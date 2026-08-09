#!/usr/bin/env python3
"""Production XFeat tracker state machine.

Production default for the ANAFI/GLOMAP runtime:

- BOOT_INIT / LOST: MegaLoc global retrieval topK=30 -> XFeat+LighterGlue -> PnP.
- TRACK / WEAK_TRACK: no global retrieval; vpr_ms is intentionally 0.  Candidate
  refs come only from pose prior, nearby camera centers, and SfM covisibility.
- TRACK / WEAK_TRACK run XFeat+LighterGlue adaptive on every frame: first top3
  local refs; if not strong enough, retry top5 local refs.

The mutual-NN fast pass ("nn_then_lg") was removed from production on 2026-07-14 on
accuracy grounds.  It remains selectable via cfg.matcher_mode so the earlier
benchmarks stay reproducible.  The temporal anchor cache is consulted ONLY by that
NN pass, so it is inert under the production default.

Deliberately NOT in the production TRACK path:
- ALIKED
- LoMa
- Efficient LoFTR
- non-MegaLoc VPR
- MegaLoc, except BOOT_INIT and LOST acquisition.

This module is intentionally additive. It reuses the existing XFeat reloc bundle
and does not modify the MV-RoMa/GLOMAP base map.
"""
from __future__ import annotations

from collections import OrderedDict
import gc
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pose_types import Localizer, Pose
from megaloc_cache import load_megaloc_cache, write_megaloc_cache
from reference_index import ReferenceIndex, open_reference_index
from reloc_localizer_xfeat import (
    Camera,
    XFeatRelocMap,
    _to_device_feats,
    extract_xfeat,
    load_xfeat,
    DEVICE,
    MEGALOC_REPO_DIR,
    MEGALOC_WEIGHTS,
    TORCH_HUB_DIR,
)

MEGALOC_REPO = "gmberton/MegaLoc"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# SFM_LOC_TIMING=1 restores full CUDA synchronize around stage timers (slower, for profiling).
# Default off: avoid device-wide sync on every feature/match/pnp boundary.
_LOC_TIMING = os.environ.get("SFM_LOC_TIMING", "0") == "1"
_PASS_TIMING_FIELDS = ("feature_ms", "match_ms", "pnp_ms")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timing_begin() -> float:
    if _LOC_TIMING:
        _sync()
    return time.perf_counter()


def _timing_ms(t0: float) -> float:
    if _LOC_TIMING:
        _sync()
    return (time.perf_counter() - t0) * 1000.0


def _temporal_diagnostics(info: dict) -> dict:
    return {k: v for k, v in info.items() if k.startswith("temporal_cache_")}


def _apply_pass_timings(info: dict, passes: list[tuple[str, dict]]) -> None:
    snapshots = [
        (label, {timing_field: pass_info.get(timing_field) for timing_field in _PASS_TIMING_FIELDS})
        for label, pass_info in passes
    ]
    for label, timing in snapshots:
        for timing_field, value in timing.items():
            info[f"{label}_{timing_field}"] = value
    for timing_field in _PASS_TIMING_FIELDS:
        values = [
            timing[timing_field]
            for _, timing in snapshots
            if timing[timing_field] is not None
        ]
        info[timing_field] = sum(float(value) for value in values) if values else None
    info["timing_synced"] = bool(_LOC_TIMING)


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _pose_center_yaw_from_ret(ret) -> tuple[np.ndarray, float]:
    T = ret["cam_from_world"]
    R = T.rotation.matrix()
    t = np.asarray(T.translation)
    C = -R.T @ t
    fwd = R.T @ np.array([0.0, 0.0, 1.0])
    yaw = float(math.atan2(fwd[1], fwd[0]))
    return C.astype(np.float32), yaw


def _map_heading_from_forward(fwd: np.ndarray, map_frame,
                              fallback_yaw: float) -> float:
    """Public pose heading; internal reference-yaw convention stays untouched."""
    if map_frame is None:
        return float(fallback_yaw)
    return float(map_frame.heading(np.asarray(fwd, dtype=float)))


def _pose_map_yaw_from_ret(ret, map_frame, fallback_yaw: float) -> float:
    if map_frame is None:
        return float(fallback_yaw)
    R = ret["cam_from_world"].rotation.matrix()
    fwd = R.T @ np.array([0.0, 0.0, 1.0])
    return _map_heading_from_forward(fwd, map_frame, fallback_yaw)


def _ret_inlier_mask(ret, n_corr: int) -> np.ndarray | None:
    if ret is None:
        return None
    for key in ("inliers", "inlier_mask"):
        if key in ret:
            mask = np.asarray(ret[key], bool)
            if mask.shape[0] == int(n_corr):
                return mask
    return None


def _inlier_distribution(
    ret, pts2d: np.ndarray, width: int, height: int,
) -> tuple[float, int]:
    """Return inlier ratio and occupied cells in the production 5x3 image grid."""
    points = np.asarray(pts2d, dtype=float)
    mask = _ret_inlier_mask(ret, len(points))
    if mask is None or not np.any(mask):
        return 0.0, 0
    inliers = points[mask]
    gx = np.clip((inliers[:, 0] * 5.0 / max(int(width), 1)).astype(int), 0, 4)
    gy = np.clip((inliers[:, 1] * 3.0 / max(int(height), 1)).astype(int), 0, 2)
    return float(np.count_nonzero(mask) / max(len(points), 1)), int(
        len(np.unique(gy * 5 + gx))
    )


def scaled_simple_radial_camera(width0: int, height0: int, params0: list[float],
                                width: int, height: int) -> Camera:
    """Scale a SIMPLE_RADIAL camera to a resized stream image.

    COLMAP SIMPLE_RADIAL params are [f, cx, cy, k]. f/cx/cy scale in pixels;
    k is normalized-coordinate radial distortion and is kept unchanged.
    """
    sx = float(width) / float(width0)
    sy = float(height) / float(height0)
    if abs(sx - sy) > 1e-6:
        raise ValueError(f"non-uniform resize is not supported: sx={sx} sy={sy}")
    f, cx, cy, k = [float(x) for x in params0]
    return Camera("SIMPLE_RADIAL", int(width), int(height),
                  [f * sx, cx * sx, cy * sy, k])


def mutual_nn_pairs(q_desc: torch.Tensor, r_desc: torch.Tensor, min_score: float = 0.8,
                    with_scores: bool = False, q_normalized: bool = False):
    """Mutual-nearest-neighbour descriptor matches for XFeat descriptors.

    Returns an int64 tensor of shape (K,2), [query_idx, ref_idx].  This is the
    cheap alternative to LighterGlue for a fast first pass; PnP/RANSAC remains
    the geometric verifier.  With with_scores=True also returns the (K,) cosine
    similarity of each pair (used by correspondence dedup to keep the best match).
    q_normalized=True reuses a caller-cached normalized query; refs are always
    normalized here so their LightGlue inputs remain untouched.
    """
    if q_desc.ndim != 2 or r_desc.ndim != 2 or len(q_desc) == 0 or len(r_desc) == 0:
        empty = torch.empty((0, 2), dtype=torch.long, device=q_desc.device)
        if with_scores:
            return empty, torch.empty((0,), dtype=torch.float32, device=q_desc.device)
        return empty
    qd = q_desc.float() if q_normalized else F.normalize(q_desc.float(), dim=-1)
    rd = F.normalize(r_desc.float(), dim=-1)
    sim = qd @ rd.T
    best_r = sim.argmax(dim=1)
    best_q = sim.argmax(dim=0)
    qi = torch.arange(sim.shape[0], device=sim.device)
    scores = sim[qi, best_r]
    ok = (best_q[best_r] == qi) & (scores >= float(min_score))
    pairs = torch.stack([qi[ok], best_r[ok]], dim=1).long()
    if with_scores:
        return pairs, scores[ok]
    return pairs


def dedup_correspondence_indices(pts2d: np.ndarray, pts3d: np.ndarray,
                                 scores: np.ndarray) -> np.ndarray:
    """Indices keeping one correspondence per 2D keypoint and per 3D anchor.

    Preference: higher score first (NN cosine similarity; LighterGlue matches
    all carry score 1.0, so ties keep the earlier = higher-priority candidate).
    Non-finite 2D rows are dropped.  3D anchors are identified by their xyz
    bytes, which also collapses the same landmark reached via the temporal
    cache and via a direct ref, or via multiple refs sharing a triangulated
    point.  PnP/RANSAC inlier counts then count distinct observations instead
    of double-counting repeats.
    """
    idx = np.flatnonzero(np.isfinite(pts2d).all(axis=1))
    if len(idx) == 0:
        return idx
    # np.unique(return_index) keeps the FIRST occurrence, so order rows by
    # (-score, original index): best score wins, ties keep the earlier row.
    # (np.unique also re-orders its output, so re-sort before the second pass --
    # a bare stable argsort on scores would inherit that lexicographic order.)
    order = idx[np.lexsort((idx, -scores[idx]))]
    _, f1 = np.unique(pts2d[order], axis=0, return_index=True)
    keep = order[f1]
    keep = keep[np.lexsort((keep, -scores[keep]))]
    _, f2 = np.unique(pts3d[keep], axis=0, return_index=True)
    keep = keep[f2]
    keep.sort()
    return keep


@dataclass
class ProductionConfig:
    boot_global_topk: int = 30
    lost_global_topk: int = 30
    weak_global_topk: int = 0          # production: MegaLoc is never used in WEAK_TRACK
    # TRACK: few nearby refs (2–3). WEAK: wider pool (weak_local_topk=8).
    # Loc ~15 FPS vs stream 30 FPS is handled outside by latest-frame coalesce
    # (never queue backlog; always track the newest image).
    local_topk: int = 3
    weak_local_topk: int = 8
    near_pool: int = 24
    covis_per_ref: int = 20
    radius: float = 0.8
    max_yaw_diff_deg: float = 90.0
    # LOST reacquisition on continuous paths: bias MegaLoc toward the last pose
    # so visually similar far places do not steal the fix. Fall back to pure
    # global MegaLoc only after an elapsed capture-time timeout.
    lost_prefer_nearby: bool = True
    lost_nearby_radius: float = 3.0
    lost_nearby_topk: int = 12
    lost_nearby_pool: int = 32
    lost_spatial_lambda: float = 0.55
    lost_pure_global_after_s: float = 3.0
    # Four-sequence and fixed-seed P124 sweeps select 1700 over 1300.
    xfeat_topk_track: int = 1700
    xfeat_topk_acquire: int = 2048
    min_conf: float = 0.1
    acquire_min_inliers: int = 80
    # Balanced gates (relaxed from the ultra-strict pass that caused extra LOST/shake).
    track_min_inliers: int = 50
    weak_min_inliers: int = 30
    good_inliers: int = 80
    min_inlier_ratio: float = 0.15
    min_inlier_grid_cells: int = 6
    max_reproj_error_acquire: float = 5.0
    max_reproj_error_track: float = 5.0
    pnp_ransac_max_error: float = 5.0
    pnp_ransac_random_seed: int = -1
    acquire_max_jump: float = 1.25
    acquire_max_yaw_diff_deg: float = 90.0
    max_jump: float = 2.0
    max_speed_mps: float = 10.0
    jump_slack_m: float = 0.4
    weak_after: int = 3
    lost_after: int = 3
    flow_enabled: bool = False       # experimental KLT path; never on in production
    # 2026-07-14: production runs LighterGlue on every TRACK/WEAK frame. The mutual-NN
    # fast pass ("nn_then_lg") was removed on accuracy grounds; it stays selectable so
    # the old benchmarks remain reproducible, but it is no longer a production default.
    matcher_mode: str = "lighterglue"  # lighterglue | nn_then_lg | nn
    # BOOT_INIT/LOST acquisition matcher. Kept separate from matcher_mode so that
    # switching TRACK matching to pure NN cannot weaken global acquisition
    # (pure-NN acquisition can livelock in BOOT_INIT if it never clears the
    # acquire gate). "" = follow matcher_mode.
    acquire_matcher_mode: str = "lighterglue"
    nn_min_score: float = 0.85
    adaptive_first_topk: int = 3
    adaptive_accept_inliers: int = 100
    adaptive_accept_reproj: float = 3.5
    max_corr_per_ref: int = 0     # 0 = unlimited
    max_corr_total: int = 0       # 0 = unlimited; spatially diversified before PnP
    dedup_corr: bool = False      # drop duplicate 2D/3D correspondences before PnP
    temporal_cache_enabled: bool = True
    # "full_ref": seed anchors from all keypoints of the used refs (default);
    # "inliers": seed from this frame's PnP-validated inliers when available.
    temporal_cache_seed_mode: str = "full_ref"
    temporal_cache_min_anchors: int = 80
    temporal_cache_max_anchors: int = 2048
    temporal_cache_max_age: int = 2
    temporal_cache_min_score: float = 0.85
    temporal_cache_seed_min_inliers: int = 150
    temporal_cache_seed_max_reproj: float = 3.5
    flow_mnn_crosscheck: bool = False
    flow_mnn_max_translation: float = 0.10
    flow_mnn_max_yaw_deg: float = 3.0
    flow_mnn_min_unique_inliers: int = 80
    flow_mnn_max_reproj: float = 3.5

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.min_inlier_ratio)) or not (
            0.0 < float(self.min_inlier_ratio) <= 1.0
        ):
            raise ValueError("min_inlier_ratio must be finite and within (0, 1]")
        if isinstance(self.min_inlier_grid_cells, bool) or not (
            1 <= int(self.min_inlier_grid_cells) <= 15
        ):
            raise ValueError("min_inlier_grid_cells must be within [1, 15]")


@dataclass
class RuntimeState:
    mode: str = "BOOT_INIT"
    last_pose: Pose | None = None
    prev_pose: Pose | None = None
    last_center: np.ndarray | None = None
    prev_center: np.ndarray | None = None
    last_yaw: float | None = None
    prev_yaw: float | None = None
    last_refs: list[int] = field(default_factory=list)
    bad_count: int = 0
    fail_count: int = 0
    # Consecutive LOST frames since last success (for nearby→global fallback).
    lost_streak: int = 0
    lost_since_stamp: float | None = None


@dataclass
class TemporalAnchorCache:
    descriptors: torch.Tensor | None = None
    xyz: np.ndarray | None = None
    ref_ids: np.ndarray | None = None
    age: int = 0
    source_stage: str = ""

    def __len__(self) -> int:
        return 0 if self.xyz is None else int(len(self.xyz))

    def clear(self) -> None:
        self.descriptors = None
        self.xyz = None
        self.ref_ids = None
        self.age = 0
        self.source_stage = ""


def predict_center(
    state: RuntimeState, capture_stamp: float | None = None,
) -> np.ndarray | None:
    if state.last_center is None:
        return None
    if state.prev_center is None:
        return state.last_center.copy()
    if capture_stamp is not None and state.last_pose is not None and state.prev_pose is not None:
        history_dt = float(state.last_pose.stamp) - float(state.prev_pose.stamp)
        future_dt = float(capture_stamp) - float(state.last_pose.stamp)
        if math.isfinite(history_dt) and math.isfinite(future_dt) and history_dt > 1e-6:
            ratio = min(3.0, max(0.0, future_dt / history_dt))
            return state.last_center + (state.last_center - state.prev_center) * ratio
    return state.last_center + (state.last_center - state.prev_center)


def lost_prior_expired(
    state: RuntimeState, capture_stamp: float, timeout_s: float,
) -> bool:
    if state.lost_since_stamp is None or timeout_s <= 0:
        return timeout_s <= 0 and state.lost_since_stamp is not None
    elapsed = float(capture_stamp) - float(state.lost_since_stamp)
    return math.isfinite(elapsed) and elapsed >= float(timeout_s)


def motion_gate_limit(
    state: RuntimeState,
    capture_stamp: float,
    hard_limit: float,
    max_speed_mps: float,
    slack_m: float,
) -> float:
    limit = float(hard_limit)
    if state.last_pose is None or max_speed_mps <= 0:
        return limit
    elapsed = float(capture_stamp) - float(state.last_pose.stamp)
    if not math.isfinite(elapsed) or elapsed < 0:
        return limit
    dynamic = max(0.0, float(slack_m)) + float(max_speed_mps) * elapsed
    return min(limit, dynamic) if limit > 0 else dynamic


def predict_yaw(state: RuntimeState) -> float | None:
    return state.last_yaw


def select_local_candidates(ref_names: list[str], ref_centers: np.ndarray,
                            ref_yaws: np.ndarray | None, covis: dict | None,
                            state: RuntimeState, cfg: ProductionConfig,
                            topk: int | None = None,
                            radius: float | None = None,
                            near_pool: int | None = None,
                            max_yaw_diff_deg: float | None = None) -> list[int]:
    """Select TRACK refs using geometry only: KDTree-equivalent distance + covis.

    Uses a brute-force distance sort because the map has only ~2k refs; this avoids
    requiring a serialized KDTree in the bundle and is fast enough (<1 ms).
    """
    C = predict_center(state)
    if C is None or ref_centers is None:
        return []
    topk = int(topk or cfg.local_topk)
    use_radius = float(cfg.radius if radius is None else radius)
    use_pool = int(cfg.near_pool if near_pool is None else near_pool)
    use_yaw = float(cfg.max_yaw_diff_deg if max_yaw_diff_deg is None else max_yaw_diff_deg)
    centers = np.asarray(ref_centers, np.float32)
    d = np.linalg.norm(centers - C[None, :], axis=1)
    finite = np.isfinite(d)
    order = [int(i) for i in np.argsort(np.where(finite, d, np.inf))]

    pred_yaw = predict_yaw(state)
    max_yaw = math.radians(use_yaw)

    near = []
    for i in order:
        if not finite[i]:
            continue
        if use_radius > 0 and d[i] > use_radius:
            break
        if pred_yaw is not None and ref_yaws is not None and np.isfinite(ref_yaws[i]):
            if abs(_angle_diff(float(ref_yaws[i]), pred_yaw)) > max_yaw:
                continue
        near.append(i)
        if len(near) >= use_pool:
            break

    covis_refs: list[int] = []
    if covis:
        for idx in state.last_refs[:3]:
            if 0 <= idx < len(ref_names):
                covis_refs.extend(int(j) for j in covis.get(ref_names[idx], [])[:cfg.covis_per_ref])
        for idx in near[:3]:
            covis_refs.extend(int(j) for j in covis.get(ref_names[idx], [])[:max(3, cfg.covis_per_ref // 2)])

    union = list(dict.fromkeys(near + covis_refs + state.last_refs))
    valid = []
    for i in union:
        if not (0 <= int(i) < len(ref_names)) or not finite[int(i)]:
            continue
        if use_radius > 0 and d[int(i)] > use_radius:
            continue
        if pred_yaw is not None and ref_yaws is not None and np.isfinite(ref_yaws[int(i)]):
            if abs(_angle_diff(float(ref_yaws[int(i)]), pred_yaw)) > max_yaw:
                continue
        valid.append(int(i))

    last_set = set(int(i) for i in state.last_refs)
    scale = max(use_radius, 1e-6)

    def score(i: int) -> float:
        dist_score = math.exp(-float(d[i]) / scale)
        if pred_yaw is not None and ref_yaws is not None and np.isfinite(ref_yaws[i]):
            yaw_score = math.exp(-abs(_angle_diff(float(ref_yaws[i]), pred_yaw)) / math.radians(45.0))
        else:
            yaw_score = 0.5
        recent = 1.0 if i in last_set else 0.0
        return 2.0 * dist_score + 0.8 * yaw_score + recent

    valid.sort(key=score, reverse=True)
    return valid[:topk]


def spatially_cap_correspondence_indices(
    pts2d: np.ndarray,
    max_total: int,
    width: int,
    height: int,
    scores: np.ndarray | None = None,
    grid_x: int = 8,
    grid_y: int = 6,
) -> np.ndarray:
    """Indices for a stable quality-first, spatially diverse correspondence cap."""
    pts2d = np.asarray(pts2d)
    if max_total <= 0 or len(pts2d) <= max_total:
        return np.arange(len(pts2d), dtype=int)
    if scores is None:
        scores = np.ones(len(pts2d), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(pts2d),):
            raise ValueError("correspondence scores must align with pts2d")
    gx = np.clip((pts2d[:, 0] / max(width, 1) * grid_x).astype(int), 0, grid_x - 1)
    gy = np.clip((pts2d[:, 1] / max(height, 1) * grid_y).astype(int), 0, grid_y - 1)
    buckets: dict[int, list[int]] = {}
    for i, (x, y) in enumerate(zip(gx, gy)):
        buckets.setdefault(int(y * grid_x + x), []).append(i)
    for bucket in buckets.values():
        bucket.sort(key=lambda idx: (-float(scores[idx]), idx))
    picked: list[int] = []
    keys = sorted(buckets)
    cursor = 0
    while len(picked) < max_total and keys:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        if bucket:
            picked.append(bucket.pop(0))
        if not bucket:
            keys.remove(key)
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor += 1
    # Keep the original correspondence ordering for deterministic PnP input.
    return np.asarray(sorted(picked), dtype=int)


class MegaLocLayer:
    """MegaLoc retrieval layer with cached reference descriptors."""

    def __init__(
        self,
        ref_desc: np.ndarray | None,
        input_size: int = 322,
        device: str = DEVICE,
        *,
        reference_index: ReferenceIndex | None = None,
        ref_names: Iterable[str] | None = None,
    ):
        if (ref_desc is None) == (reference_index is None):
            raise ValueError("provide exactly one of ref_desc or reference_index")
        self.ref_desc = None if ref_desc is None else np.asarray(ref_desc, np.float32)
        self._reference_index = reference_index
        self._reference_name_to_bundle_index: dict[str, int] | None = None
        if reference_index is not None:
            bundle_names = tuple(ref_names or ())
            if len(bundle_names) != reference_index.count:
                raise ValueError(
                    "reference index count does not match relocation bundle names"
                )
            if len(set(bundle_names)) != len(bundle_names):
                raise ValueError("relocation bundle reference names must be unique")
            if set(bundle_names) != set(reference_index.names):
                raise ValueError(
                    "reference index names do not match relocation bundle names"
                )
            self._reference_name_to_bundle_index = {
                name: index for index, name in enumerate(bundle_names)
            }
        self.input_size = int(input_size)
        self.device = device
        self.fp16 = device.startswith("cuda") and os.environ.get("SFM_MEGALOC_FP16", "0") == "1"
        self._model = None

    @property
    def reference_count(self) -> int:
        if self._reference_index is not None:
            return self._reference_index.count
        assert self.ref_desc is not None
        return int(self.ref_desc.shape[0])

    def model(self):
        if self._model is None:
            torch.hub.set_dir(str(TORCH_HUB_DIR))
            self._model = torch.hub.load(
                str(MEGALOC_REPO_DIR), "get_trained_model", source="local",
                weights_path=str(MEGALOC_WEIGHTS),
            ).eval().to(self.device)
        return self._model

    @torch.inference_mode()
    def extract_one(self, rgb: np.ndarray) -> np.ndarray:
        model = self.model()
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(self.device)[None] / 255.0
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic",
                          align_corners=False, antialias=True)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            d = model((x - mean) / std).float().squeeze(0).detach().cpu().numpy()
        return d / (np.linalg.norm(d) + 1e-12)

    def topk(self, rgb: np.ndarray, k: int) -> tuple[list[int], float]:
        scored, ms = self.topk_scored(rgb, k)
        return [i for i, _ in scored], ms

    def topk_scored(self, rgb: np.ndarray, k: int) -> tuple[list[tuple[int, float]], float]:
        """Return (ref_index, cosine_similarity) pairs sorted by score descending."""
        t0 = _timing_begin()
        q = self.extract_one(rgb)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("k must be a positive integer")
        top_k = min(int(k), self.reference_count)
        if top_k == 0:
            return [], _timing_ms(t0)
        if self._reference_index is not None:
            matches = self._reference_index.query(q, top_k=top_k)
            assert self._reference_name_to_bundle_index is not None
            scored = [
                (self._reference_name_to_bundle_index[match.name], match.score)
                for match in matches
            ]
            return scored, _timing_ms(t0)
        assert self.ref_desc is not None
        sims = self.ref_desc @ q
        order = np.argsort(-sims)[:top_k]
        scored = [(int(i), float(sims[int(i)])) for i in order]
        return scored, _timing_ms(t0)

    @staticmethod
    def load_cache(cache_path: Path, ref_names: list[str], input_size: int = 322,
                   device: str = DEVICE, meta_path: Path | None = None) -> "MegaLocLayer":
        desc = load_megaloc_cache(cache_path, ref_names, meta_path)
        return MegaLocLayer(desc, input_size=input_size, device=device)

    @staticmethod
    def load_index(
        index_path: Path,
        ref_names: list[str],
        *,
        expected_model_identity: str,
        input_size: int = 322,
        device: str = DEVICE,
    ) -> "MegaLocLayer":
        index = open_reference_index(
            index_path,
            expected_model_identity=expected_model_identity,
        )
        return MegaLocLayer(
            None,
            input_size=input_size,
            device=device,
            reference_index=index,
            ref_names=ref_names,
        )

    @staticmethod
    @torch.inference_mode()
    def build_cache(ref_names: list[str], image_root: Path, cache_path: Path,
                    meta_json: Path | None = None, input_size: int = 322,
                    batch: int = 16, device: str = DEVICE) -> "MegaLocLayer":
        layer = MegaLocLayer(np.empty((0, 0), np.float32), input_size=input_size, device=device)
        model = layer.model()
        outs = []
        buf = []
        for i, name in enumerate(ref_names, 1):
            bgr = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_root / name)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Map refs are mixed 720p/1080p. Resize each image before stacking.
            rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
            buf.append(x)
            if len(buf) >= batch or i == len(ref_names):
                x = torch.stack(buf).to(device)
                x = F.interpolate(x, size=(input_size, input_size), mode="bicubic",
                                  align_corners=False, antialias=True)
                mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
                d = model((x - mean) / std).float().detach().cpu().numpy()
                outs.append(d)
                buf.clear()
                if i % 160 == 0 or i == len(ref_names):
                    print(f"[{time.strftime('%H:%M:%S')}] MegaLoc refs {i}/{len(ref_names)}", flush=True)
        desc = np.vstack(outs).astype(np.float32)
        desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
        write_megaloc_cache(
            cache_path, desc, ref_names, meta_json,
            metadata={
                "model": "MegaLoc",
                "repo": MEGALOC_REPO,
                "input_size": input_size,
                "refs": len(ref_names),
                "image_root": str(image_root),
            },
        )
        return MegaLocLayer(desc, input_size=input_size, device=device)


class ProductionXFeatTracker(Localizer):
    def __init__(self, reloc_map: XFeatRelocMap, megaloc: MegaLocLayer,
                 frame_source, query_cam: Camera,
                 cfg: ProductionConfig | None = None,
                 map_frame=None):   # avoid shared mutable default
        n_refs = len(reloc_map.ref_names)
        if reloc_map.ref_centers is None or len(reloc_map.ref_centers) != n_refs:
            got = 0 if reloc_map.ref_centers is None else len(reloc_map.ref_centers)
            raise ValueError(
                f"reloc bundle tracking metadata mismatch: refs={n_refs} ref_centers={got}; "
                "run augment_reloc_bundle_tracking.py before real-flight deployment"
            )
        if reloc_map.ref_yaws is not None and len(reloc_map.ref_yaws) != n_refs:
            raise ValueError(
                f"reloc bundle ref_yaws mismatch: refs={n_refs} ref_yaws={len(reloc_map.ref_yaws)}"
            )
        if megaloc.reference_count != n_refs:
            raise ValueError(
                f"MegaLoc descriptor mismatch: refs={n_refs} "
                f"descriptors={megaloc.reference_count}"
            )
        self.map = reloc_map
        self.meg = megaloc
        self.frame_source = frame_source
        self.cam = query_cam
        self.cfg = cfg if cfg is not None else ProductionConfig()
        self.map_frame = map_frame
        # Cache each ref's XFeat features on the GPU so they are uploaded once, not every frame.
        self._ref_dev_cache: OrderedDict[int, dict] = OrderedDict()
        self.state = RuntimeState()
        self.temporal_cache = TemporalAnchorCache()
        self._xfeat = None
        self._last_info: dict = {}
        self._frame_capture_stamp = time.monotonic()
        # diagnostics only: per-frame model-call counters (reset each localize_frame)
        self._frame_counters: dict = {}
        self._last_inl_2d = None       # accepted inlier query 2D (for optical-flow seeding)
        self._last_inl_3d = None       # accepted inlier map 3D
        self._last_seed_refs: tuple = ()  # last temporal-cache seed ref-set (skip identical rebuilds)
        # Experimental optical-flow path. Production pins this off; the offline
        # benchmark may opt in explicitly through ProductionConfig.
        self._flow_enabled = bool(self.cfg.flow_enabled)
        self._flow_refresh_every = int(os.environ.get("SFM_FLOW_REFRESH", "6"))
        self._flow_2d = None
        self._flow_3d = None
        self._flow_gray = None
        self._flow_age = 0
        self._flow_min_seed = 100      # seed must be able to satisfy the tracked-point gate
        self._flow_min_track = 20      # tracked pts below this -> refresh next frame
        # Experimental-only gates. Defaults preserve the production-off path;
        # benchmark runs may tighten them without changing shipped settings.
        self._flow_fb_px = float(os.environ.get("SFM_FLOW_FB_PX", "1.0"))
        self._flow_qual_inl = int(os.environ.get("SFM_FLOW_QUAL_INLIERS", "50"))
        self._flow_qual_ratio = float(os.environ.get("SFM_FLOW_QUAL_RATIO", "0.25"))
        self._flow_qual_reproj = float(os.environ.get("SFM_FLOW_QUAL_REPROJ", "4.0"))
        self._flow_qual_ntrack = int(os.environ.get("SFM_FLOW_QUAL_NTRACK", "100"))
        self._flow_mnn_crosscheck = bool(self.cfg.flow_mnn_crosscheck)
        self._flow_mnn_max_translation = float(self.cfg.flow_mnn_max_translation)
        self._flow_mnn_max_yaw = math.radians(float(self.cfg.flow_mnn_max_yaw_deg))
        self._flow_mnn_min_unique_inliers = int(self.cfg.flow_mnn_min_unique_inliers)
        self._flow_mnn_max_reproj = float(self.cfg.flow_mnn_max_reproj)
        # Microbench: never leave TRACK, never call MegaLoc/BOOT/LOST acquisition.
        # Used by live_localizer_worker --force-track-bench for TRACK FPS only.
        self.lock_track_only = False

    def ensure_xfeat(self):
        if self._xfeat is None:
            self._xfeat = load_xfeat(max(self.cfg.xfeat_topk_acquire, self.cfg.xfeat_topk_track))
        return self._xfeat

    def ensure_models(self):
        self.meg.model()
        xfeat = self.ensure_xfeat()
        ensure_matcher = getattr(xfeat, "ensure_lighterglue", None)
        if callable(ensure_matcher):
            ensure_matcher()
        return xfeat

    def _clear_tracking_history(self) -> None:
        """Drop priors that must not survive LOST reacquisition or GPU OOM."""
        self._clear_pose_prior()
        self.state.lost_streak = 0
        self.state.lost_since_stamp = None
        self.temporal_cache.clear()
        self._last_seed_refs = ()
        self._last_inl_2d = None
        self._last_inl_3d = None
        self._flow_2d = None
        self._flow_3d = None
        self._flow_gray = None
        self._flow_age = 0

    def _clear_pose_prior(self) -> None:
        self.state.last_pose = None
        self.state.prev_pose = None
        self.state.last_center = None
        self.state.prev_center = None
        self.state.last_yaw = None
        self.state.prev_yaw = None
        self.state.last_refs = []

    def _clear_short_term_track_helpers(self) -> None:
        """On LOST: drop temporal/flow helpers but KEEP pose prior for nearby reacquire."""
        self.temporal_cache.clear()
        self._last_seed_refs = ()
        self._last_inl_2d = None
        self._last_inl_3d = None
        self._flow_2d = None
        self._flow_3d = None
        self._flow_gray = None
        self._flow_age = 0

    def _global_candidates(self, frame: np.ndarray, k: int) -> tuple[list[int], float]:
        self._frame_counters["megaloc_call_count"] = self._frame_counters.get("megaloc_call_count", 0) + 1
        return self.meg.topk(frame, k)

    def _lost_candidates(self, frame: np.ndarray, k: int) -> tuple[list[int], float]:
        """LOST acquisition: prefer refs near last pose on continuous paths.

        Pure MegaLoc top-K can snap to a visually similar but far place (trees,
        corridors). When a recent pose prior exists, re-rank MegaLoc by distance
        and inject geometric neighbors. After lost_pure_global_after_s of capture
        time, fall back to pure global MegaLoc (kidnapped / true teleport).
        """
        k = max(1, int(k))
        prefer = bool(self.cfg.lost_prefer_nearby)
        has_prior = self.state.last_center is not None or self.state.prev_center is not None
        if has_prior and lost_prior_expired(
            self.state,
            self._frame_capture_stamp,
            float(self.cfg.lost_pure_global_after_s),
        ):
            self._clear_pose_prior()
            self._frame_counters["lost_prior_expired"] = True
            has_prior = False
        use_nearby = prefer and has_prior

        self._frame_counters["megaloc_call_count"] = self._frame_counters.get("megaloc_call_count", 0) + 1
        fetch_k = max(k, int(self.cfg.lost_global_topk), int(self.cfg.lost_nearby_topk) * 2)
        scored, vpr_ms = self.meg.topk_scored(frame, fetch_k)

        if not use_nearby or self.map.ref_centers is None:
            self._frame_counters["lost_acquire_mode"] = "global_megaloc"
            return [i for i, _ in scored[:k]], vpr_ms

        C = predict_center(self.state)
        if C is None:
            C = np.asarray(self.state.last_center, np.float32)
        centers = np.asarray(self.map.ref_centers, np.float32)
        radius = max(float(self.cfg.lost_nearby_radius), 1e-3)
        lam = float(self.cfg.lost_spatial_lambda)

        ranked: list[tuple[float, int, float, float]] = []
        for idx, sim in scored:
            if not (0 <= idx < len(centers)):
                continue
            dist = float(np.linalg.norm(centers[idx] - C))
            # Cosine sim in ~[-1,1]; penalize far places so nearby wins ties.
            score = float(sim) - lam * min(dist / radius, 4.0)
            ranked.append((score, idx, float(sim), dist))
        ranked.sort(key=lambda t: t[0], reverse=True)
        megaloc_biased = [idx for _, idx, _, _ in ranked]

        geo = select_local_candidates(
            self.map.ref_names,
            self.map.ref_centers,
            self.map.ref_yaws,
            self.map.covis,
            self.state,
            self.cfg,
            topk=int(self.cfg.lost_nearby_topk),
            radius=float(self.cfg.lost_nearby_radius),
            near_pool=int(self.cfg.lost_nearby_pool),
            max_yaw_diff_deg=min(120.0, float(self.cfg.max_yaw_diff_deg) + 30.0),
        )
        # Geometry first (path continuity), then spatially re-ranked MegaLoc.
        candidates = list(dict.fromkeys(list(geo) + megaloc_biased))[:k]
        if not candidates:
            candidates = [i for i, _ in scored[:k]]
        self._frame_counters["lost_acquire_mode"] = "nearby_biased"
        self._frame_counters["lost_geo_refs"] = int(len(geo))
        return candidates, vpr_ms

    def _track_candidates(self, weak: bool = False) -> list[int]:
        return select_local_candidates(
            self.map.ref_names,
            self.map.ref_centers,
            self.map.ref_yaws,
            self.map.covis,
            self.state,
            self.cfg,
            topk=self.cfg.weak_local_topk if weak else self.cfg.local_topk,
        )

    def _estimate_reproj_rms(self, ret, pts2d: np.ndarray, pts3d: np.ndarray) -> float | None:
        mask = _ret_inlier_mask(ret, len(pts2d))
        if mask is None or not np.any(mask):
            return None
        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        t = np.asarray(T.translation)
        Xc = (R @ pts3d[mask].T).T + t[None, :]
        z = Xc[:, 2]
        valid = z > 1e-8
        if not np.any(valid):
            return None
        x = Xc[valid, 0] / z[valid]
        y = Xc[valid, 1] / z[valid]
        if self.cam.model.upper() == "SIMPLE_RADIAL":
            f, cx, cy, k = [float(v) for v in self.cam.params]
            r2 = x * x + y * y
            s = 1.0 + k * r2
            uv = np.stack([f * x * s + cx, f * y * s + cy], axis=1)
        elif self.cam.model.upper() == "PINHOLE":
            fx, fy, cx, cy = [float(v) for v in self.cam.params]
            uv = np.stack([fx * x + cx, fy * y + cy], axis=1)
        elif self.cam.model.upper() in ("OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = [float(v) for v in self.cam.params[:4]]
            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            dist = np.asarray(self.cam.params[4:], dtype=np.float64)
            uv, _ = cv2.projectPoints(
                Xc[valid].astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
            uv = uv.reshape(-1, 2)
        else:
            return None
        e = uv - pts2d[mask][valid]
        return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))

    def _age_temporal_cache(self) -> None:
        if len(self.temporal_cache) > 0:
            self.temporal_cache.age += 1
            if self.temporal_cache.age > self.cfg.temporal_cache_max_age:
                self.temporal_cache.clear()

    def _set_temporal_cache(self, payload: dict | None) -> bool:
        if not self.cfg.temporal_cache_enabled or not payload:
            return False
        desc = payload.get("descriptors")
        xyz = payload.get("xyz")
        ref_ids = payload.get("ref_ids")
        if desc is None or xyz is None or ref_ids is None:
            return False
        if not torch.is_tensor(desc):
            desc = torch.as_tensor(desc)
        desc = desc.detach().float().cpu()
        xyz = np.asarray(xyz, np.float32)
        ref_ids = np.asarray(ref_ids, np.int32)
        n = int(len(xyz))
        if n < self.cfg.temporal_cache_min_anchors:
            return False
        max_n = int(self.cfg.temporal_cache_max_anchors)
        if max_n > 0 and n > max_n:
            idx = np.linspace(0, n - 1, max_n).astype(int)
            desc = desc[idx.tolist()]
            xyz = xyz[idx]
            ref_ids = ref_ids[idx]
            n = max_n
        self.temporal_cache.descriptors = F.normalize(desc.to(DEVICE), dim=-1)
        self.temporal_cache.xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        self.temporal_cache.ref_ids = np.ascontiguousarray(ref_ids, dtype=np.int32)
        self.temporal_cache.age = 0
        self.temporal_cache.source_stage = str(payload.get("source_stage", ""))
        return n > 0

    def _cache_payload_from_ref_inliers(self, ret, pts3d_arr: np.ndarray,
                                        corr_ref_ids: list[int],
                                        corr_ref_kp_ids: list[int],
                                        source_stage: str) -> dict | None:
        mask = _ret_inlier_mask(ret, len(pts3d_arr))
        if mask is None or not np.any(mask):
            return None
        descs: list[torch.Tensor] = []
        xyzs: list[np.ndarray] = []
        refs: list[int] = []
        seen: set[tuple[int, int]] = set()
        for ci in np.flatnonzero(mask):
            ref_id = int(corr_ref_ids[int(ci)])
            kp_id = int(corr_ref_kp_ids[int(ci)])
            key = (ref_id, kp_id)
            if key in seen:
                continue
            seen.add(key)
            ref = self.map.refs[self.map.ref_names[ref_id]]
            desc = ref.feats["descriptors"][kp_id]
            if not torch.is_tensor(desc):
                desc = torch.as_tensor(desc)
            descs.append(desc.detach().float().cpu())
            xyzs.append(np.asarray(pts3d_arr[int(ci)], dtype=np.float32))
            refs.append(ref_id)
        if len(xyzs) < self.cfg.temporal_cache_min_anchors:
            return None
        return {
            "descriptors": torch.stack(descs, dim=0),
            "xyz": np.asarray(xyzs, dtype=np.float32),
            "ref_ids": np.asarray(refs, dtype=np.int32),
            "source_stage": source_stage,
        }

    def _cache_payload_from_temporal_inliers(self, ret, cache_indices: np.ndarray,
                                             source_stage: str) -> dict | None:
        if len(self.temporal_cache) == 0 or self.temporal_cache.descriptors is None:
            return None
        mask = _ret_inlier_mask(ret, len(cache_indices))
        if mask is None or not np.any(mask):
            return None
        idx = np.asarray(cache_indices, dtype=int)[mask]
        if len(idx) < self.cfg.temporal_cache_min_anchors:
            return None
        return {
            "descriptors": self.temporal_cache.descriptors.detach().cpu()[idx.tolist()],
            "xyz": np.asarray(self.temporal_cache.xyz[idx], dtype=np.float32),
            "ref_ids": np.asarray(self.temporal_cache.ref_ids[idx], dtype=np.int32),
            "source_stage": source_stage,
        }

    def _cache_payload_from_refs(self, ref_ids: Iterable[int], source_stage: str) -> dict | None:
        desc_chunks: list[torch.Tensor] = []
        xyz_chunks: list[np.ndarray] = []
        ref_chunks: list[np.ndarray] = []
        seen_refs = []
        for rid in ref_ids or []:
            rid = int(rid)
            if rid in seen_refs or not (0 <= rid < len(self.map.ref_names)):
                continue
            seen_refs.append(rid)
            ref = self.map.refs[self.map.ref_names[rid]]
            xyz = np.asarray(ref.xyz, dtype=np.float32)
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                continue
            valid = np.isfinite(xyz).all(axis=1)
            idx = np.flatnonzero(valid)
            if len(idx) == 0:
                continue
            desc = ref.feats["descriptors"]
            if not torch.is_tensor(desc):
                desc = torch.as_tensor(desc)
            desc_chunks.append(desc.detach().float().cpu()[idx.tolist()])
            xyz_chunks.append(xyz[idx])
            ref_chunks.append(np.full(len(idx), rid, dtype=np.int32))
        if not xyz_chunks:
            return None
        xyz_all = np.concatenate(xyz_chunks, axis=0).astype(np.float32)
        if len(xyz_all) < self.cfg.temporal_cache_min_anchors:
            return None
        return {
            "descriptors": torch.cat(desc_chunks, dim=0),
            "xyz": xyz_all,
            "ref_ids": np.concatenate(ref_chunks, axis=0).astype(np.int32),
            "source_stage": source_stage,
        }

    @torch.inference_mode()
    def _localize_with_candidates(self, frame: np.ndarray, candidates: Iterable[int],
                                  xfeat_topk: int, min_corr: int,
                                  matcher_mode: str | None = None,
                                  include_temporal_cache: bool = False,
                                  q_cache: dict | None = None,
                                  lg_pairs_in: dict | None = None,
                                  lg_pairs_out: dict | None = None) -> tuple[object | None, dict]:
        import pycolmap

        info = {
            "raw_matches": 0,
            "corr3d": 0,
            "used_refs": [],
            "timing_synced": bool(_LOC_TIMING),
        }
        active_matcher = matcher_mode or self.cfg.matcher_mode
        if active_matcher == "nn_then_lg":
            active_matcher = "lighterglue"
        info["matcher_mode"] = active_matcher
        need_quality_scores = bool(
            self.cfg.dedup_corr
            or self.cfg.max_corr_per_ref > 0
            or self.cfg.max_corr_total > 0
        )
        xfeat = self.ensure_xfeat()
        t0 = _timing_begin()
        # extract query features ONCE per frame; later composite passes reuse them (same
        # frame + topk -> identical descriptors/keypoints, so the PnP output is unchanged)
        if q_cache is not None and q_cache.get("q_dev") is not None:
            q_dev = q_cache["q_dev"]
            qkp = q_cache.get("qkp")
            if qkp is None:
                qkp = q_dev["keypoints"].detach()
        else:
            q_feats = extract_xfeat(xfeat, frame, xfeat_topk)
            self._frame_counters["xfeat_extract_count"] = self._frame_counters.get("xfeat_extract_count", 0) + 1
            q_dev = _to_device_feats(q_feats)
            # Keep keypoints on-device until correspondences actually need them.
            # Copying them here inserts a host fence between XFeat and the matcher;
            # failed/empty passes do not need a CPU copy at all.
            qkp = q_feats["keypoints"].detach()
            if q_cache is not None:
                q_cache["q_dev"], q_cache["qkp"] = q_dev, qkp

        def query_keypoints_numpy() -> np.ndarray:
            nonlocal qkp
            if torch.is_tensor(qkp):
                qkp = qkp.cpu().numpy()
                if q_cache is not None:
                    q_cache["qkp"] = qkp
            return qkp

        q_nn_desc = None
        if active_matcher == "nn":
            if q_cache is not None and q_cache.get("q_nn_desc") is not None:
                q_nn_desc = q_cache["q_nn_desc"]
            else:
                q_nn_desc = F.normalize(q_dev["descriptors"].float(), dim=-1)
                if q_cache is not None:
                    q_cache["q_nn_desc"] = q_nn_desc
        info["feature_ms"] = _timing_ms(t0)

        pts2d: list[np.ndarray] = []
        pts3d: list[np.ndarray] = []
        corr_scores: list[np.ndarray] = []   # per-correspondence match score (dedup priority)
        corr_ref_ids: list[int] = []
        corr_ref_kp_ids: list[int] = []
        t0 = _timing_begin()
        if include_temporal_cache and active_matcher == "nn" and self.cfg.temporal_cache_enabled:
            info["temporal_cache_attempted"] = False
            info["temporal_cache_size"] = int(len(self.temporal_cache))
            info["temporal_cache_age"] = int(self.temporal_cache.age)
            if (
                self.temporal_cache.descriptors is not None
                and self.temporal_cache.xyz is not None
                and len(self.temporal_cache) >= self.cfg.temporal_cache_min_anchors
                and self.temporal_cache.age <= self.cfg.temporal_cache_max_age
            ):
                info["temporal_cache_attempted"] = True
                match_result = mutual_nn_pairs(
                    q_nn_desc,
                    self.temporal_cache.descriptors,
                    self.cfg.temporal_cache_min_score,
                    with_scores=need_quality_scores,
                    q_normalized=True,
                )
                if need_quality_scores:
                    pairs_t, scores_t = match_result
                    pair_scores = scores_t.detach().cpu().numpy()
                else:
                    pairs_t = match_result
                    pair_scores = None
                pairs = pairs_t.detach().cpu().numpy()
                info["temporal_cache_raw_matches"] = int(len(pairs))
                info["raw_matches"] += int(len(pairs))
                if len(pairs):
                    cache_idx = pairs[:, 1].astype(int)
                    cache_xyz = np.asarray(self.temporal_cache.xyz[cache_idx], dtype=np.float32)
                    finite = np.isfinite(cache_xyz).all(axis=1)
                    if self.temporal_cache.ref_ids is not None:
                        cache_ref_ids = np.asarray(self.temporal_cache.ref_ids[cache_idx], dtype=np.int32)
                    else:
                        cache_ref_ids = np.full(len(cache_idx), -1, dtype=np.int32)
                    if finite.any():                     # vectorized gather (was a python loop)
                        qa_c = pairs[:, 0].astype(int)[finite]
                        pts2d.append(query_keypoints_numpy()[qa_c])
                        pts3d.append(cache_xyz[finite])
                        if need_quality_scores:
                            corr_scores.append(pair_scores[finite].astype(np.float32))
                        corr_ref_ids.extend(cache_ref_ids[finite].astype(int).tolist())
                        corr_ref_kp_ids.extend([-1] * int(finite.sum()))
                    temporal_refs = sorted(set(int(r) for r in cache_ref_ids if int(r) >= 0))
                    if temporal_refs:
                        info["temporal_cache_used_refs"] = temporal_refs
                    info["temporal_cache_corr3d"] = int(np.sum(finite))
            else:
                info["temporal_cache_skipped"] = True
        match_records: list[dict] = []
        for idx in candidates:
            ref_id = int(idx)
            ref = self.map.refs[self.map.ref_names[ref_id]]
            r_dev = self._ref_dev_cache.get(ref_id)
            if r_dev is None:
                if len(self._ref_dev_cache) >= 700:      # evict least-recently-used ref
                    self._ref_dev_cache.popitem(last=False)
                r_dev = _to_device_feats(ref.feats)
                self._ref_dev_cache[ref_id] = r_dev
            else:
                self._ref_dev_cache.move_to_end(ref_id)
            pair_scores = None
            if active_matcher == "nn":
                match_result = mutual_nn_pairs(
                    q_nn_desc, r_dev["descriptors"], self.cfg.nn_min_score,
                    with_scores=need_quality_scores, q_normalized=True)
                if need_quality_scores:
                    pairs_t, scores_t = match_result
                    pair_scores = scores_t
                else:
                    pairs_t = match_result
                pairs = pairs_t
                self._frame_counters["nn_call_count"] = self._frame_counters.get("nn_call_count", 0) + 1
            elif lg_pairs_in is not None and ref_id in lg_pairs_in:
                cached = lg_pairs_in[ref_id]
                if isinstance(cached, tuple):
                    pairs, pair_scores = cached
                else:
                    pairs = cached
                self._frame_counters["lg_reuse_count"] = self._frame_counters.get("lg_reuse_count", 0) + 1
            else:
                # The native tensor API lets all refs finish before one combined
                # device-to-host transfer. Preserve explicit instance overrides
                # used by experimental ONNX backends, which expose the NumPy API.
                instance_api = getattr(xfeat, "__dict__", {})
                index_api_overridden = "match_lighterglue_indices" in instance_api
                match_indices_tensor = getattr(xfeat, "match_lighterglue_indices_tensor", None)
                match_indices_scores_tensor = getattr(
                    xfeat, "match_lighterglue_indices_scores_tensor", None)
                match_indices = getattr(xfeat, "match_lighterglue_indices", None)
                if (need_quality_scores and callable(match_indices_scores_tensor)
                        and not index_api_overridden):
                    pairs, pair_scores = match_indices_scores_tensor(
                        q_dev, r_dev, min_conf=self.cfg.min_conf)
                elif callable(match_indices_tensor) and not index_api_overridden:
                    pairs = match_indices_tensor(q_dev, r_dev, min_conf=self.cfg.min_conf)
                elif callable(match_indices):
                    pairs = match_indices(q_dev, r_dev, min_conf=self.cfg.min_conf)
                else:
                    _mk0, _mk1, pairs = xfeat.match_lighterglue(
                        q_dev, r_dev, min_conf=self.cfg.min_conf)
                self._frame_counters["lg_call_count"] = self._frame_counters.get("lg_call_count", 0) + 1
            match_records.append({
                "ref_id": ref_id,
                "ref": ref,
                "pairs": pairs,
                "pair_scores": pair_scores,
            })

        # Keep pair indices resident until every reference match is queued, then
        # concatenate and transfer once. Concatenation preserves per-reference
        # match order and does not change matcher or PnP inputs.
        tensor_records = [record for record in match_records
                          if torch.is_tensor(record["pairs"])]
        if tensor_records:
            lengths = [len(record["pairs"]) for record in tensor_records]
            tensors = [record["pairs"] for record in tensor_records]
            merged = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            merged_np = merged.detach().cpu().numpy()
            offset = 0
            for record, length in zip(tensor_records, lengths):
                record["pairs"] = merged_np[offset:offset + length]
                offset += length
        score_records = [record for record in match_records
                         if torch.is_tensor(record["pair_scores"])]
        if score_records:
            lengths = [len(record["pair_scores"]) for record in score_records]
            tensors = [record["pair_scores"] for record in score_records]
            merged = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            merged_np = merged.detach().cpu().numpy()
            offset = 0
            for record, length in zip(score_records, lengths):
                record["pair_scores"] = merged_np[offset:offset + length]
                offset += length

        for record in match_records:
            ref_id = record["ref_id"]
            ref = record["ref"]
            pairs = record["pairs"]
            pair_scores = record["pair_scores"]
            if lg_pairs_out is not None and active_matcher != "nn":
                lg_pairs_out[ref_id] = (
                    (pairs, pair_scores) if pair_scores is not None else pairs)
            info["raw_matches"] += int(len(pairs))
            if len(pairs):                               # vectorized gather (was a python loop)
                pr = np.asarray(pairs)
                qa_i = pr[:, 0].astype(int)
                rb_i = pr[:, 1].astype(int)
                if need_quality_scores and pair_scores is None:
                    # LighterGlue exposes no per-match score; constant 1.0 means dedup
                    # ties resolve by candidate order (earlier = higher priority).
                    pair_scores = np.ones(len(pr), dtype=np.float32)
                Xr = ref.xyz[rb_i]
                fin = ~np.isnan(Xr).any(axis=1)
                qa_i, rb_i, Xr = qa_i[fin], rb_i[fin], Xr[fin]
                if need_quality_scores:
                    sc_i = pair_scores[fin]
                if self.cfg.max_corr_per_ref > 0 and len(qa_i) > self.cfg.max_corr_per_ref:
                    keep_ref = np.lexsort((np.arange(len(sc_i)), -sc_i))[
                        :self.cfg.max_corr_per_ref]
                    keep_ref.sort()
                    qa_i = qa_i[keep_ref]
                    rb_i = rb_i[keep_ref]
                    Xr = Xr[keep_ref]
                    sc_i = sc_i[keep_ref]
                if len(qa_i):
                    pts2d.append(query_keypoints_numpy()[qa_i])
                    pts3d.append(Xr)
                    if need_quality_scores:
                        corr_scores.append(sc_i.astype(np.float32))
                    corr_ref_ids.extend([ref_id] * len(qa_i))
                    corr_ref_kp_ids.extend(rb_i.tolist())
                    info["used_refs"].append(ref_id)
        info["match_ms"] = _timing_ms(t0)
        info["used_refs"] = list(dict.fromkeys(int(i) for i in info["used_refs"] if int(i) >= 0))
        # pts2d/pts3d are now lists of per-ref arrays -> concatenate once (was per-point append)
        pts2d_arr = (np.concatenate(pts2d, axis=0).astype(float)
                     if pts2d else np.zeros((0, 2), float))
        pts3d_arr = (np.concatenate(pts3d, axis=0).astype(float)
                     if pts3d else np.zeros((0, 3), float))
        sc_all = (np.concatenate(corr_scores, axis=0)
                  if need_quality_scores and corr_scores
                  else np.ones(len(pts3d_arr), dtype=np.float32))
        if self.cfg.dedup_corr and len(pts3d_arr):
            keep = dedup_correspondence_indices(pts2d_arr, pts3d_arr, sc_all)
            if len(keep) != len(pts3d_arr):
                info["corr3d_pre_dedup"] = int(len(pts3d_arr))
                pts2d_arr = pts2d_arr[keep]
                pts3d_arr = pts3d_arr[keep]
                sc_all = sc_all[keep]
                corr_ref_ids = np.asarray(corr_ref_ids, np.int32)[keep].tolist()
                corr_ref_kp_ids = np.asarray(corr_ref_kp_ids, np.int32)[keep].tolist()
        info["corr3d_pre_cap"] = int(len(pts3d_arr))
        if len(pts3d_arr) < min_corr:
            info["corr3d"] = int(len(pts3d_arr))
            info["inliers"] = 0
            info["reproj_rms"] = None
            return None, info
        if self.cfg.max_corr_total > 0 and len(pts3d_arr) > self.cfg.max_corr_total:
            keep = spatially_cap_correspondence_indices(
                pts2d_arr,
                self.cfg.max_corr_total,
                self.cam.width,
                self.cam.height,
                sc_all,
            )
            pts2d_arr = pts2d_arr[keep]
            pts3d_arr = pts3d_arr[keep]
            sc_all = sc_all[keep]
            corr_ref_ids = np.asarray(corr_ref_ids, np.int32)[keep].tolist()
            corr_ref_kp_ids = np.asarray(corr_ref_kp_ids, np.int32)[keep].tolist()
            info["corr3d_pruned"] = int(info["corr3d_pre_cap"] - len(pts3d_arr))
        surviving_refs = {int(ref_id) for ref_id in corr_ref_ids if int(ref_id) >= 0}
        info["used_refs"] = [
            ref_id for ref_id in info["used_refs"] if ref_id in surviving_refs]
        info["corr3d"] = int(len(pts3d_arr))
        if len(pts3d_arr) < min_corr:
            info["inliers"] = 0
            info["reproj_rms"] = None
            return None, info
        t0 = _timing_begin()
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        est_opts = pycolmap.AbsolutePoseEstimationOptions()
        est_opts.ransac.max_error = float(self.cfg.pnp_ransac_max_error)
        est_opts.ransac.random_seed = int(self.cfg.pnp_ransac_random_seed)
        ret = pycolmap.estimate_and_refine_absolute_pose(
            pts2d_arr, pts3d_arr, cam, estimation_options=est_opts)
        info["pnp_ms"] = _timing_ms(t0)
        info["pnp_failed"] = ret is None
        info["inliers"] = 0 if ret is None else int(ret.get("num_inliers", 0))
        info["reproj_rms"] = None if ret is None else self._estimate_reproj_rms(ret, pts2d_arr, pts3d_arr)
        ratio, cells = _inlier_distribution(
            ret, pts2d_arr, self.cam.width, self.cam.height
        )
        info["inlier_ratio"] = ratio
        info["inlier_grid_cells"] = cells
        info["inlier_coverage"] = float(cells / 15.0)
        # additive hook: expose the accepted inlier 2D<->3D so an optical-flow tracker can
        # seed on refresh (does NOT affect pose/gate/behavior; just stores the arrays)
        _m = _ret_inlier_mask(ret, len(pts3d_arr))
        inlier_2d = None
        if ret is not None and _m is not None:
            inlier_2d = pts2d_arr[_m].astype(np.float32)
            self._last_inl_2d, self._last_inl_3d = inlier_2d, pts3d_arr[_m].astype(np.float32)
            inlier_ref_ids = np.asarray(corr_ref_ids, dtype=np.int32)[_m]
            unique_refs, ref_counts = np.unique(inlier_ref_ids, return_counts=True)
            ranked = sorted(
                zip(unique_refs.tolist(), ref_counts.tolist()),
                key=lambda item: (-int(item[1]), int(item[0])),
            )
            info["inlier_refs"] = [int(ref_id) for ref_id, _count in ranked if ref_id >= 0]
            info["inliers_per_ref"] = {
                str(int(ref_id)): int(count) for ref_id, count in ranked if ref_id >= 0
            }
        elif ret is not None:
            inlier_2d = pts2d_arr.astype(np.float32)
            self._last_inl_2d, self._last_inl_3d = inlier_2d, pts3d_arr.astype(np.float32)
        else:
            self._last_inl_2d = self._last_inl_3d = None
        info["unique_query_inliers"] = (
            0 if inlier_2d is None else int(len(np.unique(inlier_2d, axis=0))))
        if (
            self.cfg.temporal_cache_enabled
            and ret is not None
            and all(kp >= 0 for kp in corr_ref_kp_ids)
        ):
            # Keep only the lightweight inputs until this pass wins the composite
            # gate and cache seeding is actually needed.  Materializing descriptors
            # for rejected NN/adaptive passes is pure CPU work.
            info["_temporal_cache_context"] = (
                ret, pts3d_arr, corr_ref_ids, corr_ref_kp_ids, active_matcher)
        return ret, info

    # NOTE: dead code — no callers on the production path (the live matcher is
    # _localize_with_candidates). Kept for reference; verify before relying on it.
    @torch.inference_mode()
    def _localize_with_temporal_cache(self, frame: np.ndarray, xfeat_topk: int,
                                      min_corr: int) -> tuple[object | None, dict]:
        import pycolmap

        info = {
            "raw_matches": 0,
            "corr3d": 0,
            "used_refs": [],
            "matcher_mode": "temporal_nn",
            "temporal_cache_attempted": False,
            "temporal_cache_size": int(len(self.temporal_cache)),
            "temporal_cache_age": int(self.temporal_cache.age),
        }
        if (
            not self.cfg.temporal_cache_enabled
            or self.temporal_cache.descriptors is None
            or self.temporal_cache.xyz is None
            or len(self.temporal_cache) < self.cfg.temporal_cache_min_anchors
            or self.temporal_cache.age > self.cfg.temporal_cache_max_age
        ):
            info["temporal_cache_skipped"] = True
            info["inliers"] = 0
            info["reproj_rms"] = None
            return None, info

        info["temporal_cache_attempted"] = True
        xfeat = self.ensure_xfeat()
        t0 = _timing_begin()
        q_feats = extract_xfeat(xfeat, frame, xfeat_topk)
        q_dev = _to_device_feats(q_feats)
        qkp = q_feats["keypoints"].detach().cpu().numpy()
        info["feature_ms"] = _timing_ms(t0)

        t0 = _timing_begin()
        pairs_t = mutual_nn_pairs(
            q_dev["descriptors"],
            self.temporal_cache.descriptors,
            self.cfg.temporal_cache_min_score,
        )
        pairs = pairs_t.detach().cpu().numpy()
        info["match_ms"] = _timing_ms(t0)
        info["raw_matches"] = int(len(pairs))

        if len(pairs) == 0:
            info["inliers"] = 0
            info["reproj_rms"] = None
            return None, info

        cache_idx = pairs[:, 1].astype(int)
        pts2d_arr = np.asarray([qkp[int(qa)] for qa in pairs[:, 0]], dtype=float)
        pts3d_arr = np.asarray(self.temporal_cache.xyz[cache_idx], dtype=float)
        finite = np.isfinite(pts3d_arr).all(axis=1)
        if not np.all(finite):
            pts2d_arr = pts2d_arr[finite]
            pts3d_arr = pts3d_arr[finite]
            cache_idx = cache_idx[finite]
        info["corr3d"] = int(len(pts3d_arr))
        if self.temporal_cache.ref_ids is not None and len(cache_idx):
            info["used_refs"] = sorted(set(int(i) for i in self.temporal_cache.ref_ids[cache_idx]))
        if len(pts3d_arr) < min_corr:
            info["inliers"] = 0
            info["reproj_rms"] = None
            return None, info

        t0 = _timing_begin()
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        est_opts = pycolmap.AbsolutePoseEstimationOptions()
        est_opts.ransac.max_error = float(self.cfg.pnp_ransac_max_error)
        est_opts.ransac.random_seed = int(self.cfg.pnp_ransac_random_seed)
        ret = pycolmap.estimate_and_refine_absolute_pose(
            pts2d_arr, pts3d_arr, cam, estimation_options=est_opts)
        info["pnp_ms"] = _timing_ms(t0)
        info["inliers"] = 0 if ret is None else int(ret.get("num_inliers", 0))
        info["reproj_rms"] = None if ret is None else self._estimate_reproj_rms(ret, pts2d_arr, pts3d_arr)
        ratio, cells = _inlier_distribution(
            ret, pts2d_arr, self.cam.width, self.cam.height
        )
        info["inlier_ratio"] = ratio
        info["inlier_grid_cells"] = cells
        info["inlier_coverage"] = float(cells / 15.0)
        if ret is not None:
            payload = self._cache_payload_from_temporal_inliers(ret, cache_idx, "temporal_nn")
            if payload:
                info["_temporal_cache_payload"] = payload
                info["temporal_cache_candidate_anchors"] = int(len(payload["xyz"]))
        return ret, info

    def _gate(self, ret, info: dict, mode: str) -> tuple[bool, bool, Pose | None]:
        if ret is None:
            return False, False, None
        center, yaw = _pose_center_yaw_from_ret(ret)
        pose_yaw = _pose_map_yaw_from_ret(
            ret, getattr(self, "map_frame", None), yaw
        )
        info["_tracker_yaw"] = float(yaw)
        pose = Pose(x=float(center[0]), y=float(center[1]), z=float(center[2]), yaw=pose_yaw,
                    stamp=float(self._frame_capture_stamp))
        ninl = int(info.get("inliers", 0))
        reproj = info.get("reproj_rms")
        ratio = info.get("inlier_ratio")
        cells = info.get("inlier_grid_cells")
        if ratio is None or not math.isfinite(float(ratio)):
            info["quality_rejected"] = "missing_inlier_ratio"
            return False, False, pose
        if float(ratio) < float(self.cfg.min_inlier_ratio):
            info["quality_rejected"] = "inlier_ratio"
            return False, False, pose
        if cells is None or int(cells) < int(self.cfg.min_inlier_grid_cells):
            info["quality_rejected"] = "inlier_spread"
            return False, False, pose
        is_acquire = mode in ("BOOT_INIT", "LOST", "GLOBAL_RETRY")
        if is_acquire:
            if ninl < self.cfg.acquire_min_inliers:
                return False, False, pose
            if reproj is not None and reproj > self.cfg.max_reproj_error_acquire:
                return False, False, pose
            has_prior = (
                self.state.last_center is not None
                and not lost_prior_expired(
                    self.state,
                    self._frame_capture_stamp,
                    float(self.cfg.lost_pure_global_after_s),
                )
            )
            if has_prior and self.cfg.acquire_max_jump > 0:
                pred = predict_center(self.state, self._frame_capture_stamp)
                if pred is not None:
                    jump = float(np.linalg.norm(center - pred))
                    jump_from_last = float(np.linalg.norm(center - self.state.last_center))
                    limit = motion_gate_limit(
                        self.state,
                        self._frame_capture_stamp,
                        self.cfg.acquire_max_jump,
                        self.cfg.max_speed_mps,
                        self.cfg.jump_slack_m,
                    )
                    info["acquire_jump_from_pred"] = jump
                    info["acquire_jump_from_last"] = jump_from_last
                    info["acquire_jump_limit"] = limit
                    if min(jump, jump_from_last) > limit:
                        info["acquire_jump_rejected"] = True
                        return False, False, pose
            if has_prior and self.state.last_yaw is not None \
                    and self.cfg.acquire_max_yaw_diff_deg > 0:
                yaw_delta = abs(_angle_diff(float(yaw), float(self.state.last_yaw)))
                info["acquire_yaw_delta_deg"] = math.degrees(yaw_delta)
                if yaw_delta > math.radians(float(self.cfg.acquire_max_yaw_diff_deg)):
                    info["acquire_yaw_rejected"] = True
                    return False, False, pose
            return True, ninl < self.cfg.good_inliers, pose

        # TRACK / WEAK_TRACK: allow weak-but-usable results down to weak_min_inliers.
        if ninl < self.cfg.weak_min_inliers:
            return False, False, pose
        if reproj is not None and reproj > self.cfg.max_reproj_error_track:
            return False, False, pose
        pred = predict_center(self.state, self._frame_capture_stamp)
        if pred is not None and self.cfg.max_jump > 0:
            jump = float(np.linalg.norm(center - pred))
            jump_from_last = float(np.linalg.norm(center - self.state.last_center))
            limit = motion_gate_limit(
                self.state,
                self._frame_capture_stamp,
                self.cfg.max_jump,
                self.cfg.max_speed_mps,
                self.cfg.jump_slack_m,
            )
            info["jump_from_pred"] = jump
            info["jump_from_last"] = jump_from_last
            info["jump_limit"] = limit
            if min(jump, jump_from_last) > limit:
                info["jump_rejected"] = True
                return False, False, pose
        weak = ninl < self.cfg.track_min_inliers or ninl < self.cfg.good_inliers
        return True, weak, pose

    def _publish_success(self, pose: Pose, info: dict, weak: bool,
                         source_mode: str | None = None):
        if source_mode in ("BOOT_INIT", "LOST", "GLOBAL_RETRY"):
            # A global acquisition is a discontinuous reset, not a velocity sample.
            # Retaining the old prev/last center makes predict_center extrapolate the
            # whole relocation jump on the next TRACK frame and can reject it again.
            self._clear_tracking_history()
        self.state.prev_pose = self.state.last_pose
        self.state.prev_center = self.state.last_center
        self.state.prev_yaw = self.state.last_yaw
        self.state.last_pose = pose
        self.state.last_center = np.array([pose.x, pose.y, pose.z], np.float32)
        self.state.last_yaw = float(info.get("_tracker_yaw", pose.yaw))
        self.state.last_refs = list(info.get("used_refs") or info.get("candidates", [])[:3])
        self.state.fail_count = 0
        if getattr(self, "lock_track_only", False):
            self.state.bad_count = 0
            self.state.mode = "TRACK"
        elif weak:
            self.state.bad_count += 1
            self.state.mode = "WEAK_TRACK" if self.state.bad_count >= self.cfg.weak_after else "TRACK"
        else:
            self.state.bad_count = 0
            self.state.mode = "TRACK"

    def localize_frame(self, frame: np.ndarray, capture_stamp: float | None = None) -> Pose | None:
        """Localize one 720p frame. Experimental flow mode runs the deep matcher only
        every N frames and tracks inliers with KLT in between. Production leaves it off
        and runs the proven XFeat+LighterGlue path every frame."""
        stamp = time.monotonic() if capture_stamp is None else float(capture_stamp)
        if not math.isfinite(stamp) or stamp > time.monotonic() + 0.05:
            self._last_info = {"mode": self.state.mode, "next_mode": self.state.mode,
                               "error": "invalid_capture_stamp", "inliers": 0}
            return None
        self._frame_capture_stamp = stamp
        if self._flow_enabled:
            return self._localize_frame_flow(frame)
        return self._localize_frame_deep(frame)

    def _pnp(self, pts2d, pts3d):
        import pycolmap
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        est = pycolmap.AbsolutePoseEstimationOptions()
        est.ransac.max_error = float(self.cfg.pnp_ransac_max_error)
        est.ransac.random_seed = int(self.cfg.pnp_ransac_random_seed)
        return pycolmap.estimate_and_refine_absolute_pose(
            pts2d.astype(float), pts3d.astype(float), cam, estimation_options=est)

    def _seed_flow_from_deep(
        self, pose: Pose | None, gray: np.ndarray,
    ) -> bool:
        """Seed LK anchors from a strong deep result without changing its gates."""
        info = self._last_info
        strong = (
            pose is not None
            and bool(info.get("accepted", False))
            and not bool(info.get("weak", False))
            and int(info.get("inliers", 0) or 0) >= self._flow_min_seed
        )
        seed2d = self._last_inl_2d
        seed3d = self._last_inl_3d
        if strong and seed2d is not None and seed3d is not None:
            _, unique_2d = np.unique(seed2d, axis=0, return_index=True)
            unique_2d = np.sort(unique_2d)
            _, unique_3d = np.unique(seed3d[unique_2d], axis=0, return_index=True)
            unique = unique_2d[np.sort(unique_3d)]
        else:
            unique = np.empty(0, dtype=np.int64)
        if len(unique) < self._flow_min_seed:
            self._flow_2d = None
            self._flow_3d = None
            self._flow_gray = None
            self._flow_age = 0
            return False
        self._flow_2d = seed2d[unique].copy()
        self._flow_3d = seed3d[unique].copy()
        self._flow_gray = gray
        self._flow_age = 0
        return True

    def _flow_fallback_deep(
        self,
        frame: np.ndarray,
        stage: str,
        *,
        q_cache: dict | None = None,
        prior_counters: dict | None = None,
        **attempt,
    ) -> Pose | None:
        """Discard an unsafe flow attempt and retry the same frame with the deep path."""
        self._flow_2d = None
        self._flow_3d = None
        self._flow_gray = None
        self._flow_age = 0
        force_lighterglue = stage == "mnn_crosscheck"
        if q_cache is None and prior_counters is None and not force_lighterglue:
            pose = self._localize_frame_deep(frame)
        else:
            pose = self._localize_frame_deep(
                frame,
                q_cache=q_cache,
                prior_counters=prior_counters,
                force_lighterglue=force_lighterglue,
            )
        self._seed_flow_from_deep(
            pose, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        self._last_info["flow_stage"] = stage
        self._last_info["flow_fallback"] = True
        self._last_info["flow_fallback_matcher"] = (
            "lighterglue" if force_lighterglue else "default")
        self._last_info.update({f"flow_{key}": value for key, value in attempt.items()})
        return pose

    def _flow_mnn_pose_agrees(
        self, frame: np.ndarray, flow_center: np.ndarray, flow_yaw: float,
    ) -> tuple[bool, dict, dict, dict]:
        """Verify an LK pose with independent map-descriptor MNN correspondences."""
        candidates = self._track_candidates(weak=False)
        if not candidates:
            return False, {"mnn_reason": "no_local_candidates"}, {}, {}
        q_cache: dict = {}
        ret, info = self._localize_with_candidates(
            frame,
            candidates,
            self.cfg.xfeat_topk_track,
            self.cfg.weak_min_inliers,
            matcher_mode="nn",
            include_temporal_cache=False,
            q_cache=q_cache,
        )
        counters = dict(self._frame_counters)
        ok, weak, pose = self._gate(ret, info, "TRACK")
        unique = int(info.get("unique_query_inliers", 0) or 0)
        reproj = info.get("reproj_rms")
        diagnostics = {
            "mnn_inliers": int(info.get("inliers", 0) or 0),
            "mnn_unique_query_inliers": unique,
            "mnn_weak": bool(weak),
            "mnn_reproj_rms": reproj,
            "mnn_feature_ms": info.get("feature_ms"),
            "mnn_match_ms": info.get("match_ms"),
            "mnn_pnp_ms": info.get("pnp_ms"),
        }
        strong = (
            ok
            and pose is not None
            and unique >= self._flow_mnn_min_unique_inliers
            and (reproj is None or float(reproj) <= self._flow_mnn_max_reproj)
        )
        if not strong:
            diagnostics["mnn_reason"] = "quality_gate"
            return False, diagnostics, q_cache, counters
        mnn_center = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        translation = float(np.linalg.norm(mnn_center - flow_center))
        yaw_delta = abs(_angle_diff(float(pose.yaw), float(flow_yaw)))
        diagnostics.update({
            "mnn_translation_delta": translation,
            "mnn_yaw_delta_deg": math.degrees(yaw_delta),
        })
        agrees = (
            translation <= self._flow_mnn_max_translation
            and yaw_delta <= self._flow_mnn_max_yaw
        )
        if not agrees:
            diagnostics["mnn_reason"] = "pose_disagreement"
        return agrees, diagnostics, q_cache, counters

    def _localize_frame_flow(self, frame: np.ndarray) -> Pose | None:
        self._frame_counters = {}
        if self.state.mode != "TRACK" and self._flow_2d is not None:
            self._flow_2d = None
            self._flow_3d = None
            self._flow_gray = None
            self._flow_age = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # deep refresh: (re)acquire reliable 2D<->3D, then track them for the next N frames
        if self._flow_2d is None or self._flow_age >= self._flow_refresh_every:
            pose = self._localize_frame_deep(frame)
            self._seed_flow_from_deep(pose, gray)
            self._last_info["flow_stage"] = "refresh"
            return pose
        # KLT track with forward-backward consistency check
        t0 = time.perf_counter()
        p0 = self._flow_2d.reshape(-1, 1, 2).astype(np.float32)
        nxt, stf, _ = cv2.calcOpticalFlowPyrLK(self._flow_gray, gray, p0, None, **_LK_PARAMS)
        if nxt is None or stf is None:
            return self._flow_fallback_deep(frame, "track_lost")
        back, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self._flow_gray, nxt, None, **_LK_PARAMS)
        if back is None or stb is None:
            return self._flow_fallback_deep(frame, "track_lost")
        fb = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
        good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < self._flow_fb_px)
        n2d, n3d = nxt.reshape(-1, 2)[good], self._flow_3d[good]
        if len(n3d) < self._flow_min_track:
            return self._flow_fallback_deep(frame, "track_lost", tracked=int(len(n3d)))
        # Weighted PnP (Codex #1): drop genuine forward-backward OUTLIER tracks -- the "biased but
        # plausible" drifters that pass the fb<1px gate but bias the pose -- from the POSE SOLVE
        # only, while still TRACKING the full good set. Guarded by an absolute floor so it is
        # DORMANT on clean footage (no regression) and activates only when real drifters appear
        # (fast motion / blur). Single solve on <= the full set -> FPS-neutral-or-positive.
        # (track-age is uniform within a flow segment; per-point 3D uncertainty is not in the bundle.)
        fbg = fb[good]
        drop = (fbg > 2.0 * (float(np.median(fbg)) + 1e-6)) & (fbg > 0.5)
        if bool(drop.any()) and int((~drop).sum()) >= self._flow_min_track:
            s2d, s3d = n2d[~drop], n3d[~drop]
        else:
            s2d, s3d = n2d, n3d
        ret = self._pnp(s2d, s3d)
        inl = int(ret.get("num_inliers", 0)) if ret is not None else 0
        reproj = self._estimate_reproj_rms(ret, s2d, s3d) if ret is not None else None
        if ret is None or inl < self.cfg.weak_min_inliers:
            return self._flow_fallback_deep(
                frame, "track_fail", inliers=inl, reproj_rms=reproj, tracked=int(len(n3d)))
        # Apply the SAME confidence gate as the deep matcher: a low-confidence flow frame is
        # NOT published (caller hovers + refreshes next), so we never fly on a weak fix.
        ratio = inl / max(1, len(s3d))
        if (inl < self._flow_qual_inl or ratio < self._flow_qual_ratio
                or (reproj is not None and (reproj > self._flow_qual_reproj
                                            or reproj > self.cfg.max_reproj_error_track))
                or len(n3d) < self._flow_qual_ntrack):
            return self._flow_fallback_deep(
                frame, "track_lowconf", inliers=inl, reproj_rms=reproj, tracked=int(len(n3d)))
        center, yaw = _pose_center_yaw_from_ret(ret)
        pose_yaw = _pose_map_yaw_from_ret(
            ret, getattr(self, "map_frame", None), yaw
        )
        # max_jump gate (same intent as the deep _gate): a KLT/PnP flip can look inlier/reproj
        # consistent yet jump far. Tracked motion is always small, so a big jump from the last
        # published center is not published (hover + deep refresh next frame).
        if (self.state.last_center is not None
                and float(np.linalg.norm(center - self.state.last_center)) > self.cfg.max_jump):
            return self._flow_fallback_deep(
                frame, "jump_reject", inliers=inl, reproj_rms=reproj, tracked=int(len(n3d)))
        crosscheck = {}
        if self._flow_mnn_crosscheck:
            agrees, crosscheck, q_cache, prior_counters = self._flow_mnn_pose_agrees(
                frame, center, yaw)
            if not agrees:
                return self._flow_fallback_deep(
                    frame,
                    "mnn_crosscheck",
                    q_cache=q_cache,
                    prior_counters=prior_counters,
                    inliers=inl,
                    reproj_rms=reproj,
                    tracked=int(len(n3d)),
                    **crosscheck,
                )
        pose = Pose(x=float(center[0]), y=float(center[1]), z=float(center[2]),
                    yaw=float(pose_yaw), stamp=float(self._frame_capture_stamp))
        self._flow_2d, self._flow_3d, self._flow_gray = n2d, n3d, gray
        self._flow_age += 1
        # keep the deep matcher's motion prior current for the next refresh
        self.state.prev_pose = self.state.last_pose
        self.state.prev_center, self.state.last_center = self.state.last_center, center.astype(np.float32)
        self.state.prev_yaw, self.state.last_yaw = self.state.last_yaw, yaw
        self.state.last_pose = pose
        self.state.fail_count = 0
        self.state.bad_count = 0
        self.state.mode = "TRACK"
        self._last_info = {"mode": "FLOW_TRACK", "next_mode": "FLOW_TRACK", "inliers": inl,
                           "reproj_rms": reproj, "corr3d": int(len(n3d)), "composite_stage": "flow",
                           "flow_stage": "track", "weak": False, "accepted": True,
                           "total_ms": (time.perf_counter() - t0) * 1000.0}
        self._last_info.update({f"flow_{key}": value for key, value in crosscheck.items()})
        self._last_info.update(self._frame_counters)
        return pose

    def _localize_frame_deep(
        self,
        frame: np.ndarray,
        q_cache: dict | None = None,
        prior_counters: dict | None = None,
        force_lighterglue: bool = False,
    ) -> Pose | None:
        """CUDA-OOM-safe wrapper: on GPU out-of-memory, free the cache and treat the
        frame as a missed fix (the flight loop then hovers) instead of letting the
        RuntimeError abort the whole mission."""
        try:
            if q_cache is None and prior_counters is None and not force_lighterglue:
                return self._localize_frame_impl(frame)
            return self._localize_frame_impl(
                frame,
                q_cache=q_cache,
                prior_counters=prior_counters,
                force_lighterglue=force_lighterglue,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
        # Deliberately outside the except block: ``exc`` and its traceback can
        # retain frame-local CUDA tensors until exception scope is exited.
        if q_cache is not None:
            q_cache.clear()
        self._ref_dev_cache.clear()
        self._clear_tracking_history()
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._last_info = {"mode": self.state.mode, "next_mode": self.state.mode,
                           "error": "cuda_oom", "inliers": 0}
        print("[tracker] CUDA OOM in localize_frame; cleared live caches, treating as missed fix",
              flush=True)
        return None

    def _localize_frame_impl(
        self,
        frame: np.ndarray,
        q_cache: dict | None = None,
        prior_counters: dict | None = None,
        force_lighterglue: bool = False,
    ) -> Pose | None:
        xfeat = self.ensure_xfeat()
        _ = xfeat
        start = time.perf_counter()
        self._frame_counters = dict(prior_counters or {})
        mode = self.state.mode
        vpr_ms = 0.0
        if getattr(self, "lock_track_only", False):
            # TRACK FPS microbench: pin TRACK for the whole frame; never MegaLoc.
            self.state.mode = "TRACK"
            mode = "TRACK"
            candidates = self._track_candidates(weak=False)
            if not candidates:
                # Geometry pool empty -> reuse last_refs only (still no MegaLoc).
                n_refs = len(self.map.ref_names)
                fallback = [int(i) for i in (self.state.last_refs or []) if 0 <= int(i) < n_refs]
                if not fallback:
                    fallback = [0]
                candidates = fallback[: max(1, int(self.cfg.local_topk))]
            xfeat_topk = self.cfg.xfeat_topk_track
            min_corr = self.cfg.track_min_inliers
        elif mode == "BOOT_INIT":
            candidates, vpr_ms = self._global_candidates(frame, self.cfg.boot_global_topk)
            xfeat_topk = self.cfg.xfeat_topk_acquire
            min_corr = self.cfg.acquire_min_inliers
        elif mode == "LOST":
            candidates, vpr_ms = self._lost_candidates(frame, self.cfg.lost_global_topk)
            xfeat_topk = self.cfg.xfeat_topk_acquire
            min_corr = self.cfg.acquire_min_inliers
        else:
            candidates = self._track_candidates(weak=(mode == "WEAK_TRACK"))
            if not candidates:
                self.state.mode = "LOST"
                candidates, vpr_ms = self._lost_candidates(frame, self.cfg.lost_global_topk)
                mode = "LOST"
                xfeat_topk = self.cfg.xfeat_topk_acquire
                min_corr = self.cfg.acquire_min_inliers
            else:
                xfeat_topk = self.cfg.xfeat_topk_track
                min_corr = self.cfg.weak_min_inliers

        # Acquisition (BOOT_INIT/LOST) uses its own matcher so a pure-NN
        # matcher_mode cannot weaken global relocalization.
        acquire_matcher = self.cfg.acquire_matcher_mode or self.cfg.matcher_mode

        used_adaptive = False
        if (mode in ("TRACK", "WEAK_TRACK")
                and self.cfg.matcher_mode == "nn_then_lg"
                and not force_lighterglue):
            qc: dict = q_cache if q_cache is not None else {}
            lgp: dict = {}               # cache LG pairs from adaptive pass -> reuse in full retry
            fast_ret, fast_info = self._localize_with_candidates(
                frame, candidates, xfeat_topk, min_corr, matcher_mode="nn",
                include_temporal_cache=self.cfg.temporal_cache_enabled and mode == "TRACK",
                q_cache=qc)
            temporal_summary = _temporal_diagnostics(fast_info)
            fast_info.update({
                "mode": mode,
                "vpr_ms": vpr_ms,
                "candidates": [int(i) for i in candidates],
                "composite_stage": "nn_fast",
                **temporal_summary,
            })
            ok, weak, pose = self._gate(fast_ret, fast_info, mode)
            reproj = fast_info.get("reproj_rms")
            fast_strong = (
                ok and not weak
                and int(fast_info.get("unique_query_inliers", 0))
                >= self.cfg.adaptive_accept_inliers
                and (reproj is None or reproj <= self.cfg.adaptive_accept_reproj)
            )
            if fast_strong:
                info = fast_info
                info["composite_stage"] = "nn_fast_accept"
                _apply_pass_timings(info, [("nn_fast", fast_info)])
            else:
                nn_summary = {
                    "nn_fast_inliers": int(fast_info.get("inliers", 0)),
                    "nn_fast_unique_query_inliers": int(
                        fast_info.get("unique_query_inliers", 0)),
                    "nn_fast_corr3d": int(fast_info.get("corr3d", 0)),
                    "nn_fast_reproj_rms": fast_info.get("reproj_rms"),
                    "nn_fast_accepted_by_gate": bool(ok),
                    **temporal_summary,
                }
                if self.cfg.adaptive_first_topk > 0 and len(candidates) > self.cfg.adaptive_first_topk:
                    first_candidates = list(candidates)[:self.cfg.adaptive_first_topk]
                    ret, info = self._localize_with_candidates(
                        frame, first_candidates, xfeat_topk, min_corr, matcher_mode="lighterglue",
                        q_cache=qc, lg_pairs_out=lgp)
                    info.update({
                        "mode": mode,
                        "vpr_ms": vpr_ms,
                        "candidates": [int(i) for i in first_candidates],
                        "adaptive_stage": "first",
                        "adaptive_full_candidates": [int(i) for i in candidates],
                        "composite_stage": "lg_adaptive_after_nn",
                        **nn_summary,
                    })
                    ok, weak, pose = self._gate(ret, info, mode)
                    reproj = info.get("reproj_rms")
                    strong_enough = (
                        ok and not weak and int(info.get("inliers", 0)) >= self.cfg.adaptive_accept_inliers
                        and (reproj is None or reproj <= self.cfg.adaptive_accept_reproj)
                    )
                    if strong_enough:
                        used_adaptive = True
                        _apply_pass_timings(
                            info, [("nn_fast", fast_info), ("lg_first", info)])
                    else:
                        first_info = info
                        ret, info = self._localize_with_candidates(
                            frame, candidates, xfeat_topk, min_corr, matcher_mode="lighterglue",
                            q_cache=qc, lg_pairs_in=lgp)
                        info.update({
                            "adaptive_stage": "full_retry",
                            "adaptive_first_accepted": bool(ok),
                            "composite_stage": "lg_full_after_nn",
                            **nn_summary,
                        })
                        ok, weak, pose = self._gate(ret, info, mode)
                        _apply_pass_timings(info, [
                            ("nn_fast", fast_info),
                            ("lg_first", first_info),
                            ("lg_full", info),
                        ])
                else:
                    ret, info = self._localize_with_candidates(
                        frame, candidates, xfeat_topk, min_corr, matcher_mode="lighterglue",
                        q_cache=qc)
                    info.update({"composite_stage": "lg_full_after_nn", **nn_summary})
                    ok, weak, pose = self._gate(ret, info, mode)
                    _apply_pass_timings(
                        info, [("nn_fast", fast_info), ("lg_full", info)])
        elif mode in ("TRACK", "WEAK_TRACK") and self.cfg.adaptive_first_topk > 0 and len(candidates) > self.cfg.adaptive_first_topk:
            first_candidates = list(candidates)[:self.cfg.adaptive_first_topk]
            include_cache = self.cfg.temporal_cache_enabled and mode == "TRACK"
            qc = q_cache if q_cache is not None else {}
            lgp: dict = {}
            forced_matcher = "lighterglue" if force_lighterglue else None
            ret, info = self._localize_with_candidates(
                frame, first_candidates, xfeat_topk, min_corr,
                matcher_mode=forced_matcher,
                include_temporal_cache=include_cache,
                q_cache=qc,
                lg_pairs_out=lgp)
            info.update({
                "mode": mode,
                "vpr_ms": vpr_ms,
                "candidates": [int(i) for i in first_candidates],
                "adaptive_stage": "first",
                "adaptive_full_candidates": [int(i) for i in candidates],
            })
            if force_lighterglue:
                info["composite_stage"] = "lg_adaptive_forced"
            ok, weak, pose = self._gate(ret, info, mode)
            reproj = info.get("reproj_rms")
            strong_enough = (
                ok and not weak and int(info.get("inliers", 0)) >= self.cfg.adaptive_accept_inliers
                and (reproj is None or reproj <= self.cfg.adaptive_accept_reproj)
            )
            if strong_enough:
                used_adaptive = True
                _apply_pass_timings(info, [("adaptive_first", info)])
            else:
                first_info = info
                ret, info = self._localize_with_candidates(
                    frame, candidates, xfeat_topk, min_corr,
                    matcher_mode=forced_matcher,
                    include_temporal_cache=include_cache,
                    q_cache=qc,
                    lg_pairs_in=lgp)
                info["adaptive_stage"] = "full_retry"
                info["adaptive_first_accepted"] = bool(ok)
                if force_lighterglue:
                    info["composite_stage"] = "lg_full_forced"
                ok, weak, pose = self._gate(ret, info, mode)
                _apply_pass_timings(info, [
                    ("adaptive_first", first_info),
                    ("adaptive_full", info),
                ])
        elif mode in ("TRACK", "WEAK_TRACK") and force_lighterglue:
            ret, info = self._localize_with_candidates(
                frame,
                candidates,
                xfeat_topk,
                min_corr,
                matcher_mode="lighterglue",
                q_cache=q_cache if q_cache is not None else {},
            )
            info["composite_stage"] = "lg_forced"
            ok, weak, pose = self._gate(ret, info, mode)
            _apply_pass_timings(info, [("lg_forced", info)])
        else:
            is_acquire = mode in ("BOOT_INIT", "LOST")
            ret, info = self._localize_with_candidates(
                frame, candidates, xfeat_topk, min_corr,
                matcher_mode=acquire_matcher if is_acquire else None,
                include_temporal_cache=(not is_acquire)
                and self.cfg.temporal_cache_enabled and mode == "TRACK")
            ok, weak, pose = self._gate(ret, info, mode)
        info.update({
            "mode": mode,
            "vpr_ms": vpr_ms,
            "candidates": [int(i) for i in candidates] if not used_adaptive else info.get("candidates", []),
        })
        cache_payload = info.pop("_temporal_cache_payload", None)
        cache_context = info.pop("_temporal_cache_context", None)
        cache_updated = False

        # TRACK failure escalates to WEAK_TRACK/LOST. BOOT/LOST stay global and wait
        # for external yaw scan / next frame.
        if not ok:
            self.state.fail_count += 1
            self.state.bad_count += 1
            if getattr(self, "lock_track_only", False):
                # Microbench: stay in TRACK forever; keep priors for next frame.
                self.state.mode = "TRACK"
                self.state.fail_count = 0
                self.state.bad_count = 0
                self.state.lost_streak = 0
                self.state.lost_since_stamp = None
                self._age_temporal_cache()
            elif mode == "TRACK":
                self.state.mode = "WEAK_TRACK" if self.state.fail_count < self.cfg.lost_after else "LOST"
            elif mode == "WEAK_TRACK":
                self.state.mode = "LOST"
            elif mode in ("BOOT_INIT", "LOST"):
                self.state.mode = mode
            info["accepted"] = False
            if self.state.mode == "LOST" and not getattr(self, "lock_track_only", False):
                # Keep last_center / last_refs so the next LOST frame prefers nearby
                # map places on a continuous path. Only drop short-term helpers.
                if mode == "LOST":
                    self.state.lost_streak = int(self.state.lost_streak) + 1
                else:
                    self.state.lost_streak = 1
                if self.state.lost_since_stamp is None:
                    self.state.lost_since_stamp = float(self._frame_capture_stamp)
                info["lost_elapsed_s"] = max(
                    0.0,
                    float(self._frame_capture_stamp) - float(self.state.lost_since_stamp),
                )
                self._clear_short_term_track_helpers()
            elif not getattr(self, "lock_track_only", False):
                self.state.lost_streak = 0
                self.state.lost_since_stamp = None
                self._age_temporal_cache()
            pose = None
        else:
            self.state.lost_streak = 0
            self.state.lost_since_stamp = None
            self._publish_success(pose, info, weak, source_mode=mode)
            reproj = info.get("reproj_rms")
            seed_ok = (
                not weak
                and int(info.get("inliers", 0)) >= self.cfg.temporal_cache_seed_min_inliers
                and (reproj is None or reproj <= self.cfg.temporal_cache_seed_max_reproj)
            )
            if seed_ok:
                if (
                    self.cfg.temporal_cache_seed_mode == "inliers"
                    and cache_payload is None
                    and cache_context is not None
                ):
                    cache_payload = self._cache_payload_from_ref_inliers(*cache_context)
                    if cache_payload:
                        info["temporal_cache_candidate_anchors"] = int(len(cache_payload["xyz"]))
                if self.cfg.temporal_cache_seed_mode == "inliers" and cache_payload:
                    # seed from this frame's PnP-validated inliers (smaller,
                    # higher-precision anchor set); falls through to full-ref
                    # seeding when no inlier payload exists (e.g. the cache
                    # itself contributed correspondences this frame).
                    cache_updated = self._set_temporal_cache(cache_payload)
                    if cache_updated:
                        self._last_seed_refs = ()        # inlier seeds are not a ref-set
                if not cache_updated:
                    seed_refs = tuple(int(r) for r in (info.get("used_refs") or info.get("candidates", [])[:3]))
                    if seed_refs and seed_refs == self._last_seed_refs and len(self.temporal_cache) > 0:
                        self.temporal_cache.age = 0      # same refs -> reuse cache, skip CPU gather + re-upload
                        cache_updated = True
                    else:
                        ref_payload = self._cache_payload_from_refs(
                            seed_refs, str(info.get("composite_stage", "success_refs")))
                        if ref_payload is None and cache_payload is None and cache_context is not None:
                            cache_payload = self._cache_payload_from_ref_inliers(*cache_context)
                            if cache_payload:
                                info["temporal_cache_candidate_anchors"] = int(len(cache_payload["xyz"]))
                        cache_updated = self._set_temporal_cache(ref_payload or cache_payload)
                        if cache_updated:
                            self._last_seed_refs = seed_refs
            if not cache_updated:
                self._age_temporal_cache()
            info["accepted"] = True
            info["weak"] = bool(weak)

        info["temporal_cache_updated"] = bool(cache_updated)
        info["temporal_cache_size_after"] = int(len(self.temporal_cache))
        info["temporal_cache_age_after"] = int(self.temporal_cache.age)
        info["temporal_cache_source_stage"] = self.temporal_cache.source_stage
        info["timing_synced"] = bool(_LOC_TIMING)
        info.update(self._frame_counters)
        # diagnostics only: composite localization quality in [0,1] from existing
        # metrics (inliers, corr, reproj, coverage, temporal consistency, weak).
        # Logged for benchmark/UI; nothing gates on it.
        if info.get("accepted"):
            inl = int(info.get("inliers", 0) or 0)
            corr = int(info.get("corr3d", 0) or 0)
            rms = info.get("reproj_rms")
            cov = info.get("inlier_coverage")
            jump = info.get("jump_from_pred")
            consistency = 1.0 if jump is None else max(0.0, 1.0 - jump / max(self.cfg.max_jump, 1e-6))
            info["quality_score"] = round(
                0.35 * min(1.0, inl / 120.0)
                + 0.15 * min(1.0, corr / 300.0)
                + 0.20 * (1.0 - min(1.0, (3.0 if rms is None else float(rms)) / 6.0))
                + 0.10 * (0.5 if cov is None else float(cov))
                + 0.10 * consistency
                + 0.10 * (0.0 if info.get("weak") else 1.0), 4)
        info["next_mode"] = self.state.mode
        info["bad_count"] = int(self.state.bad_count)
        info["fail_count"] = int(self.state.fail_count)
        info["total_ms"] = (time.perf_counter() - start) * 1000.0
        self._last_info = info
        return pose

    def get_pose(self) -> Pose | None:
        sample = self.frame_source()
        if sample is None:
            return None
        if isinstance(sample, tuple):
            if len(sample) != 2:
                self._last_info = {"mode": self.state.mode, "next_mode": self.state.mode,
                                   "error": "invalid_frame_sample", "inliers": 0}
                return None
            frame, capture_stamp = sample
        else:
            # Backward-compatible offline/simulation sources have no capture stamp;
            # their synchronous caller time is the best available value.
            frame, capture_stamp = sample, time.monotonic()
        return self.localize_frame(frame, capture_stamp=capture_stamp)

    @property
    def last_info(self) -> dict:
        return self._last_info


if __name__ == "__main__":
    print(__doc__)
