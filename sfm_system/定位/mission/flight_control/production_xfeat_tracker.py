#!/usr/bin/env python3
"""Production XFeat tracker state machine.

Production default for the ANAFI/GLOMAP runtime:

- BOOT_INIT / LOST: MegaLoc global retrieval topK=30 -> XFeat+LighterGlue -> PnP.
- TRACK / WEAK_TRACK: no global retrieval; vpr_ms is intentionally 0.  Candidate
  refs come only from pose prior, nearby camera centers, and SfM covisibility.
- TRACK / WEAK_TRACK first try XFeat mutual-NN fast pass.  Accept only when the
  geometric result is strong enough (>=100 inliers, reproj <=3.5 px, gate pass).
- If mutual-NN is not strong enough, fall back to XFeat+LighterGlue adaptive:
  first top3 local refs; if not strong enough, retry top5 local refs.

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
import json
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
from reloc_localizer_xfeat import (
    Camera,
    XFeatRelocMap,
    _to_device_feats,
    extract_xfeat,
    load_xfeat,
    DEVICE,
)

MEGALOC_REPO = "gmberton/MegaLoc"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def _ret_inlier_mask(ret, n_corr: int) -> np.ndarray | None:
    if ret is None:
        return None
    for key in ("inliers", "inlier_mask"):
        if key in ret:
            mask = np.asarray(ret[key], bool)
            if mask.shape[0] == int(n_corr):
                return mask
    return None


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


def mutual_nn_pairs(q_desc: torch.Tensor, r_desc: torch.Tensor, min_score: float = 0.8) -> torch.Tensor:
    """Mutual-nearest-neighbour descriptor matches for XFeat descriptors.

    Returns an int64 tensor of shape (K,2), [query_idx, ref_idx].  This is the
    cheap alternative to LighterGlue for a fast first pass; PnP/RANSAC remains
    the geometric verifier.
    """
    if q_desc.ndim != 2 or r_desc.ndim != 2 or len(q_desc) == 0 or len(r_desc) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=q_desc.device)
    qd = F.normalize(q_desc.float(), dim=-1)
    rd = F.normalize(r_desc.float(), dim=-1)
    sim = qd @ rd.T
    best_r = sim.argmax(dim=1)
    best_q = sim.argmax(dim=0)
    qi = torch.arange(sim.shape[0], device=sim.device)
    scores = sim[qi, best_r]
    ok = (best_q[best_r] == qi) & (scores >= float(min_score))
    return torch.stack([qi[ok], best_r[ok]], dim=1).long()


@dataclass
class ProductionConfig:
    boot_global_topk: int = 30
    lost_global_topk: int = 30
    weak_global_topk: int = 0          # production: MegaLoc is never used in WEAK_TRACK
    local_topk: int = 5
    weak_local_topk: int = 8
    near_pool: int = 24
    covis_per_ref: int = 20
    radius: float = 0.8
    max_yaw_diff_deg: float = 90.0
    xfeat_topk_track: int = 1300
    xfeat_topk_acquire: int = 2048
    min_conf: float = 0.1
    acquire_min_inliers: int = 80
    track_min_inliers: int = 50
    weak_min_inliers: int = 30
    good_inliers: int = 80
    max_reproj_error_acquire: float = 5.0
    max_reproj_error_track: float = 6.0
    pnp_ransac_max_error: float = 5.0
    max_jump: float = 2.0
    weak_after: int = 2
    lost_after: int = 2
    matcher_mode: str = "nn_then_lg"  # nn_then_lg | lighterglue | nn
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
    temporal_cache_enabled: bool = True
    temporal_cache_min_anchors: int = 80
    temporal_cache_max_anchors: int = 2048
    temporal_cache_max_age: int = 2
    temporal_cache_min_score: float = 0.85
    temporal_cache_seed_min_inliers: int = 150
    temporal_cache_seed_max_reproj: float = 3.5


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


def predict_center(state: RuntimeState) -> np.ndarray | None:
    if state.last_center is None:
        return None
    if state.prev_center is None:
        return state.last_center.copy()
    return state.last_center + (state.last_center - state.prev_center)


def predict_yaw(state: RuntimeState) -> float | None:
    return state.last_yaw


def select_local_candidates(ref_names: list[str], ref_centers: np.ndarray,
                            ref_yaws: np.ndarray | None, covis: dict | None,
                            state: RuntimeState, cfg: ProductionConfig,
                            topk: int | None = None) -> list[int]:
    """Select TRACK refs using geometry only: KDTree-equivalent distance + covis.

    Uses a brute-force distance sort because the map has only ~2k refs; this avoids
    requiring a serialized KDTree in the bundle and is fast enough (<1 ms).
    """
    C = predict_center(state)
    if C is None or ref_centers is None:
        return []
    topk = int(topk or cfg.local_topk)
    centers = np.asarray(ref_centers, np.float32)
    d = np.linalg.norm(centers - C[None, :], axis=1)
    finite = np.isfinite(d)
    order = [int(i) for i in np.argsort(np.where(finite, d, np.inf))]

    pred_yaw = predict_yaw(state)
    max_yaw = math.radians(cfg.max_yaw_diff_deg)

    near = []
    for i in order:
        if not finite[i]:
            continue
        if d[i] > cfg.radius and len(near) >= cfg.near_pool:
            break
        if pred_yaw is not None and ref_yaws is not None and np.isfinite(ref_yaws[i]):
            if abs(_angle_diff(float(ref_yaws[i]), pred_yaw)) > max_yaw:
                continue
        near.append(i)
        if len(near) >= cfg.near_pool:
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
        if pred_yaw is not None and ref_yaws is not None and np.isfinite(ref_yaws[int(i)]):
            if abs(_angle_diff(float(ref_yaws[int(i)]), pred_yaw)) > max_yaw:
                continue
        valid.append(int(i))

    last_set = set(int(i) for i in state.last_refs)

    def score(i: int) -> float:
        dist_score = math.exp(-float(d[i]) / max(cfg.radius, 1e-6))
        if pred_yaw is not None and ref_yaws is not None and np.isfinite(ref_yaws[i]):
            yaw_score = math.exp(-abs(_angle_diff(float(ref_yaws[i]), pred_yaw)) / math.radians(45.0))
        else:
            yaw_score = 0.5
        recent = 1.0 if i in last_set else 0.0
        return 2.0 * dist_score + 0.8 * yaw_score + recent

    valid.sort(key=score, reverse=True)
    return valid[:topk]


def spatially_cap_correspondences(pts2d: np.ndarray, pts3d: np.ndarray, max_total: int,
                                  width: int, height: int, grid_x: int = 8,
                                  grid_y: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most max_total 2D-3D correspondences with coarse image diversity."""
    if max_total <= 0 or len(pts3d) <= max_total:
        return pts2d, pts3d
    pts2d = np.asarray(pts2d)
    pts3d = np.asarray(pts3d)
    gx = np.clip((pts2d[:, 0] / max(width, 1) * grid_x).astype(int), 0, grid_x - 1)
    gy = np.clip((pts2d[:, 1] / max(height, 1) * grid_y).astype(int), 0, grid_y - 1)
    buckets: dict[int, list[int]] = {}
    for i, (x, y) in enumerate(zip(gx, gy)):
        buckets.setdefault(int(y * grid_x + x), []).append(i)
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
    idx = np.asarray(picked, dtype=int)
    return pts2d[idx], pts3d[idx]


class MegaLocLayer:
    """MegaLoc retrieval layer with cached reference descriptors."""

    def __init__(self, ref_desc: np.ndarray, input_size: int = 322, device: str = DEVICE):
        self.ref_desc = np.asarray(ref_desc, np.float32)
        self.input_size = int(input_size)
        self.device = device
        self._model = None

    def model(self):
        if self._model is None:
            self._model = torch.hub.load(MEGALOC_REPO, "get_trained_model", trust_repo=True).eval().to(self.device)
        return self._model

    @torch.inference_mode()
    def extract_one(self, rgb: np.ndarray) -> np.ndarray:
        model = self.model()
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(self.device)[None] / 255.0
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic",
                          align_corners=False, antialias=True)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        d = model((x - mean) / std).float().squeeze(0).detach().cpu().numpy()
        return d / (np.linalg.norm(d) + 1e-12)

    def topk(self, rgb: np.ndarray, k: int) -> tuple[list[int], float]:
        _sync(); t0 = time.perf_counter()
        q = self.extract_one(rgb)
        sims = self.ref_desc @ q
        cand = [int(i) for i in np.argsort(-sims)[:int(k)]]
        _sync(); return cand, (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def load_cache(cache_npy: Path, ref_names: list[str], input_size: int = 322,
                   device: str = DEVICE) -> "MegaLocLayer":
        desc = np.load(cache_npy).astype(np.float32)
        if desc.shape[0] != len(ref_names):
            raise ValueError(f"MegaLoc cache row mismatch: {desc.shape[0]} vs {len(ref_names)}")
        desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
        return MegaLocLayer(desc, input_size=input_size, device=device)

    @staticmethod
    @torch.inference_mode()
    def build_cache(ref_names: list[str], image_root: Path, cache_npy: Path,
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
        cache_npy.parent.mkdir(parents=True, exist_ok=True)
        # np.save appends ".npy" unless given a file object; our cache paths end
        # in ".npz", so a bare np.save(path) writes "<name>.npz.npy" and the
        # exists() check above never sees it (cache rebuilt every run).
        with open(cache_npy, "wb") as f:
            np.save(f, desc)
        if meta_json is not None:
            meta_json.write_text(json.dumps({
                "model": "MegaLoc",
                "repo": MEGALOC_REPO,
                "input_size": input_size,
                "refs": len(ref_names),
                "ref_names": ref_names,
                "image_root": str(image_root),
            }, ensure_ascii=False, indent=2))
        return MegaLocLayer(desc, input_size=input_size, device=device)


class ProductionXFeatTracker(Localizer):
    def __init__(self, reloc_map: XFeatRelocMap, megaloc: MegaLocLayer,
                 frame_source, query_cam: Camera,
                 cfg: ProductionConfig | None = None):   # avoid shared mutable default
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
        if megaloc.ref_desc.shape[0] != n_refs:
            raise ValueError(
                f"MegaLoc descriptor mismatch: refs={n_refs} descriptors={megaloc.ref_desc.shape[0]}"
            )
        self.map = reloc_map
        self.meg = megaloc
        self.frame_source = frame_source
        self.cam = query_cam
        self.cfg = cfg if cfg is not None else ProductionConfig()
        # Cache each ref's XFeat features on the GPU so they are uploaded once, not every frame.
        self._ref_dev_cache: OrderedDict[int, dict] = OrderedDict()
        self.state = RuntimeState()
        self.temporal_cache = TemporalAnchorCache()
        self._xfeat = None
        self._last_info: dict = {}
        self._last_inl_2d = None       # accepted inlier query 2D (for optical-flow seeding)
        self._last_inl_3d = None       # accepted inlier map 3D
        self._last_seed_refs: tuple = ()  # last temporal-cache seed ref-set (skip identical rebuilds)
        # optical-flow tracking mode (SFM_FLOW_TRACK=1): deep XFeat+LighterGlue refresh every
        # N frames, KLT-track the inlier keypoints in between (3.5x faster, ~0.015u drift).
        self._flow_enabled = os.environ.get("SFM_FLOW_TRACK", "0") == "1"
        self._flow_refresh_every = int(os.environ.get("SFM_FLOW_REFRESH", "6"))
        self._flow_2d = None
        self._flow_3d = None
        self._flow_gray = None
        self._flow_age = 0
        self._flow_min_seed = 20       # min deep inliers to start tracking
        self._flow_min_track = 20      # tracked pts below this -> refresh next frame
        self._flow_fb_px = 1.0         # forward-backward flow consistency (px)
        self._flow_qual_inl = 50       # refresh triggers (per the user's plan)
        self._flow_qual_ratio = 0.25
        self._flow_qual_reproj = 4.0
        self._flow_qual_ntrack = 100

    def ensure_models(self):
        self.meg.model()
        if self._xfeat is None:
            self._xfeat = load_xfeat(max(self.cfg.xfeat_topk_acquire, self.cfg.xfeat_topk_track))
        return self._xfeat

    def _global_candidates(self, frame: np.ndarray, k: int) -> tuple[list[int], float]:
        return self.meg.topk(frame, k)

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

        info = {"raw_matches": 0, "corr3d": 0, "used_refs": []}
        active_matcher = matcher_mode or self.cfg.matcher_mode
        if active_matcher == "nn_then_lg":
            active_matcher = "lighterglue"
        info["matcher_mode"] = active_matcher
        xfeat = self.ensure_models()
        _sync(); t0 = time.perf_counter()
        # extract query features ONCE per frame; later composite passes reuse them (same
        # frame + topk -> identical descriptors/keypoints, so the PnP output is unchanged)
        if q_cache is not None and q_cache.get("qkp") is not None:
            q_dev, qkp = q_cache["q_dev"], q_cache["qkp"]
        else:
            q_feats = extract_xfeat(xfeat, frame, xfeat_topk)
            q_dev = _to_device_feats(q_feats)
            qkp = q_feats["keypoints"].detach().cpu().numpy()
            if q_cache is not None:
                q_cache["q_dev"], q_cache["qkp"] = q_dev, qkp
        _sync(); info["feature_ms"] = (time.perf_counter() - t0) * 1000.0

        pts2d: list[np.ndarray] = []
        pts3d: list[np.ndarray] = []
        corr_ref_ids: list[int] = []
        corr_ref_kp_ids: list[int] = []
        _sync(); t0 = time.perf_counter()
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
                pairs_t = mutual_nn_pairs(
                    q_dev["descriptors"],
                    self.temporal_cache.descriptors,
                    self.cfg.temporal_cache_min_score,
                )
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
                        pts2d.append(qkp[qa_c])
                        pts3d.append(cache_xyz[finite])
                        corr_ref_ids.extend(cache_ref_ids[finite].astype(int).tolist())
                        corr_ref_kp_ids.extend([-1] * int(finite.sum()))
                    temporal_refs = sorted(set(int(r) for r in cache_ref_ids if int(r) >= 0))
                    if temporal_refs:
                        info["temporal_cache_used_refs"] = temporal_refs
                    info["temporal_cache_corr3d"] = int(np.sum(finite))
            else:
                info["temporal_cache_skipped"] = True
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
            if active_matcher == "nn":
                pairs_t = mutual_nn_pairs(q_dev["descriptors"], r_dev["descriptors"], self.cfg.nn_min_score)
                pairs = pairs_t.detach().cpu().numpy()
            elif lg_pairs_in is not None and ref_id in lg_pairs_in:
                pairs = lg_pairs_in[ref_id]              # reuse this ref's LG match from the adaptive pass
            else:
                _mk0, _mk1, pairs = xfeat.match_lighterglue(q_dev, r_dev, min_conf=self.cfg.min_conf)
            if lg_pairs_out is not None and active_matcher != "nn":
                lg_pairs_out[ref_id] = pairs
            info["raw_matches"] += int(len(pairs))
            if len(pairs):                               # vectorized gather (was a python loop)
                pr = np.asarray(pairs)
                qa_i = pr[:, 0].astype(int)
                rb_i = pr[:, 1].astype(int)
                Xr = ref.xyz[rb_i]
                fin = ~np.isnan(Xr).any(axis=1)
                qa_i, rb_i, Xr = qa_i[fin], rb_i[fin], Xr[fin]
                if self.cfg.max_corr_per_ref > 0 and len(qa_i) > self.cfg.max_corr_per_ref:
                    qa_i = qa_i[:self.cfg.max_corr_per_ref]
                    rb_i = rb_i[:self.cfg.max_corr_per_ref]
                    Xr = Xr[:self.cfg.max_corr_per_ref]
                if len(qa_i):
                    pts2d.append(qkp[qa_i])
                    pts3d.append(Xr)
                    corr_ref_ids.extend([ref_id] * len(qa_i))
                    corr_ref_kp_ids.extend(rb_i.tolist())
                    info["used_refs"].append(ref_id)
        _sync(); info["match_ms"] = (time.perf_counter() - t0) * 1000.0
        info["used_refs"] = list(dict.fromkeys(int(i) for i in info["used_refs"] if int(i) >= 0))
        # pts2d/pts3d are now lists of per-ref arrays -> concatenate once (was per-point append)
        pts2d_arr = (np.concatenate(pts2d, axis=0).astype(float)
                     if pts2d else np.zeros((0, 2), float))
        pts3d_arr = (np.concatenate(pts3d, axis=0).astype(float)
                     if pts3d else np.zeros((0, 3), float))
        info["corr3d"] = int(len(pts3d_arr))
        if len(pts3d_arr) < min_corr:
            info["inliers"] = 0
            info["reproj_rms"] = None
            return None, info
        if self.cfg.max_corr_total > 0 and len(pts3d_arr) > self.cfg.max_corr_total:
            pts2d_arr, pts3d_arr = spatially_cap_correspondences(
                pts2d_arr, pts3d_arr, self.cfg.max_corr_total, self.cam.width, self.cam.height)
            info["corr3d_pruned"] = int(len(pts3d_arr))
        _sync(); t0 = time.perf_counter()
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        est_opts = pycolmap.AbsolutePoseEstimationOptions()
        est_opts.ransac.max_error = float(self.cfg.pnp_ransac_max_error)
        ret = pycolmap.estimate_and_refine_absolute_pose(
            pts2d_arr, pts3d_arr, cam, estimation_options=est_opts)
        _sync(); info["pnp_ms"] = (time.perf_counter() - t0) * 1000.0
        info["inliers"] = 0 if ret is None else int(ret.get("num_inliers", 0))
        info["reproj_rms"] = None if ret is None else self._estimate_reproj_rms(ret, pts2d_arr, pts3d_arr)
        # additive hook: expose the accepted inlier 2D<->3D so an optical-flow tracker can
        # seed on refresh (does NOT affect pose/gate/behavior; just stores the arrays)
        _m = _ret_inlier_mask(ret, len(pts3d_arr))
        if ret is not None and _m is not None:
            self._last_inl_2d, self._last_inl_3d = pts2d_arr[_m].astype(np.float32), pts3d_arr[_m].astype(np.float32)
        elif ret is not None:
            self._last_inl_2d, self._last_inl_3d = pts2d_arr.astype(np.float32), pts3d_arr.astype(np.float32)
        else:
            self._last_inl_2d = self._last_inl_3d = None
        if ret is not None and self.cfg.max_corr_total <= 0 and all(kp >= 0 for kp in corr_ref_kp_ids):
            payload = self._cache_payload_from_ref_inliers(
                ret, pts3d_arr, corr_ref_ids, corr_ref_kp_ids, active_matcher)
            if payload:
                info["_temporal_cache_payload"] = payload
                info["temporal_cache_candidate_anchors"] = int(len(payload["xyz"]))
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
        xfeat = self.ensure_models()
        _sync(); t0 = time.perf_counter()
        q_feats = extract_xfeat(xfeat, frame, xfeat_topk)
        q_dev = _to_device_feats(q_feats)
        qkp = q_feats["keypoints"].detach().cpu().numpy()
        _sync(); info["feature_ms"] = (time.perf_counter() - t0) * 1000.0

        _sync(); t0 = time.perf_counter()
        pairs_t = mutual_nn_pairs(
            q_dev["descriptors"],
            self.temporal_cache.descriptors,
            self.cfg.temporal_cache_min_score,
        )
        pairs = pairs_t.detach().cpu().numpy()
        _sync(); info["match_ms"] = (time.perf_counter() - t0) * 1000.0
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

        _sync(); t0 = time.perf_counter()
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        est_opts = pycolmap.AbsolutePoseEstimationOptions()
        est_opts.ransac.max_error = float(self.cfg.pnp_ransac_max_error)
        ret = pycolmap.estimate_and_refine_absolute_pose(
            pts2d_arr, pts3d_arr, cam, estimation_options=est_opts)
        _sync(); info["pnp_ms"] = (time.perf_counter() - t0) * 1000.0
        info["inliers"] = 0 if ret is None else int(ret.get("num_inliers", 0))
        info["reproj_rms"] = None if ret is None else self._estimate_reproj_rms(ret, pts2d_arr, pts3d_arr)
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
        pose = Pose(x=float(center[0]), y=float(center[1]), z=float(center[2]), yaw=yaw, stamp=time.monotonic())
        ninl = int(info.get("inliers", 0))
        reproj = info.get("reproj_rms")
        is_acquire = mode in ("BOOT_INIT", "LOST", "GLOBAL_RETRY")
        if is_acquire:
            if ninl < self.cfg.acquire_min_inliers:
                return False, False, pose
            if reproj is not None and reproj > self.cfg.max_reproj_error_acquire:
                return False, False, pose
            return True, ninl < self.cfg.good_inliers, pose

        # TRACK / WEAK_TRACK: allow weak-but-usable results down to weak_min_inliers.
        if ninl < self.cfg.weak_min_inliers:
            return False, False, pose
        if reproj is not None and reproj > self.cfg.max_reproj_error_track:
            return False, False, pose
        pred = predict_center(self.state)
        if pred is not None and self.cfg.max_jump > 0:
            jump = float(np.linalg.norm(center - pred))
            info["jump_from_pred"] = jump
            if jump > self.cfg.max_jump:
                return False, False, pose
        weak = ninl < self.cfg.track_min_inliers or ninl < self.cfg.good_inliers
        return True, weak, pose

    def _publish_success(self, pose: Pose, info: dict, weak: bool):
        self.state.prev_pose = self.state.last_pose
        self.state.prev_center = self.state.last_center
        self.state.prev_yaw = self.state.last_yaw
        self.state.last_pose = pose
        self.state.last_center = np.array([pose.x, pose.y, pose.z], np.float32)
        self.state.last_yaw = float(pose.yaw)
        self.state.last_refs = list(info.get("used_refs") or info.get("candidates", [])[:3])
        self.state.fail_count = 0
        if weak:
            self.state.bad_count += 1
            self.state.mode = "WEAK_TRACK" if self.state.bad_count >= self.cfg.weak_after else "TRACK"
        else:
            self.state.bad_count = 0
            self.state.mode = "TRACK"

    def localize_frame(self, frame: np.ndarray) -> Pose | None:
        """Localize one 720p frame. With SFM_FLOW_TRACK=1: run the deep XFeat+LighterGlue
        matcher only every N frames and KLT-track the inlier keypoints in between (much
        faster, small drift). Otherwise run the proven deep matcher every frame."""
        if self._flow_enabled:
            return self._localize_frame_flow(frame)
        return self._localize_frame_deep(frame)

    def _pnp(self, pts2d, pts3d):
        import pycolmap
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        est = pycolmap.AbsolutePoseEstimationOptions()
        est.ransac.max_error = float(self.cfg.pnp_ransac_max_error)
        return pycolmap.estimate_and_refine_absolute_pose(
            pts2d.astype(float), pts3d.astype(float), cam, estimation_options=est)

    def _localize_frame_flow(self, frame: np.ndarray) -> Pose | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # deep refresh: (re)acquire reliable 2D<->3D, then track them for the next N frames
        if self._flow_2d is None or self._flow_age >= self._flow_refresh_every:
            pose = self._localize_frame_deep(frame)
            refresh_info = self._last_info
            strong_refresh = (
                pose is not None
                and bool(refresh_info.get("accepted", False))
                and not bool(refresh_info.get("weak", False))
                and int(refresh_info.get("inliers", 0) or 0) >= self._flow_qual_inl
            )
            if (strong_refresh and self._last_inl_2d is not None
                    and len(self._last_inl_2d) >= self._flow_min_seed):
                self._flow_2d = self._last_inl_2d.copy()
                self._flow_3d = self._last_inl_3d.copy()
                self._flow_gray = gray
                self._flow_age = 0
            else:
                self._flow_2d = None
            self._last_info["flow_stage"] = "refresh"
            return pose
        # KLT track with forward-backward consistency check
        t0 = time.perf_counter()
        p0 = self._flow_2d.reshape(-1, 1, 2).astype(np.float32)
        nxt, stf, _ = cv2.calcOpticalFlowPyrLK(self._flow_gray, gray, p0, None, **_LK_PARAMS)
        if nxt is None or stf is None:
            self._flow_2d = None
            self._last_info = {"mode": "FLOW_TRACK", "next_mode": "FLOW_TRACK", "inliers": 0,
                               "reproj_rms": None, "flow_stage": "track_lost", "weak": False,
                               "accepted": False}
            return None
        back, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self._flow_gray, nxt, None, **_LK_PARAMS)
        if back is None or stb is None:
            self._flow_2d = None
            self._last_info = {"mode": "FLOW_TRACK", "next_mode": "FLOW_TRACK", "inliers": 0,
                               "reproj_rms": None, "flow_stage": "track_lost", "weak": False,
                               "accepted": False}
            return None
        fb = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
        good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < self._flow_fb_px)
        n2d, n3d = nxt.reshape(-1, 2)[good], self._flow_3d[good]
        if len(n3d) < self._flow_min_track:
            self._flow_2d = None                     # lost -> deep refresh next frame
            self._last_info = {"mode": "FLOW_TRACK", "next_mode": "FLOW_TRACK", "inliers": 0,
                               "reproj_rms": None, "flow_stage": "track_lost", "weak": False,
                               "accepted": False}
            return None
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
            self._flow_2d = None
            self._last_info = {"mode": "FLOW_TRACK", "next_mode": "FLOW_TRACK", "inliers": inl,
                               "reproj_rms": reproj, "flow_stage": "track_fail", "weak": False,
                               "accepted": False}
            return None
        # Apply the SAME confidence gate as the deep matcher: a low-confidence flow frame is
        # NOT published (caller hovers + refreshes next), so we never fly on a weak fix.
        ratio = inl / max(1, len(s3d))
        if (inl < self._flow_qual_inl or ratio < self._flow_qual_ratio
                or (reproj is not None and (reproj > self._flow_qual_reproj
                                            or reproj > self.cfg.max_reproj_error_track))
                or len(n3d) < self._flow_qual_ntrack):
            self._flow_2d = None                          # low confidence -> deep refresh next frame
            self._last_info = {"mode": "FLOW_TRACK", "next_mode": "FLOW_TRACK", "inliers": inl,
                               "reproj_rms": reproj, "corr3d": int(len(n3d)),
                               "flow_stage": "track_lowconf", "weak": True, "accepted": False}
            return None
        center, yaw = _pose_center_yaw_from_ret(ret)
        # max_jump gate (same intent as the deep _gate): a KLT/PnP flip can look inlier/reproj
        # consistent yet jump far. Tracked motion is always small, so a big jump from the last
        # published center is not published (hover + deep refresh next frame).
        if (self.state.last_center is not None
                and float(np.linalg.norm(center - self.state.last_center)) > self.cfg.max_jump):
            self._flow_2d = None
            self._last_info = {"mode": "FLOW_TRACK", "next_mode": "FLOW_TRACK", "inliers": inl,
                               "reproj_rms": reproj, "flow_stage": "jump_reject", "weak": True,
                               "accepted": False}
            return None
        pose = Pose(x=float(center[0]), y=float(center[1]), z=float(center[2]),
                    yaw=float(yaw), stamp=time.monotonic())
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
        return pose

    def _localize_frame_deep(self, frame: np.ndarray) -> Pose | None:
        """CUDA-OOM-safe wrapper: on GPU out-of-memory, free the cache and treat the
        frame as a missed fix (the flight loop then hovers) instead of letting the
        RuntimeError abort the whole mission."""
        try:
            return self._localize_frame_impl(frame)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._last_info = {"mode": self.state.mode, "next_mode": self.state.mode,
                               "error": "cuda_oom", "inliers": 0}
            print("[tracker] CUDA OOM in localize_frame; emptied cache, treating as missed fix",
                  flush=True)
            return None

    def _localize_frame_impl(self, frame: np.ndarray) -> Pose | None:
        xfeat = self.ensure_models()
        _ = xfeat
        start = time.perf_counter()
        mode = self.state.mode
        vpr_ms = 0.0
        if mode == "BOOT_INIT":
            candidates, vpr_ms = self._global_candidates(frame, self.cfg.boot_global_topk)
            xfeat_topk = self.cfg.xfeat_topk_acquire
            min_corr = self.cfg.acquire_min_inliers
        elif mode == "LOST":
            candidates, vpr_ms = self._global_candidates(frame, self.cfg.lost_global_topk)
            xfeat_topk = self.cfg.xfeat_topk_acquire
            min_corr = self.cfg.acquire_min_inliers
        else:
            candidates = self._track_candidates(weak=(mode == "WEAK_TRACK"))
            if not candidates:
                self.state.mode = "LOST"
                candidates, vpr_ms = self._global_candidates(frame, self.cfg.lost_global_topk)
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
        if mode in ("TRACK", "WEAK_TRACK") and self.cfg.matcher_mode == "nn_then_lg":
            temporal_summary = {}
            qc: dict = {}                # extract query XFeat once, reuse across NN/adaptive/full
            lgp: dict = {}               # cache LG pairs from adaptive pass -> reuse in full retry
            fast_ret, fast_info = self._localize_with_candidates(
                frame, candidates, xfeat_topk, min_corr, matcher_mode="nn",
                include_temporal_cache=self.cfg.temporal_cache_enabled and mode == "TRACK",
                q_cache=qc)
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
                ok and not weak and int(fast_info.get("inliers", 0)) >= self.cfg.adaptive_accept_inliers
                and (reproj is None or reproj <= self.cfg.adaptive_accept_reproj)
            )
            if fast_strong:
                info = fast_info
                info["composite_stage"] = "nn_fast_accept"
            else:
                nn_summary = {
                    "nn_fast_inliers": int(fast_info.get("inliers", 0)),
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
                    else:
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
                else:
                    ret, info = self._localize_with_candidates(
                        frame, candidates, xfeat_topk, min_corr, matcher_mode="lighterglue",
                        q_cache=qc)
                    info.update({"composite_stage": "lg_full_after_nn", **nn_summary})
                    ok, weak, pose = self._gate(ret, info, mode)
        elif mode in ("TRACK", "WEAK_TRACK") and self.cfg.adaptive_first_topk > 0 and len(candidates) > self.cfg.adaptive_first_topk:
            first_candidates = list(candidates)[:self.cfg.adaptive_first_topk]
            include_cache = self.cfg.temporal_cache_enabled and mode == "TRACK"
            ret, info = self._localize_with_candidates(
                frame, first_candidates, xfeat_topk, min_corr,
                include_temporal_cache=include_cache)
            info.update({
                "mode": mode,
                "vpr_ms": vpr_ms,
                "candidates": [int(i) for i in first_candidates],
                "adaptive_stage": "first",
                "adaptive_full_candidates": [int(i) for i in candidates],
            })
            ok, weak, pose = self._gate(ret, info, mode)
            reproj = info.get("reproj_rms")
            strong_enough = (
                ok and not weak and int(info.get("inliers", 0)) >= self.cfg.adaptive_accept_inliers
                and (reproj is None or reproj <= self.cfg.adaptive_accept_reproj)
            )
            if strong_enough:
                used_adaptive = True
            else:
                ret, info = self._localize_with_candidates(
                    frame, candidates, xfeat_topk, min_corr,
                    include_temporal_cache=include_cache)
                info["adaptive_stage"] = "full_retry"
                info["adaptive_first_accepted"] = bool(ok)
                ok, weak, pose = self._gate(ret, info, mode)
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
        cache_updated = False

        # TRACK failure escalates to WEAK_TRACK/LOST. BOOT/LOST stay global and wait
        # for external yaw scan / next frame.
        if not ok:
            self.state.fail_count += 1
            self.state.bad_count += 1
            if mode == "TRACK":
                self.state.mode = "WEAK_TRACK" if self.state.fail_count < self.cfg.lost_after else "LOST"
            elif mode == "WEAK_TRACK":
                self.state.mode = "LOST"
            elif mode in ("BOOT_INIT", "LOST"):
                self.state.mode = mode
            info["accepted"] = False
            if self.state.mode == "LOST":
                self.temporal_cache.clear()
            else:
                self._age_temporal_cache()
            pose = None
        else:
            self._publish_success(pose, info, weak)
            reproj = info.get("reproj_rms")
            seed_ok = (
                not weak
                and int(info.get("inliers", 0)) >= self.cfg.temporal_cache_seed_min_inliers
                and (reproj is None or reproj <= self.cfg.temporal_cache_seed_max_reproj)
            )
            if seed_ok:
                seed_refs = tuple(int(r) for r in (info.get("used_refs") or info.get("candidates", [])[:3]))
                if seed_refs and seed_refs == self._last_seed_refs and len(self.temporal_cache) > 0:
                    self.temporal_cache.age = 0          # same refs -> reuse cache, skip CPU gather + re-upload
                    cache_updated = True
                else:
                    ref_payload = self._cache_payload_from_refs(
                        seed_refs, str(info.get("composite_stage", "success_refs")))
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
        info["next_mode"] = self.state.mode
        info["bad_count"] = int(self.state.bad_count)
        info["fail_count"] = int(self.state.fail_count)
        info["total_ms"] = (time.perf_counter() - start) * 1000.0
        self._last_info = info
        return pose

    def get_pose(self) -> Pose | None:
        frame = self.frame_source()
        if frame is None:
            return None
        return self.localize_frame(frame)

    @property
    def last_info(self) -> dict:
        return self._last_info


if __name__ == "__main__":
    print(__doc__)
