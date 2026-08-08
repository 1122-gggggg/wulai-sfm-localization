#!/usr/bin/env python3
"""Production EDM tracker: the state machine of ProductionXFeatTracker, EDM local matching.

Same shape as the XFeat tracker (BOOT_INIT / TRACK / WEAK_TRACK / LOST, MegaLoc once at
BOOT and once per LOST episode, pose-prior + covisibility for tracking), because it keeps
the per-frame cost down -- and it matters MORE for EDM, not less:

  XFeat amortises the map side (descriptors are precomputed, so an extra candidate
  reference costs one cheap LighterGlue call). EDM cannot: it is detector-free, so every
candidate reference is a full network forward. The number of candidates per frame is
therefore the dominant cost term, so normal TRACK stays at one reference.

Measured on the RTX 5060 replay: TRACK topk=1 is about 25.6 FPS; topk=2 increases
inliers but roughly doubles median latency. LOW/WEAK temporarily uses topk=3 instead.

What is deliberately NOT ported from the XFeat tracker:
  - the temporal anchor cache. It caches descriptors of inlier 3D anchors to match the
    next frame cheaply; a detector-free matcher has no descriptors to cache. The natural
    EDM equivalent (match against the previous FRAME and carry its 3D) is left out until
    it is shown to be needed -- at 1-2 references, matching is no longer the bottleneck.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pycolmap

from edm_matcher import GRID_H, GRID_W, EDMMatcher
from reloc_localizer_edm import Camera, EDMLocalizer, EDMRelocMap


@dataclass
class EDMConfig:
    # acquisition (BOOT_INIT / LOST): MegaLoc retrieval
    boot_global_topk: int = 10
    acquire_initial_topk: int = 2
    acquire_min_inliers: int = 80
    # Run MegaLoc once at BOOT and once on entry to each LOST episode. LOW/WEAK
    # remains geometry-only and raises only the local EDM candidate count.
    global_retrieval_policy: str = "boot_and_lost_once"
    lost_local_topk: int = 5
    lost_local_grace_frames: int = 12
    recovery_bank_size: int = 192
    recovery_scan_topk: int = 2
    # Bound EDM's reference batch so boot_global_topk=10 does not allocate a
    # batch-of-10 activation peak.  All retrieved references are still evaluated.
    match_batch_size: int = 2
    use_temporal_reference: bool = False
    temporal_map_topk: int = 1
    # tracking: geometry-only candidates, no MegaLoc
    local_topk: int = 1
    weak_local_topk: int = 3
    near_pool: int = 24
    covis_per_ref: int = 20
    radius: float = 0.8
    max_yaw_diff_deg: float = 90.0
    track_min_inliers: int = 50
    weak_min_inliers: int = 30
    min_inlier_ratio: float = 0.15
    min_inlier_grid_cells: int = 6
    # gates
    max_reproj_error_acquire: float = 5.0
    max_reproj_error_track: float = 6.0
    pnp_ransac_max_error: float = 5.0
    max_jump: float = 2.0
    prediction_max_dt: float = 0.25
    adaptive_jump_factor: float = 8.0
    adaptive_jump_floor: float = 0.003
    adaptive_jump_bootstrap: float = 0.02
    adaptive_jump_ceiling: float = 0.008
    adaptive_jump_min_history: int = 20
    adaptive_jump_history_size: int = 120
    weak_after: int = 2
    lost_after: int = 2
    # LOST re-acquisition bound. The continuous-trajectory gate cannot apply across a
    # LOST episode, but accepting an unbounded teleport is exactly how a wrong-place
    # retrieval reaches the controller looking like a normal TRACK fix. Bound the
    # re-acquisition against the last accepted pose while that pose is still recent;
    # once it expires, a pure global relocalization anywhere is allowed again.
    # Expressed as a FACTOR of max_jump so it follows each site's scale-dependent
    # calibration instead of needing its own per-site value.
    acquire_max_jump_factor: float = 2.0
    acquire_max_yaw_diff_deg: float = 90.0
    lost_prior_max_age_s: float = 3.0
    # EDM emits thousands of correspondences; PnP cost is linear in them and RANSAC
    # gains nothing past a well-spread ~900. Cap spatially so the cap does not bias
    # the pose toward whichever image region happened to match densely.
    max_corr_total: int = 900
    corr_grid: int = 8

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.global_retrieval_policy not in {"boot_once", "boot_and_lost_once"}:
            raise ValueError(
                "global_retrieval_policy must be 'boot_once' or 'boot_and_lost_once'"
            )
        positive_ints = (
            "boot_global_topk",
            "acquire_initial_topk",
            "match_batch_size",
            "temporal_map_topk",
            "local_topk",
            "weak_local_topk",
            "lost_local_topk",
            "near_pool",
            "covis_per_ref",
            "acquire_min_inliers",
            "track_min_inliers",
            "weak_min_inliers",
            "adaptive_jump_min_history",
            "adaptive_jump_history_size",
            "weak_after",
            "lost_after",
            "corr_grid",
            "min_inlier_grid_cells",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        nonnegative_ints = (
            "lost_local_grace_frames",
            "recovery_bank_size",
            "recovery_scan_topk",
            "max_corr_total",
        )
        for name in nonnegative_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.use_temporal_reference, bool):
            raise ValueError("use_temporal_reference must be boolean")
        positive_floats = (
            "radius",
            "max_yaw_diff_deg",
            "max_reproj_error_acquire",
            "max_reproj_error_track",
            "pnp_ransac_max_error",
            "max_jump",
            "prediction_max_dt",
            "adaptive_jump_factor",
            "adaptive_jump_floor",
            "adaptive_jump_bootstrap",
            "adaptive_jump_ceiling",
            "acquire_max_jump_factor",
            "acquire_max_yaw_diff_deg",
            "lost_prior_max_age_s",
        )
        for name in positive_floats:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.acquire_initial_topk > self.boot_global_topk:
            raise ValueError("acquire_initial_topk cannot exceed boot_global_topk")
        if (isinstance(self.min_inlier_ratio, bool)
                or not isinstance(self.min_inlier_ratio, (int, float))
                or not math.isfinite(float(self.min_inlier_ratio))
                or not 0.0 < float(self.min_inlier_ratio) <= 1.0):
            raise ValueError("min_inlier_ratio must be finite and within (0, 1]")
        if self.min_inlier_grid_cells > self.corr_grid * self.corr_grid:
            raise ValueError("min_inlier_grid_cells cannot exceed corr_grid squared")
        if self.local_topk > self.weak_local_topk or self.local_topk > self.lost_local_topk:
            raise ValueError("weak/lost local top-k cannot be smaller than TRACK local_topk")
        if self.adaptive_jump_min_history > self.adaptive_jump_history_size:
            raise ValueError(
                "adaptive_jump_min_history cannot exceed adaptive_jump_history_size"
            )
        if self.adaptive_jump_floor > self.adaptive_jump_ceiling:
            raise ValueError("adaptive_jump_floor cannot exceed adaptive_jump_ceiling")
        if self.adaptive_jump_ceiling > self.adaptive_jump_bootstrap:
            raise ValueError(
                "adaptive_jump_bootstrap cannot be smaller than adaptive_jump_ceiling"
            )
        for name in (
            "adaptive_jump_floor",
            "adaptive_jump_bootstrap",
            "adaptive_jump_ceiling",
        ):
            if float(getattr(self, name)) > float(self.max_jump):
                raise ValueError(f"{name} cannot exceed max_jump")


@dataclass
class RuntimeState:
    state: str = "BOOT_INIT"
    center: np.ndarray | None = None
    yaw: float | None = None
    velocity: np.ndarray | None = None
    last_capture_stamp: float | None = None
    last_refs: list = field(default_factory=list)
    misses: int = 0
    frame: int = 0
    accepted_step_norms: deque[float] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    observed_capture_dts: deque[float] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    last_observed_capture_stamp: float | None = None
    pending_limited_center: np.ndarray | None = None
    pending_limited_stamp: float | None = None
    pending_limited_limit: float | None = None
    boot_refs: list[str] = field(default_factory=list)
    global_retrieval_calls: int = 0
    lost_global_retrieval_done: bool = False
    lost_frames: int = 0
    recovery_cursor: int = 0


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def build_recovery_bank(ref_names: list[str], max_refs: int) -> list[str]:
    """Create a bounded, sequence-aware reference bank for EDM-only global recovery.

    Slots are allocated roughly in proportion to each sequence size, sampled uniformly
    within the sequence, then interleaved so a cyclic scan reaches every route early.
    """
    if max_refs <= 0 or not ref_names:
        return []
    if len(ref_names) <= max_refs:
        return list(ref_names)
    groups: dict[str, list[str]] = {}
    for name in ref_names:
        groups.setdefault(name.split("/", 1)[0], []).append(name)
    total = len(ref_names)
    keys = list(groups)
    raw = {key: max_refs * len(groups[key]) / total for key in keys}
    allocation = {key: min(len(groups[key]), max(1, int(math.floor(raw[key])))) for key in keys}
    while sum(allocation.values()) > max_refs:
        candidates = [key for key in keys if allocation[key] > 1]
        if not candidates:
            break
        key = max(candidates, key=lambda item: allocation[item] - raw[item])
        allocation[key] -= 1
    while sum(allocation.values()) < max_refs:
        candidates = [key for key in keys if allocation[key] < len(groups[key])]
        if not candidates:
            break
        key = max(candidates, key=lambda item: raw[item] - allocation[item])
        allocation[key] += 1
    sampled: dict[str, list[str]] = {}
    for key in keys:
        group = groups[key]
        count = allocation[key]
        indices = np.linspace(0, len(group) - 1, num=count, dtype=int)
        sampled[key] = [group[int(idx)] for idx in indices]
    bank: list[str] = []
    for offset in range(max(len(values) for values in sampled.values())):
        for key in keys:
            if offset < len(sampled[key]):
                bank.append(sampled[key][offset])
    return bank[:max_refs]


def build_temporal_lut(
    points2d_camera: np.ndarray,
    points3d: np.ndarray,
    inlier_mask: np.ndarray,
    camera_to_edm_scale: float,
) -> np.ndarray:
    """Bind successful query-frame PnP inliers to that frame's EDM coarse cells."""
    lut = np.full((GRID_W * GRID_H, 3), np.nan, dtype=np.float32)
    mask = np.asarray(inlier_mask, dtype=bool)
    if not mask.any():
        return lut
    points_edm = np.asarray(points2d_camera, dtype=np.float32)[mask] / camera_to_edm_scale
    cells = EDMMatcher.cell_ids(points_edm)
    lut[cells] = np.asarray(points3d, dtype=np.float32)[mask]
    return lut


def adaptive_jump_limit(
    step_history: Sequence[float],
    cfg: EDMConfig,
    *,
    capture_dt: float | None = None,
    capture_dt_history: Sequence[float] = (),
) -> float:
    """Return a trajectory envelope adjusted for skipped-frame capture time."""
    if len(step_history) < cfg.adaptive_jump_min_history:
        base_limit = min(
            cfg.max_jump,
            cfg.adaptive_jump_bootstrap,
        )
    else:
        typical = float(np.median(np.asarray(step_history, dtype=float)))
        base_limit = min(
            cfg.max_jump,
            cfg.adaptive_jump_ceiling,
            max(cfg.adaptive_jump_floor, cfg.adaptive_jump_factor * typical),
        )

    valid_dts = np.asarray(
        [
            value
            for value in capture_dt_history
            if math.isfinite(float(value)) and float(value) > 1e-6
        ],
        dtype=float,
    )
    if (
        capture_dt is None
        or not math.isfinite(float(capture_dt))
        or float(capture_dt) <= 1e-6
        or len(valid_dts) < cfg.adaptive_jump_min_history
    ):
        return base_limit

    typical_dt = float(np.median(valid_dts))
    effective_dt = min(float(capture_dt), float(cfg.prediction_max_dt))
    time_scale = max(1.0, effective_dt / typical_dt)
    return min(float(cfg.max_jump), base_limit * time_scale)


def limit_center_step(
    previous: np.ndarray,
    candidate: np.ndarray,
    step_history: Sequence[float],
    cfg: EDMConfig,
    *,
    capture_dt: float | None = None,
    capture_dt_history: Sequence[float] = (),
) -> tuple[np.ndarray, dict]:
    """Limit an isolated PnP center spike without turning it into a teleport."""
    previous = np.asarray(previous, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    delta = candidate - previous
    raw_step = float(np.linalg.norm(delta))
    limit = adaptive_jump_limit(
        step_history,
        cfg,
        capture_dt=capture_dt,
        capture_dt_history=capture_dt_history,
    )
    if raw_step <= limit or raw_step == 0.0:
        return candidate, {"limited": False, "raw_step": raw_step, "limit": limit}
    center = previous + delta * (limit / raw_step)
    return center, {"limited": True, "raw_step": raw_step, "limit": limit}


def spatially_cap_indices(pts2d: np.ndarray, confidence: np.ndarray, max_total: int,
                          width: int, height: int, grid: int = 8) -> np.ndarray:
    """Quality-first round-robin selection across image grid cells."""
    n = len(pts2d)
    if max_total <= 0 or n <= max_total:
        return np.arange(n, dtype=np.int64)
    score = np.asarray(confidence, dtype=float).reshape(-1)
    if len(score) != n:
        raise ValueError("confidence length must match pts2d")
    score = np.where(np.isfinite(score), score, -np.inf)
    gx = np.clip((pts2d[:, 0] / width * grid).astype(int), 0, grid - 1)
    gy = np.clip((pts2d[:, 1] / height * grid).astype(int), 0, grid - 1)
    cell = gy * grid + gx
    original = np.arange(n, dtype=np.int64)
    within = np.lexsort((original, -score, cell))
    sorted_cells = cell[within]
    starts = np.r_[0, np.flatnonzero(sorted_cells[1:] != sorted_cells[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    ranks = np.arange(n) - np.repeat(starts, lengths)
    unique_cells = sorted_cells[starts]
    top_scores = score[within[starts]]
    cell_order = np.lexsort((unique_cells, -top_scores))
    priorities = np.empty(len(unique_cells), dtype=np.int64)
    priorities[cell_order] = np.arange(len(unique_cells), dtype=np.int64)
    row_priorities = priorities[np.repeat(np.arange(len(unique_cells)), lengths)]
    selected = np.lexsort((row_priorities, ranks))[:max_total]
    return within[selected]


def spatially_cap(pts2d: np.ndarray, pts3d: np.ndarray, max_total: int,
                  width: int, height: int, grid: int = 8,
                  confidence: np.ndarray | None = None):
    """Keep quality-ranked correspondences spatially spread over the image."""
    scores = np.ones(len(pts2d), dtype=np.float32) if confidence is None else confidence
    selected = spatially_cap_indices(pts2d, scores, max_total, width, height, grid)
    return pts2d[selected], pts3d[selected]


def reprojection_metrics(ret, pts2d: np.ndarray, pts3d: np.ndarray,
                         camera: pycolmap.Camera, grid: int = 8) -> dict:
    """Compute generic pycolmap-camera RMS and inlier image coverage."""
    count = len(pts2d)
    mask = None
    if ret is not None:
        for key in ("inlier_mask", "inliers"):
            value = ret.get(key)
            if value is not None:
                candidate = np.asarray(value, dtype=bool).reshape(-1)
                if len(candidate) == count:
                    mask = candidate
                    break
    if mask is None or not mask.any():
        return {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}
    observed = np.asarray(pts2d, dtype=float)[mask]
    world = np.asarray(pts3d, dtype=float)[mask]
    camera_points = np.asarray(ret["cam_from_world"] * world, dtype=float)
    valid = np.isfinite(camera_points).all(1) & (camera_points[:, 2] > 1e-8)
    if not valid.any():
        return {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}
    observed = observed[valid]
    projected = np.asarray(camera.img_from_cam(camera_points[valid]), dtype=float)
    valid_projection = np.isfinite(projected).all(1)
    if not valid_projection.any():
        return {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}
    observed = observed[valid_projection]
    projected = projected[valid_projection]
    error = projected - observed
    gx = np.clip((observed[:, 0] / camera.width * grid).astype(int), 0, grid - 1)
    gy = np.clip((observed[:, 1] / camera.height * grid).astype(int), 0, grid - 1)
    return {
        "reproj_rms": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
        "inlier_ratio": float(len(observed) / max(count, 1)),
        "inlier_grid_cells": int(len(np.unique(gy * grid + gx))),
    }


def reprojection_rank(value) -> float:
    """Rank a valid RMS above a missing value, including the perfect 0.0 case."""
    return -math.inf if value is None else -float(value)


class ProductionEDMTracker:
    def __init__(self, reloc_map: EDMRelocMap, camera: Camera, cfg: EDMConfig | None = None,
                 matcher: EDMMatcher | None = None, megaloc=None):
        self.cfg = cfg or EDMConfig()
        self.cfg.validate()
        self.map = reloc_map
        self.cam = camera
        self.loc = EDMLocalizer(reloc_map, camera, matcher=matcher, megaloc=megaloc,
                                pnp_max_error=self.cfg.pnp_ransac_max_error)
        self.st = RuntimeState(
            accepted_step_norms=deque(maxlen=self.cfg.adaptive_jump_history_size),
            observed_capture_dts=deque(maxlen=self.cfg.adaptive_jump_history_size),
        )
        self.centers = None if reloc_map.ref_centers is None else np.asarray(reloc_map.ref_centers, np.float32)
        self.yaws = None if reloc_map.ref_yaws is None else np.asarray(reloc_map.ref_yaws, np.float32)
        self.name_of = {i: n for i, n in enumerate(reloc_map.ref_names)}
        self.idx_of = {n: i for i, n in enumerate(reloc_map.ref_names)}
        self.recovery_bank = build_recovery_bank(
            list(reloc_map.ref_names), self.cfg.recovery_bank_size
        )
        self.temporal_gray: np.ndarray | None = None
        self.temporal_xyz_by_cell: np.ndarray | None = None
        self._pcam: pycolmap.Camera | None = None
        self._pnp_options: pycolmap.AbsolutePoseEstimationOptions | None = None

    def _pose_estimation_context(
        self,
    ) -> tuple[pycolmap.Camera, pycolmap.AbsolutePoseEstimationOptions]:
        if getattr(self, "_pcam", None) is None:
            self._pcam = pycolmap.Camera(
                model=self.cam.model,
                width=self.cam.width,
                height=self.cam.height,
                params=self.cam.params,
            )
        if getattr(self, "_pnp_options", None) is None:
            self._pnp_options = pycolmap.AbsolutePoseEstimationOptions()
            self._pnp_options.ransac.max_error = self.cfg.pnp_ransac_max_error
        return self._pcam, self._pnp_options

    # ---------- candidate selection ----------
    def _predict_center(self, capture_stamp: float | None = None):
        if self.st.center is None:
            return None
        if (self.st.velocity is None or self.st.last_capture_stamp is None
                or capture_stamp is None):
            return self.st.center
        dt = float(capture_stamp) - float(self.st.last_capture_stamp)
        if not math.isfinite(dt) or dt <= 0.0:
            return self.st.center
        dt = min(dt, max(0.0, float(self.cfg.prediction_max_dt)))
        return self.st.center + self.st.velocity * dt

    def _track_candidates(self, topk: int,
                          capture_stamp: float | None = None) -> list[str]:
        """Geometry only: nearest references to the predicted pose, plus their covisibles."""
        C = self._predict_center(capture_stamp)
        if C is None or self.centers is None:
            return []
        d = np.linalg.norm(self.centers - C[None, :], axis=1)
        finite = np.isfinite(d)
        max_yaw = math.radians(self.cfg.max_yaw_diff_deg)

        near = []
        for i in np.argsort(np.where(finite, d, np.inf)):
            i = int(i)
            if not finite[i]:
                continue
            if d[i] > self.cfg.radius and len(near) >= self.cfg.near_pool:
                break
            if self.st.yaw is not None and self.yaws is not None and np.isfinite(self.yaws[i]):
                if abs(_angle_diff(float(self.yaws[i]), self.st.yaw)) > max_yaw:
                    continue
            near.append(i)
            if len(near) >= self.cfg.near_pool:
                break

        covis_refs: list[int] = []
        if self.map.covis:
            for idx in list(self.st.last_refs)[:2] + near[:2]:
                covis_refs.extend(int(j) for j in self.map.covis.get(self.name_of[idx], [])[:self.cfg.covis_per_ref])

        union = list(dict.fromkeys(near + covis_refs + list(self.st.last_refs)))
        last = set(self.st.last_refs)

        def score(i: int) -> float:
            s = 2.0 * math.exp(-float(d[i]) / max(self.cfg.radius, 1e-6))
            if self.st.yaw is not None and self.yaws is not None and np.isfinite(self.yaws[i]):
                s += 0.8 * math.exp(-abs(_angle_diff(float(self.yaws[i]), self.st.yaw)) / math.radians(45.0))
            else:
                s += 0.4
            return s + (1.0 if i in last else 0.0)

        union = [i for i in union if 0 <= i < len(self.map.ref_names) and finite[i]]
        union.sort(key=score, reverse=True)
        return [self.name_of[i] for i in union[:topk]]

    def _recovery_candidates(self, topk: int) -> list[str]:
        if topk <= 0 or not self.recovery_bank:
            return []
        start = self.st.recovery_cursor % len(self.recovery_bank)
        refs = [
            self.recovery_bank[(start + offset) % len(self.recovery_bank)]
            for offset in range(min(topk, len(self.recovery_bank)))
        ]
        self.st.recovery_cursor = (start + len(refs)) % len(self.recovery_bank)
        return refs

    # ---------- one frame ----------
    def localize(self, frame_bgr: np.ndarray,
                 capture_stamp: float | None = None) -> dict:
        import cv2
        cfg = self.st_cfg = self.cfg
        t0 = time.perf_counter()
        if capture_stamp is None:
            capture_stamp = time.monotonic()
        capture_stamp = float(capture_stamp)
        if not math.isfinite(capture_stamp):
            raise ValueError("capture_stamp must be finite")
        previous_observed_stamp = self.st.last_observed_capture_stamp
        if previous_observed_stamp is None or capture_stamp > previous_observed_stamp:
            if previous_observed_stamp is not None:
                self.st.observed_capture_dts.append(
                    capture_stamp - previous_observed_stamp
                )
            self.st.last_observed_capture_stamp = capture_stamp
        self.st.frame += 1
        gray = EDMMatcher.load_gray(frame_bgr)

        acquiring = self.st.state in ("BOOT_INIT", "LOST")
        t_vpr = 0.0
        candidate_mode = "edm_track"
        if acquiring:
            if cfg.global_retrieval_policy == "boot_and_lost_once":
                run_global = (
                    (self.st.state == "BOOT_INIT" and self.st.global_retrieval_calls == 0)
                    or (self.st.state == "LOST" and not self.st.lost_global_retrieval_done)
                )
            elif cfg.global_retrieval_policy == "boot_once":
                run_global = self.st.global_retrieval_calls == 0
            else:
                run_global = True
            if not run_global:
                if (
                    self.st.state == "LOST"
                    and self.st.lost_frames > cfg.lost_local_grace_frames
                ):
                    refs = self._recovery_candidates(cfg.recovery_scan_topk)
                    candidate_mode = "edm_map_scan"
                elif self.st.state == "LOST":
                    refs = self._track_candidates(cfg.lost_local_topk, capture_stamp)
                    candidate_mode = "edm_local_recovery"
                else:
                    refs = list(self.st.boot_refs)
                    candidate_mode = "edm_boot_refs"
                if not refs:
                    refs = list(self.st.boot_refs[: cfg.lost_local_topk])
                    candidate_mode = "edm_boot_refs"
            else:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tv = time.perf_counter()
                refs = self.loc.retrieve(rgb, cfg.boot_global_topk)
                t_vpr = (time.perf_counter() - tv) * 1e3
                self.st.global_retrieval_calls += 1
                if self.st.state == "LOST":
                    self.st.lost_global_retrieval_done = True
                if not self.st.boot_refs:
                    self.st.boot_refs = list(refs)
                candidate_mode = (
                    "megaloc_lost" if self.st.state == "LOST" else "megaloc_boot"
                )
            min_inl = cfg.acquire_min_inliers
        else:
            temporal_ready = (
                cfg.use_temporal_reference
                and self.temporal_gray is not None
                and self.temporal_xyz_by_cell is not None
            )
            topk = (
                cfg.temporal_map_topk
                if self.st.state == "TRACK" and temporal_ready
                else cfg.local_topk if self.st.state == "TRACK" else cfg.weak_local_topk
            )
            refs = self._track_candidates(topk, capture_stamp)
            candidate_mode = "edm_temporal_map" if temporal_ready else "edm_track"
            if not refs:                       # prior unusable -> acquisition candidate set
                if (
                    cfg.global_retrieval_policy in {"boot_once", "boot_and_lost_once"}
                    and self.st.global_retrieval_calls > 0
                ):
                    refs = list(self.st.boot_refs[: cfg.lost_local_topk])
                    candidate_mode = "edm_boot_refs"
                else:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    tv = time.perf_counter()
                    refs = self.loc.retrieve(rgb, cfg.boot_global_topk)
                    t_vpr = (time.perf_counter() - tv) * 1e3
                    self.st.global_retrieval_calls += 1
                    if not self.st.boot_refs:
                        self.st.boot_refs = list(refs)
                    candidate_mode = "megaloc_reacquire"
            min_inl = (cfg.track_min_inliers if self.st.state == "TRACK" else cfg.weak_min_inliers)

        requested_refs = list(refs)
        tm = time.perf_counter()
        by_ref = None
        ret = None
        selected_ref = None
        staged_early_stop = False
        pose_points2d = np.zeros((0, 2))
        pose_points3d = np.zeros((0, 3))
        pose_metrics = {
            "reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0,
        }
        pcam, opts = self._pose_estimation_context()
        t_pnp = 0.0

        def estimate_pose(points2d: np.ndarray, points3d: np.ndarray,
                          confidence: np.ndarray):
            nonlocal t_pnp
            if len(points3d) < 6:
                return None
            selected = spatially_cap_indices(
                points2d, confidence, cfg.max_corr_total, self.cam.width,
                self.cam.height, cfg.corr_grid
            )
            p2c, p3c = points2d[selected], points3d[selected]
            started = time.perf_counter()
            estimate = pycolmap.estimate_and_refine_absolute_pose(
                np.asarray(p2c, float), np.asarray(p3c, float), pcam, opts
            )
            t_pnp += (time.perf_counter() - started) * 1e3
            if estimate is None:
                return None
            metrics = reprojection_metrics(estimate, p2c, p3c, pcam, cfg.corr_grid)
            return estimate, p2c, p3c, metrics

        reproj_limit = (
            cfg.max_reproj_error_acquire if acquiring else cfg.max_reproj_error_track
        )

        temporal_used = (
            candidate_mode in ("edm_temporal_map", "edm_local_recovery")
            and cfg.use_temporal_reference
            and self.temporal_gray is not None
            and self.temporal_xyz_by_cell is not None
        )
        if temporal_used:
            source_images = [self.temporal_gray] + [self.map.images[name] for name in refs]
            source_luts = [self.temporal_xyz_by_cell] + [self.map.xyz_by_cell[name] for name in refs]
            by_source = self.loc.correspondences_for_sources(
                gray, source_images, source_luts, batch_size=cfg.match_batch_size
            )
            p2_parts = [row[0] for row in by_source if len(row[0])]
            p3_parts = [row[1] for row in by_source if len(row[1])]
            confidence_parts = [row[2] for row in by_source if len(row[2])]
            p2 = np.concatenate(p2_parts) if p2_parts else np.zeros((0, 2))
            p3 = np.concatenate(p3_parts) if p3_parts else np.zeros((0, 3))
            confidence = (
                np.concatenate(confidence_parts) if confidence_parts else np.zeros(0)
            )
            per_ref = [row[3] for row in by_source]
        elif candidate_mode == "edm_map_scan":
            by_ref = self.loc.correspondences_by_ref(
                gray, refs, batch_size=cfg.match_batch_size
            )
            p2_parts = [row[0] for row in by_ref if len(row[0])]
            p3_parts = [row[1] for row in by_ref if len(row[1])]
            confidence_parts = [row[2] for row in by_ref if len(row[2])]
            p2 = np.concatenate(p2_parts) if p2_parts else np.zeros((0, 2))
            p3 = np.concatenate(p3_parts) if p3_parts else np.zeros((0, 3))
            confidence = (
                np.concatenate(confidence_parts) if confidence_parts else np.zeros(0)
            )
            per_ref = [row[3] for row in by_ref]
        else:
            initial_topk = min(max(0, int(cfg.acquire_initial_topk)), len(refs))
            staged = acquiring and 0 < initial_topk < len(refs)
            if staged:
                first_refs = refs[:initial_topk]
                p2, p3, confidence, per_ref = self.loc.correspondences(
                    gray, first_refs, batch_size=cfg.match_batch_size
                )
                # Acquisition must consider the complete retrieved set.  A plausible
                # pose from the first two visually similar references is not evidence
                # of multi-reference agreement and used to bypass the remaining eight.
                remaining_refs = refs[initial_topk:]
                p2_rest, p3_rest, confidence_rest, counts_rest = self.loc.correspondences(
                    gray, remaining_refs, batch_size=cfg.match_batch_size
                )
                if len(p2_rest):
                    p2 = np.concatenate([p2, p2_rest])
                    p3 = np.concatenate([p3, p3_rest])
                    confidence = np.concatenate([confidence, confidence_rest])
                per_ref.extend(counts_rest)
            else:
                p2, p3, confidence, per_ref = self.loc.correspondences(
                    gray, refs, batch_size=cfg.match_batch_size
                )
        t_match = max(0.0, (time.perf_counter() - tm) * 1e3 - t_pnp)

        if by_ref is not None:
            for name, (ref_p2, ref_p3, ref_confidence, _) in zip(refs, by_ref):
                candidate_pose = estimate_pose(ref_p2, ref_p3, ref_confidence)
                if candidate_pose is not None and (
                    ret is None
                    or (
                        int(candidate_pose[0]["num_inliers"]),
                        reprojection_rank(candidate_pose[3]["reproj_rms"]),
                    ) > (
                        int(ret["num_inliers"]),
                        reprojection_rank(pose_metrics["reproj_rms"]),
                    )
                ):
                    ret, pose_points2d, pose_points3d, pose_metrics = candidate_pose
                    selected_ref = name
        elif ret is None:
            candidate_pose = estimate_pose(p2, p3, confidence)
            if candidate_pose is not None:
                ret, pose_points2d, pose_points3d, pose_metrics = candidate_pose

        info = {
            "frame": self.st.frame, "state_in": self.st.state, "refs": refs,
            "candidate_mode": candidate_mode,
            "temporal_used": temporal_used,
            "selected_ref": selected_ref,
            "requested_reference_count": len(requested_refs),
            "staged_early_stop": staged_early_stop,
            "n_corr": int(len(p3)), "per_ref": per_ref,
            "global_retrieval_calls": self.st.global_retrieval_calls,
            "lost_global_retrieval_done": self.st.lost_global_retrieval_done,
            "vpr_ms": t_vpr, "match_ms": t_match, "pnp_ms": t_pnp,
            "total_ms": (time.perf_counter() - t0) * 1e3,
        }
        info.update(pose_metrics)

        if ret is None or int(ret["num_inliers"]) < min_inl:
            self.st.pending_limited_center = None
            self.st.pending_limited_stamp = None
            self.st.pending_limited_limit = None
            info["inliers"] = 0 if ret is None else int(ret["num_inliers"])
            self._on_miss(info)
            return info

        if (info["reproj_rms"] is None
                or float(info["reproj_rms"]) > float(reproj_limit)):
            self.st.pending_limited_center = None
            self.st.pending_limited_stamp = None
            self.st.pending_limited_limit = None
            info["inliers"] = int(ret["num_inliers"])
            info["rejected"] = (
                "reprojection_unavailable" if info["reproj_rms"] is None
                else "reprojection"
            )
            info["max_reproj_error"] = float(reproj_limit)
            self._on_miss(info)
            return info

        if float(info["inlier_ratio"]) < float(cfg.min_inlier_ratio):
            self.st.pending_limited_center = None
            self.st.pending_limited_stamp = None
            self.st.pending_limited_limit = None
            info["inliers"] = int(ret["num_inliers"])
            info["rejected"] = "inlier_ratio"
            info["min_inlier_ratio"] = float(cfg.min_inlier_ratio)
            self._on_miss(info)
            return info

        if int(info["inlier_grid_cells"]) < cfg.min_inlier_grid_cells:
            self.st.pending_limited_center = None
            self.st.pending_limited_stamp = None
            self.st.pending_limited_limit = None
            info["inliers"] = int(ret["num_inliers"])
            info["rejected"] = "inlier_spread"
            info["min_inlier_grid_cells"] = cfg.min_inlier_grid_cells
            self._on_miss(info)
            return info

        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        C = -R.T @ np.asarray(T.translation)
        fwd = R.T @ np.array([0, 0, 1.0])
        yaw = float(math.atan2(fwd[1], fwd[0]))
        info["inliers"] = int(ret["num_inliers"])

        # Two-layer trajectory gate. Extreme centers are rejected. A smaller jump
        # outside the adaptive envelope must repeat before it is accepted; clipping
        # every isolated PnP spike would turn bad poses into a slowly drifting TRACK.
        if self.st.center is not None and info["state_in"] != "LOST":
            capture_dt = (
                None if self.st.last_capture_stamp is None
                else capture_stamp - self.st.last_capture_stamp
            )
            raw_step = float(np.linalg.norm(C - self.st.center))
            if raw_step > cfg.max_jump:
                self.st.pending_limited_center = None
                self.st.pending_limited_stamp = None
                self.st.pending_limited_limit = None
                info["limited_jump"] = {
                    "raw_step": raw_step,
                    "limit": float(cfg.max_jump),
                    "capture_dt": capture_dt,
                    "confirmation_model": None,
                    "confirmation_residual": None,
                    "confirmation_limit": None,
                }
                info["rejected"] = "jump"
                self._on_miss(info)
                return info
            _limited_center, jump_info = limit_center_step(
                self.st.center,
                C,
                self.st.accepted_step_norms,
                cfg,
                capture_dt=capture_dt,
                capture_dt_history=self.st.observed_capture_dts,
            )
            if jump_info["limited"]:
                info["limited_jump"] = {
                    "raw_step": jump_info["raw_step"],
                    "limit": jump_info["limit"],
                    "capture_dt": capture_dt,
                    "confirmation_model": None,
                    "confirmation_residual": None,
                    "confirmation_limit": None,
                }
                pending = self.st.pending_limited_center
                pending_stamp = self.st.pending_limited_stamp
                confirmation_limit = max(
                    float(jump_info["limit"]),
                    float(self.st.pending_limited_limit or 0.0),
                )
                confirmation_model = None
                confirmation_residual = None
                # The confirmation must come from a DIFFERENT capture. Re-localizing the
                # same frame reproduces the same centre, which would score a residual of
                # ~0 against itself and promote one bad frame to an accepted relocation.
                independent_capture = (
                    pending_stamp is not None and capture_stamp > pending_stamp
                )
                info["limited_jump"]["confirmation_independent"] = independent_capture
                if pending is not None and independent_capture:
                    stationary_residual = float(np.linalg.norm(C - pending))
                    confirmation_model = "stationary"
                    confirmation_residual = stationary_residual
                    if (
                        self.st.last_capture_stamp is not None
                        and pending_stamp is not None
                    ):
                        first_dt = pending_stamp - self.st.last_capture_stamp
                        next_dt = capture_stamp - pending_stamp
                        if (
                            math.isfinite(first_dt)
                            and math.isfinite(next_dt)
                            and first_dt > 1e-6
                            and next_dt >= 0.0
                        ):
                            velocity = (pending - self.st.center) / first_dt
                            predicted = pending + velocity * next_dt
                            motion_residual = float(np.linalg.norm(C - predicted))
                            if motion_residual < stationary_residual:
                                confirmation_model = "constant_velocity"
                                confirmation_residual = motion_residual
                info["limited_jump"].update({
                    "confirmation_model": confirmation_model,
                    "confirmation_residual": confirmation_residual,
                    "confirmation_limit": confirmation_limit,
                })
                agrees = (
                    confirmation_residual is not None
                    and confirmation_residual <= confirmation_limit
                )
                if not agrees:
                    self.st.pending_limited_center = np.asarray(C, dtype=float).copy()
                    self.st.pending_limited_stamp = capture_stamp
                    self.st.pending_limited_limit = float(jump_info["limit"])
                    info["rejected"] = "limited_jump_unconfirmed"
                    self._on_miss(info)
                    return info
                info["limited_jump_confirmed"] = True
                self.st.pending_limited_center = None
                self.st.pending_limited_stamp = None
                self.st.pending_limited_limit = None
            else:
                self.st.pending_limited_center = None
                self.st.pending_limited_stamp = None
                self.st.pending_limited_limit = None
        elif self.st.center is not None:
            # LOST re-acquisition: bound it against the last accepted pose while that
            # pose is still recent. Without this, any relocalization -- including one
            # onto a repeated-texture reference on the far side of the map -- is
            # accepted with no bound on distance or heading change and published as a
            # normal TRACK fix.
            prior_age = (
                None if self.st.last_capture_stamp is None
                else capture_stamp - self.st.last_capture_stamp
            )
            prior_fresh = (
                prior_age is not None
                and math.isfinite(prior_age)
                and 0.0 <= prior_age <= cfg.lost_prior_max_age_s
            )
            info["acquire_prior_age"] = prior_age
            if prior_fresh:
                acquire_limit = float(cfg.acquire_max_jump_factor) * float(cfg.max_jump)
                acquire_step = float(np.linalg.norm(C - self.st.center))
                info["acquire_step"] = acquire_step
                info["acquire_limit"] = acquire_limit
                if acquire_step > acquire_limit:
                    info["rejected"] = "acquire_jump"
                    self._on_miss(info)
                    return info
                prior_yaw = self.st.yaw
                if prior_yaw is not None and math.isfinite(float(prior_yaw)):
                    yaw_delta = abs(_angle_diff(yaw, float(prior_yaw)))
                    info["acquire_yaw_delta_deg"] = math.degrees(yaw_delta)
                    if yaw_delta > math.radians(float(cfg.acquire_max_yaw_diff_deg)):
                        info["rejected"] = "acquire_yaw"
                        self._on_miss(info)
                        return info

        previous_center = self.st.center
        previous_stamp = self.st.last_capture_stamp
        step = None if previous_center is None else (C - previous_center)
        measured_velocity = None
        if step is not None:
            dt = None if previous_stamp is None else capture_stamp - previous_stamp
            if (info["state_in"] != "LOST" and dt is not None
                    and math.isfinite(dt) and dt > 1e-6):
                self.st.accepted_step_norms.append(float(np.linalg.norm(step)))
                measured_velocity = step / dt
        self.st.velocity = (
            measured_velocity
            if self.st.velocity is None or measured_velocity is None
            else 0.5 * (self.st.velocity + measured_velocity)
        )
        self.st.center, self.st.yaw = C, yaw
        self.st.last_capture_stamp = capture_stamp
        accepted_refs = [selected_ref] if selected_ref is not None else refs
        self.st.last_refs = [self.idx_of[n] for n in accepted_refs]
        self.st.misses = 0
        self.st.lost_frames = 0
        self.st.lost_global_retrieval_done = False
        self.st.state = "TRACK"
        if cfg.use_temporal_reference:
            inlier_mask = np.asarray(
                ret.get("inlier_mask", np.ones(len(pose_points3d), dtype=bool)),
                dtype=bool,
            )
            self.temporal_xyz_by_cell = build_temporal_lut(
                pose_points2d, pose_points3d, inlier_mask,
                camera_to_edm_scale=self.loc.scale,
            )
            self.temporal_gray = gray.copy()
        else:
            self.temporal_xyz_by_cell = None
            self.temporal_gray = None
        info.update({"state_out": "TRACK", "center": C, "yaw": yaw, "R": R, "ok": True})
        return info

    def _on_miss(self, info: dict):
        self.st.misses += 1
        if self.st.state == "TRACK" and self.st.misses >= self.cfg.weak_after:
            self.st.state = "WEAK_TRACK"
        elif self.st.state == "WEAK_TRACK" and self.st.misses >= self.cfg.weak_after + self.cfg.lost_after:
            self.st.state = "LOST"
            self.st.velocity = None
            self.st.pending_limited_center = None
            self.st.pending_limited_stamp = None
            self.st.pending_limited_limit = None
            self.st.lost_global_retrieval_done = False
        if self.st.state == "LOST":
            self.st.lost_frames += 1
        else:
            self.st.lost_frames = 0
        info.update({"state_out": self.st.state, "ok": False})


if __name__ == "__main__":
    print(__doc__)
