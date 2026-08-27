#!/usr/bin/env python3
"""Production EDM tracker: the state machine of ProductionXFeatTracker, EDM local matching.

Same shape as the XFeat tracker (BOOT_INIT / TRACK / WEAK_TRACK / LOST, staged MegaLoc
at BOOT and per LOST episode, pose-prior + covisibility for tracking), because it keeps
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
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pycolmap

from edm_matcher import GRID_H, GRID_W, EDMMatcher
from reloc_localizer_edm import Camera, EDMLocalizer, EDMRelocMap
from reposed_motion_validator import RelativeMotionCheck, cam_from_world_matrix
from edm_pose_selection import (
    acquire_consensus_limit,
    candidate_inliers,
    _select_acquire_candidate,
    _select_track_candidate,
    _strong_centers_disagree,
)



GLOBAL_RETRIEVAL_ATTEMPTS = 2
LOST_GLOBAL_RETRIEVAL_ATTEMPTS = 1
GLOBAL_RETRIEVAL_RETRY_MULTIPLIER = 2
LOST_PRIOR_STRATEGIES = ("restrict_nearby", "full_global", "score_fusion")
ACQUIRE_STAGE_MODES = ("full_set", "initial_topk")
EDM_OPTIONAL_RUNTIME_TRACKER_KEYS = {
    "lost_prior_strategy",
    "lost_prior_fusion_weight",
    "acquire_stage_mode",
    "pose_consensus_mode",
    "consensus_max_rotation_deg",
    "reference_quality_weight",
    "reference_quality_floor",
}
POSE_CONSENSUS_MODES = ("pairwise", "cluster")


_POSITIVE_INTEGER_CONFIG_FIELDS = (
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

_NONNEGATIVE_INTEGER_CONFIG_FIELDS = (
    "lost_local_grace_frames",
    "lost_global_retrieval_interval",
    "recovery_bank_size",
    "recovery_scan_topk",
    "max_corr_total",
)

_POSITIVE_FLOAT_CONFIG_FIELDS = (
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
    "lost_prior_fusion_weight",
    "consensus_max_rotation_deg",
)


@dataclass
class EDMConfig:
    # acquisition (BOOT_INIT / LOST): MegaLoc retrieval
    boot_global_topk: int = 10
    acquire_initial_topk: int = 2
    acquire_min_inliers: int = 80
    # BOOT tries MegaLoc at boot_global_topk, then once at twice that count.
    # Each LOST episode first spends lost_local_grace_frames on nearby EDM;
    # a still-failed episode fires MegaLoc at lost_local_topk, then only retries
    # when lost_global_retrieval_interval is positive. LOW/WEAK remains geometry-only.
    global_retrieval_policy: str = "boot_and_lost_once"
    lost_local_topk: int = 5
    lost_local_grace_frames: int = 2
    # A zero interval preserves the legacy one-shot LOST retrieval. When set,
    # retry MegaLoc every interval-th LOST frame after the first grace-window shot.
    lost_global_retrieval_interval: int = 0
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
    # Default preserves today's nearby restriction. full_global and score_fusion
    # are measured A/B modes and must be selected explicitly.
    lost_prior_strategy: str = "restrict_nearby"
    lost_prior_fusion_weight: float = 1.0
    # Default evaluates the complete retrieved BOOT/LOST set. initial_topk
    # stages acquire_initial_topk first, then falls back to the full set.
    acquire_stage_mode: str = "full_set"
    # getattr-only pose selection / ranking knobs become profile-owned fields.
    pose_consensus_mode: str = "pairwise"
    consensus_max_rotation_deg: float = 90.0
    reference_quality_weight: float = 0.0
    reference_quality_floor: float = 0.0
    # EDM emits thousands of correspondences; PnP cost is linear in them and RANSAC
    # gains nothing past a well-spread ~900. Cap spatially so the cap does not bias
    # the pose toward whichever image region happened to match densely.
    max_corr_total: int = 900
    corr_grid: int = 8

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.global_retrieval_policy not in {"boot_once", "boot_and_lost_once"}:
            raise ValueError("global_retrieval_policy must be 'boot_once' or 'boot_and_lost_once'")
        if self.lost_prior_strategy not in LOST_PRIOR_STRATEGIES:
            raise ValueError(
                "lost_prior_strategy must be 'restrict_nearby', 'full_global', or 'score_fusion'"
            )
        if self.acquire_stage_mode not in ACQUIRE_STAGE_MODES:
            raise ValueError("acquire_stage_mode must be 'full_set' or 'initial_topk'")
        if self.pose_consensus_mode not in POSE_CONSENSUS_MODES:
            raise ValueError("pose_consensus_mode must be 'pairwise' or 'cluster'")
        _validate_integer_config_fields(self)
        if not isinstance(self.use_temporal_reference, bool):
            raise ValueError("use_temporal_reference must be boolean")
        _validate_float_config_fields(self)
        _validate_acquisition_config(self)
        _validate_jump_config(self)


def register_optional_runtime_tracker_keys() -> None:
    from edm_profile import EDM_OPTIONAL_TRACKER_KEYS

    EDM_OPTIONAL_TRACKER_KEYS.update(EDM_OPTIONAL_RUNTIME_TRACKER_KEYS)


register_optional_runtime_tracker_keys()


def _validate_integer_config_fields(config: EDMConfig) -> None:
    for name in _POSITIVE_INTEGER_CONFIG_FIELDS:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in _NONNEGATIVE_INTEGER_CONFIG_FIELDS:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")


def _validate_float_config_fields(config: EDMConfig) -> None:
    for name in _POSITIVE_FLOAT_CONFIG_FIELDS:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")


def _validate_acquisition_config(config: EDMConfig) -> None:
    if config.acquire_initial_topk > config.boot_global_topk:
        raise ValueError("acquire_initial_topk cannot exceed boot_global_topk")
    if (
        isinstance(config.min_inlier_ratio, bool)
        or not isinstance(config.min_inlier_ratio, (int, float))
        or not math.isfinite(float(config.min_inlier_ratio))
        or not 0.0 < float(config.min_inlier_ratio) <= 1.0
    ):
        raise ValueError("min_inlier_ratio must be finite and within (0, 1]")
    for name in ("reference_quality_weight", "reference_quality_floor"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and >= 0")
    if config.min_inlier_grid_cells > config.corr_grid * config.corr_grid:
        raise ValueError("min_inlier_grid_cells cannot exceed corr_grid squared")
    if config.local_topk > config.weak_local_topk or config.local_topk > config.lost_local_topk:
        raise ValueError("weak/lost local top-k cannot be smaller than TRACK local_topk")


def _validate_jump_config(config: EDMConfig) -> None:
    if config.adaptive_jump_min_history > config.adaptive_jump_history_size:
        raise ValueError("adaptive_jump_min_history cannot exceed adaptive_jump_history_size")
    if config.adaptive_jump_floor > config.adaptive_jump_ceiling:
        raise ValueError("adaptive_jump_floor cannot exceed adaptive_jump_ceiling")
    if config.adaptive_jump_ceiling > config.adaptive_jump_bootstrap:
        raise ValueError("adaptive_jump_bootstrap cannot be smaller than adaptive_jump_ceiling")
    for name in (
        "adaptive_jump_floor",
        "adaptive_jump_bootstrap",
        "adaptive_jump_ceiling",
    ):
        if float(getattr(config, name)) > float(config.max_jump):
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
    accepted_step_norms: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    observed_capture_dts: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    last_observed_capture_stamp: float | None = None
    pending_limited_center: np.ndarray | None = None
    pending_limited_stamp: float | None = None
    pending_limited_limit: float | None = None
    boot_refs: list[str] = field(default_factory=list)
    global_retrieval_calls: int = 0
    boot_global_retrieval_attempts: int = 0
    lost_global_retrieval_attempts: int = 0
    lost_global_retrieval_done: bool = False
    lost_frames: int = 0
    recovery_cursor: int = 0


@dataclass(frozen=True)
class _CandidateSelection:
    acquiring: bool
    refs: list[str]
    mode: str
    min_inliers: int
    vpr_ms: float


@dataclass(frozen=True)
class _CorrespondenceBatch:
    points2d: np.ndarray
    points3d: np.ndarray
    confidence: np.ndarray
    per_ref: list
    by_ref: list | None
    temporal_used: bool


@dataclass(frozen=True)
class _PoseAttempt:
    result: Any | None
    points2d: np.ndarray
    points3d: np.ndarray
    metrics: dict
    selected_ref: str | None
    pnp_ms: float
    reject_reason: str | None = None
    strong_count: int = 0


def _merge_correspondence_rows(rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    points2d_parts = [row[0] for row in rows if len(row[0])]
    points3d_parts = [row[1] for row in rows if len(row[1])]
    confidence_parts = [row[2] for row in rows if len(row[2])]
    points2d = np.concatenate(points2d_parts) if points2d_parts else np.zeros((0, 2))
    points3d = np.concatenate(points3d_parts) if points3d_parts else np.zeros((0, 3))
    confidence = np.concatenate(confidence_parts) if confidence_parts else np.zeros(0)
    return points2d, points3d, confidence, [row[3] for row in rows]


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _reference_groups(ref_names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in ref_names:
        groups.setdefault(name.split("/", 1)[0], []).append(name)
    return groups


def _recovery_allocation(
    groups: dict[str, list[str]],
    max_refs: int,
    total: int,
) -> dict[str, int]:
    keys = list(groups)
    raw = {key: max_refs * len(groups[key]) / total for key in keys}
    allocation = {key: min(len(groups[key]), max(1, int(math.floor(raw[key])))) for key in keys}
    _shrink_recovery_allocation(allocation, raw, max_refs, keys)
    _grow_recovery_allocation(allocation, raw, groups, max_refs, keys)
    return allocation


def _shrink_recovery_allocation(
    allocation: dict[str, int],
    raw: dict[str, float],
    max_refs: int,
    keys: list[str],
) -> None:
    while sum(allocation.values()) > max_refs:
        candidates = [key for key in keys if allocation[key] > 1]
        if not candidates:
            break
        key = max(candidates, key=lambda item: allocation[item] - raw[item])
        allocation[key] -= 1


def _grow_recovery_allocation(
    allocation: dict[str, int],
    raw: dict[str, float],
    groups: dict[str, list[str]],
    max_refs: int,
    keys: list[str],
) -> None:
    while sum(allocation.values()) < max_refs:
        candidates = [key for key in keys if allocation[key] < len(groups[key])]
        if not candidates:
            break
        key = max(candidates, key=lambda item: raw[item] - allocation[item])
        allocation[key] += 1


def _sample_recovery_groups(
    groups: dict[str, list[str]],
    allocation: dict[str, int],
) -> dict[str, list[str]]:
    sampled: dict[str, list[str]] = {}
    for key, group in groups.items():
        indices = np.linspace(0, len(group) - 1, num=allocation[key], dtype=int)
        sampled[key] = [group[int(index)] for index in indices]
    return sampled


def _interleave_recovery_groups(
    sampled: dict[str, list[str]],
    max_refs: int,
) -> list[str]:
    bank: list[str] = []
    for offset in range(max(len(values) for values in sampled.values())):
        for values in sampled.values():
            if offset < len(values):
                bank.append(values[offset])
    return bank[:max_refs]


def build_recovery_bank(ref_names: list[str], max_refs: int) -> list[str]:
    """Create a bounded, sequence-aware reference bank for EDM-only global recovery.

    Slots are allocated roughly in proportion to each sequence size, sampled uniformly
    within the sequence, then interleaved so a cyclic scan reaches every route early.
    """
    if max_refs <= 0 or not ref_names:
        return []
    if len(ref_names) <= max_refs:
        return list(ref_names)
    groups = _reference_groups(ref_names)
    allocation = _recovery_allocation(groups, max_refs, len(ref_names))
    return _interleave_recovery_groups(
        _sample_recovery_groups(groups, allocation),
        max_refs,
    )


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


def spatially_cap_indices(
    pts2d: np.ndarray,
    confidence: np.ndarray,
    max_total: int,
    width: int,
    height: int,
    grid: int = 8,
) -> np.ndarray:
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


def spatially_cap(
    pts2d: np.ndarray,
    pts3d: np.ndarray,
    max_total: int,
    width: int,
    height: int,
    grid: int = 8,
    confidence: np.ndarray | None = None,
):
    """Keep quality-ranked correspondences spatially spread over the image."""
    scores = np.ones(len(pts2d), dtype=np.float32) if confidence is None else confidence
    selected = spatially_cap_indices(pts2d, scores, max_total, width, height, grid)
    return pts2d[selected], pts3d[selected]


def reprojection_metrics(
    ret, pts2d: np.ndarray, pts3d: np.ndarray, camera: pycolmap.Camera, grid: int = 8
) -> dict:
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
    def __init__(
        self,
        reloc_map: EDMRelocMap,
        camera: Camera,
        cfg: EDMConfig | None = None,
        matcher: EDMMatcher | None = None,
        megaloc=None,
        megaloc_factory=None,
        reference_index=None,
        motion_validator=None,
        motion_validation_mode: str = "off",
    ):
        self.cfg = cfg or EDMConfig()
        self.cfg.validate()
        self.map = reloc_map
        self.cam = camera
        self.loc = EDMLocalizer(
            reloc_map,
            camera,
            matcher=matcher,
            megaloc=megaloc,
            megaloc_factory=megaloc_factory,
            pnp_max_error=self.cfg.pnp_ransac_max_error,
            reference_index=reference_index,
        )
        self.st = RuntimeState(
            accepted_step_norms=deque(maxlen=self.cfg.adaptive_jump_history_size),
            observed_capture_dts=deque(maxlen=self.cfg.adaptive_jump_history_size),
        )
        self.centers = (
            None if reloc_map.ref_centers is None else np.asarray(reloc_map.ref_centers, np.float32)
        )
        self.yaws = (
            None if reloc_map.ref_yaws is None else np.asarray(reloc_map.ref_yaws, np.float32)
        )
        self.name_of = {i: n for i, n in enumerate(reloc_map.ref_names)}
        self.idx_of = {n: i for i, n in enumerate(reloc_map.ref_names)}
        self.recovery_bank = build_recovery_bank(
            list(reloc_map.ref_names), self.cfg.recovery_bank_size
        )
        self.temporal_gray: np.ndarray | None = None
        self.temporal_xyz_by_cell: np.ndarray | None = None
        self._pcam: pycolmap.Camera | None = None
        self._pnp_options: pycolmap.AbsolutePoseEstimationOptions | None = None
        self.pose_guided = None
        if motion_validation_mode not in {"off", "shadow", "confirm_limited_jump"}:
            raise ValueError(
                "motion_validation_mode must be off, shadow, or confirm_limited_jump"
            )
        self.motion_validator = motion_validator
        self.motion_validation_mode = motion_validation_mode
        self._last_accepted_bgr: np.ndarray | None = None
        self._last_accepted_cam_from_world: np.ndarray | None = None
        count = len(reloc_map.ref_names)
        self._reference_quality = np.full(count, np.nan, dtype=np.float32)
        self._reference_stability = np.full(count, np.nan, dtype=np.float32)



    def attach_pose_guided(self, controller) -> None:
        self.pose_guided = controller

    def observe_fused_state(self, sample) -> None:
        controller = getattr(self, "pose_guided", None)
        if controller is not None:
            controller.observe_fused(sample)

    def _active_pose_guided(self):
        controller = getattr(self, "pose_guided", None)
        if controller is None or not getattr(controller, "enabled", False):
            return None
        return controller


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
        visual = None
        if self.st.center is not None:
            visual = self.st.center
            if (
                self.st.velocity is not None
                and self.st.last_capture_stamp is not None
                and capture_stamp is not None
            ):
                dt = float(capture_stamp) - float(self.st.last_capture_stamp)
                if math.isfinite(dt) and dt > 0.0:
                    dt = min(dt, max(0.0, float(self.cfg.prediction_max_dt)))
                    visual = self.st.center + self.st.velocity * dt
        controller = getattr(self, "pose_guided", None)
        if controller is None or capture_stamp is None:
            return visual
        prediction = controller.predict(
            float(capture_stamp),
            visual_center=None if self.st.center is None else np.asarray(self.st.center, dtype=float),
            visual_yaw=self.st.yaw,
            visual_velocity=self.st.velocity,
            visual_stamp=self.st.last_capture_stamp,
        )

        if prediction.valid and prediction.position is not None:
            return np.asarray(prediction.position, dtype=float)
        return visual


    def _near_reference_indices(
        self,
        distances: np.ndarray,
        finite: np.ndarray,
    ) -> list[int]:
        near = []
        radius = float(self.cfg.radius)
        max_yaw = math.radians(self.cfg.max_yaw_diff_deg)
        predicted_yaw = self.st.yaw
        controller = self._active_pose_guided()
        if controller is not None:
            limits = controller.search_limits(
                state=self.st.state,
                misses=int(self.st.misses),
                weak_after=int(self.cfg.weak_after),
                base_radius=float(self.cfg.radius),
                base_yaw_rad=math.radians(self.cfg.max_yaw_diff_deg),
                base_max_refs=int(self.cfg.near_pool),
            )
            radius = limits.radius
            max_yaw = limits.max_yaw_rad
            prediction = getattr(controller, "last_prediction", None)
            if prediction is not None and prediction.valid and prediction.yaw is not None:
                predicted_yaw = prediction.yaw
        for raw_index in np.argsort(np.where(finite, distances, np.inf)):
            index = int(raw_index)
            if not finite[index]:
                continue
            if distances[index] > radius and len(near) >= self.cfg.near_pool:
                break
            if (
                predicted_yaw is not None
                and self.yaws is not None
                and np.isfinite(self.yaws[index])
                and abs(_angle_diff(float(self.yaws[index]), predicted_yaw)) > max_yaw
            ):
                continue
            near.append(index)
            if len(near) >= self.cfg.near_pool:
                break
        return near


    def _covisible_reference_indices(self, near: list[int]) -> list[int]:
        covis_refs: list[int] = []
        if self.map.covis:
            for index in list(self.st.last_refs)[:2] + near[:2]:
                values = self.map.covis.get(self.name_of[index], [])[: self.cfg.covis_per_ref]
                covis_refs.extend(int(value) for value in values)
        return covis_refs

    def _track_candidate_score(
        self,
        index: int,
        distances: np.ndarray,
        last: set[int],
    ) -> float:
        score = 2.0 * math.exp(-float(distances[index]) / max(self.cfg.radius, 1e-6))
        if self.st.yaw is not None and self.yaws is not None and np.isfinite(self.yaws[index]):
            score += 0.8 * math.exp(
                -abs(_angle_diff(float(self.yaws[index]), self.st.yaw)) / math.radians(45.0)
            )
        else:
            score += 0.4
        score += 1.0 if index in last else 0.0
        weight = float(self.cfg.reference_quality_weight)
        if weight > 0.0:
            score += weight * self._reference_rank_score(index)
        return score

    def _reference_rank_score(self, index: int) -> float:
        quality_row = getattr(self, "_reference_quality", None)
        stability_row = getattr(self, "_reference_stability", None)
        if quality_row is None or stability_row is None:
            return 0.0
        quality = self._reference_row_value(quality_row, index)
        stability = self._reference_row_value(stability_row, index)
        if quality is None and stability is None:
            return 0.0
        if quality is None:
            combined = stability
        elif stability is None:
            combined = quality
        else:
            combined = 0.5 * (quality + stability)
        return max(float(combined), float(self.cfg.reference_quality_floor))

    def _reference_row_value(self, row: np.ndarray, index: int) -> float | None:
        if row is None or index < 0 or index >= len(row):
            return None
        value = float(row[index])
        if not math.isfinite(value):
            return None
        return value

    def _record_reference_stability(self, indices: Sequence[int], metrics: dict) -> None:
        quality_row = getattr(self, "_reference_quality", None)
        stability_row = getattr(self, "_reference_stability", None)
        if quality_row is None or stability_row is None:
            return
        ratio = metrics.get("inlier_ratio")
        quality = None
        if not isinstance(ratio, bool) and isinstance(ratio, (int, float)):
            number = float(ratio)
            if math.isfinite(number):
                quality = max(0.0, min(1.0, number))
        for index in indices:
            if index < 0 or index >= len(quality_row):
                continue
            previous = self._reference_row_value(stability_row, index)
            stability_row[index] = 1.0 if previous is None else 0.5 * previous + 0.5
            if quality is not None:
                quality_row[index] = quality



    def _track_candidates(self, topk: int, capture_stamp: float | None = None) -> list[str]:
        """Geometry only: nearest references to the predicted pose, plus their covisibles."""
        C = self._predict_center(capture_stamp)
        if C is None or self.centers is None:
            return []
        controller = self._active_pose_guided()
        if controller is not None and controller.last_prediction is not None:
            quality_scores = None
            stability_scores = None
            if float(self.cfg.reference_quality_weight) > 0.0:
                quality_scores = np.asarray(self._reference_quality, dtype=float)
                stability_scores = np.asarray(self._reference_stability, dtype=float)
            selected = controller.select_references(
                names=list(self.map.ref_names),
                centers=self.centers,
                yaws=self.yaws,
                prediction=controller.last_prediction,
                state=self.st.state,
                misses=int(self.st.misses),
                weak_after=int(self.cfg.weak_after),
                base_radius=float(self.cfg.radius),
                base_yaw_rad=math.radians(self.cfg.max_yaw_diff_deg),
                base_max_refs=int(topk),
                last_indices=list(self.st.last_refs),
                covisible_indices=self._covisible_reference_indices([]),
                quality_scores=quality_scores,
                stability_scores=stability_scores,
            )
            if selected:
                return selected[:topk]
        d = np.linalg.norm(self.centers - C[None, :], axis=1)
        finite = np.isfinite(d)
        near = self._near_reference_indices(d, finite)
        covis_refs = self._covisible_reference_indices(near)
        union = list(dict.fromkeys(near + covis_refs + list(self.st.last_refs)))
        last = set(self.st.last_refs)
        union = [i for i in union if 0 <= i < len(self.map.ref_names) and finite[i]]
        union.sort(
            key=lambda index: self._track_candidate_score(index, d, last),
            reverse=True,
        )
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

    def _observe_capture_stamp(self, capture_stamp: float) -> None:
        previous = self.st.last_observed_capture_stamp
        if previous is None or capture_stamp > previous:
            if previous is not None:
                self.st.observed_capture_dts.append(capture_stamp - previous)
            self.st.last_observed_capture_stamp = capture_stamp

    def _global_retrieval(
        self,
        frame_bgr: np.ndarray,
        capture_stamp: float,
    ) -> tuple[list[str], float, bool]:
        import cv2

        lost = self.st.state == "LOST"
        attempts = (
            self.st.lost_global_retrieval_attempts
            if lost
            else self.st.boot_global_retrieval_attempts
        )
        topk = (
            self.cfg.lost_local_topk
            if lost
            else self.cfg.boot_global_topk * (
                GLOBAL_RETRIEVAL_RETRY_MULTIPLIER if attempts else 1
            )
        )
        candidates = None
        prior_fresh = False
        if lost and attempts == 0:
            prior_age = (
                None
                if self.st.last_capture_stamp is None
                else capture_stamp - self.st.last_capture_stamp
            )
            prior_fresh = (
                prior_age is not None
                and math.isfinite(prior_age)
                and 0.0 <= prior_age <= self.cfg.lost_prior_max_age_s
            )
            if self.cfg.lost_prior_strategy == "restrict_nearby" and prior_fresh:
                nearby = self._track_candidates(self.cfg.near_pool, capture_stamp)
                candidates = nearby or None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        if lost and self.cfg.lost_prior_strategy == "score_fusion":
            refs = self._lost_score_fusion_refs(rgb, topk, capture_stamp, prior_fresh)
            self._last_retrieval_kind = "fused"
        elif candidates is not None:
            refs = self.loc.retrieve(rgb, topk, candidates=candidates)
            self._last_retrieval_kind = "near"
        else:
            refs = self.loc.retrieve(rgb, topk)
            self._last_retrieval_kind = "global"
        elapsed_ms = (time.perf_counter() - started) * 1e3
        self.st.global_retrieval_calls += 1
        if lost:
            self.st.lost_global_retrieval_attempts += 1
            self.st.lost_global_retrieval_done = (
                self.st.lost_global_retrieval_attempts
                >= LOST_GLOBAL_RETRIEVAL_ATTEMPTS
            )
        else:
            self.st.boot_global_retrieval_attempts += 1
        if not lost and len(refs) > len(self.st.boot_refs):
            self.st.boot_refs = list(refs)
        return refs, elapsed_ms, candidates is not None

    def _should_run_global_retrieval(self) -> bool:
        if self.cfg.global_retrieval_policy == "boot_and_lost_once":
            if (
                self.st.state == "BOOT_INIT"
                and self.st.boot_global_retrieval_attempts
                < GLOBAL_RETRIEVAL_ATTEMPTS
            ):
                return True
            return (
                self.st.state == "LOST"
                and self.st.lost_frames > self.cfg.lost_local_grace_frames
                and (
                    self.st.lost_global_retrieval_attempts == 0
                    or (
                        self.cfg.lost_global_retrieval_interval > 0
                        and (
                            self.st.lost_frames
                            - self.cfg.lost_local_grace_frames
                            - 1
                        )
                        % self.cfg.lost_global_retrieval_interval
                        == 0
                    )
                )
            )
        if self.cfg.global_retrieval_policy == "boot_once":
            return self.st.global_retrieval_calls == 0
        return True

    def _local_acquisition_candidates(
        self,
        capture_stamp: float,
    ) -> tuple[list[str], str]:
        if self.st.state == "LOST" and self.st.lost_frames > self.cfg.lost_local_grace_frames:
            refs = self._recovery_candidates(self.cfg.recovery_scan_topk)
            mode = "edm_map_scan"
        elif self.st.state == "LOST":
            refs = self._track_candidates(self.cfg.lost_local_topk, capture_stamp)
            mode = "edm_local_recovery"
        else:
            refs = list(self.st.boot_refs)
            mode = "edm_boot_refs"
        if not refs:
            refs = list(self.st.boot_refs[: self.cfg.lost_local_topk])
            mode = "edm_boot_refs"
        return refs, mode

    def _acquisition_candidates(
        self,
        frame_bgr: np.ndarray,
        capture_stamp: float,
    ) -> _CandidateSelection:
        if self._should_run_global_retrieval():
            retry = (
                self.st.lost_global_retrieval_attempts
                if self.st.state == "LOST"
                else self.st.boot_global_retrieval_attempts
            ) > 0
            refs, vpr_ms, nearby = self._global_retrieval(
                frame_bgr,
                capture_stamp,
            )
            if self.st.state == "LOST" and getattr(self, "_last_retrieval_kind", None) == "fused":
                mode = "megaloc_lost_fused"
            elif self.st.state == "LOST" and nearby:
                mode = "megaloc_lost_near"
            elif self.st.state == "LOST" and retry:
                mode = "megaloc_lost_global_retry"
            else:
                mode = "megaloc_lost" if self.st.state == "LOST" else "megaloc_boot"
            if retry and self.st.state != "LOST":
                mode += "_retry"
        else:
            refs, mode = self._local_acquisition_candidates(capture_stamp)
            vpr_ms = 0.0
        return _CandidateSelection(
            acquiring=True,
            refs=refs,
            mode=mode,
            min_inliers=self.cfg.acquire_min_inliers,
            vpr_ms=vpr_ms,
        )

    def _tracking_candidates(
        self,
        frame_bgr: np.ndarray,
        capture_stamp: float,
    ) -> _CandidateSelection:
        temporal_ready = (
            self.cfg.use_temporal_reference
            and self.temporal_gray is not None
            and self.temporal_xyz_by_cell is not None
        )
        topk = (
            self.cfg.temporal_map_topk
            if self.st.state == "TRACK" and temporal_ready
            else self.cfg.local_topk
            if self.st.state == "TRACK"
            else self.cfg.weak_local_topk
        )
        refs = self._track_candidates(topk, capture_stamp)
        mode = "edm_temporal_map" if temporal_ready else "edm_track"
        vpr_ms = 0.0
        if not refs:
            if (
                self.cfg.global_retrieval_policy in {"boot_once", "boot_and_lost_once"}
                and self.st.global_retrieval_calls > 0
            ):
                refs = list(self.st.boot_refs[: self.cfg.lost_local_topk])
                mode = "edm_boot_refs"
            else:
                refs, vpr_ms, _nearby = self._global_retrieval(
                    frame_bgr,
                    capture_stamp,
                )
                mode = "megaloc_reacquire"
        min_inliers = (
            self.cfg.track_min_inliers if self.st.state == "TRACK" else self.cfg.weak_min_inliers
        )
        return _CandidateSelection(
            acquiring=False,
            refs=refs,
            mode=mode,
            min_inliers=min_inliers,
            vpr_ms=vpr_ms,
        )

    def _select_candidates(
        self,
        frame_bgr: np.ndarray,
        capture_stamp: float,
    ) -> _CandidateSelection:
        if self.st.state in ("BOOT_INIT", "LOST"):
            return self._acquisition_candidates(frame_bgr, capture_stamp)
        return self._tracking_candidates(frame_bgr, capture_stamp)

    def _collect_correspondences(
        self,
        gray: np.ndarray,
        selection: _CandidateSelection,
    ) -> _CorrespondenceBatch:
        refs = selection.refs
        temporal_used = (
            selection.mode in ("edm_temporal_map", "edm_local_recovery")
            and self.cfg.use_temporal_reference
            and self.temporal_gray is not None
            and self.temporal_xyz_by_cell is not None
        )
        by_ref = None
        if temporal_used:
            kinds = ["temporal"] + ["map"] * len(refs)
            rows = self.loc.correspondences_for_sources(
                gray,
                [self.temporal_gray] + [self.map.images[name] for name in refs],
                [self.temporal_xyz_by_cell] + [self.map.xyz_by_cell[name] for name in refs],
                batch_size=self.cfg.match_batch_size,
                source_kinds=kinds,
            )
            points2d, points3d, confidence, per_ref = _merge_correspondence_rows(rows)
        else:
            by_ref = self.loc.correspondences_by_ref(
                gray,
                refs,
                batch_size=self.cfg.match_batch_size,
            )
            points2d, points3d, confidence, per_ref = _merge_correspondence_rows(by_ref)
        return _CorrespondenceBatch(
            points2d=points2d,
            points3d=points3d,
            confidence=confidence,
            per_ref=per_ref,
            by_ref=by_ref,
            temporal_used=temporal_used,
        )

    def _estimate_pose_candidate(
        self,
        points2d: np.ndarray,
        points3d: np.ndarray,
        confidence: np.ndarray,
        pcam: pycolmap.Camera,
        options: pycolmap.AbsolutePoseEstimationOptions,
    ) -> tuple[tuple | None, float]:
        if len(points3d) < 6:
            return None, 0.0
        selected = spatially_cap_indices(
            points2d,
            confidence,
            self.cfg.max_corr_total,
            self.cam.width,
            self.cam.height,
            self.cfg.corr_grid,
        )
        capped_points2d = points2d[selected]
        capped_points3d = points3d[selected]
        started = time.perf_counter()
        estimate = pycolmap.estimate_and_refine_absolute_pose(
            np.asarray(capped_points2d, float),
            np.asarray(capped_points3d, float),
            pcam,
            options,
        )
        elapsed_ms = (time.perf_counter() - started) * 1e3
        if estimate is None:
            return None, elapsed_ms
        metrics = reprojection_metrics(
            estimate,
            capped_points2d,
            capped_points3d,
            pcam,
            self.cfg.corr_grid,
        )
        return (
            estimate,
            capped_points2d,
            capped_points3d,
            metrics,
        ), elapsed_ms

    def _acquire_consensus_limit(self) -> float:
        return acquire_consensus_limit(self.cfg)

    @staticmethod
    def _candidate_inliers(candidate: tuple) -> int:
        return candidate_inliers(candidate)

    def _strong_centers_disagree(
        self,
        scored: list[tuple[str, tuple]],
        min_inl: int,
    ) -> bool:
        return _strong_centers_disagree(scored, min_inl, self.cfg)

    def _select_acquire_candidate(
        self,
        selection: _CandidateSelection,
        scored: list[tuple[str, tuple]],
    ) -> tuple[str, tuple] | None:
        return _select_acquire_candidate(selection, scored, self.cfg)

    def _select_track_candidate(
        self,
        selection: _CandidateSelection,
        scored: list[tuple[str, tuple]],
    ) -> tuple[str, tuple] | None:
        return _select_track_candidate(selection, scored, self.cfg)

    def _best_pose_attempt(
        self,
        selection: _CandidateSelection,
        batch: _CorrespondenceBatch,
    ) -> _PoseAttempt:
        pcam, options = self._pose_estimation_context()
        result = None
        selected_ref = None
        points2d = np.zeros((0, 2))
        points3d = np.zeros((0, 3))
        metrics = {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}
        pnp_ms = 0.0
        reject_reason = None
        strong_count = 0
        if batch.by_ref is not None:
            scored: list[tuple[str, tuple]] = []
            for name, (ref_p2, ref_p3, ref_confidence, _) in zip(
                selection.refs,
                batch.by_ref,
            ):
                candidate, elapsed_ms = self._estimate_pose_candidate(
                    ref_p2,
                    ref_p3,
                    ref_confidence,
                    pcam,
                    options,
                )
                pnp_ms += elapsed_ms
                if candidate is not None:
                    scored.append((name, candidate))
            strong_count = sum(
                1
                for _name, candidate in scored
                if self._candidate_inliers(candidate) >= selection.min_inliers
            )
            chosen = None
            if selection.acquiring:
                chosen = self._select_acquire_candidate(selection, scored)
                if scored and chosen is None:
                    reject_reason = "acquire_consensus"
            else:
                chosen = self._select_track_candidate(selection, scored)
                if scored and chosen is None:
                    reject_reason = "track_consensus"
            if chosen is not None:
                selected_ref, candidate = chosen
                result, points2d, points3d, metrics = candidate
        else:
            candidate, pnp_ms = self._estimate_pose_candidate(
                batch.points2d,
                batch.points3d,
                batch.confidence,
                pcam,
                options,
            )
            if candidate is not None:
                result, points2d, points3d, metrics = candidate
        return _PoseAttempt(
            result=result,
            points2d=points2d,
            points3d=points3d,
            metrics=metrics,
            selected_ref=selected_ref,
            pnp_ms=pnp_ms,
            reject_reason=reject_reason,
            strong_count=strong_count,
        )

    def _match_and_estimate(
        self,
        gray: np.ndarray,
        selection: _CandidateSelection,
    ) -> tuple[_CorrespondenceBatch, _PoseAttempt, float]:
        started = time.perf_counter()
        batch = self._collect_correspondences(gray, selection)
        attempt = self._best_pose_attempt(selection, batch)
        match_ms = max(0.0, (time.perf_counter() - started) * 1e3 - attempt.pnp_ms)
        return batch, attempt, match_ms

    def _lost_score_fusion_refs(
        self,
        rgb: np.ndarray,
        topk: int,
        capture_stamp: float,
        prior_fresh: bool,
    ) -> list[str]:
        pool = min(
            len(self.map.ref_names),
            max(int(topk) * 4, int(self.cfg.near_pool)),
        )
        if pool <= 0 or topk <= 0:
            return []
        descriptor = self.loc.megaloc.extract_one(rgb)
        scored = list(
            self.loc.retrieve_scored(rgb, pool, descriptor=descriptor)
        )
        if prior_fresh:
            nearby = self._track_candidates(self.cfg.near_pool, capture_stamp)
            have = {name for name, _score in scored}
            missing = [name for name in nearby if name not in have]
            if missing:
                scored.extend(
                    self.loc.retrieve_scored(
                        rgb, len(missing), candidates=missing, descriptor=descriptor
                    )
                )
        center = self._predict_center(capture_stamp)
        weight = float(self.cfg.lost_prior_fusion_weight)
        radius = max(float(self.cfg.radius), 1e-6)
        fused: list[tuple[str, float]] = []
        for name, vpr_score in scored:
            geom = 0.0
            index = self.idx_of.get(name)
            if (
                center is not None
                and self.centers is not None
                and index is not None
            ):
                distance = float(np.linalg.norm(self.centers[index] - center))
                if np.isfinite(distance):
                    geom = math.exp(-distance / radius)
            fused.append((name, float(vpr_score) + weight * geom))
        fused.sort(key=lambda item: (-item[1], item[0]))
        return [name for name, _score in fused[:topk]]

    def _pose_quality_ok(
        self,
        result: Any | None,
        metrics: dict,
        *,
        min_inliers: int,
        reproj_limit: float,
    ) -> bool:
        if result is None or int(result["num_inliers"]) < min_inliers:
            return False
        rms = metrics.get("reproj_rms")
        if rms is None or float(rms) > float(reproj_limit):
            return False
        if float(metrics.get("inlier_ratio", 0.0)) < float(self.cfg.min_inlier_ratio):
            return False
        if int(metrics.get("inlier_grid_cells", 0)) < self.cfg.min_inlier_grid_cells:
            return False
        return True

    def _trajectory_would_allow(
        self,
        center: np.ndarray,
        yaw: float,
        capture_stamp: float,
        state_in: str,
    ) -> bool:
        if self.st.center is None:
            return True
        if state_in != "LOST":
            raw_step = float(np.linalg.norm(center - self.st.center))
            if raw_step > self.cfg.max_jump:
                return False
            capture_dt = (
                None
                if self.st.last_capture_stamp is None
                else capture_stamp - self.st.last_capture_stamp
            )
            _limited, jump_info = limit_center_step(
                self.st.center,
                center,
                self.st.accepted_step_norms,
                self.cfg,
                capture_dt=capture_dt,
                capture_dt_history=self.st.observed_capture_dts,
            )
            return not jump_info["limited"]
        prior_age = (
            None
            if self.st.last_capture_stamp is None
            else capture_stamp - self.st.last_capture_stamp
        )
        prior_fresh = (
            prior_age is not None
            and math.isfinite(prior_age)
            and 0.0 <= prior_age <= self.cfg.lost_prior_max_age_s
        )
        if not prior_fresh:
            return True
        acquire_limit = float(self.cfg.acquire_max_jump_factor) * float(self.cfg.max_jump)
        if float(np.linalg.norm(center - self.st.center)) > acquire_limit:
            return False
        prior_yaw = self.st.yaw
        if prior_yaw is not None and math.isfinite(float(prior_yaw)):
            if abs(_angle_diff(yaw, float(prior_yaw))) > math.radians(
                float(self.cfg.acquire_max_yaw_diff_deg)
            ):
                return False
        return True

    def _acquire_stage_can_stop(
        self,
        selection: _CandidateSelection,
        attempt: _PoseAttempt,
        capture_stamp: float,
    ) -> bool:
        if attempt.reject_reason or attempt.result is None:
            return False
        if attempt.strong_count < 2:
            return False
        reproj_limit = (
            self.cfg.max_reproj_error_acquire
            if selection.acquiring
            else self.cfg.max_reproj_error_track
        )
        if not self._pose_quality_ok(
            attempt.result,
            attempt.metrics,
            min_inliers=selection.min_inliers,
            reproj_limit=reproj_limit,
        ):
            return False
        _rotation, center, yaw = self._pose_components(attempt.result)
        return self._trajectory_would_allow(center, yaw, capture_stamp, self.st.state)


    def _localization_info(
        self,
        selection: _CandidateSelection,
        batch: _CorrespondenceBatch,
        attempt: _PoseAttempt,
        *,
        match_ms: float,
        started: float,
    ) -> dict:
        matcher = getattr(self.loc, "matcher", None)
        cache_stats = None
        read_class_stats = getattr(matcher, "feature_cache_stats_by_class", None)
        if callable(read_class_stats):
            cache_stats = read_class_stats()
        info = {
            "frame": self.st.frame,
            "state_in": self.st.state,
            "refs": selection.refs,
            "candidate_mode": selection.mode,
            "temporal_used": batch.temporal_used,
            "selected_ref": attempt.selected_ref,
            "requested_reference_count": len(selection.refs),
            "staged_early_stop": False,
            "acquire_stage_mode": self.cfg.acquire_stage_mode,
            "lost_prior_strategy": self.cfg.lost_prior_strategy,
            "lost_prior_fusion_weight": float(self.cfg.lost_prior_fusion_weight),
            "runtime_sigma_mode": getattr(matcher, "runtime_sigma_mode", None),
            "feature_cache_classes": cache_stats,
            "motion_cache_active": self._motion_cache_active(),
            "n_corr": int(len(batch.points3d)),
            "per_ref": batch.per_ref,
            "global_retrieval_calls": self.st.global_retrieval_calls,
            "lost_global_retrieval_done": self.st.lost_global_retrieval_done,
            "vpr_ms": selection.vpr_ms,
            "match_ms": match_ms,
            "pnp_ms": attempt.pnp_ms,
            "total_ms": (time.perf_counter() - started) * 1e3,
        }
        info.update(attempt.metrics)
        controller = getattr(self, "pose_guided", None)
        if controller is not None:
            info.update(controller.predicted_only(controller.last_prediction))
        return info


    def _clear_pending_jump(self) -> None:
        self.st.pending_limited_center = None
        self.st.pending_limited_stamp = None
        self.st.pending_limited_limit = None

    def _clear_visual_motion_cache(self) -> None:
        self._last_accepted_bgr = None
        self._last_accepted_cam_from_world = None

    def _motion_cache_active(self) -> bool:
        return (
            getattr(self, "motion_validator", None) is not None
            and getattr(self, "motion_validation_mode", "off")
            in {"shadow", "confirm_limited_jump"}
        )

    def _store_visual_motion_cache(self, frame_bgr: np.ndarray, cam_from_world) -> None:
        if not self._motion_cache_active():
            return
        self._last_accepted_bgr = np.ascontiguousarray(frame_bgr).copy()
        self._last_accepted_cam_from_world = cam_from_world_matrix(cam_from_world)


    def _relative_motion_cache_ready(self) -> bool:
        return (
            getattr(self, "_last_accepted_bgr", None) is not None
            and getattr(self, "_last_accepted_cam_from_world", None) is not None
        )


    def _run_relative_motion_check(
        self,
        frame_bgr: np.ndarray,
        cam_from_world,
    ) -> RelativeMotionCheck:
        try:
            return self.motion_validator.check(
                self._last_accepted_bgr,
                frame_bgr,
                self._last_accepted_cam_from_world,
                cam_from_world,
            )
        except Exception as exc:
            return RelativeMotionCheck(
                status="unavailable",
                reason=f"validator_error:{type(exc).__name__}:{exc}",
                matches=0,
                inliers=0,
                rotation_delta_deg=None,
                translation_direction_delta_deg=None,
                depth_ms=0.0,
                match_ms=0.0,
                solver_ms=0.0,
                total_ms=0.0,
            )


    def _reject_pose(
        self,
        info: dict,
        result: Any | None,
        *,
        reason: str | None = None,
        detail: dict | None = None,
    ) -> None:
        self._clear_pending_jump()
        info["inliers"] = 0 if result is None else int(result["num_inliers"])
        if reason is not None:
            info["rejected"] = reason
        if detail:
            info.update(detail)
        self._on_miss(info)

    def _pose_quality_rejected(
        self,
        result: Any | None,
        info: dict,
        *,
        min_inliers: int,
        reproj_limit: float,
    ) -> bool:
        if result is None or int(result["num_inliers"]) < min_inliers:
            self._reject_pose(info, result)
            return True
        if info["reproj_rms"] is None or float(info["reproj_rms"]) > float(reproj_limit):
            reason = "reprojection_unavailable" if info["reproj_rms"] is None else "reprojection"
            self._reject_pose(
                info,
                result,
                reason=reason,
                detail={"max_reproj_error": float(reproj_limit)},
            )
            return True
        if float(info["inlier_ratio"]) < float(self.cfg.min_inlier_ratio):
            self._reject_pose(
                info,
                result,
                reason="inlier_ratio",
                detail={"min_inlier_ratio": float(self.cfg.min_inlier_ratio)},
            )
            return True
        if int(info["inlier_grid_cells"]) < self.cfg.min_inlier_grid_cells:
            self._reject_pose(
                info,
                result,
                reason="inlier_spread",
                detail={"min_inlier_grid_cells": self.cfg.min_inlier_grid_cells},
            )
            return True
        return False

    @staticmethod
    def _pose_components(result: Any) -> tuple[np.ndarray, np.ndarray, float]:
        transform = result["cam_from_world"]
        rotation = transform.rotation.matrix()
        center = -rotation.T @ np.asarray(transform.translation)
        forward = rotation.T @ np.array([0, 0, 1.0])
        yaw = float(math.atan2(forward[1], forward[0]))
        return rotation, center, yaw

    def _limited_jump_residual(
        self,
        center: np.ndarray,
        capture_stamp: float,
    ) -> tuple[bool, str | None, float | None]:
        pending = self.st.pending_limited_center
        pending_stamp = self.st.pending_limited_stamp
        independent = pending_stamp is not None and capture_stamp > pending_stamp
        model = None
        residual = None
        if pending is not None and independent:
            stationary_residual = float(np.linalg.norm(center - pending))
            model = "stationary"
            residual = stationary_residual
            if self.st.last_capture_stamp is not None and pending_stamp is not None:
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
                    motion_residual = float(np.linalg.norm(center - predicted))
                    if motion_residual < stationary_residual:
                        model = "constant_velocity"
                        residual = motion_residual
            return independent, model, residual
        return self._motion_prior_residual(center, capture_stamp)

    def _motion_prior_residual(
        self,
        center: np.ndarray,
        capture_stamp: float,
    ) -> tuple[bool, str | None, float | None]:
        controller = getattr(self, "pose_guided", None)
        prediction = getattr(controller, "last_prediction", None) if controller is not None else None
        if (
            prediction is not None
            and getattr(prediction, "valid", False)
            and getattr(prediction, "position", None) is not None
        ):
            predicted = np.asarray(prediction.position, dtype=float)
            if predicted.shape == (3,) and np.isfinite(predicted).all():
                return True, "predicted_center", float(np.linalg.norm(center - predicted))
        if (
            self.st.velocity is None
            or self.st.center is None
            or self.st.last_capture_stamp is None
        ):
            return False, None, None
        dt = float(capture_stamp) - float(self.st.last_capture_stamp)
        if not math.isfinite(dt) or dt <= 1e-6:
            return False, None, None
        dt = min(dt, float(self.cfg.prediction_max_dt))
        predicted = np.asarray(self.st.center, dtype=float) + np.asarray(
            self.st.velocity, dtype=float
        ) * dt
        if predicted.shape != (3,) or not np.isfinite(predicted).all():
            return False, None, None
        return True, "accepted_velocity_prior", float(np.linalg.norm(center - predicted))

    def _confirm_limited_jump(
        self,
        center: np.ndarray,
        capture_stamp: float,
        capture_dt: float | None,
        jump_info: dict,
        info: dict,
    ) -> bool:
        confirmation_limit = max(
            float(jump_info["limit"]),
            float(self.st.pending_limited_limit or 0.0),
        )
        independent, model, residual = self._limited_jump_residual(
            center,
            capture_stamp,
        )
        info["limited_jump"] = {
            "raw_step": jump_info["raw_step"],
            "limit": jump_info["limit"],
            "capture_dt": capture_dt,
            "confirmation_model": model,
            "confirmation_residual": residual,
            "confirmation_limit": confirmation_limit,
            "confirmation_independent": independent,
        }
        if residual is not None and residual <= confirmation_limit:
            info["limited_jump_confirmed"] = True
            self._clear_pending_jump()
            return True
        had_pending = bool(independent)
        self.st.pending_limited_center = np.asarray(center, dtype=float).copy()
        self.st.pending_limited_stamp = capture_stamp
        self.st.pending_limited_limit = float(jump_info["limit"])
        info["rejected"] = "limited_jump_unconfirmed"
        if had_pending:
            self._on_miss(info)
        else:
            info.update({"state_out": self.st.state, "ok": False})
            if info.get("pose_status") == "VISUALLY_CONFIRMED":
                info["pose_status"] = "NONE"
        return False

    def _continuous_trajectory_allows(
        self,
        center: np.ndarray,
        capture_stamp: float,
        info: dict,
        frame_bgr: np.ndarray,
        cam_from_world,
    ) -> bool:
        capture_dt = (
            None
            if self.st.last_capture_stamp is None
            else capture_stamp - self.st.last_capture_stamp
        )
        raw_step = float(np.linalg.norm(center - self.st.center))
        if raw_step > self.cfg.max_jump:
            self._clear_pending_jump()
            info["limited_jump"] = {
                "raw_step": raw_step,
                "limit": float(self.cfg.max_jump),
                "capture_dt": capture_dt,
                "confirmation_model": None,
                "confirmation_residual": None,
                "confirmation_limit": None,
            }
            info["rejected"] = "jump"
            self._on_miss(info)
            return False
        _limited_center, jump_info = limit_center_step(
            self.st.center,
            center,
            self.st.accepted_step_norms,
            self.cfg,
            capture_dt=capture_dt,
            capture_dt_history=self.st.observed_capture_dts,
        )
        if jump_info["limited"]:
            pending = self.st.pending_limited_center is not None
            mode = getattr(self, "motion_validation_mode", "off")
            validator = getattr(self, "motion_validator", None)
            if (
                not pending
                and validator is not None
                and mode in {"shadow", "confirm_limited_jump"}
                and self._relative_motion_cache_ready()
            ):
                check = self._run_relative_motion_check(frame_bgr, cam_from_world)
                info["relative_motion_check"] = check.as_json_dict()
                if mode == "confirm_limited_jump" and check.status == "agree":
                    info["limited_jump"] = {
                        "raw_step": jump_info["raw_step"],
                        "limit": jump_info["limit"],
                        "capture_dt": capture_dt,
                        "confirmation_model": "reposed_relative_pose",
                        "confirmation_residual": None,
                        "confirmation_limit": float(jump_info["limit"]),
                        "confirmation_independent": True,
                    }
                    info["limited_jump_confirmed"] = True
                    self._clear_pending_jump()
                    return True
            return self._confirm_limited_jump(
                center,
                capture_stamp,
                capture_dt,
                jump_info,
                info,
            )
        self._clear_pending_jump()
        return True


    def _lost_reacquisition_allows(
        self,
        center: np.ndarray,
        yaw: float,
        capture_stamp: float,
        info: dict,
    ) -> bool:
        prior_age = (
            None
            if self.st.last_capture_stamp is None
            else capture_stamp - self.st.last_capture_stamp
        )
        prior_fresh = (
            prior_age is not None
            and math.isfinite(prior_age)
            and 0.0 <= prior_age <= self.cfg.lost_prior_max_age_s
        )
        info["acquire_prior_age"] = prior_age
        if not prior_fresh:
            return True
        acquire_limit = float(self.cfg.acquire_max_jump_factor) * float(self.cfg.max_jump)
        acquire_step = float(np.linalg.norm(center - self.st.center))
        info["acquire_step"] = acquire_step
        info["acquire_limit"] = acquire_limit
        if acquire_step > acquire_limit:
            info["rejected"] = "acquire_jump"
            self._on_miss(info)
            return False
        prior_yaw = self.st.yaw
        if prior_yaw is not None and math.isfinite(float(prior_yaw)):
            yaw_delta = abs(_angle_diff(yaw, float(prior_yaw)))
            info["acquire_yaw_delta_deg"] = math.degrees(yaw_delta)
            if yaw_delta > math.radians(float(self.cfg.acquire_max_yaw_diff_deg)):
                info["rejected"] = "acquire_yaw"
                self._on_miss(info)
                return False
        return True

    def _trajectory_allows(
        self,
        center: np.ndarray,
        yaw: float,
        capture_stamp: float,
        info: dict,
        frame_bgr: np.ndarray,
        cam_from_world,
    ) -> bool:
        if self.st.center is None:
            return True
        if info["state_in"] != "LOST":
            return self._continuous_trajectory_allows(
                center,
                capture_stamp,
                info,
                frame_bgr,
                cam_from_world,
            )
        return self._lost_reacquisition_allows(center, yaw, capture_stamp, info)


    def _accept_pose(
        self,
        info: dict,
        result: Any,
        rotation: np.ndarray,
        center: np.ndarray,
        yaw: float,
        capture_stamp: float,
        selection: _CandidateSelection,
        attempt: _PoseAttempt,
        gray: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> dict:
        previous_center = self.st.center
        previous_stamp = self.st.last_capture_stamp
        step = None if previous_center is None else (center - previous_center)
        measured_velocity = None
        if step is not None:
            dt = None if previous_stamp is None else capture_stamp - previous_stamp
            if info["state_in"] != "LOST" and dt is not None and math.isfinite(dt) and dt > 1e-6:
                self.st.accepted_step_norms.append(float(np.linalg.norm(step)))
                measured_velocity = step / dt
        self.st.velocity = (
            measured_velocity
            if self.st.velocity is None or measured_velocity is None
            else 0.5 * (self.st.velocity + measured_velocity)
        )
        self.st.center, self.st.yaw = center, yaw
        self.st.last_capture_stamp = capture_stamp
        accepted_refs = (
            [attempt.selected_ref] if attempt.selected_ref is not None else selection.refs
        )
        self.st.last_refs = [self.idx_of[name] for name in accepted_refs]
        self._record_reference_stability(self.st.last_refs, attempt.metrics)
        self.st.misses = 0
        self.st.lost_frames = 0
        self.st.lost_global_retrieval_attempts = 0
        self.st.lost_global_retrieval_done = False
        self._store_visual_motion_cache(frame_bgr, result["cam_from_world"])
        self.st.state = "TRACK"
        if self.cfg.use_temporal_reference:
            inlier_mask = np.asarray(
                result.get(
                    "inlier_mask",
                    np.ones(len(attempt.points3d), dtype=bool),
                ),
                dtype=bool,
            )
            self.temporal_xyz_by_cell = build_temporal_lut(
                attempt.points2d,
                attempt.points3d,
                inlier_mask,
                camera_to_edm_scale=self.loc.scale,
            )
            self.temporal_gray = gray.copy()
        else:
            self.temporal_xyz_by_cell = None
            self.temporal_gray = None
        info.update(
            {
                "state_out": "TRACK",
                "center": center,
                "yaw": yaw,
                "R": rotation,
                "ok": True,
                "pose_status": "VISUALLY_CONFIRMED",
            }
        )
        controller = getattr(self, "pose_guided", None)
        if controller is not None:
            controller.maybe_update_anchor(
                info,
                timestamp=capture_stamp,
                rotation_cam_from_world=rotation,
                center=center,
                yaw=yaw,
            )
        return info


    # ---------- one frame ----------
    def localize(self, frame_bgr: np.ndarray, capture_stamp: float | None = None) -> dict:
        cfg = self.st_cfg = self.cfg
        t0 = time.perf_counter()
        if capture_stamp is None:
            capture_stamp = time.monotonic()
        capture_stamp = float(capture_stamp)
        if not math.isfinite(capture_stamp):
            raise ValueError("capture_stamp must be finite")
        self._observe_capture_stamp(capture_stamp)
        self.st.frame += 1
        gray = EDMMatcher.load_gray(frame_bgr)

        selection = self._select_candidates(frame_bgr, capture_stamp)
        requested = len(selection.refs)
        staged_early_stop = False
        if (
            selection.acquiring
            and self.cfg.acquire_stage_mode == "initial_topk"
            and requested > self.cfg.acquire_initial_topk
        ):
            staged = replace(
                selection, refs=list(selection.refs[: self.cfg.acquire_initial_topk])
            )
            batch, attempt, match_ms = self._match_and_estimate(gray, staged)
            if self._acquire_stage_can_stop(staged, attempt, capture_stamp):
                selection = staged
                staged_early_stop = True
            else:
                batch, attempt, match_ms = self._match_and_estimate(gray, selection)
        else:
            batch, attempt, match_ms = self._match_and_estimate(gray, selection)
        info = self._localization_info(
            selection,
            batch,
            attempt,
            match_ms=match_ms,
            started=t0,
        )
        info["requested_reference_count"] = requested
        info["staged_early_stop"] = staged_early_stop
        ret = attempt.result
        min_inl = selection.min_inliers
        reproj_limit = (
            cfg.max_reproj_error_acquire if selection.acquiring else cfg.max_reproj_error_track
        )
        if attempt.reject_reason:
            self._reject_pose(info, ret, reason=attempt.reject_reason)
            return info
        if self._pose_quality_rejected(
            ret,
            info,
            min_inliers=min_inl,
            reproj_limit=reproj_limit,
        ):
            return info

        R, C, yaw = self._pose_components(ret)
        info["inliers"] = int(ret["num_inliers"])

        # Two-layer trajectory gate: hard jumps fail immediately; adaptive jumps
        # require an independent confirming capture. LOST reacquisition uses its
        # separately bounded prior window.
        if not self._trajectory_allows(
            C,
            yaw,
            capture_stamp,
            info,
            frame_bgr,
            ret["cam_from_world"],
        ):
            return info
        return self._accept_pose(
            info,
            ret,
            R,
            C,
            yaw,
            capture_stamp,
            selection,
            attempt,
            gray,
            frame_bgr,
        )

    def _on_miss(self, info: dict):
        self.st.misses += 1
        if self.st.state == "TRACK" and self.st.misses >= self.cfg.weak_after:
            self.st.state = "WEAK_TRACK"
        elif (
            self.st.state == "WEAK_TRACK"
            and self.st.misses >= self.cfg.weak_after + self.cfg.lost_after
        ):
            self.st.state = "LOST"
            self.st.pending_limited_center = None
            self.st.pending_limited_stamp = None
            self.st.pending_limited_limit = None
            self.st.lost_global_retrieval_attempts = 0
            self.st.lost_global_retrieval_done = False
            self._clear_visual_motion_cache()
        if self.st.state == "LOST":
            self.st.lost_frames += 1
        else:
            self.st.lost_frames = 0
        info.update({"state_out": self.st.state, "ok": False})
        if info.get("pose_status") == "VISUALLY_CONFIRMED":
            info["pose_status"] = "NONE"
        controller = getattr(self, "pose_guided", None)
        if controller is not None and info.get("pose_status") != "VISUALLY_CONFIRMED":
            predicted = controller.predicted_only(controller.last_prediction)
            predicted["ok"] = False
            info.update(predicted)



if __name__ == "__main__":
    print(__doc__)
