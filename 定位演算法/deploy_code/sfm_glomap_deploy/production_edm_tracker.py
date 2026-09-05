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

import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Sequence
import cv2
import numpy as np
import pycolmap

try:
    from esekf import ESEKF, EKFConfig, adaptive_visual_covariance as _esekf_adaptive_cov
except Exception:  # pragma: no cover
    ESEKF = None  # type: ignore
    EKFConfig = None  # type: ignore
    _esekf_adaptive_cov = None  # type: ignore


_KLT_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
_KLT_FB_PX = 1.0
# Validation-only hook: when set to ``fn(prev_gray, gray, points) -> (next, status)``
# the bridge's forward and backward passes use that flow backend instead of
# pyramidal LK. ``None`` (production) keeps the LK calls exactly as they were.
# Used by 定位演算法/validation/flow_backend_experiment.py to A/B DIS and
# FastFlowNet through the replay gate.
_FLOW_BACKEND = None
_KLT_MIN_SEED = 50
_KLT_MIN_TRACK = 30
_KLT_MAX_FRAMES = 6
_KLT_MAX_AGE_S = 0.50
# 3D-aware LK window from ESEKF position-covariance trace (px): grows 15->41
# as uncertainty grows. Units: window px, trace map-units^2.
_KLT_COV_WINDOW_MIN = 15
_KLT_COV_WINDOW_MAX = 41
_KLT_COV_WINDOW_GAIN = 3.0
_KLT_COV_WINDOW_SCALE = 0.1
# KLT bridge anchor/drift gates (fractions of the EDM track gates, so they
# follow each site's scale calibration). Bridge needs a strong EDM anchor and
# trips back to EDM when already shaky; _MIN_RATIO stays env-overridable.
_KLT_BRIDGE_ANCHOR_MIN_INLIERS = 60
_KLT_BRIDGE_DRIFT_REPROJ_FACTOR = 0.60
_KLT_BRIDGE_DRIFT_STEP_FACTOR = 0.50
_KLT_BRIDGE_DRIFT_MIN_RATIO = 0.70
# KLT track confidence = W_RATIO*ratio + W_SPATIAL*(cells/SPATIAL_NORM) + W_FB*(1-fb_med).
# NOTE: SPATIAL_NORM is 64.0 while the grid is 8x6=48 cells, so the spatial
# term maxes at 0.3*48/64=0.225, not 0.3. Kept as-is (gate-verified behavior);
# change only with a replay A/B.
# RATIO_HARD_MIN drops the cache outright (needs TRACKED_MIN tracks);
# below CONF_MIN the track survives only with RATIO_SOFT_MIN.
_KLT_CONF_W_RATIO = 0.5
_KLT_CONF_W_SPATIAL = 0.3
_KLT_CONF_W_FB = 0.2
_KLT_CONF_SPATIAL_NORM = 64.0
_KLT_SPATIAL_GRID = (8, 6)
_KLT_SPATIAL_FALLBACK_CELLS = 6
_KLT_RATIO_HARD_MIN = 0.45
_KLT_TRACKED_MIN_FOR_RATIO_GATE = 50
_KLT_CONF_MIN = 0.30
_KLT_RATIO_SOFT_MIN = 0.50
from edm_matcher import GRID_H, GRID_W, EDMMatcher
from reloc_localizer_edm import Camera, EDMLocalizer, EDMRelocMap
from reposed_motion_validator import RelativeMotionCheck, cam_from_world_matrix
from edm_pose_selection import (
    acquire_consensus_limit,
    candidate_inliers,
    count_agreeing_refs,
    _select_acquire_candidate,
    _select_track_candidate,
    _strong_centers_disagree,
)


GLOBAL_RETRIEVAL_ATTEMPTS = 2
LOST_PROGRESSIVE_RADIUS_FACTORS = (1.0, 2.0, 4.0)
LOST_GLOBAL_RETRIEVAL_ATTEMPTS = len(LOST_PROGRESSIVE_RADIUS_FACTORS) + 1
LOST_PROGRESSIVE_STEP_INTERVAL = 2
GLOBAL_RETRIEVAL_RETRY_MULTIPLIER = 2
LOST_PRIOR_STRATEGIES = ("restrict_nearby", "full_global", "score_fusion")
ACQUIRE_STAGE_MODES = ("full_set", "initial_topk", "progressive")
ACQUIRE_PROGRESSIVE_TOPKS = (2, 4, 8, 20)
EDM_OPTIONAL_RUNTIME_TRACKER_KEYS = {
    "lost_prior_strategy",
    "lost_prior_fusion_weight",
    "acquire_stage_mode",
    "track_map_first",
    "pose_consensus_mode",
    "consensus_max_rotation_deg",
    "reference_quality_weight",
    "reference_quality_floor",
    "pnp_workers",
    "pnp_early_stop",
    "pnp_pipeline",
    "pnp_ranked_batches",
    "acquire_relaxed_min_inliers",
    "acquire_relaxed_max_reproj_error",
    "acquire_relaxed_min_agreeing_refs",
    "lost_starved_global_frames",
    "acquire_relaxed_probation_frames",
    "boot_relaxed_min_inliers",
    "track_miss_widen_topk",
    "weak_miss_widen_topk",
    "lost_starved_corr_max",
}
POSE_CONSENSUS_MODES = ("pairwise", "cluster")
PNP_RANSAC_SEED = 0
# Visual pose statuses: EDM-matched fixes are VISUALLY_CONFIRMED; KLT-bridged
# frames (no EDM match, optical-flow carry) are KLT_BRIDGED so downstream
# gates can tell them apart. Both flip to NONE on rejection.
_VISUAL_POSE_STATUSES = ("VISUALLY_CONFIRMED", "KLT_BRIDGED")
_VELOCITY_EMA_TAU_S = 0.25
_ACQUIRE_SINGLE_STRONG_INLIERS = 120
_ACQUIRE_SINGLE_STRONG_REPROJ = 2.0
_PNP_ELIGIBLE_RELAX_RATIO = 0.8
_PNP_ELIGIBLE_MIN_FLOOR = 10
_WEAK_HYSTERESIS_INLIER_MIN = 45
_WEAK_HYSTERESIS_REPROJ_MAX = 2.5
_VPR_BLUR_VAR_MIN: float = 50.0  # Initial Laplacian variance threshold for blur gating; frames below this skip VPR retry. To be tuned on P168 difficult frame holdouts.
PNP_RANSAC_THREADS_DEFAULT: int = 1


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _compute_laplacian_var(img: np.ndarray) -> float:
    import cv2
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pnp_eligible_threshold(min_inliers: int) -> int:
    """Return the minimum correspondence count required for a ref to be eligible for PnP.

    Relaxation allows refs with >= max(10, ceil(0.8 * min_inliers)) correspondences
    to attempt PnP, complementing the legacy rescue mechanism.
    Can be reverted to legacy min_inliers via SFM_EDM_PNP_ELIGIBLE_RELAX=0.
    """
    if (
        os.environ.get("SFM_EDM_PNP_ELIGIBLE_RELAX", "1").strip().lower()
        in ("0", "false", "no", "off")
    ):
        return int(min_inliers)
    return max(_PNP_ELIGIBLE_MIN_FLOOR, math.ceil(_PNP_ELIGIBLE_RELAX_RATIO * float(min_inliers)))


def _blend_velocity(
    old: np.ndarray | None, measured: np.ndarray, dt: float, tau: float = _VELOCITY_EMA_TAU_S
) -> np.ndarray:
    """dt-weighted velocity EMA: trust the new sample after long gaps.

    alpha = clamp(dt / tau): 30 fps frames smooth (alpha ~0.13), a 0.5 s gap
    almost fully trusts the new measurement. Non-finite dt falls back to the
    measurement. Replaces the fixed 0.5 average that biased slow-frame
    segments toward stale velocity.
    """
    measured_arr = np.asarray(measured, dtype=float)
    if old is None:
        return measured_arr
    alpha = 1.0 if not math.isfinite(dt) or dt <= 0 else min(1.0, dt / tau)
    return (1.0 - alpha) * np.asarray(old, dtype=float) + alpha * measured_arr



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
    "pnp_workers",
    "stale_reacquire_confirmations",
)

_NONNEGATIVE_INTEGER_CONFIG_FIELDS = (
    "lost_local_grace_frames",
    "lost_global_retrieval_interval",
    "recovery_bank_size",
    "recovery_scan_topk",
    "max_corr_total",
    "acquire_relaxed_min_inliers",
    "acquire_relaxed_min_agreeing_refs",
    "lost_starved_global_frames",
    "boot_relaxed_min_inliers",
    "track_miss_widen_topk",
    "weak_miss_widen_topk",
    "lost_starved_corr_max",
    "acquire_relaxed_probation_frames",
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
    "track_yaw_slack_deg",
    "track_max_yaw_rate_deg_s",
    "track_max_yaw_step_deg",
    "stale_reacquire_max_distance",
    "stale_reacquire_max_yaw_diff_deg",
    "lost_prior_fusion_weight",
    "consensus_max_rotation_deg",
    "acquire_relaxed_max_reproj_error",
)


@dataclass
class EDMConfig:
    # acquisition (BOOT_INIT / LOST): MegaLoc retrieval
    boot_global_topk: int = 10
    acquire_initial_topk: int = 2
    acquire_min_inliers: int = 80
    # Second-tier LOST re-acquisition floor for correspondence-rich near-misses.
    # The 8-video failure census has 92 LOST frames whose retrieval produced a
    # ~300-correspondence match and 60-79 inliers -- rejected only by the
    # acquire_min_inliers floor (rejected=None in every row). This floor accepts
    # that band ONLY with extra evidence: a tighter reprojection limit than the
    # standard acquire limit plus independent references agreeing on the same
    # camera center. Trajectory gates (acquire_jump / acquire_yaw / stale
    # two-frame confirmation) are untouched, which is what blocks the P119
    # vegetation false locks (those score 80-183 inliers, i.e. above this band).
    # 0 = off (single acquire threshold).
    #
    # Default 66 since 2026-09-05, on the seven-video 720p corpus rather than
    # the two- and four-segment sequential runs that had this at 0:
    #   corpus  5025/6231 (80.6%) -> 5256/6287 (83.6%), +231 frames
    #   P116 +102, P119 +96, P157 +28, P167 +25, P117/P118 flat, P168 -21
    #   P168 across seeds 0/1/2: -21 / -36 / -21 -- a real but bounded cost,
    #     and the only segment that pays one
    #   off-map hard negative (river video vs the urai map): 0/241 with the
    #     floor on AND off, so it opens no false lock
    #   quality: inliers p05 unchanged at 51, p50 77->74; reproj p50
    #     1.84->1.93 px, p95 2.80->2.77 px
    # Code default, not pinned into the site profile: pinning rotates the
    # profile SHA and the whole manifest/mission chain, which is a release
    # action. Set 0 to restore the single acquire threshold.
    acquire_relaxed_min_inliers: int = 66
    acquire_relaxed_max_reproj_error: float = 3.0
    acquire_relaxed_min_agreeing_refs: int = 2
    # Relaxed accept probation: a frame accepted on the relaxed floor re-enters
    # WEAK_TRACK (weak_local_topk references, weak_min_inliers) for this many
    # frames instead of TRACK (one reference). The 2026-09-04 gate showed the
    # P168 loss came from exactly this: the three relaxed accepts were correct
    # (0.03-0.23 m from the previous fix, 64/1/101 following successes) but
    # single-reference TRACK could not hold them, so inliers p50 fell 82->68
    # and WEAK_TRACK rose by 47. One frame of probation is provably not enough
    # (WEAK exits on the first good frame), hence a frame count. 0 = off.
    acquire_relaxed_probation_frames: int = 0
    # BOOT has no prior, so acquire_jump / acquire_yaw are not armed and the LOST
    # relaxed floor deliberately excludes it. The 8-video census still has 96 BOOT
    # failures with n_corr p50 652 and inliers 52-79 (max 79) -- the same
    # near-miss shape. This floor accepts that band at BOOT only after an
    # independent second frame lands within stale_reacquire_max_distance /
    # stale_reacquire_max_yaw_diff_deg of the first, i.e. two-frame confirmation
    # replaces the missing prior. 0 = off.
    boot_relaxed_min_inliers: int = 0
    # BOOT tries MegaLoc at boot_global_topk, then once at twice that count.
    # Each LOST episode first spends lost_local_grace_frames on nearby EDM;
    # a still-failed episode fires MegaLoc at lost_local_topk, then only retries
    # when lost_global_retrieval_interval is positive. LOW/WEAK remains geometry-only.
    global_retrieval_policy: str = "boot_and_lost_once"
    lost_local_topk: int = 5
    lost_local_grace_frames: int = 2
    # Retry MegaLoc every interval-th LOST frame after the first grace-window shot.
    # A zero interval restores the legacy one-shot LOST retrieval. Default 3 from the
    # 2026-09-02 river 1045-ref gate: P168 536->608 (+72), P117 366->366 (0), TRACK
    # path untouched (docs/verified_localization_optimization_ledger.md item 12). The
    # peak is map-dependent and non-monotonic (1/2/4/8 -> +14/+47/+47/+47); a new map
    # must re-run the interval sweep, this is only a better starting point than 0.
    lost_global_retrieval_interval: int = 3
    # Starvation-triggered full-map retry. LOST local recovery draws refs from the
    # last-known pose; inside a valley that pose is wrong, so EDM returns ~2
    # correspondences frame after frame (8-video census: 163 LOST frames with
    # n_corr<=5, 104 of them P119, all candidate_mode=edm_local_recovery with 2
    # refs). After this many consecutive starved LOST frames the next frame skips
    # the local/progressive-radius pool and runs a full-map MegaLoc retrieval.
    # 0 = off. This is starvation-triggered on purpose: the interval sweep is
    # non-monotonic (interval 1 is worse than 3), so "retrieve more often" is not
    # the lever -- "retrieve when the local pool produced nothing" is.
    lost_starved_global_frames: int = 0
    lost_starved_corr_max: int = 5
    recovery_bank_size: int = 192
    recovery_scan_topk: int = 2
    match_batch_size: int = 2
    # Temporal reference disabled by default: s3_no_temporal 520/700 (+34 vs s0 486/700),
    # LOST 168->113, p95 91.32ms, p50 35.18ms on P168 700-frame holdout (RTX 5060, 2026-09-01,
    # outputs/p168_all8_baseline_20260901/s3_no_temporal.json). Ledger evidence shows
    # temporal on is slower and less successful for this map; keep default False.
    use_temporal_reference: bool = False
    temporal_map_topk: int = 1
    track_map_first: bool = False
    local_topk: int = 1
    weak_local_topk: int = 3
    # Same-frame reference widening after a TRACK miss. TRACK runs one map
    # reference (local_topk=1, plus the temporal anchor), and the 8-video census
    # shows 212 failures -- 19.5% of all failures -- that are exactly this:
    # state TRACK, refs=1, inliers p50 34, n_corr p50 48. Retrying the SAME frame
    # against this many references costs nothing on healthy frames (it only runs
    # after a miss) and gives the geometry a second chance before the miss
    # counter advances toward WEAK/LOST. 0 = off.
    track_miss_widen_topk: int = 0
    # Same-frame widening after a WEAK miss: WEAK runs weak_local_topk=3 at
    # weak_min_inliers=30, and the census shows 83 WEAK failures at inliers p50
    # 23 with 33 frames in [25, 30). Same miss-only cost profile as above.
    # 0 = off.
    weak_miss_widen_topk: int = 0
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
    pnp_workers: int = 2
    pnp_early_stop: bool = True
    pnp_pipeline: bool = False
    pnp_ranked_batches: bool = False
    max_jump: float = 2.0
    prediction_max_dt: float = 0.25
    adaptive_jump_factor: float = 8.0
    adaptive_jump_floor: float = 0.003
    adaptive_jump_bootstrap: float = 0.02
    adaptive_jump_ceiling: float = 0.008
    adaptive_jump_min_history: int = 20
    adaptive_jump_history_size: int = 120
    track_yaw_slack_deg: float = 180.0
    track_max_yaw_rate_deg_s: float = 1.0
    track_max_yaw_step_deg: float = 180.0
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
    stale_reacquire_confirmations: int = 1
    stale_reacquire_max_distance: float = 0.3
    stale_reacquire_max_yaw_diff_deg: float = 30.0
    # Default preserves today's nearby restriction. full_global and score_fusion
    # are measured A/B modes and must be selected explicitly.
    lost_prior_strategy: str = "restrict_nearby"
    lost_prior_fusion_weight: float = 1.0
    # full_set and initial_topk preserve the prior modes. progressive evaluates
    # cumulative 2, 4, 8, and 20-reference stages, capped by the retrieved set.
    acquire_stage_mode: str = "full_set"
    pose_consensus_mode: str = "pairwise"
    consensus_max_rotation_deg: float = 90.0
    # P95 tail optimized: a2_quality 508/700 p95 89.87ms (lowest tail of all runs,
    # vs s0 486/700 p95 109.56ms), LOST 168->134, p50 24.25ms. Default 0.5 from
    # evidence outputs/p168_all8_baseline_20260901/a2_quality.json (SFM_EDM_REFERENCE_QUALITY_WEIGHT=0.5).
    # Env override SFM_EDM_REFERENCE_QUALITY_WEIGHT retained in ProductionEDMTracker.__init__ for
    # profile-hashed releases (river compat json unchanged to keep SHA 93e0c2...).
    reference_quality_weight: float = 0.5
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
            raise ValueError(
                "acquire_stage_mode must be 'full_set', 'initial_topk', or 'progressive'"
            )
        if self.pose_consensus_mode not in POSE_CONSENSUS_MODES:
            raise ValueError("pose_consensus_mode must be 'pairwise' or 'cluster'")
        _validate_integer_config_fields(self)
        if not isinstance(self.use_temporal_reference, bool):
            raise ValueError("use_temporal_reference must be boolean")
        if not isinstance(self.track_map_first, bool):
            raise ValueError("track_map_first must be boolean")
        if not isinstance(self.pnp_early_stop, bool):
            raise ValueError("pnp_early_stop must be boolean")
        if not isinstance(self.pnp_pipeline, bool):
            raise ValueError("pnp_pipeline must be boolean")
        if not isinstance(self.pnp_ranked_batches, bool):
            raise ValueError("pnp_ranked_batches must be boolean")
        _validate_float_config_fields(self)
        _validate_acquisition_config(self)
        _validate_jump_config(self)


def register_optional_runtime_tracker_keys() -> None:
    from edm_profile import EDM_OPTIONAL_TRACKER_KEYS

    EDM_OPTIONAL_TRACKER_KEYS.update(EDM_OPTIONAL_RUNTIME_TRACKER_KEYS)


register_optional_runtime_tracker_keys()


# Recovery-policy knobs the hashed release profiles do not carry. The site
# profile is SHA-pinned, and production-path replay only transmits the profile,
# so an A/B of these has to travel by environment variable (same reason
# SFM_EDM_REFERENCE_QUALITY_WEIGHT exists). Unset means "use the cfg value".
def _env_flag(raw: str) -> bool:
    """Parse an environment override for a boolean config field."""
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean flag, got {raw!r}")


_ENV_CONFIG_OVERRIDES = (
    ("SFM_EDM_ACQUIRE_RELAXED_MIN_INLIERS", "acquire_relaxed_min_inliers", int),
    ("SFM_EDM_ACQUIRE_RELAXED_REPROJ", "acquire_relaxed_max_reproj_error", float),
    ("SFM_EDM_ACQUIRE_RELAXED_AGREE", "acquire_relaxed_min_agreeing_refs", int),
    ("SFM_EDM_LOST_STARVED_FRAMES", "lost_starved_global_frames", int),
    ("SFM_EDM_LOST_STARVED_CORR", "lost_starved_corr_max", int),
    ("SFM_EDM_TRACK_WIDEN_TOPK", "track_miss_widen_topk", int),
    ("SFM_EDM_WEAK_WIDEN_TOPK", "weak_miss_widen_topk", int),
    ("SFM_EDM_BOOT_RELAXED_MIN_INLIERS", "boot_relaxed_min_inliers", int),
    # p95 lever: the wall-time tail is entirely LOST recovery frames matching
    # lost_local_topk references (P168/P119/P157 tail p95 frames: 36-50 of ~50
    # have 5 refs, match_ms p50 95-100 ms vs 22 ms overall, ~19 ms per ref).
    ("SFM_EDM_LOST_LOCAL_TOPK", "lost_local_topk", int),
    (
        "SFM_EDM_ACQUIRE_RELAXED_PROBATION_FRAMES",
        "acquire_relaxed_probation_frames",
        int,
    ),
)


def _apply_env_config_overrides(config: EDMConfig) -> None:
    """Apply environment overrides for the recovery-policy knobs, then re-validate."""
    applied = False
    for env_name, field_name, caster in _ENV_CONFIG_OVERRIDES:
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        try:
            value = caster(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{env_name} must be parseable as {caster.__name__}") from exc
        setattr(config, field_name, value)
        applied = True
    if applied:
        config.validate()


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
    if config.stale_reacquire_confirmations not in {1, 2}:
        raise ValueError("stale_reacquire_confirmations must be 1 or 2")
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
    misses: float = 0.0
    frame: int = 0
    accepted_step_norms: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    observed_capture_dts: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    last_observed_capture_stamp: float | None = None
    pending_limited_center: np.ndarray | None = None
    pending_limited_stamp: float | None = None
    pending_limited_limit: float | None = None
    pending_reacquire_center: np.ndarray | None = None
    pending_reacquire_yaw: float | None = None
    pending_reacquire_stamp: float | None = None
    boot_refs: list[str] = field(default_factory=list)
    global_retrieval_calls: int = 0
    boot_global_retrieval_attempts: int = 0
    lost_global_retrieval_attempts: int = 0
    lost_global_retrieval_done: bool = False
    lost_frames: int = 0
    # Consecutive LOST frames whose local recovery returned almost no
    # correspondences; arms the starvation-triggered full-map retrieval.
    lost_starved_frames: int = 0
    # Remaining frames of WEAK_TRACK probation after a relaxed-floor accept.
    relaxed_probation_frames: int = 0
    recovery_cursor: int = 0
    _vpr_consec_failures: int = 0
    _last_vpr_lost_frame: int | None = None


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
    pnp_candidates: int = 0
    pnp_skipped: int = 0
    # References agreeing with the chosen pose at the relaxed floor. Only
    # computed when the relaxed acquire tier is enabled and the chosen
    # candidate falls below acquire_min_inliers; 0 otherwise.
    relaxed_agree_count: int = 0


def _merge_correspondence_rows(rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    points2d_parts = [row[0] for row in rows if len(row[0])]
    points3d_parts = [row[1] for row in rows if len(row[1])]
    confidence_parts = [row[2] for row in rows if len(row[2])]
    points2d = np.concatenate(points2d_parts) if points2d_parts else np.zeros((0, 2))
    points3d = np.concatenate(points3d_parts) if points3d_parts else np.zeros((0, 3))
    confidence = np.concatenate(confidence_parts) if confidence_parts else np.zeros(0)
    return points2d, points3d, confidence, [row[3] for row in rows]


def _combine_correspondence_batches(
    first: _CorrespondenceBatch,
    second: _CorrespondenceBatch,
) -> _CorrespondenceBatch:
    if first.by_ref is None or second.by_ref is None:
        raise ValueError("only independent-reference correspondence batches can be reused")
    rows = [*first.by_ref, *second.by_ref]
    points2d, points3d, confidence, per_ref = _merge_correspondence_rows(rows)
    return _CorrespondenceBatch(
        points2d=points2d,
        points3d=points3d,
        confidence=confidence,
        per_ref=per_ref,
        by_ref=rows,
        temporal_used=False,
    )


def _reference_pnp_rank(
    row,
    *,
    width: int,
    height: int,
    grid: int,
) -> tuple[int, int, float]:
    points2d, points3d, confidence, _count = row
    spread = 0
    if len(points2d):
        points = np.asarray(points2d, dtype=float)
        gx = np.clip((points[:, 0] / width * grid).astype(int), 0, grid - 1)
        gy = np.clip((points[:, 1] / height * grid).astype(int), 0, grid - 1)
        spread = int(len(np.unique(gy * grid + gx)))
    finite_confidence = np.asarray(confidence, dtype=float)
    finite_confidence = finite_confidence[np.isfinite(finite_confidence)]
    mean_confidence = float(finite_confidence.mean()) if len(finite_confidence) else -math.inf
    return len(points3d), spread, mean_confidence


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


def track_yaw_limit(cfg: EDMConfig, capture_dt: float | None) -> float:
    """Return a capture-time-aware TRACK yaw envelope in degrees."""
    dt = 0.0
    if capture_dt is not None and math.isfinite(float(capture_dt)):
        dt = min(max(0.0, float(capture_dt)), float(cfg.prediction_max_dt))
    return min(
        float(cfg.track_max_yaw_step_deg),
        float(cfg.track_yaw_slack_deg) + float(cfg.track_max_yaw_rate_deg_s) * dt,
    )


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


def _anisotropic_corr_grid(width: int, height: int, grid: int) -> tuple[int, int]:
    """Grid dimensions for the anisotropic correspondence cap.

    Wide images get more horizontal than vertical cells. 16:9 frames use the
    canonical 16x9 layout; other aspect ratios derive gw, gh so gw/gh
    approximates width/height while gw*gh stays close to grid*grid.
    """
    if width <= 0 or height <= 0:
        return grid, grid
    aspect = float(width) / float(height)
    if abs(aspect - 16.0 / 9.0) <= 0.02 * 16.0 / 9.0:
        return 16, 9
    target = float(grid) * float(grid)
    gw = max(2, int(round(math.sqrt(target * aspect))))
    gh = max(2, int(round(math.sqrt(target / aspect))))
    return gw, gh


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
    if os.environ.get("SFM_EDM_ANISO_CORR_GRID") == "1":
        gw, gh = _anisotropic_corr_grid(width, height, grid)
    else:
        gw, gh = grid, grid
    gx = np.clip((pts2d[:, 0] / width * gw).astype(int), 0, gw - 1)
    gy = np.clip((pts2d[:, 1] / height * gh).astype(int), 0, gh - 1)
    cell = gy * gw + gx
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
        _apply_env_config_overrides(self.cfg)
        quality_weight_env = os.environ.get("SFM_EDM_REFERENCE_QUALITY_WEIGHT")
        if quality_weight_env is not None and hasattr(self.cfg, "reference_quality_weight"):
            try:
                quality_weight = float(quality_weight_env)
            except ValueError:
                quality_weight = None
            if quality_weight is not None and math.isfinite(quality_weight) and quality_weight >= 0.0:
                try:
                    self.cfg.reference_quality_weight = quality_weight
                except AttributeError:
                    pass
        # P95 tail optimized default 0.5 via tracker fallback for hashed river profile
        # (SHA 93e0c2... unchanged; evidence outputs/p168_all8_baseline_20260901/a2_quality.json
        # p95 89.87ms vs s0 109.56ms). Env SFM_EDM_REFERENCE_QUALITY_WEIGHT still overrides.
        if hasattr(self.cfg, "reference_quality_weight"):
            try:
                self._reference_quality_weight = float(self.cfg.reference_quality_weight)
            except (TypeError, ValueError):
                self._reference_quality_weight = 0.5
        else:
            self._reference_quality_weight = 0.5
        if quality_weight_env is not None:
            try:
                _qw = float(quality_weight_env)
                if math.isfinite(_qw) and _qw >= 0.0:
                    self._reference_quality_weight = _qw
                    if hasattr(self.cfg, "reference_quality_weight"):
                        try:
                            self.cfg.reference_quality_weight = _qw
                        except AttributeError:
                            pass
            except ValueError:
                pass
        self.cam = camera
        self.map = reloc_map
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
        self._klt_gray: np.ndarray | None = None
        self._klt_2d: np.ndarray | None = None
        self._klt_3d: np.ndarray | None = None
        self._klt_center: np.ndarray | None = None
        self._klt_yaw: float | None = None
        self._klt_seed_stamp: float | None = None
        self._klt_age: int = 0
        self._klt_query_gray: np.ndarray | None = None
        self._klt_query_stamp: float | None = None
        # SFM_EDM_KLT_BRIDGE_INTERVAL=N (>=2): in steady TRACK run the full EDM
        # match only every Nth frame and carry the pose with KLT optical flow on
        # the last EDM anchor's inlier 2D<->3D in between. Code default 0 = OFF
        # (see docs/verified_localization_optimization_ledger.md "KLT bridge").
        # It shipped as 3 on 2026-09-03 for the sequential-replay win (+89
        # successes / -121 LOST over 7 videos); the 2026-09-03 production-path
        # re-measurement reverted it: 4 same-setting repeats put P168 700f at
        # 516-528 with the bridge vs a rock-steady 575 without it (-47..-59
        # successes), far worse than the -18 the original single run recorded.
        # P117 is a successes wash there but 3.3x lower wall p50, so N>=2 is
        # still the right setting for smooth footage where latency dominates.
        try:
            self._klt_bridge_interval: int = int(
                os.environ.get("SFM_EDM_KLT_BRIDGE_INTERVAL", "0") or "0"
            )
        except ValueError:
            self._klt_bridge_interval = 0
        try:
            self._klt_bridge_min_ratio: float = float(
                os.environ.get("SFM_EDM_KLT_BRIDGE_MIN_RATIO", "0.66") or "0.66"
            )
        except ValueError:
            self._klt_bridge_min_ratio = 0.66
        # p95-tail guard: cap on consecutive bridge frames (0 = use N-1), and a
        # drift trip that forces a full EDM re-anchor the frame after a bridge
        # whose reproj / step / inlier-ratio is elevated (soft, below the hard
        # gates in _track_klt_prior). Both cut the worst-case drift before a
        # WEAK/LOST that would otherwise land a heavy recovery match.
        try:
            self._klt_bridge_max_consec: int = int(
                os.environ.get("SFM_EDM_KLT_BRIDGE_MAX_CONSEC", "0") or "0"
            )
        except ValueError:
            self._klt_bridge_max_consec = 0
        self._klt_bridge_drift_guard: bool = (
            os.environ.get("SFM_EDM_KLT_BRIDGE_DRIFT_GUARD", "1") != "0"
        )
        # Forward-backward reprojection gate tau (px) for the bridge's KLT: a
        # track is kept only when ||p_t - LK(LK(p_t))|| < tau. Mirrors the xfeat
        # flow path's SFM_FLOW_FB_PX so tau can be A/B'd through the replay gate.
        try:
            self._klt_fb_px: float = float(
                os.environ.get("SFM_EDM_KLT_FB_PX", str(_KLT_FB_PX)) or _KLT_FB_PX
            )
        except ValueError:
            self._klt_fb_px = float(_KLT_FB_PX)
        # LOST-side prediction. Both prediction branches (ESEKF and KLT) are
        # gated to TRACK/WEAK_TRACK, so a LOST frame currently reports no pose
        # at all. Opting in lets the KLT chain keep re-solving PnP against its
        # fixed 3D anchor set while LOST, which is reporting only: the frame
        # still carries ok=False / pose_status=PREDICTED_ONLY and never becomes
        # a visual anchor. Measured coverage is the reason this is off by
        # default -- see docs/klt_lost_prediction_experiment.md.
        self._klt_lost_predict: bool = (
            os.environ.get("SFM_EDM_KLT_LOST_PREDICT", "0") == "1"
        )
        # Horizon of the KLT chain. The defaults (6 frames / 0.50 s) predate the
        # LOST question and cover only ~4.5% of corpus LOST frames at the
        # measured 8 Hz cadence; they are overridable so the horizon can be
        # swept against drift rather than assumed.
        try:
            self._klt_max_frames: int = int(
                os.environ.get("SFM_EDM_KLT_MAX_FRAMES", str(_KLT_MAX_FRAMES))
                or _KLT_MAX_FRAMES
            )
        except ValueError:
            self._klt_max_frames = int(_KLT_MAX_FRAMES)
        try:
            self._klt_max_age_s: float = float(
                os.environ.get("SFM_EDM_KLT_MAX_AGE_S", str(_KLT_MAX_AGE_S))
                or _KLT_MAX_AGE_S
            )
        except ValueError:
            self._klt_max_age_s = float(_KLT_MAX_AGE_S)
        # Shadow evaluation: run the KLT chain on every frame purely to record
        # what it would have predicted, then restore the production cache byte
        # for byte. Used by benchmark_klt_lost_prediction.py to score the
        # prediction against the EDM fix on the same frame -- the only frames
        # where a ground truth exists. Never enabled in flight.
        self._klt_shadow_eval: bool = (
            os.environ.get("SFM_EDM_KLT_SHADOW_EVAL", "0") == "1"
        )
        self._shadow_klt: dict | None = None
        self._shadow_klt_anchor_frames: int = 0
        self._klt_bridge_run: int = 0
        self._pcam: pycolmap.Camera | None = None
        self._pnp_options: pycolmap.AbsolutePoseEstimationOptions | None = None
        self._pnp_executor = None
        self._last_lost_search_stage = None
        self._last_lost_radius_factor = None
        self.pose_guided = None
        if motion_validation_mode not in {"off", "shadow", "confirm_limited_jump"}:
            raise ValueError("motion_validation_mode must be off, shadow, or confirm_limited_jump")
        self.motion_validator = motion_validator
        self.motion_validation_mode = motion_validation_mode
        self._last_accepted_bgr: np.ndarray | None = None
        self._last_accepted_cam_from_world: np.ndarray | None = None
        self._last_accepted_rigid3d: pycolmap.Rigid3d | None = None
        # --- ESEKF minimal integration (after _klt_* and pose_guided init) ---
        self._latest_velocity_ned: np.ndarray | None = None
        self._esekf_last_predict: np.ndarray | None = None
        self._init_esekf()
        count = len(reloc_map.ref_names)
        self._reference_quality = np.full(count, np.nan, dtype=np.float32)
        self._reference_stability = np.full(count, np.nan, dtype=np.float32)
        if os.environ.get("SFM_EDM_LOCALIZABILITY_JSON"):
            self._load_localizability_quality_seed()

    def _load_localizability_quality_seed(self) -> None:
        """Seed reference quality from a localizability weak-region JSON (A/B).

        SFM_EDM_LOCALIZABILITY_JSON points at a JSON list of spheres, or a
        dict with a "spheres" list, each sphere {center: [x, y, z],
        radius: r, status: s}. References inside any
        red_intrinsic sphere score 0.2; every other reference scores 1.0.
        Unset env or an unreadable file keeps the default NaN qualities.
        """
        centers = self.centers
        if centers is None or not len(centers):
            return
        try:
            with open(os.environ["SFM_EDM_LOCALIZABILITY_JSON"], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
        if isinstance(payload, dict):
            spheres = payload.get("spheres")
        else:
            spheres = payload
        if not isinstance(spheres, list):
            return
        red_spheres: list[tuple[np.ndarray, float]] = []
        for sphere in spheres:
            if not isinstance(sphere, dict) or sphere.get("status") != "red_intrinsic":
                continue
            try:
                center = np.asarray(sphere["center"], dtype=float)
                radius = float(sphere["radius"])
            except (KeyError, TypeError, ValueError):
                continue
            if center.shape != (3,) or not math.isfinite(radius) or radius < 0.0:
                continue
            red_spheres.append((center, radius))
        if not red_spheres:
            return
        quality = np.ones(len(centers), dtype=np.float32)
        for center, radius in red_spheres:
            quality[np.linalg.norm(centers - center, axis=1) <= radius] = 0.2
        self._reference_quality = quality

    def _init_esekf(self) -> None:
        """Build ``self.esekf`` unless ``SFM_EDM_ESEKF_DISABLE=1``.

        The env toggle forces the ESEKF and the KLT 3D-aware prior (which is
        gated on ``self.esekf.prediction_allowed()``) fully off, so a live
        recording can be replayed both ways for A/B evaluation
        (``benchmark_esekf_live_replay.py``). Off by default.
        """
        self.esekf_disabled_by_env = os.environ.get("SFM_EDM_ESEKF_DISABLE") == "1"
        try:
            if self.esekf_disabled_by_env:
                self.esekf = None  # type: ignore
            elif ESEKF is not None and EKFConfig is not None:
                mpm = getattr(self.cfg, "metres_per_map_unit", None)
                if mpm is None:
                    pg = getattr(self, "pose_guided", None)
                    if pg is not None:
                        mpm_cfg = getattr(getattr(pg, "config", None), "metres_per_map_unit", None)
                        if mpm_cfg is not None:
                            mpm = mpm_cfg
                if mpm is not None:
                    try:
                        mpm_f = float(mpm)
                        if not math.isfinite(mpm_f) or mpm_f <= 1e-9:
                            mpm = None
                        else:
                            mpm = mpm_f
                    except Exception:
                        mpm = None
                if mpm is not None:
                    self.esekf = ESEKF(EKFConfig(metres_per_map_unit=mpm))  # type: ignore
                else:
                    self.esekf = ESEKF(EKFConfig())  # type: ignore
            else:
                self.esekf = None  # type: ignore
        except Exception:
            self.esekf = None  # type: ignore

    def attach_pose_guided(self, controller) -> None:
        self.pose_guided = controller

    def observe_fused_state(self, sample) -> None:
        controller = getattr(self, "pose_guided", None)
        if controller is not None:
            try:
                controller.observe_fused(sample)
            except Exception:
                pass
        # --- ESEKF velocity feed: store NED velocity for predict ---
        try:
            vel = None
            if sample is not None:
                vel = getattr(sample, "velocity_ned", None)
                if vel is None and isinstance(sample, dict):
                    vel = sample.get("velocity_ned", sample.get("velocity"))
                if vel is None:
                    # also try velocity attribute alias
                    vel = getattr(sample, "velocity", None)
            if vel is not None:
                arr = np.asarray(vel, dtype=float).reshape(3)
                if np.all(np.isfinite(arr)):
                    self._latest_velocity_ned = arr.astype(float)
                else:
                    self._latest_velocity_ned = None
        except Exception:
            pass

    def _active_pose_guided(self):
        controller = getattr(self, "pose_guided", None)
        if controller is None or not getattr(controller, "enabled", False):
            return None
        return controller

    def _pose_estimation_context(
        self,
    ) -> tuple[pycolmap.Camera, pycolmap.AbsolutePoseEstimationOptions]:
        if getattr(self, "_pcam", None) is None or getattr(self, "_pnp_options", None) is None:
            self._pcam, self._pnp_options = self._new_pose_estimation_context()
        return self._pcam, self._pnp_options

    def _pnp_ransac_threads(self) -> int:
        env_val = os.environ.get("SFM_EDM_PNP_THREADS", "1")
        try:
            val = int(env_val.strip())
            return max(1, val)
        except (ValueError, TypeError, AttributeError):
            return 1

    def _new_pose_estimation_context(
        self,
        random_seed: int = PNP_RANSAC_SEED,
    ) -> tuple[pycolmap.Camera, pycolmap.AbsolutePoseEstimationOptions]:
        camera = pycolmap.Camera(
            model=self.cam.model,
            width=self.cam.width,
            height=self.cam.height,
            params=self.cam.params,
        )
        options = pycolmap.AbsolutePoseEstimationOptions()
        options.ransac.max_error = self.cfg.pnp_ransac_max_error
        options.ransac.random_seed = int(random_seed)
        options.ransac.num_threads = self._pnp_ransac_threads()
        return camera, options

    @property
    def _vpr_consec_failures(self) -> int:
        if hasattr(self, "st") and self.st is not None:
            return getattr(self.st, "_vpr_consec_failures", 0)
        return getattr(self, "_vpr_consec_failures_val", 0)

    @_vpr_consec_failures.setter
    def _vpr_consec_failures(self, val: int) -> None:
        if hasattr(self, "st") and self.st is not None:
            self.st._vpr_consec_failures = int(val)
        self._vpr_consec_failures_val = int(val)

    @property
    def _last_vpr_lost_frame(self) -> int | None:
        if hasattr(self, "st") and self.st is not None:
            return getattr(self.st, "_last_vpr_lost_frame", None)
        return getattr(self, "_last_vpr_lost_frame_val", None)

    @_last_vpr_lost_frame.setter
    def _last_vpr_lost_frame(self, val: int | None) -> None:
        if hasattr(self, "st") and self.st is not None:
            self.st._last_vpr_lost_frame = val
        self._last_vpr_lost_frame_val = val

    def _adaptive_vpr_enabled(self) -> bool:
        return _env_bool("SFM_EDM_ADAPTIVE_VPR", False)

    def _prior_pnp_refine_enabled(self) -> bool:
        # Default ON (2026-09-03 gate: P168 608 + P117 366 zero-regression,
        # perturbation test +-0.3m/+-10deg holds; safe fallback to RANSAC).
        # Set SFM_EDM_PRIOR_PNP_REFINE=0 to restore pure-RANSAC PnP.
        return _env_bool("SFM_EDM_PRIOR_PNP_REFINE", True)

    def _is_vpr_blur_gated(self, frame_bgr: np.ndarray | None = None) -> bool:
        if not self._adaptive_vpr_enabled() or frame_bgr is None:
            return False
        try:
            var = _compute_laplacian_var(frame_bgr)
            return var < _VPR_BLUR_VAR_MIN
        except Exception:
            return False

    def _effective_lost_global_retrieval_interval(self) -> int:
        base = int(getattr(self.cfg, "lost_global_retrieval_interval", 3))
        if not self._adaptive_vpr_enabled() or base <= 0:
            return base
        consec = int(self._vpr_consec_failures)
        if consec < 2:
            return base
        elif consec == 2:
            return max(base, 5)
        else:
            return max(base, 8)

    def _lost_local_radius_factor(self) -> float:
        if not self._adaptive_vpr_enabled():
            return 1.0
        if int(self._vpr_consec_failures) >= 2:
            base_r = float(getattr(self.cfg, "radius", 0.8))
            max_j = float(getattr(self.cfg, "max_jump", base_r))
            if base_r > 1e-6:
                return min(2.0, max(1.0, max_j / base_r))
            return 2.0
        return 1.0

    def _resolve_pose_prior(self, prior: Any | None = None) -> pycolmap.Rigid3d | None:
        if prior is not None:
            if isinstance(prior, pycolmap.Rigid3d):
                return prior
            if isinstance(prior, dict) and "cam_from_world" in prior:
                val = prior["cam_from_world"]
                if isinstance(val, pycolmap.Rigid3d):
                    return val
                if hasattr(val, "rotation") and hasattr(val, "translation"):
                    return val
            if hasattr(prior, "cam_from_world"):
                val = getattr(prior, "cam_from_world")
                if isinstance(val, pycolmap.Rigid3d):
                    return val
        last_rigid = getattr(self, "_last_accepted_rigid3d", None)
        if last_rigid is not None and isinstance(last_rigid, pycolmap.Rigid3d):
            return last_rigid
        last_mat = getattr(self, "_last_accepted_cam_from_world", None)
        if last_mat is not None and hasattr(last_mat, "shape"):
            try:
                return pycolmap.Rigid3d(np.asarray(last_mat, dtype=float)[:3, :4])
            except Exception:
                pass
        return None

    def _parallel_pnp_executor(self) -> ThreadPoolExecutor | None:
        if self.cfg.pnp_workers <= 1:
            return None
        executor = getattr(self, "_pnp_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=self.cfg.pnp_workers,
                thread_name_prefix="edm-pnp",
            )
            self._pnp_executor = executor
        return executor

    def _pnp_random_seed(self) -> int:
        options = getattr(self, "_pnp_options", None)
        if options is None:
            return PNP_RANSAC_SEED
        return int(options.ransac.random_seed)

    def _klt_prior_states(self) -> frozenset[str]:
        """Tracker states the KLT chain may run in.

        LOST is opt-in: it is the state the chain was never allowed to serve,
        and turning it on is the whole subject of
        docs/klt_lost_prediction_experiment.md.
        """
        if getattr(self, "_klt_lost_predict", False) or getattr(self, "_klt_shadow_eval", False):
            return frozenset({"TRACK", "WEAK_TRACK", "LOST"})
        return frozenset({"TRACK", "WEAK_TRACK"})

    def _klt_prior_allowed(self, capture_stamp: float | None = None) -> bool:
        if getattr(self, "st", None) is None:
            return False
        if getattr(self.st, "state", None) not in self._klt_prior_states():
            return False
        klt_2d = getattr(self, "_klt_2d", None)
        klt_3d = getattr(self, "_klt_3d", None)
        klt_gray = getattr(self, "_klt_gray", None)
        if klt_2d is None or klt_3d is None or klt_gray is None:
            return False
        klt_age = getattr(self, "_klt_age", 0)
        try:
            if int(klt_age) >= int(getattr(self, "_klt_max_frames", _KLT_MAX_FRAMES)):
                return False
        except Exception:
            return False
        seed_stamp = getattr(self, "_klt_seed_stamp", None)
        if seed_stamp is None:
            return False
        if capture_stamp is None:
            capture_stamp = getattr(self, "_klt_query_stamp", None)
        try:
            dt = float(capture_stamp) - float(seed_stamp)  # type: ignore[arg-type]
        except Exception:
            return False
        if not math.isfinite(dt) or dt < 0.0 or dt > float(
            getattr(self, "_klt_max_age_s", _KLT_MAX_AGE_S)
        ):
            return False
        return True

    def _track_klt_prior(self, gray: np.ndarray | None, capture_stamp: float | None) -> dict | None:
        if gray is None or capture_stamp is None:
            return None
        if not self._klt_prior_allowed(capture_stamp):
            return None
        klt_gray = getattr(self, "_klt_gray", None)
        klt_2d = getattr(self, "_klt_2d", None)
        klt_3d = getattr(self, "_klt_3d", None)
        if klt_gray is None or klt_2d is None or klt_3d is None:
            return None
        if klt_2d.shape[0] < _KLT_MIN_TRACK:
            self._clear_klt_cache()
            return None
        try:
            # --- 3D-aware init + motion-guided window ---
            klt_params = _KLT_LK_PARAMS
            p0_hat = None
            try:
                esekf = getattr(self, "esekf", None)
                if esekf is not None and callable(getattr(esekf, "prediction_allowed", None)) and esekf.prediction_allowed():
                    # window from covariance trace 15->41
                    try:
                        P = esekf.get_covariance() if hasattr(esekf, "get_covariance") else None
                        if P is not None:
                            trace = float(np.trace(np.asarray(P)[0:3, 0:3]))
                        else:
                            trace = 0.0
                    except Exception:
                        trace = 0.0
                    try:
                        W = int(np.clip(_KLT_COV_WINDOW_MIN + _KLT_COV_WINDOW_GAIN * math.sqrt(max(trace, 0.0)) / _KLT_COV_WINDOW_SCALE, _KLT_COV_WINDOW_MIN, _KLT_COV_WINDOW_MAX))
                        if W % 2 == 0:
                            W += 1
                        klt_params = dict(_KLT_LK_PARAMS, winSize=(W, W))
                    except Exception:
                        klt_params = _KLT_LK_PARAMS
                    # 3D projection: hat p = pi(R_pred*(P - C_pred))
                    try:
                        state = esekf.get_state() if hasattr(esekf, "get_state") else None
                        if state is not None and hasattr(state, "p") and hasattr(state, "q"):
                            C_pred = np.asarray(state.p, dtype=float).reshape(3)
                            q_pred = np.asarray(state.q, dtype=float).reshape(4)
                            # quat to rot (wxyz)
                            w, x, y, z = q_pred
                            nrm = math.sqrt(w*w + x*x + y*y + z*z)
                            if nrm > 1e-12:
                                w, x, y, z = w/nrm, x/nrm, y/nrm, z/nrm
                                if w < 0:
                                    w, x, y, z = -w, -x, -y, -z
                                R_pred = np.array([
                                    [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                                    [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
                                    [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)],
                                ], dtype=float)
                            else:
                                R_pred = np.eye(3)
                            if hasattr(self, "cam") and hasattr(self.cam, "params"):
                                fx, fy, cx, cy = [float(v) for v in self.cam.params[:4]]
                                P3d = np.asarray(klt_3d, dtype=float)
                                diff = P3d - C_pred[None, :]
                                p_cam = (R_pred @ diff.T).T
                                p_hat = np.asarray(klt_2d, dtype=np.float32).copy()
                                valid = p_cam[:, 2] > 1e-3
                                if np.any(valid):
                                    p_hat[valid, 0] = (p_cam[valid, 0] / p_cam[valid, 2] * fx + cx).astype(np.float32)
                                    p_hat[valid, 1] = (p_cam[valid, 1] / p_cam[valid, 2] * fy + cy).astype(np.float32)
                                    # clamp to image (optional)
                                    # keep even if outside; LK will fail those points
                                    p0_hat = p_hat.reshape(-1, 1, 2).astype(np.float32)
                                    # use initial flow flag
                                    p0 = np.asarray(klt_2d, dtype=np.float32).reshape(-1, 1, 2)
                                    # 3D-aware LK result; the fallback below reuses nxt/stf when they are bound.
                                    nxt, stf, _ = cv2.calcOpticalFlowPyrLK(klt_gray, gray, p0, p0_hat, winSize=klt_params["winSize"], maxLevel=klt_params["maxLevel"], criteria=klt_params["criteria"], flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                    except Exception:
                        pass
            except Exception:
                pass
            # fallback: standard LK without initial guess
            p0 = np.asarray(klt_2d, dtype=np.float32).reshape(-1, 1, 2)
            # if we already computed nxt via 3D-aware path, reuse; else compute here
            backend = _FLOW_BACKEND
            try:
                nxt
            except NameError:
                if backend is None:
                    nxt, stf, _ = cv2.calcOpticalFlowPyrLK(klt_gray, gray, p0, None, **klt_params)
                else:
                    nxt, stf = backend(klt_gray, gray, p0)
            if nxt is None or stf is None:
                self._clear_klt_cache()
                return None
            if backend is None:
                back, stb, _ = cv2.calcOpticalFlowPyrLK(gray, klt_gray, nxt, None, **klt_params)
            else:
                back, stb = backend(gray, klt_gray, nxt)
            if back is None or stb is None:
                self._clear_klt_cache()
                return None
            fb = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
            fb_px = float(getattr(self, "_klt_fb_px", _KLT_FB_PX))
            good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < fb_px)
            if int(np.count_nonzero(good)) < _KLT_MIN_TRACK:
                self._clear_klt_cache()
                return None
            n2d = nxt.reshape(-1, 2)[good]
            n3d = np.asarray(klt_3d, dtype=float)[good]
            fbg = fb[good]
            # median drop: biased but plausible drifters
            try:
                median_fbg = float(np.median(fbg)) if len(fbg) else 0.0
            except Exception:
                median_fbg = 0.0
            drop = (fbg > 2.0 * (median_fbg + 1e-6)) & (fbg > 0.5)
            if bool(np.any(drop)) and int(np.count_nonzero(~drop)) >= _KLT_MIN_TRACK:
                s2d = n2d[~drop]
                s3d = n3d[~drop]
            else:
                s2d, s3d = n2d, n3d
            if s2d.shape[0] < 6:
                self._clear_klt_cache()
                return None
            pcam, options = self._new_pose_estimation_context(self._pnp_random_seed())
            candidate, _elapsed = self._estimate_pose_candidate(
                s2d, s3d, np.ones(len(s3d), dtype=float), pcam, options
            )
            if candidate is None:
                self._clear_klt_cache()
                return None
            result, _c2d, _c3d, metrics = candidate
            if result is None:
                self._clear_klt_cache()
                return None
            inliers = int(result.get("num_inliers", 0))
            if inliers < int(self.cfg.weak_min_inliers):
                self._clear_klt_cache()
                return None
            reproj_rms = metrics.get("reproj_rms")
            try:
                rms_val = float(reproj_rms) if reproj_rms is not None else float("nan")
            except Exception:
                rms_val = float("nan")
            if not math.isfinite(rms_val) or rms_val > float(self.cfg.max_reproj_error_track):
                self._clear_klt_cache()
                return None
            R_klt, center, yaw = self._pose_components(result)
            if getattr(self.st, "center", None) is not None:
                try:
                    step = float(np.linalg.norm(np.asarray(center, dtype=float) - np.asarray(self.st.center, dtype=float)))
                except Exception:
                    step = float("inf")
                if step > float(self.cfg.max_jump):
                    self._clear_klt_cache()
                    return None
            # geometric consistency confidence (not just N_track)
            try:
                tracked = int(np.count_nonzero(good))
                inlier_ratio = float(inliers) / max(1, len(s2d))
                # spatial coverage
                try:
                    h, w = gray.shape[:2]
                    gw, gh = _KLT_SPATIAL_GRID
                    cells = set()
                    for x, y in n2d:
                        gx = min(gw - 1, max(0, int(x / max(w, 1) * gw)))
                        gy = min(gh - 1, max(0, int(y / max(h, 1) * gh)))
                        cells.add((gx, gy))
                    spatial_cells = len(cells)
                except Exception:
                    spatial_cells = _KLT_SPATIAL_FALLBACK_CELLS
                try:
                    fb_med = float(np.median(fbg)) if len(fbg) else 0.0
                except Exception:
                    fb_med = 0.0
                confidence = _KLT_CONF_W_RATIO * float(inlier_ratio) + _KLT_CONF_W_SPATIAL * (float(spatial_cells) / _KLT_CONF_SPATIAL_NORM) + _KLT_CONF_W_FB * (1.0 - min(float(fb_med), 1.0))
                # 放宽：0.40→0.30, 0.60→0.50 （水面 ratio 0.5-0.6 原被误红）
                if inlier_ratio < _KLT_RATIO_HARD_MIN and tracked >= _KLT_TRACKED_MIN_FOR_RATIO_GATE:
                    self._clear_klt_cache()
                    return None
                if confidence < _KLT_CONF_MIN:
                    if inlier_ratio < _KLT_RATIO_SOFT_MIN:
                        self._clear_klt_cache()
                        return None
                # store for later info
                _conf = float(confidence)
                _spatial = int(spatial_cells)
                _ratio = float(inlier_ratio)
                _fb_med = float(fb_med)
            except Exception:
                _conf = 0.0
                _spatial = 0
                _ratio = 0.0
                _fb_med = 0.0
            self._klt_2d = n2d.copy()
            self._klt_3d = n3d.copy()
            self._klt_gray = np.ascontiguousarray(gray)
            try:
                self._klt_age = int(getattr(self, "_klt_age", 0)) + 1
            except Exception:
                self._klt_age = 1
            self._klt_center = np.asarray(center, dtype=float).copy()
            self._klt_yaw = float(yaw)
            return {
                "center": np.asarray(center, dtype=float).copy(),
                "yaw": float(yaw),
                "R": np.asarray(R_klt, dtype=float).copy(),
                "cam_from_world": result.get("cam_from_world"),
                "tracked": int(np.count_nonzero(good)),
                "inliers": int(inliers),
                "reproj_rms": float(rms_val),
                "confidence": float(_conf),
                "spatial_cells": int(_spatial),
                "inlier_ratio": float(_ratio),
                "fb_med": float(_fb_med),
            }
        except cv2.error:
            try:
                self._clear_klt_cache()
            except Exception:
                pass
            return None
        except Exception:
            try:
                self._clear_klt_cache()
            except Exception:
                pass
            return None

    # ---------- candidate selection ----------
    def _pending_reacquire_center(self) -> np.ndarray | None:
        """Anchor for the next LOST search after an unconfirmed stale reacquire.

        The accepted st.center predates the LOST episode, so predicting from it
        walks the next frame back into the same stale references that just failed.
        """
        if self.st.state != "LOST" or self.st.pending_reacquire_center is None:
            return None
        return np.asarray(self.st.pending_reacquire_center, dtype=float)

    def _search_yaw(self) -> float | None:
        """Yaw anchor for candidate filtering; an unconfirmed stale fix wins."""
        pending = self.st.pending_reacquire_yaw
        if self.st.state == "LOST" and pending is not None and math.isfinite(float(pending)):
            return float(pending)
        # EKF yaw supersedes KLT when prediction_allowed (cov trace<1.0, yaw sigma<15deg, age<6)
        try:
            ekf_yaw = self._esekf_yaw()
            if ekf_yaw is not None and self.st.state in {"TRACK", "WEAK_TRACK"} and math.isfinite(float(ekf_yaw)):
                return float(ekf_yaw)
        except Exception:
            pass
        klt_yaw = getattr(self, "_klt_yaw", None)
        if self.st.state in {"TRACK", "WEAK_TRACK"} and klt_yaw is not None and math.isfinite(float(klt_yaw)):
            return float(klt_yaw)
        return self.st.yaw

    def _predict_center(self, capture_stamp: float | None = None):
        pending = self._pending_reacquire_center()
        if pending is not None:
            return pending
        # EKF supersedes KLT when allowed (TRACK/WEAK_TRACK, cov trace<1.0, yaw sigma<15deg, age<6)
        try:
            if getattr(self.st, "state", None) in {"TRACK", "WEAK_TRACK"}:
                esekf = getattr(self, "esekf", None)
                if esekf is not None and callable(getattr(esekf, "prediction_allowed", None)) and esekf.prediction_allowed():
                    p_pred = self._esekf_predicted_center(capture_stamp)
                    if p_pred is not None and p_pred.shape == (3,) and np.all(np.isfinite(p_pred)):
                        return p_pred
        except Exception:
            pass
        klt_center = getattr(self, "_klt_center", None)
        if klt_center is not None:
            return np.asarray(klt_center, dtype=float)
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
            visual_center=None
            if self.st.center is None
            else np.asarray(self.st.center, dtype=float),
            visual_yaw=self.st.yaw,
            visual_velocity=self.st.velocity,
            visual_stamp=self.st.last_capture_stamp,
            allow_gnss_prior=self.st.state in {"WEAK", "WEAK_TRACK", "LOST"},
        )

        if prediction.valid and prediction.position is not None:
            return np.asarray(prediction.position, dtype=float)
        return visual

    def _near_reference_indices(
        self,
        distances: np.ndarray,
        finite: np.ndarray,
        *,
        radius_factor: float = 1.0,
        strict_radius: bool = False,
    ) -> list[int]:
        near = []
        radius = float(self.cfg.radius) * float(radius_factor)
        pool_limit = max(
            int(self.cfg.near_pool),
            int(math.ceil(float(self.cfg.near_pool) * float(radius_factor))),
        )
        max_yaw = math.radians(self.cfg.max_yaw_diff_deg)
        predicted_yaw = self._search_yaw()
        controller = self._active_pose_guided()
        if controller is not None:
            limits = controller.search_limits(
                state=self.st.state,
                misses=int(self.st.misses),
                weak_after=int(self.cfg.weak_after),
                base_radius=float(self.cfg.radius) * float(radius_factor),
                base_yaw_rad=math.radians(self.cfg.max_yaw_diff_deg),
                base_max_refs=pool_limit,
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
            if distances[index] > radius:
                if strict_radius or len(near) >= pool_limit:
                    break
            if (
                predicted_yaw is not None
                and self.yaws is not None
                and np.isfinite(self.yaws[index])
                and abs(_angle_diff(float(self.yaws[index]), predicted_yaw)) > max_yaw
            ):
                continue
            near.append(index)
            if len(near) >= pool_limit:
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
        yaw = self._search_yaw()
        score = 2.0 * math.exp(-float(distances[index]) / max(self.cfg.radius, 1e-6))
        if yaw is not None and self.yaws is not None and np.isfinite(self.yaws[index]):
            score += 0.8 * math.exp(
                -abs(_angle_diff(float(self.yaws[index]), yaw)) / math.radians(45.0)
            )
        else:
            score += 0.4
        score += 1.0 if index in last else 0.0
        weight = float(getattr(self, "_reference_quality_weight", getattr(self.cfg, "reference_quality_weight", 0.5)))
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

    def _track_candidates(
        self,
        topk: int,
        capture_stamp: float | None = None,
        *,
        radius_factor: float = 1.0,
        strict_radius: bool = False,
    ) -> list[str]:
        """Geometry only: nearest references to the predicted pose, plus their covisibles."""
        C = self._predict_center(capture_stamp)
        if C is None or self.centers is None:
            return []
        d = np.linalg.norm(self.centers - C[None, :], axis=1)
        controller = self._active_pose_guided()
        if controller is not None and controller.last_prediction is not None:
            quality_scores = None
            stability_scores = None
            if float(getattr(self, "_reference_quality_weight", getattr(self.cfg, "reference_quality_weight", 0.5))) > 0.0:
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
                base_radius=float(self.cfg.radius) * float(radius_factor),
                base_yaw_rad=math.radians(self.cfg.max_yaw_diff_deg),
                base_max_refs=int(topk),
                last_indices=list(self.st.last_refs),
                covisible_indices=self._covisible_reference_indices([]),
                quality_scores=quality_scores,
                stability_scores=stability_scores,
            )
            if selected:
                if strict_radius:
                    radius = float(self.cfg.radius) * float(radius_factor)
                    selected = [
                        name
                        for name in selected
                        if name in self.idx_of
                        and np.isfinite(d[self.idx_of[name]])
                        and d[self.idx_of[name]] <= radius
                    ]
                return selected[:topk]
        finite = np.isfinite(d)
        near = self._near_reference_indices(
            d,
            finite,
            radius_factor=radius_factor,
            strict_radius=strict_radius,
        )
        covis_refs = self._covisible_reference_indices(near)
        union = list(dict.fromkeys(near + covis_refs + list(self.st.last_refs)))
        last = set(self.st.last_refs)
        union = [i for i in union if 0 <= i < len(self.map.ref_names) and finite[i]]
        if strict_radius:
            radius = float(self.cfg.radius) * float(radius_factor)
            union = [index for index in union if d[index] <= radius]
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
        *,
        force_global: bool = False,
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
            else self.cfg.boot_global_topk * (GLOBAL_RETRIEVAL_RETRY_MULTIPLIER if attempts else 1)
        )
        candidates = None
        prior_fresh = False
        if lost:
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
            if force_global:
                # Starved: the nearby pool is what produced nothing, so the
                # progressive radius stages and the prior-fused ranking are
                # exactly the wrong place to look. Rank the whole map.
                self._last_lost_search_stage = "starved_global"
                self._last_lost_radius_factor = None
            elif self.cfg.lost_prior_strategy == "restrict_nearby":
                if attempts < len(LOST_PROGRESSIVE_RADIUS_FACTORS):
                    factor = LOST_PROGRESSIVE_RADIUS_FACTORS[attempts]
                    pool = min(
                        len(self.map.ref_names),
                        max(
                            int(self.cfg.lost_local_topk),
                            int(math.ceil(self.cfg.near_pool * factor)),
                        ),
                    )
                    candidates = self._track_candidates(
                        pool,
                        capture_stamp,
                        radius_factor=factor,
                        strict_radius=True,
                    )
                    self._last_lost_search_stage = f"near_{int(factor)}x"
                    self._last_lost_radius_factor = float(factor)
                else:
                    self._last_lost_search_stage = "global"
                    self._last_lost_radius_factor = None
            elif self.cfg.lost_prior_strategy == "full_global":
                self._last_lost_search_stage = "global"
                self._last_lost_radius_factor = None
        elif not lost:
            self._last_lost_search_stage = "boot_global"
            self._last_lost_radius_factor = None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        if lost and not force_global and self.cfg.lost_prior_strategy == "score_fusion":
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
                self.st.lost_global_retrieval_attempts >= LOST_GLOBAL_RETRIEVAL_ATTEMPTS
            )
            self._last_vpr_lost_frame = self.st.lost_frames
        else:
            self.st.boot_global_retrieval_attempts += 1
        if not lost and len(refs) > len(self.st.boot_refs):
            self.st.boot_refs = list(refs)
        return refs, elapsed_ms, candidates is not None

    def _lost_starvation_armed(self) -> bool:
        """True when consecutive corr-starved LOST frames have earned a full-map retry."""
        threshold = int(getattr(self.cfg, "lost_starved_global_frames", 0) or 0)
        if threshold <= 0 or self.st.state != "LOST":
            return False
        return int(getattr(self.st, "lost_starved_frames", 0) or 0) >= threshold

    def _observe_lost_starvation(
        self,
        state_in: str,
        selection: _CandidateSelection,
        info: dict,
    ) -> None:
        """Count corr-starved local-pool frames inside the current LOST episode.

        Local recovery refs come from the last-known pose. Inside a recovery
        valley that pose is wrong, so EDM returns a handful of correspondences
        and the next local scan repeats the same mistake (measured: 328
        edm_local_recovery frames across P119/P157/P168, 2 successes).

        Retrieval frames neither count nor reset: they exercise a different
        pool, and the interleaved MegaLoc cadence would otherwise clip every
        run of local frames to two and the escape hatch could never arm. Only a
        local frame that actually found geometry, or leaving LOST, clears it.
        """
        if int(getattr(self.cfg, "lost_starved_global_frames", 0) or 0) <= 0:
            return
        if state_in != "LOST":
            self.st.lost_starved_frames = 0
            return
        if selection.mode.startswith("megaloc"):
            return
        n_corr = int(info.get("n_corr", 0) or 0)
        if n_corr <= int(getattr(self.cfg, "lost_starved_corr_max", 0) or 0):
            self.st.lost_starved_frames += 1
        else:
            self.st.lost_starved_frames = 0
        info["lost_starved_frames"] = int(self.st.lost_starved_frames)

    def _should_run_global_retrieval(self, frame_bgr: np.ndarray | None = None) -> bool:
        if self.cfg.global_retrieval_policy == "boot_and_lost_once":
            if (
                self.st.state == "BOOT_INIT"
                and self.st.boot_global_retrieval_attempts < GLOBAL_RETRIEVAL_ATTEMPTS
            ):
                return True
            if self._lost_starvation_armed():
                return True
            progressive_offset = self.st.lost_frames - self.cfg.lost_local_grace_frames - 1
            if self._adaptive_vpr_enabled() and self.st.state == "LOST":
                if self.st.lost_frames <= self.cfg.lost_local_grace_frames:
                    return False
                if self.st.lost_global_retrieval_attempts == 0:
                    return True
                interval = self._effective_lost_global_retrieval_interval()
                if interval <= 0:
                    return False
                if self._last_vpr_lost_frame is not None:
                    timing_ok = (self.st.lost_frames - self._last_vpr_lost_frame) >= interval
                else:
                    timing_ok = (progressive_offset >= 0 and progressive_offset % interval == 0)
                if not timing_ok:
                    return False
                if self._is_vpr_blur_gated(frame_bgr):
                    return False
                return True
            if (
                self.st.state == "LOST"
                and self.cfg.lost_prior_strategy == "restrict_nearby"
                and self.cfg.lost_global_retrieval_interval > 0
                and self.st.lost_global_retrieval_attempts < LOST_GLOBAL_RETRIEVAL_ATTEMPTS
            ):
                return (
                    progressive_offset >= 0
                    and progressive_offset % LOST_PROGRESSIVE_STEP_INTERVAL == 0
                )
            return (
                self.st.state == "LOST"
                and self.st.lost_frames > self.cfg.lost_local_grace_frames
                and (
                    self.st.lost_global_retrieval_attempts == 0
                    or (
                        self.cfg.lost_global_retrieval_interval > 0
                        and (progressive_offset) % self.cfg.lost_global_retrieval_interval == 0
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
        if self.st.state == "LOST" and self.cfg.lost_prior_strategy == "restrict_nearby":
            attempts = int(self.st.lost_global_retrieval_attempts)
            factor = LOST_PROGRESSIVE_RADIUS_FACTORS[
                min(attempts, len(LOST_PROGRESSIVE_RADIUS_FACTORS) - 1)
            ]
            local_topk = (
                self.cfg.lost_local_topk
                if self.st.lost_frames <= self.cfg.lost_local_grace_frames
                else self.cfg.recovery_scan_topk
            )
            refs = self._track_candidates(
                local_topk,
                capture_stamp,
                radius_factor=factor,
                strict_radius=True,
            )
            self._last_lost_search_stage = f"near_{int(factor)}x"
            self._last_lost_radius_factor = float(factor)
            mode = "edm_local_recovery"
            if not refs and attempts >= len(LOST_PROGRESSIVE_RADIUS_FACTORS):
                refs = self._recovery_candidates(self.cfg.recovery_scan_topk)
                mode = "edm_map_scan"
        elif self.st.state == "LOST" and self.st.lost_frames > self.cfg.lost_local_grace_frames:
            if self._adaptive_vpr_enabled() and self.st.center is not None:
                refs = self._track_candidates(
                    self.cfg.lost_local_topk,
                    capture_stamp,
                    radius_factor=self._lost_local_radius_factor(),
                )
                if refs:
                    return refs, "edm_local_recovery"
            refs = self._recovery_candidates(self.cfg.recovery_scan_topk)
            mode = "edm_map_scan"
        elif self.st.state == "LOST":
            refs = self._track_candidates(
                self.cfg.lost_local_topk,
                capture_stamp,
                radius_factor=self._lost_local_radius_factor(),
            )
            mode = "edm_local_recovery"
        else:
            refs = list(self.st.boot_refs)
            mode = "edm_boot_refs"
        progressive_local_pending = (
            self.st.state == "LOST"
            and self.cfg.lost_prior_strategy == "restrict_nearby"
            and self.st.lost_global_retrieval_attempts < len(LOST_PROGRESSIVE_RADIUS_FACTORS)
        )
        if not refs and not progressive_local_pending:
            refs = list(self.st.boot_refs[: self.cfg.lost_local_topk])
            mode = "edm_boot_refs"
        return refs, mode

    def _acquisition_candidates(
        self,
        frame_bgr: np.ndarray,
        capture_stamp: float,
    ) -> _CandidateSelection:
        if self._should_run_global_retrieval(frame_bgr):
            retry = (
                self.st.lost_global_retrieval_attempts
                if self.st.state == "LOST"
                else self.st.boot_global_retrieval_attempts
            ) > 0
            starved = self._lost_starvation_armed()
            refs, vpr_ms, nearby = self._global_retrieval(
                frame_bgr,
                capture_stamp,
                force_global=starved,
            )
            if starved:
                # Counter is spent: the next full-map retry needs a fresh run of
                # starved frames, so one bad valley cannot pin VPR on every frame.
                self.st.lost_starved_frames = 0
                mode = "megaloc_lost_starved"
            elif self.st.state == "LOST" and getattr(self, "_last_retrieval_kind", None) == "fused":
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
        self._last_lost_search_stage = None
        self._last_lost_radius_factor = None
        if self.st.state in ("BOOT_INIT", "LOST"):
            return self._acquisition_candidates(frame_bgr, capture_stamp)
        return self._tracking_candidates(frame_bgr, capture_stamp)

    def _collect_correspondences(
        self,
        gray: np.ndarray,
        selection: _CandidateSelection,
        *,
        prepared_query=None,
        on_batch=None,
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
                prepared_query=prepared_query,
                on_batch=on_batch,
            )
            points2d, points3d, confidence, per_ref = _merge_correspondence_rows(rows)
        else:
            by_ref = self.loc.correspondences_by_ref(
                gray,
                refs,
                batch_size=self.cfg.match_batch_size,
                prepared_query=prepared_query,
                on_batch=on_batch,
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
        prior: Any | None = None,
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
        if self._prior_pnp_refine_enabled() and hasattr(pycolmap, "refine_absolute_pose"):
            resolved_prior = self._resolve_pose_prior(prior)
            if resolved_prior is not None:
                try:
                    p3d_arr = np.asarray(capped_points3d, float)
                    p2d_arr = np.asarray(capped_points2d, float)
                    pts_cam = np.asarray(resolved_prior * p3d_arr, dtype=float)
                    valid = (pts_cam[:, 2] > 1e-6)
                    if valid.any():
                        proj = np.asarray(pcam.img_from_cam(pts_cam), dtype=float)
                        valid_proj = valid & np.isfinite(proj).all(axis=1)
                        if valid_proj.any():
                            errors = np.linalg.norm(proj - p2d_arr, axis=1)
                            inlier_mask = valid_proj & (errors <= options.ransac.max_error)
                            num_inl = int(np.sum(inlier_mask))
                            inl_ratio = float(num_inl / max(1, len(capped_points2d)))
                            if num_inl >= 60 and inl_ratio >= 0.65:
                                refine_res = pycolmap.refine_absolute_pose(
                                    resolved_prior,
                                    p2d_arr,
                                    p3d_arr,
                                    inlier_mask,
                                    pcam,
                                )
                                if refine_res is not None and "cam_from_world" in refine_res:
                                    refined_cam = refine_res["cam_from_world"]
                                    p_cam_ref = np.asarray(refined_cam * p3d_arr, dtype=float)
                                    val_ref = (p_cam_ref[:, 2] > 1e-6)
                                    proj_ref = np.asarray(pcam.img_from_cam(p_cam_ref), dtype=float)
                                    val_proj_ref = val_ref & np.isfinite(proj_ref).all(axis=1)
                                    err_ref = np.linalg.norm(proj_ref - p2d_arr, axis=1)
                                    mask_ref = val_proj_ref & (err_ref <= options.ransac.max_error)
                                    ref_inliers = int(np.sum(mask_ref))
                                    if ref_inliers >= 60:
                                        estimate = {
                                            "cam_from_world": refined_cam,
                                            "num_inliers": ref_inliers,
                                            "inlier_mask": mask_ref,
                                        }
                                        metrics = reprojection_metrics(
                                            estimate,
                                            capped_points2d,
                                            capped_points3d,
                                            pcam,
                                            self.cfg.corr_grid,
                                        )
                                        if metrics.get("reproj_rms") is not None:
                                            elapsed_ms = (time.perf_counter() - started) * 1e3
                                            return (
                                                estimate,
                                                capped_points2d,
                                                capped_points3d,
                                                metrics,
                                            ), elapsed_ms
                except Exception:
                    pass

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

    def _relaxed_acquire_floor_value(self, selection: _CandidateSelection) -> int:
        """Relaxed acquire floor for this state, or 0 when the tier does not apply.

        LOST uses ``acquire_relaxed_min_inliers`` and is guarded by the armed
        trajectory gates. BOOT_INIT has no prior at all, so it uses its own
        ``boot_relaxed_min_inliers`` and pays for the missing prior with the
        two-frame confirmation in ``_boot_relaxed_confirms``.
        """
        if not selection.acquiring:
            return 0
        state = self.st.state
        if state == "LOST":
            floor = int(getattr(self.cfg, "acquire_relaxed_min_inliers", 0) or 0)
        elif state == "BOOT_INIT":
            floor = int(getattr(self.cfg, "boot_relaxed_min_inliers", 0) or 0)
        else:
            return 0
        if floor <= 0 or floor >= int(selection.min_inliers):
            return 0
        return floor

    def _count_relaxed_agreement(
        self,
        selection: _CandidateSelection,
        scored: list[tuple[str, tuple]],
        candidate: tuple,
    ) -> int:
        """Agreeing references for a below-threshold acquire candidate (0 when N/A)."""
        floor = self._relaxed_acquire_floor_value(selection)
        if floor <= 0 or self._candidate_inliers(candidate) >= int(selection.min_inliers):
            return 0
        _rotation, center, _yaw = self._pose_components(candidate[0])
        return count_agreeing_refs(scored, center, floor, self._acquire_consensus_limit())

    def _relaxed_acquire_gate(
        self,
        selection: _CandidateSelection,
        attempt: _PoseAttempt,
    ) -> tuple[int, float] | None:
        """Second-tier (min_inliers, reproj_limit) for a corroborated near-miss.

        Returns None unless every relaxed condition holds, so the standard
        acquire floor stays in force. The floor only drops for a LOST
        re-acquisition whose match is dense enough to place the pose twice:
        ``acquire_relaxed_min_agreeing_refs`` references at or above the
        relaxed floor landing inside the acquire consensus radius. The
        reprojection limit tightens at the same time, and every trajectory
        gate downstream (acquire_jump / acquire_yaw / stale confirmation)
        still runs unchanged.
        """
        floor = self._relaxed_acquire_floor_value(selection)
        if floor <= 0 or attempt.result is None:
            return None
        inliers = int(attempt.result["num_inliers"])
        if not floor <= inliers < int(selection.min_inliers):
            return None
        needed = int(getattr(self.cfg, "acquire_relaxed_min_agreeing_refs", 2) or 0)
        if int(attempt.relaxed_agree_count) < needed:
            return None
        reproj_limit = min(
            float(self.cfg.max_reproj_error_acquire),
            float(getattr(self.cfg, "acquire_relaxed_max_reproj_error", 0.0) or 0.0),
        )
        if reproj_limit <= 0.0:
            return None
        return floor, reproj_limit

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

    def _estimate_reference_row(self, row, prior: Any | None = None) -> tuple | None:
        ref_p2, ref_p3, ref_confidence, _count = row
        thread_camera, thread_options = self._new_pose_estimation_context(self._pnp_random_seed())
        if prior is not None:
            return self._estimate_pose_candidate(
                ref_p2,
                ref_p3,
                ref_confidence,
                thread_camera,
                thread_options,
                prior=prior,
            )[0]
        return self._estimate_pose_candidate(
            ref_p2,
            ref_p3,
            ref_confidence,
            thread_camera,
            thread_options,
        )[0]

    def _pnp_candidates_can_stop(
        self,
        selection: _CandidateSelection,
        scored: list[tuple[str, tuple]],
        capture_stamp: float,
    ) -> bool:
        strong_count = sum(
            1
            for _name, candidate in scored
            if self._candidate_inliers(candidate) >= selection.min_inliers
        )
        if selection.acquiring and strong_count < 2:
            return False
        chosen = (
            self._select_acquire_candidate(selection, scored)
            if selection.acquiring
            else self._select_track_candidate(selection, scored)
        )
        if chosen is None:
            return False
        _name, candidate = chosen
        result, _points2d, _points3d, metrics = candidate
        reproj_limit = (
            self.cfg.max_reproj_error_acquire
            if selection.acquiring
            else self.cfg.max_reproj_error_track
        )
        if not self._pose_quality_ok(
            result,
            metrics,
            min_inliers=selection.min_inliers,
            reproj_limit=reproj_limit,
        ):
            return False
        _rotation, center, yaw = self._pose_components(result)
        return self._trajectory_would_allow(
            center,
            yaw,
            capture_stamp,
            self.st.state,
        )

    def _best_pose_attempt(
        self,
        selection: _CandidateSelection,
        batch: _CorrespondenceBatch,
        *,
        capture_stamp: float | None = None,
        precomputed: list[tuple[int, str, tuple | None]] | None = None,
        precomputed_pnp_ms: float = 0.0,
    ) -> _PoseAttempt:
        relaxed_agree_count = 0
        result = None
        selected_ref = None
        points2d = np.zeros((0, 2))
        points3d = np.zeros((0, 3))
        metrics = {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0}
        pnp_ms = 0.0
        reject_reason = None
        strong_count = 0
        pnp_candidates = 0
        pnp_skipped = 0
        if batch.by_ref is not None:
            scored: list[tuple[str, tuple]] = []
            rows = list(enumerate(zip(selection.refs, batch.by_ref)))
            if precomputed is not None:
                ordered = sorted(precomputed, key=lambda item: item[0])
                pnp_candidates = len(ordered)
                pnp_skipped = len(rows) - pnp_candidates
                pnp_ms = float(precomputed_pnp_ms)
                for _ordinal, name, candidate in ordered:
                    if candidate is not None:
                        scored.append((name, candidate))
            else:
                if self.cfg.pnp_early_stop:
                    eligible_thr = _pnp_eligible_threshold(selection.min_inliers)
                    eligible = [row for row in rows if len(row[1][1][1]) >= eligible_thr]
                    if not eligible and rows:
                        # Every ref is corr-starved: rescue the two most
                        # correspondence-rich refs instead of forcing LOST.
                        # Downstream inlier/reproj/jump gates still apply.
                        eligible = sorted(
                            rows, key=lambda row: len(row[1][1][1]), reverse=True
                        )[:2]
                else:
                    eligible = rows
                if self.cfg.pnp_ranked_batches:
                    eligible.sort(
                        key=lambda row: _reference_pnp_rank(
                            row[1][1],
                            width=self.cam.width,
                            height=self.cam.height,
                            grid=self.cfg.corr_grid,
                        ),
                        reverse=True,
                    )
                chunk_size = (
                    max(1, self.cfg.pnp_workers)
                    if self.cfg.pnp_ranked_batches
                    else max(1, len(eligible))
                )
                pnp_started = time.perf_counter()
                executor = self._parallel_pnp_executor()
                for start in range(0, len(eligible), chunk_size):
                    chunk = eligible[start : start + chunk_size]
                    row_values = [row[1][1] for row in chunk]
                    if executor is not None and len(chunk) > 1:
                        candidates = list(executor.map(self._estimate_reference_row, row_values))
                    else:
                        candidates = [self._estimate_reference_row(row) for row in row_values]
                    pnp_candidates += len(chunk)
                    for (_ordinal, (name, _row)), candidate in zip(chunk, candidates):
                        if candidate is not None:
                            scored.append((name, candidate))
                    if (
                        self.cfg.pnp_ranked_batches
                        and capture_stamp is not None
                        and self._pnp_candidates_can_stop(
                            selection,
                            scored,
                            capture_stamp,
                        )
                    ):
                        break
                if eligible:
                    pnp_ms = (time.perf_counter() - pnp_started) * 1e3
                pnp_skipped = len(rows) - pnp_candidates
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
                relaxed_agree_count = self._count_relaxed_agreement(
                    selection, scored, candidate
                )
        else:
            pcam, options = self._pose_estimation_context()
            candidate, pnp_ms = self._estimate_pose_candidate(
                batch.points2d,
                batch.points3d,
                batch.confidence,
                pcam,
                options,
            )
            if candidate is not None:
                result, points2d, points3d, metrics = candidate
                pnp_candidates = 1
        return _PoseAttempt(
            result=result,
            points2d=points2d,
            points3d=points3d,
            metrics=metrics,
            selected_ref=selected_ref,
            pnp_ms=pnp_ms,
            reject_reason=reject_reason,
            strong_count=strong_count,
            pnp_candidates=pnp_candidates,
            pnp_skipped=pnp_skipped,
            relaxed_agree_count=relaxed_agree_count,
        )

    def _match_and_estimate(
        self,
        gray: np.ndarray,
        selection: _CandidateSelection,
        *,
        capture_stamp: float | None = None,
        prepared_query=None,
    ) -> tuple[_CorrespondenceBatch, _PoseAttempt, float]:
        started = time.perf_counter()
        pending = []
        skipped_rows: list = []
        executor = (
            self._parallel_pnp_executor()
            if (
                self.cfg.pnp_pipeline
                and not self.cfg.pnp_ranked_batches
                and self.cfg.pnp_workers > 1
            )
            else None
        )

        def submit_rows(start_index: int, rows: list) -> None:
            if executor is None:
                return
            for offset, row in enumerate(rows):
                ordinal = start_index + offset
                eligible_thr = _pnp_eligible_threshold(selection.min_inliers)
                if self.cfg.pnp_early_stop and len(row[1]) < eligible_thr:
                    skipped_rows.append((ordinal, row))
                    continue
                pending.append(
                    (
                        ordinal,
                        selection.refs[ordinal],
                        executor.submit(self._estimate_reference_row, row),
                    )
                )

        batch = self._collect_correspondences(
            gray,
            selection,
            prepared_query=prepared_query,
            on_batch=submit_rows if executor is not None else None,
        )
        if executor is not None and batch.by_ref is not None:
            wait_started = time.perf_counter()
            if not pending and skipped_rows:
                # Pipeline early-stop dropped every ref: estimate the two most
                # correspondence-rich refs inline rather than forcing LOST.
                rescue = sorted(skipped_rows, key=lambda item: len(item[1][1]), reverse=True)[:2]
                precomputed = [
                    (ordinal, selection.refs[ordinal], self._estimate_reference_row(row))
                    for ordinal, row in rescue
                ]
            else:
                precomputed = [(ordinal, name, future.result()) for ordinal, name, future in pending]
            pnp_wait_ms = (time.perf_counter() - wait_started) * 1e3
            attempt = self._best_pose_attempt(
                selection,
                batch,
                capture_stamp=capture_stamp,
                precomputed=precomputed,
                precomputed_pnp_ms=pnp_wait_ms,
            )
        else:
            attempt = self._best_pose_attempt(
                selection,
                batch,
                capture_stamp=capture_stamp,
            )
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
        extract_tensor = getattr(self.loc.megaloc, "extract_one_tensor", None)
        descriptor = (
            extract_tensor(rgb) if callable(extract_tensor) else self.loc.megaloc.extract_one(rgb)
        )
        scored = list(self.loc.retrieve_scored(rgb, pool, descriptor=descriptor))
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
            if center is not None and self.centers is not None and index is not None:
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
                if (self.st.last_capture_stamp is None or capture_stamp is None)
                else capture_stamp - self.st.last_capture_stamp
            )
            if self.st.yaw is not None and math.isfinite(float(self.st.yaw)):
                yaw_delta = abs(_angle_diff(yaw, float(self.st.yaw)))
                if yaw_delta > math.radians(track_yaw_limit(self.cfg, capture_dt)):
                    return False
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
        single_strong_ok = False
        if (
            os.environ.get("SFM_EDM_SINGLE_STRONG_EARLY_STOP", "1").strip().lower()
            not in ("0", "false", "no", "off")
            and attempt.strong_count == 1
        ):
            is_top1 = (
                attempt.selected_ref is None
                or not selection.refs
                or attempt.selected_ref == selection.refs[0]
            )
            if is_top1:
                acquire_min_inl = float(
                    getattr(self.cfg, "acquire_min_inliers", selection.min_inliers)
                )
                required_inliers = max(
                    acquire_min_inl * 1.5,
                    float(_ACQUIRE_SINGLE_STRONG_INLIERS),
                )
                if self._pose_quality_ok(
                    attempt.result,
                    attempt.metrics,
                    min_inliers=int(math.ceil(required_inliers)),
                    reproj_limit=float(_ACQUIRE_SINGLE_STRONG_REPROJ),
                ):
                    single_strong_ok = True
                    self._single_strong_stops = int(getattr(self, "_single_strong_stops", 0) or 0) + 1

        if not single_strong_ok and attempt.strong_count < 2:
            return False

        if not single_strong_ok:
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
        host_cache_stats = None
        read_class_stats = getattr(matcher, "feature_cache_stats_by_class", None)
        if callable(read_class_stats):
            cache_stats = read_class_stats()
        read_host_stats = getattr(matcher, "host_reference_feature_cache_stats", None)
        if callable(read_host_stats):
            host_cache_stats = read_host_stats()
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
            "host_feature_cache": host_cache_stats,
            "motion_cache_active": self._motion_cache_active(),
            "n_corr": int(len(batch.points3d)),
            "per_ref": batch.per_ref,
            "global_retrieval_calls": self.st.global_retrieval_calls,
            "lost_global_retrieval_done": self.st.lost_global_retrieval_done,
            "lost_search_stage": self._last_lost_search_stage,
            "lost_search_radius_factor": self._last_lost_radius_factor,
            "vpr_ms": selection.vpr_ms,
            "match_ms": match_ms,
            "pnp_ms": attempt.pnp_ms,
            "pnp_candidates": attempt.pnp_candidates,
            "pnp_skipped": attempt.pnp_skipped,
            "pnp_workers": int(self.cfg.pnp_workers),
            "pnp_pipeline": bool(self.cfg.pnp_pipeline),
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

    def _clear_pending_reacquisition(self) -> None:
        self.st.pending_reacquire_center = None
        self.st.pending_reacquire_yaw = None
        self.st.pending_reacquire_stamp = None

    def _stale_reacquisition_allows(
        self,
        center: np.ndarray,
        yaw: float,
        capture_stamp: float,
        info: dict,
    ) -> bool:
        if self.cfg.stale_reacquire_confirmations == 1:
            return True
        pending_center = self.st.pending_reacquire_center
        pending_yaw = self.st.pending_reacquire_yaw
        pending_stamp = self.st.pending_reacquire_stamp
        independent = pending_stamp is not None and capture_stamp > pending_stamp
        distance = None
        yaw_delta = None
        if pending_center is not None and independent:
            distance = float(np.linalg.norm(center - pending_center))
            if pending_yaw is not None and math.isfinite(float(pending_yaw)):
                yaw_delta = abs(_angle_diff(yaw, float(pending_yaw)))
            position_consistent = distance <= float(self.cfg.stale_reacquire_max_distance)
            yaw_consistent = yaw_delta is None or yaw_delta <= math.radians(
                float(self.cfg.stale_reacquire_max_yaw_diff_deg)
            )
            if position_consistent and yaw_consistent:
                info["stale_reacquire_confirmed"] = True
                self._clear_pending_reacquisition()
                return True
        self.st.pending_reacquire_center = np.asarray(center, dtype=float).copy()
        self.st.pending_reacquire_yaw = float(yaw)
        self.st.pending_reacquire_stamp = float(capture_stamp)
        info["stale_reacquire_confirmation"] = {
            "independent": independent,
            "distance": distance,
            "max_distance": float(self.cfg.stale_reacquire_max_distance),
            "yaw_delta_deg": None if yaw_delta is None else math.degrees(yaw_delta),
            "max_yaw_diff_deg": float(self.cfg.stale_reacquire_max_yaw_diff_deg),
        }
        info["rejected"] = "stale_reacquire_unconfirmed"
        info.update({"state_out": self.st.state, "ok": False})
        if info.get("pose_status") in _VISUAL_POSE_STATUSES:
            info["pose_status"] = "NONE"
        return False

    def _boot_relaxed_confirms(
        self,
        center: np.ndarray,
        yaw: float,
        capture_stamp: float,
        info: dict,
    ) -> bool:
        """Two-frame confirmation standing in for BOOT's missing prior.

        BOOT has no accepted pose, so acquire_jump / acquire_yaw are not armed
        and a relaxed-floor BOOT accept would be unguarded. Require instead that
        an independent later frame lands within the same stale-reacquisition
        window (distance / yaw), which is the site-calibrated definition of "two
        frames agree on where we are".
        """
        pending_center = self.st.pending_reacquire_center
        pending_yaw = self.st.pending_reacquire_yaw
        pending_stamp = self.st.pending_reacquire_stamp
        independent = pending_stamp is not None and capture_stamp > pending_stamp
        distance = None
        yaw_delta = None
        if pending_center is not None and independent:
            distance = float(np.linalg.norm(center - pending_center))
            if pending_yaw is not None and math.isfinite(float(pending_yaw)):
                yaw_delta = abs(_angle_diff(yaw, float(pending_yaw)))
            position_consistent = distance <= float(self.cfg.stale_reacquire_max_distance)
            yaw_consistent = yaw_delta is None or yaw_delta <= math.radians(
                float(self.cfg.stale_reacquire_max_yaw_diff_deg)
            )
            if position_consistent and yaw_consistent:
                info["boot_relaxed_confirmed"] = True
                self._clear_pending_reacquisition()
                return True
        self.st.pending_reacquire_center = np.asarray(center, dtype=float).copy()
        self.st.pending_reacquire_yaw = float(yaw)
        self.st.pending_reacquire_stamp = float(capture_stamp)
        info["boot_relaxed_confirmation"] = {
            "independent": independent,
            "distance": distance,
            "max_distance": float(self.cfg.stale_reacquire_max_distance),
            "yaw_delta_deg": None if yaw_delta is None else math.degrees(yaw_delta),
            "max_yaw_diff_deg": float(self.cfg.stale_reacquire_max_yaw_diff_deg),
        }
        info["rejected"] = "boot_relaxed_unconfirmed"
        info.update({"state_out": self.st.state, "ok": False})
        if info.get("pose_status") in _VISUAL_POSE_STATUSES:
            info["pose_status"] = "NONE"
        self._on_miss(info)
        return False

    def _clear_visual_motion_cache(self) -> None:
        self._last_accepted_bgr = None
        self._last_accepted_cam_from_world = None
        self._last_accepted_rigid3d = None
    def _clear_klt_cache(self) -> None:
        self._klt_gray = None
        self._klt_2d = None
        self._klt_3d = None
        self._klt_center = None
        self._klt_yaw = None
        self._klt_seed_stamp = None
        self._klt_age = 0

    def _seed_klt_from_inliers(
        self,
        gray: np.ndarray,
        points2d: np.ndarray,
        points3d: np.ndarray,
        inlier_mask: np.ndarray,
        capture_stamp: float,
    ) -> bool:
        try:
            mask = np.asarray(inlier_mask, dtype=bool)
            if mask.shape[0] != points2d.shape[0]:
                # mismatched length: fall back to all points
                mask = np.ones(points2d.shape[0], dtype=bool)
            seed2d = np.asarray(points2d, dtype=float)[mask]
            seed3d = np.asarray(points3d, dtype=float)[mask]
            if seed2d.shape[0] == 0 or seed3d.shape[0] == 0:
                self._clear_klt_cache()
                return False
            _, unique_2d = np.unique(seed2d, axis=0, return_index=True)
            unique_2d = np.sort(unique_2d)
            _, unique_3d = np.unique(seed3d[unique_2d], axis=0, return_index=True)
            unique = unique_2d[np.sort(unique_3d)]
            if len(unique) < _KLT_MIN_SEED:
                self._clear_klt_cache()
                return False
            self._klt_2d = seed2d[unique].copy()
            self._klt_3d = seed3d[unique].copy()
            self._klt_gray = np.ascontiguousarray(gray)
            self._klt_seed_stamp = float(capture_stamp)
            self._klt_age = 0
            self._klt_center = None
            self._klt_yaw = None
            return True
        except Exception:
            self._clear_klt_cache()
            return False
    _SHADOW_KLT_FIELDS = (
        "_klt_2d",
        "_klt_3d",
        "_klt_gray",
        "_klt_seed_stamp",
        "_klt_age",
        "_klt_center",
        "_klt_yaw",
    )

    def _klt_shadow_step(
        self,
        gray: np.ndarray,
        capture_stamp: float,
        accepted_center: np.ndarray | None,
        info: dict,
    ) -> None:
        """Score what the KLT chain would have predicted for a frame EDM solved.

        Runs on accepted frames only, which is the whole point: those are the
        only frames carrying a visual fix to measure the prediction against.
        The chain is deliberately NOT re-seeded here, so it ages exactly as it
        would through a long LOST episode while EDM keeps supplying truth.
        The production cache is swapped out and restored, so enabling this
        cannot change tracking.
        """
        saved = {name: getattr(self, name, None) for name in self._SHADOW_KLT_FIELDS}
        try:
            shadow = self._shadow_klt
            if shadow is None:
                return
            for name in self._SHADOW_KLT_FIELDS:
                setattr(self, name, shadow.get(name))
            prior = self._track_klt_prior(gray, capture_stamp)
            self._shadow_klt = (
                None
                if prior is None
                else {name: getattr(self, name, None) for name in self._SHADOW_KLT_FIELDS}
            )
            if prior is None:
                info["klt_shadow_alive"] = False
                return
            self._shadow_klt_anchor_frames += 1
            predicted = np.asarray(prior["center"], dtype=float)
            info["klt_shadow_alive"] = True
            info["klt_shadow_age"] = int(getattr(self, "_klt_age", 0))
            info["klt_shadow_horizon_s"] = float(
                capture_stamp - float(shadow.get("_klt_seed_stamp") or capture_stamp)
            )
            info["klt_shadow_center"] = predicted.tolist()
            info["klt_shadow_inliers"] = int(prior.get("inliers") or 0)
            info["klt_shadow_tracked"] = int(prior.get("tracked") or 0)
            info["klt_shadow_reproj_rms"] = float(prior.get("reproj_rms") or float("nan"))
            if accepted_center is not None:
                truth = np.asarray(accepted_center, dtype=float).reshape(3)
                if np.all(np.isfinite(truth)) and np.all(np.isfinite(predicted)):
                    info["klt_shadow_error"] = float(np.linalg.norm(predicted - truth))
        except Exception:
            self._shadow_klt = None
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def _klt_shadow_reseed(
        self,
        gray: np.ndarray,
        points2d: np.ndarray,
        points3d: np.ndarray,
        inlier_mask: np.ndarray,
        capture_stamp: float,
    ) -> None:
        """Restart the shadow chain from this accepted frame once it has died."""
        if self._shadow_klt is not None:
            return
        saved = {name: getattr(self, name, None) for name in self._SHADOW_KLT_FIELDS}
        try:
            if self._seed_klt_from_inliers(
                gray, points2d, points3d, inlier_mask, capture_stamp
            ):
                self._shadow_klt = {
                    name: getattr(self, name, None) for name in self._SHADOW_KLT_FIELDS
                }
                self._shadow_klt_anchor_frames = 0
        except Exception:
            self._shadow_klt = None
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    @staticmethod
    def _pose_to_quaternion(R: np.ndarray) -> np.ndarray:
        """Rotation matrix 3x3 -> wxyz quaternion normalized (robust trace method)."""
        try:
            R = np.asarray(R, dtype=float).reshape(3, 3)
            t = float(np.trace(R))
            if t > 0.0:
                s = math.sqrt(t + 1.0) * 2.0
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            q = np.array([qw, qx, qy, qz], dtype=float)
            n = float(np.linalg.norm(q))
            if n < 1e-12 or not math.isfinite(n):
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            q /= n
            if q[0] < 0:
                q = -q
            return q
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _esekf_yaw(self) -> float | None:
        """Yaw from ESEKF quaternion if prediction_allowed, else None."""
        try:
            esekf = getattr(self, "esekf", None)
            if esekf is None or not callable(getattr(esekf, "prediction_allowed", None)):
                return None
            if not esekf.prediction_allowed():
                return None
            q = getattr(esekf, "q", None)
            if q is None:
                return None
            q_arr = np.asarray(q, dtype=float).reshape(4)
            n = float(np.linalg.norm(q_arr))
            if n < 1e-12 or not math.isfinite(n):
                return None
            q_arr = q_arr / n
            R = None
            try:
                from esekf import _quat_to_rot as _q2r  # type: ignore
                R = _q2r(q_arr)
            except Exception:
                w, x, y, z = q_arr
                R = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    ],
                    dtype=float,
                )
            forward = R.T @ np.array([0.0, 0.0, 1.0], dtype=float)
            yaw = float(math.atan2(float(forward[1]), float(forward[0])))
            if math.isfinite(yaw):
                return yaw
        except Exception:
            return None
        return None

    def _esekf_predicted_center(self, capture_stamp: float | None) -> np.ndarray | None:
        """Return ESEKF predicted p if allowed (propagates only if not already)."""
        try:
            esekf = getattr(self, "esekf", None)
            if esekf is None or not callable(getattr(esekf, "prediction_allowed", None)):
                return None
            if not esekf.prediction_allowed():
                return None
            # determine stamp
            if capture_stamp is None or not math.isfinite(float(capture_stamp)):
                capture_stamp = float(getattr(esekf, "last_predict", 0.0))
            else:
                capture_stamp = float(capture_stamp)
            vel = getattr(self, "_latest_velocity_ned", None)
            # if already propagated for this stamp, return p without second predict (avoid double age increment)
            try:
                last = float(getattr(esekf, "last_predict", float("nan")))
                if math.isfinite(last) and abs(last - capture_stamp) < 1e-9:
                    p = getattr(esekf, "p", None)
                    if p is not None:
                        p_arr = np.asarray(p, dtype=float).reshape(3)
                        if p_arr.shape == (3,) and np.all(np.isfinite(p_arr)):
                            try:
                                self._esekf_last_predict = p_arr.copy()
                            except Exception:
                                self._esekf_last_predict = p_arr
                            return p_arr
            except Exception:
                pass
            # propagate
            try:
                res = esekf.predict(capture_stamp, velocity_ned=vel)
                if res is not None:
                    if isinstance(res, (tuple, list)) and len(res) >= 1:
                        p_pred = np.asarray(res[0], dtype=float).reshape(3)
                    else:
                        p_pred = getattr(res, "p", getattr(res, "position", None))
                        if p_pred is not None:
                            p_pred = np.asarray(p_pred, dtype=float).reshape(3)
                        else:
                            p_pred = None
                    if p_pred is not None and p_pred.shape == (3,) and np.all(np.isfinite(p_pred)):
                        try:
                            self._esekf_last_predict = p_pred.copy()
                        except Exception:
                            self._esekf_last_predict = p_pred
                        return p_pred
            except Exception:
                pass
            # fallback to direct state p without propagate if predict failed
            p = getattr(esekf, "p", None)
            if p is not None:
                p_arr = np.asarray(p, dtype=float).reshape(3)
                if p_arr.shape == (3,) and np.all(np.isfinite(p_arr)):
                    return p_arr
        except Exception:
            return None
        return None

    def _motion_cache_active(self) -> bool:
        return getattr(self, "motion_validator", None) is not None and getattr(
            self, "motion_validation_mode", "off"
        ) in {"shadow", "confirm_limited_jump"}

    def _store_visual_motion_cache(self, frame_bgr: np.ndarray, cam_from_world) -> None:
        if isinstance(cam_from_world, pycolmap.Rigid3d):
            self._last_accepted_rigid3d = cam_from_world
        else:
            try:
                mat = cam_from_world_matrix(cam_from_world)
                self._last_accepted_rigid3d = pycolmap.Rigid3d(mat[:3, :4])
            except Exception:
                self._last_accepted_rigid3d = None
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
        if (
            result is not None
            and isinstance(result, dict)
            and "cam_from_world" in result
            and self.st.state == "TRACK"
        ):
            try:
                _rot, center, yaw = self._pose_components(result)
                if self.st.center is not None:
                    info["step_norm"] = float(np.linalg.norm(center - self.st.center))
                capture_stamp = info.get("capture_stamp")
                info["step_legal"] = self._trajectory_would_allow(
                    center, yaw, capture_stamp, self.st.state
                )
            except Exception:
                pass
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
        prediction = (
            getattr(controller, "last_prediction", None) if controller is not None else None
        )
        if (
            prediction is not None
            and getattr(prediction, "valid", False)
            and getattr(prediction, "position", None) is not None
        ):
            predicted = np.asarray(prediction.position, dtype=float)
            if predicted.shape == (3,) and np.isfinite(predicted).all():
                return True, "predicted_center", float(np.linalg.norm(center - predicted))
        if self.st.velocity is None or self.st.center is None or self.st.last_capture_stamp is None:
            return False, None, None
        dt = float(capture_stamp) - float(self.st.last_capture_stamp)
        if not math.isfinite(dt) or dt <= 1e-6:
            return False, None, None
        dt = min(dt, float(self.cfg.prediction_max_dt))
        predicted = (
            np.asarray(self.st.center, dtype=float) + np.asarray(self.st.velocity, dtype=float) * dt
        )
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
            if info.get("pose_status") in _VISUAL_POSE_STATUSES:
                info["pose_status"] = "NONE"
        return False

    def _continuous_trajectory_allows(
        self,
        center: np.ndarray,
        yaw: float,
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
        if self.st.yaw is not None and math.isfinite(float(self.st.yaw)):
            yaw_delta = abs(_angle_diff(yaw, float(self.st.yaw)))
            yaw_limit_deg = track_yaw_limit(self.cfg, capture_dt)
            info["track_yaw_delta_deg"] = math.degrees(yaw_delta)
            info["track_yaw_limit_deg"] = yaw_limit_deg
            if yaw_delta > math.radians(yaw_limit_deg):
                self._clear_pending_jump()
                info["rejected"] = "track_yaw"
                self._on_miss(info)
                return False
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
            return self._stale_reacquisition_allows(center, yaw, capture_stamp, info)
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
            # No prior at all (BOOT). A standard-floor accept is unchanged; a
            # relaxed-floor accept pays for the missing prior with a two-frame
            # confirmation instead.
            if info.get("acquire_relaxed") and info.get("state_in") == "BOOT_INIT":
                return self._boot_relaxed_confirms(center, yaw, capture_stamp, info)
            return True
        if info["state_in"] != "LOST":
            return self._continuous_trajectory_allows(
                center,
                yaw,
                capture_stamp,
                info,
                frame_bgr,
                cam_from_world,
            )
        return self._lost_reacquisition_allows(center, yaw, capture_stamp, info)

    def _next_state_after_accept(self, info: dict) -> str:
        """TRACK, or WEAK_TRACK while a relaxed-accept probation counts down."""
        probation = int(getattr(self.cfg, "acquire_relaxed_probation_frames", 0) or 0)
        if probation > 0 and info.get("acquire_relaxed"):
            self.st.relaxed_probation_frames = probation
        remaining = int(getattr(self.st, "relaxed_probation_frames", 0) or 0)
        if remaining > 0:
            self.st.relaxed_probation_frames = remaining - 1
            info["acquire_relaxed_probation"] = remaining
            return "WEAK_TRACK"
        return "TRACK"

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
        prepared_query=None,
    ) -> dict:
        previous_center = self.st.center
        self._clear_pending_reacquisition()
        previous_stamp = self.st.last_capture_stamp
        step = None if previous_center is None else (center - previous_center)
        measured_velocity = None
        dt = None
        if step is not None:
            dt = None if previous_stamp is None else capture_stamp - previous_stamp
            if info["state_in"] != "LOST" and dt is not None and math.isfinite(dt) and dt > 1e-6:
                self.st.accepted_step_norms.append(float(np.linalg.norm(step)))
                measured_velocity = step / dt
        if measured_velocity is None:
            pass  # LOST or missing stamp: keep the previous velocity estimate
        elif self.st.velocity is None:
            self.st.velocity = np.asarray(measured_velocity, dtype=float)
        else:
            self.st.velocity = _blend_velocity(self.st.velocity, measured_velocity, dt)
        self.st.center, self.st.yaw = center, yaw
        self.st.last_capture_stamp = capture_stamp
        # --- ESEKF visual update (adaptive R, Mahalanobis gating) ---
        try:
            esekf = getattr(self, "esekf", None)
            if esekf is not None:
                q_wxyz = self._pose_to_quaternion(rotation)
                v_for_ekf = self.st.velocity
                if v_for_ekf is None:
                    v_arr = np.zeros(3, dtype=float)
                else:
                    try:
                        v_arr = np.asarray(v_for_ekf, dtype=float).reshape(3)
                        if not np.all(np.isfinite(v_arr)):
                            v_arr = np.zeros(3, dtype=float)
                    except Exception:
                        v_arr = np.zeros(3, dtype=float)
                is_init = bool(getattr(esekf, "_initialized", False))
                if not is_init:
                    try:
                        esekf.reset_from_pose(p=center, q_wxyz=q_wxyz, v=v_arr, timestamp=capture_stamp)  # type: ignore
                    except Exception:
                        try:
                            esekf.reset_from_pose(center, q_wxyz, v_arr, capture_stamp)  # type: ignore
                        except Exception:
                            pass
                else:
                    inliers_val = None
                    reproj_val = None
                    corr_val = None
                    try:
                        if isinstance(result, dict):
                            inliers_val = result.get("num_inliers", result.get("inliers"))
                    except Exception:
                        pass
                    if inliers_val is None:
                        try:
                            m = getattr(attempt, "metrics", None)
                            if isinstance(m, dict):
                                inliers_val = m.get("num_inliers", m.get("inliers"))
                        except Exception:
                            pass
                    try:
                        m = getattr(attempt, "metrics", None)
                        if isinstance(m, dict):
                            reproj_val = m.get("reproj_rms", m.get("reproj_rmse"))
                            if reproj_val is None:
                                reproj_val = info.get("reproj_rms")
                        else:
                            reproj_val = info.get("reproj_rms")
                    except Exception:
                        pass
                    try:
                        if hasattr(attempt, "points2d"):
                            corr_val = len(attempt.points2d)  # type: ignore
                        elif hasattr(attempt, "points3d"):
                            corr_val = len(attempt.points3d)  # type: ignore
                    except Exception:
                        pass
                    jump_val = None
                    try:
                        if previous_center is not None:
                            jump_val = float(np.linalg.norm(np.asarray(center, dtype=float) - np.asarray(previous_center, dtype=float)))
                    except Exception:
                        pass
                    R_pos = None
                    R_ori = None
                    if _esekf_adaptive_cov is not None:
                        try:
                            Rp, Ro = _esekf_adaptive_cov(  # type: ignore
                                inliers=inliers_val,
                                reproj_rmse=reproj_val,
                                corr_count=corr_val,
                                jump=jump_val,
                            )
                            R_pos = Rp
                            R_ori = Ro
                        except Exception:
                            R_pos = None
                            R_ori = None
                    accepted = False
                    d2_val = None
                    try:
                        if R_pos is not None and R_ori is not None:
                            accepted, d2_val, _nu = esekf.update_visual(  # type: ignore
                                z_p=center,
                                z_q_wxyz=q_wxyz,
                                R_visual_pos=R_pos,
                                R_visual_ori=R_ori,
                                timestamp=capture_stamp,
                            )
                        else:
                            accepted, d2_val, _nu = esekf.update_visual(  # type: ignore
                                z_p=center,
                                z_q_wxyz=q_wxyz,
                                timestamp=capture_stamp,
                            )
                    except TypeError:
                        try:
                            accepted, d2_val, _nu = esekf.update_visual(center, q_wxyz, timestamp=capture_stamp)  # type: ignore
                        except Exception:
                            accepted = False
                            self._esekf_update_exceptions = (
                                int(getattr(self, "_esekf_update_exceptions", 0) or 0) + 1
                            )
                    except Exception:
                        accepted = False
                        self._esekf_update_exceptions = (
                            int(getattr(self, "_esekf_update_exceptions", 0) or 0) + 1
                        )
                    if d2_val is not None:
                        try:
                            info["esekf_d2"] = float(d2_val)
                            info["esekf_update_accepted"] = bool(accepted)
                        except Exception:
                            pass
                    try:
                        info["esekf_update_exceptions"] = int(
                            getattr(self, "_esekf_update_exceptions", 0) or 0
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        accepted_refs = (
            [attempt.selected_ref] if attempt.selected_ref is not None else selection.refs
        )
        self.st.last_refs = [self.idx_of[name] for name in accepted_refs]
        self._record_reference_stability(self.st.last_refs, attempt.metrics)
        self.st.misses = 0
        self.st.lost_frames = 0
        self.st.lost_global_retrieval_attempts = 0
        self.st.lost_global_retrieval_done = False
        self._vpr_consec_failures = 0
        self._last_vpr_lost_frame = None
        self._store_visual_motion_cache(frame_bgr, result["cam_from_world"])
        # Probation: an accept that needed the relaxed floor keeps the tracker in
        # WEAK_TRACK for a few frames, so the following frames run
        # weak_local_topk references at weak_min_inliers instead of a single
        # reference. Measured cause of the P168 relaxed-floor loss: the accepts
        # themselves were correct but single-reference TRACK could not hold them
        # (inliers p50 82->68, WEAK_TRACK +47).
        self.st.state = self._next_state_after_accept(info)
        if self.st.state == "WEAK_TRACK":
            self.st.misses = float(self.cfg.weak_after)
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
            matcher = getattr(self.loc, "matcher", None)
            promote = getattr(matcher, "promote_prepared_query", None)
            if callable(promote):
                promote(self.temporal_gray, prepared_query)
        else:
            self.temporal_xyz_by_cell = None
            self.temporal_gray = None
        try:
            self._klt_anchor_inliers = int(info.get("inliers", 0) or 0)
        except (TypeError, ValueError):
            self._klt_anchor_inliers = 0
        try:
            self._klt_anchor_ratio = float(info.get("inlier_ratio") or 0.0)
        except (TypeError, ValueError):
            self._klt_anchor_ratio = 0.0
        try:
            inlier_mask_for_klt = np.asarray(
                result.get(
                    "inlier_mask",
                    np.ones(len(attempt.points3d), dtype=bool),
                ),
                dtype=bool,
            )
            if self._klt_shadow_eval:
                # Score first, then re-seed: the shadow must see this frame with
                # the age it carried in, not reset to zero by the production
                # anchor that is about to be laid down.
                self._klt_shadow_step(gray, capture_stamp, center, info)
                self._klt_shadow_reseed(
                    gray,
                    attempt.points2d,
                    attempt.points3d,
                    inlier_mask_for_klt,
                    capture_stamp,
                )
            self._seed_klt_from_inliers(
                gray, attempt.points2d, attempt.points3d, inlier_mask_for_klt, capture_stamp
            )
        except Exception:
            try:
                self._clear_klt_cache()
            except Exception:
                pass
        try:
            self._klt_center = None
        except Exception:
            pass
        info.update(
            {
                "state_out": self.st.state,
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

    def _acquisition_stage_counts(self, requested: int) -> list[int]:
        if requested <= 0 or self.cfg.acquire_stage_mode == "full_set":
            return [requested]
        if self.cfg.acquire_stage_mode == "initial_topk":
            candidates = (min(self.cfg.acquire_initial_topk, requested), requested)
        else:
            candidates = (
                *(min(topk, requested) for topk in ACQUIRE_PROGRESSIVE_TOPKS),
                requested,
            )
        return list(dict.fromkeys(count for count in candidates if count > 0))

    def _match_acquisition_stages(
        self,
        gray: np.ndarray,
        selection: _CandidateSelection,
        capture_stamp: float,
        prepared_query,
    ) -> tuple[
        _CandidateSelection,
        _CorrespondenceBatch,
        _PoseAttempt,
        float,
        list[int],
    ]:
        stage_counts = self._acquisition_stage_counts(len(selection.refs))
        evaluated: list[int] = []
        batch = None
        attempt = None
        match_ms = 0.0
        pnp_ms = 0.0
        previous_count = 0
        active_selection = selection
        for count in stage_counts:
            active_selection = replace(selection, refs=list(selection.refs[:count]))
            if batch is None:
                batch, attempt, stage_match_ms = self._match_and_estimate(
                    gray,
                    active_selection,
                    capture_stamp=capture_stamp,
                    prepared_query=prepared_query,
                )
            else:
                remaining = replace(
                    selection,
                    refs=list(selection.refs[previous_count:count]),
                )
                if (batch.by_ref is not None or batch.temporal_used) and remaining.refs:
                    expanded_started = time.perf_counter()
                    if batch.temporal_used:
                        remaining_selection = replace(remaining, mode="edm_track")
                        remaining_batch = self._collect_correspondences(
                            gray,
                            remaining_selection,
                            prepared_query=prepared_query,
                        )
                        combined_p2d = np.concatenate([batch.points2d, remaining_batch.points2d])
                        combined_p3d = np.concatenate([batch.points3d, remaining_batch.points3d])
                        combined_conf = np.concatenate([batch.confidence, remaining_batch.confidence])
                        combined_per_ref = [*batch.per_ref, *remaining_batch.per_ref]
                        batch = _CorrespondenceBatch(
                            points2d=combined_p2d,
                            points3d=combined_p3d,
                            confidence=combined_conf,
                            per_ref=combined_per_ref,
                            by_ref=None,
                            temporal_used=True,
                        )
                        attempt = self._best_pose_attempt(
                            active_selection,
                            batch,
                            capture_stamp=capture_stamp,
                        )
                    else:
                        remaining_batch = self._collect_correspondences(
                            gray,
                            remaining,
                            prepared_query=prepared_query,
                        )
                        batch = _combine_correspondence_batches(batch, remaining_batch)
                        attempt = self._best_pose_attempt(
                            active_selection,
                            batch,
                            capture_stamp=capture_stamp,
                        )
                    stage_match_ms = max(
                        0.0,
                        (time.perf_counter() - expanded_started) * 1e3 - attempt.pnp_ms,
                    )
                else:
                    batch, attempt, stage_match_ms = self._match_and_estimate(
                        gray,
                        active_selection,
                        capture_stamp=capture_stamp,
                        prepared_query=prepared_query,
                    )
            match_ms += stage_match_ms
            pnp_ms += attempt.pnp_ms
            evaluated.append(count)
            previous_count = count
            if count < len(selection.refs) and self._acquire_stage_can_stop(
                active_selection,
                attempt,
                capture_stamp,
            ):
                break
        assert batch is not None and attempt is not None
        return (
            active_selection,
            batch,
            replace(attempt, pnp_ms=pnp_ms),
            match_ms,
            evaluated,
        )

    def _track_stage_can_stop(
        self,
        selection: _CandidateSelection,
        attempt: _PoseAttempt,
        capture_stamp: float,
    ) -> bool:
        if attempt.reject_reason or attempt.result is None:
            return False
        if not self._pose_quality_ok(
            attempt.result,
            attempt.metrics,
            min_inliers=selection.min_inliers,
            reproj_limit=self.cfg.max_reproj_error_track,
        ):
            return False
        _rotation, center, yaw = self._pose_components(attempt.result)
        return self._trajectory_would_allow(
            center,
            yaw,
            capture_stamp,
            self.st.state,
        )

    def _match_track_map_first(
        self,
        gray: np.ndarray,
        selection: _CandidateSelection,
        capture_stamp: float,
        prepared_query,
    ) -> tuple[_CorrespondenceBatch, _PoseAttempt, float, bool]:
        map_selection = replace(selection, mode="edm_track")
        batch, attempt, match_ms = self._match_and_estimate(
            gray,
            map_selection,
            capture_stamp=capture_stamp,
            prepared_query=prepared_query,
        )
        if self._track_stage_can_stop(selection, attempt, capture_stamp):
            return batch, attempt, match_ms, False
        if batch.by_ref is None:
            raise ValueError("map-first TRACK requires independent map correspondences")
        expanded_started = time.perf_counter()
        temporal_rows = self.loc.correspondences_for_sources(
            gray,
            [self.temporal_gray],
            [self.temporal_xyz_by_cell],
            batch_size=self.cfg.match_batch_size,
            source_kinds=["temporal"],
            prepared_query=prepared_query,
        )
        rows = [*temporal_rows, *batch.by_ref]
        points2d, points3d, confidence, per_ref = _merge_correspondence_rows(rows)
        batch = _CorrespondenceBatch(
            points2d=points2d,
            points3d=points3d,
            confidence=confidence,
            per_ref=per_ref,
            by_ref=None,
            temporal_used=True,
        )
        fallback_attempt = self._best_pose_attempt(
            selection,
            batch,
            capture_stamp=capture_stamp,
        )
        match_ms += max(
            0.0,
            (time.perf_counter() - expanded_started) * 1e3 - fallback_attempt.pnp_ms,
        )
        fallback_attempt = replace(
            fallback_attempt,
            pnp_ms=attempt.pnp_ms + fallback_attempt.pnp_ms,
        )
        return batch, fallback_attempt, match_ms, True

    def _widen_track_refs_on_miss(
        self,
        gray: np.ndarray,
        selection: _CandidateSelection,
        capture_stamp: float,
        prepared_query,
        attempt: _PoseAttempt,
    ) -> tuple[_CandidateSelection, _CorrespondenceBatch, _PoseAttempt, float, dict] | None:
        """Retry a missed TRACK/WEAK frame against more map references, same frame.

        TRACK runs one map reference (212 census failures: refs=1, inliers p50
        34); WEAK runs three (83 census failures, refs=3, inliers p50 23 with 33
        frames sitting in [25, 30), one breath below ``weak_min_inliers``).
        This retry costs nothing on healthy frames -- it only runs after the
        frame already failed its state's quality gate -- and the wider attempt
        is kept only if it passes that same gate, so a miss can never be turned
        into a worse accept.

        Returns ``None`` when disabled, not applicable, or the retry did not
        produce an acceptable pose (the caller keeps the original attempt).
        """
        state = self.st.state
        if state == "WEAK_TRACK":
            topk = int(getattr(self.cfg, "weak_miss_widen_topk", 0) or 0)
            widen_mode = "edm_weak_widen"
        else:
            topk = int(getattr(self.cfg, "track_miss_widen_topk", 0) or 0)
            widen_mode = "edm_track_widen"
        if topk <= 0 or selection.acquiring or state not in ("TRACK", "WEAK_TRACK"):
            return None
        if attempt.reject_reason:
            return None
        reproj_limit = float(self.cfg.max_reproj_error_track)
        if self._pose_quality_ok(
            attempt.result,
            attempt.metrics,
            min_inliers=selection.min_inliers,
            reproj_limit=reproj_limit,
        ):
            return None
        refs = self._track_candidates(topk, capture_stamp)
        if len(refs) <= len(selection.refs):
            return None
        wide_selection = replace(selection, refs=refs, mode=widen_mode)
        started = time.perf_counter()
        batch, wide_attempt, match_ms = self._match_and_estimate(
            gray,
            wide_selection,
            capture_stamp=capture_stamp,
            prepared_query=prepared_query,
        )
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1e3 - wide_attempt.pnp_ms)
        self._track_widen_attempts = int(getattr(self, "_track_widen_attempts", 0)) + 1
        if wide_attempt.reject_reason or not self._pose_quality_ok(
            wide_attempt.result,
            wide_attempt.metrics,
            min_inliers=selection.min_inliers,
            reproj_limit=reproj_limit,
        ):
            return None
        _rotation, center, yaw = self._pose_components(wide_attempt.result)
        if not self._trajectory_would_allow(center, yaw, capture_stamp, self.st.state):
            return None
        self._track_widen_accepts = int(getattr(self, "_track_widen_accepts", 0)) + 1
        widen_info = {
            "refs": len(refs),
            "inliers_before": (
                None if attempt.result is None else int(attempt.result["num_inliers"])
            ),
            "inliers_after": int(wide_attempt.result["num_inliers"]),
            "match_ms": float(elapsed_ms),
        }
        return (
            wide_selection,
            batch,
            replace(wide_attempt, pnp_ms=attempt.pnp_ms + wide_attempt.pnp_ms),
            elapsed_ms,
            widen_info,
        )


    def _try_klt_bridge(
        self,
        gray: np.ndarray,
        frame_bgr: np.ndarray,
        capture_stamp: float,
        started: float,
    ) -> dict | None:
        """Carry the pose one frame with KLT instead of an EDM match.

        Active only when ``SFM_EDM_KLT_BRIDGE_INTERVAL=N`` (N>=2) and the
        tracker is in steady TRACK with a healthy KLT cache from the last EDM
        anchor. Returns an accepted-pose ``info`` dict on success, or ``None``
        to fall through to the full EDM path (keyframe, degraded state, or a
        failed / low-confidence KLT track). ``_track_klt_prior`` supplies its
        own reproj / max_jump / inlier-ratio gates; a rejection there clears the
        cache and forces EDM next frame.
        """
        n = int(getattr(self, "_klt_bridge_interval", 0) or 0)
        if n < 2 or self.st.state != "TRACK" or self.st.center is None:
            return None
        cap = n - 1
        max_consec = int(getattr(self, "_klt_bridge_max_consec", 0) or 0)
        if max_consec > 0:
            cap = min(cap, max_consec)
        if int(getattr(self, "_klt_bridge_run", 0)) >= cap:
            return None  # keyframe: re-anchor with a full EDM match
        # Only bridge out of a strong EDM anchor. P168's WEAK/LOST-prone zones
        # produce shaky anchors; bridging through them accumulates KLT drift the
        # yaw / jump gates then reject, tipping TRACK -> WEAK -> LOST.
        if int(getattr(self, "_klt_anchor_inliers", 0)) < max(
            2 * int(self.cfg.weak_min_inliers), _KLT_BRIDGE_ANCHOR_MIN_INLIERS
        ):
            return None
        if float(getattr(self, "_klt_anchor_ratio", 0.0)) < float(
            getattr(self, "_klt_bridge_min_ratio", 0.66)
        ):
            return None
        if not self._klt_prior_allowed(capture_stamp):
            return None
        lk_started = time.perf_counter()
        prior = self._track_klt_prior(gray, capture_stamp)
        lk_ms = (time.perf_counter() - lk_started) * 1e3
        if prior is None:
            self._klt_bridge_run = 0
            return None

        center = np.asarray(prior["center"], dtype=float)
        yaw = float(prior["yaw"])
        rotation = prior.get("R")
        previous_center = self.st.center
        previous_stamp = self.st.last_capture_stamp
        if previous_center is not None and previous_stamp is not None:
            dt = capture_stamp - previous_stamp
            if math.isfinite(dt) and dt > 1e-6:
                measured_velocity = (center - np.asarray(previous_center, dtype=float)) / dt
                self.st.velocity = _blend_velocity(self.st.velocity, measured_velocity, dt)
        step_norm = (
            float(np.linalg.norm(center - np.asarray(previous_center, dtype=float)))
            if previous_center is not None
            else 0.0
        )
        self.st.center, self.st.yaw = center, yaw
        self.st.last_capture_stamp = capture_stamp
        self.st.misses = 0
        self.st.lost_frames = 0
        self._klt_bridge_run = int(getattr(self, "_klt_bridge_run", 0)) + 1
        # Drift trip: if this bridge already looks shaky, re-anchor with EDM next
        # frame rather than compound the error into a WEAK/LOST + heavy recovery.
        if getattr(self, "_klt_bridge_drift_guard", True):
            reproj = float(prior.get("reproj_rms") or 0.0)
            ratio = float(prior.get("inlier_ratio") or 1.0)
            if (
                reproj > _KLT_BRIDGE_DRIFT_REPROJ_FACTOR * float(self.cfg.max_reproj_error_track)
                or step_norm > _KLT_BRIDGE_DRIFT_STEP_FACTOR * float(self.cfg.max_jump)
                or ratio < _KLT_BRIDGE_DRIFT_MIN_RATIO
            ):
                self._klt_bridge_run = cap  # force EDM keyframe next frame
        if prior.get("cam_from_world") is not None:
            self._store_visual_motion_cache(frame_bgr, prior["cam_from_world"])

        ref_names = [
            self.name_of[i]
            for i in (self.st.last_refs or [])
            if i in getattr(self, "name_of", {})
        ]
        info = {
            "frame": self.st.frame,
            "state_in": "TRACK",
            "state_out": "TRACK",
            "ok": True,
            "center": center,
            "yaw": yaw,
            "R": np.asarray(rotation, dtype=float) if rotation is not None else None,
            "pose_status": "KLT_BRIDGED",
            "candidate_mode": "klt_bridge",
            "bridge": True,
            "bridge_run": int(self._klt_bridge_run),
            "inliers": int(prior.get("inliers", 0) or 0),
            "reproj_rms": prior.get("reproj_rms"),
            "inlier_ratio": prior.get("inlier_ratio"),
            "n_corr": int(prior.get("tracked", 0) or 0),
            "refs": ref_names,
            "reference_count": len(ref_names),
            "requested_reference_count": 0,
            "vpr_ms": 0.0,
            "match_ms": lk_ms,
            "pnp_ms": None,
            "global_retrieval_calls": self.st.global_retrieval_calls,
            "total_ms": (time.perf_counter() - started) * 1e3,
        }
        controller = getattr(self, "pose_guided", None)
        if controller is not None and not info.get("bridge"):
            try:
                controller.maybe_update_anchor(
                    info,
                    timestamp=capture_stamp,
                    rotation_cam_from_world=info["R"],
                    center=center,
                    yaw=yaw,
                )
            except Exception:
                pass
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
        # Stage timers. vpr/match/pnp only cover ~73% of the worker's measured
        # core_wall (26.30 ms p50 vs 19.09 ms of instrumented stages, 1676
        # frames), so the rest was invisible to every Tier 1-3 gate. These four
        # plus the already-computed total_ms close that gap; they are
        # perf_counter reads on a path that already takes several of them.
        stage_t = time.perf_counter()
        gray = EDMMatcher.load_gray(frame_bgr)
        stage_gray_ms = (time.perf_counter() - stage_t) * 1e3
        try:
            self._klt_query_gray = gray
            self._klt_query_stamp = float(capture_stamp)
        except Exception:
            pass
        # ESEKF propagate: constant-velocity + covariance growth; dt clamped inside esekf
        try:
            esekf = getattr(self, "esekf", None)
            if esekf is not None and math.isfinite(float(capture_stamp)):
                vel = getattr(self, "_latest_velocity_ned", None)
                try:
                    esekf.predict(float(capture_stamp), velocity_ned=vel)
                    p = getattr(esekf, "p", None)
                    if p is not None:
                        self._esekf_last_predict = np.asarray(p, dtype=float).copy()
                except Exception:
                    pass
        except Exception:
            pass
        stage_t = time.perf_counter()
        bridge = self._try_klt_bridge(gray, frame_bgr, capture_stamp, t0)
        stage_bridge_ms = (time.perf_counter() - stage_t) * 1e3
        if bridge is not None:
            return bridge
        self._klt_bridge_run = 0
        matcher = getattr(self.loc, "matcher", None)
        prepare_query = getattr(matcher, "prepare_query", None)
        # This is a real backbone forward, and it sits outside match_ms.
        stage_t = time.perf_counter()
        prepared_query = prepare_query(gray) if callable(prepare_query) else None
        stage_query_ms = (time.perf_counter() - stage_t) * 1e3
        state_in = self.st.state
        stage_t = time.perf_counter()
        selection = self._select_candidates(frame_bgr, capture_stamp)
        # vpr_ms is a subset of this: MegaLoc runs inside candidate selection.
        stage_select_ms = (time.perf_counter() - stage_t) * 1e3
        requested = len(selection.refs)
        acquire_stage_counts = [requested]
        track_map_first = (
            cfg.track_map_first
            and selection.mode == "edm_temporal_map"
            and self.temporal_gray is not None
            and self.temporal_xyz_by_cell is not None
        )
        track_temporal_fallback = False
        if selection.acquiring:
            (
                selection,
                batch,
                attempt,
                match_ms,
                acquire_stage_counts,
            ) = self._match_acquisition_stages(
                gray,
                selection,
                capture_stamp,
                prepared_query,
            )
            if (
                self._adaptive_vpr_enabled()
                and state_in == "LOST"
                and selection.mode.startswith("megaloc")
            ):
                if attempt.strong_count > 0:
                    self._vpr_consec_failures = 0
                else:
                    self._vpr_consec_failures += 1
        elif track_map_first:
            batch, attempt, match_ms, track_temporal_fallback = self._match_track_map_first(
                gray,
                selection,
                capture_stamp,
                prepared_query,
            )
        else:
            batch, attempt, match_ms = self._match_and_estimate(
                gray,
                selection,
                capture_stamp=capture_stamp,
                prepared_query=prepared_query,
            )
        widen = self._widen_track_refs_on_miss(
            gray,
            selection,
            capture_stamp,
            prepared_query,
            attempt,
        )
        if widen is not None:
            selection, batch, attempt, widen_ms, widen_info = widen
            match_ms += widen_ms
        info = self._localization_info(
            selection,
            batch,
            attempt,
            match_ms=match_ms,
            started=t0,
        )
        info["stage_gray_ms"] = stage_gray_ms
        info["stage_bridge_ms"] = stage_bridge_ms
        info["stage_query_ms"] = stage_query_ms
        info["stage_select_ms"] = stage_select_ms
        info["capture_stamp"] = capture_stamp
        info["requested_reference_count"] = requested
        info["acquire_stage_counts"] = acquire_stage_counts
        info["staged_early_stop"] = acquire_stage_counts[-1] < requested
        info["track_map_first"] = track_map_first
        info["track_temporal_fallback"] = track_temporal_fallback
        info["pnp_ranked_batches"] = bool(cfg.pnp_ranked_batches)
        if widen is not None:
            info["track_widen"] = widen_info
        self._observe_lost_starvation(state_in, selection, info)
        ret = attempt.result
        min_inl = selection.min_inliers
        reproj_limit = (
            cfg.max_reproj_error_acquire if selection.acquiring else cfg.max_reproj_error_track
        )
        relaxed = self._relaxed_acquire_gate(selection, attempt)
        if relaxed is not None:
            min_inl, reproj_limit = relaxed
            info["acquire_relaxed"] = {
                "min_inliers": int(min_inl),
                "reproj_limit": float(reproj_limit),
                "agreeing_refs": int(attempt.relaxed_agree_count),
            }
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
            prepared_query,
        )

    def _is_weak_hysteresis_candidate(self, info: dict) -> bool:
        """Check if a miss qualifies for the soft hysteresis buffer [45, 50)."""
        # Default ON (2026-09-03 gate: P167 +53/+58 over two seeds, P117/P168
        # flat): set SFM_EDM_WEAK_HYSTERESIS=0 to restore legacy one-miss drop.
        if (
            (os.environ.get("SFM_EDM_WEAK_HYSTERESIS", "1") or "1").strip().lower()
            in ("0", "false", "no", "off")
        ):
            return False
        if self.st.state != "TRACK":
            return False
        inliers = info.get("inliers")
        if inliers is None:
            return False
        try:
            inliers_val = int(inliers)
        except (ValueError, TypeError):
            return False
        track_min = getattr(self.cfg, "track_min_inliers", 50)
        soft_min = min(_WEAK_HYSTERESIS_INLIER_MIN, max(0, track_min - 5))
        if not (soft_min <= inliers_val < track_min):
            return False
        reproj = info.get("reproj_rms")
        if reproj is None:
            return False
        try:
            reproj_val = float(reproj)
        except (ValueError, TypeError):
            return False
        if not (math.isfinite(reproj_val) and reproj_val <= _WEAK_HYSTERESIS_REPROJ_MAX):
            return False
        if info.get("rejected") in ("jump", "track_yaw", "acquire_jump", "acquire_yaw"):
            return False
        if info.get("step_legal") is False:
            return False
        if "step_norm" in info:
            try:
                if float(info["step_norm"]) > float(self.cfg.max_jump):
                    return False
            except (ValueError, TypeError):
                pass
        return True

    def _on_miss(self, info: dict):
        if self._is_weak_hysteresis_candidate(info):
            self.st.misses += 0.5
            info["weak_hysteresis_buffered"] = True
        else:
            self.st.misses += 1.0
        if self.st.state == "TRACK" and self.st.misses >= self.cfg.weak_after:
            self.st.state = "WEAK_TRACK"
            self.st.misses = float(self.cfg.weak_after)
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
            self._vpr_consec_failures = 0
            self._last_vpr_lost_frame = None
            self._clear_visual_motion_cache()
            try:
                self._clear_klt_cache()
            except Exception:
                pass
        if self.st.state == "LOST":
            self.st.lost_frames += 1
        else:
            self.st.lost_frames = 0
        info.update({"state_out": self.st.state, "ok": False})
        if info.get("pose_status") in _VISUAL_POSE_STATUSES:
            info["pose_status"] = "NONE"
        controller = getattr(self, "pose_guided", None)
        if controller is not None and info.get("pose_status") != "VISUALLY_CONFIRMED":
            predicted = controller.predicted_only(controller.last_prediction)
            predicted["ok"] = False
            info.update(predicted)
        # ESEKF PREDICTED_ONLY supersedes KLT/pose_guided when prediction_allowed (cov trace<1.0, yaw sigma<15deg, age<6)
        esekf_done = False
        try:
            if getattr(self.st, "state", None) in {"TRACK", "WEAK_TRACK"}:
                esekf = getattr(self, "esekf", None)
                if esekf is not None and callable(getattr(esekf, "prediction_allowed", None)) and esekf.prediction_allowed():
                    p_pred = getattr(esekf, "p", None)
                    q_pred = getattr(esekf, "q", None)
                    if p_pred is not None and q_pred is not None:
                        p_arr = np.asarray(p_pred, dtype=float).reshape(3)
                        if p_arr.shape == (3,) and np.all(np.isfinite(p_arr)):
                            yaw_val = self._esekf_yaw()
                            if yaw_val is None:
                                yaw_val = getattr(self.st, "yaw", None)
                            info["ok"] = False
                            info["pose_status"] = "PREDICTED_ONLY"
                            info["prediction_valid"] = True
                            info["prediction_mode"] = "esekf"
                            info["prediction_source"] = "esekf"
                            info["predicted_center"] = p_arr.astype(float).tolist()
                            info["predicted_yaw"] = float(yaw_val) if yaw_val is not None and math.isfinite(float(yaw_val)) else None
                            try:
                                P = getattr(esekf, "P", None)
                                if P is not None and hasattr(P, "shape") and P.shape == (15, 15):
                                    info["esekf_pos_trace"] = float(np.trace(P[0:3, 0:3]))
                            except Exception:
                                pass
                            esekf_done = True
        except Exception:
            esekf_done = False
        klt_predict_states_ok = (
            getattr(self, "st", None) is not None
            and (
                getattr(self.st, "state", None) != "LOST"
                or getattr(self, "_klt_lost_predict", False)
            )
        )
        if not esekf_done and klt_predict_states_ok:
            try:
                prior = self._track_klt_prior(
                    getattr(self, "_klt_query_gray", None),
                    getattr(self, "_klt_query_stamp", None) if getattr(self, "_klt_query_stamp", None) is not None else 0.0,
                )
            except Exception:
                prior = None
            if prior is not None:
                try:
                    center = prior.get("center")
                    yaw = prior.get("yaw")
                    tracked = prior.get("tracked")
                    inliers = prior.get("inliers")
                    info["ok"] = False
                    info["pose_status"] = "PREDICTED_ONLY"
                    info["prediction_valid"] = True
                    info["prediction_mode"] = "klt_pnp"
                    info["prediction_source"] = "klt_inlier_pnp"
                    info["predicted_center"] = np.asarray(center, dtype=float).tolist() if center is not None else None
                    info["predicted_yaw"] = float(yaw) if yaw is not None else None
                    info["klt_tracked"] = int(tracked) if tracked is not None else None
                    info["klt_inliers"] = int(inliers) if inliers is not None else None
                    info["prediction_state"] = str(getattr(self.st, "state", ""))
                    info["klt_age"] = int(getattr(self, "_klt_age", 0))
                except Exception:
                    pass

if __name__ == "__main__":
    print(__doc__)
