"""Validated EDM matcher/tracker deployment profiles."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path


EDM_PROFILE_SCHEMA = "edm-deployment-profile/v1"
EDM_REQUIRED_MATCHER_KEYS = {
    "coarse_topk",
    "mconf_thr",
    "fp16",
    "input_size",
    "reference_cache_size",
}
EDM_REQUIRED_TRACKER_KEYS = {
    "global_retrieval_policy",
    "boot_global_topk",
    "acquire_initial_topk",
    "acquire_min_inliers",
    "match_batch_size",
    "use_temporal_reference",
    "temporal_map_topk",
    "local_topk",
    "weak_local_topk",
    "lost_local_topk",
    "lost_local_grace_frames",
    "recovery_bank_size",
    "recovery_scan_topk",
    "near_pool",
    "covis_per_ref",
    "radius",
    "max_yaw_diff_deg",
    "track_min_inliers",
    "weak_min_inliers",
    "max_reproj_error_acquire",
    "max_reproj_error_track",
    "pnp_ransac_max_error",
    "max_jump",
    "prediction_max_dt",
    "adaptive_jump_factor",
    "adaptive_jump_floor",
    "adaptive_jump_bootstrap",
    "adaptive_jump_ceiling",
    "adaptive_jump_min_history",
    "adaptive_jump_history_size",
    "weak_after",
    "lost_after",
    "max_corr_total",
    "corr_grid",
}


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_edm_production_profile(path: str | Path) -> dict:
    """Load the EDM runtime contract before allocating any model."""
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise ValueError(f"cannot read EDM production profile {source}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid EDM production profile JSON {source}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != EDM_PROFILE_SCHEMA:
        raise ValueError(
            f"EDM production profile schema must be {EDM_PROFILE_SCHEMA}: {source}"
        )
    matcher = raw.get("matcher")
    tracker = raw.get("tracker")
    if not isinstance(matcher, dict) or not isinstance(tracker, dict):
        raise ValueError(f"EDM production profile needs matcher and tracker objects: {source}")
    missing_matcher = sorted(EDM_REQUIRED_MATCHER_KEYS - matcher.keys())
    unknown_matcher = sorted(set(matcher) - EDM_REQUIRED_MATCHER_KEYS)
    missing_tracker = sorted(EDM_REQUIRED_TRACKER_KEYS - tracker.keys())
    unknown_tracker = sorted(set(tracker) - EDM_REQUIRED_TRACKER_KEYS)
    if missing_matcher or unknown_matcher or missing_tracker or unknown_tracker:
        raise ValueError(
            f"incomplete EDM production profile {source}: "
            f"matcher_missing={missing_matcher}, matcher_unknown={unknown_matcher}, "
            f"tracker_missing={missing_tracker}, tracker_unknown={unknown_tracker}"
        )
    coarse_topk = matcher["coarse_topk"]
    mconf_thr = matcher["mconf_thr"]
    if isinstance(coarse_topk, bool) or not isinstance(coarse_topk, int) or coarse_topk <= 0:
        raise ValueError(f"EDM matcher coarse_topk must be a positive integer: {source}")
    if (isinstance(mconf_thr, bool) or not isinstance(mconf_thr, (int, float))
            or not math.isfinite(float(mconf_thr)) or not 0.0 <= float(mconf_thr) <= 1.0):
        raise ValueError(f"EDM matcher mconf_thr must be within [0,1]: {source}")
    if not isinstance(matcher["fp16"], bool):
        raise ValueError(f"EDM matcher fp16 must be boolean: {source}")
    if matcher["input_size"] != [1024, 576]:
        raise ValueError(
            f"EDM matcher input_size must be [1024, 576] for this deployment: {source}"
        )
    cache_size = matcher["reference_cache_size"]
    if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
        raise ValueError(f"EDM matcher reference_cache_size must be non-negative: {source}")
    for name, value in tracker.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"EDM tracker {name} must be finite: {source}")
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError(f"EDM tracker {name} has an unsupported value type: {source}")
    return raw


def apply_edm_tracker_profile(cfg, profile: dict) -> None:
    """Apply declared fields and run the selected deployment's complete validator."""
    tracker = profile["tracker"]
    allowed = {field.name for field in dataclasses.fields(cfg)}
    unknown = sorted(set(tracker) - allowed)
    missing = sorted(allowed - set(tracker))
    if unknown or missing:
        raise ValueError(
            "EDM production profile is incompatible with the selected deployment; "
            f"unknown={unknown}, missing={missing}"
        )
    for name, value in tracker.items():
        setattr(cfg, name, value)
    validate = getattr(cfg, "validate", None)
    if not callable(validate):
        raise ValueError("selected EDMConfig does not expose validate()")
    validate()
