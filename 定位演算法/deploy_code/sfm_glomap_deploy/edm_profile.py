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
EDM_OPTIONAL_MATCHER_KEYS = {
    "runtime_sigma_mode",
    "temporal_feature_cache_size",
}
EDM_RUNTIME_SIGMA_MODES = {"bidirectional", "reference_grid"}
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
    "min_inlier_ratio",
    "min_inlier_grid_cells",
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
    # LOST re-acquisition bound. Scale-FREE (a factor of max_jump, degrees, seconds),
    # so unlike radius/max_jump these carry the same value at every site.
    "acquire_max_jump_factor",
    "acquire_max_yaw_diff_deg",
    "lost_prior_max_age_s",
    "max_corr_total",
    "corr_grid",
}
EDM_OPTIONAL_TRACKER_KEYS = {
    # Added compatibly to v1: omitted legacy profiles keep one-shot LOST retrieval.
    "lost_global_retrieval_interval",
    "pose_consensus_mode",
    "consensus_max_rotation_deg",
    "reference_quality_weight",
    "reference_quality_floor",
    "acquire_stage_mode",
    "lost_prior_strategy",
    "lost_prior_fusion_weight",
}
EDM_REQUIRED_REPOSED_KEYS = {
    "mode",
    "model_path",
    "model_sha256",
    "num_tokens",
    "max_matches",
    "min_inliers",
    "max_rotation_delta_deg",
    "max_translation_direction_delta_deg",
}
EDM_OPTIONAL_REPOSED_KEYS = {
    "match_grid",
    "min_inlier_ratio",
    "min_spatial_support",
}
EDM_POSE_CONSENSUS_MODES = {"pairwise", "cluster"}
EDM_ACQUIRE_STAGE_MODES = {"full_set", "initial_topk"}
EDM_LOST_PRIOR_STRATEGIES = {"restrict_nearby", "full_global", "score_fusion"}
EDM_REPOSED_MODES = {"off", "shadow", "confirm_limited_jump"}



def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _validate_edm_profile_sections(raw: object, source: Path) -> tuple[dict, dict]:
    if not isinstance(raw, dict) or raw.get("schema") != EDM_PROFILE_SCHEMA:
        raise ValueError(
            f"EDM production profile schema must be {EDM_PROFILE_SCHEMA}: {source}"
        )
    matcher = raw.get("matcher")
    tracker = raw.get("tracker")
    if not isinstance(matcher, dict) or not isinstance(tracker, dict):
        raise ValueError(f"EDM production profile needs matcher and tracker objects: {source}")
    return matcher, tracker


def _validate_edm_profile_keys(matcher: dict, tracker: dict, source: Path) -> None:
    missing_matcher = sorted(EDM_REQUIRED_MATCHER_KEYS - matcher.keys())
    unknown_matcher = sorted(
        set(matcher) - EDM_REQUIRED_MATCHER_KEYS - EDM_OPTIONAL_MATCHER_KEYS
    )
    missing_tracker = sorted(EDM_REQUIRED_TRACKER_KEYS - tracker.keys())
    unknown_tracker = sorted(
        set(tracker) - EDM_REQUIRED_TRACKER_KEYS - EDM_OPTIONAL_TRACKER_KEYS
    )
    if missing_matcher or unknown_matcher or missing_tracker or unknown_tracker:
        raise ValueError(
            f"incomplete EDM production profile {source}: "
            f"matcher_missing={missing_matcher}, matcher_unknown={unknown_matcher}, "
            f"tracker_missing={missing_tracker}, tracker_unknown={unknown_tracker}"
        )


def _validate_edm_matcher_profile(matcher: dict, source: Path) -> None:
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
    sigma_mode = matcher.get("runtime_sigma_mode")
    if sigma_mode is not None and sigma_mode not in EDM_RUNTIME_SIGMA_MODES:
        raise ValueError(
            "EDM matcher runtime_sigma_mode must be "
            f"'bidirectional' or 'reference_grid': {source}"
        )
    temporal_cache = matcher.get("temporal_feature_cache_size")
    if temporal_cache is not None and (
        isinstance(temporal_cache, bool)
        or not isinstance(temporal_cache, int)
        or temporal_cache < 0
    ):
        raise ValueError(
            f"EDM matcher temporal_feature_cache_size must be a non-negative integer: {source}"
        )


def _is_strict_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_positive_int(value: object) -> bool:
    return _is_strict_int(value) and value > 0


def _is_nonneg_int(value: object) -> bool:
    return _is_strict_int(value) and value >= 0


def _is_positive_finite(value: object) -> bool:
    return _is_finite_number(value) and float(value) > 0.0


def _is_nonnegative_finite(value: object) -> bool:
    return _is_finite_number(value) and float(value) >= 0.0


def _is_unit_interval(value: object) -> bool:
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0


def _validate_edm_tracker_scalars(tracker: dict, source: Path) -> None:
    for name, value in tracker.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"EDM tracker {name} must be finite: {source}")
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError(f"EDM tracker {name} has an unsupported value type: {source}")


_EDM_TRACKER_CHOICES = (
    (
        "pose_consensus_mode",
        EDM_POSE_CONSENSUS_MODES,
        "EDM tracker pose_consensus_mode must be 'pairwise' or 'cluster': ",
    ),
    (
        "acquire_stage_mode",
        EDM_ACQUIRE_STAGE_MODES,
        "EDM tracker acquire_stage_mode must be 'full_set' or 'initial_topk': ",
    ),
    (
        "lost_prior_strategy",
        EDM_LOST_PRIOR_STRATEGIES,
        "EDM tracker lost_prior_strategy must be "
        "'restrict_nearby', 'full_global', or 'score_fusion': ",
    ),
)


def _validate_edm_tracker_choices(tracker: dict, source: Path) -> None:
    for name, allowed, prefix in _EDM_TRACKER_CHOICES:
        value = tracker.get(name)
        if value is not None and value not in allowed:
            raise ValueError(f"{prefix}{source}")


_EDM_TRACKER_NUMBER_CHECKS = (
    (
        ("consensus_max_rotation_deg", "lost_prior_fusion_weight"),
        _is_positive_finite,
        "must be finite and > 0",
    ),
    (
        ("reference_quality_weight", "reference_quality_floor"),
        _is_nonnegative_finite,
        "must be finite and >= 0",
    ),
)


def _validate_named_optional_numbers(
    values: dict,
    names: tuple[str, ...],
    ok,
    message: str,
    source: Path,
    kind: str,
) -> None:
    for name in names:
        if name not in values:
            continue
        if not ok(values[name]):
            raise ValueError(f"{kind} {name} {message}: {source}")


def _validate_edm_tracker_numbers(tracker: dict, source: Path) -> None:
    for names, ok, message in _EDM_TRACKER_NUMBER_CHECKS:
        _validate_named_optional_numbers(
            tracker, names, ok, message, source, "EDM tracker"
        )


def _validate_edm_tracker_profile(tracker: dict, source: Path) -> None:
    _validate_edm_tracker_scalars(tracker, source)
    _validate_edm_tracker_choices(tracker, source)
    _validate_edm_tracker_numbers(tracker, source)


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
    matcher, tracker = _validate_edm_profile_sections(raw, source)
    _validate_edm_profile_keys(matcher, tracker, source)
    _validate_edm_matcher_profile(matcher, source)
    _validate_edm_tracker_profile(tracker, source)
    _validate_edm_reposed_profile(raw.get("reposed"), source)
    return raw


def apply_edm_tracker_profile(cfg, profile: dict) -> None:
    """Apply declared fields and run the selected deployment's complete validator."""
    tracker = profile["tracker"]
    fields = {field.name: field for field in dataclasses.fields(cfg)}
    allowed = set(fields)
    unknown = sorted(set(tracker) - allowed)
    missing = sorted(allowed - EDM_OPTIONAL_TRACKER_KEYS - set(tracker))
    unsupported_optional = sorted(EDM_OPTIONAL_TRACKER_KEYS - allowed)
    if unknown or missing or unsupported_optional:
        raise ValueError(
            "EDM production profile is incompatible with the selected deployment; "
            f"unknown={unknown}, missing={missing}, "
            f"unsupported_optional={unsupported_optional}"
        )
    for name in EDM_OPTIONAL_TRACKER_KEYS - set(tracker):
        field = fields[name]
        if field.default is dataclasses.MISSING:
            raise ValueError(f"optional EDM tracker field {name} has no default")
        setattr(cfg, name, field.default)
    for name, value in tracker.items():
        setattr(cfg, name, value)
    validate = getattr(cfg, "validate", None)
    if not callable(validate):
        raise ValueError("selected EDMConfig does not expose validate()")
    validate()


def resolve_reposed_model_path(model_path: str | Path, source: Path | None = None) -> Path:
    """Resolve a profile model path against the profile file and workspace roots."""
    raw = Path(model_path).expanduser()
    if raw.is_file():
        return raw.resolve()
    candidates: list[Path] = []
    if source is not None:
        candidates.append(source.parent / raw)
    if not raw.is_absolute():
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (source.parent / raw).resolve() if source is not None else raw.resolve()


def _validate_edm_reposed_shape(reposed: object, source: Path) -> dict:
    if not isinstance(reposed, dict):
        raise ValueError(f"EDM production profile reposed must be an object: {source}")
    missing = sorted(EDM_REQUIRED_REPOSED_KEYS - reposed.keys())
    unknown = sorted(
        set(reposed) - EDM_REQUIRED_REPOSED_KEYS - EDM_OPTIONAL_REPOSED_KEYS
    )
    if missing or unknown:
        raise ValueError(
            f"incomplete EDM reposed profile {source}: "
            f"missing={missing}, unknown={unknown}"
        )
    return reposed


_EDM_REPOSED_LIMIT_CHECKS = (
    ("max_matches", True, _is_positive_int, "must be a positive integer"),
    ("min_inliers", True, _is_positive_int, "must be a positive integer"),
    (
        "max_rotation_delta_deg",
        True,
        _is_positive_finite,
        "must be finite and > 0",
    ),
    (
        "max_translation_direction_delta_deg",
        True,
        _is_positive_finite,
        "must be finite and > 0",
    ),
    ("match_grid", False, _is_positive_int, "must be a positive integer"),
    (
        "min_spatial_support",
        False,
        _is_nonneg_int,
        "must be a non-negative integer",
    ),
    ("min_inlier_ratio", False, _is_unit_interval, "must be within [0, 1]"),
)


def _validate_edm_reposed_limit_table(reposed: dict, source: Path) -> None:
    for name, required, ok, message in _EDM_REPOSED_LIMIT_CHECKS:
        if not required and name not in reposed:
            continue
        if not ok(reposed[name]):
            raise ValueError(f"EDM reposed {name} {message}: {source}")


def _validate_edm_reposed_limits(reposed: dict, source: Path) -> None:
    num_tokens = reposed["num_tokens"]
    if not _is_strict_int(num_tokens) or num_tokens != 1200:
        raise ValueError(f"EDM reposed num_tokens must be 1200: {source}")
    _validate_edm_reposed_limit_table(reposed, source)


def _validate_edm_reposed_model(reposed: dict, source: Path) -> None:
    digest = reposed["model_sha256"]
    if not isinstance(digest, str) or len(digest.strip()) != 64:
        raise ValueError(f"EDM reposed model_sha256 must be a 64-character digest: {source}")
    model_path = resolve_reposed_model_path(reposed["model_path"], source)
    if not model_path.is_file():
        raise ValueError(f"EDM reposed model file does not exist: {model_path}")
    from reposed_motion_validator import verify_model_sha256

    verify_model_sha256(model_path, digest)


def _validate_edm_reposed_profile(reposed: object, source: Path) -> None:
    if reposed is None:
        return
    reposed = _validate_edm_reposed_shape(reposed, source)
    mode = reposed["mode"]
    if mode not in EDM_REPOSED_MODES:
        raise ValueError(
            f"EDM reposed mode must be one of {sorted(EDM_REPOSED_MODES)}: {source}"
        )
    if mode == "off":
        return
    _validate_edm_reposed_limits(reposed, source)
    _validate_edm_reposed_model(reposed, source)
