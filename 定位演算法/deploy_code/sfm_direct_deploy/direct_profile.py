"""Validated deployment profile for the ``direct`` localization backend.

The production path reads **no** environment variable for anything that can
move a pose.  Every number the two-rate tracker and the relocalizer use comes
from one SHA-256 bound ``direct_localizer_profile.json`` (schema
``direct-deployment-profile/v1``), whose values are the frozen P174
configuration.

Validation is fail-closed in both directions, like
``sfm_glomap_deploy/edm_profile.py``: a missing required key is an error, and so
is an unknown key.  A knob nobody validates is a knob nobody controls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from direct_paths import verify_file_sha256


DIRECT_PROFILE_SCHEMA = "direct-deployment-profile/v1"

_TOP_KEYS = frozenset(
    {"schema", "name", "map_scale", "intrinsics", "reloc", "fast_loop", "vo", "dead_reckon"}
)
_OPTIMIZATION_KEYS = frozenset({"adaptive_retrieval", "spatial_selection", "gpu_cache_size"})
_INTRINSICS_KEYS = frozenset(
    {"schema_version", "image_width", "image_height", "K", "images_are_undistorted"}
)
_RELOC_KEYS = frozenset(
    {
        "top_k",
        "lift_distance_px",
        "period_s",
        "min_points",
        "reference_bank",
        "min_reference_occupied_bins",
        "bank_occupancy_percentile",
        "descriptor_batch_size",
        "batch_refs",
        "reference_cache_size",
        "frozen_pnp_thresholds",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "strong_inliers",
        "minimum_inlier_ratio",
        "minimum_hull_coverage",
        "minimum_occupancy_4x4",
        "minimum_positive_depth_ratio",
        "maximum_reprojection_p90_px",
    }
)
_FAST_LOOP_KEYS = frozenset(
    {"resolution", "tracker", "track_cap", "reseed_min_points", "topup", "klt", "pnp"}
)
_KLT_KEYS = frozenset({"fb_max_px", "win", "levels"})
_PNP_KEYS = frozenset(
    {
        "max_error_px",
        "min_trials",
        "max_trials",
        "confidence",
        "refine_iters",
        "min_points",
        "min_inliers",
    }
)
_VO_KEYS = frozenset(
    {
        "enabled",
        "keyframe_stride",
        "batch_lag",
        "min_live",
        "detect_cap",
        "min_distance",
        "min_parallax_deg",
        "max_reproj_px",
        "refine",
        "window_ba",
    }
)
_DEAD_RECKON_KEYS = frozenset({"enabled", "max_frames", "max_age_s"})

SUPPORTED_TRACKERS = frozenset({"klt"})


class DirectProfileError(ValueError):
    """Raised when a direct deployment profile violates its frozen contract."""


def _reject_json_constant(value: str):
    raise DirectProfileError(f"non-finite JSON number is not allowed: {value}")


def _section(
    raw: Mapping[str, Any], name: str, allowed: frozenset[str], source: Path,
    *, optional: frozenset[str] = frozenset(),
) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise DirectProfileError(f"direct profile section {name!r} must be an object: {source}")
    missing = sorted(allowed - optional - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        raise DirectProfileError(f"direct profile {name} is missing {missing}: {source}")
    if unknown:
        raise DirectProfileError(f"direct profile {name} has unknown keys {unknown}: {source}")
    return dict(value)


def _is_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _positive_int(section: str, name: str, value: object, source: Path) -> int:
    if not _is_int(value) or value <= 0:
        raise DirectProfileError(f"{section}.{name} must be a positive integer: {source}")
    return int(value)


def _nonnegative_int(section: str, name: str, value: object, source: Path) -> int:
    if not _is_int(value) or value < 0:
        raise DirectProfileError(f"{section}.{name} must be a non-negative integer: {source}")
    return int(value)


def _positive_float(section: str, name: str, value: object, source: Path) -> float:
    if not _is_number(value) or not np.isfinite(float(value)) or float(value) <= 0.0:
        raise DirectProfileError(f"{section}.{name} must be a positive finite number: {source}")
    return float(value)


def _unit_float(section: str, name: str, value: object, source: Path) -> float:
    if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
        raise DirectProfileError(f"{section}.{name} must lie in [0, 1]: {source}")
    return float(value)


def _boolean(section: str, name: str, value: object, source: Path) -> bool:
    if not isinstance(value, bool):
        raise DirectProfileError(f"{section}.{name} must be a boolean: {source}")
    return value


@dataclass(frozen=True)
class DirectIntrinsics:
    """Undistorted PINHOLE calibration of the declared stream resolution."""

    schema_version: int
    image_width: int
    image_height: int
    K: np.ndarray
    images_are_undistorted: bool

    def as_calibration(self) -> dict:
        """The exact mapping ``scaled_pinhole_parameters`` consumes."""

        return {
            "schema_version": self.schema_version,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "K": self.K.tolist(),
            "images_are_undistorted": self.images_are_undistorted,
        }


@dataclass(frozen=True)
class DirectKLT:
    fb_max_px: float
    win: int
    levels: int


@dataclass(frozen=True)
class DirectFastPnP:
    max_error_px: float
    min_trials: int
    max_trials: int
    confidence: float
    refine_iters: int
    min_points: int
    min_inliers: int


@dataclass(frozen=True)
class DirectFastLoop:
    resolution: tuple[int, int]
    tracker: str
    track_cap: int
    reseed_min_points: int
    topup: bool
    klt: DirectKLT
    pnp: DirectFastPnP


@dataclass(frozen=True)
class DirectReloc:
    top_k: int
    lift_distance_px: float
    period_s: float
    min_points: int
    reference_bank: str
    min_reference_occupied_bins: int
    bank_occupancy_percentile: int
    descriptor_batch_size: int
    batch_refs: bool
    reference_cache_size: int
    frozen_pnp_thresholds: Mapping[str, Any]


@dataclass(frozen=True)
class DirectVO:
    enabled: bool
    keyframe_stride: int
    batch_lag: int
    min_live: int
    detect_cap: int
    min_distance: int
    min_parallax_deg: float
    max_reproj_px: float
    refine: bool
    window_ba: bool


@dataclass(frozen=True)
class DirectDeadReckon:
    enabled: bool
    max_frames: int
    max_age_s: float = 10.0


@dataclass(frozen=True)
class DirectOptimizations:
    """Optional, SHA-bound experiments; legacy releases retain their policy."""

    adaptive_retrieval: bool = False
    spatial_selection: bool = False
    gpu_cache_size: int = 0


@dataclass(frozen=True)
class DirectProfile:
    """One frozen, SHA-bound production configuration of the direct backend."""

    name: str
    map_scale: float
    intrinsics: DirectIntrinsics
    reloc: DirectReloc
    fast_loop: DirectFastLoop
    vo: DirectVO
    dead_reckon: DirectDeadReckon
    raw: Mapping[str, Any]
    sha256: str
    source: Path
    optimizations: DirectOptimizations = field(default_factory=DirectOptimizations)


def _parse_intrinsics(raw: Mapping[str, Any], source: Path) -> DirectIntrinsics:
    section = _section(raw, "intrinsics", _INTRINSICS_KEYS, source)
    width = _positive_int("intrinsics", "image_width", section["image_width"], source)
    height = _positive_int("intrinsics", "image_height", section["image_height"], source)
    if section["schema_version"] != 1:
        raise DirectProfileError(f"intrinsics.schema_version must be 1: {source}")
    if section["images_are_undistorted"] is not True:
        raise DirectProfileError(f"direct localization requires undistorted images: {source}")
    matrix = np.asarray(section["K"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise DirectProfileError(f"intrinsics.K must be a finite 3x3 matrix: {source}")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise DirectProfileError(f"intrinsics.K focal lengths must be positive: {source}")
    matrix.flags.writeable = False
    return DirectIntrinsics(
        schema_version=1,
        image_width=width,
        image_height=height,
        K=matrix,
        images_are_undistorted=True,
    )


def _parse_thresholds(raw: Mapping[str, Any], source: Path) -> Mapping[str, Any]:
    thresholds = _section(raw, "frozen_pnp_thresholds", _THRESHOLD_KEYS, source)
    _positive_int("reloc.frozen_pnp_thresholds", "strong_inliers", thresholds["strong_inliers"], source)
    _positive_int(
        "reloc.frozen_pnp_thresholds",
        "minimum_occupancy_4x4",
        thresholds["minimum_occupancy_4x4"],
        source,
    )
    for name in ("minimum_inlier_ratio", "minimum_hull_coverage", "minimum_positive_depth_ratio"):
        _unit_float("reloc.frozen_pnp_thresholds", name, thresholds[name], source)
    _positive_float(
        "reloc.frozen_pnp_thresholds",
        "maximum_reprojection_p90_px",
        thresholds["maximum_reprojection_p90_px"],
        source,
    )
    return MappingProxyType(dict(thresholds))


def _parse_reloc(raw: Mapping[str, Any], source: Path) -> DirectReloc:
    section = _section(raw, "reloc", _RELOC_KEYS, source)
    bank = section["reference_bank"]
    if not isinstance(bank, str) or not bank:
        raise DirectProfileError(f"reloc.reference_bank must be a non-empty string: {source}")
    return DirectReloc(
        top_k=_positive_int("reloc", "top_k", section["top_k"], source),
        lift_distance_px=_positive_float("reloc", "lift_distance_px", section["lift_distance_px"], source),
        period_s=_positive_float("reloc", "period_s", section["period_s"], source),
        min_points=_positive_int("reloc", "min_points", section["min_points"], source),
        reference_bank=bank,
        min_reference_occupied_bins=_nonnegative_int(
            "reloc", "min_reference_occupied_bins", section["min_reference_occupied_bins"], source
        ),
        bank_occupancy_percentile=_nonnegative_int(
            "reloc", "bank_occupancy_percentile", section["bank_occupancy_percentile"], source
        ),
        descriptor_batch_size=_positive_int(
            "reloc", "descriptor_batch_size", section["descriptor_batch_size"], source
        ),
        batch_refs=_boolean("reloc", "batch_refs", section["batch_refs"], source),
        reference_cache_size=_positive_int(
            "reloc", "reference_cache_size", section["reference_cache_size"], source
        ),
        frozen_pnp_thresholds=_parse_thresholds(section, source),
    )


def _parse_fast_loop(raw: Mapping[str, Any], source: Path) -> DirectFastLoop:
    section = _section(raw, "fast_loop", _FAST_LOOP_KEYS, source)
    resolution = section["resolution"]
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(_is_int(value) and value > 0 for value in resolution)
    ):
        raise DirectProfileError(f"fast_loop.resolution must be two positive integers: {source}")
    tracker = section["tracker"]
    if tracker not in SUPPORTED_TRACKERS:
        raise DirectProfileError(
            f"fast_loop.tracker must be one of {sorted(SUPPORTED_TRACKERS)}: {source}"
        )
    klt = _section(section, "klt", _KLT_KEYS, source)
    pnp = _section(section, "pnp", _PNP_KEYS, source)
    return DirectFastLoop(
        resolution=(int(resolution[0]), int(resolution[1])),
        tracker=str(tracker),
        track_cap=_positive_int("fast_loop", "track_cap", section["track_cap"], source),
        reseed_min_points=_positive_int(
            "fast_loop", "reseed_min_points", section["reseed_min_points"], source
        ),
        topup=_boolean("fast_loop", "topup", section["topup"], source),
        klt=DirectKLT(
            fb_max_px=_positive_float("fast_loop.klt", "fb_max_px", klt["fb_max_px"], source),
            win=_positive_int("fast_loop.klt", "win", klt["win"], source),
            levels=_positive_int("fast_loop.klt", "levels", klt["levels"], source),
        ),
        pnp=DirectFastPnP(
            max_error_px=_positive_float("fast_loop.pnp", "max_error_px", pnp["max_error_px"], source),
            min_trials=_positive_int("fast_loop.pnp", "min_trials", pnp["min_trials"], source),
            max_trials=_positive_int("fast_loop.pnp", "max_trials", pnp["max_trials"], source),
            confidence=_unit_float("fast_loop.pnp", "confidence", pnp["confidence"], source),
            refine_iters=_positive_int("fast_loop.pnp", "refine_iters", pnp["refine_iters"], source),
            min_points=_positive_int("fast_loop.pnp", "min_points", pnp["min_points"], source),
            min_inliers=_positive_int("fast_loop.pnp", "min_inliers", pnp["min_inliers"], source),
        ),
    )


def _parse_vo(raw: Mapping[str, Any], source: Path) -> DirectVO:
    section = _section(raw, "vo", _VO_KEYS, source)
    return DirectVO(
        enabled=_boolean("vo", "enabled", section["enabled"], source),
        keyframe_stride=_positive_int("vo", "keyframe_stride", section["keyframe_stride"], source),
        batch_lag=_positive_int("vo", "batch_lag", section["batch_lag"], source),
        min_live=_positive_int("vo", "min_live", section["min_live"], source),
        detect_cap=_positive_int("vo", "detect_cap", section["detect_cap"], source),
        min_distance=_positive_int("vo", "min_distance", section["min_distance"], source),
        min_parallax_deg=_positive_float("vo", "min_parallax_deg", section["min_parallax_deg"], source),
        max_reproj_px=_positive_float("vo", "max_reproj_px", section["max_reproj_px"], source),
        refine=_boolean("vo", "refine", section["refine"], source),
        window_ba=_boolean("vo", "window_ba", section["window_ba"], source),
    )


def _parse_dead_reckon(raw: Mapping[str, Any], source: Path) -> DirectDeadReckon:
    section = _section(
        raw, "dead_reckon", _DEAD_RECKON_KEYS, source, optional=frozenset({"max_age_s"})
    )
    return DirectDeadReckon(
        enabled=_boolean("dead_reckon", "enabled", section["enabled"], source),
        max_frames=_positive_int("dead_reckon", "max_frames", section["max_frames"], source),
        # Legacy releases allowed 300 successful DR steps on the 30 Hz stream.
        max_age_s=_positive_float("dead_reckon", "max_age_s", section.get("max_age_s", 10.0), source),
    )


def load_direct_profile(
    path: str | Path, *, expected_sha256: str | None = None
) -> DirectProfile:
    """Load, integrity-check and validate one frozen direct deployment profile."""

    source = Path(path).expanduser().resolve(strict=True)
    digest = verify_file_sha256(source, expected_sha256)
    raw = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(raw, dict):
        raise DirectProfileError(f"direct profile must be a JSON object: {source}")
    if raw.get("schema") != DIRECT_PROFILE_SCHEMA:
        raise DirectProfileError(
            f"direct profile schema must be {DIRECT_PROFILE_SCHEMA!r}: {source}"
        )
    missing = sorted(_TOP_KEYS - raw.keys())
    unknown = sorted(raw.keys() - _TOP_KEYS - {"optimizations"})
    if missing:
        raise DirectProfileError(f"direct profile is missing {missing}: {source}")
    if unknown:
        raise DirectProfileError(f"direct profile has unknown keys {unknown}: {source}")
    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise DirectProfileError(f"direct profile name must be a non-empty string: {source}")
    optimizations = DirectOptimizations()
    if "optimizations" in raw:
        section = _section(raw, "optimizations", _OPTIMIZATION_KEYS, source)
        optimizations = DirectOptimizations(
            adaptive_retrieval=_boolean("optimizations", "adaptive_retrieval", section["adaptive_retrieval"], source),
            spatial_selection=_boolean("optimizations", "spatial_selection", section["spatial_selection"], source),
            gpu_cache_size=_nonnegative_int("optimizations", "gpu_cache_size", section["gpu_cache_size"], source),
        )
    return DirectProfile(
        name=name,
        map_scale=_positive_float("profile", "map_scale", raw["map_scale"], source),
        intrinsics=_parse_intrinsics(raw, source),
        reloc=_parse_reloc(raw, source),
        fast_loop=_parse_fast_loop(raw, source),
        vo=_parse_vo(raw, source),
        dead_reckon=_parse_dead_reckon(raw, source),
        raw=MappingProxyType(raw),
        sha256=digest,
        source=source,
        optimizations=optimizations,
    )
