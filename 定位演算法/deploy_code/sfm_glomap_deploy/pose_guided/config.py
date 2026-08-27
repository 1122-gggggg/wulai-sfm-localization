"""Feature-flagged pose-guided localization config. Default is disabled."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "pose-guided-localization/v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "pose_guided_localization.json"
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _finite(name: str, value: object, *, positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be > 0")
    if non_negative and number < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return number


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class CameraToBodyConfig:
    translation_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    approved: bool = False


@dataclass(frozen=True)
class UncertaintyConfig:
    sigma_position_base: float
    k_position: float
    sigma_rotation_base_rad: float
    k_rotation_rad: float
    mahalanobis_threshold: float
    hard_max_radius_factor: float


@dataclass(frozen=True)
class SearchStageConfig:
    radius_factor: float
    yaw_factor: float
    max_refs_factor: float


@dataclass(frozen=True)
class RankingConfig:
    w_position: float
    w_orientation: float
    w_covisibility: float
    w_temporal: float
    w_quality: float


@dataclass(frozen=True)
class AnchorGateConfig:
    min_inliers: int
    min_inlier_ratio: float
    min_inlier_grid_cells: int
    max_reproj_rms: float


@dataclass(frozen=True)
class PoseGuidedConfig:
    enabled: bool = False
    apply_yaw_increment: bool = True
    allow_weak_anchors: bool = False
    camera_center_is_body_origin: bool = True
    max_sync_error_s: float = 0.15
    max_sample_age_s: float = 0.40
    max_prediction_dt_s: float = 0.25
    max_anchor_age_s: float = 3.0
    metres_per_map_unit: float | None = None
    height_axis: int | None = None
    camera_to_body: CameraToBodyConfig = field(
        default_factory=lambda: CameraToBodyConfig((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), False)
    )
    uncertainty: UncertaintyConfig = field(
        default_factory=lambda: UncertaintyConfig(0.10, 0.40, math.radians(5.0), math.radians(15.0), 9.0, 8.0)
    )
    search: dict[str, SearchStageConfig] = field(default_factory=dict)
    ranking: RankingConfig = field(
        default_factory=lambda: RankingConfig(1.0, 0.8, 0.4, 0.5, 0.0)
    )
    anchor: AnchorGateConfig = field(
        default_factory=lambda: AnchorGateConfig(80, 0.15, 6, 5.0)
    )

    @property
    def camera_body_ready(self) -> bool:
        """True when navigation may treat the visual camera center as the UAV origin.

        This is a coincident-origin convention, not a measured axis calibration.
        Camera OpenCV axes are still not FRD body axes.
        """
        return bool(self.camera_center_is_body_origin or self.camera_to_body.approved)



def _vec3(name: str, value: object) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 list")
    return tuple(_finite(f"{name}[{i}]", item) for i, item in enumerate(value))


def _quat(name: str, value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be a wxyz length-4 list")
    numbers = tuple(_finite(f"{name}[{i}]", item) for i, item in enumerate(value))
    norm = math.sqrt(sum(item * item for item in numbers))
    if norm < 1e-12:
        raise ValueError(f"{name} has near-zero norm")
    return tuple(item / norm for item in numbers)


def _stage(name: str, raw: object) -> SearchStageConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    return SearchStageConfig(
        radius_factor=_finite(f"{name}.radius_factor", raw.get("radius_factor"), positive=True),
        yaw_factor=_finite(f"{name}.yaw_factor", raw.get("yaw_factor"), positive=True),
        max_refs_factor=_finite(f"{name}.max_refs_factor", raw.get("max_refs_factor"), positive=True),
    )


def _config_sections(
    raw: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    sections = []
    for name in ("camera_to_body", "uncertainty", "ranking", "anchor", "search"):
        section = raw.get(name)
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be an object")
        sections.append(section)
    return tuple(sections)  # type: ignore[return-value]


def _config_scalars(
    raw: dict[str, object], anchor_raw: dict[str, object]
) -> tuple[float | None, int | None, int, int]:
    metres = raw.get("metres_per_map_unit")
    height_axis = raw.get("height_axis")
    if metres is not None:
        metres = _finite("metres_per_map_unit", metres, positive=True)
    if height_axis is not None:
        if isinstance(height_axis, bool) or not isinstance(height_axis, int) or height_axis not in (0, 1, 2):
            raise ValueError("height_axis must be 0, 1, 2, or null")
    min_inliers = anchor_raw.get("min_inliers")
    min_cells = anchor_raw.get("min_inlier_grid_cells")
    if isinstance(min_inliers, bool) or not isinstance(min_inliers, int) or min_inliers <= 0:
        raise ValueError("anchor.min_inliers must be a positive integer")
    if isinstance(min_cells, bool) or not isinstance(min_cells, int) or min_cells <= 0:
        raise ValueError("anchor.min_inlier_grid_cells must be a positive integer")
    return metres, height_axis, min_inliers, min_cells


def pose_guided_config_from_dict(raw: object) -> PoseGuidedConfig:
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError(f"pose-guided config schema must be {SCHEMA}")
    extrinsic_raw, uncertainty_raw, ranking_raw, anchor_raw, search_raw = _config_sections(raw)
    metres, height_axis, min_inliers, min_cells = _config_scalars(raw, anchor_raw)
    coincident = raw.get("camera_center_is_body_origin", True)
    return PoseGuidedConfig(
        enabled=_bool("enabled", raw.get("enabled")),
        apply_yaw_increment=_bool("apply_yaw_increment", raw.get("apply_yaw_increment")),
        allow_weak_anchors=_bool("allow_weak_anchors", raw.get("allow_weak_anchors")),
        camera_center_is_body_origin=_bool("camera_center_is_body_origin", coincident),
        max_sync_error_s=_finite("max_sync_error_s", raw.get("max_sync_error_s"), positive=True),
        max_sample_age_s=_finite("max_sample_age_s", raw.get("max_sample_age_s"), positive=True),
        max_prediction_dt_s=_finite("max_prediction_dt_s", raw.get("max_prediction_dt_s"), positive=True),
        max_anchor_age_s=_finite("max_anchor_age_s", raw.get("max_anchor_age_s"), positive=True),
        metres_per_map_unit=metres,
        height_axis=height_axis,
        camera_to_body=CameraToBodyConfig(
            translation_m=_vec3("camera_to_body.translation_m", extrinsic_raw.get("translation_m")),
            quaternion_wxyz=_quat(
                "camera_to_body.quaternion_wxyz",
                extrinsic_raw.get("quaternion_wxyz"),
            ),
            approved=_bool("camera_to_body.approved", extrinsic_raw.get("approved")),
        ),
        uncertainty=UncertaintyConfig(
            sigma_position_base=_finite(
                "uncertainty.sigma_position_base",
                uncertainty_raw.get("sigma_position_base"),
                non_negative=True,
            ),
            k_position=_finite("uncertainty.k_position", uncertainty_raw.get("k_position"), non_negative=True),
            sigma_rotation_base_rad=_finite(
                "uncertainty.sigma_rotation_base_rad",
                uncertainty_raw.get("sigma_rotation_base_rad"),
                non_negative=True,
            ),
            k_rotation_rad=_finite(
                "uncertainty.k_rotation_rad",
                uncertainty_raw.get("k_rotation_rad"),
                non_negative=True,
            ),
            mahalanobis_threshold=_finite(
                "uncertainty.mahalanobis_threshold",
                uncertainty_raw.get("mahalanobis_threshold"),
                positive=True,
            ),
            hard_max_radius_factor=_finite(
                "uncertainty.hard_max_radius_factor",
                uncertainty_raw.get("hard_max_radius_factor"),
                positive=True,
            ),
        ),
        search={key: _stage(f"search.{key}", value) for key, value in search_raw.items()},
        ranking=RankingConfig(
            w_position=_finite("ranking.w_position", ranking_raw.get("w_position"), non_negative=True),
            w_orientation=_finite("ranking.w_orientation", ranking_raw.get("w_orientation"), non_negative=True),
            w_covisibility=_finite(
                "ranking.w_covisibility", ranking_raw.get("w_covisibility"), non_negative=True
            ),
            w_temporal=_finite("ranking.w_temporal", ranking_raw.get("w_temporal"), non_negative=True),
            w_quality=_finite("ranking.w_quality", ranking_raw.get("w_quality"), non_negative=True),
        ),
        anchor=AnchorGateConfig(
            min_inliers=min_inliers,
            min_inlier_ratio=_finite(
                "anchor.min_inlier_ratio", anchor_raw.get("min_inlier_ratio"), positive=True
            ),
            min_inlier_grid_cells=min_cells,
            max_reproj_rms=_finite("anchor.max_reproj_rms", anchor_raw.get("max_reproj_rms"), positive=True),
        ),
    )


def load_pose_guided_config(path: str | Path | None = None) -> PoseGuidedConfig:
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pose-guided config {source}: {exc}") from exc
    config = pose_guided_config_from_dict(raw)
    override = os.environ.get("SFM_POSE_GUIDED", "").strip()
    if override == "1":
        return PoseGuidedConfig(**{**config.__dict__, "enabled": True})
    if override == "0":
        return PoseGuidedConfig(**{**config.__dict__, "enabled": False})
    return config
