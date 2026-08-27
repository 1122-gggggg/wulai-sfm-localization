"""Typed fused-state, prediction, and visual-anchor records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PropagationMode(str, Enum):
    FUSED_POSE = "fused_pose"
    VELOCITY_INTEGRATION = "velocity_integration"
    VISUAL_VELOCITY = "visual_velocity"
    ORIENTATION_PRIOR = "orientation_prior"
    UNAVAILABLE = "unavailable"


class Confirmation(str, Enum):
    VISUALLY_CONFIRMED = "VISUALLY_CONFIRMED"
    PREDICTED_ONLY = "PREDICTED_ONLY"
    NONE = "NONE"


class OdometryFrame(str, Enum):
    MAP = "map"
    NED_FUSED = "ned_fused"
    UNKNOWN = "unknown"


def _optional_finite(name: str, value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric or null") from exc
    if not math.isfinite(number):
        return None
    return number


@dataclass(frozen=True)
class FusedOdometrySample:
    """One firmware or caller-supplied fused sample.

    ``frame=MAP`` is the only label that authorizes SE(3) composition into the
    GlueMap frame. ANAFI AttitudeChanged/SpeedChanged must use ``NED_FUSED``.
    """

    timestamp: float
    frame: OdometryFrame = OdometryFrame.UNKNOWN
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    velocity_ned: tuple[float, float, float] | None = None
    position: tuple[float, float, float] | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    source: str = "unknown"

    @property
    def has_attitude(self) -> bool:
        return self.roll is not None and self.pitch is not None and self.yaw is not None

    @property
    def has_map_pose(self) -> bool:
        return self.frame is OdometryFrame.MAP and self.position is not None and (
            self.quaternion_wxyz is not None or self.has_attitude
        )

    @property
    def has_ned_velocity(self) -> bool:
        return self.velocity_ned is not None

    @staticmethod
    def from_anafi(
        *,
        timestamp: float,
        roll: object = None,
        pitch: object = None,
        yaw: object = None,
        speed_north: object = None,
        speed_east: object = None,
        speed_down: object = None,
        source: str = "anafi_fused",
    ) -> FusedOdometrySample | None:
        if not math.isfinite(float(timestamp)):
            return None
        velocity = None
        north = _optional_finite("speed_north", speed_north)
        east = _optional_finite("speed_east", speed_east)
        down = _optional_finite("speed_down", speed_down)
        if north is not None and east is not None and down is not None:
            velocity = (north, east, down)
        sample = FusedOdometrySample(
            timestamp=float(timestamp),
            frame=OdometryFrame.NED_FUSED,
            roll=_optional_finite("roll", roll),
            pitch=_optional_finite("pitch", pitch),
            yaw=_optional_finite("yaw", yaw),
            velocity_ned=velocity,
            source=source,
        )
        if not sample.has_attitude and not sample.has_ned_velocity:
            return None
        return sample


@dataclass(frozen=True)
class VisualAnchor:
    timestamp: float
    map_center: np.ndarray
    map_rotation_cam_from_world: np.ndarray
    map_yaw: float
    odom_sample: FusedOdometrySample | None = None
    odom_pose: np.ndarray | None = None
    map_pose: np.ndarray | None = None

    @property
    def has_odom_pose(self) -> bool:
        return self.odom_pose is not None and np.asarray(self.odom_pose).shape == (4, 4)


@dataclass(frozen=True)
class PosePrediction:
    timestamp: float
    position: np.ndarray | None
    yaw: float | None
    rotation_cam_from_world: np.ndarray | None
    position_covariance: np.ndarray | None
    orientation_covariance: float | None
    age_since_visual_anchor: float | None
    propagation_mode: PropagationMode
    source: str
    confidence: float
    valid: bool
    confirmation: Confirmation = Confirmation.PREDICTED_ONLY

    def as_info(self) -> dict:
        return {
            "pose_status": self.confirmation.value,
            "prediction_valid": bool(self.valid),
            "prediction_mode": self.propagation_mode.value,
            "prediction_source": self.source,
            "prediction_confidence": self.confidence,
            "prediction_age_s": self.age_since_visual_anchor,
            "predicted_center": None if self.position is None else self.position.astype(float).tolist(),
            "predicted_yaw": self.yaw,
        }
