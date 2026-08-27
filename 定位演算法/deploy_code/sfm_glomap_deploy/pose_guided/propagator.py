"""Short-term pose prediction. Lost visual uses IMU to guess map motion."""

from __future__ import annotations

import math

import numpy as np

from pose_guided.config import PoseGuidedConfig
from pose_guided.se3 import (
    compose_transforms,
    invert_transform,
    olympe_yaw_to_map_ccw,
    transform_from_rt,
    wrap_pi,
)
from pose_guided.types import (
    Confirmation,
    FusedOdometrySample,
    OdometryFrame,
    PosePrediction,
    PropagationMode,
    VisualAnchor,
)


_MIN_NED_SPEED_MPS = 0.05


def _rotate_map_xy(vector: np.ndarray, delta_yaw: float) -> np.ndarray:
    cosine = math.cos(delta_yaw)
    sine = math.sin(delta_yaw)
    return np.array(
        [
            cosine * float(vector[0]) - sine * float(vector[1]),
            sine * float(vector[0]) + cosine * float(vector[1]),
            float(vector[2]),
        ],
        dtype=float,
    )


def _horizontal_ned_speed(sample: FusedOdometrySample | None) -> float | None:
    if sample is None or sample.velocity_ned is None:
        return None
    return math.hypot(float(sample.velocity_ned[0]), float(sample.velocity_ned[1]))


def grow_uncertainty(
    config: PoseGuidedConfig,
    *,
    age_s: float,
    speed_mps: float | None = None,
) -> tuple[np.ndarray, float]:
    age = max(0.0, float(age_s))
    sigma_pos = config.uncertainty.sigma_position_base + config.uncertainty.k_position * age
    if (
        speed_mps is not None
        and math.isfinite(speed_mps)
        and speed_mps >= 0.0
        and config.metres_per_map_unit is not None
    ):
        sigma_pos += (speed_mps * age) / config.metres_per_map_unit
    sigma_rot = (
        config.uncertainty.sigma_rotation_base_rad + config.uncertainty.k_rotation_rad * age
    )
    covariance = np.eye(3, dtype=float) * (sigma_pos * sigma_pos)
    return covariance, sigma_rot


def compose_map_from_odom_delta(anchor_map: np.ndarray, anchor_odom: np.ndarray, current_odom: np.ndarray) -> np.ndarray:
    """T_M_B(t) = T_M_B(t0) @ inv(T_O_B(t0)) @ T_O_B(t)."""
    return compose_transforms(
        anchor_map,
        compose_transforms(invert_transform(anchor_odom), current_odom),
    )


def compose_map_from_fixed_extrinsic(anchor_map: np.ndarray, anchor_odom: np.ndarray, current_odom: np.ndarray) -> np.ndarray:
    """T_M_B(t) = (T_M_B(t0) @ inv(T_O_B(t0))) @ T_O_B(t)."""
    map_from_odom = compose_transforms(anchor_map, invert_transform(anchor_odom))
    return compose_transforms(map_from_odom, current_odom)


def _propagation_mode(
    map_pose: np.ndarray | None,
    used_visual_velocity: bool,
    used_imu_motion: bool,
    used_orientation: bool,
) -> tuple[PropagationMode, str]:
    if map_pose is not None:
        return PropagationMode.FUSED_POSE, "map_aligned_odom"
    if used_visual_velocity and used_imu_motion:
        return PropagationMode.VELOCITY_INTEGRATION, "imu_compensated_visual_velocity"
    if used_visual_velocity:
        if used_orientation:
            return PropagationMode.ORIENTATION_PRIOR, "visual_velocity+yaw_increment"
        return PropagationMode.VISUAL_VELOCITY, "visual_velocity"
    if used_orientation:
        return PropagationMode.ORIENTATION_PRIOR, "yaw_increment"
    return PropagationMode.UNAVAILABLE, "none"


class PosePropagator:
    def __init__(self, config: PoseGuidedConfig) -> None:
        self.config = config

    def predict(
        self,
        *,
        query_timestamp: float,
        anchor: VisualAnchor | None,
        fused: FusedOdometrySample | None,
        visual_center: np.ndarray | None,
        visual_yaw: float | None,
        visual_velocity: np.ndarray | None,
        visual_stamp: float | None,
    ) -> PosePrediction:
        if not math.isfinite(query_timestamp):
            return self._invalid(query_timestamp, "non_finite_query_timestamp")
        position, used_visual_velocity, used_imu_motion = self._predict_position(
            query_timestamp, visual_center, visual_velocity, visual_stamp, anchor, fused
        )
        yaw, used_orientation = self._predict_yaw(query_timestamp, anchor, fused, visual_yaw)
        age = None if anchor is None else query_timestamp - anchor.timestamp
        if age is not None and (not math.isfinite(age) or age < 0.0):
            age = None
        if age is not None and age > self.config.max_anchor_age_s:
            return self._invalid(query_timestamp, "anchor_too_old", age=age)

        map_pose = self._try_map_aligned(anchor, fused)
        mode, source = _propagation_mode(
            map_pose,
            used_visual_velocity,
            used_imu_motion,
            used_orientation,
        )
        if map_pose is not None:
            position = map_pose[:3, 3]
        valid = position is not None or yaw is not None
        if not valid:
            return self._invalid(query_timestamp, source, age=age)

        speed = None
        if fused is not None and fused.velocity_ned is not None:
            speed = math.hypot(fused.velocity_ned[0], fused.velocity_ned[1])
        covariance, sigma_rot = grow_uncertainty(
            self.config,
            age_s=0.0 if age is None else age,
            speed_mps=speed,
        )
        confidence = 1.0 / (1.0 + (0.0 if age is None else age))
        if mode is PropagationMode.UNAVAILABLE:
            confidence = 0.0
        return PosePrediction(
            timestamp=query_timestamp,
            position=None if position is None else np.asarray(position, dtype=float),
            yaw=yaw,
            rotation_cam_from_world=None,
            position_covariance=covariance if position is not None else None,
            orientation_covariance=sigma_rot if yaw is not None else None,
            age_since_visual_anchor=age,
            propagation_mode=mode,
            source=source,
            confidence=confidence,
            valid=True,
            confirmation=Confirmation.PREDICTED_ONLY,
        )

    def _predict_position(
        self,
        query_timestamp: float,
        visual_center: np.ndarray | None,
        visual_velocity: np.ndarray | None,
        visual_stamp: float | None,
        anchor: VisualAnchor | None,
        fused: FusedOdometrySample | None,
    ) -> tuple[np.ndarray | None, bool, bool]:
        if visual_center is None:
            return None, False, False
        center = np.asarray(visual_center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            return None, False, False
        if visual_velocity is None or visual_stamp is None:
            return center, False, False
        velocity = np.asarray(visual_velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            return center, False, False
        dt = query_timestamp - float(visual_stamp)
        if not math.isfinite(dt) or dt <= 0.0:
            return center, False, False
        used_imu = False
        delta_yaw = self._yaw_increment(query_timestamp, anchor, fused)
        if delta_yaw is not None:
            velocity = _rotate_map_xy(velocity, delta_yaw)
            ratio = self._ned_speed_ratio(anchor, fused)
            if ratio is not None:
                velocity = velocity * ratio
            dt = min(dt, self.config.max_anchor_age_s)
            used_imu = True
        else:
            dt = min(dt, self.config.max_prediction_dt_s)
        return center + velocity * dt, True, used_imu

    def _yaw_increment(
        self,
        query_timestamp: float,
        anchor: VisualAnchor | None,
        fused: FusedOdometrySample | None,
    ) -> float | None:
        if (
            not self.config.apply_yaw_increment
            or anchor is None
            or fused is None
            or not fused.has_attitude
            or anchor.odom_sample is None
            or not anchor.odom_sample.has_attitude
        ):
            return None
        if abs(fused.timestamp - query_timestamp) > self.config.max_sync_error_s:
            return None
        try:
            return wrap_pi(
                olympe_yaw_to_map_ccw(fused.yaw)
                - olympe_yaw_to_map_ccw(anchor.odom_sample.yaw)
            )
        except ValueError:
            return None

    def _ned_speed_ratio(
        self,
        anchor: VisualAnchor | None,
        fused: FusedOdometrySample | None,
    ) -> float | None:
        if anchor is None:
            return None
        start = _horizontal_ned_speed(anchor.odom_sample)
        now = _horizontal_ned_speed(fused)
        if start is None or now is None or start < _MIN_NED_SPEED_MPS:
            return None
        return now / start

    def _predict_yaw(
        self,
        query_timestamp: float,
        anchor: VisualAnchor | None,
        fused: FusedOdometrySample | None,
        visual_yaw: float | None,
    ) -> tuple[float | None, bool]:
        if visual_yaw is None or not math.isfinite(visual_yaw):
            return None, False
        delta = self._yaw_increment(query_timestamp, anchor, fused)
        if delta is None:
            return wrap_pi(visual_yaw), False
        return wrap_pi(anchor.map_yaw + delta), True

    def _try_map_aligned(
        self,
        anchor: VisualAnchor | None,
        fused: FusedOdometrySample | None,
    ) -> np.ndarray | None:
        if (
            anchor is None
            or fused is None
            or anchor.map_pose is None
            or not anchor.has_odom_pose
            or fused.frame is not OdometryFrame.MAP
            or fused.position is None
            or fused.quaternion_wxyz is None
        ):
            return None
        from pose_guided.se3 import matrix_from_quaternion_wxyz

        current = transform_from_rt(
            matrix_from_quaternion_wxyz(np.asarray(fused.quaternion_wxyz, dtype=float)),
            np.asarray(fused.position, dtype=float),
        )
        return compose_map_from_odom_delta(anchor.map_pose, anchor.odom_pose, current)


    def _invalid(self, timestamp: float, source: str, *, age: float | None = None) -> PosePrediction:
        return PosePrediction(
            timestamp=timestamp,
            position=None,
            yaw=None,
            rotation_cam_from_world=None,
            position_covariance=None,
            orientation_covariance=None,
            age_since_visual_anchor=age,
            propagation_mode=PropagationMode.UNAVAILABLE,
            source=source,
            confidence=0.0,
            valid=False,
            confirmation=Confirmation.NONE,
        )
