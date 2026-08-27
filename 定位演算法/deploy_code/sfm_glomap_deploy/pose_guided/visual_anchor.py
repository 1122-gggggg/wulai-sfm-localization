"""Visual anchors. Never integrate IMU from power-on; only from a strong visual fix."""

from __future__ import annotations

import math

import numpy as np

from pose_guided.propagator import compose_map_from_fixed_extrinsic, compose_map_from_odom_delta
from pose_guided.se3 import require_finite, transform_from_rt, world_from_cam
from pose_guided.types import FusedOdometrySample, VisualAnchor


def map_pose_from_cam_from_world(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """T_M_C = inverse(T_C_M). Camera center is the translation of T_M_C."""
    return world_from_cam(rotation, translation)


def predict_from_odom_delta(anchor_map: np.ndarray, anchor_odom: np.ndarray, current_odom: np.ndarray) -> np.ndarray:
    return compose_map_from_odom_delta(anchor_map, anchor_odom, current_odom)


def predict_from_fixed_map_from_odom(
    anchor_map: np.ndarray,
    anchor_odom: np.ndarray,
    current_odom: np.ndarray,
) -> np.ndarray:
    return compose_map_from_fixed_extrinsic(anchor_map, anchor_odom, current_odom)


class VisualAnchorManager:
    def __init__(self) -> None:
        self.anchor: VisualAnchor | None = None

    def clear(self) -> None:
        self.anchor = None

    def update(
        self,
        *,
        timestamp: float,
        rotation_cam_from_world: np.ndarray,
        center: np.ndarray,
        yaw: float,
        fused: FusedOdometrySample | None = None,
        odom_pose: np.ndarray | None = None,
    ) -> VisualAnchor:
        if not math.isfinite(timestamp):
            raise ValueError("anchor timestamp must be finite")
        rotation = require_finite("rotation", rotation_cam_from_world)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        position = require_finite("center", center).reshape(3)
        translation = -rotation @ position
        map_pose = map_pose_from_cam_from_world(rotation, translation)
        stored_odom = None if odom_pose is None else np.asarray(odom_pose, dtype=float).copy()
        if stored_odom is not None and stored_odom.shape != (4, 4):
            raise ValueError("odom_pose must be 4x4")
        self.anchor = VisualAnchor(
            timestamp=float(timestamp),
            map_center=position.copy(),
            map_rotation_cam_from_world=np.asarray(rotation, dtype=float).copy(),
            map_yaw=float(yaw),
            odom_sample=fused,
            odom_pose=stored_odom,
            map_pose=map_pose,
        )
        return self.anchor


def odom_pose_from_map_sample(sample: FusedOdometrySample) -> np.ndarray | None:
    from pose_guided.se3 import matrix_from_quaternion_wxyz
    from pose_guided.types import OdometryFrame

    if sample.frame is not OdometryFrame.MAP or sample.position is None:
        return None
    if sample.quaternion_wxyz is None:
        return None
    return transform_from_rt(
        matrix_from_quaternion_wxyz(np.asarray(sample.quaternion_wxyz, dtype=float)),
        np.asarray(sample.position, dtype=float),
    )
