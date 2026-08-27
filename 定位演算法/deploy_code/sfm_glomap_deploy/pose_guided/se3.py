"""SE(3) helpers with explicit cam_from_world / body conventions.

Quaternions are wxyz. Matrices are active, right-handed, det=+1.
A 4x4 T_A_B maps coordinates of a point in B into A: p_A = R p_B + t.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def require_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = require_finite("quaternion", quaternion).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < _EPS:
        raise ValueError("quaternion has near-zero norm")
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def quaternion_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = require_rotation(rotation)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
    return normalize_quaternion_wxyz(quat)


def matrix_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def require_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = require_finite("rotation", rotation)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError("rotation must be right-handed")
    return matrix


def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=float)


def transform_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = identity_transform()
    matrix[:3, :3] = require_rotation(rotation)
    matrix[:3, 3] = require_finite("translation", translation).reshape(3)
    return matrix


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = require_finite("transform", transform)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    rotation = require_rotation(transform[:3, :3])
    translation = transform[:3, 3]
    inverse = identity_transform()
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def compose_transforms(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return T_A_C = T_A_B @ T_B_C."""
    left = require_finite("first", first)
    right = require_finite("second", second)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("transforms must be 4x4")
    return left @ right


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    matrix = require_finite("transform", transform)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    vector = require_finite("point", point).reshape(3)
    return matrix[:3, :3] @ vector + matrix[:3, 3]


def slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("slerp fraction must be in [0, 1]")
    q0 = normalize_quaternion_wxyz(start)
    q1 = normalize_quaternion_wxyz(end)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quaternion_wxyz(q0 + fraction * (q1 - q0))
    omega = math.acos(min(1.0, max(-1.0, dot)))
    sin_omega = math.sin(omega)
    return (
        math.sin((1.0 - fraction) * omega) / sin_omega * q0
        + math.sin(fraction * omega) / sin_omega * q1
    )


def ned_rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Aircraft ZYX: R_ned_from_body = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("NED RPY must be finite")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rotation_z @ rotation_y @ rotation_x


def olympe_yaw_to_map_ccw(ned_yaw: float) -> float:
    """NED CW-from-North → map CCW-from-East. Same as HeadingEstimator."""
    if not math.isfinite(ned_yaw):
        raise ValueError("NED yaw must be finite")
    return wrap_pi(math.pi / 2.0 - float(ned_yaw))


def cam_from_world(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """T_C_M from pycolmap R,t where p_cam = R @ p_world + t."""
    return transform_from_rt(rotation, translation)


def world_from_cam(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return invert_transform(cam_from_world(rotation, translation))


def camera_center_from_cam_from_world(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = require_rotation(rotation)
    translation = require_finite("translation", translation).reshape(3)
    return -rotation.T @ translation


def optical_axis_from_cam_from_world(rotation: np.ndarray) -> np.ndarray:
    return require_rotation(rotation).T @ np.array([0.0, 0.0, 1.0])


def heading_from_optical_axis(axis: np.ndarray) -> float:
    vector = require_finite("optical axis", axis).reshape(3)
    return wrap_pi(math.atan2(float(vector[1]), float(vector[0])))
