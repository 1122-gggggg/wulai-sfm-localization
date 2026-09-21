"""Offline visual-pose ESKF experiment.

This module is deliberately separate from the production localizer.  It is a
visual-only constant-velocity error-state filter: without raw IMU it does not
pretend to fuse accelerometer or gyroscope data, and it never treats its own
prediction as a measurement.

``cam_from_world`` follows the direct localizer convention ``p_cam = R p_world
+ t``.  The nominal state stores the camera centre in map units and the
camera-to-world orientation.  The 12-dimensional error state is
``[dp, dv, dtheta, dw]`` in map/world coordinates.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


_INNOVATION_GATE = 12.592  # chi-square 95% gate for a six-dimensional pose.


def _exp_so3(vector: np.ndarray) -> np.ndarray:
    """Return ``exp([vector]x)`` through SciPy's stable SO(3) implementation."""
    return Rotation.from_rotvec(np.asarray(vector, dtype=float).reshape(3)).as_matrix()


def _log_so3(rotation: np.ndarray) -> np.ndarray:
    """Return the principal rotation vector through SciPy's SO(3) API."""
    return Rotation.from_matrix(np.asarray(rotation, dtype=float).reshape(3, 3)).as_rotvec()


def _left_jacobian(vector: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian, with a numerically stable small-angle branch."""
    phi = np.asarray(vector, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    if not math.isfinite(theta):
        raise ValueError("SO(3) vector must be finite")
    if theta < 1e-8:
        # The skew helper is intentionally local: Rotation remains the source
        # of truth for exp/log/quaternion conversions, while this Jacobian is
        # needed by the error-state covariance transport.
        x, y, z = (float(value) for value in phi)
        skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)
        return np.eye(3) + 0.5 * skew + (skew @ skew) / 6.0
    x, y, z = (float(value) for value in phi)
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)
    return (
        np.eye(3)
        + ((1.0 - math.cos(theta)) / theta**2) * skew
        + ((theta - math.sin(theta)) / theta**3) * (skew @ skew)
    )


def _project_rotation(rotation: np.ndarray) -> np.ndarray | None:
    """Project a finite, non-reflective 3x3 matrix onto SO(3)."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or determinant <= 0.0:
        return None
    try:
        u, singular, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None
    if float(np.min(singular)) < 1e-8:
        return None
    projected = u @ vt
    if float(np.linalg.det(projected)) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    if not np.isfinite(projected).all() or float(np.linalg.det(projected)) <= 0.0:
        return None
    return projected


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Convert SO(3) to the filter's normalized wxyz quaternion."""
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=float).reshape(3, 3)).as_quat()
    q = np.array((xyzw[3], xyzw[0], xyzw[1], xyzw[2]), dtype=float)
    if q[0] < 0.0:
        q = -q
    return q


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion is not finite")
    w, x, y, z = q / norm
    return Rotation.from_quat(np.array((x, y, z, w), dtype=float)).as_matrix()


def _pose_components(cam_from_world: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    matrix = np.asarray(cam_from_world, dtype=float)
    if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
        return None
    rotation = _project_rotation(matrix[:, :3])
    if rotation is None:
        return None
    # Rcw maps world to camera; p is the camera centre in world/map units.
    center = -rotation.T @ matrix[:, 3]
    if not np.isfinite(center).all():
        return None
    return center, _quaternion_from_rotation(rotation.T)


class VisualPoseESKF:
    """Visual-only constant-velocity ESKF for offline A/B experiments.

    The process model uses white linear and angular acceleration noise.  It
    consumes no raw IMU and does not convert metric NED telemetry into map
    units.  A missing visual pose advances the prediction; the next visual
    pose is the only measurement update.
    """

    _STATE_SIZE = 12

    def __init__(
        self,
        acceleration_sigma: float = 1.0,
        angular_acceleration_sigma: float = 1.0,
        max_gap_s: float = 0.5,
    ) -> None:
        self.acceleration_sigma = self._positive_finite(acceleration_sigma, "acceleration_sigma")
        self.angular_acceleration_sigma = self._positive_finite(
            angular_acceleration_sigma, "angular_acceleration_sigma"
        )
        self.max_gap_s = self._positive_finite(max_gap_s, "max_gap_s")
        self._epoch: int | None = None
        self._stamp: float | None = None
        self._last_accepted_stamp: float | None = None
        self._p = np.zeros(3, dtype=float)
        self._v = np.zeros(3, dtype=float)
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._w = np.zeros(3, dtype=float)
        self._P = np.eye(self._STATE_SIZE, dtype=float)
        self._initialized = False

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return parsed

    @staticmethod
    def _measurement_sigma(value: float, name: str) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(parsed) or parsed <= 0.0:
            return None
        return parsed

    @staticmethod
    def _epoch_value(epoch: int) -> int | None:
        if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, (int, np.integer)):
            return None
        return int(epoch)

    @staticmethod
    def _stamp_value(stamp: float) -> float | None:
        try:
            parsed = float(stamp)
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if math.isfinite(parsed) else None

    def reset(self) -> None:
        """Drop all state and the timestamp/epoch ordering anchor."""
        self._epoch = None
        self._stamp = None
        self._last_accepted_stamp = None
        self._p.fill(0.0)
        self._v.fill(0.0)
        self._q[:] = (1.0, 0.0, 0.0, 0.0)
        self._w.fill(0.0)
        self._P = np.eye(self._STATE_SIZE, dtype=float)
        self._initialized = False

    def _pose_matrix(self) -> np.ndarray | None:
        if not self._initialized:
            return None
        rotation_wc = _rotation_from_quaternion(self._q)
        rotation_cw = rotation_wc.T
        translation = -rotation_cw @ self._p
        return np.column_stack((rotation_cw, translation))

    def _result(
        self,
        *,
        accepted: bool,
        reset: bool,
        innovation_mahalanobis: float | None,
        predicted_only: bool,
    ) -> dict:
        return {
            "pose": self._pose_matrix(),
            "accepted": bool(accepted),
            "reset": bool(reset),
            "innovation_mahalanobis": innovation_mahalanobis,
            "predicted_only": bool(predicted_only),
        }

    def _initialize(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        position_sigma: float,
        rotation_sigma: float,
    ) -> None:
        self._p = position.copy()
        self._v.fill(0.0)
        self._q = quaternion.copy()
        self._w.fill(0.0)
        self._P = np.zeros((self._STATE_SIZE, self._STATE_SIZE), dtype=float)
        self._P[0:3, 0:3] = np.eye(3) * position_sigma**2
        self._P[3:6, 3:6] = np.eye(3) * max(self.acceleration_sigma**2, 1e-6)
        self._P[6:9, 6:9] = np.eye(3) * rotation_sigma**2
        self._P[9:12, 9:12] = np.eye(3) * max(self.angular_acceleration_sigma**2, 1e-6)
        self._initialized = True

    def _process_covariance(self, dt: float) -> np.ndarray:
        q_a = self.acceleration_sigma**2
        q_alpha = self.angular_acceleration_sigma**2
        covariance = np.zeros((self._STATE_SIZE, self._STATE_SIZE), dtype=float)
        covariance[0:3, 0:3] = np.eye(3) * (q_a * dt**3 / 3.0)
        covariance[0:3, 3:6] = np.eye(3) * (q_a * dt**2 / 2.0)
        covariance[3:6, 0:3] = covariance[0:3, 3:6]
        covariance[3:6, 3:6] = np.eye(3) * (q_a * dt)
        covariance[6:9, 6:9] = np.eye(3) * (q_alpha * dt**3 / 3.0)
        covariance[6:9, 9:12] = np.eye(3) * (q_alpha * dt**2 / 2.0)
        covariance[9:12, 6:9] = covariance[6:9, 9:12]
        covariance[9:12, 9:12] = np.eye(3) * (q_alpha * dt)
        return covariance

    def _propagate(self, dt: float) -> None:
        if dt <= 0.0:
            return
        angular_step = self._w * dt
        orientation_transport = _exp_so3(angular_step)
        self._p = self._p + self._v * dt
        self._q = _quaternion_from_rotation(
            orientation_transport @ _rotation_from_quaternion(self._q)
        )
        transition = np.eye(self._STATE_SIZE, dtype=float)
        transition[0:3, 3:6] = np.eye(3) * dt
        transition[6:9, 6:9] = orientation_transport
        transition[6:9, 9:12] = _left_jacobian(angular_step) * dt
        self._P = transition @ self._P @ transition.T + self._process_covariance(dt)
        self._P = self._stabilize_covariance(self._P)

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        matrix = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
        if matrix.shape != (12, 12) or not np.isfinite(matrix).all():
            raise FloatingPointError("non-finite covariance")
        minimum = float(np.min(np.linalg.eigvalsh(matrix)))
        if minimum < 1e-12:
            matrix += np.eye(12) * (1e-12 - minimum)
        return matrix

    def _measurement_update(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        position_sigma: float,
        rotation_sigma: float,
    ) -> tuple[bool, float | None]:
        residual_position = position - self._p
        residual_rotation = _log_so3(
            _rotation_from_quaternion(quaternion) @ _rotation_from_quaternion(self._q).T
        )
        residual = np.concatenate((residual_position, residual_rotation))
        H = np.zeros((6, self._STATE_SIZE), dtype=float)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        measurement_covariance = np.zeros((6, 6), dtype=float)
        measurement_covariance[0:3, 0:3] = np.eye(3) * position_sigma**2
        measurement_covariance[3:6, 3:6] = np.eye(3) * rotation_sigma**2
        innovation_covariance = H @ self._P @ H.T + measurement_covariance
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        try:
            solved = np.linalg.solve(innovation_covariance, residual)
            d2 = float(residual @ solved)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return False, None
        if not math.isfinite(d2):
            return False, None
        if d2 > _INNOVATION_GATE:
            return False, d2
        try:
            gain = np.linalg.solve(innovation_covariance, H @ self._P).T
            delta = gain @ residual
            identity = np.eye(self._STATE_SIZE, dtype=float)
            joseph_left = identity - gain @ H
            updated = joseph_left @ self._P @ joseph_left.T + gain @ measurement_covariance @ gain.T
            updated = self._stabilize_covariance(updated)
            reset_jacobian = np.eye(self._STATE_SIZE, dtype=float)
            reset_jacobian[6:9, 6:9] = _left_jacobian(delta[6:9])
            updated = self._stabilize_covariance(reset_jacobian @ updated @ reset_jacobian.T)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return False, d2
        updated_position = self._p + delta[0:3]
        updated_velocity = self._v + delta[3:6]
        updated_quaternion = _quaternion_from_rotation(
            _exp_so3(delta[6:9]) @ _rotation_from_quaternion(self._q)
        )
        updated_angular_velocity = self._w + delta[9:12]
        if (
            not np.isfinite(updated_position).all()
            or not np.isfinite(updated_velocity).all()
            or not np.isfinite(updated_quaternion).all()
            or not np.isfinite(updated_angular_velocity).all()
        ):
            return False, d2
        self._p = updated_position
        self._v = updated_velocity
        self._q = updated_quaternion
        self._w = updated_angular_velocity
        self._P = updated
        return True, d2

    def update(
        self,
        stamp: float,
        cam_from_world: np.ndarray | None,
        *,
        epoch: int,
        position_sigma: float,
        rotation_sigma: float,
    ) -> dict:
        """Advance to ``stamp`` and optionally fuse one visual pose."""
        epoch_value = self._epoch_value(epoch)
        stamp_value = self._stamp_value(stamp)
        position_noise = self._measurement_sigma(position_sigma, "position_sigma")
        rotation_noise = self._measurement_sigma(rotation_sigma, "rotation_sigma")
        if (
            epoch_value is None
            or stamp_value is None
            or position_noise is None
            or rotation_noise is None
        ):
            return self._result(
                accepted=False,
                reset=False,
                innovation_mahalanobis=None,
                predicted_only=self._initialized,
            )

        reset_flag = False
        if self._epoch is None:
            self._epoch = epoch_value
        elif epoch_value != self._epoch:
            self.reset()
            self._epoch = epoch_value
            reset_flag = True
        elif self._stamp is not None and stamp_value <= self._stamp:
            return self._result(
                accepted=False,
                reset=False,
                innovation_mahalanobis=None,
                predicted_only=self._initialized,
            )
        if (
            self._initialized
            and self._last_accepted_stamp is not None
            and stamp_value - self._last_accepted_stamp > self.max_gap_s
        ):
            self.reset()
            self._epoch = epoch_value
            reset_flag = True

        components = None if cam_from_world is None else _pose_components(cam_from_world)
        if components is None:
            if not self._initialized or reset_flag:
                self._stamp = stamp_value
                return self._result(
                    accepted=False,
                    reset=reset_flag,
                    innovation_mahalanobis=None,
                    predicted_only=False,
                )
            self._propagate(stamp_value - float(self._stamp))
            self._stamp = stamp_value
            return self._result(
                accepted=False, reset=reset_flag, innovation_mahalanobis=None, predicted_only=True
            )

        position, quaternion = components
        if not self._initialized:
            self._initialize(position, quaternion, position_noise, rotation_noise)
            self._stamp = stamp_value
            self._last_accepted_stamp = stamp_value
            return self._result(
                accepted=True, reset=True, innovation_mahalanobis=0.0, predicted_only=False
            )
        self._propagate(stamp_value - float(self._stamp))
        self._stamp = stamp_value
        accepted, d2 = self._measurement_update(
            position, quaternion, position_noise, rotation_noise
        )
        if accepted:
            self._last_accepted_stamp = stamp_value
        return self._result(
            accepted=accepted,
            reset=reset_flag,
            innovation_mahalanobis=d2,
            predicted_only=not accepted,
        )
