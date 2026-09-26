"""Offline ESKF velocity updates using calibrated, firmware-fused telemetry.

No raw IMU integration and no connection to the production controller. Horizontal
velocity is observed in the gravity-aligned map plane; uncalibrated altitude and
body orientation are deliberately not fused into the camera pose.
"""

from __future__ import annotations

import numpy as np

from pose_filter_experiment import (
    VisualPoseESKF,
    _exp_so3,
    _left_jacobian,
    _quaternion_from_rotation,
    _rotation_from_quaternion,
)


class TelemetryPoseESKF(VisualPoseESKF):
    """Same visual filter and gap policy, plus a two-dimensional velocity factor."""

    def __init__(self, projection, **kwargs):
        super().__init__(**kwargs)
        self.projection = np.asarray(projection, dtype=float)
        if (
            self.projection.shape != (2, 3)
            or not np.isfinite(self.projection).all()
            or not np.allclose(self.projection @ self.projection.T, np.eye(2), atol=1e-6)
        ):
            raise ValueError("projection must contain two orthonormal map axes")
        self._velocity_stamp = None

    def reset(self):
        super().reset()
        self._velocity_stamp = None

    def update_velocity(self, stamp, velocity, *, epoch, sigma):
        """Fuse each source sample once; telemetry cannot extend the visual gap budget."""
        measurement = np.asarray(velocity, dtype=float)
        if (
            measurement.shape != (2,)
            or not np.isfinite(measurement).all()
            or not np.isfinite(stamp)
            or not np.isfinite(sigma)
            or sigma <= 0
        ):
            return {"accepted": False, "reason": "invalid_velocity"}
        if self._velocity_stamp is not None and stamp <= self._velocity_stamp:
            return {"accepted": False, "reason": "duplicate_or_reversed_velocity"}
        if self._stamp is not None and stamp < self._stamp:
            return {"accepted": False, "reason": "out_of_order_velocity"}
        if self._stamp != stamp or self._epoch != epoch:
            super().update(stamp, None, epoch=epoch, position_sigma=sigma, rotation_sigma=0.1)
        self._velocity_stamp = stamp
        if not self._initialized:
            return {"accepted": False, "reason": "no_recent_visual_anchor"}
        H = np.zeros((2, self._STATE_SIZE))
        H[:, 3:6] = self.projection
        residual = measurement - self.projection @ self._v
        noise = np.eye(2) * sigma**2
        innovation = H @ self._P @ H.T + noise
        d2 = float(residual @ np.linalg.solve(innovation, residual))
        result = {
            "accepted": False,
            "reason": "velocity_innovation_rejected",
            "innovation_mahalanobis": d2,
            "residual": residual.tolist(),
        }
        if not np.isfinite(d2) or d2 > 9.210340371976184:  # chi-square 99%, 2 DoF
            return result
        gain = np.linalg.solve(innovation, H @ self._P).T
        delta = gain @ residual
        left = np.eye(self._STATE_SIZE) - gain @ H
        covariance = left @ self._P @ left.T + gain @ noise @ gain.T
        reset = np.eye(self._STATE_SIZE)
        reset[6:9, 6:9] = _left_jacobian(delta[6:9])
        self._P = self._stabilize_covariance(reset @ covariance @ reset.T)
        self._p += delta[:3]
        self._v += delta[3:6]
        self._q = _quaternion_from_rotation(
            _exp_so3(delta[6:9]) @ _rotation_from_quaternion(self._q)
        )
        self._w += delta[9:12]
        return {**result, "accepted": True, "reason": "velocity_update"}

    def diagnostics(self):
        return {
            "position_covariance_trace": float(np.trace(self._P[:3, :3])),
            "covariance_diagonal": np.diag(self._P).tolist(),
            "visual_age_s": None
            if self._last_accepted_stamp is None
            else self._stamp - self._last_accepted_stamp,
        }
