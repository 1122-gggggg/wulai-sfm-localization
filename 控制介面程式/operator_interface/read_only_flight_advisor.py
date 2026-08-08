"""Pure, read-only map/telemetry fusion helpers.

This module deliberately has no Olympe, UI, route-execution, or aircraft-control
dependency.  It turns calibrated observations into estimates and *advisories*
only.  A caller must never treat its normalized axis recommendations as an
aircraft command without a separately approved flight-control integration.

Coordinate convention:

* Raw GLOMAP horizontal axes are ``x`` and ``z``; up is ``-y``.
* ``body_yaw_map_rad`` is a body-heading angle in that horizontal map plane,
  positive counter-clockwise from +x toward +z.
* A body-right velocity is positive to the aircraft's physical right.  Any
  hardware-specific sign convention stays outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


def wrap_pi(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    _require_finite("angle_rad", angle_rad)
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class MapVelocity:
    """Velocity in the declared map frame: +x, +z, and up, per second.

    Its unit is deliberately whatever the visual map declares.  A velocity
    derived from successive GLOMAP poses is therefore in map-units/s and needs
    no metre-scale calibration.
    """

    x: float
    z: float
    up: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("z", self.z), ("up", self.up)):
            _require_finite(name, value)


@dataclass(frozen=True)
class BodyVelocity:
    """Velocity in physical aircraft axes: forward, right, and up."""

    forward: float
    right: float
    up: float

    def __post_init__(self) -> None:
        for name, value in (
            ("forward", self.forward),
            ("right", self.right),
            ("up", self.up),
        ):
            _require_finite(name, value)


@dataclass(frozen=True)
class NedMapCalibration:
    """Explicit calibration needed before NED speeds can enter the map frame.

    ``north_yaw_map_rad`` is the heading of true north expressed in the map
    plane. ``map_units_per_meter`` is deliberately required: site profiles use
    scale-free map units, so assuming one map unit equals one metre is unsafe.
    """

    north_yaw_map_rad: float
    map_units_per_meter: float

    def __post_init__(self) -> None:
        _require_finite("north_yaw_map_rad", self.north_yaw_map_rad)
        _require_positive("map_units_per_meter", self.map_units_per_meter)


def ned_velocity_to_map(
    north_mps: float,
    east_mps: float,
    down_mps: float,
    calibration: NedMapCalibration,
) -> MapVelocity:
    """Convert NED velocity into calibrated map velocity.

    The result remains an observation, not a motion command.  Positive NED
    down becomes negative map-up, and the east axis is clockwise from north.
    """
    for name, value in (
        ("north_mps", north_mps),
        ("east_mps", east_mps),
        ("down_mps", down_mps),
    ):
        _require_finite(name, value)
    north_heading = calibration.north_yaw_map_rad
    east_heading = north_heading - math.pi / 2.0
    scale = calibration.map_units_per_meter
    return MapVelocity(
        x=scale * (
            float(north_mps) * math.cos(north_heading)
            + float(east_mps) * math.cos(east_heading)
        ),
        z=scale * (
            float(north_mps) * math.sin(north_heading)
            + float(east_mps) * math.sin(east_heading)
        ),
        up=-scale * float(down_mps),
    )


def map_velocity_to_body(
    target: MapVelocity,
    body_yaw_map_rad: float,
) -> BodyVelocity:
    """Rotate a desired map-frame velocity into aircraft body axes.

    At yaw=0, +x is forward and +z is left; this follows the declared
    right-handed ``(x, z, up)`` map convention.  The returned right value is
    physical body-right, rather than a device-specific roll sign.
    """
    _require_finite("body_yaw_map_rad", body_yaw_map_rad)
    cosine = math.cos(float(body_yaw_map_rad))
    sine = math.sin(float(body_yaw_map_rad))
    return BodyVelocity(
        forward=cosine * target.x + sine * target.z,
        right=sine * target.x - cosine * target.z,
        up=target.up,
    )


class AxisPid:
    """A bounded single-axis PID producing a normalized recommendation."""

    def __init__(
        self,
        *,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limit: float = 1.0,
        integral_limit: float = 1.0,
    ) -> None:
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd)):
            _require_finite(name, value)
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        _require_positive("output_limit", output_limit)
        _require_positive("integral_limit", integral_limit)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)
        self._integral = 0.0
        self._previous_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def step(self, *, target: float, measured: float, dt_s: float) -> float:
        """Return a bounded normalized advisory for one velocity axis."""
        for name, value in (("target", target), ("measured", measured)):
            _require_finite(name, value)
        _require_positive("dt_s", dt_s)
        error = float(target) - float(measured)
        candidate_integral = _clip(
            self._integral + error * float(dt_s), self.integral_limit
        )
        derivative = (
            0.0
            if self._previous_error is None
            else (error - self._previous_error) / float(dt_s)
        )
        unclipped = (
            self.kp * error
            + self.ki * candidate_integral
            + self.kd * derivative
        )
        result = _clip(unclipped, self.output_limit)
        # Do not wind the integrator farther into saturation.
        if abs(unclipped) <= self.output_limit or error * unclipped < 0.0:
            self._integral = candidate_integral
        self._previous_error = error
        return result


@dataclass(frozen=True)
class AxisRecommendation:
    """Normalized body-axis recommendations, all in [-1, 1].

    ``pitch_forward``, ``roll_right``, and ``gaz_up`` are not protocol values
    and are intentionally not converted to device command percentages here.
    """

    pitch_forward: float
    roll_right: float
    gaz_up: float
    target: BodyVelocity
    measured: BodyVelocity


class VelocityAdvisor:
    """Independent forward/right/up velocity PID advisories."""

    def __init__(
        self,
        *,
        forward: AxisPid,
        right: AxisPid,
        up: AxisPid,
    ) -> None:
        self.forward = forward
        self.right = right
        self.up = up

    def recommend(
        self,
        *,
        target: BodyVelocity,
        measured: BodyVelocity,
        dt_s: float,
    ) -> AxisRecommendation:
        return AxisRecommendation(
            pitch_forward=self.forward.step(
                target=target.forward, measured=measured.forward, dt_s=dt_s
            ),
            roll_right=self.right.step(
                target=target.right, measured=measured.right, dt_s=dt_s
            ),
            gaz_up=self.up.step(target=target.up, measured=measured.up, dt_s=dt_s),
            target=target,
            measured=measured,
        )


@dataclass(frozen=True)
class MapPose:
    """A timestamped pose in map X/Z/up coordinates."""

    x: float
    z: float
    up: float
    yaw_map_rad: float
    stamp_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x", self.x),
            ("z", self.z),
            ("up", self.up),
            ("yaw_map_rad", self.yaw_map_rad),
            ("stamp_s", self.stamp_s),
        ):
            _require_finite(name, value)


@dataclass(frozen=True)
class YawDeltaCheck:
    accepted: bool
    visual_delta_rad: float
    imu_delta_rad: float
    mismatch_rad: float
    reason: str


def check_visual_imu_yaw_delta(
    *,
    previous_visual_yaw_rad: float,
    visual_yaw_rad: float,
    previous_imu_yaw_rad: float,
    imu_yaw_rad: float,
    max_mismatch_rad: float,
) -> YawDeltaCheck:
    """Gate a visual update by its yaw increment, not a fixed sensor offset."""
    _require_positive("max_mismatch_rad", max_mismatch_rad)
    visual_delta = wrap_pi(float(visual_yaw_rad) - float(previous_visual_yaw_rad))
    imu_delta = wrap_pi(float(imu_yaw_rad) - float(previous_imu_yaw_rad))
    mismatch = abs(wrap_pi(visual_delta - imu_delta))
    accepted = mismatch <= float(max_mismatch_rad)
    return YawDeltaCheck(
        accepted=accepted,
        visual_delta_rad=visual_delta,
        imu_delta_rad=imu_delta,
        mismatch_rad=mismatch,
        reason="OK" if accepted else "VISUAL_IMU_YAW_INCREMENT_MISMATCH",
    )


def predict_short_horizon(
    *,
    pose: MapPose,
    velocity: MapVelocity,
    yaw_rate_rad_s: float,
    to_stamp_s: float,
    max_horizon_s: float = 0.20,
) -> MapPose | None:
    """Predict only over a bounded horizon; return ``None`` once it is stale."""
    _require_finite("yaw_rate_rad_s", yaw_rate_rad_s)
    _require_finite("to_stamp_s", to_stamp_s)
    _require_positive("max_horizon_s", max_horizon_s)
    dt_s = float(to_stamp_s) - pose.stamp_s
    if dt_s < 0.0 or dt_s > float(max_horizon_s):
        return None
    return MapPose(
        x=pose.x + velocity.x * dt_s,
        z=pose.z + velocity.z * dt_s,
        up=pose.up + velocity.up * dt_s,
        yaw_map_rad=wrap_pi(pose.yaw_map_rad + float(yaw_rate_rad_s) * dt_s),
        stamp_s=float(to_stamp_s),
    )


def visual_pose_velocity(
    previous: MapPose,
    current: MapPose,
    *,
    max_gap_s: float = 0.20,
) -> MapVelocity | None:
    """Derive a scale-free map velocity from two consecutive visual poses.

    ``None`` means that the pair is non-monotonic or too far apart to support
    short-horizon dead reckoning.  This intentionally does not accept ANAFI
    NED metre/s telemetry: without a measured map scale, that would mix units.
    """
    _require_positive("max_gap_s", max_gap_s)
    dt_s = current.stamp_s - previous.stamp_s
    if dt_s <= 0.0 or dt_s > float(max_gap_s):
        return None
    return MapVelocity(
        x=(current.x - previous.x) / dt_s,
        z=(current.z - previous.z) / dt_s,
        up=(current.up - previous.up) / dt_s,
    )


@dataclass(frozen=True)
class ScaleFreeFusionState:
    """Read-only state expressed solely in visual map units."""

    pose: MapPose
    velocity: MapVelocity
    roll_rad: float
    pitch_rad: float
    imu_yaw_rad: float


@dataclass(frozen=True)
class ScaleFreeFusionUpdate:
    """One fail-closed visual/IMU fusion result for shadow-mode consumers."""

    accepted: bool
    reason: str
    state: ScaleFreeFusionState | None
    yaw_check: YawDeltaCheck | None


class ScaleFreeVisualImuFusion:
    """Fuse visual map motion with IMU attitude without assuming map metres.

    Position and velocity remain in raw map units.  IMU roll/pitch are carried
    into the result and IMU yaw is used only to reject an inconsistent visual
    yaw increment.  NED velocity and barometric altitude are deliberately not
    observations of this state because neither can be aligned to a scale-free
    map without an independently measured conversion.

    This class is read-only: it has no Olympe, UI, route, or aircraft-control
    dependency and returns no protocol command.
    """

    def __init__(
        self,
        *,
        max_pose_gap_s: float = 0.20,
        max_yaw_mismatch_rad: float = math.radians(10.0),
    ) -> None:
        _require_positive("max_pose_gap_s", max_pose_gap_s)
        _require_positive("max_yaw_mismatch_rad", max_yaw_mismatch_rad)
        self.max_pose_gap_s = float(max_pose_gap_s)
        self.max_yaw_mismatch_rad = float(max_yaw_mismatch_rad)
        self._previous_visual: MapPose | None = None
        self._previous_imu_yaw_rad: float | None = None

    def reset(self) -> None:
        self._previous_visual = None
        self._previous_imu_yaw_rad = None

    def update(
        self,
        *,
        visual_pose: MapPose,
        roll_rad: float,
        pitch_rad: float,
        imu_yaw_rad: float,
    ) -> ScaleFreeFusionUpdate:
        """Accept a consecutive visual pose only when its IMU yaw delta agrees."""
        for name, value in (
            ("roll_rad", roll_rad),
            ("pitch_rad", pitch_rad),
            ("imu_yaw_rad", imu_yaw_rad),
        ):
            _require_finite(name, value)

        previous = self._previous_visual
        previous_imu_yaw = self._previous_imu_yaw_rad
        if previous is None or previous_imu_yaw is None:
            self._previous_visual = visual_pose
            self._previous_imu_yaw_rad = float(imu_yaw_rad)
            return ScaleFreeFusionUpdate(
                accepted=False,
                reason="INITIALIZING_NEEDS_SECOND_VISUAL_POSE",
                state=None,
                yaw_check=None,
            )

        dt_s = visual_pose.stamp_s - previous.stamp_s
        if dt_s <= 0.0:
            return ScaleFreeFusionUpdate(
                accepted=False,
                reason="NON_MONOTONIC_VISUAL_TIMESTAMP",
                state=None,
                yaw_check=None,
            )
        if dt_s > self.max_pose_gap_s:
            # A long gap has no valid visual velocity.  Re-seed and require one
            # further consecutive visual observation before shadow guidance.
            self._previous_visual = visual_pose
            self._previous_imu_yaw_rad = float(imu_yaw_rad)
            return ScaleFreeFusionUpdate(
                accepted=False,
                reason="VISUAL_POSE_GAP_TOO_LARGE_REINITIALIZED",
                state=None,
                yaw_check=None,
            )

        yaw_check = check_visual_imu_yaw_delta(
            previous_visual_yaw_rad=previous.yaw_map_rad,
            visual_yaw_rad=visual_pose.yaw_map_rad,
            previous_imu_yaw_rad=previous_imu_yaw,
            imu_yaw_rad=imu_yaw_rad,
            max_mismatch_rad=self.max_yaw_mismatch_rad,
        )
        if not yaw_check.accepted:
            # Preserve the last accepted baselines: a later visual fix must
            # agree with the accumulated IMU rotation before it can recover.
            return ScaleFreeFusionUpdate(
                accepted=False,
                reason=yaw_check.reason,
                state=None,
                yaw_check=yaw_check,
            )

        velocity = visual_pose_velocity(
            previous,
            visual_pose,
            max_gap_s=self.max_pose_gap_s,
        )
        assert velocity is not None
        state = ScaleFreeFusionState(
            pose=visual_pose,
            velocity=velocity,
            roll_rad=float(roll_rad),
            pitch_rad=float(pitch_rad),
            imu_yaw_rad=float(imu_yaw_rad),
        )
        self._previous_visual = visual_pose
        self._previous_imu_yaw_rad = float(imu_yaw_rad)
        return ScaleFreeFusionUpdate(
            accepted=True,
            reason="OK",
            state=state,
            yaw_check=yaw_check,
        )


@dataclass(frozen=True)
class FusedState:
    pose: MapPose
    velocity: MapVelocity
    roll_rad: float
    pitch_rad: float


class ConstantVelocityKalman:
    """Small, read-only constant-velocity Kalman estimator.

    State order is ``[x, z, up, vx, vz, vup, roll, pitch, yaw]``.  Velocity
    observations must already be calibrated into map-units per second and yaw
    must already be expressed in map coordinates.  This avoids silently mixing
    an ANAFI NED observation with an unscaled visual map.
    """

    _SIZE = 9

    def __init__(
        self,
        *,
        max_prediction_s: float = 0.20,
        initial_variance: float = 10.0,
        position_process_variance: float = 0.02,
        velocity_process_variance: float = 0.10,
        attitude_process_variance: float = math.radians(1.0) ** 2,
    ) -> None:
        for name, value in (
            ("max_prediction_s", max_prediction_s),
            ("initial_variance", initial_variance),
            ("position_process_variance", position_process_variance),
            ("velocity_process_variance", velocity_process_variance),
            ("attitude_process_variance", attitude_process_variance),
        ):
            _require_positive(name, value)
        self.max_prediction_s = float(max_prediction_s)
        self.position_process_variance = float(position_process_variance)
        self.velocity_process_variance = float(velocity_process_variance)
        self.attitude_process_variance = float(attitude_process_variance)
        self._state = np.zeros(self._SIZE, dtype=float)
        self._covariance = np.eye(self._SIZE, dtype=float) * float(initial_variance)
        self._stamp_s: float | None = None

    def _advance(self, stamp_s: float) -> bool:
        _require_finite("stamp_s", stamp_s)
        stamp_s = float(stamp_s)
        if self._stamp_s is None:
            self._stamp_s = stamp_s
            return True
        dt_s = stamp_s - self._stamp_s
        if dt_s < 0.0:
            raise ValueError("timestamps must be monotonic")
        self._stamp_s = stamp_s
        if dt_s > self.max_prediction_s:
            return False
        transition = np.eye(self._SIZE, dtype=float)
        transition[0, 3] = dt_s
        transition[1, 4] = dt_s
        transition[2, 5] = dt_s
        process = np.diag(
            [
                self.position_process_variance * dt_s * dt_s,
                self.position_process_variance * dt_s * dt_s,
                self.position_process_variance * dt_s * dt_s,
                self.velocity_process_variance * dt_s,
                self.velocity_process_variance * dt_s,
                self.velocity_process_variance * dt_s,
                self.attitude_process_variance * dt_s,
                self.attitude_process_variance * dt_s,
                self.attitude_process_variance * dt_s,
            ]
        )
        self._state = transition @ self._state
        self._state[8] = wrap_pi(float(self._state[8]))
        self._covariance = transition @ self._covariance @ transition.T + process
        return True

    def _update(
        self,
        *,
        values: tuple[float, ...],
        indices: tuple[int, ...],
        variance: float,
        angular_index: int | None = None,
    ) -> None:
        _require_positive("variance", variance)
        if len(values) != len(indices):
            raise ValueError("values and indices must have the same length")
        for value in values:
            _require_finite("measurement", value)
        observation = np.asarray(values, dtype=float)
        matrix = np.zeros((len(indices), self._SIZE), dtype=float)
        for row, index in enumerate(indices):
            matrix[row, index] = 1.0
        innovation = observation - matrix @ self._state
        if angular_index is not None:
            row = indices.index(angular_index)
            innovation[row] = wrap_pi(float(innovation[row]))
        noise = np.eye(len(indices), dtype=float) * float(variance)
        innovation_covariance = matrix @ self._covariance @ matrix.T + noise
        gain = np.linalg.solve(innovation_covariance, matrix @ self._covariance).T
        self._state = self._state + gain @ innovation
        self._state[8] = wrap_pi(float(self._state[8]))
        identity = np.eye(self._SIZE, dtype=float)
        residual = identity - gain @ matrix
        self._covariance = residual @ self._covariance @ residual.T + gain @ noise @ gain.T

    def update_visual(
        self,
        pose: MapPose,
        *,
        position_variance: float,
        yaw_variance: float,
    ) -> FusedState:
        """Fuse an accepted visual pose; a long gap does not dead-reckon it."""
        self._advance(pose.stamp_s)
        self._update(
            values=(pose.x, pose.z, pose.up),
            indices=(0, 1, 2),
            variance=position_variance,
        )
        self._update(
            values=(pose.yaw_map_rad,),
            indices=(8,),
            variance=yaw_variance,
            angular_index=8,
        )
        return self.estimate()

    def update_telemetry(
        self,
        *,
        stamp_s: float,
        velocity: MapVelocity,
        roll_rad: float,
        pitch_rad: float,
        yaw_map_rad: float,
        velocity_variance: float,
        attitude_variance: float,
    ) -> FusedState:
        """Fuse calibrated map velocity and flight-controller attitude."""
        for name, value in (
            ("roll_rad", roll_rad),
            ("pitch_rad", pitch_rad),
            ("yaw_map_rad", yaw_map_rad),
        ):
            _require_finite(name, value)
        self._advance(stamp_s)
        self._update(
            values=(velocity.x, velocity.z, velocity.up),
            indices=(3, 4, 5),
            variance=velocity_variance,
        )
        self._update(
            values=(roll_rad, pitch_rad, yaw_map_rad),
            indices=(6, 7, 8),
            variance=attitude_variance,
            angular_index=8,
        )
        return self.estimate()

    def update_altitude(
        self,
        *,
        stamp_s: float,
        up: float,
        variance: float,
    ) -> FusedState:
        """Fuse an altitude observation already aligned to the map-up origin."""
        _require_finite("up", up)
        self._advance(stamp_s)
        self._update(values=(up,), indices=(2,), variance=variance)
        return self.estimate()

    def predict(self, to_stamp_s: float) -> FusedState | None:
        """Advance briefly, or return ``None`` instead of extending dead reckoning."""
        if not self._advance(to_stamp_s):
            return None
        return self.estimate()

    def estimate(self) -> FusedState:
        if self._stamp_s is None:
            raise RuntimeError("the estimator has no timestamped observations")
        return FusedState(
            pose=MapPose(
                x=float(self._state[0]),
                z=float(self._state[1]),
                up=float(self._state[2]),
                yaw_map_rad=float(self._state[8]),
                stamp_s=self._stamp_s,
            ),
            velocity=MapVelocity(
                x=float(self._state[3]),
                z=float(self._state[4]),
                up=float(self._state[5]),
            ),
            roll_rad=float(self._state[6]),
            pitch_rad=float(self._state[7]),
        )


class SafetyRecommendation(str, Enum):
    NORMAL = "normal"
    DRIFT_WARNING = "drift_warning"
    HOVER_RECOMMENDED = "hover_recommended"
    MANUAL_TAKEOVER_RECOMMENDED = "manual_takeover_recommended"


@dataclass(frozen=True)
class SafetyAssessment:
    recommendation: SafetyRecommendation
    reason: str
    roll_deg: float
    pitch_deg: float
    ground_speed_mps: float


class SafetyAdvisor:
    """Read-only attitude and drift escalation with no automatic landing path."""

    def __init__(
        self,
        *,
        hover_tilt_deg: float = 25.0,
        manual_tilt_deg: float = 35.0,
        drift_speed_mps: float = 1.5,
        drift_samples_before_hover: int = 3,
    ) -> None:
        _require_positive("hover_tilt_deg", hover_tilt_deg)
        _require_positive("manual_tilt_deg", manual_tilt_deg)
        _require_positive("drift_speed_mps", drift_speed_mps)
        if float(manual_tilt_deg) <= float(hover_tilt_deg):
            raise ValueError("manual_tilt_deg must be greater than hover_tilt_deg")
        if not isinstance(drift_samples_before_hover, int) or drift_samples_before_hover < 1:
            raise ValueError("drift_samples_before_hover must be a positive integer")
        self.hover_tilt_deg = float(hover_tilt_deg)
        self.manual_tilt_deg = float(manual_tilt_deg)
        self.drift_speed_mps = float(drift_speed_mps)
        self.drift_samples_before_hover = drift_samples_before_hover
        self._drift_samples = 0

    def assess(
        self,
        *,
        roll_rad: float,
        pitch_rad: float,
        ground_speed_mps: float,
        hover_is_expected: bool,
        telemetry_fresh: bool = True,
    ) -> SafetyAssessment:
        """Return a recommendation only; no state transition controls an aircraft."""
        for name, value in (
            ("roll_rad", roll_rad),
            ("pitch_rad", pitch_rad),
            ("ground_speed_mps", ground_speed_mps),
        ):
            _require_finite(name, value)
        roll_deg = math.degrees(float(roll_rad))
        pitch_deg = math.degrees(float(pitch_rad))
        speed = abs(float(ground_speed_mps))
        peak_tilt = max(abs(roll_deg), abs(pitch_deg))
        if not telemetry_fresh:
            self._drift_samples = 0
            return SafetyAssessment(
                SafetyRecommendation.MANUAL_TAKEOVER_RECOMMENDED,
                "TELEMETRY_STALE",
                roll_deg,
                pitch_deg,
                speed,
            )
        if peak_tilt >= self.manual_tilt_deg:
            self._drift_samples = 0
            return SafetyAssessment(
                SafetyRecommendation.MANUAL_TAKEOVER_RECOMMENDED,
                "EXCESSIVE_TILT",
                roll_deg,
                pitch_deg,
                speed,
            )
        if peak_tilt >= self.hover_tilt_deg:
            self._drift_samples = 0
            return SafetyAssessment(
                SafetyRecommendation.HOVER_RECOMMENDED,
                "HIGH_TILT",
                roll_deg,
                pitch_deg,
                speed,
            )
        if hover_is_expected and speed >= self.drift_speed_mps:
            self._drift_samples += 1
            recommendation = (
                SafetyRecommendation.HOVER_RECOMMENDED
                if self._drift_samples >= self.drift_samples_before_hover
                else SafetyRecommendation.DRIFT_WARNING
            )
            return SafetyAssessment(
                recommendation,
                "UNEXPECTED_DRIFT_WHILE_HOVER_EXPECTED",
                roll_deg,
                pitch_deg,
                speed,
            )
        self._drift_samples = 0
        return SafetyAssessment(
            SafetyRecommendation.NORMAL,
            "OK",
            roll_deg,
            pitch_deg,
            speed,
        )
