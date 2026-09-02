"""Host-monotonic fused-state buffer. Interpolates only; never extrapolates."""

from __future__ import annotations

import math
from collections import deque

from pose_guided.se3 import ned_rpy_to_rotation, quaternion_wxyz_from_matrix, slerp
from pose_guided.types import FusedOdometrySample, OdometryFrame


class ImuStateProvider:
    """Ring buffer of fused samples. ANAFI path is NED_FUSED, not map pose."""

    def __init__(self, *, maxlen: int = 64) -> None:
        if isinstance(maxlen, bool) or not isinstance(maxlen, int) or maxlen < 2:
            raise ValueError("maxlen must be an integer >= 2")
        self._samples: deque[FusedOdometrySample] = deque(maxlen=maxlen)

    def observe(self, sample: FusedOdometrySample | None) -> None:
        if sample is None:
            return
        if not math.isfinite(sample.timestamp):
            return
        if self._samples and sample.timestamp < self._samples[-1].timestamp:
            return
        self._samples.append(sample)

    def latest(self) -> FusedOdometrySample | None:
        return self._samples[-1] if self._samples else None

    def clear(self) -> None:
        self._samples.clear()

    def interpolate(
        self, timestamp: float, *, max_sync_error_s: float
    ) -> FusedOdometrySample | None:
        if (
            not math.isfinite(timestamp)
            or not math.isfinite(max_sync_error_s)
            or max_sync_error_s <= 0.0
        ):
            return None
        if not self._samples:
            return None
        if (
            timestamp < self._samples[0].timestamp - 1e-9
            or timestamp > self._samples[-1].timestamp + 1e-9
        ):
            nearest = min(self._samples, key=lambda sample: abs(sample.timestamp - timestamp))
            if abs(nearest.timestamp - timestamp) <= max_sync_error_s:
                return nearest
            return None
        before, after = _bracket_samples(self._samples, timestamp)
        if before is None or after is None:
            return None
        if before.timestamp == after.timestamp:
            return before
        span = after.timestamp - before.timestamp
        if span <= 0.0 or span > max_sync_error_s * 2.0:
            return None
        fraction = (timestamp - before.timestamp) / span
        return _interpolate_pair(before, after, timestamp, fraction)


def _bracket_samples(
    samples: deque[FusedOdometrySample], timestamp: float
) -> tuple[FusedOdometrySample | None, FusedOdometrySample | None]:
    before: FusedOdometrySample | None = None
    after: FusedOdometrySample | None = None
    for sample in samples:
        if sample.timestamp <= timestamp:
            before = sample
        if sample.timestamp >= timestamp:
            after = sample
            break
    return before, after


def _interpolate_pair(
    before: FusedOdometrySample,
    after: FusedOdometrySample,
    timestamp: float,
    fraction: float,
) -> FusedOdometrySample:
    roll = pitch = yaw = None
    quaternion = None
    if before.has_attitude and after.has_attitude:
        q0 = quaternion_wxyz_from_matrix(ned_rpy_to_rotation(before.roll, before.pitch, before.yaw))
        q1 = quaternion_wxyz_from_matrix(ned_rpy_to_rotation(after.roll, after.pitch, after.yaw))
        quaternion = tuple(float(value) for value in slerp(q0, q1, fraction))
        roll = _lerp(before.roll, after.roll, fraction)
        pitch = _lerp(before.pitch, after.pitch, fraction)
        yaw = _lerp_angle(before.yaw, after.yaw, fraction)
    velocity = None
    if before.velocity_ned is not None and after.velocity_ned is not None:
        velocity = tuple(
            _lerp(left, right, fraction)
            for left, right in zip(before.velocity_ned, after.velocity_ned)
        )
    position = None
    if (
        before.frame is OdometryFrame.MAP
        and after.frame is OdometryFrame.MAP
        and before.position is not None
        and after.position is not None
    ):
        position = tuple(
            _lerp(left, right, fraction) for left, right in zip(before.position, after.position)
        )
    gnss = min(
        (sample for sample in (before, after) if sample.has_gnss),
        key=lambda sample: abs(float(sample.geodetic_timestamp) - timestamp),
        default=None,
    )
    frame = before.frame if before.frame == after.frame else OdometryFrame.UNKNOWN
    return FusedOdometrySample(
        timestamp=timestamp,
        frame=frame,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        velocity_ned=velocity,
        position=position,
        quaternion_wxyz=quaternion,
        geodetic_lla=None if gnss is None else gnss.geodetic_lla,
        geodetic_accuracy_m=None if gnss is None else gnss.geodetic_accuracy_m,
        geodetic_timestamp=None if gnss is None else gnss.geodetic_timestamp,
        source=f"{before.source}+interp",
    )


def _lerp(start: float, end: float, fraction: float) -> float:
    return float(start) + fraction * (float(end) - float(start))


def _lerp_angle(start: float, end: float, fraction: float) -> float:
    delta = (float(end) - float(start) + math.pi) % (2.0 * math.pi) - math.pi
    return float(start) + fraction * delta
