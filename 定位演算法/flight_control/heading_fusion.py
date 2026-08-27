"""Map-frame heading fusion and Olympe ground-speed helpers."""
from __future__ import annotations

import math
import os
import time


OFFSET_EMA = 0.25           # visual-to-attitude offset smoothing after first anchor


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {raw!r}") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be finite and in [{minimum}, {maximum}], got {raw!r}")
    return value


VISUAL_IMU_YAW_MISMATCH_RAD = math.radians(_env_float(
    "SFM_VISUAL_IMU_YAW_MISMATCH_DEG", 10.0, minimum=1.0, maximum=180.0))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class HeadingEstimator:
    """Fuse visual MapFrame heading with Olympe's clockwise-from-North NED yaw."""

    def __init__(
        self,
        map_frame=None,
        *,
        max_increment_mismatch_rad: float = VISUAL_IMU_YAW_MISMATCH_RAD,
    ):
        import real_path_follow_controller as rpf
        self.map_frame = map_frame or rpf.LEGACY_MAP_FRAME
        mismatch_limit = float(max_increment_mismatch_rad)
        if not math.isfinite(mismatch_limit) or mismatch_limit <= 0.0:
            raise ValueError("max_increment_mismatch_rad must be finite and positive")
        self.max_increment_mismatch_rad = mismatch_limit
        self.offset = None
        self._last_visual_heading = None
        self._previous_visual_heading = None
        self._previous_attitude_heading = None
        self.last_increment_mismatch_rad = None

    @staticmethod
    def _olympe_to_map_ccw(olympe_yaw: float) -> float:
        # Olympe AttitudeChanged.yaw is NED: clockwise from North. MapFrame
        # heading is counter-clockwise from East.
        return _wrap(math.pi / 2.0 - float(olympe_yaw))

    def update(self, visual_map_yaw: float, olympe_yaw: float | None) -> bool:
        """Accept one new visual anchor when its yaw increment agrees with IMU."""
        try:
            visual = _wrap(float(visual_map_yaw))
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(visual):
            return False
        if olympe_yaw is None:
            self._last_visual_heading = visual
            return True
        try:
            attitude = self._olympe_to_map_ccw(float(olympe_yaw))
        except (TypeError, ValueError, OverflowError):
            self._last_visual_heading = visual
            return True
        if not math.isfinite(attitude):
            self._last_visual_heading = visual
            return True
        if (
            self._previous_visual_heading is not None
            and self._previous_attitude_heading is not None
        ):
            visual_delta = _wrap(visual - self._previous_visual_heading)
            attitude_delta = _wrap(attitude - self._previous_attitude_heading)
            mismatch = abs(_wrap(visual_delta - attitude_delta))
            self.last_increment_mismatch_rad = mismatch
            if mismatch > self.max_increment_mismatch_rad:
                return False
        else:
            self.last_increment_mismatch_rad = None
        self._last_visual_heading = visual
        new = _wrap(visual - attitude)
        self.offset = new if self.offset is None else _wrap(
            self.offset + OFFSET_EMA * _wrap(new - self.offset)
        )
        self._previous_visual_heading = visual
        self._previous_attitude_heading = attitude
        return True

    def heading(self, olympe_yaw: float | None) -> float | None:
        if olympe_yaw is not None and self.offset is not None:
            try:
                attitude = self._olympe_to_map_ccw(float(olympe_yaw))
            except (TypeError, ValueError, OverflowError):
                attitude = float("nan")
            if math.isfinite(attitude):
                return _wrap(attitude + self.offset)
        return self._last_visual_heading


def olympe_yaw_of(drone) -> float | None:
    """Latest fused body yaw (radians, drone NED frame) or None if not yet received."""
    from olympe.messages.ardrone3.PilotingState import AttitudeChanged
    try:
        st = drone.get_state(AttitudeChanged)          # raises KeyError if never received
    except (KeyError, RuntimeError):
        return None
    return None if not st else float(st["yaw"])


class OlympeGroundSpeedTracker:
    """Read fresh NED horizontal speed from Olympe ``SpeedChanged`` events.

    ``get_state`` alone cannot distinguish a new event from a cached value.  The
    event UUID and SDK receipt time make the landing gate fail closed when speed
    telemetry stops updating.
    """

    def __init__(self, drone, message=None, *, now=time.monotonic, wall_now=time.time):
        if message is None:
            from olympe.messages.ardrone3.PilotingState import SpeedChanged
            message = SpeedChanged
        self.drone = drone
        self.message = message
        self._now = now
        self._wall_now = wall_now
        self._marker = None
        self._sample = None

    def sample(self) -> tuple[float, float] | None:
        """Return ``(horizontal_mps, receipt_monotonic_s)`` or ``None``."""
        try:
            event = self.drone.get_last_event(self.message)
            marker = str(event.uuid)
            state = dict(event.args)
            now = float(self._now())
            event_age_s = float(self._wall_now()) - float(event.date.timestamp())
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return None
        if (not math.isfinite(now) or not math.isfinite(event_age_s)
                or event_age_s < -0.05):
            return None
        if marker != self._marker:
            self._marker = marker
            self._sample = (state, now - max(0.0, event_age_s))
        if self._sample is None:
            return None
        state, stamp = self._sample
        try:
            speed_x = float(state["speedX"])
            speed_y = float(state["speedY"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in (speed_x, speed_y, stamp)):
            return None
        return math.hypot(speed_x, speed_y), float(stamp)
