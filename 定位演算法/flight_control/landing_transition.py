"""Pure route-completion landing and ground-speed transitions."""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast


LANDING_SPEED_THRESHOLD_MPS = 0.10
GROUND_SPEED_MAX_AGE_S = 0.50
LandingOutcome = Literal["land", "hover"]


@dataclass(frozen=True)
class LandingTransition:
    """Side-effect-free outcome for one route-completion command."""

    outcome: LandingOutcome
    reason: str
    ground_speed_mps: float | None
    log_fields: tuple[tuple[str, float | None], ...]


def landing_speed_allows_land(
    sample: object,
    now: float,
    *,
    threshold_mps: float = LANDING_SPEED_THRESHOLD_MPS,
    max_age_s: float = GROUND_SPEED_MAX_AGE_S,
) -> tuple[bool, str, float | None]:
    """Fail-closed physical-speed gate for normal end-of-route landing."""
    if sample is None:
        return False, "ground speed unavailable", None
    try:
        speed, stamp = cast(Iterable[object], sample)
        if isinstance(speed, bool) or isinstance(stamp, bool):
            raise ValueError
        speed = float(cast(float | str, speed))
        stamp = float(cast(float | str, stamp))
        now = float(now)
    except (TypeError, ValueError, OverflowError):
        return False, "ground speed invalid", None
    if not all(math.isfinite(value) for value in (speed, stamp, now)) or speed < 0.0:
        return False, "ground speed invalid", None
    age_s = now - stamp
    if age_s < -0.05:
        return False, "ground speed timestamp is in the future", speed
    if age_s > float(max_age_s):
        return False, f"ground speed stale ({age_s:.2f}s)", speed
    if speed > float(threshold_mps):
        return False, f"ground speed {speed:.3f} m/s > {threshold_mps:.3f} m/s", speed
    return True, f"ground speed {speed:.3f} m/s <= {threshold_mps:.3f} m/s", speed


LANDING_STABLE_S = 1.0
LANDING_STABLE_SAMPLES = 3


class RouteLandingConfirmation:
    """Continuous in-circle, low-speed confirmation before normal landing.

    Only normal route completion uses this guard; manual/EMERGENCY/ABORT and
    forced landing keep their existing single-sample semantics. The guard
    accumulates distinct fresh low-speed captures while the aircraft stays
    inside the final sphere with map-confirmed poses. Any out-of-circle pose,
    stale/missing/invalid speed, high speed, or long capture gap resets the
    window. A reseed-confirming pause holds the window without adding a
    sample; if the pause outlasts max_position_age_s the window resets.
    """

    def __init__(self, *, max_position_age_s: float = 0.5) -> None:
        try:
            age = float(max_position_age_s)
        except (TypeError, ValueError, OverflowError):
            age = 0.5
        if not math.isfinite(age) or age <= 0.0:
            age = 0.5
        self.max_position_age_s = float(age)
        self.reset()

    def reset(self) -> None:
        self._inside_since: float | None = None
        self._first_capture: float | None = None
        self._last_capture: float | None = None
        self._last_speed_stamp: float | None = None
        self._samples = 0
        self._last_position_time: float | None = None

    @property
    def samples(self) -> int:
        return int(self._samples)

    @property
    def span_s(self) -> float:
        if self._first_capture is None or self._last_capture is None:
            return 0.0
        return max(0.0, float(self._last_capture) - float(self._first_capture))

    def _accept_capture(self, sample: object, now_f: float, speed: float | None) -> bool:
        try:
            _speed_value = float(speed) if speed is not None else float("nan")
            capture = float(sample[1]) if isinstance(sample, (tuple, list)) else float("nan")
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return False
        if not math.isfinite(capture):
            self.reset()
            return False
        if self._last_speed_stamp is not None and not capture > self._last_speed_stamp:
            # Repeated or reordered captures add no evidence and cannot release.
            return False
        if self._first_capture is not None and capture < self._first_capture:
            self.reset()
            return False
        if self._last_speed_stamp is not None and capture - self._last_speed_stamp > GROUND_SPEED_MAX_AGE_S:
            self.reset()
            self._inside_since = now_f
            self._last_position_time = now_f
            self._first_capture = capture
            self._last_capture = capture
            self._last_speed_stamp = capture
            self._samples = 1
            return False
        if self._inside_since is not None and capture < self._inside_since:
            # A capture older than circle entry cannot prove settled flight.
            return False
        if self._first_capture is None:
            self._first_capture = capture
        self._last_capture = capture
        self._last_speed_stamp = capture
        self._samples += 1
        if int(self._samples) < LANDING_STABLE_SAMPLES:
            return False
        if float(self._last_capture - self._first_capture) < LANDING_STABLE_S - 1e-9:
            return False
        return True
    def update(self, sample: object, now: float, *, position_ok: bool, reseed_pause: bool = False) -> bool:
        try:
            now_f = float(now)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return False
        if not math.isfinite(now_f):
            self.reset()
            return False
        if not bool(position_ok):
            self.reset()
            return False
        if self._inside_since is None:
            self._inside_since = now_f
        self._last_position_time = now_f
        if bool(reseed_pause):
            # Hold the window during the brief reseed-confirming policy; a long
            # pause means the position evidence itself went stale.
            if self._last_capture is not None and now_f - self._last_capture > float(self.max_position_age_s):
                self.reset()
                return False
            if self._first_capture is not None and now_f - self._first_capture > float(self.max_position_age_s) and self._samples == 0:
                self.reset()
                return False
            return False
        allowed, _reason, speed = landing_speed_allows_land(
            sample, now_f, max_age_s=GROUND_SPEED_MAX_AGE_S,
        )
        if not allowed:
            self.reset()
            # Re-enter the circle on this tick so a still-valid position does
            # not need one extra tick to restart the window.
            self._inside_since = now_f
            self._last_position_time = now_f
            return False
        return self._accept_capture(sample, now_f, speed)



def decide_route_completion_landing(
    action: str,
    ground_speed_sample: object,
    now: float,
    *,
    threshold_mps: float = LANDING_SPEED_THRESHOLD_MPS,
    max_age_s: float = GROUND_SPEED_MAX_AGE_S,
) -> LandingTransition:
    """Choose LAND or hover while preserving route-completion reason strings."""
    speed_mps = None
    log_fields: tuple[tuple[str, float | None], ...] = ()
    if action == "LAND":
        speed_ok, speed_reason, speed_mps = landing_speed_allows_land(
            ground_speed_sample,
            now,
            threshold_mps=threshold_mps,
            max_age_s=max_age_s,
        )
        if not speed_ok:
            return LandingTransition(
                "hover",
                f"route complete; {speed_reason} -> hover",
                speed_mps,
                (("ground_speed_mps", speed_mps),),
            )
        log_fields = (("ground_speed_mps", speed_mps),)
    reason = "pending inspection abort -> land" if action == "ABORT" else "route complete -> land"
    return LandingTransition("land", reason, speed_mps, log_fields)
