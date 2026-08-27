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
