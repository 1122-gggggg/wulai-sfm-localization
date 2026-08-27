"""Single validation boundary for safety-relevant live operator settings."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


DEFAULT_MAX_TILT_DEG = 20.0
DEFAULT_MAX_VERTICAL_SPEED_MS = 2.0
DEFAULT_MAX_ROTATION_SPEED_DEGS = 20.0
DEFAULT_RTH_MIN_ALTITUDE_M = 20.0
DEFAULT_STREAM_LOSS_GRACE_S = 10.0
CRITICAL_BATTERY_PCT = 10.0
MIN_TAKEOFF_BATTERY_PCT = 30.0
NUDGE_PCT_MAX = 25
NUDGE_TTL_MIN_S = 0.1
NUDGE_TTL_MAX_S = 2.0


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _optional_positive(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    parsed = _finite(name, value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return parsed


@dataclass(frozen=True)
class LiveSafetyConfig:
    """Effective, normalized live-flight settings used by UI and backend."""

    nudge_pct: int
    nudge_pulse_s: float
    max_altitude_m: float | None
    max_distance_m: float | None
    max_tilt_deg: float
    max_vertical_speed_ms: float
    max_rotation_speed_degs: float
    rth_min_altitude_m: float
    stream_loss_grace_s: float
    min_takeoff_battery_pct: float
    distance_geofence: bool
    require_gps_for_geofence: bool

    @classmethod
    def resolve(
        cls,
        *,
        nudge_pct: object,
        nudge_pulse_s: object,
        max_altitude_m: object | None,
        max_distance_m: object | None,
        max_tilt_deg: object,
        max_vertical_speed_ms: object,
        max_rotation_speed_degs: object,
        rth_min_altitude_m: object,
        stream_loss_grace_s: object,
        min_takeoff_battery_pct: object,
        distance_geofence: bool,
        require_gps_for_geofence: bool,
    ) -> "LiveSafetyConfig":
        if isinstance(nudge_pct, bool) or not isinstance(nudge_pct, int):
            raise ValueError("nudge_pct must be an integer")
        pct = nudge_pct
        if not 1 <= pct <= NUDGE_PCT_MAX:
            raise ValueError(f"nudge_pct must be within [1, {NUDGE_PCT_MAX}]")

        pulse = _finite("nudge_pulse_s", nudge_pulse_s)
        pulse = min(NUDGE_TTL_MAX_S, max(NUDGE_TTL_MIN_S, pulse))
        altitude = _optional_positive("max_altitude_m", max_altitude_m)
        distance = _optional_positive("max_distance_m", max_distance_m)
        if (altitude is None) != (distance is None):
            raise ValueError("max_altitude_m and max_distance_m must be set together")

        tilt = _finite("max_tilt_deg", max_tilt_deg)
        vertical = _finite("max_vertical_speed_ms", max_vertical_speed_ms)
        rotation = _finite("max_rotation_speed_degs", max_rotation_speed_degs)
        rth = _finite("rth_min_altitude_m", rth_min_altitude_m)
        grace = _finite("stream_loss_grace_s", stream_loss_grace_s)
        battery = _finite("min_takeoff_battery_pct", min_takeoff_battery_pct)
        for name, value, maximum in (
            ("max_tilt_deg", tilt, DEFAULT_MAX_TILT_DEG),
            ("max_vertical_speed_ms", vertical, DEFAULT_MAX_VERTICAL_SPEED_MS),
            ("max_rotation_speed_degs", rotation, DEFAULT_MAX_ROTATION_SPEED_DEGS),
            ("rth_min_altitude_m", rth, DEFAULT_RTH_MIN_ALTITUDE_M),
        ):
            if not 0.0 < value <= maximum:
                raise ValueError(f"{name} must be within (0, {maximum:g}]")
        if not 0.0 <= grace <= DEFAULT_STREAM_LOSS_GRACE_S:
            raise ValueError(
                f"stream_loss_grace_s must be within [0, {DEFAULT_STREAM_LOSS_GRACE_S:g}]"
            )
        if not MIN_TAKEOFF_BATTERY_PCT <= battery <= 100.0:
            raise ValueError(
                "min_takeoff_battery_pct must be within "
                f"[{MIN_TAKEOFF_BATTERY_PCT:g}, 100]"
            )
        return cls(
            nudge_pct=pct,
            nudge_pulse_s=pulse,
            max_altitude_m=altitude,
            max_distance_m=distance,
            max_tilt_deg=tilt,
            max_vertical_speed_ms=vertical,
            max_rotation_speed_degs=rotation,
            rth_min_altitude_m=rth,
            stream_loss_grace_s=grace,
            min_takeoff_battery_pct=battery,
            distance_geofence=bool(distance_geofence),
            require_gps_for_geofence=bool(require_gps_for_geofence),
        )

    def as_log_fields(self) -> dict[str, object]:
        return asdict(self)

    def checksum(self) -> str:
        encoded = json.dumps(self.as_log_fields(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()
