from __future__ import annotations

import pytest

from live_safety_config import LiveSafetyConfig


def _resolve(**overrides) -> LiveSafetyConfig:
    values = {
        "nudge_pct": 8,
        "nudge_pulse_s": 0.2,
        "max_altitude_m": 50.0,
        "max_distance_m": 100.0,
        "max_tilt_deg": 20.0,
        "max_vertical_speed_ms": 2.0,
        "max_rotation_speed_degs": 20.0,
        "rth_min_altitude_m": 5.0,
        "stream_loss_grace_s": 10.0,
        "min_takeoff_battery_pct": 15.0,
        "distance_geofence": False,
        "require_gps_for_geofence": True,
    }
    values.update(overrides)
    return LiveSafetyConfig.resolve(**values)


def test_effective_safety_config_is_normalized_and_hash_stable() -> None:
    first = _resolve(nudge_pulse_s=99.0)
    second = _resolve(nudge_pulse_s=2.0)

    assert first.nudge_pulse_s == 2.0
    assert first == second
    assert first.checksum() == second.checksum()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tilt_deg", 20.1),
        ("max_vertical_speed_ms", 2.1),
        ("max_rotation_speed_degs", 20.1),
        ("stream_loss_grace_s", 10.1),
        ("min_takeoff_battery_pct", 9.9),
    ],
)
def test_unsafe_safety_override_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        _resolve(**{field: value})


@pytest.mark.parametrize("field", ["max_altitude_m", "max_distance_m"])
def test_limit_inputs_must_be_finite(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _resolve(**{field: float("nan")})
