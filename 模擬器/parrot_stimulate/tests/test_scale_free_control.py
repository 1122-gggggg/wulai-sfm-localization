import math

import pytest

from anafi_pcmd_sim.scale_free_control import (
    ScaleFreeConfig,
    ScaleFreeSample,
    command_is_fresh,
    decide_scale_free,
    validate_speed_limit_change,
)

NOW_NS = 10_000_000_000


def sample(**overrides) -> ScaleFreeSample:
    values = {
        "position_map": (0.0, 0.0, 0.0),
        "target_map": (10.0, 0.0, 0.0),
        "camera_yaw_rad": 0.0,
        "pose_mono_ns": NOW_NS - 100_000_000,
        "localization_state": "TRACK",
        "airframe_horizontal_speed_mps": 0.1,
        "speed_mono_ns": NOW_NS - 50_000_000,
        "route_deviation_ok": True,
        "target_reached": False,
        "approval_locked": False,
    }
    values.update(overrides)
    return ScaleFreeSample(**values)


def test_map_distance_changes_no_direction_or_physical_speed_limit() -> None:
    near = decide_scale_free(sample(target_map=(1.0, 0.0, 0.0)), now_mono_ns=NOW_NS)
    far = decide_scale_free(sample(target_map=(1000.0, 0.0, 0.0)), now_mono_ns=NOW_NS)

    assert near.direction_map == pytest.approx((1.0, 0.0, 0.0))
    assert far.direction_map == pytest.approx(near.direction_map)
    assert near.speed_limit_mps == far.speed_limit_mps == pytest.approx(0.30)
    assert not hasattr(ScaleFreeConfig(), "map_units_per_meter")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"localization_state": "WEAK"}, "LOCALIZATION_NOT_TRACK"),
        ({"pose_mono_ns": NOW_NS - 600_000_000}, "POSE_STALE"),
        ({"airframe_horizontal_speed_mps": None}, "SPEED_UNAVAILABLE"),
        ({"speed_mono_ns": NOW_NS - 600_000_000}, "SPEED_STALE"),
        ({"airframe_horizontal_speed_mps": 0.31}, "OVERSPEED"),
        ({"route_deviation_ok": False}, "ROUTE_DEVIATION"),
    ],
)
def test_any_safety_guard_forces_zero_and_manual_handoff(overrides, reason) -> None:
    decision = decide_scale_free(sample(**overrides), now_mono_ns=NOW_NS)

    assert decision.phase == "HOLD"
    assert decision.reason == reason
    assert decision.allow_translation is False
    assert decision.zero_motion is True
    assert decision.manual_handoff is True


def test_controller_turns_before_allowing_translation() -> None:
    turn = decide_scale_free(sample(target_map=(0.0, 1.0, 0.0)), now_mono_ns=NOW_NS)
    translate = decide_scale_free(sample(), now_mono_ns=NOW_NS)

    assert turn.phase == "TURN"
    assert math.degrees(turn.yaw_error_rad) == pytest.approx(90.0)
    assert turn.allow_translation is False
    assert turn.manual_handoff is False
    assert translate.phase == "TRANSLATE"
    assert translate.allow_translation is True
    assert translate.body_forward == pytest.approx(1.0)


def test_real_approval_lock_and_command_ttl_are_fail_closed() -> None:
    locked = decide_scale_free(sample(approval_locked=True), now_mono_ns=NOW_NS)
    active = decide_scale_free(sample(), now_mono_ns=NOW_NS)

    assert locked.reason == "LOCKED_EXTERNAL_APPROVAL"
    assert locked.zero_motion and locked.manual_handoff
    assert command_is_fresh(active.command_expires_mono_ns, NOW_NS + 100_000_000)
    assert not command_is_fresh(active.command_expires_mono_ns, NOW_NS + 200_000_000)


def test_speed_limit_change_requires_landed_and_invalidates_approval() -> None:
    rejected = validate_speed_limit_change(0.30, 0.20, landed=False)
    accepted = validate_speed_limit_change(0.30, 0.20, landed=True)

    assert not rejected.accepted
    assert rejected.reason == "LANDED_REQUIRED"
    assert accepted.accepted
    assert accepted.approval_invalidated
    assert accepted.new_speed_limit_mps == pytest.approx(0.20)
