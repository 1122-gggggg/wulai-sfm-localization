"""Consumer-visible synthetic visual loss and independent fused-attitude contracts."""

import math

import numpy as np
import pytest

from simulated_localization import (
    CommandChannel,
    CommandChannelConfig,
    SimulatedLocalizer,
    SimulatedLocalizerConfig,
    SimulatedPose,
)


def _localizer(**overrides):
    config = dict(
        s=5.0,
        pos_sigma_u=0.0,
        yaw_sigma_deg=0.0,
        latency_ms=0.0,
        drop_rate=0.0,
        blackout_every_s=2.0,
        blackout_dur_s=1.0,
    )
    config.update(overrides)
    return SimulatedLocalizer(
        SimulatedLocalizerConfig(**config), np.random.default_rng(7), pose_cls=SimulatedPose
    )


def _visual_anchor(loc, imu_stamp=1.0, imu_yaw=0.0):
    loc.push_imu(imu_stamp, imu_yaw)
    loc.push_truth(1.0, [5.0, -10.0, 15.0], 0.4)
    anchor = loc.report(1.0)
    assert anchor is not None and anchor.map_confirmed and anchor.position_observed
    assert anchor.stamp == pytest.approx(1.0)
    return anchor


def test_blackout_uses_wrapped_attitude_increment_and_holds_visual_center(monkeypatch):
    loc = _localizer()
    anchor = _visual_anchor(loc, imu_yaw=math.radians(170.0))
    loc.push_truth(2.0, [900.0, 800.0, -700.0], -2.0)
    loc.push_imu(2.0, math.radians(-175.0), tilt_deg=12.0, yaw_rate_deg_s=15.0)
    # Blackout must not consult either true heading or true position.
    monkeypatch.setattr(loc, "_truth_at", lambda _stamp: pytest.fail("blackout queried truth"))
    pose = loc.report(2.0)
    assert pose is not None
    np.testing.assert_array_equal(pose.xyz, anchor.xyz)
    assert pose.yaw == pytest.approx(anchor.yaw + math.radians(15.0))
    assert not pose.map_confirmed and not pose.position_observed and not loc.map_confirmed
    assert loc.fault == "blackout-imu"
    assert pose.stamp == pytest.approx(2.0)


def test_blackout_requires_fresh_sample_and_does_not_refresh_held_stamp():
    loc = _localizer()
    _visual_anchor(loc)
    loc.push_imu(1.5, 0.2)
    assert loc.report(2.0).stamp == pytest.approx(1.5)  # Exactly 0.5 s is usable.
    assert loc.report(2.001) is None
    assert loc.fault == "blackout-imu-stale"
    assert not loc.map_confirmed
    loc.push_imu(2.05, 0.3)
    assert loc.report(2.05).yaw == pytest.approx(0.7)


@pytest.mark.parametrize(
    "sample, fault",
    [
        ((2.0, float("nan"), 0.0, 0.0), "blackout-imu-invalid"),
        ((float("inf"), 0.1, 0.0, 0.0), "blackout-imu-invalid"),
        ((2.0, 0.1, float("nan"), 0.0), "blackout-imu-invalid"),
        ((2.0, 0.1, 0.0, float("inf")), "blackout-imu-invalid"),
        ((2.0, 0.1, 35.01, 0.0), "blackout-imu-tilt"),
        ((2.0, 0.1, 0.0, -180.01), "blackout-imu-yaw-rate"),
        ((2.1, 0.1, 0.0, 0.0), "blackout-imu-future"),
        ((0.9, 0.1, 0.0, 0.0), "blackout-imu-out-of-order"),
    ],
)
def test_blackout_invalid_attitude_has_no_pose(sample, fault):
    loc = _localizer()
    _visual_anchor(loc)
    stamp, yaw, tilt, rate = sample
    loc.push_imu(stamp, yaw, tilt_deg=tilt, yaw_rate_deg_s=rate)
    assert loc.report(2.0) is None
    assert loc.fault == fault
    assert not loc.map_confirmed


def test_blackout_safe_tilt_and_rate_limits_are_inclusive():
    loc = _localizer()
    _visual_anchor(loc)
    loc.push_imu(2.0, 0.1, tilt_deg=35.0, yaw_rate_deg_s=-180.0)
    assert loc.report(2.0) is not None


def test_blackout_needs_attitude_reference_at_last_visual_capture():
    loc = _localizer()
    _visual_anchor(loc, imu_stamp=0.0)
    loc.push_imu(2.0, 0.2)
    assert loc.report(2.0) is None
    assert loc.fault == "blackout-imu-no-reference"


def test_delayed_visual_delivery_keeps_capture_aligned_attitude_reference():
    loc = _localizer(latency_ms=100.0)
    loc.push_imu(1.0, 0.0)
    loc.push_truth(1.0, [5.0, 0.0, 0.0], 0.4)
    assert loc.report(1.0) is None
    loc.push_imu(1.1, 0.1)
    anchor = loc.report(1.1)
    assert anchor is not None and anchor.stamp == pytest.approx(1.0)
    loc.push_imu(2.0, 0.2)
    pose = loc.report(2.0)
    assert pose.yaw == pytest.approx(0.6)  # Reference is 1.0, not delivery-time 1.1.
    np.testing.assert_array_equal(pose.xyz, anchor.xyz)


@pytest.mark.parametrize("limits", [{"imu_bridge_max_age_s": 1.0}, {"imu_bridge_max_frames": 1}])
def test_blackout_expires_then_visual_recovery_reanchors(limits):
    loc = _localizer(reseed_confirm_frames=2, **limits)
    _visual_anchor(loc)
    loc.push_imu(2.0, 0.1)
    assert loc.report(2.0) is not None
    loc.push_imu(2.05, 0.15)
    assert loc.report(2.05) is None
    assert loc.fault == "blackout-expired"
    assert not loc.map_confirmed
    loc.push_imu(3.05, 0.2)
    loc.push_truth(3.05, [10.0, 5.0, -15.0], 0.7)
    recovered = loc.report(3.05)
    assert recovered is not None and recovered.map_confirmed and recovered.position_observed
    assert recovered.reseed_confirming
    np.testing.assert_allclose(recovered.xyz, [2.0, 1.0, -3.0])
    loc.push_imu(4.0, 0.3)
    bridged = loc.report(4.0)
    assert bridged is not None and not bridged.map_confirmed and not bridged.position_observed
    np.testing.assert_array_equal(bridged.xyz, recovered.xyz)
    assert bridged.yaw == pytest.approx(recovered.yaw + 0.1)


def test_dropout_has_no_new_measurement_or_confirmation_until_visual_recovery():
    loc = _localizer(blackout_every_s=0.0)
    _visual_anchor(loc)
    loc.drop_rate = 1.0
    assert loc.report(1.05) is None
    assert loc.fault == "dropout" and not loc.map_confirmed
    assert loc.report(1.06) is None  # No capture due: do not resurrect the old map fix.
    loc.drop_rate = 0.0
    loc.push_truth(1.1, [10.0, 0.0, 0.0], 0.1)
    recovered = loc.report(1.1)
    assert recovered is not None and recovered.map_confirmed and recovered.position_observed


def test_outage_and_blackout_without_anchor_have_no_pose_or_confirmation():
    outage = _localizer(outage_every_s=2.0, outage_dur_s=0.5)
    _visual_anchor(outage)
    assert outage.report(2.0) is None
    assert outage.fault == "outage" and not outage.map_confirmed
    no_anchor = _localizer()
    no_anchor.push_truth(2.0, [0.0, 0.0, 0.0], 0.0)
    no_anchor.push_imu(2.0, 0.0)
    assert no_anchor.report(2.0) is None
    assert no_anchor.fault == "blackout-no-anchor" and not no_anchor.map_confirmed


def test_command_channel_half_second_delay_does_not_starve_at_twenty_hz():
    channel = CommandChannel(CommandChannelConfig(latency_ms=500.0, ttl_s=0.2))
    for tick in range(41):
        now = tick / 20.0
        sent = channel.send((tick, 0, 0, 0), now)
        assert sent == tick + 1
        pcmd, age, seq = channel.applied(now)
        if tick < 10:
            assert seq is None and pcmd == (0, 0, 0, 0)
        else:
            assert seq == tick - 9
            assert pcmd == (tick - 10, 0, 0, 0)
            assert age == pytest.approx(0.0, abs=1e-12)
        assert len(channel._queue) <= 12
    assert channel.applied(3.0)[2] is None
    assert not channel._queue


def test_bounded_command_overload_preserves_earliest_arrival_not_starvation():
    channel = CommandChannel(CommandChannelConfig(latency_ms=500.0, max_pending=2))
    assert channel.send((1, 0, 0, 0), 0.0) == 1
    assert channel.send((2, 0, 0, 0), 0.01) == 2
    assert channel.send((3, 0, 0, 0), 0.02) is None
    assert len(channel._queue) == 2
    assert channel.applied(0.5)[0] == (1, 0, 0, 0)
    assert channel.applied(0.51)[0] == (2, 0, 0, 0)


@pytest.mark.parametrize("fault", ["blackout", "outage"])
def test_fault_windows_flush_pending_visual_packets_and_restart_capture_cadence(fault):
    options = dict(latency_ms=200.0)
    if fault == "outage":
        options.update(blackout_every_s=0.0, outage_every_s=2.0, outage_dur_s=1.0)
    loc = _localizer(**options)
    loc.push_truth(1.0, [5.0, 0.0, 0.0], 0.0)
    loc.push_imu(1.0, 0.0)
    assert loc.report(1.0) is None
    loc.push_imu(1.2, 0.0)
    assert loc.report(1.2).stamp == pytest.approx(1.0)
    loc.push_truth(1.9, [10.0, 0.0, 0.0], 0.0)
    loc.push_imu(1.9, 0.0)
    loc.report(1.9)
    assert loc._pending
    loc.push_imu(2.0, 0.1)
    loc.report(2.0)
    assert not loc._pending
    loc.push_truth(3.0, [15.0, 0.0, 0.0], 0.2)
    loc.push_imu(3.0, 0.2)
    assert loc.report(3.0) is None
    assert not loc.map_confirmed
    loc.push_imu(3.2, 0.2)
    recovered = loc.report(3.2)
    assert recovered is not None and recovered.map_confirmed and recovered.position_observed
    assert recovered.stamp == pytest.approx(3.0)
    np.testing.assert_allclose(recovered.xyz, [3.0, 0.0, 0.0])


def test_current_capture_dropout_still_delivers_independent_matured_visual_packet():
    loc = _localizer(blackout_every_s=0.0, latency_ms=100.0)
    loc.push_truth(1.0, [5.0, 0.0, 0.0], 0.0)
    loc.push_imu(1.0, 0.0)
    assert loc.report(1.0) is None
    loc.drop_rate = 1.0
    delivered = loc.report(1.1)
    assert delivered is not None and delivered.map_confirmed and delivered.position_observed
    assert delivered.stamp == pytest.approx(1.0)
    assert loc.report(1.15) is None
    assert not loc.map_confirmed


@pytest.mark.parametrize("stamp", [None, "invalid", float("nan"), float("inf")])
def test_invalid_report_time_produces_no_pose_or_map_confirmation(stamp):
    loc = _localizer()
    _visual_anchor(loc)
    assert loc.report(stamp) is None
    assert loc.fault == "invalid-time" and not loc.map_confirmed
