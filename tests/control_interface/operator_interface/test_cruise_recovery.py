"""Offline regression cases for drift, localization recovery, and stalled legs."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import operator_autonomy as oa
import path_follow_flight as pff
import real_path_follow_controller as rpf


def turn_fixture():
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    gate = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command("FOLLOW", np.zeros(3), math.pi / 2,
                      np.array([0., -.3, 1.]), 0., 0.)
    return gate, cmd


@pytest.mark.parametrize("forward,right", [(0.317, 0.), (0., -.317), (.2, .2)])
def test_turn_recovery_brakes_opposite_velocity_without_yaw_or_gaz(forward, right):
    gate, cmd = turn_fixture()
    pose = rpf.Pose(0, 0, 0, 0, stamp=1.)
    roll, pitch, yaw, gaz = gate.update(
        cmd, pose, 1., target_key=0, body_velocity=(forward, right, 1.)
    )
    assert pitch * forward + roll * right < 0
    assert yaw == gaz == 0
    assert max(abs(roll), abs(pitch)) <= gate.config.max_translation_pcmd
    assert gate.phase == "turn_drift_recovery"


def test_turn_recovery_restores_anchor_and_requires_fresh_stable_samples():
    gate, cmd = turn_fixture()
    gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=1.), 1., target_key=0,
                body_velocity=(.3, 0., 1.))
    result = gate.update(cmd, rpf.Pose(.08, 0, 0, 0, stamp=1.1), 1.1,
                         target_key=0, body_velocity=(0., 0., 1.1))
    assert result[1] < 0 and result[2] == result[3] == 0
    # Even if position is back, one repeated speed sample cannot finish settling.
    for index in range(5):
        now = 1.2 + .1 * index
        result = gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=now), now,
                             target_key=0, body_velocity=(0., 0., 1.2))
        assert result[2] == 0
    for index in range(7):
        now = 1.7 + .1 * index
        result = gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=now), now,
                             target_key=0, body_velocity=(0., 0., now))
    assert result[2] != 0 and result[:2] == (0, 0)
    assert result[3] != 0  # Climb can run with the eventual turn.


@pytest.mark.parametrize("velocity,confirmed", [((.3, 0., 0.), True),
                                               ((float("nan"), 0., 1.), True),
                                               ((.3, 0., 1.), False)])
def test_turn_recovery_never_translates_on_stale_or_weak_inputs(velocity, confirmed):
    gate, cmd = turn_fixture()
    result = gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=1., map_confirmed=confirmed),
                         1., target_key=0, body_velocity=velocity)
    assert result[:2] == (0, 0)


def progress_coordinator():
    clock = [0.]
    obj = object.__new__(oa.DesktopRouteAutonomy)
    obj.phase = "ROUTE"
    obj._paused = SimpleNamespace(is_set=lambda: False)
    obj._waypoint_progress = None
    obj.now = lambda: clock[0]
    obj._latch_auto_failure = Mock()
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    obj.controller = rpf.RouteAutoController(
        [np.array([0., 0., 0.]), np.array([1., 0., 0.])], config=cfg
    )
    return obj, clock


def progress_tick(now, **overrides):
    return dict({"target_index": 1, "pose_u": [0., 0., 0.], "pose_stamp": now,
                 "pcmd_phase": "translate", "pcmd": [0, 3, 0, 0],
                 "blocked": False}, **overrides)


def test_turns_cannot_erase_waypoint_stall_history():
    obj, clock = progress_coordinator()
    for index in range(401):
        clock[0] = index / 10
        turn = index % 250 >= 240
        obj._check_waypoint_progress(progress_tick(
            clock[0], pcmd_phase="turn" if turn else "translate",
            pcmd=[0, 0, 50, 0] if turn else [0, 3, 0, 0], yaw_error_deg=90.,
        ))
        if obj._latch_auto_failure.called:
            break
    assert obj._latch_auto_failure.called
    assert clock[0] <= 31.


def test_localization_wait_pauses_but_does_not_erase_stall_history():
    obj, clock = progress_coordinator()
    for index in range(804):
        clock[0] = index / 10
        waiting = 20. <= clock[0] < 70.
        obj._check_waypoint_progress(progress_tick(
            clock[0], blocked=waiting, pose_stamp=0. if waiting else clock[0],
        ))
        if waiting:
            assert not obj._latch_auto_failure.called
        if obj._latch_auto_failure.called:
            break
    assert obj._latch_auto_failure.called
    assert 79. <= clock[0] <= 80.3


def test_genuine_heading_or_rejoin_progress_keeps_the_leg_alive():
    for phase, field, start, rate in [("turn", "yaw_error_deg", 170., 2.),
                                     ("route_rejoin", "path_error_u", 1., .005)]:
        obj, clock = progress_coordinator()
        for index in range(601):
            clock[0] = index / 10
            obj._check_waypoint_progress(progress_tick(
                clock[0], pcmd_phase=phase, **{field: start - rate * clock[0]}
            ))
        assert not obj._latch_auto_failure.called


def test_localization_recovery_retains_target_and_hovers_before_resuming():
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    ctrl = rpf.RouteAutoController([np.array([x, 0., 0.]) for x in (0., 1., 2.)], config=cfg)
    ctrl.start_after_nearest_waypoint(np.array([0.1, 0., 0.]))
    clock = [1.]
    source = [None]
    records = []
    requests = []
    hooks = pff.LoopHooks(
        get_pose=lambda: source[0], olympe_yaw=lambda: None, send_pcmd=lambda *args: None,
        pose_is_weak=lambda: False, pose_confidence=lambda: 100,
        force_relocalize=lambda: requests.append(clock[0]),
        max_weak_pose_age_s=.5, land_on_localization_loss=False,
        log_tick=lambda row: records.append(dict(row)), now=lambda: clock[0],
    )
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False,
                                   enforce_weak_pose_gate=True)
    # A partial arrival must not survive a localization outage.
    ctrl._arrive_frames = 1
    ctrl._final_confirm_frames = 4
    for index in range(41):
        clock[0] = 1. + index * .05
        runner._tick(clock[0], "AUTO", {})
    assert ctrl.target_index == 0
    assert ctrl._arrive_frames == ctrl._final_confirm_frames == 0
    assert 1 <= len(requests) <= 3
    # Resume near waypoint 2: never reselect that closer point after recovery.
    for now in (3.1, 3.2):
        clock[0] = now
        source[0] = rpf.Pose(0.9, 0, 0, 0, stamp=now)
        runner._tick(now, "AUTO", {})
        assert records[-1]["pcmd"] == [0, 0, 0, 0]
        assert ctrl.target_index == 0
    assert records[-1]["localization_recovery_state"] == "recovered"
    clock[0] = 3.3
    source[0] = rpf.Pose(0.9, 0, 0, 0, stamp=3.3)
    runner._tick(3.3, "AUTO", {})
    assert ctrl.target_index == 0
    assert records[-1]["target_index"] == 0


@pytest.mark.parametrize("yaw,north,east,expected", [
    (0., .3, .2, (.3, .2)), (math.pi / 2, .3, .2, (.2, -.3)),
])
def test_firmware_velocity_is_rotated_to_body_axes(yaw, north, east, expected):
    obj, clock = progress_coordinator()
    clock[0] = 1.
    obj.backend = SimpleNamespace(state=SimpleNamespace(
        att_yaw=yaw, attitude_mono_ns=1_000_000_000,
        ground_speed_mps=math.hypot(north, east), ground_speed_mono_ns=1_000_000_000,
        speed_north_mps=north, speed_east_mps=east,
    ))
    assert obj._body_velocity()[:2] == pytest.approx(expected)
    clock[0] = 2.
    assert obj._body_velocity() is None


@pytest.mark.parametrize("wind_speed", [.15, .317])
def test_turn_completes_with_continuous_drift_and_inertia(wind_speed):
    gate, _ = turn_fixture()
    cfg = gate.config
    cmd = rpf.Command("FOLLOW", np.zeros(3), math.pi / 2,
                      np.array([0., 0., 10.]), 0., 0.)
    position = np.zeros(3)
    velocity = np.zeros(3)
    yaw = 0.
    dt = .05
    phases = set()
    for index in range(1200):
        now = index * dt
        wind = cfg.map_frame.east * wind_speed
        forward = math.cos(yaw) * cfg.map_frame.east + math.sin(yaw) * cfg.map_frame.north
        right = math.sin(yaw) * cfg.map_frame.east - math.cos(yaw) * cfg.map_frame.north
        ground = velocity + wind
        pose = rpf.Pose(*(position / 5.), yaw, stamp=now)
        roll, pitch, turn, gaz = gate.update(
            cmd, pose, now, target_key=0,
            body_velocity=(float(ground @ forward), float(ground @ right), now),
        )
        assert not ((roll or pitch) and (turn or gaz))
        phases.add(gate.phase)
        if gate.aligned_for_translation:
            break
        # 0.4 s inertia, 0.15 m/s per PCMD, with no firmware position hold.
        velocity += (.15 * (pitch * forward + roll * right) - velocity) * (1 - math.exp(-dt / .4))
        position += (velocity + wind) * dt
        yaw -= math.radians(turn * .2) * dt
        assert np.linalg.norm(position) < .2
    assert gate.aligned_for_translation
    assert {"turn_drift_recovery", "turn_settle", "turn"} <= phases


def test_mid_leg_realign_anchors_at_current_position_not_the_old_turn():
    gate, cmd = turn_fixture()
    gate.target_key = 0
    gate.aligned_for_translation = True
    gate.turn_anchor = np.zeros(3)
    position = np.array([.5, 0., 0.])
    for now in (1., 1.1, 1.2):
        gate.update(cmd, rpf.Pose(*position, 0., stamp=now), now,
                    target_key=0, body_velocity=(.317, 0., now))
    assert gate.phase == "turn_drift_recovery"
    np.testing.assert_allclose(gate.turn_anchor, position)


@pytest.mark.parametrize("phase,text", [
    ("turn_drift_recovery", "修正風漂移"), ("turn_settle", "確認水平穩定"),
    ("turn_recovery_wait", "等待可靠定位與速度"),
])
def test_drift_stages_are_visible(phase, text):
    obj, _ = progress_coordinator()
    obj.waypoints = obj.controller.wp
    key, label = obj._leg_stage(1, {"phase": phase})
    assert key == phase and text in label


def test_lost_localization_invalidates_a_pending_final_landing_decision():
    obj, _ = progress_coordinator()
    ctrl = obj.controller
    ctrl.target_index = len(ctrl.wp) - 1
    ctrl.state = "LANDING"
    ctrl._final_hold_started = 1.
    ctrl._final_confirm_frames = 5
    ctrl.reset_arrival_confirmation()
    assert ctrl.state == "CRUISE"
    assert ctrl.target_index == len(ctrl.wp) - 1
    assert ctrl._final_hold_started is None and ctrl._final_confirm_frames == 0


def test_drift_that_hits_speed_guard_cannot_wait_indefinitely():
    obj, clock = progress_coordinator()
    for index in range(311):
        clock[0] = index / 10
        obj._check_waypoint_progress(progress_tick(
            clock[0], blocked=True, pcmd_phase="turn_drift_recovery",
            reason="AUTO speed limit reached -> HOVER", pcmd=[0, 0, 0, 0],
        ))
        if obj._latch_auto_failure.called:
            break
    assert obj._latch_auto_failure.called


def test_flight_loop_uses_body_velocity_before_authorizing_drift_recovery():
    obj, clock = progress_coordinator()
    clock[0] = 1.
    sent = []
    records = []
    pose = rpf.Pose(0., 0., -.2, 0., stamp=1.)
    hooks = pff.LoopHooks(
        get_pose=lambda: pose, olympe_yaw=lambda: None, send_pcmd=lambda *cmd: sent.append(cmd),
        body_velocity=lambda: (.317, 0., clock[0]),
        pose_is_weak=lambda: False, pose_confidence=lambda: 100,
        log_tick=lambda row: records.append(dict(row)), now=lambda: clock[0],
    )
    runner = pff._FlightLoopRunner(hooks, obj.controller, obj.controller.wp,
                                   yaw_sign=1, verbose=False, enforce_weak_pose_gate=True)
    runner._tick(1., "AUTO", {})
    assert records[-1]["pcmd_phase"] == "turn_drift_recovery"
    assert sent[-1][1] < 0 and sent[-1][2:] == (0, 0)
    hooks.safety_poll = lambda: "HOVER"
    runner._tick(1., "HOVER", {})
    assert not any(sent[-1])
    sent_count = len(sent)
    hooks.safety_poll = lambda: "MANUAL"
    runner._tick(1., "MANUAL", {})
    assert len(sent) == sent_count and records[-1]["pcmd"] is None


def test_spatial_progress_allows_a_later_genuine_heading_correction():
    obj, clock = progress_coordinator()
    for index in range(500):
        clock[0] = index / 10
        if index < 100:
            values = {"pcmd_phase": "turn", "yaw_error_deg": 90. - index * .9}
        elif index < 300:
            values = {"pose_u": [.2, 0., 0.]}
        else:
            values = {"pose_u": [.2, 0., 0.], "pcmd_phase": "turn",
                      "yaw_error_deg": 90. - (index - 300) * .2}
        obj._check_waypoint_progress(progress_tick(clock[0], **values))
    assert not obj._latch_auto_failure.called
