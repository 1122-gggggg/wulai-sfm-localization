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
    gate.target_key = 0
    gate.aligned_for_translation = True
    return gate, cmd


@pytest.mark.parametrize("forward,right", [(0.317, 0.), (0., -.317), (.2, .2)])
def test_cruise_translates_toward_the_waypoint_without_yawing_for_drift(forward, right):
    gate, cmd = turn_fixture()
    pose = rpf.Pose(0, 0, 0, 0, stamp=1.)
    roll, pitch, yaw, gaz = gate.update(
        cmd, pose, 1., target_key=0, body_velocity=(forward, right, 1.)
    )
    assert yaw == 0
    assert roll or pitch
    assert max(abs(roll), abs(pitch)) <= gate.config.max_translation_pcmd
    assert gate.phase == "translate"


def test_cruise_keeps_translating_after_a_gust_without_turning():
    gate, cmd = turn_fixture()
    first = gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=1.), 1., target_key=0,
                        body_velocity=(.3, 0., 1.))
    second = gate.update(cmd, rpf.Pose(.08, 0, 0, 0, stamp=1.1), 1.1,
                         target_key=0, body_velocity=(0., 0., 1.1))
    assert first[2] == second[2] == 0
    assert gate.phase == "translate"
    assert first[:2] != (0, 0) or second[:2] != (0, 0)


@pytest.mark.parametrize("velocity,confirmed", [((.3, 0., 0.), True),
                                               ((float("nan"), 0., 1.), True),
                                               ((.3, 0., 1.), False)])
def test_cruise_still_translates_when_speed_or_map_confirmation_is_bad(velocity, confirmed):
    gate, cmd = turn_fixture()
    result = gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=1., map_confirmed=confirmed),
                         1., target_key=0, body_velocity=velocity)
    assert result[2] == 0
    assert result[:2] != (0, 0)
    assert gate.phase == "translate"


def test_near_waypoint_integral_grows_against_a_steady_offset():
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    cfg = rpf.apply_desktop_auto_authority(cfg)
    gate = rpf.YawAlignedPcmdController(cfg)
    gate.target_key = 0
    gate.aligned_for_translation = True
    cmd = rpf.Command("FOLLOW", np.zeros(3), 0.0, np.array([0.08, 0.0, 0.0]), 0.0, 0.0)
    pose = rpf.Pose(0, 0, 0, 0, stamp=1.0)
    first = gate.update(cmd, pose, 1.0, target_key=0, body_velocity=(0.0, 0.0, 1.0))
    later = first
    for index in range(1, 12):
        now = 1.0 + 0.05 * index
        later = gate.update(
            cmd, rpf.Pose(0, 0, 0, 0, stamp=now), now, target_key=0,
            body_velocity=(0.0, 0.0, now),
        )
    assert later[2] == 0
    assert abs(later[0]) + abs(later[1]) >= abs(first[0]) + abs(first[1])


def test_cruise_cancels_cross_track_wind_without_stopping():
    gate, cmd = turn_fixture()
    calm = gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=1.), 1., target_key=0,
                       body_velocity=(0., 0., 1.))
    blown = gate.update(cmd, rpf.Pose(0, 0, 0, 0, stamp=1.1), 1.1, target_key=0,
                        body_velocity=(0., 0.3, 1.1))
    assert calm[2] == blown[2] == 0
    assert blown[0] * 0.3 < 0 or blown[0] != calm[0]
    assert blown[1] or blown[0]


def test_height_adjust_holds_against_side_wind_instead_of_zeroing_horizontal():
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    gate = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command("FOLLOW", np.zeros(3), 0.0, np.array([0.0, -1.0, 0.0]), 0.0, 0.0)
    pose = rpf.Pose(0, 0, 0, 0, stamp=1.)
    roll, pitch, yaw, gaz = gate.update(
        cmd, pose, 1., target_key=0, body_velocity=(0.0, 0.3, 1.0),
    )
    assert yaw == 0
    assert gaz != 0 or gate.phase == "height_adjust"
    if math.hypot(roll, pitch) > 0:
        assert roll * 0.3 < 0 or pitch != 0


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
    # A partial arrival must survive localization recovery so a brief outage
    # cannot throw away a near-waypoint count and let wind push the aircraft out.
    ctrl._arrive_frames = 1
    ctrl._final_confirm_frames = 4
    for index in range(41):
        clock[0] = 1. + index * .05
        runner._tick(clock[0], "AUTO", {})
    assert ctrl.target_index == 0
    assert ctrl._arrive_frames == 1
    assert ctrl._final_confirm_frames == 4
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
def test_cruise_keeps_translating_under_continuous_drift(wind_speed):
    gate, _ = turn_fixture()
    cmd = rpf.Command("FOLLOW", np.zeros(3), math.pi / 2,
                      np.array([0., 0., 10.]), 0., 0.)
    pose = rpf.Pose(0., 0., 0., 0., stamp=0.)
    roll, pitch, turn, gaz = gate.update(
        cmd, pose, 0., target_key=0,
        body_velocity=(wind_speed, 0., 0.),
    )
    assert turn == 0
    assert roll or pitch
    assert gate.phase == "translate"
    assert gate.aligned_for_translation


def test_mid_leg_heading_error_does_not_stop_translation_to_turn():
    gate, cmd = turn_fixture()
    position = np.array([.5, 0., 0.])
    for now in (1., 1.1, 1.2):
        gate.update(cmd, rpf.Pose(*position, 0., stamp=now), now,
                    target_key=0, body_velocity=(.317, 0., now))
    assert gate.phase == "translate"
    assert gate.aligned_for_translation


def test_new_waypoint_yaws_toward_the_next_point_then_translates():
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    gate = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command(
        "FOLLOW", np.zeros(3), math.pi / 2, np.array([0.0, 0.0, 1.0]), 0.0, 0.0,
    )
    turning = gate.update(
        cmd, rpf.Pose(0, 0, 0, 0, stamp=1.0), 1.0, target_key=0,
        body_velocity=(0.0, 0.3, 1.0),
    )
    assert turning[2] != 0
    assert turning[0] * 0.3 <= 0
    assert gate.phase == "turn"
    assert not gate.aligned_for_translation

    now = 1.0
    for index in range(1, 6):
        now = 1.0 + 0.1 * index
        pcmd = gate.update(
            cmd, rpf.Pose(0, 0, 0, math.pi / 2, stamp=now), now, target_key=0,
        )
        if gate.aligned_for_translation:
            assert pcmd[2] == 0
            break
    else:
        raise AssertionError("aligned heading never released translation")
    flown = gate.update(
        cmd, rpf.Pose(0, 0, 0, math.pi / 2, stamp=now + 0.1), now + 0.1, target_key=0,
    )
    assert flown[2] == 0
    assert flown[0] or flown[1] or flown[3]
    assert gate.phase == "translate"

    next_cmd = rpf.Command(
        "FOLLOW", np.zeros(3), 0.0, np.array([1.0, 0.0, 0.0]), 0.0, 0.0,
    )
    next_leg = gate.update(
        next_cmd, rpf.Pose(0, 0, 0, math.pi / 2, stamp=2.0), 2.0, target_key=1,
        body_velocity=(0.0, 0.0, 2.0),
    )
    assert next_leg[2] != 0
    assert gate.phase == "turn"
    assert not gate.aligned_for_translation


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


def test_flight_loop_translates_instead_of_authorizing_drift_recovery():
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
    assert records[-1]["pcmd_phase"] != "turn_drift_recovery"
    assert sent[-1] is not None
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
