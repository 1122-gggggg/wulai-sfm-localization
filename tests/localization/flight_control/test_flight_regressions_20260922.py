"""Offline regressions for the 2026-09-22 flight failure mechanisms."""

from dataclasses import replace
import math

import numpy as np
import pytest

from control_state_estimator import ControlStateEstimator
import path_follow_flight as pff
import real_path_follow_controller as rpf


def _config():
    return replace(rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
                   waypoint_arrive_radius=.0302155, slowdown_distance=.4322555)


def _segment(action="REJOIN"):
    return rpf.Command(action, np.array([1., 0., 0.]), 0., np.array([1., 0., 0.]),
                       .06, 0., guidance_goal=np.array([.3, 0., 0.]),
                       path_tangent=np.array([1., 0., 0.]),
                       path_pull=np.array([0., 0., -.06]), path_pull_radius=.03)


def test_rejoin_cannot_interrupt_the_initial_reversal():
    gate = rpf.YawAlignedPcmdController(_config())
    for i in range(10):
        now = 1. + i * .05
        pose = rpf.Pose(.1, 0., .06, math.radians(165), stamp=now)
        pcmd = gate.update(_segment(), pose, now, target_key=1,
                           body_velocity=(0., 0., now))
        assert gate.phase == "turn"
        assert pcmd[2] != 0 and pcmd[3] == 0
        assert pcmd[:2] == (0, 0), "stationary turn must not advance backward along the leg"


def test_near_point_rejoin_does_not_discard_longitudinal_wind_compensation():
    gate = rpf.YawAlignedPcmdController(_config())
    gate.target_key, gate.aligned_for_translation = 1, True
    cmd = replace(_segment(), goal=np.zeros(3), guidance_goal=np.array([-.08, 0., 0.]))
    for i in range(50):
        now = 1. + i * .05
        gate.update(cmd, rpf.Pose(-.08, 0., .02, 0., stamp=now), now,
                    target_key=1, body_velocity=(0., 0., now))
    assert gate.phase == "waypoint_centering"
    assert gate._integral_map[0] > .1


def _contradict_height(guard):
    for height in [1.] * 5 + [2.1] * 5:
        guard.observe(0, .2, height, .03)
    assert 0 in guard.waived


def test_unobservable_height_does_not_restart_climb_at_the_next_waypoint():
    guard = rpf.VerticalConsistencyGuard(rpf.LEGACY_MAP_FRAME)
    _contradict_height(guard)
    assert guard.observe(1, .35, 2.1, .03)
    assert 1 in guard.waived


def test_unobservable_height_cannot_manufacture_final_landing_position():
    cfg = _config()
    ctrl = rpf.RouteAutoController([np.array([0., -.2, 0.]), np.array([1., -.2, 0.])], config=cfg)
    hooks = pff.LoopHooks(get_pose=lambda: None, olympe_yaw=lambda: math.pi / 2,
                          altitude=lambda: (2.1, 1.), send_pcmd=lambda *_: None)
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1,
                                   verbose=False, enforce_weak_pose_gate=False)
    _contradict_height(runner.vertical_guard)
    final = len(ctrl.wp) - 1
    runner.vertical_guard.waived.add(final)
    ctrl.target_index = final
    pose = rpf.Pose(ctrl.wp[final][0], 0., 0., 0., stamp=1.)
    guarded = runner._vertical_guarded_pose(pose, 1., {})
    assert not ctrl.waypoint_reached(guarded.xyz, final)


def test_height_hold_covers_the_tick_that_advances_to_the_next_target():
    cfg = replace(_config(), return_to_start=False, waypoint_arrive_confirm_frames=1)
    ctrl = rpf.RouteAutoController([np.array([0., -.2, 0.]), np.array([.02, -.2, 0.])], config=cfg)
    sent = []
    hooks = pff.LoopHooks(get_pose=lambda: None, olympe_yaw=lambda: math.pi / 2,
                          altitude=lambda: (2.1, 1.), ground_speed=lambda: (0., 1.),
                          body_velocity=lambda: (0., 0., 1.),
                          send_pcmd=lambda *pcmd: sent.append(pcmd))
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1,
                                   verbose=False, enforce_weak_pose_gate=False)
    _contradict_height(runner.vertical_guard)
    runner.heading.update(0., math.pi / 2)
    record = {}
    runner._localized_motion_outcome(rpf.Pose(0., 0., 0., 0., stamp=1.), math.pi / 2, 1., record)
    assert ctrl.target_index == 1
    assert sent[-1][3] == 0
    assert runner.landing_confirmation.samples == 0


def test_scale_uses_independent_metres_of_motion_without_waiting_four_seconds_per_pair():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for i in range(441):
        now = 1. + i * .05
        estimator.observe_telemetry((math.pi / 2, now), (0., .6, now), now)
        estimator.observe_pose(rpf.Pose(.6 * now / 15., 0., 0., 0., stamp=now), now, 0.)
    # 13.2 m of consistent motion supplies eight independent >=1 m pairs.
    assert len(estimator.scale_samples) >= 8
    assert sum(metres for _, metres in estimator.scale_samples) >= 10.
    assert estimator.scale_m_per_unit == pytest.approx(15.)


def test_heading_waits_for_late_attitude_and_boot_weak_heading_does_not_freeze():
    clock = [1.]
    ctrl = rpf.RouteAutoController([np.zeros(3), np.array([10., 0., 0.])], config=_config())
    hooks = pff.LoopHooks(
        get_pose=lambda: rpf.Pose(.5, 0., 0., .3 * (clock[0] - .08),
                                  stamp=clock[0] - .08, map_confirmed=False),
        olympe_yaw=lambda: math.pi / 2 - .3 * (clock[0] - .2),
        attitude_sample=lambda: (math.pi / 2 - .3 * (clock[0] - .2), clock[0] - .2),
        send_pcmd=lambda *_: None, now=lambda: clock[0],
    )
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1,
                                   verbose=False, enforce_weak_pose_gate=False)
    for i in range(40):
        clock[0] = 1. + i * .05
        runner._tick(clock[0], "AUTO", {})
    assert runner.heading.offset == pytest.approx(0., abs=1e-8)
    assert runner.heading.heading(hooks.olympe_yaw()) == pytest.approx(.3 * (clock[0] - .2))
