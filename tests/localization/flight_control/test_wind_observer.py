import math

import numpy as np
import pytest

import path_follow_flight as pff
import real_path_follow_controller as rpf
from wind_observer import ResidualDriftObserver


def test_constant_residual_converges_and_rotates_to_body():
    observer = ResidualDriftObserver()
    assert observer.feedforward(0, 0, 50) == (None, "no_velocity")
    for index in range(51):
        observer.observe_command([0, 0, 0, 0], 0, index * 0.2)
        observer.observe_velocities([(index * 0.2, 0.2, -0.1)])
        if index == 5:
            assert observer.feedforward(1.0, 0, 50)[1] == "warming_up"
    assert np.allclose(observer.drift, (0.2, -0.1), atol=0.01)
    assert observer.feedforward(10, 0, 50)[0] == pytest.approx((-5, 2.5), abs=0.2)
    assert observer.feedforward(10, math.pi / 2, 50)[0] == pytest.approx((2.5, 5), abs=0.2)
    assert observer.feedforward(10.6, 0, 50)[1] == "velocity_stale"
    assert observer.feedforward(10, None, 50)[1] == "yaw_unavailable"


def test_delayed_command_model_does_not_mislabel_maneuver():
    observer = ResidualDriftObserver()
    model = 0.0
    for index in range(101):
        stamp = index * 0.1
        if index:
            target = 0.8 if stamp > 0.5 else 0.0
            model += (target - model) * (1 - math.exp(-0.1 / 1.6))
        observer.observe_velocities([(stamp, model, 0.0)])
        observer.observe_command([0, 20, 0, 0], 0, stamp)
    assert np.linalg.norm(observer.drift) < 0.02
    assert observer.feedforward(10, 0, 50)[1] == "ready"


def test_cap_divergence_gap_and_reset():
    observer = ResidualDriftObserver()
    for index in range(101):
        observer.observe_velocities([(index * 0.1, 0.9, 0)])
    feedforward, reason = observer.feedforward(10, 0, 50)
    assert reason == "ready"
    assert math.hypot(*feedforward) == pytest.approx(15)
    observer.observe_velocities([(10.7, 0.9, 0)])
    assert observer.feedforward(10.7, 0, 50)[1] == "warming_up"
    stamp = 11.0
    while observer.last_reset != "diverged" and stamp < 30:
        observer.observe_velocities([(stamp, 1.5, 0)])
        stamp += 0.1
    assert observer.last_reset == "diverged"
    assert observer.feedforward(stamp, 0, 50)[1] == "no_velocity"
    observer.observe_velocities([(stamp, 1.5, 0)])
    assert observer.feedforward(stamp, 0, 50)[1] == "warming_up"
    observer.observe_command([0, 0, 0, 0], 0, stamp)
    observer.reset()
    assert not observer.commands and observer.last_reset == "diverged"


def _runner():
    clock = [0.0]
    cfg = rpf.ControlConfig(inspect_waypoints=(), waypoint_arrive_radius=0.02)
    ctrl = rpf.RouteAutoController([np.zeros(3), np.array([1.0, 0.0, 0.0])], config=cfg)
    hooks = pff.LoopHooks(
        get_pose=lambda: None, olympe_yaw=lambda: 0.0, send_pcmd=lambda *_: None,
        attitude_sample=lambda: (0.0, clock[0]), body_velocity=lambda: (0.0, 0.0, clock[0]),
        now=lambda: clock[0],
    )
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False,
                                   enforce_weak_pose_gate=False)
    return runner, clock


def test_runner_records_commands_and_reset_clears_observer():
    runner, clock = _runner()
    runner._clock_now = clock[0] = 1.0
    assert runner._send_command((1, 20, 0, 0))[0]
    assert list(runner.wind_observer.commands) == [(1.0, 20.0, 1.0)]
    runner._reset_pcmd_controller()
    assert not runner.wind_observer.commands


def test_shadow_mode_logs_without_changing_pcmd():
    outputs = []
    for warm in (False, True):
        runner, clock = _runner()
        if warm:
            runner.wind_observer.observe_velocities([(i * .2, .2, 0.) for i in range(21)])
        clock[0] = 4.0
        record = {}
        command = rpf.Command("FOLLOW", np.zeros(3), 0., np.zeros(3), 0., 0.)
        outputs.append(runner._compute_route_pcmd(command, rpf.Pose(-.1, 0., 0., 0., stamp=4.), 4., record))
        logged = record["wind_observer"]
        assert logged["reason"] == ("ready" if warm else "no_velocity")
        assert (logged["shadow_feedforward_body_pcmd"] is not None) is warm
    assert outputs[0] == outputs[1]
