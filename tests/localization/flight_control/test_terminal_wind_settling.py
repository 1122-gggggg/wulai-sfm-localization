"""Terminal control against a synthetic sustained drift, without hardware."""

from dataclasses import replace
import math

import numpy as np
import pytest

import path_follow_flight as pff
import real_path_follow_controller as rpf


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
@pytest.mark.parametrize("drift_mps", [-0.3, 0.0, 0.3])
def test_terminal_confirmation_keeps_wind_compensation_until_settled(yaw, drift_mps):
    cfg = replace(
        rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
        return_to_start=False,
        slowdown_distance=0.4322554798454335,
        waypoint_arrive_radius=0.051909,
        inspect_waypoints=(),
    )
    waypoints = [np.array([-1.0, 0.0, 0.0]), np.zeros(3)]
    ctrl = rpf.RouteAutoController(waypoints, poles=[], config=cfg)
    ctrl.target_index = 1
    hooks = pff.LoopHooks(get_pose=lambda: None, olympe_yaw=lambda: None,
                          send_pcmd=lambda *_args: None)
    runner = pff._FlightLoopRunner(hooks, ctrl, waypoints, yaw_sign=1,
                                   verbose=False, enforce_weak_pose_gate=True)
    x, velocity = -0.04, 0.0
    saw_settling = False
    for tick in range(1200):
        now = 1.0 + tick * 0.05
        ground_speed = velocity + drift_mps
        pose = rpf.Pose(x, 0.0, 0.0, yaw, stamp=now)
        command = ctrl.step(pose, now)
        record = {}
        settled = runner._apply_landing_settling(
            command, pose, (abs(ground_speed), now), now, record,
        )
        if command.action == "LAND" and not settled.should_land:
            saw_settling = True
        pcmd = runner.pcmd_controller.update(
            settled, pose, now, target_key=ctrl.target_index,
            body_velocity=(ground_speed * math.cos(yaw), ground_speed * math.sin(yaw), now),
        )
        assert pcmd[2:] == (0, 0)
        assert max(abs(pcmd[0]), abs(pcmd[1])) <= cfg.max_translation_pcmd
        if settled.should_land:
            assert abs(x) <= ctrl.final_arrive_tolerance()
            assert abs(ground_speed) <= 0.10
            assert record["landing_stable_samples"] >= 3
            assert record["landing_stable_span_s"] >= 1.0
            break
        east_pcmd = pcmd[1] * math.cos(yaw) + pcmd[0] * math.sin(yaw)
        velocity += (east_pcmd * 0.018 - velocity) * (1.0 - math.exp(-0.05 / 0.4))
        x += (velocity + drift_mps) * 0.05 / 15.0
    else:
        pytest.fail("terminal position/speed did not settle within 60 simulated seconds")
    if drift_mps:
        assert saw_settling


def test_terminal_settling_uses_wait_budget_and_stops_motion_on_expiry():
    cfg = rpf.ControlConfig(inspect_waypoints=())
    waypoints = [np.zeros(3), np.ones(3)]
    ctrl = rpf.RouteAutoController(waypoints, poles=[], config=cfg)
    sent, waits = [], []

    def expired(reason, elapsed):
        waits.append((reason, elapsed))
        return elapsed >= 0.2

    hooks = pff.LoopHooks(
        get_pose=lambda: None, olympe_yaw=lambda: None,
        send_pcmd=lambda *command: sent.append(command), wait_expired=expired,
    )
    runner = pff._FlightLoopRunner(hooks, ctrl, waypoints, yaw_sign=1,
                                   verbose=False, enforce_weak_pose_gate=True)
    command = rpf.Command("FINAL_HOLD", np.zeros(3), 0.0, waypoints[-1], 0.0, 1.0)
    assert runner._route_completion_outcome(command, None, 10.0, {}) is None
    assert runner._route_completion_outcome(command, None, 10.1, {}) is None
    outcome = runner._route_completion_outcome(command, None, 10.3, {})
    assert outcome is not None and outcome.reason == "landing confirmation wait expired -> manual"
    assert sent[-1] == (0, 0, 0, 0)
    assert waits[-1][1] == pytest.approx(0.3)

    # Leaving the final hold starts a new confirmation window on re-entry.
    assert runner._route_completion_outcome(replace(command, action="FOLLOW"), None, 11.0, {}) is None
    assert runner._route_completion_outcome(command, None, 12.0, {}) is None
    assert waits[-1][1] == 0.0
