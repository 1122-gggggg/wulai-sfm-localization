"""Wind memory and actuator feedback, using only the production software servo."""

from dataclasses import replace
import math

import numpy as np
import pytest

import path_follow_flight as pff
import real_path_follow_controller as rpf


def _servo():
    cfg = replace(rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
                  slowdown_distance=0.4322554798454335, waypoint_arrive_radius=0.051909)
    gate = rpf.YawAlignedPcmdController(cfg)
    command = rpf.Command("FOLLOW", np.zeros(3), 0., np.zeros(3), 0., 0.)
    for index in range(41):
        now = 1. + .05 * index
        gate.update(command, rpf.Pose(-.1, 0., 0., 0., stamp=now), now,
                    target_key=0, body_velocity=(0., 0., now))
    return gate, command


@pytest.mark.parametrize("yaw", [0., math.pi / 2])
def test_adjacent_waypoint_keeps_world_wind_bias_when_the_nose_rotates(yaw):
    gate, command = _servo()
    learned = gate._integral_map.copy()
    output = gate.update(command, rpf.Pose(0., 0., 0., yaw, stamp=3.05), 3.05,
                         target_key=1, body_velocity=(0., 0., 3.05))
    np.testing.assert_allclose(gate._integral_map, learned)
    assert output[1] * math.cos(yaw) + output[0] * math.sin(yaw) > 10
    assert output[2:] == (0, 0)


@pytest.mark.parametrize("target,observed,confirmed,gap", [
    (3, True, True, .05), (1, False, True, .05),
    (1, True, False, .05), (1, True, True, 1.),
])
def test_untrusted_or_discontinuous_transition_discards_wind_memory(target, observed, confirmed, gap):
    gate, command = _servo()
    now = 3. + gap
    gate.update(command, rpf.Pose(0, 0, 0, 0, stamp=now,
                                position_observed=observed, map_confirmed=confirmed), now,
                target_key=target, body_velocity=(0., 0., now))
    np.testing.assert_array_equal(gate._integral_map, np.zeros(2))


def test_explicit_reset_clears_wind_memory():
    gate, _ = _servo()
    gate.reset()
    np.testing.assert_array_equal(gate._integral_map, np.zeros(2))


def test_downstream_limit_prevents_more_integral_in_the_unavailable_direction():
    gate, command = _servo()
    before = gate._integral_map.copy()
    requested = gate.update(command, rpf.Pose(-.1, 0, 0, 0, stamp=3.05), 3.05,
                            target_key=0, body_velocity=(0., 0., 3.05))
    assert gate._integral_map[0] > before[0]
    gate.feedback_applied((requested[0], 0, requested[2], requested[3]))
    np.testing.assert_allclose(gate._integral_map, before)
    # A duplicate dispatch/readback cannot unwind the previously learned bias.
    gate.feedback_applied((0, 0, 0, 0))
    np.testing.assert_allclose(gate._integral_map, before)


def test_unlimited_dispatch_keeps_the_new_integral():
    gate, command = _servo()
    requested = gate.update(command, rpf.Pose(-.1, 0, 0, 0, stamp=3.05), 3.05,
                            target_key=0, body_velocity=(0., 0., 3.05))
    accumulated = gate._integral_map.copy()
    gate.feedback_applied(requested)
    np.testing.assert_allclose(gate._integral_map, accumulated)


def test_feedback_only_removes_the_blocked_component_of_diagonal_learning():
    gate, command = _servo()
    before = gate._integral_map.copy()
    requested = gate.update(command, rpf.Pose(-.1, 0, -.1, 0, stamp=3.05), 3.05,
                            target_key=0, body_velocity=(0., 0., 3.05))
    accumulated = gate._integral_map.copy()
    gate.feedback_applied((requested[0], 0, requested[2], requested[3]))
    assert gate._integral_map[0] == pytest.approx(before[0])
    assert gate._integral_map[1] == pytest.approx(accumulated[1])
    assert accumulated[1] > before[1]


def test_carried_along_track_integral_cannot_cancel_new_leg_progress():
    gate, _ = _servo()
    gate._integral_map = np.array([-.2, .05])
    command = rpf.Command(
        "FOLLOW", np.zeros(3), 0., np.array([1., 0., 0.]), 0., 0.,
        guidance_goal=np.array([.5, 0., 0.]), path_tangent=np.array([1., 0., 0.]),
        path_pull=np.zeros(3), path_pull_radius=.051909)
    for index in range(8):
        now = 3.05 + index * .05
        output = gate.update(command, rpf.Pose(.5, 0, 0, 0, stamp=now), now,
                             target_key=1, body_velocity=(0., 0., now))
    assert gate._integral_map[0] == pytest.approx(0.)
    assert gate._integral_map[1] == pytest.approx(.05)
    assert output[1] > 0 and output[0] < 0


def test_loop_feeds_final_dispatch_back_even_after_an_earlier_pose_ramp():
    gate, command = _servo()
    before = gate._integral_map.copy()
    pose = rpf.Pose(-.1, 0, 0, 0, stamp=3.05)
    gate.update(command, pose, 3.05, target_key=0, body_velocity=(0., 0., 3.05))
    ctrl = rpf.RouteAutoController([np.zeros(3), np.ones(3)], config=gate.config)
    hooks = pff.LoopHooks(get_pose=lambda: pose, olympe_yaw=lambda: 0.,
                          send_pcmd=lambda *_args: None,
                          send_authorized_pcmd=lambda _cmd: (True, 'limited', (0, 0, 0, 0)))
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False,
                                   enforce_weak_pose_gate=False)
    runner.pcmd_controller = gate
    record = {}
    runner._route_pcmd_outcome(command, pose, 0., (0, 0, 0, 0), record)
    np.testing.assert_allclose(gate._integral_map, before)
    assert record['horizontal_control_terms']['limited_pcmd'][1] > 0


def test_segment_approach_does_not_balance_headwind_outside_the_arrival_sphere():
    gate, _ = _servo()
    gate.reset()
    gate.target_key = 1
    gate.aligned_for_translation = True
    command = rpf.Command(
        'FOLLOW', np.zeros(3), 0., np.zeros(3), 0., 0.,
        guidance_goal=np.zeros(3), path_tangent=np.array([1., 0., 0.]),
        path_pull=np.zeros(3), path_pull_radius=gate.config.waypoint_arrive_radius)
    x, velocity = -.12, 0.
    errors = []
    for index in range(1200):
        now = 1. + .05 * index
        pcmd = gate.update(command, rpf.Pose(x, 0, 0, 0, stamp=now), now,
                           target_key=1, body_velocity=(velocity - .3, 0., now))
        gate.feedback_applied(pcmd)
        velocity += (pcmd[1] * .018 - velocity) * (1. - math.exp(-.05 / .4))
        x += (velocity - .3) * .05 / 15.
        if index >= 1000:
            errors.append(abs(x))
    assert max(errors) < gate.config.waypoint_arrive_radius


def test_directional_feedback_cannot_push_the_integral_outside_its_norm_bound():
    gate, _ = _servo()
    radius = max(gate._target_radius(), .5 * gate.config.slowdown_distance)
    bound = 3. * radius
    gate._integral_map = np.array([bound, 0.])
    gate._position_integral([-2. * bound, 4. * bound], rpf.Pose(0, 0, 0, 0, stamp=3.05),
                            3.05, radius, 50)
    gate._feedback_yaw = 0.
    gate._requested_horizontal = np.array([-5., 5.])
    # Blocking only the unloading axis leaves the orthogonal increment at the
    # boundary. Its combined norm must still respect the original I limit.
    gate.feedback_applied((5, 0, 0, 0))
    assert np.linalg.norm(gate._integral_map) <= bound + 1e-12
