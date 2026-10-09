"""Offline regressions for the 2026-10-03 flight failure mechanisms."""

from dataclasses import replace
import math

import numpy as np

import real_path_follow_controller as rpf


def _config():
    return replace(
        rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
        waypoint_arrive_radius=0.0303,
        slowdown_distance=0.15,
    )


def _behind_leg_start():
    # Short of the new leg's first point: the planner latches REJOIN but has no
    # tangent to give, because the projection falls outside the segment.
    return rpf.Command(
        "REJOIN",
        np.array([1.0, 0.0, 0.0]),
        0.0,
        np.array([1.0, 0.0, 0.0]),
        0.064,
        0.0,
        guidance_goal=np.array([0.03, 0.0, 0.0]),
        path_pull=np.array([0.05, 0.0, -0.04]),
        path_pull_radius=0.0303,
    )


def test_rejoin_behind_the_leg_start_cannot_interrupt_the_turn():
    gate = rpf.YawAlignedPcmdController(_config())
    for i in range(10):
        now = 1.0 + i * 0.05
        pose = rpf.Pose(-0.05, 0.0, 0.04, math.radians(73), stamp=now)
        pcmd = gate.update(
            _behind_leg_start(), pose, now, target_key=1, body_velocity=(0.0, 0.0, now)
        )
        assert gate.phase == "turn"
        assert pcmd[2] != 0 and pcmd[3] == 0


def test_rejoin_behind_the_leg_start_recovers_position_once_the_nose_is_aligned():
    gate = rpf.YawAlignedPcmdController(_config())
    pose = rpf.Pose(-0.05, 0.0, 0.04, math.radians(5), stamp=1.0)
    pcmd = gate.update(_behind_leg_start(), pose, 1.0, target_key=1, body_velocity=(0.0, 0.0, 1.0))
    assert gate.phase == "route_rejoin"
    assert pcmd[2] == 0 and (pcmd[0] or pcmd[1])


def test_fast_approach_brakes_before_entering_the_waypoint_sphere():
    cfg = replace(_config(), waypoint_arrive_radius=0.051948)
    gate = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command("FOLLOW", np.zeros(3), 0.0, np.zeros(3), 0.1, 0.0)
    pose = rpf.Pose(-0.1, 0.0, 0.0, 0.0, stamp=1.0)
    pcmd = gate.update(cmd, pose, 1.0, target_key=0, body_velocity=(0.6, 0.0, 1.0))
    assert pcmd[1] < 0  # Previously accelerated into the goal at this speed.
    assert pcmd[2:] == (0, 0)


def test_low_speed_centering_keeps_baseline_damping_inside_the_sphere():
    cfg = replace(_config(), waypoint_arrive_radius=0.051948)
    cmd = rpf.Command("FOLLOW", np.zeros(3), 0.0, np.zeros(3), 0.0, 0.0)
    for distance in (0.0001, 0.005, 0.025, cfg.waypoint_arrive_radius):
        for velocity in (0.06, 0.12, rpf.DESKTOP_AUTO_APPROACH_SPEED_MPS):
            gate = rpf.YawAlignedPcmdController(cfg)
            pose = rpf.Pose(-distance, 0.0, 0.0, 0.0, stamp=1.0)
            pcmd = gate.update(cmd, pose, 1.0, target_key=0, body_velocity=(velocity, 0.0, 1.0))
            assert gate.control_terms["velocity_pcmd"] == [
                0.0,
                -rpf.DESKTOP_AUTO_WIND_REJECT_PCMD_PER_MPS * velocity,
            ]
            assert pcmd[2:] == (0, 0)


def test_turn_confirmation_keeps_position_anchor_and_full_wind_bias():
    gate = rpf.YawAlignedPcmdController(_config())
    cmd = rpf.Command(
        "FOLLOW",
        np.array([1.0, 0.0, 0.0]),
        0.0,
        np.array([1.0, 0.0, 0.0]),
        0.0,
        0.0,
        guidance_goal=np.array([1.0, 0.0, 0.0]),
        path_tangent=np.array([1.0, 0.0, 0.0]),
        path_pull=np.zeros(3),
        path_pull_radius=0.0303,
    )
    first = rpf.Pose(0.0, 0.0, 0.0, math.radians(90), stamp=1.0)
    gate.update(cmd, first, 1.0, target_key=1, body_velocity=(0.0, 0.0, 1.0))
    gate._integral_map[:] = [-0.02, 0.0]
    later = rpf.Pose(0.03, 0.0, 0.0, math.radians(5), stamp=1.05)
    pcmd = gate.update(cmd, later, 1.05, target_key=1, body_velocity=(0.0, 0.0, 1.05))
    assert gate.phase == "yaw_alignment_hold"
    assert gate.turn_anchor_error_u == 0.03
    assert pcmd[1] < 0  # Still pulls toward turn start; not velocity-only zero.
    assert gate._integral_map[0] < -0.02
    assert pcmd[2:] == (0, 0)


def _step(gate, cmd, pose):
    return gate.update(cmd, pose, pose.stamp, target_key=1, body_velocity=(0.0, 0.0, pose.stamp))


def test_started_rejoin_turn_holds_through_first_in_band_sample_until_confirmed():
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    gate = rpf.YawAlignedPcmdController(cfg)
    goal = np.array([0.5, -0.3, 0.0])
    guidance = np.array([0.0, -0.3, 0.0])

    def rejoin(pose):
        return rpf.Command("REJOIN", goal - pose.xyz, 0.0, goal, 0.2, 0.0, guidance_goal=guidance)

    pose = rpf.Pose(-0.2, 0.0, 0.0, math.pi / 2, stamp=1.0)
    pcmd = _step(gate, rejoin(pose), pose)
    assert gate.phase == "turn" and gate.alignment_started is not None
    assert pcmd[2] != 0 and pcmd[3] == 0

    # First in-band sample: the window is not confirmed, so neither REJOIN
    # position recovery nor any Gaz/along-route advance may take over.
    pose = rpf.Pose(-0.2, 0.0, 0.0, 0.0, stamp=1.05)
    pcmd = _step(gate, rejoin(pose), pose)
    assert gate.phase == "yaw_alignment_hold" and not gate.aligned_for_translation
    assert gate.alignment_started is not None
    assert pcmd == (0, 0, 0, 0)

    # The existing window confirms on the next fresh in-band sample.
    pose = rpf.Pose(-0.2, 0.0, 0.0, 0.0, stamp=1.10)
    pcmd = _step(gate, rejoin(pose), pose)
    assert gate.phase == "yaw_alignment_confirmed" and gate.aligned_for_translation
    assert pcmd[2:] == (0, 0)

    # Translation starts on the update after confirmation.
    pose = rpf.Pose(-0.2, 0.0, 0.0, 0.0, stamp=1.15)
    pcmd = _step(gate, rejoin(pose), pose)
    assert gate.phase == "route_rejoin"
    assert pcmd[2] == 0 and (pcmd[0] or pcmd[1])


def test_started_turn_survives_drift_inside_minimum_alignment_distance():
    gate = rpf.YawAlignedPcmdController(_config())
    goal = np.array([0.3, -0.3, 0.0])

    def join(pose):  # Direct-to-point leg: no guidance goal, no tangent.
        return rpf.Command("FOLLOW", goal - pose.xyz, 0.0, goal, 0.0, 0.0)

    pose = rpf.Pose(0.0, 0.0, 0.0, math.pi / 2, stamp=1.0)
    pcmd = _step(gate, join(pose), pose)
    assert gate.phase == "turn" and pcmd[2] != 0 and pcmd[3] == 0

    # Drift to 0.2 from the goal, inside max(radius, minimum alignment
    # distance): centering must not abandon the started turn or climb.
    pose = rpf.Pose(0.1, 0.0, 0.0, math.pi / 2, stamp=1.05)
    pcmd = _step(gate, join(pose), pose)
    assert gate.phase == "turn" and pcmd[2] != 0 and pcmd[3] == 0

    pose = rpf.Pose(0.1, 0.0, 0.0, 0.0, stamp=1.10)
    pcmd = _step(gate, join(pose), pose)
    assert gate.phase == "yaw_alignment_hold" and pcmd[2:] == (0, 0)
    pose = rpf.Pose(0.1, 0.0, 0.0, 0.0, stamp=1.15)
    _step(gate, join(pose), pose)
    assert gate.phase == "yaw_alignment_confirmed" and gate.aligned_for_translation
    pose = rpf.Pose(0.1, 0.0, 0.0, 0.0, stamp=1.20)
    _step(gate, join(pose), pose)
    assert gate.phase == "waypoint_centering"


def test_started_turn_is_not_auto_aligned_when_pose_moves_over_the_goal():
    gate = rpf.YawAlignedPcmdController(_config())
    goal = np.array([0.3, -0.3, 0.0])

    def join(pose):
        return rpf.Command("FOLLOW", goal - pose.xyz, 0.0, goal, 0.0, 0.0)

    pose = rpf.Pose(0.0, 0.0, 0.0, math.pi / 2, stamp=1.0)
    _step(gate, join(pose), pose)
    assert gate.phase == "turn"

    # Directly above the goal the bearing is undefined (no projected error);
    # a started turn still needs the stable window instead of auto-aligning.
    pose = rpf.Pose(0.3, 0.0, 0.0, 0.0, stamp=1.05)
    pcmd = _step(gate, join(pose), pose)
    assert gate.phase == "yaw_alignment_hold" and not gate.aligned_for_translation
    assert pcmd[2:] == (0, 0)
    pose = rpf.Pose(0.3, 0.0, 0.0, 0.0, stamp=1.10)
    _step(gate, join(pose), pose)
    assert gate.phase == "yaw_alignment_confirmed" and gate.aligned_for_translation
    pose = rpf.Pose(0.3, 0.0, 0.0, 0.0, stamp=1.15)
    _step(gate, join(pose), pose)
    assert gate.phase == "waypoint_centering"
