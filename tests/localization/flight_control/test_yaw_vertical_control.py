"""AUTO may turn and change height together while horizontal motion is gated."""

from dataclasses import replace
import math

import numpy as np
import pytest

import real_path_follow_controller as rpf


FRAME = rpf.MapFrame.from_gravity(
    [-0.010870293826223672, 0.9965557122837028, 0.08221039488321148]
)


def route_command(goal, action="FOLLOW"):
    return rpf.Command(action, goal.copy(), FRAME.heading(goal), goal, 0.0, 0.0)


@pytest.mark.parametrize("action", ["FOLLOW", "REJOIN"])
@pytest.mark.parametrize("height", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("yaw_sign", [-1, 1])
def test_turn_preserves_vertical_command_without_horizontal_motion(action, height, yaw_sign):
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=FRAME)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.0)
    command = route_command(FRAME.north + height * FRAME.up, action)

    roll, pitch, yaw, gaz = rpf.command_to_body_percent(
        command, pose, config=cfg, yaw_sign=yaw_sign
    )

    assert (roll, pitch) == (0, 0)
    assert yaw * yaw_sign < 0
    assert abs(yaw) <= cfg.max_yaw_pcmd
    assert np.sign(gaz) == np.sign(height)
    assert gaz == rpf.command_to_body_percent(
        command, pose, config=cfg, require_yaw_alignment=False
    )[3]


@pytest.mark.parametrize("height", [-1.0, 1.0])
def test_cruise_translates_and_climbs_without_yaw_alignment(height):
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=FRAME)
    gate = rpf.YawAlignedPcmdController(cfg)
    command = route_command(FRAME.north + height * FRAME.up)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.0)
    pcmd = gate.update(command, pose, 0.0, target_key=0)
    assert pcmd[2] != 0
    assert gate.phase == "turn"

    pose = rpf.Pose(*(0.8 * FRAME.north), yaw=math.pi / 2, stamp=0.4)
    pcmd = gate.update(command, pose, 0.4, target_key=0)
    assert pcmd[2] == 0
    assert pcmd[3] * height > 0

    pose = rpf.Pose(*(height * FRAME.up), yaw=math.pi / 2, stamp=0.5)
    pcmd = gate.update(command, pose, 0.5, target_key=0)
    assert pcmd[2] == 0
    assert gate.phase in {
        "translate", "height_adjust", "waypoint_centering",
        "yaw_alignment_hold", "yaw_alignment_confirmed",
    }


def test_cruise_does_not_timeout_a_yaw_alignment_hold():
    cfg = rpf.ControlConfig(
        inspect_waypoints=(), map_frame=FRAME,
        yaw_alignment_timeout_s=0.2, yaw_alignment_max_windows=1,
    )
    gate = rpf.YawAlignedPcmdController(cfg)
    command = route_command(FRAME.north + FRAME.up)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=1.0)
    first = gate.update(command, pose, 1.0, target_key=0)
    pose.stamp = 1.3
    later = gate.update(command, pose, 1.3, target_key=0)
    assert first[2] != 0
    assert later == (0, 0, 0, 0)
    assert gate.phase == "yaw_alignment_timeout"


def test_rejoin_brakes_measured_motion_before_overshooting_the_line():
    cfg = rpf.production_auto_control_config(FRAME)
    pose = rpf.Pose(0, 0, 0, 0, stamp=1.0)
    cmd = rpf.Command("REJOIN", np.zeros(3), 0, FRAME.east, 0.03, 0,
                      guidance_goal=0.02 * FRAME.east)
    stopped = rpf.YawAlignedPcmdController(cfg).update(
        cmd, pose, 1.0, target_key=0, body_velocity=(0., 0., 1.0))
    moving = rpf.YawAlignedPcmdController(cfg).update(
        cmd, pose, 1.0, target_key=0, body_velocity=(0.3, 0., 1.0))
    assert stopped[1] > moving[1]
    assert stopped[2:] == moving[2:] == (0, 0)


@pytest.mark.parametrize("velocity", [(0.3, 0., 0.), (float("nan"), 0., 1.)])
def test_stale_velocity_does_not_zero_rejoin_translation(velocity):
    cfg = rpf.production_auto_control_config(FRAME)
    pose = rpf.Pose(0, 0, 0, 0, stamp=1.0)
    cmd = rpf.Command("REJOIN", np.zeros(3), 0, FRAME.east, 0.03, 0,
                      guidance_goal=0.02 * FRAME.east)
    pcmd = rpf.YawAlignedPcmdController(cfg).update(
        cmd, pose, 1.0, target_key=0, body_velocity=velocity)
    assert pcmd[2] == 0
    assert pcmd[0] or pcmd[1] or pcmd[3]


def test_height_correction_cannot_starve_larger_horizontal_drift():
    cfg = rpf.production_auto_control_config(FRAME)
    gate = rpf.YawAlignedPcmdController(cfg)
    pose = rpf.Pose(0, 0, 0, 0, stamp=1.0)
    cmd = rpf.Command("REJOIN", np.zeros(3), 0, FRAME.east, 0.1, 0,
                      guidance_goal=0.07 * FRAME.east + 0.04 * FRAME.up)
    pcmd = gate.update(cmd, pose, 1.0, target_key=0, body_velocity=(0., 0., 1.))
    assert pcmd[1] > 0 and pcmd[2:] == (0, 0)


def test_inspection_turn_does_not_command_a_height_change():
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=FRAME)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.0)
    command = route_command(FRAME.north + FRAME.up, "INSPECT")
    assert rpf.command_to_body_percent(command, pose, config=cfg) == (0, 0, -20, 0)


def test_recorded_near_vertical_target_centers_height_without_chasing_bearing():
    # 2026-09-14 16:06:15.258, f781c08a localization.jsonl:3015.
    cfg = replace(
        rpf.production_auto_control_config(FRAME),
        slowdown_distance=0.34757225729367,
        waypoint_arrive_radius=0.02221739130434783,
    )
    pose = rpf.Pose(0.3704, 0.1843, -1.3189, yaw=math.radians(144.3), stamp=4721.53)
    goal = np.array([0.3631, 0.0234, -1.3416])
    assert FRAME.horizontal_distance(goal - pose.xyz) < cfg.waypoint_arrive_radius
    assert FRAME.vertical(goal - pose.xyz) > 7 * cfg.waypoint_arrive_radius
    gate = rpf.YawAlignedPcmdController(cfg)
    pcmd = gate.update(route_command(goal), pose, 4721.579, target_key=0)
    assert pcmd[2] == 0
    assert pcmd[3] > 0
    assert gate.phase in {"waypoint_centering", "height_adjust"}


@pytest.mark.parametrize("height_sign", [-1, 1])
def test_near_vertical_waypoint_can_be_reached_before_yaw_aligns(height_sign):
    """Kinematic regression: frozen yaw must not prevent reaching the 3-D sphere."""
    cfg = replace(
        rpf.production_auto_control_config(FRAME), return_to_start=False,
        slowdown_distance=0.34757225729367,
        waypoint_arrive_radius=0.02221739130434783,
        waypoint_arrive_confirm_frames=1,
    )
    goal = 0.013 * FRAME.north + height_sign * 0.162 * FRAME.up
    controller = rpf.RouteAutoController([goal, goal + FRAME.east], config=cfg)
    gate = rpf.YawAlignedPcmdController(cfg)
    position = np.zeros(3)
    for tick in range(200):
        now = tick * 0.05
        pose = rpf.Pose(*position, yaw=0.0, stamp=now)
        command = controller.step(pose, now)
        if controller.target_index == 1:
            break
        roll, pitch, yaw, gaz = gate.update(command, pose, now, target_key=0)
        assert (roll, pitch, yaw) == (0, 0, 0)
        position += FRAME.up * gaz * 0.05 * 0.05
    assert controller.target_index == 1
    assert np.linalg.norm(goal - position) <= cfg.waypoint_arrive_radius

def _horizontal_strength(distance: float) -> int:
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=FRAME)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.0)
    roll, pitch, yaw, gaz = rpf.command_to_body_percent(
        route_command(np.array([distance, 0.0, 0.0]), "FOLLOW"),
        pose,
        config=cfg,
        require_yaw_alignment=False,
    )
    assert yaw == 0 and gaz == 0
    assert roll == 0
    return pitch


def test_approach_taper_slows_monotonically_toward_the_waypoint():
    """Limit-style approach: nearer distance never commands more authority."""
    distances = [2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.1, 0.05, 0.03, 0.01, 0.003]
    strengths = [_horizontal_strength(distance) for distance in distances]
    assert strengths[0] == rpf.ControlConfig(inspect_waypoints=()).max_translation_pcmd
    assert strengths == sorted(strengths, reverse=True)
    assert len(set(strengths)) >= 3
    assert strengths[-1] >= 1


def test_final_approach_keeps_cruise_authority_outside_the_sphere():
    """Floor of 2 outside 2x the arrival sphere: 0.06-0.10u must not creep.

    Recorded on the 6-point route (seed 13, 10 m/u): creeping the final
    approach at 1 capped the run at 10/11 waypoints in 300 s, 0.027u short
    of the last point. Only the sphere itself slows to 1.
    """
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=FRAME)
    sphere = 2.0 * cfg.waypoint_arrive_radius
    assert _horizontal_strength(0.10) >= 2
    assert _horizontal_strength(0.06) >= 2
    assert _horizontal_strength(sphere) >= 2


def test_approach_taper_keeps_minimal_creep_then_stops_inside_the_deadzone():
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=FRAME)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.0)
    creep = rpf.command_to_body_percent(
        route_command(np.array([2.0 * cfg.waypoint_arrive_radius, 0.0, 0.0]), "FOLLOW"),
        pose,
        config=cfg,
        require_yaw_alignment=False,
    )
    assert creep[1] >= 2
    stopped = rpf.command_to_body_percent(
        route_command(np.array([cfg.translation_arrival_tolerance, 0.0, 0.0]), "FOLLOW"),
        pose,
        config=cfg,
        require_yaw_alignment=False,
    )
    assert stopped == (0, 0, 0, 0)


def test_recovery_leg_keeps_full_authority_while_closing_a_gust_gap():
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=FRAME)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.0)
    goal = np.array([0.5, 0.0, 0.0])
    cruise = rpf.command_to_body_percent(
        rpf.Command("FOLLOW", goal.copy(), 0.0, goal, 0.0, 0.0),
        pose,
        config=cfg,
        require_yaw_alignment=False,
    )
    recovery = rpf.command_to_body_percent(
        rpf.Command("REJOIN", goal.copy(), 0.0, goal, 0.0, 0.0,
                    guidance_goal=goal.copy()),
        pose,
        config=cfg,
        require_yaw_alignment=False,
    )
    assert recovery[1] == cfg.max_translation_pcmd
    assert recovery[1] > cruise[1]
