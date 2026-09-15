"""Track the authored segments from a waypoint-1 start."""

from dataclasses import replace
import math

import numpy as np
import pytest

import real_path_follow_controller as rpf


def controller(points):
    return rpf.RouteAutoController(
        [np.array(point, float) for point in points],
        config=rpf.ControlConfig(inspect_waypoints=(), waypoint_arrive_confirm_frames=1),
    )


def test_waypoint_one_entry_stays_direct_then_follows_the_remaining_segments():
    control = controller([(0, 0, 0), (1, 0, 0), (2, -1, 0), (3, -1, 0)])
    assert control.start_after_nearest_waypoint(np.array([1.0, 0.0, 0.1])) == 0
    approach = control.step(rpf.Pose(1, 0, 0.1, 0, stamp=0.0), now=0.0)
    assert approach.guidance_goal is None
    np.testing.assert_allclose(approach.goal, control.wp[0])
    joined = control.step(rpf.Pose(0, 0, 0, 0, stamp=0.1), now=0.1)
    assert control.target_index == 1
    np.testing.assert_allclose(joined.goal, control.wp[1])
    assert joined.guidance_goal is not None
    assert 0.0 < joined.guidance_goal[0] < 1.0
    control.step(rpf.Pose(1, 0, 0, 0, stamp=0.2), now=0.2)
    assert control.target_index == 2


def test_vo_pose_cannot_consume_a_waypoint_without_a_fresh_map_fix():
    control = controller([(0, 0, 0), (1, 0, 0)])
    for stamp in (0.0, 0.1, 0.2):
        command = control.step(rpf.Pose(0, 0, 0, 0, stamp=stamp, map_confirmed=False), now=stamp)
        assert control.target_index == 0
        assert command.status == "WAIT_MAP_CONFIRMATION"
        assert command.action == "HOVER" and not command.should_land
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0.2), now=0.2)
    assert control.target_index == 0, "the old capture cannot gain new arrival evidence"
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0.3), now=0.3)
    assert control.target_index == 1


def test_vo_at_the_final_point_does_not_accumulate_landing_evidence():
    control = controller([(0, 0, 0), (1, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    for index in range(30):
        stamp = 1.0 + index * 0.1
        command = control.step(rpf.Pose(1, 0, 0, 0, stamp=stamp, map_confirmed=False), now=stamp)
        assert not command.should_land
    for index in range(11):
        stamp = 4.0 + index * 0.1
        command = control.step(rpf.Pose(1, 0, 0, 0, stamp=stamp), now=stamp)
        if stamp < 5.0:
            assert not command.should_land
    assert command.should_land


def _final_hold_with_periodic_unconfirmed_frames(*, reseed_confirming):
    """Flight 2026-09-15 14:25: two unconfirmed frames about every 0.9 s."""
    control = controller([(0, 0, 0), (1, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    for index in range(60):
        stamp = 1.0 + index * 0.06
        confirmed = index % 15 not in (13, 14)
        pose = rpf.Pose(
            1, 0, 0, 0, stamp=stamp, map_confirmed=confirmed,
            reseed_confirming=reseed_confirming and not confirmed,
        )
        if control.step(pose, now=stamp).should_land:
            return stamp
    return None


def test_reseed_confirmation_frames_pause_the_final_hold():
    assert _final_hold_with_periodic_unconfirmed_frames(reseed_confirming=True) == pytest.approx(2.02)


def test_other_unconfirmed_frames_still_restart_the_final_hold():
    assert _final_hold_with_periodic_unconfirmed_frames(reseed_confirming=False) is None


def test_a_long_reseed_confirmation_still_restarts_the_final_hold():
    control = controller([(0, 0, 0), (1, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    for index in range(40):
        stamp = 1.0 + index * 0.06
        confirmed = not 6 <= index <= 16  # 1.36-1.96 s: gap > max_pose_age_s
        pose = rpf.Pose(
            1, 0, 0, 0, stamp=stamp, map_confirmed=confirmed, reseed_confirming=not confirmed
        )
        if control.step(pose, now=stamp).should_land:
            break
    assert stamp == pytest.approx(3.04), "dwell restarts at the first confirmed pose, 2.02 s"


def test_climbing_during_a_turn_does_not_advance_horizontal_route_progress():
    control = controller([(0, 0, 0), (10, -5, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    low = control.step(rpf.Pose(3, -1.5, 0.002, 0, stamp=0.1), now=0.1)
    high = control.step(rpf.Pose(3, -1.49, 0.002, 0, stamp=0.2), now=0.2)
    assert low.action == high.action == "FOLLOW"
    np.testing.assert_allclose(low.guidance_goal, high.guidance_goal)
    assert high.guidance_goal[0] < high.goal[0]


def test_lateral_drift_changes_guidance_but_not_the_segment_heading():
    control = controller([(0, 0, 0), (10, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    for offset in (-0.5, 0.5):
        pose = rpf.Pose(3, 0, offset, 0, stamp=1.0)
        command = control.step(pose, now=1.0)
        assert command.yaw_target == pytest.approx(0.0)
        assert command.guidance_goal[2] == pytest.approx(0.0)
        roll, pitch, yaw, gaz = rpf.command_to_body_percent(command, pose, config=control.cfg)
        assert roll * offset > 0  # Body right points toward map -Z at yaw=0.
        assert command.action == "REJOIN"
        assert pitch == 0 and yaw == 0 and gaz == 0


def test_route_rejoin_holds_yaw_even_when_the_nose_is_not_aligned():
    control = controller([(0, 0, 0), (10, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    pose = rpf.Pose(3, 0, 0.5, math.pi / 2, stamp=0.1)
    command = control.step(pose, now=0.1)
    gate = rpf.YawAlignedPcmdController(control.cfg)
    roll, pitch, yaw, gaz = gate.update(command, pose, 0.1, target_key=1)
    assert gate.phase == "route_rejoin"
    assert roll or pitch
    assert yaw == 0 and gaz == 0


def test_route_rejoin_has_a_separate_return_to_follow_boundary():
    control = controller([(0, 0, 0), (10, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    for stamp, error, action in (
        (0.1, 0.03, "REJOIN"), (0.2, 0.018, "REJOIN"),
        (0.3, 0.014, "FOLLOW"), (0.4, 0.018, "FOLLOW"),
    ):
        assert control.step(rpf.Pose(3, 0, error, 0, stamp=stamp), now=stamp).action == action


def test_large_route_error_uses_the_full_bounded_recovery_authority():
    cfg = rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME)
    pose = rpf.Pose(3, 0, 0.08, 0, stamp=1)
    cmd = rpf.Command("REJOIN", np.array([0, 0, -1]), 0, np.array([10, 0, 0]), 0.08, 0.5,
                      guidance_goal=np.array([3, 0, 0]))
    output = rpf.YawAlignedPcmdController(cfg).update(cmd, pose, 1, target_key=1)
    assert output == (cfg.max_translation_pcmd, 0, 0, 0)


def test_centering_uses_the_actual_waypoint_instead_of_a_rejoin_projection():
    cfg = rpf.ControlConfig(inspect_waypoints=())
    gate = rpf.YawAlignedPcmdController(cfg)
    pose = rpf.Pose(0, 0.2, -0.01, 0, stamp=0.1)
    command = rpf.Command("REJOIN", np.ones(3), 0, np.zeros(3), 0.2, 0.5,
                          guidance_goal=np.array([0.1, 0.2, 0]))
    roll, pitch, yaw, gaz = gate.update(command, pose, 0.1, target_key=1)
    assert gate.phase == "waypoint_centering"
    assert (roll, pitch, yaw) == (0, 0, 0)
    assert gaz > 0


def test_centering_hysteresis_survives_fresh_updates_and_releases_when_far():
    config = rpf.ControlConfig(inspect_waypoints=())
    gate = rpf.YawAlignedPcmdController(config)
    goal = np.zeros(3)
    command = rpf.Command("FOLLOW", np.zeros(3), math.pi / 2, goal, 0, 0)
    for now, horizontal in ((0.0, 0.019), (0.2, 0.03), (0.4, 0.03), (0.6, 0.03)):
        pose = rpf.Pose(0, 0.1, -horizontal, 0, stamp=now)
        assert gate.update(command, pose, now, target_key=0)[2] == 0
        assert gate.centering
        assert gate.phase in {"waypoint_centering", "height_adjust"}
    pose = rpf.Pose(0, 0.1, -0.041, 0, stamp=0.8)
    roll, pitch, yaw, gaz = gate.update(command, pose, 0.8, target_key=0)
    assert not gate.centering
    assert yaw != 0 or gate.phase == "turn"


@pytest.mark.parametrize("limit", [0, -1, 101, True, 2.5])
def test_vertical_authority_must_be_a_valid_percentage(limit):
    with pytest.raises(ValueError, match="max_vertical_pcmd"):
        rpf.ControlConfig(max_vertical_pcmd=limit)


def test_vertical_authority_is_independent_of_horizontal_tilt():
    config = replace(rpf.ControlConfig(inspect_waypoints=()), max_translation_pcmd=3,
                     max_vertical_pcmd=20)
    pose = rpf.Pose(0, 0, 0, 0)
    goal = np.array([0.0, -10.0, 0.0])
    command = rpf.Command("FOLLOW", np.array([0.0, -1.0, 0.0]), 0, goal, 0, 0)
    assert rpf.command_to_body_percent(command, pose, config=config) == (0, 0, 0, 20)


def test_weak_lateral_axis_keeps_minimal_creep_near_waypoint():
    """Integer PCMD must not drop a meaningful secondary axis to zero.

    Recorded 2026-09-14 (session 100207Z): the closest approach stalled at
    0.089u (~4x the 0.0222u arrival radius) with yaw saturated at 50 while
    the weak roll axis rounded to 0, so lateral error kept swinging the
    target bearing away. An axis carrying >25% of the direction keeps +/-1.
    Legacy frame: east=+X, north=+Z, up=-Y, so lateral offset lives in Z.
    """
    config = replace(rpf.ControlConfig(inspect_waypoints=()), max_translation_pcmd=3,
                     max_vertical_pcmd=20, slowdown_distance=0.34757225729367,
                     waypoint_arrive_radius=0.02221739130434783)
    pose = rpf.Pose(0, 0, 0, 0)
    # Mostly forward, ~29% right, near-waypoint strength 1: roll would round to 0.
    goal = np.array([0.2, 0.0, 0.06])
    command = rpf.Command("FOLLOW", goal.copy(), 0, goal, 0, 0)
    roll, pitch, yaw, _gaz = rpf.command_to_body_percent(
        command, pose, config=config, require_yaw_alignment=False)
    assert yaw == 0
    assert pitch > 0
    assert roll != 0


def test_negligible_axis_stays_zero():
    """The floor only fires past a quarter of the direction, not for noise."""
    config = replace(rpf.ControlConfig(inspect_waypoints=()), max_translation_pcmd=3,
                     max_vertical_pcmd=20, slowdown_distance=0.34757225729367,
                     waypoint_arrive_radius=0.02221739130434783)
    pose = rpf.Pose(0, 0, 0, 0)
    goal = np.array([0.2, 0.0, 0.01])
    command = rpf.Command("FOLLOW", goal.copy(), 0, goal, 0, 0)
    roll, pitch, yaw, _gaz = rpf.command_to_body_percent(
        command, pose, config=config, require_yaw_alignment=False)
    assert yaw == 0
    assert pitch > 0
    assert roll == 0


def test_nonfinite_guidance_fails_closed():
    command = rpf.Command("FOLLOW", np.ones(3), 0, np.ones(3), 0, 0,
                          guidance_goal=np.array([math.nan, 0, 0]))
    assert rpf.command_to_body_percent(command, rpf.Pose(0, 0, 0, 0)) == (0, 0, 0, 0)
