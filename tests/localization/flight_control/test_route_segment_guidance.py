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


def test_vo_pose_advances_mid_route_but_not_the_final_leg():
    control = controller([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    assert control.target_index == 1
    # Sequenced VO poses fly the mid-route leg: arriving at wp1 retires it.
    mid = control.step(rpf.Pose(1, 0, 0, 0, stamp=0.1, map_confirmed=False), now=0.1)
    assert control.target_index == 2
    assert mid.action == "FOLLOW"
    # The final leg still holds for a map fix at the landing waypoint.
    held = control.step(rpf.Pose(2, 0, 0, 0, stamp=0.2, map_confirmed=False), now=0.2)
    assert control.target_index == 2
    assert held.action == "HOVER" and held.status == "WAIT_MAP_CONFIRMATION"
    control.step(rpf.Pose(2, 0, 0, 0, stamp=0.3), now=0.3)
    assert control.target_index == 2


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
        # Correction-priority allocation keeps a small along-leg share while
        # repairing: lateral correction dominates, forward progress continues.
        assert yaw == 0 and gaz == 0
        assert pitch > 0
        assert abs(roll) > abs(pitch)


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


def test_turn_phase_still_corrects_toward_the_segment():
    # 2026-09-18 real flight: a FOLLOW leg sat in turn with yaw ~45 deg off
    # while wind walked it 0.08 -> 0.23 u off route. Turn keeps yaw priority
    # and height lock, but must not hold position: correction toward the
    # segment stays on, along-leg progress stays off.
    cfg = rpf.ControlConfig(inspect_waypoints=())
    gate = rpf.YawAlignedPcmdController(cfg)
    goal = np.array([10.0, 0.0, 0.0])
    command = rpf.Command(
        "FOLLOW", np.array([1.0, 0.0, 0.0]), 0.0, goal, 0.5, 0.0,
        guidance_goal=np.array([3.0, 0.0, 0.0]),
        path_pull=np.array([0.0, 0.0, -0.5]),
        path_pull_radius=0.15,
        path_tangent=np.array([1.0, 0.0, 0.0]),
    )
    pose = rpf.Pose(3, 0, 0.5, math.pi / 2, stamp=0.1)
    roll, pitch, yaw, gaz = gate.update(command, pose, 0.1, target_key=1)
    assert gate.phase == "turn"
    assert yaw != 0 and gaz == 0
    assert roll or pitch


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
    # Arrival-sphere hysteresis only: shrink the join-leg yaw-hold zone below it.
    config = rpf.ControlConfig(inspect_waypoints=(), minimum_yaw_alignment_distance=0.01)
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

def test_follow_applies_correction_priority_with_along_leg_share():
    """FOLLOW corrects small drift every tick while keeping forward progress."""
    control = controller([(0, 0, 0), (10, 0, 0)])
    control.cfg = replace(control.cfg, slowdown_distance=0.15)
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    pose = rpf.Pose(3, 0, 0.015, 0, stamp=0.1)
    command = control.step(pose, now=0.1)
    assert command.action == "FOLLOW"
    assert command.path_pull is not None
    np.testing.assert_allclose(command.path_pull, [0, 0, -0.015], atol=1e-9)
    assert command.path_pull_radius == pytest.approx(
        control._arrive_radius_for(control.target_index)
    )
    assert command.path_tangent is not None
    parts = rpf._route_translation_components(command, pose, control.cfg)
    assert parts is not None
    correction, along, along_scale = parts
    # Lateral correction points back toward the line; along-leg demand stays
    # positive so the leg keeps advancing while the drift is repaired.
    assert correction[0] > 0
    assert along[1] > 0
    assert 0.10 <= along_scale <= 1.0
    roll, pitch, yaw, _gaz = rpf.command_to_body_percent(
        command, pose, config=control.cfg, require_yaw_alignment=False)
    assert yaw == 0 and pitch > 0 and roll > 0


def test_path_pull_gain_zero_falls_back_to_carrot_guidance():
    """Gain 0 disables the segment share; steering equals direct-to-point."""
    control = controller([(0, 0, 0), (10, 0, 0)])
    control.step(rpf.Pose(0, 0, 0, 0, stamp=0), now=0)
    pose = rpf.Pose(3, 0, 0.01, 0, stamp=0.1)
    command = control.step(pose, now=0.1)
    assert command.path_pull is not None
    blended = rpf.command_to_body_percent(
        command, pose, config=control.cfg, require_yaw_alignment=False)
    flat = replace(control.cfg, path_pull_gain=0.0)
    unblended = rpf.command_to_body_percent(
        command, pose, config=flat, require_yaw_alignment=False)
    carrot = rpf.command_to_body_percent(
        replace(command, path_pull=None, path_pull_radius=None, path_tangent=None), pose,
        config=control.cfg, require_yaw_alignment=False)
    assert unblended == carrot
    assert rpf._route_translation_components(command, pose, flat) is None
    assert blended[1] >= unblended[1] or blended[0] >= unblended[0]


@pytest.mark.parametrize("offset", [-0.06, 0.06])
def test_small_arrival_spheres_do_not_amplify_horizontal_route_feedback(offset):
    """Sep 18 flight: small spheres drove alternating ~45% cross-track pulls."""
    traces = []
    for radius in (0.030347, 0.051948):
        cfg = replace(rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
                      slowdown_distance=0.34757225729367,
                      waypoint_arrive_radius=radius)
        gate = rpf.YawAlignedPcmdController(cfg)
        gate.target_key = 1
        gate.aligned_for_translation = True
        cmd = rpf.Command(
            "REJOIN", np.zeros(3), 0.0, np.array([1., 0., 0.]), abs(offset), 0.,
            path_pull=np.array([0., 0., -offset]),
            path_pull_radius=radius, path_tangent=np.array([1., 0., 0.]),
        )
        trace = []
        for index in range(21):
            now = 1.0 + index * 0.05
            pose = rpf.Pose(0.5, 0., offset, 0., stamp=now)
            trace.append(gate.update(cmd, pose, now, target_key=1,
                                     body_velocity=(0., 0., now))[0])
        traces.append(trace)
    assert traces[0] == traces[1]
    assert 0 < abs(traces[0][0]) < 15
    assert all(value * offset > 0 for value in traces[0])
    assert abs(traces[0][-1]) > abs(traces[0][0])  # Steady wind still gets integral effort.


def test_small_arrival_spheres_do_not_amplify_centering_feedback():
    traces = []
    for radius in (0.030347, 0.051948):
        cfg = replace(rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
                      slowdown_distance=0.34757225729367,
                      waypoint_arrive_radius=radius)
        gate = rpf.YawAlignedPcmdController(cfg)
        cmd = rpf.Command("FOLLOW", np.zeros(3), 0., np.array([0.02, 0., 0.]), 0., 0.)
        trace = []
        for index in range(21):
            now = 1.0 + index * 0.05
            trace.append(gate.update(cmd, rpf.Pose(0., 0., 0., 0., stamp=now), now,
                                     target_key=0, body_velocity=(0., 0., now))[1])
            assert gate.phase == "waypoint_centering"
        traces.append(trace)
    assert traces[0] == traces[1]
    assert 0 < traces[0][0] < 5


def test_large_cross_track_error_retains_bounded_wind_authority():
    cfg = replace(rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
                  slowdown_distance=0.34757225729367, waypoint_arrive_radius=0.030347)
    gate = rpf.YawAlignedPcmdController(cfg)
    gate.target_key = 1
    gate.aligned_for_translation = True
    pose = rpf.Pose(0.5, 0., 1., 0., stamp=1.)
    cmd = rpf.Command(
        "REJOIN", np.zeros(3), 0., np.array([1., 0., 0.]), 1., 0.,
        path_pull=np.array([0., 0., -1.]),
        path_pull_radius=cfg.waypoint_arrive_radius, path_tangent=np.array([1., 0., 0.]),
    )
    roll, pitch, yaw, gaz = gate.update(cmd, pose, 1., target_key=1,
                                       body_velocity=(0., -0.3, 1.))
    assert roll >= 0.9 * cfg.max_translation_pcmd
    assert max(abs(roll), abs(pitch)) <= cfg.max_translation_pcmd == 50
    assert yaw == gaz == 0


def test_join_leg_holds_yaw_inside_minimum_yaw_alignment_distance():
    # 2026-09-18 run 2: 0.03-0.12 u from waypoint 1 the join bearing (goal -
    # pose) was noise, and a re-alignment swung the nose -52 -> +60 deg.
    cfg = rpf.ControlConfig(inspect_waypoints=())
    assert cfg.minimum_yaw_alignment_distance > 0.1
    gate = rpf.YawAlignedPcmdController(cfg)
    pose = rpf.Pose(0, 0, -0.1, 0, stamp=0.1)  # goal 0.1 u north, nose east: 90 deg off
    command = rpf.Command("FOLLOW", np.zeros(3), 0, np.zeros(3), 0.1, 0.0)
    roll, pitch, yaw, _gaz = gate.update(command, pose, 0.1, target_key=0)
    assert gate.phase == "waypoint_centering"
    assert yaw == 0 and (roll, pitch) != (0, 0)


def test_join_leg_beyond_minimum_yaw_alignment_distance_still_turns_first():
    cfg = rpf.ControlConfig(inspect_waypoints=())
    gate = rpf.YawAlignedPcmdController(cfg)
    far = cfg.minimum_yaw_alignment_distance + 0.05
    pose = rpf.Pose(0, 0, -far, 0, stamp=0.1)
    command = rpf.Command("FOLLOW", np.zeros(3), 0, np.zeros(3), far, 0.0)
    roll, pitch, yaw, _gaz = gate.update(command, pose, 0.1, target_key=0)
    assert gate.phase == "turn"
    assert yaw != 0 and (roll, pitch) == (0, 0)


def _feed_guard(guard, samples, *, target=0, radius=0.03):
    return [guard.observe(target, error, altitude, radius) for altitude, error in samples]


_HOLD = [(0.5, 0.20)] * 5  # (barometric altitude m, localized height of target above u)


def test_vertical_guard_waives_when_barometer_climbs_but_localized_height_does_not():
    # Run 2: +1.8 m barometric climb while waypoint 1 went from 0.19 to 0.29 u above.
    guard = rpf.VerticalConsistencyGuard(rpf.LEGACY_MAP_FRAME)
    climb = [(0.5 + 0.1 * k, 0.20 + 0.005 * k) for k in range(1, 16)]
    assert any(_feed_guard(guard, _HOLD + climb))
    assert guard.waived == {0}
    assert guard.last_check["baro_moved_m"] >= 1.0
    assert guard.last_check["localized_followed_u"] < 0.0


def test_vertical_guard_keeps_a_localized_height_that_follows():
    guard = rpf.VerticalConsistencyGuard(rpf.LEGACY_MAP_FRAME)
    climb = [(0.5 + 0.1 * k, 0.20 - 0.1 * k / 15.0) for k in range(1, 16)]  # 15 m per map unit
    assert not any(_feed_guard(guard, _HOLD + climb))
    assert not guard.waived


def test_vertical_guard_is_direction_free():
    # A segment pull can climb toward a lower target; the frozen localized
    # height is still contradicted by the barometer.
    guard = rpf.VerticalConsistencyGuard(rpf.LEGACY_MAP_FRAME)
    samples = [(2.0, -0.09)] * 5 + [(2.0 + 0.1 * k, -0.09) for k in range(1, 16)]
    assert any(_feed_guard(guard, samples))


def test_vertical_guard_needs_a_height_gap_and_an_altitude():
    inside = rpf.VerticalConsistencyGuard(rpf.LEGACY_MAP_FRAME)
    samples = [(0.5, 0.02)] * 5 + [(0.5 + 0.2 * k, 0.02) for k in range(1, 10)]
    assert not any(_feed_guard(inside, samples, radius=0.03))
    blind = rpf.VerticalConsistencyGuard(rpf.LEGACY_MAP_FRAME)
    assert not any(blind.observe(0, 0.2, None, 0.03) for _ in range(30))
    assert blind.last_check is None


def test_vertical_guard_judges_each_target_afresh():
    guard = rpf.VerticalConsistencyGuard(rpf.LEGACY_MAP_FRAME)
    climb = [(0.5 + 0.1 * k, 0.20) for k in range(1, 16)]
    assert any(_feed_guard(guard, _HOLD + climb, target=0))
    assert not any(_feed_guard(guard, [(2.0, 0.2)] * 10, target=1))
    assert guard.waived == {0}
