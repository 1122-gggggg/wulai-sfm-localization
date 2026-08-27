import math

import numpy as np
import pytest

from controllers import (ABORT_OR_MANUAL, ALGORITHM_ORDER, COMPLETED, CtrlParams,
                         ExpPose, LOST_OR_UNCERTAIN, NEXT_SEGMENT, SEGMENT_ALIGN,
                         SEGMENT_FOLLOW, SEGMENT_REJOIN, WAYPOINT_HOVER,
                         SegmentCorridorController, clamp_pcmd, make_controller,
                         speed_schedule)
from route_geometry import RouteModel
from telemetry_sources import KinematicAnafi


def line_route(n=4, seg=4.0, y=-2.0):
    return RouteModel([np.array([i * seg, y, 0.0]) for i in range(n)])


def pose(x, y, z, t, **kw):
    return ExpPose(x, y, z, stamp=t, **kw)


def base_params(**over):
    p = dict(use_adaptive_lookahead=True, use_hysteresis=True,
             use_anti_oscillation=False, use_smoothed_target=False,
             body_yaw_align_at_waypoint=False, waypoint_hover_s=0.0,
             rejoin_mode="lookahead", horizontal_control_mode="nose_first")
    p.update(over)
    return CtrlParams(**p)


def make_sc(route=None, **over):
    return SegmentCorridorController(route or line_route(), base_params(**over))


def step_at(ctrl, x, y, z, heading=0.0, t=0.0):
    return ctrl.step(pose(x, y, z, t), heading, t)


# ---------------------------------------------------------------------------
# PCMD limits

def test_pcmd_clamp():
    assert clamp_pcmd(1e9, -1e9, 150.4, -150.4) == (100, -100, 100, -100)
    assert clamp_pcmd(0.4, -0.4, 99.6, 0) == (0, 0, 100, 0)


def test_all_commands_within_conservative_limits():
    ctrl = make_sc()
    p = ctrl.p
    for x, z, h in [(0, 5, 0), (-5, -5, 2.0), (6, 0.2, -1.5), (2, -4, 3.0)]:
        cmd = step_at(ctrl, x, -2.0, z, heading=h, t=0.0)
        assert abs(cmd.roll) <= p.max_roll
        assert abs(cmd.pitch) <= p.max_pitch
        assert abs(cmd.yaw) <= p.max_yaw
        assert abs(cmd.gaz) <= p.max_gaz


# ---------------------------------------------------------------------------
# Speed schedules

def test_speed_schedules():
    for kind in ("cos", "threshold", "smoothstep"):
        assert speed_schedule(kind, 0.0) == pytest.approx(1.0)
        assert speed_schedule(kind, math.pi) <= 1e-9
        assert speed_schedule(kind, math.radians(60)) <= speed_schedule(kind, 0.0)
    assert speed_schedule("threshold", math.radians(24)) == 1.0
    assert speed_schedule("threshold", math.radians(26)) == 0.0
    with pytest.raises(ValueError):
        speed_schedule("bogus", 0.0)


# ---------------------------------------------------------------------------
# Hysteresis REJOIN <-> FOLLOW

def test_hysteresis_requires_hold_time():
    ctrl = make_sc()
    # start off path (1.2 m) -> INIT_REJOIN
    cmd = step_at(ctrl, 2.0, -2.0, 1.2, t=0.0)
    assert cmd.mode in ("INIT_REJOIN", SEGMENT_REJOIN)
    # inside enter threshold but not yet held long enough -> still REJOIN
    cmd = step_at(ctrl, 2.2, -2.0, 0.3, t=0.1)
    assert cmd.mode in ("INIT_REJOIN", SEGMENT_REJOIN)
    # after hold time -> FOLLOW
    cmd = step_at(ctrl, 2.4, -2.0, 0.3, t=1.3)
    assert cmd.mode == SEGMENT_FOLLOW
    # error grows past exit threshold -> back to REJOIN (hysteresis gap)
    cmd = step_at(ctrl, 2.6, -2.0, 0.85, t=1.4)
    assert cmd.mode == SEGMENT_REJOIN
    # inside the gap (0.5..0.8) -> stays REJOIN until enter+hold again
    cmd = step_at(ctrl, 2.8, -2.0, 0.7, t=1.5)
    assert cmd.mode == SEGMENT_REJOIN


def test_hysteresis_interrupted_hold_resets():
    ctrl = make_sc()
    step_at(ctrl, 2.0, -2.0, 0.3, t=0.0)
    step_at(ctrl, 2.0, -2.0, 0.9, t=0.5)       # excursion resets the hold
    cmd = step_at(ctrl, 2.0, -2.0, 0.3, t=0.9)
    assert cmd.mode in ("INIT_REJOIN", SEGMENT_REJOIN)


def test_no_hysteresis_switches_immediately():
    ctrl = make_sc(use_hysteresis=False)
    cmd = step_at(ctrl, 2.0, -2.0, 0.7, t=0.0)  # below correction threshold 0.8
    assert cmd.mode == SEGMENT_FOLLOW


# ---------------------------------------------------------------------------
# Corridor thresholds

def test_deadband_no_lateral_correction():
    ctrl = make_sc(horizontal_control_mode="hybrid")
    step_at(ctrl, 1.0, -2.0, 0.05, t=0.0)
    cmd = step_at(ctrl, 1.2, -2.0, 0.05, t=1.2)   # in FOLLOW, err < deadband
    assert cmd.mode == SEGMENT_FOLLOW
    assert cmd.roll == 0                          # no assist inside deadband


def test_horizontal_hard_abort():
    ctrl = make_sc()
    cmd = step_at(ctrl, 2.0, -2.0, ctrl.p.horizontal_hard_abort_threshold + 0.5, t=0.0)
    assert cmd.mode == ABORT_OR_MANUAL
    assert cmd.pcmd == (0, 0, 0, 0)
    assert cmd.info["abort_reason"] == "horizontal_hard_abort"
    # terminal: stays aborted
    cmd = step_at(ctrl, 2.0, -2.0, 0.0, t=1.0)
    assert cmd.mode == ABORT_OR_MANUAL


def test_route_tube_exit_requires_time_and_effective_pose_updates():
    ctrl = make_sc(route_tube_radius=2.0, route_tube_exit_s=0.3,
                   route_tube_exit_updates=2, route_tube_initial_grace_s=0.0,
                   horizontal_hard_abort_threshold=99.0, vertical_abort_threshold=99.0)
    inside = ctrl.step(pose(2.0, -2.0, 2.0, 0.0, source_seq=1), 0.0, 0.0)
    assert inside.mode != ABORT_OR_MANUAL
    cmd = ctrl.step(pose(2.0, -2.0, 2.1, 0.1, source_seq=2), 0.0, 0.1)
    assert cmd.mode != ABORT_OR_MANUAL
    assert cmd.info["tube_exit_updates"] == 1
    # Same hloc update repeated for long enough does not count as a second update.
    cmd = ctrl.step(pose(2.0, -2.0, 2.1, 0.5, source_seq=2), 0.0, 0.5)
    assert cmd.mode != ABORT_OR_MANUAL
    assert cmd.info["tube_exit_updates"] == 1
    cmd = ctrl.step(pose(2.0, -2.0, 2.1, 0.55, source_seq=3), 0.0, 0.55)
    assert cmd.mode == ABORT_OR_MANUAL
    assert cmd.pcmd == (0, 0, 0, 0)
    assert cmd.info["abort_reason"] == "route_tube_exit"
    assert cmd.info["tube_distance"] > cmd.info["tube_radius"]
    assert cmd.info["tube_exit_updates"] == 2


def test_route_tube_uses_active_segment_window_not_wrong_nearby_branch():
    route = RouteModel([
        np.array([0.0, -2.0, 0.0]),
        np.array([6.0, -2.0, 0.0]),
        np.array([6.0, -2.0, 1.0]),
        np.array([0.0, -2.0, 1.0]),
    ])
    ctrl = make_trans(route, route_tube_radius=0.3, route_tube_segment_window=0,
                      route_tube_exit_s=0.0, route_tube_exit_updates=1,
                      route_tube_initial_grace_s=0.0,
                      horizontal_hard_abort_threshold=99.0, vertical_abort_threshold=99.0)
    cmd = ctrl.step(pose(3.0, -2.0, 1.0, 0.0, source_seq=1), 0.0, 0.0)
    assert cmd.mode == ABORT_OR_MANUAL
    assert cmd.info["abort_reason"] == "route_tube_exit"
    assert cmd.info["tube_active_seg"] == 0
    assert cmd.info["tube_seg_index"] == 0
    assert cmd.info["tube_distance"] == pytest.approx(1.0)


def test_low_confidence_hloc_pose_holds_and_does_not_tube_abort():
    ctrl = make_sc(route_tube_radius=2.0, route_tube_exit_s=0.0,
                   route_tube_exit_updates=1, route_tube_initial_grace_s=0.0,
                   hloc_min_match_count=20, horizontal_hard_abort_threshold=99.0,
                   vertical_abort_threshold=99.0)
    cmd = ctrl.step(pose(2.0, -2.0, 4.0, 0.0, source_seq=1, match_count=5), 0.0, 0.0)
    assert cmd.mode == LOST_OR_UNCERTAIN
    assert cmd.pcmd == (0, 0, 0, 0)
    assert cmd.info["pose_quality"] == "low_match_count"


def test_initial_rejoin_grace_delays_route_tube_abort():
    ctrl = make_trans(route_tube_radius=2.0, route_tube_exit_s=0.0,
                      route_tube_exit_updates=1, route_tube_initial_grace_s=1.0,
                      horizontal_hard_abort_threshold=99.0, vertical_abort_threshold=99.0)
    early = ctrl.step(pose(2.0, -2.0, 3.0, 0.0, source_seq=1), 0.0, 0.0)
    assert early.mode != ABORT_OR_MANUAL
    assert early.info["tube_safety_active"] is False
    late = ctrl.step(pose(2.0, -2.0, 3.0, 1.2, source_seq=2), 0.0, 1.2)
    assert late.mode == ABORT_OR_MANUAL
    assert late.info["abort_reason"] == "route_tube_exit"


# ---------------------------------------------------------------------------
# Vertical handling

def test_gaz_sign_up_down():
    ctrl = make_sc()
    step_at(ctrl, 1.0, -2.0, 0.0, t=0.0)
    # drone BELOW profile (up=1 < 2): raw y=-1 > target -2 -> climb -> gaz > 0
    cmd = step_at(ctrl, 1.2, -1.0, 0.0, t=0.1)
    assert cmd.gaz > 0
    # drone ABOVE profile (up=3): raw y=-3 < -2 -> descend -> gaz < 0
    cmd = step_at(ctrl, 1.4, -3.0, 0.0, t=0.2)
    assert cmd.gaz < 0
    # inside vertical deadband -> no correction
    cmd = step_at(ctrl, 1.6, -2.1, 0.0, t=0.3)
    assert cmd.gaz == 0


def test_vertical_error_does_not_trigger_horizontal_rejoin():
    ctrl = make_sc()
    step_at(ctrl, 1.0, -2.0, 0.1, t=0.0)
    cmd = step_at(ctrl, 1.2, -2.0, 0.1, t=1.2)
    assert cmd.mode == SEGMENT_FOLLOW
    # big vertical error, tiny horizontal error: must stay FOLLOW
    cmd = step_at(ctrl, 1.4, -0.5, 0.1, t=1.3)
    assert cmd.mode == SEGMENT_FOLLOW
    assert cmd.info["vert_err"] == pytest.approx(1.5)


def test_high_vertical_error_slows_forward_pitch():
    ctrl_flat = make_sc()
    step_at(ctrl_flat, 1.0, -2.0, 0.1, t=0.0)
    flat = step_at(ctrl_flat, 1.2, -2.0, 0.1, t=1.2)
    ctrl_low = make_sc()
    step_at(ctrl_low, 1.0, -2.0, 0.1, t=0.0)
    low = step_at(ctrl_low, 1.2, -0.8, 0.1, t=1.2)   # 1.2 m below profile
    assert low.pitch < flat.pitch
    assert low.info["pitch_suppressed"] == "vertical_priority"
    assert low.gaz > 0


def test_extreme_vertical_error_aborts():
    ctrl = make_sc()
    step_at(ctrl, 1.0, -2.0, 0.0, t=0.0)
    cmd = step_at(ctrl, 1.0, -2.0 + ctrl.p.vertical_abort_threshold + 0.5, 0.0, t=0.1)
    assert cmd.mode == ABORT_OR_MANUAL
    assert cmd.info["abort_reason"] == "vertical_profile_failure"


# ---------------------------------------------------------------------------
# Arrival / segment switching / hover / align

def advance_to_follow(ctrl, x=1.0, t0=0.0):
    step_at(ctrl, x, -2.0, 0.0, t=t0)
    step_at(ctrl, x + 0.2, -2.0, 0.0, t=t0 + 1.1)


def test_waypoint_arrival_and_segment_switch():
    ctrl = make_sc()
    advance_to_follow(ctrl)
    # near end of segment 0 (x=4), inside arrival radius and vertical radius
    cmd = step_at(ctrl, 3.7, -2.0, 0.0, t=2.0)
    assert cmd.mode == NEXT_SEGMENT
    assert ctrl.seg == 1
    cmd = step_at(ctrl, 3.7, -2.0, 0.0, t=2.05)
    assert cmd.mode == SEGMENT_FOLLOW            # align disabled in this preset


def test_arrival_blocked_by_vertical_radius():
    ctrl = make_sc()
    advance_to_follow(ctrl)
    cmd = step_at(ctrl, 3.7, -0.9, 0.0, t=2.0)   # 1.1 m below waypoint height
    assert cmd.mode in (SEGMENT_FOLLOW, SEGMENT_REJOIN)
    assert ctrl.seg == 0                         # no switch: vertical gate failed


def test_min_segment_time_blocks_instant_switch():
    ctrl = make_sc()
    # first valid tick already at the segment end -> too early to switch
    step_at(ctrl, 3.7, -2.0, 0.0, t=0.0)
    cmd = step_at(ctrl, 3.7, -2.0, 0.0, t=0.4)
    assert ctrl.seg == 0
    cmd = step_at(ctrl, 3.7, -2.0, 0.0, t=1.4)   # after min_segment_time and hold
    assert cmd.mode in (NEXT_SEGMENT, SEGMENT_FOLLOW, "INIT_REJOIN")


def test_progress_based_switch_without_radius():
    ctrl = make_sc(arrival_radius=0.05)          # radius can't trigger
    advance_to_follow(ctrl)
    cmd = step_at(ctrl, 3.9, -2.0, 0.3, t=2.5)   # t=0.975 > 0.95 progress
    assert cmd.mode == NEXT_SEGMENT


def test_waypoint_hover_then_next_segment():
    ctrl = make_sc(waypoint_hover_s=1.0)
    advance_to_follow(ctrl)
    cmd = step_at(ctrl, 3.8, -2.0, 0.0, t=2.0)
    assert cmd.mode == WAYPOINT_HOVER
    assert cmd.pcmd == (0, 0, 0, 0)
    cmd = step_at(ctrl, 3.8, -2.0, 0.0, t=2.5)   # still hovering
    assert cmd.mode == WAYPOINT_HOVER
    cmd = step_at(ctrl, 3.8, -2.0, 0.0, t=3.1)   # hover done -> NEXT_SEGMENT
    assert cmd.mode == NEXT_SEGMENT
    assert ctrl.seg == 1


def test_segment_align_yaws_without_pitch():
    ctrl = make_sc(body_yaw_align_at_waypoint=True, waypoint_hover_s=0.0)
    advance_to_follow(ctrl)
    step_at(ctrl, 3.8, -2.0, 0.0, t=2.0)         # NEXT_SEGMENT tick
    # heading 90 deg off the segment direction -> ALIGN with zero pitch
    cmd = step_at(ctrl, 3.8, -2.0, 0.0, heading=math.pi / 2, t=2.05)
    assert cmd.mode == SEGMENT_ALIGN
    assert cmd.pitch == 0 and cmd.yaw != 0
    assert cmd.info["pitch_suppressed"] == "align"
    # aligned -> FOLLOW
    cmd = step_at(ctrl, 3.8, -2.0, 0.0, heading=0.0, t=2.1)
    assert cmd.mode == SEGMENT_FOLLOW


def test_align_timeout():
    ctrl = make_sc(body_yaw_align_at_waypoint=True, segment_align_timeout_s=2.0)
    advance_to_follow(ctrl)
    step_at(ctrl, 3.8, -2.0, 0.0, t=2.0)
    step_at(ctrl, 3.8, -2.0, 0.0, heading=math.pi, t=2.05)
    cmd = step_at(ctrl, 3.8, -2.0, 0.0, heading=math.pi, t=4.2)  # timeout
    assert cmd.mode == SEGMENT_FOLLOW


def test_route_completion():
    ctrl = make_sc(waypoint_hover_s=0.0)
    t = 0.0
    advance_to_follow(ctrl)
    t = 1.2
    for seg_end in [3.8, 7.8, 11.8]:
        t += 1.5
        cmd = step_at(ctrl, seg_end, -2.0, 0.0, t=t)
        if cmd.mode == NEXT_SEGMENT:
            t += 0.05
            cmd = step_at(ctrl, seg_end, -2.0, 0.0, t=t)
    assert ctrl.mode == COMPLETED
    assert step_at(ctrl, 11.8, -2.0, 0.0, t=t + 1).pcmd == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Nose-first vs lateral assist

def test_nose_first_never_rolls():
    ctrl = make_sc(horizontal_control_mode="nose_first")
    for t, z in [(0.0, 0.4), (1.2, 0.4), (1.3, 0.3)]:
        cmd = step_at(ctrl, 1.0 + t, -2.0, z, t=t)
        assert cmd.roll == 0


def test_lateral_assist_small_error_only_and_bounded():
    ctrl = make_sc(horizontal_control_mode="hybrid")
    advance_to_follow(ctrl)
    # moderate cross-track inside assist threshold, facing along route
    cmd = step_at(ctrl, 1.5, -2.0, 0.4, heading=0.0, t=1.3)
    assert cmd.mode == SEGMENT_FOLLOW
    assert abs(cmd.roll) <= ctrl.p.max_lateral_roll_percent
    assert cmd.roll < 0                          # route is at z<pos -> roll left
    # large error -> REJOIN -> nose-first (no roll)
    cmd = step_at(ctrl, 1.5, -2.0, 1.5, heading=0.0, t=1.4)
    assert cmd.mode == SEGMENT_REJOIN
    assert cmd.roll == 0


def test_rejoin_target_modes():
    r = line_route()
    near = SegmentCorridorController(r, base_params(rejoin_mode="nearest"))
    look = SegmentCorridorController(r, base_params(rejoin_mode="lookahead"))
    p_off = (2.0, -2.0, 2.0)                     # 2 m beside the route
    c_near = step_at(near, *p_off, t=0.0)
    c_look = step_at(look, *p_off, t=0.0)
    assert np.allclose(c_near.info["rejoin_target"][2], 0.0, atol=1e-9)
    assert c_near.info["rejoin_target"][0] == pytest.approx(2.0)   # nearest point
    assert c_look.info["rejoin_target"][0] > 2.5                   # ahead on route
    assert c_look.info["lookahead"] > 0


# ---------------------------------------------------------------------------
# Anti-oscillation and smoothed target

def test_anti_oscillation_damps_yaw():
    ctrl = make_sc(use_anti_oscillation=True, osc_flip_limit=4)
    t, z = 0.0, 0.45
    advance_to_follow(ctrl)
    yaws = []
    for i in range(40):                          # zigzag pose forces yaw flips
        t = 1.3 + i * 0.1
        z = 0.45 if i % 2 == 0 else -0.45
        cmd = step_at(ctrl, 1.5 + 0.02 * i, -2.0, z, heading=0.0, t=t)
        yaws.append(cmd.yaw)
    assert any(c.info.get("osc_damped") for c in [cmd])  # damping engaged at end


def test_smoothed_target_rate_limited():
    ctrl = make_sc(use_smoothed_target=True, max_target_speed=1.0)
    step_at(ctrl, 1.0, -2.0, 0.0, t=0.0)
    t1 = np.asarray(step_at(ctrl, 1.0, -2.0, 0.0, t=0.05).info["rejoin_target"])
    # pose teleports sideways: raw carrot jumps, filtered carrot must not
    t2 = np.asarray(step_at(ctrl, 1.0, -2.0, 3.0, t=0.10).info["rejoin_target"])
    assert np.linalg.norm(t2 - t1) <= 1.0 * 0.05 + 1e-6


# ---------------------------------------------------------------------------
# Lost / stale telemetry

def test_stale_pose_zero_pcmd_then_abort():
    ctrl = make_sc()
    step_at(ctrl, 1.0, -2.0, 0.0, t=0.0)
    cmd = ctrl.step(pose(1.0, -2.0, 0.0, 0.0), 0.0, 5.0)   # 5 s old pose
    assert cmd.mode == LOST_OR_UNCERTAIN
    assert cmd.pcmd == (0, 0, 0, 0)
    cmd = ctrl.step(None, None, 5.0 + ctrl.p.lost_abort_s + 1.0)
    assert cmd.mode == ABORT_OR_MANUAL
    assert cmd.info["abort_reason"] == "telemetry_lost"


def test_lost_recovery_resumes_rejoin():
    ctrl = make_sc(lost_abort_s=10.0)
    step_at(ctrl, 1.0, -2.0, 0.0, t=0.0)
    ctrl.step(None, None, 1.0)
    cmd = step_at(ctrl, 1.0, -2.0, 0.2, t=2.0)
    assert cmd.mode in (SEGMENT_REJOIN, SEGMENT_FOLLOW)


# ---------------------------------------------------------------------------
# Registry / presets

def test_registry_builds_all_algorithms():
    r = line_route()
    for name in ALGORITHM_ORDER:
        c = make_controller(name, r)
        cmd = c.step(pose(0.5, -2.0, 0.8, 0.0), 0.0, 0.0)
        assert cmd.pcmd == tuple(int(v) for v in cmd.pcmd)
        assert all(abs(v) <= 100 for v in cmd.pcmd)
    with pytest.raises(ValueError):
        make_controller("bogus", r)


def test_overrides_apply():
    c = make_controller("adaptive_lookahead", line_route(),
                        {"horizontal_control_mode": "hybrid", "max_lateral_roll_percent": 3})
    assert c.p.horizontal_control_mode == "hybrid"
    assert c.p.max_lateral_roll_percent == 3


# ---------------------------------------------------------------------------
# End-to-end kinematic convergence (the one runnable check that fails if the
# control logic breaks: random offset start -> converge -> complete route)

def closed_loop(algorithm, start, yaw0, route=None, T=90.0, dt=0.05):
    route = route or line_route()
    plant = KinematicAnafi(np.asarray(start, float), yaw0)
    ctrl = make_controller(algorithm, route)
    cts = []
    while plant.t < T:
        p = plant.get_pose()
        cmd = ctrl.step(p, plant.yaw(), plant.t)
        plant.send_pcmd(*cmd.pcmd)
        plant.step(dt)
        cts.append(route.project(plant.get_pose().xyz).cross_track)
        if ctrl.mode in (COMPLETED, ABORT_OR_MANUAL):
            break
    return ctrl, cts


def test_adaptive_lookahead_converges_from_offset_start():
    route = line_route()
    ctrl, cts = closed_loop("adaptive_lookahead", [-2.0, -2.0, 3.0],
                            math.radians(20.0), route)
    assert min(cts) < 0.3                        # reached the corridor
    assert cts[-1] < 0.6
    assert ctrl.mode == COMPLETED                # finished the route
    assert np.mean(cts[-40:]) < 0.5              # stayed near the path


def test_all_corridor_algorithms_make_progress():
    for name in ("segment_corridor", "segment_corridor_hover",
                 "segment_corridor_nose_first", "adaptive_smoothed"):
        ctrl, cts = closed_loop(name, [-1.5, -2.0, 2.0], 0.3)
        assert min(cts) < 0.5, name


# ---------------------------------------------------------------------------
# Yaw-locked translational controller (user variant)

from controllers import (MOVE_MODES_14, TranslationalWaypointController,
                         classify_mode_14)


def test_classify_mode_14_faces_and_corners():
    assert len(MOVE_MODES_14) == 14                    # 6 faces + 8 corners
    assert classify_mode_14(1, 0, 0) == "forward"
    assert classify_mode_14(-1, 0, 0) == "back"
    assert classify_mode_14(0, 1, 0) == "right"
    assert classify_mode_14(0, -1, 0) == "left"
    assert classify_mode_14(0, 0, 1) == "up"
    assert classify_mode_14(0, 0, -1) == "down"
    assert classify_mode_14(1, 1, 1) == "forward_up_right"
    assert classify_mode_14(1, -1, -1) == "forward_down_left"
    assert classify_mode_14(-1, 1, -1) == "back_down_right"
    assert classify_mode_14(0, 0, 0) == "hover"
    # a mostly-forward, slightly-right, slightly-up vector snaps to the corner
    assert classify_mode_14(0.9, 0.6, 0.6) == "forward_up_right"


def make_trans(route=None, **over):
    from controllers import CtrlParams
    base = dict(waypoint_hover_s=1.0, arrival_radius=0.4, arrival_vertical_radius=0.4,
                use_smoothed_target=False, use_anti_oscillation=False)
    base.update(over)
    return TranslationalWaypointController(route or line_route(), CtrlParams(**base))


def test_translational_first_target_is_second_waypoint():
    ctrl = make_trans()
    assert ctrl.k == 1                                 # connects start -> P1 (2nd point)


def test_translational_aligns_before_first_segment_then_translates_without_yaw():
    ctrl = make_trans(segment_align_yaw_tolerance_deg=3.0)

    turning = ctrl.step(pose(0.0, -2.0, 0.0, 0.0), math.pi / 2, 0.0)
    assert turning.mode == SEGMENT_ALIGN
    assert turning.roll == turning.pitch == turning.gaz == 0
    assert turning.yaw < 0
    assert turning.info["align_to_wp"] == 1
    assert ctrl.k == 1

    aligned = ctrl.step(pose(0.0, -2.0, 0.0, 0.1), 0.0, 0.1)
    assert aligned.mode == NEXT_SEGMENT
    assert aligned.pcmd == (0, 0, 0, 0)

    translating = ctrl.step(pose(0.0, -2.0, 0.0, 0.2), 0.0, 0.2)
    assert translating.mode == SEGMENT_FOLLOW
    assert translating.pitch > 0
    assert translating.yaw == 0


def test_translational_alignment_timeout_aborts_instead_of_translating():
    ctrl = make_trans(
        segment_align_yaw_tolerance_deg=3.0,
        segment_align_timeout_s=0.1,
    )

    turning = ctrl.step(pose(0.0, -2.0, 0.0, 0.0), math.pi / 2, 0.0)
    assert turning.mode == SEGMENT_ALIGN
    timed_out = ctrl.step(pose(0.0, -2.0, 0.0, 0.2), math.pi / 2, 0.2)
    assert timed_out.mode == ABORT_OR_MANUAL
    assert timed_out.pcmd == (0, 0, 0, 0)
    assert timed_out.info["abort_reason"] == "segment_alignment_timeout"


def test_translational_no_yaw_during_segment():
    ctrl = make_trans()
    # heading 0 (north); target wp[1] at x=4 ahead -> pure forward, zero yaw/roll
    cmd = ctrl.step(pose(0.5, -2.0, 0.0, 0.0), 0.0, 0.0)
    assert cmd.mode == SEGMENT_FOLLOW
    assert cmd.yaw == 0
    assert cmd.pitch > 0                               # forward translation
    assert cmd.info["move_mode"] == "forward"


def test_translational_body_decomposition_signs():
    ctrl = make_trans()
    # heading 0 (facing +x/north). Put wp[1] target; drone offset so target is
    # forward-and-right-and-up in body frame.
    # target wp[1] = (4,-2,0); drone at (2,-1,-1): dx=+2(fwd), dz=+1(right), up: target up=2 > drone up=1 -> climb
    cmd = ctrl.step(pose(2.0, -1.0, -1.0, 0.0), 0.0, 0.0)
    assert cmd.yaw == 0
    assert cmd.pitch > 0                               # forward component
    assert cmd.roll > 0                                # right component (+z at heading 0)
    assert cmd.gaz > 0                                 # target higher -> climb
    assert cmd.info["move_mode"] == "forward_up_right"


def test_translational_move_mode_is_label_not_discrete_speed():
    ctrl_a = make_trans(max_roll=10, max_gaz=10)
    ctrl_b = make_trans(max_roll=10, max_gaz=10)
    a = ctrl_a.step(pose(2.0, -1.0, -1.0, 0.0), 0.0, 0.0)
    b = ctrl_b.step(pose(2.0, -1.3, -1.5, 0.0), 0.0, 0.0)
    assert a.info["move_mode"] == b.info["move_mode"] == "forward_up_right"
    assert a.pcmd != b.pcmd
    assert a.pitch / a.roll != pytest.approx(b.pitch / b.roll)


def test_translational_left_and_down():
    ctrl = make_trans()
    # target behind-left-below in body frame at heading 0:
    # drone at (6,-3,2): target wp[1]=(4,-2,0): dx=-2(back), dz=-2(left), target up=2<drone up=3 -> descend
    cmd = ctrl.step(pose(6.0, -3.0, 2.0, 0.0), 0.0, 0.0)
    assert cmd.pitch < 0 and cmd.roll < 0 and cmd.gaz < 0
    assert cmd.info["move_mode"] == "back_down_left"


def test_translational_horizontal_resultant_capped():
    ctrl = make_trans(max_horiz_translate=10)
    cmd = ctrl.step(pose(0.0, -2.0, 5.0, 0.0), 0.0, 0.0)   # far off to the side
    assert math.hypot(cmd.pitch, cmd.roll) <= 10 + 1e-6


def test_translational_3d_resultant_capped():
    ctrl = make_trans(max_pitch=10, max_roll=10, max_gaz=10, max_horiz_translate=10)
    cmd = ctrl.step(pose(2.0, 0.0, -2.0, 0.0), 0.0, 0.0)   # forward + right + up
    assert cmd.info["move_mode"] == "forward_up_right"
    assert cmd.pitch > 0 and cmd.roll > 0 and cmd.gaz > 0
    assert max(abs(cmd.pitch), abs(cmd.roll), abs(cmd.gaz)) < 10
    assert math.sqrt(cmd.pitch ** 2 + cmd.roll ** 2 + cmd.gaz ** 2) <= 10 + 1e-6


def test_translational_route_tube_exit_aborts_to_manual_hover():
    ctrl = make_trans(route_tube_radius=2.0, route_tube_exit_s=0.0,
                      route_tube_exit_updates=1, route_tube_initial_grace_s=0.0,
                      horizontal_hard_abort_threshold=99.0, vertical_abort_threshold=99.0)
    cmd = ctrl.step(pose(2.0, -2.0, 2.1, 0.0, source_seq=1), 0.0, 0.0)
    assert cmd.mode == ABORT_OR_MANUAL
    assert cmd.pcmd == (0, 0, 0, 0)
    assert cmd.info["abort_reason"] == "route_tube_exit"
    assert cmd.info["tube_distance"] > 2.0


def test_translational_waypoint_align_uses_camera_yaw_offset():
    ctrl = make_trans(camera_yaw_offset_deg=90.0)
    # At take-off the camera ray is already aligned with the first leg when the
    # body is -90 deg and the camera mounting offset is +90 deg.
    ctrl.step(pose(0.0, -2.0, 0.0, 0.0), -math.pi / 2, 0.0)
    ctrl.step(pose(3.8, -2.0, 0.0, 1.2), -math.pi / 2, 1.2)
    cmd = ctrl.step(pose(3.8, -2.0, 0.0, 2.5), -math.pi / 2, 2.5)
    assert cmd.mode == NEXT_SEGMENT
    assert ctrl.k == 2
    assert cmd.info["camera_heading_target"] == pytest.approx(0.0)
    assert cmd.info["heading_target"] == pytest.approx(-math.pi / 2)


def test_translational_pcmd_rate_limit_smooths_direction_changes():
    ctrl = make_trans(pcmd_rate_limit_pct_per_s=20.0,
                      horizontal_hard_abort_threshold=99.0, vertical_abort_threshold=99.0)
    first = ctrl.step(pose(0.5, -2.0, 0.0, 0.0), 0.0, 0.0)
    assert first.pitch > 0
    second = ctrl.step(pose(6.0, -2.0, 0.0, 0.05), 0.0, 0.05)
    assert second.pitch >= first.pitch - 1


def test_translational_yaw_only_at_waypoint():
    ctrl = make_trans()
    # arrive at wp[1]=(4,-2,0): within arrival radius, after min_segment_time
    ctrl.step(pose(3.8, -2.0, 0.0, 0.0), 0.0, 0.0)
    cmd = ctrl.step(pose(3.8, -2.0, 0.0, 1.2), 0.0, 1.2)
    assert cmd.mode == WAYPOINT_HOVER and cmd.pcmd == (0, 0, 0, 0)
    # after hover -> ALIGN yaws (only here) toward wp[2]; heading 90deg off
    cmd = ctrl.step(pose(3.8, -2.0, 0.0, 2.5), math.pi / 2, 2.5)
    assert cmd.mode == SEGMENT_ALIGN
    assert cmd.yaw != 0 and cmd.pitch == 0 and cmd.roll == 0
    assert cmd.info["align_to_wp"] == 2
    # aligned -> advance to next segment (k=2)
    cmd = ctrl.step(pose(3.8, -2.0, 0.0, 2.6), 0.0, 2.6)
    assert ctrl.k == 2


def test_translational_waypoint_arrival_uses_3d_radius():
    ctrl = make_trans(arrival_radius=1.0, min_segment_time_s=0.0)
    cmd = ctrl.step(pose(3.4, -1.2, 0.0, 0.0), 0.0, 0.0)
    assert cmd.mode == WAYPOINT_HOVER
    assert cmd.info["d_to_wp"] == pytest.approx(1.0)

    ctrl = make_trans(arrival_radius=1.0, min_segment_time_s=0.0)
    cmd = ctrl.step(pose(3.4, -1.19, 0.0, 0.0), 0.0, 0.0)
    assert cmd.mode == SEGMENT_FOLLOW
    assert cmd.info["d_to_wp"] > 1.0


def test_translational_final_landing_uses_stricter_stable_gate():
    route = RouteModel([np.array([0.0, -2.0, 0.0]),
                        np.array([4.0, -2.0, 0.0])])
    ctrl = make_trans(route, arrival_radius=1.0, final_landing_radius=0.3,
                      final_landing_hold_s=0.2, final_landing_max_est_speed=10.0,
                      min_segment_time_s=0.0, waypoint_hover_s=0.0)

    cmd = ctrl.step(pose(3.5, -2.0, 0.0, 0.0), 0.0, 0.0)
    assert cmd.mode == SEGMENT_FOLLOW
    assert cmd.info["arrival_radius"] == pytest.approx(0.3)

    cmd = ctrl.step(pose(4.0, -2.0, 0.0, 0.1), 0.0, 0.1)
    assert cmd.mode == WAYPOINT_HOVER
    cmd = ctrl.step(pose(4.0, -2.0, 0.0, 0.41), 0.0, 0.41)
    assert cmd.mode == WAYPOINT_HOVER
    assert cmd.info["final_landing_pending"] is True
    cmd = ctrl.step(pose(4.0, -2.0, 0.0, 0.65), 0.0, 0.65)
    assert cmd.mode == COMPLETED
    assert cmd.info["should_land"] is True


def test_translational_waypoint_align_can_use_custom_yaw_target():
    route = RouteModel([
        np.array([0.0, -2.0, 0.0]),
        np.array([4.0, -2.0, 0.0]),
        np.array([4.0, -4.0, 0.0]),     # vertical inspection segment
    ])
    route.yaw_targets = {1: math.pi / 2}
    ctrl = make_trans(route, arrival_radius=0.3, min_segment_time_s=0.0)

    cmd = ctrl.step(pose(4.0, -2.0, 0.0, 0.0), 0.0, 0.0)
    assert cmd.mode == WAYPOINT_HOVER
    cmd = ctrl.step(pose(4.0, -2.0, 0.0, 1.2), 0.0, 1.2)
    assert cmd.mode == SEGMENT_ALIGN
    assert cmd.info["camera_heading_target"] == pytest.approx(math.pi / 2)
    assert cmd.yaw > 0


def test_translational_completes_route():
    route = line_route()
    from telemetry_sources import KinematicAnafi
    plant = KinematicAnafi(np.array([-1.0, -2.0, 1.5]), math.radians(12))
    ctrl = make_controller("translational_waypoint", route)
    yaws_in_seg = []
    while plant.t < 120:
        p = plant.get_pose()
        cmd = ctrl.step(p, plant.yaw(), plant.t)
        if cmd.mode == SEGMENT_FOLLOW:
            yaws_in_seg.append(cmd.yaw)
        plant.send_pcmd(*cmd.pcmd)
        plant.step(0.05)
        if ctrl.mode in (COMPLETED, ABORT_OR_MANUAL):
            break
    assert ctrl.mode == COMPLETED
    assert all(y == 0 for y in yaws_in_seg)            # never yawed mid-segment
    assert route.project(plant.get_pose().xyz).cross_track < 0.5


def test_translational_completes_height_route():
    from route_geometry import build_route
    from telemetry_sources import KinematicAnafi
    route = RouteModel(build_route("s_curve", np.array([0.0, -2.5, 0.0]),
                                   seg_len=4.0, height_amp=1.5))
    plant = KinematicAnafi(route.wp[0] + np.array([1.5, 0.3, -1.0]), 0.2)
    ctrl = make_controller("translational_waypoint", route)
    modes = set()
    while plant.t < 200:
        p = plant.get_pose()
        cmd = ctrl.step(p, plant.yaw(), plant.t)
        modes.add(cmd.info.get("move_mode"))
        plant.send_pcmd(*cmd.pcmd)
        plant.step(0.05)
        if ctrl.mode in (COMPLETED, ABORT_OR_MANUAL):
            break
    assert ctrl.mode == COMPLETED
    # a height+turning route must exercise 3D-corner modes, not just faces
    assert any("_" in m for m in modes if m), modes
