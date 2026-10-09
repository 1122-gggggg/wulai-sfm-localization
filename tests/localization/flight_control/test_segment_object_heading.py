"""Offline yaw-only object tracking on selected authored waypoint legs."""

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest

import real_path_follow_controller as rpf
from route_domain import WaypointAction


POINTS = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0))
TARGET = (4.0, 0.0, 5.0)


def _config(targets=(TARGET, None), **overrides):
    values = dict(
        inspect_waypoints=(), waypoint_arrive_confirm_frames=1,
        yaw_alignment_confirmation_updates=2, segment_look_at_points=targets,
    )
    values.update(overrides)
    return rpf.ControlConfig(**values)


def _controller(targets=(TARGET, None), *, points=POINTS, **overrides):
    return rpf.RouteAutoController(
        [np.array(point, dtype=float) for point in points],
        config=_config(targets, **overrides),
    )


def _pose(xyz, yaw=0.0, stamp=0.0, **extra):
    return rpf.Pose(*xyz, yaw, stamp=stamp, **extra)


def _step(control, pose):
    return control.step(pose, now=pose.stamp)


def _pcmd(gate, command, pose, target=1, **extra):
    return gate.update(command, pose, pose.stamp, target_key=target, **extra)


def _confirm(control, gate, xyz, *, stamp=0.0):
    heading = control.cfg.map_frame.heading(np.asarray(TARGET) - xyz)
    for index in range(control.cfg.yaw_alignment_confirmation_updates):
        pose = _pose(xyz, heading, stamp + index * 0.05)
        _pcmd(gate, _step(control, pose), pose, control.target_index)
    assert gate.aligned_for_translation
    return heading


def test_moving_pose_changes_object_yaw_without_changing_authored_guidance():
    aimed = _controller()
    normal = _controller(None)
    for control in (aimed, normal):
        _step(control, _pose(POINTS[0]))
    headings = []
    for index, xyz in enumerate(((2, 0, 0), (4, 0, 0), (6, 0, 0))):
        pose = _pose(xyz, stamp=0.1 + index * 0.1)
        command = _step(aimed, pose)
        baseline = _step(normal, pose)
        expected = aimed.cfg.map_frame.heading(np.asarray(TARGET) - pose.xyz)
        assert command.yaw_target == pytest.approx(expected)
        headings.append(command.yaw_target)
        np.testing.assert_allclose(command.look_at_target, TARGET)
        assert command.look_at_pole is None
        for field in ("goal", "guidance_goal", "path_pull", "path_tangent", "vel_map"):
            np.testing.assert_allclose(getattr(command, field), getattr(baseline, field))
        for field in ("action", "progress", "path_error", "corner_weight", "path_pull_radius"):
            assert getattr(command, field) == getattr(baseline, field)
    assert headings[0] < headings[1] < headings[2]


def _object_leg_pcmd_sequence(target):
    control = _controller((target, None))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    _step(control, _pose(POINTS[0]))
    yaw, sequence = 0.0, []
    for tick in range(60):
        pose = _pose((2.0, -0.01, 0.05), yaw, stamp=0.1 + tick * 0.05)
        command = _step(control, pose)
        pcmd = _pcmd(gate, command, pose, control.target_index)
        sequence.append((command.yaw_target, pcmd))
        yaw += -pcmd[2] / 100.0 * 0.2
    return sequence


@pytest.mark.parametrize("box_height", [-12.0, -3.0, 3.0, 12.0])
def test_object_height_does_not_change_yaw_or_translation_commands(box_height):
    """Only the box's horizontal bearing matters; its height is never flown to."""
    raised_or_lowered_box = (TARGET[0], box_height, TARGET[2])
    reference = _object_leg_pcmd_sequence(TARGET)
    assert any(pcmd[2] for _, pcmd in reference)  # the nose really turns
    assert any(pcmd[0] or pcmd[1] for _, pcmd in reference)  # and it then translates
    assert _object_leg_pcmd_sequence(raised_or_lowered_box) == reference


def test_initial_turn_faces_object_then_stable_window_unlocks_translation():
    control = _controller()
    _step(control, _pose(POINTS[0]))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    wrong = _pose((2, 0, 0), yaw=0.0, stamp=0.05)
    pcmd = _pcmd(gate, _step(control, wrong), wrong)
    assert gate.phase == "turn" and not gate.aligned_for_translation
    assert pcmd[:2] == (0, 0) and pcmd[2] < 0 and pcmd[3] == 0
    heading = control.cfg.map_frame.heading(np.asarray(TARGET) - wrong.xyz)
    for index in range(2):
        aligned = _pose(wrong.xyz, heading, 0.1 + index * 0.05)
        pcmd = _pcmd(gate, _step(control, aligned), aligned)
        assert pcmd == (0, 0, 0, 0)
    assert gate.aligned_for_translation and gate.phase == "yaw_alignment_confirmed"
    translating = _pose((2, 0, 0), heading, 0.2)
    pcmd = _pcmd(gate, _step(control, translating), translating)
    assert gate.phase == "translate" and pcmd[0] != 0 and pcmd[1] != 0


@pytest.mark.parametrize("yaw_sign", [-1, 1])
@pytest.mark.parametrize("error_deg", [-2.0, 2.0, 80.0])
def test_tracking_corrects_small_errors_and_blends_capped_yaw_with_roll_pitch(yaw_sign, error_deg):
    control = _controller()
    _step(control, _pose(POINTS[0]))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    heading = _confirm(control, gate, np.array([2.0, 0.0, 0.0]), stamp=0.05)
    pose = _pose((2, 0, 0), heading - math.radians(error_deg), 0.2)
    command = _step(control, pose)
    output = _pcmd(gate, command, pose, yaw_sign=yaw_sign)
    assert gate.phase == "translate" and gate.aligned_for_translation
    assert output[0] or output[1]
    assert output[2] * error_deg * yaw_sign < 0
    assert output[3] == 0
    # The object override only replaces yaw; identical actual body heading and
    # movement metadata produce exactly the same horizontal/vertical servo.
    normal = rpf.YawAlignedPcmdController(control.cfg)
    normal.target_key = 1
    normal.aligned_for_translation = True
    baseline = _pcmd(normal, replace(command, look_at_target=None), pose)
    assert output[:2] == baseline[:2] and output[3] == baseline[3]


def test_tracking_preserves_configured_yaw_cap():
    control = _controller(max_yaw_pcmd=3)
    _step(control, _pose(POINTS[0]))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    heading = _confirm(control, gate, np.array([2.0, 0, 0]), stamp=0.05)
    pose = _pose((2, 0, 0), heading - math.radians(80), 0.2)
    output = _pcmd(gate, _step(control, pose), pose)
    assert output[2] == -3
    assert output[0] or output[1]


def test_moving_object_bearing_is_tracked_at_production_translation_speed():
    points = ((-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5))
    config = replace(
        rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
        return_to_start=False, segment_look_at_points=((0, 0, 0), None),
    )
    control = rpf.RouteAutoController([np.asarray(point) for point in points], config=config)
    gate = rpf.YawAlignedPcmdController(control.cfg)
    yaw = math.pi / 4.0
    for index in range(config.yaw_alignment_confirmation_updates):
        pose = _pose(points[0], yaw=yaw, stamp=index * 0.05)
        _pcmd(gate, _step(control, pose), pose)
    errors = []
    yaw_rate = 0.0
    for index in range(1, 251):
        # 0.3 m/s at 4 m/u, passing 2 m horizontally from the object center.
        pose = _pose(
            (-0.5 + 0.075 * index * 0.05, 0, -0.5),
            yaw=yaw, stamp=0.2 + index * 0.05,
        )
        command = _step(control, pose)
        errors.append(abs(rpf.wrap_angle(command.yaw_target - yaw)))
        output = _pcmd(gate, command, pose)
        # Existing desktop yaw envelope: 20 deg/s at 100% with 0.2 s response.
        demand = -math.radians(20.0) * output[2] / 100.0
        yaw_rate += (demand - yaw_rate) * 0.05 / 0.2
        yaw = rpf.wrap_angle(yaw + yaw_rate * 0.05)
    assert math.degrees(max(errors)) < 10.0


def test_adapter_tracks_passed_control_pose_instead_of_stale_authored_ray():
    control = _controller()
    _step(control, _pose(POINTS[0]))
    observed = _pose((2, 0, 0), stamp=0.1)
    command = _step(control, observed)
    # Existing runtime may supply bounded smoothed/predicted control position.
    # The physical target remains unchanged; the servo uses the passed pose and
    # actual body yaw, not a substituted waypoint or travel heading.
    pose = _pose((2.005, 0, 0.005), command.yaw_target, stamp=0.1)
    expected = control.cfg.map_frame.heading(np.asarray(TARGET) - pose.xyz)
    assert expected > command.yaw_target
    output = rpf.command_to_body_percent(
        command, pose, config=control.cfg, require_yaw_alignment=False,
    )
    assert output[2] < 0
    np.testing.assert_allclose(command.look_at_target, TARGET)


@pytest.mark.parametrize("xyz, phase", [((9.95, 0, 0), "waypoint_centering"), ((3, 0, 0.5), "route_rejoin")])
def test_nearpoint_and_rejoin_continue_object_yaw_after_alignment(xyz, phase):
    control = _controller()
    _step(control, _pose(POINTS[0]))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    _confirm(control, gate, np.array(xyz, dtype=float), stamp=0.05)
    heading = control.cfg.map_frame.heading(np.asarray(TARGET) - xyz)
    pose = _pose(xyz, heading - math.radians(2), 0.2)
    command = _step(control, pose)
    output = _pcmd(gate, command, pose)
    assert gate.phase == phase
    assert output[0] or output[1]
    assert output[2] < 0 and output[3] == 0
    assert command.look_at_pole is None


def test_close_direct_resume_still_aligns_to_object_before_position_recovery():
    control = _controller()
    control.start_after_nearest_waypoint(np.array([9.95, 0, 0]), target_index=1)
    gate = rpf.YawAlignedPcmdController(control.cfg)
    pose = _pose((9.95, 0, 0), 0.0, 0.1)
    command = _step(control, pose)
    assert command.guidance_goal is None
    output = _pcmd(gate, command, pose)
    assert gate.phase == "turn" and output[:2] == (0, 0)
    assert output[2] < 0 and output[3] == 0


def test_vertical_movement_is_unchanged_while_object_yaw_tracks():
    points = ((0, 0, 0), (10, -2, 0), (20, -2, 0))
    control = _controller(points=points)
    _step(control, _pose(points[0]))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    _confirm(control, gate, np.array([2.0, -0.4, 0]), stamp=0.05)
    xyz = np.array([2.0, -0.4, 0])
    heading = control.cfg.map_frame.heading(np.asarray(TARGET) - xyz)
    pose = _pose(xyz, heading - math.radians(2), 0.2)
    command = _step(control, pose)
    output = _pcmd(gate, command, pose)
    normal = rpf.YawAlignedPcmdController(control.cfg)
    normal.target_key = 1
    normal.aligned_for_translation = True
    baseline = _pcmd(normal, replace(command, look_at_target=None), pose)
    assert output[:2] == baseline[:2] and output[3] == baseline[3]
    assert output[2] < 0 and output[3] > 0


def test_corner_and_disjoint_spans_enter_exit_without_changing_goals():
    points = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (2, 0, 1), (2, 0, 2))
    targets = (TARGET, None, TARGET, None)
    control = _controller(targets, points=points)
    # WP1 join is direct, not the first configured outbound segment.
    boot = _step(control, _pose((-1, 0, 0), stamp=0.0))
    assert control.target_index == 0 and boot.look_at_target is None
    assert boot.yaw_target == pytest.approx(0.0)
    for index, point in enumerate(points[:-1]):
        command = _step(control, _pose(point, stamp=0.1 + index * 0.1))
        assert control.target_index == index + 1
        np.testing.assert_allclose(command.goal, points[index + 1])
        if targets[index] is None:
            assert command.look_at_target is None
            assert command.yaw_target == pytest.approx(
                control.cfg.map_frame.heading(np.asarray(points[index + 1]) - point)
            )
        else:
            np.testing.assert_allclose(command.look_at_target, TARGET)
            assert command.yaw_target == pytest.approx(
                control.cfg.map_frame.heading(np.asarray(TARGET) - point)
            )


def test_after_an_aimed_leg_the_nose_turns_to_the_next_waypoint_before_translating():
    """Leaving the object-aimed leg: turn in place to the next waypoint, then fly to it."""
    points = ((0, 0, 0), (10, 0, 0), (10, 0, 10))
    control = _controller((TARGET, None), points=points)
    gate = rpf.YawAlignedPcmdController(control.cfg)
    _step(control, _pose(points[0]))  # the first leg (waypoint 1 -> 2) is the aimed one
    waypoint = np.asarray(points[1], float)
    toward_object = control.cfg.map_frame.heading(np.asarray(TARGET) - waypoint)
    next_bearing = control.cfg.map_frame.heading(np.asarray(points[2]) - waypoint)
    assert abs(rpf.wrap_angle(next_bearing - toward_object)) > math.radians(30)

    yaw, turns, moving = toward_object, [], None
    for tick in range(160):
        pose = _pose(points[1], yaw, stamp=0.1 + tick * 0.05)
        command = _step(control, pose)
        pcmd = _pcmd(gate, command, pose, control.target_index)
        assert control.target_index == 2 and command.look_at_target is None
        assert command.yaw_target == pytest.approx(next_bearing)  # no longer the object
        if pcmd[0] or pcmd[1]:
            moving = (pcmd, yaw)
            break
        assert pcmd[3] == 0  # a turn never climbs
        turns.append(pcmd[2])
        yaw += -pcmd[2] / 100.0 * 0.2  # crude yaw plant: positive PCMD is clockwise
    assert any(turns), "must turn before translating"
    assert moving is not None, "must start translating once aligned"
    assert abs(rpf.wrap_angle(next_bearing - moving[1])) < math.radians(control.cfg.yaw_tolerance_deg) + 1e-9


def test_wraparound_uses_shortest_yaw_direction():
    target = (-5.0, 0.0, 0.05)
    control = _controller((target, None))
    _step(control, _pose(POINTS[0]))
    pose = _pose((0.1, 0, 0), math.radians(-179), 0.1)
    command = _step(control, pose)
    output = rpf.command_to_body_percent(command, pose, config=control.cfg, require_yaw_alignment=False)
    assert command.yaw_target > math.radians(179)
    assert 0 < output[2] < control.cfg.max_yaw_pcmd
    assert output[0] or output[1]


def test_degenerate_horizontal_ray_retains_latest_defined_heading():
    target = (4.0, -20.0, 0.0)
    control = _controller((target, None))
    _step(control, _pose(POINTS[0]))
    before = _step(control, _pose((5, 0, 0), yaw=0.3, stamp=0.1))
    assert abs(before.yaw_target) == pytest.approx(math.pi)
    for index, x in enumerate((4.0, 4.0 + 1e-11)):
        pose = _pose((x, 0, 0), yaw=before.yaw_target, stamp=0.2 + index * 0.1)
        command = _step(control, pose)
        assert command.yaw_target == pytest.approx(before.yaw_target)
        assert rpf.command_to_body_percent(
            command, pose, config=control.cfg, require_yaw_alignment=False,
        )[2] == 0


def test_first_degenerate_ray_holds_current_pose_heading():
    control = _controller(((0, -5, 0), None))
    pose = _pose(POINTS[0], yaw=1.1)
    command = _step(control, pose)
    assert control.target_index == 1 and command.yaw_target == pytest.approx(1.1)
    assert rpf.command_to_body_percent(command, pose, config=control.cfg)[2] == 0


def test_measured_basis_controls_horizontal_bearing_not_raw_xz():
    frame = rpf.MapFrame.from_gravity((0.0, math.sqrt(0.5), math.sqrt(0.5)))
    target = frame.east * 4 + frame.north * 5 + frame.up * 100
    points = (np.zeros(3), frame.east * 10, frame.east * 20)
    control = _controller((tuple(target), None), points=points, map_frame=frame)
    _step(control, _pose(points[0]))
    pose = _pose(frame.east * 2, yaw=0.0, stamp=0.1)
    command = _step(control, pose)
    assert command.yaw_target == pytest.approx(math.atan2(5, 2))
    assert command.yaw_target != pytest.approx(math.atan2(target[2], target[0] - 2))
    output = rpf.command_to_body_percent(command, pose, config=control.cfg, require_yaw_alignment=False)
    assert output[2] < 0 and output[3] == 0


@pytest.mark.parametrize("targets", [(TARGET, None), (None, TARGET), (TARGET, TARGET)])
def test_return_legs_never_aim_at_object_and_keep_normal_heading(targets):
    control = _controller(targets, return_to_start=True)
    assert control.cfg.segment_look_at_points == targets + (None, None)
    for index, point in enumerate(POINTS):
        command = _step(control, _pose(point, stamp=index * 0.1))
    assert control.target_index == 3
    np.testing.assert_allclose(command.goal, POINTS[1])
    for index, start in enumerate((POINTS[2], POINTS[1])):
        pose = _pose(start, stamp=0.3 + index * 0.1)
        command = _step(control, pose)
        assert control.target_index == 3 + index
        np.testing.assert_allclose(command.goal, POINTS[1 - index])
        assert command.look_at_target is None
        assert abs(command.yaw_target) == pytest.approx(math.pi)


@pytest.mark.parametrize("bad", [(), (TARGET,), (TARGET, None, None), "bad", (None, [1, 2]), (None, [[1, 2, 3]]), (None, [1, 2, float("nan")]), (None, [1, 2, float("inf")])])
def test_controller_rejects_malformed_target_table_before_return_expansion(bad):
    with pytest.raises(ValueError, match="segment_look_at_points"):
        _controller(bad, return_to_start=True)


@pytest.mark.parametrize("waypoint_frame", ["glomap", "aligned"])
def test_config_path_and_snapshot_carry_resolved_measured_targets_and_clear_old_table(tmp_path, waypoint_frame):
    frame = rpf.MapFrame.from_gravity((0.0, math.sqrt(0.5), math.sqrt(0.5)))
    path = tmp_path / "route.json"
    payload = {
        "schema": "sfm-flight-route/v1", "site_id": "field-a",
        "coordinate_frame_id": "glomap-a", "frame": waypoint_frame, "align_source": "measured",
        "units": "map", "purpose": "flight", "closed": False,
        "waypoints": [[0, 0, 0], [10, 0, 0], [20, 0, 0]],
        "object_boxes": [{"center": [4, 5, 6], "size": [1, 2, 3]}],
        "segment_look_at": [0, None],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    base = _config(((999, 0, 0), (999, 0, 0)), map_frame=frame)
    config = rpf.config_for_route(path, base)
    expected = tuple(frame.east * 4 + frame.north * 5 + frame.up * 6)
    np.testing.assert_allclose(config.segment_look_at_points[0], expected)
    assert config.segment_look_at_points[1] is None
    snapshot = rpf.capture_mission_route_snapshot(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_site_id="field-a", expected_coordinate_frame_id="glomap-a", map_frame=frame,
    )
    # config_for_route must consume captured bytes, not reread the path.
    path.write_text("unreadable replacement", encoding="utf-8")
    assert rpf.config_for_route(snapshot, base).segment_look_at_points == config.segment_look_at_points
    assert rpf.config_for_route(replace(snapshot, segment_look_at_points=None), config).segment_look_at_points is None
    payload.pop("segment_look_at")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert rpf.config_for_route(path, config).segment_look_at_points is None


def test_waypoint_actions_keep_priority_and_new_leg_only_starts_after_actions():
    actions = ((), (WaypointAction("yaw", 30.0), WaypointAction("hover", 1.0)), ())
    control = _controller((TARGET, TARGET), waypoint_actions=actions)
    _step(control, _pose(POINTS[0]))
    pose = _pose(POINTS[1], yaw=0.0, stamp=0.1)
    command = _step(control, pose)
    assert command.action == "WAYPOINT_ACTION" and control.target_index == 1
    assert command.look_at_target is None and command.look_at_pole is None
    assert command.yaw_target == pytest.approx(math.radians(-30))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    assert _pcmd(gate, command, pose)[2] > 0
    assert gate.phase == "waypoint_action_yaw"
    for index in range(1, 30):
        pose = _pose(POINTS[1], math.radians(-30), 0.1 + index * 0.1)
        command = _step(control, pose)
        if command.action == "FOLLOW":
            break
        assert command.look_at_target is None
    assert command.action == "FOLLOW" and control.target_index == 2
    np.testing.assert_allclose(command.look_at_target, TARGET)


def test_inspection_keeps_existing_hold_semantics_and_priority():
    cfg = _config(inspect_waypoints=(2,))
    pole = {"id": "inspection", "center": [10, 0, 1], "base": [10, 0, 1], "top": [10, -1, 1]}
    control = rpf.RouteAutoController([np.array(p) for p in POINTS], poles=[pole], config=cfg)
    _step(control, _pose(POINTS[0]))
    command = _step(control, _pose(POINTS[1], stamp=0.1))
    assert command.action == "INSPECT" and command.look_at_target is None
    assert command.look_at_pole is not None
    np.testing.assert_allclose(command.vel_map, np.zeros(3))


def test_stale_pose_final_hold_and_manual_resume_do_not_bypass_existing_gates():
    control = _controller((TARGET, TARGET))
    _step(control, _pose(POINTS[0]))
    gate = rpf.YawAlignedPcmdController(control.cfg)
    _confirm(control, gate, np.array([2.0, 0, 0]), stamp=0.05)
    stale = _pose((2, 0, 0), stamp=0.1)
    hover = control.step(stale, now=1.0)
    assert hover.action == "HOVER" and hover.look_at_target is None
    assert gate.update(hover, stale, 1.0, target_key=1) == (0, 0, 0, 0)
    assert not gate.aligned_for_translation and gate.target_key is None
    # Resume only an explicitly retained target; proximity never skips points.
    assert control.start_after_nearest_waypoint(np.array(POINTS[2]), target_index=1) == 1
    resume_pose = _pose((2, 0, 0), yaw=0, stamp=1.1)
    resume = _step(control, resume_pose)
    assert control.target_index == 1 and resume.guidance_goal is None
    assert _pcmd(gate, resume, resume_pose)[2] < 0
    assert gate.phase == "turn"
    _step(control, _pose(POINTS[1], stamp=1.2))
    final_pose = _pose(POINTS[2], yaw=0.3, stamp=1.3)
    final = _step(control, final_pose)
    assert final.action == "FINAL_HOLD" and final.look_at_target is None
    assert _pcmd(gate, final, final_pose, target=2)[2] == 0
    assert gate.phase == "final_centering"
    assert not final.should_land
