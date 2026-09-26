"""Operator-authored waypoint actions: turn / hover at a reached waypoint."""

import json
import math

import numpy as np
import pytest

import real_path_follow_controller as rpf
from route_domain import RouteDocument, WaypointAction


def _config(actions, **overrides):
    values = dict(
        inspect_waypoints=(),
        waypoint_arrive_confirm_frames=1,
        yaw_alignment_confirmation_updates=2,
        waypoint_actions=actions,
    )
    values.update(overrides)
    return rpf.ControlConfig(**values)


def _controller(actions, **overrides):
    points = [np.array(point, float) for point in ((0, 0, 0), (1, 0, 0), (1, 0, 1))]
    return rpf.RouteAutoController(points, config=_config(actions, **overrides))


YAW_RIGHT_30_THEN_HOVER_2 = ((), (WaypointAction("yaw", 30.0), WaypointAction("hover", 2.0)), ())


def test_route_contract_parses_and_rejects_actions():
    base = {"waypoints": [[0, 0, 0], [1, 0, 0]], "frame": "glomap", "units": "map"}
    route = RouteDocument.from_data(
        {**base, "waypoint_actions": [[], [{"type": "yaw", "degrees": -45}, {"type": "hover", "seconds": 3}]]}
    )
    assert route.waypoint_actions == ((), (WaypointAction("yaw", -45.0), WaypointAction("hover", 3.0)))
    assert route.waypoint_actions[1][0].to_json() == {"type": "yaw", "degrees": -45.0}
    for bad in (
        [[]],  # one list per waypoint
        [[], [{"type": "spin", "degrees": 10}]],
        [[], [{"type": "yaw", "degrees": 0}]],
        [[], [{"type": "yaw", "degrees": 181}]],
        [[], [{"type": "hover", "seconds": 61}]],
        [[], [{"type": "hover", "seconds": 1, "degrees": 5}]],
        [[], [{"type": "yaw", "degrees": True}]],
    ):
        with pytest.raises(ValueError):
            RouteDocument.from_data({**base, "waypoint_actions": bad})


def test_actions_ride_the_route_file_into_the_controller_outbound_only(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(json.dumps({
        "waypoints": [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "frame": "glomap", "units": "map",
        "waypoint_actions": [[{"type": "hover", "seconds": 1}], [], [{"type": "yaw", "degrees": 90}]],
    }), encoding="utf-8")
    config = rpf.config_for_route(path, rpf.ControlConfig(inspect_waypoints=(), return_to_start=True))
    control = rpf.RouteAutoController(
        [np.array(p, float) for p in ((0, 0, 0), (1, 0, 0), (2, 0, 0))], config=config
    )
    # The turnaround point keeps its action; the return pass carries none.
    assert sorted(control.waypoint_actions) == [0, 2]
    assert len(control.cfg.waypoint_actions) == len(control.wp) == 5


def test_landing_waypoint_cannot_carry_actions():
    with pytest.raises(ValueError, match="landing waypoint"):
        _controller(((), (), (WaypointAction("hover", 1.0),)))


def test_yaw_then_hover_hold_the_waypoint_before_the_next_leg():
    control = _controller(YAW_RIGHT_30_THEN_HOVER_2)
    control.step(rpf.Pose(0, 0, 0, 0.0, stamp=0.0), now=0.0)
    assert control.target_index == 1

    arrive_heading = math.radians(10.0)
    command = control.step(rpf.Pose(1, 0, 0, arrive_heading, stamp=0.1), now=0.1)
    assert command.action == "WAYPOINT_ACTION"
    assert control.target_index == 1
    np.testing.assert_allclose(command.goal, control.wp[1])
    # Right is clockwise from above: the CCW map heading drops by 30 degrees.
    assert math.degrees(command.yaw_target) == pytest.approx(-20.0)
    assert command.waypoint_action["type"] == "yaw" and command.waypoint_action["step"] == 1

    # Still 30 degrees off: not done, however many samples arrive.
    command = control.step(rpf.Pose(1, 0, 0, arrive_heading, stamp=0.2), now=0.2)
    assert command.action == "WAYPOINT_ACTION" and command.waypoint_action["type"] == "yaw"
    turned = math.radians(-18.5)
    control.step(rpf.Pose(1, 0, 0, turned, stamp=0.3), now=0.3)
    command = control.step(rpf.Pose(1, 0, 0, turned, stamp=0.4), now=0.4)
    assert command.waypoint_action["type"] == "hover"
    # The hover keeps the new heading rather than snapping back.
    assert math.degrees(command.yaw_target) == pytest.approx(-20.0)

    for step in range(1, 40):
        now = 0.4 + 0.1 * step
        command = control.step(rpf.Pose(1, 0, 0, turned, stamp=now), now=now)
        if command.action != "WAYPOINT_ACTION":
            break
    assert now == pytest.approx(2.4)
    assert control.target_index == 2 and command.action == "FOLLOW"


def test_hover_time_does_not_advance_across_a_pose_outage():
    control = _controller(((), (WaypointAction("hover", 1.0),), ()))
    control.step(rpf.Pose(0, 0, 0, 0.0, stamp=0.0), now=0.0)
    control.step(rpf.Pose(1, 0, 0, 0.0, stamp=0.1), now=0.1)
    control.step(rpf.Pose(1, 0, 0, 0.0, stamp=0.4), now=0.4)
    # 5 s without a fresh pose: HOVER, and the dwell clock does not jump.
    assert control.step(rpf.Pose(1, 0, 0, 0.0, stamp=0.4), now=5.4).action == "HOVER"
    command = control.step(rpf.Pose(1, 0, 0, 0.0, stamp=5.5), now=5.5)
    assert command.action == "WAYPOINT_ACTION"
    assert command.waypoint_action["elapsed_s"] == pytest.approx(0.3)


def test_yaw_that_never_converges_times_out_and_continues():
    control = _controller(
        ((), (WaypointAction("yaw", -90.0), WaypointAction("hover", 5.0)), ()),
        yaw_alignment_timeout_s=1.0,
    )
    control.step(rpf.Pose(0, 0, 0, 0.0, stamp=0.0), now=0.0)
    command = None
    for step in range(1, 40):
        now = 0.1 * step
        command = control.step(rpf.Pose(1, 0, 0, 0.0, stamp=now), now=now)
        if command.action != "WAYPOINT_ACTION":
            break
    assert control.action_timeouts == [1]
    assert control.target_index == 2 and command.action == "FOLLOW"
    assert now < 1.5  # the remaining hover was skipped


def test_pcmd_turns_in_place_toward_the_action_heading():
    config = _config(None)
    pcmd = rpf.YawAlignedPcmdController(config)
    pose = rpf.Pose(1, 0, 0, 0.0, stamp=1.0)
    command = rpf.Command(
        "WAYPOINT_ACTION", np.zeros(3), math.radians(-30.0), np.array([1.0, 0, 0]), 0.0, 0.5,
    )
    roll, pitch, yaw, gaz = pcmd.update(command, pose, 1.0, target_key=1)
    assert yaw > 0 and gaz == 0  # clockwise for a right turn, height held
    assert pcmd.phase == "waypoint_action_yaw"
    aligned = rpf.Pose(1, 0, 0, math.radians(-28.0), stamp=1.1)
    assert pcmd.update(command, aligned, 1.1, target_key=1)[2] == 0
    assert pcmd.phase == "waypoint_action_hold"
