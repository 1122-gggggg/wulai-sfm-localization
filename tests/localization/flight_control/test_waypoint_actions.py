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


FOUR = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
HOVER_1 = WaypointAction("hover", 1.0)


def _return_controller(outbound, back, **overrides):
    config = _config(outbound, return_waypoint_actions=back, return_to_start=True, **overrides)
    return rpf.RouteAutoController([np.array(p, float) for p in FOUR], config=config)


def _fly_every_waypoint(control):
    """Hold at each (expanded) waypoint until it is retired; return where actions ran."""
    acted, now = [], 0.0
    for index, point in enumerate(control.wp):
        for _ in range(80):
            now += 0.1
            command = control.step(rpf.Pose(*point, 0.0, stamp=now), now=now)
            if command.action == "WAYPOINT_ACTION" and (not acted or acted[-1] != index):
                acted.append(index)
            if command.should_land or control.target_index > index:
                break
    return acted


def test_return_pass_runs_its_own_actions_at_the_mirrored_waypoints():
    outbound = ((), (HOVER_1,), (), ())
    back = ((), (WaypointAction("hover", 2.0),), (HOVER_1,), ())
    control = _return_controller(outbound, back)
    # 4 authored waypoints fly out 0,1,2,3 then back 2,1,0: expanded indices 4,5,6.
    assert len(control.wp) == 7
    assert control.waypoint_actions == {
        1: (HOVER_1,), 4: (HOVER_1,), 5: (WaypointAction("hover", 2.0),),
    }
    assert _fly_every_waypoint(control) == [1, 4, 5]


def test_outbound_and_return_actions_do_not_leak_into_each_other():
    only_back = _return_controller(None, ((), (HOVER_1,), (), ()))
    assert sorted(only_back.waypoint_actions) == [5]
    only_out = _return_controller(((), (HOVER_1,), (), ()), None)
    assert sorted(only_out.waypoint_actions) == [1]


@pytest.mark.parametrize("bad_index", [0, 3])
def test_return_actions_cannot_sit_on_the_landing_or_turnaround_waypoint(bad_index):
    back = [(), (), (), ()]
    back[bad_index] = (HOVER_1,)
    with pytest.raises(ValueError, match="cannot carry return actions"):
        _return_controller(None, tuple(back))
    data = {"waypoints": [list(p) for p in FOUR], "frame": "glomap", "units": "map",
            "return_waypoint_actions": [[{"type": "hover", "seconds": 1}] if i == bad_index else []
                                        for i in range(4)]}
    with pytest.raises(ValueError, match="must be empty"):
        RouteDocument.from_data(data)


def test_return_actions_need_a_return_pass_and_a_matching_length():
    back = ((), (HOVER_1,), (), ())
    points = [np.array(p, float) for p in FOUR]
    with pytest.raises(ValueError, match="require return_to_start"):
        rpf.RouteAutoController(points, config=_config(None, return_waypoint_actions=back))
    with pytest.raises(ValueError, match="entries for 4 waypoints"):
        _return_controller(None, ((), (HOVER_1,), ()))


def test_route_file_return_actions_reach_the_controller_and_are_replaced_per_route(tmp_path):
    def write(name, **extra):
        path = tmp_path / name
        path.write_text(json.dumps({
            "waypoints": [list(p) for p in FOUR], "frame": "glomap", "units": "map", **extra,
        }), encoding="utf-8")
        return path

    with_back = write("a.json", return_waypoint_actions=[[], [{"type": "hover", "seconds": 1}], [], []])
    base = rpf.ControlConfig(inspect_waypoints=(), return_to_start=True)
    config = rpf.config_for_route(with_back, base)
    assert config.return_waypoint_actions == ((), (HOVER_1,), (), ())
    control = rpf.RouteAutoController([np.array(p, float) for p in FOUR], config=config)
    assert sorted(control.waypoint_actions) == [5]
    # A route without return actions must not inherit the previous route's.
    plain = rpf.config_for_route(write("b.json"), config)
    assert plain.return_waypoint_actions is None


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


@pytest.mark.parametrize("pitch", [-90, -20, 0, 45, 90])
def test_absolute_camera_pitch_route_contract(pitch):
    route = RouteDocument.from_data({
        "waypoints": [[0, 0, 0], [1, 0, 0]], "frame": "glomap", "units": "map",
        "waypoint_actions": [[{"type": "gimbal_pitch", "pitch_deg": pitch}], []],
    })
    assert route.waypoint_actions[0][0].to_json() == {
        "type": "gimbal_pitch", "pitch_deg": float(pitch),
    }


@pytest.mark.parametrize("pitch", [True, "0", None, -91, 91, float("nan"), float("inf")])
def test_absolute_camera_pitch_rejects_invalid_values(pitch):
    with pytest.raises(ValueError):
        RouteDocument.from_data({
            "waypoints": [[0, 0, 0], [1, 0, 0]],
            "waypoint_actions": [[{"type": "gimbal_pitch", "pitch_deg": pitch}], []],
        })
    with pytest.raises(ValueError):
        _config(((WaypointAction("gimbal_pitch", pitch),), ()))


def _camera_controller(actions, request, read):
    control = _controller(((), tuple(actions), ()))
    control.request_gimbal_pitch = request
    control.read_gimbal_pitch = read
    control.step(rpf.Pose(0, 0, 0, 0.0, stamp=0.0), now=0.0)
    return control


def _camera_tick(control, now):
    return control.step(rpf.Pose(1, 0, 0, 0.3, stamp=now), now=now)


def test_adjacent_camera_actions_dispatch_once_and_never_reuse_feedback():
    sent = []
    sample = [-20.0, 0.0]
    control = _camera_controller(
        [WaypointAction("gimbal_pitch", -20), WaypointAction("gimbal_pitch", -20)],
        lambda pitch: sent.append(pitch) is None, lambda: tuple(sample),
    )
    command = _camera_tick(control, 0.1)
    assert command.yaw_target == pytest.approx(0.3)
    np.testing.assert_array_equal(command.goal, control.wp[1])
    assert sent == [-20]
    sample[1] = 0.2
    _camera_tick(control, 0.2)
    assert _camera_tick(control, 0.3).waypoint_action["step"] == 1  # duplicate
    sample[1] = 0.4
    command = _camera_tick(control, 0.4)
    assert command.waypoint_action["step"] == 2 and sent == [-20, -20]
    assert _camera_tick(control, 0.5).waypoint_action["step"] == 2
    sample[1] = 0.6
    _camera_tick(control, 0.6)
    sample[1] = 0.7
    assert _camera_tick(control, 0.7).action == "FOLLOW"
    assert control.target_index == 2
    assert [result["state"] for result in control.action_results] == ["confirmed", "confirmed"]
    assert sent == [-20, -20]


def test_camera_confirmation_resets_on_stale_or_off_target_readback():
    sample = [0.0, 0.0]
    control = _camera_controller([WaypointAction("gimbal_pitch", 0)], lambda pitch: True,
                                 lambda: tuple(sample))
    _camera_tick(control, 0.1)
    sample[1] = 0.2
    _camera_tick(control, 0.2)
    _camera_tick(control, 1.3)  # stale feedback cannot retain a confirmation
    sample[:] = [0.0, 1.4]
    assert _camera_tick(control, 1.4).action == "WAYPOINT_ACTION"
    sample[:] = [10.0, 1.5]
    _camera_tick(control, 1.5)
    sample[:] = [0.0, 1.6]
    assert _camera_tick(control, 1.6).action == "WAYPOINT_ACTION"
    sample[1] = 1.7
    assert _camera_tick(control, 1.7).action == "FOLLOW"


@pytest.mark.parametrize("mode,outcome", [
    ("rejected", "dispatch_failed"), ("missing", "dispatch_failed"),
    ("read_error", "readback_failed"), ("no_feedback", "timeout"),
])
def test_camera_failure_skips_remaining_actions_without_claiming_success(mode, outcome):
    sent = []

    def read():
        if mode == "read_error":
            raise RuntimeError("readback unavailable")
        return None

    control = _camera_controller(
        [WaypointAction("gimbal_pitch", 40), WaypointAction("hover", 60)],
        None if mode == "missing" else lambda pitch: sent.append(pitch) is None and mode != "rejected",
        read,
    )
    for index in range(1, 110):
        command = _camera_tick(control, index * 0.1)
        if command.action != "WAYPOINT_ACTION":
            break
    assert control.target_index == 2 and command.action == "FOLLOW"
    assert control.action_results[0]["state"] == outcome
    assert len(sent) == (0 if mode == "missing" else 1)


def test_camera_pause_and_same_target_reset_do_not_replay_dispatch_or_burn_budget():
    sent = []
    sample = [0.0, 0.0]
    control = _camera_controller([WaypointAction("gimbal_pitch", 0)],
                                 lambda pitch: sent.append(pitch) is None, lambda: tuple(sample))
    _camera_tick(control, 0.1)
    control.suspend_waypoint_action_clock()
    command = _camera_tick(control, 100.0)
    assert command.waypoint_action["elapsed_s"] == 0
    control.start_after_nearest_waypoint(np.array([1., 0, 0]), target_index=1)
    sample[1] = 100.1
    _camera_tick(control, 100.1)
    sample[1] = 100.2
    assert _camera_tick(control, 100.2).action == "FOLLOW"
    assert sent == [0]


def test_camera_actions_preserve_yaw_then_hover_order_and_landing_restriction():
    with pytest.raises(ValueError, match="landing waypoint"):
        _controller(((), (), (WaypointAction("gimbal_pitch", 0),)))
    sample = [-45., 0.]
    control = _camera_controller(
        [WaypointAction("yaw", 30), WaypointAction("gimbal_pitch", -45), WaypointAction("hover", 1)],
        lambda pitch: True, lambda: tuple(sample),
    )
    _camera_tick(control, 0.1)
    turned = 0.3 - math.radians(30)
    for now in (0.2, 0.3):
        command = control.step(rpf.Pose(1, 0, 0, turned, stamp=now), now=now)
    assert command.waypoint_action["type"] == "gimbal_pitch"
    for now in (0.4, 0.5):
        sample[1] = now
        command = control.step(rpf.Pose(1, 0, 0, turned, stamp=now), now=now)
    assert command.waypoint_action["type"] == "hover"
    assert command.yaw_target == pytest.approx(turned)


def test_manual_session_resume_retains_confirmed_camera_steps_but_new_mission_replays():
    sent = []
    sample = [0., 0.]
    actions = [WaypointAction("gimbal_pitch", 0), WaypointAction("hover", 60)]
    control = _camera_controller(actions, lambda pitch: sent.append(pitch) is None,
                                 lambda: tuple(sample))
    _camera_tick(control, 0.1)
    for now in (0.2, 0.3):
        sample[1] = now
        command = _camera_tick(control, now)
    assert command.waypoint_action["type"] == "hover"
    checkpoint = control.camera_action_checkpoint()
    assert checkpoint == frozenset({(1, 0)})
    resumed = rpf.RouteAutoController(
        list(control.wp), config=control.cfg,
        request_gimbal_pitch=lambda pitch: sent.append(pitch) is None,
        read_gimbal_pitch=lambda: tuple(sample),
        completed_camera_actions=checkpoint,
    )
    resumed.start_after_nearest_waypoint(np.zeros(3), target_index=1)
    # A retained camera checkpoint never bypasses the waypoint arrival gate.
    assert resumed.step(rpf.Pose(0, 0, 0, 0., stamp=1.), now=1.).action == "FOLLOW"
    assert len(sent) == 1
    assert _camera_tick(resumed, 1.1).waypoint_action["type"] == "hover"
    assert resumed.action_results[0]["state"] == "not_replayed"
    assert len(sent) == 1
    fresh = _camera_controller(actions, lambda pitch: sent.append(pitch) is None,
                               lambda: tuple(sample))
    assert _camera_tick(fresh, 2.).waypoint_action["type"] == "gimbal_pitch"
    assert sent == [0, 0]
