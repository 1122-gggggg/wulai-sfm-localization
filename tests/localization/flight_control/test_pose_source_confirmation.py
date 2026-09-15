"""No hardware: drive production control across the audited TRACK jump."""

import numpy as np
import pytest

import path_follow_flight as pff
import real_path_follow_controller as rpf
from pose_source_confirmation import PoseSourceConfirmation


RADIUS = 0.02221739130434783
JUMP = 0.1132771920737308


def test_track_jump_onto_waypoint_is_used_without_source_confirmation():
    waypoints = [np.array([JUMP, 0, 0]), np.array([1, 0, 0])]
    ctrl = rpf.RouteAutoController(
        waypoints,
        config=rpf.ControlConfig(
            inspect_waypoints=(),
            waypoint_arrive_radius=RADIUS,
            waypoint_arrive_confirm_frames=1,
        ),
    )
    state = {"now": 10.0, "stamp": 10.0, "x": 0.0, "weak": False}
    rows = []
    runner = pff._FlightLoopRunner(
        pff.LoopHooks(
            get_pose=lambda: rpf.Pose(state["x"], 0, 0, 0, stamp=state["stamp"]),
            olympe_yaw=lambda: None,
            send_pcmd=lambda *args: None,
            pose_confidence=lambda: 100,
            pose_is_weak=lambda: state["weak"],
            max_weak_pose_age_s=0.5,
            land_on_localization_loss=False,
            ground_speed=lambda: (0.0, state["now"]),
            now=lambda: state["now"],
            log_tick=rows.append,
        ),
        ctrl,
        waypoints,
        yaw_sign=1,
        verbose=False,
        enforce_weak_pose_gate=False,
    )

    def tick(x, *, stamp=None, weak=False):
        state["now"] += 0.05
        state.update(x=x, stamp=state["now"] if stamp is None else stamp, weak=weak)
        runner._tick(state["now"], "AUTO", {})
        return rows[-1]

    for _ in range(5):
        tick(0.0)
    jumped = tick(JUMP)
    assert jumped.get("pose_source_confirming") is not True
    assert runner.last_good is not None
    assert runner.last_good.x == pytest.approx(JUMP)
    tick(JUMP)
    assert ctrl.target_index == 1


@pytest.mark.parametrize("source_change", [False, True])
def test_candidate_confirmation_requires_distinct_consistent_strong_captures(source_change):
    gate = PoseSourceConfirmation()
    origin = np.zeros(3)
    candidate = np.array([0 if source_change else JUMP, 0, 0])
    source = "reloc" if source_change else "track"
    assert gate.accept(origin, 10.0, reliable=True, radius=RADIUS, source="track")
    assert not gate.accept(candidate, 10.1, reliable=True, radius=RADIUS, source=source)
    assert not gate.accept(candidate, 10.1, reliable=True, radius=RADIUS, source=source)
    assert not gate.accept(candidate, 10.05, reliable=True, radius=RADIUS, source=source)
    assert not gate.accept(candidate, 10.2, reliable=False, radius=RADIUS, source=source)
    assert not gate.accept(candidate, 10.3, reliable=True, radius=RADIUS, source=source)
    assert gate.accept(candidate, 10.4, reliable=True, radius=RADIUS, source=source)


def test_alternating_candidate_clusters_never_confirm():
    gate = PoseSourceConfirmation()
    assert gate.accept(np.zeros(3), 10.0, reliable=True, radius=RADIUS)
    for i in range(1, 10):
        assert not gate.accept(
            np.array([JUMP * (1 + i % 2), 0, 0]), 10 + i / 10, reliable=True, radius=RADIUS
        )
    np.testing.assert_array_equal(gate.accepted, np.zeros(3))


def test_ui_pending_flag_does_not_block_holdover_or_zero_pcmd():
    waypoints = [np.array([JUMP, 0, 0]), np.array([1, 0, 0])]
    ctrl = rpf.RouteAutoController(waypoints, config=rpf.ControlConfig(inspect_waypoints=()))
    state = {"now": 10.0, "pending": False}
    rows = []
    runner = pff._FlightLoopRunner(
        pff.LoopHooks(
            get_pose=lambda: None if state["pending"] else rpf.Pose(0, 0, 0, 0, stamp=state["now"]),
            pose_source_pending=lambda: state["pending"],
            olympe_yaw=lambda: None,
            send_pcmd=lambda *args: None,
            now=lambda: state["now"],
            log_tick=rows.append,
        ),
        ctrl,
        waypoints,
        yaw_sign=1,
        verbose=False,
        enforce_weak_pose_gate=False,
    )
    runner._tick(state["now"], "AUTO", {})
    assert runner.last_good is not None
    arrival = ctrl._arrive_frames
    state.update(now=10.05, pending=True)
    runner._tick(state["now"], "AUTO", {})
    assert rows[-1].get("pose_source_confirming") is not True
    assert ctrl._arrive_frames == arrival


def test_reseed_confirmation_hook_reaches_the_controller_pose():
    waypoints = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    ctrl = rpf.RouteAutoController(waypoints, config=rpf.ControlConfig(inspect_waypoints=()))
    state = {"now": 10.0, "weak": False, "reseed": False}
    rows, seen = [], []
    step = ctrl.step
    ctrl.step = lambda pose, now=None: seen.append(pose) or step(pose, now)
    runner = pff._FlightLoopRunner(
        pff.LoopHooks(
            get_pose=lambda: rpf.Pose(0, 0, 0, 0, stamp=state["now"]),
            olympe_yaw=lambda: None,
            send_pcmd=lambda *args: None,
            pose_confidence=lambda: 100,
            pose_is_weak=lambda: state["weak"],
            pose_reseed_confirming=lambda: state["reseed"],
            max_weak_pose_age_s=0.5,
            land_on_localization_loss=False,
            ground_speed=lambda: (0.0, state["now"]),
            now=lambda: state["now"],
            log_tick=rows.append,
        ),
        ctrl,
        waypoints,
        yaw_sign=1,
        verbose=False,
        enforce_weak_pose_gate=False,
    )
    for _ in range(5):
        state["now"] += 0.05
        runner._tick(state["now"], "AUTO", {})
    assert seen and seen[-1].map_confirmed and not seen[-1].reseed_confirming
    state.update(now=state["now"] + 0.05, weak=True, reseed=True)
    runner._tick(state["now"], "AUTO", {})
    assert not seen[-1].map_confirmed and seen[-1].reseed_confirming
    assert rows[-1]["reseed_confirming"] is True
