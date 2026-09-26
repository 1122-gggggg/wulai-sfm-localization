"""Control-only smoothing must not affect route/landing observation evidence."""

from dataclasses import replace

import numpy as np
import pytest

import path_follow_flight as pff
import real_path_follow_controller as rpf


def _runner(radius=0.02):
    cfg = rpf.ControlConfig(inspect_waypoints=(), waypoint_arrive_radius=radius)
    ctrl = rpf.RouteAutoController([np.zeros(3), np.array([1.0, 0.0, 0.0])], config=cfg)
    hooks = pff.LoopHooks(get_pose=lambda: None, olympe_yaw=lambda: None, send_pcmd=lambda *_: None)
    return pff._FlightLoopRunner(
        hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False, enforce_weak_pose_gate=False
    )


@pytest.mark.parametrize("radius", [0.01, 0.02, 0.05])
def test_moderate_handover_smooths_control_without_zeroing_wind_correction(radius):
    runner = _runner(radius)
    before = rpf.Pose(0.5, 0.0, 0.0, 0.0, stamp=1.0, map_confirmed=False)
    after = replace(before, x=0.56, stamp=1.05, map_confirmed=True)
    runner._note_pose_correction(before, 1.0, {})
    record = {}
    runner._note_pose_correction(after, 1.05, record)
    assert runner._cap_after_pose_correction((20, -20, 0, 0), 1.05, record) == (20, -20, 0, 0)
    command = runner.ctrl.step(after, 1.05)
    _, controlled = runner._predict_control_pose(command, after, 1.05, record)
    assert controlled.x == pytest.approx(before.x)
    assert after.x == 0.56 and controlled.stamp == after.stamp
    _, halfway = runner._predict_control_pose(command, after, 1.3, {})
    assert halfway.x == pytest.approx(0.53)
    _, settled = runner._predict_control_pose(command, after, 1.55, {})
    assert settled.x == pytest.approx(after.x)


def test_duplicate_frames_cannot_restart_handover_and_reset_clears_correction():
    runner = _runner()
    before = rpf.Pose(0.5, 0.0, 0.0, 0.0, stamp=1.0, map_confirmed=False)
    after = replace(before, x=0.56, stamp=1.05, map_confirmed=True)
    runner._note_pose_correction(before, 1.0, {})
    runner._note_pose_correction(after, 1.05, {})
    command = runner.ctrl.step(after, 1.05)
    for now in (1.1, 1.3, 1.5):
        runner._note_pose_correction(after, now, {})
    assert runner._predict_control_pose(command, after, 1.55, {})[1].x == pytest.approx(0.56)
    runner._note_pose_correction(replace(after, stamp=1.59, map_confirmed=False), 1.59, {})
    newer = replace(after, x=0.62, stamp=1.6)
    runner._note_pose_correction(newer, 1.6, {})
    assert np.linalg.norm(runner.control_smoother.correction(1.6)) > 0.0
    runner._reset_pcmd_controller()
    assert runner._predict_control_pose(command, newer, 1.6, {})[1].x == 0.62


def test_smoothing_cannot_put_raw_outside_position_inside_arrival_evidence():
    runner = _runner()
    runner.ctrl.target_index = 1
    before = rpf.Pose(0.995, 0.0, 0.0, 0.0, stamp=1.0, map_confirmed=False)
    after = replace(before, x=1.055, stamp=1.05, map_confirmed=True)
    runner._note_pose_correction(before, 1.0, {})
    runner._note_pose_correction(after, 1.05, {})
    command = runner.ctrl.step(after, 1.05)
    _, controlled = runner._predict_control_pose(command, after, 1.05, {})
    assert abs(controlled.x - 1.0) < 0.02
    assert abs(after.x - 1.0) > 0.02
    assert runner.ctrl._final_confirm_frames == 0
    assert runner.ctrl.state != "LANDING"


def test_metric_motion_is_not_mistaken_for_a_localization_correction():
    runner = _runner()
    runner.heading.offset = 0.0
    runner.control_state.scale_m_per_unit = 1.0
    runner.control_state.scale_stamp = 1.0
    for now in (1.0, 1.1, 1.2):
        runner.control_state.observe_telemetry((0.0, now), (0.0, 1.0, now), now)
        pose = rpf.Pose(now, 0.0, 0.0, 0.0, stamp=now)
        record = {}
        runner._note_pose_correction(pose, now, record)
        assert "pose_correction_u" not in record
        np.testing.assert_array_equal(runner.control_smoother.correction(now), np.zeros(3))


def test_fast_continuous_motion_without_metric_scale_does_not_start_handover():
    runner = _runner()
    for now in (1.0, 1.05, 1.10, 1.15):
        pose = rpf.Pose(now, 0.0, 0.0, 0.0, stamp=now)
        runner._note_pose_correction(pose, now, {})
        np.testing.assert_array_equal(runner.control_smoother.correction(now), np.zeros(3))


def test_reseed_flag_on_repeated_frame_is_remembered_until_new_confirmed_pose():
    runner = _runner()
    before = rpf.Pose(0.5, 0.0, 0.0, 0.0, stamp=1.0)
    runner._note_pose_correction(before, 1.0, {})
    runner._note_pose_correction(replace(before, reseed_confirming=True), 1.01, {})
    after = replace(before, x=0.56, stamp=1.05)
    runner._note_pose_correction(after, 1.05, {})
    np.testing.assert_allclose(runner.control_smoother.correction(1.05), [-0.06, 0.0, 0.0])
