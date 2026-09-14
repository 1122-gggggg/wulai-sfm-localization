"""The inertial bridge keeps a yaw reference when vision drops frames.

Fused attitude (~8 Hz) survives the fast yaw that blinds the visual loop
(log 2026-09-12: pose success 0.86 -> 0.64 once yaw rate passes ~5 deg/s, and
AUTO now aligns at 10 deg/s). The bridge rotates the last visual pose by the
fused yaw increment and holds its center. It never confirms a fix: the status
maps to WEAK_TRACK downstream, sharing the dead-reckon age budget.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from two_rate_tracker import (
    IMU_BRIDGE_MAX_SAMPLE_AGE_S,
    TwoRateTracker,
)

def _pose():
    # Camera center C=(1,2,3): cam_from_world stores t = -R @ C.
    return np.column_stack((np.eye(3), [-1.0, -2.0, -3.0]))


@pytest.fixture
def tracker():
    tracker = object.__new__(TwoRateTracker)
    tracker._reset_tracking_state()
    tracker.profile = SimpleNamespace(
        dead_reckon=SimpleNamespace(enabled=True, max_frames=300, max_age_s=10.0)
    )
    tracker.map_frame = SimpleNamespace(up=np.array([0.0, 0.0, 1.0]))
    return tracker


def seed(tracker, stamp=10.0, ordinal=1):
    gray = np.zeros((8, 8), dtype=np.uint8)
    tracker._step_bookkeeping(
        np.array([1.0, 2.0, 3.0]), _pose(), ordinal, gray, stamp, "FAST_TRACK"
    )


def sample(stamp, yaw_deg, roll_deg=0.0, pitch_deg=0.0):
    return SimpleNamespace(
        stamp=stamp,
        yaw=math.radians(yaw_deg),
        roll=math.radians(roll_deg),
        pitch=math.radians(pitch_deg),
    )


def test_bridge_rotates_last_pose_by_imu_yaw_increment(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(10.1, 10.0))
    pose, center, status, inliers, info = tracker._step_imu_bridge(10.1)
    assert status == "IMU_BRIDGE"
    assert inliers == 0
    # Clockwise-from-north +10 deg shrinks the CCW map heading by 10 deg:
    # cam_from_world stores the inverse: R_new = Rot_z(+10 deg).
    assert pose[0, 0] == pytest.approx(math.cos(math.radians(10.0)))
    assert pose[0, 1] == pytest.approx(-math.sin(math.radians(10.0)))
    assert pose[1, 0] == pytest.approx(math.sin(math.radians(10.0)))
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
    assert info["imu_bridge"] is True
    assert info["imu_sample_age_s"] == pytest.approx(0.0)
    assert info["imu_yaw_delta_deg"] == pytest.approx(-10.0)
    assert tracker._dead_reckon_age == 1


def test_bridge_rotates_camera_axes_in_the_measured_world_frame(tracker):
    from real_path_follow_controller import MapFrame, wrap_angle

    frame = MapFrame.from_gravity([0.15, 0.98, -0.12])
    tracker.map_frame = frame
    center = np.array([1.0, 2.0, 3.0])
    rotation = (
        tracker._rot_about_axis(np.array([1., 0., 0.]), 0.6)
        @ tracker._rot_about_axis(np.array([0., 1., 0.]), 1.0)
    )
    original = np.column_stack((rotation, -rotation @ center))
    tracker._step_bookkeeping(center, original, 1, np.zeros((8, 8), np.uint8), 10.0, "FAST_TRACK")
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(10.1, 10.0))
    pose, moved_center, *_ = tracker._step_imu_bridge(10.1)
    expected_forward = tracker._rot_about_axis(frame.up, math.radians(-10)) @ rotation[2]
    np.testing.assert_allclose(pose[2, :3], expected_forward, atol=1e-12)
    assert math.degrees(wrap_angle(frame.heading(pose[2, :3]) - frame.heading(rotation[2]))) == pytest.approx(-10)
    np.testing.assert_allclose(moved_center, center, atol=1e-12)
    np.testing.assert_allclose(pose[:, :3] @ pose[:, :3].T, np.eye(3), atol=1e-12)


def test_no_sample_no_bridge(tracker):
    seed(tracker)
    assert tracker._step_imu_bridge(10.1)[0] is None


def test_stale_reference_sample_no_bridge(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(9.0, 0.0))
    tracker.observe_fused_state(sample(10.1, 5.0))
    # The only sample at/below the last-pose stamp is 1.0 s old.
    assert tracker._step_imu_bridge(10.1)[0] is None


def test_sample_older_than_bridge_age_cap_no_bridge(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    pose, *_ = tracker._step_imu_bridge(10.0 + IMU_BRIDGE_MAX_SAMPLE_AGE_S + 0.1)
    assert pose is None


def test_tilted_flight_no_bridge(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0, roll_deg=40.0))
    tracker.observe_fused_state(sample(10.1, 5.0))
    assert tracker._step_imu_bridge(10.1)[0] is None


def test_yaw_glitch_no_bridge(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(10.1, 30.0))  # 300 deg/s: not airframe
    assert tracker._step_imu_bridge(10.1)[0] is None


def test_bridge_shares_the_dead_reckon_age_budget(tracker):
    tracker.profile.dead_reckon.max_frames = 1
    tracker._dead_reckon_age = 1
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(10.1, 5.0))
    assert tracker._step_imu_bridge(10.1)[0] is None


def test_bridge_needs_a_map_up_axis(tracker):
    tracker.map_frame = None
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(10.1, 5.0))
    assert tracker._step_imu_bridge(10.1)[0] is None


def test_single_sample_holds_pose_without_claiming_motion(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    pose, center, status, _inliers, info = tracker._step_imu_bridge(10.05)
    assert status == "IMU_BRIDGE"
    np.testing.assert_allclose(pose, _pose(), atol=1e-12)
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
    assert info["imu_yaw_delta_deg"] == pytest.approx(0.0)


def test_malformed_samples_are_dropped(tracker):
    tracker.observe_fused_state(SimpleNamespace(stamp=float("nan"), yaw=0.0))
    tracker.observe_fused_state(SimpleNamespace(stamp=10.0, yaw=None))
    tracker.observe_fused_state(None)
    assert list(tracker._fused_yaw) == []


def test_bridge_status_stays_weak_downstream():
    from direct_localizer_adapter import _STATUS_MODE

    assert _STATUS_MODE["IMU_BRIDGE"] == ("WEAK_TRACK", True)


def test_imu_bridge_bookkeeping_does_not_refresh_visual_stamp(tracker):
    seed(tracker, stamp=10.0)
    gray = np.zeros((8, 8), dtype=np.uint8)
    pose = _pose()
    center = np.array([1.0, 2.0, 3.0])
    for i in range(1, 102):
        t = 10.0 + i * 0.1
        tracker._step_bookkeeping(center, pose, 2, gray, t, "IMU_BRIDGE")
        assert tracker._last_visual_stamp == pytest.approx(10.0)
    tracker._last_pose_stamp = 10.0
    tracker._dead_reckon_age = 0
    assert (20.1 - 10.0) > tracker.profile.dead_reckon.max_age_s
    assert tracker._imu_bridge_armed(20.1) is False


def test_nan_roll_does_not_bridge(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(10.05, 1.0, roll_deg=float("nan")))
    pose, _center, _status, _inl, _info = tracker._step_imu_bridge(10.1)
    assert pose is None


def test_out_of_order_fused_sample_is_dropped(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(9.95, 20.0))
    assert [entry[0] for entry in tracker._fused_yaw] == pytest.approx([10.0])
    _pose_out, _center, _status, _inl, info = tracker._step_imu_bridge(10.1)
    assert abs(info["imu_yaw_delta_deg"]) < 1.0
    yaws_deg = [math.degrees(entry[1]) for entry in tracker._fused_yaw]
    assert all(abs(yaw - 20.0) > 1.0 for yaw in yaws_deg)


def test_repeated_fused_sample_does_not_create_more_measurements(tracker):
    tracker.observe_fused_state(sample(10.0, 0.0))
    tracker.observe_fused_state(sample(10.0, 0.0))
    assert len(tracker._fused_yaw) == 1


def test_bridge_logs_why_stale_attitude_was_rejected(tracker):
    seed(tracker)
    tracker.observe_fused_state(sample(10.0, 0.0))
    _pose_out, _center, _status, _inliers, info = tracker._step_imu_bridge(10.6)
    assert info["imu_bridge_reason"] == "sample_stale"
    assert info["imu_sample_stamp_mono"] == 10.0
    assert info["imu_sample_age_s"] == pytest.approx(0.6)
