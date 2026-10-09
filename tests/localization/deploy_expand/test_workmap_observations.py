"""Session workmap geometry regressions using the real bounded visual PnP."""

from dataclasses import replace

import numpy as np
import pytest

from two_rate_tracker import SeedProvenance, TwoRateTracker
from test_direct_profile import _load, _valid_profile


def _tracker(tmp_path):
    raw = _valid_profile()
    raw["fast_loop"].update(topup=False, track_cap=500)
    raw["reloc"]["frozen_pnp_thresholds"]["strong_inliers"] = 80
    # Every synthetic geometry capture is a VO keyframe in these scenarios.
    raw["vo"]["keyframe_stride"] = 1
    return TwoRateTracker(None, _load(tmp_path, raw), None, online_workmap_active=True)


def _pixels(tracker, xyz, pose):
    camera = xyz @ pose[:, :3].T + pose[:, 3]
    return camera[:, :2] * [tracker._fx, tracker._fy] / camera[:, 2, None] + [
        tracker._cx,
        tracker._cy,
    ]


def _scene(tracker, *, n=120):
    rng = np.random.default_rng(281)
    xyz = np.column_stack(
        (rng.uniform(-2.2, 2.2, n), rng.uniform(-1.2, 1.2, n), rng.uniform(4.7, 6.3, n))
    )
    origin = np.column_stack((np.eye(3), np.zeros(3)))
    moved = np.column_stack((np.eye(3), [-0.16, 0.0, 0.0]))
    tracker._live_ids = np.arange(1, n + 1, dtype=np.int64)
    tracker._live_xyz = xyz.copy()
    tracker._live_xy = np.ascontiguousarray(_pixels(tracker, xyz, moved))
    return xyz, origin, moved


def test_triangulated_pixels_are_actual_two_view_observations_and_next_pose_uses_new_ids(tmp_path):
    tracker = _tracker(tmp_path)
    xyz, seed_pose, current_pose = _scene(tracker)
    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    status = tracker._step_absolute_pose(False, 1.25)
    assert status[0] == "FAST_TRACK"
    assert tracker._workmap_pose_admissible(status[1], status[0], status[4], status[6])
    current_pose = status[1]
    new_xyz = xyz[:30] + [0.09, 0.05, 0.25]
    seed_xy = _pixels(tracker, new_xyz, seed_pose) + [0.12, -0.08]
    current_xy = _pixels(tracker, new_xyz, current_pose) + [-0.11, 0.07]
    tracker._vo_seed_xy = seed_xy.copy()
    tracker._vo_xy = current_xy.copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    tracker._vo_seed_meta = {0: SeedProvenance(0, 1.0, tracker._source_epoch, True)}
    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)
    assert len(tracker._workmap_landmarks) == len(new_xyz)
    for key, item in tracker._workmap_landmarks.items():
        index = -key - 1
        first, second = item["observations"]
        np.testing.assert_array_equal(first[2], seed_xy[index])
        np.testing.assert_array_equal(second[2], current_xy[index])
        assert first[:2] == (0, 1.0) and second[:2] == (8, 1.25)
        assert item["revision"] == 0
    next_pose = np.column_stack((np.eye(3), [-0.19, 0.0, 0.0]))
    tracker._ordinal = 9
    tracker._live_xy = np.ascontiguousarray(_pixels(tracker, tracker._live_xyz, next_pose))
    result = tracker._step_absolute_pose(False, 1.3)
    assert result[0] == "FAST_TRACK" and result[5] >= 12
    assert result[4] >= 80 and result[1] is not None


def test_cap_protects_old_inliers_but_does_not_starve_qualified_new_landmarks(tmp_path):
    tracker = _tracker(tmp_path)
    xyz, _origin, moved = _scene(tracker, n=530)
    new_ids = -np.arange(1, 51, dtype=np.int64)
    tracker._live_ids = np.concatenate((tracker._live_ids, new_ids))
    tracker._live_xyz = np.vstack((xyz, xyz[:50]))
    tracker._live_xy = np.ascontiguousarray(_pixels(tracker, tracker._live_xyz, moved))
    tracker._workmap_landmarks = {int(key): {"valid_observation_count": 2} for key in new_ids}
    selected = tracker._select_workmap_live_points(tracker.profile.fast_loop.track_cap, moved)
    assert len(selected) == tracker.profile.fast_loop.track_cap
    assert np.count_nonzero(tracker._live_ids[selected] > 0) >= 80
    assert np.any(tracker._live_ids[selected] < 0)
    assert len(np.unique(selected)) == len(selected)


def test_mixed_pnp_does_not_displace_strong_old_map_pose(tmp_path):
    tracker = _tracker(tmp_path)
    old_xyz, _origin, true_pose = _scene(tracker)
    rng = np.random.default_rng(839)
    new_xyz = np.column_stack(
        (rng.uniform(-1.9, 1.9, 145), rng.uniform(-1.1, 1.1, 145), rng.uniform(4.8, 6.2, 145))
    )
    false_pose = np.column_stack((np.eye(3), [-1.0, 0.0, 0.0]))
    tracker._live_ids = np.concatenate((tracker._live_ids, -np.arange(1, len(new_xyz) + 1)))
    tracker._live_xyz = np.vstack((old_xyz, new_xyz))
    tracker._live_xy = np.ascontiguousarray(
        np.vstack((_pixels(tracker, old_xyz, true_pose), _pixels(tracker, new_xyz, false_pose)))
    )
    status, pose, _center, _inliers, old_inliers, _new_inliers, _corr, _quality = (
        tracker._step_absolute_pose(False, 1.25)
    )
    assert status == "FAST_TRACK" and old_inliers >= 80
    np.testing.assert_allclose(pose, true_pose, atol=0.02)
    assert len(tracker._live_ids) >= old_inliers


def test_nonvisual_and_unobservable_poses_cannot_start_mapping(tmp_path):
    tracker = _tracker(tmp_path)
    xyz, _origin, pose = _scene(tracker)
    for status in ("DEAD_RECKON", "IMU_BRIDGE", "NO_POSE"):
        assert not tracker._workmap_pose_admissible(pose, status, len(xyz), len(xyz))
    assert not tracker._workmap_initial_anchor
    tracker._live_xy[:] = tracker._live_xy[0]
    assert not tracker._workmap_pose_admissible(pose, "FAST_TRACK", len(xyz), len(xyz))
    assert not tracker._workmap_initial_anchor


def test_bad_parallax_depth_and_outlier_pairs_never_become_landmarks(tmp_path):
    for source_point, baseline, outlier in (
        (np.array([0.2, 0.1, 5.0]), 0.0001, [0.0, 0.0]),
        (np.array([0.2, 0.1, -5.0]), 0.16, [0.0, 0.0]),
        (np.array([0.2, 0.1, 5.0]), 0.16, [100.0, -120.0]),
    ):
        tracker = _tracker(tmp_path)
        source = np.column_stack((np.eye(3), np.zeros(3)))
        moved = np.column_stack((np.eye(3), [-baseline, 0.0, 0.0]))
        tracker._vo_seed_xy = _pixels(tracker, source_point[None], source)
        tracker._vo_xy = _pixels(tracker, source_point[None], moved) + outlier
        tracker._vo_bid = np.array([0], dtype=np.int64)
        tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
        tracker._vo_pose_by_bid = {0: source}
        tracker._vo_seed_meta = {0: SeedProvenance(0, 1.0, tracker._source_epoch, True)}
        tracker._ordinal = 8
        tracker._previous_capture_stamp = 1.2
        tracker.approve_capture(
            moved, status="FAST_TRACK", map_inliers=100, n_corr=100, ordinal=8, stamp=1.2
        )
        tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), moved)
        assert not tracker._workmap_landmarks
        assert not np.any(tracker._live_ids < 0)


@pytest.mark.parametrize("submap_waiting", [False, True])
def test_consistent_new_tracks_are_pose_inputs_unless_a_submap_needs_the_map_only_pose(
    tmp_path, submap_waiting
):
    tracker = _tracker(tmp_path)
    tracker.profile = replace(
        tracker.profile,
        online_workmap=replace(tracker.profile.online_workmap, retain_valid_tracks=True),
    )
    old, first, pose = _scene(tracker)
    fresh = old[:30] + [0.06, 0.02, 0.15]
    ids = -np.arange(1, len(fresh) + 1)
    observed = _pixels(tracker, fresh, pose) + [0.75, -0.25]
    observed[-1] += [40.0, 30.0]
    tracker._live_ids = np.concatenate((tracker._live_ids, ids))
    tracker._live_xyz = np.vstack((old, fresh))
    tracker._live_xy = np.ascontiguousarray(np.vstack((tracker._live_xy, observed)))
    tracker._workmap_landmarks = {
        int(key): {
            "xyz": xyz,
            "observations": ((0, 0.0, xy, first), (1, 0.1, xy, pose)),
            "revision": 0,
        }
        for key, xyz, xy in zip(ids, fresh, observed, strict=True)
    }
    tracker._ordinal = 2
    if submap_waiting:
        tracker._anchors_wanted = lambda: True
    result = tracker._step_absolute_pose(False, 0.2)
    assert result[0] == "FAST_TRACK"
    assert set(tracker._live_ids[tracker._live_ids < 0]) == set(ids[:-1])
    assert tracker._workmap_landmarks[-1]["last_inlier"][0] == 2
    if submap_waiting:
        np.testing.assert_allclose(result[1], pose, atol=1e-7)
        assert result[5] == 0 and tracker._workmap_pose_ids == set()
        assert result[7]["workmap_retained_tracks"] == 29
        assert tracker._pose_basis == "old_only"
    else:
        np.testing.assert_allclose(result[1], pose, atol=2e-2)
        assert result[5] == 29 and len(tracker._workmap_pose_ids) == 29
        assert tracker._pose_basis == "mixed"
