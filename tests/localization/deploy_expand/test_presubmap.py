"""Pre-anchor submap: similarity fit, merge into the map frame, and session registration."""

from __future__ import annotations

from concurrent.futures import Future
import threading
import json
import numpy as np
import pytest

from presubmap import (
    ANCHOR_COUNT,
    GIVE_UP_AFTER_ANCHOR,
    LocalKeyframe,
    LocalPoint,
    PreAnchorSubmap,
)
from workmap_submaps import fit_similarity, map_pose
from test_workmap_promotion import _online_tracker  # noqa: F401 - pytest fixture
from workmap_stitch import align_independent_submap

K = np.array([[720.0, 0, 480.0], [0, 720.0, 270.0], [0, 0, 1.0]])
TRUE = (
    0.037,
    np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]),
    np.array([1.5, -0.4, 2.0]),
)


def _pose(center, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.hstack([rot, (-rot @ np.asarray(center, dtype=float)).reshape(3, 1)])


def _center(pose):
    return -pose[:, :3].T @ pose[:, 3]


def _local_poses(count=20):
    return {40 + i: _pose([0.4 * i, 0.02 * i, 0.1 * i], 0.01 * i) for i in range(count)}


def _pixels(pose, xyz):
    cam = xyz @ pose[:, :3].T + pose[:, 3]
    px = cam @ K.T
    return px[:, :2] / px[:, 2:3]


def test_fit_recovers_the_similarity_from_camera_poses():
    poses = list(_local_poses(10).values())
    mapped = [map_pose(p, TRUE) for p in poses]
    scale, rotation, translation, residual = fit_similarity(
        [(p[:, :3], _center(p)) for p in poses], [(p[:, :3], _center(p)) for p in mapped]
    )
    assert scale == pytest.approx(TRUE[0], rel=1e-9) and residual < 1e-9
    np.testing.assert_allclose(rotation, TRUE[1], atol=1e-9)
    np.testing.assert_allclose(translation, TRUE[2], atol=1e-9)


def test_mapped_pose_and_point_keep_the_local_projection():
    pose = _local_poses(3)[41]
    point = np.array([[1.0, 0.3, 5.0]])
    moved_point = TRUE[0] * (TRUE[1] @ point[0]) + TRUE[2]
    np.testing.assert_allclose(
        _pixels(map_pose(pose, TRUE), moved_point[None]), _pixels(pose, point), atol=1e-6
    )


def _submap_with_track(poses):
    submap = PreAnchorSubmap(K)
    submap.state = "tracking"
    submap.poses = dict(poses)
    submap.stamps = {o: o / 30.0 for o in poses}
    return submap


def test_merge_waits_for_enough_close_anchors_then_hands_over_and_stops():
    poses = _local_poses(20)
    submap = _submap_with_track(poses)
    ordinals = sorted(poses)[::2]                 # every other frame: the span is long enough
    for o in ordinals[: ANCHOR_COUNT - 1]:
        submap.add_anchor(o, map_pose(poses[o], TRUE))
        assert submap.try_merge() is None
    submap.add_anchor(ordinals[ANCHOR_COUNT - 1], map_pose(poses[ordinals[ANCHOR_COUNT - 1]], TRUE))
    result = submap.try_merge()
    assert result is not None and result.scale == pytest.approx(TRUE[0], rel=1e-6)
    assert result.residual_u < 1e-6 and not submap.active and submap.state == "merged"


def test_a_flight_that_starts_inside_the_map_needs_no_submap():
    poses = _local_poses(20)
    submap = _submap_with_track(poses)
    submap._first_stamp = 0.0
    submap.add_anchor(40, map_pose(poses[40], TRUE), stamp=1.0)     # first strong fix after 1 s
    assert submap.state == "unneeded" and not submap.active

    submap = _submap_with_track(poses)
    submap._first_stamp = 0.0
    submap.add_anchor(40, map_pose(poses[40], TRUE), stamp=3.0)     # 3 s outside the map
    assert submap.active


def test_anchors_ahead_of_the_tracker_thread_are_kept_until_it_reaches_them():
    poses = _local_poses(20)
    lagging = {o: p for o, p in poses.items() if o < 50}            # the thread has tracked to 48
    submap = _submap_with_track(lagging)
    ordinals = sorted(poses)[::2]
    for o in ordinals[:ANCHOR_COUNT]:
        submap.add_anchor(o, map_pose(poses[o], TRUE))
    assert submap.try_merge() is None                              # too few anchors tracked so far
    assert len(submap._anchors) == ANCHOR_COUNT                    # untracked ones were not dropped
    submap.poses = dict(poses)
    result = submap.try_merge()
    assert result is not None and result.scale == pytest.approx(TRUE[0], rel=1e-6)


def test_submap_gives_up_when_anchors_keep_coming_but_never_agree():
    poses = _local_poses(20)
    submap = _submap_with_track(poses)
    rng = np.random.default_rng(3)
    for o in sorted(poses)[::2][:ANCHOR_COUNT]:
        mapped = map_pose(poses[o], TRUE)
        mapped[:, 3] += rng.normal(0, 0.5, 3)
        submap.add_anchor(o, mapped)
    assert submap.try_merge() is None and submap.active
    submap.add_anchor(GIVE_UP_AFTER_ANCHOR + 42, map_pose(poses[40], TRUE))
    assert submap.try_merge() is None and submap.state == "unmerged"


def test_inconsistent_anchors_are_not_merged():
    poses = _local_poses(20)
    submap = _submap_with_track(poses)
    rng = np.random.default_rng(3)
    for o in sorted(poses)[::2][:ANCHOR_COUNT]:
        mapped = map_pose(poses[o], TRUE)
        mapped[:, 3] += rng.normal(0, 0.5, 3)               # anchors disagree with the local track
        submap.add_anchor(o, mapped)
    assert submap.try_merge() is None and submap.active


def _fake_result(poses, count=40):
    rng = np.random.default_rng(11)
    points = rng.uniform([-2, -1, 4], [2, 1, 7], (count, 3))
    local_points, keyframes = {}, []
    ordinals = sorted(poses)
    for i, xyz in enumerate(points):
        seed, cur = ordinals[i % 6], ordinals[6 + i % 8]
        local_points[i] = LocalPoint(
            seed, seed / 30.0, _pixels(poses[seed], xyz[None])[0], poses[seed],
            cur, cur / 30.0, _pixels(poses[cur], xyz[None])[0], poses[cur],
            xyz, 3.0, [10 + i, 20, 30],
        )
    rng2 = np.random.default_rng(5)
    for o in ordinals[6::4]:
        seen = [i for i, p in local_points.items() if p.ordinal <= o]
        gray = rng2.integers(0, 256, (540, 960), dtype=np.uint8)
        keyframes.append(LocalKeyframe(o, o / 30.0, gray, np.array(seen), _pixels(poses[o], points[seen])))
    submap = _submap_with_track(poses)
    for o in ordinals[::2][:ANCHOR_COUNT]:
        submap.add_anchor(o, map_pose(poses[o], TRUE))
    submap.points, submap.keyframes = local_points, keyframes
    return submap.try_merge(), points


def test_merged_points_and_keyframes_reach_the_session_in_the_map_frame(_online_tracker):  # noqa: F811
    tracker = _online_tracker
    tracker._vo_next_id = -1
    tracker._source_epoch = 0
    tracker._K = K
    tracker._w, tracker._h = 960, 540
    tracker._presubmap_pending = None
    tracker._presubmap_merged = 0
    tracker._workmap_merge_committed = 0
    result, points = _fake_result(_local_poses(24))
    assert result is not None
    tracker._queue_presubmap(result)
    for _ in range(60):
        tracker._flush_presubmap_queue()
        tracker.session_workmap.flush_writer()
        if tracker._presubmap_pending is None:
            break
    assert tracker._presubmap_pending is None
    session = tracker.session_workmap
    assert len(session.landmarks) == len(points) and all(key < 0 for key in session.landmarks)
    expected = np.array([TRUE[0] * (TRUE[1] @ p) + TRUE[2] for p in points])
    got = np.array([rec["xyz"] for rec in session.landmarks.values()])
    order = lambda a: a[np.lexsort(a.T)]  # noqa: E731
    np.testing.assert_allclose(order(got), order(expected), atol=1e-9)
    assert all(rec["rgb"][1:] == [20, 30] for rec in session.landmarks.values())
    assert len(session.keyframes) == len(result.keyframes) and session.keyframes
    frame = next(iter(session.keyframes.values()))
    # keyframe observations reproject through the mapped pose and mapped points
    err, _ = session._frame_errors(frame, frame["pose"], {})
    assert np.percentile(err, 90) < 1e-3
    ids, xyz, rgb, total = session.pop_display_points(1000)
    assert total == len(points) and len(ids) == len(points) and all(c is not None for c in rgb)


def test_cold_start_needs_shared_frozen_anchor_tracks_before_adding_points(_online_tracker):  # noqa: F811
    tracker = _online_tracker
    result, truth = _fake_result(_local_poses(24), count=40)
    assert result is not None
    unobserved = align_independent_submap(result, K)
    assert not unobserved["ok"]
    assert unobserved["reason"] == "insufficient_anchor_observations"
    assert not tracker.session_workmap.landmarks

    ids = np.arange(len(truth), dtype=np.int64)
    result.anchor_views = {
        ordinal: (ids, _pixels(result.poses_local[ordinal], truth))
        for ordinal in result.anchor_poses
    }
    aligned = align_independent_submap(result, K)
    assert aligned["ok"], aligned.get("reason")
    assert aligned["anchor_p90"] < 1e-3
    assert all(int(key) >= 0 for key in aligned["updates"])
    # This local-only track has no BA observation: it must never be promoted.
    result.points[999] = LocalPoint(
        40, 0.0, np.array([0.0, 0.0]), result.poses_local[40],
        41, 0.0, np.array([0.0, 0.0]), result.poses_local[41],
        np.array([100.0, 100.0, -1.0]), 3.0, None,
    )
    tracker._source_epoch = 0
    tracker._vo_next_id = -1
    tracker._K = K
    tracker._w, tracker._h = 960, 540
    tracker._presubmap_pending = None
    tracker._presubmap_merged = 0
    tracker._workmap_merge_committed = 0
    tracker._ordinal = 70
    tracker._presubmap_alignment_epoch = 0
    tracker._presubmap_alignment_result = result
    tracker._presubmap_alignment_future = Future()
    tracker._fast_close = threading.Event()
    tracker._trajectory_backfill = None
    tracker._presubmap_alignment_future.set_result(aligned)
    tracker.close()
    assert tracker._presubmap_pending is None
    assert tracker._trajectory_backfill is None
    assert not tracker.session_workmap.landmarks
    snapshot = json.loads(
        (tracker.session_workmap.directory / "snapshot.json").read_text()
    )
    assert snapshot["statistics"]["close_dropped"]["presubmap_alignment"] == 1


def test_backfilled_trajectory_interpolates_accepted_pose_corrections():
    from online_workmap import OnlineWorkMapTracker

    result, _ = _fake_result(_local_poses(24))
    for ordinal, shift in ((56, 0.04), (60, 0.08)):
        pose = map_pose(result.poses_local[ordinal], result.similarity)
        corrected = _center(pose) + [0.0, shift, 0.0]
        pose[:, 3] = -pose[:, :3] @ corrected
        result.poses_map[ordinal] = pose
    trail = OnlineWorkMapTracker._submap_backfill(result, 3)
    centers = {point[0]: np.asarray(point[1:]) for point in trail["points"]}
    baseline = _center(map_pose(result.poses_local[58], result.similarity))
    np.testing.assert_allclose(centers[58] - baseline, [0.0, 0.06, 0.0], atol=1e-9)
    assert trail["segment_id"] == (3 << 32) | 40


def test_loss_restarts_independent_submap_only_after_map_frame_vo_fails():
    from types import SimpleNamespace
    from online_workmap import OnlineWorkMapTracker

    class Local:
        def __init__(self, active=False):
            self.active = active
            self.state = "tracking" if active else "merged"
            self.anchors = 0
        def wants(self, ordinal):
            return False
        def enqueue(self, *args):
            raise AssertionError("no frame should be enqueued")
        def add_anchor(self, *args):
            self.anchors += 1
        def try_merge(self):
            return None
        def pop_rejections(self):
            return []

    tracker = object.__new__(OnlineWorkMapTracker)
    tracker.profile = SimpleNamespace(reloc=SimpleNamespace(frozen_pnp_thresholds={"strong_inliers": 80}))
    tracker._handover_anchor_answer = None
    tracker._presubmap = Local()
    tracker._presubmap_had_strong = True
    tracker._presubmap_no_pose_frames = 0
    tracker._presubmap_alignment_future = None
    tracker._presubmap_pending = None
    tracker._presubmap_submap_id = None
    tracker._presubmap_error = None
    tracker._presubmap_ms = 0.0
    tracker._ordinal = 220
    tracker._source_epoch = 0
    tracker._new_presubmap = lambda: Local(True)
    tracker._poll_presubmap_alignment = lambda: None
    tracker._flush_presubmap_queue = lambda: None
    tracker.session_workmap = SimpleNamespace(note_event=lambda *args: None)
    tracker._ring = [(None, None, None, np.zeros((4, 4), dtype=np.uint8))]
    tracker._frame_rgb = None
    tracker._presubmap_after_step({"status": "VO_ONLY", "map_inliers": 0}, 5.0)
    assert tracker._presubmap.state == "merged"
    for _ in range(5):
        tracker._presubmap_after_step({"status": "NO_POSE", "map_inliers": 0}, 5.1)
        assert tracker._presubmap.state == "merged"
    tracker._presubmap_after_step({"status": "NO_POSE", "map_inliers": 0}, 5.1)
    assert tracker._presubmap.active
    tracker._presubmap.anchors = 0
    tracker._presubmap_after_step({
        "status": "FAST_TRACK", "map_inliers": 90,
        "cam_from_world": np.vstack([_pose([1, 0, 0]), [0, 0, 0, 1]]),
    }, 5.2)
    assert tracker._presubmap.anchors == 1
    previous = tracker._presubmap
    previous.active = False
    previous.state = "merged"
    tracker._ordinal += 1
    for _ in range(6):
        tracker._presubmap_after_step({"status": "NO_POSE", "map_inliers": 0}, 5.3)
    assert tracker._presubmap is not previous and tracker._presubmap.active
    # A cold-start VO that loses its tracks before the first old-map fix
    # must also retry, rather than staying failed for the rest of the video.
    failed = tracker._presubmap
    failed.active = False
    failed.state = "failed"
    tracker._presubmap_had_strong = False
    tracker._presubmap_no_pose_frames = 0
    for _ in range(6):
        tracker._presubmap_after_step({"status": "NO_POSE", "map_inliers": 0}, 5.4)
    assert tracker._presubmap is not failed and tracker._presubmap.active


def _render(points, patches, pose):
    """Textured square patches at 3D points seen from ``pose`` (960x540 gray)."""
    image = np.full((540, 960), 40, np.uint8)
    cam = points @ pose[:, :3].T + pose[:, 3]
    px = cam @ K.T
    order = np.argsort(-cam[:, 2])
    for i in order:
        if cam[i, 2] < 1.0:
            continue
        u, v = px[i, :2] / px[i, 2]
        half = int(round(90.0 / cam[i, 2]))
        x0, y0 = int(round(u)) - half, int(round(v)) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half >= 960 or y0 + 2 * half >= 540:
            continue
        import cv2

        image[y0 : y0 + 2 * half, x0 : x0 + 2 * half] = cv2.resize(
            patches[i], (2 * half, 2 * half), interpolation=cv2.INTER_LINEAR
        )
    return image


def test_local_vo_recovers_a_synthetic_camera_path_up_to_similarity():
    rng = np.random.default_rng(7)
    points = np.column_stack(
        (rng.uniform(-6, 6, 220), rng.uniform(-3, 3, 220), rng.uniform(5, 14, 220))
    )
    patches = [rng.integers(0, 256, (12, 12), dtype=np.uint8) for _ in points]
    truth = {o: _pose([0.05 * o, 0.0, 0.02 * o]) for o in range(90)}
    submap = PreAnchorSubmap(K)
    for o, pose in truth.items():
        submap.enqueue(o, o / 30.0, _render(points, patches, pose), None)   # background thread
        submap.wait_idle()                     # real frames arrive at 30 fps; do not overflow the queue
    assert submap.dropped == 0
    assert submap.state == "tracking" and len(submap.poses) > 35 and len(submap.points) > 100
    common = sorted(submap.poses)[10:]
    scale, rotation, translation, residual = fit_similarity(
        [(submap.poses[o][:, :3], _center(submap.poses[o])) for o in common],
        [(truth[o][:, :3], _center(truth[o])) for o in common],
    )
    path = np.linalg.norm(_center(truth[common[-1]]) - _center(truth[common[0]]))
    assert scale > 0 and residual < 0.03 * path


def test_anchors_that_pass_the_residual_but_disagree_between_neighbours_are_blocked():
    poses = _local_poses(20)
    submap = _submap_with_track(poses)
    for index, o in enumerate(sorted(poses)[::2][:ANCHOR_COUNT]):
        mapped = map_pose(poses[o], TRUE)
        center = _center(mapped) + [0.0, 0.0, 0.015 * (1 if index % 2 else -1)]
        mapped[:, 3] = -mapped[:, :3] @ center
        submap.add_anchor(o, mapped)
    assert submap.try_merge() is None and submap.active
    assert submap.pop_rejections() == ["pair_translation"]
    assert submap.pop_rejections() == []
