"""Measured expansion paths preserve frozen-map geometry."""

import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from online_workmap import OnlineWorkMapTracker
from two_rate_tracker import KLTTracker, SeedProvenance
from workmap_query import FrozenCoverage
from workmap_submaps import SubmapRegistry
from test_workmap_observations import _pixels, _scene, _tracker


def test_guided_klt_measures_the_true_shift_from_a_nearby_guess():
    rng = np.random.default_rng(512)
    image = cv2.GaussianBlur(rng.integers(0, 256, (540, 960), dtype=np.uint8), (5, 5), 1)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 57], [0, 1, 9]]), (960, 540))
    xy = np.array([(x, y) for y in range(180, 360, 30) for x in range(180, 720, 30)], float)
    expected = xy + [57, 9]
    tracker = KLTTracker(fb_max_px=0.2, win=21, levels=3)
    tracker.set_frame(image)
    diagnostics = tracker.last_diagnostics.copy()
    measured, keep = tracker.track_guided(image, shifted, xy, expected + [3, 0])
    assert keep.all()
    assert np.max(np.linalg.norm(measured - expected, axis=1)) < 0.2
    unrelated = cv2.GaussianBlur(rng.integers(0, 256, image.shape, dtype=np.uint8), (5, 5), 1)
    _, keep = tracker.track_guided(image, unrelated, xy, expected)
    assert np.mean(~keep) >= 0.9
    assert tracker._prev_gray is image
    assert tracker.last_diagnostics == diagnostics


def test_triangulated_points_inside_frozen_region_are_not_registered(tmp_path):
    tracker = _tracker(tmp_path)
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:30] + [0.09, 0.05, 0.25]
    coverage = FrozenCoverage(np.repeat(new_xyz[:1], 3, axis=0))
    tracker._admit_new_landmarks = lambda world, good: good & ~coverage.covered(world)
    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = _pixels(tracker, new_xyz, seed_pose)
    tracker._vo_xy = _pixels(tracker, new_xyz, current_pose)
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose}
    tracker._vo_seed_meta = {0: SeedProvenance(0, 1.0, tracker._source_epoch, True)}
    tracker.approve_capture(
        current_pose, status="FAST_TRACK", map_inliers=100, n_corr=100, ordinal=8, stamp=1.25
    )
    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), np.uint8), current_pose)
    expected = new_xyz[~coverage.covered(new_xyz)]
    actual = np.array([value["xyz"] for value in tracker._workmap_landmarks.values()])
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert np.count_nonzero(tracker._live_ids < 0) == len(new_xyz)


def test_online_gate_counts_rejections_and_fails_closed_without_coverage():
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker._submaps = SubmapRegistry()
    tracker._expansion_gate = True
    tracker._workmap_frozen_gate_rejected = 0
    tracker._coverage = FrozenCoverage(np.repeat([[0.02, 0.02, 0.02]], 3, axis=0))
    world = np.array([[0.03, 0.03, 0.03], [0.5, 0.5, 0.5], [np.nan, 0, 0]])
    np.testing.assert_array_equal(
        tracker._admit_new_landmarks(world, np.ones(3, bool)), [False, True, False]
    )
    assert tracker._workmap_frozen_gate_rejected == 2
    tracker._coverage = None
    with pytest.raises(RuntimeError, match="ensure_models"):
        tracker._admit_new_landmarks(world, np.ones(3, bool))


def test_depth_job_supersedes_queued_request_and_reports_errors():
    from two_rate_tracker import RelocWorker

    entered, release = threading.Event(), threading.Event()
    callbacks = queue.Queue()
    calls = []

    def localize(_gray):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(ok=False)

    def lift(request):
        calls.append(request["bid"])
        raise ValueError("depth fixture failed")

    def callback(request, result, error):
        callbacks.put((request["bid"], result, error))

    worker = RelocWorker(SimpleNamespace(localize_array=localize), SimpleNamespace(lift=lift))
    worker.start()
    try:
        assert worker.submit(np.zeros((10, 10), np.uint8), 0, 0.0, 0)
        assert entered.wait(5)
        assert worker.submit_depth({"bid": 1}, callback)
        assert worker.submit_depth({"bid": 2}, callback)
        assert callbacks.get(timeout=5) == (1, None, "superseded")
        assert worker.workmap_diagnostics()["workmap_depth_pending"]
        release.set()
        assert callbacks.get(timeout=5) == (2, None, "ValueError: depth fixture failed")
        stats = worker.workmap_diagnostics()
        assert stats["workmap_depth_errors"] == 1
        assert stats["workmap_depth_healthy"] is False
        assert calls == [2]
    finally:
        release.set()
        worker.close()


def test_worker_close_discards_pending_depth_without_callback():
    from two_rate_tracker import RelocWorker

    entered, release = threading.Event(), threading.Event()
    calls = []

    def localize(_gray):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(ok=False)

    worker = RelocWorker(
        SimpleNamespace(localize_array=localize),
        SimpleNamespace(lift=lambda request: calls.append(request)),
    )
    worker.start()
    assert worker.submit(np.zeros((10, 10), np.uint8), 0, 0.0, 0)
    assert entered.wait(5)
    assert worker.submit_depth({"bid": 1}, lambda *args: calls.append(args))
    closer = threading.Thread(target=worker.close)
    closer.start()
    with worker._cv:
        assert worker._cv.wait_for(lambda: worker._closed, timeout=5)
    release.set()
    closer.join(5)
    assert not closer.is_alive()
    assert calls == []


def test_depth_promotion_keeps_measured_views_and_unpromoted_seeds(tmp_path):
    base = _tracker(tmp_path)
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker.__dict__.update(base.__dict__)
    tracker._expansion_query = False
    tracker._expansion_gate = True
    tracker._submaps = SubmapRegistry()
    tracker._init_expansion_state()
    xyz, seed_pose, pose = _scene(tracker)
    points = xyz[:3] + [0.09, 0.05, 0.25]
    tracker._coverage = FrozenCoverage(np.repeat(points[:1], 3, axis=0))
    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_meta = {0: SeedProvenance(0, 1.0, tracker._source_epoch, True)}
    tracker._vo_pose_by_bid = {0: seed_pose}
    tracker.approve_capture(
        pose, status="FAST_TRACK", map_inliers=100, n_corr=100, ordinal=8, stamp=1.25
    )
    tracker._vo_seed_xy = _pixels(tracker, points, seed_pose)
    tracker._vo_xy = _pixels(tracker, points, pose)
    tracker._vo_xy[2] += [5, 0]
    tracker._vo_bid = np.zeros(3, np.int64)
    seed_xy, tracked_xy = tracker._vo_seed_xy.copy(), tracker._vo_xy.copy()
    result = dict(ok=True, seed_xy=seed_xy, xyz=points, valid=np.ones(3, bool))
    assert tracker._promote_depth_batch(0, result, pose) == 1
    landmark = tracker._workmap_landmarks[-1]
    assert landmark["origin"] == "depth"
    np.testing.assert_allclose(landmark["xyz"], points[1])
    np.testing.assert_array_equal(landmark["observations"][0][2], seed_xy[1])
    np.testing.assert_array_equal(landmark["observations"][1][2], tracked_xy[1])
    np.testing.assert_array_equal(tracker._vo_seed_xy, seed_xy[[0, 2]])
    assert tracker._depth_counts["promoted"] == 1
    assert tracker._workmap_frozen_gate_rejected == 1
    assert tracker._vo_next_id == -2


def test_depth_results_reject_stale_epoch_and_scale_jump(tmp_path):
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker._w, tracker._h = 640, 480
    tracker._expansion_query = False
    tracker._init_expansion_state()
    tracker._source_epoch = 2
    for _ in range(5):
        assert tracker._depth_scale_consistent(7.5)
    accepted = dict(ok=True, scale=7.5, infer_ms=50)
    tracker._on_depth_result(dict(source_epoch=1, bid=1), accepted, None)
    tracker._on_depth_result(dict(source_epoch=2, bid=2), dict(accepted, scale=10), None)
    tracker._on_depth_result(dict(source_epoch=2, bid=3), accepted, None)
    tracker._drain_depth_results()
    assert set(tracker._depth_ready) == {3}
    assert tracker._depth_counts["rejected"] == 1
    assert tracker._depth_counts["completed"] == 1
