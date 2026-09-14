"""Real KLT translation and its diagnostic contract for the next flight."""

import cv2
import numpy as np
from types import SimpleNamespace

from two_rate_tracker import KLTTracker, TwoRateTracker


def test_klt_recovers_known_image_translation_and_records_quality():
    rng = np.random.default_rng(19)
    original = cv2.GaussianBlur(rng.integers(0, 255, (384, 512), dtype=np.uint8), (3, 3), 0)
    points = cv2.goodFeaturesToTrack(original, 100, 0.02, 8).reshape(-1, 2)
    points = points[(points[:, 0] > 64) & (points[:, 0] < 448) & (points[:, 1] > 64) & (points[:, 1] < 320)]
    moved = cv2.warpAffine(original, np.float32([[1, 0, 3], [0, 1, -2]]), (512, 384), borderMode=cv2.BORDER_REFLECT101)
    tracker = KLTTracker(fb_max_px=1, win=15, levels=3)
    tracker.set_frame(original)
    result, kept = tracker.track(moved, points)
    assert kept.sum() > 0.8 * len(points)
    np.testing.assert_allclose(np.median(result[kept] - points[kept], axis=0), [3, -2], atol=0.2)
    diagnostic = tracker.last_diagnostics
    assert diagnostic["klt_input_points"] == len(points)
    assert diagnostic["klt_kept_points"] == int(kept.sum())
    assert diagnostic["klt_fb_p95_px"] < 1.0


def test_missing_opencv_flow_is_a_rejection_not_a_worker_crash(monkeypatch):
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda *a, **kw: (None, None, None))
    tracker = KLTTracker(fb_max_px=1, win=15, levels=3)
    frame = np.zeros((32, 32), np.uint8)
    points = np.array([[10., 10.], [20., 20.]])
    tracker.set_frame(frame)
    _positions, kept = tracker.track(frame, points)
    assert not kept.any()
    assert tracker.last_diagnostics["klt_kept_points"] == 0


def test_pnp_debug_samples_are_bounded_timestamped_and_reprojectable():
    tracker = object.__new__(TwoRateTracker)
    tracker._reset_tracking_state()
    tracker.profile = SimpleNamespace(fast_loop=SimpleNamespace(pnp=SimpleNamespace(min_points=4, min_inliers=4)))
    tracker._w, tracker._h = 512, 384
    tracker._K = np.array([[300., 0, 256], [0, 300, 192], [0, 0, 1.]])
    tracker._live_xyz = np.random.default_rng(5).uniform([-1, -1, 3], [1, 1, 5], (200, 3))
    projected = (tracker._K @ tracker._live_xyz.T).T
    tracker._live_xy = projected[:, :2] / projected[:, 2:]
    tracker._live_ids = np.arange(1, 201)
    tracker._camera = tracker._pnp_estimation = tracker._pnp_refinement = None
    pose = np.column_stack((np.eye(3), np.zeros(3)))
    tracker._pycolmap = SimpleNamespace(estimate_and_refine_absolute_pose=lambda *a: {
        "inlier_mask": np.ones(200, bool), "cam_from_world": SimpleNamespace(matrix=lambda: pose)})
    sample = tracker._step_absolute_pose(False, 10.0)[-1]["pnp_observation_sample"]
    assert sample["sample_count"] == 128 and sample["total_inliers"] == 200
    assert sample["capture_stamp_mono"] == 10.0
    xyz = np.array(sample["world_xyz_u"])
    pixels = (np.array(sample["K"]) @ xyz.T).T
    np.testing.assert_allclose(pixels[:, :2] / pixels[:, 2:], sample["image_xy_px"])
    assert "pnp_observation_sample" not in tracker._step_absolute_pose(False, 10.1)[-1]
    assert "pnp_observation_sample" not in tracker._step_absolute_pose(True, 10.4)[-1]
    assert "pnp_observation_sample" in tracker._step_absolute_pose(True, 10.5)[-1]
