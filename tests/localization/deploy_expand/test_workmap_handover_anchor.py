"""Independent map anchors must not interrupt an already strong front end."""
from types import SimpleNamespace

import cv2
import numpy as np
import pycolmap

from online_workmap import OnlineWorkMapTracker
from two_rate_tracker import KLTTracker
from workmap_submaps import SubmapRegistry


def scene_tracker(*, healthy=True, spread=True, grid_size=10):
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker._reset_tracking_state()
    tracker._online_workmap_active = True
    tracker.mapping_enabled = True
    tracker._submaps = SubmapRegistry()
    tracker._presubmap = SimpleNamespace(active=True)
    tracker._w, tracker._h = 640, 480
    tracker._fx = tracker._fy = 500.0
    tracker._cx, tracker._cy = 320.0, 240.0
    tracker._K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    tracker._camera = pycolmap.Camera.create(1, "PINHOLE", 500., 640, 480)
    tracker._pycolmap = pycolmap
    tracker._pnp_estimation = pycolmap.AbsolutePoseEstimationOptions()
    tracker._pnp_estimation.ransac.max_error = 4.0
    tracker._pnp_estimation.ransac.random_seed = 0
    tracker._pnp_refinement = pycolmap.AbsolutePoseRefinementOptions()
    tracker.profile = SimpleNamespace(
        fast_loop=SimpleNamespace(pnp=SimpleNamespace(min_points=10, min_inliers=10),
                                  reseed_min_points=120, track_cap=500),
        reloc=SimpleNamespace(frozen_pnp_thresholds={
            "strong_inliers": 80, "minimum_inlier_ratio": .25,
            "minimum_hull_coverage": .15, "minimum_occupancy_4x4": 6,
            "minimum_positive_depth_ratio": .99, "maximum_reprojection_p90_px": 3.0}),
        dead_reckon=SimpleNamespace(enabled=False),
        optimizations=SimpleNamespace(spatial_selection=False),
    )
    tracker._ordinal = 11
    tracker._last_pose_ordinal = 10
    tracker._last_pose_status = "FAST_TRACK" if healthy else "NO_POSE"
    tracker._last_map_inliers = 150 if healthy else 0
    xs, ys = np.meshgrid(np.linspace(70, 570, grid_size), np.linspace(60, 420, grid_size))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    if not spread:
        xy = xy * .04 + [300., 220.]
    z = np.random.default_rng(7).uniform(4., 7., len(xy))
    xyz = np.column_stack([(xy[:, 0] - 320.) * z / 500.,
                           (xy[:, 1] - 240.) * z / 500., z])
    image = np.zeros((480, 640), np.uint8)
    for x, y in xy:
        cv2.rectangle(image, (round(x)-3, round(y)-3), (round(x)+3, round(y)+3), 255, -1)
    tracker._tracker = KLTTracker(fb_max_px=1., win=15, levels=3)
    tracker._tracker.set_frame(image)
    tracker._live_ids = np.arange(1, len(xy) + 1, dtype=np.int64)
    tracker._live_xy, tracker._live_xyz = xy.copy(), xyz.copy()
    return tracker, image, np.arange(1001, 1001 + len(xy), dtype=np.int64), xy, xyz


def test_current_map_anchor_does_not_reseed_healthy_tracking():
    tracker, image, ids, xy, xyz = scene_tracker()
    before = tracker._live_ids.copy()
    reseeded, *_ = tracker._apply_handover(image, ids, xy, xyz)
    assert not reseeded
    # Actual KLT carries the old observations; the new map fix is independent.
    assert set(before).issubset(tracker._live_ids)
    answer = tracker._handover_anchor_answer
    assert answer is not None and np.count_nonzero(answer["inlier_mask"]) >= 80
    np.testing.assert_allclose(answer["cam_from_world"].matrix(),
                               np.column_stack([np.eye(3), np.zeros(3)]), atol=1e-6)
    # No current delivery means the next capture cannot reuse this anchor.
    tracker._apply_handover(image, ids[:0], xy[:0], xyz[:0])
    assert tracker._handover_anchor_answer is None


def test_same_strong_handover_still_recovers_lost_tracking():
    tracker, image, ids, xy, xyz = scene_tracker(healthy=False)
    reseeded, *_ = tracker._apply_handover(image, ids, xy, xyz)
    assert reseeded
    assert set(tracker._live_ids) == set(ids)


def test_clustered_map_measurements_cannot_be_a_closure_anchor():
    tracker, image, ids, xy, xyz = scene_tracker(spread=False)
    tracker._apply_handover(image, ids, xy, xyz)
    assert tracker._handover_anchor_answer is None


def test_new_tracks_cannot_displace_available_main_support_during_recovery():
    tracker, _image, _ids, xy, xyz = scene_tracker()
    main_ids = np.arange(1, 201, dtype=np.int64)
    new_ids = -np.arange(1, 501, dtype=np.int64)
    ids = np.concatenate([main_ids, new_ids])
    pixels, points = np.tile(xy, (7, 1)), np.tile(xyz, (7, 1))
    kept, kept_xy, kept_xyz = tracker._cap_reseed_points(ids, pixels, points)
    # All 200 main observations fit in the existing half-cap main reservation.
    # Hundreds of new points must not reduce their support to the 80-point floor.
    assert set(main_ids).issubset(kept)
    assert np.count_nonzero(kept < 0) == 300
    assert len(kept) == 500
    rows = {int(key): row for row, key in enumerate(ids)}
    expected = [rows[int(key)] for key in kept]
    np.testing.assert_array_equal(kept_xy, pixels[expected])
    np.testing.assert_array_equal(kept_xyz, points[expected])


def test_consistent_strong_fix_refreshes_live_tracks_without_reseed():
    tracker, image, ids, xy, xyz = scene_tracker(grid_size=12)
    tracker._last_pose = np.column_stack([np.eye(3), np.zeros(3)])
    reseeded, points, dr_prev, dr_curr, _ = tracker._apply_handover(image, ids, xy, xyz)
    assert reseeded is False
    assert points == 144 and dr_prev is None and dr_curr is None
    assert set(tracker._live_ids[tracker._live_ids > 0]) == set(ids)
    assert tracker._pose_status(reseeded, 144, 0) == "FAST_TRACK"
    assert tracker._handover_anchor_answer is not None


def test_inconsistent_strong_fix_still_requires_reseed():
    tracker, image, ids, xy, xyz = scene_tracker(grid_size=12)
    tracker._last_pose = np.column_stack([np.eye(3), [0.5, 0., 0.]])
    reseeded, *_ = tracker._apply_handover(image, ids, xy, xyz)
    assert reseeded is True


def test_baseline_strong_fix_keeps_original_reseed_behavior():
    tracker, image, ids, xy, xyz = scene_tracker(grid_size=12)
    tracker._last_pose = np.column_stack([np.eye(3), np.zeros(3)])
    tracker._online_workmap_active = False
    reseeded, *_ = tracker._apply_handover(image, ids, xy, xyz)
    assert reseeded is True


def test_rotation_disagreement_cannot_bypass_reseed():
    tracker, image, ids, xy, xyz = scene_tracker(grid_size=12)
    rotation, _ = cv2.Rodrigues(np.array([0., 0., np.deg2rad(6.)]))
    tracker._last_pose = np.column_stack([rotation, np.zeros(3)])
    reseeded, *_ = tracker._apply_handover(image, ids, xy, xyz)
    assert reseeded is True
