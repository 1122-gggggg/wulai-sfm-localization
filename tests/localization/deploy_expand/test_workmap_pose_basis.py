"""Which pose the workmap front end publishes: mixed like the baseline, frozen-only when it must."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pycolmap

from online_workmap import OnlineWorkMapTracker
from workmap_submaps import SubmapRegistry

FX = FY = 500.0
CX, CY = 320.0, 240.0


def _project(xyz, pose):
    cam = xyz @ pose[:, :3].T + pose[:, 3]
    return np.column_stack([FX * cam[:, 0] / cam[:, 2] + CX, FY * cam[:, 1] / cam[:, 2] + CY])


def _tracker(*, mixed_shift):
    rng = np.random.default_rng(3)
    truth = np.hstack([np.eye(3), np.zeros((3, 1))])
    shifted = truth.copy()
    shifted[0, 3] = mixed_shift
    old_xyz = rng.uniform([-1, -1, 4], [1, 1, 6], size=(100, 3))
    new_xyz = rng.uniform([-1, -1, 4], [1, 1, 6], size=(400, 3))
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker.mapping_enabled = True
    tracker.workmap_eval_stage = "full"
    tracker._ba_executor = None
    tracker._submaps = SubmapRegistry()
    tracker._presubmap = None
    tracker._pycolmap = pycolmap
    tracker._camera = pycolmap.Camera.create(1, "PINHOLE", FX, 640, 480)
    tracker._pnp_estimation = pycolmap.AbsolutePoseEstimationOptions()
    tracker._pnp_estimation.ransac.max_error = 4.0
    tracker._pnp_estimation.ransac.random_seed = 0
    tracker._pnp_refinement = pycolmap.AbsolutePoseRefinementOptions()
    tracker._fx, tracker._fy, tracker._cx, tracker._cy = FX, FY, CX, CY
    tracker._live_xyz = np.vstack([old_xyz, new_xyz])
    tracker._live_xy = np.vstack([_project(old_xyz, truth), _project(new_xyz, shifted)])
    tracker._live_ids = np.concatenate([np.arange(1, 101), -np.arange(1, 401)]).astype(np.int64)
    tracker.profile = SimpleNamespace(
        fast_loop=SimpleNamespace(pnp=SimpleNamespace(min_inliers=10)),
        reloc=SimpleNamespace(frozen_pnp_thresholds={"strong_inliers": 80}),
    )
    return tracker


def test_consistent_new_geometry_preserves_measured_main_and_new_support():
    tracker = _tracker(mixed_shift=0.0)
    answer, mask = tracker._workmap_absolute_pose()
    pose = answer["cam_from_world"].matrix()
    error = np.linalg.norm(_project(tracker._live_xyz, pose) - tracker._live_xy, axis=1)
    assert np.max(error) < 1e-5
    assert np.count_nonzero(mask & (tracker._live_ids > 0)) >= 80
    assert np.count_nonzero(mask & (tracker._live_ids < 0)) >= 30


def test_dominant_drifted_new_points_cannot_drag_the_frozen_map_pose():
    tracker = _tracker(mixed_shift=0.4)
    answer, mask = tracker._workmap_absolute_pose()
    pose = answer["cam_from_world"].matrix()
    old = tracker._live_ids > 0
    error = np.linalg.norm(_project(tracker._live_xyz[old], pose) - tracker._live_xy[old], axis=1)
    assert np.max(error) < 1e-5
    assert np.count_nonzero(mask & old) >= 80
    assert not np.any(mask & ~old)
