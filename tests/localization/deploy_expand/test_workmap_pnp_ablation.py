from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

VALIDATION_DIR = Path(__file__).resolve().parents[3] / "定位演算法" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import workmap_pnp_ablation as experiment  # noqa: E402


def measurements():
    rng = np.random.default_rng(104)
    xyz = rng.uniform([-1.0, -0.6, 3.0], [1.0, 0.6, 6.0], size=(140, 3))
    ids = np.r_[np.arange(1, 101), -np.arange(1, 41)]
    K = np.array([[720.0, 0, 480], [0, 720.0, 270], [0, 0, 1]])
    projected = xyz @ K.T
    xy = projected[:, :2] / projected[:, 2:]
    row = {"ids": ids.tolist(), "image_xy_px": xy.tolist(), "world_xyz_u": xyz.tolist(),
           "K": K.tolist(), "image_size_px": [960, 540]}
    pnp = {"max_error_px": 4.0, "min_trials": 10, "max_trials": 100,
           "confidence": 0.999, "refine_iters": 2, "min_points": 12, "min_inliers": 12}
    return row, ids, xyz, K, pnp


def test_real_pnp_withholds_same_main_ids_and_does_not_initialize_from_pose():
    row, ids, xyz, K, pnp = measurements()
    # A deliberately wrong published pose is metadata, never an initializer.
    row["cam_from_world"] = [[1, 0, 0, 100], [0, 1, 0, 100], [0, 0, 1, 100]]
    before = np.array(row["world_xyz_u"])
    report = experiment.compare_observations(row, {"ordinal": 8}, K, ids[:100], xyz[:100], pnp, 80, 0)
    withheld = set(report["heldout_main_ids"])
    assert len(withheld) == 20
    assert report["training_main_available"] == 80
    assert report["strong_main_training_eligible"]
    for name in ("frozen_only", "mixed"):
        result = report[name]
        assert result["initializer"] is None
        assert withheld.isdisjoint(result["training_ids"])
        assert result["training_main_inliers"] == 80
        assert result["heldout_main"]["support_count"] == 20
        assert result["heldout_main"]["positive_depth_count"] == 20
        assert result["heldout_main"]["reprojection_p90_px"] < 1e-5
    assert report["frozen_only"]["training_new_available"] == 0
    assert report["mixed"]["training_new_inliers"] == 40
    np.testing.assert_array_equal(before, row["world_xyz_u"])


def test_holdout_boundary_and_order_invariance():
    ids = np.r_[np.arange(1, 100), [-1, -2]]
    main, mixed, heldout = experiment.split_observations(ids)
    assert main.sum() == 80
    assert heldout.sum() == 19
    assert mixed.sum() == 82
    reversed_ids = ids[::-1]
    reversed_split = experiment.split_observations(reversed_ids)
    assert set(ids[heldout]) == set(reversed_ids[reversed_split[2]])
    assert not np.any(main & heldout)
    assert not np.any(mixed & heldout)


def test_stale_frozen_xyz_rejected_before_any_fit():
    row, ids, xyz, K, pnp = measurements()
    row["world_xyz_u"][0][0] += 0.01
    with pytest.raises(experiment.closure.ExperimentError, match="immutable main-map coordinates"):
        experiment.compare_observations(row, {"ordinal": 8}, K, ids[:100], xyz[:100], pnp, 80, 0)
