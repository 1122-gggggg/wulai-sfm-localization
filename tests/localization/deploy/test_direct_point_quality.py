import numpy as np
import pytest

from point_quality import pose_quality, spatial_indices


def test_sampling_keeps_sparse_cells_in_a_clustered_point_set():
    dense = np.tile([10.0, 10.0], (1000, 1))
    sparse = np.array([[180.0, 10.0], [10.0, 110.0], [180.0, 110.0]])
    xy = np.concatenate([dense, sparse])
    take = spatial_indices(xy, 16, 200, 120)
    assert set(range(1000, 1003)).issubset(take)
    assert len(take) == len(np.unique(take)) == 16
    np.testing.assert_array_equal(take, spatial_indices(xy, 16, 200, 120))
    assert np.all(np.diff(take) > 0)


def test_small_point_sets_are_not_reordered():
    xy = np.array([[199.0, 119.0], [0.0, 0.0]])
    np.testing.assert_array_equal(spatial_indices(xy, 20, 200, 120), [0, 1])


def test_quality_measures_known_reprojection_error_and_coverage():
    xy = np.array([[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]])
    xyz = np.column_stack((xy + [3.0, 4.0], np.ones(4)))
    pose = np.column_stack((np.eye(3), np.zeros(3)))
    result = pose_quality(xy, xyz, pose, np.eye(3), 100, 100)
    assert result["reproj_rms"] == pytest.approx(5.0)
    assert result["reproj_p90"] == pytest.approx(5.0)
    assert result["inlier_grid_cells"] == 4
    assert result["inlier_hull_coverage"] == pytest.approx(0.64)
    assert result["positive_depth_ratio"] == 1.0


def test_no_correspondences_have_no_quality():
    assert pose_quality([], [], None, None, 100, 100) == {}
