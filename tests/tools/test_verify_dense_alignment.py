"""A dense cloud passes only when it reproduces the keyframes' observed depths."""

import numpy as np

import verify_dense_alignment as tool


def _view():
    rng = np.random.default_rng(0)
    observed = np.column_stack(
        [rng.uniform(-1, 1, 400), rng.uniform(-0.6, 0.6, 400), np.full(400, 5.0)]
    )
    return {
        "R": np.eye(3),
        "t": np.zeros(3),
        "K": (320.0, 320.0, 320.0, 180.0),
        "size": (640, 360),
        "X": observed,
    }


def _plane(z, step=0.004):
    xs, ys = np.meshgrid(np.arange(-1.2, 1.2, step), np.arange(-0.8, 0.8, step))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])


def test_cloud_on_the_observed_surface_agrees():
    fraction, observations = tool.depth_agreement(_plane(5.0), [_view()])
    assert observations == 400
    assert fraction > 0.95


def test_cloud_displaced_along_the_viewing_axis_is_rejected():
    fraction, _ = tool.depth_agreement(_plane(5.0 + 1.0), [_view()])
    assert fraction == 0.0


def test_cloud_missing_the_scene_cannot_pass_by_covering_nothing():
    far_away = _plane(5.0) + np.array([50.0, 0.0, 0.0])
    fraction, _ = tool.depth_agreement(far_away, [_view()])
    assert fraction == 0.0


def test_nearest_surface_wins_over_a_point_behind_it():
    cloud = np.vstack([_plane(5.0), _plane(9.0)])
    fraction, _ = tool.depth_agreement(cloud, [_view()])
    assert fraction > 0.95
