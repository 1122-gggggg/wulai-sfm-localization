"""Tests for spatio-temporal query and frozen region coverage in workmap expansion."""

from __future__ import annotations

import numpy as np

from workmap_query import (
    FrozenCoverage,
    SpatioTemporalQuery,
    _unpack_cells,
    cell_indices,
    pack_cells,
)


def test_frozen_coverage_marks_dense_cells_and_neighbours_only():
    """A cell with >=3 points marks itself and all 26 neighbors; sparse or distant cells are not."""
    cell_size = 0.08
    # Cell A at (0, 0, 0) with 3 points
    p1 = np.array([0.01, 0.01, 0.01])
    p2 = np.array([0.02, 0.02, 0.02])
    p3 = np.array([0.03, 0.03, 0.03])
    # Cell B at (10, 10, 10) with only 2 points
    p4 = np.array([10.0 * cell_size + 0.01, 10.0 * cell_size + 0.01, 10.0 * cell_size + 0.01])
    p5 = np.array([10.0 * cell_size + 0.02, 10.0 * cell_size + 0.02, 10.0 * cell_size + 0.02])

    pts = np.vstack([p1, p2, p3, p4, p5])
    cov = FrozenCoverage(pts, cell_size=cell_size, min_points=3)
    assert cov.cell_count == 1

    # Point in Cell A
    p_center = np.array([[0.04, 0.04, 0.04]])
    assert cov.covered(p_center)[0]

    # All 26 neighbors of Cell A are covered
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                p_neighbor = np.array(
                    [[(dx + 0.5) * cell_size, (dy + 0.5) * cell_size, (dz + 0.5) * cell_size]]
                )
                assert cov.covered(p_neighbor)[0], f"Offset ({dx}, {dy}, {dz}) should be covered"

    # Point separated by 2 cells from Cell A is NOT covered
    p_two_away = np.array([[2.5 * cell_size, 0.5 * cell_size, 0.5 * cell_size]])
    assert not cov.covered(p_two_away)[0]

    # Point in Cell B (which had only 2 points) is NOT covered
    p_in_cell_b = np.array(
        [[10.0 * cell_size + 0.04, 10.0 * cell_size + 0.04, 10.0 * cell_size + 0.04]]
    )
    assert not cov.covered(p_in_cell_b)[0]

    # Non-finite rows return True (fail-closed)
    p_nan = np.array([[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [-np.inf, 0.0, 0.0]])
    np.testing.assert_array_equal(cov.covered(p_nan), [True, True, True])

    # Empty queries
    assert len(cov.covered(np.empty((0, 3)))) == 0

    # Empty coverage
    empty_cov = FrozenCoverage(np.empty((0, 3)))
    assert empty_cov.cell_count == 0
    assert not empty_cov.covered(np.array([[1.0, 2.0, 3.0]]))[0]
    assert empty_cov.covered(np.array([[np.nan, 2.0, 3.0]]))[0]


def test_query_returns_visible_session_points_only():
    """Candidates must be visible, in front of the camera, not excluded, and have un-evicted source images."""
    query = SpatioTemporalQuery(960, 540, cell_size=0.08, image_cache=1)
    K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_from_world = np.eye(3, 4, dtype=np.float64)

    # Landmarks
    x_front_1 = np.array([0.0, 0.0, 5.0])  # In front: u=480, v=270
    x_back = np.array([0.0, 0.0, -5.0])  # Behind camera
    x_ex = np.array([0.1, 0.0, 5.0])  # To be excluded
    x_evicted = np.array([-0.1, 0.0, 5.0])  # In KF1 which will be evicted
    x_front_2 = np.array([0.2, 0.1, 4.0])  # In front: u=520, v=290

    # Keyframe 1: holds -4
    img1 = np.zeros((540, 960), dtype=np.uint8)
    query.add_keyframe(
        "kf_00000001",
        img1,
        cam_from_world,
        np.array([-4]),
        np.array([[400.0, 200.0]]),
    )
    assert query.image("kf_00000001") is not None

    # Keyframe 2: holds -1, -2, -3, -5
    img2 = np.zeros((540, 960), dtype=np.uint8)
    query.add_keyframe(
        "kf_00000002",
        img2,
        cam_from_world,
        np.array([-1, -2, -3, -5]),
        np.array([[480.0, 270.0], [480.0, 270.0], [496.0, 270.0], [520.0, 290.0]]),
    )

    # image_cache=1 evicted KF1
    assert query.image("kf_00000001") is None
    assert query.image("kf_00000002") is not None

    # Upsert landmarks (also include positive landmark 10 which must be rejected)
    query.upsert_landmarks(
        [-1, -2, -3, -4, -5, 10],
        np.array([x_front_1, x_back, x_ex, x_evicted, x_front_2, [0.0, 0.0, 5.0]]),
    )

    cands = query.candidates(
        cam_from_world,
        K,
        exclude_ids=np.array([-3]),
        occupied_xy=np.empty((0, 2)),
        min_distance_px=8,
    )

    # -1 and -5 retained; -2 (behind), -3 (excluded), -4 (evicted), 10 (positive) omitted
    assert set(cands.ids) == {-1, -5}

    idx1 = int(np.where(cands.ids == -1)[0][0])
    idx5 = int(np.where(cands.ids == -5)[0][0])
    np.testing.assert_allclose(cands.predicted_xy[idx1], [480.0, 270.0], atol=1e-5)
    np.testing.assert_allclose(cands.predicted_xy[idx5], [520.0, 290.0], atol=1e-5)
    assert cands.source_names[idx1] == "kf_00000002"
    assert cands.source_names[idx5] == "kf_00000002"


def test_query_skips_sources_seen_from_a_different_viewpoint():
    """Candidates observed from >20 deg ray angle are excluded; <=20 deg are retained."""
    query = SpatioTemporalQuery(960, 540, cell_size=0.08, image_cache=10)
    K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_from_world = np.eye(3, 4, dtype=np.float64)

    x1 = np.array([0.0, 0.0, 10.0])
    x2 = np.array([0.0, 0.5, 10.0])
    d = 10.0

    # Source KF 1: 15 deg angle from X1
    ang1 = np.radians(15.0)
    c1 = x1 - d * np.array([np.sin(ang1), 0.0, np.cos(ang1)])
    cam1 = np.column_stack((np.eye(3), -c1))

    # Source KF 2: 40 deg angle from X2
    ang2 = np.radians(40.0)
    c2 = x2 - d * np.array([np.sin(ang2), 0.0, np.cos(ang2)])
    cam2 = np.column_stack((np.eye(3), -c2))

    img = np.zeros((540, 960), dtype=np.uint8)
    query.add_keyframe("kf_00000001", img, cam1, np.array([-1]), np.array([[480.0, 270.0]]))
    query.add_keyframe("kf_00000002", img, cam2, np.array([-2]), np.array([[480.0, 270.0]]))
    query.upsert_landmarks([-1, -2], np.array([x1, x2]))

    cands = query.candidates(
        cam_from_world,
        K,
        exclude_ids=np.empty((0,), dtype=np.int64),
        occupied_xy=np.empty((0, 2)),
        min_distance_px=8,
    )

    assert -1 in cands.ids
    assert -2 not in cands.ids


def test_temporal_buffer_prefers_recent_reference_keyframes():
    """Temporal buffer prioritizes candidates from the most recent active keyframe."""
    query = SpatioTemporalQuery(960, 540, cell_size=0.08, image_cache=10)
    K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = np.eye(3, 4, dtype=np.float64)

    # KF 1 with landmarks -1, -2
    x1 = np.array([-0.2, 0.0, 5.0])
    x2 = np.array([-0.1, 0.0, 5.0])
    img1 = np.zeros((540, 960), dtype=np.uint8)
    query.add_keyframe(
        "kf_00000001",
        img1,
        cam,
        np.array([-1, -2]),
        np.array([[448.0, 270.0], [464.0, 270.0]]),
    )

    # KF 2 with landmarks -3, -4
    x3 = np.array([0.1, 0.0, 5.0])
    x4 = np.array([0.2, 0.0, 5.0])
    img2 = np.zeros((540, 960), dtype=np.uint8)
    query.add_keyframe(
        "kf_00000002",
        img2,
        cam,
        np.array([-3, -4]),
        np.array([[496.0, 270.0], [512.0, 270.0]]),
    )

    query.upsert_landmarks([-1, -2, -3, -4], np.array([x1, x2, x3, x4]))

    # Note frame with inliers from KF 2
    query.note_frame([-3, -4])

    # Without final spatial subsampling, temporal candidates lead the returned pool.
    cands = query.candidates(
        cam,
        K,
        exclude_ids=np.empty((0,), dtype=np.int64),
        occupied_xy=np.empty((0, 2)),
        min_distance_px=8,
        max_points=4,
    )
    np.testing.assert_array_equal(cands.ids[:2], [-3, -4])

    # Test tie-breaking: equal inlier counts select the lexicographically larger name
    query.note_frame([-1, -3])
    assert query._temporal[-1] == "kf_00000002"

    # Clearing temporal buffer resets temporal preference
    query.clear_temporal()
    assert len(query._temporal) == 0


def test_cell_indices_and_pack_cells_roundtrip():
    """Cell packing and unpacking handles negative indices and boundary values correctly."""
    coords = np.array([[0, 0, 0], [10, -5, 20], [-100, 50, -200]], dtype=np.int64)
    packed = pack_cells(coords)
    unpacked = _unpack_cells(packed)
    np.testing.assert_array_equal(coords, unpacked)

    # Empty inputs
    assert len(cell_indices(np.empty((0, 3)))) == 0
    assert len(pack_cells(np.empty((0, 3), dtype=np.int64))) == 0
    assert len(_unpack_cells(np.empty((0,), dtype=np.int64))) == 0


def test_query_deduplicates_against_occupied_xy():
    """Candidate within min_distance_px of occupied_xy is rejected by mask erosion."""
    query = SpatioTemporalQuery(960, 540, cell_size=0.08, image_cache=10)
    K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = np.eye(3, 4, dtype=np.float64)

    # -1 projects to (480, 270), -2 projects to (600, 270)
    x1 = np.array([0.0, 0.0, 5.0])
    x2 = np.array([0.75, 0.0, 5.0])
    img = np.zeros((540, 960), dtype=np.uint8)
    query.add_keyframe(
        "kf_00000001",
        img,
        cam,
        np.array([-1, -2]),
        np.array([[480.0, 270.0], [600.0, 270.0]]),
    )
    query.upsert_landmarks([-1, -2], np.array([x1, x2]))

    # Occupied at (481, 270) within 8 px of -1
    occupied_xy = np.array([[481.0, 270.0]])
    cands = query.candidates(
        cam,
        K,
        exclude_ids=np.empty((0,), dtype=np.int64),
        occupied_xy=occupied_xy,
        min_distance_px=8,
    )

    assert -1 not in cands.ids
    assert -2 in cands.ids


def test_spatial_pool_buckets_and_limits_occluded_cells():
    """Cells projecting to the same grid bucket are limited to nearest QUERY_CELLS_PER_BUCKET."""
    query = SpatioTemporalQuery(960, 540, cell_size=0.08, image_cache=10)
    K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = np.eye(3, 4, dtype=np.float64)

    # 3 points along the same ray at depths 2, 4, 8; all project to (480, 270)
    x1 = np.array([0.0, 0.0, 2.0])
    x2 = np.array([0.0, 0.0, 4.0])
    x3 = np.array([0.0, 0.0, 8.0])

    img = np.zeros((540, 960), dtype=np.uint8)
    query.add_keyframe(
        "kf_00000001",
        img,
        cam,
        np.array([-1, -2, -3]),
        np.array([[480.0, 270.0], [480.0, 270.0], [480.0, 270.0]]),
    )
    query.upsert_landmarks([-1, -2, -3], np.array([x1, x2, x3]))

    cands = query.candidates(cam, K, np.empty(0, np.int64), np.empty((0, 2)), 8)
    assert set(cands.ids) == {-1, -2}
