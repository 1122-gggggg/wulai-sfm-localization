"""Unit tests for pure depth sampling, scale fitting, lifting, and candidate row matching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from workmap_depth import (
    DEPTH_SCALE_MAX_SPREAD,
    MoGeDepthEstimator,
    fit_depth_scale,
    lift_pixels,
    lift_request,
    match_promotable_rows,
    require_moge,
    sample_depth,
)


def test_depth_scale_fit_rejects_inconsistent_depth():
    rng = np.random.default_rng(42)
    n = 100
    z_map = rng.uniform(1.0, 10.0, size=n)
    true_scale = 7.5
    pred_clean = z_map / true_scale

    # 2% noise + 10% outliers -> error < 1%
    noise_2pct = 1.0 + rng.uniform(-0.02, 0.02, size=n)
    pred_noisy = pred_clean * noise_2pct
    outlier_indices = rng.choice(n, size=10, replace=False)
    pred_noisy[outlier_indices] = rng.uniform(0.1, 20.0, size=10)

    scale, spread, count = fit_depth_scale(z_map, pred_noisy)
    assert scale is not None
    assert abs(scale - true_scale) / true_scale < 0.01
    assert count == n
    assert spread <= DEPTH_SCALE_MAX_SPREAD

    # 30% noise -> spread exceeds threshold -> returns None
    noise_30pct = 1.0 + rng.uniform(-0.30, 0.30, size=n)
    pred_inconsistent = pred_clean * noise_30pct
    bad_scale, bad_spread, _ = fit_depth_scale(z_map, pred_inconsistent)
    assert bad_scale is None
    assert bad_spread > DEPTH_SCALE_MAX_SPREAD


def test_sample_depth_rejects_depth_edges():
    # Step depth edge in 60x60 image: left half 2.0, right half 5.0
    depth = np.full((60, 60), 2.0, dtype=np.float32)
    depth[:, 30:] = 5.0

    # At x=30, y=30, the 5x5 window (x in [28..32]) crosses the step edge
    # (5 - 2) / median > DEPTH_EDGE_MAX_REL_SPREAD -> NaN
    edge_xy = np.array([[30.0, 30.0]], dtype=np.float64)
    edge_sampled = sample_depth(depth, edge_xy, radius=2)
    assert np.isnan(edge_sampled[0])

    # On flat plane, window is purely 2.0 -> returns median (2.0)
    flat_xy = np.array([[15.0, 15.0]], dtype=np.float64)
    flat_sampled = sample_depth(depth, flat_xy, radius=2)
    assert flat_sampled[0] == pytest.approx(2.0)

    # Out of boundary window -> NaN
    out_of_bounds = np.array([[0.0, 0.0], [59.0, 59.0]], dtype=np.float64)
    oob_sampled = sample_depth(depth, out_of_bounds, radius=2)
    assert np.isnan(oob_sampled).all()


def test_lift_request_prefers_frozen_points_for_scale():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_from_world = np.eye(4, dtype=np.float64)[:3, :4]
    depth_map = np.full((480, 640), 2.0, dtype=np.float32)

    # 30 positive IDs at true scale 7.5
    pos_ids = np.arange(1, 31, dtype=np.int64)
    pos_xy = np.column_stack([np.linspace(50, 590, 30), np.linspace(50, 430, 30)])
    pos_z = 7.5 * 2.0
    pos_xyz = lift_pixels(pos_xy, np.full(30, pos_z), K, cam_from_world)

    # 30 negative IDs at scale 9.0
    neg_ids = -np.arange(1, 31, dtype=np.int64)
    neg_xy = np.column_stack([np.linspace(60, 580, 30), np.linspace(60, 420, 30)])
    neg_z = 9.0 * 2.0
    neg_xyz = lift_pixels(neg_xy, np.full(30, neg_z), K, cam_from_world)

    seed_xy = np.array([[320.0, 240.0]], dtype=np.float64)

    req = {
        "source_epoch": 1,
        "bid": 42,
        "K": K,
        "cam_from_world": cam_from_world,
        "seed_xy": seed_xy,
        "ref_ids": np.concatenate([pos_ids, neg_ids]),
        "ref_xy": np.vstack([pos_xy, neg_xy]),
        "ref_xyz": np.vstack([pos_xyz, neg_xyz]),
    }

    # Case 1: Both positive and negative IDs -> prefers frozen (scale 7.5)
    res_frozen = lift_request(req, depth_map, 10.0)
    assert res_frozen["ok"] is True
    assert res_frozen["scale_source"] == "frozen"
    assert res_frozen["scale"] == pytest.approx(7.5, rel=1e-4)
    assert res_frozen["valid"][0]
    # Center pixel (320, 240) has ray [0, 0, 1], depth 15.0 -> world [0, 0, 15]
    np.testing.assert_allclose(res_frozen["xyz"][0], [0.0, 0.0, 15.0], atol=1e-6)

    # Case 2: Only negative IDs -> falls back to session (scale 9.0)
    req_session = dict(req)
    req_session["ref_ids"] = neg_ids
    req_session["ref_xy"] = neg_xy
    req_session["ref_xyz"] = neg_xyz
    res_session = lift_request(req_session, depth_map, 10.0)
    assert res_session["ok"] is True
    assert res_session["scale_source"] == "session"
    assert res_session["scale"] == pytest.approx(9.0, rel=1e-4)

    # Case 3: Insufficient points (< 20) -> ok=False, reason="scale_support"
    req_sparse = dict(req)
    req_sparse["ref_ids"] = pos_ids[:10]
    req_sparse["ref_xy"] = pos_xy[:10]
    req_sparse["ref_xyz"] = pos_xyz[:10]
    res_sparse = lift_request(req_sparse, depth_map, 10.0)
    assert res_sparse["ok"] is False
    assert res_sparse["reason"] == "scale_support"


def test_depth_rows_promote_only_when_tracked_pixel_agrees():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_from_world = np.eye(4, dtype=np.float64)[:3, :4]

    seed_xy = np.array([[320.0, 240.0], [200.0, 150.0], [100.0, 100.0]], dtype=np.float64)
    xyz_0 = np.array([0.0, 0.0, 10.0])
    xyz_1 = lift_pixels(seed_xy[1:2], [10.0], K, cam_from_world)[0]
    xyz_2 = lift_pixels(seed_xy[2:3], [10.0], K, cam_from_world)[0]

    result = {
        "ok": True,
        "seed_xy": seed_xy,
        "xyz": np.vstack([xyz_0, xyz_1, xyz_2]),
        "valid": np.array([True, True, True]),
    }

    # Row 0: tracked perfectly at (320, 240) -> matches
    # Row 1: tracked pixel shifted by 5 px -> reprojection error > 2.0 px -> rejected
    # Row 2: disappeared / seed (500, 500) absent in result -> unmatched
    tracker_seed_xy = np.array([[320.0, 240.0], [200.0, 150.0], [500.0, 500.0]], dtype=np.float64)
    tracker_tracked_xy = np.array(
        [[320.0, 240.0], [205.0, 150.0], [500.0, 500.0]], dtype=np.float64
    )

    mask, promoted_xyz = match_promotable_rows(
        result, tracker_seed_xy, tracker_tracked_xy, cam_from_world, K, max_reproj_px=2.0
    )

    assert mask[0]
    assert not mask[1]
    assert not mask[2]
    np.testing.assert_allclose(promoted_xyz[0], xyz_0, atol=1e-8)
    assert np.isnan(promoted_xyz[1]).all()
    assert np.isnan(promoted_xyz[2]).all()


def test_require_moge_raises_meaningful_error():
    with patch.dict("sys.modules", {"moge.model.v2": None}):
        with pytest.raises(RuntimeError, match="requirements/optional-workmap-depth-lock.txt"):
            require_moge()


def test_estimator_requires_load_before_infer():
    estimator = MoGeDepthEstimator(Path("/dummy/path.pt"))
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)
    with pytest.raises(RuntimeError, match="not loaded"):
        estimator.infer_depth(rgb, K)
