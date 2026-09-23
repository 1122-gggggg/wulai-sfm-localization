from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


VALIDATION_DIR = Path(__file__).resolve().parents[3] / "定位演算法" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from sta_map_anchored import (  # noqa: E402
    Hypothesis,
    camera_center,
    compose_query_pose,
    estimate_gamma,
    pose_consensus,
    rotation_angle_deg,
    sta_crop,
)


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _pose(rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([_rotation(rng), rng.normal(size=3)])


def test_compose_recovers_query_pose_from_scaled_relative_pose():
    rng = np.random.default_rng(3)
    for _ in range(20):
        ref, query = _pose(rng), _pose(rng)
        rel_R = ref[:, :3] @ query[:, :3].T
        rel_t = ref[:, 3] - rel_R @ query[:, 3]
        gamma = float(rng.uniform(0.2, 5.0))
        sta_T = np.eye(4)
        sta_T[:3, :3] = rel_R
        sta_T[:3, 3] = rel_t / gamma

        estimate = compose_query_pose(ref, sta_T, gamma)

        np.testing.assert_allclose(estimate, query, atol=1e-9)


def test_query_center_is_translation_seen_from_reference():
    ref = np.column_stack([np.eye(3), -np.array([10.0, 5.0, 2.0])])
    sta_T = np.eye(4)
    sta_T[:3, 3] = [0.2, -0.1, 0.05]

    center = camera_center(compose_query_pose(ref, sta_T, 2.8))

    np.testing.assert_allclose(center, [10.56, 4.72, 2.14], atol=1e-12)


def test_sta_crop_keeps_centre_and_central_square_for_anafi_frames():
    for width, height in ((2688, 1512), (1280, 720)):
        crop = sta_crop(width, height)
        centre = crop.to_crop(np.array([[width / 2, height / 2]]))[0]
        assert np.all(np.abs(centre - 112.0) <= 1.0)
        left_edge, right_edge = crop.to_crop(
            np.array([[crop.left + (crop.left2) / crop.scale_x, height / 2],
                      [crop.left + (crop.left2 + 224) / crop.scale_x, height / 2]])
        )[:, 0]
        assert abs(left_edge) < 1e-9 and abs(right_edge - 224.0) < 1e-9
        top_edge = crop.to_crop(np.array([[width / 2, crop.top]]))[0, 1]
        assert abs(top_edge) < 1e-9


def test_gamma_is_robust_to_outlier_depths():
    rng = np.random.default_rng(5)
    sta = rng.uniform(1.0, 10.0, size=400)
    true = 3.0 * sta * np.exp(rng.normal(scale=0.03, size=sta.size))
    true[:120] = rng.uniform(0.1, 100.0, size=120)

    fit = estimate_gamma(true, sta, np.ones_like(sta), min_samples=30)

    assert fit.samples == 400
    assert abs(fit.gamma - 3.0) < 0.05
    assert fit.log_mad < 0.1


def test_gamma_abstains_below_minimum_samples_and_drops_invalid_depths():
    fit = estimate_gamma(
        np.array([1.0, -1.0, np.nan, 2.0]), np.array([1.0, 1.0, 1.0, 0.0]), np.ones(4), min_samples=2
    )
    assert fit.gamma is None and fit.samples == 1


def _hypothesis(name, video, frame, center, yaw_deg, conf=0.9):
    angle = np.radians(yaw_deg)
    world_R_cam = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    cam_R_world = world_R_cam.T
    pose = np.column_stack([cam_R_world, -cam_R_world @ np.asarray(center, dtype=float)])
    return Hypothesis(name, video, frame, pose, conf)


def test_consensus_rejects_outlier_and_averages_independent_agreement():
    hypotheses = [
        _hypothesis("a", "v1", 0, [1.00, 0.0, 0.0], 10.0),
        _hypothesis("b", "v2", 0, [1.04, 0.0, 0.0], 12.0),
        _hypothesis("c", "v2", 24, [1.02, 0.02, 0.0], 11.0),
        _hypothesis("far", "v3", 0, [5.0, 0.0, 0.0], 11.0),
    ]

    result = pose_consensus(hypotheses, max_position=0.1, max_rotation_deg=5.0, min_frame_gap=72)

    assert result.accepted
    assert set(result.members) == {"a", "b", "c"}
    np.testing.assert_allclose(camera_center(result.cam_from_world), [1.02, 0.02 / 3, 0.0], atol=1e-9)
    reference = _hypothesis("ref", "v", 0, [0, 0, 0], 11.0).cam_from_world[:3, :3]
    assert rotation_angle_deg(result.cam_from_world[:3, :3], reference) < 1.0


def test_consensus_requires_an_independent_pair():
    adjacent = [
        _hypothesis("a", "v1", 0, [1.0, 0.0, 0.0], 10.0),
        _hypothesis("b", "v1", 24, [1.01, 0.0, 0.0], 10.0),
    ]
    result = pose_consensus(adjacent, max_position=0.1, max_rotation_deg=5.0, min_frame_gap=72)
    assert not result.accepted and result.reason == "no independent agreeing pair"

    single = pose_consensus(adjacent[:1], max_position=0.1, max_rotation_deg=5.0, min_frame_gap=72)
    assert not single.accepted
