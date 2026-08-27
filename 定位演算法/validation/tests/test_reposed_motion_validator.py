from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from reposed_motion_validator import (  # noqa: E402
    RePoseDMotionValidator,
    poselib_estimate_options,
    poselib_option_dicts,
    rank_and_cap_matches,
    relative_cam_from_cam,
    rotation_geodesic_deg,
    scale_edm_keypoints,
    translation_direction_delta_deg,
    verify_model_sha256,
)

VALIDATION = Path(__file__).resolve().parents[1]
if str(VALIDATION) not in sys.path:
    sys.path.insert(0, str(VALIDATION))

from eval_reposed_motion_gate import (  # noqa: E402
    evaluate_threshold_grid,
    label_first_limited_jumps,
    promotion_decision,
    select_threshold_tuple,
)


CAMERA = SimpleNamespace(
    model="PINHOLE",
    width=1280,
    height=720,
    params=[931.2057783503649, 931.2057783503649, 640.0, 360.0],
)


class FakeMatcher:
    def __init__(self, mkpts0, mkpts1, mconf):
        self.result = {
            "mkpts0": np.asarray(mkpts0, dtype=float),
            "mkpts1": np.asarray(mkpts1, dtype=float),
            "mconf": np.asarray(mconf, dtype=float),
        }
        self.device = "cpu"

    def match(self, _previous, _current):
        return self.result


class FakeMoGe:
    def __init__(self, depth: torch.Tensor, mask: torch.Tensor):
        self._param = torch.zeros(1)
        self.depth = depth
        self.mask = mask
        self.calls = []

    def parameters(self):
        yield self._param

    def infer(self, images, **kwargs):
        self.calls.append({"images": images, **kwargs})
        return {"depth": self.depth, "mask": self.mask}


class FakePose:
    def __init__(self, rotation, translation):
        self.R = np.asarray(rotation, dtype=float)
        self.t = np.asarray(translation, dtype=float)


class FakeGeometry:
    def __init__(self, rotation, translation, scale):
        self.pose = FakePose(rotation, translation)
        self.scale = scale


def _identity_pair():
    previous = np.eye(4)
    candidate = np.eye(4)
    candidate[:3, 3] = np.array([0.2, 0.0, 0.1], dtype=float)
    return previous, candidate


def _validator(tmp_path: Path, **kwargs) -> RePoseDMotionValidator:
    model = tmp_path / "model.pt"
    model.write_bytes(b"fixture")
    depth = kwargs.pop("depth", None)
    mask = kwargs.pop("mask", None)
    if depth is None:
        depth = torch.ones((2, 720, 1280), dtype=torch.float32)
    if mask is None:
        mask = torch.ones((2, 720, 1280), dtype=torch.float32)
    matcher = kwargs.pop("matcher", None)
    if matcher is None:
        matcher = FakeMatcher(
            [[100.0, 80.0], [200.0, 160.0], [300.0, 240.0]],
            [[110.0, 85.0], [210.0, 165.0], [310.0, 245.0]],
            [0.9, 0.4, 0.8],
        )
    estimate_fn = kwargs.pop("estimate_fn", None)
    if estimate_fn is None:
        previous, candidate = _identity_pair()
        rel = relative_cam_from_cam(previous, candidate)

        def estimate_fn(points1, points2, depths1, depths2, camera1, camera2, ransac, bundle):
            return FakeGeometry(rel[:3, :3], rel[:3, 3], 1.0), {"num_inliers": len(points1)}

    return RePoseDMotionValidator(
        CAMERA,
        matcher,
        model_path=model,
        model_sha256=hashlib.sha256(b"fixture").hexdigest(),
        moge_model=FakeMoGe(depth, mask),
        estimate_fn=estimate_fn,
        device="cpu",
        **kwargs,
    )


def test_scale_edm_keypoints_uses_independent_axes() -> None:
    points = np.array([[1024.0, 576.0], [0.0, 0.0]], dtype=float)
    scaled = scale_edm_keypoints(points, CAMERA)
    np.testing.assert_allclose(scaled, [[1280.0, 720.0], [0.0, 0.0]])


def test_rank_and_cap_matches_is_stable_and_capped() -> None:
    mkpts0 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    mkpts1 = mkpts0 + 1
    mconf = np.array([0.2, 0.9, 0.9, 0.1], dtype=float)
    points0, _points1, conf = rank_and_cap_matches(mkpts0, mkpts1, mconf, 2)
    np.testing.assert_allclose(conf, [0.9, 0.9])
    np.testing.assert_allclose(points0, [[1, 1], [2, 2]])


def test_relative_transform_is_candidate_times_inverse_previous() -> None:
    previous = np.eye(4)
    previous[:3, 3] = [1.0, 2.0, 3.0]
    candidate = np.eye(4)
    candidate[:3, 3] = [1.5, 2.0, 3.0]
    rel = relative_cam_from_cam(previous, candidate)
    np.testing.assert_allclose(rel[:3, 3], [0.5, 0.0, 0.0])
    assert rotation_geodesic_deg(rel[:3, :3], np.eye(3)) == pytest.approx(0.0)


def test_translation_direction_uses_pose_over_scale_and_rejects_near_zero() -> None:
    assert translation_direction_delta_deg(
        np.array([2.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ) == pytest.approx(0.0)
    assert translation_direction_delta_deg(np.zeros(3), np.array([1.0, 0.0, 0.0])) is None


def test_option_dictionaries_are_deterministic() -> None:
    ransac, bundle = poselib_option_dicts()
    assert ransac["max_epipolar_error"] == 2.0
    assert ransac["max_reproj_error"] == 16.0
    assert ransac["min_iterations"] == ransac["max_iterations"] == 1000
    assert ransac["seed"] == 0
    assert ransac["progressive_sampling"] is True
    assert ransac["monodepth_estimate_shift"] is False
    assert ransac["monodepth_weight_sampson"] == 1.0
    assert bundle == {"loss_type": "TRUNCATED_CAUCHY"}
    options = poselib_estimate_options()
    assert options["max_errors"] == [16.0, 2.0]
    assert options["estimate_shift"] is False
    assert options["weight_sampson"] == 1.0


def test_check_filters_invalid_mask_and_depth(tmp_path: Path) -> None:
    depth = torch.ones((2, 720, 1280), dtype=torch.float32)
    mask = torch.ones((2, 720, 1280), dtype=torch.float32)
    # Second correspondence is in EDM pixels (200,160) -> camera (250,200).
    depth[0, 200, 250] = -1.0
    mask[1, 200, 250] = 0.0
    captured = {}

    def estimate_fn(points1, points2, depths1, depths2, camera1, camera2, ransac, bundle):
        captured["points"] = np.asarray(points1)
        captured["depths"] = np.asarray(depths1)
        rel = relative_cam_from_cam(*_identity_pair())
        return FakeGeometry(rel[:3, :3], rel[:3, 3], 1.0), {"num_inliers": len(points1)}

    matcher = FakeMatcher(
        [[100.0, 80.0], [200.0, 160.0], [300.0, 240.0]],
        [[110.0, 85.0], [210.0, 165.0], [310.0, 245.0]],
        [0.5, 0.9, 0.4],
    )
    validator = _validator(
        tmp_path,
        matcher=matcher,
        depth=depth,
        mask=mask,
        estimate_fn=estimate_fn,
        min_inliers=2,
    )
    previous, candidate = _identity_pair()
    check = validator.check(
        np.zeros((720, 1280, 3), np.uint8),
        np.zeros((720, 1280, 3), np.uint8),
        previous,
        candidate,
    )
    assert captured["points"].shape[0] == 2
    assert check.status == "agree"
    assert check.matches == 2


def test_invalid_scale_is_unavailable(tmp_path: Path) -> None:
    def estimate_fn(*_args, **_kwargs):
        return FakeGeometry(np.eye(3), np.array([1.0, 0.0, 0.0]), 0.0), {"num_inliers": 10}

    validator = _validator(tmp_path, estimate_fn=estimate_fn, min_inliers=1)
    previous, candidate = _identity_pair()
    check = validator.check(
        np.zeros((720, 1280, 3), np.uint8),
        np.zeros((720, 1280, 3), np.uint8),
        previous,
        candidate,
    )
    assert check.status == "unavailable"
    assert check.reason == "invalid_scale"


def test_near_zero_translation_is_unavailable(tmp_path: Path) -> None:
    def estimate_fn(*_args, **_kwargs):
        return FakeGeometry(np.eye(3), np.zeros(3), 1.0), {"num_inliers": 10}

    validator = _validator(tmp_path, estimate_fn=estimate_fn, min_inliers=1)
    check = validator.check(
        np.zeros((720, 1280, 3), np.uint8),
        np.zeros((720, 1280, 3), np.uint8),
        np.eye(4),
        np.eye(4),
    )
    assert check.status == "unavailable"
    assert check.reason == "degenerate_translation"


def test_telemetry_is_json_serializable(tmp_path: Path) -> None:
    validator = _validator(tmp_path, min_inliers=1)
    previous, candidate = _identity_pair()
    check = validator.check(
        np.zeros((720, 1280, 3), np.uint8),
        np.zeros((720, 1280, 3), np.uint8),
        previous,
        candidate,
    )
    payload = json.dumps(check.as_json_dict())
    assert "agree" in payload


def test_hash_mismatch_is_detected_before_model_use(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_model_sha256(path, "0" * 64)


def test_label_first_limited_jumps_ignores_tail_and_quality() -> None:
    rows = [
        {
            "source_index": 0,
            "success": False,
            "rejected": "limited_jump_unconfirmed",
            "limited_jump": {"confirmation_model": None, "confirmation_residual": None},
            "limited_jump_confirmed": False,
        },
        {
            "source_index": 1,
            "success": True,
            "rejected": None,
            "limited_jump": {
                "confirmation_model": "stationary",
                "confirmation_residual": 0.001,
            },
            "limited_jump_confirmed": True,
        },
        {
            "source_index": 2,
            "success": False,
            "rejected": "limited_jump_unconfirmed",
            "limited_jump": {"confirmation_model": None, "confirmation_residual": None},
            "limited_jump_confirmed": False,
        },
        {
            "source_index": 3,
            "success": False,
            "rejected": "reprojection",
            "limited_jump": None,
            "limited_jump_confirmed": False,
        },
        {
            "source_index": 4,
            "success": False,
            "rejected": "limited_jump_unconfirmed",
            "limited_jump": {"confirmation_model": None, "confirmation_residual": None},
            "limited_jump_confirmed": False,
        },
        {
            "source_index": 5,
            "success": False,
            "rejected": "limited_jump_unconfirmed",
            "limited_jump": {
                "confirmation_model": "stationary",
                "confirmation_residual": 0.2,
                "confirmation_limit": 0.05,
            },
            "limited_jump_confirmed": False,
        },
        {
            "source_index": 6,
            "success": False,
            "rejected": "limited_jump_unconfirmed",
            "limited_jump": {"confirmation_model": None, "confirmation_residual": None},
            "limited_jump_confirmed": False,
        },
    ]
    labeled = label_first_limited_jumps(rows)
    assert [(item["source_index"], item["label"]) for item in labeled] == [
        (0, "confirmed"),
        (4, "rejected"),
    ]


def test_threshold_search_is_deterministic_and_refuses_insufficient_evidence() -> None:
    events = [
        {
            "label": "confirmed",
            "inliers": 90,
            "rotation_delta_deg": 2.0,
            "translation_direction_delta_deg": 8.0,
            "total_ms": 40.0,
        },
        {
            "label": "rejected",
            "inliers": 40,
            "rotation_delta_deg": 6.0,
            "translation_direction_delta_deg": 30.0,
            "total_ms": 50.0,
        },
    ]
    selected = select_threshold_tuple(evaluate_threshold_grid(events))
    assert selected["min_inliers"] == 30
    assert selected["max_rotation_delta_deg"] == 2
    assert selected["max_translation_direction_delta_deg"] == 10
    assert selected["true_fast_confirms"] == 1
    assert selected["false_fast_confirms"] == 0
    empty = select_threshold_tuple(evaluate_threshold_grid([events[1]]))
    assert empty is None


def test_promotion_requires_enough_heldout_evidence() -> None:
    decision = promotion_decision(
        heldout_labeled=19,
        false_fast_confirms=0,
        true_fast_confirms=5,
        quality_regressions=[],
        rejected_pose_introductions=0,
        throughput_ratio=1.0,
        p95_validator_ms=20.0,
        peak_gpu_fraction=0.4,
        cuda_oom=False,
        model_fallback=False,
    )
    assert decision["promotion_eligible"] is False
    assert "heldout_labeled_events" in decision["failed_gates"]
