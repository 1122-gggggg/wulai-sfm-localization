from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


VALIDATION_DIR = Path(__file__).resolve().parents[3] / "定位演算法" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from pose_window_experiment import (  # noqa: E402
    PoseWindowOptimizer,
    _reprojection_residual_loop,
    _reprojection_residual_vectorized,
)


def _project(pose: np.ndarray, points: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = (pose[:, :3] @ points.T + pose[:, 3:4]).T
    return np.column_stack(
        [
            K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2],
            K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2],
        ]
    )


def _frame(
    stamp: float,
    epoch: int,
    pose: np.ndarray,
    ids: np.ndarray,
    points: np.ndarray,
    K: np.ndarray,
    *,
    truth_index: int = 1,
):
    return {
        "stamp": stamp,
        "epoch": epoch,
        "pose": pose,
        "ids": ids,
        "xy": _project(pose=_TRUE_POSES[truth_index], points=points, K=K),
        "xyz": points,
        "K": K,
        # The prototype must never inspect this field. It represents the root's
        # separate evaluation split and is intentionally not part of its API.
        "holdout": object(),
    }


_K = np.array([[220.0, 0.0, 320.0], [0.0, 220.0, 240.0], [0.0, 0.0, 1.0]])
_TRAIN_POINTS = np.array(
    [
        [-0.9, -0.5, 4.0],
        [0.7, -0.4, 4.5],
        [-0.6, 0.6, 5.0],
        [0.8, 0.7, 5.5],
        [-0.2, 0.1, 6.2],
        [0.2, -0.7, 5.8],
    ],
    dtype=float,
)
_HOLDOUT_POINTS = np.array([[-0.4, 0.25, 4.8], [0.55, 0.3, 5.2]], dtype=float)
_IDS = np.arange(10, 10 + len(_TRAIN_POINTS), dtype=np.int64)


def _pose(tx: float, ty: float, tz: float) -> np.ndarray:
    return np.column_stack((np.eye(3), np.array([tx, ty, tz], dtype=float)))


_TRUE_POSES = {
    1: _pose(0.00, 0.00, 0.00),
    2: _pose(0.04, -0.01, 0.02),
    3: _pose(0.08, -0.01, 0.03),
    4: _pose(0.12, 0.00, 0.04),
    5: _pose(0.16, 0.01, 0.05),
}


def test_shared_train_tracks_reduce_hidden_holdout_error() -> None:
    optimizer = PoseWindowOptimizer(window_size=5, max_nfev=5, max_points=80)
    outputs = []
    for index in range(1, 6):
        # The input pose is deliberately biased.  Observations are generated
        # from the true pose, while only train IDs/points enter the optimizer.
        input_pose = _TRUE_POSES[index].copy()
        input_pose[:, 3] += np.array([0.06, -0.04, 0.03])
        outputs.append(
            optimizer.update(
                _frame(
                    1.0 + 0.04 * (index - 1),
                    7,
                    input_pose,
                    _IDS,
                    _TRAIN_POINTS,
                    _K,
                    truth_index=index,
                )
            )
        )

    result = outputs[-1]
    assert result["success"], result
    assert result["observation_count"] == len(_TRAIN_POINTS) * 5
    estimated = result["pose"]
    assert estimated is not None
    truth_xy = _project(_TRUE_POSES[5], _HOLDOUT_POINTS, _K)
    baseline_xy = _project(
        _TRUE_POSES[5] + np.column_stack((np.zeros((3, 3)), np.array([0.06, -0.04, 0.03]))),
        _HOLDOUT_POINTS,
        _K,
    )
    optimized_xy = _project(estimated, _HOLDOUT_POINTS, _K)
    baseline_error = float(np.mean(np.linalg.norm(baseline_xy - truth_xy, axis=1)))
    optimized_error = float(np.mean(np.linalg.norm(optimized_xy - truth_xy, axis=1)))
    assert optimized_error < baseline_error * 0.8, (baseline_error, optimized_error, result)


def test_epoch_gap_reset_and_bounded_window() -> None:
    optimizer = PoseWindowOptimizer(window_size=5, max_nfev=2, max_points=3)
    pose = _TRUE_POSES[1]
    first = optimizer.update(_frame(1.0, 1, pose, _IDS, _TRAIN_POINTS, _K))
    assert first["reset"] is False
    for index in range(2, 8):
        result = optimizer.update(_frame(float(index), 1, pose, _IDS, _TRAIN_POINTS, _K))
        assert result["observation_count"] <= 5 * 3
        assert len(optimizer._frames) <= 5

    epoch_reset = optimizer.update(_frame(8.0, 2, pose, _IDS, _TRAIN_POINTS, _K))
    assert epoch_reset["reset"] is True
    assert len(optimizer._frames) == 1

    optimizer.update(_frame(8.1, 2, pose, _IDS, _TRAIN_POINTS, _K))
    gap_reset = optimizer.update(_frame(9.0, 2, pose, _IDS, _TRAIN_POINTS, _K))
    assert gap_reset["reset"] is True
    assert len(optimizer._frames) == 1


def test_invalid_pose_resets_without_using_future_or_holdout() -> None:
    optimizer = PoseWindowOptimizer(window_size=5, max_nfev=2, max_points=4)
    valid = optimizer.update(_frame(1.0, 3, _TRUE_POSES[1], _IDS, _TRAIN_POINTS, _K))
    assert valid["reset"] is False
    invalid = dict(_frame(1.1, 3, _TRUE_POSES[1], _IDS, _TRAIN_POINTS, _K))
    invalid["pose"] = None
    result = optimizer.update(invalid)
    assert result == {
        "pose": None,
        "success": False,
        "reset": True,
        "nfev": 0,
        "cost": None,
        "observation_count": 0,
        "optimized": False,
        "window_frames": 0,
    }
    assert len(optimizer._frames) == 0


def test_vectorized_reprojection_matches_reference_loop() -> None:
    rotations = np.stack([np.eye(3), np.eye(3)])
    translations = np.array([[0.0, 0.0, 0.0], [0.1, -0.02, 0.03]])
    points = np.array([[-0.2, 0.1, 4.0], [0.3, -0.4, 5.0], [0.0, 0.0, -1.0]])
    frame_indices = np.array([0, 1, 1, 0], dtype=np.int64)
    point_indices = np.array([0, 0, 1, 2], dtype=np.int64)
    uv = np.array([[310.0, 245.0], [305.0, 244.0], [332.0, 223.0], [0.0, 0.0]])
    cameras = np.stack([_K, _K, _K, _K])
    reference = _reprojection_residual_loop(
        rotations, translations, points, frame_indices, point_indices, uv, cameras
    )
    vectorized = _reprojection_residual_vectorized(
        rotations, translations, points, frame_indices, point_indices, uv, cameras
    )
    np.testing.assert_allclose(vectorized, reference, rtol=0.0, atol=1e-12)
