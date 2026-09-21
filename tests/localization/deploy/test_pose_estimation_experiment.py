from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np


VALIDATION_DIR = Path(__file__).resolve().parents[3] / "定位演算法" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import pose_estimation_experiment as experiment  # noqa: E402


def test_projection_errors_marks_nonpositive_depth_as_failed() -> None:
    pose = np.column_stack((np.eye(3), np.zeros(3)))
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    xyz = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 4.0], [0.0, 0.0, -1.0]])
    xy = np.array([[50.0, 50.0], [75.0, 50.0], [0.0, 0.0]])

    errors = experiment.projection_errors(pose, xyz, xy, K)

    np.testing.assert_allclose(errors[:2], [0.0, 0.0])
    assert errors[2] == 1e6


def test_capture_stamp_rejects_invalid_fps_and_repairs_backward_pts() -> None:
    with np.testing.assert_raises(ValueError):
        experiment.capture_stamp(0.0, 0.0, 0, 0.0)

    stamp, fallback = experiment.capture_stamp(0.5, 1.0, 1, 10.0)
    assert stamp == 1.1
    assert fallback is True

    stamp, fallback = experiment.capture_stamp(1.2, 1.0, 1, 10.0)
    assert stamp == 1.2
    assert fallback is False


def test_validate_capture_rejects_offset_shape_and_timestamp_mismatch() -> None:
    rows = [{"stamp": 1.0}, {"stamp": 1.1}]
    offsets = np.array([0, 1, 2], dtype=np.int64)
    xy = np.zeros((2, 2), dtype=float)
    xyz = np.zeros((2, 3), dtype=float)
    ids = np.array([1, 2], dtype=np.int64)
    K = np.eye(3)
    experiment.validate_capture(rows, offsets, xy, xyz, ids, K)

    invalid_cases = (
        (np.array([0, 2]), xy, xyz, ids, rows),
        (offsets, xy, np.zeros((2, 2)), ids, rows),
        (offsets, xy, xyz, ids, [{"stamp": 1.0}, {"stamp": 1.0}]),
    )
    for bad_offsets, bad_xy, bad_xyz, bad_ids, bad_rows in invalid_cases:
        with np.testing.assert_raises(ValueError):
            experiment.validate_capture(bad_rows, bad_offsets, bad_xy, bad_xyz, bad_ids, K)


def test_scheduled_relocalizer_delivers_after_measured_video_time(monkeypatch) -> None:
    calls: list[np.ndarray] = []
    synchronizes: list[bool] = []
    fix = types.SimpleNamespace(
        status="LOCALIZED_STRONG",
        inliers=80,
        stage_ms={"retrieval_ms": 3.0, "match_lift_ms": 9.5},
    )

    class Provider:
        def localize_array(self, gray):
            calls.append(gray)
            return fix

    clock = iter((10.0, 10.25))
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(clock))
    worker = experiment.ScheduledRelocalizer(Provider(), lambda: synchronizes.append(True))
    worker.stamp = 5.0

    assert worker.submit(np.zeros((2, 2), dtype=np.uint8), 7, 5.0, 0)
    assert worker.busy
    worker.stamp = 5.249
    assert worker.poll() is None
    worker.stamp = 5.25
    delivered = worker.poll()

    assert delivered.fix is fix
    assert delivered.ordinal == 7
    assert delivered.capture_stamp == 5.0
    assert delivered.source_epoch == 0
    assert len(calls) == 1
    assert len(synchronizes) == 2
    assert worker.jobs[0]["runtime_ms"] == 250.0
    assert worker.jobs[0]["stage_ms"] == {
        "retrieval_ms": 3.0,
        "match_lift_ms": 9.5,
    }
    assert worker.frame_reloc_ms == 250.0
    assert not worker.busy


def test_evaluate_holds_out_id_split_for_every_estimator(monkeypatch, tmp_path: Path) -> None:
    captured_xyz: list[np.ndarray] = []

    class FakeCamera:
        def __init__(self, **kwargs):
            self.params = kwargs["params"]

    class FakeOptions:
        def __init__(self):
            self.ransac = types.SimpleNamespace()

    class FakePose:
        def matrix(self):
            return np.column_stack((np.eye(3), np.zeros(3)))

    class FakeRigid3d:
        def __init__(self, matrix):
            self.matrix_value = matrix

    refinement_iterations: list[int] = []

    def estimate(image_xy, world_xyz, camera, estimation):
        del image_xy, camera, estimation
        captured_xyz.append(np.asarray(world_xyz).copy())
        return {
            "num_inliers": len(world_xyz),
            "inlier_mask": np.ones(len(world_xyz), dtype=bool),
            "cam_from_world": FakePose(),
        }

    def refine(seed, image_xy, world_xyz, inlier_mask, camera, refinement):
        del seed, image_xy, world_xyz, inlier_mask, camera
        refinement_iterations.append(refinement.max_num_iterations)
        return {"cam_from_world": FakePose()}

    fake_pycolmap = types.SimpleNamespace(
        __version__="4.0.4",
        Camera=FakeCamera,
        Rigid3d=FakeRigid3d,
        AbsolutePoseEstimationOptions=FakeOptions,
        AbsolutePoseRefinementOptions=FakeOptions,
        estimate_absolute_pose=estimate,
        refine_absolute_pose=refine,
    )

    class FakeESKF:
        def update(self, stamp, pose, **kwargs):
            del stamp, kwargs
            return {"pose": pose, "accepted": True, "predicted_only": False}

    class FakeWindow:
        def __init__(self, window_size):
            self.window_size = window_size

        def update(self, frame):
            return {"pose": frame["pose"], "optimized": True, "nfev": 1, "cost": 0.0}

    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)
    monkeypatch.setitem(
        sys.modules,
        "pose_filter_experiment",
        types.SimpleNamespace(VisualPoseESKF=FakeESKF),
    )
    monkeypatch.setitem(
        sys.modules,
        "pose_window_experiment",
        types.SimpleNamespace(PoseWindowOptimizer=FakeWindow),
    )

    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    ids = np.arange(1, 16, dtype=np.int64)
    xyz = np.column_stack((ids * 0.01, ids * 0.02, np.full(ids.size, 5.0)))
    xy = np.column_stack((50.0 + ids * 0.2, 50.0 + ids * 0.4))
    source = tmp_path / "capture"
    source.mkdir()
    (source / "capture.json").write_text(
        json.dumps({"K": K.tolist(), "image_size": [100, 100], "video_sha256": "video"}),
        encoding="utf-8",
    )
    (source / "frames.jsonl").write_text(
        json.dumps({"stamp": 1.0, "epoch": 0}) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        source / "observations.npz",
        offsets=np.array([0, len(ids)], dtype=np.int64),
        xy=xy,
        xyz=xyz,
        ids=ids,
    )

    experiment.evaluate(
        types.SimpleNamespace(capture=source, out=tmp_path / "results", max_frames=0)
    )

    assert captured_xyz
    assert len(captured_xyz) == 1
    assert captured_xyz[0].shape == (12, 3)
    assert not np.isclose(captured_xyz[0][:, 0][:, None], [0.05, 0.10, 0.15]).any()
    assert refinement_iterations == [2, 3, 5, 10]
    records = [
        json.loads(line)
        for line in (tmp_path / "results" / "results.jsonl").read_text().splitlines()
    ]
    assert {row["train_count"] for row in records} == {12}
    assert {row["holdout_count"] for row in records} == {3}
    assert all(row["heldout_within4px"] == 1.0 for row in records)
