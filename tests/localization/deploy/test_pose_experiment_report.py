"""Guard the common-frame, map-only comparison against VO self-consistency."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "定位演算法/validation"))
from pose_experiment_report import summarize  # noqa: E402


def test_map_metrics_exclude_vo_and_use_the_same_frames_for_all_methods(tmp_path):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    xyz = np.array([[i * 0.1, 0, 2.0] for i in range(8)])
    xy = (xyz @ K.T)[:, :2] / xyz[:, 2:]
    xy[-2:] += 1000  # VO observations must not affect map-only accuracy.
    ids = np.array([5, 10, 15, 20, 25, 30, -5, -10])
    np.savez(
        capture / "observations.npz",
        xy=np.tile(xy, (2, 1)),
        xyz=np.tile(xyz, (2, 1)),
        ids=np.tile(ids, 2),
        offsets=[0, 8, 16],
    )
    (capture / "capture.json").write_text(
        json.dumps(
            {
                "K": K.tolist(),
                "video": "synthetic",
                "video_sha256": "fixture",
                "frames": 2,
                "declared_frames": 2,
                "status_counts": {"FAST_TRACK": 2},
            }
        )
    )
    metrics = {
        "frames": 2,
        "valid_poses": 2,
        "measured_poses": 2,
        "predicted_only": 0,
        "optimized_frames": 0,
        "stage_ms": {"p50": 1, "p95": 2},
        "estimator_total_ms": {"p50": 2, "p95": 3},
        "acceleration_u_s2": {"p50": None, "p95": None},
    }
    (results / "summary.json").write_text(
        json.dumps({"methods": {"refine2": metrics, "refine5": metrics}})
    )
    pose = np.column_stack((np.eye(3), np.zeros(3)))
    records = []
    for index in range(2):
        records.extend(
            [
                {"frame": index, "method": "refine2", "ok": True, "pose": pose.tolist()},
                {
                    "frame": index,
                    "method": "refine5",
                    "ok": index == 0,
                    "pose": pose.tolist() if index == 0 else None,
                },
            ]
        )
    (results / "results.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    report = summarize(capture, results)
    for row in report["methods"]:
        assert row["eligible_map_frames"] == 2
        assert row["common_map_frames"] == 1
        assert row["heldout_map_observations"] == 6
        assert row["map_reprojection_p95_px"] < 1e-10
    assert report["methods"][1]["missing_on_eligible"] == 1
