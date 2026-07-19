from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


VALIDATION = Path(__file__).resolve().parents[1]


def load_module():
    path = VALIDATION / "compare_pose_reference.py"
    spec = importlib.util.spec_from_file_location("compare_pose_reference_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result(xs: list[float], yaws_deg: list[float], *, failed: set[int] | None = None) -> dict:
    failed = failed or set()
    rows = []
    for index, (x, yaw) in enumerate(zip(xs, yaws_deg)):
        success = index not in failed
        rows.append({
            "idx": index,
            "frame": f"frame_{index:06d}.jpg",
            "capture_stamp": index / 30.0,
            "success": success,
            "mode": "TRACK",
            "wall_ms": 40.0,
            "load_ms": 2.0,
            "pose": ({"x": x, "y": 0.0, "z": 0.0, "yaw": math.radians(yaw)}
                     if success else None),
        })
    return {"rows": rows, "summary": {"actual_fps_including_io": 20.0,
                                        "wall_ms": {"median": 40.0, "p90": 50.0}}}


def test_reference_metrics_handle_wrapped_yaw_and_relative_motion():
    compare = load_module()
    reference = result([0.0, 1.0, 2.0], [179.0, -179.0, -177.0])
    candidate = result([0.1, 1.2, 2.3], [-179.0, -177.0, -175.0])
    report = compare.compare_pose_reference(reference, candidate)
    assert report["success_overlap"]["both"] == 3
    assert math.isclose(report["absolute_position_delta_m"]["median"], 0.2)
    assert math.isclose(report["absolute_yaw_delta_deg"]["median"], 2.0)
    assert math.isclose(report["frame_to_frame_translation_delta_m"]["median"], 0.1)
    assert report["frame_to_frame_yaw_delta_deg"]["max"] < 1e-12
    assert math.isclose(report["speed"]["candidate_normal_track"]["fps_from_median_wall"], 25.0)
    assert math.isclose(
        report["speed"]["candidate_normal_track"]["fps_from_median_wall_plus_load"],
        1000.0 / 42.0,
    )


def test_reference_metrics_separate_success_disagreement_and_failure_run():
    compare = load_module()
    reference = result([0.0, 1.0, 2.0, 3.0], [0.0] * 4, failed={3})
    candidate = result([0.0, 1.0, 2.0, 3.0], [0.0] * 4, failed={1, 2})
    report = compare.compare_pose_reference(reference, candidate)
    assert report["success_overlap"] == {
        "both": 1,
        "reference_only": 2,
        "candidate_only": 1,
        "neither": 0,
        "candidate_longest_failure_run": 2,
        "reference_longest_failure_run": 1,
    }


def test_reference_metrics_reject_misaligned_frames():
    compare = load_module()
    reference = result([0.0], [0.0])
    candidate = result([0.0], [0.0])
    candidate["rows"][0]["frame"] = "different.jpg"
    try:
        compare.compare_pose_reference(reference, candidate)
    except ValueError as exc:
        assert "alignment" in str(exc)
    else:
        raise AssertionError("misaligned rows must fail")
