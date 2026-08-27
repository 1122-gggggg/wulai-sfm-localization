from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


VALIDATION = Path(__file__).resolve().parents[1]


def load_module():
    path = VALIDATION / "compare_pose_reference.py"
    spec = importlib.util.spec_from_file_location("compare_pose_reference_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result(
    xs: list[float],
    yaws_deg: list[float],
    *,
    failed: set[int] | None = None,
    reference_evidence: dict | None = None,
) -> dict:
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
    payload = {
        "rows": rows,
        "summary": {
            "actual_fps_including_io": 20.0,
            "wall_ms": {"median": 40.0, "p90": 50.0},
        },
    }
    if reference_evidence is not None:
        payload["reference_evidence"] = reference_evidence
    return payload


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


def test_missing_reference_evidence_is_pseudo_continuity_only():
    compare = load_module()
    reference = result([0.0], [0.0])
    candidate = result([0.1], [1.0])

    report = compare.compare_pose_reference(reference, candidate)

    assert report["absolute_accuracy_validated"] is False
    assert report["claim_scope"]["scope"] == "pseudo/continuity-only"
    assert report["claim_scope"]["reference_evidence"] is None
    assert report["claim_scope"]["interpretation"]["absolute_accuracy_validated"] is False
    assert "pseudo-ground-truth" in report["interpretation"]


def test_fake_reference_evidence_cannot_escalate_a_claim():
    compare = load_module()
    reference = result(
        [0.0],
        [0.0],
        reference_evidence={
            "kind": "surveyed",
            "map_aligned": True,
            "independent_of_localizer": True,
            "source_sha256": "A" * 64,
        },
    )
    reference["absolute_accuracy_validated"] = True

    report = compare.compare_pose_reference(reference, result([0.1], [1.0]))

    assert report["absolute_accuracy_validated"] is False
    assert report["claim_scope"]["scope"] == "pseudo/continuity-only"
    assert "source_sha256" in report["claim_scope"]["interpretation"]["validation_errors"][0]


def test_valid_independent_map_aligned_evidence_validates_absolute_accuracy():
    compare = load_module()
    evidence = {
        "kind": "surveyed",
        "map_aligned": True,
        "independent_of_localizer": True,
        "source_sha256": "a" * 64,
    }
    report = compare.compare_pose_reference(
        result([0.0], [0.0], reference_evidence=evidence),
        result([0.1], [1.0]),
    )

    assert report["absolute_accuracy_validated"] is True
    assert report["claim_scope"]["scope"] == "absolute_accuracy"
    assert report["claim_scope"]["reference_evidence"] == evidence
    assert report["claim_scope"]["interpretation"]["validation_errors"] == []


def test_cli_gate_rejects_unqualified_reference_before_comparison(tmp_path, monkeypatch, capsys):
    compare = load_module()
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    reference_path.write_text(json.dumps(result([0.0], [0.0])), encoding="utf-8")
    candidate_path.write_text(json.dumps(result([0.1], [1.0])), encoding="utf-8")
    monkeypatch.setattr(
        compare,
        "compare_pose_reference",
        lambda *_args, **_kwargs: pytest.fail("comparison must not run before the claim gate"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_pose_reference.py",
            str(reference_path),
            str(candidate_path),
            "--require-absolute-ground-truth",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        compare.main()

    assert raised.value.code == 2
    assert "reference lacks valid absolute ground truth evidence" in capsys.readouterr().err
