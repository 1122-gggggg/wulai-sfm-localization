import json

from report_relocalization import build_report, session_capture


def test_missing_stages_are_not_zero_and_counts_are_per_job():
    result = build_report(
        {
            "frames": 100,
            "reloc_jobs": [
                {"ordinal": 1, "runtime_ms": 200, "stage_ms": {"match_lift_ms": 150}},
                {"ordinal": 10, "runtime_ms": 1600},
                {"ordinal": 20, "runtime_ms": float("nan")},
            ],
        }
    )
    assert result["runtime"]["count"] == 2
    assert result["runtime"]["max_ms"] == 1600
    assert result["stages"]["match_lift_ms"]["p50_ms"] == 150
    assert result["stages"]["match_lift_ms"]["count"] == 1
    assert result["slowest_jobs"][0]["ordinal"] == 10


def test_session_counts_each_reloc_once_and_survives_diagnostic_retention(tmp_path):
    row = {
        "event": "pose_result",
        "reloc_source_epoch": 0,
        "reloc_ordinal": 5,
        "reloc_capture_stamp_mono": 1.0,
        "reloc_ms": 250.0,
        "reloc_edm_match_ms": 100.0,
    }
    rows = [row] * 20 + [{**row, "reloc_source_epoch": 1, "reloc_ms": 500.0}]
    (tmp_path / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = build_report(session_capture(tmp_path))
    assert result["jobs"] == 2
    assert result["runtime"]["p50_ms"] == 375.0
