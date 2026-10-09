from __future__ import annotations

from pathlib import Path
import sys

import pytest

VALIDATION_DIR = Path(__file__).resolve().parents[3] / "定位演算法/validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from workmap_paired_compare import sequence_gates


def run_row(**overrides):
    return {"fresh": 100, "strong": 100, "e2e_fps": 24, "source_fps": 24,
            "core95": 5, "latency95": 100, "latency99": 200, **overrides}


def test_equal_fresh_and_exact_strong_threshold_pass():
    baseline = [run_row(fresh=value) for value in (90, 100, 110)]
    full = [run_row(fresh=value, strong=97) for value in (99, 100, 101)]
    assert sequence_gates(baseline, full)["pass"] is True
    assert sequence_gates(baseline, [run_row(strong=96)])["strong"] is False
    assert sequence_gates(baseline, [run_row(fresh=99)])["fresh"] is False


def test_one_slow_full_run_fails_throughput():
    full = [run_row(e2e_fps=0.98 * 24) for _ in range(3)]
    assert sequence_gates([run_row()], full)["throughput"] is True
    full[1]["e2e_fps"] -= 0.001
    gates = sequence_gates([run_row()], full)
    assert gates["throughput"] is False
    assert gates["pass"] is False


@pytest.mark.parametrize("baseline_core,limit", [(5, 13), (12, 24)])
def test_core_uses_larger_relative_or_absolute_limit(baseline_core, limit):
    baseline = [run_row(core95=baseline_core)]
    assert sequence_gates(baseline, [run_row(core95=limit)])["core"] is True
    assert sequence_gates(baseline, [run_row(core95=limit + 0.001)])["core"] is False


@pytest.mark.parametrize("field,limit", [("latency95", 100), ("latency99", 200)])
def test_latency_boundary(field, limit):
    assert sequence_gates([run_row()], [run_row(**{field: limit})])["latency"] is True
    assert sequence_gates([run_row()], [run_row(**{field: limit + 0.001})])["latency"] is False


def test_full_run_fields_count_events_and_read_session_meta(tmp_path):
    import json
    from workmap_paired_compare import _full_run_fields, _stage_ms, full_table_lines

    (tmp_path / "session_meta.json").write_text(json.dumps({"snapshot_write_ms": 7.5, "snapshot_bytes": 1234}))
    events = [
        {"event": "ba_geometry_committed", "updated_landmarks": {"-1": [0, 0, 0], "-2": [0, 0, 0]}, "submit_to_apply_ms": 100.0},
        {"event": "ba_geometry_committed", "updated_landmarks": {"-1": [0, 0, 0]}, "submit_to_apply_ms": 300.0},
        {"event": "ba_geometry_committed", "updated_landmarks": {}, "submit_to_apply_ms": None},
        {"event": "ba_geometry_rejected", "reason": "later_keyframe_reprojection"},
        {"event": "ba_geometry_rejected", "reason": "later_keyframe_reprojection"},
        {"event": "ba_geometry_rejected", "reason": "keyframe_pose_revision"},
        {"event": "loop_geometry_rejected", "reason": "scale"},
        {"event": "submap_anchor_rejected", "reason": "scale_range", "scale": 0.4},
        {"event": "submap_anchor_rejected", "reason": "scale_range", "scale": 0.36},
        {"event": "closure_reseeded"},
        {"event": "closure_reseed_rejected", "reason": "walk_broken"},
    ]
    fields = _full_run_fields(tmp_path, events, {"close_session_ms": {"drain": 1.0}})
    assert fields["ba_committed"] == 3
    assert fields["ba_rejected"] == {"later_keyframe_reprojection": 2, "keyframe_pose_revision": 1}
    assert fields["ba_updated_points_median"] == 1
    assert fields["ba_submit_to_apply_ms_p95"] == pytest.approx(290.0)
    assert fields["loop_rejected"] == {"scale": 1}
    assert fields["anchor_rejected"] == {"scale_range": 2}
    assert fields["anchor_rejected_scale_median"] == pytest.approx(0.38)
    assert fields["closure_reseeded"] == 1 and fields["closure_rejected"] == {"walk_broken": 1}
    assert fields["close_session_ms"] == {"drain": 1.0}
    assert (fields["snapshot_write_ms"], fields["snapshot_bytes"]) == (7.5, 1234)

    rows = [{"response": {"vo_ms": float(value), "pnp_ms": None}} for value in range(1, 101)]
    stage = _stage_ms(rows)
    assert stage["vo_ms"]["p50"] == pytest.approx(50.5)
    assert stage["vo_ms"]["p95"] == pytest.approx(95.05)
    assert stage["pnp_ms"] is None and stage["workmap_pre_ms"] is None

    full = {**fields, "repeat": 1, "merges": {}, "unmerged_reasons": {"anchor_timeout": 1},
            "stage_ms": stage, "drain_s": 0.5}
    lines = full_table_lines({"P": {"full": [full]}})
    assert len(lines) == 3 and lines[2].startswith("| P | 1 | 3 |")
