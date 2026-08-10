from __future__ import annotations

import json
import math

from localization_metrics import RESULT_FIELDS, build_localization_metric_record


def test_metric_record_has_unique_bounded_fields_and_compatibility_defaults() -> None:
    assert len(RESULT_FIELDS) == len(set(RESULT_FIELDS))
    source = {
        "success": 1,
        "wall_ms": 55.0,
        "display_seq": 42,
        "limited_jump_confirmed": True,
        "untrusted_extra": "must not be logged",
    }

    record = build_localization_metric_record(
        source,
        metric_mono_ns=2_000_000_000,
        loc_fps=12.34567,
        submit_ok=3,
        submit_skip_busy=2,
        submit_busy_attempts=5,
    )

    assert record["t_mono"] == 2.0
    assert record["success"] is True
    assert record["core_wall_ms"] == 55.0
    assert record["display_seq"] == 42
    assert record["limited_jump_confirmed"] is True
    assert record["loc_fps"] == 12.346
    assert "untrusted_extra" not in record


def test_metric_record_does_not_mutate_the_worker_result() -> None:
    source = {"success": False, "wall_ms": 1.0}
    before = source.copy()

    build_localization_metric_record(
        source,
        metric_mono_ns=1,
        loc_fps=0.0,
        submit_ok=0,
        submit_skip_busy=0,
        submit_busy_attempts=0,
    )

    assert source == before


def test_metric_record_replaces_non_finite_values_before_json_output() -> None:
    record = build_localization_metric_record(
        {
            "success": True,
            "wall_ms": float("nan"),
            "pose": {"x": float("inf"), "y": -float("inf")},
        },
        metric_mono_ns=2_000_000_000,
        loc_fps=float("nan"),
        submit_ok=0,
        submit_skip_busy=0,
        submit_busy_attempts=0,
    )

    json.dumps(record, allow_nan=False)
    assert record["wall_ms"] is None
    assert record["pose"] == {"x": None, "y": None}
    assert record["loc_fps"] is None
    assert not any(
        isinstance(value, float) and not math.isfinite(value) for value in record.values()
    )
