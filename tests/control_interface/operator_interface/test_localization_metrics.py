from __future__ import annotations

import json
import math

from localization_metrics import (
    CIRCUIT_BREAKER_STATES,
    CUDA_SAMPLE_MIN_INTERVAL_S,
    RESULT_FIELDS,
    RESTART_REASONS,
    build_localization_metric_record,
    edm_cache_metric_fields,
    sample_cuda_memory_stats,
)


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


def test_lifecycle_and_cuda_fields_are_bounded_and_copied() -> None:
    assert len(RESULT_FIELDS) == len(set(RESULT_FIELDS))
    for field in (
        "restart_reason",
        "outage_duration_s",
        "rejected_submits",
        "ready_latency_ms",
        "first_result_latency_ms",
        "circuit_breaker_state",
        "oom_transition",
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "edm_cache_hits",
        "edm_cache_misses",
        "edm_cache_evictions",
    ):
        assert field in RESULT_FIELDS
    assert RESTART_REASONS == (
        "stall",
        "worker_exit",
        "response_desync",
        "startup",
        "oom",
        "fatal",
        "circuit_open",
    )
    assert CIRCUIT_BREAKER_STATES == ("closed", "open")
    record = build_localization_metric_record(
        {
            "success": True,
            "restart_reason": "oom",
            "outage_duration_s": 1.5,
            "rejected_submits": 2,
            "ready_latency_ms": 3576.6,
            "first_result_latency_ms": 40.0,
            "circuit_breaker_state": "open",
            "oom_transition": True,
            "cuda_allocated_bytes": 1024,
            "cuda_reserved_bytes": 2048,
            "cuda_peak_allocated_bytes": 4096,
            "edm_cache_hits": 875,
            "edm_cache_misses": 125,
            "edm_cache_evictions": 3,
            "untrusted_extra": "drop",
        },
        metric_mono_ns=1_000_000_000,
        loc_fps=20.864,
        submit_ok=1,
        submit_skip_busy=0,
        submit_busy_attempts=0,
    )
    assert record["restart_reason"] == "oom"
    assert record["circuit_breaker_state"] == "open"
    assert record["oom_transition"] is True
    assert record["ready_latency_ms"] == 3576.6
    assert record["edm_cache_hits"] == 875
    assert record["cuda_peak_allocated_bytes"] == 4096
    assert "untrusted_extra" not in record


def test_cuda_memory_sample_is_interval_limited_without_sync() -> None:
    calls = []

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def memory_allocated(self) -> int:
            calls.append("allocated")
            return 10

        def memory_reserved(self) -> int:
            calls.append("reserved")
            return 20

        def max_memory_allocated(self) -> int:
            calls.append("peak")
            return 30

        def synchronize(self) -> None:
            raise AssertionError("must not force CUDA synchronization")

    cuda = FakeCuda()
    stats, sampled_at = sample_cuda_memory_stats(cuda, now_mono=10.0, last_sample_mono=None)
    assert stats == {
        "cuda_allocated_bytes": 10,
        "cuda_reserved_bytes": 20,
        "cuda_peak_allocated_bytes": 30,
    }
    skipped, previous = sample_cuda_memory_stats(
        cuda,
        now_mono=sampled_at + CUDA_SAMPLE_MIN_INTERVAL_S - 0.01,
        last_sample_mono=sampled_at,
    )
    assert skipped is None
    assert previous == sampled_at
    assert "synchronize" not in calls
    assert sample_cuda_memory_stats(None, now_mono=12.0)[0] is None


def test_edm_cache_metric_fields_are_fail_closed() -> None:
    assert edm_cache_metric_fields({"hits": 875, "misses": 125, "evictions": 3, "size": 32}) == {
        "edm_cache_hits": 875,
        "edm_cache_misses": 125,
        "edm_cache_evictions": 3,
    }
    assert edm_cache_metric_fields({"hits": -1, "misses": "x"}) == {}
    assert edm_cache_metric_fields(None) == {}
