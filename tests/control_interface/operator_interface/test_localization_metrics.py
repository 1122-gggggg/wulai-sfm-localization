from __future__ import annotations

import json
import math
from types import SimpleNamespace

from localization_metrics import (
    CIRCUIT_BREAKER_STATES,
    CUDA_SAMPLE_MIN_INTERVAL_S,
    RESULT_FIELDS,
    RESTART_REASONS,
    attach_fused_localization_telemetry,
    build_localization_metric_record,
    edm_cache_metric_fields,
    sample_cuda_memory_stats,
)


def test_fused_telemetry_includes_independent_gnss_stamp() -> None:
    timing = {}
    state = SimpleNamespace(
        att_roll=0.1,
        att_pitch=-0.2,
        att_yaw=1.3,
        speed_north_mps=0.4,
        speed_east_mps=-0.1,
        speed_down_mps=0.0,
        telemetry_read_mono_ns=2_100_000_000,
        gps_read_mono_ns=2_000_000_000,
        gps_latitude_deg=25.033,
        gps_longitude_deg=121.5654,
        gps_altitude_m=18.2,
        gps_latitude_accuracy_m=0.8,
        gps_longitude_accuracy_m=0.9,
        gps_altitude_accuracy_m=1.4,
    )

    attach_fused_localization_telemetry(timing, state)

    assert timing["fused_telemetry_mono"] == 2.1
    assert timing["fused_gps_mono"] == 2.0
    assert timing["fused_gps_latitude"] == 25.033
    assert timing["fused_gps_longitude_accuracy"] == 0.9


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
    assert "vpr_ms" not in record
    assert "refs" not in record


def test_metric_record_keeps_explicit_null_but_omits_absent_result_fields() -> None:
    record = build_localization_metric_record(
        {"success": False, "wall_ms": 1.0, "rejected": None},
        metric_mono_ns=1,
        loc_fps=0.0,
        submit_ok=0,
        submit_skip_busy=0,
        submit_busy_attempts=0,
    )

    assert "rejected" in record and record["rejected"] is None
    assert "vpr_ms" not in record


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
        adaptive_submit_interval_ms=37.5,
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
        adaptive_submit_interval_ms=37.5,
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
        "coalesced_submit_drops",
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
            "coalesced_submit_drops": 7,
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
        adaptive_submit_interval_ms=37.5,
    )
    assert record["restart_reason"] == "oom"
    assert record["circuit_breaker_state"] == "open"
    assert record["oom_transition"] is True
    assert record["ready_latency_ms"] == 3576.6
    assert record["coalesced_submit_drops"] == 7
    assert record["adaptive_submit_interval_ms"] == 37.5
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


def test_stage_breakdown_reaches_the_session_log() -> None:
    # vpr/match/pnp accounted for only ~73% of the worker's core_wall (26.30 ms
    # p50 against 19.09 ms of instrumented stages over 1676 frames), so the
    # remaining quarter was invisible to every replay gate. The stage fields
    # close that, and a field the worker sets but RESULT_FIELDS drops is silent
    # -- exactly how the fused telemetry stayed unlogged.
    stages = {
        "total_ms": 25.9,
        "stage_gray_ms": 0.9,
        "stage_bridge_ms": 0.02,
        "stage_query_ms": 5.1,
        "stage_select_ms": 0.4,
    }
    assert set(stages) <= set(RESULT_FIELDS)

    record = build_localization_metric_record(
        {
            "success": True,
            "wall_ms": 26.3,
            "core_wall_ms": 26.3,
            "vpr_ms": 0.0,
            "match_ms": 17.0,
            "pnp_ms": 2.1,
            **stages,
        },
        metric_mono_ns=1_000_000_000,
        loc_fps=8.0,
        submit_ok=1,
        submit_skip_busy=0,
        submit_busy_attempts=0,
    )
    for key, value in stages.items():
        assert record[key] == value

    # The point of the set is that it closes: named stages must not exceed the
    # tracker total, and the tracker total must not exceed the worker's span.
    named = (
        record["stage_gray_ms"]
        + record["stage_bridge_ms"]
        + record["stage_query_ms"]
        + record["stage_select_ms"]
        + record["match_ms"]
        + record["pnp_ms"]
    )
    assert named <= record["total_ms"] <= record["core_wall_ms"]


def test_live_status_is_written_for_a_typed_result(tmp_path, monkeypatch) -> None:
    """A frozen LocalizationResult is not JSON serialisable.

    Every live session logged diagnostic_write_failed with
    "Object of type LocalizationResult is not JSON serializable" and wrote no
    status file at all, because the live path hands _publish_live_result_state
    the typed result while only tests pass a dict.
    """
    import json as _json

    import operator_tick

    status = tmp_path / "live_status.json"
    monkeypatch.setattr(operator_tick, "LIVE_STATUS_PATH", status)

    class _Typed:
        def __init__(self) -> None:
            self._payload = {"frame_name": "f.jpg", "success": True, "inliers": 42}

        def get(self, key, default=None):
            return self._payload.get(key, default)

        def to_payload(self):
            return dict(self._payload)

    failures: list = []

    class _App:
        live_result = None
        live_result_frame_name = ""
        _last_status_write = -1e9

        def _apply_lost_hold_result(self, _r): ...
        def update_localization_metrics(self, _r): ...
        def _record_diagnostic_failure(self, path, exc):
            failures.append((path, exc))

    operator_tick._publish_live_result_state(_App(), _Typed())

    assert failures == []
    assert _json.loads(status.read_text(encoding="utf-8"))["inliers"] == 42
