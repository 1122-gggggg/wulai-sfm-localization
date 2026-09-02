"""Pure construction of the bounded localization telemetry record."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from numbers import Real
from typing import Any


CUDA_SAMPLE_MIN_INTERVAL_S = 1.0

RESTART_REASONS = (
    "stall",
    "worker_exit",
    "response_desync",
    "startup",
    "oom",
    "fatal",
    "circuit_open",
)
CIRCUIT_BREAKER_STATES = ("closed", "open")


RESULT_FIELDS = (
    "timing_clock",
    "client_submit_mono",
    "client_dequeue_mono",
    "client_write_start_mono",
    "client_write_done_mono",
    "worker_read_done_mono",
    "worker_core_start_mono",
    "worker_core_done_mono",
    "client_response_mono",
    "ui_arrival_mono",
    "frame_callback_enter_mono_ns",
    "frame_preprocess_start_mono_ns",
    "frame_yuv_ready_mono_ns",
    "frame_preprocess_done_mono_ns",
    "frame_store_mono_ns",
    "ui_serialize_start_mono_ns",
    "ui_serialize_done_mono_ns",
    "client_submit_mono_ns",
    "client_dequeue_mono_ns",
    "client_write_start_mono_ns",
    "client_write_done_mono_ns",
    "worker_read_done_mono_ns",
    "worker_core_start_mono_ns",
    "worker_core_done_mono_ns",
    "client_response_mono_ns",
    "ui_arrival_mono_ns",
    "ui_serialize_ms",
    "client_queue_wait_ms",
    "client_pipe_write_ms",
    "submit_to_worker_read_ms",
    "worker_done_to_client_ms",
    "client_roundtrip_ms",
    "ui_poll_delay_ms",
    "e2e_submit_to_ui_ms",
    "source_frame_stamp_mono",
    "source_stamp_semantics",
    "hold_retry",
    "hold_kind",
    "source_stamp_age_at_submit_ms",
    "source_stamp_age_at_ui_ms",
    "callback_to_preprocess_start_ms",
    "yuv_view_ms",
    "frame_preprocess_ms",
    "preprocess_done_to_submit_ms",
    "callback_to_submit_ms",
    "callback_to_inference_start_ms",
    "callback_to_localization_done_ms",
    "callback_to_ui_ms",
    "gpu_span_ms",
    "gpu_timing_profiled",
    "tracker_variant",
    "display_seq",
    "pose",
    "pose_raw",
    "pose_filter",
    "vpr_ms",
    "feature_ms",
    "match_ms",
    "pnp_ms",
    "pnp_candidates",
    "pnp_skipped",
    "pnp_workers",
    "edm_host_feature_cache",
    "mode",
    "next_mode",
    "composite_stage",
    "inliers",
    "n_corr",
    "reference_count",
    "requested_reference_count",
    "refs",
    "staged_early_stop",
    "rejected",
    "limited_jump",
    "candidate_mode",
    "global_retrieval_calls",
    "lost_search_stage",
    "lost_search_radius_factor",
    "reproj_rms",
    "inlier_ratio",
    "inlier_grid_cells",
    "neuflow_stage",
    "neuflow_anchor_count",
    "neuflow_flow_ms",
    "neuflow_gpu_ms",
    "projection_fallback",
    "projection_reason",
    "projection_radius_px",
    "projection_anchor_count",
    "projection_visible_count",
    "projection_match_count",
    "projection_best_inliers",
    "projection_project_ms",
    "projection_feature_ms",
    "projection_match_ms",
    "frame_name",
    "error",
    "force_track_bench",
    "force_track_label",
    "force_track_ref_requested",
    "force_track_ref_resolved",
    "force_track_seed_ms",
    "benchmark_mode_requested",
    "benchmark_mode_active",
    "relocalize_requested",
    "confidence_low",
    "confidence_hold_event",
    "confidence_hold_active",
    "confidence_low_streak",
    "confidence_hold_attempts",
    "benchmark_prior_kind",
    "benchmark_prior_ref",
    "benchmark_setup_ms",
    "benchmark_seeded",
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
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value


def build_localization_metric_record(
    result: dict[str, Any],
    *,
    metric_mono_ns: int,
    loc_fps: float,
    submit_ok: int,
    submit_skip_busy: int,
    submit_busy_attempts: int,
    adaptive_submit_interval_ms: float | None = None,
) -> dict[str, Any]:
    """Copy only the stable telemetry contract and add UI-side counters."""
    record = {
        "t_mono": metric_mono_ns * 1e-9,
        "t_mono_ns": metric_mono_ns,
        "success": bool(result.get("success")),
        "wall_ms": result.get("wall_ms"),
        "core_wall_ms": result.get("core_wall_ms", result.get("wall_ms")),
    }
    record.update((field, result[field]) for field in RESULT_FIELDS if field in result)
    record.update(
        {
            "limited_jump_confirmed": result.get("limited_jump_confirmed", False),
            "loc_fps": round(loc_fps, 3),
            "submit_ok": submit_ok,
            "submit_skip_busy": submit_skip_busy,
            "submit_busy_attempts": submit_busy_attempts,
            "adaptive_submit_interval_ms": adaptive_submit_interval_ms,
        }
    )
    return _json_safe(record)


def _optional_nonneg_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number < 0:
        return None
    return number


def _cuda_sample_now(now_mono: float | None) -> float:
    now = time.monotonic() if now_mono is None else float(now_mono)
    if not math.isfinite(now):
        raise ValueError(f"invalid CUDA sample timestamp: {now_mono!r}")
    return now


def _cuda_sample_interval(min_interval_s: Any) -> float:
    try:
        interval = float(min_interval_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid CUDA sample interval: {min_interval_s!r}") from exc
    if not math.isfinite(interval) or interval < 0.0:
        raise ValueError(f"invalid CUDA sample interval: {min_interval_s!r}")
    return interval


def _cuda_previous_sample(
    now: float,
    last_sample_mono: float | None,
    interval: float,
) -> float | None:
    """Return the previous stamp when the sampling interval has not elapsed."""
    if last_sample_mono is None:
        return None
    try:
        previous = float(last_sample_mono)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid CUDA last-sample timestamp: {last_sample_mono!r}") from exc
    if not math.isfinite(previous):
        raise ValueError(f"invalid CUDA last-sample timestamp: {last_sample_mono!r}")
    if now - previous < interval:
        return previous
    return None


def _cuda_available(cuda: Any) -> bool:
    if cuda is None:
        return False
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return True
    try:
        return bool(is_available())
    except Exception:
        return False


def _cuda_memory_counters(cuda: Any) -> dict[str, int] | None:
    try:
        allocated = int(cuda.memory_allocated())
        reserved = int(cuda.memory_reserved())
        peak = int(cuda.max_memory_allocated())
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if allocated < 0 or reserved < 0 or peak < 0:
        return None
    return {
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "cuda_peak_allocated_bytes": peak,
    }


def sample_cuda_memory_stats(
    cuda: Any,
    *,
    now_mono: float | None = None,
    last_sample_mono: float | None = None,
    min_interval_s: float = CUDA_SAMPLE_MIN_INTERVAL_S,
) -> tuple[dict[str, int] | None, float]:
    """Sample process CUDA counters without nvidia-smi or device synchronize."""
    now = _cuda_sample_now(now_mono)
    interval = _cuda_sample_interval(min_interval_s)
    previous = _cuda_previous_sample(now, last_sample_mono, interval)
    if previous is not None:
        return None, previous
    if not _cuda_available(cuda):
        return None, now
    return _cuda_memory_counters(cuda), now


def edm_cache_metric_fields(stats: Mapping[str, Any] | None) -> dict[str, int]:
    if not stats:
        return {}
    fields = {}
    hits = _optional_nonneg_int(stats.get("hits"))
    misses = _optional_nonneg_int(stats.get("misses"))
    evictions = _optional_nonneg_int(stats.get("evictions"))
    if hits is not None:
        fields["edm_cache_hits"] = hits
    if misses is not None:
        fields["edm_cache_misses"] = misses
    if evictions is not None:
        fields["edm_cache_evictions"] = evictions
    return fields


def _fused_telemetry_mono_s(telemetry_ns: Any) -> float | None:
    """Convert a backend poll stamp to seconds. Never use the image capture stamp."""
    if telemetry_ns is None:
        return None
    try:
        fused_mono = int(telemetry_ns) / 1_000_000_000.0
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isfinite(fused_mono) and fused_mono > 0.0:
        return fused_mono
    return None


def attach_fused_localization_telemetry(
    timing: dict[str, Any],
    state: Any,
) -> None:
    """Copy fused IMU, velocity, GNSS, and independent acquisition stamps."""
    if state is None:
        return
    timing.update(
        {
            "fused_roll": getattr(state, "att_roll", None),
            "fused_pitch": getattr(state, "att_pitch", None),
            "fused_yaw": getattr(state, "att_yaw", None),
            "fused_speed_north": getattr(state, "speed_north_mps", None),
            "fused_speed_east": getattr(state, "speed_east_mps", None),
            "fused_speed_down": getattr(state, "speed_down_mps", None),
            "fused_gps_latitude": getattr(state, "gps_latitude_deg", None),
            "fused_gps_longitude": getattr(state, "gps_longitude_deg", None),
            "fused_gps_altitude": getattr(state, "gps_altitude_m", None),
            "fused_gps_latitude_accuracy": getattr(state, "gps_latitude_accuracy_m", None),
            "fused_gps_longitude_accuracy": getattr(state, "gps_longitude_accuracy_m", None),
            "fused_gps_altitude_accuracy": getattr(state, "gps_altitude_accuracy_m", None),
        }
    )
    fused_mono = _fused_telemetry_mono_s(
        getattr(state, "telemetry_read_mono_ns", None),
    )
    if fused_mono is not None:
        timing["fused_telemetry_mono"] = fused_mono
    gps_mono = _fused_telemetry_mono_s(
        getattr(state, "gps_read_mono_ns", None),
    )
    if gps_mono is not None:
        timing["fused_gps_mono"] = gps_mono
