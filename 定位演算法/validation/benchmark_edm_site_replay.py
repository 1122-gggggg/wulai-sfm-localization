#!/usr/bin/env python3
"""Replay one video through the exact EDM site profile used by the operator UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


import cv2
import numpy as np
import torch


LOCALIZATION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = LOCALIZATION_ROOT.parent
CONTROL_ROOT = WORKSPACE_ROOT / "控制介面程式"
DEPLOY_ROOT = LOCALIZATION_ROOT / "deploy_code" / "sfm_glomap_deploy"
for candidate in (VALIDATION_ROOT, CONTROL_ROOT, DEPLOY_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from production_localizer_factory import build_production_localizer  # noqa: E402
from site_profile import load_site_profile  # noqa: E402
from stream_integrity import StreamAudit, ffprobe_frame_count  # noqa: E402


DEFAULT_SITE_PROFILE = (
    CONTROL_ROOT.parent / "地圖檔" / "場域" / "river_site" / "site_profile.json"
)


def parse_camera_params(value: str) -> list[float]:
    try:
        params = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "camera params must be four comma-separated numbers: fx,fy,cx,cy"
        ) from exc
    if len(params) != 4 or not all(math.isfinite(item) for item in params):
        raise argparse.ArgumentTypeError(
            "camera params must be four finite numbers: fx,fy,cx,cy"
        )
    if params[0] <= 0.0 or params[1] <= 0.0:
        raise argparse.ArgumentTypeError("camera focal lengths must be positive")
    return params


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_identity(camera) -> dict[str, object]:
    return {
        "model": str(camera.model),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(value) for value in camera.params],
    }


def baseline_identity_failures(
    baseline: dict,
    *,
    video_sha256: str,
    site_profile_sha256: str,
    bundle_sha256: str,
    localizer_profile_sha256: str,
    camera: dict[str, object],
) -> list[str]:
    """Fail closed unless a quality baseline names the exact replay inputs."""
    failures: list[str] = []
    required = {
        "video_sha256": video_sha256,
        "site_profile_sha256": site_profile_sha256,
        "bundle_sha256": bundle_sha256,
        "localizer_profile_sha256": localizer_profile_sha256,
    }
    for key, actual in required.items():
        expected = baseline.get(key)
        if not isinstance(expected, str) or len(expected) != 64:
            failures.append(f"baseline is missing {key}")
        elif expected != actual:
            failures.append(f"{key} mismatch: expected={expected} actual={actual}")
    expected_camera = baseline.get("camera")
    if not isinstance(expected_camera, dict):
        failures.append("baseline is missing camera identity")
    else:
        for key in ("model", "width", "height", "params"):
            if expected_camera.get(key) != camera.get(key):
                failures.append(
                    f"camera {key} mismatch: expected={expected_camera.get(key)!r} "
                    f"actual={camera.get(key)!r}"
                )
    return failures


def percentile(values: list[float], value: float) -> float | None:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=float), value))


def metric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "mean": float(np.mean(values)) if values else None,
    }


WORKER_MODES = ("sequential", "production-path")
SIGMA_MODES = ("fused", "upstream")
RUNTIME_SIGMA_MODES = ("reference_grid", "bidirectional")
ACQUIRE_STAGE_MODES = ("full_set", "initial_topk")
LOST_STRATEGIES = ("boot_and_lost_once", "boot_once")
LOST_PRIOR_STRATEGIES = ("restrict_nearby", "full_global", "score_fusion")
RECEIPT_OVERRIDE_KEYS = (
    "radius",
    "mconf_thr",
    "coarse_topk",
    "cache_capacity",
    "lost_strategy",
    "lost_global_retrieval_interval",
    "fused_coarse_mode",
    "runtime_sigma_mode",
    "temporal_feature_cache_size",
    "acquire_stage_mode",
    "lost_prior_strategy",
    "lost_prior_fusion_weight",
    "worker_mode",
)
PRODUCTION_PATH_UNTRANSMITTED_OVERRIDE_KEYS = (
    "runtime_sigma_mode",
    "temporal_feature_cache_size",
    "acquire_stage_mode",
    "lost_prior_strategy",
    "lost_prior_fusion_weight",
)
PIPELINE_AGE_KEYS = (
    "capture_source_age_ms",
    "submit_source_age_ms",
    "promote_source_age_ms",
    "inference_source_age_ms",
    "result_source_age_ms",
)
CONSUMED_WORKER_METRIC_KEYS = (
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
    "fused_telemetry_mono",
)


def _arg(args, name: str, default=None):
    if isinstance(args, dict):
        return args[name] if name in args else default
    return getattr(args, name, default)


def _maybe_gpu_sync(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def frame_identity(rgb, source_index: int) -> dict[str, object]:
    contiguous = np.ascontiguousarray(rgb)
    return {
        "frame_id": f"src_{int(source_index):08d}",
        "frame_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "source_index": int(source_index),
        "rgb": contiguous,
    }


def effective_sigma_mode(args) -> str:
    explicit = _arg(args, "sigma_mode")
    if explicit in SIGMA_MODES:
        return str(explicit)
    raw = os.environ.get("SFM_EDM_FUSED_COARSE", "1").strip().lower()
    return "upstream" if raw in {"0", "false", "no", "off"} else "fused"


def apply_sigma_mode(args) -> None:
    sigma = _arg(args, "sigma_mode")
    if sigma == "upstream":
        os.environ["SFM_EDM_FUSED_COARSE"] = "0"
    elif sigma == "fused":
        os.environ["SFM_EDM_FUSED_COARSE"] = "1"


def _matcher_from_built(built):
    explicit = getattr(built, "_matcher", None)
    if explicit is not None:
        return explicit
    return getattr(
        getattr(getattr(getattr(built, "tracker", None), "trk", None), "loc", None),
        "matcher",
        None,
    )


def _profile_temporal_feature_cache_size(tracker: dict, matcher: dict) -> int:
    if "temporal_feature_cache_size" in matcher:
        return int(matcher["temporal_feature_cache_size"])
    return 2 if tracker.get("use_temporal_reference") else 0


def _receipt_choice(args, name: str, built_value, allowed) -> str | None:
    explicit = _arg(args, name)
    if explicit in allowed:
        return str(explicit)
    if built_value is None:
        return None
    return str(built_value)


def _reject_untransmitted_production_overrides(args) -> None:
    worker_mode = str(_arg(args, "worker_mode", "sequential") or "sequential")
    if worker_mode != "production-path":
        return
    blocked = [
        name.replace("_", "-")
        for name in PRODUCTION_PATH_UNTRANSMITTED_OVERRIDE_KEYS
        if _arg(args, name) is not None
    ]
    if blocked:
        raise SystemExit(
            "production-path cannot apply "
            + ", ".join(blocked)
            + " without a worker input or profile override"
        )


def _validate_config_override(cfg) -> None:
    validate = getattr(cfg, "validate", None)
    if callable(validate):
        validate()


_CONFIG_OVERRIDE_FIELDS = (
    ("radius", "radius", float),
    ("lost_strategy", "global_retrieval_policy", str),
    ("lost_global_retrieval_interval", "lost_global_retrieval_interval", int),
    ("acquire_stage_mode", "acquire_stage_mode", str),
    ("lost_prior_strategy", "lost_prior_strategy", str),
    ("lost_prior_fusion_weight", "lost_prior_fusion_weight", float),
)
_MATCHER_OVERRIDE_FIELDS = (
    ("mconf_thr", "mconf_thr", float),
    ("coarse_topk", "topk", int),
    ("cache_capacity", "reference_cache_size", int),
    ("runtime_sigma_mode", "runtime_sigma_mode", str),
)
_MATCHER_OVERRIDE_ARG_NAMES = tuple(
    name for name, _attr, _caster in _MATCHER_OVERRIDE_FIELDS
) + ("temporal_feature_cache_size",)


def _apply_named_overrides(target, args, fields, validate=None) -> None:
    for name, attr, caster in fields:
        value = _arg(args, name)
        if value is None:
            continue
        setattr(target, attr, caster(value))
        if validate is not None:
            validate(target)


def _apply_temporal_cache_override(matcher, args) -> None:
    temporal_cache = _arg(args, "temporal_feature_cache_size")
    if temporal_cache is None:
        return
    size = int(temporal_cache)
    matcher.temporal_feature_cache_size = size
    capacity = getattr(matcher, "_feature_cache_capacity", None)
    if isinstance(capacity, dict):
        capacity["temporal"] = size


def _has_matcher_override(args) -> bool:
    return any(_arg(args, name) is not None for name in _MATCHER_OVERRIDE_ARG_NAMES)


def apply_runtime_overrides(args, built) -> None:
    """Mutate the built tracker/matcher with sequential CLI overrides."""
    cfg = getattr(built, "config", None)
    if cfg is not None:
        _apply_named_overrides(
            cfg, args, _CONFIG_OVERRIDE_FIELDS, _validate_config_override
        )
    matcher = _matcher_from_built(built)
    if _has_matcher_override(args) and matcher is None:
        raise SystemExit("matcher overrides require an EDM matcher")
    if matcher is None:
        return
    _apply_named_overrides(matcher, args, _MATCHER_OVERRIDE_FIELDS)
    _apply_temporal_cache_override(matcher, args)


def build_receipt(args, built) -> dict[str, object]:
    cfg = getattr(built, "config", None) if built is not None else None
    matcher = _matcher_from_built(built) if built is not None else None
    radius = _arg(args, "radius")
    if radius is None:
        radius = getattr(cfg, "radius", None)
    mconf = _arg(args, "mconf_thr")
    if mconf is None:
        mconf = getattr(matcher, "mconf_thr", None)
    topk = _arg(args, "coarse_topk")
    if topk is None:
        topk = getattr(matcher, "topk", None)
    cache = _arg(args, "cache_capacity")
    if cache is None:
        cache = getattr(matcher, "reference_cache_size", None)
    lost_strategy = _arg(args, "lost_strategy")
    if lost_strategy is None:
        lost_strategy = getattr(cfg, "global_retrieval_policy", None)
    interval = _arg(args, "lost_global_retrieval_interval")
    if interval is None:
        interval = getattr(cfg, "lost_global_retrieval_interval", None)
    runtime_sigma = _receipt_choice(
        args,
        "runtime_sigma_mode",
        getattr(matcher, "runtime_sigma_mode", None),
        RUNTIME_SIGMA_MODES,
    )
    temporal_cache = _arg(args, "temporal_feature_cache_size")
    if temporal_cache is None:
        temporal_cache = getattr(matcher, "temporal_feature_cache_size", None)
    acquire_mode = _receipt_choice(
        args,
        "acquire_stage_mode",
        getattr(cfg, "acquire_stage_mode", None),
        ACQUIRE_STAGE_MODES,
    )
    prior_strategy = _receipt_choice(
        args,
        "lost_prior_strategy",
        getattr(cfg, "lost_prior_strategy", None),
        LOST_PRIOR_STRATEGIES,
    )
    fusion_weight = _arg(args, "lost_prior_fusion_weight")
    if fusion_weight is None:
        fusion_weight = getattr(cfg, "lost_prior_fusion_weight", None)
    return {
        "radius": None if radius is None else float(radius),
        "mconf_thr": None if mconf is None else float(mconf),
        "coarse_topk": None if topk is None else int(topk),
        "cache_capacity": None if cache is None else int(cache),
        "lost_strategy": None if lost_strategy is None else str(lost_strategy),
        "lost_global_retrieval_interval": None if interval is None else int(interval),
        "fused_coarse_mode": effective_sigma_mode(args),
        "runtime_sigma_mode": runtime_sigma,
        "temporal_feature_cache_size": (
            None if temporal_cache is None else int(temporal_cache)
        ),
        "acquire_stage_mode": acquire_mode,
        "lost_prior_strategy": prior_strategy,
        "lost_prior_fusion_weight": (
            None if fusion_weight is None else float(fusion_weight)
        ),
        "worker_mode": str(_arg(args, "worker_mode", "sequential") or "sequential"),
    }


def _receipt_values_equal(expected, actual) -> bool:
    if expected == actual:
        return True
    try:
        left = float(expected)
        right = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left) and math.isfinite(right) and left == right


def receipt_identity_failures(baseline: dict, actual: dict | None) -> list[str]:
    """Fail closed unless both sides name the same effective receipt identity."""
    expected = baseline.get("receipt") if isinstance(baseline, dict) else None
    if expected is None:
        return ["baseline is missing receipt identity"]
    if not isinstance(expected, dict):
        return ["baseline receipt identity is not an object"]
    if not isinstance(actual, dict):
        return ["candidate is missing receipt identity"]
    failures: list[str] = []
    for key in RECEIPT_OVERRIDE_KEYS:
        if key not in expected:
            failures.append(f"baseline is missing receipt.{key}")
            continue
        if key not in actual:
            failures.append(f"candidate is missing receipt.{key}")
            continue
        if not _receipt_values_equal(expected[key], actual.get(key)):
            failures.append(
                f"receipt.{key} mismatch: expected={expected[key]!r} "
                f"actual={actual.get(key)!r}"
            )
    return failures



def pipeline_summary(rows: list[dict], *, startup_ms: float | None = None) -> dict:
    first_fix = next((row for row in rows if row.get("success")), None)
    coalesce_drops = sum(int(row.get("coalesce_drops", 0) or 0) for row in rows)
    submit_drops = sum(int(row.get("submit_drop", 0) or 0) for row in rows)
    copies = sum(1 for row in rows if row.get("copy_ms") is not None)
    restarts = sum(
        1 for row in rows
        if row.get("restart_required") or row.get("failure_kind") == "worker_unavailable"
    )
    ages = {key: metric_summary([
        float(row[key]) for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]) for key in PIPELINE_AGE_KEYS}
    return {
        **ages,
        "copy_ms": metric_summary([
            float(row["copy_ms"]) for row in rows
            if row.get("copy_ms") is not None and math.isfinite(float(row["copy_ms"]))
        ]),
        "gpu_span_ms": metric_summary([
            float(row["gpu_span_ms"]) for row in rows
            if row.get("gpu_span_ms") is not None and math.isfinite(float(row["gpu_span_ms"]))
        ]),
        "copies": copies,
        "coalesce_drops": coalesce_drops,
        "submit_drops": submit_drops,
        "restarts": restarts,
        "startup_ms": startup_ms,
        "first_fix_frame": None if first_fix is None else first_fix.get("source_index"),
        "first_fix_ms": None if first_fix is None else first_fix.get("result_source_age_ms"),
        "allocator": {
            "cuda_allocated_bytes": next(
                (row.get("cuda_allocated_bytes") for row in reversed(rows)
                 if row.get("cuda_allocated_bytes") is not None),
                None,
            ),
            "cuda_reserved_bytes": next(
                (row.get("cuda_reserved_bytes") for row in reversed(rows)
                 if row.get("cuda_reserved_bytes") is not None),
                None,
            ),
            "cuda_peak_allocated_bytes": next(
                (row.get("cuda_peak_allocated_bytes") for row in reversed(rows)
                 if row.get("cuda_peak_allocated_bytes") is not None),
                None,
            ),
        },
        "cache": {
            "hits": next((row.get("edm_cache_hits") for row in reversed(rows)
                          if row.get("edm_cache_hits") is not None), None),
            "misses": next((row.get("edm_cache_misses") for row in reversed(rows)
                            if row.get("edm_cache_misses") is not None), None),
            "evictions": next((row.get("edm_cache_evictions") for row in reversed(rows)
                               if row.get("edm_cache_evictions") is not None), None),
        },
    }


def run_source_cadence(
    decoded,
    source_fps: float,
    client,
    *,
    sleep=time.sleep,
    monotonic=time.monotonic,
    drain_timeout_s: float = 8.0,
) -> tuple[list[dict], dict]:
    """Submit frames at recorded cadence through a capacity-one client.

    `decoded` yields dicts with rgb, source_index, decode_ms, recorded_stamp,
    frame_id, frame_sha256, and copy_ms. Queues stay bounded: one frame is
    decoded, then submitted (coalesce if busy), then results are polled.
    """
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"invalid source fps: {source_fps!r}")
    origin = monotonic()
    identities: dict[str, dict] = {}
    payloads: list[dict] = []
    submit_ok = 0
    submit_rejected = 0
    seq = 0
    decoded_count = 0
    coalesce_at_submit: list[int] = []
    for item in decoded:
        decoded_count += 1
        source_index = int(item["source_index"])
        due = origin + source_index / source_fps
        delay = due - monotonic()
        if delay > 0.0:
            sleep(delay)
        capture_mono = monotonic()
        identity = {
            "source_index": source_index,
            "frame_id": item["frame_id"],
            "frame_sha256": item["frame_sha256"],
            "recorded_stamp": float(item["recorded_stamp"]),
            "decode_ms": float(item["decode_ms"]),
            "copy_ms": item.get("copy_ms"),
            "capture_mono": capture_mono,
        }
        identities[str(item["frame_id"])] = identity
        timing = {
            "source_frame_stamp_mono": capture_mono,
            "source_stamp_semantics": "recorded_cadence_host_monotonic",
            "fused_telemetry_mono": capture_mono,
        }
        drops_before = int(getattr(client, "_coalesce_drops", 0) or 0)
        accepted = client.submit(
            seq, str(item["frame_id"]), item["rgb"], timing_metadata=timing,
        )
        seq += 1
        if accepted:
            submit_ok += 1
        else:
            submit_rejected += 1
            identity["submit_drop"] = 1
        drop_delta = int(getattr(client, "_coalesce_drops", 0) or 0) - drops_before
        identity["coalesce_drops"] = drop_delta
        coalesce_at_submit.append(drop_delta)
        payloads.extend(client.poll_results())
    drain_deadline = monotonic() + max(0.0, float(drain_timeout_s))
    busy = getattr(client, "busy", None)
    while callable(busy) and busy() and monotonic() < drain_deadline:
        payloads.extend(client.poll_results())
        sleep(0.01)
    payloads.extend(client.poll_results())
    stats = {
        "decoded": decoded_count,
        "submit_ok": submit_ok,
        "submit_rejected": submit_rejected,
        "coalesce_drops": int(getattr(client, "_coalesce_drops", 0) or 0),
        "coalesce_at_submit": sum(coalesce_at_submit),
        "results": len(payloads),
    }
    rows = [
        _row_from_worker_payload(payload, identities)
        for payload in payloads
    ]
    return rows, stats


def _row_from_worker_payload(payload: dict, identities: dict) -> dict:
    frame_id = str(payload.get("frame_name") or payload.get("frame_id") or "")
    identity = identities.get(frame_id, {})
    capture_mono = identity.get("capture_mono")
    def _age_ms(end_key: str, fallback_key: str | None = None):
        end = payload.get(end_key)
        if end is None and fallback_key is not None:
            end = payload.get(fallback_key)
        if capture_mono is None or end is None:
            return None
        try:
            return max(0.0, (float(end) - float(capture_mono)) * 1000.0)
        except (TypeError, ValueError):
            return None
    pose = payload.get("pose")
    success = bool(payload.get("success"))
    row = {
        "source_index": identity.get("source_index", payload.get("display_seq")),
        "frame_id": frame_id or identity.get("frame_id"),
        "frame_sha256": identity.get("frame_sha256"),
        "capture_stamp": identity.get("recorded_stamp"),
        "success": success,
        "ok": success if payload.get("ok") is None else bool(payload.get("ok")),
        "mode": payload.get("mode") or payload.get("state_in"),
        "next_mode": payload.get("next_mode") or payload.get("state_out"),
        "state_in": payload.get("state_in") or payload.get("mode"),
        "state_out": payload.get("state_out") or payload.get("next_mode"),
        "candidate_mode": payload.get("candidate_mode"),
        "global_retrieval_calls": payload.get("global_retrieval_calls"),
        "temporal_used": payload.get("temporal_used"),
        "selected_ref": payload.get("selected_ref"),
        "rejected": payload.get("rejected"),
        "limited_jump": payload.get("limited_jump"),
        "limited_jump_confirmed": bool(payload.get("limited_jump_confirmed")),
        "inliers": int(payload.get("inliers", 0) or 0),
        "reproj_rms": payload.get("reproj_rms"),
        "n_corr": payload.get("n_corr"),
        "reference_count": payload.get("reference_count"),
        "refs": list(payload.get("refs") or []),
        "wall_ms": payload.get("wall_ms") or payload.get("core_wall_ms"),
        "decode_ms": identity.get("decode_ms"),
        "copy_ms": identity.get("copy_ms"),
        "vpr_ms": payload.get("vpr_ms"),
        "match_ms": payload.get("match_ms"),
        "pnp_ms": payload.get("pnp_ms"),
        "step": None,
        "submit_drop": int(identity.get("submit_drop", 0) or 0),
        "coalesce_drops": int(identity.get("coalesce_drops", 0) or 0),
        "restart_required": payload.get("restart_required"),
        "failure_kind": payload.get("failure_kind"),
        "capture_source_age_ms": 0.0,
        "submit_source_age_ms": payload.get("source_stamp_age_at_submit_ms"),
        "promote_source_age_ms": _age_ms("client_dequeue_mono"),
        "inference_source_age_ms": _age_ms("worker_core_start_mono"),
        "result_source_age_ms": _age_ms("client_response_mono"),
        "gpu_span_ms": payload.get("gpu_span_ms"),
        "pose": pose,
    }
    for key in CONSUMED_WORKER_METRIC_KEYS:
        if payload.get(key) is not None:
            row[key] = payload.get(key)
    return row


def attach_production_path_receipt_fields(rows: list[dict], stats: dict, client) -> None:
    """Copy coalesce/restart receipt fields from the bounded worker onto result rows."""
    if not rows:
        return
    rows[-1]["coalesce_drops"] = int(stats.get("coalesce_drops", 0) or 0)
    info = getattr(client, "startup_info", {}) or {}
    if isinstance(info, dict):
        for key in CONSUMED_WORKER_METRIC_KEYS:
            if info.get(key) is not None and rows[0].get(key) is None:
                rows[0][key] = info[key]
    if rows[0].get("restart_reason") is None:
        reason = getattr(client, "_restart_reason", None)
        if reason is not None:
            rows[0]["restart_reason"] = reason
    if rows[0].get("ready_latency_ms") is None:
        ready = getattr(client, "_ready_latency_ms", None)
        if ready is not None:
            rows[0]["ready_latency_ms"] = ready


def iter_decoded_replay_frames(args, cap, source_fps: float, site, audit):
    width = int(site.query_camera.width)
    height = int(site.query_camera.height)
    source_index = 0
    selected = 0
    while not args.max_frames or selected < args.max_frames:
        decode_started = time.perf_counter()
        try:
            ok, bgr = cap.read()
        except cv2.error:
            audit.decode_errors += 1
            break
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if not ok:
            break
        current_index = source_index
        source_index += 1
        audit.decoded_raw_frames += 1
        if current_index % args.stride:
            continue
        copy_started = time.perf_counter()
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        identity = frame_identity(rgb, current_index)
        copy_ms = (time.perf_counter() - copy_started) * 1000.0
        audit.sampled_frames += 1
        selected += 1
        yield {
            "source_index": current_index,
            "rgb": identity["rgb"],
            "decode_ms": decode_ms,
            "copy_ms": copy_ms,
            "recorded_stamp": float(current_index) / source_fps,
            "frame_id": identity["frame_id"],
            "frame_sha256": identity["frame_sha256"],
        }


def _wait_client_ready(client, timeout_s: float, *, sleep, monotonic) -> None:
    deadline = monotonic() + max(0.0, float(timeout_s))
    while not bool(getattr(client, "ready", False)):
        if bool(getattr(client, "unavailable", False)):
            raise RuntimeError(
                getattr(client, "startup_error", None) or "live localizer worker unavailable"
            )
        if monotonic() >= deadline:
            raise RuntimeError("live localizer worker did not become ready")
        sleep(0.01)


def _open_live_localizer_client(args, site):
    operator_dir = CONTROL_ROOT / "operator_interface"
    if str(operator_dir) not in sys.path:
        sys.path.insert(0, str(operator_dir))
    from live_worker_clients import LiveLocalizerClient

    worker_py = Path(
        _arg(args, "live_worker") or (operator_dir / "live_localizer_worker.py")
    ).expanduser().resolve()
    if not worker_py.is_file():
        raise SystemExit(f"live localizer worker not found: {worker_py}")
    python_bin = str(_arg(args, "live_python") or sys.executable)
    camera = site.query_camera
    deploy_dir = site.localizer_deploy_dir or DEPLOY_ROOT
    return LiveLocalizerClient(
        worker_py,
        python_bin,
        int(camera.width),
        int(camera.height),
        Path(site.localization_bundle),
        megaloc_cache=str(site.megaloc_cache or ""),
        reference_index=str(site.reference_index or ""),
        reference_index_sha256=str(site.asset_sha256.reference_index or ""),
        localizer_backend="edm",
        localizer_deploy_dir=str(deploy_dir),
        localizer_profile=str(site.localizer_profile or ""),
        bundle_sha256=str(site.asset_sha256.localization_bundle or ""),
        localizer_profile_sha256=str(site.asset_sha256.localizer_profile or ""),
        query_camera=camera,
        map_align=str(site.map_align or ""),
    )


def runtime_stub_from_profile(site, args):
    _reject_untransmitted_production_overrides(args)
    if site.localizer_profile is None:
        raise SystemExit("production-path mode requires a localizer profile")
    raw = json.loads(Path(site.localizer_profile).read_text(encoding="utf-8"))
    tracker = raw.get("tracker") if isinstance(raw, dict) else None
    matcher = raw.get("matcher") if isinstance(raw, dict) else None
    if not isinstance(tracker, dict) or not isinstance(matcher, dict):
        raise SystemExit("localizer profile must contain matcher and tracker objects")
    cfg = SimpleNamespace(
        max_corr_total=int(tracker.get("max_corr_total", 0) or 0),
        radius=float(tracker["radius"]),
        global_retrieval_policy=str(tracker.get("global_retrieval_policy")),
        lost_global_retrieval_interval=int(
            tracker.get("lost_global_retrieval_interval", 0)
        ),
        acquire_stage_mode=str(tracker.get("acquire_stage_mode", "full_set")),
        lost_prior_strategy=str(
            tracker.get("lost_prior_strategy", "restrict_nearby")
        ),
        lost_prior_fusion_weight=float(
            tracker.get("lost_prior_fusion_weight", 1.0)
        ),
    )
    matcher_obj = SimpleNamespace(
        mconf_thr=float(matcher["mconf_thr"]),
        topk=int(matcher["coarse_topk"]),
        reference_cache_size=int(matcher["reference_cache_size"]),
        runtime_sigma_mode=str(matcher.get("runtime_sigma_mode", "reference_grid")),
        temporal_feature_cache_size=_profile_temporal_feature_cache_size(
            tracker, matcher
        ),
    )
    stub = SimpleNamespace(
        tracker=None,
        config=cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
        variant="live_localizer_client",
        _matcher=matcher_obj,
    )
    apply_runtime_overrides(args, stub)
    return stub




def _result_exit_code(
    row_count: int,
    audit: StreamAudit,
    *,
    all_frames_requested: bool,
    requested_frames_complete: bool = True,
    quality_failures: list[str] | None = None,
    decode_accepted: bool = False,
) -> int:
    if row_count == 0:
        return 2
    if all_frames_requested:
        if (
            not decode_accepted
            and (audit.decode_complete is not True or audit.decode_errors != 0)
        ):
            return 3
    elif not requested_frames_complete:
        return 3
    return 4 if quality_failures else 0


def evaluate_quality(summary: dict, baseline: dict) -> list[str]:
    """Return deterministic localization regressions against a pinned baseline."""
    thresholds = baseline.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("quality baseline must contain a thresholds object")

    state_counts = summary.get("state_counts") or {}
    rejection_counts = summary.get("rejection_counts") or {}
    inliers = summary.get("inliers") or {}
    reproj = summary.get("reproj_rms") or {}
    checks = (
        ("frames", summary.get("frames"), "==", thresholds.get("frames")),
        ("successes", summary.get("successes"), ">=", thresholds.get("min_successes")),
        ("TRACK", state_counts.get("TRACK", 0), ">=", thresholds.get("min_track")),
        ("LOST", state_counts.get("LOST", 0), "<=", thresholds.get("max_lost")),
        ("inliers.p50", inliers.get("p50"), ">=", thresholds.get("min_inliers_p50")),
        ("inliers.p95", inliers.get("p95"), ">=", thresholds.get("min_inliers_p95")),
        (
            "reproj_rms.p95", reproj.get("p95"), "<=",
            thresholds.get("max_reproj_rms_p95"),
        ),
        (
            "limited_jump_unconfirmed",
            rejection_counts.get("limited_jump_unconfirmed", 0),
            "<=",
            thresholds.get("max_limited_jump_unconfirmed"),
        ),
    )
    failures = []
    for name, actual, operator, expected in checks:
        if expected is None:
            raise ValueError(f"quality baseline threshold missing for {name}")
        try:
            actual_value = float(actual)
            expected_value = float(expected)
        except (TypeError, ValueError):
            failures.append(f"{name}: unavailable (required {operator} {expected})")
            continue
        passed = (
            actual_value == expected_value if operator == "=="
            else actual_value >= expected_value if operator == ">="
            else actual_value <= expected_value
        )
        if not passed:
            failures.append(f"{name}: actual={actual} required {operator} {expected}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, default=DEFAULT_SITE_PROFILE)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--camera-params",
        type=parse_camera_params,
        metavar="FX,FY,CX,CY",
        help="benchmark-only PnP camera override; model and image size stay unchanged",
    )
    parser.add_argument(
        "--max-corr-total",
        type=int,
        default=0,
        help="override profile PnP cap; 0 keeps the profile value",
    )
    parser.add_argument(
        "--radius",
        type=float,
        help="benchmark-only local reference search radius override",
    )
    parser.add_argument("--pnp-random-seed", type=int, default=0)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail before loading assets if CUDA is unavailable",
    )
    parser.add_argument(
        "--quality-baseline",
        type=Path,
        help="fail on localization-quality regression against this pinned baseline",
    )
    parser.add_argument(
        "--accept-known-incomplete",
        action="store_true",
        help=(
            "accept exactly one declared-but-undecodable tail frame when the "
            "quality baseline pins the decoded frame count and video SHA"
        ),
    )
    parser.add_argument(
        "--worker-mode",
        choices=list(WORKER_MODES),
        default="sequential",
        help="sequential is the in-process baseline; production-path uses LiveLocalizerClient",
    )
    parser.add_argument("--mconf-thr", type=float, help="override matcher confidence threshold")
    parser.add_argument("--coarse-topk", type=int, help="override matcher coarse topk")
    parser.add_argument("--cache-capacity", type=int, help="override matcher reference cache size")
    parser.add_argument(
        "--lost-strategy",
        choices=list(LOST_STRATEGIES),
        help="override global retrieval policy",
    )
    parser.add_argument(
        "--lost-global-retrieval-interval",
        type=int,
        help="override LOST MegaLoc retry interval; 0 keeps one-shot LOST",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=list(SIGMA_MODES),
        help="fused-coarse dual-softmax policy; distinct from --runtime-sigma-mode",
    )
    parser.add_argument(
        "--runtime-sigma-mode",
        choices=list(RUNTIME_SIGMA_MODES),
        help="runtime direction-01 sigma policy; distinct from --sigma-mode",
    )
    parser.add_argument(
        "--temporal-feature-cache-size",
        type=int,
        help="override matcher temporal feature cache entries",
    )
    parser.add_argument(
        "--acquire-stage-mode",
        choices=list(ACQUIRE_STAGE_MODES),
        help="override BOOT/LOST acquire staging",
    )
    parser.add_argument(
        "--lost-prior-strategy",
        choices=list(LOST_PRIOR_STRATEGIES),
        help="override LOST prior retrieval strategy",
    )
    parser.add_argument(
        "--lost-prior-fusion-weight",
        type=float,
        help="override LOST score-fusion prior weight",
    )
    parser.add_argument(
        "--gpu-span",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="opt-in synchronized CUDA span around sequential inference",
    )
    parser.add_argument("--live-worker", type=Path, help="production-path worker script")
    parser.add_argument("--live-python", help="production-path worker interpreter")
    parser.add_argument("--pair-role", choices=["A", "B"], help="label this run in a B-A-A-B pair")
    parser.add_argument("--pair-index", type=int, help="0-based index in a paired series")

    return parser.parse_args()


def _is_finite_positive(value) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


def _is_unit_interval(value) -> bool:
    return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _is_positive_int(value) -> bool:
    return not isinstance(value, bool) and int(value) > 0


def _is_nonneg_int(value) -> bool:
    return not isinstance(value, bool) and int(value) >= 0


def _is_finite_positive_nonbool(value) -> bool:
    return (
        not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


_OPTIONAL_RUN_INPUT_CHECKS = (
    ("radius", None, None, _is_finite_positive, "radius must be finite and > 0"),
    (
        "worker_mode",
        "sequential",
        lambda value: str(value or "sequential"),
        lambda value: value in WORKER_MODES,
        f"worker-mode must be one of {WORKER_MODES}",
    ),
    (
        "sigma_mode",
        None,
        None,
        lambda value: value in SIGMA_MODES,
        f"sigma-mode must be one of {SIGMA_MODES}",
    ),
    (
        "lost_strategy",
        None,
        None,
        lambda value: value in LOST_STRATEGIES,
        f"lost-strategy must be one of {LOST_STRATEGIES}",
    ),
    ("mconf_thr", None, None, _is_unit_interval, "mconf-thr must be finite and within [0, 1]"),
    ("coarse_topk", None, None, _is_positive_int, "coarse-topk must be a positive integer"),
    (
        "cache_capacity",
        None,
        None,
        _is_nonneg_int,
        "cache-capacity must be a non-negative integer",
    ),
    (
        "lost_global_retrieval_interval",
        None,
        None,
        _is_nonneg_int,
        "lost-global-retrieval-interval must be a non-negative integer",
    ),
    (
        "runtime_sigma_mode",
        None,
        None,
        lambda value: value in RUNTIME_SIGMA_MODES,
        f"runtime-sigma-mode must be one of {RUNTIME_SIGMA_MODES}",
    ),
    (
        "temporal_feature_cache_size",
        None,
        None,
        _is_nonneg_int,
        "temporal-feature-cache-size must be a non-negative integer",
    ),
    (
        "acquire_stage_mode",
        None,
        None,
        lambda value: value in ACQUIRE_STAGE_MODES,
        f"acquire-stage-mode must be one of {ACQUIRE_STAGE_MODES}",
    ),
    (
        "lost_prior_strategy",
        None,
        None,
        lambda value: value in LOST_PRIOR_STRATEGIES,
        f"lost-prior-strategy must be one of {LOST_PRIOR_STRATEGIES}",
    ),
    (
        "lost_prior_fusion_weight",
        None,
        None,
        _is_finite_positive_nonbool,
        "lost-prior-fusion-weight must be finite and > 0",
    ),
)


def _validate_override_run_inputs(args) -> None:
    for name, default, normalize, predicate, message in _OPTIONAL_RUN_INPUT_CHECKS:
        value = _arg(args, name, default)
        if normalize is not None:
            value = normalize(value)
        if value is None:
            continue
        if not predicate(value):
            raise SystemExit(message)


def _validate_run_inputs(args) -> tuple[Path, Path]:
    if args.stride <= 0 or args.max_frames < 0 or args.max_corr_total < 0:
        raise SystemExit("stride must be > 0; frame/correspondence limits must be >= 0")
    _validate_override_run_inputs(args)
    _reject_untransmitted_production_overrides(args)
    site_path = args.site_profile.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required but unavailable. Check nvidia-smi and confirm the loaded "
            "kernel driver matches libcuda before running this smoke test."
        )
    return site_path, video_path



def _load_replay_profile(args, site_path: Path, video_path: Path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault(
        "SFM_TORCH_HUB_CACHE",
        str(WORKSPACE_ROOT / "執行環境" / "torch_hub_cache"),
    )
    site = load_site_profile(site_path)
    if site.localizer != "edm" or site.query_camera is None:
        raise SystemExit("benchmark requires an EDM site profile with query_camera")

    site_profile_sha256 = sha256_file(site_path)
    video_sha256 = sha256_file(video_path)
    camera = camera_identity(site.query_camera)
    if args.camera_params is not None:
        camera["params"] = list(args.camera_params)
    camera_tuple = (
        camera["model"],
        camera["width"],
        camera["height"],
        camera["params"],
    )
    quality_baseline = None
    if args.quality_baseline is not None:
        baseline_path = args.quality_baseline.expanduser().resolve()
        quality_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not isinstance(quality_baseline, dict):
            raise SystemExit("quality baseline must be a JSON object")
        identity_failures = baseline_identity_failures(
            quality_baseline,
            video_sha256=video_sha256,
            site_profile_sha256=site_profile_sha256,
            bundle_sha256=str(site.asset_sha256.localization_bundle or ""),
            localizer_profile_sha256=str(site.asset_sha256.localizer_profile or ""),
            camera=camera,
        )
        if identity_failures:
            raise SystemExit("quality baseline identity mismatch: " + "; ".join(identity_failures))
    return (
        site,
        camera_tuple,
        site_profile_sha256,
        video_sha256,
        camera,
        quality_baseline,
    )


def _build_replay_runtime(args, site, camera_tuple):
    apply_sigma_mode(args)
    startup_started = time.perf_counter()
    built = build_production_localizer(
        backend="edm",
        bundle=site.localization_bundle,
        bundle_sha256=site.asset_sha256.localization_bundle,
        frame_source=lambda: None,
        camera_tuple=camera_tuple,
        production_profile=site.localizer_profile,
        production_profile_sha256=site.asset_sha256.localizer_profile,
    )
    if args.max_corr_total:
        built.config.max_corr_total = int(args.max_corr_total)
        built.config.validate()
    apply_runtime_overrides(args, built)
    built.tracker.ensure_models()
    _camera, pnp_options = built.tracker.trk._pose_estimation_context()
    pnp_options.ransac.random_seed = int(args.pnp_random_seed)
    startup_ms = (time.perf_counter() - startup_started) * 1000.0
    if args.require_cuda and built.device != "cuda":
        raise SystemExit(f"production localizer selected {built.device!r}, expected 'cuda'")
    return built, startup_ms



def _open_replay_stream(args, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot decode video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        cap.release()
        raise SystemExit(f"video has invalid FPS: {source_fps!r}")
    reported_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_raw_frames = (
        int(round(reported_frames))
        if math.isfinite(reported_frames) and reported_frames > 0.0
        else None
    )
    all_frames_requested = args.max_frames == 0
    if all_frames_requested:
        expected_raw_frames, expected_source = ffprobe_frame_count(video_path)
    else:
        expected_raw_frames, expected_source = None, "partial_run"
    audit = StreamAudit(
        expected_raw_frames=expected_raw_frames,
        expected_source=expected_source,
        capture_opened=True,
        reported_raw_frames=reported_raw_frames,
    )
    return cap, source_fps, all_frames_requested, audit


def _process_replay_frame(
    bgr,
    current_index: int,
    decode_ms: float,
    source_fps: float,
    site,
    built,
    previous_center: np.ndarray | None,
    gpu_span: bool = False,
):
    copy_started = time.perf_counter()
    if bgr.shape[1] != site.query_camera.width or bgr.shape[0] != site.query_camera.height:
        bgr = cv2.resize(
            bgr,
            (site.query_camera.width, site.query_camera.height),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    identity = frame_identity(rgb, current_index)
    rgb = identity["rgb"]
    copy_ms = (time.perf_counter() - copy_started) * 1000.0
    capture_stamp = float(current_index) / source_fps
    _maybe_gpu_sync(gpu_span)
    started = time.perf_counter()
    pose = built.tracker.localize_frame(rgb, capture_stamp=capture_stamp)
    _maybe_gpu_sync(gpu_span)
    wall_ms = (time.perf_counter() - started) * 1000.0
    info = dict(built.tracker.last_info)
    center = None if pose is None else np.asarray([pose.x, pose.y, pose.z], float)
    step = None
    if center is not None and previous_center is not None:
        step = float(np.linalg.norm(center - previous_center))
    next_center = center if center is not None else previous_center
    row = {
        "source_index": current_index,
        "frame_id": identity["frame_id"],
        "frame_sha256": identity["frame_sha256"],
        "capture_stamp": capture_stamp,
        "success": pose is not None,
        "mode": info.get("mode"),
        "next_mode": info.get("next_mode"),
        "rejected": info.get("rejected"),
        "limited_jump": info.get("limited_jump"),
        "limited_jump_confirmed": bool(info.get("limited_jump_confirmed")),
        "inliers": int(info.get("inliers", 0) or 0),
        "reproj_rms": info.get("reproj_rms"),
        "n_corr": info.get("n_corr"),
        "reference_count": info.get("reference_count"),
        "refs": list(info.get("refs") or []),
        "wall_ms": wall_ms,
        "decode_ms": decode_ms,
        "copy_ms": copy_ms,
        "vpr_ms": info.get("vpr_ms"),
        "match_ms": info.get("match_ms"),
        "pnp_ms": info.get("pnp_ms"),
        "step": step,
        "capture_source_age_ms": 0.0,
        "submit_source_age_ms": 0.0,
        "promote_source_age_ms": 0.0,
        "inference_source_age_ms": wall_ms,
        "result_source_age_ms": wall_ms,
        "gpu_span_ms": wall_ms if gpu_span else None,
        "coalesce_drops": 0,
        "submit_drop": 0,
    }
    return row, next_center, step



def _update_replay_cache_stats(row, read_cache_stats, previous_cache_stats):
    if previous_cache_stats is None:
        return None
    stats = read_cache_stats()
    row["ref_feature_cache"] = {
        key: stats[key] - previous_cache_stats[key]
        for key in ("hits", "misses", "evictions")
    }
    return stats


def _append_replay_metrics(
    row,
    wall_values: list[float],
    pnp_values: list[float],
    match_values: list[float],
    inlier_values: list[float],
    reproj_values: list[float],
) -> None:
    wall_values.append(row["wall_ms"])
    for key, target in (
        ("pnp_ms", pnp_values),
        ("match_ms", match_values),
        ("inliers", inlier_values),
        ("reproj_rms", reproj_values),
    ):
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            target.append(float(value))


def _print_replay_progress(selected_count: int, row) -> None:
    if selected_count == 1 or selected_count % 100 == 0:
        print(
            f"frames={selected_count} state={row['next_mode']} "
            f"ok={row['success']} inliers={row['inliers']} wall={row['wall_ms']:.1f}ms",
            flush=True,
        )


def _run_replay(args, cap, source_fps: float, site, built, audit):
    rows = []
    wall_values: list[float] = []
    pnp_values: list[float] = []
    match_values: list[float] = []
    inlier_values: list[float] = []
    reproj_values: list[float] = []
    steps: list[float] = []
    previous_center: np.ndarray | None = None
    # The EDM matcher caches reference backbone features across frames. Its hit
    # rate depends on how often the tracker re-picks the same reference, which is
    # a property of this route and this video, so it belongs in the replay record.
    feature_cache = getattr(
        getattr(getattr(built.tracker, "trk", None), "loc", None), "matcher", None
    )
    read_cache_stats = getattr(feature_cache, "reference_feature_cache_stats", None)
    previous_cache_stats = read_cache_stats() if callable(read_cache_stats) else None
    source_index = 0
    selected_count = 0
    processing_started = time.perf_counter()
    try:
        while not args.max_frames or selected_count < args.max_frames:
            decode_started = time.perf_counter()
            try:
                ok, bgr = cap.read()
            except cv2.error:
                audit.decode_errors += 1
                break
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if not ok:
                break
            current_index = source_index
            source_index += 1
            audit.decoded_raw_frames += 1
            if current_index % args.stride:
                continue
            row, previous_center, step = _process_replay_frame(
                bgr,
                current_index,
                decode_ms,
                source_fps,
                site,
                built,
                previous_center,
                gpu_span=bool(_arg(args, "gpu_span", False)),
            )

            previous_cache_stats = _update_replay_cache_stats(
                row, read_cache_stats, previous_cache_stats)
            rows.append(row)
            selected_count += 1
            audit.sampled_frames += 1
            if step is not None:
                steps.append(step)
            _append_replay_metrics(
                row,
                wall_values,
                pnp_values,
                match_values,
                inlier_values,
                reproj_values,
            )
            _print_replay_progress(selected_count, row)
    finally:
        cap.release()
    return (
        rows,
        wall_values,
        pnp_values,
        match_values,
        inlier_values,
        reproj_values,
        steps,
        previous_cache_stats,
        selected_count,
        processing_started,
    )


def _rows_to_replay_tuple(rows, processing_started, cache_stats=None):
    wall_values: list[float] = []
    pnp_values: list[float] = []
    match_values: list[float] = []
    inlier_values: list[float] = []
    reproj_values: list[float] = []
    steps: list[float] = []
    for row in rows:
        if row.get("step") is not None:
            steps.append(float(row["step"]))
        _append_replay_metrics(
            row, wall_values, pnp_values, match_values, inlier_values, reproj_values,
        )
    return (
        rows,
        wall_values,
        pnp_values,
        match_values,
        inlier_values,
        reproj_values,
        steps,
        cache_stats,
        len(rows),
        processing_started,
    )


def _run_production_path_replay(
    args, cap, source_fps: float, site, audit, *,
    client=None, sleep=time.sleep, monotonic=time.monotonic,
):
    owns_client = client is None
    if owns_client:
        apply_sigma_mode(args)
        client = _open_live_localizer_client(args, site)
    processing_started = time.perf_counter()
    startup_timeout = float(getattr(client, "restart_warmup_s", 20.0) or 20.0)
    try:
        _wait_client_ready(client, startup_timeout, sleep=sleep, monotonic=monotonic)
        decoded = iter_decoded_replay_frames(args, cap, source_fps, site, audit)
        rows, stats = run_source_cadence(
            decoded,
            source_fps,
            client,
            sleep=sleep,
            monotonic=monotonic,
            drain_timeout_s=float(getattr(client, "timeout_s", 8.0) or 8.0),
        )
        attach_production_path_receipt_fields(rows, stats, client)
        return _rows_to_replay_tuple(rows, processing_started)
    finally:
        cap.release()
        if owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                close()



def _finish_replay_stream(args, audit, all_frames_requested, selected_count):
    requested_frames_complete = not args.max_frames or selected_count >= args.max_frames
    if all_frames_requested:
        audit.finish()
    return requested_frames_complete


def _summarize_replay(
    rows,
    wall_values,
    pnp_values,
    match_values,
    inlier_values,
    reproj_values,
    steps,
    previous_cache_stats,
    processing_s,
):
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "frames": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "state_counts": dict(Counter(str(row["next_mode"]) for row in rows)),
        "rejection_counts": dict(Counter(
            str(row["rejected"]) for row in rows if row["rejected"]
        )),
        "limited_jump_confirmed": sum(
            bool(row["limited_jump_confirmed"]) for row in rows
        ),
        "wall_ms": metric_summary(wall_values),
        "pnp_ms": metric_summary(pnp_values),
        "match_ms": metric_summary(match_values),
        "inliers": metric_summary(inlier_values),
        "reproj_rms": metric_summary(reproj_values),
        "accepted_step": metric_summary(steps),
        "ref_feature_cache": None if previous_cache_stats is None else {
            **previous_cache_stats,
            "hit_rate": previous_cache_stats["hits"] / max(
                previous_cache_stats["hits"] + previous_cache_stats["misses"], 1
            ),
        },
        "processing_s": processing_s,
        "processing_fps": len(rows) / processing_s if processing_s > 0.0 else 0.0,
        "pipeline": pipeline_summary(rows),

    }


def _evaluate_replay_quality(
    args,
    summary,
    quality_baseline,
    audit,
    all_frames_requested,
):
    quality_failures: list[str] = []
    if args.quality_baseline is not None:
        assert quality_baseline is not None
        quality_failures = evaluate_quality(summary, quality_baseline)
    if args.accept_known_incomplete and quality_baseline is None:
        raise SystemExit("--accept-known-incomplete requires --quality-baseline")
    stream_integrity = (
        quality_baseline.get("stream_integrity", {})
        if quality_baseline is not None
        else {}
    )
    expected_decoded = int(
        stream_integrity.get(
            "expected_decoded_frames",
            quality_baseline["thresholds"]["frames"] if quality_baseline is not None else 0,
        )
    )
    known_incomplete_accepted = bool(
        args.accept_known_incomplete
        and all_frames_requested
        and audit.reported_raw_frames == expected_decoded + 1
        and audit.decoded_raw_frames == expected_decoded
        and audit.sampled_frames == expected_decoded
        and audit.decode_errors == 1
    )
    return quality_failures, known_incomplete_accepted


def _build_replay_result(
    args,
    site_path,
    video_path,
    site,
    built,
    source_fps,
    site_profile_sha256,
    video_sha256,
    camera,
    startup_ms,
    audit,
    all_frames_requested,
    requested_frames_complete,
    known_incomplete_accepted,
    quality_baseline,
    summary,
    quality_failures,
    rows,
):
    return {
        "schema": "edm-site-replay/v1",
        "site_profile": str(site_path),
        "site_profile_sha256": site_profile_sha256,
        "video": str(video_path),
        "video_sha256": video_sha256,
        "bundle": str(site.localization_bundle),
        "bundle_sha256": site.asset_sha256.localization_bundle,
        "localizer_profile": str(site.localizer_profile),
        "localizer_profile_sha256": site.asset_sha256.localizer_profile,
        "camera": camera,
        "device": built.device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tracker_variant": built.variant,
        "source_fps": source_fps,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "max_corr_total": built.config.max_corr_total,
        "radius": built.config.radius,
        "pnp_random_seed": args.pnp_random_seed,
        "startup_ms": startup_ms,
        "receipt": build_receipt(args, built),
        "pair": {
            "role": _arg(args, "pair_role"),
            "index": _arg(args, "pair_index"),
        },
        "stream_audit": {
            **audit.as_dict(),
            "all_frames_requested": all_frames_requested,
            "requested_max_frames": args.max_frames,
            "requested_frames_complete": requested_frames_complete,
            "known_incomplete_accepted": known_incomplete_accepted,
        },
        "summary": summary,
        "quality_gate": {
            "enabled": quality_baseline is not None,
            "passed": not quality_failures,
            "failures": quality_failures,
            "baseline": (
                str(args.quality_baseline.expanduser().resolve())
                if args.quality_baseline is not None else None
            ),
        },
        "rows": rows,
    }


def _write_replay_report(args, result, summary):
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"localized={summary['successes']}/{summary['frames']} ({summary['success_rate']:.1%}) "
        f"p50/p95={summary['wall_ms']['p50']}/{summary['wall_ms']['p95']}ms "
        f"out={out}",
        flush=True,
    )
    return out


def _report_replay_exit(exit_code, audit, quality_failures):
    if exit_code == 3:
        print(
            "incomplete replay decode: "
            f"expected={audit.expected_raw_frames} "
            f"reported={audit.reported_raw_frames} "
            f"decoded={audit.decoded_raw_frames} "
            f"errors={audit.decode_errors}",
            file=sys.stderr,
            flush=True,
        )
    if exit_code == 4:
        for failure in quality_failures:
            print(f"localization quality regression: {failure}", file=sys.stderr)
    return exit_code


def main() -> int:
    args = parse_args()
    site_path, video_path = _validate_run_inputs(args)
    (
        site,
        camera_tuple,
        site_profile_sha256,
        video_sha256,
        camera,
        quality_baseline,
    ) = _load_replay_profile(args, site_path, video_path)
    worker_mode = str(_arg(args, "worker_mode", "sequential") or "sequential")
    cap, source_fps, all_frames_requested, audit = _open_replay_stream(
        args, video_path)
    if worker_mode == "production-path":
        built = runtime_stub_from_profile(site, args)
        startup_started = time.perf_counter()
        replayed = _run_production_path_replay(
            args, cap, source_fps, site, audit)
        startup_ms = (time.perf_counter() - startup_started) * 1000.0
        if replayed[0]:
            ready_ms = replayed[0][0].get("ready_latency_ms")
            if ready_ms is not None:
                startup_ms = float(ready_ms)
    else:
        built, startup_ms = _build_replay_runtime(args, site, camera_tuple)
        replayed = _run_replay(args, cap, source_fps, site, built, audit)
    (
        rows,
        wall_values,
        pnp_values,
        match_values,
        inlier_values,
        reproj_values,
        steps,
        previous_cache_stats,
        selected_count,
        processing_started,
    ) = replayed
    requested_frames_complete = _finish_replay_stream(
        args, audit, all_frames_requested, selected_count)
    processing_s = time.perf_counter() - processing_started

    summary = _summarize_replay(
        rows,
        wall_values,
        pnp_values,
        match_values,
        inlier_values,
        reproj_values,
        steps,
        previous_cache_stats,
        processing_s,
    )
    receipt = build_receipt(args, built)
    if quality_baseline is not None:
        receipt_failures = receipt_identity_failures(quality_baseline, receipt)
        if receipt_failures:
            raise SystemExit(
                "quality baseline identity mismatch: " + "; ".join(receipt_failures)
            )
    quality_failures, known_incomplete_accepted = _evaluate_replay_quality(
        args,
        summary,
        quality_baseline,
        audit,
        all_frames_requested,
    )
    result = _build_replay_result(
        args,
        site_path,
        video_path,
        site,
        built,
        source_fps,
        site_profile_sha256,
        video_sha256,
        camera,
        startup_ms,
        audit,
        all_frames_requested,
        requested_frames_complete,
        known_incomplete_accepted,
        quality_baseline,
        summary,
        quality_failures,
        rows,
    )
    _write_replay_report(args, result, summary)
    exit_code = _result_exit_code(
        len(rows),
        audit,
        all_frames_requested=all_frames_requested,
        requested_frames_complete=requested_frames_complete,
        quality_failures=quality_failures,
        decode_accepted=known_incomplete_accepted,
    )
    return _report_replay_exit(exit_code, audit, quality_failures)



if __name__ == "__main__":
    raise SystemExit(main())
