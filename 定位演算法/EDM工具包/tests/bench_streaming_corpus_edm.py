#!/usr/bin/env python3
"""Benchmark deployment-like video streaming with bounded MegaLoc retrieval.

The reported wall FPS includes video decoding, resize, EDM matching, PnP, metric
collection, and synchronous result transfer.  P123/P126 have no pose ground truth;
their quality metrics are support/continuity proxies, not absolute accuracy.

MegaLoc is allowed once at BOOT and once on entry to each LOST episode. TRACK and
LOW/WEAK must remain on EDM candidates only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
from edm_matcher import EDMMatcher, RUNTIME_SIGMA_MODES  # noqa: E402
from production_edm_tracker import (  # noqa: E402
    ACQUIRE_STAGE_MODES,
    EDMConfig,
    LOST_PRIOR_STRATEGIES,
    ProductionEDMTracker,
)
from reloc_localizer_edm import Camera, EDMRelocMap, MegaLocQuery  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=float), q))


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def discover_videos(roots: list[Path], mapped_paths: set[Path] | None = None) -> list[dict[str, Any]]:
    """Find videos once, deterministically, and assign non-overlapping eval cohorts."""
    mapped = {p.resolve() for p in (mapped_paths or set())}
    paths: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"video root does not exist: {root}")
        paths.update(
            p.resolve()
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )
    rows = []
    for idx, path in enumerate(sorted(paths, key=str), 1):
        parts = {part.lower() for part in path.parts}
        if path in mapped:
            cohort = "mapped_build"
        elif "test" in parts:
            cohort = "heldout_regression"
        else:
            cohort = "excluded_base"
        rows.append({"id": f"V{idx:03d}", "path": str(path), "cohort": cohort})
    return rows


def _longest_run(values: list[str], target: str) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value == target else 0
        best = max(best, current)
    return best


def _retrieval_contract(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate one BOOT retrieval and at most one retrieval per LOST episode."""
    episode = 0
    was_lost = False
    lost_calls: Counter[int] = Counter()
    boot_calls = 0
    unexpected_calls = 0
    observed_calls = 0
    for row in frames:
        state_in = str(row.get("state_in", ""))
        in_lost = state_in == "LOST"
        if in_lost and not was_lost:
            episode += 1
        mode = str(row.get("candidate_mode", ""))
        if mode.startswith("megaloc"):
            observed_calls += 1
            if mode == "megaloc_boot" and state_in == "BOOT_INIT":
                boot_calls += 1
            elif mode == "megaloc_lost" and in_lost:
                lost_calls[episode] += 1
            else:
                unexpected_calls += 1
        was_lost = in_lost
    cumulative_calls = max(
        (int(row.get("global_retrieval_calls", 0)) for row in frames), default=0
    )
    return {
        "lost_episodes": episode,
        "megaloc_boot_calls": boot_calls,
        "megaloc_lost_calls": sum(lost_calls.values()),
        "global_retrieval_calls": cumulative_calls,
        "global_retrieval_contract_ok": (
            boot_calls <= 1
            and all(count <= 1 for count in lost_calls.values())
            and unexpected_calls == 0
            and cumulative_calls == observed_calls
        ),
    }


def summarize_frames(
    frames: list[dict[str, Any]], wall_seconds: float, source_fps: float
) -> dict[str, Any]:
    n = len(frames)
    ok = [row for row in frames if row.get("ok")]
    states = [str(row.get("state_out", "UNKNOWN")) for row in frames]
    total = [float(row["total_ms"]) for row in frames]
    pipeline = [float(row.get("pipeline_ms", row["total_ms"])) for row in frames]
    inliers = [float(row.get("inliers", 0)) for row in ok]
    ncorr = [float(row.get("n_corr", 0)) for row in ok]
    first_fix = next((int(row["frame"]) for row in frames if row.get("ok")), None)
    centers = [
        (int(row["frame"]), np.asarray(row["center"], dtype=float))
        for row in ok
        if row.get("center") is not None
    ]
    steps = [float(np.linalg.norm(bc - ac)) for (_, ac), (_, bc) in zip(centers, centers[1:])]
    step_med = _percentile(steps, 50)
    wall_fps = n / wall_seconds if wall_seconds > 0 else 0.0
    retrieval = _retrieval_contract(frames)
    return {
        "frames": n,
        "localized_frames": len(ok),
        "localized_rate": _rounded(len(ok) / n if n else 0.0),
        "wall_seconds": _rounded(wall_seconds),
        "wall_fps": _rounded(wall_fps),
        "source_fps": _rounded(source_fps),
        "realtime_margin": _rounded(wall_fps / source_fps if source_fps > 0 else 0.0),
        "state_counts": dict(Counter(states)),
        "time_to_first_fix_frames": first_fix,
        "time_to_first_fix_seconds": _rounded(first_fix / source_fps) if first_fix and source_fps else None,
        "lost_recoveries": sum(
            bool(row.get("ok")) and row.get("state_in") == "LOST" for row in frames
        ),
        "longest_lost_run_frames": _longest_run(states, "LOST"),
        **retrieval,
        "latency_ms": {
            "tracker_p50": _rounded(_percentile(total, 50)),
            "tracker_p95": _rounded(_percentile(total, 95)),
            "tracker_p99": _rounded(_percentile(total, 99)),
            "pipeline_p50": _rounded(_percentile(pipeline, 50)),
            "pipeline_p95": _rounded(_percentile(pipeline, 95)),
            "vpr_total": _rounded(sum(float(row.get("vpr_ms", 0.0)) for row in frames)),
            "match_p50": _rounded(_percentile([float(row.get("match_ms", 0.0)) for row in frames], 50)),
            "match_p95": _rounded(_percentile([float(row.get("match_ms", 0.0)) for row in frames], 95)),
            "pnp_p50": _rounded(_percentile([float(row.get("pnp_ms", 0.0)) for row in frames], 50)),
            "pnp_p95": _rounded(_percentile([float(row.get("pnp_ms", 0.0)) for row in frames], 95)),
        },
        "inliers_median": _rounded(_percentile(inliers, 50)),
        "inliers_p05": _rounded(_percentile(inliers, 5)),
        "correspondences_median": _rounded(_percentile(ncorr, 50)),
        "step_median": _rounded(step_med),
        "step_p95": _rounded(_percentile(steps, 95)),
        "step_max": _rounded(max(steps) if steps else None),
        "jumps_gt_10x_median": (
            int(np.sum(np.asarray(steps) > 10.0 * step_med)) if steps and step_med is not None else None
        ),
        "limited_jumps": sum(bool(row.get("limited_jump")) for row in frames),
        "rejections": dict(Counter(str(row["rejected"]) for row in frames if row.get("rejected"))),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _nvidia_smi(query: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def hardware_info() -> dict[str, Any]:
    gpu = _nvidia_smi("name,memory.total,driver_version,pci.bus_id")
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "gpu": gpu,
    }


class GPUSampler:
    def __init__(self, interval_seconds: float = 0.5):
        self.interval = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        fields = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"
        while not self._stop.is_set():
            lines = _nvidia_smi(fields)
            if lines:
                try:
                    vals = [float(part.strip()) for part in lines[0].split(",")]
                    self.samples.append(dict(zip(
                        ("utilization_pct", "memory_mib", "power_w", "temperature_c", "sm_clock_mhz"),
                        vals,
                    )))
                except ValueError:
                    pass
            self._stop.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        out: dict[str, Any] = {"samples": len(self.samples)}
        for key in ("utilization_pct", "memory_mib", "power_w", "temperature_c", "sm_clock_mhz"):
            vals = [row[key] for row in self.samples]
            out[key] = {
                "median": _rounded(_percentile(vals, 50)),
                "p95": _rounded(_percentile(vals, 95)),
                "max": _rounded(max(vals) if vals else None),
            }
        return out


def _load_mapped_paths(manifest_path: Path, data_root: Path) -> set[Path]:
    if not manifest_path.exists():
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {(data_root / row["rel"]).resolve() for row in manifest.get("build", [])}


def _tracker_config(args: argparse.Namespace) -> EDMConfig:
    kwargs = dict(
        global_retrieval_policy=getattr(args, "lost_strategy", None) or "boot_and_lost_once",
        boot_global_topk=args.boot_global_topk,
        lost_local_topk=args.lost_local_topk,
        lost_local_grace_frames=args.lost_local_grace_frames,
        recovery_bank_size=args.recovery_bank_size,
        recovery_scan_topk=args.recovery_scan_topk,
        match_batch_size=args.match_batch_size,
        use_temporal_reference=args.temporal_reference,
        temporal_map_topk=args.temporal_map_topk,
        local_topk=args.local_topk,
        weak_local_topk=args.weak_local_topk,
        max_corr_total=args.max_corr_total,
        acquire_min_inliers=args.acquire_min_inliers,
        track_min_inliers=args.track_min_inliers,
        weak_min_inliers=args.weak_min_inliers,
    )
    radius = getattr(args, "radius", None)
    if radius is not None:
        kwargs["radius"] = float(radius)
    interval = getattr(args, "lost_global_retrieval_interval", None)
    if interval is not None:
        kwargs["lost_global_retrieval_interval"] = int(interval)
    acquire_mode = getattr(args, "acquire_stage_mode", None)
    if acquire_mode is not None:
        kwargs["acquire_stage_mode"] = str(acquire_mode)
    lost_prior = getattr(args, "lost_prior_strategy", None)
    if lost_prior is not None:
        kwargs["lost_prior_strategy"] = str(lost_prior)
    fusion_weight = getattr(args, "lost_prior_fusion_weight", None)
    if fusion_weight is not None:
        kwargs["lost_prior_fusion_weight"] = float(fusion_weight)
    cfg = EDMConfig(**kwargs)
    cfg.validate()
    return cfg


def effective_fused_coarse_mode(args) -> str:
    sigma = getattr(args, "sigma_mode", None)
    if sigma in {"fused", "upstream"}:
        return str(sigma)
    raw = os.environ.get("SFM_EDM_FUSED_COARSE", "1").strip().lower()
    return "upstream" if raw in {"0", "false", "no", "off"} else "fused"


def apply_sigma_mode(args) -> None:
    sigma = getattr(args, "sigma_mode", None)
    if sigma == "upstream":
        os.environ["SFM_EDM_FUSED_COARSE"] = "0"
    elif sigma == "fused":
        os.environ["SFM_EDM_FUSED_COARSE"] = "1"


def _fail_closed_untransmitted_production_path(args) -> None:
    pending = [
        key for key in (
            "runtime_sigma_mode",
            "temporal_feature_cache_size",
            "acquire_stage_mode",
            "lost_prior_strategy",
            "lost_prior_fusion_weight",
        )
        if getattr(args, key, None) is not None
    ]
    if pending:
        raise SystemExit(
            "production-path cannot apply untransmitted overrides: "
            + ", ".join(pending)
        )


def _matcher_overrides(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if getattr(args, "cache_capacity", None) is not None:
        kwargs["reference_cache_size"] = int(args.cache_capacity)
    if getattr(args, "runtime_sigma_mode", None) is not None:
        kwargs["runtime_sigma_mode"] = str(args.runtime_sigma_mode)
    if getattr(args, "temporal_feature_cache_size", None) is not None:
        kwargs["temporal_feature_cache_size"] = int(args.temporal_feature_cache_size)
    return kwargs


def build_receipt(args: argparse.Namespace, cfg=None, matcher=None) -> dict[str, Any]:
    radius = getattr(args, "radius", None)
    if radius is None:
        radius = getattr(cfg, "radius", None)
    mconf = getattr(args, "mconf_thr", None)
    if mconf is None:
        mconf = getattr(matcher, "mconf_thr", None)
    topk = getattr(args, "edm_topk", None)
    if topk is None:
        topk = getattr(args, "coarse_topk", None)
    if topk is None:
        topk = getattr(matcher, "topk", None)
    cache = getattr(args, "cache_capacity", None)
    if cache is None:
        cache = getattr(matcher, "reference_cache_size", None)
    lost_strategy = getattr(args, "lost_strategy", None)
    if lost_strategy is None:
        lost_strategy = getattr(cfg, "global_retrieval_policy", None)
    if lost_strategy is None:
        lost_strategy = "boot_and_lost_once"
    interval = getattr(args, "lost_global_retrieval_interval", None)
    if interval is None:
        interval = getattr(cfg, "lost_global_retrieval_interval", None)
    if interval is None:
        interval = 0
    runtime_sigma = getattr(args, "runtime_sigma_mode", None)
    if runtime_sigma is None:
        runtime_sigma = getattr(matcher, "runtime_sigma_mode", None)
    temporal_cache = getattr(args, "temporal_feature_cache_size", None)
    if temporal_cache is None:
        temporal_cache = getattr(matcher, "temporal_feature_cache_size", None)
    acquire_mode = getattr(args, "acquire_stage_mode", None)
    if acquire_mode is None:
        acquire_mode = getattr(cfg, "acquire_stage_mode", None)
    lost_prior = getattr(args, "lost_prior_strategy", None)
    if lost_prior is None:
        lost_prior = getattr(cfg, "lost_prior_strategy", None)
    fusion_weight = getattr(args, "lost_prior_fusion_weight", None)
    if fusion_weight is None:
        fusion_weight = getattr(cfg, "lost_prior_fusion_weight", None)
    return {
        "radius": None if radius is None else float(radius),
        "mconf_thr": None if mconf is None else float(mconf),
        "coarse_topk": None if topk is None else int(topk),
        "cache_capacity": None if cache is None else int(cache),
        "lost_strategy": str(lost_strategy),
        "lost_global_retrieval_interval": int(interval),
        "fused_coarse_mode": effective_fused_coarse_mode(args),
        "runtime_sigma_mode": None if runtime_sigma is None else str(runtime_sigma),
        "temporal_feature_cache_size": (
            None if temporal_cache is None else int(temporal_cache)
        ),
        "acquire_stage_mode": None if acquire_mode is None else str(acquire_mode),
        "lost_prior_strategy": None if lost_prior is None else str(lost_prior),
        "lost_prior_fusion_weight": (
            None if fusion_weight is None else float(fusion_weight)
        ),
        "worker_mode": getattr(args, "worker_mode", "sequential") or "sequential",
    }


def _replay_module():
    validation = ROOT.parent / "validation"
    if str(validation) not in sys.path:
        sys.path.insert(0, str(validation))
    return __import__("benchmark_edm_site_replay")


def _corpus_site(camera: Camera, bundle: Path) -> SimpleNamespace:
    return SimpleNamespace(
        query_camera=camera,
        localization_bundle=Path(bundle),
        megaloc_cache="",
        reference_index="",
        asset_sha256=SimpleNamespace(
            reference_index="",
            localization_bundle="",
            localizer_profile="",
        ),
        localizer_deploy_dir="",
        localizer_profile="",
        map_align="",
    )


def _corpus_frame_from_worker_row(row: dict[str, Any], frame_index: int) -> dict[str, Any]:
    pose = row.get("pose")
    center = None
    if isinstance(pose, dict) and pose.get("x") is not None:
        center = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    wall = float(row.get("wall_ms") or 0.0)
    decode = float(row.get("decode_ms") or 0.0)
    copy = float(row.get("copy_ms") or 0.0)
    source_index = row.get("source_index")
    frame = int(source_index) if source_index is not None else frame_index
    ok = bool(row.get("ok") if row.get("ok") is not None else row.get("success"))
    return {
        "frame": frame,
        "ok": ok,
        "state_in": row.get("state_in") or row.get("mode") or "",
        "state_out": row.get("state_out") or row.get("next_mode") or "",
        "candidate_mode": row.get("candidate_mode") or "",
        "n_corr": row.get("n_corr"),
        "inliers": row.get("inliers"),
        "global_retrieval_calls": int(row.get("global_retrieval_calls") or 0),
        "temporal_used": row.get("temporal_used"),
        "decode_ms": decode,
        "resize_ms": copy,
        "copy_ms": copy,
        "vpr_ms": float(row.get("vpr_ms") or 0.0),
        "match_ms": float(row.get("match_ms") or 0.0),
        "pnp_ms": float(row.get("pnp_ms") or 0.0),
        "total_ms": wall,
        "pipeline_ms": decode + copy + wall,
        "center": center,
        "limited_jump": row.get("limited_jump"),
        "rejected": row.get("rejected"),
        "refs": list(row.get("refs") or []),
        "selected_ref": row.get("selected_ref"),
        "frame_id": row.get("frame_id"),
        "frame_sha256": row.get("frame_sha256"),
        "capture_source_age_ms": row.get("capture_source_age_ms"),
        "submit_source_age_ms": row.get("submit_source_age_ms"),
        "promote_source_age_ms": row.get("promote_source_age_ms"),
        "inference_source_age_ms": row.get("inference_source_age_ms"),
        "result_source_age_ms": row.get("result_source_age_ms"),
        "coalesce_drops": int(row.get("coalesce_drops") or 0),
        "submit_drop": int(row.get("submit_drop") or 0),
        "restart_required": row.get("restart_required"),
        "restart_reason": row.get("restart_reason"),
        "gpu_span_ms": row.get("gpu_span_ms"),
    }


def run_production_path_video(
    video: dict[str, Any],
    camera: Camera,
    bundle: Path,
    args: argparse.Namespace,
    out_dir: Path,
    max_frames: int,
    progress_every: int,
) -> dict[str, Any]:
    replay = _replay_module()
    path = Path(video["path"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        cap.release()
        raise RuntimeError(f"video has invalid FPS: {source_fps!r}")
    site = _corpus_site(camera, bundle)
    client = None
    closed = False
    stats: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    sampler = GPUSampler()
    sampler.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    try:
        replay.apply_sigma_mode(args)
        client = replay._open_live_localizer_client(args, site)
        replay._wait_client_ready(
            client,
            float(getattr(client, "restart_warmup_s", 20.0) or 20.0),
            sleep=time.sleep,
            monotonic=time.monotonic,
        )
        audit = SimpleNamespace(decoded_raw_frames=0, sampled_frames=0, decode_errors=0)
        args_proxy = SimpleNamespace(max_frames=max_frames, stride=1)
        decoded = replay.iter_decoded_replay_frames(
            args_proxy, cap, source_fps, site, audit,
        )
        rows, stats = replay.run_source_cadence(decoded, source_fps, client)
        replay.attach_production_path_receipt_fields(rows, stats, client)
        frames = [
            _corpus_frame_from_worker_row(row, index)
            for index, row in enumerate(rows)
        ]
        if progress_every and frames:
            rate = 100.0 * sum(bool(x.get("ok")) for x in frames) / len(frames)
            print(
                f"  {video['id']} {len(frames)}/{source_frames}: "
                f"localized={rate:.1f}% coalesce={stats.get('coalesce_drops', 0)}",
                flush=True,
            )
    finally:
        cap.release()
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()
            closed = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    wall_seconds = time.perf_counter() - wall_start
    gpu = sampler.stop()
    summary = summarize_frames(frames, wall_seconds, source_fps)
    summary.update({
        **video,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "source_frames": source_frames,
        "source_width": source_width,
        "source_height": source_height,
        "gpu": gpu,
        "worker_closed": closed,
        "coalesce_drops": int(stats.get("coalesce_drops", 0) or 0),
        "torch_peak_allocated_mib": _rounded(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        "torch_peak_reserved_mib": _rounded(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else None,
    })
    _frame_csv(out_dir / "frames" / f"{video['id']}.csv", frames)
    _atomic_json(out_dir / "videos" / f"{video['id']}.json", summary)
    return summary




def _frame_csv(path: Path, frames: list[dict[str, Any]]) -> None:
    fields = [
        "frame", "ok", "state_in", "state_out", "candidate_mode", "n_corr", "inliers",
        "global_retrieval_calls",
        "temporal_used",
        "decode_ms", "resize_ms", "vpr_ms", "match_ms", "pnp_ms", "total_ms", "pipeline_ms",
        "center_x", "center_y", "center_z", "limited_jump", "rejected", "refs",
        "selected_ref",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in frames:
            center = row.get("center")
            flat = {key: row.get(key) for key in fields}
            flat["center_x"] = center[0] if center is not None else None
            flat["center_y"] = center[1] if center is not None else None
            flat["center_z"] = center[2] if center is not None else None
            flat["limited_jump"] = json.dumps(row.get("limited_jump"), ensure_ascii=False)
            flat["refs"] = "|".join(row.get("refs", []))
            writer.writerow(flat)


def run_video(
    video: dict[str, Any], rmap: EDMRelocMap, camera: Camera, tracker_cfg: EDMConfig,
    matcher: EDMMatcher, megaloc: MegaLocQuery, out_dir: Path, max_frames: int,
    progress_every: int, *, gpu_span: bool = False,
) -> dict[str, Any]:
    path = Path(video["path"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = ProductionEDMTracker(rmap, camera, tracker_cfg, matcher=matcher, megaloc=megaloc)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sampler = GPUSampler()
    sampler.start()
    frames: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    while not max_frames or len(frames) < max_frames:
        frame_start = time.perf_counter()
        ok_read, frame = cap.read()
        decode_end = time.perf_counter()
        if not ok_read:
            break
        if (frame.shape[1], frame.shape[0]) != (camera.width, camera.height):
            frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
        resize_end = time.perf_counter()
        if torch.cuda.is_available() and gpu_span:
            torch.cuda.synchronize()
        info = tracker.localize(frame)
        if torch.cuda.is_available() and gpu_span:
            torch.cuda.synchronize()
        end = time.perf_counter()
        row = dict(info)
        row["center"] = info["center"].tolist() if info.get("center") is not None else None
        row["decode_ms"] = (decode_end - frame_start) * 1e3
        row["resize_ms"] = (resize_end - decode_end) * 1e3
        row["copy_ms"] = row["resize_ms"]
        row["pipeline_ms"] = (end - frame_start) * 1e3
        row["gpu_span_ms"] = (end - resize_end) * 1e3 if gpu_span else None
        row["frame_id"] = f"src_{int(row.get('frame', len(frames))):08d}"
        row["capture_source_age_ms"] = 0.0
        row["submit_source_age_ms"] = 0.0
        row["promote_source_age_ms"] = 0.0
        row["inference_source_age_ms"] = float(row.get("total_ms") or 0.0)
        row["result_source_age_ms"] = float(row.get("pipeline_ms") or 0.0)
        row["coalesce_drops"] = 0
        row["submit_drop"] = 0
        frames.append(row)
        if progress_every and len(frames) % progress_every == 0:
            recent = frames[-progress_every:]
            rate = 100.0 * sum(bool(x.get("ok")) for x in recent) / len(recent)
            print(
                f"  {video['id']} {len(frames)}/{source_frames}: "
                f"localized={rate:.1f}% tracker_p50={np.median([x['total_ms'] for x in recent]):.1f}ms",
                flush=True,
            )
    wall_seconds = time.perf_counter() - wall_start
    cap.release()
    gpu = sampler.stop()
    summary = summarize_frames(frames, wall_seconds, source_fps)
    summary.update({
        **video,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "source_frames": source_frames,
        "source_width": source_width,
        "source_height": source_height,
        "gpu": gpu,
        "torch_peak_allocated_mib": _rounded(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        "torch_peak_reserved_mib": _rounded(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else None,
    })
    _frame_csv(out_dir / "frames" / f"{video['id']}.csv", frames)
    _atomic_json(out_dir / "videos" / f"{video['id']}.json", summary)
    return summary



def _aggregate(videos: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sum(int(row["frames"]) for row in videos)
    localized = sum(int(row["localized_frames"]) for row in videos)
    wall = sum(float(row["wall_seconds"]) for row in videos)
    return {
        "videos": len(videos),
        "frames": frames,
        "localized_frames": localized,
        "localized_rate": _rounded(localized / frames if frames else 0.0),
        "wall_seconds": _rounded(wall),
        "wall_fps": _rounded(frames / wall if wall else 0.0),
        "all_global_retrieval_contract_ok": all(row["global_retrieval_contract_ok"] for row in videos),
        "global_retrieval_calls_total": sum(int(row["global_retrieval_calls"]) for row in videos),
        "cohorts": dict(Counter(row["cohort"] for row in videos)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_package = ROOT / "transfer" / "edm_localization_package_target_site_v1_20260718"
    parser.add_argument("--config", default=str(default_package / "config.json"))
    parser.add_argument("--bundle")
    parser.add_argument("--base", required=True)
    parser.add_argument("--updates", required=True)
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--video", action="append", default=[], help="video ID or filename substring; repeatable")
    parser.add_argument("--cohort", action="append", choices=["mapped_build", "excluded_base", "heldout_regression"])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=300)
    parser.add_argument("--local-topk", type=int, default=1)
    parser.add_argument("--weak-local-topk", type=int, default=3)
    parser.add_argument("--lost-local-topk", type=int, default=5)
    parser.add_argument("--lost-local-grace-frames", type=int, default=12)
    parser.add_argument("--recovery-bank-size", type=int, default=192)
    parser.add_argument("--recovery-scan-topk", type=int, default=2)
    parser.add_argument("--boot-global-topk", type=int, default=10)
    parser.add_argument("--match-batch-size", type=int, default=2)
    parser.add_argument("--temporal-reference", action="store_true")
    parser.add_argument("--temporal-map-topk", type=int, default=1)
    parser.add_argument("--max-corr-total", type=int, default=900)
    parser.add_argument("--acquire-min-inliers", type=int, default=80)
    parser.add_argument("--track-min-inliers", type=int, default=50)
    parser.add_argument("--weak-min-inliers", type=int, default=30)
    parser.add_argument("--edm-topk", type=int)
    parser.add_argument("--mconf-thr", type=float, default=0.2)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--worker-mode", choices=["sequential", "production-path"], default="sequential",
    )
    parser.add_argument("--sigma-mode", choices=["fused", "upstream"])
    parser.add_argument(
        "--runtime-sigma-mode", choices=list(RUNTIME_SIGMA_MODES),
    )
    parser.add_argument("--temporal-feature-cache-size", type=int)
    parser.add_argument(
        "--acquire-stage-mode", choices=list(ACQUIRE_STAGE_MODES),
    )
    parser.add_argument(
        "--lost-prior-strategy", choices=list(LOST_PRIOR_STRATEGIES),
    )
    parser.add_argument("--lost-prior-fusion-weight", type=float)
    parser.add_argument("--cache-capacity", type=int)
    parser.add_argument("--radius", type=float)
    parser.add_argument(
        "--lost-strategy", choices=["boot_and_lost_once", "boot_once"],
    )
    parser.add_argument("--lost-global-retrieval-interval", type=int)
    parser.add_argument(
        "--gpu-span", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--pair-role", choices=["A", "B"])
    parser.add_argument("--pair-index", type=int)
    parser.add_argument("--live-worker", type=Path, help="production-path worker script")
    parser.add_argument("--live-python", help="production-path worker interpreter")
    args = parser.parse_args()
    apply_sigma_mode(args)

    out_dir = Path(args.out_dir).resolve()
    cfg_path = Path(args.config).resolve()
    package_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    pc = package_cfg["pnp_camera"]
    camera = Camera(model=pc["model"], width=pc["width"], height=pc["height"], params=list(pc["params"]))
    bundle = Path(args.bundle).resolve() if args.bundle else (cfg_path.parent / package_cfg["paths"]["bundle"]).resolve()
    roots = [Path(args.base).resolve(), Path(args.updates).resolve()]
    data_root = Path(args.base).resolve().parent
    mapped = _load_mapped_paths(Path(args.corpus_manifest), data_root)
    videos = discover_videos(roots, mapped)
    if args.cohort:
        videos = [row for row in videos if row["cohort"] in args.cohort]
    if args.video:
        terms = [term.lower() for term in args.video]
        videos = [row for row in videos if any(term == row["id"].lower() or term in row["path"].lower() for term in terms)]
    if not videos:
        raise SystemExit("no videos selected")

    production_path = str(getattr(args, "worker_mode", "sequential") or "sequential") == "production-path"
    if production_path:
        _fail_closed_untransmitted_production_path(args)
    tracker_cfg = _tracker_config(args)
    matcher = None
    if production_path:
        startup = {
            "map_load_seconds": None,
            "matcher_load_seconds": None,
            "megaloc_load_seconds": None,
        }
        print("startup: production-path launches one LiveLocalizerClient per video", flush=True)
    else:
        t0 = time.perf_counter()
        rmap = EDMRelocMap.load(bundle)
        map_load_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        matcher = EDMMatcher(
            mconf_thr=args.mconf_thr, topk=args.edm_topk, fp16=not args.fp32,
            **_matcher_overrides(args),
        )
        matcher_load_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        megaloc = MegaLocQuery()
        megaloc_load_seconds = time.perf_counter() - t0
        startup = {
            "map_load_seconds": _rounded(map_load_seconds),
            "matcher_load_seconds": _rounded(matcher_load_seconds),
            "megaloc_load_seconds": _rounded(megaloc_load_seconds),
        }
        print(f"startup: {startup}", flush=True)

    run_cfg = {
        "schema": "edm-streaming-benchmark/v1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": hardware_info(),
        "bundle": str(bundle),
        "package_config": str(cfg_path),
        "tracker": vars(tracker_cfg),
        "matcher": {"topk": args.edm_topk, "mconf_thr": args.mconf_thr, "fp16": not args.fp32},
        "receipt": build_receipt(
            args,
            tracker_cfg if not production_path else None,
            matcher,
        ),
        "pair": {"role": args.pair_role, "index": args.pair_index},
        "input": {"roots": [str(root) for root in roots], "videos": videos, "max_frames": args.max_frames},
        "metric_scope": {
            "wall_fps": "decode + resize + tracker + metric collection, sequential frames",
            "quality": "support/continuity proxy unless external pose ground truth is supplied",
            "heldout_note": "P123/P126 were used by earlier temporal-ceiling work; final use is regression, not untouched validation",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "run_config.json", run_cfg)
    print(json.dumps({"hardware": run_cfg["hardware"], "videos": videos}, ensure_ascii=False, indent=2))

    summaries = []
    for video in videos:
        print(f"\n[{video['id']}] {video['cohort']} {video['path']}", flush=True)
        if production_path:
            summary = run_production_path_video(
                video, camera, bundle, args, out_dir,
                max_frames=args.max_frames, progress_every=args.progress_every,
            )
        else:
            summary = run_video(
                video, rmap, camera, tracker_cfg, matcher, megaloc, out_dir,
                max_frames=args.max_frames, progress_every=args.progress_every,
                gpu_span=bool(args.gpu_span),
            )

        summaries.append(summary)
        _atomic_json(out_dir / "summary.json", {
            "schema": "edm-streaming-benchmark/v1",
            "status": "partial",
            "startup": startup,
            "receipt": run_cfg["receipt"],
            "aggregate": _aggregate(summaries),
            "videos": summaries,
        })
        print(
            f"done: localized={100*summary['localized_rate']:.1f}% wall={summary['wall_fps']:.1f} FPS "
            f"realtime={summary['realtime_margin']:.2f}x MegaLoc_calls={summary['global_retrieval_calls']}",
            flush=True,
        )
    result = {
        "schema": "edm-streaming-benchmark/v1",
        "status": "complete",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "startup": startup,
        "receipt": run_cfg["receipt"],
        "aggregate": _aggregate(summaries),
        "videos": summaries,
    }
    _atomic_json(out_dir / "summary.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
