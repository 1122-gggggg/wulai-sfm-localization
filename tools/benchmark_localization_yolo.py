#!/usr/bin/env python3
"""Offline production-localizer + production YOLO worker compute timing.

Throughput mode counts a frame only after both workers finish that same frame,
including decode, preprocessing, IPC and box/mask delivery. Paced mode measures
keeping up with a source, not maximum compute FPS. No aircraft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import select
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "控制介面程式"
OPERATOR = CONTROL / "operator_interface"
sys.path[:0] = [str(OPERATOR), str(CONTROL)]


def _stats(values):
    return {
        "count": len(values),
        **{
            key: float(np.percentile(values, percentile)) if values else None
            for key, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
        },
    }


def _gpu_monitor(stop, samples):
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(",")
            if len(values) == 3:
                samples.append(
                    dict(
                        zip(
                            ("util_percent", "memory_mb", "power_w"),
                            (float(value) for value in values),
                            strict=True,
                        )
                    )
                )
        stop.wait(1)


def _wait_ready(clients, timeout=120):
    deadline = time.monotonic() + timeout
    while not all(client.ready for client in clients.values()):
        for name, client in clients.items():
            if client.startup_error or client.unavailable or client.proc.poll() is not None:
                raise RuntimeError(
                    f"{name} startup failed: {client.startup_error}; {client.log_path}"
                )
        if time.monotonic() >= deadline:
            raise TimeoutError("worker startup timeout; inspect worker logs")
        time.sleep(0.01)


def _gpu_stats(gpu_samples):
    return {
        key: _stats([sample[key] for sample in gpu_samples])
        for key in ("util_percent", "memory_mb", "power_w")
    }


def _start_gpu_monitor(stop, gpu_samples):
    monitor = threading.Thread(target=_gpu_monitor, args=(stop, gpu_samples), daemon=True)
    monitor.start()
    return monitor


def _resize_rgb(bgr, camera):
    return cv2.cvtColor(cv2.resize(bgr, (camera.width, camera.height)), cv2.COLOR_BGR2RGB).tobytes()


def _check_paired_row(name, row, seq, completed):
    if row.get("error") or row.get("display_seq") != seq or name in completed:
        raise RuntimeError(f"{name} unpaired/failed result for frame {seq}: {row}")
    if row.get("coalesced_submit_drops", 0) or row.get("restart_reason"):
        raise RuntimeError(f"{name} dropped/restarted during paired measurement: {row}")


def _await_paired_results(clients, seq, frame_start):
    pending = set(clients)
    completed = {}
    deadline = frame_start + 30
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"frame {seq} incomplete: {sorted(pending)}")
        select.select([clients[name].result_notify_fd for name in pending], [], [], remaining)
        for name in tuple(pending):
            client = clients[name]
            client.drain_result_notifications()
            for row in client.poll_results():
                _check_paired_row(name, row, seq, completed)
                completed[name] = row
                pending.remove(name)
    return completed


def _submit_paired_frame(clients, seq, rgb, frame_start):
    for name, client in clients.items():
        if not client.submit(
            seq,
            f"frame-{seq}",
            rgb,
            timing_metadata={"source_frame_stamp_mono": frame_start},
        ):
            raise RuntimeError(f"{name} rejected frame {seq}")


def _throughput_client_stats(name, samples):
    return {
        "completed_frames": len(samples),
        "valid_frames": sum(row.get("success") is True for row in samples),
        "core_ms": _stats([row["wall_ms"] for row in samples]),
        "map_confirmed_frames": sum(
            row.get("success") is True and row.get("direct_status") in {"FAST_TRACK", "RELOC_SEED"}
            for row in samples
        )
        if name == "localization"
        else None,
        "mask_frames": sum(row.get("mask_count", 0) > 0 for row in samples)
        if name == "detector"
        else None,
        "inference_ms": _stats([row["speed"]["inference"] for row in samples])
        if name == "detector"
        else None,
    }


def _throughput_result(
    args, clients, rows, frame_times, completions, measured_start, measured_end, gpu_samples
):
    elapsed = measured_end - measured_start
    valid = sum(
        all(rows[name][index].get("success") is True for name in clients)
        for index in range(args.frames)
    )
    result = {
        "completed_paired_frames": len(frame_times),
        "elapsed_seconds": elapsed,
        "compute_e2e_fps": len(frame_times) / elapsed,
        "valid_paired_fps": valid / elapsed,
        "decode_to_both_results_ms": _stats(frame_times),
        "over_30fps_budget_frames": sum(value > 1000 / 30 for value in frame_times),
        "completed_1s_frames": [
            sum(index <= value < index + 1 for value in completions)
            for index in range(int(elapsed))
        ],
        "dropped_frames": 0,
        "gpu": _gpu_stats(gpu_samples),
    }
    for name, samples in rows.items():
        result[name] = _throughput_client_stats(name, samples)
    return result


def _run_throughput(args, camera, capture, clients, stop, gpu_samples):
    rows = {name: [] for name in clients}
    frame_times = []
    completions = []
    monitor = None
    measured_start = None
    measured_end = None
    try:
        for seq in range(args.warmup_frames + args.frames):
            if seq == args.warmup_frames:
                monitor = _start_gpu_monitor(stop, gpu_samples)
                measured_start = time.monotonic()
            # Begin before decode; the preceding frame has completed all stages.
            frame_start = time.monotonic()
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError("video ended before all paired frames completed")
            rgb = _resize_rgb(bgr, camera)
            _submit_paired_frame(clients, seq, rgb, frame_start)
            completed = _await_paired_results(clients, seq, frame_start)
            frame_end = time.monotonic()
            if seq >= args.warmup_frames:
                for name, row in completed.items():
                    row["decode_to_consumer_ms"] = (frame_end - frame_start) * 1000
                    rows[name].append(row)
                frame_times.append((frame_end - frame_start) * 1000)
                completions.append(frame_end - measured_start)
                measured_end = frame_end
        result = _throughput_result(
            args, clients, rows, frame_times, completions, measured_start, measured_end, gpu_samples
        )
        return result, rows
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=6)


def _start_arm_clients(clients, args, profile, label, camera):
    from live_worker_clients import LiveDetectorClient, LiveLocalizerClient

    clients["localization"] = LiveLocalizerClient(
        OPERATOR / "live_localizer_worker.py",
        args.localizer_python,
        camera.width,
        camera.height,
        profile.localization_bundle,
        megaloc_cache=str(profile.megaloc_cache or ""),
        reference_index=str(profile.reference_index or ""),
        reference_index_sha256=profile.asset_sha256.reference_index or "",
        localizer_backend=profile.localizer,
        localizer_deploy_dir=profile.localizer_deploy_dir or "",
        localizer_profile=profile.localizer_profile or "",
        bundle_sha256=profile.asset_sha256.localization_bundle or "",
        localizer_profile_sha256=profile.asset_sha256.localizer_profile or "",
        query_camera=camera,
        map_align=profile.map_align or "",
    )
    if label == "concurrent":
        clients["detector"] = LiveDetectorClient(
            OPERATOR / "object_detector_worker.py",
            args.yolo_python,
            camera.width,
            camera.height,
            args.model,
            imgsz=args.imgsz,
            task=args.task,
            half=args.precision == "fp16",
        )


def _collect_paced(clients, rows, measured_start, measured_end):
    for name, client in clients.items():
        client.drain_result_notifications()
        for row in client.poll_results():
            if row.get("error") or row.get("worker_restart_reason"):
                raise RuntimeError(f"{name}: {row}")
            stamp = row["source_frame_stamp_mono"]
            receipt = row["client_response_mono"]
            if measured_start <= stamp < measured_end and receipt < measured_end:
                row["source_to_client_ms"] = (receipt - stamp) * 1000
                row["receipt_elapsed_s"] = receipt - measured_start
                rows[name].append(row)


def _read_due_frame(capture, camera, seq, due, warmup_frames):
    # Keep the latest due source frame, never enqueue a stale backlog.
    skipped = 0
    while seq < due:
        if not capture.grab():
            raise RuntimeError("video ended before the measurement window")
        if seq >= warmup_frames:
            skipped += 1
        seq += 1
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError("video ended before the measurement window")
    return _resize_rgb(bgr, camera), seq, skipped


def _submit_paced_frame(clients, seq, frame, stamp):
    for name, client in clients.items():
        if not client.submit(
            seq,
            f"frame-{seq}",
            frame,
            timing_metadata={"source_frame_stamp_mono": stamp},
        ):
            raise RuntimeError(f"{name} rejected frame {seq}")


def _paced_client_stats(name, samples, seconds):
    windows = [
        sum(index <= row["receipt_elapsed_s"] < index + 1 for row in samples)
        for index in range(int(seconds))
    ]
    return {
        "delivered_frames": len(samples),
        "delivered_fps": len(samples) / seconds,
        "valid_fps": sum(row.get("success") is True for row in samples) / seconds,
        "map_confirmed_fps": sum(
            row.get("success") is True and row.get("direct_status") in {"FAST_TRACK", "RELOC_SEED"}
            for row in samples
        )
        / seconds
        if name == "localization"
        else None,
        "delivery_1s_frames": windows,
        "core_ms": _stats([row["wall_ms"] for row in samples]),
        "source_to_client_ms": _stats([row["source_to_client_ms"] for row in samples]),
        "mask_frames": sum(row.get("mask_count", 0) > 0 for row in samples)
        if name == "detector"
        else None,
    }


def _run_paced_window(args, camera, capture, clients, rows, stop, gpu_samples):
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = math.ceil((args.warmup_seconds + args.seconds) * fps)
    warmup_frames = math.ceil(args.warmup_seconds * fps)
    interval = 1 / fps
    started = time.monotonic()
    measured_start = started + warmup_frames * interval
    measured_end = measured_start + args.seconds
    seq = 0
    submitted = 0
    source_skipped = 0
    monitor = None
    try:
        while time.monotonic() < measured_end:
            now = time.monotonic()
            _collect_paced(clients, rows, measured_start, measured_end)
            if now >= measured_start and monitor is None:
                monitor = _start_gpu_monitor(stop, gpu_samples)
            due = min(total - 1, int((now - started) * fps))
            if seq <= due:
                frame, frame_seq, skipped = _read_due_frame(
                    capture, camera, seq, due, warmup_frames
                )
                source_skipped += skipped
                seq = frame_seq + 1
                _submit_paced_frame(clients, frame_seq, frame, started + frame_seq * interval)
                if frame_seq >= warmup_frames:
                    submitted += 1
            time.sleep(min(0.001, max(0, started + seq * interval - time.monotonic())))
        _collect_paced(clients, rows, measured_start, measured_end)
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=6)
    return {
        "source_fps": fps,
        "seconds": args.seconds,
        "submitted_frames": submitted,
        "source_skipped_frames": source_skipped,
    }


def _run_paced(args, camera, capture, clients, stop, gpu_samples, rows):
    result = _run_paced_window(args, camera, capture, clients, rows, stop, gpu_samples)
    result["gpu"] = _gpu_stats(gpu_samples)
    for name, samples in rows.items():
        if name not in clients:
            continue
        samples = list({row["display_seq"]: row for row in samples}.values())
        result[name] = _paced_client_stats(name, samples, args.seconds)
        if not samples:
            raise RuntimeError(f"{name} produced no measurement results")
    return result


def _run_arm(args, profile, label):
    capture = cv2.VideoCapture(str(args.video))
    clients = {}
    stop = threading.Event()
    gpu_samples = []
    rows = {"localization": [], "detector": []}
    try:
        camera = profile.query_camera
        _start_arm_clients(clients, args, profile, label, camera)
        _wait_ready(clients)
        if args.mode == "throughput":
            result, rows = _run_throughput(args, camera, capture, clients, stop, gpu_samples)
        else:
            result = _run_paced(args, camera, capture, clients, stop, gpu_samples, rows)
        result["startup"] = {name: client.startup_info for name, client in clients.items()}
        (args.output / f"{label}_results.json").write_text(
            json.dumps(rows, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        return result
    finally:
        stop.set()
        for client in clients.values():
            client.close()
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "執行環境/models/yolo26n-seg.pt")
    parser.add_argument("--task", choices=("detect", "segment"), default="segment")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp16",
        help="PyTorch precision; TensorRT uses its exported precision",
    )
    parser.add_argument(
        "--arms", choices=("baseline", "concurrent"), nargs="+", default=("baseline", "concurrent")
    )
    parser.add_argument("--localizer-python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--yolo-python", default=str(ROOT / ".venv-yolo/bin/python"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mode", choices=("throughput", "paced"), default="throughput")
    parser.add_argument(
        "--frames", type=int, default=900, help="paired completed frames per throughput arm"
    )
    parser.add_argument("--warmup-frames", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup-seconds", type=float, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/localization_yolo")
    args = parser.parse_args()
    if not args.model.is_file() or args.model.suffix not in {".pt", ".engine"}:
        parser.error("--model must be an existing .pt checkpoint or .engine")
    if (
        not math.isfinite(args.seconds)
        or not math.isfinite(args.warmup_seconds)
        or args.seconds < 1
        or args.warmup_seconds < 0
        or args.imgsz <= 0
        or args.frames < 1
        or args.warmup_frames < 1
    ):
        parser.error("require seconds >= 1, warmup seconds >= 0, imgsz/frames/warmup frames > 0")
    if args.site_profile is None or args.video is None or not args.video.is_file():
        parser.error("--site-profile and an existing offline --video are required")
    from site_profile import load_site_profile

    profile = load_site_profile(args.site_profile)
    if profile.query_camera is None:
        parser.error("profile requires query_camera")
    cap = cv2.VideoCapture(str(args.video))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not math.isfinite(fps) or fps <= 0:
        parser.error("video is unreadable")
    required_frames = (
        args.frames + args.warmup_frames
        if args.mode == "throughput"
        else math.ceil((args.seconds + args.warmup_seconds) * fps)
    )
    if frames < required_frames:
        parser.error("video is too short for warmup + measurement")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "video": str(args.video.resolve()),
        "site_profile": str(profile.source),
        "imgsz": args.imgsz,
        "task": args.task,
        "precision": args.precision if args.model.suffix == ".pt" else "engine-defined",
        "device": "cuda:0",
        "worker": "object_detector_worker.py",
        "mode": args.mode,
        "warmup_seconds": args.warmup_seconds if args.mode == "paced" else None,
        "warmup_frames": args.warmup_frames if args.mode == "throughput" else None,
        "scope": (
            "Unpaced single-frame-in-flight decode/resize/RGB/IPC -> both same-frame "
            "results consumed, including CPU box/mask delivery; no GUI, "
            "camera exposure, telemetry, aircraft or measured overlay."
            if args.mode == "throughput"
            else "Paced source schedule to client receipt; measures source-rate delivery, "
            "not maximum compute FPS. No GUI or camera exposure."
        ),
    }
    for label in args.arms:
        print(f"[{label}] starting", flush=True)
        report[label] = _run_arm(args, profile, label)
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(report[label], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
