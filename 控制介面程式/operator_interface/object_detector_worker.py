#!/usr/bin/env python3
"""Persistent stdin/stdout YOLO detection/instance-segmentation worker.

Protocol:
  input  : raw RGB frames, fixed width*height*3 bytes each
  output : ready JSON event after warmup, then one JSON line per frame

Model logs are redirected to stderr so stdout remains machine-readable.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


import sys as _sys
_CTRL = Path(__file__).resolve().parents[1]
if str(_CTRL) not in _sys.path:
    _sys.path.insert(0, str(_CTRL))
from workspace_layout import workspace_from_file  # noqa: E402
from backend_contract import InterfaceMode  # noqa: E402
from runtime_safety import (  # noqa: E402
    configure_offline_environment,
    install_network_guard,
)

_WS = workspace_from_file(__file__)
SYSTEM_ROOT = _WS.root
DEFAULT_MODEL = (
    _WS.runtime / "models/power_equipment_sample50_3class_yolo26n_640_aug_best.pt"
)


def read_exact_into(stream, data: bytearray) -> int:
    """Fill an existing buffer and return the number of bytes received."""
    view = memoryview(data)
    offset = 0
    readinto = getattr(stream, "readinto", None)
    while offset < len(data):
        if callable(readinto):
            count = readinto(view[offset:])
        else:
            chunk = stream.read(len(data) - offset)
            count = len(chunk)
            if count:
                view[offset:offset + count] = chunk
        if not count:
            break
        offset += int(count)
    view.release()
    return offset


def read_exact(stream, size: int) -> bytearray:
    data = bytearray(int(size))
    offset = read_exact_into(stream, data)
    return data if offset == len(data) else data[:offset]


@contextlib.contextmanager
def redirect_native_stdout_to_stderr():
    """Route native fd-1 logs away from the JSON stdout pipe."""
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def box_payload(result) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(result, "names", {}) or {}
    # Boxes.data already holds xyxy/conf/class (and optional tracking ID).
    # Transfer once; three separate .cpu() calls each fence the CUDA stream.
    data = boxes.data.detach().cpu().numpy()
    xyxy, conf, cls = data[:, :4], data[:, -2], data[:, -1]
    masks = getattr(result, "masks", None)
    # Ultralytics Masks.xy performs GPU-to-CPU contour extraction and rescales
    # letterboxed masks into result.orig_shape (the original input pixels).
    polygons = masks.xy if masks is not None else None
    if polygons is not None and len(polygons) != len(xyxy):
        raise ValueError("mask/box counts do not match")
    out = []
    for index, ((x1, y1, x2, y2), score, class_id) in enumerate(zip(xyxy, conf, cls)):
        out.append({
            "class_id": int(class_id),
            "class_name": str(names.get(int(class_id), f"class_{int(class_id)}")),
            "confidence": float(score),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "center": [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)],
            "size": [float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
            "mask_polygon": polygons[index].tolist() if polygons is not None else [],
        })
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--task", choices=("detect", "segment"), default="detect")
    ap.add_argument("--half", action="store_true", help="Use FP16 inference where supported")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-frames", type=int, default=0)
    return ap.parse_args()


def _predict(model, source, args):
    return model.predict(
        source=source,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=args.device,
        quantize=16 if args.half else None,
        verbose=False,
    )


def _load_warm_model(args):
    with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
        import torch
        from ultralytics import YOLO

        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = YOLO(str(model_path), task=args.task)
        if model.task != args.task:
            raise ValueError(
                f"checkpoint task {model.task!r} does not match requested task {args.task!r}"
            )

        # Warmup with one black 720p frame. Fixed-batch TensorRT engines expect
        # exactly one image per call, matching the live stream detector loop.
        warm = np.zeros((int(args.height), int(args.width), 3), dtype=np.uint8)
        warm_results = _predict(model, warm, args)
        backend_task = getattr(model.predictor.model, "task", None)
        if backend_task is not None and backend_task != args.task:
            raise ValueError(
                f"checkpoint task {backend_task!r} does not match requested task {args.task!r}"
            )
        box_payload(warm_results[0])
        if model.predictor.device.type == "cuda":
            torch.cuda.synchronize(model.predictor.device)
    return model, model_path, torch


def _detect_frame(model, torch, frame_bgr, args):
    with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
        results = _predict(model, frame_bgr, args)
        result = results[0]
        boxes = box_payload(result)
        if args.task == "segment" and boxes and getattr(result, "masks", None) is None:
            raise ValueError("segmentation checkpoint returned boxes without masks")
        if model.predictor.device.type == "cuda":
            torch.cuda.synchronize(model.predictor.device)
    return result, boxes


def _timing_fields(read_done_ns: int, core_start_ns: int, core_done_ns: int) -> dict:
    return {
        "worker_read_done_mono": read_done_ns * 1e-9,
        "worker_core_start_mono": core_start_ns * 1e-9,
        "worker_core_done_mono": core_done_ns * 1e-9,
        "worker_read_done_mono_ns": read_done_ns,
        "worker_core_start_mono_ns": core_start_ns,
        "worker_core_done_mono_ns": core_done_ns,
    }


def _process_frame(model, torch, frame, frame_bgr, args, seq, read_done_ns, core_start_ns) -> dict:
    t0 = time.perf_counter()
    try:
        # The pipe carries RGB; Ultralytics ndarray sources are BGR.
        # SIMD conversion reuses the destination without a strided copy.
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, dst=frame_bgr)
        result, boxes = _detect_frame(model, torch, frame_bgr, args)
        # Include CPU box/mask extraction and GPU completion in core timing.
        wall_ms = (time.perf_counter() - t0) * 1000.0
        core_done_ns = time.monotonic_ns()
        speed = getattr(result, "speed", {}) or {}
        return {
            "seq": seq,
            "task": args.task,
            "mask_count": sum(bool(box["mask_polygon"]) for box in boxes),
            "success": True,
            "wall_ms": wall_ms,
            **_timing_fields(read_done_ns, core_start_ns, core_done_ns),
            "count": len(boxes),
            "boxes": boxes,
            "speed": {k: float(v) for k, v in speed.items()},
        }
    except Exception as exc:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        core_done_ns = time.monotonic_ns()
        print(f"[detector_worker] detection error: {exc!r}", file=sys.stderr, flush=True)
        return {
            "seq": seq,
            "success": False,
            "task": args.task,
            "mask_count": 0,
            "wall_ms": wall_ms,
            **_timing_fields(read_done_ns, core_start_ns, core_done_ns),
            "count": 0,
            "boxes": [],
            "error": repr(exc),
        }


def main() -> None:
    configure_offline_environment()
    install_network_guard(InterfaceMode.SIMULATED_STREAM)
    args = _parse_args()

    json_fd = os.dup(sys.stdout.fileno())
    json_out = os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr

    model, model_path, torch = _load_warm_model(args)

    print(
        f"[detector_worker] ready model={args.model} task={args.task} "
        f"half={args.half} imgsz={args.imgsz} conf={args.conf} iou={args.iou}",
        file=sys.stderr,
        flush=True,
    )
    json_out.write(json.dumps({
        "event": "ready",
        "task": args.task,
        "half": bool(model.predictor.model.fp16),
        "model": str(model_path),
    }) + "\n")
    json_out.flush()

    frame_size = int(args.width) * int(args.height) * 3
    frame_buffer = bytearray(frame_size)
    frame_bgr = np.empty((int(args.height), int(args.width), 3), dtype=np.uint8)
    seq = 0
    while True:
        frame_bytes = read_exact_into(sys.stdin.buffer, frame_buffer)
        if not frame_bytes:
            break
        if frame_bytes != frame_size:
            print(f"[detector_worker] partial frame: {frame_bytes}/{frame_size}", file=sys.stderr, flush=True)
            break
        worker_read_done_mono_ns = time.monotonic_ns()
        # The fixed buffer is not refilled until predict() returns.
        frame = np.frombuffer(frame_buffer, dtype=np.uint8).reshape(
            (args.height, args.width, 3))
        worker_core_start_mono_ns = time.monotonic_ns()
        payload = _process_frame(
            model, torch, frame, frame_bgr, args, seq,
            worker_read_done_mono_ns, worker_core_start_mono_ns,
        )
        json_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        json_out.flush()
        seq += 1
        if args.max_frames and seq >= args.max_frames:
            break


if __name__ == "__main__":
    main()
