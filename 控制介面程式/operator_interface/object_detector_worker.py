#!/usr/bin/env python3
"""Persistent stdin/stdout YOLO detector worker.

Protocol:
  input  : raw RGB frames, fixed width*height*3 bytes each
  output : one JSON line per frame

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

import numpy as np


import sys as _sys
_CTRL = Path(__file__).resolve().parents[1]
if str(_CTRL) not in _sys.path:
    _sys.path.insert(0, str(_CTRL))
from workspace_layout import workspace_from_file  # noqa: E402

_WS = workspace_from_file(__file__)
SYSTEM_ROOT = _WS.root
DETECT_ROOT = _WS.algorithms / "object_detection"
DEFAULT_MODEL = DETECT_ROOT / "models" / "power_equipment_yolo26n_640_fp16.engine"


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
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    out = []
    for (x1, y1, x2, y2), score, class_id in zip(xyxy, conf, cls):
        out.append({
            "class_id": int(class_id),
            "class_name": str(names.get(int(class_id), f"class_{int(class_id)}")),
            "confidence": float(score),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "center": [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)],
            "size": [float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    json_fd = os.dup(sys.stdout.fileno())
    json_out = os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr

    with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO

        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = YOLO(str(model_path), task="detect")

        # Warmup with one black 720p frame. Fixed-batch TensorRT engines expect
        # exactly one image per call, matching the live stream detector loop.
        warm = np.zeros((int(args.height), int(args.width), 3), dtype=np.uint8)
        model.predict(
            source=warm,
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            device=args.device,
            verbose=False,
        )

    print(
        f"[detector_worker] ready model={args.model} imgsz={args.imgsz} "
        f"conf={args.conf} iou={args.iou}",
        file=sys.stderr,
        flush=True,
    )

    frame_size = int(args.width) * int(args.height) * 3
    frame_buffer = bytearray(frame_size)
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
        t0 = time.perf_counter()
        try:
            with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
                results = model.predict(
                    source=frame,
                    imgsz=int(args.imgsz),
                    conf=float(args.conf),
                    iou=float(args.iou),
                    max_det=int(args.max_det),
                    device=args.device,
                    verbose=False,
                )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            worker_core_done_mono_ns = time.monotonic_ns()
            result = results[0]
            boxes = box_payload(result)
            speed = getattr(result, "speed", {}) or {}
            payload = {
                "seq": seq,
                "success": True,
                "wall_ms": wall_ms,
                "worker_read_done_mono": worker_read_done_mono_ns * 1e-9,
                "worker_core_start_mono": worker_core_start_mono_ns * 1e-9,
                "worker_core_done_mono": worker_core_done_mono_ns * 1e-9,
                "worker_read_done_mono_ns": worker_read_done_mono_ns,
                "worker_core_start_mono_ns": worker_core_start_mono_ns,
                "worker_core_done_mono_ns": worker_core_done_mono_ns,
                "count": len(boxes),
                "boxes": boxes,
                "speed": {k: float(v) for k, v in speed.items()},
            }
        except Exception as exc:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            worker_core_done_mono_ns = time.monotonic_ns()
            payload = {
                "seq": seq,
                "success": False,
                "wall_ms": wall_ms,
                "worker_read_done_mono": worker_read_done_mono_ns * 1e-9,
                "worker_core_start_mono": worker_core_start_mono_ns * 1e-9,
                "worker_core_done_mono": worker_core_done_mono_ns * 1e-9,
                "worker_read_done_mono_ns": worker_read_done_mono_ns,
                "worker_core_start_mono_ns": worker_core_start_mono_ns,
                "worker_core_done_mono_ns": worker_core_done_mono_ns,
                "count": 0,
                "boxes": [],
                "error": repr(exc),
            }
            print(f"[detector_worker] detection error: {exc!r}", file=sys.stderr, flush=True)
        json_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        json_out.flush()
        seq += 1
        if args.max_frames and seq >= args.max_frames:
            break


if __name__ == "__main__":
    main()
