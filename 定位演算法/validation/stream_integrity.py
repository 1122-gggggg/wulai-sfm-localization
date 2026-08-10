"""Audited frame decoding for localization quality comparisons."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class StreamAudit:
    expected_raw_frames: int | None = None
    expected_source: str = ""
    integrity_min_sampled_frames: int = 0
    capture_opened: bool = False
    reported_raw_frames: int | None = None
    decoded_raw_frames: int = 0
    sampled_frames: int = 0
    decode_complete: bool | None = None
    decode_errors: int = 0

    def finish(self) -> None:
        bounds = {
            count
            for count in (self.expected_raw_frames, self.reported_raw_frames)
            if count is not None
        }
        if bounds:
            self.decode_complete = all(self.decoded_raw_frames == count for count in bounds)
            if not self.decode_complete:
                self.decode_errors += 1

    def as_dict(self) -> dict:
        return asdict(self)


def ffprobe_frame_count(path: str | Path) -> tuple[int | None, str]:
    """Return a decoded frame count from ffprobe, or ``(None, "")``."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        streams = json.loads(result.stdout).get("streams", [])
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return None, ""
    if len(streams) != 1:
        return None, ""
    for key in ("nb_read_frames", "nb_frames"):
        value = streams[0].get(key)
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count, f"ffprobe_{key}"
    return None, ""


def _to_rgb(
    frame: np.ndarray,
    resize_wh: tuple[int, int] | None,
    audit: StreamAudit,
) -> np.ndarray | None:
    try:
        if resize_wh:
            frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except cv2.error:
        audit.decode_errors += 1
        return None


def _iter_video_frames(
    path: Path,
    stride: int,
    resize_wh: tuple[int, int] | None,
    audit: StreamAudit,
) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    audit.capture_opened = bool(cap.isOpened())
    reported = cap.get(cv2.CAP_PROP_FRAME_COUNT) if audit.capture_opened else 0
    if math.isfinite(reported) and reported > 0:
        audit.reported_raw_frames = int(round(reported))
    if not audit.capture_opened:
        audit.decode_errors += 1
        cap.release()
        audit.finish()
        return
    try:
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error:
                audit.decode_errors += 1
                break
            if not ok:
                break
            index = audit.decoded_raw_frames
            audit.decoded_raw_frames += 1
            if index % stride != 0:
                continue
            rgb = _to_rgb(frame, resize_wh, audit)
            if rgb is None:
                continue
            audit.sampled_frames += 1
            yield rgb
    finally:
        cap.release()
        audit.finish()


def _iter_directory_frames(
    path: Path,
    stride: int,
    resize_wh: tuple[int, int] | None,
    audit: StreamAudit,
) -> Iterator[np.ndarray]:
    files = sorted(path.glob("*.jpg"))
    audit.capture_opened = bool(files)
    audit.reported_raw_frames = len(files)
    if audit.expected_raw_frames is None:
        audit.expected_raw_frames = len(files)
        audit.expected_source = "directory_listing"
    if not files:
        audit.decode_errors += 1
        audit.finish()
        return
    for index, image_path in enumerate(files):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            audit.decode_errors += 1
            continue
        audit.decoded_raw_frames += 1
        if index % stride != 0:
            continue
        rgb = _to_rgb(frame, resize_wh, audit)
        if rgb is None:
            continue
        audit.sampled_frames += 1
        yield rgb
    audit.finish()


def iter_rgb_frames(
    src: str | Path, stride: int, resize_wh: tuple[int, int] | None, audit: StreamAudit
) -> Iterator[np.ndarray]:
    """Yield sampled RGB frames and update ``audit`` through end-of-stream."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    path = Path(src)
    if path.suffix.lower() == ".mp4":
        yield from _iter_video_frames(path, stride, resize_wh, audit)
        return
    yield from _iter_directory_frames(path, stride, resize_wh, audit)
