#!/usr/bin/env python3
"""ReplaySystem — offline test from recorded camera + telemetry.

Saves `telemetry.jsonl` + images, replays with same interfaces.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class ReplayFrame:
    timestamp: float
    image: np.ndarray | None
    velocity_ned: tuple[float, float, float] | None
    attitude: tuple[float, float, float] | None  # roll,pitch,yaw
    altitude: float | None = None
class ReplaySystem:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.telemetry_path = self.log_dir / "telemetry.jsonl"
        self.images_dir = self.log_dir / "images"
        # keep explicit video_path alias for spec compliance (via cv2.VideoCapture, matched by index)
        # Use property fallback to _find_video; store None initially to allow override
        self._video_path: Path | None = None

    @property
    def video_path(self) -> Path | None:
        # expose video path per spec: self.video_path via cv2.VideoCapture
        if self._video_path is not None:
            return self._video_path
        return self._find_video()

    @video_path.setter
    def video_path(self, value: str | Path | None) -> None:
        self._video_path = Path(value) if value is not None else None

    def save_telemetry(self, sample: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.telemetry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _list_images(self) -> list[Path]:
        if not self.images_dir.exists() or not self.images_dir.is_dir():
            return []
        exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
        paths: list[Path] = []
        for pat in exts:
            paths.extend(self.images_dir.glob(pat))
        # deduplicate and sort lexicographically
        uniq = sorted({p.resolve(): p for p in paths}.values(), key=lambda p: p.name)
        # fallback: sort by name ensures timestamp index order
        return uniq

    def _find_video(self) -> Path | None:
        candidates = [
            self.log_dir / "video.mp4",
            self.log_dir / "video.avi",
            self.log_dir / "video.mkv",
            self.log_dir / "video.MP4",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        # glob any video
        for pat in ("*.mp4", "*.avi", "*.mkv", "*.MP4", "*.AVI"):
            found = sorted(self.log_dir.glob(pat))
            if found:
                return found[0]
        return None

    def load_image(self, frame_id: int | str) -> np.ndarray | None:
        """Load image by index or filename.

        - int: matched by timestamp index (sorted images_dir/*.jpg order, or video frame index)
        - str: filename or stem relative to images_dir / log_dir
        """
        # lazy import cv2 so environment without opencv doesn't crash on import
        try:
            import cv2  # type: ignore
        except Exception:
            return None

        # int -> index: try images_dir/*.jpg via cv2.imread first, then video_path via cv2.VideoCapture matched by index
        if isinstance(frame_id, int):
            idx = int(frame_id)
            images = self._list_images()
            if 0 <= idx < len(images):
                try:
                    img = cv2.imread(str(images[idx]), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                except Exception:
                    pass
            # fallback to video (self.video_path per spec, else _find_video)
            video = self.video_path
            if video is not None and video.exists():
                try:
                    cap = cv2.VideoCapture(str(video))
                    if cap.isOpened():
                        # seek to frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ok, frame = cap.read()
                        cap.release()
                        if ok and frame is not None:
                            return frame
                    else:
                        cap.release()
                except Exception:
                    try:
                        cap.release()
                    except Exception:
                        pass
            else:
                # try _find_video explicitly if property returned None
                video2 = self._find_video()
                if video2 is not None:
                    try:
                        cap = cv2.VideoCapture(str(video2))
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ok, frame = cap.read()
                            cap.release()
                            if ok and frame is not None:
                                return frame
                        else:
                            cap.release()
                    except Exception:
                        try:
                            cap.release()
                        except Exception:
                            pass
            return None
        else:
            # str -> filename
            name = str(frame_id).strip()
            if not name:
                return None
            p = Path(name)
            # if absolute and exists
            if p.is_absolute() and p.exists():
                try:
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                except Exception:
                    pass
                return None
            # try relative to images_dir and log_dir
            candidates: list[Path] = []
            # if name already has extension, try directly
            candidates.append(self.images_dir / p)
            candidates.append(self.log_dir / p)
            # without extension, try adding .jpg
            if p.suffix == "":
                candidates.append(self.images_dir / f"{name}.jpg")
                candidates.append(self.images_dir / f"{name}.jpeg")
                candidates.append(self.images_dir / f"{name}.png")
                candidates.append(self.log_dir / f"{name}.jpg")
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    try:
                        img = cv2.imread(str(cand), cv2.IMREAD_COLOR)
                        if img is not None:
                            return img
                    except Exception:
                        continue
            # maybe name is a stem that needs glob matching
            # fallback: try to interpret as integer string
            try:
                idx = int(name)
                return self.load_image(idx)
            except Exception:
                pass
            return None

    def iter_frames(self) -> Iterator[ReplayFrame]:
        if not self.telemetry_path.exists():
            return
        with open(self.telemetry_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                # support both new and legacy schema
                ts = float(row.get("t_mono") or row.get("timestamp") or 0.0)
                vel = None
                if "piloting.speed" in row and isinstance(row["piloting.speed"], dict):
                    d = row["piloting.speed"]
                    vel = (float(d.get("speedX", 0)), float(d.get("speedY", 0)), float(d.get("speedZ", 0)))
                elif "velocity_ned" in row:
                    vel = tuple(row["velocity_ned"])
                att = None
                if "piloting.attitude" in row and isinstance(row["piloting.attitude"], dict):
                    d = row["piloting.attitude"]
                    att = (float(d.get("roll", 0)), float(d.get("pitch", 0)), float(d.get("yaw", 0)))
                # image loading: prefer explicit frame_id / image_path if present, else timestamp index
                image = None
                frame_ref = row.get("frame_id")
                if frame_ref is None:
                    frame_ref = row.get("image_path") or row.get("image") or row.get("image_file")
                if frame_ref is not None:
                    image = self.load_image(frame_ref)
                    if image is None:
                        # fallback to index if explicit ref failed
                        image = self.load_image(idx)
                else:
                    image = self.load_image(idx)
                if image is None:
                    # warning log per spec: iter_frames yields None with warning log when file missing
                    _log.warning("ReplaySystem: missing image for frame %s (index %d) in %s", frame_ref if frame_ref is not None else idx, idx, self.log_dir)
                yield ReplayFrame(timestamp=ts, image=image, velocity_ned=vel, attitude=att, altitude=row.get("altitude"))
