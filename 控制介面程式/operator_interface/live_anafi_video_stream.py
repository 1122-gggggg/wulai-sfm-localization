"""Latest-frame adapter for the live Anafi grabber."""
from __future__ import annotations

import math
import time
from typing import Any


class LiveAnafiVideoStream:
    """Latest-frame stream for OperatorApp.

    Returns a **new** frame only when the grabber stamp advances (no duplicate
    work on every UI tick). Keeps numpy RGB for the UI to convert once at
    panel size — cheaper than full-res PIL every tick.
    """

    def __init__(self, grabber):
        self._grab = grabber
        self.output_index = 0
        self.last_frame_name = "live_0"
        self.fps = 0.0
        self._last_stamp: float | None = None
        self.last_stamp: float = 0.0
        self.last_timing: dict = {}
        self._last_frame_width_px: int | None = None
        self._last_frame_height_px: int | None = None

    @staticmethod
    def _valid_stamp(value: Any) -> float | None:
        try:
            stamp = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return (
            stamp
            if math.isfinite(stamp) and stamp > 0.0 and stamp <= time.monotonic()
            else None
        )

    def _has_new_frame_stamp(self) -> bool:
        peek_stamp = getattr(self._grab, "peek_stamp", None)
        if not callable(peek_stamp):
            return True
        try:
            raw_stamp_hint = peek_stamp()
        except Exception:
            raw_stamp_hint = None
        if raw_stamp_hint is not None:
            stamp_hint = self._valid_stamp(raw_stamp_hint)
            if stamp_hint is None:
                return False
            if self._last_stamp is not None and stamp_hint == self._last_stamp:
                return False
        elif self._last_stamp is not None:
            return False
        return True

    def _read_frame_sample(self):
        try:
            sample_with_timing = getattr(self._grab, "latest_frame_with_timing", None)
            sample = (sample_with_timing()
                      if callable(sample_with_timing) else self._grab())
        except Exception:
            return None
        if sample is None:
            return None
        timing = {}
        if isinstance(sample, tuple):
            frame = sample[0]
            stamp = self._valid_stamp(sample[1])
            if len(sample) >= 3 and isinstance(sample[2], dict):
                timing = dict(sample[2])
        else:
            frame, stamp = sample, time.monotonic()
        return frame, stamp, timing

    def _publish_frame(self, frame, stamp, timing):
        self._last_stamp = stamp
        self.last_stamp = stamp
        self.last_timing = timing
        # Leave as numpy HxWx3 RGB when possible (UI converts at panel size).
        try:
            import numpy as np
            if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        except Exception:
            pass
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            self._last_frame_height_px = int(shape[0])
            self._last_frame_width_px = int(shape[1])
        else:
            size = getattr(frame, "size", None)
            if isinstance(size, tuple) and len(size) == 2:
                self._last_frame_width_px = int(size[0])
                self._last_frame_height_px = int(size[1])
        self.output_index += 1
        self.last_frame_name = f"live_{self.output_index}"
        self.fps = float(getattr(self._grab, "fps", 0.0) or 0.0)
        return frame

    def next_frame(self, *, only_new: bool = True):
        if only_new and not self._has_new_frame_stamp():
            return None
        sample = self._read_frame_sample()
        if sample is None:
            return None
        frame, stamp, timing = sample
        if frame is None or stamp is None:
            return None
        if only_new and self._last_stamp is not None and stamp == self._last_stamp:
            return None
        return self._publish_frame(frame, stamp, timing)

    def metadata_snapshot(self) -> dict[str, Any]:
        raw = getattr(self._grab, "stream_metadata", {})
        try:
            raw = raw() if callable(raw) else raw
            metadata = dict(raw) if isinstance(raw, dict) else {}
        except Exception:
            metadata = {}
        metadata.update(
            {
                "ui_frame_width_px": self._last_frame_width_px,
                "ui_frame_height_px": self._last_frame_height_px,
                "observed_fps": float(self.fps),
                "frames_delivered": int(self.output_index),
            }
        )
        return metadata

    def close(self) -> None:
        try:
            self._grab.stop()
        except Exception:
            pass
