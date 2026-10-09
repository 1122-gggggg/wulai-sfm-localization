"""Bounded same-source pairing and segmentation presentation state, Tk-free.

Delivery is counted only when both successful results have actually been painted.
It is source-rate/UI-rate capped, not the unlimited throughput of either model.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class SourceSubmission:
    seq: int
    name: str
    stamp: float
    starts: dict[str, float] = field(default_factory=dict)
    received: set[str] = field(default_factory=set)
    successful: set[str] = field(default_factory=set)
    rendered: dict[str, float] = field(default_factory=dict)
    counted: bool = False


class PairedPresentationMetrics:
    """Retain frame references only for detector requests; never copy pixels."""

    def __init__(self, *, max_records: int = 256, max_frames: int = 8) -> None:
        self.max_records = max_records
        self.max_frames = max_frames
        self.session = 0
        self.reset()

    def reset(self) -> None:
        self.session += 1
        self.records: OrderedDict[tuple, SourceSubmission] = OrderedDict()
        self.frames: OrderedDict[tuple, Any] = OrderedDict()
        self.display_key: tuple | None = None
        self.display_frame: Any = None
        self.display_result: Any = None
        self.rendered_display_key: tuple | None = None
        self.rendered_mask_count = 0
        self.segmented_render_count = 0
        self.samples: deque[tuple[float, float | None, float]] = deque(maxlen=256)
        self.delivered_count = 0
        self.failures = {"localization": 0, "segmentation": 0}
        self.expired_results = 0

    def submit(self, kind: str, seq: int, name: str, stamp: float,
               start: float, *, frame: Any = None, hold_retry: bool = False) -> None:
        if hold_retry:
            return
        key = (seq, name, stamp)
        record = self.records.get(key)
        if record is None:
            record = SourceSubmission(seq, name, stamp)
            self.records[key] = record
        record.starts.setdefault(kind, start)
        if frame is not None:
            self.frames[key] = frame
            while len(self.frames) > self.max_frames:
                self.frames.popitem(last=False)
        while len(self.records) > self.max_records:
            expired, _ = self.records.popitem(last=False)
            self.frames.pop(expired, None)

    def result_key(self, result: Any) -> tuple | None:
        if result is None or result.get("hold_retry"):
            return None
        if result.get("ui_pair_session") != self.session:
            return None
        seq = result.get("display_seq")
        stamp = result.get("source_frame_stamp_mono")
        if type(seq) is not int:
            return None
        try:
            stamp = float(stamp or 0.0)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(stamp):
            return None
        return (seq, str(result.get("frame_name", "")), stamp)

    def receive(self, kind: str, result: Any) -> bool:
        key = self.result_key(result)
        record = self.records.get(key)
        if record is None or kind not in record.starts:
            if key is not None:
                self.expired_results += 1
            return False
        if kind in record.received:
            return False
        record.received.add(kind)
        if not result.get("success"):
            self.failures[kind] += 1
            return True
        record.successful.add(kind)
        if kind == "segmentation" and result.get("task") == "segment":
            frame = self.frames.pop(key, None)
            if frame is not None and (self.display_key is None or key[0] > self.display_key[0]):
                self.display_key = key
                self.display_frame = frame
                self.display_result = result
        return True

    def rendered(self, kind: str, result: Any, now: float) -> None:
        key = self.result_key(result)
        if kind == "segmentation" and key == self.display_key and key is not None:
            if key != self.rendered_display_key:
                self.segmented_render_count += 1
            self.rendered_display_key = key
            self.rendered_mask_count = sum(
                len(box.get("mask_polygon") or []) >= 3
                for box in result.get("boxes", ())
            )
        record = self.records.get(key)
        if record is None or kind not in record.successful:
            return
        record.rendered.setdefault(kind, now)
        if record.counted or set(record.rendered) != {"localization", "segmentation"}:
            return
        record.counted = True
        complete = max(record.rendered.values())
        source_ms = (complete - record.stamp) * 1000.0 if record.stamp > 0 else None
        compute_ms = (complete - min(record.starts.values())) * 1000.0
        self.samples.append((complete, source_ms, compute_ms))
        self.delivered_count += 1

    def snapshot(self, now: float) -> dict:
        recent = [sample for sample in self.samples if sample[0] >= now - 5.0]
        fps = 0.0
        if len(recent) > 1:
            span = recent[-1][0] - recent[0][0]
            if span > 0:
                fps = (len(recent) - 1) / span
        source = [sample[1] for sample in recent if sample[1] is not None]
        compute = [sample[2] for sample in recent]
        mean_compute = sum(compute) / len(compute) if compute else None
        return {
            "paired_delivery_fps": fps,
            "paired_source_to_render_ms": sum(source) / len(source) if source else None,
            "paired_submit_to_render_ms": mean_compute,
            "compute_e2e_fps": 1000.0 / mean_compute if mean_compute and mean_compute > 0 else None,
            "paired_delivered_count": self.delivered_count,
            "localization_failures": self.failures["localization"],
            "segmentation_failures": self.failures["segmentation"],
            "rendered_mask_count": self.rendered_mask_count,
            "segmented_render_count": self.segmented_render_count,
            "rendered_display_seq": None if self.rendered_display_key is None else self.rendered_display_key[0],
            "rendered_display_frame_name": "" if self.rendered_display_key is None else self.rendered_display_key[1],
            "rendered_display_stamp": None if self.rendered_display_key is None else self.rendered_display_key[2],
            "display_seq": None if self.display_key is None else self.display_key[0],
            "display_frame_name": "" if self.display_key is None else self.display_key[1],
            "display_stamp": None if self.display_key is None else self.display_key[2],
        }

    def hud_lines(self, now: float) -> tuple[str, ...]:
        data = self.snapshot(now)
        source = data["paired_source_to_render_ms"]
        compute = data["paired_submit_to_render_ms"]
        source_text = "-" if source is None else f"{source:.1f}"
        compute_text = "-" if compute is None else f"{compute:.1f}"
        equivalent = data["compute_e2e_fps"]
        equivalent_text = "-" if equivalent is None else f"{equivalent:.1f}"
        result = self.display_result
        masks = 0 if result is None else int(result.get("mask_count", 0) or 0)
        return (
            f"定位+分割 同幀交付 FPS {data['paired_delivery_fps']:.1f} (5s，受來源/UI限速) | 配對 {self.delivered_count}",
            f"雙路 E2E 來源→呈現 {source_text}ms | 最早序列化→雙路呈現 {compute_text}ms",
            f"延遲等效 FPS {equivalent_text} (1000/平均提交→呈現ms，非最大吞吐)",
            f"分割顯示幀 {data['display_seq']} | masks {masks} | 失敗 定位 {self.failures['localization']} / 分割 {self.failures['segmentation']}",
        )


class EndToEndFpsTracker:
    """Rolling rate of frames whose localization AND detection results both arrived.

    A frame is registered only when both workers accepted that same frame, so the
    rate is whole-pipeline completion (bounded by the slower stage), not either
    model's own throughput. Tk-free; holds no pixels.
    """

    KINDS = frozenset({"localization", "detection"})

    def __init__(self, *, window_s: float = 5.0, max_pending: int = 128) -> None:
        self.window_s = float(window_s)
        self.max_pending = int(max_pending)
        self.reset()

    def reset(self) -> None:
        self.pending: OrderedDict[tuple[int, str], set[str]] = OrderedDict()
        self.completed: deque[float] = deque(maxlen=512)
        self.completed_count = 0

    def expect_pair(self, seq: int, name: str) -> None:
        """Register a frame both workers accepted."""
        self.pending[(int(seq), str(name))] = set()
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)

    def receive(self, kind: str, result: Any, now: float) -> bool:
        """Record one worker result; True when it completed a registered pair."""
        if kind not in self.KINDS or result is None or result.get("hold_retry"):
            return False
        seq = result.get("display_seq")
        if type(seq) is not int:
            return False
        seen = self.pending.get((seq, str(result.get("frame_name", ""))))
        if seen is None or kind in seen:
            return False
        seen.add(kind)
        if seen != self.KINDS:
            return False
        del self.pending[(seq, str(result.get("frame_name", "")))]
        self.completed.append(float(now))
        self.completed_count += 1
        return True

    def fps(self, now: float) -> float:
        cutoff = float(now) - self.window_s
        recent = [stamp for stamp in self.completed if stamp >= cutoff]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0.0 else 0.0
