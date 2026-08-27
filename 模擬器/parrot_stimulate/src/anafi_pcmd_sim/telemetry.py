"""Capture and parse Sphinx simulator true telemetry."""

from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .models import TruePosition

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TRUE_DRONE_SECTION = "omniscient_anafi"
_TIMESTAMP_PATTERN = re.compile(rf"^{_TRUE_DRONE_SECTION}\.timestamp:\s*(?P<value>{_NUMBER})\s*$")
_POSITION_PATTERN = re.compile(
    rf"^{_TRUE_DRONE_SECTION}\.worldPosition\.(?P<axis>[xyz]):"
    rf"\s*(?P<value>{_NUMBER})\s*$"
)
_MAX_ABSOLUTE_WORLD_COORDINATE_M = 10_000.0


class TrueTelemetryParser:
    """Parse ``tlm-data-logger`` output into complete true-position samples."""

    def __init__(self) -> None:
        self._timestamp_s: float | None = None
        self._coordinates: dict[str, float] = {}

    def feed(self, line: str) -> TruePosition | None:
        timestamp_match = _TIMESTAMP_PATTERN.search(line)
        if timestamp_match:
            self._timestamp_s = float(timestamp_match.group("value"))
            return None

        position_match = _POSITION_PATTERN.search(line)
        if not position_match:
            return None

        value = float(position_match.group("value"))
        if not math.isfinite(value) or abs(value) > _MAX_ABSOLUTE_WORLD_COORDINATE_M:
            self._timestamp_s = None
            self._coordinates = {}
            return None

        self._coordinates[position_match.group("axis")] = value
        if self._timestamp_s is None or set(self._coordinates) != {"x", "y", "z"}:
            return None

        sample = TruePosition(
            timestamp_s=self._timestamp_s,
            x_m=self._coordinates["x"],
            y_m=self._coordinates["y"],
            z_m=self._coordinates["z"],
        )
        self._coordinates = {}
        return sample


class TrueTelemetryCollector:
    """Run Sphinx's true-data logger and retain parsed world-position samples."""

    def __init__(
        self,
        command: Sequence[str] = (
            "tlm-data-logger",
            "-r",
            "50",
            "inet:127.0.0.1:9060",
        ),
        *,
        stderr_path: Path | None = None,
    ) -> None:
        self._command = tuple(command)
        self._stderr_path = stderr_path
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._parser = TrueTelemetryParser()
        self._samples: list[TruePosition] = []
        self._condition = threading.Condition()
        self._stderr_handle: TextIO | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("true telemetry collector is already running")

        stderr: int | object = subprocess.DEVNULL
        if self._stderr_path is not None:
            self._stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_handle = self._stderr_path.open("w", encoding="utf-8")
            stderr = self._stderr_handle

        self._process = subprocess.Popen(
            self._command,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_stdout, name="sphinx-true-telemetry")
        self._thread.start()

    def _read_stdout(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            sample = self._parser.feed(line)
            if sample is None:
                continue
            with self._condition:
                self._samples.append(sample)
                self._condition.notify_all()

    def wait_for_sample(self, timeout_s: float) -> TruePosition:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._samples:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no Sphinx true world-position sample received")
                self._condition.wait(timeout=remaining)
            return self._samples[-1]

    def latest_sample(self) -> TruePosition:
        with self._condition:
            if not self._samples:
                raise RuntimeError("no Sphinx true world-position sample received")
            return self._samples[-1]

    def sample_at_or_before(self, timestamp_s: float) -> TruePosition | None:
        """Return the newest sample no later than the requested simulator timestamp."""
        with self._condition:
            return next(
                (sample for sample in reversed(self._samples) if sample.timestamp_s <= timestamp_s),
                None,
            )

    def interpolated_sample_at(self, timestamp_s: float) -> TruePosition | None:
        """Interpolate the true pose at a past timestamp without extrapolating."""
        with self._condition:
            before: TruePosition | None = None
            for after in self._samples:
                if after.timestamp_s < timestamp_s:
                    before = after
                    continue
                if after.timestamp_s == timestamp_s:
                    return after
                if before is None:
                    return None
                span_s = after.timestamp_s - before.timestamp_s
                if span_s <= 0:
                    return None
                fraction = (timestamp_s - before.timestamp_s) / span_s
                return TruePosition(
                    timestamp_s=timestamp_s,
                    x_m=before.x_m + fraction * (after.x_m - before.x_m),
                    y_m=before.y_m + fraction * (after.y_m - before.y_m),
                    z_m=before.z_m + fraction * (after.z_m - before.z_m),
                )
            return None

    def samples(self) -> tuple[TruePosition, ...]:
        with self._condition:
            return tuple(self._samples)

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._stderr_handle is not None:
            self._stderr_handle.close()
        self._process = None
        self._thread = None
