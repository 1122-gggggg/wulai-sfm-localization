"""Developer-only sampler that explains slow frames (SFM_STALL_SAMPLER=1).

While a frame step runs longer than ``THRESHOLD_S`` it records, every ``PERIOD_S``,
the Python stacks of all threads and the CPU time every OS thread of the process
spent since the step began. Nothing is started unless the environment asks for it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

THRESHOLD_S = 0.04
PERIOD_S = 0.02
_TICK = os.sysconf("SC_CLK_TCK")


def _thread_cpu() -> dict[int, tuple[str, float]]:
    out = {}
    for entry in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{entry}/stat") as handle:
                stat = handle.read()
            name = stat[stat.index("(") + 1 : stat.rindex(")")]
            fields = stat[stat.rindex(")") + 2 :].split()
            out[int(entry)] = (name, (int(fields[11]) + int(fields[12])) / _TICK)
        except (OSError, ValueError, IndexError):
            continue
    return out


class StallSampler:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._started: float | None = None
        self._ordinal = -1
        self._baseline: dict[int, tuple[str, float]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stall-sampler", daemon=True)
        self._thread.start()

    def begin(self, ordinal: int) -> None:
        self._baseline = _thread_cpu()
        self._ordinal = ordinal
        self._started = time.perf_counter()

    def end(self) -> None:
        self._started = None

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        names = {}
        last = (-1, 0.0)
        while not self._stop.wait(0.005):
            started, ordinal = self._started, self._ordinal
            if started is None:
                continue
            elapsed = time.perf_counter() - started
            if elapsed < THRESHOLD_S or (last[0] == ordinal and elapsed - last[1] < PERIOD_S):
                continue
            last = (ordinal, elapsed)
            for thread in threading.enumerate():
                names[thread.native_id] = thread.name
            frames = sys._current_frames()
            by_ident = {t.ident: t for t in threading.enumerate()}
            stacks = {}
            for ident, frame in frames.items():
                thread = by_ident.get(ident)
                label = thread.name if thread else str(ident)
                stacks[label] = [
                    f"{Path(item.filename).name}:{item.lineno}:{item.name}"
                    for item in traceback.extract_stack(frame)[-6:]
                ]
            now = _thread_cpu()
            cpu = {}
            for tid, (comm, ticks) in now.items():
                before = self._baseline.get(tid)
                delta = ticks - (before[1] if before else 0.0)
                if delta > 0:
                    cpu[f"{names.get(tid, comm)}#{tid}"] = round(delta, 3)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"ordinal": ordinal, "elapsed_ms": round(elapsed * 1000, 1),
                         "cpu_s_by_thread": cpu, "stacks": stacks},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
