"""Durable command-event logging; no aircraft operations."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def quiet_olympe_logs() -> None:
    """Cut Olympe/pdraw INFO spam (AVCC unsupported, renderer empty queue, etc.).

    Does not change flight commands — log level only. Call before connect.
    """
    import logging

    # Force WARNING even if handlers already exist (Olympe reconfigures often).
    logging.basicConfig(level=logging.WARNING)
    # Olympe creates per-device child loggers after connection and may attach
    # their own INFO handlers. The global threshold keeps verbose state payloads
    # out of the operator terminal regardless of later logger reconfiguration.
    logging.disable(logging.INFO)
    for name in (
        "olympe",
        "ulog",
        "olympe.pdraw",
        "olympe.drone",
        "olympe.video",
        "olympe.video.renderer",
        "olympe.backend",
        "olympe.media",
        "olympe.module_loader",
        "olympe.scheduler",
        "olympe.update",
        "olympe.flightplan",
        "olympe.missions",
        "olympe.arsdkng",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.propagate = True


class _CmdLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._f = None
        self.durable = False
        self.healthy = True
        self.last_error = ""
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._f = path.open("w", encoding="utf-8", buffering=1)
                self.event("start", sink="olympe_live_ui")
            except Exception as exc:
                self.healthy = False
                self.last_error = repr(exc)
                self._f = None
                print(
                    f"[live-ui] safety log unavailable path={path}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

    def event(self, event: str, **kw: Any) -> None:
        mono_ns = time.monotonic_ns()
        rec = {
            "t_iso": datetime.now(timezone.utc).isoformat(),
            "t_mono": mono_ns * 1e-9,
            "t_mono_ns": mono_ns,
            "event": event,
            **kw,
        }
        try:
            line = json.dumps(rec, ensure_ascii=False)
        except Exception as exc:
            self.healthy = False
            self.last_error = repr(exc)
            print(
                f"[live-ui] safety event serialization failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return
        print(f"[live-ui] {line}", flush=True)
        if self._f is not None:
            try:
                self._f.write(line + "\n")
                self._f.flush()
                os.fsync(self._f.fileno())
                self.durable = True
            except Exception as exc:
                self.durable = False
                self.healthy = False
                self.last_error = repr(exc)
                print(
                    f"[live-ui] safety log write failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

    def close(self) -> None:
        if self._f is not None:
            try:
                self.event("end")
            except Exception:  # Tier3: teardown best-effort — event flush optional
                pass
            try:
                self._f.close()
            except Exception:  # Tier3: teardown best-effort — file close optional
                pass
            self._f = None
