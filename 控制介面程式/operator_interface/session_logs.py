"""Durable per-session JSONL logs, hardware/video inventory receipts, and the
offline runtime identity recorded into the session manifest.

Split out of the former runtime_safety.py (D2): this file owns session/log
lifecycle only. Disk-pressure/retention policy lives in disk_policy.py,
offline-network enforcement in network_policy.py, and the autonomy arming
gate in arming_gate.py -- each independently readable and testable.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend_contract import InterfaceMode


_BULK_LOG_SYNC_EVERY = 16

_PERMANENT_LOG_NAMES = {
    "commands.jsonl",
    "hardware_inventory.json",
    "incidents.jsonl",
    "session_manifest.json",
    "session_summary.json",
    "video_inventory.json",
}
_INCIDENT_EVENTS = {
    "controller_disconnect",
    "fail_safe",
    "stream_lost",
    "link_lost",
    "lost_link_policy",
    "rth",
    "runtime_safety",
    "runtime_safety_latched",
    "runtime_safety_exception",
    "runtime_safety_rearmed",
    "distance_guard",
    "distance_guard_unavailable",
    "worker_exit",
    "worker_stall",
    "backend_poll_failed",
    "diagnostic_write_failed",
    "legacy_takeoff_rejected",
    "logging_failed",
    "tracker_state_transition",
    "disk_critical",
    "cleanup_begin",
    "cleanup_done",
    "land_cmd",
}

_RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "opencv-python",
    "parrot-olympe",
    "Pillow",
    "pycolmap",
    "torch",
    "torchvision",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_runtime_identity() -> dict[str, Any]:
    """Collect a JSON-safe, offline runtime identity for the session manifest."""
    packages: dict[str, str | None] = {}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None

    cuda: dict[str, Any] = {
        "available": False,
        "runtime_version": None,
        "device_names": [],
    }
    try:
        import torch

        cuda["runtime_version"] = getattr(torch.version, "cuda", None)
        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            cuda["device_names"] = [
                str(torch.cuda.get_device_name(index))
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        cuda["error"] = repr(exc)

    driver_path = Path("/proc/driver/nvidia/version")
    nvidia_driver: dict[str, Any] = {
        "path": str(driver_path),
        "version_line": None,
    }
    try:
        lines = driver_path.read_text(encoding="utf-8", errors="replace").splitlines()
        nvidia_driver["version_line"] = lines[0] if lines else None
    except OSError as exc:
        nvidia_driver["error"] = repr(exc)

    return {
        "python": {
            "version": platform.python_version(),
            "build": sys.version,
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "nvidia_driver": nvidia_driver,
    }


def _record(event: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "t_utc": _utc_now(),
        "t_mono_ns": time.monotonic_ns(),
        "event": event,
        **fields,
    }


class _JsonlSink:
    def __init__(self, path: Path, *, sync_every: int = 1):
        self.path = path
        self._handle = path.open("a", encoding="utf-8", buffering=1)
        self._sync_every = max(1, int(sync_every))
        self._pending_writes = 0
        self.healthy = True
        self.last_error = ""
        self.durable = False
        try:
            # A durable property is only truthful after the kernel accepted an
            # fsync on this exact sink.  The probe also makes a fresh session
            # fail closed before its first safety command is issued.
            self._sync()
        except Exception as exc:
            self.healthy = False
            self.last_error = repr(exc)
            print(
                f"[session-log] durability probe failed path={self.path}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending_writes = 0
        self.durable = True

    def write(self, value: dict[str, Any]) -> bool:
        if not self.healthy:
            return False
        try:
            self._handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            self._pending_writes += 1
            if self._pending_writes >= self._sync_every:
                self._sync()
            return True
        except Exception as exc:
            self.healthy = False
            self.durable = False
            self.last_error = repr(exc)
            print(
                f"[session-log] write failed path={self.path}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return False

    def close(self) -> None:
        try:
            if self.healthy and self._pending_writes:
                self._sync()
            self._handle.close()
        except Exception as exc:
            self.healthy = False
            self.durable = False
            self.last_error = repr(exc)


class SessionCommandLog:
    """Adapter used by OlympeLiveBackend in place of its legacy single JSONL."""

    def __init__(self, session: "SessionLogs"):
        self._session = session
        self.path = session.directory / "commands.jsonl"

    @property
    def healthy(self) -> bool:
        return self._session.healthy

    @property
    def durable(self) -> bool:
        return self._session.durable

    def event(self, event: str, **fields: Any) -> None:
        self._session.command(event, **fields)
        if event in _INCIDENT_EVENTS:
            self._session.incident(event, **fields)

    def close(self) -> None:
        # Session summary is finalized by the owning UI/process after cleanup.
        return None


class SessionLogs:
    REQUIRED_STREAMS = ("commands", "localization", "telemetry", "incidents")

    def __init__(self, directory: Path, mode: InterfaceMode):
        self.directory = directory
        self.mode = mode
        self._closed = False
        self._inventory_error: str | None = None
        self._counts = {name: 0 for name in self.REQUIRED_STREAMS}
        self._latency_samples_ms: list[float] = []
        self._unresolved_incidents: set[str] = set()
        self._sinks = {
            name: _JsonlSink(
                directory / f"{name}.jsonl",
                sync_every=(
                    1 if name in {"commands", "incidents"} else _BULK_LOG_SYNC_EVERY
                ),
            )
            for name in self.REQUIRED_STREAMS
        }
        self.command_log = SessionCommandLog(self)

    @classmethod
    def create(
        cls,
        log_root: str | Path,
        *,
        mode: InterfaceMode,
        manifest: dict[str, Any],
    ) -> "SessionLogs":
        root = Path(log_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = root / f"session_{stamp}_{mode.value}_{uuid.uuid4().hex[:8]}"
        directory.mkdir(mode=0o750)
        payload = {
            "schema_version": 1,
            "session_id": directory.name,
            "mode": mode.value,
            "created_utc": _utc_now(),
            "created_mono_ns": time.monotonic_ns(),
            **manifest,
        }
        temp = directory / ".session_manifest.json.tmp"
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, directory / "session_manifest.json")
        return cls(directory, mode)

    @property
    def healthy(self) -> bool:
        return (
            not self._closed
            and self._inventory_error is None
            and all(sink.healthy for sink in self._sinks.values())
        )

    @property
    def durable(self) -> bool:
        return (
            self._inventory_error is None
            and all(sink.durable for sink in self._sinks.values())
        )

    def write_inventory(self, kind: str, inventory: dict[str, Any]) -> bool:
        """Write one immutable hardware or video inventory receipt."""
        if kind not in {"hardware", "video"}:
            raise ValueError("inventory kind must be 'hardware' or 'video'")
        path = self.directory / f"{kind}_inventory.json"
        if path.exists():
            return True
        payload = {
            "schema_version": 1,
            "session_id": self.directory.name,
            "kind": kind,
            "captured_utc": _utc_now(),
            "captured_mono_ns": time.monotonic_ns(),
            "inventory": dict(inventory),
        }
        temp = self.directory / f".{kind}_inventory.json.tmp"
        try:
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temp, path)
        except Exception as exc:
            self._inventory_error = repr(exc)
            return False
        return True

    def _write(self, stream: str, event: str, fields: dict[str, Any]) -> bool:
        if self._closed:
            return False
        ok = self._sinks[stream].write(_record(event, fields))
        if ok:
            self._counts[stream] += 1
            for key in (
                "e2e_submit_to_ui_ms",
                "e2e_ms",
                "wall_ms",
                "core_wall_ms",
            ):
                value = fields.get(key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0.0:
                    self._latency_samples_ms.append(value)
                    break
        return ok

    def command(self, event: str, **fields: Any) -> bool:
        return self._write("commands", event, fields)

    def localization(self, event: str, **fields: Any) -> bool:
        return self._write("localization", event, fields)

    def telemetry(self, event: str, **fields: Any) -> bool:
        return self._write("telemetry", event, fields)

    def incident(self, event: str, **fields: Any) -> bool:
        if fields.get("resolved") is True:
            self._unresolved_incidents.discard(str(event))
        else:
            self._unresolved_incidents.add(str(event))
        return self._write("incidents", event, fields)

    def close(self, *, reason: str, extra: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        samples = sorted(self._latency_samples_ms)
        p95 = None
        maximum = None
        if samples:
            p95 = samples[min(len(samples) - 1, max(0, math.ceil(0.95 * len(samples)) - 1))]
            maximum = samples[-1]
        summary = {
            "schema_version": 1,
            "session_id": self.directory.name,
            "mode": self.mode.value,
            "reason": reason,
            "closed_utc": _utc_now(),
            "closed_mono_ns": time.monotonic_ns(),
            "event_counts": dict(self._counts),
            "metrics": {
                "p95_ms": p95,
                "max_ms": maximum,
                "sample_count": len(samples),
            },
            "p95_ms": p95,
            "max_ms": maximum,
            "unresolved": {
                "count": len(self._unresolved_incidents),
                "events": sorted(self._unresolved_incidents),
            },
            "log_health": {
                name: {
                    "healthy": sink.healthy,
                    "last_error": sink.last_error,
                    "durable": sink.durable,
                }
                for name, sink in self._sinks.items()
            },
            **(extra or {}),
        }
        temp = self.directory / ".session_summary.json.tmp"
        temp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, self.directory / "session_summary.json")
        for sink in self._sinks.values():
            sink.close()
        self._closed = True


__all__ = [
    "SessionCommandLog",
    "SessionLogs",
    "collect_runtime_identity",
]
