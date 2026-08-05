"""Offline runtime, durable session logs, retention, and disk-pressure guards."""
from __future__ import annotations

import importlib.metadata
import ipaddress
import json
import os
import platform
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from backend_contract import InterfaceMode


GIB = 1024**3
WARNING_FREE_BYTES = 20 * GIB
WARNING_FREE_PERCENT = 15.0
CRITICAL_FREE_BYTES = 5 * GIB
CRITICAL_FREE_PERCENT = 5.0
RETENTION_MAX_BYTES = 20 * GIB
RETENTION_MAX_AGE_DAYS = 30

_ELIGIBLE_LOG_NAMES = {
    "localization.jsonl",
    "performance.jsonl",
    "video_metrics.jsonl",
}
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
    "worker_exit",
    "worker_stall",
    "logging_failed",
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


@dataclass(frozen=True)
class DiskStatus:
    path: str
    total_bytes: int
    free_bytes: int
    free_percent: float
    warning: bool
    takeoff_blocked: bool
    reason: str


def assess_disk_space(
    path: str | Path,
    *,
    usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
) -> DiskStatus:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    values = usage(target)
    total = int(values.total)
    free = int(values.free)
    percent = (free / total * 100.0) if total > 0 else 0.0
    warning = free < WARNING_FREE_BYTES or percent < WARNING_FREE_PERCENT
    blocked = free < CRITICAL_FREE_BYTES or percent < CRITICAL_FREE_PERCENT
    if blocked:
        reason = (
            f"critical disk space: {free / GIB:.2f} GiB / {percent:.1f}% free"
        )
    elif warning:
        reason = f"low disk space: {free / GIB:.2f} GiB / {percent:.1f}% free"
    else:
        reason = "ok"
    return DiskStatus(
        str(target.resolve()), total, free, percent, warning, blocked, reason
    )


@dataclass(frozen=True)
class RetentionResult:
    bytes_before: int
    bytes_after: int
    removed: tuple[Path, ...]
    skipped_current: int


def _is_retention_eligible(path: Path) -> bool:
    name = path.name
    return name in _ELIGIBLE_LOG_NAMES or name.startswith("loc_metrics_")


def enforce_retention(
    log_root: str | Path,
    *,
    current_session: str | Path | None,
    now: float | None = None,
    max_age_days: int = RETENTION_MAX_AGE_DAYS,
    max_bytes: int = RETENTION_MAX_BYTES,
) -> RetentionResult:
    root = Path(log_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    current = Path(current_session).resolve() if current_session is not None else None
    cutoff = float(time.time() if now is None else now) - max_age_days * 86400
    candidates: list[tuple[float, int, Path]] = []
    skipped_current = 0
    for path in root.rglob("*"):
        if not path.is_file() or not _is_retention_eligible(path):
            continue
        resolved = path.resolve()
        if current is not None and (resolved == current or current in resolved.parents):
            skipped_current += 1
            continue
        stat = path.stat()
        candidates.append((stat.st_mtime, stat.st_size, path))

    bytes_before = sum(size for _mtime, size, _path in candidates)
    removed: list[Path] = []
    remaining = bytes_before
    for mtime, size, path in sorted(candidates):
        if mtime >= cutoff:
            continue
        path.unlink()
        removed.append(path)
        remaining -= size

    if remaining > max_bytes:
        already = set(removed)
        for _mtime, size, path in sorted(candidates):
            if remaining <= max_bytes:
                break
            if path in already or not path.exists():
                continue
            path.unlink()
            removed.append(path)
            remaining -= size

    result = RetentionResult(bytes_before, max(0, remaining), tuple(removed), skipped_current)
    audit = _record(
        "retention",
        {
            "bytes_before": result.bytes_before,
            "bytes_after": result.bytes_after,
            "removed": [str(path) for path in result.removed],
            "skipped_current": result.skipped_current,
            "max_age_days": max_age_days,
            "max_bytes": max_bytes,
        },
    )
    with (root / "retention_audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return result


class _JsonlSink:
    def __init__(self, path: Path):
        self.path = path
        self._handle = path.open("a", encoding="utf-8", buffering=1)
        self.healthy = True
        self.last_error = ""

    def write(self, value: dict[str, Any]) -> bool:
        if not self.healthy:
            return False
        try:
            self._handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            self._handle.flush()
            return True
        except Exception as exc:
            self.healthy = False
            self.last_error = repr(exc)
            print(
                f"[session-log] write failed path={self.path}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return False

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception as exc:
            self.healthy = False
            self.last_error = repr(exc)


class SessionCommandLog:
    """Adapter used by OlympeLiveBackend in place of its legacy single JSONL."""

    def __init__(self, session: "SessionLogs"):
        self._session = session
        self.path = session.directory / "commands.jsonl"
        self.durable = True

    @property
    def healthy(self) -> bool:
        return self._session.healthy

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
        self._sinks = {
            name: _JsonlSink(directory / f"{name}.jsonl")
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
        return True

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
        return ok

    def command(self, event: str, **fields: Any) -> bool:
        return self._write("commands", event, fields)

    def localization(self, event: str, **fields: Any) -> bool:
        return self._write("localization", event, fields)

    def telemetry(self, event: str, **fields: Any) -> bool:
        return self._write("telemetry", event, fields)

    def incident(self, event: str, **fields: Any) -> bool:
        return self._write("incidents", event, fields)

    def close(self, *, reason: str, extra: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        summary = {
            "schema_version": 1,
            "session_id": self.directory.name,
            "mode": self.mode.value,
            "reason": reason,
            "closed_utc": _utc_now(),
            "closed_mono_ns": time.monotonic_ns(),
            "event_counts": dict(self._counts),
            "log_health": {
                name: {
                    "healthy": sink.healthy,
                    "last_error": sink.last_error,
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


def configure_offline_environment() -> None:
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "NO_PROXY": "127.0.0.1,localhost,192.168.42.1,192.168.53.1",
    }.items():
        os.environ[name] = value


def network_destination_allowed(
    mode: InterfaceMode,
    host: str,
    *,
    allowed_real_hosts: Iterable[str],
) -> bool:
    text = str(host or "").strip().strip("[]")
    if text.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if mode is InterfaceMode.SIMULATED_STREAM:
        return False
    allowed = set(str(value).strip().strip("[]") for value in allowed_real_hosts)
    return text in allowed


_NETWORK_GUARD_INSTALLED = False


def install_network_guard(
    mode: InterfaceMode,
    *,
    allowed_real_hosts: Iterable[str] = (),
) -> None:
    """Deny Python-level outbound sockets outside the selected offline policy."""
    global _NETWORK_GUARD_INSTALLED
    if _NETWORK_GUARD_INSTALLED:
        return
    allowed = tuple(allowed_real_hosts)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection

    def address_host(address: Any) -> str | None:
        if isinstance(address, tuple) and address:
            return str(address[0])
        # AF_UNIX path: local IPC is allowed.
        if isinstance(address, (str, bytes)):
            return None
        return ""

    def require(host: str) -> None:
        if not network_destination_allowed(
            mode, host, allowed_real_hosts=allowed
        ):
            raise PermissionError(
                f"offline network policy blocked destination {host!r} in {mode.value}"
            )

    def guarded_connect(sock: socket.socket, address: Any):
        host = address_host(address)
        if host is not None:
            require(host)
        return original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any):
        host = address_host(address)
        if host is not None:
            require(host)
        return original_connect_ex(sock, address)

    def guarded_sendto(sock: socket.socket, data: Any, *args: Any):
        if not args:
            return original_sendto(sock, data)
        address = args[-1]
        host = address_host(address)
        if host is not None:
            require(host)
        return original_sendto(sock, data, *args)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any):
        require(str(host))
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any):
        host = address_host(address)
        if host is not None:
            require(host)
        return original_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = guarded_sendto  # type: ignore[method-assign]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    _NETWORK_GUARD_INSTALLED = True
