"""Disk-space fail-closed thresholds and old-log retention.

Split out of the former runtime_safety.py (D2). See session_logs.py for the
JSONL session logs this module prunes, network_policy.py for the offline
network guard, and arming_gate.py for the autonomy arming gate.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from session_logs import _record


GIB = 1024**3
WARNING_FREE_BYTES = 20 * GIB
WARNING_FREE_PERCENT = 15.0
CRITICAL_FREE_BYTES = 5 * GIB
CRITICAL_FREE_PERCENT = 5.0
RETENTION_MAX_BYTES = 20 * GIB
RETENTION_MAX_AGE_DAYS = 30

_ELIGIBLE_LOG_NAMES = {
    "localization.jsonl",
    "telemetry.jsonl",
    "performance.jsonl",
    "video_metrics.jsonl",
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
    name = path.name.removesuffix(".gz")
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


__all__ = [
    "CRITICAL_FREE_BYTES",
    "CRITICAL_FREE_PERCENT",
    "DiskStatus",
    "GIB",
    "RETENTION_MAX_AGE_DAYS",
    "RETENTION_MAX_BYTES",
    "RetentionResult",
    "WARNING_FREE_BYTES",
    "WARNING_FREE_PERCENT",
    "assess_disk_space",
    "enforce_retention",
]
