from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import disk_policy
from disk_policy import assess_disk_space, enforce_retention


def test_disk_pressure_warns_early_and_blocks_only_at_critical_threshold(
    tmp_path: Path,
) -> None:
    warning = assess_disk_space(
        tmp_path,
        usage=lambda _path: SimpleNamespace(
            total=100 * 1024**3,
            used=86 * 1024**3,
            free=14 * 1024**3,
        ),
    )
    critical = assess_disk_space(
        tmp_path,
        usage=lambda _path: SimpleNamespace(
            total=100 * 1024**3,
            used=96 * 1024**3,
            free=4 * 1024**3,
        ),
    )

    assert warning.warning is True
    assert warning.takeoff_blocked is False
    assert critical.warning is True
    assert critical.takeoff_blocked is True


def test_disk_pressure_blocks_when_percent_is_critical_even_above_five_gib(
    tmp_path: Path,
) -> None:
    status = assess_disk_space(
        tmp_path,
        usage=lambda _path: SimpleNamespace(
            total=200 * 1024**3,
            used=191 * 1024**3,
            free=9 * 1024**3,
        ),
    )

    assert status.free_bytes > disk_policy.CRITICAL_FREE_BYTES
    assert status.free_percent < disk_policy.CRITICAL_FREE_PERCENT
    assert status.takeoff_blocked is True


def test_retention_deletes_only_eligible_old_logs_and_never_current_session(
    tmp_path: Path,
) -> None:
    old = tmp_path / "session_old"
    current = tmp_path / "session_current"
    old.mkdir()
    current.mkdir()
    eligible = old / "localization.jsonl"
    compressed = old / "localization.jsonl.gz"
    telemetry = old / "telemetry.jsonl"
    performance = old / "performance.jsonl"
    video_metrics = old / "video_metrics.jsonl"
    permanent = old / "incidents.jsonl"
    current_eligible = current / "localization.jsonl"
    for path in (
        eligible, compressed, telemetry, performance, video_metrics,
        permanent, current_eligible,
    ):
        path.write_bytes(b"x" * 32)
    old_time = time.time() - 40 * 86400
    os.utime(eligible, (old_time, old_time))
    os.utime(compressed, (old_time, old_time))
    os.utime(telemetry, (old_time, old_time))
    os.utime(performance, (old_time, old_time))
    os.utime(video_metrics, (old_time, old_time))
    os.utime(permanent, (old_time, old_time))
    os.utime(current_eligible, (old_time, old_time))

    result = enforce_retention(
        tmp_path,
        current_session=current,
        now=time.time(),
        max_age_days=30,
        max_bytes=20 * 1024**3,
    )

    assert eligible in result.removed
    assert not eligible.exists()
    assert compressed in result.removed
    assert not compressed.exists()
    assert telemetry in result.removed
    assert not telemetry.exists()
    assert performance in result.removed
    assert not performance.exists()
    assert video_metrics in result.removed
    assert not video_metrics.exists()
    assert permanent.exists()
    assert current_eligible.exists()
    assert (tmp_path / "retention_audit.jsonl").is_file()
