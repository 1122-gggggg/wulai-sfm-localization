from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend_contract import InterfaceMode
from runtime_safety import (
    SessionLogs,
    assess_disk_space,
    collect_runtime_identity,
    enforce_retention,
    network_destination_allowed,
)


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


def test_retention_deletes_only_eligible_old_logs_and_never_current_session(
    tmp_path: Path,
) -> None:
    old = tmp_path / "session_old"
    current = tmp_path / "session_current"
    old.mkdir()
    current.mkdir()
    eligible = old / "localization.jsonl"
    permanent = old / "incidents.jsonl"
    current_eligible = current / "localization.jsonl"
    for path in (eligible, permanent, current_eligible):
        path.write_bytes(b"x" * 32)
    old_time = time.time() - 40 * 86400
    os.utime(eligible, (old_time, old_time))
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
    assert permanent.exists()
    assert current_eligible.exists()
    assert (tmp_path / "retention_audit.jsonl").is_file()


def test_session_logs_create_required_files_and_dual_timestamps(tmp_path: Path) -> None:
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.SIMULATED_STREAM,
        manifest={"site_id": "river_site_edm", "offline": True},
    )

    session.command("hover", request_id="r1")
    session.incident("stream_stale", detail="test")
    session.close(reason="test_complete")

    manifest = json.loads((session.directory / "session_manifest.json").read_text())
    command = json.loads(
        (session.directory / "commands.jsonl").read_text().splitlines()[0]
    )
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert manifest["mode"] == "simulated-stream"
    assert command["event"] == "hover"
    assert command["t_utc"]
    assert command["t_mono_ns"] > 0
    assert (session.directory / "localization.jsonl").is_file()
    assert (session.directory / "telemetry.jsonl").is_file()
    assert (session.directory / "incidents.jsonl").is_file()
    assert summary["reason"] == "test_complete"


def test_flight_safety_commands_are_copied_to_permanent_incidents(
    tmp_path: Path,
) -> None:
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.REAL_FLIGHT,
        manifest={"offline": True},
    )

    session.command_log.event(
        "runtime_safety_latched", reason="battery_critical"
    )
    session.command_log.event("rth", ok=True, reason="battery_critical")
    session.close(reason="test_complete")

    incidents = [
        json.loads(line)
        for line in (session.directory / "incidents.jsonl").read_text().splitlines()
    ]
    assert [item["event"] for item in incidents] == [
        "runtime_safety_latched",
        "rth",
    ]


def test_inventory_receipts_are_immutable_and_permanent(tmp_path: Path) -> None:
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.REAL_FLIGHT,
        manifest={"offline": True},
    )

    assert session.write_inventory("hardware", {"serial": "AIRCRAFT-1"})
    assert session.write_inventory("hardware", {"serial": "MUST-NOT-OVERWRITE"})
    assert session.write_inventory("video", {"codec": "H.264"})

    hardware = json.loads(
        (session.directory / "hardware_inventory.json").read_text()
    )
    video = json.loads((session.directory / "video_inventory.json").read_text())
    assert hardware["inventory"]["serial"] == "AIRCRAFT-1"
    assert video["inventory"]["codec"] == "H.264"
    assert session.healthy


def test_inventory_receipt_rejects_unknown_kind(tmp_path: Path) -> None:
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.REAL_FLIGHT,
        manifest={"offline": True},
    )

    with pytest.raises(ValueError, match="inventory kind"):
        session.write_inventory("arbitrary", {})


def test_network_policy_allows_only_loopback_for_sim_and_anafi_for_real() -> None:
    assert network_destination_allowed(
        InterfaceMode.SIMULATED_STREAM, "127.0.0.1", allowed_real_hosts=()
    )
    assert not network_destination_allowed(
        InterfaceMode.SIMULATED_STREAM, "8.8.8.8", allowed_real_hosts=()
    )
    assert network_destination_allowed(
        InterfaceMode.REAL_FLIGHT,
        "192.168.53.1",
        allowed_real_hosts=("192.168.53.1",),
    )
    assert not network_destination_allowed(
        InterfaceMode.REAL_FLIGHT,
        "example.com",
        allowed_real_hosts=("192.168.53.1",),
    )


def test_runtime_identity_records_packages_cuda_gpu_and_driver_fields() -> None:
    identity = collect_runtime_identity()

    assert identity["python"]["version"]
    assert identity["python"]["executable"]
    assert "numpy" in identity["packages"]
    assert "torch" in identity["packages"]
    assert set(identity["cuda"]) >= {
        "available",
        "runtime_version",
        "device_names",
    }
    assert "nvidia_driver" in identity
