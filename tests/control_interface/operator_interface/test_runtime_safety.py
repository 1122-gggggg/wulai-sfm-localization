from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import runtime_safety
from backend_contract import InterfaceMode
from runtime_safety import (
    SessionLogs,
    assess_disk_space,
    collect_runtime_identity,
    enforce_retention,
    install_network_guard,
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
    telemetry = old / "telemetry.jsonl"
    performance = old / "performance.jsonl"
    video_metrics = old / "video_metrics.jsonl"
    permanent = old / "incidents.jsonl"
    current_eligible = current / "localization.jsonl"
    for path in (
        eligible, telemetry, performance, video_metrics,
        permanent, current_eligible,
    ):
        path.write_bytes(b"x" * 32)
    old_time = time.time() - 40 * 86400
    os.utime(eligible, (old_time, old_time))
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
    assert telemetry in result.removed
    assert not telemetry.exists()
    assert performance in result.removed
    assert not performance.exists()
    assert video_metrics in result.removed
    assert not video_metrics.exists()
    assert permanent.exists()
    assert current_eligible.exists()
    assert (tmp_path / "retention_audit.jsonl").is_file()


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

    assert status.free_bytes > runtime_safety.CRITICAL_FREE_BYTES
    assert status.free_percent < runtime_safety.CRITICAL_FREE_PERCENT
    assert status.takeoff_blocked is True


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


def test_session_summary_reports_latency_extrema_and_unresolved_incidents(
    tmp_path: Path,
) -> None:
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.SIMULATED_STREAM,
        manifest={},
    )
    session.localization("pose_result", e2e_submit_to_ui_ms=10.0)
    session.localization("pose_result", e2e_submit_to_ui_ms=30.0)
    session.localization("pose_result", e2e_submit_to_ui_ms=20.0)
    session.incident("worker_exit", resolved=False)
    session.close(reason="test_complete")

    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["metrics"]["p95_ms"] == pytest.approx(30.0)
    assert summary["metrics"]["max_ms"] == pytest.approx(30.0)
    assert summary["unresolved"]["count"] == 1
    assert summary["unresolved"]["events"] == ["worker_exit"]


def test_session_logs_do_not_claim_durable_when_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fsync(_fd: int) -> None:
        raise OSError("fsync unavailable")

    monkeypatch.setattr(runtime_safety.os, "fsync", fail_fsync)
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.REAL_FLIGHT,
        manifest={},
    )

    assert session.durable is False
    assert session.healthy is False


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


def test_network_guard_blocks_disallowed_destinations_without_real_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def connect(self, address: object) -> tuple[str, object]:
            return "connect", address

        def connect_ex(self, address: object) -> tuple[str, object]:
            return "connect_ex", address

        def sendto(self, data: object, *args: object) -> tuple[str, object, tuple[object, ...]]:
            return "sendto", data, args

    fake_socket = SimpleNamespace(
        socket=FakeSocket,
        getaddrinfo=lambda host, *args, **kwargs: (host, args, kwargs),
        create_connection=lambda address, *args, **kwargs: (
            address,
            args,
            kwargs,
        ),
    )
    monkeypatch.setattr(runtime_safety, "socket", fake_socket)
    monkeypatch.setattr(runtime_safety, "_NETWORK_GUARD_INSTALLED", False)

    install_network_guard(
        InterfaceMode.REAL_FLIGHT,
        allowed_real_hosts=("192.168.53.1",),
    )
    sock = FakeSocket()
    with pytest.raises(
        PermissionError,
        match="offline network policy blocked destination '192.168.53.2'",
    ):
        sock.connect(("192.168.53.2", 443))
    assert sock.connect(("192.168.53.1", 443))[0] == "connect"
    assert sock.connect_ex(("127.0.0.1", 443))[0] == "connect_ex"
    assert sock.sendto(b"x", ("127.0.0.1", 443))[0] == "sendto"
    assert sock.sendto(b"x")[0] == "sendto"
    assert sock.connect("/tmp/local.sock")[0] == "connect"
    with pytest.raises(PermissionError):
        fake_socket.getaddrinfo("example.com")
    with pytest.raises(PermissionError):
        fake_socket.create_connection(("8.8.8.8", 443))

    guarded_connect = FakeSocket.connect
    install_network_guard(InterfaceMode.SIMULATED_STREAM)
    assert FakeSocket.connect is guarded_connect


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
