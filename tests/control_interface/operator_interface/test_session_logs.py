from __future__ import annotations

import json

import pytest

import session_logs
from backend_contract import InterfaceMode
from session_logs import SessionLogs, collect_runtime_identity


def test_session_logs_create_required_files_and_dual_timestamps(tmp_path) -> None:
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
    tmp_path,
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


def test_high_rate_logs_batch_fsync_but_safety_events_remain_immediate(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[int] = []
    monkeypatch.setattr(session_logs.os, "fsync", synced.append)
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.SIMULATED_STREAM,
        manifest={},
    )
    startup_syncs = len(synced)

    for index in range(session_logs._BULK_LOG_SYNC_EVERY - 1):
        session.localization("pose_result", seq=index)
    assert len(synced) == startup_syncs

    session.localization("pose_result", seq=session_logs._BULK_LOG_SYNC_EVERY)
    assert len(synced) == startup_syncs + 1

    session.telemetry("state", seq=0)
    session.command("hover")
    session.incident("stream_stale")
    assert len(synced) == startup_syncs + 3

    session.close(reason="test_complete")
    assert len(synced) == startup_syncs + 4


def test_session_logs_do_not_claim_durable_when_fsync_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fsync(_fd: int) -> None:
        raise OSError("fsync unavailable")

    monkeypatch.setattr(session_logs.os, "fsync", fail_fsync)
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.REAL_FLIGHT,
        manifest={},
    )

    assert session.durable is False
    assert session.healthy is False


def test_flight_safety_commands_are_copied_to_permanent_incidents(tmp_path) -> None:
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


def test_inventory_receipts_are_immutable_and_permanent(tmp_path) -> None:
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


def test_inventory_receipt_rejects_unknown_kind(tmp_path) -> None:
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.REAL_FLIGHT,
        manifest={"offline": True},
    )

    with pytest.raises(ValueError, match="inventory kind"):
        session.write_inventory("arbitrary", {})


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
