from __future__ import annotations

import json
import threading
from contextlib import contextmanager


import pytest

import session_logs
from backend_contract import InterfaceMode
from session_logs import SessionLogs, collect_runtime_identity


@pytest.fixture(autouse=True)
def close_session_writers(monkeypatch):
    """Finalize persistence workers even when a test assertion fails."""
    sessions = []
    original = SessionLogs.__init__

    def tracked_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(SessionLogs, "__init__", tracked_init)
    yield
    for session in sessions:
        if not session._closed:
            session.close(reason="test_cleanup")


@contextmanager
def blocked_diagnostic_disk(session, monkeypatch, *, stream="localization", method="write"):
    entered = threading.Event()
    release = threading.Event()
    sink = session._sinks[stream]
    original = getattr(sink, method)
    thread_ids = []

    def blocked(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        entered.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(sink, method, blocked)
    try:
        yield entered, release, thread_ids
    finally:
        release.set()




def test_trajectory_archive_preserves_route_pose_and_lifecycle_after_retention(tmp_path):
    from disk_policy import enforce_retention

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    session.localization("auto_route_plan", auto_run_id="a", waypoints_u=[[0, 0, 0], [1, 2, 3]])
    session.localization("pose_result", pose={"x": 1, "y": 2, "z": 3}, direct_status="IMU_BRIDGE", wall_ms=10)
    session.localization("auto_route_tick", auto_run_id="a", pose_u=[1, 2, 3], target_index=1)
    session.telemetry("autonomy_event", auto_run_id="a", kind="finished")
    session.close(reason="test")
    archived = list(map(json.loads, (session.directory / "trajectory.jsonl").read_text().splitlines()))
    assert [r["event"] for r in archived] == ["auto_route_plan", "pose_result", "auto_route_tick", "autonomy_event"]
    assert archived[1]["direct_status"] == "IMU_BRIDGE"
    assert "wall_ms" not in archived[1]
    original = json.loads((session.directory / "localization.jsonl").read_text().splitlines()[1])
    assert archived[1]["t_mono_ns"] == original["t_mono_ns"]
    enforce_retention(tmp_path, current_session=None, max_bytes=0)
    assert (session.directory / "trajectory.jsonl").exists()
    assert not (session.directory / "localization.jsonl").exists()


def test_trajectory_write_failure_is_visible_in_log_health(tmp_path, monkeypatch):
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    session._sinks["trajectory"].healthy = False
    assert session.localization("auto_route_tick", auto_run_id="a")
    assert session.wait_for_diagnostics(timeout=2)
    incidents = [json.loads(line) for line in (session.directory / "incidents.jsonl").read_text().splitlines()]
    assert any(row["failed_stream"] == "trajectory" for row in incidents)
    assert not session.healthy
    session.close(reason="test")


def test_fusion_inputs_survive_diagnostic_retention(tmp_path):
    from disk_policy import enforce_retention

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    session.localization("pose_result", display_seq=7, pose_capture_mono_ns=100,
                         camera_axes_world=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         fused_speed_mono=1., fused_altitude_m=2.,
                         reloc_ms=160., reloc_edm_match_ms=100., reloc_lift_prepare_ms=40.,
                         reloc_source_epoch=2, reloc_ordinal=7, reloc_capture_stamp_mono=.5,
                         ui_arrival_mono_ns=150, pose_status="VISUALLY_CONFIRMED")
    session.telemetry("fused_odometry", speed_mono_ns=100, speed_north_mps=.3,
                      attitude_mono_ns=90, altitude_mono_ns=80)
    session.telemetry("pcmd_dispatch", command_mono_ns=110, pcmd=[1, 2, 0, 0])
    session.close(reason="test")
    enforce_retention(tmp_path, current_session=None, max_bytes=0)
    rows = list(map(json.loads, (session.directory / "trajectory.jsonl").read_text().splitlines()))
    assert [r["event"] for r in rows] == ["pose_result", "fused_odometry", "pcmd_dispatch"]
    assert rows[0]["fused_speed_mono"] == 1.
    assert rows[0]["ui_arrival_mono_ns"] == 150
    assert rows[1]["speed_mono_ns"] == 100
    from report_relocalization import build_report, session_capture
    profile = build_report(session_capture(session.directory))
    assert profile["jobs"] == 1
    assert profile["stages"]["edm_match_ms"]["p50_ms"] == 100.


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


def test_dispatch_buffer_is_flushed_on_close_without_disk_io_at_enqueue(tmp_path, monkeypatch):
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    caller_id = threading.get_ident()
    with blocked_diagnostic_disk(session, monkeypatch, stream="telemetry") as (entered, release, ids):
        assert session.defer_telemetry("pcmd_dispatch", command_mono_ns=123, pcmd=[1, 2, 0, 3])
        assert entered.wait(2)
        assert ids == [session._writer.ident] and caller_id not in ids
        release.set()
        session.close(reason="done")
    row = json.loads((session.directory / "telemetry.jsonl").read_text())
    assert row["command_mono_ns"] == 123 and row["pcmd"] == [1, 2, 0, 3]
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["event_counts"]["telemetry"] == 1
    assert summary["debug_deferred_dropped"] == 0


def test_session_logs_keep_healthy_when_readback_is_an_olympe_enum(tmp_path):
    from types import SimpleNamespace

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    readback = {
        0: {
            "resolution": SimpleNamespace(name="resolution.res_1080p"),
            "framerate": SimpleNamespace(name="framerate.fps_30"),
        }
    }
    assert session.command("record_quality", readback=readback)
    assert session.healthy is True
    assert session.durable is True
    session.close(reason="done")
    row = json.loads((session.directory / "commands.jsonl").read_text())
    assert row["event"] == "record_quality"
    assert row["readback"]["0"]["resolution"] == "res_1080p"


def test_debug_log_uses_null_for_unavailable_numbers(tmp_path):
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    assert session.telemetry("test", sample=[float("nan"), float("inf")])
    session.close(reason="done")
    row = json.loads((session.directory / "telemetry.jsonl").read_text())
    assert row["sample"] == [None, None]


def test_concurrent_producers_leave_complete_json_lines(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda seq: session.localization("pose", seq=seq), range(100)))
    session.close(reason="done")
    assert all(results)
    rows = [json.loads(line) for line in (session.directory / "localization.jsonl").read_text().splitlines()]
    assert sorted(row["seq"] for row in rows) == list(range(100))


def test_debug_queue_overflow_is_reported(tmp_path, monkeypatch):
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    capacity = session_logs._DIAGNOSTIC_QUEUE_CAPACITY
    with blocked_diagnostic_disk(session, monkeypatch, stream="telemetry") as (entered, release, _):
        assert session.defer_telemetry("pcmd_dispatch", dispatch_seq=-1)
        assert entered.wait(2)
        for seq in range(capacity):
            assert session.defer_telemetry("pcmd_dispatch", dispatch_seq=seq)
        assert not session.defer_telemetry("pcmd_dispatch", dispatch_seq=capacity)
        assert not session.localization("auto_route_tick", display_seq=capacity)
        assert session.healthy is False
        # Overflow does not move disk I/O onto the producer/control thread.
        assert session.command("hover", request_id="overflow-safety")
        release.set()
        assert session.wait_for_diagnostics(timeout=5)
        markers = [json.loads(line) for line in (session.directory / "incidents.jsonl").read_text().splitlines()]
        assert {(row["stream"], row["dropped_total"]) for row in markers} == {
            ("telemetry", 1), ("localization", 1),
        }
        session.close(reason="done")
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["debug_deferred_dropped"] == 1
    assert summary["diagnostic_dropped"]["telemetry"] == 1
    assert summary["diagnostic_dropped"]["localization"] == 1
    assert summary["event_counts"]["telemetry"] == capacity + 1
    assert summary["recording_complete"] is False
    telemetry = [json.loads(line) for line in (session.directory / "telemetry.jsonl").read_text().splitlines()]
    assert [row["dispatch_seq"] for row in telemetry] == [-1, *range(capacity)]
    retained = [json.loads(line) for line in (session.directory / "trajectory.jsonl").read_text().splitlines()]
    assert [row["dispatch_seq"] for row in retained if row["event"] == "pcmd_dispatch"] == [-1, *range(capacity)]


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
    assert session.wait_for_diagnostics(timeout=2)
    assert len(synced) == startup_syncs

    session.localization("pose_result", seq=session_logs._BULK_LOG_SYNC_EVERY)
    assert session.wait_for_diagnostics(timeout=2)
    assert len(synced) == startup_syncs + 2  # diagnostics plus retained trajectory

    session.telemetry("state", seq=0)
    session.command("hover")
    session.incident("stream_stale")
    assert len(synced) == startup_syncs + 4

    session.close(reason="test_complete")
    assert len(synced) == startup_syncs + 7  # final telemetry plus summary file/directory


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
    with pytest.raises(OSError, match="fsync unavailable"):
        session.close(reason="probe_failed")


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


def test_tracker_transitions_stay_out_of_incidents(tmp_path) -> None:
    """Flight 2026-09-15: 3095 of 3101 incidents were NUDGE/HOVER transitions."""
    session = SessionLogs.create(
        tmp_path,
        mode=InterfaceMode.REAL_FLIGHT,
        manifest={"offline": True},
    )

    session.command_log.event("tracker_state_transition", old="HOVER", new="NUDGE", reason="nudge")
    session.command_log.event("uncommanded_landing", old="flying", new="landing")
    session.close(reason="test_complete")

    def events(name):
        text = (session.directory / name).read_text()
        return [json.loads(line)["event"] for line in text.splitlines()]

    assert events("incidents.jsonl") == ["uncommanded_landing"]
    assert events("commands.jsonl") == ["tracker_state_transition", "uncommanded_landing"]


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
    # Unknown until the model worker reports; it must never be recorded as False.
    assert identity["cuda"]["available"] is None
    assert identity["cuda"]["status"] == "pending_worker_report"
    assert set(identity["cuda"]) >= {"available", "runtime_version", "device_names"}
    assert "nvidia_driver" in identity


def test_collecting_runtime_identity_does_not_load_torch() -> None:
    import subprocess
    import sys
    from pathlib import Path

    package_dir = Path(session_logs.__file__).resolve().parent
    code = (
        "import sys\n"
        f"sys.path[:0] = [{str(package_dir.parent)!r}, {str(package_dir)!r}]\n"
        "from session_logs import collect_runtime_identity\n"
        "collect_runtime_identity()\n"
        "raise SystemExit(0 if 'torch' not in sys.modules else 3)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False, timeout=60).returncode == 0


def test_worker_cuda_report_replaces_pending_manifest_value_and_summary(tmp_path) -> None:
    session = SessionLogs.create(
        tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={"runtime": collect_runtime_identity()},
    )
    report = {"available": True, "runtime_version": "12.8", "device_names": ["Test GPU"]}

    assert session.record_worker_cuda(report)
    manifest = json.loads((session.directory / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"]["cuda"] == {
        "available": True,
        "status": "worker_reported",
        "source": "live_localizer_worker",
        "runtime_version": "12.8",
        "device_names": ["Test GPU"],
    }
    assert manifest["runtime"]["packages"]
    session.close(reason="test")
    summary = json.loads((session.directory / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_cuda"]["device_names"] == ["Test GPU"]
    assert summary["runtime_cuda_error"] is None


def test_session_without_a_worker_report_keeps_cuda_unknown_not_false(tmp_path) -> None:
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    session.close(reason="test")
    summary = json.loads((session.directory / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_cuda"]["available"] is None
    assert summary["runtime_cuda"]["status"] == "worker_not_reported"


def test_malformed_worker_cuda_report_is_rejected_and_recorded(tmp_path) -> None:
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})

    assert not session.record_worker_cuda({"available": "yes", "device_names": []})
    session.close(reason="test")
    summary = json.loads((session.directory / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_cuda"]["available"] is None
    assert summary["runtime_cuda_error"]


def test_real_offline_controller_records_survive_readback_and_retention(tmp_path):
    from verify_flight_diagnostics import verify_controller_persistence

    result = verify_controller_persistence(tmp_path)
    assert result["controller_ticks"] > 0
    assert result["observed_route_samples"] > 0
    assert result["terminal_outcome"] == "cancelled"
    assert result["scripted_iterations"] == 16


def test_deterministic_write_close_drop_and_summary_failure_receipts(tmp_path):
    from verify_flight_diagnostics import verify_failure_scenarios

    paths = verify_failure_scenarios(tmp_path)
    assert set(paths) == {"write_failure", "close_failure", "queue_drop", "summary_retry"}


def test_close_final_sync_failure_is_in_persisted_summary(tmp_path, monkeypatch):
    """Pre-fix repro: summary used to claim healthy before the final sink fsync."""
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    session.localization("auto_route_tick", auto_run_id="known-run", altitude_error_u=.125)

    def fail_final_sync():
        raise OSError("injected final close sync")

    monkeypatch.setattr(session._sinks["trajectory"], "_sync", fail_final_sync)
    session.close(reason="failure_repro")
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["log_health"]["trajectory"]["healthy"] is False
    assert summary["log_health"]["trajectory"]["durable"] is False
    assert "injected final close sync" in summary["log_health"]["trajectory"]["last_error"]
    assert summary["recording_complete"] is False


def test_immutable_terminal_mapping_stays_numeric_after_retention(tmp_path):
    from types import MappingProxyType
    from disk_policy import enforce_retention

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    result = MappingProxyType({
        "auto_run_id": "immutable-run",
        "mean_u": .125,
        "map_frame": MappingProxyType({"east": (1., 0., 0.)}),
        "planned_waypoints_u": ((0., 0., 0.), (1., 2., 3.)),
    })
    session.localization("auto_route_error_summary", **result)
    session.telemetry("autonomy_event", result=result, auto_run_id="immutable-run")
    session.close(reason="test")
    enforce_retention(tmp_path, current_session=None, max_bytes=0)
    rows = [json.loads(line) for line in (session.directory / "trajectory.jsonl").read_text().splitlines()]
    assert rows[0]["mean_u"] == .125
    assert rows[0]["map_frame"]["east"] == [1., 0., 0.]
    assert rows[1]["result"]["planned_waypoints_u"] == [[0., 0., 0.], [1., 2., 3.]]
    assert rows[1]["result"]["auto_run_id"] == "immutable-run"


def test_close_shuts_deferred_admission_without_waiting_for_sink_fsync(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    assert session.defer_telemetry("pcmd_dispatch", dispatch_seq=1)
    entered = threading.Event()
    release = threading.Event()
    original = session._sinks["trajectory"]._sync

    def slow_close_sync():
        entered.set()
        assert release.wait(2)
        original()

    monkeypatch.setattr(session._sinks["trajectory"], "_sync", slow_close_sync)
    with ThreadPoolExecutor(max_workers=1) as pool:
        closing = pool.submit(session.close, reason="concurrent_close")
        assert entered.wait(2)
        try:
            assert not session.defer_telemetry("pcmd_dispatch", dispatch_seq=2)
        finally:
            release.set()
        closing.result(timeout=2)
    rows = [json.loads(line) for line in (session.directory / "trajectory.jsonl").read_text().splitlines()]
    assert [row["dispatch_seq"] for row in rows] == [1]


def test_actual_sdk_enqueue_producer_values_survive_retained_readback(tmp_path):
    from verify_flight_diagnostics import verify_sdk_enqueue_persistence

    result = verify_sdk_enqueue_persistence(tmp_path)
    assert result["auto_run_id"] == "sdk-known-run"
    assert result["pcmd"] == [11, -12, 13, -14]


def test_numeric_scalar_metrics_are_numbers_not_repr_strings(tmp_path):
    import numpy as np

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    assert session.localization(
        "auto_route_error_summary", auto_run_id="numpy-run",
        samples=np.int64(7), mean_u=np.float32(.125),
    )
    session.close(reason="numeric_readback")
    row = json.loads((session.directory / "trajectory.jsonl").read_text())
    assert row["samples"] == 7 and isinstance(row["samples"], int)
    assert row["mean_u"] == .125 and isinstance(row["mean_u"], float)


def test_slow_diagnostic_fsync_does_not_block_producers_or_durable_safety(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    with blocked_diagnostic_disk(session, monkeypatch, method="_sync") as (entered, release, ids):
        for seq in range(session_logs._BULK_LOG_SYNC_EVERY):
            assert session.localization("pose_result", display_seq=seq)
        assert entered.wait(2)
        safety_syncs = []
        for stream in ("commands", "incidents"):
            sink = session._sinks[stream]
            original = sink._sync

            def safety_sync(original=original, stream=stream):
                original()
                safety_syncs.append(stream)

            monkeypatch.setattr(sink, "_sync", safety_sync)
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(session.telemetry, "readback", battery_pct=92).result(timeout=2)
            assert pool.submit(session.defer_telemetry, "pcmd_dispatch", pcmd=[0, 0, 0, 0]).result(timeout=2)
            assert pool.submit(session.command, "hover", request_id="durable").result(timeout=2)
            assert safety_syncs == ["commands"]
            assert pool.submit(session.incident, "stream_lost", reason="test").result(timeout=2)
            assert safety_syncs == ["commands", "incidents"]
        assert not release.is_set()
        assert set(ids) == {session._writer.ident}
        command = json.loads((session.directory / "commands.jsonl").read_text())
        assert command["request_id"] == "durable"
        incident = json.loads((session.directory / "incidents.jsonl").read_text())
        assert incident["event"] == "stream_lost"
        release.set()
        session.close(reason="slow_disk")
    assert not session._writer.is_alive()
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["recording_complete"] is True


def test_diagnostic_rows_snapshot_nested_values_and_producer_clock_in_admission_order(tmp_path, monkeypatch):
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    with blocked_diagnostic_disk(session, monkeypatch) as (entered, release, _):
        assert session.localization("warmup")
        assert entered.wait(2)
        clock = {"ns": 1234, "utc": "producer-utc"}
        monkeypatch.setattr(session_logs.time, "monotonic_ns", lambda: clock["ns"])
        monkeypatch.setattr(session_logs, "_utc_now", lambda: clock["utc"])
        pose = {"x": 1, "y": 2, "z": 3, "nested": {"values": [4, 5]}}
        samples = [[.1, .2, 1.0]]
        pcmd = [1, 2, 3, 4]
        waypoints = [[0, 0, 0], [1, 2, 3]]
        assert session.localization("pose_result", pose=pose, display_seq=7)
        assert session.telemetry("fused_odometry", samples=samples)
        assert session.defer_telemetry("pcmd_dispatch", pcmd=pcmd)
        assert session.localization("auto_route_tick", waypoints_u=waypoints)
        pose["nested"]["values"][0] = 99
        samples[0][0] = 99
        pcmd[:] = [99]
        waypoints[1][:] = [99]
        clock.update(ns=9999, utc="drain-utc")
        release.set()
        assert session.wait_for_diagnostics(timeout=2)
        localized = [json.loads(line) for line in (session.directory / "localization.jsonl").read_text().splitlines()][1:]
        telemetry = [json.loads(line) for line in (session.directory / "telemetry.jsonl").read_text().splitlines()]
        retained = [json.loads(line) for line in (session.directory / "trajectory.jsonl").read_text().splitlines()]
        assert [row["event"] for row in retained] == [
            "pose_result", "fused_odometry", "pcmd_dispatch", "auto_route_tick",
        ]
        assert retained[0]["pose"]["nested"]["values"] == [4, 5]
        assert retained[1]["samples"] == [[.1, .2, 1.0]]
        assert retained[2]["pcmd"] == [1, 2, 3, 4]
        assert retained[3]["waypoints_u"] == [[0, 0, 0], [1, 2, 3]]
        sources = [localized[0], telemetry[0], telemetry[1], localized[1]]
        for source, mirror in zip(sources, retained):
            assert source["t_mono_ns"] == mirror["t_mono_ns"] == 1234
            assert source["t_utc"] == mirror["t_utc"] == "producer-utc"
            assert source["session_id"] == mirror["session_id"] == session.directory.name
        session.close(reason="snapshot")


def test_background_bulk_fsync_failure_reports_health_and_incident_before_close(tmp_path, monkeypatch):
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})

    def failed_sync():
        raise OSError("injected diagnostic bulk fsync")

    monkeypatch.setattr(session._sinks["localization"], "_sync", failed_sync)
    for seq in range(session_logs._BULK_LOG_SYNC_EVERY):
        assert session.localization("pose_result", display_seq=seq)
    assert session.wait_for_diagnostics(timeout=2)
    assert not session.healthy and not session.durable
    incidents = [json.loads(line) for line in (session.directory / "incidents.jsonl").read_text().splitlines()]
    assert [(row["event"], row["failed_stream"]) for row in incidents] == [
        ("diagnostic_write_failed", "localization"),
    ]
    assert "injected diagnostic bulk fsync" in incidents[0]["last_error"]
    assert session.command("hover", request_id="still-durable")
    session.close(reason="bulk_failure")
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["write_failures"]["localization"] == 1
    assert summary["event_counts"]["localization"] == session_logs._BULK_LOG_SYNC_EVERY - 1
    assert not summary["recording_complete"]


def test_close_stops_all_admissions_and_drains_an_accepted_blocked_batch(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    with blocked_diagnostic_disk(session, monkeypatch) as (entered, release, _):
        assert session.localization("pose_result", display_seq=1)
        assert entered.wait(2)
        assert session.defer_telemetry("pcmd_dispatch", dispatch_seq=2)
        assert session.telemetry("readback", seq=3)
        with ThreadPoolExecutor(max_workers=1) as pool:
            closing = pool.submit(session.close, reason="drained")
            with session._condition:
                assert session._condition.wait_for(lambda: session._closing, timeout=2)
            assert not closing.done()
            assert not session.localization("pose_result", display_seq=4)
            assert not session.telemetry("readback", seq=5)
            assert not session.defer_telemetry("pcmd_dispatch", dispatch_seq=6)
            assert not session.command("hover")
            assert not session.incident("worker_exit")
            assert not session.write_inventory("video", {})
            release.set()
            closing.result(timeout=2)
    assert not session._writer.is_alive()
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["event_counts"] == {
        "commands": 0, "incidents": 0, "localization": 1, "telemetry": 2, "trajectory": 2,
    }
    assert summary["recording_complete"] is True
    retained = [json.loads(line) for line in (session.directory / "trajectory.jsonl").read_text().splitlines()]
    assert [row["event"] for row in retained] == ["pose_result", "pcmd_dispatch"]


def test_close_waits_for_an_admitted_synchronous_safety_write(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    with blocked_diagnostic_disk(session, monkeypatch, stream="commands") as (entered, release, _):
        with ThreadPoolExecutor(max_workers=2) as pool:
            command = pool.submit(session.command, "hover", request_id="accepted-before-close")
            assert entered.wait(2)
            assert session.localization("pose_result", display_seq=1)
            closing = pool.submit(session.close, reason="safety_drained")
            with session._condition:
                assert session._condition.wait_for(lambda: session._closing, timeout=2)
            assert not closing.done()
            release.set()
            assert command.result(timeout=2)
            closing.result(timeout=2)
    row = json.loads((session.directory / "commands.jsonl").read_text())
    assert row["request_id"] == "accepted-before-close"
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["event_counts"]["commands"] == 1
    assert summary["recording_complete"] is True


def test_failed_safety_fsync_returns_false_and_reports_without_diagnostic_drain(tmp_path, monkeypatch):
    from concurrent.futures import Future, ThreadPoolExecutor

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    diagnostic_entered = threading.Event()
    diagnostic_release = threading.Event()
    diagnostic_result = Future()
    original = session._sinks["localization"].write

    def blocked_diagnostic_write(row):
        diagnostic_entered.set()
        diagnostic_release.wait()
        try:
            result = original(row)
        except BaseException as exc:
            diagnostic_result.set_exception(exc)
            raise
        else:
            diagnostic_result.set_result(result)
            return result

    def failed_command_sync():
        raise OSError("injected safety fsync failure")

    monkeypatch.setattr(session._sinks["localization"], "write", blocked_diagnostic_write)
    monkeypatch.setattr(session._sinks["commands"], "_sync", failed_command_sync)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            assert session.localization("pose_result", display_seq=1)
            assert diagnostic_entered.wait(2)
            safety = pool.submit(session.command, "hover", request_id="not-durable")
            # This result must complete before releasing diagnostic I/O. The
            # diagnostic barrier has no timeout that could mask a blocking bug.
            assert safety.result(timeout=2) is False
            assert not diagnostic_release.is_set()
            assert not diagnostic_result.done()
            assert session.healthy is False and session.durable is False
            row = json.loads((session.directory / "incidents.jsonl").read_text())
            assert row["event"] == "diagnostic_write_failed"
            assert row["failed_stream"] == "commands"
            assert "injected safety fsync failure" in row["last_error"]
        finally:
            diagnostic_release.set()
    assert diagnostic_result.result(timeout=2)
    assert session.wait_for_diagnostics(timeout=2)
    session.close(reason="safety_failure")
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["write_failures"]["commands"] == 1
    assert summary["event_counts"]["commands"] == 0
    assert not summary["recording_complete"]


@pytest.mark.parametrize("fail_inventory", [False, True])
def test_inventory_disk_io_never_blocks_producers_and_close_accounts_for_it(
    tmp_path, monkeypatch, fail_inventory,
):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    entered = threading.Event()
    release = threading.Event()
    original = Path.write_text

    def blocked_inventory_write(path, *args, **kwargs):
        if path == session.directory / ".hardware_inventory.json.tmp":
            entered.set()
            release.wait()
            if fail_inventory:
                raise OSError("injected inventory failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", blocked_inventory_write)
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            inventory = pool.submit(session.write_inventory, "hardware", {"serial": "FIRST"})
            assert entered.wait(2)
            for producer, event in (
                (session.localization, "pose_result"),
                (session.telemetry, "readback"),
                (session.defer_telemetry, "pcmd_dispatch"),
                (session.command, "hover"),
                (session.incident, "stream_lost"),
            ):
                assert pool.submit(producer, event).result(timeout=2)
            assert not release.is_set() and not inventory.done()
            assert session.wait_for_diagnostics(timeout=2)
            closing = pool.submit(session.close, reason="inventory_drained")
            with session._condition:
                assert session._condition.wait_for(lambda: session._closing, timeout=2)
            assert not closing.done()
            assert not (session.directory / "session_summary.json").exists()
            assert not session.write_inventory("video", {})
        finally:
            release.set()
        assert inventory.result(timeout=2) is not fail_inventory
        closing.result(timeout=2)
    summary = json.loads((session.directory / "session_summary.json").read_text())
    assert summary["recording_complete"] is not fail_inventory
    if fail_inventory:
        assert "injected inventory failure" in summary["inventory_error"]
        assert session.durable is False
    else:
        assert summary["inventory_error"] is None
        receipt = json.loads((session.directory / "hardware_inventory.json").read_text())
        assert receipt["inventory"] == {"serial": "FIRST"}


def test_concurrent_inventory_receipts_preserve_first_success(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    original = Path.write_text
    writes = []

    def blocked_inventory_write(path, *args, **kwargs):
        if path == session.directory / ".hardware_inventory.json.tmp":
            writes.append(path)
            entered.set()
            release.wait()
        return original(path, *args, **kwargs)

    def second_inventory():
        second_started.set()
        return session.write_inventory("hardware", {"serial": "SECOND"})

    monkeypatch.setattr(Path, "write_text", blocked_inventory_write)
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            first = pool.submit(session.write_inventory, "hardware", {"serial": "FIRST"})
            assert entered.wait(2)
            second = pool.submit(second_inventory)
            assert second_started.wait(2)
            with session._condition:
                assert session._condition.wait_for(
                    lambda: session._active_inventory_writes == 2, timeout=2,
                )
            closing = pool.submit(session.close, reason="serialized_inventory_drained")
            with session._condition:
                assert session._condition.wait_for(lambda: session._closing, timeout=2)
            assert not closing.done()
        finally:
            release.set()
        assert first.result(timeout=2)
        assert second.result(timeout=2)
        closing.result(timeout=2)
    receipt = json.loads((session.directory / "hardware_inventory.json").read_text())
    assert receipt["inventory"] == {"serial": "FIRST"}
    assert len(writes) == 1


@pytest.mark.parametrize("registration_error_type", [RuntimeError, SystemExit])
@pytest.mark.parametrize("fail_summary", [False, True])
def test_rejected_exit_registration_drains_owned_session_before_propagating(
    tmp_path, monkeypatch, registration_error_type, fail_summary,
):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from types import SimpleNamespace

    import operator_launch as launch

    created = []
    created_event = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    registration_attempts = []
    registration_error = registration_error_type("injected exit registration failure")
    original_create = SessionLogs.create
    original_open = Path.open

    def create(*args, **kwargs):
        session = original_create(*args, **kwargs)
        original_write = session._sinks["telemetry"].write

        def blocked_write(row):
            entered.set()
            release.wait()
            return original_write(row)

        monkeypatch.setattr(session._sinks["telemetry"], "write", blocked_write)
        created.append(session)
        created_event.set()
        return session

    def reject_registration(callback):
        registration_attempts.append(callback)
        session = created[0]
        assert session.defer_telemetry("pcmd_dispatch", dispatch_seq=0)
        assert entered.wait(2)
        assert session.defer_telemetry("pcmd_dispatch", dispatch_seq=1)
        assert session.localization("pose_result", display_seq=2)
        assert session.command("partial_launch", evidence_kind="offline_local_acceptance")
        raise registration_error

    def open_with_summary_failure(path, *args, **kwargs):
        if fail_summary and path.name == ".session_summary.json.tmp":
            raise OSError("injected summary persistence failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(SessionLogs, "create", create)
    monkeypatch.setattr(Path, "open", open_with_summary_failure)
    monkeypatch.setattr(launch.app, "_WS", SimpleNamespace(flight_logs=tmp_path))
    monkeypatch.setattr(launch.app, "atexit", SimpleNamespace(register=reject_registration))
    monkeypatch.setattr(launch, "_operator_session_identity", lambda *args: object())
    monkeypatch.setattr(launch, "_operator_session_manifest", lambda *args: {"offline": True})
    prepared = SimpleNamespace(
        args=SimpleNamespace(),
        interface_mode=InterfaceMode.SIMULATED_STREAM,
        profile=None,
        hardware_approval=None,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        starting = pool.submit(launch._start_operator_site_runtime, prepared)
        try:
            assert created_event.wait(2)
            session = created[0]
            assert entered.wait(2)
            with session._condition:
                assert session._condition.wait_for(lambda: session._closing, timeout=2)
                assert session._diagnostic_accepted == 3
                assert session._diagnostic_completed == 0
            assert not starting.done()
            assert session._writer.is_alive()
            assert not session.telemetry("late_diagnostic")
            assert not (session.directory / "session_summary.json").exists()
        finally:
            release.set()
        with pytest.raises(registration_error_type) as raised:
            starting.result(timeout=2)

    assert raised.value is registration_error
    assert len(registration_attempts) == 1  # No hook is invoked or registered.
    assert session._closed
    assert not session._writer.is_alive()
    assert all(sink._handle.closed for sink in session._sinks.values())
    assert session._diagnostic_completed == session._diagnostic_accepted == 3
    telemetry = [
        json.loads(line)
        for line in (session.directory / "telemetry.jsonl").read_text().splitlines()
    ]
    assert [row["dispatch_seq"] for row in telemetry] == [0, 1]
    pose = json.loads((session.directory / "localization.jsonl").read_text())
    assert pose["event"] == "pose_result" and pose["display_seq"] == 2
    command = json.loads((session.directory / "commands.jsonl").read_text())
    assert command["evidence_kind"] == "offline_local_acceptance"
    trajectory = [
        json.loads(line)
        for line in (session.directory / "trajectory.jsonl").read_text().splitlines()
    ]
    assert trajectory == [*telemetry, pose]
    assert all(sink.healthy and sink.durable for sink in session._sinks.values())
    if fail_summary:
        assert "injected summary persistence failure" in session._summary_error
        assert session._summary_write_failures == 1
        assert not session.durable
        assert not (session.directory / "session_summary.json").exists()
    else:
        summary = json.loads((session.directory / "session_summary.json").read_text())
        assert summary["reason"] == "exit_registration_failed"
        assert summary["event_counts"] == {
            "commands": 1, "localization": 1, "telemetry": 2, "incidents": 0, "trajectory": 3,
        }
        assert summary["recording_complete"] is True
        assert all(
            health["healthy"] and health["durable"]
            for health in summary["log_health"].values()
        )
        assert session.durable


@pytest.mark.parametrize("failure_stage", ["backend", "retention"])
def test_partial_launch_exit_runs_real_atexit_cleanup_and_retains_logs(tmp_path, failure_stage):
    from verify_flight_diagnostics import verify_partial_launch_exit

    result = verify_partial_launch_exit(tmp_path, failure_stage=failure_stage)
    assert result["reason"] == "process_atexit"
    assert result["recording_complete"]
    assert result["accepted_diagnostics"] == 32
