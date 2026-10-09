#!/usr/bin/env python3
"""Offline producer-to-retained-JSONL smoke; never imports/connects an aircraft SDK."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "控制介面程式/operator_interface", ROOT / "定位演算法/flight_control"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import numpy as np
import path_follow_flight as pff
import session_logs as logs_module
from backend_contract import InterfaceMode
from backend_telemetry import poll_session_telemetry
from disk_policy import enforce_retention
from localization_metrics import build_localization_metric_record
from operator_autonomy import DesktopRouteAutonomy
from real_path_follow_controller import MapFrame, MissionRouteSnapshot, Pose
from session_logs import SessionLogs


def _rows(directory: Path, stream: str) -> list[dict]:
    return [json.loads(line) for line in (directory / f"{stream}.jsonl").read_text().splitlines()]


def _summary(directory: Path) -> dict:
    return json.loads((directory / "session_summary.json").read_text())


class _Clock:
    value = 10.0

    def now(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class _OfflineBackend:
    """Only records simulated local acceptance; there is no wire or aircraft."""

    def __init__(self, session, clock):
        self.session_logs = session
        self.clock = clock
        self.pilot_sticks = False
        self._runtime_safety_action_latched = False
        self.state = SimpleNamespace(
            stream="OK",
            flight_state="hovering",
            control_owner="PC",
            battery_pct=91,
            att_roll=0.02,
            att_pitch=-0.03,
            att_yaw=math.pi / 2,
            ground_speed_mps=math.hypot(0.12, -0.08),
            speed_north_mps=0.12,
            speed_east_mps=-0.08,
            speed_down_mps=0.01,
            ground_speed_mono_ns=9_900_000_000,
            attitude_mono_ns=9_910_000_000,
            altitude_mono_ns=9_920_000_000,
            telemetry_read_mono_ns=9_930_000_000,
            ground_speed_stamp_source="simulated_event_receipt",
            attitude_stamp_source="simulated_event_receipt",
            altitude_stamp_source="simulated_event_receipt",
            drone_altitude_m=1.75,
            agl_altitude_m=1.6,
            gimbal_pitch_deg=-10.0,
            gimbal_yaw_deg=1.0,
            gimbal_roll_deg=2.0,
            gimbal_mono_ns=9_940_000_000,
            gimbal_stamp_source="simulated_event_receipt",
            camera_zoom_level=1.25,
            camera_zoom_mono_ns=9_950_000_000,
            camera_zoom_stamp_source="simulated_event_receipt",
            zoom=1.25,
            autonomous_speed_limit_enabled=True,
            autonomous_speed_limit_mps=0.3,
        )
        self.dispatch_seq = 0

    def send_pcmd(self, roll, pitch, yaw, gaz, *, reason, auto_run_id=None):
        self.last_requested_pcmd = (roll, pitch, yaw, gaz)
        self.session_logs.command(
            "simulated_request",
            pcmd=[roll, pitch, yaw, gaz],
            reason=reason,
            auto_run_id=auto_run_id,
            evidence_kind="offline_local_acceptance",
        )
        return True

    def set_nudge_vector(self, roll, pitch, yaw, gaz, *, authority_pct, auto_run_id=None):
        self.dispatch_seq += 1
        pcmd = [round(axis * authority_pct) for axis in (roll, pitch, yaw, gaz)]
        self.last_requested_pcmd = tuple(pcmd)
        self.session_logs.command(
            "simulated_request",
            pcmd=pcmd,
            auto_run_id=auto_run_id,
            reason="desktop_auto_route",
            evidence_kind="offline_local_acceptance",
        )
        self.session_logs.defer_telemetry(
            "pcmd_dispatch",
            pcmd=pcmd,
            auto_run_id=auto_run_id,
            dispatch_seq=self.dispatch_seq,
            command_mono_ns=round(self.clock.now() * 1e9),
            command_set_mono_ns=round(self.clock.now() * 1e9) - 100,
            sdk_enqueue_start_mono_ns=round(self.clock.now() * 1e9),
            sdk_enqueue_return_mono_ns=round(self.clock.now() * 1e9) + 25,
            reason="desktop_auto_route",
            evidence_kind="offline_simulated_sdk_enqueue_not_physical_confirmation",
        )
        return True

    def clear_nudge_vector(self):
        self.last_requested_pcmd = (0, 0, 0, 0)

    def give_to_pilot(self, *, reason):
        self.pilot_sticks = True
        return True

    def _poll_stick_axes(self, now):
        self.session_logs.telemetry(
            "stick_axes", axes=[0.0, 0.0, 0.0, 0.0], source_mono_ns=round(now * 1e9)
        )

    def _video_delivery_snapshot(self, now):
        return {"evidence_kind": "offline_simulated_readback"}


def verify_controller_persistence(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    route_path = root / "offline_route.json"
    points = ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    route_path.write_text(json.dumps({"waypoints": points, "frame": "glomap", "units": "map"}))
    snapshot = MissionRouteSnapshot(
        path=route_path,
        sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
        site_id="offline-known-site",
        coordinate_frame_id="offline-ENU",
        waypoints=points,
    )
    session = SessionLogs.create(
        root,
        mode=InterfaceMode.SIMULATED_STREAM,
        manifest={"offline": True, "diagnostic_smoke": True},
    )
    clock = _Clock()
    backend = _OfflineBackend(session, clock)
    poll_session_telemetry(backend, clock.now())
    latest = {}
    calls = 0

    def get_pose():
        nonlocal calls
        calls += 1
        x = -0.3 if calls <= 3 else 0.0
        stamp_ns = round(clock.now() * 1e9)
        for name in ("ground_speed_mono_ns", "attitude_mono_ns", "altitude_mono_ns"):
            setattr(backend.state, name, stamp_ns)
        result = {
            "success": True,
            "display_seq": calls,
            "pose_capture_mono_ns": stamp_ns,
            "source_frame_stamp_mono": clock.now(),
            "pose_status": "VISUALLY_CONFIRMED",
            "direct_status": "TRACK",
            "pose": {"x": x, "y": 0.005, "z": 0.0, "yaw": 0.0},
            "pose_raw": {"x": x, "y": 0.005, "z": 0.0, "yaw": 0.0},
            "pose_filter": "offline_identity",
            "map_constraint_age_s": 0.025,
            "map_inliers": 100,
            "vo_inliers": 8,
            "inliers": 100,
            "reproj_rms": 0.75,
            "klt_input_points": 120,
            "klt_kept_points": 100,
            "klt_fb_p95_px": 0.4,
            "imu_bridge": False,
            "imu_sample_age_s": 0.01,
            "position_observed": True,
            "camera_attitude_kind": "full",
            "camera_axes_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
        latest.clear()
        latest.update(
            build_localization_metric_record(
                result,
                metric_mono_ns=stamp_ns + 1,
                loc_fps=20.0,
                submit_ok=calls,
                submit_skip_busy=0,
                submit_busy_attempts=0,
            )
        )
        session.localization("pose_result", **latest)
        return Pose(x, 0.005, 0.0, 0.0, stamp=clock.now())

    frame = MapFrame(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        source="offline-known-ENU",
    )
    autonomy = DesktopRouteAutonomy(
        backend=backend,
        snapshot=snapshot,
        map_frame=frame,
        get_pose=get_pose,
        pose_info=lambda: latest,
        pose_is_weak=lambda: False,
        pose_confidence=lambda: 100,
        force_relocalize=lambda: None,
        stream_healthy=lambda: True,
        takeoff=lambda: True,
        land=lambda: True,
        now=clock.now,
        sleep=clock.sleep,
    )
    backend.run_id = autonomy.run_id
    autonomy._log_route_plan()
    hooks = autonomy._route_hooks()
    original_beat = hooks.loop_beat
    iterations = 0

    def bounded_beat():
        nonlocal iterations
        iterations += 1
        original_beat()
        if iterations == 16:
            # Use the coordinator's explicit retirement contract. MANUAL is a
            # resumable suspended state in the production loop, not an exit.
            autonomy.cancel("offline_smoke_iteration_cap")

    hooks.loop_beat = bounded_beat
    with patch.object(pff.time, "sleep", clock.sleep):
        terminal = pff.run_loop(hooks, autonomy.controller, autonomy.waypoints, verbose=False)
    assert autonomy.join(timeout=1.0), "offline cancellation handoff did not stop"
    autonomy._report_route_error(terminal)
    session.telemetry(
        "autonomy_event",
        auto_run_id=autonomy.run_id,
        kind="finished",
        result=autonomy._route_error.summary(),
    )
    session.close(reason="offline_smoke_complete")

    rows = _rows(session.directory, "trajectory")
    plan = next(row for row in rows if row["event"] == "auto_route_plan")
    ticks = [row for row in rows if row["event"] == "auto_route_tick" and "pose_u" in row]
    terminal_summary = next(row for row in rows if row["event"] == "auto_route_error_summary")
    fused = next(row for row in rows if row["event"] == "fused_odometry")
    poses = [row for row in rows if row["event"] == "pose_result"]
    assert plan["route_sha256"] == snapshot.sha256
    assert plan["site_id"] == snapshot.site_id and plan["route_path"] == str(route_path)
    assert plan["control_config"]["map_frame"]["east"] == [1.0, 0.0, 0.0]
    assert plan["waypoints_u"] == [list(point) for point in autonomy.controller.wp]
    assert ticks and all(row["auto_run_id"] == autonomy.run_id for row in ticks)
    assert any(
        row["pose_u"] == [0.0, 0.005, 0.0] and abs(row["route_distance_u"] - 0.005) < 1e-10
        for row in ticks
    )
    assert any(
        all(
            key in row
            for key in (
                "pcmd_controller_raw",
                "pcmd_requested",
                "pcmd",
                "horizontal_control_terms",
            )
        )
        for row in ticks
    )
    assert terminal_summary["auto_run_id"] == autonomy.run_id
    assert terminal_summary["route_sha256"] == snapshot.sha256
    assert terminal_summary["samples"] == sum(bool(row.get("route_error_sample")) for row in ticks)
    assert terminal_summary["samples"] > 0 and terminal_summary["mean_u"] == 0.005
    assert fused["speed_north_mps"] == 0.12 and fused["speed_mono_ns"] == 9_900_000_000
    assert fused["att_roll"] == 0.02 and fused["attitude_mono_ns"] == 9_910_000_000
    assert fused["gimbal_pitch_deg"] == -10.0 and fused["gimbal_mono_ns"] == 9_940_000_000
    assert fused["camera_zoom_level"] == 1.25 and fused["camera_zoom_mono_ns"] == 9_950_000_000
    assert fused["drone_altitude_m"] == 1.75 and fused["altitude_mono_ns"] == 9_920_000_000
    assert all(row["klt_kept_points"] == 100 and row["imu_sample_age_s"] == 0.01 for row in poses)
    assert all(row["camera_attitude_kind"] == "full" for row in poses)
    dispatch = next(row for row in rows if row["event"] == "pcmd_dispatch")
    assert dispatch["auto_run_id"] == autonomy.run_id and dispatch["command_mono_ns"] > 0
    assert dispatch["command_set_mono_ns"] == dispatch["command_mono_ns"] - 100
    assert dispatch["sdk_enqueue_start_mono_ns"] == dispatch["command_mono_ns"]
    assert dispatch["sdk_enqueue_return_mono_ns"] == dispatch["command_mono_ns"] + 25
    assert dispatch["reason"] == "desktop_auto_route"
    assert _rows(session.directory, "commands")[0]["evidence_kind"] == "offline_local_acceptance"
    assert _summary(session.directory)["recording_complete"] is True
    enforce_retention(root, current_session=None, max_bytes=0)
    assert not (session.directory / "localization.jsonl").exists()
    assert not (session.directory / "telemetry.jsonl").exists()
    assert _rows(session.directory, "trajectory") == rows
    assert (session.directory / "commands.jsonl").exists()
    assert terminal_summary["outcome"] == "cancelled"
    return {
        "session": str(session.directory),
        "auto_run_id": autonomy.run_id,
        "controller_ticks": len(ticks),
        "observed_route_samples": terminal_summary["samples"],
        "terminal_outcome": terminal_summary["outcome"],
        "scripted_iterations": iterations,
    }


class _FailedWriteHandle:
    def __init__(self, handle):
        self._handle = handle

    def write(self, value):
        raise OSError("injected trajectory write failure")

    def __getattr__(self, name):
        return getattr(self._handle, name)


class _FailedCloseHandle:
    def __init__(self, handle):
        self._handle = handle

    def close(self):
        self._handle.close()
        raise OSError("injected handle close failure")

    def __getattr__(self, name):
        return getattr(self._handle, name)


def verify_failure_scenarios(root: Path) -> dict:
    def create():
        return SessionLogs.create(
            root, mode=InterfaceMode.SIMULATED_STREAM, manifest={"offline": True}
        )

    write = create()
    write._sinks["trajectory"]._handle = _FailedWriteHandle(write._sinks["trajectory"]._handle)
    assert not write.localization("auto_route_tick", auto_run_id="failure-run", pose_u=[1, 2, 3])
    write.close(reason="injected_write_failure")
    assert _summary(write.directory)["write_failures"]["trajectory"] >= 1
    assert not _summary(write.directory)["recording_complete"]
    assert any(
        row["event"] == "diagnostic_write_failed" for row in _rows(write.directory, "incidents")
    )

    close = create()
    close.localization("auto_route_tick", auto_run_id="close-run", value=7)
    close._sinks["telemetry"]._handle = _FailedCloseHandle(close._sinks["telemetry"]._handle)
    with patch.object(
        close._sinks["trajectory"], "_sync", side_effect=OSError("injected final fsync failure")
    ):
        close.close(reason="injected_close_failure")
    health = _summary(close.directory)["log_health"]["trajectory"]
    assert not health["healthy"] and not health["durable"]
    assert "injected final fsync failure" in health["last_error"]
    assert not _summary(close.directory)["recording_complete"]
    close_health = _summary(close.directory)["log_health"]["telemetry"]
    assert not close_health["healthy"] and not close_health["durable"]
    assert "injected handle close failure" in close_health["last_error"]

    drop = create()
    for seq in range(512):
        assert drop.defer_telemetry("pcmd_dispatch", dispatch_seq=seq)
    assert not drop.defer_telemetry("pcmd_dispatch", dispatch_seq=512)
    drop.flush_deferred()
    marker = next(
        row
        for row in _rows(drop.directory, "incidents")
        if row["event"] == "diagnostic_records_dropped"
    )
    assert marker["dropped_total"] == 1  # Visible before close, not just in the final summary.
    drop.close(reason="injected_queue_drop")
    assert _summary(drop.directory)["debug_deferred_dropped"] == 1
    assert not _summary(drop.directory)["recording_complete"]
    assert not drop.defer_telemetry("pcmd_dispatch", dispatch_seq=513)

    summary = create()
    summary.command("known", immutable=MappingProxyType({"value": 3.25}))
    with patch.object(
        logs_module.os, "replace", side_effect=OSError("injected summary replace failure")
    ):
        try:
            summary.close(reason="summary_failure")
        except OSError:
            pass
        else:
            raise AssertionError("summary persistence failure must propagate")
    assert summary.durable is False and not (summary.directory / "session_summary.json").exists()
    summary.close(reason="summary_retry")
    assert _summary(summary.directory)["summary_write_failures"] == 1
    assert _rows(summary.directory, "commands")[0]["immutable"] == {"value": 3.25}
    enforce_retention(root, current_session=None, max_bytes=0)
    for session in (write, close, drop, summary):
        assert (session.directory / "session_summary.json").exists()
        assert (session.directory / "incidents.jsonl").exists()
    return {
        "write_failure": str(write.directory),
        "close_failure": str(close.directory),
        "queue_drop": str(drop.directory),
        "summary_retry": str(summary.directory),
    }


def verify_sdk_enqueue_persistence(root: Path) -> dict:
    """Exercise the production SDK-boundary producer against an in-memory transport."""
    from olympe_live_backend import OlympeLiveBackend

    session = SessionLogs.create(
        root, mode=InterfaceMode.SIMULATED_STREAM, manifest={"offline": True}
    )
    backend = OlympeLiveBackend.__new__(OlympeLiveBackend)
    backend.session_logs = session
    backend.state = SimpleNamespace()
    calls = []
    backend.drone = calls.append  # No connection, SDK object, or hardware capability.
    names = (
        "olympe",
        "olympe.messages",
        "olympe.messages.ardrone3",
        "olympe.messages.ardrone3.Piloting",
    )
    modules = {name: ModuleType(name) for name in names}
    for name in names[:-1]:
        modules[name].__path__ = []
    modules[names[-1]].PCMD = lambda *values: tuple(values)
    stamps = itertools.count(2_000_000_000, 25)
    with (
        patch.dict(sys.modules, modules),
        patch.object(logs_module.time, "monotonic_ns", lambda: next(stamps)),
    ):
        returned = backend._raw_pcmd(
            11,
            -12,
            13,
            -14,
            reason="offline_known_enqueue",
            auto_run_id="sdk-known-run",
            command_set_mono_ns=1_999_999_900,
        )
    assert returned == 2_000_000_000
    assert calls == [(1, 11, -12, 13, -14, 0)]
    session.close(reason="offline_sdk_boundary")
    row = _rows(session.directory, "trajectory")[0]
    assert row["event"] == "pcmd_dispatch" and row["pcmd"] == [11, -12, 13, -14]
    assert row["auto_run_id"] == "sdk-known-run"
    assert row["command_set_mono_ns"] == 1_999_999_900
    assert row["command_mono_ns"] == row["sdk_enqueue_start_mono_ns"] == 2_000_000_000
    assert row["sdk_enqueue_return_mono_ns"] == 2_000_000_025
    assert row["reason"] == "offline_known_enqueue"
    assert row["evidence_kind"] == "sdk_enqueue_not_actuation_confirmation"
    enforce_retention(root, current_session=None, max_bytes=0)
    assert _rows(session.directory, "trajectory")[0] == row
    return {
        "session": str(session.directory),
        "auto_run_id": row["auto_run_id"],
        "pcmd": row["pcmd"],
        "transport": "in_memory_no_aircraft",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dedicated offline artifact directory; retention applies only below this directory",
    )
    args = parser.parse_args()
    controller = verify_controller_persistence(args.output_root / "controller")
    sdk = verify_sdk_enqueue_persistence(args.output_root / "sdk_boundary")
    failures = verify_failure_scenarios(args.output_root / "failures")
    print(
        json.dumps(
            {
                "offline_only": True,
                "controller": controller,
                "sdk_boundary": sdk,
                "failure_scenarios": failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
