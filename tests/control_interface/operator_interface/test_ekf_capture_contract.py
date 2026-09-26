"""Automatic capture and offline report pipeline, with no aircraft connection."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from backend_contract import InterfaceMode
from ekf_capture_contract import write_capture_contract
from imu_flight_test import create_recorder, close_recorder
import imu_flight_test
import imu_flight_test_report as report
from localization_metrics import attach_fused_localization_telemetry
from session_logs import SessionLogs


def _intrinsics():
    return {"image_width": 16, "image_height": 8,
            "K": [[10., 0., 8.], [0., 11., 4.], [0., 0., 1.]],
            "dist": [.01, .02, .03, .04, .05]}


def test_capture_snapshots_calibration_and_records_the_camera_center_approximation(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    original = json.dumps(_intrinsics()).encode()
    (root / "intrinsics.json").write_bytes(original)
    profile = root / "site.json"
    profile.write_text(json.dumps({"query_camera": {"model": "PINHOLE", "width": 16,
                       "height": 8, "params": [10., 11., 8., 4.]}}))
    session = tmp_path / "session"
    session.mkdir()
    (session / "session_manifest.json").write_text(json.dumps({"site_profile": str(profile)}))
    contract = write_capture_contract(session / "imu_test", root=root)
    assert contract["intrinsics_match_active_pinhole"]
    assert contract["intrinsics"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert contract["camera_body_translation_m"] == [0., 0., 0.]
    assert not contract["distortion_domain_verified"]
    assert not contract["ekf_running"]
    (root / "intrinsics.json").write_text("{}")
    assert (session / "imu_test/intrinsics.json").read_bytes() == original


def test_automatic_real_session_submissions_close_into_a_replay_report(tmp_path, monkeypatch):
    from flight_operator_app import OperatorApp
    from disk_policy import enforce_retention

    monkeypatch.delenv("SFM_IMU_FLIGHT_TEST", raising=False)
    calibration = json.loads((Path(__file__).resolve().parents[3] / "intrinsics.json").read_text())
    k = calibration["K"]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"query_camera": {"model": "PINHOLE", "width": 1280,
                       "height": 720, "params": [k[0][0], k[1][1], k[0][2], k[1][2]]}}))
    session = SessionLogs.create(tmp_path / "logs", mode=InterfaceMode.REAL_FLIGHT,
                                 manifest={"site_profile": str(profile)})
    recorder = create_recorder(session.directory, mode=session.mode)
    children = []
    popen = subprocess.Popen

    def start(args, **kwargs):
        process = popen(args, **kwargs)
        if (
            isinstance(args, (list, tuple))
            and len(args) > 1
            and Path(args[1]).name == "imu_flight_test_report.py"
        ):
            children.append(process)
        return process

    monkeypatch.setattr(imu_flight_test.subprocess, "Popen", start)
    app = SimpleNamespace(imu_flight_test=recorder, _submit_ok=0,
                          localizer=SimpleNamespace(submit=lambda *a, **kw: True),
                          video_frame=np.zeros((720, 1280, 3), dtype=np.uint8))
    session.localization("auto_route_plan", auto_run_id="test", waypoints_u=[[0, 0, 0], [1, 0, 0]])
    start_time = time.monotonic() - 10.
    for index in range(3):
        capture = start_time + index * .2
        stamp = int((capture - .02) * 1e9)
        state = SimpleNamespace(attitude_mono_ns=stamp, ground_speed_mono_ns=stamp,
                                altitude_mono_ns=stamp, att_roll=0., att_pitch=0., att_yaw=.2 * index,
                                speed_north_mps=.5, speed_east_mps=0., speed_down_mps=0.,
                                drone_altitude_m=1., gimbal_pitch_deg=-20., zoom=1.)
        timing = {"source_frame_stamp_mono": capture}
        attach_fused_localization_telemetry(timing, state)
        app.video_display_index = index
        OperatorApp._submit_localization_payload(
            app, (f"frame{index}", app.video_frame.tobytes(), timing), was_busy=False, lost_retry=False)
        session.localization("pose_result", display_seq=index, success=True, next_mode="TRACK",
                             pose={"x": index * .01, "y": 0., "z": 0.}, **timing)
        session.telemetry("fused_odometry", attitude_mono_ns=stamp, speed_mono_ns=stamp,
                          altitude_mono_ns=stamp, att_roll=0., att_pitch=0., att_yaw=.2 * index,
                          speed_north_mps=.5, speed_east_mps=0., speed_down_mps=0., drone_altitude_m=1.)
    session.close(reason="test")
    close_recorder(recorder)
    assert len(children) == 1 and children[0].wait(timeout=15) == 0
    output = json.loads((recorder.directory / "report.json").read_text())
    assert output["frames"]["indexed"] == 3
    assert output["localization"]["frames"] == 3  # The AUTO plan is not a pose.
    assert output["ekf_capture"]["image_replay_ready"]
    assert output["ekf_capture"]["matched_recorded_frames"] == 3
    assert not output["ekf_capture"]["formal_control_ready"]
    enforce_retention(tmp_path / "logs", current_session=None, max_bytes=0)
    assert report.build_report(session.directory)["ekf_capture"]["image_replay_ready"]


def test_source_timestamps_are_counted_independently_of_repeated_polls(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    samples = [{"event": "fused_odometry", "attitude_mono_ns": 1_000_000_000,
                "speed_mono_ns": 900_000_000, "altitude_mono_ns": 800_000_000}] * 50
    (session / "telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    (session / "localization.jsonl").write_text('{"event":"pose_result"}\n{broken\n')
    result = report.build_report(session)
    assert result["ekf_capture"]["source_stamps"]["speed"]["distinct_samples"] == 1
    assert not result["ekf_capture"]["pose_telemetry_replay_ready"]
    assert result["integrity_errors"] == ["localization.jsonl:2: invalid JSON"]


def test_rtsp_frames_expose_degraded_receipt_clock_not_sensor_time():
    from rtsp_frame_source import RtspFrameGrabber

    stream = RtspFrameGrabber.__new__(RtspFrameGrabber)
    stream._lock = threading.Lock()
    stream._latest, stream._digest, stream._n = None, None, 0
    stream._stamp = stream._receipt_stamp = 0.0
    stream._frames_received = stream._dup_n = 0
    stream._latest_timing = {}
    now = time.monotonic() - .1
    stream._store(
        np.zeros((8, 16, 3), dtype=np.uint8),
        receipt_stamp=now,
        receipt_mono_ns=round(now * 1e9),
    )
    timing = stream._latest_timing
    assert timing["source_timestamp_trusted"] is False
    assert timing["source_timestamp_degraded"] is True
    assert timing["source_ntp_timestamp_us"] is None
    assert timing["frame_receipt_mono_ns"] == round(now * 1e9)

def test_gimbal_records_all_axes_without_refreshing_an_unchanged_cache(tmp_path, monkeypatch):
    from arsdk_ng_live_backend import ArsdkNgLiveBackend
    import backend_telemetry
    from operator_flight_safety import TelemetryFreshnessStore
    from operator_state import DroneState

    monkeypatch.setattr(backend_telemetry, "arsdk_message", lambda _: object())
    backend = ArsdkNgLiveBackend.__new__(ArsdkNgLiveBackend)
    backend.state = DroneState()
    backend.drone = SimpleNamespace()
    backend.telemetry_freshness = TelemetryFreshnessStore()
    backend._poll_get_state = lambda _: {"gimbal_id": 0, "pitch_absolute": -20.,
        "yaw_absolute": 2., "roll_absolute": 1., "pitch_relative": -21.,
        "yaw_relative": 3., "roll_relative": 4.}
    backend._poll_gimbal()
    stamp = backend.state.gimbal_mono_ns
    assert stamp is not None
    backend._poll_gimbal()
    assert backend.state.gimbal_mono_ns == stamp
    assert backend.state.gimbal_relative_yaw_deg == 3.
    assert backend.state.gimbal_roll_deg == 1.


def test_zoom_readback_is_logged_without_changing_operator_control_state(monkeypatch):
    from arsdk_ng_live_backend import ArsdkNgLiveBackend
    import backend_telemetry
    from operator_flight_safety import TelemetryFreshnessStore
    from operator_state import DroneState

    monkeypatch.setattr(backend_telemetry, "arsdk_message", lambda _: object())
    backend = ArsdkNgLiveBackend.__new__(ArsdkNgLiveBackend)
    backend.state = DroneState()
    backend.drone = SimpleNamespace()
    backend.telemetry_freshness = TelemetryFreshnessStore()
    backend._poll_get_state = lambda _: {"cam_id": 0, "level": 1.5}
    backend._poll_camera_zoom()
    stamp = backend.state.camera_zoom_mono_ns
    backend._poll_camera_zoom()
    assert backend.state.camera_zoom_mono_ns == stamp
    assert backend.state.camera_zoom_level == 1.5
    assert backend.state.zoom == 1.0
