"""Offline ARSDK event/history and control-consumer contracts."""

from dataclasses import replace
from types import SimpleNamespace
import threading

import numpy as np
import pytest

import arsdk_ng as sdk
import backend_telemetry as telemetry
import path_follow_flight as pff
import real_path_follow_controller as rpf


def test_motion_events_between_polls_are_retained_with_receipt_timestamps(monkeypatch):
    spec = SimpleNamespace(
        id=1, fullname="Ardrone3.PilotingState.SpeedChanged", list_type=sdk._LIST_NONE
    )
    client = sdk.ArsdkNgClient.__new__(sdk.ArsdkNgClient)
    client._lock = threading.RLock()
    client._condition = threading.Condition(client._lock)
    client._events, client._states, client._map_states = {}, {}, {}
    client._event_history = {}
    client._event_serial = 0
    client._runtime = SimpleNamespace(_decode=lambda command: (spec, {"speedX": command}))
    for index in range(520):
        monkeypatch.setattr(sdk.time, "monotonic_ns", lambda: 1_000_000_000 + index * 10_000_000)
        client._on_recv_cmd(None, index, None)
    history = client.get_event_history(spec)
    assert len(history) == 512
    assert history[0].args["speedX"] == 8
    assert history[-1].monotonic_ns == 6_190_000_000
    assert client.get_last_event(spec).serial == history[-1].serial
    history.clear()
    assert len(client.get_event_history(spec)) == 512
    spec = SimpleNamespace(
        id=2, fullname="Common.CommonState.BatteryStateChanged", list_type=sdk._LIST_NONE
    )
    client._on_recv_cmd(None, 100, None)
    assert client.get_event_history(spec) == []


def test_backend_history_preserves_receipt_time_and_rejects_invalid_samples(monkeypatch):
    monkeypatch.setattr(telemetry, "arsdk_message", lambda name: name)
    histories = {
        "AttitudeChanged": [sdk._Event({"yaw": 0.1}, 9_800_000_000, 1)],
        "SpeedChanged": [
            sdk._Event({"speedX": 1.0, "speedY": 2.0}, 9_900_000_000, 2),
            sdk._Event({"speedX": float("nan"), "speedY": 2.0}, 9_950_000_000, 3),
        ],
    }
    backend = SimpleNamespace(
        drone=SimpleNamespace(
            get_event_history=lambda message: histories[message.rsplit(".", 1)[-1]]
        )
    )
    attitude, velocity = telemetry.control_telemetry_history(backend)
    assert attitude == [(0.1, 9.8)]
    assert velocity == [(1.0, 2.0, 9.9)]


def test_control_loop_consumes_intermediate_events_once():
    ctrl = rpf.RouteAutoController(
        [np.zeros(3), np.ones(3)], config=rpf.ControlConfig(inspect_waypoints=())
    )
    hooks = pff.LoopHooks(get_pose=lambda: None, olympe_yaw=lambda: 0.0, send_pcmd=lambda *_: None)
    hooks = replace(
        hooks,
        telemetry_history=lambda: (
            [(0.0, 1.0), (0.2, 1.1), (0.4, 1.2)],
            [(0.0, 0.0, 1.0), (0.0, 2.0, 1.1), (0.0, 0.0, 1.2)],
        ),
    )
    runner = pff._FlightLoopRunner(
        hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False, enforce_weak_pose_gate=False
    )
    runner._observe_control_telemetry(1.25)
    runner._observe_control_telemetry(1.3)
    assert len(runner.control_state.yaws) == len(runner.control_state.velocities) == 3
    assert runner.control_state.yaw_at(1.1) == pytest.approx(0.2)
    np.testing.assert_allclose(runner.control_state.displacement(1.0, 1.2), [0.0, 0.2])


def test_short_bridge_does_not_refresh_anchor_or_invent_landing_evidence():
    estimator = pff.ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    estimator.scale_m_per_unit, estimator.scale_stamp = 10.0, 10.0
    observed = rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=10.0)
    estimator.observe_pose(observed, 10.0, 0.0)
    for stamp in (10.0, 10.1, 10.2, 10.3):
        estimator.observe_telemetry((0.0, stamp), (0.0, 1.0, stamp), stamp)
    bridge = replace(observed, stamp=10.3, map_confirmed=False, position_observed=False)
    estimator.observe_pose(bridge, 10.3, 0.0)
    xyz, info = estimator.predict(bridge, 10.3, 0.0, 0.1)
    assert info["applied"] and xyz[0] == pytest.approx(0.03)
    assert estimator.last_observed[0] == 10.0
    assert not bridge.map_confirmed and not bridge.position_observed
    assert not estimator.predict(replace(bridge, stamp=10.6), 10.6, 0.0, 0.1)[1]["applied"]


def test_motion_log_records_each_event_once_without_advancing_sample_clocks(monkeypatch):
    rows = []
    backend = SimpleNamespace(
        session_logs=SimpleNamespace(telemetry=lambda event, **fields: rows.append((event, fields)))
    )
    histories = ([(0.1, 10.0), (0.2, 10.1)], [(0.0, 1.0, 10.0), (0.0, 2.0, 10.2)])
    monkeypatch.setattr(telemetry, "control_telemetry_history", lambda _: histories)
    telemetry._log_control_history(backend, 10.1)
    telemetry._log_control_history(backend, 10.15)
    telemetry._log_control_history(backend, 10.2)
    assert len(rows) == 2
    assert rows[0][1]["timestamp_kind"] == "host_monotonic_receipt"
    assert rows[0][1]["attitude_samples"] == histories[0]
    assert rows[1][1]["attitude_samples"] == []
    assert rows[1][1]["horizontal_ned_samples"] == [(0.0, 2.0, 10.2)]


def test_video_connection_passes_connected_client_to_real_grabber_constructor(monkeypatch):
    import arsdk_ng_live_backend as backend_module
    import rtsp_frame_source as rtsp

    backend = backend_module.ArsdkNgLiveBackend.__new__(backend_module.ArsdkNgLiveBackend)
    backend.ip, backend.controller, backend.with_video = "192.168.42.1", "drone", True
    backend.state = SimpleNamespace()
    backend.log = SimpleNamespace(event=lambda *_args, **_kwargs: None)
    client = SimpleNamespace(address=backend.ip)
    monkeypatch.setattr(rtsp, "connect", lambda *_: client)
    monkeypatch.setattr(rtsp.RtspFrameGrabber, "start", lambda self: self)
    monkeypatch.setattr(backend_module, "quiet_arsdk_logs", lambda: None)
    monkeypatch.setattr(backend_module, "LiveAnafiVideoStream", lambda grabber: grabber)
    monkeypatch.setattr(backend_module, "LegacyFrameSourceAdapter", lambda stream, _url: stream)
    monkeypatch.setattr(backend, "_set_piloting_source", lambda _: True)
    monkeypatch.setattr(backend, "_set_tracker_state", lambda *_args, **_kwargs: None)
    for name in (
        "_configure_lost_link_policy_if_safe",
        "read_connection_inventory",
        "_read_magnetometer_calibration_state",
        "_configure_firmware_limits_if_safe",
        "_auto_take_pc_control_if_safe",
    ):
        monkeypatch.setattr(backend, name, lambda: None)
    backend._connect()
    assert backend.state.stream == "OK"
    assert backend.grabber.client is client
