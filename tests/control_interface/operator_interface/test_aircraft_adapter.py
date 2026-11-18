from __future__ import annotations

import builtins
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import aircraft_adapter as adapter
from backend_contract import (
    AircraftCapabilities, ControlAction, ControlAdmission, ControlRequest, InterfaceMode,
    OperatorBackend, RuntimeSwitchKind, ScalarPayload, StreamOwner,
    ControlResult, execute_admitted_control,
)
from operator_state import ANAFI, AircraftProfile, DroneState
from simulated_backend import DroneBackend


def _profile():
    return AircraftProfile(
        model="Bench aircraft", aircraft_id="bench-aircraft", controller_name="Bench RC",
        stream_width=640, stream_height=480, stream_fps=24.0,
        gimbal_pitch_min_deg=-35.0, gimbal_pitch_max_deg=15.0,
        digital_zoom_min=1.0, digital_zoom_max=1.5,
        default_gimbal_pitch_deg=-5.0, default_zoom=1.0,
        max_horizontal_speed_mps=1.0, max_vertical_speed_mps=0.5,
        max_yaw_rate_dps=30.0, flight_time_s=600.0,
    )


def _build_memory(context):
    return DroneBackend(
        aircraft_profile=context.profile, capabilities=context.capabilities,
        session_logs=context.session_logs,
    )


def _definition(**kwargs):
    return adapter.AircraftDefinition(
        profile=_profile(), simulation_factory=_build_memory,
        simulation_capabilities=AircraftCapabilities(frozenset({
            ControlAction.GIMBAL_PITCH, ControlAction.ZOOM, ControlAction.START_LOCALIZATION,
        })), **kwargs,
    )


def _args(**kwargs):
    return SimpleNamespace(**{"live": False, "video": "", "video_stride": 1, **kwargs})


def _camera_request(action, value):
    return ControlRequest.create(action, payload=ScalarPayload(value), human_origin=True)


def test_custom_aircraft_offline_composition_uses_its_envelope_without_sdk(monkeypatch):
    real_import = builtins.__import__

    def forbid_sdk(name, *args, **kwargs):
        if name.startswith("olympe"):
            raise AssertionError(f"offline composition imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_sdk)
    definition = _definition()
    backend, stream, live = adapter.build_aircraft_runtime(
        _args(), None, None, definition=definition,
    )

    assert isinstance(backend, OperatorBackend)
    assert backend.aircraft_profile is definition.profile
    assert backend.connection_target == "offline:bench-aircraft"
    assert backend.state.aircraft_identity == "SIMULATED Bench aircraft"
    assert backend.state.gimbal_pitch_deg == -5.0
    assert stream is None and live is None
    assert backend.command(_camera_request(ControlAction.GIMBAL_PITCH, -80.0)).accepted
    assert backend.state.gimbal_pitch_deg == -35.0
    assert backend.command(_camera_request(ControlAction.ZOOM, 3.0)).accepted
    assert backend.state.zoom == 1.5
    assert backend.runtime_switch_status().kind is RuntimeSwitchKind.NEVER_CONNECTED


def test_undeclared_action_is_rejected_without_mutation():
    backend, _, _ = adapter.build_aircraft_runtime(_args(), None, None, definition=_definition())
    before = backend.state.tracker_state
    result = backend.command(ControlRequest.create(ControlAction.HOVER, human_origin=True))
    assert not result.accepted and not result.executed
    assert result.reason_code == "UNSUPPORTED_ACTION"
    assert backend.state.tracker_state == before


@pytest.mark.parametrize("capabilities", [None, AircraftCapabilities()])
def test_missing_capabilities_never_inherit_anafi_support(capabilities):
    backend = DroneBackend(aircraft_profile=_profile(), capabilities=capabilities)
    result = backend.command(_camera_request(ControlAction.GIMBAL_PITCH, 0.0))
    assert result.reason_code == "UNSUPPORTED_ACTION"
    assert backend.state.gimbal_pitch_deg == -5.0


def test_claimed_camera_action_with_unknown_bounds_fails_closed():
    profile = replace(_profile(), gimbal_pitch_min_deg=None)
    backend = DroneBackend(
        aircraft_profile=profile,
        capabilities=AircraftCapabilities(frozenset({ControlAction.GIMBAL_PITCH})),
    )
    result = backend.command(_camera_request(ControlAction.GIMBAL_PITCH, 0.0))
    assert result.reason_code == "CAMERA_PITCH_BOUNDS_UNKNOWN"
    assert backend.state.gimbal_pitch_deg == -5.0


@pytest.mark.parametrize("mismatch", ["model", "mode", "ownership", "capability"])
def test_factory_rejects_mismatched_model_mode_and_source_ownership(mismatch):
    definition = _definition()

    def mismatched(context):
        backend = _build_memory(context)
        if mismatch == "model":
            backend.aircraft_profile = ANAFI
        elif mismatch == "mode":
            backend.mode = InterfaceMode.REAL_FLIGHT
        elif mismatch == "ownership":
            backend.stream_owner = StreamOwner.BACKEND
        else:
            backend.capabilities = adapter.ANAFI_LIVE_CAPABILITIES
        return backend

    definition = replace(definition, simulation_factory=mismatched)
    with pytest.raises(ValueError, match="mismatch"):
        adapter.build_aircraft_runtime(_args(), None, None, definition=definition)


def test_live_source_cannot_be_replaced_with_offline_video_before_factory():
    factory = Mock(side_effect=AssertionError("factory should not run"))
    definition = _definition(live_factory=factory)
    with pytest.raises(ValueError, match="offline video"):
        adapter.build_aircraft_runtime(
            _args(live=True, video="recording.mp4"), None, None, definition=definition,
        )
    factory.assert_not_called()


def test_unsupported_live_mode_is_not_silently_simulated():
    with pytest.raises(ValueError, match="does not support"):
        adapter.build_aircraft_runtime(_args(live=True), None, None, definition=_definition())


def test_offline_display_and_typed_port_share_one_owned_source(monkeypatch, tmp_path):
    import ffmpeg_frame_stream

    stream = SimpleNamespace(next_frame=Mock(return_value=None), close=Mock())
    factory = Mock(return_value=stream)
    monkeypatch.setattr(ffmpeg_frame_stream, "FFmpegFrameStream", factory)
    path = tmp_path / "recording.mp4"
    definition = _definition()
    backend, display, live = adapter.build_aircraft_runtime(
        _args(video=str(path)), None, None, definition=definition,
    )
    assert display is stream and live is None
    assert backend.video.source is stream
    assert backend.stream_owner is StreamOwner.RUNTIME
    assert backend.video.source_identity == str(path.resolve())
    factory.assert_called_once_with(path, 640, 480, stride=1, fps=24.0, link_sim=None)


def test_current_live_factory_preserves_settings_and_approved_versions(monkeypatch, tmp_path):
    import olympe_live_backend

    constructor = Mock(return_value=object())
    monkeypatch.setattr(olympe_live_backend, "OlympeLiveBackend", constructor)
    settings = {
        "ip": "192.168.53.1", "controller": "skycontroller3", "nudge_pct": 8,
        "nudge_pulse_s": 0.2, "max_altitude_m": 10.0, "max_distance_m": 50.0,
        "distance_geofence": True, "min_takeoff_battery_pct": 30.0,
        "max_tilt_deg": 10.0, "max_vertical_speed_ms": 0.5,
        "max_rotation_speed_degs": 10.0, "max_pitch_roll_rotation_speed_degs": 200.0,
        "rth_min_altitude_m": 10.0, "auto_pc_control": True,
        "stream_loss_grace_s": 3.0, "require_gps_for_geofence": True,
    }
    args = _args(live=True, no_live_video=False, **settings)
    logs = SimpleNamespace(directory=tmp_path, command_log=object())
    approval = SimpleNamespace(
        approved=True, aircraft_firmware_versions=("1.8.2",),
        controller_firmware_versions=("1.8.1",), olympe_versions=("8.4.0",),
    )
    context = adapter.AircraftBuildContext(
        args, logs, approval, ANAFI, adapter.ANAFI_LIVE_CAPABILITIES, InterfaceMode.REAL_FLIGHT,
    )

    adapter.CURRENT_AIRCRAFT.live_factory(context)

    constructor.assert_called_once_with(
        DroneState, ANAFI, **settings,
        approved_aircraft_firmware=("1.8.2",), approved_controller_firmware=("1.8.1",),
        approved_olympe_versions=("8.4.0",), cmd_log=tmp_path / "commands.jsonl",
        event_log=logs.command_log, session_logs=logs, with_video=True, defer_connect=True,
    )


def test_factory_errors_propagate_without_fake_simulation_fallback():
    factory = Mock(side_effect=RuntimeError("SDK construction rejected"))
    with pytest.raises(RuntimeError, match="SDK construction rejected"):
        adapter.build_aircraft_runtime(
            _args(live=True), None, None, definition=_definition(live_factory=factory),
        )


def test_simulated_camera_samples_require_observation_stamp():
    backend = DroneBackend()
    assert backend.gimbal_pitch_sample() is None
    backend.state.gimbal_mono_ns = 123
    backend.state.gimbal_stamp_source = "observed_cache_value"
    assert backend.gimbal_pitch_sample() is None
    backend.state.gimbal_stamp_source = "simulated"
    assert backend.gimbal_pitch_sample() == (-20.0, 123)


def test_cancelled_admission_never_mutates_simulator():
    backend = DroneBackend()
    admission = ControlAdmission()
    request = ControlRequest(
        ControlAction.GIMBAL_PITCH, ScalarPayload(15.0), human_origin=True, admission=admission,
    )
    admission.cancel()
    result = backend.command(request)
    assert result.reason_code == "CONTROL_REQUEST_CANCELLED"
    assert backend.state.gimbal_pitch_deg == -20.0


def test_cancel_between_dispatch_and_backend_entry_is_rejected(monkeypatch):
    backend = DroneBackend()
    admission = ControlAdmission()
    request = ControlRequest(
        ControlAction.GIMBAL_PITCH, ScalarPayload(15.0), human_origin=True, admission=admission,
    )
    entered, release = threading.Event(), threading.Event()
    execute = backend._typed_command

    def delayed_entry(request):
        entered.set()
        assert release.wait(5)
        return execute(request)

    monkeypatch.setattr(backend, "_typed_command", delayed_entry)
    results = []
    worker = threading.Thread(target=lambda: results.append(backend.command(request)))
    worker.start()
    try:
        assert entered.wait(5)
        admission.cancel()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results[0].reason_code == "CONTROL_REQUEST_CANCELLED"
    assert backend.state.gimbal_pitch_deg == -20.0


def test_late_cancel_preserves_evidence_of_already_executed_command():
    admission = ControlAdmission()
    request = ControlRequest(ControlAction.HOVER, human_origin=True, admission=admission)
    state = DroneState()

    def completed_then_cancelled(_request):
        admission.cancel()
        return ControlResult.completed(state, raw_result=True)

    result = execute_admitted_control(request, state, completed_then_cancelled)
    assert result.reason_code == "CONTROL_REQUEST_CANCELLED"
    assert not result.accepted
    assert result.executed
    assert result.ack["accepted"] is True
