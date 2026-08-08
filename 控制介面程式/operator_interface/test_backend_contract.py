from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from backend_contract import (
    ControlAction,
    ControlRequest,
    ControlResult,
    FailureReason,
    InterfaceMode,
    InvalidControlRequest,
    LegacyFrameSourceAdapter,
    MissionRoutePayload,
    SessionConfig,
)
from flight_operator_app import DroneBackend


def test_session_config_is_immutable_and_has_one_fixed_interface() -> None:
    config = SessionConfig(
        session_id="session-1",
        interface_mode=InterfaceMode.SIMULATED_STREAM,
        site_profile="/tmp/site.json",
        site_profile_sha256="a" * 64,
        asset_sha256={"map": "b" * 64},
        runtime_profile_sha256="c" * 64,
        source="/tmp/video.mp4",
        offline=True,
        site_profile_schema_version=2,
        autonomous_speed_limit_mps=0.30,
        autonomous_locked=True,
    )

    assert config.site_profile_schema_version == 2
    assert config.autonomous_speed_limit_mps == pytest.approx(0.30)
    assert config.autonomous_locked is True
    backend = DroneBackend()
    assert callable(backend.start)
    assert backend.start(config).started
    with pytest.raises(FrozenInstanceError):
        config.interface_mode = InterfaceMode.REAL_FLIGHT  # type: ignore[misc]


def test_unknown_legacy_action_is_rejected_before_backend_dispatch() -> None:
    with pytest.raises(InvalidControlRequest, match="unknown control action"):
        ControlRequest.from_legacy("arbitrary-unchecked-command", human_origin=True)


def test_start_auto_requires_complete_immutable_route_identity(tmp_path) -> None:
    route = tmp_path / "route.json"
    route.write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidControlRequest, match="start_auto requires"):
        ControlRequest.from_legacy("start_auto", human_origin=True)

    request = ControlRequest.from_legacy(
        "start_auto",
        human_origin=True,
        route_path=str(route.resolve()),
        route_sha256="a" * 64,
        site_id="field-a",
        coordinate_frame_id="glomap-a",
    )

    assert isinstance(request.payload, MissionRoutePayload)
    assert request.legacy_call() == (
        "start_auto",
        {
            "route_path": str(route.resolve()),
            "route_sha256": "a" * 64,
            "site_id": "field-a",
            "coordinate_frame_id": "glomap-a",
        },
    )


@pytest.mark.parametrize(
    ("route_path", "route_sha256"),
    [("relative.json", "a" * 64), ("/tmp/route.json", "not-a-sha")],
)
def test_start_auto_rejects_ambiguous_route_identity(
    route_path, route_sha256
) -> None:
    with pytest.raises(InvalidControlRequest):
        MissionRoutePayload(
            route_path=route_path,
            route_sha256=route_sha256,
            site_id="field-a",
            coordinate_frame_id="glomap-a",
        )


def test_takeoff_requires_human_origin_even_for_simulation() -> None:
    backend = DroneBackend()
    request = ControlRequest.create(ControlAction.TAKEOFF, human_origin=False)

    result = backend.command(request)

    assert isinstance(result, ControlResult)
    assert not result.accepted
    assert result.reason_code == "HUMAN_ORIGIN_REQUIRED"
    assert backend.state.tracker_state == "HOVER"


@pytest.mark.parametrize(
    ("name", "action"),
    [
        ("drone_magnetometer_start", ControlAction.DRONE_MAGNETOMETER_START),
        ("drone_magnetometer_cancel", ControlAction.DRONE_MAGNETOMETER_CANCEL),
        (
            "skycontroller_magnetometer_start",
            ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
        ),
        (
            "skycontroller_magnetometer_cancel",
            ControlAction.SKYCONTROLLER_MAGNETOMETER_CANCEL,
        ),
    ],
)
def test_magnetometer_actions_are_typed_no_payload_controls(name, action) -> None:
    request = ControlRequest.from_legacy(name, human_origin=True)

    assert request.action is action


def test_simulation_rejects_hardware_magnetometer_calibration() -> None:
    backend = DroneBackend()

    automated = backend.command(ControlRequest.create(
        ControlAction.DRONE_MAGNETOMETER_START,
        human_origin=False,
    ))
    requested = backend.command(ControlRequest.create(
        ControlAction.DRONE_MAGNETOMETER_START,
        human_origin=True,
    ))

    assert automated.reason_code == "HUMAN_ORIGIN_REQUIRED"
    assert requested.reason_code == "LIVE_HARDWARE_REQUIRED"
    assert backend.state.tracker_state == "HOVER"


def test_typed_sim_takeoff_changes_only_simulated_state() -> None:
    backend = DroneBackend()
    request = ControlRequest.create(ControlAction.TAKEOFF, human_origin=True)

    result = backend.command(request)

    assert result.accepted and result.executed
    assert result.resulting_state is backend.state
    assert backend.mode is InterfaceMode.SIMULATED_STREAM
    assert backend.is_live is False
    assert backend.state.tracker_state == "TAKEOFF"


def test_sim_fail_safe_clears_motion_and_never_auto_resumes() -> None:
    backend = DroneBackend()
    backend.state.mode = "AUTO"
    backend.command("nudge_begin", dir="前")

    result = backend.fail_safe(FailureReason.LOCALIZATION_LOST)

    assert result.accepted and result.executed
    assert backend.state.mode == "MANUAL"
    assert backend.state.tracker_state == "FAIL_SAFE_HOVER"
    assert backend.state.active_incident == FailureReason.LOCALIZATION_LOST.value


def test_typed_sim_speed_limit_change_is_landed_only_and_invalidates_approval() -> None:
    backend = DroneBackend()
    backend.state.autonomous_approval_valid = True
    request = ControlRequest.from_legacy(
        "auto_speed_limit_apply",
        human_origin=True,
        speed_limit_mps=0.2,
    )

    result = backend.command(request)

    assert result.accepted and result.executed
    assert backend.state.autonomous_speed_limit_mps == pytest.approx(0.2)
    assert backend.state.autonomous_approval_valid is False

    backend.target_altitude_m = 1.0
    rejected = backend.command(
        ControlRequest.from_legacy(
            "auto_speed_limit_apply",
            human_origin=True,
            speed_limit_mps=0.1,
        )
    )
    assert not rejected.accepted
    assert backend.state.autonomous_speed_limit_mps == pytest.approx(0.2)


def test_legacy_frame_source_adapter_emits_typed_packet() -> None:
    class Source:
        output_index = 7
        last_stamp = 12.5
        last_timing = {"decode_ms": 2.0}
        eof = False

        def next_frame(self, *, only_new=True):
            assert only_new
            return "rgb-frame"

        def close(self):
            pass

    packet = LegacyFrameSourceAdapter(Source(), "test-source").next_frame()

    assert packet is not None
    assert packet.rgb == "rgb-frame"
    assert packet.sequence == 7
    assert packet.source_timestamp_ns == 12_500_000_000
    assert packet.source_identity == "test-source"
    assert packet.timing["decode_ms"] == 2.0
