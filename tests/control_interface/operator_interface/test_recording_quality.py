from __future__ import annotations

import pytest

from backend_contract import (
    ControlAction,
    ControlRequest,
    InvalidControlRequest,
    RecordingQualityPayload,
)
from flight_operator_app import DroneBackend
from recording_quality import (
    DEFAULT_RECORDING_PROFILE,
    FHD_30,
    UHD_4K_30,
    format_record_status,
    recording_mode_matches,
    recording_profile_by_label,
    resolve_recording_profile,
)


def test_only_sixteen_by_nine_profiles_are_listed() -> None:
    assert resolve_recording_profile("fhd_30") is FHD_30
    assert resolve_recording_profile("uhd_4k_30") is UHD_4K_30
    assert DEFAULT_RECORDING_PROFILE is FHD_30
    assert FHD_30.width / FHD_30.height == pytest.approx(16 / 9)
    assert UHD_4K_30.width / UHD_4K_30.height == pytest.approx(16 / 9)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_recording_profile("res_2_7k")
    with pytest.raises(ValueError, match="unknown"):
        recording_profile_by_label("2.7K")


def test_recording_mode_readback_accepts_enum_or_short_names() -> None:
    assert recording_mode_matches(
        {"resolution": "res_uhd_4k", "framerate": "fps_30"}, UHD_4K_30,
    ) is True
    assert recording_mode_matches(
        {"resolution": "uhd_4k", "framerate": "30"}, UHD_4K_30,
    ) is True
    assert recording_mode_matches(
        {
            0: {
                "resolution": type("E", (), {"name": "resolution.res_uhd_4k"})(),
                "framerate": type("E", (), {"name": "framerate.fps_30"})(),
            }
        },
        UHD_4K_30,
    ) is True
    assert recording_mode_matches(
        {
            0: {
                "resolution": type("E", (), {"name": "resolution.res_1080p"})(),
                "framerate": type("E", (), {"name": "framerate.fps_30"})(),
            }
        },
        UHD_4K_30,
    ) is False
    assert recording_mode_matches({}, UHD_4K_30) is None


def test_status_text_includes_selected_profile() -> None:
    assert "FHD 1080p30" in format_record_status(
        active=False, armed=True, profile=FHD_30,
    )
    assert "4K UHD 30" in format_record_status(
        active=True, armed=True, profile=UHD_4K_30,
    )


def test_record_quality_request_round_trips() -> None:
    request = ControlRequest.from_legacy(
        "record_quality", human_origin=True, profile_id="uhd_4k_30",
    )
    assert request.action is ControlAction.RECORD_QUALITY
    assert isinstance(request.payload, RecordingQualityPayload)
    assert request.legacy_call() == (
        "record_quality",
        {"profile_id": "uhd_4k_30"},
    )
    with pytest.raises(InvalidControlRequest, match="unsupported"):
        ControlRequest.from_legacy(
            "record_quality", human_origin=True, profile_id="res_dci_4k",
        )


def test_sim_backend_stores_quality_and_refuses_change_while_recording() -> None:
    backend = DroneBackend()
    assert backend.recording_profile is FHD_30
    assert backend.command("record_quality", profile_id="uhd_4k_30") is backend.state
    assert backend.recording_profile is UHD_4K_30
    assert "4K UHD 30" in backend.record_status
    backend.command("record_start")
    assert backend.recording_active is True
    assert backend.command("record_quality", profile_id="fhd_30") is False
    assert backend.recording_profile is UHD_4K_30
