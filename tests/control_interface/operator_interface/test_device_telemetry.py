"""Offline device telemetry: absent events, camera selection and operator units."""

import json
from types import SimpleNamespace

import backend_telemetry as telemetry
from operator_hud_text import format_device_telemetry
from operator_state import DroneState


def _backend(monkeypatch, states):
    monkeypatch.setattr(telemetry, "olympe_message", lambda module, name: (module, name))
    return SimpleNamespace(
        state=DroneState(),
        _poll_get_state=lambda message: states.get(message),
        via_skycontroller=lambda: True,
    )


def test_missing_camera_event_does_not_hide_other_devices_or_keep_old_values(monkeypatch):
    states = {
        ("battery", "voltage"): {"voltage": 7650},
        ("gimbal", "alert"): {0: {"gimbal_id": 0, "error": "overload"}},
    }
    backend = _backend(monkeypatch, states)
    telemetry.poll_device_info(backend, 10.0)
    assert backend.state.camera_info["exposure"] is None
    assert "7.65 V" in format_device_telemetry(backend.state)[0]
    assert "overload" in format_device_telemetry(backend.state)[-1]
    states.clear()
    telemetry.poll_device_info(backend, 11.0)
    lines = format_device_telemetry(backend.state)
    assert "未知 V" in lines[0]
    assert "錯誤 未知" in lines[-1]
    assert "overload" not in " ".join(lines)


def test_camera_and_gimbal_never_use_another_device_map_entry(monkeypatch):
    states = {
        ("camera", "exposure"): {
            1: {"cam_id": 1, "iso_sensitivity": "iso_3200"},
            0: {"cam_id": 0, "iso_sensitivity": "iso_100"},
        },
        ("gimbal", "alert"): {1: {"gimbal_id": 1, "error": "overload"}},
    }
    backend = _backend(monkeypatch, states)
    telemetry.poll_device_info(backend, 0.0)
    lines = format_device_telemetry(backend.state)
    assert "ISO 100" in lines[2]
    assert "3200" not in lines[2]
    assert "錯誤 未知" in lines[-1]
    states[("camera", "exposure")] = {"cam_id": 1, "iso_sensitivity": "iso_3200"}
    telemetry.poll_device_info(backend, 1.0)
    assert "ISO 未知" in format_device_telemetry(backend.state)[2]


def test_charge_sentinel_and_invalid_white_balance_temperature_are_not_measurements():
    state = DroneState(
        battery_info={
            "BatteryChanged": {"percent": 255},
            "alert": {"power_level": {"alert": "power_level", "level": "none"}},
        },
        camera_info={
            "white_balance": {"mode": "automatic", "temperature": "t_10000"},
            "recording_mode": {"bitrate": 0},
        },
    )
    lines = format_device_telemetry(state)
    assert "遙控器 充電中" in lines[0]
    assert "255%" not in lines[0]
    assert "電池警告 無" in lines[1]
    assert "10000" not in lines[2]
    assert "未知 Mb/s" in lines[3]
    state.battery_info["alert"] = None
    assert "電池警告 未知" in format_device_telemetry(state)[1]


def test_enum_alert_map_and_empty_bitfields_remain_json_safe(monkeypatch):
    from enum import Enum

    class Alert(Enum):
        too_hot = 1

    class Level(Enum):
        warning = 1

    class NoFlags:
        def __str__(self):
            return ""

    states = {
        ("battery", "alert"): {Alert.too_hot: {"alert": Alert.too_hot, "level": Level.warning}},
        ("gimbal", "alert"): {0: {"gimbal_id": 0, "error": NoFlags()}},
    }
    backend = _backend(monkeypatch, states)
    telemetry.poll_device_info(backend, 10.0)
    json.dumps(backend.state.battery_info)
    json.dumps(backend.state.gimbal_info)
    lines = format_device_telemetry(backend.state)
    assert "too_hot:warning" in lines[1]
    assert "錯誤 無" in lines[-1]
