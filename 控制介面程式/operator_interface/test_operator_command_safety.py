from __future__ import annotations

import inspect
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from flight_operator_app import (
    DroneState,
    OperatorApp,
    UI_LOG_MAX_LINES,
    _positive_env_float,
    _positive_env_int,
    format_magnetometer_calibration,
    format_olympe_telemetry,
    next_tick_deadline,
    video_hud_identity,
)


class _Backend:
    def __init__(self):
        self.calls = []

    def command(self, name, **payload):
        self.calls.append((name, payload))
        return name


def _bare_app(backend=None):
    app = OperatorApp.__new__(OperatorApp)
    app.backend = backend or _Backend()
    app._nudge_keys_held = set()
    app._nudge_buttons_held = set()
    app._nudge_key_map = {"w": "前"}
    app._flight_results = queue.Queue()
    app._flight_inflight = set()
    app._flight_inflight_lock = threading.Lock()
    app.write_log = lambda _message: None
    return app


def test_tick_deadline_does_not_accumulate_callback_runtime():
    deadline, delay = next_tick_deadline(1.0, 1.006, 0.010)
    assert deadline == pytest.approx(1.010)
    assert delay == 4

    deadline, delay = next_tick_deadline(deadline, 1.035, 0.010)
    assert deadline == pytest.approx(1.040)
    assert delay == 5


@pytest.mark.parametrize(
    ("reader", "value"),
    [
        (_positive_env_float, "nan"),
        (_positive_env_float, "0"),
        (_positive_env_int, "1.5"),
        (_positive_env_int, "-1"),
    ],
)
def test_safety_display_environment_rejects_invalid_values(
    monkeypatch, reader, value
) -> None:
    monkeypatch.setenv("SFM_TEST_VALUE", value)
    with pytest.raises(ValueError):
        reader("SFM_TEST_VALUE", 1)


def test_no_localization_markers_are_bounded() -> None:
    app = _bare_app()
    app.no_loc_markers = deque(maxlen=3)
    for index in range(5):
        app.live_last_xyz = np.array([float(index) * 2.0, 0.0, 0.0])
        app._record_no_loc()

    assert len(app.no_loc_markers) == 3
    assert np.allclose(app.no_loc_markers[0], [4.0, 0.0, 0.0])


def test_ui_log_is_trimmed_to_a_bounded_number_of_lines() -> None:
    class FakeLog:
        def __init__(self):
            self.insert_calls = []
            self.delete_calls = []

        def insert(self, *args):
            self.insert_calls.append(args)

        def delete(self, *args):
            self.delete_calls.append(args)

    app = OperatorApp.__new__(OperatorApp)
    app.log = FakeLog()

    app.write_log("bounded")

    assert app.log.insert_calls
    assert app.log.delete_calls == [(f"{UI_LOG_MAX_LINES + 1}.0", "end")]


def test_ui_starts_localization_without_exposing_fake_autonomy() -> None:
    source = inspect.getsource(OperatorApp._build_ui)
    assert 'text="開始定位"' in source
    assert "command=self.begin_auto_inspect" in source
    assert '"start_auto"' not in source
    assert 'text="自主巡檢未接入此介面"' in source


def test_operator_ui_cannot_force_megaloc_outside_boot_or_lost() -> None:
    source = inspect.getsource(OperatorApp._build_ui)

    assert '"BOOT/LOST", "global"' not in source
    assert "set_localization_benchmark_mode" not in source
    assert "定位測速：" not in source


def test_ui_has_textual_sim_real_identity_and_emergency_stop() -> None:
    """SIM/REAL must always be distinguishable by TEXT, not colour alone.

    2026-08-03, operator decision: the permanent top identity strip was removed
    as a duplicate of the video-panel HUD, so the identity now lives only in
    render_video via video_hud_identity. This asserts it is still textual and
    still distinguishes the two interfaces; the strip's other guarantee (visible
    even with no video frame) was accepted as lost. SYSTEM_SPEC 8.1 records it.
    """
    build_source = inspect.getsource(OperatorApp._build_ui)
    assert "緊急停止電腦動作" in build_source

    render_source = inspect.getsource(OperatorApp.render_video)
    assert "video_hud_identity" in render_source

    simulated = video_hud_identity(False, "LIVE")
    real = video_hud_identity(True, "MANUAL")
    assert "SIMULATED" in simulated and "REAL" in real
    assert simulated != real


def test_ui_exposes_read_only_olympe_flight_telemetry() -> None:
    source = inspect.getsource(OperatorApp._build_ui)
    assert "飛控遙測（Olympe 讀回）" in source
    assert "raw IMU／raw 氣壓數值" in source

    state = DroneState(
        flight_state="hovering",
        alert_state="none",
        heading_state="ok",
        navigate_home_state="available",
        navigate_home_reason="enabled",
        drone_altitude_m=2.5,
        agl_altitude_m=1.8,
        speed_north_mps=0.18,
        speed_east_mps=0.24,
        speed_down_mps=-0.04,
        ground_speed_mps=0.30,
        gps_fixed=True,
        gps_latitude_deg=25.012345,
        gps_longitude_deg=121.543210,
        gps_altitude_m=12.5,
        gps_latitude_accuracy_m=1.0,
        gps_longitude_accuracy_m=2.0,
        gps_altitude_accuracy_m=3.0,
        gps_satellites=14,
        wind_state="warning",
        vibration_state="ok",
        hover_no_gps_too_high=True,
        wifi_rssi_dbm=-55,
        link_signal_quality_raw=0x84,
        sensor_states={"IMU": True, "barometer": True, "GPS": False},
        telemetry_read_mono_ns=1_000_000_000,
    )
    state.att_roll = 0.1
    state.att_pitch = -0.2
    state.att_yaw = 0.3

    telemetry = format_olympe_telemetry(
        state, now_mono_ns=1_025_000_000
    )

    assert "飛行 hovering" in telemetry["state"]
    assert "飛控融合姿態" in telemetry["attitude"]
    assert "N 0.18" in telemetry["speed"] and "水平 0.30" in telemetry["speed"]
    assert "相對起飛點" in telemetry["altitude"] and "AGL 1.80" in telemetry["altitude"]
    assert "GPS FIX" in telemetry["gps"] and "衛星 14" in telemetry["gps"]
    assert "外部干擾" in telemetry["environment"]
    assert "GPS:FAULT" in telemetry["sensors"]
    assert "cache age 25 ms" in telemetry["sensors"]


def test_ui_separates_firmware_magnetometer_calibration_from_passive_gravity_check() -> None:
    source = inspect.getsource(OperatorApp._build_ui)

    assert "韌體羅盤校正（只允許 landed；使用者手持旋轉）" in source
    assert "姿態／重力檢查（只讀；不寫入飛機）" in source
    assert "drone_magnetometer_start" in source
    assert "skycontroller_magnetometer_start" in source
    assert "不會啟動馬達" in source

    state = DroneState(
        drone_magnetometer_required=2,
        drone_magnetometer_started=True,
        drone_magnetometer_axis="yAxis",
        drone_magnetometer_x_done=True,
        drone_magnetometer_y_done=False,
        drone_magnetometer_z_done=False,
        drone_magnetometer_failed=False,
        skycontroller_magnetometer_state="CalibratingZ",
    )

    status = format_magnetometer_calibration(state)

    assert "建議" in status["drone"]
    assert "Y/pitch" in status["drone"]
    assert "進行中：Z 軸" in status["controller"]


def test_focus_loss_clears_all_holds_and_backend_nudges():
    app = _bare_app()
    app._nudge_keys_held.add("w")
    app._nudge_buttons_held.add("上")

    app._on_input_focus_lost()

    assert not app._nudge_keys_held
    assert not app._nudge_buttons_held
    assert app.backend.calls == [("nudge_clear", {"reason": "ui_focus_lost"})]


def test_slow_backend_command_is_dispatched_without_blocking_caller():
    entered = threading.Event()
    release = threading.Event()

    class SlowBackend(_Backend):
        def command(self, name, **payload):
            entered.set()
            assert release.wait(1.0)
            return super().command(name, **payload)

    app = _bare_app(SlowBackend())
    start = time.monotonic()
    assert app._dispatch_live_command("takeoff", {})
    assert time.monotonic() - start < 0.1
    assert entered.wait(0.5)
    assert not app._dispatch_live_command("takeoff", {})

    release.set()
    command, result, error = app._flight_results.get(timeout=1.0)
    assert (command, result, error) == ("takeoff", "takeoff", None)


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_limit_editor_validates_then_sends_one_firmware_command():
    app = _bare_app()
    app.max_altitude_input_var = _Var("30")
    app.max_distance_input_var = _Var("100")
    app.distance_geofence_input_var = _Var(True)

    assert app.apply_firmware_limits_from_ui()
    assert app.backend.calls == [(
        "firmware_limits_apply",
        {
            "max_altitude_m": 30.0,
            "max_distance_m": 100.0,
            "distance_geofence": True,
        },
    )]


@pytest.mark.parametrize("altitude,distance", [("", "100"), ("nan", "100"), ("30", "0")])
def test_limit_editor_rejects_invalid_values(altitude, distance):
    app = _bare_app()
    app.max_altitude_input_var = _Var(altitude)
    app.max_distance_input_var = _Var(distance)
    app.distance_geofence_input_var = _Var(True)

    assert not app.apply_firmware_limits_from_ui()
    assert app.backend.calls == []


def test_limit_editor_can_load_current_firmware_readback_without_writing():
    backend = _Backend()
    backend.state = SimpleNamespace(
        max_altitude_m=18.0,
        max_distance_m=75.0,
        distance_geofence_enabled=False,
    )
    app = _bare_app(backend)
    app.max_altitude_input_var = _Var("")
    app.max_distance_input_var = _Var("")
    app.distance_geofence_input_var = _Var(True)

    assert app.load_firmware_limits_from_state()
    assert app.max_altitude_input_var.get() == "18"
    assert app.max_distance_input_var.get() == "75"
    assert app.distance_geofence_input_var.get() is False
    assert backend.calls == []


def test_autonomous_speed_editor_validates_then_sends_one_command():
    app = _bare_app()
    app.autonomous_speed_limit_input_var = _Var("0.30")

    assert app.apply_autonomous_speed_limit_from_ui()
    assert app.backend.calls == [(
        "auto_speed_limit_apply",
        {"speed_limit_mps": 0.3},
    )]


@pytest.mark.parametrize("value", ["", "nan", "0", "-0.1"])
def test_autonomous_speed_editor_rejects_invalid_values(value):
    app = _bare_app()
    app.autonomous_speed_limit_input_var = _Var(value)

    assert not app.apply_autonomous_speed_limit_from_ui()
    assert app.backend.calls == []
