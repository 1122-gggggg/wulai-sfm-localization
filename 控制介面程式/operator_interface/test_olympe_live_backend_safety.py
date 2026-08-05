from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

import olympe_live_backend as backend_module
from backend_contract import ControlAction, ControlRequest, InterfaceMode, SessionConfig


class _Message:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __rshift__(self, other):
        return _Composite(self, other)


class _Composite:
    def __init__(self, *parts):
        self.parts = list(parts)

    def __rshift__(self, other):
        return _Composite(*self.parts, other)


def _message_factory(name: str):
    def factory(*args, **kwargs):
        return _Message(name, *args, **kwargs)

    factory._message_name = name
    return factory


class _Expectation:
    def __init__(self, success: bool, on_wait=None):
        self._success = success
        self._on_wait = on_wait
        self._waited = False

    def wait(self, _timeout=None):
        if not self._waited:
            self._waited = True
            if self._on_wait is not None:
                self._on_wait()
        return self

    def success(self) -> bool:
        return self._success


class _FakeDrone:
    def __init__(self):
        self.events: list[str] = []
        self.pcmds: list[tuple] = []
        self.source_state = "SkyController"
        self.source_requests: list[str] = []
        self.source_ack = True
        self.source_readback_override: str | None = None
        self.flight_state = "hovering"
        self.connected = True
        self.managed_drone_connected = True
        self.battery_pct = 100.0
        self.gps_fixed = 1
        self.home_latitude = 25.0
        self.home_longitude = 121.0
        self.home_altitude = 12.0
        self.home_reachable = True
        self.rth_success = True
        self.rth_auto_trigger_mode = "on"
        self.rth_delay_s = 1
        self.rth_ending_behavior = "landing"
        self.setting_ack = {
            "MaxAltitude": True,
            "MaxDistance": True,
            "NoFlyOverMaxDistance": True,
        }
        self.apply_setting = True
        self.setting_states = {
            "MaxAltitudeChanged": {"current": 10.0, "min": 0.5, "max": 4000.0},
            "MaxDistanceChanged": {"current": 50.0, "min": 10.0, "max": 4000.0},
            "NoFlyOverMaxDistanceChanged": {"shouldNotFlyOver": 1},
            "MaxTiltChanged": {"current": 10.0, "min": 1.0, "max": 40.0},
            "MaxVerticalSpeedChanged": {"current": 0.5, "min": 0.1, "max": 4.0},
            "MaxRotationSpeedChanged": {"current": 10.0, "min": 3.0, "max": 200.0},
        }
        self.landing_success = True
        self.takeoff_success = True
        self.takeoff_wait_hook = None
        self.drone_calibration_required = 0
        self.drone_calibration_started = False
        self.drone_calibration_axis = "none"
        self.drone_calibration_x_done = True
        self.drone_calibration_y_done = True
        self.drone_calibration_z_done = True
        self.drone_calibration_failed = False
        self.drone_calibration_ack = True
        self.controller_calibration_state = "Calibrated"
        self.controller_calibration_ack = True

    @staticmethod
    def _first(message):
        return message.parts[0] if isinstance(message, _Composite) else message

    def __call__(self, message):
        first = self._first(message)
        self.events.append(first.name)
        if first.name == "PCMD":
            self.pcmds.append(first.args)
        if first.name == "setPilotingSource":
            requested = first.kwargs["source"]
            self.source_requests.append(requested)

            def confirm_source():
                if self.source_ack:
                    self.source_state = requested

            return _Expectation(self.source_ack, confirm_source)
        if first.name in {"MaxAltitude", "MaxDistance", "NoFlyOverMaxDistance"}:
            state_name = first.name + "Changed"
            field = {
                "MaxAltitude": "current",
                "MaxDistance": "current",
                "NoFlyOverMaxDistance": "shouldNotFlyOver",
            }[first.name]
            argument = {
                "MaxAltitude": "current",
                "MaxDistance": "value",
                "NoFlyOverMaxDistance": "shouldNotFlyOver",
            }[first.name]
            ok = self.setting_ack[first.name]

            def apply_setting():
                if ok and self.apply_setting:
                    self.setting_states[state_name][field] = first.kwargs[argument]

            return _Expectation(ok, apply_setting)
        if first.name == "set_auto_trigger_mode":
            return _Expectation(
                True,
                lambda: setattr(
                    self, "rth_auto_trigger_mode", first.kwargs["mode"]
                ),
            )
        if first.name == "set_delay":
            return _Expectation(
                True,
                lambda: setattr(self, "rth_delay_s", first.kwargs["delay"]),
            )
        if first.name == "set_ending_behavior":
            return _Expectation(
                True,
                lambda: setattr(
                    self, "rth_ending_behavior", first.kwargs["ending_behavior"]
                ),
            )
        if first.name == "return_to_home":
            return _Expectation(self.rth_success)
        if first.name == "Landing":
            return _Expectation(
                self.landing_success,
                lambda: setattr(
                    self, "flight_state",
                    "landed" if self.landing_success else "landing",
                ),
            )
        if first.name == "TakeOff":
            def finish_takeoff():
                if self.takeoff_success:
                    self.flight_state = "hovering"
                if self.takeoff_wait_hook is not None:
                    self.takeoff_wait_hook()

            return _Expectation(self.takeoff_success, finish_takeoff)
        if first.name == "MagnetoCalibration":
            requested = int(first.kwargs.get("calibrate", 0))

            def update_drone_calibration():
                if not self.drone_calibration_ack:
                    return
                self.drone_calibration_started = bool(requested)
                self.drone_calibration_axis = "xAxis" if requested else "none"
                if requested:
                    self.drone_calibration_x_done = False
                    self.drone_calibration_y_done = False
                    self.drone_calibration_z_done = False
                    self.drone_calibration_failed = False

            return _Expectation(
                self.drone_calibration_ack,
                update_drone_calibration,
            )
        if first.name == "StartCalibration":
            def start_controller_calibration():
                if self.controller_calibration_ack:
                    self.controller_calibration_state = "CalibratingX"

            return _Expectation(
                self.controller_calibration_ack,
                start_controller_calibration,
            )
        if first.name == "AbortCalibration":
            def abort_controller_calibration():
                if self.controller_calibration_ack:
                    self.controller_calibration_state = "NotCalibrated"

            return _Expectation(
                self.controller_calibration_ack,
                abort_controller_calibration,
            )
        return _Expectation(True)

    def get_state(self, message_type):
        name = getattr(message_type, "_message_name", "")
        if name == "pilotingSource":
            return {
                "source": self.source_readback_override or self.source_state,
            }
        if name == "FlyingStateChanged":
            return {"state": self.flight_state}
        if name == "BatteryStateChanged":
            return {"percent": self.battery_pct}
        if name == "GPSFixStateChanged":
            return {"fixed": self.gps_fixed}
        if name == "ProductNameChanged":
            return {"name": "ANAFI"}
        if name == "ProductModel":
            return {"model": "ANAFI_4K"}
        if name == "ProductVersionChanged":
            return {"software": "1.8.2", "hardware": "1"}
        if name == "ProductSerialHighChanged":
            return {"high": "PI"}
        if name == "ProductSerialLowChanged":
            return {"low": "000001"}
        if name == "BoardIdChanged":
            return {"id": "anafi-hw"}
        if name == "ProductSerialChanged":
            return {"serialNumber": "SC3-000001"}
        if name == "ProductVariantChanged":
            return {"variant": "SkyController3"}
        if name == "HomeChanged" or name == "home_location":
            return {
                "latitude": self.home_latitude,
                "longitude": self.home_longitude,
                "altitude": self.home_altitude,
            }
        if name == "ReturnHomeDelayChanged":
            return {"delay": 10}
        if name == "ReturnHomeMinAltitudeChanged":
            return {"value": 20.0, "min": 20.0, "max": 100.0}
        if name == "home_reachability":
            return {"status": "reachable" if self.home_reachable else "not_reachable"}
        if name == "auto_trigger_mode":
            return {"mode": self.rth_auto_trigger_mode}
        if name == "delay":
            return {"delay": self.rth_delay_s, "min": 1, "max": 120}
        if name == "ending_behavior":
            return {"ending_behavior": self.rth_ending_behavior}
        if name == "connection_state":
            return {
                "state": "connected" if self.managed_drone_connected else "disconnected"
            }
        if name in self.setting_states:
            return dict(self.setting_states[name])
        if name == "AttitudeChanged":
            return {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        if name == "MagnetoCalibrationRequiredState":
            return {"required": self.drone_calibration_required}
        if name == "MagnetoCalibrationStartedChanged":
            return {"started": int(self.drone_calibration_started)}
        if name == "MagnetoCalibrationAxisToCalibrateChanged":
            return {"axis": self.drone_calibration_axis}
        if name == "MagnetoCalibrationStateChanged":
            return {
                "xAxisCalibration": int(self.drone_calibration_x_done),
                "yAxisCalibration": int(self.drone_calibration_y_done),
                "zAxisCalibration": int(self.drone_calibration_z_done),
                "calibrationFailed": int(self.drone_calibration_failed),
            }
        if name == "MagnetoCalibrationStateV2":
            return {"state": self.controller_calibration_state}
        if name == "AltitudeChanged":
            return {"altitude": 0.0}
        if name == "AltitudeAboveGroundChanged":
            return {"altitude": 1.2}
        if name == "SpeedChanged":
            return {"speedX": 0.18, "speedY": 0.24, "speedZ": -0.04}
        if name == "GpsLocationChanged":
            return {
                "latitude": 25.012345,
                "longitude": 121.543210,
                "altitude": 12.5,
                "latitude_accuracy": 1,
                "longitude_accuracy": 2,
                "altitude_accuracy": 3,
            }
        if name == "AlertStateChanged":
            return {"state": "none"}
        if name == "NavigateHomeStateChanged":
            return {"state": "available", "reason": "enabled"}
        if name == "HeadingLockedStateChanged":
            return {"state": "ok"}
        if name == "HoveringWarning":
            return {"no_gps_too_dark": 0, "no_gps_too_high": 1}
        if name == "WindStateChanged":
            return {"state": "warning"}
        if name == "VibrationLevelChanged":
            return {"state": "ok"}
        if name == "NumberOfSatelliteChanged":
            return {"numberOfSatellite": 14}
        if name == "WifiSignalChanged":
            return {"rssi": -55}
        if name == "LinkSignalQuality":
            return {"value": 0x84}
        if name == "SensorsStatesListChanged":
            return {
                "IMU": {"sensorName": "IMU", "sensorState": 1},
                "barometer": {"sensorName": "barometer", "sensorState": 1},
                "GPS": {"sensorName": "GPS", "sensorState": 0},
            }
        return {}

    def connection_state(self):
        return SimpleNamespace(name="Connected" if self.connected else "Disconnected")

    def disconnect(self):
        self.events.append("disconnect")
        return True


class _FakeLog:
    def __init__(self):
        self.records: list[tuple[str, dict]] = []
        self.healthy = True
        self.durable = True
        self.path = None

    def event(self, event: str, **fields) -> None:
        self.records.append((event, fields))

    def close(self) -> None:
        self.records.append(("log_close", {}))


def _install_fake_olympe(monkeypatch: pytest.MonkeyPatch) -> None:
    package_names = (
        "olympe",
        "olympe.messages",
        "olympe.messages.ardrone3",
        "olympe.messages.common",
        "olympe.messages.skyctrl",
        "olympe.messages.drone_manager",
        "olympe.features",
    )
    for name in package_names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    modules = {
        "olympe.messages.ardrone3.Piloting": {
            "PCMD": _message_factory("PCMD"),
            "Landing": _message_factory("Landing"),
            "TakeOff": _message_factory("TakeOff"),
        },
        "olympe.messages.ardrone3.PilotingState": {
            "FlyingStateChanged": _message_factory("FlyingStateChanged"),
            "AttitudeChanged": _message_factory("AttitudeChanged"),
            "AltitudeChanged": _message_factory("AltitudeChanged"),
            "SpeedChanged": _message_factory("SpeedChanged"),
            "AltitudeAboveGroundChanged": _message_factory("AltitudeAboveGroundChanged"),
            "GpsLocationChanged": _message_factory("GpsLocationChanged"),
            "AlertStateChanged": _message_factory("AlertStateChanged"),
            "NavigateHomeStateChanged": _message_factory("NavigateHomeStateChanged"),
            "HeadingLockedStateChanged": _message_factory("HeadingLockedStateChanged"),
            "HoveringWarning": _message_factory("HoveringWarning"),
            "WindStateChanged": _message_factory("WindStateChanged"),
            "VibrationLevelChanged": _message_factory("VibrationLevelChanged"),
        },
        "olympe.messages.ardrone3.GPSState": {
            "NumberOfSatelliteChanged": _message_factory("NumberOfSatelliteChanged"),
        },
        "olympe.messages.ardrone3.PilotingSettings": {
            "MaxAltitude": _message_factory("MaxAltitude"),
            "MaxDistance": _message_factory("MaxDistance"),
            "NoFlyOverMaxDistance": _message_factory("NoFlyOverMaxDistance"),
        },
        "olympe.messages.ardrone3.PilotingSettingsState": {
            name: _message_factory(name) for name in (
                "MaxAltitudeChanged",
                "MaxDistanceChanged",
                "NoFlyOverMaxDistanceChanged",
                "MaxTiltChanged",
            )
        },
        "olympe.messages.ardrone3.SpeedSettingsState": {
            "MaxVerticalSpeedChanged": _message_factory("MaxVerticalSpeedChanged"),
            "MaxRotationSpeedChanged": _message_factory("MaxRotationSpeedChanged"),
        },
        "olympe.messages.ardrone3.GPSSettingsState": {
            "GPSFixStateChanged": _message_factory("GPSFixStateChanged"),
        },
        "olympe.messages.common.CommonState": {
            "BatteryStateChanged": _message_factory("BatteryStateChanged"),
            "WifiSignalChanged": _message_factory("WifiSignalChanged"),
            "LinkSignalQuality": _message_factory("LinkSignalQuality"),
            "SensorsStatesListChanged": _message_factory("SensorsStatesListChanged"),
        },
        "olympe.messages.common.Calibration": {
            "MagnetoCalibration": _message_factory("MagnetoCalibration"),
        },
        "olympe.messages.common.CalibrationState": {
            name: _message_factory(name) for name in (
                "MagnetoCalibrationRequiredState",
                "MagnetoCalibrationStartedChanged",
                "MagnetoCalibrationAxisToCalibrateChanged",
                "MagnetoCalibrationStateChanged",
            )
        },
        "olympe.messages.skyctrl.Calibration": {
            "StartCalibration": _message_factory("StartCalibration"),
            "AbortCalibration": _message_factory("AbortCalibration"),
        },
        "olympe.messages.skyctrl.CalibrationState": {
            "MagnetoCalibrationStateV2": _message_factory(
                "MagnetoCalibrationStateV2"
            ),
        },
        "olympe.messages.skyctrl.CoPiloting": {
            "setPilotingSource": _message_factory("setPilotingSource"),
        },
        "olympe.messages.skyctrl.CoPilotingState": {
            "pilotingSource": _message_factory("pilotingSource"),
        },
        "olympe.messages.common.SettingsState": {
            name: _message_factory(name) for name in (
                "ProductNameChanged",
                "ProductVersionChanged",
                "ProductSerialHighChanged",
                "ProductSerialLowChanged",
                "BoardIdChanged",
            )
        },
        "olympe.messages.skyctrl.SettingsState": {
            name: _message_factory(name) for name in (
                "ProductSerialChanged",
                "ProductVersionChanged",
                "ProductVariantChanged",
            )
        },
        "olympe.messages.camera": {
            "stop_recording": _message_factory("stop_recording"),
            "recording_state": _message_factory("recording_state"),
        },
        "olympe.features.media": {
            "download_media": _message_factory("download_media"),
        },
        "olympe.messages.rth": {
            name: _message_factory(name) for name in (
                "home_location",
                "home_reachability",
                "auto_trigger_mode",
                "delay",
                "ending_behavior",
                "return_to_home",
                "set_auto_trigger_mode",
                "set_delay",
                "set_ending_behavior",
            )
        },
        "olympe.messages.drone_manager": {
            "connection_state": _message_factory("connection_state"),
        },
    }
    modules["olympe.messages.common.CommonState"]["ProductModel"] = (
        _message_factory("ProductModel")
    )
    modules["olympe.messages.ardrone3.GPSSettingsState"].update({
        name: _message_factory(name) for name in (
            "HomeChanged",
            "ReturnHomeDelayChanged",
            "ReturnHomeMinAltitudeChanged",
        )
    })
    for name, attributes in modules.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def make_backend(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_fake_olympe(monkeypatch)
    monkeypatch.setattr(backend_module, "_DEFAULT_RECORD_DIR", tmp_path)
    # Unit tests must not inherit the workstation's current free-space state.
    # Dedicated storage-guard tests override this stub with warning/critical
    # results; production still uses the real 5% fail-closed threshold.
    monkeypatch.setattr(
        backend_module,
        "assess_disk_space",
        lambda _path: SimpleNamespace(
            free_bytes=50 * 1024**3,
            free_percent=50.0,
            warning=False,
            takeoff_blocked=False,
            reason="ok",
        ),
    )
    monkeypatch.setattr(
        backend_module.OlympeLiveBackend, "_connect", lambda self: None,
    )

    def make(
            *, skycontroller: bool = True, pulse_s: float = 0.2,
            max_altitude_m: float | None = None,
            max_distance_m: float | None = None,
            distance_geofence: bool = True,
            min_takeoff_battery_pct: float = 30.0,
            require_gps_for_geofence: bool = True,
            with_video: bool = False):
        state = SimpleNamespace(
            loc="", stream="", mode="", tracker_state="", last_command="",
        )
        profile = SimpleNamespace(
            gimbal_pitch_min_deg=-90.0,
            gimbal_pitch_max_deg=90.0,
            digital_zoom_max=3.0,
        )
        backend = backend_module.OlympeLiveBackend(
            lambda: state,
            profile,
            ip="192.168.53.1" if skycontroller else "192.168.42.1",
            controller="skycontroller3" if skycontroller else "drone",
            nudge_pulse_s=pulse_s,
            max_altitude_m=max_altitude_m,
            max_distance_m=max_distance_m,
            distance_geofence=distance_geofence,
            min_takeoff_battery_pct=min_takeoff_battery_pct,
            require_gps_for_geofence=require_gps_for_geofence,
            approved_aircraft_firmware=("1.8.2",),
            approved_controller_firmware=("1.8.2",),
            approved_olympe_versions=("8.4.0",),
            with_video=with_video,
        )
        backend.log = _FakeLog()
        backend.drone = _FakeDrone()
        backend._stick_monitor = SimpleNamespace(
            is_active=lambda: False,
            stop=lambda: None,
        )
        backend.state.stick_monitor_ok = True
        return backend

    return make


def test_inventory_reads_and_logs_actual_connected_hardware(make_backend):
    backend = make_backend()

    inventory = backend.read_connection_inventory()

    assert inventory["aircraft"]["name"] == "ANAFI"
    assert inventory["aircraft"]["serial"] == "PI000001"
    assert inventory["aircraft"]["software"] == "1.8.2"
    assert inventory["controller"]["serial"] == "SC3-000001"
    assert inventory["home"]["valid"] is True
    assert inventory["lost_link"]["return_home_delay_s"] == 1
    assert inventory["takeoff_inventory_ready"] is True
    assert backend.state.home_valid is True
    assert backend.state.rth_policy_valid is True
    assert any(event == "hardware_inventory" for event, _ in backend.log.records)


def test_real_backend_typed_session_start_is_not_shadowed(make_backend):
    backend = make_backend()
    config = SessionConfig(
        session_id="real-session",
        interface_mode=InterfaceMode.REAL_FLIGHT,
        site_profile="/tmp/site.json",
        site_profile_sha256="a" * 64,
        asset_sha256={},
        runtime_profile_sha256="b" * 64,
        source="192.168.53.1",
        offline=True,
    )

    assert callable(backend.start)
    assert backend.start(config).started


def test_video_inventory_is_logged_once_after_first_observed_frame(make_backend):
    backend = make_backend()

    class VideoStream:
        output_index = 1

        @staticmethod
        def metadata_snapshot():
            return {
                "codec": "H.264",
                "codec_evidence": "configured-pdraw-input-contract",
                "codec_observed": False,
                "ui_frame_width_px": 1280,
                "ui_frame_height_px": 720,
                "observed_fps": 29.5,
                "source_timestamp_capable": True,
                "source_timestamp_trusted": True,
            }

    backend.video_stream = VideoStream()

    assert backend._maybe_log_video_inventory()
    assert not backend._maybe_log_video_inventory()
    records = [fields for event, fields in backend.log.records if event == "video_inventory"]
    assert len(records) == 1
    assert records[0]["codec"] == "H.264"
    assert backend.connection_inventory["video"]["ui_frame_width_px"] == 1280


def test_preflight_blocks_firmware_not_in_approved_receipt(make_backend):
    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend.approved_aircraft_firmware = frozenset({"different-version"})

    assert not backend._takeoff_preflight()
    assert "aircraft firmware" in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events


def test_preflight_blocks_unhealthy_safety_log(make_backend):
    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend.log.healthy = False

    assert not backend._takeoff_preflight()
    assert "safety log" in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events


def test_preflight_blocks_critical_disk_pressure(
    make_backend, monkeypatch: pytest.MonkeyPatch,
):
    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    monkeypatch.setattr(
        backend_module,
        "assess_disk_space",
        lambda _path: SimpleNamespace(
            takeoff_blocked=True,
            warning=True,
            free_bytes=4 * 1024**3,
            free_percent=4.0,
            reason="critical disk test",
        ),
    )

    assert not backend._takeoff_preflight()
    assert backend.state.preflight_reason == "critical disk test"
    assert "TakeOff" not in backend.drone.events


def test_runtime_log_failure_zeros_motion_and_hands_to_manual_once(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.pilot_sticks = False
    backend.log.healthy = False

    assert backend._check_runtime_storage_health(10.0) is False
    first_zero_count = len(backend.drone.pcmds)
    assert first_zero_count >= 1
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.pilot_sticks is True
    assert backend.state.active_incident == "disk_or_log_failure"

    assert backend._check_runtime_storage_health(12.0) is False
    assert len(backend.drone.pcmds) == first_zero_count


def test_runtime_critical_disk_pressure_zeros_motion_and_never_auto_resumes(
    make_backend, monkeypatch: pytest.MonkeyPatch,
):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.pilot_sticks = False
    monkeypatch.setattr(
        backend_module,
        "assess_disk_space",
        lambda _path: SimpleNamespace(
            takeoff_blocked=True,
            warning=True,
            free_bytes=4 * 1024**3,
            free_percent=4.0,
            reason="critical disk during flight",
        ),
    )

    assert backend._check_runtime_storage_health(10.0) is False
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.state.mode == "MANUAL"
    assert backend.state.active_incident == "disk_or_log_failure"
    assert backend.state.disk_warning is True


@pytest.mark.parametrize(
    ("skycontroller", "pilot_sticks", "tracker_state"),
    [
        (True, True, "STICKS"),
        (False, False, "HOVER"),
    ],
)
def test_connect_keeps_skycontroller_sticks_until_explicit_pc_request(
        monkeypatch, tmp_path, skycontroller, pilot_sticks, tracker_state):
    _install_fake_olympe(monkeypatch)
    monkeypatch.setattr(backend_module, "_DEFAULT_RECORD_DIR", tmp_path)
    monkeypatch.setattr(backend_module, "quiet_olympe_logs", lambda: None)
    drone = _FakeDrone()
    frame_source = ModuleType("olympe_frame_source")
    frame_source.connect = lambda _ip, _controller: drone
    monkeypatch.setitem(sys.modules, "olympe_frame_source", frame_source)
    state = SimpleNamespace(
        loc="", stream="", mode="", tracker_state="", last_command="",
    )
    profile = SimpleNamespace(
        gimbal_pitch_min_deg=-90.0,
        gimbal_pitch_max_deg=90.0,
        digital_zoom_max=3.0,
    )

    backend = backend_module.OlympeLiveBackend(
        lambda: state,
        profile,
        ip="192.168.53.1" if skycontroller else "192.168.42.1",
        controller="skycontroller3" if skycontroller else "drone",
        with_video=False,
    )

    if skycontroller:
        assert drone.source_state == "SkyController"
        assert "setPilotingSource" in drone.events
    else:
        assert "setPilotingSource" not in drone.events
    assert backend.pilot_sticks is pilot_sticks
    assert backend.state.tracker_state == tracker_state
    assert "PCMD" not in drone.events
    assert "TakeOff" not in drone.events


def test_piloting_source_requires_expectation_and_matching_readback(make_backend):
    backend = make_backend()
    backend.drone.source_ack = False
    assert not backend._set_piloting_source("Controller")

    backend.drone.source_ack = True
    backend.drone.source_readback_override = "SkyController"
    assert not backend._set_piloting_source("Controller")

    backend.drone.source_readback_override = None
    assert backend._set_piloting_source("Controller")


def test_unset_limits_keep_ground_backend_available_but_block_takeoff(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "landed"

    assert not backend._configure_firmware_limits_if_safe()
    assert not backend.takeoff_cmd()

    assert backend.drone is not None
    assert "ground UI only" in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events
    assert not any(event.startswith("Max") for event in backend.drone.events)


def test_firmware_limits_write_only_when_landed_and_confirm_readback(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "flying"

    assert not backend._configure_firmware_limits_if_safe()
    assert "MaxAltitude" not in backend.drone.events

    backend.drone.flight_state = "landed"
    assert backend._configure_firmware_limits_if_safe()

    assert backend.drone.events[-3:] == [
        "MaxAltitude", "MaxDistance", "NoFlyOverMaxDistance",
    ]
    assert backend.state.max_altitude_m == pytest.approx(20.0)
    assert backend.state.max_distance_m == pytest.approx(80.0)
    assert backend.state.distance_geofence_enabled is True
    assert backend._firmware_config_ok


def test_ui_firmware_limit_apply_updates_desired_values_and_readback(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"
    epoch = backend._pulse_token

    assert backend.apply_firmware_limits(30.0, 100.0, True)

    assert backend._pulse_token == epoch + 1
    assert backend.desired_max_altitude_m == pytest.approx(30.0)
    assert backend.desired_max_distance_m == pytest.approx(100.0)
    assert backend.desired_distance_geofence is True
    assert backend.state.max_altitude_m == pytest.approx(30.0)
    assert backend.state.max_distance_m == pytest.approx(100.0)
    assert backend.state.distance_geofence_enabled is True
    assert backend.drone.events[-3:] == [
        "MaxAltitude", "MaxDistance", "NoFlyOverMaxDistance",
    ]


def test_ui_firmware_limit_apply_rejects_airborne_without_mutating_request(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "flying"
    epoch = backend._pulse_token

    assert not backend.apply_firmware_limits(30.0, 100.0, True)

    assert backend._pulse_token == epoch
    assert backend.desired_max_altitude_m == pytest.approx(20.0)
    assert backend.desired_max_distance_m == pytest.approx(80.0)
    assert "not confirmed landed" in backend.state.preflight_reason
    assert not any(event.startswith("Max") for event in backend.drone.events)


def test_ui_firmware_limit_command_returns_apply_result(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"

    assert backend.command(
        "firmware_limits_apply",
        max_altitude_m=30.0,
        max_distance_m=100.0,
        distance_geofence=True,
    ) is True

    backend.drone.flight_state = "flying"
    assert backend.command(
        "firmware_limits_apply",
        max_altitude_m=25.0,
        max_distance_m=90.0,
        distance_geofence=True,
    ) is False


def test_autonomous_speed_limit_is_landed_only_and_keeps_real_auto_locked(
    make_backend,
):
    backend = make_backend()
    backend.drone.flight_state = "landed"
    backend.state.autonomous_approval_valid = True

    assert backend.command(
        "auto_speed_limit_apply", speed_limit_mps=0.2
    ) is True
    assert backend.state.autonomous_speed_limit_mps == pytest.approx(0.2)
    assert backend.state.autonomous_approval_valid is False
    assert backend.state.autonomous_locked is True

    backend.drone.flight_state = "flying"
    assert backend.command(
        "auto_speed_limit_apply", speed_limit_mps=0.1
    ) is False
    assert backend.state.autonomous_speed_limit_mps == pytest.approx(0.2)


def test_takeoff_preflight_retries_transient_connect_state_cache_failure(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = ""

    assert not backend._configure_firmware_limits_if_safe()
    assert not backend._firmware_config_ok
    assert "MaxAltitude" not in backend.drone.events

    backend.drone.flight_state = "landed"
    assert backend._takeoff_preflight()

    assert backend._firmware_config_ok
    assert backend.state.preflight_ok is True
    assert "MaxAltitude" in backend.drone.events
    assert "MaxDistance" in backend.drone.events


def test_firmware_bounds_failure_sends_no_setting_command(make_backend):
    backend = make_backend(max_altitude_m=5000.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"

    assert not backend._configure_firmware_limits_if_safe()

    assert "outside firmware range" in backend.state.preflight_reason
    assert "MaxAltitude" not in backend.drone.events
    assert "MaxDistance" not in backend.drone.events


def test_firmware_expectation_failure_stops_remaining_writes(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"
    backend.drone.setting_ack["MaxAltitude"] = False

    assert not backend._configure_firmware_limits_if_safe()

    assert "failed or timed out" in backend.state.preflight_reason
    assert "MaxAltitude" in backend.drone.events
    assert "MaxDistance" not in backend.drone.events


def test_firmware_readback_mismatch_is_fail_closed(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"
    backend.drone.apply_setting = False

    assert not backend._configure_firmware_limits_if_safe()

    assert "readback mismatch" in backend.state.preflight_reason
    assert not backend._firmware_config_ok


def test_geofence_readback_mismatch_is_fail_closed(make_backend):
    backend = make_backend(
        max_altitude_m=10.0,
        max_distance_m=50.0,
        distance_geofence=False,
    )
    backend.drone.flight_state = "landed"
    backend.drone.apply_setting = False

    assert not backend._configure_firmware_limits_if_safe()

    assert "NoFlyOverMaxDistance readback mismatch" in backend.state.preflight_reason


def test_poll_displays_all_current_firmware_readbacks(make_backend):
    backend = make_backend()
    backend.drone.setting_states["MaxAltitudeChanged"]["current"] = 30.0
    backend.drone.setting_states["MaxDistanceChanged"]["current"] = 120.0
    backend.drone.setting_states["NoFlyOverMaxDistanceChanged"]["shouldNotFlyOver"] = 0
    backend.drone.setting_states["MaxTiltChanged"]["current"] = 12.0
    backend.drone.setting_states["MaxVerticalSpeedChanged"]["current"] = 0.7
    backend.drone.setting_states["MaxRotationSpeedChanged"]["current"] = 15.0

    state = backend.poll()

    assert state.max_altitude_m == pytest.approx(30.0)
    assert state.max_distance_m == pytest.approx(120.0)
    assert state.distance_geofence_enabled is False
    assert state.max_tilt_deg == pytest.approx(12.0)
    assert state.max_vertical_speed_mps == pytest.approx(0.7)
    assert state.max_rotation_speed_dps == pytest.approx(15.0)
    assert state.airspeed_mps == pytest.approx(0.3)
    assert state.ground_speed_mps == pytest.approx(0.3)
    assert state.speed_north_mps == pytest.approx(0.18)
    assert state.speed_east_mps == pytest.approx(0.24)
    assert state.speed_down_mps == pytest.approx(-0.04)
    assert state.agl_altitude_m == pytest.approx(1.2)
    assert state.gps_latitude_deg == pytest.approx(25.012345)
    assert state.gps_longitude_deg == pytest.approx(121.543210)
    assert state.gps_altitude_m == pytest.approx(12.5)
    assert state.gps_satellites == 14
    assert state.heading_state == "ok"
    assert state.alert_state == "none"
    assert state.navigate_home_state == "available"
    assert state.wind_state == "warning"
    assert state.vibration_state == "ok"
    assert state.hover_no_gps_too_high is True
    assert state.wifi_rssi_dbm == -55
    assert state.link_signal_quality_raw == 0x84
    assert state.sensor_states == {"IMU": True, "barometer": True, "GPS": False}
    assert state.drone_magnetometer_required == 0
    assert state.drone_magnetometer_started is False
    assert state.drone_magnetometer_axis == "none"
    assert state.drone_magnetometer_x_done is True
    assert state.skycontroller_magnetometer_state == "Calibrated"


def test_drone_magnetometer_calibration_is_human_landed_only_and_never_takes_off(
        make_backend):
    backend = make_backend()
    automated = backend.command(ControlRequest.create(
        ControlAction.DRONE_MAGNETOMETER_START,
        human_origin=False,
    ))
    assert not automated.accepted
    assert automated.reason_code == "HUMAN_ORIGIN_REQUIRED"

    request = ControlRequest.create(
        ControlAction.DRONE_MAGNETOMETER_START,
        human_origin=True,
    )
    airborne = backend.command(request)
    assert not airborne.accepted
    assert "MagnetoCalibration" not in backend.drone.events

    backend.drone.flight_state = "landed"
    started = backend.command(request)

    assert started.accepted and started.executed
    assert "PCMD" in backend.drone.events
    assert "MagnetoCalibration" in backend.drone.events
    assert "TakeOff" not in backend.drone.events
    assert backend.state.drone_magnetometer_started is True
    assert backend.state.drone_magnetometer_axis == "xAxis"
    assert backend.state.preflight_ok is False

    cancelled = backend.command(ControlRequest.create(
        ControlAction.DRONE_MAGNETOMETER_CANCEL,
        human_origin=True,
    ))
    assert cancelled.accepted
    assert backend.state.drone_magnetometer_started is False


def test_skycontroller_magnetometer_calibration_is_separate_and_landed_only(
        make_backend):
    backend = make_backend()
    backend.drone.flight_state = "landed"

    started = backend.command(ControlRequest.create(
        ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
        human_origin=True,
    ))

    assert started.accepted
    assert backend.state.skycontroller_magnetometer_state == "CalibratingX"
    assert "StartCalibration" in backend.drone.events
    assert "TakeOff" not in backend.drone.events

    cancelled = backend.command(ControlRequest.create(
        ControlAction.SKYCONTROLLER_MAGNETOMETER_CANCEL,
        human_origin=True,
    ))
    assert cancelled.accepted
    assert backend.state.skycontroller_magnetometer_state == "NotCalibrated"

    direct = make_backend(skycontroller=False)
    direct.drone.flight_state = "landed"
    rejected = direct.command(ControlRequest.create(
        ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
        human_origin=True,
    ))
    assert not rejected.accepted
    assert "StartCalibration" not in direct.drone.events


@pytest.mark.parametrize(
    ("drone_required", "controller_state", "reason"),
    [
        (None, "Calibrated", "state is unavailable"),
        (1, "Calibrated", "aircraft magnetometer calibration is required"),
        (0, "NotCalibrated", "SkyController magnetometer calibration is required"),
        (0, "CalibratingY", "SkyController magnetometer calibration is in progress"),
    ],
)
def test_takeoff_preflight_blocks_unusable_magnetometer_state(
        make_backend, drone_required, controller_state, reason):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.drone_calibration_required = drone_required
    backend.drone.controller_calibration_state = controller_state

    assert not backend._takeoff_preflight()

    assert reason in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events


def test_recommended_drone_calibration_warns_but_does_not_block_takeoff_preflight(
        make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.drone_calibration_required = 2

    assert backend._takeoff_preflight()

    warnings = [
        fields for event, fields in backend.log.records
        if event == "magnetometer_calibration_warning"
    ]
    assert warnings and warnings[-1]["requirement"] == "recommended"


def test_autonomous_request_is_rejected_first_when_magnetometer_is_required(
        make_backend):
    backend = make_backend()
    backend.drone.drone_calibration_required = 1

    result = backend.command(ControlRequest.create(
        ControlAction.START_AUTO,
        human_origin=True,
    ))

    assert not result.accepted
    assert result.reason_code == "MAGNETOMETER_CALIBRATION_REQUIRED"
    assert backend.command("start_auto") is False
    assert backend.drone.source_requests == []


def test_takeoff_preflight_success_checks_video_battery_gps_source_and_limits(
        make_backend):
    backend = make_backend(
        max_altitude_m=10.0,
        max_distance_m=50.0,
        with_video=True,
    )
    backend.drone.flight_state = "landed"
    backend.video_stream = SimpleNamespace(last_stamp=time.monotonic())

    assert backend._takeoff_preflight()

    assert backend.state.preflight_ok is True
    assert backend.state.preflight_reason == "ready"
    assert backend.drone.source_state == "Controller"


def test_limit_mismatch_never_hands_off_skycontroller(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.pilot_sticks = True
    backend._firmware_config_ok = True
    backend.drone.setting_states["MaxAltitudeChanged"]["current"] = 11.0

    assert not backend._takeoff_preflight()

    assert backend.drone.source_state == "SkyController"
    assert backend.drone.source_requests == []
    assert backend.pilot_sticks


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("airborne", "confirmed landed"),
        ("link", "healthy Olympe link"),
        ("video", "live video frame"),
        ("battery", "takeoff floor"),
        ("gps", "GPS fix"),
        ("source", "source command"),
        ("limit", "readback mismatch"),
    ],
)
def test_takeoff_preflight_fail_closed_on_each_gate(make_backend, case, reason):
    backend = make_backend(
        max_altitude_m=10.0,
        max_distance_m=50.0,
        with_video=case == "video",
    )
    backend.drone.flight_state = "landed"
    if case == "airborne":
        backend.drone.flight_state = "hovering"
    elif case == "link":
        backend.drone.connected = False
    elif case == "battery":
        backend.drone.battery_pct = 29.0
    elif case == "gps":
        backend.drone.gps_fixed = 0
    elif case == "source":
        backend.drone.source_ack = False
    elif case == "limit":
        backend._firmware_config_ok = True
        backend.drone.setting_states["MaxAltitudeChanged"]["current"] = 11.0

    assert not backend._takeoff_preflight()

    assert backend.state.preflight_ok is False
    assert reason in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events


def test_disabled_distance_geofence_does_not_require_gps(make_backend):
    backend = make_backend(
        max_altitude_m=10.0,
        max_distance_m=50.0,
        distance_geofence=False,
    )
    backend.drone.flight_state = "landed"
    backend.drone.gps_fixed = 0
    backend.drone.setting_states[
        "NoFlyOverMaxDistanceChanged"
    ]["shouldNotFlyOver"] = 0

    assert backend._takeoff_preflight()


def test_failed_pc_source_does_not_enable_pcmd_or_auto_state(make_backend):
    backend = make_backend()
    backend.pilot_sticks = True
    backend.state.tracker_state = "STICKS"
    backend.drone.source_ack = False

    backend.command("start_auto")

    assert backend.pilot_sticks
    assert backend.state.mode != "AUTO"
    assert backend.state.tracker_state == "SOURCE_FAIL"
    assert "PCMD" not in backend.drone.events


def test_stream_loss_zeros_pcmd_and_hands_back_to_skycontroller(make_backend):
    backend = make_backend()
    backend.pilot_sticks = False
    backend.state.mode = "AUTO"

    backend.stream_lost_hover("test stale")

    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.drone.source_requests[-1] == "SkyController"
    assert backend.pilot_sticks is True
    assert backend.state.mode == "MANUAL"
    assert backend.state.tracker_state == "STREAM_LOST_MANUAL"


def test_display_frame_lag_does_not_trigger_when_decoder_is_fresh(make_backend):
    backend = make_backend()
    backend.video_stream = SimpleNamespace(last_stamp=99.0)
    backend.grabber = SimpleNamespace(
        last_frame_age=lambda: 0.05,
        frame_pipeline_stats={"duplicate_run": 0, "frozen": False},
    )

    assert not backend._evaluate_video_health(100.1)
    assert backend.state.active_incident == ""


def test_decoder_frame_age_triggers_even_when_display_stamp_is_fresh(make_backend):
    backend = make_backend()
    backend.pilot_sticks = False
    backend.video_stream = SimpleNamespace(last_stamp=100.05)
    backend.grabber = SimpleNamespace(
        last_frame_age=lambda: 0.80,
        frame_pipeline_stats={"duplicate_run": 0, "frozen": False},
    )

    assert backend._evaluate_video_health(100.1)
    assert backend.state.active_incident == "stream_stale"


def test_stream_monitor_arms_only_after_first_decoded_frame(make_backend):
    backend = make_backend()
    backend.pilot_sticks = False
    backend.video_stream = SimpleNamespace(last_stamp=0.0)
    backend.grabber = SimpleNamespace(
        last_frame_age=lambda: None,
        frame_pipeline_stats={"duplicate_run": 0, "frozen": False},
    )

    assert not backend._evaluate_video_health(100.0)
    assert backend.state.active_incident == ""


def test_frozen_stream_health_triggers_manual_handoff_once(make_backend):
    backend = make_backend()
    backend.pilot_sticks = False
    backend.state.mode = "AUTO"
    backend.video_stream = SimpleNamespace(last_stamp=100.0)
    backend.grabber = SimpleNamespace(
        last_frame_age=lambda: 0.05,
        frame_pipeline_stats={"duplicate_run": 15, "frozen": True},
    )

    assert backend._evaluate_video_health(100.1)
    first_events = list(backend.drone.events)
    assert backend.pilot_sticks is True
    assert backend.state.active_incident == "stream_stale"

    assert not backend._evaluate_video_health(100.2)
    assert backend.drone.events == first_events


def test_emergency_stop_cancels_nudges_zeros_and_latches_manual(make_backend):
    backend = make_backend()
    backend.pilot_sticks = False
    backend._nudge_held.update({"前", "上"})

    result = backend.command("emergency_stop")

    assert result is True
    assert not backend._nudge_held
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.pilot_sticks is True
    assert backend.state.mode == "MANUAL"
    assert backend.state.tracker_state == "EMERGENCY_MANUAL"


def test_total_link_loss_requires_three_failed_polls_and_sends_no_host_commands(
    make_backend,
):
    backend = make_backend()
    backend.pilot_sticks = False
    backend._link_was_ok = True
    backend.drone.connected = False
    before = list(backend.drone.events)

    base_ns = time.monotonic_ns()
    for index in range(2):
        state = backend.poll(base_ns + index * 200_000_000)
        assert state.link_status == "DEGRADED"
        assert state.active_incident != "control_link_lost"
        assert backend.drone.events == before

    state = backend.poll(base_ns + 400_000_000)

    assert state.link_status == "LOST"
    assert state.mode == "MANUAL"
    assert state.tracker_state == "LINK_LOST_ONBOARD"
    assert state.active_incident == "control_link_lost"
    assert backend.pilot_sticks is True
    assert backend.drone.events == before


def test_skycontroller_drone_wifi_loss_uses_the_same_three_poll_debounce(
    make_backend,
):
    backend = make_backend()
    backend.pilot_sticks = False
    backend._link_was_ok = True
    backend.drone.connected = True
    backend.drone.managed_drone_connected = False
    before = list(backend.drone.events)
    base_ns = time.monotonic_ns()

    backend.poll(base_ns)
    backend.poll(base_ns + 200_000_000)
    state = backend.poll(base_ns + 400_000_000)

    assert state.link_status == "LOST"
    assert state.active_incident == "control_link_lost"
    assert backend.drone.events == before


def test_lost_link_policy_is_written_only_while_landed_and_read_back(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "landed"
    backend.drone.rth_auto_trigger_mode = "off"
    backend.drone.rth_delay_s = 10
    backend.drone.rth_ending_behavior = "hovering"

    assert backend._configure_lost_link_policy_if_safe()

    assert backend.drone.rth_auto_trigger_mode == "on"
    assert backend.drone.rth_delay_s == 1
    assert backend.drone.rth_ending_behavior == "landing"
    assert backend.state.rth_policy_configured is True


def test_lost_link_policy_write_is_refused_while_airborne(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "flying"

    assert not backend._configure_lost_link_policy_if_safe()
    assert "set_auto_trigger_mode" not in backend.drone.events
    assert backend.state.rth_policy_configured is False


def test_critical_battery_requests_rth_once_when_home_is_reachable(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.state.battery_pct = 10.0
    backend.state.drone_altitude_m = 5.0
    backend.state.max_altitude_m = 50.0
    backend.state.distance_from_home_m = 10.0
    backend.state.max_distance_m = 100.0

    assert backend._evaluate_runtime_safety()
    backend._runtime_safety_action_thread.join(timeout=1.0)

    assert "return_to_home" in backend.drone.events
    assert "Landing" not in backend.drone.events
    assert backend.state.active_incident == "battery_critical"
    assert backend._runtime_safety_action_latched
    before = list(backend.drone.events)
    assert not backend._evaluate_runtime_safety()
    assert backend.drone.events == before


def test_critical_battery_lands_when_home_is_not_reachable(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = False
    backend.state.flight_state = "flying"
    backend.state.battery_pct = 9.0

    assert backend._evaluate_runtime_safety()
    backend._runtime_safety_action_thread.join(timeout=1.0)

    assert "return_to_home" not in backend.drone.events
    assert "Landing" in backend.drone.events


def test_failed_rth_command_falls_back_to_in_place_landing(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.rth_success = False
    backend.state.flight_state = "flying"
    backend.state.battery_pct = 10.0

    assert backend._evaluate_runtime_safety()
    backend._runtime_safety_action_thread.join(timeout=1.0)

    assert "return_to_home" in backend.drone.events
    assert "Landing" in backend.drone.events


@pytest.mark.parametrize(
    ("altitude_m", "distance_m", "expected_reason"),
    (
        (49.0, 10.0, "altitude_limit"),
        (5.0, 95.0, "distance_limit"),
    ),
)
def test_runtime_geofence_uses_confirmed_proactive_margins(
    make_backend, altitude_m, distance_m, expected_reason,
):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.state.battery_pct = 90.0
    backend.state.drone_altitude_m = altitude_m
    backend.state.max_altitude_m = 50.0
    backend.state.distance_from_home_m = distance_m
    backend.state.max_distance_m = 100.0
    scheduled = []
    backend._schedule_runtime_safety_action = (
        lambda reason: scheduled.append(reason) or True
    )

    assert backend._evaluate_runtime_safety()
    assert scheduled == [expected_reason]


def test_manual_authority_blocks_hover_and_nudge_without_reclaim(make_backend):
    backend = make_backend()
    backend.pilot_sticks = True
    backend.state.tracker_state = "STICKS"

    assert not backend.hover_cmd("test")
    assert not backend.nudge_begin("前")

    assert backend.state.tracker_state == "STICKS"
    assert backend._nudge_held == set()
    assert backend.drone.events == []


def test_land_confirms_touchdown_before_state_and_media(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.recording_active = True

    def stop_recording(reason: str, *, download: bool = True) -> bool:
        backend.drone.events.append("media_stop")
        backend.recording_active = False
        return True

    backend.stop_flight_recording = stop_recording

    assert backend.land_cmd("test_land")

    events = backend.drone.events
    assert events.index("PCMD") < events.index("Landing") < events.index("media_stop")
    assert backend.drone.pcmds[0][1:5] == (0, 0, 0, 0)
    assert backend._landed
    assert backend.state.tracker_state == "LAND"


def test_unconfirmed_landing_is_not_reported_landed_and_defers_media(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.drone.landing_success = False
    backend.recording_active = True

    assert not backend.land_cmd("failed_land")

    assert not backend._landed
    assert backend.state.tracker_state == "LAND_UNCONFIRMED"
    assert backend.recording_active
    assert "media_stop" not in backend.drone.events


def test_cleanup_lands_before_media_and_disconnect(make_backend):
    backend = make_backend()
    drone = backend.drone
    drone.flight_state = "flying"
    backend.recording_active = True

    def stop_recording(reason: str, *, download: bool = True) -> bool:
        drone.events.append("media_stop")
        backend.recording_active = False
        return True

    backend.stop_flight_recording = stop_recording
    backend.cleanup()

    assert drone.events.index("PCMD") < drone.events.index("Landing")
    assert drone.events.index("Landing") < drone.events.index("media_stop")
    assert drone.events.index("media_stop") < drone.events.index("disconnect")
    assert backend._landed


def test_cleanup_finalizes_recording_without_synchronous_download(make_backend):
    backend = make_backend()
    drone = backend.drone
    drone.flight_state = "landed"
    backend.recording_active = True

    class NoCleanupDownload:
        def wait_for_pending_downloads(self, timeout=None):
            raise AssertionError("cleanup must not wait for media downloads")

        def __call__(self, _request):
            raise AssertionError("cleanup must not start a media download")

    drone.media = NoCleanupDownload()
    backend.cleanup()

    assert "stop_recording" in drone.events
    assert drone.events.index("stop_recording") < drone.events.index("disconnect")
    assert not backend.recording_active
    assert any(
        event == "record_saved"
        and fields.get("note") == "onboard_only_no_synchronous_download"
        for event, fields in backend.log.records
    )


def test_recording_download_waits_have_explicit_timeouts(
        make_backend, monkeypatch):
    backend = make_backend()
    pending_timeouts: list[float | None] = []
    download_timeouts: list[float | None] = []

    class TimedOutDownload:
        def wait(self, _timeout=None):
            download_timeouts.append(_timeout)
            return self

        def success(self) -> bool:
            return False

    class FakeMedia:
        download_dir = ""

        @staticmethod
        def last_media_id():
            return "media-1"

        def wait_for_pending_downloads(self, timeout=None):
            pending_timeouts.append(timeout)

        def __call__(self, request):
            assert request.name == "download_media"
            assert request.args == ("media-1",)
            return TimedOutDownload()

    backend.drone.media = FakeMedia()
    monkeypatch.setattr(backend_module.time, "sleep", lambda _seconds: None)

    assert backend._try_download_last_recording() == ""
    assert pending_timeouts == [backend_module._MEDIA_PENDING_TIMEOUT_S]
    assert download_timeouts == [backend_module._MEDIA_DOWNLOAD_TIMEOUT_S]
    assert any(
        event == "record_download" and not fields["ok"]
        for event, fields in backend.log.records
    )


def test_cleanup_does_not_claim_unconfirmed_touchdown(make_backend):
    backend = make_backend()
    drone = backend.drone
    drone.flight_state = "flying"
    drone.landing_success = False
    backend.recording_active = True

    backend.cleanup()

    assert not backend._landed
    assert backend.recording_active
    assert "Landing" in drone.events
    assert "media_stop" not in drone.events
    assert "disconnect" in drone.events


def test_nudge_heartbeat_deadman_zeros_stale_hold(make_backend):
    backend = make_backend(skycontroller=False, pulse_s=0.01)
    assert backend.nudge_pulse_s == pytest.approx(0.1)
    assert backend.nudge_begin("前")
    first_deadline = backend._nudge_deadline

    time.sleep(0.02)
    assert backend.nudge_heartbeat(["前"])
    assert backend._nudge_deadline > first_deadline
    extended_deadline = backend._nudge_deadline
    assert not backend.nudge_heartbeat(["後"])
    assert backend._nudge_deadline == extended_deadline

    deadline = time.monotonic() + 0.5
    while backend._nudge_held and time.monotonic() < deadline:
        time.sleep(0.01)
    thread = backend._nudge_loop_thread
    if thread is not None:
        thread.join(timeout=0.2)

    assert backend._nudge_held == set()
    assert backend.state.tracker_state == "HOVER"
    assert any(
        event == "nudge_deadman" and fields["ok"]
        for event, fields in backend.log.records
    )
    assert any(args[1:5] != (0, 0, 0, 0) for args in backend.drone.pcmds)
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)


def test_land_during_takeoff_wait_wins_epoch_and_prevents_recording(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.record_on_takeoff = True
    recording_starts: list[str] = []
    backend.start_flight_recording = lambda reason: recording_starts.append(reason) or True
    backend.drone.takeoff_wait_hook = lambda: backend.land_cmd("during_takeoff")

    assert not backend.takeoff_cmd()

    assert "Landing" in backend.drone.events
    assert backend._landed
    assert backend.state.tracker_state == "LAND"
    assert recording_starts == []


def test_land_before_takeoff_schedule_prevents_takeoff_command(
        make_backend, monkeypatch):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend._firmware_config_ok = True
    factory_entered = threading.Event()
    release_factory = threading.Event()
    piloting = sys.modules["olympe.messages.ardrone3.Piloting"]
    original_takeoff = piloting.TakeOff

    def blocked_takeoff_factory(*args, **kwargs):
        factory_entered.set()
        assert release_factory.wait(timeout=1.0)
        return original_takeoff(*args, **kwargs)

    monkeypatch.setattr(piloting, "TakeOff", blocked_takeoff_factory)
    results: list[bool] = []
    thread = threading.Thread(target=lambda: results.append(backend.takeoff_cmd()))
    thread.start()
    assert factory_entered.wait(timeout=1.0)

    assert backend.land_cmd("barrier_land_first")
    release_factory.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert results == [False]
    assert backend.drone.events.count("TakeOff") == 0
    assert backend.drone.source_requests[-1] == "SkyController"
    assert backend._landed
    assert backend.state.tracker_state == "LAND"


def test_takeoff_schedule_before_land_orders_landing_after_takeoff(make_backend):
    class BarrierDrone(_FakeDrone):
        def __init__(self):
            super().__init__()
            self.wait_entered = threading.Event()
            self.release_wait = threading.Event()

        def __call__(self, message):
            expectation = super().__call__(message)
            first = self._first(message)
            if first.name != "TakeOff":
                return expectation
            outer = self

            class BarrierExpectation:
                def wait(self, _timeout=None):
                    outer.wait_entered.set()
                    assert outer.release_wait.wait(timeout=1.0)
                    return expectation.wait(_timeout=_timeout)

            return BarrierExpectation()

    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone = BarrierDrone()
    backend.drone.flight_state = "landed"
    backend._firmware_config_ok = True
    results: list[bool] = []
    thread = threading.Thread(target=lambda: results.append(backend.takeoff_cmd()))
    thread.start()
    assert backend.drone.wait_entered.wait(timeout=1.0)

    backend.drone.flight_state = "takingoff"
    assert backend.land_cmd("barrier_takeoff_first")
    backend.drone.release_wait.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert results == [False]
    assert backend.drone.events.index("TakeOff") < backend.drone.events.index("Landing")
    assert backend.drone.source_requests[-1] == "SkyController"
    assert backend._landed
    assert backend.state.tracker_state == "LAND"


def test_axes_active_deadzone():
    assert not backend_module.axes_active({0: 0, 1: 100})
    assert not backend_module.axes_active({0: 2000})  # equal to default deadzone
    assert backend_module.axes_active({0: 2001})
    assert backend_module.axes_active({3: -5000, 0: 0})


def test_stick_monitor_reports_an_unexpected_device_disconnect(monkeypatch):
    disconnected = []
    monitor = backend_module.SkyControllerStickMonitor(
        lambda _axes: None,
        on_disconnect=disconnected.append,
        poll_s=0.001,
    )
    monitor._fd = 123
    monitor.healthy = True

    def unplugged(_fd, _size):
        raise OSError("device unplugged")

    monkeypatch.setattr(backend_module.os, "read", unplugged)
    monitor._loop()

    assert len(disconnected) == 1
    assert "device unplugged" in disconnected[0]
    assert monitor.healthy is False


def test_stick_monitor_disconnect_lands_when_the_control_link_still_works(
    make_backend,
):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"

    backend._on_stick_monitor_disconnect("device unplugged")
    backend._runtime_safety_action_thread.join(timeout=1.0)

    assert backend.state.stick_monitor_ok is False
    assert "Landing" in backend.drone.events
    assert backend.state.active_incident == "controller_disconnected"


def test_pc_control_is_refused_when_stick_monitor_is_unavailable(make_backend):
    backend = make_backend()
    backend._stick_monitor = None
    backend.state.stick_monitor_ok = False

    assert not backend.take_pc_control()
    assert backend.pilot_sticks is True
    assert backend.state.tracker_state == "STICK_MONITOR_FAIL"
    assert "Controller" not in backend.drone.source_requests


def test_takeoff_preflight_is_blocked_when_stick_monitor_is_unavailable(
    make_backend,
):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend._stick_monitor = None
    backend.state.stick_monitor_ok = False

    assert not backend._takeoff_preflight()
    assert "stick monitor" in backend.state.preflight_reason


def test_stick_override_reclaims_skycontroller_when_pc_controls(make_backend):
    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = False
    backend.state.tracker_state = "PC"
    backend.drone.source_state = "Controller"

    ok = backend._maybe_reclaim_from_sticks({0: 12000, 1: 0})

    assert ok
    assert backend.pilot_sticks is True
    assert backend.state.tracker_state == "STICKS"
    assert backend.state.last_command == "stick_override"
    assert backend.stick_override_count == 1
    assert backend.drone.source_requests[-1] == "SkyController"
    # PCMD was zeroed
    assert any(e == "PCMD" for e in backend.drone.events)
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)


def test_stick_override_noop_when_already_on_sticks(make_backend):
    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = True
    backend.state.tracker_state = "STICKS"
    before = list(backend.drone.source_requests)

    assert not backend._maybe_reclaim_from_sticks({0: 20000})
    assert backend.drone.source_requests == before
    assert backend.stick_override_count == 0


def test_stick_override_noop_on_direct_wifi(make_backend):
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = False

    assert not backend._maybe_reclaim_from_sticks({0: 20000})
    assert backend.pilot_sticks is False


def test_take_pc_control_refused_while_sticks_deflected(make_backend):
    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = True

    class _Mon:
        def is_active(self):
            return True

    backend._stick_monitor = _Mon()
    assert not backend.take_pc_control()
    assert backend.pilot_sticks is True
    assert backend.state.tracker_state == "STICKS"
    assert any(
        name == "pc_control" and fields.get("note") == "sticks_active_refuse_pc"
        for name, fields in backend.log.records
    )
