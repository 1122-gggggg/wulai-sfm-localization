from __future__ import annotations

import inspect
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import olympe_live_backend as backend_module
from backend_contract import (
    ControlAction,
    ControlRequest,
    ControlResult,
    FailureReason,
    InterfaceMode,
    MissionRoutePayload,
    SessionConfig,
)
from operator_state import TrackerState


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
        self.rth_min_altitude = 39.2
        self.setting_ack = {
            "MaxAltitude": True,
            "MaxDistance": True,
            "NoFlyOverMaxDistance": True,
            "MaxTilt": True,
            "MaxVerticalSpeed": True,
            "MaxRotationSpeed": True,
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
        self.controller_variant = "SkyController3"
        self.controller_calibration_ack = True
        self.event_markers: dict[str, int] = {}
        self.recording_mode_state = {
            0: {
                "cam_id": 0,
                "mode": SimpleNamespace(name="recording_mode.standard"),
                "resolution": SimpleNamespace(name="resolution.res_1080p"),
                "framerate": SimpleNamespace(name="framerate.fps_30"),
                "hyperlapse": SimpleNamespace(name="hyperlapse_value.ratio_15"),
                "bitrate": 50_000_000,
            }
        }
        self.recording_state = "inactive"
        self.set_recording_mode_ok = True



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
        if first.name in {"MaxAltitude", "MaxDistance", "NoFlyOverMaxDistance",
                          "MaxTilt", "MaxVerticalSpeed", "MaxRotationSpeed"}:
            state_name = first.name + "Changed"
            field = {
                "MaxAltitude": "current",
                "MaxDistance": "current",
                "NoFlyOverMaxDistance": "shouldNotFlyOver",
                "MaxTilt": "current",
                "MaxVerticalSpeed": "current",
                "MaxRotationSpeed": "current",
            }[first.name]
            argument = {
                "MaxAltitude": "current",
                "MaxDistance": "value",
                "NoFlyOverMaxDistance": "shouldNotFlyOver",
                "MaxTilt": "current",
                "MaxVerticalSpeed": "current",
                "MaxRotationSpeed": "current",
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
        if first.name == "set_min_altitude":
            return _Expectation(
                True,
                lambda: setattr(
                    self, "rth_min_altitude", first.kwargs["altitude"]
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
        if first.name == "set_recording_mode":
            def apply_recording_mode():
                if self.set_recording_mode_ok:
                    self.recording_mode_state[0] = {
                        "cam_id": 0,
                        "mode": SimpleNamespace(
                            name=f"recording_mode.{first.kwargs['mode']}",
                        ),
                        "resolution": SimpleNamespace(
                            name=f"resolution.{first.kwargs['resolution']}",
                        ),
                        "framerate": SimpleNamespace(
                            name=f"framerate.{first.kwargs['framerate']}",
                        ),
                        "hyperlapse": SimpleNamespace(
                            name=(
                                "hyperlapse_value."
                                f"{first.kwargs.get('hyperlapse', 'ratio_15')}"
                            ),
                        ),
                        "bitrate": 50_000_000,
                    }

            return _Expectation(self.set_recording_mode_ok, apply_recording_mode)
        if first.name == "start_recording":
            return _Expectation(
                True,
                lambda: setattr(self, "recording_state", "active"),
            )
        if first.name == "stop_recording":
            return _Expectation(
                True,
                lambda: setattr(self, "recording_state", "inactive"),
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
            # Real SkyController 3 fw 1.8.1 never emits this; None reproduces that.
            return (None if self.controller_variant is None
                    else {"variant": self.controller_variant})
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
        if name == "min_altitude":
            return {"current": self.rth_min_altitude, "min": 2.0, "max": 100.0}
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
        if name == "recording_mode":
            return dict(self.recording_mode_state)
        if name == "recording_state":
            return {"cam_id": 0, "state": self.recording_state}

        return {}

    def get_last_event(self, message_type):
        name = getattr(message_type, "_message_name", "")
        return SimpleNamespace(
            uuid=f"{name}:{self.event_markers.get(name, 1)}",
            args=self.get_state(message_type),
            date=datetime.now(timezone.utc),
        )

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



class _FakeStickMonitor:
    """Deterministic stand-in for the HID monitor.

    Without it, tests that assert the post-connect control state silently changed
    their answer depending on whether a SkyController happened to be plugged into
    the machine running the suite.
    """

    device_name = "Parrot Skycontroller3"
    healthy = True

    def __init__(self, on_active=None, on_disconnect=None, log_event=None):
        self.on_active = on_active
        self.on_disconnect = on_disconnect
        self.stopped = False

    def start(self):
        return True

    def stop(self):
        self.stopped = True

    def snapshot_axes(self):
        return {}


def _install_fake_stick_monitor(monkeypatch):
    monkeypatch.setattr(backend_module, "SkyControllerStickMonitor", _FakeStickMonitor)


def _install_fake_olympe(monkeypatch: pytest.MonkeyPatch) -> None:
    package_names = (
        "olympe",
        "olympe.messages",
        "olympe.messages.ardrone3",
        "olympe.messages.common",
        "olympe.messages.skyctrl",
        "olympe.messages.drone_manager",
        "olympe.features",
        "olympe.enums",
    )
    # olympe.messages has no directory -- module_loader.py generates it at import
    # time and inserts it straight into sys.modules. Shadowing a name that is NOT
    # there yet makes monkeypatch DELETE it on teardown, permanently, breaking
    # every later test that imports an olympe message. So import the real ones
    # FIRST (when a real olympe is present) to give monkeypatch something to
    # restore, then stub. Skipping the stub instead left the fakes uninstalled and
    # `from olympe.messages import rth` reached the real module, so the fake
    # return_to_home was never recorded.
    if "olympe" in sys.modules and getattr(sys.modules["olympe"], "__file__", None):
        import importlib

        for name in package_names:
            try:
                importlib.import_module(name)
            except Exception:
                pass
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
            "MaxTilt": _message_factory("MaxTilt"),
            "NoFlyOverMaxDistance": _message_factory("NoFlyOverMaxDistance"),
        },
        "olympe.messages.ardrone3.SpeedSettings": {
            "MaxVerticalSpeed": _message_factory("MaxVerticalSpeed"),
            "MaxRotationSpeed": _message_factory("MaxRotationSpeed"),
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
            "set_zoom_target": _message_factory("set_zoom_target"),
            "reset_zoom": _message_factory("reset_zoom"),
            "set_camera_mode": _message_factory("set_camera_mode"),
            "set_recording_mode": _message_factory("set_recording_mode"),
            "start_recording": _message_factory("start_recording"),
            "stop_recording": _message_factory("stop_recording"),
            "recording_state": _message_factory("recording_state"),
            "recording_mode": _message_factory("recording_mode"),
        },
        "olympe.messages.gimbal": {
            "set_target": _message_factory("set_target"),
        },
        "olympe.enums.camera": {
            "zoom_control_mode": SimpleNamespace(level="level"),
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
                "set_min_altitude",
                "min_altitude",
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
        # `from olympe.messages import rth` resolves the ATTRIBUTE on the parent,
        # not just the sys.modules entry. Without this link the import fell through
        # and _request_rth bailed before ever issuing return_to_home -- the RTH
        # tests then saw a Landing and blamed the flight logic.
        parent_name, _, leaf = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, leaf, module, raising=False)


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
            nudge_pct: int = backend_module.NUDGE_PCT,
            stream_loss_grace_s: float = backend_module.DEFAULT_STREAM_LOSS_GRACE_S,
            max_altitude_m: float | None = None,
            max_distance_m: float | None = None,
            distance_geofence: bool = True,
            min_takeoff_battery_pct: float = 30.0,
            require_gps_for_geofence: bool = True,
            with_video: bool = False):
        state = SimpleNamespace(
            loc="", stream="", mode="", tracker_state="", last_command="",
            zoom=1.0,
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
            nudge_pct=nudge_pct,
            stream_loss_grace_s=stream_loss_grace_s,
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


def test_legacy_takeoff_string_is_rejected_without_issuing_takeoff(make_backend):
    """Only a human-origin typed request may reach the real TakeOff path."""
    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"

    result = backend.command("takeoff")

    assert isinstance(result, ControlResult)
    assert not result.accepted
    assert result.reason_code == "TYPED_TAKEOFF_REQUIRED"
    assert "TakeOff" not in backend.drone.events


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


def test_preflight_does_not_gate_on_runtime_or_firmware_receipt_lists(make_backend):
    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend.approved_aircraft_firmware = frozenset({"different-version"})
    backend.approved_controller_firmware = frozenset({"different-version"})
    backend.approved_olympe_versions = frozenset({"different-version"})

    assert backend._takeoff_preflight(), backend.state.preflight_reason
    assert backend.connection_inventory["aircraft"]["software"] == "1.8.2"
    assert backend.connection_inventory["controller"]["software"] == "1.8.2"
    assert backend.connection_inventory["runtime"]["olympe_version"] == "8.4.0"
    assert not any(
        "approved receipt" in reason
        for reason in backend.connection_inventory["block_reasons"]
    )
    assert "TakeOff" not in backend.drone.events


def test_preflight_blocks_unhealthy_safety_log(make_backend):
    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend.log.healthy = False

    assert not backend._takeoff_preflight()
    assert "safety log" in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events


def test_critical_disk_pressure_blocks_new_takeoff_but_is_only_logged_in_flight(
    make_backend, monkeypatch: pytest.MonkeyPatch,
):
    """The documented <5 GiB/<5% contract is fail-closed before takeoff."""
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
    assert backend.state.disk_free_percent == pytest.approx(4.0)
    assert backend.state.disk_warning is True
    assert any(
        event == "disk_low_takeoff_blocked" and fields.get("reason") == "critical disk test"
        for event, fields in backend.log.records
    ), "takeoff was blocked without a durable incident"


def test_takeoff_battery_floor_is_fail_closed(make_backend):
    default = inspect.signature(
        backend_module.OlympeLiveBackend.__init__
    ).parameters["min_takeoff_battery_pct"].default
    assert default == pytest.approx(30.0)

    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.min_takeoff_battery_pct = 30.0
    backend.drone.flight_state = "landed"
    backend.drone.battery_pct = 29.0
    assert not backend._takeoff_preflight()
    assert backend.state.preflight_reason == (
        "battery 29% is below the 30% takeoff floor"
    )
    assert "TakeOff" not in backend.drone.events


@pytest.mark.parametrize("battery_pct", [None, float("nan")])
def test_unavailable_battery_is_fail_closed_during_preflight(
        make_backend, battery_pct):
    backend = make_backend(max_altitude_m=30.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend.drone.battery_pct = battery_pct

    assert not backend._takeoff_preflight()

    assert backend.state.preflight_reason == "battery state unavailable or invalid"
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


def test_low_disk_in_flight_is_reported_and_never_takes_command_away(
    make_backend, monkeypatch: pytest.MonkeyPatch,
):
    """Takeoff no longer blocks on low disk, so neither may the in-flight guard.

    Keeping the old trigger only moved the failure: the operator took off and had
    command authority torn away seconds later during climb-out. What low disk
    actually costs is recording, not control.
    """
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

    assert backend._check_runtime_storage_health(10.0) is True
    assert backend._runtime_storage_guard_latched is False, "the fail-safe latched"
    assert backend.state.active_incident != "disk_or_log_failure"
    assert backend.state.disk_warning is True
    assert any(
        event == "disk_low_inflight" and fields.get("free_percent") == 4.0
        for event, fields in backend.log.records
    ), "flying on a low disk with no record of it"


def test_unreadable_disk_in_flight_still_fails_safe(
    make_backend, monkeypatch: pytest.MonkeyPatch,
):
    """A disk that cannot be read at all means the safety log cannot be written."""
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.pilot_sticks = False

    def explode(_path):
        raise OSError("storage gone")

    monkeypatch.setattr(backend_module, "assess_disk_space", explode)

    assert backend._check_runtime_storage_health(10.0) is False
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.state.mode == "MANUAL"
    assert backend.state.active_incident == "disk_or_log_failure"


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
    _install_fake_stick_monitor(monkeypatch)
    monkeypatch.setattr(backend_module, "_DEFAULT_RECORD_DIR", tmp_path)
    monkeypatch.setattr(backend_module, "quiet_olympe_logs", lambda: None)
    drone = _FakeDrone()
    if skycontroller:
        # Firmware 1.8.1 leaves the Olympe variant state unavailable; inventory
        # must therefore see the HID monitor that _connect starts.
        drone.controller_variant = None
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
        approved_aircraft_firmware=("1.8.2",),
        approved_controller_firmware=("1.8.2",),
        approved_olympe_versions=("8.4.0",),
        with_video=False,
    )

    if skycontroller:
        assert drone.source_state == "SkyController"
        assert "setPilotingSource" in drone.events
        assert backend._inventory_takeoff_ready
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


def test_unset_limits_are_advisory_and_do_not_block_takeoff(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "landed"

    assert not backend._configure_firmware_limits_if_safe()
    assert backend.takeoff_cmd()

    assert backend.drone is not None
    assert backend.state.preflight_ok
    assert "TakeOff" in backend.drone.events
    assert not any(event.startswith("Max") for event in backend.drone.events)
    assert any(
        event == "takeoff_advisory"
        and any("max altitude and max distance are unset" in warning
                for warning in fields["warnings"])
        for event, fields in backend.log.records
    )


def test_firmware_limits_write_only_when_landed_and_confirm_readback(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "flying"

    assert not backend._configure_firmware_limits_if_safe()
    assert "MaxAltitude" not in backend.drone.events

    backend.drone.flight_state = "landed"
    assert backend._configure_firmware_limits_if_safe()

    # The speed envelope (tilt / vertical / rotation) is pinned and read back too:
    # every operator command is a percentage of those three.
    assert backend.drone.events[-6:] == [
        "MaxAltitude", "MaxDistance", "MaxTilt",
        "MaxVerticalSpeed", "MaxRotationSpeed", "NoFlyOverMaxDistance",
    ]
    assert backend.state.max_altitude_m == pytest.approx(20.0)
    assert backend.state.max_distance_m == pytest.approx(80.0)
    assert backend.state.max_tilt_deg == pytest.approx(
        backend_module.DEFAULT_MAX_TILT_DEG)
    assert backend.state.max_vertical_speed_mps == pytest.approx(
        backend_module.DEFAULT_MAX_VERTICAL_SPEED_MS)
    assert backend.state.max_rotation_speed_dps == pytest.approx(
        backend_module.DEFAULT_MAX_ROTATION_SPEED_DEGS)
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
    assert backend.drone.events[-6:] == [
        "MaxAltitude", "MaxDistance", "MaxTilt",
        "MaxVerticalSpeed", "MaxRotationSpeed", "NoFlyOverMaxDistance",
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
        "auto_speed_limit_apply", speed_limit_mps=0.2, enabled=False
    ) is True
    assert backend.state.autonomous_speed_limit_mps == pytest.approx(0.2)
    assert backend.state.autonomous_speed_limit_enabled is False
    assert backend.state.autonomous_speed_guard_status == "SPEED_LIMIT_DISABLED"
    assert backend.state.autonomous_approval_valid is False
    assert backend.state.autonomous_locked is True

    backend.drone.flight_state = "flying"
    assert backend.command(
        "auto_speed_limit_apply", speed_limit_mps=0.1, enabled=True
    ) is False
    assert backend.state.autonomous_speed_limit_mps == pytest.approx(0.2)
    assert backend.state.autonomous_speed_limit_enabled is False


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


def test_firmware_setting_timeout_is_bounded_and_fail_closed(
        make_backend, monkeypatch):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"
    wait_timeouts = []

    class NeverCompletes:
        def wait(self, _timeout=None):
            wait_timeouts.append(_timeout)
            if _timeout is None:
                raise AssertionError("firmware expectation wait must be bounded")
            return self

        def success(self) -> bool:
            return False

    original_call = type(backend.drone).__call__

    def never_complete_altitude(self, message):
        first = self._first(message)
        if first.name == "MaxAltitude":
            self.events.append(first.name)
            return NeverCompletes()
        return original_call(self, message)

    monkeypatch.setattr(type(backend.drone), "__call__", never_complete_altitude)

    assert not backend._configure_firmware_limits_if_safe()
    assert wait_timeouts == [backend_module._FIRMWARE_SETTING_TIMEOUT_S]
    assert "failed or timed out" in backend.state.preflight_reason
    assert "MaxDistance" not in backend.drone.events
    assert not backend._firmware_config_ok


def test_firmware_geofence_failure_is_bounded_and_fail_closed(
        make_backend, monkeypatch):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"
    wait_timeouts = []

    class FailedExpectation:
        def wait(self, _timeout=None):
            wait_timeouts.append(_timeout)
            return self

        def success(self) -> bool:
            return False

    original_call = type(backend.drone).__call__

    def fail_geofence(self, message):
        first = self._first(message)
        if first.name == "NoFlyOverMaxDistance":
            self.events.append(first.name)
            return FailedExpectation()
        return original_call(self, message)

    monkeypatch.setattr(type(backend.drone), "__call__", fail_geofence)

    assert not backend._configure_firmware_limits_if_safe()
    assert wait_timeouts == [backend_module._FIRMWARE_SETTING_TIMEOUT_S]
    assert "NoFlyOverMaxDistance command failed or timed out" in (
        backend.state.preflight_reason
    )
    assert not backend._firmware_config_ok


def test_firmware_command_exception_stops_remaining_writes(
        make_backend, monkeypatch):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"
    original_call = type(backend.drone).__call__

    def explode_on_altitude(self, message):
        first = self._first(message)
        if first.name == "MaxAltitude":
            raise RuntimeError("synthetic firmware write failure")
        return original_call(self, message)

    monkeypatch.setattr(type(backend.drone), "__call__", explode_on_altitude)

    assert not backend._configure_firmware_limits_if_safe()

    assert "MaxAltitude command failed" in backend.state.preflight_reason
    assert "MaxDistance" not in backend.drone.events
    assert not backend._firmware_config_ok


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
    # Pre-match everything the backend writes BEFORE the geofence, so the geofence
    # is the only readback that can mismatch and this test keeps testing the geofence.
    states = backend.drone.setting_states
    states["MaxTiltChanged"]["current"] = backend_module.DEFAULT_MAX_TILT_DEG
    states["MaxVerticalSpeedChanged"]["current"] = (
        backend_module.DEFAULT_MAX_VERTICAL_SPEED_MS)
    states["MaxRotationSpeedChanged"]["current"] = (
        backend_module.DEFAULT_MAX_ROTATION_SPEED_DEGS)
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


def test_poll_does_not_refresh_speed_age_without_a_new_olympe_event(
    make_backend,
) -> None:
    backend = make_backend()
    first_ns = time.monotonic_ns()

    backend.poll(now_mono_ns=first_ns)
    first_stamp = backend.state.ground_speed_mono_ns

    backend._last_telemetry_t = 0.0
    backend.poll(now_mono_ns=first_ns + 1_000_000_000)

    assert backend.state.ground_speed_mono_ns == first_stamp

    backend.drone.event_markers["SpeedChanged"] = 2
    backend._last_telemetry_t = 0.0
    backend.poll(now_mono_ns=first_ns + 2_000_000_000)

    assert backend.state.ground_speed_mono_ns > first_stamp


def test_future_telemetry_event_does_not_replace_last_valid_state(make_backend):
    backend = make_backend()
    old_state = {"speedX": 0.18, "speedY": 0.24, "speedZ": -0.04}
    backend.telemetry_freshness.observe(
        "ground_speed", old_state, marker="valid-event", observed_mono_ns=100
    )
    future_event = SimpleNamespace(
        uuid="future-event",
        args={"speedX": 99.0, "speedY": 99.0, "speedZ": 99.0},
        date=SimpleNamespace(timestamp=lambda: time.time() + 10.0),
    )
    backend.drone.get_last_event = lambda _message: future_event

    observed = backend._observe_telemetry(
        "ground_speed",
        object(),
        old_state,
        now_mono_ns=200,
    )

    assert observed == old_state
    sample = backend.telemetry_freshness.sample("ground_speed")
    assert sample is not None
    assert sample.value == old_state
    assert sample.marker == "valid-event"


@pytest.mark.parametrize(
    ("name", "key", "future_value"),
    [
        ("battery", "percent", 2.0),
        ("altitude", "altitude", 99.0),
    ],
)
def test_first_future_critical_telemetry_is_rejected(
        make_backend, name, key, future_value):
    backend = make_backend()
    future_state = {key: future_value}
    future_event = SimpleNamespace(
        uuid=f"future-{name}",
        args=future_state,
        date=SimpleNamespace(timestamp=lambda: time.time() + 10.0),
    )
    backend.drone.get_last_event = lambda _message: future_event

    observed = backend._observe_telemetry(
        name,
        object(),
        future_state,
        now_mono_ns=time.monotonic_ns(),
    )

    assert observed == {}
    assert backend.telemetry_freshness.sample(name) is None


def test_stale_altitude_telemetry_does_not_trigger_runtime_safety(
    make_backend,
) -> None:
    backend = make_backend(distance_geofence=False)
    first_ns = time.monotonic_ns()
    backend.drone.flight_state = "landed"
    backend.poll(now_mono_ns=first_ns)
    reasons = []
    backend._schedule_runtime_safety_action = (
        lambda reason: reasons.append(reason) or True
    )

    backend.drone.flight_state = "flying"
    backend.drone.event_markers["BatteryStateChanged"] = 2
    backend._last_telemetry_t = 0.0
    backend.poll(now_mono_ns=first_ns + 2_000_000_000)

    assert reasons == []
    assert backend.state.distance_guard_active is None


def test_unchanged_battery_event_does_not_fail_runtime_telemetry(make_backend) -> None:
    backend = make_backend(distance_geofence=False)
    now_ns = time.monotonic_ns()
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.telemetry_freshness.observe(
        "battery",
        {"percent": 71.0},
        marker="battery-unchanged",
        observed_mono_ns=now_ns - 30_000_000_000,
    )
    backend.telemetry_freshness.observe(
        "altitude",
        {"altitude": 1.0},
        marker="altitude-fresh",
        observed_mono_ns=now_ns,
    )
    reasons = []
    backend._schedule_runtime_safety_action = (
        lambda reason: reasons.append(reason) or True
    )

    assert not backend._evaluate_runtime_safety(now_ns)
    assert reasons == []


def test_unavailable_battery_does_not_trigger_runtime_safety(make_backend) -> None:
    backend = make_backend(distance_geofence=False)
    now_ns = time.monotonic_ns()
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.state.battery_pct = -1.0
    backend.telemetry_freshness.observe(
        "altitude",
        {"altitude": 1.0},
        marker="altitude-fresh",
        observed_mono_ns=now_ns,
    )
    reasons = []
    backend._schedule_runtime_safety_action = (
        lambda reason: reasons.append(reason) or True
    )

    assert not backend._evaluate_runtime_safety(now_ns)
    assert reasons == []


def test_critical_battery_does_not_trigger_runtime_safety(make_backend) -> None:
    backend = make_backend(distance_geofence=False)
    now_ns = time.monotonic_ns()
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.telemetry_freshness.observe(
        "battery",
        {"percent": 9.0},
        marker="battery-unchanged",
        observed_mono_ns=now_ns - 30_000_000_000,
    )
    backend.telemetry_freshness.observe(
        "altitude",
        {"altitude": 1.0},
        marker="altitude-fresh",
        observed_mono_ns=now_ns,
    )
    reasons = []
    backend._schedule_runtime_safety_action = (
        lambda reason: reasons.append(reason) or True
    )

    assert not backend._evaluate_runtime_safety(now_ns)
    assert reasons == []


def test_stale_ground_telemetry_does_not_interrupt_takeoff(make_backend) -> None:
    backend = make_backend(distance_geofence=False)
    first_ns = time.monotonic_ns()
    backend.drone.flight_state = "landed"
    backend.poll(now_mono_ns=first_ns)
    reasons = []
    backend._schedule_runtime_safety_action = (
        lambda reason: reasons.append(reason) or True
    )

    backend.drone.flight_state = "takingoff"
    backend._last_telemetry_t = 0.0
    backend.poll(now_mono_ns=first_ns + 2_000_000_000)

    assert reasons == []


def test_stale_distance_event_disables_geofence_without_blocking_no_gps_flight(
    make_backend,
) -> None:
    backend = make_backend(
        max_altitude_m=30.0,
        max_distance_m=100.0,
        distance_geofence=True,
    )
    now_ns = time.monotonic_ns()
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.state.link_ok = True
    backend.state.max_distance_m = 100.0
    backend.state.distance_from_home_m = 99.0
    backend.telemetry_freshness.observe(
        "battery",
        {"percent": 100.0},
        marker="battery-new",
        observed_mono_ns=now_ns,
    )
    backend.telemetry_freshness.observe(
        "altitude",
        {"altitude": 1.0},
        marker="altitude-new",
        observed_mono_ns=now_ns,
    )
    backend.telemetry_freshness.observe(
        "distance",
        {"distance": 99.0},
        marker="location-old",
        observed_mono_ns=now_ns - 2_000_000_000,
    )
    reasons = []
    backend._schedule_runtime_safety_action = (
        lambda reason: reasons.append(reason) or True
    )

    assert not backend._evaluate_runtime_safety(now_ns)

    assert reasons == []
    assert backend.state.distance_guard_active is False


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


def test_typed_live_control_request_rejects_stale_normal_action_but_not_safety(
        make_backend):
    backend = make_backend()
    stale_ns = time.monotonic_ns() - backend_module.CONTROL_REQUEST_MAX_AGE_NS - 1

    stale_hover = backend.command(ControlRequest(
        action=ControlAction.HOVER,
        submitted_mono_ns=stale_ns,
        human_origin=True,
    ))
    assert not stale_hover.accepted
    assert stale_hover.reason_code == "STALE_CONTROL_REQUEST"
    assert "PCMD" not in backend.drone.events

    stale_land = backend.command(ControlRequest(
        action=ControlAction.LAND_NOW,
        submitted_mono_ns=stale_ns,
        human_origin=True,
    ))
    assert stale_land.accepted
    assert stale_land.executed
    assert "Landing" in backend.drone.events

    stale_emergency = backend.command(ControlRequest(
        action=ControlAction.EMERGENCY_STOP,
        submitted_mono_ns=stale_ns,
        human_origin=False,
    ))
    assert stale_emergency.accepted
    assert stale_emergency.executed


@pytest.mark.parametrize(
    ("action", "method_name", "stale"),
    [
        (ControlAction.TAKEOFF, "takeoff_cmd", False),
        (ControlAction.LAND_NOW, "land_cmd", True),
    ],
)
def test_typed_live_command_exception_is_rejected_with_context(
        make_backend, monkeypatch, action, method_name, stale):
    backend = make_backend()
    backend.state.active_incident = "existing_incident"
    backend.state.tracker_state = "TEST_CONTEXT"
    submitted_mono_ns = time.monotonic_ns()
    if stale:
        submitted_mono_ns -= backend_module.CONTROL_REQUEST_MAX_AGE_NS + 1
    request = ControlRequest(
        action=action,
        submitted_mono_ns=submitted_mono_ns,
        human_origin=True,
    )

    def raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(backend, method_name, raise_backend_error)

    result = backend.command(request)

    assert isinstance(result, ControlResult)
    assert not result.accepted
    assert not result.executed
    assert result.reason_code == "BACKEND_EXCEPTION"
    assert result.resulting_state is backend.state
    assert backend.state.active_incident == "existing_incident"
    exception_records = [
        fields
        for event, fields in backend.log.records
        if event == "typed_command_exception"
    ]
    assert exception_records
    assert exception_records[-1]["request_id"] == request.request_id
    assert exception_records[-1]["action"] == action.value
    assert exception_records[-1]["active_incident"] == "existing_incident"
    assert exception_records[-1]["error"] == "RuntimeError('synthetic backend failure')"
    result_records = [
        fields
        for event, fields in backend.log.records
        if event == "control_result"
    ]
    assert result_records[-1]["accepted"] is False
    assert result_records[-1]["executed"] is False


def test_typed_live_command_mock_logs_request_and_result(make_backend) -> None:
    backend = make_backend()
    backend.log.event = Mock()

    result = backend.command(
        ControlRequest.create(ControlAction.START_LOCALIZATION, human_origin=True)
    )

    assert result.accepted and result.executed
    events = [call.args[0] for call in backend.log.event.call_args_list]
    assert events == ["control_request", "control_result"]


def test_legacy_nudge_command_mock_preserves_release_return(make_backend) -> None:
    backend = make_backend()
    backend.nudge_begin = Mock(return_value=True)
    backend.nudge_end = Mock(return_value=None)

    assert backend.command("nudge_begin", dir="前") is True
    assert backend.command("nudge_end", dir="前") is None
    backend.nudge_begin.assert_called_once_with("前")
    backend.nudge_end.assert_called_once_with("前")


def test_nudge_loop_mock_failure_releases_hold_and_sends_zero(make_backend) -> None:
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend._combined_nudge_pcmd = Mock(return_value=(backend.nudge_pct, 0, 0, 0))
    entered = threading.Event()
    release = threading.Event()

    def fail_after_observed(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=2.0)
        raise ConnectionError("mock hold failure")

    backend.send_pcmd = Mock(side_effect=fail_after_observed)
    backend._raw_pcmd = Mock()
    with backend._lock:
        backend._nudge_held.add("前")
        backend._nudge_deadline = time.monotonic() + 5.0

    backend._ensure_nudge_loop()
    thread = backend._nudge_loop_thread
    assert thread is not None
    assert entered.wait(timeout=1.0)
    release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert backend._nudge_held == set()
    backend._raw_pcmd.assert_called_once_with(0, 0, 0, 0)
    assert backend.state.last_command == "nudge_loop_error_hover"


def test_typed_land_reports_unconfirmed_touchdown_as_rejected(make_backend) -> None:
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.drone.landing_success = False

    result = backend.command(
        ControlRequest.create(ControlAction.LAND, human_origin=True)
    )

    assert not result.accepted
    assert not result.executed
    assert result.reason_code == "BACKEND_REJECTED"
    assert result.raw_result is False

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
        payload=MissionRoutePayload(
            route_path="/tmp/selected-route.json",
            route_sha256="a" * 64,
            site_id="field-a",
            coordinate_frame_id="glomap-a",
        ),
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


def test_mock_takeoff_success_records_only_after_hover_confirmation(
        make_backend, monkeypatch):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.record_on_takeoff = True
    recording_starts: list[str] = []
    monkeypatch.setattr(
        backend,
        "start_flight_recording",
        lambda reason: recording_starts.append(reason) or True,
    )

    assert backend.takeoff_cmd()

    assert backend.drone.events.index("TakeOff") < backend.drone.events.index("PCMD")
    assert recording_starts == ["post_takeoff"]
    assert backend.state.tracker_state == "HOVER"
    assert backend._maneuver_in_progress is None


def test_mock_takeoff_preflight_failure_never_schedules_takeoff(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.connected = False

    assert not backend.takeoff_cmd()

    assert "TakeOff" not in backend.drone.events
    assert backend.state.tracker_state == "TAKEOFF_BLOCKED"
    assert backend._maneuver_in_progress is None


def test_mock_takeoff_exception_after_schedule_uses_landing_fallback(
        make_backend, monkeypatch):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"

    def explode_after_hover(*_args, **_kwargs):
        raise RuntimeError("synthetic post-takeoff failure")

    monkeypatch.setattr(backend, "send_pcmd", explode_after_hover)

    assert not backend.takeoff_cmd()

    assert backend.drone.events.index("TakeOff") < backend.drone.events.index(
        "Landing"
    )
    assert backend._landed
    assert backend.drone.source_requests[-1] == "SkyController"
    assert backend._maneuver_in_progress is None
    assert any(
        event == "takeoff" and fields.get("error") ==
        "RuntimeError('synthetic post-takeoff failure')"
        for event, fields in backend.log.records
    )


def test_limit_mismatch_blocks_takeoff_preflight(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.pilot_sticks = True
    backend._firmware_config_ok = True
    backend.drone.setting_states["MaxAltitudeChanged"]["current"] = 11.0

    assert not backend._takeoff_preflight()

    assert backend.state.preflight_reason == (
        "MaxAltitude readback mismatch: requested=10 actual=11.0"
    )
    assert backend.drone.source_state == "SkyController"
    assert "TakeOff" not in backend.drone.events


def test_firmware_configuration_failure_blocks_takeoff_preflight(make_backend):
    backend = make_backend(max_altitude_m=20.0, max_distance_m=80.0)
    backend.drone.flight_state = "landed"
    backend.drone.setting_ack["MaxAltitude"] = False

    assert not backend._takeoff_preflight()

    assert "firmware limit configuration retry failed" in (
        backend.state.preflight_reason
    )
    assert "MaxAltitude command failed or timed out" in (
        backend.state.preflight_reason
    )
    assert "TakeOff" not in backend.drone.events


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("airborne", "confirmed landed"),
        ("link", "healthy Olympe link"),
        ("video", "live video frame"),
        ("battery", "takeoff floor"),
        ("source", "source command"),
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
    elif case == "source":
        backend.drone.source_ack = False

    assert not backend._takeoff_preflight()

    assert backend.state.preflight_ok is False
    assert reason in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events


def test_gps_fix_required_by_geofence_blocks_takeoff(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.gps_fixed = 0

    assert not backend._takeoff_preflight()
    assert backend.state.gps_fixed is False
    assert backend.state.preflight_reason == (
        "GPS fix unavailable; required by configured distance geofence"
    )
    assert "TakeOff" not in backend.drone.events


def test_gps_fix_is_advisory_when_geofence_policy_allows_takeoff(make_backend):
    backend = make_backend(
        max_altitude_m=10.0,
        max_distance_m=50.0,
        require_gps_for_geofence=False,
    )
    backend.drone.flight_state = "landed"
    backend.drone.gps_fixed = 0

    assert backend._takeoff_preflight()
    assert backend.state.gps_fixed is False
    assert any(
        event == "takeoff_advisory"
        and "GPS fix unavailable; takeoff remains allowed" in fields["warnings"]
        for event, fields in backend.log.records
    )


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
    assert any(
        event == "takeoff_advisory"
        and "GPS fix unavailable; takeoff remains allowed" in fields["warnings"]
        for event, fields in backend.log.records
    )


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


def test_stream_loss_zeros_pcmd_and_keeps_pc_control(make_backend):
    backend = make_backend()
    backend.pilot_sticks = False
    backend.state.mode = "AUTO"

    backend.stream_lost_hover("test stale")

    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.drone.source_requests == []
    assert backend.pilot_sticks is False
    assert backend.state.mode == "AUTO"
    assert backend.state.tracker_state == "STREAM_LOST_HOVER"


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
    # grace 0: this test is about WHICH clock decides staleness (decoder age, not
    # display stamp), not about how long the outage must persist.
    backend = make_backend(stream_loss_grace_s=0.0)
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


def test_frozen_stream_health_triggers_hover_once(make_backend):
    # grace 0 = the historical immediate trigger; the grace window itself is
    # covered by test_brief_stream_latency_does_not_take_pc_control_away.
    backend = make_backend(stream_loss_grace_s=0.0)
    backend.pilot_sticks = False
    backend.state.mode = "AUTO"
    backend.video_stream = SimpleNamespace(last_stamp=100.0)
    backend.grabber = SimpleNamespace(
        last_frame_age=lambda: 0.05,
        frame_pipeline_stats={"duplicate_run": 15, "frozen": True},
    )

    assert backend._evaluate_video_health(100.1)
    first_events = list(backend.drone.events)
    assert backend.pilot_sticks is False
    assert backend.state.active_incident == "stream_stale"

    assert not backend._evaluate_video_health(100.2)
    assert backend.drone.events == first_events


def test_rejected_zoom_is_not_recorded_as_applied(make_backend):
    """state.zoom gates localization, so a zoom the camera never applied would
    silently pause localization while the real camera is still at 1.0x."""
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = False
    assert backend.state.zoom == pytest.approx(1.0)

    original_call = type(backend.drone).__call__

    def reject(self, message):
        if getattr(backend.drone._first(message), "name", "") == "set_zoom_target":
            raise RuntimeError("camera refused zoom")
        return original_call(self, message)

    type(backend.drone).__call__ = reject
    try:
        backend.set_zoom(2.5)
    finally:
        type(backend.drone).__call__ = original_call

    assert backend.state.zoom == pytest.approx(1.0), (
        "a rejected zoom was recorded as applied and would pause localization"
    )
    assert any(
        event == "zoom" and not fields["ok"] for event, fields in backend.log.records
    )


def test_zoom_blocked_by_sticks_is_not_recorded_as_applied(make_backend):
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = True
    assert not backend.set_zoom(3.0)
    assert backend.state.zoom == pytest.approx(1.0)


def test_camera_setters_and_reset_report_real_application(make_backend):
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = False

    assert backend.set_gimbal_pitch(-30.0)
    assert backend.set_zoom(2.0)
    assert backend.reset_camera_defaults()
    assert backend.state.gimbal_pitch_deg == pytest.approx(-20.0)
    assert backend.state.zoom == pytest.approx(1.0)

    backend.pilot_sticks = True
    assert not backend.set_gimbal_pitch(-10.0)
    assert not backend.reset_camera_defaults()


@pytest.mark.parametrize(
    "terminal", ["TAKEOFF_FAIL", "TAKEOFF_BLOCKED", "SOURCE_FAIL", "LAND_UNCONFIRMED"],
)
def test_telemetry_poll_does_not_erase_a_terminal_state(make_backend, terminal):
    """The flying-state poll used to overwrite abnormal outcomes on the next tick,
    erasing the only on-screen record that something failed."""
    backend = make_backend()
    backend.drone.flight_state = "hovering"
    backend.state.tracker_state = terminal

    backend.poll()

    assert backend.state.tracker_state == terminal


def test_distance_geofence_reports_itself_inactive_without_gps(make_backend):
    """With no GPS position/Home the distance check evaluates nothing.

    The operator must not read "geofence enabled" as "distance containment active",
    and missing GPS must neither block takeoff nor force a landing.
    """
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.state.battery_pct = 80.0
    backend.state.max_distance_m = 100.0
    backend.state.distance_from_home_m = None          # no GPS fix / no home
    scheduled = []
    def schedule(reason):
        scheduled.append(reason)
        backend.state.active_incident = reason
        return True

    backend._schedule_runtime_safety_action = schedule

    assert not backend._evaluate_runtime_safety()
    assert backend.state.distance_guard_active is False
    assert scheduled == []
    assert any(
        event == "distance_guard" and fields["active"] is False
        for event, fields in backend.log.records
    )

    # GPS comes back -> the guard reports itself active again and still triggers.
    backend.state.distance_from_home_m = 10.0
    assert not backend._evaluate_runtime_safety()
    assert backend.state.distance_guard_active is True

    backend.state.distance_from_home_m = 999.0
    assert backend._evaluate_runtime_safety()
    assert backend.state.active_incident == "distance_limit"


def test_speed_envelope_is_pinned_and_read_back_not_inherited(make_backend):
    """MaxTilt / MaxVerticalSpeed / MaxRotationSpeed decide how fast every operator
    command actually moves the aircraft. They used to be READ but never WRITTEN, so
    the real speed was whatever the last FreeFlight session left behind."""
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    # Aircraft arrives with a completely different (much faster) envelope.
    states = backend.drone.setting_states
    states["MaxTiltChanged"]["current"] = 40.0
    states["MaxVerticalSpeedChanged"]["current"] = 4.0
    states["MaxRotationSpeedChanged"]["current"] = 200.0

    assert backend._configure_firmware_limits_if_safe()

    assert states["MaxTiltChanged"]["current"] == pytest.approx(
        backend_module.DEFAULT_MAX_TILT_DEG)
    assert states["MaxVerticalSpeedChanged"]["current"] == pytest.approx(
        backend_module.DEFAULT_MAX_VERTICAL_SPEED_MS)
    assert states["MaxRotationSpeedChanged"]["current"] == pytest.approx(
        backend_module.DEFAULT_MAX_ROTATION_SPEED_DEGS)
    assert any(
        event == "firmware_limits" and fields.get("ok")
        and fields.get("max_tilt_deg") == pytest.approx(
            backend_module.DEFAULT_MAX_TILT_DEG)
        for event, fields in backend.log.records
    )


def test_speed_envelope_readback_mismatch_is_fail_closed(make_backend):
    """If the aircraft refuses the envelope, takeoff must not proceed on its own."""
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.setting_ack["MaxTilt"] = False

    assert not backend._configure_firmware_limits_if_safe()
    assert "MaxTilt" in backend.state.preflight_reason
    assert not backend._firmware_config_ok


@pytest.mark.parametrize("field,bad", [
    ("max_tilt_deg", 0.0),
    ("max_tilt_deg", float("nan")),
    ("max_vertical_speed_ms", -1.0),
    ("max_rotation_speed_degs", 0.0),
])
def test_invalid_speed_envelope_is_rejected(make_backend, field, bad):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    setattr(backend, {
        "max_tilt_deg": "desired_max_tilt_deg",
        "max_vertical_speed_ms": "desired_max_vertical_speed_ms",
        "max_rotation_speed_degs": "desired_max_rotation_speed_degs",
    }[field], bad)

    assert not backend._configure_firmware_limits_if_safe()
    assert not backend._firmware_config_ok


def test_uncalibrated_skycontroller_compass_warns_but_does_not_block_takeoff(
        make_backend):
    """Operator decision 2026-08-06: the CONTROLLER compass only feeds
    pilot-referenced features (RTH preferred_home_type="pilot", follow-me,
    operator-relative modes). This system uses none of them -- it never calls
    set_preferred_home_type and flies body-frame PCMD -- so an uncalibrated
    controller compass must not block takeoff. It must still be logged.
    """
    backend = make_backend(skycontroller=True)
    backend.drone.drone_calibration_required = 0
    backend.drone.controller_calibration_state = "NotCalibrated"

    assert backend._magnetometer_control_error() is None
    assert any(
        event == "magnetometer_calibration_warning"
        and fields.get("target") == "skycontroller"
        for event, fields in backend.log.records
    )


def test_skycontroller_calibration_in_progress_still_blocks(make_backend):
    """A calibration actively running means the controller is being rotated by
    hand -- it is not in a state to pilot anything."""
    backend = make_backend(skycontroller=True)
    backend.drone.drone_calibration_required = 0
    backend.drone.controller_calibration_state = "CalibratingY"

    error = backend._magnetometer_control_error()
    assert error is not None and "in progress" in error


def test_aircraft_compass_gate_is_unchanged(make_backend):
    """Relaxing the CONTROLLER gate must not relax the AIRCRAFT one."""
    backend = make_backend(skycontroller=True)
    backend.drone.controller_calibration_state = "Calibrated"
    backend.drone.drone_calibration_required = 1

    error = backend._magnetometer_control_error()
    assert error is not None and "aircraft magnetometer" in error


def test_camera_axes_do_not_seize_flight_control(make_backend):
    """Moving the gimbal wheel is a CAMERA action, not a request for the aircraft.

    Measured on a SkyController 3 v1.8.1: axes 0-3 are the two flight sticks,
    axes 4-5 are the camera/gimbal wheel and shoulder controls. Counting 4-5 made
    every camera adjustment yank flight authority back from the PC.
    """
    # Camera axes alone, at full deflection, must not look like stick input.
    assert not backend_module.axes_active({4: 28714, 5: -28714})
    # Flight sticks still do.
    for axis in backend_module._STICK_FLIGHT_AXES:
        assert backend_module.axes_active({axis: 28714}), f"axis {axis} ignored"
    # Below the deadzone nothing counts.
    assert not backend_module.axes_active({0: 1999, 1: -1999})
    # Opting out of the filter restores the old any-axis behaviour.
    assert backend_module.axes_active({4: 28714}, flight_axes=None)

    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = False
    assert not backend._maybe_reclaim_from_sticks({4: 28714, 5: -28714}), (
        "gimbal wheel movement seized flight control from the PC"
    )
    assert backend.pilot_sticks is False
    assert backend._maybe_reclaim_from_sticks({3: -28715}), (
        "a real flight-stick deflection no longer seizes control"
    )
    assert backend.pilot_sticks is True


def test_nudge_deadman_ttl_is_bounded_above(make_backend):
    """An oversized TTL silently removes the "frozen UI decays to hover" guarantee."""
    backend = make_backend(pulse_s=600.0)
    assert backend.nudge_pulse_s == pytest.approx(backend_module.NUDGE_TTL_MAX_S)
    assert backend.nudge_pulse_s <= 2.0

    floor = make_backend(pulse_s=0.0)
    assert floor.nudge_pulse_s == pytest.approx(0.1)


@pytest.mark.parametrize("bad_pct", [0, -5, 100, 1000, True])
def test_nudge_authority_outside_the_envelope_is_rejected(make_backend, bad_pct):
    """--nudge-pct scales every held-key command; 100 is full stick, not a nudge."""
    with pytest.raises(ValueError, match="nudge_pct"):
        make_backend(nudge_pct=bad_pct)


def test_motion_is_refused_while_an_automated_maneuver_is_in_flight(make_backend):
    """TakeOff/Landing are firmware manoeuvres; a held nudge must not fight them.

    Zero PCMD stays allowed because it only reinforces hover.
    """
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = False
    with backend._lock:
        backend._maneuver_in_progress = "takeoff"

    assert not backend.nudge_begin("前")
    assert backend._nudge_held == set()
    assert not backend.send_pcmd(0, 10, 0, 0, reason="test_motion")
    assert any(
        event == "pcmd_blocked_maneuver" for event, _ in backend.log.records
    )
    # Hover/zero must still get through.
    assert backend.send_pcmd(0, 0, 0, 0, reason="test_zero")

    backend._clear_maneuver()
    assert backend.send_pcmd(0, 10, 0, 0, reason="test_motion_after")


def test_unconfirmed_landing_releases_the_maneuver_block(make_backend):
    """A failed landing may leave the aircraft airborne; the operator must keep
    the ability to reposition it rather than being stranded with hover only."""
    backend = make_backend(skycontroller=False)
    backend.drone.flight_state = "flying"
    backend.drone.landing_success = False
    backend.pilot_sticks = False

    assert not backend.land_cmd(reason="test")
    assert backend._maneuver_in_progress is None


def test_weak_gps_does_not_block_takeoff_and_arms_land_in_place(make_backend):
    """GPS exists at this site but is often weak, so a usable Home Point must not be
    a takeoff precondition. The onboard lost-link policy still must be, and with no
    usable home the armed fallback is land-in-place rather than return-home."""
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.home_reachable = False
    backend.drone.gps_fixed = 0
    backend.drone.home_latitude = None
    backend.drone.home_longitude = None

    backend.read_connection_inventory()

    blockers = backend._inventory_block_reason
    assert "Home Point" not in blockers
    assert "lost-link" not in blockers
    assert any(
        event == "lost_link_fallback" and fields["fallback"] == "land_in_place"
        for event, fields in backend.log.records
    )


def test_unconfirmed_lost_link_policy_still_blocks_takeoff(make_backend):
    """Relaxing the GPS requirement must not relax the auto-land guarantee."""
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.rth_ending_behavior = "hovering"      # never lands by itself

    backend.read_connection_inventory()

    assert not backend._inventory_takeoff_ready
    assert "lost-link auto-land policy" in backend._inventory_block_reason


def test_stream_health_rearms_after_recovery_so_a_second_freeze_still_hovers(
        make_backend):
    """One handoff per outage — but a SECOND outage must still trigger one.

    The latch used to be set once per session and never cleared, so after any
    recovery every later freeze was silently ignored for the rest of the flight.
    """
    backend = make_backend(stream_loss_grace_s=0.0)
    backend.pilot_sticks = False
    backend.state.mode = "AUTO"
    backend.video_stream = SimpleNamespace(last_stamp=100.0)
    stats = {"duplicate_run": 15, "frozen": True}
    backend.grabber = SimpleNamespace(
        last_frame_age=lambda: 0.05,
        frame_pipeline_stats=stats,
    )

    assert backend._evaluate_video_health(100.1)          # first freeze
    assert backend._stream_failure_latched

    # Stream recovers.
    stats["duplicate_run"] = 0
    stats["frozen"] = False
    assert not backend._evaluate_video_health(100.2)
    assert not backend._stream_failure_latched, "stream health never re-armed"
    assert any(event == "stream_health_rearmed" for event, _ in backend.log.records)

    # Second freeze must command hover again without changing control owner.
    stats["duplicate_run"] = 15
    stats["frozen"] = True
    assert backend._evaluate_video_health(100.3), (
        "a second stream freeze produced no hover"
    )
    assert backend.pilot_sticks is False


def test_pc_control_refuses_to_unblock_pcmd_if_safety_latched_during_handoff(
        make_backend):
    """An emergency stop during the blocking piloting-source handoff must win.

    take_pc_control() used to set pilot_sticks=False unconditionally after the
    handoff returned, re-enabling PC PCMD after the emergency had taken it away.
    """
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = True
    real_set_source = backend._set_piloting_source

    def slow_source(source: str) -> bool:
        # Simulate an EMERGENCY/LAND/stream-loss latch landing mid-handoff.
        backend._pulse_token += 1
        return real_set_source(source)

    backend._set_piloting_source = slow_source

    assert not backend.take_pc_control()
    assert backend.pilot_sticks is True, (
        "PC command authority was restored after a safety latch fired"
    )
    assert backend.state.tracker_state == "SOURCE_FAIL"
    assert any(
        event == "pc_control" and fields.get("note") == "safety_latched_during_handoff"
        for event, fields in backend.log.records
    )


def test_manual_and_pc_authority_transitions_are_serialized(
        make_backend):
    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = True
    source_entered = threading.Event()
    release_source = threading.Event()
    manual_started = threading.Event()
    manual_source_attempted = threading.Event()
    manual_done = threading.Event()
    results: dict[str, bool] = {}
    real_set_source = backend._set_piloting_source

    def gated_set_source(source: str) -> bool:
        if source == "Controller" and not source_entered.is_set():
            source_entered.set()
            assert release_source.wait(2.0)
        if source == "SkyController" and source_entered.is_set():
            manual_source_attempted.set()
        return real_set_source(source)

    backend._set_piloting_source = gated_set_source

    def request_pc_control() -> None:
        results["pc"] = backend.take_pc_control()

    def request_manual() -> None:
        manual_started.set()
        results["manual"] = backend.give_to_pilot(reason="race_manual")
        manual_done.set()

    pc_thread = threading.Thread(target=request_pc_control)
    manual_thread = threading.Thread(target=request_manual)
    pc_thread.start()
    assert source_entered.wait(2.0)
    manual_thread.start()
    assert manual_started.wait(2.0)
    # Manual intent blocks PCMD and invalidates the PC epoch immediately; only
    # the firmware source call waits for the in-flight serialized transition.
    assert backend.pilot_sticks is True
    assert not manual_source_attempted.wait(0.05)
    assert not manual_done.is_set()
    release_source.set()
    pc_thread.join(timeout=2.0)
    manual_thread.join(timeout=2.0)

    assert not pc_thread.is_alive()
    assert not manual_thread.is_alive()
    assert results == {"pc": False, "manual": True}
    assert backend.pilot_sticks is True
    assert backend.drone.source_state == "SkyController"
    assert backend.state.control_owner == "SKYCONTROLLER"
    assert any(
        event == "pc_control" and fields.get("note") == "safety_latched_during_handoff"
        for event, fields in backend.log.records
    )


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


def test_link_probe_falls_back_to_cached_state_without_issuing_commands(make_backend):
    backend = make_backend()
    backend.drone.connection_state = lambda: SimpleNamespace(name="Unknown")
    before = list(backend.drone.events)

    assert backend._probe_link_ok()
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


def test_critical_battery_does_not_request_rth_when_home_is_reachable(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    # A connected aircraft has had read_connection_inventory() fill this; the
    # RTH climb must be KNOWN and under the ceiling or the altitude guard
    # correctly refuses to return home. See
    # test_rth_is_refused_when_the_climb_was_never_read_back.
    backend.state.rth_min_altitude_m = 5.0
    backend.state.battery_pct = 10.0
    backend.state.drone_altitude_m = 5.0
    backend.state.max_altitude_m = 50.0
    backend.state.distance_from_home_m = 10.0
    backend.state.max_distance_m = 100.0

    assert not backend._evaluate_runtime_safety()
    assert "return_to_home" not in backend.drone.events
    assert "Landing" not in backend.drone.events
    assert not backend._runtime_safety_action_latched


def test_critical_battery_does_not_land_when_home_is_not_reachable(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = False
    backend.state.flight_state = "flying"
    backend.state.battery_pct = 9.0

    assert not backend._evaluate_runtime_safety()
    assert "return_to_home" not in backend.drone.events
    assert "Landing" not in backend.drone.events


def test_failed_fallback_hover_rearms_instead_of_disabling_protection(make_backend):
    """A fallback hover that did NOT succeed must not disarm runtime protection.

    The latch stops one *successful* action being re-issued. If it stayed set after
    a failed RTH+hover the aircraft would keep flying with altitude/distance
    protection permanently off for the rest of the session.
    """
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = False
    backend.state.flight_state = "flying"
    backend.state.drone_altitude_m = 49.5
    backend.state.max_altitude_m = 50.0
    hover_calls = []
    backend.hover_cmd = lambda reason: hover_calls.append(reason) or False

    assert backend._evaluate_runtime_safety()
    backend._runtime_safety_action_thread.join(timeout=2.0)
    assert hover_calls
    assert "Landing" not in backend.drone.events

    assert not backend._runtime_safety_action_latched, (
        "a failed safety action left the latch set: every later altitude/distance "
        "trigger is now permanently ignored"
    )
    assert any(event == "runtime_safety_rearmed" for event, _ in backend.log.records)

    # Re-arm must be real: while the aircraft is still airborne and still critical,
    # the next evaluation has to issue another safety action.
    backend.drone.flight_state = "flying"
    before = len(hover_calls)
    assert backend._evaluate_runtime_safety()
    backend._runtime_safety_action_thread.join(timeout=2.0)
    assert len(hover_calls) > before


def test_runtime_safety_action_exception_is_guarded_and_rearmed(make_backend):
    """An action-thread bug must not leave every later safety check disabled."""
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend.state.drone_altitude_m = 49.5
    backend.state.max_altitude_m = 50.0

    def explode():
        raise RuntimeError("pre-action failure")

    backend._is_already_landed = explode

    assert backend._evaluate_runtime_safety()
    backend._runtime_safety_action_thread.join(timeout=2.0)

    assert not backend._runtime_safety_action_thread.is_alive()
    assert not backend._runtime_safety_action_latched
    assert any(
        event == "runtime_safety_exception" for event, _ in backend.log.records
    )


def test_invalid_geofence_telemetry_hovers_without_rth_or_landing(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.state.flight_state = "flying"
    backend._execute_runtime_safety_action(FailureReason.INVALID_TELEMETRY.value)

    assert "Landing" not in backend.drone.events
    assert "return_to_home" not in backend.drone.events
    assert backend.state.tracker_state == "HOVER"


def test_failed_rth_command_falls_back_to_hover(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.rth_success = False
    backend.state.flight_state = "flying"
    # A connected aircraft has had read_connection_inventory() fill this; the
    # RTH climb must be KNOWN and under the ceiling or the altitude guard
    # correctly refuses to return home. See
    # test_rth_is_refused_when_the_climb_was_never_read_back.
    backend.state.rth_min_altitude_m = 5.0
    backend.state.drone_altitude_m = 49.5
    backend.state.max_altitude_m = 50.0

    assert backend._evaluate_runtime_safety()
    backend._runtime_safety_action_thread.join(timeout=1.0)

    assert "return_to_home" in backend.drone.events
    assert "Landing" not in backend.drone.events
    assert backend.state.tracker_state == "HOVER"


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


def test_cleanup_still_attempts_bounded_landing_after_pre_action_failure(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    attempts: list[str] = []

    def fail_nudge_clear(*, reason: str) -> None:
        raise RuntimeError("nudge cleanup failed")

    backend.nudge_clear = fail_nudge_clear
    backend._land_and_confirm = lambda reason: attempts.append(reason) or True

    backend.cleanup()

    assert attempts == ["cleanup_force_land"]


def test_cleanup_landing_holds_authority_transition_until_touchdown(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    events = []

    class Authority:
        class Transition:
            def __enter__(self):
                events.append("authority_enter")

            def __exit__(self, *_args):
                events.append("authority_exit")

        def transition(self):
            return self.Transition()

    backend.authority_controller = Authority()
    backend._set_piloting_source = (
        lambda source: events.append(("source", source)) or True
    )
    backend._is_already_landed = lambda: False
    backend._land_and_confirm = (
        lambda reason: events.append(("land", reason)) or True
    )

    assert backend._cleanup_attempt_landing() is True
    assert events == [
        "authority_enter",
        ("source", "Controller"),
        ("land", "cleanup_force_land"),
        ("source", "SkyController"),
        "authority_exit",
    ]


def test_unconfirmed_cleanup_latches_out_all_future_computer_motion(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend._land_and_confirm = lambda _reason: False

    assert backend.cleanup() is False

    assert backend._shutdown_latched is True
    assert backend.pilot_sticks is True
    assert backend.send_pcmd(10, 0, 0, 0, reason="late_auto") is False
    assert backend.set_nudge_vector(1.0, 0.0, 0.0, 0.0) is False
    assert backend.take_pc_control() is False
    assert backend.takeoff_cmd() is False


@pytest.mark.parametrize("motion", ["vector", "nudge"])
def test_shutdown_latch_wins_race_before_motion_state_is_committed(
    make_backend, motion: str,
) -> None:
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False
    underlying = backend._lock

    class LatchOnEnter:
        def __enter__(self):
            underlying.acquire()
            backend._shutdown_latched = True
            return self

        def __exit__(self, *_args):
            underlying.release()

    backend._lock = LatchOnEnter()
    backend._ensure_nudge_loop = lambda: None

    accepted = (
        backend.set_nudge_vector(1.0, 0.0, 0.0, 0.0)
        if motion == "vector"
        else backend.nudge_begin("前")
    )

    assert accepted is False
    assert backend._nudge_vector is None
    assert backend._nudge_held == set()


def test_shutdown_latch_wins_race_before_takeoff_is_scheduled(make_backend) -> None:
    backend = make_backend(skycontroller=False)
    underlying = backend._lock

    class LatchOnEnter:
        def __enter__(self):
            underlying.acquire()
            backend._shutdown_latched = True
            return self

        def __exit__(self, *_args):
            underlying.release()

    backend._lock = LatchOnEnter()
    backend.nudge_clear = lambda *, reason: None
    backend._takeoff_preflight = lambda _epoch: pytest.fail(
        "shutdown-latched takeoff must not reach preflight"
    )

    assert backend.takeoff_cmd() is False
    assert backend._maneuver_in_progress is None


def test_cleanup_completes_when_resource_close_reports_fault(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "landed"

    class BrokenGrabber:
        def stop(self):
            raise RuntimeError("pdraw close failed")

    backend.grabber = BrokenGrabber()
    backend._stop_stick_monitor = lambda: (_ for _ in ()).throw(
        RuntimeError("stick monitor close failed")
    )

    assert backend.cleanup()
    assert backend._cleanup_done
    assert backend.grabber is None
    assert backend.drone is None
    assert any(
        event == "cleanup_stick_monitor_error" for event, _ in backend.log.records
    )


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


def test_recording_quality_is_applied_before_start(make_backend):
    backend = make_backend()
    backend.drone.flight_state = "landed"

    assert backend.set_recording_quality("uhd_4k_30") is True
    assert backend.recording_profile.profile_id == "uhd_4k_30"
    assert backend.drone.recording_mode_state[0]["resolution"].name == (
        "resolution.res_uhd_4k"
    )
    assert "set_recording_mode" in backend.drone.events
    assert backend.start_flight_recording(reason="test") is True
    assert backend.recording_active is True
    assert backend.drone.events.count("set_recording_mode") >= 2
    assert "start_recording" in backend.drone.events
    assert backend.drone.events.index("set_recording_mode") < backend.drone.events.index(
        "start_recording"
    )


def test_recording_quality_change_is_refused_while_recording(make_backend):
    backend = make_backend()
    assert backend.start_flight_recording(reason="test") is True
    assert backend.set_recording_quality("uhd_4k_30") is False
    assert backend.recording_profile.profile_id == "fhd_30"


def test_start_recording_fails_closed_when_quality_readback_mismatches(
        make_backend):
    backend = make_backend()
    backend.drone.set_recording_mode_ok = False

    assert backend.set_recording_quality("uhd_4k_30") is False
    assert backend.start_flight_recording(reason="test") is False
    assert backend.recording_active is False
    assert "start_recording" not in backend.drone.events





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

    assert not backend.cleanup()

    assert not backend._landed
    assert not backend._cleanup_done
    assert backend.recording_active
    assert "Landing" in drone.events
    assert "media_stop" not in drone.events
    assert "disconnect" not in drone.events


def test_close_keeps_live_connection_for_landing_retry(make_backend):
    backend = make_backend()
    drone = backend.drone
    drone.flight_state = "flying"
    drone.landing_success = False

    first = backend.close("window")

    assert not first.closed
    assert first.reason_code == "LAND_UNCONFIRMED"
    assert backend.drone is drone
    assert not backend._cleanup_done
    assert "disconnect" not in drone.events

    drone.landing_success = True
    second = backend.close("window_retry")

    assert second.closed
    assert second.reason_code == "OK"
    assert backend.drone is None
    assert backend._cleanup_done
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


def test_hold_loop_error_drops_the_hold_and_hovers(make_backend):
    """A link error mid-hold must not strand the aircraft on its last PCMD.

    The deadman timer runs inside the hold loop, so if the loop thread dies the
    hold can never expire. Regression for that: the loop must drop the hold and
    command hover even when send_pcmd raises.
    """
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    drone = backend.drone
    calls = {"pcmd": 0}
    original_call = type(drone).__call__

    def flaky(self, message):
        first = self._first(message)
        if getattr(first, "name", "") == "PCMD" and first.args[1:5] != (0, 0, 0, 0):
            calls["pcmd"] += 1
            if calls["pcmd"] >= 2:
                raise ConnectionError("olympe link error during hold")
        return original_call(self, message)

    type(drone).__call__ = flaky
    try:
        assert backend.nudge_begin("前")
        thread = backend._nudge_loop_thread
        assert thread is not None
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "hold loop never exited"
    finally:
        type(drone).__call__ = original_call

    assert backend._nudge_held == set()
    assert backend.state.tracker_state == "HOVER"
    assert backend.state.last_command == "nudge_loop_error_hover"
    assert any(
        event == "nudge_loop_error" and not fields["ok"]
        for event, fields in backend.log.records
    )
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)


def test_nudge_loop_handoff_restarts_when_hold_survives_old_loop_exit(make_backend):
    """A stopping loop must not leave a live hold without a deadman owner."""
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    assert backend.nudge_begin("前")
    old_thread = backend._nudge_loop_thread
    assert old_thread is not None

    # Model the old loop being asked to stop while the input remains held. Its
    # finally block must atomically hand ownership to a replacement loop.
    backend._nudge_loop_stop.set()
    old_thread.join(timeout=2.0)
    assert not old_thread.is_alive()
    replacement = backend._nudge_loop_thread
    assert replacement is not None
    assert replacement is not old_thread
    assert replacement.is_alive()

    backend.nudge_clear(reason="handoff_test")
    replacement.join(timeout=2.0)
    assert backend._nudge_held == set()


@pytest.mark.parametrize(
    "invalidate",
    ["clear", "cancel", "land", "manual", "cleanup"],
)
def test_stale_nudge_snapshot_cannot_send_after_motion_invalidation(
        make_backend, invalidate):
    """A nudge snapshot must be rejected after any motion-cancelling action."""
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    drone = backend.drone
    with backend._lock:
        backend._nudge_held.add("前")
        backend._nudge_deadline = time.monotonic() + 5.0

    snapshot_ready = threading.Event()
    release_snapshot = threading.Event()
    original_combined = backend._combined_nudge_pcmd

    def gated_combined():
        pcmd = original_combined()
        snapshot_ready.set()
        if not release_snapshot.wait(timeout=2.0):
            raise AssertionError("test interleaving was not released")
        return pcmd

    backend._combined_nudge_pcmd = gated_combined
    step_result: list[tuple[bool, bool]] = []
    step_thread = threading.Thread(
        target=lambda: step_result.append(backend._nudge_loop_step()),
    )
    step_thread.start()
    assert snapshot_ready.wait(timeout=2.0)

    if invalidate == "clear":
        backend.nudge_clear(reason="race_clear")
    elif invalidate == "cancel":
        assert backend.command("nudge_clear") is True
    elif invalidate == "land":
        drone.flight_state = "flying"
        assert backend.land_cmd("race_land") is True
    elif invalidate == "manual":
        assert backend.give_to_pilot(reason="race_manual") is True
    else:
        drone.flight_state = "flying"
        assert backend.cleanup() is True

    release_snapshot.set()
    step_thread.join(timeout=2.0)
    assert not step_thread.is_alive()
    assert step_result == [(True, False)]
    assert all(
        pcmd[1:5] == (0, 0, 0, 0)
        for pcmd in drone.pcmds
    ), f"stale non-zero PCMD escaped {invalidate}: {drone.pcmds}"


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


def test_superseded_takeoff_does_not_restore_source_during_blocked_landing(
        make_backend):
    class BlockedManeuverDrone(_FakeDrone):
        def __init__(self):
            super().__init__()
            self.takeoff_wait_entered = threading.Event()
            self.release_takeoff_wait = threading.Event()
            self.landing_wait_entered = threading.Event()
            self.release_landing_wait = threading.Event()

        @staticmethod
        def _blocked_wait(expectation, entered, release):
            class BlockedExpectation:
                def wait(self, _timeout=None):
                    entered.set()
                    assert release.wait(timeout=1.0)
                    return expectation.wait(_timeout=_timeout)

            return BlockedExpectation()

        def __call__(self, message):
            expectation = super().__call__(message)
            first = self._first(message)
            if first.name == "TakeOff":
                return self._blocked_wait(
                    expectation,
                    self.takeoff_wait_entered,
                    self.release_takeoff_wait,
                )
            if first.name == "Landing":
                return self._blocked_wait(
                    expectation,
                    self.landing_wait_entered,
                    self.release_landing_wait,
                )
            return expectation

    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone = BlockedManeuverDrone()
    backend.drone.flight_state = "landed"
    backend._firmware_config_ok = True
    takeoff_results: list[bool] = []
    takeoff_thread = threading.Thread(
        target=lambda: takeoff_results.append(backend.takeoff_cmd())
    )
    takeoff_thread.start()
    assert backend.drone.takeoff_wait_entered.wait(timeout=1.0)

    backend.drone.flight_state = "flying"
    landing_results: list[bool] = []
    landing_thread = threading.Thread(
        target=lambda: landing_results.append(backend.land_cmd("blocked_landing"))
    )
    landing_thread.start()
    assert backend.drone.landing_wait_entered.wait(timeout=1.0)
    assert backend._maneuver_in_progress == "landing"
    source_requests_before_takeoff_release = list(backend.drone.source_requests)

    backend.drone.release_takeoff_wait.set()
    takeoff_thread.join(timeout=1.0)

    assert not takeoff_thread.is_alive()
    assert takeoff_results == [False]
    assert backend._maneuver_in_progress == "landing"
    assert backend.drone.source_requests == source_requests_before_takeoff_release
    assert backend.drone.source_state == "Controller"
    assert backend.pilot_sticks is False

    backend.drone.release_landing_wait.set()
    landing_thread.join(timeout=1.0)

    assert not landing_thread.is_alive()
    assert landing_results == [True]
    assert backend.drone.source_requests[-1] == "SkyController"
    assert backend._landed


def test_hover_invalidates_a_pending_takeoff_epoch(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "hovering"
    backend._maneuver_in_progress = "takeoff"
    before = backend._pulse_token

    assert backend.hover_cmd("operator_auto_cancel")

    assert backend._pulse_token == before + 1
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)


def test_nudge_clear_does_not_cancel_a_pending_takeoff(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "takingoff"
    backend._maneuver_in_progress = "takeoff"
    before = backend._pulse_token

    result = backend.command(ControlRequest.create(
        ControlAction.NUDGE_CLEAR,
        human_origin=True,
    ))

    assert result.accepted and result.executed
    assert backend._pulse_token == before
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)


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


def test_unconfirmed_hover_after_scheduled_takeoff_forces_bounded_landing(
    make_backend,
) -> None:
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.takeoff_success = False
    backend.drone.landing_success = True

    assert not backend.takeoff_cmd()

    assert backend.drone.events.index("TakeOff") < backend.drone.events.index(
        "Landing"
    )
    assert backend._landed
    assert backend.state.tracker_state == "LAND"


def test_axes_active_deadzone():
    assert not backend_module.axes_active({0: 0, 1: 100})
    assert not backend_module.axes_active({0: 2000})  # equal to default deadzone
    assert backend_module.axes_active({0: 2001})
    assert backend_module.axes_active({3: -5000, 0: 0})


def test_stick_monitor_processes_axis_data_and_notifies_override():
    observed = []
    monitor = backend_module.SkyControllerStickMonitor(observed.append)

    event = struct.pack(
        backend_module._JS_EVENT_FMT,
        0,
        2501,
        backend_module._JS_EVENT_AXIS,
        0,
    )
    monitor._process_stick_data(event)

    assert observed == [{0: 2501}]
    assert monitor.snapshot_axes() == {0: 2501}


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


@pytest.mark.parametrize(
    ("home_reachable", "expect_rth"),
    [
        (True, True),
        (False, False),
    ],
)
def test_stick_monitor_disconnect_recovers_by_gps_availability(
    make_backend, home_reachable, expect_rth,
):
    """Operator policy 2026-08-06, end to end: losing the controller while the
    control link still works uses RTH when Home is usable, otherwise it hovers."""
    backend = make_backend()
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = home_reachable
    backend.state.flight_state = "flying"

    backend._on_stick_monitor_disconnect("device unplugged")
    backend._runtime_safety_action_thread.join(timeout=2.0)

    assert backend.state.stick_monitor_ok is False
    assert ("return_to_home" in backend.drone.events) is expect_rth
    assert "Landing" not in backend.drone.events
    if not expect_rth:
        assert backend.state.tracker_state == "HOVER"
    assert backend.state.active_incident == "controller_disconnected"


def test_pc_control_is_refused_when_stick_monitor_is_unavailable(make_backend):
    backend = make_backend()
    backend._stick_monitor = None
    backend.state.stick_monitor_ok = False

    assert not backend.take_pc_control()
    assert backend.pilot_sticks is True
    assert backend.state.tracker_state == "STICK_MONITOR_FAIL"
    assert "Controller" not in backend.drone.source_requests


def test_takeoff_battery_blocker_pure_decision() -> None:
    from olympe_live_backend import takeoff_battery_blocker

    assert takeoff_battery_blocker(50.0, None) == (
        "takeoff battery floor is unavailable or invalid"
    )
    assert takeoff_battery_blocker(20.0, 30.0) == (
        "battery 20% is below the 30% takeoff floor"
    )
    assert takeoff_battery_blocker(30.0, 30.0) is None
    assert takeoff_battery_blocker(100.0, 30.0) is None


def test_takeoff_gps_outcome_pure_decision() -> None:
    from olympe_live_backend import takeoff_gps_outcome

    fixed_ok = takeoff_gps_outcome(fixed=True, gps_required=True)
    assert fixed_ok.blocker is None and fixed_ok.advisory is None

    api_unavailable_required = takeoff_gps_outcome(fixed=None, gps_required=True)
    assert api_unavailable_required.blocker == (
        "GPS state API unavailable; required by configured distance geofence"
    )
    assert api_unavailable_required.advisory is None

    api_unavailable_optional = takeoff_gps_outcome(fixed=None, gps_required=False)
    assert api_unavailable_optional.blocker is None
    assert api_unavailable_optional.advisory == (
        "GPS state API unavailable; takeoff remains allowed"
    )

    no_fix_required = takeoff_gps_outcome(fixed=False, gps_required=True)
    assert no_fix_required.blocker == (
        "GPS fix unavailable; required by configured distance geofence"
    )
    assert no_fix_required.advisory is None

    no_fix_optional = takeoff_gps_outcome(fixed=False, gps_required=False)
    assert no_fix_optional.blocker is None
    assert no_fix_optional.advisory == "GPS fix unavailable; takeoff remains allowed"


def test_takeoff_preflight_is_blocked_when_stick_monitor_is_unavailable(
    make_backend,
):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend._stick_monitor = None
    backend.state.stick_monitor_ok = False

    assert not backend._takeoff_preflight()
    assert "stick monitor" in backend.state.preflight_reason


def test_takeoff_preflight_is_blocked_while_physical_sticks_are_deflected(
    make_backend,
):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "landed"
    backend._stick_monitor = SimpleNamespace(
        healthy=True,
        is_active=lambda: True,
    )
    backend.state.stick_monitor_ok = True

    assert not backend._takeoff_preflight()
    assert "sticks are deflected" in backend.state.preflight_reason
    assert "TakeOff" not in backend.drone.events


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


def test_stick_override_blocks_all_subsequent_pc_motion(make_backend):
    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = False
    backend.drone.source_state = "Controller"

    assert backend._maybe_reclaim_from_sticks({0: 12000})

    assert not backend.send_pcmd(5, 6, 7, 8, reason="after_stick_override")
    assert not backend.set_nudge_vector(0.5, 0.5, 0.5, 0.5)
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.drone.source_state == "SkyController"


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


def test_take_pc_control_aborts_when_sticks_move_during_source_handoff(
    make_backend,
):
    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = True
    active = {"value": False}
    backend._stick_monitor = SimpleNamespace(
        healthy=True,
        is_active=lambda: active["value"],
    )
    backend.state.stick_monitor_ok = True
    real_set_source = backend._set_piloting_source

    def move_sticks_during_handoff(source: str) -> bool:
        result = real_set_source(source)
        if source == "Controller":
            active["value"] = True
        return result

    backend._set_piloting_source = move_sticks_during_handoff

    assert not backend.take_pc_control()
    assert backend.pilot_sticks is True
    assert backend.drone.source_state == "SkyController"
    assert any(
        name == "pc_control"
        and fields.get("note") == "sticks_active_during_handoff"
        for name, fields in backend.log.records
    )


def test_calibration_motion_snapshot_reports_received_pcmd_and_fresh_telemetry(
    make_backend,
):
    backend = make_backend()
    now_ns = time.monotonic_ns()
    backend._last_pcmd = (10, 0, 0, 0)
    backend._last_pcmd_mono_ns = now_ns
    backend.state.telemetry_read_mono_ns = now_ns
    backend.state.speed_north_mps = 0.0
    backend.state.speed_east_mps = 0.2
    backend.state.speed_down_mps = 0.0
    backend.state.att_yaw = 0.0
    backend.state.heading_state = "OK"
    backend.state.flight_state = "hovering"
    backend._stick_monitor = SimpleNamespace(snapshot_axes=lambda: {0: 123})

    snapshot = backend.calibration_motion_snapshot()

    assert snapshot["pcmd"] == (10, 0, 0, 0)
    assert snapshot["pcmd_age_s"] == pytest.approx(0.0, abs=0.05)
    assert snapshot["telemetry_age_s"] == pytest.approx(0.0, abs=0.05)
    assert snapshot["speed_east_mps"] == pytest.approx(0.2)
    assert snapshot["stick_axes"] == {0: 123}


# ---------------------------------------------------------------------------
# PCMD direction mapping: does each operator button produce the right motion?
#
# Drives the REAL nudge path (nudge_begin -> _combined_nudge_pcmd) and asserts the
# PCMD sign pattern for the 12 diagonal/vertical combinations used on the bench.
#
# ANAFI / ardrone3 PCMD(flag, roll, pitch, yaw, gaz, ts) convention:
#     roll  > 0 -> RIGHT      pitch > 0 -> FORWARD
#     yaw   > 0 -> CLOCKWISE  gaz   > 0 -> CLIMB
#
# roll/pitch are BODY frame: "forward" is wherever the nose points. These buttons
# therefore verify the sign mapping only; they say nothing about map-frame heading
# fusion (HeadingEstimator), which lives on the autonomous route path.



# name -> (button(s) to hold, expected sign of (roll, pitch, yaw, gaz))
# sign: +1 / -1 / 0
DIRECTION_CASES = [
    ("前右上", ["右上前"],       (+1, +1, 0, +1)),
    ("前上",   ["前", "上"],     (0, +1, 0, +1)),
    ("前左上", ["左上前"],       (-1, +1, 0, +1)),
    ("前右下", ["右下前"],       (+1, +1, 0, -1)),
    ("前下",   ["前", "下"],     (0, +1, 0, -1)),
    ("前左下", ["左下前"],       (-1, +1, 0, -1)),
    ("後右上", ["右上後"],       (+1, -1, 0, +1)),
    ("後上",   ["後", "上"],     (0, -1, 0, +1)),
    ("後左上", ["左上後"],       (-1, -1, 0, +1)),
    ("後右下", ["右下後"],       (+1, -1, 0, -1)),
    ("後下",   ["後", "下"],     (0, -1, 0, -1)),
    ("後左下", ["左下後"],       (-1, -1, 0, -1)),
]


def _sign(v: int) -> int:
    return 0 if v == 0 else (1 if v > 0 else -1)


@pytest.mark.parametrize("label,buttons,expected", DIRECTION_CASES,
                         ids=[c[0] for c in DIRECTION_CASES])
def test_nudge_button_produces_the_expected_pcmd_direction(
        make_backend, label, buttons, expected):
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False

    for name in buttons:
        assert backend.nudge_begin(name), f"{label}: button {name!r} was refused"

    pcmd = backend._combined_nudge_pcmd()
    assert tuple(_sign(v) for v in pcmd) == expected, (
        f"{label} (hold {buttons}) produced PCMD {pcmd} "
        f"whose signs {tuple(_sign(v) for v in pcmd)} != expected {expected}"
    )
    # Never yaw while translating: a diagonal must not also spin the airframe.
    assert pcmd[2] == 0, f"{label} unexpectedly commands yaw={pcmd[2]}"
    # Authority stays inside the nudge envelope on every axis.
    assert all(abs(v) <= backend.nudge_pct for v in pcmd), (
        f"{label} exceeded the nudge envelope: {pcmd} vs pct={backend.nudge_pct}"
    )


def test_every_bench_direction_is_reachable_and_distinct(make_backend):
    """All 12 bench directions must be reachable and produce 12 distinct commands."""
    seen = {}
    for label, buttons, _expected in DIRECTION_CASES:
        backend = make_backend(skycontroller=False, pulse_s=5.0)
        backend.pilot_sticks = False
        for name in buttons:
            assert backend.nudge_begin(name)
        seen[label] = backend._combined_nudge_pcmd()
    assert len(set(seen.values())) == len(DIRECTION_CASES), (
        f"two bench directions produce the same PCMD: {seen}"
    )


def test_releasing_every_button_returns_to_zero(make_backend):
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False
    assert backend.nudge_begin("右上前")
    assert backend._combined_nudge_pcmd() != (0, 0, 0, 0)
    backend.nudge_end("右上前")
    assert backend._combined_nudge_pcmd() == (0, 0, 0, 0)


def test_opposite_buttons_cancel_instead_of_stacking(make_backend):
    """Holding 前 and 後 together must not produce a runaway command."""
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False
    assert backend.nudge_begin("前")
    assert backend.nudge_begin("後")
    assert backend._combined_nudge_pcmd()[1] == 0


def test_a_short_tap_moves_briefly_and_then_zeroes(make_backend):
    """One tap must be a nudge, not a launch: PCMD stops the moment you let go."""
    backend = make_backend(skycontroller=False, pulse_s=5.0)  # deadman far away
    backend.pilot_sticks = False

    assert backend.nudge_begin("前")
    time.sleep(0.30)                      # a deliberate ~0.3 s tap
    backend.nudge_end("前")
    time.sleep(0.15)                      # let the hold loop unwind

    pcmds = [tuple(a[1:5]) for a in backend.drone.pcmds]
    assert pcmds, "no command reached the aircraft at all"
    assert pcmds[-1] == (0, 0, 0, 0), f"tap did not end in zero PCMD: {pcmds[-1]}"
    assert backend._nudge_held == set()
    assert backend.state.tracker_state == "HOVER"
    # Everything after the release must be zero -- no lingering motion command.
    last_nonzero = max(i for i, p in enumerate(pcmds) if p != (0, 0, 0, 0))
    assert all(p == (0, 0, 0, 0) for p in pcmds[last_nonzero + 1:])


def test_a_lost_release_event_still_stops_within_the_deadman(make_backend):
    """If the release event never arrives (focus loss, UI stall), the deadman
    must stop the aircraft on its own -- this is the 'runaway' guard."""
    backend = make_backend(skycontroller=False, pulse_s=0.2)
    backend.pilot_sticks = False

    assert backend.nudge_begin("前")
    started = time.monotonic()
    thread = backend._nudge_loop_thread
    assert thread is not None
    thread.join(timeout=3.0)              # NOTE: no nudge_end() at all
    elapsed = time.monotonic() - started

    assert not thread.is_alive(), "hold never expired: this WOULD run away"
    assert elapsed < 1.5, f"deadman took {elapsed:.2f}s"
    assert backend._nudge_held == set()
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.state.tracker_state == "HOVER"


def test_lost_controller_returns_home_when_gps_home_is_usable(make_backend):
    """Operator policy 2026-08-06: when nothing can command the aircraft any more,
    GPS decides -- usable Home Point means return home."""
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = True
    backend.state.flight_state = "flying"
    # A connected aircraft has had read_connection_inventory() fill this; the
    # RTH climb must be KNOWN and under the ceiling or the altitude guard
    # correctly refuses to return home. See
    # test_rth_is_refused_when_the_climb_was_never_read_back.
    backend.state.rth_min_altitude_m = 5.0

    backend._execute_runtime_safety_action("controller_disconnected")

    assert "return_to_home" in backend.drone.events
    assert "Landing" not in backend.drone.events
    assert any(
        event == "runtime_safety" and fields.get("action") == "rth"
        for event, fields in backend.log.records
    )


def test_lost_controller_hovers_without_gps_home(make_backend):
    """No usable Home Point means hover instead of landing in place."""
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = False
    backend.drone.gps_fixed = 0
    backend.state.flight_state = "flying"

    backend._execute_runtime_safety_action("controller_disconnected")

    assert "return_to_home" not in backend.drone.events
    assert "Landing" not in backend.drone.events
    assert backend.state.tracker_state == "HOVER"
    assert any(
        event == "runtime_safety" and fields.get("action") == "hover"
        for event, fields in backend.log.records
    )


def test_takeoff_needs_no_gps_when_the_distance_geofence_is_off(make_backend):
    """GPS is optional at this site: with the geofence off nothing may demand a fix."""
    backend = make_backend(
        max_altitude_m=10.0, max_distance_m=50.0, distance_geofence=False,
    )
    backend.drone.flight_state = "landed"
    backend.drone.gps_fixed = 0
    backend.drone.home_reachable = False
    backend.drone.setting_states["NoFlyOverMaxDistanceChanged"]["shouldNotFlyOver"] = 0

    assert backend._takeoff_preflight(), backend.state.preflight_reason


def test_rth_is_skipped_when_it_would_break_the_altitude_ceiling(make_backend):
    """RTH climbs to return_home_min_altitude before heading home. A 39 m climb
    under a 3 m ceiling is not a recovery, it is a new emergency."""
    backend = make_backend(max_altitude_m=3.0, max_distance_m=30.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = True          # RTH would otherwise be chosen
    backend.state.flight_state = "flying"
    backend.state.rth_min_altitude_m = 39.2

    backend._execute_runtime_safety_action("controller_disconnected")

    assert "return_to_home" not in backend.drone.events
    assert "Landing" not in backend.drone.events
    assert backend.state.tracker_state == "HOVER"
    assert any(
        event == "runtime_safety_rth_skipped" and "39.2" in fields["detail"]
        for event, fields in backend.log.records
    )


def test_rth_still_used_when_the_climb_fits_under_the_ceiling(make_backend):
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = True
    backend.state.flight_state = "flying"
    backend.state.rth_min_altitude_m = 39.2

    backend._execute_runtime_safety_action("controller_disconnected")

    assert "return_to_home" in backend.drone.events


def test_rth_min_altitude_is_pinned_and_read_back(make_backend):
    """RTH climb altitude used to be read but never written, so it was whatever
    the last FreeFlight session left behind (39.2 m on the reference aircraft)."""
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.rth_min_altitude = 39.2          # residual value

    assert backend._configure_lost_link_policy_if_safe()

    assert backend.drone.rth_min_altitude == pytest.approx(
        backend_module.DEFAULT_RTH_MIN_ALTITUDE_M)
    assert backend.state.rth_min_altitude_m == pytest.approx(
        backend_module.DEFAULT_RTH_MIN_ALTITUDE_M)
    assert any(
        event == "lost_link_policy" and fields.get("ok")
        and fields.get("rth_min_altitude_m") == pytest.approx(
            backend_module.DEFAULT_RTH_MIN_ALTITUDE_M)
        for event, fields in backend.log.records
    )


def test_rth_min_altitude_readback_mismatch_fails_closed(make_backend):
    backend = make_backend(max_altitude_m=10.0, max_distance_m=50.0)
    backend.drone.flight_state = "landed"
    backend.drone.rth_min_altitude = 39.2
    original = type(backend.drone).__call__

    def refuse(self, message):
        if getattr(self._first(message), "name", "") == "set_min_altitude":
            return _Expectation(True)          # acks but never applies
        return original(self, message)

    type(backend.drone).__call__ = refuse
    try:
        assert not backend._configure_lost_link_policy_if_safe()
    finally:
        type(backend.drone).__call__ = original
    assert not backend.state.rth_policy_configured


def test_magnetometer_state_is_logged_on_change_only(make_backend):
    """Calibration outcome must be reviewable afterwards, not only on screen.

    The reader runs every poll (~30 Hz), so it must log transitions, not ticks.
    """
    backend = make_backend()
    backend.drone.drone_calibration_required = 1

    backend._read_magnetometer_calibration_state()
    first = [f for e, f in backend.log.records if e == "magnetometer_state"]
    assert len(first) == 1
    assert first[0]["required"] == "required"
    assert first[0]["required_code"] == 1

    # Unchanged state must not add a second line, however often it is read.
    for _ in range(20):
        backend._read_magnetometer_calibration_state()
    assert len([1 for e, _ in backend.log.records if e == "magnetometer_state"]) == 1

    # A real transition is recorded.
    backend.drone.drone_calibration_required = 0
    backend._read_magnetometer_calibration_state()
    entries = [f for e, f in backend.log.records if e == "magnetometer_state"]
    assert len(entries) == 2
    assert entries[-1]["required"] == "valid"


def test_magnetometer_axis_progress_is_logged(make_backend):
    """Each axis the firmware asks for, and each one completed, must appear."""
    backend = make_backend()
    backend.drone.drone_calibration_required = 1
    backend.drone.drone_calibration_started = True
    backend.drone.drone_calibration_axis = "xAxis"
    backend._read_magnetometer_calibration_state()

    backend.drone.drone_calibration_x_done = True
    backend.drone.drone_calibration_axis = "yAxis"
    backend._read_magnetometer_calibration_state()

    entries = [f for e, f in backend.log.records if e == "magnetometer_state"]
    assert [e["axis"].rsplit(".", 1)[-1] for e in entries][-2:] == ["xAxis", "yAxis"]
    assert entries[-1]["x_done"] is True


def test_unreadable_skycontroller_compass_does_not_block_takeoff(make_backend):
    """Field bug 2026-08-06: an unreadable CONTROLLER compass state made preflight
    fail with 'aircraft magnetometer calibration state is unavailable' while the
    aircraft compass was valid -- sending the operator off to calibrate a compass
    that was already fine, four takeoff attempts in a row."""
    backend = make_backend(skycontroller=True)
    backend.drone.drone_calibration_required = 0        # aircraft compass VALID
    backend.drone.controller_calibration_state = None   # controller unreadable

    error = backend._magnetometer_control_error()

    assert error is None, f"takeoff blocked by the controller compass: {error}"
    assert backend.state.skycontroller_magnetometer_readable is False
    assert any(
        event == "magnetometer_calibration_warning"
        and fields.get("target") == "skycontroller"
        for event, fields in backend.log.records
    )


def test_unavailable_aircraft_compass_still_blocks_takeoff(make_backend):
    """The AIRCRAFT compass gate must not be relaxed by the controller fix."""
    backend = make_backend(skycontroller=True)
    backend.drone.drone_calibration_required = None     # aircraft state unreadable

    error = backend._magnetometer_control_error()

    assert error is not None and "aircraft magnetometer" in error


def test_auto_pc_control_refuses_when_not_confirmed_landed(make_backend):
    """A UI restarted mid-flight must never seize control from the flying pilot."""
    backend = make_backend(skycontroller=False)
    backend.drone.flight_state = "flying"
    backend.pilot_sticks = True

    assert not backend._auto_take_pc_control_if_safe()
    assert backend.pilot_sticks is True
    assert any(
        event == "auto_pc_control" and fields.get("reason") == "not_confirmed_landed"
        for event, fields in backend.log.records
    )


def test_auto_pc_control_refuses_while_sticks_are_deflected(make_backend):
    backend = make_backend(skycontroller=True)
    backend.drone.flight_state = "landed"
    backend._stick_monitor = SimpleNamespace(
        is_active=lambda: True, stop=lambda: None,
        snapshot_axes=lambda: {0: 28714},
    )
    backend.state.stick_monitor_ok = True

    assert not backend._auto_take_pc_control_if_safe()
    assert any(
        event == "auto_pc_control" and fields.get("reason") == "sticks_deflected"
        for event, fields in backend.log.records
    )


def test_auto_pc_control_can_be_disabled(make_backend):
    backend = make_backend(skycontroller=False)
    backend.drone.flight_state = "landed"
    backend.auto_pc_control = False
    assert not backend._auto_take_pc_control_if_safe()


def test_stick_movement_still_reclaims_after_auto_pc_control(make_backend):
    """Defaulting to PC control must not weaken the stick handback."""
    backend = make_backend(skycontroller=True)
    backend.pilot_sticks = False
    assert backend._maybe_reclaim_from_sticks({3: -28715})
    assert backend.pilot_sticks is True


def _stale_grabber(backend, age_s):
    backend.video_stream = SimpleNamespace(last_stamp=100.0)
    backend.grabber = SimpleNamespace(
        last_frame_age=lambda: age_s,
        frame_pipeline_stats={"duplicate_run": 0, "frozen": False},
    )


def test_brief_stream_latency_does_not_take_pc_control_away(make_backend):
    """Field observation 2026-08-06: the picture sometimes just lags a few seconds
    and recovers. A single stale sample used to hand control straight back."""
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = False
    backend.stream_loss_grace_s = 10.0

    _stale_grabber(backend, 0.9)                      # stale, but only just started
    assert not backend._evaluate_video_health(100.0)
    assert backend.pilot_sticks is False, "control taken away by a latency spike"
    assert any(e == "stream_stale_grace_started" for e, _ in backend.log.records)

    _stale_grabber(backend, 0.05)                     # recovers a moment later
    assert not backend._evaluate_video_health(100.1)
    assert backend.pilot_sticks is False
    recovered = [f for e, f in backend.log.records if e == "stream_recovered"]
    assert recovered and recovered[-1]["stale_for_s"] >= 0.0


def test_stream_lost_beyond_the_grace_still_hovers(make_backend):
    """A genuinely dead stream must zero motion without changing ownership."""
    backend = make_backend(skycontroller=False)
    backend.pilot_sticks = False
    backend.stream_loss_grace_s = 0.15                # short grace for the test
    backend.state.mode = "AUTO"

    _stale_grabber(backend, 0.9)
    assert not backend._evaluate_video_health(100.0)   # grace starts
    time.sleep(0.2)
    assert backend._evaluate_video_health(100.3), "dead stream never hovered"
    assert backend.pilot_sticks is False
    assert backend.state.tracker_state == "STREAM_LOST_HOVER"


def test_stream_loss_grace_is_validated(make_backend):
    with pytest.raises(ValueError, match="stream_loss_grace_s"):
        make_backend(stream_loss_grace_s=-1.0)


def test_controller_confirmed_from_hid_when_variant_is_never_emitted(make_backend):
    """SkyController 3 fw 1.8.1 / Olympe 8.4.0 never emits ProductVariantChanged
    (verified on the reference unit), so the model must be confirmable from the
    HID product string the stick monitor already resolved."""
    backend = make_backend(skycontroller=True)
    backend.drone.controller_variant = None          # as on the real unit
    backend._stick_monitor = SimpleNamespace(
        is_active=lambda: False, stop=lambda: None,
        device_name="Parrot Parrot Skycontroller 3 v1.8.1",
    )
    backend.state.stick_monitor_ok = True

    backend.read_connection_inventory()

    assert "not confirmed SkyController 3" not in backend._inventory_block_reason


def test_unknown_controller_is_still_rejected(make_backend):
    """The fallback must not confirm just any device."""
    backend = make_backend(skycontroller=True)
    backend.drone.controller_variant = None
    backend._stick_monitor = SimpleNamespace(
        is_active=lambda: False, stop=lambda: None,
        device_name="Generic USB Gamepad",
    )
    backend.state.stick_monitor_ok = True

    backend.read_connection_inventory()

    assert "not confirmed SkyController 3" in backend._inventory_block_reason


def test_virtual_stick_vector_stays_inside_the_nudge_envelope(make_backend):
    """A dragged stick must never out-command a button press."""
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False

    assert backend.set_nudge_vector(roll=1.0, pitch=-1.0, yaw=0.5, gaz=0.25)
    pcmd = backend._combined_nudge_pcmd()
    pct = backend.nudge_pct
    assert pcmd == (pct, -pct, round(0.5 * pct), round(0.25 * pct))
    assert all(abs(v) <= pct for v in pcmd)

    # Out-of-range input is clamped where it is stored, not merely on the way
    # out: a retained >1.0 axis would escape the envelope if scaling ever changed.
    assert backend.set_nudge_vector(roll=9.0, pitch=-9.0, yaw=0.0, gaz=0.0)
    assert backend._nudge_vector == (1.0, -1.0, 0.0, 0.0)
    assert backend._combined_nudge_pcmd() == (pct, -pct, 0, 0)


def test_virtual_stick_is_refused_while_the_pilot_holds_the_sticks(make_backend):
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = True

    assert not backend.set_nudge_vector(roll=1.0, pitch=0.0, yaw=0.0, gaz=0.0)
    assert backend._nudge_vector is None
    assert backend._combined_nudge_pcmd() == (0, 0, 0, 0)
    assert any(
        event == "nudge_blocked_manual" and fields.get("reason") == "vector"
        for event, fields in backend.log.records
    )


def test_virtual_stick_rejects_non_finite_axes(make_backend):
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False

    for bad in (float("nan"), float("inf"), float("-inf")):
        assert not backend.set_nudge_vector(roll=bad, pitch=0.0, yaw=0.0, gaz=0.0)
    assert backend._nudge_vector is None


def test_releasing_the_virtual_stick_zeroes_and_hovers(make_backend):
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False
    assert backend.set_nudge_vector(roll=0.8, pitch=0.0, yaw=0.0, gaz=0.0)
    assert backend._combined_nudge_pcmd() != (0, 0, 0, 0)

    # Centring the knob is reported as an all-zero vector.
    assert backend.set_nudge_vector(roll=0.0, pitch=0.0, yaw=0.0, gaz=0.0)

    assert backend._nudge_vector is None
    assert backend._nudge_held == set()
    assert backend._combined_nudge_pcmd() == (0, 0, 0, 0)
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)
    assert backend.state.tracker_state == "HOVER"


def test_virtual_stick_deadman_zeroes_a_held_stick_when_the_ui_stops(make_backend):
    """A frozen UI stops refreshing the TTL, so a held stick decays to zero."""
    backend = make_backend(skycontroller=False, pulse_s=0.01)
    backend.pilot_sticks = False
    assert backend.set_nudge_vector(roll=0.0, pitch=1.0, yaw=0.0, gaz=0.0)

    deadline = time.monotonic() + 0.5
    while backend._nudge_held and time.monotonic() < deadline:
        time.sleep(0.01)
    thread = backend._nudge_loop_thread
    if thread is not None:
        thread.join(timeout=0.2)

    assert backend._nudge_held == set()
    assert backend.state.tracker_state == "HOVER"
    assert backend.drone.pcmds[-1][1:5] == (0, 0, 0, 0)


# --- audit 2026-08-06: safety commands must never fail silently -------------

def test_a_failed_safety_zero_is_recorded_not_swallowed(make_backend):
    """Every caller proceeds as if the aircraft is holding still after this."""
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False

    def refuse(*_args, **_kwargs):
        raise RuntimeError("link down")

    backend._raw_pcmd = refuse
    assert not backend._zero_pcmd_or_log("unit_test")

    assert backend.zero_pcmd_failures == 1
    assert "link down" in (backend.last_zero_pcmd_error or "")
    assert any(
        event == "pcmd_zero_failed" and fields.get("reason") == "unit_test"
        for event, fields in backend.log.records
    ), "a failed safety zero left no record"


def test_set_tracker_state_logs_old_new_and_reason_on_a_real_transition(make_backend):
    backend = make_backend()
    backend.log.records.clear()

    result = backend._set_tracker_state(TrackerState.RTH, reason="unit_test")

    assert result is TrackerState.RTH
    assert backend.state.tracker_state == TrackerState.RTH
    assert any(
        event == "tracker_state_transition"
        and fields.get("old") == "LINK"
        and fields.get("new") == "RTH"
        and fields.get("reason") == "unit_test"
        for event, fields in backend.log.records
    ), "tracker_state transition left no audit record"


def test_set_tracker_state_does_not_log_when_the_state_does_not_change(make_backend):
    """Hot paths (poll(), the nudge hold loop) re-assert state every cycle;
    logging a no-op transition would flood commands.jsonl/incidents."""
    backend = make_backend()
    backend._set_tracker_state(TrackerState.RTH, reason="setup")
    backend.log.records.clear()

    result = backend._set_tracker_state(TrackerState.RTH, reason="unit_test_noop")

    assert result is TrackerState.RTH
    assert backend.log.records == []


def test_zero_pcmd_reports_failure_not_success_when_there_is_no_drone(make_backend):
    """drone=None means nothing was sent; True would claim a zero landed."""
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False
    backend.drone = None

    assert backend._zero_pcmd_or_log("unit_test") is False

    assert any(
        event == "pcmd_zero_skipped_no_drone" and fields.get("reason") == "unit_test"
        for event, fields in backend.log.records
    ), "a no-drone safety zero left no record"


def test_send_pcmd_reports_failure_not_success_when_there_is_no_drone(make_backend):
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.pilot_sticks = False
    backend.drone = None

    assert backend.send_pcmd(10, 0, 0, 0, reason="unit_test") is False

    assert any(
        event == "pcmd_skipped_no_drone" and fields.get("reason") == "unit_test"
        for event, fields in backend.log.records
    ), "a no-drone pcmd left no record"


def test_land_records_a_failed_safety_zero(make_backend):
    backend = make_backend(skycontroller=False, pulse_s=5.0)
    backend.drone.flight_state = "hovering"
    backend._raw_pcmd = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no link"))

    backend.land_cmd("audit")

    assert backend.zero_pcmd_failures > 0
    assert any(event == "pcmd_zero_failed" for event, _ in backend.log.records)


def test_stick_override_callback_failure_is_logged_and_latches_unhealthy():
    """The callback IS the override. Swallowing it hid a dead takeover path."""
    events = []
    failures = {"n": 0}

    def always_raises(_axes):
        failures["n"] += 1
        raise RuntimeError("give_to_pilot blew up")

    disconnects = []
    monitor = backend_module.SkyControllerStickMonitor(
        on_active=always_raises,
        on_disconnect=disconnects.append,
        log_event=lambda event, **fields: events.append((event, fields)),
    )
    monitor.healthy = True

    limit = backend_module._STICK_CALLBACK_FAIL_LIMIT
    for _ in range(limit):
        try:
            monitor._on_active(monitor.snapshot_axes())
        except Exception as exc:
            monitor._callback_failures += 1
            if monitor._log_event is not None:
                monitor._log_event("stick_override_callback_failed",
                                   error=repr(exc),
                                   consecutive=monitor._callback_failures)
            if monitor._callback_failures >= limit:
                monitor._report_disconnect(f"override callback failed: {exc!r}")

    assert failures["n"] == limit
    assert sum(1 for event, _ in events if event == "stick_override_callback_failed") == limit
    assert not monitor.healthy, "a dead override callback must fail the readiness gate"
    assert disconnects, "an airborne aircraft must be told to land"


def test_stick_monitor_loop_reports_callback_failures(monkeypatch):
    """Guards the real _loop wiring, not just the pattern above."""
    source = "\n".join(
        (
            inspect.getsource(backend_module.SkyControllerStickMonitor._loop),
            inspect.getsource(backend_module.SkyControllerStickMonitor._notify_stick_activity),
        )
    )
    assert "stick_override_callback_failed" in source
    assert "_STICK_CALLBACK_FAIL_LIMIT" in source
    assert "except Exception:\n                        pass" not in source


def test_rth_is_refused_when_the_climb_was_never_read_back(make_backend):
    """Unknown must not mean permitted.

    RTH climbs to the firmware's return_home_min_altitude first, and that value is
    set independently of MaxAltitude. Connecting to an aircraft that never ran the
    readback leaves it unknown -- the firmware keeps whatever the last FreeFlight
    session left (39.2 m on the reference unit), which is exactly the climb this
    guard exists to stop. The fallback is hover.
    """
    backend = make_backend(max_altitude_m=50.0, max_distance_m=100.0)
    backend.drone.flight_state = "flying"
    backend.drone.home_reachable = True
    backend.state.flight_state = "flying"
    assert backend.state.rth_min_altitude_m is None, "fixture already read it back"

    backend._execute_runtime_safety_action("controller_disconnected")

    assert "return_to_home" not in backend.drone.events
    assert "Landing" not in backend.drone.events
    assert backend.state.tracker_state == "HOVER"
    assert any(
        event == "runtime_safety_rth_skipped" and "never read back" in fields["detail"]
        for event, fields in backend.log.records
    ), "RTH was skipped with no record of why"
