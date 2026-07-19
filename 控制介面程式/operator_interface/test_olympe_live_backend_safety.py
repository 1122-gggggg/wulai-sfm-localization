from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

import olympe_live_backend as backend_module


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
        self.battery_pct = 100.0
        self.gps_fixed = 1
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
        if name in self.setting_states:
            return dict(self.setting_states[name])
        if name == "AttitudeChanged":
            return {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        if name == "AltitudeChanged":
            return {"altitude": 0.0}
        return {}

    def connection_state(self):
        return SimpleNamespace(name="Connected" if self.connected else "Disconnected")

    def disconnect(self):
        self.events.append("disconnect")
        return True


class _FakeLog:
    def __init__(self):
        self.records: list[tuple[str, dict]] = []

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
        },
        "olympe.messages.skyctrl.CoPiloting": {
            "setPilotingSource": _message_factory("setPilotingSource"),
        },
        "olympe.messages.skyctrl.CoPilotingState": {
            "pilotingSource": _message_factory("pilotingSource"),
        },
        "olympe.messages.camera": {
            "stop_recording": _message_factory("stop_recording"),
            "recording_state": _message_factory("recording_state"),
        },
        "olympe.features.media": {
            "download_media": _message_factory("download_media"),
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def make_backend(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_fake_olympe(monkeypatch)
    monkeypatch.setattr(backend_module, "_DEFAULT_RECORD_DIR", tmp_path)
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
            with_video=with_video,
        )
        backend.log = _FakeLog()
        backend.drone = _FakeDrone()
        return backend

    return make


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
