from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest

from flight_operator_app import OperatorApp, next_tick_deadline


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
