"""Offline Olympe motion-event history and control-consumer contracts."""

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import backend_telemetry as telemetry
import path_follow_flight as pff
import real_path_follow_controller as rpf


def _event(wall_s, **args):
    return SimpleNamespace(args=args, date=datetime.fromtimestamp(wall_s))


def _history(clock):
    return telemetry.MotionEventHistory(
        now=lambda: clock["mono"], wall_now=lambda: clock["wall"]
    )


def test_motion_events_between_polls_are_retained_with_receipt_timestamps():
    clock = {"mono": 100.0, "wall": 5000.0}
    history = _history(clock)
    for index in range(520):
        clock["mono"] += 0.01
        clock["wall"] += 0.01
        # Delivered 30 ms after receipt: the stamp is the receipt, not delivery.
        history.record(1, _event(clock["wall"] - 0.03, speedX=float(index), speedY=0.0))
    attitude, velocity = history.histories()
    assert attitude == []
    assert len(velocity) == 512
    assert velocity[0][0] == 8.0
    assert velocity[-1][-1] == pytest.approx(clock["mono"] - 0.03)
    velocity.clear()
    assert len(history.histories()[1]) == 512


def test_motion_history_rejects_untrusted_or_repeated_samples():
    clock = {"mono": 50.0, "wall": 1000.0}
    history = _history(clock)
    history.record(0, _event(999.9, yaw=0.1))
    history.record(0, _event(999.9, yaw=0.2))          # same receipt: not new
    history.record(0, _event(1000.5, yaw=0.3))         # future receipt
    history.record(0, _event(990.0, yaw=0.4))          # older than 5 s
    history.record(0, _event(999.95, yaw=float("nan")))
    history.record(0, SimpleNamespace(args={}, date=datetime.fromtimestamp(999.95)))
    assert history.histories()[0] == [(0.1, pytest.approx(49.9))]


def test_motion_history_subscribes_to_attitude_and_speed_and_detaches(monkeypatch):
    # Other tests may shadow the olympe package; the message type is irrelevant here.
    monkeypatch.setattr(
        telemetry, "olympe_message",
        lambda _module, name: lambda **kwargs: SimpleNamespace(name=name, kwargs=kwargs),
    )
    subscriptions = []

    class Subscription:
        def __init__(self, callback, expectation):
            self.callback, self.expectation, self.active = callback, expectation, True

        def unsubscribe(self):
            self.active = False

    class Drone:
        def subscribe(self, callback, expectation):
            subscriptions.append(Subscription(callback, expectation))
            return subscriptions[-1]

    clock = {"mono": 10.0, "wall": 100.0}
    history = _history(clock)
    history.attach(Drone())
    assert len(subscriptions) == 2
    subscriptions[0].callback(_event(99.98, yaw=0.5), None)
    subscriptions[1].callback(_event(99.99, speedX=1.0, speedY=2.0), None)
    backend = SimpleNamespace(motion_history=history)
    attitude, velocity = telemetry.control_telemetry_history(backend)
    assert attitude == [(0.5, pytest.approx(9.98))]
    assert velocity == [(1.0, 2.0, pytest.approx(9.99))]
    history.detach()
    assert not any(item.active for item in subscriptions)
    assert telemetry.control_telemetry_history(SimpleNamespace()) == ([], [])


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


def test_backend_keeps_connecting_when_motion_subscriptions_fail():
    import olympe_live_backend as backend_module

    backend = backend_module.OlympeLiveBackend.__new__(backend_module.OlympeLiveBackend)
    events = []
    backend.log = SimpleNamespace(event=lambda name, **fields: events.append((name, fields)))
    backend.motion_history = None
    backend.drone = SimpleNamespace()          # no subscribe(): old SDK or test fake
    backend._attach_motion_history()
    assert backend.motion_history is None
    assert events and events[0][0] == "motion_history_unavailable"
