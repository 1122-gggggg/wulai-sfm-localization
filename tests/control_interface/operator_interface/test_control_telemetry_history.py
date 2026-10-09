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
    attitude, velocity = history.since()
    assert attitude == []
    assert len(velocity) == 512
    assert velocity[0][0] == 8.0
    assert velocity[-1][-1] == pytest.approx(clock["mono"] - 0.03)
    velocity.clear()
    assert len(history.since()[1]) == 512


def test_motion_history_rejects_untrusted_or_repeated_samples():
    clock = {"mono": 50.0, "wall": 1000.0}
    history = _history(clock)
    history.record(0, _event(999.9, yaw=0.1))
    history.record(0, _event(999.9, yaw=0.2))          # same receipt: not new
    history.record(0, _event(1000.5, yaw=0.3))         # future receipt
    history.record(0, _event(990.0, yaw=0.4))          # older than 5 s
    history.record(0, _event(999.95, yaw=float("nan")))
    history.record(0, SimpleNamespace(args={}, date=datetime.fromtimestamp(999.95)))
    assert history.since()[0] == [(0.1, pytest.approx(49.9))]


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
    raw_attitudes = [(0.0, 1.0), (0.2, 1.1), (0.4, 1.2)]
    raw_velocities = [(0.0, 0.0, 1.0), (0.0, 2.0, 1.1), (0.0, 0.0, 1.2)]

    def history_provider(since=None):
        att = [r for r in raw_attitudes if since is None or since[0] is None or r[-1] > since[0]]
        vel = [r for r in raw_velocities if since is None or since[1] is None or r[-1] > since[1]]
        return att, vel

    hooks = replace(hooks, telemetry_history=history_provider)
    runner = pff._FlightLoopRunner(
        hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False, enforce_weak_pose_gate=False
    )
    runner._observe_control_telemetry(1.25)
    runner._observe_control_telemetry(1.3)
    assert len(runner.control_state.yaws) == len(runner.control_state.velocities) == 3
    assert runner.control_state.yaw_at(1.1) == pytest.approx(0.2)
    np.testing.assert_allclose(runner.control_state.displacement(1.0, 1.2), [0.0, 0.2])


def test_control_loop_future_arrival_race_held_until_eligible_tick():
    ctrl = rpf.RouteAutoController(
        [np.zeros(3), np.ones(3)], config=rpf.ControlConfig(inspect_waypoints=())
    )
    samples_att = [(0.1, 10.00), (0.2, 10.01)]
    samples_vel = [(0.0, 1.0, 10.00), (0.0, 2.0, 10.01)]

    def history_provider(since=None):
        att = [r for r in samples_att if since is None or since[0] is None or r[-1] > since[0]]
        vel = [r for r in samples_vel if since is None or since[1] is None or r[-1] > since[1]]
        return att, vel

    hooks = pff.LoopHooks(get_pose=lambda: None, olympe_yaw=lambda: 0.0, send_pcmd=lambda *_: None)
    hooks = replace(hooks, telemetry_history=history_provider)
    runner = pff._FlightLoopRunner(
        hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False, enforce_weak_pose_gate=False
    )
    runner._observe_control_telemetry(10.00)
    assert len(runner.control_state.yaws) == 1
    assert runner.control_state.yaws[0] == (10.00, 0.1)
    assert runner._control_telemetry_stamps[0] == 10.00

    runner._observe_control_telemetry(10.02)
    assert len(runner.control_state.yaws) == 2
    assert runner.control_state.yaws[1] == (10.01, 0.2)
    assert runner._control_telemetry_stamps[0] == 10.01


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
    monkeypatch.setattr(
        telemetry,
        "control_telemetry_history",
        lambda _backend, since=None: (
            [r for r in histories[0] if since is None or since[0] is None or r[-1] > since[0]],
            [r for r in histories[1] if since is None or since[1] is None or r[-1] > since[1]],
        ),
    )
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


def test_incremental_reads_independent_consumers_no_draining():
    clock = {"mono": 10.0, "wall": 100.0}
    history = _history(clock)
    for i in range(10):
        clock["mono"] += 0.05
        clock["wall"] += 0.05
        history.record(0, _event(clock["wall"], yaw=float(i) * 0.1))
        history.record(1, _event(clock["wall"], speedX=float(i), speedY=0.0))

    att_a1, vel_a1 = history.since()
    assert len(att_a1) == 10
    assert len(vel_a1) == 10
    cursor_a = (att_a1[4][-1], vel_a1[4][-1])

    att_b1, vel_b1 = history.since()
    assert len(att_b1) == 10
    assert len(vel_b1) == 10
    cursor_b = (att_b1[7][-1], vel_b1[7][-1])

    for i in range(10, 15):
        clock["mono"] += 0.05
        clock["wall"] += 0.05
        history.record(0, _event(clock["wall"], yaw=float(i) * 0.1))
        history.record(1, _event(clock["wall"], speedX=float(i), speedY=0.0))

    att_a2, vel_a2 = history.since(cursor_a)
    assert len(att_a2) == 10
    assert att_a2[0][0] == pytest.approx(0.5)
    assert att_a2[-1][0] == pytest.approx(1.4)

    att_b2, vel_b2 = history.since(cursor_b)
    assert len(att_b2) == 7
    assert att_b2[0][0] == pytest.approx(0.8)
    assert att_b2[-1][0] == pytest.approx(1.4)

    all_att, all_vel = history.since(None)
    assert len(all_att) == 15
    assert len(all_vel) == 15


def test_incremental_reads_ring_rollover_recovers_without_stalling():
    clock = {"mono": 100.0, "wall": 1000.0}
    history = _history(clock)
    for i in range(10):
        clock["mono"] += 0.01
        clock["wall"] += 0.01
        history.record(0, _event(clock["wall"], yaw=float(i)))
    att, _ = history.since()
    old_cursor = (att[0][-1], None)

    for i in range(10, 600):
        clock["mono"] += 0.01
        clock["wall"] += 0.01
        history.record(0, _event(clock["wall"], yaw=float(i)))

    att_after, _ = history.since(old_cursor)
    assert len(att_after) == 512
    assert att_after[-1][0] == pytest.approx(599.0)


def test_incremental_reads_monotonic_cutoff_and_reset():
    clock = {"mono": 100.0, "wall": 1000.0}
    history = _history(clock)
    for i in range(5):
        clock["mono"] += 0.01
        clock["wall"] += 0.01
        history.record(0, _event(clock["wall"], yaw=float(i)))
        history.record(1, _event(clock["wall"], speedX=float(i), speedY=0.0))

    future_cursor = (500.0, 500.0)
    att, vel = history.since(future_cursor)
    assert att == [] and vel == []

    att_all, vel_all = history.since(None)
    assert len(att_all) == 5 and len(vel_all) == 5

def test_future_timestamp_rejected_does_not_drop_subsequent_usable_samples():
    clock = {"mono": 50.0, "wall": 1000.0}
    history = _history(clock)

    history.record(0, _event(999.9, yaw=0.1))
    assert len(history.since()[0]) == 1

    history.record(0, _event(1000.1, yaw=99.0))
    att, _ = history.since()
    assert len(att) == 1
    assert att[0][0] == pytest.approx(0.1)

    clock["mono"] += 0.05
    clock["wall"] += 0.05
    history.record(0, _event(1000.02, yaw=0.2))
    att, _ = history.since()
    assert len(att) == 2
    assert att[1][0] == pytest.approx(0.2)


def test_missing_fields_and_corrupt_payload_rejected_safely():
    clock = {"mono": 50.0, "wall": 1000.0}
    history = _history(clock)
    history.record(0, SimpleNamespace(args=None, date=datetime.fromtimestamp(999.9)))
    history.record(0, SimpleNamespace(args={"wrong_field": 1.0}, date=datetime.fromtimestamp(999.9)))
    history.record(0, SimpleNamespace(args={"yaw": "not_a_number"}, date=datetime.fromtimestamp(999.9)))
    assert history.since() == ([], [])


def test_motion_ring_unread_eviction_marker_is_permanently_recorded(tmp_path):
    import json
    from backend_contract import InterfaceMode
    from disk_policy import enforce_retention
    from session_logs import SessionLogs

    clock = {"mono": 100., "wall": 5000.}
    history = telemetry.MotionEventHistory(
        maxlen=2, now=lambda: clock["mono"], wall_now=lambda: clock["wall"]
    )
    for index in range(3):
        clock["mono"] += .01
        clock["wall"] += .01
        history.record(0, _event(clock["wall"], yaw=float(index)))
    health = history.diagnostic_snapshot()
    assert health["evicted_total"] == [1, 0]
    assert health["unread_eviction_since_log_cursor"] == [True, False]
    assert history.diagnostic_snapshot((clock["mono"], None))["unread_eviction_since_log_cursor"] == [False, False]
    session = SessionLogs.create(tmp_path, mode=InterfaceMode.SIMULATED_STREAM, manifest={})
    backend = SimpleNamespace(session_logs=session, motion_history=history)
    telemetry._log_control_history(backend, clock["mono"])
    session.close(reason="history_loss")
    enforce_retention(tmp_path, current_session=None, max_bytes=0)
    rows = [json.loads(line) for line in (session.directory / "trajectory.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "control_motion_history"
    assert rows[0]["attitude_samples"][0][0] == 1.
    assert rows[0]["history_health"]["unread_eviction_since_log_cursor"] == [True, False]
