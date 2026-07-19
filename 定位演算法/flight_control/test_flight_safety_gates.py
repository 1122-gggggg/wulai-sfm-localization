#!/usr/bin/env python3
"""Failure-injection tests for the real-flight safety gates (pure python, no drone).

Every test drives the REAL path_follow_flight.run_loop / SafetyMonitor /
command_to_body_percent / OlympePdrawGrabber logic with mock hooks. No olympe,
no torch, no GPU, no hardware.

Run:  pytest -q sfm_system/定位/mission/flight_control/test_flight_safety_gates.py
"""
from __future__ import annotations

import builtins
import inspect
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import olympe_frame_source as ofs
import path_follow_flight as pff
import real_path_follow_controller as rpf

ZERO = (0, 0, 0, 0)


def run_ticks(max_ticks, pose_fn, *, route=((0.0, 0.0, 0.0), (8.0, 0.0, 0.0)),
              hooks_extra=None):
    """Run the real run_loop against mock hooks on a virtual clock.

    pose_fn(st) -> rpf.Pose | None; st has "t" (virtual now) and "tick".
    Returns (sent_pcmds, terminal_reason, log_records).
    """
    wp = [np.array(p, float) for p in route]
    ctrl = rpf.RouteAutoController(wp, poles=[],
                                   config=rpf.ControlConfig(inspect_waypoints=()))
    sent, records = [], []
    st = {"calls": 0, "t": 0.0, "tick": 0}

    def now():
        st["calls"] += 1
        if st["calls"] > max_ticks * 4:
            raise KeyboardInterrupt
        st["t"] += 1.0 / pff.CTRL_HZ
        return st["t"]

    def get_pose():
        st["tick"] += 1
        return pose_fn(st)

    extra = dict(hooks_extra or {})
    extra.setdefault("log_tick", records.append)
    hooks = pff.LoopHooks(
        get_pose=get_pose,
        olympe_yaw=lambda: None,
        send_pcmd=lambda r, p, y, g: sent.append((r, p, y, g)),
        now=now,
        **extra,
    )
    try:
        reason = pff.run_loop(hooks, ctrl, wp, verbose=False)
    except KeyboardInterrupt:
        reason = "tick cap"
    return sent, reason, records


def fresh_pose(st, x=0.0, y=0.0, z=0.0, yaw=0.0):
    return rpf.Pose(x=x, y=y, z=z, yaw=yaw, stamp=st["t"])


# ---------------------------------------------------------------------------
# Mode separation

def test_selftest_and_dry_run_never_import_olympe(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def reject_olympe(name, *args, **kwargs):
        if name == "olympe" or name.startswith("olympe."):
            raise AssertionError(f"offline mode attempted to import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_olympe)
    pff.selftest()
    route = tmp_path / "flight_path.json"
    route.write_text(
        json.dumps({"waypoints": [[0, 0, 0], [8, 0, 0]]}), encoding="utf-8"
    )
    monkeypatch.setattr(pff, "PATH_JSON", str(route))
    log = tmp_path / "dry_cmdlog.jsonl"
    state, progress = pff.dry_run(1, steps=300, cmd_log_path=str(log))
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert lines and all(r.get("sink") == "dry-run" for r in lines)
    assert lines[-1].get("event") == "terminal"


def test_grab_only_source_never_arms():
    src = inspect.getsource(pff.grab_only)
    for token in ("TakeOff", "PCMD", "Landing", "Emergency", "moveBy", "moveTo"):
        assert token not in src, f"grab_only must never reference {token}"


def test_fly_is_the_only_arming_entrypoint():
    module_src = inspect.getsource(pff)
    fly_src = inspect.getsource(pff.fly)
    # every arming/movement Olympe message used by the module must live in fly()
    for token in ("TakeOff", "Emergency"):
        assert module_src.count(f"drone({token}") == fly_src.count(f"drone({token}")


# ---------------------------------------------------------------------------
# Frame / stream gates

def test_stream_lost_zero_then_land():
    def never_localize(st):
        raise AssertionError("stream gate must run before localization")
    sent, reason, records = run_ticks(
        400, never_localize, hooks_extra={"stream_healthy": lambda: False})
    assert sent and all(c == ZERO for c in sent)
    assert "stream lost" in reason
    assert records and all(r["blocked"] for r in records)
    assert any("stream" in r["reason"] for r in records)


def _install_fake_pdraw_modules(monkeypatch):
    class Callbacks:
        def __init__(self, **kwargs):
            self.media_type = kwargs.get("media_type")
            self.video_frame = kwargs.get("video_frame")
            self.flush = kwargs.get("flush")

    class StreamInfoSingle:
        def __init__(self):
            self.name = None
            self.has_renderer = None
            self.callbacks = []

    class StreamInfoDefault(StreamInfoSingle):
        pass

    fake_deps = SimpleNamespace(VDEF_FRAME_TYPE_RAW="raw")
    fake_pdraw = SimpleNamespace(
        Callbacks=Callbacks,
        StreamInfoDefault=StreamInfoDefault,
        StreamInfoSingle=StreamInfoSingle,
        h264_coded_data_format=SimpleNamespace(bytestream="bytestream"),
    )
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "olympe_deps":
            return fake_deps
        if name == "olympe.features.video.pdraw":
            return fake_pdraw
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_live_multistream_keeps_raw_callbacks_without_renderer(monkeypatch):
    _install_fake_pdraw_modules(monkeypatch)

    class Streaming:
        def play_multiple_stream(self, *, streams_list, timeout):
            self.streams = streams_list
            self.timeout = timeout
            return True

    streaming = Streaming()
    grabber = ofs.OlympePdrawGrabber(SimpleNamespace(streaming=streaming), resize=None)
    grabber._start_with_play("DefaultVideo")

    assert len(streaming.streams) == 2
    for stream in streaming.streams:
        assert stream.has_renderer is False
        assert len(stream.callbacks) == 1
        callback = stream.callbacks[0]
        assert callback.media_type == ("raw", None)
        assert callable(callback.video_frame)


def test_live_single_stream_keeps_raw_callback_without_renderer(monkeypatch):
    _install_fake_pdraw_modules(monkeypatch)

    class Streaming:
        def play(self, **kwargs):
            self.kwargs = kwargs
            return True

    streaming = Streaming()
    grabber = ofs.OlympePdrawGrabber(SimpleNamespace(streaming=streaming), resize=None)
    grabber._start_with_play("DefaultVideo")

    assert streaming.kwargs["has_renderer"] is False
    assert callable(streaming.kwargs["raw_cb"])
    assert streaming.kwargs["data_formats"] == ["bytestream"]


def test_frozen_stream_goes_unhealthy():
    g = ofs.OlympePdrawGrabber.__new__(ofs.OlympePdrawGrabber)
    import threading
    g._lock = threading.Lock(); g._latest = None; g._stamp = 0.0; g._n = 0
    g._digest = None; g._dup_n = 0; g._frozen_warned = False
    g.stale_s = 5.0
    frame = (np.arange(720 * 1280 * 3, dtype=np.uint8) % 251).reshape(720, 1280, 3)
    for _ in range(ofs.FROZEN_DUP_FRAMES + 1):
        g._store(frame)
    assert g() is None, "frozen (duplicated) stream must yield no frame"
    assert not g.is_healthy(), "frozen stream must be unhealthy despite fresh stamps"
    g._store(frame.copy() + 1)
    assert g.is_healthy(), "a distinct new frame must recover the stream"


def test_frame_sample_carries_capture_timestamp():
    g = ofs.OlympePdrawGrabber.__new__(ofs.OlympePdrawGrabber)
    import threading
    g._lock = threading.Lock(); g._latest = None; g._stamp = 0.0; g._n = 0
    g._digest = None; g._dup_n = 0; g._frozen_warned = False
    g.stale_s = 5.0
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    g._store(frame)
    sample = g()
    assert isinstance(sample, tuple) and len(sample) == 2
    out, capture_stamp = sample
    assert np.array_equal(out, frame)
    assert capture_stamp == pytest.approx(g._stamp)


def test_olympe_ntp_timestamp_maps_to_callback_monotonic_and_detects_backlog():
    class _Frame:
        def __init__(self, ntp_us):
            self.ntp_us = ntp_us

        def info(self):
            return {"ntp_raw_timestamp": self.ntp_us}

    g = ofs.OlympePdrawGrabber.__new__(ofs.OlympePdrawGrabber)
    g._source_ntp_us = None
    g._source_to_monotonic = None
    stamp0, kind0, source0 = g._capture_stamp(_Frame(1_000_000), receipt_stamp=10.0)
    stamp1, kind1, source1 = g._capture_stamp(_Frame(1_100_000), receipt_stamp=10.6)
    assert stamp0 == pytest.approx(10.0) and kind0 == "ntp-anchor"
    assert stamp1 == pytest.approx(10.1) and kind1 == "ntp-mapped"
    assert (source0, source1) == (1_000_000, 1_100_000)
    assert 10.6 - stamp1 == pytest.approx(0.5), "decode/queue backlog must remain visible"


def test_olympe_timestamp_nonincrease_restart_and_missing_metadata_fail_closed():
    class _Frame:
        def __init__(self, payload=None, raises=False):
            self.payload = payload
            self.raises = raises

        def info(self):
            if self.raises:
                raise RuntimeError("metadata unavailable")
            return self.payload

    g = ofs.OlympePdrawGrabber.__new__(ofs.OlympePdrawGrabber)
    g._source_ntp_us = None
    g._source_to_monotonic = None
    g._capture_stamp(_Frame({"ntp_raw_timestamp": 2_000_000}), 20.0)
    stamp, kind, _source = g._capture_stamp(_Frame({"ntp_raw_timestamp": 2_000_000}), 20.1)
    assert stamp is None and kind == "ntp-nonincreasing"
    stamp, kind, _source = g._capture_stamp(_Frame({"ntp_raw_timestamp": 100_000}), 20.2)
    assert stamp == pytest.approx(20.2) and kind == "ntp-restart-receipt"
    stamp, kind, source = g._capture_stamp(_Frame(raises=True), 20.3)
    assert stamp == pytest.approx(20.3) and kind == "callback-receipt"
    assert source is None


def test_strict_flight_stream_rejects_degraded_receipt_timestamps():
    import threading

    g = ofs.OlympePdrawGrabber.__new__(ofs.OlympePdrawGrabber)
    g._lock = threading.Lock(); g._latest = None; g._stamp = 0.0; g._n = 0
    g._digest = None; g._dup_n = 0; g._frozen_warned = False
    g.require_source_timestamps = True
    g.stale_s = 5.0
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    g._store(frame, stamp_source="callback-receipt")
    assert g() is None and not g.is_healthy()
    assert g.timestamp_degraded
    g._store(frame + 1, stamp_source="ntp-mapped", source_ntp_us=1_100_000)
    assert g() is not None and g.is_healthy()
    assert not g.timestamp_degraded


def test_inspection_source_drain_rejects_preconfirm_pipeline_frames():
    import threading

    g = ofs.OlympePdrawGrabber.__new__(ofs.OlympePdrawGrabber)
    g._lock = threading.Lock(); g._latest = None; g._stamp = 0.0; g._n = 0
    g._digest = None; g._dup_n = 0; g._frozen_warned = False
    g.require_source_timestamps = True
    g.stale_s = 5.0
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    g._store(frame, stamp_source="ntp-mapped", source_ntp_us=1_000_000,
             receipt_stamp=99.0)
    baseline = g.latest_source_ntp_us()
    assert baseline == 1_000_000

    # A frame exposed before confirmation can arrive afterward because of the
    # fixed video delay. Source time has not advanced through the configured drain.
    g._store(frame + 1, stamp_source="ntp-mapped", source_ntp_us=1_400_000,
             receipt_stamp=100.2)
    assert g.inspection_sample_after(
        baseline, min_source_advance_s=1.0, receipt_after=101.0) is None

    # Source time alone is insufficient: this one was already queued before the
    # host-side drain deadline.
    g._store(frame + 2, stamp_source="ntp-mapped", source_ntp_us=2_100_000,
             receipt_stamp=100.9)
    assert g.inspection_sample_after(
        baseline, min_source_advance_s=1.0, receipt_after=101.0) is None

    g._store(frame + 3, stamp_source="ntp-mapped", source_ntp_us=2_200_000,
             receipt_stamp=101.1)
    assert g.inspection_sample_after(
        baseline, min_source_advance_s=1.0, receipt_after=101.0) is not None


def test_tracker_pose_uses_capture_not_inference_completion_stamp(monkeypatch):
    import production_xfeat_tracker as pxt

    class _Rotation:
        @staticmethod
        def matrix():
            return np.eye(3)

    class _Transform:
        rotation = _Rotation()
        translation = np.zeros(3)

    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig()
    tracker.state = pxt.RuntimeState()
    tracker._flow_enabled = False
    ret = {"cam_from_world": _Transform()}
    info = {"inliers": tracker.cfg.acquire_min_inliers, "reproj_rms": 0.5}
    tracker._localize_frame_deep = lambda _frame: tracker._gate(ret, info, "BOOT_INIT")[2]

    pose = tracker.localize_frame(np.zeros((4, 6, 3), np.uint8), capture_stamp=12.5)
    assert pose is not None and pose.stamp == pytest.approx(12.5)


@pytest.mark.parametrize("new_mode, expected", [("HOVER", "zero"), ("MANUAL", "silent")])
def test_mid_inference_mode_change_cannot_drive(new_mode, expected):
    mode = {"value": "AUTO"}

    def flip_mode(st):
        mode["value"] = new_mode
        return fresh_pose(st)

    sent, _reason, _records = run_ticks(
        30, flip_mode, hooks_extra={"safety_poll": lambda: mode["value"]})
    assert not any(cmd != ZERO for cmd in sent)
    if expected == "zero":
        assert ZERO in sent


def test_stream_failure_during_inference_cannot_drive():
    stream = {"healthy": True}

    def lose_stream(st):
        stream["healthy"] = False
        return fresh_pose(st)

    sent, _reason, _records = run_ticks(
        30, lose_stream, hooks_extra={"stream_healthy": lambda: stream["healthy"]})
    assert sent and all(cmd == ZERO for cmd in sent)


def test_slow_inference_cannot_refresh_old_frame_pose():
    def slow_pose(st):
        capture_stamp = st["t"]
        st["t"] += pff.POSE_STALE_S + 0.2
        return rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=capture_stamp)

    sent, reason, _records = run_ticks(200, slow_pose)
    assert sent and all(cmd == ZERO for cmd in sent)
    assert reason == "localization lost -> land"


# ---------------------------------------------------------------------------
# Localization gates

def test_lost_pose_zero_then_land_never_emergency():
    sent, reason, _ = run_ticks(200, lambda st: None)
    assert sent and all(c == ZERO for c in sent)
    assert reason == "localization lost -> land"
    assert "EMERGENCY" not in reason


def test_weak_pose_hovers():
    sent, reason, _ = run_ticks(
        600, fresh_pose, hooks_extra={"pose_is_weak": lambda: True})
    assert sent and all(c == ZERO for c in sent)
    assert reason == "low confidence -> land"


def test_low_inliers_hovers():
    sent, reason, _ = run_ticks(
        600, fresh_pose, hooks_extra={"pose_confidence": lambda: 10})
    assert sent and all(c == ZERO for c in sent)
    assert reason == "low confidence -> land"


def test_recovery_requires_two_consecutive_good_fixes_before_motion():
    quality = {"weak": True}

    def pose_fn(st):
        quality["weak"] = st["tick"] == 1
        return fresh_pose(st)

    sent, _reason, records = run_ticks(
        30, pose_fn, hooks_extra={"pose_is_weak": lambda: quality["weak"]})

    assert sent[0] == ZERO
    assert sent[1] == ZERO
    assert "recovery confirmation 1/2" in records[1]["reason"]
    assert any(cmd != ZERO for cmd in sent[2:])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_pose_hovers(bad):
    sent, reason, _ = run_ticks(200, lambda st: fresh_pose(st, x=bad))
    assert sent and all(c == ZERO for c in sent)
    assert reason == "localization lost -> land"


def test_pose_jump_rejected_then_confirmed():
    def pose_fn(st):
        return fresh_pose(st) if st["tick"] <= 3 else fresh_pose(st, x=3.0)
    sent, _reason, records = run_ticks(20, pose_fn)
    assert sent[3] == ZERO, "first jumped fix must hover, not steer"
    assert any(r.get("jump_reject_u") for r in records), "jump must be logged"
    assert any(c != ZERO for c in sent[4:]), "confirmed relocation must resume driving"


def test_stale_pose_hovers():
    # stamps frozen in the past -> freshness gate blocks, hover, then land
    sent, reason, _ = run_ticks(200, lambda st: rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=-10.0))
    assert sent and all(c == ZERO for c in sent)
    assert reason == "localization lost -> land"


# ---------------------------------------------------------------------------
# Route gates

def test_route_deviation_lands():
    sent, reason, _ = run_ticks(20, lambda st: fresh_pose(st, z=5.0))
    assert "route deviation" in reason
    assert sent[-1] == ZERO, "deviation abort must end on zero PCMD"


def test_route_completion_lands():
    sent, reason, _ = run_ticks(20, lambda st: fresh_pose(st, x=7.99))
    assert reason == "route complete -> land"
    assert sent[-1] == ZERO


def test_invalid_route_rejected(tmp_path):
    bad = tmp_path / "route.json"
    bad.write_text(json.dumps({"waypoints": [[0.0, 0.0, 0.0]]}))
    with pytest.raises(ValueError):
        rpf.load_waypoints(bad)


@pytest.mark.parametrize("waypoints", [
    [[0.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
])
def test_malformed_or_zero_length_route_rejected(tmp_path, waypoints):
    bad = tmp_path / "route.json"
    bad.write_text(json.dumps({"waypoints": waypoints}))
    with pytest.raises(ValueError):
        rpf.load_waypoints(bad)


def test_direct_controller_route_validation():
    with pytest.raises(ValueError):
        rpf.RouteAutoController([np.zeros(3), np.array([np.inf, 0.0, 0.0])])
    with pytest.raises(ValueError):
        rpf.RouteAutoController([np.zeros(3), np.zeros(3)])


def test_non_finite_controller_config_rejected():
    with pytest.raises(ValueError):
        rpf.ControlConfig(speed=float("nan"))
    with pytest.raises(ValueError):
        rpf.ControlConfig(max_pose_age_s=0.0)


def test_malformed_pole_rejected(tmp_path):
    bad = tmp_path / "poles.json"
    bad.write_text(json.dumps({"poles": [{"center": [0.0, float("nan"), 0.0]}]}))
    with pytest.raises(ValueError):
        rpf.load_poles(bad)


def test_boot_lock_requires_consecutive_fixes_near_route_start():
    lock = pff.BootPoseLock(np.zeros(3), required_fixes=3,
                            max_start_distance=1.0, max_fix_jump=0.4)
    assert not lock.observe(rpf.Pose(0.1, 0.0, 0.0, 0.0, stamp=1.0), now=1.1)
    assert not lock.observe(rpf.Pose(0.2, 0.0, 0.0, 0.0, stamp=1.1), now=1.2)
    assert lock.observe(rpf.Pose(0.25, 0.0, 0.0, 0.0, stamp=1.2), now=1.3)
    assert not lock.observe(rpf.Pose(3.0, 0.0, 0.0, 0.0, stamp=1.3), now=1.4)
    assert lock.count == 0


def test_inspection_requires_explicit_capture_ack():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([1.0, 0.0, 0.4]),
            "base": np.array([1.0, 0.0, 0.0]), "top": np.array([1.0, 0.0, 0.8])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.3))
    pose = rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=2.0)
    cmd = ctrl.step(pose, now=2.0)
    assert cmd.action == "INSPECT" and np.allclose(cmd.vel_map, 0.0)
    assert ctrl.completed_inspections == set()
    assert not ctrl.ack_inspection(cmd.look_at_pole), "orientation confirmation is mandatory"
    assert ctrl.ack_inspection(cmd.look_at_pole, orientation_confirmed=True)
    assert ctrl.completed_inspections == {1}


def test_inspection_command_targets_pole_bearing_not_route_heading():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([1.0, 0.0, 2.0]),
            "base": np.array([1.0, 0.0, 2.0]), "top": np.array([1.0, -1.0, 2.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.3))
    cmd = ctrl.step(rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=2.0), now=2.0)
    assert cmd.action == "INSPECT"
    assert cmd.yaw_target == pytest.approx(np.pi / 2.0)
    assert cmd.look_at_pole["target"] == [1.0, 0.0, 2.0]


def test_inspection_gimbal_pitch_is_derived_from_target_and_pose():
    pose = rpf.Pose(0.0, -2.0, 0.0, 0.0, stamp=1.0)  # camera up=+2 in -Y-up map
    meta = {"waypoint": 2, "pole_id": 1, "target": [2.0, 0.0, 0.0]}
    pitch = pff.inspection_gimbal_pitch_deg(meta, pose)
    assert pitch == pytest.approx(-45.0)
    assert pff.inspection_gimbal_pitch_deg({"target": [np.nan, 0.0, 0.0]}, pose) is None


@pytest.mark.parametrize("capture_ok", [False, True])
def test_run_loop_inspection_hook_controls_completion(capture_ok):
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([2.0, 0.0, 0.0]),
            "base": np.array([2.0, 0.0, 0.0]), "top": np.array([2.0, -1.0, 0.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.3))
    clock = {"t": 1.0, "calls": 0, "acks": 0}
    sent = []

    def now():
        clock["calls"] += 1
        if clock["calls"] > 30:
            raise KeyboardInterrupt
        clock["t"] += 0.05
        return clock["t"]

    def inspect(_meta, _pose):
        clock["acks"] += 1
        return capture_ok

    hooks = pff.LoopHooks(
        get_pose=lambda: rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=clock["t"]),
        olympe_yaw=lambda: 0.0,
        send_pcmd=lambda *cmd: sent.append(tuple(cmd)),
        inspection_ack=inspect,
        now=now,
    )
    with pytest.raises(KeyboardInterrupt):
        pff.run_loop(hooks, ctrl, wp, verbose=False)
    assert clock["acks"] > 0
    assert (1 in ctrl.completed_inspections) is capture_ok
    if not capture_ok:
        assert sent and all(cmd == ZERO for cmd in sent)


def test_inspection_zero_hold_precedes_blocking_capture_hook():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([2.0, 0.0, 0.0]),
            "base": np.array([2.0, 0.0, 0.0]), "top": np.array([2.0, -1.0, 0.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.3))
    events, clock = [], {"t": 1.0, "calls": 0}

    def now():
        clock["calls"] += 1
        if clock["calls"] > 12:
            raise KeyboardInterrupt
        clock["t"] += 0.05
        return clock["t"]

    def authorized(pcmd):
        events.append(("send", tuple(pcmd)))
        return True, "authorized", tuple(pcmd)

    def capture(_meta, _pose):
        events.append(("capture-start", None))
        return False

    hooks = pff.LoopHooks(
        get_pose=lambda: rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=clock["t"]),
        olympe_yaw=lambda: 0.0,
        send_pcmd=lambda *cmd: events.append(("raw", tuple(cmd))),
        send_authorized_pcmd=authorized,
        inspection_ack=capture,
        now=now,
    )
    with pytest.raises(KeyboardInterrupt):
        pff.run_loop(hooks, ctrl, wp, verbose=False)
    first_capture = next(i for i, e in enumerate(events) if e[0] == "capture-start")
    assert events[first_capture - 1] == ("send", ZERO)
    assert not any(e[0] == "raw" for e in events)


def test_unaligned_inspection_yaws_without_capture_ack():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([1.0, 0.0, 2.0]),
            "base": np.array([1.0, 0.0, 2.0]), "top": np.array([1.0, -1.0, 2.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.3))
    clock = {"t": 1.0, "calls": 0, "captures": 0}
    sent = []

    def now():
        clock["calls"] += 1
        if clock["calls"] > 15:
            raise KeyboardInterrupt
        clock["t"] += 0.05
        return clock["t"]

    hooks = pff.LoopHooks(
        get_pose=lambda: rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=clock["t"]),
        olympe_yaw=lambda: 0.0,
        send_pcmd=lambda *cmd: sent.append(tuple(cmd)),
        inspection_ack=lambda *_: clock.__setitem__("captures", clock["captures"] + 1),
        now=now,
    )
    with pytest.raises(KeyboardInterrupt):
        pff.run_loop(hooks, ctrl, wp, verbose=False)
    assert clock["captures"] == 0
    assert any(cmd[2] != 0 and cmd[1] == 0 and cmd[3] == 0 for cmd in sent)


def test_missing_body_yaw_telemetry_never_acknowledges_inspection():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([2.0, 0.0, 0.0]),
            "base": np.array([2.0, 0.0, 0.0]), "top": np.array([2.0, -1.0, 0.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.3))
    clock = {"t": 1.0, "calls": 0, "captures": 0}
    sent = []

    def now():
        clock["calls"] += 1
        if clock["calls"] > 15:
            raise KeyboardInterrupt
        clock["t"] += 0.05
        return clock["t"]

    hooks = pff.LoopHooks(
        get_pose=lambda: rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=clock["t"]),
        olympe_yaw=lambda: None,
        send_pcmd=lambda *cmd: sent.append(tuple(cmd)),
        inspection_ack=lambda *_: clock.__setitem__("captures", clock["captures"] + 1),
        now=now,
    )
    with pytest.raises(KeyboardInterrupt):
        pff.run_loop(hooks, ctrl, wp, verbose=False)
    assert clock["captures"] == 0
    assert sent and all(cmd == ZERO for cmd in sent)


def test_inspection_radius_jitter_does_not_reset_timeout(monkeypatch):
    monkeypatch.setattr(pff, "INSPECTION_TIMEOUT_S", 0.4)
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([2.0, 0.0, 0.0]),
            "base": np.array([2.0, 0.0, 0.0]), "top": np.array([2.0, -1.0, 0.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.2))
    clock = {"t": 1.0, "calls": 0, "poses": 0}

    def now():
        clock["calls"] += 1
        if clock["calls"] > 160:
            raise KeyboardInterrupt
        clock["t"] += 0.05
        return clock["t"]

    def pose():
        clock["poses"] += 1
        x = 1.0 if clock["poses"] % 2 else 1.25
        return rpf.Pose(x, 0.0, 0.0, 0.0, stamp=clock["t"])

    hooks = pff.LoopHooks(
        get_pose=pose,
        olympe_yaw=lambda: 0.0,
        send_pcmd=lambda *_: None,
        inspection_ack=lambda *_: False,
        now=now,
    )
    try:
        reason = pff.run_loop(hooks, ctrl, wp, verbose=False)
    except KeyboardInterrupt:
        reason = "tick cap"
    assert reason == "inspection alignment/capture timeout -> land"


def test_pending_inspection_at_route_end_aborts_instead_of_infinite_auto():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([1.0, 0.0, 0.0]),
            "base": np.array([1.0, 0.0, 0.0]), "top": np.array([1.0, -1.0, 0.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.2))
    cmd = ctrl.step(rpf.Pose(3.0, 0.0, 0.0, 0.0, stamp=2.0), now=2.0)
    assert cmd.should_land and cmd.action == "ABORT"
    assert "pending inspection" in cmd.status.lower()


def test_heading_unavailable_is_none_before_motion():
    he = pff.HeadingEstimator()
    assert he.heading(None) is None, "no motion, no olympe yaw -> heading must be None"


# ---------------------------------------------------------------------------
# Safety switch / watchdog

def test_manual_sends_nothing():
    sent, _, records = run_ticks(30, fresh_pose,
                                 hooks_extra={"safety_poll": lambda: "MANUAL"})
    assert sent == [], "MANUAL: autonomy must not compete with the pilot's sticks"
    assert records and all(r["pcmd"] is None for r in records)


def test_hover_sends_zero():
    sent, _, _ = run_ticks(30, fresh_pose,
                           hooks_extra={"safety_poll": lambda: "HOVER"})
    assert sent and all(c == ZERO for c in sent)


def test_land_sends_zero_then_breaks():
    sent, reason, _ = run_ticks(30, fresh_pose,
                                hooks_extra={"safety_poll": lambda: "LAND"})
    assert sent == [ZERO], "LAND: exactly one zero PCMD before Landing"
    assert reason == "safety LAND command -> land"


def test_emergency_breaks_without_pcmd():
    sent, reason, _ = run_ticks(30, fresh_pose,
                                hooks_extra={"safety_poll": lambda: "EMERGENCY"})
    assert sent == [], "EMERGENCY must not stream PCMD; motor cut is the fly() finally's job"
    assert reason.startswith("EMERGENCY")


def test_safety_switch_preserves_existing_non_auto(tmp_path):
    command = tmp_path / "safety.cmd"
    command.write_text("land\n")
    switch = pff.SafetySwitch(command, keyboard=False)
    assert switch.poll() == "LAND"
    assert command.read_text() == "land\n"


def test_safety_switch_missing_file_defaults_to_hover(tmp_path):
    command = tmp_path / "safety.cmd"
    switch = pff.SafetySwitch(command, keyboard=False)
    assert switch.poll() == "HOVER"
    assert command.read_text().strip().lower() == "hover"


def test_fly_requires_auto_written_after_this_run_started(tmp_path):
    command = tmp_path / "safety.cmd"
    command.write_text("auto\n")
    switch = pff.SafetySwitch(command, keyboard=False, require_fresh_auto=True)
    assert switch.poll() == "HOVER", "stale AUTO from a previous run must not arm"
    baseline = command.stat().st_mtime_ns
    command.write_text("auto\n")
    os.utime(command, ns=(baseline + 1_000_000, baseline + 1_000_000))
    assert switch.poll() == "AUTO"


def test_keyboard_read_failure_fails_closed(monkeypatch):
    switch = pff.SafetySwitch(path=None, keyboard=False)
    switch.keyboard = True
    switch.mode = "AUTO"
    monkeypatch.setattr(pff.select, "select", lambda *_args, **_kw: (_ for _ in ()).throw(
        OSError("stdin failed")))
    assert switch.poll() == "HOVER"


def test_safety_monitor_poll_failure_fails_closed():
    class _BrokenSafety:
        @staticmethod
        def poll():
            raise OSError("injected unreadable safety channel")

    sent = []
    mon = pff.SafetyMonitor(lambda *cmd: sent.append(tuple(cmd)), _BrokenSafety(), timeout_s=0.06)
    mon.start()
    time.sleep(0.2)
    mon.stop()
    assert mon.mode == "HOVER"
    assert ZERO in sent


def test_atomic_auto_sender_rechecks_mode_at_send_time():
    class _ChangedSafety:
        mode = "AUTO"

        @staticmethod
        def poll():
            return "MANUAL"

    sent = []
    mon = pff.SafetyMonitor(lambda *cmd: sent.append(tuple(cmd)), _ChangedSafety(), timeout_s=0.06)
    ok, reason, actual = mon.send_auto((0, 8, 0, 0), lambda: True, lambda: False)
    assert not ok and "MANUAL" in reason and actual is None
    assert sent == []


def test_terminal_expectation_must_wait_and_report_success():
    class _Confirmation:
        def __init__(self, ok):
            self.ok = ok

        def success(self):
            return self.ok

    class _Expectation:
        def __init__(self, ok):
            self.ok = ok
            self.waited = 0

        def wait(self):
            self.waited += 1
            return _Confirmation(self.ok)

    failed = _Expectation(False)
    mon = pff.SafetyMonitor(lambda *_: None, timeout_s=0.1)
    assert not mon._attempt_callback("LAND", lambda: failed, 1.0)
    assert failed.waited == 1 and not mon._land_acted
    assert mon.action_failures["LAND"] == 1

    succeeded = _Expectation(True)
    assert mon._attempt_callback("LAND", lambda: succeeded, 2.0)
    assert succeeded.waited == 1 and mon._land_acted

    with pytest.raises(RuntimeError, match="unconfirmed"):
        pff._await_confirmed_action(object(), "truthy object")


def test_piloting_source_is_confirmed_and_read_back(monkeypatch):
    state_message = object()

    def source_command(*, source):
        return SimpleNamespace(source=source)

    fake_module = SimpleNamespace(
        setPilotingSource=source_command,
        pilotingSource=state_message,
    )
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "olympe.messages.skyctrl.CoPiloting":
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    class _Confirmation:
        @staticmethod
        def success():
            return True

    class _Expectation:
        waited = 0

        def wait(self):
            self.waited += 1
            return _Confirmation()

    class _Drone:
        def __init__(self):
            self.expectation = _Expectation()
            self.requested = None

        def __call__(self, command):
            self.requested = command.source
            return self.expectation

        def get_state(self, message):
            assert message is state_message
            return {"source": self.requested}

    drone = _Drone()
    assert pff.set_piloting_source(drone, "Controller")
    assert drone.requested == "Controller" and drone.expectation.waited == 1


def _fake_firmware_preflight(monkeypatch, *, battery=80, gps_fixed=1,
                             flying_state="landed"):
    tokens = {name: object() for name in (
        "max_altitude", "max_distance", "geofence", "battery", "gps", "flying")}

    def command(kind):
        return lambda **kwargs: SimpleNamespace(kind=kind, kwargs=kwargs)

    modules = {
        "olympe.messages.ardrone3.PilotingSettings": SimpleNamespace(
            MaxAltitude=command("max_altitude"),
            MaxDistance=command("max_distance"),
            NoFlyOverMaxDistance=command("geofence"),
        ),
        "olympe.messages.ardrone3.PilotingSettingsState": SimpleNamespace(
            MaxAltitudeChanged=tokens["max_altitude"],
            MaxDistanceChanged=tokens["max_distance"],
            NoFlyOverMaxDistanceChanged=tokens["geofence"],
        ),
        "olympe.messages.ardrone3.PilotingState": SimpleNamespace(
            FlyingStateChanged=tokens["flying"]),
        "olympe.messages.ardrone3.GPSSettingsState": SimpleNamespace(
            GPSFixStateChanged=tokens["gps"]),
        "olympe.messages.common.CommonState": SimpleNamespace(
            BatteryStateChanged=tokens["battery"]),
    }
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            return modules[name]
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    class _Expectation:
        def __init__(self):
            self.waited = 0

        def wait(self):
            self.waited += 1
            return self

        @staticmethod
        def success():
            return True

    class _Drone:
        def __init__(self):
            self.calls = []
            self.reads = []
            self.states = {
                tokens["flying"]: {"state": flying_state},
                tokens["battery"]: {"percent": battery},
                tokens["gps"]: {"fixed": gps_fixed},
                tokens["max_altitude"]: {"current": 150.0, "min": 0.5, "max": 150.0},
                tokens["max_distance"]: {"current": 100.0, "min": 10.0, "max": 2000.0},
                tokens["geofence"]: {"shouldNotFlyOver": 0},
            }

        def __call__(self, message):
            expectation = _Expectation()
            self.calls.append((message.kind, dict(message.kwargs), expectation))
            if message.kind == "max_altitude":
                self.states[tokens[message.kind]]["current"] = message.kwargs["current"]
            elif message.kind == "max_distance":
                self.states[tokens[message.kind]]["current"] = message.kwargs["value"]
            else:
                self.states[tokens[message.kind]]["shouldNotFlyOver"] = message.kwargs[
                    "shouldNotFlyOver"]
            return expectation

        def get_state(self, message):
            self.reads.append(message)
            return dict(self.states[message])

    return _Drone(), tokens


def test_firmware_preflight_sets_waits_and_reads_back_limits(monkeypatch):
    drone, tokens = _fake_firmware_preflight(monkeypatch, battery=30, gps_fixed=1)
    result = pff.configure_flight_preflight(
        drone, max_altitude_m=20.0, max_distance_m=80.0, distance_geofence=True)

    assert result == {
        "battery_percent": 30,
        "max_altitude_m": 20.0,
        "max_distance_m": 80.0,
        "distance_geofence": True,
    }
    assert [(kind, args) for kind, args, _ in drone.calls] == [
        ("max_altitude", {"current": 20.0}),
        ("max_distance", {"value": 80.0}),
        ("geofence", {"shouldNotFlyOver": 1}),
    ]
    assert all(expectation.waited == 1 for _, _, expectation in drone.calls)
    assert drone.reads[0] is tokens["flying"]
    assert tokens["gps"] in drone.reads


def test_firmware_preflight_rejects_low_battery_before_commands(monkeypatch):
    drone, _ = _fake_firmware_preflight(monkeypatch, battery=29, gps_fixed=1)
    with pytest.raises(RuntimeError, match="below the 30%"):
        pff.configure_flight_preflight(drone, 20.0, 80.0, True)
    assert drone.calls == []


def test_firmware_preflight_rejects_non_landed_before_any_write(monkeypatch):
    drone, tokens = _fake_firmware_preflight(
        monkeypatch, battery=80, gps_fixed=1, flying_state="hovering")
    with pytest.raises(RuntimeError, match="only be changed while landed"):
        pff.configure_flight_preflight(drone, 20.0, 80.0, True)
    assert drone.reads == [tokens["flying"]]
    assert drone.calls == []


def test_firmware_distance_geofence_requires_gps_fix(monkeypatch):
    drone, _ = _fake_firmware_preflight(monkeypatch, battery=80, gps_fixed=0)
    with pytest.raises(RuntimeError, match="GPS fix"):
        pff.configure_flight_preflight(drone, 20.0, 80.0, True)
    assert drone.calls == []


def test_disabled_distance_geofence_does_not_require_gps(monkeypatch):
    drone, tokens = _fake_firmware_preflight(monkeypatch, battery=80, gps_fixed=0)
    result = pff.configure_flight_preflight(drone, 20.0, 80.0, False)
    assert result["distance_geofence"] is False
    assert tokens["gps"] not in drone.reads
    assert drone.calls[-1][1] == {"shouldNotFlyOver": 0}


def test_manual_handoff_and_auto_reacquire_are_atomic_with_pcmd():
    class _Safety:
        mode = "MANUAL"

        def poll(self):
            return self.mode

    safety = _Safety()
    sources, sent = [], []
    mon = pff.SafetyMonitor(
        lambda *cmd: sent.append(tuple(cmd)),
        safety,
        timeout_s=0.1,
        piloting_source_cb=lambda source: (sources.append(source) or True),
    )

    ok, reason, actual = mon.send_authorized(
        (0, 8, 0, 0), lambda: True, lambda: False)
    assert not ok and "MANUAL" in reason and actual is None
    assert sources == ["SkyController"] and sent == []

    safety.mode = "AUTO"
    ok, _, actual = mon.send_authorized(
        (0, 8, 0, 0), lambda: True, lambda: False)
    assert ok and actual == (0, 8, 0, 0)
    assert sources == ["SkyController", "Controller"]
    assert sent == [(0, 8, 0, 0)]


def test_fly_confirms_controller_before_pcmd_takeoff_and_waits_for_landed():
    src = inspect.getsource(pff.fly)
    initial_sticks_i = src.index('set_piloting_source(drone, "SkyController")')
    preflight_i = src.index("configure_flight_preflight")
    models_i = src.index("loc.ensure_models()")
    controller_i = src.index('set_piloting_source(drone, "Controller")')
    takeoff_i = src.index("TakeOff() >>")
    assert initial_sticks_i < preflight_i < models_i < controller_i
    assert controller_i < src.index("def send_pcmd") < src.index("SafetyMonitor(") < takeoff_i
    restore_i = src.rindex('set_piloting_source(drone, "SkyController")')
    assert restore_i < src.rindex("drone.disconnect()")
    assert "--fly requires explicit --max-altitude-m and --max-distance-m" in src
    assert src.count('state="landed"') >= 2


def test_dry_run_does_not_apply_yaw_sign_twice():
    src = inspect.getsource(pff.dry_run)
    assert "KY * yaw * dt" in src
    assert "yaw_sign * yaw * dt" not in src


def test_takeoff_schedule_is_atomic_with_land_refresh_and_does_not_wait():
    class _Safety:
        mode = "AUTO"

        def poll(self):
            return self.mode

    class _Expectation:
        waited = 0

        def wait(self):
            self.waited += 1
            return self

        @staticmethod
        def success():
            return True

    safety = _Safety()
    mon = pff.SafetyMonitor(lambda *_: None, safety, timeout_s=0.1)
    scheduled = threading.Event()
    release_schedule = threading.Event()
    land_finished = threading.Event()
    expectation = _Expectation()
    takeoff_result = {}
    land_result = {}

    def schedule():
        scheduled.set()
        assert release_schedule.wait(1.0)
        return expectation

    def takeoff_thread():
        takeoff_result["value"] = mon.schedule_authorized_takeoff(
            schedule, lambda: True, lambda: False)

    takeoff = threading.Thread(target=takeoff_thread)
    takeoff.start()
    assert scheduled.wait(1.0)
    safety.mode = "LAND"

    def land_thread():
        land_result["value"] = mon.arming_allowed(lambda: True, lambda: False)
        land_finished.set()

    land = threading.Thread(target=land_thread)
    land.start()
    assert not land_finished.wait(0.05), "LAND refresh entered before TakeOff schedule released its lock"
    release_schedule.set()
    takeoff.join(1.0)
    land.join(1.0)

    assert takeoff_result["value"][0] is expectation
    assert expectation.waited == 0, "TakeOff expectation wait must happen outside _io_lock"
    assert land_result["value"][0] is False
    assert mon.terminal_action == "LAND"

    # The opposite lock order is also deterministic: a latched LAND blocks the
    # schedule callback entirely.
    called = []
    blocked, reason = mon.schedule_authorized_takeoff(
        lambda: called.append(True), lambda: True, lambda: False)
    assert blocked is None and "terminated" in reason and called == []


def test_hover_to_manual_race_never_sends_zero_outside_authority():
    mode = {"value": "HOVER"}
    raw, authorized = [], []

    def authority(pcmd):
        authorized.append(tuple(pcmd))
        mode["value"] = "MANUAL"
        return False, "safety mode is MANUAL", None

    hooks = pff.LoopHooks(
        get_pose=lambda: (_ for _ in ()).throw(AssertionError("HOVER must gate localization")),
        olympe_yaw=lambda: None,
        send_pcmd=lambda *cmd: raw.append(tuple(cmd)),
        send_authorized_pcmd=authority,
        safety_poll=lambda: mode["value"],
        now=lambda: (_ for _ in ()).throw(KeyboardInterrupt()) if mode["value"] == "MANUAL" else 0.0,
    )
    with pytest.raises(KeyboardInterrupt):
        pff.run_loop(hooks, None, None, verbose=False)
    assert authorized == [ZERO]
    assert raw == [], "MANUAL must silence every PCMD, including a cached HOVER zero"


def test_land_failure_can_still_escalate_to_emergency():
    class _Safety:
        mode = "AUTO"

        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return "LAND" if self.calls == 1 else "EMERGENCY"

    calls = {"land": 0, "emergency": 0}

    def fail_land():
        calls["land"] += 1
        raise RuntimeError("injected Landing failure")

    mon = pff.SafetyMonitor(
        lambda *_: None,
        _Safety(),
        land_cb=fail_land,
        emergency_cb=lambda: calls.__setitem__("emergency", calls["emergency"] + 1),
        timeout_s=0.06,
    ).start()
    time.sleep(0.25)
    mon.stop()
    assert calls == {"land": 1, "emergency": 1}
    assert mon.reason.startswith("EMERGENCY") and mon.emergency_issued


def test_land_callback_retries_exception_until_none_success_without_busy_loop():
    class _Safety:
        mode = "LAND"

        @staticmethod
        def poll():
            return "LAND"

    attempts = []

    def land():
        attempts.append(time.monotonic())
        if len(attempts) < 3:
            raise RuntimeError("injected Landing failure")
        return None  # Existing Olympe send callbacks use None as success.

    mon = pff.SafetyMonitor(
        lambda *_: None, _Safety(), land_cb=land, timeout_s=0.06).start()
    time.sleep(0.36)
    mon.stop()
    assert len(attempts) == 3 and mon._land_acted
    assert mon.action_failures["LAND"] == 2
    assert min(b - a for a, b in zip(attempts, attempts[1:])) >= 0.08


def test_emergency_false_result_retries_and_remains_highest_priority():
    class _Safety:
        mode = "EMERGENCY"

        @staticmethod
        def poll():
            return "EMERGENCY"

    attempts = []

    def emergency():
        attempts.append(time.monotonic())
        return False if len(attempts) < 3 else None

    sent = []
    mon = pff.SafetyMonitor(
        lambda *cmd: sent.append(tuple(cmd)), _Safety(),
        land_cb=lambda: (_ for _ in ()).throw(AssertionError("LAND cannot override EMERGENCY")),
        emergency_cb=emergency, stop_requested=lambda: True,
        timeout_s=0.06,
    ).start()
    time.sleep(0.36)
    mon.stop()
    assert len(attempts) == 3 and mon._emergency_acted and mon.emergency_issued
    assert mon.action_failures["EMERGENCY"] == 2
    assert sent == [], "EMERGENCY must remain PCMD-silent even while retrying"


@pytest.mark.parametrize("terminal", ["LAND", "EMERGENCY"])
def test_one_shot_terminal_command_latches_until_callback_succeeds(terminal):
    class _Safety:
        mode = "AUTO"

        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return terminal if self.calls == 1 else "AUTO"

    attempts = []

    def action():
        attempts.append(time.monotonic())
        if len(attempts) < 3:
            raise RuntimeError(f"injected one-shot {terminal} failure")
        return None

    kwargs = {"land_cb": action} if terminal == "LAND" else {"emergency_cb": action}
    sent = []
    mon = pff.SafetyMonitor(
        lambda *cmd: sent.append(tuple(cmd)), _Safety(), timeout_s=0.06,
        **kwargs,
    ).start()
    time.sleep(0.36)
    mon.stop()
    assert len(attempts) == 3
    assert mon.terminal_action == terminal and mon.terminated.is_set()
    assert mon._land_acted is (terminal == "LAND")
    assert mon._emergency_acted is (terminal == "EMERGENCY")
    if terminal == "EMERGENCY":
        assert sent == [], "latched EMERGENCY stays PCMD-silent after input returns AUTO"


@pytest.mark.parametrize("terminal", ["LAND", "EMERGENCY"])
def test_terminal_consumed_by_authorized_send_is_still_latched(terminal):
    class _Safety:
        mode = "AUTO"

        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return terminal if self.calls == 1 else "AUTO"

    attempts = []

    def action():
        attempts.append(True)
        return None

    kwargs = {"land_cb": action} if terminal == "LAND" else {"emergency_cb": action}
    mon = pff.SafetyMonitor(lambda *_: None, _Safety(), timeout_s=0.06, **kwargs)
    sent, _reason, _actual = mon.send_authorized(
        (0, 8, 0, 0), lambda: True, lambda: False)
    assert not sent and mon.terminal_action == terminal
    mon.start()
    time.sleep(0.12)
    mon.stop()
    assert attempts == [True]


def test_blocking_inspection_sustains_zero_under_authority():
    class _Safety:
        mode = "AUTO"

        @staticmethod
        def poll():
            return "AUTO"

    sent = []
    mon = pff.SafetyMonitor(
        lambda *cmd: sent.append(tuple(cmd)), _Safety(), timeout_s=0.06).start()
    try:
        assert mon.run_while_holding_zero(
            lambda: (time.sleep(0.16) or True), lambda: True, lambda: False)
    finally:
        mon.stop()
    assert len(sent) >= 3 and all(cmd == ZERO for cmd in sent)


def test_termination_signal_lands_even_when_control_loop_is_blocked():
    sent, landed = [], []
    mon = pff.SafetyMonitor(
        lambda *cmd: sent.append(tuple(cmd)),
        land_cb=lambda: landed.append(True),
        stop_requested=lambda: True,
        timeout_s=0.06,
    ).start()
    time.sleep(0.2)
    mon.stop()
    assert mon.terminated.is_set() and landed == [True]
    assert ZERO in sent and "termination signal" in mon.reason


def test_non_finite_safety_environment_value_rejected(monkeypatch):
    monkeypatch.setenv("SFM_TEST_SAFETY", "nan")
    with pytest.raises(ValueError):
        pff._env_float("SFM_TEST_SAFETY", 1.0, minimum=0.1, maximum=10.0)


def test_takeoff_has_final_arming_gate_cleanup_and_termination_signals():
    src = inspect.getsource(pff.fly)
    assert src.index("must_land = True") < src.index("TakeOff() >>")
    assert "schedule_authorized_takeoff" in src
    assert "arming_allowed" in src
    assert "require_fresh_auto=True" in src
    assert src.rindex("monitor.stop()") < src.rindex("reason = monitor.reason")
    assert src.rindex("reason = monitor.reason") < src.rindex("if must_land")
    assert src.rindex("monitor.send_authorized") > src.index("finally:")
    assert "emergency_issued" in src
    assert "require_source_timestamps=True" in src
    assert src.index("require_confirmation=True") < src.index("baseline_source_us =")
    assert "inspection_sample_after" in src and "INSPECTION_PIPELINE_DRAIN_S" in src
    for sig in ("SIGINT", "SIGTERM", "SIGHUP"):
        assert sig in src


@pytest.mark.parametrize("mode, stream_ok, stop, terminated", [
    ("HOVER", True, False, False),
    ("MANUAL", True, False, False),
    ("LAND", True, False, False),
    ("AUTO", False, False, False),
    ("AUTO", True, True, False),
    ("AUTO", True, False, True),
])
def test_arming_gate_rejects_any_unsafe_condition(mode, stream_ok, stop, terminated):
    ok, _reason = pff.arming_allowed(mode, stream_ok, stop, terminated)
    assert not ok


def test_arming_gate_accepts_only_clean_auto_state():
    assert pff.arming_allowed("AUTO", True, False, False)[0]


def test_low_confidence_uses_weak_timeout_not_lost_timeout(monkeypatch):
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.15)
    monkeypatch.setattr(pff, "WEAK_HOVER_LAND_S", 0.55)
    sent, reason, records = run_ticks(
        100, fresh_pose, hooks_extra={"pose_is_weak": lambda: True})
    assert reason == "low confidence -> land"
    waits = [float(r["reason"].split("(")[-1].split("s")[0])
             for r in records if "low confidence -> hover" in r["reason"]]
    assert waits and max(waits) >= 0.5


def test_alternating_weak_and_lost_share_one_uncertainty_deadline(monkeypatch):
    monkeypatch.setattr(pff, "WEAK_HOVER_LAND_S", 0.8)
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.4)
    monkeypatch.setattr(pff, "LOST_MANUAL_S", 100.0)
    mode = {"weak": False}

    def alternating(st):
        mode["weak"] = bool(st["tick"] % 2)
        x = 0.0 if mode["weak"] else 5.0  # alternate WEAK with jump-rejected LOST
        return fresh_pose(st, x=x)

    sent, reason, records = run_ticks(
        100, alternating, hooks_extra={"pose_is_weak": lambda: mode["weak"]})
    assert reason.endswith("-> land") and reason != "tick cap"
    assert sent and all(cmd == ZERO for cmd in sent)
    assert any("low confidence" in r["reason"] for r in records)
    assert any("pose jump rejected" in r["reason"] for r in records)


def test_watchdog_stall_after_beat_sends_zero():
    calls = []
    mon = pff.SafetyMonitor(lambda r, p, y, g: calls.append((r, p, y, g)),
                            None, timeout_s=0.06)
    mon.beat()                     # loop started, then stalls (no more beats)
    mon.start()
    time.sleep(0.3)
    mon.stop()
    assert ZERO in calls, "a stalled loop must be overridden with zero PCMD"


def test_control_thread_repeats_latest_desired_pcmd_during_inference():
    calls = []
    mon = pff.SafetyMonitor(
        lambda *cmd: calls.append((time.monotonic(), tuple(cmd))),
        None,
        timeout_s=0.4,
    ).start()
    try:
        mon.beat()
        ok, _reason, actual = mon.send_authorized(
            (0, 8, 0, 0), lambda: True, lambda: False)
        assert ok and actual == (0, 8, 0, 0)
        time.sleep(0.18)  # simulates a blocked perception/inference call
    finally:
        mon.stop()
    nonzero = [(stamp, cmd) for stamp, cmd in calls if cmd != ZERO]
    assert len(nonzero) >= 3
    assert all(cmd == (0, 8, 0, 0) for _, cmd in nonzero)
    assert mon.last_pcmd_call_mono_ns is not None


def test_hover_clears_desired_command_before_auto_can_resume():
    class _Safety:
        mode = "AUTO"

        def poll(self):
            return self.mode

    safety = _Safety()
    calls = []
    mon = pff.SafetyMonitor(
        lambda *cmd: calls.append(tuple(cmd)), safety, timeout_s=0.5).start()
    try:
        mon.beat()
        assert mon.send_authorized(
            (0, 8, 0, 0), lambda: True, lambda: False)[0]
        time.sleep(0.08)
        safety.mode = "HOVER"
        time.sleep(0.08)
        first_zero = len(calls)
        safety.mode = "AUTO"
        time.sleep(0.10)
    finally:
        mon.stop()
    assert ZERO in calls[:first_zero + 1]
    last_zero = max(i for i, cmd in enumerate(calls) if cmd == ZERO)
    assert all(cmd == ZERO for cmd in calls[last_zero:])


def test_no_nonzero_command_reuse_after_loss():
    # The loop may drive on last_good within the POSE_STALE_S freshness window,
    # but once it first hovers after the loss, no nonzero command may ever recur.
    def pose_fn(st):
        return fresh_pose(st) if st["tick"] <= 5 else None
    sent, reason, _ = run_ticks(200, pose_fn)
    assert any(c != ZERO for c in sent[:6]), "sanity: loop was actually driving first"
    first_zero_after_loss = next(i for i in range(6, len(sent)) if sent[i] == ZERO)
    assert all(c == ZERO for c in sent[first_zero_after_loss:]), \
        "after the loss is recognized, the previous nonzero command must never be reused"
    assert reason == "localization lost -> land"


# ---------------------------------------------------------------------------
# PCMD conversion

def test_pcmd_conversion_bounds():
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.3)
    for vx in (-100.0, -1.2, 0.0, 1.2, 100.0):
        for vy in (-50.0, 0.0, 50.0):
            for yt in (-3.0, 0.0, 3.0):
                cmd = rpf.Command("FOLLOW", np.array([vx, vy, 0.7]), yaw_target=yt,
                                  goal=np.zeros(3), path_error=0.0, progress=0.0)
                r, p, y, g = rpf.command_to_body_percent(cmd, pose)
                assert all(isinstance(v, int) for v in (r, p, y, g))
                assert r == 0 and 0 <= p <= 8 and abs(y) <= 25 and abs(g) <= 12
                assert all(-100 <= v <= 100 for v in (r, p, y, g))


@pytest.mark.parametrize("field", ["velocity", "shape", "yaw", "pose"])
def test_non_finite_command_fails_closed_to_zero(field):
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0)
    cmd = rpf.Command("FOLLOW", np.array([1.0, 0.0, 0.0]), yaw_target=0.0,
                      goal=np.zeros(3), path_error=0.0, progress=0.0)
    if field == "velocity":
        cmd.vel_map[0] = np.nan
    elif field == "shape":
        cmd.vel_map = np.zeros(2)
    elif field == "yaw":
        cmd.yaw_target = np.inf
    else:
        pose.x = np.nan
    assert rpf.command_to_body_percent(cmd, pose) == ZERO


def test_lost_reacquire_clears_motion_and_flow_history():
    import production_xfeat_tracker as pxt

    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.state = pxt.RuntimeState(
        mode="LOST", last_pose=pxt.Pose(8, 0, 0, 0, 1), prev_pose=pxt.Pose(7, 0, 0, 0, 0),
        last_center=np.array([8, 0, 0], np.float32), prev_center=np.array([7, 0, 0], np.float32),
        last_yaw=0.2, prev_yaw=0.1, last_refs=[1, 2])
    tracker.temporal_cache = pxt.TemporalAnchorCache(
        descriptors=np.ones((2, 2)), xyz=np.ones((2, 3)), ref_ids=np.array([1, 2]))
    tracker._flow_2d = np.ones((2, 2)); tracker._flow_3d = np.ones((2, 3))
    tracker._flow_gray = np.ones((2, 2)); tracker._flow_age = 3
    tracker._last_inl_2d = np.ones((2, 2)); tracker._last_inl_3d = np.ones((2, 3))
    tracker._last_seed_refs = (1, 2)
    pose = pxt.Pose(20, 0, 0, 0.3, 2)
    tracker._publish_success(pose, {"used_refs": [3]}, False, source_mode="LOST")
    assert tracker.state.prev_center is None and tracker.state.prev_pose is None
    assert tracker.state.last_center.tolist() == [20.0, 0.0, 0.0]
    assert tracker._flow_2d is None and tracker._flow_3d is None and tracker._flow_gray is None
    assert len(tracker.temporal_cache) == 0 and tracker._last_seed_refs == ()


def test_flow_failure_retries_same_frame_with_deep_path():
    import production_xfeat_tracker as pxt

    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker._flow_2d = np.ones((2, 2)); tracker._flow_3d = np.ones((2, 3))
    tracker._flow_gray = np.ones((2, 2)); tracker._flow_age = 3
    expected = pxt.Pose(1, 2, 3, 0.2, 4)

    def deep(_frame):
        tracker._last_info = {"accepted": True, "inliers": 120}
        return expected

    tracker._localize_frame_deep = deep
    tracker._seed_flow_from_deep = lambda *_args: True
    actual = tracker._flow_fallback_deep(
        np.zeros((2, 2, 3), np.uint8), "track_lowconf", inliers=42)
    assert actual is expected
    assert tracker._flow_2d is None and tracker._flow_3d is None
    assert tracker._flow_gray is None and tracker._flow_age == 0
    assert tracker._last_info["flow_stage"] == "track_lowconf"
    assert tracker._last_info["flow_fallback"] is True
    assert tracker._last_info["flow_inliers"] == 42


def test_flow_cache_is_not_used_outside_track_state():
    import production_xfeat_tracker as pxt

    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.state = pxt.RuntimeState(mode="LOST")
    tracker._flow_2d = np.ones((2, 2)); tracker._flow_3d = np.ones((2, 3))
    tracker._flow_gray = np.ones((2, 2), np.uint8); tracker._flow_age = 1
    tracker._flow_refresh_every = 6; tracker._flow_min_seed = 100
    tracker._last_inl_2d = None; tracker._last_inl_3d = None
    deep_calls = []

    def deep(_frame):
        deep_calls.append(True)
        tracker._last_info = {"accepted": False, "inliers": 0}
        return None

    tracker._localize_frame_deep = deep
    assert tracker._localize_frame_flow(np.zeros((2, 2, 3), np.uint8)) is None
    assert deep_calls == [True]
    assert tracker._flow_2d is None and tracker._flow_3d is None
    assert tracker._last_info["flow_stage"] == "refresh"


def test_megaloc_fp16_is_explicit_and_cuda_only(monkeypatch):
    import production_xfeat_tracker as pxt

    descriptors = np.ones((1, 2), np.float32)
    monkeypatch.setenv("SFM_MEGALOC_FP16", "1")
    assert pxt.MegaLocLayer(descriptors, device="cuda").fp16 is True
    assert pxt.MegaLocLayer(descriptors, device="cpu").fp16 is False
    monkeypatch.setenv("SFM_MEGALOC_FP16", "0")
    assert pxt.MegaLocLayer(descriptors, device="cuda").fp16 is False


def test_cuda_oom_releases_live_gpu_caches(monkeypatch):
    import production_xfeat_tracker as pxt

    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.state = pxt.RuntimeState(mode="TRACK")
    tracker._ref_dev_cache = OrderedDict([(1, {"descriptors": object()})])
    tracker.temporal_cache = pxt.TemporalAnchorCache(
        descriptors=np.ones((2, 2)), xyz=np.ones((2, 3)), ref_ids=np.array([1, 2]))
    tracker._flow_2d = np.ones((2, 2)); tracker._flow_3d = np.ones((2, 3))
    tracker._flow_gray = np.ones((2, 2)); tracker._flow_age = 1
    tracker._last_inl_2d = np.ones((2, 2)); tracker._last_inl_3d = np.ones((2, 3))
    tracker._last_seed_refs = (1,)
    tracker._localize_frame_impl = lambda _frame: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory"))
    cleanup = []
    monkeypatch.setattr(pxt.gc, "collect", lambda: cleanup.append(("gc", sys.exc_info()[0])))
    monkeypatch.setattr(
        pxt.torch.cuda, "empty_cache",
        lambda: cleanup.append(("empty", sys.exc_info()[0])))
    assert tracker._localize_frame_deep(np.zeros((2, 2, 3), np.uint8)) is None
    assert not tracker._ref_dev_cache and len(tracker.temporal_cache) == 0
    assert tracker._flow_2d is None and tracker._last_seed_refs == ()
    assert cleanup == [("gc", None), ("empty", None)], \
        "traceback refs must be gone before gc.collect()/empty_cache()"


def test_command_log_has_final_pcmd_and_reason():
    sent, _, records = run_ticks(10, fresh_pose)
    driven = [r for r in records if not r["blocked"]]
    assert driven, "driving ticks must be logged"
    for r in driven:
        assert r["pcmd"] is not None and len(r["pcmd"]) == 4
        assert "path_error_u" in r and "progress" in r and "action" in r
        assert tuple(r["pcmd"]) in sent, "logged PCMD must be exactly what was sent"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
