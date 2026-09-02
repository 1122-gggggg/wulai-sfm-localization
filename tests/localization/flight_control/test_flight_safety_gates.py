#!/usr/bin/env python3
"""Failure-injection tests for the real-flight safety gates (pure python, no drone).

Every test drives the REAL path_follow_flight.run_loop / SafetyMonitor /
command_to_body_percent / OlympePdrawGrabber logic with mock hooks. No olympe,
no torch, no GPU, no hardware.

Run:  pytest -q tests/localization/flight_control/test_flight_safety_gates.py
"""
from __future__ import annotations

import builtins
import ast
import hashlib
import inspect
import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FLIGHT_CONTROL_ROOT = REPO_ROOT / "定位演算法" / "flight_control"
sys.path.insert(0, str(FLIGHT_CONTROL_ROOT))

import olympe_frame_source as ofs
import path_follow_flight as pff
import real_path_follow_controller as rpf

ZERO = (0, 0, 0, 0)


def test_sparse_cloud_collision_monitor_uses_adjustable_3d_radius() -> None:
    monitor = rpf.SparseCloudCollisionMonitor(
        np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
        collision_radius=0.025,
        warning_radius=0.025,
    )
    if monitor.tree is None:
        pytest.skip("scipy cKDTree is unavailable")

    clear = monitor.update(np.array([0.04, 0.0, 0.0]))
    assert clear["status"] == "CLEAR"
    assert clear["distance"] == pytest.approx(0.04)

    monitor.set_radii(0.05)
    collision = monitor.update(np.array([0.04, 0.0, 0.0]))
    assert collision["status"] == "COLLISION"
    assert collision["point"] == pytest.approx([0.0, 0.0, 0.0])


def test_sparse_cloud_collision_monitor_reports_unavailable_not_off_when_scipy_missing(
    monkeypatch,
) -> None:
    """A missing cKDTree must not read like 'no obstacle nearby' (fail-open)."""
    monkeypatch.setattr(rpf, "cKDTree", None)
    monitor = rpf.SparseCloudCollisionMonitor(
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        collision_radius=0.025,
        warning_radius=0.025,
    )
    assert monitor.tree is None
    result = monitor.update(np.array([0.0, 0.0, 0.0]))
    assert result["status"] == "UNAVAILABLE"
    assert result["severity"] == 0.0


def test_sparse_cloud_collision_monitor_reports_off_for_an_empty_cloud() -> None:
    """An empty point cloud is a different situation from scipy being absent."""
    monitor = rpf.SparseCloudCollisionMonitor(
        np.zeros((0, 3), dtype=float), collision_radius=0.025, warning_radius=0.025
    )
    assert monitor.tree is None
    result = monitor.update(np.array([0.0, 0.0, 0.0]))
    assert result["status"] == "OFF"


def test_sparse_cloud_collision_monitor_rejects_invalid_radii_and_points() -> None:
    zero_radius = rpf.SparseCloudCollisionMonitor(
        np.zeros((1, 3)), collision_radius=0.0, warning_radius=0.0
    )
    if zero_radius.tree is not None:
        assert zero_radius.update(np.zeros(3))["status"] == "COLLISION"
        assert zero_radius.update(np.array([0.001, 0.0, 0.0]))["status"] == "CLEAR"
    with pytest.raises(ValueError, match="within"):
        rpf.SparseCloudCollisionMonitor(np.zeros((1, 3)), collision_radius=-0.001)
    with pytest.raises(ValueError, match="within"):
        rpf.SparseCloudCollisionMonitor(np.zeros((1, 3)), collision_radius=0.061)
    with pytest.raises(ValueError, match="Nx3"):
        rpf.SparseCloudCollisionMonitor(np.zeros((3,)))

    monitor = rpf.SparseCloudCollisionMonitor(
        np.array([[np.nan, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        collision_radius=0.02,
        warning_radius=0.02,
    )
    assert monitor.xyz.tolist() == [[1.0, 2.0, 3.0]]
    with pytest.raises(ValueError, match="warning_radius"):
        monitor.set_radii(0.02, 0.01)


def run_ticks(max_ticks, pose_fn, *, route=((0.0, 0.0, 0.0), (8.0, 0.0, 0.0)),
              hooks_extra=None, enforce_weak_pose_gate=None, control_config=None):
    """Run the real run_loop against mock hooks on a virtual clock.

    pose_fn(st) -> rpf.Pose | None; st has "t" (virtual now) and "tick".
    Returns (sent_pcmds, terminal_reason, log_records).
    """
    wp = [np.array(p, float) for p in route]
    config = control_config or rpf.ControlConfig(inspect_waypoints=())
    ctrl = rpf.RouteAutoController(wp, poles=[], config=config)
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
    extra.setdefault("ground_speed", lambda: (0.0, st["t"]))
    olympe_yaw = extra.pop("olympe_yaw", lambda: None)
    hooks = pff.LoopHooks(
        get_pose=get_pose,
        olympe_yaw=olympe_yaw,
        send_pcmd=lambda r, p, y, g: sent.append((r, p, y, g)),
        now=now,
        **extra,
    )
    run_kwargs = {}
    if enforce_weak_pose_gate is not None:
        run_kwargs["enforce_weak_pose_gate"] = enforce_weak_pose_gate
    try:
        reason = pff.run_loop(hooks, ctrl, wp, verbose=False, **run_kwargs)
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


def test_fly_is_not_an_arming_entrypoint():
    fly_src = inspect.getsource(pff.fly)
    assert "TakeOff()" not in fly_src
    assert "operator UI" in fly_src
    tree = ast.parse(inspect.getsource(pff))
    takeoffs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TakeOff"
    ]
    assert takeoffs == []


def test_flight_localizer_uses_site_selected_edm_factory(monkeypatch):
    captured = {}
    tracker = object()

    def fake_build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            tracker=tracker,
            backend="edm",
            variant="fixture",
            camera=SimpleNamespace(model="PINHOLE", width=1280, height=720),
        )

    monkeypatch.setitem(
        sys.modules,
        "production_localizer_factory",
        SimpleNamespace(build_production_localizer=fake_build),
    )
    monkeypatch.setattr(pff, "LOCALIZER_BACKEND", "edm")
    monkeypatch.setattr(pff, "LOCALIZER_PROFILE", "/tmp/edm-profile.json")
    monkeypatch.setattr(pff, "BUNDLE_SHA256", "a" * 64)
    measured_frame = object()
    monkeypatch.setattr(pff, "MAP_ALIGN", "/tmp/measured-align.json")
    monkeypatch.setattr(rpf, "load_map_frame", lambda path: (
        measured_frame if path == "/tmp/measured-align.json" else None
    ))
    monkeypatch.setattr(pff, "FLIGHT_CONTRACT_JSON", json.dumps({
        "localizer_profile_sha256": "b" * 64,
    }))
    monkeypatch.setenv("SFM_QUERY_CAMERA_JSON", json.dumps({
        "model": "PINHOLE",
        "width": 1280,
        "height": 720,
        "params": [900.0, 900.0, 640.0, 360.0],
    }))

    assert pff.build_localizer(lambda: None) is tracker
    assert captured["backend"] == "edm"
    assert captured["camera_tuple"][0] == "PINHOLE"
    assert captured["bundle_sha256"] == "a" * 64
    assert captured["production_profile_sha256"] == "b" * 64
    assert captured["map_frame"] is measured_frame


def test_approved_route_builds_the_global_scale_free_controller(tmp_path, monkeypatch):
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "schema": "sfm-flight-route/v1",
                "site_id": "river",
                "coordinate_frame_id": "river-frame",
                "frame": "glomap",
                "units": "map",
                "purpose": "flight",
                "closed": False,
                "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "arrive_radius_map_units": 0.02,
            }
        ),
        encoding="utf-8",
    )
    alignment = tmp_path / "T_align_gravity.json"
    alignment.write_text(
        json.dumps(
            {
                "schema": "sfm-align/v2",
                "gravity_glomap": [0.0, 1.0, 0.0],
                "R": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pff, "PATH_JSON", str(route))
    monkeypatch.setattr(pff, "MAP_ALIGN", str(alignment))
    monkeypatch.setattr(
        pff,
        "FLIGHT_CONTRACT_JSON",
        json.dumps(
            {
                "schema_version": 2,
                "approved": True,
                "route_clearance_approved": True,
                "site_id": "river",
                "coordinate_frame_id": "river-frame",
                "route_sha256": hashlib.sha256(route.read_bytes()).hexdigest(),
            }
        ),
    )

    controller, waypoints = pff.build_controller()

    assert len(waypoints) == 2
    assert controller.cfg.inspect_waypoints == ()
    assert controller.cfg.waypoint_arrive_radius == pytest.approx(0.02)
    assert controller.cfg.waypoint_arrive_confirm_frames == 1


# ---------------------------------------------------------------------------
# Frame / stream gates

def test_stream_lost_continues_hovering_past_previous_land_threshold():
    def never_localize(st):
        raise AssertionError("stream gate must run before localization")
    sent, reason, records = run_ticks(
        400, never_localize, hooks_extra={"stream_healthy": lambda: False})
    assert sent and all(c == ZERO for c in sent)
    assert reason == "tick cap"
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


def test_enabled_lost_pose_search_rotates_right_once_before_land(monkeypatch):
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_DELAY_S", 0.0)
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_TIMEOUT_S", 10.0)
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.1)
    telemetry = {"yaw": 0.0}
    events = []

    def lost_pose(_st):
        telemetry["yaw"] = (telemetry["yaw"] + math.radians(45.0)) % (2.0 * math.pi)
        return None

    sent, reason, records = run_ticks(
        100,
        lost_pose,
        hooks_extra={
            "olympe_yaw": lambda: telemetry["yaw"],
            "localization_yaw_search_pcmd": 8,
            "localization_yaw_search_event": (
                lambda state, progress: events.append((state, progress))
            ),
            "request_manual": lambda: pytest.fail(
                "localization timeout must not request automatic manual takeover"
            ),
        },
    )

    yaw_commands = [command for command in sent if command != ZERO]
    assert yaw_commands
    assert all(command == (0, 0, 8, 0) for command in yaw_commands)
    assert reason == "localization lost -> land"
    assert [state for state, _progress in events] == ["started", "completed"]
    assert events[-1][1] >= 360.0
    last_yaw = max(index for index, command in enumerate(sent) if command != ZERO)
    assert all(command == ZERO for command in sent[last_yaw + 1:])
    assert any("right yaw localization search" in record["reason"] for record in records)


def test_enabled_lost_pose_search_waits_ten_seconds_before_rotating(monkeypatch):
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.1)
    manual_requests = []
    events = []

    sent, reason, records = run_ticks(
        180,
        lambda _st: None,
        hooks_extra={
            "olympe_yaw": lambda: 0.0,
            "localization_yaw_search_pcmd": 8,
            "localization_yaw_search_event": (
                lambda state, progress: events.append((state, progress))
            ),
            "request_manual": lambda: manual_requests.append(True) or True,
        },
    )

    assert pff.LOST_YAW_SEARCH_DELAY_S == 10.0
    assert reason == "tick cap"
    first_yaw = next(index for index, command in enumerate(sent) if command != ZERO)
    assert all(command == ZERO for command in sent[:first_yaw])
    assert sent[first_yaw] == (0, 0, 8, 0)
    assert any("hover (9." in record["reason"] for record in records)
    assert not manual_requests
    assert [state for state, _progress in events] == ["started"]


def test_enabled_lost_pose_search_stops_on_recovered_pose(monkeypatch):
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_DELAY_S", 0.0)
    monkeypatch.setattr(pff, "LOST_LAND_S", 5.0)
    telemetry = {"yaw": 0.0}
    events = []

    def pose_recovers(st):
        if st["tick"] <= 6:
            telemetry["yaw"] = (
                telemetry["yaw"] + math.radians(20.0)
            ) % (2.0 * math.pi)
            return None
        return fresh_pose(st, yaw=telemetry["yaw"])

    sent, reason, records = run_ticks(
        30,
        pose_recovers,
        hooks_extra={
            "olympe_yaw": lambda: telemetry["yaw"],
            "localization_yaw_search_pcmd": 8,
            "localization_yaw_search_event": (
                lambda state, progress: events.append((state, progress))
            ),
        },
    )

    assert (0, 0, 8, 0) in sent
    assert reason == "tick cap"
    assert [state for state, _progress in events] == ["started", "recovered"]
    assert any(
        "localization recovery confirmation" in record.get("reason", "")
        and record["pcmd"] == [0, 0, 0, 0]
        for record in records
    )


def test_lost_pose_search_never_rotates_without_a_healthy_stream(monkeypatch):
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_DELAY_S", 0.0)
    sent, reason, _records = run_ticks(
        30,
        lambda _st: None,
        hooks_extra={
            "olympe_yaw": lambda: 0.0,
            "localization_yaw_search_pcmd": 8,
            "stream_healthy": lambda: False,
        },
    )

    assert reason == "tick cap"
    assert sent and all(command == ZERO for command in sent)


def test_lost_pose_search_never_rotates_without_yaw_telemetry(monkeypatch):
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_DELAY_S", 0.0)
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.1)
    sent, reason, _records = run_ticks(
        30,
        lambda _st: None,
        hooks_extra={"localization_yaw_search_pcmd": 8},
    )

    assert reason == "localization lost -> land"
    assert sent and all(command == ZERO for command in sent)


def test_hover_interrupts_lost_pose_search_without_restarting_it(monkeypatch):
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_DELAY_S", 0.0)
    monkeypatch.setattr(pff, "LOST_LAND_S", 100.0)
    telemetry = {"yaw": 0.0}
    safety_calls = {"count": 0}
    events = []

    def lost_pose(st):
        telemetry["yaw"] += math.radians(5.0)
        return None

    def safety_poll():
        safety_calls["count"] += 1
        return "HOVER" if 7 <= safety_calls["count"] <= 10 else "AUTO"

    sent, reason, _records = run_ticks(
        30,
        lost_pose,
        hooks_extra={
            "olympe_yaw": lambda: telemetry["yaw"],
            "localization_yaw_search_pcmd": 8,
            "localization_yaw_search_event": (
                lambda state, progress: events.append((state, progress))
            ),
            "safety_poll": safety_poll,
        },
    )

    assert reason == "tick cap"
    assert (0, 0, 8, 0) in sent
    assert [state for state, _progress in events].count("started") == 1


def test_localization_loss_can_hover_indefinitely_without_auto_land():
    sent, reason, records = run_ticks(
        200,
        lambda st: None,
        hooks_extra={
            "land_on_localization_loss": False,
            "force_relocalize": lambda: (_ for _ in ()).throw(
                RuntimeError("relocalizer unavailable")
            ),
        },
    )
    assert sent and all(command == ZERO for command in sent)
    assert reason == "tick cap"
    assert any("no fresh pose" in record["reason"] for record in records)
    assert any("relocalizer unavailable" in record["relocalize_error"] for record in records)


def test_legacy_live_entrypoint_disables_localization_loss_auto_land():
    forced_modes = []
    hooks = pff._make_live_loop_hooks(
        localizer=SimpleNamespace(
            get_pose=lambda: None,
            last_info={},
            state=SimpleNamespace(mode="TRACK"),
        ),
        safety=SimpleNamespace(
            allow_manual=False,
            force=lambda mode: forced_modes.append(mode),
        ),
        stick_monitor=SimpleNamespace(is_active=lambda: False),
        drone=object(),
        send_pcmd=lambda *_command: None,
        monitor=SimpleNamespace(
            terminated=threading.Event(),
            send_authorized=lambda *_args: (True, "ok", ZERO),
            mode="AUTO",
            beat=lambda: None,
            run_while_holding_zero=lambda action, *_args: action(),
            pcmd_timing_snapshot=lambda: {},
        ),
        grabber=SimpleNamespace(
            is_healthy=lambda: True,
            last_frame_age=lambda: 0.0,
            stamp_source="test",
        ),
        stop={"f": False},
        ground_speed=SimpleNamespace(sample=lambda: (0.0, 0.0)),
        command_log=lambda _record: None,
        inspection_ack=lambda *_args: True,
    )

    assert hooks.land_on_localization_loss is False
    hooks.pose_jump_pause(3.0)
    assert forced_modes == ["hover"]


def test_lost_pose_search_completes_then_hovers_zero_pcmd_without_auto_land(
    monkeypatch,
):
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_DELAY_S", 0.0)
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.01)
    telemetry = {"yaw": 0.0}
    events = []

    def lost_pose(_st):
        telemetry["yaw"] = (
            telemetry["yaw"] + math.radians(90.0)
        ) % (2.0 * math.pi)
        return None

    sent, reason, records = run_ticks(
        60,
        lost_pose,
        hooks_extra={
            "olympe_yaw": lambda: telemetry["yaw"],
            "localization_yaw_search_pcmd": 8,
            "localization_yaw_search_event": (
                lambda state, progress: events.append((state, progress))
            ),
            "land_on_localization_loss": False,
        },
    )

    yaw_commands = [command for command in sent if command != ZERO]
    assert yaw_commands
    assert all(command == (0, 0, 8, 0) for command in yaw_commands)
    assert reason == "tick cap"
    assert [state for state, _progress in events] == ["started", "completed"]
    last_yaw = max(
        index for index, command in enumerate(sent) if command != ZERO
    )
    assert all(command == ZERO for command in sent[last_yaw + 1:])
    assert not any(
        "-> land" in record.get("reason", "")
        for record in records
    )


def test_position_without_orientation_holds_zero_until_same_frame_recovery(
    monkeypatch,
):
    monkeypatch.setattr(pff, "LOST_YAW_SEARCH_DELAY_S", 0.0)
    monkeypatch.setattr(pff, "LOST_LAND_S", 0.01)
    telemetry = {"yaw": 0.0}
    events = []
    lost_until_tick = 10

    def pose(st):
        # App-level same-frame contract: a fresh position WITHOUT orientation
        # yields no AUTO pose at all (None) instead of a pose with stale yaw.
        if st["tick"] < lost_until_tick:
            telemetry["yaw"] = (
                telemetry["yaw"] + math.radians(90.0)
            ) % (2.0 * math.pi)
            return None
        return fresh_pose(
            st, x=min(7.99, 0.5 * (st["tick"] - lost_until_tick))
        )

    sent, reason, records = run_ticks(
        120,
        pose,
        hooks_extra={
            "olympe_yaw": lambda: telemetry["yaw"],
            "localization_yaw_search_pcmd": 8,
            "localization_yaw_search_event": (
                lambda state, progress: events.append((state, progress))
            ),
            "land_on_localization_loss": False,
        },
    )

    lost_window = sent[:lost_until_tick]
    assert all(
        command == ZERO or command == (0, 0, 8, 0)
        for command in lost_window
    )
    assert reason == "route complete -> land"
    assert [state for state, _progress in events] == ["started", "completed"]
    moving = [command for command in sent[lost_until_tick:] if command != ZERO]
    assert moving
    assert any(
        "localization recovery confirmation" in record.get("reason", "")
        for record in records
    )


def test_weak_pose_hovers():
    sent, reason, _ = run_ticks(
        600, fresh_pose, hooks_extra={"pose_is_weak": lambda: True})
    assert sent and all(c == ZERO for c in sent)
    assert reason == "low confidence -> land"


def test_real_route_weak_gate_cannot_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("SFM_GATE_WEAK", "0")
    monkeypatch.setattr(pff, "GATE_WEAK", os.environ["SFM_GATE_WEAK"] != "0")
    sent, reason, _ = run_ticks(
        600, fresh_pose, hooks_extra={"pose_is_weak": lambda: True})
    assert sent and all(c == ZERO for c in sent)
    assert reason == "low confidence -> land"


def test_offline_runner_can_explicitly_disable_weak_pose_gate(monkeypatch):
    monkeypatch.setattr(pff, "GATE_WEAK", False)
    sent, _reason, _ = run_ticks(
        8,
        fresh_pose,
        hooks_extra={"pose_is_weak": lambda: True},
        enforce_weak_pose_gate=False,
    )
    assert any(c != ZERO for c in sent)


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


def test_brief_pose_dropout_uses_imu_heading_without_treating_failure_as_weak(
    monkeypatch,
):
    monkeypatch.setattr(pff, "POSE_HOLDOVER_S", pff.POSE_STALE_S)
    telemetry = {"yaw": math.pi / 2.0, "inliers": 100}

    def pose_fn(st):
        if st["tick"] == 8:
            telemetry.update(yaw=math.pi / 2.0 - 0.05, inliers=0)
            return None
        if st["tick"] == 9:
            telemetry.update(yaw=math.pi / 2.0 - 0.10, inliers=0)
            return None
        visual_yaw = 0.10 if st["tick"] > 9 else 0.0
        telemetry["inliers"] = 100
        return fresh_pose(st, yaw=visual_yaw)

    _sent, _reason, records = run_ticks(
        16,
        pose_fn,
        hooks_extra={
            "olympe_yaw": lambda: telemetry["yaw"],
            "pose_confidence": lambda: telemetry["inliers"],
        },
    )

    holdover = [record for record in records if record.get("pose_source") == "holdover"]
    assert holdover, "a short failed-frame run should use the bounded last-good holdover"
    assert any(abs(float(record.get("heading_deg", 0.0))) > 1.0 for record in holdover)
    assert all("low confidence" not in record.get("reason", "") for record in holdover)


def test_repeated_latest_pose_is_holdover_not_a_new_visual_yaw_anchor(monkeypatch):
    monkeypatch.setattr(pff, "POSE_HOLDOVER_S", pff.POSE_STALE_S)
    telemetry = {"yaw": math.pi / 2.0}
    frozen = {"pose": None}

    def pose_fn(st):
        if st["tick"] < 8:
            return fresh_pose(st)
        if frozen["pose"] is None:
            frozen["pose"] = fresh_pose(st)
        telemetry["yaw"] = math.pi / 2.0 - 0.05 * (st["tick"] - 7)
        return frozen["pose"]

    _sent, _reason, records = run_ticks(
        14,
        pose_fn,
        hooks_extra={"olympe_yaw": lambda: telemetry["yaw"]},
    )

    holdover = [record for record in records if record.get("pose_source") == "holdover"]
    assert holdover
    assert any(abs(float(record.get("heading_deg", 0.0))) > 1.0 for record in holdover)


def test_pose_dropout_past_holdover_hovers_then_confirms_recovery(monkeypatch):
    monkeypatch.setattr(pff, "POSE_HOLDOVER_S", 0.0)

    def pose_fn(st):
        if st["tick"] in (8, 9):
            return None
        return fresh_pose(st)

    sent, _reason, records = run_ticks(16, pose_fn)

    assert sent[7] == ZERO
    assert any("no fresh pose" in record.get("reason", "") for record in records)
    recovered = next(
        record for record in records[9:]
        if "recovery confirmation 1/2" in record.get("reason", "")
    )
    assert recovered["pcmd"] == [0, 0, 0, 0]
    assert any(command != ZERO for command in sent[10:])


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


def test_live_pose_jump_latches_hover_until_operator_resumes_auto():
    mode = {"value": "AUTO", "hover_polls": 0}
    pauses = []

    def pause_for_jump(distance):
        pauses.append(distance)
        mode["value"] = "HOVER"

    def safety_poll():
        value = mode["value"]
        if value == "HOVER":
            mode["hover_polls"] += 1
            if mode["hover_polls"] >= 4:
                mode["value"] = "AUTO"
        return value

    sent, reason, records = run_ticks(
        30,
        lambda st: fresh_pose(st, x=0.0 if st["tick"] <= 3 else 3.0),
        hooks_extra={
            "pose_jump_pause": pause_for_jump,
            "safety_poll": safety_poll,
        },
    )

    assert reason == "tick cap"
    assert pauses == [pytest.approx(3.0)]
    latched = next(record for record in records if record.get("pose_jump_pause_latched"))
    resumed = next(record for record in records if record.get("pose_jump_resume_authorized"))
    latched_index = records.index(latched)
    resumed_index = records.index(resumed)
    assert all(
        record.get("pcmd") in (None, [0, 0, 0, 0])
        for record in records[latched_index:resumed_index]
    )
    assert any(command != ZERO for command in sent[resumed_index:])


def test_isolated_low_confidence_jump_is_rejected_without_latching_hover():
    quality = {"inliers": 100}
    pauses = []

    def pose_fn(st):
        jumped = st["tick"] == 4
        quality["inliers"] = 10 if jumped else 100
        return fresh_pose(st, x=3.0 if jumped else 0.0)

    sent, reason, records = run_ticks(
        20,
        pose_fn,
        hooks_extra={
            "pose_confidence": lambda: quality["inliers"],
            "pose_jump_pause": pauses.append,
        },
    )

    assert reason == "tick cap"
    assert sent[3] == ZERO
    assert pauses == []
    rejected = next(record for record in records if record.get("jump_reject_u"))
    assert rejected["jump_low_confidence"] is True
    assert rejected["low_conf_jump_count"] == 1


def test_repeated_low_confidence_jumps_in_window_latch_hover():
    quality = {"inliers": 100}
    pauses = []

    def pose_fn(st):
        jumped = st["tick"] in {4, 6, 8}
        quality["inliers"] = 10 if jumped else 100
        return fresh_pose(st, x=3.0 if jumped else 0.0)

    _sent, reason, records = run_ticks(
        20,
        pose_fn,
        hooks_extra={
            "pose_confidence": lambda: quality["inliers"],
            "pose_jump_pause": pauses.append,
        },
    )

    assert reason == "tick cap"
    assert pauses == [pytest.approx(3.0)]
    rejected = [record for record in records if record.get("jump_low_confidence")]
    assert [record["low_conf_jump_count"] for record in rejected] == [1, 2, 3]
    assert "pose_jump_pause_latched" not in rejected[0]
    assert "pose_jump_pause_latched" not in rejected[1]
    assert rejected[2]["pose_jump_pause_latched"] is True


def test_stale_pose_hovers():
    # stamps frozen in the past -> freshness gate blocks, hover, then land
    sent, reason, _ = run_ticks(200, lambda st: rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=-10.0))
    assert sent and all(c == ZERO for c in sent)
    assert reason == "localization lost -> land"


# ---------------------------------------------------------------------------
# Route gates

def test_route_deviation_hovers_once_then_rejoins_without_yaw():
    sent, reason, records = run_ticks(20, lambda st: fresh_pose(st, z=5.0))
    assert reason == "tick cap"
    assert sent[0] == ZERO
    assert any(command != ZERO for command in sent[1:])
    assert any(
        "hover before rejoin" in record.get("reason", "") for record in records
    )
    rejoin_records = [record for record in records if record.get("action") == "REJOIN"]
    assert rejoin_records
    assert all(record["pcmd"][2] == 0 for record in rejoin_records)


def test_route_deviation_hover_uses_the_active_route_controller_limit():
    sent, reason, records = run_ticks(
        20,
        lambda st: fresh_pose(st, z=0.3),
        control_config=rpf.ControlConfig(
            inspect_waypoints=(),
            max_route_deviation=0.25,
        ),
    )

    assert reason == "tick cap"
    assert sent[0] == ZERO
    assert any(command != ZERO for command in sent[1:])
    assert any(
        "0.25u -> hover before rejoin" in record.get("reason", "")
        for record in records
    )


def test_zero_route_deviation_hovers_on_any_nonzero_cross_track_error():
    sent, reason, records = run_ticks(
        20,
        lambda st: fresh_pose(st, z=0.0001),
        control_config=rpf.ControlConfig(
            inspect_waypoints=(),
            max_route_deviation=0.0,
        ),
    )

    assert reason == "tick cap"
    assert sent[0] == ZERO
    assert any(
        "0.0u -> hover before rejoin" in record.get("reason", "")
        for record in records
    )
    assert any(
        record.get("reason") == "route rejoin complete -> hover"
        for record in records
    )


def test_route_rejoin_uses_fixed_projection_and_hovers_on_completion():
    def pose_fn(st):
        z_by_tick = {1: 0.30, 2: 0.20, 3: 0.10, 4: 0.0}
        return fresh_pose(st, x=2.0, z=z_by_tick.get(st["tick"], 0.0))

    sent, reason, records = run_ticks(
        12,
        pose_fn,
        control_config=rpf.ControlConfig(
            inspect_waypoints=(),
            max_route_deviation=0.25,
        ),
    )

    assert reason == "tick cap"
    breach = next(record for record in records if "hover before rejoin" in record.get("reason", ""))
    assert breach["pcmd"] == [0, 0, 0, 0]
    assert breach["route_rejoin_target"] == [2.0, 0.0, 0.0]
    rejoin = [record for record in records if record.get("action") == "REJOIN"]
    assert any(record["pcmd"] != [0, 0, 0, 0] for record in rejoin)
    assert all(record["pcmd"][2] == 0 for record in rejoin)
    assert any(
        record.get("reason") == "route rejoin complete -> hover"
        for record in records
    )


def test_route_tube_distance_uses_the_whole_polyline():
    route = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    _sent, reason, records = run_ticks(
        8,
        lambda st: fresh_pose(st, x=1.0, z=0.5),
        route=route,
        control_config=rpf.ControlConfig(
            inspect_waypoints=(),
            max_route_deviation=0.1,
        ),
    )

    assert reason == "tick cap"
    assert any(record.get("route_distance_u") == 0.0 for record in records)
    assert not any(
        "hover before rejoin" in record.get("reason", "") for record in records
    )


def test_route_completion_lands():
    sent, reason, _ = run_ticks(
        40,
        lambda st: fresh_pose(st, x=min(7.99, 0.5 * st["tick"])),
    )
    assert reason == "route complete -> land"
    assert sent[-1] == ZERO


def test_route_completion_holds_when_ground_speed_is_unavailable():
    sent, reason, records = run_ticks(
        80,
        lambda st: fresh_pose(st, x=min(7.99, 0.5 * st["tick"])),
        hooks_extra={"ground_speed": lambda: None},
    )

    assert reason == "tick cap"
    assert sent[-1] == ZERO
    assert any(
        "ground speed unavailable" in record.get("reason", "")
        for record in records
    )


def test_route_completion_waits_for_fresh_low_ground_speed():
    speed_state = {"stamp": 0.0, "reads": 0}

    def pose(st):
        speed_state["stamp"] = st["t"]
        return fresh_pose(st, x=min(7.99, 0.5 * st["tick"]))

    def ground_speed():
        speed_state["reads"] += 1
        speed = 0.11 if speed_state["reads"] <= 2 else 0.09
        return speed, speed_state["stamp"]

    sent, reason, records = run_ticks(
        80,
        pose,
        hooks_extra={"ground_speed": ground_speed},
    )

    assert reason == "route complete -> land"
    assert speed_state["reads"] == 3
    assert sent[-1] == ZERO
    assert sum(
        "ground speed 0.110 m/s" in record.get("reason", "")
        for record in records
    ) == 2


def test_route_completion_holds_when_ground_speed_is_stale():
    def ground_speed():
        return 0.0, clock["t"] - 1.0

    clock = {"t": 0.0}

    def pose(st):
        clock["t"] = st["t"]
        return fresh_pose(st, x=min(7.99, 0.5 * st["tick"]))

    sent, reason, records = run_ticks(
        80,
        pose,
        hooks_extra={"ground_speed": ground_speed},
    )

    assert reason == "tick cap"
    assert sent[-1] == ZERO
    assert any("ground speed stale" in record.get("reason", "") for record in records)


@pytest.mark.parametrize(
    ("sample", "allowed", "reason_fragment"),
    [
        (None, False, "unavailable"),
        ((float("nan"), 10.0), False, "invalid"),
        ((0.09, 9.49), False, "stale"),
        ((0.09, 10.1), False, "future"),
        ((0.11, 10.0), False, ">"),
        ((0.09, 10.0), True, "<="),
    ],
)
def test_landing_speed_gate_fails_closed(sample, allowed, reason_fragment):
    ok, reason, _speed = pff.landing_speed_allows_land(sample, now=10.0)

    assert ok is allowed
    assert reason_fragment in reason


def test_olympe_ground_speed_tracker_uses_event_receipt_time():
    event = SimpleNamespace(
        args={"speedX": 0.3, "speedY": 0.4, "speedZ": 0.0},
        uuid="speed-1",
        date=SimpleNamespace(timestamp=lambda: 99.8),
    )
    drone = SimpleNamespace(get_last_event=lambda _message: event)
    clock = {"mono": 10.0, "wall": 100.0}
    tracker = pff.OlympeGroundSpeedTracker(
        drone,
        message=object(),
        now=lambda: clock["mono"],
        wall_now=lambda: clock["wall"],
    )

    speed, stamp = tracker.sample()
    assert speed == pytest.approx(0.5)
    assert stamp == pytest.approx(9.8)
    clock["mono"] = 10.31
    clock["wall"] = 100.31
    speed, stamp = tracker.sample()
    ok, reason, _ = pff.landing_speed_allows_land((speed, stamp), now=clock["mono"])
    assert not ok
    assert "stale" in reason


def test_final_waypoint_requires_continuous_unique_pose_hold():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    cfg = rpf.ControlConfig(
        inspect_waypoints=(),
        waypoint_arrive_confirm_frames=1,
        final_arrive_confirm_frames=3,
        final_hold_s=0.5,
    )
    ctrl = rpf.RouteAutoController(wp, poles=[], config=cfg)

    ctrl.step(rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=0.9), now=0.9)
    repeated = rpf.Pose(0.995, 0.0, 0.0, 0.0, stamp=1.0)
    for now in (1.0, 1.3, 1.7):
        cmd = ctrl.step(repeated, now=now)
        assert not cmd.should_land, "re-reading one captured pose must not arm landing"

    for stamp in (1.8, 2.05, 2.31):
        cmd = ctrl.step(
            rpf.Pose(0.995, 0.0, 0.0, 0.0, stamp=stamp),
            now=stamp,
        )
    assert cmd.should_land


def test_final_tolerance_never_exceeds_waypoint_arrival_sphere():
    cfg = rpf.ControlConfig(
        inspect_waypoints=(),
        arrive=1.0,
        min_arrive=0.1,
        waypoint_arrive_radius=0.02,
    )
    ctrl = rpf.RouteAutoController(
        [np.zeros(3), np.array([10.0, 0.0, 0.0])], poles=[], config=cfg
    )
    assert ctrl.final_arrive_tolerance() == pytest.approx(0.02)


def test_route_completion_rejects_a_single_jump_to_the_end():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([8.0, 0.0, 0.0])]
    ctrl = rpf.RouteAutoController(
        wp,
        poles=[],
        config=rpf.ControlConfig(inspect_waypoints=()),
    )
    cmd = ctrl.step(rpf.Pose(7.99, 0.0, 0.0, 0.0, stamp=1.0), now=1.0)
    assert not cmd.should_land
    assert cmd.progress < 0.5


def test_progress_does_not_snap_to_a_nearby_return_branch():
    wp = [
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.5]),
    ]
    ctrl = rpf.RouteAutoController(
        wp,
        poles=[],
        config=rpf.ControlConfig(
            inspect_waypoints=(),
            segment_window=2,
            progress_jump_slack=1.0,
        ),
    )
    ctrl.step(rpf.Pose(0.2, 0.0, 0.0, 0.0, stamp=1.0), now=1.0)
    cmd = ctrl.step(rpf.Pose(0.2, 0.0, 0.49, 0.0, stamp=1.05), now=1.05)

    assert cmd.progress < 0.5
    assert ctrl.active_segment == 0
    assert not cmd.should_land


def test_invalid_route_rejected(tmp_path):
    bad = tmp_path / "route.json"
    bad.write_text(json.dumps({"waypoints": [[0.0, 0.0, 0.0]]}))
    with pytest.raises(ValueError):
        rpf.load_waypoints(bad)


def test_flight_route_contract_binds_site_frame_and_coordinate_id(tmp_path):
    route = tmp_path / "route.json"
    route.write_text(json.dumps({
        "schema": "sfm-flight-route/v1",
        "site_id": "alpha",
        "coordinate_frame_id": "alpha-v1",
        "frame": "glomap",
        "units": "map",
        "purpose": "flight",
        "closed": False,
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }), encoding="utf-8")

    points = rpf.load_waypoints(
        route,
        expected_site_id="alpha",
        expected_coordinate_frame_id="alpha-v1",
        require_flight_contract=True,
    )
    assert np.allclose(points[1], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="coordinate_frame_id"):
        rpf.load_waypoints(
            route,
            expected_site_id="alpha",
            expected_coordinate_frame_id="different",
            require_flight_contract=True,
        )
    raw = json.loads(route.read_text(encoding="utf-8"))
    raw["closed"] = 0
    route.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="closed"):
        rpf.load_waypoints(
            route,
            expected_site_id="alpha",
            expected_coordinate_frame_id="alpha-v1",
            require_flight_contract=True,
        )


def _write_flight_route(path: Path, *, end_x: float = 8.0) -> str:
    path.write_text(json.dumps({
        "schema": "sfm-flight-route/v1",
        "site_id": "field-a",
        "coordinate_frame_id": "glomap-a",
        "units": "map",
        "purpose": "flight",
        "closed": False,
        "frame": "glomap",
        "waypoints": [[0.0, 0.0, 0.0], [end_x, 0.0, 0.0]],
    }), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mission_route_snapshot_rejects_file_changed_before_auto(tmp_path):
    route = tmp_path / "flight_path.json"
    digest = _write_flight_route(route)
    snapshot = rpf.capture_mission_route_snapshot(
        route,
        expected_sha256=digest,
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
    )
    lock = rpf.MissionRouteLock(snapshot)

    _write_flight_route(route, end_x=9.0)

    with pytest.raises(ValueError, match="changed after selection"):
        lock.begin_auto(displayed_sha256=digest)
    assert not lock.active


def test_auto_route_lock_keeps_same_snapshot_until_confirmed_landed(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = rpf.capture_mission_route_snapshot(
        first_path,
        expected_sha256=_write_flight_route(first_path),
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
    )
    second = rpf.capture_mission_route_snapshot(
        second_path,
        expected_sha256=_write_flight_route(second_path, end_x=12.0),
        expected_site_id="field-a",
        expected_coordinate_frame_id="glomap-a",
    )
    lock = rpf.MissionRouteLock(first)

    assert lock.begin_auto(displayed_sha256=first.sha256) is first
    lock.confirm_auto_started()
    assert lock.resume_auto() is first
    lock.cancel_rejected_auto_start()
    assert lock.active
    with pytest.raises(ValueError, match="cannot switch route"):
        lock.bind(second)
    with pytest.raises(ValueError, match="confirmed landed"):
        lock.release_after_confirmed_landed("hovering")

    lock.release_after_confirmed_landed("landed")
    lock.bind(second)
    assert lock.snapshot is second


def test_route_controller_deep_copies_and_freezes_waypoints():
    source = [np.array([0.0, 0.0, 0.0]), np.array([8.0, 0.0, 0.0])]
    controller = rpf.RouteAutoController(
        source,
        config=rpf.ControlConfig(inspect_waypoints=()),
    )

    source[1][0] = 99.0

    assert controller.wp[1][0] == pytest.approx(8.0)
    with pytest.raises(ValueError, match="read-only"):
        controller.wp[1][0] = 7.0


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
        rpf.ControlConfig(lookahead=float("nan"))
    with pytest.raises(ValueError):
        rpf.ControlConfig(max_pose_age_s=0.0)


def test_malformed_pole_rejected(tmp_path):
    bad = tmp_path / "poles.json"
    bad.write_text(json.dumps({"poles": [{"center": [0.0, float("nan"), 0.0]}]}))
    with pytest.raises(ValueError):
        rpf.load_poles(bad)


def test_pole_geometry_uses_the_measured_horizontal_plane():
    frame = rpf.MapFrame.from_gravity([0.0, 0.0, 1.0])
    poles = [
        {"id": 1, "center": np.array([0.0, 0.0, 10.0])},
        {"id": 2, "center": np.array([1.0, 0.0, 0.0])},
    ]

    pole_id, _pole, distance = rpf.nearest_pole(poles, np.zeros(3), frame)

    assert pole_id == 1
    assert distance == pytest.approx(0.0)


def test_boot_lock_requires_consecutive_fixes_near_route_start():
    lock = pff.BootPoseLock(np.zeros(3), required_fixes=3,
                            max_start_distance=1.0, max_fix_jump=0.4)
    assert not lock.observe(rpf.Pose(0.1, 0.0, 0.0, 0.0, stamp=1.0), now=1.1)
    assert not lock.observe(rpf.Pose(0.2, 0.0, 0.0, 0.0, stamp=1.1), now=1.2)
    assert lock.observe(rpf.Pose(0.25, 0.0, 0.0, 0.0, stamp=1.2), now=1.3)
    assert not lock.observe(rpf.Pose(3.0, 0.0, 0.0, 0.0, stamp=1.3), now=1.4)
    assert lock.count == 0


def test_boot_lock_accepts_stable_fixes_near_any_route_waypoint():
    route = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [15.0, 0.0, 0.0],
    ])
    lock = pff.BootPoseLock(
        route,
        required_fixes=3,
        max_start_distance=1.0,
        max_fix_jump=0.4,
    )

    assert not lock.observe(rpf.Pose(9.8, 0.0, 0.0, 0.0, stamp=1.0), now=1.1)
    assert not lock.observe(rpf.Pose(9.9, 0.0, 0.0, 0.0, stamp=1.1), now=1.2)
    assert lock.observe(rpf.Pose(10.0, 0.0, 0.0, 0.0, stamp=1.2), now=1.3)
    assert lock.nearest_waypoint_index == 2
    np.testing.assert_allclose(lock.position, [10.0, 0.0, 0.0])
    assert not lock.observe(rpf.Pose(30.0, 0.0, 0.0, 0.0, stamp=1.3), now=1.4)
    assert lock.count == 0
    assert lock.nearest_waypoint_index is None


def test_boot_lock_recovers_after_initial_hover_has_no_localization():
    lock = pff.BootPoseLock(np.zeros(3), required_fixes=3)

    assert not lock.observe(None, now=1.0)
    assert not lock.observe(None, now=1.1)
    assert not lock.observe(rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.2), now=1.2)
    assert not lock.observe(rpf.Pose(0.1, 0.0, 0.0, 0.0, stamp=1.3), now=1.3)
    assert lock.observe(rpf.Pose(0.1, 0.0, 0.0, 0.0, stamp=1.4), now=1.4)


def test_inspection_requires_explicit_capture_ack():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([1.0, 0.0, 0.4]),
            "base": np.array([1.0, 0.0, 0.0]), "top": np.array([1.0, 0.0, 0.8])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.3))
    pose = rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=2.0)
    _settle(ctrl, pose, 2.0)
    cmd = _settle(ctrl, pose, 2.1)
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
    pose = rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=2.0)
    _settle(ctrl, pose, 2.0)
    cmd = _settle(ctrl, pose, 2.1)
    assert cmd.action == "INSPECT"
    assert cmd.yaw_target == pytest.approx(np.pi / 2.0)
    assert cmd.look_at_pole["target"] == [1.0, 0.0, 2.0]


def test_inspection_gimbal_pitch_is_derived_from_target_and_pose():
    pose = rpf.Pose(0.0, -2.0, 0.0, 0.0, stamp=1.0)  # camera up=+2 in -Y-up map
    meta = {"waypoint": 2, "pole_id": 1, "target": [2.0, 0.0, 0.0]}
    pitch = pff.inspection_gimbal_pitch_deg(meta, pose)
    assert pitch == pytest.approx(-45.0)
    assert pff.inspection_gimbal_pitch_deg({"target": [np.nan, 0.0, 0.0]}, pose) is None


def test_inspection_gimbal_pitch_uses_measured_map_frame():
    frame = rpf.MapFrame.from_gravity([0.0, 0.0, 1.0])
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0)
    meta = {"target": [1.0, 0.0, -1.0]}

    assert pff.inspection_gimbal_pitch_deg(meta, pose, frame) == pytest.approx(45.0)


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
        if clock["calls"] > 90:
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


def test_inspection_without_body_yaw_requirement_can_capture():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([2.0, 0.0, 0.0]),
            "base": np.array([2.0, 0.0, 0.0]), "top": np.array([2.0, -1.0, 0.0])}
    ctrl = rpf.RouteAutoController(
        wp,
        [pole],
        rpf.ControlConfig(
            inspect_waypoints=(2,), inspect_radius=0.3, pole_body_look=False,
        ),
    )
    clock = {"t": 1.0, "calls": 0, "captures": 0}

    def now():
        clock["calls"] += 1
        if clock["calls"] > 60:
            raise KeyboardInterrupt
        clock["t"] += 0.05
        return clock["t"]

    def capture(*_args):
        clock["captures"] += 1
        return True

    hooks = pff.LoopHooks(
        get_pose=lambda: rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=clock["t"]),
        olympe_yaw=lambda: 0.0,
        send_pcmd=lambda *_: None,
        inspection_ack=capture,
        now=now,
    )
    with pytest.raises(KeyboardInterrupt):
        pff.run_loop(hooks, ctrl, wp, verbose=False)

    assert clock["captures"] > 0
    assert ctrl.completed_inspections == {1}


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
        if clock["calls"] > 60:
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


def test_visual_yaw_allows_inspection_when_olympe_yaw_is_missing():
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
    assert clock["captures"] > 0
    assert sent and all(cmd == ZERO for cmd in sent)


def test_inspection_radius_jitter_never_forces_landing():
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
    assert reason == "tick cap"


def test_a_pending_inspection_waypoint_cannot_be_skipped():
    """Direct-to-waypoint replaced the old 'abort at route end' rule.

    The target holds on an un-inspected waypoint, so overshooting it commands the
    drone BACK instead of carrying on to the end with the inspection undone.
    """
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([1.0, 0.0, 0.0]),
            "base": np.array([1.0, 0.0, 0.0]), "top": np.array([1.0, -1.0, 0.0])}
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.2))

    _settle(ctrl, rpf.Pose(0.4, 0.0, 0.0, 0.0, stamp=1.0), 1.0)
    cmd = _settle(ctrl, rpf.Pose(1.4, 0.0, 0.0, 0.0, stamp=1.5), 1.5)

    assert ctrl.target_index == 1, "target advanced past an un-inspected waypoint"
    assert not cmd.should_land
    # Still aimed at the waypoint it overshot, not onward to the end.
    assert cmd.goal[0] == pytest.approx(1.0)
    assert 1 not in ctrl.completed_inspections


def test_parking_on_an_unreachable_inspection_waypoint_hovers():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pole = {"id": 1, "center": np.array([1.0, 0.0, 0.4]),
            "base": np.array([1.0, 0.0, 0.0]), "top": np.array([1.0, -1.0, 0.0])}
    # inspect_radius tighter than the arrival sphere: the drone parks on the
    # waypoint and the look-at region never opens.
    ctrl = rpf.RouteAutoController(
        wp, [pole], rpf.ControlConfig(inspect_waypoints=(2,), inspect_radius=0.005,
                                      waypoint_arrive_radius=0.05))

    _settle(ctrl, rpf.Pose(0.4, 0.0, 0.0, 0.0, stamp=1.0), 1.0)
    cmd = _settle(ctrl, rpf.Pose(1.02, 0.0, 0.0, 0.0, stamp=1.5), 1.5)

    assert not cmd.should_land and cmd.action == "HOVER"
    assert "no reachable target" in cmd.status
    assert "hover" in cmd.status


def test_waypoint_target_advances_on_overshoot_not_only_on_proximity():
    """A 20 Hz tick at speed steps over the arrival ball; proximity alone strands it."""
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([2.0, 0.0, 0.0])]
    ctrl = rpf.RouteAutoController(wp, poles=[],
                                   config=rpf.ControlConfig(inspect_waypoints=()))

    # Never within translation_arrival_tolerance (0.15) of waypoint 1, but past it.
    _settle(ctrl, rpf.Pose(0.5, 0.0, 0.0, 0.0, stamp=1.0), 1.0)
    cmd = _settle(ctrl, rpf.Pose(1.4, 0.0, 0.0, 0.0, stamp=1.05), 1.05)

    assert ctrl.target_index == 2
    assert cmd.goal[0] == pytest.approx(2.0)


def test_one_pose_update_can_advance_at_most_one_waypoint():
    wp = [np.array([float(x), 0.0, 0.0]) for x in range(4)]
    ctrl = rpf.RouteAutoController(
        wp,
        poles=[],
        config=rpf.ControlConfig(
            inspect_waypoints=(),
            waypoint_arrive_confirm_frames=1,
            progress_jump_slack=10.0,
        ),
    )

    ctrl.step(rpf.Pose(3.0, 0.0, 0.0, 0.0, stamp=1.0), now=1.0)
    assert ctrl.target_index == 1


def test_waypoint_target_never_regresses():
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([2.0, 0.0, 0.0])]
    ctrl = rpf.RouteAutoController(wp, poles=[],
                                   config=rpf.ControlConfig(inspect_waypoints=()))
    _settle(ctrl, rpf.Pose(1.1, 0.0, 0.0, 0.0, stamp=1.0), 1.0)
    reached = ctrl.target_index

    ctrl.step(rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.5), now=1.5)

    assert ctrl.target_index == reached, "a backward fix must not rewind the route"


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


def test_safety_switch_rejects_stale_non_auto_commands(tmp_path):
    command = tmp_path / "safety.cmd"
    command.write_text("land\n")
    command.chmod(0o600)
    switch = pff.SafetySwitch(command, keyboard=False)
    assert switch.poll() == "HOVER"
    assert command.read_text() == "land\n"


def test_safety_switch_missing_file_defaults_to_hover(tmp_path):
    command = tmp_path / "safety.cmd"
    switch = pff.SafetySwitch(command, keyboard=False)
    assert switch.poll() == "HOVER"
    assert command.read_text().strip().lower() == "hover"


def test_fly_requires_auto_written_after_this_run_started(tmp_path):
    command = tmp_path / "safety.cmd"
    command.write_text("auto\n")
    command.chmod(0o600)
    switch = pff.SafetySwitch(command, keyboard=False, require_fresh_auto=True)
    assert switch.poll() == "HOVER", "stale AUTO from a previous run must not arm"
    baseline = command.stat().st_mtime_ns
    command.write_text("auto\n")
    os.utime(command, ns=(baseline + 1_000_000, baseline + 1_000_000))
    assert switch.poll() == "AUTO"


@pytest.mark.parametrize(
    ("token", "expected", "allow_manual"),
    [("hover", "HOVER", False), ("manual", "MANUAL", True),
     ("land", "LAND", False), ("emergency", "EMERGENCY", False)],
)
def test_every_file_command_requires_fresh_write(tmp_path, token, expected, allow_manual):
    command = tmp_path / "safety.cmd"
    command.write_text(f"{token}\n")
    command.chmod(0o600)
    switch = pff.SafetySwitch(command, keyboard=False, allow_manual=allow_manual)
    assert switch.poll() == "HOVER", f"stale {token} must fail closed"
    baseline = command.stat().st_mtime_ns
    command.write_text(f"{token}\n")
    os.utime(command, ns=(baseline + 1_000_000, baseline + 1_000_000))
    assert switch.poll() == expected


def test_default_safety_file_is_created_in_private_runtime_dir(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    path = pff._default_safety_file()
    assert path == runtime / "sfm_drone" / "safety.cmd"
    switch = pff.SafetySwitch(path, keyboard=False)
    assert switch.poll() == "HOVER"
    assert (path.parent.stat().st_mode & 0o077) == 0
    assert (path.stat().st_mode & 0o077) == 0


def test_safety_file_rejects_symlink_group_writable_and_non_owner(tmp_path, monkeypatch):
    target = tmp_path / "target.cmd"
    target.write_text("hover\n")
    target.chmod(0o600)
    link = tmp_path / "link.cmd"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="safety command"):
        pff.SafetySwitch(link, keyboard=False)

    insecure = tmp_path / "insecure.cmd"
    insecure.write_text("hover\n")
    insecure.chmod(0o660)
    with pytest.raises(RuntimeError, match="safety command"):
        pff.SafetySwitch(insecure, keyboard=False)

    real_uid = os.getuid()
    monkeypatch.setattr(pff.os, "getuid", lambda: real_uid + 1)
    with pytest.raises(RuntimeError, match="owner-only|owner-owned"):
        pff.SafetySwitch(target, keyboard=False)


def test_keyboard_read_failure_fails_closed(monkeypatch):
    switch = pff.SafetySwitch(path=None, keyboard=False)
    switch.keyboard = True
    switch.mode = "AUTO"
    monkeypatch.setattr(pff.select, "select", lambda *_args, **_kw: (_ for _ in ()).throw(
        OSError("stdin failed")))
    assert switch.poll() == "HOVER"


def test_safety_monitor_starts_fail_closed_before_first_valid_command():
    monitor = pff.SafetyMonitor(lambda *_cmd: None)
    assert monitor.mode == "HOVER"


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


def test_dead_pcmd_channel_is_counted_and_never_stamped_as_delivered():
    """A failing command channel must not look like a healthy one.

    last_pcmd_call_mono_ns used to be stamped BEFORE the send attempt, so every
    failed PCMD still advanced the "last command reached the wire" marker and the
    failures were counted nowhere. The monitor must still keep running.
    """
    def dead_send(*_cmd):
        raise ConnectionError("injected dead command channel")

    mon = pff.SafetyMonitor(dead_send, timeout_s=0.06)
    assert mon.pcmd_send_failures == 0
    assert mon.last_pcmd_call_mono_ns is None

    sent, why, actual = mon.send_authorized((0, 0, 0, 0), lambda: True, lambda: False)

    assert mon.pcmd_send_failures >= 1
    assert "ConnectionError" in mon.last_pcmd_send_error
    assert mon.last_pcmd_call_mono_ns is None, (
        "a failed send stamped the wire-time marker, hiding a dead command channel"
    )
    assert mon.pcmd_timing_snapshot()["pcmd_send_failures"] >= 1


def test_successful_pcmd_stamps_the_wire_marker():
    sent = []
    mon = pff.SafetyMonitor(lambda *cmd: sent.append(tuple(cmd)), timeout_s=0.06)
    mon.mode = "AUTO"  # explicit test injection of a previously validated command
    ok, _why, _actual = mon.send_authorized((0, 0, 0, 0), lambda: True, lambda: False)
    assert ok
    assert mon.pcmd_send_failures == 0
    assert mon.last_pcmd_call_mono_ns is not None


@pytest.mark.parametrize("pcmd", [
    (11, 0, 0, 0),
    (0, 11, 0, 0),
    (0, 0, 21, 0),
    (0, 0, 0, 11),
    (1.5, 0, 0, 0),
    (True, 0, 0, 0),
    (0, 0, 0),
])
def test_final_command_authority_rejects_malformed_or_out_of_envelope_pcmd(pcmd):
    sent = []
    mon = pff.SafetyMonitor(
        lambda *cmd: sent.append(tuple(cmd)),
        max_translation_pcmd=10,
        max_yaw_pcmd=20,
    )
    mon.mode = "AUTO"

    ok, reason, actual = mon.send_authorized(pcmd, lambda: True, lambda: False)

    assert not ok
    assert "PCMD" in reason
    assert actual == ZERO
    assert sent[-1] == ZERO


def test_expired_nonzero_pcmd_is_not_replayed_by_monitor_thread():
    class _AutoSafety:
        @staticmethod
        def poll():
            return "AUTO"

    sent = []
    mon = pff.SafetyMonitor(
        lambda *cmd: sent.append((time.monotonic(), tuple(cmd))),
        _AutoSafety(),
        timeout_s=1.0,
        command_ttl_s=0.05,
    ).start()
    try:
        mon.beat()
        ok, _reason, _actual = mon.send_authorized(
            (1, 2, 3, 4), lambda: True, lambda: False
        )
        assert ok
        time.sleep(0.18)
    finally:
        mon.stop()

    nonzero_indices = [i for i, (_stamp, cmd) in enumerate(sent) if cmd != ZERO]
    assert nonzero_indices
    assert any(cmd == ZERO for _stamp, cmd in sent[nonzero_indices[-1] + 1:])


def test_pose_jump_cannot_be_confirmed_by_relocalizing_the_same_capture():
    """The confirming fix must come from a DIFFERENT capture.

    Re-localizing the same frame reproduces the same centre, so it would agree with
    itself and promote one bad frame straight to an accepted relocation.
    """
    jump = pff.MAX_POSE_JUMP_U * 3.0
    frozen = {"stamp": None}

    def pose(st):
        if st["tick"] <= 2:
            return rpf.Pose(x=0.0, y=0.0, z=0.0, yaw=0.0, stamp=st["t"])
        if frozen["stamp"] is None:
            frozen["stamp"] = st["t"]          # one capture, re-localized repeatedly
        return rpf.Pose(x=jump, y=0.0, z=0.0, yaw=0.0, stamp=frozen["stamp"])

    _sent, _reason, records = run_ticks(8, pose)[:3]
    first = next(
        (i for i, r in enumerate(records) if "jump_reject_u" in r), None)
    assert first is not None, "the teleport was never even flagged as a jump"
    # While the same capture keeps being re-presented it must keep being rejected.
    # Once it ages past POSE_STALE_S the loop legitimately reports "no fresh pose".
    accepted = [
        r for r in records[first + 1:]
        if "jump_reject_u" not in r
        and "no fresh pose" not in str(r.get("reason", ""))
    ]
    assert not accepted, (
        "a single teleported capture confirmed itself and was accepted: "
        f"{[str(r.get('reason'))[:60] for r in accepted]}"
    )


def test_stall_watchdog_escalates_to_land_instead_of_hovering_forever():
    """Holding zero forever leaves the aircraft airborne until the battery dies."""
    sent = []
    mon = pff.SafetyMonitor(lambda *cmd: sent.append(tuple(cmd)), timeout_s=0.05)
    mon.mode = "AUTO"  # explicit test injection of a previously validated command
    monkey_land = []
    mon._land_cb = lambda: monkey_land.append(True)
    mon.beat()
    mon.start()
    try:
        deadline = time.monotonic() + max(3.0, pff.WATCHDOG_LAND_S + 2.0)
        while time.monotonic() < deadline and not mon.terminated.is_set():
            time.sleep(0.05)
    finally:
        mon.stop()
    assert mon.terminated.is_set(), "stalled control loop never escalated past hover"
    assert "stall" in (mon.reason or "").lower()
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


def test_firmware_preflight_reports_low_battery_as_advisory(monkeypatch):
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


def test_default_preflight_does_not_require_gps(monkeypatch):
    drone, tokens = _fake_firmware_preflight(monkeypatch, battery=80, gps_fixed=0)
    result = pff.configure_flight_preflight(drone, 20.0, 80.0)
    assert result["distance_geofence"] is False
    assert tokens["gps"] not in drone.reads
    assert drone.calls[-1][1] == {"shouldNotFlyOver": 0}


def test_missing_firmware_limits_are_advisory(monkeypatch):
    drone, _ = _fake_firmware_preflight(monkeypatch, battery=80, gps_fixed=0)
    with pytest.raises(RuntimeError, match="were not provided"):
        pff.configure_flight_preflight(drone, None, None)
    assert drone.calls == []


def test_firmware_limit_failure_is_advisory(monkeypatch):
    drone, tokens = _fake_firmware_preflight(monkeypatch, battery=80, gps_fixed=0)
    drone.states[tokens["max_altitude"]]["max"] = 10.0
    with pytest.raises(RuntimeError, match="outside firmware range"):
        pff.configure_flight_preflight(drone, 20.0, 80.0)
    assert drone.calls == []


def test_fly_cli_defaults_to_no_gps_or_firmware_limits(monkeypatch):
    captured = {}
    monkeypatch.setattr(pff, "fly", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["path_follow_flight.py", "--fly"])

    pff.main()

    assert captured["max_altitude_m"] is None
    assert captured["max_distance_m"] is None
    assert captured["distance_geofence"] is False


def test_fly_refuses_live_takeoff() -> None:
    with pytest.raises(SystemExit, match="operator UI"):
        pff.fly("192.168.53.1", 1, -20.0, "skycontroller3", "/tmp/safety")



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
    firmware_src = inspect.getsource(pff._configure_flight_preflight_for_live)
    cleanup_src = inspect.getsource(pff._close_live_flight_resources)
    assert "TakeOff()" not in src
    assert "operator UI" in src
    helper = inspect.getsource(pff.start_skycontroller_stick_override)
    assert helper.index("stick_monitor.start()") < helper.index(
        'set_piloting_source(drone, "Controller")'
    )
    restore_i = cleanup_src.index('set_piloting_source(drone, "SkyController")')
    assert restore_i < cleanup_src.index("drone.disconnect()")
    assert "except Exception as exc" in firmware_src
    assert "continuing without confirmed firmware limits" in firmware_src


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


def test_physical_stick_override_zeroes_then_confirms_manual_source():
    events = []

    class _Safety:
        mode = "AUTO"

        def force(self, token):
            assert token == "manual"
            self.mode = "MANUAL"
            return True

    safety = _Safety()
    mon = pff.SafetyMonitor(
        lambda *cmd: events.append(("pcmd", tuple(cmd))),
        safety,
        piloting_source_cb=lambda source: events.append(("source", source)) or True,
        timeout_s=0.06,
    )
    mon.mode = "AUTO"
    mon._desired_pcmd = (0, 8, 0, 0)
    mon._desired_valid = True

    accepted, reason = mon.request_manual_override()

    assert accepted and "SkyController manual override" in reason
    assert events == [("pcmd", ZERO), ("source", "SkyController")]
    assert mon.mode == "MANUAL" and not mon._desired_valid


def test_physical_stick_override_source_failure_latches_land():
    class _Safety:
        mode = "AUTO"

        def force(self, _token):
            self.mode = "MANUAL"
            return True

    mon = pff.SafetyMonitor(
        lambda *_cmd: None,
        _Safety(),
        piloting_source_cb=lambda _source: False,
        timeout_s=0.06,
    )
    mon.mode = "AUTO"

    accepted, reason = mon.request_manual_override()

    assert not accepted
    assert mon.terminal_action == "LAND" and mon.terminated.is_set()
    assert "handoff" in reason


def test_non_finite_safety_environment_value_rejected(monkeypatch):
    monkeypatch.setenv("SFM_TEST_SAFETY", "nan")
    with pytest.raises(ValueError):
        pff._env_float("SFM_TEST_SAFETY", 1.0, minimum=0.1, maximum=10.0)


def test_takeoff_has_final_arming_gate_cleanup_and_termination_signals():
    src = inspect.getsource(pff.fly)
    boot_src = inspect.getsource(pff._wait_for_boot_lock)
    route_src = inspect.getsource(pff._run_live_route_after_takeoff)
    authorize_src = inspect.getsource(pff._authorize_cleanup_zero)
    terminal_src = inspect.getsource(pff._perform_terminal_flight_action)
    inspection_src = inspect.getsource(pff._capture_inspection_frame)
    signal_src = inspect.getsource(pff._install_flight_stop_handlers)
    assert "TakeOff()" not in src
    assert "operator UI" in src
    assert "arming_allowed" in boot_src and "arming_allowed" in route_src
    # Behavioral replacement for ordering text assert: SafetyMonitor latch is
    # monotonic and terminal (replaces authorize_src.index ordering check).
    mon = pff.SafetyMonitor(lambda *_: None, timeout_s=0.05)
    assert mon.terminal_action == "NONE" and not mon.terminated.is_set()
    mon._latch_terminal("LAND", "test land")
    assert mon.terminal_action == "LAND" and mon.terminated.is_set()
    assert mon.reason == "test land"
    mon._latch_terminal("HOVER", "should not downgrade terminal land")
    assert mon.terminal_action == "LAND"
    mon._latch_terminal("EMERGENCY", "upgrade to emergency")
    assert mon.terminal_action == "EMERGENCY"
    mon._latch_terminal("LAND", "cannot downgrade emergency")
    assert mon.terminal_action == "EMERGENCY"
    assert "monitor.send_authorized" in authorize_src
    assert "emergency_issued" in terminal_src
    assert inspection_src.index("require_confirmation=True") < inspection_src.index(
        "baseline_source_us =")
    assert "inspection_sample_after" in inspection_src
    assert "INSPECTION_PIPELINE_DRAIN_S" in inspection_src
    for sig in ("SIGINT", "SIGTERM", "SIGHUP"):
        assert sig in signal_src


def test_fly_arms_and_cleans_up_physical_stick_override_before_takeoff():
    src = inspect.getsource(pff.fly)
    hooks_src = inspect.getsource(pff._make_live_loop_hooks)
    assert "TakeOff()" not in src
    assert "operator UI" in src
    assert 'stick_active=getattr(stick_monitor, "is_active", None)' in hooks_src

    helper = inspect.getsource(pff.start_skycontroller_stick_override)
    assert "if not stick_monitor.start()" in helper
    assert "refusing autonomous takeoff" in helper



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
    )
    mon.mode = "AUTO"  # explicit test injection of a previously validated command
    mon.start()
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
    # 11 ticks, not 6: a waypoint now costs waypoint_arrive_confirm_frames to
    # retire and yaw_alignment_confirmation_updates to unlock translation, so the
    # loop legitimately sends zeros for the first ~6 ticks of any route.
    def pose_fn(st):
        return fresh_pose(st) if st["tick"] <= 10 else None
    sent, reason, _ = run_ticks(200, pose_fn)
    assert any(c != ZERO for c in sent[:11]), "sanity: loop was actually driving first"
    first_zero_after_loss = next(i for i in range(11, len(sent)) if sent[i] == ZERO)
    assert all(c == ZERO for c in sent[first_zero_after_loss:]), \
        "after the loss is recognized, the previous nonzero command must never be reused"
    assert reason == "localization lost -> land"


# ---------------------------------------------------------------------------
# PCMD conversion

def test_pcmd_turns_in_place_before_translation():
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0)
    cmd = rpf.Command(
        "FOLLOW",
        np.array([1.0, 0.0, 1.0]),
        yaw_target=np.pi / 4.0,
        goal=np.array([1.0, 0.0, 1.0]),
        path_error=0.0,
        progress=0.0,
    )

    assert rpf.command_to_body_percent(cmd, pose) == (0, 0, 20, 0)
    assert rpf.command_to_body_percent(cmd, pose, yaw_sign=-1) == (0, 0, -20, 0)


def test_yaw_error_uses_both_rays_projected_on_the_measured_ground_plane():
    frame = rpf.MapFrame.from_gravity([0.0, 1.0, 1.0])
    target_delta = frame.north + 2.0 * frame.up

    assert rpf.ground_projected_yaw_error(0.0, target_delta, frame) == pytest.approx(
        math.pi / 2.0
    )
    assert rpf.ground_projected_yaw_error(
        math.pi / 2.0, target_delta, frame
    ) == pytest.approx(0.0)
    assert rpf.ground_projected_yaw_error(0.0, frame.up, frame) is None


def test_camera_6dof_heading_uses_ground_projection_and_rejects_vertical_view():
    frame = rpf.MapFrame.from_gravity([0.0, 1.0, 1.0])

    assert rpf.camera_heading_from_forward(
        frame.north + 2.0 * frame.up, frame
    ) == pytest.approx(math.pi / 2.0)
    assert rpf.camera_heading_from_forward(frame.up, frame) is None


def test_pcmd_alignment_is_derived_from_the_waypoint_ray_not_stale_metadata():
    frame = rpf.MapFrame.from_gravity([0.0, 1.0, 1.0])
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0)
    goal = frame.north.copy()
    cmd = rpf.Command(
        "FOLLOW",
        goal.copy(),
        yaw_target=0.0,  # Deliberately stale: geometry must remain authoritative.
        goal=goal,
        path_error=0.0,
        progress=0.0,
    )

    assert rpf.command_to_body_percent(
        cmd,
        pose,
        config=rpf.ControlConfig(inspect_waypoints=(), map_frame=frame),
    ) == (0, 0, 20, 0)


@pytest.mark.parametrize("goal, expected", [
    ([1.0, -1.0, -1.0], (6, 6, 0, 6)),
    ([1.0, 1.0, -1.0], (6, 6, 0, -6)),
])
def test_pcmd_normalizes_right_forward_vertical_diagonals(goal, expected):
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0)
    cmd = rpf.Command(
        "FOLLOW",
        np.asarray(goal, dtype=float),
        yaw_target=0.0,
        goal=np.asarray(goal, dtype=float),
        path_error=0.0,
        progress=0.0,
    )

    assert rpf.command_to_body_percent(
        cmd,
        pose,
        require_yaw_alignment=False,
    ) == expected


def test_vertical_pcmd_uses_map_gravity_not_camera_pitch_or_roll():
    frame = rpf.MapFrame.from_gravity([0.0, 1.0, 1.0])
    cfg = rpf.ControlConfig(inspect_waypoints=(), map_frame=frame)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0)

    horizontal = rpf.Command(
        "FOLLOW", frame.east + frame.north, 0.0,
        frame.east + frame.north, 0.0, 0.0,
    )
    climb = rpf.Command("FOLLOW", frame.up, 0.0, frame.up, 0.0, 0.0)

    assert rpf.command_to_body_percent(
        horizontal, pose, config=cfg, require_yaw_alignment=False,
    )[3] == 0
    assert rpf.command_to_body_percent(
        climb, pose, config=cfg, require_yaw_alignment=False,
    )[:3] == (0, 0, 0)
    assert rpf.command_to_body_percent(
        climb, pose, config=cfg, require_yaw_alignment=False,
    )[3] > 0


def test_yaw_alignment_requires_three_quiet_updates_then_translates():
    cfg = rpf.ControlConfig(inspect_waypoints=())
    control = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command(
        "FOLLOW",
        np.array([1.0, 0.0, 0.0]),
        yaw_target=0.0,
        goal=np.array([1.0, 0.0, 0.0]),
        path_error=0.0,
        progress=0.0,
    )

    for now in (0.0, 0.05, 0.10, 0.15):
        pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=now)
        assert control.update(cmd, pose, now, target_key=0) == ZERO
    assert control.phase == "yaw_alignment_confirmed"

    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.20)
    assert control.update(cmd, pose, 0.20, target_key=0)[1] > 0
    assert control.phase == "translate"

    assert control.update(cmd, pose, 0.25, target_key=1) == ZERO
    assert control.phase == "yaw_alignment_hold"


def test_translation_keeps_yaw_zero_while_error_stays_within_thirty_degrees():
    cfg = rpf.ControlConfig(
        inspect_waypoints=(),
        yaw_alignment_confirmation_updates=1,
        yaw_alignment_max_rate_deg_s=100.0,
    )
    control = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command(
        "FOLLOW",
        np.array([1.0, 0.0, 0.0]),
        yaw_target=0.0,
        goal=np.array([1.0, 0.0, 0.0]),
        path_error=0.0,
        progress=0.0,
    )

    control.update(cmd, rpf.Pose(0, 0, 0, yaw=0.0, stamp=0.0), 0.0, target_key=1)
    control.update(cmd, rpf.Pose(0, 0, 0, yaw=0.0, stamp=0.1), 0.1, target_key=1)
    translated = control.update(
        cmd,
        rpf.Pose(0.2, 0, 0, yaw=math.radians(20.0), stamp=0.2),
        0.2,
        target_key=1,
    )

    assert translated[2] == 0
    assert translated[0] > 0
    assert translated[1] > 0
    assert control.phase == "translate"


def test_translation_over_thirty_degree_yaw_error_hovers_realigns_then_resumes():
    cfg = rpf.ControlConfig(
        inspect_waypoints=(),
        yaw_alignment_confirmation_updates=1,
        yaw_alignment_max_rate_deg_s=100.0,
    )
    control = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command(
        "FOLLOW", np.array([1.0, 0.0, 0.0]), 0.0,
        np.array([1.0, 0.0, 0.0]), 0.0, 0.0,
    )

    for now in (0.0, 0.1, 0.2):
        pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=now)
        translated = control.update(cmd, pose, now, target_key=1)
    assert translated[1] > 0 and translated[2] == 0

    pose = rpf.Pose(0.2, 0.0, 0.0, yaw=math.radians(31.0), stamp=0.3)
    correcting = control.update(cmd, pose, 0.3, target_key=1)
    assert correcting[:2] == (0, 0)
    assert correcting[2] != 0
    assert correcting[3] == 0
    assert not control.aligned_for_translation
    assert control.phase == "turn"

    for now in (0.4, 0.5):
        pose = rpf.Pose(0.2, 0.0, 0.0, yaw=0.0, stamp=now)
        assert control.update(cmd, pose, now, target_key=1) == ZERO
    resumed = control.update(
        cmd,
        rpf.Pose(0.2, 0.0, 0.0, yaw=0.0, stamp=0.6),
        0.6,
        target_key=1,
    )
    assert resumed[1] > 0 and resumed[2] == 0
    assert control.phase == "translate"


def test_repeated_capture_cannot_confirm_yaw_alignment():
    cfg = rpf.ControlConfig(inspect_waypoints=())
    control = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command(
        "FOLLOW",
        np.array([1.0, 0.0, 0.0]),
        yaw_target=0.0,
        goal=np.array([1.0, 0.0, 0.0]),
        path_error=0.0,
        progress=0.0,
    )
    frozen = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=1.0)

    for now in (1.0, 1.05, 1.10, 1.15, 1.20):
        assert control.update(cmd, frozen, now, target_key=0) == ZERO

    assert not control.aligned_for_translation
    assert control.phase == "yaw_alignment_hold"


def test_short_horizontal_leg_still_requires_nose_alignment():
    cfg = rpf.ControlConfig(inspect_waypoints=())
    control = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command(
        "FOLLOW",
        np.array([0.1, 0.0, 0.0]),
        yaw_target=0.0,
        goal=np.array([0.0, 0.0, 0.1]),
        path_error=0.0,
        progress=0.0,
    )
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=1.0)

    pcmd = control.update(cmd, pose, 1.0, target_key=1)

    assert pcmd[:2] == (0, 0)
    assert pcmd[2] != 0
    assert control.phase == "turn"


def test_yaw_alignment_has_no_timeout():
    cfg = rpf.ControlConfig(inspect_waypoints=(), yaw_alignment_timeout_s=0.2)
    control = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command(
        "FOLLOW",
        np.array([0.0, 0.0, 1.0]),
        yaw_target=math.pi / 2.0,
        goal=np.array([0.0, 0.0, 1.0]),
        path_error=0.0,
        progress=0.0,
    )
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=1.0)

    assert control.update(cmd, pose, 1.0, target_key=1)[2] != 0
    assert control.update(cmd, pose, 121.0, target_key=1)[2] != 0
    assert control.phase == "turn"

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


# --- map frame: which way is up is a MEASUREMENT, not an assumption ----------

_URAI_ALIGN = (REPO_ROOT
               / "地圖檔/場域/urai/maps/edm_v1/T_align_gravity.json")


def _write_align(tmp_path, gravity, rotation=None):
    frame = rpf.MapFrame.from_gravity(gravity)
    R = rotation if rotation is not None else [
        frame.east.tolist(), frame.north.tolist(), frame.up.tolist()]
    path = tmp_path / "T_align_gravity.json"
    path.write_text(json.dumps({
        "schema": "sfm-align/v2", "R": R,
        "gravity_glomap": list(gravity),
    }), encoding="utf-8")
    return path



def _settle(ctrl, pose, now):
    """Step through the arrival confirmation window and return the last command.

    Waypoint arrival deliberately needs consecutive in-sphere frames, so a test
    that steps once is asserting against a target that has not advanced yet.
    """
    cmd = None
    for i in range(int(ctrl.cfg.waypoint_arrive_confirm_frames)):
        sample = rpf.Pose(
            pose.x, pose.y, pose.z, pose.yaw, stamp=float(pose.stamp) + i * 1e-3
        )
        cmd = ctrl.step(sample, now=now + i * 1e-3)
    return cmd


def test_legacy_map_frame_reproduces_the_old_hardcoded_axes():
    """The legacy default must be bit-compatible, or this becomes a silent change."""
    legacy = rpf.LEGACY_MAP_FRAME
    delta = np.array([0.3, -1.2, 0.8])

    assert legacy.heading(delta) == pytest.approx(math.atan2(delta[2], delta[0]))
    assert legacy.horizontal_distance(delta) == pytest.approx(
        math.hypot(delta[0], delta[2]))
    assert legacy.vertical(delta) == pytest.approx(-delta[1])

    yaw = 0.7
    forward, right, up = legacy.body_components(delta, yaw)
    assert forward == pytest.approx(delta[0] * math.cos(yaw) + delta[2] * math.sin(yaw))
    assert right == pytest.approx(delta[0] * math.sin(yaw) - delta[2] * math.cos(yaw))
    assert up == pytest.approx(-delta[1])

    point = [1.0, 2.0, 3.0]
    assert np.allclose(rpf.aligned_to_glomap(point, legacy),
                       [point[0], -point[2], point[1]])


def test_map_frame_is_orthonormal_and_right_handed():
    for frame in (rpf.LEGACY_MAP_FRAME,
                  rpf.MapFrame.from_gravity([0.009, 0.924, 0.383])):
        for axis in (frame.east, frame.north, frame.up):
            assert float(np.linalg.norm(axis)) == pytest.approx(1.0)
        assert np.allclose(np.cross(frame.east, frame.north), frame.up, atol=1e-9)


@pytest.mark.skipif(not _URAI_ALIGN.is_file(), reason="site alignment not present")
def test_measured_frame_matches_the_sites_recorded_rotation():
    frame = rpf.load_map_frame(_URAI_ALIGN)
    R = np.asarray(json.loads(_URAI_ALIGN.read_text(encoding="utf-8"))["R"], float)

    assert np.allclose(frame.east, R[0], atol=1e-6)
    assert np.allclose(frame.north, R[1], atol=1e-6)
    assert np.allclose(frame.up, R[2], atol=1e-6)
    # The whole point: it is NOT the legacy guess.
    tilt = math.degrees(math.acos(float(np.dot(frame.up, rpf.LEGACY_MAP_FRAME.up))))
    assert tilt == pytest.approx(22.51, abs=0.05)


def test_measured_frame_changes_the_commanded_body_axes():
    """A 22.5 deg error leaks commanded climb into horizontal motion."""
    frame = rpf.MapFrame.from_gravity([0.009067509372034937,
                                       0.9237964066045453,
                                       0.38277667042064856])
    climb = -frame.up * 1.0          # a purely vertical move in the MEASURED frame

    legacy_f, legacy_r, legacy_u = rpf.LEGACY_MAP_FRAME.body_components(climb, 0.0)
    measured_f, measured_r, measured_u = frame.body_components(climb, 0.0)

    assert measured_u == pytest.approx(-1.0)
    assert abs(measured_f) < 1e-9 and abs(measured_r) < 1e-9
    # Under the legacy assumption the same motion reads as partly horizontal.
    assert math.hypot(legacy_f, legacy_r) > 0.38
    assert abs(legacy_u) < 0.93


def test_measured_route_round_trip_is_exact_and_legacy_inverse_is_not(tmp_path):
    gravity = [0.009067509372034937, 0.9237964066045453, 0.38277667042064856]
    frame = rpf.load_map_frame(_write_align(tmp_path, gravity))
    point = np.array([0.3, -1.2, 0.8])
    aligned = [float(np.dot(point, frame.east)),
               float(np.dot(point, frame.north)),
               float(np.dot(point, frame.up))]

    assert np.allclose(rpf.aligned_to_glomap(aligned, frame), point, atol=1e-12)
    # The bug this replaced: authoring with the measured basis, reading with legacy.
    wrong = rpf.aligned_to_glomap(aligned, rpf.LEGACY_MAP_FRAME)
    assert float(np.linalg.norm(wrong - point)) > 0.4


def test_map_alignment_whose_R_disagrees_with_its_gravity_is_refused(tmp_path):
    # A proper rotation (identity) whose Z row is NOT the recorded gravity's up.
    bad = _write_align(tmp_path, [0.0, 1.0, 0.0],
                       rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="disagrees with gravity_glomap"):
        rpf.load_map_frame(bad)


def test_map_alignment_with_a_non_rotation_matrix_is_refused(tmp_path):
    path = tmp_path / "T_align_gravity.json"
    path.write_text(json.dumps({
        "schema": "sfm-align/v2",
        "R": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        "gravity_glomap": [0.0, 1.0, 0.0],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="proper rotation"):
        rpf.load_map_frame(path)


def test_aligned_route_needs_an_align_source_when_the_site_measured_gravity(tmp_path):
    """'aligned' does not say WHICH alignment; guessing costs 22.5 deg."""
    frame = rpf.load_map_frame(_write_align(
        tmp_path, [0.009067509372034937, 0.9237964066045453, 0.38277667042064856]))
    route = tmp_path / "route.json"

    def write(**extra):
        route.write_text(json.dumps({
            "frame": "aligned", "units": "map",
            "waypoints": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            **extra,
        }), encoding="utf-8")

    write()
    with pytest.raises(ValueError, match="align_source"):
        rpf.load_waypoints(route, map_frame=frame)

    write(align_source="nonsense")
    with pytest.raises(ValueError, match="unknown route align_source"):
        rpf.load_waypoints(route, map_frame=frame)


def test_align_source_selects_the_basis_that_authored_the_route(tmp_path):
    """It says which basis PRODUCED the file, not which basis the site uses.

    A route drawn before the site's gravity was measured is legacy-aligned and
    inverts correctly with the legacy basis. Refusing it because the site has since
    been measured made every pre-measurement route unloadable -- which is exactly
    what happened to river_site's existing route.
    """
    frame = rpf.load_map_frame(_write_align(
        tmp_path, [0.009067509372034937, 0.9237964066045453, 0.38277667042064856]))
    route = tmp_path / "route.json"
    aligned = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]

    def write(source):
        route.write_text(json.dumps({
            "frame": "aligned", "units": "map", "align_source": source,
            "waypoints": aligned,
        }), encoding="utf-8")

    write("legacy")
    legacy_points = rpf.load_waypoints(route, map_frame=frame)
    assert np.allclose(legacy_points[1], [1.0, -3.0, 2.0]), "not the legacy inverse"

    write("measured")
    measured_points = rpf.load_waypoints(route, map_frame=frame)
    expected = frame.east * 1.0 + frame.north * 2.0 + frame.up * 3.0
    assert np.allclose(measured_points[1], expected, atol=1e-12)

    # The two really are different worlds; picking the wrong one moves the route.
    assert float(np.linalg.norm(measured_points[1] - legacy_points[1])) > 0.4


def test_measured_route_is_refused_when_the_site_has_no_measurement(tmp_path):
    route = tmp_path / "route.json"
    route.write_text(json.dumps({
        "frame": "aligned", "units": "map", "align_source": "measured",
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="no measured gravity alignment"):
        rpf.load_waypoints(route)


def test_a_legacy_route_loads_on_a_site_with_no_measurement(tmp_path):
    """Sites without T_align_gravity.json must behave exactly as before."""
    route = tmp_path / "route.json"
    route.write_text(json.dumps({
        "frame": "aligned", "units": "map",
        "waypoints": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
    }), encoding="utf-8")

    points = rpf.load_waypoints(route)

    assert np.allclose(points[1], [1.0, -3.0, 2.0])


def test_control_config_no_longer_exposes_a_dead_speed_knob():
    """`speed` scaled only vel_map, which the PCMD adapter ignores."""
    assert not hasattr(rpf.ControlConfig(), "speed")
    with pytest.raises(TypeError):
        rpf.ControlConfig(speed=1.2)


_MEASURED_GRAVITY = [0.009067509372034937, 0.9237964066045453, 0.38277667042064856]


def test_pcmd_adapter_decomposes_through_the_configured_map_frame():
    """Without this the adapter silently keeps assuming X/Z horizontal, -Y up."""
    frame = rpf.MapFrame.from_gravity(_MEASURED_GRAVITY)
    pose = rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0)
    # 3.0 map units: far enough that the integer PCMD rounding cannot hide the
    # difference between the two frames.
    goal = frame.up * 3.0                       # straight up in the MEASURED frame
    cmd = rpf.Command("FOLLOW", np.zeros(3), 0.0, goal, 0.0, 0.0)

    roll, pitch, _yaw, gaz = rpf.command_to_body_percent(
        cmd, pose, config=rpf.ControlConfig(map_frame=frame),
        require_yaw_alignment=False)
    assert (roll, pitch) == (0, 0) and gaz > 0

    legacy_roll, legacy_pitch, _y, legacy_gaz = rpf.command_to_body_percent(
        cmd, pose, config=rpf.ControlConfig(), require_yaw_alignment=False)
    assert (legacy_roll, legacy_pitch) != (0, 0), (
        "the legacy frame must read part of a pure climb as horizontal motion"
    )
    assert legacy_gaz < gaz, "the legacy frame must under-command the climb"


def test_route_heading_target_uses_the_configured_map_frame():
    frame = rpf.MapFrame.from_gravity(_MEASURED_GRAVITY)
    wp = [np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])]
    pose = rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0)

    measured = _settle(rpf.RouteAutoController(
        wp, poles=[], config=rpf.ControlConfig(inspect_waypoints=(), map_frame=frame)
    ), pose, 1.0)
    legacy = _settle(rpf.RouteAutoController(
        wp, poles=[], config=rpf.ControlConfig(inspect_waypoints=())
    ), pose, 1.0)

    assert measured.yaw_target != pytest.approx(legacy.yaw_target)


def test_heading_estimator_anchors_to_visual_map_heading_not_route_motion():
    frame = rpf.MapFrame.from_gravity(_MEASURED_GRAVITY)
    estimator = pff.HeadingEstimator(frame)
    visual_heading = math.radians(37.0)

    estimator.update(visual_heading, math.radians(-80.0))

    assert estimator.heading(math.radians(-80.0)) == pytest.approx(
        visual_heading, abs=1e-9
    )


def test_heading_estimator_converts_clockwise_ned_yaw_before_fusion():
    estimator = pff.HeadingEstimator()
    estimator.update(0.0, math.pi / 2.0)  # East in clockwise-from-North NED.

    assert estimator.heading(math.pi / 2.0) == pytest.approx(0.0, abs=1e-9)
    assert estimator.heading(math.pi) == pytest.approx(-math.pi / 2.0, abs=1e-9)


def test_heading_estimator_rejects_visual_yaw_increment_that_disagrees_with_imu():
    estimator = pff.HeadingEstimator(max_increment_mismatch_rad=math.radians(10.0))
    assert estimator.update(0.0, math.pi / 2.0)
    assert estimator.update(0.10, math.pi / 2.0 - 0.10)

    assert not estimator.update(1.0, math.pi / 2.0 - 0.20)
    assert estimator.last_increment_mismatch_rad == pytest.approx(0.80)
    assert estimator.heading(math.pi / 2.0 - 0.20) == pytest.approx(0.20)


def test_heading_estimator_has_no_position_or_path_seed_api():
    estimator = pff.HeadingEstimator()

    assert not hasattr(estimator, "seed_from_path")
    assert not hasattr(estimator, "mark_teleport")


def test_one_stray_fix_inside_the_sphere_does_not_retire_a_waypoint():
    """Arrival needs consecutive frames: a single bad fix must not advance."""
    wp = [np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0]),
          np.array([8.0, 0.0, 0.0])]
    cfg = rpf.ControlConfig(inspect_waypoints=(), waypoint_arrive_confirm_frames=3)
    ctrl = rpf.RouteAutoController(wp, poles=[], config=cfg)
    ctrl.step(rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0), now=1.0)
    ctrl.step(rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.05), now=1.05)
    ctrl.step(rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.10), now=1.10)
    settled = ctrl.target_index

    # One fix inside waypoint 2's sphere, then back out again.
    ctrl.step(rpf.Pose(3.5, 0.0, 0.0, 0.0, stamp=1.15), now=1.15)
    ctrl.step(rpf.Pose(2.0, 0.0, 0.0, 0.0, stamp=1.20), now=1.20)

    assert ctrl.target_index == settled, "a single in-sphere fix retired a waypoint"


# --- audit 2026-08-06: safety threads must not stop quietly -----------------

def test_safety_monitor_thread_crash_latches_a_terminal_land():
    """It is the only safety-switch poller; a silent death disarms LAND/EMERGENCY."""
    sent = []
    monitor = pff.SafetyMonitor(send_pcmd=lambda *cmd: sent.append(tuple(cmd)))

    boom = RuntimeError("unexpected in the safety loop")
    monitor._run_loop = lambda: (_ for _ in ()).throw(boom)

    with pytest.raises(RuntimeError):
        monitor._run()

    assert monitor.terminal_action == "LAND"
    assert "safety monitor thread died" in (monitor.reason or "")
    assert monitor.terminated.is_set()
    assert "unexpected in the safety loop" in (monitor.thread_died_error or "")


def test_safety_monitor_run_delegates_to_a_guarded_loop():
    """Guards the wiring: the loop body must sit inside the crash handler."""
    source = inspect.getsource(pff.SafetyMonitor._run)
    assert "_run_loop()" in source
    assert "_latch_terminal" in source


def test_deadman_keeps_trying_after_a_failed_land():
    """A dropped link during land() used to end the deadman after one attempt.

    The previous version of this test used a stub whose land() returned a bool and
    never touched safety.stop, so it passed against the REAL land() -- which
    returned None and set safety.stop before even attempting the Landing. It
    therefore protected nothing. This one drives the real NudgePilot.land().
    """
    import manual_nudge_pilot as mnp

    pilot = mnp.NudgePilot.__new__(mnp.NudgePilot)
    pilot.safety = SimpleNamespace(
        stop=False, landed=False, airborne=True, pilot_sticks=False,
        last_beat=0.0, last_pcmd=None,
    )
    events = []
    pilot.log = SimpleNamespace(event=lambda name=None, **fields: events.append(name))
    pilot.dry_run = False
    pilot.drone = None
    pilot._pulse_token = 0
    pilot._send_lock = threading.RLock()
    pilot._sent = []
    pilot._set_piloting_source = lambda _source: None
    pilot._raw_pcmd = lambda *_cmd: None

    attempts = {"n": 0}
    real_land = mnp.NudgePilot.land

    def flaky_land(self, reason="land"):
        attempts["n"] += 1
        if attempts["n"] < 3:
            # Touchdown not confirmed: real land() must NOT stop the world here.
            with self._send_lock:
                self.safety.landed = False
                self.safety.airborne = True
            return False
        return real_land(self, reason)

    original_sleep = mnp.time.sleep
    mnp.time.sleep = lambda _s: None
    try:
        mnp.NudgePilot.land = flaky_land
        mnp.run_deadman(pilot)
    finally:
        mnp.NudgePilot.land = real_land
        mnp.time.sleep = original_sleep

    assert attempts["n"] >= 3, "the deadman gave up before touchdown was confirmed"
    assert pilot.safety.landed is True
    assert events.count("deadman_land_retry") >= 2


def test_land_reports_confirmed_touchdown_and_only_then_stops():
    """run_deadman's retry guard depends on this return value being real."""
    import manual_nudge_pilot as mnp

    assert inspect.signature(mnp.NudgePilot.land).return_annotation == "bool"
    source = inspect.getsource(mnp.NudgePilot.land)
    assert "return landed_ok" in source
    assert "if landed_ok:" in source, (
        "safety.stop is set again before touchdown is confirmed"
    )

def test_no_developer_specific_site_probe_selects_flight_assets():
    """A path probe used to swap the bundle to another site behind the operator.

    The live launcher refuses to start without an explicit --site-profile because
    field assets must never fall back to a different site's map/route/bundle; this
    module must not reintroduce that fallback by probing the filesystem.
    """
    source = Path(pff.__file__).read_text(encoding="utf-8")
    for marker in ("FOOTBALL_FIELD_ROOT", "FOOTBALL_FIELD_BUNDLE",
                   "FOOTBALL_FIELD_MEGALOC"):
        assert marker not in source, f"{marker} reintroduces a per-developer site probe"
    assert not hasattr(pff, "FOOTBALL_FIELD_BUNDLE")
    # Defaults must resolve inside the workspace, not to whatever site directory
    # happens to exist on the machine running the flight.
    workspace = str(pff.LOC_ROOT.parent)
    for name in ("DEFAULT_BUNDLE", "DEFAULT_MEGALOC_CACHE"):
        resolved = str(getattr(pff, name))
        assert resolved.startswith(workspace), f"{name} points outside the workspace: {resolved}"


def test_pcmd_capture_is_bounded_in_real_flight():
    """It appends before the dry-run early return, so a long session grows it.

    Also guards the constant's scope: it was briefly defined inside the module
    docstring, which no test noticed because none of them sends 4097 commands.
    """
    import manual_nudge_pilot as mnp

    assert isinstance(mnp._SENT_CAPTURE_MAX, int) and mnp._SENT_CAPTURE_MAX > 0

    pilot = mnp.NudgePilot.__new__(mnp.NudgePilot)
    pilot._sent = []
    pilot.dry_run = True
    pilot.drone = None
    pilot.safety = SimpleNamespace(last_pcmd=None)

    for index in range(mnp._SENT_CAPTURE_MAX + 50):
        mnp.NudgePilot._raw_pcmd(pilot, index % 7, 0, 0, 0)

    assert len(pilot._sent) == mnp._SENT_CAPTURE_MAX
    # The tail is what every caller reads; it must be the most recent sends.
    assert pilot._sent[-1] == ((mnp._SENT_CAPTURE_MAX + 49) % 7, 0, 0, 0)


def test_heading_seed_survives_a_smaller_arrival_radius():
    """seed_goal must use the sequencer's reached test, not a bare radius check.

    With a radius smaller than the leg length, a bare check stopped skipping the
    waypoint underfoot and the seeded heading pointed backwards along the route.
    """
    wp = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
          np.array([3.0, 0.0, 0.0])]
    pos = np.array([1.0, 0.0, 0.0])

    for radius in (0.3, 1.0):
        ctrl = rpf.RouteAutoController(
            wp, poles=[],
            config=rpf.ControlConfig(inspect_waypoints=(),
                                     waypoint_arrive_radius=radius))
        goal = ctrl.seed_goal(pos)
        assert goal[0] > pos[0], (
            f"radius={radius}: seeded toward {goal.tolist()}, which is behind the drone"
        )


def test_route_start_targets_the_waypoint_after_nearest_takeoff_point():
    wp = [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
        np.array([15.0, 0.0, 0.0]),
    ]
    ctrl = rpf.RouteAutoController(
        wp,
        poles=[],
        config=rpf.ControlConfig(inspect_waypoints=()),
    )

    nearest = ctrl.start_after_nearest_waypoint(np.array([9.8, 0.0, 0.0]))

    assert nearest == 2
    assert ctrl.target_index == 3
    np.testing.assert_allclose(ctrl.current_target(np.array([9.8, 0.0, 0.0])), wp[3])


def test_route_start_at_final_waypoint_does_not_wrap_to_route_start():
    wp = [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
    ]
    ctrl = rpf.RouteAutoController(
        wp,
        poles=[],
        config=rpf.ControlConfig(inspect_waypoints=()),
    )

    nearest = ctrl.start_after_nearest_waypoint(np.array([10.0, 0.0, 0.0]))

    assert nearest == 2
    assert ctrl.target_index == 2


def test_raw_map_route_runs_nearest_next_turn_translate_then_repeats():
    waypoints = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
    ]
    cfg = rpf.ControlConfig(inspect_waypoints=())
    route = rpf.RouteAutoController(waypoints, poles=[], config=cfg)
    pcmd = rpf.YawAlignedPcmdController(cfg)
    assert route.start_after_nearest_waypoint(np.array([0.01, 0.0, 0.0])) == 0
    assert route.target_index == 1

    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=math.pi / 2.0, stamp=0.0)
    command = route.step(pose, now=0.0)
    assert pcmd.update(command, pose, 0.0, target_key=route.target_index)[2] != 0

    for now in (0.1, 0.2, 0.3, 0.4):
        pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=now)
        command = route.step(pose, now=now)
        output = pcmd.update(command, pose, now, target_key=route.target_index)
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.0, stamp=0.5)
    command = route.step(pose, now=0.5)
    output = pcmd.update(command, pose, 0.5, target_key=route.target_index)
    assert output[1] > 0 and output[2] == 0

    for now in (0.6, 0.7, 0.8):
        pose = rpf.Pose(1.0, 0.0, 0.0, yaw=0.0, stamp=now)
        command = route.step(pose, now=now)
    assert route.target_index == 2
    assert pcmd.update(command, pose, 0.8, target_key=route.target_index)[2] != 0

    for now in (0.9, 1.0, 1.1, 1.2):
        pose = rpf.Pose(1.0, 0.0, 0.0, yaw=math.pi / 2.0, stamp=now)
        command = route.step(pose, now=now)
        output = pcmd.update(command, pose, now, target_key=route.target_index)
    pose = rpf.Pose(1.0, 0.0, 0.0, yaw=math.pi / 2.0, stamp=1.3)
    command = route.step(pose, now=1.3)
    output = pcmd.update(command, pose, 1.3, target_key=route.target_index)
    assert output[1] > 0 and output[2] == 0

    for now in (1.4, 1.7, 2.0, 2.3, 2.6):
        pose = rpf.Pose(1.0, 0.0, 1.0, yaw=math.pi / 2.0, stamp=now)
        command = route.step(pose, now=now)
    assert command.action == "LAND"
    assert command.should_land


# --- physical stick override during autonomous flight -----------------------

def test_stick_override_hovers_then_suspends_autonomy():
    """The pilot taking the aircraft back must beat anything autonomy is doing."""
    thrown = {"n": 0}
    manual = {"n": 0}

    def sticks():
        thrown["n"] += 1
        return thrown["n"] >= 3          # idle for two ticks, then the pilot moves

    def request_manual():
        manual["n"] += 1
        return True

    sent, reason, records = run_ticks(
        12,
        lambda st: fresh_pose(st, x=min(7.99, 0.5 * st["tick"])),
        hooks_extra={"stick_active": sticks, "request_manual": request_manual},
    )

    assert manual["n"] >= 1, "autonomy never handed control to the pilot"
    override = [r for r in records if r.get("stick_override")]
    assert override, "no tick recorded the override"
    first = records.index(override[0])
    assert override[0]["pcmd"] == list(ZERO), "the override tick must command zero"
    # And nothing non-zero may follow it: the loop is silent once the pilot has it.
    assert all(
        record.get("pcmd") in (None, list(ZERO)) for record in records[first:]
    ), "autonomy kept commanding after the pilot took over"


def test_stick_override_without_a_manual_pilot_lands():
    """MANUAL downgrades to HOVER with no SkyController; autonomy must not continue."""
    sent, reason, _ = run_ticks(
        12,
        lambda st: fresh_pose(st, x=min(7.99, 0.5 * st["tick"])),
        hooks_extra={"stick_active": lambda: True,
                     "request_manual": lambda: False},
    )

    assert reason == "stick override without a manual pilot -> land"
    assert sent[-1] == ZERO


def test_a_stick_monitor_that_raises_is_treated_as_an_override():
    """A monitor that cannot answer is not evidence that the pilot is idle."""
    def broken():
        raise RuntimeError("HID read failed")

    sent, reason, records = run_ticks(
        8,
        lambda st: fresh_pose(st, x=0.5 * st["tick"]),
        hooks_extra={"stick_active": broken, "request_manual": lambda: True},
    )

    assert any(record.get("stick_override") for record in records)
    assert all(record.get("pcmd") in (None, list(ZERO)) for record in records)


# --- the sequencer must be able to finish a route ---------------------------

def _fly_route(config, waypoints, max_ticks=600):
    """Crude closed loop: pitch percent -> forward motion. Returns the terminal."""
    controller = rpf.RouteAutoController(waypoints, poles=[], config=config)
    x, now = 0.0, 1.0
    for _ in range(max_ticks):
        pose = rpf.Pose(x, 0.0, 0.0, 0.0, stamp=now)
        command = controller.step(pose, now=now)
        if command.should_land:
            return "LAND", x, controller.target_index
        pcmd = (0, 0, 0, 0)
        if command.action in {"FOLLOW", "REJOIN"}:
            pcmd = rpf.command_to_body_percent(
                command, pose, config=config, require_yaw_alignment=False)
        x += pcmd[1] * 0.02
        now += 0.05
    return "STALLED", x, controller.target_index


def test_autonomy_can_actually_finish_a_route():
    """The PCMD dead zone must be inside the arrival sphere, or nothing arrives.

    With translation_arrival_tolerance at 0.15 against a 0.02 sphere the drone
    stopped 0.15 short of waypoint 2, was never "reached", and the route hovered
    there for the rest of the flight.
    """
    waypoints = [np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0]),
                 np.array([8.0, 0.0, 0.0])]
    terminal, x, index = _fly_route(rpf.ControlConfig(inspect_waypoints=()), waypoints)

    assert terminal == "LAND", f"route stalled at x={x:.2f} on waypoint index {index}"
    assert index == len(waypoints) - 1
    assert x > 7.5, f"landed far short of the final waypoint: {x:.2f}"


def test_config_refuses_a_dead_zone_wider_than_the_arrival_sphere():
    with pytest.raises(ValueError, match="translation_arrival_tolerance must be smaller"):
        rpf.ControlConfig(waypoint_arrive_radius=0.02,
                          translation_arrival_tolerance=0.15)


def test_config_refuses_an_end_of_route_window_inside_the_dead_zone():
    with pytest.raises(ValueError, match="never terminate the mission"):
        rpf.ControlConfig(waypoint_arrive_radius=0.5, arrive=0.05, min_arrive=0.05,
                          translation_arrival_tolerance=0.2)


# --- audit 2026-08-07: legacy paths and terminal safety callbacks -----------

def test_legacy_entrypoints_stay_locked_even_when_legacy_env_is_enabled(monkeypatch):
    """The retired takeoff paths must not become live via one environment flag."""
    monkeypatch.setenv("SFM_ALLOW_LEGACY_FLIGHT", "1")
    import autoflight as legacy_auto
    import cruise_geofence as legacy_cruise

    with pytest.raises(SystemExit, match="permanently locked"):
        legacy_auto.run(None, None, None, dry_run=False)
    with pytest.raises(SystemExit, match="permanently locked"):
        legacy_cruise.run(None, None, dry_run=False)
    with pytest.raises(SystemExit, match="permanently locked"):
        ofs.run_real(object(), None, None, None)

def test_canonical_frame_source_has_live_hard_lock_and_timestamp_safety():
    source = (FLIGHT_CONTROL_ROOT / "olympe_frame_source.py").read_text(
        encoding="utf-8"
    )
    assert "SFM_ALLOW_LEGACY_FLIGHT" not in source
    assert "TakeOff()" not in source
    assert ".wait()" not in source
    assert ".join()" not in source
    assert ".wait(timeout=0.5)" in source
    assert ".join(timeout=5.0)" in source
    assert "require_source_timestamps" in source
    assert "_source_timing_trusted_locked" in source


def test_frame_source_has_no_unreferenced_coded_decoder_helpers():
    dead_names = {
        "_init_pyav_decoder", "_avcc_to_annexb", "_decode_h264_payload",
        "_coded_payload_bytes", "_coded_avcc_cb", "_coded_bytestream_cb",
    }
    path = FLIGHT_CONTROL_ROOT / "olympe_frame_source.py"
    source = path.read_text(encoding="utf-8")
    assert "if skipped_stale" not in source
    tree = ast.parse(source, filename=str(path))
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (dead_names & definitions), f"dead helpers remain in {path}"
    assert not (dead_names & calls), f"dead helper calls remain in {path}"


def test_sphinx_smoke_is_simulator_only_and_uses_bounded_waits():
    source = Path(
        FLIGHT_CONTROL_ROOT / "sphinx_path_follow_smoke.py"
    ).read_text(encoding="utf-8")
    assert "require_sphinx_ip" in source
    assert "this arming path has no real-aircraft override" in source
    assert ".wait(_timeout=20)" in source


def test_pole_cruise_is_controller_only_not_a_live_arming_entrypoint():
    source = Path(FLIGHT_CONTROL_ROOT / "pole_cruise.py").read_text(
        encoding="utf-8"
    )
    assert "class PoleCruise" in source
    assert "TakeOff" not in source
    assert "olympe" not in source


def test_manual_takeoff_requires_a_confirmed_controller_handoff():
    import manual_nudge_pilot as mnp

    pilot = mnp.NudgePilot(
        dry_run=False, ip="192.168.53.1", controller="skycontroller3",
        pct=8, pulse_s=0.2, cmd_log=mnp.CommandLog(None),
    )
    pilot.drone = object()
    pilot._set_piloting_source = lambda _source: False
    pilot.approve_operator_takeoff()

    assert pilot.takeoff() is False
    assert not pilot.safety.airborne
    assert pilot._sent == []


def test_manual_source_wait_timeout_is_bounded():
    import manual_nudge_pilot as mnp

    class _HungExpectation:
        def wait(self):
            time.sleep(0.5)

    t0 = time.monotonic()
    assert not mnp._wait_result(_HungExpectation(), 0.05, "source handoff")
    assert time.monotonic() - t0 < 0.2


def test_blocking_land_callback_does_not_hold_safety_io_lock():
    started = threading.Event()

    class _Safety:
        mode = "LAND"

        @staticmethod
        def poll():
            return "LAND"

    def blocking_land():
        started.set()
        time.sleep(0.5)

    monitor = pff.SafetyMonitor(
        lambda *_: None, _Safety(), land_cb=blocking_land, timeout_s=0.05,
    ).start()
    try:
        assert started.wait(0.5)
        t0 = time.monotonic()
        allowed, _reason = monitor.arming_allowed(lambda: True, lambda: False)
        elapsed = time.monotonic() - t0
        assert not allowed
        assert elapsed < 0.2, "LAND callback blocked the safety I/O lock"
    finally:
        monitor.stop()


def test_safety_monitor_crash_attempts_bounded_land_callback():
    attempts = []
    monitor = pff.SafetyMonitor(
        lambda *_: None,
        land_cb=lambda: attempts.append(time.monotonic()),
        timeout_s=0.05,
    )
    boom = RuntimeError("injected safety loop crash")
    monitor._run_loop = lambda: (_ for _ in ()).throw(boom)

    with pytest.raises(RuntimeError, match="injected safety loop crash"):
        monitor._run()

    assert attempts, "thread crash only latched LAND without trying the callback"
    assert monitor.terminal_action == "LAND"
