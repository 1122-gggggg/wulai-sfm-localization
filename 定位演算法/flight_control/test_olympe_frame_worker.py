#!/usr/bin/env python3
"""Pure-Python lifecycle tests for the capacity-one Pdraw frame worker."""
from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULES = [
    _load_module("mission_olympe_frame_source_worker_test",
                 "mission/flight_control/olympe_frame_source.py"),
    _load_module("deploy_olympe_frame_source_worker_test",
                 "deploy_code/sfm_glomap_deploy/olympe_frame_source.py"),
]


class _Streaming:
    def stop(self, *_, **__):
        return True


class _Frame:
    def __init__(self, label: int):
        self.label = label
        self.ref_calls = 0
        self.unref_calls = 0
        self.info_calls = 0
        self.info_threads = []
        self.released = threading.Event()

    def ref(self):
        self.ref_calls += 1

    def unref(self):
        self.unref_calls += 1
        if self.unref_calls > self.ref_calls:
            raise AssertionError(f"frame {self.label} unref without matching ref")
        self.released.set()

    def info(self):
        self.info_calls += 1
        self.info_threads.append(threading.current_thread().name)
        return {"ntp_raw_timestamp": 1_000_000 + self.label * 33_333}


def _grabber(module):
    return module.OlympePdrawGrabber(
        SimpleNamespace(streaming=_Streaming()), resize=None, stale_s=5.0)


def _wait_until(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for frame worker")


def _assert_balanced(*frames):
    assert all(frame.ref_calls == 1 for frame in frames)
    assert all(frame.unref_calls == 1 for frame in frames)


@pytest.mark.parametrize("module", MODULES, ids=("mission", "deploy"))
def test_latest_frame_replaces_pending_without_blocking_callback(module):
    grabber = _grabber(module)
    entered = threading.Event()
    release = threading.Event()
    processed = []

    def convert(frame, *_):
        processed.append(frame.label)
        if frame.label == 1:
            entered.set()
            assert release.wait(2.0)
        return True

    grabber._convert_yuv_frame = convert
    grabber._start_frame_worker()
    frames = [_Frame(i) for i in (1, 2, 3)]
    try:
        grabber._yuv_cb(frames[0])
        assert entered.wait(2.0)
        grabber._yuv_cb(frames[1])
        grabber._yuv_cb(frames[2])
        assert frames[1].released.wait(0.5), "replaced frame must be unref'd immediately"
        release.set()
        _wait_until(lambda: grabber.frame_pipeline_stats["converted"] == 2)
    finally:
        release.set()
        grabber.stop()

    assert processed == [1, 3]
    _assert_balanced(*frames)
    stats = grabber.frame_pipeline_stats
    assert stats["received"] == 3
    assert stats["enqueued"] == 3
    assert stats["queue_drops"] == 1
    assert stats["pending"] == stats["inflight"] == 0
    assert stats["convert_ms_mean"] >= 0.0
    assert [frame.info_calls for frame in frames] == [1, 0, 1]
    assert all(name == "olympe-yuv-latest"
               for frame in (frames[0], frames[2]) for name in frame.info_threads)


@pytest.mark.parametrize("module", MODULES, ids=("mission", "deploy"))
def test_pdraw_callback_does_not_read_metadata_or_convert(module):
    grabber = _grabber(module)
    grabber._accept_frames = True
    frame = _Frame(7)

    grabber._yuv_cb(frame)

    assert frame.info_calls == 0
    assert grabber._pending_yuv is not None
    assert grabber._pending_yuv[0] is frame
    assert len(grabber._pending_yuv) == 2
    grabber._pause_frame_input()
    _assert_balanced(frame)


@pytest.mark.parametrize("module", MODULES, ids=("mission", "deploy"))
def test_blocked_metadata_worker_never_holds_the_callback_lock(module):
    grabber = _grabber(module)
    info_entered = threading.Event()
    release_info = threading.Event()

    class BlockingInfoFrame(_Frame):
        def info(self):
            self.info_calls += 1
            self.info_threads.append(threading.current_thread().name)
            info_entered.set()
            assert release_info.wait(2.0)
            return {"ntp_raw_timestamp": 2_000_000}

    first = BlockingInfoFrame(1)
    second = _Frame(2)
    grabber._convert_yuv_frame = lambda *_args: True
    grabber._start_frame_worker()
    try:
        grabber._yuv_cb(first)
        assert info_entered.wait(1.0)
        started = time.monotonic()
        grabber._yuv_cb(second)
        assert time.monotonic() - started < 0.1
    finally:
        release_info.set()
        grabber.stop()
    _assert_balanced(first, second)


@pytest.mark.parametrize("module", MODULES, ids=("mission", "deploy"))
def test_flush_releases_pending_and_waits_for_inflight_unref(module):
    grabber = _grabber(module)
    entered = threading.Event()
    release = threading.Event()
    processed = []

    def convert(frame, *_):
        processed.append(frame.label)
        if frame.label == 1:
            entered.set()
            assert release.wait(2.0)
        return True

    grabber._convert_yuv_frame = convert
    grabber._start_frame_worker()
    first, pending, during_flush, after = (_Frame(i) for i in (1, 2, 3, 4))
    flush_thread = None
    try:
        grabber._yuv_cb(first)
        assert entered.wait(2.0)
        grabber._yuv_cb(pending)
        flush_thread = threading.Thread(target=grabber._flush_cb)
        flush_thread.start()
        assert pending.released.wait(0.5)
        assert flush_thread.is_alive(), "flush must wait until the in-flight ref is released"
        grabber._yuv_cb(during_flush)
        assert during_flush.released.wait(0.5)
        release.set()
        flush_thread.join(2.0)
        assert not flush_thread.is_alive()
        grabber._yuv_cb(after)
        _wait_until(lambda: grabber.frame_pipeline_stats["converted"] == 2)
    finally:
        release.set()
        if flush_thread is not None:
            flush_thread.join(2.0)
        grabber.stop()

    assert processed == [1, 4]
    _assert_balanced(first, pending, during_flush, after)
    assert grabber.frame_pipeline_stats["flush_drops"] == 2


@pytest.mark.parametrize("module", MODULES, ids=("mission", "deploy"))
def test_stop_releases_pending_rejects_late_callback_and_joins_worker(module):
    grabber = _grabber(module)
    entered = threading.Event()
    release = threading.Event()
    processed = []

    def convert(frame, *_):
        processed.append(frame.label)
        entered.set()
        assert release.wait(2.0)
        return True

    grabber._convert_yuv_frame = convert
    grabber._start_frame_worker()
    first, pending, late = (_Frame(i) for i in (1, 2, 3))
    stop_thread = None
    try:
        grabber._yuv_cb(first)
        assert entered.wait(2.0)
        grabber._yuv_cb(pending)
        stop_thread = threading.Thread(target=grabber.stop)
        stop_thread.start()
        assert pending.released.wait(0.5)
        assert stop_thread.is_alive(), "stop must join the in-flight conversion"
        grabber._yuv_cb(late)
        assert late.released.wait(0.5)
        release.set()
        stop_thread.join(2.0)
        assert not stop_thread.is_alive()
    finally:
        release.set()
        if stop_thread is not None:
            stop_thread.join(2.0)
        grabber.stop()

    assert processed == [1]
    _assert_balanced(first, pending, late)
    stats = grabber.frame_pipeline_stats
    assert stats["stopped_drops"] == 2
    assert stats["pending"] == stats["inflight"] == 0
    assert grabber._frame_worker is None


@pytest.mark.parametrize("module", MODULES, ids=("mission", "deploy"))
def test_store_still_enforces_stale_and_frozen_contract(module):
    grabber = _grabber(module)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    grabber._store(frame, stamp=time.monotonic())
    assert grabber() is not None

    grabber._stamp = time.monotonic() - 10.0
    assert grabber() is None

    for _ in range(module.FROZEN_DUP_FRAMES + 1):
        grabber._store(frame, stamp=time.monotonic())
    assert grabber() is None
    assert not grabber.is_healthy()


@pytest.mark.parametrize("module", MODULES, ids=("mission", "deploy"))
def test_live_ui_sample_reuses_owned_rgb_and_carries_monotonic_nodes(module):
    grabber = _grabber(module)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    timing = {
        "frame_callback_enter_mono_ns": 10,
        "frame_preprocess_done_mono_ns": 20,
    }
    grabber._store(frame, stamp=time.monotonic(), timing=timing)

    live = grabber.latest_frame_with_timing()
    assert live is not None
    out, _stamp, out_timing = live
    assert out is frame
    assert out_timing["frame_callback_enter_mono_ns"] == 10
    assert out_timing["frame_store_mono_ns"] >= out_timing["frame_preprocess_done_mono_ns"]
    defensive, _ = grabber()
    assert defensive is not frame
