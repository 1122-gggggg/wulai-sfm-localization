"""The relocalizer handover: what the background worker is actually given.

Retrieval is given the *grey* frame, which the provider replicates to three
channels.  The frozen BoQ bank was built from colour keyframes, so that is a
real distribution mismatch, and the fast loop is holding the colour frame at
the moment it submits -- but handing it over was measured on the P173 holdout
and did not improve anything (see docs/direct_backend_ledger.md).  This file
pins the grey-only contract so the mismatch is not "fixed" again without a
gate, together with bounded latest-capture queueing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

# Source modules are supplied by the repository's pytest pythonpath.
import numpy as np
import pytest
import cv2

pytestmark = pytest.mark.smoke

from live_provider import RelocFix
from two_rate_tracker import KLTTracker, RelocResult, RelocWorker, TwoRateTracker


class RecordingProvider:
    """Captures what each relocalization was given, without touching a GPU."""

    def __init__(self, *, block: threading.Event | None = None) -> None:
        self.calls: list[tuple[np.ndarray, dict]] = []
        self.entered = threading.Event()
        self._block = block

    def localize_array(self, gray, **kwargs):
        self.entered.set()
        if self._block is not None:
            self._block.wait(timeout=5.0)
        self.calls.append((np.array(gray, copy=True), dict(kwargs)))
        return SimpleNamespace(ok=False)


def drain(worker: RelocWorker, *, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        delivered = worker.poll()
        if delivered is not None:
            return delivered
        time.sleep(0.005)
    raise AssertionError("relocalizer produced no result within the timeout")


@pytest.fixture
def worker_factory():
    started: list[RelocWorker] = []

    def make(provider) -> RelocWorker:
        worker = RelocWorker(provider)
        worker.start()
        started.append(worker)
        return worker

    yield make
    for worker in started:
        worker.close()


def test_the_relocalizer_is_given_the_grey_frame_only(worker_factory) -> None:
    """Grey-only is a measured decision, not an oversight: handing the colour
    frame over was tried on the P173 holdout and did not improve anything."""

    provider = RecordingProvider()
    worker = worker_factory(provider)
    gray = np.full((6, 8), 40, dtype=np.uint8)

    assert worker.submit(gray, 17, 2.5, 3) is True
    delivered = drain(worker)

    assert delivered.ordinal == 17
    assert delivered.capture_stamp == pytest.approx(2.5)
    assert delivered.source_epoch == 3
    seen_gray, extra = provider.calls[0]
    np.testing.assert_array_equal(seen_gray, gray)
    assert extra == {}


def test_busy_worker_keeps_only_the_latest_waiting_capture(worker_factory) -> None:
    release = threading.Event()
    provider = RecordingProvider(block=release)
    worker = worker_factory(provider)
    gray = np.zeros((6, 8), dtype=np.uint8)
    assert worker.submit(gray, 0, 0.0, 0)
    assert provider.entered.wait(timeout=5.0)
    for ordinal in range(1, 100):
        gray.fill(ordinal)
        assert worker.submit(gray, ordinal, float(ordinal), 0)
    assert worker.busy and worker.queued
    gray.fill(0)  # Caller reuse must not mutate the queued capture.
    release.set()
    delivered = drain(worker)
    if delivered.ordinal == 0:
        delivered = drain(worker)
    assert delivered.ordinal == 99
    assert delivered.capture_stamp == pytest.approx(99.0)
    assert delivered.source_epoch == 0
    assert len(provider.calls) == 2
    assert np.all(provider.calls[-1][0] == 99)


def test_reset_discards_an_inflight_result(worker_factory):
    release = threading.Event()
    provider = RecordingProvider(block=release)
    worker = worker_factory(provider)
    gray = np.zeros((6, 8), dtype=np.uint8)
    assert worker.submit(gray, 1, 1.0, 0)
    assert provider.entered.wait(timeout=5.0)
    worker.reset()
    assert worker.submit(gray, 2, 2.0, 1)
    release.set()
    delivered = drain(worker)
    assert delivered.ordinal == 2
    assert delivered.capture_stamp == pytest.approx(2.0)
    assert delivered.source_epoch == 1


def test_worker_drops_an_older_ready_result_but_keeps_capture_identity() -> None:
    worker = RelocWorker(RecordingProvider())
    fix = SimpleNamespace(ok=False)

    worker._publish(fix, 4, 10.0, 2, 0)
    worker._publish(fix, 5, 10.37, 2, 0)

    delivered = worker.poll()
    assert delivered is not None
    assert delivered.ordinal == 5
    assert delivered.capture_stamp == pytest.approx(10.37)
    assert delivered.source_epoch == 2
    assert worker.poll() is None


def test_a_closed_worker_refuses_work() -> None:
    provider = RecordingProvider()
    worker = RelocWorker(provider)
    worker.start()
    worker.close()

    assert worker.submit(np.zeros((6, 8), dtype=np.uint8), 0, 0.0, 0) is False
    assert provider.calls == []


@pytest.mark.parametrize("status", ["VO_ONLY", "DEAD_RECKON", "IMU_BRIDGE", "NO_POSE"])
@pytest.mark.parametrize("occupied", [None, "pending", "busy"])
def test_map_unconfirmed_pose_relocalizes_without_waiting_for_period(status, occupied):
    tracker = object.__new__(TwoRateTracker)
    tracker._reset_tracking_state()
    tracker.profile = SimpleNamespace(reloc=SimpleNamespace(period_s=1.0, min_points=12))
    # Many tracked points do not imply a valid map-constrained solution.
    tracker._live_ids = np.arange(200)
    tracker._last_reloc_stamp = 10.0
    tracker._pending_ordinal = 1 if occupied == "pending" else None
    submitted = []
    tracker._worker = SimpleNamespace(
        busy=occupied == "busy",
        queued=occupied == "pending",
        submit=lambda gray, ordinal, stamp, epoch: submitted.append(ordinal) or True,
    )
    triggered = tracker._step_reloc_trigger(np.zeros((8, 8), np.uint8), 2, 10.1, status)
    assert triggered
    assert submitted == [2]


@pytest.mark.parametrize("status", ["FAST_TRACK", "RELOC_SEED"])
def test_map_confirmed_tracking_keeps_the_normal_relocalization_period(status):
    tracker = object.__new__(TwoRateTracker)
    tracker._reset_tracking_state()
    tracker.profile = SimpleNamespace(reloc=SimpleNamespace(period_s=1.0, min_points=12))
    tracker._live_ids = np.arange(200)
    tracker._last_reloc_stamp = 10.0
    submitted = []
    tracker._worker = SimpleNamespace(
        busy=False,
        queued=False,
        submit=lambda gray, ordinal, stamp, epoch: submitted.append(ordinal) or True,
    )
    gray = np.zeros((8, 8), np.uint8)
    assert not tracker._step_reloc_trigger(gray, 2, 10.1, status)
    assert tracker._step_reloc_trigger(gray, 3, 11.0, status)
    assert submitted == [3]


class _StubKLT:
    def __init__(self) -> None:
        self.track_calls: list[int] = []
        self.pair_calls: list[int] = []
        self.set_frames = 0

    def set_frame(self, gray) -> None:
        self.set_frames += 1

    def reset(self) -> None:
        pass

    def track_pair(self, _prev, _cur, xy):
        self.pair_calls.append(len(xy))
        shifted = np.asarray(xy, dtype=float) + 7.0
        return shifted, np.ones(len(xy), dtype=bool)

    def track(self, _gray, xy):
        self.track_calls.append(len(xy))
        shifted = np.asarray(xy, dtype=float).copy()
        shifted[:, 0] += 10.0
        return shifted, np.ones(len(xy), dtype=bool)


def _reloc_fix(n: int, *, xy0: float = 20.0, xyz0: float = 0.0, id0: int = 1000) -> RelocFix:
    ids = np.arange(id0, id0 + n, dtype=np.int64)
    xy = np.column_stack((np.full(n, xy0), np.full(n, xy0) + np.arange(n) * 0.01))
    xyz = np.column_stack((np.full(n, xyz0), np.zeros(n), np.zeros(n)))
    return RelocFix(
        ok=True,
        status="OK",
        cam_from_world=np.column_stack((np.eye(3), [0.0, 0.0, 0.0])),
        query_xy=xy,
        point3d_ids=ids,
        point_xyz=xyz,
        inliers=n,
        inlier_ratio=1.0,
        reproj_p90=0.5,
        reference_names=("r0",),
        runtime_ms=12.0,
    )


def _handover_tracker(fix: RelocFix):
    tracker = object.__new__(TwoRateTracker)
    tracker._reset_tracking_state()
    tracker._ordinal = -1
    tracker.profile = SimpleNamespace(
        fast_loop=SimpleNamespace(
            pnp=SimpleNamespace(min_points=8, min_inliers=8),
            reseed_min_points=120,
            track_cap=400,
        ),
        reloc=SimpleNamespace(
            frozen_pnp_thresholds={"strong_inliers": 80},
            min_points=12,
            period_s=1.0,
        ),
        vo=SimpleNamespace(enabled=False, keyframe_stride=100),
        dead_reckon=SimpleNamespace(enabled=False, max_frames=0, max_age_s=10.0),
        optimizations=SimpleNamespace(spatial_selection=False),
    )
    tracker._w = 64
    tracker._h = 64
    tracker._tracker = _StubKLT()
    tracker._worker = SimpleNamespace(
        poll=lambda: RelocResult(fix=fix, ordinal=0, capture_stamp=1.0, source_epoch=0),
        busy=False,
        submit=lambda *_a: False,
    )
    tracker._step_reloc_trigger = lambda *_a, **_k: False

    def fake_absolute(reseeded, stamp):
        n = int(len(tracker._live_ids))
        center = np.mean(tracker._live_xyz, axis=0) if n else np.zeros(3, dtype=float)
        pose = np.column_stack((np.eye(3), -center))
        status = "RELOC_SEED" if reseeded else "FAST_TRACK"
        return status, pose, center, n, n, 0, n, {}

    tracker._step_absolute_pose = fake_absolute
    live_n = 200
    tracker._live_ids = np.arange(1, live_n + 1, dtype=np.int64)
    tracker._live_xy = np.column_stack(
        (np.full(live_n, 5.0), np.full(live_n, 5.0) + np.arange(live_n) * 0.01)
    )
    tracker._live_xyz = np.tile(np.array([100.0, 0.0, 0.0]), (live_n, 1))
    return tracker


def test_failed_handover_does_not_report_the_previous_capture_age():
    fix = replace(_reloc_fix(120), ok=False)
    tracker = _handover_tracker(fix)
    tracker._handover_capture_age_s = 0.4
    ids, _, _ = tracker._walk_handover(fix, 0)
    assert len(ids) == 0
    assert tracker._handover_capture_age_s is None


def test_strong_reloc_handover_replaces_and_skips_klt_on_new_ids() -> None:
    n = 120
    fix = _reloc_fix(n)
    expected_xy = np.asarray(fix.query_xy, dtype=float).copy()
    expected_ids = np.asarray(fix.point3d_ids, dtype=np.int64).copy()
    tracker = _handover_tracker(fix)
    gray = np.zeros((64, 64), dtype=np.uint8)

    info = tracker.step(gray, 1.0)

    assert info["status"] == "RELOC_SEED"
    assert info["status"] != "FAST_TRACK"
    np.testing.assert_array_equal(tracker._live_ids, expected_ids)
    np.testing.assert_allclose(tracker._live_xy, expected_xy)
    assert tracker._tracker.track_calls == []
    assert tracker._tracker.pair_calls == []
    assert np.mean(tracker._live_xyz[:, 0]) == pytest.approx(0.0)
    assert info["center"][0] == pytest.approx(0.0)


def test_handover_accepts_jittered_capture_times_and_reports_elapsed_age() -> None:
    fix = _reloc_fix(120)
    tracker = _handover_tracker(fix)
    tracker._ring.extend(
        (0, ordinal, stamp, np.full((64, 64), ordinal, np.uint8))
        for ordinal, stamp in ((0, 10.0), (1, 10.07), (2, 10.37))
    )

    ids, _xy, _xyz = tracker._walk_handover(
        fix,
        0,
        trigger_epoch=0,
        trigger_stamp=10.0,
        current_epoch=0,
        current_stamp=10.37,
        current_ordinal=2,
    )

    assert len(ids) == 120
    assert tracker._handover_capture_age_s == pytest.approx(0.37)


@pytest.mark.parametrize(
    "result",
    [
        RelocResult(fix=_reloc_fix(120), ordinal=0, capture_stamp=10.0, source_epoch=1),
        RelocResult(fix=_reloc_fix(120), ordinal=0, capture_stamp=9.9, source_epoch=0),
        RelocResult(fix=_reloc_fix(120), ordinal=0, capture_stamp=10.1, source_epoch=0),
    ],
)
def test_handover_rejects_reset_or_inconsistent_capture_identity(result) -> None:
    tracker = _handover_tracker(result.fix)
    tracker._ring.extend(
        (0, ordinal, stamp, np.full((64, 64), ordinal, np.uint8))
        for ordinal, stamp in ((0, 10.0), (1, 10.07), (2, 10.37))
    )
    tracker._worker = SimpleNamespace(poll=lambda: result)

    ids, _xy, _xyz, _status, _refs, _delivered = tracker._step_handover(
        np.zeros((64, 64), np.uint8), 2, 10.37
    )

    assert ids.size == 0
    assert tracker._handover_dropped == 1


@pytest.mark.parametrize("frames", [3, 4, 7])
@pytest.mark.parametrize("motion", ["healthy", "retry_better", "retry_worse"])
def test_handover_retries_only_weak_two_frame_hops(frames, motion):
    fix = _reloc_fix(120)
    tracker = _handover_tracker(fix)
    tracker._ring.extend((0, i, float(i), np.full((64, 64), i, np.uint8)) for i in range(frames))
    calls = []

    def track(previous, current, xy):
        a, b = int(previous[0, 0]), int(current[0, 0])
        calls.append((a, b))
        keep = np.ones(len(xy), dtype=bool)
        if (a, b) == (0, 2) and motion != "healthy":
            keep[80:] = False
        elif b - a == 1 and motion == "retry_worse":
            keep[40:] = False
        return xy + np.array([b - a, 0.0]), keep

    tracker._tracker = SimpleNamespace(track_pair=track)
    ids, xy, xyz = tracker._walk_handover(fix, 0)
    count = 120 if motion != "retry_worse" else (40 if frames == 4 else 80)
    np.testing.assert_array_equal(ids, fix.point3d_ids[:count])
    np.testing.assert_array_equal(xyz, fix.point_xyz[:count])
    np.testing.assert_allclose(xy, fix.query_xy[:count] + [frames - 1, 0])
    if motion == "healthy":
        assert tracker._handover_retry_hops == 0
        assert len(calls) == frames // 2
    else:
        assert calls[:3] == [(0, 2), (0, 1), (1, 2)]
        assert tracker._handover_retry_hops >= 1
    assert len(calls) <= 3 * (frames // 2)


def test_handover_recovers_real_klt_points_through_intermediate_exposures():
    rng = np.random.default_rng(73)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (540, 960), dtype=np.uint8), (3, 3), 0)
    xy = cv2.goodFeaturesToTrack(texture, 500, 0.01, 8).reshape(-1, 2).astype(float)
    chain = [
        cv2.warpAffine(texture, np.array([[1, 0, 16 * i], [0, 1, 0]], dtype=float), (960, 540))
        for i in range(5)
    ]
    fix = replace(_reloc_fix(len(xy)), query_xy=xy)
    tracker = _handover_tracker(fix)
    tracker._w, tracker._h = 960, 540
    tracker.profile.fast_loop.track_cap = 500
    tracker._tracker = KLTTracker(fb_max_px=1, win=15, levels=3)
    tracker._ring.extend((0, i, float(i), frame) for i, frame in enumerate(chain))
    old_xy = xy.copy()
    for a, b in ((0, 2), (2, 4)):
        fwd, keep = tracker._tracker.track_pair(chain[a], chain[b], old_xy)
        old_xy = fwd[keep & tracker._inside(fwd)]
    ids, tracked, _ = tracker._walk_handover(fix, 0)
    assert len(old_xy) < tracker.profile.fast_loop.reseed_min_points <= len(ids)
    expected = xy[ids - fix.point3d_ids[0]] + [64, 0]
    assert np.percentile(np.linalg.norm(tracked - expected, axis=1), 90) < 1.0


def test_weak_reloc_handover_merges_without_klting_new_ids() -> None:
    n = 70
    fix = _reloc_fix(n)
    expected_xy = np.asarray(fix.query_xy, dtype=float).copy()
    expected_ids = np.asarray(fix.point3d_ids, dtype=np.int64).copy()
    tracker = _handover_tracker(fix)
    old_xy = tracker._live_xy.copy()
    gray = np.zeros((64, 64), dtype=np.uint8)

    info = tracker.step(gray, 1.0)

    assert tracker._tracker.track_calls == [200]
    index = {int(i): k for k, i in enumerate(tracker._live_ids)}
    for pid, xy in zip(expected_ids, expected_xy):
        np.testing.assert_allclose(tracker._live_xy[index[int(pid)]], xy)
    for pid, xy in zip(np.arange(1, 201, dtype=np.int64), old_xy):
        got = tracker._live_xy[index[int(pid)]]
        np.testing.assert_allclose(got, xy + np.array([10.0, 0.0]))
    assert info["status"] == "FAST_TRACK"


@pytest.mark.parametrize("count", [80, 119, 120])
def test_empty_track_requires_configured_reseed_minimum(count):
    tracker = _handover_tracker(_reloc_fix(count))
    tracker._live_ids = np.zeros(0, np.int64)
    tracker._live_xy = np.zeros((0, 2))
    tracker._live_xyz = np.zeros((0, 3))
    info = tracker.step(np.zeros((64, 64), np.uint8), 1.0)
    assert info["klt_reseeded"] is (count >= 120)
    assert len(tracker._live_ids) == (count if count >= 120 else 0)


@pytest.mark.parametrize("count", [12, 59, 60, 79, 80, 120])
def test_fast_track_label_does_not_bypass_strong_map_confidence(count):
    from direct_localizer_adapter import _map_confidence_mode

    profile = SimpleNamespace(reloc=SimpleNamespace(frozen_pnp_thresholds={"strong_inliers": 80}))
    mode, weak = _map_confidence_mode("FAST_TRACK", {"map_inliers": count}, profile)
    assert (mode, weak) == (("TRACK", False) if count >= 80 else ("WEAK_TRACK", True))


def test_reloc_idle_detects_hover_only():
    from two_rate_tracker import _reloc_idle

    assert _reloc_idle([]) is False
    assert _reloc_idle([0.0001, 0.0003, 0.0002]) is True
    assert _reloc_idle([0.0001, 0.05, 0.0001, 0.0001, 0.0001, 0.0001]) is True
    assert _reloc_idle([0.0001, 0.05, 0.05, 0.05, 0.0001, 0.0001]) is False
    assert _reloc_idle([0.01] * 10) is False


def test_reloc_refresh_due_stretches_only_idle_strong():
    from two_rate_tracker import _reloc_refresh_due as due

    assert due("FAST_TRACK", 100.0, None, 1.0, False) is True
    assert due("FAST_TRACK", 102.0, 100.0, 1.0, False) is True
    assert due("FAST_TRACK", 100.5, 100.0, 1.0, False) is False
    # Idle strong hover: 1 s period stretches to 4 s.
    assert due("FAST_TRACK", 102.0, 100.0, 1.0, True) is False
    assert due("FAST_TRACK", 104.0, 100.0, 1.0, True) is True
    # Weak statuses keep the aggressive cadence even when still.
    assert due("VO_ONLY", 100.5, 100.0, 1.0, True) is False
    assert due("VO_ONLY", 101.0, 100.0, 1.0, True) is True
    assert due("NO_POSE", 100.1, 100.0, 1.0, True) is False
