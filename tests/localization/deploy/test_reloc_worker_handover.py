"""The relocalizer handover: what the background worker is actually given.

Retrieval is given the *grey* frame, which the provider replicates to three
channels.  The frozen BoQ bank was built from colour keyframes, so that is a
real distribution mismatch, and the fast loop is holding the colour frame at
the moment it submits -- but handing it over was measured on the P173 holdout
and did not improve anything (see docs/direct_backend_ledger.md).  This file
pins the grey-only contract so the mismatch is not "fixed" again without a
gate, together with the one-job capacity rule.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

# Source modules are supplied by the repository's pytest pythonpath.
import numpy as np
import pytest

pytestmark = pytest.mark.smoke

from live_provider import RelocFix
from two_rate_tracker import RelocWorker, TwoRateTracker


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

    assert worker.submit(gray, 17) is True
    _, ordinal = drain(worker)

    assert ordinal == 17
    seen_gray, extra = provider.calls[0]
    np.testing.assert_array_equal(seen_gray, gray)
    assert extra == {}


def test_submit_refuses_while_the_worker_is_busy(worker_factory) -> None:
    """Capacity stays exactly one job: a queued reloc would land already stale."""

    release = threading.Event()
    provider = RecordingProvider(block=release)
    worker = worker_factory(provider)
    gray = np.zeros((6, 8), dtype=np.uint8)

    assert worker.submit(gray, 0) is True
    assert provider.entered.wait(timeout=5.0)
    assert worker.busy is True
    assert worker.submit(gray, 1) is False

    release.set()
    drain(worker)
    assert len(provider.calls) == 1


def test_a_closed_worker_refuses_work() -> None:
    provider = RecordingProvider()
    worker = RelocWorker(provider)
    worker.start()
    worker.close()

    assert worker.submit(np.zeros((6, 8), dtype=np.uint8), 0) is False
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
        submit=lambda gray, ordinal: submitted.append(ordinal) or True,
    )
    triggered = tracker._step_reloc_trigger(np.zeros((8, 8), np.uint8), 2, 10.1, status)
    assert triggered is (occupied is None)
    assert submitted == ([2] if occupied is None else [])


@pytest.mark.parametrize("status", ["FAST_TRACK", "RELOC_SEED"])
def test_map_confirmed_tracking_keeps_the_normal_relocalization_period(status):
    tracker = object.__new__(TwoRateTracker)
    tracker._reset_tracking_state()
    tracker.profile = SimpleNamespace(reloc=SimpleNamespace(period_s=1.0, min_points=12))
    tracker._live_ids = np.arange(200)
    tracker._last_reloc_stamp = 10.0
    submitted = []
    tracker._worker = SimpleNamespace(
        busy=False, submit=lambda gray, ordinal: submitted.append(ordinal) or True,
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
    tracker._worker = SimpleNamespace(poll=lambda: (fix, 0), busy=False, submit=lambda *_a: False)
    tracker._step_reloc_trigger = lambda *_a, **_k: False

    def fake_absolute(reseeded, stamp):
        n = int(len(tracker._live_ids))
        center = (
            np.mean(tracker._live_xyz, axis=0) if n else np.zeros(3, dtype=float)
        )
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


def test_strong_reloc_handover_replaces_and_skips_klt_on_new_ids() -> None:
    n = 90
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
