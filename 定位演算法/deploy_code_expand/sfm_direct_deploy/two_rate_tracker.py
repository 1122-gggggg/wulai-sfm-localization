#!/usr/bin/env python3
"""Two-rate direct localizer: CPU fast loop + background GPU relocalizer.

Fast loop (every frame, must fit the stream interval): pyramidal KLT carries the
live map-point set and the un-triangulated VO candidates one frame forward, then
pycolmap RANSAC PnP solves the absolute pose. Pure CPU, no model, no GPU.

`RelocWorker` serializes query localization, durable session-KF indexing, and
frozen-map overlap matching on one background GPU thread. Query localization
uses `LiveMapEDMProvider.localize_array`; its anchor 2D points are chained
forward through each frame the worker is busy, then handed to the fast loop.
The fast loop NEVER waits: query captures coalesce, while overlap submission is
non-blocking and maintenance jobs receive service after at most two queries.

Derived from `定位演算法/validation/replay_two_rate_reference.py` (the frozen
P174 replay harness). Behavioural differences, all deliberate:

* Handover timing is real, not simulated. The replay estimated the deployed
  latency (`SCALE_5090_TO_5060`) and installed the anchor on a computed
  `ready_frame`; here the anchor is installed on the first frame after
  `RelocWorker.poll()` returns it. The catch-up chaining is unchanged: the grey
  frames from the trigger frame to the current one are kept in a ring buffer and
  the anchor xy is walked forward through them with `track_pair`.
* `VO_REENTRY_FIT` is NOT ported. It rewrites already-emitted trajectory rows
  after a relocalization lands, which is an offline post-process: a flight
  controller has already consumed those poses, so retroactively editing them is
  meaningless at best and unsafe at worst.
* `SYNTH_HOLE`, ground-truth comparison, `queries.jsonl` generation and the
  replay summary statistics are not ported; they are experiment scaffolding.
* `VO_REFINE` (structure-only local BA) and `VO_WINDOW_BA` are not ported: both
  are frozen off, measured slower with no accuracy gain (P174_AND_NEXT.md:96).
  A profile that switches them on raises instead of silently ignoring them.
* `TOPUP` is not ported for the same reason plus a structural one: its point
  pool came from `FinalMapEDMProvider._observations`, which is not part of the
  frozen provider contract. It is frozen off (seeding points at their
  pose-predicted pixel makes the next PnP agree with the pose it already had,
  which locks in drift), and a profile that switches it on raises.
* Every knob comes from the SHA-bound `DirectProfile`. No `os.environ` reads.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait as _wait_futures
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

from direct_paths import vendor_sys_path
from point_quality import pose_quality, spatial_indices

vendor_sys_path()

from river_map_quality.pose_information import scaled_pose_information  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import graph light
    from direct_map import DirectMapAssets
    from direct_profile import DirectProfile
    from live_provider import RelocFix


#: Longest catch-up chain a handover may walk. A 5060 relocalization takes
#: ~400-450 ms, i.e. ~13 frames at 29 fps; 48 frames of 960x540 grey is ~25 MB
#: and leaves a 3x margin. A worker slower than that produces an anchor whose
#: 2D points can no longer be trusted through the accumulated tracking, so the
#: handover is dropped rather than applied stale.
HANDOVER_MAX_FRAMES = 48

#: A consistent full-mode map refresh preserves continuous visual confirmation.
REFRESH_MAX_CENTER_SHIFT_U = 0.1
REFRESH_MAX_ROTATION_DEG = 5.0
#: Dead-reckoning essential-matrix RANSAC used to run to OpenCV's default 1000 iterations,
#: which stalled a frame for 40-150 ms whenever most tracks were outliers.
DR_ESSENTIAL_MAX_ITERS = int(os.environ.get("SFM_DR_MAX_ITERS", "300"))

#: Single-frame azimuth change above which a dead-reckoned step is halved.
DEAD_RECKON_MAX_TURN_DEG = 5.0

#: Inertial bridge (fused-attitude aiding) bounds. Level D only: the fused
#: sample carries Euler angles, not gyro/accel, so the bridge rotates the last
#: visual pose about the map up axis and holds its center -- it never invents
#: translation. attitude ~8 Hz, frames ~30 Hz: 0.5 s covers a few missed polls.
IMU_FUSED_HISTORY = 64
IMU_BRIDGE_MAX_SAMPLE_AGE_S = 0.5
#: Near-level flight only: the yaw axis is approximated by the map up axis.
#: Log p95 roll/pitch rates hit 27/19 deg/s with peaks past 150 (2026-09-12),
#: but sustained tilt past 35 deg means the approximation is dishonest.
IMU_BRIDGE_MAX_TILT_DEG = 35.0
#: Glitch gate: the airframe cannot sustain this yaw rate (MaxRotationSpeed
#: caps at 200 deg/s and AUTO flies 10); a bigger implied rate is a bad sample.
IMU_BRIDGE_MAX_YAW_RATE_DEG_S = 180.0
#: Share of the live old-map points that must stay within 4 px under the mixed pose for it to be
#: published without a frozen-only comparison (the baseline always publishes the mixed solve).
OLD_SUPPORT_SHARE = 0.8

#: Background relocalization while strongly tracked and nearly stationary.
#: A 5060 relocalization costs ~250 ms of GPU for an anchor refresh the fast
#: loop does not need every second while hovering: KLT holds the tracks and
#: the map has not moved. Stretch the refresh period only in this regime --
#: weak statuses keep the aggressive every-frame trigger below, so recovery
#: after a real loss is untouched.
RELOC_STRONG_IDLE_PERIOD_S = 4.0
#: Median per-frame center step below this counts as hovering (map units).
#: 0.002 u is ~3 cm/frame at 15 m/u, i.e. ~0.6 m/s drift ceiling -- generous
#: against a hovering aircraft, tight against real translation.
RELOC_IDLE_STEP_U = 0.002
RELOC_IDLE_WINDOW = 5


def _reloc_idle(step_history) -> bool:
    """Is the recent track history a hover? Pure; empty history is not idle."""
    try:
        recent = [float(v) for v in list(step_history)[-RELOC_IDLE_WINDOW:]]
    except (TypeError, ValueError, OverflowError):
        return False
    if not recent:
        return False
    return float(np.median(recent)) < RELOC_IDLE_STEP_U


def _reloc_refresh_due(status: str, stamp: float, last_stamp, period_s: float, idle: bool) -> bool:
    """Background-refresh cadence. Pure. Idle strong tracking refreshes slower."""
    period = RELOC_STRONG_IDLE_PERIOD_S if (status == "FAST_TRACK" and idle) else float(period_s)
    return last_stamp is None or (float(stamp) - float(last_stamp)) >= period


def azimuth_turn_exceeds(
    map_frame: Any,
    prev_rotation: np.ndarray,
    relative_rotation: np.ndarray,
    max_turn_deg: float = DEAD_RECKON_MAX_TURN_DEG,
) -> bool:
    """Did the camera azimuth swing more than `max_turn_deg` in this one step?

    Measured in the map's *measured* horizontal plane, which is what MapFrame
    carries. The raw `atan2(forward[1], forward[0])` of `_yaw_from_rotation` is
    not usable for this: on a site whose gravity runs along +Y -- the river map,
    see T_align_gravity.json -- that expression spans a vertical plane, so it
    folds gimbal pitch into the heading and barely responds to real yaw.

    Returns False without a MapFrame: there is then no defensible horizontal
    basis, and a guard firing on the wrong quantity is worse than no guard. Also
    False when either optical axis is near-vertical, where azimuth is undefined.
    """

    if map_frame is None:
        return False
    forward_before = np.asarray(prev_rotation, dtype=float)[2]
    forward_after = (np.asarray(relative_rotation, dtype=float) @ prev_rotation)[2]
    if (
        map_frame.horizontal_distance(forward_before) <= 1e-9
        or map_frame.horizontal_distance(forward_after) <= 1e-9
    ):
        return False
    turn = map_frame.heading(forward_after) - map_frame.heading(forward_before)
    turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
    return abs(turn) > math.radians(max_turn_deg)


def _detect_corners(gray: np.ndarray, occupied: np.ndarray, cap: int, min_distance: int) -> np.ndarray:
    """Shi-Tomasi corners outside discs of ``min_distance`` around the in-image ``occupied`` pixels."""
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    if occupied.size:
        pts = np.asarray(occupied, dtype=int)
        valid = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < gray.shape[1])
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < gray.shape[0])
        )
        for x, y in pts[valid]:
            cv2.circle(mask, (int(x), int(y)), min_distance, 0, -1)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=int(cap), qualityLevel=0.01, minDistance=min_distance, mask=mask
    )
    if corners is None:
        return np.zeros((0, 2), dtype=float)
    return corners.reshape(-1, 2).astype(float)


class _PendingDetection(NamedTuple):
    """A VO seed detection running behind the foreground, merged at the next frame."""

    future: Future
    pose: np.ndarray
    provenance: Any


class KLTTracker:
    """Pyramidal Lucas-Kanade with a forward-backward gate. Pure CPU.

    Ported verbatim from the replay harness except that it consumes grayscale
    images directly (the fast loop already needs the grey frame for VO feature
    detection and for the relocalizer, so converting once per frame is strictly
    cheaper) and reads its window/level/gate from the deployment profile.
    """

    kind = "cpu"
    name = "klt"

    def __init__(self, *, fb_max_px: float, win: int, levels: int) -> None:
        self.fb_max_px = float(fb_max_px)
        self.win = int(win)
        self.levels = int(levels)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        self._prev_gray: np.ndarray | None = None
        self.last_diagnostics: dict = {}

    def set_frame(self, gray: np.ndarray) -> None:
        self._prev_gray = gray
        self.last_diagnostics = {"klt_input_points": 0, "klt_kept_points": 0}

    def reset(self) -> None:
        self._prev_gray = None
        self.last_diagnostics = {}

    def track_pair(
        self, prev_gray: np.ndarray, cur_gray: np.ndarray, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._lk(prev_gray, cur_gray, xy)

    def track_guided(
        self, prev_gray: np.ndarray, cur_gray: np.ndarray, xy: np.ndarray, guess: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measure from the source image; prediction only initializes bidirectional LK."""
        rejected = np.zeros(len(xy), dtype=bool)
        fallback = guess.astype(np.float64).copy()
        if not len(xy):
            return fallback, rejected
        p0 = np.asarray(xy, dtype=np.float32).reshape(-1, 1, 2)
        initial = np.asarray(guess, dtype=np.float32).reshape(-1, 1, 2).copy()
        options = dict(
            winSize=(self.win, self.win),
            maxLevel=self.levels,
            criteria=self.criteria,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, initial, **options)
        if p1 is None or st is None:
            return fallback, rejected
        back, stb, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, p0.copy(), **options)
        if back is None or stb is None:
            return fallback, rejected
        fwd = p1.reshape(-1, 2)
        keep = (
            (st.ravel() == 1)
            & (stb.ravel() == 1)
            & np.isfinite(fwd).all(axis=1)
            & (np.linalg.norm(back.reshape(-1, 2) - xy, axis=1) <= self.fb_max_px)
        )
        return fwd, keep

    def track(self, gray: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        previous = self._prev_gray
        self._prev_gray = gray
        if previous is None:
            self.last_diagnostics = {"klt_input_points": len(xy), "klt_kept_points": 0}
            return xy, np.zeros(len(xy), dtype=bool)
        return self._lk(previous, gray, xy)

    # NOTE: pre-built pyramids (cv2.buildOpticalFlowPyramid) would remove three
    # of the four pyramid builds per frame, but the Python binding rejects a
    # pyramid list for `prevImg`, so the grey images are passed directly.
    def _lk(
        self, previous: np.ndarray, current: np.ndarray, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.last_diagnostics = {"klt_input_points": len(xy), "klt_kept_points": 0}
        if xy.size == 0:
            return xy, np.zeros(len(xy), dtype=bool)
        p0 = xy.astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            p0,
            None,
            winSize=(self.win, self.win),
            maxLevel=self.levels,
            criteria=self.criteria,
        )
        if p1 is None or st is None:
            return xy, np.zeros(len(xy), dtype=bool)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(
            current,
            previous,
            p1,
            None,
            winSize=(self.win, self.win),
            maxLevel=self.levels,
            criteria=self.criteria,
        )
        if p0b is None or stb is None:
            return xy, np.zeros(len(xy), dtype=bool)
        fwd = p1.reshape(-1, 2)
        back = p0b.reshape(-1, 2)
        fb = np.linalg.norm(back - xy, axis=1)
        keep = (
            (st.reshape(-1) == 1)
            & (stb.reshape(-1) == 1)
            & np.isfinite(fwd).all(axis=1)
            & (fb <= self.fb_max_px)
        )
        valid_fb = fb[(st.reshape(-1) == 1) & (stb.reshape(-1) == 1) & np.isfinite(fb)]
        self.last_diagnostics.update(
            klt_forward_valid=int(np.count_nonzero(st)),
            klt_backward_valid=int(np.count_nonzero(stb)),
            klt_kept_points=int(np.count_nonzero(keep)),
            klt_fb_p50_px=float(np.median(valid_fb)) if valid_fb.size else None,
            klt_fb_p95_px=float(np.percentile(valid_fb, 95)) if valid_fb.size else None,
        )
        displacement = np.linalg.norm(fwd[keep] - xy[keep], axis=1)
        self.last_diagnostics["klt_displacement_p50_px"] = (
            float(np.median(displacement)) if displacement.size else None
        )
        return fwd, keep


class RelocWorker:
    """One serialized GPU worker for queries, KF indexing, and map overlap.

    Query captures coalesce to the latest image. Index submitters may wait on
    this worker from the session writer; overlap submission is always
    non-blocking and has one pending slot. When maintenance work waits, at most
    two consecutive query calls may run before the oldest maintenance job.
    """

    def __init__(self, provider: Any, depth_estimator: Any = None) -> None:
        self._provider = provider
        self._depth_estimator = depth_estimator
        self._cv = threading.Condition()
        self._pending: tuple[str, Any, float] | None = None
        self._pending_index: tuple[Mapping[str, Any], Any, float] | None = None
        self._pending_overlap: tuple[Mapping[str, Any], Any, float] | None = None
        self._pending_depth: tuple[Mapping[str, Any], Any, float] | None = None
        self._depth_healthy = True
        self._depth_errors = 0
        self._depth_last_error: str | None = None
        self._busy = False
        self._busy_kind: str | None = None
        self._busy_started: float | None = None
        self._closed = False
        self._maintenance_closed = False
        self._out: "queue.Queue[RelocResult]" = queue.Queue(maxsize=1)
        self._generation = 0
        self._thread: threading.Thread | None = None
        self._queries_since_maintenance = 0
        self._index_waiters = 0
        self._index_wait_started: float | None = None
        self._index_healthy = True
        self._index_errors = 0
        self._index_last_error: str | None = None
        self._overlap_healthy = True
        self._overlap_errors = 0
        self._overlap_last_error: str | None = None
        # While the fast loop has no strong global anchor, frozen-map overlap
        # matching (2 x top-k EDM passes) waits so relocalization gets the GPU.
        self._defer_overlap = False

    # ---------- lifecycle ----------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._closed = False
        thread = threading.Thread(target=self._run, name="direct-reloc-worker", daemon=True)
        self._thread = thread
        thread.start()

    def close(self) -> None:
        thread = self._thread
        with self._cv:
            self._closed = True
            self._generation += 1
            # A pending capture has no durable side effect. Index and overlap
            # jobs drain so callers receive their completion callback.
            self._defer_overlap = False
            self._pending_depth = None
            if self._pending is not None and self._pending[0] == "query":
                self._pending = None
            self._cv.notify_all()
        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive():
                raise RuntimeError("relocalizer thread did not stop within 30 s")
        self._thread = None

    def abandon_maintenance(self) -> dict[str, int]:
        """Cancel unclaimed maintenance without disturbing a running GPU job."""
        with self._cv:
            self._maintenance_closed = True
            dropped = {
                "index": int(self._pending_index is not None),
                "overlap": int(self._pending_overlap is not None),
                "depth": int(self._pending_depth is not None),
            }
            self._pending_index = self._pending_overlap = self._pending_depth = None
            self._cv.notify_all()
            return dropped

    def running_kind(self) -> str | None:
        with self._cv:
            return self._busy_kind

    # ---------- fast-loop facing ----------
    @property
    def busy(self) -> bool:
        with self._cv:
            return (
                self._busy
                or self._pending is not None
                or self._pending_index is not None
                or self._pending_overlap is not None
                or self._pending_depth is not None
            )

    @property
    def queued(self) -> bool:
        with self._cv:
            return (
                self._pending is not None
                or self._pending_index is not None
                or self._pending_overlap is not None
                or self._pending_depth is not None
            )

    def _maintenance_waiting_locked(self) -> bool:
        return (
            self._index_waiters > 0
            or self._pending_index is not None
            or self._runnable_overlap_locked() is not None
            or self._pending_depth is not None
        )

    def _runnable_overlap_locked(self):
        return None if self._defer_overlap else self._pending_overlap

    def set_defer_overlap(self, defer: bool) -> None:
        """Hold pending overlap matching while relocalization is the priority."""
        with self._cv:
            if self._defer_overlap != bool(defer):
                self._defer_overlap = bool(defer)
                self._cv.notify_all()

    def submit(
        self,
        gray: np.ndarray,
        ordinal: int,
        capture_stamp: float,
        source_epoch: int,
    ) -> bool:
        with self._cv:
            if self._closed or self._thread is None:
                return False
            queries_ahead = self._queries_since_maintenance
            if self._busy and self._busy_kind == "query":
                queries_ahead += 1
            elif self._busy and self._busy_kind in {"index", "overlap", "depth"}:
                # The running maintenance job resets the consecutive-query
                # count before any queued query can be dispatched.
                queries_ahead = 0
            if self._maintenance_waiting_locked() and queries_ahead >= 2:
                return False
            self._pending = (
                "query",
                (gray.copy(), int(ordinal), int(source_epoch), float(capture_stamp)),
                time.monotonic(),
            )
            self._cv.notify()
        return True

    def submit_index(self, reference: Mapping[str, Any], callback) -> bool:
        """Queue one durable reference for indexing; called by the disk writer."""
        with self._cv:
            if self._closed or self._maintenance_closed or self._thread is None:
                return False
            self._index_waiters += 1
            if self._index_waiters == 1:
                self._index_wait_started = time.monotonic()
            registered = True
            try:
                while self._pending_index is not None and not self._closed and not self._maintenance_closed:
                    self._cv.wait()
                if self._closed or self._maintenance_closed:
                    return False
                self._pending_index = (reference, callback, time.monotonic())
                self._index_waiters -= 1
                registered = False
                if self._index_waiters == 0:
                    self._index_wait_started = None
                self._cv.notify_all()
                return True
            finally:
                if registered:
                    self._index_waiters -= 1
                    if self._index_waiters == 0:
                        self._index_wait_started = None
                    self._cv.notify_all()

    def submit_overlap(self, reference: Mapping[str, Any], callback) -> bool:
        """Queue at most one non-blocking overlap request on the GPU thread."""
        with self._cv:
            if self._closed or self._maintenance_closed or self._thread is None or self._pending_overlap is not None:
                return False
            self._pending_overlap = (reference, callback, time.monotonic())
            self._cv.notify()
            return True

    def submit_depth(self, request: Mapping[str, Any], callback) -> bool:
        """Coalesce pending depth without blocking the foreground or holding its callback lock."""
        with self._cv:
            if self._closed or self._maintenance_closed or self._thread is None or self._depth_estimator is None:
                return False
            old = self._pending_depth
            self._pending_depth = (request, callback, time.monotonic())
            self._cv.notify()
        if old is not None:
            old[1](old[0], None, "superseded")
        return True

    def poll(self) -> "RelocResult | None":
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def reset(self) -> None:
        """Invalidate query results; durable index and overlap work survives."""
        with self._cv:
            self._generation += 1
            self._pending_depth = None
            if self._pending is not None and self._pending[0] == "query":
                self._pending = None
            while True:
                try:
                    self._out.get_nowait()
                except queue.Empty:
                    break
            self._cv.notify_all()

    def _job_times_locked(self, kind: str, pending) -> list[float]:
        times = [] if pending is None else [pending[2]]
        if self._busy_kind == kind and self._busy_started is not None:
            times.append(self._busy_started)
        if kind == "index" and self._index_wait_started is not None:
            times.append(self._index_wait_started)
        return times

    def workmap_diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._cv:
            result = {}
            for kind, pending in (
                ("query", self._pending),
                ("index", self._pending_index),
                ("overlap", self._pending_overlap),
                ("depth", self._pending_depth),
            ):
                times = self._job_times_locked(kind, pending)
                prefix = "workmap_reloc" if kind == "query" else f"workmap_{kind}"
                result[f"{prefix}_pending"] = bool(times)
                result[f"{prefix}_oldest_age_ms"] = (
                    max(0.0, (now - min(times)) * 1000.0) if times else 0.0
                )
                if kind != "query":
                    result[f"{prefix}_healthy"] = bool(getattr(self, f"_{kind}_healthy"))
                    result[f"{prefix}_errors"] = int(getattr(self, f"_{kind}_errors"))
            return result

    # ---------- worker thread ----------
    def _oldest_maintenance_locked(self) -> tuple[str, Any] | None:
        jobs = [
            (kind, pending)
            for kind, pending in (
                ("index", self._pending_index),
                ("overlap", self._runnable_overlap_locked()),
                ("depth", self._pending_depth),
            )
            if pending is not None
        ]
        if not jobs:
            return None
        kind, pending = min(jobs, key=lambda item: item[1][2])
        setattr(self, f"_pending_{kind}", None)
        return kind, (pending[0], pending[1])

    def _take_next_locked(self) -> tuple[str, Any] | None:
        query = self._pending
        if query is not None and (
            not self._maintenance_waiting_locked() or self._queries_since_maintenance < 2
        ):
            self._pending = None
            return "query", query[1]
        return self._oldest_maintenance_locked()

    def _run(self) -> None:
        while True:
            claimed = self._claim_next_job()
            if claimed is None:
                return
            kind, payload, generation = claimed
            if kind == "query":
                self._run_query(payload, generation)
            elif kind == "index":
                self._run_index(payload)
            elif kind == "depth":
                self._run_depth(payload)
            else:
                self._run_overlap(payload)

    def _claim_next_job(self) -> tuple[str, Any, int] | None:
        with self._cv:
            job = self._take_next_locked()
            while job is None and not self._closed:
                self._cv.wait()
                job = self._take_next_locked()
            if self._closed and job is None:
                self._busy = False
                return None
            assert job is not None
            kind, payload = job
            generation = self._generation
            self._busy = True
            self._busy_kind = kind
            self._busy_started = time.monotonic()
            self._cv.notify_all()
        return kind, payload, generation

    def _run_query(self, payload: Any, generation: int) -> None:
        started = time.perf_counter()
        gray, ordinal, source_epoch, capture_stamp = payload
        try:
            fix = self._provider.localize_array(gray)
        except BaseException as error:  # noqa: BLE001 - must never kill the thread
            fix = _failed_fix(
                (time.perf_counter() - started) * 1000.0,
                f"{type(error).__name__}: {error}",
            )
        finally:
            self._finish_job("query")
        self._publish(fix, ordinal, capture_stamp, source_epoch, generation)

    def _record_index_error(self, error_text: str) -> None:
        with self._cv:
            self._index_healthy = False
            self._index_errors += 1
            self._index_last_error = error_text

    def _record_overlap_error(self, error_text: str) -> None:
        with self._cv:
            self._overlap_healthy = False
            self._overlap_errors += 1
            self._overlap_last_error = error_text

    def _run_index(self, payload: Any) -> None:
        reference, callback = payload
        result = None
        error_text = None
        try:
            result = self._provider.index_workmap_reference(reference)
            if not isinstance(result, str) or not result:
                raise RuntimeError("provider returned an empty workmap reference name")
        except BaseException as error:  # noqa: BLE001 - report index failure
            error_text = f"{type(error).__name__}: {error}"
            self._record_index_error(error_text)
        finally:
            try:
                callback(reference, result, error_text)
            except BaseException as error:  # noqa: BLE001
                self._record_index_error(f"{type(error).__name__}: {error}")
            self._finish_job("index")

    def _run_overlap(self, payload: Any) -> None:
        reference, callback = payload
        result = None
        error_text = None
        try:
            matcher = getattr(self._provider, "match_workmap_overlap", None)
            if not callable(matcher):
                raise RuntimeError("provider does not implement workmap overlap matching")
            result = matcher(reference)
            if not isinstance(result, Mapping):
                raise RuntimeError("provider returned a non-mapping overlap result")
        except BaseException as caught:  # noqa: BLE001 - report overlap failure
            error_text = f"{type(caught).__name__}: {caught}"
            self._record_overlap_error(error_text)
        finally:
            try:
                callback(result, error_text)
            except BaseException as caught:  # noqa: BLE001
                self._record_overlap_error(f"{type(caught).__name__}: {caught}")
            self._finish_job("overlap")

    def _record_depth_error(self, error_text: str) -> None:
        with self._cv:
            self._depth_healthy = False
            self._depth_errors += 1
            self._depth_last_error = error_text

    def _run_depth(self, payload: Any) -> None:
        request, callback = payload
        result = None
        error_text = None
        try:
            result = self._depth_estimator.lift(request)
        except BaseException as caught:  # noqa: BLE001 - preserve worker and report failure
            error_text = f"{type(caught).__name__}: {caught}"
            self._record_depth_error(error_text)
        finally:
            try:
                callback(request, result, error_text)
            except BaseException as caught:  # noqa: BLE001
                self._record_depth_error(f"{type(caught).__name__}: {caught}")
            self._finish_job("depth")

    def _finish_job(self, kind: str) -> None:
        with self._cv:
            self._busy = False
            self._busy_kind = None
            self._busy_started = None
            if kind == "query":
                self._queries_since_maintenance += 1
            else:
                self._queries_since_maintenance = 0
            self._cv.notify_all()

    def _publish(
        self,
        fix,
        ordinal: int,
        capture_stamp: float,
        source_epoch: int,
        generation: int,
    ) -> None:
        with self._cv:
            if self._closed or generation != self._generation:
                return
            try:
                self._out.get_nowait()
            except queue.Empty:
                pass
            self._out.put_nowait(
                RelocResult(
                    fix=fix,
                    ordinal=int(ordinal),
                    capture_stamp=float(capture_stamp),
                    source_epoch=int(source_epoch),
                )
            )


class RelocResult(NamedTuple):
    """One worker result with the capture identity it localized."""

    fix: Any
    ordinal: int
    capture_stamp: float
    source_epoch: int


def _failed_fix(runtime_ms: float, reason: str) -> "RelocFix":
    """A well-formed ABSTAINED fix, so a worker exception looks like a refusal.

    Imported lazily: the fast loop must be constructible without the GPU
    provider module having been touched.
    """
    from live_provider import RelocFix

    return RelocFix(
        ok=False,
        status="ABSTAINED",
        cam_from_world=None,
        query_xy=np.zeros((0, 2), dtype=np.float64),
        point3d_ids=np.zeros((0,), dtype=np.int64),
        point_xyz=np.zeros((0, 3), dtype=np.float64),
        inliers=0,
        inlier_ratio=0.0,
        reproj_p90=None,
        reference_names=(reason,),
        runtime_ms=float(runtime_ms),
    )


def _center(cam_from_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(cam_from_world, dtype=float)
    return -pose[:3, :3].T @ pose[:3, 3]


def _yaw_from_rotation(rotation: np.ndarray) -> float:
    """Azimuth of the camera optical axis in the raw map's horizontal plane.

    `cam_from_world` is p_cam = R p_world + t, so row 2 of R is the camera
    forward axis expressed in world coordinates (identical to the EDM tracker's
    `R.T @ [0,0,1]`). A gravity-aligned heading needs MapFrame; the adapter
    substitutes `map_frame.heading(R[2])` when one is attached.
    """
    forward = np.asarray(rotation, dtype=float)[2]
    return float(math.atan2(float(forward[1]), float(forward[0])))

class SeedProvenance(NamedTuple):
    """Observation provenance for trusted workmap promotion."""

    ordinal: int
    stamp: float
    source_epoch: int
    admissible: bool


class CaptureApproval(NamedTuple):
    """Visual pose admission approval bound strictly to one capture identity."""

    ordinal: int
    stamp: float
    source_epoch: int
    pose: np.ndarray
    admissible: bool


class TwoRateTracker:
    """Per-frame fast loop with an asynchronous map relocalizer behind it."""

    def __init__(
        self,
        assets: "DirectMapAssets",
        profile: "DirectProfile",
        provider: Any,
        *,
        online_workmap_active: bool = False,
        depth_estimator: Any = None,
    ) -> None:
        import pycolmap

        fast = profile.fast_loop
        if str(fast.tracker) != "klt":
            raise ValueError(f"unsupported fast-loop tracker {fast.tracker!r}")
        if bool(fast.topup):
            raise ValueError(
                "fast_loop.topup is frozen off: re-projecting the local map at its "
                "pose-predicted pixel makes the next PnP agree with the pose it "
                "already had, which locks in drift"
            )
        if bool(profile.vo.refine) or bool(profile.vo.window_ba):
            raise ValueError(
                "vo.refine / vo.window_ba are frozen off (P174_AND_NEXT.md:96): "
                "measured slower with no accuracy gain"
            )

        self.assets = assets
        self.profile = profile
        self.provider = provider
        self._online_workmap_active = bool(online_workmap_active)
        #: Gravity-aligned basis for this map, or None when the site ships no
        #: alignment. Assigned through `DirectTrackerAdapter.map_frame`, which
        #: the worker reassigns after construction, so it is never snapshotted
        #: here. Configuration, not tracking state: `reset()` leaves it alone.
        self.map_frame: Any = None

        width, height = (int(v) for v in fast.resolution)
        self._w, self._h = width, height
        fx, fy, cx, cy = _scaled_intrinsics(profile, width, height)
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        self._K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        camera = pycolmap.Camera.create(1, "PINHOLE", fx, width, height)
        camera.params = [fx, fy, cx, cy]
        self._camera = camera

        # The forward-backward KLT gate already leaves >95% inliers, so
        # pycolmap's defaults buy nothing and dominate the fast loop.
        pnp = fast.pnp
        estimation = pycolmap.AbsolutePoseEstimationOptions()
        estimation.ransac.max_error = float(pnp.max_error_px)
        estimation.ransac.min_num_trials = int(pnp.min_trials)
        estimation.ransac.max_num_trials = int(pnp.max_trials)
        estimation.ransac.confidence = float(pnp.confidence)
        # Deterministic coverage: pycolmap defaults random_seed=-1
        # (nondeterministic), which alone swings P173 coverage 3-4pp
        # run-to-run. Vendor reloc (_solve via _estimate_pnp) already uses
        # seed=0; the fast loop must match so paired A/B actually measures
        # the change, not RNG noise. Not an accuracy knob (P174 ban unaffected).
        estimation.ransac.random_seed = 0
        cv2.setRNGSeed(0)
        refinement = pycolmap.AbsolutePoseRefinementOptions()
        refinement.max_num_iterations = int(pnp.refine_iters)
        self._pnp_estimation = estimation
        self._pnp_refinement = refinement
        self._pycolmap = pycolmap

        self._tracker = KLTTracker(
            fb_max_px=fast.klt.fb_max_px, win=fast.klt.win, levels=fast.klt.levels
        )
        self._worker = RelocWorker(provider, depth_estimator=depth_estimator)

        self._ordinal = -1
        self._source_epoch = 0
        self._models_ready = False
        self._detect_executor: ThreadPoolExecutor | None = None
        self._reset_tracking_state()

    # ---------- lifecycle ----------
    def ensure_models(self) -> None:
        """Load the GPU models and start the worker, outside the timed loop."""
        if not self._models_ready:
            self.provider.ensure_models()
            self._models_ready = True
        self._worker.start()

    def close(self) -> None:
        self._discard_pending_detection()
        executor, self._detect_executor = getattr(self, "_detect_executor", None), None
        if executor is not None:
            executor.shutdown(wait=True)
        self._worker.close()

    def reset(self) -> None:
        """Drop every tracking prior. The map assets and models are untouched."""
        self._discard_pending_detection()
        self._source_epoch += 1
        self._tracker.reset()
        self._worker.reset()
        self._reset_tracking_state()

    def _reset_tracking_state(self) -> None:
        if not hasattr(self, "_source_epoch"):
            self._source_epoch = 0
        self._live_xy = np.zeros((0, 2), dtype=np.float64)
        self._live_ids = np.zeros((0,), dtype=np.int64)
        self._live_xyz = np.zeros((0, 3), dtype=np.float64)
        self._vo_xy = np.zeros((0, 2), dtype=float)
        self._vo_seed_xy = np.zeros((0, 2), dtype=float)
        self._vo_bid = np.zeros((0,), dtype=np.int64)
        self._vo_pose_by_bid: dict[int, np.ndarray] = {}
        self._vo_seed_meta: dict[int, Any] = {}
        self._pending_detection: _PendingDetection | None = None
        self._workmap_landmarks: dict[int, dict[str, Any]] = {}
        self._workmap_pending_ids: set[int] = set()
        self._workmap_pose_ids: set[int] = set()
        self._current_handover_scope = "global"
        self._frame_reference_scope = "global"
        self._workmap_initial_anchor = False
        self._handover_anchor_answer: Any = None
        self._vo_bid_next = 0
        self._vo_next_id = -1
        self._current_capture_approval: CaptureApproval | None = None
        # (source epoch, ordinal, capture stamp, grayscale frame). The worker
        # must hand back the same capture identity before its anchor can cross
        # this ring; ordinal alone survives neither a reset nor timestamp jitter.
        self._ring: deque[tuple[int, int, float, np.ndarray]] = deque(maxlen=HANDOVER_MAX_FRAMES)
        self._pending_ordinal: int | None = None
        self._pending_source_epoch: int | None = None
        self._pending_capture_stamp: float | None = None
        self._last_reloc_stamp: float | None = None
        self._last_pose: np.ndarray | None = None
        self._last_gray: np.ndarray | None = None
        self._last_pose_ordinal = -(10**9)
        self._last_pose_stamp: float | None = None
        self._last_visual_stamp: float | None = None
        self._last_pose_status: str | None = None
        self._prev_center: np.ndarray | None = None
        self._step_history: list[float] = []
        self._speed_history: list[float] = []
        self._dead_reckon_age = 0
        self._last_reloc_status: str | None = None
        self._last_reloc_ms: float | None = None
        self._last_reloc_stages: dict = {}
        self._last_reloc_capture_stamp: float | None = None
        self._last_reloc_source_epoch: int | None = None
        self._last_reloc_ordinal: int | None = None
        self._last_map_stamp: float | None = None
        self._handover_dropped = 0
        self._handover_retry_hops = 0
        self._handover_capture_age_s: float | None = None
        # Cumulative DR-guard hits (hover rejects + yaw downweights). Internal
        # only: direct_localizer_adapter builds last_info from an explicit
        # key whitelist, so a new info-dict key would be silently dropped --
        # per spec the bool key is omitted and only this counter is kept.
        self._dead_reckon_guarded = 0
        # Fused-attitude samples for the inertial bridge: (stamp, yaw, roll,
        # pitch), host-monotonic seconds, yaw NED clockwise-from-north radians.
        self._fused_yaw: deque = deque(maxlen=IMU_FUSED_HISTORY)
        self._klt_diagnostics: dict = {}
        self._previous_capture_stamp: float | None = None
        self._last_observation_sample_stamp = float("-inf")

    # ---------- one frame ----------
    def _step_fast_loop(
        self, gray: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Fast loop: carry live map points and un-triangulated VO candidates forward."""
        n_live = len(self._live_xy)
        combined = (
            np.concatenate([self._live_xy, self._vo_xy]) if self._vo_xy.size else self._live_xy
        )
        fwd_all, keep_all = self._tracker.track(gray, combined)
        self._klt_diagnostics = dict(getattr(self._tracker, "last_diagnostics", {}))
        dr_prev = dr_curr = None
        if combined.size:
            ok_all = keep_all & self._inside(fwd_all)
            self._klt_diagnostics["klt_inside_kept"] = int(np.count_nonzero(ok_all))
            inside, ok_vo = ok_all[:n_live], ok_all[n_live:]
            if self.profile.dead_reckon.enabled:
                dr_prev = combined[ok_all]
                dr_curr = fwd_all[ok_all]
            self._live_xy = np.ascontiguousarray(fwd_all[:n_live][inside], dtype=np.float64)
            self._live_ids = self._live_ids[inside]
            self._live_xyz = self._live_xyz[inside]
            self._vo_xy = fwd_all[n_live:][ok_vo]
            self._vo_seed_xy = self._vo_seed_xy[ok_vo]
            self._vo_bid = self._vo_bid[ok_vo]
        t_track = time.perf_counter()
        return dr_prev, dr_curr, t_track

    def _step_handover(
        self, gray: np.ndarray, ordinal: int, capture_stamp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None, tuple[str, ...], Any]:
        """Poll the worker and walk the fix to this frame without writing _live_*."""
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_xy = np.zeros((0, 2), dtype=np.float64)
        empty_xyz = np.zeros((0, 3), dtype=np.float64)
        reloc_status: str | None = None
        reference_names: tuple[str, ...] = ()
        delivered = self._worker.poll()
        if delivered is None:
            return empty_ids, empty_xy, empty_xyz, reloc_status, reference_names, None
        fix = delivered.fix
        trigger_ordinal = delivered.ordinal
        trigger_stamp = delivered.capture_stamp
        trigger_epoch = delivered.source_epoch
        self._last_reloc_capture_stamp = float(trigger_stamp)
        self._last_reloc_source_epoch = int(trigger_epoch)
        self._last_reloc_ordinal = int(trigger_ordinal)
        if (
            self._pending_source_epoch == trigger_epoch
            and self._pending_ordinal == trigger_ordinal
            and self._pending_capture_stamp == trigger_stamp
        ):
            self._pending_ordinal = None
            self._pending_source_epoch = None
            self._pending_capture_stamp = None
        self._last_reloc_status = reloc_status = str(fix.status)
        self._last_reloc_ms = float(fix.runtime_ms)
        self._last_reloc_stages = dict(fix.stage_ms or {})
        reference_names = tuple(str(name) for name in (fix.reference_names or ()))
        ids, xy, xyz = self._walk_handover(
            fix,
            trigger_ordinal,
            trigger_epoch=trigger_epoch,
            trigger_stamp=trigger_stamp,
            current_epoch=self._source_epoch,
            current_stamp=float(capture_stamp),
            current_ordinal=ordinal,
        )
        return ids, xy, xyz, reloc_status, reference_names, delivered

    def _step_dead_reckon(
        self,
        gray: np.ndarray,
        ordinal: int,
        dr_prev: np.ndarray | None,
        dr_curr: np.ndarray | None,
        stamp: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None, int]:
        dead = self.profile.dead_reckon
        if not (
            dead.enabled
            and self._last_pose is not None
            and self._last_gray is not None
            and self._dead_reckon_age < int(dead.max_frames)
            and self._last_pose_stamp is not None
            and stamp > self._last_pose_stamp
            and self._last_visual_stamp is not None
            and stamp - self._last_visual_stamp <= float(dead.max_age_s)
        ):
            return None, None, None, 0
        prev_xy, curr_xy = dr_prev, dr_curr
        if self._last_pose_ordinal != ordinal - 1 or prev_xy is None or len(prev_xy) < 8:
            extra = self._detect_features(self._last_gray, np.zeros((0, 2), dtype=float))
            if extra.size:
                fwd, keep = self._tracker.track_pair(self._last_gray, gray, extra)
                ok = keep & self._inside(fwd)
                prev_xy, curr_xy = extra[ok], fwd[ok]
        step_len = (
            float(np.median(self._speed_history[-5:])) * (stamp - self._last_pose_stamp)
            if self._speed_history
            else 0.0
        )
        pose, n_inl = self._recover_relative(prev_xy, curr_xy, self._last_pose, step_len)
        if pose is not None:
            self._dead_reckon_age += 1
            return pose, _center(pose), "DEAD_RECKON", n_inl
        return None, None, None, 0

    def observe_fused_state(self, sample) -> None:
        """Keep one fused-attitude sample for the inertial bridge.

        Duck-typed: the worker hands over ``FusedTelemetry`` (stamp/yaw/roll/
        pitch in host-monotonic seconds and NED radians); tests hand over
        anything with the same attributes. Samples without a finite yaw and
        stamp are not attitude aiding and are dropped. A stamp older than the
        newest kept sample is out of order and is dropped, so
        ``_fused_attitude_at`` can keep taking the last entry at or before
        ``when``.
        """
        try:
            stamp = float(getattr(sample, "stamp", None))
            yaw = float(getattr(sample, "yaw", None))
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(stamp) or not math.isfinite(yaw):
            return
        if self._fused_yaw and stamp <= self._fused_yaw[-1][0]:
            return
        try:
            roll = float(getattr(sample, "roll", None))
            pitch = float(getattr(sample, "pitch", None))
        except (TypeError, ValueError, OverflowError):
            roll = pitch = float("nan")
        self._fused_yaw.append((stamp, yaw, roll, pitch))

    def _fused_attitude_at(self, when: float):
        """Latest fused sample at or before ``when``, else None."""
        best = None
        for entry in self._fused_yaw:
            if entry[0] <= when:
                best = entry
            else:
                break
        return best

    @staticmethod
    def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if not math.isfinite(norm) or norm <= 1e-12:
            return np.eye(3)
        x, y, z = axis / norm
        c, s = math.cos(angle), math.sin(angle)
        return np.array(
            [
                [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
            ]
        )

    def _step_imu_bridge(
        self, stamp: float
    ) -> tuple[np.ndarray | None, np.ndarray | None, str | None, int, dict]:
        """Rotate the last visual pose by the fused yaw increment.

        Runs only when visual PnP and visual dead reckoning both failed. Holds
        the last center (no translation source exists: NED velocity is unscaled
        on a scale-free map) and reports WEAK so the flight loop never treats
        it as a confirmed fix. Shares the dead-reckon age budget.
        """
        blank: dict = {
            "imu_bridge": False,
            "imu_sample_age_s": None,
            "imu_yaw_delta_deg": None,
            "imu_bridge_reason": "no_recent_visual_pose",
        }
        if not self._imu_bridge_armed(stamp):
            return None, None, None, 0, blank
        bridged = self._imu_bridge_delta(stamp)
        blank.update(self._imu_bridge_debug)
        if bridged is None:
            return None, None, None, 0, blank
        delta_ned, sample_age = bridged
        up = getattr(self.map_frame, "up", None)
        if up is None:
            blank["imu_bridge_reason"] = "map_frame_missing"
            return None, None, None, 0, blank
        # Map headings run counter-clockwise-from-East while fused yaw runs
        # clockwise-from-North: a clockwise turn shrinks the map heading.
        rot = self._rot_about_axis(np.asarray(up, dtype=float), -delta_ned)
        last = np.asarray(self._last_pose, dtype=float)
        if last.shape != (3, 4) or not np.isfinite(last).all():
            blank["imu_bridge_reason"] = "pose_invalid"
            return None, None, None, 0, blank
        # R is world-to-camera. A world-frame camera rotation Q transforms
        # camera-to-world as Q @ R.T, hence the new R is R @ Q.T.
        center = _center(last)
        rotation = last[:, :3] @ rot.T
        pose = np.column_stack((rotation, -rotation @ center))
        self._dead_reckon_age += 1
        return (
            pose,
            _center(pose),
            "IMU_BRIDGE",
            0,
            {
                **self._imu_bridge_debug,
                "imu_bridge_reason": "applied",
                "imu_bridge": True,
                "imu_sample_age_s": sample_age,
                "imu_yaw_delta_deg": math.degrees(-delta_ned),
            },
        )

    def _imu_bridge_armed(self, stamp: float) -> bool:
        """Is there a recent visual pose the bridge may extend?"""
        dead = self.profile.dead_reckon
        return bool(
            dead.enabled
            and self._last_pose is not None
            and self._last_pose_stamp is not None
            and self._dead_reckon_age < int(dead.max_frames)
            and stamp > self._last_pose_stamp
            and self._last_visual_stamp is not None
            and stamp - self._last_visual_stamp <= float(dead.max_age_s)
        )

    def _imu_bridge_delta(self, stamp: float) -> tuple[float, float] | None:
        """Fused yaw increment since the last visual pose, else None.

        Returns (delta_ned_radians, sample_age_seconds). Every reject is a
        sample the bridge must not steer by: missing, stale, tilted, or a yaw
        rate no airframe sustains.
        """
        now_sample = self._fused_attitude_at(stamp)
        ref_sample = self._fused_attitude_at(float(self._last_pose_stamp))
        self._imu_bridge_debug = {
            "imu_bridge_reason": "sample_unavailable",
            "imu_sample_stamp_mono": None if now_sample is None else now_sample[0],
            "imu_reference_stamp_mono": None if ref_sample is None else ref_sample[0],
            "imu_sample_age_s": None if now_sample is None else stamp - now_sample[0],
        }
        if now_sample is None or ref_sample is None:
            return None
        sample_age = stamp - now_sample[0]
        if sample_age < 0.0 or sample_age > IMU_BRIDGE_MAX_SAMPLE_AGE_S:
            self._imu_bridge_debug["imu_bridge_reason"] = "sample_stale"
            return None
        if self._last_pose_stamp - ref_sample[0] > IMU_BRIDGE_MAX_SAMPLE_AGE_S:
            self._imu_bridge_debug["imu_bridge_reason"] = "reference_stale"
            return None
        tilt_limit = math.radians(IMU_BRIDGE_MAX_TILT_DEG)
        for entry in (now_sample, ref_sample):
            roll, pitch = float(entry[2]), float(entry[3])
            if (
                not math.isfinite(roll)
                or not math.isfinite(pitch)
                or abs(roll) > tilt_limit
                or abs(pitch) > tilt_limit
            ):
                self._imu_bridge_debug["imu_bridge_reason"] = "tilt_invalid_or_excessive"
                return None
        yaw_now, yaw_ref = now_sample[1], ref_sample[1]
        if not math.isfinite(yaw_now) or not math.isfinite(yaw_ref):
            self._imu_bridge_debug["imu_bridge_reason"] = "yaw_invalid"
            return None
        delta_ned = (yaw_now - yaw_ref + math.pi) % (2.0 * math.pi) - math.pi
        sample_dt = float(now_sample[0]) - float(ref_sample[0])
        if sample_dt < 0.0 or (sample_dt == 0.0 and abs(delta_ned) > 0.0):
            self._imu_bridge_debug["imu_bridge_reason"] = "yaw_rate_excessive"
            return None
        if sample_dt > 0.0 and abs(delta_ned) / sample_dt > math.radians(
            IMU_BRIDGE_MAX_YAW_RATE_DEG_S
        ):
            self._imu_bridge_debug["imu_bridge_reason"] = "yaw_rate_excessive"
            return None
        return delta_ned, sample_age

    def _step_reloc_trigger(
        self, gray: np.ndarray, ordinal: int, stamp: float, status: str
    ) -> bool:
        if not math.isfinite(float(stamp)):
            return False
        reloc = self.profile.reloc
        # FAST_TRACK means the fast PnP minimum, not the frozen strong-map
        # admission floor. A large VO set must not hide eroding map anchors.
        map_confirmed = int(np.count_nonzero(self._live_ids > 0)) >= int(
            reloc.frozen_pnp_thresholds["strong_inliers"]
        )
        idle = status == "FAST_TRACK" and map_confirmed and _reloc_idle(self._step_history)
        due = _reloc_refresh_due(status, stamp, self._last_reloc_stamp, float(reloc.period_s), idle)
        # Retain the usable tracks while continuously replacing only the
        # worker's waiting capture during weak-map recovery.
        unconfirmed = status in {"VO_ONLY", "DEAD_RECKON", "IMU_BRIDGE", "NO_POSE"} or (
            status == "FAST_TRACK" and not map_confirmed
        )
        if len(self._live_ids) < int(reloc.min_points) or unconfirmed or due or self._worker.queued:
            if self._worker.submit(gray, ordinal, stamp, self._source_epoch):
                self._pending_ordinal = ordinal
                self._pending_source_epoch = self._source_epoch
                self._pending_capture_stamp = stamp
                self._last_reloc_stamp = stamp
                return True
        return False

    def _step_bookkeeping(
        self,
        center: np.ndarray | None,
        cam_from_world: np.ndarray | None,
        ordinal: int,
        gray: np.ndarray,
        stamp: float,
        status: str,
    ) -> float | None:
        step = None
        if center is not None and self._prev_center is not None:
            step = float(np.linalg.norm(center - self._prev_center))
        if center is not None:
            self._prev_center = center
        if step is not None:
            self._step_history.append(step)
            if len(self._step_history) > 15:
                del self._step_history[:-15]
        if status == "RELOC_SEED":
            # A map correction is not physical motion.
            self._speed_history.clear()
        elif (
            step is not None
            and status in {"FAST_TRACK", "VO_ONLY"}
            and self._last_pose_status in {"FAST_TRACK", "VO_ONLY", "RELOC_SEED"}
            and self._last_pose_ordinal == ordinal - 1
            and self._last_pose_stamp is not None
            and stamp > self._last_pose_stamp
        ):
            self._speed_history.append(step / (stamp - self._last_pose_stamp))
            del self._speed_history[:-5]
        if cam_from_world is not None:
            self._last_pose = np.asarray(cam_from_world, dtype=float).copy()
            self._last_pose_ordinal = ordinal
            self._last_pose_stamp = stamp
            self._last_pose_status = status
            # Keep the exposure paired with the pose through NO_POSE gaps.
            self._last_gray = gray
            if status not in {"DEAD_RECKON", "IMU_BRIDGE"}:
                self._last_visual_stamp = stamp
        return step

    def _apply_handover(self, gray, walked_ids, walked_xy, walked_xyz):
        self._handover_anchor_answer = None
        strong_inliers = int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        local = getattr(self, "_current_handover_scope", "global") == "workmap"
        if local:
            reseed_floor = max(
                int(self.profile.fast_loop.pnp.min_points),
                int(self.profile.fast_loop.pnp.min_inliers),
            )
        elif getattr(self, "_online_workmap_active", False) and not (
            self._last_pose_status in {"FAST_TRACK", "RELOC_SEED"}
            and self._last_pose_ordinal == self._ordinal - 1
            and getattr(self, "_last_map_inliers", 0) >= strong_inliers
        ):
            # Recovery keeps the early strong-map floor. A healthy track uses the
            # normal replacement floor; collecting a closure anchor is independent.
            reseed_floor = strong_inliers
        else:
            reseed_floor = max(strong_inliers, self.profile.fast_loop.reseed_min_points)
        reseeded = int(walked_ids.size) >= reseed_floor
        handover_points = int(walked_ids.size)
        refresh = False
        anchor = None
        if (
            self._online_workmap_active and not local and reseeded
            and self._last_pose is not None
            and self._last_pose_status in {"FAST_TRACK", "RELOC_SEED"}
            and self._last_pose_ordinal == self._ordinal - 1
            and self._last_map_inliers >= strong_inliers
        ):
            anchor = self._estimate_handover_anchor(walked_ids, walked_xy, walked_xyz)
            if anchor is not None:
                anchor_pose = np.asarray(anchor["cam_from_world"].matrix())[:3, :4]
                shift = np.linalg.norm(_center(anchor_pose) - _center(self._last_pose))
                cosine = (np.trace(anchor_pose[:, :3] @ self._last_pose[:, :3].T) - 1.0) / 2.0
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                refresh = shift <= REFRESH_MAX_CENTER_SHIFT_U and angle <= REFRESH_MAX_ROTATION_DEG
        if reseeded:
            if getattr(self, "_online_workmap_active", False) and not local:
                # Global recovery replaces old positive anchors, but first carry
                # current negative tracks through real KLT and keep only
                # revision-valid two-view geometry. PnP below validates them
                # against the newly walked global observations.
                self._step_fast_loop(gray)
                valid = []
                for row, landmark_id in enumerate(self._live_ids):
                    landmark = self._workmap_landmarks.get(int(landmark_id))
                    if (
                        landmark_id < 0
                        and landmark is not None
                        and len(landmark.get("observations", ())) >= 2
                        and int(landmark.get("revision", -1)) >= 0
                    ):
                        valid.append(row)
                old_xy = self._live_xy[valid]
                old_ids = self._live_ids[valid]
                old_xyz = self._live_xyz[valid]
                pos = walked_ids > 0
                new_ids = np.concatenate([walked_ids[pos], old_ids])
                new_xy = np.concatenate([walked_xy[pos], old_xy], axis=0)
                new_xyz = np.concatenate([walked_xyz[pos], old_xyz], axis=0)
                self._live_ids, self._live_xy, self._live_xyz = self._cap_reseed_points(
                    new_ids, new_xy, new_xyz
                )
            else:
                # A session-only fix is a weak visual seed, never a global
                # RELOC_SEED. It may replace a lost track and is re-solved by
                # the current-frame PnP below.
                self._live_xy = np.ascontiguousarray(walked_xy, dtype=np.float64)
                self._live_ids = np.asarray(walked_ids, dtype=np.int64).copy()
                self._live_xyz = np.ascontiguousarray(walked_xyz, dtype=np.float64)
            self._tracker.set_frame(gray)
            dr_prev = dr_curr = None
            t_track = time.perf_counter()
            if refresh:
                reseeded = False
        else:
            dr_prev, dr_curr, t_track = self._step_fast_loop(gray)
            if walked_ids.size and len(self._live_ids) >= self.profile.fast_loop.pnp.min_points:
                handover_points = self._merge_handover_points(walked_ids, walked_xy, walked_xyz)
        if not local and walked_ids.size and self._anchors_wanted():
            self._handover_anchor_answer = anchor if refresh else self._estimate_handover_anchor(
                walked_ids, walked_xy, walked_xyz
            )
        return reseeded, handover_points, dr_prev, dr_curr, t_track

    def _estimate_handover_anchor(self, ids, xy, xyz):
        """Solve this capture's walked frozen observations without changing live tracks."""
        positive = ids > 0
        limits = self.profile.reloc.frozen_pnp_thresholds
        strong = int(limits["strong_inliers"])
        if np.count_nonzero(positive) < strong:
            return None
        xy, xyz = xy[positive], xyz[positive]
        estimation = self._pycolmap.AbsolutePoseEstimationOptions(self._pnp_estimation.todict())
        estimation.ransac.max_error = float(limits["maximum_reprojection_p90_px"])
        answer = self._pycolmap.estimate_and_refine_absolute_pose(
            xy, xyz, self._camera, estimation, self._pnp_refinement
        )
        if answer is None:
            return None
        mask = np.asarray(answer["inlier_mask"], dtype=bool)
        if mask.sum() < strong or mask.mean() < float(limits["minimum_inlier_ratio"]):
            return None
        pose = np.asarray(answer["cam_from_world"].matrix(), dtype=np.float64)[:3, :4]
        quality = pose_quality(xy[mask], xyz[mask], pose, self._K, self._w, self._h)
        if (
            quality["positive_depth_ratio"] < float(limits["minimum_positive_depth_ratio"])
            or quality["inlier_hull_coverage"] < float(limits["minimum_hull_coverage"])
            or quality["inlier_grid_cells"] < int(limits["minimum_occupancy_4x4"])
            or quality["reproj_p90"] is None
            or quality["reproj_p90"] > float(limits["maximum_reprojection_p90_px"])
        ):
            return None
        return answer

    def _cap_reseed_points(self, ids, xy, xyz):
        """Use the normal half-cap main reservation before retaining session tracks."""
        cap = int(self.profile.fast_loop.track_cap)
        ids = np.asarray(ids, dtype=np.int64)
        xy = np.asarray(xy, dtype=np.float64)
        xyz = np.asarray(xyz, dtype=np.float64)
        positive = np.flatnonzero(ids > 0)
        negative = np.flatnonzero(ids < 0)
        strong = int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        old_floor = min(len(positive), max(strong, cap // 2), cap)
        old_keep = (
            positive[spatial_indices(xy[positive], old_floor, self._w, self._h)]
            if old_floor
            else np.zeros((0,), dtype=np.int64)
        )
        remaining = max(0, cap - len(old_keep))
        negative_keep = (
            negative[spatial_indices(xy[negative], min(remaining, len(negative)), self._w, self._h)]
            if remaining and len(negative)
            else np.zeros((0,), dtype=np.int64)
        )
        remaining -= len(negative_keep)
        extra_positive = positive[~np.isin(positive, old_keep)]
        extra_positive = (
            extra_positive[
                spatial_indices(
                    xy[extra_positive], min(remaining, len(extra_positive)), self._w, self._h
                )
            ]
            if remaining and len(extra_positive)
            else np.zeros((0,), dtype=np.int64)
        )
        keep = np.concatenate([old_keep, negative_keep, extra_positive])
        return ids[keep], np.ascontiguousarray(xy[keep]), np.ascontiguousarray(xyz[keep])

    def step(self, gray_native: np.ndarray, capture_stamp: float) -> dict:
        t0 = time.perf_counter()
        self._finish_pending_detection()
        self._ordinal += 1
        ordinal = self._ordinal
        stamp = float(capture_stamp)
        frame_dt = (
            None if self._previous_capture_stamp is None else stamp - self._previous_capture_stamp
        )
        self._previous_capture_stamp = stamp
        self._klt_diagnostics = {}
        self._current_capture_approval = None

        frame = np.asarray(gray_native)
        if (frame.shape[1], frame.shape[0]) != (self._w, self._h):
            frame = cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        self._ring.append((self._source_epoch, ordinal, stamp, gray))
        t_gray = time.perf_counter()

        self._current_handover_scope = "global"
        self._frame_reference_scope = "global"
        # 1. worker handover: poll and catch-up walk, do not write _live_* yet
        (
            walked_ids,
            walked_xy,
            walked_xyz,
            reloc_status,
            reference_names,
            delivered,
        ) = self._step_handover(gray, ordinal, stamp)
        if delivered is not None:
            fix_scope = str(getattr(delivered.fix, "reference_scope", "global"))
            self._current_handover_scope = (
                "workmap" if fix_scope == "workmap" and walked_ids.size else "global"
            )
        t_handover = time.perf_counter()

        # A partial handover can augment a viable track, but cannot seed an empty one.
        reseeded, handover_points, dr_prev, dr_curr, t_track = self._apply_handover(
            gray, walked_ids, walked_xy, walked_xyz
        )
        self._augment_live_points(gray, reseeded)
        t_query = time.perf_counter()
        # 3. absolute pose
        status, cam_from_world, center, inliers, map_inliers, vo_inliers, n_corr, quality = (
            self._step_absolute_pose(reseeded, stamp)
        )
        # A partial strong fix can lose a mixed PnP vote to a much larger,
        # drifting VO set. Try its current-frame map hypothesis independently
        # only after the mixed solution failed strong map support. An empty
        # track still requires the unchanged configured reseed minimum.
        if (
            not reseeded
            and n_corr >= int(self.profile.fast_loop.pnp.min_points)
            and map_inliers < int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        ):
            recovered = self._recover_partial_handover(walked_ids, walked_xy, walked_xyz, stamp)
            if recovered is not None:
                (
                    status,
                    cam_from_world,
                    center,
                    inliers,
                    map_inliers,
                    vo_inliers,
                    n_corr,
                    quality,
                ) = recovered
        t_pnp = time.perf_counter()

        # 4. dead reckoning
        imu_info: dict = {
            "imu_bridge": False,
            "imu_sample_age_s": None,
            "imu_yaw_delta_deg": None,
            "imu_bridge_reason": "visual_pose_available",
        }
        if cam_from_world is None:
            dr_pose, dr_center, dr_status, dr_inl = self._step_dead_reckon(
                gray, ordinal, dr_prev, dr_curr, stamp
            )
            if dr_pose is not None:
                cam_from_world = dr_pose
                center = dr_center
                status = dr_status
                inliers = dr_inl
            else:
                # 4b. inertial bridge: visual tracking is gone but the fused
                # yaw still says which way the camera turned. Holds the last
                # center, shares the dead-reckon age budget, stays WEAK.
                imu_pose, imu_center, imu_status, imu_inl, imu_info = self._step_imu_bridge(stamp)
                if imu_pose is not None:
                    cam_from_world = imu_pose
                    center = imu_center
                    status = imu_status
                    inliers = imu_inl
        else:
            self._dead_reckon_age = 0

        # 5. VO keyframes
        vo = self.profile.vo
        if (
            vo.enabled
            and cam_from_world is not None
            and ordinal % int(vo.keyframe_stride) == 0
        ):
            self._vo_keyframe(gray, cam_from_world)
        t_vo = time.perf_counter()

        # 6. relocalizer trigger
        submitted = self._step_reloc_trigger(gray, ordinal, stamp, status)

        # 7. bookkeeping
        step = self._step_bookkeeping(center, cam_from_world, ordinal, gray, stamp, status)
        t_end = time.perf_counter()

        rotation = None if cam_from_world is None else cam_from_world[:3, :3]
        return {
            **quality,
            "ok": cam_from_world is not None,
            "status": status,
            "ordinal": ordinal,
            "capture_stamp": stamp,
            "center": center,
            "R": rotation,
            "cam_from_world": cam_from_world,
            "yaw": None if rotation is None else _yaw_from_rotation(rotation),
            "inliers": int(inliers),
            "map_inliers": int(map_inliers),
            "vo_inliers": int(vo_inliers),
            "live_points": int(len(self._live_ids)),
            "vo_candidates": int(len(self._vo_xy)),
            "dead_reckon_age": int(self._dead_reckon_age),
            "imu_bridge": bool(imu_info["imu_bridge"]),
            "imu_sample_age_s": imu_info["imu_sample_age_s"],
            "imu_yaw_delta_deg": imu_info["imu_yaw_delta_deg"],
            "imu_bridge_reason": imu_info.get("imu_bridge_reason"),
            "imu_sample_stamp_mono": imu_info.get("imu_sample_stamp_mono"),
            "imu_reference_stamp_mono": imu_info.get("imu_reference_stamp_mono"),
            "map_constraint_age_s": (
                None if self._last_map_stamp is None else max(0.0, stamp - self._last_map_stamp)
            ),
            "frame_stamp_mono": stamp,
            "frame_dt_s": frame_dt,
            "klt_reseeded": reseeded,
            "pnp_image_size_px": [self._w, self._h],
            **self._klt_diagnostics,
            "handover_points": int(handover_points),
            "handover_dropped": int(self._handover_dropped),
            "handover_retry_hops": int(self._handover_retry_hops),
            "handover_capture_age_s": self._handover_capture_age_s,
            "reloc_status": reloc_status if reloc_status is not None else self._last_reloc_status,
            "reloc_delivered": delivered is not None,
            "reloc_submitted": submitted,
            "reloc_busy": self._worker.busy,
            "reference_names": reference_names,
            "step": step,
            "n_corr": n_corr,
            "loop_ms": (t_end - t0) * 1000.0,
            "gray_ms": (t_gray - t0) * 1000.0,
            "handover_ms": (t_handover - t_gray) * 1000.0,
            "track_ms": (t_track - t_handover) * 1000.0,
            "pnp_ms": (t_pnp - t_query) * 1000.0,
            "vo_ms": (t_vo - t_pnp) * 1000.0,
            "bookkeeping_ms": (t_end - t_vo) * 1000.0,
            "reloc_ms": self._last_reloc_ms,
            "reloc_retrieval_ms": self._last_reloc_stages.get("retrieval_ms"),
            "reloc_match_lift_ms": self._last_reloc_stages.get("match_lift_ms"),
            "reloc_edm_match_ms": self._last_reloc_stages.get("edm_match_ms"),
            "reloc_lift_prepare_ms": self._last_reloc_stages.get("lift_prepare_ms"),
            "reloc_pnp_ms": self._last_reloc_stages.get("reloc_pnp_ms"),
            "reloc_reference_count": self._last_reloc_stages.get("matched_references"),
            "reloc_capture_stamp_mono": self._last_reloc_capture_stamp,
            "reference_scope": self._frame_reference_scope,
            "reloc_source_epoch": self._last_reloc_source_epoch,
            "reloc_ordinal": self._last_reloc_ordinal,
        }

    def _mixed_pnp_rows(self) -> np.ndarray | None:
        """Live rows the mixed PnP may use; ``None`` means all of them."""
        return None

    def _collect_submap_anchor(self, old_only: Any, estimate: Any, args: tuple) -> None:
        """Called on every strong frozen-map solve; subclasses pair it with submap-only solves."""

    def _retained_rows(self, track_mask: np.ndarray) -> np.ndarray:
        """Extra live rows to keep for the next frame beyond the PnP inliers."""
        return np.zeros(len(self._live_ids), dtype=bool)

    def _anchors_wanted(self) -> bool:
        """Whether this frame's frozen-map-only solve feeds a submap anchor pair."""
        return False

    def _old_support(self, answer: Any, old: np.ndarray) -> tuple[int, float]:
        """Old-map points within 4 px under ``answer`` and their reprojection p90."""
        pose = np.asarray(answer["cam_from_world"].matrix(), dtype=float)[:3, :4]
        points = self._live_xyz[old] @ pose[:3, :3].T + pose[:3, 3]
        depth = points[:, 2]
        error = np.full(len(old), np.inf)
        valid = depth > 1e-6
        error[valid] = np.hypot(
            self._fx * points[valid, 0] / depth[valid] + self._cx - self._live_xy[old[valid], 0],
            self._fy * points[valid, 1] / depth[valid] + self._cy - self._live_xy[old[valid], 1],
        )
        valid_errors = error[np.isfinite(error)]
        return int(np.count_nonzero(error <= 4.0)), (
            float(np.percentile(valid_errors, 90)) if len(valid_errors) else np.inf
        )

    def _workmap_absolute_pose(self) -> tuple[Any, np.ndarray] | None:
        """Mixed PnP like the frozen baseline; the frozen-only pose replaces it only when the
        mixed one lets the old map's points lose their support (or an anchor pair needs it)."""
        estimate = self._pycolmap.estimate_and_refine_absolute_pose
        args = (self._camera, self._pnp_estimation, self._pnp_refinement)
        rows = self._mixed_pnp_rows()
        mixed: Any = None
        mixed_mask: np.ndarray | None = None
        if rows is None:
            mixed = estimate(self._live_xy, self._live_xyz, *args)
            if mixed is not None:
                mixed_mask = np.asarray(mixed["inlier_mask"], dtype=bool)
        elif len(rows):
            mixed = estimate(self._live_xy[rows], self._live_xyz[rows], *args)
            if mixed is not None:
                mixed_mask = np.zeros(len(self._live_ids), dtype=bool)
                mixed_mask[rows] = np.asarray(mixed["inlier_mask"], dtype=bool)
        min_inliers = int(self.profile.fast_loop.pnp.min_inliers)
        strong = int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        self._pose_basis = "mixed"
        old = np.flatnonzero(self._live_ids > 0)
        handover_anchor = getattr(self, "_handover_anchor_answer", None)
        if handover_anchor is not None:
            self._collect_submap_anchor(handover_anchor, estimate, args)
        if len(old) < strong or len(old) == len(self._live_ids):
            return None if mixed is None else (mixed, mixed_mask)
        mixed_ok = mixed is not None and int(mixed_mask.sum()) >= min_inliers
        if mixed_ok and (handover_anchor is not None or not self._anchors_wanted()):
            kept, _ = self._old_support(mixed, old)
            if kept >= OLD_SUPPORT_SHARE * len(old):
                return mixed, mixed_mask
        old_only = estimate(self._live_xy[old], self._live_xyz[old], *args)
        if old_only is None:
            return None if mixed is None else (mixed, mixed_mask)
        old_mask = np.asarray(old_only["inlier_mask"], dtype=bool)
        if int(old_mask.sum()) < min_inliers:
            return None if mixed is None else (mixed, mixed_mask)
        if int(old_mask.sum()) >= strong and handover_anchor is None:
            self._collect_submap_anchor(old_only, estimate, args)
        if mixed_ok:
            mixed_support, mixed_p90 = self._old_support(mixed, old)
            old_support, old_p90 = self._old_support(old_only, old)
            if mixed_support >= old_support and mixed_p90 <= old_p90 + 1e-9:
                return mixed, mixed_mask
        self._pose_basis = "old_only"
        full_mask = np.zeros(len(self._live_ids), dtype=bool)
        full_mask[old] = old_mask
        return old_only, full_mask

    def _step_absolute_pose(
        self, reseeded: bool, stamp: float
    ) -> tuple[str, np.ndarray | None, np.ndarray | None, int, int, int, int, dict]:
        """PnP the live set; keep only geometric inliers for the next frame."""
        self._workmap_pose_ids: set[int] = set()
        n_corr = int(len(self._live_xy))
        self._frame_reference_scope = "global"
        pnp = self.profile.fast_loop.pnp
        solved = self._solve_live_pose() if n_corr >= int(pnp.min_points) else None
        inliers = 0 if solved is None else int(solved[1].sum())
        if solved is None or inliers < int(pnp.min_inliers):
            self._current_capture_approval = None
            return "NO_POSE", None, None, inliers, 0, 0, n_corr, {}
        answer, mask = solved
        matrix = np.asarray(answer["cam_from_world"].matrix(), dtype=float)
        cam_from_world = matrix[:3, :4]
        center = _center(cam_from_world)
        map_inliers = int(np.count_nonzero(self._live_ids[mask] > 0))
        vo_inliers = inliers - map_inliers
        status = self._pose_status(reseeded, map_inliers, vo_inliers)
        pnp_xy, pnp_ids, pnp_xyz = self._live_xy[mask], self._live_ids[mask], self._live_xyz[mask]
        self._workmap_pose_ids = {
            int(key) for key in pnp_ids if key < 0 and int(key) in self._workmap_landmarks
        }
        if (
            self._online_workmap_active and self.profile.vo.enabled
            and self._ordinal % int(self.profile.vo.keyframe_stride) == 0
        ):
            self.approve_capture(
                cam_from_world,
                status=status,
                map_inliers=map_inliers,
                n_corr=n_corr,
                ordinal=self._ordinal,
                stamp=stamp,
            )
        else:
            self._current_capture_approval = None
        track_mask = self._keep_pose_tracks(cam_from_world, mask, map_inliers, stamp)
        quality = pose_quality(pnp_xy, pnp_xyz, cam_from_world, self._K, self._w, self._h)
        if self._online_workmap_active:
            quality["workmap_retained_tracks"] = int(track_mask.sum()) - inliers
        elapsed = stamp - self._last_observation_sample_stamp
        if elapsed >= 1.0 or (reseeded and elapsed >= 0.5):
            quality["pnp_observation_sample"] = self._pnp_observation_sample(
                stamp, inliers, pnp_ids, pnp_xy, pnp_xyz, cam_from_world
            )
            self._last_observation_sample_stamp = stamp
        if map_inliers >= (
            int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
            if self._online_workmap_active
            else int(pnp.min_inliers)
        ):
            self._last_map_stamp = stamp
        return status, cam_from_world, center, inliers, map_inliers, vo_inliers, n_corr, quality

    def _solve_live_pose(self) -> tuple[Any, np.ndarray] | None:
        if self._online_workmap_active:
            return self._workmap_absolute_pose()
        answer = self._pycolmap.estimate_and_refine_absolute_pose(
            self._live_xy,
            self._live_xyz,
            self._camera,
            self._pnp_estimation,
            self._pnp_refinement,
        )
        return None if answer is None else (answer, np.asarray(answer["inlier_mask"], dtype=bool))

    def _pose_status(self, reseeded: bool, map_inliers: int, vo_inliers: int) -> str:
        if reseeded:
            status = (
                "VO_ONLY"
                if getattr(self, "_current_handover_scope", "global") == "workmap"
                else "RELOC_SEED"
            )
        elif map_inliers >= int(self.profile.fast_loop.pnp.min_inliers):
            status = "FAST_TRACK"
        else:
            status = "VO_ONLY"
        if (
            self._online_workmap_active
            and status == "VO_ONLY"
            and vo_inliers > 0
            and map_inliers < int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        ):
            self._frame_reference_scope = "workmap"
        return status

    def _keep_pose_tracks(
        self, cam_from_world: np.ndarray, mask: np.ndarray, map_inliers: int, stamp: float
    ) -> np.ndarray:
        """Carry PnP inliers (plus validated workmap tracks) into the next frame."""
        track_mask = mask
        if self._online_workmap_active and map_inliers >= int(
            self.profile.reloc.frozen_pnp_thresholds["strong_inliers"]
        ):
            track_mask = self._retain_workmap_tracks(cam_from_world, mask)
        track_mask = track_mask | self._retained_rows(track_mask)
        self._live_xy = np.ascontiguousarray(self._live_xy[track_mask], dtype=np.float64)
        self._live_ids = self._live_ids[track_mask]
        self._live_xyz = self._live_xyz[track_mask]
        if self._online_workmap_active:
            for key, xy in zip(self._live_ids, self._live_xy, strict=True):
                if key < 0 and int(key) in self._workmap_landmarks:
                    point = self._workmap_landmarks[int(key)]
                    point["last_inlier"] = (self._ordinal, stamp, xy.copy())
                    point["valid_observation_count"] = point.get("valid_observation_count", 2) + 1
        return track_mask

    def _pnp_observation_sample(
        self,
        stamp: float,
        inliers: int,
        pnp_ids: np.ndarray,
        pnp_xy: np.ndarray,
        pnp_xyz: np.ndarray,
        cam_from_world: np.ndarray,
    ) -> dict:
        # Bounded samples for offline residual/covariance analysis.
        # Positive IDs are map anchors; negative IDs are VO points.
        selected = np.linspace(0, len(pnp_ids) - 1, min(128, len(pnp_ids)), dtype=int)
        return {
            "capture_stamp_mono": stamp,
            "total_inliers": int(inliers),
            "sample_count": len(selected),
            "ids": pnp_ids[selected].tolist(),
            "image_xy_px": pnp_xy[selected].tolist(),
            "world_xyz_u": pnp_xyz[selected].tolist(),
            "K": self._K.tolist(),
            "image_size_px": [self._w, self._h],
            "cam_from_world": cam_from_world.tolist(),
            "sampling": "uniform_index; at most 128 inliers",
        }

    def _retain_workmap_tracks(self, pose: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Keep validated KLT observations without treating them as pose inputs."""
        if not getattr(getattr(self.profile, "online_workmap", None), "retain_valid_tracks", False):
            return mask
        rows = np.flatnonzero((self._live_ids < 0) & ~mask)
        rows = np.asarray(
            [
                row
                for row in rows
                if int(self._live_ids[row]) in self._workmap_landmarks
                and len(self._workmap_landmarks[int(self._live_ids[row])].get("observations", ()))
                >= 2
            ],
            dtype=np.int64,
        )
        if not len(rows):
            return mask
        camera = self._live_xyz[rows] @ pose[:, :3].T + pose[:, 3]
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-3)
        projected = camera @ self._K.T
        xy = projected[:, :2] / np.where(valid, projected[:, 2], 1.0)[:, None]
        valid &= np.linalg.norm(xy - self._live_xy[rows], axis=1) <= float(
            self.profile.vo.max_reproj_px
        )
        retained = mask.copy()
        retained[rows[valid]] = True
        return retained

    def _recover_partial_handover(self, ids, xy, xyz, stamp):
        """Accept a walked map-only PnP only under the existing frozen gates.

        The capture pose itself is never applied: these points have already
        passed the identity/stamp/ring checks and KLT to this exposure.
        Rejected candidates leave the mixed track and its pose untouched.
        """
        thresholds = self.profile.reloc.frozen_pnp_thresholds
        if len(ids) < max(
            int(thresholds["strong_inliers"]), int(self.profile.fast_loop.pnp.min_points)
        ):
            return None
        answer = self._pycolmap.estimate_and_refine_absolute_pose(
            xy, xyz, self._camera, self._pnp_estimation, self._pnp_refinement
        )
        if answer is None:
            return None
        mask = np.asarray(answer["inlier_mask"], dtype=bool)
        inliers = int(mask.sum())
        if inliers < max(
            int(thresholds["strong_inliers"]), int(self.profile.fast_loop.pnp.min_inliers)
        ):
            return None
        pose = np.asarray(answer["cam_from_world"].matrix(), dtype=float)[:3, :4]
        if not np.isfinite(pose).all():
            return None
        quality = pose_quality(xy[mask], xyz[mask], pose, self._K, self._w, self._h)
        from sfm_diagnosis.site_pipeline.deployment_localizer import localization_is_strong

        metrics = {
            "inlier_count": inliers,
            "inlier_ratio": inliers / len(ids),
            "convex_hull_coverage": quality.get("inlier_hull_coverage"),
            "occupancy_4x4": quality.get("inlier_grid_cells"),
            "positive_depth_ratio": quality.get("positive_depth_ratio"),
            "reprojection_p90": quality.get("reproj_p90"),
        }
        if not localization_is_strong(metrics, decision_status="ACCEPT", thresholds=thresholds):
            return None
        self._live_ids = ids[mask]
        self._live_xy = np.ascontiguousarray(xy[mask], dtype=np.float64)
        self._live_xyz = xyz[mask]
        self._last_map_stamp = stamp
        # Keep the original relocalization confirmation contract even though
        # this was a hypothesis recovery rather than a 120-point KLT reseed.
        return ("RELOC_SEED", pose, _center(pose), inliers, inliers, 0, len(ids), quality)

    # ---------- handover ----------
    def _track_handover_chain(self, chain, ids, xy, xyz):
        ids, xy, xyz = self._cap_handover_points(ids, xy, xyz)
        n = len(chain)
        # Keep stride-2 while it supplies enough anchors. If a hop drops below
        # the existing reseed floor, retry through the intervening exposure.
        # Every retry uses the same FB/bounds gates; retain it only if it keeps
        # more points. No retry recursively expands the bounded ring walk.
        a, stride = 0, 2
        while a < n - 1 and xy.size:
            b = min(a + stride, n - 1)
            fwd, keep = self._tracker.track_pair(chain[a], chain[b], xy)
            inside = keep & self._inside(fwd)
            required = min(len(xy), int(self.profile.fast_loop.reseed_min_points))
            if b - a == 2 and np.count_nonzero(inside) < required:
                self._handover_retry_hops += 1
                wide_diagnostics = dict(getattr(self._tracker, "last_diagnostics", {}))
                midpoint, mid_keep = self._tracker.track_pair(chain[a], chain[a + 1], xy)
                mid_inside = mid_keep & self._inside(midpoint)
                mid_indices = np.flatnonzero(mid_inside)
                if mid_indices.size:
                    end, end_keep = self._tracker.track_pair(
                        chain[a + 1], chain[b], midpoint[mid_inside]
                    )
                    end_inside = end_keep & self._inside(end)
                    if np.count_nonzero(end_inside) > np.count_nonzero(inside):
                        selected = mid_indices[end_inside]
                        xy, ids, xyz = end[end_inside], ids[selected], xyz[selected]
                        # The intervening exposure helped. Use consecutive
                        # frames for the rest of this handover, avoiding
                        # another failed wide hop during the same motion.
                        a, stride = b, 1
                        continue
                self._tracker.last_diagnostics = wide_diagnostics
            xy, ids, xyz = fwd[inside], ids[inside], xyz[inside]
            a = b
        return ids, xy, xyz

    def _accept_handover_scope(self, fix: "RelocFix") -> str | None:
        """The fix's reference scope, or None when this tracker must drop it."""
        scope = str(getattr(fix, "reference_scope", "global"))
        if scope not in {"global", "workmap"}:
            return None
        if scope == "workmap":
            names = tuple(str(name) for name in (getattr(fix, "reference_names", ()) or ()))
            validator = getattr(self, "_validate_workmap_fix", None)
            if (
                not getattr(self, "_online_workmap_active", False)
                or not callable(validator)
                or not validator(fix, names)
            ):
                return None
        self._current_handover_scope = scope
        return scope

    def _handover_entries(
        self,
        trigger_ordinal: int,
        trigger_epoch: int,
        trigger_stamp: float | None,
        current_epoch: int,
        current_stamp: float | None,
        current_ordinal: int | None,
    ) -> tuple[list, float, float] | None:
        """Ring entries from the trigger frame to now, or None if the chain is broken."""
        entries = [
            entry
            for entry in self._ring
            if entry[0] == trigger_epoch and entry[1] >= trigger_ordinal
        ]
        first = entries[0][1] if entries else None
        if first != trigger_ordinal or trigger_epoch != current_epoch:
            return None
        if trigger_stamp is None:
            trigger_stamp = entries[0][2]
        if current_stamp is None:
            current_stamp = entries[-1][2]
        if current_ordinal is not None and entries[-1][1] != current_ordinal:
            return None
        stamps = [entry[2] for entry in entries]
        if (
            not math.isfinite(float(trigger_stamp))
            or not math.isfinite(float(current_stamp))
            or not stamps
            or stamps[0] != float(trigger_stamp)
            or (current_ordinal is not None and stamps[-1] != float(current_stamp))
            or float(current_stamp) < float(trigger_stamp)
            or any(not math.isfinite(float(stamp)) for stamp in stamps)
            or any(later < earlier for earlier, later in zip(stamps, stamps[1:]))
        ):
            return None
        return entries, float(trigger_stamp), float(current_stamp)

    def _walk_handover(
        self,
        fix: "RelocFix",
        trigger_ordinal: int,
        *,
        trigger_epoch: int | None = None,
        trigger_stamp: float | None = None,
        current_epoch: int | None = None,
        current_stamp: float | None = None,
        current_ordinal: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_xy = np.zeros((0, 2), dtype=np.float64)
        empty_xyz = np.zeros((0, 3), dtype=np.float64)
        self._handover_capture_age_s = None
        if not fix.ok or fix.cam_from_world is None:
            return empty_ids, empty_xy, empty_xyz
        scope = self._accept_handover_scope(fix)
        if scope is None:
            self._handover_dropped += 1
            return empty_ids, empty_xy, empty_xyz
        ids = np.asarray(fix.point3d_ids, dtype=np.int64)
        xy = np.asarray(fix.query_xy, dtype=float)
        xyz = np.asarray(fix.point_xyz, dtype=float)
        if scope == "workmap" and np.any(ids >= 0):
            self._handover_dropped += 1
            return empty_ids, empty_xy, empty_xyz
        if ids.size < int(self.profile.fast_loop.pnp.min_points):
            return empty_ids, empty_xy, empty_xyz

        walk = self._handover_entries(
            trigger_ordinal,
            self._source_epoch if trigger_epoch is None else trigger_epoch,
            trigger_stamp,
            self._source_epoch if current_epoch is None else current_epoch,
            current_stamp,
            current_ordinal,
        )
        if walk is None:
            self._handover_dropped += 1
            return empty_ids, empty_xy, empty_xyz
        entries, trigger_stamp, current_stamp = walk
        self._handover_capture_age_s = float(current_stamp) - float(trigger_stamp)
        chain = [entry[3] for entry in entries]
        if len(chain) > HANDOVER_MAX_FRAMES:
            # the worker outran the ring buffer: the anchor's 2D can no longer
            # be walked to this frame, so the whole handover is discarded
            self._handover_dropped += 1
            return empty_ids, empty_xy, empty_xyz

        ids, xy, xyz = self._track_handover_chain(chain, ids, xy, xyz)
        if ids.size == 0:
            return empty_ids, empty_xy, empty_xyz
        return ids, xy, xyz

    def _merge_handover_points(self, ids: np.ndarray, xy: np.ndarray, xyz: np.ndarray) -> int:
        """Append current-frame reloc ids after the old live set has been KLT'd."""
        fresh = ~np.isin(ids, self._live_ids)
        xy, ids, xyz = xy[fresh], ids[fresh], xyz[fresh]
        self._live_xy = (
            np.ascontiguousarray(np.concatenate([self._live_xy, xy]), dtype=np.float64)
            if self._live_xy.size
            else np.ascontiguousarray(xy, dtype=np.float64)
        )
        self._live_ids = np.concatenate([self._live_ids, ids]) if self._live_ids.size else ids
        self._live_xyz = np.concatenate([self._live_xyz, xyz]) if self._live_xyz.size else xyz
        return int(ids.size)

    def _cap_handover_points(
        self, ids: np.ndarray, xy: np.ndarray, xyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bound the catch-up set; spread survivors across the image when enabled."""
        cap = int(self.profile.fast_loop.track_cap)
        if len(ids) > cap:
            if self.profile.optimizations.spatial_selection:
                take = spatial_indices(xy, cap, self._w, self._h)
            else:
                take = np.linspace(0, len(ids) - 1, cap).astype(int)
            ids, xy, xyz = ids[take], xy[take], xyz[take]
        return ids, xy, xyz

    # ---------- VO ----------
    def _get_seed_provenance(self, bid: int) -> SeedProvenance | None:
        meta = self._vo_seed_meta.get(bid)
        if isinstance(meta, SeedProvenance):
            return meta
        return None

    def _is_pose_admissible(self, pose: np.ndarray | None) -> bool:
        if not getattr(self, "_online_workmap_active", False) or pose is None:
            return False
        approval = getattr(self, "_current_capture_approval", None)
        if approval is None:
            return False
        if approval.ordinal != self._ordinal:
            return False
        if approval.source_epoch != self._source_epoch:
            return False
        if (
            self._previous_capture_stamp is None
            or not math.isfinite(self._previous_capture_stamp)
            or not math.isfinite(approval.stamp)
            or not math.isclose(
                approval.stamp, float(self._previous_capture_stamp), rel_tol=0.0, abs_tol=1e-6
            )
        ):
            return False
        if not np.array_equal(pose, approval.pose):
            return False
        return bool(approval.admissible)

    def approve_capture(
        self,
        pose: np.ndarray,
        *,
        status: str,
        map_inliers: int,
        n_corr: int,
        ordinal: int | None = None,
        stamp: float | None = None,
    ) -> bool:
        """Explicitly evaluate and record capture approval for a known visual pose."""
        if not getattr(self, "_online_workmap_active", False):
            self._current_capture_approval = None
            return False
        if stamp is None:
            if self._previous_capture_stamp is None or not math.isfinite(self._previous_capture_stamp):
                self._current_capture_approval = None
                return False
            stamp_val = float(self._previous_capture_stamp)
        else:
            if not math.isfinite(stamp):
                self._current_capture_approval = None
                return False
            stamp_val = float(stamp)
        ord_val = self._ordinal if ordinal is None else int(ordinal)
        admissible = bool(self._workmap_pose_admissible(pose, status, map_inliers, n_corr))
        self._current_capture_approval = CaptureApproval(
            ordinal=ord_val,
            stamp=stamp_val,
            source_epoch=int(self._source_epoch),
            pose=np.asarray(pose, dtype=float).copy(),
            admissible=admissible,
        )
        return admissible

    def _workmap_pose_admissible(
        self, pose: np.ndarray, status: str, map_inliers: int, n_corr: int
    ) -> bool:
        """Admit keyframe geometry only from a measured, observable visual pose."""
        if status not in {"FAST_TRACK", "RELOC_SEED", "VO_ONLY"}:
            return False
        limits = self.profile.reloc.frozen_pnp_thresholds
        strong = int(limits["strong_inliers"])
        if not self._workmap_initial_anchor and map_inliers < strong:
            return False
        if len(self._live_ids) < self.profile.fast_loop.pnp.min_inliers:
            return False
        if len(self._live_ids) / max(n_corr, 1) < float(limits["minimum_inlier_ratio"]):
            return False
        old = self._live_ids > 0
        rows = old if not self._workmap_initial_anchor else np.ones(len(old), dtype=bool)
        if not rows.any():
            return False
        quality = pose_quality(
            self._live_xy[rows], self._live_xyz[rows], pose, self._K, self._w, self._h
        )
        if (
            quality["positive_depth_ratio"] < float(limits["minimum_positive_depth_ratio"])
            or quality["inlier_hull_coverage"] < float(limits["minimum_hull_coverage"])
            or quality["inlier_grid_cells"] < int(limits["minimum_occupancy_4x4"])
            or quality["reproj_p90"] is None
            or quality["reproj_p90"] > float(limits["maximum_reprojection_p90_px"])
        ):
            return False
        points = self._live_xyz[rows] @ pose[:3, :3].T + pose[:3, 3]
        if scaled_pose_information(points, fx=self._fx, fy=self._fy, pixel_sigma=1.0).rank < 6:
            return False
        if status in {"FAST_TRACK", "RELOC_SEED"} and map_inliers >= strong:
            self._workmap_initial_anchor = True
        return self._workmap_initial_anchor

    def _select_workmap_live_points(self, cap: int, pose: np.ndarray) -> np.ndarray:
        """Reserve old-map support, then spread good new/old tracks over 8x6 cells."""
        ids = self._live_ids
        xy = self._live_xy
        positive = np.flatnonzero(ids > 0)
        strong_floor = int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        floor = min(len(positive), max(strong_floor, cap // 2), cap)
        old_keep = positive[spatial_indices(xy[positive], floor, self._w, self._h)]
        selected = np.zeros(len(ids), dtype=bool)
        selected[old_keep] = True
        remaining = cap - len(old_keep)
        if remaining == 0:
            return np.sort(old_keep)
        camera = self._live_xyz @ pose[:3, :3].T + pose[:3, 3]
        depth = camera[:, 2]
        reproj = np.full(len(ids), np.inf)
        valid = depth > 1e-6
        reproj[valid] = np.hypot(
            self._fx * camera[valid, 0] / depth[valid] + self._cx - xy[valid, 0],
            self._fy * camera[valid, 1] / depth[valid] + self._cy - xy[valid, 1],
        )
        counts = np.full(len(ids), 2, dtype=int)
        parallax = np.zeros(len(ids))
        for row in np.flatnonzero(ids < 0):
            landmark = self._workmap_landmarks.get(int(ids[row]))
            if landmark is not None:
                counts[row] = int(landmark.get("valid_observation_count", 2))
                parallax[row] = float(landmark.get("parallax_deg", 0.0))
        # Reserve a slot so a high-count all-positive track set cannot starve
        # a newly triangulated point. Old-map support remains at the strong floor.
        new = np.flatnonzero((ids < 0) & ~selected & np.isfinite(reproj))
        if len(new) and remaining:
            best = new[np.lexsort((reproj[new], -parallax[new], -counts[new]))[0]]
            selected[best] = True
            remaining -= 1
        if remaining:
            candidates = np.flatnonzero(~selected & np.isfinite(reproj))
            ranked = candidates[
                np.lexsort((reproj[candidates], -parallax[candidates], -counts[candidates]))
            ]
            spread = spatial_indices(xy[ranked], remaining, self._w, self._h)
            selected[ranked[spread]] = True
        return np.flatnonzero(selected)

    def _augment_live_points(self, gray: np.ndarray, reseeded: bool) -> None:
        """Allow subclasses to add measured live points before PnP."""

    def _admit_new_landmarks(self, world: np.ndarray, good: np.ndarray) -> np.ndarray:
        """Apply a subclass region gate before allocating session landmark IDs."""
        return good

    def _live_cap_policy(self) -> str:
        """``frozen_first``: positive map points outrank VO; ``spread``: reserve then spread."""
        return "spread" if self._online_workmap_active else "frozen_first"

    def _vo_keyframe(self, gray: np.ndarray, cam_from_world: np.ndarray) -> None:
        self._finish_pending_detection()
        vo = self.profile.vo
        mature = self._vo_bid <= self._vo_bid_next - int(vo.batch_lag)
        new_xy: list[np.ndarray] = []
        new_ids: list[np.ndarray] = []
        new_xyz: list[np.ndarray] = []
        current_admissible = self._is_pose_admissible(cam_from_world)
        for bid in np.unique(self._vo_bid[mature]) if mature.any() else ():
            rows = self._vo_bid == bid
            world, good = self._triangulate_batch(
                self._vo_seed_xy[rows],
                self._vo_pose_by_bid[int(bid)],
                self._vo_xy[rows],
                cam_from_world,
            )
            seed_meta = self._get_seed_provenance(int(bid))
            eligible_source = (
                seed_meta is not None
                and seed_meta.source_epoch == self._source_epoch
                and seed_meta.admissible
            )
            can_promote = (
                self._online_workmap_active
                and eligible_source
                and current_admissible
            )
            promote_mask = (
                self._admit_new_landmarks(world, good.copy())
                if can_promote
                else np.zeros_like(good)
            )
            if good.any():
                add_xy = self._vo_xy[rows][good]
                add_xyz = world[good]
                count = int(good.sum())
                add_ids = np.arange(self._vo_next_id, self._vo_next_id - count, -1, dtype=np.int64)
                self._vo_next_id -= count
                new_xy.append(add_xy)
                new_ids.append(add_ids)
                if can_promote and promote_mask[good].any():
                    promoted_subset = promote_mask[good]
                    self._register_vo_landmarks(
                        int(bid),
                        add_ids[promoted_subset],
                        add_xyz[promoted_subset],
                        self._vo_seed_xy[rows][good][promoted_subset],
                        add_xy[promoted_subset],
                        cam_from_world,
                    )
                new_xyz.append(add_xyz)
            self._vo_pose_by_bid.pop(int(bid), None)
            self._vo_seed_meta.pop(int(bid), None)

        if new_xy:
            all_add_xy = np.concatenate(new_xy, axis=0)
            all_add_ids = np.concatenate(new_ids, axis=0)
            all_add_xyz = np.concatenate(new_xyz, axis=0)
            self._live_xy = np.ascontiguousarray(
                np.concatenate([self._live_xy, all_add_xy], axis=0)
                if self._live_xy.size
                else all_add_xy,
                dtype=np.float64,
            )
            self._live_ids = (
                np.concatenate([self._live_ids, all_add_ids], axis=0)
                if self._live_ids.size
                else all_add_ids
            )
            self._live_xyz = (
                np.concatenate([self._live_xyz, all_add_xyz], axis=0)
                if self._live_xyz.size
                else all_add_xyz
            )
        keep_vo = ~mature
        self._vo_xy = self._vo_xy[keep_vo]
        self._vo_seed_xy = self._vo_seed_xy[keep_vo]
        self._vo_bid = self._vo_bid[keep_vo]

        if len(self._live_xy) < int(vo.min_live):
            occupied = (
                np.concatenate([self._live_xy, self._vo_xy]) if self._vo_xy.size else self._live_xy
            )
            pose = np.asarray(cam_from_world, dtype=float).copy()
            provenance = (
                SeedProvenance(
                    int(self._ordinal),
                    float(self._previous_capture_stamp),
                    int(self._source_epoch),
                    bool(current_admissible),
                )
                if self._online_workmap_active
                else None
            )
            if self._vo_detection_deferrable():
                # The corner search (mask included) runs behind the foreground on a copy of
                # this frame's tracks and is merged at the start of the next step.
                executor = getattr(self, "_detect_executor", None)
                if executor is None:
                    executor = self._detect_executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="vo-detect"
                    )
                self._pending_detection = _PendingDetection(
                    executor.submit(
                        _detect_corners, gray, occupied.copy(),
                        int(vo.detect_cap), int(vo.min_distance),
                    ),
                    pose,
                    provenance,
                )
            else:
                self._register_seed_batch(self._detect_features(gray, occupied), pose, provenance)

        cap = int(self.profile.fast_loop.track_cap)
        if len(self._live_xy) > cap:
            if self._live_cap_policy() == "spread":
                order = self._select_workmap_live_points(cap, cam_from_world)
            else:
                # Frozen baseline policy: positive map points outrank VO.
                order = np.argsort(-self._live_ids, kind="stable")[:cap]
            self._live_xy = np.ascontiguousarray(self._live_xy[order], dtype=np.float64)
            self._live_ids = self._live_ids[order]
            self._live_xyz = self._live_xyz[order]

    def _sample_rgb(self, xy: np.ndarray) -> np.ndarray | None:
        """Original colour at fast-loop pixels of the current frame; None without a colour frame."""
        return None

    def _register_vo_landmarks(
        self,
        bid: int,
        add_ids: np.ndarray,
        add_xyz: np.ndarray,
        seed_xy: np.ndarray,
        add_xy: np.ndarray,
        cam_from_world: np.ndarray,
    ) -> None:
        """Record freshly triangulated VO points as two-view workmap landmarks."""
        meta = self._vo_seed_meta[bid]
        seed_ordinal = meta[0]
        seed_stamp = meta[1]
        seed_pose = self._vo_pose_by_bid[bid]
        current_pose = np.asarray(cam_from_world, dtype=float).copy()
        current_pose.setflags(write=False)
        seed_rays = add_xyz - _center(seed_pose)
        current_rays = add_xyz - _center(cam_from_world)
        ray_norms = np.linalg.norm(seed_rays, axis=1) * np.linalg.norm(current_rays, axis=1)
        angles = np.degrees(
            np.arccos(
                np.clip(np.einsum("ij,ij->i", seed_rays, current_rays) / ray_norms, -1.0, 1.0)
            )
        )
        colors = self._sample_rgb(add_xy)
        for row, (key, point, first_xy, current_xy, parallax_deg) in enumerate(
            zip(add_ids, add_xyz, seed_xy, add_xy, angles, strict=True)
        ):
            observed_xy = current_xy.copy()
            self._workmap_landmarks[int(key)] = {
                "xyz": point.copy(),
                "parallax_deg": float(parallax_deg),
                "revision": 0,
                **({} if colors is None else {"rgb": colors[row].tolist()}),
                "observations": (
                    (seed_ordinal, seed_stamp, first_xy.copy(), seed_pose),
                    (self._ordinal, self._previous_capture_stamp, observed_xy, current_pose),
                ),
                "last_inlier": (self._ordinal, self._previous_capture_stamp, observed_xy),
            }
            self._workmap_pending_ids.add(int(key))

    def _triangulate_batch(
        self,
        seed_xy: np.ndarray,
        seed_pose: np.ndarray,
        cur_xy: np.ndarray,
        cur_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World points from two known poses. Scale comes from the poses, so a
        VO point inherits the map's scale instead of inventing its own."""
        if seed_xy.shape[0] == 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=bool)
        fx, fy, cx, cy = self._fx, self._fy, self._cx, self._cy
        p_seed = self._K @ np.asarray(seed_pose, dtype=float)
        p_cur = self._K @ np.asarray(cur_pose, dtype=float)
        homogeneous = cv2.triangulatePoints(
            np.ascontiguousarray(p_seed, dtype=np.float64),
            np.ascontiguousarray(p_cur, dtype=np.float64),
            np.ascontiguousarray(seed_xy.T, dtype=np.float64),
            np.ascontiguousarray(cur_xy.T, dtype=np.float64),
        )
        w = homogeneous[3]
        good = np.abs(w) > 1e-12
        world = np.zeros((seed_xy.shape[0], 3), dtype=float)
        world[good] = (homogeneous[:3, good] / w[good]).T
        max_reproj = float(self.profile.vo.max_reproj_px)
        for pose, xy in ((seed_pose, seed_xy), (cur_pose, cur_xy)):
            pose = np.asarray(pose, dtype=float)
            camera = world @ pose[:3, :3].T + pose[:3, 3]
            depth = camera[:, 2]
            good &= depth > 1e-3
            safe = np.where(good, depth, 1.0)
            u = fx * camera[:, 0] / safe + cx
            v = fy * camera[:, 1] / safe + cy
            good &= np.hypot(u - xy[:, 0], v - xy[:, 1]) <= max_reproj
        # parallax: a nearly-zero angle triangulates to noise
        ray_a = world - _center(seed_pose)
        ray_b = world - _center(cur_pose)
        norm = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
        cosine = np.einsum("ij,ij->i", ray_a, ray_b) / np.where(norm > 0, norm, 1.0)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        good &= angle >= float(self.profile.vo.min_parallax_deg)
        return world, good

    def _detect_features(self, gray: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        vo = self.profile.vo
        return _detect_corners(gray, occupied, int(vo.detect_cap), int(vo.min_distance))

    def _vo_detection_deferrable(self) -> bool:
        """True when this frame's seeds may join the VO arrays at the next step."""
        return False

    def _register_seed_batch(
        self, seeded: np.ndarray, pose: np.ndarray, provenance: Any
    ) -> None:
        if not seeded.size:
            return
        self._vo_pose_by_bid[self._vo_bid_next] = pose
        if provenance is not None:
            self._vo_seed_meta[self._vo_bid_next] = provenance
        self._vo_xy = np.concatenate([self._vo_xy, seeded]) if self._vo_xy.size else seeded
        self._vo_seed_xy = (
            np.concatenate([self._vo_seed_xy, seeded.copy()])
            if self._vo_seed_xy.size
            else seeded.copy()
        )
        new_bid = np.full(len(seeded), self._vo_bid_next, dtype=np.int64)
        self._vo_bid = np.concatenate([self._vo_bid, new_bid]) if self._vo_bid.size else new_bid
        self._vo_bid_next += 1

    def _finish_pending_detection(self) -> None:
        pending = getattr(self, "_pending_detection", None)
        if pending is None:
            return
        self._pending_detection = None
        self._register_seed_batch(pending.future.result(), pending.pose, pending.provenance)

    def _discard_pending_detection(self) -> None:
        pending = getattr(self, "_pending_detection", None)
        self._pending_detection = None
        if pending is not None:
            _wait_futures([pending.future])

    def _recover_relative(
        self,
        prev_xy: np.ndarray | None,
        curr_xy: np.ndarray | None,
        prev_pose: np.ndarray,
        step_len: float,
    ) -> tuple[np.ndarray | None, int]:
        """Calibrated essential-matrix step. Translation magnitude from recent motion."""
        cv2.setRNGSeed(0)
        if prev_xy is None or curr_xy is None:
            return None, 0
        prev_xy = np.asarray(prev_xy, dtype=float).reshape(-1, 2)
        curr_xy = np.asarray(curr_xy, dtype=float).reshape(-1, 2)
        if len(prev_xy) < 8 or len(curr_xy) != len(prev_xy):
            return None, 0
        # Median KLT flow of this pair. Near zero means hover / static scene:
        # parallax vanishes, the essential matrix is degenerate, and
        # recoverPose then assembles a ~90 deg garbage rotation out of noise.
        flow_med = float(np.median(np.linalg.norm(curr_xy - prev_xy, axis=1)))
        matrix, mask = cv2.findEssentialMat(
            prev_xy,
            curr_xy,
            cameraMatrix=self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
            maxIters=DR_ESSENTIAL_MAX_ITERS,
        )
        if matrix is None:
            return None, 0
        e_inliers = int(np.count_nonzero(mask)) if mask is not None else 0
        if flow_med < 1.0 and e_inliers < 8:
            # Hover guard: refuse the DR update. Returning None leaves the
            # caller on NO_POSE without ageing dead_reckon_age or writing
            # last_pose, so the next frame retries from the same prior.
            self._dead_reckon_guarded += 1
            return None, e_inliers
        n_inliers, rotation, translation, _ = cv2.recoverPose(
            matrix,
            prev_xy,
            curr_xy,
            cameraMatrix=self._K,
            mask=mask,
        )
        n_inliers = int(n_inliers)
        if n_inliers < 8:
            return None, n_inliers
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float).reshape(3)
        norm = float(np.linalg.norm(translation))
        prev_r = np.asarray(prev_pose, dtype=float)[:3, :3]
        prev_t = np.asarray(prev_pose, dtype=float)[:3, 3]
        # Corner downweight: DR extrapolates with the recent step length, i.e.
        # it assumes locally constant velocity. A single-frame heading jump
        # breaks that prior, so halve the step (conservative extrapolation).
        step_scale = 1.0
        if azimuth_turn_exceeds(self.map_frame, prev_r, rotation):
            step_scale = 0.5
            self._dead_reckon_guarded += 1
        scaled = np.zeros(3) if norm < 1e-9 else translation / norm * (float(step_len) * step_scale)
        return (
            np.concatenate([rotation @ prev_r, (rotation @ prev_t + scaled).reshape(3, 1)], axis=1),
            n_inliers,
        )

    def _inside(self, xy: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < self._w)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < self._h)
        )


def _scaled_intrinsics(
    profile: "DirectProfile", width: int, height: int
) -> tuple[float, float, float, float]:
    """Fast-loop PINHOLE parameters at the fast-loop resolution.

    The vendored `scaled_pinhole_parameters` is the single definition of how a
    calibration is rescaled; the relocalizer path uses the same function, so the
    fast loop and the provider can never disagree about the query camera.
    """
    from direct_paths import vendor_sys_path

    vendor_sys_path()
    from sfm_diagnosis.site_pipeline.deployment_localizer import (
        scaled_pinhole_parameters,
    )

    return scaled_pinhole_parameters(
        profile.intrinsics.as_calibration(), width=width, height=height
    )
