#!/usr/bin/env python3
"""Two-rate direct localizer: CPU fast loop + background GPU relocalizer.

Fast loop (every frame, must fit the stream interval): pyramidal KLT carries the
live map-point set and the un-triangulated VO candidates one frame forward, then
pycolmap RANSAC PnP solves the absolute pose. Pure CPU, no model, no GPU.

Relocalizer (`RelocWorker`, one background thread): MegaLoc retrieval + official
EDM matching + PnP on a single frame through `LiveMapEDMProvider.localize_array`.
Its anchor 2D points are chained forward through every frame the worker was busy
for, then handed over to the fast loop. The fast loop NEVER waits for it:
`RelocWorker.submit` retains only the newest waiting capture while busy.

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
import queue
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

from point_quality import pose_quality, spatial_indices

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
    """One background thread running the GPU relocalizer, never blocking.

    One job may run and one latest capture may wait. New submissions replace
    the waiting capture, so triggers are retained without an aging backlog.
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._cv = threading.Condition()
        self._job: tuple[np.ndarray, int, int, float] | None = None
        self._busy = False
        self._closed = False
        self._out: "queue.Queue[RelocResult]" = queue.Queue(maxsize=1)
        self._generation = 0
        self._thread: threading.Thread | None = None

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
            self._job = None
            self._cv.notify_all()
        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive():
                raise RuntimeError("relocalizer thread did not stop within 30 s")
        self._thread = None

    # ---------- fast-loop facing ----------
    @property
    def busy(self) -> bool:
        with self._cv:
            return self._busy or self._job is not None

    @property
    def queued(self) -> bool:
        with self._cv:
            return self._job is not None

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
            self._job = (gray.copy(), int(ordinal), int(source_epoch), float(capture_stamp))
            self._cv.notify()
        return True

    def poll(self) -> "RelocResult | None":
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def reset(self) -> None:
        """Invalidate in-flight work as well as pending and undelivered fixes."""
        with self._cv:
            self._generation += 1
            self._job = None
            while True:
                try:
                    self._out.get_nowait()
                except queue.Empty:
                    break

    # ---------- worker thread ----------
    def _run(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._closed:
                    self._cv.wait()
                if self._closed:
                    self._busy = False
                    return
                gray, ordinal, source_epoch, capture_stamp = self._job
                self._job = None
                generation = self._generation
                self._busy = True
            started = time.perf_counter()
            try:
                fix = self._provider.localize_array(gray)
            except BaseException as error:  # noqa: BLE001 - must never kill the thread
                fix = _failed_fix(
                    (time.perf_counter() - started) * 1000.0,
                    f"{type(error).__name__}: {error}",
                )
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()
            self._publish(fix, ordinal, capture_stamp, source_epoch, generation)

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


class TwoRateTracker:
    """Per-frame fast loop with an asynchronous map relocalizer behind it."""

    def __init__(
        self,
        assets: "DirectMapAssets",
        profile: "DirectProfile",
        provider: Any,
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
        self._worker = RelocWorker(provider)

        self._ordinal = -1
        self._source_epoch = 0
        self._models_ready = False
        self._reset_tracking_state()

    # ---------- lifecycle ----------
    def ensure_models(self) -> None:
        """Load the GPU models and start the worker, outside the timed loop."""
        if not self._models_ready:
            self.provider.ensure_models()
            self._models_ready = True
        self._worker.start()

    def close(self) -> None:
        self._worker.close()

    def reset(self) -> None:
        """Drop every tracking prior. The map assets and models are untouched."""
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
        self._vo_bid_next = 0
        self._vo_next_id = -1
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
        idle = status == "FAST_TRACK" and _reloc_idle(self._step_history)
        due = _reloc_refresh_due(status, stamp, self._last_reloc_stamp, float(reloc.period_s), idle)
        # VO tracks can be numerous while none constrain the pose to the map.
        # Request a map fix as soon as the one-job worker is available, without
        # waiting for the normal refresh period or clearing the usable tracks.
        unconfirmed = status in {"VO_ONLY", "DEAD_RECKON", "IMU_BRIDGE", "NO_POSE"}
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
        strong_inliers = int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        reseeded = int(walked_ids.size) >= max(
            strong_inliers, self.profile.fast_loop.reseed_min_points
        )
        handover_points = int(walked_ids.size)
        if reseeded:
            self._live_xy = np.ascontiguousarray(walked_xy, dtype=np.float64)
            self._live_ids = walked_ids
            self._live_xyz = walked_xyz
            self._tracker.set_frame(gray)
            dr_prev = dr_curr = None
            t_track = time.perf_counter()
        else:
            dr_prev, dr_curr, t_track = self._step_fast_loop(gray)
            if walked_ids.size and len(self._live_ids) >= self.profile.fast_loop.pnp.min_points:
                handover_points = self._merge_handover_points(walked_ids, walked_xy, walked_xyz)
        return reseeded, handover_points, dr_prev, dr_curr, t_track

    def step(self, gray_native: np.ndarray, capture_stamp: float) -> dict:
        t0 = time.perf_counter()
        self._ordinal += 1
        ordinal = self._ordinal
        stamp = float(capture_stamp)
        frame_dt = (
            None if self._previous_capture_stamp is None else stamp - self._previous_capture_stamp
        )
        self._previous_capture_stamp = stamp
        self._klt_diagnostics = {}

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

        # 1. worker handover: poll and catch-up walk, do not write _live_* yet
        (
            walked_ids,
            walked_xy,
            walked_xyz,
            reloc_status,
            reference_names,
            delivered,
        ) = self._step_handover(gray, ordinal, stamp)
        t_handover = time.perf_counter()

        # A partial handover can augment a viable track, but cannot seed an empty one.
        reseeded, handover_points, dr_prev, dr_curr, t_track = self._apply_handover(
            gray, walked_ids, walked_xy, walked_xyz
        )
        # 3. absolute pose
        status, cam_from_world, center, inliers, map_inliers, vo_inliers, n_corr, quality = (
            self._step_absolute_pose(reseeded, stamp)
        )
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
        if vo.enabled and cam_from_world is not None and ordinal % int(vo.keyframe_stride) == 0:
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
            "pnp_ms": (t_pnp - t_track) * 1000.0,
            "vo_ms": (t_vo - t_pnp) * 1000.0,
            "bookkeeping_ms": (t_end - t_vo) * 1000.0,
            "reloc_ms": self._last_reloc_ms,
            "reloc_retrieval_ms": self._last_reloc_stages.get("retrieval_ms"),
            "reloc_match_lift_ms": self._last_reloc_stages.get("match_lift_ms"),
            "reloc_pnp_ms": self._last_reloc_stages.get("reloc_pnp_ms"),
            "reloc_reference_count": self._last_reloc_stages.get("matched_references"),
            "reloc_capture_stamp_mono": self._last_reloc_capture_stamp,
            "reloc_source_epoch": self._last_reloc_source_epoch,
            "reloc_ordinal": self._last_reloc_ordinal,
        }

    def _step_absolute_pose(
        self, reseeded: bool, stamp: float
    ) -> tuple[str, np.ndarray | None, np.ndarray | None, int, int, int, int, dict]:
        """PnP the live set; keep only geometric inliers for the next frame."""
        n_corr = int(len(self._live_xy))
        status = "NO_POSE"
        inliers = 0
        map_inliers = 0
        vo_inliers = 0
        center = None
        cam_from_world = None
        quality: dict = {}
        pnp = self.profile.fast_loop.pnp
        if n_corr >= int(pnp.min_points):
            answer = self._pycolmap.estimate_and_refine_absolute_pose(
                self._live_xy,
                self._live_xyz,
                self._camera,
                self._pnp_estimation,
                self._pnp_refinement,
            )
            if answer is not None:
                mask = np.asarray(answer["inlier_mask"], dtype=bool)
                inliers = int(mask.sum())
                if inliers >= int(pnp.min_inliers):
                    matrix = np.asarray(answer["cam_from_world"].matrix(), dtype=float)
                    cam_from_world = matrix[:3, :4]
                    center = _center(cam_from_world)
                    map_inliers = int(np.count_nonzero(self._live_ids[mask] > 0))
                    vo_inliers = inliers - map_inliers
                    if reseeded:
                        status = "RELOC_SEED"
                    elif map_inliers >= int(pnp.min_inliers):
                        status = "FAST_TRACK"
                    else:
                        status = "VO_ONLY"
                    self._live_xy = np.ascontiguousarray(self._live_xy[mask], dtype=np.float64)
                    self._live_ids = self._live_ids[mask]
                    self._live_xyz = self._live_xyz[mask]
                    quality = pose_quality(
                        self._live_xy, self._live_xyz, cam_from_world, self._K, self._w, self._h
                    )
                    elapsed = stamp - self._last_observation_sample_stamp
                    if elapsed >= 1.0 or (reseeded and elapsed >= 0.5):
                        # Bounded samples for offline residual/covariance analysis.
                        # Positive IDs are map anchors; negative IDs are VO points.
                        selected = np.linspace(
                            0, len(self._live_ids) - 1, min(128, len(self._live_ids)), dtype=int
                        )
                        quality["pnp_observation_sample"] = {
                            "capture_stamp_mono": stamp,
                            "total_inliers": int(inliers),
                            "sample_count": len(selected),
                            "ids": self._live_ids[selected].tolist(),
                            "image_xy_px": self._live_xy[selected].tolist(),
                            "world_xyz_u": self._live_xyz[selected].tolist(),
                            "K": self._K.tolist(),
                            "image_size_px": [self._w, self._h],
                            "cam_from_world": cam_from_world.tolist(),
                            "sampling": "uniform_index; at most 128 inliers",
                        }
                        self._last_observation_sample_stamp = stamp
                    if map_inliers >= int(pnp.min_inliers):
                        self._last_map_stamp = stamp
        return status, cam_from_world, center, inliers, map_inliers, vo_inliers, n_corr, quality

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
        ids = np.asarray(fix.point3d_ids, dtype=np.int64)
        xy = np.asarray(fix.query_xy, dtype=float)
        xyz = np.asarray(fix.point_xyz, dtype=float)
        if ids.size < int(self.profile.fast_loop.pnp.min_points):
            return empty_ids, empty_xy, empty_xyz

        if trigger_epoch is None:
            trigger_epoch = self._source_epoch
        if current_epoch is None:
            current_epoch = self._source_epoch
        entries = [
            entry
            for entry in self._ring
            if entry[0] == trigger_epoch and entry[1] >= trigger_ordinal
        ]
        first = entries[0][1] if entries else None
        if first != trigger_ordinal or trigger_epoch != current_epoch:
            self._handover_dropped += 1
            return empty_ids, empty_xy, empty_xyz
        if trigger_stamp is None:
            trigger_stamp = entries[0][2]
        if current_stamp is None:
            current_stamp = entries[-1][2]
        if current_ordinal is not None and entries[-1][1] != current_ordinal:
            self._handover_dropped += 1
            return empty_ids, empty_xy, empty_xyz
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
            self._handover_dropped += 1
            return empty_ids, empty_xy, empty_xyz
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
    def _vo_keyframe(self, gray: np.ndarray, cam_from_world: np.ndarray) -> None:
        vo = self.profile.vo
        mature = self._vo_bid <= self._vo_bid_next - int(vo.batch_lag)
        new_xy: list[np.ndarray] = []
        new_ids: list[np.ndarray] = []
        new_xyz: list[np.ndarray] = []
        for bid in np.unique(self._vo_bid[mature]) if mature.any() else ():
            rows = self._vo_bid == bid
            world, good = self._triangulate_batch(
                self._vo_seed_xy[rows],
                self._vo_pose_by_bid[int(bid)],
                self._vo_xy[rows],
                cam_from_world,
            )
            if good.any():
                add_xy = self._vo_xy[rows][good]
                add_xyz = world[good]
                count = int(good.sum())
                add_ids = np.arange(self._vo_next_id, self._vo_next_id - count, -1, dtype=np.int64)
                self._vo_next_id -= count
                new_xy.append(add_xy)
                new_ids.append(add_ids)
                new_xyz.append(add_xyz)
            self._vo_pose_by_bid.pop(int(bid), None)

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
            seeded = self._detect_features(gray, occupied)
            if seeded.size:
                self._vo_pose_by_bid[self._vo_bid_next] = np.asarray(
                    cam_from_world, dtype=float
                ).copy()
                self._vo_xy = np.concatenate([self._vo_xy, seeded]) if self._vo_xy.size else seeded
                self._vo_seed_xy = (
                    np.concatenate([self._vo_seed_xy, seeded.copy()])
                    if self._vo_seed_xy.size
                    else seeded.copy()
                )
                new_bid = np.full(len(seeded), self._vo_bid_next, dtype=np.int64)
                self._vo_bid = (
                    np.concatenate([self._vo_bid, new_bid]) if self._vo_bid.size else new_bid
                )
                self._vo_bid_next += 1

        cap = int(self.profile.fast_loop.track_cap)
        if len(self._live_xy) > cap:
            # PnP and the tracker are both linear in point count, so the cap is
            # the throughput knob. Map points outrank VO points.
            order = np.argsort(-self._live_ids, kind="stable")[:cap]
            self._live_xy = np.ascontiguousarray(self._live_xy[order], dtype=np.float64)
            self._live_ids = self._live_ids[order]
            self._live_xyz = self._live_xyz[order]

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
        min_distance = int(vo.min_distance)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if occupied.size:
            pts = np.asarray(occupied, dtype=int)
            valid = (
                (pts[:, 0] >= 0)
                & (pts[:, 0] < gray.shape[1])
                & (pts[:, 1] >= 0)
                & (pts[:, 1] < gray.shape[0])
            )
            pts = pts[valid]
            if len(pts):
                mask[pts[:, 1], pts[:, 0]] = 0
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * min_distance + 1, 2 * min_distance + 1)
                )
                mask = cv2.erode(mask, kernel)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(vo.detect_cap),
            qualityLevel=0.01,
            minDistance=min_distance,
            mask=mask,
        )
        if corners is None:
            return np.zeros((0, 2), dtype=float)
        return corners.reshape(-1, 2).astype(float)

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
