#!/usr/bin/env python3
"""Two-rate direct localizer: CPU fast loop + background GPU relocalizer.

Fast loop (every frame, must fit the stream interval): pyramidal KLT carries the
live map-point set and the un-triangulated VO candidates one frame forward, then
pycolmap RANSAC PnP solves the absolute pose. Pure CPU, no model, no GPU.

Relocalizer (`RelocWorker`, one background thread): MegaLoc retrieval + official
EDM matching + PnP on a single frame through `LiveMapEDMProvider.localize_array`.
Its anchor 2D points are chained forward through every frame the worker was busy
for, then handed over to the fast loop. The fast loop NEVER waits for it:
`RelocWorker.submit` refuses work while the thread is busy instead of queueing.

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
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

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

    def set_frame(self, gray: np.ndarray) -> None:
        self._prev_gray = gray

    def reset(self) -> None:
        self._prev_gray = None

    def track_pair(
        self, prev_gray: np.ndarray, cur_gray: np.ndarray, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._lk(prev_gray, cur_gray, xy)

    def track(self, gray: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        previous = self._prev_gray
        self._prev_gray = gray
        if previous is None:
            return xy, np.zeros(len(xy), dtype=bool)
        return self._lk(previous, gray, xy)

    # NOTE: pre-built pyramids (cv2.buildOpticalFlowPyramid) would remove three
    # of the four pyramid builds per frame, but the Python binding rejects a
    # pyramid list for `prevImg`, so the grey images are passed directly.
    def _lk(
        self, previous: np.ndarray, current: np.ndarray, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if xy.size == 0:
            return xy, np.zeros(len(xy), dtype=bool)
        p0 = xy.astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, p0, None, winSize=(self.win, self.win),
            maxLevel=self.levels, criteria=self.criteria,
        )
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, p1, None, winSize=(self.win, self.win),
            maxLevel=self.levels, criteria=self.criteria,
        )
        fwd = p1.reshape(-1, 2)
        back = p0b.reshape(-1, 2)
        fb = np.linalg.norm(back - xy, axis=1)
        keep = (
            (st.reshape(-1) == 1) & (stb.reshape(-1) == 1)
            & np.isfinite(fwd).all(axis=1) & (fb <= self.fb_max_px)
        )
        return fwd, keep


class RelocWorker:
    """One background thread running the GPU relocalizer, never blocking.

    Capacity is exactly one job: `submit` returns False while the thread is
    busy instead of queueing, because a queued relocalization would land with a
    catch-up chain longer than the ring buffer and be dropped anyway.
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._cv = threading.Condition()
        self._job: tuple[np.ndarray, np.ndarray | None, int] | None = None
        self._busy = False
        self._closed = False
        self._out: "queue.Queue[tuple[RelocFix, int]]" = queue.Queue()
        self._thread: threading.Thread | None = None

    # ---------- lifecycle ----------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._closed = False
        thread = threading.Thread(
            target=self._run, name="direct-reloc-worker", daemon=True
        )
        self._thread = thread
        thread.start()

    def close(self) -> None:
        thread = self._thread
        with self._cv:
            self._closed = True
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
            return self._busy

    def submit(
        self, gray: np.ndarray, color_bgr: np.ndarray | None, ordinal: int
    ) -> bool:
        with self._cv:
            if self._closed or self._thread is None or self._busy:
                return False
            # `gray` is always a fresh cv2.cvtColor output, but the colour frame
            # is the caller's array whenever no resize was needed, so it can be
            # overwritten while the worker is still reading it. Snapshot it.
            self._job = (
                gray,
                None if color_bgr is None else color_bgr.copy(),
                int(ordinal),
            )
            self._busy = True
            self._cv.notify()
        return True

    def poll(self) -> tuple["RelocFix", int] | None:
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    # ---------- worker thread ----------
    def _run(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._closed:
                    self._cv.wait()
                if self._closed:
                    self._busy = False
                    return
                gray, color_bgr, ordinal = self._job
                self._job = None
            started = time.perf_counter()
            try:
                fix = self._provider.localize_array(gray, color_bgr=color_bgr)
            except BaseException as error:  # noqa: BLE001 - must never kill the thread
                fix = _failed_fix(
                    (time.perf_counter() - started) * 1000.0,
                    f"{type(error).__name__}: {error}",
                )
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()
            self._out.put((fix, ordinal))


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
        self._tracker.reset()
        self._reset_tracking_state()

    def _reset_tracking_state(self) -> None:
        self._live_xy = np.zeros((0, 2), dtype=float)
        self._live_ids = np.zeros((0,), dtype=np.int64)
        self._live_xyz = np.zeros((0, 3), dtype=float)
        self._vo_xy = np.zeros((0, 2), dtype=float)
        self._vo_seed_xy = np.zeros((0, 2), dtype=float)
        self._vo_bid = np.zeros((0,), dtype=np.int64)
        self._vo_pose_by_bid: dict[int, np.ndarray] = {}
        self._vo_bid_next = 0
        self._vo_next_id = -1
        self._ring: deque[tuple[int, np.ndarray]] = deque(maxlen=HANDOVER_MAX_FRAMES)
        self._pending_ordinal: int | None = None
        self._last_reloc_stamp: float | None = None
        self._last_pose: np.ndarray | None = None
        self._last_gray: np.ndarray | None = None
        self._last_pose_ordinal = -10**9
        self._prev_center: np.ndarray | None = None
        self._step_history: list[float] = []
        self._dead_reckon_age = 0
        self._last_reloc_status: str | None = None
        self._last_reloc_ms: float | None = None
        self._handover_dropped = 0
        # Cumulative DR-guard hits (hover rejects + yaw downweights). Internal
        # only: direct_localizer_adapter builds last_info from an explicit
        # key whitelist, so a new info-dict key would be silently dropped --
        # per spec the bool key is omitted and only this counter is kept.
        self._dead_reckon_guarded = 0

    # ---------- one frame ----------
    def step(self, bgr_native: np.ndarray, capture_stamp: float) -> dict:
        t0 = time.perf_counter()
        self._ordinal += 1
        ordinal = self._ordinal
        stamp = float(capture_stamp)

        frame = np.asarray(bgr_native)
        if (frame.shape[1], frame.shape[0]) != (self._w, self._h):
            frame = cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._ring.append((ordinal, gray))

        # 1. fast loop: carry the live map-point set and the un-triangulated VO
        #    candidates one frame forward in a single tracker call
        n_live = len(self._live_xy)
        combined = (
            np.concatenate([self._live_xy, self._vo_xy])
            if self._vo_xy.size
            else self._live_xy
        )
        fwd_all, keep_all = self._tracker.track(gray, combined)
        dr_prev = dr_curr = None
        if combined.size:
            ok_all = keep_all & self._inside(fwd_all)
            inside, ok_vo = ok_all[:n_live], ok_all[n_live:]
            if self.profile.dead_reckon.enabled:
                dr_prev = combined[ok_all]
                dr_curr = fwd_all[ok_all]
            self._live_xy = fwd_all[:n_live][inside]
            self._live_ids = self._live_ids[inside]
            self._live_xyz = self._live_xyz[inside]
            self._vo_xy = fwd_all[n_live:][ok_vo]
            self._vo_seed_xy = self._vo_seed_xy[ok_vo]
            self._vo_bid = self._vo_bid[ok_vo]
        t_track = time.perf_counter()

        # 2. worker handover: chain the anchor through the frames the worker
        #    spent relocalizing, then REPLACE (strong) or MERGE (weak).
        reseeded = False
        handover_points = 0
        reloc_status: str | None = None
        reference_names: tuple[str, ...] = ()
        delivered = self._worker.poll()
        if delivered is not None:
            fix, trigger_ordinal = delivered
            self._pending_ordinal = None
            self._last_reloc_status = reloc_status = str(fix.status)
            self._last_reloc_ms = float(fix.runtime_ms)
            reference_names = tuple(str(name) for name in (fix.reference_names or ()))
            handover_points, reseeded = self._apply_handover(fix, trigger_ordinal, gray)

        # 3. absolute pose
        n_corr = int(len(self._live_xy))
        status = "NO_POSE"
        inliers = 0
        map_inliers = 0
        vo_inliers = 0
        center = None
        cam_from_world = None
        pnp = self.profile.fast_loop.pnp
        if n_corr >= int(pnp.min_points):
            answer = self._pycolmap.estimate_and_refine_absolute_pose(
                self._live_xy, self._live_xyz, self._camera,
                self._pnp_estimation, self._pnp_refinement,
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
                    else:
                        status = (
                            "FAST_TRACK"
                            if map_inliers >= int(pnp.min_inliers)
                            else "VO_ONLY"
                        )
                    # keep only the geometric inliers
                    self._live_xy = self._live_xy[mask]
                    self._live_ids = self._live_ids[mask]
                    self._live_xyz = self._live_xyz[mask]
        t_pnp = time.perf_counter()

        # 4. dead reckoning: calibrated essential-matrix step from the last pose
        if cam_from_world is None:
            dead = self.profile.dead_reckon
            if (
                dead.enabled
                and self._last_pose is not None
                and self._last_gray is not None
                and self._dead_reckon_age < int(dead.max_frames)
            ):
                prev_xy, curr_xy = dr_prev, dr_curr
                if (
                    self._last_pose_ordinal != ordinal - 1
                    or prev_xy is None
                    or len(prev_xy) < 8
                ):
                    extra = self._detect_features(
                        self._last_gray, np.zeros((0, 2), dtype=float)
                    )
                    if extra.size:
                        fwd, keep = self._tracker.track_pair(
                            self._last_gray, gray, extra
                        )
                        ok = keep & self._inside(fwd)
                        prev_xy, curr_xy = extra[ok], fwd[ok]
                step_len = (
                    float(np.median(self._step_history[-5:]))
                    if self._step_history
                    else 0.0
                )
                pose, n_inl = self._recover_relative(
                    prev_xy, curr_xy, self._last_pose, step_len
                )
                if pose is not None:
                    cam_from_world = pose
                    center = _center(pose)
                    status = "DEAD_RECKON"
                    inliers = n_inl
                    self._dead_reckon_age += 1
        else:
            self._dead_reckon_age = 0

        # 5. VO keyframes: candidates are seeded every keyframe and triangulated
        #    `batch_lag` keyframes later, because one keyframe of drone motion is
        #    not enough parallax. Scale comes from the poses, so a VO point is in
        #    map units. Nothing here writes to the map.
        vo = self.profile.vo
        if vo.enabled and cam_from_world is not None and ordinal % int(vo.keyframe_stride) == 0:
            self._vo_keyframe(gray, cam_from_world)
        t_end = time.perf_counter()

        # 6. relocalizer trigger
        reloc = self.profile.reloc
        submitted = False
        if self._pending_ordinal is None and not self._worker.busy:
            due = (
                self._last_reloc_stamp is None
                or (stamp - self._last_reloc_stamp) >= float(reloc.period_s)
            )
            if len(self._live_ids) < int(reloc.min_points) or status == "NO_POSE" or due:
                # The frozen BoQ reference bank was extracted from colour
                # keyframes; feeding the relocalizer a grey frame replicated
                # across three channels puts retrieval off the distribution the
                # bank was built on. The colour frame is already in hand here.
                if self._worker.submit(gray, frame, ordinal):
                    self._pending_ordinal = ordinal
                    self._last_reloc_stamp = stamp
                    submitted = True

        # 7. bookkeeping
        step = None
        if center is not None and self._prev_center is not None:
            step = float(np.linalg.norm(center - self._prev_center))
        if center is not None:
            self._prev_center = center
        if step is not None:
            self._step_history.append(step)
            if len(self._step_history) > 15:
                del self._step_history[:-15]
        if cam_from_world is not None:
            self._last_pose = np.asarray(cam_from_world, dtype=float).copy()
            self._last_pose_ordinal = ordinal
        self._last_gray = gray

        rotation = None if cam_from_world is None else cam_from_world[:3, :3]
        return {
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
            "handover_points": int(handover_points),
            "handover_dropped": int(self._handover_dropped),
            "reloc_status": reloc_status if reloc_status is not None else self._last_reloc_status,
            "reloc_delivered": delivered is not None,
            "reloc_submitted": submitted,
            "reloc_busy": self._worker.busy,
            "reference_names": reference_names,
            "step": step,
            "n_corr": n_corr,
            "loop_ms": (t_end - t0) * 1000.0,
            "track_ms": (t_track - t0) * 1000.0,
            "pnp_ms": (t_pnp - t_track) * 1000.0,
            "vo_ms": (t_end - t_pnp) * 1000.0,
            "reloc_ms": self._last_reloc_ms,
        }

    # ---------- handover ----------
    def _apply_handover(
        self, fix: "RelocFix", trigger_ordinal: int, gray: np.ndarray
    ) -> tuple[int, bool]:
        if not fix.ok or fix.cam_from_world is None:
            return 0, False
        ids = np.asarray(fix.point3d_ids, dtype=np.int64)
        xy = np.asarray(fix.query_xy, dtype=float)
        xyz = np.asarray(fix.point_xyz, dtype=float)
        if ids.size < int(self.profile.fast_loop.pnp.min_points):
            return 0, False

        chain = [frame for ordinal, frame in self._ring if ordinal >= trigger_ordinal]
        first = next(
            (ordinal for ordinal, _ in self._ring if ordinal >= trigger_ordinal), None
        )
        if first != trigger_ordinal or len(chain) > HANDOVER_MAX_FRAMES:
            # the worker outran the ring buffer: the anchor's 2D can no longer
            # be walked to this frame, so the whole handover is discarded
            self._handover_dropped += 1
            return 0, False

        cap = int(self.profile.fast_loop.track_cap)
        if len(ids) > cap:
            take = np.linspace(0, len(ids) - 1, cap).astype(int)
            ids, xy, xyz = ids[take], xy[take], xyz[take]
        n = len(chain)
        # Stride-2 catch-up: one track_pair per two frames instead of one per
        # frame. Pairs fall from (n-1) to ceil((n-1)/2) -- about half the KLT
        # calls on the delivery peak. A gap-2 hop tracks frame a straight to
        # frame b; the skipped frame's 2D is implicitly its linear midpoint
        # (constant velocity over 2 frames), so no state is stored for it --
        # the walk only lives on tracked frames. Indexing starts at 0, hence
        # the anchor frame is always tracked: odd n lands exactly on the last
        # frame, even n ends with one gap-1 hop onto the current frame.
        # Risk: a gap-2 hop doubles the per-call displacement, so fast-motion
        # handovers lose more points to the FB gate and fall through to MERGE
        # (or empty) -- reseed rate is the validation metric for this trade.
        idx = list(range(0, n, 2))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        for a, b in zip(idx[:-1], idx[1:]):
            if xy.size == 0:
                break
            fwd, keep = self._tracker.track_pair(chain[a], chain[b], xy)
            inside = keep & self._inside(fwd)
            xy, ids, xyz = fwd[inside], ids[inside], xyz[inside]
        if ids.size == 0:
            return 0, True

        if ids.size >= int(self.profile.fast_loop.reseed_min_points):
            # Strong handover: REPLACE. Merging lets an old drifted majority
            # outvote the fresh map-derived 2D, so relocalization stops
            # correcting anything -- measured, not hypothetical.
            self._live_xy, self._live_ids, self._live_xyz = xy, ids, xyz
        else:
            fresh = ~np.isin(ids, self._live_ids)
            xy, ids, xyz = xy[fresh], ids[fresh], xyz[fresh]
            self._live_xy = np.concatenate([self._live_xy, xy]) if self._live_xy.size else xy
            self._live_ids = (
                np.concatenate([self._live_ids, ids]) if self._live_ids.size else ids
            )
            self._live_xyz = (
                np.concatenate([self._live_xyz, xyz]) if self._live_xyz.size else xyz
            )
        return int(ids.size), True

    # ---------- VO ----------
    def _vo_keyframe(self, gray: np.ndarray, cam_from_world: np.ndarray) -> None:
        vo = self.profile.vo
        mature = self._vo_bid <= self._vo_bid_next - int(vo.batch_lag)
        for bid in np.unique(self._vo_bid[mature]) if mature.any() else ():
            rows = self._vo_bid == bid
            world, good = self._triangulate_batch(
                self._vo_seed_xy[rows], self._vo_pose_by_bid[int(bid)],
                self._vo_xy[rows], cam_from_world,
            )
            if good.any():
                add_xy = self._vo_xy[rows][good]
                add_xyz = world[good]
                count = int(good.sum())
                add_ids = np.arange(
                    self._vo_next_id, self._vo_next_id - count, -1, dtype=np.int64
                )
                self._vo_next_id -= count
                self._live_xy = (
                    np.concatenate([self._live_xy, add_xy]) if self._live_xy.size else add_xy
                )
                self._live_ids = (
                    np.concatenate([self._live_ids, add_ids])
                    if self._live_ids.size
                    else add_ids
                )
                self._live_xyz = (
                    np.concatenate([self._live_xyz, add_xyz])
                    if self._live_xyz.size
                    else add_xyz
                )
            self._vo_pose_by_bid.pop(int(bid), None)
        keep_vo = ~mature
        self._vo_xy = self._vo_xy[keep_vo]
        self._vo_seed_xy = self._vo_seed_xy[keep_vo]
        self._vo_bid = self._vo_bid[keep_vo]

        if len(self._live_xy) < int(vo.min_live):
            occupied = (
                np.concatenate([self._live_xy, self._vo_xy])
                if self._vo_xy.size
                else self._live_xy
            )
            seeded = self._detect_features(gray, occupied)
            if seeded.size:
                self._vo_pose_by_bid[self._vo_bid_next] = np.asarray(
                    cam_from_world, dtype=float
                ).copy()
                self._vo_xy = (
                    np.concatenate([self._vo_xy, seeded]) if self._vo_xy.size else seeded
                )
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
            self._live_xy = self._live_xy[order]
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
        for point in occupied:
            cv2.circle(mask, (int(point[0]), int(point[1])), min_distance, 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=int(vo.detect_cap), qualityLevel=0.01,
            minDistance=min_distance, mask=mask,
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
            prev_xy, curr_xy, cameraMatrix=self._K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
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
            matrix, prev_xy, curr_xy, cameraMatrix=self._K, mask=mask,
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
        scaled = (
            np.zeros(3)
            if norm < 1e-9
            else translation / norm * (float(step_len) * step_scale)
        )
        return (
            np.concatenate(
                [rotation @ prev_r, (rotation @ prev_t + scaled).reshape(3, 1)], axis=1
            ),
            n_inliers,
        )

    def _inside(self, xy: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0) & (xy[:, 0] < self._w)
            & (xy[:, 1] >= 0) & (xy[:, 1] < self._h)
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
