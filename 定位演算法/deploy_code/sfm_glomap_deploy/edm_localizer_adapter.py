#!/usr/bin/env python3
"""ProductionEDMTracker behind the ProductionXFeatTracker API.

live_localizer_worker (and the operator UI behind it) talks to the localizer through a
small, stable surface: localize_frame(rgb) -> Pose, last_info, ensure_models(), and a
RuntimeState it may seed at a frame boundary for the UI's global/weak/track benchmark
modes. The EDM tracker runs the same state machine (BOOT_INIT / TRACK / WEAK_TRACK /
LOST) but a different call shape -- BGR in, dict out, its own state object -- so this is
the only place the two conventions meet. Nothing upstream of it changes.
"""
from __future__ import annotations

import math
import os
import threading
import time
from types import SimpleNamespace
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from pose_types import Localizer, Pose
from edm_profile import apply_edm_tracker_profile, load_edm_production_profile  # noqa: F401
from production_edm_tracker import EDMConfig, ProductionEDMTracker
from reloc_localizer_edm import Camera, EDMRelocMap  # noqa: F401  (re-exported for the worker)
from edm_matcher import EDM_H, EDM_W

# Legacy EDM fallback camera. Site profiles must pass their map-specific query camera;
# a bundle's 3D geometry and PnP intrinsics must never be mixed across sites.
CAM_720_EDM = (
    "PINHOLE", 1280, 720,
    [
        1000.8124857902368, 1007.5681084778566,
        670.8167724609375, 358.71917724609375,
    ],
)


def production_edm_config() -> EDMConfig:
    """tracker_defaults from the EDM package's config.json, pinned here like
    path_follow_flight.production_config() pins the XFeat sweep.

    local_topk=1 is the measured RTX 5060 speed/accuracy balance. LOW/WEAK
    raises local top-k to 3. BOOT/LOST stage MegaLoc from top-k 10 to 20.
    """
    return EDMConfig(
        global_retrieval_policy="boot_and_lost_once",
        local_topk=1, weak_local_topk=3, boot_global_topk=10,
        max_corr_total=900, track_min_inliers=50,
    )


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


@dataclass
class RuntimeState:
    """Field-for-field the XFeat tracker's RuntimeState: the worker seeds these directly
    (live_localizer_worker.seed_local_prior / apply_runtime_benchmark_mode)."""
    mode: str = "BOOT_INIT"
    last_pose: Pose | None = None
    prev_pose: Pose | None = None
    last_center: np.ndarray | None = None
    prev_center: np.ndarray | None = None
    last_yaw: float | None = None
    prev_yaw: float | None = None
    last_refs: list = field(default_factory=list)
    bad_count: int = 0
    fail_count: int = 0


@dataclass
class InertTemporalCache:
    """EDM is detector-free, so there are no descriptors to cache and no temporal anchor
    cache to reset. Present only so the worker's force-track bench, which does
    `type(tracker.temporal_cache)()`, keeps working."""

    def __len__(self) -> int:
        return 0

    def clear(self) -> None:
        pass


class EDMTrackerAdapter(Localizer):
    def __init__(self, reloc_map: EDMRelocMap, camera: Camera,
                 cfg: EDMConfig | None = None, megaloc=None, megaloc_factory=None,
                 matcher=None,
                 frame_source=lambda: None, map_frame=None,
                 reference_index=None,
                 motion_validator=None,
                 motion_validation_mode: str = "off"):
        # matcher: EDMMatcher (torch FP16) or EDMOnnxMatcher (ORT CUDA/TensorRT).
        # None -> ProductionEDMTracker builds the default torch EDMMatcher.
        self.trk = ProductionEDMTracker(
            reloc_map, camera,
            cfg=cfg or production_edm_config(),
            matcher=matcher,
            megaloc=megaloc,
            megaloc_factory=megaloc_factory,
            reference_index=reference_index,
            motion_validator=motion_validator,
            motion_validation_mode=motion_validation_mode,
        )
        self.map = reloc_map
        self.cfg = self.trk.cfg
        self.frame_source = frame_source
        self.map_frame = map_frame
        self.state = RuntimeState()
        self.temporal_cache = InertTemporalCache()
        self._last_info: dict = {}

    @property
    def last_info(self) -> dict:
        return self._last_info

    # ---------- model lifecycle ----------
    def ensure_edm(self) -> None:
        """EDM itself is built eagerly by EDMLocalizer.__init__; MegaLoc stays off GPU.
        Mirrors ensure_xfeat(): the local matcher without the retrieval model."""

    # the worker calls ensure_xfeat() only on the force-track bench, where MegaLoc must
    # stay off the GPU so its cost is not attributed to the matching path being measured.
    ensure_xfeat = ensure_edm

    def ensure_models(self) -> None:
        self.trk.loc.megaloc  # lazy property: loads MegaLoc now, not on frame 0

    def _clear_tracking_history(self) -> None:
        """Drop priors that must not survive LOST reacquisition."""
        s = self.state
        s.last_pose = s.prev_pose = None
        s.last_center = s.prev_center = None
        s.last_yaw = s.prev_yaw = None
        s.last_refs = []
        self.trk.st = type(self.trk.st)(
            accepted_step_norms=deque(
                maxlen=self.cfg.adaptive_jump_history_size
            ),
            observed_capture_dts=deque(
                maxlen=self.cfg.adaptive_jump_history_size
            ),
        )
        clear_visual = getattr(self.trk, "_clear_visual_motion_cache", None)
        if callable(clear_visual):
            clear_visual()
        controller = getattr(self.trk, "pose_guided", None)
        if controller is not None:
            controller.reset()


    # ---------- state mirror ----------
    # self.state is the worker-facing mirror; self.trk.st is EDM's own. Push before each
    # frame so anything the worker seeded takes effect, pull after so it can be read back.
    def _push_state(self) -> None:
        st = self.trk.st
        center = (None if self.state.last_center is None
                  else np.asarray(self.state.last_center, np.float32))
        # A centre EDM did not produce itself (a seeded prior, or a reset) carries no
        # usable velocity -- carrying the old one would predict from the wrong place.
        seeded = (st.center is None or center is None
                  or not np.allclose(st.center, center))
        st.state = self.state.mode
        st.center = center
        st.yaw = self.state.last_yaw
        st.last_refs = list(self.state.last_refs)
        st.misses = int(self.state.fail_count)
        if seeded:
            st.velocity = None
            st.last_capture_stamp = None

    def _pull_state(self) -> None:
        st, s = self.trk.st, self.state
        s.prev_pose, s.prev_center, s.prev_yaw = s.last_pose, s.last_center, s.last_yaw
        s.mode = st.state
        s.last_center = None if st.center is None else np.asarray(st.center, np.float32)
        s.last_yaw = st.yaw
        s.last_refs = list(st.last_refs)
        s.fail_count = s.bad_count = int(st.misses)

    def observe_fused_state(self, sample) -> None:
        observe = getattr(self.trk, "observe_fused_state", None)
        if callable(observe):
            observe(sample)

    def attach_pose_guided(self, controller) -> None:
        attach = getattr(self.trk, "attach_pose_guided", None)
        if callable(attach):
            attach(controller)

    # ---------- one frame ----------
    def localize_frame(self, frame: np.ndarray,
                       capture_stamp: float | None = None) -> Pose | None:
        # The worker hands us RGB; the EDM package is BGR-in throughout (its grayscale
        # conversion and its MegaLoc pre-step both assume it).
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._push_state()
        info = self.trk.localize(bgr, capture_stamp=capture_stamp)
        self._pull_state()

        pose = None
        if info.get("ok"):
            c = info["center"]
            pose_yaw = float(info["yaw"])
            R = np.asarray(info.get("R"), dtype=float)
            map_frame = getattr(self, "map_frame", None)
            if (map_frame is not None and R.shape == (3, 3)
                    and np.isfinite(R).all()):
                pose_yaw = float(map_frame.heading(R[2]))
            pose = Pose(x=float(c[0]), y=float(c[1]), z=float(c[2]),
                        yaw=pose_yaw,
                        stamp=time.monotonic() if capture_stamp is None
                        else float(capture_stamp))
            self.state.last_pose = pose

        camera_axes_world = None
        camera_forward_world = None
        R = np.asarray(info.get("R"), dtype=float)
        if R.shape == (3, 3) and np.isfinite(R).all():
            # cam_from_world is p_cam = R @ p_world + t. Each row of R is one
            # camera axis expressed in world coordinates: right, down, forward.
            camera_axes_world = R.astype(float).tolist()
            camera_forward_world = R[2].astype(float).tolist()

        self._last_info = {
            "mode": info["state_in"],
            "next_mode": info["state_out"],
            # The flight loop's WEAK gate (SFM_GATE_WEAK -> LoopHooks.pose_is_weak)
            # reads this key. On success the tracker always reports state_out="TRACK",
            # so the only record that a fix was won from a degraded state is state_in:
            # WEAK_TRACK accepts at weak_min_inliers, and LOST re-acquisition skips the
            # trajectory-jump gate. Without this key the gate silently never fires.
            "weak": info["state_in"] in ("WEAK_TRACK", "LOST"),
            "inliers": int(info.get("inliers", 0) or 0),
            "reproj_rms": info.get("reproj_rms"),
            "inlier_ratio": info.get("inlier_ratio"),
            "inlier_grid_cells": info.get("inlier_grid_cells"),
            "vpr_ms": info.get("vpr_ms"),
            "feature_ms": None,      # detector-free: no separate feature stage
            "match_ms": info.get("match_ms"),
            # Stage breakdown of the worker's core_wall. vpr/match/pnp alone
            # left ~27% of it unattributed; these name the rest.
            "total_ms": info.get("total_ms"),
            "stage_gray_ms": info.get("stage_gray_ms"),
            "stage_bridge_ms": info.get("stage_bridge_ms"),
            "stage_query_ms": info.get("stage_query_ms"),
            "stage_select_ms": info.get("stage_select_ms"),
            "pnp_ms": info.get("pnp_ms"),
            "pnp_candidates": info.get("pnp_candidates"),
            "pnp_skipped": info.get("pnp_skipped"),
            "pnp_workers": info.get("pnp_workers"),
            "host_feature_cache": info.get("host_feature_cache"),
            "n_corr": info.get("n_corr"),
            "refs": info.get("refs"),
            "reference_count": len(info.get("refs") or []),
            "requested_reference_count": info.get("requested_reference_count"),
            "staged_early_stop": info.get("staged_early_stop"),
            "candidate_mode": info.get("candidate_mode"),
            "global_retrieval_calls": info.get("global_retrieval_calls"),
            "lost_search_stage": info.get("lost_search_stage"),
            "lost_search_radius_factor": info.get("lost_search_radius_factor"),
            "rejected": info.get("rejected"),
            "limited_jump": info.get("limited_jump"),
            "relative_motion_check": info.get("relative_motion_check"),
            "limited_jump_confirmed": info.get("limited_jump_confirmed", False),
            "camera_axes_world": camera_axes_world,
            "camera_forward_world": camera_forward_world,
            "pose_status": info.get(
                "pose_status",
                "VISUALLY_CONFIRMED" if info.get("ok") else "NONE",
            ),
            "prediction_valid": info.get("prediction_valid", False),
            "prediction_mode": info.get("prediction_mode"),
            "predicted_center": info.get("predicted_center"),
            "predicted_yaw": info.get("predicted_yaw"),
            # ESEKF diagnostics. The filter is fed by observe_fused_state and
            # gates itself on prediction_allowed(), so without these the live
            # session cannot say whether the IMU path ever armed -- which is the
            # whole question docs/esekf_live_eval_runbook.md asks.
            "esekf_pos_trace": info.get("esekf_pos_trace"),
            "esekf_d2": info.get("esekf_d2"),
            "esekf_update_accepted": info.get("esekf_update_accepted"),
            "esekf_update_exceptions": info.get("esekf_update_exceptions"),
            "prediction_source": info.get("prediction_source"),
        }
        return pose

    def get_pose(self) -> Pose | None:
        sample = self.frame_source()
        if sample is None:
            return None
        if isinstance(sample, tuple):
            if len(sample) != 2:
                self._last_info = {
                    "mode": self.state.mode,
                    "next_mode": self.state.mode,
                    "error": "invalid_frame_sample",
                    "inliers": 0,
                }
                return None
            frame, capture_stamp = sample
        else:
            frame, capture_stamp = sample, time.monotonic()
        return self.localize_frame(frame, capture_stamp=capture_stamp)


class AsyncEDMTrackerAdapter(Localizer):
    """Async fast/slow wrapper (code default; SFM_EDM_ASYNC_TRACKER=0 reverts).

    Fast thread (CPU KLT+PnP, 3-5ms) publishes every frame; a dedicated
    stateful ProductionEDMTracker runs slow keyframes on a background thread
    and feeds the anchor mailbox. Accepted anchors are VISUALLY_CONFIRMED
    only (quality.py semantics mirrored in SlowPath); fast frames are
    KLT_BRIDGED and never become anchors. The sync EDMTrackerAdapter path
    is untouched.

    When the fast path has nothing to publish (BOOT before the first anchor,
    or a total track loss) the keyframe runs inline on the calling thread, so
    a caller that consumes frames faster than the slow path can answer still
    gets synchronous-tracker behaviour instead of a stream of NO_ANCHOR
    (`SFM_EDM_ASYNC_INLINE_SYNC=0` disables that floor).
    """

    def __init__(self, reloc_map: EDMRelocMap, camera: Camera,
                 cfg: EDMConfig | None = None, megaloc=None, megaloc_factory=None,
                 matcher=None,
                 frame_source=lambda: None, map_frame=None,
                 reference_index=None,
                 motion_validator=None,
                 motion_validation_mode: str = "off"):
        from async_localizer import AsyncLocalizer

        self.trk = ProductionEDMTracker(
            reloc_map, camera,
            cfg=cfg or production_edm_config(),
            matcher=matcher,
            megaloc=megaloc,
            megaloc_factory=megaloc_factory,
            reference_index=reference_index,
            motion_validator=motion_validator,
            motion_validation_mode=motion_validation_mode,
        )
        params = list(camera.params[:4])
        sx = float(EDM_W) / float(camera.width)
        sy = float(EDM_H) / float(camera.height)
        fast_camera = SimpleNamespace(
            width=int(EDM_W), height=int(EDM_H),
            fx=float(params[0]) * sx, fy=float(params[1]) * sy,
            cx=float(params[2]) * sx, cy=float(params[3]) * sy,
        )
        # The slow tracker is the authority on what a fix is: every anchor it
        # publishes already passed this site's inlier / reproj / ratio / spread
        # and trajectory gates. Feed those same numbers to the async gates so
        # the anchor path cannot re-reject a site-approved fix with unrelated
        # module defaults (that dropped every recovery anchor -- inlier_ratio
        # on a 5-ref acquire frame is ~0.23, the module default asked 0.66).
        cfg_gates = self.trk.cfg
        self.async_loc = AsyncLocalizer(
            camera=fast_camera, edm_matcher=self._slow_keyframe,
            max_jump=float(cfg_gates.max_jump),
            max_yaw_jump_deg=float(cfg_gates.acquire_max_yaw_diff_deg),
            weak_min_inliers=int(cfg_gates.weak_min_inliers),
            anchor_min_inliers=int(cfg_gates.track_min_inliers),
            anchor_min_ratio=float(cfg_gates.min_inlier_ratio),
            max_reproj_error=float(cfg_gates.max_reproj_error_track),
            # Hard bound on consecutive frames served without adopting a fresh
            # visual anchor. See SyncCarry: the per-frame drift budget is
            # credited back by every gated fast fix, and a fast fix re-fits the
            # same KLT-carried 3D, so this is the only thing that stops the
            # carry from dead-reckoning indefinitely.
            max_anchorless_frames=(
                _env_int("SFM_EDM_ASYNC_MAX_ANCHORLESS")
                if _env_int("SFM_EDM_ASYNC_MAX_ANCHORLESS") is not None
                else 30
            ),
            pnp_ransac_seed=self.trk._pnp_random_seed(),
            fast_min_inliers=(
                _env_int("SFM_EDM_ASYNC_FAST_MIN_INLIERS")
                if _env_int("SFM_EDM_ASYNC_FAST_MIN_INLIERS") is not None
                else int(getattr(cfg_gates, "track_min_inliers", 50))
            ),
            fast_max_reproj_error=(
                _env_float("SFM_EDM_ASYNC_FAST_REPROJ")
                if _env_float("SFM_EDM_ASYNC_FAST_REPROJ") is not None
                else 2.5
            ),
            inline_sync_fallback=_env_bool("SFM_EDM_ASYNC_INLINE_SYNC", True),
        )
        self.async_loc.start()
        self.map = reloc_map
        self.cfg = self.trk.cfg
        self.frame_source = frame_source
        self.map_frame = map_frame
        self.state = RuntimeState()
        self.temporal_cache = InertTemporalCache()
        self._last_info: dict = {}
        # Last accepted slow-keyframe telemetry, surfaced in _last_info so the
        # production-path gate (source-age / p50) can read async runs. Updated
        # only on VISUALLY_CONFIRMED slow accepts; fast frames never clear it.
        self._last_slow_info: dict = {}
        # Worker main thread seeds self.state directly (mode/fail_count/refs);
        # slow keyframes run on a background thread. This lock coordinates the
        # adapter's own mirror accesses; the slow tracker is source of truth
        # for recovery state and the worker re-pins bench modes every frame.
        self._state_lock = threading.Lock()
        # Fast-prior feedback channel (see _apply_fast_prior_feedback): the
        # latest published fast pose, as (center float32(3,), yaw|None, stamp).
        # Plain-attribute replace is atomic under the GIL; a reader racing a
        # writer only ever sees an older-or-newer complete tuple, both valid
        # priors, so no lock (and no new deadlock surface) is required.
        self._fast_prior: tuple | None = None
        self._last_push_snapshot = None
        self._fast_prior_feedbacks: int = 0
    def _push_state_to_slow(self) -> None:
        """Propagate worker-seeded mirror state into the slow tracker.

        Same semantics as EDMTrackerAdapter._push_state, but applied at slow
        keyframe cadence (the only place the slow state machine can take it).
        """
        st = getattr(self.trk, "st", None)
        if st is None:
            return
        center = (None if self.state.last_center is None
                  else np.asarray(self.state.last_center, np.float32))
        seeded = (st.center is None or center is None
                  or not np.allclose(st.center, center))
        st.state = self.state.mode
        st.center = center
        st.yaw = self.state.last_yaw
        st.last_refs = list(self.state.last_refs)
        st.misses = int(self.state.fail_count)
        if seeded:
            st.velocity = None
            st.last_capture_stamp = None
        self._apply_fast_prior_feedback(st)

    def _mirror_snapshot(self):
        """Hashable snapshot of the worker-visible mirror (provenance check)."""
        s = self.state
        try:
            c = None if s.last_center is None else np.asarray(
                s.last_center, np.float32).tobytes()
            return (s.mode, c, s.last_yaw, tuple(s.last_refs),
                    int(s.fail_count))
        except Exception:
            return None

    def _apply_fast_prior_feedback(self, st) -> None:
        """Refresh slow PRIOR fields from fast-path progress (prior only).

        The slow tracker is driven sparsely; across a KLT-bridged stretch its
        center/velocity prior is N frames stale while the fast path kept
        tracking, so the next keyframe opens its local pool around a stale
        pose. Feed the latest published fast pose back as PRIOR
        (center/yaw/velocity/stamp) without touching recovery state
        (mode/misses/refs) or any gate.

        Arbitration: the mirror is augmented only when the worker did not seed
        it since the last pull (mirror identical to snapshot) -- bench modes,
        force-track priors and LOST injection always win verbatim. All reads
        are getattr-defensive: unit stubs may lack slow fields entirely.
        """
        try:
            if getattr(self, "_last_push_snapshot", None) is None:
                return
            if self._mirror_snapshot() != self._last_push_snapshot:
                return  # worker seeded since last pull; status quo ante
            fp = getattr(self, "_fast_prior", None)
            if fp is None:
                return
            fcenter, fyaw, fstamp = fp
            fstamp = float(fstamp)
            if not math.isfinite(fstamp):
                return
            old_stamp = getattr(st, "last_capture_stamp", None)
            if old_stamp is not None:
                old_stamp = float(old_stamp)
                if not math.isfinite(old_stamp) or fstamp <= old_stamp:
                    return  # slow already beyond this prior
            fcenter = np.asarray(fcenter, dtype=np.float32)
            if fcenter.shape != (3,) or not np.isfinite(fcenter).all():
                return
            old_center = getattr(st, "center", None)
            old_center = (None if old_center is None
                          else np.asarray(old_center, dtype=np.float32))
            if (old_center is not None and old_stamp is not None
                    and old_center.shape == (3,)
                    and np.isfinite(old_center).all()):
                dt = fstamp - old_stamp
                st.velocity = ((fcenter - old_center) / dt
                               if dt > 0.0 and math.isfinite(dt) else None)
            else:
                st.velocity = None
            st.center = fcenter
            st.yaw = None if fyaw is None else float(fyaw)
            st.last_capture_stamp = fstamp
            self._fast_prior_feedbacks = int(
                getattr(self, "_fast_prior_feedbacks", 0) or 0) + 1
        except Exception:
            return

    def _pull_state_from_slow(self) -> None:
        """Mirror slow tracker recovery state back for the worker to read."""
        st = getattr(self.trk, "st", None)
        if st is None:
            return
        s = self.state
        s.mode = st.state
        s.last_center = None if st.center is None else np.asarray(st.center, np.float32)
        s.last_yaw = st.yaw
        s.last_refs = list(st.last_refs)
        s.fail_count = s.bad_count = int(st.misses)
        # Provenance anchor for _apply_fast_prior_feedback: between keyframes
        # only worker seeds may change the mirror, so any deviation from this
        # snapshot at the next push means "worker seeded, hands off".
        self._last_push_snapshot = self._mirror_snapshot()

    @property
    def last_info(self) -> dict:
        return self._last_info

    def _slow_keyframe(self, gray: np.ndarray, stamp: float) -> dict | None:
        """Run one stateful EDM step on a keyframe; return anchor dict or None."""
        bgr = cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2BGR)
        with self._state_lock:
            self._push_state_to_slow()
            try:
                info = self.trk.localize(bgr, capture_stamp=float(stamp))
            except Exception:
                if os.environ.get("SFM_EDM_ASYNC_DEBUG", "0").strip().lower() in (
                    "1", "true", "yes", "on",
                ):
                    import traceback as _tb
                    print("[async-slow] localize EXC", flush=True)
                    _tb.print_exc()
                return None
            finally:
                self._pull_state_from_slow()
        if os.environ.get("SFM_EDM_ASYNC_DEBUG", "0").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            try:
                print(
                    "[async-slow] keyframe ok=%s status=%s mode=%s inl=%s ncorr=%s "
                    "refs=%s vpr=%s" % (
                        info.get("ok"), info.get("pose_status"),
                        info.get("state_in"),
                        info.get("inliers"), info.get("n_corr"),
                        info.get("requested_reference_count"),
                        info.get("vpr_ms"),
                    ),
                    flush=True,
                )
                self._slow_keyframes = int(getattr(self, "_slow_keyframes", 0)) + 1
                if self._slow_keyframes % 100 == 0:
                    # close() is not guaranteed to run under the worker's
                    # shutdown path, so checkpoint the accounting periodically.
                    self._debug_anchor_summary()
            except Exception:
                pass
        if not info.get("ok") or info.get("pose_status") != "VISUALLY_CONFIRMED":
            return None
        # NOTE: _last_accepted_cam_from_world is only written when the motion
        # cache is active; _last_accepted_rigid3d is written on every accept.
        mat = None
        rigid = getattr(self.trk, "_last_accepted_rigid3d", None)
        if rigid is not None:
            try:
                raw = np.asarray(rigid.matrix(), dtype=float)
                mat = np.eye(4, dtype=float)
                mat[:3, :] = raw.reshape(3, 4)
            except Exception:
                mat = None
        if mat is None:
            mat = getattr(self.trk, "_last_accepted_cam_from_world", None)
        pts2d = getattr(self.trk, "_klt_2d", None)
        pts3d = getattr(self.trk, "_klt_3d", None)
        if mat is None or pts2d is None or pts3d is None or len(pts2d) == 0:
            return None
        # Slow-tracker KLT seeds are camera-res (1280x720); the fast path tracks
        # EDM-res gray (1024x576). Rescale x/y so LK starts on the right pixels.
        # _klt_3d is resolution-independent (map-frame 3D) and passes through.
        try:
            sx = float(EDM_W) / float(self.trk.cam.width)
            sy = float(EDM_H) / float(self.trk.cam.height)
        except Exception:
            sx = sy = 1.0
        inlier_2d = np.asarray(pts2d, dtype=np.float32).copy()
        inlier_2d[:, 0] *= sx
        inlier_2d[:, 1] *= sy
        out = {
            "ok": True,
            "cam_from_world": np.asarray(mat, dtype=float),
            "inlier_2d": inlier_2d,
            "inlier_3d": np.asarray(pts3d, dtype=float),
            "inliers": int(info.get("inliers", 0) or 0),
            "inlier_ratio": float(info.get("inlier_ratio", 1.0) or 0.0),
            "reproj_rms": info.get("reproj_rms"),
            "inlier_grid_cells": int(info.get("inlier_grid_cells", 0) or 0),
            "pose_status": "VISUALLY_CONFIRMED",
            "vpr_ms": info.get("vpr_ms"),
            "match_ms": info.get("match_ms"),
            "pnp_ms": info.get("pnp_ms"),
            "refs": info.get("refs"),
            "reference_count": info.get("reference_count"),
            "requested_reference_count": info.get("requested_reference_count"),
            "candidate_mode": info.get("candidate_mode"),
            "global_retrieval_calls": info.get("global_retrieval_calls"),
        }
        try:
            self._last_slow_info = {
                "vpr_ms": out.get("vpr_ms"),
                "match_ms": out.get("match_ms"),
                "pnp_ms": out.get("pnp_ms"),
                "inliers": out.get("inliers"),
                "inlier_ratio": out.get("inlier_ratio"),
                "reproj_rms": out.get("reproj_rms"),
                "inlier_grid_cells": out.get("inlier_grid_cells"),
                "refs": out.get("refs"),
                "reference_count": out.get("reference_count"),
                "requested_reference_count": out.get("requested_reference_count"),
                "candidate_mode": out.get("candidate_mode"),
                "global_retrieval_calls": out.get("global_retrieval_calls"),
                "slow_stamp": float(stamp),
            }
        except Exception:
            pass
        return out

    def localize_frame(self, frame: np.ndarray,
                       capture_stamp: float | None = None) -> Pose | None:
        from edm_matcher import EDMMatcher

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = EDMMatcher.load_gray(bgr)
        stamp = time.monotonic() if capture_stamp is None else float(capture_stamp)
        result = self.async_loc.feed_frame(gray, stamp)
        info = result.as_dict()
        # Mirror EDMTrackerAdapter._pull_state: prev_* always trails last_*.
        self.state.prev_pose = self.state.last_pose
        pose = None
        camera_axes_world = None
        camera_forward_world = None
        if info.get("pose_status") in ("VISUALLY_CONFIRMED", "KLT_BRIDGED"):
            position = info.get("pose") or {}
            try:
                pose_yaw = float(position["yaw"])
                # Parity with EDMTrackerAdapter: map_frame.heading(R[2]) wins when
                # the fast pose carries a valid rotation matrix.
                try:
                    rot = position.get("rotation")
                    map_frame = getattr(self, "map_frame", None)
                    if rot is not None:
                        rarr = np.asarray(rot, dtype=float)
                        if rarr.shape == (3, 3) and np.isfinite(rarr).all():
                            # Same contract as EDMTrackerAdapter: rows of
                            # cam_from_world are the camera right/down/forward
                            # axes in world coordinates. The operator map
                            # overlay and _autonomy_pose both read these; an
                            # async adapter that omitted them left the UI
                            # drawing the travel direction instead of where the
                            # camera looks, and starved AUTO of a heading.
                            camera_axes_world = rarr.tolist()
                            camera_forward_world = rarr[2].tolist()
                            if map_frame is not None:
                                pose_yaw = float(map_frame.heading(rarr[2]))
                except Exception:
                    pass
                pose = Pose(x=float(position["x"]), y=float(position["y"]),
                            z=float(position["z"]), yaw=pose_yaw,
                            stamp=stamp)
                self.state.last_pose = pose
                # Feed the slow prior (see _apply_fast_prior_feedback). Only
                # published visual poses qualify; stale NEED_REANCHOR/LOST
                # payloads never become priors.
                try:
                    self._fast_prior = (
                        np.array([pose.x, pose.y, pose.z], dtype=np.float32),
                        float(pose_yaw), float(stamp))
                except Exception:
                    pass
            except (KeyError, TypeError, ValueError):
                pose = None
        slow = getattr(self, "_last_slow_info", None) or {}
        slow_path = getattr(self.async_loc, "slow_path", None)
        slow_thread = getattr(slow_path, "_worker_thread", None)
        self._last_info = {
            "mode": info.get("pose_status"),
            "next_mode": info.get("pose_status"),
            "weak": info.get("pose_status") not in ("VISUALLY_CONFIRMED",),
            "inliers": int(info.get("inliers", 0) or 0),
            "tracked": int(info.get("tracked", 0) or 0),
            "stale": bool(info.get("stale", False)),
            "reproj_rms": info.get("reproj_rms"),
            "inlier_ratio": info.get("inlier_ratio", slow.get("inlier_ratio")),
            "inlier_grid_cells": info.get("inlier_grid_cells", slow.get("inlier_grid_cells")),
            "vpr_ms": slow.get("vpr_ms"),
            "feature_ms": None,
            "match_ms": slow.get("match_ms"),
            "pnp_ms": slow.get("pnp_ms"),
            "candidate_mode": info.get("candidate_mode"),
            "camera_axes_world": camera_axes_world,
            "camera_forward_world": camera_forward_world,
            "pose_status": info.get("pose_status"),
            "sync_status": info.get("sync_status"),
            "drift_count": int(info.get("drift_count", 0) or 0),
            "refs": slow.get("refs"),
            "reference_count": slow.get("reference_count"),
            "requested_reference_count": slow.get("requested_reference_count"),
            "global_retrieval_calls": slow.get("global_retrieval_calls"),
            "slow_stamp": slow.get("slow_stamp"),
            # A dead slow thread and a slow path that never matched both look
            # like permanent NO_ANCHOR from outside; make the difference legible.
            "inline_syncs": int(getattr(self.async_loc, "inline_syncs", 0)),
            "slow_errors": int(getattr(slow_path, "errors", 0) or 0),
            "slow_last_error": getattr(slow_path, "last_error", None),
            "slow_alive": bool(slow_thread is not None and slow_thread.is_alive()),
            # Anchor supply. A run that is all NEED_REANCHOR/LOST is either a
            # slow path that never ran or one whose every result was refused;
            # only the accept/reject split says which, and which gate refused.
            "slow_anchor_accepts": int(getattr(slow_path, "anchor_accepts", 0) or 0),
            "slow_anchor_rejects": dict(getattr(slow_path, "anchor_rejects", {}) or {}),
            "slow_inline_runs": int(getattr(slow_path, "inline_runs", 0) or 0),
            "fast_prior_feedbacks": int(
                getattr(self, "_fast_prior_feedbacks", 0) or 0),
        }
        return pose

    def get_pose(self) -> Pose | None:
        sample = self.frame_source()
        if sample is None:
            return None
        if isinstance(sample, tuple):
            if len(sample) != 2:
                return None
            frame, capture_stamp = sample
        else:
            frame, capture_stamp = sample, time.monotonic()
        return self.localize_frame(frame, capture_stamp=capture_stamp)

    def close(self) -> None:
        try:
            self.async_loc.stop()
        except Exception:
            pass
        # After the slow thread joins, so the counters cannot move mid-read.
        self._debug_anchor_summary()

    def _debug_anchor_summary(self) -> None:
        """One line of anchor accounting (SFM_EDM_ASYNC_DEBUG=1).

        A starved fast path and a slow path that never matched look identical in
        the mode histogram; the accept/reject split and the measured anchor
        cadence tell them apart.
        """
        if os.environ.get("SFM_EDM_ASYNC_DEBUG", "0").strip().lower() not in (
            "1", "true", "yes", "on",
        ):
            return
        try:
            slow = self.async_loc.slow_path
            carry = self.async_loc.sync_carry
            print(
                "[async-slow] anchors accepted=%d rejects=%s mailbox=%s carry=%s "
                "frame_gap=%s stamp_gap=%s drift_budget=%d anchor_max_age=%.3f" % (
                    slow.anchor_accepts,
                    dict(slow.anchor_rejects),
                    dict(self.async_loc.mailbox.stats),
                    dict(carry.stats),
                    carry.supply.frame_gap(),
                    carry.supply.stamp_gap(),
                    carry.effective_drift_budget(),
                    carry.effective_anchor_max_age_s(),
                ),
                flush=True,
            )
        except Exception:
            pass
    def ensure_models(self) -> None:
        """Warm slow-tracker models at worker startup, not on first keyframe.

        Sync path warms MegaLoc lazily via ensure_models; without this the first
        slow keyframe pays full model-load latency while fast path reports
        NO_ANCHOR. Steady-state behavior unchanged.
        """
        try:
            self.trk.loc.megaloc  # lazy property: loads MegaLoc now
        except Exception:
            pass
        try:
            warmup = getattr(getattr(self.trk, "loc", None), "matcher", None)
            if warmup is not None and hasattr(warmup, "warmup_fused_coarse"):
                warmup.warmup_fused_coarse()
        except Exception:
            pass
    def observe_fused_state(self, sample) -> None:
        observe = getattr(self.trk, "observe_fused_state", None)
        if callable(observe):
            observe(sample)
    def attach_pose_guided(self, controller) -> None:
        attach = getattr(self.trk, "attach_pose_guided", None)
        if callable(attach):
            attach(controller)
    def _clear_tracking_history(self) -> None:
        """Drop priors that must not survive LOST reacquisition or bench switch.

        Mirrors EDMTrackerAdapter._clear_tracking_history for the slow tracker,
        plus the fast path / chain / mailbox / scheduler which have no sync
        equivalent. Slow tracker is source of truth; the worker rebinds
        tracker.state itself (reset_to_boot) right after calling this.
        """
        with self._state_lock:
            s = self.state
            s.last_pose = s.prev_pose = None
            s.last_center = s.prev_center = None
            s.last_yaw = s.prev_yaw = None
            s.last_refs = []
            try:
                self.trk.st = type(self.trk.st)(
                    accepted_step_norms=deque(
                        maxlen=self.cfg.adaptive_jump_history_size
                    ),
                    observed_capture_dts=deque(
                        maxlen=self.cfg.adaptive_jump_history_size
                    ),
                )
            except Exception:
                pass
            clear_visual = getattr(self.trk, "_clear_visual_motion_cache", None)
            if callable(clear_visual):
                try:
                    clear_visual()
                except Exception:
                    pass
            clear_klt = getattr(self.trk, "_clear_klt_cache", None)
            if callable(clear_klt):
                try:
                    clear_klt()
                except Exception:
                    pass
            controller = getattr(self.trk, "pose_guided", None)
            if controller is not None:
                try:
                    controller.reset()
                except Exception:
                    pass
            try:
                self.async_loc.reset()
            except Exception:
                pass
            self._last_slow_info = {}
            # A prior from the previous map/run must never leak into the next.
            self._fast_prior = None
            self._last_push_snapshot = None
            self._fast_prior_feedbacks = 0
