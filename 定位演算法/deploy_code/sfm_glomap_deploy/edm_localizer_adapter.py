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

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from pose_types import Localizer, Pose
from edm_profile import apply_edm_tracker_profile, load_edm_production_profile  # noqa: F401
from production_edm_tracker import EDMConfig, ProductionEDMTracker
from reloc_localizer_edm import Camera, EDMRelocMap  # noqa: F401  (re-exported for the worker)

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
