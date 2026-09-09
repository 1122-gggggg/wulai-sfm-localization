#!/usr/bin/env python3
"""DirectTrackerAdapter: the two-rate direct localizer behind the worker API.

`live_localizer_worker` drives every backend through the same duck-typed
surface: `state`, `temporal_cache`, `map_frame`, `ensure_models()`,
`localize_frame(rgb, capture_stamp)`, `last_info`, `observe_fused_state()`,
`attach_pose_guided()`. This adapter maps `TwoRateTracker` onto it without
changing anything the EDM/XFeat backends already promise.

Status mapping (safety contract, never widened):

    FAST_TRACK   -> TRACK       weak=False   PnP inliers include >=12 map points
    RELOC_SEED   -> TRACK       weak=False   handover landed on this frame
    VO_ONLY      -> WEAK_TRACK  weak=True    inliers are self-built VO points
    DEAD_RECKON  -> WEAK_TRACK  weak=True    essential-matrix dead reckoning
    NO_POSE      -> LOST        weak=True    localize_frame returns None
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from pose_types import Localizer, Pose

from two_rate_tracker import TwoRateTracker

#: status -> (worker mode, weak flag). See the module docstring.
_STATUS_MODE: dict[str, tuple[str, bool]] = {
    "FAST_TRACK": ("TRACK", False),
    "RELOC_SEED": ("TRACK", False),
    "VO_ONLY": ("WEAK_TRACK", True),
    "DEAD_RECKON": ("WEAK_TRACK", True),
    "NO_POSE": ("LOST", True),
}


@dataclass
class RuntimeState:
    """Field-for-field the XFeat/EDM tracker's RuntimeState: the worker seeds
    these directly (live_localizer_worker.seed_local_prior /
    apply_runtime_benchmark_mode), and rebuilds it with `type(state)()`."""

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
    """The direct backend keeps its temporal anchor as tracked 2D points inside
    TwoRateTracker, not as a descriptor cache. Present only so the worker's
    force-track bench, which does `type(tracker.temporal_cache)()`, keeps
    working."""

    def __len__(self) -> int:
        return 0

    def clear(self) -> None:
        pass


class DirectTrackerAdapter(Localizer):
    def __init__(
        self,
        assets,
        profile,
        provider,
        *,
        map_frame=None,
        frame_source=lambda: None,
    ) -> None:
        self.assets = assets
        self.profile = profile
        self.provider = provider
        self.trk = TwoRateTracker(assets, profile, provider)
        self.map_frame = map_frame
        self.frame_source = frame_source
        self.state = RuntimeState()
        self.temporal_cache = InertTemporalCache()
        self._last_info: dict = {}
        self._fused_sample = None

    @property
    def map_frame(self):
        return self.trk.map_frame

    @map_frame.setter
    def map_frame(self, value) -> None:
        """One copy, shared with the tracker.

        `live_localizer_worker` reassigns this after the localizer is built, and
        the tracker's dead-reckon yaw guard needs the same basis this adapter
        publishes headings in, so the attribute lives on the tracker and this is
        the view onto it.
        """
        self.trk.map_frame = value

    @property
    def last_info(self) -> dict:
        return self._last_info

    # ---------- model lifecycle ----------
    def ensure_models(self) -> None:
        """Load MegaLoc + EDM and start the relocalizer thread. The worker calls
        this once, before the first frame, so no stream frame pays the load."""
        self.trk.ensure_models()

    def ensure_edm(self) -> None:
        """No-op: the fast loop is pure CPU and owns no local matcher. The
        force-track bench calls this to load the matcher without the retrieval
        model; here there is nothing to split."""

    ensure_xfeat = ensure_edm

    def close(self) -> None:
        self.trk.close()

    def _clear_tracking_history(self) -> None:
        """Drop priors that must not survive LOST reacquisition."""
        s = self.state
        s.last_pose = s.prev_pose = None
        s.last_center = s.prev_center = None
        s.last_yaw = s.prev_yaw = None
        s.last_refs = []
        s.fail_count = s.bad_count = 0
        self.trk.reset()

    # ---------- fusion / retrieval hooks ----------
    def observe_fused_state(self, sample) -> None:
        """Record the fused navigation sample for diagnostics only.

        The frozen P174 configuration has no ESEKF and no IMU/NED coupling: the
        fast loop's prior is the tracked 2D point set, not a filtered state.
        Storing the sample without consuming it is deliberate -- pretending to
        fuse would make `last_info` claim an inertial aid that does not exist.
        """
        self._fused_sample = sample

    def attach_pose_guided(self, controller) -> None:
        """No-op. Pose-guided retrieval reorders the EDM tracker's *reference*
        candidate list by predicted pose. This backend retrieves with MegaLoc
        inside the background worker on a whole-image descriptor, and its fast
        loop does no retrieval at all, so there is no candidate ordering for a
        controller to steer."""

    # ---------- one frame ----------
    def localize_frame(
        self, frame: np.ndarray, capture_stamp: float | None = None
    ) -> Pose | None:
        # The worker hands us RGB; OpenCV and the whole direct stack are BGR-in.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        stamp = time.monotonic() if capture_stamp is None else float(capture_stamp)
        info = self.trk.step(bgr, stamp)

        status = str(info["status"])
        mode, weak = _STATUS_MODE[status]

        pose = None
        pose_yaw = info.get("yaw")
        rotation = info.get("R")
        rotation = None if rotation is None else np.asarray(rotation, dtype=float)
        if (
            self.map_frame is not None
            and rotation is not None
            and rotation.shape == (3, 3)
            and np.isfinite(rotation).all()
        ):
            pose_yaw = float(self.map_frame.heading(rotation[2]))
        if info["ok"] and info["center"] is not None and pose_yaw is not None:
            center = np.asarray(info["center"], dtype=float)
            pose = Pose(
                x=float(center[0]), y=float(center[1]), z=float(center[2]),
                yaw=float(pose_yaw), stamp=stamp,
            )

        self._advance_state(pose, info, mode, status)

        camera_axes_world = None
        camera_forward_world = None
        if rotation is not None and rotation.shape == (3, 3) and np.isfinite(rotation).all():
            # cam_from_world is p_cam = R @ p_world + t. Each row of R is one
            # camera axis expressed in world coordinates: right, down, forward.
            camera_axes_world = rotation.tolist()
            camera_forward_world = rotation[2].tolist()

        references = list(info.get("reference_names") or ())
        self._last_info = {
            "mode": mode,
            "next_mode": mode,
            # The flight loop's WEAK gate (SFM_GATE_WEAK -> LoopHooks.pose_is_weak)
            # reads this key: a VO-only or dead-reckoned pose has no map
            # constraint and must never be treated as a confirmed fix.
            "weak": weak,
            "inliers": int(info["inliers"]),
            "reproj_rms": None,          # fast-loop PnP reports no residual RMS
            "inlier_ratio": (
                None if not info["n_corr"] else float(info["inliers"]) / float(info["n_corr"])
            ),
            "inlier_grid_cells": None,
            "vpr_ms": None,              # retrieval runs off the fast loop
            "feature_ms": None,          # detector-free fast loop
            "match_ms": None,
            "total_ms": float(info["loop_ms"]),
            "stage_gray_ms": None,
            "stage_bridge_ms": None,
            "stage_query_ms": None,
            "stage_select_ms": None,
            "track_ms": float(info["track_ms"]),
            "pnp_ms": float(info["pnp_ms"]),
            "vo_ms": float(info["vo_ms"]),
            "pnp_candidates": int(info["n_corr"]),
            "pnp_skipped": None,
            "pnp_workers": None,
            "host_feature_cache": None,
            "n_corr": int(info["n_corr"]),
            "refs": references,
            "reference_count": len(references),
            "requested_reference_count": int(self.profile.reloc.top_k),
            "camera_axes_world": camera_axes_world,
            "camera_forward_world": camera_forward_world,
            "pose_status": "VISUALLY_CONFIRMED" if (info["ok"] and not weak) else "NONE",
            "prediction_valid": False,
            "prediction_mode": None,
            "predicted_center": None,
            "predicted_yaw": None,
            # --- direct-backend specific (Contract C4) ---
            "direct_status": status,
            "map_inliers": int(info["map_inliers"]),
            "vo_inliers": int(info["vo_inliers"]),
            "live_points": int(info["live_points"]),
            "dead_reckon_age": int(info["dead_reckon_age"]),
            "reloc_status": info["reloc_status"],
            "handover_points": int(info["handover_points"]),
            "handover_dropped": int(info["handover_dropped"]),
            "reloc_ms": info["reloc_ms"],
            "reloc_delivered": bool(info["reloc_delivered"]),
            "reloc_submitted": bool(info["reloc_submitted"]),
            "reloc_busy": bool(info["reloc_busy"]),
            "vo_candidates": int(info["vo_candidates"]),
            "step_norm": info["step"],
            "fused_state_seen": self._fused_sample is not None,
        }
        return pose

    def _advance_state(self, pose: Pose | None, info: dict, mode: str, status: str) -> None:
        s = self.state
        s.prev_pose, s.prev_center, s.prev_yaw = s.last_pose, s.last_center, s.last_yaw
        s.mode = mode
        if pose is not None:
            s.last_pose = pose
            s.last_center = np.asarray(info["center"], dtype=np.float32)
            s.last_yaw = float(pose.yaw)
        references = info.get("reference_names") or ()
        if references:
            s.last_refs = [str(name) for name in references]
        if mode == "TRACK":
            s.fail_count = 0
            s.bad_count = 0
        elif mode == "WEAK_TRACK":
            s.fail_count = 0
            s.bad_count += 1
        else:
            s.fail_count += 1
            s.bad_count += 1

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
