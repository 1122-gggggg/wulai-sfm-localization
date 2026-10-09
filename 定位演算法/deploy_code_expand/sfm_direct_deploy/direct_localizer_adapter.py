#!/usr/bin/env python3
"""Direct tracker selection behind the worker API.

`live_localizer_worker` drives every backend through the same duck-typed
surface: `state`, `temporal_cache`, `map_frame`, `ensure_models()`,
`localize_frame(rgb, capture_stamp)`, `last_info`, `observe_fused_state()`,
`attach_pose_guided()`. Legacy profiles and the evaluation baseline use
`TwoRateTracker`; enabled online-workmap profiles select the formal
`OnlineWorkMapTracker` lazily and fail closed when it is unavailable.

Status mapping (safety contract, never widened):

    FAST_TRACK   -> TRACK       weak=False   requires the configured strong map-inlier floor; otherwise WEAK_TRACK
    RELOC_SEED   -> TRACK       weak=False   handover landed on this frame
    VO_ONLY      -> WEAK_TRACK  weak=True    inliers are self-built VO points
    DEAD_RECKON  -> WEAK_TRACK  weak=True    essential-matrix dead reckoning
    IMU_BRIDGE   -> WEAK_TRACK  weak=True    fused-yaw rotation of last pose
    NO_POSE      -> LOST        weak=True    localize_frame returns None
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from localization_continuity import PoseSourceConfirmation
from pose_types import Localizer, Pose

from two_rate_tracker import TwoRateTracker

#: The original tracker's weak pose is no longer frozen 1.5 u away from the last strong fix
#: (SFM_ORIGINAL_NO_FREEZE=0 restores the old freeze). Experimental branch only.
ORIGINAL_NO_FREEZE = os.environ.get("SFM_ORIGINAL_NO_FREEZE", "1") != "0"

#: Display-only stream of session landmarks to the operator map pane.
WORKMAP_POINTS_EVERY = 8
WORKMAP_POINTS_PER_UPDATE = 600

_STATUS_MODE: dict[str, tuple[str, bool]] = {
    "FAST_TRACK": ("TRACK", False),
    "RELOC_SEED": ("TRACK", False),
    "VO_ONLY": ("WEAK_TRACK", True),
    "DEAD_RECKON": ("WEAK_TRACK", True),
    "IMU_BRIDGE": ("WEAK_TRACK", True),
    "NO_POSE": ("LOST", True),
}


def _same_capture_stamp(stamp_a: float | None, stamp_b: float | None) -> bool:
    if stamp_a is None or stamp_b is None:
        return False
    return abs(float(stamp_a) - float(stamp_b)) <= 2e-9

def _workmap_session_id_for_dir(directory: str | Path) -> str:
    resolved = Path(directory).expanduser().resolve()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(resolved)))


def _map_confidence_mode(status: str, info: dict, profile) -> tuple[str, bool]:
    mode, weak = _STATUS_MODE[status]
    if str(info.get("reference_scope", "global")) == "workmap":
        return "WEAK_TRACK", True
    if not weak and int(info.get("map_inliers", 0)) < int(
        profile.reloc.frozen_pnp_thresholds["strong_inliers"]
    ):
        return "WEAK_TRACK", True
    return mode, weak


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
        workmap_session_dir: str | Path | None = None,
        workmap_eval_stage: str | None = None,
    ) -> None:
        self.assets = assets
        self.profile = profile
        self.provider = provider
        online_workmap_enabled = bool(
            getattr(getattr(profile, "online_workmap", None), "enabled", False)
        )
        if workmap_eval_stage is None:
            stage = "full" if online_workmap_enabled else "baseline"
        else:
            stage = str(workmap_eval_stage)
            if stage not in {"baseline", "reuse", "ba", "full"}:
                raise ValueError("workmap_eval_stage must be one of baseline|reuse|ba|full")
            if stage != "full" and os.environ.get("SFM_EVALUATION_ONLY") != "1":
                raise RuntimeError(
                    "non-full workmap evaluation stages require SFM_EVALUATION_ONLY=1"
                )
        if stage in {"reuse", "ba", "full"} and not online_workmap_enabled:
            raise ValueError(
                f"workmap evaluation stage {stage!r} requires online_workmap.enabled=true"
            )
        self.workmap_eval_stage = stage
        self.workmap_session_dir = (
            Path(workmap_session_dir).expanduser().resolve()
            if workmap_session_dir is not None
            else None
        )
        self.workmap_session_id = (
            _workmap_session_id_for_dir(self.workmap_session_dir)
            if online_workmap_enabled
            and stage != "baseline"
            and self.workmap_session_dir is not None
            else None
        )
        self._workmap_diagnostics = (
            {
                "workmap_session_id": self.workmap_session_id,
                "workmap_session_dir": str(self.workmap_session_dir),
            }
            if self.workmap_session_id is not None
            else {}
        )
        if stage == "baseline":
            self.trk = TwoRateTracker(assets, profile, provider)
        else:
            if self.workmap_session_dir is None:
                raise ValueError("an enabled online workmap tracker requires workmap_session_dir")
            try:
                from online_workmap import OnlineWorkMapTracker
            except ModuleNotFoundError as exc:
                if exc.name != "online_workmap":
                    raise
                raise RuntimeError(
                    f"workmap stage {stage!r} requested but the formal "
                    "OnlineWorkMapTracker module is not installed"
                ) from exc
            try:
                self.trk = OnlineWorkMapTracker(
                    assets,
                    profile,
                    provider,
                    workmap_session_dir=self.workmap_session_dir,
                    workmap_eval_stage=stage,
                    workmap_session_id=self.workmap_session_id,
                )
            except Exception as exc:
                raise RuntimeError("online workmap tracker construction failed") from exc
        self.map_frame = map_frame
        self.frame_source = frame_source
        self.state = RuntimeState()
        self.temporal_cache = InertTemporalCache()
        self._last_info: dict = {}
        self._fused_sample = None
        self._pose_continuity = PoseSourceConfirmation()
        self._pose_centers: list[np.ndarray] = []
        self._reseed_confirming = False
        self._last_output_pose: Pose | None = None
        self._stall_sampler = None
        if os.environ.get("SFM_STALL_SAMPLER") == "1" and self.workmap_session_dir is not None:
            from stall_sampler import StallSampler

            self._stall_sampler = StallSampler(self.workmap_session_dir / "stall_samples.jsonl")
        self._accepted_camera_axes_world: list[list[float]] | None = None

    @property
    def map_frame(self):
        return getattr(getattr(self, "trk", None), "map_frame", None)

    @map_frame.setter
    def map_frame(self, value) -> None:
        trk = getattr(self, "trk", None)
        if trk is not None:
            trk.map_frame = value

    @property
    def last_info(self) -> dict:
        return self._last_info

    # ---------- model lifecycle ----------
    def ensure_models(self) -> None:
        """Load MegaLoc + EDM and start the relocalizer thread."""
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
        self._pose_continuity = PoseSourceConfirmation()
        self._pose_centers.clear()
        self._reseed_confirming = False
        self._last_output_pose = None
        self._accepted_camera_axes_world = None

    # ---------- fusion / retrieval hooks ----------
    def observe_fused_state(self, sample) -> None:
        """Offer this frame's fused IMU sample to the fast loop.

        The tracker keeps the sample's yaw for its inertial bridge: when both
        visual PnP and visual dead reckoning fail, a fresh fused yaw rotates
        the last pose instead of reporting LOST. The bridge never confirms a
        fix -- it reports WEAK_TRACK, and `last_info` says whether it ran.
        """
        self._fused_sample = sample
        offer = getattr(self.trk, "observe_fused_state", None)
        if callable(offer):
            offer(sample)

    def attach_pose_guided(self, controller) -> None:
        """No-op. Pose-guided retrieval reorders the EDM tracker's *reference*
        candidate list by predicted pose. This backend retrieves with MegaLoc
        inside the background worker on a whole-image descriptor, and its fast
        loop does no retrieval at all, so there is no candidate ordering for a
        controller to steer."""

    def _offer_rgb_frame(self, frame: np.ndarray) -> None:
        rgb = frame if frame.ndim == 3 else None
        offer = getattr(self.trk, "offer_rgb_frame", None)
        if callable(offer):
            offer(rgb)

    # ---------- one frame ----------
    def localize_frame(self, frame: np.ndarray, capture_stamp: float | None = None) -> Pose | None:
        sampler = getattr(self, "_stall_sampler", None)
        if sampler is None:
            return self._localize_frame(frame, capture_stamp)
        sampler.begin(int(getattr(self, "_frames_seen", 0)))
        self._frames_seen = getattr(self, "_frames_seen", 0) + 1
        try:
            return self._localize_frame(frame, capture_stamp)
        finally:
            sampler.end()

    def _lost_info(self) -> dict:
        info = {
            "mode": "LOST",
            "next_mode": "LOST",
            "weak": True,
            "direct_status": "NO_POSE",
            "inliers": 0,
            "pose_status": "NONE",
            "workmap_output_source": "baseline",
            "workmap_mapping_healthy": bool(getattr(self.trk, "mapping_enabled", True)),
            "workmap_mapping_error": getattr(self.trk, "mapping_error", None),
        }
        info.update(getattr(self, "_workmap_diagnostics", {}))
        return info

    def _localize_frame(self, frame: np.ndarray, capture_stamp: float | None = None) -> Pose | None:
        try:
            stamp = float(capture_stamp)
        except (TypeError, ValueError, OverflowError):
            stamp = float("nan")
        if not math.isfinite(stamp):
            self._last_info = self._lost_info()
            return None
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.asarray(frame)
        target_w, target_h = (int(v) for v in self.profile.fast_loop.resolution)
        if (gray.shape[1], gray.shape[0]) != (target_w, target_h):
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        self._offer_rgb_frame(frame)

        # 1. One tracker step. An online-workmap tracker that fails while mapping drops its
        # mapping side effects and keeps localizing from the next frame.
        try:
            info = self.trk.step(gray, stamp)
        except Exception as exc:
            if not callable(getattr(self.trk, "disable_mapping", None)) or not self.trk.mapping_enabled:
                raise
            self.trk.disable_mapping(f"{type(exc).__name__}: {exc}")
            self._clear_tracking_history()
            self._last_info = self._lost_info()
            return None
        status = str(info["status"])
        mode, weak = _map_confidence_mode(status, info, self.profile)

        pose = None
        yaw = info.get("yaw")
        rotation = info.get("R")
        rotation = None if rotation is None else np.asarray(rotation, dtype=float)
        if (
            self.map_frame is not None
            and rotation is not None
            and rotation.shape == (3, 3)
            and np.isfinite(rotation).all()
        ):
            yaw = float(self.map_frame.heading(rotation[2]))
        if info.get("ok") and info.get("center") is not None and yaw is not None:
            center = np.asarray(info["center"], dtype=float)
            pose = Pose(
                x=float(center[0]),
                y=float(center[1]),
                z=float(center[2]),
                yaw=float(yaw),
                stamp=stamp,
            )

        pose, mode, weak, confirming = self._confirm_pose(pose, info, mode, weak, status)
        # Advance RuntimeState every frame before output selection.
        self._advance_state(pose, info, mode, status)

        fresh = (
            pose is not None
            and _same_capture_stamp(pose.stamp, stamp)
            and status in {"FAST_TRACK", "RELOC_SEED", "VO_ONLY"}
            and not confirming
            and math.isfinite(pose.x)
            and math.isfinite(pose.y)
            and math.isfinite(pose.z)
            and math.isfinite(pose.yaw)
        )
        # "workmap" only labels a published pose that actually used session landmarks.
        workmap_output_source = (
            "workmap" if fresh and int(info.get("workmap_pose_landmarks", 0)) > 0 else "baseline"
        )
        # A rejected jump publishes the previous pose, not this frame's candidate.
        # Keep the camera basis from the same accepted pose until confirmation.
        if not confirming:
            self._accepted_camera_axes_world = (
                rotation.tolist()
                if pose is not None
                and rotation is not None
                and rotation.shape == (3, 3)
                and np.isfinite(rotation).all()
                else None
            )
        camera_axes_world = self._accepted_camera_axes_world if pose is not None else None
        camera_forward_world = camera_axes_world[2] if camera_axes_world is not None else None

        references = list(info.get("reference_names") or ())
        self._last_info = {
            "mode": mode,
            "next_mode": mode,
            "weak": weak,
            "reference_scope": str(info.get("reference_scope", "global")),
            "inliers": int(info.get("inliers", 0)),
            "reproj_rms": info.get("reproj_rms"),
            "reproj_p90": info.get("reproj_p90"),
            "inlier_hull_coverage": info.get("inlier_hull_coverage"),
            "positive_depth_ratio": info.get("positive_depth_ratio"),
            "inlier_ratio": (
                None
                if not info.get("n_corr")
                else float(info.get("inliers", 0)) / float(info["n_corr"])
            ),
            "inlier_grid_cells": info.get("inlier_grid_cells"),
            "vpr_ms": None,
            "feature_ms": None,
            "match_ms": None,
            "total_ms": float(info.get("loop_ms", 0.0)),
            "stage_gray_ms": info.get("gray_ms"),
            "stage_bridge_ms": None,
            "stage_query_ms": None,
            "stage_select_ms": None,
            "track_ms": float(info.get("track_ms", 0.0)),
            "handover_ms": info.get("handover_ms"),
            "bookkeeping_ms": info.get("bookkeeping_ms"),
            "map_constraint_age_s": info.get("map_constraint_age_s"),
            "pnp_ms": float(info.get("pnp_ms", 0.0)),
            "vo_ms": float(info.get("vo_ms", 0.0)),
            "pnp_candidates": int(info.get("n_corr", 0)),
            "pnp_skipped": None,
            "pnp_workers": None,
            "host_feature_cache": None,
            "n_corr": int(info.get("n_corr", 0)),
            "refs": references,
            "reference_count": len(references),
            "requested_reference_count": int(self.profile.reloc.top_k),
            "camera_axes_world": camera_axes_world,
            "camera_forward_world": camera_forward_world,
            # The bridge rotates the camera basis around measured map gravity,
            # but only fused yaw is observed; retained visual tilt is not a new
            # camera pitch/roll estimate.
            "camera_attitude_kind": (
                "heading_only"
                if status == "IMU_BRIDGE"
                else "full"
                if camera_axes_world is not None
                else "none"
            ),
            "pose_status": "VISUALLY_CONFIRMED" if (info.get("ok") and not weak) else "NONE",
            "prediction_valid": False,
            "prediction_mode": None,
            "predicted_center": None,
            "predicted_yaw": None,
            # --- direct-backend specific (Contract C4) ---
            "direct_status": status,
            "map_inliers": int(info.get("map_inliers", 0)),
            "vo_inliers": int(info.get("vo_inliers", 0)),
            "live_points": int(info.get("live_points", 0)),
            "dead_reckon_age": int(info.get("dead_reckon_age", 0)),
            "reloc_status": info.get("reloc_status"),
            "handover_points": int(info.get("handover_points", 0)),
            "handover_dropped": int(info.get("handover_dropped", 0)),
            "handover_capture_age_s": info.get("handover_capture_age_s"),
            "reloc_ms": info.get("reloc_ms", 0.0),
            "reloc_retrieval_ms": info.get("reloc_retrieval_ms"),
            "reloc_match_lift_ms": info.get("reloc_match_lift_ms"),
            "reloc_edm_match_ms": info.get("reloc_edm_match_ms"),
            "reloc_lift_prepare_ms": info.get("reloc_lift_prepare_ms"),
            "reloc_pnp_ms": info.get("reloc_pnp_ms"),
            "reloc_reference_count": info.get("reloc_reference_count"),
            "reloc_capture_stamp_mono": info.get("reloc_capture_stamp_mono"),
            "reloc_source_epoch": info.get("reloc_source_epoch"),
            "reloc_ordinal": info.get("reloc_ordinal"),
            "reloc_delivered": bool(info.get("reloc_delivered")),
            "fused_state_seen": self._fused_sample is not None,
            "imu_bridge": bool(info.get("imu_bridge")),
            "imu_sample_age_s": info.get("imu_sample_age_s"),
            "imu_yaw_delta_deg": info.get("imu_yaw_delta_deg"),
            "reloc_busy": bool(info.get("reloc_busy")),
            "vo_candidates": int(info.get("vo_candidates", 0)),
            "step_norm": info.get("step"),
            "reloc_reference_names": references,
            "reloc_submitted": bool(info.get("reloc_submitted")),
        }
        self._last_info["reference_scope"] = str(info.get("reference_scope", "global"))
        self._last_info.update({key: value for key, value in info.items() if key.startswith("workmap_")})
        self._last_info["workmap_output_source"] = workmap_output_source
        self._last_info["workmap_mapping_healthy"] = bool(getattr(self.trk, "mapping_enabled", True))
        self._last_info["workmap_mapping_error"] = getattr(self.trk, "mapping_error", None)
        self._last_info.update(getattr(self, "_workmap_diagnostics", {}))

        for key in (
            "frame_stamp_mono",
            "frame_dt_s",
            "klt_reseeded",
            "klt_input_points",
            "klt_forward_valid",
            "klt_backward_valid",
            "klt_kept_points",
            "klt_inside_kept",
            "klt_fb_p50_px",
            "klt_fb_p95_px",
            "klt_displacement_p50_px",
            "pnp_image_size_px",
            "imu_bridge_reason",
            "imu_reference_stamp_mono",
            "imu_sample_stamp_mono",
            "pnp_observation_sample",
        ):
            if key in info:
                self._last_info[key] = info[key]
        self._last_info["workmap_points_delta"] = self._workmap_points_delta()
        self._last_info["pose_source_confirming"] = confirming
        self._last_info["reseed_confirming"] = getattr(self, "_reseed_confirming", False)
        self._last_info["max_pose_jump_u"] = self.profile.max_jump_u
        self._last_info["pose_candidate_u"] = (
            None if info.get("center") is None else np.asarray(info["center"]).tolist()
        )
        if pose is not None and _same_capture_stamp(pose.stamp, stamp):
            self._last_output_pose = pose
        return pose

    def _workmap_points_delta(self) -> dict | None:
        """Bounded display update of session landmarks that are new or moved."""
        self._points_frame = getattr(self, "_points_frame", 0) + 1
        session = getattr(self.trk, "session_workmap", None)
        if (
            session is None
            or not getattr(self.trk, "mapping_enabled", True)
            or self._points_frame % WORKMAP_POINTS_EVERY
        ):
            return None
        ids, xyz, rgb, total = session.pop_display_points(WORKMAP_POINTS_PER_UPDATE)
        return {"ids": ids, "xyz": xyz, "rgb": rgb, "total": total} if ids else None

    def _confirm_pose(self, pose, info, mode, weak, status):
        gate = self._pose_continuity
        # True only while a strong fix is held to confirm the source a RELOC_SEED
        # started: the seed frame, then the FAST_TRACK frame whose label differs.
        reseed_run, self._reseed_confirming = self._reseed_confirming, False
        gate_source = "MAP_FIX" if str(status) in {"RELOC_SEED", "FAST_TRACK"} else status
        if str(status) == "RELOC_SEED" and not reseed_run:
            # Non-first RELOC_SEED frames still need two consistent captures;
            # mark the gate so the next consistent FAST_TRACK can release them.
            gate.needs_confirmation = True
        if pose is None:
            gate.interrupt()
            return pose, mode, weak, False
        center = np.array([pose.x, pose.y, pose.z], dtype=float)
        if not np.isfinite([pose.x, pose.y, pose.z, pose.yaw, pose.stamp]).all():
            gate.interrupt()
            return None, "LOST", True, False
        if str(info.get("reference_scope", "global")) == "workmap":
            # A current-frame local PnP is usable as a weak pose, but it neither
            # confirms nor replaces the old-release continuity anchor.
            gate.interrupt()
            return pose, "WEAK_TRACK", True, False
        if ORIGINAL_NO_FREEZE and weak:
            # A weak pose is judged against the pose published one frame earlier,
            # not against the last strong fix, so the output keeps updating while the
            # camera is outside the map. A one-frame jump above max_jump_u is still held,
            # and the next strong fix still needs its two-capture confirmation.
            previous = self.state.last_pose
            if previous is None or float(
                np.linalg.norm(center - [previous.x, previous.y, previous.z])
            ) <= self.profile.max_jump_u:
                gate.interrupt()
                return pose, mode, weak, False
        if self._pose_centers:
            reference = np.median(np.stack(self._pose_centers), axis=0)
            if np.linalg.norm(center - reference) > self.profile.max_jump_u:
                gate.needs_confirmation = True
        pending = gate.needs_confirmation
        accepted = gate.accept(
            center,
            pose.stamp,
            reliable=not weak,
            radius=self.profile.max_jump_u / 2.0,
            source=gate_source,
        )
        if not accepted:
            self._reseed_confirming = not weak and (
                str(status) in {"RELOC_SEED", "FAST_TRACK"} or reseed_run
            )
            trk = self.trk
            if hasattr(trk, "_speed_history"):
                trk._speed_history.clear()  # An estimator correction is not physical velocity.
            return self.state.last_pose, "WEAK_TRACK", True, True
        if not weak:
            if pending:
                self._pose_centers.clear()
            self._pose_centers.append(center)
            del self._pose_centers[:-5]
        return pose, mode, weak, False

    def _advance_state(self, pose: Pose | None, info: dict, mode: str, status: str) -> None:
        s = self.state
        s.prev_pose, s.prev_center, s.prev_yaw = s.last_pose, s.last_center, s.last_yaw
        s.mode = mode
        if pose is not None:
            s.last_pose = pose
            s.last_center = np.asarray([pose.x, pose.y, pose.z], dtype=np.float32)
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
        if not isinstance(sample, tuple) or len(sample) != 2:
            self._last_info = {
                "mode": "LOST",
                "next_mode": "LOST",
                "weak": True,
                "direct_status": "NO_POSE",
                "inliers": 0,
                "pose_status": "NONE",
                "error": "invalid_frame_sample",
            }
            return None
        frame, capture_stamp = sample
        return self.localize_frame(frame, capture_stamp=capture_stamp)
