"""Experimental NeuFlow-v2 keyframe refresh tracker.

This module deliberately lives under ``validation``.  Production continues to
use :class:`ProductionXFeatTracker` unchanged.  A deep localization refresh
seeds verified 2D<->3D inlier anchors; NeuFlow-v2 propagates only those anchors
on intermediate frames, followed by the existing PnP/RANSAC and safety gates.
Any failed flow attempt falls back to deep localization on the same frame.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pose_types import Pose
from production_xfeat_tracker import (
    ProductionXFeatTracker,
    _pose_center_yaw_from_ret,
    _ret_inlier_mask,
    predict_center,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_flow_at_points(
    flow: torch.Tensor,
    points_xy: np.ndarray,
    source_width: int,
    source_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample model-resolution flow at source-image points.

    The half-pixel transform matches OpenCV resize coordinates.  Only sampled
    vectors are copied to the CPU; the dense flow remains on the GPU.
    """
    if flow.ndim != 4 or flow.shape[0] != 1 or flow.shape[1] != 2:
        raise ValueError(f"expected flow [1,2,H,W], got {tuple(flow.shape)}")
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return points.copy(), np.zeros(0, dtype=bool)
    if source_width <= 1 or source_height <= 1:
        raise ValueError("source image must be at least 2x2")

    model_height, model_width = int(flow.shape[-2]), int(flow.shape[-1])
    scale_x = model_width / float(source_width)
    scale_y = model_height / float(source_height)
    model_points = torch.as_tensor(points, device=flow.device, dtype=flow.dtype)
    model_points = model_points.clone()
    model_points[:, 0] = (model_points[:, 0] + 0.5) * scale_x - 0.5
    model_points[:, 1] = (model_points[:, 1] + 0.5) * scale_y - 0.5

    grid = model_points.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(model_width - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(model_height - 1) - 1.0
    sampled = F.grid_sample(
        flow,
        grid.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, :, 0].transpose(0, 1)
    next_model = model_points + sampled
    valid = (
        torch.isfinite(next_model).all(dim=1)
        & (model_points[:, 0] >= 0.0)
        & (model_points[:, 0] <= model_width - 1.0)
        & (model_points[:, 1] >= 0.0)
        & (model_points[:, 1] <= model_height - 1.0)
        & (next_model[:, 0] >= 0.0)
        & (next_model[:, 0] <= model_width - 1.0)
        & (next_model[:, 1] >= 0.0)
        & (next_model[:, 1] <= model_height - 1.0)
    )
    next_source = next_model.float()
    next_source[:, 0] = (next_source[:, 0] + 0.5) / scale_x - 0.5
    next_source[:, 1] = (next_source[:, 1] + 0.5) / scale_y - 0.5
    return (
        next_source.detach().cpu().numpy().astype(np.float32, copy=False),
        valid.detach().cpu().numpy().astype(bool, copy=False),
    )


class NeuFlowV2Backend:
    """Official NeuFlow-v2 PyTorch model with sparse anchor readback."""

    def __init__(
        self,
        repo: Path,
        weights: Path,
        width: int = 512,
        height: int = 288,
        iters_s16: int = 1,
        iters_s8: int = 8,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.weights = Path(weights).resolve()
        self.width = int(width)
        self.height = int(height)
        self.iters_s16 = int(iters_s16)
        self.iters_s8 = int(iters_s8)
        if not (self.repo / "NeuFlow" / "neuflow.py").is_file():
            raise FileNotFoundError(f"NeuFlow-v2 source not found under {self.repo}")
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        if self.width % 16 or self.height % 16 or self.width <= 0 or self.height <= 0:
            raise ValueError("NeuFlow width/height must be positive multiples of 16")
        if not torch.cuda.is_available():
            raise RuntimeError("NeuFlow-v2 experiment requires CUDA")

        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from NeuFlow.backbone_v7 import ConvBlock
        from NeuFlow.neuflow import NeuFlow

        self._conv_block_type = ConvBlock
        self.device = torch.device("cuda")
        self.model = NeuFlow().to(self.device).eval()
        checkpoint = torch.load(
            self.weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self._fuse_backbone_batch_norms()
        self.model.half()
        self.model.init_bhwd(1, self.height, self.width, "cuda", amp=True)
        self._warmup()

    @staticmethod
    def _fused_conv(conv, batch_norm):
        fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).requires_grad_(False).to(conv.weight.device)
        weight = conv.weight.detach().clone().view(conv.out_channels, -1)
        scale = batch_norm.weight.detach() / torch.sqrt(
            batch_norm.running_var.detach() + batch_norm.eps)
        bias = (
            torch.zeros(conv.out_channels, device=conv.weight.device)
            if conv.bias is None else conv.bias.detach())
        with torch.no_grad():
            fused.weight.copy_((scale[:, None] * weight).view_as(conv.weight))
            fused.bias.copy_(
                batch_norm.bias.detach()
                + scale * (bias - batch_norm.running_mean.detach()))
        return fused

    def _fuse_backbone_batch_norms(self) -> None:
        for module in self.model.modules():
            if type(module) is self._conv_block_type:
                module.conv1 = self._fused_conv(module.conv1, module.norm1)
                module.conv2 = self._fused_conv(module.conv2, module.norm2)
                delattr(module, "norm1")
                delattr(module, "norm2")
                module.forward = module.forward_fuse

    def _prepare(self, frame_rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # The official inference script feeds cv2.imread() BGR tensors.
        bgr = np.ascontiguousarray(resized[:, :, ::-1])
        return (
            torch.from_numpy(bgr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float16)
        )

    @torch.inference_mode()
    def _warmup(self) -> None:
        image0 = torch.zeros(
            (1, 3, self.height, self.width), device=self.device, dtype=torch.float16)
        image1 = image0.clone()
        for _ in range(3):
            self.model(
                image0.clone(), image1.clone(),
                iters_s16=self.iters_s16, iters_s8=self.iters_s8)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def track_points(
        self,
        previous_rgb: np.ndarray,
        current_rgb: np.ndarray,
        points_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        if previous_rgb.shape[:2] != current_rgb.shape[:2]:
            raise ValueError("previous/current frame sizes differ")
        total_start = time.perf_counter()
        prepare_start = time.perf_counter()
        image0 = self._prepare(previous_rgb)
        image1 = self._prepare(current_rgb)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
        flow = self.model(
            image0, image1,
            iters_s16=self.iters_s16, iters_s8=self.iters_s8)[-1]
        next_points, valid = sample_flow_at_points(
            flow,
            points_xy,
            source_width=int(current_rgb.shape[1]),
            source_height=int(current_rgb.shape[0]),
        )
        gpu_end.record()
        gpu_end.synchronize()
        return next_points, valid, {
            "backend": "neuflow_v2",
            "prepare_ms": prepare_ms,
            "gpu_ms": float(gpu_start.elapsed_time(gpu_end)),
            "total_ms": (time.perf_counter() - total_start) * 1000.0,
        }

    def metadata(self) -> dict:
        return {
            "repo": str(self.repo),
            "weights": str(self.weights),
            "weights_sha256": _sha256(self.weights),
            "width": self.width,
            "height": self.height,
            "iters_s16": self.iters_s16,
            "iters_s8": self.iters_s8,
            "dtype": "float16",
        }


class NeuFlowRefreshTracker(ProductionXFeatTracker):
    """Validation-only deep-refresh / NeuFlow / PnP tracker."""

    def __init__(
        self,
        *args,
        neuflow_backend: NeuFlowV2Backend,
        refresh_interval: int = 3,
        min_seed: int = 100,
        min_track: int = 100,
        min_inliers: int = 50,
        min_inlier_ratio: float = 0.25,
        max_reproj: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.neuflow_backend = neuflow_backend
        self.neuflow_refresh_interval = int(refresh_interval)
        if self.neuflow_refresh_interval < 2:
            raise ValueError("NeuFlow refresh interval must be >= 2")
        self._nf_min_seed = int(min_seed)
        self._nf_min_track = int(min_track)
        self._nf_min_inliers = int(min_inliers)
        self._nf_min_inlier_ratio = float(min_inlier_ratio)
        self._nf_max_reproj = float(max_reproj)
        self._nf_2d: np.ndarray | None = None
        self._nf_3d: np.ndarray | None = None
        self._nf_previous_frame: np.ndarray | None = None
        self._nf_since_refresh = 0

    def _clear_tracking_history(self) -> None:
        super()._clear_tracking_history()
        if hasattr(self, "_nf_since_refresh"):
            self._clear_neuflow()

    def _clear_neuflow(self) -> None:
        self._nf_2d = None
        self._nf_3d = None
        self._nf_previous_frame = None
        self._nf_since_refresh = 0

    def _seed_neuflow(self, pose: Pose | None, frame: np.ndarray) -> bool:
        info = self._last_info
        strong = (
            pose is not None
            and bool(info.get("accepted", False))
            and not bool(info.get("weak", False))
            and int(info.get("inliers", 0) or 0) >= self._nf_min_seed
        )
        seed2d, seed3d = self._last_inl_2d, self._last_inl_3d
        if not strong or seed2d is None or seed3d is None:
            self._clear_neuflow()
            return False
        _, unique_2d = np.unique(seed2d, axis=0, return_index=True)
        unique_2d = np.sort(unique_2d)
        _, unique_3d = np.unique(seed3d[unique_2d], axis=0, return_index=True)
        unique = unique_2d[np.sort(unique_3d)]
        if len(unique) < self._nf_min_seed:
            self._clear_neuflow()
            return False
        self._nf_2d = seed2d[unique].astype(np.float32, copy=True)
        self._nf_3d = seed3d[unique].astype(np.float32, copy=True)
        self._nf_previous_frame = frame.copy()
        self._nf_since_refresh = 0
        return True

    def _deep_refresh(self, frame: np.ndarray, stage: str = "refresh") -> Pose | None:
        pose = self._localize_frame_deep(frame)
        seeded = self._seed_neuflow(pose, frame)
        self._last_info["neuflow_stage"] = stage
        self._last_info["neuflow_seeded"] = bool(seeded)
        self._last_info["neuflow_anchor_count"] = (
            0 if self._nf_2d is None else int(len(self._nf_2d)))
        return pose

    def _fallback_deep(
        self, frame: np.ndarray, stage: str, **attempt,
    ) -> Pose | None:
        self._clear_neuflow()
        pose = self._localize_frame_deep(frame)
        seeded = self._seed_neuflow(pose, frame)
        self._last_info["neuflow_stage"] = stage
        self._last_info["neuflow_fallback"] = True
        self._last_info["neuflow_seeded"] = bool(seeded)
        self._last_info["neuflow_anchor_count"] = (
            0 if self._nf_2d is None else int(len(self._nf_2d)))
        self._last_info.update(
            {f"neuflow_attempt_{key}": value for key, value in attempt.items()})
        return pose

    def localize_frame(
        self, frame: np.ndarray, capture_stamp: float | None = None,
    ) -> Pose | None:
        stamp = time.monotonic() if capture_stamp is None else float(capture_stamp)
        if not math.isfinite(stamp) or stamp > time.monotonic() + 0.05:
            self._last_info = {
                "mode": self.state.mode,
                "next_mode": self.state.mode,
                "error": "invalid_capture_stamp",
                "inliers": 0,
            }
            return None
        self._frame_capture_stamp = stamp
        self._frame_counters = {}

        if self.state.mode != "TRACK":
            self._clear_neuflow()
        refresh_due = self._nf_since_refresh >= self.neuflow_refresh_interval - 1
        if (
            self.state.mode != "TRACK"
            or self._nf_2d is None
            or self._nf_3d is None
            or self._nf_previous_frame is None
            or refresh_due
        ):
            return self._deep_refresh(frame)

        attempt_start = time.perf_counter()
        try:
            next2d_all, valid, backend_info = self.neuflow_backend.track_points(
                self._nf_previous_frame, frame, self._nf_2d)
        except RuntimeError as exc:
            return self._fallback_deep(
                frame, "backend_error", error=repr(exc))
        next2d = next2d_all[valid]
        next3d = self._nf_3d[valid]
        tracked = int(len(next3d))
        if tracked < self._nf_min_track:
            return self._fallback_deep(
                frame, "track_lost", tracked=tracked, **backend_info)

        pnp_start = time.perf_counter()
        ret = self._pnp(next2d, next3d)
        pnp_ms = (time.perf_counter() - pnp_start) * 1000.0
        inliers = int(ret.get("num_inliers", 0)) if ret is not None else 0
        reproj = (
            self._estimate_reproj_rms(ret, next2d, next3d)
            if ret is not None else None)
        ratio = inliers / max(1, tracked)
        if (
            ret is None
            or inliers < self._nf_min_inliers
            or ratio < self._nf_min_inlier_ratio
            or reproj is None
            or reproj > self._nf_max_reproj
            or reproj > self.cfg.max_reproj_error_track
        ):
            return self._fallback_deep(
                frame,
                "quality_gate",
                tracked=tracked,
                inliers=inliers,
                inlier_ratio=ratio,
                reproj_rms=reproj,
                pnp_ms=pnp_ms,
                **backend_info,
            )

        center, yaw = _pose_center_yaw_from_ret(ret)
        predicted_center = predict_center(self.state)
        jump = (
            0.0 if predicted_center is None
            else float(np.linalg.norm(center - predicted_center)))
        if predicted_center is not None and self.cfg.max_jump > 0 and jump > self.cfg.max_jump:
            return self._fallback_deep(
                frame,
                "jump_reject",
                tracked=tracked,
                inliers=inliers,
                inlier_ratio=ratio,
                reproj_rms=reproj,
                jump=jump,
                pnp_ms=pnp_ms,
                **backend_info,
            )

        pose = Pose(
            x=float(center[0]), y=float(center[1]), z=float(center[2]),
            yaw=float(yaw), stamp=float(self._frame_capture_stamp))
        inlier_mask = _ret_inlier_mask(ret, tracked)
        if inlier_mask is not None and int(inlier_mask.sum()) >= self._nf_min_track:
            self._nf_2d = next2d[inlier_mask].astype(np.float32, copy=True)
            self._nf_3d = next3d[inlier_mask].astype(np.float32, copy=True)
        else:
            self._nf_2d = next2d.astype(np.float32, copy=True)
            self._nf_3d = next3d.astype(np.float32, copy=True)
        self._nf_previous_frame = frame.copy()
        self._nf_since_refresh += 1

        self.state.prev_pose = self.state.last_pose
        self.state.prev_center = self.state.last_center
        self.state.prev_yaw = self.state.last_yaw
        self.state.last_pose = pose
        self.state.last_center = center.astype(np.float32)
        self.state.last_yaw = float(yaw)
        self.state.fail_count = 0
        self.state.bad_count = 0
        self.state.mode = "TRACK"
        self._last_info = {
            "mode": "NEUFLOW_TRACK",
            "next_mode": "TRACK",
            "inliers": inliers,
            "reproj_rms": reproj,
            "corr3d": tracked,
            "composite_stage": "neuflow",
            "neuflow_stage": "track",
            "neuflow_fallback": False,
            "neuflow_anchor_count": int(len(self._nf_2d)),
            "neuflow_inlier_ratio": ratio,
            "neuflow_jump_from_pred": jump,
            "neuflow_prepare_ms": backend_info["prepare_ms"],
            "neuflow_gpu_ms": backend_info["gpu_ms"],
            "neuflow_flow_ms": backend_info["total_ms"],
            "pnp_ms": pnp_ms,
            "weak": False,
            "accepted": True,
            "total_ms": (time.perf_counter() - attempt_start) * 1000.0,
            **self._frame_counters,
        }
        return pose
