"""On-demand RePoseD relative-motion check for the first limited jump."""
from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from edm_matcher import EDM_H, EDM_W


MOGE_REVISION = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
POSELIB_REVISION = "fa7280fee27f97aff31ae7f98bab7f583fac7d08"
TRANSLATION_NORM_EPS = 1e-8
STATUSES = frozenset({"agree", "disagree", "unavailable"})

POSELIB_RANSAC_OPTIONS = {
    "max_epipolar_error": 2.0,
    "max_reproj_error": 16.0,
    "min_iterations": 1000,
    "max_iterations": 1000,
    "seed": 0,
    "progressive_sampling": True,
    "monodepth_estimate_shift": False,
    "monodepth_weight_sampson": 1.0,
}
POSELIB_BUNDLE_OPTIONS = {
    "loss_type": "TRUNCATED_CAUCHY",
}


def poselib_option_dicts() -> tuple[dict, dict]:
    """Deterministic PoseLib option dictionaries for the pinned monodepth solver."""
    return dict(POSELIB_RANSAC_OPTIONS), dict(POSELIB_BUNDLE_OPTIONS)


def poselib_estimate_options() -> dict:
    """Pinned-commit options dict accepted by estimate_monodepth_relative_pose."""
    ransac, bundle = poselib_option_dicts()
    return {
        "ransac": {
            "min_iterations": int(ransac["min_iterations"]),
            "max_iterations": int(ransac["max_iterations"]),
            "seed": int(ransac["seed"]),
            "progressive_sampling": bool(ransac["progressive_sampling"]),
        },
        "bundle": dict(bundle),
        "max_errors": [
            float(ransac["max_reproj_error"]),
            float(ransac["max_epipolar_error"]),
        ],
        "estimate_shift": bool(ransac["monodepth_estimate_shift"]),
        "weight_sampson": float(ransac["monodepth_weight_sampson"]),
    }


@dataclass(frozen=True)
class RelativeMotionCheck:
    status: str
    reason: str
    matches: int
    inliers: int
    rotation_delta_deg: float | None
    translation_direction_delta_deg: float | None
    depth_ms: float
    match_ms: float
    solver_ms: float
    total_ms: float

    def as_json_dict(self) -> dict[str, Any]:
        if self.status not in STATUSES:
            raise ValueError(f"invalid relative-motion status: {self.status!r}")
        return {
            "status": str(self.status),
            "reason": str(self.reason),
            "matches": int(self.matches),
            "inliers": int(self.inliers),
            "rotation_delta_deg": _json_float(self.rotation_delta_deg),
            "translation_direction_delta_deg": _json_float(
                self.translation_direction_delta_deg
            ),
            "depth_ms": float(self.depth_ms),
            "match_ms": float(self.match_ms),
            "solver_ms": float(self.solver_ms),
            "total_ms": float(self.total_ms),
        }


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_sha256(path: str | Path, expected: str) -> str:
    artifact = Path(path).expanduser().resolve()
    trusted = str(expected).strip().lower()
    if len(trusted) != 64 or any(ch not in "0123456789abcdef" for ch in trusted):
        raise ValueError(f"reposed model_sha256 must be a 64-character digest: {expected!r}")
    if not artifact.is_file():
        raise ValueError(f"reposed model file does not exist: {artifact}")
    actual = _sha256_file(artifact)
    if not hmac.compare_digest(actual, trusted):
        raise ValueError(
            f"SHA-256 mismatch for {artifact}: expected {trusted}, got {actual}"
        )
    return actual


def pinhole_fov_x_deg(width: int, fx: float) -> float:
    return float(math.degrees(2.0 * math.atan(float(width) / (2.0 * float(fx)))))


def cam_from_world_matrix(transform) -> np.ndarray:
    if isinstance(transform, np.ndarray):
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape == (4, 4):
            return matrix
        raise ValueError("cam_from_world matrix must be 4x4")
    rotation = np.asarray(transform.rotation.matrix(), dtype=float)
    translation = np.asarray(transform.translation, dtype=float).reshape(3)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def relative_cam_from_cam(previous, candidate) -> np.ndarray:
    previous_mat = cam_from_world_matrix(previous)
    candidate_mat = cam_from_world_matrix(candidate)
    return candidate_mat @ np.linalg.inv(previous_mat)


def rotation_geodesic_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = first.T @ second
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def translation_direction_delta_deg(
    first: np.ndarray,
    second: np.ndarray,
    *,
    eps: float = TRANSLATION_NORM_EPS,
) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < eps or second_norm < eps:
        return None
    cosine = float(
        np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0)
    )
    return float(math.degrees(math.acos(cosine)))


def scale_edm_keypoints(keypoints: np.ndarray, camera) -> np.ndarray:
    points = np.asarray(keypoints, dtype=float)
    if points.size == 0:
        return np.zeros((0, 2), dtype=float)
    scale = np.asarray(
        (float(camera.width) / float(EDM_W), float(camera.height) / float(EDM_H)),
        dtype=float,
    )
    return points.reshape(-1, 2) * scale


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_unit_interval(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return number


def _grid_cells(points: np.ndarray, width: float, height: float, grid: int) -> np.ndarray:
    gx = np.clip((points[:, 0] / max(width, 1e-6) * grid).astype(int), 0, grid - 1)
    gy = np.clip((points[:, 1] / max(height, 1e-6) * grid).astype(int), 0, grid - 1)
    return gy * grid + gx


def _quality_round_robin(cell: np.ndarray, score: np.ndarray, max_total: int) -> np.ndarray:
    count = int(cell.shape[0])
    original = np.arange(count, dtype=np.int64)
    within = np.lexsort((original, -score, cell))
    sorted_cells = cell[within]
    starts = np.r_[0, np.flatnonzero(sorted_cells[1:] != sorted_cells[:-1]) + 1]
    lengths = np.diff(np.r_[starts, count])
    ranks = np.arange(count) - np.repeat(starts, lengths)
    unique_cells = sorted_cells[starts]
    top_scores = score[within[starts]]
    cell_order = np.lexsort((unique_cells, -top_scores))
    priorities = np.empty(len(unique_cells), dtype=np.int64)
    priorities[cell_order] = np.arange(len(unique_cells), dtype=np.int64)
    row_priorities = priorities[np.repeat(np.arange(len(unique_cells)), lengths)]
    selected = np.lexsort((row_priorities, ranks))[:max_total]
    return within[selected]


def rank_and_cap_matches(
    mkpts0,
    mkpts1,
    mconf,
    max_matches: int,
    *,
    width: int = EDM_W,
    height: int = EDM_H,
    grid: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points0 = np.asarray(mkpts0, dtype=float).reshape(-1, 2)
    points1 = np.asarray(mkpts1, dtype=float).reshape(-1, 2)
    confidence = np.asarray(mconf, dtype=float).reshape(-1)
    if points0.shape[0] != points1.shape[0] or points0.shape[0] != confidence.shape[0]:
        raise ValueError("RePoseD match arrays must have the same length")
    if points0.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, np.zeros(0, dtype=float)
    score = np.where(np.isfinite(confidence), confidence, -np.inf)
    if max_matches <= 0 or points0.shape[0] <= int(max_matches) or int(grid) <= 1:
        order = np.argsort(-score, kind="stable")
        if max_matches > 0:
            order = order[: int(max_matches)]
        return points0[order], points1[order], confidence[order]
    width = _require_positive_int("match_width", int(width))
    height = _require_positive_int("match_height", int(height))
    grid = _require_positive_int("match_grid", int(grid))
    cell0 = _grid_cells(points0, float(width), float(height), grid)
    cell1 = _grid_cells(points1, float(width), float(height), grid)
    cell = cell0.astype(np.int64) * int(grid * grid) + cell1.astype(np.int64)
    order = _quality_round_robin(cell, score, int(max_matches))
    return points0[order], points1[order], confidence[order]


def match_spatial_support(
    points0: np.ndarray,
    points1: np.ndarray,
    *,
    width: int,
    height: int,
    grid: int,
    mask: np.ndarray | None = None,
) -> int:
    pts0 = np.asarray(points0, dtype=float).reshape(-1, 2)
    pts1 = np.asarray(points1, dtype=float).reshape(-1, 2)
    if pts0.shape[0] == 0:
        return 0
    if mask is not None:
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        if keep.shape[0] != pts0.shape[0]:
            return -1
        pts0 = pts0[keep]
        pts1 = pts1[keep]
        if pts0.shape[0] == 0:
            return 0
    cell0 = _grid_cells(pts0, float(width), float(height), int(grid))
    cell1 = _grid_cells(pts1, float(width), float(height), int(grid))
    paired = cell0.astype(np.int64) * int(grid * grid) + cell1.astype(np.int64)
    return int(np.unique(paired).size)


def _inlier_mask(info, count: int) -> np.ndarray | None:
    if info is None:
        return None
    values = None
    if isinstance(info, dict):
        values = info.get("inlier_mask", info.get("inliers"))
    elif hasattr(info, "inlier_mask"):
        values = info.inlier_mask
    elif hasattr(info, "inliers"):
        values = info.inliers
    if values is None:
        return None
    mask = np.asarray(values)
    if mask.dtype != bool:
        if np.issubdtype(mask.dtype, np.integer) and mask.ndim == 1 and mask.size != count:
            binary = np.zeros(count, dtype=bool)
            valid = (mask >= 0) & (mask < count)
            binary[mask[valid]] = True
            return binary
        mask = mask.astype(bool)
    if mask.ndim != 1 or mask.shape[0] != count:
        return None
    return mask



def poselib_camera_dict(camera) -> dict[str, Any]:
    return {
        "model": str(camera.model),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(value) for value in camera.params],
    }


def _require_pinhole(camera) -> tuple[int, int, float]:
    model = str(getattr(camera, "model", "")).strip().upper()
    if model != "PINHOLE":
        raise ValueError(f"RePoseD requires a PINHOLE camera, got {model!r}")
    width = int(camera.width)
    height = int(camera.height)
    params = [float(value) for value in camera.params]
    if width <= 0 or height <= 0 or len(params) < 4:
        raise ValueError("RePoseD PINHOLE camera needs width, height, and [fx, fy, cx, cy]")
    fx = params[0]
    if not math.isfinite(fx) or fx <= 0.0:
        raise ValueError("RePoseD PINHOLE fx must be finite and positive")
    return width, height, fx


def _bgr_pair_to_rgb_nchw(previous_bgr: np.ndarray, current_bgr: np.ndarray):
    import torch

    previous = np.asarray(previous_bgr)
    current = np.asarray(current_bgr)
    if previous.ndim != 3 or current.ndim != 3 or previous.shape[2] != 3 or current.shape[2] != 3:
        raise ValueError("RePoseD expects BGR images with shape (H, W, 3)")
    stacked = np.stack((previous[..., ::-1], current[..., ::-1]), axis=0)
    return torch.from_numpy(np.ascontiguousarray(stacked)).permute(0, 3, 1, 2).float().div_(255.0)


def _sample_maps(maps, pixels):
    import torch
    import torch.nn.functional as F

    count = int(pixels.shape[0])
    if count == 0:
        return maps.new_zeros((maps.shape[0], 0))
    height = int(maps.shape[-2])
    width = int(maps.shape[-1])
    xs = pixels[:, 0]
    ys = pixels[:, 1]
    grid_x = xs.mul(2.0).div_(max(width - 1, 1)).sub_(1.0)
    grid_y = ys.mul(2.0).div_(max(height - 1, 1)).sub_(1.0)
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, count, 2)
    grid = grid.expand(maps.shape[0], -1, -1, -1)
    sampled = F.grid_sample(
        maps.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, 0, 0, :]


class RePoseDMotionValidator:
    """Lazy MoGe + PoseLib check. Construction imports neither package."""

    def __init__(
        self,
        camera,
        matcher,
        *,
        model_path: str | Path,
        model_sha256: str,
        num_tokens: int = 1200,
        max_matches: int = 1200,
        min_inliers: int = 30,
        max_rotation_delta_deg: float = 3.0,
        max_translation_direction_delta_deg: float = 15.0,
        min_inlier_ratio: float = 0.0,
        min_spatial_support: int = 0,
        match_grid: int = 1,
        support_grid: int = 8,
        moge_model=None,
        poselib_module=None,
        estimate_fn=None,
        device: str | None = None,
    ):
        _require_pinhole(camera)
        self.camera = camera
        self.matcher = matcher
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_sha256 = str(model_sha256).strip().lower()
        self.num_tokens = int(num_tokens)
        self.max_matches = int(max_matches)
        self.min_inliers = int(min_inliers)

        self.max_rotation_delta_deg = float(max_rotation_delta_deg)
        self.max_translation_direction_delta_deg = float(
            max_translation_direction_delta_deg
        )
        self.min_inlier_ratio = _require_unit_interval("min_inlier_ratio", min_inlier_ratio)
        self.min_spatial_support = _require_non_negative_int(
            "min_spatial_support", int(min_spatial_support)
        )
        self.match_grid = _require_positive_int("match_grid", int(match_grid))
        self.support_grid = _require_positive_int("support_grid", int(support_grid))
        self._injected_model = moge_model
        self._injected_poselib = poselib_module
        self._estimate_fn = estimate_fn
        self._device = device
        self._model = moge_model
        self._poselib = poselib_module
        self._model_digest: str | None = None


    def check(
        self,
        previous_bgr,
        current_bgr,
        previous_cam_from_world,
        candidate_cam_from_world,
    ) -> RelativeMotionCheck:
        started = time.perf_counter()
        depth_ms = match_ms = solver_ms = 0.0
        try:
            self._ensure_runtime()
            predicted = relative_cam_from_cam(
                previous_cam_from_world,
                candidate_cam_from_world,
            )
            depth_ms, depths, masks = self._infer_depths(previous_bgr, current_bgr)
            match_started = time.perf_counter()
            matches = self.matcher.match(previous_bgr, current_bgr)
            match_ms = (time.perf_counter() - match_started) * 1e3
            points1, points2, depths1, depths2, valid_count = self._correspondences(
                matches,
                depths,
                masks,
            )
            if valid_count < self.min_inliers:
                return self._result(
                    "unavailable",
                    "too_few_valid_matches",
                    matches=valid_count,
                    inliers=0,
                    depth_ms=depth_ms,
                    match_ms=match_ms,
                    solver_ms=0.0,
                    started=started,
                )
            solver_started = time.perf_counter()
            geometry, info = self._solve_relative_pose(
                points1,
                points2,
                depths1,
                depths2,
            )
            solver_ms = (time.perf_counter() - solver_started) * 1e3
            return self._compare_geometry(
                geometry,
                info,
                predicted,
                valid_count,
                depth_ms,
                match_ms,
                solver_ms,
                started,
                points1,
                points2,
            )

        except Exception as exc:
            return self._result(
                "unavailable",
                f"solver_error:{type(exc).__name__}:{exc}",
                matches=0,
                inliers=0,
                depth_ms=depth_ms,
                match_ms=match_ms,
                solver_ms=solver_ms,
                started=started,
            )

    def _ensure_runtime(self) -> None:
        if self._model is None:
            verify_model_sha256(self.model_path, self.model_sha256)
            from moge.model.v2 import MoGeModel

            device = self._resolve_device()
            self._model = MoGeModel.from_pretrained(self.model_path).to(device).eval()
            self._model_digest = self.model_sha256
        if self._poselib is None and self._estimate_fn is None:
            import poselib

            if not hasattr(poselib, "estimate_monodepth_relative_pose"):
                raise RuntimeError("poselib is missing estimate_monodepth_relative_pose")
            self._poselib = poselib

    def _resolve_device(self) -> str:
        if self._device:
            return str(self._device)
        matcher_device = getattr(self.matcher, "device", None)
        if matcher_device:
            return str(matcher_device)
        return "cuda"

    def _infer_depths(self, previous_bgr, current_bgr):
        import torch

        width, _height, fx = _require_pinhole(self.camera)
        images = _bgr_pair_to_rgb_nchw(previous_bgr, current_bgr)
        device = next(self._model.parameters()).device
        images = images.to(device=device, dtype=torch.float32)
        fov_x = pinhole_fov_x_deg(width, fx)
        start_event = end_event = None
        wall_started = time.perf_counter()
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        output = self._model.infer(
            images,
            num_tokens=self.num_tokens,
            fov_x=fov_x,
            force_projection=True,
            apply_mask=True,
            use_fp16=True,
        )
        if end_event is not None:
            end_event.record()
            end_event.synchronize()
            depth_ms = float(start_event.elapsed_time(end_event))
        else:
            depth_ms = (time.perf_counter() - wall_started) * 1e3
        depths = output["depth"]
        masks = output.get("mask")
        if masks is None:
            masks = torch.isfinite(depths) & (depths > 0)
        return depth_ms, depths, masks.to(dtype=depths.dtype)

    def _correspondences(self, matches: dict, depths, masks):
        import torch

        points0, points1, _confidence = rank_and_cap_matches(
            matches.get("mkpts0"),
            matches.get("mkpts1"),
            matches.get("mconf"),
            self.max_matches,
            width=EDM_W,
            height=EDM_H,
            grid=self.match_grid,
        )

        points0 = scale_edm_keypoints(points0, self.camera)
        points1 = scale_edm_keypoints(points1, self.camera)
        if points0.shape[0] == 0:
            empty = np.zeros((0, 2), dtype=np.float64)
            empty_depth = np.zeros(0, dtype=np.float64)
            return empty, empty, empty_depth, empty_depth, 0
        device = depths.device
        pixels0 = torch.from_numpy(np.ascontiguousarray(points0)).to(
            device=device, dtype=torch.float32
        )
        pixels1 = torch.from_numpy(np.ascontiguousarray(points1)).to(
            device=device, dtype=torch.float32
        )
        packed = torch.stack(
            (
                _sample_maps(depths, pixels0)[0],
                _sample_maps(depths, pixels1)[1],
                _sample_maps(masks, pixels0)[0],
                _sample_maps(masks, pixels1)[1],
            ),
            dim=0,
        )
        sampled = packed.detach().cpu().numpy()
        depth0, depth1, mask0, mask1 = sampled
        valid = (
            np.isfinite(points0).all(axis=1)
            & np.isfinite(points1).all(axis=1)
            & np.isfinite(depth0)
            & np.isfinite(depth1)
            & (depth0 > 0.0)
            & (depth1 > 0.0)
            & (mask0 > 0.5)
            & (mask1 > 0.5)
        )
        return (
            points0[valid].astype(np.float64, copy=False),
            points1[valid].astype(np.float64, copy=False),
            depth0[valid].astype(np.float64, copy=False),
            depth1[valid].astype(np.float64, copy=False),
            int(np.count_nonzero(valid)),
        )

    def _solve_relative_pose(self, points1, points2, depths1, depths2):
        camera = poselib_camera_dict(self.camera)
        ransac, bundle = poselib_option_dicts()
        if self._estimate_fn is not None:
            return self._estimate_fn(
                points1,
                points2,
                depths1,
                depths2,
                camera,
                camera,
                ransac,
                bundle,
            )
        return self._poselib.estimate_monodepth_relative_pose(
            points1,
            points2,
            depths1,
            depths2,
            camera,
            camera,
            poselib_estimate_options(),
        )

    def _compare_geometry(
        self,
        geometry,
        info,
        predicted: np.ndarray,
        matches: int,
        depth_ms: float,
        match_ms: float,
        solver_ms: float,
        started: float,
        points1=None,
        points2=None,
    ) -> RelativeMotionCheck:
        inliers = _inlier_count(info)
        if inliers < self.min_inliers:
            return self._result(
                "unavailable",
                "too_few_inliers",
                matches=matches,
                inliers=inliers,
                depth_ms=depth_ms,
                match_ms=match_ms,
                solver_ms=solver_ms,
                started=started,
            )
        ratio = float(inliers) / float(max(int(matches), 1))
        if self.min_inlier_ratio > 0.0 and ratio < self.min_inlier_ratio:
            return self._result(
                "unavailable",
                "inlier_ratio",
                matches=matches,
                inliers=inliers,
                depth_ms=depth_ms,
                match_ms=match_ms,
                solver_ms=solver_ms,
                started=started,
            )
        if self.min_spatial_support > 0:
            if points1 is None or points2 is None:
                return self._result(
                    "unavailable",
                    "spatial_support",
                    matches=matches,
                    inliers=inliers,
                    depth_ms=depth_ms,
                    match_ms=match_ms,
                    solver_ms=solver_ms,
                    started=started,
                )
            mask = _inlier_mask(info, int(np.asarray(points1).shape[0]))
            if mask is None:
                return self._result(
                    "unavailable",
                    "spatial_support",
                    matches=matches,
                    inliers=inliers,
                    depth_ms=depth_ms,
                    match_ms=match_ms,
                    solver_ms=solver_ms,
                    started=started,
                )
            support = match_spatial_support(
                points1,
                points2,
                width=int(self.camera.width),
                height=int(self.camera.height),
                grid=self.support_grid,
                mask=mask,
            )

            if support < 0 or support < self.min_spatial_support:
                return self._result(
                    "unavailable",
                    "spatial_support",
                    matches=matches,
                    inliers=inliers,
                    depth_ms=depth_ms,
                    match_ms=match_ms,
                    solver_ms=solver_ms,
                    started=started,
                )
        scale = float(getattr(geometry, "scale"))
        if not math.isfinite(scale) or scale <= 0.0:
            return self._result(
                "unavailable",
                "invalid_scale",
                matches=matches,
                inliers=inliers,
                depth_ms=depth_ms,
                match_ms=match_ms,
                solver_ms=solver_ms,
                started=started,
            )
        pose = geometry.pose
        estimated_rotation = np.asarray(pose.R, dtype=float)
        estimated_translation = np.asarray(pose.t, dtype=float).reshape(3) / scale
        predicted_rotation = predicted[:3, :3]
        predicted_translation = predicted[:3, 3]
        rotation_delta = rotation_geodesic_deg(estimated_rotation, predicted_rotation)
        translation_delta = translation_direction_delta_deg(
            estimated_translation,
            predicted_translation,
        )
        if translation_delta is None:
            return self._result(
                "unavailable",
                "degenerate_translation",
                matches=matches,
                inliers=inliers,
                rotation_delta_deg=rotation_delta,
                depth_ms=depth_ms,
                match_ms=match_ms,
                solver_ms=solver_ms,
                started=started,
            )
        agreed = (
            rotation_delta <= self.max_rotation_delta_deg
            and translation_delta <= self.max_translation_direction_delta_deg
        )
        return self._result(
            "agree" if agreed else "disagree",
            "within_thresholds" if agreed else "exceeds_thresholds",
            matches=matches,
            inliers=inliers,
            rotation_delta_deg=rotation_delta,
            translation_direction_delta_deg=translation_delta,
            depth_ms=depth_ms,
            match_ms=match_ms,
            solver_ms=solver_ms,
            started=started,
        )


    def _result(
        self,
        status: str,
        reason: str,
        *,
        matches: int,
        inliers: int,
        depth_ms: float,
        match_ms: float,
        solver_ms: float,
        started: float,
        rotation_delta_deg: float | None = None,
        translation_direction_delta_deg: float | None = None,
    ) -> RelativeMotionCheck:
        return RelativeMotionCheck(
            status=status,
            reason=reason,
            matches=int(matches),
            inliers=int(inliers),
            rotation_delta_deg=_json_float(rotation_delta_deg),
            translation_direction_delta_deg=_json_float(
                translation_direction_delta_deg
            ),
            depth_ms=float(depth_ms),
            match_ms=float(match_ms),
            solver_ms=float(solver_ms),
            total_ms=float((time.perf_counter() - started) * 1e3),
        )


def _inlier_count(info) -> int:
    if info is None:
        return 0
    if isinstance(info, dict):
        if "num_inliers" in info:
            return int(info["num_inliers"])
        inliers = info.get("inliers")
        if inliers is None:
            return 0
        return int(np.count_nonzero(np.asarray(inliers)))
    if hasattr(info, "num_inliers"):
        return int(info.num_inliers)
    return 0
