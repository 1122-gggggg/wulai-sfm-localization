"""MoGe-2 metric depth estimation and scale lifting for online workmap expansion.

This module provides pure numpy helpers for sampling depth maps, fitting metric
scale against frozen site-map or session reference points, unprojecting image
pixels into world coordinates, and matching candidate seed keypoints for promotion.

MoGe-2 inference is encapsulated in ``MoGeDepthEstimator``, which requires the
optional MoGe layer (requirements/optional-workmap-depth-lock.txt).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import direct_paths

MOGE_NUM_TOKENS = 1200  # MoGe 建議範圍下限，最快
DEPTH_WINDOW_RADIUS_PX = 2  # 5×5 取樣窗
DEPTH_EDGE_MAX_REL_SPREAD = 0.10  # (max-min)/median 超過視為深度邊緣
DEPTH_SCALE_MIN_POINTS = 20
DEPTH_SCALE_MAX_SPREAD = 0.10  # median(|r/s-1|)
DEPTH_MAX_RANGE_RATIO = 3.0  # 相對尺度參考點中位深度
DEPTH_MIN_INTERVAL_S = 0.5
DEPTH_SCALE_HISTORY = 20
DEPTH_SCALE_HISTORY_MIN = 5
DEPTH_SCALE_MAX_RATIO = 1.25


def require_moge() -> None:
    """Fail-closed check for the optional MoGe depth estimation runtime."""
    try:
        from moge.model.v2 import MoGeModel  # noqa: F401
        import utils3d  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "online_workmap.expansion.depth_seeds requires the optional MoGe layer "
            f"(requirements/optional-workmap-depth-lock.txt) on PYTHONPATH: {error}"
        ) from error


def _sample_one_patch(
    depth: np.ndarray,
    u: int,
    v: int,
    radius: int,
    max_rel_spread: float,
    height: int,
    width: int,
) -> float:
    if u - radius < 0 or u + radius >= width or v - radius < 0 or v + radius >= height:
        return float("nan")
    patch = depth[v - radius : v + radius + 1, u - radius : u + radius + 1]
    if not np.all(np.isfinite(patch)) or not np.all(patch > 0.0):
        return float("nan")
    p_min = float(np.min(patch))
    p_max = float(np.max(patch))
    p_med = float(np.median(patch))
    if p_med <= 0.0 or (p_max - p_min) / p_med > max_rel_spread:
        return float("nan")
    return p_med


def sample_depth(
    depth: np.ndarray,
    xy: np.ndarray,
    *,
    radius: int = DEPTH_WINDOW_RADIUS_PX,
    max_rel_spread: float = DEPTH_EDGE_MAX_REL_SPREAD,
) -> np.ndarray:
    """Sample depth in a (2*radius+1) window around rounded pixel coordinates."""
    pts = np.asarray(xy, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    height, width = depth.shape[:2]
    out = np.empty(len(pts), dtype=np.float64)
    for i in range(len(pts)):
        u = int(round(float(pts[i, 0])))
        v = int(round(float(pts[i, 1])))
        out[i] = _sample_one_patch(depth, u, v, radius, max_rel_spread, height, width)
    return out


def fit_depth_scale(
    z_map: np.ndarray,
    predicted: np.ndarray,
    *,
    min_points: int = DEPTH_SCALE_MIN_POINTS,
    max_spread: float = DEPTH_SCALE_MAX_SPREAD,
) -> tuple[float | None, float, int]:
    """Robust median scale fit between metric map z and predicted depth."""
    z = np.asarray(z_map, dtype=np.float64).reshape(-1)
    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    valid = np.isfinite(z) & (z > 0.0) & np.isfinite(pred) & (pred > 0.0)
    n = int(np.count_nonzero(valid))
    if n < min_points:
        return (None, float("inf"), n)
    r = z[valid] / pred[valid]
    s = float(np.median(r))
    if not np.isfinite(s) or s <= 0.0:
        return (None, float("inf"), n)
    spread = float(np.median(np.abs(r / s - 1.0)))
    if spread > max_spread:
        return (None, spread, n)
    return (s, spread, n)


def lift_pixels(
    xy: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    cam_from_world: np.ndarray,
) -> np.ndarray:
    """Unproject (u, v) pixels with camera-z into world coordinates."""
    pts = np.asarray(xy, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64).reshape(-1, 1)
    inv_K = np.linalg.inv(np.asarray(K, dtype=np.float64))
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pixels_h = np.hstack([pts, ones])
    rays = pixels_h @ inv_K.T
    p_cam = rays * d
    pose = np.asarray(cam_from_world, dtype=np.float64)
    R = pose[:3, :3]
    t = pose[:3, 3]
    return (p_cam - t) @ R


def lift_request(
    request: Mapping[str, Any],
    depth_map: np.ndarray,
    infer_ms: float,
) -> dict[str, Any]:
    """Fit depth scale using reference landmarks and lift seeds into 3D."""
    source_epoch = int(request["source_epoch"])
    bid = int(request["bid"])
    seed_xy = np.asarray(request["seed_xy"], dtype=np.float64).copy()
    ref_ids = np.asarray(request["ref_ids"], dtype=np.int64)
    ref_xy = np.asarray(request["ref_xy"], dtype=np.float64)
    ref_xyz = np.asarray(request["ref_xyz"], dtype=np.float64)
    pose = np.asarray(request["cam_from_world"], dtype=np.float64)

    ref_p_cam = ref_xyz @ pose[:3, :3].T + pose[:3, 3]
    z_map = ref_p_cam[:, 2]
    d = sample_depth(depth_map, ref_xy)

    positive_mask = ref_ids > 0
    valid_pairs = np.isfinite(z_map) & (z_map > 0.0) & np.isfinite(d) & (d > 0.0)
    frozen_count = int(np.count_nonzero(positive_mask & valid_pairs))

    if frozen_count >= DEPTH_SCALE_MIN_POINTS:
        scale_source = "frozen"
        eval_mask = positive_mask
    else:
        scale_source = "session"
        eval_mask = np.ones(len(ref_ids), dtype=bool)

    scale, scale_spread, scale_points = fit_depth_scale(z_map[eval_mask], d[eval_mask])

    if scale is None:
        reason = "scale_support" if scale_points < DEPTH_SCALE_MIN_POINTS else "scale_spread"
        return {
            "source_epoch": source_epoch,
            "bid": bid,
            "ok": False,
            "reason": reason,
            "scale": None,
            "scale_spread": float(scale_spread),
            "scale_points": int(scale_points),
            "scale_source": scale_source,
            "seed_xy": seed_xy,
            "xyz": np.full((len(seed_xy), 3), np.nan, dtype=np.float64),
            "valid": np.zeros(len(seed_xy), dtype=bool),
            "infer_ms": float(infer_ms),
        }

    used_z = z_map[eval_mask & valid_pairs]
    med_ref_z = float(np.median(used_z))
    max_allowed_z = DEPTH_MAX_RANGE_RATIO * med_ref_z

    raw_seed_d = sample_depth(depth_map, seed_xy)
    seed_depth = scale * raw_seed_d
    valid = np.isfinite(seed_depth) & (seed_depth > 1e-3) & (seed_depth <= max_allowed_z)
    xyz = lift_pixels(seed_xy, seed_depth, request["K"], request["cam_from_world"])
    xyz[~valid] = np.nan

    return {
        "source_epoch": source_epoch,
        "bid": bid,
        "ok": True,
        "reason": None,
        "scale": float(scale),
        "scale_spread": float(scale_spread),
        "scale_points": int(scale_points),
        "scale_source": scale_source,
        "seed_xy": seed_xy,
        "xyz": xyz,
        "valid": valid,
        "infer_ms": float(infer_ms),
    }


def _check_promotable_row(
    point_xyz: np.ndarray,
    tracked_pt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    max_reproj_px: float,
) -> tuple[bool, np.ndarray]:
    if not np.all(np.isfinite(point_xyz)):
        return False, np.full(3, np.nan, dtype=np.float64)
    p_cam = R @ point_xyz + t
    z = float(p_cam[2])
    if z <= 1e-3 or not math.isfinite(z):
        return False, np.full(3, np.nan, dtype=np.float64)
    u = (K[0, 0] * p_cam[0] + K[0, 1] * p_cam[1] + K[0, 2] * z) / z
    v = (K[1, 0] * p_cam[0] + K[1, 1] * p_cam[1] + K[1, 2] * z) / z
    err = math.hypot(u - float(tracked_pt[0]), v - float(tracked_pt[1]))
    if err <= max_reproj_px:
        return True, point_xyz.copy()
    return False, np.full(3, np.nan, dtype=np.float64)


def match_promotable_rows(
    result: Mapping[str, Any],
    seed_xy: np.ndarray,
    tracked_xy: np.ndarray,
    cam_from_world: np.ndarray,
    K: np.ndarray,
    max_reproj_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match tracked seeds against a depth lift result and verify reprojection."""
    seeds = np.asarray(seed_xy, dtype=np.float64)
    tracked = np.asarray(tracked_xy, dtype=np.float64)
    n = len(seeds)
    if n == 0 or not bool(result.get("ok", False)):
        return np.zeros(n, dtype=bool), np.full((n, 3), np.nan, dtype=np.float64)

    res_seeds = np.asarray(result["seed_xy"], dtype=np.float64)
    res_valid = np.asarray(result["valid"], dtype=bool)
    res_xyz = np.asarray(result["xyz"], dtype=np.float64)
    lut = {(float(pt[0]), float(pt[1])): idx for idx, pt in enumerate(res_seeds)}

    pose = np.asarray(cam_from_world, dtype=np.float64)
    R = pose[:3, :3]
    t = pose[:3, 3]
    K_mat = np.asarray(K, dtype=np.float64)

    mask = np.zeros(n, dtype=bool)
    out_xyz = np.full((n, 3), np.nan, dtype=np.float64)
    for i in range(n):
        key = (float(seeds[i, 0]), float(seeds[i, 1]))
        idx = lut.get(key)
        if idx is None or not res_valid[idx]:
            continue
        ok, xyz = _check_promotable_row(res_xyz[idx], tracked[i], R, t, K_mat, max_reproj_px)
        if ok:
            mask[i] = True
            out_xyz[i] = xyz
    return mask, out_xyz


class MoGeDepthEstimator:
    """Encapsulates the optional MoGe-2 ViT-S GPU depth estimator."""

    def __init__(self, checkpoint: Path, *, device: str = "cuda") -> None:
        self.checkpoint = Path(checkpoint)
        self._device = str(device)
        self._model: Any = None

    def load(self) -> None:
        """Verify checkpoint hash, load MoGe-2 weights, and warm up."""
        require_moge()
        direct_paths.verify_file_sha256(self.checkpoint, direct_paths.MOGE_CHECKPOINT_SHA256)
        from moge.model.v2 import MoGeModel

        self._model = MoGeModel.from_pretrained(str(self.checkpoint)).to(self._device).eval()
        warmup_rgb = np.zeros((540, 960, 3), dtype=np.uint8)
        warmup_K = np.array(
            [[700.0, 0.0, 480.0], [0.0, 700.0, 270.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.infer_depth(warmup_rgb, warmup_K)

    def infer_depth(self, rgb: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, float]:
        """Run MoGe-2 inference on an RGB image, returning (depth_map, infer_ms)."""
        if self._model is None:
            raise RuntimeError("MoGe depth model is not loaded")
        import torch

        t0 = time.perf_counter()
        H, W = rgb.shape[:2]
        tensor = torch.from_numpy(rgb).to(self._device).float().permute(2, 0, 1) / 255.0
        fov_x = math.degrees(2.0 * math.atan(float(W) / (2.0 * float(K[0, 0]))))
        with torch.inference_mode():
            out = self._model.infer(
                tensor,
                num_tokens=MOGE_NUM_TOKENS,
                fov_x=fov_x,
                apply_mask=True,
                force_projection=True,
                use_fp16=self._device.startswith("cuda"),
            )
            depth = out["depth"].float().cpu().numpy()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        depth[~np.isfinite(depth)] = np.nan
        return depth, infer_ms

    def lift(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run depth inference and lift seed pixels into 3D world landmarks."""
        depth, ms = self.infer_depth(request["rgb"], request["K"])
        return lift_request(request, depth, ms)
