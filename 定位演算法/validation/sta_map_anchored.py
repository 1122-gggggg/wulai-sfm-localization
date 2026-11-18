"""Map-anchored STA relocalization: pure geometry for the offline experiment.

ViSTA-SLAM's STA model regresses, for an image pair (i, j), the relative pose
``T_ij`` that maps points from camera i into camera j (paper Eq. 6; in the
reference code ``pose_i = pose_j @ T_ij``) plus one pointmap per view in that
view's own camera frame.  Pose translation and both pointmaps share one
arbitrary per-pass scale.

This module turns one STA pass ``(query, reference)`` into a query pose in the
map frame:

* ``gamma`` rescales the pass to map units by comparing the reference view's
  STA depth with the depth of the reference keyframe's own triangulated map
  points (z only, so STA's implicit focal length does not enter);
* the query pose is ``world_T_ref @ ref_T_query`` with the translation scaled.

Nothing here touches EDM, KLT or PnP.  It is an offline research path only.

Experiment concluded NO-GO on 2026-09-23 (docs/sta_map_anchored_eval_20260923.md);
kept only for reproduction and for its geometry unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


STA_RESOLUTION = 224
STA_EDGE_PX = 10


@dataclass(frozen=True)
class StaCrop:
    """Continuous pixel mapping of ViSTA's ``_crop_resize_if_necessary_image_only``."""

    left: float
    top: float
    right: float
    bottom: float
    scale_x: float
    scale_y: float
    left2: float
    top2: float
    size: int

    def to_crop(self, xy: np.ndarray) -> np.ndarray:
        """Map COLMAP pixel coordinates (pixel i spans [i, i+1)) into the crop."""

        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        u = (xy[:, 0] - self.left) * self.scale_x - self.left2
        v = (xy[:, 1] - self.top) * self.scale_y - self.top2
        return np.column_stack([u, v])


def sta_crop(width: int, height: int, *, size: int = STA_RESOLUTION, edge: int = STA_EDGE_PX) -> StaCrop:
    """Replicate ViSTA's centre crop + Lanczos rescale + centre crop for one image size."""

    cx, cy = int(width / 2), int(height / 2)
    margin_x = min(cx, width - cx)
    margin_y = min(cy, height - cy)
    left = max(cx - margin_x, edge)
    top = max(cy - margin_y, edge)
    right = min(cx + margin_x, width - edge)
    bottom = min(cy + margin_y, height - edge)
    crop_w, crop_h = right - left, bottom - top
    scale_final = max(size / crop_w, size / crop_h) + 1e-8
    out_w = int(np.floor(crop_w * scale_final))
    out_h = int(np.floor(crop_h * scale_final))
    left2 = int(np.int32(np.round(out_w / 2 - size / 2)))
    top2 = int(np.int32(np.round(out_h / 2 - size / 2)))
    return StaCrop(
        left=float(left),
        top=float(top),
        right=float(right),
        bottom=float(bottom),
        scale_x=out_w / crop_w,
        scale_y=out_h / crop_h,
        left2=float(left2),
        top2=float(top2),
        size=size,
    )


def camera_center(cam_from_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(cam_from_world, dtype=float)
    return -pose[:3, :3].T @ pose[:3, 3]


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)
    cosine = (np.trace(relative) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def compose_query_pose(
    ref_cam_from_world: np.ndarray, ref_T_query: np.ndarray, gamma: float
) -> np.ndarray:
    """Return the query ``cam_from_world`` (3x4) from one scaled STA pass.

    ``ref_T_query`` is STA's 4x4 (or 3x4) ``T_{query,ref}``: X_ref = R X_q + t.
    """

    ref = np.asarray(ref_cam_from_world, dtype=float)
    rel = np.asarray(ref_T_query, dtype=float)
    world_R_ref = ref[:3, :3].T
    world_R_query = world_R_ref @ rel[:3, :3]
    center = world_R_ref @ (float(gamma) * rel[:3, 3]) + camera_center(ref)
    query_R_world = world_R_query.T
    return np.column_stack([query_R_world, -query_R_world @ center])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(values[order][index])


@dataclass(frozen=True)
class ScaleFit:
    gamma: float | None
    samples: int
    log_mad: float | None


def estimate_gamma(
    map_depth: np.ndarray, sta_depth: np.ndarray, weights: np.ndarray, *, min_samples: int
) -> ScaleFit:
    """Robust ``gamma`` with ``map_depth ~= gamma * sta_depth``.

    The weighted median of the per-sample ratio is used instead of ViSTA's
    least-squares Eq. 11 because map points include outliers (water, sky edges,
    occlusion) that a squared loss would let dominate.  ``log_mad`` is the
    weighted median absolute log residual: how consistently the two depth
    fields agree up to one scale.
    """

    z_map = np.asarray(map_depth, dtype=float).reshape(-1)
    z_sta = np.asarray(sta_depth, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    keep = (z_map > 0) & (z_sta > 0) & np.isfinite(z_map) & np.isfinite(z_sta) & (w > 0)
    count = int(keep.sum())
    if count < min_samples:
        return ScaleFit(None, count, None)
    ratio = z_map[keep] / z_sta[keep]
    gamma = weighted_median(ratio, w[keep])
    log_mad = weighted_median(np.abs(np.log(ratio / gamma)), w[keep])
    return ScaleFit(gamma, count, log_mad)


@dataclass(frozen=True)
class Hypothesis:
    reference: str
    video: str
    frame_index: int
    cam_from_world: np.ndarray
    pose_conf: float

    @property
    def center(self) -> np.ndarray:
        return camera_center(self.cam_from_world)


def independent(a: Hypothesis, b: Hypothesis, *, min_frame_gap: int) -> bool:
    """Different video, or the same video far enough apart to be another view."""

    return a.video != b.video or abs(a.frame_index - b.frame_index) >= min_frame_gap


@dataclass(frozen=True)
class Consensus:
    accepted: bool
    cam_from_world: np.ndarray | None
    members: tuple[str, ...]
    reason: str


def _mean_rotation(rotations: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    total = sum(w * r for r, w in zip(rotations, weights))
    u, _, vt = np.linalg.svd(total)
    fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return u @ fix @ vt


def pose_consensus(
    hypotheses: Sequence[Hypothesis],
    *,
    max_position: float,
    max_rotation_deg: float,
    min_frame_gap: int,
) -> Consensus:
    """Pose-space vote: accept only when two independent references agree."""

    if len(hypotheses) < 2:
        return Consensus(False, None, (), "fewer than two valid hypotheses")

    def agree(a: Hypothesis, b: Hypothesis) -> bool:
        return (
            float(np.linalg.norm(a.center - b.center)) <= max_position
            and rotation_angle_deg(a.cam_from_world[:3, :3], b.cam_from_world[:3, :3])
            <= max_rotation_deg
        )

    best: list[Hypothesis] = []
    best_key = (-1, -1.0)
    for anchor in hypotheses:
        support = [other for other in hypotheses if agree(anchor, other)]
        key = (len(support), sum(item.pose_conf for item in support))
        if key > best_key:
            best, best_key = support, key
    has_independent_pair = any(
        independent(a, b, min_frame_gap=min_frame_gap)
        for index, a in enumerate(best)
        for b in best[index + 1 :]
    )
    if not has_independent_pair:
        return Consensus(False, None, tuple(h.reference for h in best), "no independent agreeing pair")
    weights = [item.pose_conf for item in best]
    center = sum(w * item.center for item, w in zip(best, weights)) / sum(weights)
    world_R_cam = _mean_rotation([item.cam_from_world[:3, :3].T for item in best], weights)
    cam_R_world = world_R_cam.T
    pose = np.column_stack([cam_R_world, -cam_R_world @ center])
    return Consensus(True, pose, tuple(h.reference for h in best), "")
