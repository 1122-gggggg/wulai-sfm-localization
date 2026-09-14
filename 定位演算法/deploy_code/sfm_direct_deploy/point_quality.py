"""Deterministic spatial sampling and inexpensive pose diagnostics."""

import cv2
import numpy as np


def spatial_indices(xy: np.ndarray, cap: int, width: int, height: int) -> np.ndarray:
    """Round-robin an 8x6 image grid, retaining input order within each cell."""
    if len(xy) <= cap:
        return np.arange(len(xy))
    cells = np.clip((xy * [8 / width, 6 / height]).astype(int), [0, 0], [7, 5])
    cell = cells[:, 1] * 8 + cells[:, 0]
    order = np.argsort(cell, kind="stable")
    sorted_cells = cell[order]
    starts = np.maximum.accumulate(
        np.where(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]], np.arange(len(xy)), 0)
    )
    within_cell = np.empty(len(xy), dtype=int)
    within_cell[order] = np.arange(len(xy)) - starts
    take = np.lexsort((np.arange(len(xy)), within_cell))[:cap]
    return np.sort(take)


def pose_quality(xy, xyz, pose, camera_matrix, width, height) -> dict:
    """Measure only the actual PnP inliers; do not relax admission thresholds."""
    if not len(xy):
        return {}
    camera = xyz @ pose[:, :3].T + pose[:, 3]
    positive = camera[:, 2] > 1e-6
    projected = camera @ camera_matrix.T
    projected = projected[:, :2] / np.where(positive, camera[:, 2], np.nan)[:, None]
    errors = np.linalg.norm(projected - xy, axis=1)
    finite = errors[np.isfinite(errors)]
    cells = np.clip((xy * [4 / width, 4 / height]).astype(int), [0, 0], [3, 3])
    hull = cv2.convexHull(np.asarray(xy, dtype=np.float32))
    return {
        "reproj_rms": float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
        "reproj_p90": float(np.percentile(finite, 90)) if len(finite) else None,
        "inlier_grid_cells": int(len(np.unique(cells[:, 1] * 4 + cells[:, 0]))),
        "inlier_hull_coverage": float(cv2.contourArea(hull) / (width * height)),
        "positive_depth_ratio": float(positive.mean()),
    }
