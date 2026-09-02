#!/usr/bin/env python3
"""LocalMapManager — radius/KNN query around predicted pose, no O(N) scan every frame.

Uses KD-tree / spatial index over SfM point cloud (3D). Configurable count 500~2000.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    KDTree = None


@dataclass(frozen=True)
class LocalMapConfig:
    num_points: int = 1000
    radius_m: float = 5.0
    min_points: int = 500
    max_points: int = 2000


class LocalMapManager:
    def __init__(self, points3d: np.ndarray, config: LocalMapConfig | None = None):
        self.config = config or LocalMapConfig()
        self._points = np.asarray(points3d, dtype=float).reshape(-1, 3)
        if self._points.shape[0] == 0 or not np.isfinite(self._points).all():
            raise ValueError("points3d must be non-empty finite")
        if KDTree is not None:
            self._tree = KDTree(self._points)
        else:
            self._tree = None

    def query(self, center: np.ndarray | None, k: int | None = None, radius: float | None = None) -> np.ndarray:
        """Return indices of local map points near center."""
        if center is None or not np.isfinite(np.asarray(center)).all():
            # fallback: first N
            n = k or self.config.num_points
            return np.arange(min(n, len(self._points)))
        center = np.asarray(center, dtype=float).reshape(3)
        n = int(k or self.config.num_points)
        n = max(self.config.min_points, min(self.config.max_points, n))
        rad = float(radius or self.config.radius_m)
        if self._tree is not None:
            # radius + KNN hybrid
            idx_radius = self._tree.query_ball_point(center, rad)
            if len(idx_radius) >= n:
                # pick closest n among radius
                dists = np.linalg.norm(self._points[idx_radius] - center, axis=1)
                order = np.argsort(dists)[:n]
                return np.array([idx_radius[i] for i in order], dtype=int)
            # supplement with KNN
            k2 = min(n, len(self._points))
            dists, idx = self._tree.query(center, k=k2)
            if k2 == 1:
                return np.array([int(idx)], dtype=int)
            return np.asarray(idx, dtype=int).reshape(-1)
        # brute fallback O(N) but small map — still OK for test
        dists = np.linalg.norm(self._points - center, axis=1)
        return np.argsort(dists)[:n]

    def points_for_indices(self, indices: np.ndarray) -> np.ndarray:
        return self._points[np.asarray(indices, dtype=int)]
