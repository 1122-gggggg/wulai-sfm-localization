#!/usr/bin/env python3
"""Robust Sim(3) NED ↔ SfM Map alignment — Umeyama with sliding window.

Uses `flight_control.site_alignment.solve_similarity_alignment` (Umeyama, proper
rotation, positive scale) as the solver. This module adds:
- sliding window 10..50 samples (attachment §22-24)
- spatial diversity check (std > threshold)
- outlier rejection via residuals (3-sigma or fixed 0.3m)
- only visually-validated poses enter (caller responsibility, enforced via `add_sample(..., is_valid)`)

Never called with a single EDM pose (§22)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from flight_control.site_alignment import solve_similarity_alignment


@dataclass(frozen=True)
class Sim3Config:
    min_samples: int = 10
    max_samples: int = 50
    min_spatial_spread_m: float = 0.5
    max_residual_m: float = 0.3
    residual_sigma_factor: float = 3.0
    min_samples_for_outlier_rejection: int = 15


@dataclass
class _Sample:
    ned: np.ndarray  # (3,)
    map: np.ndarray  # (3,)
    timestamp: float
    label: str | None = None


@dataclass(frozen=True)
class Sim3Estimate:
    scale: float
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3
    residuals: tuple[float, ...]
    rmse: float
    max_residual: float
    inliers: int
    total: int

    def map_to_ned(self, p_map: np.ndarray) -> np.ndarray:
        p = np.asarray(p_map, dtype=float).reshape(3)
        return self.scale * (self.rotation @ p) + self.translation

    def ned_to_map(self, p_ned: np.ndarray) -> np.ndarray:
        p = np.asarray(p_ned, dtype=float).reshape(3)
        # P_ned = s R P_map + t => P_map = R^T (P_ned - t)/s
        return (self.rotation.T @ (p - self.translation)) / self.scale


class Sim3Alignment:
    def __init__(self, config: Sim3Config | None = None):
        self.config = config or Sim3Config()
        self._samples: deque[_Sample] = deque(maxlen=self.config.max_samples)
        self._last_estimate: Sim3Estimate | None = None

    def clear(self) -> None:
        self._samples.clear()
        self._last_estimate = None

    def add_sample(
        self,
        p_ned: tuple[float, float, float] | np.ndarray,
        p_map: tuple[float, float, float] | np.ndarray,
        *,
        timestamp: float | None = None,
        is_valid: bool = True,
        label: str | None = None,
    ) -> bool:
        """Add a paired correspondence. Returns False if rejected (invalid, non-finite, duplicate)."""
        if not is_valid:
            return False
        try:
            ned = np.asarray(p_ned, dtype=float).reshape(3)
            mp = np.asarray(p_map, dtype=float).reshape(3)
        except Exception:
            return False
        if not np.isfinite(ned).all() or not np.isfinite(mp).all():
            return False
        # duplicate check (unique map points required by Umeyama wrapper)
        for s in self._samples:
            if np.allclose(s.map, mp, atol=1e-6):
                return False
        ts = float(timestamp) if timestamp is not None and math.isfinite(timestamp) else float(len(self._samples))
        self._samples.append(_Sample(ned=ned, map=mp, timestamp=ts, label=label))
        return True

    def enough_samples(self) -> bool:
        return len(self._samples) >= self.config.min_samples

    def spatial_spread(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        arr = np.stack([s.map for s in self._samples], axis=0)
        # std of distances from centroid
        centroid = arr.mean(axis=0)
        dists = np.linalg.norm(arr - centroid, axis=1)
        return float(dists.std())

    def estimate(self) -> Sim3Estimate | None:
        if len(self._samples) < self.config.min_samples:
            return None
        if self.spatial_spread() < self.config.min_spatial_spread_m:
            return None
        # first solve with all samples
        return self._robust_estimate(list(self._samples))

    def _robust_estimate(self, samples: list[_Sample]) -> Sim3Estimate | None:
        if len(samples) < 3:
            return None
        map_pts = [s.map for s in samples]
        ned_pts = [s.ned for s in samples]
        labels = [s.label for s in samples]
        try:
            fit = solve_similarity_alignment(map_pts, ned_pts, labels=labels)
        except Exception:
            return None
        scale = float(fit.transform.scale)
        rotation = np.asarray(fit.transform.rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(fit.transform.translation, dtype=float).reshape(3)
        residuals = np.array(fit.quality.residuals_m, dtype=float)
        rmse = float(fit.quality.rmse_m)
        max_res = float(fit.quality.max_residual_m)
        # outlier rejection if enough samples and residuals large
        if len(samples) >= self.config.min_samples_for_outlier_rejection:
            # 3-sigma or fixed max
            sigma = float(residuals.std()) if len(residuals) > 1 else rmse
            thresh = min(
                self.config.max_residual_m,
                self.config.residual_sigma_factor * max(sigma, 1e-6),
            )
            # keep inliers
            inliers_idx = [i for i, r in enumerate(residuals) if r <= thresh]
            if len(inliers_idx) >= self.config.min_samples and len(inliers_idx) < len(samples):
                # re-solve with inliers
                try:
                    fit2 = solve_similarity_alignment(
                        [map_pts[i] for i in inliers_idx],
                        [ned_pts[i] for i in inliers_idx],
                        labels=[labels[i] for i in inliers_idx],
                    )
                    scale = float(fit2.transform.scale)
                    rotation = np.asarray(fit2.transform.rotation, dtype=float).reshape(3, 3)
                    translation = np.asarray(fit2.transform.translation, dtype=float).reshape(3)
                    residuals = np.array(fit2.quality.residuals_m, dtype=float)
                    rmse = float(fit2.quality.rmse_m)
                    max_res = float(fit2.quality.max_residual_m)
                    # store estimate with inlier count
                    est = Sim3Estimate(
                        scale=scale,
                        rotation=rotation,
                        translation=translation,
                        residuals=tuple(float(x) for x in residuals),
                        rmse=rmse,
                        max_residual=max_res,
                        inliers=len(inliers_idx),
                        total=len(samples),
                    )
                    self._last_estimate = est
                    return est
                except Exception:
                    pass
        est = Sim3Estimate(
            scale=scale,
            rotation=rotation,
            translation=translation,
            residuals=tuple(float(x) for x in residuals),
            rmse=rmse,
            max_residual=max_res,
            inliers=len(samples),
            total=len(samples),
        )
        self._last_estimate = est
        return est

    def last_estimate(self) -> Sim3Estimate | None:
        return self._last_estimate

    def to_dict(self) -> dict:
        est = self._last_estimate
        if est is None:
            return {"valid": False, "samples": len(self._samples)}
        return {
            "valid": True,
            "scale": est.scale,
            "rotation": est.rotation.tolist(),
            "translation": est.translation.tolist(),
            "rmse": est.rmse,
            "max_residual": est.max_residual,
            "inliers": est.inliers,
            "total": est.total,
            "spatial_spread": self.spatial_spread(),
        }
