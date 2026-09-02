#!/usr/bin/env python3
"""KLT temporal tracker — high-frequency, EDM-parallel.

EDM (low-freq global anchor) provides 2D-3D correspondences (x_i ↔ X_i^M).
KLT maintains:
  pixel_2d + map_point_3d + track_age + status + error

Do NOT output only dx/dy. Do NOT treat as VIO.

Quality: tracked_count, ratio, error, spatial coverage, inlier_ratio.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


_KLT_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


@dataclass
class KLTConfig:
    min_tracks: int = 30
    min_track_ratio: float = 0.4
    max_klt_error_px: float = 1.0
    min_spatial_coverage: int = 6  # grid cells 8x?
    fb_px: float = 1.0
    min_seed: int = 50
    max_age_frames: int = 6
    max_age_s: float = 0.5
    grid_w: int = 8
    grid_h: int = 6


@dataclass
class KLTTrack:
    pixel: np.ndarray  # (2,)
    map_point: np.ndarray  # (3,)
    age: int = 0
    error: float = 0.0
    status: str = "active"


@dataclass
class KLTResult:
    tracks: list[KLTTrack]
    tracked_count: int
    tracking_ratio: float
    spatial_cells: int
    confidence: float
    valid: bool


class KLTTracker:
    def __init__(self, config: KLTConfig | None = None):
        self.config = config or KLTConfig()
        self._prev_gray: np.ndarray | None = None
        self._tracks: list[KLTTrack] = []
        self._seed_stamp: float | None = None
        self._frame_age: int = 0

    def clear(self) -> None:
        self._prev_gray = None
        self._tracks.clear()
        self._seed_stamp = None
        self._frame_age = 0

    def initialize(self, gray: np.ndarray, points2d: np.ndarray, points3d: np.ndarray, inlier_mask: np.ndarray | None, timestamp: float) -> bool:
        """Seed from EDM 2D-3D inliers. Returns True if enough unique points."""
        try:
            if inlier_mask is not None:
                mask = np.asarray(inlier_mask, dtype=bool)
                pts2d = np.asarray(points2d, dtype=float)[mask]
                pts3d = np.asarray(points3d, dtype=float)[mask]
            else:
                pts2d = np.asarray(points2d, dtype=float)
                pts3d = np.asarray(points3d, dtype=float)
            if len(pts2d) == 0:
                self.clear()
                return False
            # unique like XFeat _seed_flow_from_deep
            _, u2 = np.unique(pts2d, axis=0, return_index=True)
            u2 = np.sort(u2)
            _, u3 = np.unique(pts3d[u2], axis=0, return_index=True)
            uniq = u2[np.sort(u3)]
            if len(uniq) < self.config.min_seed:
                self.clear()
                return False
            self._tracks = [
                KLTTrack(pixel=pts2d[i].copy(), map_point=pts3d[i].copy(), age=0)
                for i in uniq
            ]
            self._prev_gray = np.ascontiguousarray(gray)
            self._seed_stamp = float(timestamp)
            self._frame_age = 0
            return True
        except Exception:
            self.clear()
            return False

    def track(self, gray: np.ndarray, timestamp: float) -> KLTResult:
        if self._prev_gray is None or not self._tracks:
            return KLTResult(tracks=[], tracked_count=0, tracking_ratio=0.0, spatial_cells=0, confidence=0.0, valid=False)
        # age check
        if self._seed_stamp is not None and math.isfinite(timestamp) and math.isfinite(self._seed_stamp):
            if timestamp - self._seed_stamp > self.config.max_age_s or self._frame_age >= self.config.max_age_frames:
                self.clear()
                return KLTResult(tracks=[], tracked_count=0, tracking_ratio=0.0, spatial_cells=0, confidence=0.0, valid=False)
        try:
            p0 = np.array([t.pixel for t in self._tracks], dtype=np.float32).reshape(-1, 1, 2)
            nxt, stf, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **_KLT_PARAMS)
            if nxt is None or stf is None:
                self.clear()
                return KLTResult(tracks=[], tracked_count=0, tracking_ratio=0.0, spatial_cells=0, confidence=0.0, valid=False)
            back, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, nxt, None, **_KLT_PARAMS)
            if back is None or stb is None:
                self.clear()
                return KLTResult(tracks=[], tracked_count=0, tracking_ratio=0.0, spatial_cells=0, confidence=0.0, valid=False)
            fb = np.linalg.norm((p0 - back).reshape(-1, 2), axis=1)
            good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < self.config.fb_px)
            if int(np.count_nonzero(good)) < self.config.min_tracks:
                self.clear()
                return KLTResult(tracks=[], tracked_count=0, tracking_ratio=0.0, spatial_cells=0, confidence=0.0, valid=False)
            # median drop for PnP later is handled outside; here keep full good
            new_tracks: list[KLTTrack] = []
            for idx, ok in enumerate(good):
                if not ok:
                    continue
                t = self._tracks[idx]
                new_tracks.append(
                    KLTTrack(
                        pixel=nxt[idx, 0].copy(),
                        map_point=t.map_point.copy(),
                        age=t.age + 1,
                        error=float(fb[idx]),
                        status="active",
                    )
                )
            # quality
            ratio = len(new_tracks) / max(1, len(self._tracks))
            # spatial coverage: grid cells occupied
            h, w = gray.shape[:2]
            gw, gh = self.config.grid_w, self.config.grid_h
            cells = set()
            for tr in new_tracks:
                x, y = tr.pixel
                gx = min(gw - 1, max(0, int(x / w * gw)))
                gy = min(gh - 1, max(0, int(y / h * gh)))
                cells.add((gx, gy))
            spatial = len(cells)
            confidence = ratio * (spatial / (gw * gh))
            valid = (
                len(new_tracks) >= self.config.min_tracks
                and ratio >= self.config.min_track_ratio
                and spatial >= self.config.min_spatial_coverage
                and float(np.median(fb[good])) <= self.config.max_klt_error_px
            )
            # chain
            self._tracks = new_tracks
            self._prev_gray = np.ascontiguousarray(gray)
            self._frame_age += 1
            return KLTResult(
                tracks=new_tracks,
                tracked_count=len(new_tracks),
                tracking_ratio=ratio,
                spatial_cells=spatial,
                confidence=confidence,
                valid=valid,
            )
        except cv2.error:
            self.clear()
            return KLTResult(tracks=[], tracked_count=0, tracking_ratio=0.0, spatial_cells=0, confidence=0.0, valid=False)
        except Exception:
            self.clear()
            return KLTResult(tracks=[], tracked_count=0, tracking_ratio=0.0, spatial_cells=0, confidence=0.0, valid=False)

    def correspondences(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (N,2) pixels and (N,3) map points for current tracks."""
        if not self._tracks:
            return np.zeros((0, 2)), np.zeros((0, 3))
        pts2d = np.stack([t.pixel for t in self._tracks], axis=0)
        pts3d = np.stack([t.map_point for t in self._tracks], axis=0)
        return pts2d, pts3d

    def needs_replenishment(self) -> bool:
        return len(self._tracks) < self.config.min_tracks
