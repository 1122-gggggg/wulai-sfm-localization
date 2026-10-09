"""Spatio-temporal query and frozen region coverage for online workmap expansion.

與被凍結關閉的 TOPUP 不同，這裡的像素由 LK 從該點自己的關鍵幀影像實測並經前後向檢查，
預測位置只當 LK 起點，不直接當觀測。
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from point_quality import spatial_indices

CELL_SIZE_U = 0.08  # ≈0.05 × 凍結圖中位場景深度 1.53 u；114,577 佔用 cell
FROZEN_CELL_MIN_POINTS = 3  # 0.08 u 時 44,157 個 cell 達標
QUERY_TEMPORAL_FRAMES = 100  # 論文 |B|=100
QUERY_MAX_POINTS = 150
QUERY_POOL_FACTOR = 4
QUERY_MAX_SOURCE_KEYFRAMES = 3
QUERY_STRIDE = 2
QUERY_MAX_VIEW_ANGLE_DEG = 20.0
QUERY_MAX_SCALE_RATIO = 1.4
QUERY_MAX_SHIFT_PX = 24.0
QUERY_IMAGE_MARGIN_PX = 8.0
QUERY_OCCLUSION_GRID = (16, 9)
QUERY_CELLS_PER_BUCKET = 2
KEYFRAME_IMAGE_CACHE = 192  # 960×540 灰階約 100 MB
_KEY_OFFSET = 1 << 20  # 每軸 21 bits

_OFFSETS = np.array(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=np.int64,
)


def cell_indices(xyz: np.ndarray, cell_size: float = CELL_SIZE_U) -> np.ndarray:
    """Return integer 3D grid cell indices for Cartesian points."""
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.floor(pts / float(cell_size)).astype(np.int64)


def pack_cells(ijk: np.ndarray) -> np.ndarray:
    """Pack (N, 3) integer cell coordinates into (N,) int64 keys."""
    indices = np.asarray(ijk, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices.reshape(1, 3)
    if indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    i = indices[:, 0]
    j = indices[:, 1]
    k = indices[:, 2]
    return ((i + _KEY_OFFSET) << 42) | ((j + _KEY_OFFSET) << 21) | (k + _KEY_OFFSET)


def _unpack_cells(keys: np.ndarray) -> np.ndarray:
    """Unpack (N,) int64 keys into (N, 3) integer cell coordinates."""
    k_arr = np.asarray(keys, dtype=np.int64)
    if k_arr.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    i = ((k_arr >> 42) & 0x1FFFFF) - _KEY_OFFSET
    j = ((k_arr >> 21) & 0x1FFFFF) - _KEY_OFFSET
    k = (k_arr & 0x1FFFFF) - _KEY_OFFSET
    return np.column_stack((i, j, k))


class FrozenCoverage:
    """Cell occupancy gate determining whether 3D regions are in the frozen site map."""

    def __init__(
        self,
        frozen_xyz: np.ndarray,
        *,
        cell_size: float = CELL_SIZE_U,
        min_points: int = FROZEN_CELL_MIN_POINTS,
    ) -> None:
        self._cell_size = float(cell_size)
        self._min_points = int(min_points)
        pts = np.asarray(frozen_xyz, dtype=np.float64)
        if pts.size == 0:
            self._covered = np.empty((0,), dtype=np.int64)
            return
        finite = np.all(np.isfinite(pts), axis=1)
        valid_pts = pts[finite]
        if valid_pts.size == 0:
            self._covered = np.empty((0,), dtype=np.int64)
            return
        keys = pack_cells(cell_indices(valid_pts, cell_size=self._cell_size))
        unique_keys, counts = np.unique(keys, return_counts=True)
        self._covered = unique_keys[counts >= self._min_points]

    @property
    def cell_count(self) -> int:
        """Number of dense frozen cells retained."""
        return len(self._covered)

    def covered(self, xyz: np.ndarray) -> np.ndarray:
        """Check whether each point is inside or neighboring any frozen covered cell."""
        pts = np.asarray(xyz, dtype=np.float64)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)
        result = np.ones(len(pts), dtype=bool)
        if len(self._covered) == 0:
            finite = np.all(np.isfinite(pts), axis=1)
            result[finite] = False
            return result
        finite = np.all(np.isfinite(pts), axis=1)
        if not np.any(finite):
            return result
        valid_pts = pts[finite]
        ijk = cell_indices(valid_pts, cell_size=self._cell_size)
        neighbor_ijk = ijk[:, None, :] + _OFFSETS[None, :, :]
        packed = pack_cells(neighbor_ijk.reshape(-1, 3))
        idx = np.searchsorted(self._covered, packed)
        idx = np.clip(idx, 0, len(self._covered) - 1)
        is_in = (self._covered[idx] == packed).reshape(-1, 27)
        result[finite] = np.any(is_in, axis=1)
        return result


@dataclass(frozen=True)
class QueryCandidates:
    """Landmark candidates retrieved for guided KLT tracking."""

    ids: np.ndarray
    xyz: np.ndarray
    predicted_xy: np.ndarray
    source_names: tuple[str, ...]
    source_xy: np.ndarray


def _empty_candidates() -> QueryCandidates:
    return QueryCandidates(
        ids=np.empty((0,), dtype=np.int64),
        xyz=np.empty((0, 3), dtype=np.float64),
        predicted_xy=np.empty((0, 2), dtype=np.float64),
        source_names=(),
        source_xy=np.empty((0, 2), dtype=np.float64),
    )


class SpatioTemporalQuery:
    """Spatiotemporal index for active session landmarks and keyframe sources."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        cell_size: float = CELL_SIZE_U,
        image_cache: int = KEYFRAME_IMAGE_CACHE,
    ) -> None:
        self._w = int(width)
        self._h = int(height)
        self._cell_size = float(cell_size)
        self._image_cache = int(image_cache)

        self._xyz: dict[int, np.ndarray] = {}
        self._cell_of: dict[int, int] = {}
        self._cells: dict[int, set[int]] = {}
        self._centers_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._source: dict[int, tuple[str, np.ndarray]] = {}
        self._images: OrderedDict[str, np.ndarray] = OrderedDict()
        self._kf_center: dict[str, np.ndarray] = {}
        self._kf_ids: dict[str, np.ndarray] = {}
        self._temporal: deque[str] = deque(maxlen=QUERY_TEMPORAL_FRAMES)

    def upsert_landmarks(self, ids: Iterable[int], xyz: np.ndarray) -> None:
        """Upsert negative landmark 3D positions and update cell membership."""
        pts = np.asarray(xyz, dtype=np.float64)
        if pts.ndim == 1 and pts.size == 3:
            pts = pts.reshape(1, 3)
        for lid_raw, pt in zip(ids, pts):
            lid = int(lid_raw)
            if lid >= 0 or not np.all(np.isfinite(pt)):
                continue
            self._upsert_one_landmark(lid, pt)

    def _upsert_one_landmark(self, lid: int, pt: np.ndarray) -> None:
        ijk = cell_indices(pt.reshape(1, 3), cell_size=self._cell_size)[0]
        new_cell = int(pack_cells(ijk.reshape(1, 3))[0])
        old_cell = self._cell_of.get(lid)
        if old_cell is not None and old_cell != new_cell:
            cell_set = self._cells.get(old_cell)
            if cell_set is not None:
                cell_set.discard(lid)
                if not cell_set:
                    del self._cells[old_cell]
                    self._centers_cache = None
        if old_cell != new_cell:
            if new_cell not in self._cells:
                self._cells[new_cell] = set()
                self._centers_cache = None
            self._cells[new_cell].add(lid)
            self._cell_of[lid] = new_cell
        self._xyz[lid] = pt.copy()

    def remove_landmarks(self, ids: Iterable[int]) -> None:
        """Forget landmarks so they are never proposed as tracking candidates again."""
        for lid_raw in ids:
            lid = int(lid_raw)
            cell = self._cell_of.pop(lid, None)
            if cell is not None:
                cell_set = self._cells.get(cell)
                if cell_set is not None:
                    cell_set.discard(lid)
                    if not cell_set:
                        del self._cells[cell]
                        self._centers_cache = None
            self._xyz.pop(lid, None)
            self._source.pop(lid, None)

    def add_keyframe(
        self,
        name: str,
        gray: np.ndarray,
        cam_from_world: np.ndarray,
        ids: np.ndarray,
        xy: np.ndarray,
    ) -> None:
        """Register a keyframe reference image, pose and observed negative landmarks."""
        center = -cam_from_world[:3, :3].T @ cam_from_world[:3, 3]
        self._kf_center[name] = center.astype(np.float64)

        id_arr = np.asarray(ids, dtype=np.int64)
        xy_arr = np.asarray(xy, dtype=np.float64)
        neg_mask = id_arr < 0
        neg_ids = id_arr[neg_mask]
        neg_xy = xy_arr[neg_mask]

        self._kf_ids[name] = neg_ids
        for row in range(len(neg_ids)):
            lid = int(neg_ids[row])
            self._source[lid] = (name, neg_xy[row].copy())

        self._images[name] = gray
        self._evict_oldest_keyframes()

    def _evict_oldest_keyframes(self) -> None:
        while len(self._images) > self._image_cache:
            oldest_name, _ = self._images.popitem(last=False)
            self._kf_center.pop(oldest_name, None)
            self._kf_ids.pop(oldest_name, None)

    def note_frame(self, inlier_ids: Iterable[int]) -> None:
        """Record the most frequent valid source keyframe in the temporal buffer."""
        counts: dict[str, int] = {}
        for lid_raw in inlier_ids:
            lid = int(lid_raw)
            if lid >= 0 or lid not in self._source:
                continue
            name, _ = self._source[lid]
            if name in self._images:
                counts[name] = counts.get(name, 0) + 1
        if not counts:
            return
        best_name = max(counts.keys(), key=lambda n: (counts[n], n))
        self._temporal.append(best_name)

    def clear_temporal(self) -> None:
        """Clear temporal keyframe history."""
        self._temporal.clear()

    def image(self, name: str) -> np.ndarray | None:
        """Retrieve keyframe image and move it to the most recent cache slot."""
        img = self._images.get(name)
        if img is not None:
            self._images.move_to_end(name)
        return img

    def _get_cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        if self._centers_cache is not None:
            return self._centers_cache
        if not self._cells:
            self._centers_cache = (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
            )
            return self._centers_cache
        keys = np.array(list(self._cells.keys()), dtype=np.int64)
        ijk = _unpack_cells(keys)
        centers = (ijk.astype(np.float64) + 0.5) * self._cell_size
        self._centers_cache = (keys, centers)
        return self._centers_cache

    def candidates(
        self,
        cam_from_world: np.ndarray,
        K: np.ndarray,
        exclude_ids: np.ndarray,
        occupied_xy: np.ndarray,
        min_distance_px: int,
        max_points: int = QUERY_MAX_POINTS,
    ) -> QueryCandidates:
        """Find spatio-temporal landmark candidates visible from the current camera pose."""
        limit = QUERY_POOL_FACTOR * int(max_points)
        pool_ids = self._build_pool(cam_from_world, K, limit)
        cand_ids = np.array(pool_ids, dtype=np.int64)

        ex_ids = np.asarray(exclude_ids, dtype=np.int64)
        cand_ids, xyz = self._filter_pool_metadata(cand_ids, ex_ids)
        cand_ids, xyz, pred_xy = self._project_and_filter_bounds(cand_ids, xyz, cam_from_world, K)
        cand_ids, xyz, pred_xy = self._filter_viewpoint(cand_ids, xyz, pred_xy, cam_from_world)
        cand_ids, xyz, pred_xy = self._filter_occupied(
            cand_ids, xyz, pred_xy, occupied_xy, min_distance_px
        )

        if cand_ids.size == 0:
            return _empty_candidates()

        if len(cand_ids) > max_points:
            sel = spatial_indices(pred_xy, int(max_points), self._w, self._h)
            cand_ids = cand_ids[sel]
            xyz = xyz[sel]
            pred_xy = pred_xy[sel]

        source_names = tuple(self._source[int(cid)][0] for cid in cand_ids)
        source_xy = np.array([self._source[int(cid)][1] for cid in cand_ids], dtype=np.float64)
        return QueryCandidates(
            ids=cand_ids.astype(np.int64),
            xyz=xyz.astype(np.float64),
            predicted_xy=pred_xy.astype(np.float64),
            source_names=source_names,
            source_xy=source_xy.reshape(-1, 2),
        )

    def _build_pool(
        self,
        cam_from_world: np.ndarray,
        K: np.ndarray,
        limit: int,
    ) -> list[int]:
        pool: dict[int, None] = {}
        self._collect_temporal_pool(pool, limit)
        if len(pool) < limit:
            self._collect_spatial_pool(pool, cam_from_world, K, limit)
        return list(pool.keys())

    def _collect_temporal_pool(self, pool: dict[int, None], limit: int) -> None:
        seen_kf: set[str] = set()
        for name in reversed(self._temporal):
            if name in seen_kf or name not in self._images or name not in self._kf_ids:
                continue
            seen_kf.add(name)
            for lid in self._kf_ids[name]:
                pool[int(lid)] = None
                if len(pool) >= limit:
                    return

    def _collect_spatial_pool(
        self,
        pool: dict[int, None],
        cam_from_world: np.ndarray,
        K: np.ndarray,
        limit: int,
    ) -> None:
        keys, centers = self._get_cell_centers()
        if len(keys) == 0:
            return
        R = cam_from_world[:3, :3]
        t = cam_from_world[:3, 3]
        cam_pts = centers @ R.T + t
        z = cam_pts[:, 2]
        valid_z = z > 1e-3
        if not np.any(valid_z):
            return
        u = (cam_pts[:, 0] * K[0, 0] / np.where(valid_z, z, 1.0)) + K[0, 2]
        v = (cam_pts[:, 1] * K[1, 1] / np.where(valid_z, z, 1.0)) + K[1, 2]
        in_img = valid_z & (u >= 0) & (u < self._w) & (v >= 0) & (v < self._h)
        if not np.any(in_img):
            return
        kept_keys, kept_z = self._bucket_visible_cells(
            keys[in_img], u[in_img], v[in_img], z[in_img]
        )
        order = np.argsort(kept_z)
        for cell_key in kept_keys[order]:
            cell_ids = self._cells.get(int(cell_key))
            if not cell_ids:
                continue
            for lid in sorted(cell_ids):
                pool[int(lid)] = None
                if len(pool) >= limit:
                    return

    def _bucket_visible_cells(
        self,
        keys: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = QUERY_OCCLUSION_GRID
        bx = np.clip((u * (cols / self._w)).astype(np.int64), 0, cols - 1)
        by = np.clip((v * (rows / self._h)).astype(np.int64), 0, rows - 1)
        bucket_idx = by * cols + bx
        selected: list[int] = []
        for b_id in np.unique(bucket_idx):
            in_b = np.flatnonzero(bucket_idx == b_id)
            if len(in_b) <= QUERY_CELLS_PER_BUCKET:
                selected.extend(in_b)
            else:
                top = in_b[np.argsort(z[in_b])[:QUERY_CELLS_PER_BUCKET]]
                selected.extend(top)
        sel_arr = np.array(selected, dtype=np.int64)
        return keys[sel_arr], z[sel_arr]

    def _filter_pool_metadata(
        self,
        cand_ids: np.ndarray,
        exclude_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cand_ids.size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
        arr = cand_ids
        if exclude_ids.size:
            arr = arr[~np.isin(arr, exclude_ids)]
        valid_ids: list[int] = []
        valid_xyz: list[np.ndarray] = []
        for cid in arr:
            lid = int(cid)
            if lid not in self._source or lid not in self._xyz:
                continue
            src_name, _ = self._source[lid]
            if src_name not in self._images:
                continue
            valid_ids.append(lid)
            valid_xyz.append(self._xyz[lid])
        if not valid_ids:
            return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
        return np.array(valid_ids, dtype=np.int64), np.array(valid_xyz, dtype=np.float64)

    def _project_and_filter_bounds(
        self,
        cand_ids: np.ndarray,
        xyz: np.ndarray,
        cam_from_world: np.ndarray,
        K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cand_ids.size == 0:
            return cand_ids, xyz, np.empty((0, 2), dtype=np.float64)
        R = cam_from_world[:3, :3]
        t = cam_from_world[:3, 3]
        pts_cam = xyz @ R.T + t
        z = pts_cam[:, 2]
        valid_z = z > 1e-3
        u = (pts_cam[:, 0] * K[0, 0] / np.where(valid_z, z, 1.0)) + K[0, 2]
        v = (pts_cam[:, 1] * K[1, 1] / np.where(valid_z, z, 1.0)) + K[1, 2]
        margin = float(QUERY_IMAGE_MARGIN_PX)
        in_bounds = (
            valid_z
            & (u >= margin)
            & (u <= (self._w - 1.0 - margin))
            & (v >= margin)
            & (v <= (self._h - 1.0 - margin))
        )
        if not np.any(in_bounds):
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )
        pred_xy = np.column_stack((u[in_bounds], v[in_bounds]))
        return cand_ids[in_bounds], xyz[in_bounds], pred_xy

    def _filter_viewpoint(
        self,
        cand_ids: np.ndarray,
        xyz: np.ndarray,
        pred_xy: np.ndarray,
        cam_from_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cand_ids.size == 0:
            return cand_ids, xyz, pred_xy
        c_cur = -cam_from_world[:3, :3].T @ cam_from_world[:3, 3]
        ray_cur = xyz - c_cur
        d_cur = np.linalg.norm(ray_cur, axis=1)

        c_src = np.array([self._kf_center[self._source[int(cid)][0]] for cid in cand_ids])
        ray_src = xyz - c_src
        d_src = np.linalg.norm(ray_src, axis=1)

        dot = np.sum(ray_cur * ray_src, axis=1)
        denom = d_cur * d_src + 1e-12
        cos_ang = np.clip(dot / denom, -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cos_ang))

        ratio = d_cur / (d_src + 1e-12)
        valid = (
            (ang_deg <= QUERY_MAX_VIEW_ANGLE_DEG)
            & (ratio >= (1.0 / QUERY_MAX_SCALE_RATIO))
            & (ratio <= QUERY_MAX_SCALE_RATIO)
        )
        if not np.any(valid):
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )
        return cand_ids[valid], xyz[valid], pred_xy[valid]

    def _filter_occupied(
        self,
        cand_ids: np.ndarray,
        xyz: np.ndarray,
        pred_xy: np.ndarray,
        occupied_xy: np.ndarray,
        min_distance_px: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cand_ids.size == 0:
            return cand_ids, xyz, pred_xy
        mask = np.full((self._h, self._w), 255, dtype=np.uint8)
        occ = np.asarray(occupied_xy, dtype=np.float64)
        if occ.size:
            pts = np.asarray(occ, dtype=int)
            valid = (
                (pts[:, 0] >= 0) & (pts[:, 0] < self._w) & (pts[:, 1] >= 0) & (pts[:, 1] < self._h)
            )
            pts = pts[valid]
            if len(pts):
                mask[pts[:, 1], pts[:, 0]] = 0
                ksize = max(1, 2 * int(min_distance_px) + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                mask = cv2.erode(mask, kernel)

        rounded = np.rint(pred_xy).astype(np.int64)
        valid_r = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < self._w)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < self._h)
        )
        keep = np.zeros(len(pred_xy), dtype=bool)
        keep[valid_r] = mask[rounded[valid_r, 1], rounded[valid_r, 0]] > 0
        if not np.any(keep):
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )
        return cand_ids[keep], xyz[keep], pred_xy[keep]
