"""Bounded CPU sliding-window pose experiment.

This is an offline research prototype.  It consumes only the inlier fields in
one frame record and never reads a holdout split.  A window couples the camera
poses through shared track IDs and gives every local point one correction
variable with a prior around the supplied map/triangulated point.

The first pose in each window is fixed.  This is a convenient gauge anchor, not
formal marginalisation: when the deque slides, the oldest retained pose becomes
the new fixed pose.  There is deliberately no NED velocity factor yet because
the prototype API does not accept paired velocity samples.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


_MAX_GAP_S = 0.5
_PIXEL_NOISE_PX = 1.0


@dataclass(frozen=True)
class _Frame:
    stamp: float
    epoch: int
    pose: np.ndarray
    ids: np.ndarray
    xy: np.ndarray
    xyz: np.ndarray
    K: np.ndarray


def _rotvec_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues map for the small pose corrections used by the experiment."""
    value = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(value))
    if theta < 1e-12:
        # First-order form is stable at the origin and sufficient for the
        # finite-difference Jacobian used by scipy.
        x, y, z = value
        return np.array([[1.0, -z, y], [z, 1.0, -x], [-y, x, 1.0]])
    axis = value / theta
    x, y, z = axis
    c = math.cos(theta)
    s = math.sin(theta)
    one = 1.0 - c
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=float,
    )


def _reprojection_residual_loop(
    rotations: np.ndarray,
    translations: np.ndarray,
    points: np.ndarray,
    frame_indices: np.ndarray,
    point_indices: np.ndarray,
    uv: np.ndarray,
    cameras: np.ndarray,
) -> np.ndarray:
    """Reference scalar residual used to verify the vectorized implementation."""
    residual: list[float] = []
    for frame_index, point_index, observed in zip(frame_indices, point_indices, uv):
        camera = rotations[frame_index] @ points[point_index] + translations[frame_index]
        if camera[2] <= 1e-6 or not np.isfinite(camera).all():
            residual.extend([100.0, 100.0])
            continue
        K = cameras[frame_index]
        predicted = np.array(
            [
                K[0, 0] * camera[0] / camera[2] + K[0, 2],
                K[1, 1] * camera[1] / camera[2] + K[1, 2],
            ],
            dtype=float,
        )
        residual.extend((predicted - observed).tolist())
    return np.asarray(residual, dtype=float)


def _reprojection_residual_vectorized(
    rotations: np.ndarray,
    translations: np.ndarray,
    points: np.ndarray,
    frame_indices: np.ndarray,
    point_indices: np.ndarray,
    uv: np.ndarray,
    cameras: np.ndarray,
) -> np.ndarray:
    """Batch reprojection residual with the same invalid-depth convention."""
    selected_rotations = rotations[frame_indices]
    selected_translations = translations[frame_indices]
    camera = np.einsum("nij,nj->ni", selected_rotations, points[point_indices])
    camera = camera + selected_translations
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    predicted = np.zeros_like(uv, dtype=float)
    K = cameras[frame_indices]
    predicted[:, 0] = K[:, 0, 0] * camera[:, 0] / np.where(valid, camera[:, 2], 1.0) + K[:, 0, 2]
    predicted[:, 1] = K[:, 1, 1] * camera[:, 1] / np.where(valid, camera[:, 2], 1.0) + K[:, 1, 2]
    residual = predicted - uv
    residual[~valid] = 100.0
    return residual.reshape(-1)


def _as_pose(value: Any) -> np.ndarray | None:
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if pose.shape != (3, 4) or not np.isfinite(pose).all():
        return None
    rotation = pose[:, :3]
    if not np.isfinite(np.linalg.det(rotation)):
        return None
    return np.ascontiguousarray(pose, dtype=float)


def _finite_stamp(value: Any) -> float | None:
    try:
        stamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # Offline video timestamps may legitimately start at 0.0.  Later updates
    # still enforce strict monotonicity through the gap check in ``update``.
    return stamp if math.isfinite(stamp) and stamp >= 0.0 else None


def _unique_finite_observations(ids, xy, xyz):
    # Keep the first finite observation for a repeated ID in one frame.  A
    # repeated row is not an independent sensor and would otherwise overweight
    # that track in the optimizer.
    keep: list[int] = []
    seen: set[int] = set()
    for index, point_id in enumerate(ids.tolist()):
        if int(point_id) in seen:
            continue
        if not np.isfinite(xy[index]).all() or not np.isfinite(xyz[index]).all():
            continue
        seen.add(int(point_id))
        keep.append(index)
    if keep:
        ids = ids[keep]
        xy = xy[keep]
        xyz = xyz[keep]
    else:
        ids = np.zeros((0,), dtype=np.int64)
        xy = np.zeros((0, 2), dtype=float)
        xyz = np.zeros((0, 3), dtype=float)
    return ids, xy, xyz


def _frame_from_input(frame: dict[str, Any]) -> _Frame | None:
    """Copy only the six documented input fields; holdout keys are ignored."""
    stamp = _finite_stamp(frame.get("stamp"))
    epoch = frame.get("epoch")
    if stamp is None or isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)):
        return None
    pose = _as_pose(frame.get("pose"))
    if pose is None:
        return None
    try:
        ids = np.asarray(frame.get("ids"), dtype=np.int64)
        xy = np.asarray(frame.get("xy"), dtype=float)
        xyz = np.asarray(frame.get("xyz"), dtype=float)
        K = np.asarray(frame.get("K"), dtype=float)
    except (TypeError, ValueError):
        return None
    if ids.ndim != 1 or xy.ndim != 2 or xyz.ndim != 2 or K.shape != (3, 3):
        return None
    if xy.shape != (ids.size, 2) or xyz.shape != (ids.size, 3):
        return None
    if not np.isfinite(K).all() or float(K[0, 0]) <= 0.0 or float(K[1, 1]) <= 0.0:
        return None

    ids, xy, xyz = _unique_finite_observations(ids, xy, xyz)
    return _Frame(
        stamp=stamp,
        epoch=int(epoch),
        pose=pose,
        ids=np.ascontiguousarray(ids),
        xy=np.ascontiguousarray(xy),
        xyz=np.ascontiguousarray(xyz),
        K=np.ascontiguousarray(K),
    )


class PoseWindowOptimizer:
    """Bounded visual-only pose/point optimizer for a 5 or 10 frame window.

    ``success`` means a finite bounded solver result was produced.  It does not
    claim statistical convergence; ``nfev`` and ``cost`` are returned so the
    caller can apply its own acceptance rule.  The implementation uses a sparse
    finite-difference Jacobian pattern, a soft-L1 pixel loss, and no velocity
    or holdout inputs.  Per-frame pose priors are derived from the incoming PnP
    poses; they are regularization, not additional independent measurements.
    """

    def __init__(self, window_size: int = 5, max_nfev: int = 5, max_points: int = 80):
        if isinstance(window_size, bool) or int(window_size) < 2:
            raise ValueError("window_size must be >= 2")
        if isinstance(max_nfev, bool) or int(max_nfev) <= 0:
            raise ValueError("max_nfev must be positive")
        if isinstance(max_points, bool) or int(max_points) <= 0:
            raise ValueError("max_points must be positive")
        self.window_size = int(window_size)
        self.max_nfev = int(max_nfev)
        self.max_points = int(max_points)
        self._frames: deque[_Frame] = deque(maxlen=self.window_size)
        self._epoch: int | None = None
        self._last_stamp: float | None = None

    def _clear(self) -> None:
        self._frames.clear()
        self._epoch = None
        self._last_stamp = None

    @staticmethod
    def _empty_result(
        *,
        reset: bool,
        pose: np.ndarray | None,
        observation_count: int = 0,
        window_frames: int = 0,
    ) -> dict:
        return {
            "pose": None if pose is None else np.asarray(pose, dtype=float).copy(),
            "success": False,
            "reset": bool(reset),
            "nfev": 0,
            "cost": None,
            "observation_count": int(observation_count),
            "optimized": False,
            "window_frames": int(window_frames),
        }

    def update(self, frame: dict[str, Any]) -> dict:
        """Append one current train-inlier frame and solve using past frames only."""
        if not isinstance(frame, dict):
            self._clear()
            return self._empty_result(reset=True, pose=None)
        current = _frame_from_input(frame)
        if current is None:
            self._clear()
            return self._empty_result(reset=True, pose=None)

        did_reset = False
        if self._epoch is not None and current.epoch != self._epoch:
            self._clear()
            did_reset = True
        elif self._last_stamp is not None:
            gap = current.stamp - self._last_stamp
            if gap <= 0.0 or gap > _MAX_GAP_S:
                self._clear()
                did_reset = True
        self._epoch = current.epoch
        self._last_stamp = current.stamp
        self._frames.append(current)

        if len(self._frames) < 2:
            return self._empty_result(
                reset=did_reset,
                pose=current.pose,
                window_frames=len(self._frames),
            )

        result = self._solve()
        result["reset"] = did_reset
        if result["pose"] is None:
            result["pose"] = current.pose.copy()
        return result

    def _collect_tracks(
        self,
    ) -> tuple[list[_Frame], list[int], dict[int, list[tuple[int, np.ndarray, np.ndarray]]]]:
        frames = list(self._frames)
        tracks: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for frame_index, item in enumerate(frames):
            for point_id, uv, xyz in zip(item.ids.tolist(), item.xy, item.xyz):
                tracks.setdefault(int(point_id), []).append(
                    (frame_index, np.asarray(uv, dtype=float), np.asarray(xyz, dtype=float))
                )
        # Shared tracks are most useful for cross-frame coupling.  Positive map
        # anchors win ties, then deterministic ID order keeps CPU work bounded.
        selected_ids = sorted(
            tracks,
            key=lambda point_id: (
                -min(len(tracks[point_id]), 2),
                -len(tracks[point_id]),
                0 if point_id > 0 else 1,
                point_id,
            ),
        )[: self.max_points]
        return frames, selected_ids, {point_id: tracks[point_id] for point_id in selected_ids}

    @staticmethod
    def _projection_scale(
        frames: list[_Frame], tracks: dict[int, list[tuple[int, np.ndarray, np.ndarray]]]
    ) -> tuple[float, float]:
        focals = np.asarray([(item.K[0, 0] + item.K[1, 1]) * 0.5 for item in frames], dtype=float)
        frame_indices: list[int] = []
        points: list[np.ndarray] = []
        for observations in tracks.values():
            for frame_index, _uv, xyz in observations:
                frame_indices.append(frame_index)
                points.append(xyz)
        depths: np.ndarray
        if points:
            point_array = np.asarray(points, dtype=float)
            rotations = np.asarray([frames[index].pose[:, :3] for index in frame_indices])
            translations = np.asarray([frames[index].pose[:, 3] for index in frame_indices])
            depths = np.einsum("nij,nj->ni", rotations, point_array)[:, 2] + translations[:, 2]
            depths = depths[np.isfinite(depths) & (depths > 1e-3)]
        else:
            depths = np.zeros((0,), dtype=float)
        depth = float(np.median(depths)) if depths.size else 1.0
        focal = float(np.median(focals)) if focals.size else 1.0
        return max(depth, 1e-3), max(focal, 1.0)

    @staticmethod
    def _make_sparsity(
        frame_count: int,
        point_ids: list[int],
        observations: list[tuple[int, int]],
    ):
        pose_vars = 6 * max(0, frame_count - 1)
        point_offset = pose_vars
        variable_count = pose_vars + 3 * len(point_ids)
        rows = 2 * len(observations) + 6 * max(0, frame_count - 1) + 3 * len(point_ids)
        pattern = lil_matrix((rows, variable_count), dtype=np.int8)
        row = 0
        point_index = {point_id: index for index, point_id in enumerate(point_ids)}
        for frame_index, point_id in observations:
            if frame_index > 0:
                start = 6 * (frame_index - 1)
                pattern[row : row + 2, start : start + 6] = 1
            pstart = point_offset + 3 * point_index[point_id]
            pattern[row : row + 2, pstart : pstart + 3] = 1
            row += 2
        for frame_index in range(1, frame_count):
            start = 6 * (frame_index - 1)
            pattern[row : row + 6, start : start + 6] = 1
            row += 6
        for point_index_value in range(len(point_ids)):
            start = point_offset + 3 * point_index_value
            pattern[row : row + 3, start : start + 3] = 1
            row += 3
        return pattern.tocsr()

    def _solve(self) -> dict:
        frames, point_ids, track_map = self._collect_tracks()
        observations: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for point_id in point_ids:
            for frame_index, uv, xyz in track_map[point_id]:
                observations.append((frame_index, point_id, uv, xyz))
        observation_count = len(observations)
        current_pose = frames[-1].pose.copy()
        if observation_count < 4 or len(point_ids) == 0:
            return self._empty_result(
                reset=False,
                pose=current_pose,
                observation_count=observation_count,
                window_frames=len(frames),
            )

        depth, focal = self._projection_scale(frames, track_map)
        metric_per_pixel = max(depth / focal * _PIXEL_NOISE_PX, 1e-5)
        map_point_sigma = metric_per_pixel * 1.5
        vo_point_sigma = map_point_sigma * 6.0
        rotation_sigma = max(2.0 * _PIXEL_NOISE_PX / focal, 1e-4)
        translation_sigma = metric_per_pixel * 4.0

        bases: dict[int, np.ndarray] = {}
        for point_id in point_ids:
            values = np.stack([xyz for _frame_index, _uv, xyz in track_map[point_id]], axis=0)
            bases[point_id] = np.median(values, axis=0)

        pose_count = len(frames)
        point_offset = 6 * (pose_count - 1)
        point_index = {point_id: index for index, point_id in enumerate(point_ids)}
        obs_frame_indices = np.asarray([item[0] for item in observations], dtype=np.int64)
        obs_point_indices = np.asarray(
            [point_index[item[1]] for item in observations], dtype=np.int64
        )
        obs_uv = np.asarray([item[2] for item in observations], dtype=float)
        obs_cameras = np.asarray([frames[index].K for index in obs_frame_indices], dtype=float)
        base_points = np.asarray([bases[point_id] for point_id in point_ids], dtype=float)
        point_sigmas = np.where(
            np.asarray(point_ids, dtype=np.int64)[:, None] > 0,
            map_point_sigma,
            vo_point_sigma,
        )
        base_rotations = np.asarray([item.pose[:, :3] for item in frames], dtype=float)
        base_translations = np.asarray([item.pose[:, 3] for item in frames], dtype=float)
        x0 = np.zeros(point_offset + 3 * len(point_ids), dtype=float)
        sparsity = self._make_sparsity(
            pose_count,
            point_ids,
            [(frame_index, point_id) for frame_index, point_id, _uv, _xyz in observations],
        )

        def unpack_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            pose_corrections = values[:point_offset].reshape(pose_count - 1, 6)
            correction_rotations = np.asarray(
                [_rotvec_matrix(row[:3]) for row in pose_corrections], dtype=float
            )
            corrected_rotations = np.concatenate(
                [
                    base_rotations[0:1],
                    np.einsum(
                        "nij,njk->nik",
                        correction_rotations,
                        base_rotations[1:],
                    ),
                ],
                axis=0,
            )
            corrected_translations = base_translations.copy()
            corrected_translations[1:] += pose_corrections[:, 3:6]
            point_deltas = values[point_offset:].reshape(len(point_ids), 3)
            corrected_points = base_points + point_deltas
            return corrected_rotations, corrected_translations, corrected_points

        def residuals(values: np.ndarray) -> np.ndarray:
            # One pixel reprojection factor per observed (ID, frame) pair is
            # the visual measurement.  There is no second E_flow factor for
            # the same pair, so a frame is never counted as two independent
            # sensors merely because KLT supplied its association.
            rotations, translations, points = unpack_arrays(values)
            reprojection = _reprojection_residual_vectorized(
                rotations,
                translations,
                points,
                obs_frame_indices,
                obs_point_indices,
                obs_uv,
                obs_cameras,
            )
            pose_corrections = values[:point_offset].reshape(pose_count - 1, 6)
            pose_prior = pose_corrections.copy()
            pose_prior[:, :3] /= rotation_sigma
            pose_prior[:, 3:6] /= translation_sigma
            pose_prior = pose_prior.reshape(-1)
            point_prior = (values[point_offset:].reshape(len(point_ids), 3) / point_sigmas).reshape(
                -1
            )
            return np.concatenate([reprojection, pose_prior, point_prior])

        try:
            solution = least_squares(
                residuals,
                x0,
                jac="2-point",
                jac_sparsity=sparsity,
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=self.max_nfev,
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return self._empty_result(
                reset=False,
                pose=current_pose,
                observation_count=observation_count,
                window_frames=len(frames),
            )

        rotations, translations, _points = unpack_arrays(solution.x)
        pose = np.column_stack((rotations[-1], translations[-1]))
        finite_result = np.isfinite(pose).all() and math.isfinite(float(solution.cost))
        return {
            "pose": pose if finite_result else current_pose,
            "success": bool(finite_result),
            "reset": False,
            "nfev": int(getattr(solution, "nfev", 0)),
            "cost": float(solution.cost) if finite_result else None,
            "observation_count": observation_count,
            "optimized": bool(finite_result),
            "window_frames": len(frames),
        }


__all__ = ["PoseWindowOptimizer"]
