from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from workmap_stitch import _OLD_WEIGHT

#: Most recent keyframes of one submap in a window; one older boundary keyframe may join.
BA_WINDOW_KEYFRAMES = 10
_MIN_WINDOW_KEYFRAMES = 4
#: Negative points one local BA may refine.
MAX_VARIABLE_POINTS = 160
#: Keyframes outside the window that see variable points; their poses stay fixed.
BA_MAX_OBSERVER_KEYFRAMES = 20
_MAX_OLD_OBSERVATIONS_PER_KEYFRAME = 800
#: Equals the tracker's live-point cap (``fast_loop.track_cap`` = 500), so no keyframe can exceed it.
_MAX_NEW_OBSERVATIONS_PER_KEYFRAME = 500
_MIN_OBSERVATIONS = 30
_HUBER_SCALE_PX = 2.0
_MAX_NFEV = 6
_MIN_DEPTH = 1e-6


def _integer(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _landmark_id(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value, 10)
        except ValueError:
            return None
    return None


def _numeric_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            return None
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if shape is not None and result.shape != shape:
        return None
    if not np.isfinite(result).all():
        return None
    return result


def _id_array(value: Any) -> np.ndarray | None:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        if raw.ndim == 1 and raw.size == 0:
            return np.empty((0,), dtype=np.int64)
        return None
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        return None
    try:
        return np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None


def _id_map(value: Any, *, positive: bool) -> dict[int, np.ndarray] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[int, np.ndarray] = {}
    for raw_id, raw_xyz in value.items():
        landmark_id = _landmark_id(raw_id)
        if landmark_id is None or (landmark_id > 0) != positive or landmark_id == 0:
            return None
        if landmark_id in result:
            return None
        xyz = _numeric_array(raw_xyz, (3,))
        if xyz is None:
            return None
        result[landmark_id] = xyz.copy()
    return result


def _revision_map(value: Any) -> dict[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[int, int] = {}
    for raw_id, raw_revision in value.items():
        landmark_id = _landmark_id(raw_id)
        revision = _integer(raw_revision)
        if landmark_id is None or landmark_id >= 0 or revision is None or revision < 0:
            return None
        if landmark_id in result:
            return None
        result[landmark_id] = revision
    return result


def _failure(
    *,
    session_id: Any = None,
    source_epoch: int | None = None,
    snapshot_revision: int | None = None,
    keyframe_revisions: Mapping[str, int] | None = None,
    landmark_revisions: Mapping[int, int] | None = None,
    reason: str,
    before_cost: float | None = None,
    after_cost: float | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "session_id": session_id,
        "source_epoch": source_epoch,
        "snapshot_revision": snapshot_revision,
        "keyframe_revisions": dict(keyframe_revisions or {}),
        "landmark_revisions": dict(landmark_revisions or {}),
        "poses": {},
        "updates": {},
        "before_cost": before_cost,
        "after_cost": after_cost,
        "reason": reason,
    }


def _huber_cost(residuals: np.ndarray) -> float:
    scale = _HUBER_SCALE_PX
    magnitude = np.abs(residuals)
    squared = residuals * residuals
    return float(
        0.5 * np.sum(np.where(magnitude <= scale, squared, 2.0 * scale * magnitude - scale * scale))
    )


class _BaFailure(Exception):
    """One named reason the snapshot cannot yield a BA candidate."""

    def __init__(
        self, reason: str, *, before_cost: float | None = None, after_cost: float | None = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.before_cost = before_cost
        self.after_cost = after_cost


def solve_ba_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Solve one immutable four-keyframe local BA snapshot with SciPy."""
    if not isinstance(payload, Mapping):
        return _failure(reason="payload must be a mapping")
    # Failures report whatever identity/revision context was parsed so far.
    context: dict[str, Any] = {}
    try:
        return _solve_ba(payload, context)
    except _BaFailure as failure:
        return _failure(
            **context,
            reason=failure.reason,
            before_cost=failure.before_cost,
            after_cost=failure.after_cost,
        )


def _solve_ba(payload: Mapping[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    _parse_header(payload, context)
    least_squares, Rotation = _scipy()
    frames = _parse_keyframes(payload, context)
    K, old_landmarks, new_landmarks = _parse_landmarks(payload, context)
    negative_observations, old_support = _classify_observations(
        frames, old_landmarks, new_landmarks
    )
    variable_ids = sorted(
        landmark_id
        for landmark_id, items in negative_observations.items()
        if len({frame_index for frame_index, _row in items}) >= 2
    )[:MAX_VARIABLE_POINTS]
    if not variable_ids:
        raise _BaFailure("no negative landmark has observations in two keyframes")
    observers = _fixed_observers(payload, frames, variable_ids, new_landmarks)
    problem = _BaProblem(
        frames + observers,
        K,
        old_landmarks,
        new_landmarks,
        variable_ids,
        old_support,
        Rotation,
        window_count=len(frames),
    )
    freeze_poses = payload.get("freeze_poses", False)
    if not isinstance(freeze_poses, (bool, np.bool_)):
        raise _BaFailure("freeze_poses must be boolean")
    problem.choose_variable_poses(bool(freeze_poses))
    before_cost = problem.initial_cost()
    x0, bounds = problem.start_and_bounds()
    try:
        solution = least_squares(
            problem.residuals,
            x0,
            bounds=bounds,
            method="trf",
            tr_solver="lsmr",
            jac_sparsity=problem.jacobian_sparsity(),
            x_scale="jac",
            loss="huber",
            f_scale=_HUBER_SCALE_PX,
            max_nfev=_MAX_NFEV,
            ftol=1e-3,
            xtol=1e-3,
            gtol=1e-3,
        )
    except Exception as exc:
        raise _BaFailure(
            f"SciPy BA failed: {type(exc).__name__}", before_cost=before_cost
        ) from None
    poses, points, after_cost = problem.final(solution.x, before_cost)
    return {
        "ok": True,
        "session_id": context["session_id"],
        "source_epoch": context["source_epoch"],
        "snapshot_revision": context["snapshot_revision"],
        "keyframe_revisions": dict(context["keyframe_revisions"]),
        "landmark_revisions": dict(context["landmark_revisions"]),
        "poses": {frame["name"]: poses[index].tolist() for index, frame in enumerate(frames)},
        "updates": {
            landmark_id: points[index].tolist() for index, landmark_id in enumerate(variable_ids)
        },
        "before_cost": before_cost,
        "after_cost": after_cost,
        "reason": None,
    }


def _parse_header(payload: Mapping[str, Any], context: dict[str, Any]) -> None:
    session_id = payload.get("session_id")
    source_epoch = _integer(payload.get("source_epoch"))
    snapshot_revision = _integer(payload.get("snapshot_revision"))
    if not isinstance(session_id, str) or not session_id:
        raise _BaFailure("session_id must be a non-empty string")
    context["session_id"] = session_id
    if source_epoch is None or source_epoch < 0:
        raise _BaFailure("source_epoch must be a non-negative integer")
    context["source_epoch"] = source_epoch
    if snapshot_revision is None or snapshot_revision < 0:
        raise _BaFailure("snapshot_revision must be a non-negative integer")
    context["snapshot_revision"] = snapshot_revision


def _scipy() -> tuple[Any, Any]:
    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ImportError:
        raise _BaFailure("SciPy is required for local BA") from None
    return least_squares, Rotation


def _parse_keyframes(payload: Mapping[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    keyframes_raw = payload.get("keyframes")
    if (
        not isinstance(keyframes_raw, (list, tuple))
        or not _MIN_WINDOW_KEYFRAMES <= len(keyframes_raw) <= BA_WINDOW_KEYFRAMES + 1
    ):
        raise _BaFailure(
            f"between {_MIN_WINDOW_KEYFRAMES} and {BA_WINDOW_KEYFRAMES + 1} keyframes are required"
        )
    keyframe_revisions: dict[str, int] = {}
    context["keyframe_revisions"] = keyframe_revisions
    frames: list[dict[str, Any]] = []
    ordinals: set[int] = set()
    for raw_frame in keyframes_raw:
        frame = _parse_keyframe(raw_frame, keyframe_revisions, ordinals)
        ordinals.add(frame["ordinal"])
        keyframe_revisions[frame["name"]] = frame["pose_revision"]
        frames.append(frame)
    return frames


def _parse_keyframe(
    raw_frame: Any, keyframe_revisions: Mapping[str, int], ordinals: set[int]
) -> dict[str, Any]:
    if not isinstance(raw_frame, Mapping):
        raise _BaFailure("each keyframe must be a mapping")
    name = raw_frame.get("name")
    ordinal = _integer(raw_frame.get("ordinal"))
    pose_revision = _integer(raw_frame.get("pose_revision"))
    if not isinstance(name, str) or not name or name in keyframe_revisions:
        raise _BaFailure("keyframe names must be unique non-empty strings")
    if ordinal is None or ordinal in ordinals or pose_revision is None or pose_revision < 0:
        raise _BaFailure("keyframe ordinal or pose_revision is invalid")
    pose = _numeric_array(raw_frame.get("pose"), (3, 4))
    ids = _id_array(raw_frame.get("ids"))
    xy = _numeric_array(raw_frame.get("xy"))
    if pose is None or ids is None or xy is None or xy.ndim != 2 or xy.shape != (len(ids), 2):
        raise _BaFailure(f"keyframe {name} has invalid pose, ids, or observations")
    rotation = pose[:, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-3, rtol=1e-3
    ):
        raise _BaFailure(f"keyframe {name} pose rotation is not in SO(3)")
    if np.any(ids == 0) or len(np.unique(ids)) != len(ids):
        raise _BaFailure(f"keyframe {name} ids must be unique and nonzero")
    fixed = raw_frame.get("fixed")
    if not isinstance(fixed, (bool, np.bool_)):
        raise _BaFailure(f"keyframe {name} fixed flag must be boolean")
    return {
        "name": name,
        "ordinal": ordinal,
        "pose_revision": pose_revision,
        "pose": pose.copy(),
        "ids": ids,
        "xy": xy.copy(),
        "fixed": bool(fixed),
    }


def _fixed_observers(
    payload: Mapping[str, Any],
    window: list[dict[str, Any]],
    variable_ids: list[int],
    new_landmarks: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    """Keyframes outside the window that keep their pose but add residuals of the variable points.

    Only rows of variable points enter the problem, and rows that would sit behind
    the observer's camera are dropped so one stale row cannot fail the whole solve.
    """
    raw = payload.get("observers", ())
    if not isinstance(raw, (list, tuple)) or len(raw) > BA_MAX_OBSERVER_KEYFRAMES:
        raise _BaFailure("invalid observer keyframe")
    revisions = {frame["name"]: frame["pose_revision"] for frame in window}
    ordinals = {frame["ordinal"] for frame in window}
    variable = np.asarray(variable_ids, dtype=np.int64)
    observers: list[dict[str, Any]] = []
    for raw_frame in raw:
        try:
            frame = _parse_keyframe(raw_frame, revisions, ordinals)
        except _BaFailure:
            raise _BaFailure("invalid observer keyframe") from None
        if (
            frame["fixed"] is not True
            or np.any(frame["ids"] >= 0)
            or len(frame["ids"]) > _MAX_NEW_OBSERVATIONS_PER_KEYFRAME
        ):
            raise _BaFailure("invalid observer keyframe")
        revisions[frame["name"]] = frame["pose_revision"]
        ordinals.add(frame["ordinal"])
        keep = np.isin(frame["ids"], variable)
        if not keep.any():
            continue
        ids, xy = frame["ids"][keep], frame["xy"][keep]
        points = np.stack([new_landmarks[int(key)] for key in ids])
        depth = points @ frame["pose"][2, :3] + frame["pose"][2, 3]
        keep = depth > _MIN_DEPTH
        if not keep.any():
            continue
        frame["ids"], frame["xy"] = ids[keep], xy[keep]
        observers.append(frame)
    return observers


def _parse_landmarks(
    payload: Mapping[str, Any], context: dict[str, Any]
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    K = _numeric_array(payload.get("K"), (3, 3))
    if K is None or abs(float(np.linalg.det(K))) <= 1e-12:
        raise _BaFailure("K must be a finite invertible 3x3 matrix")
    old_landmarks = _id_map(payload.get("old_landmarks"), positive=True)
    new_landmarks = _id_map(payload.get("new_landmarks"), positive=False)
    landmark_revisions = _revision_map(payload.get("landmark_revisions"))
    if old_landmarks is None or new_landmarks is None or landmark_revisions is None:
        raise _BaFailure("landmark IDs, coordinates, or revisions are invalid")
    context["landmark_revisions"] = landmark_revisions
    if set(landmark_revisions) != set(new_landmarks):
        raise _BaFailure("landmark_revisions must cover the negative landmark snapshot exactly")
    return K, old_landmarks, new_landmarks


def _classify_observations(
    frames: list[dict[str, Any]],
    old_landmarks: Mapping[int, np.ndarray],
    new_landmarks: Mapping[int, np.ndarray],
) -> tuple[dict[int, list[tuple[int, int]]], bool]:
    """Negative-point observations by landmark and whether old points support the window."""
    old_count = 0
    negative_observations: dict[int, list[tuple[int, int]]] = {}
    for frame_index, frame in enumerate(frames):
        if (
            int(np.count_nonzero(frame["ids"] > 0)) > _MAX_OLD_OBSERVATIONS_PER_KEYFRAME
            or int(np.count_nonzero(frame["ids"] < 0)) > _MAX_NEW_OBSERVATIONS_PER_KEYFRAME
        ):
            raise _BaFailure(f"keyframe {frame['name']} exceeds the BA observation limit")
        for row, raw_id in enumerate(frame["ids"]):
            landmark_id = int(raw_id)
            if landmark_id > 0 and landmark_id not in old_landmarks:
                raise _BaFailure(
                    f"keyframe {frame['name']} references missing old landmark {landmark_id}"
                )
            if landmark_id < 0 and landmark_id not in new_landmarks:
                raise _BaFailure(
                    f"keyframe {frame['name']} references missing new landmark {landmark_id}"
                )
            if landmark_id > 0:
                old_count += 1
            else:
                negative_observations.setdefault(landmark_id, []).append((frame_index, row))
    total = old_count + sum(len(items) for items in negative_observations.values())
    if total < _MIN_OBSERVATIONS:
        raise _BaFailure(f"at least {_MIN_OBSERVATIONS} valid observations are required")
    return negative_observations, bool(old_count)


class _BaProblem:
    """Reprojection residuals of one window with frozen old points."""

    def __init__(
        self,
        frames: list[dict[str, Any]],
        K: np.ndarray,
        old_landmarks: Mapping[int, np.ndarray],
        new_landmarks: Mapping[int, np.ndarray],
        variable_ids: list[int],
        old_support: bool,
        rotation_type: Any,
        window_count: int | None = None,
    ) -> None:
        #: Frames past ``window_count`` are fixed observers: residuals only.
        self.window_count = len(frames) if window_count is None else int(window_count)
        self.K = K
        self.rotation_type = rotation_type
        self.old_support = old_support
        variable_index = {landmark_id: index for index, landmark_id in enumerate(variable_ids)}
        frame_rows, pixels, ids = [], [], []
        for frame_index, frame in enumerate(frames):
            for row, landmark_id in enumerate(frame["ids"]):
                frame_rows.append(frame_index)
                pixels.append(frame["xy"][row])
                ids.append(int(landmark_id))
        self.frame_indices = np.asarray(frame_rows, dtype=np.int64)
        self.window_rows = int(np.count_nonzero(self.frame_indices < self.window_count))
        self.pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        self.point_indices = np.full(len(ids), -1, dtype=np.int64)
        self.fixed_xyz = np.empty((len(ids), 3), dtype=np.float64)
        for row, landmark_id in enumerate(ids):
            if landmark_id > 0:
                self.fixed_xyz[row] = old_landmarks[landmark_id]
                continue
            if landmark_id in variable_index:
                self.point_indices[row] = variable_index[landmark_id]
            self.fixed_xyz[row] = new_landmarks[landmark_id]
        self.fixed_flags = [bool(frame["fixed"]) for frame in frames]
        #: GlueMap residuals are constants of the window but count lambda_G times as much.
        self.row_weights = np.where(np.asarray(ids, dtype=np.int64) > 0, _OLD_WEIGHT, 1.0)
        self.initial_poses = np.stack([frame["pose"] for frame in frames])
        self.initial_points = np.asarray(
            [new_landmarks[key] for key in variable_ids], dtype=np.float64
        ).reshape(-1, 3)
        self.initial_camera_xyz = (
            np.einsum("nij,nj->ni", self.initial_poses[self.frame_indices, :, :3], self.fixed_xyz)
            + self.initial_poses[self.frame_indices, :, 3]
        )
        if not np.isfinite(self.initial_camera_xyz).all() or np.any(
            self.initial_camera_xyz[:, 2] <= _MIN_DEPTH
        ):
            raise _BaFailure("snapshot contains non-finite or non-positive observation depth")
        self.variable_pose_indices: list[int] = []

    def choose_variable_poses(self, freeze_poses: bool) -> None:
        self.variable_pose_indices = (
            []
            if freeze_poses
            else [index for index, fixed in enumerate(self.fixed_flags) if not fixed]
        )
        if self.old_support:
            return
        anchors = [
            index for index, fixed in enumerate(self.fixed_flags[: self.window_count]) if fixed
        ]
        if len(anchors) < 2:
            raise _BaFailure("without old-map support two fixed keyframes set the scale")
        centers = self._camera_centers()
        baseline = float(np.linalg.norm(centers[anchors[1]] - centers[anchors[0]]))
        if not np.isfinite(baseline) or baseline <= 1e-9:
            raise _BaFailure("without old-map support the fixed keyframes need a nonzero baseline")

    def jacobian_sparsity(self) -> Any:
        """Two residual rows per observation: its keyframe's pose block and its point block."""
        from scipy.sparse import coo_matrix

        count = len(self.frame_indices)
        pose_count = len(self.variable_pose_indices) * 6
        pose_base = np.full(count, -1, dtype=np.int64)
        for local_index, frame_index in enumerate(self.variable_pose_indices):
            pose_base[self.frame_indices == frame_index] = 6 * local_index
        observations = np.arange(count, dtype=np.int64)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        has_pose = pose_base >= 0
        has_point = self.point_indices >= 0
        for offset in range(2):
            for column in range(6):
                rows.append(2 * observations[has_pose] + offset)
                columns.append(pose_base[has_pose] + column)
            for column in range(3):
                rows.append(2 * observations[has_point] + offset)
                columns.append(pose_count + 3 * self.point_indices[has_point] + column)
        rows_all, columns_all = np.concatenate(rows), np.concatenate(columns)
        variable_count = pose_count + 3 * len(self.initial_points)
        return coo_matrix(
            (np.ones(len(rows_all), dtype=np.int8), (rows_all, columns_all)),
            shape=(2 * count, variable_count),
        ).tocsr()

    def _camera_centers(self) -> np.ndarray:
        poses = self.initial_poses
        centers: np.ndarray = -np.einsum(
            "nij,nj->ni", np.transpose(poses[:, :, :3], (0, 2, 1)), poses[:, :, 3]
        )
        return centers

    def start_and_bounds(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        pose_parameters = np.empty((0, 6), dtype=np.float64)
        if self.variable_pose_indices:
            rotvecs = self.rotation_type.from_matrix(
                self.initial_poses[self.variable_pose_indices, :, :3]
            ).as_rotvec()
            pose_parameters = np.column_stack(
                (rotvecs, self.initial_poses[self.variable_pose_indices, :, 3])
            )
        x0 = np.concatenate((pose_parameters.reshape(-1), self.initial_points.reshape(-1)))
        centers = self._camera_centers()[: self.window_count]
        max_baseline = float(np.max(np.linalg.norm(centers - centers[0], axis=1)))
        scene_scale = max(
            float(np.median(self.initial_camera_xyz[: self.window_rows, 2])), max_baseline, 1e-3
        )
        translation_radius = max(0.25 * scene_scale, 2.0 * max_baseline, 1e-3)
        point_radius = max(0.25 * scene_scale, translation_radius)
        pose_lower, pose_upper = pose_parameters.copy(), pose_parameters.copy()
        pose_lower[:, :3] -= 0.35
        pose_upper[:, :3] += 0.35
        pose_lower[:, 3:] -= translation_radius
        pose_upper[:, 3:] += translation_radius
        lower = np.concatenate(
            (pose_lower.reshape(-1), (self.initial_points - point_radius).reshape(-1))
        )
        upper = np.concatenate(
            (pose_upper.reshape(-1), (self.initial_points + point_radius).reshape(-1))
        )
        return x0, (lower, upper)

    def project(
        self, poses: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        world_xyz = self.fixed_xyz.copy()
        variable_rows = self.point_indices >= 0
        if np.any(variable_rows):
            world_xyz[variable_rows] = points[self.point_indices[variable_rows]]
        camera_xyz = (
            np.einsum("nij,nj->ni", poses[self.frame_indices, :, :3], world_xyz)
            + poses[self.frame_indices, :, 3]
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected_h = camera_xyz @ self.K.T
            denominator = projected_h[:, 2]
            valid = (
                np.isfinite(camera_xyz).all(axis=1)
                & np.isfinite(projected_h).all(axis=1)
                & (camera_xyz[:, 2] > _MIN_DEPTH)
                & (np.abs(denominator) > _MIN_DEPTH)
            )
            projected = projected_h[:, :2] / np.where(valid, denominator, 1.0)[:, None]
            valid &= np.isfinite(projected).all(axis=1)
            residuals = projected - self.pixels
        return residuals, valid, camera_xyz

    def initial_cost(self) -> float:
        residuals, valid, _camera_xyz = self.project(self.initial_poses, self.initial_points)
        if not np.all(valid) or not np.isfinite(residuals).all():
            raise _BaFailure("snapshot contains invalid projected observations")
        cost = _huber_cost(residuals.reshape(-1))
        if not np.isfinite(cost):
            raise _BaFailure("snapshot robust cost is not finite")
        return cost

    def unpack(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.variable_pose_indices) * 6
        pose_values = parameters[:count].reshape(-1, 6)
        rotations = (
            self.rotation_type.from_rotvec(pose_values[:, :3]).as_matrix()
            if self.variable_pose_indices
            else np.empty((0, 3, 3), dtype=np.float64)
        )
        poses = self.initial_poses.copy()
        for local_index, frame_index in enumerate(self.variable_pose_indices):
            poses[frame_index, :, :3] = rotations[local_index]
            poses[frame_index, :, 3] = pose_values[local_index, 3:]
        return poses, parameters[count:].reshape(-1, 3)

    def residuals(self, parameters: np.ndarray) -> np.ndarray:
        poses, points = self.unpack(parameters)
        residuals, valid, camera_xyz = self.project(poses, points)
        if not np.all(valid):
            invalid_depth = np.maximum(-camera_xyz[:, 2], 0.0)
            invalid_depth = np.nan_to_num(invalid_depth, nan=0.0, posinf=0.0, neginf=0.0)
            penalty = 50.0 + 10.0 * invalid_depth
            residuals[~valid] = penalty[~valid, None]
        return (residuals * self.row_weights[:, None]).reshape(-1)

    def final(self, x: np.ndarray, before_cost: float) -> tuple[np.ndarray, np.ndarray, float]:
        if not np.isfinite(x).all():
            raise _BaFailure("SciPy BA returned non-finite parameters", before_cost=before_cost)
        poses, points = self.unpack(x)
        residuals, valid, _camera_xyz = self.project(poses, points)
        if (
            not np.isfinite(poses).all()
            or not np.isfinite(points).all()
            or not np.all(valid)
            or not np.isfinite(residuals).all()
        ):
            raise _BaFailure(
                "candidate contains non-finite values or non-positive observation depth",
                before_cost=before_cost,
            )
        after_cost = _huber_cost(residuals.reshape(-1))
        if not np.isfinite(after_cost) or after_cost >= before_cost - 1e-9:
            raise _BaFailure(
                "candidate does not improve robust reprojection cost",
                before_cost=before_cost,
                after_cost=after_cost if np.isfinite(after_cost) else None,
            )
        return poses, points, after_cost
