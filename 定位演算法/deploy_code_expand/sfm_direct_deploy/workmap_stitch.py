"""Submap stitch: bundle-adjust a drifted session submap onto a strong frozen-map fix.

The anchor keyframe and the loop frame (whose pose comes from frozen-map PnP) are
fixed, as are all frozen-map points. Free variables are the keyframe poses between
them and every session point seen in two or more views. Runs in the BA process.
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

import cv2
import numpy as np

_HUBER_SCALE_PX = 2.0
#: Recorded 85-keyframe P230 snapshot: cost 14165 (20+5 evaluations) vs 14177 (12+3) with the same
#: loop-frame residual, for about 28% less solver time.
_MAX_NFEV = 12
_MAX_ROBUST_NFEV = 3
#: Weight of frozen-map residuals; fixed session points keep weight 1.
_OLD_WEIGHT = 2.0
_MAX_OLD_OBSERVATIONS_PER_KEYFRAME = 400
_MIN_DEPTH = 1e-6
_BAD_RESIDUAL_PX = 1e3
_MIN_POINT_VIEWS = 2
_MAX_VARIABLE_POINTS = 4000
_X_SCALE = "jac"
_LOSS = "huber"
_FTOL = 1e-8
_XTOL = 1e-8
_GTOL = 1e-8
LOOP_INLIER_PX = 3.0


def _failure(reason: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "reason": reason, "poses": {}, "updates": {}, **extra}


def _huber(residuals: np.ndarray) -> float:
    magnitude = np.abs(residuals)
    scale = _HUBER_SCALE_PX
    return float(
        0.5
        * np.sum(
            np.where(magnitude <= scale, magnitude**2, 2.0 * scale * magnitude - scale * scale)
        )
    )


def solve_stitch_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Solve one stitch snapshot; never raises, failures carry a named reason."""
    try:
        return _solve(payload)
    except _Reject as rejection:
        return _failure(rejection.reason)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return _failure(f"malformed_snapshot:{type(error).__name__}")


class _Reject(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _solve(payload: Mapping[str, Any]) -> dict[str, Any]:
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    from scipy.spatial.transform import Rotation

    K = np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
    frames = list(payload["keyframes"])
    old_points = {int(k): np.asarray(v, dtype=np.float64) for k, v in payload["old_landmarks"].items()}
    new_points = {int(k): np.asarray(v, dtype=np.float64) for k, v in payload["new_landmarks"].items()}
    views: dict[int, set[int]] = {}
    for index, frame in enumerate(frames):
        for key in np.asarray(frame["ids"], dtype=np.int64):
            if int(key) in new_points:
                views.setdefault(int(key), set()).add(index)
    loop_seen = {
        int(key)
        for frame in frames
        if frame.get("loop")
        for key in np.asarray(frame["ids"], dtype=np.int64)
    }
    candidates = [key for key, seen in views.items() if len(seen) >= _MIN_POINT_VIEWS]
    candidates.sort(key=lambda key: (key not in loop_seen, -len(views[key]), key))
    variable_ids = sorted(candidates[:_MAX_VARIABLE_POINTS])
    if not variable_ids or not any(frame["fixed"] for frame in frames):
        raise _Reject("no_variable_points")
    point_index = {key: row for row, key in enumerate(variable_ids)}
    free = [index for index, frame in enumerate(frames) if not frame["fixed"]]
    pose_index = {index: slot for slot, index in enumerate(free)}
    base_rotation = [np.asarray(f["pose"], dtype=np.float64).reshape(3, 4)[:, :3] for f in frames]
    base_translation = [np.asarray(f["pose"], dtype=np.float64).reshape(3, 4)[:, 3] for f in frames]

    rows_frame: list[int] = []
    rows_kind: list[int] = []  # 0 variable new point, 1 fixed point
    rows_weight: list[float] = []
    rows_point: list[int] = []
    rows_xyz: list[np.ndarray] = []
    rows_xy: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        ids = np.asarray(frame["ids"], dtype=np.int64)
        xy = np.asarray(frame["xy"], dtype=np.float64).reshape(-1, 2)
        old_rows = 0
        for key, pixel in zip(ids, xy):
            key = int(key)
            if key in point_index:
                rows_frame.append(index)
                rows_kind.append(0)
                rows_weight.append(1.0)
                rows_point.append(point_index[key])
                rows_xyz.append(np.zeros(3))
                rows_xy.append(pixel)
            elif key in old_points and index in pose_index and old_rows < _MAX_OLD_OBSERVATIONS_PER_KEYFRAME:
                old_rows += 1
                rows_frame.append(index)
                rows_kind.append(1)
                rows_weight.append(_OLD_WEIGHT if key > 0 else 1.0)
                rows_point.append(-1)
                rows_xyz.append(old_points[key])
                rows_xy.append(pixel)
    frame_of = np.asarray(rows_frame)
    kind = np.asarray(rows_kind)
    weight = np.asarray(rows_weight)
    point_of = np.asarray(rows_point)
    fixed_xyz = np.asarray(rows_xyz)
    observed = np.asarray(rows_xy)
    if len(observed) < 30:
        raise _Reject("too_few_observations")

    n_free = len(free)
    x0 = np.zeros(6 * n_free + 3 * len(variable_ids))
    for key, row in point_index.items():
        x0[6 * n_free + 3 * row : 6 * n_free + 3 * row + 3] = new_points[key]

    def poses_of(x: np.ndarray):
        rotations, translations = list(base_rotation), list(base_translation)
        for index, slot in pose_index.items():
            delta = Rotation.from_rotvec(x[6 * slot : 6 * slot + 3]).as_matrix()
            rotations[index] = delta @ base_rotation[index]
            translations[index] = base_translation[index] + x[6 * slot + 3 : 6 * slot + 6]
        return rotations, translations

    def residuals(x: np.ndarray) -> np.ndarray:
        rotations, translations = poses_of(x)
        world = fixed_xyz.copy()
        moving = kind == 0
        points = x[6 * n_free :].reshape(-1, 3)
        world[moving] = points[point_of[moving]]
        rot = np.stack(rotations)[frame_of]
        trans = np.stack(translations)[frame_of]
        camera = np.einsum("nij,nj->ni", rot, world) + trans
        depth = camera[:, 2]
        pixels = camera @ K.T
        safe = np.where(depth > _MIN_DEPTH, depth, 1.0)
        error = pixels[:, :2] / safe[:, None] - observed
        error[depth <= _MIN_DEPTH] = _BAD_RESIDUAL_PX
        error *= weight[:, None]
        return error.reshape(-1)

    pose_rows = np.flatnonzero(np.isin(frame_of, free))
    pose_slots = np.asarray([pose_index[int(frame_of[row])] for row in pose_rows], dtype=int)
    point_rows = np.flatnonzero(kind == 0)
    jac_rows = np.concatenate(
        [
            np.broadcast_to(
                2 * pose_rows[:, None, None] + np.arange(2)[None, :, None],
                (len(pose_rows), 2, 6),
            ).ravel(),
            np.broadcast_to(
                2 * point_rows[:, None, None] + np.arange(2)[None, :, None],
                (len(point_rows), 2, 3),
            ).ravel(),
        ]
    )
    jac_columns = np.concatenate(
        [
            np.broadcast_to(
                6 * pose_slots[:, None, None] + np.arange(6)[None, None, :],
                (len(pose_rows), 2, 6),
            ).ravel(),
            np.broadcast_to(
                6 * n_free + 3 * point_of[point_rows, None, None] + np.arange(3)[None, None, :],
                (len(point_rows), 2, 3),
            ).ravel(),
        ]
    )

    def jacobian(x: np.ndarray):
        rotations, translations = poses_of(x)
        world = fixed_xyz.copy()
        world[point_rows] = x[6 * n_free :].reshape(-1, 3)[point_of[point_rows]]
        rot = np.stack(rotations)[frame_of]
        camera = np.einsum("nij,nj->ni", rot, world) + np.stack(translations)[frame_of]
        valid = camera[:, 2] > _MIN_DEPTH
        safe = np.where(valid, camera[:, 2], 1.0)
        pixels = (camera @ K.T)[:, :2] / safe[:, None]
        projection = np.broadcast_to(K[:2], (len(observed), 2, 3)).copy()
        projection[:, :, 2] -= pixels
        projection *= (weight / safe)[:, None, None]
        projection[~valid] = 0.0  # Behind-camera residuals are constant penalties.
        derivative = np.zeros((len(frames), 3, 3, 3))
        for index, slot in pose_index.items():
            _, delta_jac = cv2.Rodrigues(x[6 * slot : 6 * slot + 3])
            derivative[index] = delta_jac.reshape(3, 3, 3)
        base_camera = np.einsum(
            "nij,nj->ni", np.stack(base_rotation)[frame_of[pose_rows]], world[pose_rows]
        )
        rotation_jac = np.einsum(
            "nkrj,nj->nrk", derivative[frame_of[pose_rows]], base_camera
        )
        pose_jac = np.concatenate(
            [projection[pose_rows] @ rotation_jac, projection[pose_rows]], axis=2
        )
        point_jac = projection[point_rows] @ rot[point_rows]
        return coo_matrix(
            (
                np.concatenate([pose_jac.ravel(), point_jac.ravel()]),
                (jac_rows, jac_columns),
            ),
            shape=(2 * len(observed), len(x)),
        ).tocsr()

    started = time.perf_counter()
    before = residuals(x0)
    common = dict(
        jac=jacobian, method="trf", tr_solver="lsmr", x_scale=_X_SCALE,
        ftol=_FTOL, xtol=_XTOL, gtol=_GTOL,
    )
    # Plain least squares first: a robust loss stalls far from the solution when
    # the whole submap is offset. Then a few robust iterations shed bad tracks.
    coarse = least_squares(residuals, x0, loss="linear", max_nfev=_MAX_NFEV, **common)
    solution = least_squares(
        residuals, coarse.x, loss="huber", f_scale=_HUBER_SCALE_PX, max_nfev=_MAX_ROBUST_NFEV,
        **common,
    )
    solution.nfev += coarse.nfev
    after = residuals(solution.x)
    if not np.isfinite(solution.x).all():
        raise _Reject("nonfinite_solution")
    rotations, translations = poses_of(solution.x)
    poses = {
        frames[index]["name"]: np.column_stack((rotations[index], translations[index])).tolist()
        for index in free
    }
    points = solution.x[6 * n_free :].reshape(-1, 3)
    loop_rows = np.asarray([bool(frames[int(f)].get("loop")) for f in frame_of]) & (kind == 0)
    loop_error = lambda residual: np.linalg.norm(residual.reshape(-1, 2), axis=1)[loop_rows]  # noqa: E731
    loop_stats = lambda residual: (  # noqa: E731
        {
            "p50": float(np.percentile(loop_error(residual), 50)),
            "p90": float(np.percentile(loop_error(residual), 90)),
            "inlier_fraction": float(np.mean(loop_error(residual) <= LOOP_INLIER_PX)),
        }
        if loop_rows.any()
        else None
    )
    return {
        "ok": True,
        "reason": "",
        "session_id": payload.get("session_id"),
        "source_epoch": payload.get("source_epoch"),
        "snapshot_revision": payload.get("snapshot_revision"),
        "poses": poses,
        "updates": {key: points[row].tolist() for key, row in point_index.items()},
        "before_cost": _huber(before),
        "after_cost": _huber(after),
        "loop_before": loop_stats(before),
        "loop_after": loop_stats(after),
        "loop_p90_before": float(np.percentile(loop_error(before), 90)) if loop_rows.any() else None,
        "loop_p90_after": float(np.percentile(loop_error(after), 90)) if loop_rows.any() else None,
        "observations": int(len(observed)),
        "points": len(variable_ids),
        "nfev": int(solution.nfev),
        "status": int(solution.status),
        "message": str(solution.message),
        "solver_ms": (time.perf_counter() - started) * 1000.0,
    }


def align_independent_submap(result: Any, K: np.ndarray) -> dict[str, Any]:
    """Anchor-fixed local BA in the map frame; never exports unaligned geometry.

    Only sampled keyframes and a capped set of shared features reach the solver.
    Frozen-map poses are constants, and no old landmark is an optimization variable.
    """
    from workmap_submaps import map_pose

    if len(result.anchor_poses) < 8 or len(result.anchor_views) < 4:
        return _failure("insufficient_anchor_observations")
    scale, rotation, translation = result.similarity
    anchor_ids = {int(key) for ids, _ in result.anchor_views.values() for key in ids}
    ordered = sorted(
        result.points, key=lambda key: (key not in anchor_ids, -result.points[key].ordinal, key)
    )
    points = {
        key: (scale * (rotation @ result.points[key].xyz) + translation).tolist()
        for key in ordered[:600]
    }
    frames = []
    for ordinal, pose in sorted(result.anchor_poses.items()):
        if ordinal not in result.anchor_views:
            continue
        ids, xy = result.anchor_views[ordinal]
        keep = np.isin(ids, list(points))
        frames.append(dict(
            name=str(ordinal), fixed=True, pose=np.asarray(pose).tolist(),
            ids=ids[keep].tolist(), xy=xy[keep].tolist(),
        ))
    fixed_ordinals = {int(frame["name"]) for frame in frames}
    for frame in result.keyframes:
        if frame.ordinal in fixed_ordinals or frame.ordinal not in result.poses_local:
            continue
        keep = np.isin(frame.point_ids, list(points))
        frames.append(dict(
            name=str(frame.ordinal), fixed=False,
            pose=map_pose(result.poses_local[frame.ordinal], result.similarity).tolist(),
            ids=frame.point_ids[keep].tolist(), xy=frame.xy[keep].tolist(),
        ))
    views: dict[int, set[str]] = {}
    for frame in frames:
        for key in frame["ids"]:
            views.setdefault(key, set()).add(frame["name"])
    supported = {key for key, seen in views.items() if len(seen) >= 2}
    anchor_ids = {key for frame in frames if frame["fixed"] for key in frame["ids"]}
    if len(supported & anchor_ids) < 30:
        return _failure("insufficient_shared_anchor_tracks")
    payload = dict(
        K=np.asarray(K).tolist(), keyframes=frames, old_landmarks={},
        new_landmarks={key: value for key, value in points.items() if key in supported},
    )
    candidate = solve_stitch_snapshot(payload)
    if not candidate["ok"]:
        return candidate
    updated = candidate["updates"]
    anchor_errors = []
    for frame in frames:
        ids = [key for key in frame["ids"] if key in updated]
        if not ids:
            continue
        pose = np.asarray(
            frame["pose"] if frame["fixed"] else candidate["poses"][frame["name"]]
        )
        selected = np.isin(frame["ids"], ids)
        xyz = np.asarray([updated[key] for key in np.asarray(frame["ids"])[selected]])
        camera = xyz @ pose[:, :3].T + pose[:, 3]
        if np.any(camera[:, 2] <= 1e-6):
            return _failure("aligned_depth")
        uv = camera @ np.asarray(K).T
        errors = np.linalg.norm(
            uv[:, :2] / uv[:, 2:3] - np.asarray(frame["xy"])[selected], axis=1
        )
        if np.percentile(errors, 90) > (3.0 if frame["fixed"] else 4.0):
            return _failure("aligned_reprojection")
        if frame["fixed"]:
            anchor_errors.extend(errors)
    if len(anchor_errors) < 30:
        return _failure("anchor_reprojection")
    candidate["anchor_p90"] = float(np.percentile(anchor_errors, 90))
    return candidate
