"""Incremental, calibrated visual BA for a session map with immutable old points.

Only measured 2D observations enter the graph. Triangulated coordinates initialize
new variables; they are not counted again as independent depth measurements.
The sole BA process retains iSAM2 between snapshots. Rejected proposals remain
private; an external scale correction invalidates the graph's coordinate gauge.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import time
from typing import Any

import numpy as np

from workmap_ba import _huber_cost, _id_array, _id_map, _integer, _numeric_array


def require_gtsam():
    try:
        import gtsam
    except ImportError as error:
        raise RuntimeError(
            "online_workmap.optimizer=isam2 requires the optional GTSAM runtime"
        ) from error
    return gtsam


def snapshot_from_keyframes(
    frames, landmarks, *, session_id, source_epoch, revision, K, anchor_min_inliers=None
):
    """Copy measured arrays once; avoid Python work for every repeated old point.

    The caller holds the session lock. All arrays returned here are detached
    before that lock is released and before the process-pool feeder serializes.
    """
    all_ids = np.concatenate([frame["ba_ids"] for frame in frames])
    new_ids, counts = np.unique(all_ids[all_ids < 0], return_counts=True)
    known = np.asarray([key in landmarks for key in new_ids], dtype=bool)
    if not np.any(counts[known] >= 2):
        return None
    new_ids = new_ids[known]
    old_rows = all_ids > 0
    old_ids, first, inverse = np.unique(all_ids[old_rows], return_index=True, return_inverse=True)
    old_xyz = np.concatenate([frame["ba_xyz"] for frame in frames])[old_rows]
    unique_xyz = old_xyz[first]
    if not np.array_equal(old_xyz, unique_xyz[inverse]):
        return None
    selected = []
    for frame in frames:
        ids = frame["ba_ids"]
        keep = (ids > 0) | np.isin(ids, new_ids)
        selected.append(
            {
                "name": frame["name"],
                "ordinal": int(frame["ordinal"]),
                "pose": frame["pose"].copy(),
                "pose_revision": int(frame["pose_revision"]),
                "ids": ids[keep].copy(),
                "xy": frame["ba_xy"][keep].copy(),
                "fixed_pose": anchor_min_inliers is not None
                and np.count_nonzero(ids > 0) >= anchor_min_inliers,
            }
        )
    if sum(len(frame["ids"]) for frame in selected) < 30:
        return None
    return {
        "session_id": session_id,
        "source_epoch": int(source_epoch),
        "snapshot_revision": int(revision),
        "K": np.asarray(K, dtype=float).copy(),
        "keyframes": selected,
        "old_landmarks": {
            int(key): xyz.copy() for key, xyz in zip(old_ids, unique_xyz, strict=True)
        },
        "new_landmarks": {int(key): landmarks[int(key)]["xyz"].copy() for key in new_ids},
        "landmark_revisions": {int(key): int(landmarks[int(key)]["revision"]) for key in new_ids},
    }


def _decode_header(payload):
    if not isinstance(payload, Mapping) or not isinstance(payload.get("session_id"), str):
        raise ValueError("invalid graph session")
    for key in ("source_epoch", "snapshot_revision", "solver_generation"):
        value = payload.get(key, 0)
        if _integer(value) is None or value < 0:
            raise ValueError(f"invalid graph {key}")
    if not isinstance(payload.get("freeze_poses", False), bool):
        raise ValueError("freeze_poses must be boolean")
    K = _numeric_array(payload["K"], (3, 3))
    if K is None or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.array_equal(K[2], [0, 0, 1]):
        raise ValueError("invalid pinhole intrinsics")
    return K


def _decode_keyframe(raw, names, ordinals, points):
    pose = _numeric_array(raw["pose"], (3, 4))
    ids = _id_array(raw["ids"])
    xy = _numeric_array(raw["xy"])
    name, ordinal = raw["name"], _integer(raw["ordinal"])
    if (
        not isinstance(name, str)
        or not name
        or name in names
        or ordinal is None
        or ordinal in ordinals
        or _integer(raw["pose_revision"]) is None
        or pose is None
        or ids is None
        or xy is None
        or xy.shape != (len(ids), 2)
        or len(set(ids)) != len(ids)
        or any(int(key) not in points for key in ids)
    ):
        raise ValueError("invalid graph keyframe observations")
    if not np.allclose(pose[:, :3] @ pose[:, :3].T, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(pose[:, :3]), 1.0
    ):
        raise ValueError("invalid graph rotation")
    return {**raw, "pose": pose, "ids": ids, "xy": xy}


def _decode(payload):
    K = _decode_header(payload)
    old = _id_map(payload["old_landmarks"], positive=True)
    fresh = _id_map(payload["new_landmarks"], positive=False)
    if old is None or fresh is None or not fresh:
        raise ValueError("invalid graph landmarks")
    points = {**old, **fresh}
    frames = []
    names, ordinals = set(), set()
    for raw in payload["keyframes"]:
        frame = _decode_keyframe(raw, names, ordinals, points)
        names.add(frame["name"])
        ordinals.add(frame["ordinal"])
        frames.append(frame)
    if len(frames) < 4 or [f["ordinal"] for f in frames] != sorted(ordinals):
        raise ValueError("graph needs at least four ordered keyframes")
    revisions = {int(key): value for key, value in payload["landmark_revisions"].items()}
    if set(revisions) != set(fresh) or any(
        _integer(v) is None or v < 0 for v in revisions.values()
    ):
        raise ValueError("invalid graph landmark revisions")
    return K, old, fresh, frames


def _visual_components(frames, variable_ids):
    """Keyframes grouped by shared variable landmarks (union-find)."""
    parents = list(range(len(frames)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, frame in enumerate(frames):
        for key in frame["ids"]:
            if key in variable_ids:
                previous = owners.setdefault(int(key), index)
                parents[root(index)] = root(previous)
    components = {}
    for index, frame in enumerate(frames):
        components.setdefault(root(index), []).append(frame)
    return list(components.values())


def _fixed_poses(frames, variable_ids, freeze):
    """Anchor each connected visual component, including scale when unobserved."""
    if freeze:
        return {f["name"] for f in frames}
    fixed = {frame["name"] for frame in frames if frame.get("fixed_pose") is True}
    for component in _visual_components(frames, variable_ids):
        fixed.add(component[0]["name"])
        # Fixed points observed only in the first fixed camera do not constrain
        # the scale of the subsequently reconstructed component.
        anchored = any(np.count_nonzero(f["ids"] > 0) >= 6 for f in component[1:])
        if len(component) > 1 and not anchored:
            a, b = (f["pose"] for f in component[:2])
            ca, cb = -a[:, :3].T @ a[:, 3], -b[:, :3].T @ b[:, 3]
            if np.linalg.norm(ca - cb) <= 1e-9:
                raise ValueError("unanchored graph needs a nonzero second-camera baseline")
            fixed.add(component[1]["name"])
    return fixed


def _cost(K, frames, points, poses):
    residuals = []
    for frame in frames:
        pose = poses[frame["name"]]
        xyz = np.asarray([points[int(key)] for key in frame["ids"]]).reshape(-1, 3)
        camera = xyz @ pose[:, :3].T + pose[:, 3]
        if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 1e-6):
            raise ValueError("graph contains non-positive or non-finite observation depth")
        projected = camera @ K.T
        # GTSAM's robust projection factor applies Huber to each 2D block.
        residuals.append(
            np.linalg.norm(projected[:, :2] / projected[:, 2, None] - frame["xy"], axis=1)
        )
    return _huber_cost(np.concatenate(residuals))


class IncrementalWorkMapGraph:
    """One session/epoch's persistent iSAM2 graph; never mutates input snapshots."""

    def __init__(self):
        self.gtsam = require_gtsam()
        self._identity = None

    def _reset(self, identity):
        g = self.gtsam
        parameters = g.ISAM2Params()
        parameters.relinearizeSkip = 1
        parameters.setRelinearizeThreshold(0.01)
        # Hard pose/point constraints and narrow baselines make the normal
        # equations poorly conditioned in the river replay. QR avoids the
        # Cholesky failures reproduced with that same immutable snapshot.
        parameters.setFactorization("QR")
        self._isam = g.ISAM2(parameters)
        self._identity = identity
        self._poses = {}
        self._points = set()
        self._observations = {}
        self._old = {}
        self._fixed = {}
        self._factor_count = 0

    def solve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        result = {
            "ok": False,
            "reason": "",
            "updates": {},
            "poses": {},
            "session_id": payload.get("session_id"),
            "source_epoch": payload.get("source_epoch"),
            "snapshot_revision": payload.get("snapshot_revision"),
            "keyframe_revisions": {},
            "landmark_revisions": {},
            "before_cost": None,
            "after_cost": None,
            "optimizer": "isam2",
        }
        try:
            self._solve(payload, result)
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
            self._identity = None
            result["reason"] = f"{type(error).__name__}: {error}"
        result["solver_ms"] = (time.monotonic() - started) * 1000.0
        return result

    def _solve(self, payload, result):
        K, old, fresh, frames = _decode(payload)
        result["keyframe_revisions"] = {f["name"]: f["pose_revision"] for f in frames}
        result["landmark_revisions"] = {int(k): v for k, v in payload["landmark_revisions"].items()}
        initial_poses = {f["name"]: f["pose"] for f in frames}
        result["before_cost"] = _cost(K, frames, {**old, **fresh}, initial_poses)
        counts = Counter(int(key) for frame in frames for key in frame["ids"] if key < 0)
        variable_ids = {key for key, count in counts.items() if count >= 2}
        if not variable_ids:
            raise ValueError("graph needs multi-view new landmarks")
        fixed = _fixed_poses(frames, variable_ids, payload.get("freeze_poses", False))
        identity = (
            payload["session_id"],
            payload["source_epoch"],
            payload.get("solver_generation", 0),
            tuple(K.ravel()),
        )
        measurements = {
            (f["name"], int(key)): tuple(xy)
            for f in frames
            for key, xy in zip(f["ids"], f["xy"], strict=True)
            if key > 0 or key in variable_ids
        }
        rebuilt = self._needs_rebuild(identity, measurements, fixed, initial_poses, old)
        if rebuilt:
            self._reset(identity)
        graph, values = self._insert_poses(frames, fixed)
        added = self._add_measurements(graph, values, K, {**old, **fresh}, old, fixed, measurements)
        poses, updates = self._estimate(frames, fixed, fresh, variable_ids)
        result["after_cost"] = _cost(K, frames, {**old, **fresh, **updates}, poses)
        result.update(
            {
                "ok": result["after_cost"] < result["before_cost"] - 1e-9,
                "poses": {name: pose.tolist() for name, pose in poses.items()},
                "updates": {key: xyz.tolist() for key, xyz in updates.items()},
                "graph_rebuilt": rebuilt,
                "graph_keyframes": len(self._poses),
                "graph_observations": len(self._observations),
                "graph_new_observations": added,
                "graph_points": len(self._points),
                "fixed_pose_names": sorted(fixed),
                "graph_factors": self._factor_count,
            }
        )
        if not result["ok"]:
            result["reason"] = "graph reprojection cost did not improve"

    def _needs_rebuild(self, identity, measurements, fixed, initial_poses, old):
        if self._identity != identity:
            return True
        rebuilt = any(measurements.get(key) != xy for key, xy in self._observations.items())
        rebuilt |= fixed.intersection(self._poses) != self._fixed.keys()
        rebuilt |= any(
            not np.array_equal(initial_poses[name], pose) for name, pose in self._fixed.items()
        )
        if any(
            key not in old or not np.array_equal(old[key], xyz) for key, xyz in self._old.items()
        ):
            raise ValueError("frozen graph landmarks changed")
        return rebuilt

    def _insert_poses(self, frames, fixed):
        """New keyframe poses and pose constraints for the next iSAM2 update."""
        g = self.gtsam
        graph, values = g.NonlinearFactorGraph(), g.Values()
        for frame in frames:
            name, pose = frame["name"], frame["pose"]
            if name in self._poses:
                continue
            pose_key = g.symbol("x", len(self._poses))
            self._poses[name] = pose_key
            camera_to_world = g.Pose3(g.Rot3(pose[:, :3].T), -pose[:, :3].T @ pose[:, 3])
            values.insert(pose_key, camera_to_world)
            if name in fixed:
                graph.add(g.NonlinearEqualityPose3(pose_key, camera_to_world))
                self._fixed[name] = pose.copy()
        return graph, values

    def _add_measurements(self, graph, values, K, points, old, fixed, measurements):
        g = self.gtsam
        calibration = g.Cal3_S2(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2])
        noise = g.noiseModel.Robust.Create(
            g.noiseModel.mEstimator.Huber.Create(2.0), g.noiseModel.Isotropic.Sigma(2, 1.0)
        )
        self._old.update({key: xyz.copy() for key, xyz in old.items() if key not in self._old})
        added = 0
        for (name, point_id), xy in measurements.items():
            if (name, point_id) in self._observations:
                continue
            self._observations[(name, point_id)] = xy
            added += 1
            # Both variables are fixed: this residual is a constant. Keep
            # it in publication cost/support checks, outside the solver.
            if name in fixed and point_id > 0:
                continue
            point_key = g.symbol("a" if point_id > 0 else "l", abs(point_id))
            if point_id not in self._points:
                values.insert(point_key, points[point_id])
                if point_id > 0:
                    graph.add(g.NonlinearEqualityPoint3(point_key, points[point_id]))
                    self._old[point_id] = points[point_id].copy()
                self._points.add(point_id)
            graph.add(
                g.GenericProjectionFactorCal3_S2(
                    np.asarray(xy), noise, self._poses[name], point_key, calibration
                )
            )
        self._factor_count += graph.size()
        self._isam.update(graph, values)
        self._isam.update()
        self._isam.update()
        return added

    def _estimate(self, frames, fixed, fresh, variable_ids):
        g = self.gtsam
        estimate = self._isam.calculateEstimate()
        poses = {}
        for frame in frames:
            name = frame["name"]
            if name in fixed:
                poses[name] = frame["pose"].copy()
                continue
            poses[name] = estimate.atPose3(self._poses[name]).inverse().matrix()[:3, :4]
            if np.allclose(poses[name], frame["pose"], rtol=0.0, atol=1e-12):
                poses[name] = frame["pose"].copy()
        updates = {key: estimate.atPoint3(g.symbol("l", abs(key))) for key in variable_ids}
        updates = {
            key: xyz
            for key, xyz in updates.items()
            if not np.allclose(xyz, fresh[key], rtol=0.0, atol=1e-12)
        }
        return poses, updates


_PROCESS_GRAPHS: dict[bool, IncrementalWorkMapGraph] = {}


def solve_graph_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    frozen = bool(payload.get("freeze_poses", False))
    if frozen not in _PROCESS_GRAPHS:
        _PROCESS_GRAPHS[frozen] = IncrementalWorkMapGraph()
    return _PROCESS_GRAPHS[frozen].solve(payload)
