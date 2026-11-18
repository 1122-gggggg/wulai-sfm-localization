#!/usr/bin/env python3
"""Durable session-local references for the direct visual tracker.

The frozen release remains read-only.  This module owns a separate, append-only
session archive and only exposes references after their image and observation
records have reached disk.  GPU indexing is deliberately delegated to the
relocalizer worker; the writer thread never touches provider models.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import json
import math
import multiprocessing
import os
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

import direct_paths
from workmap_depth import (
    DEPTH_SCALE_HISTORY,
    DEPTH_SCALE_HISTORY_MIN,
    DEPTH_SCALE_MAX_RATIO,
    DEPTH_MIN_INTERVAL_S,
    match_promotable_rows,
)
from two_rate_tracker import TwoRateTracker
from workmap_ba import (
    BA_MAX_OBSERVER_KEYFRAMES,
    BA_WINDOW_KEYFRAMES,
    MAX_VARIABLE_POINTS,
    solve_ba_snapshot,
)
from workmap_overlap import fit_overlap_sim3
from presubmap import PreAnchorSubmap
from workmap_state import MappingInputs, MappingState, MappingStateMachine, frozen_ratio
from workmap_submaps import (
    ANCHOR_MAX_SPAN,
    ANCHOR_MIN_COUNT,
    ANCHOR_MIN_LOCAL_INLIERS,
    AnchorPair,
    Sim3,
    SubmapRegistry,
    anchor_consistency,
    fit_anchor_similarity,
    map_pose,
)
from workmap_stitch import align_independent_submap, solve_stitch_snapshot
from workmap_submap_graph import solve_submap_graph
from workmap_query import (
    FrozenCoverage,
    SpatioTemporalQuery,
    QueryCandidates,
    QUERY_STRIDE,
    QUERY_MAX_SOURCE_KEYFRAMES,
    QUERY_MAX_SHIFT_PX,
)

if TYPE_CHECKING:  # pragma: no cover
    from direct_map import DirectMapAssets
    from direct_profile import DirectProfile


_WRITER_QUEUE_CAPACITY = 4
#: A fixed-pose observer of refined points must keep this many rows within this radius.
OBSERVER_SUPPORT_PX = 4.0
#: Rejections caused by newer session state rather than bad geometry.
_STALE_REJECTIONS = frozenset(
    {"session_epoch_or_writer", "keyframe_pose_revision", "landmark_revision"}
)
#: Time a signalled shutdown may spend on the session archive. The operator
#: UI escalates SIGTERM to SIGKILL after about one second.
_FAST_CLOSE_S = 0.5
#: Merged pre-anchor points converted to session landmarks per frame.
PRESUBMAP_POINTS_PER_FRAME = 100
#: Keyframe images stop (writer fails closed) before the workmap can push the
#: laptop toward the flight system's critical disk block, which
#: 控制介面程式/operator_interface/disk_policy.py sets at 5 GiB or 5 % free.
_MIN_FREE_BYTES = 8 * 1024**3
_MIN_FREE_PERCENT = 6.5
#: Submap merge: multi-frame anchor pairs fit a Sim(3) onto GlueMap; the stitch then
#: re-solves the submap against those anchors without moving GlueMap points.
LOOP_MIN_DRIFT_U = 0.02
LOOP_MIN_ANGLE_DEG = 0.5
#: VO drift measured on the boundary replays is about 0.1 u per u of path.
LOOP_MAX_DRIFT_RATIO = 0.3
LOOP_MAX_DRIFT_FLOOR_U = 0.05
LOOP_MAX_ANGLE_DEG = 10.0
LOOP_MAX_REPROJECTION_GROWTH_PX = 2.0
#: After stitching, the loop frame's session points must reproject this well
#: (robust: median and the share within the solver's inlier threshold).
LOOP_MAX_LOOP_P50_PX = 1.5
LOOP_MIN_INLIER_FRACTION = 0.8
#: Thinning the chain turns submap points seen by dropped keyframes into fixed constants, which
#: hides the anchors' points from the stitch (``loop_frame_sees_no_submap_points``); keep it wide.
LOOP_MAX_CHAIN = 120
#: A verified overlap pair fuses a session landmark into a GlueMap point only when the
#: session point already lies this close and the GlueMap point reprojects this well.
FUSE_MAX_DISTANCE_U = 0.05
FUSE_MAX_REPROJ_PX = 3.0
#: A graph-solved Sim(3) is applied to session data only when it differs from the applied
#: one by more than this (translation, rotation, log scale).
GRAPH_APPLY_MIN_TRANS_U = 0.005
GRAPH_APPLY_MIN_ROT_DEG = 0.2
GRAPH_APPLY_MIN_LOGSCALE = 0.002


def _graph_sigma(kind: str, translation_u: float | None = None) -> list[float]:
    """Edge sigma: rotation x3 (rad), translation x3 (u), log scale."""
    if kind == "sequential":
        return [math.radians(5.0)] * 3 + [0.2] * 3 + [0.2]
    return [math.radians(0.5)] * 3 + [max(float(translation_u or 0.0), 0.01)] * 3 + [0.02]


def _require_free_space(path: Path) -> None:
    usage = shutil.disk_usage(path)
    percent = 100.0 * usage.free / usage.total if usage.total else 0.0
    if usage.free < _MIN_FREE_BYTES or percent < _MIN_FREE_PERCENT:
        raise OSError(
            f"workmap keyframes stopped: {usage.free / 1024**3:.2f} GiB / {percent:.1f}% free"
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot encode workmap value of type {type(value).__name__}")


_ATOMIC_EXTRA_TYPES = (int, float, str, bool, type(None), np.generic)


def _copy_extra(value: Any) -> Any:
    """Independent copy of one extra landmark field without ``copy.deepcopy`` on the common types."""
    if isinstance(value, _ATOMIC_EXTRA_TYPES):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list) and all(isinstance(item, _ATOMIC_EXTRA_TYPES) for item in value):
        return list(value)
    if isinstance(value, tuple) and all(isinstance(item, _ATOMIC_EXTRA_TYPES) for item in value):
        return value
    return copy.deepcopy(value)


def _copy_landmark(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the small immutable geometry/observation record out of the tracker."""
    extras = {
        k: v
        for k, v in value.items()
        if k not in ("xyz", "observations", "last_inlier", "revision")
    }
    result: dict[str, Any] = {key: _copy_extra(item) for key, item in extras.items()}
    result["xyz"] = np.asarray(value["xyz"], dtype=np.float64).reshape(3).copy()
    observations = []
    for observation in value.get("observations", ()):
        ordinal, stamp, xy, pose = observation
        observations.append(
            (
                int(ordinal),
                float(stamp),
                np.asarray(xy, dtype=np.float64).reshape(2).copy(),
                np.asarray(pose, dtype=np.float64).reshape(3, 4).copy(),
            )
        )
    result["observations"] = tuple(observations)
    if "last_inlier" in value and value["last_inlier"] is not None:
        ordinal, stamp, xy = value["last_inlier"]
        result["last_inlier"] = (
            int(ordinal),
            float(stamp),
            np.asarray(xy, dtype=np.float64).reshape(2).copy(),
        )
    result["revision"] = int(value.get("revision", 0))
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_BA_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextlib.contextmanager
def _ba_spawn_environment():
    """Give only the spawned solver single-threaded BLAS, then restore ours.

    Spawned children import NumPy before the initializer runs, so the limit
    has to be in the environment the child starts with.
    """
    saved = {name: os.environ.get(name) for name in _BA_THREAD_ENV}
    os.environ.update({name: "1" for name in _BA_THREAD_ENV})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _exit_when_parent_dies(parent_pid: int, poll_s: float = 0.5) -> None:
    # A SIGTERM/SIGKILLed worker never runs executor shutdown, and the pool's
    # call queue keeps this child blocked forever. Reparenting is the signal.
    while True:
        if os.getppid() != parent_pid:
            os._exit(0)
        time.sleep(poll_s)


def _ba_process_initialize(parent_pid: int | None = None) -> None:
    for name in _BA_THREAD_ENV:
        os.environ[name] = "1"
    if parent_pid is not None:
        threading.Thread(
            target=_exit_when_parent_dies,
            args=(int(parent_pid),),
            name="workmap-ba-parent-watch",
            daemon=True,
        ).start()


def _ba_process_ready(optimizer: str = "local_ba") -> bool:
    if optimizer == "isam2":
        from workmap_graph import require_gtsam

        require_gtsam()
    return True


def _fit_overlap_job(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for frame in snapshot["evidence"]:
        for pair in frame["pairs"]:
            rows.append(
                {
                    **pair,
                    "keyframe_name": frame["keyframe_name"],
                    "keyframe_revision": frame["keyframe_revision"],
                    "landmark_revisions": frame["landmark_revisions"],
                    "pose_revision": frame["pose_revision"],
                    "keyframe_pose": frame["keyframe_pose"],
                    "keyframe_K": frame["keyframe_K"],
                    "image_size": frame["image_size"],
                    "session_id": frame["session_id"],
                    "source_epoch": frame["source_epoch"],
                }
            )
    return fit_overlap_sim3(rows)


def _transform_pose_sim3(
    pose: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    turned = pose[:, :3] @ rotation.T
    return np.column_stack((turned, scale * pose[:, 3] - turned @ translation))


class _Rejected(Exception):
    """A geometry candidate failed one named admission check."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _BaCandidate(NamedTuple):
    epoch: int
    names: tuple[str, ...]
    expected_kfs: dict[str, int]
    expected_points: dict[int, int]
    poses: dict[str, np.ndarray]
    updates: dict[int, np.ndarray]
    before: float
    after: float
    #: Fixed-pose keyframes outside the window and the pose revision the solver saw.
    expected_observers: dict[str, int]


class _ScaleCandidate(NamedTuple):
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    fit_ids: set[int]
    holdout_ids: set[int]
    before_3d: float
    after_3d: float
    before_2d: float
    after_2d: float
    expected_frames: dict[str, int]
    expected_points: dict[int, int]


def _parse_ba_candidate(
    session_id: str, payload: Mapping[str, Any], candidate: Mapping[str, Any]
) -> _BaCandidate:
    if not candidate.get("ok") or candidate.get("session_id") != session_id:
        raise _Rejected("candidate_identity")
    try:
        parsed = _BaCandidate(
            epoch=int(payload["source_epoch"]),
            names=tuple(str(frame["name"]) for frame in payload["keyframes"]),
            expected_kfs={
                str(frame["name"]): int(frame["pose_revision"]) for frame in payload["keyframes"]
            },
            expected_points={
                int(key): int(value) for key, value in payload["landmark_revisions"].items()
            },
            poses={
                str(name): np.asarray(value, dtype=np.float64).reshape(3, 4)
                for name, value in candidate["poses"].items()
            },
            updates={
                int(key): np.asarray(value, dtype=np.float64).reshape(3)
                for key, value in candidate["updates"].items()
            },
            before=float(candidate["before_cost"]),
            after=float(candidate["after_cost"]),
            expected_observers={
                str(frame["name"]): int(frame["pose_revision"])
                for frame in payload.get("observers") or ()
            },
        )
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        raise _Rejected("candidate_fields") from None
    if not _ba_candidate_matches_snapshot(payload, candidate, parsed):
        raise _Rejected("candidate_geometry")
    keyframes = payload["keyframes"]
    fixed_names = {str(keyframes[0]["name"])} | {
        str(frame["name"])
        for frame in keyframes
        if frame.get("fixed") is True or frame.get("fixed_pose") is True
    }
    if not payload["old_landmarks"]:
        fixed_names.add(str(keyframes[1]["name"]))
    for frame in keyframes:
        name = str(frame["name"])
        if name in fixed_names and not np.array_equal(
            parsed.poses[name], np.asarray(frame["pose"], dtype=np.float64)
        ):
            raise _Rejected("fixed_pose_gauge")
    return parsed


def _ba_candidate_matches_snapshot(
    payload: Mapping[str, Any], candidate: Mapping[str, Any], parsed: _BaCandidate
) -> bool:
    keyframe_revisions = {
        str(key): int(value) for key, value in candidate.get("keyframe_revisions", {}).items()
    }
    landmark_revisions = {
        int(key): int(value) for key, value in candidate.get("landmark_revisions", {}).items()
    }
    finite = all(
        np.isfinite(value).all() for value in (*parsed.poses.values(), *parsed.updates.values())
    )
    return (
        int(candidate.get("source_epoch", -1)) == parsed.epoch
        and int(candidate.get("snapshot_revision", -1)) == int(payload["snapshot_revision"])
        and keyframe_revisions == parsed.expected_kfs
        and landmark_revisions == parsed.expected_points
        and set(parsed.poses) == set(parsed.names)
        and bool(parsed.updates)
        and all(key < 0 and key in parsed.expected_points for key in parsed.updates)
        and finite
        and bool(np.isfinite([parsed.before, parsed.after]).all())
        and parsed.after < parsed.before - 1e-9
    )


def _parse_scale_candidate(
    snapshot: Mapping[str, Any], candidate: Mapping[str, Any]
) -> _ScaleCandidate:
    if not candidate.get("ok"):
        raise _Rejected(str(candidate.get("reason", "sim3_rejected")))
    try:
        parsed = _ScaleCandidate(
            scale=float(candidate["scale"]),
            rotation=np.asarray(candidate["rotation"], dtype=np.float64).reshape(3, 3),
            translation=np.asarray(candidate["translation"], dtype=np.float64).reshape(3),
            fit_ids={int(key) for key in candidate["fit_new_ids"]},
            holdout_ids={int(key) for key in candidate["holdout_new_ids"]},
            before_3d=float(candidate["before_3d_p90"]),
            after_3d=float(candidate["after_3d_p90"]),
            before_2d=float(candidate["before_2d_p90"]),
            after_2d=float(candidate["after_2d_p90"]),
            expected_frames={
                str(name): int(revision)
                for name, revision in snapshot["keyframe_revisions"].items()
            },
            expected_points={
                int(key): int(revision) for key, revision in snapshot["landmark_revisions"].items()
            },
        )
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        raise _Rejected("candidate_fields") from None
    if not _scale_candidate_is_admissible(parsed):
        raise _Rejected("candidate_geometry")
    return parsed


def _scale_candidate_is_admissible(parsed: _ScaleCandidate) -> bool:
    rotation = parsed.rotation
    numbers = [
        parsed.scale,
        *parsed.translation,
        parsed.before_3d,
        parsed.after_3d,
        parsed.before_2d,
        parsed.after_2d,
    ]
    if not np.isfinite(numbers).all() or not np.isfinite(rotation).all():
        return False
    return bool(
        parsed.scale > 0.0
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        and len(parsed.fit_ids) >= 6
        and len(parsed.holdout_ids) >= 6
        and not parsed.fit_ids & parsed.holdout_ids
        and (parsed.fit_ids | parsed.holdout_ids) <= parsed.expected_points.keys()
        and parsed.after_3d < parsed.before_3d - 1e-6
        and parsed.after_2d < parsed.before_2d - 1e-6
    )


def _require_old_support(frame: Mapping[str, Any], initial: np.ndarray, final: np.ndarray) -> None:
    """Frozen-map points must keep their 4 px support and their p90 error."""
    old = np.asarray(frame["ba_ids"]) > 0
    if not np.any(old):
        return
    if np.count_nonzero(final[old] <= 4.0) < np.count_nonzero(initial[old] <= 4.0):
        raise _Rejected("old_support_count")
    if np.percentile(final[old], 90) > np.percentile(initial[old], 90) + 1e-6:
        raise _Rejected("old_reprojection_p90")


def _write_frame_points(
    frame: dict[str, Any], updates: Mapping[int, np.ndarray], revision: int
) -> None:
    for row, landmark_id in enumerate(frame["ba_ids"]):
        if int(landmark_id) in updates:
            frame["ba_xyz"][row] = updates[int(landmark_id)]
    for row, landmark_id in enumerate(frame["ids"]):
        key = int(landmark_id)
        if key in updates:
            frame["xyz"][row] = updates[key]
            frame["landmark_revisions"][key] = revision


def _parse_workmap_snapshot(session_id: str, snapshot: Any):
    """(revision, pose revision, landmark revisions, keyframe revisions, anchor)."""
    if not isinstance(snapshot, Mapping):
        return None
    if str(snapshot.get("session_id", "")) != session_id:
        return None
    try:
        revision = int(snapshot["revision"])
        pose_revision = int(snapshot["keyframe_revision"])
        revisions = {int(key): int(value) for key, value in snapshot["landmark_revisions"].items()}
        raw_keyframe_revisions = snapshot.get("keyframe_revisions")
        keyframe_revisions = (
            {str(key): int(value) for key, value in raw_keyframe_revisions.items()}
            if raw_keyframe_revisions is not None
            else None
        )
        anchor_name = str(snapshot.get("reference_name", ""))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not revisions or any(key >= 0 for key in revisions):
        return None
    return revision, pose_revision, revisions, keyframe_revisions, anchor_name


def _select_keyframe_points(
    ids: np.ndarray,
    xy: np.ndarray,
    xyz: np.ndarray,
    landmarks: Mapping[int, Mapping[str, Any]],
    ordinal: int,
    candidate_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Session points this exposure really measured, sorted by landmark ID."""
    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for landmark_id, point_xy, point_xyz in zip(ids, xy, xyz, strict=True):
        key = int(landmark_id)
        observations = (
            tuple(landmarks[key].get("observations", ())) if key < 0 and key in landmarks else ()
        )
        if len(observations) < 2 or int(observations[0][0]) == int(observations[1][0]):
            continue
        if np.isfinite(point_xy).all() and np.isfinite(point_xyz).all():
            selected[key] = (point_xy.copy(), point_xyz.copy())
    # Include newly triangulated points even when the active PnP cap did not
    # keep them: their second true observation is still from this exposure.
    for raw_key in candidate_ids:
        key = int(raw_key)
        if key >= 0 or key in selected or key not in landmarks:
            continue
        landmark = landmarks[key]
        observations = tuple(landmark.get("observations", ()))
        if len(observations) < 2 or int(observations[-1][0]) != ordinal:
            continue
        current_xy = np.asarray(observations[-1][2], dtype=np.float64).reshape(2)
        current_xyz = np.asarray(landmark["xyz"], dtype=np.float64).reshape(3)
        if np.isfinite(current_xy).all() and np.isfinite(current_xyz).all():
            selected[key] = (current_xy.copy(), current_xyz.copy())
    selected_ids = np.asarray(sorted(selected), dtype=np.int64)
    selected_xy = np.asarray(
        [selected[int(key)][0] for key in selected_ids], dtype=np.float64
    ).reshape(-1, 2)
    selected_xyz = np.asarray(
        [selected[int(key)][1] for key in selected_ids], dtype=np.float64
    ).reshape(-1, 3)
    return selected_ids, selected_xy, selected_xyz


def _landmark_observation_rows(
    session_id: str,
    landmarks: Mapping[int, Mapping[str, Any]],
    epoch_by_capture: Mapping[tuple[int, float], int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for landmark_id, landmark in sorted(landmarks.items()):
        revision = int(landmark.get("revision", 0))
        for index, (ordinal, stamp, xy, pose) in enumerate(landmark.get("observations", ())):
            obs_ord = int(ordinal)
            obs_stamp = float(stamp)
            obs_row: dict[str, Any] = {
                "session_id": session_id,
                "landmark_id": int(landmark_id),
                "landmark_revision": revision,
                "view_index": int(index),
                "ordinal": obs_ord,
                "stamp": obs_stamp,
                "xy": xy,
                "xyz": landmark["xyz"],
                "cam_from_world": pose,
                "observation_source": "tracked_image_measurement",
            }
            if epoch_by_capture is not None:
                epoch = epoch_by_capture.get((obs_ord, round(obs_stamp, 6)))
                if epoch is not None:
                    obs_row["source_epoch"] = int(epoch)
            rows.append(obs_row)
        if landmark.get("last_inlier") is not None:
            inlier_ord, inlier_stamp, inlier_xy = landmark["last_inlier"]
            inlier_ord_int = int(inlier_ord)
            inlier_stamp_flt = float(inlier_stamp)
            inlier_row: dict[str, Any] = {
                "session_id": session_id,
                "landmark_id": int(landmark_id),
                "landmark_revision": revision,
                "ordinal": inlier_ord_int,
                "stamp": inlier_stamp_flt,
                "xy": inlier_xy,
                "xyz": landmark["xyz"],
                "observation_source": "visual_pnp_inlier",
            }
            if epoch_by_capture is not None:
                epoch = epoch_by_capture.get((inlier_ord_int, round(inlier_stamp_flt, 6)))
                if epoch is not None:
                    inlier_row["source_epoch"] = int(epoch)
            rows.append(inlier_row)
    return rows


def _keyframe_observation_rows(
    session_id: str, task: Mapping[str, Any], name: str, image_path: Path
) -> list[dict[str, Any]]:
    common = {
        "session_id": session_id,
        "keyframe_name": name,
        "ordinal": int(task["ordinal"]),
        "stamp": float(task["stamp"]),
        "cam_from_world": task["pose"],
        "image_path": str(image_path),
    }
    rows = [
        {
            **common,
            "landmark_id": int(landmark_id),
            "landmark_revision": int(task["landmark_revisions"][int(landmark_id)]),
            "xy": point_xy,
            "xyz": point_xyz,
            "observation_source": "keyframe_image_measurement",
        }
        for landmark_id, point_xy, point_xyz in zip(
            task["ids"], task["xy"], task["xyz"], strict=True
        )
    ]
    rows.extend(
        {
            **common,
            "landmark_id": int(landmark_id),
            "xy": point_xy,
            "xyz": point_xyz,
            "observation_source": "frozen_map_keyframe_image_measurement",
        }
        for landmark_id, point_xy, point_xyz in zip(
            task["all_ids"], task["all_xy"], task["all_xyz"], strict=True
        )
        if landmark_id > 0
    )
    return rows


class SessionWorkMap:
    """Append-only session geometry and durable keyframe archive.

    ``submit_keyframe`` is nonblocking and is safe on the frame thread.  The
    bounded writer queue carries copied records; overflow is an explicit health
    failure and the archive stops claiming durable expansion rather than
    deleting/overwriting older references.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        session_id: str,
        base_release_id: str,
        coordinate_frame_id: str,
        base_release_root: str | Path | None = None,
        base_bundle_sha256: str | None = None,
        on_durable_reference=None,
        min_reference_points: int = 8,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if base_release_root is not None:
            release_root = Path(base_release_root).expanduser().resolve()
            try:
                self.directory.relative_to(release_root)
            except ValueError:
                pass
            else:
                raise ValueError("workmap session directory must be outside the base release")
        if not session_id or not base_release_id or not coordinate_frame_id:
            raise ValueError("session, base release, and coordinate frame identities are required")
        self.session_id = str(session_id)
        self.base_release_id = str(base_release_id)
        self.coordinate_frame_id = str(coordinate_frame_id)
        if base_bundle_sha256 is not None and (
            len(base_bundle_sha256) != 64
            or any(c not in "0123456789abcdef" for c in base_bundle_sha256)
        ):
            raise ValueError("base release bundle SHA-256 must be a lowercase hex digest")
        self.base_bundle_sha256 = base_bundle_sha256
        self.min_reference_points = max(1, int(min_reference_points))
        self._submap_records: tuple[dict[str, Any], ...] = ()
        self._fusion_candidates: list[dict[str, Any]] = []
        self._on_durable_reference = on_durable_reference
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError("workmap session directory is not empty")
        self.images_dir = self.directory / "keyframes"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        # Serializes events.jsonl appends; never held together with a caller's
        # _lock so the frame thread cannot wait on an fsync.
        self._event_io_lock = threading.Lock()
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=_WRITER_QUEUE_CAPACITY)
        self._closed = False
        # Set by a shutdown signal: every drain gives up, the snapshot is still
        # written, and it records that queued work may not have been drained.
        self._abort = threading.Event()
        self._writer_stop = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._writer_job_queued_mono: float | None = None
        self._writer_error: str | None = None
        self._writer_overflow_count = 0
        self._index_condition = threading.Condition(self._lock)
        self._index_dispatch_thread: threading.Thread | None = None
        self._index_dispatch_stopping = False
        self._index_dispatch_closing = False
        self._close_report: dict[str, Any] | None = None
        self._index_job_started: float | None = None
        self._index_active_name: str | None = None
        # Names whose requested index revision is newer than the submitted one.
        # Per-frame diagnostics read this instead of scanning every keyframe.
        self._index_waiting: set[str] = set()
        self._events_pending: list[dict] = []
        self._source_epoch = 0
        self._next_keyframe = 0
        self._revision = 0
        self.landmarks: dict[int, dict[str, Any]] = {}
        # Candidate-only traversal must retain archive insertion order, not ID order.
        self._landmark_insertion_rank: dict[int, int] = {}
        #: Landmarks new or moved since the operator display last drained them.
        self._display_dirty: dict[int, None] = {}
        self.keyframes: dict[str, dict[str, Any]] = {}
        # Capture ordinals may restart in a new source epoch.
        self._keyframe_insertion_rank: dict[str, int] = {}
        self._keyframe_by_capture: dict[tuple[int, int], str] = {}
        # Negative landmark ID -> keyframes whose BA observations include it.
        self._frames_by_landmark: dict[int, set[str]] = {}
        self._pending_landmarks: dict[int, dict[str, Any]] = {}
        self._seed_reindex_pending: set[str] = set()
        self._candidate_landmark_ids: set[int] = set()
        self._last_candidate_ids: set[int] = set()
        self._last_candidate_cells: set[int] = set()

        self._epoch_by_capture: dict[tuple[int, float], int] = {}
        self._keyframes_ready = 0
        self._points_created = 0
        self._first_usage_logged: set[int] = set()
        self._index_errors = 0
        self._overlap_evidence: dict[str, dict[str, Any]] = {}
        self._overlap_revision = 0
        self._overlap_durable_revision = 0
        self._overlap_error: str | None = None

        metadata_path = self.directory / "session_meta.json"
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                existing.get("session_id") != self.session_id
                or existing.get("base_release_id") != self.base_release_id
                or existing.get("coordinate_frame_id") != self.coordinate_frame_id
            ):
                raise ValueError("existing workmap session metadata does not match this session")
            raise FileExistsError("workmap session directory already contains a prior session")
        self._write_atomic_json(
            metadata_path,
            {
                "schema": "direct-online-workmap-session/v1",
                "session_id": self.session_id,
                "base_release_id": self.base_release_id,
                "coordinate_frame_id": self.coordinate_frame_id,
                **(
                    {"base_bundle_sha256": self.base_bundle_sha256}
                    if self.base_bundle_sha256
                    else {}
                ),
                "source_epoch": self._source_epoch,
                "created_unix_s": time.time(),
                "writer_queue_capacity": _WRITER_QUEUE_CAPACITY,
            },
        )
        if self._on_durable_reference is not None:
            self._index_dispatch_thread = threading.Thread(
                target=self._index_dispatch_loop,
                name="direct-workmap-index-dispatch",
                daemon=True,
            )
            self._index_dispatch_thread.start()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="direct-workmap-writer", daemon=True
        )
        self._writer_thread.start()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def writer_healthy(self) -> bool:
        with self._lock:
            return self._writer_error is None and not self._writer_overflow_count

    @property
    def geometry_commit_ready(self) -> bool:
        """Keep three of four writer slots for real KF/observation evidence."""
        with self._lock:
            return self._queue.qsize() <= 1

    @property
    def keyframes_ready(self) -> int:
        with self._lock:
            return self._keyframes_ready

    @property
    def points_created(self) -> int:
        with self._lock:
            return self._points_created

    @property
    def index_errors(self) -> int:
        with self._lock:
            return self._index_errors

    @property
    def writer_pending(self) -> bool:
        with self._lock:
            return self._writer_job_queued_mono is not None or not self._queue.empty()

    def writer_oldest_age_ms(self) -> float:
        now = time.monotonic()
        with self._lock:
            oldest = self._writer_job_queued_mono
            with self._queue.mutex:
                if self._queue.queue and self._queue.queue[0] is not None:
                    queued = self._queue.queue[0]["queued_mono"]
                    oldest = queued if oldest is None else min(oldest, queued)
            return 0.0 if oldest is None else max(0.0, (now - oldest) * 1000.0)

    @property
    def index_handoff_pending(self) -> bool:
        if self._index_dispatch_thread is None:
            return False
        with self._lock:
            return self._index_active_name is not None or self._has_index_handoff_pending_locked()

    def index_handoff_oldest_age_ms(self) -> float:
        if self._index_dispatch_thread is None:
            return 0.0
        with self._lock:
            oldest = self._index_job_started
            for name in self._index_waiting:
                frame = self.keyframes[name]
                requested = frame.get("index_requested_revision")
                submitted = int(frame.get("index_submitted_revision", -1))
                requested_mono = frame.get("index_requested_mono")
                if (
                    requested is not None
                    and int(requested) > submitted
                    and requested_mono is not None
                ):
                    oldest = (
                        float(requested_mono)
                        if oldest is None
                        else min(oldest, float(requested_mono))
                    )
        return 0.0 if oldest is None else max(0.0, (time.monotonic() - oldest) * 1000.0)

    def _index_frame_landmarks_locked(self, name: str, ids) -> None:
        for key in ids:
            if int(key) < 0:
                self._frames_by_landmark.setdefault(int(key), set()).add(name)

    def frames_observing(self, landmark_ids) -> set[str]:
        """Keyframes whose BA observations include any of these new points."""
        with self._lock:
            names: set[str] = set()
            for key in landmark_ids:
                names |= self._frames_by_landmark.get(int(key), set())
            return names

    def _has_index_handoff_pending_locked(self) -> bool:
        return bool(self._index_waiting)

    def _refresh_index_waiting_locked(self, frame: Mapping[str, Any]) -> None:
        requested = frame.get("index_requested_revision")
        if requested is not None and int(requested) > int(
            frame.get("index_submitted_revision", -1)
        ):
            self._index_waiting.add(str(frame["name"]))
        else:
            self._index_waiting.discard(str(frame["name"]))

    def _waiting_frames_locked(self) -> list[dict[str, Any]]:
        """Waiting keyframes in capture order, as the full scan used to visit them."""
        return sorted(
            (self.keyframes[name] for name in self._index_waiting),
            key=lambda frame: int(frame["ordinal"]),
        )

    def _has_durable_index_candidate_locked(self) -> bool:
        return any(
            frame.get("durable")
            and not frame.get("index_dropped_at_close")
            and frame.get("index_requested_revision") is not None
            and int(frame["index_requested_revision"]) <= int(frame.get("durable_revision", -1))
            and int(frame["index_requested_revision"])
            > int(frame.get("index_submitted_revision", -1))
            for frame in self._waiting_frames_locked()
        )

    def _next_index_reference_locked(self):
        if self._index_dispatch_closing:
            return None
        for frame in self._waiting_frames_locked():
            requested = frame.get("index_requested_revision")
            if requested is None:
                continue
            revision = int(requested)
            if (
                not frame.get("durable")
                or revision > int(frame.get("durable_revision", -1))
                or revision <= int(frame.get("index_submitted_revision", -1))
            ):
                continue
            return (
                str(frame["name"]),
                revision,
                self._reference_from_frame_locked(frame),
            )
        return None

    def _request_index_locked(self, frame: dict[str, Any], revision: int) -> None:
        if self._index_dispatch_thread is None:
            return
        revision = int(revision)
        requested = frame.get("index_requested_revision")
        submitted = int(frame.get("index_submitted_revision", -1))
        if requested is None or int(requested) <= submitted:
            frame["index_requested_mono"] = None
        if requested is None or revision > int(requested):
            frame["index_requested_revision"] = revision
        self._refresh_index_waiting_locked(frame)
        self._index_condition.notify_all()

    def _mark_revision_durable_locked(self, frame: dict[str, Any], revision: int) -> None:
        frame["durable_revision"] = max(int(frame.get("durable_revision", -1)), int(revision))
        requested = frame.get("index_requested_revision")
        if (
            self._index_dispatch_closing and requested is not None
            and int(requested) > int(frame.get("index_completed_revision", -1))
        ):
            frame["index_dropped_at_close"] = True
        if (
            requested is not None
            and int(requested) > int(frame.get("index_submitted_revision", -1))
            and int(requested) <= int(frame["durable_revision"])
            and frame.get("index_requested_mono") is None
        ):
            frame["index_requested_mono"] = time.monotonic()
            if not frame.get("index_first_request_logged", False):
                frame["index_first_request_logged"] = True
                self._events_pending.append(
                    {
                        "event": "reference_index_requested",
                        "name": str(frame["name"]),
                        "revision": int(requested),
                        "ordinal": int(frame["ordinal"]),
                        "stamp_unix_s": time.time(),
                    }
                )

    def update_source_epoch(self, source_epoch: int) -> None:
        with self._lock:
            epoch = int(source_epoch)
            if epoch <= self._source_epoch:
                return
            previous = self._source_epoch
            self._source_epoch = epoch
            self._events_pending.append(
                {
                    "event": "source_epoch_changed",
                    "previous_source_epoch": previous,
                    "source_epoch": epoch,
                    "stamp_unix_s": time.time(),
                }
            )

    def _resolve_seed_keyframe_locked(self, seed_ordinal: int, seed_stamp: float) -> str | None:
        seed_name = self._keyframe_by_capture.get((self._source_epoch, seed_ordinal))
        seed = self.keyframes.get(seed_name) if seed_name is not None else None
        if seed is not None and abs(float(seed["stamp"]) - seed_stamp) <= 1e-6:
            return seed_name
        for (_epoch, kf_ord), kf_name in self._keyframe_by_capture.items():
            if kf_ord == seed_ordinal:
                cand = self.keyframes.get(kf_name)
                if cand is not None and abs(float(cand["stamp"]) - seed_stamp) <= 1e-6:
                    return kf_name
        return None

    def _archive_landmark_record_locked(
        self,
        raw_id: Any,
        raw: Mapping[str, Any],
        seed_additions: dict[str, list[tuple[int, np.ndarray, np.ndarray, int]]],
        revisions: dict[int, int],
        changed_ids: list[int],
        submap_id: int | None = None,
    ) -> None:
        landmark_id = int(raw_id)
        if landmark_id >= 0:
            raise ValueError("session workmap landmark IDs must be negative")
        record = _copy_landmark(raw)
        previous = self.landmarks.get(landmark_id)
        if submap_id is not None:
            record["submap_id"] = int(submap_id)
        elif "submap_id" not in record:
            record["submap_id"] = int(previous.get("submap_id", 0)) if previous is not None else 0
        is_new = previous is None
        if is_new:
            self._landmark_insertion_rank[landmark_id] = len(self._landmark_insertion_rank)
            self._points_created += 1
            self._revision += 1
            record["revision"] = self._revision
        else:
            record["revision"] = int(previous["revision"])
        self.landmarks[landmark_id] = record
        if is_new or not np.array_equal(previous["xyz"], record["xyz"]):
            self._display_dirty[landmark_id] = None
        revisions[landmark_id] = int(record["revision"])
        obs = record.get("observations", ())
        if is_new and len(obs) >= 2:
            seed_ordinal, seed_stamp, seed_xy, _ = obs[0]
            seed_name = self._resolve_seed_keyframe_locked(int(seed_ordinal), float(seed_stamp))
            if seed_name is not None:
                seed_additions.setdefault(seed_name, []).append(
                    (landmark_id, seed_xy, record["xyz"], int(record["revision"]))
                )
        if is_new or self._last_observation_signature(previous) != self._last_observation_signature(
            record
        ):
            self._pending_landmarks[landmark_id] = _copy_landmark(record)
            changed_ids.append(landmark_id)

    def archive_landmarks(
        self, landmarks: Mapping[int, Mapping[str, Any]], *, submap_id: int | None = None
    ) -> dict[int, int]:
        """Retain newly triangulated negative IDs even if the active cap drops them."""
        revisions: dict[int, int] = {}
        if not landmarks:
            return revisions
        with self._lock:
            changed_ids: list[int] = []
            seed_additions: dict[str, list[tuple[int, np.ndarray, np.ndarray, int]]] = {}
            for raw_id, raw in landmarks.items():
                self._archive_landmark_record_locked(
                    raw_id, raw, seed_additions, revisions, changed_ids, submap_id
                )
            for seed_name, additions in seed_additions.items():
                self._attach_seed_points_locked(seed_name, additions)
            if changed_ids:
                self._events_pending.append(
                    {
                        "event": "landmarks_archived",
                        "revision": self._revision,
                        "landmark_ids": sorted(changed_ids),
                        "stamp_unix_s": time.time(),
                    }
                )
        return revisions

    def _attach_seed_points_locked(
        self, seed_name: str, additions: list[tuple[int, np.ndarray, np.ndarray, int]]
    ) -> None:
        """A keyframe that was a new point's first view gains that observation."""
        seed = self.keyframes[seed_name]
        ids = np.asarray([item[0] for item in additions], dtype=np.int64)
        xy = np.stack([item[1] for item in additions])
        xyz = np.stack([item[2] for item in additions])
        seed["ba_ids"] = np.concatenate((seed["ba_ids"], ids))
        self._index_frame_landmarks_locked(seed_name, ids)
        seed["ba_xy"] = np.concatenate((seed["ba_xy"], xy))
        seed["ba_xyz"] = np.concatenate((seed["ba_xyz"], xyz))
        seed["ids"] = np.concatenate((seed["ids"], ids))
        seed["xy"] = np.concatenate((seed["xy"], xy))
        seed["xyz"] = np.concatenate((seed["xyz"], xyz))
        seed["landmark_revisions"].update((item[0], item[3]) for item in additions)
        seed["revision"] = self._revision
        if len(seed["ids"]) >= self.min_reference_points:
            self._seed_reindex_pending.add(seed_name)

    def record_geometry_updates(self) -> bool:
        """Queue pending true observations which were not followed by a keyframe."""
        with self._lock:
            if not self._pending_landmarks:
                return True
            landmarks = {
                key: _copy_landmark(value) for key, value in self._pending_landmarks.items()
            }
            latest = max(
                (
                    value["last_inlier"]
                    for value in landmarks.values()
                    if value.get("last_inlier") is not None
                ),
                key=lambda observation: int(observation[0]),
                default=(0, 0.0, np.zeros(2, dtype=np.float64)),
            )
            job_epochs: dict[tuple[int, float], int] = {}
            for lm in landmarks.values():
                for obs in lm.get("observations", ()):
                    k = (int(obs[0]), round(float(obs[1]), 6))
                    if k in self._epoch_by_capture:
                        job_epochs[k] = self._epoch_by_capture[k]
            job = {
                "kind": "geometry",
                "name": f"observations_{self.session_id[:8]}_{uuid.uuid4().hex}",
                "revision": int(self._revision),
                "ordinal": int(latest[0]),
                "stamp": float(latest[1]),
                "landmarks": landmarks,
                "epoch_by_capture": job_epochs,
                "queued_mono": time.monotonic(),
            }
            if not self._enqueue_locked(job):
                return False
            self._pending_landmarks.clear()
            return True

    @staticmethod
    def _last_observation_signature(landmark: Mapping[str, Any] | None):
        if landmark is None or landmark.get("last_inlier") is None:
            return None
        ordinal, stamp, xy = landmark["last_inlier"]
        return int(ordinal), float(stamp), tuple(np.asarray(xy, dtype=float).reshape(2))

    def note_event(self, event: Mapping[str, Any]) -> None:
        """Append one diagnostic event to the session log."""
        with self._lock:
            self._events_pending.append({**event, "stamp_unix_s": time.time()})

    def pop_display_points(
        self, limit: int
    ) -> tuple[list[int], list[list[float]], list[list[int] | None], int]:
        """Oldest ``limit`` new or moved landmarks (id, xyz, original RGB) and the total."""
        with self._lock:
            ids = list(itertools.islice(self._display_dirty, int(limit)))
            for key in ids:
                del self._display_dirty[key]
            xyz = [self.landmarks[key]["xyz"].round(5).tolist() for key in ids]
            rgb = [self.landmarks[key].get("rgb") for key in ids]
            return ids, xyz, rgb, len(self.landmarks)

    def tracker_landmarks(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return {key: _copy_landmark(value) for key, value in self.landmarks.items()}

    def set_submap_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Submap bookkeeping of the tracker, written to ``snapshot.json`` on close."""
        with self._lock:
            self._submap_records = tuple(dict(record) for record in records)

    def record_keyframe(
        self,
        gray: np.ndarray,
        *,
        ordinal: int,
        stamp: float,
        source_epoch: int,
        pose: np.ndarray,
        K: np.ndarray,
        image_size: tuple[int, int],
        ids: np.ndarray,
        xy: np.ndarray,
        xyz: np.ndarray,
        landmarks: Mapping[int, Mapping[str, Any]],
        pending_ids: Iterable[int] | None = None,
        submap_id: int = 0,
    ) -> str | None:
        """Queue one nonredundant visual keyframe and its complete point records."""
        image = np.asarray(gray, dtype=np.uint8)
        if image.ndim != 2:
            raise ValueError("workmap keyframe image must be grayscale")
        if not np.isfinite([stamp]).all():
            return None
        pose_array = np.asarray(pose, dtype=np.float64).reshape(3, 4)
        K_array = np.asarray(K, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(pose_array).all() or not np.isfinite(K_array).all():
            return None
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if len(ids) != len(xy) or len(ids) != len(xyz):
            raise ValueError("workmap keyframe IDs, image points, and world points must align")
        with self._lock:
            if pending_ids is not None:
                candidates = set(pending_ids)
            else:
                candidates = {
                    key
                    for key, lm in self._pending_landmarks.items()
                    if len(lm.get("observations", ())) >= 2
                    and int(lm["observations"][-1][0]) == int(ordinal)
                    and math.isclose(
                        float(lm["observations"][-1][1]), float(stamp), rel_tol=0.0, abs_tol=1e-6
                    )
                }
        selected_ids, selected_xy, selected_xyz = _select_keyframe_points(
            ids, xy, xyz, landmarks, int(ordinal), candidates
        )
        revisions = {int(key): int(landmarks[int(key)].get("revision", 0)) for key in selected_ids}

        with self._lock:
            self._epoch_by_capture[(int(ordinal), round(float(stamp), 6))] = int(source_epoch)
            self._source_epoch = max(self._source_epoch, int(source_epoch))
            self._revision += 1
            revision = self._revision
            cells = self._image_cells(selected_xy, int(image_size[0]), int(image_size[1]))
            should_index = self._should_index_locked(selected_ids, cells)
            pending_landmarks = {
                key: _copy_landmark(value) for key, value in self._pending_landmarks.items()
            }
            self._pending_landmarks.clear()
            job_epochs: dict[tuple[int, float], int] = {}
            for lm in pending_landmarks.values():
                for obs in lm.get("observations", ()):
                    k = (int(obs[0]), round(float(obs[1]), 6))
                    if k in self._epoch_by_capture:
                        job_epochs[k] = self._epoch_by_capture[k]
            # A later triangulation may use this exposure as its first view.
            # Keep the image even if it is not yet a searchable reference.
            self._next_keyframe += 1
            name = f"workmap_{self.session_id[:8]}_{self._next_keyframe:08d}_{uuid.uuid4().hex[:8]}"
            if should_index:
                self._candidate_landmark_ids.update(map(int, selected_ids))
                self._last_candidate_ids = set(map(int, selected_ids))
                self._last_candidate_cells = cells
            promoted_names = sorted(
                self._seed_reindex_pending,
                key=lambda key: int(self.keyframes[key]["ordinal"]),
            )
            reference_revisions = {
                key: int(self.keyframes[key]["revision"]) for key in promoted_names
            }
            if should_index:
                reference_revisions[name] = int(revision)
            job = {
                "kind": "keyframe",
                "name": name,
                "image": image.copy(),
                "ordinal": int(ordinal),
                "stamp": float(stamp),
                "source_epoch": int(source_epoch),
                "pose": pose_array.copy(),
                "K": K_array.copy(),
                "epoch_by_capture": job_epochs,
                "submap_id": int(submap_id),
                "image_size": [int(image_size[0]), int(image_size[1])],
                "ids": selected_ids.copy(),
                "xy": selected_xy.copy(),
                "xyz": selected_xyz.copy(),
                "all_ids": ids.copy(),
                "all_xy": xy.copy(),
                "all_xyz": xyz.copy(),
                "landmark_revisions": revisions,
                "landmarks": pending_landmarks,
                "reference_revisions": reference_revisions,
                "revision": int(revision),
                "queued_mono": time.monotonic(),
            }
            if not self._enqueue_locked(job):
                return None
            self._seed_reindex_pending.difference_update(promoted_names)
            self.keyframes[name] = {
                "name": name,
                "ordinal": int(ordinal),
                "stamp": float(stamp),
                "source_epoch": int(source_epoch),
                "pose": pose_array.copy(),
                "K": K_array.copy(),
                "image_size": [int(image_size[0]), int(image_size[1])],
                "ids": selected_ids.copy(),
                "xy": selected_xy.copy(),
                "xyz": selected_xyz.copy(),
                "ba_ids": ids.copy(),
                "ba_xy": xy.copy(),
                "ba_xyz": xyz.copy(),
                "revision": int(revision),
                "pose_revision": 0,
                "submap_id": int(submap_id),
                "durable": False,
                "index_ready": False,
                "index_requested_revision": (
                    int(revision)
                    if should_index and self._index_dispatch_thread is not None
                    else None
                ),
                "index_requested_mono": None,
                "index_first_request_logged": False,
                "index_submitted_revision": -1,
                "durable_revision": -1,
                "image_path": str(self.images_dir / f"{name}.png"),
                "landmark_revisions": dict(revisions),
            }
            self._keyframe_insertion_rank[name] = len(self._keyframe_insertion_rank)
            self._refresh_index_waiting_locked(self.keyframes[name])
            self._index_frame_landmarks_locked(name, ids)
            for promoted_name in promoted_names:
                self._request_index_locked(
                    self.keyframes[promoted_name],
                    reference_revisions[promoted_name],
                )
            self._keyframe_by_capture[(int(source_epoch), int(ordinal))] = name
            self._events_pending.append(
                {
                    "event": "keyframe_queued",
                    "name": name,
                    "revision": int(revision),
                    "ordinal": int(ordinal),
                    "landmark_count": int(len(selected_ids)),
                    "index_requested": bool(
                        should_index and self._index_dispatch_thread is not None
                    ),
                    "stamp_unix_s": time.time(),
                }
            )
            return name

    def _should_index_locked(self, selected_ids: np.ndarray, cells: set[int]) -> bool:
        """Index only keyframes that add session geometry or image coverage."""
        if len(selected_ids) < self.min_reference_points:
            return False
        if not self.keyframes:
            return True
        chosen = set(map(int, selected_ids))
        previous_ids = self._last_candidate_ids
        if chosen - self._candidate_landmark_ids:
            return True
        if len(cells - self._last_candidate_cells) >= 4:
            return True
        overlap = len(chosen & previous_ids) / max(1, len(chosen | previous_ids))
        return bool(previous_ids) and overlap < 0.65

    def ba_snapshot(
        self,
        source_epoch: int,
        K: np.ndarray,
        *,
        incremental: bool = False,
        anchor_min_inliers: int | None = None,
    ) -> dict[str, Any] | None:
        """Select measured views independently of the retrieval catalog."""
        with self._lock:
            available = [
                frame
                for frame in self.keyframes.values()
                if frame["source_epoch"] == source_epoch and len(frame["ba_ids"])
            ]
            if len(available) < 4 or not self.writer_healthy:
                return None
            if incremental:
                if not self.landmarks:
                    return None
                from workmap_graph import snapshot_from_keyframes

                return snapshot_from_keyframes(
                    available,
                    self.landmarks,
                    session_id=self.session_id,
                    source_epoch=source_epoch,
                    revision=self._revision,
                    K=K,
                    anchor_min_inliers=anchor_min_inliers,
                )
            return self._local_ba_snapshot_locked(available, source_epoch, K, anchor_min_inliers)

    def _local_ba_snapshot_locked(
        self,
        available: list[dict[str, Any]],
        source_epoch: int,
        K: np.ndarray,
        anchor_min_inliers: int | None,
    ) -> dict[str, Any] | None:
        """Latest keyframes of one submap plus the older one sharing the most points.

        The boundary keyframe, keyframes held by at least ``anchor_min_inliers`` GlueMap
        observations, and the scale-gauge keyframe of a window without GlueMap points are
        fixed; GlueMap points are constants of the problem.
        """
        submap_id = int(available[-1].get("submap_id", 0))
        available = [f for f in available if int(f.get("submap_id", 0)) == submap_id]
        if len(available) < 4:
            return None
        recent = available[-BA_WINDOW_KEYFRAMES:]
        window = list(recent)
        older = available[:-BA_WINDOW_KEYFRAMES]
        if older:
            shared = np.unique(
                np.concatenate([np.asarray(f["ba_ids"], dtype=np.int64) for f in recent])
            )
            window.insert(
                0,
                max(
                    older,
                    key=lambda frame: (
                        int(np.count_nonzero(np.isin(frame["ba_ids"], shared))),
                        int(frame["ordinal"]),
                    ),
                ),
            )
        old_points: dict[int, list[float]] = {}
        new_points: dict[int, list[float]] = {}
        observed_new: dict[int, set[int]] = {}
        keyframes: list[dict[str, Any]] = []
        for index, frame in enumerate(window):
            kept = self._ba_frame_observations_locked(
                frame, index, old_points, new_points, observed_new
            )
            if kept is None:
                return None
            old_count = sum(1 for key in kept[0] if key > 0)
            keyframes.append(
                {
                    "name": frame["name"],
                    "ordinal": int(frame["ordinal"]),
                    "pose": np.asarray(frame["pose"], dtype=float).tolist(),
                    "ids": kept[0],
                    "xy": kept[1],
                    "pose_revision": int(frame["pose_revision"]),
                    "fixed": index == 0
                    or (anchor_min_inliers is not None and old_count >= anchor_min_inliers),
                }
            )
        if not old_points:
            keyframes[1]["fixed"] = True
        if sum(len(frame["ids"]) for frame in keyframes) < 30 or not any(
            len(views) >= 2 for views in observed_new.values()
        ):
            return None
        return {
            "session_id": self.session_id,
            "source_epoch": int(source_epoch),
            "snapshot_revision": int(self._revision),
            "K": np.asarray(K, dtype=float).tolist(),
            "keyframes": keyframes,
            "observers": self._ba_observers_locked(
                window,
                sorted(key for key, views in observed_new.items() if len(views) >= 2)[
                    :MAX_VARIABLE_POINTS
                ],
                source_epoch,
                submap_id,
            ),
            "old_landmarks": old_points,
            "new_landmarks": new_points,
            "landmark_revisions": {key: int(self.landmarks[key]["revision"]) for key in new_points},
        }

    def _ba_observers_locked(
        self,
        window: list[dict[str, Any]],
        variable_ids: list[int],
        source_epoch: int,
        submap_id: int,
    ) -> list[dict[str, Any]]:
        """Keyframes outside the window that see the refined points, as fixed cameras.

        Local BA keeps their poses and adds their residuals of the refined points, so the
        window cannot move a point away from views the solver was not allowed to move.
        """
        window_names = {frame["name"] for frame in window}
        window_ordinals = {int(frame["ordinal"]) for frame in window}
        shared: dict[str, int] = {}
        for key in variable_ids:
            for name in self._frames_by_landmark.get(key, ()):
                if name not in window_names:
                    shared[name] = shared.get(name, 0) + 1
        ranked = sorted(
            (
                (count, int(self.keyframes[name]["ordinal"]), name)
                for name, count in shared.items()
                if self.keyframes[name]["source_epoch"] == source_epoch
                and int(self.keyframes[name].get("submap_id", 0)) == submap_id
                and int(self.keyframes[name]["ordinal"]) not in window_ordinals
            ),
            reverse=True,
        )
        variable = np.asarray(variable_ids, dtype=np.int64)
        observers: list[dict[str, Any]] = []
        taken = set(window_ordinals)
        for _count, ordinal, name in ranked:
            if len(observers) >= BA_MAX_OBSERVER_KEYFRAMES:
                break
            if ordinal in taken:
                continue
            frame = self.keyframes[name]
            ids, xy, _xyz = self._fused_view(frame)
            keep = np.isin(ids, variable)
            if not keep.any():
                continue
            taken.add(ordinal)
            observers.append(
                {
                    "name": name,
                    "ordinal": ordinal,
                    "pose": np.asarray(frame["pose"], dtype=float).tolist(),
                    "ids": ids[keep].tolist(),
                    "xy": xy[keep].tolist(),
                    "pose_revision": int(frame["pose_revision"]),
                    "fixed": True,
                }
            )
        return observers

    def _fused_view(
        self, frame: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Measured rows with fused session points relabelled as their GlueMap points."""
        ids = np.asarray(frame["ba_ids"], dtype=np.int64)
        xy = np.asarray(frame["ba_xy"], dtype=np.float64)
        xyz = np.asarray(frame["ba_xyz"], dtype=np.float64)
        fused: dict[int, int] = {}
        for key in ids[ids < 0]:
            landmark = self.landmarks.get(int(key))
            if landmark is not None and landmark.get("fused_to") is not None:
                fused[int(key)] = int(landmark["fused_to"])
        if not fused:
            return ids, xy, xyz
        mapped = np.array([fused.get(int(key), int(key)) for key in ids], dtype=np.int64)
        _, first = np.unique(mapped, return_index=True)
        keep = np.sort(first)
        return mapped[keep], xy[keep], xyz[keep]

    def _ba_frame_observations_locked(
        self,
        frame: Mapping[str, Any],
        index: int,
        old_points: dict[int, list[float]],
        new_points: dict[int, list[float]],
        observed_new: dict[int, set[int]],
    ) -> tuple[list[int], list[list[float]]] | None:
        """Measured rows of one window frame; None if a frozen point disagrees."""
        kept_ids: list[int] = []
        kept_xy: list[list[float]] = []
        ids_array, xy_array, xyz_array = self._fused_view(frame)
        ids = ids_array.tolist()
        pixels = xy_array.tolist()
        points = xyz_array.tolist()
        for key, pixel, xyz in zip(ids, pixels, points, strict=True):
            if key > 0:
                if old_points.setdefault(key, xyz) != xyz:
                    return None
            elif key < 0 and key in self.landmarks:
                if key not in new_points:
                    new_points[key] = np.asarray(self.landmarks[key]["xyz"], dtype=float).tolist()
                observed_new.setdefault(key, set()).add(index)
            else:
                continue
            kept_ids.append(key)
            kept_xy.append(pixel)
        return kept_ids, kept_xy

    @staticmethod
    def _frame_errors(
        frame: Mapping[str, Any],
        pose: np.ndarray,
        new_points: Mapping[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(frame["ba_ids"], dtype=np.int64)
        world = np.asarray(frame["ba_xyz"], dtype=np.float64).copy()
        for row in np.flatnonzero(ids < 0):
            replacement = new_points.get(int(ids[row]))
            if replacement is not None:
                world[row] = replacement
        camera = world @ pose[:, :3].T + pose[:, 3]
        depth = camera[:, 2]
        projected = camera @ np.asarray(frame["K"], dtype=np.float64).T
        error = np.full(len(ids), np.inf, dtype=np.float64)
        valid = (
            np.isfinite(projected).all(axis=1) & np.isfinite(camera).all(axis=1) & (depth > 1e-6)
        )
        error[valid] = np.linalg.norm(
            projected[valid, :2] / projected[valid, 2, None]
            - np.asarray(frame["ba_xy"], dtype=np.float64)[valid],
            axis=1,
        )
        return error, valid

    def apply_ba_candidate(self, payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
        """Commit a read-versioned copy only if all affected evidence still agrees."""
        self.last_ba_rejection_reason: str | None = None
        try:
            parsed = _parse_ba_candidate(self.session_id, payload, candidate)
            with self._lock:
                self._check_ba_revisions_locked(payload, parsed)
                changed_kfs = self._ba_changed_keyframes_locked(parsed)
                self._commit_ba_locked(payload, candidate, parsed, changed_kfs)
        except _Rejected as rejection:
            self.last_ba_rejection_reason = rejection.reason
            return False
        return True

    def _check_ba_revisions_locked(
        self, payload: Mapping[str, Any], parsed: "_BaCandidate"
    ) -> None:
        if self._closed or not self.writer_healthy or self._source_epoch != parsed.epoch:
            raise _Rejected("session_epoch_or_writer")
        if any(
            name not in self.keyframes or self.keyframes[name]["pose_revision"] != revision
            for name, revision in (*parsed.expected_kfs.items(), *parsed.expected_observers.items())
        ):
            raise _Rejected("keyframe_pose_revision")
        if any(
            key not in self.landmarks or self.landmarks[key]["revision"] != revision
            for key, revision in parsed.expected_points.items()
        ):
            raise _Rejected("landmark_revision")
        for name in parsed.names:
            frame = self.keyframes[name]
            old = frame["ba_ids"] > 0
            expected = np.asarray(
                [payload["old_landmarks"][int(key)] for key in frame["ba_ids"][old]],
                dtype=np.float64,
            ).reshape(-1, 3)
            if not np.array_equal(frame["ba_xyz"][old], expected):
                raise _Rejected("frozen_landmark_changed")

    def _ba_changed_keyframes_locked(self, parsed: "_BaCandidate") -> set[str]:
        """Every keyframe the candidate moves must keep its measured support."""
        poses, updates = parsed.poses, parsed.updates
        update_ids = set(updates)
        update_id_array = np.asarray(tuple(update_ids), dtype=np.int64)
        before_points = {key: self.landmarks[key]["xyz"] for key in update_ids}
        candidates = set(poses)
        for key in update_ids:
            candidates |= self._frames_by_landmark.get(key, set())
        changed_kfs = set()
        for name in sorted(
            (name for name in candidates if name in self.keyframes),
            key=self._keyframe_insertion_rank.__getitem__,
        ):
            frame = self.keyframes[name]
            changed_pose = name in poses and not np.array_equal(frame["pose"], poses[name])
            affected = np.isin(frame["ba_ids"], update_id_array)
            if not changed_pose and not np.any(affected):
                continue
            initial_error, initial_depth = self._frame_errors(frame, frame["pose"], before_points)
            final_error, final_depth = self._frame_errors(
                frame, poses.get(name, frame["pose"]), updates
            )
            _require_old_support(frame, initial_error, final_error)
            if np.any(affected & initial_depth & ~final_depth):
                raise _Rejected("new_depth_lost")
            if np.any(affected & ~final_depth):
                raise _Rejected("new_depth_invalid")
            if name not in poses:
                if name in parsed.expected_observers:
                    # A fixed camera may see some refined points worse, but not fewer of
                    # them within the support radius than before.
                    before_count = np.count_nonzero(initial_error[affected] <= OBSERVER_SUPPORT_PX)
                    after_count = np.count_nonzero(final_error[affected] <= OBSERVER_SUPPORT_PX)
                    if after_count < before_count:
                        raise _Rejected("observer_support_count")
                elif np.any(final_error[affected] > initial_error[affected] + 1e-6):
                    raise _Rejected("later_keyframe_reprojection")
            changed_kfs.add(name)
        return changed_kfs

    def _commit_ba_locked(
        self,
        payload: Mapping[str, Any],
        candidate: Mapping[str, Any],
        parsed: "_BaCandidate",
        changed_kfs: set[str],
    ) -> None:
        poses, updates = parsed.poses, parsed.updates
        revision = self._revision + 1
        result_rows = {
            "event": "ba_geometry_committed",
            "session_id": self.session_id,
            "revision": revision,
            "source_epoch": parsed.epoch,
            "snapshot_revision": int(payload["snapshot_revision"]),
            "poses": {name: poses[name].tolist() for name in parsed.names},
            "updated_landmarks": {str(key): point.tolist() for key, point in updates.items()},
            "before_cost": parsed.before,
            "after_cost": parsed.after,
            "optimizer": candidate.get("optimizer", "local_ba"),
            "solver_ms": candidate.get("solver_ms"),
            "graph": {key: value for key, value in candidate.items() if key.startswith("graph_")},
            "submit_to_apply_ms": candidate.get("submit_to_apply_ms"),
            "stamp_unix_s": time.time(),
        }
        job = {
            "kind": "ba_update",
            "name": f"ba_{revision}",
            "revision": revision,
            "record": result_rows,
            "queued_mono": time.monotonic(),
            "reference_revisions": {
                name: revision
                for name in changed_kfs
                if len(self.keyframes[name]["ids"]) >= self.min_reference_points
            },
        }
        if not self._enqueue_locked(job):
            raise _Rejected("writer_queue_full")
        self._revision = revision
        self._store_landmark_updates_locked(updates, revision, copy_points=True)
        submitted = {str(item["name"]): item["pose"] for item in payload["keyframes"]}
        for name in changed_kfs:
            frame = self.keyframes[name]
            if name in poses:
                frame["pose"] = poses[name].copy()
                if not np.array_equal(poses[name], submitted[name]):
                    frame["pose_revision"] += 1
            frame["revision"] = revision
            _write_frame_points(frame, updates, revision)
            if name in job["reference_revisions"]:
                self._request_index_locked(frame, revision)
        self._events_pending.append(result_rows)

    def _store_landmark_updates_locked(
        self, updates: Mapping[int, np.ndarray], revision: int, *, copy_points: bool
    ) -> None:
        for key, point in updates.items():
            record = _copy_landmark(self.landmarks[key])
            record["xyz"] = point.copy() if copy_points else point
            record["revision"] = revision
            self.landmarks[key] = record
            self._display_dirty[key] = None
            self._pending_landmarks[key] = _copy_landmark(record)

    def apply_scale_candidate(
        self, snapshot: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> set[int]:
        """Weld one connected session component without moving frozen points."""
        self.last_scale_rejection_reason: str | None = None
        try:
            parsed = _parse_scale_candidate(snapshot, candidate)
            with self._lock:
                self._check_scale_revisions_locked(snapshot, parsed)
                connected, affected_names = self._scale_component_locked(parsed)
                updates, poses = self._scaled_component_locked(parsed, connected, affected_names)
                self._commit_scale_locked(snapshot, parsed, updates, poses)
        except _Rejected as rejection:
            self.last_scale_rejection_reason = rejection.reason
            return set()
        return connected

    def _check_scale_revisions_locked(
        self, snapshot: Mapping[str, Any], parsed: "_ScaleCandidate"
    ) -> None:
        if (
            self._closed
            or not self.writer_healthy
            or self._overlap_error is not None
            or snapshot["session_id"] != self.session_id
            or int(snapshot["source_epoch"]) != self._source_epoch
            or int(snapshot["overlap_revision"]) != self._overlap_revision
        ):
            raise _Rejected("session_epoch_or_writer")
        if any(
            name not in self.keyframes or int(self.keyframes[name]["pose_revision"]) != revision
            for name, revision in parsed.expected_frames.items()
        ):
            raise _Rejected("keyframe_pose_revision")
        if any(
            key not in self.landmarks or int(self.landmarks[key]["revision"]) != revision
            for key, revision in parsed.expected_points.items()
        ):
            raise _Rejected("landmark_revision")

    def _scale_component_locked(self, parsed: "_ScaleCandidate") -> tuple[set[int], set[str]]:
        # A frame sharing any negative observation belongs to the same
        # gauge component; leaving it unchanged would break its geometry.
        connected = {next(iter(parsed.fit_ids))}
        affected_names: set[str] = set()
        frontier = list(connected)
        while frontier:
            for name in self._frames_by_landmark.get(frontier.pop(), ()):
                if name in affected_names:
                    continue
                affected_names.add(name)
                negatives = {int(key) for key in self.keyframes[name]["ba_ids"] if int(key) < 0}
                frontier.extend(negatives - connected)
                connected |= negatives
        if not (parsed.fit_ids | parsed.holdout_ids) <= connected:
            raise _Rejected("disconnected_fit")
        if not affected_names or any(key not in self.landmarks for key in connected):
            raise _Rejected("component_missing")
        return connected, affected_names

    def _scaled_component_locked(
        self, parsed: "_ScaleCandidate", connected: set[int], affected_names: set[str]
    ) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
        scale, rotation, translation = parsed.scale, parsed.rotation, parsed.translation
        updates = {
            key: scale * (rotation @ np.asarray(self.landmarks[key]["xyz"], dtype=np.float64))
            + translation
            for key in connected
        }
        poses = {
            name: _transform_pose_sim3(
                np.asarray(self.keyframes[name]["pose"], dtype=np.float64),
                scale,
                rotation,
                translation,
            )
            for name in affected_names
        }
        if any(not np.isfinite(value).all() for value in (*updates.values(), *poses.values())):
            raise _Rejected("nonfinite_component")
        for name in affected_names:
            frame = self.keyframes[name]
            initial, initial_depth = self._frame_errors(frame, frame["pose"], {})
            final, final_depth = self._frame_errors(frame, poses[name], updates)
            new = np.asarray(frame["ba_ids"], dtype=np.int64) < 0
            _require_old_support(frame, initial, final)
            if np.any(new & initial_depth & ~final_depth) or np.any(new & ~final_depth):
                raise _Rejected("new_depth_lost")
            if np.any(new & (final > initial + 1e-5)):
                raise _Rejected("component_reprojection")
        return updates, poses

    def _commit_scale_locked(
        self,
        snapshot: Mapping[str, Any],
        parsed: "_ScaleCandidate",
        updates: dict[int, np.ndarray],
        poses: dict[str, np.ndarray],
    ) -> None:
        revision = self._revision + 1
        event = {
            "event": "scale_geometry_committed",
            "session_id": self.session_id,
            "revision": revision,
            "source_epoch": self._source_epoch,
            "snapshot_revision": int(snapshot["snapshot_revision"]),
            "overlap_revision": self._overlap_revision,
            "scale": parsed.scale,
            "rotation": parsed.rotation.tolist(),
            "translation": parsed.translation.tolist(),
            "fit_new_ids": sorted(parsed.fit_ids),
            "holdout_new_ids": sorted(parsed.holdout_ids),
            "poses": {name: pose.tolist() for name, pose in poses.items()},
            "updated_landmarks": {str(key): point.tolist() for key, point in updates.items()},
            "before_3d_p90": parsed.before_3d,
            "after_3d_p90": parsed.after_3d,
            "before_2d_p90": parsed.before_2d,
            "after_2d_p90": parsed.after_2d,
            "stamp_unix_s": time.time(),
        }
        job = {
            "kind": "ba_update",
            "name": f"scale_{revision}",
            "record": event,
            "reference_revisions": {
                name: revision
                for name in poses
                if (
                    self.keyframes[name]["durable"]
                    and len(self.keyframes[name]["ids"]) >= self.min_reference_points
                )
            },
            "queued_mono": time.monotonic(),
        }
        if not self._enqueue_locked(job):
            raise _Rejected("writer_queue_full")
        self._revision = revision
        self._store_landmark_updates_locked(updates, revision, copy_points=False)
        for name, pose in poses.items():
            frame = self.keyframes[name]
            frame["pose"] = pose
            frame["pose_revision"] += 1
            frame["revision"] = revision
            _write_frame_points(frame, updates, revision)
            if name in job["reference_revisions"]:
                self._request_index_locked(frame, revision)
        self._events_pending.append(event)

    def apply_submap_transform(self, submap_id: int, delta: Sim3) -> set[int]:
        """Move one submap's points and keyframes by ``delta``; GlueMap points never move."""
        self.last_scale_rejection_reason = None
        try:
            with self._lock:
                if self._closed or not self.writer_healthy:
                    raise _Rejected("session_epoch_or_writer")
                connected = {
                    key
                    for key, landmark in self.landmarks.items()
                    if int(landmark.get("submap_id", 0)) == int(submap_id)
                    and landmark.get("fused_to") is None
                }
                tagged = {
                    name
                    for name, frame in self.keyframes.items()
                    if int(frame.get("submap_id", 0)) == int(submap_id)
                }
                affected = set(tagged)
                for key in connected:
                    affected |= self._frames_by_landmark.get(key, set())
                if not connected or not affected:
                    raise _Rejected("component_missing")
                updates = {
                    key: delta.apply_points(np.asarray(self.landmarks[key]["xyz"], dtype=np.float64))[0]
                    for key in connected
                }
                poses = {
                    name: delta.apply_pose(np.asarray(self.keyframes[name]["pose"], dtype=np.float64))
                    for name in tagged
                }
                if any(not np.isfinite(value).all() for value in (*updates.values(), *poses.values())):
                    raise _Rejected("nonfinite_component")
                for name in affected:
                    frame = self.keyframes[name]
                    initial, initial_depth = self._frame_errors(frame, frame["pose"], {})
                    final, final_depth = self._frame_errors(
                        frame, poses.get(name, frame["pose"]), updates
                    )
                    new = np.asarray(frame["ba_ids"], dtype=np.int64) < 0
                    _require_old_support(frame, initial, final)
                    if np.any(new & ~final_depth):
                        raise _Rejected("new_depth_lost")
                self._commit_submap_transform_locked(int(submap_id), delta, updates, poses, affected)
        except _Rejected as rejection:
            self.last_scale_rejection_reason = rejection.reason
            return set()
        return connected

    def _commit_submap_transform_locked(
        self,
        submap_id: int,
        delta: Sim3,
        updates: dict[int, np.ndarray],
        poses: dict[str, np.ndarray],
        affected: set[str],
    ) -> None:
        revision = self._revision + 1
        event = {
            "event": "submap_transform_committed",
            "session_id": self.session_id,
            "revision": revision,
            "submap_id": submap_id,
            "delta": [float(v) for v in delta.to_vector()],
            "poses": {name: pose.tolist() for name, pose in poses.items()},
            "updated_landmarks": {str(key): point.tolist() for key, point in updates.items()},
            "stamp_unix_s": time.time(),
        }
        job = {
            "kind": "ba_update",
            "name": f"submap_{revision}",
            "record": event,
            "reference_revisions": {
                name: revision
                for name in poses
                if (
                    self.keyframes[name]["durable"]
                    and len(self.keyframes[name]["ids"]) >= self.min_reference_points
                )
            },
            "queued_mono": time.monotonic(),
        }
        if not self._enqueue_locked(job):
            raise _Rejected("writer_queue_full")
        self._revision = revision
        self._store_landmark_updates_locked(updates, revision, copy_points=False)
        for name in affected:
            frame = self.keyframes[name]
            if name in poses:
                frame["pose"] = poses[name]
                frame["pose_revision"] += 1
            frame["revision"] = revision
            _write_frame_points(frame, updates, revision)
            if name in job["reference_revisions"]:
                self._request_index_locked(frame, revision)
        self._events_pending.append(event)

    def merge_snapshot(
        self,
        submap_id: int,
        sim: Sim3,
        anchors: list[AnchorPair],
        *,
        source_epoch: int,
        K: np.ndarray,
        anchor_min_old: int,
    ) -> dict[str, Any] | None:
        """Snapshot for stitching submap ``submap_id`` onto GlueMap with the anchor Sim(3).

        The chain is the submap's keyframes; chain poses and the submap's points start at
        ``sim`` applied to their current values. Fixed are the last earlier keyframe of
        another submap holding at least ``anchor_min_old`` GlueMap observations (if any)
        and one loop frame per anchor pair (GlueMap pose, the submap points it saw).
        """
        self.last_loop_rejection_reason: str | None = None
        try:
            with self._lock:
                if (
                    self._closed
                    or not self.writer_healthy
                    or int(source_epoch) != self._source_epoch
                ):
                    raise _Rejected("session_epoch_or_writer")
                parent, chain, path, drift = self._merge_chain_locked(
                    int(submap_id), sim, int(source_epoch), int(anchor_min_old)
                )
                angle = sim.rotation_deg()
                if (
                    drift > max(LOOP_MAX_DRIFT_FLOOR_U, LOOP_MAX_DRIFT_RATIO * path)
                    or angle > LOOP_MAX_ANGLE_DEG
                ):
                    raise _Rejected("implausible_drift")
                return self._merge_snapshot_locked(
                    int(submap_id), sim, anchors, parent, chain,
                    (np.asarray(K, dtype=float), path, drift, angle),
                )
        except _Rejected as rejection:
            self.last_loop_rejection_reason = rejection.reason
            with self._lock:
                self._events_pending.append(
                    {
                        "event": "loop_geometry_rejected",
                        "reason": rejection.reason,
                        "submap_id": int(submap_id),
                        "scale": float(sim.scale),
                        "angle_deg": float(sim.rotation_deg()),
                        "stamp_unix_s": time.time(),
                    }
                )
            return None

    def _merge_chain_locked(
        self, submap_id: int, sim: Sim3, epoch: int, anchor_min_old: int
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float, float]:
        frames = sorted(
            (frame for frame in self.keyframes.values() if int(frame["source_epoch"]) == epoch),
            key=lambda frame: int(frame["ordinal"]),
        )
        chain = [frame for frame in frames if int(frame["submap_id"]) == submap_id]
        if not chain:
            raise _Rejected("empty_submap")
        first = int(chain[0]["ordinal"])
        anchored = [
            frame
            for frame in frames
            if int(frame["submap_id"]) != submap_id
            and int(frame["ordinal"]) < first
            and np.count_nonzero(np.asarray(frame["ba_ids"]) > 0) >= anchor_min_old
        ]
        parent = anchored[-1] if anchored else None
        centers = np.stack(
            [-np.asarray(f["pose"])[:, :3].T @ np.asarray(f["pose"])[:, 3] for f in chain]
        )
        path = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
        if not math.isfinite(path) or path <= 1e-9:
            raise _Rejected("zero_path")
        mapped_poses = [sim.apply_pose(f["pose"]) for f in chain]
        mapped = np.stack([-pose[:, :3].T @ pose[:, 3] for pose in mapped_poses])
        drift = float(np.max(np.linalg.norm(mapped - centers, axis=1)))
        if len(chain) > LOOP_MAX_CHAIN:
            # Optimize a time-spanning sparse chain, not an unbounded BA graph.
            # Unsampled frames/points stay fixed and admission checks their shared tracks.
            samples = np.linspace(0, len(chain) - 1, LOOP_MAX_CHAIN, dtype=int)
            chain = [chain[int(index)] for index in samples]
        return parent, chain, path, drift

    def _merge_snapshot_locked(
        self,
        submap_id: int,
        sim: Sim3,
        anchors: list[AnchorPair],
        parent: Mapping[str, Any] | None,
        chain: list[dict[str, Any]],
        geometry: tuple[np.ndarray, float, float, float],
    ) -> dict[str, Any]:
        K, path, drift, angle = geometry
        allowed = {frame["name"] for frame in chain}
        if parent is not None:
            allowed.add(parent["name"])
        old_points: dict[int, list[float]] = {}
        new_points0: dict[int, list[float]] = {}
        frames = []
        for frame in ([parent] if parent is not None else []) + chain:
            free = frame is not parent
            pose0 = np.asarray(frame["pose"], dtype=np.float64)
            pose = sim.apply_pose(pose0) if free else pose0
            ids, ba_xy, ba_xyz = self._fused_view(frame)
            for key, xyz in zip(ids, ba_xyz):
                key = int(key)
                if key > 0:
                    old_points.setdefault(key, np.asarray(xyz, dtype=float).tolist())
                elif key in self.landmarks:
                    if key in new_points0 or key in old_points:
                        continue
                    point = np.asarray(self.landmarks[key]["xyz"], dtype=float).tolist()
                    if (
                        int(self.landmarks[key].get("submap_id", 0)) == submap_id
                        and self._frames_by_landmark.get(key, set()) <= allowed
                    ):
                        new_points0[key] = point
                    else:
                        old_points[key] = point
            frames.append(
                {
                    "name": frame["name"],
                    "ordinal": int(frame["ordinal"]),
                    "fixed": not free,
                    "pose": pose.tolist(),
                    "pose0": pose0.tolist(),
                    "ids": ids.tolist(),
                    "xy": ba_xy.tolist(),
                }
            )
        new_points = (
            dict(
                zip(
                    new_points0,
                    sim.apply_points(np.asarray(list(new_points0.values()))).tolist(),
                    strict=True,
                )
            )
            if new_points0
            else {}
        )
        seen_new = False
        for anchor in anchors:
            ids = np.asarray(anchor.ids, dtype=np.int64)
            keep = np.fromiter(
                (int(k) in new_points or int(k) in old_points for k in ids),
                dtype=bool,
                count=len(ids),
            )
            seen_new = seen_new or any(int(k) in new_points for k in ids[keep])
            frames.append(
                {
                    "name": f"__anchor_{int(anchor.ordinal)}__",
                    "loop": True,
                    "fixed": True,
                    "pose": np.asarray(anchor.map_pose, dtype=float).tolist(),
                    "ids": ids[keep].tolist(),
                    "xy": np.asarray(anchor.xy, dtype=float)[keep].tolist(),
                }
            )
        if not seen_new:
            raise _Rejected("loop_frame_sees_no_submap_points")
        return {
            "session_id": self.session_id,
            "source_epoch": self._source_epoch,
            "snapshot_revision": int(self._revision),
            "K": K.tolist(),
            "keyframes": frames,
            "old_landmarks": old_points,
            "new_landmarks": new_points,
            "new_landmarks0": new_points0,
            "pose_revisions": {f["name"]: int(f["pose_revision"]) for f in chain},
            "landmark_revisions": {key: int(self.landmarks[key]["revision"]) for key in new_points},
            "chain": [f["name"] for f in chain],
            "drift_u": drift,
            "angle_deg": angle,
            "scale": float(sim.scale),
            "path_u": path,
            "loop_ordinal": int(anchors[-1].ordinal),
            "submap_id": int(submap_id),
        }

    def apply_loop_candidate(
        self, snapshot: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Commit a stitch solution only if all touched evidence still agrees."""
        self.last_loop_rejection_reason = None
        self._loop_reject_detail = None
        try:
            with self._lock:
                updates, poses = self._checked_stitch_locked(snapshot, candidate)
                self._commit_loop_locked(
                    updates,
                    poses,
                    {
                        "loop_ordinal": int(snapshot["loop_ordinal"]),
                        "drift_u": float(snapshot["drift_u"]),
                        "angle_deg": float(snapshot["angle_deg"]),
                        "path_u": float(snapshot["path_u"]),
                        "loop_before": candidate.get("loop_before"),
                        "loop_after": candidate.get("loop_after"),
                        "cost_before": candidate.get("before_cost"),
                        "cost_after": candidate.get("after_cost"),
                        "solver_ms": candidate.get("solver_ms"),
                        "nfev": candidate.get("nfev"),
                    },
                )
        except _Rejected as rejection:
            self.last_loop_rejection_reason = rejection.reason
            with self._lock:
                self._events_pending.append(
                    {
                        "event": "loop_geometry_rejected",
                        "reason": rejection.reason,
                        "loop_ordinal": int(snapshot["loop_ordinal"]),
                        "drift_u": float(snapshot["drift_u"]),
                        "chain_keyframes": len(snapshot["chain"]),
                        "detail": getattr(self, "_loop_reject_detail", None),
                        "path_u": float(snapshot["path_u"]),
                        **{
                            key: candidate.get(key)
                            for key in (
                                "loop_before", "loop_after", "before_cost",
                                "after_cost", "nfev", "points", "observations",
                            )
                        },
                        "stamp_unix_s": time.time(),
                    }
                )
            return None
        return {"ids": set(updates), "poses": poses}

    def _checked_stitch_locked(
        self, snapshot: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
        if not candidate.get("ok"):
            raise _Rejected(str(candidate.get("reason") or "solver_rejected"))
        if (
            self._closed
            or not self.writer_healthy
            or snapshot["session_id"] != self.session_id
            or int(snapshot["source_epoch"]) != self._source_epoch
        ):
            raise _Rejected("session_epoch_or_writer")
        if any(
            name not in self.keyframes or int(self.keyframes[name]["pose_revision"]) != revision
            for name, revision in snapshot["pose_revisions"].items()
        ) or any(
            key not in self.landmarks or int(self.landmarks[key]["revision"]) != revision
            for key, revision in snapshot["landmark_revisions"].items()
        ):
            raise _Rejected("keyframe_pose_revision")
        before, after = candidate.get("loop_before"), candidate.get("loop_after")
        if (
            not before
            or not after
            or after["inlier_fraction"] < LOOP_MIN_INLIER_FRACTION
            or after["p50"] > LOOP_MAX_LOOP_P50_PX
            or after["p50"] > before["p50"] + 0.5
        ):
            raise _Rejected("loop_frame_not_stitched")
        poses = {name: np.asarray(pose, dtype=np.float64) for name, pose in candidate["poses"].items()}
        updates = {int(key): np.asarray(v, dtype=np.float64) for key, v in candidate["updates"].items()}
        chain_names = set(snapshot["chain"])
        if set(poses) != chain_names:
            raise _Rejected("pose_set_mismatch")
        self._carry_single_view_points_locked(chain_names, poses, updates)
        if any(not np.isfinite(v).all() for v in (*updates.values(), *poses.values())):
            raise _Rejected("nonfinite_loop")
        for name in self._loop_affected_locked(poses, updates):
            frame = self.keyframes[name]
            initial, initial_depth = self._frame_errors(frame, frame["pose"], {})
            final, final_depth = self._frame_errors(frame, poses.get(name, frame["pose"]), updates)
            new = np.asarray(frame["ba_ids"], dtype=np.int64) < 0
            self._require_loop_old_support(frame, initial, final)
            if np.any(new & initial_depth & ~final_depth):
                raise _Rejected("new_depth_lost")
            before_rows, after_rows = new & initial_depth, new & final_depth
            if np.any(before_rows) and np.percentile(final[after_rows], 90) > (
                np.percentile(initial[before_rows], 90) + LOOP_MAX_REPROJECTION_GROWTH_PX
            ):
                self._loop_reject_detail = {
                    "keyframe_ordinal": int(frame["ordinal"]),
                    "in_chain": name in poses,
                    "p90_before": float(np.percentile(initial[before_rows], 90)),
                    "p90_after": float(np.percentile(final[after_rows], 90)),
                    "new_rows": int(np.count_nonzero(new)),
                }
                raise _Rejected("loop_reprojection")
        return updates, poses

    @staticmethod
    def _require_loop_old_support(
        frame: Mapping[str, Any], initial: np.ndarray, final: np.ndarray
    ) -> None:
        """Frozen-map points keep their support up to a small tolerance.

        The stitch itself weighs these residuals; this is the independent check.
        """
        old = np.asarray(frame["ba_ids"]) > 0
        if not np.any(old):
            return
        kept, before = np.count_nonzero(final[old] <= 4.0), np.count_nonzero(initial[old] <= 4.0)
        if kept < before - max(2, int(0.1 * before)):
            raise _Rejected("old_support_count")
        if np.percentile(final[old], 90) > np.percentile(initial[old], 90) + 1.0:
            raise _Rejected("old_reprojection_p90")

    def _loop_affected_locked(
        self, poses: Mapping[str, np.ndarray], updates: Mapping[int, np.ndarray]
    ) -> list[str]:
        """Moved keyframes plus every other keyframe that observes a moved point."""
        names = set(poses)
        for key in updates:
            names |= self._frames_by_landmark.get(int(key), set())
        return sorted(names, key=lambda name: int(self.keyframes[name]["ordinal"]))

    def _carry_single_view_points_locked(
        self, chain_names: set[str], poses: dict[str, np.ndarray], updates: dict[int, np.ndarray]
    ) -> None:
        """Session points the solver left fixed follow the pose of their first keyframe."""
        candidates = {
            int(key)
            for name in chain_names
            for key in self.keyframes[name]["ba_ids"]
            if int(key) < 0
        }
        for key in sorted(
            (key for key in candidates if key in self.landmarks),
            key=self._landmark_insertion_rank.__getitem__,
        ):
            landmark = self.landmarks[key]
            names = self._frames_by_landmark.get(int(key), set())
            if key >= 0 or key in updates or not names or not names <= chain_names:
                continue
            host = min(names, key=lambda name: int(self.keyframes[name]["ordinal"]))
            old = np.asarray(self.keyframes[host]["pose"], dtype=np.float64)
            new = poses[host]
            camera = old[:, :3] @ np.asarray(landmark["xyz"], dtype=np.float64) + old[:, 3]
            updates[int(key)] = new[:, :3].T @ (camera - new[:, 3])

    def _commit_loop_locked(
        self,
        updates: dict[int, np.ndarray],
        poses: dict[str, np.ndarray],
        summary: Mapping[str, Any],
    ) -> None:
        revision = self._revision + 1
        affected = self._loop_affected_locked(poses, updates)
        event = {
            "event": "loop_geometry_committed",
            "session_id": self.session_id,
            "revision": revision,
            "source_epoch": self._source_epoch,
            **summary,
            "poses": {name: pose.tolist() for name, pose in poses.items()},
            "updated_landmarks": {str(key): point.tolist() for key, point in updates.items()},
            "stamp_unix_s": time.time(),
        }
        job = {
            "kind": "ba_update",
            "name": f"loop_{revision}",
            "record": event,
            "reference_revisions": {
                name: revision
                for name in affected
                if (
                    self.keyframes[name]["durable"]
                    and len(self.keyframes[name]["ids"]) >= self.min_reference_points
                )
            },
            "queued_mono": time.monotonic(),
        }
        if not self._enqueue_locked(job):
            raise _Rejected("writer_queue_full")
        self._revision = revision
        self._store_landmark_updates_locked(updates, revision, copy_points=False)
        for name in affected:
            frame = self.keyframes[name]
            if name in poses:
                frame["pose"] = poses[name]
                frame["pose_revision"] += 1
            frame["revision"] = revision
            _write_frame_points(frame, updates, revision)
            if name in job["reference_revisions"]:
                self._request_index_locked(frame, revision)
        self._events_pending.append(
            {key: value for key, value in event.items() if key not in ("poses", "updated_landmarks")}
        )

    def _reference_from_frame_locked(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        ids = np.asarray(frame["ids"], dtype=np.int64)
        return {
            "name": frame["name"],
            "image_path": frame["image_path"],
            "ids": ids.copy(),
            "xy": np.asarray(frame["xy"], dtype=np.float64).copy(),
            "xyz": np.asarray(frame["xyz"], dtype=np.float64).copy(),
            "pose": np.asarray(frame["pose"], dtype=np.float64).copy(),
            "K": np.asarray(frame["K"], dtype=np.float64).copy(),
            "revision": int(frame["revision"]),
            "session_id": self.session_id,
            "source_epoch": int(frame["source_epoch"]),
            "landmark_revisions": dict(frame["landmark_revisions"]),
            "pose_revision": int(frame["pose_revision"]),
            "ordinal": int(frame["ordinal"]),
            "stamp": float(frame["stamp"]),
            "image_size": tuple(frame["image_size"]),
        }

    def validate_snapshot(
        self,
        snapshot: Mapping[str, Any] | None,
        reference_names: tuple[str, ...] = (),
    ) -> bool:
        """Reject stale local fixes without invalidating unrelated newer KFs."""
        parsed = _parse_workmap_snapshot(self.session_id, snapshot)
        if parsed is None:
            return False
        revision, pose_revision, revisions, keyframe_revisions, anchor_name = parsed
        aliases = {str(value) for value in reference_names}
        if keyframe_revisions is not None:
            aliases.update(keyframe_revisions)
        if anchor_name:
            aliases.add(anchor_name)
        with self._lock:
            alias_to_keyframe = self._keyframe_aliases_locked()
            keyframe_names = (
                {alias_to_keyframe[alias] for alias in aliases if alias in alias_to_keyframe}
                if aliases
                else {name for name, record in self.keyframes.items() if record.get("index_ready")}
            )
            if not keyframe_names or any(
                not self.keyframes[name].get("index_ready") for name in keyframe_names
            ):
                return False
            if keyframe_revisions is not None:
                current = self._explicit_revisions_current_locked(
                    alias_to_keyframe,
                    keyframe_names,
                    keyframe_revisions,
                    revision,
                    anchor_name,
                    pose_revision,
                )
            else:
                current = any(
                    int(self.keyframes[name]["revision"]) == revision
                    and int(self.keyframes[name]["pose_revision"]) == pose_revision
                    for name in keyframe_names
                )
            return current and self._landmarks_current_locked(keyframe_names, revisions)

    def _keyframe_aliases_locked(self) -> dict[str, str]:
        aliases = {}
        for name, keyframe in self.keyframes.items():
            aliases[name] = name
            provider_name = keyframe.get("provider_reference_name")
            if provider_name:
                aliases[str(provider_name)] = name
        return aliases

    def _explicit_revisions_current_locked(
        self,
        alias_to_keyframe: Mapping[str, str],
        keyframe_names: set[str],
        keyframe_revisions: Mapping[str, int],
        revision: int,
        anchor_name: str,
        pose_revision: int,
    ) -> bool:
        if not keyframe_revisions or any(
            alias not in alias_to_keyframe
            or alias_to_keyframe[alias] not in keyframe_names
            or int(self.keyframes[alias_to_keyframe[alias]]["pose_revision"]) != expected
            for alias, expected in keyframe_revisions.items()
        ):
            return False
        if max(int(self.keyframes[name]["revision"]) for name in keyframe_names) != revision:
            return False
        anchor_keyframe = alias_to_keyframe.get(anchor_name)
        return (
            anchor_keyframe is not None
            and anchor_keyframe in keyframe_names
            and int(self.keyframes[anchor_keyframe]["pose_revision"]) == pose_revision
        )

    def _landmarks_current_locked(
        self, keyframe_names: set[str], revisions: Mapping[int, int]
    ) -> bool:
        referenced = {int(key) for name in keyframe_names for key in self.keyframes[name]["ids"]}
        for landmark_id, expected in revisions.items():
            current = self.landmarks.get(landmark_id)
            if current is None or int(current.get("revision", -1)) != expected:
                return False
            if landmark_id not in referenced:
                return False
        return True

    def mark_indexed(
        self, reference: Mapping[str, Any], result: str | None, error: str | None = None
    ) -> None:
        name = str(reference.get("name", ""))
        with self._lock:
            keyframe = self.keyframes.get(name)
            if keyframe is None:
                self._index_errors += 1
                self._events_pending.append(
                    {
                        "event": "index_orphaned",
                        "name": name,
                        "error": error,
                        "stamp_unix_s": time.time(),
                    }
                )
                return
            if int(reference.get("revision", -1)) != int(keyframe["revision"]):
                return
            if error is not None or not result:
                self._index_errors += 1
                keyframe["index_error"] = str(error or "provider returned no reference name")
                self._events_pending.append(
                    {
                        "event": "index_failed",
                        "name": name,
                        "error": keyframe["index_error"],
                        "stamp_unix_s": time.time(),
                    }
                )
                return
            was_ready = bool(keyframe["index_ready"])
            keyframe["index_ready"] = True
            keyframe.pop("index_dropped_at_close", None)
            keyframe["index_completed_revision"] = int(reference["revision"])
            keyframe["provider_reference_name"] = str(result)
            if not was_ready:
                self._keyframes_ready += 1
            self._events_pending.append(
                {
                    "event": "reference_indexed",
                    "name": name,
                    "provider_reference_name": str(result),
                    "revision": int(keyframe["revision"]),
                    "pose_revision": int(keyframe["pose_revision"]),
                    "stamp_unix_s": time.time(),
                }
            )

    def can_match_overlap(
        self,
        reference: Mapping[str, Any],
        *,
        strong_old_inliers: int,
        maximum_old_p90_px: float,
    ) -> bool:
        with self._lock:
            frame = self.keyframes.get(str(reference.get("name", "")))
            if not (
                frame is not None
                and frame["durable"]
                and frame["index_ready"]
                and int(reference["revision"]) == int(frame["revision"])
                and int(reference["pose_revision"]) == int(frame["pose_revision"])
                and int(reference["source_epoch"]) == self._source_epoch
                and len(frame["ids"]) >= self.min_reference_points
            ):
                return False
            old = np.asarray(frame["ba_ids"]) > 0
            if np.any(old):
                error, _ = self._frame_errors(frame, frame["pose"], {})
                if (
                    np.count_nonzero(error[old] <= 4.0) >= strong_old_inliers
                    and np.percentile(error[old], 90) <= maximum_old_p90_px
                ):
                    # Strong frozen-map PnP/BA already defines this KF gauge;
                    # defer expensive overlap until support weakens.
                    return False
            return True

    def record_visual_use(
        self, ids: Iterable[int], *, ordinal: int, stamp: float, ba_ids: set[int]
    ) -> set[int]:
        """Log only actual next-frame PnP inliers, never newly appended tracks."""
        reused: set[int] = set()
        with self._lock:
            for landmark_id in ids:
                key = int(landmark_id)
                record = self.landmarks.get(key)
                if record is None or key >= 0:
                    continue
                created = int(record["observations"][-1][0])
                if ordinal <= created:
                    continue
                if key not in self._first_usage_logged:
                    self._first_usage_logged.add(key)
                    self._events_pending.append(
                        {
                            "event": "first_new_landmark_pnp_use",
                            "landmark_id": key,
                            "created_ordinal": created,
                            "used_ordinal": int(ordinal),
                            "stamp": float(stamp),
                            "revision": int(record["revision"]),
                        }
                    )
                if key in ba_ids:
                    reused.add(key)
            if reused:
                self._events_pending.append(
                    {
                        "event": "ba_geometry_used",
                        "landmark_ids": sorted(reused),
                        "used_ordinal": int(ordinal),
                        "stamp": float(stamp),
                        "revision": int(self._revision),
                    }
                )
        return reused

    def record_scale_visual_use(self, ids: Iterable[int], *, ordinal: int, stamp: float) -> None:
        used = sorted({int(key) for key in ids})
        if not used:
            return
        with self._lock:
            self._events_pending.append(
                {
                    "event": "scale_geometry_used",
                    "landmark_ids": used,
                    "used_ordinal": int(ordinal),
                    "stamp": float(stamp),
                    "revision": self._revision,
                }
            )

    def record_ba_rejection(
        self,
        reason: str,
        payload: Mapping[str, Any] | None,
        submit_to_apply_ms: float | None = None,
    ) -> None:
        with self._lock:
            self._events_pending.append(
                {
                    "event": "ba_geometry_rejected",
                    "reason": str(reason),
                    "snapshot_revision": None
                    if payload is None
                    else int(payload["snapshot_revision"]),
                    "source_epoch": None if payload is None else int(payload["source_epoch"]),
                    "submit_to_apply_ms": None if submit_to_apply_ms is None else float(submit_to_apply_ms),
                    "stamp_unix_s": time.time(),
                }
            )

    def record_scale_rejection(self, reason: str, snapshot: Mapping[str, Any] | None) -> None:
        with self._lock:
            self._events_pending.append(
                {
                    "event": "scale_geometry_rejected",
                    "reason": str(reason),
                    "snapshot_revision": (
                        None if snapshot is None else int(snapshot["snapshot_revision"])
                    ),
                    "overlap_revision": (
                        None if snapshot is None else int(snapshot["overlap_revision"])
                    ),
                    "stamp_unix_s": time.time(),
                }
            )

    @property
    def overlap_revision(self) -> int:
        with self._lock:
            return self._overlap_durable_revision

    def record_overlap_evidence(self, result: Mapping[str, Any] | None, error: str | None) -> bool:
        """Admit actual cross-map matches only from the current indexed KF."""
        with self._lock:
            if error is not None:
                self._overlap_error = str(error)
                self._events_pending.append(
                    {
                        "event": "overlap_match_failed",
                        "error": str(error),
                        "stamp_unix_s": time.time(),
                    }
                )
                return False
            if result is None or result.get("session_id") != self.session_id:
                return False
            name = str(result.get("keyframe_name", ""))
            frame = self.keyframes.get(name)
            if (
                frame is None
                or not frame["index_ready"]
                or int(result.get("source_epoch", -1)) != self._source_epoch
                or int(result.get("keyframe_revision", -1)) != int(frame["revision"])
                or int(result.get("pose_revision", -1)) != int(frame["pose_revision"])
            ):
                return False
            try:
                expected = {
                    int(key): int(value) for key, value in result["landmark_revisions"].items()
                }
                measured_pose = np.asarray(result["keyframe_pose"], dtype=float).reshape(3, 4)
                measured_K = np.asarray(result["keyframe_K"], dtype=float).reshape(3, 3)
                measured_size = tuple(int(value) for value in result["image_size"])
            except (KeyError, TypeError, ValueError, AttributeError):
                return False
            if (
                not np.array_equal(measured_pose, frame["pose"])
                or not np.array_equal(measured_K, frame["K"])
                or measured_size != tuple(frame["image_size"])
            ):
                return False
            pairs = result.get("pairs")
            if not isinstance(pairs, (tuple, list)):
                return False
            if not pairs:
                self._events_pending.append(
                    {
                        "event": "overlap_no_matches",
                        "keyframe_name": name,
                        "stamp_unix_s": time.time(),
                    }
                )
                return True
            if not all(self._overlap_pair_valid_locked(pair, frame, expected) for pair in pairs):
                return False
            stored = _json_value({**result, "pairs": [dict(pair) for pair in pairs]})
            stored["overlap_revision"] = self._overlap_revision + 1
            job = {
                "kind": "overlap_evidence",
                "name": f"overlap_{name}_{self._overlap_revision + 1}",
                "record": stored,
                "queued_mono": time.monotonic(),
            }
            if not self._enqueue_locked(job):
                return False
            self._overlap_evidence[name] = stored
            self._overlap_revision += 1
            for pair in pairs:
                self._fusion_candidates.append(
                    {
                        "new_id": int(pair["new_id"]),
                        "old_id": int(pair["old_id"]),
                        "old_xyz": np.asarray(pair["old_xyz"], dtype=np.float64).reshape(3).copy(),
                        "new_xy": np.asarray(pair["new_xy"], dtype=np.float64).reshape(2).copy(),
                        "pose": np.asarray(frame["pose"], dtype=np.float64).copy(),
                        "K": np.asarray(frame["K"], dtype=np.float64).copy(),
                        "keyframe": name,
                    }
                )
            self._events_pending.append(
                {
                    "event": "overlap_matches_durable_queued",
                    "keyframe_name": name,
                    "pair_count": len(pairs),
                    "overlap_revision": self._overlap_revision,
                    "stamp_unix_s": time.time(),
                }
            )
            return True

    def take_fusion_candidates(self) -> list[dict[str, Any]]:
        """Verified (new, GlueMap) pairs recorded since the last call."""
        with self._lock:
            candidates, self._fusion_candidates = self._fusion_candidates, []
        return candidates

    def fuse_landmark(
        self, new_id: int, old_id: int, old_xyz: np.ndarray, *, distance_u: float, reprojection_px: float
    ) -> bool:
        """Point session landmark ``new_id`` at GlueMap point ``old_id`` (GlueMap never moves)."""
        with self._lock:
            current = self.landmarks.get(int(new_id))
            if current is None or current.get("fused_to") is not None or not self.writer_healthy:
                return False
            revision = self._revision + 1
            event = {
                "event": "landmark_fused",
                "kind": "landmark_fused",
                "session_id": self.session_id,
                "revision": revision,
                "new_id": int(new_id),
                "old_id": int(old_id),
                "distance_u": float(distance_u),
                "reprojection_px": float(reprojection_px),
                "stamp_unix_s": time.time(),
            }
            job = {
                "kind": "ba_update",
                "name": f"fuse_{revision}",
                "revision": revision,
                "record": event,
                "reference_revisions": {},
                "queued_mono": time.monotonic(),
            }
            if not self._enqueue_locked(job):
                return False
            self._revision = revision
            record = _copy_landmark(current)
            record["xyz"] = np.asarray(old_xyz, dtype=np.float64).reshape(3).copy()
            record["revision"] = revision
            record["fused_to"] = int(old_id)
            self.landmarks[int(new_id)] = record
            self._display_dirty[int(new_id)] = None
            self._pending_landmarks[int(new_id)] = _copy_landmark(record)
            for name in self._frames_by_landmark.get(int(new_id), set()):
                frame = self.keyframes[name]
                _write_frame_points(frame, {int(new_id): record["xyz"]}, revision)
                frame["revision"] = revision
            self._events_pending.append(event)
        return True

    def _overlap_pair_valid_locked(
        self, pair: Mapping[str, Any], frame: Mapping[str, Any], expected: Mapping[int, int]
    ) -> bool:
        """One measured new<->frozen pair, consistent with current revisions."""
        try:
            key = int(pair["new_id"])
            old = int(pair["old_id"])
            xyz = np.asarray(pair["new_xyz"], dtype=float).reshape(3)
            target = np.asarray(pair["old_xyz"], dtype=float).reshape(3)
            pixel = np.asarray(pair["new_xy"], dtype=float).reshape(2)
            old_pixel = np.asarray(pair["old_xy"], dtype=float).reshape(2)
            confidence = float(pair["confidence"])
        except (KeyError, TypeError, ValueError, AttributeError):
            return False
        if (
            key >= 0
            or old <= 0
            or key not in self.landmarks
            or key not in frame["landmark_revisions"]
        ):
            return False
        revision = int(self.landmarks[key]["revision"])
        name = pair.get("old_reference_name")
        return bool(
            expected.get(key) == revision
            and expected[key] == int(frame["landmark_revisions"][key])
            and np.array_equal(xyz, self.landmarks[key]["xyz"])
            and isinstance(name, str)
            and name
            and np.isfinite(confidence)
            and all(np.isfinite(value).all() for value in (old_pixel, xyz, target, pixel))
        )

    def overlap_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if len(self._overlap_evidence) < 3 or not self.writer_healthy or self._overlap_error:
                return None
            names = sorted(
                (
                    name
                    for name, item in self._overlap_evidence.items()
                    if int(item["overlap_revision"]) <= self._overlap_durable_revision
                ),
                key=lambda name: int(self.keyframes[name]["ordinal"]),
            )[-8:]
            current: list[dict[str, Any]] = []
            point_revisions: dict[int, int] = {}
            keyframe_revisions: dict[str, int] = {}
            for name in names:
                frame = self.keyframes[name]
                original = self._overlap_evidence[name]
                pairs: list[dict[str, Any]] = []
                for pair in original["pairs"]:
                    key = int(pair["new_id"])
                    point = self.landmarks.get(key)
                    if point is None:
                        continue
                    point_revisions[key] = int(point["revision"])
                    pairs.append(
                        {
                            **pair,
                            "new_xyz": np.asarray(point["xyz"], dtype=float).tolist(),
                        }
                    )
                if not pairs:
                    continue
                keyframe_revisions[name] = int(frame["pose_revision"])
                current.append(
                    {
                        **original,
                        "pairs": pairs,
                        "source_epoch": self._source_epoch,
                        "keyframe_revision": int(frame["revision"]),
                        "pose_revision": int(frame["pose_revision"]),
                        "landmark_revisions": {
                            str(pair["new_id"]): int(
                                self.landmarks[int(pair["new_id"])]["revision"]
                            )
                            for pair in pairs
                        },
                        "keyframe_pose": np.asarray(frame["pose"], dtype=float).tolist(),
                        "keyframe_K": np.asarray(frame["K"], dtype=float).tolist(),
                        "image_size": list(frame["image_size"]),
                    }
                )
            if len(current) < 3:
                return None
            return {
                "session_id": self.session_id,
                "source_epoch": self._source_epoch,
                "snapshot_revision": self._revision,
                "overlap_revision": self._overlap_durable_revision,
                "keyframe_revisions": keyframe_revisions,
                "landmark_revisions": point_revisions,
                "evidence": current,
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "workmap_revision": int(self._revision),
                "workmap_points_created": int(self._points_created),
                "workmap_keyframes_ready": int(self._keyframes_ready),
                "workmap_writer_healthy": self._writer_error is None
                and self._writer_overflow_count == 0,
                "workmap_writer_pending": (
                    self._writer_job_queued_mono is not None or not self._queue.empty()
                ),
                "workmap_writer_oldest_age_ms": self.writer_oldest_age_ms(),
                "workmap_writer_overflow_count": int(self._writer_overflow_count),
                "workmap_index_errors": int(self._index_errors),
                "workmap_overlap_evidence_count": len(self._overlap_evidence),
                "workmap_overlap_healthy": self._overlap_error is None,
            }

    def request_abort(self) -> None:
        """Stop waiting for drains; close() still writes a (partial) snapshot."""
        self._abort.set()
        with self._index_condition:
            self._index_condition.notify_all()

    def flush_writer(self, timeout: float | None = None) -> bool:
        """Wait for accepted durable writes; False on abort or timeout."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
                if self._abort.is_set() or remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def stop_index_dispatch(self) -> None:
        """Stop new submissions; running index callbacks may still finish."""
        with self._index_condition:
            self._index_dispatch_closing = True
            for frame in self.keyframes.values():
                if (
                    frame.get("durable")
                    and frame.get("index_requested_revision") is not None
                    and int(frame["index_requested_revision"])
                    > int(frame.get("index_completed_revision", -1))
                ):
                    frame["index_dropped_at_close"] = True
            self._index_condition.notify_all()

    def set_close_report(self, report: Mapping[str, Any]) -> None:
        with self._lock:
            self._close_report = dict(report)

    def flush_index_dispatcher(self, timeout: float | None = None) -> bool:
        """Wait until durable reference revisions reach the relocator."""
        if self._index_dispatch_thread is None:
            return True
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._index_condition:
            while self._index_active_name is not None or self._has_durable_index_candidate_locked():
                remaining = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
                if self._abort.is_set() or remaining <= 0:
                    return False
                self._index_condition.wait(remaining)
        return True

    def close(self, *, shutdown_timeout_s: float = 30.0) -> None:
        if self._closed:
            return
        # Drain first so the final geometry record cannot overflow the queue.
        drain_started = time.perf_counter()
        drained = self.flush_writer(timeout=shutdown_timeout_s)
        if not self._abort.is_set():
            self.record_geometry_updates()
            drained = self.flush_writer(timeout=shutdown_timeout_s) and drained
        drain_ms = (time.perf_counter() - drain_started) * 1000.0
        with self._index_condition:
            self._closed = True
            writer_thread = self._writer_thread
            if drained and not self._abort.is_set() and writer_thread is not None:
                self._queue.put_nowait(None)
            else:
                self._writer_stop.set()
        join_started = time.perf_counter()
        if writer_thread is not None:
            writer_thread.join(timeout=shutdown_timeout_s)
        writer_join_ms = (time.perf_counter() - join_started) * 1000.0
        if writer_thread is not None and writer_thread.is_alive():
            drained = False
            self._set_writer_error(
                f"writer thread did not stop within {shutdown_timeout_s:g} s"
            )
        index_started = time.perf_counter()
        index_thread = self._index_dispatch_thread
        if index_thread is not None:
            with self._index_condition:
                self._index_dispatch_stopping = True
                self._index_condition.notify_all()
            drained = self.flush_index_dispatcher(timeout=shutdown_timeout_s) and drained
            index_thread.join(timeout=0.0 if self._abort.is_set() else shutdown_timeout_s)
            drained = drained and not index_thread.is_alive()
        index_join_ms = (time.perf_counter() - index_started) * 1000.0
        drained = drained and not self._abort.is_set()
        events_started = time.perf_counter()
        self._flush_pending_events()
        events_flush_ms = (time.perf_counter() - events_started) * 1000.0
        with self._lock:
            finished = {
                "schema": "direct-online-workmap-session/v1",
                "session_id": self.session_id,
                "base_release_id": self.base_release_id,
                "coordinate_frame_id": self.coordinate_frame_id,
                **(
                    {"base_bundle_sha256": self.base_bundle_sha256}
                    if self.base_bundle_sha256
                    else {}
                ),
                "revision": int(self._revision),
                "source_epoch": int(self._source_epoch),
                "landmarks": {
                    str(key): self._landmark_json(value)
                    for key, value in sorted(self.landmarks.items())
                },
                "keyframes": self.keyframes,
                "submaps": list(self._submap_records),
                "statistics": {
                    "points_created": int(self._points_created),
                    "keyframes_ready": int(self._keyframes_ready),
                    "writer_healthy": self._writer_error is None
                    and self._writer_overflow_count == 0,
                    "writer_error": self._writer_error,
                    "writer_overflow_count": int(self._writer_overflow_count),
                    "index_errors": int(self._index_errors),
                    "drain_complete": bool(drained),
                    "close_session_ms": {
                        "drain": drain_ms,
                        "writer_join": writer_join_ms,
                        "index_join": index_join_ms,
                        "events_flush": events_flush_ms,
                    },
                    **(
                        {
                            "close_dropped": {
                                **self._close_report["dropped"],
                                "index": sum(
                                    bool(frame.get("index_dropped_at_close"))
                                    for frame in self.keyframes.values()
                                ),
                            },
                            "close_phase_ms": self._close_report["phase_ms"],
                        }
                        if self._close_report is not None else {}
                    ),
                },
                "finished_unix_s": time.time(),
            }
            snapshot_path = self.directory / "snapshot.json"
            write_started = time.perf_counter()
            self._write_atomic_json(snapshot_path, finished)
            snapshot_write_ms = (time.perf_counter() - write_started) * 1000.0
            snapshot_bytes = snapshot_path.stat().st_size
            self._write_atomic_json(
                self.directory / "session_meta.json",
                {
                    "schema": "direct-online-workmap-session/v1",
                    "session_id": self.session_id,
                    "base_release_id": self.base_release_id,
                    "coordinate_frame_id": self.coordinate_frame_id,
                    **(
                        {"base_bundle_sha256": self.base_bundle_sha256}
                        if self.base_bundle_sha256
                        else {}
                    ),
                    "source_epoch": int(self._source_epoch),
                    "writer_healthy": self._writer_error is None
                    and self._writer_overflow_count == 0,
                    "drain_complete": bool(drained),
                    "snapshot_write_ms": snapshot_write_ms,
                    "snapshot_bytes": int(snapshot_bytes),
                    "finished_unix_s": time.time(),
                },
            )
        if not drained and not self._abort.is_set():
            raise RuntimeError("workmap session did not drain before its shutdown deadline")

    # ---------- writer internals ----------

    @staticmethod
    def _readonly(value: np.ndarray) -> np.ndarray:
        value.setflags(write=False)
        return value

    @staticmethod
    def _image_cells(xy: np.ndarray, width: int, height: int) -> set[int]:
        if not len(xy) or width <= 0 or height <= 0:
            return set()
        x = np.clip((xy[:, 0] * 8.0 / width).astype(int), 0, 7)
        y = np.clip((xy[:, 1] * 6.0 / height).astype(int), 0, 5)
        return set(map(int, y * 8 + x))

    def _enqueue_locked(self, job: dict) -> bool:
        if self._closed or self._writer_error is not None or self._writer_overflow_count:
            return False
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._writer_overflow_count += 1
            self._writer_error = "durable writer queue capacity exceeded"
            self._events_pending.append(
                {
                    "event": "writer_overflow",
                    "capacity": _WRITER_QUEUE_CAPACITY,
                    "revision": int(self._revision),
                    "stamp_unix_s": time.time(),
                }
            )
            return False
        return True

    # Pending references remain in the durable frame catalog; this thread only
    # owns the active submission, so GPU backpressure never holds the writer.
    def _index_dispatch_loop(self) -> None:
        while True:
            with self._index_condition:
                candidate = self._next_index_reference_locked()
                while candidate is None or self._abort.is_set():
                    if self._index_dispatch_stopping or self._abort.is_set():
                        return
                    self._index_condition.wait()
                    candidate = self._next_index_reference_locked()
                name, revision, reference = candidate
                frame = self.keyframes[name]
                frame["index_submitted_revision"] = revision
                self._refresh_index_waiting_locked(frame)
                self._index_active_name = name
                self._index_job_started = float(
                    frame.get("index_requested_mono") or time.monotonic()
                )
            try:
                callback = self._on_durable_reference
                if callback is None:
                    accepted = False
                else:
                    accepted = callback(reference, self.mark_indexed)
                if accepted is False:
                    with self._index_condition:
                        if self._index_dispatch_closing:
                            frame["index_dropped_at_close"] = True
                        else:
                            self.mark_indexed(reference, None, "relocalizer rejected index job")
            except BaseException as error:  # noqa: BLE001 - preserve index health
                self.mark_indexed(reference, None, f"{type(error).__name__}: {error}")
            finally:
                with self._index_condition:
                    self._index_active_name = None
                    self._index_job_started = None
                    self._index_condition.notify_all()

    def _writer_loop(self) -> None:
        while True:
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                try:
                    self._flush_pending_events()
                except BaseException as error:  # noqa: BLE001 - preserve writer health
                    self._set_writer_error(f"{type(error).__name__}: {error}")
                if self._writer_stop.is_set():
                    return
                continue
            if task is None:
                self._writer_stop.set()
                self._queue.task_done()
                return
            with self._lock:
                self._writer_job_queued_mono = float(task["queued_mono"])
            try:
                self._write_task(task)
            except BaseException as error:  # noqa: BLE001 - health must outlive writer errors
                self._set_writer_error(f"{type(error).__name__}: {error}")
                with self._lock:
                    self._events_pending.append(
                        {
                            "event": "writer_error",
                            "error": self._writer_error,
                            "name": task.get("name"),
                            "stamp_unix_s": time.time(),
                        }
                    )
            finally:
                with self._lock:
                    self._writer_job_queued_mono = None
                self._queue.task_done()
            if self._abort.is_set() and self._writer_stop.is_set():
                return  # shutdown deadline: leave queued work, keep the snapshot

    def _write_task(self, task: dict) -> None:
        kind = task["kind"]
        if kind == "overlap_evidence":
            self._write_overlap_task(task)
        elif kind == "ba_update":
            self._write_geometry_update_task(task)
        elif kind == "keyframe":
            self._write_keyframe_task(task)
        else:
            rows = _landmark_observation_rows(
                self.session_id, task["landmarks"], epoch_by_capture=task.get("epoch_by_capture")
            )
            rows.append(
                {
                    "event": "geometry_archive_complete",
                    "name": str(task["name"]),
                    "revision": int(task["revision"]),
                    "ordinal": int(task["ordinal"]),
                    "stamp": float(task["stamp"]),
                }
            )
            self._append_jsonl_rows(self.directory / "observations.jsonl", rows, str_keyed_rows=True)
        self._flush_pending_events()

    def _write_overlap_task(self, task: dict) -> None:
        self._append_jsonl(self.directory / "overlaps.jsonl", task["record"])
        with self._lock:
            self._overlap_durable_revision = max(
                self._overlap_durable_revision, int(task["record"]["overlap_revision"])
            )

    def _write_geometry_update_task(self, task: dict) -> None:
        self._append_jsonl(self.directory / "geometry_updates.jsonl", task["record"])
        with self._lock:
            for name, revision in task.get("reference_revisions", {}).items():
                frame = self.keyframes.get(str(name))
                if frame is not None:
                    self._mark_revision_durable_locked(frame, revision)
            self._index_condition.notify_all()

    def _write_keyframe_image(self, name: str, image: np.ndarray) -> Path:
        _require_free_space(self.images_dir)
        image_path = self.images_dir / f"{name}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError("OpenCV failed to encode workmap keyframe")
        temporary = image_path.with_suffix(".png.tmp")
        with temporary.open("wb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, image_path)
        _fsync_directory(self.images_dir)
        return image_path

    def _write_keyframe_task(self, task: dict) -> None:
        name = str(task["name"])
        image_path = self._write_keyframe_image(name, task["image"])
        rows = _landmark_observation_rows(
            self.session_id, task["landmarks"], epoch_by_capture=task.get("epoch_by_capture")
        )
        rows.extend(_keyframe_observation_rows(self.session_id, task, name, image_path))
        self._append_jsonl_rows(self.directory / "observations.jsonl", rows, str_keyed_rows=True)
        self._append_jsonl(
            self.directory / "keyframes.jsonl",
            {
                "name": name,
                "session_id": self.session_id,
                "revision": int(task["revision"]),
                "ordinal": int(task["ordinal"]),
                "stamp": float(task["stamp"]),
                "source_epoch": int(task["source_epoch"]),
                "image_path": str(image_path),
                "image_size": task["image_size"],
                "cam_from_world": task["pose"],
                "K": task["K"],
                "ids": task["ids"],
                "xy": task["xy"],
                "xyz": task["xyz"],
                "all_ids": task["all_ids"],
                "all_xy": task["all_xy"],
                "all_xyz": task["all_xyz"],
                "landmark_revisions": task["landmark_revisions"],
                "pose_revision": 0,
                "submap_id": int(task.get("submap_id", 0)),
                "durable_complete": True,
            },
        )
        with self._lock:
            frame = self.keyframes[name]
            frame["durable"] = True
            self._mark_revision_durable_locked(frame, int(task["revision"]))
            for reference_name, revision in task.get("reference_revisions", {}).items():
                reference_frame = self.keyframes.get(str(reference_name))
                if reference_frame is not None:
                    self._mark_revision_durable_locked(reference_frame, revision)
            self._index_condition.notify_all()
            self._events_pending.append(
                {
                    "event": "keyframe_durable",
                    "name": name,
                    "revision": int(task["revision"]),
                    "ordinal": int(task["ordinal"]),
                    "image_path": str(image_path),
                    "stamp_unix_s": time.time(),
                }
            )

    def _set_writer_error(self, message: str) -> None:
        with self._lock:
            if self._writer_error is None:
                self._writer_error = str(message)

    def _flush_pending_events(self) -> None:
        """Append queued events; the geometry lock is released before fsync."""
        with self._event_io_lock:
            with self._lock:
                if not self._events_pending:
                    return
                events, self._events_pending = self._events_pending, []
            try:
                self._append_jsonl_rows(
                    self.directory / "events.jsonl", events, session_id=self.session_id
                )
            except BaseException:
                with self._lock:
                    self._events_pending[:0] = events
                raise

    @staticmethod
    def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
        SessionWorkMap._append_jsonl_rows(path, (row,))

    @staticmethod
    def _append_jsonl_rows(
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        session_id: str | None = None,
        str_keyed_rows: bool = False,
    ) -> None:
        # ``str_keyed_rows``: every mapping key is already a str, so ``json.dumps`` can
        # convert NumPy leaves itself and skip the recursive copy made by ``_json_value``.
        with path.open("a", encoding="utf-8") as stream:
            for row in rows:
                data = {"session_id": session_id, **row} if session_id is not None else row
                encoded = (
                    json.dumps(
                        data, default=_json_value, sort_keys=True, separators=(",", ":"),
                        allow_nan=False,
                    )
                    if str_keyed_rows
                    else json.dumps(
                        _json_value(data), sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                )
                stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        # Snapshot collections are already JSON-native; convert only NumPy/Path leaves.
        encoded = json.dumps(
            value, default=_json_value, separators=(",", ":"), sort_keys=True, allow_nan=False
        )
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)

    @staticmethod
    def _landmark_json(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "xyz": value["xyz"],
            "revision": int(value.get("revision", 0)),
            "origin": str(value.get("origin", "triangulated")),
            "parallax_deg": float(value.get("parallax_deg", 0.0)),
            "valid_observation_count": int(value.get("valid_observation_count", 2)),
            "observations": value.get("observations", ()),
            "last_inlier": value.get("last_inlier"),
            "rgb": value.get("rgb"),
            "submap_id": int(value.get("submap_id", 0)),
        }


class OnlineWorkMapTracker(TwoRateTracker):
    """Production direct tracker with a persistent, session-only revisit map."""

    #: False after a mapping-side failure: the tracker keeps localizing, nothing new is mapped.
    mapping_enabled: bool = True
    mapping_error: str | None = None
    _mapping_state: MappingState = MappingState.ANCHORED_EXPAND
    _frozen_ratio: float = 0.0

    def __init__(
        self,
        assets: "DirectMapAssets",
        profile: "DirectProfile",
        provider: Any,
        *,
        workmap_session_dir: str | Path,
        workmap_eval_stage: str = "full",
        workmap_session_id: str | None = None,
    ) -> None:
        if workmap_eval_stage not in {"reuse", "ba", "full"}:
            raise ValueError("OnlineWorkMapTracker requires a workmap-enabled evaluation stage")
        if workmap_eval_stage != "full" and os.environ.get("SFM_EVALUATION_ONLY") != "1":
            raise RuntimeError("non-full workmap evaluation stages require SFM_EVALUATION_ONLY=1")
        if not bool(getattr(getattr(profile, "online_workmap", None), "enabled", False)):
            raise ValueError("OnlineWorkMapTracker requires online_workmap.enabled=true")
        if workmap_session_id is None:
            resolved = Path(workmap_session_dir).expanduser().resolve()
            workmap_session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(resolved)))
        self.workmap_session_dir = Path(workmap_session_dir).expanduser().resolve()
        self.workmap_session_id = str(workmap_session_id)
        self.workmap_eval_stage = str(workmap_eval_stage)
        self._ba_optimizer = getattr(profile.online_workmap, "optimizer", "local_ba")
        self._graph_generation = 0
        self._graph_stats: dict[str, Any] = {}
        if self._ba_optimizer == "isam2" and workmap_eval_stage in {"ba", "full"}:
            from workmap_graph import require_gtsam

            require_gtsam()
        expansion = getattr(profile.online_workmap, "expansion", None)
        self._expansion_gate = bool(getattr(expansion, "frozen_region_gate", False))
        self._expansion_query = bool(getattr(expansion, "spatiotemporal_query", False))
        self._expansion_depth = bool(getattr(expansion, "depth_seeds", False))
        depth_estimator = self._make_depth_estimator()
        super().__init__(
            assets, profile, provider, online_workmap_active=True, depth_estimator=depth_estimator
        )
        self._depth_estimator = depth_estimator
        self._init_expansion_state()
        base_release_id = str(getattr(assets, "map_revision_id", "unknown-base-release"))
        coordinate_frame_id = str(
            getattr(assets, "coordinate_frame_id", "unknown-coordinate-frame")
        )
        base_root = getattr(assets, "root", None)
        self.session_workmap = SessionWorkMap(
            self.workmap_session_dir,
            session_id=self.workmap_session_id,
            base_release_id=base_release_id,
            coordinate_frame_id=coordinate_frame_id,
            base_release_root=base_root,
            base_bundle_sha256=str(assets.sha256),
            on_durable_reference=self._submit_index_reference,
            min_reference_points=max(
                int(profile.reloc.min_points),
                int(profile.fast_loop.pnp.min_points),
                int(profile.fast_loop.pnp.min_inliers),
            ),
        )
        self.workmap = self.session_workmap
        self._workmap_points_used = 0
        self._workmap_reloc_accepted = 0
        self._workmap_index_healthy = True
        self._workmap_ba_submitted = 0
        self._workmap_ba_applied = 0
        self._workmap_ba_stale = 0
        self._workmap_scale_applied = 0
        self._ba_pending_use_ids: set[int] = set()
        self._ba_executor = (
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_ba_process_initialize,
                initargs=(os.getpid(),),
            )
            if self.workmap_eval_stage in {"ba", "full"}
            else None
        )
        self._ba_ready: Future | None = None
        self._fast_close = threading.Event()
        self._closing = False
        self._ba_future: Future | None = None
        self._ba_payload: dict[str, Any] | None = None
        self._ba_pending = False
        self._ba_started_mono: float | None = None
        self._ba_pending_mono: float | None = None
        self._ba_error: str | None = None
        self._workmap_ba_rejected = 0
        self._overlap_future: Future | None = None
        self._overlap_payload: dict[str, Any] | None = None
        self._overlap_started_mono: float | None = None
        self._overlap_pending = False
        self._overlap_pending_mono: float | None = None
        self._overlap_last_submitted_version = 0
        self._overlap_attempted: dict[str, int] = {}
        self._overlap_retry_lock = threading.Lock()
        self._overlap_retry_ref: Mapping[str, Any] | None = None
        self._overlap_retry_started_mono: float | None = None
        self._overlap_superseded = 0
        thresholds = profile.reloc.frozen_pnp_thresholds
        self._overlap_strong_old_inliers = int(thresholds["strong_inliers"])
        self._overlap_max_old_p90_px = float(thresholds["maximum_reprojection_p90_px"])
        self._overlap_match_spacing_s = float(profile.reloc.period_s)
        self._overlap_last_match_submitted_mono: float | None = None
        self._overlap_error: str | None = None
        self._workmap_overlap_stale = 0
        self._workmap_overlap_rejected = 0
        self._scale_pending_use_ids: set[int] = set()
        self._submaps = SubmapRegistry()
        self._submap_archive: list[dict[str, Any]] = []
        self._reloc_started: dict[int, int] = {}
        self._reloc_first_anchor: dict[int, int] = {}
        self._merge_failures: dict[int, int] = {}
        self._merge_residual: dict[int, float] = {}
        self._merge_context: dict[int, tuple[Sim3, list[AnchorPair], float]] = {}
        self._merge_ready: dict[int, Sim3] = {}
        self._graph_edges: list[dict[str, Any]] = []
        self._graph_future: Future | None = None
        self._graph_payload: dict[str, Any] | None = None
        self._graph_pending = False
        self._graph_error: str | None = None
        self._graph_solves = 0
        self._graph_applied = 0
        self._unmerged_dirty = False
        self._last_stitch_s: float | None = None
        self._anchor_rows = self._anchor_local_inliers = -1
        self._graph_solver = str(
            getattr(profile.online_workmap, "submap_graph_solver", "batch")
        )
        self._presubmap_submap_id: int | None = None
        self._presubmap = self._new_presubmap()
        self._presubmap_pending: dict[str, Any] | None = None
        self._presubmap_merged = 0
        self._presubmap_error: str | None = None
        self._presubmap_had_strong = False
        self._presubmap_no_pose_frames = 0
        self._presubmap_alignment_future: Future | None = None
        self._presubmap_alignment_result = None
        self._presubmap_alignment_epoch = 0
        self._trajectory_backfill: dict[str, Any] | None = None
        self._backfill_repeats = 0
        self._backfill_next_ordinal = 0
        self._presubmap_ms = 0.0
        self._workmap_merge_submitted = 0
        self._workmap_merge_committed = 0
        self._workmap_merge_rejected = 0
        self._workmap_merge_noop = 0
        self._workmap_landmarks_fused = 0
        self._workmap_merge_scale: float | None = None
        self._stitch_future: Future | None = None
        self._stitch_snapshot: dict[str, Any] | None = None
        self._stitch_sim: Sim3 | None = None
        self._stitch_started_mono: float | None = None
        self._mapping = MappingStateMachine()

    def _make_depth_estimator(self):
        if not self._expansion_depth:
            return None
        from workmap_depth import MoGeDepthEstimator, require_moge

        require_moge()
        return MoGeDepthEstimator(direct_paths.resolve_moge_checkpoint())

    def _init_expansion_state(self) -> None:
        self._coverage: FrozenCoverage | None = None
        self._query = SpatioTemporalQuery(self._w, self._h) if self._expansion_query else None
        self._last_map_inliers = 0
        self._query_added_ids: set[int] = set()
        self._query_frame = self._empty_query_frame()
        self._query_added_total = 0
        self._query_inliers_total = 0
        self._workmap_frozen_gate_rejected = 0
        self._frame_rgb = None
        self._depth_results = queue.SimpleQueue()
        self._depth_ready: dict[int, dict] = {}
        self._depth_scales = deque(maxlen=DEPTH_SCALE_HISTORY)
        self._last_depth_submit_stamp: float | None = None
        self._depth_counts = dict.fromkeys(
            ("submitted", "completed", "rejected", "superseded", "expired", "promoted"), 0
        )
        self._depth_last_scale = None
        self._depth_last_infer_ms = None

    @staticmethod
    def _empty_query_frame() -> dict:
        return {
            "workmap_query_candidates": 0,
            "workmap_query_added": 0,
            "workmap_query_inliers": 0,
            "workmap_query_ms": 0.0,
        }

    def _admit_new_landmarks(self, world: np.ndarray, good: np.ndarray) -> np.ndarray:
        if (
            not self.mapping_enabled
            or self._mapping_state is MappingState.LOCALIZE
            or self._submap_busy()
        ):
            return np.zeros_like(good)
        if not self._expansion_gate:
            return good
        if self._coverage is None:
            raise RuntimeError(
                "online_workmap.expansion.frozen_region_gate needs ensure_models() "
                "to build frozen coverage"
            )
        covered = self._coverage.covered(world)
        self._workmap_frozen_gate_rejected += int(np.count_nonzero(good & covered))
        return good & ~covered

    def _query_due(self) -> bool:
        return (
            self._query is not None
            and self._last_pose is not None
            and self._last_pose_ordinal == self._ordinal - 1
            and self._last_pose_status in {"FAST_TRACK", "RELOC_SEED", "VO_ONLY", "DEAD_RECKON"}
            and self._last_map_inliers
            < int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
            and self._ordinal % QUERY_STRIDE == 0
        )

    def _augment_live_points(self, gray: np.ndarray, reseeded: bool) -> None:
        self._query_frame = self._empty_query_frame()
        self._query_added_ids = set()
        if reseeded or self._mapping_state is MappingState.LOCALIZE or not self._query_due():
            return
        started = time.perf_counter()
        occupied = (
            np.concatenate([self._live_xy, self._vo_xy]) if self._vo_xy.size else self._live_xy
        )
        cands = self._query.candidates(
            self._last_pose, self._K, self._live_ids, occupied, int(self.profile.vo.min_distance)
        )
        ids, xy, xyz = self._track_query_groups(gray, cands)
        if len(ids):
            self._live_ids = np.concatenate([self._live_ids, ids])
            self._live_xy = np.ascontiguousarray(
                np.concatenate([self._live_xy, xy]), dtype=np.float64
            )
            self._live_xyz = np.concatenate([self._live_xyz, xyz])
            self._query_added_ids = set(map(int, ids))
        self._query_added_total += len(ids)
        self._query_frame.update(
            workmap_query_candidates=len(cands.ids),
            workmap_query_added=len(ids),
            workmap_query_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _track_query_groups(
        self, gray: np.ndarray, cands: QueryCandidates
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        groups: dict[str, list[int]] = {}
        for row, name in enumerate(cands.source_names):
            groups.setdefault(name, []).append(row)
        names = sorted(groups, key=lambda name: (len(groups[name]), name), reverse=True)
        selected, measured = [], []
        for name in names[:QUERY_MAX_SOURCE_KEYFRAMES]:
            rows = np.asarray(groups[name], dtype=np.int64)
            source = self._query.image(name)
            if source is None:
                continue
            fwd, keep = self._tracker.track_guided(
                source, gray, cands.source_xy[rows], cands.predicted_xy[rows]
            )
            keep &= self._inside(fwd)
            keep &= np.linalg.norm(fwd - cands.predicted_xy[rows], axis=1) <= QUERY_MAX_SHIFT_PX
            selected.extend(rows[keep])
            measured.extend(fwd[keep])
        rows = np.asarray(selected, dtype=np.int64)
        return (
            cands.ids[rows],
            np.asarray(measured, dtype=np.float64).reshape(-1, 2),
            cands.xyz[rows],
        )

    def _note_expansion_frame(self, info: dict) -> None:
        ok = bool(info.get("ok"))
        self._last_map_inliers = int(info.get("map_inliers", 0)) if ok else 0
        if self._query is not None:
            if info.get("status") in {"FAST_TRACK", "VO_ONLY", "RELOC_SEED"}:
                self._query.note_frame(self._workmap_pose_ids)
            inliers = len(self._query_added_ids & self._workmap_pose_ids) if ok else 0
            self._query_frame["workmap_query_inliers"] = inliers
            self._query_inliers_total += inliers

    def _expansion_diagnostics(self) -> dict:
        return {
            "workmap_frozen_cells": self._coverage.cell_count if self._coverage is not None else 0,
            "workmap_frozen_gate_rejected": self._workmap_frozen_gate_rejected,
            **self._query_frame,
            "workmap_query_added_total": self._query_added_total,
            "workmap_query_inliers_total": self._query_inliers_total,
            **{f"workmap_depth_{key}": value for key, value in self._depth_counts.items()},
            "workmap_depth_last_scale": self._depth_last_scale,
            "workmap_depth_last_infer_ms": self._depth_last_infer_ms,
        }

    def _sample_rgb(self, xy: np.ndarray) -> np.ndarray | None:
        """3x3 mean of the offered RGB capture at fast-loop pixels (uint8, n x 3)."""
        frame = self._frame_rgb
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not len(xy):
            return None
        height, width = frame.shape[:2]
        columns = np.rint(np.asarray(xy, dtype=float)[:, 0] * (width / self._w)).astype(int)
        rows = np.rint(np.asarray(xy, dtype=float)[:, 1] * (height / self._h)).astype(int)
        total = np.zeros((len(xy), 3), dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                total += frame[np.clip(rows + dy, 0, height - 1), np.clip(columns + dx, 0, width - 1)]
        return np.clip(np.rint(total / 9.0), 0, 255).astype(np.uint8)

    def offer_rgb_frame(self, frame: np.ndarray | None) -> None:
        """Borrow the capture only until this step; queued depth owns its resized copy."""
        self._frame_rgb = frame

    def _depth_submit_due(self) -> bool:
        frame = self._frame_rgb
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            return False
        return (
            self._last_depth_submit_stamp is None
            or self._previous_capture_stamp - self._last_depth_submit_stamp >= DEPTH_MIN_INTERVAL_S
        )

    def _submit_depth_seeds(self, bid: int, cam_from_world: np.ndarray) -> None:
        if self._closing:
            return
        if not self._depth_submit_due():
            return
        frame = self._frame_rgb
        rgb = (
            frame.copy()
            if frame.shape[:2] == (self._h, self._w)
            else cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_AREA)
        )
        request = {
            "source_epoch": self._source_epoch,
            "bid": int(bid),
            "rgb": rgb,
            "K": self._K.copy(),
            "cam_from_world": cam_from_world.copy(),
            "seed_xy": self._vo_seed_xy[self._vo_bid == bid].copy(),
            "ref_ids": self._live_ids[
                (self._live_ids > 0) | np.isin(self._live_ids, list(self._workmap_landmarks.keys()))
            ].copy(),
            "ref_xy": self._live_xy[
                (self._live_ids > 0) | np.isin(self._live_ids, list(self._workmap_landmarks.keys()))
            ].copy(),
            "ref_xyz": self._live_xyz[
                (self._live_ids > 0) | np.isin(self._live_ids, list(self._workmap_landmarks.keys()))
            ].copy(),
        }
        if self._worker.submit_depth(request, self._on_depth_result):
            self._depth_counts["submitted"] += 1
            self._last_depth_submit_stamp = self._previous_capture_stamp

    def _on_depth_result(self, request, result, error) -> None:
        self._depth_results.put((int(request["source_epoch"]), int(request["bid"]), result, error))

    def _drain_depth_results(self) -> None:
        while True:
            try:
                epoch, bid, result, error = self._depth_results.get_nowait()
            except queue.Empty:
                return
            if error == "superseded":
                self._depth_counts["superseded"] += 1
                continue
            if error is not None or epoch != self._source_epoch:
                continue
            if not result["ok"] or not self._depth_scale_consistent(result["scale"]):
                self._depth_counts["rejected"] += 1
                continue
            self._depth_ready[bid] = result
            self._depth_counts["completed"] += 1
            self._depth_last_scale = result["scale"]
            self._depth_last_infer_ms = result["infer_ms"]

    def _depth_scale_consistent(self, scale: float) -> bool:
        if len(self._depth_scales) >= DEPTH_SCALE_HISTORY_MIN:
            median = float(np.median(self._depth_scales))
            if max(scale / median, median / scale) > DEPTH_SCALE_MAX_RATIO:
                return False
        self._depth_scales.append(scale)
        return True

    def _promote_depth_seeds(self, cam_from_world: np.ndarray) -> int:
        self._drain_depth_results()
        promoted = sum(
            self._promote_depth_batch(bid, result, cam_from_world)
            for bid, result in self._depth_ready.items()
        )
        self._depth_ready.clear()
        return promoted

    def _promote_depth_batch(self, bid: int, result: dict, cam_from_world: np.ndarray) -> int:
        rows = np.flatnonzero(self._vo_bid == bid)
        if not len(rows) or bid not in self._vo_seed_meta:
            self._depth_counts["expired"] += 1
            return 0
        meta = self._get_seed_provenance(bid)
        if meta is None or meta.source_epoch != self._source_epoch or not meta.admissible:
            self._depth_counts["expired"] += 1
            return 0
        if not self._is_pose_admissible(cam_from_world):
            self._depth_counts["rejected"] += 1
            return 0
        mask, xyz = match_promotable_rows(
            result,
            self._vo_seed_xy[rows],
            self._vo_xy[rows],
            cam_from_world,
            self._K,
            float(self.profile.vo.max_reproj_px),
        )
        mask = self._admit_new_landmarks(xyz, mask)
        take = rows[mask]
        count = len(take)
        if not count:
            return 0
        ids = np.arange(self._vo_next_id, self._vo_next_id - count, -1, dtype=np.int64)
        self._vo_next_id -= count
        self._register_vo_landmarks(
            bid, ids, xyz[mask], self._vo_seed_xy[take], self._vo_xy[take], cam_from_world
        )
        for key in ids:
            self._workmap_landmarks[int(key)]["origin"] = "depth"
        self._live_ids = np.concatenate([self._live_ids, ids])
        self._live_xy = np.ascontiguousarray(
            np.concatenate([self._live_xy, self._vo_xy[take]]), dtype=np.float64
        )
        self._live_xyz = np.concatenate([self._live_xyz, xyz[mask]])
        keep = np.ones(len(self._vo_bid), dtype=bool)
        keep[take] = False
        self._vo_xy = self._vo_xy[keep]
        self._vo_seed_xy = self._vo_seed_xy[keep]
        self._vo_bid = self._vo_bid[keep]
        self._depth_counts["promoted"] += count
        return count

    def ensure_models(self) -> None:
        super().ensure_models()
        if self._ba_executor is not None:
            self._ensure_ba_process().result()
        if self._expansion_gate and self._coverage is None:
            self._coverage = FrozenCoverage(self.provider.frozen_point_xyz())
        if self._depth_estimator is not None:
            self._depth_estimator.load()

    def _ensure_ba_process(self) -> Future:
        """Spawn the single solver process once, inside its thread limits."""
        assert self._ba_executor is not None
        if self._ba_ready is None:
            with _ba_spawn_environment():
                self._ba_ready = self._ba_executor.submit(_ba_process_ready, self._ba_optimizer)
        return self._ba_ready

    def _submit_ba_if_ready(self) -> None:
        if self._closing:
            return
        if self._ba_executor is None or self._ba_error is not None:
            return
        if self._ba_future is not None or self._stitch_future is not None:
            # A stitch in flight owns the keyframes it will move; BA waits for it.
            self._ba_pending = True
            if self._ba_pending_mono is None:
                self._ba_pending_mono = time.monotonic()
            return
        payload = self.session_workmap.ba_snapshot(
            self._source_epoch,
            self._K,
            incremental=self._ba_optimizer == "isam2",
            anchor_min_inliers=int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"]),
        )
        if payload is None:
            return
        self._submit_ba_payload(payload)

    def _submit_ba_payload(self, payload: dict[str, Any]) -> None:
        assert self._ba_executor is not None
        solver = solve_ba_snapshot
        if self._ba_optimizer == "isam2":
            from workmap_graph import solve_graph_snapshot

            solver = solve_graph_snapshot
            payload = {**payload, "solver_generation": self._graph_generation}
        try:
            self._ensure_ba_process()
            self._ba_future = self._ba_executor.submit(solver, payload)
        except Exception as error:
            self._ba_error = f"{type(error).__name__}: {error}"
            self._workmap_ba_rejected += 1
            return
        self._ba_payload = payload
        self._ba_started_mono = time.monotonic()
        self._workmap_ba_submitted += 1

    def _poll_ba(self) -> bool:
        future = self._ba_future
        if future is None or not future.done():
            return False
        if self.session_workmap.writer_healthy and not self.session_workmap.geometry_commit_ready:
            return False
        payload = self._ba_payload
        submitted_at = self._ba_started_mono
        self._ba_future = None
        self._ba_payload = None
        self._ba_started_mono = None
        applied = False
        try:
            candidate = dict(future.result())
            self._graph_stats = {
                f"workmap_{key}": value
                for key, value in candidate.items()
                if key.startswith("graph_") or key == "solver_ms"
            }
            candidate["submit_to_apply_ms"] = (
                max(0.0, (time.monotonic() - submitted_at) * 1000.0)
                if submitted_at is not None
                else None
            )
        except Exception as error:
            self._ba_error = f"{type(error).__name__}: {error}"
            self._workmap_ba_rejected += 1
        else:
            applied = self._apply_ba_result(payload, candidate)
        if self._ba_pending and self._ba_future is None and self.workmap_eval_stage != "full":
            self._ba_pending = False
            self._ba_pending_mono = None
            self._submit_ba_if_ready()
        return applied

    def _apply_ba_result(self, payload: dict[str, Any] | None, candidate: dict[str, Any]) -> bool:
        if payload is None or int(payload["source_epoch"]) != self._source_epoch:
            self._workmap_ba_stale += 1
            self.session_workmap.record_ba_rejection("source_epoch", payload)
            return False
        if not candidate.get("ok"):
            self._workmap_ba_rejected += 1
            self.session_workmap.record_ba_rejection(
                str(candidate.get("reason", "solver_rejected")), payload,
                candidate.get("submit_to_apply_ms"),
            )
            return False
        if not self.session_workmap.apply_ba_candidate(payload, candidate):
            reason = self.session_workmap.last_ba_rejection_reason or "unknown_rejection"
            if reason in _STALE_REJECTIONS:
                self._workmap_ba_stale += 1
            else:
                self._workmap_ba_rejected += 1
            self.session_workmap.record_ba_rejection(reason, payload, candidate.get("submit_to_apply_ms"))
            if (
                reason in {"old_support_count", "old_reprojection_p90"}
                and not payload.get("freeze_poses", False)
                and self._ba_executor is not None
            ):
                self._submit_ba_payload({**payload, "freeze_poses": True})
            return False
        update_ids = {int(key) for key in candidate["updates"]}
        self._ba_pending_use_ids.update(update_ids)
        self._refresh_live_points(update_ids)
        # Mature features must triangulate from the same corrected camera poses as
        # the session landmarks. Only exact captured keyframes have a BA correction.
        seed_poses = {
            int(frame["ordinal"]): np.asarray(candidate["poses"][frame["name"]], dtype=np.float64)
            for frame in payload["keyframes"]
        }
        for bid, meta in self._vo_seed_meta.items():
            pose = seed_poses.get(int(meta[0]))
            if bid in self._vo_pose_by_bid and pose is not None:
                self._vo_pose_by_bid[bid] = pose.copy()
        for frame in payload["keyframes"]:
            if self._last_pose_ordinal == int(frame["ordinal"]):
                pose = np.asarray(candidate["poses"][frame["name"]], dtype=np.float64)
                self._last_pose = pose.copy()
                self._prev_center = -pose[:, :3].T @ pose[:, 3]
        self._workmap_ba_applied += 1
        return True

    def _refresh_live_points(self, update_ids: set[int]) -> None:
        """Carry committed session geometry into the tracker's working copies."""
        for key in update_ids:
            self._workmap_landmarks[key] = _copy_landmark(self.session_workmap.landmarks[key])
        if len(self._live_ids) and update_ids:
            refreshed = np.isin(
                self._live_ids, np.fromiter(update_ids, dtype=np.int64, count=len(update_ids))
            )
            for row in np.flatnonzero(refreshed):
                self._live_xyz[row] = self.session_workmap.landmarks[int(self._live_ids[row])]["xyz"]
        if self._query is not None and update_ids:
            ids = sorted(update_ids)
            self._query.upsert_landmarks(
                ids, np.array([self._workmap_landmarks[k]["xyz"] for k in ids])
            )

    def _submit_overlap_if_ready(self) -> None:
        if self._closing:
            return
        if self.workmap_eval_stage != "full" or self._ba_executor is None or self._overlap_error:
            return
        version = self.session_workmap.overlap_revision
        if version <= self._overlap_last_submitted_version:
            return
        if self._overlap_future is not None:
            self._overlap_pending = True
            if self._overlap_pending_mono is None:
                self._overlap_pending_mono = time.monotonic()
            return
        snapshot = self.session_workmap.overlap_snapshot()
        if snapshot is None:
            return
        try:
            self._ensure_ba_process()
            future = self._ba_executor.submit(_fit_overlap_job, snapshot)
        except Exception as error:
            self._overlap_error = f"{type(error).__name__}: {error}"
            return
        self._overlap_future = future
        self._overlap_payload = snapshot
        self._overlap_started_mono = time.monotonic()
        self._overlap_last_submitted_version = int(snapshot["overlap_revision"])

    def _poll_overlap(self) -> bool:
        future = self._overlap_future
        if future is None or not future.done():
            return False
        if self.session_workmap.writer_healthy and not self.session_workmap.geometry_commit_ready:
            return False
        snapshot = self._overlap_payload
        self._overlap_future = None
        self._overlap_payload = None
        self._overlap_started_mono = None
        applied = False
        try:
            candidate = dict(future.result())
        except Exception as error:
            self._overlap_error = f"{type(error).__name__}: {error}"
            self._workmap_overlap_rejected += 1
            self.session_workmap.record_scale_rejection(self._overlap_error, snapshot)
        else:
            applied = self._apply_overlap_result(snapshot, candidate)
        if self._overlap_pending and self._overlap_future is None:
            self._overlap_pending = False
            self._overlap_pending_mono = None
            self._submit_overlap_if_ready()
        return applied

    def _apply_overlap_result(
        self, snapshot: dict[str, Any] | None, candidate: dict[str, Any]
    ) -> bool:
        if snapshot is None or int(snapshot["source_epoch"]) != self._source_epoch:
            self._workmap_overlap_stale += 1
            self.session_workmap.record_scale_rejection("source_epoch", snapshot)
            return False
        if not candidate.get("ok"):
            self._workmap_overlap_rejected += 1
            self.session_workmap.record_scale_rejection(
                str(candidate.get("reason", "insufficient_overlap")), snapshot
            )
            return False
        owners = {
            self._submaps.submap_of(int(key))
            for key in (*candidate["fit_new_ids"], *candidate["holdout_new_ids"])
        }
        detached = {
            owner
            for owner in owners - {0}
            if self._submaps.get(owner)["state"] != "merged"
        }
        if detached:
            return self._route_overlap_to_graph(snapshot, candidate, owners)
        update_ids = self.session_workmap.apply_scale_candidate(snapshot, candidate)
        if not update_ids:
            reason = self.session_workmap.last_scale_rejection_reason or "unknown_rejection"
            if reason in _STALE_REJECTIONS:
                self._workmap_overlap_stale += 1
            else:
                self._workmap_overlap_rejected += 1
            self.session_workmap.record_scale_rejection(reason, snapshot)
            return False
        self._scale_pending_use_ids.update(update_ids)
        self._refresh_live_points(update_ids)
        scale = float(candidate["scale"])
        rotation = np.asarray(candidate["rotation"], dtype=np.float64)
        translation = np.asarray(candidate["translation"], dtype=np.float64)
        connected_frames = self.session_workmap.frames_observing(update_ids)
        for bid, meta in self._vo_seed_meta.items():
            ordinal = meta[0]
            seed_name = self.session_workmap._keyframe_by_capture.get(
                (self._source_epoch, int(ordinal))
            )
            if seed_name in connected_frames and bid in self._vo_pose_by_bid:
                self._vo_pose_by_bid[bid] = _transform_pose_sim3(
                    self._vo_pose_by_bid[bid], scale, rotation, translation
                )
        self._rescale_live_prior(update_ids, scale, rotation, translation)
        self._workmap_scale_applied += 1
        self._graph_generation += 1
        self._submit_ba_if_ready()
        return True

    def _route_overlap_to_graph(
        self, snapshot: dict[str, Any] | None, candidate: dict[str, Any], owners: set[int]
    ) -> bool:
        """Overlap evidence for a not-yet-merged submap becomes a graph edge, not a commit."""
        if len(owners) != 1:
            self._workmap_overlap_rejected += 1
            self.session_workmap.record_scale_rejection("mixed_submaps", snapshot)
            return False
        submap_id = next(iter(owners))
        applied = self._submaps.get(submap_id)["applied"] or Sim3.identity()
        fit = Sim3(
            float(candidate["scale"]),
            np.asarray(candidate["rotation"], dtype=np.float64),
            np.asarray(candidate["translation"], dtype=np.float64),
        )
        if not self._add_graph_edge(
            0,
            submap_id,
            "overlap",
            fit.compose(applied),
            _graph_sigma("overlap", float(candidate.get("after_3d_p90") or 0.0)),
        ):
            self._workmap_overlap_rejected += 1
            self.session_workmap.record_scale_rejection("graph_unavailable", snapshot)
            return False
        return False                    # accepted as an edge; nothing was applied this frame

    def _mixed_pnp_rows(self) -> np.ndarray | None:
        """While a submap awaits confirmation its points stay out of the pose solve."""
        submap_id = self._submaps.awaiting_id
        if submap_id is None:
            return None
        return np.flatnonzero(~np.isin(self._live_ids, self._submaps.landmark_ids(submap_id)))

    def _retained_rows(self, track_mask: np.ndarray) -> np.ndarray:
        submap_id = self._submaps.awaiting_id
        if submap_id is None:
            return super()._retained_rows(track_mask)
        return np.isin(self._live_ids, self._submaps.landmark_ids(submap_id))

    def _anchors_wanted(self) -> bool:
        """Anchor pairs need the map-only pose: a detached submap awaits, or the pre-anchor
        submap takes the published pose as its map anchor."""
        registry = self._submaps
        presubmap = self._presubmap
        return self.mapping_enabled and (
            registry.active_id != 0
            or registry.relocalizing_id is not None
            or (presubmap is not None and presubmap.active)
        )

    def _collect_submap_anchor(self, old_only: Any, estimate: Any, args: tuple) -> None:
        """A strong frozen-map solve while a detached submap is open or waiting.

        The first one starts relocalization of the open submap. Every one pairs the
        GlueMap pose with a submap-only pose of the same frame; once enough consistent
        pairs exist the submap is merged. Nothing is applied here.
        """
        if (
            not self.mapping_enabled
            or self.workmap_eval_stage != "full"
            or self._ba_executor is None
        ):
            return
        registry = self._submaps
        if registry.relocalizing_id is None:
            submap_id = registry.active_id
            if submap_id == 0 or registry.get(submap_id)["state"] != "active":
                return
            parent = registry.get(submap_id)["parent"] or 0
            registry.set_state(submap_id, "relocalizing")
            registry.set_active(parent)
            self._reloc_started[submap_id] = int(self._ordinal)
            self.session_workmap.note_event(
                {"event": "submap_relocalizing", "submap_id": submap_id, "ordinal": int(self._ordinal)}
            )
            self._publish_submaps()
        submap_id = int(registry.relocalizing_id)
        rows = np.flatnonzero(np.isin(self._live_ids, registry.landmark_ids(submap_id)))
        self._anchor_rows = int(len(rows))
        if len(rows) < ANCHOR_MIN_LOCAL_INLIERS:
            return
        local = estimate(self._live_xy[rows], self._live_xyz[rows], *args)
        if local is None:
            return
        mask = np.asarray(local["inlier_mask"], dtype=bool)
        self._anchor_local_inliers = int(mask.sum())
        if int(mask.sum()) < ANCHOR_MIN_LOCAL_INLIERS:
            return
        registry.add_anchor(
            submap_id,
            AnchorPair(
                int(self._ordinal),
                np.asarray(old_only["cam_from_world"].matrix(), dtype=np.float64)[:3, :4],
                np.asarray(local["cam_from_world"].matrix(), dtype=np.float64)[:3, :4],
                self._live_ids[rows][mask].copy(),
                self._live_xy[rows][mask].copy(),
            ),
        )
        self._reloc_first_anchor.setdefault(submap_id, int(self._ordinal))
        self._try_merge_submap(submap_id)

    def _expire_relocalization(self) -> None:
        """No consistent anchors within ANCHOR_MAX_SPAN frames of the first one: give up."""
        submap_id = self._submaps.relocalizing_id
        if submap_id is None:
            return
        start = self._reloc_first_anchor.get(
            submap_id, self._reloc_started.get(submap_id, int(self._ordinal))
        )
        if int(self._ordinal) - start > ANCHOR_MAX_SPAN:
            self._mark_unmerged(submap_id, "anchor_timeout")

    def _try_merge_submap(self, submap_id: int) -> None:
        registry = self._submaps
        anchors = registry.anchors(submap_id)
        if len(anchors) < ANCHOR_MIN_COUNT:
            return
        window = [a for a in anchors if a.ordinal >= anchors[-1].ordinal - ANCHOR_MAX_SPAN]
        if len(window) < ANCHOR_MIN_COUNT:
            return
        sim, residual = fit_anchor_similarity(window)
        reason = anchor_consistency(window, sim)
        if reason == "insufficient_anchors":
            return
        if reason is not None:
            registry.drop_oldest_anchor(submap_id)
            self.session_workmap.note_event(
                {
                    "event": "submap_anchor_rejected",
                    "submap_id": submap_id,
                    "reason": reason,
                    "anchors": len(window),
                    "residual_u": float(residual),
                    "scale": float(sim.scale),
                    "rotation_deg": float(sim.rotation_deg()),
                    "ordinal": int(self._ordinal),
                }
            )
            return
        # Consistent anchors become a merge edge; the solved Sim(3) starts the stitch.
        registry.set_state(submap_id, "merging")
        self._merge_context[submap_id] = (sim, window, float(residual))
        self._publish_submaps()
        if not self._add_graph_edge(0, submap_id, "merge", sim, _graph_sigma("merge", residual)):
            self._merge_ready[submap_id] = sim

    def _advance_pending_merges(self) -> None:
        """Start the stitch of a merging submap once its graph result (or fallback) is in."""
        if self._stitch_future is not None:
            return
        for submap_id in list(self._merge_ready):
            sim = self._merge_ready.pop(submap_id)
            context = self._merge_context.pop(submap_id, None)
            if (
                context is None
                or submap_id not in self._submaps
                or self._submaps.get(submap_id)["state"] != "merging"
            ):
                continue
            self._submit_merge_stitch(submap_id, sim, context[1], context[2])
            if self._stitch_future is not None:
                return

    def _submit_merge_stitch(
        self, submap_id: int, sim: Sim3, window: list[AnchorPair], residual: float
    ) -> None:
        registry = self._submaps
        strong = int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
        snapshot = self.session_workmap.merge_snapshot(
            submap_id,
            sim,
            window,
            source_epoch=self._source_epoch,
            K=self._K,
            anchor_min_old=strong,
        )
        if snapshot is None:
            self._merge_rejected(
                submap_id, self.session_workmap.last_loop_rejection_reason or "merge_snapshot"
            )
            self._mark_unmerged(submap_id, "merge_snapshot")
            return
        self._merge_residual[submap_id] = float(residual)
        if (
            abs(math.log(sim.scale)) < 0.01
            and snapshot["angle_deg"] < LOOP_MIN_ANGLE_DEG
            and snapshot["drift_u"] < LOOP_MIN_DRIFT_U
        ):
            registry.set_applied(submap_id, sim)
            registry.set_state(submap_id, "merged")
            self._workmap_merge_noop += 1
            self.session_workmap.note_event(
                {
                    "event": "submap_merged",
                    "submap_id": submap_id,
                    "noop": True,
                    "scale": float(sim.scale),
                    "anchors": len(window),
                    "residual_u": float(residual),
                }
            )
            self._publish_submaps()
            return
        try:
            self._ensure_ba_process()
            self._stitch_future = self._ba_executor.submit(solve_stitch_snapshot, snapshot)
        except Exception as error:
            self._ba_error = f"{type(error).__name__}: {error}"
            self._merge_rejected(submap_id, "submit_failed")
            return
        self._stitch_snapshot = snapshot
        self._stitch_sim = sim
        self._stitch_started_mono = time.monotonic()
        self._workmap_merge_submitted += 1

    def _add_graph_edge(
        self, a: int, b: int, kind: str, z: Sim3, sigma: list[float]
    ) -> bool:
        """Queue one submap-graph edge; False when the graph layer cannot run."""
        if self._ba_executor is None or self._graph_error is not None or not self.mapping_enabled:
            return False
        self._graph_edges.append(
            {
                "a": int(a),
                "b": int(b),
                "kind": kind,
                "z": [float(v) for v in z.to_vector()],
                "sigma": [float(v) for v in sigma],
            }
        )
        return self._submit_graph()

    def _submit_graph(self) -> bool:
        if self._graph_future is not None:
            self._graph_pending = True
            return True
        nodes = sorted({0, *(e["a"] for e in self._graph_edges), *(e["b"] for e in self._graph_edges)})
        initial = {}
        for node in nodes:
            applied = self._submaps.get(node)["applied"] if node in self._submaps else None
            initial[str(node)] = [float(v) for v in (applied.to_vector() if applied else np.zeros(7))]
        payload = {
            "session_id": self.workmap_session_id,
            "source_epoch": int(self._source_epoch),
            "solver": self._graph_solver,
            "nodes": nodes,
            "initial": initial,
            "edges": [dict(edge) for edge in self._graph_edges],
        }
        try:
            self._ensure_ba_process()
            self._graph_future = self._ba_executor.submit(solve_submap_graph, payload)
        except Exception as error:
            self._graph_error = f"{type(error).__name__}: {error}"
            return False
        self._graph_payload = payload
        return True

    def _poll_graph(self) -> None:
        future = self._graph_future
        if future is None or not future.done():
            return
        payload = self._graph_payload
        self._graph_future = None
        self._graph_payload = None
        if int(payload["source_epoch"]) != self._source_epoch:
            self._graph_pending = False
            return
        try:
            result = dict(future.result())
        except Exception as error:
            result = {"ok": False, "reason": f"{type(error).__name__}: {error}", "nodes": {}}
        self._graph_solves += 1
        if not result.get("ok") and result.get("reason") == "gtsam_unavailable":
            self._graph_error = "gtsam_unavailable"
        self.session_workmap.note_event(
            {"event": "submap_graph_solved", "payload": payload, "result": result}
        )
        solved = {
            int(node): Sim3.from_vector(vector) for node, vector in (result.get("nodes") or {}).items()
        }
        for submap_id in payload["nodes"]:
            if submap_id == 0 or submap_id not in self._submaps:
                continue
            state = self._submaps.get(submap_id)["state"]
            if state == "merging":
                context = self._merge_context.get(submap_id)
                if context is not None:
                    self._merge_ready[submap_id] = solved.get(submap_id, context[0])
            elif state in {"merged", "unmerged", "active"} and submap_id in solved:
                self._apply_graph_transform(submap_id, solved[submap_id])
        if self._graph_pending:
            self._graph_pending = False
            if not self._submit_graph():
                for submap_id, context in self._merge_context.items():
                    self._merge_ready.setdefault(submap_id, context[0])

    def _apply_graph_transform(self, submap_id: int, solved: Sim3) -> None:
        """Move a submap's session data to its graph-solved Sim(3); GlueMap stays put."""
        registry = self._submaps
        applied = registry.get(submap_id)["applied"] or Sim3.identity()
        change = solved.to_vector() - applied.to_vector()
        if (
            float(np.linalg.norm(change[3:6])) < GRAPH_APPLY_MIN_TRANS_U
            and math.degrees(float(np.linalg.norm(change[:3]))) < GRAPH_APPLY_MIN_ROT_DEG
            and abs(float(change[6])) < GRAPH_APPLY_MIN_LOGSCALE
        ):
            return
        delta = solved.compose(applied.inverse())
        ids = self.session_workmap.apply_submap_transform(submap_id, delta)
        if not ids:
            self.session_workmap.note_event(
                {
                    "event": "submap_transform_rejected",
                    "submap_id": submap_id,
                    "reason": self.session_workmap.last_scale_rejection_reason,
                }
            )
            return
        self._scale_pending_use_ids.update(ids)
        self._refresh_live_points(ids)
        connected_frames = self.session_workmap.frames_observing(ids)
        for bid, meta in self._vo_seed_meta.items():
            seed_name = self.session_workmap._keyframe_by_capture.get(
                (self._source_epoch, int(meta[0]))
            )
            if seed_name in connected_frames and bid in self._vo_pose_by_bid:
                self._vo_pose_by_bid[bid] = _transform_pose_sim3(
                    self._vo_pose_by_bid[bid], delta.scale, delta.rotation, delta.translation
                )
        self._rescale_live_prior(ids, delta.scale, delta.rotation, delta.translation)
        registry.set_applied(submap_id, solved)
        registry.set_state(submap_id, "merged")
        if registry.active_id == submap_id:
            registry.set_active(registry.get(submap_id)["parent"] or 0)
        self._graph_applied += 1
        self._graph_generation += 1
        self._publish_submaps()
        self._submit_ba_if_ready()

    def _merge_rejected(self, submap_id: int, reason: str) -> None:
        """One failed merge attempt; three in a row give the submap up."""
        self._workmap_merge_rejected += 1
        self._merge_context.pop(submap_id, None)
        self._merge_ready.pop(submap_id, None)
        failures = self._merge_failures.get(submap_id, 0) + 1
        self._merge_failures[submap_id] = failures
        self.session_workmap.note_event(
            {"event": "submap_merge_rejected", "submap_id": submap_id, "reason": reason,
             "failures": failures, "stitch_s": getattr(self, "_last_stitch_s", None)}
        )
        registry = self._submaps
        if submap_id in registry and registry.get(submap_id)["state"] == "merging":
            if failures >= 3:
                self._mark_unmerged(submap_id, reason)
            else:
                registry.set_state(submap_id, "relocalizing")
                # Each attempt gets a full anchor window; the stale pairs would refit the same Sim(3).
                registry.clear_anchors(submap_id)
                self._reloc_first_anchor.pop(submap_id, None)
                self._reloc_started[submap_id] = int(self._ordinal)

    def _mark_unmerged(self, submap_id: int, reason: str) -> None:
        """The submap never joins GlueMap: its points leave the live set and the query."""
        registry = self._submaps
        self._merge_context.pop(submap_id, None)
        self._merge_ready.pop(submap_id, None)
        registry.set_state(submap_id, "unmerged")
        if registry.active_id == submap_id:
            registry.set_active(registry.get(submap_id)["parent"] or 0)
        ids = registry.landmark_ids(submap_id)
        self._unmerged_dirty = True        # live rows are purged at the next frame, never mid-solve
        if self._query is not None and len(ids):
            self._query.remove_landmarks(ids.tolist())
        self.session_workmap.note_event(
            {"event": "submap_unmerged", "submap_id": submap_id, "reason": reason,
             "ordinal": int(self._ordinal)}
        )
        self._publish_submaps()

    def _purge_unmerged_rows(self) -> None:
        """Drop live rows of submaps that gave up merging (between frames only)."""
        if not self._unmerged_dirty:
            return
        self._unmerged_dirty = False
        unmerged = [r["id"] for r in self._submaps.records() if r["state"] == "unmerged"]
        if not unmerged:
            return
        drop = np.isin(
            self._live_ids, np.concatenate([self._submaps.landmark_ids(k) for k in unmerged])
        )
        if drop.any():
            keep = ~drop
            self._live_xy = np.ascontiguousarray(self._live_xy[keep], dtype=np.float64)
            self._live_ids = self._live_ids[keep]
            self._live_xyz = self._live_xyz[keep]

    def _publish_submaps(self) -> None:
        self.session_workmap.set_submap_records(
            [*self._submap_archive, *self._submaps.records()]
        )

    def _apply_fusions(self, candidates: list[dict[str, Any]]) -> None:
        """Point duplicate session landmarks at the GlueMap point they re-observe."""
        for candidate in candidates:
            new_id, old_id = int(candidate["new_id"]), int(candidate["old_id"])
            landmark = self._workmap_landmarks.get(new_id)
            if landmark is None or landmark.get("fused_to") is not None:
                continue
            owner = self._submaps.submap_of(new_id)
            if owner != 0 and self._submaps.get(owner)["state"] != "merged":
                continue
            old_xyz = np.asarray(candidate["old_xyz"], dtype=np.float64)
            distance = float(np.linalg.norm(np.asarray(landmark["xyz"], dtype=np.float64) - old_xyz))
            pose, K = np.asarray(candidate["pose"]), np.asarray(candidate["K"])
            camera = pose[:, :3] @ old_xyz + pose[:, 3]
            if distance > FUSE_MAX_DISTANCE_U or camera[2] <= 1e-6:
                continue
            pixel = (K @ camera)[:2] / camera[2]
            reprojection = float(np.linalg.norm(pixel - np.asarray(candidate["new_xy"])))
            if reprojection > FUSE_MAX_REPROJ_PX:
                continue
            if not self.session_workmap.fuse_landmark(
                new_id, old_id, old_xyz, distance_u=distance, reprojection_px=reprojection
            ):
                continue
            rows = np.flatnonzero(self._live_ids == new_id)
            if len(rows):
                if np.any(self._live_ids == old_id):
                    keep = np.ones(len(self._live_ids), dtype=bool)
                    keep[rows] = False
                    self._live_xy = np.ascontiguousarray(self._live_xy[keep], dtype=np.float64)
                    self._live_ids = self._live_ids[keep]
                    self._live_xyz = self._live_xyz[keep]
                else:
                    self._live_ids[rows] = old_id
                    self._live_xyz[rows] = old_xyz
            self._workmap_landmarks.pop(new_id, None)
            self._workmap_pending_ids.discard(new_id)
            if self._query is not None:
                self._query.remove_landmarks([new_id])
            self._workmap_landmarks_fused += 1

    def _carry_seed_poses(self, snapshot: dict, poses: dict) -> None:
        """Pending VO seed poses follow the correction of the latest earlier keyframe."""
        chain = [f for f in snapshot["keyframes"] if f["name"] in poses]
        if not chain:
            return

        def lift(pose):
            return np.vstack((np.asarray(pose, dtype=np.float64).reshape(3, 4), [0, 0, 0, 1]))

        for bid, meta in self._vo_seed_meta.items():
            if bid not in self._vo_pose_by_bid:
                continue
            earlier = [f for f in chain if f["ordinal"] <= int(meta[0])]
            host = earlier[-1] if earlier else chain[0]
            correction = np.linalg.inv(lift(host["pose0"])) @ lift(poses[host["name"]])
            self._vo_pose_by_bid[bid] = (lift(self._vo_pose_by_bid[bid]) @ correction)[:3]

    def _poll_stitch(self) -> bool:
        future = self._stitch_future
        if future is None or not future.done():
            return False
        if self.session_workmap.writer_healthy and not self.session_workmap.geometry_commit_ready:
            return False
        self._last_stitch_s = time.monotonic() - float(self._stitch_started_mono or time.monotonic())
        snapshot = self._stitch_snapshot
        sim = self._stitch_sim
        self._stitch_future = None
        self._stitch_snapshot = None
        self._stitch_sim = None
        submap_id = int(snapshot["submap_id"])
        try:
            candidate = dict(future.result())
        except Exception as error:
            self._ba_error = f"{type(error).__name__}: {error}"
            self._merge_rejected(submap_id, "solver_error")
            return False
        if int(snapshot["source_epoch"]) != self._source_epoch:
            self._merge_rejected(submap_id, "source_epoch")
            return False
        closed = self.session_workmap.apply_loop_candidate(snapshot, candidate)
        if self._ba_pending and self._ba_future is None and self.workmap_eval_stage != "full":
            self._ba_pending = False
            self._ba_pending_mono = None
            self._submit_ba_if_ready()
        if os.environ.get("SFM_WORKMAP_DUMP_STITCH") == "1":
            import pickle

            dump = self.workmap_session_dir / f"stitch_{snapshot['loop_ordinal']}.pkl"
            dump.write_bytes(pickle.dumps({"snapshot": snapshot, "candidate": candidate,
                                           "committed": closed is not None}))
        if closed is None:
            self._merge_rejected(
                submap_id, self.session_workmap.last_loop_rejection_reason or "stitch_rejected"
            )
            return False
        self._refresh_live_points(closed["ids"])
        self._carry_seed_poses(snapshot, closed["poses"])
        self._speed_history.clear()
        self._graph_generation += 1
        self._workmap_merge_committed += 1
        self._workmap_merge_scale = float(snapshot["scale"])
        self._merge_failures.pop(submap_id, None)
        if submap_id in self._submaps:
            self._submaps.set_applied(submap_id, sim)
            self._submaps.set_state(submap_id, "merged")
        self.session_workmap.note_event(
            {
                "event": "submap_merged",
                "submap_id": submap_id,
                "noop": False,
                "scale": float(snapshot["scale"]),
                "anchors": sum(1 for frame in snapshot["keyframes"] if frame.get("loop")),
                "residual_u": self._merge_residual.get(submap_id),
                "stitch_s": self._last_stitch_s,
            }
        )
        self._publish_submaps()
        frames = [frame for frame in snapshot["keyframes"] if not frame.get("loop")]
        if frames:
            points = []
            sampled = np.linspace(0, len(frames) - 1, min(300, len(frames)), dtype=int)
            for index in sampled:
                frame = frames[int(index)]
                pose = np.asarray(closed["poses"].get(frame["name"], frame["pose0"]))
                center = -pose[:, :3].T @ pose[:, 3]
                points.append([int(frame["ordinal"]), *center.tolist()])
            self._trajectory_backfill = {
                "segment_id": (int(self._source_epoch) << 32) | (int(frames[0]["ordinal"]) + 1),
                "points": points,
            }
            self._backfill_repeats = 3
            self._backfill_next_ordinal = self._ordinal
        return True

    def _rescale_live_prior(
        self, update_ids: set[int], scale: float, rotation: np.ndarray, translation: np.ndarray
    ) -> None:
        # The last issued pose stays in the output history, but its live
        # tracking prior must follow a corrected gauge -- unless that would
        # cost frozen-map support.
        if (
            self._last_pose is None
            or self._last_pose_ordinal != self._ordinal
            or not any(int(key) in update_ids for key in self._live_ids)
        ):
            return
        transformed = _transform_pose_sim3(self._last_pose, scale, rotation, translation)
        positive = np.asarray(self._live_ids) > 0
        if np.any(positive):
            support = {
                "ba_ids": self._live_ids[positive],
                "ba_xy": self._live_xy[positive],
                "ba_xyz": self._live_xyz[positive],
                "K": self._K,
            }
            before, _ = self.session_workmap._frame_errors(support, self._last_pose, {})
            after, _ = self.session_workmap._frame_errors(support, transformed, {})
            if (
                np.count_nonzero(after <= 4.0) < np.count_nonzero(before <= 4.0)
                or np.percentile(after, 90) > np.percentile(before, 90) + 1e-6
            ):
                return
        self._last_pose = transformed
        self._prev_center = -transformed[:, :3].T @ transformed[:, 3]
        self._speed_history = [speed * scale for speed in self._speed_history]

    def _on_overlap_match(self, result: Mapping[str, Any] | None, error: str | None) -> None:
        if not self.session_workmap.record_overlap_evidence(result, error) and error:
            self._overlap_error = str(error)

    def _submit_or_retry_overlap(self, ref: Mapping[str, Any]) -> None:
        with self._overlap_retry_lock:
            name = str(ref["name"])
            count = len(ref["ids"])
            if count <= self._overlap_attempted.get(name, -1):
                return
            waiting = self._overlap_retry_ref
            if waiting is not None and str(waiting["name"]) != name:
                # One retry slot, newest wins: the older keyframe's overlap
                # evidence is dropped (as a rejected candidate), not the branch.
                self._overlap_superseded += 1
                self._overlap_retry_ref = None
                self._overlap_retry_started_mono = None
            elif waiting is not None and count < len(waiting["ids"]):
                return
            submit = getattr(self._worker, "submit_overlap", None)
            if not callable(submit):
                self._overlap_error = "relocalizer has no overlap matching support"
                return
            try:
                accepted = submit(ref, self._on_overlap_match)
            except Exception as error:
                self._overlap_error = f"{type(error).__name__}: {error}"
                return
            if accepted:
                self._overlap_attempted[name] = count
                self._overlap_last_match_submitted_mono = time.monotonic()
                self._overlap_retry_ref = None
                self._overlap_retry_started_mono = None
            else:
                self._overlap_retry_ref = ref
                if self._overlap_retry_started_mono is None:
                    self._overlap_retry_started_mono = time.monotonic()

    def _can_match_overlap_reference(self, reference: Mapping[str, Any]) -> bool:
        return self.session_workmap.can_match_overlap(
            reference,
            strong_old_inliers=self._overlap_strong_old_inliers,
            maximum_old_p90_px=self._overlap_max_old_p90_px,
        )

    def _retry_overlap_reference(self) -> None:
        if self._closing:
            return
        with self._overlap_retry_lock:
            reference = self._overlap_retry_ref
        if reference is None:
            return
        if not self._can_match_overlap_reference(reference):
            with self._overlap_retry_lock:
                if self._overlap_retry_ref is reference:
                    self._overlap_retry_ref = None
                    self._overlap_retry_started_mono = None
            return
        self._submit_or_retry_overlap(reference)

    def _submit_index_reference(self, reference: Mapping[str, Any], callback) -> bool:
        submit = getattr(self._worker, "submit_index", None)
        if not callable(submit):
            return False

        def indexed(ref: Mapping[str, Any], name: str | None, error: str | None) -> None:
            callback(ref, name, error)
            if (
                self.workmap_eval_stage != "full"
                or error is not None
                or not name
                or not self._can_match_overlap_reference(ref)
            ):
                return
            if (
                self._overlap_last_match_submitted_mono is not None
                and time.monotonic() - self._overlap_last_match_submitted_mono
                < self._overlap_match_spacing_s
            ):
                return
            self._submit_or_retry_overlap(ref)

        return bool(submit(reference, indexed))

    def request_fast_close(self) -> None:
        """Shutdown signal: abandon drains, still write the session snapshot."""
        self._fast_close.set()
        self.session_workmap.request_abort()

    @staticmethod
    def _terminate_ba_executor(executor: ProcessPoolExecutor) -> None:
        # Executor internals: shutdown(wait=False) alone leaves a solver that
        # is mid-job running after this process exits.
        processes = list(getattr(executor, "_processes", {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=0.2)

    def disable_mapping(self, reason: str) -> None:
        """Stop mapping side effects after a failure; localization stays up."""
        self.mapping_enabled = False
        self.mapping_error = str(reason)
        if getattr(self, "_presubmap", None) is not None:
            self._presubmap.close("failed")
        self._query = None
        try:
            self.session_workmap.note_event({"event": "mapping_disabled", "reason": self.mapping_error})
        except Exception:  # noqa: BLE001 - the session writer may be the failing part
            pass

    def close(self) -> None:
        if hasattr(self, "_submaps"):
            self._publish_submaps()
        if getattr(self, "_presubmap", None) is not None:
            self._presubmap.close("failed")
        defer_overlap = getattr(self._worker, "set_defer_overlap", None)
        if callable(defer_overlap):
            defer_overlap(False)
        archive_started = time.perf_counter()
        self.session_workmap.archive_landmarks(self._workmap_landmarks)
        archive_ms = (time.perf_counter() - archive_started) * 1000.0
        if self._fast_close.is_set():
            executor, self._ba_executor = getattr(self, "_ba_executor", None), None
            if executor is not None:
                self._terminate_ba_executor(executor)
            # Archive first: the relocalizer may still be inside a GPU job.
            try:
                self.session_workmap.close(shutdown_timeout_s=_FAST_CLOSE_S)
            finally:
                super().close()
            return

        stop_started = time.perf_counter()
        self._closing = True
        dropped = self._worker.abandon_maintenance()
        self.session_workmap.stop_index_dispatch()
        stop_done = time.perf_counter()
        deadline = time.monotonic() + 2.0
        while self._worker.running_kind() is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        gpu_done = time.perf_counter()
        cpu_dropped = {}
        for name, field in (
            ("ba", "_ba_future"), ("stitch", "_stitch_future"),
            ("overlap_fit", "_overlap_future"),
            ("presubmap_alignment", "_presubmap_alignment_future"),
        ):
            cpu_dropped[name] = int(getattr(self, field, None) is not None)
            setattr(self, field, None)
        executor, self._ba_executor = getattr(self, "_ba_executor", None), None
        if executor is not None:
            self._terminate_ba_executor(executor)
        cpu_done = time.perf_counter()
        self.session_workmap.flush_writer()
        writer_done = time.perf_counter()
        phase_ms: dict[str, float | None] = {
            "stop": (stop_done - stop_started) * 1000.0,
            "gpu_wait": (gpu_done - stop_done) * 1000.0,
            "cpu_terminate": (cpu_done - gpu_done) * 1000.0,
            "writer": (writer_done - cpu_done) * 1000.0,
            "archive": archive_ms,
            "index_flush": None,
            "reloc_close": None,
        }
        try:
            phase_started = time.perf_counter()
            self.session_workmap.flush_index_dispatcher()
            phase_ms["index_flush"] = (time.perf_counter() - phase_started) * 1000.0
        finally:
            try:
                phase_started = time.perf_counter()
                super().close()
                phase_ms["reloc_close"] = (time.perf_counter() - phase_started) * 1000.0
            finally:
                self.session_workmap.set_close_report({
                    "dropped": {
                        "overlap": dropped["overlap"], "depth": dropped["depth"],
                        **cpu_dropped,
                    },
                    "phase_ms": phase_ms,
                })
                self.session_workmap.close()

    def _new_presubmap(self) -> PreAnchorSubmap | None:
        """Local VO only before the first map fix or during a pose-less outage."""
        if (
            os.environ.get("SFM_WORKMAP_PRESUBMAP", "1") == "0"
            or self.workmap_eval_stage != "full"
            or not self.mapping_enabled
        ):
            return None
        submap = PreAnchorSubmap(self._K)
        self._presubmap_submap_id = self._submaps.open("independent", None, int(self._ordinal))
        return submap

    def _presubmap_after_step(self, info: dict, capture_stamp: float) -> None:
        """Track independently only when necessary; publish nothing before map alignment."""
        started = time.perf_counter()
        try:
            strong = int(self.profile.reloc.frozen_pnp_thresholds["strong_inliers"])
            handover_anchor = self._handover_anchor_answer
            anchor_pose = (
                np.asarray(handover_anchor["cam_from_world"].matrix(), dtype=np.float64)[:3, :4]
                if handover_anchor is not None else info.get("cam_from_world")
            )
            anchored = (
                handover_anchor is not None or (
                    anchor_pose is not None
                    and info.get("status") in {"FAST_TRACK", "RELOC_SEED"}
                    and int(info.get("map_inliers", 0)) >= strong
                )
            )
            if anchored:
                self._presubmap_had_strong = True
            submap = self._presubmap
            # Existing map-frame workmap VO/loop BA is cheaper and preserves map scale.
            no_map_vo = info.get("status") not in {"FAST_TRACK", "RELOC_SEED", "VO_ONLY"}
            self._presubmap_no_pose_frames = (
                self._presubmap_no_pose_frames + 1 if no_map_vo else 0
            )
            if (
                submap is not None and submap.active and self._presubmap_had_strong
                and info.get("status") == "VO_ONLY" and submap._first_anchor is None
            ):
                submap.close("unneeded")
            self._sync_presubmap_state()
            if (
                submap is not None and not submap.active
                and (self._presubmap_had_strong or submap.state in {"failed", "unmerged"})
                and self._presubmap_no_pose_frames >= 6
                and self._presubmap_alignment_future is None
                and self._presubmap_pending is None
            ):
                self._presubmap = submap = self._new_presubmap()
            if submap is not None and submap.active:
                if submap.wants(self._ordinal):
                    frame = self._frame_rgb
                    small = None
                    if frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                        small = (
                            frame.copy()
                            if frame.shape[:2] == (self._h, self._w)
                            else cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_AREA)
                        )
                    submap.enqueue(self._ordinal, float(capture_stamp), self._ring[-1][3], small)
                if anchored:
                    submap.add_anchor(
                        self._ordinal, np.asarray(anchor_pose, dtype=np.float64)[:3, :4],
                        float(capture_stamp),
                    )
                result = submap.try_merge()
                for reason in submap.pop_rejections():
                    self.session_workmap.note_event(
                        {"event": "presubmap_anchor_rejected", "reason": reason,
                         "ordinal": int(self._ordinal)}
                    )
                if result is not None and self._ba_executor is not None:
                    self._presubmap_alignment_epoch = int(self._source_epoch)
                    self._presubmap_alignment_result = result
                    self._ensure_ba_process()
                    anchor_ids = {
                        int(key) for ids, _ in result.anchor_views.values() for key in ids
                    }
                    point_ids = sorted(
                        result.points,
                        key=lambda key: (
                            key not in anchor_ids, -result.points[key].ordinal, key
                        ),
                    )[:600]
                    geometry = replace(
                        result,
                        points={key: result.points[key] for key in point_ids},
                        keyframes=[replace(frame, gray=None) for frame in result.keyframes],
                    )
                    self._presubmap_alignment_future = self._ba_executor.submit(
                        align_independent_submap, geometry, self._K
                    )
            self._sync_presubmap_state()
            self._poll_presubmap_alignment()
            self._flush_presubmap_queue()
        except Exception as error:  # noqa: BLE001 - independent VO is optional
            self._presubmap_error = f"{type(error).__name__}: {error}"
            if self._presubmap is not None:
                self._presubmap.close("failed")
            self._presubmap_alignment_future = None
            self._presubmap_alignment_result = None
            self.session_workmap.note_event(
                {"event": "presubmap_failed", "error": self._presubmap_error}
            )
        self._presubmap_ms = (time.perf_counter() - started) * 1000.0

    def _poll_presubmap_alignment(self) -> None:
        future = self._presubmap_alignment_future
        if future is None or not future.done() or self._presubmap_pending is not None:
            return
        result = self._presubmap_alignment_result
        epoch = self._presubmap_alignment_epoch
        self._presubmap_alignment_future = None
        self._presubmap_alignment_result = None
        if epoch != self._source_epoch:
            return
        candidate = future.result()
        submap_id = self._presubmap_submap_id
        if not candidate.get("ok"):
            self._presubmap_error = str(candidate.get("reason"))
            self.session_workmap.note_event(
                {"event": "presubmap_alignment_rejected", "reason": self._presubmap_error}
            )
            if submap_id is not None:
                self._submaps.set_state(submap_id, "unmerged")
                self._publish_submaps()
            return
        result.poses_map = {int(key): np.asarray(pose) for key, pose in candidate["poses"].items()}
        result.points_map = {int(key): np.asarray(point) for key, point in candidate["updates"].items()}
        # Admit only tracks actually solved and reprojection-checked by local BA.
        # The remaining local points have merely inherited a Sim3 guess.
        result.points = {
            key: point for key, point in result.points.items() if key in result.points_map
        }
        self._queue_presubmap(result)
        if submap_id is not None:
            self._submaps.set_applied(submap_id, Sim3(*result.similarity))
            self._submaps.set_state(submap_id, "merged")
            self._add_graph_edge(
                0, submap_id, "merge", Sim3(*result.similarity), _graph_sigma("merge", result.residual_u)
            )
            self._publish_submaps()
        self._trajectory_backfill = self._submap_backfill(result, epoch)
        self._backfill_repeats = 3 if self._trajectory_backfill is not None else 0
        self._backfill_next_ordinal = self._ordinal

    def _sync_presubmap_state(self) -> None:
        """A pre-anchor submap that was abandoned never joins the map."""
        submap, submap_id = self._presubmap, self._presubmap_submap_id
        if (
            submap is not None
            and submap_id is not None
            and submap.state in {"failed", "unmerged", "unneeded"}
            and self._submaps.get(submap_id)["state"] == "active"
        ):
            self._submaps.set_state(submap_id, "unmerged")
            self._publish_submaps()

    @staticmethod
    def _submap_backfill(result, epoch: int) -> dict[str, Any] | None:
        ordinals = sorted(result.poses_local)
        if not ordinals:
            return None
        chosen = np.linspace(0, len(ordinals) - 1, min(300, len(ordinals)), dtype=int)
        baseline = {}
        for ordinal in ordinals:
            pose = map_pose(result.poses_local[ordinal], result.similarity)
            baseline[ordinal] = -pose[:, :3].T @ pose[:, 3]
        # Display-only interpolation of the accepted BA correction; switching
        # abruptly between optimized keyframes and raw Sim3 poses draws jumps.
        adjusted = {**result.poses_map, **result.anchor_poses}
        control = sorted(set(ordinals) & set(adjusted))
        offsets = np.array([
            -adjusted[ordinal][:, :3].T @ adjusted[ordinal][:, 3] - baseline[ordinal]
            for ordinal in control
        ])
        points = []
        for index in chosen:
            ordinal = ordinals[int(index)]
            correction = np.array([
                np.interp(ordinal, control, offsets[:, axis]) for axis in range(3)
            ]) if control else np.zeros(3)
            points.append([int(ordinal), *(baseline[ordinal] + correction).tolist()])
        return {"segment_id": (int(epoch) << 32) | int(ordinals[0]), "points": points}

    def _queue_presubmap(self, result) -> None:
        """Queue a merged local submap; landmarks and keyframes reach the session over later frames."""
        ordered = sorted(result.points.items(), key=lambda item: (item[1].ordinal, item[0]))
        self._presubmap_pending = {
            "result": result,
            "points": ordered,
            "converted": 0,
            "landmarks": {},
            "id_of": {},
            "created": [],
            "next": 0,
            "keyframes": [k for k in result.keyframes if k.gray is not None],
            "epoch": int(self._source_epoch),
            "retries": 0,
            "recorded": 0,
            "submap_id": int(self._presubmap_submap_id or 0),
        }
        scale = result.similarity[0]
        self.session_workmap.note_event(
            {
                "event": "presubmap_merged",
                "scale": float(scale),
                "anchor_residual_u": float(result.residual_u),
                "anchors": int(result.anchors),
                "points": len(ordered),
                "keyframes": len(self._presubmap_pending["keyframes"]),
                "frames": len(result.poses_local),
                "first_ordinal": int(min(result.poses_local)),
            }
        )
        self._workmap_merge_committed += 1
        self.session_workmap.note_event(
            {
                "event": "submap_merged",
                "submap_id": int(self._presubmap_submap_id or 0),
                "kind": "independent",
                "noop": False,
                "scale": float(scale),
                "anchors": int(result.anchors),
                "residual_u": float(result.residual_u),
            }
        )
        self._presubmap_merged += 1

    def _convert_presubmap_points(self, budget: int) -> None:
        """Turn up to ``budget`` queued local points into map-frame landmark records."""
        pending = self._presubmap_pending
        similarity = pending["result"].similarity
        scale, rotation, translation = similarity
        stop = min(len(pending["points"]), pending["converted"] + budget)
        for local_id, point in pending["points"][pending["converted"] : stop]:
            key = int(self._vo_next_id)
            self._vo_next_id -= 1
            seed_pose = pending["result"].poses_map.get(
                point.seed_ordinal, pending["result"].anchor_poses.get(point.seed_ordinal)
            )
            if seed_pose is None:
                seed_pose = map_pose(point.seed_pose, similarity)
            pose = pending["result"].poses_map.get(
                point.ordinal, pending["result"].anchor_poses.get(point.ordinal)
            )
            if pose is None:
                pose = map_pose(point.pose, similarity)
            seed_pose = np.array(seed_pose, copy=True)
            pose = np.array(pose, copy=True)
            seed_pose.setflags(write=False)
            pose.setflags(write=False)
            record: dict[str, Any] = {
                "xyz": pending["result"].points_map.get(local_id, scale * (rotation @ point.xyz) + translation),
                "parallax_deg": float(point.parallax_deg),
                "revision": 0,
                "origin": "presubmap",
                "observations": (
                    (point.seed_ordinal, point.seed_stamp, point.seed_xy.copy(), seed_pose),
                    (point.ordinal, point.stamp, point.xy.copy(), pose),
                ),
                "last_inlier": (point.ordinal, point.stamp, point.xy.copy()),
            }
            if point.rgb is not None:
                record["rgb"] = list(point.rgb)
            pending["landmarks"][key] = record
            pending["id_of"][local_id] = key
            pending["created"].append((point.ordinal, key))
        pending["converted"] = stop

    def _flush_presubmap_queue(self) -> None:
        """Convert a chunk of merged points, then record at most one pre-anchor keyframe."""
        pending = self._presubmap_pending
        if pending is None:
            return
        if pending["epoch"] != self._source_epoch:
            self._presubmap_pending = None
            return
        self._convert_presubmap_points(PRESUBMAP_POINTS_PER_FRAME)
        similarity = pending["result"].similarity
        landmarks, created = pending["landmarks"], pending["created"]
        keyframes = pending["keyframes"]
        done = pending["converted"] >= len(pending["points"])

        def archive_until(ordinal: int) -> set[int]:
            added: dict[int, dict[str, Any]] = {}
            while pending["next"] < len(created) and created[pending["next"]][0] <= ordinal:
                key = created[pending["next"]][1]
                added[key] = landmarks[key]
                pending["next"] += 1
            for key, revision in self.session_workmap.archive_landmarks(
                added, submap_id=pending["submap_id"]
            ).items():
                added[key]["revision"] = int(revision)
            self._submaps.tag_landmarks(list(added), pending["submap_id"])
            self._workmap_landmarks.update(added)
            return set(added)

        if not keyframes:
            if done:
                archive_until(2**62)
                self._presubmap_pending = None
            return
        frame = keyframes[0]
        covered = done or pending["points"][pending["converted"]][1].ordinal > frame.ordinal
        if not covered:
            return
        added = archive_until(frame.ordinal)
        ids = np.array([pending["id_of"].get(int(i), 0) for i in frame.point_ids], dtype=np.int64)
        known = np.array([int(i) in self._workmap_landmarks for i in ids], dtype=bool)
        pose = pending["result"].poses_map.get(
            frame.ordinal, pending["result"].anchor_poses.get(frame.ordinal)
        )
        if pose is None:
            pose = map_pose(pending["result"].poses_local[frame.ordinal], similarity)
        name = self.session_workmap.record_keyframe(
            frame.gray,
            ordinal=frame.ordinal,
            stamp=frame.stamp,
            source_epoch=self._source_epoch,
            pose=pose,
            K=self._K,
            image_size=(self._w, self._h),
            ids=ids[known],
            xy=frame.xy[known],
            xyz=np.array([self._workmap_landmarks[int(i)]["xyz"] for i in ids[known]]).reshape(-1, 3),
            landmarks=self._workmap_landmarks,
            pending_ids=added,
            submap_id=pending["submap_id"],
        )
        if name is not None:
            self._submaps.tag_keyframe(name, pending["submap_id"])
        if name is None:
            pending["retries"] += 1
            if pending["retries"] < 60:
                return
        else:
            pending["recorded"] += 1
        pending["retries"] = 0
        keyframes.pop(0)
        if not keyframes and done:
            archive_until(2**62)
            self.session_workmap.note_event(
                {"event": "presubmap_registered", "keyframes": pending["recorded"], "points": len(landmarks)}
            )
            self._presubmap_pending = None

    def reset(self) -> None:
        """Clear only current-frame priors; session IDs and references survive."""
        self._mapping = MappingStateMachine()
        self._mapping_state = MappingState.ANCHORED_EXPAND
        if hasattr(self, "_submaps"):
            self._reset_submaps()
        if not self.mapping_enabled:
            super().reset()
            self._frame_rgb = None
            return
        if hasattr(self, "_presubmap"):
            if self._presubmap is not None:
                self._presubmap.close("failed")
            self._presubmap = self._new_presubmap()
            self._presubmap_pending = None
            self._presubmap_alignment_future = None
            self._presubmap_alignment_result = None
            self._trajectory_backfill = None
            self._backfill_repeats = 0
            self._presubmap_had_strong = False
            self._presubmap_no_pose_frames = 0
        if hasattr(self, "_ba_pending"):
            self._ba_pending = False
            self._ba_pending_mono = None
        self.session_workmap.archive_landmarks(self._workmap_landmarks)
        archived = self.session_workmap.tracker_landmarks()
        next_id = int(getattr(self, "_vo_next_id", -1))
        anchored = bool(getattr(self, "_workmap_initial_anchor", False))
        super().reset()
        self.session_workmap.update_source_epoch(self._source_epoch)
        self._workmap_landmarks = archived
        self._vo_next_id = min(next_id, min(archived, default=0) - 1)
        self._workmap_initial_anchor = anchored
        if getattr(self, "_query", None) is not None:
            self._query.clear_temporal()
        self._depth_ready.clear()
        self._last_depth_submit_stamp = None
        self._frame_rgb = None

    def _live_cap_policy(self) -> str:
        # Strong old-map coverage keeps the original frozen-first cap; otherwise spread.
        return "frozen_first" if self._mapping_state is MappingState.LOCALIZE else "spread"

    def _reset_submaps(self) -> None:
        """A reset ends every open submap: archive its record, start a fresh registry."""
        for record in self._submaps.records():
            if record["id"] == 0:
                continue
            if record["state"] in {"active", "relocalizing", "merging"}:
                record["state"] = "unmerged"
            self._submap_archive.append({**record, "epoch": int(self._source_epoch)})
        self._submaps = SubmapRegistry()
        self._reloc_started.clear()
        self._reloc_first_anchor.clear()
        self._merge_failures.clear()
        self._merge_residual.clear()
        self._merge_context.clear()
        self._merge_ready.clear()
        self._graph_edges.clear()
        self._graph_pending = False
        self._presubmap_submap_id = None
        self._publish_submaps()

    def _detached_open(self) -> bool:
        registry = self._submaps
        return registry.active_id != 0 and registry.get(registry.active_id)["state"] == "active"

    def _merge_pending(self) -> bool:
        return self._submaps.has_state("merging")

    def _presubmap_collecting(self) -> bool:
        """The pre-anchor submap holds anchor pairs and waits for a consistent set."""
        submap = self._presubmap
        return submap is not None and submap.state == "tracking" and bool(submap._anchors)

    def _step_absolute_pose(self, reseeded: bool, stamp: float):
        result = super()._step_absolute_pose(reseeded, stamp)
        if self.mapping_enabled:
            self._advance_mapping(result)
        return result

    def _advance_mapping(self, result: tuple) -> None:
        """Update the mapping state and open a detached submap once the old map is lost."""
        status, cam_from_world, _, _, map_inliers, _, n_corr, quality = result
        self._expire_relocalization()
        thresholds = self.profile.reloc.frozen_pnp_thresholds
        inputs = MappingInputs(
            pose_ok=cam_from_world is not None,
            status=str(status),
            map_inliers=int(map_inliers),
            n_corr=int(n_corr),
            hull_coverage=(quality or {}).get("inlier_hull_coverage"),
            detached_open=self._detached_open(),
            relocalizing=self._submaps.relocalizing_id is not None or self._presubmap_collecting(),
            merge_pending=self._merge_pending() or self._presubmap_pending is not None,
            strong_inliers=int(thresholds["strong_inliers"]),
            min_inliers=int(self.profile.fast_loop.pnp.min_inliers),
        )
        previous = self._mapping_state
        self._frozen_ratio = frozen_ratio(inputs)
        self._mapping_state = self._mapping.update(inputs)
        if (
            self._mapping_state is MappingState.DETACHED_EXPAND
            and cam_from_world is not None
            and self._submaps.active_id == 0
            and self._submaps.relocalizing_id is None
            and not inputs.merge_pending
        ):
            submap_id = self._submaps.open("detached", 0, int(self._ordinal))
            self._submaps.set_active(submap_id)
            self._add_graph_edge(0, submap_id, "sequential", Sim3.identity(), _graph_sigma("sequential"))
            self.session_workmap.note_event(
                {
                    "event": "submap_opened",
                    "submap_id": submap_id,
                    "kind": "detached",
                    "parent": 0,
                    "ordinal": int(self._ordinal),
                }
            )
            self._publish_submaps()
        if self._mapping_state is not previous:
            self.session_workmap.note_event(
                {
                    "event": "mapping_state",
                    "from": previous.value,
                    "to": self._mapping_state.value,
                    "ordinal": int(self._ordinal),
                }
            )

    def _submap_busy(self) -> bool:
        """Points and keyframes are not added while a submap awaits or undergoes its merge."""
        return self._submaps.relocalizing_id is not None or self._submaps.has_state("merging")

    def _vo_keyframe(self, gray: np.ndarray, cam_from_world: np.ndarray) -> None:
        if (
            not self.mapping_enabled
            or self._mapping_state is MappingState.LOCALIZE
            or self._submap_busy()
        ):
            super()._vo_keyframe(gray, cam_from_world)
            return
        submap_id = int(self._submaps.active_id)
        current_admissible = self._is_pose_admissible(cam_from_world)
        if self._depth_estimator is not None and current_admissible:
            self._promote_depth_seeds(cam_from_world)
        bid_before = self._vo_bid_next
        super()._vo_keyframe(gray, cam_from_world)
        added = {
            key: self._workmap_landmarks[key]
            for key in self._workmap_pending_ids
            if key in self._workmap_landmarks
        }
        revisions = self.session_workmap.archive_landmarks(added, submap_id=submap_id)
        self._submaps.tag_landmarks(list(revisions), submap_id)
        pending_ids = set(self._workmap_pending_ids)
        self._workmap_pending_ids.clear()
        for key, revision in revisions.items():
            if key in self._workmap_landmarks:
                self._workmap_landmarks[key]["revision"] = int(revision)
        name = None
        if current_admissible:
            trusted = (self._live_ids > 0) | np.isin(
                self._live_ids, list(self._workmap_landmarks.keys())
            )
            name = self.session_workmap.record_keyframe(
                gray,
                ordinal=self._ordinal,
                stamp=float(self._previous_capture_stamp),
                source_epoch=self._source_epoch,
                pose=np.asarray(cam_from_world, dtype=np.float64),
                K=self._K,
                image_size=(self._w, self._h),
                ids=self._live_ids[trusted],
                xy=self._live_xy[trusted],
                xyz=self._live_xyz[trusted],
                landmarks=self._workmap_landmarks,
                pending_ids=pending_ids,
                submap_id=submap_id,
            )
            if name is not None:
                self._submaps.tag_keyframe(name, submap_id)
        if self._query is not None:
            if added:
                self._query.upsert_landmarks(
                    list(added), np.array([added[k]["xyz"] for k in added])
                )
            if name is not None:
                trusted = (self._live_ids > 0) | np.isin(
                    self._live_ids, list(self._workmap_landmarks.keys())
                )
                self._query.add_keyframe(
                    name, gray, cam_from_world, self._live_ids[trusted], self._live_xy[trusted]
                )
        if (
            self._depth_estimator is not None
            and self._vo_bid_next > bid_before
            and current_admissible
        ):
            self._submit_depth_seeds(bid_before, cam_from_world)
        if self._ba_executor is not None and current_admissible:
            if self.workmap_eval_stage == "full":
                self._ba_pending = True
                if self._ba_pending_mono is None:
                    self._ba_pending_mono = time.monotonic()
            else:
                self._submit_ba_if_ready()

    def _validate_workmap_handover(self, fix, reference_names: tuple[str, ...]) -> bool:
        if str(getattr(fix, "reference_scope", "global")) != "workmap":
            return True
        snapshot = getattr(fix, "workmap_snapshot", None)
        if not isinstance(snapshot, Mapping):
            return False
        raw_revisions = snapshot.get("landmark_revisions")
        if not isinstance(raw_revisions, Mapping):
            return False
        try:
            revision_ids = {int(key) for key in raw_revisions}
            ids = np.asarray(fix.point3d_ids, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError, AttributeError):
            return False
        if not revision_ids or np.any(ids >= 0) or any(int(key) not in revision_ids for key in ids):
            return False
        return self.session_workmap.validate_snapshot(snapshot, reference_names)

    def _vo_detection_deferrable(self) -> bool:
        # The depth path consumes the new batch id in the same frame.
        return getattr(self, "_depth_estimator", None) is None

    def step(self, gray_native: np.ndarray, capture_stamp: float) -> dict:
        step_started = time.perf_counter()
        self._finish_pending_detection()
        mapping = self.mapping_enabled
        self._anchor_rows = self._anchor_local_inliers = -1    # -1: no anchor attempt this frame
        if mapping:
            self._purge_unmerged_rows()
            self._apply_fusions(self.session_workmap.take_fusion_candidates())
        full = self.workmap_eval_stage == "full"
        ba_applied_this_frame = (
            self._poll_ba() if mapping and self._ba_executor is not None else False
        )
        if mapping:
            self._poll_stitch()
        if mapping:
            self._poll_graph()
            self._advance_pending_merges()
        scale_applied_this_frame = self._poll_overlap() if mapping and full else False
        if mapping and full:
            self._retry_overlap_reference()
            self._submit_overlap_if_ready()
            stride = int(self.profile.vo.keyframe_stride)
            if (
                self._ba_executor is not None and self._ba_pending
                and self._ba_future is None and self._stitch_future is None
                and (stride == 1 or (self._ordinal + 1) % stride != 0)
            ):
                # Prepare the previous VO capture's BA separately from its KLT/VO work.
                self._ba_pending = False
                self._ba_pending_mono = None
                self._submit_ba_if_ready()
        core_started = time.perf_counter()
        try:
            info = super().step(gray_native, capture_stamp)
            if mapping:
                self._presubmap_after_step(info, capture_stamp)
        finally:
            self._frame_rgb = None
        core_done = time.perf_counter()
        self._note_expansion_frame(info)
        info.update(self._expansion_diagnostics())
        defer_overlap = getattr(self._worker, "set_defer_overlap", None)
        if callable(defer_overlap):
            # Only a strong global fix frees the GPU for frozen-map overlap.
            defer_overlap(info.get("status") not in {"FAST_TRACK", "RELOC_SEED"})
        actual_ids: tuple[int, ...] = ()
        if info.get("status") in {"FAST_TRACK", "VO_ONLY", "RELOC_SEED"}:
            actual_ids = tuple(
                int(key)
                for key in self._live_ids
                if key < 0
                and int(key) in self._workmap_pose_ids
                and int(key) in self._workmap_landmarks
                and int(self._workmap_landmarks[int(key)]["last_inlier"][0]) == self._ordinal
            )
            self._workmap_points_used += len(actual_ids)
            if mapping:
                reused = self.session_workmap.record_visual_use(
                    actual_ids,
                    ordinal=self._ordinal,
                    stamp=float(capture_stamp),
                    ba_ids=self._ba_pending_use_ids,
                )
                self._ba_pending_use_ids.difference_update(reused)
                scaled = self._scale_pending_use_ids.intersection(actual_ids)
                if scaled:
                    self.session_workmap.record_scale_visual_use(
                        scaled, ordinal=self._ordinal, stamp=float(capture_stamp)
                    )
                    self._scale_pending_use_ids.difference_update(scaled)
        info["workmap_pose_landmarks"] = len(actual_ids)
        info["workmap_mapping_state"] = self._mapping_state.value
        info["workmap_pose_basis"] = str(getattr(self, "_pose_basis", "mixed"))
        info["workmap_frozen_ratio"] = float(self._frozen_ratio)
        if (
            info.get("reference_scope") == "workmap"
            and bool(info.get("reloc_delivered"))
            and bool(info.get("klt_reseeded"))
            and info.get("status") == "VO_ONLY"
            and int(info.get("vo_inliers", 0)) >= int(self.profile.fast_loop.pnp.min_inliers)
        ):
            self._workmap_reloc_accepted += 1
        worker_stats = getattr(self._worker, "workmap_diagnostics", None)
        worker_info = worker_stats() if callable(worker_stats) else {}
        info.update(
            {
                "workmap_depth_pending": bool(worker_info.get("workmap_depth_pending", False)),
                "workmap_depth_oldest_age_ms": float(
                    worker_info.get("workmap_depth_oldest_age_ms", 0.0)
                ),
                "workmap_depth_healthy": bool(worker_info.get("workmap_depth_healthy", True)),
                "workmap_depth_errors": int(worker_info.get("workmap_depth_errors", 0)),
            }
        )
        session_stats = self.session_workmap.stats()
        ba_times = [
            stamp for stamp in (self._ba_started_mono, self._ba_pending_mono) if stamp is not None
        ]
        overlap_times = [
            stamp
            for stamp in (
                self._overlap_started_mono,
                self._overlap_pending_mono,
                self._overlap_retry_started_mono,
            )
            if stamp is not None
        ]
        index_handoff_pending = self.session_workmap.index_handoff_pending
        index_handoff_age_ms = self.session_workmap.index_handoff_oldest_age_ms()
        info.update(
            {
                "workmap_session_id": self.workmap_session_id,
                "workmap_session_dir": str(self.workmap_session_dir),
                "workmap_revision": session_stats["workmap_revision"],
                "workmap_points_created": session_stats["workmap_points_created"],
                "workmap_points_used": self._workmap_points_used,
                "workmap_keyframes_ready": session_stats["workmap_keyframes_ready"],
                "workmap_reloc_accepted": self._workmap_reloc_accepted,
                "workmap_ba_submitted": self._workmap_ba_submitted,
                "workmap_optimizer": self._ba_optimizer,
                **self._graph_stats,
                "workmap_ba_applied": self._workmap_ba_applied,
                "workmap_ba_stale": self._workmap_ba_stale,
                "workmap_scale_applied": self._workmap_scale_applied,
                "workmap_scale_applied_this_frame": scale_applied_this_frame,
                "workmap_presubmap_state": self._presubmap.state if self._presubmap else "off",
                "workmap_presubmap_points": len(self._presubmap.points) if self._presubmap else 0,
                "workmap_presubmap_merged": self._presubmap_merged,
                "workmap_presubmap_ms": self._presubmap_ms,
                "workmap_pre_ms": (core_started - step_started) * 1000.0,
                "workmap_core_ms": (core_done - core_started) * 1000.0,
                "workmap_post_ms": (time.perf_counter() - core_done) * 1000.0,
                "workmap_merge_submitted": self._workmap_merge_submitted,
                "workmap_merge_committed": self._workmap_merge_committed,
                "workmap_merge_rejected": self._workmap_merge_rejected,
                "workmap_merge_noop": self._workmap_merge_noop,
                "workmap_landmarks_fused": self._workmap_landmarks_fused,
                "workmap_graph_solves": self._graph_solves,
                "workmap_graph_applied": self._graph_applied,
                "workmap_graph_solver": self._graph_solver,
                "workmap_merge_scale": self._workmap_merge_scale,
                "workmap_active_submap": int(self._submaps.active_id),
                "workmap_submaps": len(self._submaps),
                "workmap_anchor_count": (
                    len(self._submaps.anchors(self._submaps.relocalizing_id))
                    if self._submaps.relocalizing_id is not None
                    else 0
                ),
                "workmap_anchor_rows": self._anchor_rows,
                "workmap_anchor_local_inliers": self._anchor_local_inliers,
                "workmap_overlap_evidence_count": session_stats["workmap_overlap_evidence_count"],
                "workmap_overlap_stale": self._workmap_overlap_stale,
                "workmap_overlap_rejected": self._workmap_overlap_rejected,
                "workmap_overlap_superseded": self._overlap_superseded,
                "workmap_overlap_healthy": (
                    self._overlap_error is None
                    and session_stats["workmap_overlap_healthy"]
                    and bool(worker_info.get("workmap_overlap_healthy", True))
                ),
                "workmap_overlap_pending": (
                    bool(overlap_times) or bool(worker_info.get("workmap_overlap_pending", False))
                ),
                "workmap_overlap_oldest_age_ms": max(
                    max(0.0, (time.monotonic() - min(overlap_times)) * 1000.0)
                    if overlap_times
                    else 0.0,
                    float(worker_info.get("workmap_overlap_oldest_age_ms", 0.0)),
                ),
                "workmap_writer_healthy": session_stats["workmap_writer_healthy"],
                "workmap_writer_overflow_count": session_stats["workmap_writer_overflow_count"],
                "workmap_writer_pending": session_stats["workmap_writer_pending"],
                "workmap_writer_oldest_age_ms": session_stats["workmap_writer_oldest_age_ms"],
                "workmap_index_errors": session_stats["workmap_index_errors"],
                "workmap_index_healthy": (
                    bool(worker_info.get("workmap_index_healthy", self._workmap_index_healthy))
                    and self._workmap_index_healthy
                    and session_stats["workmap_index_errors"] == 0
                ),
                "workmap_index_pending": (
                    bool(worker_info.get("workmap_index_pending", False)) or index_handoff_pending
                ),
                "workmap_index_oldest_age_ms": max(
                    float(worker_info.get("workmap_index_oldest_age_ms", 0.0)),
                    index_handoff_age_ms,
                ),
                "workmap_reloc_pending": bool(
                    worker_info.get("workmap_reloc_pending", self._worker.busy)
                ),
                "workmap_reloc_oldest_age_ms": float(
                    worker_info.get("workmap_reloc_oldest_age_ms", 0.0)
                ),
                "workmap_ba_pending": bool(ba_times),
                "workmap_ba_oldest_age_ms": (
                    max(0.0, (time.monotonic() - min(ba_times)) * 1000.0) if ba_times else 0.0
                ),
                "workmap_ba_healthy": self._ba_error is None,
                "workmap_ba_rejected": self._workmap_ba_rejected,
                "workmap_ba_applied_this_frame": ba_applied_this_frame,
                "reference_scope": str(info.get("reference_scope", "global")),
            }
        )
        if (
            self._trajectory_backfill is not None
            and self._backfill_repeats > 0
            and self._ordinal >= self._backfill_next_ordinal
        ):
            info["workmap_trajectory_backfill"] = self._trajectory_backfill
            self._backfill_repeats -= 1
            self._backfill_next_ordinal = self._ordinal + 30
            if not self._backfill_repeats:
                self._trajectory_backfill = None
        if self.mapping_enabled:
            for health_key in (
                "workmap_writer_healthy",
                "workmap_index_healthy",
                "workmap_ba_healthy",
                "workmap_overlap_healthy",
                "workmap_depth_healthy",
            ):
                if not bool(info.get(health_key, True)):
                    self.disable_mapping(f"unhealthy:{health_key}")
                    break
        return info

    def _validate_workmap_fix(self, fix, reference_names: tuple[str, ...]) -> bool:
        if any(
            self._submaps.is_unmerged(int(key))
            for key in np.asarray(getattr(fix, "point3d_ids", ()), dtype=np.int64)
            if key < 0
        ):
            return False
        return self._validate_workmap_handover(fix, reference_names)
