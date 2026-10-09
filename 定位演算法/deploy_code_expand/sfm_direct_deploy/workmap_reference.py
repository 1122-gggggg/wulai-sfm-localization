"""Validation of session-workmap keyframe records handed to the relocalizer.

Both entry points -- indexing a durable keyframe and matching it against the
frozen map -- receive the same record shape from ``SessionWorkMap``. Messages
keep the caller's prefix so a failure names the operation that rejected it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

REFERENCE_FIELDS = frozenset(
    {
        "name",
        "image_path",
        "ids",
        "xy",
        "xyz",
        "pose",
        "K",
        "revision",
        "session_id",
        "source_epoch",
        "landmark_revisions",
        "pose_revision",
        "ordinal",
        "stamp",
        "image_size",
    }
)


class OverlapQuery(NamedTuple):
    keyframe_name: str
    session_id: str
    keyframe_revision: int
    source_epoch: int
    pose_revision: int
    width: int
    height: int
    pose: np.ndarray
    camera_matrix: np.ndarray
    new_ids: np.ndarray
    new_xy: np.ndarray
    new_xyz: np.ndarray
    landmark_revisions: dict[int, int]
    image_path: Path


def require_reference_fields(reference: Any, label: str) -> None:
    if not isinstance(reference, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = REFERENCE_FIELDS.difference(reference)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def reference_integer(value: Any, prefix: str, field: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{prefix} {field} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{prefix} {field} must be nonnegative")
    return result


def reference_names(reference: Mapping[str, Any], message: str) -> tuple[str, str]:
    name = str(reference["name"]).strip()
    session_id = str(reference["session_id"]).strip()
    if not name or not session_id:
        raise ValueError(message)
    return name, session_id


def reference_image_size(reference: Mapping[str, Any], prefix: str) -> tuple[int, int]:
    image_size = np.asarray(reference["image_size"])
    if (
        image_size.shape != (2,)
        or not np.issubdtype(image_size.dtype, np.integer)
        or np.any(image_size <= 0)
    ):
        raise ValueError(f"{prefix} image_size must be positive integer [width, height]")
    width, height = (int(value) for value in image_size)
    return width, height


def reference_camera(
    reference: Mapping[str, Any], prefix: str, pose_description: str
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(reference["pose"], dtype=np.float64)
    if pose.shape == (4, 4):
        pose = pose[:3, :4]
    if pose.shape != (3, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{prefix} pose must be a finite 3x4 or 4x4 {pose_description}")
    camera_matrix = np.asarray(reference["K"], dtype=np.float64)
    if (
        camera_matrix.shape != (3, 3)
        or not np.isfinite(camera_matrix).all()
        or camera_matrix[0, 0] <= 0
        or camera_matrix[1, 1] <= 0
    ):
        raise ValueError(f"{prefix} K must be a finite pinhole 3x3 matrix")
    return pose, camera_matrix


def _raw_rows(reference: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(reference["ids"]),
        np.asarray(reference["xy"], dtype=np.float64),
        np.asarray(reference["xyz"], dtype=np.float64),
    )


def _negative_rows(
    raw_ids: np.ndarray, raw_xy: np.ndarray, raw_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of session points only; call after the arrays were validated."""
    negative = raw_ids < 0
    return (
        np.ascontiguousarray(raw_ids[negative], dtype=np.int64),
        np.ascontiguousarray(raw_xy[negative], dtype=np.float64),
        np.ascontiguousarray(raw_xyz[negative], dtype=np.float64),
    )


def index_reference_points(
    reference: Mapping[str, Any], width: int, height: int, minimum_tracks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Session landmarks of one keyframe; positive IDs never enter the catalog."""
    raw_ids, raw_xy, raw_xyz = _raw_rows(reference)
    if raw_ids.ndim != 1 or not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("workmap ids must be a one-dimensional integer array")
    if raw_xy.shape != (len(raw_ids), 2) or raw_xyz.shape != (len(raw_ids), 3):
        raise ValueError("workmap ids, xy, and xyz must have matching rows")
    if not np.isfinite(raw_xy).all() or not np.isfinite(raw_xyz).all():
        raise ValueError("workmap observations and points must be finite")
    ids, xy, xyz = _negative_rows(raw_ids, raw_xy, raw_xyz)
    if len(ids) < minimum_tracks:
        raise ValueError(
            f"workmap reference has {len(ids)} negative landmarks; "
            f"at least {minimum_tracks} are required"
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError("workmap reference contains duplicate negative landmark IDs")
    if np.any(xy < 0) or np.any(xy[:, 0] >= width) or np.any(xy[:, 1] >= height):
        raise ValueError("workmap observations lie outside the declared image size")
    landmark_revisions = reference["landmark_revisions"]
    if not isinstance(landmark_revisions, Mapping):
        raise ValueError("workmap landmark_revisions must be a mapping")
    revisions = {
        reference_integer(key, "workmap", "landmark ID", nonnegative=False): reference_integer(
            value, "workmap", "landmark revision"
        )
        for key, value in landmark_revisions.items()
    }
    if set(revisions) != set(int(point_id) for point_id in ids):
        raise ValueError("workmap landmark revisions must cover exactly the negative IDs")
    return ids, xy, xyz, revisions


def _overlap_landmark_id(raw_id: Any, raw_revision: Any) -> int:
    if isinstance(raw_id, (bool, np.bool_)) or isinstance(raw_revision, (bool, np.bool_)):
        raise ValueError("workmap overlap landmark revisions must be integer keyed")
    if isinstance(raw_id, (int, np.integer)):
        return int(raw_id)
    if isinstance(raw_id, str):
        try:
            return int(raw_id)
        except ValueError as error:
            raise ValueError("workmap overlap landmark revisions must be integers") from error
    raise ValueError("workmap overlap landmark revisions must be integer keyed")


def overlap_landmark_revisions(raw_revisions: Any, new_ids: np.ndarray) -> dict[int, int]:
    if not isinstance(raw_revisions, Mapping):
        raise ValueError("workmap overlap landmark_revisions must be a mapping")
    landmark_revisions: dict[int, int] = {}
    for raw_id, raw_revision in raw_revisions.items():
        point_id = _overlap_landmark_id(raw_id, raw_revision)
        if not isinstance(raw_revision, (int, np.integer)):
            raise ValueError("workmap overlap landmark revision must be an integer")
        point_revision = int(raw_revision)
        if point_id >= 0 or point_revision < 0:
            raise ValueError("workmap overlap landmark revisions require negative IDs")
        if point_id in landmark_revisions:
            raise ValueError("workmap overlap landmark revisions contain duplicate IDs")
        landmark_revisions[point_id] = point_revision
    if set(landmark_revisions) != set(int(point_id) for point_id in new_ids):
        raise ValueError("workmap overlap landmark revisions must cover the negative IDs")
    return landmark_revisions


def parse_overlap_reference(reference: Any) -> OverlapQuery:
    require_reference_fields(reference, "workmap overlap reference")
    keyframe_name, session_id = reference_names(
        reference, "workmap overlap name and session_id must be non-empty"
    )
    revision = reference_integer(reference["revision"], "workmap overlap", "revision")
    source_epoch = reference_integer(reference["source_epoch"], "workmap overlap", "source_epoch")
    pose_revision = reference_integer(
        reference["pose_revision"], "workmap overlap", "pose_revision"
    )
    width, height = reference_image_size(reference, "workmap overlap")
    pose, camera_matrix = reference_camera(reference, "workmap overlap", "matrix")
    raw_ids, raw_xy, raw_xyz = _raw_rows(reference)
    if (
        raw_ids.ndim != 1
        or not np.issubdtype(raw_ids.dtype, np.integer)
        or raw_xy.shape != (len(raw_ids), 2)
        or raw_xyz.shape != (len(raw_ids), 3)
        or not np.isfinite(raw_xy).all()
        or not np.isfinite(raw_xyz).all()
    ):
        raise ValueError("workmap overlap ids, xy, and xyz must be finite aligned arrays")
    new_ids, new_xy, new_xyz = _negative_rows(raw_ids, raw_xy, raw_xyz)
    if len(np.unique(new_ids)) != len(new_ids):
        raise ValueError("workmap overlap contains duplicate negative landmark IDs")
    if np.any(new_xy < 0) or np.any(new_xy[:, 0] >= width) or np.any(new_xy[:, 1] >= height):
        raise ValueError("workmap overlap observations lie outside image_size")
    return OverlapQuery(
        keyframe_name=keyframe_name,
        session_id=session_id,
        keyframe_revision=revision,
        source_epoch=source_epoch,
        pose_revision=pose_revision,
        width=width,
        height=height,
        pose=pose,
        camera_matrix=camera_matrix,
        new_ids=new_ids,
        new_xy=new_xy,
        new_xyz=new_xyz,
        landmark_revisions=overlap_landmark_revisions(reference["landmark_revisions"], new_ids),
        image_path=Path(str(reference["image_path"])).expanduser().resolve(strict=True),
    )


def unique_overlap_pairs(
    candidates: Mapping[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop every new or frozen point that matched more than one partner."""
    new_to_old: dict[int, set[int]] = {}
    old_to_new: dict[int, set[int]] = {}
    for new_id, old_id in candidates:
        new_to_old.setdefault(new_id, set()).add(old_id)
        old_to_new.setdefault(old_id, set()).add(new_id)
    pairs = [
        pair
        for (new_id, old_id), pair in candidates.items()
        if len(new_to_old[new_id]) == 1 and len(old_to_new[old_id]) == 1
    ]
    pairs.sort(
        key=lambda pair: (-float(pair["confidence"]), int(pair["new_id"]), int(pair["old_id"]))
    )
    return pairs
