"""Build an offline COLMAP release model from a completed online workmap session.

Only measured session observations are admitted.  The frozen base reconstruction is
read-only; all geometry changes happen in memory and the optimized model directory is
published by an atomic directory rename after every geometric gate succeeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEPLOY_DIR = _REPO_ROOT / "定位演算法" / "deploy_code_expand" / "sfm_direct_deploy"
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from direct_paths import default_runtime_paths, sha256_file, vendor_sys_path  # noqa: E402

vendor_sys_path()

_MIN_PARALLAX_DEG = 1.5
_MAX_INITIAL_REPROJECTION_P90_PX = 6.0
_MAX_FILTER_REPROJECTION_PX = 3.0
_SIM3_MAX_RELATIVE_HOLDOUT = 0.05
_SIM3_SCALE_BOUNDS = (0.5, 2.0)
_MIN_COMPONENT_OLD_IDS = 6
#: Submaps whose points and keyframes may enter the offline model.
_KEPT_SUBMAP_STATES = frozenset({"anchored", "merged"})


class _UnusableTrack(ValueError):
    """One measured track fails a geometry gate, not the session provenance gate."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"measured track fails {reason}")
        self.reason = reason


@dataclass(frozen=True)
class _Keyframe:
    name: str
    ordinal: int
    revision: int
    pose_revision: int
    source_epoch: int
    stamp: float
    pose: np.ndarray
    K: np.ndarray
    width: int
    height: int
    image_path: Path
    negative_ids: np.ndarray
    negative_xy: np.ndarray
    landmark_revisions: Mapping[int, int]


@dataclass(frozen=True)
class _Session:
    session_id: str
    source_epoch: int
    keyframes: tuple[_Keyframe, ...]
    points_xyz: Mapping[int, np.ndarray]
    negative_observations: Mapping[int, tuple[tuple[str, np.ndarray], ...]]
    positive_observations: Mapping[str, tuple[tuple[int, np.ndarray], ...]]
    input_hashes: Mapping[str, str]


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path.name}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number} must contain a JSON object")
            yield row


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _inside(path: Path, root: Path, *, label: str) -> Path:
    candidate = path.expanduser().resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside the completed session directory") from error
    return candidate


def _session_keyframes(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = snapshot.get("keyframes")
    if isinstance(raw, Mapping):
        rows = []
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError("snapshot keyframes must contain mapping records")
            row = dict(value)
            if str(row.get("name", key)) != str(key):
                raise ValueError("snapshot keyframe mapping key and name disagree")
            row.setdefault("name", str(key))
            rows.append(row)
        return rows
    if isinstance(raw, list) and all(isinstance(row, Mapping) for row in raw):
        return [dict(row) for row in raw]
    raise ValueError("snapshot keyframes must be a mapping or list")


def _validate_pose(value: Any, name: str) -> np.ndarray:
    pose = _array(value, (3, 4), name)
    rotation = pose[:, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-3)
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError(f"{name} rotation must be proper and orthonormal")
    return pose


def _validate_K(value: Any, name: str) -> np.ndarray:
    K = _array(value, (3, 3), name)
    if (
        K[0, 0] <= 0.0
        or K[1, 1] <= 0.0
        or not np.allclose(K[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-7)
        or not np.allclose(K[[0, 1], [1, 0]], 0.0, rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"{name} must be a zero-skew pinhole calibration")
    return K


def _pair_arrays(ids_value: Any, xy_value: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        ids = np.asarray(ids_value)
        xy = np.asarray(xy_value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} IDs and measured pixels are invalid") from error
    if ids.ndim != 1:
        raise ValueError(f"{name} IDs must be a one-dimensional integer array")
    if len(ids) == 0:
        if xy.size != 0 or xy.shape not in {(0,), (0, 2)}:
            raise ValueError(f"{name} empty IDs do not align with measured pixels")
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float64)
    if (
        not np.issubdtype(ids.dtype, np.integer)
        or np.issubdtype(ids.dtype, np.bool_)
        or xy.shape != (len(ids), 2)
        or not np.isfinite(xy).all()
    ):
        raise ValueError(f"{name} IDs and measured pixels must be aligned finite arrays")
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate point IDs in one image")
    return ids, xy


_MEASURED_OBSERVATION_SOURCES = frozenset(
    {
        "keyframe_image_measurement",
        "frozen_map_keyframe_image_measurement",
        "tracked_image_measurement",
    }
)


@dataclass(frozen=True)
class _CaptureEntry:
    name: str
    source_epoch: int
    ordinal: int
    stamp: float
    pose: np.ndarray


class _CaptureIndex:
    """Index durable keyframes by (source_epoch, ordinal) and candidate ordinals."""

    def __init__(self, keyframes: Mapping[str, Any]) -> None:
        self._by_capture: dict[tuple[int, int], str] = {}
        self._by_ordinal: dict[int, list[str]] = {}
        self._entries: dict[str, _CaptureEntry] = {}
        for name, item in keyframes.items():
            if isinstance(item, _Keyframe):
                epoch = item.source_epoch
                ordinal = item.ordinal
                stamp = item.stamp
                pose = item.pose
            elif isinstance(item, Mapping):
                epoch = _integer(item.get("source_epoch"), f"{name} source_epoch", minimum=0)
                ordinal = _integer(item.get("ordinal"), f"{name} ordinal", minimum=0)
                try:
                    stamp = float(item["stamp"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"invalid stamp for keyframe {name}") from exc
                if not math.isfinite(stamp):
                    raise ValueError(f"non-finite stamp for keyframe {name}")
                raw_pose = (
                    item.get("cam_from_world")
                    if item.get("cam_from_world") is not None
                    else item.get("pose")
                )
                pose = _validate_pose(raw_pose, f"keyframe {name} pose")
            else:
                raise ValueError(f"unsupported keyframe capture type: {type(item).__name__}")
            cap = (epoch, ordinal)
            if cap in self._by_capture:
                raise ValueError(f"durable keyframes have duplicate capture identity {cap}")
            self._by_capture[cap] = name
            self._by_ordinal.setdefault(ordinal, []).append(name)
            self._entries[name] = _CaptureEntry(
                name=name, source_epoch=epoch, ordinal=ordinal, stamp=stamp, pose=pose
            )

    def get_by_capture(self, epoch: int, ordinal: int) -> str | None:
        return self._by_capture.get((epoch, ordinal))

    def get_candidates_by_ordinal(self, ordinal: int) -> list[str]:
        return list(self._by_ordinal.get(ordinal, []))

    def get_entry(self, name: str) -> _CaptureEntry:
        return self._entries[name]

    def has_name(self, name: str) -> bool:
        return name in self._entries


def resolve_tracked_capture(
    index: _CaptureIndex,
    ordinal: int,
    stamp: float,
    pose: np.ndarray,
    *,
    source_epoch: int | None = None,
    expected_name: str | None = None,
) -> str | None:
    """Resolve durable keyframe for a tracked observation row, rejecting ambiguities and conflicts."""
    name: str | None = None
    if source_epoch is not None:
        name = index.get_by_capture(source_epoch, ordinal)
        if name is None:
            return None
        entry = index.get_entry(name)
        if not math.isclose(stamp, entry.stamp, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"tracked observation stamp does not match keyframe {name}")
        if not np.allclose(pose, entry.pose, rtol=0.0, atol=1e-4):
            raise ValueError(f"tracked observation pose does not match keyframe {name}")
    else:
        candidates = index.get_candidates_by_ordinal(ordinal)
        if not candidates:
            return None
        matching = [
            cand
            for cand in candidates
            if math.isclose(stamp, index.get_entry(cand).stamp, rel_tol=0.0, abs_tol=1e-6)
            and np.allclose(pose, index.get_entry(cand).pose, rtol=0.0, atol=1e-4)
        ]
        if not matching:
            return None
        if len(matching) > 1:
            raise ValueError(
                f"ambiguous keyframe binding for observation at ordinal {ordinal}: matches {matching}"
            )
        name = matching[0]

    if expected_name is not None and expected_name != name:
        raise ValueError(
            f"forged or conflicting keyframe_name in tracked observation: {expected_name!r} != {name!r}"
        )
    return name


def _tracked_observation_keyframe(
    row: Mapping[str, Any], landmark_id: int, capture_index: _CaptureIndex
) -> str | None:
    if landmark_id > 0:
        raise ValueError("tracked-image archive contains a non-session landmark ID")
    ordinal = _integer(row.get("ordinal"), "observation ordinal", minimum=0)
    raw_epoch = row.get("source_epoch")
    epoch = (
        _integer(raw_epoch, "observation source_epoch", minimum=0)
        if raw_epoch is not None
        else None
    )
    try:
        stamp = float(row.get("stamp"))
    except (TypeError, ValueError, OverflowError):
        return None
    pose = _validate_pose(row.get("cam_from_world"), "observation pose")
    return resolve_tracked_capture(
        capture_index,
        ordinal,
        stamp,
        pose,
        source_epoch=epoch,
        expected_name=row.get("keyframe_name"),
    )


def _durable_observation_keyframe(
    row: Mapping[str, Any],
    source: str,
    landmark_id: int,
    capture_index: _CaptureIndex,
    image_paths: Mapping[str, Path] | None = None,
) -> str:
    name = row.get("keyframe_name")
    if not isinstance(name, str) or not capture_index.has_name(name):
        raise ValueError("measured observation references an absent durable keyframe image")
    entry = capture_index.get_entry(name)
    ordinal = _integer(row.get("ordinal"), "observation ordinal", minimum=0)
    if ordinal != entry.ordinal:
        raise ValueError(f"observation ordinal does not match keyframe {name!r}")
    if "source_epoch" in row:
        row_epoch = _integer(row["source_epoch"], "observation source_epoch", minimum=0)
        if row_epoch != entry.source_epoch:
            raise ValueError(f"observation source_epoch does not match keyframe {name!r}")
    expected_source = (
        "keyframe_image_measurement" if landmark_id < 0 else "frozen_map_keyframe_image_measurement"
    )
    if source != expected_source:
        raise ValueError(
            f"{source!r} is not a measured keyframe image observation for ID {landmark_id}"
        )
    if image_paths is not None and name in image_paths:
        image_path = Path(str(row.get("image_path", ""))).expanduser().resolve(strict=True)
        if image_path != image_paths[name]:
            raise ValueError(f"measured observation image differs from durable keyframe {name!r}")
    return name


def _observation_keyframe(
    row: Mapping[str, Any],
    source: str,
    landmark_id: int,
    capture_index: _CaptureIndex,
    image_paths: Mapping[str, Path] | None = None,
) -> str | None:
    """The durable image a measured row belongs to; None for an unbound tracked row."""
    if source == "tracked_image_measurement":
        return _tracked_observation_keyframe(row, landmark_id, capture_index)
    return _durable_observation_keyframe(row, source, landmark_id, capture_index, image_paths)


def _check_observation_capture(
    row: Mapping[str, Any], frame: _Keyframe, manifest: Mapping[str, Any], name: str
) -> None:
    try:
        stamp = float(row.get("stamp"))
        manifest_stamp = float(manifest.get("stamp"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"observation for {name!r} has an invalid capture stamp") from error
    if (
        not math.isfinite(stamp)
        or not math.isfinite(manifest_stamp)
        or not math.isclose(stamp, frame.stamp, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(manifest_stamp, frame.stamp, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(f"observation stamp does not identify durable keyframe {name!r}")
    measured_pose = _validate_pose(row.get("cam_from_world"), f"observation {name} pose")
    manifest_pose = _validate_pose(manifest.get("cam_from_world"), f"manifest {name} pose")
    if not np.allclose(measured_pose, manifest_pose, rtol=0.0, atol=1e-4):
        raise ValueError(f"measured observation pose does not identify durable keyframe {name!r}")


def _observation_row(
    row: Mapping[str, Any],
    keyframes: Mapping[str, _Keyframe],
    session_id: str,
    manifest_by_name: Mapping[str, Mapping[str, Any]],
    capture_index: _CaptureIndex,
):
    source = row.get("observation_source")
    if source not in _MEASURED_OBSERVATION_SOURCES:
        return None
    if row.get("session_id") != session_id:
        raise ValueError("observations.jsonl spans a different session")
    landmark_id = _integer(row.get("landmark_id"), "observation landmark_id")
    if landmark_id == 0:
        raise ValueError("zero landmark IDs are invalid")
    image_paths = {name: frame.image_path for name, frame in keyframes.items()}
    name = _observation_keyframe(row, source, landmark_id, capture_index, image_paths=image_paths)
    if name is None:
        return None
    frame = keyframes[name]
    _check_observation_capture(row, frame, manifest_by_name[name], name)
    xy = _array(row.get("xy"), (2,), f"observation {name}:{landmark_id} xy")
    if not (0.0 <= xy[0] < frame.width and 0.0 <= xy[1] < frame.height):
        raise ValueError(f"observation for {landmark_id} falls outside {name!r}")
    revision = (
        _integer(row.get("landmark_revision"), "observation landmark_revision", minimum=0)
        if landmark_id < 0
        else None
    )
    return name, landmark_id, xy, revision


def _check_snapshot_header(
    session_snapshot: Any, base_release_id: str, coordinate_frame_id: str
) -> tuple[str, int]:
    if not isinstance(session_snapshot, Mapping):
        raise ValueError("session snapshot must be a mapping")
    if session_snapshot.get("schema") != "direct-online-workmap-session/v1":
        raise ValueError("unsupported or incomplete workmap snapshot schema")
    session_id = session_snapshot.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("snapshot session_id must be a non-empty string")
    source_epoch = _integer(
        session_snapshot.get("source_epoch"), "snapshot source_epoch", minimum=0
    )
    stats = session_snapshot.get("statistics")
    if not isinstance(stats, Mapping) or stats.get("writer_healthy") is not True:
        raise ValueError("workmap writer did not finish healthy")
    if session_snapshot.get("base_release_id") != base_release_id:
        raise ValueError("session base_release_id does not match the selected base release")
    if session_snapshot.get("coordinate_frame_id") != coordinate_frame_id:
        raise ValueError("session coordinate_frame_id does not match the selected base map")
    return session_id, source_epoch


def _check_session_meta(
    root: Path,
    session_snapshot: Mapping[str, Any],
    session_id: str,
    source_epoch: int,
    base_release_id: str,
    coordinate_frame_id: str,
    bundle_sha256: str,
) -> tuple[Path, Path, Path]:
    meta_path = root / "session_meta.json"
    keyframes_path = root / "keyframes.jsonl"
    observations_path = root / "observations.jsonl"
    for required in (meta_path, keyframes_path, observations_path):
        if not required.is_file():
            raise ValueError(f"completed session is missing {required.name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        not isinstance(meta, Mapping)
        or meta.get("schema") != "direct-online-workmap-session/v1"
        or meta.get("session_id") != session_id
        or meta.get("base_release_id") != base_release_id
        or meta.get("coordinate_frame_id") != coordinate_frame_id
        or meta.get("writer_healthy") is not True
    ):
        raise ValueError("session_meta.json does not certify this healthy base-bound session")
    if _integer(meta.get("source_epoch"), "session_meta source_epoch", minimum=0) != source_epoch:
        raise ValueError("session_meta.json source epoch differs from the final snapshot")
    if meta.get("base_bundle_sha256") != bundle_sha256:
        raise ValueError("session_meta.json lacks the matching base bundle SHA")
    if session_snapshot.get("base_bundle_sha256") != bundle_sha256:
        raise ValueError("final snapshot lacks the matching base bundle SHA")
    return meta_path, keyframes_path, observations_path


def _durable_manifest(keyframes_path: Path, session_id: str) -> dict[str, dict[str, Any]]:
    manifest_by_name: dict[str, dict[str, Any]] = {}
    for row in _jsonl(keyframes_path):
        name = row.get("name")
        if not isinstance(name, str) or not name or name in manifest_by_name:
            raise ValueError("keyframes.jsonl contains a missing or duplicate keyframe name")
        if row.get("session_id") != session_id or row.get("durable_complete") is not True:
            raise ValueError(f"keyframe {name!r} is not a durable complete session image")
        manifest_by_name[name] = row
    return manifest_by_name


def _keyframe_manifest(
    raw: Mapping[str, Any],
    manifest_by_name: Mapping[str, dict[str, Any]],
    snapshot_by_name: dict[str, Mapping[str, Any]],
    source_epoch: int,
    seen_captures: set[tuple[int, int]],
) -> tuple[str, Mapping[str, Any]]:
    name = raw.get("name")
    if not isinstance(name, str) or not name or name in snapshot_by_name:
        raise ValueError("snapshot contains a missing or duplicate keyframe name")
    snapshot_by_name[name] = raw
    manifest = manifest_by_name.get(name)
    if manifest is None:
        raise ValueError(f"snapshot keyframe {name!r} has no durable manifest record")
    if raw.get("durable") is not True:
        raise ValueError(f"snapshot keyframe {name!r} is not durable")
    raw_source_epoch = _integer(raw.get("source_epoch"), f"{name} source_epoch", minimum=0)
    manifest_source_epoch = _integer(
        manifest.get("source_epoch"), f"{name} manifest source_epoch", minimum=0
    )
    raw_ordinal = _integer(raw.get("ordinal"), f"{name} ordinal", minimum=0)
    manifest_ordinal = _integer(manifest.get("ordinal"), f"{name} manifest ordinal", minimum=0)
    if (
        raw_ordinal != manifest_ordinal
        or raw_source_epoch != manifest_source_epoch
        or raw_source_epoch > source_epoch
    ):
        raise ValueError(f"snapshot and durable manifest identity differ for {name!r}")
    capture_id = (raw_source_epoch, raw_ordinal)
    if capture_id in seen_captures:
        raise ValueError(f"snapshot contains duplicate keyframe capture identity {capture_id}")
    seen_captures.add(capture_id)
    return name, manifest


def _keyframe_camera(
    raw: Mapping[str, Any], manifest: Mapping[str, Any], name: str
) -> tuple[int, int, np.ndarray, np.ndarray, float]:
    size = np.asarray(raw.get("image_size"), dtype=np.float64)
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or np.any(size != np.floor(size))
    ):
        raise ValueError(f"keyframe {name!r} image_size must be positive integer [width, height]")
    width, height = (int(size[0]), int(size[1]))
    if list(manifest.get("image_size", ())) != [width, height]:
        raise ValueError(f"snapshot and manifest image sizes differ for {name!r}")
    K = _validate_K(raw.get("K"), f"keyframe {name} K")
    if not np.allclose(K, _validate_K(manifest.get("K"), f"manifest {name} K"), rtol=0, atol=1e-9):
        raise ValueError(f"snapshot and manifest intrinsics differ for {name!r}")
    _validate_pose(manifest.get("cam_from_world"), f"manifest {name} capture pose")
    try:
        manifest_stamp = float(manifest.get("stamp"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"manifest keyframe {name!r} has an invalid capture stamp") from error
    pose = _validate_pose(raw.get("pose"), f"keyframe {name} pose")
    try:
        stamp = float(raw.get("stamp"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"keyframe {name!r} has an invalid capture stamp") from error
    if (
        not math.isfinite(stamp)
        or not math.isfinite(manifest_stamp)
        or not math.isclose(stamp, manifest_stamp, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(f"snapshot and durable manifest capture stamps differ for {name!r}")
    return width, height, K, pose, stamp


def _keyframe_image(
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    name: str,
    root: Path,
    width: int,
    height: int,
) -> Path:
    raw_image_path = Path(str(raw.get("image_path", ""))).expanduser()
    image_path = _inside(
        raw_image_path if raw_image_path.is_absolute() else root / raw_image_path,
        root,
        label=f"keyframe {name} image",
    )
    manifest_image_path = Path(str(manifest.get("image_path", ""))).expanduser()
    if not manifest_image_path.is_absolute():
        manifest_image_path = root / manifest_image_path
    if manifest_image_path.resolve(strict=True) != image_path:
        raise ValueError(f"snapshot and manifest image paths differ for {name!r}")
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3) or image.shape[:2] != (height, width):
        raise ValueError(f"keyframe image {name!r} is missing, corrupt, or has the wrong size")
    return image_path


def _keyframe_observations(
    raw: Mapping[str, Any], manifest: Mapping[str, Any], name: str
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    ids, xy = _pair_arrays(raw.get("ids"), raw.get("xy"), f"keyframe {name} selected observations")
    if np.any(ids >= 0):
        raise ValueError(f"keyframe {name!r} selected session IDs must be negative")
    manifest_ids, manifest_xy = _pair_arrays(
        manifest.get("ids"), manifest.get("xy"), f"manifest {name} initial observations"
    )
    if np.any(manifest_ids >= 0):
        raise ValueError(f"manifest keyframe {name!r} selected session IDs must be negative")
    # archive_landmarks may append a true seed-view observation to the final
    # snapshot after this immutable image-manifest row was first written.
    final_xy_by_id = {int(point_id): xy[index] for index, point_id in enumerate(ids)}
    for point_id, pixel in zip(manifest_ids, manifest_xy, strict=True):
        if int(point_id) not in final_xy_by_id or not np.array_equal(
            final_xy_by_id[int(point_id)], pixel
        ):
            raise ValueError(
                f"final snapshot changed an immutable manifest observation for {name!r}"
            )
    manifest_selected = {
        int(point_id): manifest_xy[index].copy() for index, point_id in enumerate(manifest_ids)
    }
    return ids, xy, manifest_selected


def _keyframe_revisions(
    raw: Mapping[str, Any], name: str, ids: np.ndarray
) -> tuple[int, int, dict[int, int]]:
    revision = _integer(raw.get("revision"), f"{name} revision", minimum=0)
    pose_revision = _integer(raw.get("pose_revision"), f"{name} pose_revision", minimum=0)
    raw_revisions = raw.get("landmark_revisions")
    if not isinstance(raw_revisions, Mapping):
        raise ValueError(f"keyframe {name!r} has no landmark revision map")
    revisions: dict[int, int] = {}
    for raw_id, raw_revision in raw_revisions.items():
        point_id = _integer(int(raw_id), f"{name} landmark ID")
        if point_id >= 0 or point_id in revisions:
            raise ValueError(f"keyframe {name!r} has an invalid/duplicate landmark revision ID")
        revisions[point_id] = _integer(raw_revision, f"{name} landmark revision", minimum=0)
    if set(revisions) != set(map(int, ids)):
        raise ValueError(f"keyframe {name!r} revisions do not cover selected negative IDs")
    return revision, pose_revision, revisions


def _parse_keyframes(
    snapshot_rows: list[dict[str, Any]],
    manifest_by_name: Mapping[str, dict[str, Any]],
    source_epoch: int,
    root: Path,
) -> tuple[dict[str, _Keyframe], dict[str, Mapping[str, Any]], dict[str, dict[int, np.ndarray]]]:
    frame_records: dict[str, _Keyframe] = {}
    snapshot_by_name: dict[str, Mapping[str, Any]] = {}
    manifest_selected: dict[str, dict[int, np.ndarray]] = {}
    seen_captures: set[tuple[int, int]] = set()
    for raw in snapshot_rows:
        name, manifest = _keyframe_manifest(
            raw, manifest_by_name, snapshot_by_name, source_epoch, seen_captures
        )
        width, height, K, pose, stamp = _keyframe_camera(raw, manifest, name)
        image_path = _keyframe_image(raw, manifest, name, root, width, height)
        ids, xy, manifest_selected[name] = _keyframe_observations(raw, manifest, name)
        revision, pose_revision, revisions = _keyframe_revisions(raw, name, ids)
        frame_records[name] = _Keyframe(
            name=name,
            ordinal=int(raw["ordinal"]),
            revision=revision,
            pose_revision=pose_revision,
            source_epoch=int(raw["source_epoch"]),
            stamp=stamp,
            pose=pose,
            K=K,
            width=width,
            height=height,
            image_path=image_path,
            negative_ids=ids,
            negative_xy=xy,
            landmark_revisions=revisions,
        )
    return frame_records, snapshot_by_name, manifest_selected


def _ordinal_index(
    frame_records: Mapping[str, _Keyframe], manifest_by_name: Mapping[str, Any]
) -> _CaptureIndex:
    if set(manifest_by_name) != set(frame_records):
        raise ValueError(
            "keyframes.jsonl and final snapshot do not contain the same durable images"
        )
    return _CaptureIndex(frame_records)


def _snapshot_points(
    session_snapshot: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    raw_points = session_snapshot.get("landmarks")
    if not isinstance(raw_points, Mapping) or not raw_points:
        raise ValueError("workmap snapshot contains no landmarks")
    points_xyz: dict[int, np.ndarray] = {}
    point_revisions: dict[int, int] = {}
    for raw_id, value in raw_points.items():
        point_id = _integer(int(raw_id), "snapshot landmark ID")
        if point_id >= 0 or point_id in points_xyz or not isinstance(value, Mapping):
            raise ValueError("snapshot landmarks contain invalid or duplicate negative IDs")
        points_xyz[point_id] = _array(value.get("xyz"), (3,), f"landmark {point_id} xyz")
        point_revisions[point_id] = _integer(
            value.get("revision"), f"landmark {point_id} revision", minimum=0
        )
    return points_xyz, point_revisions


def _check_session_point_revision(
    point_id: int,
    frame: _Keyframe,
    observation_revision: int | None,
    points_xyz: Mapping[int, np.ndarray],
    point_revisions: Mapping[int, int],
) -> None:
    if point_id not in points_xyz:
        raise ValueError(f"measured observation refers to absent session point {point_id}")
    if point_id not in frame.landmark_revisions:
        raise ValueError(
            f"measured session point {point_id} is absent from {frame.name!r} snapshot"
        )
    if (
        observation_revision is None
        or observation_revision > frame.landmark_revisions[point_id]
        or observation_revision > point_revisions[point_id]
    ):
        raise ValueError(
            f"measured observation for {point_id} has a stale/future geometry revision"
        )


def _measured_observations(
    observations_path: Path,
    frame_records: Mapping[str, _Keyframe],
    session_id: str,
    manifest_by_name: Mapping[str, Mapping[str, Any]],
    capture_index: _CaptureIndex,
    points_xyz: Mapping[int, np.ndarray],
    point_revisions: Mapping[int, int],
    skip_row: Any = None,
) -> dict[tuple[str, int], tuple[np.ndarray, int | None]]:
    measured_by_key: dict[tuple[str, int], tuple[np.ndarray, int | None]] = {}
    # Archived tracked-image rows omit keyframe_name; only bind them to an image
    # when ordinal, capture stamp, and original camera pose identify that durable
    # capture. Projected PnP coordinates are deliberately not accepted here.
    for row in _iter_jsonl(observations_path):
        if skip_row is not None and skip_row(row):
            continue
        measured = _observation_row(row, frame_records, session_id, manifest_by_name, capture_index)
        if measured is None:
            continue
        name, point_id, xy, observation_revision = measured
        key = (name, point_id)
        if point_id < 0:
            _check_session_point_revision(
                point_id, frame_records[name], observation_revision, points_xyz, point_revisions
            )
        existing = measured_by_key.get(key)
        if existing is not None and not np.array_equal(existing[0], xy):
            raise ValueError(
                f"conflicting true image observations for point {point_id} in {name!r}"
            )
        if existing is None:
            measured_by_key[key] = (xy.copy(), observation_revision)
    return measured_by_key


def _check_manifest_history(
    manifest_selected: Mapping[str, Mapping[int, np.ndarray]],
    measured_by_key: Mapping[tuple[str, int], tuple[np.ndarray, int | None]],
) -> None:
    for name, manifest_observations in manifest_selected.items():
        for point_id, xy in manifest_observations.items():
            measured = measured_by_key.get((name, point_id))
            if measured is None or not np.array_equal(measured[0], xy):
                raise ValueError(
                    f"immutable keyframes.jsonl observation {point_id} in {name!r} is not present in true observation history"
                )


def _frame_negative_observations(
    frame: _Keyframe,
    measured_by_key: Mapping[tuple[str, int], tuple[np.ndarray, int | None]],
    negative_observations: dict[int, list[tuple[str, np.ndarray]]],
) -> None:
    for point_id, xy in zip(frame.negative_ids, frame.negative_xy, strict=True):
        measured = measured_by_key.get((frame.name, int(point_id)))
        if measured is None or not np.array_equal(measured[0], xy):
            raise ValueError(
                f"negative ID {point_id} in {frame.name!r} lacks its true durable image observation"
            )
        negative_observations.setdefault(int(point_id), []).append((frame.name, xy))


def _frame_positive_observations(
    frame: _Keyframe,
    raw: Mapping[str, Any],
    measured_by_key: Mapping[tuple[str, int], tuple[np.ndarray, int | None]],
) -> list[tuple[int, np.ndarray]]:
    raw_ba_ids, raw_ba_xy = _pair_arrays(
        raw.get("ba_ids"),
        raw.get("ba_xy"),
        f"keyframe {frame.name} all measured observations",
    )
    ba_by_id = {int(point_id): raw_ba_xy[index] for index, point_id in enumerate(raw_ba_ids)}
    positive: list[tuple[int, np.ndarray]] = []
    for (name, point_id), (xy, _revision) in measured_by_key.items():
        if name != frame.name or point_id < 0:
            continue
        if point_id not in ba_by_id or not np.array_equal(ba_by_id[point_id], xy):
            raise ValueError(
                f"old positive ID {point_id} is not an exact measured session observation"
            )
        positive.append((point_id, xy))
    return positive


def _check_negative_coverage(
    points_xyz: Mapping[int, np.ndarray],
    frame_records: Mapping[str, _Keyframe],
    negative_observations: Mapping[int, list[tuple[str, np.ndarray]]],
) -> None:
    eligible_negative_ids = {
        int(point_id) for frame in frame_records.values() for point_id in frame.negative_ids
    }
    if set(points_xyz) != eligible_negative_ids or eligible_negative_ids != set(
        negative_observations
    ):
        absent = sorted(set(points_xyz).symmetric_difference(eligible_negative_ids))
        raise ValueError(
            f"snapshot and measured negative observation identities disagree: {absent[:8]}"
        )
    for point_id, observations in negative_observations.items():
        frames = {name for name, _xy in observations}
        if len(frames) < 2:
            raise ValueError(f"new landmark {point_id} lacks two distinct measured keyframe images")


def _strip_points(row: Mapping[str, Any], drop: set[int]) -> dict[str, Any]:
    """A copy of one keyframe record without the observations of dropped session points."""
    result = dict(row)
    for ids_key, other_keys in (("ids", ("xy", "xyz")), ("ba_ids", ("ba_xy", "ba_xyz"))):
        ids = row.get(ids_key)
        if ids is None:
            continue
        keep = [index for index, point_id in enumerate(ids) if int(point_id) not in drop]
        if len(keep) == len(ids):
            continue
        result[ids_key] = [ids[index] for index in keep]
        for key in other_keys:
            if row.get(key) is not None:
                result[key] = [row[key][index] for index in keep]
    revisions = row.get("landmark_revisions")
    if isinstance(revisions, Mapping):
        result["landmark_revisions"] = {
            key: value for key, value in revisions.items() if int(key) not in drop
        }
    return result


def _without_unmerged_submaps(
    snapshot: Mapping[str, Any], manifest_by_name: Mapping[str, dict[str, Any]]
) -> tuple[Mapping[str, Any], Mapping[str, dict[str, Any]], Any]:
    """Drop what never joined GlueMap: unmerged submaps and landmarks fused into GlueMap IDs."""
    unmerged = {
        int(record["id"])
        for record in snapshot.get("submaps") or ()
        if record.get("state") not in _KEPT_SUBMAP_STATES
    }
    raw_points = snapshot.get("landmarks")
    drop_points = (
        {
            int(key)
            for key, value in raw_points.items()
            if int(value.get("submap_id", 0)) in unmerged or value.get("fused_to") is not None
        }
        if isinstance(raw_points, Mapping)
        else set()
    )
    rows = _session_keyframes(snapshot)
    drop_frames = {str(row["name"]) for row in rows if int(row.get("submap_id", 0)) in unmerged}
    if not drop_points and not drop_frames:
        return snapshot, manifest_by_name, None
    filtered = dict(snapshot)
    if isinstance(raw_points, Mapping):
        filtered["landmarks"] = {
            key: value for key, value in raw_points.items() if int(key) not in drop_points
        }
    filtered["keyframes"] = [
        _strip_points(row, drop_points) for row in rows if str(row["name"]) not in drop_frames
    ]
    manifest = {
        name: _strip_points(row, drop_points)
        for name, row in manifest_by_name.items()
        if name not in drop_frames
    }

    def skip_row(row: Mapping[str, Any]) -> bool:
        landmark_id = row.get("landmark_id")
        return (isinstance(landmark_id, int) and landmark_id in drop_points) or row.get(
            "keyframe_name"
        ) in drop_frames

    return filtered, manifest, skip_row


def _parse_session(
    session_snapshot: Mapping[str, Any],
    session_dir: Path,
    base_release_id: str,
    coordinate_frame_id: str,
    bundle_sha256: str,
) -> _Session:
    root = session_dir.expanduser().resolve(strict=True)
    session_id, source_epoch = _check_snapshot_header(
        session_snapshot, base_release_id, coordinate_frame_id
    )
    meta_path, keyframes_path, observations_path = _check_session_meta(
        root,
        session_snapshot,
        session_id,
        source_epoch,
        base_release_id,
        coordinate_frame_id,
        bundle_sha256,
    )
    manifest_by_name = _durable_manifest(keyframes_path, session_id)
    original_snapshot = session_snapshot
    session_snapshot, manifest_by_name, skip_row = _without_unmerged_submaps(
        session_snapshot, manifest_by_name
    )

    snapshot_rows = _session_keyframes(session_snapshot)
    if not snapshot_rows:
        raise ValueError("workmap snapshot contains no keyframes")
    frame_records, snapshot_by_name, manifest_selected = _parse_keyframes(
        snapshot_rows, manifest_by_name, source_epoch, root
    )
    capture_index = _ordinal_index(frame_records, manifest_by_name)
    points_xyz, point_revisions = _snapshot_points(session_snapshot)
    measured_by_key = _measured_observations(
        observations_path,
        frame_records,
        session_id,
        manifest_by_name,
        capture_index,
        points_xyz,
        point_revisions,
        skip_row,
    )
    _check_manifest_history(manifest_selected, measured_by_key)

    negative_observations: dict[int, list[tuple[str, np.ndarray]]] = {}
    positive_observations: dict[str, list[tuple[int, np.ndarray]]] = {
        name: [] for name in frame_records
    }
    for frame in frame_records.values():
        _frame_negative_observations(frame, measured_by_key, negative_observations)
        positive_observations[frame.name].extend(
            _frame_positive_observations(frame, snapshot_by_name[frame.name], measured_by_key)
        )
    _check_negative_coverage(points_xyz, frame_records, negative_observations)

    try:
        snapshot_payload = json.dumps(
            original_snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("final session snapshot is not canonical JSON data") from error
    return _Session(
        session_id=session_id,
        source_epoch=source_epoch,
        keyframes=tuple(sorted(frame_records.values(), key=lambda row: (row.ordinal, row.name))),
        points_xyz=points_xyz,
        negative_observations={
            point_id: tuple(sorted(rows, key=lambda row: (frame_records[row[0]].ordinal, row[0])))
            for point_id, rows in negative_observations.items()
        },
        positive_observations={name: tuple(rows) for name, rows in positive_observations.items()},
        input_hashes={
            "session_snapshot_sha256": hashlib.sha256(snapshot_payload).hexdigest(),
            "session_meta_sha256": sha256_file(meta_path),
            "keyframes_jsonl_sha256": sha256_file(keyframes_path),
            "observations_jsonl_sha256": sha256_file(observations_path),
            "keyframe_images_sha256": {
                frame.name: sha256_file(frame.image_path)
                for frame in sorted(frame_records.values(), key=lambda row: row.name)
            },
        },
    )


def _camera_center(pose: np.ndarray) -> np.ndarray:
    rotation = pose[:, :3]
    translation = pose[:, 3]
    return -rotation.T @ translation


def _project(K: np.ndarray, pose: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, float]:
    camera = pose[:, :3] @ xyz + pose[:, 3]
    if not np.isfinite(camera).all() or camera[2] <= 1e-8:
        raise _UnusableTrack("positive_depth")
    projected = K @ camera
    return projected[:2] / projected[2], float(camera[2])


def _triangulate_measured_track(
    observations: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    minimum_angle_deg: float = _MIN_PARALLAX_DEG,
    maximum_p90_px: float = _MAX_INITIAL_REPROJECTION_P90_PX,
) -> tuple[np.ndarray, dict[str, float]]:
    """Triangulate one point only from real 2D measurements and registered poses."""
    if len(observations) < 2:
        raise ValueError("a measured 3D track needs at least two distinct keyframe images")
    camera_centers = []
    rays = []
    rows = []
    for xy, K, pose in observations:
        pixel = _array(xy, (2,), "measured pixel")
        intrinsics = _validate_K(K, "measured keyframe K")
        cam_from_world = _validate_pose(pose, "measured keyframe pose")
        camera_centers.append(_camera_center(cam_from_world))
        ray_camera = np.linalg.inv(intrinsics) @ np.array([pixel[0], pixel[1], 1.0])
        ray_world = cam_from_world[:, :3].T @ ray_camera
        ray_world /= np.linalg.norm(ray_world)
        rays.append(ray_world)
        projection = intrinsics @ cam_from_world
        rows.extend(
            (pixel[0] * projection[2] - projection[0], pixel[1] * projection[2] - projection[1])
        )
    centers = np.asarray(camera_centers, dtype=np.float64)
    if len(np.unique(np.round(centers, decimals=10), axis=0)) < 2:
        raise _UnusableTrack("camera_baseline")
    parallax = max(
        math.degrees(math.acos(float(np.clip(np.dot(rays[first], rays[second]), -1.0, 1.0))))
        for first in range(len(rays))
        for second in range(first + 1, len(rays))
    )
    if not math.isfinite(parallax) or parallax < minimum_angle_deg:
        raise _UnusableTrack("minimum_parallax")
    design = np.asarray(rows, dtype=np.float64)
    _u, singular_values, right = np.linalg.svd(design, full_matrices=False)
    if len(singular_values) < 4 or singular_values[-2] <= 1e-12 * singular_values[0]:
        raise _UnusableTrack("rank_deficient")
    homogeneous = right[-1]
    if abs(float(homogeneous[3])) <= 1e-12:
        raise _UnusableTrack("at_infinity")
    xyz = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(xyz).all():
        raise _UnusableTrack("finite_triangulation")
    errors = []
    for xy, K, pose in observations:
        projected, _depth = _project(K, pose, xyz)
        errors.append(float(np.linalg.norm(projected - xy)))
    p90 = float(np.percentile(errors, 90.0))
    maximum = float(max(errors))
    if not math.isfinite(p90) or p90 > maximum_p90_px:
        raise _UnusableTrack("maximum_reprojection")
    return xyz, {
        "observation_count": len(observations),
        "parallax_deg": float(parallax),
        "reprojection_p50_px": float(np.percentile(errors, 50.0)),
        "reprojection_p90_px": p90,
        "reprojection_max_px": maximum,
    }


def _triangulate_session(
    session: _Session,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]], dict[str, int]]:
    frames = {frame.name: frame for frame in session.keyframes}
    triangulated: dict[int, np.ndarray] = {}
    metrics: dict[int, dict[str, float]] = {}
    rejected: dict[str, int] = {}
    for point_id, observations in sorted(session.negative_observations.items()):
        geometry = [(xy, frames[name].K, frames[name].pose) for name, xy in observations]
        try:
            xyz, record = _triangulate_measured_track(geometry)
            snapshot_xyz = session.points_xyz[point_id]
            # Persisted geometry is a consistency check, not a substitute for pixels.
            scale = max(
                1.0,
                float(
                    np.median(
                        [
                            np.linalg.norm(xyz - _camera_center(frames[name].pose))
                            for name, _xy in observations
                        ]
                    )
                ),
            )
            if float(np.linalg.norm(xyz - snapshot_xyz)) > max(0.25, 0.02 * scale):
                raise _UnusableTrack("archive_geometry_conflict")
        except _UnusableTrack as exc:
            rejected[exc.reason] = rejected.get(exc.reason, 0) + 1
            continue
        triangulated[point_id] = xyz
        metrics[point_id] = record
    if not triangulated:
        raise ValueError(f"no measured new landmark passes triangulation gates: {rejected}")
    return triangulated, metrics, rejected


def _retain_triangulated(session: _Session, triangulated: Mapping[int, np.ndarray]) -> _Session:
    accepted = set(triangulated)
    frames = []
    for frame in session.keyframes:
        selected = np.fromiter(
            (int(point_id) in accepted for point_id in frame.negative_ids),
            dtype=np.bool_,
            count=len(frame.negative_ids),
        )
        frames.append(
            replace(
                frame,
                negative_ids=frame.negative_ids[selected],
                negative_xy=frame.negative_xy[selected],
                landmark_revisions={
                    point_id: revision
                    for point_id, revision in frame.landmark_revisions.items()
                    if point_id in accepted
                },
            )
        )
    return replace(
        session,
        keyframes=tuple(frames),
        points_xyz={point_id: session.points_xyz[point_id] for point_id in accepted},
        negative_observations={
            point_id: session.negative_observations[point_id] for point_id in accepted
        },
    )


def _frame_components(session: _Session) -> list[dict[str, Any]]:
    parent = {frame.name: frame.name for frame in session.keyframes}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: str, second: str) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_point: dict[int, list[str]] = {}
    for point_id, observations in session.negative_observations.items():
        by_point[point_id] = [name for name, _xy in observations]
    for names in by_point.values():
        for name in names[1:]:
            union(names[0], name)
    groups: dict[str, list[str]] = {}
    for name in parent:
        groups.setdefault(find(name), []).append(name)
    return [{"keyframe_names": sorted(names)} for _root, names in sorted(groups.items())]


def _resolve_base_release(base_model: Path):
    from direct_map import DirectMapAssets
    from direct_profile import load_direct_profile

    model = base_model.expanduser().resolve(strict=True)
    if not model.is_dir():
        raise ValueError(f"base model must be an existing COLMAP model directory: {model}")
    bundle_path = None
    for ancestor in (model, *model.parents):
        candidate = ancestor / "localization" / "direct_bundle.json"
        if candidate.is_file():
            bundle_path = candidate
            break
    if bundle_path is None:
        raise ValueError(
            "base model has no owning localization/direct_bundle.json for real EDM validation"
        )
    assets = DirectMapAssets.load(bundle_path)
    if assets.model_dir.resolve() != model:
        raise ValueError("base model path differs from the immutable release bundle model")
    profile_path = assets.root / "localization" / "direct_localizer_profile.json"
    if not profile_path.is_file():
        raise ValueError("base release has no SHA-bound direct localizer profile")
    profile = load_direct_profile(profile_path)
    if profile.online_workmap.enabled is not True:
        raise ValueError(
            "base release profile is not the approved online-workmap evaluation profile"
        )
    return assets, profile


def _overlap_reference(
    frame: _Keyframe,
    triangulated: Mapping[int, np.ndarray],
    session: _Session,
) -> dict[str, Any]:
    ids = frame.negative_ids
    if len(ids) == 0:
        raise ValueError(f"keyframe {frame.name!r} has no persistent new landmarks for EDM overlap")
    return {
        "name": frame.name,
        "image_path": str(frame.image_path),
        "ids": ids,
        "xy": frame.negative_xy,
        "xyz": np.asarray([triangulated[int(point_id)] for point_id in ids], dtype=np.float64),
        "pose": frame.pose,
        "K": frame.K,
        "revision": frame.revision,
        "session_id": session.session_id,
        "source_epoch": session.source_epoch,
        "landmark_revisions": dict(frame.landmark_revisions),
        "pose_revision": frame.pose_revision,
        "ordinal": frame.ordinal,
        "stamp": frame.stamp,
        "image_size": [frame.width, frame.height],
    }


def _verified_overlap_pair(
    pair: Any, frame: _Keyframe, triangulated: Mapping[int, np.ndarray], old_reconstruction: Any
) -> dict[str, Any]:
    if not isinstance(pair, Mapping):
        raise ValueError("EDM overlap matcher returned a malformed pair")
    new_id = _integer(pair.get("new_id"), "EDM overlap new_id")
    old_id = _integer(pair.get("old_id"), "EDM overlap old_id")
    if new_id >= 0 or old_id <= 0 or new_id not in triangulated:
        raise ValueError("EDM overlap pair has invalid new/old landmark identity")
    if not old_reconstruction.exists_point3D(old_id):
        raise ValueError(f"EDM overlap references absent frozen Point3D ID {old_id}")
    old_xyz = np.asarray(old_reconstruction.point3D(old_id).xyz, dtype=np.float64)
    reported_old_xyz = _array(pair.get("old_xyz"), (3,), "EDM old_xyz")
    reported_new_xyz = _array(pair.get("new_xyz"), (3,), "EDM new_xyz")
    if not np.allclose(reported_old_xyz, old_xyz, rtol=0.0, atol=1e-7):
        raise ValueError(f"EDM overlap old ID {old_id} geometry differs from base model")
    if not np.allclose(reported_new_xyz, triangulated[new_id], rtol=0.0, atol=1e-7):
        raise ValueError(
            f"EDM overlap new ID {new_id} geometry differs from measured triangulation"
        )
    if pair.get("keyframe_name") != frame.name:
        raise ValueError("EDM overlap pair has a mismatched source keyframe")
    return dict(pair)


def _overlap_alignment(rows: list[dict[str, Any]], references_used: set[str]) -> dict[str, Any]:
    from workmap_overlap import fit_overlap_sim3

    if not rows:
        raise ValueError("real EDM produced no reciprocal, geometrically verified new-to-old pairs")
    alignment = fit_overlap_sim3(rows)
    if alignment.get("ok") is not True:
        raise ValueError(
            f"EDM overlap Sim3 fit/holdout rejected: {alignment.get('reason', 'unknown reason')}"
        )
    scale = float(alignment["scale"])
    if not math.isfinite(scale) or not _SIM3_SCALE_BOUNDS[0] <= scale <= _SIM3_SCALE_BOUNDS[1]:
        raise ValueError(f"EDM overlap Sim3 scale {scale!r} is outside the safe finite bounds")
    alignment["verified_rows"] = len(rows)
    alignment["unique_new_old_pairs"] = len(
        {(int(row["new_id"]), int(row["old_id"])) for row in rows}
    )
    alignment["source_keyframe_count"] = len({str(row["keyframe_name"]) for row in rows})
    alignment["old_reference_names"] = sorted(references_used)
    return alignment


def _make_overlap_evidence(
    provider: Any,
    session: _Session,
    triangulated: Mapping[int, np.ndarray],
    old_reconstruction: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    references_used: set[str] = set()
    for frame in session.keyframes:
        if not len(frame.negative_ids):
            continue
        result = provider.match_workmap_overlap(_overlap_reference(frame, triangulated, session))
        if not isinstance(result, Mapping) or result.get("session_id") != session.session_id:
            raise ValueError(f"EDM overlap matcher returned invalid provenance for {frame.name!r}")
        for pair in result.get("pairs", ()):
            row = _verified_overlap_pair(pair, frame, triangulated, old_reconstruction)
            references_used.add(str(row.get("old_reference_name", "")))
            rows.append(row)
    return rows, _overlap_alignment(rows, references_used)


def _validate_component_anchors(
    session: _Session,
    components: Sequence[Mapping[str, Any]],
    overlap_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_frame = {frame.name: frame for frame in session.keyframes}
    records: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        names = [str(name) for name in component["keyframe_names"]]
        new_ids = {int(point_id) for name in names for point_id in by_frame[name].negative_ids}
        matched = {
            int(row["new_id"]): int(row["old_id"])
            for row in overlap_rows
            if str(row["keyframe_name"]) in names
        }
        matched_frames = {
            str(row["keyframe_name"]) for row in overlap_rows if str(row["keyframe_name"]) in names
        }
        positive_ids = {
            int(point_id) for name in names for point_id, _xy in session.positive_observations[name]
        }
        if not new_ids:
            if len(positive_ids) < _MIN_COMPONENT_OLD_IDS:
                raise ValueError(
                    f"old-only component {component_index} has fewer than six measured base anchors"
                )
            records.append(
                {
                    "component_index": int(component_index),
                    "keyframe_names": names,
                    "edm_anchored_negative_ids": [],
                    "edm_old_point_ids": [],
                    "edm_source_keyframes": [],
                    "exact_old_track_point_ids": sorted(positive_ids),
                }
            )
            continue
        exact_old_ids = set(matched.values()) & positive_ids
        if len(set(matched) & new_ids) < _MIN_COMPONENT_OLD_IDS or len(matched_frames) < 2:
            raise ValueError(
                f"new component {component_index} lacks six distinct EDM-verified old anchors across two keyframes"
            )
        if not exact_old_ids:
            raise ValueError(
                f"new component {component_index} has no matching old landmark in a true measured session track"
            )
        records.append(
            {
                "component_index": int(component_index),
                "keyframe_names": names,
                "edm_anchored_negative_ids": sorted(set(matched) & new_ids),
                "edm_old_point_ids": sorted(set(matched.values())),
                "edm_source_keyframes": sorted(matched_frames),
                "exact_old_track_point_ids": sorted(exact_old_ids),
            }
        )
    return records


def _new_new_pair_candidates(session: _Session) -> list[tuple[_Keyframe, _Keyframe, set[int]]]:
    frames = {frame.name: frame for frame in session.keyframes}
    ids_by_frame = {frame.name: set(map(int, frame.negative_ids)) for frame in session.keyframes}
    edges = []
    names = sorted(frames)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            shared = ids_by_frame[first_name] & ids_by_frame[second_name]
            if shared:
                edges.append((-len(shared), first_name, second_name, shared))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))

    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    selected = []
    for _negative_count, first_name, second_name, shared in edges:
        first_root, second_root = find(first_name), find(second_name)
        if first_root == second_root:
            continue
        parent[max(first_root, second_root)] = min(first_root, second_root)
        selected.append((frames[first_name], frames[second_name], shared))
    return selected


def _validate_new_new_edm(
    provider: Any,
    session: _Session,
) -> tuple[int, list[dict[str, Any]]]:
    if not session.keyframes:
        raise ValueError("session has no keyframes for new-to-new EDM validation")
    from river_map_quality.official_edm_adapter import (
        _nearest_observation_within,
        prepare_official_megadepth_image_from_array,
    )
    from sfm_diagnosis.site_pipeline.deployment_localizer import _valid_matches
    from live_provider import _reciprocal_edm_matches, _strict_two_view_inliers
    import cv2

    runtime = provider._matcher_runtime()
    lift_distance = float(provider.profile.reloc.lift_distance_px)
    reciprocal_tolerance = min(2.0, lift_distance)
    if not math.isfinite(lift_distance) or lift_distance <= 0:
        raise ValueError("profile workmap lift distance must be finite and positive")
    validated_count = 0
    validated_edges: list[dict[str, Any]] = []
    candidates = _new_new_pair_candidates(session)
    if not candidates:
        raise ValueError("session has no covisible new-keyframe pairs for new-to-new EDM")
    for first, second, common_ids in candidates:
        first_gray = cv2.imread(str(first.image_path), cv2.IMREAD_GRAYSCALE)
        second_gray = cv2.imread(str(second.image_path), cv2.IMREAD_GRAYSCALE)
        if (
            first_gray is None
            or second_gray is None
            or first_gray.shape != (first.height, first.width)
            or second_gray.shape != (second.height, second.width)
        ):
            raise ValueError("a measured new-to-new EDM image could not be decoded")
        prepared_first = prepare_official_megadepth_image_from_array(first_gray, runtime)
        prepared_second = prepare_official_megadepth_image_from_array(second_gray, runtime)
        forward_raw = provider._match_prepared(runtime, prepared_first, prepared_second)
        reverse_raw = provider._match_prepared(runtime, prepared_second, prepared_first)
        first_shape = (first.height, first.width)
        second_shape = (second.height, second.width)
        forward0 = np.asarray(forward_raw["mkpts0_f"], dtype=np.float64).reshape(-1, 2)
        forward1 = np.asarray(forward_raw["mkpts1_f"], dtype=np.float64).reshape(-1, 2)
        forward_conf = np.asarray(forward_raw["mconf"], dtype=np.float64).reshape(-1)
        reverse0 = np.asarray(reverse_raw["mkpts0_f"], dtype=np.float64).reshape(-1, 2)
        reverse1 = np.asarray(reverse_raw["mkpts1_f"], dtype=np.float64).reshape(-1, 2)
        reverse_conf = np.asarray(reverse_raw["mconf"], dtype=np.float64).reshape(-1)
        threshold = float(provider.edm_config.get("confidence_threshold") or 0.0)
        forward_valid = _valid_matches(
            forward0,
            forward1,
            forward_conf,
            query_shape=first_shape,
            reference_shape=second_shape,
            confidence_threshold=threshold,
        )
        reverse_valid = _valid_matches(
            reverse0,
            reverse1,
            reverse_conf,
            query_shape=second_shape,
            reference_shape=first_shape,
            confidence_threshold=threshold,
        )
        forward = {
            "mkpts0_f": forward0[forward_valid],
            "mkpts1_f": forward1[forward_valid],
            "mconf": forward_conf[forward_valid],
        }
        reverse = {
            "mkpts0_f": reverse0[reverse_valid],
            "mkpts1_f": reverse1[reverse_valid],
            "mconf": reverse_conf[reverse_valid],
        }
        reciprocal_rows, _confidences = _reciprocal_edm_matches(
            forward, reverse, tolerance_px=reciprocal_tolerance
        )
        if len(reciprocal_rows) < 8:
            raise ValueError(
                f"new-to-new EDM has fewer than eight reciprocal matches for {first.name!r}/{second.name!r}"
            )
        first_xy = forward["mkpts0_f"][reciprocal_rows]
        second_xy = forward["mkpts1_f"][reciprocal_rows]
        geometric = _strict_two_view_inliers(first_xy, second_xy)
        if int(geometric.sum()) < 6:
            raise ValueError(
                f"new-to-new EDM geometric validation failed for {first.name!r}/{second.name!r}"
            )
        first_row_ids = np.asarray(first.negative_ids, dtype=np.int64)
        second_row_ids = np.asarray(second.negative_ids, dtype=np.int64)
        lifted_first, _d0, first_found = _nearest_observation_within(
            first_xy[geometric], first.negative_xy, first_row_ids, lift_distance
        )
        lifted_second, _d1, second_found = _nearest_observation_within(
            second_xy[geometric], second.negative_xy, second_row_ids, lift_distance
        )
        accepted = first_found & second_found & (lifted_first < 0) & (lifted_first == lifted_second)
        accepted_ids = sorted(set(map(int, lifted_first[accepted])) & common_ids)
        if not accepted_ids:
            raise ValueError(
                f"new-to-new EDM did not lift to the same measured persistent landmarks for {first.name!r}/{second.name!r}"
            )
        validated_count += len(accepted_ids)
        validated_edges.append(
            {
                "keyframe_names": [first.name, second.name],
                "covisible_negative_ids": sorted(common_ids),
                "validated_negative_ids": accepted_ids,
                "reciprocal_match_count": int(len(reciprocal_rows)),
                "geometric_inlier_count": int(geometric.sum()),
                "validated_pair_count": len(accepted_ids),
            }
        )
    return validated_count, validated_edges


def _sim3_transform_points(points: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    scale = float(transform["scale"])
    rotation = _array(transform["rotation"], (3, 3), "Sim3 rotation")
    translation = _array(transform["translation"], (3,), "Sim3 translation")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Sim3 scale must be finite and positive")
    return scale * (rotation @ points.T).T + translation


def _sim3_transform_pose(pose: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    scale = float(transform["scale"])
    rotation = _array(transform["rotation"], (3, 3), "Sim3 rotation")
    translation = _array(transform["translation"], (3,), "Sim3 translation")
    cam_rotation = pose[:, :3] @ rotation.T
    cam_translation = scale * pose[:, 3] - cam_rotation @ translation
    return np.column_stack((cam_rotation, cam_translation))


def _split_image_ids(names: Sequence[str]) -> tuple[list[str], list[str]]:
    if len(names) < 12:
        raise ValueError(
            "global BA gauge validation requires at least twelve original registered images"
        )
    ordered = sorted(names, key=lambda name: hashlib.sha256(name.encode("utf-8")).digest())
    fit_count = int(math.floor(0.7 * len(ordered)))
    fit_count = min(len(ordered) - 6, max(6, fit_count))
    return sorted(ordered[:fit_count]), sorted(ordered[fit_count:])


def _fit_original_gauge(
    candidate_centers: Mapping[str, np.ndarray],
    original_centers: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    from sfm_diagnosis.sim3 import evaluate_holdout_similarity

    names = sorted(set(candidate_centers) & set(original_centers))
    fit_names, holdout_names = _split_image_ids(names)
    fit_source = np.asarray([candidate_centers[name] for name in fit_names], dtype=np.float64)
    fit_target = np.asarray([original_centers[name] for name in fit_names], dtype=np.float64)
    holdout_source = np.asarray(
        [candidate_centers[name] for name in holdout_names], dtype=np.float64
    )
    holdout_target = np.asarray(
        [original_centers[name] for name in holdout_names], dtype=np.float64
    )
    evaluation = evaluate_holdout_similarity(
        fit_source,
        fit_target,
        holdout_source,
        holdout_target,
        max_relative_residual=_SIM3_MAX_RELATIVE_HOLDOUT,
    )
    transform = evaluation["transform"]
    scale = float(transform.scale)
    if not _SIM3_SCALE_BOUNDS[0] <= scale <= _SIM3_SCALE_BOUNDS[1]:
        raise ValueError(f"post-BA gauge Sim3 scale {scale:.6g} is outside safe bounds")
    residuals = np.asarray(evaluation["holdout_residuals"], dtype=np.float64)
    relative = np.asarray(evaluation["holdout_relative_residuals"], dtype=np.float64)
    if not evaluation["consistent"]:
        raise ValueError(
            f"global BA changed the original map by non-Sim3 distortion; holdout relative p90={np.percentile(relative, 90):.6g}"
        )
    return {
        "pass": True,
        "scale": scale,
        "rotation": np.asarray(transform.rotation, dtype=np.float64).tolist(),
        "translation": np.asarray(transform.translation, dtype=np.float64).tolist(),
        "fit": {
            "image_names": fit_names,
            "count": len(fit_names),
            "rms_residual": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                scale * (transform.rotation @ fit_source.T).T
                                + transform.translation
                                - fit_target
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
            ),
        },
        "holdout": {
            "image_names": holdout_names,
            "count": len(holdout_names),
            "residuals": residuals.tolist(),
            "relative_residuals": relative.tolist(),
            "residual_p50": float(np.percentile(residuals, 50.0)),
            "residual_p90": float(np.percentile(residuals, 90.0)),
            "relative_residual_p90": float(np.percentile(relative, 90.0)),
            "target_anchor_span": float(evaluation["target_anchor_span"]),
            "max_relative_residual": _SIM3_MAX_RELATIVE_HOLDOUT,
        },
    }


def _camera_calibrations(reconstruction: Any) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    return {
        int(camera_id): (
            str(camera.model_name),
            int(camera.width),
            int(camera.height),
            tuple(float(value) for value in camera.params),
        )
        for camera_id, camera in reconstruction.cameras.items()
    }


def _base_camera_centers(reconstruction: Any) -> dict[str, np.ndarray]:
    return {
        str(image.name): np.asarray(image.projection_center(), dtype=np.float64).reshape(3)
        for image in reconstruction.images.values()
        if image.has_pose
    }


def _gauge_anchor_frames(reconstruction: Any) -> tuple[int, int, float]:
    images = [image for image in reconstruction.images.values() if image.has_pose]
    if len(images) < 2:
        raise ValueError(
            "base reconstruction needs at least two registered posed images to fix BA gauge"
        )
    centers = np.asarray([image.projection_center() for image in images], dtype=np.float64)
    best_distance_sq = -1.0
    best_pair: tuple[Any, Any] | None = None
    for start in range(0, len(images), 256):
        deltas = centers[start : start + 256, None, :] - centers[None, :, :]
        distances_sq = np.einsum("ijk,ijk->ij", deltas, deltas)
        local = np.unravel_index(int(np.argmax(distances_sq)), distances_sq.shape)
        distance_sq = float(distances_sq[local])
        if distance_sq > best_distance_sq:
            best_distance_sq = distance_sq
            best_pair = (images[start + local[0]], images[local[1]])
    baseline = math.sqrt(max(0.0, best_distance_sq))
    if best_pair is None or not math.isfinite(baseline) or baseline <= 1e-6:
        raise ValueError("base reconstruction has no nonzero baseline to fix Sim3 gauge")
    if int(best_pair[0].frame_id) == int(best_pair[1].frame_id):
        # A shared rig frame has one world pose; choose a second frame with a
        # distinct center rather than pretending that its sensors fix scale.
        first = best_pair[0]
        candidates = [image for image in images if int(image.frame_id) != int(first.frame_id)]
        if not candidates:
            raise ValueError("base reconstruction has no independent frame for scale gauge")
        first_center = np.asarray(first.projection_center(), dtype=np.float64)
        second = max(
            candidates,
            key=lambda image: float(np.linalg.norm(image.projection_center() - first_center)),
        )
        baseline = float(np.linalg.norm(second.projection_center() - first_center))
        best_pair = (first, second)
    if baseline <= 1e-6:
        raise ValueError("selected base camera frames have a degenerate scale baseline")
    return int(best_pair[0].frame_id), int(best_pair[1].frame_id), baseline


def _fixed_k_global_ba(reconstruction: Any, filter_config: Any) -> dict[str, Any]:
    """Run fixed-calibration BA with two separated original poses fixing Sim3 gauge."""
    import pycolmap
    from sfm_diagnosis.site_pipeline.robust_filter import filter_reconstruction

    filtered_before = filter_reconstruction(reconstruction, filter_config)
    if reconstruction.num_points3D() == 0:
        raise ValueError("robust filtering removed every base and workmap point")
    anchor_frames = _gauge_anchor_frames(reconstruction)
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in sorted(reconstruction.images):
        config.add_image(int(image_id))
    for point_id in sorted(reconstruction.points3D):
        config.add_variable_point(int(point_id))
    for camera_id in sorted(reconstruction.cameras):
        config.set_constant_cam_intrinsics(int(camera_id))
    config.set_constant_rig_from_world_pose(anchor_frames[0])
    config.set_constant_rig_from_world_pose(anchor_frames[1])

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = True
    options.print_summary = True
    options.ceres.solver_options.max_num_iterations = int(
        filter_config.bundle_adjustment_iterations
    )
    options.ceres.solver_options.num_threads = int(filter_config.bundle_adjustment_threads)
    reconstruction.update_point_3d_errors()
    before_mean = float(reconstruction.compute_mean_reprojection_error())
    started = time.monotonic()
    adjuster = pycolmap.create_default_bundle_adjuster(options, config, reconstruction)
    summary = adjuster.solve()
    elapsed = time.monotonic() - started
    reconstruction.update_point_3d_errors()
    after_mean = float(reconstruction.compute_mean_reprojection_error())
    if not math.isfinite(before_mean) or not math.isfinite(after_mean):
        raise ValueError("global fixed-intrinsics BA returned a non-finite reprojection metric")
    if not summary.is_solution_usable() or int(summary.num_residuals) <= 0:
        raise RuntimeError("pycolmap global fixed-intrinsics BA did not solve measured residuals")
    ceres = summary.ceres_summary
    initial_cost = float(ceres.initial_cost)
    final_cost = float(ceres.final_cost)
    if not math.isfinite(initial_cost) or not math.isfinite(final_cost):
        raise RuntimeError(
            "pycolmap global BA did not report finite initial/final objective values"
        )
    if final_cost > initial_cost + max(1e-10, abs(initial_cost) * 1e-8):
        raise ValueError("global fixed-intrinsics BA increased the robust objective")
    filtered_after = filter_reconstruction(reconstruction, filter_config)
    return {
        "invoked": True,
        "solver": "pycolmap.create_default_bundle_adjuster",
        "global_registered_images": int(reconstruction.num_reg_images()),
        "global_points_before_filter": int(filtered_before["points_before"]),
        "global_points_after_filter": int(filtered_after["points_after"]),
        "observations_before_filter": int(filtered_before["observations_before"]),
        "observations_after_filter": int(filtered_after["observations_after"]),
        "first_filter": filtered_before,
        "second_filter": filtered_after,
        "fixed_camera_frames": [anchor_frames[0], anchor_frames[1]],
        "fixed_camera_baseline": float(anchor_frames[2]),
        "fixed_intrinsics": True,
        "iterations": int(filter_config.bundle_adjustment_iterations),
        "threads": int(filter_config.bundle_adjustment_threads),
        "elapsed_seconds": float(elapsed),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "objective_improvement": float(initial_cost - final_cost),
        "mean_reprojection_before_px": before_mean,
        "mean_reprojection_after_px": after_mean,
        "summary": {
            "num_residuals": int(summary.num_residuals),
            "termination_type": str(summary.termination_type),
            "successful_steps": int(ceres.num_successful_steps),
            "unsuccessful_steps": int(ceres.num_unsuccessful_steps),
            "total_time_seconds": float(ceres.total_time_in_seconds),
        },
    }


def _workmap_image_name(
    frame: _Keyframe, session: _Session, existing: set[str], assigned: Mapping[str, str]
) -> str:
    safe_name = Path(frame.name).name
    if not safe_name or safe_name in {".", ".."} or "/" in safe_name or "\\" in safe_name:
        raise ValueError(f"keyframe name {frame.name!r} cannot form a stable COLMAP image identity")
    image_name = f"workmap/{session.session_id[:8]}_{safe_name}.png"
    if image_name in existing or image_name in assigned.values():
        raise ValueError(f"new COLMAP image name collides with an existing image: {image_name}")
    return image_name


def _frame_measured_rows(frame: _Keyframe, session: _Session) -> tuple[list[int], np.ndarray]:
    observations: dict[int, np.ndarray] = {
        int(point_id): xy.copy()
        for point_id, xy in zip(frame.negative_ids, frame.negative_xy, strict=True)
    }
    for point_id, xy in session.positive_observations[frame.name]:
        if point_id in observations:
            raise ValueError(f"image {frame.name!r} repeats an old/new Point3D identity")
        observations[int(point_id)] = xy.copy()
    ordered_ids = sorted(observations)
    xy_rows = np.asarray(
        [observations[point_id] for point_id in ordered_ids], dtype=np.float64
    ).reshape(-1, 2)
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError(f"image {frame.name!r} has duplicate Point3D observations")
    for xy in xy_rows:
        if not (0.0 <= xy[0] < frame.width and 0.0 <= xy[1] < frame.height):
            raise ValueError(
                f"measured observation in {frame.name!r} falls outside its camera image"
            )
    return ordered_ids, xy_rows


def _attach_frame_tracks(
    reconstruction: Any,
    pycolmap: Any,
    image_id: int,
    ordered_ids: list[int],
    mapped_ids: Mapping[int, int],
    added_observation_tracks: dict[int, list[Any]],
) -> dict[int, int]:
    index_by_point = {point_id: index for index, point_id in enumerate(ordered_ids)}
    for point_id in ordered_ids:
        if point_id > 0:
            if not reconstruction.exists_point3D(point_id):
                raise ValueError(
                    f"measured old Point3D ID {point_id} is absent from the base model"
                )
            reconstruction.add_observation(
                point_id, pycolmap.TrackElement(image_id, index_by_point[point_id])
            )
        else:
            added_observation_tracks[mapped_ids[point_id]].append(
                pycolmap.TrackElement(image_id, index_by_point[point_id])
            )
    return index_by_point


def _add_workmap_points(
    reconstruction: Any,
    pycolmap: Any,
    mapped_ids: Mapping[int, int],
    added_observation_tracks: Mapping[int, list[Any]],
    triangulated: Mapping[int, np.ndarray],
    alignment: Mapping[str, Any],
) -> None:
    for negative_id in sorted(mapped_ids):
        observations = added_observation_tracks[mapped_ids[negative_id]]
        image_ids_seen = [int(element.image_id) for element in observations]
        if len(image_ids_seen) < 2 or len(set(image_ids_seen)) != len(image_ids_seen):
            raise ValueError(
                f"new landmark {negative_id} lacks distinct measured COLMAP image tracks"
            )
        xyz = _sim3_transform_points(triangulated[negative_id][None, :], alignment)[0]
        point = pycolmap.Point3D(
            xyz=np.asarray(xyz, dtype=np.float64),
            track=pycolmap.Track(observations),
            color=np.zeros(3, dtype=np.uint8),
        )
        reconstruction.add_point3D_with_id(mapped_ids[negative_id], point)


def _add_session_geometry(
    reconstruction: Any,
    session: _Session,
    triangulated: Mapping[int, np.ndarray],
    alignment: Mapping[str, Any],
    old_model_point_ids: set[int],
) -> tuple[dict[int, int], dict[str, str], dict[str, dict[int, int]]]:
    import pycolmap

    mapped_ids = {
        negative_id: max(old_model_point_ids, default=0) + index + 1
        for index, negative_id in enumerate(sorted(triangulated))
    }
    if (
        len(set(mapped_ids.values())) != len(mapped_ids)
        or set(mapped_ids.values()) & old_model_point_ids
    ):
        raise RuntimeError(
            "deterministic workmap Point3D ID allocation collided with the base model"
        )
    base_camera_id = (
        max(
            [int(value) for value in reconstruction.cameras]
            + [int(value) for value in reconstruction.rigs]
            + [0]
        )
        + 1
    )
    base_image_id = (
        max(
            [int(value) for value in reconstruction.images]
            + [int(value) for value in reconstruction.frames]
            + [0]
        )
        + 1
    )
    image_names: dict[str, str] = {}
    point2d_index: dict[str, dict[int, int]] = {}
    existing_image_names = {str(image.name) for image in reconstruction.images.values()}
    added_observation_tracks: dict[int, list[Any]] = {
        int(value): [] for value in mapped_ids.values()
    }

    for frame_index, frame in enumerate(session.keyframes):
        image_name = _workmap_image_name(frame, session, existing_image_names, image_names)
        image_id = base_image_id + frame_index
        camera_id = base_camera_id + frame_index
        camera = pycolmap.Camera.create_from_model_name(
            camera_id, "PINHOLE", float(frame.K[0, 0]), frame.width, frame.height
        )
        camera.params = np.asarray(
            [frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]], dtype=np.float64
        )
        reconstruction.add_camera_with_trivial_rig(camera)
        ordered_ids, xy_rows = _frame_measured_rows(frame, session)
        pose = _sim3_transform_pose(frame.pose, alignment)
        image = pycolmap.Image(
            name=image_name,
            keypoints=xy_rows,
            camera_id=camera_id,
            image_id=image_id,
        )
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pose))
        image_names[frame.name] = image_name
        point2d_index[frame.name] = _attach_frame_tracks(
            reconstruction, pycolmap, image_id, ordered_ids, mapped_ids, added_observation_tracks
        )

    _add_workmap_points(
        reconstruction, pycolmap, mapped_ids, added_observation_tracks, triangulated, alignment
    )
    return mapped_ids, image_names, point2d_index


def _point_image_components(
    reconstruction: Any, base_names: set[str], new_names: Sequence[str]
) -> list[str]:
    image_ids = {int(image.image_id): str(image.name) for image in reconstruction.images.values()}
    parent = {image_id: image_id for image_id in image_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for point in reconstruction.points3D.values():
        observed = [int(element.image_id) for element in point.track.elements]
        if observed:
            for image_id in observed[1:]:
                union(observed[0], image_id)
    base_roots = {find(image_id) for image_id, name in image_ids.items() if name in base_names}
    unanchored = []
    for name in new_names:
        matches = [image_id for image_id, image_name in image_ids.items() if image_name == name]
        if len(matches) != 1 or find(matches[0]) not in base_roots:
            unanchored.append(name)
    return unanchored


def _atomic_write_reconstruction(reconstruction: Any, output_model: Path) -> None:
    output = output_model.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent))
    try:
        reconstruction.write(str(temporary))
        for path in temporary.iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        dir_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _runtime_provenance(
    assets: Any, profile: Any, runtime_paths: Any, session: _Session
) -> dict[str, Any]:
    import cv2
    import pycolmap
    import torch
    import direct_map
    import direct_profile
    from river_map_quality import official_edm_adapter
    from sfm_diagnosis import sim3
    from sfm_diagnosis.site_pipeline import deployment_localizer, robust_filter

    source_paths = {
        "reconstruction_tool": Path(__file__).resolve(),
        "direct_map": Path(direct_map.__file__).resolve(),
        "direct_profile": Path(direct_profile.__file__).resolve(),
        "live_provider": (_DEPLOY_DIR / "live_provider.py").resolve(),
        "workmap_overlap": (_DEPLOY_DIR / "workmap_overlap.py").resolve(),
        "official_edm_adapter": Path(official_edm_adapter.__file__).resolve(),
        "deployment_localizer_match_validation": Path(deployment_localizer.__file__).resolve(),
        "robust_filter": Path(robust_filter.__file__).resolve(),
        "sim3": Path(sim3.__file__).resolve(),
    }
    return {
        "implementation": "LiveMapEDMProvider.match_workmap_overlap + reciprocal fixed-pair EDM",
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "runtime_versions": {
            "pycolmap": str(getattr(pycolmap, "__version__", "unknown")),
            "opencv": str(cv2.__version__),
            "torch": str(torch.__version__),
        },
        "base_bundle_sha256": str(assets.sha256),
        "base_model_sha256": dict(assets.raw["model_sha256"]),
        "base_reference_manifest_sha256": sha256_file(assets.reference_manifest),
        "edm_checkpoint_sha256": sha256_file(runtime_paths.edm_checkpoint),
        "boq_checkpoint_sha256": sha256_file(runtime_paths.boq_weights),
        "profile_sha256": str(profile.sha256),
        "session_id": session.session_id,
        "session_input_sha256": dict(session.input_hashes),
        "matcher": {
            "reciprocal_tolerance_px": min(2.0, float(profile.reloc.lift_distance_px)),
            "observation_lift_distance_px": float(profile.reloc.lift_distance_px),
            "strict_two_view_reprojection_px": 3.0,
            "minimum_reciprocal_matches": 8,
            "minimum_two_view_inliers": 6,
            "new_old_candidates": int(profile.reloc.top_k),
        },
    }


def _prepare_reconstruction_inputs(
    base_model: Path, session_snapshot: Any, session_dir: Path, output_model: Path
) -> tuple[Path, Any, Any, str, _Session]:
    if not isinstance(session_snapshot, Mapping):
        raise ValueError("session snapshot must be a mapping")
    base_path = base_model.expanduser().resolve(strict=True)
    output_path = output_model.expanduser().absolute()
    output_resolved = output_path.resolve(strict=False)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    assets, profile = _resolve_base_release(base_path)
    base_bundle_sha = str(assets.sha256)
    if session_snapshot.get("base_bundle_sha256") != base_bundle_sha:
        raise ValueError("session base bundle SHA does not match the selected immutable release")
    session_root = Path(session_dir).expanduser().resolve(strict=True)
    release_root = assets.root.parent.resolve()
    if output_resolved.is_relative_to(release_root):
        raise ValueError("optimized output must not be written inside the frozen base release")
    if output_resolved.is_relative_to(session_root):
        raise ValueError("optimized output must not be written into or mutate the source session")
    session = _parse_session(
        session_snapshot,
        session_dir,
        assets.map_revision_id,
        assets.coordinate_frame_id,
        base_bundle_sha,
    )
    return base_path, assets, profile, base_bundle_sha, session


def _open_base_reconstruction(pycolmap: Any, base_path: Path) -> tuple[Any, set[str]]:
    reconstruction = pycolmap.Reconstruction(str(base_path))
    base_names = {str(image.name) for image in reconstruction.images.values()}
    if len(base_names) != len(reconstruction.images):
        raise ValueError("base model has duplicate registered image names")
    if (
        reconstruction.num_points3D() <= 0
        or reconstruction.num_reg_images() <= 0
        or int(reconstruction.num_reg_images()) != len(base_names)
    ):
        raise ValueError("base model contains no complete registered geometry")
    return reconstruction, base_names


def _edm_validation(
    assets: Any,
    profile: Any,
    session: _Session,
    triangulated: Mapping[int, np.ndarray],
    reconstruction: Any,
    components: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, list[dict[str, Any]], dict[str, Any]]:
    # No database is reconstructed for the immutable old map: its original measured
    # tracks are retained.  EDM is run only for new↔old overlap and covisible new↔new.
    from live_provider import LiveMapEDMProvider

    runtime_paths = default_runtime_paths()
    shm = Path("/dev/shm")
    temporary_root = tempfile.TemporaryDirectory(
        prefix="workmap_reconstruct_",
        dir=str(shm) if shm.is_dir() and os.access(shm, os.W_OK) else None,
    )
    provider = None
    try:
        provider = LiveMapEDMProvider(
            assets,
            profile,
            edm_repo=runtime_paths.edm_repo,
            edm_checkpoint=runtime_paths.edm_checkpoint,
            boq_weights=runtime_paths.boq_weights,
            cache_dir=Path(temporary_root.name) / "runtime_cache",
        )
        provider.ensure_models()
        if hasattr(provider, "_workmap_session_id"):
            provider._workmap_session_id = session.session_id
        overlap_rows, overlap = _make_overlap_evidence(
            provider, session, triangulated, reconstruction
        )
        component_anchors = _validate_component_anchors(session, components, overlap_rows)
        new_new_count, new_new_edges = _validate_new_new_edm(provider, session)
        if new_new_count <= 0:
            raise ValueError("no new-to-new measured landmarks passed reciprocal EDM validation")
        edm_provenance = _runtime_provenance(assets, profile, runtime_paths, session)
    finally:
        if provider is not None:
            provider.close()
        temporary_root.cleanup()
    return overlap, component_anchors, new_new_count, new_new_edges, edm_provenance


def _apply_original_gauge(
    pycolmap: Any,
    reconstruction: Any,
    original_centers: Mapping[str, np.ndarray],
    camera_calibrations_before_ba: Mapping[int, Any],
) -> dict[str, Any]:
    camera_centers_after_ba = _base_camera_centers(reconstruction)
    gauge = _fit_original_gauge(camera_centers_after_ba, original_centers)
    pycolmap_transform = pycolmap.Sim3d(
        float(gauge["scale"]),
        pycolmap.Rotation3d(np.asarray(gauge["rotation"], dtype=np.float64)),
        np.asarray(gauge["translation"], dtype=np.float64),
    )
    reconstruction.transform(pycolmap_transform)
    if _camera_calibrations(reconstruction) != camera_calibrations_before_ba:
        raise ValueError("global fixed-intrinsics BA changed one or more camera calibrations")
    return gauge


def _check_optimized_model(
    reconstruction: Any,
    session: _Session,
    image_names: Mapping[str, str],
    base_names: set[str],
    mapped_ids: Mapping[int, int],
) -> dict[str, Any]:
    from sfm_diagnosis.site_pipeline.robust_filter import reconstruction_metrics

    for frame in session.keyframes:
        model_image_name = image_names[frame.name]
        image = reconstruction.find_image_with_name(model_image_name)
        if image is None or int(image.num_points3D) < 2:
            raise ValueError(
                f"global filtering left fewer than two actual tracks in {frame.name!r}"
            )
    unanchored = _point_image_components(reconstruction, base_names, list(image_names.values()))
    if unanchored:
        raise ValueError(
            f"new registered image components are not connected to the unchanged base model: {unanchored[:8]}"
        )
    final_metrics = reconstruction_metrics(reconstruction)
    if final_metrics["registered_images"] != len(base_names) + len(session.keyframes):
        raise ValueError("optimized model does not contain every original and new registered image")
    point_ids = {int(point_id) for point_id in reconstruction.points3D}
    if any(point_id <= 0 for point_id in point_ids):
        raise ValueError("optimized model contains a non-positive COLMAP Point3D ID")
    missing_new_points = sorted(set(mapped_ids.values()) - point_ids)
    if missing_new_points:
        raise ValueError(
            f"global filtering removed newly reconstructed points: {missing_new_points[:8]}"
        )
    return final_metrics


def _gauge_receipt(overlap: Mapping[str, Any], gauge: Mapping[str, Any]) -> dict[str, Any]:
    fit_ids = [int(value) for value in overlap["fit_ids"]]
    holdout_ids = [int(value) for value in overlap["holdout_ids"]]
    if len(fit_ids) < 6 or len(holdout_ids) < 6 or set(fit_ids) & set(holdout_ids):
        raise ValueError(
            "EDM Sim3 gauge fit and holdout landmark IDs are not disjoint with six each"
        )
    return {
        "pass": True,
        "scale": float(overlap["scale"]),
        "rotation": overlap["rotation"],
        "translation": overlap["translation"],
        "fit": {
            "landmark_ids": fit_ids,
            "count": len(fit_ids),
            "residual": float(overlap["fit_residual"]),
        },
        "holdout": {
            "landmark_ids": holdout_ids,
            "count": len(holdout_ids),
            "residuals_3d": overlap["holdout_after"],
            "residual_p90_3d": float(overlap["after_3d_p90"]),
            "residual_p90_2d_before_px": float(overlap["before_2d_p90"]),
            "residual_p90_2d_after_px": float(overlap["after_2d_p90"]),
        },
        "fit_holdout_source_keyframes": overlap["source_keyframes"],
        "initial_workmap_alignment": {
            "fit_holdout_landmark_ids_disjoint": True,
            "holdout_before_3d": overlap["holdout_before"],
            "holdout_after_3d": overlap["holdout_after"],
            "before_3d_p90": float(overlap["before_3d_p90"]),
            "after_3d_p90": float(overlap["after_3d_p90"]),
            "before_2d_p90_px": float(overlap["before_2d_p90"]),
            "after_2d_p90_px": float(overlap["after_2d_p90"]),
            "post_ba_original_image_fit_holdout": gauge,
        },
    }


def reconstruct_workmap(
    base_model: Path,
    session_snapshot: Mapping[str, Any],
    session_dir: Path,
    output_model: Path,
) -> dict[str, Any]:
    """Merge measured workmap tracks into a copy of the immutable base COLMAP model.

    The returned receipt is JSON-serializable.  This function never writes the
    base model and never exposes a partial output directory when any validation,
    EDM, gauge, filtering, BA, or connectivity check rejects the candidate.
    """
    import pycolmap
    from sfm_diagnosis.site_pipeline.robust_filter import (
        RobustFilterConfig,
        reconstruction_metrics,
    )

    base_path, assets, profile, base_bundle_sha, session = _prepare_reconstruction_inputs(
        base_model, session_snapshot, session_dir, output_model
    )
    input_landmarks = len(session.points_xyz)
    input_observations = sum(len(items) for items in session.negative_observations.values())
    triangulated, triangulation_metrics, rejected_tracks = _triangulate_session(session)
    session = _retain_triangulated(session, triangulated)
    components = _frame_components(session)
    reconstruction, base_names = _open_base_reconstruction(pycolmap, base_path)
    original_centers = _base_camera_centers(reconstruction)
    before_metrics = reconstruction_metrics(reconstruction)
    overlap, component_anchors, new_new_count, new_new_edges, edm_provenance = _edm_validation(
        assets, profile, session, triangulated, reconstruction, components
    )

    mapped_ids, image_names, _point2d_index = _add_session_geometry(
        reconstruction,
        session,
        triangulated,
        overlap["transform"],
        {int(point_id) for point_id in reconstruction.points3D},
    )
    camera_calibrations_before_ba = _camera_calibrations(reconstruction)
    config = RobustFilterConfig(
        max_reprojection_error_px=_MAX_FILTER_REPROJECTION_PX,
        minimum_triangulation_angle_deg=_MIN_PARALLAX_DEG,
        minimum_track_length=2,
        bundle_adjustment_iterations=100,
        bundle_adjustment_threads=min(8, max(1, os.cpu_count() or 1)),
    )
    ba = _fixed_k_global_ba(reconstruction, config)
    gauge = _apply_original_gauge(
        pycolmap, reconstruction, original_centers, camera_calibrations_before_ba
    )
    final_metrics = _check_optimized_model(
        reconstruction, session, image_names, base_names, mapped_ids
    )
    gauge_receipt = _gauge_receipt(overlap, gauge)
    receipt = {
        "schema": "workmap-reconstruction/v1",
        "base_model": str(base_path),
        "base_bundle_sha256": base_bundle_sha,
        "base_release_id": assets.map_revision_id,
        "coordinate_frame_id": assets.coordinate_frame_id,
        "source_epoch": session.source_epoch,
        "session_id": session.session_id,
        "session_snapshot_revision": _integer(
            session_snapshot.get("revision"), "snapshot revision", minimum=0
        ),
        "output_model": str(output_model.expanduser().absolute()),
        "negative_id_map": {str(key): int(value) for key, value in sorted(mapped_ids.items())},
        "keyframe_image_names": dict(sorted(image_names.items())),
        "anchored_keyframe_image_names": sorted(image_names.values()),
        "anchored_components": component_anchors,
        "gauge": gauge_receipt,
        "edm_validation": {
            "new_old_pairs": int(overlap["unique_new_old_pairs"]),
            "new_old_evidence_rows": int(overlap["verified_rows"]),
            "new_old_source_keyframes": int(overlap["source_keyframe_count"]),
            "new_new_pairs": int(new_new_count),
            "new_new_edges": new_new_edges,
            "provenance": edm_provenance,
            "old_reference_names": overlap["old_reference_names"],
        },
        "triangulation": {
            "input_negative_landmarks": input_landmarks,
            "triangulated_negative_landmarks": len(triangulated),
            "rejected_tracks_by_reason": rejected_tracks,
            "input_measured_keyframe_observations": input_observations,
            "distinct_measured_keyframe_observations": int(
                sum(len(items) for items in session.negative_observations.values())
            ),
            "parallax_deg_p50": float(
                np.percentile([row["parallax_deg"] for row in triangulation_metrics.values()], 50.0)
            ),
            "reprojection_p90_px_p90": float(
                np.percentile(
                    [row["reprojection_p90_px"] for row in triangulation_metrics.values()], 90.0
                )
            ),
        },
        "filtering": {
            "max_reprojection_error_px": config.max_reprojection_error_px,
            "minimum_triangulation_angle_deg": config.minimum_triangulation_angle_deg,
            "minimum_track_length": config.minimum_track_length,
            "before": ba["first_filter"],
            "after": ba["second_filter"],
            "global_model_before": before_metrics,
            "global_model_after": final_metrics,
            "new_images_connected_to_base": len(image_names),
        },
        "bundle_adjustment": ba,
    }
    _atomic_write_reconstruction(reconstruction, output_model)
    return receipt


__all__ = ["reconstruct_workmap"]
