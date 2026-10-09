#!/usr/bin/env python3
"""Assemble a geometry-consistent deploy pack from a completed online workmap."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

try:
    from tools.workmap_reconstruction import _CaptureIndex, resolve_tracked_capture
except ImportError:
    from workmap_reconstruction import _CaptureIndex, resolve_tracked_capture

MODEL_BINS = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")
FRAME_DIGEST_BINS = ("cameras.bin", "images.bin", "points3D.bin")
BUNDLE_SCHEMA = "direct-localization-bundle/v1"
INTERSECTION_SCHEMA = "fim-lwtl-scored-grid/v1"
BOQ_DIM = 16384
DEPLOY_DIR = Path(__file__).resolve().parents[1] / "定位演算法/deploy_code_expand/sfm_direct_deploy"


class WorkmapPackError(RuntimeError):
    """A workmap pack cannot be assembled without violating its asset contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkmapPackError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkmapPackError(f"{label} must contain a JSON object: {path}")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(value, dict):
                    raise WorkmapPackError(f"{label} row {line_no} must be a JSON object")
                rows.append(value)
    except WorkmapPackError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkmapPackError(f"invalid {label} JSONL {path}: {exc}") from exc
    return rows


def _safe_relative(raw: Any, *, label: str, components: int | None = None) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise WorkmapPackError(f"{label} must be a non-empty POSIX relative path")
    parsed = PurePosixPath(raw)
    if parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
        raise WorkmapPackError(f"unsafe {label}: {raw!r}")
    if components is not None and len(parsed.parts) != components:
        raise WorkmapPackError(f"{label} must have exactly {components} path components: {raw!r}")
    return parsed.as_posix()


def _ensure_regular_file(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise WorkmapPackError(f"{label} must be a regular non-symlink file: {candidate}")
    return candidate


def _ensure_no_symlink_tree(root: Path, *, label: str) -> list[Path]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise WorkmapPackError(f"{label} must be a directory without symlink root: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WorkmapPackError(f"{label} must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise WorkmapPackError(f"{label} contains a non-regular entry: {path}")
    return files


def _identity_from_keyframe(row: Mapping[str, Any], *, label: str) -> str:
    output_name = row.get("output_name")
    if isinstance(output_name, str) and output_name.count("/") == 1:
        return _safe_relative(output_name, label=f"{label} output_name", components=2)
    uri = row.get("image_uri")
    if not isinstance(uri, str) or not uri:
        raise WorkmapPackError(f"{label} has neither output_name nor image_uri")
    parts = PurePosixPath(uri.replace("\\", "/")).parts
    if len(parts) < 2:
        raise WorkmapPackError(f"{label} image_uri cannot identify an image: {uri!r}")
    return _safe_relative("/".join(parts[-2:]), label=f"{label} image identity", components=2)


def _open_base_bundle(base_release: Path) -> tuple[dict[str, Any], Path, Path, str]:
    raw_release = Path(base_release).expanduser()
    if raw_release.is_symlink():
        raise WorkmapPackError(f"base release must not be a symlink: {raw_release}")
    release = raw_release.resolve(strict=True)
    if not release.is_dir():
        raise WorkmapPackError(f"base release must be a directory: {release}")
    bundle_path = _ensure_regular_file(
        release / "localization/direct_bundle.json", label="base bundle"
    )
    bundle_bytes = bundle_path.read_bytes()
    try:
        bundle = json.loads(bundle_bytes, parse_constant=_reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkmapPackError(f"invalid base direct bundle: {bundle_path}: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise WorkmapPackError(f"base bundle schema must be {BUNDLE_SCHEMA!r}: {bundle_path}")
    bundle_sha = _bytes_sha256(bundle_bytes)
    if not isinstance(bundle.get("map_revision_id"), str) or not bundle["map_revision_id"]:
        raise WorkmapPackError("base bundle has no map_revision_id")
    if not isinstance(bundle.get("coordinate_frame_id"), str) or not bundle["coordinate_frame_id"]:
        raise WorkmapPackError("base bundle has no coordinate_frame_id")
    return bundle, release, bundle_path, bundle_sha


def _verify_base_model_digests(bundle: Mapping[str, Any], release: Path) -> None:
    declared_model = bundle.get("model_sha256")
    if not isinstance(declared_model, dict):
        raise WorkmapPackError("base bundle has no model_sha256 map")
    base_model = release / "model"
    for name in FRAME_DIGEST_BINS:
        path = _ensure_regular_file(base_model / name, label=f"base model {name}")
        if declared_model.get(name) != _sha256(path):
            raise WorkmapPackError(f"base bundle model digest mismatch: {name}")


def _verify_base_bundle_file(entry: Any, bundle_path: Path, release: Path) -> None:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise WorkmapPackError("base bundle has a malformed files entry")
    raw_path = Path(entry["path"])
    if raw_path.is_absolute():
        raise WorkmapPackError(f"base bundle file path must be relative: {entry['path']}")
    lexical = bundle_path.parent
    for part in raw_path.parts:
        if part == "..":
            lexical = lexical.parent
        elif part not in ("", "."):
            lexical = lexical / part
        if lexical.is_symlink():
            raise WorkmapPackError(f"base release assets must not traverse symlinks: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(release)
    except ValueError as exc:
        raise WorkmapPackError(f"base bundle file escapes release: {entry['path']}") from exc
    if not resolved.is_file():
        raise WorkmapPackError(f"base bundle asset is missing: {resolved}")
    if resolved.stat().st_size != entry.get("size_bytes") or _sha256(resolved) != entry.get(
        "sha256"
    ):
        raise WorkmapPackError(f"base bundle file digest/size mismatch: {entry['path']}")


def _base_bundle(base_release: Path) -> tuple[dict[str, Any], Path, str]:
    bundle, release, bundle_path, bundle_sha = _open_base_bundle(base_release)
    _verify_base_model_digests(bundle, release)
    entries = bundle.get("files")
    if not isinstance(entries, list):
        raise WorkmapPackError("base bundle files must be a list")
    for entry in entries:
        _verify_base_bundle_file(entry, bundle_path, release)
    return bundle, release, bundle_sha


def _session_paths(session_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    raw_session = Path(session_dir).expanduser()
    if raw_session.is_symlink():
        raise WorkmapPackError(f"session root must not be a symlink: {raw_session}")
    session = raw_session.resolve(strict=True)
    if not session.is_dir():
        raise WorkmapPackError(f"session must be a directory: {session}")
    meta_path = _ensure_regular_file(session / "session_meta.json", label="session metadata")
    snapshot_path = _ensure_regular_file(session / "snapshot.json", label="final session snapshot")
    keyframes_path = _ensure_regular_file(session / "keyframes.jsonl", label="session keyframes")
    observations_path = _ensure_regular_file(
        session / "observations.jsonl", label="session observations"
    )
    return session, meta_path, snapshot_path, keyframes_path, observations_path


def _check_session_context(
    meta: Mapping[str, Any], snapshot: Mapping[str, Any], bundle: Mapping[str, Any], bundle_sha: str
) -> None:
    required_context = {
        "base_release_id": bundle.get("map_revision_id"),
        "coordinate_frame_id": bundle.get("coordinate_frame_id"),
    }
    for source_name, source in (("metadata", meta), ("snapshot", snapshot)):
        for key, expected in required_context.items():
            if source.get(key) != expected:
                raise WorkmapPackError(
                    f"session {source_name} {key} does not match the bound base release: "
                    f"{source.get(key)!r} != {expected!r}"
                )
        if source.get("base_bundle_sha256") != bundle_sha:
            raise WorkmapPackError(
                f"session {source_name} is not bound to the exact base direct_bundle.json SHA-256"
            )


def _check_session_completion(meta: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
    if not isinstance(meta.get("session_id"), str) or not meta["session_id"]:
        raise WorkmapPackError("session metadata has no session_id")
    if snapshot.get("session_id") != meta["session_id"]:
        raise WorkmapPackError("session metadata and final snapshot session_id differ")
    if (
        meta.get("schema") != "direct-online-workmap-session/v1"
        or snapshot.get("schema") != "direct-online-workmap-session/v1"
    ):
        raise WorkmapPackError(
            "session metadata/snapshot schema is not a completed direct workmap session"
        )
    if not meta.get("finished_unix_s") or not snapshot.get("finished_unix_s"):
        raise WorkmapPackError(
            "session metadata and snapshot must record a completed writer shutdown"
        )
    if meta.get("writer_healthy") is not True:
        raise WorkmapPackError("session writer is not marked healthy")
    if not isinstance(snapshot.get("keyframes"), dict):
        raise WorkmapPackError("final session snapshot has no keyframe table")


def _check_session_keyframes(
    keyframes: list[dict[str, Any]], snapshot: Mapping[str, Any]
) -> set[str]:
    if not keyframes:
        raise WorkmapPackError("session has no durable keyframe records")
    seen: set[str] = set()
    for row in keyframes:
        name = row.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise WorkmapPackError(f"session keyframe name is missing or duplicated: {name!r}")
        seen.add(name)
        if row.get("durable_complete") is not True:
            raise WorkmapPackError(f"session keyframe is not durably complete: {name}")
        final_row = snapshot["keyframes"].get(name)
        if not isinstance(final_row, dict) or final_row.get("durable") is not True:
            raise WorkmapPackError(f"final snapshot lacks a durable keyframe: {name}")
        if final_row.get("image_size") != row.get("image_size") or final_row.get("K") != row.get(
            "K"
        ):
            raise WorkmapPackError(
                f"session keyframe metadata disagrees with final snapshot: {name}"
            )
    if set(snapshot["keyframes"]) != seen:
        raise WorkmapPackError(
            "final snapshot and durable keyframe manifest contain different keyframes"
        )
    return seen


def _observation_measurement(row: Mapping[str, Any], line_no: int) -> tuple[int, tuple[float, ...]]:
    try:
        landmark_id = int(row["landmark_id"])
        xy = tuple(float(value) for value in row["xy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkmapPackError(f"malformed measured observation at row {line_no}") from exc
    if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
        raise WorkmapPackError(f"non-finite measured observation at row {line_no}")
    return landmark_id, xy


def _resolve_pack_tracked_row(
    index: _CaptureIndex, row: Mapping[str, Any], line_no: int
) -> str | None:
    raw_ordinal = row.get("ordinal")
    raw_stamp = row.get("stamp")
    raw_pose = row.get("cam_from_world")
    if raw_ordinal is None or raw_stamp is None or raw_pose is None:
        return None
    try:
        obs_ord = int(raw_ordinal)
        obs_stamp = float(raw_stamp)
        raw_epoch = row.get("source_epoch")
        obs_epoch = int(raw_epoch) if raw_epoch is not None else None
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkmapPackError(f"malformed tracked observation capture at row {line_no}") from exc
    if not math.isfinite(obs_stamp):
        raise WorkmapPackError(f"non-finite tracked observation stamp at row {line_no}")
    try:
        pose = np.asarray(raw_pose, dtype=np.float64)
        if pose.shape != (3, 4) or not np.isfinite(pose).all():
            raise WorkmapPackError(f"invalid pose at row {line_no}")
        return resolve_tracked_capture(
            index,
            obs_ord,
            obs_stamp,
            pose,
            source_epoch=obs_epoch,
            expected_name=row.get("keyframe_name"),
        )
    except ValueError as exc:
        raise WorkmapPackError(f"tracked observation at row {line_no}: {exc}") from exc


_VALID_OBSERVATION_SOURCES = frozenset(
    {
        "keyframe_image_measurement",
        "frozen_map_keyframe_image_measurement",
        "tracked_image_measurement",
    }
)


def _check_pack_keyframe_row(
    index: _CaptureIndex | None, name: str, row: Mapping[str, Any], line_no: int
) -> None:
    if index is None or not index.has_name(name):
        return
    entry = index.get_entry(name)
    if "ordinal" in row and int(row["ordinal"]) != entry.ordinal:
        raise WorkmapPackError(
            f"observation ordinal conflicts with keyframe {name} at row {line_no}"
        )
    if "source_epoch" in row and int(row["source_epoch"]) != entry.source_epoch:
        raise WorkmapPackError(
            f"observation source_epoch conflicts with keyframe {name} at row {line_no}"
        )


def _process_pack_observation(
    row: Mapping[str, Any],
    line_no: int,
    seen: set[str],
    capture_index: _CaptureIndex | None,
) -> tuple[str, int, tuple[float, float]] | None:
    source = row.get("observation_source")
    if source not in _VALID_OBSERVATION_SOURCES:
        return None
    landmark_id, xy = _observation_measurement(row, line_no)
    if source == "tracked_image_measurement":
        if landmark_id >= 0:
            raise WorkmapPackError(
                f"tracked-image observation has non-negative landmark ID at row {line_no}"
            )
        if capture_index is None:
            return None
        name = _resolve_pack_tracked_row(capture_index, row, line_no)
        return (name, landmark_id, xy) if name is not None else None
    name = row.get("keyframe_name")
    if not isinstance(name, str) or name not in seen:
        return None
    _check_pack_keyframe_row(capture_index, name, row, line_no)
    return (name, landmark_id, xy)


def _build_pack_capture_index(
    keyframes: list[dict[str, Any]] | None, seen: set[str]
) -> _CaptureIndex | None:
    if keyframes is None:
        return None
    try:
        return _CaptureIndex(
            {str(row["name"]): row for row in keyframes if row.get("name") in seen}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkmapPackError(f"invalid session keyframe captures: {exc}") from exc


def _measured_session_observations(
    observations_path: Path,
    seen: set[str],
    keyframes: list[dict[str, Any]] | None = None,
) -> dict[str, dict[int, tuple[float, float]]]:
    measured_rows: dict[str, dict[int, tuple[float, float]]] = {name: {} for name in seen}
    capture_index = _build_pack_capture_index(keyframes, seen)
    try:
        with observations_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(row, dict):
                    raise WorkmapPackError(f"session observations row {line_no} is not an object")
                parsed = _process_pack_observation(row, line_no, seen, capture_index)
                if parsed is None:
                    continue
                name, landmark_id, xy = parsed
                previous = measured_rows[name].get(landmark_id)
                if previous is not None and not np.allclose(previous, xy, rtol=0.0, atol=1e-4):
                    raise WorkmapPackError(
                        f"conflicting true measurements for landmark {landmark_id} in {name}"
                    )
                measured_rows[name][landmark_id] = (xy[0], xy[1])
    except WorkmapPackError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkmapPackError(f"invalid session observations: {exc}") from exc
    return measured_rows


def _merge_keyframe_measurement(
    measured: dict[int, tuple[float, float]], name: str, raw_id: Any, raw_xy: Any
) -> None:
    try:
        landmark_id = int(raw_id)
        xy = tuple(float(value) for value in raw_xy)
    except (TypeError, ValueError) as exc:
        raise WorkmapPackError(f"malformed measured keyframe observation: {name}") from exc
    if landmark_id == 0 or len(xy) != 2 or not all(math.isfinite(value) for value in xy):
        raise WorkmapPackError(f"invalid measured keyframe observation: {name}")
    previous = measured.get(landmark_id)
    if previous is not None and not np.allclose(previous, xy, rtol=0.0, atol=1e-4):
        raise WorkmapPackError(
            f"keyframe snapshot conflicts with measured observation {landmark_id} in {name}"
        )
    if previous is None:
        measured[landmark_id] = (xy[0], xy[1])


def _merge_keyframe_measurements(
    keyframes: list[dict[str, Any]], measured_rows: dict[str, dict[int, tuple[float, float]]]
) -> None:
    for keyframe_row in keyframes:
        name = str(keyframe_row["name"])
        point_ids = keyframe_row.get("all_ids")
        measured_xy = keyframe_row.get("all_xy")
        if point_ids is None and measured_xy is None:
            point_ids = keyframe_row.get("ids")
            measured_xy = keyframe_row.get("xy")
        if point_ids is None and measured_xy is None:
            continue
        if (
            not isinstance(point_ids, list)
            or not isinstance(measured_xy, list)
            or len(point_ids) != len(measured_xy)
        ):
            raise WorkmapPackError(f"session keyframe measured ID/XY arrays disagree: {name}")
        for raw_id, raw_xy in zip(point_ids, measured_xy, strict=True):
            _merge_keyframe_measurement(measured_rows[name], name, raw_id, raw_xy)


def _validate_session(
    session_dir: Path, bundle: Mapping[str, Any], bundle_sha: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[int, tuple[float, float]]],
    dict[str, Any],
]:
    session, meta_path, snapshot_path, keyframes_path, observations_path = _session_paths(
        session_dir
    )
    meta = _read_json(meta_path, label="session metadata")
    snapshot = _read_json(snapshot_path, label="session snapshot")
    _check_session_context(meta, snapshot, bundle, bundle_sha)
    _check_session_completion(meta, snapshot)
    keyframes = _read_jsonl(keyframes_path, label="session keyframes")
    seen = _check_session_keyframes(keyframes, snapshot)

    keyframe_image_hashes = {
        str(row["name"]): _sha256(
            _session_image_path(session, row.get("image_path"), name=str(row["name"]))
        )
        for row in keyframes
    }
    measured_rows = _measured_session_observations(observations_path, seen, keyframes=keyframes)
    _merge_keyframe_measurements(keyframes, measured_rows)
    source_count = {name: len(measured_rows[name]) for name in seen}
    if not any(source_count.values()):
        raise WorkmapPackError("session has no true keyframe image measurements")

    return (
        meta,
        snapshot,
        keyframes,
        measured_rows,
        {
            "meta_path": meta_path,
            "snapshot_path": snapshot_path,
            "keyframes_path": keyframes_path,
            "observations_path": observations_path,
            "measured_row_counts": source_count,
            "directory": session,
            "keyframe_images_sha256": keyframe_image_hashes,
        },
    )


def _copy_file(source: Path, target: Path, *, label: str) -> str:
    source = _ensure_regular_file(source, label=f"source {label}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    source_stat, target_stat = source.stat(), target.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (target_stat.st_dev, target_stat.st_ino):
        raise WorkmapPackError(f"pack must own a non-shared copy of {label}: {target}")
    source_sha = _sha256(source)
    target_sha = _sha256(target)
    if source_sha != target_sha or source_stat.st_size != target_stat.st_size:
        raise WorkmapPackError(f"copied {label} differs from its verified source: {source}")
    return target_sha


def _npy_header(stream: Any, path: Path, label: str) -> tuple[Any, bool, Any]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version in ((2, 0), (3, 0)):
        return np.lib.format.read_array_header_2_0(stream)
    raise WorkmapPackError(f"unsupported {label} NPY version {version}: {path}")


def _read_npz_headers(path: Path) -> tuple[tuple[int, ...], str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 2 or set(names) != {"depth.npy", "source.npy"}:
                raise WorkmapPackError(
                    f"raw MoGe3-G archive must contain depth.npy and source.npy only: {path}"
                )
            with archive.open("depth.npy") as stream:
                depth_shape, fortran_order, dtype = _npy_header(stream, path, "raw depth")
                if (
                    fortran_order
                    or np.dtype(dtype) != np.dtype(np.float16)
                    or len(depth_shape) != 2
                ):
                    raise WorkmapPackError(
                        f"raw depth must be a two-dimensional float16 array: {path}"
                    )
            with archive.open("source.npy") as stream:
                shape, fortran_order, dtype = _npy_header(stream, path, "raw depth source")
                if shape != () or fortran_order or np.dtype(dtype).kind not in ("U", "S"):
                    raise WorkmapPackError(f"raw depth source tag must be a scalar string: {path}")
            # read_array parses the magic and header itself, so it needs a fresh stream.
            with archive.open("source.npy") as stream:
                source_value = np.lib.format.read_array(stream, allow_pickle=False).item()
    except WorkmapPackError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise WorkmapPackError(f"invalid raw MoGe3-G depth archive {path}: {exc}") from exc
    if source_value != "moge3-g":
        raise WorkmapPackError(
            f"depth archive source must be 'moge3-g', got {source_value!r}: {path}"
        )
    return tuple(int(v) for v in depth_shape), str(source_value)


def _load_keyframe_image(
    path: Path, *, name: str, expected_size: Any = None, expected_sha: Any = None
) -> tuple[int, int, str]:
    try:
        import cv2
    except ImportError as exc:
        raise WorkmapPackError("OpenCV is required to verify keyframe image dimensions") from exc
    path = _ensure_regular_file(path, label=f"keyframe image {name}")
    digest = _sha256(path)
    if expected_sha is not None and (not isinstance(expected_sha, str) or digest != expected_sha):
        raise WorkmapPackError(f"keyframe image SHA-256 mismatch: {name}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise WorkmapPackError(f"keyframe image is unreadable: {name}")
    height, width = image.shape[:2]
    if expected_size is not None:
        if not isinstance(expected_size, (list, tuple)) or len(expected_size) != 2:
            raise WorkmapPackError(f"keyframe image_size must be [width, height]: {name}")
        try:
            declared = tuple(int(value) for value in expected_size)
        except (TypeError, ValueError) as exc:
            raise WorkmapPackError(f"invalid keyframe image_size: {name}") from exc
        if declared != (width, height):
            raise WorkmapPackError(
                f"keyframe image dimensions disagree for {name}: declared {declared}, actual {(width, height)}"
            )
    return width, height, digest


def _model_frame_sha(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in FRAME_DIGEST_BINS:
        with (model_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _intersection_evidence_parts(
    model_receipt: Mapping[str, Any], model_frame_sha: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Any], Mapping[str, Any]]:
    evidence = model_receipt.get("fim_lwtl_scored_grid")
    if not isinstance(evidence, Mapping):
        raise WorkmapPackError(
            "cannot rebuild geometry-dependent FIM∩LWTL intersection cells: the optimized-model "
            "FIM/LWTL scored grid and its frozen scoring configuration/provenance are absent. "
            "The base release only carries derived weak-cell labels; those labels cannot be copied or treated as recomputed."
        )
    if evidence.get("schema") != INTERSECTION_SCHEMA:
        raise WorkmapPackError(f"FIM/LWTL evidence schema must be {INTERSECTION_SCHEMA!r}")
    if evidence.get("model_frame_sha256") != model_frame_sha:
        raise WorkmapPackError(
            "FIM/LWTL scored grid is not bound to the optimized model frame digest"
        )
    config = evidence.get("config")
    grid = evidence.get("grid")
    provenance = evidence.get("provenance")
    if (
        not isinstance(config, Mapping)
        or not isinstance(grid, list)
        or not isinstance(provenance, Mapping)
    ):
        raise WorkmapPackError(
            "FIM/LWTL evidence must contain config, grid, and reproducible provenance"
        )
    for field in ("tool", "version", "source_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value:
            raise WorkmapPackError(f"FIM/LWTL provenance lacks {field}")
    if not _is_sha256(provenance["source_sha256"]):
        raise WorkmapPackError("FIM/LWTL provenance source_sha256 is malformed")
    return evidence, config, grid, provenance


def _intersection_threshold(config: Mapping[str, Any], metric: str) -> tuple[str, float]:
    item = config.get(metric)
    if not isinstance(item, Mapping) or item.get("weak_when") not in ("<=", ">="):
        raise WorkmapPackError(f"FIM/LWTL config lacks an explicit {metric} weak-score rule")
    try:
        threshold = float(item["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkmapPackError(f"FIM/LWTL {metric} threshold is invalid") from exc
    if not math.isfinite(threshold):
        raise WorkmapPackError(f"FIM/LWTL {metric} threshold must be finite")
    if not isinstance(item.get("score_name"), str) or not item["score_name"]:
        raise WorkmapPackError(f"FIM/LWTL {metric} score_name is required")
    if item["score_name"] != f"{metric}_score":
        raise WorkmapPackError(f"FIM/LWTL score grid must bind the configured {metric}_score field")
    return str(item["weak_when"]), threshold


def _check_intersection_radius(config: Mapping[str, Any]) -> None:
    radius = config.get("intersection_radius")
    try:
        radius_value = float(radius)
    except (TypeError, ValueError) as exc:
        raise WorkmapPackError("FIM/LWTL intersection_radius is invalid") from exc
    if not math.isfinite(radius_value) or radius_value <= 0:
        raise WorkmapPackError("FIM/LWTL intersection_radius must be positive and finite")


def _score_intersection_grid(
    grid: list[Any], thresholds: Mapping[str, tuple[str, float]]
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    positions: list[list[float]] = []
    score_rows: list[dict[str, Any]] = []
    unique_positions: set[tuple[float, float, float]] = set()
    for index, row in enumerate(grid):
        if not isinstance(row, Mapping):
            raise WorkmapPackError(f"FIM/LWTL score-grid row {index} must be an object")
        try:
            xyz = tuple(float(value) for value in row["xyz"])
            fim_score = float(row["fim_score"])
            lwtl_score = float(row["lwtl_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkmapPackError(f"FIM/LWTL score-grid row {index} is incomplete") from exc
        if len(xyz) != 3 or not all(
            math.isfinite(value) for value in (*xyz, fim_score, lwtl_score)
        ):
            raise WorkmapPackError(f"FIM/LWTL score-grid row {index} contains non-finite values")
        if xyz in unique_positions:
            raise WorkmapPackError(f"duplicate FIM/LWTL score-grid position at row {index}")
        unique_positions.add(xyz)
        fim_weak = (
            fim_score <= thresholds["fim"][1]
            if thresholds["fim"][0] == "<="
            else fim_score >= thresholds["fim"][1]
        )
        lwtl_weak = (
            lwtl_score <= thresholds["lwtl"][1]
            if thresholds["lwtl"][0] == "<="
            else lwtl_score >= thresholds["lwtl"][1]
        )
        score_rows.append({"xyz": list(xyz), "fim_score": fim_score, "lwtl_score": lwtl_score})
        if fim_weak and lwtl_weak:
            positions.append(list(xyz))
    return positions, score_rows


def _validate_intersection_evidence(
    model_receipt: Mapping[str, Any], model_frame_sha: str
) -> tuple[list[list[float]], dict[str, Any]]:
    evidence, config, grid, provenance = _intersection_evidence_parts(
        model_receipt, model_frame_sha
    )
    thresholds = {metric: _intersection_threshold(config, metric) for metric in ("fim", "lwtl")}
    if evidence.get("config_sha256") != _bytes_sha256(_json_bytes(config)):
        raise WorkmapPackError(
            "FIM/LWTL config_sha256 does not match the supplied frozen score configuration"
        )
    _check_intersection_radius(config)
    positions, score_rows = _score_intersection_grid(grid, thresholds)
    if not grid:
        raise WorkmapPackError(
            "FIM/LWTL scored grid is empty; cannot prove weak-region recomputation"
        )
    score_sha = _bytes_sha256(_json_bytes(score_rows))
    if evidence.get("score_grid_sha256") != score_sha:
        raise WorkmapPackError(
            "FIM/LWTL score_grid_sha256 does not match the supplied raw scored grid"
        )
    return positions, {
        "status": "recomputed",
        "source_model_frame_sha256": model_frame_sha,
        "source_score_grid_sha256": score_sha,
        "source_provenance": dict(provenance),
        "config": dict(config),
        "scored_positions": len(grid),
        "intersection_positions": len(positions),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _intersection_payload(
    positions: list[list[float]], *, map_name: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "FIM_LWTL_INTERSECTION_CELLS",
        "map": map_name,
        "n_cells": len(positions),
        "radius": float(evidence["config"]["intersection_radius"]),
        "meaning": "positions weak under BOTH FIM and LWTL, recomputed against this optimized model",
        "cell_positions": positions,
        "recompute": {
            "source_model_frame_sha256": evidence["source_model_frame_sha256"],
            "source_score_grid_sha256": evidence["source_score_grid_sha256"],
            "config": evidence["config"],
            "provenance": evidence["source_provenance"],
        },
    }


def _check_gauge_receipt(model_receipt: Mapping[str, Any]) -> None:
    gauge = model_receipt.get("gauge")
    if not isinstance(gauge, Mapping) or gauge.get("pass") is not True:
        raise WorkmapPackError(
            "optimized model receipt does not prove a passing, anchored Sim3 gauge"
        )
    fit = gauge.get("fit")
    holdout = gauge.get("holdout")
    if not isinstance(fit, Mapping) or not isinstance(holdout, Mapping):
        raise WorkmapPackError(
            "optimized model receipt must include disjoint gauge fit and holdout evidence"
        )
    fit_ids = fit.get("landmark_ids")
    holdout_ids = holdout.get("landmark_ids")
    if not isinstance(fit_ids, list) or not isinstance(holdout_ids, list):
        raise WorkmapPackError("gauge fit and holdout must list the landmark IDs used in each set")
    fit_set, holdout_set = {str(value) for value in fit_ids}, {str(value) for value in holdout_ids}
    if len(fit_set) < 6 or len(holdout_set) < 6 or fit_set.intersection(holdout_set):
        raise WorkmapPackError(
            "gauge fit/holdout must each have at least six disjoint landmark IDs"
        )


def _check_edm_receipt_counts(model_receipt: Mapping[str, Any]) -> None:
    edm = model_receipt.get("edm_validation")
    if not isinstance(edm, Mapping):
        raise WorkmapPackError("optimized model receipt lacks actual EDM validation evidence")
    try:
        new_old = edm["new_old_pairs"]
        new_new = edm["new_new_pairs"]
    except KeyError as exc:
        raise WorkmapPackError(
            "EDM validation must report new_old_pairs and new_new_pairs"
        ) from exc
    if (
        isinstance(new_old, bool)
        or isinstance(new_new, bool)
        or not isinstance(new_old, int)
        or not isinstance(new_new, int)
        or new_old <= 0
        or new_new <= 0
    ):
        raise WorkmapPackError(
            "EDM validation must contain positive new↔old and new↔new accepted-pair counts"
        )
    provenance = edm.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise WorkmapPackError("EDM validation lacks reproducible provenance")


def _receipt_negative_id_map(model_receipt: Mapping[str, Any]) -> dict[int, int]:
    raw_map = model_receipt.get("negative_id_map")
    if not isinstance(raw_map, Mapping):
        raise WorkmapPackError("optimized model receipt lacks negative_id_map")
    negative_map: dict[int, int] = {}
    for old, new in raw_map.items():
        try:
            old_id, new_id = int(old), int(new)
        except (TypeError, ValueError) as exc:
            raise WorkmapPackError("negative_id_map contains a non-integer ID") from exc
        if old_id >= 0 or new_id <= 0 or old_id in negative_map:
            raise WorkmapPackError(
                "negative_id_map must map unique negative IDs to positive COLMAP IDs"
            )
        negative_map[old_id] = new_id
    if len(set(negative_map.values())) != len(negative_map):
        raise WorkmapPackError(
            "negative_id_map must assign a distinct positive COLMAP ID to each negative landmark"
        )
    return negative_map


def _receipt_keyframe_image_names(model_receipt: Mapping[str, Any]) -> dict[str, str]:
    names = model_receipt.get("keyframe_image_names")
    if not isinstance(names, Mapping) or not names:
        raise WorkmapPackError("optimized model receipt lacks keyframe_image_names")
    image_names: dict[str, str] = {}
    for original, output in names.items():
        if not isinstance(original, str) or not original or original in image_names:
            raise WorkmapPackError(
                "keyframe_image_names contains a duplicate or invalid source name"
            )
        normalized = _safe_relative(output, label="optimized keyframe image name", components=2)
        if PurePosixPath(normalized).parts[0] != "workmap":
            raise WorkmapPackError(
                f"new COLMAP keyframe must use the stable workmap namespace: {normalized}"
            )
        image_names[original] = normalized
    if len(set(image_names.values())) != len(image_names):
        raise WorkmapPackError(
            "keyframe_image_names must assign a distinct COLMAP image name to each session keyframe"
        )
    return image_names


def _validate_reconstruction_receipt(
    model_receipt: Mapping[str, Any],
) -> tuple[dict[int, int], dict[str, str]]:
    _check_gauge_receipt(model_receipt)
    _check_edm_receipt_counts(model_receipt)
    return _receipt_negative_id_map(model_receipt), _receipt_keyframe_image_names(model_receipt)


def _check_edm_binding(
    provenance: Any,
    bundle: Mapping[str, Any],
    bundle_sha: str,
    base_release: Path,
    session_meta: Mapping[str, Any],
    session_inputs: Mapping[str, Any],
) -> None:
    if not isinstance(provenance, Mapping):
        raise WorkmapPackError("EDM validation provenance must be a mapping")
    if provenance.get("base_bundle_sha256") != bundle_sha:
        raise WorkmapPackError("EDM evidence is not bound to the exact base direct_bundle.json")
    if provenance.get("session_id") != session_meta.get("session_id"):
        raise WorkmapPackError("EDM evidence is not bound to this completed session")
    if not isinstance(provenance.get("implementation"), str) or not provenance["implementation"]:
        raise WorkmapPackError("EDM provenance must identify the actual matcher implementation")
    if provenance.get("base_model_sha256") != bundle.get("model_sha256"):
        raise WorkmapPackError("EDM evidence base-model hashes do not match the bound release")
    reference_manifest = _ensure_regular_file(
        base_release / "localization/reference_manifest.jsonl",
        label="base reference manifest for EDM provenance",
    )
    if provenance.get("base_reference_manifest_sha256") != _sha256(reference_manifest):
        raise WorkmapPackError(
            "EDM evidence reference-manifest SHA does not match the base release"
        )
    expected_inputs = {
        "session_meta_sha256": _sha256(session_inputs["meta_path"]),
        "keyframes_jsonl_sha256": _sha256(session_inputs["keyframes_path"]),
        "observations_jsonl_sha256": _sha256(session_inputs["observations_path"]),
        "keyframe_images_sha256": session_inputs["keyframe_images_sha256"],
    }
    if provenance.get("session_input_sha256") != expected_inputs:
        raise WorkmapPackError(
            "EDM evidence session-input hashes do not match the completed session files"
        )


def _check_edm_runtime(provenance: Mapping[str, Any]) -> None:
    for field in ("edm_checkpoint_sha256", "boq_checkpoint_sha256", "profile_sha256"):
        if not _is_sha256(provenance.get(field)):
            raise WorkmapPackError(f"EDM evidence provenance lacks a valid {field}")
    if not isinstance(provenance.get("matcher"), Mapping) or not provenance["matcher"]:
        raise WorkmapPackError("EDM evidence provenance lacks the matcher configuration")
    source_hashes = provenance.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise WorkmapPackError("EDM provenance lacks pinned matcher source digests")
    for field in (
        "reconstruction_tool",
        "live_provider",
        "workmap_overlap",
        "official_edm_adapter",
    ):
        if not _is_sha256(source_hashes.get(field)):
            raise WorkmapPackError(f"EDM provenance lacks a valid {field} source SHA-256")
    versions = provenance.get("runtime_versions")
    if not isinstance(versions, Mapping) or any(
        not isinstance(versions.get(field), str) or not versions[field]
        for field in ("pycolmap", "opencv", "torch")
    ):
        raise WorkmapPackError("EDM provenance lacks pycolmap/OpenCV/PyTorch runtime versions")


def _validate_edm_provenance(
    edm: Mapping[str, Any],
    bundle: Mapping[str, Any],
    bundle_sha: str,
    base_release: Path,
    session_meta: Mapping[str, Any],
    session_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = edm.get("provenance")
    _check_edm_binding(provenance, bundle, bundle_sha, base_release, session_meta, session_inputs)
    _check_edm_runtime(provenance)
    try:
        return json.loads(_json_bytes(dict(provenance)))
    except (TypeError, ValueError) as exc:
        raise WorkmapPackError(f"EDM provenance is not finite JSON data: {exc}") from exc


def _image_observations(
    image: Any, *, unique_tracks: bool = True
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Registered 2D observations and the first one per Point3D ID.

    Dense EDM matching leaves nearby keypoints on the same track in base
    images, so counting callers pass ``unique_tracks=False``; identity checks
    against measured session pixels keep the one-observation-per-track rule.
    """
    xy: list[np.ndarray] = []
    by_point: dict[int, np.ndarray] = {}
    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue
        point_id = int(point2d.point3D_id)
        measured = np.asarray(point2d.xy, dtype=np.float64).reshape(2)
        if (
            not np.isfinite(measured).all()
            or point_id <= 0
            or (unique_tracks and point_id in by_point)
        ):
            raise WorkmapPackError(
                f"invalid or repeated 3D observation in COLMAP image {image.name}"
            )
        xy.append(measured)
        by_point.setdefault(point_id, measured)
    array = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return array, by_point


def _occupancy_rows(
    reconstruction: Any, images: list[Any]
) -> tuple[list[tuple[str, np.ndarray, int, int]], list[int]]:
    rows: list[tuple[str, np.ndarray, int, int]] = []
    observation_counts: list[int] = []
    for image in images:
        camera = reconstruction.cameras[image.camera_id]
        points, _ = _image_observations(image, unique_tracks=False)
        width, height = int(camera.width), int(camera.height)
        if width <= 0 or height <= 0:
            raise WorkmapPackError(f"COLMAP image camera has invalid size: {image.name}")
        if len(points) and (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= height).any()
        ):
            raise WorkmapPackError(f"COLMAP observations escape the image bounds: {image.name}")
        observation_counts.append(len(points))
        rows.append((str(image.name), points, width, height))
    return rows, observation_counts


def _sampled_occupancy(
    sampled: list[tuple[str, np.ndarray, int, int]],
) -> tuple[list[np.ndarray], dict[int, list[float]]]:
    from scipy.spatial import cKDTree

    nn_all: list[np.ndarray] = []
    occupied: dict[int, list[float]] = {size: [] for size in (2, 8, 16)}
    for _name, xy, width, height in sampled:
        sampled_xy = xy[np.linspace(0, len(xy) - 1, 2000, dtype=np.int64)] if len(xy) > 2000 else xy
        if len(sampled_xy) > 1:
            distances = cKDTree(sampled_xy).query(sampled_xy, k=2)[0][:, 1]
            nn_all.append(np.asarray(distances, dtype=np.float64))
        for cell_px in occupied:
            nx = max(1, int(math.ceil(width / cell_px)))
            ny = max(1, int(math.ceil(height / cell_px)))
            cells = np.floor(xy / float(cell_px)).astype(np.int64)
            occupied[cell_px].append(
                float(len(set(map(tuple, cells.tolist()))) / (nx * ny)) if len(xy) else 0.0
            )
    return nn_all, occupied


def _occupancy_payload(reconstruction: Any) -> dict[str, Any]:
    images = sorted(reconstruction.images.values(), key=lambda image: str(image.name))
    if not images:
        raise WorkmapPackError("optimized COLMAP model contains no images")
    if int(reconstruction.num_points3D()) <= 0:
        raise WorkmapPackError("optimized COLMAP model contains no 3D landmarks")
    rows, observation_counts = _occupancy_rows(reconstruction, images)
    counts = np.asarray(observation_counts, dtype=np.float64)
    if not counts.any():
        raise WorkmapPackError("optimized COLMAP model has no registered 2D↔3D observations")
    sampled = [rows[index] for index in _stratified_indices(len(rows), 40)]
    nn_all, occupied = _sampled_occupancy(sampled)
    nn = np.concatenate(nn_all) if nn_all else np.empty(0, dtype=np.float64)
    return {
        "n_images": len(images),
        "n_points3D": int(reconstruction.num_points3D()),
        "observation_count": {
            "p10": float(np.percentile(counts, 10)),
            "median": float(np.percentile(counts, 50)),
            "p90": float(np.percentile(counts, 90)),
        },
        "nn_spacing_px": {
            "p10": float(np.percentile(nn, 10)) if nn.size else None,
            "median": float(np.percentile(nn, 50)) if nn.size else None,
            "p90": float(np.percentile(nn, 90)) if nn.size else None,
            "n_images_sampled": len(sampled),
            "max_points_per_image": 2000,
        },
        "occupied_cell_fraction": {
            str(cell_px): float(np.mean(values)) if values else 0.0
            for cell_px, values in occupied.items()
        },
        "derivation": "recomputed from optimized COLMAP image observations; base baseline_smoke metadata was not carried forward",
    }


def _stratified_indices(n: int, k: int) -> list[int]:
    sample = min(k, n)
    if sample <= 0:
        return []
    if sample == 1:
        return [0]
    return [int(round(index * (n - 1) / (sample - 1))) for index in range(sample)]


def _load_base_keyframes(
    base_rows: list[dict[str, Any]], keyframes_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    keyframe_by_name: dict[str, dict[str, Any]] = {}
    image_paths: dict[str, Path] = {}
    for row in base_rows:
        name = _identity_from_keyframe(row, label="base keyframe")
        if name in keyframe_by_name:
            raise WorkmapPackError(f"duplicate base keyframe identity: {name}")
        image_path = keyframes_root / name
        _ensure_regular_file(image_path, label=f"base image {name}")
        expected_sha = row.get("image_sha256")
        if not _is_sha256(expected_sha):
            raise WorkmapPackError(f"base keyframe has no valid image SHA-256: {name}")
        width, height, _digest = _load_keyframe_image(
            image_path,
            name=name,
            expected_size=row.get("image_size"),
            expected_sha=expected_sha,
        )
        row["image_size"] = [width, height]
        keyframe_by_name[name] = row
        image_paths[name] = image_path
    return keyframe_by_name, image_paths


def _load_base_references(
    ref_rows: list[dict[str, Any]], keyframe_by_name: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for row in ref_rows:
        name = _safe_relative(
            row.get("image_name"), label="base reference image_name", components=2
        )
        if name in references or name not in keyframe_by_name:
            raise WorkmapPackError(f"base reference is duplicated or absent from keyframes: {name}")
        references[name] = row
    return references


def _load_base_bank(
    names_path: Path, bank_path: Path, references: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    try:
        bank_names = json.loads(names_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkmapPackError(f"invalid base BoQ names: {names_path}: {exc}") from exc
    if not isinstance(bank_names, list) or any(not isinstance(name, str) for name in bank_names):
        raise WorkmapPackError("base BoQ names must be a JSON string list")
    if bank_names != sorted(bank_names) or len(bank_names) != len(set(bank_names)):
        raise WorkmapPackError("base BoQ bank names must be unique and lexicographically sorted")
    if any(name not in references for name in bank_names):
        raise WorkmapPackError("base BoQ bank names do not match the base reference catalog")
    try:
        bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise WorkmapPackError(f"invalid base BoQ descriptors: {bank_path}: {exc}") from exc
    if bank.ndim != 2 or bank.shape != (len(bank_names), BOQ_DIM) or bank.dtype != np.float32:
        raise WorkmapPackError(
            f"base BoQ bank violates (N,{BOQ_DIM}) float32 contract: {bank.shape} {bank.dtype}"
        )
    if not np.isfinite(bank).all():
        raise WorkmapPackError("base BoQ descriptors contain non-finite values")
    norms = np.linalg.norm(np.asarray(bank), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise WorkmapPackError("base BoQ descriptors are not L2-normalized")
    return {
        name: np.asarray(bank[index], dtype=np.float32) for index, name in enumerate(bank_names)
    }


def _load_base_depth(
    base_release: Path, keyframe_by_name: Mapping[str, Any], image_paths: Mapping[str, Path]
) -> dict[str, Path]:
    import cv2

    depth_root = base_release / "depth_moge3"
    if not depth_root.is_dir() or depth_root.is_symlink():
        raise WorkmapPackError(
            f"base release is missing raw image-only MoGe3-G depth: {depth_root}"
        )
    depth_paths: dict[str, Path] = {}
    for path in _ensure_no_symlink_tree(depth_root, label="base raw depth"):
        if path.suffix != ".npz" or not path.name.endswith(".jpg.npz"):
            raise WorkmapPackError(f"unexpected non-MoGe3 depth asset: {path}")
        name = _safe_relative(
            path.relative_to(depth_root).as_posix()[:-4], label="depth image identity", components=2
        )
        if name not in keyframe_by_name or name in depth_paths:
            raise WorkmapPackError(
                f"depth archive is duplicated or has no matching base image: {name}"
            )
        decoded = cv2.imread(str(image_paths[name]), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise WorkmapPackError(f"cannot read base image for raw depth size check: {name}")
        height, width = decoded.shape[:2]
        depth_shape, _ = _read_npz_headers(path)
        if depth_shape != (height, width):
            raise WorkmapPackError(
                f"raw depth shape {depth_shape} does not match source image {(height, width)}: {name}"
            )
        depth_paths[name] = path
    return depth_paths


def _shard_entries(
    shard_manifest: Mapping[str, Any], label: str, field: str
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    groups = shard_manifest.get(field)
    if not isinstance(groups, list):
        raise WorkmapPackError(f"base shard manifest has no {field}")
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("files"), list):
            raise WorkmapPackError(f"malformed {label} shard entry")
        for entry in group["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise WorkmapPackError(f"malformed {label} file entry in shard manifest")
            rel = _safe_relative(entry["path"], label=f"{label} shard path", components=2)
            if rel in entries:
                raise WorkmapPackError(f"duplicate {label} shard path: {rel}")
            entries[rel] = entry
    return entries


def _verify_shard_entries(
    entries: Mapping[str, Mapping[str, Any]], expected_paths: Mapping[str, Path], label: str
) -> None:
    if set(entries) != set(expected_paths):
        raise WorkmapPackError(f"base shard manifest {label} names disagree with the source tree")
    for rel, source in expected_paths.items():
        entry = entries[rel]
        if entry.get("size_bytes") != source.stat().st_size or entry.get("sha256") != _sha256(
            source
        ):
            raise WorkmapPackError(f"base shard manifest {label} SHA/size mismatch: {rel}")


def _verify_base_shards(
    base_release: Path, image_paths: Mapping[str, Path], depth_paths: Mapping[str, Path]
) -> dict[str, str]:
    """Check both shard catalogs against the source tree; return raw-depth hashes."""
    shard_manifest_path = _ensure_regular_file(
        base_release / "localization/shard_manifest.json", label="base release shard manifest"
    )
    shard_manifest = _read_json(shard_manifest_path, label="base release shard manifest")
    image_entries = _shard_entries(shard_manifest, "keyframe images", "shards")
    _verify_shard_entries(image_entries, image_paths, "keyframe images")
    depth_expected = {f"{name}.npz": source for name, source in depth_paths.items()}
    depth_entries = _shard_entries(shard_manifest, "raw depth", "depth_shards")
    _verify_shard_entries(depth_entries, depth_expected, "raw depth")
    return {rel[:-4]: str(depth_entries[rel]["sha256"]) for rel in depth_expected}


def _load_base_assets(
    base_release: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Path],
    dict[str, Path],
    dict[str, str],
]:
    keyframes_root = base_release / "keyframes/images"
    base_keyframes_path = _ensure_regular_file(
        base_release / "keyframes/keyframes.jsonl", label="base keyframes manifest"
    )
    localization = base_release / "localization"
    ref_path = _ensure_regular_file(
        localization / "reference_manifest.jsonl", label="base reference manifest"
    )
    names_path = _ensure_regular_file(
        localization / "boq_references_sideview.names.json", label="base BoQ names"
    )
    bank_path = _ensure_regular_file(
        localization / "boq_references_sideview.npy", label="base BoQ bank"
    )
    image_files = _ensure_no_symlink_tree(keyframes_root, label="base keyframe images")
    base_rows = _read_jsonl(base_keyframes_path, label="base keyframes")
    keyframe_by_name, image_paths = _load_base_keyframes(base_rows, keyframes_root)
    actual_images = {path.relative_to(keyframes_root).as_posix() for path in image_files}
    if actual_images != set(keyframe_by_name):
        missing = sorted(set(keyframe_by_name) - actual_images)[:5]
        extra = sorted(actual_images - set(keyframe_by_name))[:5]
        raise WorkmapPackError(
            f"base image tree and keyframe manifest differ (missing={missing}, extra={extra})"
        )

    references = _load_base_references(
        _read_jsonl(ref_path, label="base reference manifest"), keyframe_by_name
    )
    old_descriptors = _load_base_bank(names_path, bank_path, references)
    depth_paths = _load_base_depth(base_release, keyframe_by_name, image_paths)
    depth_asset_hashes = _verify_base_shards(base_release, image_paths, depth_paths)
    return base_rows, references, old_descriptors, image_paths, depth_paths, depth_asset_hashes


def _session_image_path(session_dir: Path, raw_path: Any, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise WorkmapPackError(f"session keyframe has no image_path: {name}")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = session_dir / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(session_dir)
    except ValueError as exc:
        raise WorkmapPackError(
            f"session keyframe image escapes the session directory: {resolved}"
        ) from exc
    lexical = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        if part == "..":
            lexical = lexical.parent
        elif part != ".":
            lexical = lexical / part
        if lexical.is_symlink():
            target = lexical.resolve(strict=True)
            # Archived image paths may use an alias for the moved worktree.
            # Never allow links within the actual session, or links to a
            # different tree, even if the final path happens to exist.
            if lexical.is_relative_to(session_dir) or (
                target != session_dir and target not in session_dir.parents
            ):
                raise WorkmapPackError(f"session image path must not traverse symlinks: {lexical}")
    return _ensure_regular_file(resolved, label=f"session image {name}")


def _matched_measurements(
    measured: Mapping[int, tuple[float, float]],
    negative_map: Mapping[int, int],
    observed_points: Mapping[int, np.ndarray],
) -> tuple[int, list[int], list[int]]:
    matches = 0
    matched_negative_ids: list[int] = []
    matched_positive_ids: list[int] = []
    for source_id, source_xy in measured.items():
        point_id = negative_map.get(source_id, source_id)
        model_xy = observed_points.get(point_id)
        if model_xy is not None and np.allclose(model_xy, source_xy, rtol=0.0, atol=1e-3):
            matches += 1
            (matched_negative_ids if source_id < 0 else matched_positive_ids).append(source_id)
    return matches, matched_negative_ids, matched_positive_ids


def _new_keyframe_camera(
    row: Mapping[str, Any], reconstruction: Any, image: Any, source_name: str, model_name: str
) -> tuple[np.ndarray, int, int]:
    try:
        K = np.asarray(row["K"], dtype=np.float64).reshape(3, 3)
        width, height = (int(value) for value in row["image_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkmapPackError(
            f"new keyframe lacks valid K/image_size metadata: {source_name}"
        ) from exc
    if not np.isfinite(K).all() or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise WorkmapPackError(f"new keyframe has invalid measured K: {source_name}")
    camera = reconstruction.cameras[image.camera_id]
    if int(camera.width) != width or int(camera.height) != height:
        raise WorkmapPackError(
            f"optimized COLMAP camera image size disagrees with session: {model_name}"
        )
    camera_name = str(getattr(camera, "model_name", ""))
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if camera_name != "PINHOLE" or len(params) != 4:
        raise WorkmapPackError(
            f"new workmap camera must be an independent PINHOLE camera: {model_name}"
        )
    model_K = np.asarray(
        [[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.allclose(model_K, K, rtol=1e-5, atol=1e-4):
        raise WorkmapPackError(
            f"optimized COLMAP intrinsics disagree with measured session K: {model_name}"
        )
    return K, width, height


def _optimized_pose(image: Any, model_name: str) -> np.ndarray:
    try:
        optimized_pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)[:3, :4]
    except (AttributeError, TypeError, ValueError) as exc:
        raise WorkmapPackError(
            f"optimized workmap keyframe lacks a valid COLMAP pose: {model_name}"
        ) from exc
    if optimized_pose.shape != (3, 4) or not np.isfinite(optimized_pose).all():
        raise WorkmapPackError(
            f"optimized workmap keyframe has a non-finite COLMAP pose: {model_name}"
        )
    return optimized_pose


def _validate_new_keyframes(
    session_dir: Path,
    keyframe_rows: list[dict[str, Any]],
    keyframe_image_names: Mapping[str, str],
    reconstruction: Any,
    negative_map: Mapping[int, int],
    measured_by_source: Mapping[str, Mapping[int, tuple[float, float]]],
    session_image_hashes: Mapping[str, str],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], dict[str, int]]:
    model_images = {str(image.name): image for image in reconstruction.images.values()}
    images_by_source: dict[str, Path] = {}
    output_rows: dict[str, dict[str, Any]] = {}
    observation_counts: dict[str, int] = {}
    names: set[str] = set()
    for row in keyframe_rows:
        source_name = str(row["name"])
        model_name = keyframe_image_names.get(source_name)
        if model_name is None:
            raise WorkmapPackError(
                f"optimized model receipt does not map session keyframe {source_name}"
            )
        if model_name in names or model_name not in model_images:
            raise WorkmapPackError(
                f"optimized COLMAP model lacks or duplicates new keyframe image {model_name}"
            )
        names.add(model_name)
        source_path = _session_image_path(session_dir, row.get("image_path"), name=source_name)
        _image_width, _image_height, image_sha = _load_keyframe_image(
            source_path,
            name=source_name,
            expected_size=row.get("image_size"),
            expected_sha=session_image_hashes[source_name],
        )
        image = model_images[model_name]
        observed_xy, observed_points = _image_observations(image)
        if len(observed_xy) == 0:
            raise WorkmapPackError(
                f"new keyframe has no optimized 2D↔3D observations: {model_name}"
            )
        matches, matched_negative_ids, matched_positive_ids = _matched_measurements(
            measured_by_source[source_name], negative_map, observed_points
        )
        K, width, height = _new_keyframe_camera(row, reconstruction, image, source_name, model_name)
        optimized_pose = _optimized_pose(image, model_name)
        output_rows[model_name] = {
            "name": model_name,
            "source_session_keyframe": source_name,
            "ordinal": int(row["ordinal"]),
            "stamp": float(row["stamp"]),
            "source_epoch": int(row.get("source_epoch", 0)),
            "K": K.tolist(),
            "image_size": [width, height],
            "cam_from_world": optimized_pose.tolist(),
            "image_sha256": image_sha,
            "image_bytes": source_path.stat().st_size,
            "matched_measured_observations": matches,
            "matched_negative_ids": sorted(matched_negative_ids),
            "matched_positive_ids": sorted(matched_positive_ids),
        }
        images_by_source[model_name] = source_path
        observation_counts[model_name] = len(observed_xy)
    return images_by_source, output_rows, observation_counts


def _load_boq_extractor() -> Any:
    import sys

    deploy_path = str(DEPLOY_DIR)
    if deploy_path not in sys.path:
        sys.path.insert(0, deploy_path)
    from boq_extractor import BOQ_WEIGHTS_SHA256, BoQExtractor

    extractor = BoQExtractor(device="cuda")
    if _sha256(Path(extractor.weights_path)) != BOQ_WEIGHTS_SHA256:
        raise WorkmapPackError("BoQ weights are not the approved pinned checkpoint")
    return extractor, BOQ_WEIGHTS_SHA256


def _write_boq_bank(
    localization: Path,
    old_descriptors: Mapping[str, np.ndarray],
    old_image_hashes: Mapping[str, str],
    new_names: list[str],
    new_image_paths: Mapping[str, Path],
    image_hashes: Mapping[str, str],
) -> dict[str, Any]:
    preserved: dict[str, np.ndarray] = {}
    for name, descriptor in old_descriptors.items():
        if (
            name not in old_image_hashes
            or name not in image_hashes
            or old_image_hashes[name] != image_hashes[name]
        ):
            continue
        preserved[name] = descriptor
    if len(preserved) != len(old_descriptors):
        missing = sorted(set(old_descriptors) - set(preserved))[:5]
        raise WorkmapPackError(
            f"refusing to preserve old BoQ rows whose image bytes changed or are absent: {missing}"
        )
    if not new_names:
        raise WorkmapPackError("no new keyframe references are available for BoQ extraction")
    extractor, weights_sha = _load_boq_extractor()
    ordered_new = sorted(new_names)
    descriptor_rows = extractor.extract_bank(
        [new_image_paths[name] for name in ordered_new], batch_size=8
    )
    descriptor_rows = np.asarray(descriptor_rows)
    if descriptor_rows.dtype != np.float32 or descriptor_rows.shape != (len(ordered_new), BOQ_DIM):
        raise WorkmapPackError(
            f"actual BoQExtractor returned unexpected shape/dtype: {descriptor_rows.shape} {descriptor_rows.dtype}"
        )
    if not np.isfinite(descriptor_rows).all() or not np.allclose(
        np.linalg.norm(descriptor_rows, axis=1), 1.0, rtol=1e-4, atol=1e-4
    ):
        raise WorkmapPackError(
            "actual BoQExtractor returned non-finite or non-normalized descriptors"
        )
    descriptors: dict[str, np.ndarray] = dict(preserved)
    descriptors.update({name: descriptor_rows[index] for index, name in enumerate(ordered_new)})
    names = sorted(descriptors)
    if len(names) != len(set(names)):
        raise WorkmapPackError("BoQ names are not unique")
    bank = np.ascontiguousarray(np.stack([descriptors[name] for name in names]), dtype=np.float32)
    if bank.shape != (len(names), BOQ_DIM) or not np.isfinite(bank).all():
        raise WorkmapPackError("assembled BoQ bank violates the finite named-bank contract")
    localization.mkdir(parents=True, exist_ok=True)
    np.save(localization / "boq_references_sideview.npy", bank, allow_pickle=False)
    _write_json(localization / "boq_references_sideview.names.json", names)
    del extractor
    return {
        "count": len(names),
        "new_count": len(ordered_new),
        "preserved_old_count": len(preserved),
        "dimension": BOQ_DIM,
        "dtype": "float32",
        "l2_normalized": True,
        "weights_sha256": weights_sha,
        "names": names,
    }


def _check_staging_tree(output: Path, model: Path) -> None:
    for path in output.rglob("*"):
        if path.is_symlink():
            raise WorkmapPackError(f"output staging tree must not contain symlinks: {path}")
        if path.is_file() and not path.is_relative_to(model):
            raise WorkmapPackError(f"output_pack already contains non-model files: {path}")
        if (
            path.is_dir()
            and path != model
            and not path.is_relative_to(model)
            and path not in model.parents
        ):
            raise WorkmapPackError(f"output_pack already contains unrelated directories: {path}")


def _check_pack_staging(
    base_release: Path, output_pack: Path, optimized_model: Path
) -> tuple[Path, Path, Path]:
    raw_base = Path(base_release).expanduser()
    raw_output = Path(output_pack).expanduser()
    raw_model = Path(optimized_model).expanduser()
    if raw_base.is_symlink() or raw_output.is_symlink() or raw_model.is_symlink():
        raise WorkmapPackError(
            "base release, output staging root, and optimized model must not be symlinks"
        )
    base = raw_base.resolve(strict=True)
    output = raw_output.resolve(strict=True)
    model = raw_model.resolve(strict=True)
    expected_model = output / "map/experiments/edm_fixedpose_retriangulate/model"
    if not output.is_dir():
        raise WorkmapPackError(f"output_pack must be an existing staging directory: {output}")
    if output == base or output.is_relative_to(base) or base.is_relative_to(output):
        raise WorkmapPackError(
            "output pack and frozen base release must use disjoint directory trees"
        )
    if model != expected_model or not model.is_dir():
        raise WorkmapPackError(
            f"optimized_model must be the pre-existing model directory under output_pack: {expected_model}"
        )
    if (model / "occupancy.json").exists():
        raise WorkmapPackError(
            "optimized model staging directory already contains occupancy.json; refusing stale/overwritten output"
        )
    _check_staging_tree(output, model)
    for name in MODEL_BINS:
        _ensure_regular_file(model / name, label=f"optimized COLMAP {name}")
    actual_model_files = {
        path.relative_to(model).as_posix()
        for path in _ensure_no_symlink_tree(model, label="optimized COLMAP model")
    }
    if actual_model_files != set(MODEL_BINS):
        raise WorkmapPackError(
            "optimized COLMAP model staging must contain exactly the five model binaries and no stale assets"
        )
    return base, output, model


def _check_model_receipt(
    model_receipt: Mapping[str, Any], bundle_sha: str, meta: Mapping[str, Any]
) -> dict[str, Any]:
    if model_receipt.get("base_bundle_sha256") != bundle_sha:
        raise WorkmapPackError(
            "optimized model receipt is not bound to the exact base release bundle"
        )
    if model_receipt.get("session_id") != meta.get("session_id"):
        raise WorkmapPackError(
            "optimized model receipt belongs to a different online workmap session"
        )
    filtering_receipt = model_receipt.get("filtering")
    ba_receipt = model_receipt.get("bundle_adjustment")
    if not isinstance(filtering_receipt, Mapping) or not isinstance(ba_receipt, Mapping):
        raise WorkmapPackError("optimized model receipt lacks global filtering/BA counts")
    if ba_receipt.get("fixed_intrinsics") is not True:
        raise WorkmapPackError(
            "optimized model receipt does not prove fixed-intrinsics global bundle adjustment"
        )
    optimizer_receipt = {"filtering": filtering_receipt, "bundle_adjustment": ba_receipt}
    try:
        _json_bytes(optimizer_receipt)
    except (TypeError, ValueError) as exc:
        raise WorkmapPackError(f"filtering/BA receipt is not finite JSON data: {exc}") from exc
    try:
        _json_bytes(
            {
                "gauge": model_receipt["gauge"],
                "triangulation": model_receipt.get("triangulation"),
            }
        )
    except (TypeError, ValueError) as exc:
        raise WorkmapPackError(
            f"gauge/triangulation receipt is not finite JSON data: {exc}"
        ) from exc
    return optimizer_receipt


def _open_optimized_model(model: Path) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import pycolmap
    except ImportError as exc:
        raise WorkmapPackError(
            "pycolmap is required to verify and count optimized COLMAP observations"
        ) from exc
    try:
        reconstruction = pycolmap.Reconstruction(str(model))
    except Exception as exc:
        raise WorkmapPackError(
            f"optimized COLMAP model cannot be reopened: {model}: {exc}"
        ) from exc
    model_images = {str(image.name): image for image in reconstruction.images.values()}
    if len(model_images) != int(reconstruction.num_images()):
        raise WorkmapPackError("optimized COLMAP model has duplicate image names")
    return pycolmap, reconstruction, model_images


def _check_model_catalog(
    model_names: set[str],
    base_names: set[str],
    expected_new_names: set[str],
    keyframe_image_names: Mapping[str, str],
    session_keyframes: list[dict[str, Any]],
) -> None:
    if len(keyframe_image_names) != len(session_keyframes):
        raise WorkmapPackError(
            "optimized model receipt does not map every durable session keyframe"
        )
    if model_names != base_names | expected_new_names:
        missing = sorted((base_names | expected_new_names) - model_names)[:5]
        extra = sorted(model_names - (base_names | expected_new_names))[:5]
        raise WorkmapPackError(
            f"optimized model images do not match base+session catalogs (missing={missing}, extra={extra})"
        )
    if len(model_names) != len(base_names) + len(expected_new_names):
        raise WorkmapPackError("optimized model contains duplicate base/session image identities")


def _base_anchor_points(
    pycolmap: Any,
    base_root: Path,
    negative_map: Mapping[int, int],
    new_metadata: dict[str, dict[str, Any]],
) -> tuple[Any, set[int]]:
    """Frozen base Point3D IDs; marks each new keyframe that can serve retrieval."""
    base_point_ids: set[int] = set()
    try:
        base_reconstruction = pycolmap.Reconstruction(str(base_root / "model"))
        base_point_ids = {int(point_id) for point_id in base_reconstruction.points3D}
    except Exception as exc:
        raise WorkmapPackError(
            f"base COLMAP model cannot be reopened for anchor verification: {exc}"
        ) from exc
    collisions = sorted(set(negative_map.values()).intersection(base_point_ids))
    if collisions:
        raise WorkmapPackError(
            f"new negative landmark IDs collide with frozen base Point3D IDs: {collisions[:5]}"
        )
    for image_name, metadata in new_metadata.items():
        positive_ids = set(metadata["matched_positive_ids"])
        unanchored = sorted(positive_ids - base_point_ids)
        if unanchored:
            raise WorkmapPackError(
                f"measured positive IDs in {image_name} are not frozen base landmarks: {unanchored[:5]}"
            )
        metadata["reference_ready"] = bool(metadata["matched_negative_ids"] or positive_ids)
    if not any(metadata["reference_ready"] for metadata in new_metadata.values()):
        raise WorkmapPackError(
            "no session keyframe has verified measured old/new map support for retrieval"
        )
    return base_reconstruction, base_point_ids


def _check_base_cameras(
    base_reconstruction: Any,
    reconstruction: Any,
    model_images: Mapping[str, Any],
    base_rows: list[dict[str, Any]],
    expected_new_names: set[str],
) -> None:
    base_rows_by_name = {
        _identity_from_keyframe(row, label="base keyframe"): row for row in base_rows
    }
    for old_image in base_reconstruction.images.values():
        name = str(old_image.name)
        optimized_image = model_images.get(name)
        if optimized_image is None:
            raise WorkmapPackError(f"optimized model omits base image: {name}")
        old_camera = base_reconstruction.cameras[old_image.camera_id]
        new_camera = reconstruction.cameras[optimized_image.camera_id]
        if (
            int(old_camera.width) != int(new_camera.width)
            or int(old_camera.height) != int(new_camera.height)
            or str(old_camera.model_name) != str(new_camera.model_name)
            or not np.array_equal(np.asarray(old_camera.params), np.asarray(new_camera.params))
        ):
            raise WorkmapPackError(f"optimized model changed frozen base intrinsics: {name}")
        if (int(new_camera.width), int(new_camera.height)) != tuple(
            base_rows_by_name[name]["image_size"]
        ):
            raise WorkmapPackError(
                f"base keyframe image size disagrees with optimized camera: {name}"
            )
    base_camera_ids = {int(camera_id) for camera_id in base_reconstruction.cameras}
    new_camera_ids = [int(model_images[name].camera_id) for name in sorted(expected_new_names)]
    if len(set(new_camera_ids)) != len(expected_new_names) or set(new_camera_ids) & base_camera_ids:
        raise WorkmapPackError(
            "each new workmap keyframe must use a camera ID independent of the frozen base"
        )


def _check_negative_tracks(
    negative_map: Mapping[int, int],
    reconstruction: Any,
    new_image_ids: Mapping[int, str],
    source_name_by_image: Mapping[str, str],
    new_metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    for source_id, point_id in negative_map.items():
        if point_id not in reconstruction.points3D:
            raise WorkmapPackError(
                f"negative landmark {source_id} is absent from the optimized model"
            )
        point = reconstruction.points3D[point_id]
        tracked_session_images: set[str] = set()
        for element in point.track.elements:
            image_name = new_image_ids.get(int(element.image_id))
            if image_name is None:
                continue
            if (
                source_name_by_image.get(image_name) is None
                or source_id not in new_metadata[image_name]["matched_negative_ids"]
            ):
                raise WorkmapPackError(
                    f"optimized track for negative landmark {source_id} lacks a matching measured session observation"
                )
            tracked_session_images.add(image_name)
        if len(tracked_session_images) < 2:
            raise WorkmapPackError(
                f"negative landmark {source_id} has no verified measured two-image COLMAP track"
            )


def _image_track_links(
    image: Any, reconstruction: Any, base_image_ids: set[int], base_point_ids: set[int]
) -> tuple[set[int], bool, list[int]]:
    """Point IDs seen by one image, whether one is a base point in a base image,
    and every other tracked image ID in visiting order."""
    points: set[int] = set()
    reaches_base = False
    tracked: list[int] = []
    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue
        point_id = int(point2d.point3D_id)
        points.add(point_id)
        if point_id not in reconstruction.points3D:
            raise WorkmapPackError(f"COLMAP image references missing Point3D ID {point_id}")
        for element in reconstruction.points3D[point_id].track.elements:
            tracked_image_id = int(element.image_id)
            if tracked_image_id in base_image_ids and point_id in base_point_ids:
                reaches_base = True
            else:
                tracked.append(tracked_image_id)
    return points, reaches_base, tracked


def _unanchored_workmap_images(
    new_image_ids: Mapping[int, str],
    model_images: Mapping[str, Any],
    reconstruction: Any,
    base_image_ids: set[int],
    base_point_ids: set[int],
) -> set[str]:
    unanchored: set[str] = set()
    remaining = set(new_image_ids)
    while remaining:
        component = {remaining.pop()}
        pending = list(component)
        component_points: set[int] = set()
        reaches_base = False
        while pending:
            image_id = pending.pop()
            points, reaches, tracked = _image_track_links(
                model_images[new_image_ids[image_id]],
                reconstruction,
                base_image_ids,
                base_point_ids,
            )
            component_points |= points
            reaches_base = reaches_base or reaches
            for tracked_image_id in tracked:
                if tracked_image_id in remaining:
                    remaining.remove(tracked_image_id)
                    component.add(tracked_image_id)
                    pending.append(tracked_image_id)
        if not reaches_base or len(component_points.intersection(base_point_ids)) < 3:
            unanchored.update(new_image_ids[image_id] for image_id in component)
    return unanchored


@dataclass
class _ValidatedPack:
    """Every verified input the staging step copies or records."""

    output: Path
    model: Path
    bundle: dict[str, Any]
    bundle_sha: str
    meta: dict[str, Any]
    session_inputs: dict[str, Any]
    model_receipt: Mapping[str, Any]
    negative_map: dict[int, int]
    keyframe_image_names: dict[str, str]
    optimizer_receipt: dict[str, Any]
    edm_receipt: dict[str, Any]
    model_frame_sha: str
    intersection_positions: list[list[float]]
    intersection_receipt: dict[str, Any]
    reconstruction: Any
    model_images: dict[str, Any]
    base_rows: list[dict[str, Any]]
    base_references: dict[str, dict[str, Any]]
    old_descriptors: dict[str, np.ndarray]
    base_image_paths: dict[str, Path]
    base_depth_paths: dict[str, Path]
    base_depth_hashes: dict[str, str]
    expected_new_names: set[str]
    images_by_model_name: dict[str, Path]
    new_metadata: dict[str, dict[str, Any]]
    new_observation_counts: dict[str, int]


def _validate_pack_inputs(
    base_release: Path,
    session_dir: Path,
    optimized_model: Path,
    output_pack: Path,
    model_receipt: Mapping[str, Any],
) -> _ValidatedPack:
    if not isinstance(model_receipt, Mapping):
        raise WorkmapPackError("model_receipt must be a mapping")
    base, output, model = _check_pack_staging(base_release, output_pack, optimized_model)
    bundle, base_root, bundle_sha = _base_bundle(base)
    meta, snapshot, session_keyframes, measured_by_source, session_inputs = _validate_session(
        Path(session_dir), bundle, bundle_sha
    )
    negative_map, keyframe_image_names = _validate_reconstruction_receipt(model_receipt)
    optimizer_receipt = _check_model_receipt(model_receipt, bundle_sha, meta)
    edm_receipt = _validate_edm_provenance(
        model_receipt["edm_validation"], bundle, bundle_sha, base_root, meta, session_inputs
    )

    model_frame_sha = _model_frame_sha(model)
    intersection_positions, intersection_receipt = _validate_intersection_evidence(
        model_receipt, model_frame_sha
    )
    pycolmap, reconstruction, model_images = _open_optimized_model(model)
    model_names = set(model_images)

    (
        base_rows,
        base_references,
        old_descriptors,
        base_image_paths,
        base_depth_paths,
        base_depth_hashes,
    ) = _load_base_assets(base_root)
    base_names = set(base_image_paths)
    expected_new_names = set(keyframe_image_names.values())
    _check_model_catalog(
        model_names, base_names, expected_new_names, keyframe_image_names, session_keyframes
    )

    images_by_model_name, new_metadata, new_observation_counts = _validate_new_keyframes(
        session_inputs["directory"],
        session_keyframes,
        keyframe_image_names,
        reconstruction,
        negative_map,
        measured_by_source,
        session_inputs["keyframe_images_sha256"],
    )
    if set(images_by_model_name) != expected_new_names:
        raise WorkmapPackError("validated session image set differs from optimized model receipt")

    # Check that each newly admitted workmap component reaches unchanged base geometry.
    base_reconstruction, base_point_ids = _base_anchor_points(
        pycolmap, base_root, negative_map, new_metadata
    )
    _check_base_cameras(
        base_reconstruction, reconstruction, model_images, base_rows, expected_new_names
    )
    new_image_ids = {
        int(image.image_id): name
        for name, image in model_images.items()
        if name in expected_new_names
    }
    base_image_ids = {
        int(image.image_id) for name, image in model_images.items() if name in base_names
    }
    source_name_by_image = {
        image_name: source_name for source_name, image_name in keyframe_image_names.items()
    }
    _check_negative_tracks(
        negative_map, reconstruction, new_image_ids, source_name_by_image, new_metadata
    )
    unanchored = _unanchored_workmap_images(
        new_image_ids, model_images, reconstruction, base_image_ids, base_point_ids
    )
    if unanchored:
        raise WorkmapPackError(
            f"optimized workmap images are not connected by verified tracks to the base model: {sorted(unanchored)[:5]}"
        )
    return _ValidatedPack(
        output=output,
        model=model,
        bundle=bundle,
        bundle_sha=bundle_sha,
        meta=meta,
        session_inputs=session_inputs,
        model_receipt=model_receipt,
        negative_map=negative_map,
        keyframe_image_names=keyframe_image_names,
        optimizer_receipt=optimizer_receipt,
        edm_receipt=edm_receipt,
        model_frame_sha=model_frame_sha,
        intersection_positions=intersection_positions,
        intersection_receipt=intersection_receipt,
        reconstruction=reconstruction,
        model_images=model_images,
        base_rows=base_rows,
        base_references=base_references,
        old_descriptors=old_descriptors,
        base_image_paths=base_image_paths,
        base_depth_paths=base_depth_paths,
        base_depth_hashes=base_depth_hashes,
        expected_new_names=expected_new_names,
        images_by_model_name=images_by_model_name,
        new_metadata=new_metadata,
        new_observation_counts=new_observation_counts,
    )


def _check_pack_disk_space(pack: _ValidatedPack) -> None:
    staging_parent = pack.output.parent
    required_copy_bytes = sum(path.stat().st_size for path in pack.base_image_paths.values())
    required_copy_bytes += sum(path.stat().st_size for path in pack.base_depth_paths.values())
    required_copy_bytes += sum(path.stat().st_size for path in pack.images_by_model_name.values())
    # Leave room for the merged BoQ bank, manifests, and filesystem metadata.
    required_free_bytes = int(required_copy_bytes * 1.05) + 128 * 1024 * 1024
    if shutil.disk_usage(staging_parent).free < required_free_bytes:
        raise WorkmapPackError(
            f"insufficient free space for independent pack copies: need about {required_free_bytes} bytes, "
            f"have {shutil.disk_usage(staging_parent).free} bytes; hardlinks are forbidden"
        )


@dataclass(frozen=True)
class _StagingLayout:
    root: Path

    @property
    def map(self) -> Path:
        return self.root / "map"

    @property
    def keyframes(self) -> Path:
        return self.map / "keyframes"

    @property
    def depth(self) -> Path:
        return self.map / "experiments/edm_fixedpose_retriangulate/depth_moge3"

    @property
    def localization(self) -> Path:
        return self.map / "localization"

    @property
    def model(self) -> Path:
        return self.map / "experiments/edm_fixedpose_retriangulate/model"

    @property
    def images(self) -> Path:
        return self.keyframes / "images"


def _stage_base_images(
    pack: _ValidatedPack, layout: _StagingLayout
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    base_image_hashes: dict[str, str] = {}
    base_rewritten_rows: list[dict[str, Any]] = []
    for original in pack.base_rows:
        name = _identity_from_keyframe(original, label="base keyframe")
        source = pack.base_image_paths[name]
        width, height = original["image_size"]
        digest = original["image_sha256"]
        target = layout.images / name
        copied_sha = _copy_file(source, target, label=f"base image {name}")
        if copied_sha != digest:
            raise WorkmapPackError(f"base image hash changed during pack copy: {name}")
        base_image_hashes[name] = digest
        row = dict(original)
        row["output_name"] = name
        row["image_uri"] = f"images/{name}"
        row["image_sha256"] = digest
        row["image_size"] = [width, height]
        base_rewritten_rows.append(row)
    return base_image_hashes, base_rewritten_rows


def _stage_new_images(
    pack: _ValidatedPack, layout: _StagingLayout
) -> tuple[dict[str, Path], dict[str, str], list[dict[str, Any]]]:
    new_image_paths: dict[str, Path] = {}
    new_image_hashes: dict[str, str] = {}
    new_rows: list[dict[str, Any]] = []
    for name in sorted(pack.expected_new_names):
        source = pack.images_by_model_name[name]
        metadata = pack.new_metadata[name]
        destination = layout.images / name
        digest = _copy_file(source, destination, label=f"new image {name}")
        if digest != metadata["image_sha256"]:
            raise WorkmapPackError(f"new image hash changed during pack copy: {name}")
        new_image_paths[name] = destination
        new_image_hashes[name] = digest
        new_rows.append(
            {
                "frame_id": name,
                "keyframe_id": name,
                "output_name": name,
                "image_uri": f"images/{name}",
                "image_sha256": digest,
                "image_size": metadata["image_size"],
                "video_id": "workmap",
                "frame_index": metadata["ordinal"],
                "session_id": pack.meta["session_id"],
                "source_epoch": metadata["source_epoch"],
                "stamp": metadata["stamp"],
                "K": metadata["K"],
                "cam_from_world": metadata["cam_from_world"],
                "mapping_mode": "OPTIMIZED_ONLINE_WORKMAP",
                "observation_count": pack.new_observation_counts[name],
            }
        )
    return new_image_paths, new_image_hashes, new_rows


def _stage_raw_depth(pack: _ValidatedPack, layout: _StagingLayout) -> dict[str, str]:
    # Bind every raw depth archive to its original image; depth is image-only,
    # and is deliberately not rescaled or rewritten after map optimization.
    depth_hashes: dict[str, str] = {}
    for name, source in sorted(pack.base_depth_paths.items()):
        target = layout.depth / f"{name}.npz"
        copied_hash = _copy_file(source, target, label=f"raw depth {name}")
        if copied_hash != pack.base_depth_hashes[name]:
            raise WorkmapPackError(f"base raw depth hash changed during pack copy: {name}")
        depth_hashes[name] = copied_hash
    depth_names = set(pack.base_depth_paths)
    if depth_names.intersection(pack.expected_new_names):
        raise WorkmapPackError(
            "new workmap image unexpectedly collides with a base raw-depth identity"
        )
    return depth_hashes


def _old_reference_rows(
    pack: _ValidatedPack, copied_image_hashes: Mapping[str, str], depth_hashes: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    # All old base image identities must be byte-identical before preserving a row.
    old_reference_rows: dict[str, dict[str, Any]] = {}
    for name in pack.base_references:
        if name not in pack.model_images:
            raise WorkmapPackError(f"base reference image is absent from optimized model: {name}")
        _xy, _by_point = _image_observations(pack.model_images[name], unique_tracks=False)
        count = len(_xy)
        if count <= 0:
            raise WorkmapPackError(f"base reference lost all optimized 2D↔3D observations: {name}")
        old_reference_rows[name] = {
            "image_name": name,
            "image_path": f"images/{name}",
            "video_id": name.split("/", 1)[0],
            "observations": count,
            "image_sha256": copied_image_hashes[name],
            "depth_path": f"{name}.npz" if name in depth_hashes else None,
            "depth_source": "moge3-g-raw-image-only" if name in depth_hashes else None,
        }
    return old_reference_rows


def _new_reference_rows(
    pack: _ValidatedPack, new_image_hashes: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "image_name": name,
            "image_path": f"images/{name}",
            "video_id": "workmap",
            "observations": pack.new_observation_counts[name],
            "image_sha256": new_image_hashes[name],
            "depth_path": None,
            "depth_source": "optimized-colmap-sparse-observations",
            "dense_depth_note": "No MoGe3-G weights/output were available for this session reference; sparse optimized tracks only.",
        }
        for name in sorted(pack.expected_new_names)
        if pack.new_metadata[name]["reference_ready"]
    }


def _reference_catalog(
    old_reference_rows: dict[str, dict[str, Any]], new_reference_rows: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    all_reference_rows = {**old_reference_rows, **new_reference_rows}
    if len(all_reference_rows) != len(old_reference_rows) + len(new_reference_rows):
        raise WorkmapPackError("duplicate reference identity across base and workmap catalogs")
    ref_names = sorted(all_reference_rows)
    for name in ref_names:
        _safe_relative(name, label="reference image name", components=2)
    return all_reference_rows, ref_names


def _write_catalog_manifests(
    layout: _StagingLayout,
    keyframe_rows: list[dict[str, Any]],
    all_reference_rows: Mapping[str, dict[str, Any]],
    ref_names: list[str],
) -> None:
    layout.localization.mkdir(parents=True, exist_ok=True)
    with (layout.keyframes / "keyframes.jsonl").open("wb") as stream:
        for row in sorted(keyframe_rows, key=lambda item: item["output_name"]):
            stream.write(_json_bytes(row))
    with (layout.localization / "reference_manifest.jsonl").open("wb") as stream:
        for name in ref_names:
            stream.write(_json_bytes(all_reference_rows[name]))


def _check_staged_catalog(
    pack: _ValidatedPack,
    layout: _StagingLayout,
    image_sha_by_name: Mapping[str, str],
    all_reference_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    # The pack must never carry builder outputs from the base release.
    forbidden = ("direct_geometry.npz", "map.ply", "reference_poses.json", "T_align_gravity.json")
    for name in forbidden:
        if (layout.map / name).exists() or (layout.localization / name).exists():
            raise WorkmapPackError(f"stale builder-derived asset was unexpectedly staged: {name}")
    missing_manifest = set(pack.model_images) - set(image_sha_by_name)
    if missing_manifest:
        raise WorkmapPackError(
            f"keyframe image manifest does not cover optimized model images: {sorted(missing_manifest)[:5]}"
        )
    for name, image in pack.model_images.items():
        if (
            name in pack.expected_new_names
            and pack.new_metadata[name]["reference_ready"]
            and name not in all_reference_rows
        ):
            raise WorkmapPackError(
                f"measured optimized keyframe was not added to reference catalog: {name}"
            )
        if name in all_reference_rows:
            expected_count = int(all_reference_rows[name]["observations"])
            actual_count = len(_image_observations(image, unique_tracks=False)[0])
            if expected_count != actual_count:
                raise WorkmapPackError(
                    f"reference observation count changed while assembling: {name}"
                )


def _pack_notes(expected_new_names: set[str]) -> tuple[str, str]:
    # Human-readable pack notes explicitly distinguish raw depth from
    # map-aligned sparse geometry and refuse flight/activation claims.
    missing_dense = sorted(expected_new_names)
    readme = (
        "# Offline optimized workmap deploy pack\n\n"
        "This pack was assembled from a completed, base-SHA-bound online workmap and a globally optimized COLMAP model. "
        "It is an offline next-mission candidate only; this build does not select, activate, or approve it for flight.\n\n"
        "The base release keyframe images and image-only MoGe3-G raw depth predictions are independent copies. Raw depth is not "
        "geometry-aligned; runtime derives its sparse-depth alignment from this pack's optimized model. New workmap references "
        "have measured image observations and optimized sparse COLMAP tracks but no dense MoGe3-G depth prediction.\n\n"
        f"New workmap keyframe images without dense depth ({len(missing_dense)}):\n"
        + "\n".join(f"- `{name}`" for name in missing_dense)
        + "\n"
    )
    p174 = (
        "# P174 and next-mission workmap release\n\n"
        "This pack is generated offline after session completion. It includes the globally optimized model, verified base and "
        "session images, rebuilt occupancy, recomputed FIM∩LWTL weak cells, and an actual BoQ bank. The frozen source release "
        "was not modified or hard-linked.\n\n"
        "The MoGe3-G archives retained from the base release are raw per-image predictions and are not copied as sparse/model-aligned "
        "depth. New session keyframes have no dense depth; they rely on their measured optimized sparse tracks.\n\n"
        "This is not an accuracy or site-approval claim. No mission is selected or activated by pack assembly. The next mission still "
        "requires independent coordinate/route/obstacle/safety review and operator selection after the current mission is stopped.\n"
    )
    return readme, p174


def _pack_outputs(layout: _StagingLayout, intersection_path: Path) -> dict[str, Any]:
    output_keyframes = layout.keyframes / "keyframes.jsonl"
    output_reference_manifest = layout.localization / "reference_manifest.jsonl"
    output_names = layout.localization / "boq_references_sideview.names.json"
    output_bank = layout.localization / "boq_references_sideview.npy"
    outputs: dict[str, Any] = {
        "keyframes_manifest": {
            "path": "map/keyframes/keyframes.jsonl",
            "sha256": _sha256(output_keyframes),
        },
        "reference_manifest": {
            "path": "map/localization/reference_manifest.jsonl",
            "sha256": _sha256(output_reference_manifest),
        },
        "boq_descriptors": {
            "path": "map/localization/boq_references_sideview.npy",
            "sha256": _sha256(output_bank),
        },
        "boq_names": {
            "path": "map/localization/boq_references_sideview.names.json",
            "sha256": _sha256(output_names),
        },
        "intersection_cells": {
            "path": "map/localization/fim_lwtl_intersection_cells.json",
            "sha256": _sha256(intersection_path),
        },
    }
    for root_name, root in (("keyframe_images", layout.images), ("depth", layout.depth)):
        files = _ensure_no_symlink_tree(
            root, label="staged images" if root_name == "keyframe_images" else "staged raw depth"
        )
        outputs[root_name] = {
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "files": {path.relative_to(root).as_posix(): _sha256(path) for path in files},
        }
    return outputs


def _write_staged_pack(pack: _ValidatedPack, layout: _StagingLayout) -> dict[str, Any]:
    base_image_hashes, base_rewritten_rows = _stage_base_images(pack, layout)
    new_image_paths, new_image_hashes, new_rows = _stage_new_images(pack, layout)
    depth_hashes = _stage_raw_depth(pack, layout)
    copied_image_hashes = {**base_image_hashes, **new_image_hashes}
    old_reference_rows = _old_reference_rows(pack, copied_image_hashes, depth_hashes)
    new_reference_rows = _new_reference_rows(pack, new_image_hashes)
    all_reference_rows, ref_names = _reference_catalog(old_reference_rows, new_reference_rows)
    _write_catalog_manifests(
        layout, [*base_rewritten_rows, *new_rows], all_reference_rows, ref_names
    )

    # Validate every retained base row against the byte-identical image and
    # current observation catalog before preserving its original descriptor.
    old_image_hashes = {name: _sha256(path) for name, path in pack.base_image_paths.items()}
    bank_summary = _write_boq_bank(
        layout.localization,
        pack.old_descriptors,
        old_image_hashes,
        sorted(new_reference_rows),
        new_image_paths,
        copied_image_hashes,
    )
    if bank_summary["names"] != sorted(set(pack.old_descriptors) | set(new_reference_rows)):
        raise WorkmapPackError(
            "assembled BoQ bank does not match the base+measured-workmap reference catalog"
        )

    occupancy = _occupancy_payload(pack.reconstruction)
    _write_json(layout.model / "occupancy.json", occupancy)
    intersection_path = layout.localization / "fim_lwtl_intersection_cells.json"
    _write_json(
        intersection_path,
        _intersection_payload(
            pack.intersection_positions,
            map_name=f"optimized_{pack.model_frame_sha[:12]}",
            evidence=pack.intersection_receipt,
        ),
    )
    _check_staged_catalog(
        pack, layout, {**base_image_hashes, **new_image_hashes}, all_reference_rows
    )

    readme, p174 = _pack_notes(pack.expected_new_names)
    (layout.root / "README.md").write_text(readme, encoding="utf-8")
    (layout.root / "P174_AND_NEXT.md").write_text(p174, encoding="utf-8")
    outputs = _pack_outputs(layout, intersection_path)
    session_inputs = pack.session_inputs
    model = pack.model
    new_metadata = pack.new_metadata
    receipt = {
        "schema": "online-workmap-pack-assets/v1",
        "base_release_id": pack.bundle["map_revision_id"],
        "base_coordinate_frame_id": pack.bundle["coordinate_frame_id"],
        "base_bundle_sha256": pack.bundle_sha,
        "session_id": pack.meta["session_id"],
        "session_sha256": {
            "session_meta.json": _sha256(session_inputs["meta_path"]),
            "snapshot.json": _sha256(session_inputs["snapshot_path"]),
            "keyframes.jsonl": _sha256(session_inputs["keyframes_path"]),
            "observations.jsonl": _sha256(session_inputs["observations_path"]),
        },
        "model_frame_sha256": pack.model_frame_sha,
        "model_bins": {
            name: {"sha256": _sha256(model / name), "size_bytes": (model / name).stat().st_size}
            for name in MODEL_BINS
        },
        "model_counts": {
            "images": int(pack.reconstruction.num_images()),
            "points3D": int(pack.reconstruction.num_points3D()),
        },
        "negative_id_map": {str(key): value for key, value in sorted(pack.negative_map.items())},
        "keyframe_image_names": dict(sorted(pack.keyframe_image_names.items())),
        "gauge": pack.model_receipt["gauge"],
        "edm_validation": pack.edm_receipt,
        "triangulation": pack.model_receipt.get("triangulation"),
        "filtering": pack.optimizer_receipt["filtering"],
        "bundle_adjustment": pack.optimizer_receipt["bundle_adjustment"],
        "new_keyframes": [
            {
                "session_name": original,
                "image_name": name,
                "image_sha256": new_image_hashes[name],
                "image_size": new_metadata[name]["image_size"],
                "observations": pack.new_observation_counts[name],
                "measured_observations_matching_model": new_metadata[name][
                    "matched_measured_observations"
                ],
                "matched_negative_ids": new_metadata[name]["matched_negative_ids"],
                "matched_positive_ids": new_metadata[name]["matched_positive_ids"],
                "reference_ready": new_metadata[name]["reference_ready"],
                "dense_depth": None,
            }
            for original, name in sorted(pack.keyframe_image_names.items(), key=lambda row: row[1])
        ],
        "base_assets": {
            "images": {
                name: {
                    "sha256": base_image_hashes[name],
                    "size_bytes": pack.base_image_paths[name].stat().st_size,
                }
                for name in sorted(pack.base_image_paths)
            },
            "raw_depth_moge3_g": {
                name: {
                    "sha256": depth_hashes[name],
                    "size_bytes": pack.base_depth_paths[name].stat().st_size,
                }
                for name in sorted(pack.base_depth_paths)
            },
        },
        "reference_catalog": {
            "count": len(ref_names),
            "names": ref_names,
            "new_reference_count": len(new_reference_rows),
        },
        "boq_bank": {key: value for key, value in bank_summary.items() if key != "names"},
        "occupancy_recomputed": occupancy,
        "intersection_cells_status": pack.intersection_receipt,
        "outputs": outputs,
        "copy_mode": "independent-copy-no-hardlinks-no-symlinks",
        "safety": {"selected": False, "activated": False, "flight_approved": False},
    }
    _write_json(layout.root / "workmap_pack_receipt.json", receipt)
    return receipt


def _promote_pack(pack: _ValidatedPack, layout: _StagingLayout, committed: list[Path]) -> None:
    # Commit all assets except the pre-existing optimized model; the caller
    # rolls back promoted paths if a move fails, leaving only its own model.
    output = pack.output
    promotion = (
        (layout.keyframes, output / "map/keyframes"),
        (layout.depth, output / "map/experiments/edm_fixedpose_retriangulate/depth_moge3"),
        (layout.localization, output / "map/localization"),
        (layout.model / "occupancy.json", pack.model / "occupancy.json"),
        (layout.root / "README.md", output / "README.md"),
        (layout.root / "P174_AND_NEXT.md", output / "P174_AND_NEXT.md"),
        (layout.root / "workmap_pack_receipt.json", output / "workmap_pack_receipt.json"),
    )
    for source, destination in promotion:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise WorkmapPackError(f"refusing to overwrite staged pack asset: {destination}")
        os.replace(source, destination)
        committed.append(destination)


def assemble_workmap_pack(
    base_release: Path,
    session_dir: Path,
    optimized_model: Path,
    output_pack: Path,
    model_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete deploy-pack assets around the already optimized COLMAP model.

    ``output_pack`` is an existing staging directory. Its only pre-existing content
    may be ``map/experiments/edm_fixedpose_retriangulate/model`` written by
    ``reconstruct_workmap``. The model is never replaced or modified here.
    """
    pack = _validate_pack_inputs(
        base_release, session_dir, optimized_model, output_pack, model_receipt
    )
    _check_pack_disk_space(pack)
    temp_root = Path(tempfile.mkdtemp(prefix=".workmap-pack-assets-", dir=pack.output.parent))
    committed: list[Path] = []
    try:
        layout = _StagingLayout(temp_root)
        receipt = _write_staged_pack(pack, layout)
        _promote_pack(pack, layout, committed)
        return receipt
    except BaseException:
        for path in reversed(committed):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
