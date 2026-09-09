"""Relocatable, integrity-checked map assets for the ``direct`` backend.

``direct_bundle.json`` (schema ``direct-localization-bundle/v1``) is the single
entry point: every path inside it is relative to the bundle file itself, so a
release directory can be copied anywhere and still verify.  The mapping machine
wrote absolute ``/home/cihcilab/...`` URIs into ``keyframes.jsonl``; those are
deliberately ignored here and re-derived from the bundle root.

Two integrity tiers:

* ``files[]`` — model binaries, manifests, MegaLoc bank, intersection cells.
  Digest and size are checked eagerly at load (~600 MB, once per process).
* keyframe JPEGs — far too many to hash eagerly, so each one is verified
  against ``keyframes.jsonl``'s ``image_sha256`` the first time the relocalizer
  actually prepares it (:meth:`DirectMapAssets.verify_reference_image`).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from direct_paths import sha256_file, verify_file_sha256


DIRECT_BUNDLE_SCHEMA = "direct-localization-bundle/v1"
REFERENCE_POSES_SCHEMA = "reference-poses/v2"
REFERENCE_POSES_NAME = "reference_poses.json"

_BUNDLE_KEYS = frozenset(
    {
        "schema",
        "map_revision_id",
        "coordinate_frame_id",
        "model_dir",
        "keyframes_manifest",
        "keyframes_images_root",
        "reference_manifest",
        "reference_bank",
        "intersection_cells",
        "reference_depth_dir",
        "model_sha256",
        "files",
    }
)
_BANK_KEYS = frozenset({"name", "descriptors", "names"})
_FILE_ENTRY_KEYS = frozenset({"path", "sha256", "size_bytes"})
_MODEL_BINARIES = ("cameras.bin", "images.bin", "points3D.bin")


class DirectMapError(ValueError):
    """Raised when a direct localization bundle violates its frozen contract."""


def _reject_json_constant(value: str):
    raise DirectMapError(f"non-finite JSON number is not allowed: {value}")


def release_root_of(bundle_dir: Path) -> Path:
    """The release directory that owns a ``localization/`` bundle directory."""

    return Path(bundle_dir).parent


def reject_symlink_components(path: Path, *, root: Path) -> Path:
    """Refuse a path that leaves ``root`` or traverses a symlink below it.

    A release directory is a signed artefact.  Allowing a symlink anywhere below
    it would let a digest computed through one inode be served from another.
    """

    current = Path(root)
    if current.is_symlink():
        raise DirectMapError(f"release root must not be a symlink: {current}")
    try:
        relative = Path(path).relative_to(current)
    except ValueError as exc:
        raise DirectMapError(f"{path} escapes the release root {current}") from exc
    for part in relative.parts:
        if part in ("..", ""):
            raise DirectMapError(f"bundle path may not traverse upwards: {path}")
        current = current / part
        if current.is_symlink():
            raise DirectMapError(f"bundle path component is a symlink: {current}")
    return current


def _relative_path(raw: object, *, field: str, root: Path, source: Path) -> Path:
    """Resolve one bundle-relative path and hold it inside the release root."""

    if not isinstance(raw, str) or not raw:
        raise DirectMapError(f"bundle {field} must be a non-empty relative path: {source}")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise DirectMapError(f"bundle {field} must be relative to the bundle: {source}")
    # normpath, not resolve: ".." is a lexical step inside the release, while
    # resolve() would silently follow a symlink before the check below runs.
    joined = Path(os.path.normpath(str(root / candidate)))
    reject_symlink_components(joined, root=release_root_of(root))
    if not joined.exists():
        raise DirectMapError(f"bundle {field} does not exist: {joined}")
    return joined


def _verify_declared_files(entries: object, *, root: Path, source: Path) -> tuple[Path, ...]:
    if not isinstance(entries, list) or not entries:
        raise DirectMapError(f"bundle files[] must be a non-empty array: {source}")
    verified: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise DirectMapError(f"bundle files[] entries must be objects: {source}")
        missing = sorted(_FILE_ENTRY_KEYS - entry.keys())
        unknown = sorted(entry.keys() - _FILE_ENTRY_KEYS)
        if missing or unknown:
            raise DirectMapError(
                f"bundle files[] entry missing={missing} unknown={unknown}: {source}"
            )
        path = _relative_path(entry["path"], field="files[].path", root=root, source=source)
        if not path.is_file():
            raise DirectMapError(f"bundle files[] entry is not a regular file: {path}")
        size = entry["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise DirectMapError(f"bundle files[].size_bytes must be an integer: {path}")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise DirectMapError(
                f"bundle files[] size mismatch for {path}: expected {size}, got {actual_size}"
            )
        verify_file_sha256(path, entry["sha256"])
        verified.append(path)
    return tuple(verified)


def _verify_model_binaries(declared: object, *, model_dir: Path, source: Path) -> None:
    if not isinstance(declared, dict):
        raise DirectMapError(f"bundle model_sha256 must be an object: {source}")
    missing = sorted(set(_MODEL_BINARIES) - declared.keys())
    unknown = sorted(declared.keys() - set(_MODEL_BINARIES))
    if missing or unknown:
        raise DirectMapError(
            f"bundle model_sha256 missing={missing} unknown={unknown}: {source}"
        )
    for name in _MODEL_BINARIES:
        binary = model_dir / name
        if not binary.is_file():
            raise DirectMapError(f"COLMAP model binary is absent: {binary}")
        verify_file_sha256(binary, declared[name])


def _load_reference_names(names_path: Path, descriptors_path: Path) -> tuple[str, ...]:
    payload = json.loads(names_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise DirectMapError(f"MegaLoc bank names must be a non-empty array: {names_path}")
    names = tuple(str(name) for name in payload)
    if len(set(names)) != len(names):
        raise DirectMapError(f"MegaLoc bank names must be unique: {names_path}")
    # mmap keeps this to a header read; the bank is ~32 MB but the check is
    # only about row count agreeing with the identity list.
    shape = np.load(descriptors_path, mmap_mode="r", allow_pickle=False).shape
    if len(shape) != 2 or shape[0] != len(names):
        raise DirectMapError(
            f"MegaLoc bank {descriptors_path} has shape {shape} but "
            f"{len(names)} identities in {names_path}"
        )
    return names


def _reference_pose_arrays(
    poses_path: Path, ref_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(poses_path.read_text(encoding="utf-8"))
    if payload.get("schema") != REFERENCE_POSES_SCHEMA:
        raise DirectMapError(
            f"reference poses schema must be {REFERENCE_POSES_SCHEMA!r}: {poses_path}"
        )
    poses = payload.get("poses")
    if not isinstance(poses, dict):
        raise DirectMapError(f"reference poses payload must map names to poses: {poses_path}")
    centers = np.zeros((len(ref_names), 3), dtype=np.float32)
    yaws = np.zeros(len(ref_names), dtype=np.float32)
    for index, name in enumerate(ref_names):
        entry = poses.get(name)
        if not isinstance(entry, dict):
            raise DirectMapError(f"reference pose is absent for {name}: {poses_path}")
        rotation = np.asarray(entry["R"], dtype=np.float64)
        translation = np.asarray(entry["t"], dtype=np.float64).reshape(3)
        if rotation.shape != (3, 3):
            raise DirectMapError(f"reference pose R must be 3x3 for {name}: {poses_path}")
        centers[index] = -rotation.T @ translation
        forward = rotation.T[:, 2]
        yaws[index] = math.degrees(math.atan2(float(forward[1]), float(forward[0]))) % 360.0
    centers.flags.writeable = False
    yaws.flags.writeable = False
    return centers, yaws


@dataclass(frozen=True)
class DirectMapAssets:
    """Verified, relocatable handles to one direct localization release."""

    bundle_path: Path
    root: Path
    map_revision_id: str
    coordinate_frame_id: str
    model_dir: Path
    keyframes_manifest: Path
    keyframes_images_root: Path
    reference_manifest: Path
    bank_name: str
    bank_descriptors: Path
    bank_names: Path
    intersection_cells: Path | None
    reference_depth_dir: Path | None
    ref_names: tuple[str, ...]
    ref_centers: np.ndarray | None
    ref_yaws: np.ndarray | None
    sha256: str
    raw: Mapping[str, Any]

    @classmethod
    def load(
        cls, bundle_json: str | Path, *, expected_sha256: str | None = None
    ) -> "DirectMapAssets":
        source = Path(bundle_json).expanduser().resolve(strict=True)
        digest = verify_file_sha256(source, expected_sha256)
        root = source.parent
        raw = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        if not isinstance(raw, dict):
            raise DirectMapError(f"direct bundle must be a JSON object: {source}")
        if raw.get("schema") != DIRECT_BUNDLE_SCHEMA:
            raise DirectMapError(f"direct bundle schema must be {DIRECT_BUNDLE_SCHEMA!r}: {source}")
        missing = sorted(_BUNDLE_KEYS - raw.keys())
        unknown = sorted(raw.keys() - _BUNDLE_KEYS)
        if missing:
            raise DirectMapError(f"direct bundle is missing {missing}: {source}")
        if unknown:
            raise DirectMapError(f"direct bundle has unknown keys {unknown}: {source}")
        for field in ("map_revision_id", "coordinate_frame_id"):
            if not isinstance(raw[field], str) or not raw[field]:
                raise DirectMapError(f"direct bundle {field} must be a non-empty string: {source}")

        model_dir = _relative_path(raw["model_dir"], field="model_dir", root=root, source=source)
        keyframes_manifest = _relative_path(
            raw["keyframes_manifest"], field="keyframes_manifest", root=root, source=source
        )
        keyframes_images_root = _relative_path(
            raw["keyframes_images_root"], field="keyframes_images_root", root=root, source=source
        )
        reference_manifest = _relative_path(
            raw["reference_manifest"], field="reference_manifest", root=root, source=source
        )
        bank = raw["reference_bank"]
        if not isinstance(bank, dict):
            raise DirectMapError(f"direct bundle reference_bank must be an object: {source}")
        bank_missing = sorted(_BANK_KEYS - bank.keys())
        bank_unknown = sorted(bank.keys() - _BANK_KEYS)
        if bank_missing or bank_unknown:
            raise DirectMapError(
                f"reference_bank missing={bank_missing} unknown={bank_unknown}: {source}"
            )
        bank_descriptors = _relative_path(
            bank["descriptors"], field="reference_bank.descriptors", root=root, source=source
        )
        bank_names = _relative_path(
            bank["names"], field="reference_bank.names", root=root, source=source
        )
        intersection_cells = (
            None
            if raw["intersection_cells"] is None
            else _relative_path(
                raw["intersection_cells"], field="intersection_cells", root=root, source=source
            )
        )
        reference_depth_dir = (
            None
            if raw["reference_depth_dir"] is None
            else _relative_path(
                raw["reference_depth_dir"], field="reference_depth_dir", root=root, source=source
            )
        )
        if not model_dir.is_dir():
            raise DirectMapError(f"model_dir must be a directory: {model_dir}")
        if not keyframes_images_root.is_dir():
            raise DirectMapError(f"keyframes_images_root must be a directory: {keyframes_images_root}")
        if reference_depth_dir is not None and not reference_depth_dir.is_dir():
            raise DirectMapError(f"reference_depth_dir must be a directory: {reference_depth_dir}")

        _verify_model_binaries(raw["model_sha256"], model_dir=model_dir, source=source)
        _verify_declared_files(raw["files"], root=root, source=source)

        ref_names = _load_reference_names(bank_names, bank_descriptors)
        poses_path = root / REFERENCE_POSES_NAME
        if poses_path.is_file():
            ref_centers, ref_yaws = _reference_pose_arrays(poses_path, ref_names)
        else:
            ref_centers, ref_yaws = None, None

        return cls(
            bundle_path=source,
            root=root,
            map_revision_id=str(raw["map_revision_id"]),
            coordinate_frame_id=str(raw["coordinate_frame_id"]),
            model_dir=model_dir,
            keyframes_manifest=keyframes_manifest,
            keyframes_images_root=keyframes_images_root,
            reference_manifest=reference_manifest,
            bank_name=str(bank["name"]),
            bank_descriptors=bank_descriptors,
            bank_names=bank_names,
            intersection_cells=intersection_cells,
            reference_depth_dir=reference_depth_dir,
            ref_names=ref_names,
            ref_centers=ref_centers,
            ref_yaws=ref_yaws,
            sha256=digest,
            raw=raw,
        )

    # -- keyframes ---------------------------------------------------------

    def keyframe_index(self) -> dict[str, dict]:
        """Map ``<video_id>/<frame>.jpg`` to a relocatable keyframe record.

        The mapping machine's absolute ``image_uri`` is discarded: the last two
        components of it name the image, and the bytes live under this
        release's ``keyframes_images_root``.
        """

        cached = self.__dict__.get("_keyframe_index_cache")
        if cached is not None:
            return cached
        index: dict[str, dict] = {}
        with self.keyframes_manifest.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                name = _identity_from_row(row, self.keyframes_manifest)
                if name in index:
                    raise DirectMapError(f"keyframe identity is not unique: {name}")
                digest = row.get("image_sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    raise DirectMapError(f"keyframe {name} has no usable image_sha256")
                index[name] = {
                    "image_uri": str(self.keyframes_images_root / name),
                    "video_id": str(row["video_id"]) if row.get("video_id") else name.split("/")[0],
                    "image_sha256": digest,
                }
        if not index:
            raise DirectMapError(f"keyframe manifest is empty: {self.keyframes_manifest}")
        object.__setattr__(self, "_keyframe_index_cache", index)
        return index

    def verify_reference_image(self, name: str) -> Path:
        """Verify one keyframe JPEG against the manifest digest, once."""

        verified = self.__dict__.get("_verified_images")
        if verified is None:
            verified = {}
            object.__setattr__(self, "_verified_images", verified)
        cached = verified.get(name)
        if cached is not None:
            return cached
        record = self.keyframe_index().get(name)
        if record is None:
            raise DirectMapError(f"reference identity is absent from the keyframe manifest: {name}")
        path = Path(record["image_uri"])
        reject_symlink_components(path, root=release_root_of(self.root))
        if not path.is_file():
            raise DirectMapError(f"reference image is absent: {path}")
        actual = sha256_file(path)
        if actual != record["image_sha256"]:
            raise DirectMapError(
                f"reference image digest mismatch for {name}: "
                f"expected {record['image_sha256']}, got {actual}"
            )
        verified[name] = path
        return path


def _identity_from_row(row: Mapping[str, Any], source: Path) -> str:
    """``<video_id>/<frame>.jpg`` — the identity the COLMAP model uses."""

    output_name = row.get("output_name")
    if isinstance(output_name, str) and output_name.count("/") == 1:
        return output_name
    uri = row.get("image_uri")
    if not isinstance(uri, str) or not uri:
        raise DirectMapError(f"keyframe row has neither output_name nor image_uri: {source}")
    parts = Path(uri).parts
    if len(parts) < 2:
        raise DirectMapError(f"keyframe image_uri is too shallow to identify: {uri}")
    return f"{parts[-2]}/{parts[-1]}"
