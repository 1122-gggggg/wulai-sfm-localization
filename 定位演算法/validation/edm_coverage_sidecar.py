"""Exact production-EDM landmark observation tracks for offline coverage analysis."""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SIDECAR_SCHEMA = "edm-coverage-observations"
SIDECAR_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_names_sha256(names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        encoded = str(name).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class CoverageRecords:
    """CSR observations for landmarks identified by production EDM anchor cells."""

    anchor_ref_idx: np.ndarray
    anchor_cell_idx: np.ndarray
    obs_offsets: np.ndarray
    obs_ref_idx: np.ndarray

    def as_payload(self) -> dict[str, np.ndarray]:
        return {
            "anchor_ref_idx": self.anchor_ref_idx,
            "anchor_cell_idx": self.anchor_cell_idx,
            "obs_offsets": self.obs_offsets,
            "obs_ref_idx": self.obs_ref_idx,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "CoverageRecords":
        return cls(
            np.asarray(payload["anchor_ref_idx"], dtype=np.int32),
            np.asarray(payload["anchor_cell_idx"], dtype=np.int32),
            np.asarray(payload["obs_offsets"], dtype=np.int64),
            np.asarray(payload["obs_ref_idx"], dtype=np.int32),
        )


def _record_arrays(records: CoverageRecords) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(records.anchor_ref_idx, dtype=np.int32).reshape(-1),
        np.ascontiguousarray(records.anchor_cell_idx, dtype=np.int32).reshape(-1),
        np.ascontiguousarray(records.obs_offsets, dtype=np.int64).reshape(-1),
        np.ascontiguousarray(records.obs_ref_idx, dtype=np.int32).reshape(-1),
    )


def _validate_record_structure(
    anchor_cell: np.ndarray,
    offsets: np.ndarray,
    observations: np.ndarray,
    count: int,
) -> None:
    if len(anchor_cell) != count or len(offsets) != count + 1:
        raise ValueError("coverage sidecar anchor/offset shapes are inconsistent")
    if len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != len(observations):
        raise ValueError("coverage sidecar observation offsets are invalid")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("every coverage landmark must have at least one observation")


def _validate_record_indices(
    anchor_ref: np.ndarray,
    anchor_cell: np.ndarray,
    observations: np.ndarray,
    *,
    ref_count: int,
    cell_count: int | None,
) -> None:
    if np.any((anchor_ref < 0) | (anchor_ref >= int(ref_count))):
        raise ValueError("coverage sidecar anchor reference index is out of range")
    if np.any((observations < 0) | (observations >= int(ref_count))):
        raise ValueError("coverage sidecar observation reference index is out of range")
    if np.any(anchor_cell < 0) or (
        cell_count is not None and np.any(anchor_cell >= int(cell_count))
    ):
        raise ValueError("coverage sidecar anchor cell index is out of range")


def _validate_record_tracks(
    anchor_ref: np.ndarray,
    anchor_cell: np.ndarray,
    offsets: np.ndarray,
    observations: np.ndarray,
) -> None:
    pairs = np.stack((anchor_ref, anchor_cell), axis=1)
    if np.any(pairs[1:, 0] < pairs[:-1, 0]) or np.any(
        (pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] <= pairs[:-1, 1])
    ):
        raise ValueError("coverage sidecar anchors must be unique and canonically sorted")
    lengths = np.diff(offsets)
    observation_anchors = np.repeat(anchor_ref, lengths)
    contains_anchor = np.logical_or.reduceat(
        observations == observation_anchors, offsets[:-1]
    )
    if not np.all(contains_anchor):
        raise ValueError("coverage landmark track does not contain its anchor reference")
    if len(observations) > 1:
        boundary = np.zeros(len(observations) - 1, dtype=bool)
        boundary[offsets[1:-1] - 1] = True
        if np.any((observations[1:] <= observations[:-1]) & ~boundary):
            raise ValueError("coverage landmark tracks must contain sorted unique references")


def validate_records(
    records: CoverageRecords,
    *,
    ref_count: int,
    cell_count: int | None = None,
) -> CoverageRecords:
    anchor_ref, anchor_cell, offsets, observations = _record_arrays(records)
    count = len(anchor_ref)
    _validate_record_structure(anchor_cell, offsets, observations, count)
    _validate_record_indices(
        anchor_ref,
        anchor_cell,
        observations,
        ref_count=ref_count,
        cell_count=cell_count,
    )
    if count:
        _validate_record_tracks(anchor_ref, anchor_cell, offsets, observations)
    return CoverageRecords(anchor_ref, anchor_cell, offsets, observations)


def merge_records(
    chunks: Sequence[CoverageRecords],
    *,
    ref_count: int,
    cell_count: int | None = None,
) -> CoverageRecords:
    checked = [
        validate_records(chunk, ref_count=ref_count, cell_count=cell_count)
        for chunk in chunks
    ]
    if not checked:
        checked = [
            CoverageRecords(
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(1, dtype=np.int64),
                np.zeros(0, dtype=np.int32),
            )
        ]
    anchor_ref = np.concatenate([chunk.anchor_ref_idx for chunk in checked])
    anchor_cell = np.concatenate([chunk.anchor_cell_idx for chunk in checked])
    lengths = np.concatenate([np.diff(chunk.obs_offsets) for chunk in checked])
    source_offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    source_offsets[0] = 0
    np.cumsum(lengths, out=source_offsets[1:])
    source_observations = np.concatenate([chunk.obs_ref_idx for chunk in checked])
    order = np.lexsort((anchor_cell, anchor_ref))
    sorted_ref = anchor_ref[order]
    sorted_cell = anchor_cell[order]
    if len(order) > 1:
        duplicate = (sorted_ref[1:] == sorted_ref[:-1]) & (
            sorted_cell[1:] == sorted_cell[:-1]
        )
        if duplicate.any():
            index = int(np.flatnonzero(duplicate)[0])
            raise ValueError(
                "duplicate coverage anchor while merging: "
                f"({int(sorted_ref[index])}, {int(sorted_cell[index])})"
            )
    sorted_lengths = lengths[order]
    offsets = np.empty(len(sorted_lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sorted_lengths, out=offsets[1:])
    if np.array_equal(order, np.arange(len(order))):
        observations = source_observations
    else:
        observations = np.empty(len(source_observations), dtype=np.int32)
        for target_index, source_index in enumerate(order):
            source_start, source_end = source_offsets[int(source_index) : int(source_index) + 2]
            target_start = offsets[target_index]
            target_end = offsets[target_index + 1]
            observations[int(target_start) : int(target_end)] = source_observations[
                int(source_start) : int(source_end)
            ]
    return validate_records(
        CoverageRecords(
            sorted_ref,
            sorted_cell,
            offsets,
            observations,
        ),
        ref_count=ref_count,
        cell_count=cell_count,
    )


def extract_coverage_records(
    reconstruction,
    tables: dict[str, dict[str, np.ndarray]],
    anchor_names: Sequence[str],
    ref_names: Sequence[str],
) -> CoverageRecords:
    """Extract exact observation image tracks for finite EDM anchor-cell landmarks."""
    ref_index = {name: index for index, name in enumerate(ref_names)}
    image_by_name = {image.name: image for image in reconstruction.images.values()}
    image_name_by_id = {
        int(image.image_id): image.name for image in reconstruction.images.values()
    }
    anchor_ref: list[int] = []
    anchor_cell: list[int] = []
    observations: list[int] = []
    offsets = [0]

    for name in sorted(anchor_names, key=ref_index.__getitem__):
        image = image_by_name.get(name)
        if image is None:
            continue
        idx_of_cell = np.asarray(tables[name]["idx_of_cell"], dtype=np.int64)
        for cell_idx in np.flatnonzero(idx_of_cell >= 0):
            point2d_idx = int(idx_of_cell[cell_idx])
            if point2d_idx >= len(image.points2D):
                continue
            point2d = image.points2D[point2d_idx]
            if not point2d.has_point3D():
                continue
            point3d = reconstruction.points3D[int(point2d.point3D_id)]
            observed = sorted(
                {
                    ref_index[image_name_by_id[int(element.image_id)]]
                    for element in point3d.track.elements
                    if image_name_by_id.get(int(element.image_id)) in ref_index
                }
            )
            if ref_index[name] not in observed:
                raise ValueError(f"triangulated track for {name} cell {cell_idx} omits its anchor")
            anchor_ref.append(ref_index[name])
            anchor_cell.append(int(cell_idx))
            observations.extend(observed)
            offsets.append(len(observations))

    cell_count = max((len(table["idx_of_cell"]) for table in tables.values()), default=0)
    return validate_records(
        CoverageRecords(
            np.asarray(anchor_ref, dtype=np.int32),
            np.asarray(anchor_cell, dtype=np.int32),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(observations, dtype=np.int32),
        ),
        ref_count=len(ref_names),
        cell_count=cell_count,
    )


def write_coverage_sidecar(
    path: str | Path,
    records: CoverageRecords,
    *,
    bundle_path: str | Path,
    ref_names: Sequence[str],
    cell_count: int,
) -> Path:
    target = Path(path).resolve()
    if target.suffix != ".npz":
        raise ValueError("coverage sidecar path must end in .npz")
    checked = validate_records(records, ref_count=len(ref_names), cell_count=cell_count)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_name=np.array(SIDECAR_SCHEMA),
            schema_version=np.array(SIDECAR_VERSION, dtype=np.int64),
            bundle_sha256=np.array(file_sha256(bundle_path)),
            ref_names_sha256=np.array(ordered_names_sha256(ref_names)),
            cell_count=np.array(cell_count, dtype=np.int64),
            **checked.as_payload(),
        )
    os.replace(temporary, target)
    return target


def load_coverage_sidecar(
    path: str | Path,
    *,
    bundle_path: str | Path,
    ref_names: Sequence[str],
    cell_count: int,
) -> CoverageRecords:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "schema_name",
            "schema_version",
            "bundle_sha256",
            "ref_names_sha256",
            "cell_count",
            "anchor_ref_idx",
            "anchor_cell_idx",
            "obs_offsets",
            "obs_ref_idx",
        }
        if set(archive.files) != required:
            raise ValueError("coverage sidecar fields do not match its schema")
        if str(np.asarray(archive["schema_name"]).item()) != SIDECAR_SCHEMA:
            raise ValueError("unsupported coverage sidecar schema")
        if int(np.asarray(archive["schema_version"]).item()) != SIDECAR_VERSION:
            raise ValueError("unsupported coverage sidecar version")
        if str(np.asarray(archive["bundle_sha256"]).item()) != file_sha256(bundle_path):
            raise ValueError("coverage sidecar bundle SHA-256 mismatch")
        if str(np.asarray(archive["ref_names_sha256"]).item()) != ordered_names_sha256(
            ref_names
        ):
            raise ValueError("coverage sidecar ordered reference names mismatch")
        if int(np.asarray(archive["cell_count"]).item()) != int(cell_count):
            raise ValueError("coverage sidecar EDM cell count mismatch")
        records = CoverageRecords(
            archive["anchor_ref_idx"],
            archive["anchor_cell_idx"],
            archive["obs_offsets"],
            archive["obs_ref_idx"],
        )
    return validate_records(records, ref_count=len(ref_names), cell_count=cell_count)
