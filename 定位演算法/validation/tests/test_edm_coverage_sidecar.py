from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


VALIDATION_DIR = Path(__file__).resolve().parents[1]
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from edm_coverage_sidecar import (
    CoverageRecords,
    extract_coverage_records,
    load_coverage_sidecar,
    merge_records,
    write_coverage_sidecar,
)



class _Point2D:
    def __init__(self, point3d_id: int | None):
        self.point3D_id = point3d_id

    def has_point3D(self) -> bool:
        return self.point3D_id is not None


def _reconstruction(track_image_ids: list[int]):
    images = {
        1: SimpleNamespace(
            image_id=1,
            name="a.jpg",
            points2D=[_Point2D(10), _Point2D(None)],
        ),
        2: SimpleNamespace(
            image_id=2,
            name="b.jpg",
            points2D=[_Point2D(None), _Point2D(20)],
        ),
    }
    points = {
        10: SimpleNamespace(
            track=SimpleNamespace(
                elements=[SimpleNamespace(image_id=value) for value in track_image_ids]
            )
        ),
        20: SimpleNamespace(
            track=SimpleNamespace(
                elements=[SimpleNamespace(image_id=1), SimpleNamespace(image_id=2)]
            )
        ),
    }
    return SimpleNamespace(images=images, points3D=points)


def test_extracts_canonical_exact_observation_tracks() -> None:
    tables = {
        "a.jpg": {"idx_of_cell": np.array([0, -1, -1], dtype=np.int32)},
        "b.jpg": {"idx_of_cell": np.array([-1, 1, -1], dtype=np.int32)},
    }

    records = extract_coverage_records(
        _reconstruction([2, 1, 2]),
        tables,
        ["b.jpg", "a.jpg"],
        ["a.jpg", "b.jpg"],
    )

    assert records.anchor_ref_idx.tolist() == [0, 1]
    assert records.anchor_cell_idx.tolist() == [0, 1]
    assert records.obs_offsets.tolist() == [0, 2, 4]
    assert records.obs_ref_idx.tolist() == [0, 1, 0, 1]


def test_extract_rejects_track_that_omits_anchor() -> None:
    tables = {
        "a.jpg": {"idx_of_cell": np.array([0], dtype=np.int32)},
        "b.jpg": {"idx_of_cell": np.array([-1], dtype=np.int32)},
    }

    with pytest.raises(ValueError, match="omits its anchor"):
        extract_coverage_records(
            _reconstruction([2]), tables, ["a.jpg"], ["a.jpg", "b.jpg"]
        )


def test_sidecar_round_trip_binds_bundle_and_ordered_names(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.pt"
    bundle.write_bytes(b"production-bundle")
    sidecar = tmp_path / "coverage.npz"
    records = CoverageRecords(
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int32),
    )

    write_coverage_sidecar(
        sidecar,
        records,
        bundle_path=bundle,
        ref_names=["a.jpg", "b.jpg"],
        cell_count=3,
    )
    loaded = load_coverage_sidecar(
        sidecar,
        bundle_path=bundle,
        ref_names=["a.jpg", "b.jpg"],
        cell_count=3,
    )

    assert loaded.obs_ref_idx.tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="ordered reference names"):
        load_coverage_sidecar(
            sidecar,
            bundle_path=bundle,
            ref_names=["b.jpg", "a.jpg"],
            cell_count=3,
        )
    bundle.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bundle SHA-256"):
        load_coverage_sidecar(
            sidecar,
            bundle_path=bundle,
            ref_names=["a.jpg", "b.jpg"],
            cell_count=3,
        )


def test_merge_rejects_duplicate_anchor() -> None:
    chunk = CoverageRecords(
        np.array([0], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([0, 1], dtype=np.int64),
        np.array([0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="duplicate coverage anchor"):
        merge_records([chunk, chunk], ref_count=1, cell_count=3)


def test_merge_reorders_ragged_observations_with_their_anchor() -> None:
    later = CoverageRecords(
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0, 1], dtype=np.int64),
        np.array([1], dtype=np.int32),
    )
    earlier = CoverageRecords(
        np.array([0], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([0, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int32),
    )

    merged = merge_records([later, earlier], ref_count=2, cell_count=3)

    assert merged.anchor_ref_idx.tolist() == [0, 1]
    assert merged.anchor_cell_idx.tolist() == [2, 0]
    assert merged.obs_offsets.tolist() == [0, 2, 3]
    assert merged.obs_ref_idx.tolist() == [0, 1, 1]


