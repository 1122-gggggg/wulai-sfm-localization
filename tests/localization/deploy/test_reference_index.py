from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

DEPLOY_ROOT = (
    Path(__file__).resolve().parents[3]
    / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
)
sys.path.insert(0, str(DEPLOY_ROOT))

from reference_index import (
    IndexFormatError,
    IVFReferenceIndex,
    ReferenceMatch,
)


def _descriptors(count: int, dimension: int = 4) -> np.ndarray:
    rng = np.random.default_rng(20260809)
    values = rng.normal(size=(count, dimension)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def _names(count: int) -> list[str]:
    return [f"ref-{index:06d}.jpg" for index in range(count)]


def _build(tmp_path: Path, *, count: int = 32, dimension: int = 4) -> Path:
    path = tmp_path / "index"
    IVFReferenceIndex.build(
        path,
        _descriptors(count, dimension),
        _names(count),
        model_identity="edm:test-model:v1",
        nlist=min(4, count),
        seed=17,
        kmeans_iterations=4,
        batch_size=8,
        max_query_probes=4,
        max_query_candidates=12,
    )
    return path


def test_build_open_and_query_returns_stable_matches(tmp_path: Path) -> None:
    path = _build(tmp_path)
    index = IVFReferenceIndex.open(path, expected_model_identity="edm:test-model:v1")

    query = _descriptors(32)[7]
    results = index.query(query, top_k=5, probes=4, max_candidates=12)

    assert len(results) == 5
    assert all(isinstance(match, ReferenceMatch) for match in results)
    assert results[0].name == "ref-000007.jpg"
    assert results == index.query(query, top_k=5, probes=4, max_candidates=12)
    assert index.last_query_stats.candidate_count <= 12
    assert index.last_query_stats.probes == 4


def test_query_ties_are_broken_by_stable_name(tmp_path: Path) -> None:
    descriptors = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    path = tmp_path / "index"
    IVFReferenceIndex.build(
        path,
        descriptors,
        ["z-ref", "a-ref", "m-ref"],
        model_identity="model",
        nlist=1,
        seed=1,
        kmeans_iterations=2,
        max_query_probes=1,
        max_query_candidates=3,
    )
    index = IVFReferenceIndex.open(path)

    assert [match.name for match in index.query([1.0, 0.0], top_k=3)] == [
        "a-ref",
        "z-ref",
        "m-ref",
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("digest", "digest"),
        ("metadata", "model identity"),
        ("descriptor", "digest"),
    ],
)
def test_open_fails_closed_on_corrupt_digest_or_identity(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    path = _build(tmp_path)
    if mutation == "digest":
        manifest = path / "SHA256SUMS.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["files"]["names.json"] = "0" * 64
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "metadata":
        metadata = path / "metadata.json"
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["model_identity"] = "different-model"
        metadata.write_text(json.dumps(payload), encoding="utf-8")
    else:
        descriptor_path = path / "descriptors.npy"
        descriptor_path.write_bytes(descriptor_path.read_bytes() + b"corrupt")

    with pytest.raises(IndexFormatError, match=expected):
        IVFReferenceIndex.open(path, expected_model_identity="edm:test-model:v1")


def test_open_rejects_wrong_dimension(tmp_path: Path) -> None:
    path = _build(tmp_path, dimension=4)
    with pytest.raises(IndexFormatError, match="dimension"):
        IVFReferenceIndex.open(path, expected_dimension=3)


def test_build_rejects_nan_and_non_normalized_descriptors(tmp_path: Path) -> None:
    values = _descriptors(4)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        IVFReferenceIndex.build(tmp_path / "nan", values, _names(4), model_identity="model")

    values = _descriptors(4)
    values[0] *= 2.0
    with pytest.raises(ValueError, match="normalized"):
        IVFReferenceIndex.build(
            tmp_path / "not-normalized", values, _names(4), model_identity="model"
        )


def test_query_recall_sanity_with_bounded_candidates(tmp_path: Path) -> None:
    descriptors = _descriptors(256, dimension=8)
    path = tmp_path / "index"
    IVFReferenceIndex.build(
        path,
        descriptors,
        _names(len(descriptors)),
        model_identity="model",
        nlist=16,
        seed=7,
        kmeans_iterations=5,
        batch_size=32,
        max_query_probes=4,
        max_query_candidates=24,
    )
    index = IVFReferenceIndex.open(path)
    query = descriptors[123] + np.float32(0.005)
    result = index.query(query, top_k=1, probes=4, max_candidates=24)

    assert result[0].name == "ref-000123.jpg"
    assert index.last_query_stats.candidate_count <= 24
    assert index.last_query_stats.candidate_count < len(descriptors)


def test_100k_metadata_and_projection_smoke_is_bounded(tmp_path: Path) -> None:
    count = 100_000
    dimension = 4
    descriptors = _descriptors(count, dimension)
    path = tmp_path / "index"
    IVFReferenceIndex.build(
        path,
        descriptors,
        _names(count),
        model_identity="model:100k",
        nlist=64,
        seed=3,
        kmeans_iterations=2,
        batch_size=2048,
        max_query_probes=4,
        max_query_candidates=4096,
    )
    index = IVFReferenceIndex.open(path, expected_model_identity="model:100k")
    result = index.query(descriptors[50_000], top_k=3, probes=4, max_candidates=4096)

    assert result[0].name == "ref-050000.jpg"
    assert index.count == count
    assert index.dimension == dimension
    assert index.last_query_stats.candidate_count <= 4096
    assert index.last_query_stats.exact_rerank_count == index.last_query_stats.candidate_count


def test_manifest_contains_sha256_for_every_required_file(tmp_path: Path) -> None:
    path = _build(tmp_path)
    manifest = json.loads((path / "SHA256SUMS.json").read_text(encoding="utf-8"))
    required = {
        "metadata.json",
        "centroids.npy",
        "postings_offsets.npy",
        "postings_indices.npy",
        "descriptors.npy",
        "names.json",
    }

    assert set(manifest["files"]) == required
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        assert digest == expected


def test_default_seed_index_can_be_reopened(tmp_path: Path) -> None:
    path = tmp_path / "default-seed"
    IVFReferenceIndex.build(
        path,
        _descriptors(8),
        _names(8),
        model_identity="model:default-seed",
    )

    index = IVFReferenceIndex.open(
        path,
        expected_model_identity="model:default-seed",
    )

    assert index.count == 8
    assert index.names == tuple(_names(8))


def test_megaloc_layer_queries_index_and_maps_names_to_bundle_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    bundle_names = ["z-ref.jpg", "a-ref.jpg", "m-ref.jpg"]
    path = tmp_path / "megaloc-index"
    IVFReferenceIndex.build(
        path,
        descriptors,
        bundle_names,
        model_identity="megaloc:test:v1",
        nlist=1,
        max_query_probes=1,
        max_query_candidates=3,
    )
    from production_xfeat_tracker import MegaLocLayer

    layer = MegaLocLayer.load_index(
        path,
        bundle_names,
        expected_model_identity="megaloc:test:v1",
        device="cpu",
    )
    monkeypatch.setattr(layer, "extract_one", lambda _rgb: descriptors[0])

    scored, _elapsed_ms = layer.topk_scored(np.empty((0, 0, 3), np.uint8), 2)

    assert scored[0][0] == 0
    assert scored[0][1] == pytest.approx(1.0)
    assert layer.reference_count == 3
    assert layer.ref_desc is None


def test_edm_retrieval_uses_index_and_honors_exclusions(tmp_path: Path) -> None:
    descriptors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    names = ["z-ref.jpg", "a-ref.jpg", "m-ref.jpg"]
    path = tmp_path / "edm-index"
    IVFReferenceIndex.build(
        path,
        descriptors,
        names,
        model_identity="megaloc:test:v1",
        nlist=1,
        max_query_probes=1,
        max_query_candidates=3,
    )
    index = IVFReferenceIndex.open(path)
    from reloc_localizer_edm import Camera, EDMLocalizer

    localizer = EDMLocalizer(
        SimpleNamespace(
            ref_names=names,
            ref_global=SimpleNamespace(
                __matmul__=lambda _other: (_ for _ in ()).throw(
                    AssertionError("full scan must not run")
                )
            ),
        ),
        Camera("PINHOLE", 1280, 720, [1000.0, 1000.0, 640.0, 360.0]),
        matcher=object(),
        megaloc=SimpleNamespace(extract_one=lambda _frame: descriptors[0]),
        reference_index=index,
    )

    result = localizer.retrieve(
        np.empty((1, 1, 3), np.uint8),
        2,
        exclude={"z-ref.jpg"},
    )

    assert result == ["a-ref.jpg", "m-ref.jpg"]


def test_factory_opens_profile_pinned_index_manifest(tmp_path: Path) -> None:
    descriptors = _descriptors(8, dimension=4)
    path = tmp_path / "profile-index"
    IVFReferenceIndex.build(
        path,
        descriptors,
        _names(8),
        model_identity="megaloc:test:v1",
    )
    from production_localizer_factory import _load_bound_reference_index

    manifest_sha256 = hashlib.sha256(
        (path / "SHA256SUMS.json").read_bytes()
    ).hexdigest()

    index = _load_bound_reference_index(
        path / "SHA256SUMS.json",
        ref_names=_names(8),
        dimension=4,
        expected_sha256=manifest_sha256,
    )

    assert index is not None
    assert index.count == 8
    with pytest.raises(ValueError, match="names"):
        _load_bound_reference_index(
            path / "SHA256SUMS.json",
            ref_names=[*_names(7), "wrong.jpg"],
            dimension=4,
            expected_sha256=manifest_sha256,
        )
