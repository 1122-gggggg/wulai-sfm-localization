"""The reference backbone cache must not change what a match depends on.

edm_matcher keeps the image0 side's ResNet18 output across frames, because map
references are immutable and BatchNorm runs on stored statistics in eval(). Two
properties make that safe to ship, and both are cheap to break by accident:

  - purity: a frame's matches must not depend on which references happened to be
    cached, so every image is extracted in a batch of one.
  - layout: the levels handed to the neck must stay channels_last. Concatenating
    the reference and query features silently produces NCHW, which is slower than
    the cache is fast.

Needs CUDA and the pinned EDM checkpoint; skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from edm_matcher import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_HOST_REF_FEATURE_CACHE,
    DEFAULT_REF_FEATURE_CACHE,
    EDM_H,
    EDM_W,
    EDMMatcher,
)
import edm_matcher as edm_module  # noqa: E402

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Path(DEFAULT_CKPT).is_file(),
    reason="the EDM reference feature cache needs CUDA and the pinned checkpoint",
)


def test_validated_production_feature_cache_defaults() -> None:
    assert DEFAULT_REF_FEATURE_CACHE == 192
    assert DEFAULT_HOST_REF_FEATURE_CACHE == 0


def test_match_output_pack_preserves_values_and_batch_ids() -> None:
    batch = {
        "m_bids": torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        "mkpts0_f": torch.tensor([[1.25, 2.5], [3.75, 4.0]], device="cuda"),
        "mkpts1_f": torch.tensor([[5.0, 6.5], [7.25, 8.0]], device="cuda"),
        "mconf": torch.tensor([0.9, 0.7], device="cuda"),
    }

    bids, k0, k1, confidence = EDMMatcher._match_outputs_to_numpy(batch)

    np.testing.assert_array_equal(bids, [0, 1])
    np.testing.assert_array_equal(k0, batch["mkpts0_f"].cpu().numpy())
    np.testing.assert_array_equal(k1, batch["mkpts1_f"].cpu().numpy())
    np.testing.assert_array_equal(confidence, batch["mconf"].cpu().numpy())


def test_packed_match_outputs_split_into_exact_batch_views() -> None:
    bids = np.array([0, 0, 2], dtype=np.int64)
    k0 = np.arange(6, dtype=np.float32).reshape(3, 2)
    k1 = k0 + 10
    confidence = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    rows = EDMMatcher._split_match_outputs(bids, k0, k1, confidence, 3)

    np.testing.assert_array_equal(rows[0]["mkpts0"], k0[:2])
    assert rows[1]["mkpts0"].shape == (0, 2)
    np.testing.assert_array_equal(rows[2]["mkpts1"], k1[2:])
    assert np.shares_memory(rows[0]["mconf"], confidence)


def _textured_pair() -> tuple[np.ndarray, np.ndarray]:
    """A reference and a shifted copy, so real correspondences exist."""
    import cv2

    rng = np.random.default_rng(20260803)
    base = rng.integers(0, 255, (EDM_H, EDM_W), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.7)
    query = np.roll(np.roll(base, 8, axis=1), 4, axis=0)
    return base, query


def _second_pair() -> tuple[np.ndarray, np.ndarray]:
    import cv2

    rng = np.random.default_rng(20260804)
    base = rng.integers(0, 255, (EDM_H, EDM_W), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.1)
    return base, np.roll(base, 6, axis=1)


def _anchors(result: dict) -> dict:
    """{reference cell id: query point}, direction-01 only.

    This is what reloc_localizer_edm keeps: the reference side identifies the 3D
    anchor, the query side becomes a PnP observation.
    """
    k0, k1 = result["mkpts0"], result["mkpts1"]
    if len(k0) == 0:
        return {}
    keep = ~EDMMatcher.is_refined(k0)
    return dict(zip(EDMMatcher.cell_ids(k0[keep]).tolist(), k1[keep]))


def _agreement(left: dict, right: dict) -> tuple[float, float]:
    shared = set(left) & set(right)
    if not shared:
        return 0.0, float("inf")
    deltas = np.array([np.linalg.norm(left[cell] - right[cell]) for cell in shared])
    return len(shared) / max(len(left), 1), float(deltas.mean())


@pytest.fixture(scope="module")
def matcher() -> EDMMatcher:
    return EDMMatcher()


@pytest.fixture(scope="module")
def uncached_matcher() -> EDMMatcher:
    return EDMMatcher(reference_feature_cache_size=0)


@pytest.fixture(scope="module")
def repeated_matcher() -> EDMMatcher:
    """A matcher whose query tiling is the allocating repeat, for exact A/B."""
    old = os.environ.get("SFM_EDM_QUERY_BATCH_EXPAND")
    os.environ["SFM_EDM_QUERY_BATCH_EXPAND"] = "0"
    try:
        return EDMMatcher(runtime_sigma_mode="reference_grid")
    finally:
        if old is None:
            os.environ.pop("SFM_EDM_QUERY_BATCH_EXPAND", None)
        else:
            os.environ["SFM_EDM_QUERY_BATCH_EXPAND"] = old


def test_cache_is_installed_and_starts_empty(matcher: EDMMatcher) -> None:
    stats = matcher.reference_feature_cache_stats()
    assert stats["capacity"] > 0
    assert matcher.model.backbone.forward.__name__ == "cached_forward"


def test_disabled_cache_leaves_the_backbone_alone(uncached_matcher: EDMMatcher) -> None:
    assert uncached_matcher.reference_feature_cache_stats()["capacity"] == 0
    assert uncached_matcher.model.backbone.forward.__name__ != "cached_forward"
    ref, query = _textured_pair()
    before = dict(uncached_matcher.reference_feature_cache_stats())
    uncached_matcher.match_many_to_one([ref], query)
    assert uncached_matcher.reference_feature_cache_stats() == before


@pytest.mark.parametrize("batch", [1, 2])
def test_a_cache_hit_matches_a_cache_miss_exactly(batch: int) -> None:
    """Purity: the output may not depend on what the cache happened to hold."""
    matcher = EDMMatcher(runtime_sigma_mode="reference_grid")
    ref, query = _textured_pair()
    matcher._reference_feature_cache.clear()
    cold = matcher.match_many_to_one([ref] * batch, query)
    warm = matcher.match_many_to_one([ref] * batch, query)
    assert len(cold) == len(warm) == batch
    for slot, (a, b) in enumerate(zip(cold, warm)):
        for key in ("mkpts0", "mkpts1", "mconf"):
            assert np.array_equal(a[key], b[key]), f"{key} differs in slot {slot}"


def test_cached_anchors_track_the_uncached_backbone(
    matcher: EDMMatcher, uncached_matcher: EDMMatcher
) -> None:
    """The query is extracted alone rather than beside the reference, so a small
    fp16 batch-shape difference is expected. Anchor identity must survive it."""
    ref, query = _textured_pair()
    matcher._reference_feature_cache.clear()
    matcher.match_many_to_one([ref], query)  # populate
    cached = _anchors(matcher.match_many_to_one([ref], query)[0])
    plain = _anchors(uncached_matcher.match_many_to_one([ref], query)[0])

    assert len(plain) > 200, "degenerate pair: too few anchors to compare"
    shared, mean_delta = _agreement(plain, cached)
    assert shared > 0.95, f"only {100 * shared:.1f}% of anchor cells survived"
    assert mean_delta < 0.05, f"query points moved {mean_delta:.4f}px on average"


def test_hit_miss_and_eviction_accounting() -> None:
    small = EDMMatcher(reference_feature_cache_size=1)
    first_ref, query = _textured_pair()
    second_ref, _ = _second_pair()

    small.match_many_to_one([first_ref], query)
    assert small.reference_feature_cache_stats() == {
        "hits": 0,
        "misses": 1,
        "evictions": 0,
        "size": 1,
        "capacity": 1,
    }
    small.match_many_to_one([first_ref], query)
    assert small.reference_feature_cache_stats()["hits"] == 1

    small.match_many_to_one([second_ref], query)  # evicts the first
    stats = small.reference_feature_cache_stats()
    assert stats["misses"] == 2 and stats["evictions"] == 1 and stats["size"] == 1

    small.match_many_to_one([first_ref], query)  # gone again
    assert small.reference_feature_cache_stats()["misses"] == 3


def test_assembled_levels_stay_channels_last(matcher: EDMMatcher) -> None:
    ref, query = _textured_pair()
    matcher._reference_feature_cache.clear()
    matcher.match_many_to_one([ref], query)  # populate
    seen: list[bool] = []
    original = matcher.model.neck.forward

    def spy(ms_feats, mask_c0=None, mask_c1=None):
        seen.extend(level.is_contiguous(memory_format=torch.channels_last) for level in ms_feats)
        return original(ms_feats, mask_c0, mask_c1)

    matcher.model.neck.forward = spy
    try:
        matcher.match_many_to_one([ref], query)
    finally:
        matcher.model.neck.forward = original

    assert seen, "the neck was never called"
    assert all(seen), "a cached level reached the neck as NCHW"


def test_prepared_query_features_are_reused_without_changing_matches(
    matcher: EDMMatcher,
) -> None:
    ref, query = _textured_pair()
    matcher.match_many_to_one([ref], query)  # warm the reference side
    original_extract = matcher._extract_one
    extracted = []

    def counted_extract(original, image):
        extracted.append(tuple(image.shape))
        return original_extract(original, image)

    matcher._extract_one = counted_extract
    try:
        prepared = matcher.prepare_query(query)
        first = matcher.match_many_to_one([ref], query, prepared_query=prepared)[0]
        second = matcher.match_many_to_one([ref], query, prepared_query=prepared)[0]
    finally:
        matcher._extract_one = original_extract

    assert extracted == [(1, 1, EDM_H, EDM_W)]
    for key in ("mkpts0", "mkpts1", "mconf"):
        assert np.array_equal(first[key], second[key])


@pytest.mark.parametrize("batch", [1, 2, 3, 5])
def test_query_batch_broadcast_is_a_zero_copy_view(batch: int) -> None:
    query = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)

    batched = EDMMatcher._broadcast_query(query, batch)

    assert batched.shape == (batch, 1, 3, 4)
    assert batched.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
    assert torch.equal(batched, query.repeat(batch, 1, 1, 1))


@pytest.mark.parametrize("batch", [1, 2])
def test_query_batch_expand_matches_the_allocating_repeat_exactly(
    repeated_matcher: EDMMatcher, batch: int
) -> None:
    """The broadcast shortcut is only legal if it is the allocating repeat, exactly."""
    matcher = EDMMatcher(runtime_sigma_mode="reference_grid")
    ref, query = _textured_pair()
    imgs0 = [ref] if batch == 1 else [ref, _second_pair()[0]]

    assert matcher.query_batch_expand, "the expand matcher must broadcast the query"
    assert not repeated_matcher.query_batch_expand, "the A/B matcher must allocate a repeat"
    prepared = matcher.prepare_query(query)
    expanded = matcher.match_many_to_one(imgs0, query, prepared_query=prepared)
    prepared_repeat = repeated_matcher.prepare_query(query)
    repeated = repeated_matcher.match_many_to_one(imgs0, query, prepared_query=prepared_repeat)
    for expanded_row, repeated_row in zip(expanded, repeated):
        for key in ("mkpts0", "mkpts1", "mconf"):
            assert np.array_equal(expanded_row[key], repeated_row[key]), (
                f"{key} differs between expand and repeat at batch {batch}"
            )


def test_pinned_host_l2_restores_an_evicted_reference_exactly() -> None:
    matcher = EDMMatcher(
        reference_feature_cache_size=1,
        host_reference_feature_cache_size=1,
    )
    first_ref, query = _textured_pair()
    second_ref, _ = _second_pair()

    first = matcher.match_many_to_one([first_ref], query)[0]
    matcher.match_many_to_one([second_ref], query)
    host_stats = matcher.host_reference_feature_cache_stats()
    assert host_stats["size"] == 1
    assert host_stats["stores"] == 1
    host_entry = next(iter(matcher._host_reference_feature_cache.values()))
    assert all(level.is_pinned() for level in host_entry[1])

    restored = matcher.match_many_to_one([first_ref], query)[0]

    assert matcher.host_reference_feature_cache_stats()["hits"] == 1
    for key in ("mkpts0", "mkpts1", "mconf"):
        assert np.array_equal(first[key], restored[key])


def test_inclusive_host_cache_does_not_copy_an_immutable_hit_back_again() -> None:
    matcher = EDMMatcher(
        reference_feature_cache_size=1,
        host_reference_feature_cache_size=2,
        inclusive_host_feature_cache=True,
    )
    first_ref, query = _textured_pair()
    second_ref, _ = _second_pair()

    matcher.match_many_to_one([first_ref], query)
    matcher.match_many_to_one([second_ref], query)
    matcher.match_many_to_one([first_ref], query)
    before = matcher.host_reference_feature_cache_stats()["stores"]

    matcher.match_many_to_one([second_ref], query)

    stats = matcher.host_reference_feature_cache_stats()
    assert stats["stores"] == before
    assert stats["hits"] >= 2


def test_legacy_stream_reference_feature_store_loads(
    matcher: EDMMatcher,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFM_EDM_REFERENCE_FEATURE_STORE_DIR", str(tmp_path))
    reference, _query = _textured_pair()
    path = matcher.bind_reference_feature_store(
        {"reference": reference},
        bundle_sha256="b" * 64,
    )
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_key": matcher.reference_feature_model_key,
            "source_names": ["reference"],
            "source_sha256": [matcher._reference_source_sha256(reference)],
            "levels": [torch.zeros((1, 1, 1, 1), dtype=torch.float16)],
        },
        path,
        _use_new_zipfile_serialization=False,
    )

    matcher._load_reference_feature_store(path)

    assert matcher._persistent_feature_levels is not None
    assert matcher._persistent_feature_levels[0].shape == (1, 1, 1, 1)


def test_persistent_reference_features_restore_exact_matches(
    matcher: EDMMatcher,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFM_EDM_REFERENCE_FEATURE_STORE_DIR", str(tmp_path))
    reference, query = _textured_pair()
    sources = {"reference": reference}
    baseline = matcher.match_many_to_one([reference], query)[0]

    path = matcher.bind_reference_feature_store(
        sources,
        bundle_sha256="a" * 64,
        build_if_missing=True,
    )
    assert path is not None and path.is_file()
    matcher._reference_feature_cache.clear()
    before = matcher.persistent_reference_feature_cache_stats()["hits"]

    restored = matcher.match_many_to_one([reference], query)[0]

    assert matcher.persistent_reference_feature_cache_stats()["hits"] == before + 1
    for key in ("mkpts0", "mkpts1", "mconf"):
        assert np.array_equal(baseline[key], restored[key])


def test_large_reference_feature_store_is_sharded_mmap_and_exact(
    matcher: EDMMatcher,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFM_EDM_REFERENCE_FEATURE_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(edm_module, "REFERENCE_FEATURE_STORE_ZIP_MAX_BYTES", 1)
    monkeypatch.setattr(edm_module, "REFERENCE_FEATURE_SHARD_ENTRIES", 1)
    reference, query = _textured_pair()
    second_reference, _ = _second_pair()
    baseline = matcher.match_many_to_one([reference], query)[0]

    stored = matcher.bind_reference_feature_store(
        {"reference": reference, "second": second_reference},
        bundle_sha256="c" * 64,
        build_if_missing=True,
    )

    assert stored is not None and stored.is_dir()
    shards = sorted(stored.glob("shard_*.pt"))
    assert len(shards) == 2
    assert max(path.stat().st_size for path in shards) < 12 * 1024 * 1024
    assert matcher._persistent_feature_levels is None
    assert matcher._persistent_feature_shards is not None
    matcher._reference_feature_cache.clear()
    restored = matcher.match_many_to_one([reference], query)[0]
    for key in ("mkpts0", "mkpts1", "mconf"):
        assert np.array_equal(baseline[key], restored[key])


def test_accepted_query_features_become_an_exact_temporal_cache_hit() -> None:
    matcher = EDMMatcher(
        temporal_feature_cache_size=2,
        temporal_feature_promotion=True,
    )
    temporal, query = _textured_pair()
    prepared_temporal = matcher.prepare_query(temporal)
    source = temporal.copy()
    matcher.promote_prepared_query(source, prepared_temporal)
    prepared_query = matcher.prepare_query(query)
    before = matcher.feature_cache_stats_by_class()["temporal"]["hits"]

    promoted = matcher.match_many_to_one(
        [source],
        query,
        source_kinds=["temporal"],
        prepared_query=prepared_query,
    )[0]
    direct = matcher.match_many_to_one(
        [temporal],
        query,
        source_kinds=["map"],
        prepared_query=prepared_query,
    )[0]

    assert matcher.feature_cache_stats_by_class()["temporal"]["hits"] == before + 1
    for key in ("mkpts0", "mkpts1", "mconf"):
        assert np.array_equal(promoted[key], direct[key])


def test_query_cuda_graph_matches_eager_backbone_for_changing_frames() -> None:
    matcher = EDMMatcher(query_cuda_graph=True)
    first, second = _textured_pair()

    graph_first = tuple(level.clone() for level in matcher.prepare_query(first)[1])
    matcher.query_cuda_graph = False
    eager_first = matcher.prepare_query(first)[1]
    matcher.query_cuda_graph = True
    graph_second = tuple(level.clone() for level in matcher.prepare_query(second)[1])
    matcher.query_cuda_graph = False
    eager_second = matcher.prepare_query(second)[1]

    for graph, eager in zip(graph_first, eager_first):
        assert torch.equal(graph, eager)
    for graph, eager in zip(graph_second, eager_second):
        assert torch.equal(graph, eager)
