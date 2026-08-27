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

from pathlib import Path
import sys

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from edm_matcher import DEFAULT_CKPT, EDM_H, EDM_W, EDMMatcher  # noqa: E402

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Path(DEFAULT_CKPT).is_file(),
    reason="the EDM reference feature cache needs CUDA and the pinned checkpoint",
)


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
def test_a_cache_hit_matches_a_cache_miss_exactly(matcher: EDMMatcher, batch: int) -> None:
    """Purity: the output may not depend on what the cache happened to hold."""
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
    matcher.match_many_to_one([ref], query)                    # populate
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
        "hits": 0, "misses": 1, "evictions": 0, "size": 1, "capacity": 1,
    }
    small.match_many_to_one([first_ref], query)
    assert small.reference_feature_cache_stats()["hits"] == 1

    small.match_many_to_one([second_ref], query)               # evicts the first
    stats = small.reference_feature_cache_stats()
    assert stats["misses"] == 2 and stats["evictions"] == 1 and stats["size"] == 1

    small.match_many_to_one([first_ref], query)                # gone again
    assert small.reference_feature_cache_stats()["misses"] == 3


def test_assembled_levels_stay_channels_last(matcher: EDMMatcher) -> None:
    ref, query = _textured_pair()
    matcher._reference_feature_cache.clear()
    matcher.match_many_to_one([ref], query)                    # populate
    seen: list[bool] = []
    original = matcher.model.neck.forward

    def spy(ms_feats, mask_c0=None, mask_c1=None):
        seen.extend(
            level.is_contiguous(memory_format=torch.channels_last) for level in ms_feats
        )
        return original(ms_feats, mask_c0, mask_c1)

    matcher.model.neck.forward = spy
    try:
        matcher.match_many_to_one([ref], query)
    finally:
        matcher.model.neck.forward = original

    assert seen, "the neck was never called"
    assert all(seen), "a cached level reached the neck as NCHW"
