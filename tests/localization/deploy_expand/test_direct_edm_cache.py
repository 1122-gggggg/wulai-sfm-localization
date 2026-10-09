from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edm_inference import CachedEDM
from river_map_quality.official_edm_adapter import match_official_prepared


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return tuple(images * value for value in (1, 2, 3, 4))


class Matcher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.calls = 0
        self.fail = False

    def forward(self, batch):
        self.calls += 1
        features = getattr(self, "_cached_pyramid", None)
        if features is None:
            features = self.backbone(torch.cat([batch["image0"], batch["image1"]]))
        if self.fail:
            raise RuntimeError("matcher failure")
        pair = features[0][0] + features[0][1] * 2
        batch["mkpts0_f"] = pair.flatten().reshape(-1, 2)
        batch["mkpts1_f"] = batch["mkpts0_f"] + 1
        batch["mconf"] = features[1].flatten()


def prepared(value):
    return SimpleNamespace(
        pixels=np.full((4, 4), value, dtype=np.uint8),
        coarse_mask=np.ones((1, 1), dtype=bool),
        scale=(1.0, 1.0),
    )


@pytest.fixture
def runtime():
    return SimpleNamespace(torch=torch, device="cpu", matcher=Matcher().eval())


def test_cache_matches_the_uncached_oracle_for_changing_queries_and_references(runtime):
    cache = CachedEDM(runtime, capacity=2)
    refs = [prepared(30), prepared(90)]
    for query in (prepared(10), prepared(20)):
        for reference in [*refs, refs[0]]:
            expected = match_official_prepared(runtime, query, reference)
            actual = cache.match(query, reference)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
    assert cache.backbone_hits > 0


def test_cache_reuses_backbone_but_runs_pair_interaction_every_time(runtime):
    cache = CachedEDM(runtime, capacity=2)
    query, reference = prepared(10), prepared(30)
    cache.match(query, reference)
    cache.match(query, reference)
    assert runtime.matcher.backbone.calls == 1
    assert runtime.matcher.calls == 2


def test_reference_cache_is_bounded_and_evicted_reference_is_recomputed(runtime):
    cache = CachedEDM(runtime, capacity=1)
    query, first, second = prepared(10), prepared(30), prepared(60)
    for reference in (first, second, first):
        cache.match(query, reference)
    assert len(cache.references) == 1
    assert runtime.matcher.backbone.calls == 3


def test_exception_restores_model_hooks_and_cached_plan(runtime):
    cache = CachedEDM(runtime, capacity=2)
    query, reference = prepared(10), prepared(30)
    cache.match(query, reference)
    runtime.matcher.fail = True
    with pytest.raises(RuntimeError, match="matcher failure"):
        cache.match(query, reference)
    assert runtime.matcher._backbone_plan is None
    assert runtime.matcher._cached_pyramid is None
    assert not runtime.matcher.backbone._forward_hooks
