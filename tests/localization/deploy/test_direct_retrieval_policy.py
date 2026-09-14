from types import SimpleNamespace

import numpy as np
import pytest

import live_provider
from live_provider import LiveMapEDMProvider, RelocFix
from river_map_quality import official_edm_adapter


@pytest.fixture
def provider(monkeypatch):
    provider = object.__new__(LiveMapEDMProvider)
    provider._index = SimpleNamespace(index_id="test")
    provider._subsets = {"test": SimpleNamespace(excluded_sessions=frozenset(), indices=tuple(range(6)))}
    provider._reference_names = tuple(str(i) for i in range(6))
    provider._reference_descriptors = np.eye(6, dtype=np.float32)
    provider._reference_sessions = ("map",) * 6
    provider._reference_occupied_bins = (1,) * 6
    provider.min_reference_occupied_bins = 1
    provider.top_k = 2
    provider._query_matches = {}
    provider._reference_norms = None
    provider._last_strong_refs = ()
    provider.profile = SimpleNamespace(
        optimizations=SimpleNamespace(adaptive_retrieval=True),
        reloc=SimpleNamespace(period_s=100.0),
    )
    provider._vpr = lambda: None
    provider._matcher_runtime = lambda: None
    monkeypatch.setattr(live_provider, "boq_descriptor_from_array", lambda *_: np.arange(6, 0, -1, dtype=np.float32))
    monkeypatch.setattr(official_edm_adapter, "prepare_official_megadepth_image_from_array", lambda *_: object())
    return provider


def run(provider, statuses):
    attempts = []

    def solve(image, prepared, ranked, started, fallback):
        attempts.append(ranked)
        fix = RelocFix.abstained("test", 0.0)
        if statuses[len(attempts) - 1]:
            from dataclasses import replace
            fix = replace(fix, ok=True, status="LOCALIZED_STRONG", reference_names=ranked)
        return fix

    provider._solve_references = solve
    fix = provider._localize_array(np.zeros((8, 8), dtype=np.uint8), None)
    return fix, attempts


def test_a_strong_baseline_is_preserved_without_extra_matching(provider):
    fix, attempts = run(provider, [True])
    assert fix.ok
    assert attempts == [(0, 1)]


def test_a_failed_pair_expands_with_the_same_original_references(provider):
    fix, attempts = run(provider, [False, True])
    assert fix.ok
    assert attempts == [(0, 1), (0, 1, 2, 3)]


def test_exhausted_time_budget_does_not_add_gpu_work(provider):
    provider.profile.reloc.period_s = 0.0
    fix, attempts = run(provider, [False])
    assert not fix.ok
    assert attempts == [(0, 1)]


def test_legacy_policy_uses_only_the_frozen_pair(provider):
    provider.profile.optimizations.adaptive_retrieval = False
    fix, attempts = run(provider, [False])
    assert not fix.ok
    assert attempts == [(0, 1)]


def test_covisibility_never_removes_the_best_remaining_global_candidate(provider):
    provider._last_strong_refs = (0,)
    provider._observations = {
        str(i): (None, np.array(ids))
        for i, ids in enumerate([[10, 11], [], [22], [33], [10, 11], [44]])
    }
    assert provider._recovery_references(tuple(range(6))) == (2, 4)


def test_cached_norms_preserve_ranking_including_session_exclusion():
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(30, 64)).astype(np.float32)
    query = rng.normal(size=64).astype(np.float32)
    options = dict(reference_sessions=("a", "b") * 15, excluded_sessions=frozenset({"b"}), top_k=8)
    expected = live_provider.rank_reference_indices(bank, query, **options)
    actual = live_provider.rank_reference_indices(
        bank, query, **options, reference_norms=np.linalg.norm(bank, axis=1)
    )
    assert actual == expected
    assert all(index % 2 == 0 for index in actual[0])
