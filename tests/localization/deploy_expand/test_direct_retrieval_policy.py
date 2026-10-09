from types import SimpleNamespace

import numpy as np
import pytest

import live_provider
from live_provider import LiveMapEDMProvider, RelocFix
from river_map_quality import official_edm_adapter


@pytest.fixture
def provider(monkeypatch):
    provider = object.__new__(LiveMapEDMProvider)
    provider._workmap_descriptor_count = 0
    provider._index = SimpleNamespace(index_id="test")
    provider._subsets = {
        "test": SimpleNamespace(excluded_sessions=frozenset(), indices=tuple(range(6)))
    }
    provider._reference_names = tuple(str(i) for i in range(6))
    provider._reference_descriptors = np.eye(6, dtype=np.float32)
    provider._reference_sessions = ("map",) * 6
    provider._reference_occupied_bins = (1,) * 6
    provider.min_reference_occupied_bins = 1
    provider.top_k = 2
    provider._query_matches = {}
    provider._query_lifts = {}
    provider._reference_norms = None
    provider._last_strong_refs = ()
    provider._recovery_attempted = set()
    provider._observations = {str(i): (None, np.zeros(0, dtype=np.int64)) for i in range(6)}
    provider.profile = SimpleNamespace(
        optimizations=SimpleNamespace(adaptive_retrieval=True),
        reloc=SimpleNamespace(period_s=100.0, frozen_pnp_thresholds={"strong_inliers": 80}),
    )
    provider._vpr = lambda: None
    provider._matcher_runtime = lambda: None
    monkeypatch.setattr(
        live_provider, "boq_descriptor_from_array", lambda *_: np.arange(6, 0, -1, dtype=np.float32)
    )
    monkeypatch.setattr(
        official_edm_adapter, "prepare_official_megadepth_image_from_array", lambda *_: object()
    )
    return provider


def run(provider, statuses):
    attempts = []

    def solve(image, prepared, ranked, started, fallback):
        attempts.append(ranked)
        fix = RelocFix.abstained("test", 0.0)
        if statuses[len(attempts) - 1]:
            from dataclasses import replace

            fix = replace(
                fix,
                ok=True,
                status="LOCALIZED_STRONG",
                reference_names=ranked,
                point3d_ids=np.arange(1, 81),
            )
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


def test_failed_return_views_rotate_candidates_and_reset_after_recovery(provider):
    attempts = []

    def solve(image, prepared, ranked, started, fallback):
        attempts.append(ranked)
        fix = RelocFix.abstained("no return-view support", 0.0)
        if 4 in ranked:
            from dataclasses import replace

            return replace(fix, ok=True, status="LOCALIZED_STRONG", point3d_ids=np.arange(1, 81))
        return fix

    provider._solve_references = solve
    image = np.zeros((8, 8), dtype=np.uint8)
    assert not provider._localize_array(image, None).ok
    assert provider._localize_array(image, None).ok
    assert attempts == [(0, 1), (0, 1, 2, 3), (0, 1), (0, 1, 4, 5)]
    # A successful anchor must not blacklist references on the next turn.
    assert provider._recovery_references(tuple(range(6))) == (2, 3)


def test_exhausted_recovery_cycle_retries_new_exposures(provider):
    image = np.zeros((8, 8), dtype=np.uint8)
    attempts = []

    def solve(image, prepared, ranked, started, fallback):
        attempts.append(ranked)
        return RelocFix.abstained("view not yet usable", 0.0)

    provider._solve_references = solve
    for _ in range(3):
        assert not provider._localize_array(image, None).ok
    assert attempts[1::2] == [(0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 2, 3)]


def test_return_reference_beyond_original_shortlist_recovers_geometric_pose(provider, monkeypatch):
    """A valid return-view hypothesis must be reachable beyond rank six."""
    pycolmap = pytest.importorskip("pycolmap")
    from dataclasses import replace
    from direct_profile import DirectOptimizations
    from point_quality import pose_quality
    from sfm_diagnosis.site_pipeline.deployment_localizer import localization_is_strong

    provider.profile.optimizations = DirectOptimizations()
    provider._reference_names = tuple(str(i) for i in range(8))
    provider._reference_descriptors = np.eye(8, dtype=np.float32)
    provider._reference_sessions = ("map",) * 8
    provider._reference_occupied_bins = (1,) * 8
    provider._subsets["test"].indices = tuple(range(8))
    monkeypatch.setattr(
        live_provider, "boq_descriptor_from_array", lambda *_: np.arange(8, 0, -1, dtype=np.float32)
    )
    rng = np.random.default_rng(73)
    xyz = rng.uniform([-3.0, -2.0, 7.0], [3.0, 2.0, 12.0], size=(160, 3))
    K = np.array([[90.0, 0.0, 64.0], [0.0, 90.0, 48.0], [0.0, 0.0, 1.0]])
    projected = xyz @ K.T
    xy = projected[:, :2] / projected[:, 2:]
    camera = pycolmap.Camera(model="PINHOLE", width=128, height=96, params=[90, 90, 64, 48])
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.random_seed = 0
    thresholds = {
        "strong_inliers": 80,
        "minimum_inlier_ratio": 0.25,
        "minimum_hull_coverage": 0.15,
        "minimum_occupancy_4x4": 6,
        "minimum_positive_depth_ratio": 0.99,
        "maximum_reprojection_p90_px": 3.0,
    }
    attempts = []

    def solve(image, prepared, ranked, started, fallback):
        attempts.append(ranked)
        fix = RelocFix.abstained("outbound references have no return-view correspondences", 0.0)
        if 6 not in ranked:
            return fix
        answer = pycolmap.estimate_and_refine_absolute_pose(xy, xyz, camera, options)
        assert answer is not None
        mask = np.asarray(answer["inlier_mask"], dtype=bool)
        pose = np.asarray(answer["cam_from_world"].matrix())
        quality = pose_quality(xy[mask], xyz[mask], pose, K, 128, 96)
        strong = localization_is_strong(
            {
                "inlier_count": int(mask.sum()),
                "inlier_ratio": float(mask.mean()),
                "convex_hull_coverage": quality["inlier_hull_coverage"],
                "occupancy_4x4": quality["inlier_grid_cells"],
                "positive_depth_ratio": quality["positive_depth_ratio"],
                "reprojection_p90": quality["reproj_p90"],
            },
            decision_status="ACCEPT",
            thresholds=thresholds,
        )
        return replace(
            fix,
            ok=strong,
            status="LOCALIZED_STRONG" if strong else "POSE_ESTIMATED_WEAK",
            cam_from_world=pose,
            query_xy=xy[mask],
            point_xyz=xyz[mask],
            point3d_ids=np.arange(1, len(xyz) + 1)[mask],
        )

    provider._solve_references = solve
    image = np.zeros((96, 128), dtype=np.uint8)
    assert not provider._localize_array(image, None).ok
    assert not provider._localize_array(image, None).ok
    recovered = provider._localize_array(image, None)
    assert recovered.ok
    assert attempts[1::2] == [(0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 6, 7)]
    np.testing.assert_allclose(
        recovered.cam_from_world, np.column_stack((np.eye(3), np.zeros(3))), atol=1e-5
    )


@pytest.mark.parametrize("recovered", [False, True])
def test_seedless_strong_keeps_searching_without_losing_original_decision(provider, recovered):
    from dataclasses import replace

    baseline = replace(RelocFix.abstained("depth-only", 0.0), ok=True, status="LOCALIZED_STRONG")
    calls = []

    def solve(image, prepared, ranked, started, fallback):
        calls.append(ranked)
        if len(ranked) > provider.top_k and recovered:
            return replace(baseline, point3d_ids=np.arange(1, 81))
        return baseline

    provider._solve_references = solve
    fix = provider._localize_array(np.zeros((8, 8), dtype=np.uint8), None)
    assert fix.ok
    assert calls == [(0, 1), (0, 1, 2, 3)]
    assert len(fix.point3d_ids) == (80 if recovered else 0)
    assert provider._recovery_attempted == (set() if recovered else {2, 3})


def test_seedless_strong_does_not_restart_candidate_rotation(provider):
    from dataclasses import replace

    attempts = []

    def solve(image, prepared, ranked, started, fallback):
        attempts.append(ranked)
        return replace(RelocFix.abstained("depth-only", 0.0), ok=True, status="LOCALIZED_STRONG")

    provider._solve_references = solve
    for _ in range(2):
        assert provider._localize_array(np.zeros((8, 8), dtype=np.uint8), None).ok
    assert attempts == [(0, 1), (0, 1, 2, 3), (0, 1), (0, 1, 4, 5)]


def test_new_capture_discards_lift_evidence(provider):
    provider._query_lifts["previous_frame"] = (["must not survive"], 1)
    run(provider, [True])
    assert provider._query_lifts == {}


def test_close_releases_lift_and_match_evidence(provider, monkeypatch):
    provider._query_lifts["last_frame"] = (["must not survive shutdown"], 1)
    provider._query_matches[1] = object()
    provider._prepared_reference_cache = SimpleNamespace(clear=lambda: None)
    provider._workmap_references = {}
    provider._workmap_order = []
    provider._workmap_landmark_refs = {}
    monkeypatch.setattr(provider, "_empty_cuda_cache", lambda: None)
    provider.close()
    assert provider._query_lifts == {}
    assert provider._query_matches == {}
    assert not provider._models_ready
