from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


TRACKER_DIR = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(TRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKER_DIR))

import production_xfeat_tracker as pxt  # noqa: E402


def test_output_heading_uses_supplied_measured_map_frame() -> None:
    calls = []
    frame = SimpleNamespace(
        heading=lambda forward: calls.append(np.asarray(forward, float)) or 0.75
    )

    yaw = pxt._map_heading_from_forward(
        np.array([0.0, 1.0, 0.0]), frame, fallback_yaw=-1.0
    )

    assert yaw == pytest.approx(0.75)
    assert np.allclose(calls[0], [0.0, 1.0, 0.0])


def _candidate_tracker(monkeypatch, *, refs: int = 2, dedup_corr: bool = False):
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig(
        dedup_corr=dedup_corr,
        temporal_cache_enabled=False,
    )
    tracker._frame_counters = {}
    tracker._ref_dev_cache = OrderedDict()
    tracker.temporal_cache = pxt.TemporalAnchorCache()
    tracker.cam = pxt.Camera("PINHOLE", 16, 12, [8.0, 8.0, 8.0, 6.0])

    q_feats = {
        "keypoints": torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
        "descriptors": torch.tensor([[3.0, 4.0], [4.0, -3.0]]),
        "image_size": (16, 12),
    }
    ref_names = [f"r{i}" for i in range(refs)]
    ref_map = {}
    for i, name in enumerate(ref_names):
        ref_map[name] = SimpleNamespace(
            feats={
                "keypoints": torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
                "descriptors": torch.tensor([[5.0, 12.0], [12.0, -5.0]]),
                "image_size": (16, 12),
            },
            xyz=np.array([[float(i), 0.0, 3.0], [float(i), 1.0, 3.0]], np.float32),
        )
    tracker.map = SimpleNamespace(ref_names=ref_names, refs=ref_map)
    tracker._xfeat = object()

    extract_calls = []

    def fake_extract(_xfeat, _frame, _topk):
        extract_calls.append(1)
        return q_feats

    monkeypatch.setattr(pxt, "extract_xfeat", fake_extract)
    monkeypatch.setattr(pxt, "_to_device_feats", lambda feats: dict(feats))
    return tracker, extract_calls


def _composite_tracker():
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig()
    tracker.state = pxt.RuntimeState(mode="TRACK")
    tracker.temporal_cache = pxt.TemporalAnchorCache()
    tracker._last_seed_refs = ()
    tracker._frame_capture_stamp = 1.0
    tracker._frame_counters = {}
    tracker._xfeat = object()
    tracker._track_candidates = lambda weak=False: [0, 1, 2, 3, 4]
    tracker._publish_success = lambda *_args, **_kwargs: None
    return tracker


def test_flow_experiment_gates_are_read_from_environment(monkeypatch):
    values = {
        "SFM_FLOW_REFRESH": "2",
        "SFM_FLOW_FB_PX": "0.35",
        "SFM_FLOW_QUAL_INLIERS": "100",
        "SFM_FLOW_QUAL_RATIO": "0.5",
        "SFM_FLOW_QUAL_REPROJ": "3.5",
        "SFM_FLOW_QUAL_NTRACK": "120",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    reloc_map = SimpleNamespace(
        ref_names=[], ref_centers=np.empty((0, 3)), ref_yaws=None,
    )
    megaloc = SimpleNamespace(
        ref_desc=np.empty((0, 4), np.float32),
        reference_count=0,
    )

    tracker = pxt.ProductionXFeatTracker(
        reloc_map, megaloc, None, None, pxt.ProductionConfig(flow_enabled=True),
    )

    assert tracker._flow_enabled is True
    assert tracker._flow_refresh_every == 2
    assert tracker._flow_fb_px == pytest.approx(0.35)
    assert tracker._flow_qual_inl == 100
    assert tracker._flow_qual_ratio == pytest.approx(0.5)
    assert tracker._flow_qual_reproj == pytest.approx(3.5)
    assert tracker._flow_qual_ntrack == 120


def test_xfeat_only_warmup_does_not_load_megaloc(monkeypatch):
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig()
    tracker._xfeat = None
    megaloc_calls = []
    tracker.meg = SimpleNamespace(model=lambda: megaloc_calls.append(1))
    xfeat = object()
    monkeypatch.setattr(pxt, "load_xfeat", lambda _topk: xfeat)

    assert tracker.ensure_xfeat() is xfeat
    assert megaloc_calls == []
    assert tracker.ensure_models() is xfeat
    assert megaloc_calls == [1]


def test_ensure_models_preloads_lighterglue():
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig()
    matcher_calls = []

    class FakeXFeat:
        def ensure_lighterglue(self):
            matcher_calls.append(1)

    xfeat = FakeXFeat()
    tracker._xfeat = xfeat
    tracker.meg = SimpleNamespace(model=lambda: None)

    assert tracker.ensure_models() is xfeat
    assert matcher_calls == [1]


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_pre_normalized_query_preserves_mutual_nn_results(device):
    q_desc = torch.tensor(
        [[3.0, 4.0], [4.0, -3.0], [-5.0, 12.0]], device=device)
    r_desc = torch.tensor(
        [[5.0, 12.0], [12.0, -5.0], [-8.0, 15.0]], device=device)

    expected_pairs, expected_scores = pxt.mutual_nn_pairs(
        q_desc, r_desc, min_score=-1.0, with_scores=True)
    q_normalized = pxt.F.normalize(q_desc.float(), dim=-1)
    actual_pairs, actual_scores = pxt.mutual_nn_pairs(
        q_normalized, r_desc, min_score=-1.0, with_scores=True,
        q_normalized=True)

    assert torch.equal(actual_pairs, expected_pairs)
    assert torch.equal(actual_scores, expected_scores)


def test_default_nn_reuses_query_normalization_and_does_not_request_scores(monkeypatch):
    tracker, extract_calls = _candidate_tracker(monkeypatch)
    tracker.cfg.temporal_cache_enabled = True
    tracker.cfg.temporal_cache_min_anchors = 1
    tracker.temporal_cache = pxt.TemporalAnchorCache(
        descriptors=torch.tensor([[5.0, 12.0]]),
        xyz=np.array([[0.0, 0.0, 3.0]], np.float32),
        ref_ids=np.array([0], np.int32),
    )
    q_cache = {}
    calls = []
    normalize = pxt.F.normalize
    normalize_calls = []

    def tracked_normalize(tensor, *args, **kwargs):
        normalize_calls.append(tensor)
        return normalize(tensor, *args, **kwargs)

    def fake_mutual_nn(q_desc, _r_desc, _min_score, with_scores=False,
                       q_normalized=False):
        calls.append((q_desc, with_scores, q_normalized))
        return torch.empty((0, 2), dtype=torch.long)

    monkeypatch.setattr(pxt.F, "normalize", tracked_normalize)
    monkeypatch.setattr(pxt, "mutual_nn_pairs", fake_mutual_nn)

    for _ in range(2):
        ret, _info = tracker._localize_with_candidates(
            np.zeros((12, 16, 3), np.uint8), [0, 1], 1700, 1,
            matcher_mode="nn", include_temporal_cache=True, q_cache=q_cache)
        assert ret is None

    assert len(extract_calls) == 1
    assert len(normalize_calls) == 1
    assert calls and all(with_scores is False for _, with_scores, _ in calls)
    assert all(q_normalized is True for _, _, q_normalized in calls)
    assert len({id(q_desc) for q_desc, _, _ in calls}) == 1


def test_dedup_enabled_still_requests_scores(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=1, dedup_corr=True)
    requested = []

    def fake_mutual_nn(_q_desc, _r_desc, _min_score, with_scores=False,
                       q_normalized=False):
        requested.append((with_scores, q_normalized))
        empty_pairs = torch.empty((0, 2), dtype=torch.long)
        empty_scores = torch.empty((0,), dtype=torch.float32)
        return empty_pairs, empty_scores

    monkeypatch.setattr(pxt, "mutual_nn_pairs", fake_mutual_nn)
    ret, _info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0], 1700, 1,
        matcher_mode="nn", q_cache={})

    assert ret is None
    assert requested == [(True, True)]


def test_lighterglue_uses_index_only_api_when_available(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=1)
    calls = []

    class FakeXFeat:
        def match_lighterglue_indices(self, _query, _ref, min_conf):
            calls.append(min_conf)
            return np.empty((0, 2), dtype=np.int64)

        def match_lighterglue(self, *_args, **_kwargs):
            raise AssertionError("legacy keypoint-returning API should not be used")

    tracker._xfeat = FakeXFeat()
    ret, _info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0], 1700, 1,
        matcher_mode="lighterglue", q_cache={})

    assert ret is None
    assert calls == [tracker.cfg.min_conf]


def test_lighterglue_device_indices_preserve_candidate_order(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=2)
    calls = []

    class FakeXFeat:
        def match_lighterglue_indices_tensor(self, _query, _ref, min_conf):
            calls.append(min_conf)
            return [
                torch.tensor([[1, 0]], dtype=torch.long),
                torch.tensor([[0, 1]], dtype=torch.long),
            ][len(calls) - 1]

        def match_lighterglue_indices(self, *_args, **_kwargs):
            raise AssertionError("device-index API should avoid per-ref D2H")

        def match_lighterglue(self, *_args, **_kwargs):
            raise AssertionError("legacy keypoint-returning API should not be used")

    tracker._xfeat = FakeXFeat()
    cached_pairs = {}
    ret, info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0, 1], 1700, 99,
        matcher_mode="lighterglue", q_cache={}, lg_pairs_out=cached_pairs)

    assert ret is None
    assert calls == [tracker.cfg.min_conf, tracker.cfg.min_conf]
    assert info["raw_matches"] == 2
    assert info["used_refs"] == [0, 1]
    assert np.array_equal(cached_pairs[0], np.array([[1, 0]]))
    assert np.array_equal(cached_pairs[1], np.array([[0, 1]]))


def test_lighterglue_scores_drive_per_ref_cap_without_reordering(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=1)
    tracker.cfg.max_corr_per_ref = 1
    tracker.cfg.temporal_cache_enabled = True
    tracker.cfg.temporal_cache_min_anchors = 1

    class FakeXFeat:
        def match_lighterglue_indices_scores_tensor(self, _query, _ref, min_conf):
            assert min_conf == tracker.cfg.min_conf
            return (
                torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
                torch.tensor([0.1, 0.9], dtype=torch.float32),
            )

    tracker._xfeat = FakeXFeat()
    tracker._estimate_reproj_rms = lambda *_args: 1.0

    class FakeOptions:
        def __init__(self):
            self.ransac = SimpleNamespace(max_error=None, random_seed=None)

    fake_ret = {"num_inliers": 1, "inliers": np.ones(1, dtype=bool)}
    fake_pycolmap = SimpleNamespace(
        Camera=lambda **_kwargs: object(),
        AbsolutePoseEstimationOptions=FakeOptions,
        estimate_and_refine_absolute_pose=lambda *_args, **_kwargs: fake_ret,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    ret, info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0], 1700, 1,
        matcher_mode="lighterglue", q_cache={})

    assert ret is fake_ret
    assert info["corr3d_pre_cap"] == 1
    context = info["_temporal_cache_context"]
    assert context[2] == [0]
    assert context[3] == [1]


def test_spatial_cap_keeps_quality_and_metadata_aligned(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=2)
    tracker.cfg.max_corr_total = 2
    tracker.cfg.temporal_cache_enabled = True
    tracker.cfg.temporal_cache_min_anchors = 1
    calls = 0

    def fake_mutual_nn(*_args, with_scores=False, **_kwargs):
        nonlocal calls
        scores = ([0.1, 0.9], [0.8, 0.2])[calls]
        calls += 1
        pairs = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
        assert with_scores is True
        return pairs, torch.tensor(scores, dtype=torch.float32)

    monkeypatch.setattr(pxt, "mutual_nn_pairs", fake_mutual_nn)
    tracker._estimate_reproj_rms = lambda *_args: 1.0

    class FakeOptions:
        def __init__(self):
            self.ransac = SimpleNamespace(max_error=None, random_seed=None)

    fake_ret = {"num_inliers": 2, "inliers": np.ones(2, dtype=bool)}
    fake_pycolmap = SimpleNamespace(
        Camera=lambda **_kwargs: object(),
        AbsolutePoseEstimationOptions=FakeOptions,
        estimate_and_refine_absolute_pose=lambda *_args, **_kwargs: fake_ret,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    ret, info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0, 1], 1700, 1,
        matcher_mode="nn", q_cache={})

    assert ret is fake_ret
    assert info["corr3d_pre_cap"] == 4
    assert info["corr3d"] == 2
    assert info["corr3d_pruned"] == 2
    context = info["_temporal_cache_context"]
    assert context[2] == [0, 1]
    assert context[3] == [1, 0]


def test_spatial_cap_removes_refs_without_surviving_correspondences(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=2)
    tracker.cfg.max_corr_total = 1
    tracker._estimate_reproj_rms = lambda *_args: 1.0
    calls = 0

    def fake_mutual_nn(*_args, with_scores=False, **_kwargs):
        nonlocal calls
        scores = ([0.1, 0.2], [0.9, 0.8])[calls]
        calls += 1
        assert with_scores is True
        return (
            torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
            torch.tensor(scores, dtype=torch.float32),
        )

    monkeypatch.setattr(pxt, "mutual_nn_pairs", fake_mutual_nn)

    class FakeOptions:
        def __init__(self):
            self.ransac = SimpleNamespace(max_error=None, random_seed=None)

    fake_ret = {"num_inliers": 1, "inliers": np.ones(1, dtype=bool)}
    fake_pycolmap = SimpleNamespace(
        Camera=lambda **_kwargs: object(),
        AbsolutePoseEstimationOptions=FakeOptions,
        estimate_and_refine_absolute_pose=lambda *_args, **_kwargs: fake_ret,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    ret, info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0, 1], 1700, 1,
        matcher_mode="nn", q_cache={})

    assert ret is fake_ret
    assert info["corr3d_pre_cap"] == 4
    assert info["corr3d"] == 1
    assert info["used_refs"] == [1]


def test_query_keypoint_d2h_is_lazy_and_cached(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=1)
    q_cache = {}
    pairs = torch.empty((0, 2), dtype=torch.long)
    monkeypatch.setattr(pxt, "mutual_nn_pairs", lambda *_args, **_kwargs: pairs)

    tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0], 1700, 99,
        matcher_mode="nn", q_cache=q_cache)
    query_keypoints = q_cache["qkp"]
    assert torch.is_tensor(query_keypoints)
    expected_keypoints = query_keypoints.numpy().copy()

    query_cpu_calls = 0
    events = []
    tensor_cpu = torch.Tensor.cpu

    def tracked_cpu(tensor, *args, **kwargs):
        nonlocal query_cpu_calls
        if tensor is query_keypoints:
            query_cpu_calls += 1
            events.append("query_d2h")
        return tensor_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", tracked_cpu)
    pairs = torch.tensor([[0, 0]], dtype=torch.long)
    monkeypatch.setattr(
        pxt,
        "mutual_nn_pairs",
        lambda *_args, **_kwargs: events.append("match") or pairs,
    )
    tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0], 1700, 99,
        matcher_mode="nn", q_cache=q_cache)
    cached_numpy = q_cache["qkp"]
    assert isinstance(cached_numpy, np.ndarray)
    assert np.array_equal(cached_numpy, expected_keypoints)
    assert query_cpu_calls == 1
    assert events == ["match", "query_d2h"]

    tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0], 1700, 99,
        matcher_mode="nn", q_cache=q_cache)
    assert q_cache["qkp"] is cached_numpy
    assert query_cpu_calls == 1


def test_candidate_pass_defers_inlier_payload_materialization(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=1)
    tracker.cfg.temporal_cache_enabled = True
    tracker._estimate_reproj_rms = lambda *_args: 1.0
    payload_calls = []
    tracker._cache_payload_from_ref_inliers = lambda *_args: payload_calls.append(1)

    monkeypatch.setattr(
        pxt,
        "mutual_nn_pairs",
        lambda *_args, **_kwargs: torch.tensor([[0, 0]], dtype=torch.long),
    )

    class FakeOptions:
        def __init__(self):
            self.ransac = SimpleNamespace(max_error=None)

    fake_ret = {
        "num_inliers": 1,
        "inliers": np.array([True]),
    }
    fake_pycolmap = SimpleNamespace(
        Camera=lambda **_kwargs: object(),
        AbsolutePoseEstimationOptions=FakeOptions,
        estimate_and_refine_absolute_pose=lambda *_args, **_kwargs: fake_ret,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    ret, info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0], 1700, 1,
        matcher_mode="nn", q_cache={})

    assert ret is fake_ret
    assert payload_calls == []
    assert "_temporal_cache_context" in info
    assert "_temporal_cache_payload" not in info


def test_composite_fallback_keeps_temporal_diagnostics_and_aggregates_timings(monkeypatch):
    tracker = _composite_tracker()
    tracker.cfg.matcher_mode = "nn_then_lg"
    monkeypatch.setattr(pxt, "_LOC_TIMING", False)
    pass_infos = [
        {
            "tag": "fast", "inliers": 60, "corr3d": 80, "reproj_rms": 2.0,
            "feature_ms": 10.0, "match_ms": 20.0, "pnp_ms": 30.0,
            "temporal_cache_attempted": True, "temporal_cache_corr3d": 44,
        },
        {
            "tag": "first", "inliers": 90, "corr3d": 110, "reproj_rms": 2.0,
            "feature_ms": 1.0, "match_ms": 2.0, "pnp_ms": 3.0,
        },
        {
            "tag": "full", "inliers": 120, "corr3d": 150, "reproj_rms": 2.0,
            "feature_ms": 4.0, "match_ms": 5.0, "pnp_ms": 6.0,
            "used_refs": [0, 1],
        },
    ]
    call_index = []

    def fake_localize(*_args, **_kwargs):
        info = dict(pass_infos[len(call_index)])
        call_index.append(info["tag"])
        return object(), info

    pose = pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0)
    tracker._localize_with_candidates = fake_localize
    tracker._gate = lambda _ret, info, _mode: {
        "fast": (True, True, pose),
        "first": (True, False, pose),
        "full": (True, False, pose),
    }[info["tag"]]

    result = tracker._localize_frame_impl(np.zeros((12, 16, 3), np.uint8))
    info = tracker.last_info

    assert result is pose
    assert call_index == ["fast", "first", "full"]
    assert info["feature_ms"] == pytest.approx(15.0)
    assert info["match_ms"] == pytest.approx(27.0)
    assert info["pnp_ms"] == pytest.approx(39.0)
    assert info["nn_fast_feature_ms"] == pytest.approx(10.0)
    assert info["lg_first_match_ms"] == pytest.approx(2.0)
    assert info["lg_full_pnp_ms"] == pytest.approx(6.0)
    assert info["temporal_cache_attempted"] is True
    assert info["temporal_cache_corr3d"] == 44
    assert info["timing_synced"] is False


@pytest.mark.parametrize("full_ref_available", [True, False])
def test_full_ref_seed_materializes_inliers_only_as_fallback(full_ref_available):
    tracker = _composite_tracker()
    tracker.cfg.adaptive_first_topk = 0
    pose = pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0)
    context = (object(), np.zeros((1, 3), np.float32), [0], [0], "nn")
    info = {
        "inliers": 200,
        "corr3d": 200,
        "reproj_rms": 1.0,
        "feature_ms": 1.0,
        "match_ms": 2.0,
        "pnp_ms": 3.0,
        "used_refs": [0],
        "_temporal_cache_context": context,
    }
    tracker._localize_with_candidates = lambda *_args, **_kwargs: (object(), dict(info))
    tracker._gate = lambda *_args, **_kwargs: (True, False, pose)

    full_payload = {
        "descriptors": torch.ones((2, 2)),
        "xyz": np.ones((2, 3), np.float32),
        "ref_ids": np.zeros(2, np.int32),
        "source_stage": "full_ref",
    }
    inlier_payload = {
        "descriptors": torch.ones((1, 2)),
        "xyz": np.ones((1, 3), np.float32),
        "ref_ids": np.zeros(1, np.int32),
        "source_stage": "nn",
    }
    inlier_calls = []
    tracker._cache_payload_from_refs = (
        lambda *_args: full_payload if full_ref_available else None)

    def materialize(*args):
        inlier_calls.append(args)
        return inlier_payload

    tracker._cache_payload_from_ref_inliers = materialize
    stored = []
    tracker._set_temporal_cache = lambda payload: stored.append(payload) or True

    result = tracker._localize_frame_impl(np.zeros((12, 16, 3), np.uint8))

    assert result is pose
    assert stored == [full_payload if full_ref_available else inlier_payload]
    assert len(inlier_calls) == (0 if full_ref_available else 1)
    assert "_temporal_cache_context" not in tracker.last_info


def test_production_track_topk_is_pinned_to_1700():
    cfg = pxt.ProductionConfig()
    assert cfg.xfeat_topk_track == 1700
    assert cfg.adaptive_first_topk == 3
    assert cfg.flow_enabled is False


def test_predict_center_scales_velocity_by_capture_time() -> None:
    state = pxt.RuntimeState(
        prev_pose=pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=10.0),
        last_pose=pxt.Pose(1.0, 0.0, 0.0, 0.0, stamp=10.5),
        prev_center=np.array([0.0, 0.0, 0.0], np.float32),
        last_center=np.array([1.0, 0.0, 0.0], np.float32),
    )

    predicted = pxt.predict_center(state, capture_stamp=10.75)

    assert predicted == pytest.approx([1.5, 0.0, 0.0])


def test_lost_prior_expires_by_elapsed_capture_time_not_retry_count() -> None:
    state = pxt.RuntimeState(
        last_center=np.zeros(3, np.float32),
        lost_streak=100,
        lost_since_stamp=10.0,
    )

    assert not pxt.lost_prior_expired(state, 12.99, timeout_s=3.0)
    assert pxt.lost_prior_expired(state, 13.0, timeout_s=3.0)


def test_track_gate_rejects_physically_impossible_short_interval_jump(monkeypatch) -> None:
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig(
        max_jump=2.0,
        max_speed_mps=3.0,
        jump_slack_m=0.2,
    )
    tracker.state = pxt.RuntimeState(
        prev_pose=pxt.Pose(-0.1, 0.0, 0.0, 0.0, stamp=0.9),
        last_pose=pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0),
        prev_center=np.array([-0.1, 0.0, 0.0], np.float32),
        last_center=np.zeros(3, np.float32),
        last_yaw=0.0,
    )
    tracker._frame_capture_stamp = 1.1
    monkeypatch.setattr(
        pxt, "_pose_center_yaw_from_ret",
        lambda _ret: (np.array([1.0, 0.0, 0.0], np.float32), 0.0),
    )
    info = {
        "inliers": 100, "reproj_rms": 1.0,
        "inlier_ratio": 0.8, "inlier_grid_cells": 12,
    }

    accepted, _weak, _pose = tracker._gate(object(), info, "TRACK")

    assert not accepted
    assert info["jump_rejected"] is True
    assert info["jump_limit"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("quality", "reason"),
    [
        ({"inlier_ratio": 0.05, "inlier_grid_cells": 12}, "inlier_ratio"),
        ({"inlier_ratio": 0.8, "inlier_grid_cells": 2}, "inlier_spread"),
    ],
)
def test_track_gate_rejects_weak_or_concentrated_inlier_geometry(
    monkeypatch, quality, reason,
) -> None:
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig()
    tracker.state = pxt.RuntimeState()
    tracker._frame_capture_stamp = 1.0
    monkeypatch.setattr(
        pxt,
        "_pose_center_yaw_from_ret",
        lambda _ret: (np.zeros(3, np.float32), 0.0),
    )
    info = {"inliers": 100, "reproj_rms": 1.0, **quality}

    accepted, _weak, _pose = tracker._gate(object(), info, "TRACK")

    assert not accepted
    assert info["quality_rejected"] == reason


def test_xfeat_quality_gate_configuration_is_bounded() -> None:
    with pytest.raises(ValueError, match="min_inlier_ratio"):
        pxt.ProductionConfig(min_inlier_ratio=0.0)
    with pytest.raises(ValueError, match="min_inlier_grid_cells"):
        pxt.ProductionConfig(min_inlier_grid_cells=16)


def test_track_gate_accepts_pose_near_last_when_velocity_prediction_overshoots(
    monkeypatch,
) -> None:
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig(
        max_jump=2.0,
        max_speed_mps=3.0,
        jump_slack_m=0.2,
    )
    tracker.state = pxt.RuntimeState(
        prev_pose=pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=0.9),
        last_pose=pxt.Pose(1.0, 0.0, 0.0, 0.0, stamp=1.0),
        prev_center=np.zeros(3, np.float32),
        last_center=np.array([1.0, 0.0, 0.0], np.float32),
        last_yaw=0.0,
    )
    tracker._frame_capture_stamp = 1.1
    monkeypatch.setattr(
        pxt, "_pose_center_yaw_from_ret",
        lambda _ret: (np.array([1.1, 0.0, 0.0], np.float32), 0.0),
    )
    info = {
        "inliers": 100, "reproj_rms": 1.0,
        "inlier_ratio": 0.8, "inlier_grid_cells": 12,
    }

    accepted, _weak, _pose = tracker._gate(object(), info, "TRACK")

    assert accepted
    assert info["jump_from_pred"] == pytest.approx(0.9)
    assert info["jump_from_last"] == pytest.approx(0.1)


def test_lost_gate_rejects_far_or_reverse_reacquisition_with_prior(monkeypatch) -> None:
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig(
        acquire_max_jump=0.8,
        acquire_max_yaw_diff_deg=60.0,
    )
    tracker.state = pxt.RuntimeState(
        last_pose=pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0),
        last_center=np.zeros(3, np.float32),
        last_yaw=0.0,
        lost_since_stamp=1.0,
    )
    tracker._frame_capture_stamp = 1.1
    info = {
        "inliers": 200, "reproj_rms": 1.0,
        "inlier_ratio": 0.8, "inlier_grid_cells": 12,
    }
    monkeypatch.setattr(
        pxt, "_pose_center_yaw_from_ret",
        lambda _ret: (np.array([1.2, 0.0, 0.0], np.float32), 0.0),
    )

    accepted, _weak, _pose = tracker._gate(object(), info, "LOST")

    assert not accepted
    assert info["acquire_jump_rejected"] is True

    info = {
        "inliers": 200, "reproj_rms": 1.0,
        "inlier_ratio": 0.8, "inlier_grid_cells": 12,
    }
    monkeypatch.setattr(
        pxt, "_pose_center_yaw_from_ret",
        lambda _ret: (np.array([0.1, 0.0, 0.0], np.float32), math.pi),
    )
    accepted, _weak, _pose = tracker._gate(object(), info, "LOST")

    assert not accepted
    assert info["acquire_yaw_rejected"] is True


def test_flow_mnn_disagreement_falls_back_to_deep_same_frame(monkeypatch):
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker.cfg = pxt.ProductionConfig(flow_enabled=True, flow_mnn_crosscheck=True)
    tracker.state = pxt.RuntimeState(
        mode="TRACK", last_center=np.zeros(3, np.float32), last_yaw=0.0)
    tracker._flow_2d = np.column_stack((
        np.linspace(1.0, 14.0, 100), np.linspace(1.0, 10.0, 100))).astype(np.float32)
    tracker._flow_3d = np.column_stack((
        np.linspace(0.0, 1.0, 100), np.zeros(100), np.ones(100))).astype(np.float32)
    tracker._flow_gray = np.zeros((12, 16), np.uint8)
    tracker._flow_age = 0
    tracker._flow_refresh_every = 6
    tracker._flow_min_track = 20
    tracker._flow_fb_px = 0.5
    tracker._flow_qual_inl = 50
    tracker._flow_qual_ratio = 0.25
    tracker._flow_qual_reproj = 4.0
    tracker._flow_qual_ntrack = 100
    tracker._flow_mnn_crosscheck = True
    tracker._frame_capture_stamp = 1.0
    tracker._pnp = lambda *_args: {
        "num_inliers": 100,
        "inliers": np.ones(100, dtype=bool),
        "cam_from_world": SimpleNamespace(
            rotation=SimpleNamespace(matrix=lambda: np.eye(3)),
            translation=np.zeros(3),
        ),
    }
    tracker._estimate_reproj_rms = lambda *_args: 1.0
    cached_query = {"q_dev": object()}
    prior_counters = {"xfeat_extract_count": 1, "nn_call_count": 5}
    tracker._flow_mnn_pose_agrees = lambda *_args: (
        False,
        {"mnn_reason": "pose_disagreement"},
        cached_query,
        prior_counters,
    )
    fallback = []
    sentinel = object()
    tracker._flow_fallback_deep = (
        lambda _frame, stage, **info: fallback.append((stage, info)) or sentinel)

    def fake_lk(_src, _dst, points, _unused, **_kwargs):
        return points.copy(), np.ones((len(points), 1), np.uint8), None

    monkeypatch.setattr(pxt.cv2, "calcOpticalFlowPyrLK", fake_lk)

    result = tracker._localize_frame_flow(np.zeros((12, 16, 3), np.uint8))

    assert result is sentinel
    assert fallback[0][0] == "mnn_crosscheck"
    assert fallback[0][1]["mnn_reason"] == "pose_disagreement"
    assert fallback[0][1]["q_cache"] is cached_query
    assert fallback[0][1]["prior_counters"] is prior_counters


def test_flow_fallback_reuses_crosscheck_query_features():
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    tracker._flow_2d = np.ones((2, 2), np.float32)
    tracker._flow_3d = np.ones((2, 3), np.float32)
    tracker._flow_gray = np.ones((2, 2), np.uint8)
    tracker._flow_age = 1
    tracker._last_info = {}
    cached_query = {"q_dev": object()}
    prior_counters = {"xfeat_extract_count": 1, "nn_call_count": 5}
    calls = []
    sentinel = pxt.Pose(1.0, 2.0, 3.0, 0.1, stamp=1.0)
    tracker._flow_min_seed = 80
    tracker._last_inl_2d = np.column_stack((
        np.arange(80, dtype=np.float32),
        np.arange(80, dtype=np.float32) + 0.5,
    ))
    tracker._last_inl_3d = np.column_stack((
        np.arange(80, dtype=np.float32),
        np.zeros(80, np.float32),
        np.ones(80, np.float32),
    ))

    def deep(_frame, q_cache=None, prior_counters=None, force_lighterglue=False):
        calls.append((q_cache, prior_counters, force_lighterglue))
        tracker._last_info = {"accepted": True, "weak": False, "inliers": 80}
        return sentinel

    tracker._localize_frame_deep = deep
    result = tracker._flow_fallback_deep(
        np.zeros((2, 2, 3), np.uint8),
        "mnn_crosscheck",
        q_cache=cached_query,
        prior_counters=prior_counters,
        mnn_reason="quality_gate",
    )

    assert result is sentinel
    assert calls == [(cached_query, prior_counters, True)]
    assert len(tracker._flow_2d) == 80
    assert len(tracker._flow_3d) == 80
    assert tracker._flow_gray.shape == (2, 2)
    assert tracker._flow_age == 0
    assert tracker.last_info["flow_mnn_reason"] == "quality_gate"
    assert tracker.last_info["flow_fallback_matcher"] == "lighterglue"


def test_local_candidates_enforce_radius_for_near_covis_and_recent_refs():
    cfg = pxt.ProductionConfig(radius=0.8, near_pool=24, local_topk=5)
    state = pxt.RuntimeState(
        last_center=np.zeros(3, np.float32),
        last_refs=[2],
    )
    names = ["r0", "r1", "r2"]
    centers = np.array([
        [0.1, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.9, 0.0, 0.0],
    ], np.float32)

    selected = pxt.select_local_candidates(
        names, centers, None, {"r0": [2]}, state, cfg)

    assert selected == [0, 1]

    state.last_center = np.array([10.0, 0.0, 0.0], np.float32)
    assert pxt.select_local_candidates(
        names, centers, None, {"r2": [0]}, state, cfg) == []


def test_full_opencv_reprojection_rms_uses_calibrated_distortion():
    tracker = pxt.ProductionXFeatTracker.__new__(pxt.ProductionXFeatTracker)
    params = [
        960.4853099760471, 958.1961747147875,
        670.8167651412149, 358.7191813450141,
        -0.016359355216362784, 0.256336300878371,
        -0.006099082030819077, 0.019509803298460405,
        -0.1198628127364991, 0.0, 0.0, 0.0,
    ]
    tracker.cam = pxt.Camera("FULL_OPENCV", 1280, 720, params)
    pts3d = np.array([
        [0.0, 0.0, 2.0],
        [0.2, 0.0, 2.0],
        [0.0, 0.2, 2.0],
        [-0.2, -0.1, 2.5],
    ], np.float64)
    K = np.array([
        [params[0], 0.0, params[2]],
        [0.0, params[1], params[3]],
        [0.0, 0.0, 1.0],
    ])
    pts2d, _ = pxt.cv2.projectPoints(
        pts3d, np.zeros(3), np.zeros(3), K, np.asarray(params[4:]))
    ret = {
        "cam_from_world": SimpleNamespace(
            rotation=SimpleNamespace(matrix=lambda: np.eye(3)),
            translation=np.zeros(3),
        ),
        "inliers": np.ones(len(pts3d), dtype=bool),
    }

    assert tracker._estimate_reproj_rms(
        ret, pts2d.reshape(-1, 2), pts3d) == pytest.approx(0.0, abs=1e-6)


def test_candidate_pass_reports_unique_query_inliers(monkeypatch):
    tracker, _extract_calls = _candidate_tracker(monkeypatch, refs=2)
    tracker._estimate_reproj_rms = lambda *_args: 1.0
    monkeypatch.setattr(
        pxt,
        "mutual_nn_pairs",
        lambda *_args, **_kwargs: torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
    )

    class FakeOptions:
        def __init__(self):
            self.ransac = SimpleNamespace(max_error=None)

    fake_ret = {
        "num_inliers": 4,
        "inliers": np.ones(4, dtype=bool),
    }
    fake_pycolmap = SimpleNamespace(
        Camera=lambda **_kwargs: object(),
        AbsolutePoseEstimationOptions=FakeOptions,
        estimate_and_refine_absolute_pose=lambda *_args, **_kwargs: fake_ret,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    ret, info = tracker._localize_with_candidates(
        np.zeros((12, 16, 3), np.uint8), [0, 1], 1300, 1,
        matcher_mode="nn", q_cache={})

    assert ret is fake_ret
    assert info["inliers"] == 4
    assert info["unique_query_inliers"] == 2
    assert info["inlier_refs"] == [0, 1]


@pytest.mark.parametrize("unique_inliers, expected_passes", [(99, 2), (100, 1)])
def test_nn_fast_accept_uses_unique_query_inliers(unique_inliers, expected_passes):
    tracker = _composite_tracker()
    tracker.cfg.matcher_mode = "nn_then_lg"
    pose = pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0)
    calls = []

    def fake_localize(*_args, matcher_mode=None, **_kwargs):
        calls.append(matcher_mode)
        return object(), {
            "inliers": 120,
            "unique_query_inliers": unique_inliers,
            "corr3d": 140,
            "reproj_rms": 2.0,
            "feature_ms": 1.0,
            "match_ms": 1.0,
            "pnp_ms": 1.0,
            "used_refs": [0],
        }

    tracker._localize_with_candidates = fake_localize
    tracker._gate = lambda *_args, **_kwargs: (True, False, pose)

    assert tracker._localize_frame_impl(
        np.zeros((12, 16, 3), np.uint8)) is pose
    assert len(calls) == expected_passes


def test_forced_lighterglue_fallback_cannot_reaccept_nn_pose():
    tracker = _composite_tracker()
    pose = pxt.Pose(0.0, 0.0, 0.0, 0.0, stamp=1.0)
    cached_query = {"q_dev": object()}
    calls = []

    def fake_localize(_frame, candidates, *_args, matcher_mode=None,
                      q_cache=None, **_kwargs):
        calls.append((list(candidates), matcher_mode, q_cache))
        return object(), {
            "inliers": 120,
            "unique_query_inliers": 120,
            "corr3d": 140,
            "reproj_rms": 2.0,
            "feature_ms": 1.0,
            "match_ms": 1.0,
            "pnp_ms": 1.0,
            "used_refs": [0],
        }

    tracker._localize_with_candidates = fake_localize
    tracker._gate = lambda *_args, **_kwargs: (True, False, pose)

    result = tracker._localize_frame_impl(
        np.zeros((12, 16, 3), np.uint8),
        q_cache=cached_query,
        prior_counters={"nn_call_count": 5},
        force_lighterglue=True,
    )

    assert result is pose
    assert calls == [([0, 1, 2], "lighterglue", cached_query)]
    assert tracker.last_info["composite_stage"] == "lg_adaptive_forced"
    assert tracker.last_info["nn_call_count"] == 5
