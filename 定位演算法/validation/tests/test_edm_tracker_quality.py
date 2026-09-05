from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import sys

import numpy as np
import pycolmap
import pytest


DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import production_edm_tracker as tracker_module  # noqa: E402
import reloc_localizer_edm as edm_localizer_module  # noqa: E402
import edm_matcher as edm_matcher_module  # noqa: E402
from edm_matcher import (  # noqa: E402
    EDMMatcher,
    select_reference_grid_matches,
    _validate_matcher_cache_budget,
)
from production_edm_tracker import (  # noqa: E402
    adaptive_jump_limit,
    track_yaw_limit,
    EDMConfig,
    ProductionEDMTracker,
    RuntimeState,
    _CandidateSelection,
    _CorrespondenceBatch,
    reprojection_metrics,
    reprojection_rank,
    spatially_cap_indices,
)
from reloc_localizer_edm import (  # noqa: E402
    EDMLocalizer,
    MEGALOC_TENSORRT_GPU_IDENTITY,
    _validate_edm_bundle_schema,
    _resolve_cuda_device,
    _validate_megaloc_engine_file,
    _validate_megaloc_plan_size,
    _validate_megaloc_tensorrt_compatibility,
    deserialize_megaloc_cuda_engine,
    normalize_gpu_identity,
)


def test_edm_megaloc_loads_only_from_verified_local_assets(monkeypatch) -> None:
    monkeypatch.delenv("SFM_MEGALOC_TENSORRT", raising=False)
    verified = []
    loaded = {}

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

    monkeypatch.setattr(
        edm_localizer_module,
        "verify_sha256",
        lambda path, digest: verified.append((Path(path), digest)),
    )
    monkeypatch.setattr(edm_localizer_module.torch.hub, "set_dir", lambda path: None)

    def fake_load(*args, **kwargs):
        loaded["args"] = args
        loaded["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(edm_localizer_module.torch.hub, "load", fake_load)

    query = edm_localizer_module.MegaLocQuery(device="cpu")

    assert loaded["args"] == (
        str(edm_localizer_module.MEGALOC_REPO_DIR),
        "get_trained_model",
    )
    assert loaded["kwargs"] == {
        "source": "local",
        "weights_path": str(edm_localizer_module.MEGALOC_WEIGHTS),
    }
    assert verified == [
        (
            edm_localizer_module.MEGALOC_REPO_DIR / "hubconf.py",
            edm_localizer_module.MEGALOC_HUBCONF_SHA256,
        ),
        (
            edm_localizer_module.MEGALOC_REPO_DIR / "megaloc_model.py",
            edm_localizer_module.MEGALOC_MODEL_SOURCE_SHA256,
        ),
        (
            edm_localizer_module.MEGALOC_WEIGHTS,
            edm_localizer_module.MEGALOC_WEIGHTS_SHA256,
        ),
    ]
    assert query.fp16 is False
    assert query.tensorrt is False
    assert query.backend == "pytorch"

    monkeypatch.setenv("SFM_MEGALOC_FP16", "1")
    assert edm_localizer_module.MegaLocQuery(device="cuda").fp16 is True
    assert edm_localizer_module.MegaLocQuery(device="cpu").fp16 is False

    explicit = edm_localizer_module.MegaLocQuery(
        device="cuda", backend="pytorch"
    )
    assert explicit.fp16 is False
    assert explicit.backend == "pytorch"


def test_edm_megaloc_tensorrt_is_cuda_default_and_hash_verified(
    monkeypatch, tmp_path: Path
) -> None:
    engine_path = tmp_path / "megaloc.engine"
    engine_sha256 = "a" * 64
    verified = []
    created = []

    class FakeEngine:
        def __init__(self, path, device, *, require_sanctioned_size):
            created.append((path, device, require_sanctioned_size))

    monkeypatch.delenv("SFM_MEGALOC_BACKEND", raising=False)
    monkeypatch.delenv("SFM_MEGALOC_TENSORRT", raising=False)
    monkeypatch.delenv("SFM_MEGALOC_FP16", raising=False)
    monkeypatch.setenv("SFM_MEGALOC_TENSORRT_ENGINE", str(engine_path))
    monkeypatch.setenv("SFM_MEGALOC_TENSORRT_ENGINE_SHA256", engine_sha256)
    monkeypatch.setattr(
        edm_localizer_module,
        "verify_sha256",
        lambda path, digest: verified.append((Path(path), digest)),
    )
    monkeypatch.setattr(edm_localizer_module, "_MegaLocTensorRTEngine", FakeEngine)
    monkeypatch.setattr(
        edm_localizer_module.torch.hub,
        "load",
        lambda *_args, **_kwargs: pytest.fail("PyTorch model must not load"),
    )

    query = edm_localizer_module.MegaLocQuery(device="cuda")

    assert query.tensorrt is True
    assert query.fp16 is False
    assert query.model is None
    assert verified == [(engine_path, engine_sha256)]
    assert created == [(engine_path, "cuda", False)]


def test_edm_megaloc_tensorrt_requires_cuda(monkeypatch) -> None:
    monkeypatch.setenv("SFM_MEGALOC_TENSORRT", "1")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        edm_localizer_module.MegaLocQuery(device="cpu")


def test_edm_retrieval_can_rank_only_nearby_candidates() -> None:
    reloc_map = SimpleNamespace(
        ref_names=["near_bad", "global_best", "near_good"],
        ref_global=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]],
            dtype=np.float32,
        ),
    )
    megaloc = SimpleNamespace(
        extract_one=lambda _frame: np.asarray([1.0, 0.0], dtype=np.float32)
    )
    localizer = EDMLocalizer(
        reloc_map,
        SimpleNamespace(width=1280, height=720),
        matcher=object(),
        megaloc=megaloc,
    )

    refs = localizer.retrieve(
        np.zeros((322, 322, 3), dtype=np.uint8),
        1,
        candidates=["near_bad", "near_good"],
    )

    assert refs == ["near_good"]


def test_edm_megaloc_fp16_autocast_keeps_fp32_output(monkeypatch) -> None:
    autocast_calls = []

    class FakeImage:
        @staticmethod
        def fromarray(_array):
            return SimpleNamespace(convert=lambda _mode: object())

    class FakeAutocast:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    def fake_autocast(*, device_type, dtype, enabled):
        autocast_calls.append((device_type, dtype, enabled))
        return FakeAutocast()

    query = edm_localizer_module.MegaLocQuery.__new__(
        edm_localizer_module.MegaLocQuery
    )
    query._Image = FakeImage
    query.device = "cpu"
    query.fp16 = True
    query._trt_engine = None
    query.tf = lambda _image: edm_localizer_module.torch.ones((3, 2, 2))
    query.model = lambda _tensor: edm_localizer_module.torch.ones(
        (1, 4), dtype=edm_localizer_module.torch.float16
    )
    monkeypatch.setattr(edm_localizer_module.torch, "autocast", fake_autocast)

    descriptor = query.extract_one(np.zeros((2, 2, 3), dtype=np.uint8))

    assert autocast_calls == [
        ("cuda", edm_localizer_module.torch.float16, True)
    ]
    assert descriptor.dtype == np.float32
    np.testing.assert_allclose(descriptor, np.full(4, 0.5, dtype=np.float32))


def test_edm_megaloc_tensorrt_keeps_fp32_output(monkeypatch) -> None:
    class FakeImage:
        @staticmethod
        def fromarray(_array):
            return SimpleNamespace(convert=lambda _mode: object())

    query = edm_localizer_module.MegaLocQuery.__new__(
        edm_localizer_module.MegaLocQuery
    )
    query._Image = FakeImage
    query.device = "cpu"
    query.fp16 = False
    query.tf = lambda _image: edm_localizer_module.torch.ones((3, 2, 2))
    query.model = None
    query._trt_engine = lambda _tensor: edm_localizer_module.torch.tensor(
        [[3.0, 4.0]], dtype=edm_localizer_module.torch.float16
    )
    monkeypatch.setattr(
        edm_localizer_module.torch,
        "autocast",
        lambda **_kwargs: pytest.fail("TensorRT must not enter torch.autocast"),
    )

    descriptor = query.extract_one(np.zeros((2, 2, 3), dtype=np.uint8))

    assert descriptor.dtype == np.float32
    np.testing.assert_allclose(descriptor, np.array([0.6, 0.8], dtype=np.float32))


def test_edm_localizer_keeps_configured_megaloc_lazy() -> None:
    created = []
    sentinel = object()
    reloc_map = edm_localizer_module.EDMRelocMap(
        ref_names=[],
        ref_global=np.empty((0, edm_localizer_module.EDM_GLOBAL_DESCRIPTOR_DIM), np.float32),
        xyz_by_cell={},
        images={},
        meta={},
    )
    camera = edm_localizer_module.Camera("PINHOLE", 1280, 720, [1, 1, 1, 1])
    localizer = edm_localizer_module.EDMLocalizer(
        reloc_map,
        camera,
        matcher=object(),
        megaloc_factory=lambda: created.append(True) or sentinel,
    )

    assert created == []
    assert localizer.megaloc is sentinel
    assert localizer.megaloc is sentinel
    assert created == [True]


def test_edm_config_rejects_non_finite_and_relational_values() -> None:
    with pytest.raises(ValueError, match="max_jump"):
        EDMConfig(max_jump=float("nan"))
    with pytest.raises(ValueError, match="acquire_initial_topk"):
        EDMConfig(boot_global_topk=2, acquire_initial_topk=3)
    with pytest.raises(ValueError, match="history"):
        EDMConfig(adaptive_jump_min_history=20, adaptive_jump_history_size=10)
    with pytest.raises(ValueError, match="bootstrap"):
        EDMConfig(adaptive_jump_bootstrap=0.005, adaptive_jump_ceiling=0.008)
    with pytest.raises(ValueError, match="min_inlier_ratio"):
        EDMConfig(min_inlier_ratio=0.0)
    with pytest.raises(ValueError, match="min_inlier_grid_cells"):
        EDMConfig(corr_grid=2, min_inlier_grid_cells=5)


def test_lost_global_retrieval_interval_is_nonnegative_and_defaults_to_three() -> None:
    # Default 3 from the 2026-09-02 river 1045-ref gate (ledger item 12); 0 restores
    # the legacy one-shot behavior and is still accepted per site profile.
    assert EDMConfig().lost_global_retrieval_interval == 3
    assert EDMConfig(lost_global_retrieval_interval=0).lost_global_retrieval_interval == 0

    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="lost_global_retrieval_interval"):
            EDMConfig(lost_global_retrieval_interval=value)


def _lost_retrieval_policy_tracker(*, interval: int) -> ProductionEDMTracker:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        global_retrieval_policy="boot_and_lost_once",
        lost_global_retrieval_interval=interval,
    )
    tracker.st = RuntimeState(state="LOST")
    return tracker


def test_lost_global_retrieval_starts_after_local_grace() -> None:
    tracker = _lost_retrieval_policy_tracker(interval=2)
    tracker.st.lost_frames = tracker.cfg.lost_local_grace_frames

    assert not tracker._should_run_global_retrieval()

    tracker.st.lost_frames += 1
    assert tracker._should_run_global_retrieval()


def test_lost_global_retrieval_retries_only_when_interval_is_reached(
    monkeypatch,
) -> None:
    tracker = _lost_retrieval_policy_tracker(interval=2)
    calls = []

    def fake_global_retrieval(_frame, _capture_stamp):
        calls.append(tracker.st.lost_frames)
        tracker.st.lost_global_retrieval_attempts += 1
        return [], 0.0, False

    monkeypatch.setattr(tracker, "_global_retrieval", fake_global_retrieval)
    for lost_frame in range(1, 6):
        tracker.st.lost_frames = lost_frame
        if tracker._should_run_global_retrieval():
            tracker._global_retrieval(None, float(lost_frame))

    assert calls == [tracker.cfg.lost_local_grace_frames + 1, 5]


def test_progressive_lost_stages_advance_every_two_frames_before_global(
    monkeypatch,
) -> None:
    tracker = _lost_retrieval_policy_tracker(interval=15)
    calls = []

    def fake_global_retrieval(_frame, _capture_stamp):
        calls.append(tracker.st.lost_frames)
        tracker.st.lost_global_retrieval_attempts += 1
        return [], 0.0, True

    monkeypatch.setattr(tracker, "_global_retrieval", fake_global_retrieval)
    for lost_frame in range(1, 11):
        tracker.st.lost_frames = lost_frame
        if tracker._should_run_global_retrieval():
            tracker._global_retrieval(None, float(lost_frame))

    assert calls == [3, 5, 7, 9]


def test_lost_global_retrieval_interval_zero_keeps_one_shot_behavior() -> None:
    tracker = _lost_retrieval_policy_tracker(interval=0)
    tracker.st.lost_frames = tracker.cfg.lost_local_grace_frames + 1
    assert tracker._should_run_global_retrieval()

    tracker.st.lost_global_retrieval_attempts = 1
    tracker.st.lost_frames += 100
    assert not tracker._should_run_global_retrieval()


def test_lost_megaloc_progresses_through_local_rings_before_global(
    monkeypatch,
) -> None:
    import cv2

    monkeypatch.setattr(cv2, "cvtColor", lambda frame, _code: frame)
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(lost_prior_strategy="restrict_nearby")
    tracker.st = RuntimeState(
        state="LOST",
        center=np.zeros(3, np.float32),
        last_capture_stamp=0.0,
    )
    tracker.map = SimpleNamespace(ref_names=[f"ref{i}" for i in range(340)])
    calls = []

    def track_candidates(topk, _stamp=None, *, radius_factor=1.0, strict_radius=False):
        calls.append((topk, radius_factor, strict_radius))
        return [f"near-{radius_factor}"]

    retrieved = []
    tracker._track_candidates = track_candidates
    tracker.loc = SimpleNamespace(
        retrieve=lambda _rgb, _k, candidates=None: (
            retrieved.append(candidates) or (["global"] if candidates is None else candidates[:1])
        )
    )

    stages = []
    for attempt in range(4):
        tracker.st.lost_global_retrieval_attempts = attempt
        refs, _ms, nearby = tracker._global_retrieval(
            np.zeros((8, 8, 3), np.uint8), 100.0
        )
        stages.append(
            (
                refs,
                nearby,
                tracker._last_lost_search_stage,
                tracker._last_lost_radius_factor,
            )
        )

    assert [call[1] for call in calls] == [1.0, 2.0, 4.0]
    assert all(call[2] for call in calls)
    assert [item[2] for item in stages] == ["near_1x", "near_2x", "near_4x", "global"]
    assert [item[3] for item in stages] == [1.0, 2.0, 4.0, None]
    assert [item[1] for item in stages] == [True, True, True, False]
    assert retrieved[-1] is None


def test_lost_local_recovery_uses_the_next_progressive_ring() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(lost_prior_strategy="restrict_nearby")
    tracker.st = RuntimeState(
        state="LOST",
        lost_frames=tracker.cfg.lost_local_grace_frames + 2,
    )
    calls = []
    tracker._track_candidates = lambda topk, _stamp=None, **kwargs: (
        calls.append((topk, kwargs)) or ["near"]
    )
    tracker._recovery_candidates = lambda _topk: pytest.fail(
        "recovery-bank must wait until progressive local rings are exhausted"
    )

    tracker.st.lost_global_retrieval_attempts = 1
    refs, mode = tracker._local_acquisition_candidates(1.0)
    tracker.st.lost_global_retrieval_attempts = 2
    refs2, mode2 = tracker._local_acquisition_candidates(2.0)

    assert refs == refs2 == ["near"]
    assert mode == mode2 == "edm_local_recovery"
    assert [call[0] for call in calls] == [tracker.cfg.recovery_scan_topk] * 2
    assert [call[1]["radius_factor"] for call in calls] == [2.0, 4.0]
    assert all(call[1]["strict_radius"] for call in calls)


def test_progressive_local_search_does_not_fall_back_to_boot_refs_early() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(lost_prior_strategy="restrict_nearby")
    tracker.st = RuntimeState(
        state="LOST",
        lost_frames=tracker.cfg.lost_local_grace_frames + 2,
        lost_global_retrieval_attempts=1,
        boot_refs=["global-boot-ref"],
    )
    tracker._track_candidates = lambda *_args, **_kwargs: []

    refs, mode = tracker._local_acquisition_candidates(1.0)

    assert refs == []
    assert mode == "edm_local_recovery"


def test_track_does_not_schedule_lost_global_retrieval_retry() -> None:
    tracker = _lost_retrieval_policy_tracker(interval=1)
    tracker.st.state = "TRACK"
    tracker.st.lost_frames = 100
    tracker.st.lost_global_retrieval_attempts = 1

    assert not tracker._should_run_global_retrieval()


def test_adaptive_jump_uses_bootstrap_before_learning_then_tightens() -> None:
    cfg = EDMConfig()

    assert adaptive_jump_limit([], cfg) == pytest.approx(
        cfg.adaptive_jump_bootstrap
    )
    assert adaptive_jump_limit([0.0001] * cfg.adaptive_jump_min_history, cfg) == pytest.approx(
        cfg.adaptive_jump_floor
    )


def test_adaptive_jump_scales_with_capture_interval_and_clamps_stale_time() -> None:
    cfg = EDMConfig(
        adaptive_jump_min_history=3,
        adaptive_jump_factor=2.0,
        adaptive_jump_floor=0.001,
        adaptive_jump_ceiling=0.02,
        prediction_max_dt=0.25,
    )
    steps = [0.002] * 3
    capture_dts = [0.05] * 3

    assert adaptive_jump_limit(
        steps, cfg, capture_dt=0.20, capture_dt_history=capture_dts
    ) == pytest.approx(0.016)
    assert adaptive_jump_limit(
        steps, cfg, capture_dt=2.0, capture_dt_history=capture_dts
    ) == pytest.approx(0.020)
    assert adaptive_jump_limit(
        steps, cfg, capture_dt=0.01, capture_dt_history=capture_dts
    ) == pytest.approx(0.004)


def test_track_yaw_limit_scales_with_capture_time_and_has_a_hard_cap() -> None:
    cfg = EDMConfig(
        track_yaw_slack_deg=8.0,
        track_max_yaw_rate_deg_s=120.0,
        track_max_yaw_step_deg=30.0,
    )

    assert track_yaw_limit(cfg, 1.0 / 24.0) == pytest.approx(13.0)
    assert track_yaw_limit(cfg, 1.0) == pytest.approx(30.0)


def test_runtime_step_history_is_bounded() -> None:
    state = RuntimeState()
    for value in range(1000):
        state.accepted_step_norms.append(float(value))
        state.observed_capture_dts.append(float(value))
    assert len(state.accepted_step_norms) == 120
    assert state.accepted_step_norms[0] == 880.0
    assert len(state.observed_capture_dts) == 120
    assert state.observed_capture_dts[0] == 880.0


def test_edm_bundle_schema_rejects_non_finite_descriptors() -> None:
    bundle = {
        "meta": {
            "feature": "edm",
            "edm_grid_w": 128,
            "edm_grid_h": 72,
            "edm_input_w": 1024,
            "edm_input_h": 576,
        },
        "ref_names": ["ref"],
        "ref_global": np.array([[float("nan")]], dtype=np.float32),
        "refs": {
            "ref": {
                "xyz_by_cell": np.full((128 * 72, 3), np.nan, np.float32),
                "image_jpg": np.array([1], dtype=np.uint8),
            },
        },
    }
    with pytest.raises(ValueError, match="ref_global"):
        _validate_edm_bundle_schema(bundle)


def test_reprojection_metrics_use_the_pycolmap_camera_model() -> None:
    camera = pycolmap.Camera(
        model="SIMPLE_RADIAL", width=1280, height=720,
        params=[930.0, 640.0, 360.0, 0.02],
    )
    points3d = np.array([
        [-1.0, -0.5, 4.0], [0.0, 0.0, 3.0], [0.8, 0.3, 5.0], [1.2, -0.7, 6.0],
    ])
    points2d = camera.img_from_cam(points3d)
    ret = {
        "cam_from_world": pycolmap.Rigid3d(),
        "inlier_mask": np.ones(len(points3d), dtype=bool),
    }

    metrics = reprojection_metrics(ret, points2d, points3d, camera, grid=4)

    assert metrics["reproj_rms"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["inlier_ratio"] == 1.0
    assert metrics["inlier_grid_cells"] >= 2


def test_perfect_reprojection_ranks_above_missing_or_nonzero_error() -> None:
    assert reprojection_rank(0.0) > reprojection_rank(0.1)
    assert reprojection_rank(0.0) > reprojection_rank(None)


def test_spatial_cap_keeps_highest_confidence_from_each_cell_first() -> None:
    points2d = np.array([[10, 10], [20, 20], [700, 10], [710, 20]], dtype=float)
    confidence = np.array([0.1, 0.9, 0.8, 0.7], dtype=float)

    selected = spatially_cap_indices(
        points2d, confidence, max_total=2, width=1280, height=720, grid=2)

    assert selected.tolist() == [1, 2]


def test_prediction_scales_velocity_by_capture_dt_and_clamps_stale_time() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(prediction_max_dt=0.25)
    tracker.st = RuntimeState(
        center=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([2.0, 0.0, -4.0]),
        last_capture_stamp=10.0,
    )

    assert np.allclose(tracker._predict_center(10.1), [1.2, 2.0, 2.6])
    assert np.allclose(tracker._predict_center(11.0), [1.5, 2.0, 2.0])
    assert np.allclose(tracker._predict_center(9.0), tracker.st.center)


def test_reference_tensor_lru_reuses_the_map_image_device_tensor() -> None:
    matcher = object.__new__(EDMMatcher)
    matcher.device = "cpu"
    matcher.reference_cache_size = 2
    matcher._reference_tensor_cache = OrderedDict()
    source = np.full((8, 8), 127, dtype=np.uint8)

    first = matcher.reference_tensor(source)
    second = matcher.reference_tensor(source)

    assert first is second
    assert first.device.type == "cpu"
    assert len(matcher._reference_tensor_cache) == 1


def test_edm_checkpoint_is_verified_before_weights_only_load(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "fixture.ckpt"
    checkpoint.write_bytes(b"trusted checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls = {}

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"state_dict": OrderedDict({"weight": object()})}

    monkeypatch.setattr(edm_matcher_module.torch, "load", fake_load)
    state = edm_matcher_module._load_edm_state_dict(checkpoint, digest)

    assert list(state) == ["weight"]
    assert calls == {
        "path": str(checkpoint),
        "kwargs": {"map_location": "cpu", "weights_only": True},
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        edm_matcher_module._load_edm_state_dict(checkpoint, "0" * 64)


def test_default_edm_checkpoint_is_automatically_hash_pinned() -> None:
    assert edm_matcher_module._checkpoint_sha256_for_load(
        edm_matcher_module.DEFAULT_CKPT,
        None,
    ) == edm_matcher_module.DEFAULT_CKPT_SHA256


def test_correspondence_confidence_stays_aligned_after_3d_filtering() -> None:
    localizer = object.__new__(EDMLocalizer)
    localizer.scale = 1.25
    localizer.matcher = SimpleNamespace(
        match_many_to_one=lambda _images, _query: [{
            "mkpts0": np.array([[8.0, 8.0], [16.0, 8.0]], dtype=np.float32),
            "mkpts1": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            "mconf": np.array([0.9, 0.4], dtype=np.float32),
        }]
    )
    xyz = np.full((128 * 72, 3), np.nan, dtype=np.float32)
    xyz[1 * 128 + 1] = [1.0, 2.0, 3.0]

    rows = localizer.correspondences_for_sources(
        np.zeros((576, 1024), np.uint8),
        [np.zeros((576, 1024), np.uint8)], [xyz],
    )

    points2d, points3d, confidence, count = rows[0]
    assert points2d.tolist() == [[12.5, 25.0]]
    assert points3d.tolist() == [[1.0, 2.0, 3.0]]
    assert confidence.tolist() == pytest.approx([0.9])
    assert count == 1


def test_correspondence_chunks_keep_query_preparation_isolated() -> None:
    class FakeMatcher:
        def __init__(self):
            self.calls = []

        def match_many_to_one(self, images, query, *, prepared_query=None):
            self.calls.append((len(images), query, prepared_query))
            empty = np.zeros((0, 2), np.float32)
            return [
                {"mkpts0": empty, "mkpts1": empty, "mconf": np.zeros(0, np.float32)}
                for _image in images
            ]

    matcher = FakeMatcher()
    localizer = object.__new__(EDMLocalizer)
    localizer.scale = np.asarray((1.0, 1.0), np.float32)
    localizer.matcher = matcher
    query = np.zeros((576, 1024), np.uint8)
    refs = [np.full_like(query, index) for index in range(5)]
    xyz = [np.full((128 * 72, 3), np.nan, np.float32) for _ref in refs]

    rows = localizer.correspondences_for_sources(query, refs, xyz, batch_size=2)

    assert len(rows) == 5
    assert [call[0] for call in matcher.calls] == [2, 2, 1]
    assert all(call[1] is query and call[2] is None for call in matcher.calls)


def test_correspondence_chunks_reuse_one_prepared_query() -> None:
    prepared = object()

    class FakeMatcher:
        def __init__(self):
            self.prepared = 0
            self.calls = []

        def prepare_query(self, query):
            self.prepared += 1
            return prepared

        def match_many_to_one(self, images, query, *, prepared_query=None):
            self.calls.append((len(images), query, prepared_query))
            empty = np.zeros((0, 2), np.float32)
            return [
                {"mkpts0": empty, "mkpts1": empty, "mconf": np.zeros(0, np.float32)}
                for _image in images
            ]

    matcher = FakeMatcher()
    localizer = object.__new__(EDMLocalizer)
    localizer.scale = np.asarray((1.0, 1.0), np.float32)
    localizer.matcher = matcher
    query = np.zeros((576, 1024), np.uint8)
    refs = [np.full_like(query, index) for index in range(5)]
    xyz = [np.full((128 * 72, 3), np.nan, np.float32) for _ref in refs]

    rows = localizer.correspondences_for_sources(query, refs, xyz, batch_size=2)

    assert len(rows) == 5
    assert matcher.prepared == 1
    assert [call[0] for call in matcher.calls] == [2, 2, 1]
    assert all(call[1] is query and call[2] is prepared for call in matcher.calls)


def test_edm_retrieval_keeps_similarity_and_topk_on_the_tensor_device() -> None:
    class FakeMegaLoc:
        device = edm_localizer_module.torch.device("cpu")

        def __init__(self):
            self.tensor_calls = 0

        def extract_one_tensor(self, _frame):
            self.tensor_calls += 1
            return edm_localizer_module.torch.tensor([1.0, 0.0])

        def extract_one(self, _frame):
            raise AssertionError("device retrieval must not round-trip the descriptor")

    reloc_map = SimpleNamespace(
        ref_names=["second", "best", "third"],
        ref_global=np.asarray(
            [[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]], dtype=np.float32
        ),
    )
    megaloc = FakeMegaLoc()
    localizer = EDMLocalizer(
        reloc_map,
        SimpleNamespace(width=1280, height=720),
        matcher=object(),
        megaloc=megaloc,
    )

    scored = localizer.retrieve_scored(
        np.zeros((720, 1280, 3), np.uint8), 2
    )

    assert [name for name, _score in scored] == ["best", "second"]
    assert megaloc.tensor_calls == 1
    assert localizer._ref_global_tensor.device.type == "cpu"


def test_edm_config_validates_pnp_acceleration_fields() -> None:
    with pytest.raises(ValueError, match="pnp_workers"):
        EDMConfig(pnp_workers=0)
    with pytest.raises(ValueError, match="pnp_early_stop"):
        EDMConfig(pnp_early_stop=1)
    with pytest.raises(ValueError, match="pnp_pipeline"):
        EDMConfig(pnp_pipeline=1)


def test_progressive_acquisition_uses_cumulative_2_4_8_20_stages() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        boot_global_topk=20,
        acquire_stage_mode="progressive",
    )
    names = [f"ref{index}" for index in range(20)]
    selection = _CandidateSelection(True, names, "megaloc_boot", 80, 0.0)
    calls = []
    empty = np.zeros((0, 2), np.float32)

    def batch(count):
        rows = [(empty, np.zeros((0, 3)), np.zeros(0), 0)] * count
        return _CorrespondenceBatch(empty, np.zeros((0, 3)), np.zeros(0), [0] * count, rows, False)

    attempt = tracker_module._PoseAttempt(
        None,
        empty,
        np.zeros((0, 3)),
        {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0},
        None,
        1.0,
    )
    tracker._match_and_estimate = lambda _gray, staged, **_kwargs: (
        calls.append(list(staged.refs)) or batch(len(staged.refs)),
        attempt,
        2.0,
    )
    tracker._collect_correspondences = lambda _gray, staged, **_kwargs: (
        calls.append(list(staged.refs)) or batch(len(staged.refs))
    )
    tracker._best_pose_attempt = lambda *_args, **_kwargs: attempt
    tracker._acquire_stage_can_stop = (
        lambda staged, _attempt, _stamp: len(staged.refs) >= 8
    )

    active, _batch, result, match_ms, evaluated = tracker._match_acquisition_stages(
        np.zeros((576, 1024), np.uint8),
        selection,
        1.0,
        None,
    )

    assert [len(refs) for refs in calls] == [2, 2, 4]
    assert calls == [names[:2], names[2:4], names[4:8]]
    assert active.refs == names[:8]
    assert evaluated == [2, 4, 8]
    assert result.pnp_ms == pytest.approx(3.0)
    assert match_ms >= 2.0


def test_lost_grace_progressive_acquisition_avoids_duplicate_matching_and_matches_full_batch() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        acquire_stage_mode="progressive",
        use_temporal_reference=True,
    )
    tracker.st = RuntimeState(state="LOST", lost_frames=1)
    tracker.temporal_gray = np.ones((576, 1024), np.uint8)
    tracker.temporal_xyz_by_cell = np.ones((128 * 72, 3), np.float32)

    ref_names = ["ref0", "ref1", "ref2", "ref3", "ref4"]
    selection = _CandidateSelection(True, ref_names, "edm_local_recovery", 50, 0.0)

    def make_row(marker: int, size: int = 10):
        p2 = np.full((size, 2), marker, dtype=np.float32)
        p3 = np.full((size, 3), marker, dtype=np.float32)
        conf = np.full(size, 0.5 + marker * 0.05, dtype=np.float32)
        return (p2, p3, conf, size)

    source_rows = {
        "temporal": make_row(0),
        "ref0": make_row(1),
        "ref1": make_row(2),
        "ref2": make_row(3),
        "ref3": make_row(4),
        "ref4": make_row(5),
    }

    forward_calls = []

    def mock_correspondences_for_sources(
        _gray,
        images,
        xyz_luts,
        batch_size=None,
        source_kinds=None,
        **kwargs,
    ):
        kinds = source_kinds or ["map"] * len(images)
        rows = []
        for i, kind in enumerate(kinds):
            if kind == "temporal":
                forward_calls.append("temporal")
                rows.append(source_rows["temporal"])
            else:
                name = None
                for ref_name in ref_names:
                    if images[i] is tracker.map.images[ref_name]:
                        name = ref_name
                        break
                forward_calls.append(name or f"map_{i}")
                rows.append(source_rows[name])
        return rows

    def mock_correspondences_by_ref(_gray, refs, **kwargs):
        rows = []
        for name in refs:
            forward_calls.append(name)
            rows.append(source_rows[name])
        return rows

    tracker.map = SimpleNamespace(
        images={name: np.ones((576, 1024), np.uint8) * (idx + 2) for idx, name in enumerate(ref_names)},
        xyz_by_cell={name: np.ones((128 * 72, 3), np.float32) for name in ref_names},
    )
    tracker.loc = SimpleNamespace(
        map=tracker.map,
        correspondences_for_sources=mock_correspondences_for_sources,
        correspondences_by_ref=mock_correspondences_by_ref,
    )
    tracker.cam = SimpleNamespace(width=1280, height=720)
    tracker._pose_estimation_context = lambda: (object(), object())
    tracker._new_pose_estimation_context = lambda _seed=-1: (object(), object())
    tracker._acquire_stage_can_stop = lambda _staged, _attempt, _stamp: False

    def mock_estimate_pose(p2, p3, conf, _cam, _opt):
        return ({"num_inliers": len(p3), "center": np.mean(p3, axis=0)}, p2, p3, {}), 1.0

    tracker._estimate_pose_candidate = mock_estimate_pose

    active, staged_batch, staged_attempt, match_ms, evaluated = tracker._match_acquisition_stages(
        np.zeros((576, 1024), np.uint8),
        selection,
        1.0,
        None,
    )

    assert forward_calls == ["temporal", "ref0", "ref1", "ref2", "ref3", "ref4"]
    assert len(forward_calls) == 6
    assert evaluated == [2, 4, 5]

    full_sources_rows = [
        source_rows["temporal"],
        source_rows["ref0"],
        source_rows["ref1"],
        source_rows["ref2"],
        source_rows["ref3"],
        source_rows["ref4"],
    ]
    p2_full, p3_full, conf_full, per_ref_full = tracker_module._merge_correspondence_rows(full_sources_rows)

    assert np.array_equal(staged_batch.points2d, p2_full)
    assert np.array_equal(staged_batch.points3d, p3_full)
    assert np.array_equal(staged_batch.confidence, conf_full)
    assert staged_batch.per_ref == per_ref_full
    assert staged_batch.temporal_used is True
    assert staged_batch.by_ref is None

    full_attempt_result, _ = mock_estimate_pose(p2_full, p3_full, conf_full, None, None)
    assert staged_attempt.result["num_inliers"] == full_attempt_result[0]["num_inliers"]
    assert np.array_equal(staged_attempt.result["center"], full_attempt_result[0]["center"])

def test_ranked_pnp_runs_best_batch_first_and_stops_after_gates() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        pnp_workers=1,
        pnp_early_stop=False,
        pnp_ranked_batches=True,
    )
    tracker.cam = SimpleNamespace(width=1280, height=720)
    tracker.st = RuntimeState(state="TRACK")
    calls = []

    def estimate(row):
        marker = int(row[0][0, 0])
        calls.append(marker)
        return (
            {"num_inliers": len(row[1])},
            row[0],
            row[1],
            {"reproj_rms": 1.0, "inlier_ratio": 1.0, "inlier_grid_cells": 8},
        )

    tracker._estimate_reference_row = estimate
    tracker._pnp_candidates_can_stop = lambda *_args: True
    tracker._select_track_candidate = lambda _selection, scored: scored[0]
    selection = _CandidateSelection(
        False,
        ["small", "best", "middle"],
        "edm_map_scan",
        50,
        0.0,
    )

    attempt = tracker._best_pose_attempt(
        selection,
        _pnp_batch([80, 120, 100]),
        capture_stamp=1.0,
    )

    assert calls == [1]
    assert attempt.selected_ref == "best"
    assert attempt.pnp_candidates == 1
    assert attempt.pnp_skipped == 2


def test_track_map_first_reuses_map_correspondences_for_temporal_fallback() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        use_temporal_reference=True,
        track_map_first=True,
    )
    tracker.temporal_gray = np.ones((576, 1024), np.uint8)
    tracker.temporal_xyz_by_cell = np.ones((128 * 72, 3), np.float32)
    map_batch = _pnp_batch([60])
    empty_attempt = tracker_module._PoseAttempt(
        None,
        np.zeros((0, 2)),
        np.zeros((0, 3)),
        {"reproj_rms": None, "inlier_ratio": 0.0, "inlier_grid_cells": 0},
        None,
        1.0,
    )
    tracker._match_and_estimate = lambda *_args, **_kwargs: (
        map_batch,
        empty_attempt,
        2.0,
    )
    tracker._track_stage_can_stop = lambda *_args: False
    temporal_row = _pnp_batch([70]).by_ref[0]
    source_calls = []
    tracker.loc = SimpleNamespace(
        correspondences_for_sources=lambda _gray, images, xyz, **kwargs: (
            source_calls.append((images, xyz, kwargs)) or [temporal_row]
        )
    )
    full_batches = []
    fallback_attempt = replace(empty_attempt, pnp_ms=3.0)
    tracker._best_pose_attempt = lambda _selection, batch, **_kwargs: (
        full_batches.append(batch) or fallback_attempt
    )
    selection = _CandidateSelection(
        False,
        ["map"],
        "edm_temporal_map",
        50,
        0.0,
    )

    batch, attempt, _match_ms, fallback = tracker._match_track_map_first(
        np.zeros((576, 1024), np.uint8),
        selection,
        1.0,
        object(),
    )

    assert fallback
    assert len(source_calls) == 1
    assert source_calls[0][2]["source_kinds"] == ["temporal"]
    assert batch.temporal_used and batch.by_ref is None
    assert batch.per_ref == [70, 60]
    assert attempt.pnp_ms == pytest.approx(4.0)


def _pnp_batch(row_sizes: list[int]) -> _CorrespondenceBatch:
    rows = []
    for marker, size in enumerate(row_sizes):
        points2d = np.full((size, 2), marker, dtype=np.float32)
        points3d = np.full((size, 3), marker, dtype=np.float32)
        rows.append((points2d, points3d, np.ones(size, np.float32), size))
    return _CorrespondenceBatch(
        points2d=np.zeros((0, 2), np.float32),
        points3d=np.zeros((0, 3), np.float32),
        confidence=np.zeros(0, np.float32),
        per_ref=row_sizes,
        by_ref=rows,
        temporal_used=False,
    )


def test_pnp_early_stop_skips_rows_that_cannot_reach_the_inlier_gate() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(pnp_workers=2, pnp_early_stop=True)
    tracker.cam = SimpleNamespace(width=1280, height=720)
    tracker._pose_estimation_context = lambda: (object(), object())
    tracker._new_pose_estimation_context = lambda _seed=-1: (object(), object())
    tracker._estimate_pose_candidate = lambda *_args: pytest.fail(
        "a row with fewer correspondences than min_inliers cannot pass PnP"
    )
    selection = _CandidateSelection(
        acquiring=False,
        refs=["a", "b", "c"],
        mode="edm_map_scan",
        min_inliers=50,
        vpr_ms=0.0,
    )

    # Only the 120-corr ref is eligible; the two starved rows stay skipped
    # because a viable candidate exists.
    calls = []
    tracker._estimate_reference_row = lambda row: calls.append(len(row[1])) or None
    attempt = tracker._best_pose_attempt(selection, _pnp_batch([20, 39, 120]))

    assert attempt.result is None
    assert attempt.pnp_candidates == 1
    assert attempt.pnp_skipped == 2
    assert calls == [120]


def test_pnp_early_stop_rescues_best_two_when_every_ref_is_starved() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(pnp_workers=2, pnp_early_stop=True)
    tracker.cam = SimpleNamespace(width=1280, height=720)
    tracker._pose_estimation_context = lambda: (object(), object())
    tracker._new_pose_estimation_context = lambda _seed=-1: (object(), object())
    selection = _CandidateSelection(
        acquiring=False,
        refs=["a", "b"],
        mode="edm_map_scan",
        min_inliers=50,
        vpr_ms=0.0,
    )

    # Both refs are corr-starved: the two best still get a PnP chance instead
    # of forcing LOST. Downstream gates decide; here the mock yields nothing.
    calls = []
    tracker._estimate_reference_row = lambda row: calls.append(len(row[1])) or None
    attempt = tracker._best_pose_attempt(selection, _pnp_batch([20, 39]))

    assert attempt.result is None
    assert attempt.pnp_candidates == 2
    assert attempt.pnp_skipped == 0
    assert sorted(calls) == [20, 39]


def test_blend_velocity_trusts_new_sample_after_long_gaps() -> None:
    blend = tracker_module._blend_velocity
    old = np.array([1.0, 0.0, 0.0])
    measured = np.array([3.0, 0.0, 0.0])
    # 30 fps frame: mostly the old estimate (alpha = dt/tau ~ 0.13).
    slow = blend(old, measured, 1.0 / 30.0)
    assert slow[0] == pytest.approx(1.0 + (3.0 - 1.0) * (1.0 / 30.0 / 0.25))
    # 0.5 s gap: almost fully the new measurement.
    assert blend(old, measured, 0.5)[0] == pytest.approx(3.0)
    # No history or bad dt: fall back to the measurement.
    assert blend(None, measured, 0.1)[0] == pytest.approx(3.0)
    assert blend(old, measured, float("nan"))[0] == pytest.approx(3.0)


def test_parallel_pnp_preserves_candidate_order() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(pnp_workers=2, pnp_early_stop=False)
    tracker.cam = SimpleNamespace(width=1280, height=720)
    tracker._pose_estimation_context = lambda: (object(), object())
    tracker._new_pose_estimation_context = lambda _seed=-1: (object(), object())
    thread_ids = []

    def estimate(points2d, points3d, _confidence, _pcam, _options):
        marker = int(points2d[0, 0])
        thread_ids.append(threading.get_ident())
        time.sleep(0.02 if marker == 0 else 0.005)
        candidate = (
            {"num_inliers": len(points3d)},
            points2d,
            points3d,
            {"reproj_rms": 1.0, "inlier_ratio": 1.0, "inlier_grid_cells": 8},
        )
        return candidate, 1.0

    selected_order = []
    tracker._estimate_pose_candidate = estimate
    tracker._select_track_candidate = lambda _selection, scored: (
        selected_order.extend(name for name, _candidate in scored) or scored[0]
    )
    selection = _CandidateSelection(
        acquiring=False,
        refs=["first", "second", "third"],
        mode="edm_map_scan",
        min_inliers=50,
        vpr_ms=0.0,
    )

    attempt = tracker._best_pose_attempt(selection, _pnp_batch([80, 90, 100]))

    assert selected_order == ["first", "second", "third"]
    assert attempt.selected_ref == "first"
    assert len(set(thread_ids)) == 2


def test_pnp_pipeline_starts_before_the_next_edm_batch_finishes() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        pnp_workers=2,
        pnp_early_stop=False,
        pnp_pipeline=True,
    )
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    tracker.map = SimpleNamespace()
    first_started = threading.Event()
    release_first = threading.Event()
    rows = _pnp_batch([80, 90]).by_ref

    class FakeLocalizer:
        def correspondences_by_ref(
            self, _gray, _refs, *, batch_size, prepared_query, on_batch
        ):
            assert batch_size == tracker.cfg.match_batch_size
            on_batch(0, rows[:1])
            assert first_started.wait(0.5), "PnP did not overlap the next EDM batch"
            on_batch(1, rows[1:])
            release_first.set()
            return rows

    tracker.loc = FakeLocalizer()

    def estimate(row):
        marker = int(row[0][0, 0])
        if marker == 0:
            first_started.set()
            assert release_first.wait(0.5)
        return (
            {"num_inliers": len(row[1])},
            row[0],
            row[1],
            {"reproj_rms": 1.0, "inlier_ratio": 1.0, "inlier_grid_cells": 8},
        )

    tracker._estimate_reference_row = estimate
    tracker._select_track_candidate = lambda _selection, scored: scored[0]
    selection = _CandidateSelection(
        acquiring=False,
        refs=["first", "second"],
        mode="edm_map_scan",
        min_inliers=50,
        vpr_ms=0.0,
    )

    _batch, attempt, _match_ms = tracker._match_and_estimate(
        np.zeros((576, 1024), np.uint8),
        selection,
        prepared_query=object(),
    )

    assert attempt.selected_ref == "first"
    assert attempt.pnp_candidates == 2


def _jump_gate_tracker(
    monkeypatch,
    candidate_centers: list[float],
    candidate_inliers: list[int] | None = None,
):
    name = "ref0"
    camera = SimpleNamespace(
        model="PINHOLE",
        width=1280,
        height=720,
        params=[900.0, 900.0, 640.0, 360.0],
    )
    points3d = np.stack([
        np.linspace(-1.0, 1.0, 100),
        np.linspace(-0.5, 0.5, 100),
        np.linspace(4.0, 6.0, 100),
    ], axis=1)
    points2d = np.zeros((100, 2), dtype=float)

    class FakeLocalizer:
        scale = 1.25

        def correspondences(self, _gray, _refs, **_kwargs):
            return (
                points2d.copy(),
                points3d.copy(),
                np.ones(100, np.float32),
                [100],
            )

        def correspondences_by_ref(self, gray, refs, **kwargs):
            points2d_all, points3d_all, confidence, counts = self.correspondences(
                gray, refs, **kwargs)
            return [
                (points2d_all, points3d_all, confidence, counts[0] if counts else 0)
            ]

    candidate_poses = iter(
        pycolmap.Rigid3d(
            pycolmap.Rotation3d(),
            np.array([-center, 0.0, 0.0]),
        )
        for center in candidate_centers
    )
    inliers = iter(candidate_inliers or [100] * len(candidate_centers))
    monkeypatch.setattr(
        tracker_module.pycolmap,
        "estimate_and_refine_absolute_pose",
        lambda *_args, **_kwargs: {
            "cam_from_world": next(candidate_poses),
            "num_inliers": next(inliers),
            "inlier_mask": np.ones(100, dtype=bool),
        },
    )
    monkeypatch.setattr(
        tracker_module,
        "reprojection_metrics",
        lambda *_args, **_kwargs: {
            "reproj_rms": 0.0,
            "inlier_ratio": 1.0,
            "inlier_grid_cells": 16,
        },
    )
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(track_min_inliers=10, max_corr_total=100)
    tracker.st = RuntimeState(
        state="TRACK",
        center=np.zeros(3),
        yaw=0.0,
        last_refs=[0],
    )
    tracker.map = SimpleNamespace(
        ref_names=[name],
        images={name: np.zeros((576, 1024), np.uint8)},
        xyz_by_cell={name: np.zeros((128 * 72, 3), np.float32)},
        covis={},
    )
    tracker.cam = camera
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((1, 3), np.float32)
    tracker.yaws = np.zeros(1, np.float32)
    tracker.name_of = {0: name}
    tracker.idx_of = {name: 0}
    tracker.recovery_bank = [name]
    tracker.temporal_gray = None
    tracker.motion_validator = None
    tracker.motion_validation_mode = "off"
    tracker._last_accepted_bgr = None
    tracker._last_accepted_cam_from_world = None
    tracker.temporal_xyz_by_cell = None
    return tracker


def _add_map_reference(
    tracker,
    name: str,
    center: list[float],
    yaw: float = 0.0,
) -> None:
    """Grow the fake map so a reacquired pose has a reference to search against."""
    index = len(tracker.centers)
    tracker.centers = np.vstack([tracker.centers, np.asarray(center, dtype=np.float32)])
    tracker.yaws = np.append(tracker.yaws, np.float32(yaw))
    tracker.name_of[index] = name
    tracker.idx_of[name] = index
    tracker.map.ref_names = list(tracker.map.ref_names) + [name]
    tracker.map.images[name] = np.zeros((576, 1024), np.uint8)
    tracker.map.xyz_by_cell[name] = np.zeros((128 * 72, 3), np.float32)


def test_limited_jump_requires_a_second_consistent_pose(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.05])

    first = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=1.0,
    )
    second = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=1.1,
    )

    assert not first["ok"]
    assert first["rejected"] == "limited_jump_unconfirmed"
    assert second["ok"]
    assert second["limited_jump_confirmed"]
    assert np.allclose(second["center"], [0.05, 0.0, 0.0])


def test_limited_jump_confirms_constant_motion_using_capture_time(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.10])
    tracker.st.last_capture_stamp = 0.9

    first = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=1.0,
    )
    second = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=1.1,
    )

    assert not first["ok"]
    assert first["rejected"] == "limited_jump_unconfirmed"
    assert second["ok"]
    assert second["limited_jump_confirmed"]
    assert second["limited_jump"]["confirmation_model"] == "constant_velocity"
    assert second["limited_jump"]["confirmation_residual"] == pytest.approx(0.0)
    assert second["limited_jump"]["capture_dt"] == pytest.approx(0.2)
    assert np.allclose(second["center"], [0.10, 0.0, 0.0])



def test_limited_jump_hold_does_not_degrade_track(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.05])

    first = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=1.0,
    )

    assert not first["ok"]
    assert first["rejected"] == "limited_jump_unconfirmed"
    assert tracker.st.state == "TRACK"
    assert tracker.st.misses == 0


def test_limited_jump_accepts_velocity_prior(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05])
    tracker.st.last_capture_stamp = 0.9
    tracker.st.velocity = np.array([0.5, 0.0, 0.0], dtype=float)

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=1.0,
    )

    assert result["ok"]
    assert result["limited_jump_confirmed"]
    assert result["limited_jump"]["confirmation_model"] == "accepted_velocity_prior"
    assert result["limited_jump"]["confirmation_residual"] == pytest.approx(0.0)
    assert np.allclose(result["center"], [0.05, 0.0, 0.0])


def test_jump_time_history_uses_every_observed_submission(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.10])

    tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)
    tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.05)

    assert list(tracker.st.observed_capture_dts) == pytest.approx([0.05])


def test_quality_rejection_invalidates_pending_jump_candidate(monkeypatch) -> None:
    tracker = _jump_gate_tracker(
        monkeypatch,
        [0.05, 0.06, 0.10],
        candidate_inliers=[100, 0, 100],
    )

    first = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )
    middle = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.05
    )
    third = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.10
    )

    assert first["rejected"] == "limited_jump_unconfirmed"
    assert not middle["ok"]
    assert third["rejected"] == "limited_jump_unconfirmed"
    assert third["limited_jump"]["confirmation_model"] is None


def test_lost_recovery_does_not_train_continuous_step_history(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [1.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )

    assert result["ok"]
    assert not tracker.st.accepted_step_norms


def test_long_capture_gap_expands_only_the_adaptive_jump_gate(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.02])
    tracker.st.last_capture_stamp = 1.0
    tracker.st.accepted_step_norms.extend([0.001] * 20)
    tracker.st.observed_capture_dts.extend([0.05] * 20)

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=1.2,
    )

    assert result["ok"]
    assert result.get("limited_jump") is None
    assert np.allclose(result["center"], [0.02, 0.0, 0.0])


def test_long_capture_gap_never_weakens_the_hard_jump_gate(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [3.0])
    tracker.st.last_capture_stamp = 1.0
    tracker.st.accepted_step_norms.extend([0.001] * 20)
    tracker.st.observed_capture_dts.extend([0.05] * 20)

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8),
        capture_stamp=2.0,
    )

    assert not result["ok"]
    assert result["rejected"] == "jump"


@pytest.mark.parametrize("pixel_offset", [0.0, 10.0])
def test_boot_staging_always_evaluates_the_complete_retrieved_set(
    monkeypatch, pixel_offset: float,
) -> None:
    names = [f"ref{i}" for i in range(10)]
    camera = SimpleNamespace(
        model="PINHOLE", width=1280, height=720,
        params=[900.0, 900.0, 640.0, 360.0],
    )
    pycamera = pycolmap.Camera(
        model=camera.model, width=camera.width, height=camera.height,
        params=camera.params,
    )
    points3d = np.stack([
        np.linspace(-1.0, 1.0, 100),
        np.linspace(-0.5, 0.5, 100),
        np.linspace(4.0, 6.0, 100),
    ], axis=1)
    points2d = pycamera.img_from_cam(points3d) + np.array([pixel_offset, 0.0])
    calls = []

    class FakeLocalizer:
        scale = 1.25

        def retrieve(self, _rgb, _topk):
            return names

        def correspondences_by_ref(self, _gray, refs, **_kwargs):
            calls.append(list(refs))
            return [
                (
                    points2d.copy(),
                    points3d.copy(),
                    np.ones(100, np.float32),
                    100,
                )
                for _name in refs
            ]

    def estimate(_points2d, _points3d, _camera, _options):
        return {
            "cam_from_world": pycolmap.Rigid3d(),
            "num_inliers": len(_points3d),
            "inlier_mask": np.ones(len(_points3d), dtype=bool),
        }

    monkeypatch.setattr(tracker_module.pycolmap, "estimate_and_refine_absolute_pose", estimate)
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        boot_global_topk=10, acquire_initial_topk=2, acquire_min_inliers=80,
        max_corr_total=100, min_inlier_grid_cells=1,
    )
    tracker.st = RuntimeState()
    tracker.map = SimpleNamespace(
        ref_names=names,
        images={name: np.zeros((576, 1024), np.uint8) for name in names},
        xyz_by_cell={name: np.zeros((128 * 72, 3), np.float32) for name in names},
        covis={},
    )
    tracker.cam = camera
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((10, 3), np.float32)
    tracker.yaws = np.zeros(10, np.float32)
    tracker.name_of = dict(enumerate(names))
    tracker.idx_of = {name: index for index, name in enumerate(names)}
    tracker.recovery_bank = names
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None

    info = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert info["requested_reference_count"] == 10
    if pixel_offset == 0.0:
        assert info["ok"]
        assert not info["staged_early_stop"]
        assert info["refs"] == names
        assert calls == [names]
        assert tracker.temporal_gray is None
        assert tracker.temporal_xyz_by_cell is None
    else:
        assert not info["ok"]
        assert not info["staged_early_stop"]
        assert info["rejected"] == "reprojection"
        assert info["reproj_rms"] == pytest.approx(pixel_offset)
        assert calls == [names]


@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        ({"reproj_rms": 0.0, "inlier_ratio": 0.05, "inlier_grid_cells": 16},
         "inlier_ratio"),
        ({"reproj_rms": 0.0, "inlier_ratio": 1.0, "inlier_grid_cells": 2},
         "inlier_spread"),
    ],
)
def test_pose_quality_rejects_weak_or_spatially_concentrated_inliers(
    monkeypatch, metrics: dict, reason: str,
) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    monkeypatch.setattr(
        tracker_module,
        "reprojection_metrics",
        lambda *_args, **_kwargs: dict(metrics),
    )

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )

    assert not result["ok"]
    assert result["rejected"] == reason


# ---------------------------------------------------------------------------
# Adapter -> flight-loop WEAK gate wiring


def test_adapter_reports_degraded_state_fixes_as_weak_for_the_flight_gate() -> None:
    """path_follow_flight's SFM_GATE_WEAK hover gate reads last_info["weak"].

    On success the tracker always reports state_out="TRACK", so state_in is the only
    record that a fix was won from a degraded state: WEAK_TRACK accepts at
    weak_min_inliers (30, below the loop's LOW_CONF_INLIERS of 60) and LOST
    re-acquisition skips the trajectory-jump gate entirely. Without this key the
    documented WEAK gate silently never fires on the EDM backend.
    """
    import edm_localizer_adapter as adapter_module

    cases = {
        "TRACK": False,
        "BOOT_INIT": False,
        "WEAK_TRACK": True,
        "LOST": True,
    }
    for state_in, expected_weak in cases.items():
        adapter = object.__new__(adapter_module.EDMTrackerAdapter)
        adapter.state = adapter_module.RuntimeState()
        adapter.state.mode = state_in
        adapter._last_info = {}
        info = {
            "state_in": state_in,
            "state_out": "TRACK",
            "ok": True,
            "center": np.zeros(3, float),
            "yaw": 0.0,
            "R": np.eye(3),
            "inliers": 120,
        }
        adapter.trk = SimpleNamespace(
            st=RuntimeState(state=state_in),
            localize=lambda _bgr, capture_stamp=None, _i=info: _i,
        )
        pose = adapter.localize_frame(
            np.zeros((8, 8, 3), np.uint8), capture_stamp=1.0
        )
        assert pose is not None
        assert adapter.last_info["weak"] is expected_weak, (
            f"state_in={state_in!r} -> weak={adapter.last_info['weak']!r}, "
            f"expected {expected_weak!r}"
        )


def test_edm_adapter_publishes_heading_from_measured_map_frame() -> None:
    import edm_localizer_adapter as adapter_module

    adapter = object.__new__(adapter_module.EDMTrackerAdapter)
    adapter.state = adapter_module.RuntimeState()
    adapter._last_info = {}
    observed = []
    adapter.map_frame = SimpleNamespace(
        heading=lambda forward: observed.append(np.asarray(forward, float)) or 1.25
    )
    adapter.trk = SimpleNamespace(
        st=RuntimeState(state="TRACK"),
        localize=lambda _bgr, capture_stamp=None: {
            "state_in": "TRACK",
            "state_out": "TRACK",
            "ok": True,
            "center": np.zeros(3, float),
            "yaw": -0.5,
            "R": np.eye(3),
            "inliers": 120,
        },
    )

    pose = adapter.localize_frame(np.zeros((8, 8, 3), np.uint8), capture_stamp=1.0)

    assert pose.yaw == pytest.approx(1.25)
    assert np.allclose(observed[0], [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# LOST re-acquisition bound (was: unbounded teleport + no yaw sanity gate)


def test_lost_reacquisition_rejects_a_teleport_beyond_the_acquire_bound(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [100.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]
    tracker.st.last_capture_stamp = 0.95          # prior still fresh

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )

    assert not result["ok"]
    assert result["rejected"] == "acquire_jump"
    assert result["acquire_limit"] == pytest.approx(
        tracker.cfg.acquire_max_jump_factor * tracker.cfg.max_jump
    )


def test_lost_reacquisition_within_the_bound_is_still_accepted(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [1.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]
    tracker.st.last_capture_stamp = 0.95

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )

    assert result["ok"]
    assert result.get("rejected") is None


def test_lost_reacquisition_is_unbounded_once_the_prior_expires(monkeypatch) -> None:
    """A genuinely long LOST episode must still be able to relocalize anywhere."""
    tracker = _jump_gate_tracker(monkeypatch, [100.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]
    tracker.st.last_capture_stamp = 1.0 - (tracker.cfg.lost_prior_max_age_s + 1.0)

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )

    assert result["ok"]
    assert result.get("rejected") is None


def test_stale_lost_reacquisition_requires_two_consistent_captures(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [100.0, 100.1])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.cfg.stale_reacquire_confirmations = 2
    tracker.cfg.stale_reacquire_max_distance = 0.3
    tracker.cfg.stale_reacquire_max_yaw_diff_deg = 30.0
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]
    tracker.st.last_capture_stamp = 0.0
    _add_map_reference(tracker, "ref1", [100.0, 0.0, 0.0])

    first = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=4.0
    )
    second = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=4.1
    )

    assert not first["ok"]
    assert first["rejected"] == "stale_reacquire_unconfirmed"
    assert second["ok"]
    assert second["stale_reacquire_confirmed"]


def test_stale_lost_reacquisition_rejects_inconsistent_second_fix(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [100.0, 101.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.cfg.stale_reacquire_confirmations = 2
    tracker.cfg.stale_reacquire_max_distance = 0.3
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]
    tracker.st.last_capture_stamp = 0.0
    _add_map_reference(tracker, "ref1", [100.0, 0.0, 0.0])

    tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=4.0)
    second = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=4.1
    )

    assert not second["ok"]
    assert second["rejected"] == "stale_reacquire_unconfirmed"
    assert second["stale_reacquire_confirmation"]["distance"] == pytest.approx(1.0)


def test_stale_lost_reacquisition_anchors_the_next_search_at_the_pending_pose(monkeypatch) -> None:
    """P119: an unconfirmed reacquire must not send the next frame back to the stale refs."""
    tracker = _jump_gate_tracker(monkeypatch, [100.0, 100.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.cfg.stale_reacquire_confirmations = 2
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref0"]
    tracker.st.last_capture_stamp = 0.0
    _add_map_reference(tracker, "ref1", [100.0, 0.0, 0.0])
    # ref0 keeps the stale accepted heading; the reacquired heading is 0 degrees.
    tracker.st.yaw = 120.0
    tracker.yaws[0] = 120.0

    predicted: list[np.ndarray] = []
    searched: list[list[str]] = []
    predict_center = tracker._predict_center
    track_candidates = tracker._track_candidates

    def _record_prediction(stamp=None):
        center = predict_center(stamp)
        predicted.append(np.array(center, dtype=float))
        return center

    def _record_candidates(topk, stamp=None, **kwargs):
        refs = track_candidates(topk, stamp, **kwargs)
        searched.append(list(refs))
        return refs

    tracker._predict_center = _record_prediction
    tracker._track_candidates = _record_candidates

    first = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=4.0)

    assert first["rejected"] == "stale_reacquire_unconfirmed"
    # Unconfirmed: the fix is parked, never accepted as the tracked pose.
    assert np.allclose(tracker.st.center, [0.0, 0.0, 0.0])
    assert np.allclose(tracker.st.pending_reacquire_center, [100.0, 0.0, 0.0])
    assert np.allclose(predicted[-1], [0.0, 0.0, 0.0])

    second = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=4.1)

    assert np.allclose(predicted[-1], [100.0, 0.0, 0.0])
    assert searched[-1] == ["ref1"]
    assert second["ok"]
    assert second["stale_reacquire_confirmed"]


def test_lost_reacquisition_rejects_an_impossible_heading_flip(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]
    tracker.st.last_capture_stamp = 0.95
    tracker.st.yaw = 0.0

    # Camera centre stays put (0.1 u) but the heading flips by 180 degrees.
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    monkeypatch.setattr(
        tracker_module.pycolmap,
        "estimate_and_refine_absolute_pose",
        lambda *_a, **_k: {
            "cam_from_world": pycolmap.Rigid3d(
                pycolmap.Rotation3d(rotation), np.array([0.0, 0.0, 0.1])
            ),
            "num_inliers": 100,
            "inlier_mask": np.ones(100, dtype=bool),
        },
    )

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )

    assert not result["ok"]
    assert result["rejected"] == "acquire_yaw"
    assert result["acquire_yaw_delta_deg"] == pytest.approx(180.0, abs=1e-6)


def test_track_rejects_capture_time_scaled_yaw_spike(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.track_yaw_slack_deg = 8.0
    tracker.cfg.track_max_yaw_rate_deg_s = 120.0
    tracker.cfg.track_max_yaw_step_deg = 30.0
    tracker.st.last_capture_stamp = 0.95
    rotation = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
    )
    monkeypatch.setattr(
        tracker_module.pycolmap,
        "estimate_and_refine_absolute_pose",
        lambda *_a, **_k: {
            "cam_from_world": pycolmap.Rigid3d(
                pycolmap.Rotation3d(rotation), np.zeros(3)
            ),
            "num_inliers": 100,
            "inlier_mask": np.ones(100, dtype=bool),
        },
    )

    result = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )

    assert not result["ok"]
    assert result["rejected"] == "track_yaw"
    assert result["track_yaw_limit_deg"] == pytest.approx(14.0)


def test_limited_jump_cannot_be_confirmed_by_relocalizing_the_same_capture(
        monkeypatch) -> None:
    """The two-frame confirmation must need two DIFFERENT captures.

    Re-running the same frame reproduces the same centre, which would score a
    residual of ~0 against itself and promote one bad frame to an accepted relocation.
    """
    tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.05])

    first = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0
    )
    second = tracker.localize(
        np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0   # SAME capture
    )

    assert first["rejected"] == "limited_jump_unconfirmed"
    assert not second["ok"]
    assert second["rejected"] == "limited_jump_unconfirmed"
    assert second["limited_jump"]["confirmation_independent"] is False


class _FakeMotionValidator:
    def __init__(self, status: str = "agree", reason: str = "within_thresholds"):
        self.status = status
        self.reason = reason
        self.calls: list[tuple] = []

    def check(self, previous_bgr, current_bgr, previous_cam_from_world, candidate_cam_from_world):
        from reposed_motion_validator import RelativeMotionCheck

        self.calls.append(
            (previous_bgr, current_bgr, previous_cam_from_world, candidate_cam_from_world)
        )
        return RelativeMotionCheck(
            status=self.status,
            reason=self.reason,
            matches=80,
            inliers=60,
            rotation_delta_deg=0.1,
            translation_direction_delta_deg=1.0,
            depth_ms=1.0,
            match_ms=1.0,
            solver_ms=1.0,
            total_ms=3.0,
        )


def _seed_visual_cache(tracker) -> None:
    tracker._last_accepted_bgr = np.zeros((720, 1280, 3), np.uint8)
    tracker._last_accepted_cam_from_world = np.eye(4, dtype=float)


def test_normal_pose_does_not_invoke_motion_validator(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.motion_validator = _FakeMotionValidator()
    tracker.motion_validation_mode = "confirm_limited_jump"
    _seed_visual_cache(tracker)

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert result["ok"]
    assert tracker.motion_validator.calls == []


def test_hard_jump_does_not_invoke_motion_validator(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [3.0])
    tracker.motion_validator = _FakeMotionValidator()
    tracker.motion_validation_mode = "confirm_limited_jump"
    _seed_visual_cache(tracker)

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert result["rejected"] == "jump"
    assert tracker.motion_validator.calls == []


def test_quality_failure_does_not_invoke_motion_validator(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05], candidate_inliers=[0])
    tracker.motion_validator = _FakeMotionValidator()
    tracker.motion_validation_mode = "confirm_limited_jump"
    _seed_visual_cache(tracker)

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert not result["ok"]
    assert tracker.motion_validator.calls == []

def test_lost_reacquisition_does_not_invoke_motion_validator(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [1.0])
    tracker.cfg.global_retrieval_policy = "boot_once"
    tracker.st.state = "LOST"
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref"]
    tracker.st.last_capture_stamp = 0.95
    tracker.motion_validator = _FakeMotionValidator()
    tracker.motion_validation_mode = "confirm_limited_jump"
    _seed_visual_cache(tracker)

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert result["ok"]

    assert tracker.motion_validator.calls == []


def test_pending_limited_jump_does_not_invoke_motion_validator(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.05])
    tracker.motion_validator = _FakeMotionValidator()
    tracker.motion_validation_mode = "confirm_limited_jump"
    _seed_visual_cache(tracker)
    tracker.st.pending_limited_center = np.array([0.04, 0.0, 0.0], dtype=float)
    tracker.st.pending_limited_stamp = 0.9
    tracker.st.pending_limited_limit = 0.02

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert tracker.motion_validator.calls == []
    assert result["limited_jump"]["confirmation_model"] in {"stationary", "constant_velocity"}


def test_shadow_mode_cannot_fast_confirm_first_limited_jump(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05])
    tracker.motion_validator = _FakeMotionValidator("agree")
    tracker.motion_validation_mode = "shadow"
    _seed_visual_cache(tracker)

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert not result["ok"]
    assert result["rejected"] == "limited_jump_unconfirmed"
    assert result["relative_motion_check"]["status"] == "agree"
    assert not result.get("limited_jump_confirmed")


def test_off_mode_keeps_first_limited_jump_unconfirmed(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05])
    tracker.motion_validator = _FakeMotionValidator("agree")
    tracker.motion_validation_mode = "off"
    _seed_visual_cache(tracker)

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert result["rejected"] == "limited_jump_unconfirmed"
    assert tracker.motion_validator.calls == []


def test_confirm_mode_agree_fast_confirms_first_adaptive_jump(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05])
    tracker.motion_validator = _FakeMotionValidator("agree")
    tracker.motion_validation_mode = "confirm_limited_jump"
    _seed_visual_cache(tracker)

    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert result["ok"]
    assert result["limited_jump_confirmed"] is True
    assert result["limited_jump"]["confirmation_model"] == "reposed_relative_pose"
    assert len(tracker.motion_validator.calls) == 1


@pytest.mark.parametrize("status", ["disagree", "unavailable"])
def test_confirm_mode_non_agree_keeps_second_frame_path(monkeypatch, status: str) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.05, 0.05])
    tracker.motion_validator = _FakeMotionValidator(status, "exceeds_thresholds")
    tracker.motion_validation_mode = "confirm_limited_jump"
    _seed_visual_cache(tracker)

    first = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)
    second = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.1)

    assert first["rejected"] == "limited_jump_unconfirmed"
    assert first["relative_motion_check"]["status"] == status
    assert second["ok"]
    assert second["limited_jump_confirmed"]
    assert second["limited_jump"]["confirmation_model"] in {"stationary", "constant_velocity"}
    assert len(tracker.motion_validator.calls) == 1


def test_lost_entry_clears_visual_motion_cache(monkeypatch) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    _seed_visual_cache(tracker)
    tracker.st.state = "WEAK_TRACK"
    tracker.st.misses = tracker.cfg.weak_after + tracker.cfg.lost_after - 1

    tracker._on_miss({"pose_status": "NONE"})

    assert tracker.st.state == "LOST"
    assert tracker._last_accepted_bgr is None
    assert tracker._last_accepted_cam_from_world is None


def test_adapter_reset_clears_visual_motion_cache(monkeypatch) -> None:
    import edm_localizer_adapter as adapter_module

    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    _seed_visual_cache(tracker)
    adapter = object.__new__(adapter_module.EDMTrackerAdapter)
    adapter.state = adapter_module.RuntimeState()
    adapter.trk = tracker
    adapter.cfg = tracker.cfg

    adapter._clear_tracking_history()

    assert tracker._last_accepted_bgr is None
    assert tracker._last_accepted_cam_from_world is None


def test_track_picks_the_better_single_reference_pose(monkeypatch) -> None:
    good, bad = "ref_good", "ref_bad"
    camera = SimpleNamespace(
        model="PINHOLE", width=1280, height=720,
        params=np.array([900.0, 900.0, 640.0, 360.0]),
    )
    pycamera = pycolmap.Camera(
        model=camera.model, width=camera.width, height=camera.height,
        params=camera.params,
    )
    good_xyz = np.stack([
        np.linspace(-1.0, 1.0, 80),
        np.linspace(-0.5, 0.5, 80),
        np.linspace(4.0, 6.0, 80),
    ], axis=1)
    bad_xyz = good_xyz[:20].copy()
    good_xy = np.asarray(pycamera.img_from_cam(good_xyz), dtype=float)
    bad_xy = np.asarray(pycamera.img_from_cam(bad_xyz), dtype=float)

    class FakeLocalizer:
        scale = 1.0

        def correspondences_by_ref(self, _gray, refs, **_kwargs):
            rows = {
                good: (good_xy.copy(), good_xyz.copy(), np.ones(80, np.float32), 80),
                bad: (bad_xy.copy(), bad_xyz.copy(), np.ones(20, np.float32), 20),
            }
            return [rows[name] for name in refs]

    monkeypatch.setattr(
        tracker_module.pycolmap,
        "estimate_and_refine_absolute_pose",
        lambda points2d, points3d, _camera, _options: {
            "cam_from_world": pycolmap.Rigid3d(),
            "num_inliers": len(points3d),
            "inlier_mask": np.ones(len(points3d), dtype=bool),
        },
    )
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        local_topk=2, track_min_inliers=10, max_corr_total=100,
        min_inlier_grid_cells=1, min_inlier_ratio=0.01,
    )
    tracker.st = RuntimeState(
        state="TRACK",
        last_refs=[0, 1],
        center=np.zeros(3),
        last_capture_stamp=1.0,
        global_retrieval_calls=1,
        boot_refs=[good, bad],
    )
    tracker.map = SimpleNamespace(
        ref_names=[good, bad],
        images={name: np.zeros((576, 1024), np.uint8) for name in (good, bad)},
        xyz_by_cell={name: np.zeros((128 * 72, 3), np.float32) for name in (good, bad)},
        covis={},
    )
    tracker.cam = camera
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((2, 3), np.float32)
    tracker.yaws = np.zeros(2, np.float32)
    tracker.name_of = {0: good, 1: bad}
    tracker.idx_of = {good: 0, bad: 1}
    tracker.recovery_bank = [good, bad]
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    tracker.motion_validator = None
    tracker.motion_validation_mode = "off"
    tracker._last_accepted_bgr = None
    tracker._last_accepted_cam_from_world = None

    info = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert info["ok"]
    assert info["candidate_mode"] == "edm_track"
    assert info["selected_ref"] == good
    assert info["inliers"] == 80
    assert tracker.st.last_refs == [0]


def _acquire_candidate(center: list[float], inliers: int, reproj: float = 1.0):
    translation = -np.asarray(center, dtype=float)
    estimate = {
        "cam_from_world": pycolmap.Rigid3d(pycolmap.Rotation3d(), translation),
        "num_inliers": inliers,
        "inlier_mask": np.ones(inliers, dtype=bool),
    }
    metrics = {
        "reproj_rms": reproj,
        "inlier_ratio": 1.0,
        "inlier_grid_cells": 8,
    }
    return (
        estimate,
        np.zeros((inliers, 2)),
        np.zeros((inliers, 3)),
        metrics,
    )


def test_acquire_keeps_strong_megaloc_top1() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    selection = _CandidateSelection(True, ["top", "other"], "megaloc_boot", 80, 0.0)
    chosen = tracker._select_acquire_candidate(
        selection,
        [
            ("top", _acquire_candidate([0.0, 0.0, 0.0], 90)),
            ("other", _acquire_candidate([0.1, 0.0, 0.0], 200)),
        ],
    )
    assert chosen is not None
    assert chosen[0] == "top"


def test_acquire_reranks_when_megaloc_top1_is_weak() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    selection = _CandidateSelection(True, ["top", "other"], "megaloc_lost", 80, 0.0)
    chosen = tracker._select_acquire_candidate(
        selection,
        [
            ("top", _acquire_candidate([0.0, 0.0, 0.0], 20)),
            ("other", _acquire_candidate([0.1, 0.0, 0.0], 90)),
        ],
    )
    assert chosen is not None
    assert chosen[0] == "other"


def test_acquire_vetoes_disagreeing_strong_references() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    selection = _CandidateSelection(True, ["a", "b"], "megaloc_lost", 80, 0.0)
    chosen = tracker._select_acquire_candidate(
        selection,
        [
            ("a", _acquire_candidate([0.0, 0.0, 0.0], 90)),
            ("b", _acquire_candidate([5.0, 0.0, 0.0], 90)),
        ],
    )
    assert chosen is None
    assert tracker._acquire_consensus_limit() == pytest.approx(4.0)


def test_track_keeps_best_when_strong_refs_agree() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    selection = _CandidateSelection(False, ["weak", "strong"], "edm_track", 50, 0.0)
    chosen = tracker._select_track_candidate(
        selection,
        [
            ("weak", _acquire_candidate([0.0, 0.0, 0.0], 20)),
            ("strong", _acquire_candidate([0.1, 0.0, 0.0], 80)),
        ],
    )
    assert chosen is not None
    assert chosen[0] == "strong"


def test_track_vetoes_disagreeing_strong_references() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    selection = _CandidateSelection(False, ["a", "b"], "edm_track", 50, 0.0)
    chosen = tracker._select_track_candidate(
        selection,
        [
            ("a", _acquire_candidate([0.0, 0.0, 0.0], 80)),
            ("b", _acquire_candidate([5.0, 0.0, 0.0], 80)),
        ],
    )
    assert chosen is None


def test_new_runtime_knobs_default_to_current_behavior_and_reject_invalid() -> None:
    cfg = EDMConfig()
    assert cfg.acquire_stage_mode == "full_set"
    assert cfg.lost_prior_strategy == "restrict_nearby"
    assert cfg.lost_prior_fusion_weight == 1.0
    assert cfg.pose_consensus_mode == "pairwise"
    assert cfg.consensus_max_rotation_deg == 90.0
    assert cfg.reference_quality_weight == 0.5
    assert cfg.reference_quality_floor == 0.0
    with pytest.raises(ValueError, match="acquire_stage_mode"):
        EDMConfig(acquire_stage_mode="skip_gates")
    with pytest.raises(ValueError, match="lost_prior_strategy"):
        EDMConfig(lost_prior_strategy="teleport")
    with pytest.raises(ValueError, match="lost_prior_fusion_weight"):
        EDMConfig(lost_prior_fusion_weight=0.0)
    with pytest.raises(ValueError, match="pose_consensus_mode"):
        EDMConfig(pose_consensus_mode="best_effort")
    with pytest.raises(ValueError, match="consensus_max_rotation_deg"):
        EDMConfig(consensus_max_rotation_deg=0.0)
    with pytest.raises(ValueError, match="reference_quality_weight"):
        EDMConfig(reference_quality_weight=-0.1)


def test_reference_grid_sigma_keeps_d01_when_d10_score_is_higher() -> None:
    torch = edm_matcher_module.torch
    fine = SimpleNamespace(
        local_resolution=8.0,
        bi_directional_refine=True,
        mconf_thr=0.2,
        sigma_thr=1e-6,
        border_rm=0.0,
    )
    data = {
        "pred_coord": torch.tensor([[0.1, 0.0], [0.4, 0.0]], dtype=torch.float32),
        "pred_score": torch.tensor([0.3, 0.95], dtype=torch.float32),
        "hw0_i": (576, 1024),
        "hw1_i": (576, 1024),
        "mkpts0_c": torch.tensor([[64.0, 64.0]]),
        "mkpts1_c": torch.tensor([[80.0, 64.0]]),
        "mconf": torch.tensor([0.8]),
        "b_ids": torch.tensor([0]),
    }
    select_reference_grid_matches(fine, data)
    assert data["mkpts0_f"].shape[0] == 1
    assert torch.allclose(data["mkpts0_f"][0], torch.tensor([64.0, 64.0]))
    assert torch.allclose(data["mkpts1_f"][0], torch.tensor([80.8, 64.0]))

    data["mconf"] = torch.tensor([float("nan")])
    data["pred_coord"] = torch.tensor([[0.1, 0.0], [0.4, 0.0]], dtype=torch.float32)
    data["pred_score"] = torch.tensor([0.3, 0.95], dtype=torch.float32)
    data["mkpts0_c"] = torch.tensor([[64.0, 64.0]])
    data["mkpts1_c"] = torch.tensor([[80.0, 64.0]])
    data["b_ids"] = torch.tensor([0])
    select_reference_grid_matches(fine, data)
    assert data["mkpts0_f"].shape[0] == 0


def test_temporal_feature_cache_is_isolated_from_map_lru() -> None:
    matcher = object.__new__(EDMMatcher)
    matcher.reference_feature_cache_size = 1
    matcher.temporal_feature_cache_size = 1
    matcher._feature_caches = {"map": OrderedDict(), "temporal": OrderedDict()}
    matcher._feature_cache_capacity = {"map": 1, "temporal": 1}
    matcher._feature_cache_stats = {
        "map": {"hits": 0, "misses": 0, "evictions": 0},
        "temporal": {"hits": 0, "misses": 0, "evictions": 0},
    }
    map_src = object()
    temporal_src = object()
    matcher._store_reference_features(map_src, ("map-features",), "map")
    matcher._store_reference_features(temporal_src, ("temporal-features",), "temporal")
    extra_temporal = object()
    matcher._store_reference_features(extra_temporal, ("temporal-2",), "temporal")
    assert id(map_src) in matcher._feature_caches["map"]
    assert id(temporal_src) not in matcher._feature_caches["temporal"]
    assert id(extra_temporal) in matcher._feature_caches["temporal"]
    stats = matcher.feature_cache_stats_by_class()
    assert stats["map"] == {
        "hits": 0, "misses": 0, "evictions": 0, "size": 1, "capacity": 1,
    }
    assert stats["temporal"]["evictions"] == 1
    assert stats["temporal"]["size"] == 1


def test_matcher_cache_budget_stays_inside_8_gib() -> None:
    _validate_matcher_cache_budget(
        map_feature_entries=32,
        temporal_feature_entries=2,
        map_tensor_entries=32,
        temporal_tensor_entries=2,
    )
    with pytest.raises(ValueError, match="8 GiB"):
        _validate_matcher_cache_budget(
            map_feature_entries=2000,
            temporal_feature_entries=0,
            map_tensor_entries=0,
            temporal_tensor_entries=0,
        )


def test_acquire_stage_early_stops_only_after_consensus_and_gates(
    monkeypatch,
) -> None:
    names = [f"ref{i}" for i in range(10)]
    camera = SimpleNamespace(
        model="PINHOLE", width=1280, height=720,
        params=[900.0, 900.0, 640.0, 360.0],
    )
    pycamera = pycolmap.Camera(
        model=camera.model, width=camera.width, height=camera.height,
        params=camera.params,
    )
    points3d = np.stack([
        np.linspace(-1.0, 1.0, 100),
        np.linspace(-0.5, 0.5, 100),
        np.linspace(4.0, 6.0, 100),
    ], axis=1)
    points2d = pycamera.img_from_cam(points3d)
    calls = []

    class FakeLocalizer:
        scale = 1.25

        def retrieve(self, _rgb, _topk):
            return names

        def correspondences_by_ref(self, _gray, refs, **_kwargs):
            calls.append(list(refs))
            return [
                (
                    np.asarray(points2d, dtype=float).copy(),
                    points3d.copy(),
                    np.ones(100, np.float32),
                    100,
                )
                for _name in refs
            ]

    monkeypatch.setattr(
        tracker_module.pycolmap,
        "estimate_and_refine_absolute_pose",
        lambda points2d, points3d, _camera, _options: {
            "cam_from_world": pycolmap.Rigid3d(),
            "num_inliers": len(points3d),
            "inlier_mask": np.ones(len(points3d), dtype=bool),
        },
    )
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        boot_global_topk=10,
        acquire_initial_topk=2,
        acquire_min_inliers=80,
        max_corr_total=100,
        min_inlier_grid_cells=1,
        acquire_stage_mode="initial_topk",
    )
    tracker.st = RuntimeState()
    tracker.map = SimpleNamespace(
        ref_names=names,
        images={name: np.zeros((576, 1024), np.uint8) for name in names},
        xyz_by_cell={name: np.zeros((128 * 72, 3), np.float32) for name in names},
        covis={},
    )
    tracker.cam = camera
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((10, 3), np.float32)
    tracker.yaws = np.zeros(10, np.float32)
    tracker.name_of = dict(enumerate(names))
    tracker.idx_of = {name: index for index, name in enumerate(names)}
    tracker.recovery_bank = names
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    tracker.motion_validator = None
    tracker.motion_validation_mode = "off"

    info = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert info["ok"]
    assert info["staged_early_stop"]
    assert info["requested_reference_count"] == 10
    assert info["acquire_stage_mode"] == "initial_topk"
    assert calls == [names[:2]]


def test_acquire_stage_falls_back_to_full_set_when_quality_fails(
    monkeypatch,
) -> None:
    names = [f"ref{i}" for i in range(4)]
    camera = SimpleNamespace(
        model="PINHOLE", width=1280, height=720,
        params=[900.0, 900.0, 640.0, 360.0],
    )
    pycamera = pycolmap.Camera(
        model=camera.model, width=camera.width, height=camera.height,
        params=camera.params,
    )
    points3d = np.stack([
        np.linspace(-1.0, 1.0, 100),
        np.linspace(-0.5, 0.5, 100),
        np.linspace(4.0, 6.0, 100),
    ], axis=1)
    points2d = pycamera.img_from_cam(points3d) + np.array([10.0, 0.0])
    calls = []

    class FakeLocalizer:
        scale = 1.25

        def retrieve(self, _rgb, _topk):
            return names

        def correspondences_by_ref(self, _gray, refs, **_kwargs):
            calls.append(list(refs))
            return [
                (
                    np.asarray(points2d, dtype=float).copy(),
                    points3d.copy(),
                    np.ones(100, np.float32),
                    100,
                )
                for _name in refs
            ]

    monkeypatch.setattr(
        tracker_module.pycolmap,
        "estimate_and_refine_absolute_pose",
        lambda points2d, points3d, _camera, _options: {
            "cam_from_world": pycolmap.Rigid3d(),
            "num_inliers": len(points3d),
            "inlier_mask": np.ones(len(points3d), dtype=bool),
        },
    )
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        boot_global_topk=4,
        acquire_initial_topk=2,
        acquire_min_inliers=80,
        max_corr_total=100,
        min_inlier_grid_cells=1,
        acquire_stage_mode="initial_topk",
    )
    tracker.st = RuntimeState()
    tracker.map = SimpleNamespace(
        ref_names=names,
        images={name: np.zeros((576, 1024), np.uint8) for name in names},
        xyz_by_cell={name: np.zeros((128 * 72, 3), np.float32) for name in names},
        covis={},
    )
    tracker.cam = camera
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((4, 3), np.float32)
    tracker.yaws = np.zeros(4, np.float32)
    tracker.name_of = dict(enumerate(names))
    tracker.idx_of = {name: index for index, name in enumerate(names)}
    tracker.recovery_bank = names
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    tracker.motion_validator = None
    tracker.motion_validation_mode = "off"

    info = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)

    assert not info["ok"]
    assert not info["staged_early_stop"]
    assert info["rejected"] == "reprojection"
    # The failed first stage is reused; only the remaining references are matched.
    assert calls == [names[:2], names[2:]]


def test_acquire_stage_can_stop_single_strong_fast_path(monkeypatch) -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        acquire_min_inliers=80,
        max_reproj_error_acquire=5.0,
        min_inlier_ratio=0.2,
        min_inlier_grid_cells=1,
    )
    tracker.st = RuntimeState()

    selection = _CandidateSelection(
        acquiring=True,
        refs=["ref0", "ref1"],
        mode="megaloc_boot",
        min_inliers=80,
        vpr_ms=0.0,
    )

    def make_attempt(
        inliers: int,
        reproj_rms: float,
        selected_ref: str = "ref0",
        strong_count: int = 1,
    ) -> "tracker_module._PoseAttempt":
        result = {
            "num_inliers": inliers,
            "cam_from_world": pycolmap.Rigid3d(),
        }
        metrics = {
            "reproj_rms": reproj_rms,
            "inlier_ratio": 0.5,
            "inlier_grid_cells": 4,
        }
        return tracker_module._PoseAttempt(
            result=result,
            points2d=np.zeros((inliers, 2)),
            points3d=np.zeros((inliers, 3)),
            metrics=metrics,
            selected_ref=selected_ref,
            pnp_ms=1.0,
            strong_count=strong_count,
        )

    # Super-strong single candidate: 120 inliers, 1.5px reproj stops early
    attempt_120_1p5 = make_attempt(120, 1.5, selected_ref="ref0", strong_count=1)
    assert tracker._acquire_stage_can_stop(selection, attempt_120_1p5, capture_stamp=1.0) is True

    # Super-strong single candidate: 150 inliers, 1.0px reproj stops early
    attempt_150_1p0 = make_attempt(150, 1.0, selected_ref="ref0", strong_count=1)
    assert tracker._acquire_stage_can_stop(selection, attempt_150_1p0, capture_stamp=1.0) is True

    # 119 inliers (< 120 threshold) does not stop
    attempt_119 = make_attempt(119, 1.5, selected_ref="ref0", strong_count=1)
    assert tracker._acquire_stage_can_stop(selection, attempt_119, capture_stamp=1.0) is False

    # Reproj 2.1 (> 2.0 threshold) does not stop
    attempt_2p1 = make_attempt(120, 2.1, selected_ref="ref0", strong_count=1)
    assert tracker._acquire_stage_can_stop(selection, attempt_2p1, capture_stamp=1.0) is False

    # Reproj 2.0 (exact threshold) stops
    attempt_2p0 = make_attempt(120, 2.0, selected_ref="ref0", strong_count=1)
    assert tracker._acquire_stage_can_stop(selection, attempt_2p0, capture_stamp=1.0) is True

    # Non-Top-1 candidate (ref1) does not stop even with strong numbers
    attempt_ref1 = make_attempt(120, 1.5, selected_ref="ref1", strong_count=1)
    assert tracker._acquire_stage_can_stop(selection, attempt_ref1, capture_stamp=1.0) is False

    # When acquire_min_inliers is higher (e.g. 100 -> max(150, 120) = 150):
    tracker_100 = object.__new__(ProductionEDMTracker)
    tracker_100.cfg = EDMConfig(
        acquire_min_inliers=100,
        max_reproj_error_acquire=5.0,
        min_inlier_ratio=0.2,
        min_inlier_grid_cells=1,
    )
    tracker_100.st = RuntimeState()
    selection_100 = replace(selection, min_inliers=100)
    attempt_140 = make_attempt(140, 1.5, selected_ref="ref0", strong_count=1)
    assert tracker_100._acquire_stage_can_stop(selection_100, attempt_140, capture_stamp=1.0) is False
    attempt_150 = make_attempt(150, 1.5, selected_ref="ref0", strong_count=1)
    assert tracker_100._acquire_stage_can_stop(selection_100, attempt_150, capture_stamp=1.0) is True

    # Env switch SFM_EDM_SINGLE_STRONG_EARLY_STOP=0 restores old 2-consensus requirement
    monkeypatch.setenv("SFM_EDM_SINGLE_STRONG_EARLY_STOP", "0")
    assert tracker._acquire_stage_can_stop(selection, attempt_120_1p5, capture_stamp=1.0) is False
    # Under env=0, 2-consensus still succeeds
    attempt_2cons = make_attempt(85, 3.5, selected_ref="ref0", strong_count=2)
    assert tracker._acquire_stage_can_stop(selection, attempt_2cons, capture_stamp=1.0) is True


def test_acquire_stage_single_strong_skips_stages_and_obeys_env(monkeypatch) -> None:
    names = [f"ref{i}" for i in range(4)]
    camera = SimpleNamespace(
        model="PINHOLE", width=1280, height=720,
        params=[900.0, 900.0, 640.0, 360.0],
    )
    pycamera = pycolmap.Camera(
        model=camera.model, width=camera.width, height=camera.height,
        params=camera.params,
    )
    points3d = np.stack([
        np.linspace(-1.0, 1.0, 120),
        np.linspace(-0.5, 0.5, 120),
        np.linspace(4.0, 6.0, 120),
    ], axis=1)
    points2d = pycamera.img_from_cam(points3d)
    calls = []

    class FakeLocalizer:
        scale = 1.25

        def retrieve(self, _rgb, _topk):
            return names

        def correspondences_by_ref(self, _gray, refs, **_kwargs):
            calls.append(list(refs))
            rows = []
            for name in refs:
                if name == "ref0":
                    rows.append((
                        np.asarray(points2d, dtype=float).copy(),
                        points3d.copy(),
                        np.ones(120, np.float32),
                        120,
                    ))
                else:
                    rows.append((
                        np.zeros((10, 2), dtype=float),
                        np.zeros((10, 3), dtype=float),
                        np.ones(10, np.float32),
                        10,
                    ))
            return rows

    monkeypatch.setattr(
        tracker_module.pycolmap,
        "estimate_and_refine_absolute_pose",
        lambda p2, p3, _camera, _options: {
            "cam_from_world": pycolmap.Rigid3d(),
            "num_inliers": len(p3),
            "inlier_mask": np.ones(len(p3), dtype=bool),
        },
    )
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        boot_global_topk=4,
        acquire_initial_topk=2,
        acquire_min_inliers=80,
        max_corr_total=200,
        min_inlier_grid_cells=1,
        acquire_stage_mode="initial_topk",
    )
    tracker.st = RuntimeState()
    tracker.map = SimpleNamespace(
        ref_names=names,
        images={name: np.zeros((576, 1024), np.uint8) for name in names},
        xyz_by_cell={name: np.zeros((128 * 72, 3), np.float32) for name in names},
        covis={},
    )
    tracker.cam = camera
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((4, 3), np.float32)
    tracker.yaws = np.zeros(4, np.float32)
    tracker.name_of = dict(enumerate(names))
    tracker.idx_of = {name: index for index, name in enumerate(names)}
    tracker.recovery_bank = names
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    tracker.motion_validator = None
    tracker.motion_validation_mode = "off"

    # Default env: ref0 has 120 inliers (strong_count=1), stops at stage 1
    info = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)
    assert info["ok"]
    assert info["staged_early_stop"]
    assert calls == [names[:2]]

    # Env SFM_EDM_SINGLE_STRONG_EARLY_STOP=0: strong_count=1 cannot early stop, falls back to remaining refs
    calls.clear()
    monkeypatch.setenv("SFM_EDM_SINGLE_STRONG_EARLY_STOP", "0")
    tracker.st = RuntimeState()
    info = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)
    assert calls == [names[:2], names[2:]]

def test_full_global_lost_prior_does_not_restrict_megaloc(monkeypatch) -> None:
    import cv2

    monkeypatch.setattr(cv2, "cvtColor", lambda frame, _code: frame)
    calls = []
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(lost_prior_strategy="full_global")
    tracker.st = RuntimeState(state="LOST", last_capture_stamp=1.0)
    tracker.loc = SimpleNamespace(
        retrieve=lambda _rgb, _k, candidates=None: calls.append(candidates) or ["ref"]
    )
    tracker._track_candidates = lambda *_args, **_kwargs: ["near"]

    refs, _ms, nearby = tracker._global_retrieval(
        np.zeros((8, 8, 3), np.uint8), 1.1
    )

    assert refs == ["ref"]
    assert nearby is False
    assert calls == [None]
    assert tracker._last_retrieval_kind == "global"


def test_score_fusion_lost_prior_ranks_geometry_with_vpr(monkeypatch) -> None:
    import cv2

    monkeypatch.setattr(cv2, "cvtColor", lambda frame, _code: frame)
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(lost_prior_strategy="score_fusion")
    tracker.st = RuntimeState(
        state="LOST",
        last_capture_stamp=1.0,
        center=np.zeros(3, dtype=float),
    )
    tracker.map = SimpleNamespace(ref_names=["near", "far"])
    tracker.centers = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    tracker.idx_of = {"near": 0, "far": 1}
    tracker._predict_center = lambda _stamp=None: tracker.st.center
    tracker._track_candidates = lambda *_args, **_kwargs: ["near"]
    tracker.loc = SimpleNamespace(
        megaloc=SimpleNamespace(extract_one=lambda _rgb: np.ones(2, np.float32)),
        retrieve_scored=lambda _rgb, _k, candidates=None, descriptor=None: [
            ("far", 0.95),
            ("near", 0.70),
        ],
    )

    refs, _ms, nearby = tracker._global_retrieval(
        np.zeros((8, 8, 3), np.uint8), 1.1
    )

    assert nearby is False
    assert tracker._last_retrieval_kind == "fused"
    assert refs[0] == "near"


def test_accepted_frame_skips_bgr_cache_when_motion_validation_is_off(
    monkeypatch,
) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)
    assert result["ok"]
    assert result["motion_cache_active"] is False
    assert tracker._last_accepted_bgr is None
    assert tracker._last_accepted_cam_from_world is None


def test_accepted_frame_stores_bgr_cache_when_motion_validation_is_active(
    monkeypatch,
) -> None:
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.motion_validator = _FakeMotionValidator()
    tracker.motion_validation_mode = "shadow"
    result = tracker.localize(np.zeros((720, 1280, 3), np.uint8), capture_stamp=1.0)
    assert result["ok"]
    assert result["motion_cache_active"] is True
    assert tracker._last_accepted_bgr is not None
    assert tracker._last_accepted_cam_from_world is not None


def test_megaloc_engine_rejects_oversized_plan(tmp_path: Path, monkeypatch) -> None:
    engine_path = tmp_path / "megaloc.engine"
    engine_path.write_bytes(b"x" * 64)
    monkeypatch.setattr(
        edm_localizer_module, "MEGALOC_TENSORRT_MAX_ENGINE_BYTES", 16
    )
    with pytest.raises(ValueError, match="exceeds"):
        _validate_megaloc_engine_file(engine_path)


def test_megaloc_engine_stream_never_slurps_plan_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    trt = sys.modules.get("tensorrt")
    if trt is None or not hasattr(trt, "IStreamReaderV2"):
        class _IStreamReaderV2:
            def __init__(self) -> None:
                pass

        trt = SimpleNamespace(IStreamReaderV2=_IStreamReaderV2)
        monkeypatch.setitem(sys.modules, "tensorrt", trt)
    import tensorrt as trt

    engine_path = tmp_path / "megaloc.engine"
    payload = b"engine-bytes-0123456789abcd"
    engine_path.write_bytes(payload)
    slurps = []

    def forbidden(self):
        slurps.append(str(self))
        raise AssertionError("TensorRT loading must not slurp the plan")

    monkeypatch.setattr(Path, "read_bytes", forbidden)

    class FakeRuntime:
        def deserialize_cuda_engine(self, stream):
            chunks = []
            while True:
                data = stream.read(8, 0)
                if not data:
                    break
                chunks.append(data)
            assert b"".join(chunks) == payload
            return "ok"

    result = deserialize_megaloc_cuda_engine(FakeRuntime(), engine_path, trt)
    assert result == "ok"
    assert slurps == []


def test_megaloc_tensorrt_device_and_sm_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        edm_localizer_module.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(
        edm_localizer_module.torch.cuda, "device_count", lambda: 1
    )
    with pytest.raises(ValueError, match="out of range"):
        _resolve_cuda_device("cuda:3")
    monkeypatch.setattr(
        edm_localizer_module.torch.cuda,
        "get_device_capability",
        lambda _index: (7, 0),
    )
    with pytest.raises(ValueError, match="compute capability"):
        _validate_megaloc_tensorrt_compatibility(
            0, SimpleNamespace(__version__="11.2.1")
        )



def test_reference_stability_reaches_ranking_when_weight_enabled() -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(reference_quality_weight=2.0, reference_quality_floor=0.0)
    tracker._reference_quality = np.array([0.1, 0.9], dtype=np.float32)
    tracker._reference_stability = np.array([0.1, 1.0], dtype=np.float32)
    tracker.st = RuntimeState(yaw=None)
    tracker.yaws = None
    distances = np.array([0.10, 0.11], dtype=float)
    last: set[int] = set()
    near = tracker._track_candidate_score(0, distances, last)
    far = tracker._track_candidate_score(1, distances, last)
    assert far > near
    tracker.cfg = EDMConfig(reference_quality_weight=0.0)
    assert tracker._track_candidate_score(0, distances, last) > tracker._track_candidate_score(
        1, distances, last
    )


def test_profile_optional_knobs_apply_to_config_and_reposed(tmp_path: Path) -> None:
    import json

    from edm_profile import apply_edm_tracker_profile, load_edm_production_profile
    from production_localizer_factory import build_reposed_motion_validator

    official = Path(
        "/home/allen/localization/定位演算法/configs/edm_production_profile.json"
    )
    raw = json.loads(official.read_text(encoding="utf-8"))
    raw["tracker"].update(
        {
            "pose_consensus_mode": "cluster",
            "consensus_max_rotation_deg": 15.0,
            "reference_quality_weight": 0.5,
            "reference_quality_floor": 0.1,
            "acquire_stage_mode": "initial_topk",
            "lost_prior_strategy": "score_fusion",
            "lost_prior_fusion_weight": 2.0,
        }
    )
    raw["matcher"]["runtime_sigma_mode"] = "bidirectional"
    raw["matcher"]["temporal_feature_cache_size"] = 3
    model = tmp_path / "model.pt"
    payload = b"fixture-weights"
    model.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    raw["reposed"] = {
        "mode": "off",
        "model_path": str(model),
        "model_sha256": digest,
        "num_tokens": 1200,
        "max_matches": 1200,
        "min_inliers": 30,
        "max_rotation_delta_deg": 3.0,
        "max_translation_direction_delta_deg": 15.0,
        "match_grid": 4,
        "min_inlier_ratio": 0.2,
        "min_spatial_support": 3,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    profile = load_edm_production_profile(path)
    cfg = EDMConfig()
    apply_edm_tracker_profile(cfg, profile)
    assert cfg.pose_consensus_mode == "cluster"
    assert cfg.consensus_max_rotation_deg == 15.0
    assert cfg.reference_quality_weight == 0.5
    assert cfg.reference_quality_floor == 0.1
    assert cfg.acquire_stage_mode == "initial_topk"
    assert cfg.lost_prior_strategy == "score_fusion"
    assert cfg.lost_prior_fusion_weight == 2.0
    camera = SimpleNamespace(
        model="PINHOLE", width=1280, height=720, params=[900.0, 900.0, 640.0, 360.0]
    )
    raw["reposed"]["mode"] = "shadow"
    validator, mode = build_reposed_motion_validator(raw, camera, object(), source=model)
    assert mode == "shadow"
    assert validator.match_grid == 4
    assert validator.min_inlier_ratio == pytest.approx(0.2)
    assert validator.min_spatial_support == 3


def test_megaloc_plan_identity_is_artifact_specific_and_fail_closed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        edm_localizer_module.torch.cuda, "get_device_capability", lambda _index: (12, 0)
    )
    monkeypatch.setattr(
        edm_localizer_module.torch.cuda,
        "get_device_name",
        lambda _index: "NVIDIA GeForce RTX 5060 Laptop GPU",
    )
    assert normalize_gpu_identity("  NVIDIA   GeForce RTX 5060 Laptop GPU ") == (
        MEGALOC_TENSORRT_GPU_IDENTITY
    )
    capability = _validate_megaloc_tensorrt_compatibility(
        0, SimpleNamespace(__version__="11.2.1")
    )
    assert capability == (12, 0)
    with pytest.raises(ValueError, match="sanctioned MegaLoc plan"):
        _validate_megaloc_tensorrt_compatibility(
            0, SimpleNamespace(__version__="10.0.1")
        )
    monkeypatch.setattr(
        edm_localizer_module.torch.cuda,
        "get_device_name",
        lambda _index: "NVIDIA GeForce RTX 4090",
    )
    with pytest.raises(ValueError, match="GPU identity"):
        _validate_megaloc_tensorrt_compatibility(
            0, SimpleNamespace(__version__="11.2.1")
        )
    engine_path = tmp_path / "megaloc.engine"
    engine_path.write_bytes(b"x" * 64)
    with pytest.raises(ValueError, match="sanctioned"):
        _validate_megaloc_plan_size(engine_path)



# ---------------------------------------------------------------------------
# Flight-disjoint LOO helpers (eval_edm_loo.py)


def _eval_edm_loo():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "EDM工具包" / "tests" / "eval_edm_loo.py"
    spec = importlib.util.spec_from_file_location("eval_edm_loo_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flight_group_keeps_p116_p120_folds_separate_for_flat_and_nested_names() -> None:
    loo = _eval_edm_loo()
    nested = {
        "P1160116/0001.jpg": "P116",
        "site/P1170117/frame_0002.jpg": "P117",
        "P118/img.jpg": "P118",
        r"maps\P1190119\0033.jpg": "P119",
        "river/P1200120/frame.jpg": "P120",
    }
    flat = {
        "P1160116_0001.jpg": "P116",
        "P117_0002.jpg": "P117",
        "P1180118.jpg": "P118",
        "P1190119_frame0033.jpg": "P119",
        "P1200120_9.jpg": "P120",
    }
    for name, flight in {**nested, **flat}.items():
        assert loo.flight_group(name) == flight
    groups = loo.group_refs_by_flight(list(nested) + list(flat))
    assert sorted(groups) == ["P116", "P117", "P118", "P119", "P120"]
    assert loo.flight_group("P1570157/0001.jpg") == "P157"
    assert loo.flight_group("P1670167_0001.jpg") == "P167"


def test_same_flight_candidates_are_dropped_before_megaloc_ranking() -> None:
    loo = _eval_edm_loo()
    names = [
        "P1160116/q.jpg",
        "P1160116/near.jpg",
        "P1170117/other.jpg",
        "P1180118/far.jpg",
    ]
    scores = np.array([0.99, 0.95, 0.10, 0.05], dtype=float)
    masked = loo.mask_retrieval_scores(
        scores, names, 0, exclude_same_flight=True
    )
    assert not np.isfinite(masked[0])
    assert not np.isfinite(masked[1])
    refs = loo.retrieve_topk(masked, names, topk=2)
    assert refs == ["P1170117/other.jpg", "P1180118/far.jpg"]
    assert all(loo.flight_group(name) != "P116" for name in refs)


def test_standard_loo_only_excludes_the_query_reference() -> None:
    loo = _eval_edm_loo()
    names = [
        "P1160116/q.jpg",
        "P1160116/near.jpg",
        "P1170117/other.jpg",
    ]
    scores = np.array([0.99, 0.95, 0.10], dtype=float)
    masked = loo.mask_retrieval_scores(
        scores, names, 0, exclude_same_flight=False
    )
    assert not np.isfinite(masked[0])
    assert masked[1] == pytest.approx(0.95)
    assert loo.retrieve_topk(masked, names, topk=1) == ["P1160116/near.jpg"]
    assert loo.STANDARD_LOO_ACCURACY_SCOPE == (
        "leave-one-reference-out proxy against map poses"
    )
    assert "exclude the query flight" in loo.FLIGHT_DISJOINT_ACCURACY_SCOPE


def test_flight_disjoint_pool_fails_closed_without_two_flights_or_candidates() -> None:
    loo = _eval_edm_loo()
    with pytest.raises(ValueError, match="at least two flights"):
        loo.require_flight_disjoint_pool(["P1160116/a.jpg", "P1160116/b.jpg"])
    names = ["P1160116/q.jpg", "P1170117/other.jpg"]
    loo.require_flight_disjoint_pool(names)
    empty = loo.mask_retrieval_scores(
        np.array([0.9, 0.1], dtype=float), names, 0, exclude_same_flight=True
    )
    empty[1] = -np.inf
    with pytest.raises(ValueError, match="no candidates"):
        loo.retrieve_topk(empty, names, topk=3)


def test_flight_fold_report_exposes_macro_and_worst_fold_metrics() -> None:
    loo = _eval_edm_loo()
    folds = {
        "P116": loo.fold_report(4, 4, [0.01, 0.02], [0.1, 0.2], [80, 90], [10.0, 12.0]),
        "P117": loo.fold_report(4, 2, [0.05, 0.08], [0.4, 0.6], [60, 70], [20.0, 22.0]),
        "P118": loo.fold_report(2, 2, [0.03], [0.2], [75], [15.0]),
        "P119": loo.fold_report(4, 1, [0.04], [0.3], [55], [18.0]),
        "P120": loo.fold_report(2, 2, [0.02, 0.03], [0.15, 0.25], [85, 88], [11.0, 13.0]),
    }
    assert list(folds) == ["P116", "P117", "P118", "P119", "P120"]
    for fold in folds.values():
        assert set(fold) >= {
            "queries", "localized", "success_rate", "inliers",
            "position_map_units", "rotation_degrees", "match_pnp_ms",
        }
        for key in ("inliers", "position_map_units", "rotation_degrees", "match_pnp_ms"):
            assert set(fold[key]) == {"median", "p90", "max"}
    summary = loo.macro_and_worst(folds)
    assert summary["macro_success"] == pytest.approx(np.mean([
        folds[flight]["success_rate"] for flight in folds
    ]))
    assert summary["worst_fold"] == "P119"
    assert summary["worst_fold_success"] == pytest.approx(0.25)
    assert summary["worst_fold_error"]["position_map_units"]["median"] == pytest.approx(0.04)
    assert summary["worst_fold_error"]["rotation_degrees"]["median"] == pytest.approx(0.3)


def test_pnp_early_stop_relaxed_eligibility_boundary(monkeypatch) -> None:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(pnp_workers=2, pnp_early_stop=True)
    tracker.cam = SimpleNamespace(width=1280, height=720)
    tracker._pose_estimation_context = lambda: (object(), object())
    tracker._new_pose_estimation_context = lambda _seed=-1: (object(), object())

    # Case 1: min_inliers = 50 -> threshold is max(10, ceil(0.8*50)) = 40.
    # 40 is eligible, 39 is skipped.
    selection = _CandidateSelection(
        acquiring=False,
        refs=["a", "b"],
        mode="edm_map_scan",
        min_inliers=50,
        vpr_ms=0.0,
    )
    calls = []
    tracker._estimate_reference_row = lambda row: calls.append(len(row[1])) or None
    attempt = tracker._best_pose_attempt(selection, _pnp_batch([39, 40]))
    assert attempt.pnp_candidates == 1
    assert attempt.pnp_skipped == 1
    assert calls == [40]

    # Case 2: min_inliers = 10 -> threshold is max(10, ceil(0.8*10)) = 10.
    # 10 is eligible, 9 is skipped.
    selection_10 = _CandidateSelection(
        acquiring=False,
        refs=["a", "b"],
        mode="edm_map_scan",
        min_inliers=10,
        vpr_ms=0.0,
    )
    calls.clear()
    attempt = tracker._best_pose_attempt(selection_10, _pnp_batch([9, 10]))
    assert attempt.pnp_candidates == 1
    assert attempt.pnp_skipped == 1
    assert calls == [10]

    # Case 3: min_inliers = 8 (small value) -> threshold is max(10, ceil(0.8*8)) = 10.
    # Floor of 10 applies: 10 is eligible, 9 is skipped.
    selection_8 = _CandidateSelection(
        acquiring=False,
        refs=["a", "b"],
        mode="edm_map_scan",
        min_inliers=8,
        vpr_ms=0.0,
    )
    calls.clear()
    attempt = tracker._best_pose_attempt(selection_8, _pnp_batch([9, 10]))
    assert attempt.pnp_candidates == 1
    assert attempt.pnp_skipped == 1
    assert calls == [10]

    # Case 4: env SFM_EDM_PNP_ELIGIBLE_RELAX=0 restores legacy min_inliers threshold.
    # With min_inliers=50, 40 is now skipped and only >=50 is eligible.
    monkeypatch.setenv("SFM_EDM_PNP_ELIGIBLE_RELAX", "0")
    calls.clear()
    attempt = tracker._best_pose_attempt(selection, _pnp_batch([40, 50]))
    assert attempt.pnp_candidates == 1
    assert attempt.pnp_skipped == 1
    assert calls == [50]


def test_weak_hysteresis_single_frame_near_miss_buffered(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_WEAK_HYSTERESIS", "1")
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.weak_after = 1
    tracker.cfg.track_min_inliers = 50
    tracker.st.state = "TRACK"
    tracker.st.misses = 0

    # 49 inliers in [45, 50), reproj 1.5 <= 2.5, step is legal -> buffered, stays TRACK
    info = {
        "pose_status": "NONE",
        "inliers": 49,
        "reproj_rms": 1.5,
        "step_legal": True,
    }
    tracker._on_miss(info)

    assert tracker.st.state == "TRACK"
    assert tracker.st.misses == 0.5
    assert info.get("weak_hysteresis_buffered") is True
    assert info.get("state_out") == "TRACK"


def test_weak_hysteresis_two_consecutive_near_misses_downgrades(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_WEAK_HYSTERESIS", "1")
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.weak_after = 1
    tracker.cfg.track_min_inliers = 50
    tracker.st.state = "TRACK"
    tracker.st.misses = 0

    # Frame 1: buffered, stays TRACK
    info1 = {
        "pose_status": "NONE",
        "inliers": 49,
        "reproj_rms": 1.5,
        "step_legal": True,
    }
    tracker._on_miss(info1)
    assert tracker.st.state == "TRACK"
    assert tracker.st.misses == 0.5

    # Frame 2: second near-miss pushes misses to 1.0 >= weak_after -> downgrades to WEAK_TRACK
    info2 = {
        "pose_status": "NONE",
        "inliers": 49,
        "reproj_rms": 1.5,
        "step_legal": True,
    }
    tracker._on_miss(info2)
    assert tracker.st.state == "WEAK_TRACK"
    assert tracker.st.misses == 1.0
    assert info2.get("state_out") == "WEAK_TRACK"


def test_weak_hysteresis_reproj_exceeded_downgrades_immediately(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_WEAK_HYSTERESIS", "1")
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.weak_after = 1
    tracker.cfg.track_min_inliers = 50
    tracker.st.state = "TRACK"
    tracker.st.misses = 0

    # 49 inliers but reproj 3.0 > 2.5 -> not buffered, drops to WEAK_TRACK immediately
    info = {
        "pose_status": "NONE",
        "inliers": 49,
        "reproj_rms": 3.0,
        "step_legal": True,
    }
    tracker._on_miss(info)
    assert tracker.st.state == "WEAK_TRACK"
    assert tracker.st.misses == 1.0
    assert not info.get("weak_hysteresis_buffered")
    assert info.get("state_out") == "WEAK_TRACK"


def test_weak_hysteresis_step_illegal_downgrades_immediately(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_WEAK_HYSTERESIS", "1")
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.weak_after = 1
    tracker.cfg.track_min_inliers = 50
    tracker.st.state = "TRACK"
    tracker.st.misses = 0

    # Step illegal (step_legal=False) -> drops to WEAK_TRACK immediately
    info = {
        "pose_status": "NONE",
        "inliers": 49,
        "reproj_rms": 1.5,
        "step_legal": False,
    }
    tracker._on_miss(info)
    assert tracker.st.state == "WEAK_TRACK"
    assert tracker.st.misses == 1.0

    # Also test rejected="jump"
    tracker.st.state = "TRACK"
    tracker.st.misses = 0
    info_jump = {
        "pose_status": "NONE",
        "inliers": 49,
        "reproj_rms": 1.5,
        "rejected": "jump",
    }
    tracker._on_miss(info_jump)
    assert tracker.st.state == "WEAK_TRACK"
    assert tracker.st.misses == 1.0


def test_weak_hysteresis_below_soft_interval_downgrades_immediately(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_WEAK_HYSTERESIS", "1")
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.weak_after = 1
    tracker.cfg.track_min_inliers = 50
    tracker.st.state = "TRACK"
    tracker.st.misses = 0

    # 44 inliers < 45 -> below soft interval, drops to WEAK_TRACK immediately
    info = {
        "pose_status": "NONE",
        "inliers": 44,
        "reproj_rms": 1.5,
        "step_legal": True,
    }
    tracker._on_miss(info)
    assert tracker.st.state == "WEAK_TRACK"
    assert tracker.st.misses == 1.0
    assert not info.get("weak_hysteresis_buffered")
    assert info.get("state_out") == "WEAK_TRACK"


def test_weak_hysteresis_env_disabled_restores_legacy_behavior(monkeypatch) -> None:
    # Default/env 0: hysteresis off
    monkeypatch.setenv("SFM_EDM_WEAK_HYSTERESIS", "0")
    tracker = _jump_gate_tracker(monkeypatch, [0.0])
    tracker.cfg.weak_after = 1
    tracker.cfg.track_min_inliers = 50
    tracker.st.state = "TRACK"
    tracker.st.misses = 0

    info = {
        "pose_status": "NONE",
        "inliers": 49,
        "reproj_rms": 1.5,
        "step_legal": True,
    }
    tracker._on_miss(info)
    assert tracker.st.state == "WEAK_TRACK"
    assert tracker.st.misses == 1.0
    assert not info.get("weak_hysteresis_buffered")
    assert info.get("state_out") == "WEAK_TRACK"


def test_adaptive_vpr_blur_gate_boundary(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_ADAPTIVE_VPR", "1")
    tracker = _lost_retrieval_policy_tracker(interval=3)
    tracker.st.lost_frames = 6
    tracker.st.lost_global_retrieval_attempts = 1
    tracker._last_vpr_lost_frame = 3

    # Blurry frame (var == 0.0 < _VPR_BLUR_VAR_MIN): skips global retrieval retry
    blurry = np.full((100, 100), 128, dtype=np.uint8)
    assert not tracker._should_run_global_retrieval(blurry)

    # Sharp frame (var >> _VPR_BLUR_VAR_MIN): does not skip
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[::2, ::2] = 255
    assert tracker._should_run_global_retrieval(sharp)

    # Disabled env: blur gate does not trigger
    monkeypatch.setenv("SFM_EDM_ADAPTIVE_VPR", "0")
    tracker.st.lost_frames = 5
    assert tracker._should_run_global_retrieval(blurry)


def test_adaptive_vpr_backoff_and_reset(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_ADAPTIVE_VPR", "1")
    tracker = _lost_retrieval_policy_tracker(interval=3)
    tracker.cfg.radius = 0.8
    tracker.cfg.max_jump = 2.0

    # Initial state: 0 failures -> interval=3, radius=1.0x
    assert tracker._vpr_consec_failures == 0
    assert tracker._effective_lost_global_retrieval_interval() == 3
    assert tracker._lost_local_radius_factor() == 1.0

    # 1 failure: interval=3, radius=1.0x
    tracker._vpr_consec_failures = 1
    assert tracker._effective_lost_global_retrieval_interval() == 3
    assert tracker._lost_local_radius_factor() == 1.0

    # 2 consecutive failures: interval=5, radius=2.0x (capped at max_jump)
    tracker._vpr_consec_failures = 2
    assert tracker._effective_lost_global_retrieval_interval() == 5
    assert tracker._lost_local_radius_factor() == 2.0

    # 3 consecutive failures: interval=8, radius=2.0x
    tracker._vpr_consec_failures = 3
    assert tracker._effective_lost_global_retrieval_interval() == 8
    assert tracker._lost_local_radius_factor() == 2.0

    # Success reset: resets back to 0 -> interval=3, radius=1.0x
    tracker._vpr_consec_failures = 0
    assert tracker._effective_lost_global_retrieval_interval() == 3
    assert tracker._lost_local_radius_factor() == 1.0


def test_adaptive_vpr_env_disabled_keeps_fixed_interval(monkeypatch) -> None:
    monkeypatch.setenv("SFM_EDM_ADAPTIVE_VPR", "0")
    tracker = _lost_retrieval_policy_tracker(interval=3)
    tracker.cfg.radius = 0.8
    tracker.cfg.max_jump = 2.0
    tracker._vpr_consec_failures = 5

    # Even with 5 failures, env=0 keeps fixed interval=3 and radius=1.0x
    assert tracker._effective_lost_global_retrieval_interval() == 3
    assert tracker._lost_local_radius_factor() == 1.0


def test_prior_pnp_refine_fast_path_selection(monkeypatch) -> None:
    import pycolmap
    from types import SimpleNamespace

    camera = pycolmap.Camera(model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240])
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cam = camera
    tracker.cfg = SimpleNamespace(max_corr_total=900, corr_grid=8, pnp_ransac_max_error=5.0)
    tracker.st = RuntimeState(state="TRACK")

    rng = np.random.RandomState(42)
    pts3D = rng.uniform(-2, 2, size=(100, 3))
    pts3D[:, 2] += 5.0
    pts2D = camera.img_from_cam(pts3D)
    conf = np.ones(100, dtype=float)
    prior = pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([0.01, -0.01, 0.01]))

    refine_calls = []
    orig_refine = pycolmap.refine_absolute_pose
    def spy_refine(*args, **kwargs):
        refine_calls.append(True)
        return orig_refine(*args, **kwargs)

    ransac_calls = []
    orig_ransac = pycolmap.estimate_and_refine_absolute_pose
    def spy_ransac(*args, **kwargs):
        ransac_calls.append(True)
        return orig_ransac(*args, **kwargs)

    monkeypatch.setattr(pycolmap, "refine_absolute_pose", spy_refine)
    monkeypatch.setattr(pycolmap, "estimate_and_refine_absolute_pose", spy_ransac)

    # 1. Qualified (inliers=100 >= 60, ratio=1.0 >= 0.65) with env ON -> calls refine, skips RANSAC
    monkeypatch.setenv("SFM_EDM_PRIOR_PNP_REFINE", "1")
    _cam, options = tracker._new_pose_estimation_context()
    cand, _elapsed = tracker._estimate_pose_candidate(pts2D, pts3D, conf, camera, options, prior=prior)
    assert cand is not None
    assert len(refine_calls) == 1
    assert len(ransac_calls) == 0

    # 2. Unqualified (inliers=40 < 60) -> skips refine, calls full RANSAC
    refine_calls.clear()
    ransac_calls.clear()
    cand, _elapsed = tracker._estimate_pose_candidate(pts2D[:40], pts3D[:40], conf[:40], camera, options, prior=prior)
    assert cand is not None
    assert len(refine_calls) == 0
    assert len(ransac_calls) == 1

    # 3. Refine fails (returns None) -> falls back to full RANSAC
    monkeypatch.setattr(pycolmap, "refine_absolute_pose", lambda *args, **kwargs: None)
    refine_calls.clear()
    ransac_calls.clear()
    cand, _elapsed = tracker._estimate_pose_candidate(pts2D, pts3D, conf, camera, options, prior=prior)
    assert cand is not None
    assert len(ransac_calls) == 1

    # 4. Env OFF -> skips refine even if qualified, runs full RANSAC
    monkeypatch.setattr(pycolmap, "refine_absolute_pose", spy_refine)
    monkeypatch.setenv("SFM_EDM_PRIOR_PNP_REFINE", "0")
    refine_calls.clear()
    ransac_calls.clear()
    cand, _elapsed = tracker._estimate_pose_candidate(pts2D, pts3D, conf, camera, options, prior=prior)
    assert cand is not None
    assert len(refine_calls) == 0
    assert len(ransac_calls) == 1


def test_pnp_threads_env_passthrough(monkeypatch) -> None:
    import pycolmap
    from types import SimpleNamespace

    tracker = object.__new__(ProductionEDMTracker)
    tracker.cam = pycolmap.Camera(model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240])
    tracker.cfg = SimpleNamespace(pnp_ransac_max_error=5.0)

    # Set threads via env
    monkeypatch.setenv("SFM_EDM_PNP_THREADS", "4")
    _cam, options = tracker._new_pose_estimation_context()
    assert options.ransac.num_threads == 4

    # Default without env
    monkeypatch.delenv("SFM_EDM_PNP_THREADS", raising=False)
    _cam, options = tracker._new_pose_estimation_context()
    assert options.ransac.num_threads == 1



def test_prior_refine_survives_noisy_prior(monkeypatch) -> None:
    """Perturbed prior (+-0.3m, +-10deg) must still yield an accurate pose.

    Either the 1-step refine converges or the code falls back to full RANSAC;
    both paths must stay within tight error bounds (no local-minimum latch).
    """
    import pycolmap

    monkeypatch.setenv("SFM_EDM_PRIOR_PNP_REFINE", "1")
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    tracker.cam = SimpleNamespace(width=1280, height=720)
    pcam = pycolmap.Camera(model="PINHOLE", width=1280, height=720,
                           params=[900.0, 900.0, 640.0, 360.0])
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = 5.0
    options.ransac.random_seed = 0
    tracker._pose_estimation_context = lambda: (pcam, options)
    tracker._new_pose_estimation_context = lambda _seed=-1: (pcam, options)

    rng = np.random.default_rng(0)
    pts3d = np.column_stack([
        rng.uniform(-3.0, 3.0, 300),
        rng.uniform(-2.0, 2.0, 300),
        rng.uniform(4.0, 10.0, 300),
    ])
    q_true = pycolmap.Rotation3d()
    t_true = np.zeros(3)
    cam_true = pycolmap.Rigid3d(q_true, t_true)
    pts_cam = np.asarray(cam_true * pts3d)
    uv = np.column_stack([
        pts_cam[:, 0] / pts_cam[:, 2] * 900.0 + 640.0,
        pts_cam[:, 1] / pts_cam[:, 2] * 900.0 + 360.0,
    ]) + rng.normal(0.0, 0.5, (300, 2))
    # 20% gross outliers.
    uv[:60] = rng.uniform([0, 0], [1280, 720], (60, 2))
    conf = np.ones(300, np.float32)

    # Perturbed prior: 0.3 m translation + 10 deg yaw.
    half = np.radians(10.0) / 2.0
    dq = pycolmap.Rotation3d(np.array([np.cos(half), 0.0, 0.0, np.sin(half)]))
    noisy = pycolmap.Rigid3d(dq * q_true, t_true + np.array([0.3, 0.0, 0.0]))

    out, _ms = tracker._estimate_pose_candidate(
        uv, pts3d, conf, pcam, options, prior={"cam_from_world": noisy})
    assert out is not None
    result, _p2d, _p3d, metrics = out
    est = result["cam_from_world"]
    got_t = np.asarray(est.translation)
    assert np.linalg.norm(got_t - t_true) < 0.3
    assert metrics["reproj_rms"] is not None and float(metrics["reproj_rms"]) < 5.0
    assert int(result["num_inliers"]) >= 60


def _acquire_selection(min_inliers: int = 80, refs: list[str] | None = None):
    return _CandidateSelection(
        acquiring=True,
        refs=list(refs or ["a", "b", "c"]),
        mode="megaloc_lost_global_retry",
        min_inliers=min_inliers,
        vpr_ms=1.0,
    )


def _lost_tracker(**cfg_kwargs) -> ProductionEDMTracker:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(**cfg_kwargs)
    tracker.st = RuntimeState(state="LOST", lost_frames=9, last_capture_stamp=1.0)
    return tracker


def _relaxed_attempt(inliers: int, agree: int):
    return tracker_module._PoseAttempt(
        result={
            "cam_from_world": pycolmap.Rigid3d(),
            "num_inliers": inliers,
            "inlier_mask": np.ones(inliers, dtype=bool),
        },
        points2d=np.zeros((inliers, 2)),
        points3d=np.zeros((inliers, 3)),
        metrics={"reproj_rms": 2.0, "inlier_ratio": 0.25, "inlier_grid_cells": 8},
        selected_ref="a",
        pnp_ms=1.0,
        strong_count=0,
        relaxed_agree_count=agree,
    )


def test_relaxed_acquire_floor_requires_agreeing_references() -> None:
    tracker = _lost_tracker(acquire_relaxed_min_inliers=66)
    selection = _acquire_selection()

    assert tracker._relaxed_acquire_gate(selection, _relaxed_attempt(70, 1)) is None
    assert tracker._relaxed_acquire_gate(selection, _relaxed_attempt(70, 2)) == (66, 3.0)


def test_relaxed_acquire_floor_is_bounded_to_the_near_miss_band() -> None:
    tracker = _lost_tracker(acquire_relaxed_min_inliers=66)
    selection = _acquire_selection()

    # Below the relaxed floor and at/above the standard floor both fall through
    # to the unchanged single-threshold behaviour.
    assert tracker._relaxed_acquire_gate(selection, _relaxed_attempt(65, 3)) is None
    assert tracker._relaxed_acquire_gate(selection, _relaxed_attempt(80, 3)) is None


def test_relaxed_acquire_floor_is_off_for_boot_and_tracking() -> None:
    tracker = _lost_tracker(acquire_relaxed_min_inliers=66)
    selection = _acquire_selection()
    attempt = _relaxed_attempt(70, 3)

    tracker.st.state = "BOOT_INIT"
    assert tracker._relaxed_acquire_gate(selection, attempt) is None

    tracker.st.state = "LOST"
    track_selection = _CandidateSelection(
        acquiring=False, refs=["a"], mode="edm_track", min_inliers=80, vpr_ms=0.0
    )
    assert tracker._relaxed_acquire_gate(track_selection, attempt) is None


def test_relaxed_acquire_floor_defaults_to_66() -> None:
    # Shipped 2026-09-05 on the seven-video 720p corpus: +231 frames overall,
    # P168 the only segment that pays (-21/-36/-21 across seeds 0/1/2), and the
    # off-map hard negative stays 0/241 with the floor on. See EDMConfig.
    tracker = _lost_tracker()
    assert tracker.cfg.acquire_relaxed_min_inliers == 66
    assert tracker._relaxed_acquire_gate(
        _acquire_selection(), _relaxed_attempt(70, 5)) == (66, 3.0)


def test_relaxed_acquire_floor_can_be_turned_off() -> None:
    tracker = _lost_tracker(acquire_relaxed_min_inliers=0)
    assert tracker._relaxed_acquire_gate(_acquire_selection(), _relaxed_attempt(70, 5)) is None


def test_relaxed_acquire_reproj_limit_never_widens_the_standard_limit() -> None:
    tracker = _lost_tracker(
        acquire_relaxed_min_inliers=66,
        acquire_relaxed_max_reproj_error=9.0,
        max_reproj_error_acquire=5.0,
    )
    relaxed = tracker._relaxed_acquire_gate(_acquire_selection(), _relaxed_attempt(70, 2))
    assert relaxed == (66, 5.0)


def _scored_candidate(inliers: int, x: float):
    result = {
        "cam_from_world": pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([-x, 0.0, 0.0])),
        "num_inliers": inliers,
        "inlier_mask": np.ones(inliers, dtype=bool),
    }
    return (
        result,
        np.zeros((inliers, 2)),
        np.zeros((inliers, 3)),
        {"reproj_rms": 2.0, "inlier_ratio": 0.25, "inlier_grid_cells": 8},
    )


def test_relaxed_agreement_counts_only_references_on_the_same_center() -> None:
    tracker = _lost_tracker(acquire_relaxed_min_inliers=66, max_jump=2.0)
    selection = _acquire_selection()
    chosen = _scored_candidate(70, 0.0)
    scored = [
        ("a", chosen),
        ("b", _scored_candidate(68, 0.5)),  # agrees (0.5 m < 4.0 m consensus limit)
        ("c", _scored_candidate(90, 40.0)),  # strong but somewhere else
        ("d", _scored_candidate(20, 0.1)),  # same place, below the relaxed floor
    ]

    assert tracker._count_relaxed_agreement(selection, scored, chosen) == 2


def test_relaxed_agreement_is_not_computed_for_strong_candidates() -> None:
    tracker = _lost_tracker(acquire_relaxed_min_inliers=66)
    selection = _acquire_selection()
    chosen = _scored_candidate(120, 0.0)
    scored = [("a", chosen), ("b", _scored_candidate(110, 0.2))]

    assert tracker._count_relaxed_agreement(selection, scored, chosen) == 0


def _starvation_tracker(**cfg_kwargs) -> ProductionEDMTracker:
    tracker = _lost_tracker(**cfg_kwargs)
    tracker._last_lost_search_stage = None
    tracker._last_lost_radius_factor = None
    return tracker


def _local_selection():
    return _CandidateSelection(
        acquiring=True,
        refs=["r0", "r1"],
        mode="edm_local_recovery",
        min_inliers=80,
        vpr_ms=0.0,
    )


def test_consecutive_starved_lost_frames_arm_global_retrieval() -> None:
    tracker = _starvation_tracker(lost_starved_global_frames=3, lost_starved_corr_max=5)
    selection = _local_selection()

    for expected in (1, 2):
        info: dict = {"n_corr": 2}
        tracker._observe_lost_starvation("LOST", selection, info)
        assert info["lost_starved_frames"] == expected
        assert tracker._lost_starvation_armed() is False

    info = {"n_corr": 0}
    tracker._observe_lost_starvation("LOST", selection, info)
    assert tracker._lost_starvation_armed() is True
    assert tracker._should_run_global_retrieval(None) is True


def test_starvation_counter_resets_on_dense_matches_and_leaving_lost() -> None:
    tracker = _starvation_tracker(lost_starved_global_frames=2, lost_starved_corr_max=5)
    selection = _local_selection()

    tracker._observe_lost_starvation("LOST", selection, {"n_corr": 1})
    tracker._observe_lost_starvation("LOST", selection, {"n_corr": 300})
    assert tracker.st.lost_starved_frames == 0

    tracker._observe_lost_starvation("LOST", selection, {"n_corr": 1})
    tracker._observe_lost_starvation("TRACK", selection, {"n_corr": 1})
    assert tracker.st.lost_starved_frames == 0


def test_retrieval_frames_do_not_clip_a_starved_run() -> None:
    # The interleaved MegaLoc cadence puts a retrieval frame between every two
    # local frames, so resetting on retrieval frames would cap the counter at
    # two and a k>=3 escape hatch could never arm.
    tracker = _starvation_tracker(lost_starved_global_frames=3, lost_starved_corr_max=5)
    selection = _local_selection()
    retrieval = replace(selection, mode="megaloc_lost_near")

    tracker._observe_lost_starvation("LOST", selection, {"n_corr": 2})
    tracker._observe_lost_starvation("LOST", selection, {"n_corr": 2})
    tracker._observe_lost_starvation("LOST", retrieval, {"n_corr": 340})
    assert tracker.st.lost_starved_frames == 2
    assert tracker._lost_starvation_armed() is False

    tracker._observe_lost_starvation("LOST", selection, {"n_corr": 3})
    assert tracker.st.lost_starved_frames == 3
    assert tracker._lost_starvation_armed() is True


def test_starvation_accounting_is_off_by_default() -> None:
    tracker = _starvation_tracker()
    info: dict = {"n_corr": 0}
    tracker._observe_lost_starvation("LOST", _local_selection(), info)

    assert "lost_starved_frames" not in info
    assert tracker.st.lost_starved_frames == 0
    assert tracker._lost_starvation_armed() is False


def test_starved_retrieval_ranks_the_whole_map_instead_of_the_nearby_pool() -> None:
    tracker = _starvation_tracker(lost_starved_global_frames=2, lost_starved_corr_max=5)
    tracker.st.lost_starved_frames = 2
    tracker.map = SimpleNamespace(ref_names=[f"r{i}" for i in range(50)])
    tracker._track_candidates = lambda *_a, **_k: pytest.fail(
        "a starved retry must not restrict retrieval to the last-known pose"
    )

    retrieve_calls = []

    def retrieve(_rgb, topk, candidates=None):
        retrieve_calls.append((topk, candidates))
        return ["g0", "g1"]

    tracker.loc = SimpleNamespace(retrieve=retrieve)
    refs, _vpr_ms, nearby = tracker._global_retrieval(
        np.zeros((8, 8, 3), np.uint8), 2.0, force_global=True
    )

    assert refs == ["g0", "g1"]
    assert nearby is False
    assert retrieve_calls == [(tracker.cfg.lost_local_topk, None)]
    assert tracker._last_lost_search_stage == "starved_global"
    assert tracker._last_lost_radius_factor is None


def _widen_tracker(**cfg_kwargs) -> ProductionEDMTracker:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(**cfg_kwargs)
    tracker.st = RuntimeState(state="TRACK", last_capture_stamp=1.0)
    tracker.cam = SimpleNamespace(width=1280, height=720)
    return tracker


def _track_selection(refs: list[str] | None = None):
    return _CandidateSelection(
        acquiring=False,
        refs=list(refs or ["a"]),
        mode="edm_temporal_map",
        min_inliers=50,
        vpr_ms=0.0,
    )


def _attempt_with(inliers: int, reproj: float = 1.5):
    return tracker_module._PoseAttempt(
        result={
            "cam_from_world": pycolmap.Rigid3d(),
            "num_inliers": inliers,
            "inlier_mask": np.ones(max(inliers, 1), dtype=bool),
        },
        points2d=np.zeros((max(inliers, 1), 2)),
        points3d=np.zeros((max(inliers, 1), 3)),
        metrics={"reproj_rms": reproj, "inlier_ratio": 0.8, "inlier_grid_cells": 8},
        selected_ref="a",
        pnp_ms=1.0,
    )


def test_track_widening_is_skipped_when_the_frame_already_passes() -> None:
    tracker = _widen_tracker(track_miss_widen_topk=3)
    tracker._track_candidates = lambda *_a, **_k: pytest.fail("healthy frames must not retry")
    assert tracker._widen_track_refs_on_miss(
        np.zeros((576, 1024), np.uint8),
        _track_selection(),
        1.1,
        None,
        _attempt_with(80),
    ) is None


def test_track_widening_retries_the_same_frame_with_more_refs() -> None:
    tracker = _widen_tracker(track_miss_widen_topk=3)
    tracker._track_candidates = lambda topk, _stamp, **_k: ["a", "b", "c"][:topk]
    tracker._trajectory_would_allow = lambda *_a, **_k: True
    calls = []

    def fake_match(gray, selection, *, capture_stamp=None, prepared_query=None):
        calls.append(list(selection.refs))
        return (
            _pnp_batch([120, 90, 60]),
            _attempt_with(74),
            7.0,
        )

    tracker._match_and_estimate = fake_match
    widened = tracker._widen_track_refs_on_miss(
        np.zeros((576, 1024), np.uint8),
        _track_selection(),
        1.1,
        None,
        _attempt_with(34),
    )

    assert widened is not None
    selection, _batch, attempt, _ms, widen_info = widened
    assert calls == [["a", "b", "c"]]
    assert selection.mode == "edm_track_widen"
    assert widen_info["inliers_before"] == 34
    assert widen_info["inliers_after"] == 74
    assert int(attempt.result["num_inliers"]) == 74
    assert tracker._track_widen_accepts == 1


def test_track_widening_keeps_the_original_miss_when_the_retry_also_fails() -> None:
    tracker = _widen_tracker(track_miss_widen_topk=3)
    tracker._track_candidates = lambda topk, _stamp, **_k: ["a", "b", "c"][:topk]
    tracker._match_and_estimate = lambda *_a, **_k: (
        _pnp_batch([40]),
        _attempt_with(41),
        7.0,
    )
    assert tracker._widen_track_refs_on_miss(
        np.zeros((576, 1024), np.uint8),
        _track_selection(),
        1.1,
        None,
        _attempt_with(34),
    ) is None
    assert tracker._track_widen_attempts == 1
    assert getattr(tracker, "_track_widen_accepts", 0) == 0


def test_track_widening_defaults_to_off_and_skips_non_track_states() -> None:
    tracker = _widen_tracker()
    tracker._track_candidates = lambda *_a, **_k: pytest.fail("widening is off by default")
    assert tracker._widen_track_refs_on_miss(
        np.zeros((576, 1024), np.uint8), _track_selection(), 1.1, None, _attempt_with(34)
    ) is None

    tracker = _widen_tracker(track_miss_widen_topk=3)
    tracker.st.state = "WEAK_TRACK"
    tracker._track_candidates = lambda *_a, **_k: pytest.fail("only TRACK widens")
    assert tracker._widen_track_refs_on_miss(
        np.zeros((576, 1024), np.uint8), _track_selection(), 1.1, None, _attempt_with(34)
    ) is None


def test_weak_widening_retries_the_same_frame_with_more_refs() -> None:
    tracker = _widen_tracker(weak_miss_widen_topk=5)
    tracker.st.state = "WEAK_TRACK"
    tracker._track_candidates = lambda topk, _stamp, **_k: ["a", "b", "c", "d", "e"][:topk]
    tracker._trajectory_would_allow = lambda *_a, **_k: True
    calls = []

    def fake_match(gray, selection, *, capture_stamp=None, prepared_query=None):
        calls.append(list(selection.refs))
        return (_pnp_batch([80, 70, 60, 50, 40]), _attempt_with(32), 7.0)

    tracker._match_and_estimate = fake_match
    weak_selection = _CandidateSelection(
        acquiring=False, refs=["a", "b", "c"], mode="edm_track",
        min_inliers=30, vpr_ms=0.0,
    )
    widened = tracker._widen_track_refs_on_miss(
        np.zeros((576, 1024), np.uint8),
        weak_selection,
        1.1,
        None,
        _attempt_with(27),
    )

    assert widened is not None
    selection, _batch, attempt, _ms, widen_info = widened
    assert calls == [["a", "b", "c", "d", "e"]]
    assert selection.mode == "edm_weak_widen"
    assert widen_info["inliers_before"] == 27
    assert int(attempt.result["num_inliers"]) == 32


def test_weak_widening_is_independent_of_the_track_knob() -> None:
    tracker = _widen_tracker(track_miss_widen_topk=3)
    tracker.st.state = "WEAK_TRACK"
    tracker._track_candidates = lambda *_a, **_k: pytest.fail(
        "WEAK widening uses its own knob, not track_miss_widen_topk"
    )
    weak_selection = _CandidateSelection(
        acquiring=False, refs=["a", "b", "c"], mode="edm_track",
        min_inliers=30, vpr_ms=0.0,
    )
    assert tracker._widen_track_refs_on_miss(
        np.zeros((576, 1024), np.uint8),
        weak_selection,
        1.1,
        None,
        _attempt_with(27),
    ) is None


def test_boot_relaxed_floor_applies_only_at_boot() -> None:
    # LOST's floor is turned off here so the assertion below tests the two knobs'
    # independence rather than the LOST default.
    tracker = _lost_tracker(boot_relaxed_min_inliers=60, acquire_relaxed_min_inliers=0)
    selection = _acquire_selection()

    tracker.st.state = "BOOT_INIT"
    assert tracker._relaxed_acquire_gate(selection, _relaxed_attempt(70, 2)) == (60, 3.0)

    # LOST keeps its own (here disabled) floor: the two knobs are independent.
    tracker.st.state = "LOST"
    assert tracker._relaxed_acquire_gate(selection, _relaxed_attempt(70, 2)) is None


def test_boot_relaxed_accept_needs_a_second_confirming_frame() -> None:
    tracker = _lost_tracker(boot_relaxed_min_inliers=60)
    tracker.st.state = "BOOT_INIT"
    tracker.st.center = None
    center = np.array([1.0, 2.0, 3.0])
    info: dict = {"state_in": "BOOT_INIT", "acquire_relaxed": {"min_inliers": 60}}

    # First frame: nothing to confirm against -> held, and counted as a miss.
    assert tracker._trajectory_allows(center, 0.1, 10.0, info, None, None) is False
    assert info["rejected"] == "boot_relaxed_unconfirmed"
    assert tracker.st.pending_reacquire_center is not None

    # Second, independent frame within the stale-reacquisition window -> accept.
    info2: dict = {"state_in": "BOOT_INIT", "acquire_relaxed": {"min_inliers": 60}}
    assert tracker._trajectory_allows(
        center + np.array([0.05, 0.0, 0.0]), 0.12, 10.1, info2, None, None
    ) is True
    assert info2["boot_relaxed_confirmed"] is True
    assert tracker.st.pending_reacquire_center is None


def test_boot_relaxed_accept_rejects_a_disagreeing_second_frame() -> None:
    tracker = _lost_tracker(boot_relaxed_min_inliers=60)
    tracker.st.state = "BOOT_INIT"
    tracker.st.center = None
    info: dict = {"state_in": "BOOT_INIT", "acquire_relaxed": {"min_inliers": 60}}
    tracker._trajectory_allows(np.zeros(3), 0.0, 10.0, info, None, None)

    info2: dict = {"state_in": "BOOT_INIT", "acquire_relaxed": {"min_inliers": 60}}
    assert tracker._trajectory_allows(
        np.array([5.0, 0.0, 0.0]), 0.0, 10.1, info2, None, None
    ) is False
    assert info2["rejected"] == "boot_relaxed_unconfirmed"


def test_standard_boot_accept_is_unchanged() -> None:
    tracker = _lost_tracker(boot_relaxed_min_inliers=60)
    tracker.st.state = "BOOT_INIT"
    tracker.st.center = None
    info: dict = {"state_in": "BOOT_INIT"}
    assert tracker._trajectory_allows(np.zeros(3), 0.0, 10.0, info, None, None) is True
    assert "rejected" not in info


def test_relaxed_probation_holds_weak_track_for_the_configured_frames() -> None:
    tracker = _lost_tracker(acquire_relaxed_probation_frames=3, weak_after=2)

    states = [
        tracker._next_state_after_accept(
            {"acquire_relaxed": {"min_inliers": 66}} if frame == 0 else {}
        )
        for frame in range(5)
    ]

    assert states == ["WEAK_TRACK", "WEAK_TRACK", "WEAK_TRACK", "TRACK", "TRACK"]
    assert tracker.st.relaxed_probation_frames == 0


def test_relaxed_probation_off_keeps_track_after_a_relaxed_accept() -> None:
    tracker = _lost_tracker()
    assert tracker.cfg.acquire_relaxed_probation_frames == 0
    assert (
        tracker._next_state_after_accept({"acquire_relaxed": {"min_inliers": 66}})
        == "TRACK"
    )
