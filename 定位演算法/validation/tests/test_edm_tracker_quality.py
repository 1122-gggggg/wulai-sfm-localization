from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
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
from edm_matcher import EDMMatcher  # noqa: E402
from production_edm_tracker import (  # noqa: E402
    adaptive_jump_limit,
    EDMConfig,
    ProductionEDMTracker,
    RuntimeState,
    reprojection_metrics,
    reprojection_rank,
    spatially_cap_indices,
)
from reloc_localizer_edm import EDMLocalizer, _validate_edm_bundle_schema  # noqa: E402


def test_edm_megaloc_loads_only_from_verified_local_assets(monkeypatch) -> None:
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

    edm_localizer_module.MegaLocQuery(device="cpu")

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


def test_edm_config_rejects_non_finite_and_relational_values() -> None:
    with pytest.raises(ValueError, match="max_jump"):
        EDMConfig(max_jump=float("nan"))
    with pytest.raises(ValueError, match="acquire_initial_topk"):
        EDMConfig(boot_global_topk=2, acquire_initial_topk=3)
    with pytest.raises(ValueError, match="history"):
        EDMConfig(adaptive_jump_min_history=20, adaptive_jump_history_size=10)
    with pytest.raises(ValueError, match="bootstrap"):
        EDMConfig(adaptive_jump_bootstrap=0.005, adaptive_jump_ceiling=0.008)


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
    tracker.temporal_xyz_by_cell = None
    return tracker


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
def test_boot_staging_stops_on_good_geometry_and_expands_on_bad_geometry(
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

        def correspondences(self, _gray, refs, **_kwargs):
            calls.append(list(refs))
            return (
                points2d.copy(), points3d.copy(), np.ones(100, np.float32),
                [50] * len(refs),
            )

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
        max_corr_total=100,
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
        assert info["staged_early_stop"]
        assert info["refs"] == names[:2]
        assert calls == [names[:2]]
        assert tracker.temporal_gray is None
        assert tracker.temporal_xyz_by_cell is None
    else:
        assert not info["ok"]
        assert not info["staged_early_stop"]
        assert info["rejected"] == "reprojection"
        assert info["reproj_rms"] == pytest.approx(pixel_offset)
        assert calls == [names[:2], names[2:]]
