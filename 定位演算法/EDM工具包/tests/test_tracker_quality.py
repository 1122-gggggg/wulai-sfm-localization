from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pycolmap
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

import production_edm_tracker as tracker_module  # noqa: E402
from edm_matcher import EDMMatcher  # noqa: E402
from production_edm_tracker import (  # noqa: E402
    EDMConfig,
    ProductionEDMTracker,
    RuntimeState,
    reprojection_rank,
    reprojection_metrics,
    spatially_cap_indices,
)
from reloc_localizer_edm import EDMLocalizer  # noqa: E402


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


@pytest.mark.parametrize("pixel_offset", [0.0, 10.0])
def test_boot_staging_evaluates_the_complete_retrieved_set(
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


sys.path.insert(0, str(ROOT / "tests"))
from bench_video_edm import (  # noqa: E402
    STRESS_MATRIX,
    STRESS_SEED,
    apply_stress_transform,
    build_stress_report,
    camera_contract,
    frame_sha256,
    require_frame_camera_contract,
    sequence_sha256,
    stress_deltas,
    summarize_stress_frames,
    transform_record,
)


def _stress_camera(width: int = 64, height: int = 48):
    return SimpleNamespace(
        model="PINHOLE",
        width=width,
        height=height,
        params=[50.0, 50.0, width / 2.0, height / 2.0],
    )


def _stress_frame(width: int = 64, height: int = 48) -> np.ndarray:
    column = np.linspace(0, 255, width, dtype=np.uint8)
    row = np.linspace(32, 223, height, dtype=np.uint8)
    plane = (column[None, :] // 2 + row[:, None] // 2).astype(np.uint8)
    high = ((np.arange(height)[:, None] * 13 + np.arange(width)[None, :] * 17) % 251).astype(np.uint8)
    return np.stack([plane, np.flipud(plane) ^ high, np.fliplr(plane)], axis=2)


def test_stress_matrix_starts_with_unmodified_baseline() -> None:
    assert STRESS_MATRIX[0]["id"] == "identity"
    assert tuple(STRESS_MATRIX[0]["ops"]) == ()
    families = {op["op"] for spec in STRESS_MATRIX for op in spec["ops"]}
    assert families == {"crop", "rotate", "exposure", "gamma", "contrast", "blur", "jpeg"}
    assert any(len(spec["ops"]) > 1 for spec in STRESS_MATRIX)


def test_each_stress_transform_is_deterministic_and_preserves_camera_frame() -> None:
    camera = _stress_camera()
    frame = _stress_frame()
    before = camera_contract(camera)
    original = frame.copy()
    for spec in STRESS_MATRIX:
        first = apply_stress_transform(frame, spec, camera=camera)
        second = apply_stress_transform(frame, spec, camera=camera)
        require_frame_camera_contract(first, camera)
        assert first.shape == frame.shape
        assert first.dtype == np.uint8
        assert np.isfinite(first.astype(np.float32)).all()
        assert frame_sha256(first) == frame_sha256(second)
        assert np.array_equal(first, second)
        if spec["id"] == "identity":
            assert np.array_equal(first, frame)
            assert frame_sha256(first) == frame_sha256(frame)
        else:
            assert frame_sha256(first) != frame_sha256(frame)
    assert np.array_equal(frame, original)
    assert camera_contract(camera) == before


def test_stress_transform_records_identity_params_and_hashes() -> None:
    frame = _stress_frame()
    spec = next(item for item in STRESS_MATRIX if item["id"] == "hcrop_0.10+jpeg_70")
    warped = apply_stress_transform(frame, spec)
    record = transform_record(frame, warped, spec, seed=STRESS_SEED)
    assert record["id"] == "hcrop_0.10+jpeg_70"
    assert [op["op"] for op in record["ops"]] == ["crop", "jpeg"]
    assert record["ops"][0]["axis"] == "horizontal"
    assert record["ops"][0]["fraction"] == 0.10
    assert record["ops"][1]["quality"] == 70
    assert record["seed"] == STRESS_SEED
    assert record["input_sha256"] == frame_sha256(frame)
    assert record["output_sha256"] == frame_sha256(warped)
    assert record["input_sha256"] != record["output_sha256"]
    sequential = apply_stress_transform(
        apply_stress_transform(frame, {"id": "hcrop_0.10", "ops": (spec["ops"][0],)}),
        {"id": "jpeg_70", "ops": (spec["ops"][1],)},
    )
    assert frame_sha256(sequential) == record["output_sha256"]


def test_stress_summary_reports_accepted_states_lost_and_deltas() -> None:
    baseline_infos = [
        {"ok": True, "state_out": "TRACK", "inliers": 120, "reproj_rms": 0.4, "total_ms": 10.0},
        {"ok": True, "state_out": "TRACK", "inliers": 100, "reproj_rms": 0.6, "total_ms": 12.0},
        {"ok": True, "state_out": "WEAK_TRACK", "inliers": 80, "reproj_rms": 0.8, "total_ms": 11.0},
        {"ok": False, "state_out": "LOST", "inliers": 0, "reproj_rms": None, "total_ms": 9.0},
    ]
    stressed_infos = [
        {"ok": True, "state_out": "TRACK", "inliers": 90, "reproj_rms": 1.0, "total_ms": 14.0},
        {"ok": False, "state_out": "LOST", "inliers": 0, "reproj_rms": None, "total_ms": 13.0},
        {"ok": False, "state_out": "LOST", "inliers": 0, "reproj_rms": None, "total_ms": 15.0},
        {"ok": False, "state_out": "LOST", "inliers": 0, "reproj_rms": None, "total_ms": 16.0},
    ]
    baseline = summarize_stress_frames(baseline_infos)
    stressed = summarize_stress_frames(stressed_infos)
    assert baseline["accepted_rate"] == pytest.approx(0.75)
    assert baseline["states"] == {"TRACK": 2, "WEAK_TRACK": 1, "LOST": 1, "WEAK": 1}
    assert baseline["longest_lost"] == 1
    assert baseline["inliers_median"] == pytest.approx(100.0)
    assert baseline["reproj_rms_median"] == pytest.approx(0.6)
    assert baseline["latency_median_ms"] == pytest.approx(10.5)
    assert stressed["accepted_rate"] == pytest.approx(0.25)
    assert stressed["longest_lost"] == 3
    deltas = stress_deltas(stressed, baseline)
    assert deltas["accepted_rate"] == pytest.approx(-0.5)
    assert deltas["longest_lost"] == pytest.approx(2.0)
    assert deltas["inliers_median"] == pytest.approx(-10.0)
    assert deltas["reproj_rms_median"] == pytest.approx(0.4)
    assert deltas["latency_median_ms"] == pytest.approx(4.0)
    assert deltas["states"]["LOST"] == 2
    assert deltas["states"]["TRACK"] == -1

    camera = _stress_camera()
    frame = _stress_frame()
    identity = STRESS_MATRIX[0]
    crop = next(item for item in STRESS_MATRIX if item["id"] == "hcrop_0.10")
    identity_out = apply_stress_transform(frame, identity, camera=camera)
    crop_out = apply_stress_transform(frame, crop, camera=camera)
    report = build_stress_report(
        video="probe.mp4",
        bundle="unused-bundle.pt",
        camera=camera,
        seed=STRESS_SEED,
        rows=[
            {
                "id": identity["id"],
                "ops": [],
                "input_sha256": sequence_sha256([frame_sha256(frame)], STRESS_SEED),
                "output_sha256": sequence_sha256([frame_sha256(identity_out)], STRESS_SEED),
                "probe": transform_record(frame, identity_out, identity),
                "metrics": baseline,
            },
            {
                "id": crop["id"],
                "ops": [dict(crop["ops"][0])],
                "input_sha256": sequence_sha256([frame_sha256(frame)], STRESS_SEED),
                "output_sha256": sequence_sha256([frame_sha256(crop_out)], STRESS_SEED),
                "probe": transform_record(frame, crop_out, crop),
                "metrics": stressed,
            },
        ],
    )
    assert report["schema"] == "edm-video-stress-matrix/v1"
    assert report["robustness_kind"] == "2d_appearance_image_plane"
    assert "not proof of 3D parallax" in report["note"]
    assert report["camera"] == camera_contract(camera)
    assert report["transforms"][0]["id"] == "identity"
    assert report["transforms"][0]["delta_vs_baseline"]["accepted_rate"] == pytest.approx(0.0)
    assert report["transforms"][1]["delta_vs_baseline"]["accepted_rate"] == pytest.approx(-0.5)
    assert report["transforms"][1]["probe"]["input_sha256"] != report["transforms"][1]["probe"]["output_sha256"]


def test_stress_helpers_leave_map_and_profile_assets_untouched(tmp_path) -> None:
    bundle = tmp_path / "bundle.pt"
    profile = tmp_path / "localizer_profile.json"
    bundle.write_bytes(b"map-bytes")
    profile.write_text("{}", encoding="utf-8")
    stamp = (bundle.stat().st_mtime_ns, profile.stat().st_mtime_ns)
    camera = _stress_camera()
    frame = _stress_frame()
    for spec in STRESS_MATRIX:
        apply_stress_transform(frame, spec, camera=camera)
    summarize_stress_frames([])
    assert bundle.read_bytes() == b"map-bytes"
    assert profile.read_text(encoding="utf-8") == "{}"
    assert (bundle.stat().st_mtime_ns, profile.stat().st_mtime_ns) == stamp


def test_stress_transform_rejects_camera_frame_mismatch() -> None:
    camera = _stress_camera(width=64, height=48)
    with pytest.raises(ValueError, match="does not match camera"):
        apply_stress_transform(_stress_frame(width=32, height=24), STRESS_MATRIX[0], camera=camera)
    with pytest.raises(ValueError, match="must start with the unmodified identity"):
        build_stress_report(
            video="probe.mp4",
            bundle="unused-bundle.pt",
            camera=camera,
            seed=STRESS_SEED,
            rows=[{
                "id": "hcrop_0.10",
                "ops": [],
                "input_sha256": "0" * 64,
                "output_sha256": "1" * 64,
                "probe": {"input_sha256": "0" * 64, "output_sha256": "1" * 64},
                "metrics": summarize_stress_frames([]),
            }],
        )


