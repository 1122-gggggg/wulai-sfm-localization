from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


VALIDATION = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[4]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_result(path: Path, rows: list[dict], **extra) -> None:
    path.write_text(json.dumps({"rows": rows, **extra}), encoding="utf-8")


def complete_stream_fields(n: int) -> dict:
    return {
        "capture_opened": True,
        "reported_raw_frames": n,
        "expected_raw_frames": n,
        "decoded_raw_frames": n,
        "sampled_frames": n,
        "expected_sampled_frames": n,
        "decode_complete": True,
        "decode_errors": 0,
        "integrity_min_sampled_frames": 0,
    }


def test_quality_gate_rejects_empty_results(tmp_path, monkeypatch):
    pipeline = load_module("localize_pipeline_empty", VALIDATION.parent / "pipeline" / "localize_pipeline.py")
    monkeypatch.setattr(pipeline, "LOC_ROOT", tmp_path)
    result = tmp_path / "empty.json"
    write_result(result, [])

    assert not pipeline.write_quality_report(result, 0.9, 0, 30)
    report = result.with_suffix(".quality_report.md").read_text(encoding="utf-8")
    assert "Overall: FAIL" in report
    assert "no result rows" in report


def test_quality_gate_rejects_zero_frames_and_error_counters(tmp_path, monkeypatch):
    pipeline = load_module("localize_pipeline_bad", VALIDATION.parent / "pipeline" / "localize_pipeline.py")
    monkeypatch.setattr(pipeline, "LOC_ROOT", tmp_path)
    result = tmp_path / "bad.json"
    write_result(
        result,
        [{
            "set": "zero",
            "n": 0,
            "base_success": 0.0,
            "final_success": 0.0,
            "gain_pp": 0.0,
            "ok_to_fail": 0,
            "base_max_fail_run": 0,
            "final_max_fail_run": 0,
        }],
        error_frame_counters={"match_error": 1},
    )

    assert not pipeline.write_quality_report(result, 0.9, 0, 30)
    report = result.with_suffix(".quality_report.md").read_text(encoding="utf-8")
    assert "| zero | 0 |" in report
    assert "| FAIL |" in report
    assert "error-frame counters" in report


def test_quality_report_never_labels_a_regression_pass(tmp_path, monkeypatch):
    pipeline = load_module("localize_pipeline_reg", VALIDATION.parent / "pipeline" / "localize_pipeline.py")
    monkeypatch.setattr(pipeline, "LOC_ROOT", tmp_path)
    result = tmp_path / "regression.json"
    write_result(
        result,
        [{
            "set": "regressed",
            "n": 10,
            "base_success": 1.0,
            "final_success": 0.9,
            "gain_pp": -10.0,
            "ok_to_fail": 1,
            "base_max_fail_run": 0,
            "final_max_fail_run": 1,
            **complete_stream_fields(10),
        }],
    )

    assert not pipeline.write_quality_report(result, 0.9, 0, 30)
    report = result.with_suffix(".quality_report.md").read_text(encoding="utf-8")
    assert "| regressed | 10 |" in report
    assert "| FAIL |" in report
    assert "PASS |" not in report


def test_quality_gate_rejects_truncated_or_unbounded_stream(tmp_path, monkeypatch):
    pipeline = load_module("localize_pipeline_stream", VALIDATION.parent / "pipeline" / "localize_pipeline.py")
    monkeypatch.setattr(pipeline, "LOC_ROOT", tmp_path)
    row = {
        "set": "truncated",
        "n": 1,
        "base_n": 1,
        "final_n": 1,
        "base_success": 1.0,
        "final_success": 1.0,
        "gain_pp": 0.0,
        "ok_to_fail": 0,
        "base_max_fail_run": 0,
        "final_max_fail_run": 0,
        "capture_opened": True,
        "reported_raw_frames": 1,
        "expected_raw_frames": 1,
        "decoded_raw_frames": 1,
        "sampled_frames": 1,
        "expected_sampled_frames": 1,
        "decode_complete": True,
        "decode_errors": 0,
        "integrity_min_sampled_frames": 30,
    }
    result = tmp_path / "truncated.json"
    write_result(result, [row])
    assert not pipeline.write_quality_report(result, 0.9, 0, 30)

    row.update({
        "set": "unknown_length",
        "reported_raw_frames": None,
        "expected_raw_frames": None,
        "decoded_raw_frames": 1,
        "decode_complete": None,
        "decode_errors": 0,
    })
    write_result(result, [row])
    assert not pipeline.write_quality_report(result, 0.9, 0, 30)


def test_quality_gate_accepts_unknown_length_only_with_explicit_minimum(tmp_path, monkeypatch):
    pipeline = load_module("localize_pipeline_min", VALIDATION.parent / "pipeline" / "localize_pipeline.py")
    monkeypatch.setattr(pipeline, "LOC_ROOT", tmp_path)
    row = {
        "set": "bounded_stream",
        "n": 10,
        "base_n": 10,
        "final_n": 10,
        "base_success": 1.0,
        "final_success": 1.0,
        "gain_pp": 0.0,
        "ok_to_fail": 0,
        "base_max_fail_run": 0,
        "final_max_fail_run": 0,
        "capture_opened": True,
        "reported_raw_frames": None,
        "expected_raw_frames": None,
        "decoded_raw_frames": 10,
        "sampled_frames": 10,
        "expected_sampled_frames": None,
        "decode_complete": None,
        "decode_errors": 0,
        "integrity_min_sampled_frames": 10,
    }
    result = tmp_path / "bounded.json"
    write_result(result, [row])
    assert pipeline.write_quality_report(result, 0.9, 0, 30)


def test_stream_audit_detects_early_decode_stop(monkeypatch):
    audit_mod = load_module("stream_integrity", VALIDATION / "stream_integrity.py")

    class FakeCapture:
        def __init__(self, _path):
            self.frames = [np.zeros((2, 2, 3), dtype=np.uint8)]

        def isOpened(self):
            return True

        def get(self, _key):
            return 100

        def read(self):
            return (True, self.frames.pop()) if self.frames else (False, None)

        def release(self):
            pass

    monkeypatch.setattr(audit_mod.cv2, "VideoCapture", FakeCapture)
    audit = audit_mod.StreamAudit(expected_raw_frames=100, expected_source="test")
    frames = list(audit_mod.iter_rgb_frames("cut.mp4", 1, None, audit))
    assert len(frames) == 1
    assert audit.decoded_raw_frames == 1
    assert audit.sampled_frames == 1
    assert audit.decode_complete is False
    assert audit.decode_errors == 1


def test_megaloc_cache_binds_descriptors_to_exact_names_and_hash(tmp_path, monkeypatch):
    deploy = VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy"
    monkeypatch.syspath_prepend(str(deploy))
    cache_io = load_module("megaloc_cache", deploy / "megaloc_cache.py")
    desc = np.eye(3, 4, dtype=np.float32)
    names = ["a.jpg", "b.jpg", "c.jpg"]

    npz = tmp_path / "cache.npz"
    cache_io.write_megaloc_cache(npz, desc, names)
    np.testing.assert_array_equal(cache_io.load_megaloc_cache(npz, names), desc)
    production = load_module("production_tracker_cache_test", deploy / "production_xfeat_tracker.py")
    np.testing.assert_array_equal(production.MegaLocLayer.load_cache(npz, names).ref_desc, desc)
    try:
        cache_io.load_megaloc_cache(npz, list(reversed(names)))
    except ValueError as exc:
        assert "ordered reference names" in str(exc)
    else:
        raise AssertionError("named NPZ was silently reordered")

    disguised = tmp_path / "raw_content.npz"
    with disguised.open("wb") as stream:
        np.save(stream, desc)
    try:
        cache_io.load_megaloc_cache(disguised, names)
    except ValueError as exc:
        assert "raw MegaLoc NPY content" in str(exc)
    else:
        raise AssertionError("raw NPY content disguised with .npz was accepted")

    npy = tmp_path / "cache.npy"
    meta = tmp_path / "cache.json"
    with npy.open("wb") as stream:
        np.save(stream, desc)
    try:
        cache_io.load_megaloc_cache(npy, names)
    except ValueError as exc:
        assert "sidecar" in str(exc)
    else:
        raise AssertionError("unbound NPY cache was accepted")

    cache_io.write_megaloc_cache(npy, desc, names, meta)
    np.testing.assert_array_equal(cache_io.load_megaloc_cache(npy, names, meta), desc)
    sidecar = json.loads(meta.read_text(encoding="utf-8"))
    sidecar["cache_sha256"] = "0" * 64
    meta.write_text(json.dumps(sidecar), encoding="utf-8")
    try:
        cache_io.load_megaloc_cache(npy, names, meta)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered NPY cache was accepted")


def test_megaloc_cache_loads_portable_updated_map_schema(tmp_path, monkeypatch):
    deploy = VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy"
    monkeypatch.syspath_prepend(str(deploy))
    cache_io = load_module("megaloc_cache_portable", deploy / "megaloc_cache.py")
    desc = np.eye(2, 3, dtype=np.float32)
    names = ["a.jpg", "b.jpg"]
    path = tmp_path / "portable.npz"
    np.savez(
        path,
        schema_name=np.array("sfm_system.megaloc_cache"),
        schema_version=np.array(1, dtype=np.int64),
        model=np.array("MegaLoc"),
        input_size=np.array(322, dtype=np.int64),
        desc=desc,
        names=np.asarray(names),
    )

    np.testing.assert_array_equal(cache_io.load_megaloc_cache(path, names), desc)


def test_intrinsics_are_loaded_from_json_without_sparse_model(tmp_path):
    intrinsics = load_module("camera_intrinsics", VALIDATION / "camera_intrinsics.py")
    path = tmp_path / "map_intrinsics.json"
    path.write_text(json.dumps({
        "model": "SIMPLE_RADIAL",
        "cameras": {
            "cam720": {"width": 1280, "height": 720, "params": [934.0, 640.0, 360.0, 0.001]},
            "cam1080": {"width": 1920, "height": 1080, "params": [1401.0, 960.0, 540.0, 0.002]},
        },
        "query_recommended": {"1920x1080": {"use_scene": "cam1080"}},
    }), encoding="utf-8")

    assert intrinsics.load_scaled_simple_radial(path, (1920, 1080)) == [1401.0, 960.0, 540.0, 0.002]
    assert intrinsics.load_scaled_simple_radial(path, (960, 540)) == [700.5, 480.0, 270.0, 0.001]


def test_operator_worker_defaults_to_current_validated_interpreter(monkeypatch):
    app = load_module(
        "flight_operator_interpreter_test",
        VALIDATION.parent / "mission" / "operator_interface" / "flight_operator_app.py",
    )
    monkeypatch.delenv("SFM_LOCALIZER_PYTHON", raising=False)
    assert app.default_worker_python("SFM_LOCALIZER_PYTHON") == sys.executable
    monkeypatch.setenv("SFM_LOCALIZER_PYTHON", "/trusted/python")
    assert app.default_worker_python("SFM_LOCALIZER_PYTHON") == "/trusted/python"


def test_mission_pipeline_defaults_to_current_validated_interpreter(monkeypatch):
    path = VALIDATION.parent / "mission" / "mission_pipeline.py"
    monkeypatch.delenv("SFM_LOCALIZER_PYTHON", raising=False)
    mission = load_module("mission_pipeline_interpreter_current", path)
    assert mission.DEFAULT_PYTHON == sys.executable

    monkeypatch.setenv("SFM_LOCALIZER_PYTHON", "/trusted/python")
    mission = load_module("mission_pipeline_interpreter_override", path)
    assert mission.DEFAULT_PYTHON == "/trusted/python"


def test_manifest_detects_content_and_file_set_changes(tmp_path):
    manifest = load_module("package_manifest", PACKAGE_ROOT / "tools" / "package_manifest.py")
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("b", encoding="utf-8")
    manifest.generate(tmp_path)

    assert manifest.verify(tmp_path) == []
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    assert any("a.txt" in issue for issue in manifest.verify(tmp_path))

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    assert any("extra.txt" in issue for issue in manifest.verify(tmp_path))


def test_bundle_hash_preflight_rejects_mismatch(tmp_path):
    integrity = load_module(
        "artifact_integrity",
        VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy" / "artifact_integrity.py",
    )
    bundle = tmp_path / "bundle.pt"
    bundle.write_bytes(b"trusted fixture")
    expected = hashlib.sha256(bundle.read_bytes()).hexdigest()

    integrity.verify_sha256(bundle, expected)
    bundle.write_bytes(b"tampered")
    try:
        integrity.verify_sha256(bundle, expected)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered bundle was accepted")


def test_football_field_bundle_hash_is_trusted():
    integrity = load_module(
        "artifact_integrity_football",
        VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy" / "artifact_integrity.py",
    )

    assert integrity.KNOWN_SHA256["football_field_reloc_map_xfeat_tri.pt"] == (
        "1c1774318a71ac29870f78ccb67001150edd934141e4aa288765e745e72db46f"
    )


def test_production_flight_uses_calibrated_720p_full_opencv_camera(monkeypatch):
    deploy = VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy"
    monkeypatch.syspath_prepend(str(deploy))
    flight = load_module("path_follow_flight_football", deploy / "path_follow_flight.py")

    assert flight.CAM_720 == (
        "FULL_OPENCV",
        1280,
        720,
        [
            960.4853099760471, 958.1961747147875,
            670.8167651412149, 358.7191813450141,
            -0.016359355216362784, 0.256336300878371,
            -0.006099082030819077, 0.019509803298460405,
            -0.1198628127364991, 0.0, 0.0, 0.0,
        ],
    )
    cfg = flight.production_config()
    expected = {
        "boot_global_topk": 30,
        "lost_global_topk": 30,
        "weak_global_topk": 0,
        "local_topk": 5,
        "weak_local_topk": 8,
        "near_pool": 24,
        "covis_per_ref": 20,
        "radius": 0.8,
        "max_yaw_diff_deg": 90.0,
        "xfeat_topk_track": 1700,
        "xfeat_topk_acquire": 2048,
        "matcher_mode": "nn_then_lg",
        "acquire_matcher_mode": "lighterglue",
        "nn_min_score": 0.85,
        "adaptive_first_topk": 3,
        "adaptive_accept_inliers": 100,
        "adaptive_accept_reproj": 3.5,
        "temporal_cache_enabled": True,
        "temporal_cache_seed_mode": "full_ref",
        "temporal_cache_min_anchors": 80,
        "temporal_cache_max_anchors": 2048,
        "temporal_cache_max_age": 2,
        "temporal_cache_min_score": 0.85,
        "temporal_cache_seed_min_inliers": 150,
        "temporal_cache_seed_max_reproj": 3.5,
        "flow_enabled": False,
    }
    assert {name: getattr(cfg, name) for name in expected} == expected


def test_stream_benchmark_loads_full_opencv_calibration(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(VALIDATION))
    benchmark = load_module(
        "benchmark_full_opencv_test", VALIDATION / "benchmark_production_stream.py")
    calibration = tmp_path / "intrinsics.json"
    calibration.write_text(json.dumps({
        "image_width": 1280,
        "image_height": 720,
        "K": [
            [960.4853099760471, 0.0, 670.8167651412149],
            [0.0, 958.1961747147875, 358.7191813450141],
            [0.0, 0.0, 1.0],
        ],
        "dist": [
            -0.016359355216362784, 0.256336300878371,
            -0.006099082030819077, 0.019509803298460405,
            -0.1198628127364991,
        ],
    }), encoding="utf-8")

    cam = benchmark.load_query_camera(calibration, (1280, 720), (1280, 720))

    assert cam.model == "FULL_OPENCV"
    assert (cam.width, cam.height) == (1280, 720)
    assert cam.params == [
        960.4853099760471, 958.1961747147875,
        670.8167651412149, 358.7191813450141,
        -0.016359355216362784, 0.256336300878371,
        -0.006099082030819077, 0.019509803298460405,
        -0.1198628127364991, 0.0, 0.0, 0.0,
    ]
    assert benchmark.intrinsics_check(cam)["matches_production"] is True


def test_stream_benchmark_reads_video_stride_without_materializing_frames(
    tmp_path, monkeypatch,
):
    monkeypatch.syspath_prepend(str(VALIDATION))
    benchmark = load_module(
        "benchmark_video_input_test", VALIDATION / "benchmark_production_stream.py")
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 12))
    assert writer.isOpened()
    for value in (10, 20, 30, 40):
        writer.write(np.full((12, 16, 3), value, np.uint8))
    writer.release()

    rows = list(benchmark.iter_video_frames(video, width=8, stride=2, limit=0))

    assert [row[0] for row in rows] == ["frame_000001.jpg", "frame_000003.jpg"]
    assert all(row[1].shape == (6, 8, 3) for row in rows)
    assert [row[5] for row in rows] == pytest.approx([0.0, 0.2])
    assert benchmark.video_frame_rate(video) == pytest.approx(10.0)
    assert benchmark.video_frame_count(video, stride=2, limit=1) == 1


def test_verified_bundle_uses_restricted_schema(tmp_path, monkeypatch):
    deploy = VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy"
    monkeypatch.syspath_prepend(str(deploy))
    reloc = load_module("reloc_localizer_schema_test", deploy / "reloc_localizer_xfeat.py")
    name = "ref.jpg"
    payload = {
        "meta": {"bundle_vpr": "megaloc"},
        "ref_names": [name],
        "ref_global": np.ones((1, 4), dtype=np.float32),
        "refs": {
            name: {
                "feats": {
                    "keypoints": torch.zeros((2, 2), dtype=torch.float32),
                    "scores": torch.ones(2, dtype=torch.float32),
                    "descriptors": torch.zeros((2, 64), dtype=torch.float32),
                    "image_size": (1280, 720),
                },
                "xyz": np.zeros((2, 3), dtype=np.float32),
            },
        },
    }
    bundle = tmp_path / "fixture.pt"
    torch.save(payload, bundle)
    expected = hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert reloc.load_verified_bundle(bundle, expected)["ref_names"] == [name]

    payload["unexpected"] = "rejected"
    torch.save(payload, bundle)
    expected = hashlib.sha256(bundle.read_bytes()).hexdigest()
    try:
        reloc.load_verified_bundle(bundle, expected)
    except ValueError as exc:
        assert "unexpected relocation bundle keys" in str(exc)
    else:
        raise AssertionError("bundle with an unrestricted schema was accepted")


def test_bundle_schema_rejects_nonfinite_features_bad_dtype_and_covis(tmp_path, monkeypatch):
    deploy = VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy"
    monkeypatch.syspath_prepend(str(deploy))
    reloc = load_module("reloc_localizer_malicious_test", deploy / "reloc_localizer_xfeat.py")
    name = "ref.jpg"

    def payload():
        return {
            "meta": {"bundle_vpr": "megaloc"},
            "ref_names": [name],
            "ref_global": np.ones((1, 4), dtype=np.float32),
            "refs": {
                name: {
                    "feats": {
                        "keypoints": torch.zeros((2, 2), dtype=torch.float32),
                        "scores": torch.ones(2, dtype=torch.float32),
                        "descriptors": torch.zeros((2, 64), dtype=torch.float32),
                        "image_size": (1280, 720),
                    },
                    "xyz": np.zeros((2, 3), dtype=np.float32),
                },
            },
            "covis": {name: []},
        }

    bad_payloads = []
    bad = payload()
    bad["refs"][name]["feats"]["descriptors"][0, 0] = float("nan")
    bad_payloads.append((bad, "non-finite"))
    bad = payload()
    bad["refs"][name]["feats"]["descriptors"] = torch.zeros((2, 64), dtype=torch.int32)
    bad_payloads.append((bad, "dtype"))
    bad = payload()
    bad["covis"][name] = [3]
    bad_payloads.append((bad, "covis"))

    for index, (bad, message) in enumerate(bad_payloads):
        bundle = tmp_path / f"bad_{index}.pt"
        torch.save(bad, bundle)
        expected = hashlib.sha256(bundle.read_bytes()).hexdigest()
        try:
            reloc.load_verified_bundle(bundle, expected)
        except ValueError as exc:
            assert message in str(exc).lower()
        else:
            raise AssertionError(f"malicious bundle {index} was accepted")
