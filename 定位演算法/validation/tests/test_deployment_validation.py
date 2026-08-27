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
PACKAGE_ROOT = Path(__file__).resolve().parents[3]


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


def test_stream_audit_rejects_declared_and_decodable_count_disagreement():
    audit_mod = load_module("stream_integrity_count_mismatch", VALIDATION / "stream_integrity.py")
    audit = audit_mod.StreamAudit(
        expected_raw_frames=1,
        expected_source="ffprobe_nb_read_frames",
        capture_opened=True,
        reported_raw_frames=2,
        decoded_raw_frames=1,
        sampled_frames=1,
    )

    audit.finish()

    assert audit.decode_complete is False
    assert audit.decode_errors == 1


def test_site_replay_full_run_fails_closed_on_incomplete_decode():
    benchmark = load_module(
        "benchmark_edm_site_replay_exit",
        VALIDATION / "benchmark_edm_site_replay.py",
    )
    audit_mod = load_module("stream_integrity_benchmark_exit", VALIDATION / "stream_integrity.py")
    audit = audit_mod.StreamAudit(decode_complete=False, decode_errors=1)

    assert benchmark._result_exit_code(1, audit, all_frames_requested=True) == 3
    assert benchmark._result_exit_code(
        1,
        audit,
        all_frames_requested=True,
        decode_accepted=True,
    ) == 0
    assert benchmark._result_exit_code(1, audit, all_frames_requested=False) == 0
    assert benchmark._result_exit_code(0, audit, all_frames_requested=True) == 2


def test_site_replay_camera_override_requires_valid_pinhole_params():
    benchmark = load_module(
        "benchmark_edm_site_replay_camera_override",
        VALIDATION / "benchmark_edm_site_replay.py",
    )

    assert benchmark.parse_camera_params("960.5,958.2,670.8,358.7") == [
        960.5,
        958.2,
        670.8,
        358.7,
    ]
    with pytest.raises(Exception, match="four finite numbers"):
        benchmark.parse_camera_params("960,958,nan,359")
    with pytest.raises(Exception, match="focal lengths must be positive"):
        benchmark.parse_camera_params("0,958,671,359")


def test_site_replay_quality_gate_rejects_baseline_regressions():
    benchmark = load_module(
        "benchmark_edm_site_replay_quality",
        VALIDATION / "benchmark_edm_site_replay.py",
    )
    baseline = {
        "thresholds": {
            "frames": 100,
            "min_successes": 80,
            "min_track": 90,
            "max_lost": 2,
            "min_inliers_p50": 50.0,
            "min_inliers_p95": 70.0,
            "max_reproj_rms_p95": 3.0,
            "max_limited_jump_unconfirmed": 10,
        }
    }
    summary = {
        "frames": 100,
        "successes": 79,
        "state_counts": {"TRACK": 89, "LOST": 3},
        "rejection_counts": {"limited_jump_unconfirmed": 11},
        "inliers": {"p50": 49.0, "p95": 69.0},
        "reproj_rms": {"p95": 3.1},
    }

    failures = benchmark.evaluate_quality(summary, baseline)

    assert len(failures) == 7
    assert any("successes" in item for item in failures)
    assert any("limited_jump_unconfirmed" in item for item in failures)


def test_site_replay_quality_gate_accepts_equal_or_better_result():
    benchmark = load_module(
        "benchmark_edm_site_replay_quality_pass",
        VALIDATION / "benchmark_edm_site_replay.py",
    )
    baseline = {
        "thresholds": {
            "frames": 100,
            "min_successes": 80,
            "min_track": 90,
            "max_lost": 2,
            "min_inliers_p50": 50.0,
            "min_inliers_p95": 70.0,
            "max_reproj_rms_p95": 3.0,
            "max_limited_jump_unconfirmed": 10,
        }
    }
    summary = {
        "frames": 100,
        "successes": 81,
        "state_counts": {"TRACK": 91, "LOST": 1},
        "rejection_counts": {"limited_jump_unconfirmed": 9},
        "inliers": {"p50": 51.0, "p95": 71.0},
        "reproj_rms": {"p95": 2.9},
    }

    assert benchmark.evaluate_quality(summary, baseline) == []


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
        PACKAGE_ROOT / "控制介面程式" / "operator_interface" / "flight_operator_app.py",
    )
    monkeypatch.delenv("SFM_LOCALIZER_PYTHON", raising=False)
    assert app.default_worker_python("SFM_LOCALIZER_PYTHON") == sys.executable
    monkeypatch.setenv("SFM_LOCALIZER_PYTHON", "/trusted/python")
    assert app.default_worker_python("SFM_LOCALIZER_PYTHON") == "/trusted/python"


def test_mission_pipeline_defaults_to_current_validated_interpreter(monkeypatch):
    path = PACKAGE_ROOT / "控制介面程式" / "mission_pipeline.py"
    monkeypatch.delenv("SFM_LOCALIZER_PYTHON", raising=False)
    mission = load_module("mission_pipeline_interpreter_current", path)
    assert mission.DEFAULT_PYTHON == sys.executable

    monkeypatch.setenv("SFM_LOCALIZER_PYTHON", "/trusted/python")
    mission = load_module("mission_pipeline_interpreter_override", path)
    assert mission.DEFAULT_PYTHON == "/trusted/python"


def test_manifest_detects_content_and_file_set_changes(tmp_path):
    manifest_path = PACKAGE_ROOT / "tools" / "package_manifest.py"
    manifest = load_module("package_manifest", manifest_path)
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


def test_edm_checkpoint_hash_is_trusted():
    integrity = load_module(
        "artifact_integrity_checkpoint",
        VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy" / "artifact_integrity.py",
    )

    assert integrity.KNOWN_SHA256["edm_outdoor.ckpt"] == (
        "f686bebdd9705bf6918621a1a83695f83d698cbd8c3eed932847fe3678d13a97"
    )


def test_production_flight_uses_calibrated_720p_full_opencv_camera(monkeypatch):
    deploy = VALIDATION.parent / "deploy_code" / "sfm_glomap_deploy"
    flight_control = VALIDATION.parent / "flight_control"
    monkeypatch.syspath_prepend(str(deploy))
    monkeypatch.syspath_prepend(str(flight_control))
    flight = load_module(
        "path_follow_flight_camera_calibration", flight_control / "path_follow_flight.py"
    )

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
        "matcher_mode": "lighterglue",
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


def test_localizer_factory_refuses_unverified_default_thresholds() -> None:
    """No profile means no SHA verification AND target_site-scaled defaults.

    The EDMConfig dataclass defaults (radius 0.8 / max_jump 2.0 / ...) are the
    target_site scale, so another site silently inherits gates several times too
    loose. Falling back to them must be an explicit, deliberate opt-in.
    """
    deploy = PACKAGE_ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    import production_localizer_factory as factory

    with pytest.raises(ValueError, match="production localizer profile is required"):
        factory.build_production_localizer(
            backend="edm",
            bundle="/nonexistent/bundle.pt",
            frame_source=lambda: None,
            camera_tuple=("PINHOLE", 1280, 720, [900.0, 900.0, 640.0, 360.0]),
            production_profile=None,
        )

    # The opt-in still exists for deliberate non-flight experiments: it must get
    # past the profile gate (and fail later, on the missing bundle).
    with pytest.raises(Exception) as excinfo:
        factory.build_production_localizer(
            backend="edm",
            bundle="/nonexistent/bundle.pt",
            frame_source=lambda: None,
            camera_tuple=("PINHOLE", 1280, 720, [900.0, 900.0, 640.0, 360.0]),
            production_profile=None,
            allow_profile_defaults=True,
        )
    assert "production localizer profile is required" not in str(excinfo.value)


def _official_edm_profile() -> dict:
    deploy = PACKAGE_ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    from edm_profile import load_edm_production_profile

    return load_edm_production_profile(
        PACKAGE_ROOT / "定位演算法" / "configs" / "edm_production_profile.json"
    )


def test_legacy_profile_uses_one_shot_lost_retrieval_default() -> None:
    from edm_profile import apply_edm_tracker_profile
    from production_edm_tracker import EDMConfig

    profile = _official_edm_profile()
    assert "lost_global_retrieval_interval" not in profile["tracker"]
    cfg = EDMConfig(lost_global_retrieval_interval=9)

    apply_edm_tracker_profile(cfg, profile)

    assert cfg.lost_global_retrieval_interval == 0


def test_absent_or_off_reposed_profile_imports_no_optional_package() -> None:
    before = {name for name in sys.modules if name.split(".", 1)[0] in {"moge", "poselib"}}
    profile = _official_edm_profile()
    assert "reposed" not in profile
    deploy = PACKAGE_ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    from production_localizer_factory import build_reposed_motion_validator

    validator, mode = build_reposed_motion_validator(profile, object(), object())
    assert validator is None
    assert mode == "off"
    after = {name for name in sys.modules if name.split(".", 1)[0] in {"moge", "poselib"}}
    assert after == before


def test_unknown_or_missing_reposed_keys_fail(tmp_path: Path) -> None:
    deploy = PACKAGE_ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    from edm_profile import load_edm_production_profile

    raw = _official_edm_profile()
    raw["reposed"] = {
        "mode": "off",
        "model_path": "missing.pt",
        "model_sha256": "0" * 64,
        "num_tokens": 1200,
        "max_matches": 1200,
        "min_inliers": 30,
        "max_rotation_delta_deg": 3.0,
        "max_translation_direction_delta_deg": 15.0,
        "extra": True,
    }
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_edm_production_profile(path)

    raw["reposed"].pop("extra")
    raw["reposed"].pop("min_inliers")
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_edm_production_profile(path)


def test_enabled_reposed_hash_mismatch_fails_before_model_allocation(tmp_path: Path, monkeypatch) -> None:
    deploy = PACKAGE_ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    import reposed_motion_validator as validator_module
    from edm_profile import load_edm_production_profile

    model = tmp_path / "model.pt"
    model.write_bytes(b"not-the-real-weights")
    raw = _official_edm_profile()
    raw["reposed"] = {
        "mode": "shadow",
        "model_path": str(model),
        "model_sha256": "0" * 64,
        "num_tokens": 1200,
        "max_matches": 1200,
        "min_inliers": 30,
        "max_rotation_delta_deg": 3.0,
        "max_translation_direction_delta_deg": 15.0,
    }
    path = tmp_path / "bad_hash.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise AssertionError("model allocation must not run after a hash mismatch")

    monkeypatch.setattr(validator_module.RePoseDMotionValidator, "_ensure_runtime", _boom)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_edm_production_profile(path)


def test_factory_passes_one_validated_wrapper_to_the_tracker(tmp_path: Path) -> None:
    deploy = PACKAGE_ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
    if str(deploy) not in sys.path:
        sys.path.insert(0, str(deploy))
    from edm_localizer_adapter import EDMTrackerAdapter
    from production_edm_tracker import EDMConfig
    from production_localizer_factory import build_reposed_motion_validator

    model = tmp_path / "model.pt"
    payload = b"fixture-weights"
    model.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    camera = type(
        "Cam",
        (),
        {"model": "PINHOLE", "width": 1280, "height": 720, "params": [900.0, 900.0, 640.0, 360.0]},
    )()
    profile = {
        "reposed": {
            "mode": "shadow",
            "model_path": str(model),
            "model_sha256": digest,
            "num_tokens": 1200,
            "max_matches": 1200,
            "min_inliers": 30,
            "max_rotation_delta_deg": 3.0,
            "max_translation_direction_delta_deg": 15.0,
        }
    }
    validator, mode = build_reposed_motion_validator(profile, camera, object(), source=model)
    assert mode == "shadow"
    assert validator is not None
    created = []

    class _Map:
        ref_centers = None
        ref_yaws = None
        ref_names = ["ref0"]

    original = EDMTrackerAdapter.__init__

    def wrapped(self, *args, **kwargs):
        created.append(kwargs.get("motion_validator"))
        self.trk = type("T", (), {"cfg": EDMConfig(), "st": None})()
        self.map = args[0]
        self.cfg = self.trk.cfg
        self.frame_source = kwargs.get("frame_source")
        self.map_frame = kwargs.get("map_frame")
        self.state = None
        self.temporal_cache = None
        self._last_info = {}

    EDMTrackerAdapter.__init__ = wrapped
    try:
        adapter = EDMTrackerAdapter(
            _Map(),
            camera,
            motion_validator=validator,
            motion_validation_mode=mode,
        )
    finally:
        EDMTrackerAdapter.__init__ = original
    assert created == [validator]
    assert adapter.trk is not None
