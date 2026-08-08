from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "定位演算法/validation/benchmark_edm_site_replay.py"
SPEC = importlib.util.spec_from_file_location("benchmark_edm_site_replay_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

STREAM_SPEC = importlib.util.spec_from_file_location(
    "benchmark_production_stream_identity",
    ROOT / "定位演算法/validation/benchmark_production_stream.py",
)
assert STREAM_SPEC is not None and STREAM_SPEC.loader is not None
stream_module = importlib.util.module_from_spec(STREAM_SPEC)
sys.modules[STREAM_SPEC.name] = stream_module
STREAM_SPEC.loader.exec_module(stream_module)


def _baseline() -> dict:
    return {
        "video_sha256": "a" * 64,
        "site_profile_sha256": "b" * 64,
        "bundle_sha256": "c" * 64,
        "localizer_profile_sha256": "d" * 64,
        "camera": {"model": "PINHOLE", "width": 1280, "height": 720, "params": [1.0]},
    }


def test_baseline_identity_accepts_all_bound_inputs() -> None:
    baseline = _baseline()
    assert module.baseline_identity_failures(
        baseline,
        video_sha256="a" * 64,
        site_profile_sha256="b" * 64,
        bundle_sha256="c" * 64,
        localizer_profile_sha256="d" * 64,
        camera=baseline["camera"],
    ) == []


def test_baseline_identity_rejects_camera_and_asset_mismatch() -> None:
    failures = module.baseline_identity_failures(
        _baseline(),
        video_sha256="e" * 64,
        site_profile_sha256="f" * 64,
        bundle_sha256="g" * 64,
        localizer_profile_sha256="h" * 64,
        camera={"model": "FULL_OPENCV", "width": 640, "height": 360, "params": [2.0]},
    )
    assert any("video_sha256 mismatch" in item for item in failures)
    assert any("camera model mismatch" in item for item in failures)


def test_baseline_identity_rejects_legacy_unbound_baseline() -> None:
    failures = module.baseline_identity_failures(
        {},
        video_sha256="a" * 64,
        site_profile_sha256="b" * 64,
        bundle_sha256="c" * 64,
        localizer_profile_sha256="d" * 64,
        camera={"model": "PINHOLE", "width": 1280, "height": 720, "params": [1.0]},
    )
    assert "baseline is missing video_sha256" in failures
    assert "baseline is missing camera identity" in failures


def test_production_stream_camera_gate_fails_closed() -> None:
    class Camera:
        model = "SIMPLE_RADIAL"
        width = 1280
        height = 720
        params = [1.0, 2.0, 3.0, 4.0]

    report = stream_module.intrinsics_check(Camera())
    assert report["matches_production"] is False
    try:
        stream_module.require_camera_match(Camera())
    except ValueError as exc:
        assert "refusing the run" in str(exc)
    else:
        raise AssertionError("camera mismatch was accepted")
