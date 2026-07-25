from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import flight_operator_app as app
import mission_pipeline
from live_localizer_worker import load_edm_production_profile
from site_profile import load_site_profile


def _write_profile(root: Path, *, missing_bundle: bool = False) -> Path:
    assets = root / "assets"
    assets.mkdir()
    for name in ("map.ply", "route.json", "poles.json", "cache.npy", "landmarks.npz"):
        (assets / name).write_bytes(b"x")
    if not missing_bundle:
        (assets / "bundle.pt").write_bytes(b"x")
    profile = root / "site.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site_id": "test_site",
                "display_name": "Test Site",
                "assets": {
                    "map_ply": "assets/map.ply",
                    "route_json": "assets/route.json",
                    "poles_json": "assets/poles.json",
                    "localization_bundle": "assets/bundle.pt",
                    "megaloc_cache": "assets/cache.npy",
                    "track_landmarks": "assets/landmarks.npz",
                },
            }
        ),
        encoding="utf-8",
    )
    return profile


def _args(profile: Path, **overrides):
    values = {
        "site_profile": str(profile),
        "map_ply": None,
        "route_json": None,
        "bundle": None,
        "megaloc_cache": None,
        "track_landmarks": None,
        "localizer_backend": None,
        "localizer_deploy_dir": None,
        "localizer_profile": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_site_profile_resolves_all_paths_relative_to_profile(tmp_path: Path) -> None:
    profile = load_site_profile(_write_profile(tmp_path))

    assert profile.site_id == "test_site"
    assert profile.map_ply == (tmp_path / "assets/map.ply").resolve()
    assert profile.route_json == (tmp_path / "assets/route.json").resolve()
    assert profile.localization_bundle == (tmp_path / "assets/bundle.pt").resolve()
    assert profile.megaloc_cache == (tmp_path / "assets/cache.npy").resolve()
    assert profile.track_landmarks == (tmp_path / "assets/landmarks.npz").resolve()
    assert profile.poles_json == (tmp_path / "assets/poles.json").resolve()


def test_site_profile_loads_query_camera_calibration(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["query_camera"] = {
        "model": "SIMPLE_RADIAL",
        "width": 1280,
        "height": 720,
        "params": [934.139423, 640.0, 360.0, 0.001061],
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_site_profile(profile_path)

    assert profile.query_camera is not None
    assert profile.query_camera.model == "SIMPLE_RADIAL"
    assert profile.query_camera.width == 1280
    assert profile.query_camera.height == 720
    assert profile.query_camera.params == (934.139423, 640.0, 360.0, 0.001061)


def test_site_profile_loads_edm_deployment_directory(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    deploy_dir = tmp_path / "edm-deploy"
    deploy_dir.mkdir()
    runtime_profile = deploy_dir / "production.json"
    runtime_profile.write_text("{}", encoding="utf-8")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer"] = "edm"
    raw["localizer_deploy_dir"] = "edm-deploy"
    raw["localizer_profile"] = "edm-deploy/production.json"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_site_profile(profile_path)

    assert profile.localizer_deploy_dir == deploy_dir.resolve()
    assert profile.localizer_profile == runtime_profile.resolve()


@pytest.mark.parametrize(
    "query_camera",
    [
        {"model": "", "width": 1280, "height": 720, "params": [1.0]},
        {"model": "SIMPLE_RADIAL", "width": 0, "height": 720, "params": [1.0]},
        {"model": "SIMPLE_RADIAL", "width": 1280, "height": 720, "params": []},
    ],
)
def test_site_profile_rejects_invalid_query_camera(
    tmp_path: Path, query_camera: dict,
) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["query_camera"] = query_camera
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="query_camera"):
        load_site_profile(profile_path)


def test_site_profile_fails_before_start_when_an_asset_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="localization_bundle"):
        load_site_profile(_write_profile(tmp_path, missing_bundle=True))


def test_operator_profile_switches_all_localization_assets_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in app._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    profile_path = _write_profile(tmp_path)
    args = _args(profile_path)

    selected = app.resolve_operator_site_assets(args, argparse.ArgumentParser())

    assert selected is not None
    assert args.map_ply == str((tmp_path / "assets/map.ply").resolve())
    assert args.route_json == str((tmp_path / "assets/route.json").resolve())
    assert args.bundle == str((tmp_path / "assets/bundle.pt").resolve())
    assert args.megaloc_cache == str((tmp_path / "assets/cache.npy").resolve())
    assert args.track_landmarks == str((tmp_path / "assets/landmarks.npz").resolve())
    assert args.localizer_deploy_dir == ""
    assert args.localizer_profile == ""


def test_operator_rejects_profile_mixed_with_per_asset_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in app._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    args = _args(_write_profile(tmp_path), bundle="other.pt")

    with pytest.raises(SystemExit):
        app.resolve_operator_site_assets(args, argparse.ArgumentParser())


def test_mission_pipeline_uses_the_same_profile_route_and_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in mission_pipeline._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(
        site_profile=str(_write_profile(tmp_path)),
        bundle=None,
        megaloc_cache=None,
        track_landmarks=None,
        path_json=None,
        poles_json=None,
        safezone_dir=None,
    )

    selected = mission_pipeline.resolve_mission_site_assets(
        args, argparse.ArgumentParser(), mode="dry-run"
    )

    assert selected is not None
    assert args.map_ply == str((tmp_path / "assets/map.ply").resolve())
    assert args.path_json == str((tmp_path / "assets/route.json").resolve())
    assert args.poles_json == str((tmp_path / "assets/poles.json").resolve())
    assert args.bundle == str((tmp_path / "assets/bundle.pt").resolve())


def test_example_edm_profile_uses_portable_relative_paths() -> None:
    path = Path(__file__).resolve().parents[1] / "site_profiles" / "example_site_edm.json"
    profile = load_site_profile(path, validate_files=False)

    assert profile.site_id == "your_site_edm"
    assert profile.localizer == "edm"
    assert profile.route_json is None
    assert profile.map_ply.name == "your_site.ply"
    assert profile.localization_bundle.name == "your_site_reloc_map_edm.pt"
    assert profile.localizer_deploy_dir is not None
    assert profile.localizer_profile is not None


def test_final_edm_profile_pins_the_validated_runtime_parameters() -> None:
    profile = load_edm_production_profile(
        Path(__file__).resolve().parents[2]
        / "定位演算法/configs/edm_production_profile.json"
    )

    assert profile["matcher"] == {
        "coarse_topk": 3225,
        "mconf_thr": 0.2,
        "fp16": True,
        "input_size": [1024, 576],
        "reference_cache_size": 32,
    }
    tracker = profile["tracker"]
    assert (tracker["local_topk"], tracker["weak_local_topk"], tracker["lost_local_topk"]) == (1, 3, 5)
    assert (tracker["boot_global_topk"], tracker["match_batch_size"]) == (10, 2)
    assert tracker["acquire_initial_topk"] == 2
    assert (tracker["lost_local_grace_frames"], tracker["recovery_bank_size"], tracker["recovery_scan_topk"]) == (12, 192, 2)
    assert tracker["use_temporal_reference"] is False
    assert (tracker["max_reproj_error_acquire"], tracker["max_reproj_error_track"], tracker["pnp_ransac_max_error"]) == (5.0, 6.0, 5.0)
    assert tracker["prediction_max_dt"] == 0.25


def test_safety_command_bypasses_invalid_site_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    safety_file = tmp_path / "safety.cmd"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mission_pipeline.py",
            "--mode",
            "safety-hover",
            "--site-profile",
            str(tmp_path / "missing-profile.json"),
            "--safety-file",
            str(safety_file),
        ],
    )

    mission_pipeline.main()

    assert safety_file.read_text(encoding="utf-8") == "hover\n"


def test_localizer_explicitly_disables_inherited_megaloc_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_worker_init(self, cmd, *_args, **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.setattr(app.LiveWorkerClient, "__init__", fake_worker_init)
    app.LiveLocalizerClient(
        Path("worker.py"),
        sys.executable,
        1280,
        720,
        Path("bundle.pt"),
        megaloc_cache="",
        localizer_backend="xfeat",
    )

    index = captured["cmd"].index("--megaloc-cache")
    assert captured["cmd"][index + 1] == ""
    assert captured["cmd"][captured["cmd"].index("--localizer-backend") + 1] == "xfeat"


def test_edm_localizer_passes_profile_deployment_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_worker_init(self, cmd, *_args, **_kwargs):
        captured["cmd"] = cmd

    monkeypatch.setattr(app.LiveWorkerClient, "__init__", fake_worker_init)
    app.LiveLocalizerClient(
        Path("worker.py"), sys.executable, 1280, 720, Path("bundle_edm.pt"),
        localizer_backend="edm", localizer_deploy_dir="/tmp/edm-deploy",
        localizer_profile="/tmp/edm-production.json",
    )

    index = captured["cmd"].index("--deploy-dir")
    assert captured["cmd"][index + 1] == "/tmp/edm-deploy"
    matcher_index = captured["cmd"].index("--edm-matcher")
    assert captured["cmd"][matcher_index + 1] == "torch"
    profile_index = captured["cmd"].index("--production-profile")
    assert captured["cmd"][profile_index + 1] == "/tmp/edm-production.json"
    assert "--edm-onnx" not in captured["cmd"]


def test_localizer_backend_auto_detects_edm_bundle_name() -> None:
    assert app.resolve_localizer_backend("auto", Path("your_site_reloc_map_edm.pt")) == "edm"
    assert app.resolve_localizer_backend("auto", Path("current_reloc_map_updated_v3.pt")) == "xfeat"
    assert app.resolve_localizer_backend("edm", Path("anything.pt")) == "edm"
