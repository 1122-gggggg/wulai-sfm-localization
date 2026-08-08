from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import flight_operator_app as app
import mission_pipeline
from live_localizer_worker import (
    apply_edm_tracker_profile,
    load_edm_production_profile,
    resolve_query_camera_override,
)
from edm_localizer_adapter import production_edm_config
from production_localizer_factory import validate_camera_tuple
from site_profile import (
    flight_readiness_errors,
    load_hardware_approval_receipt,
    load_site_profile,
)


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
    assert profile.localizer == "edm"
    assert profile.map_ply == (tmp_path / "assets/map.ply").resolve()
    assert profile.route_json == (tmp_path / "assets/route.json").resolve()
    assert profile.localization_bundle == (tmp_path / "assets/bundle.pt").resolve()
    assert profile.megaloc_cache == (tmp_path / "assets/cache.npy").resolve()
    assert profile.track_landmarks == (tmp_path / "assets/landmarks.npz").resolve()
    assert profile.poles_json == (tmp_path / "assets/poles.json").resolve()


def test_site_profile_rejects_non_edm_localizer(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer"] = "xfeat"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must be 'edm'"):
        load_site_profile(profile_path)


@pytest.mark.parametrize(
    "display_name",
    [
        "River's Edge $(touch pwned)",
        'both " and \' $(id)',
        "back\\slash",
        "tick `id`",
        "two\nlines",
    ],
)
def test_site_profile_rejects_shell_unsafe_display_name(
    tmp_path: Path, display_name: str
) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["display_name"] = display_name
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="display_name"):
        load_site_profile(profile_path)


@pytest.mark.parametrize(
    "display_name",
    [
        "烏來（目標場域）EDM v1 — 2026-07-19 驗證",
        "足球場 EDM",
        "Your site EDM profile",
        "Site A (north) [v2]",
    ],
)
def test_site_profile_keeps_ordinary_display_names(
    tmp_path: Path, display_name: str
) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["display_name"] = display_name
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_site_profile(profile_path).display_name == display_name


def test_site_profile_schema_version_rejects_boolean_alias(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = True
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_site_profile(profile_path)


def test_schema_v2_scale_free_contract_has_no_map_scale(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["flight"] = {
        "approved": False,
        "coordinate_frame_id": "test-reconstruction-v1",
        "route_clearance_approved": False,
        "approval_note": "Ground localization only.",
        "controller": {
            "model": "scale_free_direction_speed_guard_v1",
            "speed_limit_mps": 0.30,
            "pose_max_age_ms": 500,
            "speed_max_age_ms": 500,
            "command_ttl_ms": 150,
            "yaw_tolerance_deg": 3.0,
            "horizontal_axes": [0, 1],
            "vertical_axis": 2,
            "camera_to_body_yaw_deg": 0.0,
            "body_right_sign": -1,
            "lookahead_map_units": 0.8,
            "rejoin_tolerance_map_units": 0.4,
            "arrival_tolerance_map_units": 0.2,
            "inspect_radius_map_units": 0.5,
            "inspect_resume_margin_map_units": 0.2,
            "max_pose_jump_map_units": 1.0,
            "max_route_deviation_map_units": 1.5,
            "progress_jump_slack_map_units": 0.5,
            "max_progress_regression_map_units": 0.1,
            "segment_window": 2,
            "progress_speed_factor": 2.0,
            "inspect_waypoints": [],
        },
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_site_profile(profile_path)

    assert profile.schema_version == 2
    assert profile.flight is not None
    assert not hasattr(profile.flight, "map_units_per_meter")
    assert profile.flight.controller is not None
    assert profile.flight.controller.speed_limit_mps == pytest.approx(0.30)
    assert profile.flight.controller.command_ttl_ms == pytest.approx(150)
    assert "flight.approved is false" in flight_readiness_errors(profile)
    assert all("map_units_per_meter" not in item for item in flight_readiness_errors(profile))


def test_schema_v2_rejects_metric_map_scale_field(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["flight"] = {
        "approved": False,
        "map_units_per_meter": 1.0,
        "route_clearance_approved": False,
        "approval_note": "must be rejected",
        "controller": None,
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="map_units_per_meter"):
        load_site_profile(profile_path)


def test_legacy_schema_v1_is_localization_only(tmp_path: Path) -> None:
    profile = load_site_profile(_write_profile(tmp_path))

    assert any(
        "schema v1 is localization-only" in error
        for error in flight_readiness_errors(profile)
    )


def test_hardware_approval_receipt_is_hash_verified(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    receipt_path = tmp_path / "hardware-approval.json"
    receipt_path.write_text(json.dumps({
        "schema": "anafi-hardware-approval/v1",
        "approved": True,
        "aircraft_product": "ANAFI 4K",
        "controller_product": "SkyController 3",
        "aircraft_firmware_versions": ["1.8.2"],
        "controller_firmware_versions": ["1.8.2"],
        "olympe_versions": ["8.4.0"],
        "approval_note": "Props-off and ground checks recorded.",
    }), encoding="utf-8")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["hardware_approval"] = {
        "receipt": receipt_path.name,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_site_profile(profile_path)
    receipt = load_hardware_approval_receipt(profile.hardware_approval)

    assert receipt.approved is True
    assert receipt.aircraft_firmware_versions == ("1.8.2",)
    assert receipt.controller_firmware_versions == ("1.8.2",)
    assert receipt.olympe_versions == ("8.4.0",)

    receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_hardware_approval_receipt(profile.hardware_approval)


def test_all_shipped_profiles_are_scale_free_and_unapproved() -> None:
    profiles_dir = Path(__file__).resolve().parents[1] / "site_profiles"
    for path in profiles_dir.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == 2, path
        assert "map_units_per_meter" not in raw.get("flight", {}), path
        assert raw["flight"]["approved"] is False, path
        profile = load_site_profile(path, validate_files=False)
        assert profile.schema_version == 2


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


def test_site_profile_loads_explicit_glomap_coordinate_frame(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["coordinate_frame"] = {
        "id": "test-reconstruction-v1",
        "convention": "glomap",
        "horizontal_axes": ["x", "z"],
        "up_axis": "-y",
        "handedness": "right",
        "units": "map",
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_site_profile(profile_path)

    assert profile.coordinate_frame is not None
    assert profile.coordinate_frame.id == "test-reconstruction-v1"
    assert profile.coordinate_frame.horizontal_axes == ("x", "z")
    assert profile.coordinate_frame.up_axis == "-y"


def test_site_profile_rejects_ambiguous_coordinate_frame(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["coordinate_frame"] = {
        "id": "test-reconstruction-v1",
        "convention": "glomap",
        "horizontal_axes": ["x", "y"],
        "up_axis": "z",
        "handedness": "right",
        "units": "map",
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="raw GLOMAP"):
        load_site_profile(profile_path)


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
        {"model": "SIMPLE_RADIAL", "width": 1280, "height": 720, "params": ["1.0"]},
        {
            "model": "SIMPLE_RADIAL",
            "width": 1280,
            "height": 720,
            "params": [1.0],
            "typo": True,
        },
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


@pytest.mark.parametrize(
    ("section", "key"),
    [("top", "site_typo"), ("assets", "route_typo")],
)
def test_site_profile_rejects_unknown_keys(tmp_path: Path, section: str, key: str) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    target = raw if section == "top" else raw["assets"]
    target[key] = "ignored-before-this-fix"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown site profile"):
        load_site_profile(profile_path)


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


@pytest.mark.parametrize("asset_name", ["map.ply", "route.json"])
def test_operator_rejects_tampered_display_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset_name: str,
) -> None:
    for name in app._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["asset_sha256"] = {
        "map_ply": hashlib.sha256(b"x").hexdigest(),
        "route_json": hashlib.sha256(b"x").hexdigest(),
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "assets" / asset_name).write_bytes(b"tampered")

    with pytest.raises(SystemExit):
        app.resolve_operator_site_assets(
            _args(profile_path), argparse.ArgumentParser()
        )


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


def test_edm_tracker_profile_rejects_non_finite_values() -> None:
    profile = load_edm_production_profile(
        Path(__file__).resolve().parents[2]
        / "定位演算法/configs/edm_production_profile.json"
    )
    profile["tracker"]["max_jump"] = float("nan")
    with pytest.raises(ValueError, match="max_jump"):
        apply_edm_tracker_profile(production_edm_config(), profile)


def test_query_camera_is_constructed_before_model_startup() -> None:
    with pytest.raises(ValueError, match="pycolmap"):
        resolve_query_camera_override(
            "PINHOLE",
            1280,
            720,
            [900.0, 640.0, 360.0],
            1280,
            720,
        )


@pytest.mark.parametrize(
    "camera",
    [
        ("PINHOLE", 1280.5, 720, [900.0, 900.0, 640.0, 360.0]),
        ("PINHOLE", True, 720, [900.0, 900.0, 640.0, 360.0]),
        ("PINHOLE", 1280, 720, [900.0, False, 640.0, 360.0]),
        ("PINHOLE", 1280, 720, [900.0, float("nan"), 640.0, 360.0]),
    ],
)
def test_query_camera_rejects_coercible_or_nonfinite_values(camera) -> None:
    with pytest.raises(ValueError, match="query camera"):
        validate_camera_tuple(camera)


def test_site_profile_json_rejects_nonfinite_constants(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["flight"] = {"map_units_per_meter": float("nan")}
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON"):
        load_site_profile(profile_path)


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
        bundle_sha256="a" * 64,
        localizer_profile_sha256="b" * 64,
    )

    index = captured["cmd"].index("--deploy-dir")
    assert captured["cmd"][index + 1] == "/tmp/edm-deploy"
    matcher_index = captured["cmd"].index("--edm-matcher")
    assert captured["cmd"][matcher_index + 1] == "torch"
    profile_index = captured["cmd"].index("--production-profile")
    assert captured["cmd"][profile_index + 1] == "/tmp/edm-production.json"
    assert captured["cmd"][captured["cmd"].index("--bundle-sha256") + 1] == "a" * 64
    assert (
        captured["cmd"][
            captured["cmd"].index("--production-profile-sha256") + 1
        ]
        == "b" * 64
    )
    assert "--edm-onnx" not in captured["cmd"]


def test_localizer_backend_auto_detects_edm_bundle_name() -> None:
    assert app.resolve_localizer_backend("auto", Path("your_site_reloc_map_edm.pt")) == "edm"
    assert app.resolve_localizer_backend("auto", Path("current_reloc_map_updated_v3.pt")) == "xfeat"
    assert app.resolve_localizer_backend("edm", Path("anything.pt")) == "edm"


def _approved_profile(root: Path) -> Path:
    """A profile that passes every readiness gate, so one removal shows up alone."""
    profile_path = _write_profile(root)
    assets = root / "assets"
    (assets / "refs.json").write_bytes(b"x")
    (assets / "edm.json").write_bytes(b"x")
    (assets / "T_align_gravity.json").write_bytes(b"x")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["localizer"] = "edm"
    raw["localizer_profile"] = "assets/edm.json"
    raw["map_reference_poses"] = "assets/refs.json"
    raw["map_align"] = "assets/T_align_gravity.json"
    raw["asset_sha256"] = {
        "map_ply": "a" * 64,
        "localization_bundle": "b" * 64,
        "route_json": "c" * 64,
        "map_reference_poses": "d" * 64,
        "localizer_profile": "e" * 64,
        "map_align": "f" * 64,
    }
    raw["flight"] = {
        "approved": True,
        "coordinate_frame_id": "test-frame-v1",
        "route_clearance_approved": True,
        "approval_note": "test",
        "controller": {
            "model": "scale_free_direction_speed_guard_v1",
            "speed_limit_mps": 0.30,
            "pose_max_age_ms": 500,
            "speed_max_age_ms": 500,
            "command_ttl_ms": 150,
            "yaw_tolerance_deg": 3.0,
            "lookahead_map_units": 0.8,
            "rejoin_tolerance_map_units": 0.45,
            "arrival_tolerance_map_units": 0.15,
            "inspect_radius_map_units": 0.75,
            "inspect_resume_margin_map_units": 0.25,
            "max_pose_jump_map_units": 1.5,
            "max_route_deviation_map_units": 3.0,
            "progress_jump_slack_map_units": 1.5,
            "max_progress_regression_map_units": 0.2,
            "progress_speed_factor": 2.0,
            # Axis INDICES: this schema can only express axis-aligned frames, which
            # is why the measured rotation lives in map_align instead.
            "horizontal_axes": [0, 2],
            "vertical_axis": 1,
            "camera_to_body_yaw_deg": 0.0,
            "body_right_sign": 1,
            "segment_window": 2,
            "inspect_waypoints": [],
        },
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    return profile_path


def test_autonomous_flight_requires_a_measured_map_alignment(tmp_path: Path) -> None:
    """Without it the controller silently assumes GLOMAP -Y is up.

    Every site measured so far is 1.99 to 22.51 degrees away from that guess, which
    tilts the horizontal plane every commanded body axis is decomposed against.
    """
    profile_path = _approved_profile(tmp_path)
    assert not any("map_align" in item
                   for item in flight_readiness_errors(load_site_profile(profile_path)))

    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    del raw["map_align"]
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    errors = flight_readiness_errors(load_site_profile(profile_path))
    assert any("missing map_align" in item for item in errors), errors


def test_autonomous_flight_requires_the_alignment_to_be_hash_pinned(tmp_path: Path) -> None:
    profile_path = _approved_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    del raw["asset_sha256"]["map_align"]
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    errors = flight_readiness_errors(load_site_profile(profile_path))
    assert any("asset_sha256.map_align" in item for item in errors), errors
