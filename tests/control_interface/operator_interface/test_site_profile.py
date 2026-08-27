from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa

import flight_operator_app as app
import mission_pipeline
from live_localizer_worker import (
    apply_edm_tracker_profile,
    load_edm_production_profile,
    resolve_query_camera_override,
)
from edm_localizer_adapter import production_edm_config
from production_localizer_factory import validate_camera_tuple
from hardware_approval_trust import (
    TRUST_STORE_SCHEMA,
    attach_detached_signature,
    canonical_json_bytes,
    create_unsigned_signing_payload,
)
from site_profile import (
    bind_camera_center_navigation_calibration,
    flight_readiness_errors,
    load_hardware_approval_receipt,
    load_site_profile,
    site_profile_approval_sha256,
)
from pose_frame_chain import CameraBodyExtrinsic, save_camera_body_extrinsic
from site_alignment import SiteAlignment, save_site_alignment, solve_similarity_alignment


def _write_profile(root: Path, *, missing_bundle: bool = False) -> Path:
    # Model the real workspace layout so profile resolution can use the same
    # discovered root as bundled profiles.
    for name in ("控制介面程式", "定位演算法", "地圖檔"):
        (root / name).mkdir(parents=True, exist_ok=True)
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


def test_site_profile_accepts_registered_xfeat_localizer(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer"] = "xfeat"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_site_profile(profile_path)
    assert profile.localizer == "xfeat"
    assert all("localizer_profile" not in error for error in flight_readiness_errors(profile))


def test_site_profile_rejects_cache_and_reference_index_together(
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(tmp_path)
    index_dir = tmp_path / "assets" / "index"
    index_dir.mkdir()
    (index_dir / "SHA256SUMS.json").write_text("{}", encoding="utf-8")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer"] = "xfeat"
    raw["assets"]["reference_index"] = "assets/index/SHA256SUMS.json"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="either assets.reference_index"):
        load_site_profile(profile_path)


def test_site_profile_reference_index_must_name_its_manifest(
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer"] = "xfeat"
    raw["assets"].pop("megaloc_cache")
    raw["assets"]["reference_index"] = "assets/index.json"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256SUMS.json manifest"):
        load_site_profile(profile_path, validate_files=False)


def test_site_profile_requires_backend_specific_edm_profile(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["localizer"] = "edm"
    raw["flight"] = {
        "approved": False,
        "coordinate_frame_id": "test-frame-v1",
        "route_clearance_approved": False,
        "approval_note": "ground only",
        "controller": None,
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    profile = load_site_profile(profile_path)
    assert "missing EDM localizer_profile" in flight_readiness_errors(profile)


def test_xfeat_profile_rejects_edm_only_profile_asset(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    runtime_profile = tmp_path / "edm.json"
    runtime_profile.write_text("{}", encoding="utf-8")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer"] = "xfeat"
    raw["localizer_profile"] = runtime_profile.name
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="does not support localizer_profile"):
        load_site_profile(profile_path)


def test_site_profile_rejects_unknown_localizer(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer"] = "unknown"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="registered localizer backend"):
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
        "一般場域 EDM",
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


def _attach_v2_hardware_receipt(profile_path: Path, **overrides: object) -> Path:
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["query_camera"] = {
        "model": "PINHOLE",
        "width": 1280,
        "height": 720,
        "params": [931.2, 931.2, 640.0, 360.0],
    }
    raw.pop("hardware_approval", None)
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    receipt = {
        "schema": "anafi-hardware-approval/v2",
        "approved": True,
        "site_id": "test_site",
        "coordinate_frame_id": "test-frame-v1",
        "profile_sha256": site_profile_approval_sha256(profile_path),
        "route_sha256": "c" * 64,
        "bundle_sha256": "b" * 64,
        "approved_mode": "auto",
        "aircraft": {
            "product": "ANAFI",
            "serial": "ANAFI-L105488",
            "firmware_versions": ["1.8.2"],
        },
        "controller": {
            "product": "SkyController 3",
            "serial": "SC3-001",
            "firmware_versions": ["1.8.1"],
        },
        "olympe_versions": ["8.4.0"],
        "approved_envelope": {
            "max_altitude_m": 20.0,
            "max_distance_m": 50.0,
            "max_tilt_deg": 15.0,
            "max_vertical_speed_ms": 2.0,
            "max_rotation_speed_degs": 20.0,
            "gps_required": False,
        },
        "approval_note": "Props-off and ground checks recorded; AUTO not implied.",
    }
    receipt.update(overrides)
    receipt_path = profile_path.parent / "hardware-approval-v2.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    raw["hardware_approval"] = {
        "receipt": receipt_path.name,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    return receipt_path


def _attach_signed_v2_hardware_receipt(profile_path: Path) -> None:
    receipt_path = _attach_v2_hardware_receipt(profile_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)

    def timestamp(value: datetime) -> str:
        return value.isoformat(timespec="seconds").replace("+00:00", "Z")

    key = ECC.generate(curve="ed25519")
    unsigned = create_unsigned_signing_payload(
        receipt,
        key_id="test-field-authority",
        issued_at=timestamp(now - timedelta(minutes=1)),
        expires_at=timestamp(now + timedelta(hours=1)),
    )
    signature = eddsa.new(key, "rfc8032").sign(canonical_json_bytes(unsigned))
    signature_path = profile_path.parent / "hardware-approval-v2.signature.json"
    signature_path.write_text(
        json.dumps(attach_detached_signature(unsigned, signature)),
        encoding="utf-8",
    )
    trust_path = profile_path.parent / "hardware-approval-trust.json"
    trust_path.write_text(
        json.dumps({
            "schema": TRUST_STORE_SCHEMA,
            "keys": [{
                "key_id": "test-field-authority",
                "public_key": key.public_key().export_key(format="raw").hex(),
                "role": "field_hardware_authority",
                "permissions": ["hardware_approval:auto"],
                "not_before": timestamp(now - timedelta(days=1)),
                "not_after": timestamp(now + timedelta(days=1)),
                "revoked": False,
            }],
        }),
        encoding="utf-8",
    )
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["hardware_approval"].update({
        "signature": signature_path.name,
        "signature_sha256": hashlib.sha256(
            signature_path.read_bytes()
        ).hexdigest(),
        "trust_store": trust_path.name,
        "trust_store_sha256": hashlib.sha256(trust_path.read_bytes()).hexdigest(),
    })
    profile_path.write_text(json.dumps(raw), encoding="utf-8")


def test_hardware_approval_receipt_v2_loads_bound_identity_and_envelope(
    tmp_path: Path,
) -> None:
    profile_path = _approved_profile(tmp_path)
    _attach_v2_hardware_receipt(profile_path)

    profile = load_site_profile(profile_path, validate_files=False)
    receipt = load_hardware_approval_receipt(profile.hardware_approval)

    assert receipt.site_id == "test_site"
    assert receipt.coordinate_frame_id == "test-frame-v1"
    assert receipt.profile_sha256 == site_profile_approval_sha256(profile_path)
    assert receipt.route_sha256 == "c" * 64
    assert receipt.bundle_sha256 == "b" * 64
    assert receipt.approved_mode == "auto"
    assert receipt.aircraft_serial == "ANAFI-L105488"
    assert receipt.controller_serial == "SC3-001"
    assert receipt.approved_envelope["gps_required"] is False


def test_site_profile_approval_digest_ignores_only_hardware_reference(
    tmp_path: Path,
) -> None:
    profile_path = _approved_profile(tmp_path)
    before = site_profile_approval_sha256(profile_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["hardware_approval"] = {
        "receipt": "receipt.json",
        "sha256": "a" * 64,
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    assert site_profile_approval_sha256(profile_path) == before

    raw["flight"]["approval_note"] = "changed safety decision"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    assert site_profile_approval_sha256(profile_path) != before


def test_auto_readiness_does_not_require_signed_v2_receipt(tmp_path: Path) -> None:
    profile_path = _approved_profile(tmp_path)
    _attach_v2_hardware_receipt(profile_path)

    errors = flight_readiness_errors(
        load_site_profile(profile_path, validate_files=False)
    )

    assert not any("hardware approval" in error for error in errors), errors


def test_auto_readiness_accepts_valid_signed_v2_receipt(tmp_path: Path) -> None:
    profile_path = _approved_profile(tmp_path)
    _attach_signed_v2_hardware_receipt(profile_path)

    errors = flight_readiness_errors(
        load_site_profile(profile_path, validate_files=False)
    )

    assert errors == []


@pytest.mark.parametrize(
    ("filename", "reason"),
    [
        ("hardware-approval-v2.signature.json", "signature SHA-256 mismatch"),
        ("hardware-approval-trust.json", "trust store SHA-256 mismatch"),
    ],
)
def test_auto_readiness_does_not_gate_on_tampered_signed_material(
    tmp_path: Path, filename: str, reason: str
) -> None:
    profile_path = _approved_profile(tmp_path)
    _attach_signed_v2_hardware_receipt(profile_path)
    profile = load_site_profile(profile_path, validate_files=False)
    (tmp_path / filename).write_text("{}", encoding="utf-8")

    errors = flight_readiness_errors(profile)

    assert not any("hardware approval" in error for error in errors), errors


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("site_id", "other_site", "hardware approval site_id mismatch"),
        (
            "coordinate_frame_id",
            "other-frame",
            "hardware approval coordinate_frame_id mismatch",
        ),
        ("profile_sha256", "f" * 64, "hardware approval profile_sha256 mismatch"),
        ("route_sha256", "d" * 64, "hardware approval route_sha256 mismatch"),
        ("bundle_sha256", "a" * 64, "hardware approval bundle_sha256 mismatch"),
        ("approved_mode", "manual", "hardware approval approved_mode is not auto"),
    ],
)
def test_auto_readiness_does_not_gate_on_unbound_or_manual_hardware_receipt(
    tmp_path: Path, field: str, value: object, reason: str,
) -> None:
    profile_path = _approved_profile(tmp_path)
    _attach_v2_hardware_receipt(profile_path, **{field: value})

    errors = flight_readiness_errors(
        load_site_profile(profile_path, validate_files=False)
    )

    assert not any("hardware approval" in error for error in errors), errors


def test_all_shipped_profiles_are_scale_free() -> None:
    root = Path(__file__).resolve().parents[3]
    profiles = (root / "控制介面程式" / "site_profiles").glob("*.json")
    for path in profiles:
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["schema_version"] == 2, path
        assert "map_units_per_meter" not in raw.get("flight", {}), path
        assert raw["flight"]["approved"] is False, path
        assert raw["flight"]["route_clearance_approved"] is False, path
        profile = load_site_profile(path, validate_files=False)
        assert profile.schema_version == 2


def test_current_river_map_route_and_approval_are_consistent() -> None:
    profile_path = (
        Path(__file__).resolve().parents[3]
        / "地圖檔"
        / "場域"
        / "river_site"
        / "site_profile.json"
    )

    profile = load_site_profile(profile_path)

    assert profile.hardware_approval is None
    assert profile.flight is not None
    if profile.route_json is None:
        assert profile.flight.approved is False
        assert profile.flight.route_clearance_approved is False
        assert profile.asset_sha256.route_json is None
    else:
        assert profile.asset_sha256.route_json is not None
        if profile.flight.approved:
            assert profile.flight.route_clearance_approved is True
    assert profile.coordinate_frame is not None
    assert profile.coordinate_frame.id == (
        "river_site_b0_p116_p117_reconstruction_18056f835daa"
    )
    assert not any("pose_chain" in error for error in flight_readiness_errors(profile))
    assert not any("hardware approval" in error for error in flight_readiness_errors(profile))




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


def test_site_profile_rejects_asset_path_traversal(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    escaped = tmp_path.parent / "escaped-map.ply"
    escaped.write_bytes(b"x")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["assets"]["map_ply"] = "../escaped-map.ply"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="outside workspace root"):
        load_site_profile(profile_path)


def test_site_profile_rejects_absolute_external_asset_path(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    external = tmp_path.parent / "absolute-external-map.ply"
    external.write_bytes(b"x")
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["assets"]["map_ply"] = str(external)
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="outside workspace root"):
        load_site_profile(profile_path)


def test_site_profile_rejects_symlink_asset_file(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    target = tmp_path / "assets" / "target-map.ply"
    target.write_bytes(b"x")
    link = tmp_path / "assets" / "linked-map.ply"
    link.symlink_to(target)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["assets"]["map_ply"] = "assets/linked-map.ply"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must not use symlinks"):
        load_site_profile(profile_path)


def test_site_profile_rejects_symlink_asset_ancestor(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    linked_assets = tmp_path / "linked-assets"
    linked_assets.symlink_to(tmp_path / "assets", target_is_directory=True)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["assets"]["map_ply"] = "linked-assets/map.ply"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must not use symlinks"):
        load_site_profile(profile_path)


def test_site_profile_rejects_external_localizer_deploy_directory(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    external_deploy = tmp_path.parent / "external-edm-deploy"
    external_deploy.mkdir()
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["localizer_deploy_dir"] = str(external_deploy)
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="outside workspace root"):
        load_site_profile(profile_path)


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


def test_real_operator_requires_mission_selection_even_with_a_site_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in app._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SFM_MISSION_SELECTION", raising=False)
    args = _args(_write_profile(tmp_path), live=True)

    with pytest.raises(SystemExit, match="SFM_MISSION_SELECTION"):
        app._resolve_startup_site(args, argparse.ArgumentParser())


def test_real_operator_binds_profile_and_session_identity_to_mission_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in app._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    profile_path = _write_profile(tmp_path).resolve()
    selection_path = (tmp_path / "mission selection.json").resolve()
    selection_path.write_text("{}", encoding="utf-8")
    mission = SimpleNamespace(
        identity="test-selection@123456789abc",
        selection=SimpleNamespace(source=selection_path),
        selection_sha256="1" * 64,
        readiness=SimpleNamespace(
            localization_ready=True,
            localization_errors=(),
            flight_ready=False,
            flight_errors=("test readiness blocker",),
        ),
        materialize_legacy_site_profile=lambda _output_dir: profile_path,
    )
    monkeypatch.setenv("SFM_MISSION_SELECTION", str(selection_path))
    monkeypatch.setattr(app, "resolve_mission", lambda *_args, **_kwargs: mission)
    args = _args(profile_path, live=True)

    selected, approval = app._resolve_startup_site(
        args, argparse.ArgumentParser()
    )

    assert selected is not None
    assert approval is None
    assert args.mission_selection == str(selection_path)
    assert args.mission_selection_sha256 == "1" * 64
    assert args.mission_snapshot_id == "test-selection@123456789abc"
    assert args.mission_flight_ready is False
    assert args.mission_flight_errors == ("test readiness blocker",)


def test_real_operator_rejects_profile_not_materialized_from_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in app._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    profile_path = _write_profile(tmp_path).resolve()
    other_profile = tmp_path / "different.snapshot.json"
    other_profile.write_text(profile_path.read_text(encoding="utf-8"), encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}", encoding="utf-8")
    mission = SimpleNamespace(
        identity="selection@abcdef123456",
        selection=SimpleNamespace(source=selection_path.resolve()),
        selection_sha256="2" * 64,
        readiness=SimpleNamespace(
            localization_ready=True,
            localization_errors=(),
            flight_ready=True,
            flight_errors=(),
        ),
        materialize_legacy_site_profile=lambda _output_dir: other_profile.resolve(),
    )
    monkeypatch.setenv("SFM_MISSION_SELECTION", str(selection_path))
    monkeypatch.setattr(app, "resolve_mission", lambda *_args, **_kwargs: mission)

    with pytest.raises(SystemExit, match="does not match mission selection snapshot"):
        app._resolve_startup_site(
            _args(profile_path, live=True), argparse.ArgumentParser()
        )


def test_real_operator_session_manifest_records_mission_selection_identity() -> None:
    args = SimpleNamespace(
        mission_selection="/missions/field-a.json",
        mission_selection_sha256="3" * 64,
        mission_snapshot_id="field-a@333333333333",
        mission_flight_ready=False,
        mission_flight_errors=("approval missing",),
        max_altitude_m=None,
        max_distance_m=None,
        distance_geofence=True,
    )
    identity = app.OperatorSessionIdentity(
        asset_hashes={},
        site_profile_sha256="",
        runtime_profile_sha256="",
        site_profile_schema_version=0,
        autonomous_speed_limit_mps=0.3,
        autonomy_profile_errors=("approval missing",),
        source_identity="192.168.53.1",
        source_sha256="",
    )

    manifest = app._operator_session_manifest(
        args,
        app.InterfaceMode.REAL_FLIGHT,
        None,
        None,
        identity,
    )

    assert manifest["mission_selection"] == "/missions/field-a.json"
    assert manifest["mission_selection_sha256"] == "3" * 64
    assert manifest["mission_snapshot_id"] == "field-a@333333333333"
    assert manifest["mission_flight_ready"] is False
    assert manifest["mission_flight_errors"] == ["approval missing"]


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
    path = (
        Path(__file__).resolve().parents[3]
        / "控制介面程式"
        / "site_profiles"
        / "example_site_edm.json"
    )
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
        Path(__file__).resolve().parents[3]
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
    assert (tracker["lost_local_grace_frames"], tracker["recovery_bank_size"], tracker["recovery_scan_topk"]) == (2, 192, 2)
    assert tracker["use_temporal_reference"] is False
    assert (tracker["max_reproj_error_acquire"], tracker["max_reproj_error_track"], tracker["pnp_ransac_max_error"]) == (5.0, 6.0, 5.0)
    assert tracker["prediction_max_dt"] == 0.25
    assert "reposed" not in profile



def test_edm_tracker_profile_rejects_non_finite_values() -> None:
    profile = load_edm_production_profile(
        Path(__file__).resolve().parents[3]
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
    """A complete flight contract; receipt helpers add the hardware trust gate."""
    profile_path = _write_profile(root)
    assets = root / "assets"
    (assets / "refs.json").write_bytes(b"x")
    (assets / "edm.json").write_bytes(b"x")
    (assets / "T_align_gravity.json").write_bytes(b"x")
    control_points = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 0.0],
    ]
    alignment = SiteAlignment.from_fit(
        map_frame_id="test-frame-v1",
        site_frame_id="test-site-enu-v1",
        fit=solve_similarity_alignment(control_points, control_points),
        approved=True,
    )
    save_site_alignment(alignment, assets / "site_alignment.json")
    save_camera_body_extrinsic(
        CameraBodyExtrinsic(
            vehicle_id="test-anafi-720p-v1",
            body_frame_id="test-anafi-frd",
            camera_frame_id="test-camera-opencv",
            rotation_camera_from_body=(
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
            ),
            translation_camera_from_body_m=(0.0, 0.0, 0.0),
            fixed_gimbal_pitch_deg=-10.0,
            gimbal_pitch_tolerance_deg=1.0,
            evidence="test fixture",
            approved=True,
        ),
        assets / "camera_body_extrinsic.json",
    )
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["localizer"] = "edm"
    raw["localizer_profile"] = "assets/edm.json"
    raw["map_reference_poses"] = "assets/refs.json"
    raw["map_align"] = "assets/T_align_gravity.json"
    raw["coordinate_frame"] = {
        "id": "test-frame-v1",
        "convention": "glomap",
        "horizontal_axes": ["x", "z"],
        "up_axis": "-y",
        "handedness": "right",
        "units": "map",
    }
    raw["pose_chain"] = {
        "site_frame_id": "test-site-enu-v1",
        "vehicle_id": "test-anafi-720p-v1",
        "body_frame_id": "test-anafi-frd",
        "camera_frame_id": "test-camera-opencv",
        "site_alignment": "assets/site_alignment.json",
        "camera_body_extrinsic": "assets/camera_body_extrinsic.json",
    }
    raw["asset_sha256"] = {
        "map_ply": "a" * 64,
        "localization_bundle": "b" * 64,
        "route_json": "c" * 64,
        "map_reference_poses": "d" * 64,
        "localizer_profile": "e" * 64,
        "map_align": "f" * 64,
        "site_alignment": "1" * 64,
        "camera_body_extrinsic": "2" * 64,
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


def test_autonomous_flight_does_not_require_signed_hardware_approval(
    tmp_path: Path,
) -> None:
    errors = flight_readiness_errors(load_site_profile(_approved_profile(tmp_path)))

    assert not any("hardware approval" in error for error in errors), errors


def test_camera_center_field_calibration_atomically_binds_and_unlocks_profile(
    tmp_path: Path,
) -> None:
    profile_path = _approved_profile(tmp_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw.pop("pose_chain")
    raw["asset_sha256"].pop("site_alignment")
    raw["asset_sha256"].pop("camera_body_extrinsic")
    raw["query_camera"] = {
        "model": "PINHOLE",
        "width": 1280,
        "height": 720,
        "params": [900.0, 900.0, 640.0, 360.0],
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    controls = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 0.3),
        (0.0, 0.0, -0.3),
    )
    alignment_path = save_site_alignment(
        SiteAlignment.from_fit(
            map_frame_id="test-frame-v1",
            site_frame_id="test-site-camera-navigation-v1",
            fit=solve_similarity_alignment(controls, controls),
            approved=True,
        ),
        calibration / "site_alignment.json",
    )
    extrinsic_path = save_camera_body_extrinsic(
        CameraBodyExtrinsic(
            vehicle_id="test-anafi-camera-centre-v1",
            body_frame_id="test-anafi-camera-centre-frd",
            camera_frame_id="test-camera-opencv",
            rotation_camera_from_body=(
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
            ),
            translation_camera_from_body_m=(0.0, 0.0, 0.0),
            fixed_gimbal_pitch_deg=-20.0,
            gimbal_pitch_tolerance_deg=1.0,
            evidence="camera optical centre is the navigation origin",
            approved=True,
        ),
        calibration / "camera_center_extrinsic.json",
    )

    profile = bind_camera_center_navigation_calibration(
        profile_path,
        site_alignment_path=alignment_path,
        camera_body_extrinsic_path=extrinsic_path,
    )

    assert profile.pose_chain is not None
    assert profile.pose_chain.site_alignment == alignment_path
    assert profile.pose_chain.camera_body_extrinsic == extrinsic_path
    assert flight_readiness_errors(profile) == []
    persisted = json.loads(profile_path.read_text(encoding="utf-8"))
    assert persisted["pose_chain"]["body_frame_id"].endswith("camera-centre-frd")
    assert persisted["asset_sha256"]["site_alignment"] == hashlib.sha256(
        alignment_path.read_bytes()
    ).hexdigest()
    assert persisted["asset_sha256"]["camera_body_extrinsic"] == hashlib.sha256(
        extrinsic_path.read_bytes()
    ).hexdigest()
    assert not list(tmp_path.glob(".*.calibration.tmp"))


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
