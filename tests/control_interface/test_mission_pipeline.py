from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa

CONTROL_ROOT = Path(__file__).resolve().parents[2] / "控制介面程式"
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from operator_interface.hardware_approval_trust import (  # noqa: E402
    TRUST_STORE_SCHEMA,
    attach_detached_signature,
    canonical_json_bytes,
    create_unsigned_signing_payload,
)
from site_profile import (  # noqa: E402
    _load_flight_controller,
    site_profile_approval_sha256,
)
from pose_frame_chain import (  # noqa: E402
    CameraBodyExtrinsic,
    save_camera_body_extrinsic,
)
from site_alignment import (  # noqa: E402
    SiteAlignment,
    save_site_alignment,
    solve_similarity_alignment,
)

SPEC = importlib.util.spec_from_file_location(
    "mission_pipeline_under_test", CONTROL_ROOT / "mission_pipeline.py"
)
assert SPEC is not None and SPEC.loader is not None
mission_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mission_pipeline)


@pytest.fixture(autouse=True)
def clear_site_asset_environment(monkeypatch):
    monkeypatch.delenv("SFM_SITE_PROFILE", raising=False)
    for name in mission_pipeline._SITE_ASSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _args(**overrides):
    values = {
        "site_profile": "",
        "mission_selection": "",
        "allow_legacy_assets": False,
        "map_ply": None,
        "bundle": None,
        "megaloc_cache": None,
        "track_landmarks": None,
        "path_json": None,
        "poles_json": None,
        "safezone_dir": None,
        "controller": "auto",
        "safety_file": "/tmp/test-safety.cmd",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="mission-test")


def _scale_free_controller() -> dict:
    return {
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
    }


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("speed_limit_mps", 2.01),
        ("lookahead_map_units", 5.01),
        ("max_pose_jump_map_units", 5.01),
        ("max_route_deviation_map_units", 10.01),
    ],
)
def test_site_controller_rejects_values_outside_the_safety_envelope(
    field: str, unsafe: float, tmp_path: Path,
) -> None:
    controller = _scale_free_controller()
    controller[field] = unsafe

    with pytest.raises(ValueError, match=field):
        _load_flight_controller(controller, tmp_path / "site.json")


def test_operational_mode_requires_site_profile():
    with pytest.raises(SystemExit) as exc:
        mission_pipeline.resolve_mission_site_assets(
            _args(), _parser(), mode="dry-run"
        )
    assert exc.value.code == 2


def test_selftest_remains_profile_free():
    args = _args()
    assert (
        mission_pipeline.resolve_mission_site_assets(
            args, _parser(), mode="flight-selftest"
        )
        is None
    )
    assert args.bundle.endswith("your_site_reloc_map_edm.pt")


def test_main_dispatches_profile_free_selftest_without_external_process(monkeypatch):
    commands = []
    monkeypatch.setattr(
        mission_pipeline,
        "run",
        lambda command, env: commands.append((command, env)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mission_pipeline.py", "--mode", "flight-selftest", "--python", "python-test"],
    )

    mission_pipeline.main()

    assert len(commands) == 1
    command, env = commands[0]
    assert command == [
        "python-test",
        str(mission_pipeline.FLIGHT / "path_follow_flight.py"),
        "--selftest",
    ]
    assert "SFM_SITE_PROFILE" not in env


def test_mode_fly_is_rejected_before_exec(monkeypatch):
    commands = []
    monkeypatch.setattr(
        mission_pipeline,
        "run",
        lambda command, env: commands.append((command, env)),
    )
    args = argparse.Namespace(mode="fly")
    with pytest.raises(SystemExit, match="operator UI"):
        mission_pipeline._run_mission_mode(args, [], {})
    assert commands == []

def test_main_mode_fly_is_rejected_before_resolve_or_exec(monkeypatch):
    commands = []
    monkeypatch.setattr(
        mission_pipeline,
        "run",
        lambda command, env: commands.append((command, env)),
    )
    monkeypatch.setattr(
        mission_pipeline,
        "resolve_mission_site_assets",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fly must not resolve assets")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["mission_pipeline.py", "--mode", "fly", "--allow-legacy-assets"],
    )
    with pytest.raises(SystemExit, match="operator UI"):
        mission_pipeline.main()
    assert commands == []



def test_main_shadow_readiness_preserves_blocked_output_and_exit(
    monkeypatch, capsys, tmp_path
):
    profile = SimpleNamespace(
        site_id="alpha",
        display_name="Alpha",
        source=tmp_path / "profile.json",
    )
    monkeypatch.setattr(
        mission_pipeline,
        "resolve_mission_site_assets",
        lambda args, parser, mode: profile,
    )
    monkeypatch.setattr(
        mission_pipeline,
        "shadow_readiness_errors",
        lambda value: ["bad route"],
    )
    monkeypatch.setattr(
        mission_pipeline,
        "shadow_authorization_blockers",
        lambda value: ["not approved"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mission_pipeline.py",
            "--mode",
            "shadow-readiness",
            "--site-profile",
            str(profile.source),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mission_pipeline.main()

    assert exc.value.code == 1
    assert capsys.readouterr().out == (
        f"[mission_pipeline] site='alpha' name='Alpha' profile={profile.source}\n"
        "[shadow-readiness] package=BLOCKED\n"
        "  package blocker: bad route\n"
        "[shadow-readiness] autonomous execution=BLOCKED\n"
        "  human/field blocker: not approved\n"
    )


def test_safety_commands_use_one_private_default_path(tmp_path):
    runtime = tmp_path / "runtime"
    environment = {"XDG_RUNTIME_DIR": str(runtime)}

    path = mission_pipeline.safety_file_from_environment(environment)
    mission_pipeline.write_safety_command(path, "land")

    assert path == runtime / "sfm_drone" / "safety.cmd"
    assert path.read_text(encoding="utf-8") == "land\n"
    assert path.parent.stat().st_mode & 0o077 == 0
    assert path.stat().st_mode & 0o077 == 0


def test_safety_command_writer_rejects_relative_and_symlink_paths(tmp_path):
    with pytest.raises(RuntimeError, match="absolute"):
        mission_pipeline.write_safety_command(Path("relative.cmd"), "hover")

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = private / "target.cmd"
    target.write_text("hover\n", encoding="utf-8")
    target.chmod(0o600)
    link = private / "safety.cmd"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="safety command"):
        mission_pipeline.write_safety_command(link, "land")


def test_profile_uses_canonical_route_paths(monkeypatch, tmp_path):
    site_packages = tmp_path / "場域"
    package_root = site_packages / "alpha-package"
    profile = SimpleNamespace(
        source=tmp_path / "site.json",
        schema_version=2,
        site_id="alpha_edm",
        display_name="Alpha",
        map_ply=package_root / "maps" / "alpha.ply",
        localization_bundle=package_root / "bundles" / "alpha.pt",
        route_json=None,
        poles_json=None,
        megaloc_cache=None,
        track_landmarks=None,
        flight=None,
    )
    monkeypatch.setattr(mission_pipeline, "load_site_profile", lambda _: profile)
    monkeypatch.setattr(
        mission_pipeline,
        "_WS",
        SimpleNamespace(site_packages=site_packages),
    )

    args = _args(site_profile=str(profile.source))
    result = mission_pipeline.resolve_mission_site_assets(
        args, _parser(), mode="draw-path"
    )

    assert result is profile
    assert Path(args.path_json) == package_root / "routes" / "flight_path.json"
    assert Path(args.poles_json) == package_root / "routes" / "poles.json"


def test_mission_selection_materializes_one_legacy_profile(monkeypatch, tmp_path):
    profile_path = tmp_path / "snapshot.site_profile.json"
    selection_path = tmp_path / "mission.json"
    route = tmp_path / "route.json"
    poles = tmp_path / "poles.json"
    profile = SimpleNamespace(
        source=profile_path,
        schema_version=2,
        site_id="alpha",
        display_name="Alpha",
        map_ply=tmp_path / "map.ply",
        localization_bundle=tmp_path / "bundle.pt",
        route_json=route,
        poles_json=poles,
        megaloc_cache=None,
        reference_index=None,
        track_landmarks=None,
        asset_sha256=SimpleNamespace(reference_index=None),
        flight=None,
    )
    resolved = SimpleNamespace(
        readiness=SimpleNamespace(flight_ready=False, flight_errors=("blocked",)),
        materialize_legacy_site_profile=lambda _directory: profile_path,
    )
    monkeypatch.setattr(
        mission_pipeline,
        "resolve_mission",
        lambda path, workspace_root: resolved,
    )
    monkeypatch.setattr(mission_pipeline, "load_site_profile", lambda path: profile)

    args = _args(mission_selection=str(selection_path))
    result = mission_pipeline.resolve_mission_site_assets(
        args,
        _parser(),
        mode="draw-path",
    )

    assert result is profile
    assert args.site_profile == str(profile_path)
    assert args.map_ply == str(profile.map_ply)
    assert args.bundle == str(profile.localization_bundle)


def test_flight_mode_rejects_files_without_an_approved_contract(monkeypatch, tmp_path):
    route = tmp_path / "route.json"
    poles = tmp_path / "poles.json"
    profile = SimpleNamespace(
        source=tmp_path / "site.json",
        schema_version=2,
        site_id="alpha",
        display_name="Alpha",
        map_ply=tmp_path / "alpha.ply",
        localization_bundle=tmp_path / "alpha.pt",
        route_json=route,
        poles_json=poles,
        megaloc_cache=None,
        track_landmarks=None,
        flight=None,
    )
    monkeypatch.setattr(mission_pipeline, "load_site_profile", lambda _: profile)

    with pytest.raises(SystemExit):
        mission_pipeline.resolve_mission_site_assets(
            _args(site_profile=str(profile.source)), _parser(), mode="fly"
        )

    route.write_text("[]", encoding="utf-8")
    poles.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        mission_pipeline.resolve_mission_site_assets(
            _args(site_profile=str(profile.source)), _parser(), mode="fly"
        )
    assert exc.value.code == 2


def test_scale_free_flight_contract_exports_no_map_scale_and_verifies_hashes(tmp_path):
    for name in ("控制介面程式", "定位演算法", "地圖檔"):
        (tmp_path / name).mkdir()
    map_ply = tmp_path / "map.ply"
    bundle = tmp_path / "bundle.pt"
    reference_poses = tmp_path / "reference_poses.json"
    localizer_profile = tmp_path / "edm.json"
    map_align = tmp_path / "T_align_gravity.json"
    for path in (map_ply, bundle, reference_poses, localizer_profile, map_align):
        path.write_bytes(b"x")
    control_points = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 0.0],
    ]
    site_alignment = tmp_path / "site_alignment.json"
    save_site_alignment(
        SiteAlignment.from_fit(
            map_frame_id="alpha-reconstruction-v1",
            site_frame_id="alpha-site-enu-v1",
            fit=solve_similarity_alignment(control_points, control_points),
            approved=True,
        ),
        site_alignment,
    )
    camera_body_extrinsic = tmp_path / "camera_body_extrinsic.json"
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
        camera_body_extrinsic,
    )
    route = tmp_path / "route.json"
    route.write_text(json.dumps({
        "schema": "sfm-flight-route/v1",
        "site_id": "alpha",
        "coordinate_frame_id": "alpha-reconstruction-v1",
        "frame": "glomap",
        "units": "map",
        "purpose": "flight",
        "closed": False,
        "waypoints": [[0, 0, 0], [1, 0, 0]],
    }), encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    profile_path = tmp_path / "site.json"
    profile_path.write_text(json.dumps({
        "schema_version": 2,
        "site_id": "alpha",
        "display_name": "Alpha",
        "localizer": "edm",
        "localizer_profile": "edm.json",
        "map_reference_poses": "reference_poses.json",
        "map_align": "T_align_gravity.json",
        "coordinate_frame": {
            "id": "alpha-reconstruction-v1",
            "convention": "glomap",
            "horizontal_axes": ["x", "z"],
            "up_axis": "-y",
            "handedness": "right",
            "units": "map",
        },
        "pose_chain": {
            "site_frame_id": "alpha-site-enu-v1",
            "vehicle_id": "test-anafi-720p-v1",
            "body_frame_id": "test-anafi-frd",
            "camera_frame_id": "test-camera-opencv",
            "site_alignment": site_alignment.name,
            "camera_body_extrinsic": camera_body_extrinsic.name,
        },
        "query_camera": {
            "model": "PINHOLE",
            "width": 1280,
            "height": 720,
            "params": [900, 900, 640, 360],
        },
        "asset_sha256": {
            "localization_bundle": digest(bundle),
            "route_json": digest(route),
            "map_reference_poses": digest(reference_poses),
            "map_align": digest(map_align),
            "localizer_profile": digest(localizer_profile),
            "site_alignment": digest(site_alignment),
            "camera_body_extrinsic": digest(camera_body_extrinsic),
        },
        "flight": {
            "approved": True,
            "coordinate_frame_id": "alpha-reconstruction-v1",
            "route_clearance_approved": True,
            "approval_note": "fixture",
            "controller": _scale_free_controller(),
        },
        "assets": {
            "map_ply": "map.ply",
            "route_json": "route.json",
            "poles_json": None,
            "localization_bundle": "bundle.pt",
            "megaloc_cache": None,
            "track_landmarks": None,
        },
    }), encoding="utf-8")

    now = datetime.now(timezone.utc).replace(microsecond=0)

    def timestamp(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    receipt = {
        "schema": "anafi-hardware-approval/v2",
        "approved": True,
        "site_id": "alpha",
        "coordinate_frame_id": "alpha-reconstruction-v1",
        "profile_sha256": site_profile_approval_sha256(profile_path),
        "route_sha256": digest(route),
        "bundle_sha256": digest(bundle),
        "approved_mode": "auto",
        "aircraft": {
            "product": "ANAFI",
            "serial": "fixture-aircraft",
            "firmware_versions": ["fixture"],
        },
        "controller": {
            "product": "SkyController 3",
            "serial": "fixture-controller",
            "firmware_versions": ["fixture"],
        },
        "olympe_versions": ["fixture"],
        "approved_envelope": {
            "max_altitude_m": 20.0,
            "max_distance_m": 50.0,
            "max_tilt_deg": 15.0,
            "max_vertical_speed_ms": 2.0,
            "max_rotation_speed_degs": 20.0,
            "gps_required": False,
        },
        "approval_note": "test-only signed AUTO readiness fixture",
    }
    receipt_path = tmp_path / "hardware-approval-v2.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    key = ECC.generate(curve="ed25519")
    unsigned_envelope = create_unsigned_signing_payload(
        receipt,
        key_id="test-field-authority",
        issued_at=timestamp(now - timedelta(minutes=1)),
        expires_at=timestamp(now + timedelta(hours=1)),
    )
    signature = eddsa.new(key, "rfc8032").sign(
        canonical_json_bytes(unsigned_envelope)
    )
    signature_path = tmp_path / "hardware-approval-v2.signature.json"
    signature_path.write_text(
        json.dumps(attach_detached_signature(unsigned_envelope, signature)),
        encoding="utf-8",
    )
    trust_path = tmp_path / "hardware-approval-trust.json"
    trust_path.write_text(json.dumps({
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
    }), encoding="utf-8")

    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["hardware_approval"] = {
        "receipt": receipt_path.name,
        "sha256": digest(receipt_path),
        "signature": signature_path.name,
        "signature_sha256": digest(signature_path),
        "trust_store": trust_path.name,
        "trust_store_sha256": digest(trust_path),
    }
    profile_path.write_text(json.dumps(raw), encoding="utf-8")

    args = _args(site_profile=str(profile_path))
    profile = mission_pipeline.resolve_mission_site_assets(
        args, _parser(), mode="label-route"
    )
    mission_pipeline.validate_profile_flight_assets(profile)
    env = mission_pipeline.env_with_mission(args, profile)

    assert profile is not None
    assert env["SFM_LOCALIZER_BACKEND"] == "edm"
    assert env["SFM_BUNDLE_SHA256"] == digest(bundle)
    contract = json.loads(env["SFM_FLIGHT_CONTRACT_JSON"])
    assert "map_units_per_meter" not in contract
    assert contract["schema_version"] == 2
    assert contract["controller"]["model"] == "scale_free_direction_speed_guard_v1"
    assert contract["controller"]["speed_limit_mps"] == pytest.approx(0.30)
    assert contract["controller"]["inspect_waypoints"] == []
    assert contract["approval_note"] == "fixture"
    assert contract["query_camera"]["model"] == "PINHOLE"
    assert contract["localization_bundle_sha256"] == digest(bundle)


def test_autonomous_route_flight_delegates_to_the_approved_asset_gate(monkeypatch):
    profile = object()
    validated = []
    monkeypatch.setattr(
        mission_pipeline,
        "validate_profile_flight_assets",
        lambda value: validated.append(value),
    )

    mission_pipeline.validate_profile_for_flight(profile)

    assert validated == [profile]


def test_shadow_readiness_verifies_pinned_assets_without_granting_flight_approval(
    tmp_path: Path,
) -> None:
    paths = {
        "map_ply": tmp_path / "map.ply",
        "localization_bundle": tmp_path / "bundle.pt",
        "map_reference_poses": tmp_path / "reference_poses.json",
        "map_align": tmp_path / "T_align_gravity.json",
        "localizer_profile": tmp_path / "edm.json",
    }
    for path in paths.values():
        path.write_bytes(path.name.encode("utf-8"))
    route = tmp_path / "route.json"
    route.write_text(json.dumps({
        "schema": "sfm-flight-route/v1",
        "site_id": "alpha",
        "coordinate_frame_id": "alpha-frame-v1",
        "frame": "glomap",
        "units": "map",
        "purpose": "flight",
        "closed": False,
        "waypoints": [[0, 0, 0], [1, 0, 0]],
    }), encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    profile = SimpleNamespace(
        schema_version=2,
        site_id="alpha",
        localizer="edm",
        map_ply=paths["map_ply"],
        localization_bundle=paths["localization_bundle"],
        route_json=route,
        map_reference_poses=paths["map_reference_poses"],
        map_align=paths["map_align"],
        localizer_profile=paths["localizer_profile"],
        query_camera=SimpleNamespace(
            model="PINHOLE", width=1280, height=720, params=(900, 900, 640, 360),
        ),
        poles_json=None,
        hardware_approval=None,
        flight=SimpleNamespace(
            approved=False,
            coordinate_frame_id="alpha-frame-v1",
            route_clearance_approved=False,
            approval_note="shadow only",
            controller=None,
        ),
        asset_sha256=SimpleNamespace(
            map_ply=digest(paths["map_ply"]),
            localization_bundle=digest(paths["localization_bundle"]),
            route_json=digest(route),
            map_reference_poses=digest(paths["map_reference_poses"]),
            map_align=digest(paths["map_align"]),
            localizer_profile=digest(paths["localizer_profile"]),
            poles_json=None,
        ),
    )

    assert mission_pipeline.shadow_readiness_errors(profile) == []
    blockers = mission_pipeline.shadow_authorization_blockers(profile)
    assert "flight.approved is false" in blockers
    assert "flight.route_clearance_approved is false" in blockers


@pytest.mark.parametrize(
    ("waypoints", "closed", "message"),
    [
        ([[0, 0, 0], [0, 0, 0]], False, "zero length"),
        ([[0, 0, 0], [True, 0, 1]], False, "finite numeric 3-vector"),
        ([[0, 0, 0], [float("nan"), 0, 1]], False, "non-finite JSON"),
        ([[0, 0, 0], [1, 0, 0]], 0, "closed"),
    ],
)
def test_flight_route_rejects_invalid_geometry(
    tmp_path, waypoints, closed, message
):
    route = tmp_path / "route.json"
    route.write_text(json.dumps({
        "schema": "sfm-flight-route/v1",
        "site_id": "alpha",
        "coordinate_frame_id": "alpha-v1",
        "frame": "glomap",
        "units": "map",
        "purpose": "flight",
        "closed": closed,
        "waypoints": waypoints,
    }), encoding="utf-8")
    profile = SimpleNamespace(
        site_id="alpha",
        route_json=route,
        flight=SimpleNamespace(coordinate_frame_id="alpha-v1"),
    )

    with pytest.raises(ValueError, match=message):
        mission_pipeline._validate_flight_route(profile)


def test_flight_profile_rejects_invalid_pycolmap_camera_before_launch(
    tmp_path, monkeypatch
):
    profile = SimpleNamespace(
        flight=SimpleNamespace(
            approved=True,
            coordinate_frame_id="frame",
            route_clearance_approved=True,
            approval_note="",
            controller=SimpleNamespace(inspect_waypoints=()),
        ),
        site_id="alpha",
        localizer="edm",
        query_camera=SimpleNamespace(
            model="PINHOLE",
            width=1280,
            height=720,
            params=(900.0, 640.0, 360.0),
        ),
        route_json=tmp_path / "route.json",
        map_reference_poses=tmp_path / "refs.json",
        map_align=tmp_path / "T_align_gravity.json",
        localization_bundle=tmp_path / "bundle.pt",
        localizer_profile=tmp_path / "edm.json",
        poles_json=None,
        asset_sha256=SimpleNamespace(
            localization_bundle="a" * 64,
            route_json="b" * 64,
            map_reference_poses="c" * 64,
            map_align="e" * 64,
            localizer_profile="d" * 64,
            poles_json=None,
        ),
    )
    monkeypatch.setattr(
        mission_pipeline,
        "flight_readiness_errors",
        lambda _profile: [],
    )

    with pytest.raises(ValueError, match="pycolmap"):
        mission_pipeline.validate_profile_flight_assets(profile)


def test_inspection_flight_contract_requires_hashed_poles() -> None:
    profile = SimpleNamespace(
        schema_version=2,
        flight=SimpleNamespace(
            approved=True,
            coordinate_frame_id="frame",
            route_clearance_approved=True,
            approval_note="fixture",
            controller=SimpleNamespace(inspect_waypoints=(1,)),
        ),
        localizer="edm",
        route_json=Path("route.json"),
        map_reference_poses=Path("refs.json"),
        map_align=Path("T_align_gravity.json"),
        query_camera=SimpleNamespace(
            model="PINHOLE",
            width=1280,
            height=720,
            params=(900.0, 900.0, 640.0, 360.0),
        ),
        localizer_profile=Path("edm.json"),
        poles_json=Path("poles.json"),
        asset_sha256=SimpleNamespace(
            localization_bundle="a" * 64,
            route_json="b" * 64,
            map_reference_poses="c" * 64,
            map_align="e" * 64,
            localizer_profile="d" * 64,
            poles_json=None,
        ),
    )

    assert (
        "inspection waypoints require asset_sha256.poles_json"
        in mission_pipeline.flight_readiness_errors(profile)
    )


def test_approved_flight_profile_rejects_tampered_asset(tmp_path, monkeypatch):
    profile = SimpleNamespace(
        flight=SimpleNamespace(
            approved=True,
            coordinate_frame_id="frame",
            route_clearance_approved=True,
            approval_note="fixture",
            controller=SimpleNamespace(inspect_waypoints=()),
        ),
        site_id="alpha",
        localizer="edm",
        query_camera=SimpleNamespace(
            model="PINHOLE",
            width=1280,
            height=720,
            params=(900.0, 900.0, 640.0, 360.0),
        ),
        route_json=tmp_path / "route.json",
        map_reference_poses=tmp_path / "refs.json",
        map_align=tmp_path / "T_align_gravity.json",
        localization_bundle=tmp_path / "bundle.pt",
        localizer_profile=tmp_path / "edm.json",
        poles_json=None,
        asset_sha256=SimpleNamespace(
            localization_bundle="0" * 64,
            route_json="0" * 64,
            map_reference_poses="0" * 64,
            map_align="0" * 64,
            localizer_profile="0" * 64,
            poles_json=None,
        ),
    )
    profile.route_json.write_text(json.dumps({
        "schema": "sfm-flight-route/v1",
        "site_id": "alpha",
        "coordinate_frame_id": "frame",
        "frame": "glomap",
        "units": "map",
        "purpose": "flight",
        "closed": False,
        "waypoints": [[0, 0, 0], [1, 0, 0]],
    }), encoding="utf-8")
    for path in (
        profile.map_reference_poses,
        profile.localization_bundle,
        profile.localizer_profile,
    ):
        path.write_bytes(b"x")

    monkeypatch.setattr(
        mission_pipeline,
        "flight_readiness_errors",
        lambda _profile: [],
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        mission_pipeline.validate_profile_flight_assets(profile)
