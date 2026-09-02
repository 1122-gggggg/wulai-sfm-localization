from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "控制介面程式"
FLIGHT_CONTROL = ROOT / "定位演算法" / "flight_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))
if str(FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(FLIGHT_CONTROL))

from mission_manifest import ManifestError  # noqa: E402
from mission_resolver import resolve_mission  # noqa: E402
from pose_frame_chain import CameraBodyExtrinsic, save_camera_body_extrinsic  # noqa: E402
from site_alignment import (  # noqa: E402
    SiteAlignment,
    save_site_alignment,
    solve_similarity_alignment,
)


NOW = datetime(2026, 8, 11, tzinfo=timezone.utc)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_asset(path: Path, value: bytes = b"asset") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(path: Path, *, base: Path) -> dict[str, str]:
    return {
        "path": os.path.relpath(path, base),
        "sha256": _digest(path),
    }


def _build_selection(
    root: Path,
    *,
    include_route: bool = False,
    include_imu: bool = False,
    include_pose_chain: bool = False,
    include_approval: bool = False,
    localizer_map_revision: str = "map_v1",
) -> Path:
    assets = root / "assets"
    map_ply = _write_asset(assets / "map.ply")
    poses = _write_asset(assets / "poses.json", b"{}")
    align = _write_asset(assets / "align.json", b"{}")
    bundle = _write_asset(assets / "bundle.pt")
    profile = _write_asset(assets / "profile.json", b"{}")

    vehicle = _write_json(
        root / "components" / "vehicle.json",
        {
            "schema": "sfm-vehicle/v1",
            "vehicle_id": "aircraft_a",
            "revision": "r1",
            "adapter": "test_adapter",
            "model": "Test aircraft",
            "serials": [],
            "capabilities": ["rgb_720p", "imu"],
            "required_calibrations": ["imu", "camera_body_extrinsic"],
            "limits": {"max_speed_mps": 2.0},
            "camera": {
                "camera_id": "front",
                "pipeline_id": "camera_pipeline_v1",
                "model": "PINHOLE",
                "width": 1280,
                "height": 720,
                "params": [900.0, 900.0, 640.0, 360.0],
            },
        },
    )
    site = _write_json(
        root / "components" / "site.json",
        {
            "schema": "sfm-site/v1",
            "site_id": "site_a",
            "display_name": "Site A",
            "site_frame_id": "site_a_enu",
            "units": "m",
        },
    )
    map_manifest = _write_json(
        root / "components" / "map.json",
        {
            "schema": "sfm-map-revision/v1",
            "site_id": "site_a",
            "map_revision_id": "map_v1",
            "coordinate_frame": {
                "id": "map_v1_frame",
                "convention": "glomap",
                "horizontal_axes": ["x", "z"],
                "up_axis": "-y",
                "handedness": "right",
                "units": "map",
            },
            "assets": {
                "map_ply": _reference(map_ply, base=root / "components"),
                "reference_poses": _reference(poses, base=root / "components"),
                "map_align": _reference(align, base=root / "components"),
            },
        },
    )
    localizer = _write_json(
        root / "components" / "localizer.json",
        {
            "schema": "sfm-localizer-variant/v1",
            "algorithm_id": "test_localizer",
            "variant_id": "variant_v1",
            "provider_api_version": 1,
            "pose_contract_version": 1,
            "map_revision_id": localizer_map_revision,
            "coordinate_frame_id": "map_v1_frame",
            "camera_profiles": ["camera_pipeline_v1"],
            "required_vehicle_capabilities": ["rgb_720p"],
            "quality_gate_id": "quality_gate_v1",
            "artifacts": {
                "bundle": _reference(bundle, base=root / "components"),
                "profile": _reference(profile, base=root / "components"),
            },
            "runtime": {"device": "test"},
        },
    )
    quality = _write_json(
        root / "receipts" / "quality.json",
        {
            "schema": "sfm-calibration-receipt/v1",
            "receipt_id": "quality_v1",
            "kind": "localizer_quality",
            "subject": "quality_gate_v1",
            "passed": True,
            "issued_at": "2026-08-01T00:00:00Z",
            "expires_at": None,
            "details": {},
        },
    )

    route_manifest = None
    if include_route:
        route_asset = _write_json(
            root / "routes" / "flight_route.json",
            {
                "schema": "sfm-flight-route/v1",
                "site_id": "site_a",
                "coordinate_frame_id": "map_v1_frame",
                "frame": "aligned",
                "align_source": "measured",
                "units": "map",
                "purpose": "flight",
                "closed": False,
                "waypoints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            },
        )
        route_manifest = _write_json(
            root / "components" / "route.json",
            {
                "schema": "sfm-route-package/v1",
                "route_id": "route_a",
                "revision": "r1",
                "site_id": "site_a",
                "frame": {"kind": "map", "id": "map_v1_frame"},
                "route": _reference(route_asset, base=root / "components"),
            },
        )

    calibrations = [quality]
    if include_imu:
        calibrations.append(
            _write_json(
                root / "receipts" / "imu.json",
                {
                    "schema": "sfm-calibration-receipt/v1",
                    "receipt_id": "imu_v1",
                    "kind": "imu",
                    "subject": "aircraft_a@r1",
                    "passed": True,
                    "issued_at": "2026-08-01T00:00:00Z",
                    "expires_at": "2027-08-01T00:00:00Z",
                    "details": {},
                },
            )
        )

    if include_pose_chain:
        controls = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
        )
        fit = solve_similarity_alignment(controls, controls)
        site_alignment = save_site_alignment(
            SiteAlignment.from_fit(
                map_frame_id="map_v1_frame",
                site_frame_id="site_a_enu",
                fit=fit,
                approved=True,
            ),
            root / "calibration" / "site_alignment.json",
        )
        camera_body = save_camera_body_extrinsic(
            CameraBodyExtrinsic(
                vehicle_id="aircraft_a@r1",
                body_frame_id="body_frd",
                camera_frame_id="camera_opencv",
                rotation_camera_from_body=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                translation_camera_from_body_m=(0.0, 0.0, 0.0),
                fixed_gimbal_pitch_deg=0.0,
                gimbal_pitch_tolerance_deg=1.0,
                evidence="test fixture",
                approved=True,
            ),
            root / "calibration" / "camera_body_extrinsic.json",
        )
        calibrations.extend(
            (
                _write_json(
                    root / "receipts" / "site_alignment.json",
                    {
                        "schema": "sfm-calibration-receipt/v1",
                        "receipt_id": "site_alignment_v1",
                        "kind": "site_alignment",
                        "subject": "site_a@map_v1",
                        "passed": True,
                        "issued_at": "2026-08-01T00:00:00Z",
                        "expires_at": None,
                        "details": {
                            "artifact": _reference(
                                site_alignment,
                                base=root / "receipts",
                            )
                        },
                    },
                ),
                _write_json(
                    root / "receipts" / "camera_body.json",
                    {
                        "schema": "sfm-calibration-receipt/v1",
                        "receipt_id": "camera_body_v1",
                        "kind": "camera_body_extrinsic",
                        "subject": "aircraft_a@r1",
                        "passed": True,
                        "issued_at": "2026-08-01T00:00:00Z",
                        "expires_at": None,
                        "details": {
                            "artifact": _reference(
                                camera_body,
                                base=root / "receipts",
                            )
                        },
                    },
                ),
            )
        )

    selection_dir = root / "selections"
    components = {
        "vehicle": _reference(vehicle, base=selection_dir),
        "site": _reference(site, base=selection_dir),
        "map": _reference(map_manifest, base=selection_dir),
        "localizer": _reference(localizer, base=selection_dir),
        "route": (
            None if route_manifest is None else _reference(route_manifest, base=selection_dir)
        ),
        "calibrations": [_reference(receipt, base=selection_dir) for receipt in calibrations],
    }
    approval = None
    if include_approval:
        bindings = {
            "vehicle": components["vehicle"]["sha256"],
            "site": components["site"]["sha256"],
            "map": components["map"]["sha256"],
            "localizer": components["localizer"]["sha256"],
            "route": components["route"]["sha256"],
        }
        for index, receipt in enumerate(components["calibrations"]):
            bindings[f"calibration_{index}"] = receipt["sha256"]
        approval = _write_json(
            root / "approvals" / "approval.json",
            {
                "schema": "sfm-mission-approval/v1",
                "approval_id": "approval_v1",
                "approved": True,
                "route_clearance_approved": True,
                "component_sha256": bindings,
                "note": "test approval",
            },
        )

    return _write_json(
        selection_dir / "mission.json",
        {
            "schema": "sfm-mission-selection/v1",
            "selection_id": "mission_a",
            **components,
            "approval": (None if approval is None else _reference(approval, base=selection_dir)),
        },
    )


def test_localization_can_be_ready_while_flight_remains_blocked(tmp_path: Path) -> None:
    selection = _build_selection(tmp_path)

    mission = resolve_mission(selection, workspace_root=tmp_path, now=NOW)

    assert mission.readiness.localization_ready
    assert not mission.readiness.flight_ready
    assert mission.readiness.flight_errors == (
        "no route package selected",
        "missing current imu calibration for aircraft_a@r1",
    )
    document = mission.legacy_site_profile_document()
    assert document["localizer"] == "test_localizer"
    assert document["flight"]["approved"] is False
    assert document["assets"]["route_json"] is None


def test_full_selection_is_flight_ready_without_static_mission_approval(
    tmp_path: Path,
) -> None:
    selection = _build_selection(
        tmp_path,
        include_route=True,
        include_imu=True,
        include_pose_chain=True,
    )

    mission = resolve_mission(selection, workspace_root=tmp_path, now=NOW)

    assert mission.readiness.localization_ready
    assert mission.readiness.flight_ready
    assert mission.pose_chain is not None
    document = mission.legacy_site_profile_document()
    assert document["pose_chain"]["body_frame_id"] == "body_frd"
    assert document["asset_sha256"]["site_alignment"] == mission.pose_chain.site_alignment.sha256
    mission.verify_unchanged()


def test_explicit_incomplete_pose_chain_blocks_flight_without_blocking_localization(
    tmp_path: Path,
) -> None:
    selection = _build_selection(
        tmp_path,
        include_route=True,
        include_imu=True,
        include_pose_chain=True,
    )
    raw = json.loads(selection.read_text(encoding="utf-8"))
    raw["calibrations"] = raw["calibrations"][:-1]
    _write_json(selection, raw)

    mission = resolve_mission(selection, workspace_root=tmp_path, now=NOW)

    assert mission.readiness.localization_ready
    assert not mission.readiness.flight_ready
    assert mission.pose_chain is None
    assert "camera_body_extrinsic" in " ".join(mission.readiness.flight_errors)


def test_localizer_built_for_another_map_is_rejected(tmp_path: Path) -> None:
    selection = _build_selection(tmp_path, localizer_map_revision="map_v2")

    mission = resolve_mission(selection, workspace_root=tmp_path, now=NOW)

    assert not mission.readiness.localization_ready
    assert "localizer map revision does not match" in " ".join(
        mission.readiness.localization_errors
    )


def test_snapshot_detects_artifact_change_after_resolution(tmp_path: Path) -> None:
    selection = _build_selection(tmp_path)
    mission = resolve_mission(selection, workspace_root=tmp_path, now=NOW)
    mission.localizer.artifacts["bundle"].path.write_bytes(b"changed")

    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        mission.verify_unchanged()


def test_selection_and_materialized_snapshot_paths_with_spaces(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace with spaces"
    selection = _build_selection(workspace)

    mission = resolve_mission(selection, workspace_root=workspace, now=NOW)
    snapshot_dir = workspace / "runtime with spaces" / "mission snapshots"
    snapshot = mission.materialize_legacy_site_profile(snapshot_dir)

    assert " " in str(selection)
    assert snapshot.parent == snapshot_dir
    assert snapshot.is_file()
    assert mission.readiness.localization_ready


def test_invalid_selection_path_is_rejected_before_materialization(tmp_path: Path) -> None:
    missing = tmp_path / "selection with spaces.json"

    with pytest.raises(ManifestError, match="selection manifest is not a file"):
        resolve_mission(missing, workspace_root=tmp_path, now=NOW)


def test_not_localization_ready_selection_is_explicitly_blocked(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace with spaces"
    selection = _build_selection(workspace, localizer_map_revision="map_v2")

    mission = resolve_mission(selection, workspace_root=workspace, now=NOW)

    assert not mission.readiness.localization_ready
    assert "localizer map revision does not match" in " ".join(
        mission.readiness.localization_errors
    )


def test_bundled_anafi_selection_is_blocked_until_new_map_is_validated() -> None:
    selection = (
        CONTROL
        / "mission_selections"
        / "river_gluemap_all8_direct_localization.json"
    )

    mission = resolve_mission(selection, workspace_root=ROOT, now=NOW)

    assert "imu" not in mission.vehicle.required_calibrations
    assert not mission.readiness.localization_ready
    assert not mission.readiness.flight_ready
    assert "localizer_quality calibration is failed" in " ".join(
        mission.readiness.localization_errors
    )
    assert "no route package selected" in mission.readiness.flight_errors
