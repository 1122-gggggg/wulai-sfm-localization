from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from local_site_assets import (
    LocalRouteProvider,
    LocalSitePackageProvider,
    LocalTargetProvider,
)
from operator_actions import require_safe_site_switch, replace_site_profile_argument
from site_profile import load_site_profile


CAMERA = {
    "model": "SIMPLE_RADIAL",
    "width": 1280,
    "height": 720,
    "params": [934.139423, 640.0, 360.0, 0.001061],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_site_package(root: Path) -> Path:
    package = root / "builder-output"
    package.mkdir()
    (package / "map.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 0 255 255 255\n",
        encoding="ascii",
    )
    (package / "localization_bundle.pt").write_bytes(b"test-edm-bundle")
    (package / "edm_runtime_profile.json").write_text(
        json.dumps({"schema": "edm-deployment-profile/v1"}), encoding="utf-8"
    )
    (package / "reference_poses.json").write_text(
        json.dumps(
            {
                "camera": CAMERA,
                "poses": {
                    "ref-a.jpg": {
                        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "t": [0, 0, 0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    profile = {
        "schema_version": 2,
        "site_id": "test-yard",
        "display_name": "Test Yard",
        "localizer": "edm",
        "localizer_profile": "edm_runtime_profile.json",
        "map_reference_poses": "reference_poses.json",
        "query_camera": CAMERA,
        "coordinate_frame": {
            "id": "test-yard-reconstruction-v1",
            "convention": "glomap",
            "horizontal_axes": ["x", "z"],
            "up_axis": "-y",
            "handedness": "right",
            "units": "map",
        },
        "assets": {
            "map_ply": "map.ply",
            "route_json": None,
            "localization_bundle": "localization_bundle.pt",
            "megaloc_cache": None,
            "track_landmarks": None,
            "poles_json": None,
        },
    }
    profile["asset_sha256"] = {
        "map_ply": _sha(package / "map.ply"),
        "localization_bundle": _sha(package / "localization_bundle.pt"),
        "localizer_profile": _sha(package / "edm_runtime_profile.json"),
        "map_reference_poses": _sha(package / "reference_poses.json"),
    }
    (package / "site_profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    return package


def _profile_loader(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == "edm-deployment-profile/v1"
    return raw


def test_site_package_is_validated_and_imported_atomically(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )

    report = provider.validate_folder(package)
    imported = provider.import_folder(package)
    profile = load_site_profile(imported.profile_path)

    assert report.site_id == "test-yard"
    assert report.reference_count == 1
    assert imported.profile_path == managed / "test-yard" / "site_profile.json"
    assert profile.map_ply == managed / "test-yard" / "map" / "map.ply"
    assert profile.flight is not None
    assert profile.flight.approved is False
    assert profile.flight.route_clearance_approved is False


def test_site_package_rejects_reference_names_not_in_bundle(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path: ("different.jpg",),
        edm_profile_loader=_profile_loader,
    )

    with pytest.raises(ValueError, match="reference image names"):
        provider.validate_folder(package)


def test_site_package_rejects_asset_path_outside_selected_folder(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    outside = tmp_path / "outside.ply"
    outside.write_bytes((package / "map.ply").read_bytes())
    profile_path = package / "site_profile.json"
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["assets"]["map_ply"] = "../outside.ply"
    raw["asset_sha256"]["map_ply"] = _sha(outside)
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )

    with pytest.raises(ValueError, match="inside the selected folder"):
        provider.validate_folder(package)


def test_route_and_target_are_independent_profile_ports(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    profile_path = site_provider.import_folder(package).profile_path
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "schema": "sfm-flight-route/v1",
                "site_id": "test-yard",
                "coordinate_frame_id": "test-yard-reconstruction-v1",
                "frame": "glomap",
                "units": "map",
                "purpose": "flight",
                "closed": False,
                "waypoints": [[0, 0, 0], [1, 0, 0]],
            }
        ),
        encoding="utf-8",
    )
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "schema": "sfm-inspection-targets/v1",
                "site_id": "test-yard",
                "coordinate_frame_id": "test-yard-reconstruction-v1",
                "frame": "aligned",
                "units": "map",
                "poles": [{"base": [0, 0, 0], "top": [0, 0, 2]}],
            }
        ),
        encoding="utf-8",
    )

    LocalRouteProvider(managed).import_file(route, profile_path)
    LocalTargetProvider(managed).import_file(targets, profile_path)
    profile = load_site_profile(profile_path)

    assert profile.route_json == managed / "test-yard" / "routes" / "flight_route.json"
    assert profile.poles_json == managed / "test-yard" / "targets" / "inspection_targets.json"
    assert profile.asset_sha256.route_json == _sha(profile.route_json)
    assert profile.asset_sha256.poles_json == _sha(profile.poles_json)
    assert profile.flight is not None and profile.flight.approved is False


def test_restart_argument_replaces_only_site_profile() -> None:
    argv = ["app.py", "--live", "--site-profile", "old.json", "--auto-inspect"]

    assert replace_site_profile_argument(argv, Path("/tmp/new.json")) == [
        "app.py",
        "--live",
        "--auto-inspect",
        "--site-profile",
        "/tmp/new.json",
    ]


@pytest.mark.parametrize("state", [None, "unknown", "hovering", "flying"])
def test_live_site_switch_requires_confirmed_landed_state(state: str | None) -> None:
    with pytest.raises(ValueError, match="只有 landed"):
        require_safe_site_switch(
            is_live=True, flight_state=state, commands_inflight=False
        )


def test_live_site_switch_rejects_pending_flight_command() -> None:
    with pytest.raises(ValueError, match="飛行指令處理中"):
        require_safe_site_switch(
            is_live=True, flight_state="landed", commands_inflight=True
        )
