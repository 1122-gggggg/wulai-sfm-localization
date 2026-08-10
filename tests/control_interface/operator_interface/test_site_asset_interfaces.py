from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import local_site_assets as local_assets
from local_site_assets import (
    LocalRouteProvider,
    LocalSitePackageProvider,
    LocalTargetProvider,
    discover_ply_files,
    inspect_edm_bundle,
    load_edm_runtime_profile,
    match_managed_site_profile_for_map,
)
from operator_actions import replace_site_profile_argument, require_safe_site_switch
from site_profile import load_site_profile

CAMERA = {
    "model": "SIMPLE_RADIAL",
    "width": 1280,
    "height": 720,
    "params": [934.139423, 640.0, 360.0, 0.001061],
}


def test_edm_asset_inspection_does_not_mutate_ui_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        payload = (
            '["ref-a.jpg"]'
            if "EDMRelocMap" in args[3]
            else '{"schema":"edm-deployment-profile/v1"}'
        )
        return subprocess.CompletedProcess(
            args, 0, stdout=f"noise\nSFM_ASSET_RESULT:{payload}\n", stderr=""
        )

    monkeypatch.setattr(local_assets.subprocess, "run", fake_run)
    before = list(sys.path)

    assert inspect_edm_bundle(tmp_path / "bundle.pt") == ("ref-a.jpg",)
    assert load_edm_runtime_profile(tmp_path / "profile.json")["schema"].endswith("/v1")
    assert sys.path == before
    assert len(calls) == 2
    assert all(call[1]["cwd"].name == "sfm_glomap_deploy" for call in calls)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_site_package(root: Path) -> Path:
    # Site packages are loaded from a workspace, not an arbitrary filesystem
    # folder; mirror the production marker directories in this fixture.
    for name in ("控制介面程式", "定位演算法", "地圖檔"):
        (root / name).mkdir(parents=True, exist_ok=True)
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
        "map_align": "T_align_gravity.json",
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
    # A real package carries its measured gravity alignment; without it every
    # imported site silently reverts to the legacy "-Y is up" guess.
    (package / "T_align_gravity.json").write_text(
        json.dumps({
            "schema": "sfm-align/v2",
            "R": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            "gravity_glomap": [0.0, 1.0, 0.0],
        }),
        encoding="utf-8",
    )
    profile["asset_sha256"] = {
        "map_align": _sha(package / "T_align_gravity.json"),
        "map_ply": _sha(package / "map.ply"),
        "localization_bundle": _sha(package / "localization_bundle.pt"),
        "localizer_profile": _sha(package / "edm_runtime_profile.json"),
        "map_reference_poses": _sha(package / "reference_poses.json"),
    }
    (package / "site_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return package


def _profile_loader(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == "edm-deployment-profile/v1"
    return raw


def _write_valid_route(path: Path) -> Path:
    path.write_text(
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
    return path


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


def test_ply_digest_uniquely_matches_an_imported_site(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    provider.import_folder(package)

    profile = match_managed_site_profile_for_map(package / "map.ply", managed)

    assert profile is not None
    assert profile.site_id == "test-yard"
    assert profile.coordinate_frame is not None
    assert profile.coordinate_frame.id == "test-yard-reconstruction-v1"


def test_map_folder_discovers_nested_ply_files_in_stable_order(tmp_path: Path) -> None:
    folder = tmp_path / "maps"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "z.ply").write_bytes(b"z")
    (nested / "A.PLY").write_bytes(b"a")
    (folder / "ignore.txt").write_bytes(b"x")

    found = discover_ply_files(folder)

    assert [path.relative_to(folder).as_posix() for path in found] == [
        "nested/A.PLY",
        "z.ply",
    ]


def test_map_folder_with_no_ply_returns_empty_list(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("none", encoding="utf-8")

    assert discover_ply_files(tmp_path) == []


def test_map_folder_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        discover_ply_files(tmp_path / "missing")


def test_map_binding_rejects_unsupported_ply_format(tmp_path: Path) -> None:
    path = tmp_path / "big-endian.ply"
    path.write_text(
        "ply\nformat binary_big_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="ASCII or binary little-endian"):
        match_managed_site_profile_for_map(path, tmp_path / "managed")


def test_map_binding_requires_rgb_vertex_properties(tmp_path: Path) -> None:
    path = tmp_path / "xyz-only.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="missing UI vertex properties"):
        match_managed_site_profile_for_map(path, tmp_path / "managed")


def test_unmatched_ply_stays_unbound(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    provider.import_folder(package)
    other = tmp_path / "other.ply"
    other.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n1 2 3 4 5 6\n",
        encoding="ascii",
    )

    assert match_managed_site_profile_for_map(other, managed) is None


def test_ambiguous_ply_match_is_rejected(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    imported = provider.import_folder(package)
    shutil.copytree(imported.profile_path.parent, managed / "duplicate")

    with pytest.raises(ValueError, match="同時匹配多個"):
        match_managed_site_profile_for_map(package / "map.ply", managed)


def test_site_package_rejects_reference_names_not_in_bundle(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path: ("different.jpg",),
        edm_profile_loader=_profile_loader,
    )

    # The bundle offers a reference the poses file cannot place. That direction is
    # the dangerous one and is still refused; extra poses are now allowed.
    with pytest.raises(ValueError, match="have no pose"):
        provider.validate_folder(package)


def test_site_package_rejects_asset_path_outside_selected_folder(
    tmp_path: Path,
) -> None:
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
    assert (
        profile.poles_json
        == managed / "test-yard" / "targets" / "inspection_targets.json"
    )
    assert profile.asset_sha256.route_json == _sha(profile.route_json)
    assert profile.asset_sha256.poles_json == _sha(profile.poles_json)
    assert profile.flight is not None and profile.flight.approved is False


def test_route_commit_rolls_back_asset_when_profile_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    profile_path = site_provider.import_folder(package).profile_path
    profile_before = profile_path.read_bytes()
    route = _write_valid_route(tmp_path / "route.json")
    real_replace = local_assets.os.replace

    def fail_profile_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == profile_path:
            raise OSError("forced profile replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(local_assets.os, "replace", fail_profile_replace)

    with pytest.raises(OSError, match="forced profile replace failure"):
        LocalRouteProvider(managed).import_file(route, profile_path)

    route_target = managed / "test-yard" / "routes" / "flight_route.json"
    assert not route_target.exists()
    assert profile_path.read_bytes() == profile_before
    assert not [path for path in managed.rglob("*") if ".tmp" in path.name]


def test_concurrent_route_imports_commit_without_temp_collision(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    profile_path = site_provider.import_folder(package).profile_path
    route = _write_valid_route(tmp_path / "route.json")
    copied = threading.Barrier(2)
    real_copy2 = local_assets.shutil.copy2

    def synchronized_copy2(source: str | Path, destination: str | Path, *args, **kwargs):
        result = real_copy2(source, destination, *args, **kwargs)
        if Path(destination).name.endswith(".tmp"):
            copied.wait(timeout=5)
        return result

    monkeypatch.setattr(local_assets.shutil, "copy2", synchronized_copy2)

    def import_route() -> object:
        return LocalRouteProvider(managed).import_file(route, profile_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: import_route(), range(2)))

    target = managed / "test-yard" / "routes" / "flight_route.json"
    profile = load_site_profile(profile_path)
    assert all(result.asset_path == target for result in results)
    assert target.read_bytes() == route.read_bytes()
    assert profile.route_json == target
    assert profile.asset_sha256.route_json == _sha(target)
    assert not [path for path in managed.rglob("*") if ".tmp" in path.name]


def test_concurrent_site_package_imports_have_one_winner(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    validated = threading.Barrier(2)
    real_validate = provider.validate_folder

    def synchronized_validate(folder: str | Path):
        report = real_validate(folder)
        validated.wait(timeout=5)
        return report

    monkeypatch.setattr(provider, "validate_folder", synchronized_validate)

    def import_package() -> object:
        return provider.import_folder(package)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: import_package(), range(2)))

    assert sorted(result.already_present for result in results) == [False, True]
    profile_path = managed / "test-yard" / "site_profile.json"
    assert profile_path.is_file()
    assert not [path for path in managed.rglob("*") if ".tmp" in path.name]


def test_route_import_resolves_system_profile_to_its_managed_site(
    tmp_path: Path,
) -> None:
    """The operator commonly starts from control/site_profiles, not the pack."""
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )
    imported_profile_path = site_provider.import_folder(package).profile_path
    site_folder = managed / "operator-folder"
    imported_profile_path.parent.rename(site_folder)
    managed_profile_path = site_folder / "site_profile.json"
    managed_profile = load_site_profile(managed_profile_path)

    raw = json.loads(managed_profile_path.read_text(encoding="utf-8"))
    raw["localizer_profile"] = str(managed_profile.localizer_profile)
    raw["map_reference_poses"] = str(managed_profile.map_reference_poses)
    raw["map_align"] = str(managed_profile.map_align)
    raw["assets"]["map_ply"] = str(managed_profile.map_ply)
    raw["assets"]["localization_bundle"] = str(
        managed_profile.localization_bundle
    )
    system_profiles = tmp_path / "control" / "site_profiles"
    system_profiles.mkdir(parents=True)
    system_profile_path = system_profiles / "test_yard_edm.json"
    system_profile_path.write_text(json.dumps(raw), encoding="utf-8")

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

    imported = LocalRouteProvider(managed).import_file(route, system_profile_path)

    assert imported.profile_path == managed_profile_path
    assert imported.asset_path == site_folder / "routes" / "flight_route.json"
    updated = load_site_profile(managed_profile_path)
    assert updated.route_json == imported.asset_path
    assert updated.asset_sha256.route_json == _sha(imported.asset_path)

    mismatched = json.loads(system_profile_path.read_text(encoding="utf-8"))
    mismatched["site_id"] = "wrong-site"
    system_profile_path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="尚未對應"):
        LocalRouteProvider(managed).import_file(route, system_profile_path)


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


def test_site_package_requires_and_carries_its_gravity_alignment(tmp_path: Path) -> None:
    """Dropping map_align on import made every managed site revert to the legacy
    "GLOMAP -Y is up" guess, which is 22.51 deg wrong on target_site_v1."""
    package = _write_site_package(tmp_path)
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )

    report = provider.validate_folder(package)
    assert any(check.key == "map_align" for check in report.checks), (
        "the alignment is not a required asset"
    )

    imported = provider.import_folder(package)
    managed = json.loads(imported.profile_path.read_text(encoding="utf-8"))
    assert managed.get("map_align"), "the managed profile has no map_align"
    assert (imported.profile_path.parent / managed["map_align"]).is_file()
    assert managed["asset_sha256"].get("map_align"), "the alignment is not pinned"


def test_site_package_without_a_gravity_alignment_is_refused(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    raw = json.loads((package / "site_profile.json").read_text(encoding="utf-8"))
    del raw["map_align"]
    (package / "site_profile.json").write_text(json.dumps(raw), encoding="utf-8")
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path: ("ref-a.jpg",),
        edm_profile_loader=_profile_loader,
    )

    with pytest.raises(ValueError, match="map_align"):
        provider.validate_folder(package)
