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
    inspect_direct_bundle,
    load_direct_runtime_profile,
    match_managed_site_profile_for_map,
    preview_site_folder,
    profile_for_selected_route,
)
from operator_actions import replace_site_profile_argument, require_safe_site_switch
from site_profile import load_site_profile

CAMERA = {
    "model": "SIMPLE_RADIAL",
    "width": 1280,
    "height": 720,
    "params": [934.139423, 640.0, 360.0, 0.001061],
}


def test_direct_asset_inspection_does_not_mutate_ui_import_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        payload = (
            '["ref-a.jpg"]'
            if "DirectMapAssets" in args[3]
            else '{"name": "test-direct", "map_scale": 1.0, "reloc_top_k": 2}'
        )
        return subprocess.CompletedProcess(
            args, 0, stdout=f"noise\nSFM_ASSET_RESULT:{payload}\n", stderr=""
        )

    monkeypatch.setattr(local_assets.subprocess, "run", fake_run)
    before = list(sys.path)

    assert inspect_direct_bundle(tmp_path / "bundle.json", "a" * 64) == ("ref-a.jpg",)
    assert load_direct_runtime_profile(tmp_path / "profile.json")["reloc_top_k"] == 2
    assert sys.path == before
    assert len(calls) == 2
    assert calls[0][0][-1] == "a" * 64
    assert all(call[1]["cwd"].name == "sfm_direct_deploy" for call in calls)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_site_package(root: Path) -> Path:
    # Site packages are loaded from a workspace, not an arbitrary filesystem
    # folder; mirror the production marker directories in this fixture.
    for name in ("控制介面程式", "定位演算法", "地圖檔"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "定位演算法" / "deploy_code" / "sfm_direct_deploy").mkdir(
        parents=True, exist_ok=True
    )
    package = root / "builder-output"
    package.mkdir()
    (package / "map.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n0 0 0 255 255 255\n",
        encoding="ascii",
    )
    (package / "direct_bundle.json").write_text(
        json.dumps({"schema": "direct-localization-bundle/v1"}), encoding="utf-8"
    )
    (package / "direct_runtime_profile.json").write_text(
        json.dumps({"schema": "direct-deployment-profile/v1"}), encoding="utf-8"
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
        "localizer": "direct",
        "localizer_deploy_dir": "../定位演算法/deploy_code/sfm_direct_deploy",
        "localizer_profile": "direct_runtime_profile.json",
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
            "localization_bundle": "direct_bundle.json",
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
        "localization_bundle": _sha(package / "direct_bundle.json"),
        "localizer_profile": _sha(package / "direct_runtime_profile.json"),
        "map_reference_poses": _sha(package / "reference_poses.json"),
    }
    (package / "site_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return package


def _profile_loader(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == "direct-deployment-profile/v1"
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

    def inspect_bundle(path: Path, expected_sha256: str) -> tuple[str, ...]:
        assert path == package / "direct_bundle.json"
        assert expected_sha256 == _sha(path)
        return ("ref-a.jpg",)

    provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=inspect_bundle,
        profile_loader=_profile_loader,
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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )
    imported = provider.import_folder(package)
    shutil.copytree(imported.profile_path.parent, managed / "duplicate")

    with pytest.raises(ValueError, match="同時匹配多個"):
        match_managed_site_profile_for_map(package / "map.ply", managed)


def test_site_package_rejects_reference_names_not_in_bundle(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path, _sha256: ("different.jpg",),
        profile_loader=_profile_loader,
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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )

    with pytest.raises(ValueError, match="inside the selected folder"):
        provider.validate_folder(package)


def test_folder_preview_offers_import_only_for_a_complete_site_folder(
    tmp_path: Path,
) -> None:
    package = _write_site_package(tmp_path)

    complete = preview_site_folder(package)
    assert complete.ready
    assert (complete.site_id, complete.display_name) == ("test-yard", "Test Yard")
    assert complete.coordinate_frame_id == "test-yard-reconstruction-v1"

    (package / "reference_poses.json").unlink()
    missing = preview_site_folder(package)
    assert not missing.ready
    assert [item.label for item in missing.files if item.problem] == ["參考影像位姿"]

    no_manifest = preview_site_folder(tmp_path)
    assert not no_manifest.ready and not no_manifest.files


def test_folder_preview_refuses_what_the_importer_would_refuse(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    outside = tmp_path / "outside.ply"
    outside.write_bytes((package / "map.ply").read_bytes())
    profile_path = package / "site_profile.json"
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["assets"]["map_ply"] = "../outside.ply"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    outside_preview = preview_site_folder(package)
    assert not outside_preview.ready
    assert [item.label for item in outside_preview.files if item.problem] == ["點雲地圖"]

    raw["assets"]["map_ply"] = "map.ply"
    raw["assets"]["route_json"] = "map.ply"
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    embedded = preview_site_folder(package)
    assert not embedded.ready
    assert "route_json" in (embedded.error or "")

    profile_path.write_text("not JSON", encoding="utf-8")
    unreadable = preview_site_folder(package)
    assert not unreadable.ready and unreadable.error


def test_route_and_target_are_independent_profile_ports(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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
    route_profile = load_site_profile(profile_path)
    assert route_profile.flight is not None
    assert route_profile.flight.approved is False
    assert route_profile.flight.route_clearance_approved is False

    LocalTargetProvider(managed).import_file(targets, profile_path)
    profile = load_site_profile(profile_path)

    assert profile.route_json is not None
    assert profile.route_json.parent == managed / "test-yard" / "routes"
    assert profile.route_json.name.startswith("flight_route_")
    assert (
        profile.poles_json
        == managed / "test-yard" / "targets" / "inspection_targets.json"
    )
    assert profile.asset_sha256.route_json == _sha(profile.route_json)
    assert profile.asset_sha256.poles_json == _sha(profile.poles_json)
    assert profile.flight is not None and profile.flight.approved is False


def test_editor_authored_route_can_be_approved_for_auto(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )
    profile_path = site_provider.import_folder(package).profile_path
    route = _write_valid_route(tmp_path / "route.json")
    route_payload = json.loads(route.read_text(encoding="utf-8"))
    route_payload["source"] = "operator_route_editor"
    route.write_text(json.dumps(route_payload), encoding="utf-8")

    imported = LocalRouteProvider(managed).import_file(
        route,
        profile_path,
        approve_for_auto=True,
    )
    profile = load_site_profile(profile_path)

    assert imported.asset_path == profile.route_json
    assert profile.flight is not None
    assert profile.flight.approved is True
    assert profile.flight.route_clearance_approved is True
    assert "operator route editor" in profile.flight.approval_note.lower()
    assert profile.asset_sha256.route_json == _sha(imported.asset_path)


def test_route_import_updates_linked_component_and_selection_sha(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "地圖檔" / "場域"
    profile_path = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    ).import_folder(package).profile_path
    component_path = (
        tmp_path
        / "控制介面程式"
        / "mission_components"
        / "routes"
        / "test_route.json"
    )
    selection_path = (
        tmp_path
        / "控制介面程式"
        / "mission_selections"
        / "test_selection.json"
    )
    component_path.parent.mkdir(parents=True)
    selection_path.parent.mkdir(parents=True)
    component_path.write_text(
        json.dumps(
            {
                "schema": "sfm-route-package/v1",
                "route": {
                    "path": "../../../地圖檔/場域/test-yard/routes/flight_route.json",
                    "sha256": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(
            {
                "schema": "sfm-mission-selection/v1",
                "route": {
                    "path": "../mission_components/routes/test_route.json",
                    "sha256": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    route = _write_valid_route(tmp_path / "route.json")
    route_payload = json.loads(route.read_text(encoding="utf-8"))
    route_payload["source"] = "operator_route_editor"
    route.write_text(json.dumps(route_payload), encoding="utf-8")

    imported = LocalRouteProvider(managed).import_file(
        route, profile_path, approve_for_auto=True
    )

    component = json.loads(component_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert Path(component["route"]["path"]).name == imported.asset_path.name
    assert component["route"]["sha256"] == _sha(imported.asset_path)
    assert selection["route"]["sha256"] == _sha(component_path)


def test_auto_approval_rejects_a_route_not_authored_by_the_editor(
    tmp_path: Path,
) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )
    profile_path = site_provider.import_folder(package).profile_path
    route = _write_valid_route(tmp_path / "external-route.json")
    payload = json.loads(route.read_text(encoding="utf-8"))
    payload["source"] = "operator_route_editor"
    route.write_text(json.dumps(payload), encoding="utf-8")
    imported = LocalRouteProvider(managed).import_file(route, profile_path)
    profile = load_site_profile(profile_path)
    assert profile.flight is not None
    assert profile.flight.approved is False
    assert profile.flight.route_clearance_approved is False
    assert imported.approved_for_auto is False
    assert not profile_for_selected_route(profile_path, imported.asset_path).flight.approved

    invalid_route = tmp_path / "invalid-route.json"
    invalid_route.write_text(
        json.dumps(
            {
                "schema": "sfm-flight-route/v1",
                "site_id": "test-yard",
                "coordinate_frame_id": "test-yard-reconstruction-v1",
                "frame": "glomap",
                "units": "map",
                "purpose": "flight",
                "closed": False,
                "waypoints": [[0, 0, 0], [0, 0, 0]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="zero length"):
        LocalRouteProvider(managed).import_file(invalid_route, profile_path)

def test_route_commit_rolls_back_asset_when_profile_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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
        LocalRouteProvider(managed).import_file(route, profile_path, approve_for_auto=True)

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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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

    profile = load_site_profile(profile_path)
    targets = {result.asset_path for result in results}
    assert len(targets) == 2
    assert all(target.read_bytes() == route.read_bytes() for target in targets)
    assert profile.route_json in targets
    assert profile.asset_sha256.route_json == _sha(profile.route_json)
    assert not [path for path in managed.rglob("*") if ".tmp" in path.name]


def test_adding_route_preserves_existing_route_files(
    tmp_path: Path,
) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    site_provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )
    profile_path = site_provider.import_folder(package).profile_path
    site_root = profile_path.parent
    draft = site_root / "route_drafts" / "route_old.json"
    authored = site_root / "routes" / "authored" / "route_old.json"
    draft.parent.mkdir(parents=True)
    authored.parent.mkdir(parents=True)
    draft.write_text("{}", encoding="utf-8")
    authored.write_text("{}", encoding="utf-8")
    route = _write_valid_route(tmp_path / "latest_route.json")

    imported = LocalRouteProvider(managed).import_file(route, profile_path)

    assert imported.asset_path.name.startswith("flight_route_")
    assert imported.asset_path.is_file()
    assert draft.exists()
    assert authored.exists()


def test_editing_route_replaces_only_the_selected_route(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    profile_path = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    ).import_folder(package).profile_path
    provider = LocalRouteProvider(managed)
    first = provider.import_file(_write_valid_route(tmp_path / "first.json"), profile_path)
    second_source = _write_valid_route(tmp_path / "second.json")
    second_payload = json.loads(second_source.read_text(encoding="utf-8"))
    second_payload["waypoints"] = [[0, 0, 0], [2, 0, 0]]
    second_source.write_text(json.dumps(second_payload), encoding="utf-8")
    second = provider.import_file(second_source, profile_path)
    first_before = first.asset_path.read_bytes()

    edited_source = _write_valid_route(tmp_path / "edited.json")
    edited_payload = json.loads(edited_source.read_text(encoding="utf-8"))
    edited_payload["waypoints"] = [[0, 0, 0], [3, 0, 0]]
    edited_payload["source"] = "operator_route_editor"
    edited_source.write_text(json.dumps(edited_payload), encoding="utf-8")
    edited = provider.import_file(
        edited_source,
        profile_path,
        approve_for_auto=True,
        replace_route=second.asset_path,
    )

    profile = load_site_profile(profile_path)
    assert edited.asset_path == second.asset_path
    assert first.asset_path.read_bytes() == first_before
    assert len(list((profile_path.parent / "routes").glob("*.json"))) == 2
    assert profile.route_json == second.asset_path
    assert json.loads(second.asset_path.read_text(encoding="utf-8"))["waypoints"] == [
        [0, 0, 0],
        [3, 0, 0],
    ]


def _route_deletion_site(tmp_path: Path) -> tuple[LocalRouteProvider, Path]:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "地圖檔" / "場域"
    profile_path = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    ).import_folder(package).profile_path
    return LocalRouteProvider(managed), profile_path


def _write_route_links(
    tmp_path: Path, route: Path, profile_path: Path, name: str = "selected"
) -> tuple[Path, Path, Path]:
    profile = load_site_profile(profile_path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    for key in ("localizer_profile", "map_reference_poses", "map_align", "localizer_deploy_dir"):
        value = getattr(profile, key)
        raw[key] = None if value is None else str(value)
    for key, value in raw["assets"].items():
        if value is not None:
            raw["assets"][key] = str(getattr(profile, key))
    raw["assets"]["route_json"] = str(route)
    raw["asset_sha256"]["route_json"] = _sha(route)
    system_profile = tmp_path / "控制介面程式" / "site_profiles" / f"{name}.json"
    system_profile.parent.mkdir(parents=True, exist_ok=True)
    system_profile.write_text(json.dumps(raw), encoding="utf-8")
    component = tmp_path / "控制介面程式" / "mission_components" / "routes" / f"{name}.json"
    component.parent.mkdir(parents=True, exist_ok=True)
    component.write_text(json.dumps({
        "schema": "sfm-route-package/v1",
        "route_id": name,
        "revision": "test-v1",
        "site_id": profile.site_id,
        "frame": {"kind": "map", "id": profile.coordinate_frame.id},
        "route": {"path": str(route), "sha256": _sha(route)},
    }), encoding="utf-8")
    selection = tmp_path / "控制介面程式" / "mission_selections" / f"{name}.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    # Selection loading validates artifact refs, not component schemas; the
    # unrelated slots deliberately reuse the real map artifact in this fixture.
    artifact = {"path": str(profile.map_ply), "sha256": _sha(profile.map_ply)}
    selection.write_text(json.dumps({
        "schema": "sfm-mission-selection/v1",
        "selection_id": name,
        "vehicle": artifact,
        "site": artifact,
        "map": artifact,
        "localizer": artifact,
        "route": {"path": str(component), "sha256": _sha(component)},
        "calibrations": [],
        "approval": artifact,
    }), encoding="utf-8")
    return system_profile, component, selection


def test_delete_old_display_route_preserves_current_route_and_approval(tmp_path: Path) -> None:
    provider, profile_path = _route_deletion_site(tmp_path)
    current = provider.import_file(
        _write_valid_route(tmp_path / "current.json"), profile_path, approve_for_auto=True,
    )
    old = profile_path.parent / "old_flight_path.json"
    old.write_text(json.dumps({
        "schema": "legacy-route/v1",
        "purpose": "display",
        "waypoints": [[0, 0, 0], [4, 0, 0]],
    }), encoding="utf-8")
    # Unrelated, incomplete JSON should not become owned by route deletion.
    (profile_path.parent / "unfinished.json").write_text("[", encoding="utf-8")
    profile_before = profile_path.read_bytes()
    current_before = current.asset_path.read_bytes()

    deleted = provider.delete_file(old, profile_path)

    assert deleted.asset_path == old
    assert deleted.profile_path == profile_path
    assert deleted.approved_for_auto is True
    assert not old.exists()
    assert profile_path.read_bytes() == profile_before
    assert current.asset_path.read_bytes() == current_before


def test_delete_current_route_clears_profile_without_selecting_remaining(tmp_path: Path) -> None:
    provider, profile_path = _route_deletion_site(tmp_path)
    first = provider.import_file(_write_valid_route(tmp_path / "first.json"), profile_path)
    current = provider.import_file(_write_valid_route(tmp_path / "current.json"), profile_path)
    first_before = first.asset_path.read_bytes()

    deleted = provider.delete_file(current.asset_path, profile_path)

    profile = load_site_profile(profile_path)
    assert not current.asset_path.exists()
    assert first.asset_path.read_bytes() == first_before
    assert deleted.approved_for_auto is False
    assert profile.route_json is None
    assert profile.asset_sha256.route_json is None
    assert profile.flight.approved is False
    assert profile.flight.route_clearance_approved is False


def test_delete_last_route_detaches_linked_manifests_and_keeps_site_editable(tmp_path: Path) -> None:
    from mission_manifest import load_mission_selection, load_route_manifest
    from operator_actions import SiteAssetActions

    provider, profile_path = _route_deletion_site(tmp_path)
    current = provider.import_file(_write_valid_route(tmp_path / "current.json"), profile_path)
    system_profile, component, selection = _write_route_links(
        tmp_path, current.asset_path, profile_path
    )
    load_route_manifest(component, workspace_root=tmp_path)
    load_mission_selection(selection, workspace_root=tmp_path)
    actions = SiteAssetActions(None, provider, None, current_profile=system_profile)

    result = actions.delete_route(current.asset_path)

    assert result.kind == "route_deleted"
    assert result.asset_path == current.asset_path
    assert result.profile_path == actions.current_profile == profile_path
    assert result.approved_for_auto is False
    for path in (profile_path, system_profile):
        profile = load_site_profile(path)
        assert profile.route_json is None
        assert profile.asset_sha256.route_json is None
        assert profile.flight.approved is False
        assert profile.flight.route_clearance_approved is False
    assert not component.exists()
    detached = load_mission_selection(selection, workspace_root=tmp_path)
    assert detached.route is None
    assert detached.approval is None
    assert not local_assets.describe_site_routes(profile_path.parent)

    # Editing can resume by an explicit new import, never by resurrecting links
    # or choosing some other route on the operator's behalf.
    new = provider.import_file(_write_valid_route(tmp_path / "new.json"), profile_path)
    assert load_site_profile(profile_path).route_json == new.asset_path
    assert not current.asset_path.exists()
    assert not component.exists()
    assert load_mission_selection(selection, workspace_root=tmp_path).route is None


def test_delete_linked_old_route_preserves_unrelated_links_and_active_profile(tmp_path: Path) -> None:
    provider, profile_path = _route_deletion_site(tmp_path)
    old = provider.import_file(_write_valid_route(tmp_path / "old.json"), profile_path)
    current = provider.import_file(_write_valid_route(tmp_path / "current.json"), profile_path)
    old_links = _write_route_links(tmp_path, old.asset_path, profile_path, "old")
    current_links = _write_route_links(tmp_path, current.asset_path, profile_path, "current")
    unchanged = {
        path: path.read_bytes()
        for path in (profile_path, current.asset_path, *current_links)
    }

    provider.delete_file(old.asset_path, profile_path)

    assert not old.asset_path.exists()
    assert not old_links[1].exists()
    assert load_site_profile(old_links[0]).route_json is None
    assert json.loads(old_links[2].read_text(encoding="utf-8"))["route"] is None
    for path, before in unchanged.items():
        assert path.read_bytes() == before


@pytest.mark.parametrize("unsafe", [
    "outside", "other-site", "file-symlink", "directory-symlink", "non-route",
    "draft", "profile", "directory", "missing", "shared-target",
])
def test_route_deletion_rejects_unsafe_or_non_route_paths(tmp_path: Path, unsafe: str) -> None:
    provider, profile_path = _route_deletion_site(tmp_path)
    current = provider.import_file(_write_valid_route(tmp_path / "current.json"), profile_path)
    routes = current.asset_path.parent
    if unsafe == "outside":
        source = _write_valid_route(tmp_path / "outside_route.json")
    elif unsafe == "other-site":
        other = provider.managed_root / "other"
        other.mkdir()
        source = _write_valid_route(other / "flight_route.json")
    elif unsafe == "file-symlink":
        source = routes / "linked_route.json"
        source.symlink_to(current.asset_path)
    elif unsafe == "directory-symlink":
        linked = profile_path.parent / "linked-routes"
        linked.symlink_to(routes, target_is_directory=True)
        source = linked / current.asset_path.name
    elif unsafe == "non-route":
        source = routes / "wires.json"
        source.write_text(json.dumps({"wires": []}), encoding="utf-8")
    elif unsafe == "draft":
        folder = profile_path.parent / "route_drafts"
        folder.mkdir()
        source = _write_valid_route(folder / "flight_route.json")
    elif unsafe == "profile":
        source = profile_path
    elif unsafe == "directory":
        source = routes
    elif unsafe == "shared-target":
        source = current.asset_path
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
        raw["assets"]["poles_json"] = str(source)
        profile_path.write_text(json.dumps(raw), encoding="utf-8")
    else:
        source = routes / "missing_route.json"
    profile_before = profile_path.read_bytes()
    route_before = current.asset_path.read_bytes()

    with pytest.raises(ValueError):
        provider.delete_file(source, profile_path)

    assert profile_path.read_bytes() == profile_before
    assert current.asset_path.read_bytes() == route_before


@pytest.mark.parametrize("failure", [
    "profile-replace", "selection-replace", "component-delete", "route-delete",
    "post-delete-validation",
])
def test_route_deletion_rolls_back_all_files_on_transaction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    provider, profile_path = _route_deletion_site(tmp_path)
    current = provider.import_file(_write_valid_route(tmp_path / "current.json"), profile_path)
    links = _write_route_links(tmp_path, current.asset_path, profile_path)
    snapshots = {
        path: path.read_bytes() for path in (profile_path, current.asset_path, *links)
    }
    real_replace = local_assets.os.replace
    real_unlink = Path.unlink
    real_load = local_assets.load_site_profile
    injected = False

    def fail_replace(source, destination):
        nonlocal injected
        fail_target = profile_path if failure == "profile-replace" else links[2]
        if failure.endswith("replace") and Path(destination) == fail_target and not injected:
            injected = True
            raise OSError("forced route deletion failure")
        real_replace(source, destination)

    def fail_unlink(path, *args, **kwargs):
        nonlocal injected
        fail_target = links[1] if failure == "component-delete" else current.asset_path
        if failure.endswith("delete") and path == fail_target and not injected:
            injected = True
            raise OSError("forced route deletion failure")
        return real_unlink(path, *args, **kwargs)

    def fail_load(path, *args, **kwargs):
        nonlocal injected
        if (
            failure == "post-delete-validation" and Path(path) == profile_path
            and not current.asset_path.exists() and not injected
        ):
            injected = True
            raise OSError("forced route deletion failure")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(local_assets.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(local_assets, "load_site_profile", fail_load)

    with pytest.raises(OSError, match="forced route deletion failure"):
        provider.delete_file(current.asset_path, profile_path)

    assert injected
    for path, before in snapshots.items():
        assert path.read_bytes() == before
    assert not [
        path for path in tmp_path.rglob("*") if path.suffix in {".tmp", ".bak"}
    ]


def test_delete_route_clears_explicit_runtime_snapshot_outside_site_profiles(tmp_path: Path) -> None:
    provider, profile_path = _route_deletion_site(tmp_path)
    current = provider.import_file(_write_valid_route(tmp_path / "current.json"), profile_path)
    system_profile, _component, _selection = _write_route_links(
        tmp_path, current.asset_path, profile_path
    )
    snapshot = tmp_path / "執行環境" / "mission_snapshots" / "active.site_profile.json"
    snapshot.parent.mkdir(parents=True)
    system_profile.rename(snapshot)

    deleted = provider.delete_file(current.asset_path, snapshot)

    assert deleted.profile_path == profile_path
    cleared = load_site_profile(snapshot)
    assert cleared.route_json is None
    assert cleared.asset_sha256.route_json is None
    assert cleared.flight.approved is False


def test_route_delete_retains_recovery_backups_if_rollback_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, profile_path = _route_deletion_site(tmp_path)
    current = provider.import_file(_write_valid_route(tmp_path / "current.json"), profile_path)
    profile_before = profile_path.read_bytes()
    real_replace = local_assets.os.replace
    real_unlink = Path.unlink

    def fail_replace(source, destination):
        if Path(destination) == profile_path and Path(source).suffix == ".bak":
            raise OSError("forced rollback failure")
        real_replace(source, destination)

    def fail_unlink(path, *args, **kwargs):
        if path == current.asset_path:
            raise OSError("forced delete failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(local_assets.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(RuntimeError, match="recovery backups retained"):
        provider.delete_file(current.asset_path, profile_path)

    assert current.asset_path.exists()
    assert load_site_profile(profile_path).route_json is None
    profile_backups = list(profile_path.parent.glob(".site_profile.json.*.bak"))
    assert len(profile_backups) == 1
    assert profile_backups[0].read_bytes() == profile_before


def test_concurrent_site_package_imports_have_one_winner(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    provider = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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
    system_profile_path = system_profiles / "test_yard_direct.json"
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
    assert imported.asset_path.parent == site_folder / "routes"
    assert imported.asset_path.name.startswith("flight_route_")
    updated = load_site_profile(managed_profile_path)
    assert updated.route_json == imported.asset_path
    assert updated.asset_sha256.route_json == _sha(imported.asset_path)

    mismatched = json.loads(system_profile_path.read_text(encoding="utf-8"))
    mismatched["site_id"] = "wrong-site"
    system_profile_path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong-site"):
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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
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
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )

    with pytest.raises(ValueError, match="map_align"):
        provider.validate_folder(package)


def test_site_package_rejects_bundle_sha_mismatch(tmp_path: Path) -> None:
    package = _write_site_package(tmp_path)
    profile_path = package / "site_profile.json"
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    raw["asset_sha256"]["localization_bundle"] = "0" * 64
    profile_path.write_text(json.dumps(raw), encoding="utf-8")
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        provider.validate_folder(package)


def test_site_package_validate_rejects_non_directory(tmp_path: Path) -> None:
    provider = LocalSitePackageProvider(
        tmp_path / "managed",
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )

    with pytest.raises(ValueError, match="not a folder"):
        provider.validate_folder(tmp_path / "missing")


def test_authored_route_action_propagates_committed_auto_approval(tmp_path: Path) -> None:
    from operator_actions import SiteAssetActions

    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    sites = LocalSitePackageProvider(
        managed,
        bundle_inspector=lambda _path, _sha256: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )
    profile = sites.import_folder(package).profile_path
    actions = SiteAssetActions(
        sites, LocalRouteProvider(managed), LocalTargetProvider(managed), profile,
    )
    result = actions.import_authored_route(_write_valid_route(tmp_path / "route.json"))
    assert result.approved_for_auto is True
    assert result.asset_path == load_site_profile(profile).route_json
    assert "已自動核准" in result.message
    assert "請完成三項驗證" in result.message


def test_route_import_action_does_not_invent_provider_approval(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from operator_actions import SiteAssetActions
    from site_asset_interfaces import ImportedAsset

    profile = tmp_path / "site_profile.json"
    route = tmp_path / "route.json"
    provider = SimpleNamespace(import_file=lambda *_: ImportedAsset(profile, route))
    actions = SiteAssetActions(None, provider, None, profile)
    result = actions.import_route(route)
    assert result.approved_for_auto is False
    assert "未核准自動飛行" in result.message


@pytest.mark.parametrize("live", [False, True], ids=["simulated", "real-flight"])
@pytest.mark.parametrize("edit", [False, True], ids=["new-route", "edit-route"])
def test_editor_save_automatically_approves_in_both_interfaces(tmp_path, monkeypatch, live, edit):
    import queue
    from types import SimpleNamespace
    from unittest.mock import Mock
    import flight_operator_app as app
    from operator_actions import SiteAssetActions
    from site_assets_panel import SiteAssetsPanel

    package = _write_site_package(tmp_path)
    managed = tmp_path / "managed"
    sites = LocalSitePackageProvider(
        managed, bundle_inspector=lambda *_: ("ref-a.jpg",),
        profile_loader=_profile_loader,
    )
    profile_path = sites.import_folder(package).profile_path
    actions = SiteAssetActions(
        sites, LocalRouteProvider(managed), LocalTargetProvider(managed), profile_path,
    )
    source = _write_valid_route(tmp_path / "drawn-route.json")
    previous = actions.import_authored_route(source).asset_path if edit else None
    results = []
    panel = SimpleNamespace(actions=actions, _finish_ok=results.append, set_status=Mock())
    panel.import_authored_route = lambda path, **kw: SiteAssetsPanel.import_authored_route(panel, path, **kw)
    operator = app.OperatorApp.__new__(app.OperatorApp)
    operator.backend = SimpleNamespace(
        is_live=live,
        mode=app.InterfaceMode.REAL_FLIGHT if live else app.InterfaceMode.SIMULATED_STREAM,
    )
    operator.site_assets_panel = panel
    operator._route_editor_load_queue = queue.Queue()
    operator._route_editor_load_queue.put((load_site_profile(profile_path), [], None, previous, None))
    operator.route_editor_safety_check = lambda: (True, "landed")
    editor_args = {}
    monkeypatch.setattr(app, "RouteEditorWindow", lambda *_args, **kw: editor_args.update(kw))

    operator._poll_route_editor_load()
    result = editor_args["import_route"](source)
    assert result.approved_for_auto is True
    assert results == [result]
    assert "已自動核准" in result.message
    profile = load_site_profile(profile_path)
    assert profile.flight.approved is True
    assert profile.flight.route_clearance_approved is True
    assert profile.asset_sha256.route_json == _sha(result.asset_path)
    if edit:
        assert result.asset_path == previous
    assert (editor_args["test_route"] is None) == live


def test_authored_routes_remain_approved_after_reopen_and_reselection(tmp_path):
    from operator_actions import SiteAssetActions
    from site_profile import flight_readiness_errors

    provider, profile_path = _route_deletion_site(tmp_path)
    actions = SiteAssetActions(None, provider, None, profile_path)
    first = actions.import_authored_route(_write_valid_route(tmp_path / "first.json"))
    second = actions.import_authored_route(_write_valid_route(tmp_path / "second.json"))
    saved_profile = profile_path.read_bytes()
    listed = {item.path for item in local_assets.describe_site_routes(profile_path.parent)}
    assert {first.asset_path, second.asset_path} <= listed
    assert load_site_profile(profile_path).route_json == second.asset_path
    for route in (first, second):
        selected = profile_for_selected_route(profile_path, route.asset_path)
        assert selected.flight.approved and selected.flight.route_clearance_approved
        assert selected.route_json == route.asset_path
        assert selected.asset_sha256.route_json == _sha(route.asset_path)
        assert "flight.approved is false" not in flight_readiness_errors(selected)
        assert "flight.route_clearance_approved is false" not in flight_readiness_errors(selected)
        # Route authoring does not invent the independent controller contract.
        assert "missing flight.controller" in flight_readiness_errors(selected)
    assert profile_path.read_bytes() == saved_profile


def test_legacy_current_editor_approval_migrates_when_another_route_is_saved(tmp_path):
    provider, profile_path = _route_deletion_site(tmp_path)
    first = provider.import_file(
        _write_valid_route(tmp_path / "first.json"), profile_path, approve_for_auto=True,
    )
    legacy = json.loads(profile_path.read_bytes())
    legacy["flight"].pop("authored_routes")
    profile_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert profile_for_selected_route(profile_path, first.asset_path).flight.approved
    second = provider.import_file(
        _write_valid_route(tmp_path / "second.json"), profile_path, approve_for_auto=True,
    )
    assert profile_for_selected_route(profile_path, first.asset_path).flight.approved
    assert profile_for_selected_route(profile_path, second.asset_path).flight.approved


def test_deleted_authored_route_cannot_reuse_its_old_approval(tmp_path):
    provider, profile_path = _route_deletion_site(tmp_path)
    first = provider.import_file(
        _write_valid_route(tmp_path / "first.json"), profile_path, approve_for_auto=True,
    )
    second = provider.import_file(
        _write_valid_route(tmp_path / "second.json"), profile_path, approve_for_auto=True,
    )
    original = first.asset_path.read_bytes()
    provider.delete_file(first.asset_path, profile_path)
    assert profile_for_selected_route(profile_path, second.asset_path).flight.approved
    first.asset_path.write_bytes(original)
    assert not profile_for_selected_route(profile_path, first.asset_path).flight.approved


def test_editing_authored_route_replaces_sha_and_preserves_other_approvals(tmp_path):
    provider, profile_path = _route_deletion_site(tmp_path)
    first = provider.import_file(
        _write_valid_route(tmp_path / "first.json"), profile_path, approve_for_auto=True,
    )
    second = provider.import_file(
        _write_valid_route(tmp_path / "second.json"), profile_path, approve_for_auto=True,
    )
    old_bytes, old_sha = first.asset_path.read_bytes(), _sha(first.asset_path)
    source = _write_valid_route(tmp_path / "edited.json")
    raw = json.loads(source.read_bytes())
    raw["waypoints"] = [[0, 0, 0], [2, 0, 0]]
    source.write_text(json.dumps(raw), encoding="utf-8")
    provider.import_file(source, profile_path, approve_for_auto=True, replace_route=first.asset_path)
    selected = profile_for_selected_route(profile_path, first.asset_path)
    assert selected.flight.approved
    assert selected.asset_sha256.route_json != old_sha
    assert profile_for_selected_route(profile_path, second.asset_path).flight.approved
    first.asset_path.write_bytes(old_bytes)
    with pytest.raises(ValueError, match="SHA-256"):
        profile_for_selected_route(profile_path, first.asset_path)


def test_external_edit_cannot_reuse_an_authored_approval(tmp_path):
    provider, profile_path = _route_deletion_site(tmp_path)
    authored = provider.import_file(
        _write_valid_route(tmp_path / "authored.json"), profile_path, approve_for_auto=True,
    )
    replacement = _write_valid_route(tmp_path / "external.json")
    imported = provider.import_file(replacement, profile_path, replace_route=authored.asset_path)
    assert not imported.approved_for_auto
    assert not profile_for_selected_route(profile_path, imported.asset_path).flight.approved


@pytest.mark.parametrize("change", [
    {"site_id": "different-map"},
    {"coordinate_frame_id": "different-frame"},
    {"map_ply_sha256": "b" * 64},
    {"purpose": "preview_only"},
    {"waypoints": [[0, 0, 0], [float("nan"), 0, 0]]},
    {"waypoint_actions": [[{"type": "hover", "seconds": 61}], []]},
])
def test_invalid_authored_save_never_changes_approval(tmp_path, change):
    provider, profile_path = _route_deletion_site(tmp_path)
    existing = provider.import_file(
        _write_valid_route(tmp_path / "existing.json"), profile_path, approve_for_auto=True,
    )
    before = profile_path.read_bytes()
    source = _write_valid_route(tmp_path / "invalid.json")
    raw = json.loads(source.read_bytes())
    raw.update(change)
    source.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        provider.import_file(source, profile_path, approve_for_auto=True, replace_route=existing.asset_path)
    assert profile_path.read_bytes() == before
    assert profile_for_selected_route(profile_path, existing.asset_path).flight.approved


def test_changed_map_rejects_both_saved_approval_and_new_save(tmp_path):
    provider, profile_path = _route_deletion_site(tmp_path)
    source = _write_valid_route(tmp_path / "source.json")
    existing = provider.import_file(source, profile_path, approve_for_auto=True)
    before = profile_path.read_bytes()
    profile = load_site_profile(profile_path)
    profile.map_ply.write_bytes(profile.map_ply.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="map SHA-256"):
        profile_for_selected_route(profile_path, existing.asset_path)
    with pytest.raises(ValueError, match="map SHA-256"):
        provider.import_file(source, profile_path, approve_for_auto=True)
    assert profile_path.read_bytes() == before


def test_save_durability_failure_rolls_back_route_and_approval(tmp_path, monkeypatch):
    provider, profile_path = _route_deletion_site(tmp_path)
    source = _write_valid_route(tmp_path / "source.json")
    existing = provider.import_file(source, profile_path, approve_for_auto=True)
    profile_before, route_before = profile_path.read_bytes(), existing.asset_path.read_bytes()
    raw = json.loads(source.read_bytes())
    raw["waypoints"] = [[0, 0, 0], [2, 0, 0]]
    source.write_text(json.dumps(raw), encoding="utf-8")
    real_sync = local_assets._sync_path

    def fail_directory_sync(path):
        if path.is_dir():
            raise OSError("forced durability failure")
        real_sync(path)

    monkeypatch.setattr(local_assets, "_sync_path", fail_directory_sync)
    with pytest.raises(OSError, match="forced durability failure"):
        provider.import_file(source, profile_path, approve_for_auto=True, replace_route=existing.asset_path)
    assert profile_path.read_bytes() == profile_before
    assert existing.asset_path.read_bytes() == route_before
    assert profile_for_selected_route(profile_path, existing.asset_path).flight.approved


def test_source_mutation_during_save_does_not_approve_different_bytes(tmp_path, monkeypatch):
    provider, profile_path = _route_deletion_site(tmp_path)
    source = _write_valid_route(tmp_path / "source.json")
    before = profile_path.read_bytes()
    real_copy = local_assets.shutil.copy2

    def changed_copy(origin, destination, *args, **kwargs):
        if Path(origin) == source:
            raw = json.loads(source.read_bytes())
            raw["waypoints"] = [[0, 0, 0], [4, 0, 0]]
            source.write_text(json.dumps(raw), encoding="utf-8")
        return real_copy(origin, destination, *args, **kwargs)

    monkeypatch.setattr(local_assets.shutil, "copy2", changed_copy)
    with pytest.raises(ValueError, match="changed while saving"):
        provider.import_file(source, profile_path, approve_for_auto=True)
    assert profile_path.read_bytes() == before


@pytest.mark.parametrize("already_managed", [False, True], ids=["adopt", "existing-pack"])
def test_deployed_profile_save_adopts_managed_site_without_source_mutation(
    tmp_path, already_managed,
):
    from operator_actions import SiteAssetActions

    package = _write_site_package(tmp_path)
    source_profile = package / "site_profile.json"
    raw = json.loads(source_profile.read_bytes())
    raw["flight"] = {
        "approved": False,
        "coordinate_frame_id": "test-yard-reconstruction-v1",
        "route_clearance_approved": False,
        "approval_note": "Ground-only deployed profile.",
        "controller": {
            "model": "scale_free_direction_speed_guard_v1",
            "speed_limit_mps": 0.30,
            "pose_max_age_ms": 500,
            "speed_max_age_ms": 500,
            "command_ttl_ms": 150,
            "yaw_tolerance_deg": 3.0,
            "horizontal_axes": [0, 2],
            "vertical_axis": 1,
            "camera_to_body_yaw_deg": 0.0,
            "body_right_sign": 1,
            "lookahead_map_units": 0.8,
            "rejoin_tolerance_map_units": 0.4,
            "arrival_tolerance_map_units": 0.2,
            "inspect_radius_map_units": 0.5,
            "inspect_resume_margin_map_units": 0.2,
            "max_pose_jump_map_units": 1.0,
            "progress_jump_slack_map_units": 0.5,
            "max_progress_regression_map_units": 0.1,
            "segment_window": 2,
            "progress_speed_factor": 2.0,
            "inspect_waypoints": [],
        },
    }
    source_profile.write_text(json.dumps(raw), encoding="utf-8")
    source_controller = load_site_profile(source_profile).flight.controller
    source_before = source_profile.read_bytes()
    managed = tmp_path / "地圖檔" / "場域"
    if already_managed:
        existing = LocalSitePackageProvider(
            managed, bundle_inspector=lambda *_: ("ref-a.jpg",),
            profile_loader=_profile_loader,
        ).import_folder(package).profile_path
        assert load_site_profile(existing).flight.controller is None
    actions = SiteAssetActions(None, LocalRouteProvider(managed), None, source_profile)
    first = actions.import_authored_route(_write_valid_route(tmp_path / "first.json"))
    second = actions.import_authored_route(_write_valid_route(tmp_path / "second.json"))
    assert first.profile_path == second.profile_path == actions.current_profile
    assert first.profile_path == managed / "test-yard" / "site_profile.json"
    assert source_profile.read_bytes() == source_before
    assert local_assets.site_pack_root_for_profile(source_profile, managed) == first.profile_path.parent
    for route in (first, second):
        # Reopening the deployed profile can select either managed-site route.
        selected = profile_for_selected_route(source_profile, route.asset_path)
        assert selected.flight.approved and selected.flight.route_clearance_approved
        assert selected.flight.controller == source_controller
        assert profile_for_selected_route(first.profile_path, route.asset_path).flight.controller == source_controller


def test_authored_action_does_not_claim_an_uncommitted_approval(tmp_path):
    from types import SimpleNamespace
    from operator_actions import SiteAssetActions
    from site_asset_interfaces import ImportedAsset

    profile, route = tmp_path / "site_profile.json", tmp_path / "route.json"
    provider = SimpleNamespace(import_file=lambda *_args, **_kw: ImportedAsset(profile, route))
    result = SiteAssetActions(None, provider, None, profile).import_authored_route(route)
    assert not result.approved_for_auto
    assert "已自動核准" not in result.message
    assert "不會解鎖飛行" in result.message
