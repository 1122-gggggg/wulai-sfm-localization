from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import update_mission_route
from mission_manifest import ManifestError
from mission_resolver import resolve_mission

SITE = "site_a"
FRAME = "map_v1_frame"


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path, base: Path) -> dict[str, str]:
    return {"path": os.path.relpath(path, base), "sha256": _sha(path)}


def _route_asset(
    path: Path, *, frame: str = FRAME, waypoints: list[list[float]] | None = None
) -> Path:
    return _write_json(
        path,
        {
            "schema": "sfm-flight-route/v1",
            "site_id": SITE,
            "coordinate_frame_id": frame,
            "frame": "aligned",
            "align_source": "measured",
            "units": "map",
            "purpose": "flight",
            "closed": False,
            "waypoints": waypoints or [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
    )


def _build_workspace(root: Path) -> Path:
    """Smallest routed selection that resolves flight-ready (no pose-chain or IMU receipts)."""
    components = root / "components"
    assets = root / "assets"
    for name, content in (
        ("map.ply", b"ply"),
        ("poses.json", b"{}"),
        ("align.json", b"{}"),
        ("bundle.pt", b"bundle"),
        ("profile.json", b"{}"),
    ):
        (assets / name).parent.mkdir(parents=True, exist_ok=True)
        (assets / name).write_bytes(content)
    vehicle = _write_json(
        components / "vehicle.json",
        {
            "schema": "sfm-vehicle/v1",
            "vehicle_id": "aircraft_a",
            "revision": "r1",
            "adapter": "test_adapter",
            "model": "Test aircraft",
            "serials": [],
            "capabilities": ["rgb_720p"],
            "required_calibrations": [],
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
        components / "site.json",
        {
            "schema": "sfm-site/v1",
            "site_id": SITE,
            "display_name": "Site A",
            "site_frame_id": "site_a_enu",
            "units": "m",
        },
    )
    map_manifest = _write_json(
        components / "map.json",
        {
            "schema": "sfm-map-revision/v1",
            "site_id": SITE,
            "map_revision_id": "map_v1",
            "coordinate_frame": {
                "id": FRAME,
                "convention": "glomap",
                "horizontal_axes": ["x", "z"],
                "up_axis": "-y",
                "handedness": "right",
                "units": "map",
            },
            "assets": {
                "map_ply": _ref(assets / "map.ply", components),
                "reference_poses": _ref(assets / "poses.json", components),
                "map_align": _ref(assets / "align.json", components),
            },
        },
    )
    localizer = _write_json(
        components / "localizer.json",
        {
            "schema": "sfm-localizer-variant/v1",
            "algorithm_id": "test_localizer",
            "variant_id": "variant_v1",
            "provider_api_version": 1,
            "pose_contract_version": 1,
            "map_revision_id": "map_v1",
            "coordinate_frame_id": FRAME,
            "camera_profiles": ["camera_pipeline_v1"],
            "required_vehicle_capabilities": ["rgb_720p"],
            "quality_gate_id": "quality_gate_v1",
            "artifacts": {
                "bundle": _ref(assets / "bundle.pt", components),
                "profile": _ref(assets / "profile.json", components),
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
    route = _route_asset(root / "routes" / "flight_route_20260101_old00000.json")
    package = _write_json(
        components / "route.json",
        {
            "schema": "sfm-route-package/v1",
            "route_id": "route_a",
            "revision": "20260101_old00000",
            "site_id": SITE,
            "frame": {"kind": "map", "id": FRAME},
            "route": _ref(route, components),
        },
    )
    selections = root / "selections"
    return _write_json(
        selections / "mission.json",
        {
            "schema": "sfm-mission-selection/v1",
            "selection_id": "mission_a",
            "vehicle": _ref(vehicle, selections),
            "site": _ref(site, selections),
            "map": _ref(map_manifest, selections),
            "localizer": _ref(localizer, selections),
            "route": _ref(package, selections),
            "calibrations": [_ref(quality, selections)],
            "approval": None,
        },
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_update_rebinds_route_package_and_selection(tmp_path: Path) -> None:
    selection = _build_workspace(tmp_path)
    assert resolve_mission(selection, workspace_root=tmp_path).readiness.flight_ready
    new_route = _route_asset(
        tmp_path / "routes" / "flight_route_20261101_abcd1234.json",
        waypoints=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    )

    report = update_mission_route.update_route(selection, new_route, root=tmp_path)

    mission = resolve_mission(selection, workspace_root=tmp_path)
    package = tmp_path / "components" / "route.json"
    assert mission.readiness.flight_ready and report["flight_ready"] is True
    assert mission.route is not None
    assert mission.route.route.path == new_route
    assert mission.route.route.sha256 == _sha(new_route) == report["route_sha256"]
    assert mission.route.revision == "20261101_abcd1234"
    assert mission.selection.route is not None
    assert mission.selection.route.sha256 == _sha(package) == report["route_package_sha256"]
    assert report["dry_run"] is False
    # Only the two documented files change; the old route asset stays for history.
    assert (tmp_path / "routes" / "flight_route_20260101_old00000.json").is_file()
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".candidate")]


def test_update_keeps_other_selection_fields_and_key_order(tmp_path: Path) -> None:
    selection = _build_workspace(tmp_path)
    before = json.loads(selection.read_text(encoding="utf-8"))
    new_route = _route_asset(tmp_path / "routes" / "custom_name.json")

    update_mission_route.update_route(selection, new_route, root=tmp_path)

    after = json.loads(selection.read_text(encoding="utf-8"))
    assert list(after) == list(before)
    assert {k: v for k, v in after.items() if k != "route"} == {
        k: v for k, v in before.items() if k != "route"
    }
    assert after["route"]["path"] == before["route"]["path"]
    # A file name outside the route-editor convention does not rewrite the revision.
    package = json.loads((tmp_path / "components" / "route.json").read_text(encoding="utf-8"))
    assert package["revision"] == "20260101_old00000"


def test_dry_run_validates_without_writing(tmp_path: Path) -> None:
    selection = _build_workspace(tmp_path)
    new_route = _route_asset(
        tmp_path / "routes" / "flight_route_20261101_abcd1234.json",
        waypoints=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    before = _snapshot(tmp_path)

    report = update_mission_route.update_route(selection, new_route, root=tmp_path, dry_run=True)

    assert report["dry_run"] is True and report["flight_ready"] is True
    assert _snapshot(tmp_path) == before
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".candidate")]


@pytest.mark.parametrize(
    ("route_kwargs", "message"),
    [
        ({"frame": "another_map_frame"}, "coordinate_frame_id does not match"),
        # Loads, but flight-route geometry rejects zero-length legs: a new blocker.
        ({"waypoints": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}, "new route is not acceptable"),
    ],
)
def test_unacceptable_route_changes_nothing(
    tmp_path: Path, route_kwargs: dict[str, Any], message: str
) -> None:
    selection = _build_workspace(tmp_path)
    bad_route = _route_asset(tmp_path / "routes" / "flight_route_bad.json", **route_kwargs)
    before = _snapshot(tmp_path)
    leftovers = {p.name for p in tmp_path.rglob("*")}

    with pytest.raises(ManifestError, match=message):
        update_mission_route.update_route(selection, bad_route, root=tmp_path)

    assert _snapshot(tmp_path) == before
    assert {p.name for p in tmp_path.rglob("*")} == leftovers


def test_failed_post_write_validation_restores_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _build_workspace(tmp_path)
    new_route = _route_asset(tmp_path / "routes" / "flight_route_20261101_abcd1234.json")
    before = _snapshot(tmp_path) | {
        new_route.relative_to(tmp_path).as_posix(): new_route.read_bytes()
    }
    real_resolve = update_mission_route.resolve_mission

    def resolve(path: Path, **kwargs: Any):
        if Path(path) == selection and _snapshot(tmp_path) != before:
            raise ManifestError("simulated failure after the files were replaced")
        return real_resolve(path, **kwargs)

    monkeypatch.setattr(update_mission_route, "resolve_mission", resolve)

    with pytest.raises(ManifestError, match="simulated failure"):
        update_mission_route.update_route(selection, new_route, root=tmp_path)

    assert _snapshot(tmp_path) == before


def test_other_selections_bound_to_the_package_are_reported_stale(tmp_path: Path) -> None:
    selection = _build_workspace(tmp_path)
    sibling = selection.with_name("sibling.json")
    shutil.copyfile(selection, sibling)
    new_route = _route_asset(
        tmp_path / "routes" / "flight_route_20261101_abcd1234.json",
        waypoints=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    )

    report = update_mission_route.update_route(selection, new_route, root=tmp_path)

    assert report["stale_selections"] == [str(sibling)]


def test_selection_without_route_is_rejected(tmp_path: Path) -> None:
    selection = _build_workspace(tmp_path)
    document = json.loads(selection.read_text(encoding="utf-8"))
    document["route"] = None
    _write_json(selection, document)
    new_route = _route_asset(tmp_path / "routes" / "flight_route_20261101_abcd1234.json")

    with pytest.raises(ManifestError, match="no route package"):
        update_mission_route.update_route(selection, new_route, root=tmp_path)


def test_route_outside_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    selection = _build_workspace(workspace)
    outside = _route_asset(tmp_path / "elsewhere" / "flight_route_20261101_abcd1234.json")

    with pytest.raises(ManifestError, match="must stay inside the workspace"):
        update_mission_route.update_route(selection, outside, root=workspace)
