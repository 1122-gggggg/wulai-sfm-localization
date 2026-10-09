from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import build_direct_site_release as builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Rotation:
    def matrix(self) -> np.ndarray:
        return np.eye(3)


class _ImagePose:
    def __init__(self, center: tuple[float, float, float]) -> None:
        self.rotation = _Rotation()
        self.translation = -np.asarray(center, dtype=np.float64)


class _Image:
    def __init__(self, name: str, center: tuple[float, float, float]) -> None:
        self.name = name
        self._pose = _ImagePose(center)

    def cam_from_world(self) -> _ImagePose:
        return self._pose


class _Point3D:
    xyz = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    color = np.asarray((10, 20, 30), dtype=np.uint8)


class _Reconstruction:
    def __init__(self, _model_dir: str) -> None:
        self.images = {
            1: _Image("camera_0/frame.jpg", (0.0, 0.0, 0.0)),
            2: _Image("camera_1/frame.jpg", (1.0, 0.0, 0.0)),
        }
        self.points3D = {1: _Point3D()}

    def num_images(self) -> int:
        return len(self.images)

    def num_points3D(self) -> int:
        return len(self.points3D)


def _make_pack(root: Path) -> Path:
    pack = root / "pack"
    map_dir = pack / "map"
    model = map_dir / "experiments/edm_fixedpose_retriangulate/model"
    depth = map_dir / "experiments/edm_fixedpose_retriangulate/depth_moge3/camera_0"
    keyframes = map_dir / "keyframes"
    localization = map_dir / "localization"
    model.mkdir(parents=True)
    depth.mkdir(parents=True)
    (keyframes / "images/camera_0").mkdir(parents=True)
    localization.mkdir(parents=True)

    for name in builder.MODEL_BINS:
        (model / name).write_bytes(f"fixture {name}".encode())
    (model / "occupancy.json").write_text(
        json.dumps({"n_images": 2, "n_points3D": 1}), encoding="utf-8"
    )
    (depth / "frame.npz").write_bytes(b"depth fixture")
    (keyframes / "images/camera_0/frame.jpg").write_bytes(b"image fixture")
    (keyframes / "keyframes.jsonl").write_text(
        json.dumps({"image_uri": "images/camera_0/frame.jpg"}) + "\n", encoding="utf-8"
    )
    (localization / "reference_manifest.jsonl").write_text(
        json.dumps({"image_name": "camera_0/frame.jpg"}) + "\n", encoding="utf-8"
    )
    (localization / "boq_references_sideview.npy").write_bytes(b"descriptor fixture")
    (localization / "boq_references_sideview.names.json").write_text(
        json.dumps(["camera_0/frame.jpg"]), encoding="utf-8"
    )
    (localization / "fim_lwtl_intersection_cells.json").write_text("{}\n", encoding="utf-8")
    (pack / "README.md").write_text("fixture pack\n", encoding="utf-8")
    (pack / "P174_AND_NEXT.md").write_text("fixture provenance\n", encoding="utf-8")
    return pack


def _install_build_fixtures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    pack = _make_pack(tmp_path)
    release_root = tmp_path / "地圖檔/場域/demo/releases"
    release_root.mkdir(parents=True)
    monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=_Reconstruction))

    def derive_gravity(_poses_path: Path, output_path: Path, frame_name: str) -> dict:
        payload = {
            "schema": "map-gravity-alignment/v1",
            "coordinate_frame_id": frame_name,
            "gravity_up_map": [0.0, -1.0, 0.0],
            "derivation": "fixture world -y aligned with gravity",
        }
        builder.write_json(output_path, payload)
        return payload

    def prepare_geometry(bundle_path: Path, _package: str = "") -> str:
        archive = bundle_path.parent / "geometry.npz"
        archive.write_bytes(b"fixture prepared geometry")
        manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
        manifest["files"].append(
            {
                "path": archive.name,
                "sha256": _sha256(archive),
                "size_bytes": archive.stat().st_size,
            }
        )
        return builder.write_json(bundle_path, manifest)

    monkeypatch.setattr(builder, "derive_gravity", derive_gravity)
    monkeypatch.setattr(builder, "prepare_release_geometry", prepare_geometry)
    return pack


def _build_release(
    pack: Path,
    release_id: str,
    *,
    online_workmap: bool,
    optimizer=None,
    retain=False,
    expansion=False,
    graph_solver=None,
) -> Path:
    args = [
        "--pack",
        str(pack),
        "--site",
        "demo",
        "--release-id",
        release_id,
        "--map-dir",
        "map",
        "--mode",
        "copy",
        "--issued-at",
        "2026-09-26T00:00:00+08:00",
    ]
    if online_workmap:
        args.append("--online-workmap")
    if optimizer is not None:
        args.extend(["--workmap-optimizer", optimizer])
    if graph_solver is not None:
        args.extend(["--workmap-submap-graph-solver", graph_solver])
    if retain:
        args.append("--workmap-retain-tracks")
    if expansion:
        args.append("--workmap-expansion")
    assert builder.main(args) == 0
    return builder.REPO_ROOT / "地圖檔/場域/demo/releases" / release_id


def test_build_workmap_expansion_profile(tmp_path, monkeypatch):
    from direct_profile import load_direct_profile

    pack = _install_build_fixtures(monkeypatch, tmp_path)
    release = _build_release(pack, "expansion", online_workmap=True, expansion=True)
    flags = load_direct_profile(
        release / "localization/direct_localizer_profile.json"
    ).online_workmap.expansion
    assert flags.frozen_region_gate and flags.spatiotemporal_query and flags.depth_seeds


def test_expansion_requires_online_workmap(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _build_release(tmp_path, "expansion", online_workmap=False, expansion=True)
    assert exc.value.code == 2


def _assert_profile_bindings(release: Path, *, enabled: bool, optimizer=None, retain=False) -> dict:
    profile_path = release / "localization/direct_localizer_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if enabled:
        expected = {"enabled": True}
        if optimizer is not None:
            expected["optimizer"] = optimizer
        if retain:
            expected["retain_valid_tracks"] = True
        assert profile["online_workmap"] == expected
    else:
        assert "online_workmap" not in profile
        assert profile == {
            "schema": "direct-deployment-profile/v1",
            "name": "river_p174_frozen",
            "map_scale": 1.0,
            "intrinsics": builder.QUERY_INTRINSICS,
            "reloc": builder.FROZEN_RELOC,
            "fast_loop": builder.FROZEN_FAST_LOOP,
            "vo": builder.FROZEN_VO,
            "dead_reckon": builder.FROZEN_DEAD_RECKON,
            "max_jump_u": 1.5,
        }

    profile_sha = _sha256(profile_path)
    site_profile = json.loads((release / "site_profile.json").read_text(encoding="utf-8"))
    localizer = json.loads(
        (release / "compat/localizer_direct_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        (release / "provenance/source_manifest.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (release / "compat/localizer_quality_receipt.json").read_text(encoding="utf-8")
    )
    bundle_path = release / "localization/direct_bundle.json"
    bundle_sha = _sha256(bundle_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert site_profile["asset_sha256"]["localizer_profile"] == profile_sha
    assert site_profile["asset_sha256"]["localization_bundle"] == bundle_sha
    assert localizer["artifacts"]["profile"]["sha256"] == profile_sha
    assert localizer["artifacts"]["bundle"]["sha256"] == bundle_sha
    assert source_manifest["generated"]["localization/direct_localizer_profile.json"] == profile_sha
    assert source_manifest["generated"]["localization/direct_bundle.json"] == bundle_sha
    assert receipt["details"]["profile_sha256"] == profile_sha
    assert receipt["details"]["bundle_sha256"] == bundle_sha
    assert receipt["passed"] is False
    assert site_profile["flight"]["approved"] is False
    assert site_profile["flight"]["route_clearance_approved"] is False
    for entry in bundle["files"]:
        asset = (bundle_path.parent / entry["path"]).resolve()
        assert _sha256(asset) == entry["sha256"]
        assert asset.stat().st_size == entry["size_bytes"]
    return profile


def test_online_workmap_flag_changes_only_profile_and_rebinds_release_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _install_build_fixtures(monkeypatch, tmp_path)
    source_model_file = pack / "map/experiments/edm_fixedpose_retriangulate/model/cameras.bin"
    original_source_hash = _sha256(source_model_file)

    default_release = _build_release(pack, "default_release", online_workmap=False)
    default_profile = _assert_profile_bindings(default_release, enabled=False)
    enabled_release = _build_release(pack, "enabled_release", online_workmap=True)
    enabled_profile = _assert_profile_bindings(enabled_release, enabled=True)

    assert default_profile.keys() < enabled_profile.keys()
    assert _sha256(default_release / "localization/direct_localizer_profile.json") != _sha256(
        enabled_release / "localization/direct_localizer_profile.json"
    )
    assert _sha256(source_model_file) == original_source_hash
    assert not (default_release / "model/cameras.bin").samefile(source_model_file)
    assert not (enabled_release / "model/cameras.bin").samefile(source_model_file)
    for release, package in (
        (default_release, "deploy_code"),
        (enabled_release, "deploy_code_expand"),
    ):
        deploy_dir = json.loads((release / "site_profile.json").read_text(encoding="utf-8"))[
            "localizer_deploy_dir"
        ]
        assert deploy_dir == f"../../../../../定位演算法/{package}/sfm_direct_deploy"


def test_incremental_graph_release_binds_optimizer_without_flight_approval(tmp_path, monkeypatch):
    pack = _install_build_fixtures(monkeypatch, tmp_path)
    release = _build_release(
        pack, "graph_release", online_workmap=True, optimizer="isam2", retain=True
    )
    _assert_profile_bindings(release, enabled=True, optimizer="isam2", retain=True)


def test_optimizer_requires_explicit_workmap_opt_in():
    with pytest.raises(SystemExit) as result:
        builder.main(
            [
                "--pack",
                "unused",
                "--site",
                "demo",
                "--release-id",
                "unused",
                "--workmap-optimizer",
                "isam2",
            ]
        )
    assert result.value.code == 2


def test_submap_graph_solver_flag_needs_online_workmap_and_is_written(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as result:
        builder.main(
            ["--pack", "unused", "--site", "demo", "--release-id", "unused",
             "--workmap-submap-graph-solver", "isam2"]
        )
    assert result.value.code == 2

    from direct_profile import load_direct_profile

    pack = _install_build_fixtures(monkeypatch, tmp_path)
    release = _build_release(pack, "graph", online_workmap=True, graph_solver="isam2")
    profile = json.loads(
        (release / "localization/direct_localizer_profile.json").read_text(encoding="utf-8")
    )
    assert profile["online_workmap"] == {"enabled": True, "submap_graph_solver": "isam2"}
    loaded = load_direct_profile(release / "localization/direct_localizer_profile.json")
    assert loaded.online_workmap.submap_graph_solver == "isam2"
