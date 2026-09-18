"""Geometry archives are equivalent, relocatable and digest-bound, without GPU."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from direct_geometry import GEOMETRY_ARCHIVE, load_geometry_archive, prepare_geometry_archive
from direct_map import DirectMapAssets
from live_provider import LiveMapEDMProvider
from sfm_diagnosis.site_pipeline.deployment_localizer import FinalMapEDMProvider


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def assets(tmp_path, monkeypatch):
    root = tmp_path / "release"
    model = root / "model"
    model.mkdir(parents=True)
    local = root / "localization"
    local.mkdir()
    images = root / "keyframes/images/session"
    images.mkdir(parents=True)
    names = ["session/a.jpg", "session/b.jpg", "session/empty.jpg"]
    records = []
    for name in names:
        path = images / Path(name).name
        path.write_bytes(b"image")
        records.append({"output_name": name, "video_id": "session", "image_sha256": digest(path)})
    keyframes = root / "keyframes/keyframes.jsonl"
    keyframes.write_text("".join(json.dumps(row) + "\n" for row in records))
    files = [keyframes]
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = model / name
        path.write_bytes(name.encode())
        files.append(path)
    for name, content in (("names.json", json.dumps(names[:2])), ("references.jsonl", "{}\n")):
        path = local / name
        path.write_text(content)
        files.append(path)
    bank = local / "bank.npy"
    np.save(bank, np.eye(2, dtype=np.float32))
    files.append(bank)
    raw = {
        "schema": "direct-localization-bundle/v1",
        "map_revision_id": "fixture",
        "coordinate_frame_id": "fixture-frame",
        "model_dir": "../model",
        "keyframes_manifest": "../keyframes/keyframes.jsonl",
        "keyframes_images_root": "../keyframes/images",
        "reference_manifest": "references.jsonl",
        "reference_bank": {"name": "fixture", "names": "names.json", "descriptors": "bank.npy"},
        "intersection_cells": None,
        "reference_depth_dir": None,
        "model_sha256": {p.name: digest(p) for p in model.iterdir()},
        "files": [
            {
                "path": str(Path("..") / p.relative_to(root)),
                "sha256": digest(p),
                "size_bytes": p.stat().st_size,
            }
            for p in files
        ],
    }
    bundle = local / "direct_bundle.json"
    bundle.write_text(json.dumps(raw))
    rows = {}
    for i, name in enumerate(reversed(names)):
        observations = (
            []
            if name.endswith("empty.jpg")
            else [
                SimpleNamespace(xy=np.array([10.0, 20.0]), point3D_id=9, has_point3D=lambda: True),
                SimpleNamespace(xy=np.array([60.0, 70.0]), point3D_id=2, has_point3D=lambda: True),
            ]
        )
        matrix = np.column_stack((np.eye(3), [i * 0.1, 0.0, 0.0]))
        rows[i] = SimpleNamespace(
            name=name,
            camera_id=1,
            points2D=observations,
            cam_from_world=lambda matrix=matrix: SimpleNamespace(matrix=lambda: matrix),
        )
    reconstruction = SimpleNamespace(
        images=rows,
        cameras={1: SimpleNamespace(width=100, height=80, params=[50, 51, 50, 40])},
        points3D={
            9: SimpleNamespace(xyz=np.array([1.0, 2.0, 3.0])),
            2: SimpleNamespace(xyz=np.array([4.0, 5.0, 6.0])),
        },
    )
    monkeypatch.setitem(
        sys.modules, "pycolmap", SimpleNamespace(Reconstruction=lambda _: reconstruction)
    )
    return DirectMapAssets.load(bundle)


def provider(assets, minimum=1):
    result = object.__new__(LiveMapEDMProvider)
    result.assets = assets
    result.map_model = assets.model_dir
    result._keyframes = assets.keyframe_index()
    result._geometry_locked = False
    result._reference_names = ()
    result._requested_min_reference_occupied_bins = minimum
    return result


@pytest.mark.parametrize("minimum", [1, 2])
def test_precomputed_geometry_matches_authoritative_loader(assets, minimum):
    archive = assets.root / GEOMETRY_ARCHIVE
    prepare_geometry_archive(assets, archive)
    expected, actual = provider(assets, minimum), provider(assets, minimum)
    FinalMapEDMProvider._load_geometry(expected)
    load_geometry_archive(actual, archive, assets.raw["model_sha256"])
    for field in (
        "_reference_names",
        "_reference_sessions",
        "_reference_paths",
        "_reference_occupied_bins",
        "min_reference_occupied_bins",
    ):
        assert getattr(actual, field) == getattr(expected, field)
    for field in ("_point_ids", "_point_xyz"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    for field in ("_cam_from_world", "_native_wh", "_camera_params", "_median_sparse_z"):
        for name, value in getattr(expected, field).items():
            np.testing.assert_array_equal(getattr(actual, field)[name], value)
    for name, observation in expected._observations.items():
        for got, want in zip(actual._observations[name], observation):
            np.testing.assert_array_equal(got, want)


def test_archive_requires_bundle_declaration_and_matching_model(assets, monkeypatch):
    archive = assets.root / GEOMETRY_ARCHIVE
    archive.write_bytes(b"untrusted undeclared archive")
    provider(assets)._load_geometry()  # Ignores undeclared files.
    prepare_geometry_archive(assets, archive)
    raw = dict(assets.raw)
    raw["files"] = [
        *raw["files"],
        {"path": archive.name, "sha256": digest(archive), "size_bytes": archive.stat().st_size},
    ]
    assets.bundle_path.write_text(json.dumps(raw))
    verified = DirectMapAssets.load(assets.bundle_path)

    def no_parse(*_args):
        raise AssertionError("verified archive must avoid COLMAP parsing")

    monkeypatch.setattr(sys.modules["pycolmap"], "Reconstruction", no_parse)
    restored = provider(verified)
    restored._load_geometry()
    assert restored._reference_names == ("session/a.jpg", "session/b.jpg")
    with pytest.raises(ValueError, match="model/schema"):
        load_geometry_archive(provider(verified), archive, {"cameras.bin": "wrong"})
    archive.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        DirectMapAssets.load(assets.bundle_path)


def test_geometry_archive_uses_current_release_paths(assets, tmp_path):
    archive = assets.root / GEOMETRY_ARCHIVE
    prepare_geometry_archive(assets, archive)
    destination = tmp_path / "new-location"
    assets.root.parent.rename(destination)
    moved = DirectMapAssets.load(destination / "localization/direct_bundle.json")
    restored = provider(moved)
    load_geometry_archive(restored, moved.root / GEOMETRY_ARCHIVE, moved.raw["model_sha256"])
    assert all(path.is_relative_to(destination) for path in restored._reference_paths)


def test_prepared_keyframe_index_avoids_parsing_manifest_twice(assets, monkeypatch):
    from sfm_diagnosis.site_pipeline import deployment_localizer as vendor

    def forbidden(*_args):
        raise AssertionError("the verified keyframe index must be reused")

    monkeypatch.setattr(vendor, "_keyframe_index", forbidden)
    monkeypatch.setattr(vendor, "_query_index", lambda _: {})
    monkeypatch.setattr(vendor, "_resolve_audited_site_packages", lambda _: None)
    monkeypatch.setattr(vendor, "_configure_audited_runtime_site_packages", lambda _: None)
    monkeypatch.setattr(FinalMapEDMProvider, "_compute_fingerprint", lambda _: "fixture")
    index = assets.keyframe_index()
    result = FinalMapEDMProvider(
        map_model=str(assets.model_dir),
        keyframes=str(assets.keyframes_manifest),
        query_manifest=str(assets.keyframes_manifest),
        cache_dir=str(assets.root / "scratch"),
        edm_config={},
        megaloc_source=str(assets.keyframes_manifest),
        megaloc_checkpoint=str(assets.keyframes_manifest),
        intrinsics_calibration={},
        keyframe_index=index,
    )
    assert result._keyframes is index


def test_release_builder_binds_geometry_and_reuses_existing_archive(assets, monkeypatch):
    import build_direct_site_release as builder
    import direct_geometry

    first = builder.prepare_release_geometry(assets.bundle_path)
    verified = DirectMapAssets.load(assets.bundle_path, expected_sha256=first)
    restored = provider(verified)
    restored._load_geometry()
    assert restored._reference_names == ("session/a.jpg", "session/b.jpg")

    def forbidden(*_args):
        raise AssertionError("an already verified archive must not be regenerated")

    monkeypatch.setattr(direct_geometry, "prepare_geometry_archive", forbidden)
    assert builder.prepare_release_geometry(assets.bundle_path) == first
