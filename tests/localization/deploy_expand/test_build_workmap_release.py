from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from build_direct_site_release import frame_digest, sha256_file
from build_workmap_release import WorkmapBuildError, validate_pack, validate_source


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _source(tmp_path: Path) -> tuple[Path, Path]:
    release = tmp_path / "frozen_release"
    model = release / "model"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(f"source {name}".encode())
    local = release / "localization"
    local.mkdir()
    np.save(local / "boq.npy", np.ones((1, 3), dtype=np.float32))
    _json(local / "names.json", ["first.jpg"])
    _json(
        local / "reference_poses.json",
        {
            "schema": "reference-poses/v2",
            "source_model_sha256": frame_digest(model),
            "poses": {"first.jpg": {"R": np.eye(3).tolist(), "t": [0, 0, 0]}},
        },
    )
    (local / "reference_manifest.jsonl").write_text('{"image_name":"first.jpg"}\n')
    images = release / "keyframes/images"
    images.mkdir(parents=True)
    (release / "keyframes/keyframes.jsonl").write_text("\n")
    bank = local / "boq.npy"
    bundle = {
        "schema": "direct-localization-bundle/v1",
        "map_revision_id": "frozen_release",
        "coordinate_frame_id": f"frozen_release_{frame_digest(model)[:12]}",
        "model_dir": "../model",
        "keyframes_manifest": "../keyframes/keyframes.jsonl",
        "keyframes_images_root": "../keyframes/images",
        "reference_manifest": "reference_manifest.jsonl",
        "reference_bank": {"name": "sideview-boq", "descriptors": "boq.npy", "names": "names.json"},
        "intersection_cells": None,
        "reference_depth_dir": None,
        "model_sha256": {
            name: sha256_file(model / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "files": [
            {"path": "boq.npy", "size_bytes": bank.stat().st_size, "sha256": sha256_file(bank)}
        ],
    }
    _json(local / "direct_bundle.json", bundle)
    _json(
        release / "site_profile.json",
        {
            "site_id": "frozen_release",
            "coordinate_frame": {"id": bundle["coordinate_frame_id"]},
            "asset_sha256": {"localization_bundle": sha256_file(local / "direct_bundle.json")},
        },
    )
    session = tmp_path / "completed_session"
    session.mkdir()
    record = {
        "schema": "direct-online-workmap-session/v1",
        "session_id": "a-session",
        "base_release_id": "frozen_release",
        "coordinate_frame_id": bundle["coordinate_frame_id"],
        "base_bundle_sha256": sha256_file(local / "direct_bundle.json"),
        "source_epoch": 0,
        "finished_unix_s": 1.0,
    }
    _json(session / "session_meta.json", {**record, "writer_healthy": True})
    _json(
        session / "snapshot.json",
        {
            **record,
            "keyframes": {"measured": {"durable": True}},
            "landmarks": {"-1": {"xyz": [0, 0, 1]}},
            "statistics": {
                "writer_healthy": True,
                "writer_overflow_count": 0,
                "writer_error": None,
            },
        },
    )
    return release, session


def test_session_requires_exact_source_bundle_sha(tmp_path: Path) -> None:
    release, session = _source(tmp_path)
    assert validate_source(session, release)["snapshot"]["session_id"] == "a-session"
    metadata_path = session / "session_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["base_bundle_sha256"] = "0" * 64
    _json(metadata_path, metadata)
    with pytest.raises(WorkmapBuildError):
        validate_source(session, release)


def test_changed_base_release_is_rejected_before_pack_construction(tmp_path: Path) -> None:
    release, session = _source(tmp_path)
    bundle = release / "localization/direct_bundle.json"
    bundle.write_text(bundle.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(WorkmapBuildError):
        validate_source(session, release)


def test_pack_must_not_share_frozen_release_inodes(tmp_path: Path) -> None:
    release, _ = _source(tmp_path)
    staging = tmp_path / "staging"
    map_dir = staging / "map"
    for relative in (
        "keyframes/keyframes.jsonl",
        "localization/reference_manifest.jsonl",
        "localization/boq_references_sideview.npy",
        "localization/boq_references_sideview.names.json",
        "localization/fim_lwtl_intersection_cells.json",
        "experiments/edm_fixedpose_retriangulate/model/occupancy.json",
    ):
        path = map_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"independent pack asset")
    for name in ("README.md", "P174_AND_NEXT.md"):
        (staging / name).write_text("independent pack source", encoding="utf-8")
    validate_pack(staging, release)
    image = map_dir / "keyframes/images/first.jpg"
    image.parent.mkdir(parents=True)
    image.hardlink_to(release / "model/cameras.bin")
    with pytest.raises(WorkmapBuildError):
        validate_pack(staging, release)
