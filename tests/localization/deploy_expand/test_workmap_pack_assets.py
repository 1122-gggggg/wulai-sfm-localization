from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.workmap_pack_assets import WorkmapPackError, _session_image_path, assemble_workmap_pack


_MODEL_BINS = ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")
_FRAME_BINS = ("cameras.bin", "images.bin", "points3D.bin")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    base = tmp_path / "base-release"
    base_model = base / "model"
    base_localization = base / "localization"
    base_model.mkdir(parents=True)
    for name in _FRAME_BINS:
        (base_model / name).write_bytes(f"base-{name}".encode())
    reference_manifest = base_localization / "reference_manifest.jsonl"
    reference_manifest.parent.mkdir(parents=True)
    reference_manifest.write_text('{"image_name":"base/frame.jpg"}\n', encoding="utf-8")
    bundle = {
        "schema": "direct-localization-bundle/v1",
        "map_revision_id": "base-test",
        "coordinate_frame_id": "base-frame-test",
        "model_sha256": {name: _sha256(base_model / name) for name in _FRAME_BINS},
        "files": [
            {
                "path": "reference_manifest.jsonl",
                "sha256": _sha256(reference_manifest),
                "size_bytes": reference_manifest.stat().st_size,
            }
        ],
    }
    bundle_path = base_localization / "direct_bundle.json"
    _write_json(bundle_path, bundle)
    bundle_sha = _sha256(bundle_path)

    session = tmp_path / "session"
    session.mkdir()
    keyframe_name = "workmap_test_00000001_aaaaaaaa"
    image_size = [64, 48]
    K = [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
    image_path = session / "keyframes/new.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"durable test session image")
    image_sha256 = _sha256(image_path)
    context = {
        "schema": "direct-online-workmap-session/v1",
        "session_id": "session-test",
        "base_release_id": "base-test",
        "coordinate_frame_id": "base-frame-test",
        "base_bundle_sha256": bundle_sha,
    }
    _write_json(
        session / "session_meta.json",
        {**context, "writer_healthy": True, "finished_unix_s": 10.0},
    )
    _write_json(
        session / "snapshot.json",
        {
            **context,
            "finished_unix_s": 10.0,
            "keyframes": {keyframe_name: {"durable": True, "image_size": image_size, "K": K}},
        },
    )
    eye_pose = np.column_stack((np.eye(3), np.zeros(3))).tolist()
    (session / "keyframes.jsonl").write_text(
        json.dumps(
            {
                "name": keyframe_name,
                "durable_complete": True,
                "image_size": image_size,
                "K": K,
                "all_ids": [42],
                "all_xy": [[20.0, 15.0]],
                "image_path": "keyframes/new.png",
                "ordinal": 1,
                "stamp": 1.0,
                "source_epoch": 0,
                "cam_from_world": eye_pose,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session / "observations.jsonl").write_text(
        json.dumps(
            {
                "keyframe_name": keyframe_name,
                "landmark_id": 42,
                "xy": [20.0, 15.0],
                "observation_source": "frozen_map_keyframe_image_measurement",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "staging"
    model = output / "map/experiments/edm_fixedpose_retriangulate/model"
    model.mkdir(parents=True)
    for name in _MODEL_BINS:
        (model / name).write_bytes(f"optimized-{name}".encode())

    receipt: dict[str, object] = {
        "base_bundle_sha256": bundle_sha,
        "session_id": "session-test",
        "gauge": {
            "pass": True,
            "fit": {"landmark_ids": list(range(1, 7))},
            "holdout": {"landmark_ids": list(range(7, 13))},
            "scale": 1.0,
        },
        "edm_validation": {
            "new_old_pairs": 8,
            "new_new_pairs": 8,
            "provenance": {
                "implementation": "test fixture",
                "base_bundle_sha256": bundle_sha,
                "base_model_sha256": bundle["model_sha256"],
                "base_reference_manifest_sha256": _sha256(reference_manifest),
                "edm_checkpoint_sha256": "1" * 64,
                "boq_checkpoint_sha256": "2" * 64,
                "profile_sha256": "3" * 64,
                "session_id": "session-test",
                "session_input_sha256": {
                    "session_meta_sha256": _sha256(session / "session_meta.json"),
                    "keyframes_jsonl_sha256": _sha256(session / "keyframes.jsonl"),
                    "observations_jsonl_sha256": _sha256(session / "observations.jsonl"),
                    "keyframe_images_sha256": {keyframe_name: image_sha256},
                },
                "matcher": {"reciprocal_tolerance_px": 2.0},
                "source_sha256": {
                    "reconstruction_tool": "4" * 64,
                    "live_provider": "5" * 64,
                    "workmap_overlap": "6" * 64,
                    "official_edm_adapter": "7" * 64,
                },
                "runtime_versions": {"pycolmap": "4.0.4", "opencv": "4.11", "torch": "2.7"},
            },
        },
        "negative_id_map": {"-1": 10001},
        "keyframe_image_names": {keyframe_name: "workmap/session-test_frame.png"},
        "filtering": {"before": {"points": 10}, "after": {"points": 9}},
        "bundle_adjustment": {"fixed_intrinsics": True, "iterations": 1},
    }
    return base, session, model, output, receipt


def test_missing_optimized_fim_lwtl_scores_fail_closed_without_half_pack(tmp_path: Path) -> None:
    base, session, model, output, receipt = _inputs(tmp_path)
    base_bundle_before = _sha256(base / "localization/direct_bundle.json")
    model_hashes_before = {name: _sha256(model / name) for name in _MODEL_BINS}

    with pytest.raises(WorkmapPackError):
        assemble_workmap_pack(base, session, model, output, receipt)

    assert _sha256(base / "localization/direct_bundle.json") == base_bundle_before
    assert {name: _sha256(model / name) for name in _MODEL_BINS} == model_hashes_before
    assert not (model / "occupancy.json").exists()
    assert not (output / "map/keyframes").exists()
    assert not (output / "map/localization").exists()
    assert not (output / "README.md").exists()
    assert not (output / "P174_AND_NEXT.md").exists()
    assert not (output / "workmap_pack_receipt.json").exists()


def test_session_must_bind_exact_base_bundle_sha_before_any_asset_copy(tmp_path: Path) -> None:
    base, session, model, output, receipt = _inputs(tmp_path)
    snapshot_path = session / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["base_bundle_sha256"] = "0" * 64
    _write_json(snapshot_path, snapshot)
    bundle_before = _sha256(base / "localization/direct_bundle.json")
    model_hashes_before = {name: _sha256(model / name) for name in _MODEL_BINS}

    with pytest.raises(WorkmapPackError):
        assemble_workmap_pack(base, session, model, output, receipt)

    assert _sha256(base / "localization/direct_bundle.json") == bundle_before
    assert {name: _sha256(model / name) for name in _MODEL_BINS} == model_hashes_before
    assert not (model / "occupancy.json").exists()
    assert not (output / "map/keyframes").exists()
    assert not (output / "map/localization").exists()


def test_archived_session_image_uses_worktree_alias_but_not_session_symlinks(
    tmp_path: Path,
) -> None:
    session = tmp_path / "relocated" / "session"
    images = session / "keyframes"
    images.mkdir(parents=True)
    image = images / "new.png"
    image.write_bytes(b"original image")
    alias = tmp_path / "old-worktree"
    alias.symlink_to(tmp_path / "relocated", target_is_directory=True)
    historical_path = alias / "session/keyframes/new.png"

    assert _session_image_path(session, str(historical_path), name="new") == image

    (images / "linked.png").symlink_to(image)
    with pytest.raises(WorkmapPackError, match="symlink"):
        _session_image_path(session, str(images / "linked.png"), name="new")
    other = tmp_path / "unrelated"
    other.write_bytes(b"unexpected")
    with pytest.raises(WorkmapPackError, match="escapes"):
        _session_image_path(session, str(other), name="new")


_PACK_K = [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
_PACK_SIZE = (64, 48)


def _pack_pixel(center_x: float, xyz: tuple[float, float, float]) -> list[float]:
    x, y, z = xyz
    return [32.0 + 50.0 * (x - center_x) / z, 24.0 + 50.0 * y / z]


def _write_image(path: Path, seed: int) -> str:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pattern = (np.arange(_PACK_SIZE[0] * _PACK_SIZE[1] * 3, dtype=np.uint32) * (seed + 3)) % 251
    assert cv2.imwrite(str(path), pattern.reshape(_PACK_SIZE[1], _PACK_SIZE[0], 3).astype(np.uint8))
    return _sha256(path)


def _pack_model(images: list[dict[str, object]], points: dict[int, tuple[float, float, float]]):
    import numpy as np
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    for camera_id in sorted({int(image["camera_id"]) for image in images}):
        camera = pycolmap.Camera.create_from_model_name(camera_id, "PINHOLE", 50.0, *_PACK_SIZE)
        camera.params = np.array([50.0, 50.0, 32.0, 24.0])
        reconstruction.add_camera_with_trivial_rig(camera)
    tracks: dict[int, list[object]] = {point_id: [] for point_id in points}
    for image in images:
        keypoints = [xy for _point_id, xy in image["observations"]]
        colmap_image = pycolmap.Image(
            name=str(image["name"]),
            keypoints=np.asarray(keypoints, dtype=np.float64),
            camera_id=int(image["camera_id"]),
            image_id=int(image["image_id"]),
        )
        pose = np.column_stack((np.eye(3), [-float(image["center_x"]), 0.0, 0.0]))
        reconstruction.add_image_with_trivial_frame(colmap_image, pycolmap.Rigid3d(pose))
        for index, (point_id, _xy) in enumerate(image["observations"]):
            tracks[point_id].append(pycolmap.TrackElement(int(image["image_id"]), index))
    for point_id, xyz in points.items():
        reconstruction.add_point3D_with_id(
            point_id,
            pycolmap.Point3D(
                xyz=np.asarray(xyz, dtype=np.float64),
                track=pycolmap.Track(tracks[point_id]),
                color=np.zeros(3, dtype=np.uint8),
            ),
        )
    return reconstruction


def _complete_pack_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    import numpy as np

    from tools.workmap_pack_assets import _bytes_sha256, _json_bytes

    base_points = {
        point_id: (-1.0 + 0.25 * point_id, -0.6 + 0.15 * point_id, 5.0) for point_id in range(1, 9)
    }
    new_points = {101: (0.3, 0.5, 5.0), 102: (-0.3, -0.5, 5.0)}
    base_names = ("vidA/frame_0.jpg", "vidA/frame_1.jpg")
    base_images = []
    for index, name in enumerate(base_names):
        center_x = 0.2 * index
        observations = [
            (point_id, _pack_pixel(center_x, xyz)) for point_id, xyz in base_points.items()
        ]
        if index == 0:
            # EDM dense matching leaves nearby duplicate keypoints on one track.
            duplicate = list(observations[0][1])
            observations.append((1, [duplicate[0] + 1.2, duplicate[1]]))
        base_images.append(
            {
                "name": name,
                "camera_id": 1,
                "image_id": index + 1,
                "center_x": center_x,
                "observations": observations,
            }
        )

    base = tmp_path / "base-release"
    model_dir = base / "model"
    model_dir.mkdir(parents=True)
    _pack_model(base_images, base_points).write(str(model_dir))
    localization = base / "localization"
    localization.mkdir()
    keyframe_rows, shard_files, depth_files = [], [], []
    for index, name in enumerate(base_names):
        image_path = base / "keyframes/images" / name
        digest = _write_image(image_path, index)
        keyframe_rows.append(
            {"output_name": name, "image_sha256": digest, "image_size": list(_PACK_SIZE)}
        )
        shard_files.append(
            {"path": name, "size_bytes": image_path.stat().st_size, "sha256": digest}
        )
        depth_path = base / "depth_moge3" / f"{name}.npz"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        with depth_path.open("wb") as stream:
            np.savez(
                stream,
                depth=np.ones((_PACK_SIZE[1], _PACK_SIZE[0]), dtype=np.float16),
                source=np.array("moge3-g"),
            )
        depth_files.append(
            {
                "path": f"{name}.npz",
                "size_bytes": depth_path.stat().st_size,
                "sha256": _sha256(depth_path),
            }
        )
    (base / "keyframes/keyframes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in keyframe_rows), encoding="utf-8"
    )
    reference_manifest = localization / "reference_manifest.jsonl"
    reference_manifest.write_text(
        "".join(json.dumps({"image_name": name}) + "\n" for name in base_names), encoding="utf-8"
    )
    _write_json(localization / "boq_references_sideview.names.json", sorted(base_names))
    bank = np.zeros((2, 16384), dtype=np.float32)
    bank[0, 0] = bank[1, 1] = 1.0
    np.save(localization / "boq_references_sideview.npy", bank)
    _write_json(
        localization / "shard_manifest.json",
        {"shards": [{"files": shard_files}], "depth_shards": [{"files": depth_files}]},
    )
    bundle = {
        "schema": "direct-localization-bundle/v1",
        "map_revision_id": "base-test",
        "coordinate_frame_id": "base-frame-test",
        "model_sha256": {name: _sha256(model_dir / name) for name in _FRAME_BINS},
        "files": [
            {
                "path": "reference_manifest.jsonl",
                "sha256": _sha256(reference_manifest),
                "size_bytes": reference_manifest.stat().st_size,
            }
        ],
    }
    bundle_path = localization / "direct_bundle.json"
    _write_json(bundle_path, bundle)
    bundle_sha = _sha256(bundle_path)

    session = tmp_path / "session"
    context = {
        "schema": "direct-online-workmap-session/v1",
        "session_id": "session-test",
        "base_release_id": "base-test",
        "coordinate_frame_id": "base-frame-test",
        "base_bundle_sha256": bundle_sha,
    }
    session_rows, observation_rows, session_images, new_images = [], [], {}, []
    snapshot_keyframes = {}
    negative_map = {-1: 101, -2: 102}
    for index, name in enumerate(("kf_a", "kf_b")):
        center_x = 0.4 + 0.2 * index
        measured = [
            (point_id, _pack_pixel(center_x, base_points[point_id])) for point_id in range(1, 5)
        ]
        measured += [
            (source_id, _pack_pixel(center_x, new_points[point_id]))
            for source_id, point_id in negative_map.items()
        ]
        image_path = session / "keyframes" / f"{name}.png"
        session_images[name] = _write_image(image_path, 10 + index)
        session_rows.append(
            {
                "name": name,
                "durable_complete": True,
                "image_size": list(_PACK_SIZE),
                "K": _PACK_K,
                "all_ids": [point_id for point_id, _xy in measured],
                "all_xy": [xy for _point_id, xy in measured],
                "image_path": f"keyframes/{name}.png",
                "ordinal": index + 1,
                "stamp": 1.0 + index,
                "source_epoch": 0,
                "cam_from_world": np.column_stack(
                    (np.eye(3), [-float(center_x), 0.0, 0.0])
                ).tolist(),
            }
        )
        observation_rows += [
            {
                "keyframe_name": name,
                "landmark_id": point_id,
                "xy": xy,
                "observation_source": "keyframe_image_measurement"
                if point_id < 0
                else "frozen_map_keyframe_image_measurement",
            }
            for point_id, xy in measured
        ]
        snapshot_keyframes[name] = {"durable": True, "image_size": list(_PACK_SIZE), "K": _PACK_K}
        new_images.append(
            {
                "name": f"workmap/session-test_{name}.png",
                "camera_id": 2 + index,
                "image_id": 3 + index,
                "center_x": center_x,
                "observations": [
                    (negative_map.get(point_id, point_id), xy) for point_id, xy in measured
                ],
            }
        )
    _write_json(
        session / "session_meta.json", {**context, "writer_healthy": True, "finished_unix_s": 10.0}
    )
    _write_json(
        session / "snapshot.json",
        {**context, "finished_unix_s": 10.0, "keyframes": snapshot_keyframes},
    )
    (session / "keyframes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in session_rows), encoding="utf-8"
    )
    (session / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observation_rows), encoding="utf-8"
    )

    output = tmp_path / "staging"
    model = output / "map/experiments/edm_fixedpose_retriangulate/model"
    model.mkdir(parents=True)
    _pack_model([*base_images, *new_images], {**base_points, **new_points}).write(str(model))
    frame_digest = hashlib.sha256(
        b"".join((model / name).read_bytes() for name in _FRAME_BINS)
    ).hexdigest()
    config = {
        "fim": {"weak_when": "<=", "threshold": 0.5, "score_name": "fim_score"},
        "lwtl": {"weak_when": "<=", "threshold": 0.5, "score_name": "lwtl_score"},
        "intersection_radius": 0.5,
    }
    grid = [
        {"xyz": [0.0, 0.0, 5.0], "fim_score": 0.1, "lwtl_score": 0.2},
        {"xyz": [1.0, 0.0, 5.0], "fim_score": 0.9, "lwtl_score": 0.1},
    ]
    receipt: dict[str, object] = {
        "base_bundle_sha256": bundle_sha,
        "session_id": "session-test",
        "gauge": {
            "pass": True,
            "fit": {"landmark_ids": list(range(1, 7))},
            "holdout": {"landmark_ids": list(range(7, 13))},
        },
        "edm_validation": {
            "new_old_pairs": 8,
            "new_new_pairs": 2,
            "provenance": {
                "implementation": "test fixture",
                "base_bundle_sha256": bundle_sha,
                "base_model_sha256": bundle["model_sha256"],
                "base_reference_manifest_sha256": _sha256(reference_manifest),
                "edm_checkpoint_sha256": "1" * 64,
                "boq_checkpoint_sha256": "2" * 64,
                "profile_sha256": "3" * 64,
                "session_id": "session-test",
                "session_input_sha256": {
                    "session_meta_sha256": _sha256(session / "session_meta.json"),
                    "keyframes_jsonl_sha256": _sha256(session / "keyframes.jsonl"),
                    "observations_jsonl_sha256": _sha256(session / "observations.jsonl"),
                    "keyframe_images_sha256": session_images,
                },
                "matcher": {"reciprocal_tolerance_px": 2.0},
                "source_sha256": {
                    field: str(index) * 64
                    for index, field in enumerate(
                        (
                            "reconstruction_tool",
                            "live_provider",
                            "workmap_overlap",
                            "official_edm_adapter",
                        ),
                        start=4,
                    )
                },
                "runtime_versions": {"pycolmap": "4.0.4", "opencv": "4.11", "torch": "2.7"},
            },
        },
        "negative_id_map": {str(key): value for key, value in negative_map.items()},
        "keyframe_image_names": {
            "kf_a": "workmap/session-test_kf_a.png",
            "kf_b": "workmap/session-test_kf_b.png",
        },
        "filtering": {"before": {"points": 10}, "after": {"points": 10}},
        "bundle_adjustment": {"fixed_intrinsics": True, "iterations": 1},
        "fim_lwtl_scored_grid": {
            "schema": "fim-lwtl-scored-grid/v1",
            "model_frame_sha256": frame_digest,
            "config": config,
            "config_sha256": _bytes_sha256(_json_bytes(config)),
            "grid": grid,
            "score_grid_sha256": _bytes_sha256(_json_bytes(grid)),
            "provenance": {"tool": "fixture-scorer", "version": "1", "source_sha256": "8" * 64},
        },
    }
    return base, session, model, output, receipt


class _FakeBoQ:
    def extract_bank(self, paths: list[Path], batch_size: int = 8):
        import numpy as np

        rows = np.zeros((len(paths), 16384), dtype=np.float32)
        for index in range(len(paths)):
            rows[index, 100 + index] = 1.0
        return rows


def test_assemble_workmap_pack_builds_complete_pack_from_synthetic_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.workmap_pack_assets as pack

    base, session, model, output, receipt = _complete_pack_inputs(tmp_path)
    monkeypatch.setattr(pack, "_load_boq_extractor", lambda: (_FakeBoQ(), "f" * 64))
    base_files_before = {path: _sha256(path) for path in base.rglob("*") if path.is_file()}
    model_hashes_before = {name: _sha256(model / name) for name in _MODEL_BINS}

    result = assemble_workmap_pack(base, session, model, output, receipt)

    assert result["schema"] == "online-workmap-pack-assets/v1"
    assert {path: _sha256(path) for path in base.rglob("*") if path.is_file()} == base_files_before
    assert {name: _sha256(model / name) for name in _MODEL_BINS} == model_hashes_before
    keyframes = [
        json.loads(line)
        for line in (output / "map/keyframes/keyframes.jsonl").read_text().splitlines()
    ]
    references = [
        json.loads(line)
        for line in (output / "map/localization/reference_manifest.jsonl").read_text().splitlines()
    ]
    names = json.loads((output / "map/localization/boq_references_sideview.names.json").read_text())
    assert [row["output_name"] for row in keyframes] == [
        "vidA/frame_0.jpg",
        "vidA/frame_1.jpg",
        "workmap/session-test_kf_a.png",
        "workmap/session-test_kf_b.png",
    ]
    assert (
        [row["image_name"] for row in references]
        == names
        == sorted(row["output_name"] for row in keyframes)
    )
    assert references[0]["observations"] == 9
    assert sorted(
        path.name
        for path in (output / "map/experiments/edm_fixedpose_retriangulate/depth_moge3").rglob(
            "*.npz"
        )
    ) == [
        "frame_0.jpg.npz",
        "frame_1.jpg.npz",
    ]
    occupancy = json.loads((model / "occupancy.json").read_text())
    assert occupancy["n_images"] == 4 and occupancy["n_points3D"] == 10
    cells = json.loads((output / "map/localization/fim_lwtl_intersection_cells.json").read_text())
    assert cells["cell_positions"] == [[0.0, 0.0, 5.0]]
    assert [row["matched_negative_ids"] for row in result["new_keyframes"]] == [[-2, -1], [-2, -1]]
    assert all(row["reference_ready"] for row in result["new_keyframes"])
    assert result["boq_bank"]["new_count"] == 2 and result["boq_bank"]["preserved_old_count"] == 2
    copied = output / "map/keyframes/images/vidA/frame_0.jpg"
    source = base / "keyframes/images/vidA/frame_0.jpg"
    assert copied.stat().st_ino != source.stat().st_ino
    assert not any(path.is_symlink() for path in output.rglob("*"))


def test_measured_session_observations_recovers_seed_rows_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    import pytest
    from tools.workmap_pack_assets import (
        WorkmapPackError,
        _measured_session_observations,
    )

    eye_pose = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    pose_2 = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    keyframes = [
        {
            "name": "kf_epoch0",
            "ordinal": 0,
            "source_epoch": 0,
            "stamp": 10.0,
            "cam_from_world": eye_pose,
        },
        {
            "name": "kf_epoch1",
            "ordinal": 0,
            "source_epoch": 1,
            "stamp": 20.0,
            "cam_from_world": pose_2,
        },
    ]
    seen = {"kf_epoch0", "kf_epoch1"}

    # 1. Valid seed observations (with source_epoch and historical without source_epoch)
    obs_file = tmp_path / "obs_valid.jsonl"
    obs_file.write_text(
        json.dumps(
            {
                "landmark_id": -101,
                "xy": [12.0, 34.0],
                "observation_source": "tracked_image_measurement",
                "source_epoch": 0,
                "ordinal": 0,
                "stamp": 10.0,
                "cam_from_world": eye_pose,
            }
        )
        + "\n"
        + json.dumps(
            {
                "landmark_id": -102,
                "xy": [56.0, 78.0],
                "observation_source": "tracked_image_measurement",
                "ordinal": 0,
                "stamp": 20.0,
                "cam_from_world": pose_2,
            }
        )
        + "\n"
        + json.dumps(
            {
                "landmark_id": -103,
                "xy": [90.0, 12.0],
                "observation_source": "visual_pnp_inlier",
                "ordinal": 0,
                "stamp": 10.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    measured = _measured_session_observations(obs_file, seen, keyframes=keyframes)
    assert measured["kf_epoch0"][-101] == (12.0, 34.0)
    assert measured["kf_epoch1"][-102] == (56.0, 78.0)
    # visual_pnp_inlier projected fallback must not be admitted
    assert -103 not in measured["kf_epoch0"] and -103 not in measured["kf_epoch1"]

    # 2. Forged keyframe_name binding rejected
    obs_forged = tmp_path / "obs_forged.jsonl"
    obs_forged.write_text(
        json.dumps(
            {
                "keyframe_name": "kf_epoch1",
                "landmark_id": -104,
                "xy": [1.0, 2.0],
                "observation_source": "tracked_image_measurement",
                "source_epoch": 0,
                "ordinal": 0,
                "stamp": 10.0,
                "cam_from_world": eye_pose,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkmapPackError, match="forged or conflicting keyframe_name"):
        _measured_session_observations(obs_forged, seen, keyframes=keyframes)

    # 3. Non-negative landmark ID in tracked_image_measurement rejected
    obs_positive_id = tmp_path / "obs_pos.jsonl"
    obs_positive_id.write_text(
        json.dumps(
            {
                "landmark_id": 42,
                "xy": [1.0, 2.0],
                "observation_source": "tracked_image_measurement",
                "source_epoch": 0,
                "ordinal": 0,
                "stamp": 10.0,
                "cam_from_world": eye_pose,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkmapPackError, match="non-negative landmark ID"):
        _measured_session_observations(obs_positive_id, seen, keyframes=keyframes)

    # 4. Conflicting measurement for same landmark in same keyframe rejected
    obs_conflict = tmp_path / "obs_conflict.jsonl"
    obs_conflict.write_text(
        json.dumps(
            {
                "landmark_id": -105,
                "xy": [10.0, 20.0],
                "observation_source": "tracked_image_measurement",
                "source_epoch": 0,
                "ordinal": 0,
                "stamp": 10.0,
                "cam_from_world": eye_pose,
            }
        )
        + "\n"
        + json.dumps(
            {
                "landmark_id": -105,
                "xy": [99.0, 99.0],
                "observation_source": "tracked_image_measurement",
                "source_epoch": 0,
                "ordinal": 0,
                "stamp": 10.0,
                "cam_from_world": eye_pose,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkmapPackError, match="conflicting true measurements"):
        _measured_session_observations(obs_conflict, seen, keyframes=keyframes)
