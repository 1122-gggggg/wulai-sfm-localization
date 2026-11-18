"""The dense display cloud is the whole sparse model, bound to its map.ply."""

import hashlib
import json
import struct

import numpy as np
import pytest

import build_dense_display_ply as tool
from map_point_io import dense_display_ply, read_ply_points


def _write_points3d(path, xyz, rgb, ids, tracks):
    chunks = [struct.pack("<Q", len(xyz))]
    for point_id, point, color, track in zip(ids, xyz, rgb, tracks):
        chunks.append(
            struct.pack("<QdddBBBdQ", point_id, *point, *color, 0.5, track) + b"\0" * (8 * track)
        )
    path.write_bytes(b"".join(chunks))


def _write_ply(path, xyz, rgb):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<fffBBB", *p, *c) for p, c in zip(xyz, rgb))
    path.write_bytes(header + body)


def _release(tmp_path, *, sparse_offset=0.0):
    rng = np.random.default_rng(3)
    count = 40
    xyz = rng.normal(size=(count, 3))
    rgb = rng.integers(0, 256, size=(count, 3))
    ids = rng.permutation(np.arange(1, count + 1))
    (tmp_path / "model").mkdir()
    (tmp_path / "map").mkdir()
    _write_points3d(tmp_path / "model" / "points3D.bin", xyz, rgb, ids, [2, 3, 0, 5] * 10)
    order = np.argsort(ids)
    sparse = xyz[order][::7] + sparse_offset
    _write_ply(tmp_path / "map" / "map.ply", sparse, rgb[order][::7])
    digest = hashlib.sha256((tmp_path / "map" / "map.ply").read_bytes()).hexdigest()
    (tmp_path / "site_profile.json").write_text(
        json.dumps(
            {
                "coordinate_frame": {"id": "frame_1"},
                "assets": {"map_ply": "map/map.ply"},
                "asset_sha256": {"map_ply": digest},
            }
        ),
        encoding="utf-8",
    )
    return xyz[order], rgb[order], digest


def test_writes_every_model_point_bound_to_frame_and_map(tmp_path):
    xyz, rgb, digest = _release(tmp_path)

    summary = tool.build(tmp_path)

    assert summary["dense_points"] == 40 and summary["map_ply_points"] == 6
    dense = dense_display_ply(
        tmp_path / "map" / "map.ply", coordinate_frame_id="frame_1", map_ply_sha256=digest
    )
    assert dense == tmp_path / "map" / "map_dense.ply"
    points = read_ply_points(dense, max_points=1000)
    np.testing.assert_array_equal(points[:, :3], xyz.astype(np.float32))
    np.testing.assert_array_equal(points[:, 3:], rgb)


def test_refuses_a_map_ply_from_another_reconstruction(tmp_path):
    _release(tmp_path, sparse_offset=0.25)

    with pytest.raises(ValueError, match="different reconstructions"):
        tool.build(tmp_path)
    assert not (tmp_path / "map" / "map_dense.ply").exists()


def test_refuses_a_map_ply_that_no_longer_matches_the_profile(tmp_path):
    _release(tmp_path)
    (tmp_path / "map" / "map.ply").write_bytes(b"edited")

    with pytest.raises(ValueError, match="does not match site_profile"):
        tool.build(tmp_path)


def _write_fused(path, xyz, rgb):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<ffffffBBB", *p, 0.0, 0.0, 1.0, *c) for p, c in zip(xyz, rgb))
    path.write_bytes(header + body)


def _mvs_release(tmp_path, *, sidecar_frame="frame_1"):
    _release(tmp_path)
    profile = json.loads((tmp_path / "site_profile.json").read_text(encoding="utf-8"))
    profile["assets"]["localization_bundle"] = "localization/direct_bundle.json"
    (tmp_path / "site_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (tmp_path / "localization").mkdir()
    (tmp_path / "localization" / "dense_surface.json").write_text(
        json.dumps({"coordinate_frame_id": sidecar_frame}), encoding="utf-8"
    )
    rng = np.random.default_rng(5)
    xyz = rng.normal(scale=0.05, size=(400, 3)).astype(np.float32)
    xyz[0] = (50.0, 50.0, 50.0)  # floating speck far from the surface
    rgb = rng.integers(0, 256, size=(400, 3)).astype(np.uint8)
    fused = tmp_path / "fused.ply"
    _write_fused(fused, xyz, rgb)
    return fused, xyz, rgb, profile["asset_sha256"]["map_ply"]


def test_mvs_display_cloud_drops_outliers_and_keeps_raw_colors(tmp_path):
    fused, xyz, rgb, digest = _mvs_release(tmp_path)

    summary = tool.build_from_mvs(tmp_path, fused, max_points=10_000)

    assert (
        summary["outliers_dropped"] >= 1
        and summary["dense_points"] == 400 - summary["outliers_dropped"]
    )
    dense = dense_display_ply(
        tmp_path / "map" / "map.ply", coordinate_frame_id="frame_1", map_ply_sha256=digest
    )
    assert dense is not None
    points = read_ply_points(dense, max_points=10_000)
    assert points[:, :3].max() < 1.0  # the 50-unit speck is gone
    # Surviving points keep their own position and color pairing.
    by_position = {tuple(p): tuple(c) for p, c in zip(xyz, rgb)}
    for row in points:
        assert by_position[tuple(row[:3])] == tuple(row[3:])


def test_mvs_display_cloud_never_exceeds_the_editor_point_cap(tmp_path):
    fused, _, _, _ = _mvs_release(tmp_path)

    summary = tool.build_from_mvs(tmp_path, fused, max_points=100)

    assert summary["dense_points"] == 100


def test_mvs_display_cloud_refuses_a_sidecar_from_another_frame(tmp_path):
    fused, _, _, _ = _mvs_release(tmp_path, sidecar_frame="other_frame")

    with pytest.raises(ValueError, match="is not frame_1"):
        tool.build_from_mvs(tmp_path, fused)
    assert not (tmp_path / "map" / "map_dense.ply").exists()


def test_mvs_display_cloud_reads_a_bounded_stride_sample_of_a_huge_source(tmp_path):
    fused, _, _, _ = _mvs_release(tmp_path)

    summary = tool.build_from_mvs(tmp_path, fused, max_points=50)

    # 400 source points, read at most PRESAMPLE_FACTOR * cap before thinning.
    assert summary["sampled_points"] == 50 * tool.PRESAMPLE_FACTOR
    assert summary["dense_points"] <= 50


def test_sidecarless_cloud_must_overlap_map_ply(tmp_path):
    xyz, rgb, _ = _release(tmp_path)
    profile = json.loads((tmp_path / "site_profile.json").read_text(encoding="utf-8"))
    profile["assets"]["localization_bundle"] = "localization/direct_bundle.json"
    (tmp_path / "site_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    xyz = xyz.astype(np.float32)
    aligned, shifted = tmp_path / "aligned.ply", tmp_path / "shifted.ply"
    _write_fused(aligned, xyz, rgb)
    _write_fused(shifted, xyz + 1000.0, rgb)

    with pytest.raises(ValueError, match="another frame or unit"):
        tool.build_from_mvs(tmp_path, shifted)
    assert not (tmp_path / "map" / "map_dense.ply").exists()

    summary = tool.build_from_mvs(tmp_path, aligned)
    assert summary["frame_checked_by"] == "map_overlap"
    assert summary["map_overlap_fraction"] >= tool.MIN_MAP_OVERLAP
