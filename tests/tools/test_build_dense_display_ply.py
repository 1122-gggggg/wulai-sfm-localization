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
