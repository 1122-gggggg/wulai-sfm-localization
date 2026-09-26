#!/usr/bin/env python3
"""Write the route editor's dense display cloud for one direct site release.

A release's ``map/map.ply`` is a 1-in-N stride sample of the sparse model,
sized for the operator map pane. Route authoring needs to see the objects
themselves, so this writes every point of ``model/points3D.bin`` to
``map/map_dense.ply``. Its header binds it to the release's coordinate frame
and to the exact ``map.ply`` digest; the route editor ignores the file when
either does not match. Display only: localization never reads it.

The tool refuses to write unless every ``map.ply`` point appears bit-exactly in
the dense cloud, which proves both come from the same model coordinates.

Usage
-----
    .venv/bin/python tools/build_dense_display_ply.py \\
        --release 地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260908
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_OPERATOR = REPO_ROOT / "控制介面程式" / "operator_interface"
if str(_OPERATOR) not in sys.path:
    sys.path.insert(0, str(_OPERATOR))

from map_point_io import (  # noqa: E402
    DENSE_DISPLAY_PLY_NAME,
    DENSE_FRAME_COMMENT,
    DENSE_MAP_SHA_COMMENT,
    read_ply_points,
)

# point3D_id, xyz, rgb, error, track length; the track itself follows.
_POINT_HEAD = struct.Struct("<QdddBBBdQ")
_TRACK_ENTRY_BYTES = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP points3D.bin -> (xyz float64 Nx3, rgb uint8 Nx3) sorted by id."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"points3D.bin is truncated: {path}")
    (count,) = struct.unpack_from("<Q", data, 0)
    ids = np.empty(count, dtype=np.int64)
    xyz = np.empty((count, 3), dtype=np.float64)
    rgb = np.empty((count, 3), dtype=np.uint8)
    offset = 8
    for index in range(count):
        if offset + _POINT_HEAD.size > len(data):
            raise ValueError(f"points3D.bin ends inside point {index}: {path}")
        point_id, x, y, z, red, green, blue, _error, track = _POINT_HEAD.unpack_from(data, offset)
        ids[index] = point_id
        xyz[index] = (x, y, z)
        rgb[index] = (red, green, blue)
        offset += _POINT_HEAD.size + _TRACK_ENTRY_BYTES * track
    if offset != len(data):
        raise ValueError(f"points3D.bin has {len(data) - offset} trailing bytes: {path}")
    if not np.isfinite(xyz).all():
        raise ValueError(f"points3D.bin contains non-finite coordinates: {path}")
    order = np.argsort(ids, kind="stable")
    return xyz[order], rgb[order]


def _row_keys(xyz: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(np.asarray(xyz, dtype="<f4"))
    return packed.view(np.dtype((np.void, 12))).ravel()


def require_map_subset(map_ply: Path, dense_xyz: np.ndarray) -> int:
    """Every map.ply point must be a bit-exact dense point (same model, same frame)."""
    sparse = read_ply_points(map_ply, max_points=1 << 62)
    if not len(sparse):
        raise ValueError(f"map.ply has no points: {map_ply}")
    missing = int((~np.isin(_row_keys(sparse[:, :3]), _row_keys(dense_xyz))).sum())
    if missing:
        raise ValueError(
            f"{missing} of {len(sparse)} map.ply points are not in the sparse model; "
            "map.ply and model/points3D.bin come from different reconstructions"
        )
    return len(sparse)


def write_dense_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    coordinate_frame_id: str,
    map_ply_sha256: str,
) -> None:
    records = np.empty(len(xyz), dtype=[("pos", "<f4", (3,)), ("rgb", "u1", (3,))])
    records["pos"] = xyz
    records["rgb"] = rgb
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment all {len(xyz)} points of the sparse model sorted by point3D_id;"
        " route editor display only\n"
        f"comment {DENSE_FRAME_COMMENT} {coordinate_frame_id}\n"
        f"comment {DENSE_MAP_SHA_COMMENT} {map_ply_sha256}\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())
    os.replace(temporary, path)


def build(release: Path, out: Path | None = None) -> dict:
    release = Path(release).expanduser().resolve()
    profile = json.loads((release / "site_profile.json").read_text(encoding="utf-8"))
    frame_id = str(profile["coordinate_frame"]["id"])
    map_ply = release / profile["assets"]["map_ply"]
    declared = str(profile["asset_sha256"]["map_ply"])
    actual = sha256_file(map_ply)
    if actual != declared:
        raise ValueError(f"map.ply digest {actual} does not match site_profile.json {declared}")
    xyz, rgb = read_points3d_bin(release / "model" / "points3D.bin")
    sparse_count = require_map_subset(map_ply, xyz)
    target = out or map_ply.with_name(DENSE_DISPLAY_PLY_NAME)
    write_dense_ply(target, xyz, rgb, coordinate_frame_id=frame_id, map_ply_sha256=actual)
    return {
        "output": str(target),
        "dense_points": int(len(xyz)),
        "map_ply_points": sparse_count,
        "coordinate_frame_id": frame_id,
        "map_ply_sha256": actual,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True, help="direct site release folder")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output PLY (default: <release>/map/{DENSE_DISPLAY_PLY_NAME})",
    )
    args = parser.parse_args(argv)
    summary = build(args.release, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
