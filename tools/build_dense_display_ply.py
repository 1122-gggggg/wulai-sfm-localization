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

``--mvs-fused`` writes the colored MVS fused cloud (``fused.ply`` from
``tools/build_mvs_surface.py``) instead. It stays in the release's raw map
coordinates, so waypoints clicked on it are valid. Statistical outliers are
dropped and the cloud is thinned to ``--max-points`` (the editor's own cap) so the
editor never has to stride it. The release must already carry a
``dense_surface.json`` for the same coordinate frame. Needs scipy; use ``.venv-mvs``.

Usage
-----
    .venv/bin/python tools/build_dense_display_ply.py \\
        --release 地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260908
    .venv-mvs/bin/python tools/build_dense_display_ply.py --release <release> \\
        --mvs-fused /path/to/fused.ply
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
MVS_MANIFEST_NAME = "dense_surface.json"
DEFAULT_MAX_POINTS = 3_000_000  # flight_operator_app.ROUTE_EDITOR_POINTS
OUTLIER_NEIGHBORS = 8
OUTLIER_STD_RATIO = 3.0


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


def drop_outliers(xyz: np.ndarray, *, neighbors: int, std_ratio: float) -> np.ndarray:
    """Keep mask: mean distance to the k nearest neighbours <= mean + ratio * std."""
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(xyz).query(xyz, k=neighbors + 1, workers=2)
    mean_distance = distances[:, 1:].mean(axis=1)
    return mean_distance <= mean_distance.mean() + std_ratio * mean_distance.std()


def build_from_mvs(
    release: Path, fused: Path, out: Path | None = None, *, max_points: int = DEFAULT_MAX_POINTS
) -> dict:
    release = Path(release).expanduser().resolve()
    profile = json.loads((release / "site_profile.json").read_text(encoding="utf-8"))
    frame_id = str(profile["coordinate_frame"]["id"])
    map_ply = release / profile["assets"]["map_ply"]
    declared = str(profile["asset_sha256"]["map_ply"])
    actual = sha256_file(map_ply)
    if actual != declared:
        raise ValueError(f"map.ply digest {actual} does not match site_profile.json {declared}")
    manifest_path = (release / profile["assets"]["localization_bundle"]).with_name(
        MVS_MANIFEST_NAME
    )
    try:
        manifest_frame = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "coordinate_frame_id"
        )
    except OSError as exc:
        raise ValueError(f"release has no MVS sidecar {manifest_path}: {exc}") from exc
    if manifest_frame != frame_id:
        raise ValueError(f"{MVS_MANIFEST_NAME} frame {manifest_frame} is not {frame_id}")
    if max_points < 1:
        raise ValueError("--max-points must be positive")
    points = read_ply_points(Path(fused), max_points=1 << 62)
    if not len(points) or not np.isfinite(points[:, :3]).all():
        raise ValueError(f"fused cloud is empty or has non-finite coordinates: {fused}")
    keep = drop_outliers(points[:, :3], neighbors=OUTLIER_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    points = points[keep]
    if len(points) > max_points:
        chosen = np.sort(np.random.default_rng(0).choice(len(points), max_points, replace=False))
        points = points[chosen]
    target = out or map_ply.with_name(DENSE_DISPLAY_PLY_NAME)
    write_dense_ply(
        target,
        points[:, :3],
        points[:, 3:6].astype(np.uint8),
        coordinate_frame_id=frame_id,
        map_ply_sha256=actual,
    )
    return {
        "output": str(target),
        "dense_points": int(len(points)),
        "fused_points": int(len(keep)),
        "outliers_dropped": int((~keep).sum()),
        "fused_sha256": sha256_file(Path(fused)),
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
    parser.add_argument(
        "--mvs-fused",
        type=Path,
        default=None,
        help="write this MVS fused.ply (outliers dropped, thinned) instead of the sparse model",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="point cap for --mvs-fused (default: the route editor's 3,000,000 cap)",
    )
    args = parser.parse_args(argv)
    if args.mvs_fused is not None:
        summary = build_from_mvs(args.release, args.mvs_fused, args.out, max_points=args.max_points)
    else:
        summary = build(args.release, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
