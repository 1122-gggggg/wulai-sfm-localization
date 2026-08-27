"""Map point loaders for the operator map pane."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _read_ply_header(source) -> tuple[str, int, bool]:
    header: list[str] = []
    while True:
        line = source.readline()
        if not line:
            raise ValueError("PLY ended before end_header")
        text = line.decode("ascii", "replace").strip()
        header.append(text)
        if text == "end_header":
            break
    fmt = next((h.split()[1] for h in header if h.startswith("format ")), "")
    vertex_line = next((h for h in header if h.startswith("element vertex ")), "")
    n_vertices = int(vertex_line.split()[-1]) if vertex_line else 0
    return fmt, n_vertices, "property float nx" in header


def _read_binary_ply_points(source, n_vertices: int, step: int,
                            *, has_normals: bool) -> list[tuple]:
    record = struct.Struct("<ffffffBBB" if has_normals else "<fffBBB")
    points: list[tuple] = []
    for index in range(n_vertices):
        raw = source.read(record.size)
        if len(raw) != record.size:
            break
        if index % step:
            continue
        values = record.unpack(raw)
        points.append((*values[:3], *values[-3:]))
    return points


def _read_ascii_ply_points(source, n_vertices: int, step: int,
                           *, has_normals: bool) -> list[tuple]:
    points: list[tuple] = []
    minimum_values = 9 if has_normals else 6
    for index in range(n_vertices):
        raw = source.readline()
        if not raw:
            break
        if index % step:
            continue
        values = raw.split()
        if len(values) < minimum_values:
            continue
        rgb = values[-3:]
        points.append((
            float(values[0]), float(values[1]), float(values[2]),
            int(rgb[0]), int(rgb[1]), int(rgb[2]),
        ))
    return points


def read_ply_points(path: Path, max_points: int) -> np.ndarray:
    with path.open("rb") as source:
        fmt, n_vertices, has_normals = _read_ply_header(source)
        limit = max(1, int(max_points))
        step = max(1, (n_vertices + limit - 1) // limit)
        if fmt == "binary_little_endian":
            points = _read_binary_ply_points(
                source, n_vertices, step, has_normals=has_normals,
            )
        elif fmt == "ascii":
            points = _read_ascii_ply_points(
                source, n_vertices, step, has_normals=has_normals,
            )
        else:
            raise ValueError(f"unsupported PLY format: {fmt}")
    return np.asarray(points, dtype=np.float32)


def read_reference_pose_points(path: Path, max_points: int) -> np.ndarray:
    """Render EDM's reference-pose map as camera centres in the operator map pane."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    poses = raw.get("poses") if isinstance(raw, dict) else None
    if not isinstance(poses, dict) or not poses:
        raise ValueError("reference-pose map must contain a non-empty poses object")
    names = sorted(poses)
    limit = max(1, int(max_points))
    step = max(1, (len(names) + limit - 1) // limit)
    points = []
    for index, name in enumerate(names):
        if index % step:
            continue
        pose = poses[name]
        if not isinstance(pose, dict):
            raise ValueError(f"reference pose {name!r} must be an object")
        R = np.asarray(pose.get("R"), dtype=float)
        t = np.asarray(pose.get("t"), dtype=float)
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError(f"reference pose {name!r} has invalid R/t")
        center = -R.T @ t
        points.append((*center, 84.0, 216.0, 255.0))
    return np.asarray(points, dtype=np.float32)


def read_map_points(path: Path, max_points: int) -> np.ndarray:
    if path.suffix.lower() == ".json":
        return read_reference_pose_points(path, max_points)
    return read_ply_points(path, max_points)
