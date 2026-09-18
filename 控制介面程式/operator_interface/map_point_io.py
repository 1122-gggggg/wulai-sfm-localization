"""Map point loaders for the operator map pane."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_ply_header(source) -> tuple[str, int, bool, bool, int, bool]:
    header: list[str] = []
    header_bytes_len = 0
    while True:
        line = source.readline()
        if not line:
            raise ValueError("PLY ended before end_header")
        header_bytes_len += len(line)
        text = line.decode("ascii", "replace").strip()
        header.append(text)
        if text == "end_header":
            break
    fmt = next((h.split()[1] for h in header if h.startswith("format ")), "")
    vertex_line = next((h for h in header if h.startswith("element vertex ")), "")
    n_vertices = int(vertex_line.split()[-1]) if vertex_line else 0
    has_normals = any(
        "property float nx" in h
        or "property float normal_x" in h
        or h == "property float nx"
        or h == "property float normal_x"
        for h in header
    )
    has_colors = any(
        any(k in h for k in ("red", "diffuse_red"))
        for h in header
    )
    has_alpha = any("alpha" in h for h in header)
    return fmt, n_vertices, has_normals, has_colors, header_bytes_len, has_alpha


def _read_binary_ply_points(
    source,
    n_vertices: int,
    step: int,
    *,
    has_normals: bool,
    has_colors: bool = True,
    has_alpha: bool = False,
    header_pos: int | None = None,
) -> list[tuple] | np.ndarray:
    if header_pos is not None and hasattr(source, "seek"):
        try:
            source.seek(header_pos)
        except Exception:
            pass

    fields = [("pos", "<f4", (3,))]
    if has_normals:
        fields.append(("norm", "<f4", (3,)))
    if has_colors:
        fields.append(("rgb", "u1", (4 if has_alpha else 3,)))
    dt = np.dtype(fields)
    itemsize = dt.itemsize

    # Bound the temporary payload independently of the cloud size. Keep the
    # sampling phase relative to the whole file, including across chunks.
    points = np.empty(((n_vertices + step - 1) // step, 6), dtype=np.float32)
    written = 0
    for offset in range(0, n_vertices, 65536):
        count = min(65536, n_vertices - offset)
        raw = source.read(count * itemsize)
        complete = len(raw) // itemsize
        sliced = np.frombuffer(raw, dtype=dt, count=complete)[(-offset) % step::step]
        end = written + len(sliced)
        points[written:end, :3] = sliced["pos"]
        if has_colors:
            points[written:end, 3:] = sliced["rgb"][:, :3]
        else:
            points[written:end, 3:] = (200.0, 200.0, 200.0)
        written = end
        if complete < count:
            break
    return points[:written]


def _read_ascii_ply_points(
    source,
    n_vertices: int,
    step: int,
    *,
    has_normals: bool,
    has_colors: bool = True,
    has_alpha: bool = False,
) -> list[tuple]:
    points: list[tuple] = []
    min_vals = 3
    if has_normals:
        min_vals += 3
    if has_colors:
        min_vals += (4 if has_alpha else 3)
    for index in range(n_vertices):
        raw = source.readline()
        if not raw:
            break
        if index % step:
            continue
        values = raw.split()
        if len(values) < min_vals:
            continue
        xyz = (float(values[0]), float(values[1]), float(values[2]))
        if has_colors:
            rgb_parts = values[-4:-1] if has_alpha else values[-3:]
            rgb = (int(rgb_parts[0]), int(rgb_parts[1]), int(rgb_parts[2]))
        else:
            rgb = (200.0, 200.0, 200.0)
        points.append((*xyz, *rgb))
    return points


def read_ply_points(path: Path, max_points: int) -> np.ndarray:
    with path.open("rb") as source:
        fmt, n_vertices, has_normals, has_colors, header_pos, has_alpha = (
            _read_ply_header(source)
        )
        limit = max(1, int(max_points))
        step = max(1, (n_vertices + limit - 1) // limit)
        if fmt == "binary_little_endian":
            points = _read_binary_ply_points(
                source,
                n_vertices,
                step,
                has_normals=has_normals,
                has_colors=has_colors,
                has_alpha=has_alpha,
                header_pos=header_pos,
            )
        elif fmt == "ascii":
            points = _read_ascii_ply_points(
                source,
                n_vertices,
                step,
                has_normals=has_normals,
                has_colors=has_colors,
                has_alpha=has_alpha,
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

