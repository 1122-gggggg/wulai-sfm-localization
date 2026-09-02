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


def _red_sphere_search_paths(map_ply: Path | None) -> list[Path]:
    """Candidate overlay locations for red intrinsic difficult spheres.

    Primary is the site-package overlay (地圖檔/場域/<site>/overlay/...),
    fallback is the repository workspace outputs for the river_site demo.
    The workspace-wide fallbacks are only probed for river_site maps so a
    synthetic test cloud does not suddenly render the river demo spheres.
    """
    candidates: list[Path] = []
    is_river = False
    if map_ply is not None:
        try:
            lower = str(map_ply).lower()
            is_river = "river" in lower or "river_site" in lower
        except Exception:
            is_river = False
        try:
            resolved = Path(map_ply).resolve()
            # Walk the map path's ancestors (map folder -> release -> site root)
            # and probe each overlay/red_spheres_only.ply.
            seen_parents: set[Path] = set()
            for parent in [resolved.parent, *list(resolved.parents)]:
                if parent in seen_parents:
                    continue
                seen_parents.add(parent)
                candidates.append(parent / "overlay" / "red_spheres_only.ply")
                candidates.append(parent / "overlay" / "red_spheres.ply")
                candidates.append(parent / "red_spheres_only.ply")
                # Stop once we have climbed past the workspace root marker.
                if (parent / "控制介面程式").is_dir():
                    break
                if len(seen_parents) > 8:
                    break
        except Exception:
            pass
    # Workspace river_site fallbacks — only for river_site maps (real demo),
    # otherwise a synthetic test cloud would inherit the river demo overlay.
    if is_river:
        try:
            ws_root = Path(__file__).resolve().parents[2]
            candidates.append(ws_root / "地圖檔" / "場域" / "river_site" / "overlay" / "red_spheres_only.ply")
            candidates.append(ws_root / "地圖檔" / "場域" / "river_site" / "overlay" / "red_spheres.ply")
            candidates.append(ws_root / "outputs" / "river_map_localizability" / "river_site_edm_red_only" / "red_spheres_only.ply")
            candidates.append(ws_root / "outputs" / "river_map_localizability" / "river_site_edm" / "weak_spheres_only.ply")
        except Exception:
            pass
    # Deduplicate preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            uniq.append(path)
    return uniq


def load_red_sphere_points(map_ply: Path | None, max_points: int) -> np.ndarray:
    """Load red intrinsic spheres for the site, or empty array if none.

    Probes overlay/red_spheres_only.ply beside the site map, falling back to
    outputs/river_map_localizability/... when the site overlay is missing.
    Never raises for a missing file; a corrupt file is also treated as empty
    so the base map continues to render.
    """
    for cand in _red_sphere_search_paths(map_ply):
        if cand.is_file():
            try:
                return read_ply_points(cand, max_points)
            except Exception:
                continue
    return np.empty((0, 6), dtype=np.float32)


# Backwards-compatible alias
read_red_sphere_points = load_red_sphere_points
try_load_red_spheres = load_red_sphere_points
