#!/usr/bin/env python3
"""Mark outdoor map locations where visual localization is likely to fail.

Geometric proxies of:
  FIF  arXiv:2008.03324  positional Fisher information of visible bearings
  LWL  arXiv:2407.15593  30x30 image-bin occupancy of projected landmarks

ActLoc is not used: its LocMap was trained on indoor HM3D/ScanNet scenes.

Writes a PLY overlay (map points + coloured spheres) and a JSON report.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch


WORLD_UP = {
    "x": np.array([1.0, 0.0, 0.0], np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], np.float64),
    "y": np.array([0.0, 1.0, 0.0], np.float64),
    "-y": np.array([0.0, -1.0, 0.0], np.float64),
    "z": np.array([0.0, 0.0, 1.0], np.float64),
    "-z": np.array([0.0, 0.0, -1.0], np.float64),
}

STATUS_RGB = {
    "red_intrinsic": (220, 32, 32),
    "orange_directional": (255, 140, 0),
    "yellow_marginal": (255, 210, 40),
}

# Outdoor drone FoV: horizon ±, look-down allowed, sky discarded.
PITCH_DEG = np.array([-50.0, -30.0, -15.0, 0.0, 15.0], np.float64)
YAW_DEG = np.arange(-180.0, 180.0, 30.0, dtype=np.float64)
BIN_ROWS = 30
BIN_COLS = 30
BIN_COUNT = BIN_ROWS * BIN_COLS


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _axis_vector(name: str) -> np.ndarray:
    try:
        return WORLD_UP[name].copy()
    except KeyError as exc:
        raise ValueError(f"unsupported up axis {name!r}") from exc


def load_site(profile_path: Path) -> dict:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    root = profile_path.parent
    camera = profile["query_camera"]
    fx, fy, cx, cy = (float(v) for v in camera["params"][:4])
    return {
        "site_id": profile.get("site_id", profile_path.stem),
        "ply": (root / profile["assets"]["map_ply"]).resolve(),
        "poses": (root / profile["map_reference_poses"]).resolve(),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "up": _axis_vector(profile["coordinate_frame"]["up_axis"]),
        "frame_id": profile["coordinate_frame"]["id"],
    }


def load_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as source:
        header: list[str] = []
        while True:
            line = source.readline()
            if not line:
                raise ValueError(f"{path} ended before end_header")
            text = line.decode("ascii", "replace").strip()
            header.append(text)
            if text == "end_header":
                break
        fmt = next((h.split()[1] for h in header if h.startswith("format ")), "")
        vertex_line = next((h for h in header if h.startswith("element vertex ")), "")
        n = int(vertex_line.split()[-1])
        names = [h.split()[-1] for h in header if h.startswith("property ")]
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {fmt}")
        if names[:6] != ["x", "y", "z", "red", "green", "blue"]:
            raise ValueError(f"expected xyz rgb vertices, got {names[:6]}")
        raw = np.fromfile(source, dtype=np.dtype("<f4, <f4, <f4, u1, u1, u1"), count=n)
    xyz = np.stack([raw["f0"], raw["f1"], raw["f2"]], axis=1).astype(np.float64)
    rgb = np.stack([raw["f3"], raw["f4"], raw["f5"]], axis=1)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], rgb[finite]


def load_mapping_cameras(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload["poses"]
    centers = []
    axes = []
    for name in payload.get("ref_names", sorted(poses)):
        pose = poses[name]
        rotation = np.asarray(pose["R"], np.float64)
        translation = np.asarray(pose["t"], np.float64)
        centers.append(-rotation.T @ translation)
        axes.append(rotation[2])
    centers = np.asarray(centers, np.float64)
    axes = np.asarray(axes, np.float64)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True).clip(min=1e-12)
    return centers, axes


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(points / voxel).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def camera_spacing(centers: np.ndarray) -> float:
    if len(centers) < 2:
        raise ValueError("need at least two mapping cameras")
    from scipy.spatial import cKDTree

    nearest = cKDTree(centers).query(centers, k=2)[0][:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-8)]
    if not len(nearest):
        raise ValueError("mapping cameras are coincident")
    return float(np.median(nearest))


def query_positions(centers: np.ndarray, spacing: float, voxel: float) -> np.ndarray:
    keys = np.unique(np.floor(centers / voxel).astype(np.int64), axis=0)
    occupied = {tuple(int(v) for v in key) for key in keys}
    dilated: set[tuple[int, int, int]] = set()
    radius = max(1, int(round(2.0 * spacing / voxel)))
    for x, y, z in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz <= radius * radius:
                        dilated.add((x + dx, y + dy, z + dz))
    grid = (np.asarray(list(dilated), dtype=np.float64) + 0.5) * voxel
    merged = np.concatenate([centers, grid], axis=0)
    return voxel_downsample(merged, voxel * 0.5)


def yaw_pitch_rotations(up: np.ndarray) -> np.ndarray:
    up = up / np.linalg.norm(up)
    ref = np.array([0.0, 0.0, 1.0], np.float64)
    if abs(float(ref @ up)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], np.float64)
    east = np.cross(ref, up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    north /= np.linalg.norm(north)
    yaw, pitch = np.meshgrid(np.deg2rad(YAW_DEG), np.deg2rad(PITCH_DEG), indexing="xy")
    yaw = yaw.reshape(-1)
    pitch = pitch.reshape(-1)
    forward_h = np.cos(yaw)[:, None] * north + np.sin(yaw)[:, None] * east
    forward = np.cos(pitch)[:, None] * forward_h + np.sin(pitch)[:, None] * up
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    down = np.broadcast_to(-up, forward.shape).copy()
    right = np.cross(down, forward)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1).astype(np.float32)


def _snap_view(axis: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    optical = rotations[:, 2, :]
    return (axis @ optical.T).argmax(axis=1)


def evaluate_views(
    positions: np.ndarray,
    rotations: np.ndarray,
    landmarks: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    max_range: float,
    device: torch.device,
    batch: int = 24,
) -> dict[str, np.ndarray]:
    n_pos = len(positions)
    n_dir = len(rotations)
    pos = torch.as_tensor(positions, dtype=torch.float32, device=device)
    rot = torch.as_tensor(rotations, dtype=torch.float32, device=device)
    pts = torch.as_tensor(landmarks, dtype=torch.float32, device=device)
    eye = torch.eye(3, device=device, dtype=torch.float32)

    vis_count = torch.zeros(n_pos, n_dir, device=device)
    occ_ratio = torch.zeros(n_pos, n_dir, device=device)
    lambda_min = torch.zeros(n_pos, n_dir, device=device)

    for start in range(0, n_pos, batch):
        sl = slice(start, min(start + batch, n_pos))
        t = pos[sl]
        rel = pts[None, :, :] - t[:, None, :]
        dist = torch.linalg.norm(rel, dim=2).clamp_min(1e-6)
        near = dist <= max_range
        for d in range(n_dir):
            p_cam = torch.einsum("ij,bnj->bni", rot[d], rel)
            z = p_cam[..., 2]
            u = fx * p_cam[..., 0] / z.clamp_min(1e-6) + cx
            v = fy * p_cam[..., 1] / z.clamp_min(1e-6) + cy
            vis = near & (z > 1e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            vis_f = vis.float()
            vis_count[sl, d] = vis_f.sum(dim=1)
            bx = (u / width * BIN_COLS).long().clamp(0, BIN_COLS - 1)
            by = (v / height * BIN_ROWS).long().clamp(0, BIN_ROWS - 1)
            bins = (by * BIN_COLS + bx).clamp(0, BIN_COUNT - 1)
            occ = torch.zeros(len(t), BIN_COUNT, device=device)
            occ.scatter_add_(1, bins, vis_f)
            occ_ratio[sl, d] = (occ > 0).float().mean(dim=1)

            r = dist.clamp_min(1e-6)
            bvec = rel / r[..., None]
            weight = vis_f / (r * r)
            moment = torch.einsum("bn,bni,bnj->bij", weight, bvec, bvec)
            fim = weight.sum(dim=1)[:, None, None] * eye - moment
            fim = 0.5 * (fim + fim.transpose(1, 2))
            eig = torch.linalg.eigvalsh(fim)
            lambda_min[sl, d] = eig[:, 0].clamp_min(0.0)

    fif = torch.log1p(lambda_min)
    lwl = occ_ratio * torch.log1p(vis_count)
    return {
        "visible": vis_count.cpu().numpy(),
        "occupancy": occ_ratio.cpu().numpy(),
        "fif": fif.cpu().numpy(),
        "lwl": lwl.cpu().numpy(),
        "lambda_min": lambda_min.cpu().numpy(),
    }


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    rank = np.empty(len(values), np.float64)
    rank[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    return rank


def classify_voxels(
    scores: dict[str, np.ndarray],
    snap_index: np.ndarray,
    *,
    weak_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    methods = ("fif", "lwl")
    best = {name: values.max(axis=1) for name, values in scores.items() if name in methods}
    aligned = {
        name: values[np.arange(len(values)), snap_index]
        for name, values in scores.items()
        if name in methods
    }
    best_rank = {name: _percentile_rank(best[name]) for name in methods}
    aligned_rank = {name: _percentile_rank(aligned[name]) for name in methods}
    best_flags = np.stack([best_rank[name] <= weak_q for name in methods], axis=1)
    aligned_flags = np.stack([aligned_rank[name] <= weak_q for name in methods], axis=1)
    best_mean = np.mean([best_rank[name] for name in methods], axis=0)
    aligned_mean = np.mean([aligned_rank[name] for name in methods], axis=0)
    intrinsic = best_flags.all(axis=1) | (best_mean <= weak_q)
    directional = (~intrinsic) & (aligned_flags.all(axis=1) | (aligned_mean <= weak_q))
    marginal = (~intrinsic) & (~directional) & (best_flags.any(axis=1) | aligned_flags.any(axis=1))
    return intrinsic, directional, marginal


def _component_record(
    members: list[int],
    kind: str,
    positions: np.ndarray,
    voxel: float,
    scores: dict[str, np.ndarray],
    snap_index: np.ndarray,
) -> dict:
    pts = positions[members]
    center = pts.mean(axis=0)
    radius = max(float(np.linalg.norm(pts - center, axis=1).max()), voxel)
    radius += 0.6 * voxel
    return {
        "segment_id": 0,
        "status": kind,
        "center": [float(v) for v in center],
        "radius": radius,
        "voxel_count": len(members),
        "best_fif": float(scores["fif"][members].max()),
        "best_lwl": float(scores["lwl"][members].max()),
        "aligned_fif": float(scores["fif"][members, snap_index[members]].mean()),
        "aligned_lwl": float(scores["lwl"][members, snap_index[members]].mean()),
        "best_visible": float(scores["visible"][members].max()),
        "best_occupancy": float(scores["occupancy"][members].max()),
        "best_lambda_min": float(scores["lambda_min"][members].max()),
    }


def _split_members(members: list[int], positions: np.ndarray, voxel: float) -> list[list[int]]:
    pts = positions[members]
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max()) if len(pts) else 0.0
    max_radius = 2.2 * voxel
    if radius <= max_radius or len(members) < 4:
        return [members]
    k = min(len(members), max(2, int(np.ceil(radius / max_radius))))
    from scipy.cluster.vq import kmeans2

    _, labels = kmeans2(pts, k, minit="points", rng=0)
    groups: dict[int, list[int]] = defaultdict(list)
    for member, label in zip(members, labels, strict=True):
        groups[int(label)].append(member)
    return [group for group in groups.values() if group]


def cluster_spheres(
    positions: np.ndarray,
    voxel: float,
    intrinsic: np.ndarray,
    directional: np.ndarray,
    marginal: np.ndarray,
    scores: dict[str, np.ndarray],
    snap_index: np.ndarray,
) -> list[dict]:
    origin = positions.min(axis=0)
    ijk = np.round((positions - origin) / voxel).astype(np.int64)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    spheres: list[dict] = []
    for kind, mask in (
        ("red_intrinsic", intrinsic),
        ("orange_directional", directional),
        ("yellow_marginal", marginal),
    ):
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, key in enumerate(map(tuple, ijk)):
            if mask[index]:
                buckets[key].append(index)
        seen: set[tuple[int, int, int]] = set()
        for seed in buckets:
            if seed in seen:
                continue
            queue = deque([seed])
            seen.add(seed)
            members: list[int] = []
            while queue:
                key = queue.popleft()
                members.extend(buckets[key])
                x, y, z = key
                for dx, dy, dz in neighbors:
                    nxt = (x + dx, y + dy, z + dz)
                    if nxt in buckets and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            for group in _split_members(members, positions, voxel):
                spheres.append(
                    _component_record(group, kind, positions, voxel, scores, snap_index)
                )
    priority = {"red_intrinsic": 0, "orange_directional": 1, "yellow_marginal": 2}
    spheres.sort(key=lambda row: (priority[row["status"]], -row["radius"]))
    for index, row in enumerate(spheres):
        row["segment_id"] = index
    return spheres


def _fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.column_stack((radial * np.cos(theta), radial * np.sin(theta), z))


def _unit_uv_sphere(lat: int = 12, lon: int = 24) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    vertices = []
    for i in range(lat + 1):
        phi = math.pi * i / lat
        for j in range(lon):
            theta = 2.0 * math.pi * j / lon
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            vertices.append((x, y, z))
    faces = []
    for i in range(lat):
        for j in range(lon):
            a = i * lon + j
            b = i * lon + (j + 1) % lon
            c = (i + 1) * lon + j
            d = (i + 1) * lon + (j + 1) % lon
            if i:
                faces.append((a, c, b))
            if i != lat - 1:
                faces.append((b, c, d))
    return np.asarray(vertices, np.float64), faces


def write_overlay_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    spheres: list[dict],
    *,
    samples: int = 1024,
) -> None:
    shells = _fibonacci_sphere(samples)
    count = len(xyz) + samples * len(spheres)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment FIF+LWL localizability spheres on map points\n"
        "comment red=intrinsic weak  orange=viewpoint-only  yellow=marginal\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex = struct.Struct("<fffBBB")
    with path.open("wb") as out:
        out.write(header)
        for point, color in zip(xyz, rgb, strict=True):
            out.write(
                vertex.pack(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )
            )
        for sphere in spheres:
            center = np.asarray(sphere["center"], np.float64)
            color = STATUS_RGB[sphere["status"]]
            pts = center + sphere["radius"] * shells
            for point in pts:
                out.write(vertex.pack(float(point[0]), float(point[1]), float(point[2]), *color))


def write_sphere_mesh_ply(path: Path, spheres: list[dict]) -> None:
    unit, faces = _unit_uv_sphere()
    lines = [
        "ply",
        "format ascii 1.0",
        "comment FIF+LWL localizability failure spheres",
        f"element vertex {len(unit) * len(spheres)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces) * len(spheres)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for sphere in spheres:
        color = STATUS_RGB[sphere["status"]]
        verts = np.asarray(sphere["center"]) + sphere["radius"] * unit
        for x, y, z in verts:
            lines.append(f"{x:.6g} {y:.6g} {z:.6g} {color[0]} {color[1]} {color[2]}")
    for index, _ in enumerate(spheres):
        offset = index * len(unit)
        for a, b, c in faces:
            lines.append(f"3 {a + offset} {b + offset} {c + offset}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(site: dict, out_dir: Path, weak_q: float) -> dict:
    from scipy.spatial import cKDTree

    xyz, rgb = load_ply_xyz_rgb(site["ply"])
    centers, axes = load_mapping_cameras(site["poses"])
    spacing = camera_spacing(centers)
    scene = float(
        np.percentile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 95)
    )
    voxel = float(np.clip(0.06 * scene, 8.0 * spacing, 0.12 * scene))
    landmarks = voxel_downsample(xyz, max(0.012 * scene, 1e-3))
    if len(landmarks) > 20000:
        keep = np.linspace(0, len(landmarks) - 1, 20000, dtype=np.int64)
        landmarks = landmarks[keep]
    queries = query_positions(centers, spacing, voxel)
    if len(queries) > 4000:
        keep = np.linspace(0, len(queries) - 1, 4000, dtype=np.int64)
        queries = queries[keep]
    rotations = yaw_pitch_rotations(site["up"])
    nearest = cKDTree(centers).query(queries, k=1)[1]
    query_snap = _snap_view(axes[nearest], rotations)
    max_range = float(np.clip(0.5 * scene, 20.0 * spacing, 1.2 * scene))
    device = _device()
    print(
        f"map={len(xyz)} landmarks={len(landmarks)} cameras={len(centers)} "
        f"queries={len(queries)} views={len(rotations)} spacing={spacing:.4f} "
        f"scene={scene:.4f} voxel={voxel:.4f} device={device}",
        flush=True,
    )
    scores = evaluate_views(
        queries,
        rotations,
        landmarks,
        fx=site["fx"],
        fy=site["fy"],
        cx=site["cx"],
        cy=site["cy"],
        width=site["width"],
        height=site["height"],
        max_range=max_range,
        device=device,
    )
    occupied = scores["visible"].max(axis=1) >= 8
    if occupied.sum() < 16:
        raise RuntimeError("too few query voxels see landmarks; check camera/map frame")
    masked = {name: values.copy() for name, values in scores.items()}
    for name in ("fif", "lwl"):
        masked[name][~occupied] = masked[name][occupied].min()
    intrinsic, directional, marginal = classify_voxels(masked, query_snap, weak_q=weak_q)
    intrinsic &= occupied
    directional &= occupied
    marginal &= occupied
    spheres = cluster_spheres(queries, voxel, intrinsic, directional, marginal, scores, query_snap)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = out_dir / "map_with_weak_spheres.ply"
    mesh = out_dir / "weak_spheres_only.ply"
    write_overlay_ply(overlay, xyz, rgb, spheres)
    write_sphere_mesh_ply(mesh, spheres)
    report = {
        "site_id": site["site_id"],
        "frame_id": site["frame_id"],
        "papers": {
            "fif": "arXiv:2008.03324 positional bearing Fisher information",
            "lwl": "arXiv:2407.15593 30x30 landmark image-bin occupancy",
        },
        "excluded": {
            "actloc": "arXiv:2508.20981 skipped; LocMap trained on indoor HM3D/ScanNet"
        },
        "counts": {
            "map_points": int(len(xyz)),
            "landmarks": int(len(landmarks)),
            "mapping_cameras": int(len(centers)),
            "query_voxels": int(len(queries)),
            "occupied_voxels": int(occupied.sum()),
            "intrinsic_voxels": int(intrinsic.sum()),
            "directional_voxels": int(directional.sum()),
            "marginal_voxels": int(marginal.sum()),
            "spheres": len(spheres),
        },
        "geometry": {
            "camera_spacing": spacing,
            "scene_p95": scene,
            "voxel": voxel,
            "max_range": max_range,
            "weak_percentile": weak_q,
            "pitch_deg": PITCH_DEG.tolist(),
            "yaw_deg": YAW_DEG.tolist(),
        },
        "outputs": {"overlay_ply": str(overlay), "spheres_ply": str(mesh)},
        "spheres": spheres,
        "legend": {
            "red_intrinsic": "best outdoor yaw/pitch still weak on FIF and LWL",
            "orange_directional": "a better look direction exists; mapping-aligned view is weak",
            "yellow_marginal": "single-method warning",
        },
    }
    (out_dir / "weak_regions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-profile",
        type=Path,
        default=Path("地圖檔/場域/river_site/site_profile.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--map-ply", type=Path, default=None)
    parser.add_argument("--weak-percentile", type=float, default=0.10)
    args = parser.parse_args(argv)
    site = load_site(args.site_profile.resolve())
    if args.map_ply is not None:
        site["ply"] = args.map_ply.resolve()
    out_dir = args.out_dir or Path("outputs/river_map_localizability") / site["site_id"]
    report = run(site, out_dir, args.weak_percentile)
    print(json.dumps(report["counts"], ensure_ascii=False), flush=True)
    print(report["outputs"]["overlay_ply"], flush=True)
    print(report["outputs"]["spheres_ply"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
