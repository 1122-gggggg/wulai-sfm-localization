#!/usr/bin/env python3
"""Build a measured, support-trimmed COLMAP surface in the original map frame.

Run separately from localization, for example::

    .venv-mvs/bin/python tools/build_mvs_surface.py \
        --site-profile <release>/site_profile.json --output-dir /data/mvs-work

The environment needs pycolmap-cuda12==4.0.4, Open3D, numpy, Pillow, scipy and trimesh.
The source direct bundle must contain its complete, hash-bound COLMAP model and
registered source images at the model camera resolution. No display/gravity
transform or metric scale is applied to geometry. Unknown scale stays unknown.

Poisson reconstructs across holes. Before publication we discard triangles
without nearby fused MVS measurements and triangles exceeding local sample
spacing. This is a conservative sampling heuristic, not proof of visibility,
complete geometry, obstacle clearance, or flight safety. Small holes and wrong
MVS correspondences can survive; textureless, reflective and moving surfaces
are particularly unreliable. The input release and flight approvals are not
modified; only a mesh and dense_surface.json are added beside direct_bundle.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "定位演算法" / "deploy_code" / "sfm_direct_deploy"
SURFACE_SCHEMA = "mvs-surface/v1"
MANIFEST_NAME = "dense_surface.json"
CACHE_GB = 1.0
NUM_THREADS = 2
NUM_SOURCE_IMAGES = 8
POISSON_DEPTH = 12
FACE_BATCH_SIZE = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"Non-finite JSON value {value!r} in {path}")

    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw


def validate_scale(meters_per_map_unit: float | None, scale_source: str | None) -> None:
    if (meters_per_map_unit is None) != (scale_source is None):
        raise ValueError("Supply --meters-per-map-unit and --scale-source together, or neither")
    if meters_per_map_unit is not None:
        if not math.isfinite(meters_per_map_unit) or meters_per_map_unit <= 0:
            raise ValueError("--meters-per-map-unit must be positive and finite")
        if not isinstance(scale_source, str) or not scale_source.strip():
            raise ValueError("--scale-source must identify the verified calibration")


def load_cuda_pycolmap():
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            "Use a separate .venv-mvs with pycolmap-cuda12==4.0.4, numpy, Pillow, "
            "scipy and trimesh; the localization environment is not changed."
        ) from exc
    if pycolmap.__version__ != "4.0.4":
        raise RuntimeError("This tool requires pycolmap-cuda12==4.0.4 (verified MVS API)")
    if not pycolmap.has_cuda:
        raise RuntimeError(
            "Installed pycolmap is CPU-only; patch_match_stereo requires CUDA. "
            "Create .venv-mvs and install pycolmap-cuda12==4.0.4 there, not plain "
            "pycolmap. A compatible NVIDIA driver and available CUDA GPU are required."
        )
    return pycolmap


def load_source(site_profile: Path):
    """Use the production bundle loader, including eager model/manifest digests."""
    if str(DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(DEPLOY_DIR))
    from direct_map import DirectMapAssets

    profile_path = Path(site_profile).expanduser().resolve(strict=True)
    profile = _read_json(profile_path)
    if profile.get("localizer") != "direct":
        raise ValueError("--site-profile must select the direct localization backend")
    bundle_name = profile.get("assets", {}).get("localization_bundle")
    expected = profile.get("asset_sha256", {}).get("localization_bundle")
    if not isinstance(bundle_name, str) or not bundle_name:
        raise ValueError("site profile has no assets.localization_bundle")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("site profile has no trusted asset_sha256.localization_bundle")
    # Mission snapshots can point outside their own folder. The bundle loader
    # enforces the release boundary of the actual bundle, not the snapshot.
    bundle_path = Path(bundle_name).expanduser()
    if not bundle_path.is_absolute():
        bundle_path = profile_path.parent / bundle_path
    assets = DirectMapAssets.load(bundle_path, expected_sha256=expected)
    if profile.get("coordinate_frame", {}).get("id") != assets.coordinate_frame_id:
        raise ValueError("site profile and direct bundle coordinate frames disagree")
    return assets


def verify_registered_images(assets, reconstruction) -> list[str]:
    """Verify ALL registered images, even when reconstructing a debug subset."""
    from PIL import Image

    names: list[str] = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        name = image.name
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not name:
            raise ValueError(f"Unsafe registered image name: {name!r}")
        path = assets.verify_reference_image(name)
        camera = reconstruction.camera(image.camera_id)
        with Image.open(path) as source:
            size = source.size
            source.verify()
        expected = (int(camera.width), int(camera.height))
        if size != expected:
            raise ValueError(
                f"Registered image {name} is {size[0]}x{size[1]}, but its COLMAP "
                f"camera is {expected[0]}x{expected[1]}; supply the original "
                "hash-bound source images, not resized runtime/display images"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Registered COLMAP image names must be unique")
    if len(names) < 3:
        raise ValueError("MVS requires at least three registered source images")
    return sorted(names)


def select_images(registered: list[str], selected: list[str] | None) -> list[str]:
    if selected is None:
        return registered
    names = sorted(set(selected))
    unknown = sorted(set(names) - set(registered))
    if unknown:
        raise ValueError(f"--image-name is not registered in this model: {unknown}")
    if len(names) < 3:
        raise ValueError("A debug reconstruction needs at least three distinct --image-name values")
    return names


def write_patch_match_config(
    path: Path, reconstruction, selected: list[str], *, partial: bool
) -> None:
    if not partial:
        path.write_text(
            "".join(f"{name}\n__auto__, {NUM_SOURCE_IMAGES}\n" for name in selected),
            encoding="utf-8",
        )
        return
    # COLMAP 4.0.4 undistortion writes the FULL sparse reconstruction even
    # when image_names selects only some bitmaps. __auto__ could therefore
    # choose nonexistent source bitmaps. Rank ONLY selected images by shared
    # sparse points and explicitly name the strongest overlapping sources.
    selected_set = set(selected)
    tracks = {
        image.name: {point.point3D_id for point in image.points2D if point.has_point3D()}
        for image in (reconstruction.image(i) for i in reconstruction.reg_image_ids())
        if image.name in selected_set
    }
    rows = []
    for name in selected:
        overlap = [
            (len(tracks[name] & tracks[source]), source) for source in selected if source != name
        ]
        sources = [
            source
            for count, source in sorted(overlap, key=lambda item: (-item[0], item[1]))
            if count > 0
        ][:NUM_SOURCE_IMAGES]
        if len(sources) < 2:
            raise ValueError(
                f"Debug image {name} needs at least two overlapping selected sources; "
                "select adjacent, co-visible registered frames"
            )
        rows.append(f"{name}\n{', '.join(sources)}\n")
    path.write_text("".join(rows), encoding="utf-8")


def supported_faces(vertices, faces, measurements):
    """Reject unsupported Poisson triangles using a bounded local sampling rule.

    Every vertex, edge midpoint and centroid must be within 1.5 local fused
    sample spacings. Every edge must be <= 3 times the smallest vertex spacing.
    Local spacing is the third nearest DISTINCT neighbor, capped at the global
    median spacing so isolated measurements cannot authorize large triangles.
    No faces are added, holes filled, coordinates transformed or units assumed.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    points = np.unique(np.asarray(measurements, dtype=np.float64), axis=0)
    if len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Stereo fusion produced fewer than four finite distinct measurements")
    tree = cKDTree(points)
    sample = points[:: max(1, math.ceil(len(points) / 100_000))]
    distances, _ = tree.query(sample, k=4, workers=NUM_THREADS)
    global_spacing = float(np.median(distances[:, 3]))
    if not math.isfinite(global_spacing) or global_spacing <= 0:
        raise ValueError("Cannot determine positive fused MVS sample spacing")
    keep = np.zeros(len(faces), dtype=bool)
    for start in range(0, len(faces), FACE_BATCH_SIZE):
        triangles = vertices[faces[start : start + FACE_BATCH_SIZE]]
        probes = np.concatenate(
            [
                triangles,
                (triangles + np.roll(triangles, -1, axis=1)) / 2,
                triangles.mean(axis=1, keepdims=True),
            ],
            axis=1,
        )
        flat_probes = probes.reshape(-1, 3)
        nearest_distance, nearest_id = tree.query(flat_probes, workers=NUM_THREADS)
        spacing, _ = tree.query(points[nearest_id], k=4, workers=NUM_THREADS)
        local_spacing = np.minimum(spacing[:, 3], global_spacing).reshape(-1, 7)
        near = (nearest_distance.reshape(-1, 7) <= 1.5 * local_spacing).all(axis=1)
        edges = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
        small = edges.max(axis=1) <= 3.0 * local_spacing[:, :3].min(axis=1)
        area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        keep[start : start + len(triangles)] = near & small & (area > 0)
    return keep, global_spacing


def poisson_surface(fused_cloud: Path, target: Path) -> dict[str, Any]:
    """Reconstruct oriented MVS measurements with the CPU Open3D Poisson writer."""
    import numpy as np
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(fused_cloud))
    if cloud.is_empty() or not cloud.has_normals():
        raise ValueError("Poisson reconstruction requires fused points with oriented normals")
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    if (
        not np.isfinite(points).all()
        or not np.isfinite(normals).all()
        or np.any(np.linalg.norm(normals, axis=1) == 0)
    ):
        raise ValueError("Poisson reconstruction requires finite points and nonzero finite normals")
    mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=POISSON_DEPTH,
        n_threads=NUM_THREADS,
    )
    if not o3d.io.write_triangle_mesh(str(target), mesh, write_ascii=False):
        raise OSError(f"Could not write Poisson surface: {target}")
    return {
        "implementation": "open3d",
        "version": o3d.__version__,
        "depth": POISSON_DEPTH,
        "n_threads": NUM_THREADS,
    }


def filter_surface(raw_mesh: Path, fused_cloud: Path, target: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    mesh = trimesh.load(raw_mesh, process=False)
    cloud = trimesh.load(fused_cloud, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Poisson meshing did not produce a nonempty triangle mesh")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if not np.isfinite(vertices).all() or (np.abs(vertices) > np.finfo(np.float32).max).any():
        raise ValueError("Poisson mesh contains non-finite or float32-unrepresentable coordinates")
    # PLY exports and the runtime consume float32 vertices. Filter degeneracy
    # in that representation, not in higher precision that publication loses.
    vertices = vertices.astype(np.float32).astype(np.float64)
    mesh.vertices = vertices
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Poisson mesh contains out-of-range triangle indices")
    if not hasattr(cloud, "vertices"):
        raise ValueError("Stereo fusion did not produce a readable PLY point cloud")
    keep, spacing = supported_faces(vertices, faces, cloud.vertices)
    original_count = int(len(faces))
    if not keep.any():
        raise ValueError(
            "No measurement-supported triangles remain; no manifest was published. "
            "Inspect the workspace depth maps, fused.ply and poisson.ply."
        )
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.export(target, file_type="ply", encoding="binary_little_endian")
    return {
        "fused_points": int(len(cloud.vertices)),
        "poisson_triangles": original_count,
        "supported_triangles": int(len(mesh.faces)),
        "supported_vertices": int(len(mesh.vertices)),
        "support_filter": {
            "method": "fused-spacing-probes/v1",
            "probes": ["vertices", "edge_midpoints", "centroid"],
            "nearest_measurement_radius_in_local_spacings": 1.5,
            "max_edge_in_local_spacings": 3.0,
            "local_spacing_neighbor": 3,
            "global_spacing_cap_map_units": spacing,
        },
    }


def publish_surface(
    assets,
    mesh: Path,
    metadata: dict,
    *,
    meters_per_map_unit: float | None,
    scale_source: str | None,
) -> dict:
    """Write immutable mesh first, then atomically replace the sidecar commit."""
    if sha256_file(assets.bundle_path) != assets.sha256:
        raise ValueError("direct_bundle.json changed during reconstruction; refusing publication")
    mesh_sha = sha256_file(mesh)
    destination_dir = assets.bundle_path.parent / "dense_surface"
    if destination_dir.is_symlink():
        raise ValueError(f"Surface output directory must not be a symlink: {destination_dir}")
    destination_dir.mkdir(exist_ok=True)
    destination = destination_dir / f"{mesh_sha}.ply"
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != mesh_sha:
            raise ValueError(f"Existing content-addressed mesh is corrupt: {destination}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".mesh-", dir=destination_dir)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output, mesh.open("rb") as source:
                shutil.copyfileobj(source, output, length=1 << 20)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {
        "schema": SURFACE_SCHEMA,
        "coordinate_frame_id": assets.coordinate_frame_id,
        "source_bundle_sha256": assets.sha256,
        "mesh": destination.relative_to(assets.bundle_path.parent).as_posix(),
        "mesh_sha256": mesh_sha,
        "geometry_units": "map",
        "meters_per_map_unit": meters_per_map_unit,
        "scale_source": scale_source,
        "reconstruction": metadata,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".dense_surface-", dir=assets.bundle_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, assets.bundle_path.parent / MANIFEST_NAME)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def build(
    site_profile: Path,
    output_dir: Path,
    *,
    max_image_size: int = 960,
    meters_per_map_unit: float | None = None,
    scale_source: str | None = None,
    image_names: list[str] | None = None,
) -> dict:
    validate_scale(meters_per_map_unit, scale_source)
    if max_image_size <= 0:
        raise ValueError("--max-image-size must be positive")
    pycolmap = load_cuda_pycolmap()
    assets = load_source(site_profile)
    reconstruction = pycolmap.Reconstruction(assets.model_dir)
    registered = verify_registered_images(assets, reconstruction)
    selected = select_images(registered, image_names)
    workspace_root = Path(output_dir).expanduser().resolve()
    for protected in (assets.model_dir, assets.keyframes_images_root, assets.bundle_path.parent):
        if workspace_root == protected or protected in workspace_root.parents:
            raise ValueError("--output-dir must be outside source model/images/bundle directories")
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="mvs-", dir=workspace_root))
    # Fresh workspaces cannot reuse stale COLMAP products from another bundle.
    # COPY prevents undistortion from sharing mutable source-image inodes.
    pycolmap.undistort_images(
        output_path=workspace,
        input_path=assets.model_dir,
        image_path=assets.keyframes_images_root,
        image_names=selected,
        output_type="COLMAP",
        copy_policy=pycolmap.FileCopyType.copy,
        num_patch_match_src_images=NUM_SOURCE_IMAGES,
        undistort_options=pycolmap.UndistortCameraOptions(max_image_size=max_image_size),
        num_threads=NUM_THREADS,
    )
    config = workspace / "stereo" / "patch-match.cfg"
    write_patch_match_config(
        config, reconstruction, selected, partial=len(selected) != len(registered)
    )
    patch_options = pycolmap.PatchMatchOptions(
        max_image_size=max_image_size,
        cache_size=CACHE_GB,
        gpu_index="0",
        geom_consistency=True,
        filter=True,
        filter_min_num_consistent=2,
        num_threads=NUM_THREADS,
    )
    pycolmap.patch_match_stereo(
        workspace_path=workspace,
        workspace_format="COLMAP",
        options=patch_options,
        config_path=config,
    )
    fused = workspace / "fused.ply"
    fusion_options = pycolmap.StereoFusionOptions(
        max_image_size=max_image_size,
        use_cache=False,
        num_threads=NUM_THREADS,
        min_num_pixels=3,
    )
    pycolmap.stereo_fusion(
        output_path=fused,
        workspace_path=workspace,
        workspace_format="COLMAP",
        input_type="geometric",
        output_type="ply",
        options=fusion_options,
    )
    raw_mesh = workspace / "poisson.ply"
    poisson_options = poisson_surface(fused, raw_mesh)
    mesh = workspace / "supported_surface.ply"
    geometry = filter_surface(raw_mesh, fused, mesh)
    metadata = {
        "pipeline": "pycolmap-mvs-supported-poisson/v1",
        "pycolmap_version": pycolmap.__version__,
        "source_model_sha256": dict(assets.raw["model_sha256"]),
        "registered_image_count": len(registered),
        "selected_image_count": len(selected),
        "partial_reconstruction": len(selected) != len(registered),
        "selected_image_names": selected,
        "selected_image_sha256": {
            name: assets.keyframe_index()[name]["image_sha256"] for name in selected
        },
        "max_image_size": max_image_size,
        "patch_match_options": patch_options.todict(),
        "stereo_fusion_options": {
            "use_cache": False,
            "num_threads": NUM_THREADS,
            "min_num_pixels": 3,
            "max_image_size": max_image_size,
            "input_type": "geometric",
        },
        "poisson_options": poisson_options,
        "max_source_images": NUM_SOURCE_IMAGES,
        "geometry_transform": "identity",
        "coverage_complete": False,
        "obstacle_safety_validated": False,
        "limitations": [
            "Unobserved areas are unknown, not free space; no hole filling is added.",
            "Support trimming is a sampling heuristic: small holes, Poisson smoothing "
            "and erroneous MVS correspondences may survive.",
            "Textureless, reflective and moving surfaces may be absent or incorrect.",
            "Cache/image/source limits reduce memory use, but do not guarantee a GPU memory ceiling.",
        ],
        **geometry,
    }
    manifest = publish_surface(
        assets, mesh, metadata, meters_per_map_unit=meters_per_map_unit, scale_source=scale_source
    )
    return {
        "manifest": str(assets.bundle_path.parent / MANIFEST_NAME),
        "workspace": str(workspace),
        **manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--site-profile", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="workspace root; each run uses a fresh subdirectory",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=960,
        help="maximum width/height for MVS (default: 960)",
    )
    parser.add_argument(
        "--meters-per-map-unit",
        type=float,
        help="verified calibration only; never inferred from flight settings",
    )
    parser.add_argument(
        "--scale-source", help="provenance for --meters-per-map-unit (both flags required together)"
    )
    parser.add_argument(
        "--image-name",
        action="append",
        help="registered name; repeat for a partial debug reconstruction (minimum three)",
    )
    args = parser.parse_args(argv)
    try:
        summary = build(
            args.site_profile,
            args.output_dir,
            max_image_size=args.max_image_size,
            meters_per_map_unit=args.meters_per_map_unit,
            scale_source=args.scale_source,
            image_names=args.image_name,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"MVS build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
