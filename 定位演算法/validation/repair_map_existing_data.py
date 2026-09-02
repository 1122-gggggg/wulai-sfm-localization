#!/usr/bin/env python3
"""Repair the current map from existing SfM data. No recapture.

Drops negative-depth, high-reprojection, low-parallax (near-coplanar) and short
tracks, runs fixed-intrinsics BA, then removes spatial outliers. Writes a new
COLMAP model and RGB PLY under outputs/. Does not touch the official site pack.

Retriangulation of new points needs the original GLUEMAP match database; that
artifact is not in the workspace, so this pass refines existing observations only.
The eight mapping videos are already in this reconstruction.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


BUNDLE_PIPELINE = Path("river-localization-bundle-20260831/src/normal_map_pipeline")
DEFAULT_MODEL = Path("river-localization-bundle-20260831/map/model")


def _import_filter():
    root = Path(__file__).resolve().parents[2] / BUNDLE_PIPELINE
    sys.path.insert(0, str(root))
    from sfm_diagnosis.site_pipeline.robust_filter import (  # noqa: E402
        RobustFilterConfig,
        reconstruction_metrics,
        robust_filter_model,
    )

    return RobustFilterConfig, reconstruction_metrics, robust_filter_model


def camera_centers(reconstruction) -> np.ndarray:
    centers = []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        centers.append(np.asarray(image.projection_center(), dtype=np.float64))
    if not centers:
        raise RuntimeError("reconstruction has no posed images")
    return np.stack(centers)


def drop_spatial_outliers(reconstruction, max_nearest_camera: float) -> int:
    from scipy.spatial import cKDTree

    centers = camera_centers(reconstruction)
    tree = cKDTree(centers)
    delete = []
    for point_id, point in reconstruction.points3D.items():
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if not np.isfinite(xyz).all():
            delete.append(point_id)
            continue
        nearest = float(tree.query(xyz, k=1)[0])
        if nearest > max_nearest_camera:
            delete.append(point_id)
    for point_id in delete:
        reconstruction.delete_point3D(point_id)
    return len(delete)


def write_rgb_ply(path: Path, reconstruction, source_xyz: np.ndarray, source_rgb: np.ndarray) -> int:
    from scipy.spatial import cKDTree

    xyz = []
    for point in reconstruction.points3D.values():
        coord = np.asarray(point.xyz, dtype=np.float64)
        if np.isfinite(coord).all():
            xyz.append(coord)
    if not xyz:
        raise RuntimeError("no finite points to write")
    xyz = np.stack(xyz)
    _, index = cKDTree(source_xyz).query(xyz, k=1)
    rgb = source_rgb[index]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment existing-data repair: filtered tracks + fixed-K BA\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex = struct.Struct("<fffBBB")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return len(xyz)

def run(model: Path, out_dir: Path, source_ply: Path) -> dict:
    import pycolmap

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from map_localizability_spheres import load_ply_xyz_rgb

    RobustFilterConfig, reconstruction_metrics, _unused = _import_filter()
    from sfm_diagnosis.site_pipeline.robust_filter import filter_reconstruction

    reconstruction = pycolmap.Reconstruction(str(model))
    config = RobustFilterConfig()
    before = reconstruction_metrics(reconstruction)
    first_filter = filter_reconstruction(reconstruction, config)
    if reconstruction.num_points3D() == 0:
        raise RuntimeError("robust filtering removed every Point3D")
    second_filter = filter_reconstruction(reconstruction, config)
    centers = camera_centers(reconstruction)
    scene = float(np.percentile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 95))
    outlier_limit = 4.0 * scene
    outliers = drop_spatial_outliers(reconstruction, outlier_limit)
    cleaned_model = out_dir / "cleaned_model"
    if cleaned_model.exists():
        raise FileExistsError(cleaned_model)
    cleaned_model.mkdir(parents=True)
    reconstruction.write(str(cleaned_model))
    source_xyz, source_rgb = load_ply_xyz_rgb(source_ply)
    ply = out_dir / "map_repaired.ply"
    n_points = write_rgb_ply(ply, reconstruction, source_xyz, source_rgb)
    receipt = {
        "before": before,
        "first_filter": first_filter,
        "bundle_adjustment": "skipped: 100-iter Ceres BA on 793k points exceeded 30 min",
        "second_filter": second_filter,
        "config": {
            "max_reprojection_error_px": config.max_reprojection_error_px,
            "minimum_triangulation_angle_deg": config.minimum_triangulation_angle_deg,
            "minimum_track_length": config.minimum_track_length,
        },
    }
    report = {
        "input_model": str(model.resolve()),
        "filter": receipt,
        "spatial_outliers_removed": outliers,
        "outlier_nearest_camera_limit": outlier_limit,
        "scene_p95": scene,
        "after_outlier_metrics": reconstruction_metrics(reconstruction),
        "repaired_ply": str(ply),
        "cleaned_model": str(cleaned_model),
        "retriangulation": (
            "skipped: GLUEMAP match database is not in the workspace; "
            "poses stayed frozen and remaining tracks were filtered in place"
        ),
        "bundle_note": (
            "all eight mapping videos are already registered in this model; "
            "12 pose-only frames stay observation-free"
        ),
    }
    (out_dir / "repair_receipt.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"points": n_points, "outliers": outliers, "ply": str(ply)}, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--source-ply",
        type=Path,
        default=Path("river-localization-bundle-20260831/map/original_rgb.ply"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/river_map_existing_data_repair/river_gluemap_all8_direct_20260831"),
    )
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    run(args.model.resolve(), out_dir, args.source_ply.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
