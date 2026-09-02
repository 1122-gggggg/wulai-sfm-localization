#!/usr/bin/env python3
"""Convert a portable direct MegaLoc/EDM/COLMAP map into the production EDM bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch

EDM_W = 1024
EDM_H = 576
COARSE_STRIDE = 8
GRID_W = EDM_W // COARSE_STRIDE
GRID_H = EDM_H // COARSE_STRIDE
N_CELLS = GRID_W * GRID_H
DESCRIPTOR_DIM = 8448
COVIS_KEEP = 40


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_sha256(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_dir.glob("*.bin"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def query_camera(source: Path) -> dict:
    raw = json.loads((source / "config/river_intrinsics_undistorted.json").read_text())
    matrix = raw["K"]
    return {
        "model": "PINHOLE",
        "width": int(raw["image_width"]),
        "height": int(raw["image_height"]),
        "params": [
            float(matrix[0][0]),
            float(matrix[1][1]),
            float(matrix[0][2]),
            float(matrix[1][2]),
        ],
    }


def load_reference_bank(source: Path) -> tuple[list[str], np.ndarray]:
    localization = source / "map/localization"
    names = json.loads((localization / "megaloc_references.names.json").read_text())
    descriptors = np.load(localization / "megaloc_references.npy", allow_pickle=False)
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("reference names must be unique and non-empty")
    if descriptors.shape != (len(names), DESCRIPTOR_DIM):
        raise ValueError(
            f"descriptor shape {descriptors.shape} != ({len(names)}, {DESCRIPTOR_DIM})"
        )
    descriptors = np.asarray(descriptors, dtype=np.float32)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    if not np.isfinite(descriptors).all() or np.any(norms <= 1e-12):
        raise ValueError("MegaLoc descriptors are not finite and non-zero")
    return [str(name) for name in names], descriptors / norms


def build_covisibility(
    reconstruction: pycolmap.Reconstruction,
    names: list[str],
    images_by_name: dict,
) -> dict[str, list[int]]:
    index_by_image_id = {
        int(images_by_name[name].image_id): index for index, name in enumerate(names)
    }
    shared = np.zeros((len(names), len(names)), dtype=np.int32)
    for point in reconstruction.points3D.values():
        viewers = sorted(
            {
                index_by_image_id[int(element.image_id)]
                for element in point.track.elements
                if int(element.image_id) in index_by_image_id
            }
        )
        if len(viewers) < 2:
            continue
        viewer_array = np.asarray(viewers, dtype=np.int64)
        for left in viewers:
            shared[left, viewer_array] += 1
    np.fill_diagonal(shared, 0)
    covis: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        order = np.argsort(-shared[index], kind="stable")
        covis[name] = [
            int(other)
            for other in order
            if shared[index, other] > 0
        ][:COVIS_KEEP]
    return covis


def observation_lut(
    reconstruction: pycolmap.Reconstruction,
    image,
    camera,
    *,
    lift_distance_px: float,
) -> np.ndarray:
    scale_x = EDM_W / float(camera.width)
    scale_y = EDM_H / float(camera.height)
    candidates: list[tuple[float, int, np.ndarray]] = []
    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue
        xy = np.asarray(point2d.xy, dtype=np.float64)
        edm_xy = xy * np.asarray((scale_x, scale_y))
        cell_x = int(np.rint(edm_xy[0] / COARSE_STRIDE))
        cell_y = int(np.rint(edm_xy[1] / COARSE_STRIDE))
        if not (0 <= cell_x < GRID_W and 0 <= cell_y < GRID_H):
            continue
        anchor_source = np.asarray(
            (
                cell_x * COARSE_STRIDE / scale_x,
                cell_y * COARSE_STRIDE / scale_y,
            )
        )
        distance = float(np.linalg.norm(xy - anchor_source))
        if distance > lift_distance_px:
            continue
        xyz = np.asarray(
            reconstruction.points3D[int(point2d.point3D_id)].xyz,
            dtype=np.float32,
        )
        if np.isfinite(xyz).all():
            candidates.append((distance, cell_y * GRID_W + cell_x, xyz))

    lut = np.full((N_CELLS, 3), np.nan, dtype=np.float32)
    for _distance, cell, xyz in sorted(candidates, key=lambda item: item[0]):
        if not np.isfinite(lut[cell]).all():
            lut[cell] = xyz
    return lut


def embedded_reference(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode reference image: {image_path}")
    if image.shape != (EDM_H, EDM_W):
        image = cv2.resize(image, (EDM_W, EDM_H), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )
    if not ok:
        raise ValueError(f"cannot encode reference image: {image_path}")
    return np.frombuffer(encoded.tobytes(), dtype=np.uint8)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build(source: Path, out_dir: Path, lift_distance_px: float) -> dict:
    source = source.expanduser().resolve(strict=True)
    manifest = json.loads((source / "BUNDLE_MANIFEST.json").read_text())
    if manifest.get("artifact_type") != "PORTABLE_MEGALOC_EDM_PNP_BUNDLE":
        raise ValueError("source is not a portable MegaLoc/EDM/PnP bundle")

    model_dir = source / "map/model"
    image_root = source / "map/keyframes/images"
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    names, descriptors = load_reference_bank(source)
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    missing = sorted(set(names) - set(images_by_name))
    if missing:
        raise ValueError(f"reference absent from reconstruction: {missing[0]}")

    camera_doc = query_camera(source)
    model_digest = model_sha256(model_dir)
    frame_id = f"{manifest['map_id']}_{model_digest[:12]}"
    centers = np.empty((len(names), 3), dtype=np.float32)
    yaws = np.empty(len(names), dtype=np.float32)
    poses: dict[str, dict] = {}
    refs: dict[str, dict] = {}
    anchored_counts: list[int] = []

    for index, name in enumerate(names):
        image = images_by_name[name]
        camera = reconstruction.cameras[image.camera_id]
        cam_from_world = image.cam_from_world()
        rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(cam_from_world.translation, dtype=np.float64)
        centers[index] = (-rotation.T @ translation).astype(np.float32)
        forward = rotation[2]
        yaws[index] = np.float32(np.arctan2(forward[0], forward[2]))
        poses[name] = {
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "camera_id": "query_camera",
        }
        xyz = observation_lut(
            reconstruction,
            image,
            camera,
            lift_distance_px=lift_distance_px,
        )
        anchored_counts.append(int(np.isfinite(xyz[:, 0]).sum()))
        refs[name] = {
            "xyz_by_cell": xyz,
            "image_jpg": embedded_reference(image_root / name),
        }
        if (index + 1) % 100 == 0 or index + 1 == len(names):
            print(
                f"packed {index + 1}/{len(names)} "
                f"anchors={anchored_counts[-1]}",
                flush=True,
            )

    covis = build_covisibility(reconstruction, names, images_by_name)
    scale_center = np.median(centers, axis=0)
    scale_s = float(
        2.0 * np.percentile(np.linalg.norm(centers - scale_center, axis=1), 95)
    )
    bundle = {
        "meta": {
            "feature": "edm",
            "matcher": "edm",
            "vpr": "MegaLoc",
            "bundle_vpr": "megaloc",
            "vpr_input": 322,
            "edm_input_w": EDM_W,
            "edm_input_h": EDM_H,
            "edm_grid_w": GRID_W,
            "edm_grid_h": GRID_H,
            "keypoint_identity": (
                "direction-01 reference grid anchor; nearest registered COLMAP "
                f"observation within {lift_distance_px:g}px at source resolution"
            ),
            "xyz_source": "portable direct-map registered observations",
            "refs": len(names),
            "total_3d_anchored_cells": int(sum(anchored_counts)),
            "mean_3d_anchored_per_ref": float(np.mean(anchored_counts)),
            "median_3d_anchored_per_ref": float(np.median(anchored_counts)),
            "reference_images_embedded": True,
            "source_map_id": manifest["map_id"],
            "source_map_status": manifest.get("map_status"),
            "source_validation": manifest.get("validation"),
            "source_model_sha256": model_digest,
            "lift_distance_px": float(lift_distance_px),
        },
        "ref_names": names,
        "ref_global": descriptors,
        "refs": refs,
        "ref_centers": centers,
        "ref_yaws": yaws,
        "covis": covis,
    }
    poses_document = {
        "schema": "reference-poses/v2",
        "schema_version": 2,
        "coordinate_frame_id": frame_id,
        "source_model_sha256": model_digest,
        "camera": camera_doc,
        "cameras": {"query_camera": camera_doc},
        "ref_names": names,
        "poses": poses,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "localization_bundle.pt"
    temporary = bundle_path.with_name(f".{bundle_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(bundle, temporary)
        os.replace(temporary, bundle_path)
    finally:
        temporary.unlink(missing_ok=True)
    poses_path = out_dir / "reference_poses.json"
    write_json(poses_path, poses_document)
    shutil.copy2(source / "map/original_rgb.ply", out_dir / "map.ply")

    summary = {
        "source": str(source),
        "source_archive_sha256": manifest.get("archive_sha256"),
        "source_map_id": manifest["map_id"],
        "source_map_status": manifest.get("map_status"),
        "source_validation": manifest.get("validation"),
        "frame_id": frame_id,
        "source_model_sha256": model_digest,
        "reference_count": len(names),
        "registered_images": reconstruction.num_reg_images(),
        "points3D": reconstruction.num_points3D(),
        "anchored_cells_total": int(sum(anchored_counts)),
        "anchored_cells_mean": float(np.mean(anchored_counts)),
        "anchored_cells_median": float(np.median(anchored_counts)),
        "anchored_cells_min": int(min(anchored_counts)),
        "lift_distance_px": float(lift_distance_px),
        "S": scale_s,
        "radius_candidate": 0.40 * scale_s,
        "max_jump_candidate": 0.40 * scale_s,
        "query_camera": camera_doc,
        "bundle_sha256": sha256_file(bundle_path),
        "poses_sha256": sha256_file(poses_path),
        "ply_sha256": sha256_file(out_dir / "map.ply"),
        "mean_covis": float(np.mean([len(value) for value in covis.values()])),
    }
    write_json(out_dir / "import_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lift-distance-px", type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.lift_distance_px) or args.lift_distance_px <= 0:
        raise SystemExit("--lift-distance-px must be finite and positive")
    summary = build(args.source, args.out_dir, args.lift_distance_px)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
