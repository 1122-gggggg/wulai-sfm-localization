#!/usr/bin/env python3
"""Compare production EDM 1024x576 with letterboxed 640x384 and 640x480."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONTROL = WORKSPACE / "控制介面程式"
DEPLOY = ROOT / "deploy_code" / "sfm_glomap_deploy"
for candidate in (CONTROL, DEPLOY):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from edm_matcher import EDM_H, EDM_W, EDMMatcher  # noqa: E402
from production_edm_tracker import reprojection_metrics, spatially_cap_indices  # noqa: E402
from reloc_localizer_edm import EDMRelocMap, MegaLocQuery  # noqa: E402
from site_profile import load_site_profile  # noqa: E402


DEFAULT_SITE = WORKSPACE / "地圖檔" / "場域" / "river_site" / "site_profile.json"
SIZES = ((1024, 576), (640, 384), (640, 480))


def _metric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    array = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "mean": float(np.mean(array)),
    }


def _letterbox(gray: np.ndarray, size: tuple[int, int]):
    width, height = size
    scale = min(width / EDM_W, height / EDM_H)
    content_w = int(round(EDM_W * scale))
    content_h = int(round(EDM_H * scale))
    resized = cv2.resize(gray, (content_w, content_h), interpolation=cv2.INTER_AREA)
    pad_x = (width - content_w) // 2
    pad_y = (height - content_h) // 2
    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[pad_y : pad_y + content_h, pad_x : pad_x + content_w] = resized
    return canvas, (scale, pad_x, pad_y, content_w, content_h)


def _to_source(points: np.ndarray, geometry) -> tuple[np.ndarray, np.ndarray]:
    scale, pad_x, pad_y, content_w, content_h = geometry
    valid = (
        (points[:, 0] >= pad_x)
        & (points[:, 0] < pad_x + content_w)
        & (points[:, 1] >= pad_y)
        & (points[:, 1] < pad_y + content_h)
    )
    source = (points - np.asarray((pad_x, pad_y), np.float32)) / float(scale)
    return source, valid


def _rows_from_matches(reloc_map, ref_names, results, geometry, camera_scale):
    rows = []
    for name, result in zip(ref_names, results):
        k0 = np.asarray(result["mkpts0"], np.float32)
        k1 = np.asarray(result["mkpts1"], np.float32)
        confidence = np.asarray(result["mconf"], np.float32)
        if not len(k0):
            rows.append((np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)))
            continue
        ref_source, ref_inside = _to_source(k0, geometry)
        query_source, query_inside = _to_source(k1, geometry)
        keep = ~EDMMatcher.is_refined(k0)
        keep &= ref_inside & query_inside
        if not keep.any():
            rows.append((np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)))
            continue
        cells = EDMMatcher.cell_ids(ref_source[keep])
        xyz = reloc_map.xyz_by_cell[name][cells]
        points2d = query_source[keep] * camera_scale
        conf = confidence[keep]
        finite = np.isfinite(xyz).all(1) & np.isfinite(points2d).all(1) & np.isfinite(conf)
        rows.append((points2d[finite], xyz[finite], conf[finite]))
    return rows


def _estimate_best(rows, camera, min_inliers: int = 80):
    best = None
    for points2d, points3d, confidence in rows:
        if len(points3d) < 6:
            continue
        selected = spatially_cap_indices(
            points2d, confidence, 900, camera.width, camera.height, 8
        )
        p2 = np.asarray(points2d[selected], float)
        p3 = np.asarray(points3d[selected], float)
        options = pycolmap.AbsolutePoseEstimationOptions()
        options.ransac.max_error = 5.0
        options.ransac.random_seed = 0
        estimate = pycolmap.estimate_and_refine_absolute_pose(p2, p3, camera, options)
        if estimate is None:
            continue
        metrics = reprojection_metrics(estimate, p2, p3, camera, 8)
        candidate = (int(estimate["num_inliers"]), estimate, metrics)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return {"accepted": False, "inliers": 0, "center": None, "rotation": None}
    inliers, estimate, metrics = best
    transform = estimate["cam_from_world"]
    rotation = transform.rotation.matrix()
    translation = np.asarray(transform.translation)
    center = -rotation.T @ translation
    accepted = (
        inliers >= min_inliers
        and metrics["inlier_ratio"] >= 0.15
        and metrics["inlier_grid_cells"] >= 6
        and metrics["reproj_rms"] <= 5.0
    )
    return {
        "accepted": bool(accepted),
        "inliers": inliers,
        "reproj_rms": float(metrics["reproj_rms"]),
        "inlier_ratio": float(metrics["inlier_ratio"]),
        "inlier_grid_cells": int(metrics["inlier_grid_cells"]),
        "center": center.tolist(),
        "rotation": rotation.tolist(),
    }


def _rotation_delta(left, right) -> float:
    relative = np.asarray(left) @ np.asarray(right).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _samples(args, site, reloc_map):
    query = MegaLocQuery(backend="tensorrt")
    refs = torch.from_numpy(reloc_map.ref_global).to(query.device)
    capture = cv2.VideoCapture(str(args.video))
    samples = []
    source_index = 0
    try:
        while len(samples) < args.max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            index = source_index
            source_index += 1
            if index % args.stride:
                continue
            bgr = cv2.resize(
                bgr,
                (site.query_camera.width, site.query_camera.height),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            descriptor = query.extract_one_tensor(rgb)
            order = torch.topk(refs @ descriptor, args.references).indices.cpu().tolist()
            gray = EDMMatcher.load_gray(bgr)
            samples.append(
                {
                    "source_index": index,
                    "frame_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
                    "gray": gray,
                    "refs": [reloc_map.ref_names[item] for item in order],
                }
            )
    finally:
        capture.release()
    return samples


def _run_size(size, samples, reloc_map, camera, camera_scale, batch_size: int):
    matcher = EDMMatcher(
        input_size=size,
        reference_feature_cache_size=max(64, len(reloc_map.ref_names)),
        host_reference_feature_cache_size=0,
        runtime_sigma_mode="reference_grid",
    )
    transformed_refs = {}
    results = []
    timings = []
    for sample in samples:
        query_input, geometry = _letterbox(sample["gray"], size)
        reference_inputs = []
        for name in sample["refs"]:
            if name not in transformed_refs:
                transformed_refs[name] = _letterbox(reloc_map.images[name], size)[0]
            reference_inputs.append(transformed_refs[name])
        prepared = matcher.prepare_query(query_input)
        torch.cuda.synchronize()
        started = time.perf_counter()
        matches = []
        for offset in range(0, len(reference_inputs), batch_size):
            matches.extend(
                matcher.match_many_to_one(
                    reference_inputs[offset : offset + batch_size],
                    query_input,
                    prepared_query=prepared,
                )
            )
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1e3)
        rows = _rows_from_matches(
            reloc_map, sample["refs"], matches, geometry, camera_scale
        )
        results.append(_estimate_best(rows, camera))
    del matcher
    torch.cuda.empty_cache()
    return {
        "input_size": list(size),
        "coarse_cells": size[0] // 8 * (size[1] // 8),
        "match_ms": _metric(timings),
        "accepted": sum(item["accepted"] for item in results),
        "inliers": _metric([item["inliers"] for item in results]),
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--references", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    site = load_site_profile(args.site_profile.expanduser().resolve())
    reloc_map = EDMRelocMap.load(
        site.localization_bundle,
        expected_sha256=site.asset_sha256.localization_bundle,
    )
    camera = pycolmap.Camera(
        model=site.query_camera.model,
        width=site.query_camera.width,
        height=site.query_camera.height,
        params=site.query_camera.params,
    )
    camera_scale = np.asarray(
        (site.query_camera.width / EDM_W, site.query_camera.height / EDM_H),
        np.float32,
    )
    samples = _samples(args, site, reloc_map)
    runs = [
        _run_size(size, samples, reloc_map, camera, camera_scale, args.batch_size)
        for size in SIZES
    ]
    baseline = runs[0]
    for candidate in runs[1:]:
        center_delta = []
        rotation_delta = []
        for dense, reduced in zip(baseline["results"], candidate["results"]):
            if dense["accepted"] and reduced["accepted"]:
                center_delta.append(
                    float(
                        np.linalg.norm(
                            np.asarray(dense["center"]) - np.asarray(reduced["center"])
                        )
                    )
                )
                rotation_delta.append(
                    _rotation_delta(dense["rotation"], reduced["rotation"])
                )
        candidate["center_delta"] = _metric(center_delta)
        candidate["rotation_delta_deg"] = _metric(rotation_delta)
        candidate["speedup"] = baseline["match_ms"]["p50"] / candidate["match_ms"]["p50"]
        candidate["no_loss"] = candidate["accepted"] >= baseline["accepted"]
    for run in runs:
        run.pop("results")
    schedule = [
        (sample["source_index"], sample["frame_sha256"], sample["refs"])
        for sample in samples
    ]
    result = {
        "schema": "edm-input-resolution-benchmark/v1",
        "video": str(args.video.expanduser().resolve()),
        "schedule_sha256": hashlib.sha256(
            json.dumps(schedule, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "frames": len(samples),
        "references_per_frame": args.references,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if all(run.get("no_loss", True) for run in runs) else 4


if __name__ == "__main__":
    raise SystemExit(main())
