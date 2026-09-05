#!/usr/bin/env python3
"""Augment the EDM map with references inside the measured failure spheres.

The 2026-09-05 corpus census showed 95% of localizer failures clustered in the
14 ``red_intrinsic`` spheres of ``empirical_fail_regions.json``. This tool adds
references there without re-running SfM:

``select``
    Mine the corpus replay rows for *successful* frames whose accepted pose
    falls inside a failure sphere, preferring the videos reserved for reference
    building so the remaining videos stay an uncontaminated holdout.

``register``
    For each candidate frame: match it against the existing references, lift
    the reference-side coarse cells to map-frame 3D through their
    ``xyz_by_cell`` LUTs, PnP a cam_from_world pose, and -- after the same
    inlier / reprojection quality floors the tracker uses for acquire -- pack
    the frame as a new bundle reference (image JPEG, own 3D anchor LUT, MegaLoc
    descriptor, covisibility). Writes a complete candidate release directory
    plus a candidate site profile; the shipped site profile is not touched.

New-reference geometry mirrors ``import_direct_edm_bundle.observation_lut``: a
cell is anchored only by a 3D point observed within ``lift_distance_px`` of the
cell centre, so the runtime's "reference pixel == cell centre" contract holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

LOCALIZATION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = LOCALIZATION_ROOT.parent
DEPLOY_ROOT = LOCALIZATION_ROOT / "deploy_code" / "sfm_glomap_deploy"
for candidate in (VALIDATION_ROOT, DEPLOY_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

EDM_W, EDM_H = 1024, 576
COARSE_STRIDE = 8
GRID_W, GRID_H = EDM_W // COARSE_STRIDE, EDM_H // COARSE_STRIDE
N_CELLS = GRID_W * GRID_H
COVIS_KEEP = 40
PNP_RANSAC_SEED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _video_tag(name: str) -> str:
    return re.sub(r"_720p$", "", Path(name).stem)


def _load_raw_bundle(path: Path):
    """Load one EDM bundle with the same trust boundary as EDMRelocMap.load."""
    import torch  # noqa: PLC0415

    dtypes = (np.float16, np.float32, np.float64, np.uint8, np.int32, np.int64, np.bool_)
    numpy_safe_globals = [
        np._core.multiarray._reconstruct,
        np._core.multiarray.scalar,
        np.ndarray,
        np.dtype,
        *(type(np.dtype(dtype)) for dtype in dtypes),
    ]
    with torch.serialization.safe_globals(numpy_safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def _success_rows(corpus_json: Path, min_inliers: int, max_reproj: float) -> list[dict]:
    document = json.loads(corpus_json.read_text(encoding="utf-8"))
    rows = document.get("rows") or []
    out = []
    for row in rows:
        if not row.get("success"):
            continue
        center = row.get("center")
        if not isinstance(center, list) or len(center) != 3:
            continue
        xyz = np.asarray(center, dtype=float)
        if not np.isfinite(xyz).all():
            continue
        inliers = row.get("inliers")
        if not isinstance(inliers, (int, float)) or float(inliers) < min_inliers:
            continue
        reproj = row.get("reproj_rms")
        if isinstance(reproj, (int, float)) and float(reproj) > max_reproj:
            continue
        frame_id = str(row.get("frame_id") or "")
        match = re.fullmatch(r"src_(\d+)", frame_id)
        if match is None:
            continue
        out.append(
            {
                "src_index": int(match.group(1)),
                "center": xyz.tolist(),
                "inliers": int(inliers),
                "reproj_rms": None if reproj is None else float(reproj),
            }
        )
    return out


def _frames_by_tag(args: argparse.Namespace, required: list[str]) -> dict[str, list[dict]]:
    frames_by_tag: dict[str, list[dict]] = {}
    for corpus_json in sorted(args.corpus_dir.glob("*.json")):
        frames_by_tag[_video_tag(corpus_json.stem)] = _success_rows(
            corpus_json, args.min_inliers, args.max_reproj
        )
    missing = [tag for tag in required if not frames_by_tag.get(tag)]
    if missing:
        raise SystemExit(
            f"corpus rows missing for videos: {missing}; check --corpus-dir"
        )
    return frames_by_tag


def _reference_video_paths(args: argparse.Namespace, ref_tags: list[str]) -> dict[str, str]:
    video_paths: dict[str, str] = {}
    for tag in ref_tags:
        for suffix in (".MP4", "_720p.MP4"):
            candidate_path = Path(args.video_root) / f"{tag}{suffix}"
            if candidate_path.is_file():
                video_paths[tag] = str(candidate_path)
                break
        else:
            raise SystemExit(f"reference video not found under {args.video_root}: {tag}")
    return video_paths


def _pick_for_sphere(
    sphere: dict,
    args: argparse.Namespace,
    ref_tags: list[str],
    frames_by_tag: dict[str, list[dict]],
    video_paths: dict[str, str],
) -> tuple[list[tuple[float, dict, str]], int]:
    """Frames inside one sphere, nearest first and no closer than min_separation."""
    center = np.asarray(sphere["center"], dtype=float)
    reach = float(sphere["radius"]) * args.radius_factor
    pool: list[tuple[float, dict, str]] = []
    for tag in ref_tags:
        for frame in frames_by_tag[tag]:
            distance = float(np.linalg.norm(np.asarray(frame["center"]) - center))
            if distance <= reach:
                pool.append((distance, frame, video_paths[tag]))
    pool.sort(key=lambda item: item[0])
    picked: list[tuple[float, dict, str]] = []
    for distance, frame, video_path in pool:
        xyz = np.asarray(frame["center"], dtype=float)
        if all(
            float(np.linalg.norm(xyz - np.asarray(prev[1]["center"])))
            >= args.min_separation
            for prev in picked
        ):
            picked.append((distance, frame, video_path))
        if len(picked) >= args.per_sphere:
            break
    return picked, len(pool)


def select_candidates(args: argparse.Namespace) -> None:
    spheres_doc = json.loads(args.spheres.read_text(encoding="utf-8"))
    spheres = [
        sphere
        for sphere in spheres_doc["spheres"]
        if sphere.get("status") == "red_intrinsic"
    ]
    ref_tags = [_video_tag(tag) for tag in args.ref_videos.split(",") if tag.strip()]
    holdout_tags = [
        _video_tag(tag) for tag in args.holdout_videos.split(",") if tag.strip()
    ]

    frames_by_tag = _frames_by_tag(args, ref_tags + holdout_tags)
    video_paths = _reference_video_paths(args, ref_tags)

    candidates: list[dict] = []
    per_sphere: list[dict] = []
    seen: set[tuple[str, object]] = set()
    for sphere in spheres:
        picked, pool_size = _pick_for_sphere(
            sphere, args, ref_tags, frames_by_tag, video_paths
        )
        per_sphere.append(
            {
                "segment_id": sphere["segment_id"],
                "sphere_center": sphere["center"],
                "sphere_radius": sphere["radius"],
                "candidate_pool": pool_size,
                "selected": len(picked),
            }
        )
        for distance, frame, video_path in picked:
            # A frame can fall inside two adjacent spheres; register it once.
            key = (video_path, frame["src_index"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "sphere_id": sphere["segment_id"],
                    "video_tag": _video_tag(video_path),
                    "video_path": video_path,
                    "src_index": frame["src_index"],
                    "row_center": frame["center"],
                    "row_inliers": frame["inliers"],
                    "distance_to_sphere_center": distance,
                }
            )
    report = {
        "spheres": len(spheres),
        "ref_source_videos": ref_tags,
        "holdout_videos": holdout_tags,
        "per_sphere": per_sphere,
        "candidates": candidates,
        "note": (
            "candidates come only from ref-source videos; holdout videos keep "
            "their rows untouched for uncontaminated evaluation"
        ),
    }
    write_json(args.out, report)
    covered = sum(1 for entry in per_sphere if entry["selected"] > 0)
    print(
        f"spheres covered: {covered}/{len(spheres)}, "
        f"candidates: {len(candidates)} -> {args.out}"
    )


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def _merge_rows(rows: list, max_corr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = [row for row in rows if len(row[0])]
    if not usable:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float32),
        )
    p2d = np.concatenate([row[0] for row in usable], axis=0)
    p3d = np.concatenate([row[1] for row in usable], axis=0)
    confidence = np.concatenate([row[2] for row in usable], axis=0)
    if max_corr > 0 and len(p2d) > max_corr:
        keep = np.argsort(-confidence, kind="stable")[:max_corr]
        p2d, p3d, confidence = p2d[keep], p3d[keep], confidence[keep]
    return p2d, p3d, confidence


def _build_anchor_lut(
    estimate,
    p2d: np.ndarray,
    p3d: np.ndarray,
    scale: np.ndarray,
    lift_px: float,
) -> tuple[np.ndarray, int]:
    """Anchor coarse cells from inlier correspondences, observation_lut rules.

    The runtime contract (``EDMMatcher.is_refined``) puts the reference-side
    keypoint exactly on a multiple of COARSE_STRIDE -- the cell's top-left
    corner in EDM pixels, the same anchor the original import measured against
    (``cell_x * COARSE_STRIDE / scale_x`` in source pixels). A cell is
    therefore anchored only by a 3D point observed within ``lift_px`` (camera
    resolution) of that corner, nearest first; the rest stay NaN.
    """
    inlier_mask = np.asarray(estimate["inlier_mask"], dtype=bool)
    lut = np.full((N_CELLS, 3), np.nan, dtype=np.float32)
    p2d_in = np.asarray(p2d, dtype=float)[inlier_mask]
    p3d_in = np.asarray(p3d, dtype=float)[inlier_mask]
    edm_pixels = p2d_in / scale[None, :]
    cells_xy = np.rint(edm_pixels / COARSE_STRIDE).astype(np.int64)
    valid = (cells_xy[:, 0] >= 0) & (cells_xy[:, 0] < GRID_W)
    valid &= (cells_xy[:, 1] >= 0) & (cells_xy[:, 1] < GRID_H)
    cell_anchors_px = cells_xy * COARSE_STRIDE
    distance = np.linalg.norm(edm_pixels - cell_anchors_px, axis=1) * scale[0]
    valid &= distance <= lift_px
    for (cell_x, cell_y), xyz in zip(
        cells_xy[valid], p3d_in[valid], strict=True
    ):
        cell = int(cell_y) * GRID_W + int(cell_x)
        if not np.isfinite(lut[cell]).all():
            lut[cell] = np.asarray(xyz, dtype=np.float32)
    return lut, int(np.isfinite(lut[:, 0]).sum())


def _covis_neighbors(
    lut: np.ndarray,
    own_name: str,
    new_entries: dict[str, np.ndarray],
    original_names: list[str],
    original_luts: dict[str, np.ndarray],
    radius: float,
) -> list[int]:
    """Covisible references by shared anchored cells with agreeing 3D.

    Index space is the FINAL ref_names order: original references first, then
    the new ones in registration order. ``own_name`` is excluded here so the
    returned offsets are final indices that never point back at self.
    Neighbors are ranked by overlap and truncated to the same COVIS_KEEP
    budget the map build used.
    """
    all_names = original_names + list(new_entries)
    own_finite = np.isfinite(lut[:, 0])
    scored: list[tuple[int, int]] = []
    for offset, name in enumerate(all_names):
        if name == own_name:
            continue
        other = new_entries.get(name)
        if other is None:
            other = original_luts[name]
        shared = own_finite & np.isfinite(other[:, 0])
        if not shared.any():
            continue
        delta = np.linalg.norm(lut[shared] - other[shared], axis=1)
        count = int((delta <= radius).sum())
        if count > 0:
            scored.append((-count, offset))
    scored.sort()
    return [offset for _negative_count, offset in scored[:COVIS_KEEP]]


def register_candidates(args: argparse.Namespace) -> None:
    import pycolmap  # noqa: PLC0415

    from edm_matcher import EDMMatcher  # noqa: PLC0415
    from production_edm_tracker import reprojection_metrics  # noqa: PLC0415
    from reloc_localizer_edm import (  # noqa: PLC0415
        Camera,
        EDMRelocMap,
        EDMLocalizer,
        MegaLocQuery,
    )

    candidates_doc = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = candidates_doc["candidates"]
    if not candidates:
        raise SystemExit("no candidates to register")

    site_profile_path = args.site_profile.resolve()
    site_dir = site_profile_path.parent
    site_raw = json.loads(site_profile_path.read_text(encoding="utf-8"))
    bundle_path = (site_dir / site_raw["assets"]["localization_bundle"]).resolve()
    bundle_sha = str(site_raw["asset_sha256"]["localization_bundle"])
    runtime_profile_path = (site_dir / site_raw["localizer_profile"]).resolve()
    runtime_profile = json.loads(runtime_profile_path.read_text(encoding="utf-8"))
    camera_raw = site_raw["query_camera"]
    tracker_cfg = runtime_profile["tracker"]
    matcher_cfg = runtime_profile["matcher"]

    camera = Camera(
        model=str(camera_raw["model"]),
        width=int(camera_raw["width"]),
        height=int(camera_raw["height"]),
        params=[float(value) for value in camera_raw["params"]],
    )
    scale = np.asarray((camera.width / EDM_W, camera.height / EDM_H), dtype=np.float64)

    reloc_map = EDMRelocMap.load(bundle_path, expected_sha256=bundle_sha)
    matcher = EDMMatcher(
        mconf_thr=float(matcher_cfg["mconf_thr"]),
        topk=int(matcher_cfg["coarse_topk"]),
        fp16=bool(matcher_cfg["fp16"]),
        reference_cache_size=int(matcher_cfg["reference_cache_size"]),
        runtime_sigma_mode=str(matcher_cfg.get("runtime_sigma_mode", "reference_grid")),
        temporal_feature_cache_size=0,
        query_cuda_graph=False,
    )
    matcher.bind_reference_feature_store(
        reloc_map.images, bundle_sha256=bundle_sha, build_if_missing=False
    )
    localizer = EDMLocalizer(
        reloc_map,
        camera,
        matcher=matcher,
        megaloc=MegaLocQuery(),
        pnp_max_error=float(tracker_cfg["pnp_ransac_max_error"]),
    )
    pcam = pycolmap.Camera(
        model=camera.model,
        width=camera.width,
        height=camera.height,
        params=list(camera.params),
    )
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = float(tracker_cfg["pnp_ransac_max_error"])
    options.ransac.random_seed = PNP_RANSAC_SEED
    options.ransac.num_threads = 1

    original_names = list(reloc_map.ref_names)
    centers = np.asarray(reloc_map.ref_centers, dtype=np.float64)
    original_luts = reloc_map.xyz_by_cell

    stack = {
        "matcher": matcher,
        "localizer": localizer,
        "reloc_map": reloc_map,
        "pcam": pcam,
        "options": options,
        "scale": scale,
        "tracker_cfg": tracker_cfg,
        "centers": centers,
        "original_names": original_names,
        "original_luts": original_luts,
        "reprojection_metrics": reprojection_metrics,
        "pycolmap": pycolmap,
    }
    registered: list[dict] = []
    rejected: list[dict] = []
    registered_keys: set[tuple[str, int]] = set()
    for candidate in candidates:
        if (candidate["video_path"], int(candidate["src_index"])) in registered_keys:
            continue
        entry, rejection = _attempt_registration(candidate, args, stack)
        if rejection is not None:
            rejected.append(rejection)
            continue
        assert entry is not None
        registered_keys.add((candidate["video_path"], int(candidate["src_index"])))
        registered.append(entry)
        print(
            f"registered {entry['name']}: inliers={entry['inliers']} "
            f"reproj={entry['reproj_rms']:.3f} anchored={entry['anchored_cells']}",
            flush=True,
        )

    if not registered:
        from collections import Counter  # noqa: PLC0415

        reasons = Counter(entry["reason"] for entry in rejected)
        for entry in rejected[:10]:
            print("REJECTED:", json.dumps(entry, ensure_ascii=False, default=str))
        raise SystemExit(
            f"no candidate passed registration; nothing to write; reasons={dict(reasons)}"
        )
    _write_augmented_release(
        args,
        site_raw,
        runtime_profile,
        registered,
        rejected,
        original_names,
        original_luts,
        bundle_path,
        bundle_sha,
    )


def _attempt_registration(
    candidate: dict, args: argparse.Namespace, stack: dict
) -> tuple[dict | None, dict | None]:
    """Register one candidate frame, or say why it cannot be one.

    Returns ``(entry, None)`` on success and ``(None, rejection)`` otherwise.
    Split out of register_candidates so each rejection gate stays readable.
    """
    matcher = stack["matcher"]
    localizer = stack["localizer"]
    reloc_map = stack["reloc_map"]
    pcam = stack["pcam"]
    options = stack["options"]
    scale = stack["scale"]
    tracker_cfg = stack["tracker_cfg"]
    centers = stack["centers"]
    original_names = stack["original_names"]
    original_luts = stack["original_luts"]
    reprojection_metrics = stack["reprojection_metrics"]
    pycolmap = stack["pycolmap"]

    capture = cv2.VideoCapture(candidate["video_path"])
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(candidate["src_index"]))
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        return None, ({**_public(candidate), "reason": "frame_decode_failed"})
    gray = matcher.load_gray(frame_bgr)
    pose_center = np.asarray(candidate["row_center"], dtype=float)
    distances = np.linalg.norm(centers - pose_center[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    ref_indices = [
        int(index)
        for index in order
        if distances[index] <= args.match_radius
    ][: args.max_match_refs]
    if not ref_indices:
        return None, ({**_public(candidate), "reason": "no_nearby_reference"})
    ref_images = [reloc_map.images[original_names[index]] for index in ref_indices]
    ref_luts = [original_luts[original_names[index]] for index in ref_indices]
    rows = localizer.correspondences_for_sources(
        gray, ref_images, ref_luts, batch_size=2
    )
    p2d, p3d, _confidence = _merge_rows(
        rows, int(tracker_cfg.get("max_corr_total", 900))
    )
    if len(p2d) < args.min_inliers:
        return None, (
            {
                **_public(candidate),
                "reason": "corr_starved",
                "correspondences": int(len(p2d)),
            }
        )
    estimate = pycolmap.estimate_and_refine_absolute_pose(
        np.asarray(p2d, dtype=float),
        np.asarray(p3d, dtype=float),
        pcam,
        options,
    )
    if estimate is None:
        return None, ({**_public(candidate), "reason": "pnp_failed"})
    metrics = reprojection_metrics(
        estimate,
        np.asarray(p2d, dtype=float),
        np.asarray(p3d, dtype=float),
        pcam,
        int(tracker_cfg.get("corr_grid", 8)),
    )
    inliers = int(estimate.get("num_inliers", 0))
    reproj = metrics.get("reproj_rms")
    cam_from_world = estimate["cam_from_world"]
    rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    translation = np.asarray(cam_from_world.translation, dtype=np.float64)
    new_center = (-rotation.T @ translation).astype(float)
    center_error = float(np.linalg.norm(new_center - pose_center))
    if inliers < args.min_inliers or reproj is None or float(reproj) > args.max_reproj:
        return None, (
            {
                **_public(candidate),
                "reason": "quality_gate",
                "inliers": inliers,
                "reproj_rms": None if reproj is None else float(reproj),
            }
        )
    if center_error > args.max_center_error:
        return None, (
            {
                **_public(candidate),
                "reason": "center_mismatch",
                "center_error": center_error,
            }
        )
    lut, anchored = _build_anchor_lut(
        estimate, p2d, p3d, scale, float(args.lift_distance_px)
    )
    if anchored < args.min_anchored_cells:
        # Single video frames carry ~1e2 inliers, not the thousands of
        # COLMAP observations the original build had, so even a healthy
        # registration anchors only a handful of cells once the
        # lift-distance contract is applied. Sparse-but-clean LUTs are the
        # expected shape here; the runtime simply drops unanchored cells.
        print(
            f"rejected {candidate['video_tag']}#{candidate['src_index']}: "
            f"anchored={anchored} inliers={inliers}",
            flush=True,
        )
        return None, {
            **_public(candidate),
            "reason": "too_few_anchored_cells",
            "anchored_cells": anchored,
            "inliers": inliers,
        }

    descriptor = np.asarray(
        localizer.megaloc.extract_one(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ),
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(descriptor))
    if not np.isfinite(descriptor).all() or norm <= 1e-12:
        return None, ({**_public(candidate), "reason": "descriptor_invalid"})
    ok_encoded, encoded = cv2.imencode(
        ".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )
    if not ok_encoded:
        return None, ({**_public(candidate), "reason": "jpeg_encode_failed"})
    forward = rotation[2]
    return (
        {
            "name": f"aug_{candidate['video_tag']}_{candidate['src_index']:06d}",
            "video_tag": candidate["video_tag"],
            "src_index": int(candidate["src_index"]),
            "sphere_id": candidate["sphere_id"],
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "center": new_center.tolist(),
            "yaw": float(np.arctan2(forward[0], forward[2])),
            "inliers": inliers,
            "reproj_rms": float(reproj),
            "anchored_cells": anchored,
            "matched_ref_indices": ref_indices,
            "distance_to_nearest_existing_ref": float(distances[ref_indices[0]]),
            "xyz_lut": lut,
            "descriptor": (descriptor / norm).astype(np.float32),
            "image_jpg": np.frombuffer(encoded.tobytes(), dtype=np.uint8),
        },
        None,
    )


def _public(candidate: dict) -> dict:
    return {
        key: candidate[key]
        for key in ("sphere_id", "video_tag", "src_index", "row_center")
        if key in candidate
    }


def _write_augmented_release(
    args: argparse.Namespace,
    site_raw: dict,
    runtime_profile: dict,
    registered: list[dict],
    rejected: list[dict],
    original_names: list[str],
    original_luts: dict[str, np.ndarray],
    bundle_path: Path,
    bundle_sha: str,
) -> None:
    original = _load_raw_bundle(bundle_path)
    site_dir = args.site_profile.resolve().parent

    names = list(original_names)
    new_entries: dict[str, np.ndarray] = {}
    centers = np.asarray(original["ref_centers"], np.float32)
    yaws = np.asarray(original["ref_yaws"], np.float32)
    descriptors = [np.asarray(original["ref_global"], np.float32)]
    covis = dict(original.get("covis") or {})
    refs = dict(original["refs"])

    for entry in registered:
        name = entry["name"]
        new_entries[name] = entry["xyz_lut"]
        refs[name] = {
            "xyz_by_cell": entry["xyz_lut"],
            "image_jpg": entry["image_jpg"],
        }
        centers = np.vstack([centers, np.asarray(entry["center"], np.float32)[None, :]])
        yaws = np.append(yaws, np.float32(entry["yaw"]))
        descriptors.append(entry["descriptor"][None, :])

    # Covisibility is computed once the final index space is known: original
    # references keep their existing entries, new references get overlap-based
    # neighbors, and each new reference index is appended to the lists of the
    # references it overlaps (beyond the [:covis_per_ref] window, so existing
    # behaviour is untouched).
    names = original_names + list(new_entries)
    for entry in registered:
        index = names.index(entry["name"])
        covis[entry["name"]] = _covis_neighbors(
            entry["xyz_lut"],
            entry["name"],
            new_entries,
            original_names,
            original_luts,
            float(args.covis_radius),
        )
        for neighbor in covis[entry["name"]]:
            neighbor_name = names[neighbor]
            existing_list = covis.get(neighbor_name)
            if existing_list is not None and index not in existing_list:
                existing_list.append(index)

    augmented = {
        "meta": {
            **original["meta"],
            "refs": len(names),
            "augmented_from_bundle_sha256": bundle_sha,
            "augmentation": {
                "tool": "validation/augment_failure_sphere_references.py",
                "registered": len(registered),
                "rejected": len(rejected),
                "ref_source_videos": sorted({entry["video_tag"] for entry in registered}),
                "lift_distance_px": float(args.lift_distance_px),
                "min_inliers": int(args.min_inliers),
                "max_reproj_rms": float(args.max_reproj),
                "registration": (
                    "EDM match against existing references; 3D lifted through "
                    "reference xyz_by_cell LUTs; pycolmap RANSAC PnP seed 0"
                ),
            },
        },
        "ref_names": names,
        "ref_global": np.concatenate(descriptors, axis=0).astype(np.float32),
        "refs": refs,
        "ref_centers": centers,
        "ref_yaws": yaws,
        "covis": covis,
    }

    release_dir = args.release_out.resolve()
    localization_dir = release_dir / "localization"
    localization_dir.mkdir(parents=True, exist_ok=True)
    bundle_out = localization_dir / "localization_bundle.pt"
    temporary = bundle_out.with_name(f".{bundle_out.name}.{os.getpid()}.tmp")
    try:
        import torch  # noqa: PLC0415

        torch.save(augmented, temporary)
        os.replace(temporary, bundle_out)
    finally:
        temporary.unlink(missing_ok=True)

    poses_document = json.loads(
        (bundle_path.parent / "reference_poses.json").read_text(encoding="utf-8")
    )
    for entry in registered:
        poses_document["poses"][entry["name"]] = {
            "R": entry["R"],
            "t": entry["t"],
            "camera_id": "query_camera",
        }
    poses_document["ref_names"] = names
    poses_path = localization_dir / "reference_poses.json"
    write_json(poses_path, poses_document)

    map_ply_source = bundle_path.parent.parent / "map" / "map.ply"
    map_dir = release_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(map_ply_source, map_dir / "map.ply")

    align_source = bundle_path.parent.parent / "compat" / "T_align_gravity.json"
    compat_dir = release_dir / "compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(align_source, compat_dir / "T_align_gravity.json")

    scale_center = np.median(centers.astype(np.float64), axis=0)
    scale_s = float(
        2.0
        * np.percentile(
            np.linalg.norm(centers.astype(np.float64) - scale_center, axis=1), 95
        )
    )
    previous_s = float(runtime_profile["map_scale"]["S"])
    if abs(scale_s - previous_s) / previous_s <= 0.01:
        scale_note = (
            f"S recomputed {scale_s:.10f} within 1% of previous {previous_s:.10f}; "
            "profile keeps the previous S/radius/max_jump to avoid perturbing "
            "validated gates"
        )
        new_s = previous_s
        candidate_tracker = runtime_profile["tracker"]
    else:
        scale_note = f"S recomputed {scale_s:.10f} vs previous {previous_s:.10f}"
        new_s = scale_s
        candidate_tracker = {
            **runtime_profile["tracker"],
            "radius": 0.40 * new_s,
            "max_jump": 0.40 * new_s,
        }
    candidate_profile = {
        **runtime_profile,
        "name": f"{runtime_profile['name']}_aug{len(registered)}",
        "map_scale": {**runtime_profile["map_scale"], "S": new_s, "note": scale_note},
        "tracker": candidate_tracker,
    }
    profile_path = compat_dir / "edm_runtime_profile.json"
    write_json(profile_path, candidate_profile)

    provenance_dir = release_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        provenance_dir / "augmentation_receipt.json",
        {
            "original_bundle_sha256": bundle_sha,
            "augmented_bundle_sha256": sha256_file(bundle_out),
            "reference_count": len(names),
            "added_references": len(registered),
            "rejected_candidates": rejected,
            "S_recomputed": scale_s,
            "S_used": new_s,
            "scale_note": scale_note,
            "registered": [
                {
                    key: (
                        value
                        if key
                        not in ("xyz_lut", "descriptor", "image_jpg", "R", "t")
                        else f"<{key}:{len(value) if hasattr(value, '__len__') else value}>"
                    )
                    for key, value in entry.items()
                }
                for entry in registered
            ],
        },
    )

    relative = release_dir.relative_to(site_dir)
    candidate_site = {
        **site_raw,
        "display_name": f"{site_raw.get('display_name', '')}（候選：補失敗球 reference）",
        "localizer_profile": f"{relative}/compat/edm_runtime_profile.json",
        "map_reference_poses": f"{relative}/localization/reference_poses.json",
        "map_align": f"{relative}/compat/T_align_gravity.json",
        "asset_sha256": {
            "map_ply": sha256_file(map_dir / "map.ply"),
            "localization_bundle": sha256_file(bundle_out),
            "localizer_profile": sha256_file(profile_path),
            "map_reference_poses": sha256_file(poses_path),
            "map_align": site_raw["asset_sha256"]["map_align"],
        },
        "flight": {
            **site_raw.get("flight", {}),
            "approved": False,
            "approval_note": (
                "augmented map candidate: registration receipt in provenance/, "
                "corpus gate pending; flight approval stays disabled"
            ),
        },
        "assets": {
            **site_raw.get("assets", {}),
            "map_ply": f"{relative}/map/map.ply",
            "localization_bundle": f"{relative}/localization/localization_bundle.pt",
        },
    }
    write_json(site_dir / "site_profile_augmented.json", candidate_site)
    print(
        f"augmented release: {release_dir} ({len(registered)} new references, "
        f"{len(names)} total); candidate site profile: "
        f"{site_dir / 'site_profile_augmented.json'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="pick candidate frames from corpus rows")
    select.add_argument("--corpus-dir", type=Path, required=True)
    select.add_argument(
        "--spheres",
        type=Path,
        default=Path("地圖檔/場域/river_site/overlay/empirical_fail_regions.json"),
    )
    select.add_argument("--video-root", type=Path, default=Path("模擬器/測試影片/720p"))
    select.add_argument(
        "--ref-videos", default="P1680168,P1570157,P1670167,河濱_P1170117"
    )
    select.add_argument("--holdout-videos", default="P1160116,P1180118,P1190119")
    select.add_argument("--per-sphere", type=int, default=8)
    select.add_argument("--min-separation", type=float, default=0.2)
    select.add_argument("--radius-factor", type=float, default=1.5)
    select.add_argument("--min-inliers", type=int, default=80)
    select.add_argument("--max-reproj", type=float, default=3.0)
    select.add_argument("--out", type=Path, required=True)

    register = sub.add_parser(
        "register", help="register candidates and write the augmented release"
    )
    register.add_argument("--candidates", type=Path, required=True)
    register.add_argument(
        "--site-profile",
        type=Path,
        default=Path("地圖檔/場域/river_site/site_profile.json"),
    )
    register.add_argument("--release-out", type=Path, required=True)
    register.add_argument("--min-inliers", type=int, default=80)
    register.add_argument("--max-reproj", type=float, default=3.0)
    register.add_argument("--max-center-error", type=float, default=1.0)
    register.add_argument("--min-anchored-cells", type=int, default=12)
    register.add_argument("--lift-distance-px", type=float, default=2.0)
    register.add_argument("--max-match-refs", type=int, default=8)
    register.add_argument("--match-radius", type=float, default=2.0)
    register.add_argument("--covis-radius", type=float, default=0.2)

    args = parser.parse_args(argv)
    if args.command == "select":
        select_candidates(args)
    else:
        register_candidates(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
