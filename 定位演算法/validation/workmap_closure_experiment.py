#!/usr/bin/env python3
"""Offline closure association experiment; no aircraft SDK, replay, or map writes.

Inputs are a protocol, one replay run, its *unmerged* session snapshot, and saved
reference/query images. --query-images-dir accepts ordinal.png or 000928.png;
images must depict that exact frames.jsonl capture (no seeking/interpolation).
Without it, queries must have an existing saved keyframe PNG. Native-resolution
images are converted to gray and INTER_AREA resized to the protocol's 960x540
fast-loop camera. Prefer the original saved PNG for the first 928 experiment.

Optional --map-observations-json has schema workmap-closure-map-observations/v1,
run_id, coordinate_frame_id, and records containing ordinal, source_epoch,
capture_stamp_mono, ids, image_xy_px, world_xyz_u, K, image_size_px, provenance.
Only positive IDs are accepted, and their map-only PnP is solved here. Otherwise
use same-capture recorded positive observations. No published-pose fallback is
used: every exported anchor must retain its full raw positive-ID PnP support.

Each matcher/association path is evaluated separately, not pooled. nearest uses
unique <=2px landmark association; guided transfers an actual saved observation
using a bounded image-only local warp and bidirectional, unique patch evidence.
Both require consistent matches from at least two covisible saved references.
Local PnP receives only owned, revision-checked child XYZ and measured query
pixels. Reference self-PnP can diagnose/validate inconsistent saved reference
poses without changing any source geometry. No main-map pose initializes a
child or reference PnP. All poses are 3x4 camera-from-world matrices.

Outputs: manifest.json, references.json, pairs.jsonl, queries.jsonl, anchors.json,
summary.json. Reprojection/held-out diagnostics are not position ground truth.
An empty anchor list is a completed strict-failure experiment, not an exception.
Missing/malformed inputs fail before model loading. Existing outputs are refused.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "定位演算法/deploy_code_expand/sfm_direct_deploy"
SIZE = (960, 540)
LOCAL_MIN_INLIERS = 30
MAX_REPROJ_PX = 4.0
MIN_REFERENCE_SUPPORT = 2
NEAREST_PX = 2.0
NEAREST_RATIO = 0.7
NEAREST_MARGIN_PX = 0.75
QUERY_AGREEMENT_PX = 2.0
GUIDED_RADIUS_PX = 32.0
PATCH_RADIUS = 5
PATCH_SEARCH_PX = 6
PATCH_MIN_NCC = 0.8
PATCH_PEAK_MARGIN = 0.05
PATCH_MAX_FB_PX = 0.75
PATCH_MAX_SHIFT_PX = 4.0


class ExperimentError(ValueError):
    pass


def no_constant(value):
    raise ExperimentError(f"non-finite JSON constant: {value}")


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=no_constant)


def json_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def progress(event, **fields):
    print(json_line({"offline_only": True, "event": event, **fields}), flush=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def required_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ExperimentError(f"missing input file: {path}")
    return path


def asset(protocol_path, item):
    path = Path(item["path"])
    if not path.is_absolute():
        path = protocol_path.parent / path
    path = required_file(path)
    actual = digest(path)
    if item.get("sha256") != actual:
        raise ExperimentError(f"protocol asset hash mismatch: {path}")
    return path


def finite_array(value, shape, label, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if array.size == 0 and 0 in shape:
        array = array.reshape(shape)
    if array.shape != shape or not np.isfinite(array).all():
        raise ExperimentError(f"{label}: expected finite shape {shape}, got {array.shape}")
    return array


def integer_ids(value, label):
    raw = np.asarray(value)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        if raw.size == 0 and raw.ndim == 1:
            return np.empty(0, dtype=np.int64)
        raise ExperimentError(f"{label}: IDs must be an integer vector")
    ids = raw.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ExperimentError(f"{label}: duplicate point IDs")
    return ids


def pose_array(value, label):
    pose = finite_array(value, (3, 4), label)
    rotation = pose[:, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ExperimentError(f"{label}: rotation is not SO(3)")
    return pose


def percentiles(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "p50_px": None if not len(finite) else float(np.median(finite)),
        "p90_px": None if not len(finite) else float(np.percentile(finite, 90)),
        "p95_px": None if not len(finite) else float(np.percentile(finite, 95)),
        "within4px_count": int(np.count_nonzero(finite <= MAX_REPROJ_PX)),
    }


def coverage(xy):
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if not len(xy):
        return {"hull_fraction": 0.0, "grid_cells_4x4": 0, "bbox_fraction": 0.0}
    cells = np.clip((xy / np.asarray(SIZE) * 4).astype(int), 0, 3)
    hull = cv2.convexHull(xy) if len(xy) >= 3 else None
    return {
        "hull_fraction": 0.0 if hull is None else float(cv2.contourArea(hull) / np.prod(SIZE)),
        "grid_cells_4x4": int(len(np.unique(cells, axis=0))),
        "bbox_fraction": float(np.prod(np.ptp(xy, axis=0)) / np.prod(SIZE)),
    }


def project(xyz, pose, K):
    camera = xyz @ pose[:, :3].T + pose[:, 3]
    positive = camera[:, 2] > 1e-8
    homogeneous = camera @ K.T
    xy = homogeneous[:, :2] / np.where(positive, homogeneous[:, 2], 1.0)[:, None]
    return xy, positive


def residuals(xyz, xy, pose, K):
    projected, positive = project(xyz, pose, K)
    error = np.linalg.norm(projected - xy, axis=1)
    error[~positive] = np.inf
    return error, positive


def inside(xy, margin=0):
    return (
        (xy[:, 0] >= margin) & (xy[:, 0] < SIZE[0] - margin)
        & (xy[:, 1] >= margin) & (xy[:, 1] < SIZE[1] - margin)
    )


def solve_pnp(ids, xyz, xy, K, *, minimum=LOCAL_MIN_INLIERS, seed=0, max_error_px=MAX_REPROJ_PX):
    """No pose guess: RANSAC EPNP followed by inlier-only LM refinement."""
    n = len(ids)
    result = {
        "ok": False, "rejection_reasons": [], "input_count": n,
        "inlier_count": 0, "point_ids": ids.tolist(), "image_xy_px": xy.tolist(),
        "world_xyz_u": xyz.tolist(), "inlier_point_ids": [], "inlier_indices": [],
        "cam_from_world": None, "camera_center": None,
        "solver": "opencv_EPNP_RANSAC+inlier_LM; no_extrinsic_guess",
        "max_error_px": max_error_px, "minimum_inliers": minimum,
        "coverage": coverage(xy),
    }
    if n < max(6, minimum):
        result["rejection_reasons"].append("insufficient_correspondences")
        return result
    cv2.setRNGSeed(int(seed) & 0x7fffffff)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(xyz, dtype=np.float64),
            np.ascontiguousarray(xy, dtype=np.float64), K, None,
            iterationsCount=2000, reprojectionError=max_error_px, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 6:
            result["rejection_reasons"].append("ransac_no_pose")
            return result
        selected = inliers.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(xyz[selected], xy[selected], K, None, rvec, tvec)
    except cv2.error as exc:
        result["rejection_reasons"].append("opencv_pnp_failure")
        result["solver_error"] = str(exc)
        return result
    pose = np.column_stack((cv2.Rodrigues(rvec)[0], tvec.reshape(3)))
    if not np.isfinite(pose).all():
        result["rejection_reasons"].append("nonfinite_pnp_pose")
        return result
    error, positive = residuals(xyz, xy, pose, K)
    mask = positive & (error <= max_error_px)
    positive_ratio = float(np.mean(positive))
    ratio = float(np.mean(mask))
    result.update({
        "cam_from_world": pose.tolist(), "camera_center": (-pose[:, :3].T @ pose[:, 3]).tolist(),
        "inlier_count": int(mask.sum()), "inlier_ratio": ratio,
        "inlier_indices": np.flatnonzero(mask).tolist(), "inlier_point_ids": ids[mask].tolist(),
        "positive_depth_ratio": positive_ratio, "inlier_positive_depth_ratio": 1.0 if mask.any() else None,
        "reprojection_all": percentiles(error), "reprojection_inliers": percentiles(error[mask]),
        "errors_px": [float(e) if math.isfinite(e) else None for e in error],
        "inlier_coverage": coverage(xy[mask]),
    })
    if int(mask.sum()) < minimum:
        result["rejection_reasons"].append("insufficient_pnp_inliers")
    if ratio < 0.25:
        result["rejection_reasons"].append("low_pnp_inlier_ratio")
    if positive_ratio < 0.99:
        result["rejection_reasons"].append("cheirality")
    result["ok"] = not result["rejection_reasons"]
    return result


def heldout_pnp(ids, xyz, xy, K, modulus, seed):
    if not modulus:
        return {"enabled": False}
    # Split by identity, unchanged across captures/references/methods. Never select
    # held-out rows by a fitted residual or use their pixels to initialize PnP.
    held = np.asarray([int(hashlib.sha256(str(int(key)).encode()).hexdigest()[:8], 16) % modulus == 0
                       for key in ids], dtype=bool)
    fit = solve_pnp(ids[~held], xyz[~held], xy[~held], K, minimum=12, seed=seed)
    report = {"enabled": True, "split": f"sha256(point_id)[:8] modulo {modulus} == 0",
              "train_count": int((~held).sum()), "heldout_point_ids": ids[held].tolist(),
              "heldout_count": int(held.sum()), "train_pnp": fit,
              "heldout_reprojection": None, "heldout_positive_depth_ratio": None,
              "diagnostic_only": True}
    if fit["ok"] and held.any():
        error, positive = residuals(xyz[held], xy[held], np.asarray(fit["cam_from_world"]), K)
        report["heldout_reprojection"] = percentiles(error)
        report["heldout_positive_depth_ratio"] = float(np.mean(positive))
    return report


def parse_ordinals(values):
    result = []
    for token in values:
        for item in token.split(","):
            parts = item.split(":")
            if len(parts) == 1:
                result.append(int(item))
            elif len(parts) in (2, 3):
                start, stop = int(parts[0]), int(parts[1])
                step = int(parts[2]) if len(parts) == 3 else 1
                if start > stop or step <= 0:
                    raise ExperimentError(f"invalid inclusive ordinal range: {item}")
                result.extend(range(start, stop + 1, step))
            else:
                raise ExperimentError(f"invalid ordinal: {item}")
    if not result or any(n < 0 for n in result) or len(set(result)) != len(result):
        raise ExperimentError("ordinals must be nonnegative, nonempty, and unique")
    return sorted(result)


def load_image(path, *, reference=False):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ExperimentError(f"image is not a readable uint8 gray/BGR image: {path}")
    original_size = (int(image.shape[1]), int(image.shape[0]))
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ExperimentError(f"unsupported channel count: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if original_size != SIZE:
        if reference or not np.isclose(original_size[0] / original_size[1], SIZE[0] / SIZE[1]):
            raise ExperimentError(f"wrong image coordinate frame: {path}: {original_size}")
        image = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image), {"path": str(path), "sha256": digest(path),
                                        "original_size": list(original_size), "image_size_px": list(SIZE)}


def keyframe_for_ordinal(keyframes, ordinal, epoch):
    matches = [row for row in keyframes.values()
               if row["ordinal"] == ordinal and row["source_epoch"] == epoch]
    if len(matches) > 1:
        raise ExperimentError(f"ambiguous saved capture: ordinal={ordinal}, source_epoch={epoch}")
    return matches[0] if matches else None


def query_path(directory, frame):
    if directory is None:
        if frame is None:
            raise ExperimentError("query has no saved keyframe; supply --query-images-dir with exact decoded captures")
        return required_file(frame["image_path"])
    ordinal = frame["ordinal"]
    paths = list(dict.fromkeys([directory / f"{ordinal}.png", directory / f"{ordinal:06d}.png"]))
    found = [path for path in paths if path.is_file()]
    if not found:
        raise ExperimentError(f"missing query image: {paths}")
    if len(found) == 2 and digest(found[0]) != digest(found[1]):
        raise ExperimentError(f"ambiguous query image filenames for ordinal {ordinal}")
    return found[0].resolve()


def camera_matrix(protocol, profile):
    camera = protocol["query_camera"]
    if camera["model"] != "PINHOLE" or len(camera["params"]) != 4:
        raise ExperimentError("this experiment requires the protocol PINHOLE camera, without distortion")
    if tuple(profile["fast_loop"]["resolution"]) != SIZE:
        raise ExperimentError("protocol profile does not use the expected 960x540 fast-loop camera")
    fx, fy, cx, cy = camera["params"]
    sx, sy = SIZE[0] / camera["width"], SIZE[1] / camera["height"]
    K = finite_array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], (3, 3), "camera K")
    if min(K[0, 0], K[1, 1]) <= 0:
        raise ExperimentError("camera focal length must be positive")
    return K


def check_camera(row, K, *, saved=False):
    size_key = "image_size" if saved else "image_size_px"
    if tuple(row[size_key]) != SIZE or not np.allclose(
        finite_array(row["K"], (3, 3), "observation K"), K, rtol=0, atol=1e-7
    ):
        raise ExperimentError("observation camera disagrees with resized protocol query_camera")


def observation_arrays(row, *, saved=False):
    keys = ("ba_ids", "ba_xy", "ba_xyz") if saved else ("ids", "image_xy_px", "world_xyz_u")
    ids = integer_ids(row[keys[0]], "observation")
    xy = finite_array(row[keys[1]], (len(ids), 2), "observation pixels")
    xyz = finite_array(row[keys[2]], (len(ids), 3), "observation XYZ")
    if not inside(xy).all():
        raise ExperimentError("observed pixels outside 960x540 image")
    return ids, xy, xyz


def _landmark_rejections(point, key, expected_xyz, snapshot, frame, submap, revisions):
    if point is None:
        return ["missing_snapshot_landmark"]
    rejection = []
    if point["submap_id"] != submap["id"]:
        rejection.append("wrong_landmark_owner")
    if revisions.get(str(int(key))) != point["revision"]:
        rejection.append("coordinate_revision_mismatch")
    if not np.allclose(finite_array(point["xyz"], (3,), "snapshot landmark XYZ"), expected_xyz,
                       rtol=1e-10, atol=1e-10):
        rejection.append("snapshot_ba_xyz_mismatch")
    if point["revision"] > snapshot["revision"] or point["revision"] > frame["revision"]:
        rejection.append("coordinate_revision_from_future")
    return rejection


def _reference_point_record(key, point, xy_row, xyz_row, revisions, saved_error, saved_positive,
                            rejection):
    return {"point_id": int(key), "observed_xy_px": xy_row.tolist(),
            "world_xyz_u": xyz_row.tolist(),
            "landmark_revision": None if point is None else point["revision"],
            "reference_landmark_revision": revisions.get(str(int(key))),
            "owner_submap_id": None if point is None else point["submap_id"],
            "saved_reprojection_px": float(saved_error) if saved_positive else None,
            "saved_positive_depth": bool(saved_positive),
            "rejection_reasons": rejection}


def _annotate_validation_residuals(point_records, coordinate_ok, geometry_ok, validation_error,
                                   validation_positive, reasons):
    for index, row in enumerate(point_records):
        row["validation_reprojection_px"] = float(validation_error[index]) if validation_positive[index] else None
        row["geometry_eligible"] = bool(geometry_ok[index])
        if coordinate_ok[index] and not geometry_ok[index]:
            row["rejection_reasons"].append("reference_reprojection_inconsistent")
            reasons["reference_reprojection_inconsistent"] += 1


def prepare_reference(frame, snapshot, submap, K, seed):
    check_camera(frame, K, saved=True)
    ids, xy, xyz = observation_arrays(frame, saved=True)
    child = ids < 0
    ids, xy, xyz = ids[child], xy[child], xyz[child]
    if not len(ids):
        raise ExperimentError(f"reference {frame['ordinal']} has no child landmark observations")
    if not frame["durable"] or frame["durable_revision"] != frame["revision"]:
        raise ExperimentError(f"reference {frame['ordinal']} has no durable current coordinate revision")
    pose = pose_array(frame["pose"], "saved reference pose")
    image, image_info = load_image(required_file(frame["image_path"]), reference=True)
    coordinate_ok = np.ones(len(ids), dtype=bool)
    point_records = []
    reasons = Counter()
    saved_error, saved_positive = residuals(xyz, xy, pose, K)
    revisions = frame["landmark_revisions"]
    for index, key in enumerate(ids):
        point = snapshot["landmarks"].get(str(int(key)))
        rejection = _landmark_rejections(point, key, xyz[index], snapshot, frame, submap, revisions)
        coordinate_ok[index] = not rejection
        reasons.update(rejection)
        point_records.append(_reference_point_record(
            key, point, xy[index], xyz[index], revisions, saved_error[index], saved_positive[index],
            rejection))
    diagnostic = solve_pnp(ids[coordinate_ok], xyz[coordinate_ok], xy[coordinate_ok], K, seed=seed)
    saved_good = coordinate_ok & saved_positive & (saved_error <= MAX_REPROJ_PX)
    inconsistent = (not coordinate_ok.any() or not np.all(saved_good[coordinate_ok]))
    validation_pose = pose
    validation_source = "saved_reference_pose"
    # A source inconsistency is made explicit before association, not silently
    # interpreted as evidence that descriptors could not cross the lost interval.
    if inconsistent:
        progress("saved_reference_geometry_inconsistent", ordinal=frame["ordinal"],
                 owned_revision_consistent=int(coordinate_ok.sum()),
                 saved_within4px=int(saved_good.sum()), reprojection=percentiles(saved_error),
                 self_pnp_inliers=diagnostic["inlier_count"], self_pnp_ok=diagnostic["ok"])
        if diagnostic["ok"]:
            validation_pose = np.asarray(diagnostic["cam_from_world"])
            validation_source = "independent_fixed_child_xyz_reference_self_pnp"
    validation_error, validation_positive = residuals(xyz, xy, validation_pose, K)
    geometry_ok = coordinate_ok & validation_positive & (validation_error <= MAX_REPROJ_PX)
    _annotate_validation_residuals(point_records, coordinate_ok, geometry_ok, validation_error,
                                   validation_positive, reasons)
    report = {"ordinal": frame["ordinal"], "name": frame["name"], "submap_id": frame["submap_id"],
              "source_epoch": frame["source_epoch"], "capture_stamp_mono": frame["stamp"],
              "revision": frame["revision"], "pose_revision": frame["pose_revision"],
              "durable_revision": frame["durable_revision"], "snapshot_revision": snapshot["revision"],
              "image": image_info, "child_observation_count": len(ids),
              "saved_cam_from_world": pose.tolist(), "saved_reprojection": percentiles(saved_error),
              "saved_positive_depth_ratio": float(np.mean(saved_positive)),
              "initial_geometry_inconsistent": inconsistent, "self_pnp_diagnostic": diagnostic,
              "validation_pose_source": validation_source, "validation_cam_from_world": validation_pose.tolist(),
              "source_geometry_mutated": False, "geometry_eligible_count": int(geometry_ok.sum()),
              "rejection_counts": dict(reasons), "points": point_records}
    return {"frame": frame, "ids": ids, "xy": xy, "xyz": xyz, "image": image,
            "coordinate_ok": coordinate_ok, "geometry_ok": geometry_ok,
            "validation_pose": validation_pose, "tree": cKDTree(xy), "report": report}


class SiftMatcher:
    def __init__(self, references):
        self.detector = cv2.SIFT_create(nfeatures=8000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.references = {ref["frame"]["ordinal"]: self.features(ref["image"]) for ref in references}
        self.metadata = {"implementation": "opencv_CPU_SIFT", "max_features": 8000,
                         "ratio": 0.75, "mutual_ratio_required": True, "reference_features_cached": True}

    def features(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return np.asarray([key.pt for key in keypoints], dtype=float).reshape(-1, 2), descriptors

    def prepare_query(self, image):
        return self.features(image)

    def match(self, query, reference):
        qxy, qdesc = query
        rxy, rdesc = self.references[reference["frame"]["ordinal"]]
        if qdesc is None or rdesc is None or min(len(qdesc), len(rdesc)) < 2:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0), {"query_features": len(qxy), "reference_features": len(rxy)}
        forward = self.matcher.knnMatch(rdesc, qdesc, k=2)
        backward = self.matcher.knnMatch(qdesc, rdesc, k=2)
        reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward
                   if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}
        matches = [pair[0] for pair in forward if len(pair) == 2
                   and pair[0].distance < 0.75 * pair[1].distance
                   and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
        ri = np.asarray([m.queryIdx for m in matches], dtype=int)
        qi = np.asarray([m.trainIdx for m in matches], dtype=int)
        scores = np.asarray([1.0 / (1.0 + m.distance) for m in matches])
        return rxy[ri], qxy[qi], scores, {"query_features": len(qxy), "reference_features": len(rxy)}


def geometric_matches(reference_xy, query_xy, scores, seed):
    counts = {"raw_matches": len(scores)}
    valid = np.isfinite(reference_xy).all(axis=1) & np.isfinite(query_xy).all(axis=1) & np.isfinite(scores)
    valid &= inside(reference_xy) & inside(query_xy) & (scores > 0)
    indices = np.flatnonzero(valid)
    reference_xy, query_xy, scores = reference_xy[valid], query_xy[valid], scores[valid]
    counts["finite_inside_matches"] = len(scores)
    if len(scores) < 8:
        counts["two_view_rejection"] = "fewer_than_eight_matches"
        return reference_xy[:0], query_xy[:0], scores[:0], indices[:0], counts
    cv2.setRNGSeed(int(seed) & 0x7fffffff)
    matrix, mask = cv2.findFundamentalMat(reference_xy, query_xy, cv2.FM_RANSAC, 2.0, 0.999, 10000)
    if matrix is None or matrix.shape != (3, 3) or mask is None or not np.isfinite(matrix).all():
        counts["two_view_rejection"] = "fundamental_ransac_failed"
        return reference_xy[:0], query_xy[:0], scores[:0], indices[:0], counts
    keep = mask.ravel().astype(bool)
    counts["two_view_inliers"] = int(keep.sum())
    counts["fundamental_matrix"] = matrix.tolist()
    return reference_xy[keep], query_xy[keep], scores[keep], indices[keep], counts


def nearest_seeds(ref, rxy, qxy, scores, match_indices):
    seeds, rejected = [], Counter()
    if not len(rxy):
        return seeds, rejected
    distances, neighbors = ref["tree"].query(rxy, k=2)
    for index, (d, nearest) in enumerate(zip(distances, neighbors, strict=True)):
        if d[0] > NEAREST_PX:
            rejected["source_nearest_too_far"] += 1
            continue
        if d[0] >= NEAREST_RATIO * d[1] or d[1] - d[0] < NEAREST_MARGIN_PX:
            rejected["ambiguous_source_landmark"] += 1
            continue
        row = int(nearest[0])
        seeds.append({"row": row, "source_xy": ref["xy"][row].tolist(),
                      "query_seed_xy": (qxy[index] + ref["xy"][row] - rxy[index]).tolist(),
                      "source_match_xy": rxy[index].tolist(), "query_match_xy": qxy[index].tolist(),
                      "source_distance_px": float(d[0]), "second_source_distance_px": float(d[1]) if math.isfinite(d[1]) else None,
                      "match_index": int(match_indices[index]), "match_score": float(scores[index]),
                      "association": "nearest"})
    return seeds, rejected


def guided_seeds(ref, rxy, qxy, scores, match_indices):
    """Interpolate image matches only; no reference/query/main-map pose enters."""
    seeds, rejected = [], Counter()
    if len(rxy) < 4:
        rejected["insufficient_local_warp_donors"] = len(ref["ids"])
        return seeds, rejected
    tree = cKDTree(rxy)
    distances, neighbors = tree.query(ref["xy"], k=min(8, len(rxy)))
    for row, (distance, neighbor) in enumerate(zip(distances, neighbors, strict=True)):
        donor = neighbor[distance <= GUIDED_RADIUS_PX]
        if len(donor) < 4:
            rejected["insufficient_local_warp_donors"] += 1
            continue
        source = rxy[donor]
        if len(np.unique(np.round(source, 2), axis=0)) < 4:
            rejected["duplicate_local_warp_donors"] += 1
            continue
        hull = cv2.convexHull(source.astype(np.float32))
        if cv2.pointPolygonTest(hull, tuple(map(float, ref["xy"][row])), False) < 0:
            rejected["local_warp_would_extrapolate"] += 1
            continue
        design = np.column_stack((source - ref["xy"][row], np.ones(len(source))))
        transform, _, rank, _ = np.linalg.lstsq(design, qxy[donor], rcond=None)
        singular = np.linalg.svd(transform[:2], compute_uv=False)
        fit_errors = np.linalg.norm(design @ transform - qxy[donor], axis=1)
        if rank != 3 or singular[-1] < 0.5 or singular[0] > 2.0 or fit_errors.max() > 2.0:
            rejected["inconsistent_local_image_warp"] += 1
            continue
        seeds.append({"row": row, "source_xy": ref["xy"][row].tolist(),
                      "query_seed_xy": transform[2].tolist(), "association": "guided",
                      "donor_match_indices": match_indices[donor].tolist(),
                      "donor_reference_xy": source.tolist(), "donor_query_xy": qxy[donor].tolist(),
                      "donor_scores": scores[donor].tolist(),
                      "warp_max_residual_px": float(fit_errors.max()),
                      "warp_singular_values": singular.tolist()})
    return seeds, rejected


def patch_evidence(reference_image, query_image, source, target):
    side = 2 * PATCH_RADIUS + 1
    template = cv2.getRectSubPix(reference_image, (side, side), tuple(map(float, source)))
    search = cv2.getRectSubPix(query_image, (side + 2 * PATCH_SEARCH_PX,) * 2, tuple(map(float, target)))
    if float(np.std(template)) < 5.0:
        return {"ok": False, "reason": "low_texture_source_patch"}
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(scores).all():
        return {"ok": False, "reason": "nonfinite_patch_scores"}
    y, x = np.unravel_index(np.argmax(scores), scores.shape)
    yy, xx = np.indices(scores.shape)
    alternate = scores[(xx - x) ** 2 + (yy - y) ** 2 > 4]
    margin = float(scores[y, x] - np.max(alternate))
    ncc = float(scores[PATCH_SEARCH_PX, PATCH_SEARCH_PX])
    peak_offset = math.hypot(x - PATCH_SEARCH_PX, y - PATCH_SEARCH_PX)
    reason = None
    if ncc < PATCH_MIN_NCC:
        reason = "patch_ncc"
    elif peak_offset > 1.5:
        reason = "patch_peak_disagrees_with_flow"
    elif margin < PATCH_PEAK_MARGIN:
        reason = "ambiguous_patch_peak"
    return {"ok": reason is None, "reason": reason, "ncc": ncc,
            "unique_peak_margin": margin, "peak_offset_px": peak_offset}


def refine_seeds(ref, query_image, seeds):
    rejected = Counter()
    if not seeds:
        return [], rejected
    source = np.asarray([seed["source_xy"] for seed in seeds], dtype=np.float32).reshape(-1, 1, 2)
    initial = np.asarray([seed["query_seed_xy"] for seed in seeds], dtype=np.float32).reshape(-1, 1, 2)
    # Bounded residual motion, not long-gap uninitialized KLT. Match-derived
    # initial flow is mandatory, and both directions must return the same point.
    options = {"winSize": (21, 21), "maxLevel": 1,
               "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
               "flags": cv2.OPTFLOW_USE_INITIAL_FLOW, "minEigThreshold": 1e-4}
    target, forward, _ = cv2.calcOpticalFlowPyrLK(ref["image"], query_image, source, initial.copy(), **options)
    if target is None or forward is None:
        rejected["patch_flow_failed"] = len(seeds)
        return [], rejected
    backward_target = target.copy()
    bad = ~np.isfinite(backward_target).all(axis=(1, 2))
    backward_target[bad] = initial[bad]
    back, backward, _ = cv2.calcOpticalFlowPyrLK(query_image, ref["image"], backward_target, source.copy(), **options)
    if back is None or backward is None:
        rejected["patch_backward_flow_failed"] = len(seeds)
        return [], rejected
    fb = np.linalg.norm(back[:, 0] - source[:, 0], axis=1)
    shift = np.linalg.norm(target[:, 0] - initial[:, 0], axis=1)
    valid = forward.ravel().astype(bool) & backward.ravel().astype(bool)
    valid &= np.isfinite(target).all(axis=(1, 2)) & np.isfinite(fb) & np.isfinite(shift)
    valid &= inside(source[:, 0], PATCH_RADIUS + 1) & inside(target[:, 0], PATCH_RADIUS + PATCH_SEARCH_PX + 1)
    accepted = []
    for index, seed in enumerate(seeds):
        if not valid[index] or fb[index] > PATCH_MAX_FB_PX or shift[index] > PATCH_MAX_SHIFT_PX:
            rejected["patch_bounds_or_bidirectional_flow"] += 1
            continue
        patch = patch_evidence(ref["image"], query_image, source[index, 0], target[index, 0])
        if not patch["ok"]:
            rejected[patch["reason"]] += 1
            continue
        row = seed["row"]
        candidate = {**seed, "point_id": int(ref["ids"][row]), "query_xy": target[index, 0].astype(float).tolist(),
                     "world_xyz_u": ref["xyz"][row].tolist(), "reference_ordinal": ref["frame"]["ordinal"],
                     "reference_revision": ref["frame"]["revision"], "fb_error_px": float(fb[index]),
                     "refinement_shift_px": float(shift[index]), "patch": patch,
                     "source_geometry_eligible": bool(ref["geometry_ok"][row])}
        accepted.append(candidate)
    return accepted, rejected


def unique_reference_candidates(candidates):
    groups = defaultdict(list)
    for row in candidates:
        groups[row["point_id"]].append(row)
    accepted, rejected = [], Counter()
    for rows in groups.values():
        xy = np.asarray([row["query_xy"] for row in rows])
        if len(xy) > 1 and np.max(np.linalg.norm(xy[:, None] - xy[None, :], axis=2)) > QUERY_AGREEMENT_PX:
            rejected["same_id_conflicting_query_matches"] += len(rows)
            continue
        if not rows[0]["source_geometry_eligible"]:
            rejected["reference_geometry_inconsistent"] += len(rows)
            continue
        accepted.append(max(rows, key=lambda row: row["patch"]["ncc"]))
    return accepted, rejected


def aggregate_candidates(candidates, eligible_covisibility):
    groups = defaultdict(list)
    for row in candidates:
        groups[row["point_id"]].append(row)
    records, rejected = [], Counter()
    for key, rows in sorted(groups.items()):
        support = sorted({row["reference_ordinal"] for row in rows})
        if len(support) < MIN_REFERENCE_SUPPORT or len(eligible_covisibility[key]) < MIN_REFERENCE_SUPPORT:
            rejected["insufficient_covisible_match_support"] += 1
            continue
        xy = np.asarray([row["query_xy"] for row in rows])
        dispersion = float(np.max(np.linalg.norm(xy[:, None] - xy[None, :], axis=2)))
        if dispersion > QUERY_AGREEMENT_PX:
            rejected["covisible_query_pixel_disagreement"] += 1
            continue
        # Choose an actually measured refined pixel, not an averaged/fabricated
        # pixel. All supported references must agree; no winning-cluster pruning.
        best = max(rows, key=lambda row: row["patch"]["ncc"])
        records.append({"point_id": key, "image_xy_px": best["query_xy"],
                        "world_xyz_u": best["world_xyz_u"], "reference_ordinals": support,
                        "reference_revisions": {str(row["reference_ordinal"]): row["reference_revision"] for row in rows},
                        "query_dispersion_px": dispersion,
                        "chosen_reference_ordinal": best["reference_ordinal"],
                        "association": best["association"]})
    if records:
        xy = np.asarray([row["image_xy_px"] for row in records])
        conflicts = cKDTree(xy).query_pairs(QUERY_AGREEMENT_PX)
        ambiguous = {index for pair in conflicts for index in pair}
        rejected["different_ids_same_query_pixel"] += len(ambiguous)
        records = [row for index, row in enumerate(records) if index not in ambiguous]
    return records, dict(rejected)


def rows_arrays(rows):
    return (np.asarray([row["point_id"] for row in rows], dtype=np.int64),
            np.asarray([row["world_xyz_u"] for row in rows], dtype=float).reshape(-1, 3),
            np.asarray([row["image_xy_px"] for row in rows], dtype=float).reshape(-1, 2))


class EdmMatcher:
    def __init__(self, references, device):
        # Use the audited loader and already-present configs; do not call the
        # provider config generator, create a map provider, or download weights.
        sys.path.insert(0, str(DEPLOY))
        from direct_paths import vendor_sys_path
        vendor_sys_path()
        from sfm_diagnosis.site_pipeline.deployment_localizer import (
            _configure_audited_runtime_site_packages, _resolve_audited_site_packages,
        )
        _configure_audited_runtime_site_packages(_resolve_audited_site_packages(None))
        from river_map_quality.official_edm_adapter_loo import load_official_edm_runtime
        from river_map_quality.official_edm_adapter import prepare_official_megadepth_image_from_array
        from edm_inference import CachedEDM

        paths = edm_paths()
        progress("edm_load", device=device)
        runtime = load_official_edm_runtime(
            edm_repo=paths["repo"], checkpoint=paths["checkpoint"],
            config_path=paths["config"], data_config_path=paths["data_config"], device=device,
        )
        if (runtime.input_width, runtime.input_height, runtime.coarse_topk) != (640, 640, 2240):
            raise ExperimentError("EDM runtime is not the frozen 640x640 / TOPK2240 configuration")
        self.runtime = runtime
        self.prepare = prepare_official_megadepth_image_from_array
        self.cache = CachedEDM(runtime, capacity=len(references))
        self.references = {ref["frame"]["ordinal"]: self.prepare(ref["image"], runtime) for ref in references}
        self.metadata = {
            "implementation": "audited_official_EDM+CachedEDM", "device": device,
            "input_size": [640, 640], "coarse_topk": 2240,
            "match_threshold": runtime.match_threshold, "native_pixel_outputs": True,
            "reference_features_cached": True, "model_load_count": 1,
            "assets": {key: {"path": str(path), "sha256": digest(path)}
                       for key, path in paths.items() if key != "repo"},
        }

    def prepare_query(self, image):
        return self.prepare(image, self.runtime)

    def match(self, query, reference):
        result = self.cache.match(query, self.references[reference["frame"]["ordinal"]])
        # EDM fine_matching has already applied scale0/scale1. No second scale.
        qxy = np.asarray(result["mkpts0_f"], dtype=float).reshape(-1, 2)
        rxy = np.asarray(result["mkpts1_f"], dtype=float).reshape(-1, 2)
        scores = np.asarray(result["mconf"], dtype=float).reshape(-1)
        if qxy.shape != rxy.shape or len(qxy) != len(scores):
            raise ExperimentError("EDM returned inconsistent correspondence arrays")
        return rxy, qxy, scores, {"backbone_cache_hits": self.cache.backbone_hits}


def edm_paths():
    repo = DEPLOY.parent / "runtime/EDM"
    return {
        "repo": repo,
        "checkpoint": required_file(repo / "weights/edm_outdoor.ckpt"),
        "config": required_file(ROOT / "執行環境/direct_reloc_cache/edm_runtime_config/edm_base_640.py"),
        "data_config": required_file(ROOT / "執行環境/direct_reloc_cache/edm_runtime_config/megadepth_640.py"),
    }


def frozen_geometry(bundle_path, bundle):
    entries = [entry for entry in bundle["files"] if entry["path"] == "direct_geometry.npz"]
    if len(entries) != 1:
        raise ExperimentError("a bundle-verified direct_geometry.npz is required to check positive IDs/XYZ")
    entry = entries[0]
    path = required_file(bundle_path.parent / entry["path"])
    actual = digest(path)
    if actual != entry["sha256"]:
        raise ExperimentError("frozen geometry archive SHA256 does not match the protocol bundle")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()), parse_constant=no_constant)
        if metadata != {"schema": "direct-geometry/v1", "model_sha256": bundle["model_sha256"]}:
            raise ExperimentError("frozen geometry archive model/schema mismatch")
        ids = archive["point_ids"]
        xyz = archive["point_xyz"]
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all()
            or not len(ids) or np.any(ids <= 0) or np.any(np.diff(ids) <= 0)):
        raise ExperimentError("frozen geometry must have sorted unique positive IDs and finite XYZ")
    return ids, xyz, {"path": str(path), "sha256": actual, "model_sha256": bundle["model_sha256"]}


def verify_frozen_points(ids, xyz, frozen_ids, frozen_xyz):
    indices = np.searchsorted(frozen_ids, ids)
    if np.any(indices >= len(frozen_ids)) or not np.array_equal(frozen_ids[indices], ids):
        raise ExperimentError("positive map observation ID is absent from the immutable main map")
    if not np.allclose(frozen_xyz[indices], xyz, rtol=1e-10, atol=1e-10):
        raise ExperimentError("positive map observation XYZ disagrees with immutable main-map coordinates")


def capture_identity(row, snapshot):
    response = row["response"]
    stamp = float(response["frame_stamp_mono"])
    if not math.isfinite(stamp) or abs(stamp - int(response["capture_mono_ns"]) * 1e-9) > 1e-6:
        raise ExperimentError(f"frame {row['ordinal']} has inconsistent capture timestamps")
    if response.get("workmap_session_id") != snapshot["session_id"]:
        raise ExperimentError("frames and snapshot belong to different sessions")
    # This replay records no separate foreground epoch field. Require the
    # recorded relocalizer epoch to corroborate the session epoch; do not
    # manufacture capture identity across a stream reset or missing metadata.
    if response.get("reloc_source_epoch") != snapshot["source_epoch"]:
        raise ExperimentError("capture source epoch is missing or disagrees with the saved session")
    return {"ordinal": row["ordinal"], "source_epoch": snapshot["source_epoch"],
            "source_epoch_evidence": "snapshot corroborated by recorded reloc_source_epoch",
            "capture_stamp_mono": stamp, "capture_mono_ns": response["capture_mono_ns"],
            "video_time_s": row["video_time_s"], "run_id": row["run_id"],
            "sequence": row["sequence"], "repeat": row["repeat"]}


def load_external_observations(path, run_id, snapshot, captures, K):
    if path is None:
        return {}, None
    path = required_file(path)
    document = load_json(path)
    if (document.get("schema") != "workmap-closure-map-observations/v1"
            or document.get("run_id") != run_id
            or document.get("coordinate_frame_id") != snapshot["coordinate_frame_id"]):
        raise ExperimentError("external map observations schema/run/coordinate identity mismatch")
    records = {}
    for row in document["records"]:
        ordinal = row["ordinal"]
        if ordinal in records:
            raise ExperimentError(f"duplicate external map-observation capture: {ordinal}")
        records[ordinal] = row
        if ordinal not in captures:
            continue
        identity = captures[ordinal]
        if (row["source_epoch"] != identity["source_epoch"]
                or not math.isfinite(float(row["capture_stamp_mono"]))
                or abs(row["capture_stamp_mono"] - identity["capture_stamp_mono"]) > 1e-6):
            raise ExperimentError(f"external map observations do not match capture {ordinal}")
        check_camera(row, K)
        ids, _, _ = observation_arrays(row)
        if np.any(ids <= 0):
            raise ExperimentError("external main-map observations must contain only positive IDs")
        if not isinstance(row.get("provenance"), dict) or not row["provenance"]:
            raise ExperimentError("external observations require explicit measurement provenance")
    return records, {"path": str(path), "sha256": digest(path)}


def apply_frozen_gates(result, thresholds):
    if result["cam_from_world"] is None:
        return result
    reasons = result["rejection_reasons"]
    if result["inlier_count"] < int(thresholds["strong_inliers"]):
        reasons.append("frozen_strong_inliers")
    if result["inlier_ratio"] < thresholds["minimum_inlier_ratio"]:
        reasons.append("frozen_inlier_ratio")
    if result["positive_depth_ratio"] < thresholds["minimum_positive_depth_ratio"]:
        reasons.append("frozen_positive_depth")
    if result["inlier_coverage"]["hull_fraction"] < thresholds["minimum_hull_coverage"]:
        reasons.append("frozen_pixel_hull")
    if result["inlier_coverage"]["grid_cells_4x4"] < thresholds["minimum_occupancy_4x4"]:
        reasons.append("frozen_pixel_grid")
    p90 = result["reprojection_inliers"]["p90_px"]
    if p90 is None or p90 > thresholds["maximum_reprojection_p90_px"]:
        reasons.append("frozen_reprojection_p90")
    result["ok"] = not reasons
    result["frozen_quality_gates"] = dict(thresholds)
    return result


def map_observation_sources(query, external):
    ordinal = query["identity"]["ordinal"]
    sources = []
    if ordinal in external:
        sources.append(("external_positive_map_measurements", external[ordinal], False))
    sample = query["record"]["response"].get("pnp_observation_sample")
    if sample is not None:
        if abs(float(sample["capture_stamp_mono"]) - query["identity"]["capture_stamp_mono"]) > 1e-6:
            raise ExperimentError(f"PnP observation sample is not capture {ordinal}")
        sources.append(("recorded_sample_positive_subset", sample, False))
    saved = query["saved_keyframe"]
    if saved is not None:
        if abs(float(saved["stamp"]) - query["identity"]["capture_stamp_mono"]) > 1e-6:
            raise ExperimentError(f"saved map observations are not capture {ordinal}")
        sources.append(("saved_keyframe_positive_KLT_measurements", saved, True))
    return sources


def prepare_map_pose(query, external, K, thresholds, frozen_ids, frozen_xyz, seed):
    attempts = []
    for source, observations, saved in map_observation_sources(query, external):
        check_camera(observations, K, saved=saved)
        ids, xy, xyz = observation_arrays(observations, saved=saved)
        positive = ids > 0
        ignored_negative = int(np.count_nonzero(ids < 0))
        ids, xy, xyz = ids[positive], xy[positive], xyz[positive]
        verify_frozen_points(ids, xyz, frozen_ids, frozen_xyz)
        result = solve_pnp(
            ids, xyz, xy, K, minimum=int(thresholds["strong_inliers"]), seed=seed,
            max_error_px=float(thresholds["maximum_reprojection_p90_px"]),
        )
        result = apply_frozen_gates(result, thresholds)
        result["observation_source"] = source
        result["ignored_negative_count"] = ignored_negative
        result["negative_xyz_used"] = False
        result["independence"] = "fresh_positive_only_PnP; no_published_pose_initialization"
        result["provenance"] = observations.get("provenance", {
            "capture_stamp_mono": observations["stamp"] if saved else observations["capture_stamp_mono"],
            "saved_keyframe_revision": observations.get("revision") if saved else None,
            "sampling": observations.get("sampling"),
        })
        attempts.append(result)
    successful = next((result for result in attempts if result["ok"]), None)
    response = query["record"]["response"]
    return {"ok": successful is not None, "selected": successful, "attempts": attempts,
            "rejection_reasons": [] if successful else ["no_independent_strong_positive_only_map_pnp"],
            "recorded_pose_basis": response.get("workmap_pose_basis"),
            "recorded_pose_used": False, "recorded_direct_status": response.get("direct_status"),
            "recorded_map_inliers": response.get("map_inliers"),
            "recorded_pose_capture_mono_ns": response.get("pose_capture_mono_ns"),
            "capture_identity": query["identity"]}


def _check_snapshot_identity(snapshot, protocol, bundle):
    if snapshot.get("schema") != "direct-online-workmap-session/v1":
        raise ExperimentError("unsupported workmap snapshot schema")
    if (snapshot["base_bundle_sha256"] != protocol["bundle"]["sha256"]
            or snapshot["coordinate_frame_id"] != bundle["coordinate_frame_id"]):
        raise ExperimentError("snapshot and protocol do not identify the same frozen main map")


def _check_profile_gates(profile):
    if (profile["fast_loop"]["pnp"]["max_error_px"] != MAX_REPROJ_PX
            or profile["reloc"]["frozen_pnp_thresholds"]["strong_inliers"] != 80):
        raise ExperimentError("this experiment requires the unchanged 4px PnP and 80-point frozen gates")


def _check_ordinals(ref_ordinals, query_ordinals):
    if len(ref_ordinals) < MIN_REFERENCE_SUPPORT:
        raise ExperimentError("at least two covisible saved reference keyframes are required")
    if set(ref_ordinals) & set(query_ordinals):
        raise ExperimentError("reference and query captures must be disjoint for the closure experiment")
    if min(query_ordinals) <= max(ref_ordinals):
        raise ExperimentError("closure queries must occur after the selected saved reference captures")


def _reference_frames_and_submap(snapshot, ref_ordinals, epoch):
    frames = [keyframe_for_ordinal(snapshot["keyframes"], ordinal, epoch) for ordinal in ref_ordinals]
    if any(frame is None for frame in frames):
        raise ExperimentError("one or more requested reference keyframes are absent from the snapshot")
    submap_ids = {frame["submap_id"] for frame in frames}
    if len(submap_ids) != 1 or 0 in submap_ids:
        raise ExperimentError("references must belong to exactly one non-root child submap")
    submap_id = next(iter(submap_ids))
    submaps = [row for row in snapshot["submaps"] if row["id"] == submap_id]
    if len(submaps) != 1:
        raise ExperimentError("missing/ambiguous child submap metadata")
    submap = submaps[0]
    if submap["state"] == "merged" or submap["applied"] is not None:
        raise ExperimentError("merged snapshot coordinates are not pre-merge truth; provide an unmerged snapshot/run")
    return frames, submap


def _read_capture_records(frames_path, required_ordinals):
    records = {}
    with frames_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line, parse_constant=no_constant)
            if row["ordinal"] in required_ordinals:
                if row["ordinal"] in records:
                    raise ExperimentError("duplicate capture ordinal in frames.jsonl")
                records[row["ordinal"]] = row
    if set(records) != required_ordinals:
        raise ExperimentError(f"missing frames.jsonl captures: {sorted(required_ordinals - records.keys())}")
    return records


def _capture_identities(records, snapshot, frames):
    identities = {ordinal: capture_identity(row, snapshot) for ordinal, row in records.items()}
    run_ids = {row["run_id"] for row in records.values()}
    if len(run_ids) != 1:
        raise ExperimentError("selected captures span multiple runs")
    for frame in frames:
        if abs(frame["stamp"] - identities[frame["ordinal"]]["capture_stamp_mono"]) > 1e-6:
            raise ExperimentError("saved reference identity differs from frames.jsonl capture")
        required_file(frame["image_path"])
    return identities, run_ids


def _load_queries(args, snapshot, query_ordinals, epoch, identities, records):
    directory = None if args.query_images_dir is None else Path(args.query_images_dir).expanduser().resolve()
    queries = []
    for ordinal in query_ordinals:
        saved = keyframe_for_ordinal(snapshot["keyframes"], ordinal, epoch)
        image_path = query_path(directory, saved if directory is None else {"ordinal": ordinal})
        image, image_info = load_image(image_path, reference=directory is None)
        queries.append({"identity": identities[ordinal], "record": records[ordinal],
                        "saved_keyframe": saved, "image": image, "image_info": image_info})
    return queries


def _check_map_observations(queries, external, K, frozen_ids, frozen_xyz):
    # All selected map measurements, including fallback sources, are checked now
    # so malformed captures do not fail halfway through a GPU matching run.
    for query in queries:
        for _, observations, saved in map_observation_sources(query, external):
            check_camera(observations, K, saved=saved)
            ids, _, xyz = observation_arrays(observations, saved=saved)
            verify_frozen_points(ids[ids > 0], xyz[ids > 0], frozen_ids, frozen_xyz)


def _new_output_dir(args, run_dir):
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() or output_dir.is_relative_to(run_dir):
        raise ExperimentError("output directory must be a new research path outside the source run")
    return output_dir


def _closure_thresholds(profile):
    return {
        "local_min_inliers": LOCAL_MIN_INLIERS, "pnp_max_error_px": MAX_REPROJ_PX,
        "min_reference_support": MIN_REFERENCE_SUPPORT, "nearest_max_px": NEAREST_PX,
        "nearest_ratio": NEAREST_RATIO, "nearest_margin_px": NEAREST_MARGIN_PX,
        "query_agreement_px": QUERY_AGREEMENT_PX, "guided_radius_px": GUIDED_RADIUS_PX,
        "guided_min_donors": 4, "guided_max_residual_px": 2.0,
        "patch_min_ncc": PATCH_MIN_NCC, "patch_unique_peak_margin": PATCH_PEAK_MARGIN,
        "patch_max_fb_px": PATCH_MAX_FB_PX, "patch_max_shift_px": PATCH_MAX_SHIFT_PX,
        "frozen_pnp": profile["reloc"]["frozen_pnp_thresholds"],
        "sim3_downstream_unchanged": {"min_anchors": 8, "min_span_frames": 8,
                                     "max_span_frames": 48, "scale_bounds": [0.5, 2.0]},
    }


def load_inputs(args):
    protocol_path = required_file(args.protocol)
    run_dir = Path(args.run_dir).expanduser().resolve()
    snapshot_path = required_file(run_dir / "session/snapshot.json")
    frames_path = required_file(run_dir / "frames.jsonl")
    protocol = load_json(protocol_path)
    bundle_path, profile_path = asset(protocol_path, protocol["bundle"]), asset(protocol_path, protocol["profile"])
    bundle, profile, snapshot = load_json(bundle_path), load_json(profile_path), load_json(snapshot_path)
    _check_snapshot_identity(snapshot, protocol, bundle)
    K = camera_matrix(protocol, profile)
    _check_profile_gates(profile)
    ref_ordinals, query_ordinals = parse_ordinals(args.reference_ordinals), parse_ordinals(args.query_ordinals)
    _check_ordinals(ref_ordinals, query_ordinals)
    epoch = snapshot["source_epoch"]
    frames, submap = _reference_frames_and_submap(snapshot, ref_ordinals, epoch)
    required_ordinals = set(ref_ordinals + query_ordinals)
    records = _read_capture_records(frames_path, required_ordinals)
    identities, run_ids = _capture_identities(records, snapshot, frames)
    queries = _load_queries(args, snapshot, query_ordinals, epoch, identities, records)
    external, external_info = load_external_observations(
        args.map_observations_json, next(iter(run_ids)), snapshot, identities, K,
    )
    frozen_ids, frozen_xyz, geometry_info = frozen_geometry(bundle_path, bundle)
    _check_map_observations(queries, external, K, frozen_ids, frozen_xyz)
    if "edm" in args.methods:
        edm_paths()
    output_dir = _new_output_dir(args, run_dir)
    manifest = {
        "schema": "workmap-closure-experiment/v1", "offline_only": True,
        "protocol": {"path": str(protocol_path), "sha256": digest(protocol_path)},
        "frames": {"path": str(frames_path), "sha256": digest(frames_path)},
        "snapshot": {"path": str(snapshot_path), "sha256": digest(snapshot_path),
                     "revision": snapshot["revision"], "source_epoch": epoch,
                     "session_id": snapshot["session_id"], "coordinate_frame_id": snapshot["coordinate_frame_id"],
                     "geometry_time": "latest_unmerged_snapshot; not claimed capture-time geometry"},
        "bundle": {"path": str(bundle_path), "sha256": protocol["bundle"]["sha256"]},
        "profile": {"path": str(profile_path), "sha256": protocol["profile"]["sha256"]},
        "frozen_geometry": geometry_info, "external_map_observations": external_info,
        "source_code_sha256": digest(__file__), "run_id": next(iter(run_ids)),
        "submap": submap, "K": K.tolist(), "image_size_px": list(SIZE),
        "reference_ordinals": ref_ordinals, "query_ordinals": query_ordinals,
        "query_images": [query["image_info"] for query in queries],
        "methods": args.methods, "association_modes": args.association_modes,
        "holdout_modulus": args.holdout_modulus, "seed": args.seed,
        "thresholds": _closure_thresholds(profile),
        "versions": {"python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__},
        "limitations": [
            "No independent position ground truth; reprojection is not localization accuracy.",
            "Final unmerged geometry is explicitly revision-checked, not reconstructed historical geometry.",
            "Map-only anchors require raw positive observations; no mixed or held published-pose fallback.",
            "Nearest and guided paths are separate ablations and must not be counted as independent captures.",
            "Query-directory capture identity is supplied by the caller; filenames do not prove video decoding.",
            "No Sim3, BA, graph update, map write, replay, flight, or control action is performed.",
        ],
    }
    return {"snapshot": snapshot, "submap": submap, "frames": frames, "queries": queries,
            "K": K, "external": external, "frozen_ids": frozen_ids, "frozen_xyz": frozen_xyz,
            "profile": profile, "output_dir": output_dir, "manifest": manifest}


def homogeneous_pose(pose):
    result = np.eye(4)
    result[:3] = np.asarray(pose)
    return result.tolist()


def make_anchor(query, local, main, correspondences, method, association, inputs):
    local_rows, main_rows = local["inlier_indices"], main["inlier_indices"]
    local_xyz = np.asarray(local["world_xyz_u"])[local_rows]
    local_xy = np.asarray(local["image_xy_px"])[local_rows]
    main_xyz = np.asarray(main["world_xyz_u"])[main_rows]
    main_xy = np.asarray(main["image_xy_px"])[main_rows]
    return {
        "ordinal": query["identity"]["ordinal"], "method": f"{method}/{association}",
        "matcher": method, "association": association, "submap_id": inputs["submap"]["id"],
        "source_epoch": query["identity"]["source_epoch"],
        "capture_stamp_mono": query["identity"]["capture_stamp_mono"],
        "local_pose": homogeneous_pose(local["cam_from_world"]),
        "map_pose": homogeneous_pose(main["cam_from_world"]),
        "local_inliers": local["inlier_count"], "main_inliers": main["inlier_count"],
        "local_reprojection_px": local["reprojection_inliers"]["p50_px"],
        "main_reprojection_px": main["reprojection_inliers"]["p50_px"],
        "local_reprojection_p95_px": local["reprojection_inliers"]["p95_px"],
        "main_reprojection_p95_px": main["reprojection_inliers"]["p95_px"],
        "local_point_ids": local["inlier_point_ids"], "main_point_ids": main["inlier_point_ids"],
        "local_xyz": local_xyz.tolist(), "local_xy": local_xy.tolist(),
        "main_xyz": main_xyz.tolist(), "main_xy": main_xy.tolist(),
        "K": inputs["K"].tolist(), "image_size_px": list(SIZE),
        "provenance": {
            "capture": query["identity"], "query_image": query["image_info"],
            "snapshot": inputs["manifest"]["snapshot"],
            "main_source": main["observation_source"], "main_measurements": main["provenance"],
            "main_coordinate_verification": inputs["manifest"]["frozen_geometry"],
            "local_independence": "child_owned_revision_checked_XYZ+image_only_matches; no_main_pose_prior",
            "main_independence": main["independence"],
            "local_correspondences": [correspondences[index] for index in local_rows],
            "pose_convention": "4x4 camera_from_world; local_child and frozen_main frames differ by Sim3",
        },
    }


def run(args):
    started = time.perf_counter()
    cv2.setNumThreads(1)
    inputs = load_inputs(args)
    output_dir, K = inputs["output_dir"], inputs["K"]
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", inputs["manifest"])
    progress("inputs_validated", references=len(inputs["frames"]), queries=len(inputs["queries"]),
             methods=args.methods, association_modes=args.association_modes, output_dir=str(output_dir))
    references = [prepare_reference(frame, inputs["snapshot"], inputs["submap"], K,
                                    args.seed + frame["ordinal"]) for frame in inputs["frames"]]
    eligible_covisibility = defaultdict(set)
    for ref in references:
        for key in ref["ids"][ref["geometry_ok"]]:
            eligible_covisibility[int(key)].add(ref["frame"]["ordinal"])
    write_json(output_dir / "references.json", {
        "schema": "workmap-closure-references/v1", "records": [ref["report"] for ref in references],
    })
    thresholds = inputs["profile"]["reloc"]["frozen_pnp_thresholds"]
    map_poses = {}
    for query in inputs["queries"]:
        ordinal = query["identity"]["ordinal"]
        map_poses[ordinal] = prepare_map_pose(query, inputs["external"], K, thresholds,
                                            inputs["frozen_ids"], inputs["frozen_xyz"], args.seed + ordinal)
        progress("map_only_pnp", ordinal=ordinal, ok=map_poses[ordinal]["ok"],
                 attempts=[{"source": row["observation_source"], "inliers": row["inlier_count"],
                            "reasons": row["rejection_reasons"]} for row in map_poses[ordinal]["attempts"]])
    anchors, summaries, matcher_metadata = [], [], {}
    with (output_dir / "pairs.jsonl").open("w", encoding="utf-8") as pair_stream, (
        output_dir / "queries.jsonl"
    ).open("w", encoding="utf-8") as query_stream:
        for method in args.methods:
            matcher = SiftMatcher(references) if method == "sift" else EdmMatcher(references, args.edm_device)
            matcher_metadata[method] = matcher.metadata
            for query in inputs["queries"]:
                ordinal = query["identity"]["ordinal"]
                prepared = matcher.prepare_query(query["image"])
                gathered = {association: [] for association in args.association_modes}
                for ref in references:
                    pair_started = time.perf_counter()
                    rxy, qxy, scores, matcher_info = matcher.match(prepared, ref)
                    rxy, qxy, scores, match_indices, raw_counts = geometric_matches(
                        rxy, qxy, scores, args.seed + ordinal + ref["frame"]["ordinal"],
                    )
                    for association in args.association_modes:
                        seeds, rejected = (
                            nearest_seeds(ref, rxy, qxy, scores, match_indices) if association == "nearest"
                            else guided_seeds(ref, rxy, qxy, scores, match_indices)
                        )
                        candidates, patch_rejected = refine_seeds(ref, query["image"], seeds)
                        unique, unique_rejected = unique_reference_candidates(candidates)
                        rejected.update(patch_rejected)
                        rejected.update(unique_rejected)
                        gathered[association].extend(unique)
                        pair_rows = [{"point_id": row["point_id"], "image_xy_px": row["query_xy"],
                                      "world_xyz_u": row["world_xyz_u"]} for row in unique]
                        ids, xyz, xy = rows_arrays(pair_rows)
                        pnp = solve_pnp(ids, xyz, xy, K, seed=args.seed + ordinal)
                        pair = {
                            "schema": "workmap-closure-pair/v1", "method": method, "association": association,
                            "query_capture": query["identity"], "reference_ordinal": ref["frame"]["ordinal"],
                            "reference_revision": ref["frame"]["revision"],
                            "reference_pose_revision": ref["frame"]["pose_revision"],
                            "source_snapshot_revision": inputs["snapshot"]["revision"],
                            "source_geometry": ref["report"]["validation_pose_source"],
                            "source_initial_geometry_inconsistent": ref["report"]["initial_geometry_inconsistent"],
                            **raw_counts, "matcher": matcher_info, "association_seed_count": len(seeds),
                            "patch_verified_count": len(candidates), "trusted_unique_child_ids": ids.tolist(),
                            "trusted_unique_child_count": len(unique), "rejection_counts": dict(rejected),
                            "source_association_candidates": candidates, "pair_pnp": pnp,
                            "pair_pnp_is_not_covisibility_confirmed": True,
                            "elapsed_ms_since_match": (time.perf_counter() - pair_started) * 1000.0,
                        }
                        pair_stream.write(json_line(pair) + "\n")
                        pair_stream.flush()
                        progress("reference_query_matches", method=method, association=association,
                                 reference_ordinal=ref["frame"]["ordinal"], query_ordinal=ordinal,
                                 raw_matches=raw_counts["raw_matches"], seeds=len(seeds),
                                 trusted_ids=len(unique), pnp_inliers=pnp["inlier_count"])
                for association, candidates in gathered.items():
                    correspondences, rejected = aggregate_candidates(candidates, eligible_covisibility)
                    ids, xyz, xy = rows_arrays(correspondences)
                    local = solve_pnp(ids, xyz, xy, K, seed=args.seed + ordinal)
                    holdout = heldout_pnp(ids, xyz, xy, K, args.holdout_modulus, args.seed + ordinal)
                    main = map_poses[ordinal]
                    accepted = bool(local["ok"] and main["ok"])
                    reasons = list(local["rejection_reasons"]) + list(main["rejection_reasons"])
                    report = {
                        "schema": "workmap-closure-query/v1", "method": method, "association": association,
                        "capture": query["identity"], "image": query["image_info"],
                        "submap_id": inputs["submap"]["id"], "snapshot_revision": inputs["snapshot"]["revision"],
                        "trusted_unique_child_count": len(ids), "correspondences": correspondences,
                        "association_rejection_counts": rejected, "local_pnp": local,
                        "heldout_pnp": holdout, "map_only": main,
                        "anchor_accepted": accepted, "rejection_reasons": reasons,
                    }
                    query_stream.write(json_line(report) + "\n")
                    query_stream.flush()
                    if accepted:
                        anchors.append(make_anchor(query, local, main["selected"], correspondences,
                                                   method, association, inputs))
                    summary = {"ordinal": ordinal, "method": method, "association": association,
                               "trusted_child_ids": len(ids), "local_pnp_inliers": local["inlier_count"],
                               "local_pnp_ok": local["ok"], "main_pnp_ok": main["ok"],
                               "anchor_accepted": accepted, "rejection_reasons": reasons}
                    summaries.append(summary)
                    progress("query_complete", **summary)
            del matcher
    write_json(output_dir / "anchors.json", {
        "schema": "workmap-closure-anchors/v1", "records": anchors,
        "run_id": inputs["manifest"]["run_id"], "coordinate_frame_id": inputs["snapshot"]["coordinate_frame_id"],
        "independent_groups": "group by method; never pool matcher/association alternatives as extra captures",
    })
    summary = {"schema": "workmap-closure-summary/v1", "completed": True, "offline_only": True,
               "wall_s": time.perf_counter() - started, "records": summaries, "matchers": matcher_metadata,
               "accepted_anchor_count_by_method": dict(Counter(row["method"] for row in anchors)),
               "source_geometry_written": False, "accuracy_claim": None}
    write_json(output_dir / "summary.json", summary)
    progress("complete", output_dir=str(output_dir), anchor_count=len(anchors), wall_s=summary["wall_s"])
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--protocol", required=True, type=Path)
    result.add_argument("--run-dir", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--methods", nargs="+", choices=("sift", "edm"), default=["sift", "edm"])
    result.add_argument("--reference-ordinals", nargs="+", default=["848", "856", "864"],
                        help="saved child keyframes; comma lists or inclusive START:STOP[:STEP]")
    result.add_argument("--query-ordinals", nargs="+", default=["928"],
                        help="exact captured queries; comma lists or inclusive START:STOP[:STEP]")
    result.add_argument("--query-images-dir", type=Path,
                        help="exact decoded query images named ordinal.png or six-digit ordinal.png")
    result.add_argument("--map-observations-json", type=Path,
                        help="capture-linked positive main-map observations; schema documented above")
    result.add_argument("--association-modes", nargs="+", choices=("nearest", "guided"), default=["nearest", "guided"],
                        help="separate ablations, never merged into one pose")
    result.add_argument("--edm-device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--holdout-modulus", type=int, default=5,
                        help="stable point-ID split for extra diagnostic PnP; zero disables it")
    result.add_argument("--seed", type=int, default=0)
    return result


def main():
    args = parser().parse_args()
    if args.holdout_modulus != 0 and args.holdout_modulus < 2:
        raise SystemExit("--holdout-modulus must be zero or >=2")
    if len(set(args.methods)) != len(args.methods) or len(set(args.association_modes)) != len(args.association_modes):
        raise SystemExit("methods and association modes must be unique")
    try:
        return run(args)
    except (ExperimentError, OSError, KeyError, TypeError, ValueError, RuntimeError, cv2.error) as exc:
        print(json_line({"offline_only": True, "event": "fatal_input_or_runtime_error",
                         "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
