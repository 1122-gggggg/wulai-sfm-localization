#!/usr/bin/env python3
"""Offline fixed-measurement frozen/mixed PnP; never updates a map or pose.

Example: workmap_pnp_ablation.py --protocol control_protocol.json
  --run-dir control_replay/full/P2310231/repeat-1 --output ablation.json
Repeat --run-dir for independent reports. Saved inlier samples are selected,
truncated evidence, not the original live correspondence set. Ordinary durable
keyframe writer records preserve larger observation-time sets. Final snapshot
coordinates and postmerge pre-anchor keyframes are never historical substitutes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pycolmap

import workmap_closure_experiment as closure

SCHEMA = "workmap-pnp-ablation/v1"


def read_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line, parse_constant=closure.no_constant)


def split_observations(ids):
    """Withhold floor(Nmain/5) real IDs by stable hash, independent of row order.

    All negatives train in mixed only. >=100 positives guarantees >=80 main
    training observations; smaller sets remain exploratory, never strong evidence.
    """
    positive = np.flatnonzero(ids > 0)
    ranked = sorted(positive, key=lambda row: (
        hashlib.sha256(f"workmap-pnp-holdout/v1:{int(ids[row])}".encode()).digest(),
        int(ids[row]),
    ))
    heldout = np.zeros(len(ids), dtype=bool)
    heldout[ranked[:len(positive) // 5]] = True
    return (ids > 0) & ~heldout, (ids != 0) & ~heldout, heldout


def support(xyz, xy, pose, K, threshold):
    errors, positive = closure.residuals(xyz, xy, pose, K)
    finite = errors[np.isfinite(errors)]
    return {
        "available": len(xyz), "positive_depth_count": int(positive.sum()),
        "positive_depth_ratio": float(positive.mean()) if len(positive) else None,
        "support_count": int(np.count_nonzero(errors <= threshold)),
        "unsupported_count": int(np.count_nonzero(errors > threshold)),
        "reprojection_finite_count": len(finite),
        "reprojection_p50_px": float(np.percentile(finite, 50)) if len(finite) else None,
        "reprojection_p90_px": float(np.percentile(finite, 90)) if len(finite) else None,
        "reprojection_nonfinite_count": int(len(errors) - len(finite)),
    }


def solve(ids, xy, xyz, fit, heldout, K, pnp, seed):
    rows = np.flatnonzero(fit)
    result = {
        "initializer": None, "initializer_free": True,
        "training_main_available": int(np.count_nonzero(ids[rows] > 0)),
        "training_new_available": int(np.count_nonzero(ids[rows] < 0)),
        "training_ids": ids[rows].tolist(), "status": "insufficient_training_points",
    }
    if len(rows) < int(pnp["min_points"]):
        return result
    camera = pycolmap.Camera.create(1, "PINHOLE", float(K[0, 0]), *closure.SIZE)
    camera.params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    estimation = pycolmap.AbsolutePoseEstimationOptions()
    estimation.ransac.max_error = float(pnp["max_error_px"])
    estimation.ransac.min_num_trials = int(pnp["min_trials"])
    estimation.ransac.max_num_trials = int(pnp["max_trials"])
    estimation.ransac.confidence = float(pnp["confidence"])
    estimation.ransac.random_seed = int(seed)
    refinement = pycolmap.AbsolutePoseRefinementOptions()
    refinement.max_num_iterations = int(pnp["refine_iters"])
    answer = pycolmap.estimate_and_refine_absolute_pose(
        np.ascontiguousarray(xy[rows]), np.ascontiguousarray(xyz[rows]),
        camera, estimation, refinement,
    )
    if answer is None:
        result["status"] = "ransac_no_pose"
        return result
    pose = closure.pose_array(answer["cam_from_world"].matrix(), "fitted pose")
    mask = np.asarray(answer["inlier_mask"], dtype=bool)
    if mask.shape != (len(rows),):
        raise closure.ExperimentError("unexpected pycolmap inlier mask shape")
    main = ids[rows] > 0
    new = ids[rows] < 0
    result.update({
        "status": "fit", "cam_from_world": pose.tolist(),
        "training_main_inliers": int(np.count_nonzero(mask & main)),
        "training_new_inliers": int(np.count_nonzero(mask & new)),
        "training_main_inliers_lost": int(np.count_nonzero(~mask & main)),
        "training_new_inliers_lost": int(np.count_nonzero(~mask & new)),
        "training_main_inlier_ids": ids[rows][mask & main].tolist(),
        "training_new_inlier_ids": ids[rows][mask & new].tolist(),
        "meets_fast_loop_min_inliers": int(mask.sum()) >= int(pnp["min_inliers"]),
        "heldout_main": support(xyz[heldout], xy[heldout], pose, K, pnp["max_error_px"]),
        "all_recorded_new": support(xyz[ids < 0], xy[ids < 0], pose, K, pnp["max_error_px"]),
    })
    return result


def compare_observations(row, identity, K, frozen_ids, frozen_xyz, pnp, strong, seed):
    closure.check_camera(row, K)
    ids, xy, xyz = closure.observation_arrays(row)
    if np.any(ids == 0):
        raise closure.ExperimentError("zero point ID has no valid landmark identity")
    closure.verify_frozen_points(ids[ids > 0], xyz[ids > 0], frozen_ids, frozen_xyz)
    old_fit, mixed_fit, heldout = split_observations(ids)
    report = {
        "capture_identity": identity,
        "main_available": int(np.count_nonzero(ids > 0)),
        "new_available": int(np.count_nonzero(ids < 0)),
        "training_main_available": int(old_fit.sum()),
        "strong_main_training_eligible": int(old_fit.sum()) >= strong,
        "strong_admission_evaluated": False,
        "heldout_main_ids": ids[heldout].tolist(),
        "observations": {"ids": ids.tolist(), "image_xy_px": xy.tolist(),
                         "world_xyz_u": xyz.tolist(), "K": K.tolist()},
        "status": "compared" if heldout.any() else "skipped_no_main_holdout",
    }
    if not heldout.any():
        return report
    report["frozen_only"] = solve(ids, xy, xyz, old_fit, heldout, K, pnp, seed)
    report["mixed"] = solve(ids, xy, xyz, mixed_fit, heldout, K, pnp, seed)
    a, b = report["frozen_only"], report["mixed"]
    if a["status"] == b["status"] == "fit":
        deltas = {}
        for field in ("reprojection_p50_px", "reprojection_p90_px", "support_count", "positive_depth_count"):
            va, vb = a["heldout_main"][field], b["heldout_main"][field]
            deltas[f"heldout_main_{field}"] = vb - va if va is not None and vb is not None else None
        deltas["training_main_inliers"] = b["training_main_inliers"] - a["training_main_inliers"]
        deltas["new_support_count"] = b["all_recorded_new"]["support_count"] - a["all_recorded_new"]["support_count"]
        report["mixed_minus_frozen"] = deltas
    return report


def summarize(records):
    counts = Counter(row["status"] for row in records)
    paired = [row for row in records if "mixed_minus_frozen" in row]
    mixed = [row for row in paired if row["new_available"] > 0]
    strong = [row for row in mixed if row["strong_main_training_eligible"]]
    def subset(rows):
        changes = [row["mixed_minus_frozen"] for row in rows]
        result = {"count": len(rows)}
        for field in ("heldout_main_reprojection_p90_px", "heldout_main_support_count", "training_main_inliers", "new_support_count"):
            values = [x[field] for x in changes if x[field] is not None]
            result[field] = {"finite_count": len(values), "median": float(np.median(values)) if values else None,
                             "sum": float(np.sum(values)) if values else None,
                             "negative_count": sum(v < -1e-9 for v in values),
                             "positive_count": sum(v > 1e-9 for v in values)}
        return result
    return {"records": len(records), "status_counts": dict(counts),
            "immutable_positive_xyz_verified_count": sum(r.get("immutable_positive_xyz_verified", False) for r in records),
            "candidate_status_counts": {
                name: dict(Counter(r[name]["status"] for r in records if name in r))
                for name in ("frozen_only", "mixed")
            },
            "valid_main_observation_incidences": sum(r.get("main_available", 0) for r in records),
            "valid_new_observation_incidences": sum(r.get("new_available", 0) for r in records),
            "paired_fits": subset(paired), "paired_with_new": subset(mixed),
            "paired_with_new_and_80_main_training": subset(strong)}


def _load_frame_samples(frames_path, snapshot):
    frames = {}
    identity_errors = []
    samples = []
    missing_samples = 0
    for line, frame in read_lines(frames_path):
        try:
            identity = closure.capture_identity(frame, snapshot)
            if identity["ordinal"] in frames:
                raise closure.ExperimentError("duplicate capture ordinal within run")
            frames[identity["ordinal"]] = (frame, identity)
        except (ValueError, KeyError, TypeError) as exc:
            identity_errors.append({"line": line, "reason": str(exc)})
            continue
        sample = frame["response"].get("pnp_observation_sample")
        if sample is None:
            missing_samples += 1
            continue
        samples.append((line, sample, identity, "recorded_inlier_sample", {}))
    return frames, identity_errors, samples, missing_samples


def _keyframe_records(keyframes_path, frames, snapshot, samples):
    records = []
    submaps = {int(row["id"]): row for row in snapshot["submaps"]}
    independent_sidecars = []
    if not keyframes_path.is_file():
        return records, independent_sidecars
    for line, row in read_lines(keyframes_path):
        meta = {"path": str(keyframes_path), "line": line, "name": row.get("name"),
                "revision": row.get("revision"), "submap_id": row.get("submap_id")}
        ordinal = row.get("ordinal")
        if ordinal not in frames:
            records.append({"status": "invalid", "source": "keyframe_writer", "provenance": meta,
                            "reason": "keyframe_missing_matching_capture"})
            continue
        _, identity = frames[ordinal]
        submap = submaps.get(row.get("submap_id"), {})
        if submap.get("kind") == "independent":
            independent_sidecars.append({"ordinal": ordinal, "submap_id": row["submap_id"],
                                        "main_available": sum(i > 0 for i in row.get("all_ids", []))})
            records.append({"status": "skipped_postmerge_coordinates", "source": "keyframe_writer",
                            "capture_identity": identity, "provenance": meta,
                            "reason": "presubmap writer stores merged points_map, not observation-time local XYZ"})
            continue
        observation = {"ids": row.get("all_ids"), "image_xy_px": row.get("all_xy"),
                       "world_xyz_u": row.get("all_xyz"), "K": row.get("K"),
                       "image_size_px": row.get("image_size"), "capture_stamp_mono": row.get("stamp")}
        if (row.get("source_epoch") != identity["source_epoch"]
                or row.get("session_id") != snapshot["session_id"] or not row.get("durable_complete")
                or submap.get("kind") != "anchored"):
            records.append({"status": "invalid", "source": "keyframe_writer", "provenance": meta,
                            "reason": "missing_durable_capture_or_coordinate_provenance"})
            continue
        samples.append((line, observation, identity, "keyframe_writer", meta))
    return records, independent_sidecars


def _sample_provenance(frames_path, keyframes_path, frames, sample):
    line, observation, identity, source, meta = sample
    provenance = {"path": str(frames_path) if source == "recorded_inlier_sample" else str(keyframes_path),
                  "line": line, "sampling": observation.get("sampling"),
                  "coordinate_origin": "as_recorded_capture_sample" if source == "recorded_inlier_sample" else "writer_copied_at_keyframe_observation", **meta}
    response = frames[identity["ordinal"]][0]["response"]
    provenance.update({"recorded_total_live_correspondences": response.get("n_corr"),
                       "recorded_main_inliers": response.get("map_inliers"),
                       "recorded_new_inliers": response.get("vo_inliers"),
                       "prefit_live_main_available": None})
    if source == "recorded_inlier_sample":
        provenance.update({"sample_count": observation.get("sample_count"), "total_inliers": observation.get("total_inliers")})
    return provenance


def _sample_report(sample, K, frozen_ids, frozen_xyz, pnp, strong, seed):
    _, observation, identity, source, _ = sample
    try:
        stamp = float(observation["capture_stamp_mono"])
        if not np.isfinite(stamp) or abs(stamp - identity["capture_stamp_mono"]) > 1e-6:
            raise closure.ExperimentError("observation timestamp differs from capture")
        if source == "recorded_inlier_sample" and observation.get("sample_count") != len(observation["ids"]):
            raise closure.ExperimentError("sample_count differs from saved observation count")
        report = compare_observations(observation, identity, K, frozen_ids, frozen_xyz, pnp, strong, seed)
        report["immutable_positive_xyz_verified"] = True
    except (ValueError, KeyError, TypeError) as exc:
        report = {"status": "invalid", "capture_identity": identity, "reason": str(exc)}
    return report


def _best_available_window(records, strong):
    # Availability upper bound, not admitted anchors and not pooled fits:
    # sparse captures can rule out a >=12-capture / <=48-frame comparison
    # without manufacturing local fits or assuming unit child scale.
    possible = sorted({row["capture_identity"]["ordinal"] for row in records
                       if row.get("immutable_positive_xyz_verified")
                       and row.get("main_available", 0) >= strong
                       and row.get("new_available", 0) >= closure.LOCAL_MIN_INLIERS})
    windows = [[ordinal for ordinal in possible if start <= ordinal <= start + 48]
               for start in possible]
    best_window = max(windows, key=len) if windows else []
    return possible, best_window


def _graph_prior_feasibility(independent_sidecars, possible, best_window, strong):
    return {
        "status": "missing_independent_premerge_evidence", "corrections_applied": False,
        "independent_submap_writer_records": independent_sidecars,
        "ordinary_main_aligned_diagnostic": {
            "available_capture_upper_bound": len(possible),
            "maximum_available_within_48_frames": len(best_window),
            "best_window_ordinals": best_window, "required_captures": 12,
            "main_minimum_available": strong,
            "new_minimum_available": closure.LOCAL_MIN_INLIERS,
            "status": "insufficient_available_anchor_window" if len(best_window) < 12 else "needs_independent_local_fits",
            "fits_performed": False,
            "interpretation": "Unique captures meeting saved main/new availability minima, before fit losses. Main-aligned negative XYZ cannot recover original independent-child scale. No scale=1 assumption or graph correction is made.",
        },
        "missing_fields": ["premerge local_xyz with real local point IDs and measured pixels",
                           "premerge local cam_from_world or initializer-free local PnP measurements",
                           "local-to-real-landmark ID mapping and local coordinate revision/basis",
                           "same-capture independent frozen-main correspondences for >=8 train + >=4 heldout anchors within 48 frames"],
        "reason": "Presubmap archival converts points/poses to main coordinates before writing; merged coordinates are not premerge truth. Main-frame negative fits cannot establish an independent child scale. Published poses, event scale, and arbitrary scale=1 are not used as closure measurements.",
    }


def analyze_run(run_dir, protocol, bundle, K, frozen_ids, frozen_xyz, pnp, strong, seed):
    run_dir = Path(run_dir).expanduser().resolve()
    snapshot_path = closure.required_file(run_dir / "session/snapshot.json")
    snapshot = closure.load_json(snapshot_path)
    if (snapshot.get("schema") != "direct-online-workmap-session/v1"
            or snapshot["base_bundle_sha256"] != protocol["bundle"]["sha256"]
            or snapshot["coordinate_frame_id"] != bundle["coordinate_frame_id"]):
        raise closure.ExperimentError("snapshot does not identify the immutable protocol main map")
    frames_path = closure.required_file(run_dir / "frames.jsonl")
    frames, identity_errors, samples, missing_samples = _load_frame_samples(frames_path, snapshot)
    keyframes_path = run_dir / "session/keyframes.jsonl"
    records, independent_sidecars = _keyframe_records(keyframes_path, frames, snapshot, samples)
    for sample in samples:
        provenance = _sample_provenance(frames_path, keyframes_path, frames, sample)
        report = _sample_report(sample, K, frozen_ids, frozen_xyz, pnp, strong, seed)
        report.update({"source": sample[3], "provenance": provenance})
        records.append(report)
    sources = sorted({row["source"] for row in records})
    possible, best_window = _best_available_window(records, strong)
    return {
        "run_dir": str(run_dir), "session_id": snapshot["session_id"], "source_epoch": snapshot["source_epoch"],
        "coordinate_frame_id": snapshot["coordinate_frame_id"],
        "inputs": {"frames_sha256": closure.digest(frames_path), "snapshot_sha256": closure.digest(snapshot_path),
                   "keyframes_sha256": closure.digest(keyframes_path) if keyframes_path.is_file() else None},
        "capture_count": len(frames), "invalid_capture_count": len(identity_errors), "invalid_captures": identity_errors,
        "missing_sample_count": missing_samples, "keyframe_writer_missing": not keyframes_path.is_file(),
        "summary_by_source": {source: summarize([r for r in records if r["source"] == source]) for source in sources},
        "graph_prior_feasibility": _graph_prior_feasibility(
            independent_sidecars, possible, best_window, strong),
        "records": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ransac-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; refusing overwrite")
    if args.ransac_seed < 0:
        parser.error("RANSAC seed must be nonnegative")
    if len({path.resolve() for path in args.run_dir}) != len(args.run_dir):
        parser.error("duplicate --run-dir")
    protocol_path = closure.required_file(args.protocol)
    protocol = closure.load_json(protocol_path)
    bundle_path = closure.asset(protocol_path, protocol["bundle"])
    profile_path = closure.asset(protocol_path, protocol["profile"])
    bundle, profile = closure.load_json(bundle_path), closure.load_json(profile_path)
    K = closure.camera_matrix(protocol, profile)
    intrinsics = profile["intrinsics"]
    deployed_K = closure.finite_array(intrinsics["K"], (3, 3), "profile camera K").copy()
    deployed_K[0] *= closure.SIZE[0] / intrinsics["image_width"]
    deployed_K[1] *= closure.SIZE[1] / intrinsics["image_height"]
    if not intrinsics["images_are_undistorted"] or not np.allclose(K, deployed_K, rtol=0, atol=1e-7):
        raise closure.ExperimentError("protocol camera differs from deployed undistorted PINHOLE profile")
    frozen_ids, frozen_xyz, geometry = closure.frozen_geometry(bundle_path, bundle)
    pnp = profile["fast_loop"]["pnp"]
    strong = int(profile["reloc"]["frozen_pnp_thresholds"]["strong_inliers"])
    report = {
        "schema": SCHEMA, "offline_only": True, "corrections_applied": False,
        "protocol": {"path": str(protocol_path), "sha256": closure.digest(protocol_path)},
        "immutable_geometry": geometry, "profile_sha256": closure.digest(profile_path),
        "solver": {"library": "pycolmap", "version": pycolmap.__version__, "camera_model": "PINHOLE",
                   "K": K.tolist(), "image_size_px": list(closure.SIZE), "pnp": pnp,
                   "ransac_seed": args.ransac_seed, "initializer": None},
        "split": "per-capture, floor(Npositive/5) withheld by SHA256(real ID); no withheld positives in either fit; negatives only in mixed",
        "limitations": ["Samples contain at most 128 already-selected PnP inliers; missing rejected/live inputs and selection bias preclude causal all-run claims.",
                        "Ordinary keyframe writer sets are measured but selected by capture/keyframe admission; they are not all raw live inputs.",
                        "main_available counts saved measured points, not the missing prefit live-main set; disappearance before capture/sample selection cannot be attributed to mixed-PnP bias.",
                        "Final snapshot XYZ are never used. Postmerge independent-submap writer records are skipped as historical-coordinate evidence.",
                        "Support/reprojection are internal diagnostics, not position ground truth. New-point support is in-sample for mixed, not independent validation.",
                        "Strong training eligibility means >=80 measured main training points only; full production admission gates are not evaluated or lowered.",
                        "Each source/capture/run is fit separately; no pooling across captures/epochs. Aggregate counts are observation incidences, not unique landmarks; sources may overlap."],
        "runs": [analyze_run(path, protocol, bundle, K, frozen_ids, frozen_xyz, pnp, strong, args.ransac_seed) for path in args.run_dir],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    closure.write_json(args.output, report)
    print(closure.json_line({"output": str(args.output), "runs": [
        {"run_dir": run["run_dir"], "missing_sample_count": run["missing_sample_count"],
         "summary_by_source": run["summary_by_source"]} for run in report["runs"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
