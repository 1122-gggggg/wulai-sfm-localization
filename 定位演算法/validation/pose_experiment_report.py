#!/usr/bin/env python3
"""Summarize paired estimator trials on map-only held-out observations.

No sensor ground truth is inferred from these residuals. All method columns
use exactly the same eligible frames; absent outputs are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pose_estimation_experiment import projection_errors, quantiles, write_json


def summarize(capture_path: Path, result_path: Path):
    capture = json.loads((capture_path / "capture.json").read_text())
    summary = json.loads((result_path / "summary.json").read_text())
    records = [json.loads(line) for line in (result_path / "results.jsonl").read_text().splitlines()]
    data = np.load(capture_path / "observations.npz", allow_pickle=False)
    xy, xyz, ids, offsets = (data[key] for key in ("xy", "xyz", "ids", "offsets"))
    K = np.asarray(capture["K"])
    methods = list(summary["methods"])
    by_method = {name: {row["frame"]: row for row in records if row["method"] == name} for name in methods}
    eligible, common = [], []
    for index in by_method["refine2"]:
        a, b = offsets[index:index + 2]
        holdout = (ids[a:b] > 0) & (ids[a:b] % 5 == 0)
        if holdout.sum() >= 6 and by_method["refine2"][index]["ok"]:
            eligible.append(index)
            if all(by_method[name][index]["ok"] for name in methods):
                common.append(index)
    frame_errors = {name: [] for name in methods}
    rows = []
    for name in methods:
        pooled_errors = []
        for index in common:
            a, b = offsets[index:index + 2]
            holdout = (ids[a:b] > 0) & (ids[a:b] % 5 == 0)
            errors = projection_errors(np.asarray(by_method[name][index]["pose"]),
                                       xyz[a:b][holdout], xy[a:b][holdout], K)
            pooled_errors.append(errors)
            frame_errors[name].append(float(np.median(errors)))
        pooled = np.concatenate(pooled_errors) if pooled_errors else np.zeros(0)
        measures = summary["methods"][name]
        rows.append({
            "method": name, "frames": measures["frames"], "pose_outputs": measures["valid_poses"],
            "measured_outputs": measures["measured_poses"], "predicted_outputs": measures["predicted_only"],
            "eligible_map_frames": len(eligible), "common_map_frames": len(common),
            "missing_on_eligible": sum(not by_method[name][i]["ok"] for i in eligible),
            "heldout_map_observations": len(pooled),
            "map_reprojection_p50_px": float(np.percentile(pooled, 50)) if len(pooled) else None,
            "map_reprojection_p95_px": float(np.percentile(pooled, 95)) if len(pooled) else None,
            "estimator_p50_ms": measures["estimator_total_ms"]["p50"],
            "estimator_p95_ms": measures["estimator_total_ms"]["p95"],
            "additional_stage_p50_ms": measures["stage_ms"]["p50"],
            "acceleration_proxy_p50_u_s2": measures["acceleration_u_s2"]["p50"],
            "acceleration_proxy_p95_u_s2": measures["acceleration_u_s2"]["p95"],
            "optimized_frames": measures["optimized_frames"],
        })
    baseline = np.asarray(frame_errors["refine2"])
    for row in rows:
        differences = np.asarray(frame_errors[row["method"]]) - baseline
        row["paired_frame_error_delta_px"] = quantiles(differences.tolist())
        row["paired_frame_improved_fraction"] = float(np.mean(differences < -1e-6)) if len(differences) else None
    report = {
        "video": capture["video"], "video_sha256": capture["video_sha256"],
        "decoded_frames": capture["frames"], "declared_frames": capture["declared_frames"],
        "capture_status_counts": capture["status_counts"],
        "metrics_contract": {
            "accuracy": "image reprojection consistency only; no metric pose ground truth",
            "map_holdout": "positive map IDs divisible by 5, absent from all estimator inputs",
            "common_frames": "baseline pose and >=6 map holdout points, outputs from every method",
            "frontend_conditioning": "KLT/VO/admission history is captured from baseline, not independently rerun per method",
            "latency": "shared RANSAC plus refinement, plus optional temporal stage; excludes KLT/decode/relocalizer",
            "smoothness": "second difference of estimated center, includes actual motion; not a truth error",
            "window": "fixed first pose, local point corrections and PnP regularization; no marginalization or NED factor",
        },
        "methods": rows,
    }
    write_json(result_path / "comparison.json", report)
    with (result_path / "comparison.csv").open("w", newline="") as stream:
        columns = [key for key in rows[0] if key != "paired_frame_error_delta_px"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.capture, args.results)
