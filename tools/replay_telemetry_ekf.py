#!/usr/bin/env python3
"""Offline visual versus telemetry ESKF comparison; never imports the flight SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "定位演算法/validation"))
from fusion_session_data import (  # noqa: E402
    calibration_segments,
    file_digest,
    fit_horizontal_calibration,
    load_session,
)
from pose_filter_experiment import VisualPoseESKF  # noqa: E402
from telemetry_pose_experiment import TelemetryPoseESKF  # noqa: E402


def distribution(values):
    return {
        "count": len(values),
        **{
            name: float(np.percentile(values, q)) if values else None
            for name, q in (("p50", 50), ("p95", 95), ("max", 100))
        },
    }


def _evaluate_frame(row, index, filters, samples, errors, timings):
    # Every fifth observed pose is held out from BOTH filters, including gates.
    held_out = row["pose"] is not None and index % 5 == 4
    results = {}
    for name, estimator in filters.items():
        start = time.perf_counter()
        result = estimator.update(
            row["stamp"],
            None if held_out else row["pose"],
            epoch=0,
            position_sigma=0.02,
            rotation_sigma=0.1,
        )
        timings[name].append((time.perf_counter() - start) * 1000.0)
        matrix = result.pop("pose")
        results[name] = {
            **result,
            "pose": None if matrix is None else matrix.tolist(),
            "covariance_diagonal": np.diag(estimator._P).tolist(),
            "visual_age_s": None
            if estimator._last_accepted_stamp is None
            else row["stamp"] - estimator._last_accepted_stamp,
        }
    if held_out and all(results[name]["pose"] is not None for name in filters):
        truth = row["pose"]
        center = -truth[:, :3].T @ truth[:, 3]
        for name in filters:
            pose = np.asarray(results[name]["pose"])
            errors[name].append(float(np.linalg.norm((-pose[:, :3].T @ pose[:, 3] - center)[:])))
    samples.append(
        {"stamp": row["stamp"], "held_out": held_out, "source_status": row["status"], **results}
    )


def evaluate(frames, velocities, basis, calibration, cutoff):
    scale = calibration["scale_m_per_unit"]
    rotation = np.asarray(calibration["rotation_en_to_map"])
    filters = {"visual_esekf": VisualPoseESKF(), "telemetry_esekf": TelemetryPoseESKF(basis[:2])}
    samples, velocity_updates = [], []
    errors, timings = {name: [] for name in filters}, {name: [] for name in filters}
    events = [(row["stamp"], 0, row) for row in frames if row["stamp"] > cutoff]
    events += [(row[0], 1, row) for row in velocities if cutoff < row[0] <= frames[-1]["stamp"]]
    observed_index = 0
    for stamp, kind, row in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == 1:
            start = time.perf_counter()
            result = filters["telemetry_esekf"].update_velocity(
                stamp, rotation @ row[[2, 1]] / scale, epoch=0, sigma=0.2 / scale
            )
            velocity_updates.append(
                {"stamp": stamp, **result, "wall_ms": (time.perf_counter() - start) * 1000.0}
            )
        else:
            _evaluate_frame(row, observed_index, filters, samples, errors, timings)
            observed_index += int(row["pose"] is not None)
    return (
        {
            "common_heldout_position_residual_u": {
                name: distribution(values) for name, values in errors.items()
            },
            "pose_update_wall_ms": {name: distribution(values) for name, values in timings.items()},
            "velocity_updates": len(velocity_updates),
            "velocity_accepted": sum(row["accepted"] for row in velocity_updates),
            "velocity_update_wall_ms": distribution([r["wall_ms"] for r in velocity_updates]),
            "coverage": {
                name: {
                    "outputs": sum(r[name]["pose"] is not None for r in samples),
                    "accepted_visual": sum(r[name]["accepted"] for r in samples),
                    "predicted_only": sum(r[name]["predicted_only"] for r in samples),
                }
                for name in filters
            },
        },
        samples,
        velocity_updates,
    )


def run(session, out):
    frames, velocities, basis, identity = load_session(session)
    if len(frames) < 2:
        raise ValueError("at least two source-timestamped frames are required")
    cutoff = frames[0]["stamp"] + 0.4 * (frames[-1]["stamp"] - frames[0]["stamp"])
    calibration = fit_horizontal_calibration(
        calibration_segments(frames, velocities, basis, cutoff)
    )
    report = {
        "schema": "sfm-telemetry-esekf-experiment/v1",
        "session": str(Path(session).resolve()),
        "source_sha256": identity,
        "calibration_cutoff_mono": cutoff,
        "frames": len(frames),
        "distinct_velocity_samples": len(velocities),
        "calibration": calibration,
        "formal_control_ready": False,
        "assumed_noise": {
            "position_sigma_u": 0.02,
            "rotation_sigma_rad": 0.1,
            "velocity_sigma_mps": 0.2,
            "acceleration_sigma_u_s2": 1.0,
        },
        "limitations": [
            "Measurement-time offline replay; transport delays are not an online shadow test.",
            "Calibration uses only the first 40% of source time; evaluation uses the remaining 60%.",
            "Every fifth observed evaluation pose is withheld from both filters; residual is not ground truth.",
            "Telemetry is firmware-fused and may be correlated with visual motion; covariance is assumed.",
            "Only horizontal velocity is fused. Altitude and body attitude are not camera measurements.",
            "An inconsistent fitted calibration remains a hypothesis, never a deployment calibration.",
        ],
        "implementation_sha256": {
            str(p.relative_to(ROOT)): file_digest(p)
            for p in (
                Path(__file__).resolve(),
                ROOT / "定位演算法/validation/fusion_session_data.py",
                ROOT / "定位演算法/validation/telemetry_pose_experiment.py",
                ROOT / "定位演算法/validation/pose_filter_experiment.py",
            )
        },
    }
    samples, updates = [], []
    if "scale_m_per_unit" in calibration:
        metrics, samples, updates = evaluate(frames, velocities, basis, calibration, cutoff)
        report["evaluation"] = metrics
    else:
        report["evaluation"] = {"unavailable": calibration["reason"]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("poses.jsonl", samples), ("velocity_updates.jsonl", updates)):
        (out / name).write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.session, args.out)
    print(json.dumps({"calibration": result["calibration"], "evaluation": result["evaluation"]}))


if __name__ == "__main__":
    main()
