#!/usr/bin/env python3
"""Offline horizontal outage replay. No flight SDK or production-policy changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "定位演算法/validation"))
from fusion_session_data import (  # noqa: E402
    calibration_segments,
    file_digest,
    fit_horizontal_calibration,
    load_session,
)


def causal_displacement(velocities, start, end):
    """Zero-order hold of received NED samples; never read past the gap endpoint."""
    if not len(velocities) or not np.isfinite([start, end]).all() or end <= start:
        return None
    times = velocities[:, 0]
    first = int(np.searchsorted(times, start, side="right")) - 1
    stop = int(np.searchsorted(times, end, side="right"))
    if first < 0 or start - times[first] > 0.5:
        return None
    samples = velocities[first:stop]
    if end - samples[-1, 0] > 0.5 or np.any(np.diff(samples[:, 0]) > 0.5):
        return None
    knots = np.r_[start, samples[1:, 0], end]
    return np.sum(samples[:, 1:3] * np.diff(knots)[:, None], axis=0)


def _center(row, basis):
    pose = row["pose"]
    return basis[:2] @ (-pose[:, :3].T @ pose[:, 3])


def evaluate_outages(frames, velocities, basis, calibration, cutoff, horizons):
    """Hold out whole windows after calibration; score only common endpoint pairs."""
    observed = [
        row
        for row in frames
        if row["pose"] is not None and row["stamp"] > cutoff and row["status"] == "FAST_TRACK"
    ]
    times = np.array([row["stamp"] for row in observed])
    transform = np.asarray(calibration["rotation_en_to_map"]) / calibration["scale_m_per_unit"]
    results, samples = [], []
    for horizon in horizons:
        next_start = cutoff
        errors = {name: [] for name in ("frozen", "visual_velocity", "ned_velocity")}
        rejected = 0
        for row in observed:
            start = row["stamp"]
            if start < next_start:
                continue
            end_index = int(np.searchsorted(times, start + horizon))
            previous = int(np.searchsorted(times, start - 0.5, side="right")) - 1
            if previous < 0 or end_index >= len(observed):
                continue
            end = times[end_index]
            # Endpoint references and visual velocity must have close timestamp
            # support. Do not substitute a distant post-recovery observation.
            if end - start - horizon > 0.1 or start - times[previous] > 0.6:
                continue
            next_start = end  # independent windows, including telemetry failures
            displacement = causal_displacement(velocities, start, end)
            if displacement is None:
                rejected += 1
                continue
            origin = _center(row, basis)
            reference = _center(observed[end_index], basis)
            velocity = (origin - _center(observed[previous], basis)) / (start - times[previous])
            predictions = {
                "frozen": origin,
                "visual_velocity": origin + velocity * (end - start),
                "ned_velocity": origin + transform @ displacement[[1, 0]],
            }
            residuals = {
                name: float(np.linalg.norm(xyz - reference)) for name, xyz in predictions.items()
            }
            for name, residual in residuals.items():
                errors[name].append(residual)
            samples.append(
                {
                    "start_mono": start,
                    "end_mono": float(end),
                    "horizon_s": horizon,
                    "horizontal_predictions_u": {
                        name: xyz.tolist() for name, xyz in predictions.items()
                    },
                    "reference_horizontal_u": reference.tolist(),
                    "residual_u": residuals,
                    "map_confirmed": False,
                    "arrival_evidence": False,
                }
            )
        results.append(
            {
                "horizon_s": horizon,
                "telemetry_rejected": rejected,
                "horizontal_reference_residual_u": {
                    name: {
                        "count": len(values),
                        **{
                            label: float(np.percentile(values, q)) if values else None
                            for label, q in (("p50", 50), ("p95", 95), ("max", 100))
                        },
                    }
                    for name, values in errors.items()
                },
            }
        )
    return results, samples


def run(session, out, *, horizons=(0.25, 0.5, 1.0, 2.0, 3.0, 5.0)):
    horizons = tuple(float(value) for value in horizons)
    if not horizons or any(not np.isfinite(value) or not 0.1 <= value <= 5.0 for value in horizons):
        raise ValueError("outage horizons must be finite and between 0.1 and 5 seconds")
    frames, velocities, basis, identity = load_session(session)
    if len(frames) < 2:
        raise ValueError("at least two timestamped frames are required")
    cutoff = frames[0]["stamp"] + 0.4 * (frames[-1]["stamp"] - frames[0]["stamp"])
    calibration = fit_horizontal_calibration(
        calibration_segments(frames, velocities, basis, cutoff)
    )
    report = {
        "schema": "sfm-outage-bridge-replay/v1",
        "session": str(Path(session).resolve()),
        "source_sha256": identity,
        "calibration_cutoff_mono": cutoff,
        "calibration": calibration,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): file_digest(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "定位演算法/validation/fusion_session_data.py",
            )
        },
        "production_prediction_horizon_s": 0.5,
        "formal_control_ready": False,
        "limitations": [
            "The reference is withheld visual localization, not independent ground truth.",
            "Calibration uses the first 40% only; inconsistent calibration prevents evaluation.",
            "Synthetic outages in supported views do not reproduce actual visual failure geometry.",
            "Only horizontal firmware-fused velocity is used; no raw IMU or height fusion.",
            "Receipt clocks can contain unequal transport delay; this replay does not calibrate exposure time.",
            "Horizons above 0.5 seconds are offline candidates, never flight or landing evidence.",
        ],
    }
    samples = []
    if calibration.get("ready"):
        report["evaluation"], samples = evaluate_outages(
            frames, velocities, basis, calibration, cutoff, horizons
        )
    else:
        report["evaluation"] = {"unavailable": calibration["reason"]}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (out / "outages.jsonl").open("w") as stream:
        for row in samples:
            stream.write(json.dumps(row) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.session, args.out)
    print(json.dumps({"calibration": report["calibration"], "evaluation": report["evaluation"]}))


if __name__ == "__main__":
    main()
