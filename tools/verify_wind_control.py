#!/usr/bin/env python3
"""Offline wind-drift matrix using the production controller, without aircraft IO."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import sim_route_autoflight as sim


def scenarios():
    pulses = [(0.0, 0.0, 0.0, 0.0)]
    for start in (35.0, 105.0, 175.0):
        pulses.extend(
            [
                (start, 0.0, 0.0, 0.0),
                (start + 1.0, 0.0, 0.35, 0.0),
                (start + 5.0, 0.0, 0.35, 0.0),
                (start + 6.0, 0.0, 0.0, 0.0),
            ]
        )
    reversal = (
        (0.0, 0.0, 0.3, 0.0),
        (60.0, 0.0, 0.3, 0.0),
        (62.0, 0.0, -0.3, 0.0),
        (120.0, 0.0, -0.3, 0.0),
        (122.0, 0.3, 0.0, 0.0),
    )
    return {
        "calm": ({}, ()),
        "steady_east": ({"wind_mean_mps": (0.3, 0.0, 0.0)}, ()),
        "steady_west": ({"wind_mean_mps": (-0.3, 0.0, 0.0)}, ()),
        "direction_changes": ({"wind_profile": reversal, "wind_tau_s": 0.3}, (62.0, 122.0)),
        "gusts": ({"wind_profile": tuple(pulses), "wind_tau_s": 0.3}, (41.0, 111.0, 181.0)),
        "gusts_delayed_dropout": (
            {
                "wind_profile": tuple(pulses),
                "wind_tau_s": 0.3,
                "latency_ms": 280.0,
                "telemetry_latency_ms": 250.0,
                "drop_rate": 0.08,
                "latency_jitter_ms": 40.0,
                "outage_every_s": 50.0,
                "outage_dur_s": 0.3,
            },
            (41.0, 111.0, 181.0),
        ),
    }


def score(result, params, controller, frame, recovery_times):
    samples = []
    for row in result.trace:
        target = row["target_idx"]
        if target <= 0 or row.get("true_map_u") is None:
            continue  # Initial joining error is not route cross-track error.
        position = np.asarray(frame.horizontal(np.asarray(row["true_map_u"])))
        start = np.asarray(frame.horizontal(controller.wp[target - 1]))
        end = np.asarray(frame.horizontal(controller.wp[target]))
        delta = end - start
        fraction = np.clip(
            np.dot(position - start, delta) / max(np.dot(delta, delta), 1e-12), 0.0, 1.0
        )
        error = float(np.linalg.norm(position - start - fraction * delta))
        samples.append((row["t"], error, error / controller._arrive_radius_for(target)))
    recoveries = []
    for event in recovery_times:
        if not samples or samples[-1][0] < event:
            continue
        since = None
        recovered = None
        for stamp, _error, ratio in samples:
            if stamp < event:
                continue
            since = (stamp if since is None else since) if ratio <= 1.0 else None
            if since is not None and stamp - since >= 0.5:
                recovered = round(since - event, 3)
                break
        recoveries.append({"event_s": event, "recovery_s": recovered})
    speeds = [
        row["ground_speed_mps"] for row in result.trace if row.get("ground_speed_mps") is not None
    ]
    cap = controller.cfg.max_translation_pcmd
    saturation = [max(abs(row["pcmd"][0]), abs(row["pcmd"][1])) >= cap for row in result.trace]
    aligned = [
        row["heading_time_aligned"]
        for row in result.trace
        if row.get("heading_time_aligned") is not None
    ]
    radius_p95 = float(np.percentile([row[2] for row in samples], 95)) if samples else None
    bounded = all(max(abs(row["pcmd"][0]), abs(row["pcmd"][1])) <= cap for row in result.trace)
    recovered = len(recoveries) == len(recovery_times) and all(
        row["recovery_s"] is not None and row["recovery_s"] <= 10.0 for row in recoveries
    )
    return {
        "accepted": bool(
            result.success
            and radius_p95 is not None
            and radius_p95 <= 1.0
            and bounded
            and recovered
        ),
        "completed": result.success,
        "reason": result.reason,
        "time_s": result.time_s,
        "cross_track_p95_m": float(np.percentile([row[1] for row in samples], 95))
        * params.meters_per_unit
        if samples
        else None,
        "cross_track_p95_radius": radius_p95,
        "pcmd_within_bounds": bounded,
        "wind_events_covered": len(recoveries),
        "peak_ground_speed_mps": max(speeds, default=0.0),
        "pcmd_saturation_fraction": float(np.mean(saturation)) if saturation else 0.0,
        "prediction_ticks": sum(
            bool((row.get("control_prediction") or {}).get("applied")) for row in result.trace
        ),
        "aligned_heading_fraction": float(np.mean(aligned)) if aligned else None,
        "recovery": recoveries,
    }


def run_matrix(base, out, seeds, names):
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        "定位演算法/flight_control/real_path_follow_controller.py",
        "定位演算法/flight_control/path_follow_flight.py",
        "定位演算法/flight_control/control_state_estimator.py",
        "定位演算法/flight_control/wind_observer.py",
        "控制介面程式/operator_interface/operator_autonomy.py",
        "tools/sim_route_autoflight.py",
        "tools/verify_wind_control.py",
    ]
    hashes = {
        name: hashlib.sha256((sim.REPO_ROOT / name).read_bytes()).hexdigest() for name in sources
    }
    asset_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (base.route, base.align)
        if path is not None
    }
    controller, _, frame, _ = sim.build_production_controller(
        base.route, base.align, return_to_start=base.return_to_start
    )
    rows = []
    for name in names:
        changes, recoveries = scenarios()[name]
        for seed in seeds:
            params = replace(base, seed=seed, **changes)
            result, _ = sim.run_sim(params, record_trace=True)
            metrics = score(result, params, controller, frame, recoveries)
            rows.append(
                {"scenario": name, "seed": seed, "metrics": metrics, "params": asdict(params)}
            )
            with (out / f"{name}_{seed}.jsonl").open("w") as stream:
                for sample in result.trace:
                    stream.write(json.dumps(sample) + "\n")
            print(json.dumps({"scenario": name, "seed": seed, **metrics}), flush=True)
    report = {
        "scope": "offline residual-ground-drift proxy; not aerodynamic wind or aircraft certification",
        "acceptance": "complete route; post-join horizontal cross-track p95 <= authored radius; bounded PCMD; all wind events exercised and within radius for 0.5s within 10s after event",
        "source_sha256": hashes,
        "input_sha256": asset_hashes,
        "cases": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--align", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meters-per-unit", type=float, default=15.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument("--cases", nargs="+", choices=list(scenarios()), default=list(scenarios()))
    parser.add_argument("--cruise-mps", type=float, default=0.90)
    parser.add_argument("--tau-h-s", type=float, default=0.4)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=sim.SIM_DEFAULT_DURATION_S,
        help="finite offline analysis horizon, not a production AUTO flight deadline",
    )
    parser.add_argument(
        "--speed-limit-mps",
        type=float,
        required=True,
        help="approved site AUTO speed limit; never infer it from the simulator default",
    )
    args = parser.parse_args()
    if not math.isfinite(args.speed_limit_mps) or args.speed_limit_mps <= 0:
        parser.error("--speed-limit-mps must be finite and positive")
    base = sim.SimParams(
        route=args.route,
        align=args.align,
        meters_per_unit=args.meters_per_unit,
        duration_s=args.duration_s,
        wind_sigma_mps=0.0,
        cruise_mps=args.cruise_mps,
        speed_limit_mps=args.speed_limit_mps,
        tau_h_s=args.tau_h_s,
    )
    report = run_matrix(base, args.out, args.seeds, args.cases)
    return 0 if all(row["metrics"]["accepted"] for row in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
