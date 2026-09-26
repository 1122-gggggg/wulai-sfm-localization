#!/usr/bin/env python3
"""Identify a delayed first-order PCMD-to-fused-NED-speed response offline."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter

REPO_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_ROOT = REPO_ROOT / "定位演算法" / "flight_control"
if str(FLIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(FLIGHT_ROOT))
from wind_observer import ResidualDriftObserver  # noqa: E402


SCOPE = "closed-loop fused telemetry; host receipt clocks; not an aircraft certification"
GRID_S = 0.01


def load_ticks(path):
    ticks = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "auto_route_tick":
                continue
            try:
                pcmd = row["pcmd"]
                if not isinstance(pcmd, list) or len(pcmd) != 4:
                    continue
                values = [float(v) for v in pcmd]
                yaw = float(row["att_yaw"])
                north = float(row["speed_north_mps"])
                east = float(row["speed_east_mps"])
                speed_stamp = float(row["ground_speed_mono_ns"]) * 1e-9
                command_stamp = float(row.get("command_authorized_mono_ns") or row["t_mono_ns"]) * 1e-9
                tick_stamp = float(row["t_mono_ns"]) * 1e-9
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(v) for v in (*values, yaw, north, east, speed_stamp, command_stamp, tick_stamp)):
                continue
            ticks.append((tick_stamp, command_stamp, speed_stamp, yaw, values, north, east))
    return sorted(ticks)


def build_segments(ticks):
    commands = []
    velocities = {}
    for tick_stamp, command_stamp, speed_stamp, yaw, pcmd, north, east in ticks:
        forward, right = pcmd[1], pcmd[0]
        commands.append((command_stamp, math.cos(yaw) * forward - math.sin(yaw) * right,
                         math.sin(yaw) * forward + math.cos(yaw) * right))
        velocities.setdefault(speed_stamp, (north, east))
    commands.sort()
    speeds = sorted((stamp, *value) for stamp, value in velocities.items())
    segments = []
    current = []
    for command in commands:
        if current and command[0] - current[-1][0] > 0.5:
            segments.append(current)
            current = []
        current.append(command)
    if current:
        segments.append(current)
    prepared = []
    for segment in segments:
        start, end = segment[0][0], segment[-1][0]
        usable = [row for row in speeds if start + 3.0 <= row[0] <= end]
        if not usable:
            continue
        grid = start + np.arange(int((end - start) / GRID_S) + 1) * GRID_S
        indices = np.clip(np.rint((np.asarray([row[0] for row in usable]) - start) / GRID_S).astype(int), 0, len(grid) - 1)
        prepared.append((np.asarray(segment), grid, indices, np.asarray([row[1:] for row in usable])))
    return speeds, segments, prepared


def fit_response(prepared):
    measured = np.concatenate([part[3] for part in prepared])
    mean = measured.mean(axis=0)
    centered = measured - mean
    total = float(np.sum(centered * centered))
    candidates = []
    for delay_step in range(11):
        delay = delay_step * 0.1
        inputs = []
        for commands, grid, indices, _ in prepared:
            ix = np.searchsorted(commands[:, 0], grid - delay, side="right") - 1
            signal = np.zeros((len(grid), 2))
            valid = ix >= 0
            signal[valid] = commands[ix[valid], 1:3]
            inputs.append((signal, indices))
        for tau_step in range(1, 16):
            tau = tau_step * 0.2
            alpha = 1 - math.exp(-GRID_S / tau)
            filtered = np.concatenate([
                lfilter([0, alpha], [1, -(1 - alpha)], signal, axis=0)[indices]
                for signal, indices in inputs
            ])
            avg = filtered.mean(axis=0)
            centered_input = filtered - avg
            denom = float(np.sum(centered_input * centered_input))
            if denom <= 1e-12 or total <= 1e-12:
                continue
            gain = float(np.sum(centered_input * centered) / denom)
            bias = mean - gain * avg
            residual = measured - (gain * filtered + bias)
            candidates.append({"delay_s": round(delay, 2), "tau_s": round(tau, 2),
                               "gain_mps_per_pct": gain, "bias_ned_mps": bias.tolist(),
                               "r2": 1 - float(np.sum(residual * residual)) / total})
    return sorted(candidates, key=lambda row: row["r2"], reverse=True)


def replay_observer(ticks):
    observer = ResidualDriftObserver()
    squares = np.zeros(2)
    low = np.full(2, math.inf)
    high = np.full(2, -math.inf)
    innovations = ready = resets = 0
    last_speed_stamp = -math.inf
    for now, command_stamp, speed_stamp, yaw, pcmd, north, east in ticks:
        if speed_stamp > last_speed_stamp:
            previous_samples = observer.samples
            observer.observe_velocities([(speed_stamp, north, east)])
            last_speed_stamp = speed_stamp
            if previous_samples and not observer.samples and observer.last_reset == "diverged":
                resets += 1
            if observer.innovation is not None:
                squares += observer.innovation * observer.innovation
                innovations += 1
            low = np.minimum(low, observer.drift)
            high = np.maximum(high, observer.drift)
        observer.observe_command(pcmd, yaw, command_stamp)
        _, reason = observer.feedforward(now, yaw, 50)
        ready += reason == "ready"
    return {"innovation_rms_ned_mps": np.sqrt(squares / max(innovations, 1)).tolist(),
            "drift_min_ned_mps": np.where(np.isfinite(low), low, 0).tolist(),
            "drift_max_ned_mps": np.where(np.isfinite(high), high, 0).tolist(),
            "divergence_resets": resets, "ready_fraction": ready / len(ticks)}


def identify(path):
    ticks = load_ticks(path)
    if len(ticks) < 200:
        raise ValueError("insufficient auto_route_tick rows")
    speeds, segments, prepared = build_segments(ticks)
    if not prepared or sum(len(part[3]) for part in prepared) < 10:
        raise ValueError("insufficient auto_route_tick rows")
    candidates = fit_response(prepared)
    if not candidates:
        raise ValueError("insufficient auto_route_tick rows")
    return {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(ticks), "velocity_samples": len(speeds), "segments": len(segments),
            "best": candidates[0], "top5": candidates[:5],
            "observer_replay": replay_observer(ticks), "scope": SCOPE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = identify(args.trajectory)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
