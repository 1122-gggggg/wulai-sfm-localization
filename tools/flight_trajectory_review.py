#!/usr/bin/env python3
"""Export an offline AUTO trajectory replay, CSV samples and route snapshots."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

from autoflight_error_report import read_jsonl, _percentile


def xyz(value):
    if isinstance(value, dict):
        value = [value.get(k) for k in ("x", "y", "z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        point = [float(v) for v in value[:3]]
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(v) for v in point) else None


def moment(row):
    return float(row.get("t_mono_ns", 0)) / 1e9 or float(row.get("t", 0))


def segment_error(point, start, end):
    delta = [b - a for a, b in zip(start, end)]
    length2 = sum(v * v for v in delta)
    fraction = (
        max(0.0, min(1.0, sum((p - a) * d for p, a, d in zip(point, start, delta)) / length2))
        if length2
        else 0.0
    )
    offset = [p - (a + fraction * d) for p, a, d in zip(point, start, delta)]
    return math.sqrt(sum(v * v for v in offset))


def quality(row):
    if (
        row.get("imu_bridge")
        or row.get("direct_status") == "IMU_BRIDGE"
        or row.get("pose_status") == "PREDICTED_ONLY"
        or row.get("position_observed") is False
    ):
        return "IMU／預測"
    if row.get("map_pose_confirmed") is True or (
        row.get("mode") == "TRACK" and not row.get("confidence_low")
    ):
        return "地圖定位"
    return "弱定位／未確認"


def build_replay(session: Path):
    archive = session / "trajectory.jsonl"
    paths = (
        [archive]
        if archive.is_file()
        else [session / "localization.jsonl", session / "telemetry.jsonl"]
    )
    rows = sorted(
        (
            r
            for path in paths
            for r in read_jsonl(path)
            if r.get("event")
            in {"auto_route_plan", "auto_route_tick", "pose_result", "autonomy_event"}
        ),
        key=moment,
    )
    runs = {}
    poses = [r for r in rows if r["event"] == "pose_result"]
    pose_times = [moment(r) for r in poses]
    current = None
    for row in rows:
        if row["event"] == "pose_result":
            continue
        run_id = row.get("auto_run_id")
        if not run_id and row["event"] == "auto_route_plan":
            run_id = f"legacy-{len(runs) + 1}"
        run_id = run_id or current
        if run_id is None:
            continue
        current = run_id
        run = runs.setdefault(run_id, {"id": run_id, "plan": None, "ticks": [], "events": []})
        if row["event"] == "auto_route_plan":
            run["plan"] = row
        elif row["event"] == "auto_route_tick":
            run["ticks"].append(row)
        else:
            run["events"].append(row)
    result = {
        "session": session.name,
        "units": "map units",
        "source": [p.name for p in paths],
        "runs": [],
    }
    for run in runs.values():
        boundaries = run["ticks"] + run["events"]
        if not boundaries:
            continue
        start, end = min(map(moment, boundaries)), max(map(moment, boundaries))
        plan = run["plan"] or {}
        route = [xyz(p) for p in plan.get("waypoints_u", [])]
        if any(p is None for p in route):
            route = []
        frame = plan.get("control_config", {}).get("map_frame", {})
        basis = [xyz(frame.get(k)) for k in ("east", "north", "up")]

        def project(p):
            return [sum(a * b for a, b in zip(p, axis)) for axis in basis] if all(basis) else p

        samples = []
        # Route-loop samples retain the exact target and pose used by control.
        # Pose results fill BOOT / post-route portions that have no route ticks.
        ticks = run["ticks"]
        first = moment(ticks[0]) if ticks else float("inf")
        last = moment(ticks[-1]) if ticks else float("inf")
        outside = [
            p
            for p in poses[
                bisect.bisect_left(pose_times, start) : bisect.bisect_right(pose_times, end)
            ]
            if moment(p) < first or moment(p) > last
        ]
        for row in sorted(ticks + outside, key=moment):
            p = xyz(row.get("pose_u", row.get("pose")))
            if row.get("success") is False or (row.get("pose_age") or 0) > 0.5:
                p = None
            index = row.get("target_index")
            error = None
            if p is not None and isinstance(index, int) and 0 <= index < len(route):
                initial = plan.get("target_index", 0)
                begin = route[index] if index == initial else route[max(0, index - 1)]
                error = segment_error(p, begin, route[index])
            samples.append(
                {
                    "t": moment(row) - start,
                    "utc": row.get("t_utc"),
                    "capture_mono": row.get("pose_stamp", row.get("source_frame_stamp_mono")),
                    "xyz": p,
                    "view": project(p) if p else None,
                    "quality": quality(row),
                    "target": index,
                    "error_u": error,
                    "phase": row.get("pcmd_phase", "BOOT／等待"),
                    "blocked": row.get("blocked"),
                    "reason": row.get("reason"),
                    "speed_mps": row.get("ground_speed_mps"),
                    "battery_pct": row.get("battery_pct"),
                }
            )
        errors = sorted(s["error_u"] for s in samples if s["error_u"] is not None)
        result["runs"].append(
            {
                "id": run["id"],
                "plan": plan,
                "route": [project(p) for p in route],
                "axes": ["East", "North", "Up"] if all(basis) else ["X", "Y", "Z"],
                "samples": samples,
                "events": run["events"],
                "stats": {
                    "samples": len(errors),
                    "missing": sum(s["xyz"] is None for s in samples),
                    "mean_u": sum(errors) / len(errors) if errors else None,
                    "p95_u": _percentile(errors, 95),
                    "max_u": max(errors) if errors else None,
                },
            }
        )
    return result


def export_review(session: Path, output: Path):
    result = build_replay(session)
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output / "trajectory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "auto_run_id",
                "elapsed_s",
                "utc",
                "capture_mono",
                "x_u",
                "y_u",
                "z_u",
                "source",
                "target_index",
                "active_segment_error_u",
                "phase",
                "speed_mps",
                "battery_pct",
            ]
        )
        for run in result["runs"]:
            for s in run["samples"]:
                writer.writerow(
                    [
                        run["id"],
                        s["t"],
                        s["utc"],
                        s["capture_mono"],
                        *(s["xyz"] or [None] * 3),
                        s["quality"],
                        s["target"],
                        s["error_u"],
                        s["phase"],
                        s["speed_mps"],
                        s["battery_pct"],
                    ]
                )
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("flight_trajectory_review.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace("__TRAJECTORY_DATA__", payload), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.session.is_dir():
        parser.error("session directory does not exist")
    result = export_review(args.session, args.out)
    print(f"{len(result['runs'])} AUTO runs: {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
