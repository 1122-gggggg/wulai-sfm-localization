#!/usr/bin/env python3
"""Offline acceptance of the operator's routes from a waypoint-1 start.

The plant is an approximation, not a flight clearance. Completion requires the
full waypoint sequence and return leg; tracking is checked against plant
truth after the waypoint-1 start, independently of the simulated localization estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import sim_route_autoflight as sim


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "地圖檔/場域/river_site/routes"


def evaluate(params: sim.SimParams, *, start_label: str) -> dict:
    controller, config, frame, document = sim.build_production_controller(
        params.route, params.align, return_to_start=params.return_to_start
    )
    result, info = sim.run_sim(params, record_trace=True)
    join = info["join_target_1based"] - 1
    expected = list(range(len(controller.wp)))
    errors = []
    target_order = []
    overlap = 0
    vertical_overlap = 0
    for tick in result.trace:
        roll, pitch, yaw, gaz = tick["pcmd"]
        translating = sim.adds_horizontal_speed(
            roll, pitch, tick.get("body_velocity_mps", (0.0, 0.0))
        )
        overlap += bool(yaw and translating)
        vertical_overlap += bool(gaz and translating)
        index = tick["target_idx"]
        if not target_order or target_order[-1] != index:
            target_order.append(index)
        if index <= join or tick["phase"] == "landing_hold":
            continue
        distance, _nearest, _fraction = sim.rpf.point_segment_distance(
            np.array(tick["true_map_u"]), controller.wp[index - 1], controller.wp[index]
        )
        errors.append(float(distance))
    radius = min(controller._arrive_radius_for(i) for i in expected)
    p95 = float(np.percentile(errors, 95)) if errors else 0.0
    maximum = max(errors, default=0.0)
    p95_limit = max(2.0 * radius, 4.0 * params.pos_noise_u)
    maximum_limit = max(3.0 * radius, 6.0 * params.pos_noise_u)
    end_distance = float(
        np.linalg.norm(np.array(result.trace[-1]["true_map_u"]) - controller.wp[-1])
    )
    expected_join = 0
    start_distance = (
        float(np.linalg.norm(np.array(result.trace[0]["true_map_u"]) - controller.wp[0]))
        if result.trace
        else float("inf")
    )
    # A takeoff inside waypoint 1's arrival sphere may retire it before the first
    # logged tick, or a tick later once delayed localization confirms arrival.
    accepted_orders = (
        [expected, expected[1:]]
        if start_distance <= controller._arrive_radius_for(0)
        else [expected]
    )
    sequence_ok = (
        result.waypoints_reached == expected
        and target_order in accepted_orders
        and join == expected_join
    )
    accepted = bool(
        result.success
        and sequence_ok
        and not overlap
        and not vertical_overlap
        and p95 <= p95_limit
        and maximum <= maximum_limit
        and end_distance <= controller.final_arrive_tolerance() + 3.0 * params.pos_noise_u
    )
    return {
        "route": params.route.name,
        "route_sha256": hashlib.sha256(params.route.read_bytes()).hexdigest(),
        "waypoint_count": len(document.waypoints),
        "start": start_label,
        "seed": params.seed,
        "meters_per_unit": params.meters_per_unit,
        "join_waypoint": join + 1,
        "expected_join_waypoint": expected_join + 1,
        "accepted": accepted,
        "completed": result.success,
        "reason": result.reason,
        "time_s": result.time_s,
        "speed_guard_interventions": result.speed_guard_interventions,
        "max_waypoint_no_progress_s": result.max_waypoint_no_progress_s,
        "sequence_ok": sequence_ok,
        "observed_target_order": target_order,
        "accepted_target_orders": accepted_orders,
        "waypoint_indices_reached": result.waypoints_reached,
        "expected_waypoint_indices": expected,
        "post_join_error_p95_u": p95,
        "post_join_error_max_u": maximum,
        "p95_limit_u": p95_limit,
        "maximum_limit_u": maximum_limit,
        "final_true_distance_u": end_distance,
        "yaw_translation_overlap_samples": overlap,
        "vertical_translation_overlap_samples": vertical_overlap,
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(params).items()},
    }


def evaluate_gust_recovery(params: sim.SimParams) -> dict:
    """Check transient recovery separately from the continuous-wind error bound."""
    controller, _config, _frame, _doc = sim.build_production_controller(
        params.route, params.align, return_to_start=params.return_to_start
    )
    result, info = sim.run_sim(params)
    expected = list(range(len(controller.wp)))
    recoveries = []
    for index, tick in enumerate(result.trace):
        displacement = float(np.linalg.norm(tick.get("gust_displacement_m", [0, 0, 0])))
        if displacement < 1e-9:
            continue
        target = tick["target_idx"]
        radius = controller._arrive_radius_for(target)
        recovery = None
        for future in result.trace[index:]:
            distance, _, _ = sim.rpf.point_segment_distance(
                np.array(future["true_map_u"]),
                controller.wp[max(0, target - 1)],
                controller.wp[target],
            )
            if distance <= 2 * radius or future["target_idx"] != target:
                recovery = round(future["t"] - tick["t"], 3)
                break
        recoveries.append({"t": tick["t"], "displacement_m": displacement, "recovery_s": recovery})
    overlap = 0
    for tick in result.trace:
        roll, pitch, yaw, gaz = tick["pcmd"]
        velocity = tick.get("body_velocity_mps", (0.0, 0.0))
        overlap += bool((yaw or gaz) and sim.adds_horizontal_speed(roll, pitch, velocity))
    accepted = bool(
        result.success
        and result.waypoints_reached == expected
        and not overlap
        and recoveries
        and all(item["recovery_s"] is not None and item["recovery_s"] <= 5.0 for item in recoveries)
    )
    return {
        "route": params.route.name,
        "accepted": accepted,
        "completed": result.success,
        "reason": result.reason,
        "gust_m": params.gust_m,
        "seed": params.seed,
        "meters_per_unit": params.meters_per_unit,
        "gust_every_s": params.gust_every_s,
        "time_s": result.time_s,
        "waypoints_reached": result.waypoints_reached,
        "expected_waypoint_indices": expected,
        "axis_overlap_samples": overlap,
        "recovery_limit_s": 5.0,
        "recovery_band": "2 * authored waypoint radius",
        "gusts": recoveries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one seed/scale, every start")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--localization-wait-s",
        type=float,
        required=True,
        help="explicit initial localization wait included in the 300s budget",
    )
    parser.add_argument("--flight-logs", type=Path, default=ROOT / "outputs/flight_logs")
    args = parser.parse_args(argv)
    routes = sorted(ROUTES.glob("*.json"))
    if not routes:
        parser.error(f"no preset routes in {ROUTES}")
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__).resolve(),
        Path(sim.__file__).resolve(),
        ROOT / "定位演算法/flight_control/real_path_follow_controller.py",
        ROOT / "定位演算法/flight_control/path_follow_flight.py",
        ROOT / "定位演算法/flight_control/landing_transition.py",
        ROOT / "定位演算法/flight_control/heading_fusion.py",
        ROOT / "控制介面程式/operator_interface/operator_autonomy.py",
        sim._default_align(),
    ]
    sources = [path for path in sources if path is not None]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    rows = []

    def record(params: sim.SimParams, label: str) -> None:
        params = replace(params, localization_wait_s=args.localization_wait_s)
        row = evaluate(params, start_label=label)
        rows.append(row)
        if not row["accepted"]:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        (args.out / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for route in routes:
        controller, _config, frame, doc = sim.build_production_controller(
            route, sim._default_align(), return_to_start=True
        )
        for index in range(len(doc.waypoints)):
            offset = controller.wp[index] - controller.wp[0] + 0.06 * frame.east - 0.03 * frame.up
            for scale in [5.0] if args.quick else [5.0, 10.0]:
                for seed in [7] if args.quick else [7, 13, 23]:
                    record(
                        sim.SimParams(
                            route=route,
                            align=sim._default_align(),
                            meters_per_unit=scale,
                            duration_s=300.0,
                            seed=seed,
                            start_offset_frame="raw",
                            start_offset_u=tuple(offset),
                            initial_yaw_error_deg=30.0 + 60.0 * (index % 3),
                        ),
                        f"start_near_waypoint_{index + 1}",
                    )
        if not args.quick:
            for index in sorted({0, len(doc.waypoints) // 2, len(doc.waypoints) - 1}):
                offset = controller.wp[index] - controller.wp[0] - 0.08 * frame.east
                record(
                    sim.SimParams(
                        route=route,
                        align=sim._default_align(),
                        meters_per_unit=5.0,
                        duration_s=300.0,
                        seed=41,
                        start_offset_frame="raw",
                        start_offset_u=tuple(offset),
                        initial_yaw_error_deg=170.0,
                        latency_ms=300.0,
                        pos_noise_u=0.006,
                        yaw_noise_deg=2.0,
                        drop_rate=0.05,
                        wind_sigma_mps=0.08,
                        outage_every_s=60.0,
                        outage_dur_s=0.7,
                    ),
                    f"delayed_noisy_start_near_waypoint_{index + 1}",
                )
            for index in (0, len(doc.waypoints) - 1):
                for seed in (101, 137):
                    offset = (
                        controller.wp[index]
                        - controller.wp[0]
                        + 0.09 * frame.east
                        + 0.02 * frame.up
                    )
                    record(
                        sim.SimParams(
                            route=route,
                            align=sim._default_align(),
                            meters_per_unit=7.5,
                            duration_s=300.0,
                            seed=seed,
                            start_offset_frame="raw",
                            start_offset_u=tuple(offset),
                            initial_yaw_error_deg=110.0,
                            latency_ms=200.0,
                            pos_noise_u=0.004,
                            yaw_bias_deg=5.0,
                            wind_sigma_mps=0.06,
                            drop_rate=0.03,
                        ),
                        f"held_out_start_near_waypoint_{index + 1}",
                    )
        print(
            f"{route.name}: {sum(r['accepted'] for r in rows if r['route'] == route.name)}/"
            f"{sum(r['route'] == route.name for r in rows)} accepted",
            flush=True,
        )

    _record_session_starts(args, routes, record)
    gust_rows = []
    if not args.quick:
        for route in routes:
            for gust_m in (0.2, 0.5):
                gust_rows.append(
                    evaluate_gust_recovery(
                        sim.SimParams(
                            route=route,
                            align=sim._default_align(),
                            meters_per_unit=5.0,
                            duration_s=300.0,
                            seed=53,
                            gust_m=gust_m,
                            gust_every_s=20.0,
                            localization_wait_s=args.localization_wait_s,
                        )
                    )
                )
        (args.out / "wind_results.json").write_text(
            json.dumps(gust_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"WIND {sum(row['accepted'] for row in gust_rows)}/{len(gust_rows)} accepted",
            flush=True,
        )
    accepted = sum(row["accepted"] for row in rows)
    unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in source_hashes.items()
    )
    (args.out / "verification_manifest.json").write_text(
        json.dumps(
            {
                "cases": len(rows),
                "accepted": accepted,
                "sources_unchanged_during_run": unchanged,
                "wind_cases": len(gust_rows),
                "wind_accepted": sum(row["accepted"] for row in gust_rows),
                "source_sha256": source_hashes,
                "flight_logs": str(args.flight_logs.resolve()),
                "verification": "offline controller and production safety checks; simulated plant",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"TOTAL {accepted}/{len(rows)} accepted", flush=True)
    if not unchanged:
        print(
            "Source files changed during verification; rerun before using these results.",
            flush=True,
        )
    return (
        0
        if accepted == len(rows) and all(row["accepted"] for row in gust_rows) and unchanged
        else 1
    )


def _record_session_starts(args, routes, record):
    if not args.quick:
        for session in sorted(args.flight_logs.glob("session_20260914T08*")):
            awaiting = False
            count = 0
            for line in (session / "localization.jsonl").open(encoding="utf-8"):
                tick = json.loads(line)
                if tick["event"] == "auto_route_plan":
                    count += 1
                    awaiting = True
                    matches = []
                    for candidate in routes:
                        candidate_controller, _cfg, _frame, _doc = sim.build_production_controller(
                            candidate, sim._default_align(), return_to_start=True
                        )
                        points = np.asarray(candidate_controller.wp)
                        recorded = np.asarray(tick["waypoints_u"])
                        if points.shape == recorded.shape and np.allclose(
                            points, recorded, atol=5.1e-5
                        ):
                            matches.append(candidate)
                    if len(matches) != 1:
                        raise ValueError(f"cannot uniquely match recorded AUTO plan in {session}")
                    route = matches[0]
                elif awaiting and tick["event"] == "auto_route_tick" and tick.get("pose_u"):
                    awaiting = False
                    controller, _cfg, frame, _doc = sim.build_production_controller(
                        route, sim._default_align(), return_to_start=True
                    )
                    yaw_error = tick["heading_deg"] - math.degrees(
                        frame.heading(controller.wp[1] - controller.wp[0])
                    )
                    record(
                        sim.SimParams(
                            route=route,
                            align=sim._default_align(),
                            meters_per_unit=5.0,
                            duration_s=300.0,
                            seed=7,
                            start_offset_frame="raw",
                            start_offset_u=tuple(np.array(tick["pose_u"]) - controller.wp[0]),
                            initial_yaw_error_deg=yaw_error,
                        ),
                        f"{session.name}_auto_{count}",
                    )


if __name__ == "__main__":
    raise SystemExit(main())
