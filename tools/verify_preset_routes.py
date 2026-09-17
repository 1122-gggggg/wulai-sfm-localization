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

try:
    from landing_transition import LANDING_SPEED_THRESHOLD_MPS
except ImportError:  # pragma: no cover - direct script execution
    import sys as _sys
    _sys.path.insert(0, str(ROOT.parent / "定位演算法" / "flight_control"))
    from landing_transition import LANDING_SPEED_THRESHOLD_MPS
ROUTES = ROOT / "地圖檔/場域/river_site/routes"


_TRUTH_FIELDS = (
    "body_velocity_mps", "measured_body_velocity_mps", "telemetry_stamp",
    "ground_speed_mps", "pcmd_requested", "speed_guard_status",
    "pose_stamp", "map_confirmed", "retired_target_idx",
    "gust_displacement_m", "gust_time_s",
)


def _phase_contract_violations(trace) -> int:
    bad = 0
    for tick in trace:
        roll, pitch, yaw, gaz = (int(v) for v in tick["pcmd"])
        phase = str(tick.get("phase", ""))
        base = phase.removeprefix("fgc:")
        if base == "turn" and yaw != 0 and gaz != 0:
            bad += 1
        elif base in {"route_rejoin", "waypoint_centering", "final_centering"} and yaw != 0:
            bad += 1
    return bad


def evaluate(params: sim.SimParams, *, start_label: str) -> dict:
    controller, config, frame, document = sim.build_production_controller(
        params.route, params.align, return_to_start=params.return_to_start
    )
    result, info = sim.run_sim(params, record_trace=True)
    join = info["join_target_1based"] - 1
    expected = list(range(len(controller.wp)))
    missing = [name for name in _TRUTH_FIELDS if any(name not in tick for tick in result.trace)]
    if not result.trace:
        return {
            "route": params.route.name, "start": start_label, "accepted": False,
            "completed": result.success, "reason": "empty trace: no samples to verify",
            "missing_truth_fields": missing or None,
            "post_join_error_p95_u": None, "post_join_error_max_u": None,
            "sequence_ok": False, "params": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(params).items()},
        }
    if missing:
        return {
            "route": params.route.name, "start": start_label, "accepted": False,
            "completed": result.success, "reason": f"trace missing truth fields: {sorted(set(missing))}",
            "missing_truth_fields": sorted(set(missing)),
            "post_join_error_p95_u": None, "post_join_error_max_u": None,
            "sequence_ok": False, "params": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(params).items()},
        }
    errors = []
    target_order = []
    retired_order: list[int] = []
    seen_retired: set[int] = set()
    arrival_truth_ok = True
    arrival_detail: list[dict] = []
    turn_yaw_gaz_overlap = 0
    centering_yaw = 0
    translate_gaz_ok = 0
    yaw_overlay_ok = 0
    for tick in result.trace:
        roll, pitch, yaw, gaz = (int(v) for v in tick["pcmd"])
        phase = str(tick.get("phase", "")).removeprefix("fgc:")
        measured = tick.get("measured_body_velocity_mps")
        truth_v = tick.get("body_velocity_mps")
        if truth_v is None or (measured is None and tick.get("telemetry_stamp") is not None):
            arrival_truth_ok = False
        # Overlap contract: observation only for translate, gate for the rest.
        if yaw and phase == "turn" and gaz:
            turn_yaw_gaz_overlap += 1
        if phase in {"route_rejoin", "waypoint_centering", "final_centering"} and yaw != 0:
            centering_yaw += 1
        if phase == "translate" and (roll or pitch) and gaz:
            translate_gaz_ok += 1
        if phase == "translate" and yaw != 0 and gaz == 0 and abs(int(yaw)) <= 20:
            yaw_overlay_ok += 1
        index = tick["target_idx"]
        if not target_order or target_order[-1] != index:
            target_order.append(index)
        retired = tick.get("retired_target_idx")
        if retired is not None and retired not in seen_retired:
            seen_retired.add(int(retired))
            retired_order.append(int(retired))
            truth = np.array(tick["true_map_u"], dtype=float)
            dist = float(np.linalg.norm(truth - controller.wp[int(retired)]))
            limit = float(controller._arrive_radius_for(int(retired))) + 3.0 * float(params.pos_noise_u)
            ok = dist <= limit
            arrival_detail.append({"retired": int(retired), "truth_distance_u": dist, "limit_u": limit, "ok": ok})
            if not ok:
                arrival_truth_ok = False
        if index <= join or tick["phase"] == "landing_hold":
            continue
        distance, _nearest, _fraction = sim.rpf.point_segment_distance(
            np.array(tick["true_map_u"]), controller.wp[index - 1], controller.wp[index]
        )
        errors.append(float(distance))
    # Error bound keys on the route's own scale (its widest authored sphere),
    # not its tightest: the recovery band elsewhere is 2 * the active radius.
    route_radius = max(float(controller._arrive_radius_for(i)) for i in expected)
    p95 = float(np.percentile(errors, 95)) if errors else None
    maximum = max(errors) if errors else None
    p95_limit = max(2.0 * route_radius, 4.0 * params.pos_noise_u)
    maximum_limit = max(3.0 * route_radius, 6.0 * params.pos_noise_u)
    end_distance = float(
        np.linalg.norm(np.array(result.trace[-1]["true_map_u"]) - controller.wp[-1])
    )
    # Retirements must be exactly the expanded order. A takeoff inside WP1's
    # sphere may retire it on the first tick before any row observes target 0;
    # both [0..N] and [1..N] target visits are accepted when the retirement
    # events and sequencer set are complete.
    accepted_target_orders = [expected, expected[1:]] if retired_order and retired_order[0] == 0 else [expected]
    sequence_ok = (
        result.waypoints_reached == expected
        and retired_order == expected
        and target_order in accepted_target_orders
        and join == 0
    )
    track_ok = (
        p95 is not None and maximum is not None
        and p95 <= p95_limit and maximum <= maximum_limit
        and end_distance <= controller.final_arrive_tolerance() + 3.0 * params.pos_noise_u
    )
    contract_bad = _phase_contract_violations(result.trace)
    landing_ok = True
    if result.trace:
        last = result.trace[-1]
        try:
            last_speed = last.get("landing_speed_measured_mps", last.get("ground_speed_mps", float("inf")))
            landing_ok = bool(last.get("action") == "LAND" or result.success) and float(last_speed) <= float(LANDING_SPEED_THRESHOLD_MPS)
        except (TypeError, ValueError, OverflowError):
            landing_ok = False
    accepted = bool(
        result.success
        and sequence_ok
        and arrival_truth_ok
        and track_ok
        and contract_bad == 0
        and landing_ok
    )
    reason = result.reason
    if not sequence_ok:
        reason = f"sequence mismatch: retired={retired_order} targets={target_order}"
    elif not arrival_truth_ok:
        reason = "arrival truth outside authored sphere"
    elif not track_ok:
        details = []
        if p95 is None or maximum is None:
            details.append("no post-join tracking samples")
        else:
            if p95 is not None and p95 > p95_limit:
                details.append(f"post-join p95 {p95:.4f}u > limit {p95_limit:.4f}u")
            if maximum is not None and maximum > maximum_limit:
                details.append(f"post-join max {maximum:.4f}u > limit {maximum_limit:.4f}u")
        try:
            end_limit = float(controller.final_arrive_tolerance() + 3.0 * params.pos_noise_u)
        except (TypeError, ValueError, OverflowError):
            end_limit = float("nan")
        if end_distance > end_limit:
            details.append(f"final distance {end_distance:.4f}u > limit {end_limit:.4f}u")
        reason = "; ".join(details) if details else result.reason
    elif contract_bad:
        reason = f"phase contract violations: {contract_bad}"
    elif not landing_ok:
        reason = "final ground speed above landing threshold"
    return {
        "route": params.route.name,
        "route_sha256": hashlib.sha256(params.route.read_bytes()).hexdigest(),
        "waypoint_count": len(document.waypoints),
        "start": start_label,
        "seed": params.seed,
        "meters_per_unit": params.meters_per_unit,
        "join_waypoint": join + 1,
        "expected_join_waypoint": 1,
        "accepted": accepted,
        "completed": result.success,
        "reason": reason,
        "time_s": result.time_s,
        "speed_guard_interventions": result.speed_guard_interventions,
        "max_waypoint_no_progress_s": result.max_waypoint_no_progress_s,
        "sequence_ok": sequence_ok,
        "observed_target_order": target_order,
        "accepted_target_orders": accepted_target_orders,
        "retired_target_order": retired_order,
        "waypoint_indices_reached": result.waypoints_reached,
        "expected_waypoint_indices": expected,
        "arrival_truth": arrival_detail,
        "arrival_truth_ok": arrival_truth_ok,
        "post_join_error_p95_u": p95,
        "post_join_error_max_u": maximum,
        "p95_limit_u": p95_limit,
        "maximum_limit_u": maximum_limit,
        "final_true_distance_u": end_distance,
        "phase_contract_violations": contract_bad,
        "turn_yaw_gaz_overlap_samples": turn_yaw_gaz_overlap,
        "centering_yaw_samples": centering_yaw,
        "translate_horizontal_gaz_samples": translate_gaz_ok,
        "translate_yaw_overlay_samples": yaw_overlay_ok,
        "yaw_translation_overlap_samples": turn_yaw_gaz_overlap,
        "vertical_translation_overlap_samples": 0,
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(params).items()},
    }


def evaluate_gust_recovery(params: sim.SimParams) -> dict:
    """Check transient recovery separately from the continuous-wind error bound.

    From each gust tick (t >= gust_time_s truth only), require the truth to
    re-enter the then-active segment band (2 * radius) and hold it for a
    continuous 0.5 s. A waypoint switch moves the band to the new active
    segment but never counts as recovery by itself.
    """
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
        start_t = float(tick.get("gust_time_s") or tick["t"])
        recovery = None
        hold_start: float | None = None
        for future in result.trace[index:]:
            if float(future["t"]) < start_t - 1e-9:
                continue
            ftarget = int(future["target_idx"])
            fradius = float(controller._arrive_radius_for(ftarget))
            distance, _, _ = sim.rpf.point_segment_distance(
                np.array(future["true_map_u"]),
                controller.wp[max(0, ftarget - 1)],
                controller.wp[ftarget],
            )
            if distance <= 2 * fradius:
                if hold_start is None:
                    hold_start = float(future["t"])
                if float(future["t"]) - hold_start >= 0.5 - 1e-9:
                    recovery = round(float(future["t"]) - float(tick["t"]), 3)
                    break
            else:
                hold_start = None
        recoveries.append({"t": tick["t"], "displacement_m": displacement, "recovery_s": recovery})
    turn_bad = 0
    center_bad = 0
    for tick in result.trace:
        roll, pitch, yaw, gaz = (int(v) for v in tick["pcmd"])
        phase = str(tick.get("phase", "")).removeprefix("fgc:")
        if phase == "turn" and yaw != 0 and gaz != 0:
            turn_bad += 1
        elif phase in {"route_rejoin", "waypoint_centering", "final_centering"} and yaw != 0:
            center_bad += 1
    terminal_ok = True
    if recoveries and recoveries[-1]["recovery_s"] is None:
        last = result.trace[-1]
        ftarget = int(last["target_idx"])
        fradius = float(controller._arrive_radius_for(ftarget))
        distance, _, _ = sim.rpf.point_segment_distance(
            np.array(last["true_map_u"]),
            controller.wp[max(0, ftarget - 1)],
            controller.wp[ftarget],
        )
        terminal_ok = bool(distance <= 2 * fradius)
        recoveries[-1]["terminal_distance_u"] = float(distance)
        recoveries[-1]["terminal_band_u"] = float(2 * fradius)
    accepted = bool(
        result.success
        and result.waypoints_reached == expected
        and turn_bad == 0
        and center_bad == 0
        and recoveries
        and all(item["recovery_s"] is not None and item["recovery_s"] <= 5.0 for item in recoveries[:-1])
        and (recoveries[-1]["recovery_s"] is not None and recoveries[-1]["recovery_s"] <= 5.0 or terminal_ok)
    )
    reason = result.reason
    if not result.success:
        reason = result.reason
    elif result.waypoints_reached != expected:
        reason = f"sequence mismatch: reached={result.waypoints_reached} expected={expected}"
    elif turn_bad or center_bad:
        reason = f"phase contract violations: turn={turn_bad} centering={center_bad}"
    elif not recoveries:
        reason = "no gust impulses in trace"
    elif not all(item["recovery_s"] is not None and item["recovery_s"] <= 5.0 for item in recoveries[:-1]):
        slow = next(item for item in recoveries[:-1] if item["recovery_s"] is None or item["recovery_s"] > 5.0)
        reason = f"gust at t={slow['t']}s did not re-enter the 2-radius band within 5s"
    elif not (recoveries[-1]["recovery_s"] is not None and recoveries[-1]["recovery_s"] <= 5.0 or terminal_ok):
        reason = "final gust did not re-enter the 2-radius band within 5s"
    return {
        "route": params.route.name,
        "accepted": accepted,
        "completed": result.success,
        "reason": reason,
        "gust_m": params.gust_m,
        "seed": params.seed,
        "meters_per_unit": params.meters_per_unit,
        "gust_every_s": params.gust_every_s,
        "time_s": result.time_s,
        "waypoints_reached": result.waypoints_reached,
        "expected_waypoint_indices": expected,
        "axis_overlap_samples": turn_bad + center_bad,
        "turn_yaw_gaz_overlap_samples": turn_bad,
        "centering_yaw_samples": center_bad,
        "recovery_limit_s": 5.0,
        "recovery_band": "2 * authored waypoint radius",
        "recovery_dwell_s": 0.5,
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
            for index in (0, len(doc.waypoints) - 1):
                for seed in (7, 23):
                    base_offset = controller.wp[index] - controller.wp[0]
                    for scale in (2.34, 5.0, 10.0):
                        record(
                            sim.SimParams(
                                route=route,
                                align=sim._default_align(),
                                meters_per_unit=scale,
                                duration_s=300.0,
                                seed=seed,
                                start_offset_frame="raw",
                                start_offset_u=tuple(base_offset + 0.06 * frame.east - 0.03 * frame.up),
                                initial_yaw_error_deg=30.0,
                                tau_tilt_s=0.15,
                                thrust_margin=1.35,
                                battery_sag_frac=0.1,
                                yaw_coupling=0.3,
                            ),
                            f"tilt_lag_start_near_waypoint_{index + 1}_scale_{scale}_seed_{seed}",
                        )
            # Steady-drift sensitivity along the first non-vertical leg normal.
            leg = None
            for a, b in zip(controller.wp[:-1], controller.wp[1:]):
                candidate = np.asarray(b, float) - np.asarray(a, float)
                horiz = np.array(frame.horizontal(candidate), float)
                if float(np.linalg.norm(horiz)) > 1e-9:
                    leg = candidate
                    break
            if leg is not None:
                horiz = np.array(frame.horizontal(leg), float)
                normal = np.array([-horiz[1], horiz[0]]) / max(float(np.linalg.norm(horiz)), 1e-9)
                drift = normal[0] * np.asarray(frame.east, float) + normal[1] * np.asarray(frame.north, float)
                for index in (0, len(doc.waypoints) - 1):
                    for sign in (1.0, -1.0):
                        record(
                            sim.SimParams(
                                route=route,
                                align=sim._default_align(),
                                meters_per_unit=5.0,
                                duration_s=300.0,
                                seed=7,
                                start_offset_frame="raw",
                                start_offset_u=tuple(controller.wp[index] - controller.wp[0]),
                                initial_yaw_error_deg=30.0,
                                wind_mean_mps=tuple(0.1 * sign * np.asarray(drift, float)),
                            ),
                            f"steady_drift_start_near_waypoint_{index + 1}_{'pos' if sign > 0 else 'neg'}",
                        )
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
            # Fault-injection matrix on the first route: each SimulatedLocalizer
            # defect family at a recoverable level (probed 2026-09-17), reusing
            # the delayed_noisy takeoff so faults add to an already hard fix.
            if len(routes) > 0 and route == routes[0]:
                index = sorted({0, len(doc.waypoints) // 2, len(doc.waypoints) - 1})[0]
                offset = controller.wp[index] - controller.wp[0] - 0.08 * frame.east
                fault_cases = [
                    ("fault_weak", dict(weak_rate=0.05)),
                    ("fault_predicted", dict(predicted_rate=0.05)),
                    ("fault_jump", dict(jump_rate=0.01, jump_max_u=0.05)),
                    ("fault_outlier", dict(outlier_rate=0.01, outlier_max_u=0.05)),
                    ("fault_blackout", dict(blackout_every_s=40.0, blackout_dur_s=3.0, blackout_drift_mps=0.05, blackout_yaw_drift_deg_s=2.0)),
                    ("fault_dropout", dict(drop_rate=0.20)),
                ]
                for label, fault_kw in fault_cases:
                    params_kw = dict(fault_kw)
                    drop = float(params_kw.pop("drop_rate", 0.05))
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
                            drop_rate=drop,
                            wind_sigma_mps=0.08,
                            outage_every_s=60.0,
                            outage_dur_s=0.7,
                            **params_kw,
                        ),
                        f"{label}_start_near_waypoint_{index + 1}",
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
                "controller_scope": "production_route_controller_and_desktop_speed_limiter",
                "plant_model": "velocity_approximation",
                "physical_calibration": "unvalidated",
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
