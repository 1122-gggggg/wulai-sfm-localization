#!/usr/bin/env python3
"""Sphinx ANAFI path-convergence Monte Carlo harness.

SPHINX ANAFI SIMULATION ONLY. This never flies real hardware: real ANAFI /
SkyController IPs are refused (telemetry_sources.check_simulator_ip).

Backends:
  --backend sphinx     Parrot Sphinx ANAFI via Olympe (firmware-fused
                       telemetry; no independent simulator ground truth).
  --backend kinematic  pure-Python first-order ANAFI approximation (NOT
                       Sphinx physics; used for wide Monte Carlo sweeps and
                       for running without a simulator). Results are labeled.

Scale: everything this harness measures is Sphinx/simulation METERS.
The real monocular SfM/GLOMAP map has arbitrary units on ALL axes (height
included); no threshold below transfers to the real map without a calibrated
scale factor (see scale_utils.py and --map-units-per-meter).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anafi_profile import ANAFI_PROFILE
from controllers import (ABORT_OR_MANUAL, COMPLETED, ALGORITHM_ORDER,
                         make_controller)
from metrics import aggregate_trials, compute_trial_metrics, rank_algorithms
from route_geometry import (RouteModel, build_route,
                            sample_start_offset, wrap_angle)
from scale_utils import NO_SCALE_WARNING, ScaleContext, provisional_map_unit_thresholds
from telemetry_sources import (SPHINX_IP_DEFAULT, HeadingEstimator, KinematicAnafi,
                               PerturbationConfig, PerturbedPoseSource,
                               check_simulator_ip, print_sim_banner)

CTRL_HZ = 20.0
DT = 1.0 / CTRL_HZ
SPHINX_START_POSITION_TOL_M = 0.75
SPHINX_START_YAW_TOL_DEG = 15.0

PERTURBATION_SUITE = [
    PerturbationConfig(),
    PerturbationConfig(yaw_bias_deg=5.0),
    PerturbationConfig(yaw_bias_deg=10.0),
    PerturbationConfig(yaw_bias_deg=15.0),
    PerturbationConfig(yaw_bias_deg=30.0),
    PerturbationConfig(pose_noise_m=0.1),
    PerturbationConfig(pose_noise_m=0.2),
    PerturbationConfig(telemetry_delay_ms=100.0),
    PerturbationConfig(telemetry_delay_ms=200.0),
    PerturbationConfig(telemetry_drop_rate=0.05),
]

YAW_SWEEP_DEG = [0, 5, -5, 10, -10, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90]


# ---------------------------------------------------------------------------
# Shared trial loop (kinematic and sphinx call this with different hooks)

def run_trial(*, ctrl, route: RouteModel, get_pose, get_yaw, get_vel, send_pcmd,
              now_fn, advance_fn, duration_s: float, yaw_sign: int,
              hard_stop_ct_m: float, trial_meta: dict, tick_sink) -> list[dict]:
    """One convergence trial. Returns the tick list (also streamed to tick_sink).
    advance_fn(dt) progresses sim time (kinematic) or sleeps (sphinx)."""
    est = HeadingEstimator()
    t0 = now_fn()
    seeded = False
    prev_yaw, prev_t = None, None
    terminal_since = None
    ticks: list[dict] = []
    while True:
        now = now_fn()
        if now - t0 > duration_s:
            break
        pose = get_pose(now)
        fused_yaw = get_yaw(now)
        if pose is not None:
            if not seeded:
                # operator premise: the nose roughly faces the ROUTE direction
                # at takeoff (not the rejoin goal) -> seed with the segment
                # heading; motion refinement absorbs the residual yaw error.
                seg_head = route.segment_heading(route.project(pose.xyz).seg_index)
                goal = pose.xyz + np.array([math.cos(seg_head), 0.0, math.sin(seg_head)])
                est.seed_from_path(pose.xyz, goal, fused_yaw)
                seeded = True
            if getattr(ctrl, "refine_heading_from_motion", True):
                est.update(pose.xyz, fused_yaw)     # nose-first: refine map-yaw offset
        heading = est.heading(fused_yaw)
        if not getattr(ctrl, "refine_heading_from_motion", True) and fused_yaw is not None:
            # translational controller strafes/backs up, so motion != heading; use
            # the drone's actual body yaw (Sphinx fused yaw == map heading; the real
            # GLOMAP map needs a one-time constant NED->map yaw calibration instead).
            heading = fused_yaw

        cmd = ctrl.step(pose, heading, now)
        send_pcmd(cmd.roll, cmd.pitch, yaw_sign * cmd.yaw, cmd.gaz)

        yaw_rate = None
        if fused_yaw is not None and prev_yaw is not None and now > prev_t:
            yaw_rate = wrap_angle(fused_yaw - prev_yaw) / (now - prev_t)
        prev_yaw, prev_t = fused_yaw, now
        vel = get_vel(now) if get_vel else None

        rec = {
            "t": round(now - t0, 4),
            **trial_meta,
            "mode": cmd.mode,
            "pcmd": list(cmd.pcmd),
            "fused_yaw": fused_yaw,
            "heading": heading,
            "heading_offset": est.offset,
            "motion_heading": est.motion_heading,
            "yaw_rate": yaw_rate,
            "vel_ned": list(vel) if vel else None,
            "pose": [pose.x, pose.y, pose.z] if pose else None,
            "route_p0": route.wp[0].tolist(),
            "current_height": (-pose.y) if pose else None,   # up = -y
            "target_height": (-cmd.info["target_y"]) if cmd.info.get("target_y") is not None else None,
            "telemetry_age_s": (now - pose.stamp) if pose else None,
            "telemetry_stale": pose is None or
            (now - pose.stamp) > getattr(ctrl.p, "max_pose_age_s", 0.6),
            **cmd.info,
        }
        ticks.append(rec)
        if tick_sink:
            tick_sink(rec)

        ct = rec.get("cross_track_full", rec.get("cross_track"))
        if ct is not None and ct > hard_stop_ct_m:
            rec["harness_stop"] = "runaway_cross_track"
            break
        if cmd.mode in (COMPLETED, ABORT_OR_MANUAL):
            if terminal_since is None:
                terminal_since = now
            elif now - terminal_since > 1.0:
                break
        advance_fn(DT)
    send_pcmd(0, 0, 0, 0)
    return ticks


def _ground_truth_augment(route: RouteModel, plant) -> dict:
    gt = plant.get_pose()
    proj = route.project(gt.xyz)
    tube = route.project_tube(gt.xyz)
    return {
        "gt_pose": [gt.x, gt.y, gt.z],
        "cross_track_true": proj.cross_track,
        "route_progress_true": proj.s / max(1e-9, route.length),
        "vert_err_true": float(gt.y - proj.target_y),
        "tube_distance_true": tube.distance,
    }


# ---------------------------------------------------------------------------
# Trial plan

def build_trial_plan(args, rng: np.random.Generator) -> list[dict]:
    algorithms = ALGORITHM_ORDER if args.rejoin_algorithm == "compare" \
        else [args.rejoin_algorithm]
    perturbations = PERTURBATION_SUITE if args.perturbation_suite else [
        PerturbationConfig(pose_noise_m=args.pose_noise_m,
                           yaw_bias_deg=args.yaw_bias_deg,
                           yaw_noise_deg=args.yaw_noise_deg,
                           telemetry_delay_ms=args.telemetry_delay_ms,
                           telemetry_drop_rate=args.telemetry_drop_rate,
                           speed_noise_mps=args.speed_noise_mps)]
    if args.yaw_error_list:
        yaw_errors = [float(x) for x in args.yaw_error_list.split(",") if x.strip() != ""]
    elif args.yaw_sweep:
        yaw_errors = list(YAW_SWEEP_DEG)
    else:
        yaw_errors = None                     # random within +/- range per trial
    # Generate scenarios once, then reuse each exact scenario for every
    # algorithm. This makes compare mode a paired experiment rather than giving
    # later algorithms different yaw errors, starts, noise and random seeds.
    scenarios = []
    for pert in perturbations:
        for i in range(args.num_trials):
            scenario_id = len(scenarios)
            if yaw_errors is not None:
                yerr = yaw_errors[i % len(yaw_errors)]
            else:
                yerr = float(rng.uniform(-args.random_yaw_error_deg,
                                         args.random_yaw_error_deg))
            scenarios.append({
                "scenario_id": scenario_id,
                "perturbation_cfg": pert,
                "perturbation": pert.label,
                "yaw_error_deg": yerr,
                "quadrant": i % 4,        # stratified: front/left/back/right
                "seed": args.random_seed + 1000 * scenario_id,
            })

    plan = []
    for scenario in scenarios:
        # Scenario-major execution keeps matched inputs adjacent on a physical
        # Sphinx run. Rotate the first algorithm to counterbalance battery,
        # thermal and world-order effects that cannot be reset reliably.
        shift = scenario["scenario_id"] % len(algorithms)
        ordered_algorithms = algorithms[shift:] + algorithms[:shift]
        for order_index, algo in enumerate(ordered_algorithms):
            plan.append({
                "trial_id": len(plan),
                "algorithm": algo,
                "scenario_order_index": order_index,
                **scenario,
            })
    return plan


def controller_overrides(args) -> dict:
    ov = {}
    if args.horizontal_control_mode:
        ov["horizontal_control_mode"] = args.horizontal_control_mode
    if args.disable_lateral_assist:
        ov["horizontal_control_mode"] = "nose_first"
    for cli, field in [
        ("max_lateral_roll_percent", "max_lateral_roll_percent"),
        ("lateral_assist_threshold", "lateral_assist_threshold"),
        ("waypoint_hover_s", "waypoint_hover_s"),
        ("segment_align_yaw_tolerance_deg", "segment_align_yaw_tolerance_deg"),
        ("segment_align_timeout_s", "segment_align_timeout_s"),
        ("arrival_radius", "arrival_radius"),
        ("arrival_vertical_radius", "arrival_vertical_radius"),
        ("segment_switch_progress", "segment_switch_progress"),
        ("min_segment_time_s", "min_segment_time_s"),
        ("max_segment_time_s", "max_segment_time_s"),
        ("route_tube_radius", "route_tube_radius"),
        ("route_tube_segment_window", "route_tube_segment_window"),
        ("route_tube_exit_s", "route_tube_exit_s"),
        ("route_tube_exit_updates", "route_tube_exit_updates"),
        ("route_tube_initial_grace_s", "route_tube_initial_grace_s"),
        ("hloc_min_match_count", "hloc_min_match_count"),
        ("hloc_min_inlier_ratio", "hloc_min_inlier_ratio"),
        ("hloc_max_reprojection_error", "hloc_max_reprojection_error"),
        ("hloc_max_pose_jump", "hloc_max_pose_jump"),
        ("camera_yaw_offset_deg", "camera_yaw_offset_deg"),
        ("pcmd_rate_limit_pct_per_s", "pcmd_rate_limit_pct_per_s"),
        ("speed_schedule", "speed_schedule"),
        ("base_lookahead", "base_lookahead"),
        ("min_lookahead", "min_lookahead"),
        ("max_lookahead", "max_lookahead"),
        ("k_error", "k_error"),
    ]:
        v = getattr(args, cli)
        if v is not None:
            ov[field] = v
    if args.body_yaw_align_at_waypoint is not None:
        ov["body_yaw_align_at_waypoint"] = bool(args.body_yaw_align_at_waypoint)
    return ov


# ---------------------------------------------------------------------------
# Kinematic backend

def run_kinematic(args, plan, route_c0, out_dir: Path, scale: ScaleContext) -> list[dict]:
    trials = []
    ticks_path = out_dir / "ticks.jsonl"
    with ticks_path.open("w") as tick_f:
        for spec in plan:
            rng = np.random.default_rng(spec["seed"])
            route = RouteModel(build_route(args.pattern, route_c0,
                                           seg_len=args.segment_length_m,
                                           height_amp=args.height_amp_m))
            r_head = route.segment_heading(0)
            off, rel, radius = sample_start_offset(rng, args.start_radius_m, r_head,
                                                   quadrant=spec["quadrant"])
            start = route.wp[0] + off
            start[1] += float(rng.uniform(-args.start_vert_jitter_m,
                                          args.start_vert_jitter_m))
            start_yaw = wrap_angle(r_head + math.radians(spec["yaw_error_deg"]))
            plant = KinematicAnafi(start, start_yaw)
            pert = spec["perturbation_cfg"]
            src = PerturbedPoseSource(plant, PerturbationConfig(
                **{**pert.__dict__, "seed": spec["seed"]})) if pert.any_active else plant

            ctrl = make_controller(spec["algorithm"], route, controller_overrides(args))
            meta = {"trial_id": spec["trial_id"], "algorithm_name": spec["algorithm"],
                    "scenario_id": spec["scenario_id"], "scenario_seed": spec["seed"],
                    "scenario_order_index": spec["scenario_order_index"],
                    "horizontal_control_mode": getattr(ctrl.p, "horizontal_control_mode", "n/a"),
                    "backend": plant.backend_name, "scale_mode": "sphinx_meters" if
                    not scale.has_scale else "meters+map_units"}

            def sink(rec, plant=plant, route=route, tick_f=tick_f):
                rec.update(_ground_truth_augment(route, plant))
                tick_f.write(json.dumps(rec) + "\n")

            ticks = run_trial(
                ctrl=ctrl, route=route,
                get_pose=(src.get_pose if hasattr(src, "get_pose") else plant.get_pose),
                get_yaw=src.yaw, get_vel=getattr(src, "velocity_ned", None),
                send_pcmd=plant.send_pcmd,
                now_fn=lambda plant=plant: plant.t,
                advance_fn=plant.step,
                duration_s=args.duration, yaw_sign=args.yaw_sign,
                hard_stop_ct_m=args.max_allowed_error_m + args.start_radius_m + 4.0,
                trial_meta=meta, tick_sink=sink)

            # Kinematic backend only: evaluate against the clean plant state;
            # retain the perturbed controller observation for diagnostics.
            for r in ticks:
                if "cross_track_true" in r:
                    r["cross_track_perceived"] = r.get("cross_track")
                    r["cross_track"] = r["cross_track_true"]
                    r["route_progress"] = r.get("route_progress_true",
                                                r.get("route_progress"))
                    r["vert_err"] = r.get("vert_err_true", r.get("vert_err"))
            m = compute_trial_metrics(
                ticks, ideal_success_error_m=args.ideal_success_error_m,
                max_allowed_error_m=args.max_allowed_error_m,
                min_corridor_time_ratio=args.min_corridor_time_ratio,
                success_hold_s=args.success_hold_s)
            trials.append({
                "trial_id": spec["trial_id"], "algorithm": spec["algorithm"],
                "scenario_id": spec["scenario_id"], "seed": spec["seed"],
                "scenario_order_index": spec["scenario_order_index"],
                "backend": plant.backend_name,
                "perturbation": spec["perturbation"],
                "yaw_error_deg": spec["yaw_error_deg"],
                "start_offset_m": radius,
                "start_bearing_rel_route_deg": math.degrees(rel),
                "quadrant": spec["quadrant"],
                "valid_for_comparison": not m["trivial_start_on_path"],
                "metrics": m,
            })
            print(f"[trial {spec['trial_id']:03d}] {spec['algorithm']:28s} "
                  f"pert={spec['perturbation']:14s} yaw={spec['yaw_error_deg']:+6.1f}deg "
                  f"r={radius:4.2f}m -> {'PASS' if m['pass'] else 'FAIL'} "
                  f"conv={m['time_to_converge_s'] if m['converged'] else 'never'} "
                  f"prog={m['route_progress']:.2f} {m['labels']}", flush=True)
    return trials


# ---------------------------------------------------------------------------
# Sphinx backend

def _sphinx_cli_version() -> str | None:
    try:
        result = subprocess.run(
            ["sphinx", "--version"], capture_output=True, text=True,
            timeout=3.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = "\n".join(x for x in (result.stdout.strip(), result.stderr.strip()) if x)
    return output.splitlines()[0] if output else None


def _firmware_selector_is_explicit(value: str) -> bool:
    selector = str(value).strip().lower()
    prefix = "https://firmware.parrot.com/versions/anafi/pc/"
    if not selector.startswith(prefix) or "latest" in selector:
        return False
    parts = selector[len(prefix):].split("/")
    return (
        len(parts) == 3
        and bool(parts[0])
        and parts[1] == "images"
        and parts[2].endswith(".zip")
        and all(part not in {".", ".."} for part in parts)
        and not any(char in selector for char in "?#")
    )


def _require_explicit_sphinx_firmware_selector() -> str:
    selector = os.environ.get("FIRMWARE_URL", "").strip()
    if not _firmware_selector_is_explicit(selector):
        raise SystemExit(
            "Sphinx runs require FIRMWARE_URL with an explicit reviewed Parrot "
            "firmware revision; missing, malformed, and latest selectors are refused"
        )
    return selector


def _sphinx_runtime_metadata(drone, olympe_module, ProductVersionChanged) -> dict:
    try:
        firmware_state = drone.get_state(ProductVersionChanged)
        firmware_version = str(firmware_state.get("software") or "unknown")
    except (AttributeError, KeyError, RuntimeError, TypeError):
        firmware_version = "unknown"
    try:
        from importlib.metadata import version
        olympe_version = version("parrot-olympe")
    except Exception:
        olympe_version = str(getattr(olympe_module, "__version__", "unknown"))
    firmware_selector = os.environ.get("FIRMWARE_URL", "").strip()
    firmware_image_pinned = _firmware_selector_is_explicit(firmware_selector)
    return {
        "sphinx_version": _sphinx_cli_version() or "unknown",
        "firmware_version": firmware_version,
        "olympe_version": olympe_version,
        "firmware_selector": firmware_selector or "not provided to harness process",
        "firmware_image_pinned": firmware_image_pinned,
        "reproducibility_note": (
            "Explicit firmware selector and runtime versions were recorded."
            if firmware_image_pinned else
            "FIRMWARE_URL was not provided to this harness process; runtime versions "
            "were recorded, but exact firmware selection is not reproducible."
        ),
    }


def _product_version_message():
    """Load the firmware-version state from its Olympe 8.x namespace."""
    from olympe.messages.common.SettingsState import ProductVersionChanged
    return ProductVersionChanged


def run_sphinx(args, plan, out_dir: Path, scale: ScaleContext) -> list[dict]:
    _require_explicit_sphinx_firmware_selector()
    import olympe
    from olympe.messages.ardrone3.Piloting import Landing, PCMD, TakeOff, moveBy
    from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
    from telemetry_sources import SphinxTelemetrySource

    ProductVersionChanged = _product_version_message()

    olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})
    drone = olympe.Drone(args.ip)
    if not drone.connect(retry=3):
        raise SystemExit(
            f"could not connect to Sphinx ANAFI at {args.ip}. Is Sphinx running? "
            "Start it with 定位演算法/flight_control/launch_sphinx_anafi_empty.sh "
            "(sphinx anafi.drone + parrot-ue4-empty). Pure-Python tests run without it: "
            "pytest -q tests/")

    args.sphinx_runtime_metadata = _sphinx_runtime_metadata(
        drone, olympe, ProductVersionChanged)
    print(f"[runtime] {json.dumps(args.sphinx_runtime_metadata, sort_keys=True)}", flush=True)

    trials = []
    ticks_path = out_dir / "ticks.jsonl"

    def send_pcmd(r, p, y, g):
        drone(PCMD(1, int(r), int(p), int(y), int(g), 0))

    try:
        _ensure_hovering(drone, TakeOff, Landing, FlyingStateChanged)
        src0 = SphinxTelemetrySource(drone)
        t_wait = time.monotonic()
        while src0.get_pose(time.monotonic()) is None:
            if time.monotonic() - t_wait > 15:
                raise SystemExit("no valid Sphinx/Olympe position after 15s")
            time.sleep(0.2)

        yaw_sign = args.yaw_sign
        if args.auto_yaw_calibration:
            yaw_sign = calibrate_yaw_sign(send_pcmd, lambda: src0.yaw(time.monotonic()),
                                          args.calibration_duration_s)
            print(f"[yaw-calibration] inferred yaw_sign={yaw_sign}", flush=True)

        p0 = src0.get_pose(time.monotonic())
        route_c0 = p0.xyz
        with ticks_path.open("w") as tick_f:
            for spec in plan:
                rng = np.random.default_rng(spec["seed"])
                route = RouteModel(build_route(args.pattern, route_c0,
                                               seg_len=args.segment_length_m,
                                               height_amp=args.height_amp_m))
                r_head = route.segment_heading(0)
                off, rel, radius = sample_start_offset(rng, args.start_radius_m, r_head,
                                                       quadrant=spec["quadrant"])
                start = route.wp[0] + off
                start_yaw = wrap_angle(r_head + math.radians(spec["yaw_error_deg"]))
                try:
                    setup = _goto(
                        drone, src0, start, start_yaw, moveBy, FlyingStateChanged)
                except TrialSetupError as exc:
                    send_pcmd(0, 0, 0, 0)
                    trials.append({
                        "trial_id": spec["trial_id"], "algorithm": spec["algorithm"],
                        "scenario_id": spec["scenario_id"], "seed": spec["seed"],
                        "scenario_order_index": spec["scenario_order_index"],
                        "backend": "sphinx (invalid setup)",
                        "perturbation": spec["perturbation"],
                        "yaw_error_deg": spec["yaw_error_deg"],
                        "start_offset_m": radius,
                        "start_bearing_rel_route_deg": math.degrees(rel),
                        "quadrant": spec["quadrant"],
                        "valid_for_comparison": False,
                        "setup_valid": False,
                        "setup_error": str(exc),
                        "metrics": {
                            "pass": False,
                            "labels": ["invalid_trial_setup"],
                            "failure_reasons": [str(exc)],
                        },
                    })
                    print(f"[sphinx trial {spec['trial_id']:03d}] INVALID SETUP: {exc}; "
                          "stopping physical run", flush=True)
                    break

                pert = spec["perturbation_cfg"]
                src = PerturbedPoseSource(src0, PerturbationConfig(
                    **{**pert.__dict__, "seed": spec["seed"]})) if pert.any_active else src0
                ctrl = make_controller(spec["algorithm"], route, controller_overrides(args))
                meta = {"trial_id": spec["trial_id"], "algorithm_name": spec["algorithm"],
                        "scenario_id": spec["scenario_id"], "scenario_seed": spec["seed"],
                        "scenario_order_index": spec["scenario_order_index"],
                        "horizontal_control_mode": getattr(ctrl.p, "horizontal_control_mode", "n/a"),
                        "backend": "sphinx (Parrot Sphinx ANAFI, Olympe fused telemetry)",
                        "evaluation_reference": "controller_fused_telemetry_not_ground_truth",
                        "scale_mode": "sphinx_meters" if not scale.has_scale
                        else "meters+map_units"}

                def sink(rec, route=route, tick_f=tick_f):
                    ref = src0.get_fused_reference(time.monotonic())
                    if ref is not None:
                        proj = route.project(ref.xyz)
                        rec.update({
                            "fused_reference_pose": [ref.x, ref.y, ref.z],
                            "fused_reference_stamp": ref.stamp,
                            "fused_reference_seq": ref.source_seq,
                            "cross_track_fused_reference": proj.cross_track,
                            "route_progress_fused_reference": (
                                proj.s / max(1e-9, route.length)
                            ),
                            "vert_err_fused_reference": float(ref.y - proj.target_y),
                        })
                    tick_f.write(json.dumps(rec) + "\n")

                tick_period = DT

                def advance(_dt):
                    time.sleep(tick_period)

                ticks = run_trial(
                    ctrl=ctrl, route=route, get_pose=src.get_pose, get_yaw=src.yaw,
                    get_vel=getattr(src, "velocity_ned", None), send_pcmd=send_pcmd,
                    now_fn=time.monotonic, advance_fn=advance,
                    duration_s=args.duration, yaw_sign=yaw_sign,
                    hard_stop_ct_m=args.max_allowed_error_m + args.start_radius_m + 4.0,
                    trial_meta=meta, tick_sink=sink)
                m = compute_trial_metrics(
                    ticks, ideal_success_error_m=args.ideal_success_error_m,
                    max_allowed_error_m=args.max_allowed_error_m,
                    min_corridor_time_ratio=args.min_corridor_time_ratio,
                    success_hold_s=args.success_hold_s)
                trials.append({
                    "trial_id": spec["trial_id"], "algorithm": spec["algorithm"],
                    "scenario_id": spec["scenario_id"], "seed": spec["seed"],
                    "scenario_order_index": spec["scenario_order_index"],
                    "backend": meta["backend"], "perturbation": spec["perturbation"],
                    "yaw_error_deg": spec["yaw_error_deg"], "start_offset_m": radius,
                    "start_bearing_rel_route_deg": math.degrees(rel),
                    "quadrant": spec["quadrant"],
                    "valid_for_comparison": not m["trivial_start_on_path"],
                    **setup, "metrics": m,
                })
                print(f"[sphinx trial {spec['trial_id']:03d}] {spec['algorithm']:28s} "
                      f"yaw={spec['yaw_error_deg']:+6.1f}deg -> "
                      f"{'PASS' if m['pass'] else 'FAIL'} {m['labels']}", flush=True)
    finally:
        # simulator-only safety epilogue: zero PCMD, then Landing, then disconnect.
        try:
            send_pcmd(0, 0, 0, 0)
        except Exception:
            pass
        try:
            landing = drone(
                Landing() >> FlyingStateChanged(state="landed", _timeout=15)
            ).wait()
            if hasattr(landing, "success") and not landing.success():
                print("warning: Landing did not confirm within 15 seconds", flush=True)
        except Exception as exc:
            print(f"warning: Landing failed: {exc}", flush=True)
        drone.disconnect()
    return trials


def calibrate_yaw_sign(send_pcmd, read_yaw, duration_s: float = 1.0,
                       yaw_pct: int = 15) -> int:
    """SIMULATOR-ONLY yaw-sign check: small positive yaw PCMD while hovering,
    compare fused yaw before/after, restore zero PCMD. Never uses Emergency."""
    y0 = read_yaw()
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        send_pcmd(0, 0, min(20, yaw_pct), 0)
        time.sleep(0.05)
    send_pcmd(0, 0, 0, 0)
    time.sleep(0.6)
    y1 = read_yaw()
    if y0 is None or y1 is None:
        print("[yaw-calibration] no fused yaw; keeping yaw_sign=+1", flush=True)
        return 1
    delta = wrap_angle(y1 - y0)
    print(f"[yaw-calibration] +{yaw_pct}% PCMD for {duration_s:.1f}s -> "
          f"fused yaw delta {math.degrees(delta):+.1f} deg", flush=True)
    return 1 if delta >= 0 else -1


def _ensure_hovering(drone, TakeOff, Landing, FlyingStateChanged, tries: int = 3):
    """Bring the drone to a hovering state, self-healing from whatever state a
    prior aborted run left it in (already flying, mid-landing, emergency).
    Sim-only; never uses Emergency."""
    for attempt in range(tries):
        try:
            state = str(drone.get_state(FlyingStateChanged).get("state"))
        except (KeyError, RuntimeError):
            state = "unknown"
        if "hovering" in state or "flying" in state:
            return
        if "landing" in state or "takingoff" in state:
            # wait out the transition, then settle to landed
            drone(FlyingStateChanged(state="landed", _timeout=12)).wait()
        r = drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=20)).wait()
        if getattr(r, "success", lambda: False)():
            return
        drone(Landing() >> FlyingStateChanged(state="landed", _timeout=15)).wait()
        time.sleep(1.0)
    raise SystemExit("takeoff failed: could not reach hovering after retries "
                     "(is the Sphinx ANAFI healthy? try relaunching the simulator)")


class TrialSetupError(RuntimeError):
    """Sphinx could not establish the requested physical trial start state."""


def _wait_expectation(expectation, label: str):
    try:
        result = expectation.wait()
    except Exception as exc:
        raise TrialSetupError(f"{label} expectation raised: {exc}") from exc
    if hasattr(result, "success") and not result.success():
        raise TrialSetupError(f"{label} expectation failed or timed out")
    return result


def _goto(drone, src, target, target_yaw, moveBy, FlyingStateChanged):
    """Establish and verify the requested Sphinx trial start pose.

    This setup is outside the evaluated controller. A command ACK alone is not
    enough: every move expectation and the final measured position/yaw must be
    within tolerance, otherwise the trial is invalid and the physical run stops.
    """
    target = np.asarray(target, float)
    for attempt in range(3):
        now = time.monotonic()
        pose = src.get_pose(now)
        yaw = src.yaw(now)
        if pose is None or yaw is None:
            raise TrialSetupError("moveBy setup has no fresh pose/yaw telemetry")
        d = np.asarray(target, float) - pose.xyz
        dn, de, ddown = float(d[0]), float(d[2]), float(d[1])
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx_body = cy * dn + sy * de
        dy_body = -sy * dn + cy * de
        if math.hypot(dx_body, dy_body) < 0.3 and abs(ddown) < 0.3:
            break
        _wait_expectation(
            drone(moveBy(dx_body, dy_body, ddown, 0.0)
                  >> FlyingStateChanged(state="hovering", _timeout=25)),
            f"moveBy translation attempt {attempt + 1}",
        )
    now = time.monotonic()
    yaw = src.yaw(now)
    if yaw is None:
        raise TrialSetupError("moveBy yaw setup has no fresh attitude telemetry")
    yaw_delta = wrap_angle(target_yaw - yaw)
    if abs(yaw_delta) > math.radians(2.0):
        _wait_expectation(
            drone(moveBy(0, 0, 0, yaw_delta)
                  >> FlyingStateChanged(state="hovering", _timeout=15)),
            "moveBy yaw",
        )

    now = time.monotonic()
    actual_pose = src.get_pose(now)
    actual_yaw = src.yaw(now)
    if actual_pose is None or actual_yaw is None:
        raise TrialSetupError("post-moveBy validation has no fresh pose/yaw telemetry")
    position_error = float(np.linalg.norm(actual_pose.xyz - target))
    yaw_error_deg = abs(math.degrees(wrap_angle(actual_yaw - target_yaw)))
    if position_error > SPHINX_START_POSITION_TOL_M:
        raise TrialSetupError(
            f"start position error {position_error:.3f}m exceeds "
            f"{SPHINX_START_POSITION_TOL_M:.3f}m"
        )
    if yaw_error_deg > SPHINX_START_YAW_TOL_DEG:
        raise TrialSetupError(
            f"start yaw error {yaw_error_deg:.2f}deg exceeds "
            f"{SPHINX_START_YAW_TOL_DEG:.2f}deg"
        )
    return {
        "setup_valid": True,
        "setup_position_error_m": position_error,
        "setup_yaw_error_deg": yaw_error_deg,
        "actual_start_pose": actual_pose.xyz.tolist(),
        "actual_start_yaw": float(actual_yaw),
    }


# ---------------------------------------------------------------------------
# CLI

def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Return argparse destinations explicitly present as --flag or --flag=value."""
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    given = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            given.add(dest)
    return given


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    for dest, action in actions.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        if action.choices is not None and value not in action.choices:
            parser.error(
                f"--{dest.replace('_', '-')} must be one of {list(action.choices)}"
            )
        if isinstance(action.default, bool) and not isinstance(value, bool):
            parser.error(f"--{dest.replace('_', '-')} must be boolean")
        if action.type not in (int, float):
            continue
        expected = action.type
        valid_type = (isinstance(value, int) if expected is int
                      else isinstance(value, (int, float)))
        if isinstance(value, bool) or not valid_type:
            parser.error(f"--{dest.replace('_', '-')} must be a {expected.__name__}")
        if not math.isfinite(float(value)):
            parser.error(f"--{dest.replace('_', '-')} must be finite")

    def bounded(name, lo, hi, *, lo_open=False):
        value = getattr(args, name)
        if value is None:
            return
        too_low = value <= lo if lo_open else value < lo
        if too_low or value > hi:
            left = "(" if lo_open else "["
            parser.error(
                f"--{name.replace('_', '-')} must be in {left}{lo}, {hi}]"
            )

    bounded("num_trials", 1, 10_000)
    bounded("random_seed", 0, 2**63 - 1)
    for name in ("duration", "segment_length_m", "calibration_duration_s",
                 "ideal_success_error_m", "max_allowed_error_m", "success_hold_s"):
        bounded(name, 0.0, 3600.0, lo_open=True)
    for name in ("height_amp_m", "start_vert_jitter_m", "pose_noise_m"):
        bounded(name, 0.0, 1000.0)
    bounded("start_radius_m", 0.75, 1000.0)
    bounded("yaw_noise_deg", 0.0, 180.0)
    bounded("telemetry_delay_ms", 0.0, 60_000.0)
    bounded("speed_noise_mps", 0.0, 100.0)
    bounded("random_yaw_error_deg", 0.0, 180.0)
    bounded("yaw_bias_deg", -180.0, 180.0)
    bounded("camera_yaw_offset_deg", -180.0, 180.0)
    bounded("segment_align_yaw_tolerance_deg", 0.0, 180.0)
    bounded("telemetry_drop_rate", 0.0, 1.0)
    bounded("min_corridor_time_ratio", 0.0, 1.0)
    for name in ("hloc_min_inlier_ratio", "segment_switch_progress"):
        bounded(name, 0.0, 1.0)
    for name in ("max_lateral_roll_percent",):
        bounded(name, 0, 100)
    for name in ("route_tube_segment_window", "route_tube_exit_updates",
                 "hloc_min_match_count"):
        bounded(name, 0, 100_000)
    for name in ("lateral_assist_threshold", "waypoint_hover_s",
                 "segment_align_timeout_s", "base_lookahead", "min_lookahead",
                 "max_lookahead", "k_error", "arrival_radius",
                 "arrival_vertical_radius", "min_segment_time_s",
                 "max_segment_time_s", "route_tube_exit_s",
                 "route_tube_initial_grace_s", "hloc_max_reprojection_error",
                 "hloc_max_pose_jump", "pcmd_rate_limit_pct_per_s"):
        bounded(name, 0.0, 60_000.0)
    if args.route_tube_radius is not None:
        bounded("route_tube_radius", 0.0, 1000.0, lo_open=True)
    for name in ("map_units_per_meter", "meters_per_map_unit"):
        if getattr(args, name) is not None:
            bounded(name, 0.0, 1e9, lo_open=True)
    if args.map_units_per_meter is not None and args.meters_per_map_unit is not None:
        parser.error("provide only one map scale factor")
    if args.max_allowed_error_m < args.ideal_success_error_m:
        parser.error("--max-allowed-error-m must be >= --ideal-success-error-m")
    if args.success_hold_s > args.duration:
        parser.error("--success-hold-s must not exceed --duration")
    if (args.min_segment_time_s is not None and args.max_segment_time_s is not None
            and args.max_segment_time_s < args.min_segment_time_s):
        parser.error("--max-segment-time-s must be >= --min-segment-time-s")
    if args.yaw_error_list:
        try:
            values = [float(x) for x in args.yaw_error_list.split(",") if x.strip()]
        except ValueError:
            parser.error("--yaw-error-list must contain only numbers")
        if not values or any(not math.isfinite(x) or abs(x) > 180.0 for x in values):
            parser.error("--yaw-error-list values must be finite and within [-180, 180]")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file; explicit CLI flags win")
    ap.add_argument("--backend", choices=("sphinx", "kinematic"), default="sphinx")
    ap.add_argument("--ip", default=SPHINX_IP_DEFAULT)
    ap.add_argument("--pattern", choices=("line", "square", "s_curve"), default="s_curve")
    ap.add_argument("--segment-length-m", type=float, default=4.0)
    ap.add_argument("--height-amp-m", type=float, default=0.0,
                    help="route climb amplitude (Sphinx meters)")
    ap.add_argument("--start-radius-m", type=float, default=5.0)
    ap.add_argument("--start-vert-jitter-m", type=float, default=0.3)
    ap.add_argument("--num-trials", type=int, default=20)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=1)
    ap.add_argument("--auto-yaw-calibration", action="store_true")
    ap.add_argument("--calibration-duration-s", type=float, default=1.0)
    # initial yaw error
    ap.add_argument("--initial-yaw-mode", choices=("route",), default="route",
                    help="operator roughly faces the route at takeoff")
    ap.add_argument("--random-yaw-error-deg", type=float, default=15.0)
    ap.add_argument("--yaw-sweep", action="store_true",
                    help=f"cycle the spec buckets {YAW_SWEEP_DEG}")
    ap.add_argument("--yaw-error-list", default="",
                    help="comma-separated explicit yaw errors (deg), cycled")
    # algorithm
    ap.add_argument("--rejoin-algorithm", default="adaptive_lookahead",
                    choices=ALGORITHM_ORDER + ["compare"])
    ap.add_argument("--horizontal-control-mode",
                    choices=("nose_first", "small_lateral_assist", "hybrid"), default=None)
    ap.add_argument("--disable-lateral-assist", action="store_true")
    ap.add_argument("--max-lateral-roll-percent", type=int, default=None)
    ap.add_argument("--lateral-assist-threshold", type=float, default=None)
    ap.add_argument("--body-yaw-align-at-waypoint", type=int, choices=(0, 1), default=None)
    ap.add_argument("--waypoint-hover-s", type=float, default=None)
    ap.add_argument("--segment-align-yaw-tolerance-deg", type=float, default=None)
    ap.add_argument("--segment-align-timeout-s", type=float, default=None)
    ap.add_argument("--speed-schedule", choices=("cos", "threshold", "smoothstep"),
                    default=None)
    ap.add_argument("--base-lookahead", type=float, default=None)
    ap.add_argument("--min-lookahead", type=float, default=None)
    ap.add_argument("--max-lookahead", type=float, default=None)
    ap.add_argument("--k-error", type=float, default=None)
    ap.add_argument("--arrival-radius", type=float, default=None)
    ap.add_argument("--arrival-vertical-radius", type=float, default=None)
    ap.add_argument("--segment-switch-progress", type=float, default=None)
    ap.add_argument("--min-segment-time-s", type=float, default=None)
    ap.add_argument("--max-segment-time-s", type=float, default=None)
    ap.add_argument("--route-tube-radius", type=float, default=None,
                    help="optional full-3D route tube radius in route units; exits abort to manual hover")
    ap.add_argument("--route-tube-segment-window", type=int, default=None,
                    help="tube checks active segment +/- this many segments")
    ap.add_argument("--route-tube-exit-s", type=float, default=None,
                    help="continuous outside-tube seconds required before route-tube abort")
    ap.add_argument("--route-tube-exit-updates", type=int, default=None,
                    help="effective pose updates outside tube required before route-tube abort")
    ap.add_argument("--route-tube-initial-grace-s", type=float, default=None,
                    help="allow initial rejoin outside tube for this many seconds before arming tube abort")
    ap.add_argument("--hloc-min-match-count", type=int, default=None)
    ap.add_argument("--hloc-min-inlier-ratio", type=float, default=None)
    ap.add_argument("--hloc-max-reprojection-error", type=float, default=None)
    ap.add_argument("--hloc-max-pose-jump", type=float, default=None)
    ap.add_argument("--camera-yaw-offset-deg", type=float, default=None,
                    help="camera optical yaw offset relative to body yaw; used for waypoint yaw alignment")
    ap.add_argument("--pcmd-rate-limit-pct-per-s", type=float, default=None,
                    help="rate limit commanded PCMD percent changes; safety zero commands bypass this")
    # corridor evaluation
    ap.add_argument("--ideal-success-error-m", type=float, default=0.5)
    ap.add_argument("--max-allowed-error-m", type=float, default=3.0)
    ap.add_argument("--min-corridor-time-ratio", type=float, default=0.8)
    ap.add_argument("--success-hold-s", type=float, default=3.0)
    # telemetry perturbation
    ap.add_argument("--pose-noise-m", type=float, default=0.0)
    ap.add_argument("--yaw-bias-deg", type=float, default=0.0)
    ap.add_argument("--yaw-noise-deg", type=float, default=0.0)
    ap.add_argument("--telemetry-delay-ms", type=float, default=0.0)
    ap.add_argument("--telemetry-drop-rate", type=float, default=0.0)
    ap.add_argument("--speed-noise-mps", type=float, default=0.0)
    ap.add_argument("--perturbation-suite", action="store_true",
                    help="run the spec's perturbation set instead of single values")
    # scale-aware reporting
    ap.add_argument("--sim-meters", type=int, choices=(0, 1), default=1)
    ap.add_argument("--map-units-per-meter", type=float, default=None)
    ap.add_argument("--meters-per-map-unit", type=float, default=None)
    ap.add_argument("--report-scale-warning", type=int, choices=(0, 1), default=1)
    ap.add_argument("--csv", action="store_true", help="also write trials.csv")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs" / "run"))
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(cli_argv)

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        defaults = {a.dest: a.default for a in ap._actions if a.dest != "help"}
        given = _explicit_cli_dests(ap, cli_argv)
        for k, v in cfg.items():
            k = k.replace("-", "_")
            if k not in defaults:
                raise SystemExit(f"unknown config key {k!r} in {args.config}")
            if k not in given:
                setattr(args, k, v)
    _validate_args(ap, args)
    return args


def apply_backend_interpretation(summary: dict, args, trials: list[dict]) -> None:
    """Label ranking strength without overstating sequential Sphinx trials."""
    raw_ranking = rank_algorithms(summary)
    scenario_inputs_matched = args.rejoin_algorithm == "compare"
    summary["scenario_inputs_matched"] = scenario_inputs_matched
    if args.backend == "sphinx":
        runtime = getattr(args, "sphinx_runtime_metadata", {})
        summary["paired_scenarios"] = False
        summary["physical_trials_independent"] = False
        summary["physical_fairness"] = (
            "Scenario inputs are matched and algorithm order is cyclically "
            "counterbalanced, but trials share sequential firmware/world/battery "
            "state and are not physically paired or independent."
        )
        summary["evaluation_reference"] = (
            "controller firmware-fused telemetry; no independent simulator ground truth"
        )
        summary["exploratory_ranking"] = raw_ranking
        summary["ranking"] = []
        summary["recommended_algorithm"] = None
        summary["runtime_versions"] = runtime
        summary["fully_reproducible"] = bool(runtime.get("firmware_image_pinned", False))
        summary["physical_run_complete"] = (
            summary.get("n_invalid_trials", 0) == 0 and len(trials) > 0 and
            len(trials) == getattr(args, "expected_trial_count", len(trials))
        )
    else:
        summary["paired_scenarios"] = scenario_inputs_matched
        summary["physical_trials_independent"] = True
        summary["ranking"] = raw_ranking
        summary["recommended_algorithm"] = raw_ranking[0][0] if raw_ranking else None
        summary["fully_reproducible"] = True


def system_validation_scope(backend: str) -> dict:
    """Machine-readable guard against treating a controller trial as E2E evidence."""
    return {
        "control_only": True,
        "visual_localization_closed_loop": False,
        "independent_pose_reference": backend == "kinematic",
        "eligible_as_end_to_end_system_evidence": False,
        "blocking_requirements": [
            "simulator scene rendered from the same mapped environment",
            "production EDM localizer driven by simulator camera frames",
            (
                "independent simulator truth separate from controller telemetry"
                if backend == "sphinx"
                else "Parrot Sphinx firmware and dynamics"
            ),
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print_sim_banner()
    check_simulator_ip(args.ip)

    scale = ScaleContext(map_units_per_meter=args.map_units_per_meter,
                         meters_per_map_unit=args.meters_per_map_unit)
    for line in scale.report_header():
        print(line, flush=True)
    if not scale.has_scale and args.report_scale_warning:
        print(NO_SCALE_WARNING, flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    plan = build_trial_plan(args, rng)
    args.expected_trial_count = len(plan)
    print(f"[plan] {len(plan)} trials "
          f"({args.rejoin_algorithm}, backend={args.backend}, pattern={args.pattern})",
          flush=True)

    # route validation report (steep segments etc.)
    route_probe = RouteModel(build_route(args.pattern, np.zeros(3),
                                         seg_len=args.segment_length_m,
                                         height_amp=args.height_amp_m))
    validation = route_probe.validate()
    for seg in validation:
        if seg.get("warning"):
            print(f"[route] segment {seg['segment']}: {seg['warning']}", flush=True)

    if args.backend == "kinematic":
        print("[backend] KINEMATIC approximation -- NOT Sphinx physics. Use "
              "--backend sphinx for Parrot ANAFI simulator validation.", flush=True)
        trials = run_kinematic(args, plan, np.array([0.0, -2.5, 0.0]), out_dir, scale)
    else:
        trials = run_sphinx(args, plan, out_dir, scale)

    summary = aggregate_trials(trials)
    summary["backend"] = args.backend
    summary["anafi_profile"] = ANAFI_PROFILE.to_metadata()
    apply_backend_interpretation(summary, args, trials)
    summary["model_fidelity"] = (
        "Parrot Sphinx software-in-the-loop with ANAFI firmware"
        if args.backend == "sphinx"
        else "first-order kinematic approximation; not Sphinx physics or firmware"
    )
    summary["validation_scope"] = system_validation_scope(args.backend)
    summary["scale"] = {
        "sphinx_validation_scale": "meters",
        "real_sfm_map_scale": "arbitrary units",
        "real_altitude_scale": "arbitrary map units, not meters",
        "scale_factor_supplied": scale.has_scale,
        "meters_per_map_unit": scale.meters_per_map_unit,
        "map_units_per_meter": scale.map_units_per_meter,
    }
    if scale.has_scale:
        summary["scale"]["examples"] = {
            "ideal_success_error": scale.dual(args.ideal_success_error_m),
            "max_allowed_error": scale.dual(args.max_allowed_error_m),
            "base_lookahead": scale.dual(args.base_lookahead or 0.8),
            "route_length": scale.dual(route_probe.length),
            "arrival_radius": scale.dual(args.arrival_radius or 0.5),
        }
    else:
        summary["scale"]["warning"] = NO_SCALE_WARNING
        summary["scale"]["provisional_real_map_thresholds"] = \
            provisional_map_unit_thresholds(route_probe.length, route_probe.n_segments)
    summary["route_validation"] = validation
    summary["config"] = {k: v for k, v in vars(args).items() if k != "config"}

    (out_dir / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.csv:
        with (out_dir / "trials.csv").open("w", newline="") as f:
            cols = ["trial_id", "scenario_id", "seed", "algorithm", "backend",
                    "perturbation", "yaw_error_deg",
                    "start_offset_m", "quadrant"]
            mcols = ["pass", "converged", "time_to_converge_s", "route_progress",
                     "cross_track_mean_m", "cross_track_p90_m", "vertical_mean_m",
                     "corridor_time_ratio", "yaw_flips_per_min", "stop_and_go_score",
                     "labels"]
            w = csv.writer(f)
            w.writerow(cols + mcols)
            for t in trials:
                w.writerow([t[c] for c in cols] +
                           [t["metrics"].get(c) for c in mcols])

    print(f"\n[done] {len(trials)} trials -> {out_dir}")
    display_ranking = summary.get("ranking") or summary.get("exploratory_ranking", [])
    ranking_label = "exploratory " if args.backend == "sphinx" else ""
    for name, score in display_ranking:
        a = summary["by_algorithm"][name]
        print(f"  {ranking_label}{name:28s} score={score:7.2f} pass={a['pass_rate']:.2f} "
              f"complete={a['route_completion_rate']:.2f} "
              f"ct_mean={a['cross_track_mean_m'] and round(a['cross_track_mean_m'],3)}m")
    if not scale.has_scale and args.report_scale_warning:
        print("\n" + NO_SCALE_WARNING, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
