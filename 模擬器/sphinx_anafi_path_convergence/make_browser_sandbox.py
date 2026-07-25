#!/usr/bin/env python3
"""Generate a browser-based kinematic sandbox for waypoint-controller review.

The sandbox is intentionally static: this script runs the real experiment
controller code against the pure-Python kinematic plant, embeds the resulting
frames into one HTML file, and the browser only visualizes those frames. Re-run
this script after editing controllers.py.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from anafi_profile import ANAFI_PROFILE
from controllers import ABORT_OR_MANUAL, COMPLETED, ALGORITHM_ORDER, make_controller
from metrics import compute_trial_metrics
from route_geometry import RouteModel, heading_of, horiz, sample_start_offset, wrap_angle
from run_sphinx_anafi_convergence import DT, _ground_truth_augment, run_trial
from telemetry_sources import KinematicAnafi, PerturbationConfig, PerturbedPoseSource


def build_random_waypoints(rng: np.random.Generator, n: int, seg_len: float,
                           height_amp: float) -> list[np.ndarray]:
    """Deterministic random-ish route in raw frame: x/z horizontal, y=-up."""
    if n < 2:
        raise ValueError("--num-waypoints must be >= 2")
    pts = [np.array([0.0, -2.0, 0.0], dtype=float)]
    heading = 0.0
    up = 2.0
    for i in range(1, n):
        heading = wrap_angle(heading + float(rng.uniform(-0.55, 0.55)))
        step = np.array([math.cos(heading) * seg_len, 0.0,
                         math.sin(heading) * seg_len], dtype=float)
        if height_amp:
            up = 2.0 + height_amp * math.sin(i * 0.85) + float(rng.uniform(-0.15, 0.15))
        pts.append(pts[-1] + step)
        pts[-1][1] = -up
    return pts


def build_complex_waypoints(rng: np.random.Generator, n: int, map_size_m: float,
                            margin_m: float) -> list[np.ndarray]:
    """Nearly straight horizontal corridor with mild lateral weave and random heights."""
    if n < 2:
        raise ValueError("--num-waypoints must be >= 2")
    half = max(1.0, float(map_size_m) / 2.0 - float(margin_m))
    pts = []
    for i in range(n):
        t = i / max(1, n - 1)
        x = -0.80 + 1.60 * t
        z = 0.10 * math.sin(t * math.pi * 3.0) + float(rng.normal(0.0, 0.012))
        up = float(rng.uniform(1.2, 3.8))
        x = max(-0.82, min(0.82, x))
        z = max(-0.18, min(0.18, z))
        pts.append(np.array([x * half, -float(up), z * half], dtype=float))
    return pts


def fit_waypoints_to_square(points: list[np.ndarray], map_size_m: float | None,
                            margin_m: float) -> tuple[list[np.ndarray], float, float]:
    if map_size_m is None or map_size_m <= 0:
        return points, 1.0, 0.0
    half = float(map_size_m) / 2.0
    margin = max(0.0, min(float(margin_m), half - 1e-6))
    usable = max(1e-6, float(map_size_m) - 2.0 * margin)
    xs = [float(p[0]) for p in points]
    zs = [float(p[2]) for p in points]
    cx = (min(xs) + max(xs)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    span = max(max(xs) - min(xs), max(zs) - min(zs), 1e-6)
    scale = min(1.0, usable / span)
    fitted = []
    for p in points:
        q = np.asarray(p, dtype=float).copy()
        q[0] = (q[0] - cx) * scale
        q[2] = (q[2] - cz) * scale
        fitted.append(q)
    return fitted, scale, margin


def build_inspection_poles(points: list[np.ndarray], waypoint_numbers: list[int],
                           right_offset_m: float, top_above_waypoint_m: float) -> list[dict]:
    poles = []
    n = len(points)
    for wp_num in waypoint_numbers:
        i = wp_num - 1
        if i < 0 or i >= n:
            continue
        prev_i = max(0, i - 1)
        next_i = min(n - 1, i + 1)
        tangent = horiz(points[next_i]) - horiz(points[prev_i])
        if float(tangent @ tangent) < 1e-12:
            tangent = np.array([1.0, 0.0], dtype=float)
        tangent = tangent / float(np.linalg.norm(tangent))
        right = np.array([-tangent[1], tangent[0]], dtype=float)
        wp = np.asarray(points[i], dtype=float)
        pole_h = horiz(wp) + right * float(right_offset_m)
        top_up = -float(wp[1]) + float(top_above_waypoint_m)
        base = np.array([pole_h[0], 0.0, pole_h[1]], dtype=float)
        top = np.array([pole_h[0], -top_up, pole_h[1]], dtype=float)
        poles.append({
            "waypoint_index": i,
            "waypoint_label": f"W{wp_num}",
            "base": base,
            "top": top,
            "heading": heading_of(top - wp),
            "top_up": top_up,
        })
    return poles


def _append_planned_with_inspection(out_points: list[np.ndarray], out_labels: list[str],
                                    yaw_targets: dict[int, float],
                                    points: list[np.ndarray], indices: list[int],
                                    poles_by_index: dict[int, dict]) -> None:
    for i in indices:
        p = np.asarray(points[i], dtype=float).copy()
        out_points.append(p)
        out_labels.append(f"W{i + 1}")
        pole = poles_by_index.get(i)
        if pole is None:
            continue
        yaw_targets[len(out_points) - 1] = float(pole["heading"])
        top = p.copy()
        top[1] = float(pole["top"][1])
        out_points.append(top)
        out_labels.append(f"W{i + 1} pole top")
        yaw_targets[len(out_points) - 1] = float(pole["heading"])
        out_points.append(p.copy())
        out_labels.append(f"W{i + 1} return height")


def build_control_route(points: list[np.ndarray], return_to_start: bool,
                        poles: list[dict]) -> tuple[list[np.ndarray], list[str], dict[int, float]]:
    poles_by_index = {int(p["waypoint_index"]): p for p in poles}
    route_points: list[np.ndarray] = []
    labels: list[str] = []
    yaw_targets: dict[int, float] = {}
    _append_planned_with_inspection(route_points, labels, yaw_targets, points,
                                    list(range(len(points))), poles_by_index)
    if return_to_start and len(points) >= 2:
        _append_planned_with_inspection(route_points, labels, yaw_targets, points,
                                        list(reversed(range(len(points) - 1))), poles_by_index)
    return route_points, labels, yaw_targets


def wind_gust_vector(yaw: float, rng: np.random.Generator,
                     magnitude_m: float) -> tuple[np.ndarray, str]:
    directions = [
        ("forward", np.array([math.cos(yaw), 0.0, math.sin(yaw)], dtype=float)),
        ("back", np.array([-math.cos(yaw), 0.0, -math.sin(yaw)], dtype=float)),
        ("right", np.array([-math.sin(yaw), 0.0, math.cos(yaw)], dtype=float)),
        ("left", np.array([math.sin(yaw), 0.0, -math.cos(yaw)], dtype=float)),
        ("up", np.array([0.0, -1.0, 0.0], dtype=float)),
        ("down", np.array([0.0, 1.0, 0.0], dtype=float)),
    ]
    label, unit = directions[int(rng.integers(0, len(directions)))]
    return unit * float(magnitude_m), label


def make_frame(rec: dict) -> dict:
    truth_pose = rec.get("gt_pose")
    control_pose = rec.get("pose")
    pose = truth_pose or control_pose or [0.0, 0.0, 0.0]
    perceived = control_pose or pose
    control_yaw = rec.get("fused_yaw")
    truth_yaw = rec.get("gt_yaw", control_yaw)
    yaw_error = None
    if truth_yaw is not None and control_yaw is not None:
        yaw_error = math.degrees(wrap_angle(float(control_yaw) - float(truth_yaw)))
    pose_error = None
    if truth_pose is not None and control_pose is not None:
        pose_error = math.sqrt(sum((float(control_pose[i]) - float(truth_pose[i])) ** 2
                                   for i in range(3)))
    target = rec.get("target") or [None, None, None]
    pcmd = rec.get("pcmd") or [0, 0, 0, 0]
    return {
        "t": float(rec.get("t", 0.0)),
        "x": float(pose[0]),
        "y": float(pose[1]),
        "z": float(pose[2]),
        "px": float(perceived[0]),
        "py": float(perceived[1]),
        "pz": float(perceived[2]),
        "poseAvailable": control_pose is not None,
        "poseError": None if pose_error is None else float(pose_error),
        "poseAgeS": None if rec.get("telemetry_age_s") is None else float(rec.get("telemetry_age_s")),
        "targetX": None if target[0] is None else float(target[0]),
        "targetY": None if target[1] is None else float(target[1]),
        "targetZ": None if target[2] is None else float(target[2]),
        "yaw": float(truth_yaw or 0.0),
        "controlYaw": float(control_yaw or truth_yaw or 0.0),
        "yawErrorDeg": None if yaw_error is None else float(yaw_error),
        "heading": float(rec.get("heading") or 0.0),
        "mode": rec.get("mode") or "",
        "moveMode": rec.get("move_mode") or "",
        "segIndex": rec.get("seg_index"),
        "targetWp": rec.get("target_wp"),
        "roll": int(pcmd[0]) if len(pcmd) > 0 else 0,
        "pitch": int(pcmd[1]) if len(pcmd) > 1 else 0,
        "yawCmd": int(pcmd[2]) if len(pcmd) > 2 else 0,
        "gaz": int(pcmd[3]) if len(pcmd) > 3 else 0,
        "crossTrack": float(rec.get("cross_track_true", rec.get("cross_track", 0.0)) or 0.0),
        "crossTrackControl": float(rec.get("cross_track_control",
                                           rec.get("cross_track_perceived",
                                                   rec.get("cross_track", 0.0))) or 0.0),
        "routeProgress": float(rec.get("route_progress_true", rec.get("route_progress", 0.0)) or 0.0),
        "dToWp": float(rec.get("d_to_wp", 0.0) or 0.0),
        "bodyFwd": float(rec.get("body_fwd", 0.0) or 0.0),
        "bodyRight": float(rec.get("body_right", 0.0) or 0.0),
        "bodyUp": float(rec.get("body_up", 0.0) or 0.0),
        "tubeDistance": float(rec.get("tube_distance_true", rec.get("tube_distance", 0.0)) or 0.0),
        "tubeDistanceControl": float(rec.get("tube_distance_control",
                                             rec.get("tube_distance", 0.0)) or 0.0),
        "tubeExitS": float(rec.get("tube_exit_s", 0.0) or 0.0),
        "tubeExitUpdates": int(rec.get("tube_exit_updates", 0) or 0),
        "tubeExitUpdateLimit": int(rec.get("tube_exit_update_limit", 0) or 0),
        "tubeSafetyActive": bool(rec.get("tube_safety_active", True)),
        "tubeGraceSRemaining": float(rec.get("tube_grace_s_remaining", 0.0) or 0.0),
        "tubeActiveSeg": rec.get("tube_active_seg"),
        "tubeSegmentWindow": rec.get("tube_segment_window"),
        "abortReason": rec.get("abort_reason") or "",
        "telemetryStale": bool(rec.get("telemetry_stale", False)),
        "poseQuality": rec.get("pose_quality") or "",
        "poseLossStage": rec.get("pose_loss_stage") or "",
        "lostDurationS": None if rec.get("lost_duration_s") is None else float(rec.get("lost_duration_s")),
        "finalLandingPending": bool(rec.get("final_landing_pending", False)),
        "finalLandingReady": bool(rec.get("final_landing_ready", False)),
        "finalLandingDistance": rec.get("final_landing_distance"),
        "finalLandingRadius": rec.get("final_landing_radius"),
        "finalLandingHoldS": rec.get("final_landing_hold_s"),
        "finalLandingRequiredHoldS": rec.get("final_landing_required_hold_s"),
        "finalLandingEstSpeed": rec.get("final_landing_est_speed"),
        "finalLandingMaxEstSpeed": rec.get("final_landing_max_est_speed"),
        "finalLandingYawSpanDeg": rec.get("final_landing_yaw_span_deg"),
        "finalLandingYawStableDeg": rec.get("final_landing_yaw_stable_deg"),
        "windGust": bool(rec.get("wind_gust", False)),
        "windDirection": rec.get("wind_direction") or "",
        "windMagnitude": float(rec.get("wind_magnitude", 0.0) or 0.0),
    }


def run_sandbox(args) -> dict:
    explicit_total_delay = getattr(args, "telemetry_delay_ms", None)
    video_e2e_latency_ms = float(getattr(
        args, "video_e2e_latency_ms", ANAFI_PROFILE.video_latency_ms))
    decode_localization_latency_ms = float(getattr(
        args, "decode_localization_latency_ms", 0.0))
    telemetry_delay_ms = (
        float(explicit_total_delay) if explicit_total_delay is not None
        else video_e2e_latency_ms + decode_localization_latency_ms
    )
    delay_is_lower_bound = (
        explicit_total_delay is None and decode_localization_latency_ms == 0.0
    )
    rng = np.random.default_rng(args.seed)
    map_margin = args.map_margin_m
    if map_margin is None:
        map_margin = max(float(args.start_radius_m), float(args.route_tube_radius or 0.0)) + 0.5
    if args.route_style == "complex":
        raw_points = build_complex_waypoints(
            rng, args.num_waypoints, float(args.map_size_m or 20.0), map_margin)
    else:
        raw_points = build_random_waypoints(rng, args.num_waypoints,
                                            args.segment_length_m,
                                            args.height_amp_m)
    route_points_fitted, map_fit_scale, map_margin = fit_waypoints_to_square(
        raw_points, args.map_size_m, map_margin)
    pole_waypoints = [int(v) for v in args.inspection_pole_waypoints.split(",") if v.strip()]
    poles = build_inspection_poles(route_points_fitted, pole_waypoints,
                                   args.inspection_pole_right_offset_m,
                                   args.inspection_pole_top_above_waypoint_m) \
        if args.inspection_poles else []
    control_points, route_labels, yaw_targets = build_control_route(
        route_points_fitted, args.return_to_start, poles)
    route = RouteModel(control_points)
    route.yaw_targets = yaw_targets
    route_head = route.segment_heading(0)
    off, rel, radius = sample_start_offset(rng, args.start_radius_m, route_head,
                                           quadrant=args.start_quadrant)
    start = route.wp[0] + off
    start[1] += float(rng.uniform(-args.start_vert_jitter_m, args.start_vert_jitter_m))
    yaw_error = math.radians(args.yaw_error_deg)
    start_yaw = wrap_angle(route_head + yaw_error)
    plant = KinematicAnafi(start, start_yaw)
    pose_noise_seed = args.pose_noise_seed if args.pose_noise_seed is not None else args.seed + 10007
    wind_rng = np.random.default_rng(args.wind_gust_seed if args.wind_gust_seed is not None
                                     else args.seed + 20011)
    wind_state = {
        "next_t": float(args.wind_gust_interval_s) if args.wind_gust_interval_s > 0 else math.inf,
        "last_t": None,
        "reported_t": None,
        "last_vec": np.zeros(3, dtype=float),
        "last_label": "",
    }
    pose_cfg = PerturbationConfig()
    control_src = plant
    if args.pose_source == "noisy_estimated":
        pose_cfg = PerturbationConfig(pose_noise_m=args.pose_noise_m,
                                      pose_noise_max_m=args.pose_error_max_m,
                                      yaw_noise_deg=args.camera_yaw_noise_deg,
                                      yaw_noise_max_deg=args.camera_yaw_error_max_deg,
                                      telemetry_delay_ms=telemetry_delay_ms,
                                      telemetry_delay_jitter_ms=args.telemetry_delay_jitter_ms,
                                      hloc_outage_interval_s=args.hloc_outage_interval_s,
                                      hloc_outage_duration_s=args.hloc_outage_duration_s,
                                      hloc_outage_start_s=args.hloc_outage_start_s,
                                      seed=pose_noise_seed)
        control_src = PerturbedPoseSource(plant, pose_cfg)
    ctrl = make_controller(args.algorithm, route, {
        "min_segment_time_s": args.min_segment_time_s,
        "max_segment_time_s": args.max_segment_time_s,
        "waypoint_hover_s": args.waypoint_hover_s,
        "route_tube_radius": args.route_tube_radius,
        "route_tube_segment_window": args.route_tube_segment_window,
        "route_tube_exit_s": args.route_tube_exit_s,
        "route_tube_exit_updates": args.route_tube_exit_updates,
        "route_tube_initial_grace_s": args.route_tube_initial_grace_s,
        "arrival_radius": args.arrival_radius,
        "arrival_vertical_radius": args.arrival_vertical_radius,
        "camera_yaw_offset_deg": args.camera_yaw_offset_deg,
        "pcmd_rate_limit_pct_per_s": args.pcmd_rate_limit_pct_per_s,
        "max_pose_age_s": args.max_pose_age_s,
        "pose_loss_short_s": args.pose_loss_short_s,
        "lost_abort_s": args.lost_abort_s,
        "final_landing_radius": args.final_landing_radius,
        "final_landing_hold_s": args.final_landing_hold_s,
        "final_landing_max_est_speed": args.final_landing_max_est_speed,
        "final_landing_yaw_stable_deg": args.final_landing_yaw_stable_deg,
    })

    meta = {
        "trial_id": 0,
        "algorithm_name": args.algorithm,
        "anafi_profile_id": ANAFI_PROFILE.profile_id,
        "horizontal_control_mode": getattr(ctrl.p, "horizontal_control_mode", "n/a"),
        "backend": plant.backend_name,
        "control_pose_source": args.pose_source,
        "pose_perturbation": pose_cfg.label,
        "telemetry_delay_is_lower_bound": delay_is_lower_bound,
        "scale_mode": "route_map_units",
    }

    def sink(rec):
        rec.update(_ground_truth_augment(route, plant))
        rec["gt_yaw"] = plant.yaw()
        if (wind_state["last_t"] is not None and
                wind_state["reported_t"] != wind_state["last_t"] and
                abs(plant.t - wind_state["last_t"]) <= DT * 1.5):
            vec = wind_state["last_vec"]
            rec["wind_gust"] = True
            rec["wind_direction"] = wind_state["last_label"]
            rec["wind_magnitude"] = float(np.linalg.norm(vec))
            wind_state["reported_t"] = wind_state["last_t"]

    def advance(dt):
        plant.step(dt)
        if args.wind_gust_m <= 0 or args.wind_gust_interval_s <= 0:
            return
        while plant.t + 1e-9 >= wind_state["next_t"]:
            vec, label = wind_gust_vector(plant.yaw(), wind_rng, args.wind_gust_m)
            plant.pos += vec
            wind_state["last_t"] = plant.t
            wind_state["last_vec"] = vec
            wind_state["last_label"] = label
            wind_state["next_t"] += float(args.wind_gust_interval_s)

    ticks = run_trial(
        ctrl=ctrl, route=route,
        get_pose=control_src.get_pose, get_yaw=control_src.yaw,
        get_vel=control_src.velocity_ned,
        send_pcmd=plant.send_pcmd, now_fn=lambda: plant.t, advance_fn=advance,
        duration_s=args.duration, yaw_sign=1,
        hard_stop_ct_m=args.max_allowed_error_m + args.start_radius_m + 4.0,
        trial_meta=meta, tick_sink=sink)

    for rec in ticks:
        if "cross_track_true" in rec:
            rec["cross_track_control"] = rec.get("cross_track")
            rec["cross_track_perceived"] = rec.get("cross_track")
            rec["tube_distance_control"] = rec.get("tube_distance")
            rec["cross_track"] = rec["cross_track_true"]
            rec["route_progress"] = rec.get("route_progress_true", rec.get("route_progress"))
            rec["vert_err"] = rec.get("vert_err_true", rec.get("vert_err"))
            rec["tube_distance"] = rec.get("tube_distance_true", rec.get("tube_distance"))

    metrics = compute_trial_metrics(
        ticks,
        ideal_success_error_m=args.ideal_success_error_m,
        max_allowed_error_m=args.max_allowed_error_m,
        min_corridor_time_ratio=args.min_corridor_time_ratio,
        success_hold_s=args.success_hold_s,
    )

    stride = max(1, len(ticks) // args.max_frames)
    sampled = ticks[::stride]
    if ticks and sampled[-1] is not ticks[-1]:
        sampled.append(ticks[-1])

    planned_points = [[float(p[0]), float(p[1]), float(p[2])] for p in route_points_fitted]
    route_points = [[float(p[0]), float(p[1]), float(p[2])] for p in route.wp]
    frames = [make_frame(rec) for rec in sampled]
    all_x = [p[0] for p in route_points] + [f["x"] for f in frames]
    all_z = [p[2] for p in route_points] + [f["z"] for f in frames]
    pad = max(1.0, float(args.route_tube_radius or 0.0))
    if args.map_size_m and args.map_size_m > 0:
        half = float(args.map_size_m) / 2.0
        bounds = {
            "minX": -half,
            "maxX": half,
            "minZ": -half,
            "maxZ": half,
        }
    else:
        bounds = {
            "minX": min(all_x) - pad,
            "maxX": max(all_x) + pad,
            "minZ": min(all_z) - pad,
            "maxZ": max(all_z) + pad,
        }
    return {
        "meta": {
            "algorithm": args.algorithm,
            "anafiProfile": ANAFI_PROFILE.to_metadata(),
            "backendKind": "kinematic_approximation",
            "seed": args.seed,
            "numWaypoints": args.num_waypoints,
            "controlWaypointCount": len(route_points),
            "returnToStart": args.return_to_start,
            "inspectionPoles": args.inspection_poles,
            "inspectionPoleWaypoints": pole_waypoints if args.inspection_poles else [],
            "inspectionPoleRightOffsetM": args.inspection_pole_right_offset_m,
            "inspectionPoleTopAboveWaypointM": args.inspection_pole_top_above_waypoint_m,
            "routeStyle": args.route_style,
            "distanceUnitLabel": "map u",
            "startRadiusM": args.start_radius_m,
            "startOffsetM": radius,
            "startBearingRelRouteDeg": math.degrees(rel),
            "yawErrorDeg": args.yaw_error_deg,
            "dt": DT,
            "rawTickCount": len(ticks),
            "frameCount": len(frames),
            "terminalMode": getattr(ctrl, "mode", ""),
            "routeTubeRadius": args.route_tube_radius,
            "routeTubeSegmentWindow": args.route_tube_segment_window,
            "routeTubeExitS": args.route_tube_exit_s,
            "routeTubeExitUpdates": args.route_tube_exit_updates,
            "routeTubeInitialGraceS": args.route_tube_initial_grace_s,
            "arrivalRadius": args.arrival_radius,
            "arrivalVerticalRadius": args.arrival_vertical_radius,
            "mapSizeM": args.map_size_m,
            "mapMarginM": map_margin,
            "mapFitScale": map_fit_scale,
            "poseSource": args.pose_source,
            "poseSourceLabel": ("noisy estimated pose (controller input)"
                                if args.pose_source == "noisy_estimated"
                                else "truth pose (controller input)"),
            "controlPoseUsesTruth": args.pose_source == "truth",
            "poseNoiseM": args.pose_noise_m if args.pose_source == "noisy_estimated" else 0.0,
            "poseErrorMaxM": args.pose_error_max_m if args.pose_source == "noisy_estimated" else 0.0,
            "cameraYawNoiseDeg": args.camera_yaw_noise_deg if args.pose_source == "noisy_estimated" else 0.0,
            "cameraYawErrorMaxDeg": args.camera_yaw_error_max_deg if args.pose_source == "noisy_estimated" else 0.0,
            "videoE2ELatencyMs": video_e2e_latency_ms,
            "decodeLocalizationLatencyMs": decode_localization_latency_ms,
            "telemetryDelayMs": telemetry_delay_ms if args.pose_source == "noisy_estimated" else 0.0,
            "telemetryDelayIsLowerBound": (
                delay_is_lower_bound if args.pose_source == "noisy_estimated" else False
            ),
            "telemetryDelayJitterMs": args.telemetry_delay_jitter_ms if args.pose_source == "noisy_estimated" else 0.0,
            "hlocOutageIntervalS": args.hloc_outage_interval_s if args.pose_source == "noisy_estimated" else 0.0,
            "hlocOutageDurationS": args.hloc_outage_duration_s if args.pose_source == "noisy_estimated" else 0.0,
            "hlocOutageStartS": args.hloc_outage_start_s if args.pose_source == "noisy_estimated" else 0.0,
            "maxPoseAgeS": args.max_pose_age_s,
            "poseLossShortS": args.pose_loss_short_s,
            "lostAbortS": args.lost_abort_s,
            "finalLandingRadius": args.final_landing_radius,
            "finalLandingHoldS": args.final_landing_hold_s,
            "finalLandingMaxEstSpeed": args.final_landing_max_est_speed,
            "finalLandingYawStableDeg": args.final_landing_yaw_stable_deg,
            "poseNoiseSeed": pose_noise_seed if args.pose_source == "noisy_estimated" else None,
            "windGustIntervalS": args.wind_gust_interval_s,
            "windGustM": args.wind_gust_m,
        },
        "plannedRoute": planned_points,
        "route": route_points,
        "routeWaypointLabels": route_labels,
        "poles": [{
            "waypoint": p["waypoint_label"],
            "base": [float(x) for x in p["base"]],
            "top": [float(x) for x in p["top"]],
            "heading": float(p["heading"]),
            "topUp": float(p["top_up"]),
        } for p in poles],
        "mapSizeM": args.map_size_m,
        "mapMarginM": map_margin,
        "tubeRadius": args.route_tube_radius,
        "start": [float(start[0]), float(start[1]), float(start[2])],
        "bounds": bounds,
        "metrics": {
            "pass": bool(metrics.get("pass")),
            "labels": metrics.get("labels", []),
            "failureReasons": metrics.get("failure_reasons", []),
            "converged": bool(metrics.get("converged")),
            "completedRoute": bool(metrics.get("completed_route")),
            "timeToConverge": metrics.get("time_to_converge_s"),
            "duration": metrics.get("duration_s"),
            "crossTrackMean": metrics.get("cross_track_mean_m"),
            "crossTrackP90": metrics.get("cross_track_p90_m"),
            "crossTrackMax": metrics.get("cross_track_max_m"),
            "routeProgress": metrics.get("route_progress"),
            "telemetryLostTimeS": metrics.get("telemetry_lost_time_s"),
            "yawFlipsPerMin": metrics.get("yaw_flips_per_min"),
            "stopAndGo": metrics.get("stop_and_go_score"),
        },
        "frames": frames,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Waypoint Controller Browser Sandbox</title>
<style>
:root {
  --bg: #0d1422;
  --panel: #ffffff;
  --ink: #141b2d;
  --muted: #667085;
  --line: #d8dee8;
  --route: #101828;
  --path: #15895d;
  --target: #d97706;
  --drone: #2456c5;
  --danger: #b42318;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: #f7f9fc;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.app {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  min-height: 100vh;
  background: #0d1422;
}
.stage {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  min-height: 100vh;
  overflow: hidden;
  background: #0d1422;
}
.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid rgba(216, 222, 232, 0.18);
  background: rgba(13, 20, 34, 0.88);
  backdrop-filter: blur(8px);
}
h1 {
  margin: 0;
  font-size: 16px;
  letter-spacing: 0;
}
.sub {
  margin-top: 3px;
  font-size: 12px;
  color: #b8c2d4;
}
#scene3d {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 520px;
}
#scene3d canvas {
  width: 100%;
  height: 100%;
  display: block;
  touch-action: none;
}
.controls {
  display: grid;
  grid-template-columns: auto auto auto 1fr auto;
  gap: 10px;
  align-items: center;
  padding: 12px 14px;
  border-top: 1px solid rgba(216, 222, 232, 0.18);
  background: rgba(13, 20, 34, 0.9);
  backdrop-filter: blur(8px);
}
button, select {
  min-height: 34px;
  border: 1px solid #c8cfda;
  border-radius: 6px;
  background: #ffffff;
  color: var(--ink);
  padding: 0 12px;
  font: inherit;
}
button.primary {
  color: #ffffff;
  border-color: var(--drone);
  background: var(--drone);
}
input[type="range"] { width: 100%; }
.side {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px 14px 14px 0;
  color: var(--ink);
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.panel h2 {
  margin: 0 0 10px;
  font-size: 13px;
  letter-spacing: 0;
}
.metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.metric {
  border-top: 1px solid #eef1f5;
  padding-top: 8px;
}
.label {
  color: var(--muted);
  font-size: 11px;
}
.value {
  margin-top: 2px;
  font-size: 18px;
  font-weight: 700;
  overflow-wrap: anywhere;
}
.note-value {
  font-size: 13px;
  line-height: 1.15;
}
.kv {
  display: grid;
  grid-template-columns: 124px 1fr;
  gap: 8px;
  padding: 5px 0;
  border-top: 1px solid #eef1f5;
  font-size: 13px;
}
.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 4px 9px;
  background: #eef5ff;
  color: #194185;
  font-size: 12px;
  font-weight: 700;
}
.pill.fail {
  background: #fef3f2;
  color: var(--danger);
}
.pill.pass {
  background: #e8f7ef;
  color: #067647;
}
.legend {
  display: grid;
  gap: 7px;
  color: var(--muted);
  font-size: 13px;
}
.swatch {
  display: inline-block;
  width: 32px;
  height: 3px;
  margin-right: 8px;
  vertical-align: middle;
  border-radius: 2px;
}
@media (max-width: 980px) {
  .app { grid-template-columns: 1fr; }
  .stage { min-height: 72vh; }
  .side { padding: 0 14px 14px; }
}
</style>
</head>
<body>
<div class="app">
  <main class="stage">
    <div class="bar">
      <div>
        <h1>Waypoint Controller Browser Sandbox</h1>
        <div id="subtitle" class="sub"></div>
      </div>
      <span id="status" class="pill"></span>
    </div>
    <div id="scene3d"></div>
    <div class="controls">
      <button id="play" class="primary">Play</button>
      <button id="stepBack">Step -</button>
      <button id="stepForward">Step +</button>
      <input id="slider" type="range" min="0" max="0" value="0">
      <select id="speed" aria-label="Playback speed">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
        <option value="8">8x</option>
      </select>
    </div>
  </main>
  <aside class="side">
    <section class="panel">
      <h2>Run Summary</h2>
      <div class="metrics">
        <div class="metric"><div class="label">Cross-track mean</div><div id="mCt" class="value"></div></div>
        <div class="metric"><div class="label">Cross-track p90</div><div id="mP90" class="value"></div></div>
        <div class="metric"><div class="label">Route progress</div><div id="mProg" class="value"></div></div>
        <div class="metric"><div class="label">Converged</div><div id="mConv" class="value"></div></div>
        <div class="metric"><div class="label">Pose stale</div><div id="mPoseLost" class="value"></div></div>
        <div class="metric"><div class="label">Metric notes</div><div id="mNotes" class="value note-value"></div></div>
      </div>
    </section>
    <section class="panel">
      <h2>Current Tick</h2>
      <div class="kv"><div class="label">time</div><div id="tTime"></div></div>
      <div class="kv"><div class="label">state</div><div id="tState"></div></div>
        <div class="kv"><div class="label">move mode</div><div id="tMove"></div></div>
        <div class="kv"><div class="label">segment target</div><div id="tSeg"></div></div>
        <div class="kv"><div class="label">pose source</div><div id="tPoseSource"></div></div>
        <div class="kv"><div class="label">pose error</div><div id="tPoseErr"></div></div>
        <div class="kv"><div class="label">pose age</div><div id="tPoseAge"></div></div>
        <div class="kv"><div class="label">pose loss</div><div id="tPoseLoss"></div></div>
        <div class="kv"><div class="label">yaw error</div><div id="tYawErr"></div></div>
        <div class="kv"><div class="label">wind gust</div><div id="tWind"></div></div>
        <div class="kv"><div class="label">PCMD</div><div id="tPcmd"></div></div>
        <div class="kv"><div class="label">body vector</div><div id="tBody"></div></div>
        <div class="kv"><div class="label">tube distance</div><div id="tTube"></div></div>
        <div class="kv"><div class="label">tube outside</div><div id="tTubeOutside"></div></div>
        <div class="kv"><div class="label">tube safety</div><div id="tTubeSafety"></div></div>
        <div class="kv"><div class="label">cross-track</div><div id="tCt"></div></div>
        <div class="kv"><div class="label">target distance</div><div id="tDist"></div></div>
        <div class="kv"><div class="label">landing gate</div><div id="tLanding"></div></div>
        <div class="kv"><div class="label">abort reason</div><div id="tAbort"></div></div>
    </section>
    <section class="panel">
      <h2>Legend</h2>
      <div class="legend">
        <div><span class="swatch" style="background:#101828"></span>planned route and waypoints</div>
        <div><span class="swatch" style="background:#facc15"></span>20 x 20 map-unit boundary</div>
        <div><span class="swatch" style="background:#38bdf8"></span>route safety tube</div>
        <div><span class="swatch" style="background:#15895d"></span>simulated drone path</div>
        <div><span class="swatch" style="background:#f43f5e"></span>controller estimated pose, hidden</div>
        <div><span class="swatch" style="background:#a16207"></span>inspection pole</div>
        <div><span class="swatch" style="background:#d97706"></span>current target waypoint</div>
        <div><span class="swatch" style="background:#2456c5"></span>drone body heading</div>
        <div><span class="swatch" style="background:#7c3aed"></span>body-frame command vector</div>
      </div>
    </section>
  </aside>
</div>
<script type="importmap">
{"imports":{"three":"./vendor/three/three.module.js"}}
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";

const DATA = __DATA__;
window.__sandboxData = DATA;
const frames = DATA.frames;
let idx = 0;
let playing = false;
let lastTs = 0;
let carry = 0;
const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
const sceneRoot = document.getElementById("scene3d");
const unitLabel = DATA.meta.distanceUnitLabel || "map u";
slider.max = Math.max(0, frames.length - 1);

function fmt(v, d = 2) {
  return Number.isFinite(v) ? Number(v).toFixed(d) : "n/a";
}
function up(y) {
  return -y;
}
function toV3(p) {
  return new THREE.Vector3(p[0], -p[1], p[2]);
}
function frameV3(r) {
  return new THREE.Vector3(r.x, up(r.y), r.z);
}
function perceivedV3(r) {
  return new THREE.Vector3(r.px, up(r.py), r.pz);
}
function routeLabel(idx) {
  if (idx === null || idx === undefined) return "n/a";
  return (DATA.routeWaypointLabels && DATA.routeWaypointLabels[idx]) || ("W" + (Number(idx) + 1));
}
function makeLine(points, color, width = 2) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color, linewidth: width });
  return new THREE.Line(geometry, material);
}
function makeTextSprite(text, color = "#f8fafc") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 96;
  const c = canvas.getContext("2d");
  c.font = "600 42px Inter, Arial";
  c.fillStyle = "rgba(7, 12, 22, 0.72)";
  c.fillRect(0, 14, 256, 62);
  c.fillStyle = color;
  c.textAlign = "center";
  c.textBaseline = "middle";
  c.fillText(text, 128, 46);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.8, 0.3, 1);
  return sprite;
}
function makeArrow(color) {
  return new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, color, 0.18, 0.09);
}
function makeTubeBetween(a, b, radius, material) {
  const d = new THREE.Vector3().subVectors(b, a);
  const len = d.length();
  if (len < 1e-6) return null;
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, len, 32, 1, true), material);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  return mesh;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1422);
scene.fog = new THREE.Fog(0x0d1422, 40, 90);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
  preserveDrawingBuffer: true
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(sceneRoot.clientWidth, sceneRoot.clientHeight);
sceneRoot.appendChild(renderer.domElement);

const center = new THREE.Vector3(
  (DATA.bounds.minX + DATA.bounds.maxX) / 2,
  1.5,
  (DATA.bounds.minZ + DATA.bounds.maxZ) / 2
);
const span = Math.max(DATA.bounds.maxX - DATA.bounds.minX, DATA.bounds.maxZ - DATA.bounds.minZ, 8);
const camera = new THREE.PerspectiveCamera(48, 1, 0.05, 300);
camera.position.set(center.x - span * 0.8, Math.max(8, span * 0.75), center.z + span * 1.25);
camera.lookAt(center);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = false;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 2.5;
controls.maxDistance = 90;
controls.update();
window.__sandbox3d = { camera, controls, renderer, scene };

scene.add(new THREE.HemisphereLight(0xe8f2ff, 0x2c3443, 1.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.8);
sun.position.set(8, 18, 11);
scene.add(sun);

const grid = new THREE.GridHelper(Math.ceil(span + 8), Math.ceil(span + 8), 0x4b5563, 0x243044);
grid.position.set(center.x, 0, center.z);
scene.add(grid);

const axes = new THREE.AxesHelper(1.6);
axes.position.set(DATA.bounds.minX + 0.5, 0.02, DATA.bounds.minZ + 0.5);
scene.add(axes);

if (Number(DATA.mapSizeM || 0) > 0) {
  const b = DATA.bounds;
  const boundary = [
    new THREE.Vector3(b.minX, 0.035, b.minZ),
    new THREE.Vector3(b.maxX, 0.035, b.minZ),
    new THREE.Vector3(b.maxX, 0.035, b.maxZ),
    new THREE.Vector3(b.minX, 0.035, b.maxZ),
    new THREE.Vector3(b.minX, 0.035, b.minZ)
  ];
  scene.add(makeLine(boundary, 0xfacc15, 3));
}

const routePoints = DATA.route.map(toV3);
const tubeRadius = Number(DATA.tubeRadius || 0);
if (tubeRadius > 0) {
  const tubeGroup = new THREE.Group();
  const tubeMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.16,
    roughness: 0.7,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  const capMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.12,
    roughness: 0.7,
    depthWrite: false
  });
  for (let i = 0; i < routePoints.length - 1; i++) {
    const segTube = makeTubeBetween(routePoints[i], routePoints[i + 1], tubeRadius, tubeMat);
    if (segTube) tubeGroup.add(segTube);
  }
  for (const p of routePoints) {
    const cap = new THREE.Mesh(new THREE.SphereGeometry(tubeRadius, 32, 16), capMat);
    cap.position.copy(p);
    tubeGroup.add(cap);
  }
  scene.add(tubeGroup);
}
scene.add(makeLine(routePoints, 0xf8fafc, 4));

if (Array.isArray(DATA.poles)) {
  const poleMat = new THREE.MeshStandardMaterial({ color: 0xa16207, roughness: 0.65 });
  const poleTopMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.45 });
  for (const pole of DATA.poles) {
    const base = toV3(pole.base);
    const top = toV3(pole.top);
    const poleMesh = makeTubeBetween(base, top, 0.07, poleMat);
    if (poleMesh) scene.add(poleMesh);
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.13, 18, 12), poleTopMat);
    cap.position.copy(top);
    scene.add(cap);
    const label = makeTextSprite(pole.waypoint + " pole", "#fde68a");
    label.position.copy(top).add(new THREE.Vector3(0, 0.38, 0));
    scene.add(label);
  }
}

const trailLine = makeLine([frameV3(frames[0] || { x: 0, y: -2, z: 0 })], 0x19a974, 4);
scene.add(trailLine);

const wpMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.55 });
const startMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.55 });
const plannedRoute = DATA.plannedRoute || DATA.route;
plannedRoute.forEach((p, i) => {
  const pos = toV3(p);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 16), i === 0 ? startMat : wpMat);
  marker.position.copy(pos);
  scene.add(marker);
  const label = makeTextSprite("W" + (i + 1));
  label.position.copy(pos).add(new THREE.Vector3(0, 0.45, 0));
  scene.add(label);
});

const targetMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.2, 32, 18),
  new THREE.MeshStandardMaterial({ color: 0xd97706, emissive: 0x4a2300, roughness: 0.35 })
);
scene.add(targetMarker);

const drone = new THREE.Group();
const body = new THREE.Mesh(
  new THREE.ConeGeometry(0.22, 0.55, 3),
  new THREE.MeshStandardMaterial({ color: 0x2f6fed, roughness: 0.32, metalness: 0.1 })
);
body.rotation.z = -Math.PI / 2;
drone.add(body);
const bodyBar = new THREE.Mesh(
  new THREE.BoxGeometry(0.18, 0.08, 0.72),
  new THREE.MeshStandardMaterial({ color: 0x93c5fd, roughness: 0.4 })
);
drone.add(bodyBar);
scene.add(drone);

const headingArrow = makeArrow(0x4da3ff);
const commandArrow = makeArrow(0xa855f7);
scene.add(headingArrow);
scene.add(commandArrow);

const perceivedMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.16, 20, 14),
  new THREE.MeshStandardMaterial({ color: 0xf43f5e, emissive: 0x4a0c18, roughness: 0.35 })
);
scene.add(perceivedMarker);
const perceivedLine = makeLine([new THREE.Vector3(), new THREE.Vector3()], 0xf43f5e, 2);
perceivedLine.material.transparent = true;
perceivedLine.material.opacity = 0.72;
scene.add(perceivedLine);
const showEstimatedPoseMarker = false;
perceivedMarker.visible = false;
perceivedLine.visible = false;

const targetLine = makeLine([new THREE.Vector3(), new THREE.Vector3()], 0xd97706, 2);
targetLine.material.transparent = true;
targetLine.material.opacity = 0.55;
scene.add(targetLine);

function resizeRenderer() {
  const w = Math.max(1, sceneRoot.clientWidth);
  const h = Math.max(1, sceneRoot.clientHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resizeRenderer);
resizeRenderer();

function updatePanel(r) {
  const m = DATA.metrics;
  const status = document.getElementById("status");
  const aborted = DATA.meta.terminalMode === "ABORT_OR_MANUAL";
  status.textContent = aborted ? "ABORT" : (m.pass ? "PASS" : "CHECK");
  status.className = "pill " + (!aborted && m.pass ? "pass" : "fail");
  document.getElementById("mCt").textContent = fmt(m.crossTrackMean, 3) + " " + unitLabel;
  document.getElementById("mP90").textContent = fmt(m.crossTrackP90, 3) + " " + unitLabel;
  document.getElementById("mProg").textContent = fmt((m.routeProgress || 0) * 100, 1) + "%";
  document.getElementById("mConv").textContent = m.converged ? fmt(m.timeToConverge, 1) + " s" : "no";
  document.getElementById("mPoseLost").textContent = fmt(m.telemetryLostTimeS, 1) + " s";
  document.getElementById("mNotes").textContent =
    (m.failureReasons && m.failureReasons.length) ? m.failureReasons.join(", ") : "none";
  document.getElementById("tTime").textContent = fmt(r.t, 2) + " s";
  document.getElementById("tState").textContent = r.mode;
  document.getElementById("tMove").textContent = r.moveMode || "n/a";
  document.getElementById("tSeg").textContent = "seg " + r.segIndex + " -> " + routeLabel(r.targetWp);
  document.getElementById("tPoseSource").textContent = DATA.meta.poseSourceLabel || "n/a";
  document.getElementById("tPoseErr").textContent =
    r.poseAvailable ? fmt(r.poseError, 3) + " " + unitLabel : "no pose";
  document.getElementById("tPoseAge").textContent =
    r.poseAgeS === null ? "n/a" : fmt(r.poseAgeS, 2) + "s / " + fmt(DATA.meta.maxPoseAgeS, 2) + "s";
  document.getElementById("tPoseLoss").textContent =
    r.poseLossStage || (r.telemetryStale ? "stale" : "ok");
  document.getElementById("tYawErr").textContent =
    fmt(r.yawErrorDeg, 2) + " deg";
  document.getElementById("tWind").textContent =
    r.windGust ? (r.windDirection + " " + fmt(r.windMagnitude, 2) + " " + unitLabel) : "none";
  document.getElementById("tPcmd").textContent = "roll " + r.roll + ", pitch " + r.pitch + ", yaw " + r.yawCmd + ", gaz " + r.gaz;
  document.getElementById("tBody").textContent = "fwd " + fmt(r.bodyFwd, 2) + ", right " + fmt(r.bodyRight, 2) + ", up " + fmt(r.bodyUp, 2);
  document.getElementById("tTube").textContent =
    "true " + fmt(r.tubeDistance, 3) + " " + unitLabel + ", control " +
    fmt(r.tubeDistanceControl, 3) + " " + unitLabel + " / " + fmt(DATA.tubeRadius, 2);
  document.getElementById("tTubeOutside").textContent =
    fmt(r.tubeExitS, 2) + "s / " + fmt(DATA.meta.routeTubeExitS, 2) + "s, " +
    r.tubeExitUpdates + " / " + (r.tubeExitUpdateLimit || DATA.meta.routeTubeExitUpdates || 0) + " updates";
  document.getElementById("tTubeSafety").textContent =
    r.tubeSafetyActive ? "active" : ("initial grace " + fmt(r.tubeGraceSRemaining, 1) + "s");
  document.getElementById("tCt").textContent =
    "true " + fmt(r.crossTrack, 3) + " " + unitLabel + ", control " +
    fmt(r.crossTrackControl, 3) + " " + unitLabel;
  document.getElementById("tDist").textContent = fmt(r.dToWp, 3) + " " + unitLabel;
  if (r.finalLandingPending || r.finalLandingReady) {
    document.getElementById("tLanding").textContent =
      (r.finalLandingReady ? "ready" : "pending") +
      " d " + fmt(r.finalLandingDistance, 3) + " / " + fmt(r.finalLandingRadius, 2) +
      ", hold " + fmt(r.finalLandingHoldS, 2) + " / " + fmt(r.finalLandingRequiredHoldS, 2) + "s" +
      ", speed " + fmt(r.finalLandingEstSpeed, 2) + " / " + fmt(r.finalLandingMaxEstSpeed, 2);
  } else {
    document.getElementById("tLanding").textContent = "n/a";
  }
  document.getElementById("tAbort").textContent = r.abortReason || "n/a";
}
function draw() {
  const r = frames[idx];
  const pos = frameV3(r);
  drone.position.copy(pos);
  drone.rotation.set(0, -r.yaw, 0);
  const perceived = perceivedV3(r);
  const showPerceived = showEstimatedPoseMarker && Boolean(r.poseAvailable);
  perceivedMarker.visible = showPerceived;
  perceivedLine.visible = showPerceived;
  if (showPerceived) {
    perceivedMarker.position.copy(perceived);
    perceivedLine.geometry.setFromPoints([pos, perceived]);
  }
  const heading = new THREE.Vector3(Math.cos(r.yaw), 0, Math.sin(r.yaw)).normalize();
  headingArrow.position.copy(pos);
  headingArrow.setDirection(heading);
  headingArrow.setLength(0.9, 0.18, 0.09);
  const cmd = new THREE.Vector3(
    r.bodyFwd * Math.cos(r.yaw) - r.bodyRight * Math.sin(r.yaw),
    r.bodyUp,
    r.bodyFwd * Math.sin(r.yaw) + r.bodyRight * Math.cos(r.yaw)
  );
  commandArrow.position.copy(pos);
  if (cmd.length() > 1e-6) {
    commandArrow.visible = true;
    commandArrow.setDirection(cmd.clone().normalize());
    commandArrow.setLength(Math.min(2.2, Math.max(0.35, cmd.length() * 0.32)), 0.18, 0.09);
  } else {
    commandArrow.visible = false;
  }
  if (r.targetX === null) {
    targetMarker.visible = false;
    targetLine.visible = false;
  } else {
    const t = new THREE.Vector3(r.targetX, up(r.targetY), r.targetZ);
    targetMarker.visible = true;
    targetMarker.position.copy(t);
    targetLine.visible = true;
    targetLine.geometry.setFromPoints([pos, t]);
  }
  trailLine.geometry.setFromPoints(frames.slice(0, idx + 1).map(frameV3));
  updatePanel(r);
}
function advance(n) {
  idx = Math.max(0, Math.min(frames.length - 1, idx + n));
  slider.value = idx;
  draw();
}
function loop(ts) {
  if (!lastTs) lastTs = ts;
  const dt = ts - lastTs;
  lastTs = ts;
  if (playing) {
    carry += dt * Number(document.getElementById("speed").value);
    while (carry >= 50) {
      carry -= 50;
      if (idx >= frames.length - 1) {
        playing = false;
        playBtn.textContent = "Play";
        break;
      }
      advance(1);
    }
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
document.getElementById("subtitle").textContent =
  DATA.meta.algorithm + " | " + DATA.meta.numWaypoints + " waypoints | seed " + DATA.meta.seed +
  " | " + (DATA.meta.returnToStart ? "out-and-back " : "") +
  (DATA.meta.routeStyle || "route") + " route | map " + fmt(DATA.meta.mapSizeM, 0) + " " + unitLabel + " square" +
  " | start offset " + fmt(DATA.meta.startOffsetM, 2) + " " + unitLabel +
  " | yaw error " + fmt(DATA.meta.yawErrorDeg, 1) + " deg";
playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
};
document.getElementById("stepBack").onclick = () => advance(-1);
document.getElementById("stepForward").onclick = () => advance(1);
slider.oninput = () => {
  idx = Number(slider.value);
  draw();
};
window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    playBtn.click();
  } else if (event.key === "ArrowLeft") {
    advance(-1);
  } else if (event.key === "ArrowRight") {
    advance(1);
  }
});
draw();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def write_html(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"))
    out_path.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")


def ensure_three_vendor(out_dir: Path) -> None:
    """Copy the local Three.js runtime needed by the static sandbox."""
    candidates = [
        Path(__file__).resolve().parent / "node_modules" / "three",
        Path.cwd() / "node_modules" / "three",
        Path("/home/allen/.hermes/hermes-agent/node_modules/three"),
    ]
    three_root = next((p for p in candidates if (p / "build" / "three.module.js").exists()), None)
    if three_root is None:
        raise FileNotFoundError(
            "Three.js was not found locally. Install it under this experiment "
            "folder with `npm install three`, then rerun make_browser_sandbox.py."
        )

    vendor_dir = out_dir / "vendor" / "three"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(three_root / "build" / "three.module.js", vendor_dir / "three.module.js")
    shutil.copy2(three_root / "build" / "three.core.js", vendor_dir / "three.core.js")
    shutil.copy2(
        three_root / "examples" / "jsm" / "controls" / "OrbitControls.js",
        vendor_dir / "OrbitControls.js",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build a browser waypoint-controller sandbox.")
    p.add_argument("--algorithm", choices=ALGORITHM_ORDER, default="translational_waypoint")
    p.add_argument("--route-style", choices=["complex", "random"], default="complex",
                   help="preplanned waypoint pattern for the browser sandbox")
    p.add_argument("--num-waypoints", type=int, default=10)
    p.add_argument("--return-to-start", dest="return_to_start", action="store_true",
                   default=True,
                   help="after the final planned waypoint, reverse through prior waypoints back to W1")
    p.add_argument("--no-return-to-start", dest="return_to_start", action="store_false",
                   help="fly only W1 -> final waypoint, then complete")
    p.add_argument("--inspection-poles", dest="inspection_poles", action="store_true",
                   default=True,
                   help="insert pole-inspection climb/descend actions at selected planned waypoints")
    p.add_argument("--no-inspection-poles", dest="inspection_poles", action="store_false",
                   help="disable pole-inspection actions")
    p.add_argument("--inspection-pole-waypoints", default="6,7",
                   help="comma-separated 1-based planned waypoint numbers that inspect poles")
    p.add_argument("--inspection-pole-right-offset-m", type=float, default=1.2,
                   help="pole horizontal offset to the route-right side of each inspection waypoint")
    p.add_argument("--inspection-pole-top-above-waypoint-m", type=float, default=2.0,
                   help="pole top height above its associated waypoint height")
    p.add_argument("--map-size-m", type=float, default=20.0,
                   help="fixed square map side length in route/map units; <=0 uses auto bounds")
    p.add_argument("--map-margin-m", type=float, default=None,
                   help="horizontal margin reserved inside the fixed square map")
    p.add_argument("--segment-length-m", type=float, default=3.0)
    p.add_argument("--height-amp-m", type=float, default=0.8)
    p.add_argument("--start-radius-m", type=float, default=1.5)
    p.add_argument("--start-vert-jitter-m", type=float, default=0.3)
    p.add_argument("--start-quadrant", type=int, default=1,
                   help="0=front, 1=side, 2=behind, 3=other side relative to first segment")
    p.add_argument("--yaw-error-deg", type=float, default=12.0)
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-frames", type=int, default=1800)
    p.add_argument("--min-segment-time-s", type=float, default=0.6)
    p.add_argument("--max-segment-time-s", type=float, default=90.0)
    p.add_argument("--waypoint-hover-s", type=float, default=1.0)
    p.add_argument("--arrival-radius", type=float, default=1.0,
                   help="3D waypoint arrival radius in route/map units for translational_waypoint")
    p.add_argument("--arrival-vertical-radius", type=float, default=1.0,
                   help="legacy vertical arrival radius for non-translational controllers")
    p.add_argument("--pose-source", choices=["truth", "noisy_estimated"],
                   default="noisy_estimated",
                   help="pose stream delivered to the controller; truth is for controller-only debugging")
    p.add_argument("--pose-noise-m", type=float, default=0.25,
                   help="Gaussian xyz noise sigma in route/map units, clamped by --pose-error-max-m")
    p.add_argument("--pose-error-max-m", type=float, default=1.0,
                   help="bounded random 3D pose error radius in route/map units")
    p.add_argument("--camera-yaw-noise-deg", type=float, default=5.0,
                   help="Gaussian camera/body yaw estimate noise sigma, clamped by --camera-yaw-error-max-deg")
    p.add_argument("--camera-yaw-error-max-deg", type=float, default=20.0,
                   help="bounded camera/body yaw estimate error in degrees for --pose-source noisy_estimated")
    p.add_argument("--pose-noise-seed", type=int, default=None,
                   help="random seed for estimated-pose noise; defaults to seed + 10007")
    p.add_argument("--video-e2e-latency-ms", type=float,
                   default=ANAFI_PROFILE.video_latency_ms,
                   help="ANAFI video end-to-end latency lower bound before decode/localization")
    p.add_argument("--decode-localization-latency-ms", type=float, default=0.0,
                   help="measured decode plus MegaLoc/XFeat/PnP processing latency; 0 means unknown/not modeled")
    p.add_argument("--telemetry-delay-ms", type=float, default=None,
                   help="explicit synthetic total pose delay override; otherwise video + decode/localization")
    p.add_argument("--telemetry-delay-jitter-ms", type=float, default=100.0,
                   help="uniform +/- jitter around --telemetry-delay-ms")
    p.add_argument("--hloc-outage-interval-s", type=float, default=10.0,
                   help="periodic complete hloc outage interval; <=0 disables outages")
    p.add_argument("--hloc-outage-duration-s", type=float, default=1.0,
                   help="duration of each periodic complete hloc outage")
    p.add_argument("--hloc-outage-start-s", type=float, default=10.0,
                   help="time of the first periodic complete hloc outage")
    p.add_argument("--wind-gust-interval-s", type=float, default=5.0,
                   help="seconds between simulated wind drift impulses; <=0 disables wind")
    p.add_argument("--wind-gust-m", type=float, default=0.5,
                   help="position-only wind drift impulse magnitude in route/map units")
    p.add_argument("--wind-gust-seed", type=int, default=None,
                   help="random seed for wind drift directions; defaults to seed + 20011")
    p.add_argument("--route-tube-radius", type=float, default=1.0,
                   help="full-3D safety tube radius in route/map units; exits abort to manual hover")
    p.add_argument("--route-tube-segment-window", type=int, default=1,
                   help="tube checks active segment +/- this many segments")
    p.add_argument("--route-tube-exit-s", type=float, default=0.8,
                   help="continuous outside-tube seconds required before abort")
    p.add_argument("--route-tube-exit-updates", type=int, default=6,
                   help="effective pose updates outside tube required before abort")
    p.add_argument("--route-tube-initial-grace-s", type=float, default=10.0,
                   help="allow initial rejoin outside tube for this many seconds")
    p.add_argument("--camera-yaw-offset-deg", type=float, default=0.0)
    p.add_argument("--pcmd-rate-limit-pct-per-s", type=float, default=200.0)
    p.add_argument("--max-pose-age-s", type=float, default=0.6,
                   help="maximum hloc update age before hover/lost handling")
    p.add_argument("--pose-loss-short-s", type=float, default=1.0,
                   help="pose-age threshold for short loss stage before medium wait")
    p.add_argument("--lost-abort-s", type=float, default=8.0,
                   help="pose-age threshold for long loss manual handoff")
    p.add_argument("--final-landing-radius", type=float, default=0.8,
                   help="stricter final waypoint radius in route/map units")
    p.add_argument("--final-landing-hold-s", type=float, default=0.5,
                   help="continuous valid final-landing hold time before should_land")
    p.add_argument("--final-landing-max-est-speed", type=float, default=1.5,
                   help="max net estimated pose speed over final hold window")
    p.add_argument("--final-landing-yaw-stable-deg", type=float, default=20.0,
                   help="max yaw estimate span over final hold window")
    p.add_argument("--ideal-success-error-m", type=float, default=0.5)
    p.add_argument("--max-allowed-error-m", type=float, default=3.0)
    p.add_argument("--min-corridor-time-ratio", type=float, default=0.8)
    p.add_argument("--success-hold-s", type=float, default=3.0)
    p.add_argument("--out", default="outputs/browser_sandbox/sandbox.html")
    args = p.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            p.error(f"--{name.replace('_', '-')} must be finite")

    def bounded(name, lo, hi, *, lo_open=False):
        value = getattr(args, name)
        if value is None:
            return
        too_low = value <= lo if lo_open else value < lo
        if too_low or value > hi:
            left = "(" if lo_open else "["
            p.error(f"--{name.replace('_', '-')} must be in {left}{lo}, {hi}]")

    bounded("num_waypoints", 2, 10_000)
    bounded("max_frames", 1, 1_000_000)
    bounded("seed", 0, 2**63 - 1)
    for name in ("pose_noise_seed", "wind_gust_seed"):
        if getattr(args, name) is not None:
            bounded(name, 0, 2**63 - 1)
    bounded("start_quadrant", 0, 3)
    bounded("duration", 0.0, 3600.0, lo_open=True)
    for name in ("video_e2e_latency_ms", "decode_localization_latency_ms",
                 "telemetry_delay_ms", "telemetry_delay_jitter_ms"):
        bounded(name, 0.0, 60_000.0)
    for name in ("pose_noise_m", "pose_error_max_m", "camera_yaw_noise_deg",
                 "camera_yaw_error_max_deg", "hloc_outage_interval_s",
                 "hloc_outage_duration_s", "hloc_outage_start_s",
                 "wind_gust_interval_s", "wind_gust_m", "start_radius_m",
                 "start_vert_jitter_m", "route_tube_exit_s",
                 "route_tube_initial_grace_s", "max_pose_age_s",
                 "pose_loss_short_s", "lost_abort_s"):
        bounded(name, 0.0, 60_000.0)
    if args.route_tube_radius is not None:
        bounded("route_tube_radius", 0.0, 1000.0, lo_open=True)
    bounded("route_tube_segment_window", 0, 100_000)
    bounded("route_tube_exit_updates", 1, 100_000)
    if args.max_segment_time_s < args.min_segment_time_s:
        p.error("--max-segment-time-s must be >= --min-segment-time-s")
    return args


def main() -> None:
    args = parse_args()
    data = run_sandbox(args)
    out_path = Path(args.out)
    ensure_three_vendor(out_path.parent)
    write_html(data, out_path)
    index_path = out_path.parent / "index.html"
    if out_path.name != "index.html":
        write_html(data, index_path)
    m = data["metrics"]
    print(f"[sandbox] wrote {out_path}")
    if out_path.name != "index.html":
        print(f"[sandbox] wrote {index_path}")
    print(f"[sandbox] {data['meta']['algorithm']} pass={m['pass']} "
          f"complete={m['completedRoute']} progress={m['routeProgress']:.2f} "
          f"ct_mean={m['crossTrackMean']:.3f}{data['meta']['distanceUnitLabel']} "
          f"frames={data['meta']['frameCount']}")


if __name__ == "__main__":
    main()
