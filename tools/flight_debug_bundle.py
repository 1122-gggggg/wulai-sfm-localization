#!/usr/bin/env python3
"""Turn-gate debug bundle for one flight session (read-only).

Why this exists: the AUTO turn gate spins in place (pure-yaw PCMD) while the
signed yaw error stays outside the tolerance band. The old logs showed only
the spin; the error had to be recomputed offline under an assumed frame, so
nobody could tell a walking offset (fusion fault) from a walking target
bearing (position drift), or a hardware yaw drift from a software gate fault.

Reads the four session JSONL streams plus the manifest/inventories and writes
one self-contained ``debug_bundle.json`` (plus this text report) into the
session directory. New sessions log the reconciliation fields directly
(``yaw_target_deg`` / ``yaw_error_deg`` / ``visual_yaw_deg`` /
``attitude_map_deg`` / ``heading_offset_deg`` on ``auto_route_tick``,
``camera_axes_world`` / ``camera_forward_world`` on ``pose_result``);
older sessions fall back to recomputation under the legacy [x,z] azimuth,
flagged per-section in the bundle via ``"frame": "legacy_assumption"``.

Usage:
    python3 tools/flight_debug_bundle.py <session_dir> [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                rows.append(record)
    return rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[int(rank)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def _legacy_target_bearing_deg(pose: list[float], target: list[float]) -> float | None:
    """Fallback target bearing under the legacy [x,z] azimuth (deg)."""
    try:
        dx = float(target[0]) - float(pose[0])
        dz = float(target[2]) - float(pose[2])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(dx) or not math.isfinite(dz):
        return None
    if math.hypot(dx, dz) < 1e-12:
        return None
    return math.degrees(math.atan2(dz, dx))


def _signed_err_deg(bearing_deg: float, heading_deg: float) -> float:
    return math.degrees(_wrap(math.radians(bearing_deg - heading_deg)))


def _angle_span_deg(values: list[float]) -> float:
    unwrapped = [values[0]]
    for value in values[1:]:
        unwrapped.append(unwrapped[-1] + math.degrees(_wrap(math.radians(value - unwrapped[-1]))))
    return max(unwrapped) - min(unwrapped)


def analyze_turn_gate(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-tick turn accounting with a logged-vs-recomputed error cross-check."""
    out: dict[str, Any] = {
        "tick_count": len(ticks),
        "phases": dict(Counter(str(t.get("pcmd_phase")) for t in ticks)),
        "logged_error_ticks": 0,
        "recomputed_error_ticks": 0,
        "error_agreement_deg": None,
        "max_abs_yaw_error_deg": None,
        "turn_seconds": 0.0,
        "translate_seconds": 0.0,
        "convergence": None,
    }
    diffs: list[float] = []
    max_err: float | None = None
    for tick in ticks:
        logged = _finite(tick.get("yaw_error_deg"))
        if logged is not None:
            out["logged_error_ticks"] += 1
            max_err = abs(logged) if max_err is None else max(max_err, abs(logged))
        heading = _finite(tick.get("heading_deg"))
        pose = tick.get("pose_u")
        target = tick.get("target_u")
        recomputed = None
        if heading is not None and isinstance(pose, list) and isinstance(target, list):
            bearing = _finite(tick.get("yaw_target_deg"))
            if bearing is None:
                bearing = _legacy_target_bearing_deg(pose, target)
            if bearing is not None:
                recomputed = _signed_err_deg(bearing, heading)
                if tick.get("yaw_error_deg") is None:
                    tick["yaw_error_recomputed_deg"] = round(recomputed, 2)
                    tick["yaw_error_frame"] = "legacy_assumption"
                    out["recomputed_error_ticks"] += 1
                else:
                    diffs.append(abs(_wrap(math.radians(recomputed - logged))))
    if diffs:
        diffs_sorted = sorted(math.degrees(d) for d in diffs)
        out["error_agreement_deg"] = {
            "p50": round(_percentile(diffs_sorted, 50) or 0.0, 2),
            "p95": round(_percentile(diffs_sorted, 95) or 0.0, 2),
            "max": round(diffs_sorted[-1], 2),
        }
    if max_err is not None:
        out["max_abs_yaw_error_deg"] = round(max_err, 2)
    _turn_timing(ticks, out)
    return out


def analyze_heading_reconciliation(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Split a non-converging turn into offset walk vs bearing walk vs yaw drift.

    - offset walk: visual-vs-IMU fusion offset drifts -> fusion fault.
    - bearing walk: target bearing drifts while offset holds -> position drift.
    - yaw drift: physical heading (IMU map attitude) walks under zero or
      steady yaw command, cross-checked against GPS excursion below.
    """
    visuals = [
        t
        for t in ticks
        if _finite(t.get("heading_deg")) is not None
        and _finite(t.get("imu_yaw_ned_rad")) is not None
    ]
    out: dict[str, Any] = {
        "samples": len(visuals),
        "offset_walk_deg": None,
        "bearing_walk_deg": None,
        "imu_walk_deg": None,
        "visual_imu_increment_mismatch_deg": None,
        "verdict": "insufficient_data",
    }
    if len(visuals) < 2:
        return out
    offsets: list[float] = []
    for tick in visuals:
        heading = math.radians(float(tick["heading_deg"]))
        attitude = math.pi / 2.0 - float(tick["imu_yaw_ned_rad"])
        offsets.append(math.degrees(_wrap(heading - attitude)))
    bearings = _heading_bearings(visuals)
    imus = [math.degrees(math.pi / 2.0 - float(t["imu_yaw_ned_rad"])) for t in visuals]
    out["offset_walk_deg"] = round(_angle_span_deg(offsets), 2)
    if bearings:
        out["bearing_walk_deg"] = round(_angle_span_deg(bearings), 2)
    out["imu_walk_deg"] = round(_angle_span_deg(imus), 2)
    mismatches: list[float] = []
    for first, second in zip(visuals, visuals[1:]):
        visual_delta = _wrap(
            math.radians(float(second["heading_deg"])) - math.radians(float(first["heading_deg"]))
        )
        imu_delta = _wrap(
            (math.pi / 2.0 - float(second["imu_yaw_ned_rad"]))
            - (math.pi / 2.0 - float(first["imu_yaw_ned_rad"]))
        )
        mismatches.append(abs(math.degrees(_wrap(visual_delta - imu_delta))))
    mismatches.sort()
    out["visual_imu_increment_mismatch_deg"] = {
        "p50": round(_percentile(mismatches, 50) or 0.0, 3),
        "p95": round(_percentile(mismatches, 95) or 0.0, 3),
        "max": round(mismatches[-1], 2) if mismatches else None,
    }
    offset_walk = out["offset_walk_deg"] or 0.0
    bearing_walk = out["bearing_walk_deg"] or 0.0
    if offset_walk > 10.0 and offset_walk > 2.0 * max(bearing_walk, 1.0):
        out["verdict"] = "fusion_offset_walk"
    elif bearing_walk > 10.0:
        out["verdict"] = "target_bearing_walk_position_drift"
    elif (out["imu_walk_deg"] or 0.0) < 5.0:
        out["verdict"] = "heading_static_gate_blocked"
    else:
        out["verdict"] = "heading_walks_error_persists"
    if len({tick.get("target_index") for tick in visuals}) > 1:
        out["verdict"] = "multiple_targets_review_per_leg"
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    mean = math.radians((lat1 + lat2) / 2.0)
    return radius * math.hypot(dlat * math.cos(mean), dlon)


def _event_time_s(record: dict[str, Any]) -> float | None:
    value = _finite(record.get("t_mono_ns"))
    if value is not None:
        return value / 1e9
    value = _finite(record.get("t"))
    return value


def _ownership_windows(
    commands: list[dict[str, Any]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split the session clock into AUTO vs manual(RC) windows.

    Piloting-source readbacks are the authority signal: Controller owns the
    aircraft between a Controller readback and the next SkyController one.
    tracker_state transitions are too noisy for this (the 20 Hz NUDGE spam).
    """
    edges: list[tuple[float, str]] = []
    for record in commands:
        if record.get("event") != "piloting_source":
            continue
        stamp = _event_time_s(record)
        source = str(record.get("source") or "")
        if stamp is None or not source:
            continue
        edges.append((stamp, source))
    edges.sort()
    auto_windows: list[tuple[float, float]] = []
    manual_windows: list[tuple[float, float]] = []
    for index, (start, source) in enumerate(edges):
        end = edges[index + 1][0] if index + 1 < len(edges) else float("inf")
        if end <= start:
            continue
        if source == "Controller":
            auto_windows.append((start, end))
        elif source == "SkyController":
            manual_windows.append((start, end))
    return auto_windows, manual_windows


def _excursion_in_windows(
    samples: list[tuple[float, float, float]],
    windows: list[tuple[float, float]],
) -> float | None:
    """Max GPS excursion from the window-entry fix, over all windows."""
    best: float | None = None
    for start, end in windows:
        fixes = [(t, lat, lon) for t, lat, lon in samples if start <= t < end]
        if len(fixes) < 2:
            continue
        _, lat0, lon0 = fixes[0]
        excursion = max(_haversine_m(lat0, lon0, lat, lon) for _, lat, lon in fixes)
        best = excursion if best is None else max(best, excursion)
    return round(best, 2) if best is not None else None


def analyze_drift(
    telemetry: list[dict[str, Any]],
    ticks: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    """Physical drift split by who owned the aircraft: AUTO vs RC manual.

    Whole-session GPS excursion mixes the AUTO spin with the RC spin, so it
    cannot answer "does the hardware drift under pure yaw". Ownership comes
    from piloting_source readbacks; the manual pure-yaw PCMD share is counted
    from stick_axes movement flags, not from tracker_state spam.
    """
    out: dict[str, Any] = {
        "gps_excursion_m": None,
        "gps_excursion_auto_m": None,
        "gps_excursion_manual_m": None,
        "gps_samples": 0,
        "visual_excursion_u": None,
        "manual_yaw_segments": [],
    }
    gps: list[tuple[float, float, float]] = []
    for record in telemetry:
        if record.get("event") != "fused_odometry":
            continue
        lat = _finite(record.get("gps_latitude_deg"))
        lon = _finite(record.get("gps_longitude_deg"))
        t = _finite(record.get("t_mono_ns"))
        if lat is not None and lon is not None and t is not None:
            gps.append((t / 1e9, lat, lon))
    out["gps_samples"] = len(gps)
    if len(gps) >= 2:
        lat0, lon0 = gps[0][1], gps[0][2]
        out["gps_excursion_m"] = round(
            max(_haversine_m(lat0, lon0, lat, lon) for _, lat, lon in gps), 2
        )
    auto_windows, manual_windows = _ownership_windows(commands)
    out["gps_excursion_auto_m"] = _excursion_in_windows(gps, auto_windows)
    out["gps_excursion_manual_m"] = _excursion_in_windows(gps, manual_windows)
    poses: list[list[float]] = [
        t["pose_u"] for t in ticks if isinstance(t.get("pose_u"), list) and len(t["pose_u"]) == 3
    ]
    if len(poses) >= 2:
        origin = poses[0]
        out["visual_excursion_u"] = round(
            max(
                math.dist(
                    (float(p[0]), float(p[1]), float(p[2])),
                    (float(origin[0]), float(origin[1]), float(origin[2])),
                )
                for p in poses
            ),
            3,
        )
    # Manual pure-yaw segments: bounded by stick_override pairs in commands.
    overrides = [c for c in commands if c.get("event") in ("stick_override", "manual")]
    out["manual_yaw_segments"] = [
        {"t_utc": c.get("t_utc"), "event": c.get("event")} for c in overrides[:20]
    ]
    return out


def analyze_auto_runs(localization: list[dict]) -> list[dict]:
    runs = []
    by_id = {}
    current = None
    for row in localization:
        run_id = row.get("auto_run_id")
        if row.get("event") == "auto_route_plan":
            current = {"auto_run_id": run_id, "plan": row, "ticks": []}
            runs.append(current)
            if run_id:
                by_id[run_id] = current
        elif row.get("event") == "auto_route_tick":
            selected = by_id.get(run_id) if run_id else current
            if selected is None or (not run_id and row.get("step") == 0 and selected["ticks"]):
                selected = {"auto_run_id": run_id, "plan": None, "ticks": []}
                runs.append(selected)
                current = selected
                if run_id:
                    by_id[run_id] = selected
            selected["ticks"].append(row)
    return _summarize_auto_runs(runs)


def debug_input_inventory(poses: list[dict], telemetry: list[dict], session: Path) -> dict:
    dispatches = [row for row in telemetry if row.get("event") == "pcmd_dispatch"]
    odometry = [row for row in telemetry if row.get("event") == "fused_odometry"]
    summary_path = session / "session_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    manifest_path = session / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return {
        "debug_contract": manifest.get("debug_contract"),
        "sdk_dispatches": len(dispatches),
        "debug_deferred_dropped": summary.get("debug_deferred_dropped"),
        "source_stamp_samples": {
            key: sum(row.get(key) is not None for row in odometry)
            for key in ("attitude_mono_ns", "speed_mono_ns", "altitude_mono_ns", "gps_mono_ns")
        },
        "klt_diagnostic_samples": sum(row.get("klt_input_points") is not None for row in poses),
        "pnp_observation_samples": sum(bool(row.get("pnp_observation_sample")) for row in poses),
        "imu_bridge_reasons": dict(
            Counter(str(row["imu_bridge_reason"]) for row in poses if row.get("imu_bridge_reason"))
        ),
        "covariance_note": "FC covariance unavailable; PnP samples permit offline visual-noise analysis",
        "scale_note": "Map units and NED metres must not be fused without measured scale/frame calibration",
    }


def analyze_session(session: Path) -> dict[str, Any]:
    commands = read_jsonl(session / "commands.jsonl")
    localization = read_jsonl(session / "localization.jsonl")
    telemetry = read_jsonl(session / "telemetry.jsonl")
    incidents = read_jsonl(session / "incidents.jsonl")
    ticks = [r for r in localization if r.get("event") == "auto_route_tick"]
    poses = [r for r in localization if r.get("event") == "pose_result"]
    plan = next((r for r in localization if r.get("event") == "auto_route_plan"), None)
    pose_modes = Counter(str(r.get("mode") or "UNKNOWN") for r in poses)
    successes = sum(1 for r in poses if r.get("success"))
    magnetometer = [c for c in commands if c.get("event") == "magnetometer_state"]
    pcmds = [row for row in telemetry if row.get("event") == "pcmd_dispatch"]
    pcmd_source = "sdk_dispatch" if pcmds else "legacy_sampled_commands"
    if not pcmds:
        pcmds = [c for c in commands if c.get("event") == "pcmd"]
    pure_yaw = [
        c
        for c in pcmds
        if isinstance(c.get("pcmd"), list)
        and c["pcmd"][0] == 0
        and c["pcmd"][1] == 0
        and c["pcmd"][3] == 0
        and c["pcmd"][2] != 0
    ]
    report: dict[str, Any] = {
        "session": session.name,
        "has_auto": plan is not None and len(ticks) > 0,
        "tick_count": len(ticks),
        "pose_results": len(poses),
        "pose_success": successes,
        "pose_modes": dict(pose_modes),
        "magnetometer_state": magnetometer[-1] if magnetometer else None,
        "pure_yaw_pcmd_count": len(pure_yaw),
        "pcmd_count": len(pcmds),
        "pcmd_source": pcmd_source,
        "autonomy_events": [row for row in telemetry if row.get("event") == "autonomy_event"],
        "incidents": dict(Counter(str(r.get("event")) for r in incidents).most_common(10)),
    }
    if plan is not None:
        report["plan"] = {
            "arrive_radius_u": plan.get("waypoint_arrive_radius_u"),
            "drawn_waypoint_count": plan.get("drawn_waypoint_count"),
            "return_to_start": bool(plan.get("return_to_start")),
            "target_index": plan.get("target_index"),
        }
    if ticks:
        report["turn_gate"] = analyze_turn_gate(ticks)
        report["heading"] = analyze_heading_reconciliation(ticks)
    report["drift"] = analyze_drift(telemetry, ticks, commands)
    report["auto_runs"] = analyze_auto_runs(localization)
    report["debug_inputs"] = debug_input_inventory(poses, telemetry, session)
    _session_verdict(report, ticks)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"轉向除錯：{report['session']}"]
    inventory = report.get("debug_inputs", {})
    lines.append(
        f"資料：SDK dispatch {inventory.get('sdk_dispatches', 0)} 筆，"
        f"KLT 品質 {inventory.get('klt_diagnostic_samples', 0)} 筆，"
        f"PnP 樣本 {inventory.get('pnp_observation_samples', 0)} 筆；"
        f"debug 丟棄 {inventory.get('debug_deferred_dropped')}。"
    )
    if len(report.get("auto_runs", [])) > 1:
        lines.append(
            "本 session 有多段 AUTO；分段起點與參數請看 auto_runs，勿把人工操作間隔當成巡航時間。"
        )
    if report.get("verdict") == "NO_AUTO":
        return "\n".join(lines + ["此班沒有 AUTO（無 auto_route_plan/tick），只看漂移與羅盤。"])
    lines.append(
        f"AUTO {report['tick_count']} tick，定位 {report['pose_success']}/{report['pose_results']}，"
        f"純 yaw PCMD {report['pure_yaw_pcmd_count']}/{report['pcmd_count']}。"
    )
    mag = report.get("magnetometer_state") or {}
    lines.append(
        f"羅盤：required={mag.get('required')} started={mag.get('started')} "
        f"failed={mag.get('failed')}（韌體自報；valid 才能排除硬體羅盤）。"
    )
    gate = report.get("turn_gate", {})
    lines.append(
        f"轉向門：turn {gate.get('turn_seconds')}s / translate {gate.get('translate_seconds')}s，"
        f"phase {gate.get('phases')}，最大 |err| {gate.get('max_abs_yaw_error_deg')}°。"
    )
    agreement = gate.get("error_agreement_deg")
    if agreement:
        lines.append(
            f"記帳誤差對帳（記 vs 重算）：p50 {agreement['p50']}° / p95 {agreement['p95']}° / "
            f"max {agreement['max']}°（請核對座標與量測時刻是否一致）。"
        )
    elif gate.get("logged_error_ticks"):
        lines.append("記帳 yaw_error 直接可用（新 session），無需重算。")
    else:
        lines.append("舊 session：誤差為 legacy [x,z] 重算（frame=legacy_assumption）。")
    heading = report.get("heading", {})
    lines.append(
        f"對帳：offset 走 {heading.get('offset_walk_deg')}° / 方位走 "
        f"{heading.get('bearing_walk_deg')}° / IMU 走 {heading.get('imu_walk_deg')}°；"
        f"visual-IMU 增量 mismatch "
        f"{(heading.get('visual_imu_increment_mismatch_deg') or {}).get('p95')}°(p95)。"
        f"判定：{heading.get('verdict')}。"
    )
    drift = report.get("drift", {})
    lines.append(
        f"漂移：GPS 全程 {drift.get('gps_excursion_m')}m / AUTO 段 "
        f"{drift.get('gps_excursion_auto_m')}m / 手動段 "
        f"{drift.get('gps_excursion_manual_m')}m（{drift.get('gps_samples')} 樣本）/ "
        f"visual {drift.get('visual_excursion_u')}u。"
    )
    lines.append(f"判定：{report.get('verdict')}")
    if report.get("verdict") == "ROUTE_FINAL_REACHED":
        lines.append("導航已確認終點；實際降落結果另核對飛行狀態與降落事件。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="flight session directory")
    parser.add_argument("--out", type=Path, default=None, help="write debug_bundle.json here")
    args = parser.parse_args(argv)
    report = analyze_session(args.session)
    print(format_report(report))
    out = args.out or (args.session / "debug_bundle.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"bundle -> {out}")
    return 0


def _turn_timing(ticks, out):
    ordered = sorted(ticks, key=lambda t: float(t.get("t", 0.0) or 0.0))
    for tick, nxt in zip(ordered, ordered[1:]):
        span = float(nxt.get("t", 0.0) or 0.0) - float(tick.get("t", 0.0) or 0.0)
        if not math.isfinite(span) or span < 0.0 or span > 5.0:
            continue
        phase = tick.get("pcmd_phase")
        if phase == "turn":
            out["turn_seconds"] += span
        elif phase == "translate":
            out["translate_seconds"] += span
    out["turn_seconds"] = round(out["turn_seconds"], 3)
    out["translate_seconds"] = round(out["translate_seconds"], 3)
    # Convergence: first tick where the gate actually released translation
    # (phase translate / yaw_alignment_confirmed) with the error inside the
    # 20 deg band. An error-only check false-positives: a 12 deg starting
    # error is "in band" at tick 0 while the gate still spins (measured
    # 2026-09-14: 21 s of pure-yaw PCMD before the first translate tick).
    translated = False
    for tick in ordered:
        phase = tick.get("pcmd_phase")
        if phase not in ("translate", "yaw_alignment_confirmed"):
            continue
        value = _finite(tick.get("yaw_error_deg"))
        if value is None:
            value = _finite(tick.get("yaw_error_recomputed_deg"))
        if value is not None and abs(value) <= 20.0:
            out["convergence"] = {
                "tick": tick.get("step"),
                "t": tick.get("t"),
                "time_to_converge_s": round(
                    float(tick.get("t", 0.0) or 0.0) - float(ordered[0].get("t", 0.0) or 0.0),
                    3,
                ),
            }
            # A single translate burst is not a converged leg: the gate can
            # re-open mid-leg (mid_leg_realign). Only a translate-held ending
            # counts; otherwise the report names the ending phase.
            translated = True
            break
    motion_phases = [
        tick.get("pcmd_phase")
        for tick in ordered
        if tick.get("pcmd_phase") in ("turn", "translate")
    ]
    out["final_phase"] = motion_phases[-1] if motion_phases else None
    out["ever_translated"] = translated


def _heading_bearings(visuals):
    bearings: list[float] = []
    for tick in visuals:
        pose = tick.get("pose_u")
        target = tick.get("target_u")
        heading = _finite(tick.get("heading_deg"))
        if isinstance(pose, list) and isinstance(target, list) and heading is not None:
            bearing = _finite(tick.get("yaw_target_deg"))
            if bearing is None:
                bearing = _legacy_target_bearing_deg(pose, target)
            if bearing is not None:
                bearings.append(bearing)
    return bearings


def _summarize_auto_runs(runs):
    result = []
    for run in runs:
        if not run["ticks"]:
            continue
        targets = {}
        for tick in run["ticks"]:
            if tick.get("target_index") is not None:
                targets.setdefault(tick["target_index"], []).append(tick)
        result.append(
            {
                "auto_run_id": run["auto_run_id"],
                "plan": run["plan"],
                "tick_count": len(run["ticks"]),
                "turn_gate": analyze_turn_gate(run["ticks"]),
                "last_action": run["ticks"][-1].get("action"),
                "target_indices": list(targets),
                "targets": [
                    {
                        "target_index": target,
                        "turn_gate": analyze_turn_gate(ticks),
                        "heading": analyze_heading_reconciliation(ticks),
                    }
                    for target, ticks in targets.items()
                ],
            }
        )
    return result


def _session_verdict(report, ticks):
    if not ticks:
        report["verdict"] = "NO_AUTO"
    elif any(row.get("kind") == "auto_failed" for row in report["autonomy_events"]):
        report["verdict"] = "AUTO_FAILED"
    elif ticks[-1].get("action") in {"LAND", "final path reached -> LAND"}:
        report["verdict"] = "ROUTE_FINAL_REACHED"
    elif ticks[-1].get("pcmd_phase") in {
        "height_adjust",
        "route_rejoin",
        "waypoint_centering",
        "final_centering",
    }:
        report["verdict"] = "POSITION_CORRECTION_ACTIVE"
    elif report["turn_gate"].get("final_phase") == "translate":
        report["verdict"] = "TURN_CONVERGED"
    elif report["turn_gate"].get("convergence") is None:
        report["verdict"] = "TURN_NEVER_CONVERGED"
    else:
        report["verdict"] = "TURN_REGRESSED"
    if len(report["auto_runs"]) > 1:
        report["verdict"] = "MULTIPLE_AUTO_RUNS_SEE_DETAILS"


if __name__ == "__main__":
    raise SystemExit(main())
