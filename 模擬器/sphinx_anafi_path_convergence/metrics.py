#!/usr/bin/env python3
"""Convergence / smoothness / stability metrics for the path-convergence trials.

All distance metrics are Sphinx METERS (simulation only). None of them may be
reinterpreted as real monocular SfM map units without a calibrated scale
factor (scale_utils).
"""
from __future__ import annotations

import numpy as np

from controllers import (ABORT_OR_MANUAL, COMPLETED, LOST_OR_UNCERTAIN,
                         SEGMENT_FOLLOW, SEGMENT_REJOIN, WAYPOINT_HOVER)

MOTION_MODES = {SEGMENT_FOLLOW, SEGMENT_REJOIN, "INIT_REJOIN"}


def _pct(arr, q):
    return float(np.percentile(np.asarray(arr, float), q)) if len(arr) else None


def yaw_flip_count(yaw_cmds, min_mag: int = 4) -> int:
    flips, prev = 0, 0
    for c in yaw_cmds:
        if abs(c) < min_mag:
            continue
        s = 1 if c > 0 else -1
        if prev and s != prev:
            flips += 1
        prev = s
    return flips


def stop_and_go_transitions(pitches, modes, on_thresh: int = 2, off_thresh: int = 1) -> int:
    """Count moving<->stopped pitch transitions while in motion modes."""
    n, state = 0, None
    for p, m in zip(pitches, modes):
        if m not in MOTION_MODES:
            continue
        s = "go" if abs(p) >= on_thresh else ("stop" if abs(p) <= off_thresh else state)
        if state is not None and s is not None and s != state:
            n += 1
        if s is not None:
            state = s
    return n


def first_hold_time(ts, ok_flags, hold_s: float):
    """First t at which ok stays True for >= hold_s. None if never."""
    start = None
    for t, ok in zip(ts, ok_flags):
        if ok:
            if start is None:
                start = t
            if t - start >= hold_s:
                return start
        else:
            start = None
    return None


def compute_trial_metrics(ticks: list[dict], *, ideal_success_error_m: float = 0.5,
                          max_allowed_error_m: float = 3.0,
                          min_corridor_time_ratio: float = 0.8,
                          success_hold_s: float = 3.0,
                          osc_flips_per_min_limit: float = 20.0,
                          pitch_zero_ratio_limit: float = 0.5) -> dict:
    """Reduce one trial's per-tick records to the spec's metrics + pass/fail.

    Expected tick keys: t, mode, pcmd=(r,p,y,g), cross_track (may be None in
    LOST ticks), vert_err, route_progress, plus optional segment_switch,
    lateral_assist, pitch_suppressed, transitions.
    """
    if not ticks:
        return {"pass": False, "failure_reasons": ["no_ticks"], "labels": ["no_ticks"]}

    ts = [t["t"] for t in ticks]
    dur = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    dt_med = float(np.median(np.diff(ts))) if len(ts) > 2 else 0.05
    modes = [t["mode"] for t in ticks]
    ct = [t.get("cross_track") for t in ticks]
    ct_valid = [(t, c) for t, c in zip(ts, ct) if c is not None]
    vert = [abs(t["vert_err"]) for t in ticks if t.get("vert_err") is not None]
    pitches = [t["pcmd"][1] for t in ticks]
    yaws = [t["pcmd"][2] for t in ticks]
    progress = [t.get("route_progress") or 0.0 for t in ticks]

    initial_ct = ct_valid[0][1] if ct_valid else None
    trivial = initial_ct is not None and initial_ct <= ideal_success_error_m

    # convergence: inside the ideal corridor, held for success_hold_s
    ok_flags = [(c is not None and c <= ideal_success_error_m) for c in ct]
    t_conv = first_hold_time(ts, ok_flags, success_hold_s)
    converged = t_conv is not None

    # cross-track statistics after convergence (tracking quality), plus overall
    ct_post = [c for t, c in ct_valid if converged and t >= t_conv]
    ct_all = [c for _, c in ct_valid]
    in_ideal_post = [c <= ideal_success_error_m for c in ct_post]
    in_max_all = [c <= max_allowed_error_m for c in ct_all]
    corridor_ratio = float(np.mean(in_ideal_post)) if ct_post else 0.0
    in_max_ratio = float(np.mean(in_max_all)) if ct_all else 0.0

    # mode-derived counts
    follow_reached = SEGMENT_FOLLOW in modes
    completed = COMPLETED in modes
    aborted = ABORT_OR_MANUAL in modes
    abort_reason = next((t.get("abort_reason") for t in reversed(ticks)
                         if t.get("abort_reason")), None)
    hover_count = sum(1 for a, b in zip(modes[:-1], modes[1:])
                      if b == WAYPOINT_HOVER and a != WAYPOINT_HOVER)
    lost_time = sum(dt_med for m in modes if m == LOST_OR_UNCERTAIN)
    transitions = max((t.get("transitions") or 0) for t in ticks)

    switches = [t["segment_switch"] for t in ticks if t.get("segment_switch")]
    switch_meta = [(t.get("segment_switch"), t.get("switch_d_end"), t.get("switch_progress"))
                   for t in ticks if t.get("segment_switch")]
    early_switches = sum(1 for r, d, p in switch_meta
                         if r == "arrived" and d is not None and p is not None
                         and d > 1.0 and p < 0.9)
    late_switches = sum(1 for r, _d, _p in switch_meta if r == "forced_timeout")

    flips = yaw_flip_count(yaws)
    osc_per_min = flips / max(1e-9, dur / 60.0)
    sng = stop_and_go_transitions(pitches, modes)
    sng_per_min = sng / max(1e-9, dur / 60.0)
    motion_ticks = [i for i, m in enumerate(modes) if m in MOTION_MODES]
    pitch_zero_ratio = (float(np.mean([abs(pitches[i]) < 1 for i in motion_ticks]))
                        if motion_ticks else 0.0)
    suppress_time = sum(dt_med for t in ticks if t.get("pitch_suppressed"))
    lateral_count = sum(1 for t in ticks if t.get("lateral_assist"))
    pcmd_max = max(max(abs(v) for v in t["pcmd"]) for t in ticks)

    route_progress_final = max(progress) if progress else 0.0
    progress_after_conv = (route_progress_final - progress[ts.index(t_conv)]
                           if converged and t_conv in ts else None)

    # ---- failure classification (spec labels) ----
    labels, reasons = [], []
    if trivial:
        labels.append("trivial_start_on_path")
        reasons.append("trivial_start_on_path")
    if completed:
        labels.append("completed_route")
    if aborted and abort_reason == "horizontal_hard_abort":
        labels.append("diverged"); reasons.append("horizontal_hard_abort")
    if aborted and abort_reason == "vertical_profile_failure":
        labels.append("vertical_profile_failure"); reasons.append("vertical_profile_failure")
    if aborted and abort_reason == "route_tube_exit":
        labels.append("route_tube_exit"); reasons.append("route_tube_exit")
    if aborted and abort_reason == "telemetry_lost":
        labels.append("telemetry_safety_block"); reasons.append("telemetry_lost")
    if not converged:
        reasons.append("never_converged")
        if initial_ct is not None and ct_all and min(ct_all[:max(1, int(5.0 / max(dt_med,1e-3)))]) > initial_ct + 0.5:
            labels.append("heading_seed_failure")
        if not aborted:
            labels.append("diverged" if (ct_all and ct_all[-1] > max_allowed_error_m)
                          else "no_convergence")
    if osc_per_min > osc_flips_per_min_limit:
        labels.append("oscillation_failure"); reasons.append(f"yaw_flips_per_min={osc_per_min:.1f}")
    if pitch_zero_ratio > pitch_zero_ratio_limit and not completed:
        labels.append("stop_and_go_failure"); reasons.append(f"pitch_zero_ratio={pitch_zero_ratio:.2f}")
    if early_switches:
        labels.append("early_segment_switch")
    if late_switches:
        labels.append("late_segment_switch")
    if converged and corridor_ratio < min_corridor_time_ratio:
        reasons.append(f"corridor_ratio={corridor_ratio:.2f}<{min_corridor_time_ratio}")
    if converged and not follow_reached:
        reasons.append("no_rejoin_to_follow_transition")
    if lost_time > 2.0:
        reasons.append(f"telemetry_stale_{lost_time:.1f}s")
    if pcmd_max > 100:
        reasons.append("pcmd_out_of_range")

    passed = (not trivial and converged and follow_reached
              and corridor_ratio >= min_corridor_time_ratio
              and not aborted and osc_per_min <= osc_flips_per_min_limit
              and pitch_zero_ratio <= pitch_zero_ratio_limit
              and (progress_after_conv is None or progress_after_conv > 0.02
                   or route_progress_final > 0.95)
              and pcmd_max <= 100 and lost_time <= 2.0)

    return {
        "pass": bool(passed),
        "labels": labels,
        "failure_reasons": reasons,
        "trivial_start_on_path": trivial,
        "initial_cross_track_m": initial_ct,
        "converged": converged,
        "time_to_converge_s": (t_conv - ts[0]) if converged else None,
        "follow_reached": follow_reached,
        "completed_route": completed,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "route_progress": route_progress_final,
        "cross_track_mean_m": float(np.mean(ct_post)) if ct_post else
                              (float(np.mean(ct_all)) if ct_all else None),
        "cross_track_p90_m": _pct(ct_post or ct_all, 90),
        "cross_track_max_m": float(np.max(ct_all)) if ct_all else None,
        "vertical_mean_m": float(np.mean(vert)) if vert else None,
        "vertical_p90_m": _pct(vert, 90),
        "corridor_time_ratio": corridor_ratio,
        "in_max_corridor_ratio": in_max_ratio,
        "rejoin_follow_transitions": transitions,
        "yaw_flips": flips,
        "yaw_flips_per_min": osc_per_min,
        "stop_and_go_transitions": sng,
        "stop_and_go_per_min": sng_per_min,
        "stop_and_go_score": pitch_zero_ratio + sng_per_min / 60.0,
        "pitch_zero_ratio": pitch_zero_ratio,
        "pitch_suppression_time_s": suppress_time,
        "lateral_assist_ticks": lateral_count,
        "waypoint_hover_count": hover_count,
        "early_segment_switches": early_switches,
        "late_segment_switches": late_switches,
        "segment_switches": len(switches),
        "telemetry_lost_time_s": lost_time,
        "pcmd_max_abs": pcmd_max,
        "duration_s": dur,
    }


# ---------------------------------------------------------------------------
# Aggregation across trials

def _rate(items):
    return float(np.mean([1.0 if x else 0.0 for x in items])) if items else None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def aggregate_trials(trials: list[dict]) -> dict:
    """trials: [{algorithm, yaw_error_deg, perturbation, metrics, ...}]"""
    attempted = list(trials)
    trials = [t for t in attempted if t.get("valid_for_comparison", True)]
    out: dict = {
        "n_trials": len(trials),
        "n_attempted_trials": len(attempted),
        "n_invalid_trials": len(attempted) - len(trials),
        "by_algorithm": {},
    }
    algos = sorted({t["algorithm"] for t in trials})
    for a in algos:
        rows = [t for t in trials if t["algorithm"] == a]
        ms = [t["metrics"] for t in rows]
        n_seg_done = _mean([m.get("segment_switches") for m in ms])
        agg = {
            "n": len(rows),
            "pass_rate": _rate([m["pass"] for m in ms]),
            "convergence_rate": _rate([m["converged"] for m in ms]),
            "route_completion_rate": _rate([m["completed_route"] for m in ms]),
            "mean_route_progress": _mean([m["route_progress"] for m in ms]),
            "mean_segment_switches": n_seg_done,
            "time_to_converge_mean_s": _mean([m["time_to_converge_s"] for m in ms]),
            "cross_track_mean_m": _mean([m["cross_track_mean_m"] for m in ms]),
            "cross_track_p90_m": _mean([m["cross_track_p90_m"] for m in ms]),
            "cross_track_max_m": max([m["cross_track_max_m"] or 0.0 for m in ms], default=None),
            "vertical_mean_m": _mean([m["vertical_mean_m"] for m in ms]),
            "vertical_p90_m": _mean([m["vertical_p90_m"] for m in ms]),
            "corridor_time_ratio_mean": _mean([m["corridor_time_ratio"] for m in ms]),
            "yaw_flips_per_min_mean": _mean([m["yaw_flips_per_min"] for m in ms]),
            "stop_and_go_score_mean": _mean([m["stop_and_go_score"] for m in ms]),
            "rejoin_follow_transitions_mean": _mean([m["rejoin_follow_transitions"] for m in ms]),
            "pitch_suppression_time_mean_s": _mean([m["pitch_suppression_time_s"] for m in ms]),
            "lateral_assist_ticks_mean": _mean([m["lateral_assist_ticks"] for m in ms]),
            "waypoint_hover_count_mean": _mean([m["waypoint_hover_count"] for m in ms]),
            "early_segment_switches": int(sum(m["early_segment_switches"] for m in ms)),
            "late_segment_switches": int(sum(m["late_segment_switches"] for m in ms)),
            "hard_aborts": int(sum(1 for m in ms if m["aborted"])),
            "labels": sorted({l for m in ms for l in m["labels"]}),
        }
        # pass rate by initial-yaw-error bucket
        buckets: dict[str, list] = {}
        for t in rows:
            b = f"{abs(t.get('yaw_error_deg') or 0.0):g}"
            buckets.setdefault(b, []).append(t["metrics"]["pass"])
        agg["pass_rate_by_abs_yaw_deg"] = {k: _rate(v) for k, v in sorted(
            buckets.items(), key=lambda kv: float(kv[0]))}
        # pass rate by perturbation label
        pbuckets: dict[str, list] = {}
        for t in rows:
            pbuckets.setdefault(t.get("perturbation", "clean"), []).append(t["metrics"]["pass"])
        agg["pass_rate_by_perturbation"] = {k: _rate(v) for k, v in sorted(pbuckets.items())}
        out["by_algorithm"][a] = agg
    return out


def rank_algorithms(summary: dict) -> list[tuple[str, float]]:
    """Composite score for the recommendation: pass rate dominates, then
    convergence quality, smoothness and completion. Higher is better."""
    ranked = []
    for name, a in summary.get("by_algorithm", {}).items():
        if a["n"] == 0:
            continue
        cross_track = a["cross_track_mean_m"]
        if cross_track is None:
            cross_track = 1.0
        score = (100.0 * (a["pass_rate"] or 0.0)
                 + 40.0 * (a["route_completion_rate"] or 0.0)
                 + 20.0 * (a["corridor_time_ratio_mean"] or 0.0)
                 - 10.0 * min(1.0, cross_track)
                 - 0.5 * min(20.0, a["yaw_flips_per_min_mean"] or 0.0)
                 - 10.0 * min(1.0, a["stop_and_go_score_mean"] or 0.0)
                 - 2.0 * a["hard_aborts"] / max(1, a["n"]))
        ranked.append((name, round(score, 2)))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked
