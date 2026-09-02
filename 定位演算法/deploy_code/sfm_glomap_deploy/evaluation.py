#!/usr/bin/env python3
"""Evaluation — metrics for global/continuous localization, drift vs failure duration.

Must report: success rate, pos/ori error, max/avg drift, recovery time, jump, FPS, latency,
and failure duration (0.5~10s) vs drift + ablation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FrameLog:
    timestamp: float
    edm_success: bool
    retrieval_score: float | None = None
    match_count: int | None = None
    pnp_inliers: int | None = None
    reproj: float | None = None
    klt_tracked: int | None = None
    klt_ratio: float | None = None
    local_inliers: int | None = None
    position: np.ndarray | None = None
    gt_position: np.ndarray | None = None
    # --- extended observability fields ---
    yaw: float | None = None
    gt_yaw: float | None = None
    wall_ms: float | None = None
    latency_ms: float | None = None
    state: str | None = None
    tracker_state: str | None = None


@dataclass
class BenchmarkResult:
    global_success_rate: float
    continuous_success_rate: float
    position_rmse_m: float
    orientation_rmse_deg: float
    max_drift_m: float
    avg_drift_m: float
    recovery_time_s: float | None
    max_jump_m: float
    fps: float
    latency_ms: float
    failure_duration_vs_drift: dict[float, float] = field(default_factory=dict)


def _get(log, key, default=None):
    if isinstance(log, dict):
        return log.get(key, default)
    return getattr(log, key, default)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _get_state_str(log) -> str:
    s = _get(log, "state", None)
    if s is None:
        s = _get(log, "tracker_state", None)
    if s is None:
        # fallback to edm_success
        ok = _get(log, "edm_success", False)
        return "TRACK" if bool(ok) else "LOST"
    try:
        return str(s).strip().upper()
    except Exception:
        return "LOST" if not bool(_get(log, "edm_success", False)) else "TRACK"


def _compute_orientation_rmse(logs: list[FrameLog]) -> float:
    errs_deg: list[float] = []
    for l in logs:
        y = _get(l, "yaw", None)
        if y is None:
            y = _get(l, "yaw_rad", None)
        gy = _get(l, "gt_yaw", None)
        if gy is None:
            gy = _get(l, "gt_yaw_rad", None)
        if y is None or gy is None:
            continue
        try:
            yf = float(y)
            gyf = float(gy)
        except Exception:
            continue
        if not (math.isfinite(yf) and math.isfinite(gyf)):
            continue
        diff = _wrap_pi(yf - gyf)
        errs_deg.append(math.degrees(diff))
    if not errs_deg:
        return 0.0
    # RMSE
    mse = sum(e * e for e in errs_deg) / len(errs_deg)
    return float(math.sqrt(mse))


def _compute_latency_ms(logs: list[FrameLog]) -> float:
    vals: list[float] = []
    # latency sources per spec: wall_ms, pipeline result_source_age_ms, etc.
    # Order matters: prefer explicit latency_ms / wall_ms first, then pipeline age fields
    latency_keys = (
        "latency_ms",
        "wall_ms",
        "core_wall_ms",
        "latency",
        "result_source_age_ms",
        "pipeline_result_source_age_ms",
        "source_age_ms",
        "result_age_ms",
        "pipeline_latency_ms",
        "age_ms",
    )
    for l in logs:
        found = None
        for key in latency_keys:
            v = _get(l, key, None)
            if v is not None:
                try:
                    vf = float(v)
                    if math.isfinite(vf):
                        found = vf
                        break
                except Exception:
                    continue
        if found is not None:
            vals.append(found)
    if vals:
        return float(sum(vals) / len(vals))
    # fallback: try to read localization.jsonl if present
    # search a few candidate locations
    candidates = [
        Path("localization.jsonl"),
        Path.cwd() / "localization.jsonl",
        Path(__file__).parent / "localization.jsonl",
    ]
    # also check env var or common outputs
    for cand in candidates:
        try:
            if cand.exists() and cand.is_file():
                file_vals: list[float] = []
                with open(cand, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        for k in ("wall_ms", "core_wall_ms", "latency_ms", "latency", "result_source_age_ms", "source_age_ms"):
                            if k in row and row[k] is not None:
                                try:
                                    vf = float(row[k])
                                    if math.isfinite(vf):
                                        file_vals.append(vf)
                                        break
                                except Exception:
                                    continue
                if file_vals:
                    return float(sum(file_vals) / len(file_vals))
        except Exception:
            continue
    return 0.0


def _compute_recovery_time(logs: list[FrameLog]) -> float | None:
    if not logs:
        return None
    # sort by timestamp; spec: recovery_time via first LOST->TRACK timestamp delta
    try:
        ordered = sorted(logs, key=lambda x: float(_get(x, "timestamp", 0.0)))
    except Exception:
        ordered = logs
    lost_start: float | None = None
    recoveries: list[float] = []
    for l in ordered:
        ts = float(_get(l, "timestamp", 0.0))
        st = _get_state_str(l)
        is_lost = st == "LOST"
        # WEAK_TRACK is not LOST; BOOT is not LOST for recovery purposes
        if is_lost:
            if lost_start is None:
                lost_start = ts
        else:
            if lost_start is not None:
                # transition LOST -> non-LOST (consider TRACK/ WEAK_TRACK as recovery)
                is_track = st in ("TRACK", "WEAK_TRACK", "TRACKING")
                has_explicit_state = _get(l, "state", None) is not None or _get(l, "tracker_state", None) is not None
                if has_explicit_state:
                    if is_track:
                        recoveries.append(ts - lost_start)
                        lost_start = None
                    else:
                        if bool(_get(l, "edm_success", False)):
                            recoveries.append(ts - lost_start)
                            lost_start = None
                else:
                    if bool(_get(l, "edm_success", False)):
                        recoveries.append(ts - lost_start)
                        lost_start = None
    if not recoveries:
        return None
    # spec: recovery_time via first LOST->TRACK timestamp delta (keep fallback 0 when data missing via caller)
    # Use first recovery delta; fallback to median only if multiple? Per spec first.
    # For determinism, sort recoveries by occurrence order already; first is earliest LOST->TRACK
    # But ordered ensures chronological, so recoveries[0] is first delta
    return float(recoveries[0])



def _compute_failure_duration_vs_drift(
    logs: list[FrameLog], failure_bins: list[float] | None
) -> dict[float, float]:
    if not logs:
        return {}
    try:
        ordered = sorted(logs, key=lambda x: float(_get(x, "timestamp", 0.0)))
    except Exception:
        ordered = logs

    # Find contiguous LOST segments
    segments: list[list] = []
    cur: list = []
    for l in ordered:
        # determine if this frame is LOST
        st = _get_state_str(l)
        has_state = _get(l, "state", None) is not None or _get(l, "tracker_state", None) is not None
        if has_state:
            is_lost = st == "LOST"
        else:
            is_lost = not bool(_get(l, "edm_success", False))
        if is_lost:
            cur.append(l)
        else:
            if cur:
                segments.append(cur)
                cur = []
    if cur:
        segments.append(cur)

    # For each segment compute duration and drift
    seg_infos: list[tuple[float, float]] = []  # (duration, drift)
    for seg in segments:
        if not seg:
            continue
        try:
            start = float(_get(seg[0], "timestamp", 0.0))
            end = float(_get(seg[-1], "timestamp", 0.0))
        except Exception:
            continue
        dur = float(end - start)
        # if single-frame segment, duration is 0; estimate using neighboring dt if possible?
        # If dur == 0 and len(seg)==1, keep 0 - tests may expect 0 or small.
        # Compute drift = mean norm(pos - gt) during LOST
        drifts: list[float] = []
        for ll in seg:
            pos = _get(ll, "position", None)
            gt = _get(ll, "gt_position", None)
            if pos is None or gt is None:
                continue
            try:
                p = np.asarray(pos, dtype=float)
                g = np.asarray(gt, dtype=float)
                if p.shape != g.shape:
                    # try to flatten
                    p = p.reshape(-1)
                    g = g.reshape(-1)
                    if p.shape != g.shape:
                        continue
                if p.size == 0:
                    continue
                if not (np.all(np.isfinite(p)) and np.all(np.isfinite(g))):
                    continue
                d = float(np.linalg.norm(p - g))
                if math.isfinite(d):
                    drifts.append(d)
            except Exception:
                continue
        if drifts:
            avg_drift = float(sum(drifts) / len(drifts))
        else:
            # fallback: gt missing -> drift = norm(delta pos during LOST) (keep fallback 0 when data missing)
            # Collect valid positions during segment
            positions: list[np.ndarray] = []
            for ll in seg:
                pos = _get(ll, "position", None)
                if pos is None:
                    continue
                try:
                    p = np.asarray(pos, dtype=float).reshape(-1)
                    if p.size == 0 or not np.all(np.isfinite(p)):
                        continue
                    positions.append(p)
                except Exception:
                    continue
            if len(positions) >= 2:
                # ensure same shape; if shapes differ, truncate to min size
                try:
                    # compute straight-line displacement from first to last (delta pos)
                    # also compute path length as alternative; use max of displacement vs avg step?
                    # Use norm(delta pos) = norm(last - first); if shapes mismatch, use per-element.
                    first = positions[0]
                    last = positions[-1]
                    if first.shape != last.shape:
                        # align by flatten and truncate
                        n = min(first.size, last.size)
                        first = first.reshape(-1)[:n]
                        last = last.reshape(-1)[:n]
                    disp = float(np.linalg.norm(last - first))
                    # if disp is zero but path moved (e.g., loop), consider path length
                    if disp == 0.0 and len(positions) > 2:
                        # sum of sequential steps
                        steps = 0.0
                        for i in range(1, len(positions)):
                            a = positions[i-1]
                            b = positions[i]
                            if a.shape != b.shape:
                                n = min(a.size, b.size)
                                a = a.reshape(-1)[:n]
                                b = b.reshape(-1)[:n]
                            steps += float(np.linalg.norm(b - a))
                        # use steps if larger
                        if steps > disp:
                            disp = steps
                    if math.isfinite(disp):
                        avg_drift = float(disp)
                    else:
                        avg_drift = 0.0
                except Exception:
                    avg_drift = 0.0
            elif len(positions) == 1:
                # single point -> no delta, drift 0 (fallback 0 when data missing)
                avg_drift = 0.0
            else:
                avg_drift = 0.0
        seg_infos.append((dur, avg_drift))

    fmap: dict[float, float] = {}
    if failure_bins:
        # For each requested bin, find the segment whose duration is closest
        # If seg_infos empty, fallback to 0 for each bin? But spec says real calculation -> empty or 0
        for dur in failure_bins:
            try:
                bin_dur = float(dur)
            except Exception:
                continue
            if not seg_infos:
                # no LOST segments -> drift 0
                fmap[bin_dur] = 0.0
                continue
            # find closest segment
            best = min(seg_infos, key=lambda x: abs(x[0] - bin_dur))
            # Optionally average all segments that map to this bin (closest)
            # Collect all segments where this bin is the closest bin among failure_bins
            # Simpler: assign best drift
            # If multiple segments equally close to same bin, average their drifts
            # Find all segments for which this bin is the nearest bin
            candidates: list[float] = []
            for sd, drift in seg_infos:
                # find nearest bin for this segment
                nearest_bin = min(failure_bins, key=lambda b: abs(float(b) - sd))
                if abs(float(nearest_bin) - bin_dur) < 1e-9:
                    candidates.append(drift)
            if candidates:
                fmap[bin_dur] = float(sum(candidates) / len(candidates))
            else:
                fmap[bin_dur] = float(best[1])
    else:
        # No bins requested: return mapping of actual durations to drifts
        for dur, drift in seg_infos:
            # round dur to avoid floating noise for dict key
            key = float(dur)
            # if duplicate duration, average
            if key in fmap:
                fmap[key] = float((fmap[key] + drift) / 2.0)
            else:
                fmap[key] = float(drift)
    return fmap


def compute_metrics(logs: list[FrameLog], failure_bins: list[float] | None = None) -> BenchmarkResult:
    if not logs:
        return BenchmarkResult(0,0,0,0,0,0,None,0,0,0,{})
    n = len(logs)
    global_ok = sum(1 for l in logs if bool(_get(l, "edm_success", False))) / n
    # continuous: local or edm produced pose
    cont_ok = sum(1 for l in logs if _get(l, "position", None) is not None) / n
    # pos error where gt available
    errs = []
    for l in logs:
        pos = _get(l, "position", None)
        gt = _get(l, "gt_position", None)
        if pos is not None and gt is not None:
            try:
                p = np.asarray(pos, dtype=float)
                g = np.asarray(gt, dtype=float)
                if p.shape != g.shape:
                    p = np.asarray(p).reshape(-1)
                    g = np.asarray(g).reshape(-1)
                if p.size == 0 or p.shape != g.shape:
                    continue
                if not (np.all(np.isfinite(p)) and np.all(np.isfinite(g))):
                    continue
                errs.append(float(np.linalg.norm(p - g)))
            except Exception:
                continue
    rmse = float(math.sqrt(sum(e*e for e in errs)/len(errs))) if errs else 0.0
    max_d = float(max(errs)) if errs else 0.0
    avg_d = float(sum(errs)/len(errs)) if errs else 0.0
    # jump
    jumps = []
    for i in range(1, len(logs)):
        p1 = _get(logs[i], "position", None)
        p0 = _get(logs[i-1], "position", None)
        if p1 is not None and p0 is not None:
            try:
                a = np.asarray(p1, dtype=float)
                b = np.asarray(p0, dtype=float)
                if a.shape != b.shape:
                    a = a.reshape(-1); b = b.reshape(-1)
                if a.shape == b.shape and a.size>0 and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                    jumps.append(float(np.linalg.norm(a - b)))
            except Exception:
                continue
    max_jump = float(max(jumps)) if jumps else 0.0
    # fps from timestamps
    try:
        t0 = float(_get(logs[0], "timestamp", 0.0))
        t1 = float(_get(logs[-1], "timestamp", 0.0))
        dt = t1 - t0
    except Exception:
        dt = 0.0
    fps = n / dt if dt>0 else 0.0
    # orientation RMSE via yaw diff wrap pi -> degrees
    orientation_rmse = _compute_orientation_rmse(logs)
    # latency from wall_ms / localization.jsonl
    latency = _compute_latency_ms(logs)
    # recovery time via LOST->TRACK transitions
    recovery = _compute_recovery_time(logs)
    # failure duration vs drift via actual LOST segments
    fmap = _compute_failure_duration_vs_drift(logs, failure_bins)

    return BenchmarkResult(
        global_success_rate=global_ok,
        continuous_success_rate=cont_ok,
        position_rmse_m=rmse,
        orientation_rmse_deg=orientation_rmse,
        max_drift_m=max_d,
        avg_drift_m=avg_d,
        recovery_time_s=recovery,
        max_jump_m=max_jump,
        fps=fps,
        latency_ms=latency,
        failure_duration_vs_drift=fmap,
    )


def ablation_report(results: dict[str, BenchmarkResult]) -> str:
    lines = ["Ablation | global% | cont% | rmse(m) | max_drift | fps"]
    for name, r in results.items():
        lines.append(f"{name} | {r.global_success_rate:.2f} | {r.continuous_success_rate:.2f} | {r.position_rmse_m:.2f} | {r.max_drift_m:.2f} | {r.fps:.1f}")
    return "\n".join(lines)
