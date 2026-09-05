#!/usr/bin/env python3
"""Offline A/B of the ESEKF / KLT-3D-aware paths on a recorded live flight.

The pure replay gate (benchmark_edm_site_replay.py) cannot evaluate the ESEKF:
it has no live NED velocity, so ``ESEKF.prediction_allowed()`` never turns True
and every ESEKF branch stays dormant (docs/localization_optimization_runbook.md
Tier 3). This script closes that gap.

Inputs (from one manual flight with ``--live-localize`` and onboard recording):
  * the onboard ANAFI video (full-rate frames)
  * the session ``telemetry.jsonl`` — olympe_live_backend now logs a 10 Hz
    ``fused_odometry`` event (NED velocity + attitude); the 1 Hz ``readback``
    event is used as a fallback.

It replays the video through ``ProductionEDMTracker`` twice — once as shipped
(ESEKF on) and once with ``SFM_EDM_ESEKF_DISABLE=1`` — feeding the recorded
velocity into ``observe_fused_state`` before each frame, then writes
``esekf_on.json`` / ``esekf_off.json`` / ``SUMMARY.md``.

Metrics are RELATIVE only (no external ground truth): successes, state counts,
LOST-run recovery behaviour, wall p50/p95, and ESEKF activity (how often the
predictor armed / superseded, predicted-vs-next-measured centre gap). Frames are
the onboard recording, not the exact live-submitted frames, so both variants see
identical inputs but the run is not bit-comparable to the live session; that is
fine for an on/off differential. Admission still follows runbook section 5:
completed replay must not lower an approved recovery/quality gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from bisect import bisect_left
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VALIDATION_ROOT = Path(__file__).resolve().parent
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

import benchmark_edm_site_replay as base  # noqa: E402

DEFAULT_SITE_PROFILE = base.WORKSPACE_ROOT / "地圖檔" / "場域" / "river_site" / "site_profile.json"
FUSED_EVENT = "fused_odometry"
READBACK_EVENT = "readback"


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------
def load_velocity_track(source: Path) -> tuple[list[float], np.ndarray, str]:
    """Return (t_rel_s sorted, velocity_ned Nx3, event_kind) from a session.

    ``source`` may be a session directory (uses ``telemetry.jsonl``) or a JSONL
    file directly. Prefers 10 Hz ``fused_odometry`` events, falls back to 1 Hz
    ``readback``. ``t_rel_s`` is seconds since the first usable sample.
    """
    path = source / "telemetry.jsonl" if source.is_dir() else source
    if not path.is_file():
        raise SystemExit(f"no telemetry jsonl at {path}")

    fused: list[tuple[int, float, float, float]] = []
    readback: list[tuple[int, float, float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            event = rec.get("event")
            if event not in (FUSED_EVENT, READBACK_EVENT):
                continue
            vn = rec.get("speed_north_mps")
            ve = rec.get("speed_east_mps")
            vd = rec.get("speed_down_mps")
            if vn is None or ve is None or vd is None:
                continue
            try:
                vn, ve, vd = float(vn), float(ve), float(vd)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (vn, ve, vd)):
                continue
            stamp = rec.get("t_mono_ns")
            if stamp is None:
                continue
            bucket = fused if event == FUSED_EVENT else readback
            bucket.append((int(stamp), vn, ve, vd))

    rows = fused or readback
    kind = FUSED_EVENT if fused else (READBACK_EVENT if readback else "none")
    if not rows:
        raise SystemExit(
            f"{path} has no {FUSED_EVENT}/{READBACK_EVENT} events with NED velocity; "
            "was the session flown live (not simulated-stream)?"
        )
    rows.sort(key=lambda item: item[0])
    t0 = rows[0][0]
    t_rel = [(stamp - t0) * 1e-9 for stamp, *_ in rows]
    vel = np.asarray([[vn, ve, vd] for _, vn, ve, vd in rows], dtype=float)
    return t_rel, vel, kind


def velocity_at(
    t_rel: list[float], vel: np.ndarray, query: float, max_gap_s: float
) -> np.ndarray | None:
    """Linearly interpolate NED velocity at ``query`` s; None if nearest sample
    is farther than ``max_gap_s`` (mirrors a real telemetry dropout)."""
    if not t_rel:
        return None
    idx = bisect_left(t_rel, query)
    if idx == 0:
        return vel[0].copy() if abs(t_rel[0] - query) <= max_gap_s else None
    if idx >= len(t_rel):
        return vel[-1].copy() if abs(t_rel[-1] - query) <= max_gap_s else None
    lo, hi = t_rel[idx - 1], t_rel[idx]
    if min(query - lo, hi - query) > max_gap_s:
        return None
    span = hi - lo
    if span <= 0.0:
        return vel[idx].copy()
    w = (query - lo) / span
    return (1.0 - w) * vel[idx - 1] + w * vel[idx]


# --------------------------------------------------------------------------
# one variant
# --------------------------------------------------------------------------
def _replay_args(a: argparse.Namespace) -> SimpleNamespace:
    """Minimal args namespace for the benchmark_edm_site_replay helpers."""
    return SimpleNamespace(
        site_profile=str(a.site_profile),
        video=str(a.video),
        max_frames=int(a.max_frames),
        stride=int(a.stride),
        camera_params=None,
        max_corr_total=0,
        pnp_random_seed=int(a.pnp_random_seed),
        require_cuda=bool(a.require_cuda),
        quality_baseline=None,
        megaloc_backend=a.megaloc_backend,
        megaloc_engine=None,
        megaloc_engine_sha256=None,
        megaloc_token_reduction="none",  # noqa: S106 - ViT token pruning, not a credential
        megaloc_token_keep_ratio=1.0,
        megaloc_token_layer=6,
        sigma_mode="fused",
        runtime_sigma_mode=None,
        temporal_feature_cache_size=None,
        reference_feature_cache_size=None,
        host_reference_feature_cache_size=None,
        inclusive_host_feature_cache=None,
        temporal_feature_promotion=None,
        query_feature_reuse=None,
        query_cuda_graph=None,
        track_map_first=None,
        pnp_ranked_batches=None,
        pnp_pipeline=None,
        acquire_stage_mode=None,
        lost_prior_strategy=None,
        lost_prior_fusion_weight=None,
        use_temporal_reference=None,
        reference_quality_weight=None,
        lost_strategy=None,
        lost_global_retrieval_interval=None,
        radius=None,
        local_topk=None,
        mconf_thr=None,
        coarse_topk=None,
        cache_capacity=None,
    )


def run_variant(a: argparse.Namespace, esekf_on: bool) -> dict:
    os.environ["SFM_EDM_ESEKF_DISABLE"] = "0" if esekf_on else "1"
    t_rel, vel, kind = load_velocity_track(Path(a.session))

    args = _replay_args(a)
    site_path = Path(a.site_profile).expanduser().resolve()
    video_path = Path(a.video).expanduser().resolve()
    site, camera_tuple, site_sha, video_sha, camera, _ = base._load_replay_profile(
        args, site_path, video_path
    )
    built, startup_ms = base._build_replay_runtime(args, site, camera_tuple)
    trk = getattr(built.tracker, "trk", built.tracker)
    cap, source_fps, _all, audit = base._open_replay_stream(args, video_path)

    rows: list[dict] = []
    prev_center: np.ndarray | None = None
    fed = 0
    try:
        for frame in base.iter_decoded_replay_frames(args, cap, source_fps, site, audit):
            stamp = float(frame["recorded_stamp"])
            query = stamp * float(a.telemetry_scale) + float(a.telemetry_offset_s)
            v = velocity_at(t_rel, vel, query, float(a.max_gap_s))
            if v is not None and esekf_on:
                fed += 1
                built.tracker.observe_fused_state(
                    {
                        "velocity_ned": [float(v[0]), float(v[1]), float(v[2])],
                        "velocity": [float(v[0]), float(v[1]), float(v[2])],
                        "stamp": stamp,
                    }
                )
            started = time.perf_counter()
            pose = built.tracker.localize_frame(frame["rgb"], capture_stamp=stamp)
            wall_ms = (time.perf_counter() - started) * 1000.0
            info = dict(getattr(built.tracker, "last_info", {}) or {})
            center = None if pose is None else np.asarray([pose.x, pose.y, pose.z], float)
            esekf = getattr(trk, "esekf", None)
            pred_allowed = bool(
                esekf is not None
                and callable(getattr(esekf, "prediction_allowed", None))
                and esekf.prediction_allowed()
            )
            step = None
            if center is not None and prev_center is not None:
                step = float(np.linalg.norm(center - prev_center))
            rows.append(
                {
                    "source_index": frame["source_index"],
                    "capture_stamp": stamp,
                    "telemetry_query_s": query,
                    "fed_velocity": bool(v is not None),
                    "success": pose is not None,
                    "center": None if center is None else [float(c) for c in center],
                    "mode": info.get("mode"),
                    "next_mode": info.get("next_mode"),
                    "inliers": int(info.get("inliers", 0) or 0),
                    "reproj_rms": info.get("reproj_rms"),
                    "n_corr": info.get("n_corr"),
                    "step": step,
                    "wall_ms": wall_ms,
                    "match_ms": info.get("match_ms"),
                    "pnp_ms": info.get("pnp_ms"),
                    "prediction_allowed": pred_allowed,
                    "prediction_mode": info.get("prediction_mode"),
                    "prediction_source": info.get("prediction_source"),
                    "esekf_pos_trace": info.get("esekf_pos_trace"),
                    "esekf_d2": info.get("esekf_d2"),
                    "esekf_update_accepted": info.get("esekf_update_accepted"),
                }
            )
            if center is not None:
                prev_center = center
    finally:
        cap.release()

    return {
        "schema": "esekf-live-replay/v1",
        "esekf_on": esekf_on,
        "esekf_disabled_by_env": bool(getattr(trk, "esekf_disabled_by_env", not esekf_on)),
        "site_profile": str(site_path),
        "site_profile_sha256": site_sha,
        "video": str(video_path),
        "video_sha256": video_sha,
        "session": str(Path(a.session).resolve()),
        "telemetry_event_kind": kind,
        "telemetry_samples": len(t_rel),
        "telemetry_span_s": (t_rel[-1] - t_rel[0]) if t_rel else 0.0,
        "telemetry_offset_s": float(a.telemetry_offset_s),
        "telemetry_scale": float(a.telemetry_scale),
        "max_gap_s": float(a.max_gap_s),
        "frames_fed_velocity": fed,
        "source_fps": source_fps,
        "stride": int(a.stride),
        "max_frames": int(a.max_frames),
        "pnp_random_seed": int(a.pnp_random_seed),
        "startup_ms": startup_ms,
        "device": built.device,
        "rows": rows,
        "summary": summarize(rows),
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _pct(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return None
    pos = q / 100.0 * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _recovery_stats(rows: list[dict]) -> dict:
    """LOST-run count, longest run, and mean frames from LOST onset to next
    TRACK. A run is consecutive frames with next_mode == 'LOST'."""
    runs: list[int] = []
    current = 0
    recovered: list[int] = []
    for row in rows:
        if row.get("next_mode") == "LOST":
            current += 1
        else:
            if current:
                runs.append(current)
                if row.get("next_mode") == "TRACK":
                    recovered.append(current)
            current = 0
    if current:
        runs.append(current)
    return {
        "lost_runs": len(runs),
        "longest_lost_run": max(runs) if runs else 0,
        "lost_frames_total": sum(runs),
        "recoveries_to_track": len(recovered),
        "mean_frames_to_recover": (sum(recovered) / len(recovered)) if recovered else None,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    successes = sum(1 for r in rows if r["success"])
    state = {}
    for row in rows:
        key = row.get("next_mode") or "NONE"
        state[key] = state.get(key, 0) + 1
    walls = [r["wall_ms"] for r in rows]
    pred_frames = [r for r in rows if r["prediction_allowed"]]
    superseded = [r for r in rows if r.get("prediction_source") == "esekf"]
    return {
        "frames": n,
        "successes": successes,
        "success_rate": (successes / n) if n else 0.0,
        "state_counts": state,
        "wall_ms": {"p50": _pct(walls, 50), "p95": _pct(walls, 95)},
        "recovery": _recovery_stats(rows),
        "esekf_prediction_allowed_frames": len(pred_frames),
        "esekf_superseded_frames": len(superseded),
        "esekf_update_accepted_frames": sum(
            1 for r in rows if r.get("esekf_update_accepted") is True
        ),
    }


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def compare_markdown(on: dict, off: dict) -> str:
    s_on, s_off = on["summary"], off["summary"]
    r_on, r_off = s_on["recovery"], s_off["recovery"]
    lines = [
        "# ESEKF live-replay A/B",
        "",
        f"- site profile: `{on['site_profile']}` ({on['site_profile_sha256'][:12]})",
        f"- video: `{on['video']}` ({on['video_sha256'][:12]})",
        f"- session: `{on['session']}`",
        f"- telemetry: {on['telemetry_samples']} `{on['telemetry_event_kind']}` samples "
        f"over {on['telemetry_span_s']:.1f}s; velocity fed on "
        f"{on['frames_fed_velocity']}/{s_on['frames']} frames "
        f"(offset {on['telemetry_offset_s']}s, scale {on['telemetry_scale']}, "
        f"max_gap {on['max_gap_s']}s)",
        f"- stride {on['stride']}, seed {on['pnp_random_seed']}, "
        f"{s_on['frames']} frames, device {on['device']}",
        "",
        "| metric | ESEKF on | ESEKF off | delta (on-off) |",
        "|---|--:|--:|--:|",
    ]

    def row(name: str, a: object, b: object, delta: object = None) -> str:
        if delta is None and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = a - b
        return f"| {name} | {_fmt(a)} | {_fmt(b)} | {_fmt(delta)} |"

    lines += [
        row("successes", s_on["successes"], s_off["successes"]),
        row("success rate", s_on["success_rate"], s_off["success_rate"]),
        row("TRACK", s_on["state_counts"].get("TRACK", 0), s_off["state_counts"].get("TRACK", 0)),
        row(
            "WEAK_TRACK",
            s_on["state_counts"].get("WEAK_TRACK", 0),
            s_off["state_counts"].get("WEAK_TRACK", 0),
        ),
        row("LOST", s_on["state_counts"].get("LOST", 0), s_off["state_counts"].get("LOST", 0)),
        row("lost runs", r_on["lost_runs"], r_off["lost_runs"]),
        row("longest lost run", r_on["longest_lost_run"], r_off["longest_lost_run"]),
        row("recoveries to TRACK", r_on["recoveries_to_track"], r_off["recoveries_to_track"]),
        row(
            "mean frames to recover",
            r_on["mean_frames_to_recover"],
            r_off["mean_frames_to_recover"],
        ),
        row("wall p50 ms", s_on["wall_ms"]["p50"], s_off["wall_ms"]["p50"]),
        row("wall p95 ms", s_on["wall_ms"]["p95"], s_off["wall_ms"]["p95"]),
        row(
            "ESEKF prediction_allowed frames",
            s_on["esekf_prediction_allowed_frames"],
            s_off["esekf_prediction_allowed_frames"],
        ),
        row(
            "ESEKF superseded frames",
            s_on["esekf_superseded_frames"],
            s_off["esekf_superseded_frames"],
        ),
        row(
            "ESEKF visual updates accepted",
            s_on["esekf_update_accepted_frames"],
            s_off["esekf_update_accepted_frames"],
        ),
        "",
    ]

    verdict = "INCONCLUSIVE"
    note = ""
    if on["frames_fed_velocity"] == 0:
        verdict = "INVALID"
        note = (
            "No velocity was fed (telemetry empty or unaligned). Check the session "
            "has fused_odometry events and adjust --telemetry-offset-s."
        )
    elif s_on["esekf_prediction_allowed_frames"] == 0:
        verdict = "DORMANT"
        note = (
            "ESEKF never armed (prediction_allowed stayed False all run) — its "
            "covariance/yaw-sigma/age gate was not met. No differential to judge."
        )
    else:
        d_succ = s_on["successes"] - s_off["successes"]
        d_lost = s_on["state_counts"].get("LOST", 0) - s_off["state_counts"].get("LOST", 0)
        d_p95 = (s_on["wall_ms"]["p95"] or 0) - (s_off["wall_ms"]["p95"] or 0)
        if d_succ > 0 and d_lost <= 0 and d_p95 <= 5.0:
            verdict = "ESEKF HELPS (candidate — needs full runbook section 5 gate)"
        elif d_succ < 0 or d_lost > 0:
            verdict = "ESEKF REGRESSES — do not ship"
        else:
            verdict = "NEUTRAL"
        note = (
            f"successes {d_succ:+d}, LOST {d_lost:+d}, wall p95 {d_p95:+.1f} ms. "
            "Relative metrics only; not an approval."
        )
    lines += [f"## Verdict: {verdict}", "", note, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True, help="flight_logs session dir or telemetry.jsonl")
    p.add_argument("--video", required=True, help="onboard ANAFI recording from that flight")
    p.add_argument(
        "--out", required=True, help="output directory for esekf_on/off.json + SUMMARY.md"
    )
    p.add_argument("--site-profile", default=str(DEFAULT_SITE_PROFILE))
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--pnp-random-seed", type=int, default=0)
    p.add_argument("--require-cuda", action="store_true", default=True)
    p.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    p.add_argument(
        "--megaloc-backend", default="tensorrt", choices=("pytorch", "pytorch_fp16", "tensorrt")
    )
    p.add_argument(
        "--telemetry-offset-s",
        type=float,
        default=0.0,
        help="add to each frame stamp before looking up velocity (video start vs telemetry start)",
    )
    p.add_argument(
        "--telemetry-scale",
        type=float,
        default=1.0,
        help="multiply frame stamp before lookup (only if clocks drift)",
    )
    p.add_argument(
        "--max-gap-s",
        type=float,
        default=2.0,
        help="feed nothing if nearest telemetry sample is farther than this",
    )
    p.add_argument(
        "--variant",
        choices=("on", "off"),
        default=None,
        help="internal: run a single variant (default runs both as subprocesses)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    out_dir = Path(a.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.variant is not None:
        result = run_variant(a, esekf_on=(a.variant == "on"))
        target = out_dir / f"esekf_{a.variant}.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}")
        return 0

    # orchestrate: one clean subprocess per variant (no double model load)
    for variant in ("on", "off"):
        cmd = [sys.executable, __file__, *(sys.argv[1:]), "--variant", variant]
        print(f"[esekf-replay] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"variant {variant} failed (exit {proc.returncode})", file=sys.stderr)
            return proc.returncode

    on = json.loads((out_dir / "esekf_on.json").read_text(encoding="utf-8"))
    off = json.loads((out_dir / "esekf_off.json").read_text(encoding="utf-8"))
    summary = compare_markdown(on, off)
    (out_dir / "SUMMARY.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
