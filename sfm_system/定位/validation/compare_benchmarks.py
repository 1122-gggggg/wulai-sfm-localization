#!/usr/bin/env python3
"""Compare two benchmark_production_stream.py result JSONs and judge a candidate
against the optimization acceptance criteria.

Usage:
  python3 compare_benchmarks.py baseline.json candidate.json --kind speed
  python3 compare_benchmarks.py baseline.json candidate.json --kind accuracy

Criteria (tolerances configurable via CLI):

shared gates (both kinds; each FAILS when the data for it is missing):
  - baseline and candidate cover the same frame count
  - success rate does not decrease (beyond --success-tol)
  - LOST / WEAK_TRACK frames do not increase (beyond --lost-weak-tol)
  - median inliers do not drop by more than --inlier-drop-pct (default 5%)
  - reproj RMS does not worsen by more than --rms-worsen-pct (default 5%)

speed candidate additionally requires:
  - p50 or p90 wall latency improves by at least --latency-improve-pct

accuracy candidate additionally requires:
  - median-wall FPS does not drop by more than --fps-drop-pct (default 2%)
  - p50/p90 latency does not worsen by more than --latency-worsen-pct (default 2%)
  - at least one of success rate / median inliers / reproj RMS / quality score
    improves

Safety tests are NOT run here; run them separately and treat any failure as
an automatic reject regardless of this report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["summary"]


def g(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def fmt(x, nd=3):
    return "NA" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--kind", choices=["speed", "accuracy"], required=True)
    ap.add_argument("--success-tol", type=float, default=0.0,
                    help="allowed absolute success-rate drop (default 0)")
    ap.add_argument("--lost-weak-tol", type=int, default=0,
                    help="allowed increase in LOST+WEAK_TRACK frame count (default 0)")
    ap.add_argument("--inlier-drop-pct", type=float, default=5.0)
    ap.add_argument("--rms-worsen-pct", type=float, default=5.0)
    ap.add_argument("--fps-drop-pct", type=float, default=2.0)
    ap.add_argument("--latency-worsen-pct", type=float, default=2.0)
    ap.add_argument("--latency-improve-pct", type=float, default=1.0,
                    help="min p50-or-p90 improvement for a speed candidate")
    ap.add_argument("--json-out", default="", help="optionally save the verdict as JSON")
    args = ap.parse_args()

    b = load_summary(args.baseline)
    c = load_summary(args.candidate)

    def mode_count(s, key):
        # None (missing mode_counts) must FAIL the gate downstream, not pass as 0
        if not isinstance(s.get("mode_counts"), dict):
            return None
        return int(s["mode_counts"].get(key, 0) or 0)

    rows = []

    def add(name, bv, cv, nd=3):
        delta = None if (bv is None or cv is None) else cv - bv
        rows.append((name, bv, cv, delta, nd))
        return delta

    d_n = add("frames", g(b, "n"), g(c, "n"), 0)
    d_succ = add("success_rate", g(b, "success_rate"), g(c, "success_rate"), 4)
    add("fps_actual_incl_io", g(b, "actual_fps_including_io"), g(c, "actual_fps_including_io"), 2)
    d_fps = add("fps_from_median_wall", g(b, "fps_from_median_wall"), g(c, "fps_from_median_wall"), 2)
    d_p50 = add("wall_ms_p50", g(b, "wall_ms", "median"), g(c, "wall_ms", "median"), 2)
    d_p90 = add("wall_ms_p90", g(b, "wall_ms", "p90"), g(c, "wall_ms", "p90"), 2)
    add("wall_ms_p95", g(b, "wall_ms", "p95"), g(c, "wall_ms", "p95"), 2)
    d_inl = add("inliers_median", g(b, "inliers", "median"), g(c, "inliers", "median"), 1)
    add("inliers_p90", g(b, "inliers", "p90"), g(c, "inliers", "p90"), 1)
    d_rms = add("reproj_rms_median", g(b, "reproj_rms", "median"), g(c, "reproj_rms", "median"), 3)
    def lw(s):
        lost, weak = mode_count(s, "LOST"), mode_count(s, "WEAK_TRACK")
        return None if (lost is None or weak is None) else lost + weak
    b_lw, c_lw = lw(b), lw(c)
    d_lw = add("lost_plus_weak_frames", b_lw, c_lw, 0)
    add("lost_frames", mode_count(b, "LOST"), mode_count(c, "LOST"), 0)
    add("weak_track_frames", mode_count(b, "WEAK_TRACK"), mode_count(c, "WEAK_TRACK"), 0)
    add("nn_fast_accept", g(b, "nn_fast_accept"), g(c, "nn_fast_accept"), 0)
    add("lg_fallback", g(b, "lg_fallback"), g(c, "lg_fallback"), 0)
    add("temporal_cache_attempted", g(b, "temporal_cache_attempted_frames"),
        g(c, "temporal_cache_attempted_frames"), 0)
    add("temporal_cache_used", g(b, "temporal_cache_used_frames"),
        g(c, "temporal_cache_used_frames"), 0)
    add("pnp_failures", g(b, "pnp_failures"), g(c, "pnp_failures"), 0)
    add("jump_rejections", g(b, "jump_rejections"), g(c, "jump_rejections"), 0)
    d_q = add("quality_score_median", g(b, "quality_score", "median"), g(c, "quality_score", "median"), 4)
    add("corr3d_median", g(b, "corr3d", "median"), g(c, "corr3d", "median"), 1)
    add("xfeat_extracts_per_frame", g(b, "xfeat_extracts_per_frame"), g(c, "xfeat_extracts_per_frame"), 3)
    add("lg_calls_per_frame", g(b, "lg_calls_per_frame"), g(c, "lg_calls_per_frame"), 3)

    print(f"{'metric':28} | {'baseline':>12} | {'candidate':>12} | {'delta':>12}")
    print("-" * 74)
    for name, bv, cv, delta, nd in rows:
        print(f"{name:28} | {fmt(bv, nd):>12} | {fmt(cv, nd):>12} | {fmt(delta, nd):>12}")

    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail):
        checks.append((name, bool(ok), detail))

    # shared gates (both kinds)
    check("same_frame_count",
          d_n is not None and d_n == 0,
          f"baseline n={fmt(g(b,'n'),0)} candidate n={fmt(g(c,'n'),0)} "
          "(absolute LOST/WEAK counts are only comparable on identical streams)")
    check("success_rate_not_worse",
          d_succ is not None and d_succ >= -args.success_tol,
          f"delta={fmt(d_succ, 4)} tol={args.success_tol}")
    check("lost_weak_not_increased",
          d_lw is not None and d_lw <= args.lost_weak_tol,
          f"delta={fmt(d_lw, 0)} tol={args.lost_weak_tol}")
    b_inl, c_inl = g(b, "inliers", "median"), g(c, "inliers", "median")
    inl_ok = (b_inl is not None and c_inl is not None
              and c_inl >= b_inl * (1 - args.inlier_drop_pct / 100.0))
    b_rms, c_rms = g(b, "reproj_rms", "median"), g(c, "reproj_rms", "median")
    rms_ok = (b_rms is not None and c_rms is not None
              and c_rms <= b_rms * (1 + args.rms_worsen_pct / 100.0))
    check("inliers_within_drop_tol", inl_ok,
          f"median {fmt(b_inl,1)} -> {fmt(c_inl,1)} tol={args.inlier_drop_pct}%")
    check("rms_within_worsen_tol", rms_ok,
          f"median {fmt(b_rms)} -> {fmt(c_rms)} tol={args.rms_worsen_pct}%")

    if args.kind == "speed":
        b50, c50 = g(b, "wall_ms", "median"), g(c, "wall_ms", "median")
        b90, c90 = g(b, "wall_ms", "p90"), g(c, "wall_ms", "p90")
        p50_gain = None if (b50 is None or c50 is None) else 100.0 * (b50 - c50) / b50
        p90_gain = None if (b90 is None or c90 is None) else 100.0 * (b90 - c90) / b90
        improved = ((p50_gain is not None and p50_gain >= args.latency_improve_pct)
                    or (p90_gain is not None and p90_gain >= args.latency_improve_pct))
        check("latency_improved", improved,
              f"p50 gain={fmt(p50_gain,1)}% p90 gain={fmt(p90_gain,1)}% "
              f"need>={args.latency_improve_pct}%")
    else:  # accuracy
        b_fps, c_fps = g(b, "fps_from_median_wall"), g(c, "fps_from_median_wall")
        fps_ok = (b_fps is not None and c_fps is not None
                  and c_fps >= b_fps * (1 - args.fps_drop_pct / 100.0))
        check("fps_within_drop_tol", fps_ok,
              f"{fmt(b_fps,2)} -> {fmt(c_fps,2)} tol={args.fps_drop_pct}%")
        b50, c50 = g(b, "wall_ms", "median"), g(c, "wall_ms", "median")
        b90, c90 = g(b, "wall_ms", "p90"), g(c, "wall_ms", "p90")
        lat_ok = (b50 is not None and c50 is not None and b90 is not None and c90 is not None
                  and c50 <= b50 * (1 + args.latency_worsen_pct / 100.0)
                  and c90 <= b90 * (1 + args.latency_worsen_pct / 100.0))
        check("latency_within_worsen_tol", lat_ok,
              f"p50 {fmt(b50,2)}->{fmt(c50,2)} p90 {fmt(b90,2)}->{fmt(c90,2)} "
              f"tol={args.latency_worsen_pct}%")
        improves = []
        if d_succ is not None and d_succ > 0:
            improves.append("success_rate")
        if d_inl is not None and d_inl > 0:
            improves.append("inliers")
        if d_rms is not None and d_rms < 0:
            improves.append("reproj_rms")
        if d_q is not None and d_q > 0:
            improves.append("quality_score")
        check("something_improved", bool(improves),
              f"improved={improves or 'nothing'}")

    verdict = all(ok for _, ok, _ in checks)
    print(f"\n=== acceptance ({args.kind}) ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:28} {detail}")
    print(f"\nVERDICT: {'ACCEPT' if verdict else 'REJECT'} "
          f"(safety tests must also pass; not checked here)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "baseline": args.baseline,
            "candidate": args.candidate,
            "kind": args.kind,
            "tolerances": {k: v for k, v in vars(args).items()
                           if k not in ("baseline", "candidate", "json_out")},
            "deltas": {name: delta for name, _, _, delta, _ in rows},
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
            "verdict": "ACCEPT" if verdict else "REJECT",
        }, indent=2))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
