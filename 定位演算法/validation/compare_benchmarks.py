#!/usr/bin/env python3
"""Compare two benchmark_production_stream.py result JSONs and judge a candidate
against the optimization acceptance criteria.

Usage:
  python3 compare_benchmarks.py baseline.json candidate.json --kind speed
  python3 compare_benchmarks.py baseline.json candidate.json --kind accuracy
  python3 compare_benchmarks.py baseline.json candidate.json --kind speed \
      --require-frame-equivalence

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

import argparse
import json
import random
from pathlib import Path



EXACT_FRAME_FIELDS = (
    "idx",
    "frame",
    "success",
    "mode",
    "next_mode",
    "composite_stage",
    "accepted",
    "weak",
    "inliers",
    "corr3d",
    "pnp_failed",
    "jump_rejected",
    "candidates",
    "used_refs",
)

RECEIPT_OVERRIDE_KEYS = (
    "radius",
    "mconf_thr",
    "coarse_topk",
    "cache_capacity",
    "lost_strategy",
    "lost_global_retrieval_interval",
    "fused_coarse_mode",
    "runtime_sigma_mode",
    "temporal_feature_cache_size",
    "acquire_stage_mode",
    "lost_prior_strategy",
    "lost_prior_fusion_weight",
    "worker_mode",
)


def result_receipt(result: dict) -> dict | None:
    if not isinstance(result, dict):
        return None
    receipt = result.get("receipt")
    if isinstance(receipt, dict):
        return receipt
    identity = result.get("input_identity")
    if isinstance(identity, dict) and isinstance(identity.get("receipt"), dict):
        return identity["receipt"]
    return None


def _receipt_values_equal(expected, actual) -> bool:
    if expected == actual:
        return True
    try:
        return float(expected) == float(actual)
    except (TypeError, ValueError):
        return False


def receipt_identity_failures(baseline: dict, actual: dict | None) -> list[str]:
    """Missing receipt is an identity mismatch, not a SHA-only pass."""
    expected = None
    if isinstance(baseline, dict):
        expected = baseline.get("receipt")
        if expected is None:
            expected = result_receipt(baseline)
    if expected is None:
        return ["baseline is missing receipt identity"]
    if not isinstance(expected, dict):
        return ["baseline receipt identity is not an object"]
    if not isinstance(actual, dict):
        return ["candidate is missing receipt identity"]
    failures: list[str] = []
    for key in RECEIPT_OVERRIDE_KEYS:
        if key not in expected:
            failures.append(f"baseline is missing receipt.{key}")
            continue
        if key not in actual:
            failures.append(f"candidate is missing receipt.{key}")
            continue
        if not _receipt_values_equal(expected[key], actual[key]):
            failures.append(
                f"receipt.{key} mismatch: expected={expected[key]!r} "
                f"actual={actual[key]!r}"
            )
    return failures



def load_result(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare_exact_fields(b_row: dict, c_row: dict, frame: str, mismatch) -> None:
    for field in EXACT_FRAME_FIELDS:
        if b_row.get(field) != c_row.get(field):
            mismatch(f"{frame}: {field} {b_row.get(field)!r} != {c_row.get(field)!r}")


def _compare_reprojection(
    b_row: dict,
    c_row: dict,
    frame: str,
    reproj_atol: float,
    mismatch,
) -> float:
    b_reproj, c_reproj = b_row.get("reproj_rms"), c_row.get("reproj_rms")
    if (b_reproj is None) != (c_reproj is None):
        mismatch(f"{frame}: reproj_rms {b_reproj!r} != {c_reproj!r}")
        return 0.0
    if b_reproj is None:
        return 0.0

    delta = abs(float(c_reproj) - float(b_reproj))
    if delta > reproj_atol:
        mismatch(f"{frame}: reproj_rms delta {delta:.9g} > {reproj_atol:.9g}")
    return delta


def _compare_pose(
    b_row: dict,
    c_row: dict,
    frame: str,
    pose_atol: float,
    mismatch,
) -> float:
    b_pose, c_pose = b_row.get("pose"), c_row.get("pose")
    if (b_pose is None) != (c_pose is None):
        mismatch(f"{frame}: pose presence differs")
        return 0.0
    if b_pose is None:
        return 0.0

    max_delta = 0.0
    for field in ("x", "y", "z", "yaw"):
        delta = abs(float(c_pose[field]) - float(b_pose[field]))
        max_delta = max(max_delta, delta)
        if delta > pose_atol:
            mismatch(f"{frame}: pose.{field} delta {delta:.9g} > {pose_atol:.9g}")
    return max_delta


def compare_frame_rows(
    baseline: dict,
    candidate: dict,
    *,
    pose_atol: float,
    reproj_atol: float,
    max_examples: int = 8,
) -> dict:
    """Check accuracy-critical per-frame output, ignoring timing/diagnostic fields."""
    b_rows = baseline.get("rows")
    c_rows = candidate.get("rows")
    if not isinstance(b_rows, list) or not isinstance(c_rows, list):
        return {
            "ok": False,
            "mismatch_count": 1,
            "examples": ["baseline/candidate rows are missing or not lists"],
        }

    mismatch_count = 0
    examples: list[str] = []
    max_pose_delta = 0.0
    max_reproj_delta = 0.0

    def mismatch(message: str) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(examples) < max_examples:
            examples.append(message)

    if len(b_rows) != len(c_rows):
        mismatch(f"row count {len(b_rows)} != {len(c_rows)}")

    for row_index, (b_row, c_row) in enumerate(zip(b_rows, c_rows)):
        frame = b_row.get("frame", f"row {row_index}")
        _compare_exact_fields(b_row, c_row, frame, mismatch)
        max_reproj_delta = max(
            max_reproj_delta,
            _compare_reprojection(b_row, c_row, frame, reproj_atol, mismatch),
        )
        max_pose_delta = max(
            max_pose_delta,
            _compare_pose(b_row, c_row, frame, pose_atol, mismatch),
        )

    return {
        "ok": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "max_pose_abs_delta": max_pose_delta,
        "max_reproj_abs_delta": max_reproj_delta,
        "examples": examples,
    }


def frame_metric_series(result: dict, key: str) -> list[float]:
    rows = result.get("rows")
    if not isinstance(rows, list):
        return []
    values = []
    for row in rows:
        if not isinstance(row, dict) or key not in row or row[key] is None:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if value == value:  # not NaN
            values.append(value)
    return values


def contiguous_block_bootstrap(
    values,
    *,
    block_size: int,
    n_resamples: int = 1000,
    seed: int = 0,
    q_low: float = 2.5,
    q_high: float = 97.5,
) -> dict:
    """Resample contiguous blocks. Reports a CI; does not invent a pass gate."""
    if block_size <= 0 or n_resamples <= 0:
        raise ValueError("block_size and n_resamples must be positive")
    series = [float(v) for v in values if v is not None]
    finite = [v for v in series if v == v]
    n = len(finite)
    if n == 0:
        return {
            "mean": None,
            "bootstrap_mean": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
            "block_size": int(block_size),
            "n_resamples": int(n_resamples),
            "q_low": float(q_low),
            "q_high": float(q_high),
        }
    rng = random.Random(int(seed))
    n_blocks = max(1, (n + block_size - 1) // block_size)
    means = []
    for _ in range(int(n_resamples)):
        pieces = []
        for _block in range(n_blocks):
            start = rng.randrange(n)
            end = min(start + block_size, n)
            pieces.extend(finite[start:end])
            need = block_size - (end - start)
            if need:
                pieces.extend(finite[:need])
        sample = pieces[:n]
        means.append(sum(sample) / n)
    means.sort()

    def _percentile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return float("nan")
        if len(sorted_vals) == 1:
            return float(sorted_vals[0])
        rank = (q / 100.0) * (len(sorted_vals) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = rank - lo
        return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)

    return {
        "mean": sum(finite) / n,
        "bootstrap_mean": sum(means) / len(means),
        "ci_low": _percentile(means, q_low),
        "ci_high": _percentile(means, q_high),
        "n": n,
        "block_size": int(block_size),
        "n_resamples": int(n_resamples),
        "q_low": float(q_low),
        "q_high": float(q_high),
    }


def paired_block_bootstrap(baseline, candidate, **kwargs) -> dict:
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired series length mismatch: {len(baseline)} != {len(candidate)}"
        )
    delta = [c - b for b, c in zip(baseline, candidate)]
    return contiguous_block_bootstrap(delta, **kwargs)


def compare_baab_runs(
    b1: dict,
    a1: dict,
    a2: dict,
    b2: dict,
    *,
    metrics: tuple[str, ...] = ("wall_ms", "match_ms", "pnp_ms"),
    block_size: int = 50,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """B-A-A-B paired comparison. Reports drift and effect; no pass threshold."""
    labeled = (("B1", b1), ("A1", a1), ("A2", a2), ("B2", b2))
    receipts = {name: result_receipt(result) for name, result in labeled}
    receipt_failures: list[str] = []
    for name, receipt in receipts.items():
        if receipt is None:
            receipt_failures.append(f"{name} is missing receipt identity")
    if not receipt_failures:
        reference = receipts["B1"]
        for name, receipt in receipts.items():
            receipt_failures.extend(
                f"{name}: {item}"
                for item in receipt_identity_failures({"receipt": reference}, receipt)
            )
    report = {
        "order": ["B", "A", "A", "B"],
        "metrics": {},
        "verdict": None,
        "receipts": receipts,
        "receipt_identity_failures": receipt_failures,
        "notes": (
            "B-A-A-B reports drift (B2-B1) and candidate effect "
            "((A1+A2)/2 - (B1+B2)/2). Contiguous-block bootstrap CIs "
            "are descriptive; this path does not invent a pass threshold. "
            "Receipt identity is reported and required for comparability; "
            "a missing receipt is an identity mismatch."
        ),
    }
    for metric in metrics:
        series = {name: frame_metric_series(result, metric) for name, result in labeled}
        lengths = {name: len(values) for name, values in series.items()}
        entry = {
            "n": lengths,
            "comparable": (
                not receipt_failures
                and len(set(lengths.values())) == 1
                and lengths["B1"] > 0
            ),
        }
        means = {
            name: (sum(values) / len(values) if values else None)
            for name, values in series.items()
        }
        entry["means"] = means
        if means["B1"] is not None and means["B2"] is not None:
            entry["drift"] = means["B2"] - means["B1"]
        else:
            entry["drift"] = None
        a_mean = None
        b_mean = None
        if means["A1"] is not None and means["A2"] is not None:
            a_mean = 0.5 * (means["A1"] + means["A2"])
        if means["B1"] is not None and means["B2"] is not None:
            b_mean = 0.5 * (means["B1"] + means["B2"])
        entry["effect"] = (
            None if a_mean is None or b_mean is None else a_mean - b_mean
        )
        if entry["comparable"]:
            entry["bootstrap"] = {
                "A1_minus_B1": paired_block_bootstrap(
                    series["B1"], series["A1"],
                    block_size=block_size, n_resamples=n_resamples, seed=seed,
                ),
                "A2_minus_B2": paired_block_bootstrap(
                    series["B2"], series["A2"],
                    block_size=block_size, n_resamples=n_resamples, seed=seed + 1,
                ),
                "B2_minus_B1": paired_block_bootstrap(
                    series["B1"], series["B2"],
                    block_size=block_size, n_resamples=n_resamples, seed=seed + 2,
                ),
            }
        else:
            entry["bootstrap"] = None
        report["metrics"][metric] = entry
    return report



def g(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def fmt(x, nd=3):
    return "NA" if x is None else f"{x:.{nd}f}"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("baseline", nargs="?", default=None)
    ap.add_argument("candidate", nargs="?", default=None)
    ap.add_argument(
        "--baab",
        nargs=4,
        metavar=("B1", "A1", "A2", "B2"),
        help="B-A-A-B JSON paths; reports drift/effect/bootstrap without a pass gate",
    )
    ap.add_argument("--kind", choices=["speed", "accuracy"])
    ap.add_argument(
        "--success-tol",
        type=float,
        default=0.0,
        help="allowed absolute success-rate drop (default 0)",
    )
    ap.add_argument(
        "--lost-weak-tol",
        type=int,
        default=0,
        help="allowed increase in LOST+WEAK_TRACK frame count (default 0)",
    )
    ap.add_argument("--inlier-drop-pct", type=float, default=5.0)
    ap.add_argument("--rms-worsen-pct", type=float, default=5.0)
    ap.add_argument("--fps-drop-pct", type=float, default=2.0)
    ap.add_argument("--latency-worsen-pct", type=float, default=2.0)
    ap.add_argument(
        "--latency-improve-pct",
        type=float,
        default=1.0,
        help="min p50-or-p90 improvement for a speed candidate",
    )
    ap.add_argument(
        "--require-frame-equivalence",
        action="store_true",
        help="require accuracy-critical output to match on every frame",
    )
    ap.add_argument(
        "--pose-atol", type=float, default=1e-5, help="absolute tolerance for each pose component"
    )
    ap.add_argument(
        "--reproj-atol",
        type=float,
        default=1e-5,
        help="absolute tolerance for per-frame reprojection RMS",
    )
    ap.add_argument("--json-out", default="", help="optionally save the verdict as JSON")
    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--bootstrap-resamples", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    return ap.parse_args()



def _mode_count(summary: dict, key: str):
    # None (missing mode_counts) must FAIL the gate downstream, not pass as 0
    if not isinstance(summary.get("mode_counts"), dict):
        return None
    return int(summary["mode_counts"].get(key, 0) or 0)


def _lost_weak_count(summary: dict):
    lost = _mode_count(summary, "LOST")
    weak = _mode_count(summary, "WEAK_TRACK")
    return None if (lost is None or weak is None) else lost + weak


def _add_metric(rows: list, name: str, baseline, candidate, nd: int = 3):
    delta = None if (baseline is None or candidate is None) else candidate - baseline
    rows.append((name, baseline, candidate, delta, nd))
    return delta


def _build_metric_rows(b: dict, c: dict) -> tuple[list, dict]:
    rows = []
    deltas = {}
    deltas["frames"] = _add_metric(rows, "frames", g(b, "n"), g(c, "n"), 0)
    deltas["success"] = _add_metric(
        rows,
        "success_rate",
        g(b, "success_rate"),
        g(c, "success_rate"),
        4,
    )
    _add_metric(
        rows,
        "fps_actual_incl_io",
        g(b, "actual_fps_including_io"),
        g(c, "actual_fps_including_io"),
        2,
    )
    _add_metric(
        rows, "fps_from_median_wall", g(b, "fps_from_median_wall"), g(c, "fps_from_median_wall"), 2
    )
    _add_metric(rows, "wall_ms_p50", g(b, "wall_ms", "median"), g(c, "wall_ms", "median"), 2)
    _add_metric(rows, "wall_ms_p90", g(b, "wall_ms", "p90"), g(c, "wall_ms", "p90"), 2)
    _add_metric(rows, "wall_ms_p95", g(b, "wall_ms", "p95"), g(c, "wall_ms", "p95"), 2)
    deltas["inliers"] = _add_metric(
        rows,
        "inliers_median",
        g(b, "inliers", "median"),
        g(c, "inliers", "median"),
        1,
    )
    _add_metric(rows, "inliers_p90", g(b, "inliers", "p90"), g(c, "inliers", "p90"), 1)
    deltas["rms"] = _add_metric(
        rows,
        "reproj_rms_median",
        g(b, "reproj_rms", "median"),
        g(c, "reproj_rms", "median"),
        3,
    )
    deltas["lost_weak"] = _add_metric(
        rows,
        "lost_plus_weak_frames",
        _lost_weak_count(b),
        _lost_weak_count(c),
        0,
    )
    _add_metric(rows, "lost_frames", _mode_count(b, "LOST"), _mode_count(c, "LOST"), 0)
    _add_metric(
        rows, "weak_track_frames", _mode_count(b, "WEAK_TRACK"), _mode_count(c, "WEAK_TRACK"), 0
    )
    _add_metric(rows, "nn_fast_accept", g(b, "nn_fast_accept"), g(c, "nn_fast_accept"), 0)
    _add_metric(rows, "lg_fallback", g(b, "lg_fallback"), g(c, "lg_fallback"), 0)
    _add_metric(
        rows,
        "temporal_cache_attempted",
        g(b, "temporal_cache_attempted_frames"),
        g(c, "temporal_cache_attempted_frames"),
        0,
    )
    _add_metric(
        rows,
        "temporal_cache_used",
        g(b, "temporal_cache_used_frames"),
        g(c, "temporal_cache_used_frames"),
        0,
    )
    _add_metric(rows, "pnp_failures", g(b, "pnp_failures"), g(c, "pnp_failures"), 0)
    _add_metric(rows, "jump_rejections", g(b, "jump_rejections"), g(c, "jump_rejections"), 0)
    deltas["quality"] = _add_metric(
        rows,
        "quality_score_median",
        g(b, "quality_score", "median"),
        g(c, "quality_score", "median"),
        4,
    )
    _add_metric(rows, "corr3d_median", g(b, "corr3d", "median"), g(c, "corr3d", "median"), 1)
    _add_metric(
        rows,
        "xfeat_extracts_per_frame",
        g(b, "xfeat_extracts_per_frame"),
        g(c, "xfeat_extracts_per_frame"),
        3,
    )
    _add_metric(
        rows, "lg_calls_per_frame", g(b, "lg_calls_per_frame"), g(c, "lg_calls_per_frame"), 3
    )
    return rows, deltas


def _print_metric_rows(rows: list) -> None:
    print(f"{'metric':28} | {'baseline':>12} | {'candidate':>12} | {'delta':>12}")
    print("-" * 74)
    for name, bv, cv, delta, nd in rows:
        print(f"{name:28} | {fmt(bv, nd):>12} | {fmt(cv, nd):>12} | {fmt(delta, nd):>12}")


def _add_check(checks: list, name: str, ok, detail: str) -> None:
    checks.append((name, bool(ok), detail))


def _shared_checks(b: dict, c: dict, args: argparse.Namespace, deltas: dict) -> list:
    checks = []
    _add_check(
        checks,
        "same_frame_count",
        deltas["frames"] is not None and deltas["frames"] == 0,
        f"baseline n={fmt(g(b, 'n'), 0)} candidate n={fmt(g(c, 'n'), 0)} "
        "(absolute LOST/WEAK counts are only comparable on identical streams)",
    )
    _add_check(
        checks,
        "success_rate_not_worse",
        deltas["success"] is not None and deltas["success"] >= -args.success_tol,
        f"delta={fmt(deltas['success'], 4)} tol={args.success_tol}",
    )
    _add_check(
        checks,
        "lost_weak_not_increased",
        deltas["lost_weak"] is not None and deltas["lost_weak"] <= args.lost_weak_tol,
        f"delta={fmt(deltas['lost_weak'], 0)} tol={args.lost_weak_tol}",
    )
    b_inl, c_inl = g(b, "inliers", "median"), g(c, "inliers", "median")
    inl_ok = (
        b_inl is not None
        and c_inl is not None
        and c_inl >= b_inl * (1 - args.inlier_drop_pct / 100.0)
    )
    b_rms, c_rms = g(b, "reproj_rms", "median"), g(c, "reproj_rms", "median")
    rms_ok = (
        b_rms is not None
        and c_rms is not None
        and c_rms <= b_rms * (1 + args.rms_worsen_pct / 100.0)
    )
    _add_check(
        checks,
        "inliers_within_drop_tol",
        inl_ok,
        f"median {fmt(b_inl, 1)} -> {fmt(c_inl, 1)} tol={args.inlier_drop_pct}%",
    )
    _add_check(
        checks,
        "rms_within_worsen_tol",
        rms_ok,
        f"median {fmt(b_rms)} -> {fmt(c_rms)} tol={args.rms_worsen_pct}%",
    )
    return checks


def _speed_checks(b: dict, c: dict, args: argparse.Namespace) -> list:
    b50, c50 = g(b, "wall_ms", "median"), g(c, "wall_ms", "median")
    b90, c90 = g(b, "wall_ms", "p90"), g(c, "wall_ms", "p90")
    p50_gain = None if (b50 is None or c50 is None) else 100.0 * (b50 - c50) / b50
    p90_gain = None if (b90 is None or c90 is None) else 100.0 * (b90 - c90) / b90
    improved = (p50_gain is not None and p50_gain >= args.latency_improve_pct) or (
        p90_gain is not None and p90_gain >= args.latency_improve_pct
    )
    checks = []
    _add_check(
        checks,
        "latency_improved",
        improved,
        f"p50 gain={fmt(p50_gain, 1)}% p90 gain={fmt(p90_gain, 1)}% "
        f"need>={args.latency_improve_pct}%",
    )
    return checks


def _accuracy_improvements(deltas: dict) -> list[str]:
    improves = []
    if deltas["success"] is not None and deltas["success"] > 0:
        improves.append("success_rate")
    if deltas["inliers"] is not None and deltas["inliers"] > 0:
        improves.append("inliers")
    if deltas["rms"] is not None and deltas["rms"] < 0:
        improves.append("reproj_rms")
    if deltas["quality"] is not None and deltas["quality"] > 0:
        improves.append("quality_score")
    return improves


def _accuracy_checks(b: dict, c: dict, args: argparse.Namespace, deltas: dict) -> list:
    checks = []
    b_fps, c_fps = g(b, "fps_from_median_wall"), g(c, "fps_from_median_wall")
    fps_ok = (
        b_fps is not None and c_fps is not None and c_fps >= b_fps * (1 - args.fps_drop_pct / 100.0)
    )
    _add_check(
        checks,
        "fps_within_drop_tol",
        fps_ok,
        f"{fmt(b_fps, 2)} -> {fmt(c_fps, 2)} tol={args.fps_drop_pct}%",
    )
    b50, c50 = g(b, "wall_ms", "median"), g(c, "wall_ms", "median")
    b90, c90 = g(b, "wall_ms", "p90"), g(c, "wall_ms", "p90")
    lat_ok = (
        b50 is not None
        and c50 is not None
        and b90 is not None
        and c90 is not None
        and c50 <= b50 * (1 + args.latency_worsen_pct / 100.0)
        and c90 <= b90 * (1 + args.latency_worsen_pct / 100.0)
    )
    _add_check(
        checks,
        "latency_within_worsen_tol",
        lat_ok,
        f"p50 {fmt(b50, 2)}->{fmt(c50, 2)} p90 {fmt(b90, 2)}->{fmt(c90, 2)} "
        f"tol={args.latency_worsen_pct}%",
    )
    improves = _accuracy_improvements(deltas)
    _add_check(checks, "something_improved", bool(improves), f"improved={improves or 'nothing'}")
    return checks


def _frame_equivalence_check(
    baseline_result: dict,
    candidate_result: dict,
    args: argparse.Namespace,
) -> tuple[dict | None, tuple | None]:
    if not args.require_frame_equivalence:
        return None, None

    report = compare_frame_rows(
        baseline_result,
        candidate_result,
        pose_atol=args.pose_atol,
        reproj_atol=args.reproj_atol,
    )
    check = (
        "per_frame_equivalence",
        bool(report["ok"]),
        f"mismatches={report['mismatch_count']} "
        f"max_pose_delta={report.get('max_pose_abs_delta', 0):.3g} "
        f"max_reproj_delta={report.get('max_reproj_abs_delta', 0):.3g}",
    )
    return report, check


def _print_acceptance(
    kind: str,
    checks: list,
    frame_equivalence: dict | None,
    verdict: bool,
) -> None:
    print(f"\n=== acceptance ({kind}) ===")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:28} {detail}")
    if frame_equivalence and frame_equivalence["examples"]:
        print("\nper-frame mismatch examples:")
        for example in frame_equivalence["examples"]:
            print(f"  - {example}")
    print(
        f"\nVERDICT: {'ACCEPT' if verdict else 'REJECT'} "
        f"(safety tests must also pass; not checked here)"
    )


def _write_json_verdict(
    args: argparse.Namespace,
    rows: list,
    checks: list,
    frame_equivalence: dict | None,
    verdict: bool,
) -> None:
    if not args.json_out:
        return
    Path(args.json_out).write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "candidate": args.candidate,
                "kind": args.kind,
                "tolerances": {
                    k: v
                    for k, v in vars(args).items()
                    if k not in ("baseline", "candidate", "json_out")
                },
                "deltas": {name: delta for name, _, _, delta, _ in rows},
                "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
                "frame_equivalence": frame_equivalence,
                "verdict": "ACCEPT" if verdict else "REJECT",
            },
            indent=2,
        )
    )


def main() -> int:
    args = _parse_args()
    if args.baab:
        if args.block_size <= 0 or args.bootstrap_resamples <= 0:
            raise SystemExit("block-size and bootstrap-resamples must be positive")
        b1, a1, a2, b2 = (load_result(path) for path in args.baab)
        report = compare_baab_runs(
            b1, a1, a2, b2,
            block_size=args.block_size,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        print(json.dumps(report, indent=2))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0
    if not args.baseline or not args.candidate or not args.kind:
        raise SystemExit("baseline candidate --kind are required unless --baab is used")
    baseline_result = load_result(args.baseline)
    candidate_result = load_result(args.candidate)
    b = baseline_result["summary"]
    c = candidate_result["summary"]

    rows, deltas = _build_metric_rows(b, c)
    _print_metric_rows(rows)

    checks = _shared_checks(b, c, args, deltas)
    receipt_failures = receipt_identity_failures(
        baseline_result, result_receipt(candidate_result),
    )
    _add_check(
        checks,
        "receipt_identity",
        not receipt_failures,
        "; ".join(receipt_failures) or "receipt identity matches",
    )
    if args.kind == "speed":
        checks.extend(_speed_checks(b, c, args))
    else:
        checks.extend(_accuracy_checks(b, c, args, deltas))

    frame_equivalence, frame_check = _frame_equivalence_check(
        baseline_result,
        candidate_result,
        args,
    )
    if frame_check is not None:
        checks.append(frame_check)

    verdict = all(ok for _, ok, _ in checks)
    _print_acceptance(args.kind, checks, frame_equivalence, verdict)
    _write_json_verdict(args, rows, checks, frame_equivalence, verdict)
    return 0 if verdict else 1



if __name__ == "__main__":
    raise SystemExit(main())
