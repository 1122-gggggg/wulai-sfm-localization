#!/usr/bin/env python3
"""Paired randomized A/B gate over the 720p test-video corpus.

Runs ``benchmark_edm_site_replay.py`` twice per (video, trial) -- once with a
baseline environment and once with a candidate environment -- on the exact
same frame schedule, then decides per pair whether the candidate regressed.

Why paired and randomized: the ledger's repeated lesson is that a single
segment decides nothing (2026-09-05 reverted the async default after seven
segments contradicted two). Each trial varies the PnP RANSAC seed and the
frame schedule (stride / frame budget), and every video is judged on its own
row. One regressing row fails the whole gate; there is no averaging across
videos.

Accuracy is exact-or-better: ``successes`` and ``verified`` (frames whose pose
came from a fresh match on that frame, same definition as
``summarize_corpus_runs.is_verified``) may not drop on any row. Timing is
compared with an explicit tolerance because wall-clock percentiles are not
deterministic on a laptop GPU; the tolerance is reported with the verdict
instead of being hidden.

Example::

    定位演算法/validation/randomized_ab_gate.py \\
        --videos 模擬器/測試影片/720p \\
        --candidate-env SFM_EDM_TRAJECTORY_VETO=1 \\
        --trials 3 --out outputs/ab_traj_veto
"""
from __future__ import annotations

import argparse
import math
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


VALIDATION_ROOT = Path(__file__).resolve().parent
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from summarize_corpus_runs import is_verified  # noqa: E402
from stream_integrity import ffprobe_frame_count  # noqa: E402

REPLAY = VALIDATION_ROOT / "benchmark_edm_site_replay.py"

#: Frame schedules a trial may draw. Each is (stride, max_frames); 0 frames
#: means the whole video. Stride 3 is the corpus standard; 2 and 4 shift which
#: source frames the tracker ever sees, so a candidate cannot be tuned to one
#: sampling phase.
SCHEDULES: tuple[tuple[int, int], ...] = (
    (3, 700),
    (2, 700),
    (4, 700),
    (3, 0),
)


def parse_env_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    key, _, val = value.partition("=")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"empty env key in {value!r}")
    return key, val


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    rows = payload.get("rows") or []
    inliers = [int(r.get("inliers") or 0) for r in rows if r.get("success")]
    reproj = [
        float(r["reproj_rms"])
        for r in rows
        if r.get("success") and isinstance(r.get("reproj_rms"), (int, float))
    ]
    inliers.sort()
    reproj.sort()

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        idx = min(len(values) - 1, max(0, round(q * len(values)) - 1))
        return float(values[idx])

    return {
        "worker_mode": (payload.get("receipt") or {}).get("worker_mode", "sequential"),
        "frames": int(summary["frames"]),
        "successes": int(summary["successes"]),
        "verified": sum(1 for r in rows if is_verified(r)),
        "wall_p50_ms": summary["wall_ms"]["p50"],
        "wall_p95_ms": summary["wall_ms"]["p95"],
        "inliers_p05": pct(inliers, 0.05),
        "inliers_p50": pct(inliers, 0.50),
        "reproj_p50": pct(reproj, 0.50),
        "reproj_p95": pct(reproj, 0.95),
        "paired": {
            "src": [r.get("source_index") for r in rows],
            "succ": [bool(r.get("success")) for r in rows],
            "ver": [bool(is_verified(r)) for r in rows],
            "wall": [
                float(r["wall_ms"]) if isinstance(r.get("wall_ms"), (int, float)) else None
                for r in rows
            ],
            "veto": [
                r.get("trajectory_veto_reason") or r.get("gravity_veto_reason") or None
                for r in rows
            ],
            "veto_step": [
                r.get("trajectory_veto_step") for r in rows
            ],
            "veto_inl": [
                r.get("inliers") for r in rows
            ],
            "veto_reproj": [
                r.get("reproj_rms") for r in rows
            ],
        },
    }


def run_arm(
    *,
    video: Path,
    out_path: Path,
    stride: int,
    max_frames: int,
    pnp_seed: int,
    env_overrides: dict[str, str],
    extra_args: list[str],
    site_profile: Path | None,
    python: str,
) -> dict[str, Any]:
    cmd = [
        python,
        str(REPLAY),
        "--video",
        str(video),
        "--out",
        str(out_path),
        "--stride",
        str(stride),
        "--max-frames",
        str(max_frames),
        "--pnp-random-seed",
        str(pnp_seed),
        "--worker-mode",
        "sequential",
    ]
    if site_profile is not None:
        cmd.extend(["--site-profile", str(site_profile)])
    cmd.extend(extra_args)
    env = os.environ.copy()
    env.update(env_overrides)
    started = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        payload = None
        if proc.returncode == 3 and max_frames:
            # Production-path coalesce drops: every sampled frame is either
            # a row or a counted drop. Accept when rows + drops account for
            # the budget (within 2); compare() inner-joins on source_index
            # so drop-set differences cannot fake a verdict.
            try:
                candidate_payload = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                candidate_payload = None
            if candidate_payload is not None:
                rows = candidate_payload.get("rows", [])
                pipe = (candidate_payload.get("summary") or {}).get("pipeline") or {}
                accounted = (
                    len(rows)
                    + int(pipe.get("coalesce_drops") or 0)
                    + int(pipe.get("submit_drops") or 0)
                )
                if accounted >= max_frames - 2 and len(rows) >= 0.5 * max_frames:
                    payload = candidate_payload
        if payload is None:
            raise SystemExit(
                f"replay failed for {video.name} (rc={proc.returncode}):\n"
                + "\n".join(tail)
            )
        result = summarize(payload)
        result["run_seconds"] = round(elapsed, 2)
        result["shortfall_frames"] = max_frames - len(payload.get("rows", []))
        return result
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    result = summarize(payload)
    result["run_seconds"] = round(elapsed, 2)
    return result
def _pct_sorted(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))
    return float(ordered[idx])

def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    p50_tolerance: float,
    p95_tolerance: float,
) -> dict[str, Any]:
    failures: list[str] = []
    base_paired = baseline.get("paired") or {}
    cand_paired = candidate.get("paired") or {}
    base_src = base_paired.get("src", [])
    cand_src = cand_paired.get("src", [])
    production = (
        baseline.get("worker_mode", "sequential") != "sequential"
        or candidate.get("worker_mode", "sequential") != "sequential"
    )
    if (base_src == cand_src or not base_src or not cand_src) and not production:
        # Sequential determinism: identical sets and exact counts.
        paired_frames = len(base_src)
        base_succ, cand_succ = baseline["successes"], candidate["successes"]
        base_ver, cand_ver = baseline["verified"], candidate["verified"]
        base_p50, cand_p50 = float(baseline["wall_p50_ms"]), float(candidate["wall_p50_ms"])
        base_p95, cand_p95 = float(baseline["wall_p95_ms"]), float(candidate["wall_p95_ms"])
        paired_mode = "exact"
        veto_note = ""
    else:
        # Production-path coalesce drops differ per arm: inner-join on
        # source_index so the veto verdict compares identical frames.
        cand_by_src = {}
        for i, src in enumerate(cand_src):
            succ = bool((cand_paired.get("succ") or [None] * len(cand_src))[i])
            ver = bool((cand_paired.get("ver") or [None] * len(cand_src))[i])
            cand_by_src[src] = {
                "succ": succ, "ver": ver,
                "veto": (cand_paired.get("veto") or [None] * len(cand_src))[i],
                "step": (cand_paired.get("veto_step") or [None] * len(cand_src))[i],
                "inl": (cand_paired.get("veto_inl") or [None] * len(cand_src))[i],
                "reproj": (cand_paired.get("veto_reproj") or [None] * len(cand_src))[i],
            }
        common = [src for src in base_src if src in cand_by_src]
        floor = math.ceil(0.9 * min(len(base_src), len(cand_src)))
        if len(common) < floor:
            return {
                "pass": False,
                "failures": [
                    f"paired frames {len(common)} below floor {floor} "
                    f"(base {len(base_src)}, cand {len(cand_src)})"
                ],
                "d_successes": 0,
                "d_verified": 0,
                "d_wall_p50_pct": 0.0,
                "d_wall_p95_pct": 0.0,
                "identical_rows": False,
                "paired_frames": len(common),
            }
        paired_frames = len(common)
        base_rows = {
            src: (bool(succ), bool(ver))
            for src, succ, ver in zip(
                base_src, base_paired.get("succ", []), base_paired.get("ver", [])
            )
        }
        base_succ = sum(1 for src in common if base_rows[src][0])
        base_ver = sum(1 for src in common if base_rows[src][1])
        cand_succ = sum(1 for src in common if cand_by_src[src]["succ"])
        cand_ver = sum(1 for src in common if cand_by_src[src]["ver"])
        # Veto-misfire rule: a baseline success the candidate vetoed is only
        # a misfire when the vetoed pose looked healthy (small step, enough
        # inliers, sane reproj). Vetoing an actual flyaway is the gate's job.
        misfires = []
        for src in common:
            if base_rows[src][0] and not cand_by_src[src]["succ"] and cand_by_src[src]["veto"]:
                step = cand_by_src[src]["step"]
                inl = cand_by_src[src]["inl"]
                reproj = cand_by_src[src]["reproj"]
                healthy = (
                    isinstance(step, (int, float)) and step <= 2.0
                    and isinstance(inl, (int, float)) and inl >= 30
                    and isinstance(reproj, (int, float)) and reproj <= 6.0
                )
                if healthy:
                    misfires.append(src)
        butterfly = sum(
            1 for src in common
            if base_rows[src][0] != cand_by_src[src]["succ"]
            and not cand_by_src[src]["veto"]
        )
        # Timing over identical frames only: full-set percentiles inherit
        # whichever arm dropped its slow tail, which is not veto cost.
        base_wall_by_src = dict(zip(base_src, base_paired.get("wall", [])))
        cand_wall_by_src = {
            src: w for src, w in zip(cand_src, cand_paired.get("wall", []))
        }
        base_p50 = _pct_sorted(
            [w for src in common if (w := base_wall_by_src.get(src)) is not None], 0.50
        ) or 0.0
        base_p95 = _pct_sorted(
            [w for src in common if (w := base_wall_by_src.get(src)) is not None], 0.95
        ) or 0.0
        cand_p50 = _pct_sorted(
            [w for src in common if (w := cand_wall_by_src.get(src)) is not None], 0.50
        ) or 0.0
        cand_p95 = _pct_sorted(
            [w for src in common if (w := cand_wall_by_src.get(src)) is not None], 0.95
        ) or 0.0
        n_veto_fires = sum(1 for v in (cand_paired.get("veto") or []) if v)
        veto_note = f" veto_misfires={len(misfires)} butterfly={butterfly} veto_fires={n_veto_fires}"
        paired_mode = "intersect"
    if paired_mode == "exact":
        misfires = []
        butterfly = 0
        n_veto_fires = 0
        if cand_succ < base_succ:
            failures.append(f"successes {base_succ} -> {cand_succ}")
        if cand_ver < base_ver:
            failures.append(f"verified {base_ver} -> {cand_ver}")
    else:
        if misfires:
            failures.append(
                f"veto misfires on {len(misfires)} healthy frames: {misfires[:8]}"
            )
    if cand_p50 > base_p50 * (1.0 + p50_tolerance):
        failures.append(f"wall_p50 {base_p50:.2f} -> {cand_p50:.2f}ms")
    if cand_p95 > base_p95 * (1.0 + p95_tolerance):
        failures.append(f"wall_p95 {base_p95:.2f} -> {cand_p95:.2f}ms")
    return {
        "pass": not failures,
        "veto_misfires": list(misfires),
        "butterfly_flips": butterfly,
        "veto_fires": n_veto_fires,
        "failures": failures,
        "d_successes": cand_succ - base_succ,
        "d_verified": cand_ver - base_ver,
        "d_wall_p50_pct": 100.0 * (cand_p50 - base_p50) / base_p50 if base_p50 else None,
        "d_wall_p95_pct": 100.0 * (cand_p95 - base_p95) / base_p95 if base_p95 else None,
        "identical_rows": baseline == {**candidate, "run_seconds": baseline.get("run_seconds")},
        "paired_frames": paired_frames,
        "paired_mode": paired_mode,
        "veto_note": veto_note,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--videos", type=Path, required=True, help="directory of .MP4")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--candidate-env",
        action="append",
        type=parse_env_assignment,
        default=[],
        metavar="KEY=VALUE",
        help="env for the candidate arm; repeatable",
    )
    parser.add_argument(
        "--candidate-arg",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "extra replay CLI flag for the candidate arm; repeatable. Use this "
            "for settings the replay tool exposes as flags rather than env, "
            "e.g. --candidate-arg --no-query-cuda-graph."
        ),
    )
    parser.add_argument(
        "--baseline-arg",
        action="append",
        default=[],
        metavar="FLAG",
        help="extra replay CLI flag for the baseline arm; repeatable",
    )
    parser.add_argument(
        "--baseline-env",
        action="append",
        type=parse_env_assignment,
        default=[],
        metavar="KEY=VALUE",
        help="env for the baseline arm; repeatable (default: unset candidate keys)",
    )
    parser.add_argument("--trials", type=int, default=3, help="randomized trials/video")
    parser.add_argument("--seed", type=int, default=0, help="trial-draw seed")
    parser.add_argument("--site-profile", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--p50-tolerance", type=float, default=0.05)
    parser.add_argument("--p95-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--aa",
        action="store_true",
        help=(
            "null-band mode: run BOTH arms with the baseline environment. "
            "Any timing delta it reports is pure measurement noise, which is "
            "what a real tolerance has to be derived from. Accuracy deltas "
            "must be exactly zero here; a nonzero one means the replay is "
            "not deterministic and no A/B verdict from it can be trusted."
        ),
    )
    args = parser.parse_args(argv)

    videos = sorted(p for p in args.videos.glob("*.MP4"))
    if not videos:
        raise SystemExit(f"no .MP4 under {args.videos}")
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")

    if not args.candidate_env and not args.candidate_arg:
        raise SystemExit("need at least one --candidate-env or --candidate-arg")
    candidate_env = dict(args.candidate_env)
    baseline_env = dict(args.baseline_env)
    candidate_args = list(args.candidate_arg)
    baseline_args = list(args.baseline_arg)
    # A baseline that does not name a candidate key must actively unset it, so
    # an exported shell variable cannot silently make both arms identical.
    for key in candidate_env:
        baseline_env.setdefault(key, "")
    if args.aa:
        candidate_env = dict(baseline_env)
        candidate_args = list(baseline_args)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    trials = []
    for index in range(args.trials):
        stride, max_frames = SCHEDULES[index % len(SCHEDULES)]
        trials.append(
            {
                "trial": index,
                "stride": stride,
                "max_frames": max_frames,
                "pnp_seed": rng.randrange(0, 10_000),
            }
        )

    results: list[dict[str, Any]] = []
    for video in videos:
        # The replay tool fails closed when it decodes fewer sampled frames
        # than --max-frames asked for, so a short clip cannot silently run a
        # different schedule than the one recorded in the report. Clamp the
        # budget to what this video actually holds; a clamped run is still a
        # paired comparison because both arms get the same number.
        raw_frames, _source = ffprobe_frame_count(video)
        for trial in trials:
            stride = int(trial["stride"])
            requested = int(trial["max_frames"])
            available = None if raw_frames is None else max(1, raw_frames // stride)
            if requested and available is not None:
                max_frames = min(requested, available)
            else:
                max_frames = requested
            tag = f"{video.stem}_t{trial['trial']}"
            base = run_arm(
                video=video,
                out_path=args.out / f"{tag}_baseline.json",
                stride=stride,
                max_frames=max_frames,
                pnp_seed=trial["pnp_seed"],
                env_overrides=baseline_env,
                extra_args=baseline_args,
                site_profile=args.site_profile,
                python=args.python,
            )
            cand = run_arm(
                video=video,
                out_path=args.out / f"{tag}_candidate.json",
                stride=stride,
                max_frames=max_frames,
                pnp_seed=trial["pnp_seed"],
                env_overrides=candidate_env,
                extra_args=candidate_args,
                site_profile=args.site_profile,
                python=args.python,
            )
            verdict = compare(
                base,
                cand,
                p50_tolerance=args.p50_tolerance,
                p95_tolerance=args.p95_tolerance,
            )
            row = {
                "video": video.name,
                **trial,
                "effective_max_frames": max_frames,
                "source_raw_frames": raw_frames,
                "baseline": base,
                "candidate": cand,
                "verdict": verdict,
            }
            results.append(row)
            status = "PASS" if verdict["pass"] else "FAIL"
            print(
                f"[{status}] {video.stem} t{trial['trial']} "
                f"stride={stride} frames={max_frames or 'all'} "
                f"seed={trial['pnp_seed']} "
                f"succ {base['successes']}->{cand['successes']} "
                f"({verdict['d_successes']:+d}) "
                f"verified {base['verified']}->{cand['verified']} "
                f"({verdict['d_verified']:+d}) "
                f"p50 {verdict['d_wall_p50_pct']:+.1f}% "
                f"p95 {verdict['d_wall_p95_pct']:+.1f}%"
                f"{verdict.get('veto_note', '')}",
                flush=True,
            )
            if not verdict["pass"]:
                print("        " + "; ".join(verdict["failures"]), flush=True)

    failed = [r for r in results if not r["verdict"]["pass"]]
    accuracy_failed = [
        r
        for r in results
        if r["verdict"].get("veto_misfires")
        or (
            r["verdict"].get("paired_mode", "exact") == "exact"
            and (r["verdict"]["d_successes"] < 0 or r["verdict"]["d_verified"] < 0)
        )
    ]
    p50_deltas = sorted(
        abs(float(r["verdict"]["d_wall_p50_pct"]))
        for r in results
        if r["verdict"]["d_wall_p50_pct"] is not None
    )
    p95_deltas = sorted(
        abs(float(r["verdict"]["d_wall_p95_pct"]))
        for r in results
        if r["verdict"]["d_wall_p95_pct"] is not None
    )

    def band(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p50": None, "p95": None, "max": None}
        return {
            "p50": values[len(values) // 2],
            "p95": values[min(len(values) - 1, max(0, round(0.95 * len(values)) - 1))],
            "max": values[-1],
        }

    report = {
        "schema": "randomized-ab-gate/v1",
        "mode": "aa_null_band" if args.aa else "ab",
        "candidate_env": candidate_env,
        "baseline_env": baseline_env,
        "candidate_args": candidate_args,
        "baseline_args": baseline_args,
        "trial_seed": args.seed,
        "p50_tolerance": args.p50_tolerance,
        "p95_tolerance": args.p95_tolerance,
        "pairs": len(results),
        "failed_pairs": len(failed),
        "accuracy_failed_pairs": len(accuracy_failed),
        # Absolute |delta| distribution over every pair. In --aa mode this IS
        # the measurement noise floor; in A/B mode a candidate whose band sits
        # inside the A/A band has no detectable timing effect.
        "abs_d_wall_p50_pct": band(p50_deltas),
        "abs_d_wall_p95_pct": band(p95_deltas),
        "gate": "PASS" if not failed else "FAIL",
        "results": results,
    }
    report_path = args.out / "randomized_ab_gate.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"\nmode={report['mode']} gate={report['gate']} pairs={len(results)} "
        f"failed={len(failed)} accuracy_failed={len(accuracy_failed)}\n"
        f"|d p50| p50/p95/max = {report['abs_d_wall_p50_pct']}\n"
        f"|d p95| p50/p95/max = {report['abs_d_wall_p95_pct']}\n"
        f"report={report_path}",
        flush=True,
    )
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
