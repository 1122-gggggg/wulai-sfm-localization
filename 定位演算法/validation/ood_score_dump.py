#!/usr/bin/env python3
"""OOD threshold sweep for the R2 early-exit gate (B5).

Two phases, split so the GPU-bound collect runs once and the sweep re-runs free:

collect: for every pair in outputs/r2_ood_20260906/corpus_list.json, run
    benchmark_edm_site_replay.py with SFM_EDM_OOD_MODE=audit (record only,
    zero behavior change) and store the replay JSON under
    outputs/r2_ood_20260906/sweep/.
dump: read the sweep replays, extract per-frame OOD features
    (vpr_* + match_* row keys written by the audit hook) with in_map labels.
sweep: grid-search OODThresholds, print the FRR/FAR curve, and report the
    operating point: in-map FRR <= 2% with off-map FAR == 0.

Gate: FRR<=2% at FAR==0 on the 14-pair cross-product corpus, plus PnP
call-count parity in audit mode (no skipped PnP).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VALIDATION = REPO / "定位演算法" / "validation"
DEPLOY = REPO / "定位演算法" / "deploy_code" / "sfm_glomap_deploy"
CORPUS = REPO / "outputs" / "r2_ood_20260906" / "corpus_list.json"
SWEEP_DIR = REPO / "outputs" / "r2_ood_20260906" / "sweep"

sys.path.insert(0, str(DEPLOY))
import ood_early_exit as ood  # noqa: E402


def collect(args) -> None:
    corpus = json.loads(CORPUS.read_text())
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SFM_EDM_OOD_MODE="audit")
    for entry in corpus["pairs"]:
        tag = f"{Path(entry['video']).stem}__{entry['replay_map']}"
        out = SWEEP_DIR / f"{tag}.json"
        if out.exists() and not args.redo:
            print(f"skip {tag} (exists, --redo to rerun)")
            continue
        cmd = [
            sys.executable, str(VALIDATION / "benchmark_edm_site_replay.py"),
            "--video", entry["video"],
            "--site-profile", entry["site_profile"],
            "--out", str(out),
            "--stride", "3",
        ]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env,
                              capture_output=True, text=True)
        print(proc.stdout[-2000:] if proc.stdout else "")
        if proc.returncode != 0:
            print(f"FAIL {tag} rc={proc.returncode}\n{proc.stderr[-3000:]}")
            if not args.keep_going:
                raise SystemExit(proc.returncode)


def load_frames():
    corpus = {f"{Path(e['video']).stem}__{e['replay_map']}": e["in_map"]
              for e in json.loads(CORPUS.read_text())["pairs"]}
    frames = []
    for path in sorted(SWEEP_DIR.glob("*.json")):
        in_map = corpus.get(path.stem)
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            feat = {k: row.get(k) for k in (
                "vpr_top1", "vpr_margin", "vpr_entropy",
                "match_corr_best", "match_mconf_best_p50",
                "ood_verdict", "ood_vpr_vote", "ood_match_vote")}
            if feat["vpr_top1"] is None and feat["match_corr_best"] is None:
                continue
            frames.append({"file": path.name, "in_map": in_map, **feat})
    return frames


def sweep(args) -> None:
    import math
    frames = load_frames()
    print(f"frames with features: {len(frames)}")
    pool = [f for f in frames]
    in_frames = [f for f in pool if f["in_map"]]
    out_frames = [f for f in pool if f["in_map"] is False]
    print(f"in_map frames: {len(in_frames)}, off_map frames: {len(out_frames)}")
    best = None
    for top1, margin, mconf, corr in itertools.product(
            [0.1, 0.2, 0.3], [0.02, 0.05, 0.1], [0.1, 0.2], [10, 30]):
        th = ood.OODThresholds(
            vpr_top1_out=top1, vpr_margin_out=margin,
            match_mconf_out=mconf, match_corr_out=corr,
            vpr_entropy_out=1.0)
        fp = fn = ti = to = 0
        for f in pool:
            v = ood.evaluate(
                {k: f[k] for k in ("vpr_top1", "vpr_margin", "vpr_entropy")
                 if f[k] is not None} or None,
                {"match_corr_best": f["match_corr_best"],
                 "match_mconf_best_p50": f["match_mconf_best_p50"]},
                th, ood.MODE_HARD)
            hard = v.decision == ood.DECISION_HARD
            if f["in_map"]:
                ti += 1
                fn += hard
            elif f["in_map"] is False:
                to += 1
                fp += not hard
        frr = fn / max(ti, 1)
        far = 1 - (to - fp) / max(to, 1) if to else float("nan")
        tag = "OK " if (frr <= 0.02 and (to == 0 or fp == 0)) else "   "
        print(f"{tag}top1<{top1} margin<{margin} mconf<{mconf} corr<{corr}: "
              f"FRR={frr:.3f} FAR~{far:.3f} (n_in={ti} n_out={to})")
        if frr <= 0.02 and (to == 0 or fp == 0):
            best = (top1, margin, mconf, corr, frr)
    if best:
        print("OPERATING POINT:", best)
    else:
        print("NO operating point at FRR<=2%/FAR==0 in this grid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()
    if args.collect:
        collect(args)
    if args.sweep:
        sweep(args)
    if not (args.collect or args.sweep):
        ap.print_help()


if __name__ == "__main__":
    main()
