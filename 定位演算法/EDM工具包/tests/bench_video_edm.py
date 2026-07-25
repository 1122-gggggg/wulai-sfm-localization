#!/usr/bin/env python3
"""Run the EDM tracker over a held-out query video. Site-agnostic: bundle and camera come
from config.json, so it works on any packaged map.

These videos were never mapped, so there is NO ground-truth pose to score against. What is
measurable without one, and is reported here:

  - localization rate and the tracker's state distribution (TRACK / WEAK_TRACK / LOST)
  - PnP inliers and correspondence counts
  - latency breakdown (retrieval / matching / PnP)
  - trajectory continuity: the per-frame step in map units. At 24 fps a real flight moves a
    small, smooth amount per frame, so a heavy tail here means the pose is jumping around --
    the one failure mode a rate-only summary would hide.

Nothing here proves absolute accuracy. It shows whether the map SUPPORTS this flight.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
from production_edm_tracker import EDMConfig, ProductionEDMTracker  # noqa: E402
from reloc_localizer_edm import Camera, EDMRelocMap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--local-topk", type=int, default=1)
    ap.add_argument("--boot-global-topk", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    pc = cfg["pnp_camera"]
    cam = Camera(model=pc["model"], width=pc["width"], height=pc["height"], params=list(pc["params"]))
    W, H = pc["width"], pc["height"]

    bundle = args.bundle or str(Path(args.config).parent / cfg["paths"]["bundle"])
    if not Path(bundle).exists():
        bundle = str(ROOT / "outputs" / Path(cfg["paths"]["bundle"]).name)
    rmap = EDMRelocMap.load(bundle)
    print(f"map: {len(rmap.ref_names)} refs   camera: {pc['model']} {W}x{H} f={pc['params'][0]}")
    print(f"video: {Path(args.video).name}   local_topk={args.local_topk}")

    trk = ProductionEDMTracker(rmap, cam, EDMConfig(local_topk=args.local_topk,
                                                    boot_global_topk=args.boot_global_topk))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    states, lat, vpr, match, pnp, inl, ncorr, centers = [], [], [], [], [], [], [], []
    reference_sequences, rejections = Counter(), Counter()
    limited_jumps = 0
    i = n_seen = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n_seen >= args.max_frames):
            break
        i += 1
        if (i - 1) % args.stride:
            continue
        n_seen += 1
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        info = trk.localize(frame)
        reference_sequences.update(name.split("/", 1)[0] for name in info.get("refs", []))
        if info.get("rejected"):
            rejections[str(info["rejected"])] += 1
        if info.get("limited_jump"):
            limited_jumps += 1
        states.append(info["state_out"])
        lat.append(info["total_ms"])
        vpr.append(info["vpr_ms"])
        match.append(info["match_ms"])
        pnp.append(info["pnp_ms"])
        if info.get("ok"):
            inl.append(info["inliers"])
            ncorr.append(info["n_corr"])
            centers.append((n_seen, np.asarray(info["center"], float)))
        if n_seen % 300 == 0:
            print(f"  {n_seen}  state={info['state_out']} inliers={info.get('inliers', 0)} "
                  f"{np.median(lat[-300:]):.0f}ms", flush=True)
    cap.release()
    wall = time.perf_counter() - t0

    n_ok = len(inl)
    print("\n" + "=" * 68)
    print(f"frames={n_seen}  localized={n_ok} ({100*n_ok/max(n_seen,1):.1f}%)  wall={wall:.0f}s")
    print(f"states: {dict(Counter(states))}")
    print(f"limited pose-center spikes: {limited_jumps}")
    if n_ok:
        print(f"correspondences median={np.median(ncorr):.0f}   PnP inliers median={np.median(inl):.0f}"
              f"  p05={np.percentile(inl,5):.0f}")
    print(f"latency ms (median): total={np.median(lat):.1f}  retrieval={np.median(vpr):.1f}  "
          f"match={np.median(match):.1f}  pnp={np.median(pnp):.1f}   -> {1000/max(np.median(lat),1e-6):.1f} FPS")

    steps = []
    for (a, ca), (b, cb) in zip(centers, centers[1:]):
        if b - a == 1:
            steps.append(float(np.linalg.norm(cb - ca)))
    if steps:
        s = np.array(steps)
        print(f"trajectory step (map-u, consecutive localized frames, n={len(s)}): "
              f"median={np.median(s):.4f} p95={np.percentile(s,95):.4f} max={s.max():.4f}")
        print(f"  steps > 10x median (pose jumps): {int((s > 10*np.median(s)).sum())}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "video": args.video, "bundle": bundle, "local_topk": args.local_topk,
            "frames": n_seen, "localized": n_ok, "rate": n_ok / max(n_seen, 1),
            "states": dict(Counter(states)),
            "inliers_median": float(np.median(inl)) if n_ok else None,
            "inliers_p05": float(np.percentile(inl, 5)) if n_ok else None,
            "correspondences_median": float(np.median(ncorr)) if n_ok else None,
            "reference_sequence_counts": dict(reference_sequences),
            "rejections": dict(rejections),
            "limited_jumps": limited_jumps,
            "latency_median_ms": {"total": float(np.median(lat)), "retrieval": float(np.median(vpr)),
                                  "match": float(np.median(match)), "pnp": float(np.median(pnp))},
            "step_median": float(np.median(steps)) if steps else None,
            "step_p95": float(np.percentile(steps, 95)) if steps else None,
            "step_max": float(np.max(steps)) if steps else None,
            "jumps_gt_10x_median": int((np.array(steps) > 10 * np.median(steps)).sum()) if steps else None,
        }, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
