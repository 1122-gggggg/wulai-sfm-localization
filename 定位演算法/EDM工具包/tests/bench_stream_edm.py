#!/usr/bin/env python3
"""Run the EDM tracker over the real P1180118 24fps flight and compare it, frame by frame,
against the XFeat baseline's own recorded run.

The baseline (P1180118_full_24fps_map_ba) localized all 1895 frames and recorded, per
frame, its pose AND its latency breakdown. Comparing against that file needs no frame
alignment guessing -- frame_index i is frame_index i -- and it is the same yardstick the
baseline's own accuracy (0.0013 map-units / 0.050 deg vs the map references) was measured
on.

Reported: pose agreement with the baseline, tracker state distribution, and the latency
breakdown of both routes on the same frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
from production_edm_tracker import EDMConfig, ProductionEDMTracker  # noqa: E402
from reloc_localizer_edm import Camera, EDMRelocMap  # noqa: E402

VIDEO = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/raw/河濱樹24fps 2.7k/P1180118.MP4")
SITE = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/runs/river_site_pi3_1fps_smoke_pinhole_fix")
MODEL = SITE / "gluemap" / "gluemap_aba"
BASE_CSV = Path("/media/cihcilab/新增磁碟區/河濱場域/gluemap_build/localization/"
                "P1180118_megaloc_xfeat_lighterglue_mnn_20260713/outputs/"
                "P1180118_full_24fps_map_ba/frames.csv")
BUNDLE = ROOT / "outputs" / "river_site_reloc_map_edm.pt"
W, H = 1280, 720


def rot_err_deg(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1))))


def load_baseline(path: Path):
    poses, timing = {}, []
    with open(path) as f:
        for row in csv.DictReader(f):
            i = int(row["frame_index"])
            if row["accepted"].lower() == "true":
                poses[i] = (
                    np.array(json.loads(row["cam_from_world_rotation"]), float),
                    np.array([float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]),
                )
            timing.append({k: float(row[k] or 0) for k in
                           ("vpr_ms", "feature_ms", "match_ms", "pnp_ms", "total_ms")})
    return poses, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--local-topk", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "stream_edm_result.json"))
    args = ap.parse_args()

    base_poses, base_timing = load_baseline(BASE_CSV)
    print(f"baseline: {len(base_poses)} accepted poses, {len(base_timing)} frames")

    rec = pycolmap.Reconstruction(str(MODEL))
    c0 = list(rec.cameras.values())[0]
    cam = Camera(model=c0.model.name, width=c0.width, height=c0.height, params=list(c0.params))

    rmap = EDMRelocMap.load(BUNDLE)
    trk = ProductionEDMTracker(rmap, cam, EDMConfig(local_topk=args.local_topk))

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {VIDEO}")

    states, lat, vpr, match, pnp, inl, ncorr = [], [], [], [], [], [], []
    perr, rerr = [], []
    i = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        info = trk.localize(frame)
        states.append(info["state_out"])
        lat.append(info["total_ms"])
        vpr.append(info["vpr_ms"])
        match.append(info["match_ms"])
        pnp.append(info["pnp_ms"])
        if info.get("ok"):
            inl.append(info["inliers"])
            ncorr.append(info["n_corr"])
            if i in base_poses:
                R_b, C_b = base_poses[i]
                perr.append(float(np.linalg.norm(info["center"] - C_b)))
                rerr.append(rot_err_deg(info["R"], R_b))
        i += 1
        if i % 300 == 0:
            print(f"  {i}  state={info['state_out']} inliers={info.get('inliers',0)} "
                  f"{np.median(lat[-300:]):.0f}ms")
    cap.release()
    wall = time.perf_counter() - t0

    n_ok = len(inl)
    print("\n" + "=" * 66)
    print(f"frames={i}  localized={n_ok} ({100*n_ok/max(i,1):.1f}%)  wall={wall:.0f}s")
    print(f"states: {dict(Counter(states))}")
    print(f"correspondences median={np.median(ncorr):.0f}   PnP inliers median={np.median(inl):.0f}")

    def med(a, k):
        return np.median([x[k] for x in a])

    print(f"\nlatency (ms, median)      EDM      XFeat baseline (same frames, same GPU)")
    print(f"  retrieval (vpr)   {np.median(vpr):>8.1f}   {med(base_timing,'vpr_ms'):>8.1f}")
    print(f"  feature           {'-':>8}   {med(base_timing,'feature_ms'):>8.1f}")
    print(f"  match             {np.median(match):>8.1f}   {med(base_timing,'match_ms'):>8.1f}")
    print(f"  pnp               {np.median(pnp):>8.1f}   {med(base_timing,'pnp_ms'):>8.1f}")
    print(f"  TOTAL             {np.median(lat):>8.1f}   {med(base_timing,'total_ms'):>8.1f}")
    print(f"  FPS               {1000/np.median(lat):>8.1f}   {1000/med(base_timing,'total_ms'):>8.1f}")

    if perr:
        perr, rerr = np.array(perr), np.array(rerr)
        print(f"\npose agreement with the baseline run ({len(perr)} frames)")
        print(f"  position (map-u): median={np.median(perr):.4f}  p90={np.percentile(perr,90):.4f}  max={perr.max():.4f}")
        print(f"  rotation (deg)  : median={np.median(rerr):.3f}  p90={np.percentile(rerr,90):.3f}  max={rerr.max():.3f}")

    Path(args.out).write_text(json.dumps({
        "frames": i, "localized": n_ok,
        "states": dict(Counter(states)),
        "median_ms": {"vpr": float(np.median(vpr)), "match": float(np.median(match)),
                      "pnp": float(np.median(pnp)), "total": float(np.median(lat))},
        "baseline_median_ms": {k: float(med(base_timing, k)) for k in
                               ("vpr_ms", "feature_ms", "match_ms", "pnp_ms", "total_ms")},
        "inliers_median": float(np.median(inl)) if inl else 0,
        "pose_vs_baseline": {"position_median": float(np.median(perr)) if len(perr) else None,
                             "rotation_median_deg": float(np.median(rerr)) if len(rerr) else None},
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
