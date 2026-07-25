#!/usr/bin/env python3
"""End-to-end accuracy check of the EDM localizer. Runs with nothing but the package.

Every reference is re-localized as if it were an unseen query, with ITSELF excluded from
retrieval, and the recovered pose is compared to its known map pose. This exercises the
whole chain -- retrieval, EDM matching, the cell->xyz lookup, PnP -- and is where a wrong
2D-3D correspondence shows up as a pose error instead of as a plausible inlier count.

Self-contained by construction:
  - the reference images are embedded in the bundle (EDM must see them at flight time)
  - the reference poses ship as maps/river_site_ref_poses.json
  - retrieval needs no MegaLoc model: the query IS reference i, so its global descriptor
    is already ref_global[i]. Cosine over ref_global reproduces MegaLoc's ranking exactly.

So this needs no COLMAP model, no image directory, and no network. Reference numbers from
the build machine (RTX 5090, topk=5): 98.2% localized, 0.0006 map-unit / 0.044 deg median.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for cand in (ROOT / "deploy", ROOT):
    if (cand / "reloc_localizer_edm.py").exists():
        sys.path.insert(0, str(cand))
        break
from edm_matcher import EDMMatcher  # noqa: E402
from reloc_localizer_edm import Camera, EDMLocalizer, EDMRelocMap  # noqa: E402


def _default(kind, fallback):
    """Resolve bundle / ref_poses from config.json. Hardcoding the site's filenames here
    breaks every other packaged map -- the default must follow the package it ships in."""
    cfg = ROOT / "config.json"
    rel = fallback
    if cfg.exists():
        rel = json.loads(cfg.read_text())["paths"].get(kind, fallback)
    for base in (ROOT, ROOT / "outputs"):
        p = base / rel
        if p.exists():
            return str(p)
        p = base / Path(rel).name          # dev tree: outputs/<name>, no bundles/ dir
        if p.exists():
            return str(p)
    return str(ROOT / rel)


def rot_err_deg(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=_default("bundle", "bundles/river_site_reloc_map_edm.pt"))
    ap.add_argument("--poses", default=_default("ref_poses", "maps/river_site_ref_poses.json"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--edm-topk", type=int, default=None,
                    help="override EDM coarse match top-k")
    ap.add_argument("--every", type=int, default=8, help="evaluate every Nth reference")
    ap.add_argument("--min-inliers", type=int, default=50)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--out", help="optional JSON result path")
    args = ap.parse_args()

    meta = json.loads(Path(args.poses).read_text())
    gt = {n: (np.array(v["R"]), np.array(v["t"])) for n, v in meta["poses"].items()}

    # A map may mix source resolutions (target_site: 720p / 1080p / 2.7K). Here the QUERY is
    # a reference image, so it must be posed with its OWN camera. (At flight time this does
    # not arise: the query is always the 720p live stream, and the references only ever
    # contribute 3D points, which carry no resolution.)
    cams = meta.get("cameras") or {"1": meta["camera"]}
    cam_id_of = {n: v.get("camera_id", "1") for n, v in meta["poses"].items()}

    def make_cam(c):
        return Camera(model=c["model"], width=c["width"], height=c["height"], params=list(c["params"]))

    rmap = EDMRelocMap.load(args.bundle)
    anchored = np.mean([np.isfinite(v[:, 0]).sum() for v in rmap.xyz_by_cell.values()])
    print(f"bundle: {len(rmap.ref_names)} refs, {anchored:.0f} 3D-anchored cells/ref, "
          f"{len(cams)} camera(s)")
    matcher = EDMMatcher(mconf_thr=0.2, topk=args.edm_topk, fp16=not args.no_fp16)
    locs = {cid: EDMLocalizer(rmap, make_cam(c), matcher=matcher, min_inliers=args.min_inliers)
            for cid, c in cams.items()}

    G = rmap.ref_global
    queries = [(i, n) for i, n in enumerate(rmap.ref_names)][::args.every]
    perr, rerr, inl, ncorr, fails, times = [], [], [], [], [], []

    for step, (i, name) in enumerate(queries, 1):
        if name not in gt:
            continue
        # retrieval without MegaLoc: the query's own global descriptor is ref_global[i]
        sim = G @ G[i]
        sim[i] = -np.inf                       # leave-one-out: self is not a candidate
        refs = [rmap.ref_names[j] for j in np.argsort(-sim)[:args.topk]]

        loc = locs[cam_id_of[name]]
        cam = loc.cam
        t0 = time.perf_counter()
        p2, p3, _confidence, _ = loc.correspondences(rmap.images[name], refs)
        ret = None
        if len(p3) >= 6:
            pcam = pycolmap.Camera(model=cam.model, width=cam.width, height=cam.height,
                                   params=cam.params)
            opts = pycolmap.AbsolutePoseEstimationOptions()
            opts.ransac.max_error = 5.0
            ret = pycolmap.estimate_and_refine_absolute_pose(
                np.asarray(p2, float), np.asarray(p3, float), pcam, opts)
        times.append((time.perf_counter() - t0) * 1e3)

        if ret is None or int(ret["num_inliers"]) < args.min_inliers:
            fails.append(name)
            continue
        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        C = -R.T @ np.asarray(T.translation)
        R_gt, t_gt = gt[name]
        C_gt = -R_gt.T @ t_gt
        perr.append(float(np.linalg.norm(C - C_gt)))
        rerr.append(rot_err_deg(R, R_gt))
        inl.append(int(ret["num_inliers"]))
        ncorr.append(len(p3))
        if step % 25 == 0:
            print(f"  {step}/{len(queries)}  inliers={inl[-1]} perr={perr[-1]:.4f}")

    n, ok = len(queries), len(perr)
    print("\n" + "=" * 60)
    print(f"queries={n}  localized={ok} ({100*ok/max(n,1):.1f}%)  failed={len(fails)}")
    if ok:
        perr, rerr = np.array(perr), np.array(rerr)
        print(f"correspondences : median={np.median(ncorr):.0f}  min={min(ncorr)}")
        print(f"PnP inliers     : median={np.median(inl):.0f}  min={min(inl)}")
        print(f"position (map-u): median={np.median(perr):.4f}  p90={np.percentile(perr,90):.4f}  max={perr.max():.4f}")
        print(f"rotation (deg)  : median={np.median(rerr):.3f}  p90={np.percentile(rerr,90):.3f}  max={rerr.max():.3f}")
        print(f"match+PnP (ms)  : median={np.median(times):.0f}   (topk={args.topk})")
        print(f"within 0.01 map-u & 0.5 deg: {100*np.mean((perr<0.01)&(rerr<0.5)):.1f}%")
    if fails:
        print(f"failed: {fails[:10]}{' ...' if len(fails) > 10 else ''}")
    if args.out:
        result = {
            "bundle": str(Path(args.bundle).resolve()),
            "poses": str(Path(args.poses).resolve()),
            "retrieval_topk": args.topk,
            "edm_topk": matcher.topk,
            "every": args.every,
            "min_inliers": args.min_inliers,
            "fp16": not args.no_fp16,
            "queries": n,
            "localized": ok,
            "rate": ok / max(n, 1),
            "failed": fails,
            "correspondences_median": float(np.median(ncorr)) if ok else None,
            "inliers_median": float(np.median(inl)) if ok else None,
            "position_map_units": {
                "median": float(np.median(perr)) if ok else None,
                "p90": float(np.percentile(perr, 90)) if ok else None,
                "max": float(np.max(perr)) if ok else None,
            },
            "rotation_degrees": {
                "median": float(np.median(rerr)) if ok else None,
                "p90": float(np.percentile(rerr, 90)) if ok else None,
                "max": float(np.max(rerr)) if ok else None,
            },
            "match_pnp_ms": {
                "median": float(np.median(times)) if times else None,
                "p95": float(np.percentile(times, 95)) if times else None,
            },
            "within_0_01_map_units_and_0_5_degrees": (
                float(np.mean((perr < 0.01) & (rerr < 0.5))) if ok else None
            ),
            "absolute_accuracy_available": False,
            "accuracy_scope": "leave-one-reference-out proxy against map poses",
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0 if ok >= 0.95 * n else 1


if __name__ == "__main__":
    raise SystemExit(main())
