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
import hashlib
import json
import math
import re
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


_FLIGHT_TOKEN = re.compile(r"P(\d{3})", re.IGNORECASE)
STANDARD_LOO_ACCURACY_SCOPE = "leave-one-reference-out proxy against map poses"
FLIGHT_DISJOINT_ACCURACY_SCOPE = (
    "leave-one-flight-out proxy against map poses; candidates exclude the query flight"
)


def flight_group(name: str) -> str:
    """Extract a flight id from nested paths or flat filenames.

    DJI-style tokens (P1160116, P116_0001.jpg, site/P117/frame.jpg) collapse to
    P116/P117/... so those map flights stay separate folds. Nested names without
    a P-token use the parent directory; flat names use the stem minus a trailing
    frame index.
    """
    text = str(name).replace("\\", "/")
    parts = [part for part in text.split("/") if part and part not in {".", ".."}]
    ordered = []
    if len(parts) >= 2:
        ordered.append(parts[-2])
        ordered.extend(parts[:-2])
        ordered.append(parts[-1])
    elif parts:
        ordered.append(parts[0])
    else:
        ordered.append(text)
    for raw in ordered:
        stem = raw.rsplit(".", 1)[0] if "." in raw else raw
        match = _FLIGHT_TOKEN.search(stem)
        if match:
            return f"P{match.group(1)}"
    if len(parts) >= 2:
        return parts[-2]
    stem = parts[-1].rsplit(".", 1)[0] if parts else str(name)
    stripped = re.sub(r"[_-]*\d+$", "", stem)
    return stripped or stem


def group_refs_by_flight(ref_names) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in ref_names:
        groups.setdefault(flight_group(name), []).append(name)
    return groups


def require_flight_disjoint_pool(ref_names) -> dict[str, list[str]]:
    groups = group_refs_by_flight(ref_names)
    if len(groups) < 2:
        raise ValueError(
            "leave-one-flight-out needs at least two flights; found "
            f"{sorted(groups)}"
        )
    for flight in groups:
        n_other = sum(len(names) for key, names in groups.items() if key != flight)
        if n_other == 0:
            raise ValueError(f"fold {flight} has no candidates")
    return groups


def mask_retrieval_scores(
    scores, ref_names, query_index: int, *, exclude_same_flight: bool
):
    """Mask illegal retrieval candidates before ranking."""
    masked = np.asarray(scores, dtype=float).copy()
    n = len(ref_names)
    if query_index < 0 or query_index >= n:
        raise IndexError("query index out of range")
    masked[query_index] = -np.inf
    if exclude_same_flight:
        query_flight = flight_group(ref_names[query_index])
        for index, name in enumerate(ref_names):
            if flight_group(name) == query_flight:
                masked[index] = -np.inf
    return masked


def retrieve_topk(scores, ref_names, topk: int) -> list[str]:
    if topk <= 0:
        return []
    ranking = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    refs: list[str] = []
    for index in ranking:
        if not np.isfinite(scores[int(index)]):
            continue
        refs.append(ref_names[int(index)])
        if len(refs) >= topk:
            break
    if not refs:
        raise ValueError("fold has no candidates")
    return refs


def _error_stats(values) -> dict[str, float | None]:
    if len(values) == 0:
        return {"median": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def fold_report(n_queries: int, localized: int, perr, rerr, inl, times) -> dict:
    return {
        "queries": int(n_queries),
        "localized": int(localized),
        "success_rate": localized / max(n_queries, 1),
        "inliers": _error_stats(inl),
        "position_map_units": _error_stats(perr),
        "rotation_degrees": _error_stats(rerr),
        "match_pnp_ms": _error_stats(times),
    }


def macro_and_worst(folds: dict) -> dict:
    if not folds:
        return {
            "macro_success": None,
            "worst_fold": None,
            "worst_fold_success": None,
            "worst_fold_error": {
                "position_map_units": {"median": None, "p90": None, "max": None},
                "rotation_degrees": {"median": None, "p90": None, "max": None},
            },
        }

    def worst_key(flight: str):
        fold = folds[flight]
        pos = fold["position_map_units"]["median"]
        rot = fold["rotation_degrees"]["median"]
        return (
            fold["success_rate"],
            -(pos if pos is not None else -np.inf),
            -(rot if rot is not None else -np.inf),
            flight,
        )

    worst = min(folds, key=worst_key)
    fold = folds[worst]
    return {
        "macro_success": float(np.mean([item["success_rate"] for item in folds.values()])),
        "worst_fold": worst,
        "worst_fold_success": fold["success_rate"],
        "worst_fold_error": {
            "position_map_units": dict(fold["position_map_units"]),
            "rotation_degrees": dict(fold["rotation_degrees"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=_default("bundle", "bundles/river_site_reloc_map_edm.pt"))
    ap.add_argument("--poses", default=_default("ref_poses", "maps/river_site_ref_poses.json"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--edm-topk",
        type=int,
        default=None,
        help="override EDM coarse match top-k",
    )
    ap.add_argument("--mconf-thr", type=float, default=0.2)
    ap.add_argument(
        "--runtime-sigma-mode",
        choices=("reference_grid", "bidirectional"),
        default="reference_grid",
    )
    ap.add_argument("--every", type=int, default=8, help="evaluate every Nth reference")
    ap.add_argument("--min-inliers", type=int, default=50)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--out", help="optional JSON result path")
    ap.add_argument(
        "--exclude-same-flight",
        action="store_true",
        help="leave-one-flight-out: drop every same-flight ref before MegaLoc ranking",
    )
    args = ap.parse_args()
    if not math.isfinite(args.mconf_thr) or not 0.0 <= args.mconf_thr <= 1.0:
        raise SystemExit("--mconf-thr must be finite and within [0, 1]")

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

    bundle_path = Path(args.bundle)
    digest = hashlib.sha256()
    with bundle_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    rmap = EDMRelocMap.load(bundle_path, expected_sha256=digest.hexdigest())
    anchored = np.mean([np.isfinite(v[:, 0]).sum() for v in rmap.xyz_by_cell.values()])
    print(f"bundle: {len(rmap.ref_names)} refs, {anchored:.0f} 3D-anchored cells/ref, "
          f"{len(cams)} camera(s)")
    matcher = EDMMatcher(
        mconf_thr=args.mconf_thr,
        topk=args.edm_topk,
        fp16=not args.no_fp16,
        runtime_sigma_mode=args.runtime_sigma_mode,
    )
    locs = {cid: EDMLocalizer(rmap, make_cam(c), matcher=matcher, min_inliers=args.min_inliers)
            for cid, c in cams.items()}

    G = rmap.ref_global
    names = list(rmap.ref_names)
    queries = [(i, n) for i, n in enumerate(names)][::args.every]
    if args.exclude_same_flight:
        groups = require_flight_disjoint_pool(names)
        print(f"flights: {len(groups)} {sorted(groups)}")

    perr, rerr, inl, ncorr, fails, times = [], [], [], [], [], []
    fold_acc: dict[str, dict] = {}

    for step, (i, name) in enumerate(queries, 1):
        if name not in gt:
            continue
        flight = flight_group(name)
        if args.exclude_same_flight:
            slot = fold_acc.setdefault(
                flight,
                {"queries": 0, "perr": [], "rerr": [], "inl": [], "ncorr": [],
                 "fails": [], "times": []},
            )
            slot["queries"] += 1

        sim = G @ G[i]
        if args.exclude_same_flight:
            sim = mask_retrieval_scores(sim, names, i, exclude_same_flight=True)
            try:
                refs = retrieve_topk(sim, names, args.topk)
            except ValueError as exc:
                raise ValueError(f"fold {flight} has no candidates") from exc
            if any(flight_group(ref) == flight for ref in refs):
                raise ValueError(f"same-flight leakage in fold {flight}")
        else:
            sim[i] = -np.inf                       # leave-one-out: self is not a candidate
            refs = [names[j] for j in np.argsort(-sim)[:args.topk]]

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
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        times.append(elapsed_ms)
        if args.exclude_same_flight:
            fold_acc[flight]["times"].append(elapsed_ms)

        if ret is None or int(ret["num_inliers"]) < args.min_inliers:
            fails.append(name)
            if args.exclude_same_flight:
                fold_acc[flight]["fails"].append(name)
            continue
        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        C = -R.T @ np.asarray(T.translation)
        R_gt, t_gt = gt[name]
        C_gt = -R_gt.T @ t_gt
        pe = float(np.linalg.norm(C - C_gt))
        re = rot_err_deg(R, R_gt)
        n_inl = int(ret["num_inliers"])
        n_p3 = len(p3)
        perr.append(pe)
        rerr.append(re)
        inl.append(n_inl)
        ncorr.append(n_p3)
        if args.exclude_same_flight:
            fold_acc[flight]["perr"].append(pe)
            fold_acc[flight]["rerr"].append(re)
            fold_acc[flight]["inl"].append(n_inl)
            fold_acc[flight]["ncorr"].append(n_p3)
        if step % 25 == 0:
            print(f"  {step}/{len(queries)}  inliers={inl[-1]} perr={perr[-1]:.4f}")

    n, ok = len(queries), len(perr)
    folds = {
        flight: fold_report(
            slot["queries"], len(slot["perr"]), slot["perr"], slot["rerr"],
            slot["inl"], slot["times"],
        )
        for flight, slot in sorted(fold_acc.items())
    }
    summary = macro_and_worst(folds) if args.exclude_same_flight else None
    accuracy_scope = (
        FLIGHT_DISJOINT_ACCURACY_SCOPE if args.exclude_same_flight
        else STANDARD_LOO_ACCURACY_SCOPE
    )

    print("\n" + "=" * 60)
    if args.exclude_same_flight:
        print(f"mode=leave-one-flight-out  accuracy_scope={accuracy_scope}")
        for flight, fold in folds.items():
            pos = fold["position_map_units"]
            rot = fold["rotation_degrees"]
            print(
                f"  fold {flight}: queries={fold['queries']}  "
                f"localized={fold['localized']} ({100 * fold['success_rate']:.1f}%)  "
                f"pos med/p90/max="
                f"{pos['median']}/{pos['p90']}/{pos['max']}  "
                f"rot med/p90/max="
                f"{rot['median']}/{rot['p90']}/{rot['max']}  "
                f"inliers med={fold['inliers']['median']}  "
                f"match+PnP med={fold['match_pnp_ms']['median']}"
            )
        print(f"macro success: {100 * summary['macro_success']:.1f}%")
        worst_pos = summary["worst_fold_error"]["position_map_units"]
        worst_rot = summary["worst_fold_error"]["rotation_degrees"]
        print(
            f"worst fold: {summary['worst_fold']}  "
            f"success={100 * summary['worst_fold_success']:.1f}%  "
            f"pos median={worst_pos['median']}  rot median={worst_rot['median']}"
        )
    else:
        print(f"mode=leave-one-reference-out  accuracy_scope={accuracy_scope}")
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
            "exclude_same_flight": bool(args.exclude_same_flight),
            "queries": n,
            "localized": ok,
            "rate": ok / max(n, 1),
            "mconf_thr": matcher.mconf_thr,
            "runtime_sigma_mode": matcher.runtime_sigma_mode,
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
            "accuracy_scope": accuracy_scope,
        }
        if args.exclude_same_flight:
            result["folds"] = folds
            result["macro_success"] = summary["macro_success"]
            result["worst_fold"] = summary["worst_fold"]
            result["worst_fold_success"] = summary["worst_fold_success"]
            result["worst_fold_error"] = summary["worst_fold_error"]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0 if ok >= 0.95 * n else 1


if __name__ == "__main__":
    raise SystemExit(main())
