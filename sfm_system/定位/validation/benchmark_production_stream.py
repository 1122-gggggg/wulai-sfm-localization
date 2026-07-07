#!/usr/bin/env python3
"""Benchmark the production XFeat tracker on a directory as a 720p stream.

Default query stream:
  /media/cihcilab/新增磁碟區/目標場域影片/驗證/P0710071_frames

Frames are read sequentially, resized to 1280x720 by default, and fed through:
  BOOT_INIT/LOST: MegaLoc topK=30 -> XFeat+LighterGlue -> PnP
  TRACK/WEAK_TRACK: XFeat mutual-NN fast pass; if not strong enough,
                    XFeat+LighterGlue adaptive top3 -> top5.  No MegaLoc here,
                    so vpr_ms is expected to be 0 in TRACK/WEAK_TRACK.

This reports runtime health (FPS, latency, state distribution, inliers). External
P0710071 frames have no ground-truth pose here, so it does not claim metric error.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch

# This file is shipped both inside deploy_code/sfm_glomap_deploy/ (imports work
# from the script dir) and as a real file under 定位/validation/ in the transfer
# package, where the tracker modules live one level up in deploy_code/. Make the
# sibling imports work from either location without requiring PYTHONPATH.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "deploy_code" / "sfm_glomap_deploy"):
    if (_cand / "production_xfeat_tracker.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from production_xfeat_tracker import (
    MegaLocLayer,
    ProductionConfig,
    ProductionXFeatTracker,
    scaled_simple_radial_camera,
)
from reloc_localizer_xfeat import XFeatRelocMap, DEVICE

ROOT = Path(__file__).resolve().parents[1]
QUERY_DIR = ROOT.parents[2] / "建圖" / "inputs" / "目標場域影片" / "驗證" / "P0710071_frames"
MODEL = ROOT / "maps" / "base_glomap_fused_0"
BUNDLE = ROOT / "bundles" / "current_reloc_map_updated_v3.pt"
IMAGES_FUSED = ROOT / "maps" / "base_images_fused"
CACHE_NPY = ROOT / "bundles" / "base_megaloc_cache_v3.npz"
CACHE_META = ROOT / "bundles" / "base_megaloc_cache_v3.json"
OUT_DIR = ROOT / "outputs" / "production_stream_bench"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_rgb_resized(path: Path, width: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    h0, w0 = bgr.shape[:2]
    if width > 0 and w0 != width:
        height = int(round(h0 * (width / w0)))
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, (w0, h0), (w, h)


def robust_1080_camera_params(model_path: Path, orig_size: tuple[int, int]) -> list[float]:
    rec = pycolmap.Reconstruction(str(model_path))
    w0, h0 = orig_size
    vals = []
    for cam in rec.cameras.values():
        if int(cam.width) == int(w0) and int(cam.height) == int(h0) and str(cam.model).endswith("SIMPLE_RADIAL"):
            vals.append([float(x) for x in cam.params])
    if vals:
        arr = np.asarray(vals, float)
        return [float(np.median(arr[:, i])) for i in range(arr.shape[1])]
    # Conservative fallback for this ANAFI map if the model only has 720p cams.
    return [1401.8, w0 / 2.0, h0 / 2.0, 0.002]


def pct(vals, p):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, float), p))


def stat(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"median": None, "p75": None, "p90": None, "mean": None}
    arr = np.asarray(vals, float)
    return {
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(np.mean(arr)),
    }


def count_by(rows, key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        val = str(r.get(key, "?"))
        out[val] = out.get(val, 0) + 1
    return out


def write_camera_positions_ply(rows: list[dict], out_ply: Path) -> int:
    colors = {
        "temporal_nn_cache_accept": (0, 220, 120),
        "nn_fast_accept": (40, 120, 255),
        "nn_fast": (90, 160, 255),
        "lg_adaptive_after_nn": (255, 190, 40),
        "lg_full_after_nn": (255, 80, 40),
        "temporal_nn_cache": (120, 240, 180),
    }
    default_color = (180, 180, 180)
    start_color = (40, 230, 100)
    end_color = (255, 60, 50)
    marker_size = 0.18
    vertices = []
    faces = []
    pose_rows = [r for r in rows if r.get("pose")]

    def add_vec(a, b, s):
        return (a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s)

    def add_frustum(row: dict, color: tuple[int, int, int]) -> None:
        p = row["pose"]
        center = (float(p["x"]), float(p["y"]), float(p["z"]))
        yaw = float(p.get("yaw", 0.0))
        forward = (math.cos(yaw), 0.0, math.sin(yaw))
        right = (-math.sin(yaw), 0.0, math.cos(yaw))
        up = (0.0, -1.0, 0.0)
        plane = add_vec(center, forward, marker_size * 1.8)
        half_w = marker_size * 0.65
        half_h = marker_size * 0.45
        corners = [
            add_vec(add_vec(plane, right, -half_w), up, -half_h),
            add_vec(add_vec(plane, right, half_w), up, -half_h),
            add_vec(add_vec(plane, right, half_w), up, half_h),
            add_vec(add_vec(plane, right, -half_w), up, half_h),
        ]
        top = add_vec(center, up, marker_size * 0.7)
        base = len(vertices)
        for v in [center, *corners, top]:
            vertices.append((*v, *color))
        faces.extend([
            (base + 0, base + 1, base + 2),
            (base + 0, base + 2, base + 3),
            (base + 0, base + 3, base + 4),
            (base + 0, base + 4, base + 1),
            (base + 1, base + 2, base + 3),
            (base + 1, base + 3, base + 4),
            (base + 0, base + 5, base + 2),
            (base + 0, base + 3, base + 5),
        ])

    for i, r in enumerate(pose_rows):
        pose = r.get("pose")
        if not pose:
            continue
        if i == 0:
            color = start_color
        elif i == len(pose_rows) - 1:
            color = end_color
        else:
            color = colors.get(str(r.get("composite_stage", "")), default_color)
        add_frustum(r, color)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    with out_ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")
    return len(pose_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-dir", default=str(QUERY_DIR))
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--image-root", default=str(IMAGES_FUSED))
    ap.add_argument("--megaloc-cache", default=str(CACHE_NPY))
    ap.add_argument("--megaloc-meta", default=str(CACHE_META))
    ap.add_argument("--resize-width", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--boot-topk", type=int, default=30)
    ap.add_argument("--lost-topk", type=int, default=30)
    ap.add_argument("--local-topk", type=int, default=5)
    ap.add_argument("--weak-local-topk", type=int, default=8)
    ap.add_argument("--near-pool", type=int, default=24)
    ap.add_argument("--radius", type=float, default=0.8)
    ap.add_argument("--max-yaw-diff", type=float, default=90.0)
    ap.add_argument("--track-xfeat-topk", type=int, default=1300)
    ap.add_argument("--acquire-xfeat-topk", type=int, default=2048)
    ap.add_argument("--matcher-mode", choices=["nn_then_lg", "lighterglue", "nn"], default="nn_then_lg")
    ap.add_argument("--nn-min-score", type=float, default=0.85)
    ap.add_argument("--adaptive-first-topk", type=int, default=3)
    ap.add_argument("--adaptive-accept-inliers", type=int, default=100)
    ap.add_argument("--adaptive-accept-reproj", type=float, default=3.5)
    ap.add_argument("--max-corr-per-ref", type=int, default=0)
    ap.add_argument("--max-corr-total", type=int, default=0)
    ap.add_argument("--disable-temporal-cache", action="store_true")
    ap.add_argument("--temporal-cache-min-anchors", type=int, default=80)
    ap.add_argument("--temporal-cache-max-anchors", type=int, default=2048)
    ap.add_argument("--temporal-cache-max-age", type=int, default=2)
    ap.add_argument("--temporal-cache-min-score", type=float, default=0.85)
    ap.add_argument("--temporal-cache-seed-min-inliers", type=int, default=150)
    ap.add_argument("--temporal-cache-seed-max-reproj", type=float, default=3.5)
    ap.add_argument("--acquire-min-inliers", type=int, default=80)
    ap.add_argument("--track-min-inliers", type=int, default=50)
    ap.add_argument("--weak-min-inliers", type=int, default=30)
    ap.add_argument("--good-inliers", type=int, default=80)
    ap.add_argument("--acquire-max-reproj", type=float, default=5.0)
    ap.add_argument("--track-max-reproj", type=float, default=6.0)
    ap.add_argument("--pnp-ransac-max-error", type=float, default=5.0)
    ap.add_argument("--max-jump", type=float, default=2.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--out-ply", default="")
    args = ap.parse_args()

    qdir = Path(args.query_dir)
    frames = sorted(qdir.glob("*.jpg"))
    if args.stride > 1:
        frames = frames[::args.stride]
    if args.limit and args.limit > 0:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"no jpg frames found under {qdir}")

    first_rgb, orig_size, stream_size = load_rgb_resized(frames[0], args.resize_width)
    params0 = robust_1080_camera_params(Path(args.model), orig_size)
    cam = scaled_simple_radial_camera(orig_size[0], orig_size[1], params0, stream_size[0], stream_size[1])

    print(f"device={DEVICE} cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    print(f"frames={len(frames)} query_dir={qdir}")
    print(f"resize {orig_size[0]}x{orig_size[1]} -> {stream_size[0]}x{stream_size[1]}")
    print(f"query camera {cam.model} {cam.width}x{cam.height} params={cam.params}")

    print(f"loading bundle {args.bundle}", flush=True)
    xmap = XFeatRelocMap.load(args.bundle)
    print("bundle meta", xmap.meta, flush=True)

    cache = Path(args.megaloc_cache)
    meta = Path(args.megaloc_meta)
    if cache.exists():
        print(f"loading MegaLoc cache {cache}", flush=True)
        try:
            megaloc = MegaLocLayer.load_cache(cache, xmap.ref_names, input_size=322, device=DEVICE)
        except ValueError as exc:
            print(f"MegaLoc cache incompatible ({exc}); using bundle ref_global descriptors", flush=True)
            megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)
    elif Path(args.image_root).exists():
        print(f"building MegaLoc cache {cache} from {args.image_root}", flush=True)
        megaloc = MegaLocLayer.build_cache(
            xmap.ref_names, Path(args.image_root), cache, meta,
            input_size=322, batch=16, device=DEVICE)
    else:
        print("MegaLoc cache and reference images missing; using bundle ref_global descriptors", flush=True)
        megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)

    cfg = ProductionConfig(
        boot_global_topk=args.boot_topk,
        lost_global_topk=args.lost_topk,
        local_topk=args.local_topk,
        weak_local_topk=args.weak_local_topk,
        near_pool=args.near_pool,
        radius=args.radius,
        max_yaw_diff_deg=args.max_yaw_diff,
        xfeat_topk_track=args.track_xfeat_topk,
        xfeat_topk_acquire=args.acquire_xfeat_topk,
        matcher_mode=args.matcher_mode,
        nn_min_score=args.nn_min_score,
        adaptive_first_topk=args.adaptive_first_topk,
        adaptive_accept_inliers=args.adaptive_accept_inliers,
        adaptive_accept_reproj=args.adaptive_accept_reproj,
        max_corr_per_ref=args.max_corr_per_ref,
        max_corr_total=args.max_corr_total,
        temporal_cache_enabled=not args.disable_temporal_cache,
        temporal_cache_min_anchors=args.temporal_cache_min_anchors,
        temporal_cache_max_anchors=args.temporal_cache_max_anchors,
        temporal_cache_max_age=args.temporal_cache_max_age,
        temporal_cache_min_score=args.temporal_cache_min_score,
        temporal_cache_seed_min_inliers=args.temporal_cache_seed_min_inliers,
        temporal_cache_seed_max_reproj=args.temporal_cache_seed_max_reproj,
        acquire_min_inliers=args.acquire_min_inliers,
        track_min_inliers=args.track_min_inliers,
        weak_min_inliers=args.weak_min_inliers,
        good_inliers=args.good_inliers,
        max_reproj_error_acquire=args.acquire_max_reproj,
        max_reproj_error_track=args.track_max_reproj,
        pnp_ransac_max_error=args.pnp_ransac_max_error,
        max_jump=args.max_jump,
    )
    tracker = ProductionXFeatTracker(xmap, megaloc, frame_source=lambda: None, query_cam=cam, cfg=cfg)
    print("loading models ...", flush=True)
    tracker.ensure_models()
    sync()

    rows = []
    t_all0 = time.perf_counter()
    for i, path in enumerate(frames, 1):
        frame, _orig, _stream = load_rgb_resized(path, args.resize_width)
        sync(); t0 = time.perf_counter()
        pose = tracker.localize_frame(frame)
        sync(); wall_ms = (time.perf_counter() - t0) * 1000.0
        info = dict(tracker.last_info)
        row = {
            "idx": i - 1,
            "frame": path.name,
            "success": pose is not None,
            "wall_ms": wall_ms,
            "pose": None if pose is None else {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw},
            **info,
        }
        rows.append(row)
        if i <= 5 or i % 25 == 0 or i == len(frames):
            print(
                f"{i:4d}/{len(frames)} {path.name} mode={row.get('mode')} -> {row.get('next_mode')} "
                f"ok={row['success']} inl={row.get('inliers')} corr={row.get('corr3d')} "
                f"ms={wall_ms:.1f} vpr={row.get('vpr_ms',0):.1f} "
                f"feat={row.get('feature_ms',0):.1f} match={row.get('match_ms',0):.1f}",
                flush=True,
            )
    sync(); t_all = time.perf_counter() - t_all0

    n = len(rows)
    ok = [r for r in rows if r["success"]]
    by_mode = {}
    for r in rows:
        by_mode[r.get("mode", "?")] = by_mode.get(r.get("mode", "?"), 0) + 1
    by_stage = count_by(rows, "composite_stage")
    steady = rows[min(5, len(rows)):]
    summary = {
        "n": n,
        "success": len(ok),
        "success_rate": len(ok) / max(n, 1),
        "actual_wall_s": float(t_all),
        "actual_fps_including_io": float(n / max(t_all, 1e-9)),
        "mode_counts": by_mode,
        "composite_stage_counts": by_stage,
        # The tracker folds temporal-cache anchors into the NN pass (there is no
        # separate "temporal_nn_cache_accept" stage), so count frames where the
        # cache actually contributed correspondences.
        "temporal_cache_accept": int(by_stage.get("temporal_nn_cache_accept", 0)),
        "temporal_cache_used_frames": sum(
            1 for r in rows if (r.get("temporal_cache_corr3d") or 0) > 0),
        "temporal_cache_attempted_frames": sum(
            1 for r in rows if r.get("temporal_cache_attempted")),
        "nn_fast_accept": int(by_stage.get("nn_fast_accept", 0)),
        "lg_fallback": int(by_stage.get("lg_adaptive_after_nn", 0) + by_stage.get("lg_full_after_nn", 0)),
        "wall_ms": stat([r.get("wall_ms") for r in rows]),
        "wall_ms_steady_excluding_first5": stat([r.get("wall_ms") for r in steady]),
        "fps_from_median_wall": None if pct([r.get("wall_ms") for r in rows], 50) is None else 1000.0 / pct([r.get("wall_ms") for r in rows], 50),
        "fps_from_steady_median_wall": None if pct([r.get("wall_ms") for r in steady], 50) is None else 1000.0 / pct([r.get("wall_ms") for r in steady], 50),
        "inliers": stat([r.get("inliers") for r in rows]),
        "corr3d": stat([r.get("corr3d") for r in rows]),
        "raw_matches": stat([r.get("raw_matches") for r in rows]),
        "vpr_ms": stat([r.get("vpr_ms") for r in rows]),
        "feature_ms": stat([r.get("feature_ms") for r in rows]),
        "match_ms": stat([r.get("match_ms") for r in rows]),
        "pnp_ms": stat([r.get("pnp_ms") for r in rows]),
        "reproj_rms": stat([r.get("reproj_rms") for r in rows]),
    }
    payload = {
        "args": vars(args),
        "device": DEVICE,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bundle_meta": xmap.meta,
        "query_camera": {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params},
        "summary": summary,
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / "current_production_default_nn_then_lg_720p_stream.json"
    out_ply = Path(args.out_ply) if args.out_ply else out.with_name(out.stem + "_camera_positions.ply")
    ply_vertices = write_camera_positions_ply(rows, out_ply)
    payload["camera_positions_ply"] = str(out_ply)
    payload["camera_positions_ply_vertices"] = int(ply_vertices)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\n=== production stream summary ===")
    print(f"localized {summary['success']}/{summary['n']} = {100*summary['success_rate']:.1f}%")
    print(f"mode_counts {summary['mode_counts']}")
    print(f"stage_counts {summary['composite_stage_counts']}")
    print(f"actual FPS incl I/O {summary['actual_fps_including_io']:.2f}")
    def fmt(x, nd=2):
        return "NA" if x is None else f"{x:.{nd}f}"
    print(f"median wall FPS {fmt(summary['fps_from_median_wall'])}  steady(excl first5) {fmt(summary['fps_from_steady_median_wall'])}")
    print(f"wall_ms median/p90 {fmt(summary['wall_ms']['median'],1)}/{fmt(summary['wall_ms']['p90'],1)}")
    print(f"inliers median/p90 {fmt(summary['inliers']['median'],0)}/{fmt(summary['inliers']['p90'],0)}")
    print(f"match_ms median/p90 {fmt(summary['match_ms']['median'],1)}/{fmt(summary['match_ms']['p90'],1)}")
    print(f"saved {out}")
    print(f"saved camera positions {out_ply} ({ply_vertices} vertices)")


if __name__ == "__main__":
    main()
