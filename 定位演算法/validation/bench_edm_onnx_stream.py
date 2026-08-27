#!/usr/bin/env python3
"""Stream-localize a 720p validation video with torch / ORT-CUDA / ORT-TensorRT EDM.

Reports median total latency, end-to-end FPS, and whether the 17 FPS target is met.

Example:
  python bench_edm_onnx_stream.py \\
    --video /path/to/video.mp4 \\
    --bundle /path/to/localization_bundle.pt \\
    --backend onnx_tensorrt --max-frames 300 --target-fps 17
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

LOC_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
sys.path.insert(0, str(DEPLOY))

from edm_localizer_adapter import CAM_720_EDM, production_edm_config  # noqa: E402
from edm_onnx_matcher import make_matcher  # noqa: E402
from production_edm_tracker import ProductionEDMTracker  # noqa: E402
from reloc_localizer_edm import Camera, EDMRelocMap  # noqa: E402

STREAM_W, STREAM_H = 1280, 720


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    a = np.asarray(xs, dtype=float)
    return float(np.percentile(a, p))


def _run_stream(
    args: argparse.Namespace,
    trk: ProductionEDMTracker,
) -> tuple[list[str], list[float], list[float], list[float], list[float], list[int], int, int, int, float]:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    states: list[str] = []
    totals: list[float] = []
    vprs: list[float] = []
    matches: list[float] = []
    pnps: list[float] = []
    inliers: list[int] = []
    n_ok = 0
    frame_i = 0
    used = 0
    timed = 0
    t_wall0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and used >= args.max_frames + max(0, args.warmup_frames):
            break
        if frame_i % max(1, args.stride) != 0:
            frame_i += 1
            continue
        frame = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_AREA)
        info = trk.localize(frame)
        used += 1
        frame_i += 1

        is_warmup = used <= args.warmup_frames
        if not is_warmup:
            timed += 1
            states.append(str(info.get("state_out")))
            totals.append(float(info.get("total_ms") or 0.0))
            vprs.append(float(info.get("vpr_ms") or 0.0))
            matches.append(float(info.get("match_ms") or 0.0))
            pnps.append(float(info.get("pnp_ms") or 0.0))
            if info.get("ok"):
                n_ok += 1
                inliers.append(int(info.get("inliers") or 0))

        if used % 50 == 0 or used == 1:
            print(
                f"  frame used={used} timed={timed} state={info.get('state_out')} "
                f"ok={bool(info.get('ok'))} inliers={info.get('inliers', 0)} "
                f"total={info.get('total_ms', 0):.1f}ms "
                f"match={info.get('match_ms', 0):.1f}ms",
                flush=True,
            )

    cap.release()
    wall_s = time.perf_counter() - t_wall0
    return states, totals, vprs, matches, pnps, inliers, n_ok, used, timed, wall_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument(
        "--backend",
        default="onnx_tensorrt",
        choices=(
            "torch", "torch_fp32",
            "onnx_cuda", "onnx_tensorrt", "onnx_cpu",
        ),
        help="EDM matching backend",
    )
    ap.add_argument("--onnx", type=Path, default=None,
                    help="Override ONNX path (default flight 1024x576 outdoor export)")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--warmup-frames", type=int, default=5,
                    help="Skip first N timed frames (TRT build / CUDA warmup)")
    ap.add_argument("--local-topk", type=int, default=2)
    ap.add_argument("--boot-global-topk", type=int, default=10)
    ap.add_argument("--target-fps", type=float, default=17.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="Keep every Nth source frame (1 = every frame)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    if not args.bundle.is_file():
        raise SystemExit(f"bundle not found: {args.bundle}")

    print(f"[bench] backend={args.backend} video={args.video} "
          f"max_frames={args.max_frames} target_fps={args.target_fps}", flush=True)

    print("[bench] loading map ...", flush=True)
    rmap = EDMRelocMap.load(args.bundle)
    cam = Camera(*CAM_720_EDM)
    cfg = production_edm_config()
    cfg.local_topk = int(args.local_topk)
    cfg.boot_global_topk = int(args.boot_global_topk)

    print(f"[bench] loading matcher backend={args.backend} ...", flush=True)
    matcher_kwargs = {}
    if args.onnx is not None:
        matcher_kwargs["onnx_path"] = args.onnx
    matcher = make_matcher(args.backend, **matcher_kwargs)

    trk = ProductionEDMTracker(rmap, cam, cfg=cfg, matcher=matcher)
    # Force MegaLoc load before timed loop so VPR init is not attributed to frame 0 only.
    print("[bench] warming MegaLoc ...", flush=True)
    _ = trk.loc.megaloc

    states, totals, vprs, matches, pnps, inliers, n_ok, used, timed, wall_s = _run_stream(args, trk)

    n = max(timed, 1)
    med_total = percentile(totals, 50)
    p90_total = percentile(totals, 90)
    mean_total = float(np.mean(totals)) if totals else float("nan")
    # End-to-end FPS from timed wall segments would need per-frame wall clocks;
    # report both inverse-median and overall processed rate.
    fps_from_median = 1000.0 / med_total if med_total > 0 else 0.0
    fps_wall = timed / max(wall_s, 1e-6)
    target = float(args.target_fps)
    pass_median = fps_from_median >= target
    pass_wall = fps_wall >= target

    providers = getattr(matcher, "active_providers", ["torch"])
    result = {
        "backend": args.backend,
        "active_providers": providers,
        "video": str(args.video),
        "bundle": str(args.bundle),
        "frames_used": used,
        "frames_timed": timed,
        "warmup_frames": args.warmup_frames,
        "localized": n_ok,
        "localized_pct": 100.0 * n_ok / n,
        "states": dict(Counter(states)),
        "median_total_ms": med_total,
        "p90_total_ms": p90_total,
        "mean_total_ms": mean_total,
        "median_vpr_ms": percentile(vprs, 50),
        "median_match_ms": percentile(matches, 50),
        "median_pnp_ms": percentile(pnps, 50),
        "median_inliers": percentile(inliers, 50) if inliers else None,
        "fps_from_median_total_ms": fps_from_median,
        "fps_wall_including_io": fps_wall,
        "target_fps": target,
        "meets_target_median": pass_median,
        "meets_target_wall": pass_wall,
        "wall_s": wall_s,
        "local_topk": cfg.local_topk,
        "boot_global_topk": cfg.boot_global_topk,
    }

    print("\n" + "=" * 72)
    print(f"backend          : {args.backend}  providers={providers}")
    print(f"frames timed     : {timed}  (warmup skipped {args.warmup_frames})")
    print(f"localized        : {n_ok}/{timed} ({result['localized_pct']:.1f}%)")
    print(f"states           : {result['states']}")
    print(f"median total     : {med_total:.2f} ms   (p90 {p90_total:.2f}, mean {mean_total:.2f})")
    print(f"median match     : {result['median_match_ms']:.2f} ms")
    print(f"median vpr/pnp   : {result['median_vpr_ms']:.2f} / {result['median_pnp_ms']:.2f} ms")
    print(f"median inliers   : {result['median_inliers']}")
    print(f"FPS (1/med_ms)   : {fps_from_median:.2f}")
    print(f"FPS (wall/IO)    : {fps_wall:.2f}")
    print(f"target           : {target:.1f} FPS")
    print(f"PASS median>=tgt : {pass_median}")
    print(f"PASS wall>=tgt   : {pass_wall}")
    print("=" * 72)

    out = args.out or (
        LOC_ROOT / "outputs" / "edm_onnx_bench"
        / f"{args.backend}_{args.video.stem}_n{timed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out}")

    # Non-zero exit if neither metric meets target (caller can ignore).
    if not (pass_median or pass_wall):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
