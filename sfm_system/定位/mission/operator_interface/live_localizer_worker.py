#!/usr/bin/env python3
"""Persistent stdin/stdout production localizer worker.

Protocol:
  input  : raw RGB frames, fixed width*height*3 bytes each
  output : one JSON line per frame

All model logs are redirected to stderr so stdout remains machine-readable.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def find_system_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "sfm_system":
            return p
    env = os.environ.get("SFM_SYSTEM_ROOT")
    if env:
        return Path(env)
    raise SystemExit(f"could not locate sfm_system root from {start}; set SFM_SYSTEM_ROOT")


SYSTEM_ROOT = find_system_root(Path(__file__).resolve())
LOC_ROOT = SYSTEM_ROOT / "定位"
DEPLOY_DIR = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
if not DEPLOY_DIR.exists():
    DEPLOY_DIR = LOC_ROOT / "source" / "sfm_glomap" / "deploy"
DEFAULT_BUNDLE = LOC_ROOT / "bundles" / "current_reloc_map_updated_v3.pt"
DEFAULT_MEGALOC = ""


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@contextlib.contextmanager
def redirect_native_stdout_to_stderr():
    """Route native fd-1 logs away from the JSON stdout pipe."""
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--megaloc-cache", default=DEFAULT_MEGALOC)
    ap.add_argument("--deploy-dir", default=str(DEPLOY_DIR))
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    json_fd = os.dup(sys.stdout.fileno())
    json_out = os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    sys.path.insert(0, str(Path(args.deploy_dir)))

    with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
        from path_follow_flight import CAM_720, production_config
        from production_xfeat_tracker import MegaLocLayer, ProductionXFeatTracker
        from reloc_localizer_xfeat import Camera, DEVICE, XFeatRelocMap
        import torch

        xmap = XFeatRelocMap.load(args.bundle)
        cache = Path(args.megaloc_cache) if args.megaloc_cache else None
        if cache is None:
            print("[live_worker] using bundle ref_global MegaLoc descriptors", file=sys.stderr, flush=True)
            megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)
        elif cache.exists():
            try:
                megaloc = MegaLocLayer.load_cache(cache, xmap.ref_names, input_size=322, device=DEVICE)
            except Exception as exc:
                print(f"[live_worker] MegaLoc cache incompatible ({exc}); using bundle ref_global", file=sys.stderr, flush=True)
                megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)
        else:
            print(f"[live_worker] MegaLoc cache missing: {cache}; using bundle ref_global", file=sys.stderr, flush=True)
            megaloc = MegaLocLayer(xmap.ref_global, input_size=322, device=DEVICE)

        cam = Camera(*CAM_720)
        tracker = ProductionXFeatTracker(
            xmap, megaloc, frame_source=lambda: None, query_cam=cam, cfg=production_config()
        )
        print(
            f"[live_worker] ready device={DEVICE} cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} "
            f"bundle={args.bundle}",
            file=sys.stderr,
            flush=True,
        )
        tracker.ensure_models()

    frame_size = int(args.width) * int(args.height) * 3
    seq = 0
    while True:
        raw = read_exact(sys.stdin.buffer, frame_size)
        if not raw:
            break
        if len(raw) != frame_size:
            print(f"[live_worker] partial frame: {len(raw)}/{frame_size}", file=sys.stderr, flush=True)
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((args.height, args.width, 3)).copy()
        t0 = time.perf_counter()
        try:
            with redirect_native_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
                pose = tracker.localize_frame(frame)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            info = dict(tracker.last_info)
            # A non-finite pose (NaN/inf) would serialize as the bare token `NaN`
            # (invalid JSON) and could steer the UI trajectory -> report it as no fix.
            pose_ok = pose is not None and bool(
                np.isfinite([pose.x, pose.y, pose.z, pose.yaw]).all())
            payload = {
                "seq": seq,
                "success": pose_ok,
                "wall_ms": wall_ms,
                "pose": None if not pose_ok else {
                    "x": float(pose.x),
                    "y": float(pose.y),
                    "z": float(pose.z),
                    "yaw_raw": float(pose.yaw),
                },
                "mode": info.get("mode"),
                "next_mode": info.get("next_mode"),
                "inliers": int(info.get("inliers", 0) or 0),
                "reproj_rms": info.get("reproj_rms"),
                "composite_stage": info.get("composite_stage"),
                "vpr_ms": info.get("vpr_ms"),
                "feature_ms": info.get("feature_ms"),
                "match_ms": info.get("match_ms"),
                "pnp_ms": info.get("pnp_ms"),
            }
        except Exception as exc:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "seq": seq,
                "success": False,
                "wall_ms": wall_ms,
                "error": repr(exc),
            }
            print(f"[live_worker] localization error: {exc!r}", file=sys.stderr, flush=True)
        json_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        json_out.flush()
        seq += 1
        if args.max_frames and seq >= args.max_frames:
            break


if __name__ == "__main__":
    main()
