#!/usr/bin/env python3
"""Benchmark the production XFeat tracker on images or a video as a 720p stream.

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
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from camera_intrinsics import load_scaled_simple_radial

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
from reloc_localizer_xfeat import Camera, XFeatRelocMap, DEVICE

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parents[1]
QUERY_DIR = ROOT.parents[2] / "建圖" / "inputs" / "目標場域影片" / "驗證" / "P0710071_frames"
BUNDLE = ROOT / "bundles" / "current_reloc_map_updated_v3.pt"
INTRINSICS = ROOT / "deploy_code" / "sfm_glomap_deploy" / "map_intrinsics.json"
IMAGES_FUSED = ROOT / "maps" / "base_images_fused"
CACHE_PATH = ROOT / "bundles" / "base_megaloc_cache_v3.npz"
OUT_DIR = ROOT / "outputs" / "production_stream_bench"
NEUFLOW_REPO = PACKAGE_ROOT / ".experiment_deps" / "neuflow_v2"
NEUFLOW_WEIGHTS = NEUFLOW_REPO / "neuflow_mixed.pth"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_rgb_resized(path: Path, width: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return resize_bgr_to_rgb(bgr, width)


def resize_bgr_to_rgb(
    bgr: np.ndarray, width: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    h0, w0 = bgr.shape[:2]
    if width > 0 and w0 != width:
        height = int(round(h0 * (width / w0)))
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, (w0, h0), (w, h)


def iter_video_frames(path: Path, width: int, stride: int, limit: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"video has invalid frame rate: {path}")
    selected = 0
    raw_index = 0
    try:
        while not limit or selected < limit:
            t0 = time.perf_counter()
            ok, bgr = cap.read()
            if not ok:
                break
            current = raw_index
            raw_index += 1
            if current % stride:
                continue
            rgb, orig_size, stream_size = resize_bgr_to_rgb(bgr, width)
            load_ms = (time.perf_counter() - t0) * 1000.0
            yield (
                f"frame_{current + 1:06d}.jpg",
                rgb,
                orig_size,
                stream_size,
                load_ms,
                float(current) / source_fps,
            )
            selected += 1
    finally:
        cap.release()


def video_frame_count(path: Path, stride: int, limit: int) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        raw = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    count = (raw + stride - 1) // stride
    return min(count, limit) if limit > 0 else count


def video_frame_rate(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"video has invalid frame rate: {path}")
    return fps


def load_query_camera(path: Path, orig_size: tuple[int, int],
                      stream_size: tuple[int, int]) -> Camera:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "K" not in data or "dist" not in data:
        params = load_scaled_simple_radial(path, orig_size)
        return scaled_simple_radial_camera(
            orig_size[0], orig_size[1], params, stream_size[0], stream_size[1])

    source_width = int(data.get("image_width", 0))
    source_height = int(data.get("image_height", 0))
    K = np.asarray(data["K"], dtype=float)
    dist = [float(value) for value in data["dist"]]
    if source_width <= 0 or source_height <= 0 or K.shape != (3, 3) or len(dist) > 8:
        raise ValueError(f"invalid FULL_OPENCV calibration in {path}")
    scale_x = stream_size[0] / source_width
    scale_y = stream_size[1] / source_height
    if not math.isclose(scale_x, scale_y, rel_tol=1e-3, abs_tol=1e-6):
        raise ValueError(
            f"cannot scale intrinsics across aspect ratios: "
            f"{source_width}x{source_height} -> {stream_size[0]}x{stream_size[1]}"
        )
    dist.extend([0.0] * (8 - len(dist)))
    params = [
        float(K[0, 0]) * scale_x,
        float(K[1, 1]) * scale_y,
        float(K[0, 2]) * scale_x,
        float(K[1, 2]) * scale_y,
        *dist,
    ]
    if not all(math.isfinite(value) for value in params):
        raise ValueError(f"non-finite FULL_OPENCV calibration in {path}")
    return Camera("FULL_OPENCV", stream_size[0], stream_size[1], params)


def intrinsics_check(cam) -> dict:
    """Cross-check the benchmark query camera against the deployed CAM_720."""
    out = {"benchmark_cam": {"model": cam.model, "width": cam.width,
                             "height": cam.height, "params": list(cam.params)}}
    try:
        from path_follow_flight import CAM_720
        model, width, height, params = CAM_720
        expected = np.asarray(params, dtype=float)
        actual = np.asarray(cam.params, dtype=float)
        out["production_720_cam"] = {
            "model": model, "width": width, "height": height, "params": list(params)}
        out["model_matches"] = cam.model == model
        out["size_matches"] = (cam.width, cam.height) == (width, height)
        out["params_max_abs_diff"] = (
            None if actual.shape != expected.shape
            else float(np.max(np.abs(actual - expected))))
        out["matches_production"] = bool(
            out["model_matches"] and out["size_matches"]
            and actual.shape == expected.shape
            and np.allclose(actual, expected, rtol=0.0, atol=1e-9))
        if not out["matches_production"]:
            print("WARNING: benchmark camera does not match deployed CAM_720", flush=True)
    except Exception as exc:                       # diagnostic failure must not kill the run
        out["warning"] = f"intrinsics check failed: {exc!r}"
    return out


def require_camera_match(cam) -> dict:
    """Return the camera audit or raise when production and benchmark differ."""
    result = intrinsics_check(cam)
    if not result.get("matches_production"):
        raise ValueError(
            "benchmark camera does not match deployed CAM_720; refusing the run"
        )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_identity_failures(
    baseline: dict,
    *,
    video_sha256: str | None,
    site_profile_sha256: str | None,
    bundle_sha256: str,
    localizer_profile_sha256: str | None,
    camera: dict[str, object],
) -> list[str]:
    """Return mismatches for the exact inputs used by a benchmark receipt."""
    failures: list[str] = []
    actual_values = {
        "video_sha256": video_sha256,
        "site_profile_sha256": site_profile_sha256,
        "bundle_sha256": bundle_sha256,
        "localizer_profile_sha256": localizer_profile_sha256,
    }
    for key, actual in actual_values.items():
        expected = baseline.get(key)
        if not isinstance(expected, str) or len(expected) != 64:
            failures.append(f"baseline is missing {key}")
        elif not isinstance(actual, str) or expected != actual:
            failures.append(f"{key} mismatch: expected={expected} actual={actual}")
    expected_camera = baseline.get("camera")
    if not isinstance(expected_camera, dict):
        failures.append("baseline is missing camera identity")
    else:
        for key in ("model", "width", "height", "params"):
            if expected_camera.get(key) != camera.get(key):
                failures.append(
                    f"camera {key} mismatch: expected={expected_camera.get(key)!r} "
                    f"actual={camera.get(key)!r}"
                )
    return failures


def pct(vals, p):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, float), p))


def stat(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"median": None, "p75": None, "p90": None, "p95": None, "mean": None}
    arr = np.asarray(vals, float)
    return {
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
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
    ap.add_argument("--video", default="", help="video input; overrides --query-dir")
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--bundle-sha256", default="",
                    help="trusted SHA-256 for a non-package bundle")
    ap.add_argument("--intrinsics", default=str(INTRINSICS))
    ap.add_argument(
        "--strict-camera",
        action="store_true",
        help="fail closed if benchmark intrinsics differ from deployed CAM_720",
    )
    ap.add_argument(
        "--allow-camera-mismatch",
        action="store_true",
        help="development-only opt-out from camera gates (never a production claim)",
    )
    ap.add_argument("--model", default="",
                    help="deprecated and ignored; retained for command compatibility")
    ap.add_argument("--image-root", default=str(IMAGES_FUSED))
    ap.add_argument("--megaloc-cache", default=str(CACHE_PATH))
    ap.add_argument("--megaloc-meta", default="",
                    help="required binding JSON when --megaloc-cache is a legacy NPY")
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
    ap.add_argument("--track-xfeat-topk", type=int, default=1700)
    ap.add_argument("--acquire-xfeat-topk", type=int, default=2048)
    ap.add_argument("--matcher-mode", choices=["nn_then_lg", "lighterglue", "nn"], default="nn_then_lg")
    ap.add_argument("--nn-min-score", type=float, default=0.85)
    ap.add_argument("--adaptive-first-topk", type=int, default=3)
    ap.add_argument("--adaptive-accept-inliers", type=int, default=100)
    ap.add_argument("--adaptive-accept-reproj", type=float, default=3.5)
    ap.add_argument("--max-corr-per-ref", type=int, default=0)
    ap.add_argument("--max-corr-total", type=int, default=0)
    ap.add_argument("--dedup-corr", action="store_true",
                    help="drop duplicate 2D/3D correspondences before PnP")
    ap.add_argument("--cache-seed-mode", choices=["full_ref", "inliers"], default="full_ref")
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
    ap.add_argument("--pnp-ransac-random-seed", type=int, default=0,
                    help="Fixed seed for reproducible offline comparisons; production default is -1")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--max-jump", type=float, default=2.0)
    ap.add_argument(
        "--flow-track", action="store_true",
        default=os.environ.get("SFM_FLOW_TRACK", "0") == "1",
    )
    ap.add_argument(
        "--flow-refresh", type=int,
        default=int(os.environ.get("SFM_FLOW_REFRESH", "1")),
    )
    ap.add_argument(
        "--flow-fb-px", type=float,
        default=float(os.environ.get("SFM_FLOW_FB_PX", "0.5")),
    )
    ap.add_argument("--flow-mnn-crosscheck", action="store_true")
    ap.add_argument("--flow-mnn-max-translation", type=float, default=0.10)
    ap.add_argument("--flow-mnn-max-yaw-deg", type=float, default=3.0)
    ap.add_argument("--flow-mnn-min-unique-inliers", type=int, default=80)
    ap.add_argument("--flow-mnn-max-reproj", type=float, default=3.5)
    ap.add_argument(
        "--neuflow-track", action="store_true",
        help="validation-only NeuFlow-v2 anchors between deep refresh frames",
    )
    ap.add_argument("--neuflow-repo", default=str(NEUFLOW_REPO))
    ap.add_argument("--neuflow-weights", default=str(NEUFLOW_WEIGHTS))
    ap.add_argument("--neuflow-width", type=int, default=512)
    ap.add_argument("--neuflow-height", type=int, default=288)
    ap.add_argument("--neuflow-refresh-interval", type=int, default=3)
    ap.add_argument("--neuflow-min-seed", type=int, default=100)
    ap.add_argument("--neuflow-min-track", type=int, default=100)
    ap.add_argument("--neuflow-min-inliers", type=int, default=50)
    ap.add_argument("--neuflow-min-inlier-ratio", type=float, default=0.25)
    ap.add_argument("--neuflow-max-reproj", type=float, default=4.0)
    ap.add_argument(
        "--projection-track", action="store_true",
        help="validation-only projected XFeat landmark matching before deep fallback",
    )
    ap.add_argument(
        "--track-landmarks", default="",
        help="validated XFeat TRACK landmark sidecar required by --projection-track",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--out-ply", default="")
    ap.add_argument(
        "--site-profile",
        default="",
        help="site profile to bind into a reproducibility receipt",
    )
    ap.add_argument(
        "--quality-baseline",
        default="",
        help="baseline JSON whose video/site/bundle/profile identity must match",
    )
    ap.add_argument("--lg-onnx-track-model", default="",
                    help="Experimental static ONNX matcher for TRACK query top-K")
    ap.add_argument("--lg-onnx-acquire-model", default="",
                    help="Experimental static ONNX matcher for BOOT/LOST query top-K")
    ap.add_argument(
        "--lg-onnx-provider", choices=["cuda", "tensorrt", "tensorrt_fp32"], default="tensorrt"
    )
    ap.add_argument("--lg-onnx-ref-topk", type=int, default=2048)
    args = ap.parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    if args.stride <= 0:
        ap.error("--stride must be positive")
    if sum(bool(value) for value in (
        args.flow_track, args.neuflow_track, args.projection_track,
    )) > 1:
        ap.error("--flow-track, --neuflow-track, and --projection-track are alternatives")
    if args.projection_track and not args.track_landmarks:
        ap.error("--projection-track requires --track-landmarks")
    qdir = Path(args.query_dir)
    video = Path(args.video) if args.video else None
    if video is not None:
        frame_total = video_frame_count(video, args.stride, args.limit)
        source_fps = video_frame_rate(video)
        first = next(iter_video_frames(video, args.resize_width, args.stride, 1), None)
        if first is None:
            raise SystemExit(f"no decodable video frames found in {video}")
        first_rgb, orig_size, stream_size = first[1], first[2], first[3]
        source_label = f"video={video} source_fps={source_fps:.6f}"
    else:
        source_fps = None
        frames = sorted(qdir.glob("*.jpg"))
        if args.stride > 1:
            frames = frames[::args.stride]
        if args.limit and args.limit > 0:
            frames = frames[:args.limit]
        if not frames:
            raise SystemExit(f"no jpg frames found under {qdir}")
        frame_total = len(frames)
        first_rgb, orig_size, stream_size = load_rgb_resized(frames[0], args.resize_width)
        source_label = f"query_dir={qdir}"
    cam = load_query_camera(Path(args.intrinsics), orig_size, stream_size)
    camera_audit = intrinsics_check(cam)
    if (args.strict_camera or args.quality_baseline) and not args.allow_camera_mismatch:
        try:
            require_camera_match(cam)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    site_profile_sha256 = None
    localizer_profile_sha256 = None
    if args.site_profile:
        site_profile_path = Path(args.site_profile).expanduser().resolve()
        try:
            site_raw = json.loads(site_profile_path.read_text(encoding="utf-8"))
            profile_value = site_raw.get("localizer_profile")
            if not isinstance(profile_value, str) or not profile_value:
                raise ValueError("site profile has no localizer_profile")
            localizer_profile_path = (site_profile_path.parent / profile_value).resolve()
            site_profile_sha256 = sha256_file(site_profile_path)
            localizer_profile_sha256 = sha256_file(localizer_profile_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot bind site profile: {exc}") from exc

    bundle_sha256 = sha256_file(Path(args.bundle).expanduser().resolve())
    camera_identity = {
        "model": cam.model,
        "width": cam.width,
        "height": cam.height,
        "params": [float(value) for value in cam.params],
    }
    baseline_identity = None
    if args.quality_baseline:
        if video is None or not args.site_profile:
            raise SystemExit(
                "--quality-baseline requires --video and --site-profile"
            )
        baseline_path = Path(args.quality_baseline).expanduser().resolve()
        try:
            baseline_identity = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read quality baseline: {exc}") from exc
        failures = baseline_identity_failures(
            baseline_identity,
            video_sha256=sha256_file(video),
            site_profile_sha256=site_profile_sha256,
            bundle_sha256=bundle_sha256,
            localizer_profile_sha256=localizer_profile_sha256,
            camera=camera_identity,
        )
        if failures:
            raise SystemExit("quality baseline identity mismatch: " + "; ".join(failures))

    print(f"device={DEVICE} cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    print(f"frames={frame_total} {source_label}")
    print(f"resize {orig_size[0]}x{orig_size[1]} -> {stream_size[0]}x{stream_size[1]}")
    print(f"query camera {cam.model} {cam.width}x{cam.height} params={cam.params}")

    projection_types = None
    if args.projection_track:
        from projection_guided_tracker import ProjectionGuidedTracker, TrackLandmarkSidecar

        landmark_path = Path(args.track_landmarks)
        if not landmark_path.is_file():
            raise SystemExit(f"TRACK landmark sidecar not found: {landmark_path}")
        sidecar_bundle_sha256, _sidecar_names_sha256 = (
            TrackLandmarkSidecar.read_binding(landmark_path))
        if args.bundle_sha256 and args.bundle_sha256 != sidecar_bundle_sha256:
            ap.error("--bundle-sha256 does not match --track-landmarks binding")
        projection_types = (
            ProjectionGuidedTracker,
            TrackLandmarkSidecar,
            landmark_path,
            sidecar_bundle_sha256,
        )

    print(f"loading bundle {args.bundle}", flush=True)
    expected_bundle_sha256 = (
        projection_types[3] if projection_types is not None
        else args.bundle_sha256 or None
    )
    xmap = XFeatRelocMap.load(args.bundle, expected_bundle_sha256)
    print("bundle meta", xmap.meta, flush=True)

    cache = Path(args.megaloc_cache)
    meta = Path(args.megaloc_meta) if args.megaloc_meta else cache.with_suffix(".json")
    if cache.exists():
        print(f"loading MegaLoc cache {cache}", flush=True)
        try:
            megaloc = MegaLocLayer.load_cache(
                cache, xmap.ref_names, input_size=322, device=DEVICE,
                meta_path=meta if args.megaloc_meta else None,
            )
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
        dedup_corr=args.dedup_corr,
        temporal_cache_seed_mode=args.cache_seed_mode,
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
        pnp_ransac_random_seed=args.pnp_ransac_random_seed,
        max_jump=args.max_jump,
        flow_enabled=args.flow_track,
        flow_mnn_crosscheck=args.flow_mnn_crosscheck,
        flow_mnn_max_translation=args.flow_mnn_max_translation,
        flow_mnn_max_yaw_deg=args.flow_mnn_max_yaw_deg,
        flow_mnn_min_unique_inliers=args.flow_mnn_min_unique_inliers,
        flow_mnn_max_reproj=args.flow_mnn_max_reproj,
    )
    os.environ["SFM_FLOW_REFRESH"] = str(args.flow_refresh)
    os.environ["SFM_FLOW_FB_PX"] = str(args.flow_fb_px)
    neuflow_metadata = None
    projection_metadata = None
    tracker_variant = (
        f"deep_{args.matcher_mode}_{args.adaptive_first_topk}_{args.local_topk}")
    if args.neuflow_track:
        from neuflow_refresh_experiment import NeuFlowRefreshTracker, NeuFlowV2Backend

        print(
            f"loading NeuFlow-v2 {args.neuflow_width}x{args.neuflow_height} "
            f"refresh={args.neuflow_refresh_interval}",
            flush=True,
        )
        neuflow_backend = NeuFlowV2Backend(
            repo=Path(args.neuflow_repo),
            weights=Path(args.neuflow_weights),
            width=args.neuflow_width,
            height=args.neuflow_height,
        )
        neuflow_metadata = neuflow_backend.metadata()
        tracker = NeuFlowRefreshTracker(
            xmap,
            megaloc,
            frame_source=lambda: None,
            query_cam=cam,
            cfg=cfg,
            neuflow_backend=neuflow_backend,
            refresh_interval=args.neuflow_refresh_interval,
            min_seed=args.neuflow_min_seed,
            min_track=args.neuflow_min_track,
            min_inliers=args.neuflow_min_inliers,
            min_inlier_ratio=args.neuflow_min_inlier_ratio,
            max_reproj=args.neuflow_max_reproj,
        )
        tracker_variant = (
            f"neuflow_v2_r{args.neuflow_refresh_interval}"
            f"_s{args.neuflow_min_seed}_t{args.neuflow_min_track}")
    elif args.projection_track:
        ProjectionGuidedTracker, TrackLandmarkSidecar, landmark_path, bundle_sha256 = (
            projection_types)
        landmark_sidecar = TrackLandmarkSidecar.load(
            landmark_path,
            xmap.ref_names,
            verified_bundle_sha256=bundle_sha256,
        )
        tracker = ProjectionGuidedTracker(
            xmap,
            megaloc,
            frame_source=lambda: None,
            query_cam=cam,
            cfg=cfg,
            landmark_sidecar=landmark_sidecar,
            search_radii=(15.0, 25.0, 40.0),
            min_score=0.60,
            ratio=0.95,
        )
        projection_metadata = {
            "landmarks": str(landmark_path),
            "bundle_sha256": bundle_sha256,
            "search_radii_px": [15.0, 25.0, 40.0],
            "min_score": 0.60,
            "ratio": 0.95,
        }
        tracker_variant = "projection_guided_r15_25_40_s060_ratio095"
    else:
        tracker = ProductionXFeatTracker(
            xmap, megaloc, frame_source=lambda: None, query_cam=cam, cfg=cfg)
    print("loading models ...", flush=True)
    tracker.ensure_models()
    if args.lg_onnx_track_model or args.lg_onnx_acquire_model:
        from benchmark_xfeat_lg_onnx import OnnxXFeatMatcher
        xfeat = tracker.ensure_xfeat()
        models = {}
        if args.lg_onnx_track_model:
            models[args.track_xfeat_topk] = Path(args.lg_onnx_track_model)
        if args.lg_onnx_acquire_model:
            models[args.acquire_xfeat_topk] = Path(args.lg_onnx_acquire_model)
        backend = OnnxXFeatMatcher(
            models,
            provider=args.lg_onnx_provider,
            reference_topk=args.lg_onnx_ref_topk,
        )
        backend.fallback = xfeat.match_lighterglue_indices
        xfeat.match_lighterglue_indices = backend
        print(
            f"experimental ONNX matcher provider={args.lg_onnx_provider} "
            f"query_shapes={sorted(models)} "
            f"ref_topk={args.lg_onnx_ref_topk}",
            flush=True,
        )
    sync()

    rows = []
    t_all0 = time.perf_counter()
    if video is not None:
        frame_iter = iter_video_frames(
            video, args.resize_width, args.stride, args.limit)
    else:
        def image_frames():
            for path in frames:
                t_load0 = time.perf_counter()
                frame, frame_orig, frame_stream = load_rgb_resized(path, args.resize_width)
                yield (
                    path.name,
                    frame,
                    frame_orig,
                    frame_stream,
                    (time.perf_counter() - t_load0) * 1000.0,
                    time.monotonic(),
                )
        frame_iter = image_frames()
    for i, (frame_name, frame, _orig, _stream, load_ms, capture_stamp) in enumerate(frame_iter, 1):
        sync(); t0 = time.perf_counter()
        pose = tracker.localize_frame(frame, capture_stamp=capture_stamp)
        sync(); wall_ms = (time.perf_counter() - t0) * 1000.0
        info = dict(tracker.last_info)
        row = {
            "idx": i - 1,
            "frame": frame_name,
            "success": pose is not None,
            "wall_ms": wall_ms,
            "load_ms": load_ms,
            "capture_stamp": capture_stamp,
            "pose": None if pose is None else {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw},
            **info,
        }
        rows.append(row)
        if i <= 5 or i % 25 == 0 or i == frame_total:
            print(
                f"{i:4d}/{frame_total} {frame_name} mode={row.get('mode')} -> {row.get('next_mode')} "
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
    flow_rows = [r for r in rows if r.get("mode") == "FLOW_TRACK"]
    flow_fallback_rows = [r for r in rows if r.get("flow_fallback")]
    flow_mnn_rows = [
        r for r in rows
        if r.get("flow_mnn_inliers") is not None
        or r.get("flow_stage") == "mnn_crosscheck"
        or r.get("flow_mnn_reason") is not None
    ]
    flow_mnn_failure_reasons = {}
    for row in flow_fallback_rows:
        reason = row.get("flow_mnn_reason")
        if reason:
            flow_mnn_failure_reasons[reason] = (
                flow_mnn_failure_reasons.get(reason, 0) + 1)
    neuflow_rows = [r for r in rows if r.get("mode") == "NEUFLOW_TRACK"]
    neuflow_fallback_rows = [r for r in rows if r.get("neuflow_fallback")]
    neuflow_fallback_reasons = count_by(neuflow_fallback_rows, "neuflow_stage")
    projection_rows = [
        r for r in rows
        if r.get("composite_stage") == "projection_guided"
        and not r.get("projection_fallback")
    ]
    projection_fallback_rows = [r for r in rows if r.get("projection_fallback")]
    projection_fallback_reasons = count_by(
        projection_fallback_rows, "projection_reason")
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
        "lg_fallback": int(sum(by_stage.get(stage, 0) for stage in (
            "lg_adaptive_after_nn", "lg_full_after_nn",
            "lg_adaptive_forced", "lg_full_forced", "lg_forced",
        ))),
        "pnp_failures": sum(1 for r in rows if r.get("pnp_failed")),
        "jump_rejections": sum(1 for r in rows if r.get("jump_rejected")),
        "xfeat_extracts_total": sum(int(r.get("xfeat_extract_count") or 0) for r in rows),
        "lg_calls_total": sum(int(r.get("lg_call_count") or 0) for r in rows),
        "lg_reuse_total": sum(int(r.get("lg_reuse_count") or 0) for r in rows),
        "nn_calls_total": sum(int(r.get("nn_call_count") or 0) for r in rows),
        "megaloc_calls_total": sum(int(r.get("megaloc_call_count") or 0) for r in rows),
        "xfeat_extracts_per_frame": sum(int(r.get("xfeat_extract_count") or 0) for r in rows) / max(n, 1),
        "lg_calls_per_frame": sum(int(r.get("lg_call_count") or 0) for r in rows) / max(n, 1),
        "megaloc_calls_per_frame": sum(int(r.get("megaloc_call_count") or 0) for r in rows) / max(n, 1),
        "flow_accept_frames": len(flow_rows),
        "flow_fallback_frames": len(flow_fallback_rows),
        "flow_mnn_attempted_frames": len(flow_mnn_rows),
        "flow_mnn_failure_reasons": flow_mnn_failure_reasons,
        "flow_accept_wall_ms": stat([r.get("wall_ms") for r in flow_rows]),
        "flow_fallback_wall_ms": stat([r.get("wall_ms") for r in flow_fallback_rows]),
        "flow_mnn_translation_delta": stat([
            r.get("flow_mnn_translation_delta") for r in flow_rows]),
        "flow_mnn_yaw_delta_deg": stat([
            r.get("flow_mnn_yaw_delta_deg") for r in flow_rows]),
        "neuflow_accept_frames": len(neuflow_rows),
        "neuflow_fallback_frames": len(neuflow_fallback_rows),
        "neuflow_fallback_reasons": neuflow_fallback_reasons,
        "neuflow_accept_wall_ms": stat([
            r.get("wall_ms") for r in neuflow_rows]),
        "neuflow_flow_ms": stat([
            r.get("neuflow_flow_ms") for r in neuflow_rows]),
        "neuflow_gpu_ms": stat([
            r.get("neuflow_gpu_ms") for r in neuflow_rows]),
        "neuflow_pnp_ms": stat([
            r.get("pnp_ms") for r in neuflow_rows]),
        "projection_accept_frames": len(projection_rows),
        "projection_fallback_frames": len(projection_fallback_rows),
        "projection_fallback_reasons": projection_fallback_reasons,
        "projection_accept_wall_ms": stat([
            r.get("wall_ms") for r in projection_rows]),
        "projection_match_ms": stat([
            r.get("match_ms") for r in projection_rows]),
        "projection_pnp_ms": stat([
            r.get("pnp_ms") for r in projection_rows]),
        "corr3d_pre_dedup": stat([r.get("corr3d_pre_dedup") for r in rows]),
        "inlier_coverage": stat([r.get("inlier_coverage") for r in rows]),
        "quality_score": stat([r.get("quality_score") for r in rows]),
        "load_ms": stat([r.get("load_ms") for r in rows]),
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
        "runtime_env": {
            "SFM_FLOW_TRACK": "1" if args.flow_track else "0",
            "SFM_FLOW_REFRESH": str(args.flow_refresh),
            "SFM_FLOW_FB_PX": str(args.flow_fb_px),
            "SFM_FLOW_MNN_CROSSCHECK": "1" if args.flow_mnn_crosscheck else "0",
            "SFM_FLOW_QUAL_INLIERS": os.environ.get("SFM_FLOW_QUAL_INLIERS", "50"),
            "SFM_FLOW_QUAL_RATIO": os.environ.get("SFM_FLOW_QUAL_RATIO", "0.25"),
            "SFM_FLOW_QUAL_REPROJ": os.environ.get("SFM_FLOW_QUAL_REPROJ", "4.0"),
            "SFM_FLOW_QUAL_NTRACK": os.environ.get("SFM_FLOW_QUAL_NTRACK", "100"),
            "SFM_MEGALOC_FP16": os.environ.get("SFM_MEGALOC_FP16", "0"),
        },
        "neuflow": neuflow_metadata,
        "projection": projection_metadata,
        "tracker_variant": tracker_variant,
        "source_fps": source_fps,
        "device": DEVICE,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bundle_meta": xmap.meta,
        "query_camera": {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params},
        "intrinsics_check": camera_audit,
        "input_identity": {
            "video_sha256": None if video is None else sha256_file(video),
            "site_profile_sha256": site_profile_sha256,
            "bundle_sha256": bundle_sha256,
            "localizer_profile_sha256": localizer_profile_sha256,
            "camera": camera_identity,
            "baseline": None if not args.quality_baseline else str(Path(args.quality_baseline).expanduser().resolve()),
        },
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
