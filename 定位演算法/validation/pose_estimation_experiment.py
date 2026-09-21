#!/usr/bin/env python3
"""Offline, paired pose experiments. Never imports an aircraft SDK.

Capture the production frontend once; compare estimators on identical raw
correspondences. This is a conditional estimator experiment, not a closed-loop
frontend or flight acceptance test. Reprojection holdout is not position GT.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for relative in (
    "定位演算法/flight_control",
    "定位演算法/deploy_code/sfm_glomap_deploy",
    "定位演算法/deploy_code/sfm_direct_deploy",
    "控制介面程式",
):
    sys.path.insert(0, str(ROOT / relative))

from two_rate_tracker import RelocResult  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def capture_stamp(pts, previous, ordinal, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video FPS must be finite and positive")
    if math.isfinite(pts) and pts > previous:
        return pts, False
    return max(previous + 1.0 / fps, ordinal / fps), True


def validate_capture(rows, offsets, xy, xyz, ids, K):
    if (
        offsets.ndim != 1
        or len(offsets) != len(rows) + 1
        or not np.issubdtype(offsets.dtype, np.integer)
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] != len(ids)
        or xy.shape != (len(ids), 2)
        or xyz.shape != (len(ids), 3)
        or ids.ndim != 1
    ):
        raise ValueError("capture offsets and observation shapes do not agree")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("invalid capture intrinsics")
    if not np.isfinite(xy).all() or not np.isfinite(xyz).all():
        raise ValueError("non-finite captured observations")
    stamps = np.asarray([row["stamp"] for row in rows], dtype=float)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError("capture timestamps must be finite and strictly increasing")


class ScheduledRelocalizer:
    """Measure GPU jobs inline and deliver after their measured video-time lag.

    No CPU/GPU concurrency is claimed. A busy job rejects submissions; the
    frontend retries on subsequent frames. All estimator variants share this
    one captured schedule, so scheduling cannot favor one estimator.
    """

    def __init__(self, provider, synchronize):
        self.provider = provider
        self.synchronize = synchronize
        self.stamp = 0.0
        self.pending = None
        self.jobs = []
        self.frame_reloc_ms = 0.0

    @property
    def queued(self):
        return False

    @property
    def busy(self):
        return self.pending is not None

    def start(self):
        pass

    def close(self):
        self.pending = None

    reset = close

    def submit(self, gray, ordinal, capture_stamp, source_epoch=0):
        if self.pending is not None:
            return False
        self.synchronize()
        started = time.perf_counter()
        fix = self.provider.localize_array(gray)
        self.synchronize()
        elapsed = time.perf_counter() - started
        self.frame_reloc_ms += elapsed * 1000.0
        capture_stamp = float(capture_stamp)
        source_epoch = int(source_epoch)
        self.pending = (capture_stamp + elapsed, fix, int(ordinal), capture_stamp, source_epoch)
        job = {
            "ordinal": int(ordinal),
            "capture_stamp": capture_stamp,
            "source_epoch": source_epoch,
            "runtime_ms": elapsed * 1000.0,
            "status": fix.status,
            "inliers": int(fix.inliers),
        }
        stage_ms = getattr(fix, "stage_ms", None)
        if isinstance(stage_ms, dict):
            job["stage_ms"] = {
                str(key): float(value)
                for key, value in stage_ms.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
        self.jobs.append(job)
        return True

    def poll(self):
        if self.pending is None or self.stamp < self.pending[0]:
            return None
        _, fix, ordinal, capture_stamp, source_epoch = self.pending
        self.pending = None
        return RelocResult(
            fix=fix,
            ordinal=ordinal,
            capture_stamp=capture_stamp,
            source_epoch=source_epoch,
        )


def capture(args):
    import torch
    from production_localizer_factory import build_production_localizer
    from real_path_follow_controller import load_map_frame

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    site_path = Path(args.site).resolve()
    site = json.loads(site_path.read_text())
    asset = lambda value: (site_path.parent / value).resolve()
    camera = site["query_camera"]
    built = build_production_localizer(
        backend="direct",
        bundle=asset(site["assets"]["localization_bundle"]),
        bundle_sha256=site["asset_sha256"]["localization_bundle"],
        production_profile=asset(site["localizer_profile"]),
        production_profile_sha256=site["asset_sha256"]["localizer_profile"],
        camera_tuple=(camera["model"], camera["width"], camera["height"], camera["params"]),
        frame_source=lambda: None,
        map_frame=load_map_frame(asset(site["map_align"])),
    )
    tracker = built.tracker.trk
    # Load/warm models without starting the production background worker.
    started = time.perf_counter()
    tracker.provider.ensure_models()
    torch.cuda.synchronize()
    worker = ScheduledRelocalizer(tracker.provider, torch.cuda.synchronize)
    tracker._worker = worker
    warmup_s = time.perf_counter() - started
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"cannot decode {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    capture_stamp(0.0, -1.0, 0, fps)  # Validate before decoding or dividing by FPS.
    metadata = {
        "video": str(Path(args.video).resolve()),
        "video_sha256": digest(args.video),
        "fps": fps,
        "declared_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "source_size": [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ],
        "profile_sha256": site["asset_sha256"]["localizer_profile"],
        "bundle_sha256": site["asset_sha256"]["localization_bundle"],
        "K": tracker._K.tolist(),
        "image_size": [tracker._w, tracker._h],
        "warmup_s": warmup_s,
        "gpu": torch.cuda.get_device_name(),
        "versions": {"torch": torch.__version__, "opencv": cv2.__version__},
        "telemetry": None,
        "independent_ground_truth": None,
        "schedule": "inline measured GPU latency, delivered on video timestamps; not concurrent end-to-end timing",
    }
    batches, rows, offsets = [], [], [0]
    original = tracker._step_absolute_pose
    current = {}
    epoch = 0

    def record_pnp(reseeded, stamp):
        nonlocal epoch
        if reseeded:
            epoch += 1
        current.update(
            xy=tracker._live_xy.copy(),
            xyz=tracker._live_xyz.copy(),
            ids=tracker._live_ids.copy(),
            epoch=epoch,
        )
        return original(reseeded, stamp)

    tracker._step_absolute_pose = record_pnp
    previous_stamp = -1.0
    timestamp_fallbacks = 0
    try:
        while not args.max_frames or len(rows) < args.max_frames:
            started = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter() - started) * 1000.0
            if not ok:
                break
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            stamp, fallback = capture_stamp(pts, previous_stamp, len(rows), fps)
            timestamp_fallbacks += int(fallback)
            previous_stamp = stamp
            worker.stamp, worker.frame_reloc_ms = stamp, 0.0
            started = time.perf_counter()
            info = tracker.step(frame, stamp)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            batches.append((current["xy"], current["xyz"], current["ids"]))
            offsets.append(offsets[-1] + len(current["ids"]))
            pose = info["cam_from_world"]
            rows.append(
                {
                    "ordinal": len(rows),
                    "stamp": stamp,
                    "epoch": current["epoch"],
                    "status": info["status"],
                    "map_inliers": info["map_inliers"],
                    "inliers": info["inliers"],
                    "reseeded": bool(info["klt_reseeded"]),
                    "pose": None if pose is None else pose.tolist(),
                    "decode_ms": decode_ms,
                    "frontend_wall_ms": elapsed_ms,
                    "frontend_without_inline_reloc_ms": elapsed_ms - worker.frame_reloc_ms,
                    "pnp_ms": info["pnp_ms"],
                    "track_ms": info["track_ms"],
                }
            )
            if len(rows) % 240 == 0:
                print(
                    json.dumps(
                        {
                            "frames": len(rows),
                            "status": dict(Counter(r["status"] for r in rows)),
                            "reloc_jobs": len(worker.jobs),
                        }
                    ),
                    flush=True,
                )
    finally:
        cap.release()
        tracker.close()
    np.savez_compressed(
        out / "observations.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        xy=np.concatenate([b[0] for b in batches]),
        xyz=np.concatenate([b[1] for b in batches]),
        ids=np.concatenate([b[2] for b in batches]),
    )
    (out / "frames.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    metadata.update(
        frames=len(rows),
        timestamp_fallbacks=timestamp_fallbacks,
        status_counts=dict(Counter(r["status"] for r in rows)),
        reloc_jobs=worker.jobs,
    )
    write_json(out / "capture.json", metadata)
    print(
        json.dumps({"out": str(out), "frames": len(rows), "status": metadata["status_counts"]}),
        flush=True,
    )


def projection_errors(pose, xyz, xy, K):
    camera_points = xyz @ pose[:, :3].T + pose[:, 3]
    depth = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (depth > 1e-8)
    errors = np.full(len(xyz), 1e6, dtype=float)
    projected = camera_points[valid] @ K.T
    errors[valid] = np.linalg.norm(projected[:, :2] / projected[:, 2:] - xy[valid], axis=1)
    return errors


def quantiles(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return {
        "n": len(a),
        **(
            {k: float(np.percentile(a, p)) for k, p in (("p50", 50), ("p95", 95))}
            if len(a)
            else {"p50": None, "p95": None}
        ),
    }


def _motion_metrics(previous, name, row, pose, center):
    from scipy.spatial.transform import Rotation

    velocity = acceleration = turn_rate = None
    old = previous.get(name)
    if old is not None and center is not None and old["epoch"] == row["epoch"]:
        dt = row["stamp"] - old["stamp"]
        if 0 < dt <= 0.5:
            velocity = (center - old["center"]) / dt
            turn_rate = float(
                np.linalg.norm(Rotation.from_matrix(pose[:, :3] @ old["pose"][:, :3].T).as_rotvec())
                / dt
            )
            if old["velocity"] is not None:
                acceleration = float(np.linalg.norm(velocity - old["velocity"]) / dt)
    if center is not None:
        previous[name] = {
            "stamp": row["stamp"],
            "epoch": row["epoch"],
            "center": center,
            "pose": pose,
            "velocity": velocity,
        }
    else:
        previous.pop(name, None)
    return velocity, acceleration, turn_rate


def evaluate(args):
    import pycolmap
    from pose_filter_experiment import VisualPoseESKF
    from pose_window_experiment import PoseWindowOptimizer

    source, out = Path(args.capture), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "capture.json").read_text())
    rows = [json.loads(line) for line in (source / "frames.jsonl").read_text().splitlines()]
    data = np.load(source / "observations.npz", allow_pickle=False)
    arrays = (data["xy"], data["xyz"], data["ids"])
    offsets = data["offsets"]
    K = np.asarray(meta["K"])
    validate_capture(rows, offsets, *arrays, K)
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=meta["image_size"][0],
        height=meta["image_size"][1],
        params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
    )
    estimation = pycolmap.AbsolutePoseEstimationOptions()
    estimation.ransac.max_error = 4.0
    estimation.ransac.min_num_trials = 10
    estimation.ransac.max_num_trials = 100
    estimation.ransac.confidence = 0.999
    estimation.ransac.random_seed = 0
    methods = (
        "refine0",
        "refine2",
        "refine3",
        "refine5",
        "refine10",
        "visual_esekf",
        "window5",
        "window10",
    )
    filters = {
        "visual_esekf": VisualPoseESKF(),
        "window5": PoseWindowOptimizer(window_size=5),
        "window10": PoseWindowOptimizer(window_size=10),
    }
    records = []
    previous = {}
    for index, row in enumerate(rows):
        if args.max_frames and index >= args.max_frames:
            break
        a, b = offsets[index : index + 2]
        # Materialize once outside timing; NPZ members otherwise decompress on every lookup.
        xy, xyz, ids = arrays[0][a:b], arrays[1][a:b], arrays[2][a:b]
        held = np.remainder(ids, 5) == 0
        train = ~held
        frame_results = {}
        started = time.perf_counter()
        seed = (
            pycolmap.estimate_absolute_pose(xy[train], xyz[train], camera, estimation)
            if train.sum() >= 12
            else None
        )
        ransac_ms = (time.perf_counter() - started) * 1000.0
        if seed is not None and seed["num_inliers"] < 12:
            seed = None
        seed_pose = None if seed is None else np.asarray(seed["cam_from_world"].matrix()).copy()
        seed_mask = (
            np.zeros(train.sum(), dtype=bool)
            if seed is None
            else np.asarray(seed["inlier_mask"], dtype=bool)
        )
        iteration_order = (0, 2, 3, 5, 10)
        order_start = index % len(iteration_order)
        for iterations in iteration_order[order_start:] + iteration_order[:order_start]:
            answer = None
            started = time.perf_counter()
            if seed is not None and iterations:
                refinement = pycolmap.AbsolutePoseRefinementOptions()
                refinement.max_num_iterations = iterations
                answer = pycolmap.refine_absolute_pose(
                    pycolmap.Rigid3d(seed_pose.copy()),
                    xy[train],
                    xyz[train],
                    seed_mask,
                    camera,
                    refinement,
                )
            elapsed = (time.perf_counter() - started) * 1000.0
            pose = seed_pose.copy() if seed is not None and iterations == 0 else None
            if answer is not None:
                pose = np.asarray(answer["cam_from_world"].matrix()).copy()
            frame_results[f"refine{iterations}"] = {"pose": pose, "ms": elapsed}
            if iterations == 2:
                baseline = pose
                inliers = np.zeros(train.sum(), dtype=bool) if pose is None else seed_mask
        accepted_xyz = xyz[train][inliers]
        depth_scale = 1.0
        if baseline is not None and len(accepted_xyz):
            depth_scale = float(
                np.median(np.linalg.norm(accepted_xyz @ baseline[:, :3].T + baseline[:, 3], axis=1))
            )
        position_sigma = max(1e-5, 2.0 * depth_scale / K[0, 0])
        started = time.perf_counter()
        filtered = filters["visual_esekf"].update(
            row["stamp"],
            baseline,
            epoch=row["epoch"],
            position_sigma=position_sigma,
            rotation_sigma=2.0 / K[0, 0],
        )
        frame_results["visual_esekf"] = {**filtered, "ms": (time.perf_counter() - started) * 1000.0}
        frame = {
            "stamp": row["stamp"],
            "epoch": row["epoch"],
            "pose": baseline,
            "ids": ids[train][inliers],
            "xy": xy[train][inliers],
            "xyz": accepted_xyz,
            "K": K,
        }
        for name in ("window5", "window10"):
            started = time.perf_counter()
            result = filters[name].update(frame)
            frame_results[name] = {**result, "ms": (time.perf_counter() - started) * 1000.0}
        for name in methods:
            result = frame_results[name]
            pose = result.pop("pose")
            errors = (
                None
                if pose is None or not held.any()
                else projection_errors(pose, xyz[held], xy[held], K)
            )
            center = None if pose is None else -pose[:, :3].T @ pose[:, 3]
            velocity, acceleration, turn_rate = _motion_metrics(previous, name, row, pose, center)
            records.append(
                {
                    "frame": index,
                    "stamp": row["stamp"],
                    "epoch": row["epoch"],
                    "method": name,
                    "ok": pose is not None,
                    "pose": None if pose is None else pose.tolist(),
                    "train_count": int(train.sum()),
                    "upstream_pose_available": baseline is not None,
                    "holdout_count": int(held.sum()),
                    "heldout_median_px": None if errors is None else float(np.median(errors)),
                    "heldout_p90_px": None if errors is None else float(np.percentile(errors, 90)),
                    "heldout_within4px": None if errors is None else float(np.mean(errors <= 4.0)),
                    "speed_u_s": None if velocity is None else float(np.linalg.norm(velocity)),
                    "acceleration_u_s2": acceleration,
                    "rotation_rate_rad_s": turn_rate,
                    "ransac_ms": ransac_ms,
                    "estimator_total_ms": ransac_ms
                    + result["ms"]
                    + (0.0 if name.startswith("refine") else frame_results["refine2"]["ms"]),
                    **{
                        k: v
                        for k, v in result.items()
                        if isinstance(v, (bool, int, float, str)) or v is None
                    },
                }
            )
        if index % 240 == 0:
            print(json.dumps({"evaluated": index + 1, "total": len(rows)}), flush=True)
    summaries = {}
    for name in methods:
        entries = [r for r in records if r["method"] == name]
        # Exclude no-work frames and the first ten eligible calls from steady timing.
        timed = [r for r in entries if r["train_count"] >= 12][10:]
        summaries[name] = {
            "frames": len(entries),
            "valid_poses": sum(r["ok"] for r in entries),
            "measured_poses": sum(r["ok"] and not r.get("predicted_only", False) for r in entries),
            "upstream_pose_available": sum(r["upstream_pose_available"] for r in entries),
            "predicted_only": sum(bool(r.get("predicted_only")) for r in entries),
            "optimized_frames": sum(bool(r.get("optimized")) for r in entries),
            "heldout_median_px": quantiles([r["heldout_median_px"] for r in entries]),
            "heldout_p90_px": quantiles([r["heldout_p90_px"] for r in entries]),
            "stage_ms": quantiles([r["ms"] for r in timed]),
            "ransac_ms": quantiles([r["ransac_ms"] for r in timed]),
            "estimator_total_ms": quantiles([r["estimator_total_ms"] for r in timed]),
            "optimized_stage_ms": quantiles([r["ms"] for r in timed if r.get("optimized")]),
            "acceleration_u_s2": quantiles([r["acceleration_u_s2"] for r in entries]),
        }
    (out / "results.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    )
    write_json(
        out / "summary.json",
        {
            "capture": str(source.resolve()),
            "video_sha256": meta["video_sha256"],
            "paired_input": True,
            "split": "ids % 5 == 0 held out from every estimator",
            "refinement_comparison": "one shared RANSAC pose and mask per frame; each refinement starts from a fresh copy",
            "source_sha256": {
                name: digest(Path(__file__).with_name(name))
                for name in (
                    "pose_estimation_experiment.py",
                    "pose_filter_experiment.py",
                    "pose_window_experiment.py",
                )
            },
            "versions": {"numpy": np.__version__, "pycolmap": pycolmap.__version__},
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "independent_ground_truth": False,
            "methods": summaries,
        },
    )
    print(json.dumps(summaries), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--video", type=Path, required=True)
    cap.add_argument("--site", type=Path, default=ROOT / "地圖檔/場域/river_site/site_profile.json")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--max-frames", type=int, default=0)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--capture", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)
    return capture(args) if args.command == "capture" else evaluate(args)


if __name__ == "__main__":
    main()
