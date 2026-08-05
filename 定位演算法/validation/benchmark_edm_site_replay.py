#!/usr/bin/env python3
"""Replay one video through the exact EDM site profile used by the operator UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch


LOCALIZATION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = LOCALIZATION_ROOT.parent
CONTROL_ROOT = WORKSPACE_ROOT / "控制介面程式"
DEPLOY_ROOT = LOCALIZATION_ROOT / "deploy_code" / "sfm_glomap_deploy"
for candidate in (VALIDATION_ROOT, CONTROL_ROOT, DEPLOY_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from production_localizer_factory import build_production_localizer  # noqa: E402
from site_profile import load_site_profile  # noqa: E402
from stream_integrity import StreamAudit, ffprobe_frame_count  # noqa: E402


DEFAULT_SITE_PROFILE = CONTROL_ROOT / "site_profiles" / "river_site_edm.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], value: float) -> float | None:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=float), value))


def metric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "mean": float(np.mean(values)) if values else None,
    }


def _result_exit_code(
    row_count: int,
    audit: StreamAudit,
    *,
    all_frames_requested: bool,
    requested_frames_complete: bool = True,
    quality_failures: list[str] | None = None,
    decode_accepted: bool = False,
) -> int:
    if row_count == 0:
        return 2
    if all_frames_requested:
        if (
            not decode_accepted
            and (audit.decode_complete is not True or audit.decode_errors != 0)
        ):
            return 3
    elif not requested_frames_complete:
        return 3
    return 4 if quality_failures else 0


def evaluate_quality(summary: dict, baseline: dict) -> list[str]:
    """Return deterministic localization regressions against a pinned baseline."""
    thresholds = baseline.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("quality baseline must contain a thresholds object")

    state_counts = summary.get("state_counts") or {}
    rejection_counts = summary.get("rejection_counts") or {}
    inliers = summary.get("inliers") or {}
    reproj = summary.get("reproj_rms") or {}
    checks = (
        ("frames", summary.get("frames"), "==", thresholds.get("frames")),
        ("successes", summary.get("successes"), ">=", thresholds.get("min_successes")),
        ("TRACK", state_counts.get("TRACK", 0), ">=", thresholds.get("min_track")),
        ("LOST", state_counts.get("LOST", 0), "<=", thresholds.get("max_lost")),
        ("inliers.p50", inliers.get("p50"), ">=", thresholds.get("min_inliers_p50")),
        ("inliers.p95", inliers.get("p95"), ">=", thresholds.get("min_inliers_p95")),
        (
            "reproj_rms.p95", reproj.get("p95"), "<=",
            thresholds.get("max_reproj_rms_p95"),
        ),
        (
            "limited_jump_unconfirmed",
            rejection_counts.get("limited_jump_unconfirmed", 0),
            "<=",
            thresholds.get("max_limited_jump_unconfirmed"),
        ),
    )
    failures = []
    for name, actual, operator, expected in checks:
        if expected is None:
            raise ValueError(f"quality baseline threshold missing for {name}")
        try:
            actual_value = float(actual)
            expected_value = float(expected)
        except (TypeError, ValueError):
            failures.append(f"{name}: unavailable (required {operator} {expected})")
            continue
        passed = (
            actual_value == expected_value if operator == "=="
            else actual_value >= expected_value if operator == ">="
            else actual_value <= expected_value
        )
        if not passed:
            failures.append(f"{name}: actual={actual} required {operator} {expected}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, default=DEFAULT_SITE_PROFILE)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-corr-total",
        type=int,
        default=0,
        help="override profile PnP cap; 0 keeps the profile value",
    )
    parser.add_argument("--pnp-random-seed", type=int, default=0)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail before loading assets if CUDA is unavailable",
    )
    parser.add_argument(
        "--quality-baseline",
        type=Path,
        help="fail on localization-quality regression against this pinned baseline",
    )
    parser.add_argument(
        "--accept-known-incomplete",
        action="store_true",
        help=(
            "accept exactly one declared-but-undecodable tail frame when the "
            "quality baseline pins the decoded frame count and video SHA"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stride <= 0 or args.max_frames < 0 or args.max_corr_total < 0:
        raise SystemExit("stride must be > 0; frame/correspondence limits must be >= 0")
    site_path = args.site_profile.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required but unavailable. Check nvidia-smi and confirm the loaded "
            "kernel driver matches libcuda before running this smoke test."
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault(
        "SFM_TORCH_HUB_CACHE",
        str(WORKSPACE_ROOT / "執行環境" / "torch_hub_cache"),
    )
    site = load_site_profile(site_path)
    if site.localizer != "edm" or site.query_camera is None:
        raise SystemExit("benchmark requires an EDM site profile with query_camera")

    camera_tuple = (
        site.query_camera.model,
        site.query_camera.width,
        site.query_camera.height,
        list(site.query_camera.params),
    )
    startup_started = time.perf_counter()
    built = build_production_localizer(
        backend="edm",
        bundle=site.localization_bundle,
        bundle_sha256=site.asset_sha256.localization_bundle,
        frame_source=lambda: None,
        camera_tuple=camera_tuple,
        production_profile=site.localizer_profile,
        production_profile_sha256=site.asset_sha256.localizer_profile,
    )
    if args.max_corr_total:
        built.config.max_corr_total = int(args.max_corr_total)
        built.config.validate()
    built.tracker.ensure_models()
    _camera, pnp_options = built.tracker.trk._pose_estimation_context()
    pnp_options.ransac.random_seed = int(args.pnp_random_seed)
    startup_ms = (time.perf_counter() - startup_started) * 1000.0
    if args.require_cuda and built.device != "cuda":
        raise SystemExit(f"production localizer selected {built.device!r}, expected 'cuda'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot decode video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        cap.release()
        raise SystemExit(f"video has invalid FPS: {source_fps!r}")
    reported_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_raw_frames = (
        int(round(reported_frames))
        if math.isfinite(reported_frames) and reported_frames > 0.0
        else None
    )
    all_frames_requested = args.max_frames == 0
    if all_frames_requested:
        expected_raw_frames, expected_source = ffprobe_frame_count(video_path)
    else:
        expected_raw_frames, expected_source = None, "partial_run"
    audit = StreamAudit(
        expected_raw_frames=expected_raw_frames,
        expected_source=expected_source,
        capture_opened=True,
        reported_raw_frames=reported_raw_frames,
    )
    rows = []
    wall_values: list[float] = []
    pnp_values: list[float] = []
    match_values: list[float] = []
    inlier_values: list[float] = []
    reproj_values: list[float] = []
    steps: list[float] = []
    previous_center: np.ndarray | None = None
    # The EDM matcher caches reference backbone features across frames. Its hit
    # rate depends on how often the tracker re-picks the same reference, which is
    # a property of this route and this video, so it belongs in the replay record.
    feature_cache = getattr(
        getattr(getattr(built.tracker, "trk", None), "loc", None), "matcher", None
    )
    read_cache_stats = getattr(feature_cache, "reference_feature_cache_stats", None)
    previous_cache_stats = read_cache_stats() if callable(read_cache_stats) else None
    source_index = 0
    selected_count = 0
    processing_started = time.perf_counter()
    try:
        while not args.max_frames or selected_count < args.max_frames:
            decode_started = time.perf_counter()
            try:
                ok, bgr = cap.read()
            except cv2.error:
                audit.decode_errors += 1
                break
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if not ok:
                break
            current_index = source_index
            source_index += 1
            audit.decoded_raw_frames += 1
            if current_index % args.stride:
                continue
            if bgr.shape[1] != site.query_camera.width or bgr.shape[0] != site.query_camera.height:
                bgr = cv2.resize(
                    bgr,
                    (site.query_camera.width, site.query_camera.height),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            capture_stamp = float(current_index) / source_fps
            started = time.perf_counter()
            pose = built.tracker.localize_frame(rgb, capture_stamp=capture_stamp)
            wall_ms = (time.perf_counter() - started) * 1000.0
            info = dict(built.tracker.last_info)
            center = None if pose is None else np.asarray([pose.x, pose.y, pose.z], float)
            step = None
            if center is not None and previous_center is not None:
                step = float(np.linalg.norm(center - previous_center))
                steps.append(step)
            if center is not None:
                previous_center = center
            row = {
                "source_index": current_index,
                "capture_stamp": capture_stamp,
                "success": pose is not None,
                "mode": info.get("mode"),
                "next_mode": info.get("next_mode"),
                "rejected": info.get("rejected"),
                "limited_jump": info.get("limited_jump"),
                "limited_jump_confirmed": bool(info.get("limited_jump_confirmed")),
                "inliers": int(info.get("inliers", 0) or 0),
                "reproj_rms": info.get("reproj_rms"),
                "n_corr": info.get("n_corr"),
                "reference_count": info.get("reference_count"),
                "refs": list(info.get("refs") or []),
                "wall_ms": wall_ms,
                "decode_ms": decode_ms,
                "vpr_ms": info.get("vpr_ms"),
                "match_ms": info.get("match_ms"),
                "pnp_ms": info.get("pnp_ms"),
                "step": step,
            }
            if previous_cache_stats is not None:
                stats = read_cache_stats()
                row["ref_feature_cache"] = {
                    key: stats[key] - previous_cache_stats[key]
                    for key in ("hits", "misses", "evictions")
                }
                previous_cache_stats = stats
            rows.append(row)
            selected_count += 1
            audit.sampled_frames += 1
            wall_values.append(wall_ms)
            for key, target in (
                ("pnp_ms", pnp_values),
                ("match_ms", match_values),
                ("inliers", inlier_values),
                ("reproj_rms", reproj_values),
            ):
                value = row.get(key)
                if value is not None and math.isfinite(float(value)):
                    target.append(float(value))
            if selected_count == 1 or selected_count % 100 == 0:
                print(
                    f"frames={selected_count} state={row['next_mode']} "
                    f"ok={row['success']} inliers={row['inliers']} wall={wall_ms:.1f}ms",
                    flush=True,
                )
    finally:
        cap.release()
    requested_frames_complete = not args.max_frames or selected_count >= args.max_frames
    if all_frames_requested:
        audit.finish()
    processing_s = time.perf_counter() - processing_started

    successes = sum(bool(row["success"]) for row in rows)
    summary = {
        "frames": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "state_counts": dict(Counter(str(row["next_mode"]) for row in rows)),
        "rejection_counts": dict(Counter(
            str(row["rejected"]) for row in rows if row["rejected"]
        )),
        "limited_jump_confirmed": sum(
            bool(row["limited_jump_confirmed"]) for row in rows
        ),
        "wall_ms": metric_summary(wall_values),
        "pnp_ms": metric_summary(pnp_values),
        "match_ms": metric_summary(match_values),
        "inliers": metric_summary(inlier_values),
        "reproj_rms": metric_summary(reproj_values),
        "accepted_step": metric_summary(steps),
        "ref_feature_cache": None if previous_cache_stats is None else {
            **previous_cache_stats,
            "hit_rate": previous_cache_stats["hits"] / max(
                previous_cache_stats["hits"] + previous_cache_stats["misses"], 1
            ),
        },
        "processing_s": processing_s,
        "processing_fps": len(rows) / processing_s if processing_s > 0.0 else 0.0,
    }
    quality_baseline = None
    quality_failures: list[str] = []
    if args.quality_baseline is not None:
        baseline_path = args.quality_baseline.expanduser().resolve()
        quality_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        expected_video_sha = str(quality_baseline.get("video_sha256") or "")
        actual_video_sha = sha256_file(video_path)
        if expected_video_sha and expected_video_sha != actual_video_sha:
            raise SystemExit(
                "quality baseline video SHA-256 mismatch: "
                f"expected={expected_video_sha} actual={actual_video_sha}"
            )
        quality_failures = evaluate_quality(summary, quality_baseline)
    if args.accept_known_incomplete and quality_baseline is None:
        raise SystemExit("--accept-known-incomplete requires --quality-baseline")
    expected_decoded = (
        int(quality_baseline["thresholds"]["frames"])
        if quality_baseline is not None else 0
    )
    known_incomplete_accepted = bool(
        args.accept_known_incomplete
        and all_frames_requested
        and audit.reported_raw_frames == expected_decoded + 1
        and audit.decoded_raw_frames == expected_decoded
        and audit.sampled_frames == expected_decoded
        and audit.decode_errors == 1
    )
    result = {
        "schema": "edm-site-replay/v1",
        "site_profile": str(site_path),
        "site_profile_sha256": sha256_file(site_path),
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "bundle": str(site.localization_bundle),
        "bundle_sha256": site.asset_sha256.localization_bundle,
        "localizer_profile": str(site.localizer_profile),
        "localizer_profile_sha256": site.asset_sha256.localizer_profile,
        "device": built.device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tracker_variant": built.variant,
        "source_fps": source_fps,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "max_corr_total": built.config.max_corr_total,
        "pnp_random_seed": args.pnp_random_seed,
        "startup_ms": startup_ms,
        "stream_audit": {
            **audit.as_dict(),
            "all_frames_requested": all_frames_requested,
            "requested_max_frames": args.max_frames,
            "requested_frames_complete": requested_frames_complete,
            "known_incomplete_accepted": known_incomplete_accepted,
        },
        "summary": summary,
        "quality_gate": {
            "enabled": quality_baseline is not None,
            "passed": not quality_failures,
            "failures": quality_failures,
            "baseline": (
                str(args.quality_baseline.expanduser().resolve())
                if args.quality_baseline is not None else None
            ),
        },
        "rows": rows,
    }
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"localized={successes}/{len(rows)} ({summary['success_rate']:.1%}) "
        f"p50/p95={summary['wall_ms']['p50']}/{summary['wall_ms']['p95']}ms "
        f"out={out}",
        flush=True,
    )
    exit_code = _result_exit_code(
        len(rows),
        audit,
        all_frames_requested=all_frames_requested,
        requested_frames_complete=requested_frames_complete,
        quality_failures=quality_failures,
        decode_accepted=known_incomplete_accepted,
    )
    if exit_code == 3:
        print(
            "incomplete replay decode: "
            f"expected={audit.expected_raw_frames} "
            f"reported={audit.reported_raw_frames} "
            f"decoded={audit.decoded_raw_frames} "
            f"errors={audit.decode_errors}",
            file=sys.stderr,
            flush=True,
        )
    if exit_code == 4:
        for failure in quality_failures:
            print(f"localization quality regression: {failure}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
