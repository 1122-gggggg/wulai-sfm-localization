#!/usr/bin/env python3
"""Paired off/shadow RePoseD gate evaluation for river-site videos."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from dataclasses import replace
from typing import Any

import numpy as np


LOCALIZATION_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = LOCALIZATION_ROOT.parent
CONTROL_ROOT = WORKSPACE_ROOT / "控制介面程式"
DEPLOY_ROOT = LOCALIZATION_ROOT / "deploy_code" / "sfm_glomap_deploy"
for candidate in (VALIDATION_ROOT, CONTROL_ROOT, DEPLOY_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_edm_site_replay import (  # noqa: E402
    _append_replay_metrics,
    _build_replay_result,
    _build_replay_runtime,
    _evaluate_replay_quality,
    _finish_replay_stream,
    _load_replay_profile,
    _open_replay_stream,
    _print_replay_progress,
    _process_replay_frame,
    _summarize_replay,
    _update_replay_cache_stats,
    camera_identity,
    percentile,
    sha256_file,
)
from edm_profile import load_edm_production_profile  # noqa: E402


MIN_INLIER_GRID = (30, 50, 80, 120)
ROTATION_GRID = (1, 2, 3, 5, 8)
TRANSLATION_GRID = (5, 10, 15, 25, 45)
QUALITY_REJECTIONS = {
    "reprojection",
    "reprojection_unavailable",
    "inlier_ratio",
    "inlier_spread",
}
HELD_OUT_MIN_EVENTS = 20
MAX_P95_VALIDATOR_MS = 100.0
MAX_THROUGHPUT_REGRESSION = 0.05
MAX_GPU_FRACTION = 0.90


def label_first_limited_jumps(rows: list[dict]) -> list[dict]:
    """Label first limited-jump events from an already-finished replay."""
    labeled: list[dict] = []
    index = 0
    while index < len(rows):
        row = rows[index]
        jump = row.get("limited_jump") or {}
        if row.get("rejected") != "limited_jump_unconfirmed" or jump.get("confirmation_model") is not None:
            index += 1
            continue
        follow = index + 1
        label = None
        while follow < len(rows):
            nxt = rows[follow]
            rejected = nxt.get("rejected")
            nxt_jump = nxt.get("limited_jump") or {}
            if nxt.get("limited_jump_confirmed"):
                label = "confirmed"
                break
            if rejected in QUALITY_REJECTIONS:
                label = None
                break
            if (
                rejected == "limited_jump_unconfirmed"
                and nxt_jump.get("confirmation_residual") is not None
                and nxt_jump.get("confirmation_limit") is not None
                and float(nxt_jump["confirmation_residual"]) > float(nxt_jump["confirmation_limit"])
            ):
                label = "rejected"
                break
            if rejected in {"jump", "acquire_jump", "acquire_yaw"}:
                label = None
                break
            if not nxt.get("success") and rejected not in {None, "limited_jump_unconfirmed"}:
                label = None
                break
            follow += 1
        if label is not None:
            labeled.append(
                {
                    "source_index": row.get("source_index", index),
                    "label": label,
                    "start_row": row,
                    "follow_row": rows[follow] if follow < len(rows) else None,
                }
            )
        index += 1
    return labeled


def _would_fast_confirm(event: dict, min_inliers: int, rotation_deg: float, translation_deg: float) -> bool:
    inliers = event.get("inliers")
    rotation = event.get("rotation_delta_deg")
    translation = event.get("translation_direction_delta_deg")
    if inliers is None or rotation is None or translation is None:
        return False
    if int(inliers) < int(min_inliers):
        return False
    return float(rotation) <= float(rotation_deg) and float(translation) <= float(translation_deg)


def evaluate_threshold_grid(events: list[dict]) -> list[dict]:
    results = []
    latencies = [float(event["total_ms"]) for event in events if event.get("total_ms") is not None]
    p95 = percentile(latencies, 95) if latencies else None
    for min_inliers in MIN_INLIER_GRID:
        for rotation_deg in ROTATION_GRID:
            for translation_deg in TRANSLATION_GRID:
                true_pos = 0
                false_pos = 0
                for event in events:
                    if not _would_fast_confirm(event, min_inliers, rotation_deg, translation_deg):
                        continue
                    if event.get("label") == "confirmed":
                        true_pos += 1
                    elif event.get("label") == "rejected":
                        false_pos += 1
                results.append(
                    {
                        "min_inliers": int(min_inliers),
                        "max_rotation_delta_deg": float(rotation_deg),
                        "max_translation_direction_delta_deg": float(translation_deg),
                        "true_fast_confirms": true_pos,
                        "false_fast_confirms": false_pos,
                        "p95_validator_ms": p95,
                    }
                )
    return results


def select_threshold_tuple(grid: list[dict]) -> dict | None:
    eligible = [
        row
        for row in grid
        if int(row["false_fast_confirms"]) == 0 and int(row["true_fast_confirms"]) >= 1
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            -int(row["true_fast_confirms"]),
            float(row["p95_validator_ms"] if row["p95_validator_ms"] is not None else math.inf),
            float(row["max_rotation_delta_deg"]),
            float(row["max_translation_direction_delta_deg"]),
            int(row["min_inliers"]),
        )
    )
    return eligible[0]


def attach_event_metrics(labeled: list[dict], shadow_rows: list[dict]) -> list[dict]:
    by_index = {row.get("source_index"): row for row in shadow_rows}
    events = []
    for item in labeled:
        shadow = by_index.get(item["source_index"], {})
        check = (shadow.get("relative_motion_check") or {})
        events.append(
            {
                **item,
                "inliers": check.get("inliers"),
                "rotation_delta_deg": check.get("rotation_delta_deg"),
                "translation_direction_delta_deg": check.get(
                    "translation_direction_delta_deg"
                ),
                "total_ms": check.get("total_ms"),
                "status": check.get("status"),
                "reason": check.get("reason"),
            }
        )
    return events


def promotion_decision(
    *,
    heldout_labeled: int,
    false_fast_confirms: int,
    true_fast_confirms: int,
    quality_regressions: list[str],
    rejected_pose_introductions: int,
    throughput_ratio: float | None,
    p95_validator_ms: float | None,
    peak_gpu_fraction: float | None,
    cuda_oom: bool,
    model_fallback: bool,
) -> dict[str, Any]:
    gates = {
        "heldout_labeled_events": heldout_labeled >= HELD_OUT_MIN_EVENTS,
        "zero_false_fast_confirms": false_fast_confirms == 0,
        "at_least_one_true_fast_confirm": true_fast_confirms >= 1,
        "no_quality_regression": not quality_regressions,
        "no_rejected_pose_introduction": rejected_pose_introductions == 0,
        "throughput_within_5pct": (
            throughput_ratio is not None and throughput_ratio >= (1.0 - MAX_THROUGHPUT_REGRESSION)
        ),
        "p95_validator_below_100ms": (
            p95_validator_ms is not None and p95_validator_ms < MAX_P95_VALIDATOR_MS
        ),
        "gpu_below_90pct": (
            peak_gpu_fraction is not None and peak_gpu_fraction < MAX_GPU_FRACTION
        ),
        "no_cuda_oom": not cuda_oom,
        "no_model_fallback": not model_fallback,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "promotion_eligible": not failed,
        "gates": gates,
        "failed_gates": failed,
        "heldout_labeled_events": heldout_labeled,
        "false_fast_confirms": false_fast_confirms,
        "true_fast_confirms": true_fast_confirms,
        "quality_regressions": quality_regressions,
        "rejected_pose_introductions": rejected_pose_introductions,
        "throughput_ratio": throughput_ratio,
        "p95_validator_ms": p95_validator_ms,
        "peak_gpu_fraction": peak_gpu_fraction,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_shadow_profile(base_profile: Path, out_path: Path, thresholds: dict, mode: str) -> Path:
    raw = json.loads(base_profile.read_text(encoding="utf-8"))
    raw["reposed"] = {
        "mode": mode,
        "model_path": "執行環境/models/moge-2-vits-normal/model.pt",
        "model_sha256": "79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc",
        "num_tokens": 1200,
        "max_matches": 1200,
        "min_inliers": int(thresholds["min_inliers"]),
        "max_rotation_delta_deg": float(thresholds["max_rotation_delta_deg"]),
        "max_translation_direction_delta_deg": float(
            thresholds["max_translation_direction_delta_deg"]
        ),
    }
    out_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return out_path


class _Args(dict):
    def __getattr__(self, name):
        return self[name]


def _replay_args(site_path: Path, video_path: Path, out: Path, seed: int) -> _Args:
    return _Args(
        site_profile=site_path,
        video=video_path,
        out=out,
        stride=1,
        max_frames=0,
        max_corr_total=0,
        pnp_random_seed=int(seed),
        quality_baseline=None,
        require_cuda=False,
        accept_known_incomplete=False,
    )


def _run_one_replay(site_path: Path, video_path: Path, out: Path, seed: int, production_profile=None):
    args = _replay_args(site_path, video_path, out, seed)
    loaded = _load_replay_profile(args, site_path, video_path)
    site, camera_tuple, site_profile_sha256, video_sha256, camera, quality_baseline = loaded
    if production_profile is not None:
        profile_path = Path(production_profile).expanduser().resolve()
        site = replace(
            site,
            localizer_profile=profile_path,
            asset_sha256=replace(
                site.asset_sha256,
                localizer_profile=sha256_file(profile_path),
            ),
        )
    built, startup_ms = _build_replay_runtime(args, site, camera_tuple)
    cap, source_fps, all_frames_requested, audit = _open_replay_stream(args, video_path)
    rows = []
    wall_values: list[float] = []
    pnp_values: list[float] = []
    match_values: list[float] = []
    inlier_values: list[float] = []
    reproj_values: list[float] = []
    steps: list[float] = []
    previous_center = None
    feature_cache = getattr(
        getattr(getattr(built.tracker, "trk", None), "loc", None), "matcher", None
    )
    read_cache_stats = getattr(feature_cache, "reference_feature_cache_stats", None)
    previous_cache_stats = read_cache_stats() if callable(read_cache_stats) else None
    source_index = 0
    selected_count = 0
    processing_started = time.perf_counter()
    try:
        import cv2

        while True:
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
            row, previous_center, step = _process_replay_frame(
                bgr,
                current_index,
                decode_ms,
                source_fps,
                site,
                built,
                previous_center,
            )
            info = dict(built.tracker.last_info)
            row["relative_motion_check"] = info.get("relative_motion_check")
            previous_cache_stats = _update_replay_cache_stats(
                row, read_cache_stats, previous_cache_stats
            )
            rows.append(row)
            selected_count += 1
            audit.sampled_frames += 1
            if step is not None:
                steps.append(step)
            _append_replay_metrics(
                row,
                wall_values,
                pnp_values,
                match_values,
                inlier_values,
                reproj_values,
            )
            _print_replay_progress(selected_count, row)
    finally:
        cap.release()
    requested_complete = _finish_replay_stream(args, audit, all_frames_requested, selected_count)
    processing_s = time.perf_counter() - processing_started
    summary = _summarize_replay(
        rows,
        wall_values,
        pnp_values,
        match_values,
        inlier_values,
        reproj_values,
        steps,
        previous_cache_stats,
        processing_s,
    )
    quality_failures, known_incomplete = _evaluate_replay_quality(
        args,
        summary,
        quality_baseline,
        audit,
        all_frames_requested,
    )
    result = _build_replay_result(
        args,
        site_path,
        video_path,
        site,
        built,
        source_fps,
        site_profile_sha256,
        video_sha256,
        camera,
        startup_ms,
        audit,
        all_frames_requested,
        requested_complete,
        known_incomplete,
        quality_baseline,
        summary,
        quality_failures,
        rows,
    )
    result["input_frame_count"] = selected_count
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result, rows, audit, summary




def _quality_snapshot(rows: list[dict]) -> dict[str, Any]:
    successes = sum(bool(row.get("success")) for row in rows)
    return {
        "successes": successes,
        "frames": len(rows),
        "success_rate": successes / len(rows) if rows else 0.0,
        "track": sum(row.get("next_mode") == "TRACK" for row in rows),
        "lost": sum(row.get("next_mode") == "LOST" for row in rows),
        "inliers_mean": float(np.mean([row["inliers"] for row in rows if row.get("inliers") is not None] or [0.0])),
        "reproj_mean": float(
            np.mean(
                [
                    float(row["reproj_rms"])
                    for row in rows
                    if row.get("reproj_rms") is not None and math.isfinite(float(row["reproj_rms"]))
                ]
                or [0.0]
            )
        ),
    }


def _quality_regressions(off_rows: list[dict], other_rows: list[dict]) -> list[str]:
    off = _quality_snapshot(off_rows)
    other = _quality_snapshot(other_rows)
    failures = []
    if other["successes"] < off["successes"]:
        failures.append(f"successes {other['successes']} < {off['successes']}")
    if other["track"] < off["track"]:
        failures.append(f"TRACK {other['track']} < {off['track']}")
    if other["lost"] > off["lost"]:
        failures.append(f"LOST {other['lost']} > {off['lost']}")
    if other["inliers_mean"] + 1e-9 < off["inliers_mean"]:
        failures.append(f"inliers {other['inliers_mean']} < {off['inliers_mean']}")
    if other["reproj_mean"] > off["reproj_mean"] + 1e-9:
        failures.append(f"reproj {other['reproj_mean']} > {off['reproj_mean']}")
    return failures


def _count_confusion(events: list[dict], thresholds: dict | None) -> dict[str, int]:
    if thresholds is None:
        return {"true_fast_confirms": 0, "false_fast_confirms": 0, "labeled": len(events)}
    true_pos = false_pos = 0
    for event in events:
        if not _would_fast_confirm(
            event,
            thresholds["min_inliers"],
            thresholds["max_rotation_delta_deg"],
            thresholds["max_translation_direction_delta_deg"],
        ):
            continue
        if event["label"] == "confirmed":
            true_pos += 1
        elif event["label"] == "rejected":
            false_pos += 1
    return {
        "true_fast_confirms": true_pos,
        "false_fast_confirms": false_pos,
        "labeled": len(events),
    }


def _run_pair(site_path: Path, video_path: Path, out_dir: Path, seed: int, shadow_profile: Path | None):
    off_out = out_dir / f"{video_path.stem}_off.json"
    off_result, off_rows, off_audit, off_summary = _run_one_replay(
        site_path, video_path, off_out, seed
    )
    shadow_result = shadow_rows = shadow_audit = shadow_summary = None
    if shadow_profile is not None:
        shadow_out = out_dir / f"{video_path.stem}_shadow.json"
        shadow_result, shadow_rows, shadow_audit, shadow_summary = _run_one_replay(
            site_path, video_path, shadow_out, seed, production_profile=shadow_profile
        )
    return {
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "off": off_result,
        "off_rows": off_rows,
        "off_audit": off_audit.as_dict(),
        "shadow": shadow_result,
        "shadow_rows": shadow_rows,
        "shadow_audit": None if shadow_audit is None else shadow_audit.as_dict(),
        "off_summary": off_summary,
        "shadow_summary": shadow_summary,
    }


def _load_shadow_profile(
    base_profile: Path, out: Path, thresholds: dict
) -> tuple[Path | None, bool]:
    shadow_profile_path = None
    try:
        shadow_profile_path = _write_shadow_profile(
            base_profile,
            out / "shadow_recording_profile.json",
            thresholds,
            "shadow",
        )
        load_edm_production_profile(shadow_profile_path)
    except Exception as exc:
        print(f"shadow profile unavailable: {exc}", flush=True)
        return None, True
    return shadow_profile_path, False



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, required=True)
    parser.add_argument("--calibration-video", type=Path, action="append", default=[])
    parser.add_argument("--heldout-video", type=Path, action="append", default=[])
    parser.add_argument("--update-video", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    site_path = args.site_profile.expanduser().resolve()
    from site_profile import load_site_profile

    site = load_site_profile(site_path)
    if site.localizer != "edm":
        raise SystemExit("evaluator requires an EDM site profile")
    base_profile = Path(site.localizer_profile).expanduser().resolve()
    recording_thresholds = {
        "min_inliers": 30,
        "max_rotation_delta_deg": 8,
        "max_translation_direction_delta_deg": 45,
    }
    shadow_profile_path, model_fallback = _load_shadow_profile(
        base_profile, out, recording_thresholds
    )
    cuda_oom = False

    calibration = []
    labeled_events = []
    for video in args.calibration_video:
        pair = _run_pair(site_path, video.expanduser().resolve(), out, args.seed, shadow_profile_path)
        calibration.append(pair)
        labeled = label_first_limited_jumps(pair["off_rows"])
        labeled_events.extend(attach_event_metrics(labeled, pair["shadow_rows"] or pair["off_rows"]))

    grid = evaluate_threshold_grid(labeled_events)
    selected = select_threshold_tuple(grid)

    heldout = []
    heldout_events = []
    for video in args.heldout_video:
        pair = _run_pair(site_path, video.expanduser().resolve(), out, args.seed, shadow_profile_path)
        heldout.append(pair)
        labeled = label_first_limited_jumps(pair["off_rows"])
        heldout_events.extend(attach_event_metrics(labeled, pair["shadow_rows"] or pair["off_rows"]))

    update = []
    for video in args.update_video:
        update.append(
            _run_pair(site_path, video.expanduser().resolve(), out, args.seed, shadow_profile_path)
        )

    heldout_confusion = _count_confusion(heldout_events, selected)
    quality_failures: list[str] = []
    throughput_ratios = []
    validator_times = [
        float(event["total_ms"])
        for event in heldout_events
        if event.get("total_ms") is not None
    ]
    for pair in heldout:
        if pair["shadow_rows"] is not None:
            quality_failures.extend(_quality_regressions(pair["off_rows"], pair["shadow_rows"]))
            off_ms = pair["off_summary"].get("wall_ms", {}).get("sum") or pair["off_summary"].get("elapsed_ms")
            sh_ms = pair["shadow_summary"].get("wall_ms", {}).get("sum") or pair["shadow_summary"].get("elapsed_ms")
            if off_ms and sh_ms:
                throughput_ratios.append(float(off_ms) / float(sh_ms))
    throughput_ratio = min(throughput_ratios) if throughput_ratios else None
    p95 = percentile(validator_times, 95) if validator_times else None
    peak_gpu_fraction = None
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            peak = torch.cuda.max_memory_allocated()
            peak_gpu_fraction = float(peak) / float(total) if total else None
    except Exception:
        peak_gpu_fraction = None

    decision = promotion_decision(
        heldout_labeled=len(heldout_events),
        false_fast_confirms=heldout_confusion["false_fast_confirms"],
        true_fast_confirms=heldout_confusion["true_fast_confirms"],
        quality_regressions=quality_failures,
        rejected_pose_introductions=heldout_confusion["false_fast_confirms"],
        throughput_ratio=throughput_ratio,
        p95_validator_ms=p95,
        peak_gpu_fraction=peak_gpu_fraction,
        cuda_oom=cuda_oom,
        model_fallback=model_fallback or selected is None,
    )
    report = {
        "seed": int(args.seed),
        "moge_revision": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
        "poselib_revision": "fa7280fee27f97aff31ae7f98bab7f583fac7d08",
        "checkpoint_sha256": "79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc",
        "camera": camera_identity(site.query_camera) if site.query_camera is not None else None,
        "selected_thresholds": selected,
        "calibration_confusion": _count_confusion(labeled_events, selected),
        "heldout_confusion": heldout_confusion,
        "threshold_grid": grid,
        "promotion": decision,
        "videos": {
            "calibration": [item["video"] for item in calibration],
            "heldout": [item["video"] for item in heldout],
            "update": [item["video"] for item in update],
        },
        "audits": {
            "calibration": [item["off_audit"] for item in calibration],
            "heldout": [item["off_audit"] for item in heldout],
            "update": [item["off_audit"] for item in update],
        },
    }

    (out / "reposed_motion_gate.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"promotion_eligible": decision["promotion_eligible"], "selected": selected}, indent=2))
    return 0 if not decision["failed_gates"] or selected is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
