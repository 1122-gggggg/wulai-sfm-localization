#!/usr/bin/env python3
"""Measure a TRACK candidate against a full XFeat+LightGlue reference run.

The reference is a repeatable image-only pseudo-ground-truth, not surveyed ground
truth.  Metrics therefore describe disagreement with the reference trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


SUPPORTED_REFERENCE_EVIDENCE_KINDS = frozenset({
    "surveyed",
    "motion_capture",
    "map_aligned_apriltag",
    "rtk_map_aligned",
})
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


def _reference_claim_scope(reference: dict) -> dict:
    """Classify the strongest claim supported by independent reference evidence."""
    evidence = reference.get("reference_evidence")
    errors: list[str] = []
    if not isinstance(evidence, dict):
        errors.append("reference_evidence is missing or is not an object")
        evidence_view = None
    else:
        kind = evidence.get("kind")
        if not isinstance(kind, str) or kind not in SUPPORTED_REFERENCE_EVIDENCE_KINDS:
            errors.append("kind is not a supported independent reference type")
        if evidence.get("map_aligned") is not True:
            errors.append("map_aligned must be true")
        if evidence.get("independent_of_localizer") is not True:
            errors.append("independent_of_localizer must be true")
        source_sha256 = evidence.get("source_sha256")
        if not isinstance(source_sha256, str) or _LOWER_SHA256.fullmatch(source_sha256) is None:
            errors.append("source_sha256 must be 64 lowercase hexadecimal characters")
        evidence_view = {
            "kind": kind,
            "map_aligned": evidence.get("map_aligned"),
            "independent_of_localizer": evidence.get("independent_of_localizer"),
            "source_sha256": source_sha256,
        }

    absolute_accuracy_validated = not errors
    scope = "absolute_accuracy" if absolute_accuracy_validated else "pseudo/continuity-only"
    interpretation = {
        "absolute_accuracy_validated": absolute_accuracy_validated,
        "reference_type": (
            "independent_map_aligned_ground_truth"
            if absolute_accuracy_validated
            else "image_only_pseudo_reference"
        ),
        "basis": (
            "reference_evidence satisfies the independent, map-aligned evidence gate"
            if absolute_accuracy_validated
            else "reference_evidence does not satisfy the independent, map-aligned evidence gate"
        ),
        "validation_errors": errors,
    }
    return {
        "scope": scope,
        "absolute_accuracy_validated": absolute_accuracy_validated,
        "reference_evidence": evidence_view,
        "interpretation": interpretation,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


def _pose(row: dict) -> dict | None:
    pose = row.get("pose")
    return pose if row.get("success") and isinstance(pose, dict) else None


def _xyz(pose: dict) -> tuple[float, float, float]:
    return float(pose["x"]), float(pose["y"]), float(pose["z"])


def _sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _norm(v: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _yaw_error_rad(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _longest_failure_run(rows: list[dict]) -> int:
    longest = current = 0
    for row in rows:
        if _pose(row) is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _track_latency(rows: list[dict]) -> dict:
    """Latency for normal high-frequency tracking, excluding BOOT/LOST/WEAK."""
    track_rows = [row for row in rows if row.get("mode") in {"TRACK", "NEUFLOW_TRACK"}]
    wall = [float(row["wall_ms"]) for row in track_rows if row.get("wall_ms") is not None]
    sequential = [
        float(row["wall_ms"]) + float(row.get("load_ms", 0.0) or 0.0)
        for row in track_rows if row.get("wall_ms") is not None
    ]
    wall_stats = _stats(wall)
    sequential_stats = _stats(sequential)
    return {
        "frames": len(track_rows),
        "success": sum(_pose(row) is not None for row in track_rows),
        "wall_ms": wall_stats,
        "fps_from_median_wall": (
            1000.0 / wall_stats["median"] if wall_stats["median"] else None
        ),
        "wall_plus_load_ms": sequential_stats,
        "fps_from_median_wall_plus_load": (
            1000.0 / sequential_stats["median"] if sequential_stats["median"] else None
        ),
    }


def _validate_alignment(reference_rows: list[dict], candidate_rows: list[dict]) -> None:
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(
            f"row count differs: reference={len(reference_rows)} candidate={len(candidate_rows)}"
        )
    for index, (ref, cand) in enumerate(zip(reference_rows, candidate_rows)):
        if ref.get("idx") != cand.get("idx") or ref.get("frame") != cand.get("frame"):
            raise ValueError(f"frame alignment differs at row {index}")
        ref_stamp, cand_stamp = ref.get("capture_stamp"), cand.get("capture_stamp")
        if ref_stamp is not None and cand_stamp is not None:
            if abs(float(ref_stamp) - float(cand_stamp)) > 1e-9:
                raise ValueError(f"capture timestamp differs at row {index}")


def compare_pose_reference(reference: dict, candidate: dict) -> dict:
    """Return trajectory disagreement metrics for two aligned benchmark results."""
    ref_rows = reference.get("rows")
    cand_rows = candidate.get("rows")
    if not isinstance(ref_rows, list) or not isinstance(cand_rows, list):
        raise ValueError("both benchmark results must contain rows lists")
    _validate_alignment(ref_rows, cand_rows)

    position_delta: list[float] = []
    yaw_delta_deg: list[float] = []
    common_indices: list[int] = []
    ref_only = candidate_only = neither = 0
    for index, (ref_row, cand_row) in enumerate(zip(ref_rows, cand_rows)):
        ref_pose, cand_pose = _pose(ref_row), _pose(cand_row)
        if ref_pose is not None and cand_pose is not None:
            common_indices.append(index)
            position_delta.append(_norm(_sub(_xyz(cand_pose), _xyz(ref_pose))))
            yaw_delta_deg.append(
                math.degrees(_yaw_error_rad(float(cand_pose["yaw"]), float(ref_pose["yaw"])))
            )
        elif ref_pose is not None:
            ref_only += 1
        elif cand_pose is not None:
            candidate_only += 1
        else:
            neither += 1

    translation_step_error: list[float] = []
    yaw_step_error_deg: list[float] = []
    ref_step: list[float] = []
    candidate_step: list[float] = []
    for index in range(1, len(ref_rows)):
        poses = (
            _pose(ref_rows[index - 1]), _pose(ref_rows[index]),
            _pose(cand_rows[index - 1]), _pose(cand_rows[index]),
        )
        if any(pose is None for pose in poses):
            continue
        ref_prev, ref_now, cand_prev, cand_now = poses
        ref_motion = _sub(_xyz(ref_now), _xyz(ref_prev))
        cand_motion = _sub(_xyz(cand_now), _xyz(cand_prev))
        ref_step.append(_norm(ref_motion))
        candidate_step.append(_norm(cand_motion))
        translation_step_error.append(_norm(_sub(cand_motion, ref_motion)))
        ref_yaw_step = float(ref_now["yaw"]) - float(ref_prev["yaw"])
        cand_yaw_step = float(cand_now["yaw"]) - float(cand_prev["yaw"])
        yaw_step_error_deg.append(math.degrees(_yaw_error_rad(cand_yaw_step, ref_yaw_step)))

    position_second_difference_error: list[float] = []
    yaw_second_difference_error_deg: list[float] = []
    for index in range(2, len(ref_rows)):
        rp0, rp1, rp2 = (_pose(ref_rows[index - 2]), _pose(ref_rows[index - 1]),
                         _pose(ref_rows[index]))
        cp0, cp1, cp2 = (_pose(cand_rows[index - 2]), _pose(cand_rows[index - 1]),
                         _pose(cand_rows[index]))
        if any(pose is None for pose in (rp0, rp1, rp2, cp0, cp1, cp2)):
            continue
        ref_second = _sub(_sub(_xyz(rp2), _xyz(rp1)), _sub(_xyz(rp1), _xyz(rp0)))
        cand_second = _sub(_sub(_xyz(cp2), _xyz(cp1)), _sub(_xyz(cp1), _xyz(cp0)))
        position_second_difference_error.append(_norm(_sub(cand_second, ref_second)))
        ref_second_yaw = (float(rp2["yaw"]) - float(rp1["yaw"])) - (
            float(rp1["yaw"]) - float(rp0["yaw"])
        )
        cand_second_yaw = (float(cp2["yaw"]) - float(cp1["yaw"])) - (
            float(cp1["yaw"]) - float(cp0["yaw"])
        )
        yaw_second_difference_error_deg.append(
            math.degrees(_yaw_error_rad(cand_second_yaw, ref_second_yaw))
        )

    common = len(common_indices)
    large = {
        "position_gt_0_25m": sum(value > 0.25 for value in position_delta),
        "position_gt_0_5m": sum(value > 0.5 for value in position_delta),
        "position_gt_1m": sum(value > 1.0 for value in position_delta),
        "yaw_gt_5deg": sum(value > 5.0 for value in yaw_delta_deg),
        "yaw_gt_10deg": sum(value > 10.0 for value in yaw_delta_deg),
    }
    large["denominator_common_success"] = common

    ref_summary = reference.get("summary", {})
    cand_summary = candidate.get("summary", {})
    claim_scope = _reference_claim_scope(reference)
    legacy_interpretation = (
        "Independent map-aligned reference evidence supports absolute-accuracy validation; "
        "the report also describes trajectory disagreement and continuity."
        if claim_scope["absolute_accuracy_validated"]
        else "Image-only disagreement against full XFeat+LightGlue pseudo-ground-truth; "
        "this is not surveyed ground truth."
    )
    return {
        # Keep the legacy prose field stable for pseudo references.  The structured
        # claim_scope below is the machine-readable authority for new consumers.
        "interpretation": legacy_interpretation,
        "claim_scope": claim_scope,
        "absolute_accuracy_validated": claim_scope["absolute_accuracy_validated"],
        "frames": len(ref_rows),
        "success_overlap": {
            "both": common,
            "reference_only": ref_only,
            "candidate_only": candidate_only,
            "neither": neither,
            "candidate_longest_failure_run": _longest_failure_run(cand_rows),
            "reference_longest_failure_run": _longest_failure_run(ref_rows),
        },
        "absolute_position_delta_m": _stats(position_delta),
        "absolute_yaw_delta_deg": _stats(yaw_delta_deg),
        "frame_to_frame_translation_delta_m": _stats(translation_step_error),
        "frame_to_frame_yaw_delta_deg": _stats(yaw_step_error_deg),
        "second_difference_position_delta_m": _stats(position_second_difference_error),
        "second_difference_yaw_delta_deg": _stats(yaw_second_difference_error_deg),
        "reference_motion_step_m": _stats(ref_step),
        "candidate_motion_step_m": _stats(candidate_step),
        "large_disagreements": large,
        "speed": {
            "reference_actual_fps": ref_summary.get("actual_fps_including_io"),
            "candidate_actual_fps": cand_summary.get("actual_fps_including_io"),
            "reference_wall_ms_p50": ref_summary.get("wall_ms", {}).get("median"),
            "candidate_wall_ms_p50": cand_summary.get("wall_ms", {}).get("median"),
            "reference_wall_ms_p90": ref_summary.get("wall_ms", {}).get("p90"),
            "candidate_wall_ms_p90": cand_summary.get("wall_ms", {}).get("p90"),
            "reference_normal_track": _track_latency(ref_rows),
            "candidate_normal_track": _track_latency(cand_rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--require-absolute-ground-truth",
        action="store_true",
        help="fail before comparison unless reference_evidence passes the absolute-accuracy gate",
    )
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    claim_scope = _reference_claim_scope(reference)
    if args.require_absolute_ground_truth and not claim_scope["absolute_accuracy_validated"]:
        details = "; ".join(claim_scope["interpretation"]["validation_errors"])
        parser.error(f"reference lacks valid absolute ground truth evidence: {details}")
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = compare_pose_reference(reference, candidate)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
