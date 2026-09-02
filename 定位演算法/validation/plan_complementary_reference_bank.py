#!/usr/bin/env python3
"""Plan a complementary EDM reference bank without crossing map frames.

This is a shadow planning tool. It never writes a bundle or authorizes deployment.
Two telemetry streams may contribute references only when their site profiles name
the same coordinate frame. Different-frame inputs produce a blocked report instead
of silently concatenating incompatible 3D anchors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _finite_rate(successes: int, total: int) -> float:
    return 0.0 if total <= 0 else float(successes / total)


def summarize_windows(rows: list[dict[str, Any]], width: int) -> dict[int, dict[str, Any]]:
    if width <= 0:
        raise ValueError("window width must be positive")
    windows: dict[int, dict[str, Any]] = {}
    for row in rows:
        sequence = row.get("display_seq")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            continue
        start = sequence // width * width
        bucket = windows.setdefault(
            start,
            {"results": 0, "successes": 0, "lost": 0, "successful_refs": set()},
        )
        bucket["results"] += 1
        if row.get("success") is True:
            bucket["successes"] += 1
            refs = row.get("refs")
            if isinstance(refs, list):
                bucket["successful_refs"].update(
                    value for value in refs if isinstance(value, str) and value
                )
        if row.get("mode") == "LOST":
            bucket["lost"] += 1
    return windows


def build_plan(
    *,
    frame_a: str,
    rows_a: list[dict[str, Any]],
    frame_b: str,
    rows_b: list[dict[str, Any]],
    window_width: int = 250,
    independent_holdout: str | None = None,
) -> dict[str, Any]:
    first = summarize_windows(rows_a, window_width)
    second = summarize_windows(rows_b, window_width)
    comparisons = []
    selected_refs: set[str] = set()
    reference_evidence = False
    for start in sorted(set(first) | set(second)):
        a = first.get(start, {"results": 0, "successes": 0, "lost": 0, "successful_refs": set()})
        b = second.get(start, {"results": 0, "successes": 0, "lost": 0, "successful_refs": set()})
        rate_a = _finite_rate(a["successes"], a["results"])
        rate_b = _finite_rate(b["successes"], b["results"])
        preferred = "A" if rate_a >= rate_b else "B"
        chosen = a if preferred == "A" else b
        if chosen["successful_refs"]:
            reference_evidence = True
            selected_refs.update(chosen["successful_refs"])
        comparisons.append(
            {
                "window": [start, start + window_width - 1],
                "A": {
                    "results": a["results"],
                    "success_rate": rate_a,
                    "lost_rate": _finite_rate(a["lost"], a["results"]),
                },
                "B": {
                    "results": b["results"],
                    "success_rate": rate_b,
                    "lost_rate": _finite_rate(b["lost"], b["results"]),
                },
                "preferred": preferred,
                "success_rate_margin": abs(rate_a - rate_b),
            }
        )

    same_frame = bool(frame_a) and frame_a == frame_b
    if not same_frame:
        status = "BLOCKED_DIFFERENT_COORDINATE_FRAMES"
        selected_refs.clear()
        next_action = (
            "Register and triangulate both image/reference sets inside one frozen "
            "reconstruction; do not concatenate the existing bundles."
        )
    elif not reference_evidence:
        status = "NEEDS_REFERENCE_TELEMETRY"
        next_action = "Replay both candidates with localization telemetry that includes refs."
    elif not independent_holdout:
        status = "NEEDS_INDEPENDENT_HOLDOUT"
        next_action = "Provide a mapping-disjoint replay before promotion."
    else:
        status = "SHADOW_PLAN_READY"
        next_action = "Rebuild EDM anchors for candidate_reference_names and run the holdout."

    return {
        "schema": "sfm-complementary-reference-bank-plan/v1",
        "status": status,
        "coordinate_frames": {"A": frame_a, "B": frame_b, "same": same_frame},
        "window_width": window_width,
        "window_comparisons": comparisons,
        "reference_evidence_available": reference_evidence,
        "candidate_reference_names": sorted(selected_refs),
        "independent_holdout": independent_holdout,
        "deployment_authorized": False,
        "next_action": next_action,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _profile_frame(path: Path) -> str:
    control = Path(__file__).resolve().parents[2] / "控制介面程式"
    if str(control) not in sys.path:
        sys.path.insert(0, str(control))
    from site_profile import load_site_profile

    profile = load_site_profile(path)
    if profile.coordinate_frame is None:
        raise ValueError(f"profile has no coordinate frame: {path}")
    return profile.coordinate_frame.id


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-a", type=Path, required=True)
    parser.add_argument("--telemetry-a", type=Path, required=True)
    parser.add_argument("--profile-b", type=Path, required=True)
    parser.add_argument("--telemetry-b", type=Path, required=True)
    parser.add_argument("--window-width", type=int, default=250)
    parser.add_argument("--independent-holdout", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan(
        frame_a=_profile_frame(args.profile_a),
        rows_a=_read_jsonl(args.telemetry_a),
        frame_b=_profile_frame(args.profile_b),
        rows_b=_read_jsonl(args.telemetry_b),
        window_width=args.window_width,
        independent_holdout=args.independent_holdout or None,
    )
    _write_json_atomic(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0 if plan["status"] == "SHADOW_PLAN_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
