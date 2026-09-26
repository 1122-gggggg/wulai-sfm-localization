#!/usr/bin/env python3
"""Summarize distinct, synchronized relocalization jobs from an offline capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def quantiles(values):
    values = [
        float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
    ]
    return {
        "count": len(values),
        **{
            name: float(np.percentile(values, q)) if values else None
            for name, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99), ("max_ms", 100))
        },
    }


def build_report(capture):
    jobs = capture.get("reloc_jobs", [])
    keys = sorted({key for row in jobs for key in row.get("stage_ms", {}) if key.endswith("_ms")})
    stages = {key: quantiles([row.get("stage_ms", {}).get(key) for row in jobs]) for key in keys}
    timed = [row for row in jobs if quantiles([row.get("runtime_ms")])["count"]]
    slowest = sorted(timed, key=lambda row: row["runtime_ms"], reverse=True)[:10]
    return {
        "schema": "sfm-reloc-profile/v1",
        "schedule": capture.get("schedule"),
        "frames": capture.get("frames"),
        "jobs": len(jobs),
        "runtime": quantiles([row.get("runtime_ms") for row in jobs]),
        "stages": stages,
        "slowest_jobs": slowest,
        "status_counts": capture.get("status_counts"),
        "video_sha256": capture.get("video_sha256"),
        "profile_sha256": capture.get("profile_sha256"),
        "bundle_sha256": capture.get("bundle_sha256"),
        "gpu": capture.get("gpu"),
        "versions": capture.get("versions"),
        "limitations": [
            "Scheduled offline capture is not concurrent UI end-to-end latency.",
            "edm_match_ms and lift_prepare_ms partition match_lift_ms; do not add them twice.",
            "Capture mode synchronizes each outer GPU job; runtime log timers remain host wall time.",
            "Missing stage samples are counted as missing, never as zero.",
        ],
    }


def session_capture(session):
    path = Path(session) / "localization.jsonl"
    if not path.is_file():
        path = Path(session) / "trajectory.jsonl"
    jobs, source_ages, missing_identity = {}, [], 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "pose_result":
                continue
            source_ages.append(row.get("source_stamp_age_at_ui_ms"))
            key = tuple(
                row.get(name)
                for name in ("reloc_source_epoch", "reloc_ordinal", "reloc_capture_stamp_mono")
            )
            if any(value is None for value in key):
                missing_identity += 1
                continue
            jobs.setdefault(
                key,
                {
                    "source_epoch": key[0],
                    "ordinal": key[1],
                    "capture_stamp": key[2],
                    "runtime_ms": row.get("reloc_ms"),
                    "stage_ms": {
                        name: row.get(field)
                        for name, field in (
                            ("retrieval_ms", "reloc_retrieval_ms"),
                            ("match_lift_ms", "reloc_match_lift_ms"),
                            ("edm_match_ms", "reloc_edm_match_ms"),
                            ("lift_prepare_ms", "reloc_lift_prepare_ms"),
                            ("reloc_pnp_ms", "reloc_pnp_ms"),
                        )
                    },
                },
            )
    return {
        "reloc_jobs": list(jobs.values()),
        "schedule": "recorded runtime, distinct reloc identity",
        "source_age_at_ui": quantiles(source_ages),
        "missing_identity_rows": missing_identity,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path)
    source.add_argument("--session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.capture is not None:
        data = args.capture.read_bytes()
        result = build_report(json.loads(data))
        result["capture_sha256"] = hashlib.sha256(data).hexdigest()
    else:
        capture = session_capture(args.session)
        result = build_report(capture)
        result.update(
            source_age_at_ui=capture["source_age_at_ui"],
            missing_identity_rows=capture["missing_identity_rows"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"jobs": result["jobs"], "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
