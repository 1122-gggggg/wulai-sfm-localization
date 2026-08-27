#!/usr/bin/env python3
"""Pose-guided vs MegaLoc-every-frame replay.

Current flight logs do not pair query frames with AttitudeChanged, so this
entry refuses to invent a benchmark. Unit tests cover SE(3), anchors, and
fail-closed prediction labeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=None)
    args = parser.parse_args()
    report = {
        "schema": "pose-guided-replay/v1",
        "runnable": False,
        "reason": (
            "No paired query-frame + fused-attitude corpus is checked in. "
            "Real session telemetry JSONL omits att_roll/pitch/yaw and has no images."
        ),
        "required_for_baseline_ab": [
            "query RGB frames with capture_stamp",
            "AttitudeChanged at those stamps",
            "optional SpeedChanged NED",
            "GlueMap EDM bundle used for the same flight",
        ],
        "session": None if args.session is None else str(args.session),
        "metrics": {},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
