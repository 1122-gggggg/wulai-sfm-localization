#!/usr/bin/env python3
"""Resolve a component mission selection and report startup readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "控制介面程式"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from mission_manifest import ManifestError  # noqa: E402
from mission_resolver import resolve_mission  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", help="sfm-mission-selection/v1 JSON")
    parser.add_argument(
        "--require",
        choices=("valid", "localization", "flight"),
        default="valid",
        help="exit nonzero unless this readiness level is satisfied",
    )
    parser.add_argument(
        "--materialize-site-profile",
        action="store_true",
        help="write the immutable schema-v2 compatibility profile under 執行環境",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        mission = resolve_mission(args.selection, workspace_root=ROOT)
        profile_path = None
        if args.materialize_site_profile:
            profile_path = mission.materialize_legacy_site_profile(
                ROOT / "執行環境" / "mission_snapshots"
            )
        report = {
            "selection": str(mission.selection.source),
            "snapshot_id": mission.identity,
            "vehicle": mission.vehicle.identity,
            "site_id": mission.site.site_id,
            "map_revision_id": mission.map_revision.map_revision_id,
            "localizer": {
                "algorithm_id": mission.localizer.algorithm_id,
                "variant_id": mission.localizer.variant_id,
            },
            "route": None if mission.route is None else mission.route.identity,
            "localization": {
                "ready": mission.readiness.localization_ready,
                "errors": list(mission.readiness.localization_errors),
            },
            "flight": {
                "ready": mission.readiness.flight_ready,
                "errors": list(mission.readiness.flight_errors),
            },
            "compatibility_site_profile": (None if profile_path is None else str(profile_path)),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    except (ManifestError, OSError, ValueError) as exc:
        print(f"mission resolution failed: {exc}", file=sys.stderr)
        return 2

    if args.require == "localization" and not mission.readiness.localization_ready:
        return 1
    if args.require == "flight" and not mission.readiness.flight_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
