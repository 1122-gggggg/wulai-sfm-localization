#!/usr/bin/env python3
"""Launch the operator UI from one verified mission selection snapshot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from mission_manifest import ManifestError
from mission_resolver import evaluation_only_admission, resolve_mission
from workspace_layout import workspace_from_file


WORKSPACE = workspace_from_file(__file__)
OPERATOR = WORKSPACE.operator_interface / "flight_operator_app.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", help="sfm-mission-selection/v1 JSON")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="resolve and materialize the snapshot without launching the UI",
    )
    return parser


def _has_direct_site_profile(arguments: list[str]) -> bool:
    return any(
        argument == "--site-profile" or argument.startswith("--site-profile=")
        for argument in arguments
    )


def main(argv: list[str] | None = None) -> int:
    args, operator_args = _parser().parse_known_args(argv)
    if operator_args[:1] == ["--"]:
        operator_args.pop(0)
    if _has_direct_site_profile(operator_args):
        print(
            "mission launcher owns --site-profile; pass SFM_MISSION_SELECTION",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("SFM_SITE_PROFILE", "").strip():
        print(
            "mission launcher rejects existing SFM_SITE_PROFILE; "
            "pass SFM_MISSION_SELECTION",
            file=sys.stderr,
        )
        return 2
    try:
        mission = resolve_mission(args.selection, workspace_root=WORKSPACE.root)
        evaluation_only, waived = evaluation_only_admission(mission.readiness)
        if not mission.readiness.localization_ready and not evaluation_only:
            raise ManifestError(
                "mission is not localization-ready: "
                + "; ".join(mission.readiness.localization_errors)
            )
        profile = mission.materialize_legacy_site_profile(WORKSPACE.runtime / "mission_snapshots")
    except (ManifestError, OSError, ValueError) as exc:
        print(f"mission launch rejected: {exc}", file=sys.stderr)
        return 2

    if evaluation_only:
        print(
            "EVALUATION-ONLY session: localization runs for measurement only. "
            "Waived: " + "; ".join(waived),
            file=sys.stderr,
        )

    report = {
        "snapshot_id": mission.identity,
        "site_profile": str(profile),
        "localization_ready": not evaluation_only,
        "evaluation_only": evaluation_only,
        "evaluation_waived_errors": list(waived),
        "flight_ready": mission.readiness.flight_ready,
        "flight_errors": list(mission.readiness.flight_errors),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if args.check_only:
        return 0

    environment = os.environ.copy()
    environment.pop("SFM_SITE_PROFILE", None)
    environment["SFM_MISSION_SELECTION"] = str(mission.selection.source)
    return subprocess.call(
        [sys.executable, str(OPERATOR), "--site-profile", str(profile), *operator_args],
        cwd=str(WORKSPACE.root),
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
