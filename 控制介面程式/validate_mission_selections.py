#!/usr/bin/env python3
"""Validate every shipped mission selection and its complete artifact graph."""

from __future__ import annotations

import json
from pathlib import Path

from mission_manifest import ManifestError
from mission_resolver import resolve_mission


CONTROL = Path(__file__).resolve().parent
WORKSPACE = CONTROL.parent
SELECTIONS = CONTROL / "mission_selections"


def validate() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    sources = sorted(SELECTIONS.glob("*.json"))
    if not sources:
        return rows, ["no mission selections found"]
    for source in sources:
        try:
            mission = resolve_mission(source, workspace_root=WORKSPACE)
            mission.verify_unchanged()
        except (ManifestError, OSError, ValueError) as exc:
            failures.append(f"{source.name}: {exc}")
            continue
        if not mission.readiness.localization_ready:
            failures.append(
                f"{source.name}: localization: " + "; ".join(mission.readiness.localization_errors)
            )
        rows.append(
            {
                "selection": source.name,
                "snapshot_id": mission.identity,
                "vehicle": mission.vehicle.identity,
                "site_id": mission.site.site_id,
                "map_revision_id": mission.map_revision.map_revision_id,
                "localizer": mission.localizer.algorithm_id,
                "localization_ready": mission.readiness.localization_ready,
                "flight_ready": mission.readiness.flight_ready,
                "flight_errors": list(mission.readiness.flight_errors),
            }
        )
    return rows, failures


def main() -> int:
    rows, failures = validate()
    print(
        json.dumps(
            {"selections": rows, "failures": failures, "ok": not failures},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
