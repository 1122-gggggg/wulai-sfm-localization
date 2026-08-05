#!/usr/bin/env python3
"""Validate shipped site-profile schemas, assets, digests, and safety locks."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from site_profile import SCHEMA_VERSION, load_site_profile


ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "site_profiles"
TEMPLATE_NAMES = {"example_site_edm.json"}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for source in sorted(PROFILES.glob("*.json")):
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            profile = load_site_profile(
                source, validate_files=source.name not in TEMPLATE_NAMES
            )
        except Exception as exc:
            failures.append(f"{source.name}: {exc}")
            continue
        if profile.schema_version != SCHEMA_VERSION:
            failures.append(
                f"{source.name}: schema v{profile.schema_version} is not v{SCHEMA_VERSION}"
            )
        if "map_units_per_meter" in raw.get("flight", {}):
            failures.append(f"{source.name}: metric map scale is prohibited")
        if profile.flight is None or profile.flight.approved:
            failures.append(f"{source.name}: flight.approved must remain false")

        checked = 0
        for key, path in (
            ("map_ply", profile.map_ply),
            ("localization_bundle", profile.localization_bundle),
            ("route_json", profile.route_json),
            ("map_reference_poses", profile.map_reference_poses),
            ("localizer_profile", profile.localizer_profile),
            ("poles_json", profile.poles_json),
        ):
            expected = getattr(profile.asset_sha256, key)
            if expected is None:
                continue
            if path is None or not path.is_file():
                failures.append(f"{source.name}: hashed asset {key} is missing")
                continue
            actual = _digest(path)
            checked += 1
            if actual != expected:
                failures.append(
                    f"{source.name}: {key} SHA-256 mismatch expected={expected} actual={actual}"
                )
        rows.append(
            {
                "profile": source.name,
                "site_id": profile.site_id,
                "schema_version": profile.schema_version,
                "flight_approved": bool(profile.flight and profile.flight.approved),
                "digests_checked": checked,
                "template": source.name in TEMPLATE_NAMES,
            }
        )
    return rows, failures


def main() -> int:
    rows, failures = validate()
    print(
        json.dumps(
            {"profiles": rows, "failures": failures, "ok": not failures},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
