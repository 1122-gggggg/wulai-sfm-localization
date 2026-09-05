#!/usr/bin/env python3
"""Validate shipped site-profile schemas, assets, digests, and approvals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from site_profile import (
    ROUTE_EDITOR_AUTO_APPROVAL_NOTE,
    SCHEMA_VERSION,
    flight_readiness_errors,
    load_site_profile,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
PROFILES = ROOT / "site_profiles"
CANONICAL_PROFILE_DIR = WORKSPACE_ROOT / "地圖檔" / "場域"
TEMPLATE_NAMES = {"example_site_edm.json"}
# A profile may be AUTO-approved only once a route has been redrawn on the
# current map. Operator decision 2026-09-05: rather than hand-editing a list of
# sites for every route, accept the approval the route editor itself recorded --
# a profile is expected to be approved exactly when it carries the editor's AUTO
# approval note. That is not a blanket pass: an approved profile must also
# survive flight_readiness_errors() below, and _validate_asset_digests re-hashes
# route_json, so a route edited outside the editor, or a profile approved by
# hand, still fails.
#
# This authorizes AUTO through the site profile only -- the in-app AUTO and
# simulated-route runs. It is NOT flight authorization for the real-aircraft
# entry point: that path flies the profile materialized from the mission
# selection, whose flight.approved follows the localizer_quality receipt. No
# route-editor approval can unblock 一鍵啟動.sh.
def _expects_auto_approval(profile: object) -> bool:
    flight = getattr(profile, "flight", None)
    note = str(getattr(flight, "approval_note", "") or "").strip()
    return note == ROUTE_EDITOR_AUTO_APPROVAL_NOTE


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _validate_profile_fields(source: Path, raw: dict, profile: object) -> list[str]:
    failures: list[str] = []
    if profile.schema_version != SCHEMA_VERSION:
        failures.append(
            f"{source.name}: schema v{profile.schema_version} is not v{SCHEMA_VERSION}"
        )
    if "map_units_per_meter" in raw.get("flight", {}):
        failures.append(f"{source.name}: metric map scale is prohibited")
    approved = bool(profile.flight and profile.flight.approved)
    expected_approved = _expects_auto_approval(profile)
    if approved is not expected_approved:
        failures.append(f"{source.name}: flight.approved must be {expected_approved}")
    if expected_approved:
        failures.extend(
            f"{source.name}: AUTO readiness: {error}"
            for error in flight_readiness_errors(profile)
        )
    return failures


def _validate_asset_digests(source: Path, profile: object) -> tuple[list[str], int]:
    failures: list[str] = []
    checked = 0
    for key, path in (
        ("map_ply", profile.map_ply),
        ("localization_bundle", profile.localization_bundle),
        ("route_json", profile.route_json),
        ("map_reference_poses", profile.map_reference_poses),
        ("localizer_profile", profile.localizer_profile),
        ("reference_index", profile.reference_index),
        ("poles_json", profile.poles_json),
        ("map_align", profile.map_align),
        (
            "site_alignment",
            None if profile.pose_chain is None else profile.pose_chain.site_alignment,
        ),
        (
            "camera_body_extrinsic",
            None
            if profile.pose_chain is None
            else profile.pose_chain.camera_body_extrinsic,
        ),
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
                f"{source.name}: {key} SHA-256 mismatch "
                f"expected={expected} actual={actual}"
            )
    return failures, checked


def _validate_hardware_digests(source: Path, profile: object) -> tuple[list[str], int]:
    hardware = profile.hardware_approval
    if hardware is None:
        return [], 0
    failures: list[str] = []
    checked = 0
    for key, path, expected in (
        ("hardware_approval.receipt", hardware.receipt, hardware.sha256),
        (
            "hardware_approval.signature",
            hardware.signature,
            hardware.signature_sha256,
        ),
        (
            "hardware_approval.trust_store",
            hardware.trust_store,
            hardware.trust_store_sha256,
        ),
    ):
        if path is None and expected is None:
            continue
        if path is None or expected is None or not path.is_file():
            failures.append(f"{source.name}: hashed asset {key} is missing")
            continue
        actual = _digest(path)
        checked += 1
        if actual != expected:
            failures.append(
                f"{source.name}: {key} SHA-256 mismatch "
                f"expected={expected} actual={actual}"
            )
    return failures, checked


def _validate_one(source: Path) -> tuple[dict[str, object] | None, list[str]]:
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        profile = load_site_profile(
            source, validate_files=source.name not in TEMPLATE_NAMES
        )
    except Exception as exc:
        return None, [f"{source.name}: {exc}"]

    failures = _validate_profile_fields(source, raw, profile)
    asset_failures, asset_count = _validate_asset_digests(source, profile)
    hardware_failures, hardware_count = _validate_hardware_digests(source, profile)
    failures.extend(asset_failures)
    failures.extend(hardware_failures)
    return {
        "profile": str(source.relative_to(WORKSPACE_ROOT)),
        "site_id": profile.site_id,
        "schema_version": profile.schema_version,
        "flight_approved": bool(profile.flight and profile.flight.approved),
        "digests_checked": asset_count + hardware_count,
        "template": source.name in TEMPLATE_NAMES,
    }, failures


def validate() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    sources = [
        *PROFILES.glob("*.json"),
        *CANONICAL_PROFILE_DIR.glob("*/site_profile.json"),
    ]
    for source in sorted(sources):
        row, profile_failures = _validate_one(source)
        failures.extend(profile_failures)
        if row is not None:
            rows.append(row)
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
