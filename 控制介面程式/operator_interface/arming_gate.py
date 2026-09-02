"""Manual vs autonomous authority.

Policy (operator decision, 2026-08-06): MANUAL flight does not require any
localization safety review -- the pilot is the control loop, and localization is
display-only. Switching to AUTONOMOUS makes localization the control input, so
every localization safety condition must hold before autonomy may be armed.

This gate is deliberately fail-closed and takes a plain snapshot dict so it can be
unit-tested without a drone, a worker or a map.

Split out of the former runtime_safety.py (D2). See session_logs.py, disk_policy.py
and network_policy.py for the other former responsibilities of that file.
"""
from __future__ import annotations

import math
from typing import Any


#: Localization states from which autonomy may be armed. BOOT_INIT / WEAK_TRACK /
#: LOST all mean the fix is provisional or degraded.
AUTONOMY_OK_LOC_STATES = frozenset({"TRACK"})

#: Consecutive good fixes required before autonomy trusts the pose. Mirrors the
#: flight loop's RECOVERY_GOOD_FIXES: one lucky frame is not a lock.
AUTONOMY_MIN_CONSECUTIVE_FIXES = 2


def autonomous_approval_blockers(snapshot: dict[str, Any]) -> list[str]:
    """Return profile/external-approval reasons AUTO may not even start."""
    blockers: list[str] = []
    if snapshot.get("autonomous_locked", True):
        blockers.append("autonomous route flight is locked pending external approval")
    if not snapshot.get("autonomous_approval_valid", False):
        blockers.append("autonomous route approval is not valid")
    if not snapshot.get("profile_verified", False):
        blockers.append(
            "localizer profile was not verified (running on unpinned defaults)"
        )
    return blockers


def _finite_snapshot_number(snapshot: dict[str, Any], key: str) -> float | None:
    value = snapshot.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_limit_blocker(
    snapshot: dict[str, Any],
    *,
    value_key: str,
    limit_key: str,
    unknown_message: str,
    violation_message: str,
    minimum: bool = False,
) -> str | None:
    value = _finite_snapshot_number(snapshot, value_key)
    limit = _finite_snapshot_number(snapshot, limit_key)
    if value is None or limit is None:
        return unknown_message
    violates = value < limit if minimum else value > limit
    if violates:
        return violation_message.format(value=value, limit=limit)
    return None


def _consecutive_fix_blocker(snapshot: dict[str, Any]) -> str | None:
    fixes = _finite_snapshot_number(snapshot, "consecutive_good_fixes")
    if fixes is None or fixes < AUTONOMY_MIN_CONSECUTIVE_FIXES:
        return (
            f"needs {AUTONOMY_MIN_CONSECUTIVE_FIXES} consecutive good fixes "
            f"(have {'unknown' if fixes is None else int(fixes)})"
        )
    return None


def autonomous_arming_blockers(snapshot: dict[str, Any]) -> list[str]:
    """Return every reason autonomy may NOT be armed. Empty list == may arm.

    Unknown/missing fields are treated as blocking: a gate that cannot see the
    evidence must not conclude the evidence is good.
    """
    blockers = autonomous_approval_blockers(snapshot)

    if not snapshot.get("localizer_ready", False):
        blockers.append("localizer worker is not ready")
    if snapshot.get("zoom_paused", False):
        blockers.append("localization is paused by an uncalibrated camera zoom")

    state = str(snapshot.get("loc_state") or "")
    if state not in AUTONOMY_OK_LOC_STATES:
        blockers.append(f"localization state {state or 'unknown'!s} is not TRACK")

    metric_blockers = (
        _metric_limit_blocker(
            snapshot,
            value_key="pose_age_s",
            limit_key="max_pose_age_s",
            unknown_message="pose age is unknown",
            violation_message="pose is stale ({value:.2f}s > {limit:.2f}s)",
        ),
        _metric_limit_blocker(
            snapshot,
            value_key="inliers",
            limit_key="min_inliers",
            unknown_message="PnP inlier count is unknown",
            violation_message="PnP inliers {value:.0f} < {limit:.0f}",
            minimum=True,
        ),
        _metric_limit_blocker(
            snapshot,
            value_key="reproj_rms",
            limit_key="max_reproj_rms",
            unknown_message="reprojection error is unknown",
            violation_message="reprojection RMS {value:.2f} > {limit:.2f}",
        ),
        _consecutive_fix_blocker(snapshot),
    )
    blockers.extend(blocker for blocker in metric_blockers if blocker is not None)
    return blockers


def manual_flight_blockers(snapshot: dict[str, Any]) -> list[str]:
    """Manual flight is NOT gated on localization — by explicit operator policy.

    Present so the asymmetry is expressed in code rather than left implicit, and so a
    regression that starts gating manual flight on localization is caught by a test.
    """
    return []


__all__ = [
    "AUTONOMY_MIN_CONSECUTIVE_FIXES",
    "AUTONOMY_OK_LOC_STATES",
    "autonomous_approval_blockers",
    "autonomous_arming_blockers",
    "manual_flight_blockers",
]
