"""Manual flight needs no localization review; autonomous flight needs all of it.

Operator policy, 2026-08-06: in MANUAL the pilot is the control loop and
localization is display-only. Switching to AUTONOMOUS makes localization the
control input, so every localization safety condition must hold first.
"""
from __future__ import annotations

import pytest

import runtime_safety as rs


def _good() -> dict:
    """A snapshot in which autonomy may legitimately be armed."""
    return {
        "autonomous_locked": False,
        "autonomous_approval_valid": True,
        "localizer_ready": True,
        "profile_verified": True,
        "zoom_paused": False,
        "loc_state": "TRACK",
        "pose_age_s": 0.05,
        "max_pose_age_s": 0.5,
        "inliers": 250,
        "min_inliers": 60,
        "reproj_rms": 1.3,
        "max_reproj_rms": 6.0,
        "consecutive_good_fixes": 3,
    }


def test_all_localization_conditions_good_allows_arming():
    assert rs.autonomous_arming_blockers(_good()) == []


@pytest.mark.parametrize("field,bad,expect", [
    ("autonomous_locked", True, "locked"),
    ("autonomous_approval_valid", False, "approval"),
    ("localizer_ready", False, "localizer worker"),
    ("profile_verified", False, "profile"),
    ("zoom_paused", True, "zoom"),
    ("loc_state", "WEAK_TRACK", "not TRACK"),
    ("loc_state", "LOST", "not TRACK"),
    ("loc_state", "BOOT_INIT", "not TRACK"),
    ("pose_age_s", 2.0, "stale"),
    ("inliers", 10, "inliers"),
    ("reproj_rms", 9.9, "reprojection"),
    ("consecutive_good_fixes", 1, "consecutive"),
])
def test_each_localization_condition_blocks_autonomy(field, bad, expect):
    snap = _good()
    snap[field] = bad
    blockers = rs.autonomous_arming_blockers(snap)
    assert blockers, f"{field}={bad!r} did not block autonomy"
    assert any(expect in b for b in blockers), blockers


@pytest.mark.parametrize("missing", [
    "pose_age_s", "inliers", "reproj_rms", "consecutive_good_fixes",
    "loc_state", "localizer_ready", "profile_verified",
    "autonomous_locked", "autonomous_approval_valid",
])
def test_unknown_evidence_fails_closed(missing):
    """A gate that cannot see the evidence must not conclude the evidence is good."""
    snap = _good()
    del snap[missing]
    assert rs.autonomous_arming_blockers(snap), f"missing {missing} did not block"


def test_empty_snapshot_blocks_everything():
    assert len(rs.autonomous_arming_blockers({})) >= 6


@pytest.mark.parametrize("field", [
    "pose_age_s", "max_pose_age_s", "inliers", "min_inliers",
    "reproj_rms", "max_reproj_rms", "consecutive_good_fixes",
])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_evidence_fails_closed(field, bad):
    """NaN/Inf must read as UNKNOWN, never as a satisfied threshold.

    A non-finite number silently wins every comparison it is asked to lose
    (NaN > x and NaN < x are both False; -inf passes any upper bound). If such a
    value reached the gate as a real reading, autonomy would arm on evidence
    nobody actually has. _num() rejects them, and this pins that behaviour on
    BOTH sides of each comparison -- the measured value and its own limit.
    """
    snap = _good()
    snap[field] = bad
    assert rs.autonomous_arming_blockers(snap), (
        f"{field}={bad!r} did not block autonomy"
    )


def test_bool_is_not_accepted_as_a_numeric_reading():
    """True is 1 in Python; a bool where a measurement belongs is a bug, not a 1."""
    snap = _good()
    snap["inliers"] = True
    assert any("inlier" in b for b in rs.autonomous_arming_blockers(snap))


def test_manual_flight_is_never_gated_on_localization():
    """Explicit operator policy: manual flight must not require localization."""
    for snap in ({}, _good(), {"loc_state": "LOST", "localizer_ready": False,
                               "pose_age_s": 99.0, "inliers": 0}):
        assert rs.manual_flight_blockers(snap) == [], (
            "manual flight became gated on localization; operator policy is that "
            "the pilot is the control loop and localization is display-only"
        )
