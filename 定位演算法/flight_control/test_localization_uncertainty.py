from __future__ import annotations

import pytest

from localization_uncertainty import (
    LocalizationState,
    decide_localization_transition,
)


@pytest.mark.parametrize(
    "state",
    (
        {"uncertain_since": 10.0},
        {"uncertain_land_after": 4.0},
        {"recovery_good_fixes": -1},
    ),
)
def test_localization_state_rejects_inconsistent_values(state):
    with pytest.raises(ValueError):
        LocalizationState(**state)


@pytest.mark.parametrize(
    (
        "label",
        "state",
        "now",
        "fresh",
        "low_confidence",
        "expected_action",
        "expected_since",
        "expected_land_after",
        "expected_recovery_fixes",
        "expected_waited",
    ),
    [
        (
            "first lost fix starts LOST timer",
            LocalizationState(),
            10.0,
            False,
            False,
            "uncertain_hover",
            10.0,
            4.0,
            0,
            0.0,
        ),
        (
            "first weak fix starts WEAK timer",
            LocalizationState(),
            10.0,
            True,
            True,
            "uncertain_hover",
            10.0,
            8.0,
            0,
            0.0,
        ),
        (
            "weak and lost keep shortest fail-safe",
            LocalizationState(10.0, 8.0, 0),
            11.0,
            False,
            False,
            "uncertain_hover",
            10.0,
            4.0,
            0,
            1.0,
        ),
        (
            "LOST timer reaches LAND",
            LocalizationState(10.0, 4.0, 0),
            14.0,
            False,
            False,
            "land",
            10.0,
            4.0,
            0,
            4.0,
        ),
        (
            "WEAK timer reaches LAND",
            LocalizationState(10.0, 8.0, 0),
            18.0,
            True,
            True,
            "land",
            10.0,
            8.0,
            0,
            8.0,
        ),
        (
            "first good fix after uncertainty confirms in hover",
            LocalizationState(10.0, 8.0, 0),
            11.0,
            True,
            False,
            "recovery_hover",
            10.0,
            8.0,
            1,
            1.0,
        ),
        (
            "second good fix resumes TRACK",
            LocalizationState(10.0, 8.0, 1),
            12.0,
            True,
            False,
            "track",
            None,
            None,
            0,
            None,
        ),
        (
            "good fix with no uncertainty tracks immediately",
            LocalizationState(),
            10.0,
            True,
            False,
            "track",
            None,
            None,
            0,
            None,
        ),
    ],
)
def test_localization_transition_table(
    label,
    state,
    now,
    fresh,
    low_confidence,
    expected_action,
    expected_since,
    expected_land_after,
    expected_recovery_fixes,
    expected_waited,
):
    decision = decide_localization_transition(
        state,
        now=now,
        fresh=fresh,
        low_confidence=low_confidence,
        weak_hover_land_s=8.0,
        lost_land_s=4.0,
        recovery_good_fixes_required=2,
    )

    assert decision.action == expected_action, label
    assert decision.state.uncertain_since == expected_since, label
    assert decision.state.uncertain_land_after == expected_land_after, label
    assert decision.state.recovery_good_fixes == expected_recovery_fixes, label
    assert decision.waited_s == expected_waited, label
