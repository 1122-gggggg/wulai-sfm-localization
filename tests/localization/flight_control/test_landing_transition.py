from __future__ import annotations

# Source modules are supplied by the repository's pytest pythonpath.
import pytest

from landing_transition import decide_route_completion_landing


@pytest.mark.parametrize(
    (
        "label",
        "action",
        "sample",
        "expected_outcome",
        "expected_speed",
        "expected_reason",
        "expected_log_fields",
    ),
    [
        (
            "ABORT bypasses ground-speed gate",
            "ABORT",
            None,
            "land",
            None,
            "pending inspection abort -> land",
            (),
        ),
        (
            "LAND unavailable speed hovers",
            "LAND",
            None,
            "hover",
            None,
            "route complete; ground speed unavailable -> hover",
            (("ground_speed_mps", None),),
        ),
        (
            "LAND invalid speed hovers",
            "LAND",
            (float("nan"), 10.0),
            "hover",
            None,
            "route complete; ground speed invalid -> hover",
            (("ground_speed_mps", None),),
        ),
        (
            "LAND stale speed hovers",
            "LAND",
            (0.09, 9.49),
            "hover",
            0.09,
            "route complete; ground speed stale (0.51s) -> hover",
            (("ground_speed_mps", 0.09),),
        ),
        (
            "LAND future speed hovers",
            "LAND",
            (0.09, 10.1),
            "hover",
            0.09,
            "route complete; ground speed timestamp is in the future -> hover",
            (("ground_speed_mps", 0.09),),
        ),
        (
            "LAND high speed hovers",
            "LAND",
            (0.11, 10.0),
            "hover",
            0.11,
            "route complete; ground speed 0.110 m/s > 0.100 m/s -> hover",
            (("ground_speed_mps", 0.11),),
        ),
        (
            "LAND fresh low speed lands",
            "LAND",
            (0.09, 10.0),
            "land",
            0.09,
            "route complete -> land",
            (("ground_speed_mps", 0.09),),
        ),
    ],
)
def test_route_completion_landing_transition_table(
    label,
    action,
    sample,
    expected_outcome,
    expected_speed,
    expected_reason,
    expected_log_fields,
):
    transition = decide_route_completion_landing(action, sample, now=10.0)

    assert transition.outcome == expected_outcome, label
    assert transition.ground_speed_mps == expected_speed, label
    assert transition.reason == expected_reason, label
    assert transition.log_fields == expected_log_fields, label
