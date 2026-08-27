from anafi_pcmd_sim.assessment import (
    DiagonalCriteria,
    DirectionalCriteria,
    assess_diagonal_motion,
    assess_directional_motion,
)
from anafi_pcmd_sim.models import MotionDelta, PilotingCommand


def test_accepts_forward_up_motion_with_small_lateral_drift() -> None:
    result = assess_diagonal_motion(
        MotionDelta(forward_m=1.2, right_m=0.05, up_m=0.8, yaw_change_deg=1.0),
        DiagonalCriteria(min_forward_m=0.2, min_up_m=0.2, max_lateral_m=0.2, max_yaw_deg=5.0),
    )

    assert result.passed is True
    assert result.failures == ()


def test_rejects_forward_only_motion() -> None:
    result = assess_diagonal_motion(
        MotionDelta(forward_m=1.2, right_m=0.0, up_m=0.01, yaw_change_deg=0.0),
        DiagonalCriteria(min_forward_m=0.2, min_up_m=0.2, max_lateral_m=0.2, max_yaw_deg=5.0),
    )

    assert result.passed is False
    assert "upward displacement" in result.failures


def test_rejects_uncontrolled_yaw_drift() -> None:
    result = assess_diagonal_motion(
        MotionDelta(forward_m=1.2, right_m=0.0, up_m=0.8, yaw_change_deg=8.0),
        DiagonalCriteria(min_forward_m=0.2, min_up_m=0.2, max_lateral_m=0.2, max_yaw_deg=5.0),
    )

    assert result.passed is False
    assert "yaw drift" in result.failures


def test_directional_assessment_checks_the_sign_of_all_commanded_axes() -> None:
    command = PilotingCommand(roll=20, pitch=-20, yaw=0, gaz=-20)
    criteria = DirectionalCriteria(
        min_commanded_axis_m=0.2,
        max_uncommanded_axis_m=0.2,
        max_yaw_deg=5.0,
    )

    accepted = assess_directional_motion(
        MotionDelta(forward_m=-0.6, right_m=0.5, up_m=-0.4, yaw_change_deg=1.0),
        command,
        criteria,
    )
    assert accepted.passed is True

    rejected = assess_directional_motion(
        MotionDelta(forward_m=0.6, right_m=0.5, up_m=-0.4, yaw_change_deg=1.0),
        command,
        criteria,
    )
    assert rejected.passed is False
    assert "backward displacement" in rejected.failures
