"""Deterministic acceptance criteria for the diagonal PCMD probe."""

from __future__ import annotations

from dataclasses import dataclass

from .models import MotionDelta, PilotingCommand


@dataclass(frozen=True)
class DiagonalCriteria:
    min_forward_m: float = 0.2
    min_up_m: float = 0.2
    max_lateral_m: float = 0.2
    max_yaw_deg: float = 5.0

    def __post_init__(self) -> None:
        if min(self.min_forward_m, self.min_up_m, self.max_lateral_m, self.max_yaw_deg) < 0:
            raise ValueError("all criteria must be non-negative")


@dataclass(frozen=True)
class DiagonalAssessment:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class DirectionalCriteria:
    """Fixed acceptance gates for one body-frame PCMD direction."""

    min_commanded_axis_m: float = 0.2
    max_uncommanded_axis_m: float = 0.2
    max_yaw_deg: float = 5.0

    def __post_init__(self) -> None:
        if (
            min(
                self.min_commanded_axis_m,
                self.max_uncommanded_axis_m,
                self.max_yaw_deg,
            )
            < 0
        ):
            raise ValueError("all criteria must be non-negative")


def assess_diagonal_motion(
    delta: MotionDelta,
    criteria: DiagonalCriteria | None = None,
) -> DiagonalAssessment:
    """Assess forward/up motion without treating point count or speed as success."""
    if criteria is None:
        criteria = DiagonalCriteria()
    failures: list[str] = []
    if delta.forward_m < criteria.min_forward_m:
        failures.append("forward displacement")
    if delta.up_m < criteria.min_up_m:
        failures.append("upward displacement")
    if abs(delta.right_m) > criteria.max_lateral_m:
        failures.append("lateral drift")
    if abs(delta.yaw_change_deg) > criteria.max_yaw_deg:
        failures.append("yaw drift")
    return DiagonalAssessment(passed=not failures, failures=tuple(failures))


def assess_directional_motion(
    delta: MotionDelta,
    command: PilotingCommand,
    criteria: DirectionalCriteria | None = None,
) -> DiagonalAssessment:
    """Verify signs and minimum movement for arbitrary roll/pitch/gaz PCMD.

    ``yaw`` must remain zero for this suite: a yaw command would rotate the body
    frame while testing translation and invalidate a direct sign comparison.
    """

    if criteria is None:
        criteria = DirectionalCriteria()
    if command.yaw != 0:
        raise ValueError("direction sweep requires yaw=0")
    axes = (
        ("right", "left", command.roll, delta.right_m),
        ("forward", "backward", command.pitch, delta.forward_m),
        ("up", "down", command.gaz, delta.up_m),
    )
    failures: list[str] = []
    for positive_name, negative_name, command_value, displacement in axes:
        if command_value > 0 and displacement < criteria.min_commanded_axis_m:
            failures.append(f"{positive_name} displacement")
        elif command_value < 0 and displacement > -criteria.min_commanded_axis_m:
            failures.append(f"{negative_name} displacement")
        elif command_value == 0 and abs(displacement) > criteria.max_uncommanded_axis_m:
            failures.append(f"{positive_name}/{negative_name} drift")
    if abs(delta.yaw_change_deg) > criteria.max_yaw_deg:
        failures.append("yaw drift")
    return DiagonalAssessment(passed=not failures, failures=tuple(failures))
