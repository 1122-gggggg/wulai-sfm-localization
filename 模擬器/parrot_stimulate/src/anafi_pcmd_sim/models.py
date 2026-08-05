"""Small, simulator-agnostic data objects for the PCMD experiment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


def _validate_pcmd_value(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not -100 <= value <= 100:
        raise ValueError(f"{name} must be in the inclusive range -100..100")


@dataclass(frozen=True)
class PilotingCommand:
    """Manual ANAFI PCMD components expressed in firmware percentages."""

    roll: int
    pitch: int
    yaw: int
    gaz: int

    def __post_init__(self) -> None:
        for name, value in (
            ("roll", self.roll),
            ("pitch", self.pitch),
            ("yaw", self.yaw),
            ("gaz", self.gaz),
        ):
            _validate_pcmd_value(name, value)

    @property
    def horizontal_enabled(self) -> bool:
        """Whether ARSDK's PCMD flag must enable roll/pitch control."""
        return bool(self.roll or self.pitch)

    @classmethod
    def zero(cls) -> PilotingCommand:
        return cls(roll=0, pitch=0, yaw=0, gaz=0)


@dataclass(frozen=True)
class Scenario:
    """One bounded PCMD test case."""

    name: str
    command: PilotingCommand
    duration_s: float
    settle_s: float = 2.0

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.settle_s < 0:
            raise ValueError("settle_s must not be negative")

    @classmethod
    def forward_up(
        cls,
        *,
        duration_s: float = 1.5,
        pitch: int = 20,
        gaz: int = 20,
        settle_s: float = 2.0,
    ) -> Scenario:
        return cls(
            name="forward-up",
            command=PilotingCommand(roll=0, pitch=pitch, yaw=0, gaz=gaz),
            duration_s=duration_s,
            settle_s=settle_s,
        )


def _direction_name(*, right: int, forward: int, up: int) -> str:
    labels = (
        "right" if right > 0 else "left" if right < 0 else "",
        "forward" if forward > 0 else "backward" if forward < 0 else "",
        "up" if up > 0 else "down" if up < 0 else "",
    )
    return "-".join(label for label in labels if label)


def all_direction_scenarios(
    *,
    duration_s: float = 1.5,
    magnitude: int = 20,
    settle_s: float = 2.0,
) -> tuple[Scenario, ...]:
    """Return the 26 non-zero body-frame movement directions at one magnitude.

    The sweep deliberately holds yaw at zero.  That keeps ``roll`` and ``pitch``
    aligned with the deterministic initial body frame used by the simulator
    telemetry assessment.
    """

    _validate_pcmd_value("magnitude", magnitude)
    if magnitude <= 0:
        raise ValueError("magnitude must be positive for a direction sweep")
    scenarios = tuple(
        Scenario(
            name=_direction_name(right=right, forward=forward, up=up),
            command=PilotingCommand(
                roll=right * magnitude,
                pitch=forward * magnitude,
                yaw=0,
                gaz=up * magnitude,
            ),
            duration_s=duration_s,
            settle_s=settle_s,
        )
        for right, forward, up in product((-1, 0, 1), repeat=3)
        if right or forward or up
    )
    if len(scenarios) != 26 or len({scenario.name for scenario in scenarios}) != 26:
        raise RuntimeError("direction sweep did not produce a unique 26-case matrix")
    return scenarios


@dataclass(frozen=True)
class TruePosition:
    """Sphinx true world position in Gazebo ENU metres."""

    timestamp_s: float
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class MotionDelta:
    """Movement expressed in the deterministic yaw-zero probe frame."""

    forward_m: float
    right_m: float
    up_m: float
    yaw_change_deg: float
