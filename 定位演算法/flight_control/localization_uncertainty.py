"""Pure localization uncertainty/recovery state transitions.

The flight loop owns all side effects (PCMD, relocalization, and pilot handoff).
This module only computes the next state and the gate that the loop should take.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LocalizationAction = Literal["uncertain_hover", "recovery_hover", "track", "land"]


@dataclass(frozen=True)
class LocalizationState:
    """State carried between localization observations."""

    uncertain_since: float | None = None
    uncertain_land_after: float | None = None
    recovery_good_fixes: int = 0
    dirty_streak: int = 0

    def __post_init__(self) -> None:
        if (self.uncertain_since is None) != (self.uncertain_land_after is None):
            raise ValueError("uncertain_since and uncertain_land_after must be present together")
        if self.recovery_good_fixes < 0:
            raise ValueError("recovery_good_fixes cannot be negative")
        if self.dirty_streak < 0:
            raise ValueError("dirty_streak cannot be negative")


@dataclass(frozen=True)
class LocalizationDecision:
    """Pure output of one localization observation."""

    action: LocalizationAction
    state: LocalizationState
    waited_s: float | None


def decide_localization_transition(
    state: LocalizationState,
    *,
    now: float,
    fresh: bool,
    low_confidence: bool,
    weak_hover_land_s: float,
    lost_land_s: float,
    recovery_good_fixes_required: int,
) -> LocalizationDecision:
    """Return the next uncertainty state and side-effect-free gate decision."""
    if low_confidence or not fresh:
        current_limit = weak_hover_land_s if low_confidence else lost_land_s
        if state.uncertain_since is None:
            uncertain_since = now
            uncertain_land_after = current_limit
        else:
            uncertain_since = state.uncertain_since
            assert state.uncertain_land_after is not None
            uncertain_land_after = min(state.uncertain_land_after, current_limit)
        waited = now - uncertain_since
        # Hysteresis: one isolated dirty tick during recovery (flicker) keeps the
        # confirmation count; the land timers above are untouched, and a second
        # consecutive dirty tick resets as before.
        dirty = state.dirty_streak + 1
        kept = state.recovery_good_fixes if (
            state.recovery_good_fixes > 0 and dirty == 1
        ) else 0
        next_state = LocalizationState(uncertain_since, uncertain_land_after, kept, dirty)
        action: LocalizationAction = "land" if waited >= uncertain_land_after else "uncertain_hover"
        return LocalizationDecision(action, next_state, waited)

    if (
        state.uncertain_since is not None
        and state.recovery_good_fixes + 1 < recovery_good_fixes_required
    ):
        next_state = LocalizationState(
            state.uncertain_since,
            state.uncertain_land_after,
            state.recovery_good_fixes + 1,
            0,
        )
        return LocalizationDecision(
            "recovery_hover",
            next_state,
            now - state.uncertain_since,
        )

    return LocalizationDecision("track", LocalizationState(), None)
