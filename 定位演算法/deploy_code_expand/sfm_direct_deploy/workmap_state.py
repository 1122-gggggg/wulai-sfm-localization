"""Mapping state of the online workmap front end (pure logic, no I/O).

States: LOCALIZE (strong old-map coverage, nothing is mapped), ANCHORED_EXPAND (old
map still constrains the pose, new points are mapped), DETACHED_EXPAND (outside the
old map, a detached submap collects points), RELOCALIZE (back in the old map, the
detached submap waits for multi-frame confirmation), MERGE (the confirmed submap is
being sewn in).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: Old-map share R_G of all valid tracked correspondences that starts / keeps LOCALIZE.
LOCALIZE_ENTER_RATIO = 0.8
LOCALIZE_EXIT_RATIO = 0.7
LOCALIZE_HOLD_FRAMES = 8
#: Same floor as the profile's ``minimum_hull_coverage`` for an accepted old-map pose.
LOCALIZE_MIN_HULL = 0.15
#: Consecutive frames without an old-map anchor before the camera counts as detached.
DETACH_FRAMES = 12

_ANCHOR_STATUSES = frozenset({"FAST_TRACK", "RELOC_SEED"})


class MappingState(str, Enum):
    LOCALIZE = "LOCALIZE"
    ANCHORED_EXPAND = "ANCHORED_EXPAND"
    DETACHED_EXPAND = "DETACHED_EXPAND"
    RELOCALIZE = "RELOCALIZE"
    MERGE = "MERGE"


@dataclass(frozen=True)
class MappingInputs:
    pose_ok: bool
    status: str
    map_inliers: int
    n_corr: int
    hull_coverage: float | None
    detached_open: bool
    relocalizing: bool
    merge_pending: bool
    strong_inliers: int
    min_inliers: int


def frozen_ratio(x: MappingInputs) -> float:
    """R_G: old-map inliers over all valid tracked correspondences."""
    return x.map_inliers / max(1, x.n_corr)


class MappingStateMachine:
    def __init__(self) -> None:
        self.state = MappingState.ANCHORED_EXPAND
        self._unanchored = 0
        self._enter_run = 0
        self._detach_due = False

    @property
    def detach_due(self) -> bool:
        """True on the frame the unanchored run just reached ``DETACH_FRAMES``."""
        return self._detach_due

    def update(self, x: MappingInputs) -> MappingState:
        anchor_status = x.pose_ok and x.status in _ANCHOR_STATUSES
        strong = anchor_status and x.map_inliers >= x.strong_inliers
        anchored = anchor_status and x.map_inliers >= x.min_inliers
        ratio = frozen_ratio(x)
        hull_ok = x.hull_coverage is not None and x.hull_coverage >= LOCALIZE_MIN_HULL
        self._unanchored = 0 if anchored else self._unanchored + 1
        self._detach_due = self._unanchored == DETACH_FRAMES
        enter = strong and ratio >= LOCALIZE_ENTER_RATIO and hull_ok
        self._enter_run = self._enter_run + 1 if enter else 0

        if x.merge_pending:
            new = MappingState.MERGE
        elif x.relocalizing:
            new = MappingState.RELOCALIZE
        elif x.detached_open:
            new = MappingState.DETACHED_EXPAND
        elif self.state is MappingState.LOCALIZE and strong and ratio >= LOCALIZE_EXIT_RATIO:
            new = MappingState.LOCALIZE
        elif self._enter_run >= LOCALIZE_HOLD_FRAMES:
            new = MappingState.LOCALIZE
        elif anchored:
            new = MappingState.ANCHORED_EXPAND
        elif self._unanchored >= DETACH_FRAMES:
            new = MappingState.DETACHED_EXPAND
        elif self.state is MappingState.DETACHED_EXPAND:
            new = MappingState.DETACHED_EXPAND
        else:
            new = MappingState.ANCHORED_EXPAND
        self.state = new
        return new
