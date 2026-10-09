"""Confirm a changed localization anchor before publishing it to control."""

from __future__ import annotations

import numpy as np


class PoseSourceConfirmation:
    """Require two distinct, consistent strong captures after a discontinuity.

    The candidate stays anchored to its first capture, not a moving average.
    Neither repeated timestamps nor weak fixes can confirm a new source.
    """

    def __init__(self) -> None:
        self.accepted: np.ndarray | None = None
        self.source: str | None = None
        self.stamp = float("-inf")
        self.pending: np.ndarray | None = None
        self.pending_source: str | None = None
        self.pending_stamp = float("-inf")
        self.needs_confirmation = False

    def interrupt(self) -> None:
        self.needs_confirmation = self.accepted is not None
        self.pending_stamp = float("-inf")

    def accept(
        self,
        xyz: np.ndarray,
        stamp: float,
        *,
        reliable: bool,
        radius: float,
        source: str | None = None,
    ) -> bool:
        if stamp <= self.stamp:
            return False
        self.stamp = stamp
        if not reliable:
            was_pending = self.pending is not None
            self.interrupt()
            displaced = self.accepted is not None and float(np.linalg.norm(xyz - self.accepted)) > 2.0 * radius
            if displaced:
                self.pending = xyz.copy()
                self.pending_source = source
            return not was_pending and not displaced
        changed = self.accepted is not None and (
            self.needs_confirmation
            or source != self.source
            or float(np.linalg.norm(xyz - self.accepted)) > 2.0 * radius
        )
        if changed:
            confirmed = (
                self.pending is not None
                and source == self.pending_source
                and 0.0 < stamp - self.pending_stamp <= 2.0
                and float(np.linalg.norm(xyz - self.pending)) <= radius
            )
            if not confirmed:
                self.pending = xyz.copy()
                self.pending_source = source
                self.pending_stamp = stamp
                self.needs_confirmation = True
                return False
        self.accepted = xyz.copy()
        self.source = source
        self.pending = None
        self.needs_confirmation = False
        return True
