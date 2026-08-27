"""Bounded LOST/LOW recovery policy for simulated and live operator inputs."""
from __future__ import annotations


class LostHoldPolicy:
    """Bound sustained-LOW and LOST recovery for simulated and live inputs.

    A real aircraft hovers on a LOST pose, so MegaLoc
    reacquires from the scene the camera is still pointed at. A file stream has
    no such feedback:
    it would run ~10 frames ahead during a ~350 ms global relocalization. Holding
    the frame the stream stopped on keeps the offline run faithful to flight.

    Bounded on purpose: one hold per lost episode, released by a fix, by
    max_attempts or by timeout_s, and re-armed only by the next fix. Retrying one
    frozen frame is near-deterministic (only RANSAC sampling differs), so a frame
    the one-shot MegaLoc plus EDM recovery cannot solve must never pause the
    stream indefinitely.

    A file/video source freezes its current frame. A live drone stream keeps
    advancing; the backend first sends zero PCMD and hands control to the pilot.
    """

    def __init__(self, max_attempts: int = 5, timeout_s: float = 10.0,
                 low_confidence_results: int = 2,
                 hold_on_low_confidence: bool = False):
        self.max_attempts = max(1, int(max_attempts))
        self.timeout_s = max(0.0, float(timeout_s))
        self.low_confidence_results = max(1, int(low_confidence_results))
        self.hold_on_low_confidence = bool(hold_on_low_confidence)
        self.active = False
        self.armed = True
        self.attempts = 0
        self.low_streak = 0
        self.frame_index = -1
        self.started = 0.0

    def reset(self) -> None:
        self.active = False
        self.armed = True
        self.attempts = 0
        self.low_streak = 0
        self.frame_index = -1
        self.started = 0.0

    def on_result(self, *, success: bool, low_confidence: bool,
                  strong_relocalize: bool, next_mode: str, frame_index: int,
                  now: float) -> str | None:
        """Fold in one localizer result; returns an event name on a state change."""
        # Results are ordered by the synchronous worker.  If the tracker's own
        # first LOST recovery already moved it out of LOST, accept that fix before
        # the hold can enqueue a duplicate explicit recovery request.
        # LOW/WEAK never starts a hold or runs MegaLoc.
        trustworthy = (
            bool(success)
            and not bool(low_confidence)
            and (
                not self.active
                or bool(strong_relocalize)
                or str(next_mode) != "LOST"
            )
        )
        if trustworthy:
            was_active = self.active
            self.reset()
            return "RELEASE_FIX" if was_active else None
        if self.active:
            if self.attempts >= self.max_attempts:
                self.active = False
                self.armed = False          # only a fix re-arms the next hold
                return "RELEASE_ATTEMPTS"
            return None
        if not self.armed:
            return None
        if success:
            self.low_streak = self.low_streak + 1 if low_confidence else 0
            if (self.hold_on_low_confidence
                    and self.low_streak >= self.low_confidence_results):
                self.active = True
                self.attempts = 0
                self.frame_index = int(frame_index)
                self.started = float(now)
                self.low_streak = 0
                return "ENGAGE_LOW_CONF"
            return None
        self.low_streak = 0
        if str(next_mode) != "LOST":
            return None
        event = "ENGAGE_FAIL"
        self.active = True
        self.attempts = 0
        self.frame_index = int(frame_index)
        self.started = float(now)
        return event

    def check_timeout(self, now: float) -> str | None:
        """Release a hold whose retries stalled (e.g. worker restart warmup)."""
        if not self.active or self.timeout_s <= 0.0:
            return None
        if float(now) - self.started < self.timeout_s:
            return None
        self.active = False
        self.armed = False
        return "RELEASE_TIMEOUT"

    def wants_retry(self) -> bool:
        return self.active and self.attempts < self.max_attempts

    def note_submit(self) -> None:
        if self.active:
            self.attempts += 1
