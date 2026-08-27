"""Small flight-safety modules shared by the live backend and its tests."""
from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Hashable


@dataclass(frozen=True)
class TelemetrySample:
    value: Any
    marker: Hashable
    observed_mono_ns: int


class TelemetryFreshnessStore:
    """Keep event receipt age separate from UI polling age.

    Re-reading the same Olympe cache entry must not make telemetry fresh.  A
    field advances only when its event marker changes.
    """

    def __init__(self) -> None:
        self._samples: dict[str, TelemetrySample] = {}
        self._lock = threading.RLock()

    def observe(
        self,
        name: str,
        value: Any,
        *,
        marker: Hashable,
        observed_mono_ns: int,
    ) -> TelemetrySample:
        key = str(name)
        stamp = int(observed_mono_ns)
        if stamp < 0:
            raise ValueError("observed_mono_ns cannot be negative")
        with self._lock:
            previous = self._samples.get(key)
            if previous is not None and previous.marker == marker:
                return previous
            sample = TelemetrySample(value, marker, stamp)
            self._samples[key] = sample
            return sample

    def sample(self, name: str) -> TelemetrySample | None:
        with self._lock:
            return self._samples.get(str(name))

    def fresh(
        self,
        name: str,
        *,
        max_age_s: float,
        now_mono_ns: int,
    ) -> Any | None:
        age_limit = float(max_age_s)
        if not math.isfinite(age_limit) or age_limit < 0.0:
            raise ValueError("max_age_s must be finite and non-negative")
        sample = self.sample(name)
        if sample is None:
            return None
        now_ns = int(now_mono_ns)
        if now_ns < sample.observed_mono_ns:
            return None
        age_ns = now_ns - sample.observed_mono_ns
        if age_ns > int(age_limit * 1_000_000_000):
            return None
        return sample.value


class AuthorityController:
    """Serialize complete manual/PC authority transitions.

    The live backend can receive a physical-stick callback while a blocking
    piloting-source handoff is in progress.  A re-entrant lock keeps those
    transitions whole without deadlocking deliberate nested manual handoffs.
    Safety epoch checks still belong to the backend because they can invalidate
    a transition from outside this lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @contextmanager
    def transition(self) -> Iterator[None]:
        with self._lock:
            yield


@dataclass(frozen=True)
class LandingOutcome:
    confirmed: bool
    reason_code: str


class TakeoffLandingSupervisor:
    """Serialize landing confirmation shared by failure and shutdown paths."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def ensure_landed(
        self,
        *,
        reason: str,
        is_landed: Callable[[], bool],
        land_and_confirm: Callable[[str], bool],
        force_command: bool = False,
    ) -> LandingOutcome:
        with self._lock:
            if not force_command and bool(is_landed()):
                return LandingOutcome(True, "ALREADY_LANDED")
            confirmed = bool(land_and_confirm(str(reason)))
            return LandingOutcome(
                confirmed,
                "LANDED_CONFIRMED" if confirmed else "LAND_UNCONFIRMED",
            )
