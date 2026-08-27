"""Bounded yaw-only search used while waiting for an initial visual pose.

The state machine owns no aircraft connection and never takes off or changes
piloting authority.  Callers remain responsible for those safety boundaries.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class YawSearchConfig:
    wait_before_search_s: float = 10.0
    max_search_s: float = 12.0
    sweep_angle_deg: float = 25.0
    yaw_pcmd: int = 8
    tick_s: float = 0.1

    def __post_init__(self) -> None:
        for name in ("wait_before_search_s", "max_search_s", "sweep_angle_deg", "tick_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.yaw_pcmd, bool) or not isinstance(self.yaw_pcmd, int):
            raise ValueError("yaw_pcmd must be an integer")
        if not 1 <= self.yaw_pcmd <= 20:
            raise ValueError("yaw_pcmd must be between 1 and 20")
        if self.sweep_angle_deg > 45.0:
            raise ValueError("sweep_angle_deg must not exceed 45 degrees")


@dataclass(frozen=True, slots=True)
class YawSearchStep:
    yaw_pcmd: int
    state: str
    elapsed_s: float
    relative_yaw_deg: float | None

    @property
    def terminal(self) -> bool:
        return self.state in {"pose_found", "cancelled", "timed_out", "unsafe"}


class BoundedYawSearch:
    """Alternate a small left/right sweep and stop after a fixed duration."""

    def __init__(self, config: YawSearchConfig = YawSearchConfig()):
        self.config = config
        self._began_at: float | None = None
        self._last_now: float | None = None
        self._last_yaw: float | None = None
        self._relative_yaw = 0.0
        self._direction = 1
        self._terminal_state: str | None = None

    def begin(self, now: float) -> None:
        stamp = self._finite_now(now)
        self._began_at = stamp
        self._last_now = stamp
        self._last_yaw = None
        self._relative_yaw = 0.0
        self._direction = 1
        self._terminal_state = None

    @staticmethod
    def _finite_now(value: float) -> float:
        stamp = float(value)
        if not math.isfinite(stamp):
            raise ValueError("search time must be finite")
        return stamp

    @staticmethod
    def _finite_yaw(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            yaw = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return yaw if math.isfinite(yaw) else None

    @staticmethod
    def _wrapped_delta(current: float, previous: float) -> float:
        return (current - previous + math.pi) % (2.0 * math.pi) - math.pi

    def stop(self, state: str = "cancelled") -> None:
        if state not in {"pose_found", "cancelled", "timed_out", "unsafe"}:
            raise ValueError(f"invalid terminal search state: {state}")
        self._terminal_state = state

    def _update_step_state(
        self,
        stamp: float,
        elapsed: float,
        *,
        pose_found: bool,
        safe: bool,
    ) -> None:
        if self._last_now is not None and stamp < self._last_now:
            self.stop("unsafe")
        self._last_now = stamp
        if pose_found:
            self.stop("pose_found")
        elif not safe:
            self.stop("unsafe")
        elif elapsed >= self.config.wait_before_search_s + self.config.max_search_s:
            self.stop("timed_out")

    def _update_yaw(self, yaw: float) -> None:
        if self._last_yaw is None:
            self._last_yaw = yaw
        else:
            self._relative_yaw += self._wrapped_delta(yaw, self._last_yaw)
            self._last_yaw = yaw

        span = math.radians(self.config.sweep_angle_deg)
        if self._direction > 0 and self._relative_yaw >= span:
            self._direction = -1
        elif self._direction < 0 and self._relative_yaw <= -span:
            self._direction = 1

    def step(
        self,
        *,
        now: float,
        aircraft_yaw_rad: float | None,
        pose_found: bool = False,
        safe: bool = True,
    ) -> YawSearchStep:
        if self._began_at is None:
            self.begin(now)
        stamp = self._finite_now(now)
        assert self._began_at is not None
        elapsed = max(0.0, stamp - self._began_at)
        self._update_step_state(
            stamp,
            elapsed,
            pose_found=pose_found,
            safe=safe,
        )
        if self._terminal_state is not None:
            return YawSearchStep(0, self._terminal_state, elapsed, None)
        if elapsed < self.config.wait_before_search_s:
            return YawSearchStep(0, "waiting", elapsed, None)

        yaw = self._finite_yaw(aircraft_yaw_rad)
        if yaw is None:
            # Without yaw feedback there is no enforceable angular bound.
            return YawSearchStep(0, "no_yaw_telemetry", elapsed, None)
        self._update_yaw(yaw)
        return YawSearchStep(
            self._direction * self.config.yaw_pcmd,
            "searching_right" if self._direction > 0 else "searching_left",
            elapsed,
            math.degrees(self._relative_yaw),
        )


class ManualLocalizationSearch:
    """Run an explicitly requested search without taking authority or taking off."""

    def __init__(
        self,
        *,
        send_pcmd: Callable[[int, int, int, int, str], bool],
        aircraft_yaw: Callable[[], float | None],
        pose_found: Callable[[], bool],
        safe_to_search: Callable[[], bool],
        force_relocalize: Callable[[], None],
        on_finished: Callable[[str], None] | None = None,
        config: YawSearchConfig = YawSearchConfig(),
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.send_pcmd = send_pcmd
        self.aircraft_yaw = aircraft_yaw
        self.pose_found = pose_found
        self.safe_to_search = safe_to_search
        self.force_relocalize = force_relocalize
        self.on_finished = on_finished
        self.config = config
        self.now = now
        self.sleep = sleep
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._finish_lock = threading.Lock()
        self._finished_state: str | None = None

    @property
    def active(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def finished_state(self) -> str | None:
        with self._finish_lock:
            return self._finished_state

    def start(self) -> bool:
        if self.active or not self.safe_to_search():
            return False
        self._cancel.clear()
        with self._finish_lock:
            self._finished_state = None
        self._thread = threading.Thread(
            target=self._run,
            name="manual-localization-yaw-search",
            daemon=True,
        )
        self._thread.start()
        return True

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancel.set()
        self._send_zero(f"manual_localization_search_{reason}")

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _send_zero(self, reason: str) -> None:
        try:
            self.send_pcmd(0, 0, 0, 0, reason)
        except Exception:
            pass

    def _finish(self, state: str) -> None:
        self._send_zero(f"manual_localization_search_{state}")
        with self._finish_lock:
            self._finished_state = state
        if self.on_finished is not None:
            try:
                self.on_finished(state)
            except Exception:
                pass

    def _run(self) -> None:
        search = BoundedYawSearch(self.config)
        search.begin(self.now())
        next_relocalize = float("-inf")
        terminal = "unsafe"
        try:
            while True:
                now = self.now()
                if self._cancel.is_set():
                    search.stop("cancelled")
                safe = bool(self.safe_to_search())
                found = bool(self.pose_found())
                step = search.step(
                    now=now,
                    aircraft_yaw_rad=self.aircraft_yaw(),
                    pose_found=found,
                    safe=safe,
                )
                if step.terminal:
                    terminal = step.state
                    break
                if now >= next_relocalize:
                    try:
                        self.force_relocalize()
                    except Exception:
                        pass
                    next_relocalize = now + 0.5
                try:
                    accepted = bool(
                        self.send_pcmd(
                            0,
                            0,
                            int(step.yaw_pcmd),
                            0,
                            "manual_localization_search",
                        )
                    )
                except Exception:
                    accepted = False
                if not accepted:
                    terminal = "unsafe"
                    break
                self.sleep(self.config.tick_s)
        finally:
            self._finish(terminal)


__all__ = [
    "BoundedYawSearch",
    "ManualLocalizationSearch",
    "YawSearchConfig",
    "YawSearchStep",
]
