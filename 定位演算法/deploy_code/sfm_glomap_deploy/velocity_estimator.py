#!/usr/bin/env python3
"""Parrot ANAFI NED velocity integration — firmware-fused, not raw IMU.

Parrot Olympe exposes SpeedChanged.speedX/Y/Z already in NED/world frame (m/s),
verified via `log_anafi_telemetry.py` and `pose_guided/types.FusedOdometrySample`.
Do NOT rotate this velocity again with yaw — it is already metric world.

Implements trapezoidal integration:
  p_{t+1} = p_t + (v_t + v_{t+1})/2 * dt
with timestamp-derived dt and freshness gating.

References: white-paper_anafi-v1.4 (2500mA battery etc not needed) and
https://developer.parrot.com/docs/olympe/index.html (Olympe API).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from pose_guided.types import FusedOdometrySample


@dataclass(frozen=True)
class NEDVelocityConfig:
    max_velocity_age_s: float = 0.5
    max_dt_s: float = 0.3
    min_speed_for_integration_mps: float = 0.02
    history_len: int = 128


class NEDVelocityEstimator:
    """Integrate NED fused velocity into NED position with freshness check.

    Keeps a monotonic NED position by integrating (v_t+v_{t+1})/2*dt from an
    arbitrary origin (0,0,0) at first valid sample.  Caller may re-anchor
    via `reset(origin, timestamp)` when a valid Sim(3) is available.
    """

    def __init__(self, config: NEDVelocityConfig | None = None):
        self.config = config or NEDVelocityConfig()
        self._history: deque[FusedOdometrySample] = deque(maxlen=self.config.history_len)
        self._ned_position: np.ndarray | None = None
        self._last_timestamp: float | None = None
        self._last_velocity: tuple[float, float, float] | None = None

    def reset(self, origin_ned: tuple[float, float, float] | np.ndarray | None = None, timestamp: float | None = None) -> None:
        self._history.clear()
        if origin_ned is None:
            self._ned_position = None
        else:
            self._ned_position = np.asarray(origin_ned, dtype=float).reshape(3)
        self._last_timestamp = float(timestamp) if timestamp is not None and math.isfinite(timestamp) else None
        self._last_velocity = None

    def observe(self, sample: FusedOdometrySample | None) -> None:
        if sample is None or not sample.has_ned_velocity or sample.velocity_ned is None:
            return
        if not math.isfinite(sample.timestamp):
            return
        # out-of-order or duplicate
        if self._last_timestamp is not None and sample.timestamp <= self._last_timestamp + 1e-9:
            # still keep in history for interpolation but don't integrate
            self._history.append(sample)
            return
        # integrate
        if self._ned_position is None:
            self._ned_position = np.zeros(3, dtype=float)
            self._last_timestamp = float(sample.timestamp)
            self._last_velocity = tuple(float(v) for v in sample.velocity_ned)
            self._history.append(sample)
            return
        dt = float(sample.timestamp) - float(self._last_timestamp)  # type: ignore[arg-type]
        if not math.isfinite(dt) or dt <= 0 or dt > self.config.max_dt_s * 2:
            # gap too large → re-anchor without displacement, but keep freshness
            self._last_timestamp = float(sample.timestamp)
            self._last_velocity = tuple(float(v) for v in sample.velocity_ned)
            self._history.append(sample)
            return
        dt = min(dt, self.config.max_dt_s)
        v_prev = np.asarray(self._last_velocity, dtype=float)
        v_curr = np.asarray(sample.velocity_ned, dtype=float)
        if not np.isfinite(v_prev).all() or not np.isfinite(v_curr).all():
            self._last_timestamp = float(sample.timestamp)
            self._last_velocity = tuple(float(v) for v in sample.velocity_ned)
            self._history.append(sample)
            return
        # trapezoidal
        avg_v = 0.5 * (v_prev + v_curr)
        # suppress drift from tiny noisy velocities
        speed = float(np.linalg.norm(avg_v))
        if speed < self.config.min_speed_for_integration_mps:
            avg_v = np.zeros(3, dtype=float)
        self._ned_position = self._ned_position + avg_v * dt
        self._last_timestamp = float(sample.timestamp)
        self._last_velocity = tuple(float(v) for v in sample.velocity_ned)
        self._history.append(sample)

    def velocity_valid(self, query_timestamp: float) -> bool:
        if self._last_timestamp is None or self._last_velocity is None:
            return False
        if not math.isfinite(query_timestamp):
            return False
        age = float(query_timestamp) - float(self._last_timestamp)
        if not math.isfinite(age) or age < -1e-6:
            return False
        return age <= self.config.max_velocity_age_s

    def integrated_position(self) -> np.ndarray | None:
        if self._ned_position is None:
            return None
        return self._ned_position.copy()

    def predict_ned_position(self, query_timestamp: float) -> np.ndarray | None:
        """Extrapolate one step from last sample using its velocity (for fusion)."""
        if self._ned_position is None or self._last_timestamp is None or self._last_velocity is None:
            return None
        if not self.velocity_valid(query_timestamp):
            return None
        dt = float(query_timestamp) - float(self._last_timestamp)
        if not math.isfinite(dt) or dt < 0:
            return None
        dt = min(dt, self.config.max_dt_s)
        v = np.asarray(self._last_velocity, dtype=float)
        return self._ned_position + v * dt

    def interpolate_velocity(self, timestamp: float, max_sync_error_s: float = 0.05) -> tuple[float, float, float] | None:
        """Linear interpolate velocity between bracketing samples (trapezoidal helper)."""
        if len(self._history) < 2:
            latest = self._history[-1] if self._history else None
            if latest is None or latest.velocity_ned is None:
                return None
            if abs(float(latest.timestamp) - float(timestamp)) > max_sync_error_s:
                return None
            return latest.velocity_ned
        # find bracket
        before = None
        after = None
        for s in self._history:
            if s.timestamp <= timestamp:
                before = s
            elif s.timestamp >= timestamp:
                after = s
                break
        if before is None or after is None or before.velocity_ned is None or after.velocity_ned is None:
            # fallback to nearest
            nearest = before if before is not None else after
            if nearest is None or nearest.velocity_ned is None:
                return None
            if abs(float(nearest.timestamp) - float(timestamp)) > max_sync_error_s:
                return None
            return nearest.velocity_ned
        if abs(before.timestamp - after.timestamp) < 1e-9:
            return before.velocity_ned
        frac = (timestamp - before.timestamp) / (after.timestamp - before.timestamp)
        frac = max(0.0, min(1.0, frac))
        v0 = np.asarray(before.velocity_ned, dtype=float)
        v1 = np.asarray(after.velocity_ned, dtype=float)
        v = v0 + frac * (v1 - v0)
        return tuple(float(x) for x in v)
