#!/usr/bin/env python3
"""Timestamp synchronization for camera / NED velocity / attitude / GNSS.

All streams carry monotonic timestamps; fusion must not assume fixed dt.
Checks max_sync_error_s and velocity freshness before using prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pose_guided.types import FusedOdometrySample


@dataclass(frozen=True)
class SyncConfig:
    max_sync_error_s: float = 0.05
    max_velocity_age_s: float = 0.5
    max_attitude_age_s: float = 0.2


@dataclass(frozen=True)
class SyncStatus:
    valid: bool
    age_velocity_s: float | None
    age_attitude_s: float | None
    reason: str | None = None


class TelemetrySynchronizer:
    def __init__(self, config: SyncConfig | None = None):
        self.config = config or SyncConfig()

    def check(
        self,
        camera_timestamp: float,
        fused: FusedOdometrySample | None,
    ) -> SyncStatus:
        if fused is None:
            return SyncStatus(valid=False, age_velocity_s=None, age_attitude_s=None, reason="no_fused")
        if not math.isfinite(camera_timestamp) or not math.isfinite(fused.timestamp):
            return SyncStatus(valid=False, age_velocity_s=None, age_attitude_s=None, reason="non_finite_timestamp")
        dt = abs(float(camera_timestamp) - float(fused.timestamp))
        if dt > self.config.max_sync_error_s:
            return SyncStatus(valid=False, age_velocity_s=dt, age_attitude_s=dt, reason=f"sync_error_{dt:.3f}s")
        age_vel = None
        age_att = None
        if fused.velocity_ned is not None:
            age_vel = abs(float(camera_timestamp) - float(fused.timestamp))
            if age_vel > self.config.max_velocity_age_s:
                return SyncStatus(valid=False, age_velocity_s=age_vel, age_attitude_s=age_att, reason="velocity_stale")
        if fused.has_attitude:
            age_att = abs(float(camera_timestamp) - float(fused.timestamp))
            if age_att > self.config.max_attitude_age_s:
                # attitude stale is warning, not fatal for position
                pass
        return SyncStatus(valid=True, age_velocity_s=age_vel, age_attitude_s=age_att, reason=None)

    def velocity_age(self, camera_timestamp: float, fused: FusedOdometrySample | None) -> float | None:
        if fused is None or not math.isfinite(camera_timestamp) or not math.isfinite(fused.timestamp):
            return None
        return abs(float(camera_timestamp) - float(fused.timestamp))
