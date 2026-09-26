"""Timestamped telemetry for heading alignment and bounded control prediction.

Only strong, observed map poses may teach metric scale. Prediction never changes
the source pose, its timestamp, or route/landing confirmation evidence.
"""

from collections import deque
import math

import numpy as np


class ControlPoseSmoother:
    """Blend horizontal observation corrections for the servo only, over 0.5 s."""

    def __init__(self, map_frame):
        self.frame = map_frame
        self.reset()

    def reset(self):
        self.previous = None
        self.started = None
        self.offset = np.zeros(3)
        self.recovery_pending = False

    def correction(self, now):
        if self.started is None:
            return np.zeros(3)
        weight = max(0., 1. - max(0., now - self.started) / .5)
        return self.offset * weight

    def observe(self, pose, now, expected_motion=None):
        if not 0. <= now - pose.stamp <= .5 or not np.isfinite(pose.xyz).all():
            self.reset()
            return 0.
        confirmed = pose.map_confirmed and pose.position_observed and not pose.reseed_confirming
        if not confirmed:
            self.recovery_pending = True
        previous = self.previous
        if previous is not None and pose.stamp <= previous[0]:
            return 0.
        self.previous = (pose.stamp, pose.xyz.copy())
        if previous is None or pose.stamp - previous[0] > .5:
            self.started = None
            self.recovery_pending = not confirmed
            return 0.
        delta = pose.xyz - previous[1]
        if expected_motion is not None:
            delta = delta - expected_motion
        step = float(np.linalg.norm(delta))
        recovered = confirmed and self.recovery_pending
        if confirmed:
            self.recovery_pending = False
        if step > .02 and recovered:
            # Height and evidence remain untouched. Accumulated corrections
            # cannot hide more than 0.1 map unit, even under repeated reseeds.
            east, north = self.frame.horizontal(delta)
            horizontal = east * np.asarray(self.frame.east) + north * np.asarray(self.frame.north)
            offset = self.correction(now) - horizontal
            norm = float(np.linalg.norm(offset))
            self.offset = offset * min(1., .1 / max(norm, 1e-12))
            self.started = now
        return step


class ControlStateEstimator:
    def __init__(self, map_frame):
        self.frame = map_frame
        self.yaws = deque(maxlen=512)
        self.velocities = deque(maxlen=512)
        self.reset_position()

    def reset_position(self):
        self.reset_observation()
        self.scale_samples = deque(maxlen=12)
        self.scale_m_per_unit = None
        self.scale_stamp = None

    def reset_observation(self):
        """Break displacement pairs at a pose correction, retaining metric scale."""
        self.last_observed = None
        self.pending = deque(maxlen=128)
        self.scale_anchor = None

    @staticmethod
    def _append(history, sample, now, size):
        if sample is None:
            return
        try:
            values = tuple(float(value) for value in sample)
        except (TypeError, ValueError, OverflowError):
            return
        if len(values) != size or not all(math.isfinite(value) for value in values):
            return
        stamp = values[-1]
        if not 0.0 <= now - stamp <= .5 or (history and stamp <= history[-1][0]):
            return
        history.append((stamp, *values[:-1]))
        while history and now - history[0][0] > 6.0:
            history.popleft()

    def observe_telemetry(self, attitude, velocity, now):
        """Input yaw/stamp and NED north/east/stamp; cached stamps add nothing."""
        self._append(self.yaws, attitude, now, 2)
        self._append(self.velocities, velocity, now, 3)

    def yaw_at(self, stamp):
        if not self.yaws or not self.yaws[0][0] <= stamp <= self.yaws[-1][0]:
            return None
        for index, (end, yaw) in enumerate(self.yaws):
            if end == stamp:
                return yaw
            if end > stamp:
                start, previous = self.yaws[index - 1]
                if end - start > .5:
                    return None
                delta = (yaw - previous + math.pi) % (2 * math.pi) - math.pi
                return (previous + delta * (stamp - start) / (end - start) + math.pi) % (2 * math.pi) - math.pi
        return None

    def displacement(self, start, end, *, extrapolate=False):
        """Integrate NED velocity; refuse gaps or unobserved interval starts."""
        if not self.velocities or end < start:
            return None
        rows = np.asarray(self.velocities, dtype=float)
        times = rows[:, 0]
        if start < times[0] or end > times[-1] + (.5 if extrapolate else 0.):
            return None
        gaps = np.diff(times)
        overlap = (times[:-1] < end) & (times[1:] > start)
        if np.any(gaps[overlap] > .5):
            return None
        knots = np.concatenate(([start], times[(times > start) & (times < end)], [end]))
        values = np.column_stack([np.interp(knots, times, rows[:, axis]) for axis in (1, 2)])
        return np.sum(.5 * (values[:-1] + values[1:]) * np.diff(knots)[:, None], axis=0)

    @staticmethod
    def _map_horizontal(ned, offset):
        north, east = ned
        c, s = math.cos(offset), math.sin(offset)
        return np.array([c * east - s * north, s * east + c * north])

    def observe_pose(self, pose, now, heading_offset):
        trusted = (pose.map_confirmed and pose.position_observed and not pose.reseed_confirming
                   and 0.0 <= now - pose.stamp <= .5)
        # A weak intervening frame is not a displacement endpoint. It must not
        # discard already observed endpoints awaiting delayed telemetry.
        if trusted and (self.last_observed is None or pose.stamp > self.last_observed[0]):
            self.last_observed = (float(pose.stamp), pose.xyz.copy())
            self.pending.append(self.last_observed)
        if not self.velocities or heading_offset is None:
            return
        # Telemetry may arrive later than vision. Match on sample time, never
        # fill the missing tail with extrapolation while learning a scale.
        while self.pending and self.pending[0][0] <= self.velocities[-1][0]:
            sample = self.pending.popleft()
            if self.scale_anchor is None:
                self.scale_anchor = sample
                continue
            elapsed = sample[0] - self.scale_anchor[0]
            if elapsed < 1.0:
                continue
            start, xyz = self.scale_anchor
            displacement = self.displacement(start, sample[0]) if elapsed <= 4.5 else None
            if displacement is None:
                self.scale_anchor = sample
                continue
            metric = self._map_horizontal(displacement, heading_offset)
            visual = np.asarray(self.frame.horizontal(sample[1] - xyz))
            metres, units = float(np.linalg.norm(metric)), float(np.linalg.norm(visual))
            if metres < 1.0 or units < .02:
                continue
            # Independent intervals end on observed displacement, not a fixed
            # wall-clock bucket that discards short legs or slow approaches.
            self.scale_anchor = sample
            if float(np.dot(metric, visual)) < math.cos(math.radians(20)) * metres * units:
                self.scale_samples.clear()
                self.scale_m_per_unit = None
                continue
            self.scale_samples.append((metres / units, metres))
            self._update_scale(sample[0])

    def _update_scale(self, stamp):
        if len(self.scale_samples) < 8 or sum(row[1] for row in self.scale_samples) < 10.0:
            return
        ratios = np.array([row[0] for row in self.scale_samples])
        median = float(np.median(ratios))
        spread = float(np.percentile(ratios, 75) - np.percentile(ratios, 25)) / median
        half = len(ratios) // 2
        drift = abs(float(np.median(ratios[half:]) / np.median(ratios[:half])) - 1.0)
        if spread > .20 or drift > .20:
            self.scale_m_per_unit = None
            return
        self.scale_m_per_unit, self.scale_stamp = median, stamp

    def observed_motion(self, start, end, now, heading_offset):
        """Expected inter-frame motion from fully covered velocity and trusted scale."""
        if (heading_offset is None or self.scale_m_per_unit is None or self.scale_stamp is None
                or not 0. <= now - self.scale_stamp <= 30. or not 0. < end - start <= .5):
            return None
        displacement = self.displacement(start, end)
        if displacement is None:
            return None
        east, north = self._map_horizontal(displacement, heading_offset) / self.scale_m_per_unit
        return east * np.asarray(self.frame.east) + north * np.asarray(self.frame.north)

    def predict(self, pose, now, heading_offset, radius):
        diagnostics = {"scale_m_per_unit": self.scale_m_per_unit,
                       "scale_segments": len(self.scale_samples), "applied": False}
        if (heading_offset is None or self.scale_m_per_unit is None or self.scale_stamp is None
                or not 0.0 <= now - self.scale_stamp <= 30.0):
            diagnostics["reason"] = "scale_or_heading_unavailable"
            return pose.xyz, diagnostics
        # IMU_BRIDGE holds the last center but publishes a newer frame stamp.
        # Propagate from the actual observed anchor, not the bridge timestamp.
        anchor = (pose.stamp, pose.xyz) if pose.position_observed else self.last_observed
        if anchor is None or not 0.0 <= now - anchor[0] <= .5:
            diagnostics["reason"] = "prediction_horizon_expired"
            return pose.xyz, diagnostics
        displacement = self.displacement(anchor[0], now, extrapolate=True)
        if displacement is None:
            diagnostics["reason"] = "velocity_history_unavailable"
            return pose.xyz, diagnostics
        horizontal = self._map_horizontal(displacement, heading_offset) / self.scale_m_per_unit
        shift = horizontal[0] * np.asarray(self.frame.east) + horizontal[1] * np.asarray(self.frame.north)
        xyz = anchor[1] + shift
        # Keep the control-only correction inside half the authored sphere.
        correction = xyz - pose.xyz
        distance = float(np.linalg.norm(correction))
        cap = .5 * float(radius)
        if distance > cap:
            correction *= cap / distance
        diagnostics.update(applied=True, reason="telemetry_prediction",
                           horizon_s=now - anchor[0], limited=distance > cap,
                           correction_u=correction.tolist())
        return pose.xyz + correction, diagnostics
