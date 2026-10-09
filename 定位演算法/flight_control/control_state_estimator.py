"""Timestamped telemetry for heading alignment and bounded control prediction.

Only strong, observed map poses may teach metric scale. Prediction never changes
the source pose, its timestamp, or route/landing confirmation evidence.
"""

from collections import deque
import math
import threading

from map_scale import D_MIN_M, N_MIN_SEGMENTS, SEG_MIN_M, SEG_MIN_S, SEG_MIN_VISUAL_U

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
        # Session UI and AUTO share one estimator. Callers hold this lock over
        # their entire observation/control transaction, including resets.
        self.lock = threading.RLock()
        self.yaws = deque(maxlen=512)
        self.velocities = deque(maxlen=512)
        self._invalid_sources = {
            "attitude": deque(maxlen=512), "velocity": deque(maxlen=512),
        }
        self._invalid_source_keys = {}
        self.reset_position()

    def reset_position(self):
        """Start evidence for a new map/session; never carry scale across maps."""
        self.reset_observation()
        self.scale_samples = []
        self.scale_m_per_unit = None
        self.scale_stamp = None
        self.scale_candidate = None
        self.scale_distance = 0.
        self.scale_spread = None
        self.scale_drift = None
        self.scale_reason = "insufficient_segments"
        self.scale_rejections = {}
        self._velocity_cache = None

    def reset_observation(self):
        """Break displacement pairs at a pose correction, retaining map evidence."""
        self.last_observed = None
        self.pending = deque(maxlen=128)
        self.scale_anchor = None
        self.scale_endpoints = deque()
        self._scale_pose_stamp = None

    def _break_scale_interval(self, reason):
        """Break only calibration pairs; retain the bounded prediction anchor."""
        if self.pending or self.scale_anchor is not None:
            self._reject_scale(reason)
        self.pending.clear()
        self.scale_anchor = None
        self.scale_endpoints.clear()

    @staticmethod
    def _append(history, sample, now, size):
        if sample is None:
            return False
        try:
            values = tuple(float(value) for value in sample)
        except (TypeError, ValueError, OverflowError):
            return False
        if len(values) != size or not all(math.isfinite(value) for value in values):
            return False
        stamp = values[-1]
        if not 0.0 <= now - stamp <= .5 or (history and stamp <= history[-1][0]):
            return False
        history.append((stamp, *values[:-1]))
        while history and now - history[0][0] > 6.0:
            history.popleft()
        return True

    def _mark_invalid_source(self, source, history, sample, now, size):
        """Fence a genuinely invalid new event, not a cached/late valid sample.

        Retain its time even though it cannot enter the numerical history:
        otherwise interpolation silently bridges the invalid source interval.
        An unstampable payload is fenced once at receipt, not every poll.
        """
        if sample is None:
            return
        try:
            values = tuple(float(value) for value in sample)
        except (TypeError, ValueError, OverflowError):
            values = ()
        if len(values) == size and all(math.isfinite(value) for value in values):
            return
        stamp = values[-1] if len(values) == size and math.isfinite(values[-1]) else now
        if not 0. <= now - stamp <= .5 or (history and stamp <= history[-1][0]):
            return
        key = repr(sample)
        if self._invalid_source_keys.get(source) == key:
            return
        self._invalid_source_keys[source] = key
        barriers = self._invalid_sources[source]
        if not barriers or stamp > barriers[-1]:
            barriers.append(stamp)
            if source == "attitude":
                self._reject_scale("invalid_attitude_sample")

    def _source_interval_invalid(self, source, start, end):
        barriers = self._invalid_sources[source]
        return bool(barriers and barriers[-1] > start and barriers[0] <= end
                    and any(start < stamp <= end for stamp in barriers))

    def observe_telemetry(self, attitude, velocity, now):
        """Input yaw/stamp and NED north/east/stamp; cached stamps add nothing."""
        if self._append(self.yaws, attitude, now, 2):
            self._invalid_source_keys.pop("attitude", None)
        else:
            self._mark_invalid_source("attitude", self.yaws, attitude, now, 2)
        if self._append(self.velocities, velocity, now, 3):
            self._velocity_cache = None
            self._invalid_source_keys.pop("velocity", None)
        elif velocity is not None:
            self._mark_invalid_source("velocity", self.velocities, velocity, now, 3)
            try:
                values = tuple(float(value) for value in velocity)
                if len(values) != 3 or not all(math.isfinite(value) for value in values):
                    reason = "invalid_velocity_sample"
                elif now < values[-1]:
                    reason = "velocity_future"
                elif now - values[-1] > .5:
                    reason = "velocity_stale"
                else:
                    reason = "velocity_not_new"
            except (TypeError, ValueError, OverflowError):
                reason = "invalid_velocity_sample"
            self._reject_scale(reason)

    def yaw_at(self, stamp):
        if not self.yaws or not self.yaws[0][0] <= stamp <= self.yaws[-1][0]:
            return None
        for index, (end, yaw) in enumerate(self.yaws):
            if end == stamp:
                return yaw
            if end > stamp:
                start, previous = self.yaws[index - 1]
                if (end - start > .5
                        or self._source_interval_invalid("attitude", start, end)):
                    return None
                delta = (yaw - previous + math.pi) % (2 * math.pi) - math.pi
                return (previous + delta * (stamp - start) / (end - start) + math.pi) % (2 * math.pi) - math.pi
        return None

    def displacement(self, start, end, *, extrapolate=False):
        """Integrate timestamped NED velocity without bridging missing samples."""
        if not self.velocities or not math.isfinite(start) or not math.isfinite(end) or end < start:
            return None
        # Vision queries reuse the same telemetry several times. Build the
        # trapezoid prefix once per new velocity, not arrays for every pose.
        if self._velocity_cache is None:
            rows = np.asarray(self.velocities, dtype=float)
            times, values = rows[:, 0], rows[:, 1:]
            gaps = np.diff(times)
            integral = np.zeros_like(values)
            integral[1:] = np.cumsum(.5 * (values[:-1] + values[1:]) * gaps[:, None], axis=0)
            self._velocity_cache = times, values, integral, gaps
        times, values, integral, gaps = self._velocity_cache
        if start < times[0] or end > times[-1] + (.5 if extrapolate else 0.):
            return None
        first = max(0, int(np.searchsorted(times, start, side="right")) - 1)
        last = int(np.searchsorted(times, end, side="left"))
        if np.any(gaps[first:last] > .5):
            return None
        # Partial integration still depends on both interpolation brackets.
        # The extrapolated tail also cannot cross a known invalid event.
        coverage_end = max(end, times[min(last, len(times) - 1)])
        if self._source_interval_invalid("velocity", times[first], coverage_end):
            return None

        def primitive(stamp):
            index = min(len(times) - 1, int(np.searchsorted(times, stamp, side="right")) - 1)
            dt = stamp - times[index]
            result = integral[index] + values[index] * dt
            if index + 1 < len(times):
                result = result + .5 * (values[index + 1] - values[index]) * dt * dt / gaps[index]
            return result

        return primitive(end) - primitive(start)

    @staticmethod
    def _map_horizontal(ned, offset):
        north, east = ned
        c, s = math.cos(offset), math.sin(offset)
        return np.array([c * east - s * north, s * east + c * north])

    def _reject_scale(self, reason, *, invalidate=False):
        self.scale_rejections[reason] = self.scale_rejections.get(reason, 0) + 1
        if invalidate:
            # A bad segment never erases coherent whole-flight evidence, but
            # cannot leave a previously trusted control scale enabled either.
            self.scale_m_per_unit = None
            self.scale_reason = reason

    @staticmethod
    def _pose_trusted(pose, now):
        return (pose is not None and pose.map_confirmed and pose.position_observed
                and not pose.reseed_confirming and 0.0 <= now - pose.stamp <= .5
                and np.isfinite(pose.xyz).all())

    def _queue_scale_pose(self, pose, heading_offset):
        if not (heading_offset is not None and math.isfinite(heading_offset)
                and (self._scale_pose_stamp is None or pose.stamp > self._scale_pose_stamp)):
            return
        if self._scale_pose_stamp is not None and pose.stamp - self._scale_pose_stamp > .5:
            self._break_scale_interval("visual_interval_gap")
        self._scale_pose_stamp = float(pose.stamp)
        if len(self.pending) == self.pending.maxlen:
            self._break_scale_interval("pending_overflow")
        # Freeze the capture-aligned map/NED offset with each endpoint.
        # A later heading anchor must not rotate earlier queued evidence.
        self.pending.append((float(pose.stamp), pose.xyz.copy(), float(heading_offset)))

    def _restart_scale_interval(self, sample):
        self.scale_anchor = sample
        self.scale_endpoints.clear()
        self.scale_endpoints.append(sample)

    def _expire_scale_endpoints(self, sample):
        # Keep the oldest covered endpoint in the existing 4.5s bound.
        expired = False
        while (self.scale_endpoints and
               (sample[0] - self.scale_endpoints[0][0] > 4.5 or
                self.scale_endpoints[0][0] < self.velocities[0][0])):
            self.scale_endpoints.popleft()
            expired = True
        if expired:
            self._reject_scale("interval_coverage_expired")
            self.scale_anchor = self.scale_endpoints[0] if self.scale_endpoints else sample

    def _heading_interval_inconsistent(self, offset):
        # Physical yaw may turn freely, but a map/NED frame offset cannot.
        # Inspect every endpoint, not just a candidate's preferred direction.
        return any(abs((row[2] - offset + math.pi) % (2 * math.pi) - math.pi)
                   > math.radians(20) for row in self.scale_endpoints)

    def _observe_scale_sample(self, sample):
        self.scale_endpoints.append(sample)
        if self.scale_anchor is None:
            self.scale_anchor = sample
            return
        self._expire_scale_endpoints(sample)
        start, xyz, offset = self.scale_anchor
        if sample[0] - start < SEG_MIN_S:
            return
        if self._heading_interval_inconsistent(offset):
            self._reject_scale("heading_interval_inconsistent", invalidate=True)
            self._restart_scale_interval(sample)
            return
        displacement = self.displacement(start, sample[0])
        if displacement is None:
            self._reject_scale("telemetry_gap")
            self._restart_scale_interval(sample)
            return
        self._score_scale_interval(sample, xyz, offset, displacement)

    def _score_scale_interval(self, sample, xyz, offset, displacement):
        metric = self._map_horizontal(displacement, offset)
        visual = np.asarray(self.frame.horizontal(sample[1] - xyz))
        metres, units = float(np.linalg.norm(metric)), float(np.linalg.norm(visual))
        if metres < SEG_MIN_M or units < SEG_MIN_VISUAL_U:
            return
        # Independent intervals share at most one endpoint, even on rejection.
        self._restart_scale_interval(sample)
        if float(np.dot(metric, visual)) < math.cos(math.radians(20)) * metres * units:
            self._reject_scale("direction_inconsistent", invalidate=True)
            return
        ratio = metres / units
        outlier = (self.scale_m_per_unit is not None
                   and abs(ratio / self.scale_m_per_unit - 1.) > .20)
        # Include ALL eligible ratios in dispersion/drift. Even a formerly
        # reliable estimate must not hide a newly changing visual scale by
        # rejecting every ratio that disagrees with its own median.
        self.scale_samples.append((ratio, metres))
        self.scale_distance += metres
        self._update_scale(sample[0])
        if outlier:
            self._reject_scale("scale_outlier", invalidate=True)

    def observe_pose(self, pose, now, heading_offset):
        if not self._pose_trusted(pose, now):
            # Strong endpoints cannot certify a displacement through a weak,
            # missing, stale or reseeding visual interval.
            self._break_scale_interval("visual_interval_untrusted")
            return
        if self.last_observed is None or pose.stamp > self.last_observed[0]:
            self.last_observed = (float(pose.stamp), pose.xyz.copy())
        self._queue_scale_pose(pose, heading_offset)
        if not self.velocities:
            return
        while self.pending and self.pending[0][0] <= self.velocities[-1][0]:
            sample = self.pending.popleft()
            if not 0. <= now - sample[0] <= .5:
                self._break_scale_interval("pending_pose_stale")
                continue
            self._observe_scale_sample(sample)

    def _update_scale(self, stamp):
        # Same cumulative endpoint-ratio median/IQR convention as map_scale;
        # online control retains its stricter 20% spread AND drift limits.
        ratios = np.fromiter((row[0] for row in self.scale_samples), dtype=float)
        median = float(np.median(ratios))
        self.scale_candidate = median
        self.scale_spread = float(np.percentile(ratios, 75) - np.percentile(ratios, 25)) / median
        half = len(ratios) // 2
        self.scale_drift = (abs(float(np.median(ratios[half:]) / np.median(ratios[:half])) - 1.)
                            if half else 0.)
        self.scale_stamp = stamp
        self.scale_m_per_unit = None
        if len(ratios) < N_MIN_SEGMENTS:
            self.scale_reason = "insufficient_segments"
        elif self.scale_distance < D_MIN_M:
            self.scale_reason = "insufficient_displacement"
        elif self.scale_spread > .20:
            self.scale_reason = "scale_iqr_unstable"
        elif self.scale_drift > .20:
            self.scale_reason = "scale_drift_unstable"
        else:
            self.scale_m_per_unit = median
            self.scale_reason = "converged"

    def scale_diagnostics(self, now):
        """Read a single session/map-wide scalar; candidates are NOT conversions."""
        fresh = self.scale_stamp is not None and 0. <= now - self.scale_stamp <= 30.
        reliable = self.scale_m_per_unit is not None and fresh
        return {
            "candidate_m_per_unit": self.scale_candidate,
            "scale_m_per_unit": self.scale_m_per_unit if reliable else None,
            "reliable": reliable,
            "scale_segments": len(self.scale_samples),
            "cumulative_distance_m": self.scale_distance,
            # Empirical segment dispersion, not a fabricated confidence interval.
            "relative_iqr": self.scale_spread,
            "relative_drift": self.scale_drift,
            "last_evidence_stamp": self.scale_stamp,
            "reason": self.scale_reason if fresh or self.scale_stamp is None else "scale_stale",
            "rejection_counts": dict(self.scale_rejections),
        }

    def metric_distance(self, distance_u, now):
        """Convert any distance in this map frame only while scale is reliable."""
        scale = self.scale_m_per_unit
        if (scale is None or self.scale_stamp is None or
                not 0. <= now - self.scale_stamp <= 30. or
                not math.isfinite(distance_u) or distance_u < 0.):
            return None
        return float(distance_u) * scale

    def metric_xyz(self, xyz_u, now):
        """Uniform map-origin-relative XYZ in metres; no world/GPS claim."""
        scale = self.metric_distance(1., now)
        xyz = np.asarray(xyz_u, dtype=float)
        if scale is None or xyz.shape != (3,) or not np.isfinite(xyz).all():
            return None
        return xyz * scale

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
        diagnostics = {**self.scale_diagnostics(now), "applied": False}
        diagnostics["scale_reason"] = diagnostics["reason"]
        if (heading_offset is None or self.scale_m_per_unit is None or self.scale_stamp is None
                or not 0.0 <= now - self.scale_stamp <= 30.0):
            diagnostics["reason"] = "scale_or_heading_unavailable"
            return pose.xyz, diagnostics
        if not math.isfinite(heading_offset) or not math.isfinite(radius) or radius <= 0.:
            diagnostics["reason"] = "invalid_prediction_geometry"
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
