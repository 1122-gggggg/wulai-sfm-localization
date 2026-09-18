#!/usr/bin/env python3
"""Shared synthetic visual-localizer boundary for offline route autoflight.

Wraps true meters into reported map-unit poses with misalignment residue,
capture cadence, latency/jitter delivery, correlated OU noise, bias walk,
dropouts/outages, the production IMU-bridge law during blackout, drifting
weak estimates during localization-failure windows, and a height gain. Owns
the capture/receive contract used by the fast simulator and the Sphinx
production-route adapter so both consume one implementation.

Only numpy, Pose, and the standard library. Never Olympe or UI.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class SimulatedPose:
    x: float
    y: float
    z: float
    yaw: float
    stamp: float
    map_confirmed: bool = True
    reseed_confirming: bool = False
    position_observed: bool = True

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


def _make_pose(pose_cls, x, y, z, yaw, stamp, *, map_confirmed=True, position_observed=True):
    try:
        return pose_cls(float(x), float(y), float(z), float(yaw), stamp=float(stamp),
                        map_confirmed=bool(map_confirmed),
                        position_observed=bool(position_observed))
    except TypeError:
        pose = pose_cls(float(x), float(y), float(z), float(yaw), stamp=float(stamp))
        try:
            pose.map_confirmed = bool(map_confirmed)
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            pose.position_observed = bool(position_observed)
        except (AttributeError, TypeError, ValueError):
            pass
        return pose


@dataclass
class SimulatedLocalizerConfig:
    s: float = 10.0
    yaw_bias_deg: float = 0.0
    pos_bias_u: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_error_pct: float = 0.0
    pos_sigma_u: float = 0.004
    yaw_sigma_deg: float = 1.0
    latency_ms: float = 120.0
    drop_rate: float = 0.02
    outage_every_s: float = 0.0
    outage_dur_s: float = 0.0
    jump_rate: float = 0.0
    jump_max_u: float = 0.0
    outlier_rate: float = 0.0
    outlier_max_u: float = 0.0
    weak_rate: float = 0.0
    weak_noise_gain: float = 3.0
    predicted_rate: float = 0.0
    blackout_every_s: float = 0.0
    blackout_dur_s: float = 0.0
    blackout_drift_mps: float = 0.0
    blackout_yaw_drift_deg_s: float = 0.0
    pose_rate_hz: float = 20.0
    latency_jitter_ms: float = 0.0
    position_correlation_s: float = 0.0
    yaw_correlation_s: float = 0.0
    bias_walk_u_sqrt_s: float = 0.0
    reseed_confirm_frames: int = 0
    imu_bridge_max_age_s: float = 10.0
    imu_bridge_max_frames: int = 300
    # Localization-failure windows: every drift_every_s (first one at
    # drift_offset_s) for drift_dur_s the aircraft does not know where it is.
    # Production keeps publishing VO / dead-reckon estimates then, so each
    # capture is a weak, position-observed pose that drifts from the last map
    # fix: drift_gain x true displacement plus drift_mps in a random
    # horizontal direction per window. Before any map fix there is no pose.
    drift_every_s: float = 0.0
    drift_dur_s: float = 0.0
    drift_offset_s: float = 0.0
    drift_gain: float = 1.0
    drift_mps: float = 0.0
    # Reported height change = vertical_gain x true change since the first
    # capture. 1 is exact; 0 means the localized height never moves.
    vertical_gain: float = 1.0
    # Map up axis; required by drift_mps > 0 or vertical_gain != 1.
    map_up_u: tuple[float, float, float] | None = None

class SimulatedLocalizer:
    """Capture-cadence pose source with bounded jittered delivery."""

    def __init__(self, config=None, rng=None, *, pose_cls=None, tick_dt=0.05, **overrides):
        self.config = config if config is not None else SimulatedLocalizerConfig(**overrides)
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._validate()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._pose_cls = pose_cls
        self._tick_dt = float(tick_dt) if tick_dt else 0.05
        self.s = float(self.config.s)
        self.yaw_bias = math.radians(float(self.config.yaw_bias_deg))
        self.pos_bias = np.asarray(self.config.pos_bias_u, float)
        self.scale_err = float(self.config.scale_error_pct) / 100.0
        self.pos_sigma = float(self.config.pos_sigma_u)
        self.yaw_sigma = math.radians(float(self.config.yaw_sigma_deg))
        self.latency = float(self.config.latency_ms) / 1000.0
        self.drop_rate = float(self.config.drop_rate)
        self.outage_every = float(self.config.outage_every_s)
        self.outage_dur = float(self.config.outage_dur_s)
        self.jump_rate = float(self.config.jump_rate)
        self.jump_max_u = float(self.config.jump_max_u)
        self.outlier_rate = float(self.config.outlier_rate)
        self.outlier_max_u = float(self.config.outlier_max_u)
        self.weak_rate = float(self.config.weak_rate)
        self.weak_noise_gain = float(self.config.weak_noise_gain)
        self.predicted_rate = float(self.config.predicted_rate)
        self.blackout_every = float(self.config.blackout_every_s)
        self.blackout_dur = float(self.config.blackout_dur_s)
        self.blackout_drift = float(self.config.blackout_drift_mps)
        self.blackout_yaw_drift = math.radians(float(self.config.blackout_yaw_drift_deg_s))
        self.pose_rate = float(self.config.pose_rate_hz)
        self.jitter = float(self.config.latency_jitter_ms) / 1000.0
        self.pos_tau = float(self.config.position_correlation_s)
        self.yaw_tau = float(self.config.yaw_correlation_s)
        self.bias_walk = float(self.config.bias_walk_u_sqrt_s)
        self.reseed_confirm_frames = int(self.config.reseed_confirm_frames)
        self.imu_bridge_max_age_s = float(self.config.imu_bridge_max_age_s)
        self.imu_bridge_max_frames = int(self.config.imu_bridge_max_frames)
        self.drift_every = float(self.config.drift_every_s)
        self.drift_dur = float(self.config.drift_dur_s)
        self.drift_offset = float(self.config.drift_offset_s)
        self.drift_gain = float(self.config.drift_gain)
        self.drift_mps = float(self.config.drift_mps)
        self.vertical_gain = float(self.config.vertical_gain)
        self._up = None
        if self.config.map_up_u is not None:
            up = np.asarray(self.config.map_up_u, float)
            self._up = up / float(np.linalg.norm(up))
        self._height_ref: float | None = None
        # Last map fix (reported, observed truth) and the active drift window.
        self._map_fix: tuple[np.ndarray, np.ndarray] | None = None
        self._drift: tuple[np.ndarray, np.ndarray, float, np.ndarray] | None = None
        self._truth: list[tuple[float, np.ndarray, float]] = []
        self._pending: list[tuple[float, float, object]] = []
        self._next_capture = 0.0
        self._capture_initialized = False
        self._pos_err = np.zeros(3)
        self._yaw_err = 0.0
        self._bias = np.zeros(3)
        self._last_capture_pose = None
        self._last_report = None
        self._prev_report = None
        self._last_delivered_capture: float | None = None
        self._dropped_late = 0
        self.map_confirmed = True
        self.fault = "ok"
        self._blackout_active = False
        self._blackout_anchor: np.ndarray | None = None
        self._blackout_anchor_yaw = 0.0
        self._blackout_start = 0.0
        self._blackout_true_yaw = 0.0
        self._blackout_visual_capture = 0.0
        self._blackout_frames = 0
        self._reseed_remaining = 0
        self._last_seq = 0

    def _validate(self) -> None:
        cfg = self.config
        for name in ("s", "pose_rate_hz"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("pos_sigma_u", "yaw_sigma_deg", "latency_ms", "outage_every_s",
                     "outage_dur_s", "jump_rate", "jump_max_u", "outlier_rate",
                     "outlier_max_u", "weak_rate", "weak_noise_gain", "predicted_rate",
                     "blackout_every_s", "blackout_dur_s", "blackout_drift_mps",
                     "blackout_yaw_drift_deg_s", "latency_jitter_ms",
                     "position_correlation_s", "yaw_correlation_s", "bias_walk_u_sqrt_s"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not math.isfinite(float(cfg.yaw_bias_deg)):
            raise ValueError("yaw_bias_deg must be finite")
        if not math.isfinite(float(cfg.scale_error_pct)):
            raise ValueError("scale_error_pct must be finite")
        if not 0.0 <= float(cfg.drop_rate) <= 1.0:
            raise ValueError("drop_rate must be in [0, 1]")
        frames = cfg.reseed_confirm_frames
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise ValueError("reseed_confirm_frames must be a non-negative int")
        if not math.isfinite(float(cfg.imu_bridge_max_age_s)) or float(cfg.imu_bridge_max_age_s) <= 0.0:
            raise ValueError("imu_bridge_max_age_s must be finite and > 0")
        bridge_frames = cfg.imu_bridge_max_frames
        if isinstance(bridge_frames, bool) or not isinstance(bridge_frames, int) or bridge_frames <= 0:
            raise ValueError("imu_bridge_max_frames must be a positive int")
        for name in ("drift_every_s", "drift_dur_s", "drift_offset_s", "drift_gain", "drift_mps"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not math.isfinite(float(cfg.vertical_gain)):
            raise ValueError("vertical_gain must be finite")
        if cfg.map_up_u is not None:
            up = np.asarray(cfg.map_up_u, float)
            if up.shape != (3,) or not np.all(np.isfinite(up)) or float(np.linalg.norm(up)) <= 1e-9:
                raise ValueError("map_up_u must be a finite nonzero 3-vector")
        elif float(cfg.drift_mps) > 0.0 or float(cfg.vertical_gain) != 1.0:
            raise ValueError("drift_mps and vertical_gain need map_up_u")

    @property
    def dropped_late(self) -> int:
        return int(self._dropped_late)

    def push_truth(self, t: float, position_m, yaw: float) -> None:
        try:
            from real_path_follow_controller import Pose as _RpfPose
            pose_cls = self._pose_cls or _RpfPose
        except ImportError:
            pose_cls = self._pose_cls or SimulatedPose
        self._pose_cls = pose_cls
        self._truth.append((float(t), np.array(position_m, float).copy(), float(yaw)))
        horizon = float(t) - self.latency - abs(self.jitter) - 2.0
        while len(self._truth) > 2 and self._truth[0][0] < horizon:
            self._truth.pop(0)
        capacity = int(math.ceil(self.pose_rate * (self.latency + abs(self.jitter)))) + 2
        while len(self._pending) > max(2, capacity):
            self._pending.pop(0)
            self._dropped_late += 1

    def _in_outage(self, t: float) -> bool:
        return (self.outage_every > 0 and self.outage_dur > 0
                and (float(t) % self.outage_every) < self.outage_dur)

    def _in_blackout(self, t: float) -> bool:
        return (self.blackout_every > 0 and self.blackout_dur > 0
                and (float(t) % self.blackout_every) < self.blackout_dur)

    def _truth_at(self, want: float):
        sample = self._truth[0]
        for entry in self._truth:
            if entry[0] <= want:
                sample = entry
            else:
                break
        return sample

    def _capture_due(self, t: float) -> bool:
        if not self._capture_initialized:
            self._next_capture = float(t)
            self._capture_initialized = True
            return True
        period = 1.0 / self.pose_rate
        if float(t) + 1e-12 >= self._next_capture:
            while self._next_capture <= float(t) + 1e-12:
                self._next_capture += period
            return True
        return False

    def _ou_step(self, current: float, sigma: float, tau: float, dt: float) -> float:
        if tau <= 0.0 or sigma <= 0.0 or dt <= 0.0:
            return 0.0 if sigma <= 0.0 else float(self.rng.normal(0.0, sigma))
        alpha = math.exp(-dt / tau)
        scale = sigma * math.sqrt(max(0.0, 1.0 - alpha * alpha))
        return float(current * alpha + self.rng.normal(0.0, scale))

    def _make_capture(self, capture_t: float):
        _, pos_m, yaw = self._truth_at(capture_t)
        dt_capture = 1.0 / self.pose_rate
        pos_noise = np.zeros(3)
        for axis in range(3):
            pos_noise[axis] = self._ou_step(float(self._pos_err[axis]), self.pos_sigma, self.pos_tau, dt_capture)
        self._pos_err = pos_noise.copy()
        yaw_noise = self._ou_step(self._yaw_err, self.yaw_sigma, self.yaw_tau, dt_capture)
        self._yaw_err = float(yaw_noise)
        if self.bias_walk > 0.0 and dt_capture > 0.0:
            self._bias = self._bias + self.rng.normal(0.0, self.bias_walk * math.sqrt(dt_capture), 3)
        noise_gain = 1.0
        map_confirmed = True
        fault = "ok"
        if self.weak_rate > 0 and self.rng.random() < self.weak_rate:
            noise_gain = max(1.0, float(self.weak_noise_gain))
            fault = "weak"
            map_confirmed = False
        observed = self._observed_position(pos_m)
        if self._in_drift(capture_t):
            if self._map_fix is None:
                return None, None, "drift-no-fix"
            noise_gain = max(1.0, float(self.weak_noise_gain))
            fault = "drift"
            map_confirmed = False
            p_map = self._drift_estimate(capture_t, observed) + pos_noise * noise_gain
        else:
            self._drift = None
            p_map = observed + self.pos_bias + self._bias + pos_noise * noise_gain
            if map_confirmed:
                self._map_fix = (p_map.copy(), observed.copy())
        yaw_rep = yaw + self.yaw_bias + yaw_noise * noise_gain
        if self.outlier_rate > 0 and self.rng.random() < self.outlier_rate:
            base = np.array(self._last_report.xyz, float) if self._last_report is not None else p_map
            p_map = base + self.rng.uniform(-self.outlier_max_u, self.outlier_max_u, 3)
            fault = "outlier"
        if self.jump_rate > 0 and self.rng.random() < self.jump_rate:
            jdir = self.rng.normal(0, 1, 3)
            jdir /= float(np.linalg.norm(jdir)) or 1.0
            p_map = p_map + jdir * float(self.rng.uniform(0, self.jump_max_u))
            fault = "jump" if fault == "ok" else fault + "+jump"
        pose = _make_pose(self._pose_cls, float(p_map[0]), float(p_map[1]), float(p_map[2]),
                          float(yaw_rep), float(capture_t), map_confirmed=map_confirmed)
        jitter_s = float(self.rng.uniform(-self.jitter, self.jitter)) if self.jitter > 0 else 0.0
        receive = float(capture_t) + max(0.0, self.latency + jitter_s)
        self._last_capture_pose = pose
        self.fault = fault
        self.map_confirmed = bool(map_confirmed)
        return receive, pose, fault

    def _observed_position(self, pos_m) -> np.ndarray:
        """True map position as the localizer resolves it (height gain applied)."""
        p_map = np.asarray(pos_m, float) / (self.s * (1.0 + self.scale_err))
        if self.vertical_gain == 1.0:
            return p_map
        height = float(np.dot(p_map, self._up))
        if self._height_ref is None:
            self._height_ref = height
        return p_map + (self.vertical_gain - 1.0) * (height - self._height_ref) * self._up

    def _in_drift(self, t: float) -> bool:
        if self.drift_every <= 0.0 or self.drift_dur <= 0.0 or float(t) < self.drift_offset:
            return False
        return ((float(t) - self.drift_offset) % self.drift_every) < self.drift_dur

    def _drift_estimate(self, capture_t: float, observed: np.ndarray) -> np.ndarray:
        """Weak estimate carried from the last map fix; the window opens on first use."""
        if self._drift is None:
            fix_est, fix_observed = self._map_fix
            velocity = np.zeros(3)
            if self.drift_mps > 0.0:
                direction = self.rng.normal(0.0, 1.0, 3)
                direction -= float(np.dot(direction, self._up)) * self._up
                direction /= float(np.linalg.norm(direction)) or 1.0
                velocity = direction * self.drift_mps / self.s
            self._drift = (fix_est, fix_observed, float(capture_t), velocity)
        fix_est, fix_observed, start, velocity = self._drift
        return (fix_est + self.drift_gain * (observed - fix_observed)
                + velocity * (float(capture_t) - start))

    def report(self, t: float):
        self.fault = "ok"
        self.map_confirmed = True
        if not self._truth:
            return None
        if self._in_outage(t):
            self._blackout_active = False
            self.fault = "outage"
            return None
        if self._in_blackout(t):
            return self._report_blackout(t)
        self._blackout_active = False
        capture_tick = None
        if self._capture_due(t):
            capture_tick = self._next_capture - 1.0 / self.pose_rate
            if capture_tick > float(t):
                capture_tick = float(t)
        if capture_tick is not None and self.rng.random() >= self.drop_rate:
            receive, pose, fault = self._make_capture(float(capture_tick))
            if pose is None:
                # No position at all (lost before any map fix): like a dropout.
                self.fault = fault
                self.map_confirmed = bool(getattr(self._last_report, "map_confirmed", True))
                return self._last_report
            # Capture stamp is the cadence tick, not the poll time.
            capture = float(pose.stamp)
            self._pending.append((receive, capture, pose))
            self._pending.sort(key=lambda row: (row[0], row[1]))
            self.fault = fault
        elif capture_tick is not None:
            self.fault = "dropout"
            self.map_confirmed = bool(getattr(self._last_report, "map_confirmed", True))
            return self._last_report
        ready = [row for row in self._pending if row[0] <= float(t) + 1e-12]
        if not ready:
            # Hold the last delivered pose value/stamp across polls with no
            # newly matured capture (e.g. 10 Hz poses polled at 20 Hz).
            return self._last_report
        # Deliver newest ready capture; count older late arrivals as dropped.
        ready.sort(key=lambda row: (row[1], row[0]))
        newest = ready[-1]
        for row in ready[:-1]:
            if self._last_delivered_capture is None or row[1] > self._last_delivered_capture:
                self._dropped_late += 1
        self._pending = [row for row in self._pending if row[0] > float(t) + 1e-12
                         or (row[1] == newest[1] and row[0] == newest[0])]
        # Remove all other ready rows.
        self._pending = [row for row in self._pending if row[0] > float(t) + 1e-12]
        _, capture, pose = newest
        if self._last_delivered_capture is not None and capture <= self._last_delivered_capture:
            return None
        if self.predicted_rate > 0 and self._prev_report is not None and self._last_report is not None \
                and self.rng.random() < self.predicted_rate:
            dt = self._last_report.stamp - self._prev_report.stamp
            if dt > 1e-6:
                vel = (np.array(self._last_report.xyz) - np.array(self._prev_report.xyz)) / dt
                pred = np.array(self._last_report.xyz) + vel * (t - self._last_report.stamp)
                pose = _make_pose(self._pose_cls, float(pred[0]), float(pred[1]), float(pred[2]),
                                  float(self._last_report.yaw), float(t), map_confirmed=False,
                                  position_observed=False)
                self._prev_report = self._last_report
                self._last_report = pose
                self._last_delivered_capture = float(capture)
                self.fault = "predicted"
                self.map_confirmed = False
                return pose
        self._last_delivered_capture = float(capture)
        try:
            position_observed = bool(getattr(pose, "position_observed", True))
        except (AttributeError, TypeError, ValueError):
            position_observed = True
        if position_observed:
            try:
                self._blackout_visual_capture = float(pose.stamp)
            except (AttributeError, TypeError, ValueError, OverflowError):
                pass
            self._blackout_frames = 0
        if self._reseed_remaining > 0:
            self._reseed_remaining -= 1
            try:
                pose.reseed_confirming = True
            except (AttributeError, TypeError, ValueError):
                pass
        self._prev_report = self._last_report
        self._last_report = pose
        self.fault = getattr(pose, "_sim_fault", self.fault)
        return pose

    def _report_blackout(self, t: float):
        if self._last_report is None:
            self.fault = "blackout-no-anchor"
            return None
        if not self._blackout_active:
            self._blackout_active = True
            self._blackout_anchor = np.array(self._last_report.xyz, float)
            self._blackout_anchor_yaw = float(self._last_report.yaw)
            self._blackout_start = float(t)
            _, _, yaw = self._truth_at(float(t) - self.latency)
            self._blackout_true_yaw = float(yaw)
            try:
                self._blackout_visual_capture = float(self._last_report.stamp)
            except (AttributeError, TypeError, ValueError, OverflowError):
                self._blackout_visual_capture = float(t)
            self._blackout_frames = 0
        age = max(0.0, float(t) - float(self._blackout_visual_capture))
        if age > float(self.imu_bridge_max_age_s) or int(self._blackout_frames) >= int(self.imu_bridge_max_frames):
            self.fault = "blackout-expired"
            return None
        self._blackout_frames = int(self._blackout_frames) + 1
        dt = max(0.0, float(t) - self._blackout_start)
        _, _, yaw = self._truth_at(float(t) - self.latency)
        true_dyaw = float(yaw) - float(self._blackout_true_yaw)
        yaw_walk = self.rng.normal(0.0, math.radians(1.0) * math.sqrt(max(dt, 0.0)))
        yaw_walk += self.rng.normal(0.0, float(self.blackout_yaw_drift) * math.sqrt(max(dt, 0.0)))
        est_yaw = float(self._blackout_anchor_yaw) + true_dyaw + yaw_walk
        drift_step = float(self.blackout_drift) / max(self.s, 1e-9) * 0.05
        self._blackout_anchor = np.array(self._blackout_anchor, float) + self.rng.normal(0.0, drift_step, 3)
        pose = _make_pose(self._pose_cls, float(self._blackout_anchor[0]),
                          float(self._blackout_anchor[1]), float(self._blackout_anchor[2]),
                          float(est_yaw), float(t), map_confirmed=False,
                          position_observed=False)
        self._prev_report = self._last_report
        self._last_report = pose
        self.fault = "blackout-imu"
        self.map_confirmed = False
        self._reseed_remaining = int(self.reseed_confirm_frames)
        return pose

    def note_recovery(self) -> None:
        self._reseed_remaining = int(self.reseed_confirm_frames)


@dataclass
class CommandChannelConfig:
    latency_ms: float = 0.0
    drop_rate: float = 0.0
    ttl_s: float = 0.2


class CommandChannel:
    """Bounded command FIFO: plant applies the newest unexpired arrived command."""

    def __init__(self, config=None, rng=None, **overrides):
        self.config = config if config is not None else CommandChannelConfig(**overrides)
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._validate()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._queue: list[tuple[float, int, tuple[int, int, int, int]]] = []
        self._seq = 0
        self._applied_seq = -1

    def _validate(self) -> None:
        latency = float(self.config.latency_ms)
        drop = float(self.config.drop_rate)
        ttl = float(self.config.ttl_s)
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("command latency must be finite and >= 0")
        if not math.isfinite(drop) or not 0.0 <= drop <= 1.0:
            raise ValueError("command drop rate must be in [0, 1]")
        if not math.isfinite(ttl) or ttl <= 0.0:
            raise ValueError("command TTL must be finite and > 0")

    def send(self, pcmd, now: float) -> int | None:
        if self.rng.random() < float(self.config.drop_rate):
            return None
        self._seq += 1
        receive = float(now) + float(self.config.latency_ms) / 1000.0
        self._queue.append((receive, int(self._seq), (int(pcmd[0]), int(pcmd[1]), int(pcmd[2]), int(pcmd[3]))))
        capacity = 8
        while len(self._queue) > capacity:
            self._queue.pop(0)
        return int(self._seq)

    def applied(self, now: float) -> tuple[tuple[int, int, int, int], float, int | None]:
        ready = [row for row in self._queue if row[0] <= float(now) + 1e-12]
        if not ready:
            return (0, 0, 0, 0), float("inf"), None
        ready.sort(key=lambda row: (row[1], row[0]))
        receive, seq, pcmd = ready[-1]
        age = float(now) - float(receive)
        if age > float(self.config.ttl_s):
            return (0, 0, 0, 0), age, None
        # Older sequences never override a newer applied command.
        if seq < self._applied_seq:
            return (0, 0, 0, 0), age, None
        self._applied_seq = int(seq)
        return pcmd, age, int(seq)
