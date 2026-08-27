#!/usr/bin/env python3
"""Pose/telemetry sources for the Sphinx ANAFI path-convergence experiment.

Sources (all return raw-map-frame ExpPose: x=north, z=east, y=-up, meters):

  * SphinxTelemetrySource   -- Olympe fused telemetry from the Sphinx ANAFI
                               (PositionChanged / AttitudeChanged / SpeedChanged).
                               This is FUSED simulator telemetry, not raw IMU.
  * KinematicAnafi          -- pure-Python ANAFI-like kinematic plant for
                               Monte Carlo comparison WITHOUT Sphinx. Clearly a
                               kinematic approximation: it is NOT the Sphinx
                               physics engine and is labeled as such in outputs.
  * PerturbedPoseSource     -- wrapper-side fault injection (noise/bias/delay/
                               dropout). Applied ONLY here in the experiment;
                               production code is never modified to pass noise.

Also here (pure logic, testable without Olympe):
  * HeadingEstimator        -- map-frame heading = fused yaw + learned offset
                               (condensed copy of production path_follow_flight
                               logic; copied, not imported, for isolation).
  * check_simulator_ip      -- SIMULATOR-ONLY guard: refuse real ANAFI /
                               SkyController IPs with no override.
"""
from __future__ import annotations

import ipaddress
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from anafi_profile import ANAFI_PROFILE
from controllers import ExpPose
from route_geometry import wrap_angle

# ---------------------------------------------------------------------------
# Simulator-only IP guard (spec "ANAFI / Sphinx / Olympe requirements")

SPHINX_IP_DEFAULT = "10.202.0.1"
REAL_ANAFI_IP = "192.168.42.1"
SKYCONTROLLER_IP = "192.168.53.1"
SPHINX_NETWORK = ipaddress.ip_network("10.202.0.0/16")

BANNER = r"""
############################################################
#                                                          #
#            SPHINX ANAFI SIMULATION ONLY                  #
#                                                          #
#  This harness arms a SIMULATED Parrot ANAFI in Sphinx.   #
#  It must never be pointed at real aircraft.              #
#  Real ANAFI (192.168.42.1) and SkyController             #
#  (192.168.53.1) addresses are refused.                   #
#                                                          #
############################################################
"""


def print_sim_banner() -> None:
    print(BANNER, flush=True)


def check_simulator_ip(ip: str) -> str:
    """Return a validated Sphinx IPv4 address in 10.202.0.0/16.

    There is intentionally no real-aircraft override in this experiment.
    ``ipaddress`` parsing prevents prefix lookalikes such as
    ``10.202.0.1.invalid`` from passing the simulator-only guard.
    """
    raw = str(ip)
    try:
        parsed = ipaddress.ip_address(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, ipaddress.IPv4Address) and parsed in SPHINX_NETWORK:
        return str(parsed)
    kind = ("a REAL ANAFI" if raw == REAL_ANAFI_IP else
            "a SkyController" if raw == SKYCONTROLLER_IP else "not a valid Sphinx simulator")
    raise SystemExit(
        f"REFUSED: --ip {raw} is {kind} address. This experiment is SPHINX "
        f"ANAFI SIMULATION ONLY ({SPHINX_NETWORK}); no real-aircraft override exists.")


# ---------------------------------------------------------------------------
# Heading estimator (condensed from production path_follow_flight.py)

# Refine the map-frame yaw offset only over a motion baseline long enough that
# the true displacement dominates telemetry position noise (a short baseline
# under noisy position turns the motion heading into noise). 0.30 m tolerates
# ~0.1 m position noise; the offset is seeded from the route at takeoff so a
# slower refinement rate costs nothing.
MOVE_EPS_M = 0.30
OFFSET_EMA = 0.20


class HeadingEstimator:
    """Map-frame heading = wrap(fused_yaw + offset). The offset is seeded from
    the route direction at takeoff and refined whenever the drone translates
    (motion heading vs fused yaw). Validates the production heading-offset
    logic against Sphinx fused telemetry."""

    def __init__(self):
        self.offset: float | None = None
        self._last_c: np.ndarray | None = None
        self.motion_heading: float | None = None

    def seed_from_path(self, c0: np.ndarray, path_goal: np.ndarray, fused_yaw: float | None):
        h = math.atan2(float(path_goal[2] - c0[2]), float(path_goal[0] - c0[0]))
        self.motion_heading = h
        if fused_yaw is not None:
            self.offset = wrap_angle(h - fused_yaw)

    def update(self, c: np.ndarray, fused_yaw: float | None):
        c = np.asarray(c, float)
        if self._last_c is None:
            self._last_c = c
            return
        d = c - self._last_c
        if math.hypot(float(d[0]), float(d[2])) < MOVE_EPS_M:
            return          # keep the anchor: displacement accumulates across
                            # high-rate ticks instead of resetting every call
        h = math.atan2(float(d[2]), float(d[0]))
        self.motion_heading = h
        if fused_yaw is not None:
            new = wrap_angle(h - fused_yaw)
            self.offset = new if self.offset is None else \
                wrap_angle(self.offset + OFFSET_EMA * wrap_angle(new - self.offset))
        self._last_c = c

    def heading(self, fused_yaw: float | None) -> float | None:
        if fused_yaw is not None and self.offset is not None:
            return wrap_angle(fused_yaw + self.offset)
        return self.motion_heading


# ---------------------------------------------------------------------------
# Kinematic ANAFI-like plant (NOT Sphinx physics; labeled everywhere)

@dataclass
class KinematicParams:
    """First-order-lag kinematic approximation of a PCMD-flown ANAFI.
    100% stick scales use the public ANAFI capability profile, while the
    controllers retain their conservative ~10-25% command caps. This remains
    a linear plant without aerodynamics; Sphinx is the fidelity step above it.
    """
    v_max_horiz: float = ANAFI_PROFILE.max_horizontal_speed_mps
    v_max_up: float = ANAFI_PROFILE.max_ascent_speed_mps
    v_max_down: float = ANAFI_PROFILE.max_descent_speed_mps
    yaw_rate_max: float = math.radians(ANAFI_PROFILE.max_angular_speed_deg_s)
    integration_hz: float = ANAFI_PROFILE.internal_control_loop_hz
    tau_horiz: float = 0.6          # s, velocity response lag
    tau_vert: float = 0.4
    tau_yaw: float = 0.25


class KinematicAnafi:
    """Plant + pose source in one. Positions in raw map frame (meters)."""

    backend_name = "kinematic (NOT Sphinx -- first-order ANAFI approximation)"

    def __init__(self, start_pos, start_yaw: float, params: KinematicParams | None = None):
        self.kp = params or KinematicParams()
        self.pos = np.asarray(start_pos, dtype=float).copy()
        self.yaw_v = float(start_yaw)
        self.u = 0.0            # body forward velocity (m/s)
        self.v = 0.0            # body right velocity
        self.w_up = 0.0         # vertical velocity, + up
        self.r = 0.0            # yaw rate
        self._cmd = (0, 0, 0, 0)
        self.t = 0.0

    def send_pcmd(self, roll: int, pitch: int, yaw: int, gaz: int):
        c = lambda v: max(-100, min(100, int(v)))
        self._cmd = (c(roll), c(pitch), c(yaw), c(gaz))

    def step(self, dt: float):
        kp = self.kp
        roll, pitch, yaw, gaz = self._cmd
        tgt_u = kp.v_max_horiz * pitch / 100.0
        tgt_v = kp.v_max_horiz * roll / 100.0
        vert_limit = kp.v_max_up if gaz >= 0 else kp.v_max_down
        tgt_w = vert_limit * gaz / 100.0
        tgt_r = kp.yaw_rate_max * yaw / 100.0
        remaining = max(0.0, float(dt))
        substep = 1.0 / max(1.0, float(kp.integration_hz))
        while remaining > 1e-12:
            h = min(substep, remaining)
            self.u += (tgt_u - self.u) * min(1.0, h / kp.tau_horiz)
            self.v += (tgt_v - self.v) * min(1.0, h / kp.tau_horiz)
            self.w_up += (tgt_w - self.w_up) * min(1.0, h / kp.tau_vert)
            self.r += (tgt_r - self.r) * min(1.0, h / kp.tau_yaw)
            self.yaw_v = wrap_angle(self.yaw_v + self.r * h)
            cy, sy = math.cos(self.yaw_v), math.sin(self.yaw_v)
            # body fwd = (cos, sin), body right = (-sin, cos) in (x, z); up = -y
            self.pos[0] += (self.u * cy - self.v * sy) * h
            self.pos[2] += (self.u * sy + self.v * cy) * h
            self.pos[1] -= self.w_up * h
            remaining -= h
        self.t += dt

    # pose-source interface -------------------------------------------------
    def get_pose(self, now: float | None = None) -> ExpPose:
        return ExpPose(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                       stamp=self.t if now is None else float(now))

    def yaw(self, now: float | None = None) -> float:
        """Fused-yaw analogue. In this plant, map frame == body NED frame, so
        the true heading is returned (offset = 0 unless perturbed)."""
        return self.yaw_v

    def velocity_ned(self, now: float | None = None):
        cy, sy = math.cos(self.yaw_v), math.sin(self.yaw_v)
        vn = self.u * cy - self.v * sy
        ve = self.u * sy + self.v * cy
        return (vn, ve, -self.w_up)     # NED: +down


# ---------------------------------------------------------------------------
# Telemetry perturbation wrapper (experiment-side fault injection ONLY)

@dataclass
class PerturbationConfig:
    pose_noise_m: float = 0.0
    pose_noise_max_m: float = 0.0
    yaw_bias_deg: float = 0.0
    yaw_noise_deg: float = 0.0
    yaw_noise_max_deg: float = 0.0
    telemetry_delay_ms: float = 0.0
    telemetry_delay_jitter_ms: float = 0.0
    telemetry_drop_rate: float = 0.0
    hloc_outage_interval_s: float = 0.0
    hloc_outage_duration_s: float = 0.0
    hloc_outage_start_s: float = 0.0
    speed_noise_mps: float = 0.0
    seed: int = 0

    @property
    def label(self) -> str:
        parts = []
        if self.pose_noise_m: parts.append(f"noise{self.pose_noise_m:g}m")
        if self.pose_noise_max_m: parts.append(f"boundednoise{self.pose_noise_max_m:g}m")
        if self.yaw_bias_deg: parts.append(f"yawbias{self.yaw_bias_deg:g}deg")
        if self.yaw_noise_deg: parts.append(f"yawnoise{self.yaw_noise_deg:g}deg")
        if self.yaw_noise_max_deg: parts.append(f"boundedyawnoise{self.yaw_noise_max_deg:g}deg")
        if self.telemetry_delay_ms: parts.append(f"delay{self.telemetry_delay_ms:g}ms")
        if self.telemetry_delay_jitter_ms: parts.append(f"delayjitter{self.telemetry_delay_jitter_ms:g}ms")
        if self.telemetry_drop_rate: parts.append(f"drop{self.telemetry_drop_rate:g}")
        if self.hloc_outage_interval_s and self.hloc_outage_duration_s:
            parts.append(f"outage{self.hloc_outage_duration_s:g}s_every{self.hloc_outage_interval_s:g}s")
        if self.speed_noise_mps: parts.append(f"vnoise{self.speed_noise_mps:g}")
        return "+".join(parts) or "clean"

    @property
    def any_active(self) -> bool:
        return self.label != "clean"


class PerturbedPoseSource:
    """Wraps any pose source; injects noise, yaw bias/noise, delay and dropout
    on the telemetry the CONTROLLER sees. The underlying source stays clean.
    The kinematic backend can log its independent plant state; Sphinx cannot."""

    def __init__(self, inner, cfg: PerturbationConfig):
        self.inner = inner
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._queue: deque = deque()      # (deliver_at, ExpPose, yaw)
        self._last_delivered: ExpPose | None = None
        self._last_yaw: float | None = None

    @staticmethod
    def _copy_pose(raw: ExpPose, x=None, y=None, z=None) -> ExpPose:
        return ExpPose(raw.x if x is None else float(x),
                       raw.y if y is None else float(y),
                       raw.z if z is None else float(z),
                       raw.stamp,
                       source_seq=raw.source_seq,
                       valid=raw.valid,
                       match_count=raw.match_count,
                       inlier_ratio=raw.inlier_ratio,
                       reprojection_error=raw.reprojection_error)

    def _in_hloc_outage(self, now: float) -> bool:
        cfg = self.cfg
        if cfg.hloc_outage_interval_s <= 0 or cfg.hloc_outage_duration_s <= 0:
            return False
        t = max(0.0, float(now) - float(cfg.hloc_outage_start_s))
        if float(now) < float(cfg.hloc_outage_start_s):
            return False
        phase = t % float(cfg.hloc_outage_interval_s)
        return phase < min(float(cfg.hloc_outage_duration_s),
                           float(cfg.hloc_outage_interval_s))

    def _sample_delay_s(self) -> float:
        cfg = self.cfg
        delay_ms = float(cfg.telemetry_delay_ms)
        jitter_ms = float(cfg.telemetry_delay_jitter_ms)
        if jitter_ms > 0:
            delay_ms += float(self.rng.uniform(-jitter_ms, jitter_ms))
        return max(0.0, delay_ms / 1000.0)

    def _enqueue(self, deliver_at: float, pose: ExpPose, yaw: float | None) -> None:
        item = (float(deliver_at), pose, yaw)
        if not self._queue or deliver_at >= self._queue[-1][0]:
            self._queue.append(item)
            return
        for i, (queued_at, _, _) in enumerate(self._queue):
            if deliver_at < queued_at:
                self._queue.insert(i, item)
                return
        self._queue.append(item)

    def _perturb_yaw(self, y: float | None) -> float | None:
        if y is None:
            return None
        cfg = self.cfg
        y = y + math.radians(cfg.yaw_bias_deg)
        if cfg.yaw_noise_max_deg > 0:
            if cfg.yaw_noise_deg > 0:
                err = float(self.rng.normal(0.0, math.radians(cfg.yaw_noise_deg)))
                limit = math.radians(cfg.yaw_noise_max_deg)
                err = max(-limit, min(limit, err))
            else:
                err = float(self.rng.uniform(-math.radians(cfg.yaw_noise_max_deg),
                                             math.radians(cfg.yaw_noise_max_deg)))
            y += err
        elif cfg.yaw_noise_deg > 0:
            y += self.rng.normal(0.0, math.radians(cfg.yaw_noise_deg))
        return wrap_angle(y)

    def get_pose(self, now: float) -> ExpPose | None:
        cfg = self.cfg
        raw = None if self._in_hloc_outage(now) else self.inner.get_pose(now)
        if raw is not None:
            if cfg.telemetry_drop_rate > 0 and self.rng.uniform() < cfg.telemetry_drop_rate:
                raw = None                 # sample lost in transit
        if raw is not None:
            p = self._copy_pose(raw)
            if cfg.pose_noise_max_m > 0:
                if cfg.pose_noise_m > 0:
                    n = self.rng.normal(0.0, cfg.pose_noise_m, size=3)
                else:
                    d = self.rng.normal(0.0, 1.0, size=3)
                    norm = float(np.linalg.norm(d))
                    if norm < 1e-12:
                        n = np.zeros(3)
                    else:
                        radius = float(cfg.pose_noise_max_m) * (float(self.rng.random()) ** (1.0 / 3.0))
                        n = d / norm * radius
                norm = float(np.linalg.norm(n))
                if norm > float(cfg.pose_noise_max_m):
                    n = n / norm * float(cfg.pose_noise_max_m)
                p = self._copy_pose(p, p.x + n[0], p.y + n[1], p.z + n[2])
            elif cfg.pose_noise_m > 0:
                n = self.rng.normal(0.0, cfg.pose_noise_m, size=3)
                p = self._copy_pose(p, p.x + n[0], p.y + n[1], p.z + n[2])
            raw_yaw = self.inner.yaw(now) if hasattr(self.inner, "yaw") else None
            self._enqueue(now + self._sample_delay_s(), p, self._perturb_yaw(raw_yaw))
        out = None
        out_yaw = None
        while self._queue and self._queue[0][0] <= now:
            _, out, out_yaw = self._queue.popleft()
        if out is not None:
            self._last_delivered = out
            self._last_yaw = out_yaw
        return self._last_delivered

    def yaw(self, now: float) -> float | None:
        cfg = self.cfg
        delayed = (cfg.telemetry_delay_ms > 0 or cfg.telemetry_delay_jitter_ms > 0 or
                   cfg.telemetry_drop_rate > 0 or
                   (cfg.hloc_outage_interval_s > 0 and cfg.hloc_outage_duration_s > 0))
        if delayed:
            return self._last_yaw
        return self._perturb_yaw(self.inner.yaw(now))

    def velocity_ned(self, now: float):
        if not hasattr(self.inner, "velocity_ned"):
            return None
        v = self.inner.velocity_ned(now)
        if v is None or self.cfg.speed_noise_mps <= 0:
            return v
        n = self.rng.normal(0.0, self.cfg.speed_noise_mps, size=3)
        return (v[0] + n[0], v[1] + n[1], v[2] + n[2])


# ---------------------------------------------------------------------------
# Sphinx / Olympe fused-telemetry source (imports olympe lazily)

EARTH_R_M = 6378137.0


@dataclass
class GpsOrigin:
    lat: float
    lon: float
    alt: float


def _freeze_payload(value):
    """Hashable marker for get_state-only test doubles or older Olympe APIs."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze_payload(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class OlympeStateTracker:
    """Track reception freshness of one cached Olympe state message.

    ``Drone.get_state`` returns the last cached payload and provides no proof
    that a new event arrived. Modern Olympe exposes ``get_last_event`` whose
    event UUID changes on every reception, including identical hover payloads.
    A get_state-only fallback refreshes only when the payload changes, which is
    intentionally conservative and fail-closed.
    """

    def __init__(self, drone, message):
        self.drone = drone
        self.message = message
        self.marker = None
        self.observed_at: float | None = None
        self.source_date = None
        self.state = None
        self.sequence = 0
        self.last_read_ok = False

    def read(self, now: float):
        try:
            event = self.drone.get_last_event(self.message)
            state = dict(event.args)
            marker = ("event", str(event.uuid))
            source_date = getattr(event, "date", None)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            try:
                state = dict(self.drone.get_state(self.message))
            except (AttributeError, KeyError, RuntimeError, TypeError):
                self.last_read_ok = False
                return None, self.observed_at, self.sequence, False
            marker = ("payload", _freeze_payload(state))
            source_date = None

        is_new = marker != self.marker
        if is_new:
            self.marker = marker
            self.observed_at = float(now)
            self.source_date = source_date
            self.state = state
            self.sequence += 1
        self.last_read_ok = True
        return dict(self.state), self.observed_at, self.sequence, is_new

    def fresh_state(self, now: float, max_age_s: float):
        state, stamp, seq, is_new = self.read(now)
        if state is None or stamp is None or now - stamp > max_age_s:
            return None, stamp, seq, is_new
        return state, stamp, seq, is_new


class SphinxTelemetrySource:
    """Fused Olympe telemetry from the SIMULATED ANAFI (same GPS->local
    conversion as the production sphinx smoke test): raw map frame is
    x=north, z=east, y=-up, meters, origin at the first valid fix.

    Not raw IMU: AttitudeChanged / PositionChanged / SpeedChanged are the
    drone firmware's fused states.

    Rate fusion: Sphinx PositionChanged (GPS) reports at ~1 Hz while the
    control loop runs at 20 Hz and SpeedChanged/AttitudeChanged report at
    ~5 Hz. `get_pose` dead-reckons the position between GPS fixes by
    integrating fresh NED velocity events and snapping to each fresh GPS event.
    Pose stamps advance only when a new Olympe event UUID is observed; repeatedly
    reading a cached state cannot hide a frozen telemetry stream.

    `get_fused_reference` exposes the firmware-fused PositionChanged state for
    diagnostics. It is neither independent nor exact simulator ground truth and
    must not be labeled or scored as such.
    """

    backend_name = "sphinx (Parrot Sphinx ANAFI, Olympe fused telemetry, 1Hz GPS + 5Hz velocity DR)"

    def __init__(self, drone, messages=None, max_event_age_s: float = 0.6,
                 max_position_age_s: float = 2.0):
        if messages is None:
            from olympe.messages.ardrone3.PilotingState import (AttitudeChanged,
                                                                PositionChanged,
                                                                SpeedChanged)
            messages = (PositionChanged, AttitudeChanged, SpeedChanged)
        self._Pos, self._Att, self._Spd = messages
        self.drone = drone
        self.max_event_age_s = float(max_event_age_s)
        self.max_position_age_s = float(max_position_age_s)
        self._pos_events = OlympeStateTracker(drone, self._Pos)
        self._att_events = OlympeStateTracker(drone, self._Att)
        self._spd_events = OlympeStateTracker(drone, self._Spd)
        self.origin: GpsOrigin | None = None
        self._last_fix_seq = None      # PositionChanged event sequence of DR anchor
        self._est = None               # dead-reckoned [x, y, z] raw-frame estimate
        self._last_now = None          # time of last get_pose (DR integration base)
        self._pose_seq = 0
        self._last_pose_stamp = None

    @staticmethod
    def _valid(st) -> bool:
        if not st:
            return False
        lat, lon, alt = (float(st.get(k, 500.0)) for k in ("latitude", "longitude", "altitude"))
        return all(math.isfinite(v) for v in (lat, lon, alt)) and abs(lat) <= 90 and abs(lon) <= 180

    def _raw_fix(self, st):
        """Firmware-fused position as raw-frame (north, east, up) or None."""
        if not self._valid(st):
            return None
        if self.origin is None:
            self.origin = GpsOrigin(float(st["latitude"]), float(st["longitude"]),
                                    float(st["altitude"]))
        lat = math.radians(float(st["latitude"]))
        lon = math.radians(float(st["longitude"]))
        lat0 = math.radians(self.origin.lat)
        lon0 = math.radians(self.origin.lon)
        north = (lat - lat0) * EARTH_R_M
        east = (lon - lon0) * EARTH_R_M * math.cos(lat0)
        up = float(st["altitude"]) - self.origin.alt
        return north, east, up

    def get_fused_reference(self, now: float) -> ExpPose | None:
        """Latest firmware-fused GPS position, preserving event freshness."""
        st, stamp, seq, _ = self._pos_events.read(now)
        if stamp is None or now - stamp > self.max_position_age_s:
            return None
        f = self._raw_fix(st)
        if f is None:
            return None
        north, east, up = f
        return ExpPose(north, -up, east, stamp=stamp, source_seq=seq)

    def get_pose(self, now: float) -> ExpPose | None:
        """Controller input: GPS position dead-reckoned with NED velocity to the
        control rate, snapped to each fresh GPS event. The stamp is the newest
        contributing PositionChanged or SpeedChanged event observation."""
        st, pos_stamp, pos_seq, _ = self._pos_events.read(now)
        f = self._raw_fix(st)
        if f is None:
            return None
        if pos_stamp is None or now - pos_stamp > self.max_position_age_s:
            return None
        north, east, up = f
        if self._est is None or pos_seq != self._last_fix_seq:
            # First sample or fresh GPS event: snap the DR anchor to the fused fix.
            self._est = [north, -up, east]
            self._last_fix_seq = pos_seq

        speed_state, speed_stamp, _speed_seq, _ = self._spd_events.read(now)
        speed_fresh = (speed_state is not None and speed_stamp is not None and
                       now - speed_stamp <= self.max_event_age_s)
        if self._last_now is not None and speed_fresh:
            dt = now - self._last_now
            if dt > 0:
                try:
                    vn, ve, vd = (float(speed_state["speedX"]),
                                  float(speed_state["speedY"]),
                                  float(speed_state["speedZ"]))
                except (KeyError, TypeError, ValueError):
                    speed_fresh = False
                else:
                    if all(math.isfinite(v) for v in (vn, ve, vd)):
                        self._est[0] += vn * dt      # x = north
                        self._est[2] += ve * dt      # z = east
                        self._est[1] += vd * dt      # y = -up; +speedZ is down
                    else:
                        speed_fresh = False
        self._last_now = now
        pose_stamp = max(pos_stamp, speed_stamp) if speed_fresh else pos_stamp
        if pose_stamp != self._last_pose_stamp:
            self._pose_seq += 1
            self._last_pose_stamp = pose_stamp
        return ExpPose(self._est[0], self._est[1], self._est[2],
                       stamp=pose_stamp, source_seq=self._pose_seq)

    def yaw(self, now: float) -> float | None:
        st, _stamp, _seq, _ = self._att_events.fresh_state(
            now, self.max_event_age_s)
        if st is None:
            return None
        try:
            y = float(st["yaw"])
        except (KeyError, TypeError, ValueError):
            return None
        return y if math.isfinite(y) else None

    def velocity_ned(self, now: float):
        st, _stamp, _seq, _ = self._spd_events.fresh_state(
            now, self.max_event_age_s)
        if st is None:
            return None
        try:
            v = (float(st["speedX"]), float(st["speedY"]), float(st["speedZ"]))
        except (KeyError, TypeError, ValueError):
            return None
        return v if all(math.isfinite(x) for x in v) else None

    def telemetry_healthy(self, now: float | None = None) -> bool:
        """True only while fresh position/speed-derived pose and yaw events exist."""
        if now is None:
            import time
            now = time.monotonic()
        pose = self.get_pose(now)
        return (pose is not None and now - pose.stamp <= self.max_event_age_s and
                self.yaw(now) is not None)
