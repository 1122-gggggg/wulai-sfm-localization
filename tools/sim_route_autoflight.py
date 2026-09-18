#!/usr/bin/env python3
"""Scale-free route-autoflight closed-loop simulator (no drone, no Sphinx).

What it proves
-------------
The production AUTO law (`RouteAutoController.step` + `YawAlignedPcmdController`
/ `command_to_body_percent` in `定位演算法/flight_control/`) only ever sees:

  * the drone's localized point (map frame, arbitrary map units), and
  * the drawn route (same map frame, same arbitrary units).

It never sees a scale. This script closes the loop around the REAL controller
(no reimplementation of the control law) with a realistic ANAFI plant:

  * true drone state lives in METERS in its own world frame,
  * the localizer reports that state back in MAP UNITS through an unknown
    Sim(3)-ish misalignment (scale + yaw bias + position bias + noise +
    latency + dropouts),
  * the controller answers with body PCMD (roll/pitch/yaw/gaz %) and
    separate yaw/horizontal-motion gates, exactly as in the controller,
  * the plant integrates those PCMD % into meter motion.

Success = start at waypoint 1 and fly every waypoint in order, then return to
waypoint 1 (`return_to_start`, the production default) and LAND there, while
the route-authored arrival spheres (`arrive_radius_map_units` / per-point
radii) gate every waypoint. The controller prioritizes returning to the active
segment when tracking error grows; this simulator adds no hard safety tube.

Frames ("兩者並沒有對齊")
------------------------
Map frame (route + localization output, map units) and the drone's true world
frame (meters) differ by scale ``s`` (meters per map unit) plus optional
residual calibration biases (``--yaw-bias-deg``, ``--pos-bias-u``,
``--scale-error-pct``). A perfect localizer inverts the misalignment; the
bias knobs leave a residual error in to prove the direction-based law still
converges ("應該是沒問題"). Takeoff away from waypoint 1 is the normal case:
AUTO still starts at waypoint 1 (`start_after_nearest_waypoint`), and the
deviation failsafe stays disarmed until that start target is reached -- same as
`path_follow_flight._FlightLoopRunner`.

Gaz/yaw percentages use configured desktop envelopes (2 m/s and 20 deg/s by
default, matching the 2026-09-14 flight logs). Roll/pitch are tilt percentages,
not velocity percentages: the horizontal plant approximates 0.45 m/s at the
3% tilt cap. This is not an identified aircraft model. First-order velocity
lag, wind, per-axis hover anchors, integer PCMD, 20 Hz control / 80 Hz integration,
and localization defects are included. Long outages fail this simulation;
they do not issue a live landing.

Usage
-----
    .venv/bin/python tools/sim_route_autoflight.py \\
        --route 地圖檔/場域/river_site/routes/flight_route_20260912_144954_c495094b.json \\
        --out outputs/sim_autoflight/demo

    # robustness sweep over unknown scales (headless, no plots per run):
    .venv/bin/python tools/sim_route_autoflight.py --sweep-scales 5,10,20 --seeds 3 --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FLIGHT_ROOT = REPO_ROOT / "定位演算法" / "flight_control"
if str(FLIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(FLIGHT_ROOT))
OPERATOR_ROOT = REPO_ROOT / "控制介面程式/operator_interface"
if str(OPERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_ROOT))

# Never touch hardware SDKs in the simulator.
sys.modules.setdefault("olympe", None)

import real_path_follow_controller as rpf  # noqa: E402
from route_domain import RouteDocument  # noqa: E402
from operator_autonomy import DesktopRouteAutonomy, _fresh_route_progress  # noqa: E402

CTRL_HZ = 20
DT = 1.0 / CTRL_HZ
VH_MAX_MPS = 15.0
VV_MAX_MPS = 4.0
YAW_MAX_DEG_S = 200.0

# --------------------------------------------------------------------------
# Config helpers (mirror path_follow_flight.build_controller, minus Olympe env)
# --------------------------------------------------------------------------


def build_production_controller(
    route_path: Path, align_path: Path | None, *, return_to_start: bool
):
    """Build the exact controller production flies: measured MapFrame +
    route-authored radii + out-and-back expansion. No poles here (patrol-only
    routes); pass poles explicitly if the route gains inspection points."""
    map_frame = rpf.load_map_frame(align_path) if align_path else rpf.LEGACY_MAP_FRAME
    doc = RouteDocument.from_path(route_path, require_map_units=True, map_frame=map_frame)
    base = rpf.production_auto_control_config(map_frame)
    if not return_to_start:
        base = replace(base, return_to_start=False)
    # DesktopRouteAutonomy widens authored arrival spheres the same way.
    cfg = rpf.apply_desktop_auto_authority(rpf.config_for_route(route_path, base))
    ctrl = rpf.RouteAutoController(doc.controller_waypoints(), poles=[], config=cfg)
    return ctrl, ctrl.cfg, map_frame, doc


# --------------------------------------------------------------------------
# Plant: true drone state in meters, world axes parallel to map east/north/up
# --------------------------------------------------------------------------


@dataclass
class DroneState:
    pos_m: np.ndarray  # (3,) meters, world frame parallel to map basis
    yaw: float  # map-heading convention, radians
    vel_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rate: float = 0.0  # rad/s, persistent first-order yaw response

    def copy(self) -> "DroneState":
        d = DroneState(self.pos_m.copy(), float(self.yaw), self.vel_m.copy())
        d.yaw_rate = float(self.yaw_rate)
        return d


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass
class SimParams:
    route: Path
    align: Path | None
    meters_per_unit: float = 10.0
    return_to_start: bool = True
    yaw_sign: int = 1
    seed: int = 7
    duration_s: float = 600.0
    localization_wait_s: float = 0.0  # Included in the total mission budget.
    # takeoff
    start_offset_u: tuple[float, float, float] = (0.25, 0.0, 0.0)
    start_offset_frame: str = "lateral"  # lateral | raw
    initial_yaw_error_deg: float = 30.0
    # misalignment residue ("沒對齊")
    yaw_bias_deg: float = 0.0
    pos_bias_u: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_error_pct: float = 0.0
    # localization defects
    pos_noise_u: float = 0.004
    yaw_noise_deg: float = 1.0  # visual yaw anchor noise (smoothed by fusion)
    oly_noise_deg: float = 0.1  # Olympe/IMU yaw telemetry noise (fast, smooth)
    latency_ms: float = 120.0
    drop_rate: float = 0.02
    outage_every_s: float = 0.0
    outage_dur_s: float = 0.0
    # Sudden pose jumps (relocalization snaps): teleport up to jump_max_u for one tick.
    jump_rate: float = 0.0
    jump_max_u: float = 0.0
    # Outlier frames: uniform noise over +-outlier_max_u (mismatch proxy).
    outlier_rate: float = 0.0
    outlier_max_u: float = 0.0
    # WEAK_TRACK: low-confidence but reported; noise x weak_noise_gain that tick.
    weak_rate: float = 0.0
    weak_noise_gain: float = 3.0
    # PREDICTED_ONLY: dead-reckoning extrapolation instead of a fresh fix.
    predicted_rate: float = 0.0
    # Total visual blackout: no fix at all, IMU-only estimate from the last
    # visual anchor (production IMU_BRIDGE law: hold center, rotate by fused
    # yaw, random-walk drift). Periodic: every blackout_every_s for
    # blackout_dur_s. Defaults off.
    blackout_every_s: float = 0.0
    blackout_dur_s: float = 0.0
    blackout_drift_mps: float = 0.0
    blackout_yaw_drift_deg_s: float = 0.0
    # Localization-failure windows: the aircraft does not know where it is,
    # but VO / dead reckoning keep reporting a weak estimate that drifts
    # (drift_gain x true motion + drift_mps). See SimulatedLocalizerConfig.
    drift_every_s: float = 0.0
    drift_dur_s: float = 0.0
    drift_offset_s: float = 0.0
    drift_gain: float = 1.0
    drift_mps: float = 0.0
    # Reported height change = vertical_gain x true height change.
    vertical_gain: float = 1.0
    # Firmware altitude fed to the barometric height cross-check: truth plus
    # white noise and a constant drift (m/s since the start).
    baro_noise_m: float = 0.0
    baro_drift_mps: float = 0.0
    # plant
    cruise_mps: float = 0.90  # horizontal speed at max_translation_pcmd
    max_vertical_speed_mps: float = 2.0  # configured desktop envelope, not capability maximum
    max_rotation_speed_deg_s: float = 20.0
    speed_limit_mps: float = 0.6
    tau_h_s: float = 0.4
    tau_v_s: float = 0.3
    tau_yaw_s: float = 0.2
    wind_sigma_mps: float = 0.05
    wind_tau_s: float = 3.0
    # Firmware hover stabilization: a real ANAFI holds position (optical
    # flow / GPS / barometer) when PCMD translation is zero; zero PCMD is NOT
    # open-loop drift. Emulated as a capped spring to the hover-entry anchor.
    # 0 disables it (pure drift, pessimistic).
    hover_hold_mps: float = 0.2
    hover_hold_gain: float = 0.5
    # --- plant fidelity (all default to previous behavior) ---
    # Tilt follows PCMD with its own lag; horizontal accel = tilt * g.
    tau_tilt_s: float = 1e-3
    thrust_margin: float = 99.0
    # Battery sag: linear thrust loss over mission; 0 disables.
    battery_sag_frac: float = 0.0
    # Ground effect: lift cushion near takeoff altitude; 0 disables.
    ground_effect_frac: float = 0.0
    ground_effect_height_m: float = 1.0
    # Yaw/translation coupling: yaw rate bleeds tilt authority; 0 disables.
    yaw_coupling: float = 0.0
    gust_every_s: float = 0.0
    gust_m: float = 0.0
    ground_altitude_m: float | None = None
    telemetry_rate_hz: float = 5.0
    telemetry_latency_ms: float = 200.0
    telemetry_noise_mps: float = 0.0
    telemetry_drop_rate: float = 0.0
    wind_mean_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose_rate_hz: float = 20.0
    latency_jitter_ms: float = 0.0
    position_correlation_s: float = 0.0
    yaw_correlation_s: float = 0.0
    bias_walk_u_sqrt_s: float = 0.0
    reseed_confirm_frames: int = 0
    command_latency_ms: float = 0.0
    command_drop_rate: float = 0.0
    command_ttl_s: float = 0.2
    wind_model: str = "residual_ground_drift_proxy"
    def __post_init__(self):
        sag = float(self.battery_sag_frac)
        if not __import__("math").isfinite(sag) or not 0.0 <= sag < 1.0:
            raise ValueError("battery_sag_frac must be finite and in [0, 1)")
        if not __import__("math").isfinite(float(self.duration_s)) or float(self.duration_s) <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not __import__("math").isfinite(float(self.thrust_margin)) or float(self.thrust_margin) < 1.0:
            raise ValueError("thrust_margin must be finite and >= 1")
        yaw_c = float(self.yaw_coupling)
        if not __import__("math").isfinite(yaw_c) or not 0.0 <= yaw_c <= 1.0:
            raise ValueError("yaw_coupling must be finite and in [0, 1]")
        if (sag != 0.0 or yaw_c != 0.0 or float(self.thrust_margin) != 99.0) and float(self.tau_tilt_s) <= 1e-3 + 1e-9:
            raise ValueError("tilt dynamics require tau_tilt_s > 0.001")
        for name in ("telemetry_rate_hz", "pose_rate_hz"):
            rate = float(getattr(self, name))
            if not __import__("math").isfinite(rate) or rate <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("telemetry_latency_ms", "telemetry_noise_mps", "latency_jitter_ms",
                     "position_correlation_s", "yaw_correlation_s", "bias_walk_u_sqrt_s",
                     "command_latency_ms", "baro_noise_m"):
            value = float(getattr(self, name))
            if not __import__("math").isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not __import__("math").isfinite(float(self.baro_drift_mps)):
            raise ValueError("baro_drift_mps must be finite")
        drop = float(self.telemetry_drop_rate)
        if not __import__("math").isfinite(drop) or not 0.0 <= drop <= 1.0:
            raise ValueError("telemetry_drop_rate must be finite and in [0, 1]")
        command_drop = float(self.command_drop_rate)
        if not __import__("math").isfinite(command_drop) or not 0.0 <= command_drop <= 1.0:
            raise ValueError("command_drop_rate must be finite and in [0, 1]")
        if not __import__("math").isfinite(float(self.command_ttl_s)) or float(self.command_ttl_s) <= 0.0:
            raise ValueError("command_ttl_s must be finite and > 0")
        frames = self.reseed_confirm_frames
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise ValueError("reseed_confirm_frames must be a non-negative int")
        if str(self.wind_model) != "residual_ground_drift_proxy":
            raise ValueError("wind_model must be residual_ground_drift_proxy")
        if float(self.ground_effect_frac) > 0:
            ground = self.ground_altitude_m
            if ground is None or not __import__("math").isfinite(float(ground)):
                raise ValueError("ground_altitude_m must be finite when ground_effect_frac > 0")


# --------------------------------------------------------------------------
# Localization-defect regression scenarios (recorded real-flight regimes).
#
# R1_flight20260918 replays the 2026-09-18 real flight that stalled at
# waypoint 1: ~23 s of NO_POSE after takeoff, then repeating ~24 s stretches
# with zero strong (FAST_TRACK/RELOC_SEED) fixes while KLT/IMU weak poses keep
# arriving. R2_weak80 is the milder flicker regime: mostly weak poses with
# enough strong fixes interleaved that recovery can still confirm.
#
# R3/R4 replay the two failure modes of the 2026-09-18 13:01 flight on the
# route it flew (start offsets are map x/y/z from its waypoint 1; tuned at
# 15 m/u). R3 (run 1): takeoff 0.68 u from waypoint 1, 0.3 u below it; ~2 s
# in the aircraft stops knowing where it is for 24 s while VO / dead reckon
# keep reporting ~5x its real motion plus 0.5 m/s drift. R4 (run 2): the
# localized height never follows the climb, so waypoint 1 stays above; once
# over it (40 s here) the position drifts again and a >0.5 s gap forces a
# re-alignment toward a bearing that is only noise at that range.
# --------------------------------------------------------------------------
_FLIGHT_20260918_1301_DELIVERY = {"pose_rate_hz": 16.0, "latency_jitter_ms": 60.0, "drop_rate": 0.08}
REGRESSION_SCENARIOS: dict[str, dict] = {
    "R1_flight20260918": {
        "localization_wait_s": 23.0,
        "outage_every_s": 40.0,
        "outage_dur_s": 24.0,
        "start_offset_u": (0.68, 0.0, 0.0),
    },
    "R2_weak80": {
        "start_offset_u": (0.68, 0.0, 0.0),
        "weak_rate": 0.8,
    },
    "R3_flight20260918_1301_drift": {
        **_FLIGHT_20260918_1301_DELIVERY,
        "start_offset_frame": "raw",
        "start_offset_u": (-0.0463, 0.2957, -0.6162),
        "drift_offset_s": 2.0,
        "drift_every_s": 40.0,
        "drift_dur_s": 24.0,
        "drift_gain": 5.0,
        "drift_mps": 0.5,
    },
    "R4_flight20260918_1301_height": {
        **_FLIGHT_20260918_1301_DELIVERY,
        "start_offset_frame": "raw",
        "start_offset_u": (-0.0654, 0.2154, -0.3268),
        "vertical_gain": 0.0,
        "drift_offset_s": 40.0,
        "drift_every_s": 60.0,
        "drift_dur_s": 14.0,
        "drift_gain": 5.0,
        "drift_mps": 0.5,
        "outage_every_s": 45.0,
        "outage_dur_s": 1.0,
    },
}


def apply_regression_scenario(params: SimParams, name: str) -> SimParams:
    """Return params with a REGRESSION_SCENARIOS preset applied. Unknown names raise KeyError."""
    try:
        overrides = REGRESSION_SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown regression scenario: {name!r} (have: {sorted(REGRESSION_SCENARIOS)})") from None
    return replace(params, **overrides)


@dataclass
class SimResult:
    success: bool
    reason: str
    time_s: float
    steps: int
    progress_pct: float
    dist_flown_m: float
    dist_flown_u: float
    max_dev_u: float
    max_dev_m: float
    hover_ticks: int
    yaw_align_ticks: int
    waypoints_reached: list[int]
    speed_guard_interventions: int = 0
    max_waypoint_no_progress_s: float = 0.0
    yaw_overlap_ticks: int = 0
    # Truth-side metrics (the estimate-side max_dev cannot see drift).
    max_true_dev_m: float = 0.0
    max_est_error_m: float = 0.0
    max_climb_m: float = 0.0
    # Longest run of route commands issued on a pose that is not map-confirmed.
    weak_steer_max_s: float = 0.0
    # Join-leg yaw-alignment ticks while the target is horizontally inside
    # minimum_yaw_alignment_distance. That leg aims at goal - pose, so its
    # bearing is ill-conditioned there; later legs aim along the segment.
    near_waypoint_turn_ticks: int = 0
    # Waypoints whose localized height the barometric cross-check waived.
    vertical_waivers: int = 0
    trace: list[dict] = field(default_factory=list)

class _SimulationAutonomyChecks(DesktopRouteAutonomy):
    """Run production speed/stall checks without a flight worker or transport."""

    def __init__(self, limit: float, controller):
        import queue as _queue
        import threading as _threading
        import time as _time
        self.events: _queue.Queue = _queue.Queue()
        self._cancel = _threading.Event()
        self._paused = _threading.Event()
        self._cancel_reason = ""
        self._manual_handoff_attempted = False
        self._manual_handoff_lock = _threading.Lock()
        self._manual_handoff_thread = None
        self._thread = None
        self._watchdog_thread = None
        self._watchdog_stop = _threading.Event()
        self._heartbeat_lock = _threading.Lock()
        self._heartbeat_mono = _time.monotonic()
        self._landing_confirmed = None
        self._pcmd_failure_streak = 0
        self._auto_failed = False
        self._auto_failure_detail = ""
        self._failure_lock = _threading.Lock()
        self._command_epoch = 0
        self._mission_started = None
        self._route_progress = _fresh_route_progress()
        self.backend = SimpleNamespace(
            state=SimpleNamespace(
                autonomous_speed_limit_enabled=True,
                autonomous_speed_limit_mps=limit,
                att_yaw=0.0,
                attitude_mono_ns=0,
                speed_north_mps=0.0,
                speed_east_mps=0.0,
                ground_speed_mps=0.0,
                ground_speed_mono_ns=0,
            )
        )
        self.stamp = 0.0
        self.now = lambda: self.stamp
        self._speed_guard_latched = False
        self._speed_command_scale = 1.0
        self._speed_sample = None
        self._speed_rate_mps2 = 0.0
        self.controller = controller
        self.phase = "ROUTE"
        self._waypoint_progress = None
        self.failure_reason = None

    def _clear_motion(self):
        pass

    def _send_zero(self, reason):
        return True

    def _dispatch_route_pcmd(self, pcmd):
        limited = tuple(int(v) for v in pcmd)
        self._pcmd_failure_streak = 0
        return True, "sim command channel", limited

    def _latch_auto_failure(self, detail, *, error=None, zero_reason="auto_failure_hover"):
        super()._latch_auto_failure(detail, error=error, zero_reason=zero_reason)
        self.failure_reason = self._auto_failure_detail

    def _session_logs(self):
        return None

    def stream_healthy(self):
        return True

    def _olympe_yaw(self):
        # Simulation feeds yaw explicitly per tick; bypass the wall-clock
        # attitude-age gate used on the live link.
        try:
            yaw = float(self.backend.state.att_yaw)
        except (TypeError, ValueError, OverflowError):
            return None
        return yaw if math.isfinite(yaw) else None

    def apply(self, pcmd, stamp: float):
        self.stamp = stamp
        return self._apply_speed_limit(pcmd)


def _body_axes(yaw: float, map_frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = math.cos(yaw) * map_frame.east + math.sin(yaw) * map_frame.north
    right = math.sin(yaw) * map_frame.east - math.cos(yaw) * map_frame.north
    return (np.asarray(fwd, float), np.asarray(right, float), np.asarray(map_frame.up, float))


def adds_horizontal_speed(roll: int, pitch: int, body_velocity) -> bool:
    """True when horizontal PCMD pushes along the current motion.

    The controller may hold horizontal PCMD against measured velocity while it
    yaws or changes height (wind hold); that only brakes. Translating while
    yawing or climbing is what the two-phase law forbids.
    """
    forward_v, right_v = body_velocity
    return pitch * forward_v + roll * right_v > 0.0


def _integrate_plant(
    state,
    wind,
    hover_anchor,
    params,
    map_frame,
    rng,
    pcmd,
    vh_per_pct,
    vv_per_pct,
    yaw_per_pct,
    gust_phase,
    *,
    elapsed_s: float,
):
    """Advance the physical plant by one control period, including wind.

    Fidelity model: PCMD roll/pitch are tilt percentages. Commanded tilt
    (rad) = pct/100 * MaxTilt; achieved tilt lags with tau_tilt_s and slews
    at the firmware pitch/roll rate; horizontal accel = achieved tilt * g *
    yaw-coupling (tilt approximation, not a 200 Hz firmware PID replica).
    Total thrust (hover + tilt + climb) saturates at thrust_margin; battery
    sag scales it down over the mission; ground effect adds a lift cushion
    near takeoff altitude. The horizontal velocity loop is a speed
    approximation keyed to cruise_mps, not a calibrated airframe model.
    """
    roll, pitch, yaw_p, gaz = pcmd
    sub = 10
    sdt = DT / sub
    sim_now = float(elapsed_s)
    wind_mean = np.asarray(params.wind_mean_mps, dtype=float).reshape(3)
    wind_mean_world = (float(wind_mean[0]) * np.asarray(map_frame.east, dtype=float)
                       + float(wind_mean[1]) * np.asarray(map_frame.north, dtype=float)
                       + float(wind_mean[2]) * np.asarray(map_frame.up, dtype=float))
    try:
        from live_safety_config import (
            DEFAULT_MAX_PITCH_ROLL_ROTATION_SPEED_DEGS as _MAX_TILT_RATE,
            DEFAULT_MAX_TILT_DEG as _MAX_TILT,
        )
        max_tilt_deg = float(_MAX_TILT)
        max_tilt_rate_deg_s = float(_MAX_TILT_RATE)
    except (ImportError, AttributeError, TypeError, ValueError):
        max_tilt_deg, max_tilt_rate_deg_s = 20.0, 60.0
    max_tilt_rad = math.radians(max_tilt_deg)
    max_tilt_rate = math.radians(max_tilt_rate_deg_s)
    g0 = 9.81
    tilt_cmd_fwd = pitch / 100.0 * max_tilt_rad
    tilt_cmd_right = roll / 100.0 * max_tilt_rad
    k_tilt = 1.0 - math.exp(-sdt / max(float(params.tau_tilt_s), 1e-6))
    fwd_w, right_w, up_w = _body_axes(state.yaw, map_frame)
    v_tgt = fwd_w * (pitch * vh_per_pct) + right_w * (roll * vh_per_pct) + up_w * (gaz * vv_per_pct)
    yaw_rate_tgt = -math.radians(yaw_p * yaw_per_pct)
    if hover_anchor is not None and params.hover_hold_mps > 0:
        hold = (hover_anchor - state.pos_m) * params.hover_hold_gain
        hn = float(np.linalg.norm(hold))
        if hn > params.hover_hold_mps:
            hold = hold / hn * params.hover_hold_mps
        # Vertical part only: the horizontal hover correction is added once
        # per substep after the speed/tilt mode is chosen (see hover_hold
        # below). Adding the full vector here would apply it twice.
        v_tgt = v_tgt + np.dot(hold, up_w) * up_w
    vh_cap = v_tgt - np.dot(v_tgt, up_w) * up_w  # horizontal part
    if float(np.linalg.norm(vh_cap)) > VH_MAX_MPS:
        v_tgt = vh_cap / float(np.linalg.norm(vh_cap)) * VH_MAX_MPS + np.dot(v_tgt, up_w) * up_w
    vv_n = float(np.dot(v_tgt, up_w))
    if abs(vv_n) > VV_MAX_MPS:
        v_tgt = v_tgt - up_w * (vv_n - math.copysign(VV_MAX_MPS, vv_n))
    k_h = 1.0 - math.exp(-sdt / params.tau_h_s)
    k_v = 1.0 - math.exp(-sdt / params.tau_v_s)
    k_y = 1.0 - math.exp(-sdt / params.tau_yaw_s)
    tilt_fwd = float(getattr(state, "tilt_fwd", 0.0))
    tilt_right = float(getattr(state, "tilt_right", 0.0))
    gust_displacement = np.zeros(3)
    for _ in range(sub):
        # wind OU drift around the configured mean (ground-drift proxy)
        wind += (
            (
                (wind_mean_world - wind) / params.wind_tau_s * sdt
                + math.sqrt(2.0 / params.wind_tau_s)
                * params.wind_sigma_mps
                * math.sqrt(max(sdt, 1e-9))
                * rng.normal(0, 1, 3)
            )
            if params.wind_tau_s > 0
            else 0.0
        )
        gust = np.zeros(3)
        if params.gust_every_s > 0 and params.gust_m > 0:
            gust_phase += sdt
            if gust_phase + 1e-9 >= params.gust_every_s:
                gust_phase = 0.0
                gdir = rng.normal(0, 1, 3)
                gdir /= np.linalg.norm(gdir) or 1.0
                gust = gdir * (params.gust_m / sdt)  # exact metre displacement per impulse
                gust_displacement += gust * sdt
        # achieved tilt lags command (first-order) and slews at the firmware
        # pitch/roll rate limit; horizontal accel = achieved tilt * g.
        tilt_fwd += (tilt_cmd_fwd - tilt_fwd) * k_tilt
        tilt_right += (tilt_cmd_right - tilt_right) * k_tilt
        step_limit = max_tilt_rate * sdt
        tilt_fwd = max(tilt_cmd_fwd - step_limit, min(tilt_cmd_fwd + step_limit, tilt_fwd))
        tilt_right = max(tilt_cmd_right - step_limit, min(tilt_cmd_right + step_limit, tilt_right))
        tilt_mag = math.hypot(float(tilt_fwd), float(tilt_right))
        if tilt_mag > max_tilt_rad and tilt_mag > 0.0:
            tilt_fwd *= max_tilt_rad / tilt_mag
            tilt_right *= max_tilt_rad / tilt_mag
        fwd_w, right_w, up_w = _body_axes(state.yaw, map_frame)
        couple = 1.0 - float(params.yaw_coupling) * abs(float(state.yaw_rate)) / math.radians(200.0)
        couple = max(0.0, min(1.0, couple))
        a_h = (fwd_w * tilt_fwd + right_w * tilt_right) * g0 * couple
        # thrust saturation + sag: tilt + climb share the margin over hover
        climb_demand = abs(float(gaz * vv_per_pct)) / max(VV_MAX_MPS, 1e-6)
        tilt_demand = float(np.linalg.norm(a_h)) / g0
        sag_now = float(params.battery_sag_frac) * min(1.0, max(0.0, (sim_now + sdt) / max(float(params.duration_s), 1e-6)))
        avail = max(0.0, float(params.thrust_margin) * (1.0 - sag_now) - 1.0)
        if tilt_demand + climb_demand > avail and (tilt_demand + climb_demand) > 0:
            a_h = a_h * max(0.0, avail / (tilt_demand + climb_demand))
        # ground effect: lift cushion within ground_effect_height_m above the
        # takeoff altitude (vertical only, no horizontal effect). Scaled to
        # 10% of the vertical envelope so frac=1.0 ≈ 0.2 m/s, not a takeoff.
        vv_lift = 0.0
        if float(params.ground_effect_frac) > 0 and params.ground_altitude_m is not None:
            hagl = float(np.dot(state.pos_m, up_w)) - float(params.ground_altitude_m)
            if 0.0 <= hagl <= float(params.ground_effect_height_m):
                vv_lift = float(params.ground_effect_frac) * (
                    1.0 - hagl / max(float(params.ground_effect_height_m), 1e-6)
                ) * 0.1 * float(params.max_vertical_speed_mps)
        hover_hold = np.zeros(3)
        if hover_anchor is not None and params.hover_hold_mps > 0:
            hold = (hover_anchor - state.pos_m) * params.hover_hold_gain
            hn = float(np.linalg.norm(hold))
            if hn > params.hover_hold_mps:
                hold = hold / hn * params.hover_hold_mps
            hover_hold = hold - np.dot(hold, up_w) * up_w
        vh = v_tgt - np.dot(v_tgt, up_w) * up_w  # horizontal part (legacy velocity loop)
        vv = np.dot(v_tgt, up_w) * up_w
        # tilt path: with tau_tilt_s at default (~0) this is an exact no-op and
        # the legacy velocity loop above runs unchanged; with tau_tilt_s set,
        # the horizontal target comes from achieved tilt (tilt * g * tau_h).
        if float(params.tau_tilt_s) > 1e-3 + 1e-9:
            vh = a_h * float(params.tau_h_s)
        vh = vh + hover_hold
        vv = vv + up_w * vv_lift
        cur_h = state.vel_m - np.dot(state.vel_m, up_w) * up_w
        cur_v = np.dot(state.vel_m, up_w) * up_w
        cur_h += (vh - cur_h) * k_h
        cur_v += (vv - cur_v) * k_v
        state.vel_m = cur_h + cur_v
        state.pos_m += (state.vel_m + wind) * sdt + gust * sdt
        state.yaw_rate += (yaw_rate_tgt - state.yaw_rate) * k_y
        state.yaw = rpf.wrap_angle(state.yaw + state.yaw_rate * sdt)
    state.tilt_fwd = float(tilt_fwd)
    state.tilt_right = float(tilt_right)
    return wind, gust_phase, gust_displacement


def _inspection_dwell(ctrl, cmd, pose, cfg, t, land_hold_since):
    phase = "inspect_hold"
    if pose is not None and cmd.look_at_pole is not None:
        err = abs(math.degrees(rpf.wrap_angle(float(cmd.yaw_target) - float(pose.yaw))))
        if err <= float(cfg.inspect_yaw_tolerance_deg):
            if land_hold_since is None:
                land_hold_since = t
            if t - land_hold_since >= 1.0:
                ctrl.ack_inspection(cmd.look_at_pole, orientation_confirmed=True)
                land_hold_since = None
                phase = "inspect_acked"
        else:
            land_hold_since = None
    return land_hold_since, phase


def _formal_terminal_success(reason, ctrl, n_wp):
    """Map the formal loop terminal reason onto the offline result contract."""
    text = str(reason or "")
    if "route complete -> land" in text:
        if int(ctrl.target_index) == int(n_wp) - 1:
            return True, text
        return False, text
    return False, text


def _formal_landing_settled(runner_record):
    try:
        samples = int(runner_record.get("landing_stable_samples", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    try:
        span = float(runner_record.get("landing_stable_span_s", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return False
    try:
        speed = float(runner_record.get("landing_speed_mps"))
    except (TypeError, ValueError, OverflowError):
        return False
    return samples >= 3 and span >= 1.0 and math.isfinite(speed) and speed <= 0.10


def _formal_pose_from_record(runner_record, fallback):
    """Pose the formal loop actually steered from, not the sensor-cell value."""
    try:
        pose_u = runner_record.get("pose_u")
    except AttributeError:
        return fallback
    if pose_u is None:
        return fallback
    try:
        xyz = [float(v) for v in pose_u]
    except (TypeError, ValueError, OverflowError):
        return fallback
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        return fallback
    if fallback is None:
        return fallback
    try:
        yaw = float(runner_record.get("pose_yaw", fallback.yaw))
    except (TypeError, ValueError, OverflowError):
        yaw = float(fallback.yaw)
    if not math.isfinite(yaw):
        yaw = float(fallback.yaw)
    try:
        return replace(
            fallback,
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
            yaw=float(yaw),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return fallback

def _inspection_command_from_record(runner_record, pose):
    import numpy as _np
    goal = _np.asarray(runner_record.get("target_u", [0.0, 0.0, 0.0]), dtype=float)
    yaw_target = float(runner_record.get("yaw_target_deg", 0.0) or 0.0)
    return rpf.Command(
        "INSPECT",
        _np.zeros(3),
        _np.radians(yaw_target),
        goal,
        float(runner_record.get("path_error_u", 0.0) or 0.0),
        float(runner_record.get("progress", 0.0) or 0.0),
        look_at_pole={"waypoint": int(runner_record.get("target_index", 0) or 0) + 1},
        should_land=False,
        status=str(runner_record.get("action") or "INSPECT"),
    )


_TURN_PHASES = ("turn", "yaw_alignment_hold", "yaw_alignment_confirmed", "yaw_alignment_timeout")


def _phase_ticks(phase):
    text = str(phase or "")
    base = text.removeprefix("fgc:")
    if base in (*_TURN_PHASES, "final_centering"):
        return 0, 1
    if base in ("translate", "route_rejoin", "waypoint_centering", "height_adjust"):
        return 0, 0
    return 1, 0


def _update_hover_anchor(state, hover_anchor, map_frame, roll, pitch, gaz):
    if hover_anchor is None:
        hover_anchor = state.pos_m.copy()
    anchor_delta = state.pos_m - hover_anchor
    anchor_vertical = np.dot(anchor_delta, map_frame.up) * map_frame.up
    if roll != 0 or pitch != 0:
        hover_anchor += anchor_delta - anchor_vertical
    if gaz != 0:
        hover_anchor += anchor_vertical

    return hover_anchor


def _retired_target_from_record(runner_record, prev_target):
    try:
        current = int(runner_record.get("target_index", prev_target))
    except (TypeError, ValueError, OverflowError):
        return None
    if int(current) != int(prev_target):
        return int(prev_target)
    return None


def _initial_sim_state(params, wp, map_frame, s):
    if params.start_offset_frame == "lateral":
        # offset perpendicular to the first leg's horizontal direction: a real
        # cross-track start, not a slide along the route.
        leg = wp[1] - wp[0]
        e, n = map_frame.horizontal(leg)
        norm = math.hypot(e, n) or 1.0
        perp_map = (-n / norm) * map_frame.east + (e / norm) * map_frame.north
        up_map = np.asarray(map_frame.up, float)
        off_map = (
            float(params.start_offset_u[0]) * perp_map
            + float(params.start_offset_u[1]) * (leg / (float(np.linalg.norm(leg)) or 1.0))
            + float(params.start_offset_u[2]) * up_map
        )
    else:
        off_map = np.array(params.start_offset_u, float)
    start_map = wp[0] + off_map
    state = DroneState(
        pos_m=start_map * s,
        yaw=float(map_frame.heading(wp[1] - wp[0]) + math.radians(params.initial_yaw_error_deg)),
    )

    return state, start_map


class _TruthDebrief:
    """Truth-side metrics. run_sim's max_dev is measured on the estimate, which
    a drifting weak pose can keep on the route while the aircraft is not."""

    def __init__(self, ctrl, cfg, map_frame, s, start_pos_m, join_target):
        self.ctrl, self.map_frame, self.s = ctrl, map_frame, float(s)
        self.join_target = int(join_target)
        self.min_yaw_distance_u = float(cfg.minimum_yaw_alignment_distance)
        self.up = np.asarray(map_frame.up, dtype=float) / float(np.linalg.norm(map_frame.up))
        self.start_height_m = float(np.dot(start_pos_m, self.up))
        self.max_true_dev_u = self.max_est_error_u = self.max_climb_m = 0.0
        self.weak_steer_s = self.weak_steer_max_s = 0.0
        self.last_pose_weak = False
        self.near_waypoint_turn_ticks = 0
        self.vertical_waivers = 0

    def update(self, pos_m, pose, live_pose, record, phase) -> None:
        true_u = np.asarray(pos_m, dtype=float) / self.s
        dist, _nearest, _segment, _s = rpf.project_to_path(true_u, list(self.ctrl.wp), self.ctrl.cum)
        self.max_true_dev_u = max(self.max_true_dev_u, float(dist))
        if pose is not None:
            self.max_est_error_u = max(self.max_est_error_u, float(np.linalg.norm(pose.xyz - true_u)))
        self.max_climb_m = max(self.max_climb_m, float(np.dot(pos_m, self.up)) - self.start_height_m)
        # Longest stretch the route controller keeps flying (no localization
        # hold) while the pose it is given is not map-confirmed.
        if live_pose is not None:  # None = no delivery this tick (e.g. out-of-order capture)
            self.last_pose_weak = not bool(getattr(live_pose, "map_confirmed", True))
        if "command_action" in record and self.last_pose_weak:
            self.weak_steer_s += DT
            self.weak_steer_max_s = max(self.weak_steer_max_s, self.weak_steer_s)
        else:
            self.weak_steer_s = 0.0
        if record.get("vertical_waived_now"):
            self.vertical_waivers += 1
        error = record.get("target_error_map_u")
        if (
            error is not None
            and record.get("target_index") == self.join_target
            and str(phase).removeprefix("fgc:") in _TURN_PHASES
            and self.map_frame.horizontal_distance(np.asarray(error, dtype=float)) < self.min_yaw_distance_u
        ):
            self.near_waypoint_turn_ticks += 1


def _barometer(params: SimParams, rng, up_axis):
    """Firmware altitude reader for the height cross-check: truth plus configured error."""
    noise, drift = float(params.baro_noise_m), float(params.baro_drift_mps)

    def read(pos_m, now) -> float:
        altitude = float(np.dot(pos_m, up_axis)) + drift * float(now)
        if noise > 0.0:
            altitude += float(rng.normal(0.0, noise))
        return altitude

    return read


def _simulation_budget(params: SimParams) -> tuple[float, int]:
    duration, wait = float(params.duration_s), float(params.localization_wait_s)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not math.isfinite(wait) or wait < 0:
        raise ValueError("localization_wait_s must be finite and nonnegative")
    initial_wait = min(wait, duration)
    return initial_wait, int((duration - initial_wait) * CTRL_HZ)


def run_sim(params: SimParams, *, record_trace: bool = True) -> tuple[SimResult, dict]:
    seed_seq = np.random.SeedSequence(params.seed)
    rng = np.random.default_rng(seed_seq.spawn(1)[0])
    tele_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    loc_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    cmd_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    baro_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    ctrl, cfg, map_frame, doc = build_production_controller(
        params.route, params.align, return_to_start=params.return_to_start
    )
    s = float(params.meters_per_unit)
    if not math.isfinite(s) or s <= 0:
        raise ValueError("meters_per_unit must be finite and > 0")

    wp = [np.array(p, float) for p in ctrl.wp]
    n_wp = len(wp)

    # --- takeoff state (meters, world frame parallel to map basis) ---
    state, start_map = _initial_sim_state(params, wp, map_frame, s)

    # AUTO start rule: waypoint 1 first, even when takeoff sits nearer a later point.
    ctrl.start_after_nearest_waypoint(np.asarray(start_map, float))
    join_target = int(ctrl.target_index)
    from simulated_localization import CommandChannel, CommandChannelConfig, SimulatedLocalizer, SimulatedLocalizerConfig
    loc_cfg = SimulatedLocalizerConfig(
        s=s, yaw_bias_deg=params.yaw_bias_deg, pos_bias_u=tuple(params.pos_bias_u),
        scale_error_pct=params.scale_error_pct, pos_sigma_u=params.pos_noise_u,
        yaw_sigma_deg=params.yaw_noise_deg, latency_ms=params.latency_ms,
        drop_rate=params.drop_rate, outage_every_s=params.outage_every_s,
        outage_dur_s=params.outage_dur_s, jump_rate=params.jump_rate, jump_max_u=params.jump_max_u,
        outlier_rate=params.outlier_rate, outlier_max_u=params.outlier_max_u,
        weak_rate=params.weak_rate, weak_noise_gain=params.weak_noise_gain,
        predicted_rate=params.predicted_rate, blackout_every_s=params.blackout_every_s,
        blackout_dur_s=params.blackout_dur_s, blackout_drift_mps=params.blackout_drift_mps,
        blackout_yaw_drift_deg_s=params.blackout_yaw_drift_deg_s,
        pose_rate_hz=params.pose_rate_hz, latency_jitter_ms=params.latency_jitter_ms,
        position_correlation_s=params.position_correlation_s, yaw_correlation_s=params.yaw_correlation_s,
        bias_walk_u_sqrt_s=params.bias_walk_u_sqrt_s,
        reseed_confirm_frames=params.reseed_confirm_frames,
        drift_every_s=params.drift_every_s, drift_dur_s=params.drift_dur_s,
        drift_offset_s=params.drift_offset_s, drift_gain=params.drift_gain,
        drift_mps=params.drift_mps, vertical_gain=params.vertical_gain,
        map_up_u=tuple(float(v) for v in map_frame.up),
    )
    loc = SimulatedLocalizer(loc_cfg, loc_rng)
    cmd_channel = CommandChannel(CommandChannelConfig(
        latency_ms=params.command_latency_ms, drop_rate=params.command_drop_rate,
        ttl_s=params.command_ttl_s), cmd_rng)
    import path_follow_flight as pff

    # per-% speed calibration: max_translation_pcmd flies cruise_mps horizontally.
    vh_per_pct = params.cruise_mps / max(1, int(cfg.max_translation_pcmd))
    vv_per_pct = params.max_vertical_speed_mps / 100.0
    yaw_per_pct = params.max_rotation_speed_deg_s / 100.0
    initial_wait, max_steps = _simulation_budget(params)
    wind_mean = np.asarray(params.wind_mean_mps, dtype=float).reshape(3)
    if not np.all(np.isfinite(wind_mean)):
        raise ValueError("wind_mean_mps must be finite")
    for k in range(8):
        loc.push_truth(initial_wait - (8 - k) * DT, state.pos_m, state.yaw)
    loc.push_truth(initial_wait, state.pos_m, state.yaw)
    wind_mean_world = (float(wind_mean[0]) * np.asarray(map_frame.east, dtype=float)
                       + float(wind_mean[1]) * np.asarray(map_frame.north, dtype=float)
                       + float(wind_mean[2]) * np.asarray(map_frame.up, dtype=float))
    wind = np.array(wind_mean_world, dtype=float)
    # Sampled, delayed NED telemetry (independent RNG stream).
    tele_queue: list[tuple[float, float, float]] = []  # (capture_t, north, east)
    tele_next = initial_wait
    tele_last: tuple[float, float, float] | None = None  # (north, east, stamp)
    hover_anchor: np.ndarray | None = None

    reached: set[int] = set()
    hover_ticks = yaw_ticks = 0
    yaw_overlap_ticks = 0
    land_hold_since: float | None = None
    speed_guard = _SimulationAutonomyChecks(params.speed_limit_mps, ctrl)
    speed_guard_interventions = 0
    max_waypoint_no_progress_s = 0.0
    max_dev = 0.0
    debrief = _TruthDebrief(ctrl, cfg, map_frame, s, state.pos_m, join_target)
    barometer = _barometer(params, baro_rng, debrief.up)
    reason = "step cap" if max_steps else "localization wait exhausted total budget"
    success = False
    dist_m = 0.0
    t = initial_wait
    step = -1
    prev_pos = state.pos_m.copy()
    gust_phase = 0.0
    safety_mode = {"mode": "AUTO"}
    pose_cell: dict = {"pose": None}
    oly_cell: dict = {"yaw": 0.0}
    runner_cell: dict = {"runner": None}
    sim_clock = {"t": float(initial_wait)}

    def _safety_poll():
        return str(safety_mode["mode"])

    def _jump_pause(distance):
        safety_mode["mode"] = "HOVER"

    def _request_manual():
        safety_mode["mode"] = "MANUAL"
        return True

    def _wait_expired(wait_reason, elapsed_s):
        return speed_guard._check_wait_budget(wait_reason, elapsed_s)

    def _send_authorized(pcmd):
        runner_now = float(sim_clock["t"])
        speed_guard.stamp = runner_now
        authorized = speed_guard._send_authorized(tuple(int(v) for v in pcmd))
        if not authorized[0]:
            return False, str(authorized[1]), (0, 0, 0, 0)
        sent = tuple(int(v) for v in (authorized[2] if authorized[2] is not None else (0, 0, 0, 0)))
        cmd_channel.send(sent, t)
        return True, "sim authorized", sent
    logged: list = []
    sim_trace: list[dict] = []

    def _log_tick(record):
        logged.append(record)

    hooks = pff.LoopHooks(
        get_pose=lambda: pose_cell["pose"],
        olympe_yaw=lambda: float(oly_cell["yaw"]),
        send_pcmd=lambda r, p, y, g: cmd_channel.send((r, p, y, g), t),
        send_authorized_pcmd=_send_authorized,
        pose_is_weak=lambda: bool(pose_cell["pose"] is not None and not bool(getattr(pose_cell["pose"], "map_confirmed", True))),
        pose_is_predicted=lambda: bool(pose_cell["pose"] is not None and not bool(getattr(pose_cell["pose"], "position_observed", True))),
        pose_reseed_confirming=lambda: bool(pose_cell["pose"] is not None and bool(getattr(pose_cell["pose"], "reseed_confirming", False))),
        pose_confidence=lambda: 0 if pose_cell["pose"] is not None and not bool(getattr(pose_cell["pose"], "map_confirmed", True)) else 300,
        force_relocalize=lambda: loc.note_recovery(),
        request_manual=_request_manual,
        stick_active=lambda: False,
        safety_poll=_safety_poll,
        stream_healthy=lambda: True,
        stream_status=lambda: "sim",
        loop_beat=lambda: None,
        ground_speed=lambda: speed_guard._ground_speed(),
        body_velocity=lambda: speed_guard._body_velocity(),
        altitude=lambda: (barometer(state.pos_m, sim_clock["t"]), float(sim_clock["t"])),
        max_weak_pose_age_s=pff.POSE_STALE_S,
        log_tick=_log_tick,
        land_on_localization_loss=False,
        wait_expired=_wait_expired,
        now=lambda: float(sim_clock["t"]),
    )
    runner = pff._FlightLoopRunner(
        hooks, ctrl, [np.array(point, float) for point in ctrl.wp],
        yaw_sign=int(params.yaw_sign), verbose=False, enforce_weak_pose_gate=False,
    )
    runner_cell["runner"] = runner
    # Offline plant: simulated time advances explicitly via t += DT. Never
    # wall-clock sleep inside the formal tick or a full route costs minutes.
    runner.period = 0.0
    speed_guard._mission_started = initial_wait
    arrive_r = [float(ctrl._arrive_radius_for(i)) for i in range(n_wp)]

    def _olympe_report() -> float:
        # NED clockwise-from-North, inverse of HeadingEstimator._olympe_to_map_ccw
        oly = (math.pi / 2.0 - state.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return float(oly + math.radians(rng.normal(0.0, params.oly_noise_deg)))

    def _pump_telemetry(now: float) -> None:
        nonlocal tele_next, tele_last
        rate = float(params.telemetry_rate_hz)
        period = 1.0 / rate
        while tele_next <= now + 1e-12:
            capture = tele_next
            tele_next += period
            if tele_rng.random() < float(params.telemetry_drop_rate):
                continue
            true_ned_n = float(np.dot(state.vel_m + wind, np.asarray(map_frame.north, dtype=float)))
            true_ned_e = float(np.dot(state.vel_m + wind, np.asarray(map_frame.east, dtype=float)))
            noise = float(params.telemetry_noise_mps)
            if noise > 0:
                true_ned_n += float(tele_rng.normal(0.0, noise))
                true_ned_e += float(tele_rng.normal(0.0, noise))
            tele_queue.append((capture, true_ned_n, true_ned_e))
        latency = float(params.telemetry_latency_ms) / 1000.0
        while tele_queue and tele_queue[0][0] + latency <= now + 1e-12:
            capture, north, east = tele_queue.pop(0)
            tele_last = (north, east, capture)

    def _trace_body_velocity():
        body = speed_guard._body_velocity()
        if body is None:
            return None
        return (float(body[0]), float(body[1]), float(body[2]))

    def _publish_telemetry(now: float, oly_yaw_now: float) -> tuple[tuple[float, float] | None, float | None]:
        _pump_telemetry(now)
        speed_guard.stamp = now
        speed_guard.backend.state.att_yaw = float(oly_yaw_now)
        if tele_last is None:
            speed_guard.backend.state.ground_speed_mps = float("nan")
            speed_guard.backend.state.ground_speed_mono_ns = 0
            speed_guard.backend.state.speed_north_mps = float("nan")
            speed_guard.backend.state.speed_east_mps = float("nan")
            return None, None
        north, east, capture = tele_last
        speed = math.hypot(north, east)
        speed_guard.backend.state.speed_north_mps = north
        speed_guard.backend.state.speed_east_mps = east
        speed_guard.backend.state.ground_speed_mps = speed
        speed_guard.backend.state.ground_speed_mono_ns = int(capture * 1e9)
        body = speed_guard._body_velocity()
        if body is None:
            return None, None
        return (body[0], body[1]), body[2]

    for step in range(max_steps):
        oly_yaw = _olympe_report()
        sim_clock["t"] = float(t)
        oly_cell["yaw"] = float(oly_yaw)
        prev_target = int(ctrl.target_index)
        logged_before = len(logged)
        pose_cell["pose"] = loc.report(t)
        _publish_telemetry(t, oly_yaw)
        runner_outcome = runner.step()
        sim_clock["t"] = float(t)
        runner_record = logged[logged_before] if len(logged) > logged_before else None
        if runner_record is None:
            # wait_expired latches AUTO failure and returns a terminal outcome
            # without emitting a per-tick row (same as the live loop: nothing
            # is dispatched after the handoff). Surface the latched detail.
            reason = str(getattr(speed_guard, "failure_reason", None) or getattr(speed_guard, "_auto_failure_detail", None) or runner_outcome.reason or "formal flight loop did not log this tick")
            success = False
            break
        pose = _formal_pose_from_record(runner_record, pose_cell["pose"])
        if str(safety_mode["mode"]) == "MANUAL":
            reason = "manual handoff: autonomy sends nothing after pilot takeover"
            break
        if runner_outcome.reason is not None:
            success, reason = _formal_terminal_success(runner_outcome.reason, ctrl, n_wp)
            if "route complete -> land" in str(runner_outcome.reason or "") and success:
                reached.add(int(ctrl.target_index))
                if record_trace:
                    try:
                        dist_term, _, _, _ = rpf.project_to_path(
                            np.array([pose.x, pose.y, pose.z]), list(ctrl.wp), ctrl.cum
                        )
                    except (AttributeError, TypeError, ValueError, OverflowError):
                        dist_term = float("nan")
                    body_forward, body_right, _up = _body_axes(state.yaw, map_frame)
                    gv = state.vel_m + wind
                    tele_term = _trace_body_velocity()
                    sim_trace.append(
                        {
                            "t": round(t, 3),
                            "step": step,
                            "true_m": state.pos_m.tolist(),
                            "true_map_u": (state.pos_m / s).tolist(),
                            "est_map_u": pose.xyz.tolist() if pose is not None else [float("nan")] * 3,
                            "yaw_deg": round(math.degrees(state.yaw), 2),
                            "action": "LAND",
                            "phase": str(runner_record.get("pcmd_phase") or "final_centering"),
                            "target_idx": int(ctrl.target_index),
                            "progress": 1.0,
                            "path_err_u": float(runner_record.get("path_error_u", 0.0) or 0.0),
                            "dev_u": float(dist_term),
                            "ground_speed_mps": float(map_frame.horizontal_distance(gv)),
                            "body_velocity_mps": [float(np.dot(gv, body_forward)), float(np.dot(gv, body_right))],
                            "measured_body_velocity_mps": None if tele_term is None else [float(tele_term[0]), float(tele_term[1])],
                            "telemetry_stamp": None if tele_term is None else float(tele_term[2]),
                            "pcmd_requested": list(tuple(int(v) for v in runner_record.get("pcmd_requested", (0, 0, 0, 0)))),
                            "pcmd": list(tuple(int(v) for v in runner_record.get("pcmd_requested", (0, 0, 0, 0)))),
                            "pcmd_applied": list(tuple(int(v) for v in runner_record.get("pcmd_requested", (0, 0, 0, 0)))),
                            "speed_guard_status": str(getattr(speed_guard.backend.state, "autonomous_speed_guard_status", "")),
                            "pose_stamp": float(pose.stamp) if pose is not None else None,
                            "map_confirmed": bool(getattr(pose, "map_confirmed", True)) if pose is not None else False,
                            "retired_target_idx": int(ctrl.target_index),
                            "gust_displacement_m": [0.0, 0.0, 0.0],
                            "gust_time_s": None,
                            "loc_fault": loc.fault,
                            "landing_stable_samples": int(runner_record.get("landing_stable_samples", 0) or 0),
                            "landing_stable_span_s": round(float(runner_record.get("landing_stable_span_s", 0.0) or 0.0), 3),
                            "landing_speed_measured_mps": runner_record.get("landing_speed_mps"),
                            "landing_speed_reason": runner_record.get("landing_speed_reason"),
                        }
                    )
            break
        cmd_action = str(runner_record.get("command_action") or "")
        if cmd_action == "ABORT_PENDING_INSPECTION":
            reason = "ABORT: pending inspection at route end"
            break
        if "command_action" not in runner_record:
            # Formal safety hold (BOOT/recovery/jump/HOVER): apply the logged
            # zero and integrate the plant without a navigation decision.
            action_pcmd = tuple(int(v) for v in runner_record.get("pcmd", (0, 0, 0, 0)))
            phase = str(runner_record.get("localization_recovery_state") or runner_record.get("reason") or "safety_hold")
        else:
            # The controller owns segment recovery; the simulator measures its result.
            action_pcmd = tuple(int(v) for v in runner_record.get("pcmd_requested", (0, 0, 0, 0)))
            phase = str(runner_record.get("pcmd_phase") or "")
        tele_body = _trace_body_velocity()
        if pose is not None:
            dist, _nearest_pt, _seg, _s = rpf.project_to_path(
                np.array([pose.x, pose.y, pose.z]), list(ctrl.wp), ctrl.cum
            )
            if dist > max_dev:
                max_dev = dist
        else:
            dist = float("nan")
        debrief.update(state.pos_m, pose, pose_cell["pose"], runner_record, phase)

        if cmd_action == "INSPECT":
            cmd = _inspection_command_from_record(runner_record, pose)
            land_hold_since, phase = _inspection_dwell(ctrl, cmd, pose, cfg, t, land_hold_since)
            action_pcmd = tuple(int(v) for v in runner_record.get("pcmd_requested", (0, 0, 0, 0)))
            phase = str(runner_record.get("pcmd_phase") or phase)
        elif cmd_action == "LAND":
            # The formal loop only emits LAND after the landing confirmation
            # guard passes, and then it returns the terminal outcome on the
            # same tick; a bare LAND log without one is a stale snapshot, so
            # keep integrating instead of ending the offline run early.
            hover_inc, yaw_inc = _phase_ticks(phase)
            hover_ticks += hover_inc
            yaw_ticks += yaw_inc
        else:
            hover_inc, yaw_inc = _phase_ticks(phase)
            hover_ticks += hover_inc
            yaw_ticks += yaw_inc

        max_waypoint_no_progress_s = max(
            max_waypoint_no_progress_s, float(runner_record.get("waypoint_no_progress_s", 0.0) or 0.0)
        )
        if speed_guard.failure_reason is not None:
            reason = speed_guard.failure_reason
            break
        if speed_guard._auto_failed:
            reason = str(speed_guard._auto_failure_detail)
            break
        applied_now, _age_s, _seq = cmd_channel.applied(t)
        roll, pitch, yaw_p, gaz = (int(v) for v in applied_now)
        requested = tuple(int(v) for v in runner_record.get("pcmd_requested", (0, 0, 0, 0)))
        if (roll, pitch, yaw_p, gaz) != requested and any(v != 0 for v in requested):
            speed_guard_interventions += 1
        body_forward, body_right, _body_up = _body_axes(state.yaw, map_frame)
        ground_velocity = state.vel_m + wind
        body_velocity = (
            float(np.dot(ground_velocity, body_forward)),
            float(np.dot(ground_velocity, body_right)),
        )
        # Production removed the turn gate (SAFETY.md: no yaw+horizontal FAIL);
        # the estimator can disagree with truth for a tick and a wind-hold can
        # read as along-track on the true velocity. Count it for the debrief,
        # never fail the run on it.
        hover_anchor = _update_hover_anchor(state, hover_anchor, map_frame, roll, pitch, gaz)

        # Record-before-integrate: one row = time t, pre-control state, the
        # formal loop decision, and the command actually applied. Gust
        # displacement is backfilled after integration with its own time base.
        rec: dict | None = None
        retired_now = _retired_target_from_record(runner_record, prev_target)
        if retired_now is not None:
            reached.add(int(retired_now))
        if record_trace:
            pre_m = state.pos_m.copy()
            pre_u = (state.pos_m / s).tolist()
            est = np.array([pose.x, pose.y, pose.z]) if pose is not None else np.full(3, np.nan)
            rec = {
                "t": round(t, 3),
                "step": step,
                "true_m": [round(float(v), 4) for v in pre_m],
                "true_map_u": [float(v) for v in pre_u],
                "est_map_u": [round(float(v), 5) for v in est],
                "yaw_deg": round(math.degrees(state.yaw), 2),
                "action": cmd_action,
                "phase": phase,
                "target_idx": int(ctrl.target_index),
                "progress": round(float(runner_record.get("progress", 0.0) or 0.0), 4),
                "path_err_u": round(float(runner_record.get("path_error_u", 0.0) or 0.0), 5),
                "dev_u": round(float(dist), 5) if math.isfinite(dist) else None,
                "ground_speed_mps": float(map_frame.horizontal_distance(ground_velocity)),
                "body_velocity_mps": [float(body_velocity[0]), float(body_velocity[1])],
                "measured_body_velocity_mps": None if tele_body is None else [float(tele_body[0]), float(tele_body[1])],
                "telemetry_stamp": None if tele_body is None else float(tele_body[2]),
                "pcmd_requested": list(action_pcmd),
                "pcmd": [roll, pitch, yaw_p, gaz],
                "pcmd_applied": [roll, pitch, yaw_p, gaz],
                "speed_guard_status": str(getattr(speed_guard.backend.state, "autonomous_speed_guard_status", "")),
                "pose_stamp": None if pose is None else float(pose.stamp),
                "map_confirmed": bool(getattr(pose, "map_confirmed", True)) if pose is not None else False,
                "position_observed": bool(getattr(pose, "position_observed", True)) if pose is not None else False,
                "retired_target_idx": retired_now,
                "gust_displacement_m": [0.0, 0.0, 0.0],
                "gust_time_s": None,
                "loc_fault": loc.fault,
                "landing_stable_samples": int(runner_record.get("landing_stable_samples", 0) or 0),
                "landing_stable_span_s": round(float(runner_record.get("landing_stable_span_s", 0.0) or 0.0), 3),
                "landing_speed_measured_mps": runner_record.get("landing_speed_mps"),
                "landing_speed_reason": runner_record.get("landing_speed_reason"),
            }

        wind, gust_phase, gust_displacement = _integrate_plant(
            state,
            wind,
            hover_anchor,
            params,
            map_frame,
            rng,
            (roll, pitch, yaw_p, gaz),
            vh_per_pct,
            vv_per_pct,
            yaw_per_pct,
            gust_phase,
            elapsed_s=t,
        )
        dist_m += float(np.linalg.norm(state.pos_m - prev_pos))
        prev_pos = state.pos_m.copy()

        t += DT
        loc.push_truth(t, state.pos_m, state.yaw)

        if rec is not None:
            rec["gust_displacement_m"] = gust_displacement.tolist()
            rec["gust_time_s"] = round(t, 3) if float(np.linalg.norm(gust_displacement)) > 1e-12 else None
            sim_trace.append(rec)

    progress_pct = 100.0 * float(ctrl.progress_s) / float(ctrl.path_len)
    # waypoints actually retired by the sequencer (monotonic front), plus sphere hits
    result = SimResult(
        success=bool(success),
        reason=reason,
        time_s=round(t, 2),
        steps=step + 1,
        progress_pct=round(progress_pct, 1),
        dist_flown_m=round(dist_m, 2),
        dist_flown_u=round(dist_m / s, 3),
        max_dev_u=round(max_dev, 4),
        max_dev_m=round(max_dev * s, 3),
        hover_ticks=hover_ticks,
        yaw_align_ticks=yaw_ticks,
        waypoints_reached=sorted(reached),
        speed_guard_interventions=speed_guard_interventions,
        max_waypoint_no_progress_s=max_waypoint_no_progress_s,
        yaw_overlap_ticks=yaw_overlap_ticks,
        max_true_dev_m=round(debrief.max_true_dev_u * s, 3),
        max_est_error_m=round(debrief.max_est_error_u * s, 3),
        max_climb_m=round(debrief.max_climb_m, 3),
        weak_steer_max_s=round(debrief.weak_steer_max_s, 2),
        near_waypoint_turn_ticks=debrief.near_waypoint_turn_ticks,
        vertical_waivers=debrief.vertical_waivers,
        trace=sim_trace if record_trace else [],
    )
    home_label = 1 if params.return_to_start else (n_wp - 1) // 2 + 1
    info = {
        "route": str(params.route),
        "meters_per_unit": s,
        "path_len_u": round(float(ctrl.path_len), 4),
        "path_len_m": round(float(ctrl.path_len) * s, 2),
        "n_waypoints_flown": n_wp,
        "outbound_n": (n_wp + 1) // 2,
        "arrive_radius_u": round(float(arrive_r[0]), 5),
        "arrive_radius_m": round(float(arrive_r[0]) * s, 3),
        "join_target_1based": join_target + 1,
        "final_target_1based": home_label if success else int(ctrl.target_index) + 1,
        "seed": params.seed,
        "localization_wait_s": initial_wait,
        "route_time_s": round(t - initial_wait, 2),
    }
    return result, info


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _save_plots(
    outdir: Path, result: SimResult, info: dict, ctrl_wp: list[np.ndarray], params: SimParams
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    s = info["meters_per_unit"]
    tr = result.trace
    if not tr:
        return []
    true_u = np.array([r["true_map_u"] for r in tr])
    est_u = np.array([r["est_map_u"] for r in tr])
    W = np.array([np.asarray(p, float) for p in ctrl_wp])
    arr = info["arrive_radius_u"]
    made: list[str] = []

    # 1) top-down: map east (x) vs north-ish. Raw GLOMAP horizontal = X/Z, so
    # plot X vs Z with per-waypoint arrival spheres.
    fig, ax = plt.subplots(figsize=(9, 7))
    for a, b in zip(W[:-1], W[1:]):
        ax.plot([a[0], b[0]], [a[2], b[2]], "k-", lw=1.2, alpha=0.5)
    th = np.linspace(0, 2 * math.pi, 49)
    n_drawn = (len(W) + 1) // 2  # return leg revisits the same geometry
    for i, p in enumerate(W):
        ax.plot(
            p[0] + arr * np.cos(th), p[2] + arr * np.sin(th), color="tab:green", lw=0.8, alpha=0.7
        )
        if i < n_drawn:  # label drawn waypoints once; W{n_drawn+1}.. revisit them
            ax.text(
                p[0], p[2], f"W{i + 1}", fontsize=8, ha="center", va="bottom", color="tab:green"
            )
    ax.plot(true_u[:, 0], true_u[:, 2], color="tab:red", lw=1.0, label="true (m→u)")
    ax.plot(est_u[:, 0], est_u[:, 2], color="tab:orange", lw=0.6, alpha=0.6, label="localizer")
    ax.scatter([true_u[0, 0]], [true_u[0, 2]], c="blue", s=60, marker="o", label="takeoff")
    ax.scatter([true_u[-1, 0]], [true_u[-1, 2]], c="red", s=60, marker="X", label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map east X [map units]")
    ax.set_ylabel("map horizontal Z [map units]")
    verdict = "SUCCESS" if result.success else "FAIL"
    ax.set_title(
        f"route patrol + return-to-start [{verdict}]  "
        f"arrive={arr:.3f}u ({arr * s:.2f}m)  "
        f"s={s:g}m/u"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    p1 = outdir / "map_topdown.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    made.append(p1.name)

    # 2) 3-D view
    fig = plt.figure(figsize=(9, 7))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot(W[:, 0], W[:, 2], W[:, 1], "k--", lw=1.2, label="route")
    ax3.scatter(W[:, 0], W[:, 2], W[:, 1], c="green", s=25)
    ax3.plot(true_u[:, 0], true_u[:, 2], true_u[:, 1], color="tab:red", lw=1.0, label="true")
    ax3.set_xlabel("X [u]")
    ax3.set_ylabel("Z [u]")
    ax3.set_zlabel("Y [u]")
    ax3.set_title(
        f"3-D patrol ({result.progress_pct:.0f}% route, {result.dist_flown_m:.1f} m flown)"
    )
    ax3.legend(fontsize=8)
    p2 = outdir / "route_3d.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    made.append(p2.name)

    # 3) commands / tracking over time
    tt = np.array([r["t"] for r in tr])
    dev = np.array([r["dev_u"] if r["dev_u"] is not None else np.nan for r in tr])
    pcmd = np.array([r["pcmd"] for r in tr])
    prog = np.array([r["progress"] for r in tr])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(tt, dev, color="tab:blue", lw=0.8, label="route deviation (telemetry)")
    axes[0].set_ylabel("map units")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"tracking + PCMD  ({result.reason})")
    for j, name in enumerate(["roll", "pitch", "yaw", "gaz"]):
        axes[1].plot(tt, pcmd[:, j], lw=0.7, label=name)
    axes[1].set_ylabel("PCMD %")
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].grid(alpha=0.3)
    axes[2].plot(tt, prog * 100.0, color="tab:green", lw=1.0)
    axes[2].set_ylabel("route progress %")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)
    p3 = outdir / "tracking_pcmd.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    made.append(p3.name)
    return made


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _default_route() -> Path:
    return (
        REPO_ROOT
        / "地圖檔"
        / "場域"
        / "river_site"
        / "routes"
        / "flight_route_20260913_145051_ec20651d.json"
    )


def _default_align() -> Path | None:
    p = (
        REPO_ROOT
        / "地圖檔"
        / "場域"
        / "river_site"
        / "releases"
        / "river_gluemap_all8_direct_20260908"
        / "localization"
        / "T_align_gravity.json"
    )
    return p if p.exists() else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", type=Path, default=_default_route())
    ap.add_argument("--align", type=Path, default=_default_align())
    ap.add_argument("--no-align", action="store_true", help="use legacy map frame")
    ap.add_argument(
        "--meters-per-unit",
        type=float,
        default=10.0,
        help="true scale s (unknown to the controller)",
    )
    ap.add_argument(
        "--sweep-scales", default="", help="comma list, e.g. 5,10,20: headless robustness sweep"
    )
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--no-return-to-start", action="store_true")
    ap.add_argument("--yaw-sign", type=int, default=1, choices=[-1, 1])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration-s", type=float, default=900.0)
    ap.add_argument(
        "--localization-wait-s",
        type=float,
        default=0.0,
        help="initial wait charged to the total budget",
    )
    ap.add_argument(
        "--start-offset-u",
        type=float,
        nargs=3,
        default=[0.25, 0.0, 0.0],
        metavar=("LATERAL", "ALONG", "UP"),
        help="takeoff offset from WP1 in map units",
    )
    ap.add_argument("--initial-yaw-error-deg", type=float, default=30.0)
    ap.add_argument(
        "--yaw-bias-deg", type=float, default=0.0, help="residual map-vs-body yaw misalignment"
    )
    ap.add_argument("--pos-bias-u", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--scale-error-pct", type=float, default=0.0)
    ap.add_argument("--pos-noise-u", type=float, default=0.004)
    ap.add_argument("--yaw-noise-deg", type=float, default=1.0)
    ap.add_argument("--oly-noise-deg", type=float, default=0.1)
    ap.add_argument("--latency-ms", type=float, default=120.0)
    ap.add_argument("--drop-rate", type=float, default=0.02)
    ap.add_argument("--outage-every-s", type=float, default=0.0)
    ap.add_argument("--outage-dur-s", type=float, default=0.0)
    ap.add_argument(
        "--cruise-mps",
        type=float,
        default=0.90,
        help="horizontal speed at max_translation_pcmd in the plant model",
    )
    ap.add_argument("--wind-sigma-mps", type=float, default=0.05)
    ap.add_argument(
        "--hover-hold-mps",
        type=float,
        default=0.2,
        help="firmware hover-hold authority in m/s; 0 = pure drift",
    )
    ap.add_argument("--gust-every-s", type=float, default=0.0)
    ap.add_argument("--gust-m", type=float, default=0.0)
    ap.add_argument("--tau-tilt-s", type=float, default=1e-3)
    ap.add_argument("--thrust-margin", type=float, default=99.0)
    ap.add_argument("--battery-sag-frac", type=float, default=0.0)
    ap.add_argument("--ground-effect-frac", type=float, default=0.0)
    ap.add_argument("--ground-effect-height-m", type=float, default=1.0)
    ap.add_argument("--yaw-coupling", type=float, default=0.0)
    ap.add_argument("--jump-rate", type=float, default=0.0)
    ap.add_argument("--jump-max-u", type=float, default=0.0)
    ap.add_argument("--outlier-rate", type=float, default=0.0)
    ap.add_argument("--outlier-max-u", type=float, default=0.0)
    ap.add_argument("--weak-rate", type=float, default=0.0)
    ap.add_argument("--weak-noise-gain", type=float, default=3.0)
    ap.add_argument("--predicted-rate", type=float, default=0.0)
    ap.add_argument("--blackout-every-s", type=float, default=0.0)
    ap.add_argument("--blackout-dur-s", type=float, default=0.0)
    ap.add_argument("--blackout-drift-mps", type=float, default=0.0)
    ap.add_argument("--blackout-yaw-drift-deg-s", type=float, default=0.0)
    ap.add_argument("--drift-every-s", type=float, default=0.0,
                    help="localization-failure window period (0 = off)")
    ap.add_argument("--drift-dur-s", type=float, default=0.0,
                    help="seconds per window with no map fix, only a drifting weak estimate")
    ap.add_argument("--drift-offset-s", type=float, default=0.0, help="start of the first window")
    ap.add_argument("--drift-gain", type=float, default=1.0,
                    help="weak-estimate displacement per unit of true displacement")
    ap.add_argument("--drift-mps", type=float, default=0.0,
                    help="extra weak-estimate drift speed, random horizontal direction per window")
    ap.add_argument("--vertical-gain", type=float, default=1.0,
                    help="reported height change per unit of true height change (0 = frozen)")
    ap.add_argument("--baro-noise-m", type=float, default=0.0,
                    help="white noise on the firmware altitude fed to the height cross-check")
    ap.add_argument("--baro-drift-mps", type=float, default=0.0,
                    help="constant firmware-altitude drift (m/s) fed to the height cross-check")
    ap.add_argument("--start-offset-frame", choices=["lateral", "raw"], default="lateral",
                    help="lateral: --start-offset-u is (lateral, along, up); raw: map x/y/z")
    ap.add_argument("--ground-altitude-m", type=float, default=None)
    ap.add_argument("--telemetry-rate-hz", type=float, default=5.0)
    ap.add_argument("--telemetry-latency-ms", type=float, default=200.0)
    ap.add_argument("--telemetry-noise-mps", type=float, default=0.0)
    ap.add_argument("--telemetry-drop-rate", type=float, default=0.0)
    ap.add_argument("--wind-mean-mps", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("E", "N", "U"))
    ap.add_argument("--pose-rate-hz", type=float, default=20.0)
    ap.add_argument("--latency-jitter-ms", type=float, default=0.0)
    ap.add_argument("--position-correlation-s", type=float, default=0.0)
    ap.add_argument("--yaw-correlation-s", type=float, default=0.0)
    ap.add_argument("--bias-walk-u-sqrt-s", type=float, default=0.0)
    ap.add_argument("--reseed-confirm-frames", type=int, default=0)
    ap.add_argument("--command-latency-ms", type=float, default=0.0)
    ap.add_argument("--command-drop-rate", type=float, default=0.0)
    ap.add_argument("--command-ttl-s", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None,
                    help="artifact directory (default: outputs/sim_autoflight/run_<timestamp>)")
    ap.add_argument("--scenario", type=str, default="",
                    choices=["", *sorted(REGRESSION_SCENARIOS)],
                    help="apply a localization-defect regression preset (wins over the "
                         "matching defect flags for its keys)")
    ap.add_argument("--no-plot", action="store_true")
    return ap


def _params_from_args(a: argparse.Namespace, *, seed: int, scale: float) -> SimParams:
    return SimParams(
        route=a.route,
        align=None if a.no_align else a.align,
        meters_per_unit=scale,
        return_to_start=not a.no_return_to_start,
        yaw_sign=a.yaw_sign,
        seed=seed,
        duration_s=a.duration_s,
        start_offset_u=tuple(a.start_offset_u),  # type: ignore[arg-type]
        start_offset_frame=a.start_offset_frame,
        initial_yaw_error_deg=a.initial_yaw_error_deg,
        yaw_bias_deg=a.yaw_bias_deg,
        pos_bias_u=tuple(a.pos_bias_u),  # type: ignore[arg-type]
        scale_error_pct=a.scale_error_pct,
        pos_noise_u=a.pos_noise_u,
        yaw_noise_deg=a.yaw_noise_deg,
        oly_noise_deg=a.oly_noise_deg,
        latency_ms=a.latency_ms,
        drop_rate=a.drop_rate,
        outage_every_s=a.outage_every_s,
        outage_dur_s=a.outage_dur_s,
        cruise_mps=a.cruise_mps,
        wind_sigma_mps=a.wind_sigma_mps,
        hover_hold_mps=a.hover_hold_mps,
        gust_every_s=a.gust_every_s,
        gust_m=a.gust_m,
        tau_tilt_s=a.tau_tilt_s,
        thrust_margin=a.thrust_margin,
        battery_sag_frac=a.battery_sag_frac,
        ground_effect_frac=a.ground_effect_frac,
        ground_effect_height_m=a.ground_effect_height_m,
        yaw_coupling=a.yaw_coupling,
        jump_rate=a.jump_rate,
        jump_max_u=a.jump_max_u,
        outlier_rate=a.outlier_rate,
        outlier_max_u=a.outlier_max_u,
        weak_rate=a.weak_rate,
        weak_noise_gain=a.weak_noise_gain,
        predicted_rate=a.predicted_rate,
        blackout_every_s=a.blackout_every_s,
        blackout_dur_s=a.blackout_dur_s,
        blackout_drift_mps=a.blackout_drift_mps,
        blackout_yaw_drift_deg_s=a.blackout_yaw_drift_deg_s,
        drift_every_s=a.drift_every_s,
        drift_dur_s=a.drift_dur_s,
        drift_offset_s=a.drift_offset_s,
        drift_gain=a.drift_gain,
        drift_mps=a.drift_mps,
        vertical_gain=a.vertical_gain,
        baro_noise_m=a.baro_noise_m,
        baro_drift_mps=a.baro_drift_mps,
        ground_altitude_m=a.ground_altitude_m,
        telemetry_rate_hz=a.telemetry_rate_hz,
        telemetry_latency_ms=a.telemetry_latency_ms,
        telemetry_noise_mps=a.telemetry_noise_mps,
        telemetry_drop_rate=a.telemetry_drop_rate,
        wind_mean_mps=tuple(a.wind_mean_mps),
        pose_rate_hz=a.pose_rate_hz,
        latency_jitter_ms=a.latency_jitter_ms,
        position_correlation_s=a.position_correlation_s,
        yaw_correlation_s=a.yaw_correlation_s,
        bias_walk_u_sqrt_s=a.bias_walk_u_sqrt_s,
        reseed_confirm_frames=a.reseed_confirm_frames,
        command_latency_ms=a.command_latency_ms,
        command_drop_rate=a.command_drop_rate,
        command_ttl_s=a.command_ttl_s,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.route.exists():
        print(f"[sim] route not found: {args.route}", flush=True)
        return 2
    if args.align is not None and not args.no_align and not args.align.exists():
        print(f"[sim] align not found: {args.align}", flush=True)
        return 2

    scales = [float(x) for x in str(args.sweep_scales).split(",") if x.strip()] or [
        float(args.meters_per_unit)
    ]
    sweep = len(scales) > 1 or int(args.seeds) > 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out or (REPO_ROOT / "outputs" / "sim_autoflight" / f"run_{stamp}")
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rc = 0
    for scale in scales:
        for k in range(max(1, int(args.seeds))):
            seed = int(args.seed) + k
            params = _params_from_args(args, seed=seed, scale=scale)
            if args.scenario:
                params = apply_regression_scenario(params, args.scenario)
            result, info = run_sim(params, record_trace=not sweep)
            row = {
                "scale_m_per_u": scale,
                "seed": seed,
                "scenario": args.scenario,
                "success": result.success,
                "reason": result.reason,
                "time_s": result.time_s,
                "progress_pct": result.progress_pct,
                "dist_m": result.dist_flown_m,
                "dist_u": result.dist_flown_u,
                "max_dev_m": result.max_dev_m,
                "hover_ticks": result.hover_ticks,
                "yaw_align_ticks": result.yaw_align_ticks,
                "n_wp_hits": len(result.waypoints_reached),
                "path_len_m": info["path_len_m"],
                "arrive_m": info["arrive_radius_m"],
                "max_true_dev_m": result.max_true_dev_m,
                "max_est_error_m": result.max_est_error_m,
                "max_climb_m": result.max_climb_m,
                "weak_steer_max_s": result.weak_steer_max_s,
                "near_waypoint_turn_ticks": result.near_waypoint_turn_ticks,
                "vertical_waivers": result.vertical_waivers,
            }
            rows.append(row)
            flag = "OK " if result.success else "FAIL"
            print(
                f"[sim] [{flag}] s={scale:g}m/u seed={seed} "
                f"t={result.time_s:.0f}s prog={result.progress_pct:.0f}% "
                f"flown={result.dist_flown_m:.1f}m max_dev={result.max_dev_m:.2f}m "
                f"true_dev={result.max_true_dev_m:.2f}m est_err={result.max_est_error_m:.2f}m "
                f"climb={result.max_climb_m:.2f}m weak_steer={result.weak_steer_max_s:.1f}s "
                f"near_wp_turn={result.near_waypoint_turn_ticks} waivers={result.vertical_waivers} "
                f"-> {result.reason}",
                flush=True,
            )
            if not result.success:
                rc = 1
            if not sweep:
                # full artifacts for the single run
                ctrl, _cfg, _mf, _doc = build_production_controller(
                    params.route, params.align, return_to_start=params.return_to_start
                )
                summary = {
                    "result": dict(result.__dict__),
                    "info": info,
                    "scenario": args.scenario,
                    "ctrl_waypoints_u": [[float(v) for v in p] for p in ctrl.wp],
                    "params": {
                        k: (
                            str(v)
                            if isinstance(v, Path)
                            else list(v)
                            if isinstance(v, tuple)
                            else v
                        )
                        for k, v in params.__dict__.items()
                    },
                }
                summary["result"].pop("trace", None)
                (outdir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                with (outdir / "trace.csv").open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(
                        [
                            "t",
                            "step",
                            "true_x_m",
                            "true_y_m",
                            "true_z_m",
                            "true_x_u",
                            "true_y_u",
                            "true_z_u",
                            "est_x_u",
                            "est_y_u",
                            "est_z_u",
                            "yaw_deg",
                            "action",
                            "phase",
                            "target_idx",
                            "progress",
                            "path_err_u",
                            "dev_u",
                            "roll",
                            "pitch",
                            "yaw",
                            "gaz",
                            "body_fwd_mps",
                            "body_right_mps",
                            "meas_fwd_mps",
                            "meas_right_mps",
                            "telemetry_stamp",
                            "ground_speed_mps",
                            "req_roll",
                            "req_pitch",
                            "req_yaw",
                            "req_gaz",
                            "speed_guard_status",
                            "pose_stamp",
                            "map_confirmed",
                            "retired_target_idx",
                            "gust_time_s",
                            "gust_dx_m",
                            "gust_dy_m",
                            "gust_dz_m",
                            "loc_fault",
                        ]
                    )
                    for r in result.trace:
                        body = r.get("body_velocity_mps") or (float("nan"), float("nan"))
                        meas = r.get("measured_body_velocity_mps") or (float("nan"), float("nan"))
                        gust = r.get("gust_displacement_m") or (float("nan"), float("nan"), float("nan"))
                        w.writerow(
                            [
                                r["t"],
                                r["step"],
                                *r["true_m"],
                                *r["true_map_u"],
                                *r["est_map_u"],
                                r["yaw_deg"],
                                r["action"],
                                r["phase"],
                                r["target_idx"],
                                r["progress"],
                                r["path_err_u"],
                                r["dev_u"],
                                *r["pcmd"],
                                body[0],
                                body[1],
                                meas[0],
                                meas[1],
                                r.get("telemetry_stamp"),
                                r.get("ground_speed_mps"),
                                *(r.get("pcmd_requested") or r["pcmd"]),
                                r.get("speed_guard_status", ""),
                                r.get("pose_stamp"),
                                r.get("map_confirmed"),
                                r.get("retired_target_idx"),
                                r.get("gust_time_s"),
                                gust[0],
                                gust[1],
                                gust[2],
                                r.get("loc_fault", "ok"),
                            ]
                        )
                if not args.no_plot:
                    try:
                        made = _save_plots(
                            outdir, result, info, [np.array(p, float) for p in ctrl.wp], params
                        )
                    except Exception as exc:  # plots never fail the verdict
                        print(f"[sim] plot failed ({exc!r}); csv/json kept", flush=True)
                        made = []
                else:
                    made = []
                print(
                    f"[sim] artifacts: {outdir} (summary.json trace.csv {' '.join(made)})",
                    flush=True,
                )
                print(
                    f"[sim] route {info['path_len_m']:.1f}m over {info['n_waypoints_flown']} flown "
                    f"waypoints (out-and-back), arrive {info['arrive_radius_m']:.2f}m, "
                    f"max excursion {result.max_dev_m:.2f}m",
                    flush=True,
                )

    if sweep:
        with (outdir / "sweep.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        ok = sum(1 for r in rows if r["success"])
        print(f"[sim] sweep {ok}/{len(rows)} success -> {outdir}/sweep.csv", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
