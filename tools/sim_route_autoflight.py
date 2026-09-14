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
from heading_fusion import HeadingEstimator  # noqa: E402
from route_domain import RouteDocument  # noqa: E402
from operator_autonomy import DesktopRouteAutonomy  # noqa: E402
from landing_transition import decide_route_completion_landing  # noqa: E402

CTRL_HZ = 20
DT = 1.0 / CTRL_HZ

# ANAFI white-paper capability maxima (see module docstring).
VH_MAX_MPS = 15.0
VV_MAX_MPS = 4.0
YAW_MAX_DEG_S = 200.0

MAX_SIM_OUTAGE_S = 4.0  # reject an unrecovered simulation, without a flight command


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
    cfg = rpf.config_for_route(route_path, base)
    ctrl = rpf.RouteAutoController(doc.controller_waypoints(), poles=[], config=cfg)
    return ctrl, cfg, map_frame, doc


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


class Localizer:
    """True meters -> reported map-units with misalignment residue + defects.

    Truth: p_map_truth = pos_m / s. Report inverts with a wrong scale
    (1+scale_err), then adds constant yaw/position biases, Gaussian noise,
    latency (delay line) and dropouts/outages (stale stamp reuse).
    """

    def __init__(
        self,
        s: float,
        yaw_bias: float,
        pos_bias_u: np.ndarray,
        scale_err: float,
        pos_sigma_u: float,
        yaw_sigma: float,
        latency_s: float,
        drop_rate: float,
        outage_every_s: float,
        outage_dur_s: float,
        rng: np.random.Generator,
    ):
        self.s = float(s)
        self.yaw_bias = float(yaw_bias)
        self.pos_bias = np.asarray(pos_bias_u, float)
        self.scale_err = float(scale_err)
        self.pos_sigma = float(pos_sigma_u)
        self.yaw_sigma = float(yaw_sigma)
        self.latency = float(latency_s)
        self.drop_rate = float(drop_rate)
        self.outage_every = float(outage_every_s)
        self.outage_dur = float(outage_dur_s)
        self.rng = rng
        self._buf: list[tuple[float, np.ndarray, float]] = []  # (t, pos_m, yaw)
        self._last_report: rpf.Pose | None = None

    def push_truth(self, t: float, state: DroneState) -> None:
        self._buf.append((t, state.pos_m.copy(), float(state.yaw)))
        horizon = t - self.latency - 2.0
        while len(self._buf) > 2 and self._buf[0][0] < horizon:
            self._buf.pop(0)

    def _in_outage(self, t: float) -> bool:
        return (
            self.outage_every > 0
            and self.outage_dur > 0
            and (t % self.outage_every) < self.outage_dur
        )

    def report(self, t: float) -> rpf.Pose | None:
        """Pose in the controller's map frame, or None when nothing fresh."""
        if self._in_outage(t):
            return None  # total loss: no frame at all this tick
        # delayed truth sample
        want = t - self.latency
        sample = self._buf[0]
        for entry in self._buf:
            if entry[0] <= want:
                sample = entry
            else:
                break
        _, pos_m, yaw = sample
        if self.rng.random() < self.drop_rate and self._last_report is not None:
            return self._last_report  # stale reuse: same stamp -> controller HOVERs
        p_map = pos_m / (self.s * (1.0 + self.scale_err))
        p_map = p_map + self.pos_bias + self.rng.normal(0.0, self.pos_sigma, 3)
        yaw_rep = yaw + self.yaw_bias + math.radians(self.rng.normal(0.0, self.yaw_sigma))
        pose = rpf.Pose(
            float(p_map[0]),
            float(p_map[1]),
            float(p_map[2]),
            float(yaw_rep),
            stamp=float(sample[0]),
        )
        self._last_report = pose
        return pose


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
    # plant
    cruise_mps: float = 0.45  # horizontal speed at max_translation_pcmd
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
    gust_every_s: float = 0.0
    gust_m: float = 0.0


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
    trace: list[dict] = field(default_factory=list)


class _SimulationAutonomyChecks(DesktopRouteAutonomy):
    """Run production speed/stall checks without a flight worker or transport."""

    def __init__(self, limit: float, controller):
        self.backend = SimpleNamespace(state=SimpleNamespace(
            autonomous_speed_limit_enabled=True, autonomous_speed_limit_mps=limit,
        ))
        self.stamp = 0.0
        self.now = lambda: self.stamp
        self._speed_guard_latched = False
        self._speed_command_scale = 1.0
        self.controller = controller
        self.phase = "ROUTE"
        self._paused = SimpleNamespace(is_set=lambda: False)
        self._waypoint_progress = None
        self.failure_reason = None

    def _clear_motion(self):
        pass

    def _send_zero(self, reason):
        return True

    def _latch_auto_failure(self, detail, *, zero_reason):
        self.failure_reason = detail

    def apply(self, pcmd, speed: float, stamp: float):
        self.stamp = stamp
        self.backend.state.ground_speed_mps = speed
        self.backend.state.ground_speed_mono_ns = int(stamp * 1e9)
        _active, rejection = self._apply_speed_guard(pcmd)
        if rejection is not None:
            return rejection[2]
        return self._apply_turn_speed_guard((
            round(pcmd[0] * self._speed_command_scale),
            round(pcmd[1] * self._speed_command_scale), pcmd[2], pcmd[3],
        ))


def _body_axes(yaw: float, map_frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = math.cos(yaw) * map_frame.east + math.sin(yaw) * map_frame.north
    right = math.sin(yaw) * map_frame.east - math.cos(yaw) * map_frame.north
    return (np.asarray(fwd, float), np.asarray(right, float), np.asarray(map_frame.up, float))


def run_sim(params: SimParams, *, record_trace: bool = True) -> tuple[SimResult, dict]:
    rng = np.random.default_rng(params.seed)
    ctrl, cfg, map_frame, doc = build_production_controller(
        params.route, params.align, return_to_start=params.return_to_start
    )
    s = float(params.meters_per_unit)
    if not math.isfinite(s) or s <= 0:
        raise ValueError("meters_per_unit must be finite and > 0")

    wp = [np.array(p, float) for p in ctrl.wp]
    n_wp = len(wp)

    # --- takeoff state (meters, world frame parallel to map basis) ---
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

    # AUTO start rule: waypoint 1 first, even when takeoff sits nearer a later point.
    ctrl.start_after_nearest_waypoint(np.asarray(start_map, float))
    join_target = int(ctrl.target_index)

    loc = Localizer(
        s,
        math.radians(params.yaw_bias_deg),
        np.array(params.pos_bias_u, float),
        params.scale_error_pct / 100.0,
        params.pos_noise_u,
        params.yaw_noise_deg,
        params.latency_ms / 1000.0,
        params.drop_rate,
        params.outage_every_s,
        params.outage_dur_s,
        rng,
    )
    yaw_ctl = rpf.YawAlignedPcmdController(cfg)
    heading_est = HeadingEstimator(map_frame)

    # per-% speed calibration: max_translation_pcmd flies cruise_mps horizontally.
    vh_per_pct = params.cruise_mps / max(1, int(cfg.max_translation_pcmd))
    vv_per_pct = params.max_vertical_speed_mps / 100.0
    yaw_per_pct = params.max_rotation_speed_deg_s / 100.0

    loc.push_truth(0.0, state)
    wind = np.zeros(3)
    hover_anchor: np.ndarray | None = None
    max_steps = int(params.duration_s * CTRL_HZ)
    sub = 4
    sdt = DT / sub

    trace: list[dict] = []
    reached: set[int] = set()
    last_fresh_stamp: float | None = None
    lost_since: float | None = None
    hover_ticks = yaw_ticks = 0
    speed_guard = _SimulationAutonomyChecks(params.speed_limit_mps, ctrl)
    speed_guard_interventions = 0
    max_waypoint_no_progress_s = 0.0
    max_dev = 0.0
    land_hold_since: float | None = None
    reason = "step cap"
    success = False
    dist_m = 0.0
    t = 0.0
    prev_pos = state.pos_m.copy()
    gust_phase = 0.0

    arrive_r = [float(ctrl._arrive_radius_for(i)) for i in range(n_wp)]

    def _olympe_report() -> float:
        # NED clockwise-from-North, inverse of HeadingEstimator._olympe_to_map_ccw
        oly = (math.pi / 2.0 - state.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return float(oly + math.radians(rng.normal(0.0, params.oly_noise_deg)))

    for step in range(max_steps):
        oly_yaw = _olympe_report()
        visual = loc.report(t)
        fresh = visual is not None and (last_fresh_stamp is None or visual.stamp > last_fresh_stamp)
        if fresh:
            heading_est.update(float(visual.yaw), oly_yaw)
        fused = heading_est.heading(oly_yaw)
        pose = (
            rpf.Pose(
                float(visual.x),
                float(visual.y),
                float(visual.z),
                float(fused),
                stamp=float(visual.stamp),
            )
            if visual is not None and fused is not None
            else None
        )
        if fresh:
            last_fresh_stamp = float(pose.stamp)
            lost_since = None
        else:
            if last_fresh_stamp is None:
                lost_since = 0.0 if lost_since is None else lost_since
            else:
                # age of the newest fix the localizer ever produced
                lost_since = t - last_fresh_stamp if lost_since is None else lost_since
                if fresh is False and pose is None:
                    pass
            # total-loss timer uses sim time since last fresh stamp
            if last_fresh_stamp is not None and t - last_fresh_stamp >= MAX_SIM_OUTAGE_S:
                reason = f"simulation localization outage exceeded {MAX_SIM_OUTAGE_S:.0f}s"
                break

        prev_target = int(ctrl.target_index)
        cmd = ctrl.step(pose, t)
        if pose is not None and fresh:
            now_target = int(ctrl.target_index)
            if prev_target != now_target:
                reached.add(prev_target)
            elif now_target not in reached and ctrl.waypoint_reached(
                np.array([pose.x, pose.y, pose.z]), now_target
            ):
                reached.add(now_target)

        # The controller owns segment recovery; the simulator measures its result.
        action_pcmd: tuple[int, int, int, int] = (0, 0, 0, 0)
        phase = ""
        if pose is not None and fresh:
            dist, _nearest_pt, _seg, _s = rpf.project_to_path(
                np.array([pose.x, pose.y, pose.z]), list(ctrl.wp), ctrl.cum
            )
            if dist > max_dev:
                max_dev = dist
        else:
            dist = float("nan")

        if cmd.should_land and cmd.action == "ABORT_PENDING_INSPECTION":
            reason = "ABORT: pending inspection at route end"
            break

        if cmd.action == "INSPECT":
            # patrol routes carry no poles; if they ever do, auto-ack like the
            # gimbal/capture pipeline would after alignment + drain dwell.
            action_pcmd = (
                rpf.command_to_body_percent(
                    cmd, pose, config=cfg, yaw_sign=params.yaw_sign, require_yaw_alignment=True
                )
                if pose is not None
                else (0, 0, 0, 0)
            )
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
        elif cmd.should_land and cmd.action == "LAND":
            action_pcmd = (0, 0, 0, 0)
            phase = "landing_hold"
            speed_mps = map_frame.horizontal_distance(state.vel_m + wind)
            transition = decide_route_completion_landing(cmd.action, (speed_mps, t), t)
            if transition.outcome == "land":
                success = True
                reason = transition.reason
                if record_trace:
                    trace.append({
                        "t": round(t, 3), "step": step,
                        "true_m": state.pos_m.tolist(),
                        "true_map_u": (state.pos_m / s).tolist(),
                        "est_map_u": pose.xyz.tolist(),
                        "yaw_deg": round(math.degrees(state.yaw), 2),
                        "action": "LAND", "phase": phase,
                        "target_idx": int(ctrl.target_index), "progress": 1.0,
                        "path_err_u": float(cmd.path_error), "dev_u": float(dist),
                        "ground_speed_mps": speed_mps, "pcmd": list(action_pcmd),
                    })
                break
        else:
            land_hold_since = None if cmd.action != "INSPECT" else land_hold_since
            if pose is None:
                action_pcmd = (0, 0, 0, 0)
                phase = "stale_hover"
            elif cmd.action in {"FOLLOW", "REJOIN", "FINAL_HOLD"}:
                body_forward, body_right, _body_up = _body_axes(state.yaw, map_frame)
                ground_velocity = state.vel_m + wind
                action_pcmd = yaw_ctl.update(
                    cmd, pose, t, target_key=int(ctrl.target_index), yaw_sign=params.yaw_sign,
                    body_velocity=(float(np.dot(ground_velocity, body_forward)),
                                   float(np.dot(ground_velocity, body_right)), t),
                )
                phase = f"fgc:{yaw_ctl.phase}"
                if yaw_ctl.phase in (
                    "turn",
                    "yaw_alignment_hold",
                    "yaw_alignment_confirmed",
                    "yaw_alignment_timeout",
                    "final_centering",
                ):
                    yaw_ticks += 1
            else:  # ARRIVING / HOVER
                action_pcmd = (0, 0, 0, 0)
                phase = cmd.action.lower()
                hover_ticks += 1

        guarded = speed_guard.apply(
            action_pcmd, map_frame.horizontal_distance(state.vel_m + wind), t
        )
        speed_guard_interventions += guarded != action_pcmd
        progress_record = {
            "pcmd": guarded, "pcmd_phase": phase.removeprefix("fgc:"),
            "target_index": int(ctrl.target_index),
            "pose_u": None if pose is None else pose.xyz,
            "pose_stamp": None if pose is None else float(pose.stamp),
            "path_error_u": float(cmd.path_error),
            "yaw_error_deg": None if pose is None else math.degrees(rpf.wrap_angle(cmd.yaw_target - pose.yaw)),
            "turn_anchor_error_u": yaw_ctl.turn_anchor_error_u,
        }
        speed_guard._check_waypoint_progress(progress_record)
        max_waypoint_no_progress_s = max(
            max_waypoint_no_progress_s, progress_record.get("waypoint_no_progress_s", 0.0)
        )
        if speed_guard.failure_reason is not None:
            reason = speed_guard.failure_reason
            break
        roll, pitch, yaw_p, gaz = (int(v) for v in guarded)
        if yaw_p and (roll or pitch):
            reason = "invalid simultaneous yaw and horizontal PCMD"
            break
        if hover_anchor is None:
            hover_anchor = state.pos_m.copy()
        anchor_delta = state.pos_m - hover_anchor
        anchor_vertical = np.dot(anchor_delta, map_frame.up) * map_frame.up
        if roll != 0 or pitch != 0:
            hover_anchor += anchor_delta - anchor_vertical
        if gaz != 0:
            hover_anchor += anchor_vertical

        # --- plant integration (meters, 80 Hz substeps) ---
        fwd_w, right_w, up_w = _body_axes(state.yaw, map_frame)
        # vh_per_pct encodes the cruise calibration (max_translation_pcmd ->
        # cruise_mps); vertical uses the configured firmware speed envelope.
        v_tgt = (
            fwd_w * (pitch * vh_per_pct) + right_w * (roll * vh_per_pct) + up_w * (gaz * vv_per_pct)
        )
        yaw_rate_tgt = -math.radians(yaw_p * yaw_per_pct)
        # hard capability caps (white-paper maxima, never exceeded even by gusts)
        if hover_anchor is not None and params.hover_hold_mps > 0:
            hold = (hover_anchor - state.pos_m) * params.hover_hold_gain
            hn = float(np.linalg.norm(hold))
            if hn > params.hover_hold_mps:
                hold = hold / hn * params.hover_hold_mps
            v_tgt = v_tgt + hold
        vh_cap = v_tgt - np.dot(v_tgt, up_w) * up_w
        if float(np.linalg.norm(vh_cap)) > VH_MAX_MPS:
            v_tgt = vh_cap / float(np.linalg.norm(vh_cap)) * VH_MAX_MPS + np.dot(v_tgt, up_w) * up_w
        vv_n = float(np.dot(v_tgt, up_w))
        if abs(vv_n) > VV_MAX_MPS:
            v_tgt = v_tgt - up_w * (vv_n - math.copysign(VV_MAX_MPS, vv_n))
        k_h = 1.0 - math.exp(-sdt / params.tau_h_s)
        k_v = 1.0 - math.exp(-sdt / params.tau_v_s)
        k_y = 1.0 - math.exp(-sdt / params.tau_yaw_s)
        gust_displacement = np.zeros(3)
        for _ in range(sub):
            # wind OU in world meters
            wind += (
                (
                    -wind / params.wind_tau_s * sdt
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
                if gust_phase >= params.gust_every_s:
                    gust_phase = 0.0
                    gdir = rng.normal(0, 1, 3)
                    gdir /= np.linalg.norm(gdir) or 1.0
                    gust = gdir * (params.gust_m / sdt)  # exact metre displacement per impulse
                    gust_displacement += gust * sdt
            vh = v_tgt - np.dot(v_tgt, up_w) * up_w  # horizontal part
            vv = np.dot(v_tgt, up_w) * up_w
            cur_h = state.vel_m - np.dot(state.vel_m, up_w) * up_w
            cur_v = np.dot(state.vel_m, up_w) * up_w
            cur_h += (vh - cur_h) * k_h
            cur_v += (vv - cur_v) * k_v
            state.vel_m = cur_h + cur_v
            state.pos_m += (state.vel_m + wind) * sdt + gust * sdt
            state.yaw_rate += (yaw_rate_tgt - state.yaw_rate) * k_y
            state.yaw = rpf.wrap_angle(state.yaw + state.yaw_rate * sdt)
        dist_m += float(np.linalg.norm(state.pos_m - prev_pos))
        prev_pos = state.pos_m.copy()

        t += DT
        loc.push_truth(t, state)

        if record_trace:
            est = np.array([pose.x, pose.y, pose.z]) if pose is not None else np.full(3, np.nan)
            trace.append(
                {
                    "t": round(t, 3),
                    "step": step,
                    "true_m": [round(float(v), 4) for v in state.pos_m],
                    "true_map_u": [float(v) for v in state.pos_m / s],
                    "est_map_u": [round(float(v), 5) for v in est],
                    "yaw_deg": round(math.degrees(state.yaw), 2),
                    "action": cmd.action,
                    "phase": phase,
                    "target_idx": int(ctrl.target_index),
                    "progress": round(float(cmd.progress), 4),
                    "path_err_u": round(float(cmd.path_error), 5),
                    "dev_u": round(float(dist), 5) if math.isfinite(dist) else None,
                    "pcmd": [roll, pitch, yaw_p, gaz],
                    "gust_displacement_m": gust_displacement.tolist(),
                }
            )

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
        trace=trace,
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
    axes[0].legend(fontsize=8)
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
        default=0.45,
        help="horizontal speed at max_translation_pcmd; "
        "default 0.45 m/s = white-paper linear mapping "
        "(3%% of 15 m/s firmware maximum, matching "
        "DESKTOP_AUTO_MAX_TRANSLATION_PCMD=3)",
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
    ap.add_argument("--out", type=Path, default=None)
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
            result, info = run_sim(params, record_trace=not sweep)
            row = {
                "scale_m_per_u": scale,
                "seed": seed,
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
            }
            rows.append(row)
            flag = "OK " if result.success else "FAIL"
            print(
                f"[sim] [{flag}] s={scale:g}m/u seed={seed} "
                f"t={result.time_s:.0f}s prog={result.progress_pct:.0f}% "
                f"flown={result.dist_flown_m:.1f}m max_dev={result.max_dev_m:.2f}m "
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
                        ]
                    )
                    for r in result.trace:
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
