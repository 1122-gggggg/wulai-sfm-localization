#!/usr/bin/env python3
"""Path-convergence controllers for the Sphinx ANAFI experiment.

Ten comparable algorithms (spec "Algorithms to compare"):

  1. naive_waypoint             fly straight at the next waypoint
  2. continuous_path            production-style full-polyline FOLLOW/REJOIN
                                (REJOIN chases the moving nearest point)
  3. segment_corridor           active-segment corridor, fixed lookahead,
                                nearest-point REJOIN, no hover/hysteresis
  4. segment_corridor_hover     3 + waypoint hover + yaw re-align per segment
  5. segment_corridor_lateral   4 + limited lateral (roll) assist
  6. segment_corridor_nose_first  4 + nose-first lookahead REJOIN (pure pursuit)
  7. adaptive_lookahead         6 + adaptive lookahead + REJOIN/FOLLOW
                                hysteresis + anti-oscillation
  8. adaptive_smoothed          7 + low-pass + rate-limited carrot target
  9. translational_waypoint     yaw-locked body-frame translation; yaw only
                                while hovering at a waypoint
 10. translational_smoothed     9 + filtered pose/heading/target

All corridor controllers share one state machine (SegmentCorridorController)
with feature flags, so differences between algorithms are exactly the feature
under test.

Units: distance thresholds are expressed in the route/model coordinate unit.
In Sphinx that unit is a meter; in real monocular SfM/hloc it is an arbitrary
map unit, so values such as tube radius 1.0 mean "1 map unit", not "1 meter".

Frame: raw map frame (X/Z horizontal, up = -Y, yaw = atan2(z, x)).
PCMD: percents in [-100, 100]; conservative caps; roll stays 0 unless lateral
assist is explicitly enabled. No Emergency anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from route_geometry import (RouteModel, adaptive_lookahead, heading_of,
                            horiz, wrap_angle)

# state names (spec)
INIT_REJOIN = "INIT_REJOIN"
SEGMENT_ALIGN = "SEGMENT_ALIGN"
SEGMENT_FOLLOW = "SEGMENT_FOLLOW"
SEGMENT_REJOIN = "SEGMENT_REJOIN"
WAYPOINT_HOVER = "WAYPOINT_HOVER"
NEXT_SEGMENT = "NEXT_SEGMENT"
LOST_OR_UNCERTAIN = "LOST_OR_UNCERTAIN"
ABORT_OR_MANUAL = "ABORT_OR_MANUAL"
COMPLETED = "COMPLETED"          # terminal success (not a spec state; reporting only)


@dataclass
class ExpPose:
    """Raw-map-frame position sample. Heading is supplied separately (fused)."""
    x: float
    y: float
    z: float
    stamp: float
    source_seq: int | None = None
    valid: bool = True
    match_count: int | None = None
    inlier_ratio: float | None = None
    reprojection_error: float | None = None

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class TickCommand:
    roll: int
    pitch: int
    yaw: int
    gaz: int
    mode: str
    info: dict = field(default_factory=dict)

    @property
    def pcmd(self):
        return (self.roll, self.pitch, self.yaw, self.gaz)


def clamp_pcmd(roll: float, pitch: float, yaw: float, gaz: float) -> tuple[int, int, int, int]:
    """Final wire clamp to the PCMD percent domain [-100, 100]."""
    c = lambda v: int(round(max(-100.0, min(100.0, float(v)))))
    return c(roll), c(pitch), c(yaw), c(gaz)


def speed_schedule(kind: str, yaw_err: float, threshold_deg: float = 25.0,
                   full_suppress_deg: float = 90.0) -> float:
    """Forward-speed scale as a function of |yaw error| (spec section 8).
    Compared kinds: cos | threshold | smoothstep."""
    e = abs(wrap_angle(float(yaw_err)))
    if kind == "cos":
        return max(0.0, math.cos(e))
    if kind == "threshold":
        return 1.0 if e < math.radians(threshold_deg) else 0.0
    if kind == "smoothstep":
        x = 1.0 - min(1.0, e / math.radians(full_suppress_deg))
        return x * x * (3.0 - 2.0 * x)
    raise ValueError(f"unknown speed schedule {kind!r}")


@dataclass
class CtrlParams:
    """All tunables. Distance fields are route/model units, not physical meters."""
    # PCMD caps (percent) -- conservative, never raised for real flight here
    max_pitch: int = 10
    max_roll: int = 6
    max_yaw: int = 25
    max_gaz: int = 15
    k_yaw: float = 1.4
    # translational (yaw-locked) mode: the normalized 3-D command resultant is
    # capped here, so diagonal modes do not stack full per-axis commands.
    # Simulator-only; not a real-flight speed limit.
    max_horiz_translate: int = 10
    translate_min_speed_frac: float = 0.25
    translate_deadband: float = 0.12    # route/map units; don't push horizontal inside this
    # lookahead (route/map units)
    base_lookahead: float = 0.8
    min_lookahead: float = 0.5
    max_lookahead: float = 3.0
    k_error: float = 0.5
    use_adaptive_lookahead: bool = True
    # corridor thresholds (route/map units)
    horizontal_deadband: float = 0.15
    horizontal_correction_threshold: float = 0.8   # == exit_follow_error
    horizontal_hard_abort_threshold: float = 6.0
    route_tube_radius: float | None = None         # full-3D route tube; route/map units
    route_tube_segment_window: int = 1             # active segment +/- this many segments
    route_tube_exit_s: float = 0.3                 # continuous outside-tube time before abort
    route_tube_exit_updates: int = 2               # effective pose updates outside tube before abort
    route_tube_initial_grace_s: float = 10.0       # allow initial rejoin before tube is armed
    enter_follow_error: float = 0.5
    follow_hold_s: float = 1.0
    use_hysteresis: bool = True
    # vertical (route/map units)
    vertical_deadband: float = 0.25
    vertical_norm: float = 1.0          # route/map-unit error that maps to full gaz
    vertical_priority_threshold: float = 1.0
    vertical_priority_pitch_scale: float = 0.4
    vertical_abort_threshold: float = 4.0
    k_vert: float = 1.0
    # arrival / segment switching
    arrival_radius: float = 0.5
    arrival_vertical_radius: float = 0.6
    segment_switch_progress: float = 0.95
    min_segment_time_s: float = 1.0
    max_segment_time_s: float = 90.0
    waypoint_hover_s: float = 1.0
    # align state
    body_yaw_align_at_waypoint: bool = True
    segment_align_yaw_tolerance_deg: float = 12.0
    segment_align_timeout_s: float = 6.0
    # rejoin
    rejoin_mode: str = "lookahead"      # "nearest" | "lookahead"
    rejoin_speed_scale: float = 0.7
    # horizontal control mode
    horizontal_control_mode: str = "nose_first"  # nose_first|small_lateral_assist|hybrid
    lateral_assist_threshold: float = 0.6        # only assist when |e_lat| below (m)
    max_lateral_roll_percent: int = 6
    k_lateral: float = 1.2
    # speed scheduling
    speed_schedule: str = "cos"          # cos | threshold | smoothstep
    speed_schedule_threshold_deg: float = 25.0
    slow_radius: float = 1.5             # decelerate when carrot closer than this
    min_speed_frac: float = 0.25
    # anti-oscillation
    use_anti_oscillation: bool = True
    osc_window_s: float = 4.0
    osc_flip_limit: int = 6              # sign flips in window before damping
    osc_min_yaw_cmd: int = 4             # ignore tiny yaw cmds when counting flips
    osc_cooldown_s: float = 3.0
    osc_lookahead_boost: float = 1.5
    osc_yaw_gain_scale: float = 0.7
    osc_pitch_scale: float = 0.7
    yaw_slew_rate_pct_per_s: float = 200.0   # cap yaw-command change rate (anti-jitter)
    # smoothed carrot target + telemetry low-pass (noise rejection)
    use_smoothed_target: bool = False
    target_filter_alpha: float = 0.35    # EMA per tick
    max_target_speed: float = 2.5        # route/map units per second carrot displacement limit
    pose_filter_alpha: float = 0.30      # EMA on the pose used for CONTROL (not eval)
    heading_filter_alpha: float = 0.30   # EMA on the heading used for CONTROL
    # pose staleness / lost handling
    max_pose_age_s: float = 0.6
    pose_loss_short_s: float = 1.0
    lost_abort_s: float = 8.0
    hloc_min_match_count: int | None = None
    hloc_min_inlier_ratio: float | None = None
    hloc_max_reprojection_error: float | None = None
    hloc_max_pose_jump: float | None = None
    # hover/align yaw hold
    align_yaw_gain_scale: float = 1.0
    camera_yaw_offset_deg: float = 0.0
    pcmd_rate_limit_pct_per_s: float | None = 200.0
    # final landing gate: stricter than ordinary waypoint arrival
    final_landing_radius: float = 0.5
    final_landing_hold_s: float = 1.0
    final_landing_max_est_speed: float = 0.8
    final_landing_yaw_stable_deg: float = 20.0


# ---------------------------------------------------------------------------
# Shared low-level mapping helpers

def _body_axes(heading: float):
    fwd = np.array([math.cos(heading), math.sin(heading)], dtype=float)
    right = np.array([-math.sin(heading), math.cos(heading)], dtype=float)
    return fwd, right


def _vertical_gaz(vert_err_y: float, p: CtrlParams) -> float:
    """vert_err_y = pose.y - target_y (raw frame; +y is DOWN, so err>0 means
    the drone is BELOW the profile -> climb -> positive gaz)."""
    if abs(vert_err_y) < p.vertical_deadband:
        return 0.0
    frac = max(-1.0, min(1.0, p.k_vert * vert_err_y / max(1e-9, p.vertical_norm)))
    return frac * p.max_gaz


def _camera_yaw_offset(p: CtrlParams) -> float:
    return math.radians(float(p.camera_yaw_offset_deg))


def _pose_update_key(pose: ExpPose) -> tuple:
    return ("seq", pose.source_seq) if pose.source_seq is not None else ("stamp", pose.stamp)


def _tube_info(route: RouteModel, pos: np.ndarray, p: CtrlParams,
               active_seg: int | None = None) -> dict:
    if p.route_tube_radius is None:
        return {}
    tube = route.project_tube(pos, active_seg, p.route_tube_segment_window)
    return {
        "tube_distance": tube.distance,
        "tube_radius": p.route_tube_radius,
        "tube_seg_index": tube.seg_index,
        "tube_nearest_point": tube.point.tolist(),
        "tube_active_seg": active_seg,
        "tube_segment_window": p.route_tube_segment_window,
    }


def _outside_route_tube(info: dict) -> bool:
    radius = info.get("tube_radius")
    return radius is not None and info.get("tube_distance", 0.0) > radius


def _route_tube_should_abort(ctrl, info: dict, p: CtrlParams, pose: ExpPose, now: float) -> bool:
    if p.route_tube_radius is None:
        ctrl._tube_exit_started_t = None
        ctrl._tube_exit_update_count = 0
        return False
    outside = _outside_route_tube(info)
    if not outside:
        ctrl._tube_entered_once = True
        ctrl._tube_exit_started_t = None
        ctrl._tube_exit_update_count = 0
        info.update({"tube_safety_active": True, "tube_exit_s": 0.0,
                     "tube_exit_updates": 0, "tube_exit_update_limit": p.route_tube_exit_updates})
        return False

    if not getattr(ctrl, "_tube_entered_once", False):
        if getattr(ctrl, "_tube_grace_started_t", None) is None:
            ctrl._tube_grace_started_t = now
        elapsed = now - ctrl._tube_grace_started_t
        if elapsed < p.route_tube_initial_grace_s:
            info.update({"tube_safety_active": False, "tube_grace_s_remaining": p.route_tube_initial_grace_s - elapsed,
                         "tube_exit_s": 0.0, "tube_exit_updates": 0,
                         "tube_exit_update_limit": p.route_tube_exit_updates})
            return False

    key = _pose_update_key(pose)
    if ctrl._tube_exit_started_t is None:
        ctrl._tube_exit_started_t = now
        ctrl._tube_exit_update_count = 0
        ctrl._tube_last_update_key = None
    if key != ctrl._tube_last_update_key:
        ctrl._tube_exit_update_count += 1
        ctrl._tube_last_update_key = key
    elapsed = now - ctrl._tube_exit_started_t
    info.update({"tube_safety_active": True, "tube_exit_s": elapsed,
                 "tube_exit_updates": ctrl._tube_exit_update_count,
                 "tube_exit_update_limit": p.route_tube_exit_updates})
    return (elapsed >= p.route_tube_exit_s and
            ctrl._tube_exit_update_count >= max(1, int(p.route_tube_exit_updates)))


def _pose_quality_issue(ctrl, pose: ExpPose, p: CtrlParams, now: float) -> str | None:
    if not getattr(pose, "valid", True):
        return "pose_invalid"
    if p.hloc_min_match_count is not None and pose.match_count is not None \
            and pose.match_count < p.hloc_min_match_count:
        return "low_match_count"
    if p.hloc_min_inlier_ratio is not None and pose.inlier_ratio is not None \
            and pose.inlier_ratio < p.hloc_min_inlier_ratio:
        return "low_inlier_ratio"
    if p.hloc_max_reprojection_error is not None and pose.reprojection_error is not None \
            and pose.reprojection_error > p.hloc_max_reprojection_error:
        return "high_reprojection_error"
    key = _pose_update_key(pose)
    last_key = getattr(ctrl, "_last_good_pose_key", None)
    if p.hloc_max_pose_jump is not None and key != last_key \
            and getattr(ctrl, "_last_good_pose_xyz", None) is not None:
        jump = float(np.linalg.norm(pose.xyz - ctrl._last_good_pose_xyz))
        if jump > p.hloc_max_pose_jump:
            return "pose_jump"
    ctrl._last_good_pose_xyz = pose.xyz.copy()
    ctrl._last_good_pose_key = key
    return None


def _rate_limit_pcmd(ctrl, roll: int, pitch: int, yaw: int, gaz: int,
                     now: float, p: CtrlParams) -> tuple[int, int, int, int]:
    if p.pcmd_rate_limit_pct_per_s is None:
        ctrl._last_pcmd = (roll, pitch, yaw, gaz)
        ctrl._last_pcmd_t = now
        return roll, pitch, yaw, gaz
    cur = [int(roll), int(pitch), int(yaw), int(gaz)]
    last = getattr(ctrl, "_last_pcmd", None)
    last_t = getattr(ctrl, "_last_pcmd_t", None)
    if last is None or last_t is None:
        ctrl._last_pcmd = tuple(cur)
        ctrl._last_pcmd_t = now
        return tuple(cur)
    max_step = max(0.0, now - last_t) * float(p.pcmd_rate_limit_pct_per_s)
    out = []
    for prev, val in zip(last, cur):
        lo, hi = prev - max_step, prev + max_step
        out.append(int(round(max(lo, min(hi, val)))))
    ctrl._last_pcmd = tuple(out)
    ctrl._last_pcmd_t = now
    return tuple(out)


def _init_runtime_safety(ctrl) -> None:
    ctrl._tube_exit_started_t = None
    ctrl._tube_exit_update_count = 0
    ctrl._tube_last_update_key = None
    ctrl._tube_entered_once = False
    ctrl._tube_grace_started_t = None
    ctrl._last_good_pose_xyz = None
    ctrl._last_good_pose_key = None
    ctrl._last_pcmd = None
    ctrl._last_pcmd_t = None


def _pose_gate_or_lost(ctrl, pose: ExpPose | None, heading: float | None,
                       now: float) -> TickCommand | None:
    p = ctrl.p
    if pose is None:
        return _lost_tick(ctrl, now, "pose_missing")
    pose_age = now - pose.stamp
    if pose_age > p.max_pose_age_s:
        return _lost_tick(ctrl, now, "pose_stale", pose_age)
    if heading is None:
        return _lost_tick(ctrl, now, "heading_missing")
    issue = _pose_quality_issue(ctrl, pose, p, now)
    if issue:
        return _lost_tick(ctrl, now, issue)
    return None


# ---------------------------------------------------------------------------
# 1. naive waypoint-to-waypoint baseline

class NaiveWaypointController:
    """Fly the nose straight at the next waypoint; switch on arrival radius.
    No projection, no corridor, no rejoin logic (deliberately naive)."""

    name = "naive_waypoint"

    def __init__(self, route: RouteModel, params: CtrlParams):
        self.route = route
        self.p = params
        self.k = 1                      # target waypoint index
        self.mode = INIT_REJOIN
        self._lost_since = None
        self.abort_reason = None
        _init_runtime_safety(self)

    def step(self, pose: ExpPose | None, heading: float | None, now: float) -> TickCommand:
        p = self.p
        gated = _pose_gate_or_lost(self, pose, heading, now)
        if gated is not None:
            return gated
        self._lost_since = None
        if self.mode in (ABORT_OR_MANUAL, COMPLETED):
            return TickCommand(0, 0, 0, 0, self.mode, {"target": None, "abort_reason": self.abort_reason})

        tgt = self.route.wp[self.k]
        tube = _tube_info(self.route, pose.xyz, p, max(0, self.k - 1))
        if _route_tube_should_abort(self, tube, p, pose, now):
            self.abort_reason = "route_tube_exit"
            self.mode = ABORT_OR_MANUAL
            return TickCommand(0, 0, 0, 0, self.mode, {**tube, "abort_reason": self.abort_reason})
        d_h = float(np.linalg.norm(horiz(tgt) - horiz(pose.xyz)))
        vert_err = float(pose.y - tgt[1])
        if d_h < p.arrival_radius and abs(vert_err) < p.arrival_vertical_radius:
            if self.k >= len(self.route.wp) - 1:
                self.mode = COMPLETED
                return TickCommand(0, 0, 0, 0, self.mode, {"target": tgt.tolist()})
            self.k += 1
            tgt = self.route.wp[self.k]
            d_h = float(np.linalg.norm(horiz(tgt) - horiz(pose.xyz)))
            vert_err = float(pose.y - tgt[1])
        self.mode = SEGMENT_FOLLOW

        heading_target = heading_of(tgt - pose.xyz)
        yaw_err = wrap_angle(heading_target - heading)
        yaw_cmd = max(-1.0, min(1.0, p.k_yaw * yaw_err)) * p.max_yaw
        sched = speed_schedule(p.speed_schedule, yaw_err, p.speed_schedule_threshold_deg)
        speed_frac = max(p.min_speed_frac, min(1.0, d_h / p.slow_radius))
        pitch = p.max_pitch * sched * speed_frac
        gaz = _vertical_gaz(vert_err, p)
        r, pi, y, g = clamp_pcmd(0, pitch, yaw_cmd, gaz)
        r, pi, y, g = _rate_limit_pcmd(self, r, pi, y, g, now, p)
        proj = self.route.project(pose.xyz)
        return TickCommand(r, pi, y, g, self.mode, {
            "algorithm": self.name, "target": tgt.tolist(), "target_wp": self.k,
            "heading_target": heading_target, "yaw_err": yaw_err,
            "cross_track": proj.cross_track, "s": proj.s, "seg_index": proj.seg_index,
            "vert_err": vert_err, "target_y": float(tgt[1]),
            "lookahead": 0.0, "lateral_assist": False, "pitch_suppressed": None,
            **tube,
        })


# ---------------------------------------------------------------------------
# 2. production-style continuous path following (mirror of
#    real_path_follow_controller.RouteAutoController semantics)

class ContinuousPathController:
    """Full-polyline FOLLOW with fixed lookahead; REJOIN aims at the moving
    nearest point when cross-track > rejoin_tol. Mirrors the production
    RouteAutoController step + command_to_body_percent mapping (rejoin_tol
    0.45, fixed lookahead, cos facing), with the experiment's conservative
    PCMD caps. This is the "chases the nearest point" baseline."""

    name = "continuous_path"
    REJOIN_TOL = 0.45
    FIXED_LOOKAHEAD = 0.8

    def __init__(self, route: RouteModel, params: CtrlParams):
        self.route = route
        self.p = params
        self.mode = INIT_REJOIN
        self._lost_since = None
        self.abort_reason = None
        _init_runtime_safety(self)

    def step(self, pose: ExpPose | None, heading: float | None, now: float) -> TickCommand:
        p = self.p
        gated = _pose_gate_or_lost(self, pose, heading, now)
        if gated is not None:
            return gated
        self._lost_since = None
        if self.mode in (ABORT_OR_MANUAL, COMPLETED):
            return TickCommand(0, 0, 0, 0, self.mode, {"abort_reason": self.abort_reason})

        proj = self.route.project(pose.xyz)
        tube = _tube_info(self.route, pose.xyz, p, proj.seg_index)
        if _route_tube_should_abort(self, tube, p, pose, now):
            self.abort_reason = "route_tube_exit"
            self.mode = ABORT_OR_MANUAL
            return TickCommand(0, 0, 0, 0, self.mode, {**tube, "abort_reason": self.abort_reason})
        remaining = self.route.length - proj.s
        if remaining <= max(0.2, min(1.0, self.route.length * 0.02)) \
                and proj.cross_track <= self.REJOIN_TOL:
            self.mode = COMPLETED
            return TickCommand(0, 0, 0, 0, self.mode, {"cross_track": proj.cross_track,
                                                       "s": proj.s})
        if proj.cross_track > self.REJOIN_TOL:
            self.mode = SEGMENT_REJOIN
            goal, gi = proj.point, proj.seg_index      # nearest point (moving!)
        else:
            self.mode = SEGMENT_FOLLOW
            goal, gi = self.route.point_at_s(proj.s + self.FIXED_LOOKAHEAD)

        heading_target = heading_of(goal - pose.xyz) \
            if np.linalg.norm(horiz(goal) - horiz(pose.xyz)) > 1e-6 else heading
        yaw_err = wrap_angle(heading_target - heading)
        yaw_cmd = max(-1.0, min(1.0, p.k_yaw * yaw_err)) * p.max_yaw
        facing = max(0.0, math.cos(yaw_err))           # production cos schedule
        d_h = float(np.linalg.norm(horiz(goal) - horiz(pose.xyz)))
        speed_frac = max(p.min_speed_frac, min(1.0, d_h / p.slow_radius))
        pitch = p.max_pitch * facing * speed_frac
        vert_err = float(pose.y - proj.target_y)
        gaz = _vertical_gaz(vert_err, p)
        r, pi, y, g = clamp_pcmd(0, pitch, yaw_cmd, gaz)
        r, pi, y, g = _rate_limit_pcmd(self, r, pi, y, g, now, p)
        return TickCommand(r, pi, y, g, self.mode, {
            "algorithm": self.name, "target": np.asarray(goal, float).tolist(),
            "heading_target": heading_target, "yaw_err": yaw_err,
            "cross_track": proj.cross_track, "s": proj.s, "seg_index": proj.seg_index,
            "vert_err": vert_err, "target_y": proj.target_y,
            "lookahead": 0.0 if self.mode == SEGMENT_REJOIN else self.FIXED_LOOKAHEAD,
            "lateral_assist": False, "pitch_suppressed": None,
            "nearest_point": proj.point.tolist(),
            **tube,
        })


# ---------------------------------------------------------------------------
# 3-8. segment-corridor state machine (feature-flagged)

def _lost_tick(ctrl, now: float, reason: str = "stale_or_missing",
               pose_age_s: float | None = None) -> TickCommand:
    """Shared stale/missing-pose handling: zero PCMD; abort if lost too long."""
    ctrl._tube_exit_started_t = None
    ctrl._tube_exit_update_count = 0
    if ctrl._lost_since is None:
        ctrl._lost_since = now
    lost_duration = now - ctrl._lost_since
    age_for_stage = pose_age_s if pose_age_s is not None else lost_duration
    if age_for_stage > ctrl.p.lost_abort_s and ctrl.mode != ABORT_OR_MANUAL:
        ctrl.mode = ABORT_OR_MANUAL
        return TickCommand(0, 0, 0, 0, ABORT_OR_MANUAL,
                           {"abort_reason": "telemetry_lost", "pose_quality": reason,
                            "pose_loss_stage": "long_manual_handoff",
                            "pose_age_s": pose_age_s, "lost_duration_s": lost_duration})
    if ctrl.mode not in (ABORT_OR_MANUAL, COMPLETED):
        ctrl.mode = LOST_OR_UNCERTAIN
    stage = ("short_hover_hold" if age_for_stage <= ctrl.p.pose_loss_short_s
             else "medium_wait_relocalize")
    return TickCommand(0, 0, 0, 0, ctrl.mode,
                       {"telemetry": "stale_or_missing", "pose_quality": reason,
                        "pose_loss_stage": stage,
                        "pose_age_s": pose_age_s, "lost_duration_s": lost_duration})


class SegmentCorridorController:
    """Waypoint-segment corridor tracker with the spec's state machine.

    Feature flags (CtrlParams) select the algorithm variant:
      use_adaptive_lookahead, use_hysteresis, use_anti_oscillation,
      use_smoothed_target, body_yaw_align_at_waypoint + waypoint_hover_s,
      rejoin_mode (nearest|lookahead), horizontal_control_mode.
    """

    name = "segment_corridor"

    def __init__(self, route: RouteModel, params: CtrlParams, name: str | None = None):
        self.route = route
        self.p = params
        if name:
            self.name = name
        self.mode = INIT_REJOIN
        self.seg = 0                     # active segment index
        self._seg_entered_t = None
        self._mode_entered_t = None
        self._follow_ok_since = None     # hysteresis: err below enter threshold since
        self._hover_until = None
        self._lost_since = None
        self._filt_target = None         # smoothed carrot
        self._last_tick_t = None
        self._yaw_cmd_hist: list[tuple[float, int]] = []   # (t, yaw_cmd)
        self._osc_active_until = 0.0
        self._transitions = 0            # REJOIN<->FOLLOW count (for logging)
        self.abort_reason = None
        self._pose_filt = None           # EMA pose (control-only, noise rejection)
        self._head_filt = None           # EMA heading (control-only)
        self._last_yaw_cmd = 0.0         # for yaw-command slew limiting
        self._last_yaw_now = None
        _init_runtime_safety(self)

    # -- bookkeeping ---------------------------------------------------------

    def _set_mode(self, mode: str, now: float):
        if mode != self.mode:
            prev = self.mode
            self.mode = mode
            self._mode_entered_t = now
            if {prev, mode} == {SEGMENT_FOLLOW, SEGMENT_REJOIN}:
                self._transitions += 1

    def _filter_pose(self, pos: np.ndarray, now: float) -> np.ndarray:
        """Low-pass only the CONTROL pose for the smoothed variant.

        Evaluation uses the unfiltered reference supplied by the harness: clean
        plant state for kinematic runs, firmware-fused telemetry for Sphinx.
        """
        if not self.p.use_smoothed_target:
            return pos
        if self._pose_filt is None:
            self._pose_filt = np.asarray(pos, float).copy()
        else:
            a = self.p.pose_filter_alpha
            self._pose_filt = self._pose_filt + a * (np.asarray(pos, float) - self._pose_filt)
        return self._pose_filt.copy()

    def _filter_heading(self, heading: float, now: float) -> float:
        """Wrapped EMA on the control heading (rejects motion-offset jitter)."""
        if not self.p.use_smoothed_target or heading is None:
            return heading
        if self._head_filt is None:
            self._head_filt = float(heading)
        else:
            a = self.p.heading_filter_alpha
            self._head_filt = wrap_angle(self._head_filt + a * wrap_angle(heading - self._head_filt))
        return self._head_filt

    def _slew_yaw(self, yaw_cmd: float, now: float) -> float:
        """Rate-limit the yaw command (spec anti-oscillation: 'limit yaw command
        rate'). Bounds tick-to-tick change so noisy heading can't produce large
        yaw swings; a long gap (after hover) allows a full re-aim."""
        if not self.p.use_anti_oscillation:
            self._last_yaw_cmd = yaw_cmd
            self._last_yaw_now = now
            return yaw_cmd
        if self._last_yaw_now is not None:
            dt = now - self._last_yaw_now
            max_step = self.p.yaw_slew_rate_pct_per_s * max(0.0, dt)
            yaw_cmd = max(self._last_yaw_cmd - max_step,
                          min(self._last_yaw_cmd + max_step, yaw_cmd))
        self._last_yaw_cmd = yaw_cmd
        self._last_yaw_now = now
        return yaw_cmd

    def _osc_update(self, now: float, yaw_cmd: int):
        p = self.p
        if not p.use_anti_oscillation:
            return False
        h = self._yaw_cmd_hist
        h.append((now, yaw_cmd))
        cutoff = now - p.osc_window_s
        while h and h[0][0] < cutoff:
            h.pop(0)
        flips, prev = 0, 0
        for _, c in h:
            if abs(c) < p.osc_min_yaw_cmd:
                continue
            s = 1 if c > 0 else -1
            if prev and s != prev:
                flips += 1
            prev = s
        if flips >= p.osc_flip_limit:
            self._osc_active_until = now + p.osc_cooldown_s
        return now < self._osc_active_until

    # -- main step ------------------------------------------------------------

    def step(self, pose: ExpPose | None, heading: float | None, now: float) -> TickCommand:
        p = self.p
        if self._seg_entered_t is None:
            self._seg_entered_t = now
            self._mode_entered_t = now
        gated = _pose_gate_or_lost(self, pose, heading, now)
        if gated is not None:
            return gated
        self._lost_since = None
        if self.mode in (ABORT_OR_MANUAL, COMPLETED):
            return TickCommand(0, 0, 0, 0, self.mode, {"abort_reason": self.abort_reason})

        pos = self._filter_pose(pose.xyz, now)      # control-only low-pass (eval uses truth)
        heading = self._filter_heading(heading, now)
        # returning from LOST: resume the pre-lost segment in REJOIN posture
        if self.mode == LOST_OR_UNCERTAIN:
            self._set_mode(SEGMENT_REJOIN, now)

        # INIT: pick the active segment from the full-polyline projection once
        if self.mode == INIT_REJOIN and self._last_tick_t is None:
            self.seg = self.route.project(pos).seg_index
            self._seg_entered_t = now
        self._last_tick_t = now

        proj_full = self.route.project(pos)
        if self.mode in (INIT_REJOIN, SEGMENT_REJOIN) and proj_full.seg_index > self.seg:
            # rejoin catch-up: while off-route the drone may cross segment
            # boundaries; adopt the route-forward segment so the carrot (and
            # the active-segment cross-track) never falls behind the drone.
            self.seg = proj_full.seg_index
            self._seg_entered_t = now
        proj_act = self.route.project_active(pos, self.seg)
        tube = _tube_info(self.route, pos, p, self.seg)
        err_h = proj_act.cross_track
        info: dict = {
            "algorithm": self.name, "seg_index": self.seg,
            "seg_a": self.route.wp[self.seg].tolist(),
            "seg_b": self.route.wp[self.seg + 1].tolist(),
            "cross_track": err_h, "cross_track_full": proj_full.cross_track,
            "s": proj_act.s, "seg_progress": proj_act.t,
            "nearest_point": proj_act.point.tolist(),
            "route_progress": proj_act.s / max(1e-9, self.route.length),
            "transitions": self._transitions,
            **tube,
        }

        # ---- hard aborts (simulator-only safety policy) ----
        if _route_tube_should_abort(self, info, p, pose, now):
            self.abort_reason = "route_tube_exit"
            self._set_mode(ABORT_OR_MANUAL, now)
            return TickCommand(0, 0, 0, 0, self.mode, {**info, "abort_reason": self.abort_reason})
        if proj_full.cross_track >= p.horizontal_hard_abort_threshold:
            self.abort_reason = "horizontal_hard_abort"
            self._set_mode(ABORT_OR_MANUAL, now)
            return TickCommand(0, 0, 0, 0, self.mode, {**info, "abort_reason": self.abort_reason})

        # ---- vertical profile ----
        vert_err = float(pose.y - proj_act.target_y)     # +: below profile -> climb
        info["target_y"] = proj_act.target_y
        info["vert_err"] = vert_err
        if abs(vert_err) >= p.vertical_abort_threshold:
            self.abort_reason = "vertical_profile_failure"
            self._set_mode(ABORT_OR_MANUAL, now)
            return TickCommand(0, 0, 0, 0, self.mode, {**info, "abort_reason": self.abort_reason})

        # ---- arrival / segment switch ----
        end_wp = self.route.wp[self.seg + 1]
        d_end = float(np.linalg.norm(horiz(end_wp) - horiz(pos)))
        vert_err_wp = float(pose.y - end_wp[1])
        seg_time = now - self._seg_entered_t
        # progress-based switching only counts inside the corridor: a clamped
        # projection (t==1) far off-path must not "arrive" at the waypoint.
        progress_switch = (proj_act.t >= p.segment_switch_progress
                           and err_h < p.horizontal_correction_threshold)
        arrived = ((d_end < p.arrival_radius or progress_switch)
                   and abs(vert_err_wp) < p.arrival_vertical_radius
                   and seg_time >= p.min_segment_time_s
                   and self.mode in (INIT_REJOIN, SEGMENT_FOLLOW, SEGMENT_REJOIN))
        forced = (seg_time > p.max_segment_time_s
                  and self.mode in (INIT_REJOIN, SEGMENT_FOLLOW, SEGMENT_REJOIN))
        if arrived or forced:
            info["segment_switch"] = "forced_timeout" if forced else "arrived"
            info["switch_d_end"] = d_end
            info["switch_progress"] = proj_act.t
            if self.seg >= self.route.n_segments - 1:
                self._set_mode(COMPLETED, now)
                return TickCommand(0, 0, 0, 0, self.mode, info)
            if p.waypoint_hover_s > 0.0:
                self._hover_until = now + p.waypoint_hover_s
                self._set_mode(WAYPOINT_HOVER, now)
            else:
                self._advance_segment(now, info)

        # ---- states with fixed commands ----
        if self.mode == WAYPOINT_HOVER:
            if now >= (self._hover_until or now):
                self._advance_segment(now, info)     # emits NEXT_SEGMENT this tick
            else:
                return TickCommand(0, 0, 0, 0, WAYPOINT_HOVER, info)

        if self.mode == NEXT_SEGMENT:
            # one transient tick, then align or follow
            nxt = SEGMENT_ALIGN if p.body_yaw_align_at_waypoint else SEGMENT_FOLLOW
            self._set_mode(nxt, now)
            return TickCommand(0, 0, 0, 0, NEXT_SEGMENT, info)

        if self.mode == SEGMENT_ALIGN:
            seg_head = self.route.segment_heading(self.seg)
            body_target = wrap_angle(seg_head - _camera_yaw_offset(p))
            camera_heading = wrap_angle(heading + _camera_yaw_offset(p))
            yaw_err = wrap_angle(seg_head - camera_heading)
            if abs(yaw_err) <= math.radians(p.segment_align_yaw_tolerance_deg) or \
                    (now - self._mode_entered_t) > p.segment_align_timeout_s:
                self._set_mode(SEGMENT_FOLLOW, now)
            else:
                yaw_cmd = max(-1.0, min(1.0, p.k_yaw * p.align_yaw_gain_scale * yaw_err)) * p.max_yaw
                yaw_cmd = self._slew_yaw(yaw_cmd, now)
                gaz = _vertical_gaz(vert_err, p)
                r, pi, y, g = clamp_pcmd(0, 0, yaw_cmd, gaz)
                r, pi, y, g = _rate_limit_pcmd(self, r, pi, y, g, now, p)
                info.update({"heading_target": body_target, "camera_heading_target": seg_head,
                             "camera_yaw_offset_deg": p.camera_yaw_offset_deg,
                             "yaw_err": yaw_err, "pitch_suppressed": "align"})
                return TickCommand(r, pi, y, g, SEGMENT_ALIGN, info)

        # ---- FOLLOW / REJOIN corridor policy ----
        mode = self.mode
        if mode in (INIT_REJOIN, SEGMENT_REJOIN):
            if p.use_hysteresis:
                if err_h < p.enter_follow_error:
                    if self._follow_ok_since is None:
                        self._follow_ok_since = now
                    if now - self._follow_ok_since >= p.follow_hold_s:
                        self._set_mode(SEGMENT_FOLLOW, now)
                        mode = SEGMENT_FOLLOW
                else:
                    self._follow_ok_since = None
            else:
                if err_h < p.horizontal_correction_threshold:
                    self._set_mode(SEGMENT_FOLLOW, now)
                    mode = SEGMENT_FOLLOW
        elif mode == SEGMENT_FOLLOW:
            self._follow_ok_since = None
            if err_h >= p.horizontal_correction_threshold:
                self._set_mode(SEGMENT_REJOIN, now)
                mode = SEGMENT_REJOIN

        rejoining = mode in (INIT_REJOIN, SEGMENT_REJOIN)

        # ---- carrot target ----
        osc_damped = now < self._osc_active_until
        L = (adaptive_lookahead(err_h, p.base_lookahead, p.k_error,
                                p.min_lookahead, p.max_lookahead)
             if p.use_adaptive_lookahead else p.base_lookahead)
        if osc_damped:
            L = min(p.max_lookahead, L * p.osc_lookahead_boost)
        if rejoining and p.rejoin_mode == "nearest":
            target, _ = proj_act.point, self.seg          # chases nearest point
            L_used = 0.0
        else:
            target, _gi = self.route.point_at_s(proj_act.s + L)
            L_used = L
        if p.use_smoothed_target:
            target = self._smooth_target(target, now)
        info.update({"lookahead": L_used, "rejoin_target": np.asarray(target, float).tolist(),
                     "osc_damped": osc_damped})

        # ---- nose-first command ----
        delta_h = horiz(target) - horiz(pos)
        heading_target = heading_of(np.array([delta_h[0], 0.0, delta_h[1]])) \
            if np.linalg.norm(delta_h) > 1e-6 else heading
        yaw_err = wrap_angle(heading_target - heading)
        yaw_gain = p.k_yaw * (p.osc_yaw_gain_scale if osc_damped else 1.0)
        yaw_cmd = max(-1.0, min(1.0, yaw_gain * yaw_err)) * p.max_yaw
        yaw_cmd = self._slew_yaw(yaw_cmd, now)

        sched = speed_schedule(p.speed_schedule, yaw_err, p.speed_schedule_threshold_deg)
        d_target = float(np.linalg.norm(delta_h))
        speed_frac = max(p.min_speed_frac, min(1.0, d_target / p.slow_radius))
        pitch = p.max_pitch * sched * speed_frac
        suppressed = None
        if rejoining:
            pitch *= p.rejoin_speed_scale
        if err_h < p.horizontal_deadband and not rejoining:
            pass                                          # deadband: carrot only
        if abs(vert_err) >= p.vertical_priority_threshold:
            pitch *= p.vertical_priority_pitch_scale      # prioritize altitude
            suppressed = "vertical_priority"
        if sched < 0.05:
            suppressed = suppressed or "yaw_error_large"
        if osc_damped:
            pitch *= p.osc_pitch_scale

        # ---- optional bounded lateral assist ----
        roll = 0.0
        lateral_used = False
        if p.horizontal_control_mode in ("small_lateral_assist", "hybrid") and not rejoining:
            _fwd, right = _body_axes(heading)
            e_lat = float(delta_h @ right)                # + = target to body right
            if abs(e_lat) <= p.lateral_assist_threshold and err_h >= p.horizontal_deadband:
                roll = max(-1.0, min(1.0, p.k_lateral * e_lat)) * \
                    min(p.max_lateral_roll_percent, p.max_roll)
                lateral_used = abs(roll) >= 0.5
        gaz = _vertical_gaz(vert_err, p)
        r, pi, y, g = clamp_pcmd(roll, pitch, yaw_cmd, gaz)
        r, pi, y, g = _rate_limit_pcmd(self, r, pi, y, g, now, p)
        osc_damped = self._osc_update(now, y) or osc_damped

        info.update({"heading_target": heading_target, "yaw_err": yaw_err,
                     "pitch_suppressed": suppressed, "lateral_assist": lateral_used,
                     "speed_schedule": p.speed_schedule, "osc_damped": osc_damped})
        return TickCommand(r, pi, y, g, mode, info)

    def _advance_segment(self, now: float, info: dict):
        self.seg += 1
        self._seg_entered_t = now
        self._filt_target = None
        self._follow_ok_since = None
        info["next_segment"] = self.seg
        self._set_mode(NEXT_SEGMENT, now)

    def _smooth_target(self, target: np.ndarray, now: float) -> np.ndarray:
        p = self.p
        t = np.asarray(target, dtype=float)
        if self._filt_target is None:
            self._filt_target = t.copy()
            self._filt_t = now
            return t
        dt = max(1e-3, now - getattr(self, "_filt_t", now))
        self._filt_t = now
        step = p.target_filter_alpha * (t - self._filt_target)
        max_step = p.max_target_speed * dt
        n = float(np.linalg.norm(step))
        if n > max_step:
            step *= max_step / n
        self._filt_target = self._filt_target + step
        return self._filt_target.copy()


# ---------------------------------------------------------------------------
# Yaw-locked translational controller (user variant): the drone NEVER yaws
# while flying a segment. It reaches the next waypoint by pure body-frame
# translation -- combining forward/back (pitch), left/right (roll) and up/down
# (gaz) velocity components -- and only rotates (yaw) while hovering AT a
# waypoint, to face the following waypoint. The body-frame direction to the
# target falls into one of 14 movement modes (6 axis faces + 8 3D corners);
# the continuous per-axis velocity components realize it.

# 14 canonical body directions (F=forward +x_body, R=right +y_body, U=up).
_S3 = 1.0 / math.sqrt(3.0)
MOVE_MODES_14 = {
    "forward": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0), "left": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0),
    "forward_up_right": (_S3, _S3, _S3), "forward_up_left": (_S3, -_S3, _S3),
    "forward_down_right": (_S3, _S3, -_S3), "forward_down_left": (_S3, -_S3, -_S3),
    "back_up_right": (-_S3, _S3, _S3), "back_up_left": (-_S3, -_S3, _S3),
    "back_down_right": (-_S3, _S3, -_S3), "back_down_left": (-_S3, -_S3, -_S3),
}


def classify_mode_14(fwd: float, right: float, up: float) -> str:
    """Nearest of the 14 movement modes (max cosine) for a body-frame direction
    (forward, right, up). 'hover' if the vector is ~zero. This is a LABEL for
    analysis; the controller applies the continuous per-axis components."""
    n = math.sqrt(fwd * fwd + right * right + up * up)
    if n < 1e-9:
        return "hover"
    f, r, u = fwd / n, right / n, up / n
    best, best_dot = None, -2.0
    for name, (cf, cr, cu) in MOVE_MODES_14.items():
        d = f * cf + r * cr + u * cu
        if d > best_dot:
            best, best_dot = name, d
    return best


def _translational_segment_pcmd(fwd: float, right: float, up: float,
                                d_to_wp: float, p: CtrlParams) -> tuple[float, float, float]:
    """Continuous body-frame translation command for a segment.

    The 14 movement modes are labels only. This function keeps the command
    continuous by normalizing the body-frame vector and scaling each axis by its
    conservative PCMD cap. The normalized 3-D resultant is <= speed_frac, so
    forward+right+up never means "full pitch + full roll + full gaz".
    """
    mag_h = math.hypot(fwd, right)
    if mag_h < p.translate_deadband:
        fwd = right = 0.0
    if abs(up) < p.vertical_deadband:
        up = 0.0

    mag = math.sqrt(fwd * fwd + right * right + up * up)
    if mag < 1e-9:
        return 0.0, 0.0, 0.0

    speed_frac = min(1.0, d_to_wp / p.slow_radius)
    if d_to_wp > 2.0 * p.arrival_radius:
        speed_frac = max(p.translate_min_speed_frac, speed_frac)

    # Unit direction in body-frame route space; axis caps convert it into PCMD.
    pitch = (fwd / mag) * min(p.max_pitch, p.max_horiz_translate) * speed_frac
    roll = (right / mag) * min(p.max_roll, p.max_horiz_translate) * speed_frac
    gaz = (up / mag) * min(p.max_gaz, p.max_horiz_translate) * speed_frac
    return roll, pitch, gaz


def _cap_int_vector_norm(roll: int, pitch: int, gaz: int, max_norm: float) -> tuple[int, int, int]:
    vals = [int(roll), int(pitch), int(gaz)]
    while math.sqrt(sum(v * v for v in vals)) > max_norm + 1e-9:
        i = max(range(3), key=lambda j: abs(vals[j]))
        if vals[i] == 0:
            break
        vals[i] -= 1 if vals[i] > 0 else -1
    return vals[0], vals[1], vals[2]


class TranslationalWaypointController:
    """Waypoint-to-waypoint, body-frame translation, yaw ONLY at waypoints.

    Before the first segment, and after every arrival, hover and yaw so the
    body/camera heading projected onto the ground plane aligns with the next
    waypoint vector projected onto the same plane.  Only after alignment is the
    vector decomposed on the drone's fixed body axes into forward/right/up PCMD
    components (pitch, roll, gaz), with yaw held at zero during translation.
    Repeat to the last waypoint, then hover for landing.
    """

    name = "translational_waypoint"
    # motion is NOT nose-first here (the drone strafes/backs up), so the run loop
    # must NOT refine the map-yaw offset from motion direction -- it would flip
    # the heading. Use the drone's actual body yaw directly.
    refine_heading_from_motion = False

    def __init__(self, route: RouteModel, params: CtrlParams, name: str | None = None):
        self.route = route
        self.p = params
        if name:
            self.name = name
        self.k = 1                       # target = SECOND waypoint (spec: connect start->P1)
        # The first leg obeys the same hover/align/translate contract as every
        # later leg.  Starting directly in SEGMENT_FOLLOW let a misaligned drone
        # translate immediately after take-off.
        self.mode = SEGMENT_ALIGN
        self._initial_alignment_pending = True
        self._initial_turn_commanded = False
        self._seg_entered_t = None
        self._mode_entered_t = None
        self._hover_until = None
        self._lost_since = None
        self._pose_filt = None
        self._landing_samples = []
        self.abort_reason = None
        _init_runtime_safety(self)

    def _adv_mode(self, mode, now):
        self.mode = mode
        self._mode_entered_t = now

    def _final_landing_ready(self, pos: np.ndarray, heading: float, now: float,
                             info: dict) -> bool:
        p = self.p
        target = self.route.wp[-1]
        d = float(np.linalg.norm(np.asarray(pos, float) - target))
        inside = d <= p.final_landing_radius
        if not inside:
            self._landing_samples = []
        else:
            self._landing_samples.append((now, np.asarray(pos, float).copy(), float(heading)))
        elapsed = 0.0
        est_speed = float("inf")
        yaw_span = 0.0
        if len(self._landing_samples) >= 2:
            t0, p0, h0 = self._landing_samples[0]
            t1, p1, _ = self._landing_samples[-1]
            elapsed = max(0.0, t1 - t0)
            est_speed = float(np.linalg.norm(p1 - p0) / max(1e-9, elapsed))
            yaw_span = max(abs(wrap_angle(s[2] - h0)) for s in self._landing_samples)
        ready = (inside and elapsed >= p.final_landing_hold_s and
                 est_speed <= p.final_landing_max_est_speed and
                 yaw_span <= math.radians(p.final_landing_yaw_stable_deg))
        info.update({
            "final_landing_ready": ready,
            "final_landing_distance": d,
            "final_landing_radius": p.final_landing_radius,
            "final_landing_hold_s": elapsed,
            "final_landing_required_hold_s": p.final_landing_hold_s,
            "final_landing_est_speed": None if not math.isfinite(est_speed) else est_speed,
            "final_landing_max_est_speed": p.final_landing_max_est_speed,
            "final_landing_yaw_span_deg": math.degrees(yaw_span),
            "final_landing_yaw_stable_deg": p.final_landing_yaw_stable_deg,
        })
        return ready

    def step(self, pose: ExpPose | None, heading: float | None, now: float) -> TickCommand:
        p = self.p
        if self._seg_entered_t is None:
            self._seg_entered_t = now
            self._mode_entered_t = now
        gated = _pose_gate_or_lost(self, pose, heading, now)
        if gated is not None:
            return gated
        self._lost_since = None
        if self.mode in (ABORT_OR_MANUAL, COMPLETED):
            return TickCommand(0, 0, 0, 0, self.mode, {"abort_reason": self.abort_reason})
        if self.mode == LOST_OR_UNCERTAIN:
            self._adv_mode(
                SEGMENT_ALIGN if self._initial_alignment_pending else SEGMENT_FOLLOW,
                now,
            )

        pos = pose.xyz
        if p.use_smoothed_target:                       # optional control-pose low-pass
            if self._pose_filt is None:
                self._pose_filt = np.asarray(pos, float).copy()
            else:
                self._pose_filt += p.pose_filter_alpha * (np.asarray(pos, float) - self._pose_filt)
            pos = self._pose_filt.copy()

        target = self.route.wp[self.k]
        proj = self.route.project(pos)                  # for reporting cross-track
        active_seg = max(0, self.k - 1)
        tube = _tube_info(self.route, pos, p, active_seg)
        d_h = float(np.linalg.norm(horiz(target) - horiz(pos)))
        vert_err = float(pos[1] - target[1])            # +: below target -> climb
        d_wp_3d = math.sqrt(d_h * d_h + vert_err * vert_err)
        fwd, right = _body_axes(heading)                # fixed body axes (no yaw in segment)
        delta_h = horiz(target) - horiz(pos)
        f_comp = float(delta_h @ fwd)
        r_comp = float(delta_h @ right)
        up_comp = -(target[1] - pos[1])                 # + = need to climb
        mode14 = classify_mode_14(f_comp, r_comp, up_comp)

        info = {
            "algorithm": self.name, "seg_index": max(0, self.k - 1),
            "target_wp": self.k, "target": target.tolist(),
            "seg_a": self.route.wp[self.k - 1].tolist(), "seg_b": target.tolist(),
            "cross_track": proj.cross_track, "cross_track_full": proj.cross_track,
            "s": proj.s, "route_progress": proj.s / max(1e-9, self.route.length),
            "target_y": float(target[1]), "vert_err": vert_err,
            "move_mode": mode14, "body_fwd": f_comp, "body_right": r_comp, "body_up": up_comp,
            "d_to_wp": d_wp_3d, "d_to_wp_horiz": d_h,
            "arrival_radius": p.arrival_radius,
            "lookahead": 0.0, "transitions": 0,
            **tube,
        }

        # hard aborts
        if _route_tube_should_abort(self, info, p, pose, now):
            self.abort_reason = "route_tube_exit"
            self._adv_mode(ABORT_OR_MANUAL, now)
            return TickCommand(0, 0, 0, 0, self.mode, {**info, "abort_reason": self.abort_reason})
        if proj.cross_track >= p.horizontal_hard_abort_threshold:
            self.abort_reason = "horizontal_hard_abort"
            self._adv_mode(ABORT_OR_MANUAL, now)
            return TickCommand(0, 0, 0, 0, self.mode, {**info, "abort_reason": self.abort_reason})
        if abs(vert_err) >= p.vertical_abort_threshold:
            self.abort_reason = "vertical_profile_failure"
            self._adv_mode(ABORT_OR_MANUAL, now)
            return TickCommand(0, 0, 0, 0, self.mode, {**info, "abort_reason": self.abort_reason})

        # ---- WAYPOINT_HOVER: settle before yawing ----
        if self.mode == WAYPOINT_HOVER:
            if now < (self._hover_until or now):
                return TickCommand(0, 0, 0, 0, WAYPOINT_HOVER, info)
            if self.k >= len(self.route.wp) - 1:
                if self._final_landing_ready(pos, heading, now, info):
                    self._adv_mode(COMPLETED, now)      # last waypoint -> hover + land
                    return TickCommand(0, 0, 0, 0, COMPLETED, {**info, "should_land": True})
                if info["final_landing_distance"] > p.final_landing_radius:
                    self._seg_entered_t = now
                    self._adv_mode(SEGMENT_FOLLOW, now)
                    return TickCommand(0, 0, 0, 0, SEGMENT_FOLLOW,
                                       {**info, "final_landing_reacquire": True})
                return TickCommand(0, 0, 0, 0, WAYPOINT_HOVER,
                                   {**info, "final_landing_pending": True})
            self._adv_mode(SEGMENT_ALIGN, now)

        # ---- SEGMENT_ALIGN: yaw ONLY, using both rays' ground-plane projection ----
        if self.mode == SEGMENT_ALIGN:
            initial_alignment = self._initial_alignment_pending
            segment_start = 0 if initial_alignment else self.k
            next_index = 1 if initial_alignment else self.k + 1
            nxt = self.route.wp[next_index]
            yaw_targets = getattr(self.route, "yaw_targets", {})
            seg_head = yaw_targets.get(
                segment_start,
                heading_of(nxt - self.route.wp[segment_start]),
            )
            body_target = wrap_angle(seg_head - _camera_yaw_offset(p))
            camera_heading = wrap_angle(heading + _camera_yaw_offset(p))
            yaw_err = wrap_angle(seg_head - camera_heading)        # camera XY vs segment XY
            info.update({"heading_target": body_target, "camera_heading_target": seg_head,
                         "camera_yaw_offset_deg": p.camera_yaw_offset_deg,
                         "yaw_err": yaw_err, "align_to_wp": next_index,
                         "pitch_suppressed": "align"})
            aligned = abs(yaw_err) <= math.radians(p.segment_align_yaw_tolerance_deg)
            timed_out = (now - self._mode_entered_t) > p.segment_align_timeout_s
            if aligned:
                if initial_alignment:
                    self._initial_alignment_pending = False
                else:
                    self.k += 1                          # advance to next segment
                self._seg_entered_t = now
                self._adv_mode(SEGMENT_FOLLOW, now)
                # At take-off, an already aligned aircraft may enter the first
                # translation in this tick. Later waypoints still emit one full
                # zero-PCMD transition tick after their hover/alignment phase.
                if not initial_alignment or self._initial_turn_commanded:
                    self._initial_turn_commanded = False
                    self._last_pcmd = (0, 0, 0, 0)
                    self._last_pcmd_t = now
                    return TickCommand(
                        0, 0, 0, 0, NEXT_SEGMENT,
                        {**info, "next_segment": self.k},
                    )
            elif timed_out:
                # Never convert a failed turn into translation. The two projected
                # rays must actually overlap within tolerance before a leg starts.
                self.abort_reason = "segment_alignment_timeout"
                self._adv_mode(ABORT_OR_MANUAL, now)
                return TickCommand(
                    0, 0, 0, 0, self.mode,
                    {**info, "abort_reason": self.abort_reason},
                )
            else:
                gaz = _vertical_gaz(vert_err, p)         # hold altitude while yawing
                yaw_cmd = max(-1.0, min(1.0, p.k_yaw * yaw_err)) * p.max_yaw
                r_, pi_, y_, g_ = clamp_pcmd(0, 0, yaw_cmd, gaz)
                r_, pi_, y_, g_ = _rate_limit_pcmd(self, r_, pi_, y_, g_, now, p)
                if initial_alignment:
                    self._initial_turn_commanded = True
                return TickCommand(r_, pi_, y_, g_, SEGMENT_ALIGN, info)

        # ---- SEGMENT_FOLLOW: body translation, NO yaw ----
        arrival_radius = p.final_landing_radius if self.k >= len(self.route.wp) - 1 else p.arrival_radius
        info["arrival_radius"] = arrival_radius
        arrived = (d_wp_3d <= arrival_radius
                   and (now - self._seg_entered_t) >= p.min_segment_time_s)
        forced = (now - self._seg_entered_t) > p.max_segment_time_s
        if arrived or forced:
            info["segment_switch"] = "forced_timeout" if forced else "arrived"
            info["switch_d_end"] = d_wp_3d
            info["switch_progress"] = 1.0
            self._hover_until = now + max(0.3, p.waypoint_hover_s)
            self._adv_mode(WAYPOINT_HOVER, now)
            return TickCommand(0, 0, 0, 0, WAYPOINT_HOVER, info)

        d_3d = math.sqrt(f_comp * f_comp + r_comp * r_comp + up_comp * up_comp)
        roll, pitch, gaz = _translational_segment_pcmd(f_comp, r_comp, up_comp, d_3d, p)
        r_, pi_, y_, g_ = clamp_pcmd(roll, pitch, 0, gaz)   # yaw = 0 (no rotation)
        r_, pi_, g_ = _cap_int_vector_norm(r_, pi_, g_, p.max_horiz_translate)
        r_, pi_, y_, g_ = _rate_limit_pcmd(self, r_, pi_, y_, g_, now, p)
        info.update({"lateral_assist": abs(r_) >= 1, "pitch_suppressed": None,
                     "heading_target": heading, "yaw_err": 0.0})
        return TickCommand(r_, pi_, y_, g_, SEGMENT_FOLLOW, info)


# ---------------------------------------------------------------------------
# Algorithm registry (spec "Algorithms to compare")

def _preset(**over) -> CtrlParams:
    base = CtrlParams(
        use_adaptive_lookahead=False, use_hysteresis=False,
        use_anti_oscillation=False, use_smoothed_target=False,
        body_yaw_align_at_waypoint=False, waypoint_hover_s=0.0,
        rejoin_mode="nearest", horizontal_control_mode="nose_first",
    )
    return replace(base, **over)


ALGORITHM_PRESETS: dict[str, dict] = {
    "naive_waypoint": {},
    "continuous_path": {},
    "segment_corridor": {},
    "segment_corridor_hover": dict(body_yaw_align_at_waypoint=True, waypoint_hover_s=1.0),
    "segment_corridor_lateral": dict(body_yaw_align_at_waypoint=True, waypoint_hover_s=1.0,
                                     horizontal_control_mode="small_lateral_assist"),
    "segment_corridor_nose_first": dict(body_yaw_align_at_waypoint=True,
                                        waypoint_hover_s=1.0, rejoin_mode="lookahead"),
    "adaptive_lookahead": dict(body_yaw_align_at_waypoint=True, waypoint_hover_s=1.0,
                               rejoin_mode="lookahead", use_adaptive_lookahead=True,
                               use_hysteresis=True, use_anti_oscillation=True),
    "adaptive_smoothed": dict(body_yaw_align_at_waypoint=True, waypoint_hover_s=1.0,
                              rejoin_mode="lookahead", use_adaptive_lookahead=True,
                              use_hysteresis=True, use_anti_oscillation=True,
                              use_smoothed_target=True),
    # yaw-locked body-translation, yaw only at waypoints (user variant)
    "translational_waypoint": dict(waypoint_hover_s=1.0, arrival_radius=0.4,
                                   arrival_vertical_radius=0.4),
    "translational_smoothed": dict(waypoint_hover_s=1.0, arrival_radius=0.4,
                                   arrival_vertical_radius=0.4, use_smoothed_target=True),
}

ALGORITHM_ORDER = list(ALGORITHM_PRESETS.keys())


def make_controller(name: str, route: RouteModel, overrides: dict | None = None):
    """Build a controller by algorithm name. `overrides` are CtrlParams fields
    (e.g. horizontal_control_mode from the CLI) applied on top of the preset."""
    if name not in ALGORITHM_PRESETS:
        raise ValueError(f"unknown algorithm {name!r}; choose from {ALGORITHM_ORDER}")
    params = _preset(**ALGORITHM_PRESETS[name])
    if overrides:
        params = replace(params, **overrides)
    if name == "naive_waypoint":
        return NaiveWaypointController(route, params)
    if name == "continuous_path":
        return ContinuousPathController(route, params)
    if name.startswith("translational"):
        return TranslationalWaypointController(route, params, name=name)
    return SegmentCorridorController(route, params, name=name)
