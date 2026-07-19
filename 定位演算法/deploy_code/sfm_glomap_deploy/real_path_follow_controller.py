#!/usr/bin/env python3
"""Real-flight path-follow controller extracted from sim_operator_dashboard.py.

Purpose
-------
This file is the deployment-facing AUTO controller logic, without the simulator
UI.  It is intended to be wired to:

  1. a visual localizer that returns the current camera/drone pose in the SAME
     GLOMAP map frame, and
  2. a drone adapter that converts the returned map-frame command into Olympe
     PCMD / velocity commands.

Current frame convention (matches the present operator dashboard)
-----------------------------------------------------------------
- `safezone/flight_path.json` and `safezone/poles.json` are Blender aligned
  coordinates: X/Y horizontal, Z up.
- The sparse reloc map and simulator run in original GLOMAP coordinates:
  horizontal = X/Z, gravity-up = -Y.
- Therefore this controller converts aligned -> GLOMAP as:

      aligned (x, y, z)  ->  glomap (x, -z, y)

If later your real localizer returns a gravity-aligned Z-up pose instead, either
remove this conversion or use T_align consistently before calling this module.

What this controller does now
-----------------------------
- Follow the drawn route as one continuous polyline.
- If HLoc/visual pose is far from the path: REJOIN the nearest point.
- Once close: FOLLOW a lookahead carrot point along the route.
- Near selected inspection waypoints (default labels 9,10,11,15), hold position
  and request a gimbal/capture acknowledgement for the nearest drawn pole box.
- No vertical up/down inspection maneuver by default.
- Landing is allowed only after every configured inspection waypoint has an
  explicit successful gimbal/capture acknowledgement.

Safety notes
------------
This is not a complete flight app by itself.  It does not arm motors or talk to
Olympe.  It outputs a Command.  A real-flight runner must:
- keep a manual override pilot ready,
- verify yaw/body-frame signs with props off first,
- set conservative speed gains,
- enforce geofence / obstacle safety with a real safety layer, not just sparse
  point-cloud proximity.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # optional; collision monitor becomes unavailable
    cKDTree = None


# ---------------------------------------------------------------------------
# Data contracts

@dataclass
class Pose:
    """Visual-localizer pose in current GLOMAP map frame.

    Current dashboard convention: x/z are horizontal, -y is up, yaw is heading in
    the X/Z plane measured by atan2(z, x).
    """
    x: float
    y: float
    z: float
    yaw: float
    stamp: float = field(default_factory=time.monotonic)

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Command:
    """Map-frame command produced by the controller.

    `vel_map` is a desired per-second velocity in current GLOMAP map units.
    `yaw_target` is the heading the camera/drone should face in the X/Z plane.
    A drone adapter can convert this to PCMD or velocity-control commands.
    """
    action: str
    vel_map: np.ndarray
    yaw_target: float
    goal: np.ndarray
    path_error: float
    progress: float
    look_at_pole: Optional[dict] = None
    should_land: bool = False
    status: str = ""


@dataclass
class ControlConfig:
    speed: float = 1.2                  # map-units/sec
    lookahead: float = 0.8              # map-units ahead on route after rejoin
    rejoin_tol: float = 0.45            # cross-track error before REJOIN
    arrive: float = 1.0                 # absolute cap for end tolerance
    arrive_fraction: float = 0.02       # end tolerance <= 2% of route length
    min_arrive: float = 0.20            # but never smaller than this map-unit value
    inspect_waypoints: tuple[int, ...] = (9, 10, 11, 15)  # 1-based labels in current 15-point route
    inspect_radius: float = 0.75        # map-unit trigger around each label
    inspect_resume_margin: float = 0.25 # arclength margin after last inspected wp
    inspect_yaw_tolerance_deg: float = 6.0
    max_pose_age_s: float = 0.5         # stale visual pose -> HOVER
    # ANAFI's inspection path uses body yaw for horizontal aim and gimbal pitch
    # for vertical aim. Translation remains zero while aligning/capturing.
    pole_body_look: bool = True

    def __post_init__(self):
        limits = {
            "speed": (0.0, 10.0),
            "lookahead": (0.0, 100.0),
            "rejoin_tol": (0.0, 100.0),
            "arrive": (0.0, 100.0),
            "arrive_fraction": (0.0, 1.0),
            "min_arrive": (0.0, 100.0),
            "inspect_radius": (0.0, 100.0),
            "inspect_resume_margin": (0.0, 100.0),
            "inspect_yaw_tolerance_deg": (0.0, 45.0),
            "max_pose_age_s": (0.0, 10.0),
        }
        for name, (lo, hi) in limits.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= lo or value > hi:
                raise ValueError(f"{name} must be finite and in ({lo}, {hi}], got {value!r}")
        if (not isinstance(self.inspect_waypoints, tuple)
                or any(not isinstance(i, int) or isinstance(i, bool) or i < 1
                       for i in self.inspect_waypoints)):
            raise ValueError("inspect_waypoints must be a tuple of positive 1-based integers")


# ---------------------------------------------------------------------------
# Frame / geometry helpers

def _finite_vec3(value, label: str) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 3-vector") from exc
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be a finite numeric 3-vector, got shape={out.shape}")
    return out


def _validated_waypoints(waypoints) -> list[np.ndarray]:
    if not isinstance(waypoints, (list, tuple)) or len(waypoints) < 2:
        raise ValueError(f"need >=2 waypoints, got {0 if waypoints is None else len(waypoints)}")
    out = [_finite_vec3(p, f"waypoint[{i}]") for i, p in enumerate(waypoints)]
    for i, (a, b) in enumerate(zip(out[:-1], out[1:])):
        if float(np.linalg.norm(b - a)) <= 1e-9:
            raise ValueError(f"route segment {i}->{i + 1} has zero length")
    return out


def aligned_to_glomap(p: Iterable[float]) -> np.ndarray:
    """Blender aligned Z-up -> current original GLOMAP frame."""
    p = _finite_vec3(p, "aligned coordinate")
    return np.array([p[0], -p[2], p[1]], dtype=float)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_from_to(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(b, float) - np.asarray(a, float)
    return float(math.atan2(d[2], d[0]))


def path_arclengths(wp: list[np.ndarray]) -> np.ndarray:
    s = [0.0]
    for a, b in zip(wp[:-1], wp[1:]):
        s.append(s[-1] + float(np.linalg.norm(b - a)))
    return np.asarray(s, dtype=float)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    l2 = float(ab @ ab)
    t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, float((p - a) @ ab) / l2))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q, t


def project_to_path(p: np.ndarray, wp: list[np.ndarray], cum: np.ndarray):
    """Return (distance, nearest_point, segment_index, arclength_s)."""
    best = None
    for i, (a, b) in enumerate(zip(wp[:-1], wp[1:])):
        d, q, t = point_segment_distance(p, a, b)
        seg_len = float(np.linalg.norm(b - a))
        s = float(cum[i] + t * seg_len)
        if best is None or d < best[0]:
            best = (d, q, i, s)
    return best


def point_at_s(wp: list[np.ndarray], cum: np.ndarray, s: float):
    """Return (point, segment_index) at path arclength s."""
    s = max(0.0, min(float(s), float(cum[-1])))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(wp) - 2))
    a, b = wp[i], wp[i + 1]
    seg_len = float(cum[i + 1] - cum[i])
    t = 0.0 if seg_len < 1e-12 else (s - float(cum[i])) / seg_len
    return a + t * (b - a), i


# ---------------------------------------------------------------------------
# Loading JSON artifacts from Blender

def load_waypoints(path_json: str | Path) -> list[np.ndarray]:
    data = json.loads(Path(path_json).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("waypoints"), list):
        raise ValueError("route JSON must contain a waypoints list")
    return _validated_waypoints([aligned_to_glomap(p) for p in data["waypoints"]])


def load_poles(poles_json: str | Path) -> list[dict]:
    p = Path(poles_json)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    poles = data.get("poles") if isinstance(data, dict) else None
    if not isinstance(poles, list):
        raise ValueError("poles JSON must contain a poles list")
    out = []
    for i, pole in enumerate(poles, start=1):
        if not isinstance(pole, dict):
            raise ValueError(f"pole[{i}] must be an object")
        center = pole.get("center")
        if center is None:
            base = _finite_vec3(pole.get("base"), f"pole[{i}].base")
            top = _finite_vec3(pole.get("top"), f"pole[{i}].top")
            center = ((base + top) * 0.5).tolist()
        center_g = aligned_to_glomap(center)
        base_g = aligned_to_glomap(pole.get("base", center))
        top_g = aligned_to_glomap(pole.get("top", center))
        out.append({
            "id": i,
            "center": center_g,
            "base": base_g,
            "top": top_g,
            "raw": pole,
        })
    return out


def nearest_pole(poles: list[dict], ref: np.ndarray):
    """Return (pole_id, pole_dict, horizontal_distance) nearest to ref in X/Z."""
    if not poles:
        return None, None, None
    ref = np.asarray(ref, float)
    best = None
    for pole in poles:
        c = np.asarray(pole["center"], float)
        d = float(np.linalg.norm(c[[0, 2]] - ref[[0, 2]]))
        if best is None or d < best[0]:
            best = (d, pole)
    d, pole = best
    return pole["id"], pole, d


# ---------------------------------------------------------------------------
# Optional sparse-cloud collision/clearance monitor.
# NOTE: NOT wired into the production flight path — RouteAutoController.step never calls it.
# Real-flight obstacle safety currently relies on the route-deviation bound + the human pilot.
# Wiring this in as a hover/land trigger is a deliberate design decision (deferred, not done).

class SparseCloudCollisionMonitor:
    """Same collision heuristic as the current simulator dashboard.

    It queries the nearest point in the sparse/colored point cloud to the drone
    position.  This is only an operator warning layer: the point cloud is not a
    watertight obstacle model and should not be the only real-flight safety check.
    """
    def __init__(self, xyz: np.ndarray, collision_radius: float = 0.06,
                 warning_radius: float = 0.20, max_points: int = 0):
        self.status = "OFF"
        self.collision_radius = float(collision_radius)
        self.warning_radius = float(warning_radius)
        self.xyz = np.asarray(xyz, dtype=np.float32)
        if max_points and len(self.xyz) > max_points:
            idx = np.linspace(0, len(self.xyz) - 1, int(max_points), dtype=np.int64)
            self.xyz = self.xyz[idx]
        self.tree = cKDTree(self.xyz) if (cKDTree is not None and len(self.xyz)) else None

    def update(self, pos: np.ndarray) -> dict:
        if self.tree is None:
            return {"status": "OFF", "distance": None, "point": None, "severity": 0.0}
        d, idx = self.tree.query(np.asarray(pos, float), k=1)
        d = float(d)
        point = self.xyz[int(idx)].astype(float).tolist() if np.isfinite(d) else None
        if d <= self.collision_radius:
            status, severity = "COLLISION", 1.0
        elif d <= self.warning_radius:
            status = "WARNING"
            denom = max(1e-6, self.warning_radius - self.collision_radius)
            severity = max(0.0, min(1.0, (self.warning_radius - d) / denom))
        else:
            status, severity = "CLEAR", 0.0
        return {"status": status, "distance": d, "point": point, "severity": severity}


# ---------------------------------------------------------------------------
# Main controller

class RouteAutoController:
    """Path-follow + look-at-pole controller for real deployment wiring."""

    def __init__(self, waypoints: list[np.ndarray], poles: list[dict] | None = None,
                 config: ControlConfig | None = None):
        self.wp = _validated_waypoints(waypoints)
        self.cum = path_arclengths(self.wp)
        self.path_len = float(self.cum[-1])
        self.cfg = config or ControlConfig()
        invalid_inspections = [i for i in self.cfg.inspect_waypoints if i > len(self.wp)]
        if invalid_inspections:
            raise ValueError(
                f"inspection waypoint labels exceed route length {len(self.wp)}: {invalid_inspections}")
        self.inspect_waypoints = {i - 1 for i in self.cfg.inspect_waypoints}
        self.poles = []
        for i, pole in enumerate(poles or []):
            if not isinstance(pole, dict):
                raise ValueError(f"pole[{i}] must be an object")
            checked = dict(pole)
            for key in ("center", "base", "top"):
                checked[key] = _finite_vec3(pole.get(key), f"pole[{i}].{key}")
            self.poles.append(checked)
        if self.inspect_waypoints and not self.poles:
            raise ValueError("inspection waypoints require at least one validated pole")
        self.completed_inspections: set[int] = set()
        self.state = "CRUISE"  # CRUISE, INSPECT, LANDING, DONE, HOVER
        self.last_inspect: Optional[dict] = None

    def final_arrive_tolerance(self) -> float:
        """End-of-route tolerance for LAND.

        The legacy fixed 1.0 map-unit value was too loose for short routes and
        could mark route completion around 90% progress. Keep 1.0 as a hard cap,
        but scale the active tolerance to the current route length.
        """
        return max(float(self.cfg.min_arrive),
                   min(float(self.cfg.arrive), self.path_len * float(self.cfg.arrive_fraction)))

    def _all_inspections_done(self) -> bool:
        valid = {i for i in self.inspect_waypoints if 0 <= i < len(self.wp)}
        return (not valid) or valid.issubset(self.completed_inspections)

    def _inspection_resume_reached(self, pos: np.ndarray) -> bool:
        if not self._all_inspections_done():
            return False
        if not self.inspect_waypoints:
            return False
        _d, _nearest, _seg, s = project_to_path(pos, self.wp, self.cum)
        last_s = max(float(self.cum[i]) for i in self.inspect_waypoints)
        return s >= last_s + self.cfg.inspect_resume_margin

    def _look_at_pole_target(self, pos: np.ndarray):
        """Return (wp_1based, pole_center, pole_id) if inside a look-at region."""
        if not self.poles or not self.inspect_waypoints:
            self.last_inspect = None
            return None
        if self._inspection_resume_reached(pos):
            self.last_inspect = None
            return None
        _path_dist, _nearest, _seg, s = project_to_path(pos, self.wp, self.cum)
        best = None
        for idx in self.inspect_waypoints:
            if idx in self.completed_inspections:
                continue
            eu = float(np.linalg.norm(pos - self.wp[idx]))
            ar = abs(float(s - self.cum[idx]))
            score = min(eu, ar)
            if best is None or score < best[0]:
                best = (score, idx, eu, ar)
        if best is None or best[0] > self.cfg.inspect_radius:
            self.last_inspect = None
            return None
        _score, idx, eu, ar = best
        pole_id, pole, pole_dist = nearest_pole(self.poles, self.wp[idx])
        if pole is None:
            self.last_inspect = None
            return None
        self.last_inspect = {
            "waypoint": idx + 1,
            "pole_id": pole_id,
            "pole_dist": pole_dist,
            "euclid_to_wp": eu,
            "arclength_to_wp": ar,
        }
        return idx + 1, np.asarray(pole["center"], float), pole_id

    def ack_inspection(self, metadata: dict | None,
                       orientation_confirmed: bool = False) -> bool:
        """Mark only the currently pending inspection complete after capture acknowledgement."""
        if (not orientation_confirmed or not isinstance(metadata, dict)
                or self.last_inspect is None):
            return False
        waypoint = metadata.get("waypoint")
        pole_id = metadata.get("pole_id")
        if (waypoint != self.last_inspect.get("waypoint")
                or pole_id != self.last_inspect.get("pole_id")):
            return False
        idx = int(waypoint) - 1
        if idx not in self.inspect_waypoints:
            return False
        self.completed_inspections.add(idx)
        self.last_inspect = None
        return True

    def _auto_goal(self, pos: np.ndarray):
        """Return (goal, segment, path_error, progress_s, action)."""
        path_dist, nearest, seg, s = project_to_path(pos, self.wp, self.cum)
        remaining = self.path_len - s
        pending_inspections = any(idx not in self.completed_inspections
                                  for idx in self.inspect_waypoints)
        at_end = remaining <= self.final_arrive_tolerance() and path_dist <= self.cfg.rejoin_tol
        if at_end:
            return self.wp[-1].copy(), seg, path_dist, s, \
                ("ABORT_PENDING_INSPECTION" if pending_inspections else "LAND")
        if path_dist > self.cfg.rejoin_tol:
            return nearest, seg, path_dist, s, "REJOIN"
        goal_s = min(self.path_len, s + self.cfg.lookahead)
        goal, gi = point_at_s(self.wp, self.cum, goal_s)
        return goal, gi, path_dist, s, "FOLLOW"

    def step(self, pose: Pose | None, now: Optional[float] = None) -> Command:
        """Compute one control command from the latest visual pose.

        If pose is missing or stale, returns HOVER with zero velocity.  The caller
        should keep sending hover/zero PCMD and decide when to auto-land.
        """
        now = time.monotonic() if now is None else float(now)
        pose_values = () if pose is None else (pose.x, pose.y, pose.z, pose.yaw, pose.stamp)
        pose_ok = bool(pose is not None and math.isfinite(now)
                       and all(math.isfinite(float(v)) for v in pose_values))
        age = float("inf") if not pose_ok else now - float(pose.stamp)
        if not pose_ok or age < -0.05 or age > self.cfg.max_pose_age_s:
            self.state = "HOVER"
            return Command("HOVER", np.zeros(3), 0.0, np.zeros(3), float("inf"), 0.0,
                           status="pose missing/stale -> HOVER")

        pos = pose.xyz
        goal, seg, path_error, progress_s, action = self._auto_goal(pos)
        progress = progress_s / max(1e-9, self.path_len)

        look = self._look_at_pole_target(pos)
        if look:
            wpnum, pole_center, pole_id = look
            look_meta = {"waypoint": wpnum, "pole_id": pole_id,
                         "target": pole_center.tolist(), "capture_ack_required": True,
                         "body_yaw_required": bool(self.cfg.pole_body_look)}
            yaw_target = (heading_from_to(pos, pole_center) if self.cfg.pole_body_look
                          else float(pose.yaw))
            self.state = "INSPECT"
            return Command("INSPECT", np.zeros(3), yaw_target, pos.copy(), path_error, progress,
                           look_at_pole=look_meta, should_land=False,
                           status=f"INSPECT wp={wpnum} pole={pole_id} awaiting gimbal/capture ack")

        if action == "ABORT_PENDING_INSPECTION":
            self.state = "LANDING"
            return Command("ABORT", np.zeros(3), pose.yaw, goal, path_error, progress,
                           should_land=True,
                           status="pending inspection missed at route end -> abort/land")
        if action == "LAND":
            self.state = "LANDING"
            return Command("LAND", np.zeros(3), pose.yaw, goal, path_error, progress,
                           should_land=True, status="final path reached -> LAND")

        yaw_target = heading_from_to(pos, goal)
        look_meta = None
        status = action

        d = goal - pos
        dist = float(np.linalg.norm(d))
        vel = np.zeros(3, dtype=float) if dist < 1e-9 else d / dist * self.cfg.speed
        self.state = "CRUISE"
        return Command(action, vel, yaw_target, goal, path_error, progress,
                       look_at_pole=look_meta, should_land=False, status=status)


def command_to_body_percent(cmd: Command, pose: Pose, max_pitch: int = 8,
                            max_roll: int = 0, max_yaw: int = 25,
                            max_gaz: int = 12, k_yaw: float = 1.4,
                            k_vert: float = 1.0, yaw_sign: int = 1):
    """Optional helper: convert a map-frame command to ANAFI-style PCMD percent.

    This mirrors the old slow/safe PCMD mapping: yaw first, forward pitch only
    when the drone is roughly facing the desired yaw.  Roll remains zero by
    default.  VERIFY signs on props-off bench before real flight.
    """
    try:
        vel = _finite_vec3(cmd.vel_map, "command velocity")
        _finite_vec3(cmd.goal, "command goal")
        scalars = [cmd.yaw_target, cmd.path_error, cmd.progress,
                   pose.x, pose.y, pose.z, pose.yaw, pose.stamp,
                   max_pitch, max_roll, max_yaw, max_gaz, k_yaw, k_vert, yaw_sign]
        if not all(math.isfinite(float(v)) for v in scalars):
            return 0, 0, 0, 0
        if (not all(0 <= int(v) <= 100 for v in (max_pitch, max_roll, max_yaw, max_gaz))
                or float(k_yaw) < 0.0 or float(k_vert) < 0.0 or int(yaw_sign) not in (-1, 1)):
            return 0, 0, 0, 0
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0, 0, 0, 0
    yaw_err = wrap_angle(cmd.yaw_target - pose.yaw)
    yaw_cmd = yaw_sign * max(-1.0, min(1.0, k_yaw * yaw_err)) * max_yaw
    facing = max(0.0, math.cos(yaw_err))
    # pitch reacts to HORIZONTAL motion only (X/Z); the vertical Y component drives gaz below.
    # (Using the full 3D norm made a pure altitude correction produce spurious forward pitch.)
    horiz_speed = math.hypot(float(vel[0]), float(vel[2]))
    speed_frac = min(1.0, horiz_speed / max(1e-9, 1.2))
    pitch_cmd = max_pitch * speed_frac * facing
    # Current GLOMAP up is -Y; positive vertical desire means target y lower.
    gaz_cmd = max(-1.0, min(1.0, k_vert * (-vel[1]))) * max_gaz
    return int(round(max_roll * 0)), int(round(pitch_cmd)), int(round(yaw_cmd)), int(round(gaz_cmd))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test the route controller without flying.")
    ap.add_argument("--path", default=str(Path(__file__).resolve().parents[1] / "safezone/flight_path.json"))
    ap.add_argument("--poles", default=str(Path(__file__).resolve().parents[1] / "safezone/poles.json"))
    args = ap.parse_args()
    wp = load_waypoints(args.path)
    poles = load_poles(args.poles)
    ctrl = RouteAutoController(wp, poles)
    print(f"loaded waypoints={len(wp)} poles={len(poles)} inspect_valid={sorted(i+1 for i in ctrl.inspect_waypoints)}")
    print("This module outputs commands; it does not connect to or fly the drone.")
