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


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


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
    # Map-unit arrival tolerance. Scale-dependent like radius/max_jump: 0.1 was
    # chosen against urai EDM v1, whose camera track spans 6.07 map units and
    # whose consecutive references sit 0.027 apart. Re-derive it per site.
    arrive: float = 0.1                 # absolute cap for end tolerance
    arrive_fraction: float = 0.02       # end tolerance <= 2% of route length
    min_arrive: float = 0.1             # but never smaller than this map-unit value
    inspect_waypoints: tuple[int, ...] = (9, 10, 11, 15)  # 1-based labels in current 15-point route
    inspect_radius: float = 0.75        # map-unit trigger around each label
    inspect_resume_margin: float = 0.25 # arclength margin after last inspected wp
    inspect_yaw_tolerance_deg: float = 6.0
    max_pose_age_s: float = 0.5         # stale visual pose -> HOVER
    segment_window: int = 2             # active segment +/- this many route segments
    max_progress_regression: float = 0.2
    progress_jump_slack: float = 1.5    # initial/cumulative map-unit progress budget
    progress_speed_factor: float = 2.0  # observed motion replenishes progress budget
    # ANAFI's inspection path uses body yaw for horizontal aim and gimbal pitch
    # for vertical aim. Translation remains zero while aligning/capturing.
    pole_body_look: bool = True
    # Simulator-validated body PCMD mapping. Distance values remain map units;
    # the production builder scales the physical metre values per site.
    yaw_tolerance_deg: float = 3.0
    yaw_alignment_confirmation_updates: int = 3
    yaw_alignment_max_rate_deg_s: float = 5.0
    yaw_command_period_s: float = 1.0
    minimum_yaw_alignment_distance: float = 0.25
    max_translation_pcmd: int = 10
    max_yaw_pcmd: int = 20
    slowdown_distance: float = 1.5
    translation_arrival_tolerance: float = 0.15

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
            "max_progress_regression": (0.0, 100.0),
            "progress_jump_slack": (0.0, 100.0),
            "progress_speed_factor": (0.0, 10.0),
            "yaw_tolerance_deg": (0.0, 45.0),
            "yaw_alignment_max_rate_deg_s": (0.0, 180.0),
            "yaw_command_period_s": (0.0, 10.0),
            "minimum_yaw_alignment_distance": (0.0, 100.0),
            "slowdown_distance": (0.0, 100.0),
            "translation_arrival_tolerance": (0.0, 100.0),
        }
        for name, (lo, hi) in limits.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= lo or value > hi:
                raise ValueError(f"{name} must be finite and in ({lo}, {hi}], got {value!r}")
        if (not isinstance(self.inspect_waypoints, tuple)
                or any(not isinstance(i, int) or isinstance(i, bool) or i < 1
                       for i in self.inspect_waypoints)):
            raise ValueError("inspect_waypoints must be a tuple of positive 1-based integers")
        if (isinstance(self.segment_window, bool) or not isinstance(self.segment_window, int)
                or not 0 <= self.segment_window <= 20):
            raise ValueError("segment_window must be an integer in [0,20]")
        for name in ("yaw_alignment_confirmation_updates", "max_translation_pcmd",
                     "max_yaw_pcmd"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not 1 <= value <= 100):
                raise ValueError(f"{name} must be an integer in [1,100]")


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

def load_waypoints(
    path_json: str | Path,
    *,
    expected_site_id: str | None = None,
    expected_coordinate_frame_id: str | None = None,
    require_flight_contract: bool = False,
) -> list[np.ndarray]:
    data = json.loads(
        Path(path_json).read_text(),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(data, dict) or not isinstance(data.get("waypoints"), list):
        raise ValueError("route JSON must contain a waypoints list")
    if require_flight_contract:
        expected = {
            "schema": "sfm-flight-route/v1",
            "site_id": expected_site_id,
            "coordinate_frame_id": expected_coordinate_frame_id,
            "units": "map",
            "purpose": "flight",
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(
                    f"flight route {key} must be {value!r}, got {data.get(key)!r}"
                )
        if data.get("closed") is not False:
            raise ValueError(
                f"flight route closed must be False, got {data.get('closed')!r}"
            )
    frame = data.get("frame", "aligned")
    if frame not in {"aligned", "glomap"}:
        raise ValueError("route frame must be 'aligned' or 'glomap'")
    if data.get("units", "map") != "map":
        raise ValueError("route units must be 'map'")
    points = (
        [aligned_to_glomap(point) for point in data["waypoints"]]
        if frame == "aligned"
        else data["waypoints"]
    )
    return _validated_waypoints(points)


def load_poles(poles_json: str | Path) -> list[dict]:
    p = Path(poles_json)
    if not p.exists():
        return []
    data = json.loads(
        p.read_text(),
        parse_constant=_reject_json_constant,
    )
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
        self.active_segment = 0
        self.progress_s = 0.0
        self._progress_budget = float(self.cfg.progress_jump_slack)
        self._last_progress_position: np.ndarray | None = None

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
        last_s = max(float(self.cum[i]) for i in self.inspect_waypoints)
        return self.progress_s >= last_s + self.cfg.inspect_resume_margin

    def _look_at_pole_target(self, pos: np.ndarray):
        """Return (wp_1based, pole_center, pole_id) if inside a look-at region."""
        if not self.poles or not self.inspect_waypoints:
            self.last_inspect = None
            return None
        if self._inspection_resume_reached(pos):
            self.last_inspect = None
            return None
        s = self.progress_s
        best = None
        for idx in self.inspect_waypoints:
            if idx in self.completed_inspections:
                continue
            eu = float(np.linalg.norm(pos - self.wp[idx]))
            ar = abs(float(s - self.cum[idx]))
            score = max(eu, ar)
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

    def _project_with_progress(self, pos: np.ndarray):
        if self._last_progress_position is not None:
            movement = float(np.linalg.norm(pos - self._last_progress_position))
            if math.isfinite(movement):
                self._progress_budget += movement * self.cfg.progress_speed_factor
        self._last_progress_position = pos.copy()

        lo = max(0, self.active_segment - self.cfg.segment_window)
        hi = min(len(self.wp) - 2, self.active_segment + self.cfg.segment_window)
        min_s = max(0.0, self.progress_s - self.cfg.max_progress_regression)
        max_s = min(self.path_len, self.progress_s + self._progress_budget)
        candidates = []
        for index in range(lo, hi + 1):
            distance, nearest, t = point_segment_distance(
                pos, self.wp[index], self.wp[index + 1]
            )
            segment_length = float(self.cum[index + 1] - self.cum[index])
            s = float(self.cum[index] + t * segment_length)
            if min_s <= s <= max_s + 1e-9:
                candidates.append((distance, abs(s - self.progress_s), nearest, index, s))
        if not candidates:
            index = self.active_segment
            distance, nearest, t = point_segment_distance(
                pos, self.wp[index], self.wp[index + 1]
            )
            segment_length = float(self.cum[index + 1] - self.cum[index])
            s = min(
                max_s,
                max(min_s, float(self.cum[index] + t * segment_length)),
            )
            nearest, _ = point_at_s(self.wp, self.cum, s)
            distance = float(np.linalg.norm(pos - nearest))
            chosen = (distance, abs(s - self.progress_s), nearest, index, s)
        else:
            chosen = min(candidates, key=lambda item: (item[0], item[1], item[3]))
        path_dist, _continuity, nearest, segment, projected_s = chosen
        if projected_s > self.progress_s:
            self._progress_budget = max(
                0.0,
                self._progress_budget - (projected_s - self.progress_s),
            )
            self.progress_s = projected_s
        safe_s = self.progress_s
        self.active_segment = int(
            max(
                0,
                min(
                    len(self.wp) - 2,
                    np.searchsorted(self.cum, safe_s, side="right") - 1,
                ),
            )
        )
        return path_dist, nearest, segment, safe_s

    def _auto_goal(self, pos: np.ndarray):
        """Return (goal, segment, path_error, progress_s, action)."""
        path_dist, nearest, seg, s = self._project_with_progress(pos)
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


def command_to_body_percent(
    cmd: Command,
    pose: Pose,
    *,
    config: ControlConfig | None = None,
    yaw_sign: int = 1,
    require_yaw_alignment: bool = True,
) -> tuple[int, int, int, int]:
    """Map the latest camera-target line to ANAFI body PCMD percentages.

    The route planner stays in the raw GLOMAP frame (horizontal X/Z, up=-Y).
    This adapter rotates that 3-D target vector into body forward/right/up,
    normalizes the resultant, and assigns pitch/roll/gaz from one shared bounded
    strength.  When requested, yaw is corrected by itself before translation.
    """
    cfg = config or ControlConfig()
    zero = (0, 0, 0, 0)
    try:
        _finite_vec3(cmd.vel_map, "command velocity")
        goal = _finite_vec3(cmd.goal, "command goal")
        scalars = [cmd.yaw_target, cmd.path_error, cmd.progress,
                   pose.x, pose.y, pose.z, pose.yaw, pose.stamp]
        if not all(math.isfinite(float(value)) for value in scalars):
            return zero
        if int(yaw_sign) not in (-1, 1):
            return zero
    except (AttributeError, TypeError, ValueError, OverflowError):
        return zero

    movement = cmd.action in {"FOLLOW", "REJOIN"}
    inspection = cmd.action == "INSPECT"
    if cmd.should_land or (not movement and not inspection):
        return zero

    yaw_error = wrap_angle(float(cmd.yaw_target) - float(pose.yaw))
    yaw_error_deg = math.degrees(yaw_error)
    if inspection or (
        require_yaw_alignment
        and math.hypot(float(goal[0] - pose.x), float(goal[2] - pose.z))
        > cfg.minimum_yaw_alignment_distance
        and abs(yaw_error_deg) > cfg.yaw_tolerance_deg
    ):
        if abs(yaw_error_deg) <= cfg.yaw_tolerance_deg:
            return zero
        yaw_strength = min(
            cfg.max_yaw_pcmd,
            max(
                1,
                round(
                    100.0 * abs(yaw_error_deg)
                    / 70.0
                    / cfg.yaw_command_period_s
                ),
            ),
        )
        yaw = int(yaw_sign) * (yaw_strength if yaw_error > 0.0 else -yaw_strength)
        return 0, 0, int(yaw), 0

    if not movement:
        return zero

    delta = goal - pose.xyz
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= cfg.translation_arrival_tolerance:
        return zero

    # Body forward in the GLOMAP X/Z horizontal plane is (cos(yaw), sin(yaw));
    # body right is (sin(yaw), -cos(yaw)). GLOMAP gravity-up is -Y.
    dx, dy, dz = (float(value) for value in delta)
    forward = dx * math.cos(pose.yaw) + dz * math.sin(pose.yaw)
    right = dx * math.sin(pose.yaw) - dz * math.cos(pose.yaw)
    up = -dy
    forward_fraction = forward / distance
    right_fraction = right / distance
    up_fraction = up / distance

    raw_strength = min(
        cfg.max_translation_pcmd,
        max(2, round(cfg.max_translation_pcmd * distance / cfg.slowdown_distance)),
    )
    remaining = max(0.0, distance - cfg.translation_arrival_tolerance)
    slowdown_span = max(
        1e-6,
        cfg.slowdown_distance - cfg.translation_arrival_tolerance,
    )
    slowdown_scale = max(0.25, min(1.0, remaining / slowdown_span))
    strength = max(1, round(raw_strength * slowdown_scale))

    clamp = lambda value: int(round(max(-100.0, min(100.0, float(value)))))
    return (
        clamp(strength * right_fraction),
        clamp(strength * forward_fraction),
        0,
        clamp(strength * up_fraction),
    )


class YawAlignedPcmdController:
    """Stateful hover/turn/translate gate shared by dry-run and real flight."""

    def __init__(self, config: ControlConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.target_key = None
        self.aligned_for_translation = False
        self.confirmation_count = 0
        self.previous_timestamp: float | None = None
        self.previous_yaw: float | None = None
        self.phase = "idle"

    def update(
        self,
        cmd: Command,
        pose: Pose,
        now: float,
        *,
        target_key,
        yaw_sign: int = 1,
    ) -> tuple[int, int, int, int]:
        try:
            now = float(now)
            if not math.isfinite(now):
                raise ValueError("non-finite control timestamp")
        except (TypeError, ValueError):
            self.reset()
            return 0, 0, 0, 0

        if (self.previous_timestamp is not None
                and now - self.previous_timestamp > self.config.max_pose_age_s):
            self.reset()
        if target_key != self.target_key:
            self.reset()
            self.target_key = target_key

        if cmd.action not in {"FOLLOW", "REJOIN"} or cmd.should_land:
            self.reset()
            return 0, 0, 0, 0

        try:
            goal = _finite_vec3(cmd.goal, "command goal")
            horizontal = math.hypot(float(goal[0] - pose.x), float(goal[2] - pose.z))
            yaw_error = wrap_angle(float(cmd.yaw_target) - float(pose.yaw))
        except (AttributeError, TypeError, ValueError, OverflowError):
            self.reset()
            return 0, 0, 0, 0

        yaw_rate_deg_s = None
        if self.previous_timestamp is not None and self.previous_yaw is not None:
            dt = now - self.previous_timestamp
            if dt > 1e-6:
                yaw_rate_deg_s = math.degrees(
                    wrap_angle(float(pose.yaw) - self.previous_yaw)
                ) / dt
        self.previous_timestamp = now
        self.previous_yaw = float(pose.yaw)

        if horizontal <= self.config.minimum_yaw_alignment_distance:
            self.aligned_for_translation = True

        if not self.aligned_for_translation:
            stable = (
                abs(math.degrees(yaw_error)) <= self.config.yaw_tolerance_deg
                and yaw_rate_deg_s is not None
                and abs(yaw_rate_deg_s) <= self.config.yaw_alignment_max_rate_deg_s
            )
            self.confirmation_count = self.confirmation_count + 1 if stable else 0
            if abs(math.degrees(yaw_error)) > self.config.yaw_tolerance_deg:
                self.phase = "turn"
                return command_to_body_percent(
                    cmd,
                    pose,
                    config=self.config,
                    yaw_sign=yaw_sign,
                    require_yaw_alignment=True,
                )
            if self.confirmation_count >= self.config.yaw_alignment_confirmation_updates:
                self.aligned_for_translation = True
                self.phase = "yaw_alignment_confirmed"
            else:
                self.phase = "yaw_alignment_hold"
            return 0, 0, 0, 0

        self.phase = "translate"
        return command_to_body_percent(
            cmd,
            pose,
            config=self.config,
            yaw_sign=yaw_sign,
            require_yaw_alignment=False,
        )


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
