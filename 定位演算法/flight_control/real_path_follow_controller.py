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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from route_domain import (
    MissionRouteLock as _MissionRouteLock,
    MissionRouteSnapshot,
    RouteDocument,
    _validated_waypoints as _validated_route_waypoints,
    aligned_to_glomap as _domain_aligned_to_glomap,
    capture_mission_route_snapshot as _capture_mission_route_snapshot,
)

# Preserve the historical public import path while keeping the implementation
# in the shared route domain module.
MissionRouteLock = _MissionRouteLock

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
    """Visual-localizer pose in the raw GLOMAP map frame.

    x/y/z are raw GLOMAP coordinates and carry NO axis convention of their own: a
    reconstruction is gauge-free, so which direction is up is a per-site
    MEASUREMENT (see MapFrame / T_align_gravity.json), not a property of the axes.
    ``yaw`` is the heading in that measured horizontal plane, produced by
    MapFrame.heading(); it is NOT atan2(z, x) unless the site's measured basis
    happens to be the legacy one.

    Field-compatible with pose_types.Pose, which is what the operator interface's
    localizer worker produces.
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

    `vel_map` is the UNIT direction toward the goal in current GLOMAP map units.
    It carries no speed: the map is scale-free, so the commanded rate lives in the
    PCMD adapter (max_translation_pcmd / slowdown_distance), not here.
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


@dataclass(frozen=True, eq=False)
class MapFrame:
    """Which way is horizontal, and which way is up, in the raw GLOMAP frame.

    The legacy convention (X/Z horizontal, -Y up) is an ASSUMPTION about how the
    reconstruction happened to be oriented, not a measurement. On target_site_v1 it
    is wrong by 22.5 deg (see T_align_gravity.json), which tilts every
    forward/right/up decomposition built from it: part of a commanded climb leaks
    into horizontal motion and vice versa. Prefer the measured frame.

    ``east`` keeps the GLOMAP +X azimuth, ``north`` completes the horizontal pair
    such that east x north == up, which reproduces the legacy convention exactly
    when gravity really is +Y.
    """

    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    source: str = "legacy_assumption"

    # eq=False on the dataclass: the generated __eq__ compares the fields as a
    # tuple, and comparing two numpy arrays yields an array, so `a == b` raised
    # ValueError (ambiguous truth value) and the frozen __hash__ raised TypeError
    # (unhashable ndarray). Both surface through ControlConfig, which holds a
    # MapFrame -- comparing or caching two configs built from separately-loaded
    # measured frames raised instead of answering. Compare by value instead.
    def _key(self) -> tuple:
        return (
            tuple(np.asarray(self.east, dtype=float).ravel().tolist()),
            tuple(np.asarray(self.north, dtype=float).ravel().tolist()),
            tuple(np.asarray(self.up, dtype=float).ravel().tolist()),
            str(self.source),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MapFrame):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    @classmethod
    def legacy(cls) -> "MapFrame":
        return cls(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, -1.0, 0.0]),
            source="legacy_assumption",
        )

    @classmethod
    def from_gravity(cls, gravity_glomap, *, source: str = "measured") -> "MapFrame":
        """Build the basis from a measured gravity (DOWN) vector in GLOMAP units."""
        g = _finite_vec3(gravity_glomap, "gravity vector")
        norm = float(np.linalg.norm(g))
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError("gravity vector must be non-zero and finite")
        up = -g / norm
        x_axis = np.array([1.0, 0.0, 0.0])
        east = x_axis - float(np.dot(x_axis, up)) * up
        east_norm = float(np.linalg.norm(east))
        if east_norm < 1e-6:
            # Gravity parallel to +X: the azimuth reference is degenerate, so there
            # is no defensible heading convention to fall back on.
            raise ValueError("gravity is parallel to GLOMAP +X; azimuth undefined")
        east = east / east_norm
        return cls(east, np.cross(up, east), up, source=source)

    def horizontal(self, delta) -> tuple[float, float]:
        d = np.asarray(delta, float)
        return float(np.dot(d, self.east)), float(np.dot(d, self.north))

    def vertical(self, delta) -> float:
        return float(np.dot(np.asarray(delta, float), self.up))

    def heading(self, delta) -> float:
        east, north = self.horizontal(delta)
        return float(math.atan2(north, east))

    def horizontal_distance(self, delta) -> float:
        east, north = self.horizontal(delta)
        return float(math.hypot(east, north))

    def body_components(self, delta, yaw: float) -> tuple[float, float, float]:
        """Split a map-frame displacement into body forward / right / up."""
        east, north = self.horizontal(delta)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        forward = east * cos_yaw + north * sin_yaw
        right = east * sin_yaw - north * cos_yaw
        return forward, right, self.vertical(delta)


LEGACY_MAP_FRAME = MapFrame.legacy()


def load_map_frame(path: str | Path) -> MapFrame:
    """Load the measured gravity direction from a site's T_align_gravity.json."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"),
                      parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise ValueError(f"map alignment must be a JSON object: {source}")
    if data.get("schema") != "sfm-align/v2":
        raise ValueError(
            f"map alignment schema must be sfm-align/v2, got {data.get('schema')!r}: {source}"
        )
    gravity = data.get("gravity_glomap")
    rotation = data.get("R")
    if gravity is None or rotation is None:
        raise ValueError(f"map alignment needs both R and gravity_glomap: {source}")
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError(f"map alignment R must be a finite 3x3 matrix: {source}")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or float(np.linalg.det(R)) < 0.0:
        raise ValueError(
            f"map alignment R must be a proper rotation (orthonormal, det>0): {source}"
        )
    frame = MapFrame.from_gravity(gravity, source=str(source))
    # R's third row is the aligned Z axis, i.e. up. Cross-checking it against the
    # basis rebuilt from gravity_glomap catches a file whose two representations
    # disagree -- which would silently pick whichever one the reader happened to use.
    if not np.allclose(R[2], frame.up, atol=1e-6):
        raise ValueError(
            f"map alignment R row 2 disagrees with gravity_glomap: {source}"
        )
    return frame



@dataclass
class ControlConfig:
    # NOTE: there is deliberately no `speed` here. The map is scale-free, so a
    # map-units/sec figure cannot be converted into a tilt percentage. Commanded
    # speed is set by max_translation_pcmd (a % of the firmware MaxTilt /
    # MaxVerticalSpeed) and shaped by slowdown_distance. A `speed` field existed
    # until 2026-08-06 but only ever scaled Command.vel_map, which the PCMD
    # adapter ignores -- tuning it changed nothing about how the drone flew.
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
    #: Capture radius (map units): how close the drone must be before a waypoint
    #: counts as reached. Bounds how far one fix may advance the target.
    progress_jump_slack: float = 1.5
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
    yaw_alignment_timeout_s: float = 6.0
    # Retained for profile compatibility. Horizontal legs no longer bypass the
    # mandatory turn gate based on distance; only a purely vertical ray may do so.
    minimum_yaw_alignment_distance: float = 0.25
    max_translation_pcmd: int = 10
    max_yaw_pcmd: int = 20
    slowdown_distance: float = 1.5
    #: Distance at which the PCMD adapter stops commanding. It MUST stay strictly
    #: inside waypoint_arrive_radius: at 0.15 against a 0.02 sphere the drone
    #: stopped 0.15 short of every waypoint, was never "reached", and autonomy
    #: hovered there forever. __post_init__ enforces the ordering.
    translation_arrival_tolerance: float = 0.002
    #: Radius of the sphere around a clicked waypoint that counts as "arrived"
    #: (MAP UNITS -- scale-dependent, re-derive per site). Operator decision
    #: 2026-08-06, after viewing the map: 0.02, inside the operator's 0.005-0.05
    #: range. 1.0 (68% of urai's usable height) and 0.3 (5% of its 5.62-unit camera
    #: trajectory) were both tried first and were far too coarse. Distinct from translation_arrival_tolerance, which is where the
    #: PCMD adapter stops commanding; the target normally switches well before then.
    waypoint_arrive_radius: float = 0.02
    #: Consecutive ticks the drone must stay inside that sphere before the target
    #: advances. One frame is a single visual fix, and a single bad fix landing in
    #: the sphere would otherwise retire a waypoint the drone never reached.
    waypoint_arrive_confirm_frames: int = 3
    #: Exactly one waypoint may retire per pose update. This preserves a distinct
    #: hover/turn gate for every clicked point, including tightly spaced points.
    waypoint_advance_budget: int = 1
    #: End-of-route LAND needs several distinct fresh captures over a dwell period.
    final_arrive_confirm_frames: int = 5
    final_hold_s: float = 1.0
    #: Measured map basis. The default is the legacy X/Z-horizontal, -Y-up guess.
    map_frame: MapFrame = LEGACY_MAP_FRAME

    def __post_init__(self):
        limits = {
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
            "yaw_alignment_timeout_s": (0.0, 120.0),
            "minimum_yaw_alignment_distance": (0.0, 100.0),
            "slowdown_distance": (0.0, 100.0),
            "translation_arrival_tolerance": (0.0, 100.0),
            "waypoint_arrive_radius": (0.0, 100.0),
            "final_hold_s": (0.0, 30.0),
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
        if float(self.translation_arrival_tolerance) >= float(self.waypoint_arrive_radius):
            # The adapter must keep commanding until the drone is INSIDE the sphere,
            # or the sequencer can never retire the waypoint and the route stalls.
            raise ValueError(
                "translation_arrival_tolerance must be smaller than "
                f"waypoint_arrive_radius ({self.translation_arrival_tolerance} >= "
                f"{self.waypoint_arrive_radius}); autonomy would stall short of "
                "every waypoint"
            )
        final_floor = min(float(self.arrive), float(self.min_arrive))
        if final_floor <= float(self.translation_arrival_tolerance):
            raise ValueError(
                "the end-of-route arrival window must be larger than "
                f"translation_arrival_tolerance ({final_floor} <= "
                f"{self.translation_arrival_tolerance}); the final waypoint would "
                "never terminate the mission"
            )
        if float(self.progress_jump_slack) < float(self.waypoint_arrive_radius):
            # The capture bound must not be tighter than the arrival sphere, or a
            # waypoint could be "arrived at" and simultaneously refused as too far.
            raise ValueError(
                "progress_jump_slack must be >= waypoint_arrive_radius "
                f"({self.progress_jump_slack} < {self.waypoint_arrive_radius})"
            )
        if not isinstance(self.map_frame, MapFrame):
            raise ValueError("map_frame must be a MapFrame")
        for name in ("yaw_alignment_confirmation_updates", "max_translation_pcmd",
                     "max_yaw_pcmd", "waypoint_arrive_confirm_frames",
                     "waypoint_advance_budget", "final_arrive_confirm_frames"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not 1 <= value <= 100):
                raise ValueError(f"{name} must be an integer in [1,100]")
        if self.waypoint_advance_budget != 1:
            raise ValueError("waypoint_advance_budget must be exactly 1")


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
    """Compatibility adapter to the shared route-domain validator."""
    return [np.array(point, dtype=float, copy=True)
            for point in _validated_route_waypoints(waypoints)]


def aligned_to_glomap(p, frame: MapFrame = LEGACY_MAP_FRAME) -> np.ndarray:
    """Compatibility adapter to the shared route-domain conversion."""
    return _domain_aligned_to_glomap(p, frame)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi



def heading_from_to(a: np.ndarray, b: np.ndarray,
                    frame: MapFrame = LEGACY_MAP_FRAME) -> float:
    return frame.heading(np.asarray(b, float) - np.asarray(a, float))


def ground_projected_yaw_error(
    nose_yaw: float,
    target_delta: np.ndarray,
    frame: MapFrame = LEGACY_MAP_FRAME,
) -> float | None:
    """Signed angle from the ground-projected nose ray to the target ray.

    ``nose_yaw`` already describes the aircraft's forward ray in ``frame``'s
    measured horizontal basis.  The target displacement is projected onto that
    same plane before the signed angle is computed.  A purely vertical target
    has no defensible yaw direction and returns ``None``.
    """
    yaw = float(nose_yaw)
    if not math.isfinite(yaw):
        raise ValueError("nose yaw must be finite")
    east, north = frame.horizontal(_finite_vec3(target_delta, "target displacement"))
    target_norm = math.hypot(east, north)
    if not math.isfinite(target_norm) or target_norm < 1e-9:
        return None
    target_east = east / target_norm
    target_north = north / target_norm
    nose_east, nose_north = math.cos(yaw), math.sin(yaw)
    dot = nose_east * target_east + nose_north * target_north
    cross = nose_east * target_north - nose_north * target_east
    return float(math.atan2(cross, dot))


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

def _waypoints_from_route_data(
    data: object,
    *,
    expected_site_id: str | None = None,
    expected_coordinate_frame_id: str | None = None,
    require_flight_contract: bool = False,
    map_frame: MapFrame = LEGACY_MAP_FRAME,
) -> list[np.ndarray]:
    return RouteDocument.from_data(
        data,
        expected_site_id=expected_site_id,
        expected_coordinate_frame_id=expected_coordinate_frame_id,
        require_flight_contract=require_flight_contract,
        require_map_units=True,
        map_frame=map_frame,
    ).controller_waypoints()


def load_waypoints(
    path_json: str | Path,
    *,
    expected_site_id: str | None = None,
    expected_coordinate_frame_id: str | None = None,
    require_flight_contract: bool = False,
    map_frame: MapFrame = LEGACY_MAP_FRAME,
) -> list[np.ndarray]:
    return RouteDocument.from_path(
        path_json,
        expected_site_id=expected_site_id,
        expected_coordinate_frame_id=expected_coordinate_frame_id,
        require_flight_contract=require_flight_contract,
        require_map_units=True,
        map_frame=map_frame,
    ).controller_waypoints()


def capture_mission_route_snapshot(
    path_json: str | Path,
    *,
    expected_sha256: str,
    expected_site_id: str,
    expected_coordinate_frame_id: str,
    map_frame: MapFrame = LEGACY_MAP_FRAME,
) -> MissionRouteSnapshot:
    """Compatibility adapter to the shared route-domain snapshot."""
    return _capture_mission_route_snapshot(
        path_json,
        expected_sha256=expected_sha256,
        expected_site_id=expected_site_id,
        expected_coordinate_frame_id=expected_coordinate_frame_id,
        map_frame=map_frame,
    )


def load_route_arrive_radius(path_json: str | Path) -> float | None:
    """The arrival sphere the operator confirmed in the route editor, if declared.

    The editor writes `arrive_radius_map_units` into every route it exports. Read
    it here, or the operator sets a radius on a slider and the drone flies the
    ControlConfig default instead -- the two silently disagree.
    """
    return RouteDocument.from_path(path_json, require_map_units=True).arrive_radius_map_units


def config_for_route(
    path_json: str | Path | MissionRouteSnapshot, base: "ControlConfig | None" = None
) -> "ControlConfig":
    """Apply a route's arrival radius without rereading a validated snapshot."""
    cfg = ControlConfig() if base is None else base
    if isinstance(path_json, MissionRouteSnapshot):
        radius = path_json.arrive_radius_map_units
    else:
        radius = load_route_arrive_radius(path_json)
    if radius is None:
        return cfg
    # progress_jump_slack must stay >= the arrival sphere (__post_init__ enforces
    # it), so widen the capture bound rather than reject the operator's radius.
    return replace(
        cfg,
        waypoint_arrive_radius=radius,
        progress_jump_slack=max(float(cfg.progress_jump_slack), radius),
    )


def load_poles(poles_json: str | Path,
               map_frame: MapFrame = LEGACY_MAP_FRAME) -> list[dict]:
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
        center_g = aligned_to_glomap(center, map_frame)
        base_g = aligned_to_glomap(pole.get("base", center), map_frame)
        top_g = aligned_to_glomap(pole.get("top", center), map_frame)
        out.append({
            "id": i,
            "center": center_g,
            "base": base_g,
            "top": top_g,
            "raw": pole,
        })
    return out


def nearest_pole(poles: list[dict], ref: np.ndarray,
                 map_frame: MapFrame = LEGACY_MAP_FRAME):
    """Return the pole nearest to ref in the site's measured horizontal plane."""
    if not poles:
        return None, None, None
    ref = np.asarray(ref, float)
    best = None
    for pole in poles:
        c = np.asarray(pole["center"], float)
        d = map_frame.horizontal_distance(c - ref)
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
        checked_waypoints = _validated_waypoints(waypoints)
        frozen_waypoints = []
        for point in checked_waypoints:
            copied = np.array(point, dtype=float, copy=True)
            copied.setflags(write=False)
            frozen_waypoints.append(copied)
        self.wp = tuple(frozen_waypoints)
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
        # Direct-to-waypoint: the operator's clicked points ARE the plan. The
        # target index only ever advances, so a relocation jump cannot send the
        # drone back to an earlier leg.
        self.target_index = 0
        self.holding_for_inspection = False
        self._arrive_frames = 0
        self._last_sequence_pose_stamp: float | None = None
        self._final_hold_started: float | None = None
        self._final_pose_stamp: float | None = None
        self._final_confirm_frames = 0
        self.active_segment = 0
        self.progress_s = 0.0

    def final_arrive_tolerance(self) -> float:
        """End-of-route tolerance for LAND.

        The legacy fixed 1.0 map-unit value was too loose for short routes and
        could mark route completion around 90% progress. Keep 1.0 as a hard cap,
        but scale the active tolerance to the current route length.
        """
        scaled = max(
            float(self.cfg.min_arrive),
            min(float(self.cfg.arrive), self.path_len * float(self.cfg.arrive_fraction)),
        )
        return min(float(self.cfg.waypoint_arrive_radius), scaled)

    def _reset_final_hold(self) -> None:
        self._final_hold_started = None
        self._final_pose_stamp = None
        self._final_confirm_frames = 0

    def _final_hold_complete(self, pos: np.ndarray, pose_stamp: float,
                             now: float) -> bool:
        if float(np.linalg.norm(self.wp[-1] - pos)) > self.final_arrive_tolerance():
            self._reset_final_hold()
            return False
        # Reprocessing one captured frame must never accumulate landing evidence.
        if self._final_pose_stamp is not None and pose_stamp <= self._final_pose_stamp:
            return False
        if self._final_hold_started is None:
            self._final_hold_started = now
            self._final_confirm_frames = 1
        else:
            self._final_confirm_frames += 1
        self._final_pose_stamp = pose_stamp
        return (
            self._final_confirm_frames >= int(self.cfg.final_arrive_confirm_frames)
            and now - self._final_hold_started >= float(self.cfg.final_hold_s)
        )

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
        pole_id, pole, pole_dist = nearest_pole(
            self.poles, self.wp[idx], self.cfg.map_frame
        )
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

    def _leg_direction(self, index: int) -> np.ndarray | None:
        """Unit direction of the leg leading INTO waypoint `index`."""
        start = self.wp[index - 1] if index >= 1 else self.wp[0]
        end = self.wp[index] if index >= 1 else self.wp[min(1, len(self.wp) - 1)]
        leg = end - start
        length = float(np.linalg.norm(leg))
        return None if length < 1e-9 else leg / length

    def _waypoint_reached(self, pos: np.ndarray, index: int) -> bool:
        if float(np.linalg.norm(self.wp[index] - pos)) <= float(
                self.cfg.waypoint_arrive_radius):
            return True
        # A 20 Hz tick at real speed steps straight over a small arrival ball, so
        # also accept the waypoint once the drone is past the plane through it
        # perpendicular to its leg. Proximity alone would strand the target behind
        # the drone forever and the cross-track error would then grow without bound.
        # Bounded by the capture radius: without it, one teleported fix sitting far
        # past the plane would consume waypoints it never flew.
        if float(np.linalg.norm(self.wp[index] - pos)) > float(
                self.cfg.progress_jump_slack):
            return False
        direction = self._leg_direction(index)
        if direction is None:
            return True
        reference = self.wp[index] if index >= 1 else self.wp[0]
        return float(np.dot(pos - reference, direction)) >= 0.0

    def _advance_to_reached(self, pos: np.ndarray,
                            pose_stamp: float | None = None) -> None:
        """Consume at most one reached waypoint per distinct pose. Never regress."""
        if pose_stamp is not None:
            if (self._last_sequence_pose_stamp is not None
                    and pose_stamp <= self._last_sequence_pose_stamp):
                return
            self._last_sequence_pose_stamp = pose_stamp
        last = len(self.wp) - 1
        self.holding_for_inspection = False
        # Bounded per tick: on a route that doubles back, one pose can sit "past"
        # the plane of many later waypoints at once, and an unbounded loop would
        # let a single jump declare most of the route flown.
        budget = int(self.cfg.waypoint_advance_budget)
        confirmed = False
        while self.target_index < last and budget > 0:
            if not self._waypoint_reached(pos, self.target_index):
                self._arrive_frames = 0
                break
            if not confirmed:
                self._arrive_frames += 1
                if self._arrive_frames < int(self.cfg.waypoint_arrive_confirm_frames):
                    # Inside the sphere, but not for long enough yet. One stray fix
                    # landing in it must not retire a waypoint the drone never reached.
                    break
            if (self.target_index in self.inspect_waypoints
                    and self.target_index not in self.completed_inspections):
                # Reached an inspection point: hold the target here until the
                # capture is acknowledged, or advancing would fly away mid-shot.
                self.holding_for_inspection = True
                break
            # The configured budget is exactly one. Keeping this bounded loop
            # makes that invariant explicit and preserves an alignment phase for
            # the next clicked point even after a slow or relocated pose update.
            self.target_index += 1
            self._arrive_frames = 0
            confirmed = True
            budget -= 1
        # Kept in the LEG sense (the segment being flown), which is what the PCMD
        # controller keys its per-waypoint yaw realignment on.
        self.active_segment = max(0, self.target_index - 1)
        self.progress_s = float(self.cum[self.target_index])

    def seed_goal(self, pos: np.ndarray) -> np.ndarray:
        """Waypoint to seed the heading estimate from.

        NOT current_target: the arrival confirmation window means target_index can
        still point at a waypoint the drone is already sitting on (or has just
        passed), and seeding the heading from that yields a meaningless -- often
        reversed -- direction that the yaw controller then spends the whole flight
        correcting.
        """
        pos = _finite_vec3(pos, "position")
        index = self.target_index
        last = len(self.wp) - 1
        # Reuse the sequencer's own reached test rather than a bare radius check:
        # tuning waypoint_arrive_radius down otherwise stops this from skipping a
        # waypoint the drone has already flown past, and the seeded heading then
        # points backwards along the route for the rest of the flight.
        while index < last and self._waypoint_reached(pos, index):
            index += 1
        return self.wp[index].copy()

    def current_target(self, pos: np.ndarray) -> np.ndarray:
        """The clicked waypoint the drone is currently flying at."""
        self._advance_to_reached(_finite_vec3(pos, "position"))
        return self.wp[self.target_index].copy()

    def _auto_goal(self, pos: np.ndarray, pose_stamp: float, now: float):
        """Return (goal, target_index, path_error, progress, action).

        Every tick re-draws the line from the CURRENT localized position to the
        current waypoint, so drift is corrected continuously without a separate
        REJOIN mode: flying straight at the point is the rejoin.
        """
        self._advance_to_reached(pos, pose_stamp)
        last = len(self.wp) - 1
        goal = self.wp[self.target_index].copy()
        # Cross-track against the leg being flown, so the route-corridor failsafe
        # still bites even though the controller no longer projects onto the path.
        leg_start = self.wp[max(0, self.target_index - 1)]
        path_error, _nearest, _t = point_segment_distance(pos, leg_start, goal)
        progress = float(self.target_index) / float(max(1, last))
        if (self.target_index < last and self._arrive_frames > 0
                and not self.holding_for_inspection):
            # Inside the arrival sphere, waiting out the confirmation window. The
            # sphere IS the arrival criterion, so settle here rather than chase the
            # exact centre -- on a sphere this large the centre can be BEHIND the
            # drone, and chasing it commands a turn-around every single waypoint.
            #
            # NOT while holding for inspection: _advance_to_reached breaks into that
            # hold with _arrive_frames already at the confirmation count and never
            # clears it, so this gate latched and returned a zero-velocity ARRIVING
            # forever. A drone that crossed the waypoint's plane while further from
            # the pole than inspect_radius could then never close in, never inspect
            # and never abort -- no watchdog covers it, because the inspection
            # timeout only arms once look_at_pole has been non-None at least once.
            return goal, self.target_index, path_error, progress, "ARRIVING"
        if self.target_index >= last:
            if self._final_hold_complete(pos, pose_stamp, now):
                pending = any(index not in self.completed_inspections
                              for index in self.inspect_waypoints)
                return (goal, self.target_index, path_error, progress,
                        "ABORT_PENDING_INSPECTION" if pending else "LAND")
            if float(np.linalg.norm(goal - pos)) <= self.final_arrive_tolerance():
                return goal, self.target_index, path_error, progress, "FINAL_HOLD"
        else:
            self._reset_final_hold()
        return goal, self.target_index, path_error, progress, "FOLLOW"

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
        goal, seg, path_error, progress, action = self._auto_goal(
            pos, float(pose.stamp), now
        )

        look = self._look_at_pole_target(pos)
        if (look is None and self.holding_for_inspection
                and float(np.linalg.norm(goal - pos))
                <= float(self.cfg.waypoint_arrive_radius)):
            # Parked ON an inspection waypoint that yields no look-at target (pole
            # out of range, or already consumed). Holding forever would sit there
            # until an unrelated failsafe fired, so abort explicitly instead.
            # Gated on the ARRIVAL SPHERE, the same notion of "at the waypoint" that
            # set holding_for_inspection -- not on the PCMD dead zone, which is now
            # far tighter and would leave the hold unbounded.
            self.state = "LANDING"
            return Command("ABORT", np.zeros(3), pose.yaw, goal, path_error, progress,
                           should_land=True,
                           status="inspection waypoint has no reachable target -> abort/land")
        if look:
            wpnum, pole_center, pole_id = look
            look_meta = {"waypoint": wpnum, "pole_id": pole_id,
                         "target": pole_center.tolist(), "capture_ack_required": True,
                         "body_yaw_required": bool(self.cfg.pole_body_look)}
            yaw_target = (heading_from_to(pos, pole_center, self.cfg.map_frame)
                          if self.cfg.pole_body_look else float(pose.yaw))
            self.state = "INSPECT"
            return Command("INSPECT", np.zeros(3), yaw_target, pos.copy(), path_error, progress,
                           look_at_pole=look_meta, should_land=False,
                           status=f"INSPECT wp={wpnum} pole={pole_id} awaiting gimbal/capture ack")

        if action == "ARRIVING":
            self.state = "CRUISE"
            return Command(
                "ARRIVING", np.zeros(3), float(pose.yaw), goal, path_error, progress,
                status=(f"waypoint {self.target_index + 1} arrival confirm "
                        f"{self._arrive_frames}/{self.cfg.waypoint_arrive_confirm_frames}"),
            )
        if action == "FINAL_HOLD":
            self.state = "CRUISE"
            return Command(
                "FINAL_HOLD", np.zeros(3), float(pose.yaw), goal,
                path_error, progress,
                status=(f"final hold {self._final_confirm_frames}/"
                        f"{self.cfg.final_arrive_confirm_frames}"),
            )
        if action == "ABORT_PENDING_INSPECTION":
            self.state = "LANDING"
            return Command("ABORT", np.zeros(3), pose.yaw, goal, path_error, progress,
                           should_land=True,
                           status="pending inspection missed at route end -> abort/land")
        if action == "LAND":
            self.state = "LANDING"
            return Command("LAND", np.zeros(3), pose.yaw, goal, path_error, progress,
                           should_land=True, status="final path reached -> LAND")

        yaw_target = heading_from_to(pos, goal, self.cfg.map_frame)
        look_meta = None
        status = action

        d = goal - pos
        dist = float(np.linalg.norm(d))
        vel = np.zeros(3, dtype=float) if dist < 1e-9 else d / dist
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

    delta = goal - pose.xyz
    if inspection:
        yaw_error = wrap_angle(float(cmd.yaw_target) - float(pose.yaw))
    else:
        try:
            projected_error = ground_projected_yaw_error(
                float(pose.yaw), delta, cfg.map_frame
            )
        except (TypeError, ValueError, OverflowError):
            return zero
        # No horizontal target ray exists for a purely vertical move. In that
        # case yaw is intentionally left unchanged and only the body decomposition
        # below may produce a vertical command.
        yaw_error = 0.0 if projected_error is None else projected_error
    yaw_error_deg = math.degrees(yaw_error)
    if inspection or (
        require_yaw_alignment
        and projected_error is not None
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

    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= cfg.translation_arrival_tolerance:
        return zero

    # Split into body forward/right/up using the site's MEASURED horizontal plane.
    # Assuming X/Z is horizontal costs 22.5 deg on target_site_v1, which leaks
    # commanded climb into horizontal motion and vice versa.
    forward, right, up = cfg.map_frame.body_components(delta, float(pose.yaw))
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
        self.alignment_started_at: float | None = None
        self.abort_reason: str | None = None
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
            self.alignment_started_at = now

        if cmd.action not in {"FOLLOW", "REJOIN"} or cmd.should_land:
            self.reset()
            return 0, 0, 0, 0

        try:
            goal = _finite_vec3(cmd.goal, "command goal")
            pose_stamp = float(pose.stamp)
            if not math.isfinite(pose_stamp):
                raise ValueError("non-finite pose timestamp")
            projected_error = ground_projected_yaw_error(
                float(pose.yaw), goal - pose.xyz, self.config.map_frame
            )
            yaw_error = 0.0 if projected_error is None else projected_error
        except (AttributeError, TypeError, ValueError, OverflowError):
            self.reset()
            return 0, 0, 0, 0

        fresh_alignment_sample = (
            self.previous_timestamp is None
            or pose_stamp > self.previous_timestamp
        )
        yaw_rate_deg_s = None
        if (fresh_alignment_sample and self.previous_timestamp is not None
                and self.previous_yaw is not None):
            dt = pose_stamp - self.previous_timestamp
            if dt > 1e-6:
                yaw_rate_deg_s = math.degrees(
                    wrap_angle(float(pose.yaw) - self.previous_yaw)
                ) / dt
        if fresh_alignment_sample:
            self.previous_timestamp = pose_stamp
            self.previous_yaw = float(pose.yaw)

        if projected_error is None:
            self.aligned_for_translation = True

        if not self.aligned_for_translation:
            if self.abort_reason is not None:
                self.phase = "yaw_alignment_timeout"
                return 0, 0, 0, 0
            if self.alignment_started_at is None:
                self.alignment_started_at = now
            if now - self.alignment_started_at >= self.config.yaw_alignment_timeout_s:
                self.abort_reason = (
                    f"waypoint yaw alignment exceeded "
                    f"{self.config.yaw_alignment_timeout_s:.1f}s"
                )
                self.phase = "yaw_alignment_timeout"
                return 0, 0, 0, 0
            stable = (
                fresh_alignment_sample
                and abs(math.degrees(yaw_error)) <= self.config.yaw_tolerance_deg
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
    ctrl = RouteAutoController(wp, poles, config_for_route(args.path))
    print(f"loaded waypoints={len(wp)} poles={len(poles)} inspect_valid={sorted(i+1 for i in ctrl.inspect_waypoints)}")
    print("This module outputs commands; it does not connect to or fly the drone.")
