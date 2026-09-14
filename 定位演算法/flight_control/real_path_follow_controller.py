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

- Start at waypoint 1 and fly the authored points in order (1 -> 2 -> ...),
  even when takeoff is closer to a later waypoint. The leg from takeoff to
  waypoint 1 is flown direct; segment guidance engages after that.
- A local guidance point keeps position and height on the segment; excessive
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
    map_confirmed: bool = True

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Command:
    """Map-frame command produced by the controller.

    `vel_map` is the UNIT direction toward the guidance point in GLOMAP map units.
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
    # The authored waypoint stays in goal for arrival checks and logging.
    # Segment guidance controls the approach without changing that waypoint.
    guidance_goal: Optional[np.ndarray] = None


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


def camera_heading_from_forward(
    camera_forward,
    map_frame: MapFrame,
    *,
    minimum_horizontal_component: float = 0.1,
) -> float | None:
    """Project the camera optical axis onto the map ground plane.

    A nearly vertical optical axis has no reliable left/right heading.  Returning
    ``None`` keeps AUTO hovering instead of silently treating ``atan2(0, 0)`` as
    a valid zero-degree nose direction.
    """
    forward = _finite_vec3(camera_forward, "camera forward")
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    forward = forward / norm
    if map_frame.horizontal_distance(forward) < float(minimum_horizontal_component):
        return None
    heading = map_frame.heading(forward)
    return heading if math.isfinite(heading) else None


def load_map_frame(path: str | Path) -> MapFrame:
    """Load the measured gravity direction from a site's T_align_gravity.json."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
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
        raise ValueError(f"map alignment R row 2 disagrees with gravity_glomap: {source}")
    return frame


_CONTROL_FLOAT_LIMITS = {
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
    "mid_leg_realign_deg": (0.0, 90.0),
    "yaw_alignment_max_rate_deg_s": (0.0, 180.0),
    "yaw_command_period_s": (0.0, 10.0),
    "yaw_alignment_timeout_s": (0.0, 120.0),
    "minimum_yaw_alignment_distance": (0.0, 100.0),
    "slowdown_distance": (0.0, 100.0),
    "translation_arrival_tolerance": (0.0, 100.0),
    "waypoint_arrive_radius": (0.0, 100.0),
    "final_hold_s": (0.0, 30.0),
}

_CONTROL_INTEGER_LIMITS = (
    "yaw_alignment_confirmation_updates",
    "max_translation_pcmd",
    "max_vertical_pcmd",
    "max_yaw_pcmd",
    "waypoint_arrive_confirm_frames",
    "waypoint_advance_budget",
    "final_arrive_confirm_frames",
    "yaw_alignment_max_windows",
)


@dataclass
class ControlConfig:
    # NOTE: there is deliberately no `speed` here. The map is scale-free, so a
    # map-units/sec figure cannot be converted into a tilt percentage. Commanded
    # speed is set by max_translation_pcmd (a % of the firmware MaxTilt /
    # MaxVerticalSpeed) and shaped by slowdown_distance. A `speed` field existed
    # until 2026-08-06 but only ever scaled Command.vel_map, which the PCMD
    # adapter ignores -- tuning it changed nothing about how the drone flew.
    lookahead: float = 0.8  # map-units ahead on route after rejoin
    rejoin_tol: float = 0.45  # cross-track error before REJOIN
    # Map-unit arrival tolerance. Scale-dependent like radius/max_jump: 0.1 was
    # chosen against urai EDM v1, whose camera track spans 6.07 map units and
    # whose consecutive references sit 0.027 apart. Re-derive it per site.
    arrive: float = 0.1  # absolute cap for end tolerance
    arrive_fraction: float = 0.02  # end tolerance <= 2% of route length
    min_arrive: float = 0.1  # but never smaller than this map-unit value
    inspect_waypoints: tuple[int, ...] = (9, 10, 11, 15)  # 1-based labels in current 15-point route
    inspect_radius: float = 0.75  # map-unit trigger around each label
    inspect_resume_margin: float = 0.25  # arclength margin after last inspected wp
    inspect_yaw_tolerance_deg: float = 6.0
    max_pose_age_s: float = 0.5  # stale visual pose -> HOVER
    segment_window: int = 2  # active segment +/- this many route segments
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
    yaw_tolerance_deg: float = 7.5
    yaw_alignment_confirmation_updates: int = 3
    yaw_alignment_max_rate_deg_s: float = 5.0
    yaw_command_period_s: float = 1.0
    #: Per-window budget for one yaw-alignment attempt. A full course
    #: reversal needs most of it at the capped yaw rate; the attempt is
    #: retried (not latched) until yaw_alignment_max_windows is exhausted.
    yaw_alignment_timeout_s: float = 6.0
    # Retained for profile compatibility. Near-point centering uses the authored
    # arrival radius rather than this legacy distance.
    minimum_yaw_alignment_distance: float = 0.25
    max_translation_pcmd: int = 10
    max_vertical_pcmd: int = 10
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
    #: Optional per-waypoint arrival spheres, index-aligned with the route.
    #: A route drawn with per-point radii sets this via config_for_route;
    #: entries must each clear translation_arrival_tolerance like the global
    #: sphere. None means every waypoint uses waypoint_arrive_radius.
    waypoint_arrive_radii: tuple[float, ...] | None = None
    #: Fly the authored route out, then back along it to waypoint 1 and land
    #: there instead of landing at the last waypoint. See _return_to_start_route.
    return_to_start: bool = False
    #: Legacy/default confirmation count. A route-authored arrival sphere overrides
    #: this to one update: entering the drawn sphere is the arrival condition.
    waypoint_arrive_confirm_frames: int = 3
    #: Exactly one waypoint may retire per pose update. This preserves a distinct
    #: hover/turn gate for every clicked point, including tightly spaced points.
    waypoint_advance_budget: int = 1
    #: End-of-route LAND needs several distinct fresh captures over a dwell period.
    final_arrive_confirm_frames: int = 5
    final_hold_s: float = 1.0
    #: Measured map basis. The default is the legacy X/Z-horizontal, -Y-up guess.
    map_frame: MapFrame = LEGACY_MAP_FRAME
    #: Consecutive alignment windows before a never-converging turn stops
    #: commanding and holds for the operator. 1 restores the old latch-at-once
    #: behavior; the default 3 lets a slow-but-converging reversal finish.
    yaw_alignment_max_windows: int = 3
    #: Mid-leg heading drift that re-opens the turn gate, in degrees. Small
    #: drift stays on strafing correction; this many degrees off the target
    #: ray for yaw_alignment_confirmation_updates consecutive fresh samples
    #: parks translation and turns again. Must clear yaw_tolerance_deg or the
    #: gate would chatter at the alignment boundary.
    mid_leg_realign_deg: float = 25.0

    def __post_init__(self):
        _validate_control_float_limits(self)
        _validate_control_shape(self)
        _validate_arrival_contract(self)
        _validate_map_frame(self)
        _validate_control_integer_limits(self)
        _validate_realign_threshold(self)


DESKTOP_AUTO_MAX_TRANSLATION_PCMD = 3
DESKTOP_AUTO_MAX_VERTICAL_PCMD = 20
DESKTOP_AUTO_MAX_YAW_PCMD = 50
DESKTOP_AUTO_YAW_ALIGNMENT_TIMEOUT_S = 30.0
DESKTOP_AUTO_YAW_TOLERANCE_DEG = 20.0

# Horizontal tilt remains capped at 3%; telemetry enforces the physical speed
# limit. Gaz has its own speed-percentage cap. Turns keep horizontal PCMD at
# zero while allowing height correction; route recovery holds yaw instead.


def production_auto_control_config(map_frame: MapFrame) -> ControlConfig:
    """Shared AUTO ControlConfig for desktop, dry-run, and the offline sim."""
    return ControlConfig(
        inspect_waypoints=(),
        map_frame=map_frame,
        max_translation_pcmd=DESKTOP_AUTO_MAX_TRANSLATION_PCMD,
        # Gaz is a percentage of vertical speed, unlike pitch/roll tilt.
        # 20% of the desktop's 2 m/s envelope permits up to 0.4 m/s vertically.
        max_vertical_pcmd=DESKTOP_AUTO_MAX_VERTICAL_PCMD,
        max_yaw_pcmd=DESKTOP_AUTO_MAX_YAW_PCMD,
        yaw_tolerance_deg=DESKTOP_AUTO_YAW_TOLERANCE_DEG,
        yaw_alignment_timeout_s=DESKTOP_AUTO_YAW_ALIGNMENT_TIMEOUT_S,
        return_to_start=True,
    )


def _validate_control_float_limits(config: ControlConfig) -> None:
    for name, (lo, hi) in _CONTROL_FLOAT_LIMITS.items():
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= lo or value > hi:
            raise ValueError(f"{name} must be finite and in ({lo}, {hi}], got {value!r}")


def _validate_control_shape(config: ControlConfig) -> None:
    if not isinstance(config.inspect_waypoints, tuple) or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 1
        for index in config.inspect_waypoints
    ):
        raise ValueError("inspect_waypoints must be a tuple of positive 1-based integers")
    if (
        isinstance(config.segment_window, bool)
        or not isinstance(config.segment_window, int)
        or not 0 <= config.segment_window <= 20
    ):
        raise ValueError("segment_window must be an integer in [0,20]")


def _validate_arrival_contract(config: ControlConfig) -> None:
    # The adapter must keep commanding until the drone is inside the arrival sphere,
    # or the sequencer can never retire the waypoint and the route stalls.
    if float(config.translation_arrival_tolerance) >= float(config.waypoint_arrive_radius):
        raise ValueError(
            "translation_arrival_tolerance must be smaller than "
            f"waypoint_arrive_radius ({config.translation_arrival_tolerance} >= "
            f"{config.waypoint_arrive_radius}); autonomy would stall short of "
            "every waypoint"
        )
    final_floor = min(float(config.arrive), float(config.min_arrive))
    if final_floor <= float(config.translation_arrival_tolerance):
        raise ValueError(
            "the end-of-route arrival window must be larger than "
            f"translation_arrival_tolerance ({final_floor} <= "
            f"{config.translation_arrival_tolerance}); the final waypoint would "
            "never terminate the mission"
        )
    # The capture bound cannot be tighter than the arrival sphere, or a waypoint
    # could be accepted and simultaneously rejected as too far away.
    if float(config.progress_jump_slack) < float(config.waypoint_arrive_radius):
        raise ValueError(
            "progress_jump_slack must be >= waypoint_arrive_radius "
            f"({config.progress_jump_slack} < {config.waypoint_arrive_radius})"
        )
    radii = config.waypoint_arrive_radii
    if radii is not None:
        if not isinstance(radii, tuple) or not radii:
            raise ValueError("waypoint_arrive_radii must be a non-empty tuple or None")
        for index, radius in enumerate(radii):
            if (
                isinstance(radius, bool)
                or not isinstance(radius, (int, float))
                or not math.isfinite(float(radius))
                or float(radius) <= float(config.translation_arrival_tolerance)
            ):
                raise ValueError(
                    f"waypoint_arrive_radii[{index}] must clear "
                    "translation_arrival_tolerance "
                    f"({radius!r} <= {config.translation_arrival_tolerance}); "
                    "autonomy would stall short of that waypoint"
                )


def _validate_map_frame(config: ControlConfig) -> None:
    if not isinstance(config.map_frame, MapFrame):
        raise ValueError("map_frame must be a MapFrame")


def _validate_control_integer_limits(config: ControlConfig) -> None:
    for name in _CONTROL_INTEGER_LIMITS:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
            raise ValueError(f"{name} must be an integer in [1,100]")
    if config.waypoint_advance_budget != 1:
        raise ValueError("waypoint_advance_budget must be exactly 1")


def _validate_realign_threshold(config: ControlConfig) -> None:
    if float(config.mid_leg_realign_deg) <= float(config.yaw_tolerance_deg):
        raise ValueError(
            "mid_leg_realign_deg must clear yaw_tolerance_deg "
            f"({config.mid_leg_realign_deg} <= {config.yaw_tolerance_deg}); "
            "the re-alignment gate would chatter at the alignment boundary"
        )


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
    return [
        np.array(point, dtype=float, copy=True) for point in _validated_route_waypoints(waypoints)
    ]


def aligned_to_glomap(p, frame: MapFrame = LEGACY_MAP_FRAME) -> np.ndarray:
    """Compatibility adapter to the shared route-domain conversion."""
    return _domain_aligned_to_glomap(p, frame)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_from_to(a: np.ndarray, b: np.ndarray, frame: MapFrame = LEGACY_MAP_FRAME) -> float:
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


def _path_length(points) -> float | None:
    """Drawn-route length in map units, or None when unmeasurable."""
    try:
        vectors = [np.asarray(point, dtype=float).reshape(3) for point in points]
    except (TypeError, ValueError):
        return None
    try:
        total = sum(float(np.linalg.norm(b - a)) for a, b in zip(vectors, vectors[1:]))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(total) or total <= 0.0:
        return None
    return total


def config_for_route(
    path_json: str | Path | MissionRouteSnapshot, base: "ControlConfig | None" = None
) -> "ControlConfig":
    """Apply route-authored limits without rereading a validated snapshot."""
    cfg = ControlConfig() if base is None else base
    if isinstance(path_json, MissionRouteSnapshot):
        radius = path_json.arrive_radius_map_units
        point_radii = path_json.waypoint_arrive_radii
    else:
        # Only the scalar limits are wanted here, but parsing the document also
        # converts its waypoints, and that needs the frame the route was authored
        # in. cfg already carries it; leaving it out silently substituted the
        # legacy [x,z,-y] assumption and rejected every route the editor stamped
        # align_source='measured' -- i.e. every route at a site that has a
        # measured T_align_gravity.json.
        route = RouteDocument.from_path(path_json, require_map_units=True, map_frame=cfg.map_frame)
        radius = route.arrive_radius_map_units
        point_radii = route.waypoint_arrive_radii
    if isinstance(path_json, MissionRouteSnapshot):
        drawn_points = list(path_json.waypoints)
    else:
        drawn_points = list(route.controller_points)
    path_len = _path_length(drawn_points)
    changes = {}
    if radius is not None:
        # progress_jump_slack must stay >= the arrival sphere (__post_init__
        # enforces it), so widen the capture bound rather than reject the radius.
        changes.update(
            waypoint_arrive_radius=radius,
            # Operator decision 2026-09-13: entering the drawn sphere retires
            # the waypoint immediately (touch counts). A dwell was tried the
            # same day and reverted: with small operator-drawn spheres the
            # settle wait costs more than the stray-fix protection is worth.
            waypoint_arrive_confirm_frames=1,
            progress_jump_slack=max(float(cfg.progress_jump_slack), radius),
        )
        # The 1.5 default slowdown was derived for 5-6 unit routes. On a 1.03
        # unit route it caps every in-corridor fix at 1-2% PCMD, which cannot
        # close a 0.3-0.5 unit gap against wind (measured 2026-09-13: 140 s of
        # translate at pcmd 1 with the error stuck at ~0.38). Scale it to the
        # drawn route so full authority resumes beyond ~1/4 route length.
        if path_len is not None:
            changes["slowdown_distance"] = min(
                float(cfg.slowdown_distance), max(0.15, 0.25 * path_len)
            )
    if point_radii is not None:
        # Per-point spheres replace the global one index-wise; the arrival
        # contract validator still guards every entry, and entering a drawn
        # sphere is an immediate arrival like the scalar route radius.
        changes.update(
            waypoint_arrive_radii=point_radii,
            waypoint_arrive_confirm_frames=1,
        )
        widest = max(float(value) for value in point_radii)
        changes["progress_jump_slack"] = max(float(cfg.progress_jump_slack), widest)
    if not changes:
        return cfg
    return replace(cfg, **changes)


def load_poles(poles_json: str | Path, map_frame: MapFrame = LEGACY_MAP_FRAME) -> list[dict]:
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
        out.append(
            {
                "id": i,
                "center": center_g,
                "base": base_g,
                "top": top_g,
                "raw": pole,
            }
        )
    return out


def nearest_pole(poles: list[dict], ref: np.ndarray, map_frame: MapFrame = LEGACY_MAP_FRAME):
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
# Main controller


def _return_to_start_route(
    waypoints: list[np.ndarray], config: ControlConfig
) -> tuple[list[np.ndarray], ControlConfig]:
    """Append the reverse pass so the route ends back at waypoint 1.

    The authored waypoints keep their indices, so the waypoint-1 start, the
    inspection labels and the per-point arrival spheres all still mean what they
    meant on the drawn route; the mirrored tail simply re-visits them in reverse
    order.  Expanding the waypoint list -- rather than teaching the controller to
    walk backwards -- keeps ``target_index`` monotonic and leaves arclength,
    progress and the route-deviation corridor derived from one polyline.

    The turnaround waypoint is not repeated, so the aircraft does not have to
    re-confirm arrival at the point it is already sitting on.
    """
    outbound = list(waypoints)
    if len(outbound) < 2:
        return outbound, config
    full = outbound + [np.array(point, dtype=float, copy=True) for point in reversed(outbound[:-1])]
    radii = config.waypoint_arrive_radii
    if radii is None:
        return full, config
    drawn = tuple(radii)
    return full, replace(config, waypoint_arrive_radii=drawn + tuple(reversed(drawn[:-1])))


class RouteAutoController:
    """Path-follow + look-at-pole controller for real deployment wiring."""

    def __init__(
        self,
        waypoints: list[np.ndarray],
        poles: list[dict] | None = None,
        config: ControlConfig | None = None,
    ):
        self.cfg = config or ControlConfig()
        checked_waypoints = _validated_waypoints(waypoints)
        if self.cfg.return_to_start:
            checked_waypoints, self.cfg = _return_to_start_route(checked_waypoints, self.cfg)
        frozen_waypoints = []
        for point in checked_waypoints:
            copied = np.array(point, dtype=float, copy=True)
            copied.setflags(write=False)
            frozen_waypoints.append(copied)
        self.wp = tuple(frozen_waypoints)
        self.cum = path_arclengths(self.wp)
        self.path_len = float(self.cum[-1])
        radii = self.cfg.waypoint_arrive_radii
        if radii is not None and len(radii) != len(self.wp):
            raise ValueError(
                f"waypoint_arrive_radii has {len(radii)} entries for "
                f"{len(self.wp)} waypoints; radii must be per-waypoint aligned"
            )
        invalid_inspections = [i for i in self.cfg.inspect_waypoints if i > len(self.wp)]
        if invalid_inspections:
            raise ValueError(
                f"inspection waypoint labels exceed route length {len(self.wp)}: {invalid_inspections}"
            )
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
        self._join_target_index = 0
        self._rejoining = False

    def start_after_nearest_waypoint(self, pos: np.ndarray) -> int:
        """Start at waypoint 1, even when the takeoff pose is nearer a later point.

        Kept under the historical name because desktop autonomy, the offline sim,
        and this repo's tests all call it. The position only validates; the
        target is always waypoint index 0, so every authored waypoint is flown in
        order. Only ``_advance_to_reached`` retires a waypoint.
        """
        _finite_vec3(pos, "takeoff position")
        self.target_index = 0
        self._join_target_index = 0
        self._rejoining = False
        self.active_segment = 0
        self.progress_s = 0.0
        self.holding_for_inspection = False
        self._arrive_frames = 0
        self._last_sequence_pose_stamp = None
        self._reset_final_hold()
        return 0

    def _arrive_radius_for(self, index: int) -> float:
        """Arrival sphere for one waypoint: drawn per-point value or global."""
        radii = self.cfg.waypoint_arrive_radii
        if radii is not None and 0 <= int(index) < len(radii):
            return float(radii[int(index)])
        return float(self.cfg.waypoint_arrive_radius)

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
        radii = self.cfg.waypoint_arrive_radii
        if radii is not None and len(radii) == len(self.wp):
            # The operator drew this sphere for the final waypoint: honor it.
            return float(radii[-1])
        return min(float(self.cfg.waypoint_arrive_radius), scaled)

    def _reset_final_hold(self) -> None:
        self._final_hold_started = None
        self._final_pose_stamp = None
        self._final_confirm_frames = 0

    def reset_arrival_confirmation(self) -> None:
        """Discard pre-outage evidence without reselecting or advancing a waypoint."""
        self._arrive_frames = 0
        self._last_sequence_pose_stamp = None
        self._reset_final_hold()
        self._rejoining = False  # Recompute the active segment error on recovery.
        if self.state == "LANDING":
            self.state = "CRUISE"  # A pre-outage LAND decision needs new arrival evidence.

    def _final_hold_complete(self, pos: np.ndarray, pose_stamp: float, now: float) -> bool:
        if float(np.linalg.norm(self.wp[-1] - pos)) > self.final_arrive_tolerance():
            self._reset_final_hold()
            return False
        # Reprocessing one captured frame must never accumulate landing evidence.
        if self._final_pose_stamp is not None and pose_stamp <= self._final_pose_stamp:
            return False
        if self._final_pose_stamp is not None and pose_stamp - self._final_pose_stamp > float(
            self.cfg.max_pose_age_s
        ):
            self._reset_final_hold()
            self._final_hold_started = now
            self._final_confirm_frames = 1
            self._final_pose_stamp = pose_stamp
            return False
        if self._final_hold_started is None:
            self._final_hold_started = now
            self._final_confirm_frames = 1
        else:
            self._final_confirm_frames += 1
        self._final_pose_stamp = pose_stamp
        return self._final_confirm_frames >= int(
            self.cfg.final_arrive_confirm_frames
        ) and now - self._final_hold_started >= float(self.cfg.final_hold_s)

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
        pole_id, pole, pole_dist = nearest_pole(self.poles, self.wp[idx], self.cfg.map_frame)
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

    def ack_inspection(self, metadata: dict | None, orientation_confirmed: bool = False) -> bool:
        """Mark only the currently pending inspection complete after capture acknowledgement."""
        if not orientation_confirmed or not isinstance(metadata, dict) or self.last_inspect is None:
            return False
        waypoint = metadata.get("waypoint")
        pole_id = metadata.get("pole_id")
        if waypoint != self.last_inspect.get("waypoint") or pole_id != self.last_inspect.get(
            "pole_id"
        ):
            return False
        idx = int(waypoint) - 1
        if idx not in self.inspect_waypoints:
            return False
        self.completed_inspections.add(idx)
        self.last_inspect = None
        return True

    def _waypoint_reached(self, pos: np.ndarray, index: int) -> bool:
        # Overshooting outside the authored sphere keeps this waypoint as the
        # target, so the controller corrects back toward it instead of skipping it.
        return float(np.linalg.norm(self.wp[index] - pos)) <= self._arrive_radius_for(index)

    def waypoint_reached(self, pos: np.ndarray, index: int) -> bool:
        """Return whether the position is inside this waypoint's arrival sphere."""
        return self._waypoint_reached(_finite_vec3(pos, "position"), int(index))

    def _advance_to_reached(
        self, pos: np.ndarray, pose_stamp: float | None = None, *, allow_arrival: bool = True
    ) -> None:
        """Consume at most one reached waypoint per distinct pose. Never regress."""
        if pose_stamp is not None:
            if (
                self._last_sequence_pose_stamp is not None
                and pose_stamp <= self._last_sequence_pose_stamp
            ):
                return
            if (
                self._last_sequence_pose_stamp is not None
                and pose_stamp - self._last_sequence_pose_stamp > float(self.cfg.max_pose_age_s)
            ):
                self._arrive_frames = 0
            self._last_sequence_pose_stamp = pose_stamp
        if not allow_arrival:
            self._arrive_frames = 0
            return
        last = len(self.wp) - 1
        self.holding_for_inspection = False
        # Overlapping arrival spheres must still retire at most one waypoint
        # per distinct pose, preserving each waypoint's alignment phase.
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
            if (
                self.target_index in self.inspect_waypoints
                and self.target_index not in self.completed_inspections
            ):
                # Reached an inspection point: hold the target here until the
                # capture is acknowledged, or advancing would fly away mid-shot.
                self.holding_for_inspection = True
                break
            # The configured budget is exactly one. Keeping this bounded loop
            # makes that invariant explicit and preserves an alignment phase for
            # the next clicked point even after a slow or relocated pose update.
            self.target_index += 1
            self._rejoining = False
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
        still point at a waypoint the drone is already sitting on, and seeding
        the heading from that yields a meaningless -- often
        reversed -- direction that the yaw controller then spends the whole flight
        correcting.
        """
        pos = _finite_vec3(pos, "position")
        index = self.target_index
        last = len(self.wp) - 1
        # Skip only targets whose arrival sphere contains this position, using
        # the same per-waypoint radius as the sequencer.
        while index < last and self._waypoint_reached(pos, index):
            index += 1
        return self.wp[index].copy()

    def current_target(self, pos: np.ndarray) -> np.ndarray:
        """The clicked waypoint the drone is currently flying at."""
        self._advance_to_reached(_finite_vec3(pos, "position"))
        return self.wp[self.target_index].copy()

    def _auto_goal(self, pos: np.ndarray, pose_stamp: float, now: float, *, map_confirmed=True):
        """Return (goal, target_index, path_error, progress, action).

        Arrival always uses the authored waypoint. step() supplies segment
        guidance and route recovery separately from waypoint sequencing.
        """
        self._advance_to_reached(pos, pose_stamp, allow_arrival=map_confirmed)
        last = len(self.wp) - 1
        goal = self.wp[self.target_index].copy()
        # Cross-track against the active leg also drives segment recovery.
        leg_start = self.wp[max(0, self.target_index - 1)]
        path_error, _nearest, _t = point_segment_distance(pos, leg_start, goal)
        progress = float(self.target_index) / float(max(1, last))
        if not map_confirmed:
            self._reset_final_hold()
            if self._waypoint_reached(pos, self.target_index):
                return goal, self.target_index, path_error, progress, "WAIT_MAP_CONFIRMATION"
        if self.target_index < last and self._arrive_frames > 0 and not self.holding_for_inspection:
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
                pending = any(
                    index not in self.completed_inspections for index in self.inspect_waypoints
                )
                return (
                    goal,
                    self.target_index,
                    path_error,
                    progress,
                    "ABORT_PENDING_INSPECTION" if pending else "LAND",
                )
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
        pose_ok = bool(
            pose is not None
            and math.isfinite(now)
            and all(math.isfinite(float(v)) for v in pose_values)
        )
        age = float("inf") if not pose_ok else now - float(pose.stamp)
        if not pose_ok or age < -0.05 or age > self.cfg.max_pose_age_s:
            self.state = "HOVER"
            return Command(
                "HOVER",
                np.zeros(3),
                0.0,
                np.zeros(3),
                float("inf"),
                0.0,
                status="pose missing/stale -> HOVER",
            )

        pos = pose.xyz
        goal, seg, path_error, progress, action = self._auto_goal(
            pos, float(pose.stamp), now, map_confirmed=bool(getattr(pose, "map_confirmed", True))
        )

        if action == "WAIT_MAP_CONFIRMATION":
            self.state = "HOVER"
            return Command(
                "HOVER", np.zeros(3), pose.yaw, goal, path_error, progress,
                status="WAIT_MAP_CONFIRMATION",
            )

        look = self._look_at_pole_target(pos)
        if (
            look is None
            and self.holding_for_inspection
            and float(np.linalg.norm(goal - pos)) <= self._arrive_radius_for(self.target_index)
        ):
            # Parked ON an inspection waypoint that yields no look-at target (pole
            # out of range, or already consumed). Hold here without turning the
            # missing target into an automatic landing.
            # Gated on the ARRIVAL SPHERE, the same notion of "at the waypoint" that
            # set holding_for_inspection -- not on the PCMD dead zone, which is now
            # far tighter and would leave the hold unbounded.
            self.state = "HOVER"
            return Command(
                "HOVER",
                np.zeros(3),
                pose.yaw,
                goal,
                path_error,
                progress,
                should_land=False,
                status="inspection waypoint has no reachable target -> hover",
            )
        if look:
            wpnum, pole_center, pole_id = look
            look_meta = {
                "waypoint": wpnum,
                "pole_id": pole_id,
                "target": pole_center.tolist(),
                "capture_ack_required": True,
                "body_yaw_required": bool(self.cfg.pole_body_look),
            }
            yaw_target = (
                heading_from_to(pos, pole_center, self.cfg.map_frame)
                if self.cfg.pole_body_look
                else float(pose.yaw)
            )
            self.state = "INSPECT"
            return Command(
                "INSPECT",
                np.zeros(3),
                yaw_target,
                pos.copy(),
                path_error,
                progress,
                look_at_pole=look_meta,
                should_land=False,
                status=f"INSPECT wp={wpnum} pole={pole_id} awaiting gimbal/capture ack",
            )

        if action == "ARRIVING":
            self.state = "CRUISE"
            return Command(
                "ARRIVING",
                np.zeros(3),
                float(pose.yaw),
                goal,
                path_error,
                progress,
                status=(
                    f"waypoint {self.target_index + 1} arrival confirm "
                    f"{self._arrive_frames}/{self.cfg.waypoint_arrive_confirm_frames}"
                ),
            )
        if action == "FINAL_HOLD":
            self.state = "CRUISE"
            return Command(
                "FINAL_HOLD",
                np.zeros(3),
                float(pose.yaw),
                goal,
                path_error,
                progress,
                status=(
                    f"final hold {self._final_confirm_frames}/"
                    f"{self.cfg.final_arrive_confirm_frames}"
                ),
            )
        if action == "ABORT_PENDING_INSPECTION":
            self.state = "LANDING"
            return Command(
                "ABORT",
                np.zeros(3),
                pose.yaw,
                goal,
                path_error,
                progress,
                should_land=True,
                status="pending inspection missed at route end -> abort/land",
            )
        if action == "LAND":
            self.state = "LANDING"
            return Command(
                "LAND",
                np.zeros(3),
                pose.yaw,
                goal,
                path_error,
                progress,
                should_land=True,
                status="final path reached -> LAND",
            )

        guidance_goal = None
        if self.target_index > self._join_target_index:
            start = self.wp[self.target_index - 1]
            segment = goal - start
            length = float(np.linalg.norm(segment))
            if length > 1e-9:
                _distance, _nearest, fraction = point_segment_distance(pos, start, goal)
                horizontal_segment = np.array(self.cfg.map_frame.horizontal(segment))
                horizontal_squared = float(np.dot(horizontal_segment, horizontal_segment))
                if horizontal_squared > 1e-12:
                    horizontal_offset = np.array(self.cfg.map_frame.horizontal(pos - start))
                    fraction = float(np.clip(
                        np.dot(horizontal_offset, horizontal_segment) / horizontal_squared,
                        0.0, 1.0,
                    ))
                lead = min(
                    float(self.cfg.lookahead),
                    max(self._arrive_radius_for(self.target_index), 0.1 * length),
                )
                guidance_goal = start + segment * min(1.0, fraction + lead / length)
                radius = self._arrive_radius_for(self.target_index)
                if path_error > radius:
                    self._rejoining = True
                elif path_error <= 0.75 * radius:
                    self._rejoining = False
                if self._rejoining:
                    guidance_goal = _nearest
                    remaining = goal - _nearest
                    remaining_length = float(np.linalg.norm(remaining))
                    if path_error < 2.0 * radius and remaining_length > 1e-9:
                        # Keep making bounded along-leg progress during small
                        # drift repair instead of stalling at a noisy projection.
                        guidance_goal = _nearest + remaining * min(1.0, radius / remaining_length)
                    action = "REJOIN"
        steering_goal = goal if guidance_goal is None else guidance_goal
        yaw_delta = steering_goal - pos if guidance_goal is None else goal - start
        yaw_target = (
            self.cfg.map_frame.heading(yaw_delta)
            if self.cfg.map_frame.horizontal_distance(yaw_delta) > 1e-9
            else float(pose.yaw)
        )
        look_meta = None
        status = action

        d = steering_goal - pos
        dist = float(np.linalg.norm(d))
        vel = np.zeros(3, dtype=float) if dist < 1e-9 else d / dist
        self.state = "CRUISE"
        return Command(
            action,
            vel,
            yaw_target,
            goal,
            path_error,
            progress,
            look_at_pole=look_meta,
            should_land=False,
            status=status,
            guidance_goal=guidance_goal,
        )


def _validated_command_goal(cmd: Command, pose: Pose, yaw_sign: int) -> np.ndarray:
    _finite_vec3(cmd.vel_map, "command velocity")
    goal = _finite_vec3(cmd.goal, "command goal")
    if cmd.guidance_goal is not None:
        _finite_vec3(cmd.guidance_goal, "guidance goal")
    scalars = [
        cmd.yaw_target,
        cmd.path_error,
        cmd.progress,
        pose.x,
        pose.y,
        pose.z,
        pose.yaw,
        pose.stamp,
    ]
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("command and pose scalars must be finite")
    if int(yaw_sign) not in (-1, 1):
        raise ValueError("yaw sign must be -1 or 1")
    return goal


def _command_yaw_context(
    cmd: Command,
    pose: Pose,
    delta: np.ndarray,
    config: ControlConfig,
    *,
    inspection: bool,
) -> tuple[float, float | None]:
    if inspection:
        return wrap_angle(float(cmd.yaw_target) - float(pose.yaw)), None
    if cmd.guidance_goal is not None:
        error = wrap_angle(float(cmd.yaw_target) - float(pose.yaw))
        return error, error
    projected_error = ground_projected_yaw_error(float(pose.yaw), delta, config.map_frame)
    yaw_error = 0.0 if projected_error is None else projected_error
    return yaw_error, projected_error


def _yaw_percent_command(
    yaw_error: float,
    projected_error: float | None,
    *,
    config: ControlConfig,
    yaw_sign: int,
    inspection: bool,
    require_yaw_alignment: bool,
) -> tuple[int, int, int, int] | None:
    yaw_error_deg = math.degrees(yaw_error)
    needs_yaw = inspection or (
        require_yaw_alignment
        and projected_error is not None
        and abs(yaw_error_deg) > config.yaw_tolerance_deg
    )
    if not needs_yaw:
        return None
    if abs(yaw_error_deg) <= config.yaw_tolerance_deg:
        return 0, 0, 0, 0
    yaw_strength = min(
        config.max_yaw_pcmd,
        max(
            1,
            round(100.0 * abs(yaw_error_deg) / 70.0 / config.yaw_command_period_s),
        ),
    )
    # PCMD yaw is positive clockwise; ground_projected_yaw_error is CCW.
    yaw = int(yaw_sign) * (-yaw_strength if yaw_error > 0.0 else yaw_strength)
    return 0, 0, int(yaw), 0


def _translation_percent_command(
    delta: np.ndarray,
    pose: Pose,
    config: ControlConfig,
    *,
    distance_to_waypoint: float | None = None,
    recovering: bool = False,
) -> tuple[int, int, int, int]:
    """Limit-style approach taper: speed shrinks as the waypoint gets closer.

    Strength steps down with remaining distance (``distance -> 0`` implies
    ``strength -> 1 -> 0``), quantized to integer PCMD. Farther than
    ``slowdown_distance`` flies at full authority; inside the deadzone it
    stops. Outside twice the arrival sphere the floor is 2 so the final
    approach keeps cruising; only the sphere itself slows to creep. A
    recovery leg far from the line keeps full authority so a gust can
    still be closed.
    """
    zero = (0, 0, 0, 0)
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= config.translation_arrival_tolerance:
        return zero

    forward, right, up = config.map_frame.body_components(delta, float(pose.yaw))
    forward_fraction = forward / distance
    right_fraction = right / distance
    up_fraction = up / distance
    speed_distance = distance if distance_to_waypoint is None else distance_to_waypoint
    if not math.isfinite(speed_distance):
        return zero
    slowdown_span = max(
        1e-6,
        config.slowdown_distance - config.translation_arrival_tolerance,
    )
    remaining = max(0.0, speed_distance - config.translation_arrival_tolerance)
    ratio = max(0.0, min(1.0, remaining / slowdown_span))
    # Limit-style taper with a floor of 2 outside the arrival sphere: far
    # legs keep full authority, the final approach keeps a steady cruise
    # instead of dropping to creep early, and only the sphere itself (plus
    # the deadzone) slows to 1 -> 0. The 6-point route's last legs sit in
    # the 0.06-0.10u band; creeping that whole band at 1 misses the 300 s
    # budget (measured: step cap at 300 s, 10/11 reached, 0.027u short).
    if speed_distance <= 2.0 * config.waypoint_arrive_radius:
        strength = 1
    else:
        strength = min(
            config.max_translation_pcmd,
            max(2, int(math.ceil(ratio * config.max_translation_pcmd - 1e-9))),
        )
    if recovering and speed_distance > 2.0 * config.waypoint_arrive_radius:
        strength = config.max_translation_pcmd
    clamp = lambda value: int(round(max(-100.0, min(100.0, float(value)))))
    roll = clamp(strength * right_fraction)
    pitch = clamp(strength * forward_fraction)
    # Weak-axis floor: integer PCMD otherwise drops a meaningful secondary
    # axis near the waypoint (at strength 1 any |fraction| < 0.5 rounds to 0),
    # so the drone slides along one body axis while the uncorrected lateral
    # error swings the target bearing away (recorded 2026-09-14: the closest
    # approach stalled at 4x the arrival radius with yaw saturated). An axis
    # carrying more than a quarter of the direction keeps a minimal +/-1
    # creep. Yaw is untouched (translation never yaws) and gaz keeps its own
    # vertical authority scale.
    if roll == 0 and abs(right_fraction) > 0.25:
        roll = 1 if right_fraction > 0 else -1
    if pitch == 0 and abs(forward_fraction) > 0.25:
        pitch = 1 if forward_fraction > 0 else -1
    return (
        roll,
        pitch,
        0,
        clamp(strength * up_fraction * config.max_vertical_pcmd / config.max_translation_pcmd),
    )


def command_to_body_percent(
    cmd: Command,
    pose: Pose,
    *,
    config: ControlConfig | None = None,
    yaw_sign: int = 1,
    require_yaw_alignment: bool = True,
) -> tuple[int, int, int, int]:
    """Map the latest camera-target line to ANAFI body PCMD percentages.

    The route planner stays in the raw GLOMAP frame and uses the site's measured
    gravity only to define its horizontal plane and up direction. This adapter
    rotates that 3-D target vector into body forward/right/up,
    normalizes the resultant, and assigns pitch/roll/gaz from one shared bounded
    strength. When requested, yaw gates horizontal motion while height correction
    continues during the turn.
    """
    cfg = config or ControlConfig()
    zero = (0, 0, 0, 0)
    try:
        goal = _validated_command_goal(cmd, pose, yaw_sign)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return zero

    movement = cmd.action in {"FOLLOW", "REJOIN"}
    inspection = cmd.action == "INSPECT"
    if cmd.should_land or (not movement and not inspection):
        return zero

    distance_to_waypoint = float(np.linalg.norm(goal - pose.xyz))
    steering_goal = goal if cmd.guidance_goal is None else np.asarray(cmd.guidance_goal, float)
    delta = steering_goal - pose.xyz
    recovering = cmd.action == "REJOIN" and cmd.guidance_goal is not None
    if recovering:
        distance_to_waypoint = float(np.linalg.norm(delta))
    try:
        yaw_error, projected_error = _command_yaw_context(
            cmd,
            pose,
            delta,
            cfg,
            inspection=inspection,
        )
    except (TypeError, ValueError, OverflowError):
        return zero
    # A purely vertical target has no defensible yaw direction, so translation
    # may proceed without changing yaw.
    yaw_command = _yaw_percent_command(
        yaw_error,
        projected_error,
        config=cfg,
        yaw_sign=yaw_sign,
        inspection=inspection,
        require_yaw_alignment=require_yaw_alignment,
    )
    if yaw_command is not None:
        if movement:
            gaz = _translation_percent_command(
                delta, pose, cfg, distance_to_waypoint=distance_to_waypoint, recovering=recovering
            )[3]
            return 0, 0, yaw_command[2], gaz
        return yaw_command

    if not movement:
        return zero
    return _translation_percent_command(
        delta, pose, cfg, distance_to_waypoint=distance_to_waypoint, recovering=recovering
    )


class YawAlignedPcmdController:
    """Yaw gate shared by dry-run and real flight, with bounded mid-leg repair.

    Heading is corrected once per waypoint (takeoff counts as the first one)
    while height correction continues; horizontal cruise follows alignment.
    Small mid-leg drift is closed by strafing. A large drift (gust-rotated, or a
    fused heading that walked away) re-opens the turn gate after
    ``yaw_alignment_confirmation_updates`` consecutive fresh samples past
    ``mid_leg_realign_deg``. The leg's original window/timeout budget is NOT
    reset, so a flapping heading ends in hold-for-operator, never chatter.
    """

    def __init__(self, config: ControlConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.target_key = None
        self.aligned_for_translation = False
        self.confirmation_count = 0
        self.previous_timestamp: float | None = None
        self.previous_yaw: float | None = None
        self.alignment_started: float | None = None
        self.alignment_windows = 1
        self.phase = "idle"
        self._alignment_samples: list[tuple[float, float, float]] = []
        self._realign_strikes = 0
        self.centering = False
        self.vertical_adjusting = False
        self._reset_turn_recovery()

    def _reset_turn_recovery(self) -> None:
        self.turn_anchor: np.ndarray | None = None
        self.turn_recovering = False
        self.turn_anchor_error_u = 0.0
        self._turn_settle_stamp: float | None = None
        self._turn_velocity_stamp: float | None = None

    def _turn_drift_command(self, cmd, pose, now, body_velocity, radius):
        """Bounded position/velocity feedback before allowing a stationary turn.

        Position feedback uses map units/radius; damping uses body m/s with
        10 PCMD per m/s. No metric velocity is added to map coordinates.
        """
        if body_velocity is None:
            if not self.turn_recovering:
                return None
            self._turn_settle_stamp = None
            self.phase = "turn_recovery_wait"
            return 0, 0, 0, 0
        forward_v, right_v, stamp = body_velocity
        if (
            not all(math.isfinite(v) for v in body_velocity)
            or not 0.0 <= now - stamp <= self.config.max_pose_age_s
            or not pose.map_confirmed
        ):
            self._turn_settle_stamp = None
            self.phase = "turn_recovery_wait"
            return 0, 0, 0, 0
        if self.turn_anchor is None:
            self.turn_anchor = pose.xyz.copy()
        error = self.turn_anchor - pose.xyz
        forward, right, _up = self.config.map_frame.body_components(error, pose.yaw)
        distance = math.hypot(forward, right)
        self.turn_anchor_error_u = distance
        speed = math.hypot(forward_v, right_v)
        if speed > 0.10 or distance > 2.0 * radius:
            self.turn_recovering = True
        if not self.turn_recovering:
            return None

        fresh = self._turn_velocity_stamp is None or stamp > self._turn_velocity_stamp
        if fresh:
            self._turn_velocity_stamp = stamp
        stable = speed <= 0.08 and distance <= radius
        if stable and fresh:
            if self._turn_settle_stamp is None:
                self._turn_settle_stamp = stamp
            elif stamp - self._turn_settle_stamp >= 0.3:
                self.turn_recovering = False
                self._turn_settle_stamp = None
        elif not stable:
            self._turn_settle_stamp = None
        if stable and not self.turn_recovering:
            self.phase = "turn_settle"
            # One zero-horizontal tick separates braking from resumed yaw.
            gaz = command_to_body_percent(cmd, pose, config=self.config)[3]
            return 0, 0, 0, gaz

        self.phase = "turn_settle" if stable else "turn_drift_recovery"
        cap = self.config.max_translation_pcmd
        # Leave a small position deadband so map noise cannot drive a brake pulse.
        position_scale = 0.0 if distance <= 0.5 * radius else cap / radius
        pitch = int(round(np.clip(forward * position_scale - 10.0 * forward_v, -cap, cap)))
        roll = int(round(np.clip(right * position_scale - 10.0 * right_v, -cap, cap)))
        return roll, pitch, 0, 0

    def _separate_translation_axes(self, command, cmd: Command, pose: Pose, now, body_velocity):
        """Separate height/horizontal repair, with measured horizontal damping."""
        roll, pitch, yaw, gaz = command
        goal = cmd.goal if cmd.guidance_goal is None else cmd.guidance_goal
        delta = np.asarray(goal) - pose.xyz
        height_error = self.config.map_frame.vertical(delta)
        radius = float(self.config.waypoint_arrive_radius)
        radii = self.config.waypoint_arrive_radii
        if radii is not None and isinstance(self.target_key, int) and 0 <= self.target_key < len(radii):
            radius = float(radii[self.target_key])
        if (roll or pitch) and gaz:
            if abs(height_error) > radius:
                self.vertical_adjusting = True
            elif abs(height_error) <= 0.5 * radius:
                self.vertical_adjusting = False
            horizontal_error = self.config.map_frame.horizontal_distance(delta)
            if horizontal_error > max(radius, abs(height_error)):
                # A smaller height residual must not starve horizontal gust repair.
                self.vertical_adjusting = False
            if self.vertical_adjusting:
                self.phase = "height_adjust"
                strength = max(1, round(self.config.max_vertical_pcmd * min(
                    1.0, abs(height_error) / (3.0 * radius)
                )))
                return 0, 0, yaw, int(math.copysign(strength, height_error))
            horizontal = _translation_percent_command(
                delta - height_error * self.config.map_frame.up, pose, self.config,
                distance_to_waypoint=(horizontal_error if cmd.action == "REJOIN"
                                      else float(np.linalg.norm(cmd.goal - pose.xyz))),
                recovering=cmd.action == "REJOIN",
            )
            command = horizontal[0], horizontal[1], 0, 0
        close_or_recovering = self.phase != "translate" or float(np.linalg.norm(cmd.goal - pose.xyz)) <= 3.0 * radius
        if body_velocity is not None and close_or_recovering and command[2] == command[3] == 0:
            forward_v, right_v, stamp = body_velocity
            if (not all(math.isfinite(v) for v in body_velocity)
                    or not 0.0 <= now - stamp <= self.config.max_pose_age_s
                    or not pose.map_confirmed):
                return 0, 0, 0, 0
            forward, right, _up = self.config.map_frame.body_components(delta, pose.yaw)
            cap = self.config.max_translation_pcmd
            # Position/radius is dimensionless; body velocity provides damping
            # in 10 PCMD per m/s, without integrating metres into the raw map.
            pitch = int(np.clip(round(cap * forward / (2.0 * radius) - 10.0 * forward_v), -cap, cap))
            roll = int(np.clip(round(cap * right / (2.0 * radius) - 10.0 * right_v), -cap, cap))
            return roll, pitch, 0, 0
        return command

    def _normalized_now(self, now: float) -> float | None:
        try:
            value = float(now)
            if not math.isfinite(value):
                raise ValueError("non-finite control timestamp")
        except (TypeError, ValueError):
            self.reset()
            return None
        return value

    def _prepare_target(self, now: float, target_key) -> None:
        if (
            self.previous_timestamp is not None
            and now - self.previous_timestamp > self.config.max_pose_age_s
        ):
            self.reset()
        if target_key != self.target_key:
            self.reset()
            self.target_key = target_key

    def _alignment_measurement(
        self,
        cmd: Command,
        pose: Pose,
    ) -> tuple[float, float | None, float] | None:
        try:
            goal = _finite_vec3(
                cmd.goal if cmd.guidance_goal is None else cmd.guidance_goal, "guidance goal"
            )
            pose_stamp = float(pose.stamp)
            if not math.isfinite(pose_stamp):
                raise ValueError("non-finite pose timestamp")
            yaw_error, projected_error = _command_yaw_context(
                cmd, pose, goal - pose.xyz, self.config, inspection=False
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            self.reset()
            return None
        return pose_stamp, projected_error, yaw_error

    def _record_alignment_sample(
        self,
        pose_stamp: float,
        pose_yaw: float,
    ) -> tuple[bool, float | None]:
        fresh = self.previous_timestamp is None or pose_stamp > self.previous_timestamp
        yaw_rate_deg_s = None
        if fresh and self.previous_timestamp is not None and self.previous_yaw is not None:
            dt = pose_stamp - self.previous_timestamp
            if dt > 1e-6:
                yaw_rate_deg_s = math.degrees(wrap_angle(float(pose_yaw) - self.previous_yaw)) / dt
        if fresh:
            self.previous_timestamp = pose_stamp
            self.previous_yaw = float(pose_yaw)
        return fresh, yaw_rate_deg_s

    def _alignment_window_stable(
        self,
        *,
        stamp: float,
        yaw: float,
        yaw_error: float,
        fresh_sample: bool,
    ) -> bool:
        """Confirm from a bounded in-band window rate, not adjacent-sample rate."""
        in_band = abs(math.degrees(yaw_error)) <= self.config.yaw_tolerance_deg
        needed = int(self.config.yaw_alignment_confirmation_updates)
        if not in_band:
            self._alignment_samples = []
            self.confirmation_count = 0
            return False
        if not fresh_sample:
            return False
        self._alignment_samples.append((float(stamp), float(yaw), float(yaw_error)))
        if needed > 0 and len(self._alignment_samples) > needed:
            del self._alignment_samples[:-needed]
        self.confirmation_count = len(self._alignment_samples)
        if len(self._alignment_samples) != needed:
            return False
        if any(
            abs(math.degrees(error)) > self.config.yaw_tolerance_deg
            for _stamp, _yaw, error in self._alignment_samples
        ):
            return False
        stamp_first, yaw_first, _err_first = self._alignment_samples[0]
        stamp_last, yaw_last, _err_last = self._alignment_samples[-1]
        dt = stamp_last - stamp_first
        if dt <= 1e-6:
            # A one-sample window has no span; confirm the in-band sample.
            return needed == 1
        window_rate = abs(math.degrees(wrap_angle(yaw_last - yaw_first))) / dt
        return window_rate <= float(self.config.yaw_alignment_max_rate_deg_s)

    def _alignment_hold_command(
        self,
        cmd: Command,
        pose: Pose,
        now: float,
        yaw_error: float,
        fresh_sample: bool,
        yaw_rate_deg_s: float | None,
        yaw_sign: int,
    ) -> tuple[int, int, int, int]:
        zero = (0, 0, 0, 0)
        if self.alignment_started is None:
            self.alignment_started = now
        elif now - self.alignment_started > self.config.yaw_alignment_timeout_s:
            if int(self.alignment_windows) < int(self.config.yaw_alignment_max_windows):
                # Still converging, just slow (e.g. a full course reversal at
                # the capped yaw rate): open a fresh window and keep turning
                # instead of stranding the mission on the first budget.
                self.alignment_windows = int(self.alignment_windows) + 1
                self.alignment_started = now
                self.confirmation_count = 0
                self._alignment_samples = []
            else:
                # Out of windows: a turn that never converges commands yaw
                # forever while translation stays at zero. Stop commanding and
                # hold: a stuck alignment is the operator's call, and the tick
                # log carries this phase.
                self.phase = "yaw_alignment_timeout"
                return zero
        # Adjacent two-point yaw_rate_deg_s is log-only; do not use it to confirm.
        _ = yaw_rate_deg_s
        stable = self._alignment_window_stable(
            stamp=float(pose.stamp),
            yaw=float(pose.yaw),
            yaw_error=yaw_error,
            fresh_sample=fresh_sample,
        )
        if abs(math.degrees(yaw_error)) > self.config.yaw_tolerance_deg:
            self.phase = "turn"
            return command_to_body_percent(
                cmd,
                pose,
                config=self.config,
                yaw_sign=yaw_sign,
                require_yaw_alignment=True,
            )
        if stable:
            self.aligned_for_translation = True
            self.phase = "yaw_alignment_confirmed"
        else:
            self.phase = "yaw_alignment_hold"
        vertical = command_to_body_percent(
            cmd, pose, config=self.config, yaw_sign=yaw_sign,
            require_yaw_alignment=False,
        )[3]
        return 0, 0, 0, vertical

    def update(
        self,
        cmd: Command,
        pose: Pose,
        now: float,
        *,
        target_key,
        yaw_sign: int = 1,
        body_velocity: tuple[float, float, float] | None = None,
    ) -> tuple[int, int, int, int]:
        normalized_now = self._normalized_now(now)
        if normalized_now is None:
            return 0, 0, 0, 0
        now = normalized_now

        self._prepare_target(now, target_key)

        if cmd.action == "FINAL_HOLD" and not cmd.should_land:
            # Terminal centering: the landing point is known and fresh poses
            # keep arriving, so hold it with the visual servo instead of an
            # open-loop hover that wind walks off the arrival sphere. Yaw
            # stays locked; translation stops inside the arrival deadband.
            # (ARRIVING keeps its deliberate settle-in-place behavior: on a
            # wide sphere the center can sit behind the drone.)
            measurement = self._alignment_measurement(cmd, pose)
            if measurement is None:
                return 0, 0, 0, 0
            self._record_alignment_sample(measurement[0], float(pose.yaw))
            self.centering = True
            hold_cmd = Command(
                "FOLLOW",
                np.zeros(3, dtype=float),
                float(pose.yaw),
                np.array(cmd.goal, dtype=float, copy=True),
                cmd.path_error,
                cmd.progress,
                status="final centering",
            )
            centered = command_to_body_percent(
                hold_cmd,
                pose,
                config=self.config,
                yaw_sign=yaw_sign,
                require_yaw_alignment=False,
            )
            self.phase = "final_centering"
            return self._separate_translation_axes(centered, hold_cmd, pose, now, body_velocity)

        if cmd.action not in {"FOLLOW", "REJOIN"} or cmd.should_land:
            self.reset()
            return 0, 0, 0, 0

        measurement = self._alignment_measurement(cmd, pose)
        if measurement is None:
            return 0, 0, 0, 0
        pose_stamp, projected_error, yaw_error = measurement

        fresh_alignment_sample, yaw_rate_deg_s = self._record_alignment_sample(
            pose_stamp,
            float(pose.yaw),
        )

        radius = float(self.config.waypoint_arrive_radius)
        radii = self.config.waypoint_arrive_radii
        if radii is not None and isinstance(target_key, int) and 0 <= target_key < len(radii):
            radius = float(radii[target_key])
        horizontal_distance = self.config.map_frame.horizontal_distance(cmd.goal - pose.xyz)
        if horizontal_distance <= radius:
            self.centering = True
        elif horizontal_distance > 2.0 * radius:
            self.centering = False
        if self.centering:
            # Near a waypoint the target bearing is ill-conditioned. Hold yaw
            # and close position/height instead of turning around the sphere.
            # The wider exit boundary prevents localization noise from toggling
            # this phase on every capture.
            self.phase = "waypoint_centering"
            self._reset_turn_recovery()
            centering_cmd = replace(cmd, guidance_goal=None, action="FOLLOW")
            return self._separate_translation_axes(command_to_body_percent(
                centering_cmd, pose,
                config=self.config, yaw_sign=yaw_sign,
                require_yaw_alignment=False,
            ), centering_cmd, pose, now, body_velocity)
        if cmd.action == "REJOIN" and cmd.guidance_goal is not None:
            # Position recovery takes priority over nose alignment. Translate
            # back to the segment with yaw held, then resume the turn gate.
            self.phase = "route_rejoin"
            self._reset_turn_recovery()
            return self._separate_translation_axes(command_to_body_percent(
                cmd, pose, config=self.config, yaw_sign=yaw_sign,
                require_yaw_alignment=False,
            ), cmd, pose, now, body_velocity)

        if projected_error is None:
            self.aligned_for_translation = True

        # Bounded mid-leg repair: small drift keeps strafing correction, but a
        # large drift would fly a long rotated leg. Strikes need consecutive
        # FRESH samples so one noisy frame cannot park translation; the leg's
        # window/timeout budget is deliberately not reset, so a flapping
        # heading degrades to hold-for-operator instead of chattering.
        # Repeated captures neither add nor clear strikes when control runs
        # faster than localization.
        if (
            self.aligned_for_translation
            and projected_error is not None
            and fresh_alignment_sample
            and abs(math.degrees(yaw_error)) > float(self.config.mid_leg_realign_deg)
        ):
            self._realign_strikes += 1
        elif fresh_alignment_sample:
            self._realign_strikes = 0
        if self._realign_strikes >= int(self.config.yaw_alignment_confirmation_updates):
            self._realign_strikes = 0
            self.aligned_for_translation = False
            self.confirmation_count = 0
            self._alignment_samples = []
            self._reset_turn_recovery()
        if not self.aligned_for_translation:
            if self.turn_recovering or abs(math.degrees(yaw_error)) > self.config.yaw_tolerance_deg:
                recovery = self._turn_drift_command(cmd, pose, now, body_velocity, radius)
                if recovery is not None:
                    return recovery
            return self._alignment_hold_command(
                cmd,
                pose,
                now,
                yaw_error,
                fresh_alignment_sample,
                yaw_rate_deg_s,
                yaw_sign,
            )
        self.phase = "translate"
        self._reset_turn_recovery()
        return self._separate_translation_axes(command_to_body_percent(
            cmd,
            pose,
            config=self.config,
            yaw_sign=yaw_sign,
            require_yaw_alignment=False,
        ), cmd, pose, now, body_velocity)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the route controller without flying.")
    ap.add_argument(
        "--path", default=str(Path(__file__).resolve().parents[1] / "safezone/flight_path.json")
    )
    ap.add_argument(
        "--poles", default=str(Path(__file__).resolve().parents[1] / "safezone/poles.json")
    )
    args = ap.parse_args()
    wp = load_waypoints(args.path)
    poles = load_poles(args.poles)
    ctrl = RouteAutoController(wp, poles, config_for_route(args.path))
    print(
        f"loaded waypoints={len(wp)} poles={len(poles)} inspect_valid={sorted(i + 1 for i in ctrl.inspect_waypoints)}"
    )
    print("This module outputs commands; it does not connect to or fly the drone.")
