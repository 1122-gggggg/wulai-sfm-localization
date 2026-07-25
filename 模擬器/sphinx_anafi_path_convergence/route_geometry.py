#!/usr/bin/env python3
"""Route geometry for the Sphinx ANAFI path-convergence experiment.

Frame convention (copied from production real_path_follow_controller.py so the
experiment controllers speak the same language as the production stack):

  * controller "raw map" frame: horizontal plane = X/Z, vertical/up = -Y
    (larger y  = lower). yaw/heading = atan2(z, x) in the X/Z plane, which for
    the Sphinx telemetry mapping (x=north, z=east) equals the ANAFI NED yaw.
  * path JSON "aligned" frame: X/Y horizontal, Z up.
    aligned (x, y, z) -> raw (x, -z, y)      [same as production aligned_to_glomap]

All distances here are in whatever unit the route is expressed in: Sphinx
meters during simulation, arbitrary map units for the real SfM map. Geometry
does not know physical scale; thresholds such as tube radius 1.0 mean one
route/map unit unless a separate scale calibration is explicitly applied.

Pure functions/classes only. No Olympe. Small helpers are deliberately copied
from production (point/segment projection) instead of imported, so the
experiment cannot drift production behavior and vice versa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Angles / frames

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_of(v: np.ndarray) -> float:
    """Heading of a raw-frame vector in the X/Z horizontal plane."""
    return float(math.atan2(float(v[2]), float(v[0])))


def aligned_to_raw(p) -> np.ndarray:
    """Blender aligned Z-up -> controller raw frame (production convention)."""
    p = list(p)
    return np.array([p[0], -p[2], p[1]], dtype=float)


def raw_to_aligned(p) -> list[float]:
    p = np.asarray(p, float)
    return [float(p[0]), float(p[2]), float(-p[1])]


def horiz(p: np.ndarray) -> np.ndarray:
    """Horizontal (x, z) components of a raw-frame point."""
    p = np.asarray(p, float)
    return np.array([p[0], p[2]], dtype=float)


def up_of(p: np.ndarray) -> float:
    """Altitude ("up" coordinate) of a raw-frame point: up = -y."""
    return -float(np.asarray(p, float)[1])


# ---------------------------------------------------------------------------
# Projection primitives (horizontal-plane only, per spec)

@dataclass
class Projection:
    """Result of projecting a pose onto the route (horizontal plane)."""
    cross_track: float      # horizontal distance to nearest route point
    point: np.ndarray       # nearest route point, full 3D (height interpolated)
    seg_index: int
    t: float                # progress within segment [0, 1]
    s: float                # horizontal arclength from route start
    tangent: np.ndarray     # unit horizontal tangent (x, z) of the segment
    target_y: float         # interpolated route y (raw frame) at the projection


@dataclass
class TubeProjection:
    """Result of projecting a pose onto the route in full 3D."""
    distance: float
    point: np.ndarray
    seg_index: int
    t: float


def _project_horiz_to_segment(c_h: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Project horizontal point c_h onto segment a->b using ONLY X/Z.
    Returns (dist, t) with t clamped to [0, 1]."""
    a_h, b_h = horiz(a), horiz(b)
    ab = b_h - a_h
    l2 = float(ab @ ab)
    t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, float((c_h - a_h) @ ab) / l2))
    q = a_h + t * ab
    return float(np.linalg.norm(c_h - q)), t


def _project_3d_to_segment(pos: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Project a raw-frame 3D point onto a 3D segment. Returns (dist, point, t)."""
    pos = np.asarray(pos, float)
    ab = np.asarray(b, float) - np.asarray(a, float)
    l2 = float(ab @ ab)
    t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, float((pos - a) @ ab) / l2))
    q = a + t * ab
    return float(np.linalg.norm(pos - q)), q, t


class RouteModel:
    """Waypoint polyline with horizontal-arclength parameterization and a
    vertical profile interpolated per segment (spec section 7)."""

    def __init__(self, waypoints):
        self.wp = [np.asarray(p, dtype=float) for p in waypoints]
        if len(self.wp) < 2:
            raise ValueError(f"need >= 2 waypoints, got {len(self.wp)}")
        cum = [0.0]
        for a, b in zip(self.wp[:-1], self.wp[1:]):
            cum.append(cum[-1] + float(np.linalg.norm(horiz(b) - horiz(a))))
        self.cum = np.asarray(cum, dtype=float)          # horizontal arclength
        self.length = float(self.cum[-1])
        self.n_segments = len(self.wp) - 1

    # -- projection ---------------------------------------------------------

    def _segment_projection(self, pos: np.ndarray, i: int) -> Projection:
        a, b = self.wp[i], self.wp[i + 1]
        d, t = _project_horiz_to_segment(horiz(pos), a, b)
        seg_h = horiz(b) - horiz(a)
        seg_len = float(np.linalg.norm(seg_h))
        tangent = seg_h / seg_len if seg_len > 1e-12 else np.array([1.0, 0.0])
        target_y = float(a[1] + t * (b[1] - a[1]))
        q = a + t * (b - a)
        s = float(self.cum[i] + t * seg_len)
        return Projection(cross_track=d, point=q, seg_index=i, t=t, s=s,
                          tangent=tangent, target_y=target_y)

    def project(self, pos: np.ndarray) -> Projection:
        """Nearest point on the FULL polyline (horizontal distance), not just
        the nearest waypoint."""
        best = None
        for i in range(self.n_segments):
            pr = self._segment_projection(pos, i)
            if best is None or pr.cross_track < best.cross_track:
                best = pr
        return best

    def project_active(self, pos: np.ndarray, seg_index: int) -> Projection:
        """Projection clamped to a specific active segment."""
        i = max(0, min(int(seg_index), self.n_segments - 1))
        return self._segment_projection(pos, i)

    def project_tube(self, pos: np.ndarray, seg_index: int | None = None,
                     segment_window: int | None = None) -> TubeProjection:
        """Nearest 3D route point for tube/corridor safety checks.

        By default this checks the full polyline. When seg_index/window are
        supplied, it only checks [seg_index-window, seg_index+window]. That avoids
        accepting the drone inside the wrong branch of a self-crossing or U-turn
        route.
        """
        if seg_index is None or segment_window is None:
            lo, hi = 0, self.n_segments - 1
        else:
            w = max(0, int(segment_window))
            c = max(0, min(int(seg_index), self.n_segments - 1))
            lo, hi = max(0, c - w), min(self.n_segments - 1, c + w)
        best = None
        for i in range(lo, hi + 1):
            d, q, t = _project_3d_to_segment(pos, self.wp[i], self.wp[i + 1])
            if best is None or d < best.distance:
                best = TubeProjection(distance=d, point=q, seg_index=i, t=t)
        return best

    # -- arclength sampling --------------------------------------------------

    def point_at_s(self, s: float):
        """(point3d, seg_index) at horizontal arclength s (clamped to route).
        Height is interpolated along the segment (spec: lookahead height)."""
        s = max(0.0, min(float(s), self.length))
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = max(0, min(i, self.n_segments - 1))
        seg_len = float(self.cum[i + 1] - self.cum[i])
        t = 0.0 if seg_len < 1e-12 else (s - float(self.cum[i])) / seg_len
        a, b = self.wp[i], self.wp[i + 1]
        return a + t * (b - a), i

    def segment_heading(self, i: int) -> float:
        i = max(0, min(int(i), self.n_segments - 1))
        d = self.wp[i + 1] - self.wp[i]
        return heading_of(d)

    # -- validation (steepness etc.) -----------------------------------------

    def validate(self, steep_slope_ratio: float = 1.0) -> list[dict]:
        """Per-segment geometry report. A segment whose |height change| exceeds
        steep_slope_ratio * horizontal length is flagged as steep (near-vertical
        motion) and should get intermediate waypoints or a separate climb/hover
        behavior."""
        out = []
        for i in range(self.n_segments):
            a, b = self.wp[i], self.wp[i + 1]
            h_len = float(np.linalg.norm(horiz(b) - horiz(a)))
            dz_up = up_of(b) - up_of(a)                     # + = climb
            len3d = float(np.linalg.norm(b - a))
            slope = abs(dz_up) / max(1e-9, h_len)
            steep = slope > steep_slope_ratio
            rec = {
                "segment": i,
                "horizontal_length": h_len,
                "height_change_up": dz_up,
                "length_3d": len3d,
                "slope_ratio": slope,
                "climb_per_unit_horizontal": dz_up / max(1e-9, h_len),
                "steep": steep,
            }
            if steep:
                rec["warning"] = ("This segment requires near-vertical motion. "
                                  "Add intermediate waypoints or separate "
                                  "climb/hover behavior.")
            out.append(rec)
        return out


# ---------------------------------------------------------------------------
# Adaptive lookahead (spec section 3)

def adaptive_lookahead(cross_track: float, base: float = 0.8, k_error: float = 0.5,
                       lo: float = 0.5, hi: float = 3.0) -> float:
    """L = clamp(base + k_error * cross_track, lo, hi). Values are route/map
    units; in Sphinx those units are meters, but in real SfM they are arbitrary
    map units."""
    return max(float(lo), min(float(hi), float(base) + float(k_error) * float(cross_track)))


# ---------------------------------------------------------------------------
# Route patterns for the Sphinx trials (built relative to a start pose, raw frame)

def build_route(pattern: str, c0: np.ndarray, seg_len: float = 4.0,
                height_amp: float = 0.0) -> list[np.ndarray]:
    """Waypoints in raw frame starting at c0. `height_amp` > 0 adds a climb /
    descend profile across the route (up-units; raw y decreases when climbing)."""
    c0 = np.asarray(c0, dtype=float)
    L = float(seg_len)
    if pattern == "line":
        offs = [(0, 0), (1, 0), (2, 0), (3, 0)]
    elif pattern == "square":
        offs = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    elif pattern == "s_curve":
        offs = [(0, 0), (1, 0), (1.5, 0.7), (2.0, 1.4), (3.0, 1.4), (3.5, 0.7), (4.0, 0.0)]
    else:
        raise ValueError(f"unknown pattern {pattern!r} (line|square|s_curve)")
    n = len(offs)
    ups = [0.0] * n
    if height_amp:
        # triangle profile: climb to exactly height_amp mid-route, back to 0
        tri = [1.0 - abs(2.0 * (i / max(1, n - 1)) - 1.0) for i in range(n)]
        peak = max(tri) or 1.0
        ups = [float(height_amp) * v / peak for v in tri]
    return [c0 + np.array([dx * L, -up, dz * L], dtype=float)
            for (dx, dz), up in zip(offs, ups)]


# ---------------------------------------------------------------------------
# Random start sampling (spec: random point within 5 m of P0, all directions)

def sample_start_offset(rng: np.random.Generator, radius: float, route_heading: float,
                        quadrant: int | None = None, min_radius: float = 0.75):
    """Random horizontal start offset around P0.

    quadrant (0..3) stratifies the direction RELATIVE TO THE ROUTE HEADING into
    front / left / back / right so trials provably cover all sides. Returns
    (offset_raw_xyz, bearing_rel_route_rad, radius)."""
    r = float(rng.uniform(min_radius, radius))
    if quadrant is None:
        rel = float(rng.uniform(-math.pi, math.pi))
    else:
        # quadrant q covers rel in [q*90 - 45, q*90 + 45) deg relative to the
        # route heading: 0=front, 1=left-or-right (frame handed), 2=behind, 3=other side
        q = int(quadrant) % 4
        rel = wrap_angle(q * (math.pi / 2.0) + float(rng.uniform(-math.pi / 4.0, math.pi / 4.0)))
    bearing = wrap_angle(route_heading + rel)
    off = np.array([r * math.cos(bearing), 0.0, r * math.sin(bearing)], dtype=float)
    return off, rel, r
