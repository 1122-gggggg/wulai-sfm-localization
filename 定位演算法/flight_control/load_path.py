#!/usr/bin/env python3
"""Load the hand-drawn flight path (scripts/draw_path.py output) for the executor.

The path is a list of 3D waypoints in the ALIGNED map frame (ground z=0, +Z up),
same frame the geofence / safe zone use and the same frame align_pose.AlignedPose
maps live localizer poses into. So it feeds straight into cruise_geofence:

    from load_path import load_waypoints
    from cruise_geofence import PathFollower
    follower = PathFollower(load_waypoints())

Returns a list of np.ndarray(3,) (what PathFollower expects). If `closed` was set
in the JSON, the start waypoint is appended so the loop closes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from route_domain import RouteDocument

PATH_JSON = os.environ.get("SFM_FLIGHT_PATH_JSON", "").strip()


def load_waypoints(path_json: str | None = None, close: bool | None = None):
    path_json = str(path_json or PATH_JSON).strip()
    if not path_json:
        raise ValueError(
            "flight path must be explicit: pass path_json or set SFM_FLIGHT_PATH_JSON"
        )
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"no drawn path at {path_json} -- run scripts/draw_path.py first")
    route = RouteDocument.from_path(Path(path_json))
    return [np.array(point, dtype=float, copy=True)
            for point in route.source_waypoints(close=close)]


def _pt_seg_dist(p, a, b):
    """Shortest 3D distance from point p to segment a-b."""
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-12 else _clamp01(float((p - a) @ ab) / L2)
    return float(np.linalg.norm(p - (a + t * ab)))


def _clamp01(x):
    return max(0.0, min(1.0, x))


def cross_track(P, waypoints=None):
    """Perpendicular distance (map units) from point P to the path centerline =
    min distance to any segment. This is the only thing that decides 'on path';
    it does NOT depend on the camera's physical size -- the path is a 1D line."""
    wp = waypoints or load_waypoints()
    return min(_pt_seg_dist(P, wp[i], wp[i + 1]) for i in range(len(wp) - 1))


def on_path(P, tol, waypoints=None):
    """True if P is within `tol` (the corridor radius you choose) of the path.
    Pick tol from your deviation budget (localization error + safety slack),
    NOT from camera size."""
    return cross_track(P, waypoints) <= tol


if __name__ == "__main__":
    wp = load_waypoints()
    total = sum(float(np.linalg.norm(b - a)) for a, b in zip(wp[:-1], wp[1:]))
    print(f"[load_path] {len(wp)} waypoints, total length {total:.2f} map units")
    for i, p in enumerate(wp):
        print(f"  {i:3d}  [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]")
