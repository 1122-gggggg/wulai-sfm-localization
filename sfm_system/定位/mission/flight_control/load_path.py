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

import json
import os

import numpy as np

ROOT = os.environ.get("SFM_MAP_ROOT", "/media/cihcilab/新增磁碟區/sfm_glomap")
PATH_JSON = os.environ.get("SFM_FLIGHT_PATH_JSON", f"{ROOT}/safezone/flight_path.json")


def load_waypoints(path_json: str = PATH_JSON, close: bool | None = None):
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"no drawn path at {path_json} -- run scripts/draw_path.py first")
    d = json.load(open(path_json))
    wp = [np.asarray(p, float) for p in d.get("waypoints", [])]
    if len(wp) < 2:
        raise ValueError(f"path has {len(wp)} waypoints, need >= 2")
    closed = d.get("closed", False) if close is None else close
    if closed and not np.allclose(wp[0], wp[-1]):
        wp.append(wp[0].copy())
    return wp


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
