#!/usr/bin/env python3
"""Per-(x,y) altitude band from a hand-drawn height profile.

Two profile formats (both produced by the safezone pickers) are supported:

  * single  : one (s,z) floor + ceiling along the corridor centreline
              (scripts/13_pick_profile.py, scripts/15_pick_3d_height.py)
  * two_rail : separate LEFT and RIGHT elevations; the band is interpolated
              across the corridor width between them
              (scripts/16_pick_views.py)

s is distance along the corridor long-axis, w is the signed perpendicular
offset. floor/ceiling are linearly interpolated in s (and, for two_rail, across
w), then a clearance is added above the floor and a margin left below the ceiling.

Usage in cruise_geofence.py:

    from height_profile import HeightProfile
    hp = HeightProfile("safezone/height_profile.json", clearance=0.4, margin=0.3)
    # in CruiseController.compute(), replace the fixed band with:
    z_lo, z_hi = hp.band(pose.x, pose.y)
    z_target = _clamp(self.alt_hold, z_lo, z_hi)

All map units (scale-free); clearance/margin tuned empirically.
"""
from __future__ import annotations

import json

import numpy as np


def _xy(arr):
    a = np.asarray(sorted(arr), float)
    return a[:, 0], a[:, 1]


class HeightProfile:
    def __init__(self, path: str, clearance: float = 0.4, margin: float = 0.3):
        d = json.load(open(path))
        self.origin = np.asarray(d["axis_origin"], float)
        self.dir = np.asarray(d["axis_dir"], float)
        self.clearance = float(clearance)
        self.margin = float(margin)
        self.two = d.get("mode") == "two_rail"
        if self.two:
            self.perp = np.asarray(d["perp"], float)
            self.wl, self.wr = float(d["w_left"]), float(d["w_right"])
            self._fLs, self._fLz = _xy(d["floor_left"])
            self._fRs, self._fRz = _xy(d["floor_right"])
            self._cLs, self._cLz = _xy(d["ceil_left"])
            self._cRs, self._cRz = _xy(d["ceil_right"])
        else:
            self._fs, self._fz = _xy(d["floor"])
            self._cs, self._cz = _xy(d["ceiling"])

    def s_of(self, x: float, y: float) -> float:
        return float((np.array([x, y]) - self.origin) @ self.dir)

    def band(self, x: float, y: float) -> tuple[float, float]:
        """(z_min, z_max) safe altitude band at map point (x, y).

        Out-of-range s clamps to the nearest endpoint (np.interp). z_min is kept
        below z_max; if the band collapses, returns a thin slot at the midpoint.
        """
        p = np.array([x, y], float)
        s = float((p - self.origin) @ self.dir)
        if not self.two:
            floor = float(np.interp(s, self._fs, self._fz)) + self.clearance
            ceil = float(np.interp(s, self._cs, self._cz)) - self.margin
        else:
            w = float((p - self.origin) @ self.perp)
            t = (w - self.wl) / (self.wr - self.wl + 1e-9)
            t = 0.0 if t < 0 else 1.0 if t > 1 else t
            fL = np.interp(s, self._fLs, self._fLz)
            fR = np.interp(s, self._fRs, self._fRz)
            cL = np.interp(s, self._cLs, self._cLz)
            cR = np.interp(s, self._cRs, self._cRz)
            floor = float((1 - t) * fL + t * fR) + self.clearance
            ceil = float((1 - t) * cL + t * cR) - self.margin
        if ceil < floor:
            mid = 0.5 * (floor + ceil)
            return mid - 1e-3, mid + 1e-3
        return floor, ceil
