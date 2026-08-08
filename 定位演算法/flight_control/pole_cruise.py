#!/usr/bin/env python3
"""Pole-inspection cruise: fly the hand-drawn path, but turn the CAMERA to face
each boxed pole while passing it, then snap back to looking straight ahead.

Inputs (both ALIGNED map frame, scale-free, from the Blender tools):
  safezone/flight_path.json  -> the route waypoints (load_path)
  safezone/poles.json        -> the marked pole boxes (load_poles)

Behaviour, as a function of the live pose:
  * a carrot point LOOKAHEAD ahead on the path gives the travel direction +
    target altitude (this is what actually moves the drone).
  * HEADING schedule:
      - far from every pole         -> face the path forward tangent.
      - within RELEASE_R of a pole   -> start blending toward "look at pole".
      - within ENGAGE_R of a pole    -> fully locked on the pole bearing.
    The blend is distance-based with hysteresis, so as the drone approaches a
    pole it locks on, and once it has passed (distance grows again) it eases
    back to forward -- exactly "face the pole, recover to straight ahead after".
  * the ANAFI gimbal only pitches, so horizontal aim = drone YAW (body turns);
    the gimbal PITCH tilts up/down to the pole mid-height while engaged.
  * because the drone is yawed at the pole but must still track the path, the
    world travel direction is split into body forward (pitch) + lateral (roll)
    using the current yaw, so it strafes along the route while looking sideways.

PCMD percents + sign conventions mirror cruise_geofence.py. VERIFY YAW_SIGN /
ROLL_SIGN / gimbal sign on YOUR airframe before flying (they match the built-in
sim here, noted the same way the existing controllers are).

Standalone self-check:  python3 deploy/pole_cruise.py
"""
# NOTE: not wired into the production flight path (path_follow_flight). Experimental/unused as of this package.
from __future__ import annotations

import math
import os

import numpy as np

ROOT = os.environ.get("SFM_MAP_ROOT", "").strip()

# --- tunables (MAP UNITS / percents; same spirit as cruise_geofence) ----------
LOOKAHEAD = 1.5          # carrot distance along the path (map units)
ENGAGE_R = 2.0           # within this XY dist of a pole -> fully locked on it
RELEASE_R = 3.5          # outside this -> pure forward; band between = blend
CRUISE_PITCH = 8         # forward body tilt % at full speed (slow)
MAX_ROLL = 8             # lateral (strafe) command % cap
MAX_YAW_RATE = 25        # yaw command % cap
ALT_GAZ = 12             # vertical command % cap
K_YAW = 1.4              # heading P-gain
K_ALT = 1.0              # altitude P-gain
YAW_SIGN = -1            # PCMD yaw>0 = CW; map yaw grows CCW. VERIFY on airframe!
ROLL_SIGN = +1           # PCMD roll>0 = right. VERIFY on airframe!
GIMBAL_SIGN = -1         # gimbal pitch>0 = up? VERIFY on airframe!


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _angle_lerp(a, b, t):
    """Shortest-arc interpolation from angle a to b by fraction t in [0,1]."""
    return a + _wrap_pi(b - a) * t


def _smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PoleCruise:
    def __init__(self, path_json=None, poles_json=None,
                 lookahead=LOOKAHEAD, engage_r=ENGAGE_R, release_r=RELEASE_R):
        from load_path import load_waypoints
        # load_poles.py does not exist; the pole loader lives in the deployment
        # controller. (Legacy module; verify center/base/top axis convention before use.)
        from real_path_follow_controller import load_poles as load_pole_boxes
        if not path_json and not ROOT:
            raise ValueError("path_json or SFM_MAP_ROOT is required")
        if not poles_json and not ROOT:
            raise ValueError("poles_json or SFM_MAP_ROOT is required")
        self.path = [np.asarray(p, float) for p in load_waypoints(
            path_json or f"{ROOT}/safezone/flight_path.json"
        )]
        boxes = load_pole_boxes(poles_json or f"{ROOT}/safezone/poles.json")
        self.pole_xy = np.array([[b["center"][0], b["center"][1]] for b in boxes])
        self.pole_midz = np.array([0.5 * (b["base"][2] + b["top"][2]) for b in boxes])
        self.lookahead = lookahead
        self.engage_r = engage_r
        self.release_r = release_r

    # --- path: carrot point ahead + forward tangent + target altitude ---------
    def carrot(self, P):
        P = np.asarray(P, float)
        pts = self.path
        # nearest waypoint, then walk forward by lookahead
        di = min(range(len(pts)), key=lambda i: np.linalg.norm(pts[i][:2] - P[:2]))
        idx, acc = di, 0.0
        while idx < len(pts) - 1 and acc < self.lookahead:
            acc += np.linalg.norm(pts[idx + 1][:2] - pts[idx][:2])
            idx += 1
        carrot = pts[idx]
        fwd = carrot[:2] - P[:2]
        n = np.linalg.norm(fwd)
        fwd_heading = math.atan2(fwd[1], fwd[0]) if n > 1e-6 else math.atan2(
            (pts[min(di + 1, len(pts) - 1)] - pts[di])[1],
            (pts[min(di + 1, len(pts) - 1)] - pts[di])[0])
        return carrot, fwd_heading

    # --- nearest marked pole in XY -------------------------------------------
    def nearest_pole(self, P):
        if len(self.pole_xy) == 0:
            return None, np.inf
        d = np.linalg.norm(self.pole_xy - np.asarray(P, float)[:2], axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    # --- heading schedule: forward vs look-at-pole, with hysteresis blend -----
    def schedule(self, P, fwd_heading):
        """Returns (desired_heading, gimbal_pitch_deg, blend, pole_idx)."""
        i, d = self.nearest_pole(P)
        if i is None or d >= self.release_r:
            return fwd_heading, 0.0, 0.0, None
        blend = _smoothstep((self.release_r - d) / (self.release_r - self.engage_r))
        to_pole = self.pole_xy[i] - np.asarray(P, float)[:2]
        bearing = math.atan2(to_pole[1], to_pole[0])
        desired = _angle_lerp(fwd_heading, bearing, blend)
        # gimbal tilts toward pole mid-height while engaged
        dz = self.pole_midz[i] - float(P[2])
        gimbal = math.degrees(math.atan2(dz, max(d, 1e-3))) * blend
        return desired, gimbal, blend, i

    # --- full PCMD: strafe along the path while heading faces the pole --------
    def command(self, P, yaw):
        """P=(x,y,z) aligned, yaw=current heading (rad). Returns
        (roll,pitch,yaw,gaz, gimbal_pitch_deg, info)."""
        P = np.asarray(P, float)
        carrot, fwd_heading = self.carrot(P)
        desired, gimbal, blend, pidx = self.schedule(P, fwd_heading)

        # yaw command toward the desired heading
        yaw_err = _wrap_pi(desired - yaw)
        yaw_cmd = YAW_SIGN * _clamp(K_YAW * yaw_err, -1, 1) * MAX_YAW_RATE

        # world travel direction (toward carrot) -> body forward + lateral
        move = carrot[:2] - P[:2]
        mn = np.linalg.norm(move)
        if mn > 1e-6:
            move = move / mn
        fwd_axis = np.array([math.cos(yaw), math.sin(yaw)])
        right_axis = np.array([math.sin(yaw), -math.cos(yaw)])   # +right of heading
        f = float(np.dot(move, fwd_axis))                        # forward share
        l = float(np.dot(move, right_axis))                      # lateral share
        # slow down translation while still swinging onto the heading
        gate = max(0.0, math.cos(yaw_err))
        pitch_cmd = CRUISE_PITCH * f * gate
        roll_cmd = ROLL_SIGN * MAX_ROLL * l * gate

        # altitude hold on the carrot's height
        gaz_cmd = _clamp(K_ALT * (carrot[2] - P[2]), -1, 1) * ALT_GAZ

        info = ("NAV->fwd" if pidx is None
                else f"LOCK pole#{pidx} blend={blend:.2f}")
        return (int(round(roll_cmd)), int(round(pitch_cmd)), int(round(yaw_cmd)),
                int(round(gaz_cmd)), round(GIMBAL_SIGN * gimbal, 1), info)


# ----------------------------------------------------------------------------
def _demo():
    """Self-check on a synthetic straight path with one pole offset to the side.
    Asserts: heading ~forward when far, locks toward the pole at closest pass,
    and recovers to ~forward after passing."""
    import tempfile, json

    # straight path along +X at y=0,z=1 ; one pole at (5, 1.0)
    wp = [[float(x), 0.0, 1.0] for x in range(0, 11)]
    pole = {"center": [5.0, 1.0, 1.2], "half_extents": [0.1, 0.1, 1.2],
            "base": [5.0, 1.0, 0.0], "top": [5.0, 1.0, 2.4], "radius": 0.1}
    d = tempfile.mkdtemp()
    pj, qj = f"{d}/path.json", f"{d}/poles.json"
    json.dump({"waypoints": wp, "closed": False}, open(pj, "w"))
    json.dump({"poles": [pole]}, open(qj, "w"))

    pc = PoleCruise(path_json=pj, poles_json=qj, engage_r=1.2, release_r=3.0)

    def heading_at(x):
        _, fwd = pc.carrot([x, 0.0, 1.0])
        des, gim, blend, idx = pc.schedule([x, 0.0, 1.0], fwd)
        return des, blend

    h_far, b_far = heading_at(0.5)      # far before pole
    h_near, b_near = heading_at(5.0)    # beside the pole (pole is at +Y -> bearing ~ +90deg)
    h_after, b_after = heading_at(9.5)  # well past

    print(f"far  : blend={b_far:.2f} heading={math.degrees(h_far):+6.1f}")
    print(f"near : blend={b_near:.2f} heading={math.degrees(h_near):+6.1f}")
    print(f"after: blend={b_after:.2f} heading={math.degrees(h_after):+6.1f}")

    assert b_far < 0.2, "should be ~forward far from pole"
    assert b_near > 0.8, "should lock on pole at closest pass"
    assert abs(math.degrees(h_near)) > 30, "heading should swing toward the side pole"
    assert b_after < 0.2, "should recover to forward after passing"

    # full command sanity: yaw command non-zero while engaged, gimbal tilts up
    r, p, y, g, gim, info = pc.command([5.0, 0.0, 1.0], yaw=0.0)
    print(f"cmd beside pole: roll={r} pitch={p} yaw={y} gaz={g} gimbal={gim} [{info}]")
    assert y != 0, "yaw should command a turn toward the pole"
    print("OK: lock-on / recover behaviour verified")


if __name__ == "__main__":
    _demo()
