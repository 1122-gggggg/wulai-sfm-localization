#!/usr/bin/env python3
"""Cruise + geofence controller for Parrot ANAFI 4K via Olympe.

Design (matches the agreed plan):
  * The drone is localized IN THE SfM MAP FRAME by a visual relocalizer
    (MegaLoc + ALIKED/XFeat + LightGlue + PnP -- a SEPARATE module, see Localizer).
  * Safe zone = an INNER trigger polygon (the true flyable zone shrunk inward by
    >= the drone's stopping/turning distance) + an altitude band [z_min, z_max].
    Working with the INNER boundary makes the test scale-free: we only ever ask
    "inside polygon?" (point-in-polygon) -- no metric scale needed.
  * "Approaching the boundary" is handled by a SLOW BAND just inside the inner
    polygon: speed ramps to 0 as the signed distance to the boundary shrinks,
    and the heading steers back toward the interior. Because we fly slowly,
    overshoot past the inner boundary stays inside the TRUE zone.

All geofence distances (SLOW_BAND, insets) are in MAP UNITS and tuned empirically
by flying at max speed and watching the overshoot -- you never need meters.

!!! SAFETY -- READ BEFORE FLYING !!!
  * Test FIRST with --dry-run (no takeoff: prints commands), then PROPS-OFF, then
    in a large open area with a human holding the controller for manual override.
  * If localization is stale/lost the controller HOVERS (never flies blind), and
    lands after LOST_LAND_S seconds of no pose.
  * Speeds are intentionally tiny. Increase only after validating behavior.
  * Verify the PCMD sign conventions (YAW_SIGN, pitch/gaz) on YOUR airframe first.
"""
from __future__ import annotations

import argparse
import math
import signal
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CONFIG -- tune these on your setup (all geofence lengths are in MAP UNITS)
# ---------------------------------------------------------------------------
DRONE_IP = "192.168.42.1"      # ANAFI direct WiFi; via SkyController use 192.168.53.1
CTRL_HZ = 20                   # PCMD send rate (ANAFI hovers if you stop sending)

# PCMD magnitudes are PERCENT of max tilt/rate [-100,100]. Keep SMALL = slow.
CRUISE_PITCH = 8               # forward tilt % at full cruise (slow!)
MAX_YAW_RATE = 25              # yaw command % cap
ALT_GAZ = 12                   # vertical command % cap

# Geofence (MAP UNITS). SLOW_BAND: distance inside the inner polygon where we
# start slowing + steering in. Tune so that at top speed the drone stops before
# leaving the TRUE zone. Bigger = safer/more conservative.
SLOW_BAND = 0.6
# Altitude band margins (MAP UNITS) inside [Z_MIN, Z_MAX] where we slow vertical.
ALT_BAND = 0.3

# Control gains (tune empirically; units are map-frame).
K_YAW = 1.4                    # heading P-gain (rad -> command scale)
K_ALT = 1.0                    # altitude P-gain (map-units -> gaz scale)
YAW_SIGN = -1                  # PCMD yaw>0 = clockwise; map yaw grows CCW. VERIFY!

# Localization-loss safety.
POSE_STALE_S = 0.5             # pose older than this -> treat as no pose -> hover
LOST_LAND_S = 4.0              # no valid pose for this long -> auto-land

# Altitude hold target (MAP UNITS, inside the band). Set to your cruise height.
ALT_HOLD = None                # None -> hold first observed z


# ---------------------------------------------------------------------------
# Pose + Localizer interface  (the visual reloc module plugs in HERE)
# ---------------------------------------------------------------------------
@dataclass
class Pose:
    x: float            # map-frame position (map units)
    y: float
    z: float            # map-frame height (up positive)
    yaw: float          # map-frame heading, radians, CCW from +X (forward axis)
    stamp: float        # time.monotonic() when this pose was produced


class Localizer:
    """Interface to the visual relocalizer running in the SfM map frame.

    Implement get_pose() to return the LATEST drone Pose in MAP coordinates, or
    None if no fresh fix is available. Wire this to:
        MegaLoc retrieval -> ALIKED/XFeat + LightGlue -> PnP against the reloc map
    (optionally fused with onboard VIO/IMU for smooth high-rate pose; the map
    reloc supplies the absolute, drift-free anchor at a few Hz).

    The controller is rate-decoupled from the localizer: it reuses the last pose
    until POSE_STALE_S, then hovers.
    """

    def get_pose(self) -> Pose | None:        # pragma: no cover - integration point
        raise NotImplementedError("Plug in the visual relocalizer here.")


# ---------------------------------------------------------------------------
# 2.5D safe zone:  horizontal polygon (INNER trigger) + altitude band
# ---------------------------------------------------------------------------
class SafeZone2p5D:
    """Inner trigger polygon (map XY) + [z_min, z_max]. Scale-free inside test
    plus a signed distance for the slow band."""

    def __init__(self, polygon_xy: list[tuple[float, float]], z_min: float, z_max: float):
        assert len(polygon_xy) >= 3, "polygon needs >= 3 vertices"
        self.poly = [(float(x), float(y)) for x, y in polygon_xy]
        self.z_min, self.z_max = float(z_min), float(z_max)

    def _point_in_poly(self, x: float, y: float) -> bool:
        inside = False
        n = len(self.poly)
        j = n - 1
        for i in range(n):
            xi, yi = self.poly[i]
            xj, yj = self.poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _closest_on_segment(px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        t = 0.0 if denom < 1e-12 else ((px - ax) * abx + (py - ay) * aby) / denom
        t = max(0.0, min(1.0, t))
        return ax + t * abx, ay + t * aby

    def signed_distance_inward(self, x: float, y: float):
        """Return (d, ux, uy): signed distance to the polygon boundary (>0 inside)
        and a unit vector pointing toward the interior (deeper-in if inside, back
        in if outside)."""
        best_d2 = float("inf")
        bx_, by_ = x, y
        n = len(self.poly)
        j = n - 1
        for i in range(n):
            cx, cy = self._closest_on_segment(x, y, *self.poly[j], *self.poly[i])
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if d2 < best_d2:
                best_d2, bx_, by_ = d2, cx, cy
            j = i
        dist = math.sqrt(best_d2)
        sign = 1.0 if self._point_in_poly(x, y) else -1.0
        dx, dy = x - bx_, y - by_
        norm = math.hypot(dx, dy)
        if norm < 1e-9:                      # on the boundary -> aim at centroid
            cx = sum(p[0] for p in self.poly) / n
            cy = sum(p[1] for p in self.poly) / n
            dx, dy = cx - x, cy - y
            norm = math.hypot(dx, dy) + 1e-9
            sign = 1.0
        ux, uy = sign * dx / norm, sign * dy / norm
        return sign * dist, ux, uy


# ---------------------------------------------------------------------------
# Cruise controller: pose -> (roll, pitch, yaw, gaz) PCMD percents
# ---------------------------------------------------------------------------
def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class CruiseController:
    def __init__(self, zone: SafeZone2p5D):
        self.zone = zone
        self.alt_hold = ALT_HOLD

    def compute(self, pose: Pose) -> tuple[int, int, int, int]:
        if self.alt_hold is None:
            self.alt_hold = pose.z

        d, ux, uy = self.zone.signed_distance_inward(pose.x, pose.y)

        # --- horizontal: pick travel direction + speed factor ---
        if d >= SLOW_BAND:
            # deep inside: patrol straight ahead (current heading), full slow-cruise
            tx, ty = math.cos(pose.yaw), math.sin(pose.yaw)
            speed = 1.0
        elif d > 0.0:
            # in the slow band: ramp speed down, steer increasingly inward
            speed = d / SLOW_BAND
            fwd_x, fwd_y = math.cos(pose.yaw), math.sin(pose.yaw)
            w = 1.0 - speed                       # more inward as we near the edge
            tx, ty = (1 - w) * fwd_x + w * ux, (1 - w) * fwd_y + w * uy
        else:
            # crossed the inner boundary: turn around, head straight back in
            tx, ty = ux, uy
            speed = 1.0

        tnorm = math.hypot(tx, ty) + 1e-9
        tx, ty = tx / tnorm, ty / tnorm

        # face the target direction (yaw), advance only when roughly facing it
        yaw_err = _wrap_pi(math.atan2(ty, tx) - pose.yaw)
        yaw_cmd = YAW_SIGN * _clamp(K_YAW * yaw_err, -1.0, 1.0) * MAX_YAW_RATE
        facing = max(0.0, math.cos(yaw_err))      # 1 when facing, 0 at 90deg+
        pitch_cmd = CRUISE_PITCH * speed * facing

        # --- vertical: hold ALT_HOLD, slow near the altitude band edges ---
        z_lo, z_hi = self.zone.z_min + ALT_BAND, self.zone.z_max - ALT_BAND
        if z_lo > z_hi:                          # band thinner than 2*ALT_BAND -> collapse to mid
            z_lo = z_hi = 0.5 * (self.zone.z_min + self.zone.z_max)
        z_target = _clamp(self.alt_hold, z_lo, z_hi)
        gaz_cmd = _clamp(K_ALT * (z_target - pose.z), -1.0, 1.0) * ALT_GAZ

        return (0, int(round(pitch_cmd)), int(round(yaw_cmd)), int(round(gaz_cmd)))


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------
# Olympe wiring
# ---------------------------------------------------------------------------
def run(localizer: Localizer, zone: SafeZone2p5D, dry_run: bool = False):
    ctrl = CruiseController(zone)
    period = 1.0 / CTRL_HZ
    last_good = None       # (Pose) most recent fresh pose
    lost_since = None

    drone = None
    if not dry_run:
        raise SystemExit(
            "cruise_geofence.py legacy TakeOff is permanently locked; "
            "use the operator-approved flight path instead"
        )

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    def send(roll, pitch, yaw, gaz):
        if dry_run:
            print(f"[PCMD] roll={roll:+d} pitch={pitch:+d} yaw={yaw:+d} gaz={gaz:+d}")
            return
        from olympe.messages.ardrone3.Piloting import PCMD
        drone(PCMD(1, roll, pitch, yaw, gaz, 0))

    try:
        while not stop["flag"]:
            t0 = time.monotonic()
            p = localizer.get_pose()
            now = time.monotonic()
            fresh = p is not None and (now - p.stamp) <= POSE_STALE_S
            if fresh:
                last_good, lost_since = p, None
            elif last_good is not None and (now - last_good.stamp) <= POSE_STALE_S:
                p, fresh = last_good, True       # reuse very recent pose

            if fresh:
                roll, pitch, yaw, gaz = ctrl.compute(p)
                send(roll, pitch, yaw, gaz)
            else:
                send(0, 0, 0, 0)                 # HOVER -- never fly blind
                lost_since = lost_since or now
                if (now - lost_since) >= LOST_LAND_S:
                    print("[cruise] localization lost too long -> landing")
                    break

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        if not dry_run and drone is not None:
            from olympe.messages.ardrone3.Piloting import Landing
            print("[cruise] landing ...")
            try:
                send(0, 0, 0, 0)                 # zero PCMD before landing
                res = drone(Landing()).wait(_timeout=20)
                if hasattr(res, "success") and not res.success():
                    print("[cruise] warning: legacy cruise_geofence landing did not report success")
            except (Exception, SystemExit) as exc:
                print(f"[cruise] warning: landing raised: {exc}")
            try:
                drone.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Example wiring (replace DemoLocalizer + polygon with your real ones)
# ---------------------------------------------------------------------------
class DemoLocalizer(Localizer):
    """Dry-run only: a fake circular flight to exercise the controller logic."""

    def __init__(self):
        self.t0 = time.monotonic()

    def get_pose(self) -> Pose | None:
        t = time.monotonic() - self.t0
        return Pose(x=2.0 * math.cos(0.3 * t), y=2.0 * math.sin(0.3 * t),
                    z=1.0, yaw=0.3 * t + math.pi / 2, stamp=time.monotonic())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no takeoff; print PCMD")
    args = ap.parse_args()

    # TODO: replace with your INNER trigger polygon (map XY) + altitude band,
    # drawn on the dense MVS map (CloudCompare/Blender), in MAP UNITS.
    inner_polygon = [(-3, -3), (3, -3), (3, 3), (-3, 3)]
    zone = SafeZone2p5D(inner_polygon, z_min=0.4, z_max=2.5)

    # TODO: replace DemoLocalizer() with your visual relocalizer.
    localizer = DemoLocalizer()

    run(localizer, zone, dry_run=args.dry_run)
