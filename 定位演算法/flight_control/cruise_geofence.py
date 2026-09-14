#!/usr/bin/env python3
"""Cruise + geofence controller for Parrot ANAFI 4K via Olympe.

Legacy patrol controller (TakeOff permanently locked). The old SLOW BAND is
gone: there is no gradual slowdown or inward steering near the polygon or
altitude edges. Inside the inner trigger polygon the drone patrols straight
ahead at full slow-cruise; outside it heads straight back in at full speed,
and the altitude hold clamps directly to [z_min, z_max]. Production AUTO
uses real_path_follow_controller, not this file.

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
DRONE_IP = "192.168.42.1"  # ANAFI direct WiFi; via SkyController use 192.168.53.1
CTRL_HZ = 20  # PCMD send rate (ANAFI hovers if you stop sending)

# PCMD magnitudes are PERCENT of max tilt/rate [-100,100]. Keep SMALL = slow.
CRUISE_PITCH = 8  # forward tilt % at full cruise (slow!)
MAX_YAW_RATE = 25  # yaw command % cap
ALT_GAZ = 12  # vertical command % cap

# The old SLOW_BAND/ALT_BAND gradual slowdown is removed: boundary proximity
# never scales speed. The polygon test below is a hard inside/outside switch.

# Control gains (tune empirically; units are map-frame).
K_YAW = 1.4  # heading P-gain (rad -> command scale)
K_ALT = 1.0  # altitude P-gain (map-units -> gaz scale)
YAW_SIGN = -1  # PCMD yaw>0 = clockwise; map yaw grows CCW. VERIFY!

# Localization-loss safety.
POSE_STALE_S = 0.5  # pose older than this -> treat as no pose -> hover
LOST_LAND_S = 4.0  # no valid pose for this long -> auto-land

# Altitude hold target (MAP UNITS, inside the band). Set to your cruise height.
ALT_HOLD = None  # None -> hold first observed z


# ---------------------------------------------------------------------------
# Pose + Localizer interface  (the visual reloc module plugs in HERE)
# ---------------------------------------------------------------------------
@dataclass
class Pose:
    x: float  # map-frame position (map units)
    y: float
    z: float  # map-frame height (up positive)
    yaw: float  # map-frame heading, radians, CCW from +X (forward axis)
    stamp: float  # time.monotonic() when this pose was produced


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

    def get_pose(self) -> Pose | None:  # pragma: no cover - integration point
        raise NotImplementedError("Plug in the visual relocalizer here.")


# ---------------------------------------------------------------------------
# 2.5D safe zone:  horizontal polygon (INNER trigger) + altitude band
# ---------------------------------------------------------------------------
class SafeZone2p5D:
    """Inner trigger polygon (map XY) + [z_min, z_max]. Hard inside/outside only."""

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

    def inside(self, x: float, y: float) -> bool:
        """Hard inside/outside test only; no distance, no slow band."""
        return self._point_in_poly(x, y)


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

        inside = self.zone.inside(pose.x, pose.y)
        if inside:
            # Inside: patrol straight ahead at full slow-cruise, no slow band.
            tx, ty = math.cos(pose.yaw), math.sin(pose.yaw)
        else:
            # Outside: head straight back toward the polygon centroid at full speed.
            n = len(self.zone.poly)
            cx = sum(p[0] for p in self.zone.poly) / n
            cy = sum(p[1] for p in self.zone.poly) / n
            tx, ty = cx - pose.x, cy - pose.y

        tnorm = math.hypot(tx, ty) + 1e-9
        tx, ty = tx / tnorm, ty / tnorm

        # face the target direction (yaw), advance only when roughly facing it
        yaw_err = _wrap_pi(math.atan2(ty, tx) - pose.yaw)
        yaw_cmd = YAW_SIGN * _clamp(K_YAW * yaw_err, -1.0, 1.0) * MAX_YAW_RATE
        facing = max(0.0, math.cos(yaw_err))  # 1 when facing, 0 at 90deg+
        pitch_cmd = CRUISE_PITCH * facing

        # --- vertical: hold ALT_HOLD clamped directly to [z_min, z_max] ---
        z_target = _clamp(self.alt_hold, self.zone.z_min, self.zone.z_max)
        gaz_cmd = _clamp(K_ALT * (z_target - pose.z), -1.0, 1.0) * ALT_GAZ

        return (0, int(round(pitch_cmd)), int(round(yaw_cmd)), int(round(gaz_cmd)))


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------
# Olympe wiring
# ---------------------------------------------------------------------------
def _send_cruise_command(dry_run, drone, roll, pitch, yaw, gaz) -> None:
    if dry_run:
        print(f"[PCMD] roll={roll:+d} pitch={pitch:+d} yaw={yaw:+d} gaz={gaz:+d}")
        return
    from olympe.messages.ardrone3.Piloting import PCMD

    drone(PCMD(1, roll, pitch, yaw, gaz, 0))


def _run_cruise_loop(localizer, ctrl, *, dry_run, drone, stop, period) -> None:
    last_good = None
    lost_since = None
    while not stop["flag"]:
        started = time.monotonic()
        pose = localizer.get_pose()
        now = time.monotonic()
        fresh = pose is not None and (now - pose.stamp) <= POSE_STALE_S
        if fresh:
            last_good, lost_since = pose, None
        elif last_good is not None and (now - last_good.stamp) <= POSE_STALE_S:
            pose, fresh = last_good, True

        if fresh:
            command = ctrl.compute(pose)
            _send_cruise_command(dry_run, drone, *command)
        else:
            _send_cruise_command(dry_run, drone, 0, 0, 0, 0)
            lost_since = lost_since or now
            if (now - lost_since) >= LOST_LAND_S:
                print("[cruise] localization lost too long -> landing")
                break

        elapsed = time.monotonic() - started
        if elapsed < period:
            time.sleep(period - elapsed)


def _cleanup_cruise_flight(dry_run: bool, drone) -> None:
    if dry_run or drone is None:
        return
    from olympe.messages.ardrone3.Piloting import Landing

    print("[cruise] landing ...")
    try:
        _send_cruise_command(False, drone, 0, 0, 0, 0)
        result = drone(Landing()).wait(_timeout=20)
        if hasattr(result, "success") and not result.success():
            print("[cruise] warning: legacy cruise_geofence landing did not report success")
    except (Exception, SystemExit) as exc:
        print(f"[cruise] warning: landing raised: {exc}")
    try:
        drone.disconnect()
    except Exception:
        pass


def run(localizer: Localizer, zone: SafeZone2p5D, dry_run: bool = False):
    ctrl = CruiseController(zone)
    period = 1.0 / CTRL_HZ
    drone = None
    if not dry_run:
        raise SystemExit(
            "cruise_geofence.py legacy TakeOff is permanently locked; "
            "use the operator-approved flight path instead"
        )
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    try:
        _run_cruise_loop(
            localizer,
            ctrl,
            dry_run=dry_run,
            drone=drone,
            stop=stop,
            period=period,
        )
    finally:
        _cleanup_cruise_flight(dry_run, drone)


# ---------------------------------------------------------------------------
# Example wiring (replace DemoLocalizer + polygon with your real ones)
# ---------------------------------------------------------------------------
class DemoLocalizer(Localizer):
    """Dry-run only: a fake circular flight to exercise the controller logic."""

    def __init__(self):
        self.t0 = time.monotonic()

    def get_pose(self) -> Pose | None:
        t = time.monotonic() - self.t0
        return Pose(
            x=2.0 * math.cos(0.3 * t),
            y=2.0 * math.sin(0.3 * t),
            z=1.0,
            yaw=0.3 * t + math.pi / 2,
            stamp=time.monotonic(),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no takeoff; print PCMD")
    args = ap.parse_args()

    # Fixed demonstration geometry for the dry-run harness. The legacy live
    # path is permanently locked in run(); approved routes use the operator
    # pipeline instead of accepting ad-hoc polygons here.
    inner_polygon = [(-3, -3), (3, -3), (3, 3), (-3, 3)]
    zone = SafeZone2p5D(inner_polygon, z_min=0.4, z_max=2.5)

    # The deterministic demo localizer is intentional for this dry-run harness.
    localizer = DemoLocalizer()

    run(localizer, zone, dry_run=args.dry_run)
