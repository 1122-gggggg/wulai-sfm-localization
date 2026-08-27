#!/usr/bin/env python3
"""Autonomous pole-inspection flight for Parrot ANAFI 4K (Olympe).

Two layers (agreed design):
  GLOBAL  : plan_path.plan_tour -> a clearance-preferring path visiting the pole
            STANDOFF points (safe zone + pole no-go baked into the SDF).
  LOCAL   : this reactive controller follows the path while adjusting live to
            SDF clearance (geofence repulsion), YOLO pole bearing+area (final
            approach to standoff), and localization. Hard safety override =
            stop / hover / recover-toward-open-space.

State machine:  NAV -> APPROACH -> INSPECT -> (next target) -> ... -> DONE
                any state -> RECOVER (clearance <= STOP_MARGIN) -> back
                any state -> HOVER (pose stale) -> LAND (lost too long)

Frames: map XY horizontal, Z up; yaw = heading of forward axis, CCW from +X.
All geofence/path lengths are in MAP UNITS (scale-free). Only the YOLO standoff
(bbox area target) is a metric-ish proxy and is *backed up* by the SDF no-go.

!!! SAFETY: validate with --dry-run (built-in sim), then PROPS-OFF, then open
area with manual-override pilot. VERIFY PCMD signs (YAW_SIGN/pitch/gaz) first.
Speeds are deliberately tiny. !!!
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEPLOY_ROOT = Path(__file__).resolve().parents[1] / "deploy_code" / "sfm_glomap_deploy"
if DEPLOY_ROOT.is_dir() and str(DEPLOY_ROOT) not in sys.path:
    sys.path.append(str(DEPLOY_ROOT))

from plan_path import SDFGrid, plan_tour

# ----------------------------- CONFIG --------------------------------------
DRONE_IP = "192.168.42.1"  # ANAFI WiFi; SkyController -> 192.168.53.1
CTRL_HZ = 20

CRUISE_PITCH = 8  # forward tilt % at full cruise (slow!)
MAX_YAW_RATE = 25  # yaw command % cap
ALT_GAZ = 12  # vertical command % cap
K_YAW = 1.4  # heading P-gain
K_VERT = 1.0  # vertical P-gain (map-units -> gaz scale)
YAW_SIGN = +1  # matches the built-in sim; VERIFY on real ANAFI!

# SDF geofence thresholds (MAP UNITS)
SLOW_BAND = 1.0  # start slowing + repelling below this clearance
STOP_MARGIN = 0.35  # <= this -> stop/hover/recover (touched boundary)
RESUME_MARGIN = 0.8  # recover until clearance >= this (hysteresis)
REPULSE_W = 1.5  # weight of SDF-gradient repulsion vs path direction

# path following / mission (MAP UNITS unless noted)
LOOKAHEAD = 1.5  # carrot distance along the global path
ARRIVAL_R = 1.0  # within this of a target standoff -> APPROACH
AREA_TARGET = 0.18  # YOLO bbox area fraction == standoff reached
APPROACH_AREA = 0.04  # bbox area fraction that means "pole is close enough to approach"
INSPECT_SECS = 4.0
DETECT_LOST_S = 1.0  # APPROACH: detection lost this long -> back to NAV

# safety
POSE_STALE_S = 0.5
LOST_LAND_S = 4.0


# ----------------------------- interfaces ----------------------------------
from pose_types import (
    Pose,
    Localizer,
)  # re-export; types moved to pose_types (decouples localizer from plan_path)


@dataclass
class PoleDetection:
    u_off: float  # horizontal bbox-center offset in [-1,1] (0 = image center)
    v_off: float  # vertical offset in [-1,1] (0 = center, + = upper)
    area_frac: float  # bbox area / image area  (proxy for inverse distance)
    score: float


class PoleDetector:
    def detect(self) -> PoleDetection | None:
        return None  # default: no YOLO -> NAV arrives by proximity only


# ----------------------------- helpers -------------------------------------
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PathFollower:
    """Carrot-point follower over the global polyline (world coords)."""

    def __init__(self, path):
        self.path = [np.asarray(p, float) for p in path]

    def carrot(self, P):
        P = np.asarray(P, float)
        # nearest vertex, then look ahead LOOKAHEAD along the polyline
        di = min(range(len(self.path)), key=lambda i: np.linalg.norm(self.path[i] - P))
        acc, idx = 0.0, di
        while idx < len(self.path) - 1 and acc < LOOKAHEAD:
            acc += np.linalg.norm(self.path[idx + 1] - self.path[idx])
            idx += 1
        return self.path[idx]


# ----------------------------- controller ----------------------------------
class AutoFlight:
    def __init__(self, sdf: SDFGrid, follower: PathFollower, targets):
        self.sdf = sdf
        self.follower = follower
        self.targets = [np.asarray(t, float) for t in targets]  # pole standoff pts
        self.cur = 0
        self.state = "NAV"
        self.prev_state = "NAV"
        self._t_inspect = None
        self._t_detect_seen = None

    def _horizontal_dir(self, P, base_dir_xy, clearance):
        """Blend path/goal direction with SDF repulsion (toward open space)."""
        d = np.asarray(base_dir_xy, float)
        n = np.linalg.norm(d)
        d = d / n if n > 1e-9 else d
        if clearance < SLOW_BAND:
            g = self.sdf.gradient([P.x, P.y, P.z])[:2]  # toward higher clearance
            w = REPULSE_W * (1.0 - max(clearance, 0.0) / SLOW_BAND)
            d = d + w * g
            n = np.linalg.norm(d)
            d = d / n if n > 1e-9 else d
        return d

    def _to_pcmd(self, P, dir_xy, target_z, speed):
        """world horizontal dir + target altitude + speed -> (roll,pitch,yaw,gaz)."""
        yaw_err = _wrap(math.atan2(dir_xy[1], dir_xy[0]) - P.yaw)
        yaw_cmd = YAW_SIGN * _clamp(K_YAW * yaw_err, -1, 1) * MAX_YAW_RATE
        facing = max(0.0, math.cos(yaw_err))  # advance only when facing
        pitch_cmd = CRUISE_PITCH * speed * facing
        gaz_cmd = _clamp(K_VERT * (target_z - P.z), -1, 1) * ALT_GAZ
        return (0, int(round(pitch_cmd)), int(round(yaw_cmd)), int(round(gaz_cmd)))

    def _recover_command(self, clearance: float):
        if clearance >= RESUME_MARGIN:
            self.state = self.prev_state
            return None
        # On a clearance breach, hover for the operator. Autonomous gradient
        # motion would be an uncommanded move inside an uncertain obstacle region.
        return (0, 0, 0, 0, f"RECOVER hover c={clearance:.2f}")

    def _inspect_command(self, det: PoleDetection | None, now: float):
        self._t_inspect = self._t_inspect or now
        if now - self._t_inspect >= INSPECT_SECS:
            self._t_inspect = None
            self.cur += 1
            self.state = "DONE" if self.cur >= len(self.targets) else "NAV"
            return (0, 0, 0, 0, f"INSPECT done -> {self.state}")
        if det:
            yaw_cmd = YAW_SIGN * _clamp(-1.5 * det.u_off, -1, 1) * MAX_YAW_RATE
            gaz = _clamp(1.5 * det.v_off, -1, 1) * ALT_GAZ
            return (0, 0, int(yaw_cmd), int(gaz), "INSPECT hold")
        return (0, 0, 0, 0, "INSPECT hold")

    def _approach_command(
        self,
        det: PoleDetection | None,
        now: float,
        clearance: float,
    ):
        if det and det.score > 0.0:
            self._t_detect_seen = now
            if det.area_frac >= AREA_TARGET:
                self.state = "INSPECT"
                return (0, 0, 0, 0, "APPROACH -> INSPECT (standoff)")
            yaw_err_img = -det.u_off
            yaw_cmd = YAW_SIGN * _clamp(1.5 * yaw_err_img, -1, 1) * MAX_YAW_RATE
            speed = _clamp(1.0 - det.area_frac / AREA_TARGET, 0.1, 1.0)
            facing = max(0.0, 1.0 - abs(det.u_off))
            pitch = CRUISE_PITCH * speed * facing
            pitch *= _clamp(clearance / SLOW_BAND, 0.0, 1.0)
            gaz = _clamp(1.2 * det.v_off, -1, 1) * ALT_GAZ
            return (
                0,
                int(round(pitch)),
                int(round(yaw_cmd)),
                int(round(gaz)),
                f"APPROACH area={det.area_frac:.2f} c={clearance:.2f}",
            )
        if self._t_detect_seen and now - self._t_detect_seen > DETECT_LOST_S:
            self.state = "NAV"
        return (0, 0, 0, 0, "APPROACH wait-detect")

    def _nav_command(self, pose: Pose, now: float, clearance: float):
        target = self.targets[self.cur]
        position = np.array([pose.x, pose.y, pose.z])
        if np.linalg.norm(position - target) < ARRIVAL_R:
            self.state = "APPROACH"
            self._t_detect_seen = now
            return (0, 0, 0, 0, "NAV arrived -> APPROACH")
        carrot = self.follower.carrot(position)
        base = (carrot - position)[:2]
        direction = self._horizontal_dir(pose, base, clearance)
        speed = _clamp(clearance / SLOW_BAND, 0.0, 1.0)
        command = self._to_pcmd(pose, direction, carrot[2], speed)
        return (*command, f"NAV->t{self.cur} c={clearance:.2f}")

    def step(self, P: Pose, det: PoleDetection | None, now: float):
        """Return (roll,pitch,yaw,gaz, info)."""
        clearance = self.sdf.clearance([P.x, P.y, P.z])

        # ---- hard safety override: clearance breached -> RECOVER ----
        if clearance <= STOP_MARGIN and self.state != "RECOVER":
            self.prev_state = self.state
            self.state = "RECOVER"

        if self.state == "RECOVER":
            command = self._recover_command(clearance)
            if command is not None:
                return command

        if self.state == "DONE":
            return (0, 0, 0, 0, "DONE hover")

        if self.state == "INSPECT":
            return self._inspect_command(det, now)

        if self.state == "APPROACH":
            return self._approach_command(det, now, clearance)
        return self._nav_command(P, now, clearance)


# ----------------------------- run loop ------------------------------------
def _send_command(dry_run, sim, drone, period, roll, pitch, yaw, gaz) -> None:
    if dry_run:
        if sim:
            sim.step(roll, pitch, yaw, gaz, period)
        return
    from olympe.messages.ardrone3.Piloting import PCMD

    drone(PCMD(1, roll, pitch, yaw, gaz, 0))


def _run_controller_loop(
    loc: Localizer,
    detector: PoleDetector,
    ctrl: AutoFlight,
    *,
    dry_run: bool,
    sim,
    drone,
    stop: dict,
    period: float,
) -> None:
    last_good, lost_since = None, None
    steps = 0
    while not stop["f"]:
        started = time.monotonic()
        pose = loc.get_pose()
        now = time.monotonic()
        fresh = pose is not None and (now - pose.stamp) <= POSE_STALE_S
        if fresh:
            last_good, lost_since = pose, None
        elif last_good and (now - last_good.stamp) <= POSE_STALE_S:
            pose, fresh = last_good, True

        if not fresh:
            _send_command(dry_run, sim, drone, period, 0, 0, 0, 0)
            lost_since = lost_since or now
            if now - lost_since >= LOST_LAND_S:
                print("[auto] localization lost -> land")
                break
        else:
            roll, pitch, yaw, gaz, info = ctrl.step(pose, detector.detect(), now)
            _send_command(dry_run, sim, drone, period, roll, pitch, yaw, gaz)
            if dry_run and steps % 10 == 0:
                print(
                    f"[{info:28s}] pos=({pose.x:5.1f},{pose.y:5.1f},{pose.z:4.1f}) "
                    f"yaw={math.degrees(pose.yaw):6.1f} "
                    f"PCMD(p={pitch:+d},y={yaw:+d},g={gaz:+d})"
                )
            if ctrl.state == "DONE":
                print("[auto] all targets inspected -> land")
                break
        steps += 1
        elapsed = time.monotonic() - started
        if elapsed < period and not dry_run:
            time.sleep(period - elapsed)


def _cleanup_legacy_flight(dry_run: bool, drone, airborne: bool, period: float) -> None:
    if dry_run or drone is None:
        return
    from olympe.messages.ardrone3.Piloting import Landing

    if airborne:
        print("[auto] landing")
        try:
            _send_command(False, None, drone, period, 0, 0, 0, 0)
        except Exception as exc:
            print(f"[auto] zero PCMD before landing FAILED: {exc!r}", flush=True)
        try:
            result = drone(Landing()).wait(_timeout=20)
            if hasattr(result, "success") and not result.success():
                print("[auto] warning: legacy autoflight landing did not report success")
        except (Exception, SystemExit) as exc:
            print(f"[auto] warning: landing raised: {exc}")
    try:
        drone.disconnect()
    except Exception:
        pass


def run(loc: Localizer, det: PoleDetector, ctrl: AutoFlight, dry_run=False, sim=None):
    period = 1.0 / CTRL_HZ
    drone = None
    airborne = False
    stop = {"f": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("f", True))
    try:
        if not dry_run:
            raise SystemExit(
                "autoflight.py legacy TakeOff is permanently locked; "
                "use the operator-approved flight path instead"
            )
        _run_controller_loop(
            loc,
            det,
            ctrl,
            dry_run=dry_run,
            sim=sim,
            drone=drone,
            stop=stop,
            period=period,
        )
    finally:
        _cleanup_legacy_flight(dry_run, drone, airborne, period)


# ----------------------------- dry-run sim ---------------------------------
class _Sim:
    """Toy kinematics so --dry-run exercises the full state machine (NOT physics)."""

    KP, KG, KY = 0.06, 0.05, 0.06  # forward / vertical / yaw response per % per step

    def __init__(self, x, y, z, yaw):
        self.x, self.y, self.z, self.yaw = x, y, z, yaw

    def step(self, roll, pitch, yaw, gaz, dt):
        self.yaw = _wrap(self.yaw + self.KY * yaw * dt)
        v = self.KP * pitch
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.z += self.KG * gaz * dt

    def pose(self):
        return Pose(self.x, self.y, self.z, self.yaw, time.monotonic())


class _SimLocalizer(Localizer):
    def __init__(self, sim):
        self.sim = sim

    def get_pose(self):
        return self.sim.pose()


class _SimPoleDetector(PoleDetector):
    """Fakes YOLO from the sim pose vs a known pole world position."""

    def __init__(self, sim, pole_xy, max_see=8.0):
        self.sim, self.pole = sim, np.asarray(pole_xy, float)
        self.max_see = max_see

    def detect(self):
        dxy = self.pole - np.array([self.sim.x, self.sim.y])
        dist = float(np.linalg.norm(dxy))
        if dist > self.max_see:
            return None
        bearing = _wrap(math.atan2(dxy[1], dxy[0]) - self.sim.yaw)
        if abs(bearing) > math.radians(60):
            return None  # outside FOV
        u = _clamp(bearing / math.radians(40), -1, 1)
        area = _clamp((1.0 / max(dist, 0.5)) * 1.5, 0.0, 0.5)  # closer -> bigger
        return PoleDetection(u_off=u, v_off=0.0, area_frac=area, score=0.9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # synthetic SDF: 30x30x10 box, pole no-go cylinder (r+buffer=3) at (15,15)
    vox, L, H = 0.5, 30.0, 10.0
    nx, ny, nz = int(L / vox), int(L / vox), int(H / vox)
    sdf = np.empty((nx, ny, nz), np.float32)
    for i in range(nx):
        x = i * vox
        for j in range(ny):
            y = j * vox
            dpole = math.hypot(x - 15, y - 15) - 3.0
            for k in range(nz):
                z = k * vox
                sdf[i, j, k] = min(min(x, L - x, y, L - y, z, H - z), dpole)
    grid = SDFGrid(sdf, np.zeros(3), vox)

    start = (4, 4, 5)
    # pole standoff points = ring around the pole at the box, here just two probe spots
    targets = [(11.5, 15, 5), (15, 11.5, 5)]  # standoff points near the pole no-go
    path, order = plan_tour(grid, start, targets, standoff_min=0.0, pref_clear=2.0, risk_w=3.0)
    # NOTE: plan_tour already smooths each leg; do NOT smooth the full tour again
    # (a global shortcut could skip a mandatory target).
    print(f"[plan] tour pts={len(path)} order={[tuple(map(float, o)) for o in order]}")

    follower = PathFollower(path)
    ctrl = AutoFlight(grid, follower, targets)

    if args.dry_run:
        sim = _Sim(start[0], start[1], start[2], 0.0)
        run(_SimLocalizer(sim), _SimPoleDetector(sim, (15, 15)), ctrl, dry_run=True, sim=sim)
    else:
        # This legacy harness intentionally has no live integration. Approved
        # localization and detection are wired through the operator pipeline.
        raise SystemExit(
            "real flight needs a Localizer + PoleDetector; use --dry-run to test logic"
        )
