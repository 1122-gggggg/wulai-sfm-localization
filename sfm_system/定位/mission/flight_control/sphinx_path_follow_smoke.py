#!/usr/bin/env python3
"""Path-follow smoke test against Parrot Sphinx.

This uses Sphinx/Olympe telemetry as the pose source. It verifies that the
existing route controller can command a simulated ANAFI along a preplanned path.
It does not validate SfM/XFeat visual relocalization in an arbitrary UE scene.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import olympe
from olympe.messages.ardrone3.Piloting import Landing, PCMD, TakeOff
from olympe.messages.ardrone3.PilotingState import (
    AttitudeChanged,
    FlyingStateChanged,
    PositionChanged,
)

import real_path_follow_controller as rpf
from path_follow_flight import DRONE_IP_SIM, LoopHooks, run_loop


EARTH_R_M = 6378137.0


@dataclass
class Origin:
    lat: float
    lon: float
    alt: float


def _valid_position(st: dict | None) -> bool:
    if not st:
        return False
    lat = float(st.get("latitude", 500.0))
    lon = float(st.get("longitude", 500.0))
    alt = float(st.get("altitude", 500.0))
    return all(math.isfinite(v) for v in (lat, lon, alt)) and abs(lat) <= 90.0 and abs(lon) <= 180.0


def _gps_to_raw_map(st: dict, origin: Origin) -> np.ndarray:
    lat = math.radians(float(st["latitude"]))
    lon = math.radians(float(st["longitude"]))
    lat0 = math.radians(origin.lat)
    lon0 = math.radians(origin.lon)
    north = (lat - lat0) * EARTH_R_M
    east = (lon - lon0) * EARTH_R_M * math.cos(lat0)
    up = float(st["altitude"]) - origin.alt
    return np.array([north, -up, east], dtype=float)


class TelemetryPoseSource:
    def __init__(self, drone):
        self.drone = drone
        self.origin: Origin | None = None

    def get_pose(self):
        try:
            st = self.drone.get_state(PositionChanged)
        except (KeyError, RuntimeError):
            return None
        if not _valid_position(st):
            return None
        if self.origin is None:
            self.origin = Origin(
                lat=float(st["latitude"]),
                lon=float(st["longitude"]),
                alt=float(st["altitude"]),
            )
        c = _gps_to_raw_map(st, self.origin)
        return rpf.Pose(x=float(c[0]), y=float(c[1]), z=float(c[2]), yaw=0.0, stamp=time.monotonic())

    def yaw(self) -> float | None:
        try:
            st = self.drone.get_state(AttitudeChanged)
        except (KeyError, RuntimeError):
            return None
        if not st:
            return None
        yaw = float(st["yaw"])
        return yaw if math.isfinite(yaw) else None


def wait_success(expectation, label: str) -> None:
    res = expectation.wait()
    if hasattr(res, "success") and not res.success():
        raise SystemExit(f"{label} failed or timed out")


def wait_for_pose(source: TelemetryPoseSource, timeout_s: float):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        pose = source.get_pose()
        if pose is not None:
            return pose
        time.sleep(0.1)
    raise SystemExit(f"no valid Sphinx/Olympe position after {timeout_s:.1f}s")


def build_path(c0: np.ndarray, side_m: float, pattern: str) -> list[np.ndarray]:
    side = float(side_m)
    if pattern == "line":
        offsets = [(0.0, 0.0), (side, 0.0)]
    else:
        offsets = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)]
    return [c0 + np.array([north, 0.0, east], dtype=float) for north, east in offsets]


def raw_to_aligned(p: np.ndarray) -> list[float]:
    return [float(p[0]), float(p[2]), float(-p[1])]


def save_path(path: list[np.ndarray], out: Path, pattern: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "waypoints": [raw_to_aligned(p) for p in path],
        "closed": False,
        "frame": "aligned",
        "units": "meters-local-sphinx",
        "source": "sphinx_path_follow_smoke.py",
        "pattern": pattern,
        "note": "Controller test path generated relative to the simulated ANAFI takeoff pose.",
    }
    out.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sphinx ANAFI route-follow smoke test.")
    parser.add_argument("--ip", default=DRONE_IP_SIM)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--side", type=float, default=2.5)
    parser.add_argument("--pattern", choices=("line", "square"), default="line")
    parser.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--path-out",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "sphinx_empty" / "flight_path.json"),
    )
    args = parser.parse_args()

    drone = olympe.Drone(args.ip)
    if not drone.connect():
        raise SystemExit(f"could not connect to Sphinx ANAFI at {args.ip}")

    reason = "stopped"
    try:
        wait_success(drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=15)), "takeoff")
        source = TelemetryPoseSource(drone)
        p0 = wait_for_pose(source, 10.0)
        c0 = np.array([p0.x, p0.y, p0.z], dtype=float)
        path = build_path(c0, args.side, args.pattern)
        save_path(path, Path(args.path_out), args.pattern)

        cfg = rpf.ControlConfig(
            speed=0.7,
            lookahead=0.6,
            rejoin_tol=0.75,
            arrive=0.35,
            arrive_fraction=0.08,
            min_arrive=0.20,
            inspect_waypoints=(),
        )
        ctrl = rpf.RouteAutoController(path, poles=[], config=cfg)

        def send_pcmd(roll, pitch, yaw, gaz):
            drone(PCMD(1, int(roll), int(pitch), int(yaw), int(gaz), 0))

        end_t = time.monotonic() + float(args.duration)

        def capped_now():
            if time.monotonic() >= end_t:
                raise KeyboardInterrupt
            return time.monotonic()

        hooks = LoopHooks(
            get_pose=source.get_pose,
            olympe_yaw=source.yaw,
            send_pcmd=send_pcmd,
            stream_healthy=lambda: True,
            now=capped_now,
        )
        try:
            reason = run_loop(hooks, ctrl, path, yaw_sign=args.yaw_sign, verbose=True)
        except KeyboardInterrupt:
            reason = f"duration {args.duration:.1f}s reached -> land"
        print(f"[sphinx-smoke] {reason}")
        print(f"[sphinx-smoke] path saved: {args.path_out}")
        return 0
    finally:
        try:
            drone(PCMD(1, 0, 0, 0, 0, 0))
        except Exception:
            pass
        try:
            drone(Landing()).wait()
        except Exception as exc:
            print(f"[sphinx-smoke] warning: Landing failed: {exc}", flush=True)
        drone.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
