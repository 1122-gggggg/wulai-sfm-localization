#!/usr/bin/env python3
"""Path-follow smoke test against Parrot Sphinx.

This uses Sphinx/Olympe telemetry as the pose source. It verifies that the
existing route controller can command a simulated ANAFI along a preplanned path.
It does not validate SfM/XFeat visual relocalization in an arbitrary UE scene.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")


def _find_workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "定位演算法").is_dir()
            and (parent / "模擬器" / "sphinx_anafi_path_convergence").is_dir()
        ):
            return parent
    raise RuntimeError("could not locate localization workspace root")


WORKSPACE_ROOT = _find_workspace_root()
EXPERIMENT_DIR = WORKSPACE_ROOT / "模擬器" / "sphinx_anafi_path_convergence"
FLIGHT_CONTROL_DIR = WORKSPACE_ROOT / "定位演算法" / "flight_control"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(FLIGHT_CONTROL_DIR))
from telemetry_sources import SphinxTelemetrySource  # noqa: E402

import real_path_follow_controller as rpf
from path_follow_flight import DRONE_IP_SIM, LoopHooks, run_loop


SPHINX_NETWORK = ipaddress.ip_network("10.202.0.0/16")


class TelemetryPoseSource:
    def __init__(self, drone):
        self.inner = SphinxTelemetrySource(drone)

    def get_pose(self):
        pose = self.inner.get_pose(time.monotonic())
        if pose is None:
            return None
        return rpf.Pose(
            x=pose.x, y=pose.y, z=pose.z, yaw=0.0,
            stamp=pose.stamp,
        )

    def yaw(self) -> float | None:
        return self.inner.yaw(time.monotonic())

    def telemetry_healthy(self) -> bool:
        return self.inner.telemetry_healthy(time.monotonic())


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


def require_sphinx_ip(value: str) -> str:
    """Accept only a valid IPv4 address in the Sphinx 10.202.0.0/16 network."""
    try:
        parsed = ipaddress.ip_address(str(value))
    except ValueError:
        parsed = None
    if isinstance(parsed, ipaddress.IPv4Address) and parsed in SPHINX_NETWORK:
        return str(parsed)
    raise SystemExit(
        f"sphinx_path_follow_smoke: --ip {value} is outside Sphinx network "
        f"{SPHINX_NETWORK}; this arming path has no real-aircraft override."
    )


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

    for name in ("duration", "side"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name} must be finite and > 0")

    # This script arms motors with no localization safety gates. It is a Sphinx
    # simulator smoke test only and has no real-aircraft escape hatch.
    args.ip = require_sphinx_ip(args.ip)
    print("[mode] SPHINX-SMOKE: simulator-only arming path (sim IP enforced)", flush=True)

    import olympe
    from olympe.messages.ardrone3.Piloting import Landing, PCMD, TakeOff
    from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

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
            stream_healthy=source.telemetry_healthy,
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
