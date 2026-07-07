#!/usr/bin/env python3
"""Mission authoring and ANAFI flight entrypoint.

This is the non-expert facing wrapper for:
  - drawing inspection routes;
  - marking poles/wires;
  - planning/checking the route;
  - running dry-run, live grab-only, or real ANAFI flight.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MISSION_ROOT = Path(__file__).resolve().parent
LOC_ROOT = MISSION_ROOT.parents[0]
SYSTEM_ROOT = LOC_ROOT.parent
AUTHOR = MISSION_ROOT / "authoring"
FLIGHT = MISSION_ROOT / "flight_control"
DEPLOY = LOC_ROOT / "deploy_code" / "sfm_glomap_deploy"
SCRIPTS = LOC_ROOT / "source" / "sfm_glomap" / "scripts"

# Prefer the current updated reloc bundle; only fall back to the legacy base bundle
# if v3 is absent. (Previously defaulted to the base bundle, silently overriding
# path_follow_flight.py's own current-v3 default via SFM_RELOC_BUNDLE.)
_V3_FLIGHT_BUNDLE = LOC_ROOT / "bundles" / "current_reloc_map_updated_v3.pt"
DEFAULT_FLIGHT_BUNDLE = (
    _V3_FLIGHT_BUNDLE if _V3_FLIGHT_BUNDLE.exists()
    else LOC_ROOT / "bundles" / "base_reloc_map_xfeat_tri.pt"
)
DEFAULT_MEGALOC_CACHE = DEPLOY / "megaloc_ref_desc_glomap_fused_322.npy"
DEFAULT_PATH_JSON = MISSION_ROOT / "outputs" / "current_safezone" / "flight_path.json"
DEFAULT_POLES_JSON = MISSION_ROOT / "outputs" / "current_safezone" / "poles.json"
DEFAULT_SAFEZONE_DIR = MISSION_ROOT / "outputs" / "current_safezone"
DEFAULT_SAFETY_FILE = Path(os.environ.get("SFM_SAFETY_FILE", "/tmp/sfm_drone_safety.cmd"))


DRAW_SCRIPTS = {
    "draw-path": AUTHOR / "draw_path.py",
    "draw-poles": AUTHOR / "draw_poles.py",
    "draw-wires": AUTHOR / "draw_wires.py",
    "label-route": AUTHOR / "label_route_points.py",
    "edit-height": AUTHOR / "edit_path_height.py",
    "sync-path": AUTHOR / "sync_path.py",
    "sync-poles": AUTHOR / "sync_poles.py",
}


def env_with_mission(args) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(FLIGHT) + os.pathsep +
        str(DEPLOY) + os.pathsep +
        str(SCRIPTS) + os.pathsep +
        env.get("PYTHONPATH", "")
    )
    env["SFM_RELOC_BUNDLE"] = args.bundle
    env["SFM_MEGALOC_CACHE"] = args.megaloc_cache
    env["SFM_FLIGHT_PATH_JSON"] = args.path_json
    env["SFM_POLES_JSON"] = args.poles_json
    env["SFM_SAFEZONE_DIR"] = str(Path(args.path_json).resolve().parent)
    env["SFM_MAP_ROOT"] = str((LOC_ROOT / "source" / "sfm_glomap").resolve())
    env["SFM_OLYMPE_CONTROLLER"] = args.controller
    env["SFM_SAFETY_FILE"] = str(args.safety_file)
    return env


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[mission_pipeline] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


# env override > original machine's known-good python (if present) > current python.
_LEGACY_PY = "/usr/bin/python3.12"
DEFAULT_PYTHON = os.environ.get("SFM_LOCALIZER_PYTHON") or (_LEGACY_PY if Path(_LEGACY_PY).exists() else sys.executable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified mission authoring / flight pipeline.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--mode",
        choices=[
            "launch-blender",
            "draw-path",
            "draw-poles",
            "draw-wires",
            "label-route",
            "edit-height",
            "sync-path",
            "sync-poles",
            "plan-path",
            "flight-selftest",
            "dry-run",
            "grab-only",
            "fly",
            "safety-auto",
            "safety-hover",
            "safety-manual",
            "safety-land",
        ],
        required=True,
    )
    parser.add_argument("--bundle", default=str(DEFAULT_FLIGHT_BUNDLE))
    parser.add_argument("--megaloc-cache", default=str(DEFAULT_MEGALOC_CACHE))
    parser.add_argument("--path-json", default=str(DEFAULT_PATH_JSON))
    parser.add_argument("--poles-json", default=str(DEFAULT_POLES_JSON))
    parser.add_argument("--safezone-dir", default=None)
    parser.add_argument("--ip", default="192.168.42.1")
    parser.add_argument("--controller", default=os.environ.get("SFM_OLYMPE_CONTROLLER", "auto"))
    parser.add_argument("--safety-file", default=str(DEFAULT_SAFETY_FILE))
    parser.add_argument("--secs", type=float, default=20.0)
    parser.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=1)
    args, passthrough = parser.parse_known_args()

    if args.safezone_dir:
        safezone = Path(args.safezone_dir)
        args.path_json = str(safezone / "flight_path.json")
        args.poles_json = str(safezone / "poles.json")

    env = env_with_mission(args)

    if args.mode.startswith("safety-"):
        cmd = args.mode.removeprefix("safety-")
        safety_path = Path(args.safety_file)
        safety_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: the flight-side poller must never read a half-written command.
        tmp = safety_path.with_suffix(safety_path.suffix + ".tmp")
        tmp.write_text(cmd + "\n")
        os.replace(tmp, safety_path)
        print(f"[mission_pipeline] safety command -> {cmd} ({args.safety_file})", flush=True)
        return

    if args.mode == "launch-blender":
        run([args.python, str(AUTHOR / "blender_mcp_launch.py"), *passthrough], env)
        return

    if args.mode in DRAW_SCRIPTS:
        run([args.python, str(AUTHOR / "blender_send.py"), "codefile", str(DRAW_SCRIPTS[args.mode]), *passthrough], env)
        return

    if args.mode == "plan-path":
        run([args.python, str(FLIGHT / "plan_path.py"), *passthrough], env)
        return

    flight_script = FLIGHT / "path_follow_flight.py"
    if args.mode == "flight-selftest":
        run([args.python, str(flight_script), "--selftest", *passthrough], env)
    elif args.mode == "dry-run":
        run([args.python, str(flight_script), "--dry-run", "--yaw-sign", str(args.yaw_sign), *passthrough], env)
    elif args.mode == "grab-only":
        run([
            args.python, str(flight_script), "--grab-only",
            "--ip", args.ip,
            "--controller", args.controller,
            "--secs", str(args.secs),
            *passthrough,
        ], env)
    elif args.mode == "fly":
        run([
            args.python, str(flight_script), "--fly",
            "--ip", args.ip,
            "--controller", args.controller,
            "--safety-file", args.safety_file,
            "--yaw-sign", str(args.yaw_sign),
            *passthrough,
        ], env)


if __name__ == "__main__":
    main()
