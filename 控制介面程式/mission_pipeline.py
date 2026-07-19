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

from site_profile import SiteProfile, load_site_profile


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
DEFAULT_MAP_PLY = LOC_ROOT / "maps" / "current_realrgb_v3.ply"
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

_SITE_ASSET_ENV_VARS = (
    "SFM_MAP_PLY",
    "SFM_FLIGHT_PATH_JSON",
    "SFM_POLES_JSON",
    "SFM_RELOC_BUNDLE",
    "SFM_MEGALOC_CACHE",
    "SFM_TRACK_LANDMARKS",
)


def resolve_mission_site_assets(args, parser: argparse.ArgumentParser) -> SiteProfile | None:
    """Apply one site profile, or retain the legacy per-asset arguments."""
    profile_arg = str(getattr(args, "site_profile", "") or "").strip()
    cli_overrides = [
        flag
        for flag, value in (
            ("--bundle", getattr(args, "bundle", None)),
            ("--megaloc-cache", getattr(args, "megaloc_cache", None)),
            ("--track-landmarks", getattr(args, "track_landmarks", None)),
            ("--path-json", getattr(args, "path_json", None)),
            ("--poles-json", getattr(args, "poles_json", None)),
            ("--safezone-dir", getattr(args, "safezone_dir", None)),
        )
        if value is not None
    ]
    env_overrides = [name for name in _SITE_ASSET_ENV_VARS if os.environ.get(name)]
    if profile_arg:
        conflicts = cli_overrides + env_overrides
        if conflicts:
            parser.error(
                "--site-profile atomically selects mission assets; "
                f"remove these per-asset overrides: {', '.join(conflicts)}"
            )
        try:
            profile = load_site_profile(profile_arg)
        except ValueError as exc:
            parser.error(str(exc))
        if profile.poles_json is None:
            parser.error("mission_pipeline site profile requires assets.poles_json")
        args.site_profile = str(profile.source)
        args.map_ply = str(profile.map_ply)
        args.bundle = str(profile.localization_bundle)
        args.megaloc_cache = str(profile.megaloc_cache or "")
        args.track_landmarks = str(profile.track_landmarks or "")
        args.path_json = str(profile.route_json)
        args.poles_json = str(profile.poles_json)
        return profile

    args.site_profile = ""
    args.map_ply = str(DEFAULT_MAP_PLY)
    args.bundle = str(DEFAULT_FLIGHT_BUNDLE if args.bundle is None else args.bundle)
    args.megaloc_cache = str(
        DEFAULT_MEGALOC_CACHE if args.megaloc_cache is None else args.megaloc_cache
    )
    args.track_landmarks = str(
        os.environ.get("SFM_TRACK_LANDMARKS", "")
        if args.track_landmarks is None else args.track_landmarks
    )
    args.path_json = str(DEFAULT_PATH_JSON if args.path_json is None else args.path_json)
    args.poles_json = str(DEFAULT_POLES_JSON if args.poles_json is None else args.poles_json)
    return None


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
    env["SFM_TRACK_LANDMARKS"] = args.track_landmarks
    env["SFM_MAP_PLY"] = args.map_ply
    env["SFM_FLIGHT_PATH_JSON"] = args.path_json
    env["SFM_POLES_JSON"] = args.poles_json
    env["SFM_SAFEZONE_DIR"] = str(Path(args.path_json).resolve().parent)
    env["SFM_MAP_ROOT"] = str((LOC_ROOT / "source" / "sfm_glomap").resolve())
    env["SFM_OLYMPE_CONTROLLER"] = args.controller
    env["SFM_SAFETY_FILE"] = str(args.safety_file)
    if args.site_profile:
        env["SFM_SITE_PROFILE"] = args.site_profile
    else:
        env.pop("SFM_SITE_PROFILE", None)
    return env


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[mission_pipeline] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


# Keep every mission subcommand in the validated environment that launched this
# wrapper unless the operator explicitly selects another localizer runtime.
DEFAULT_PYTHON = os.environ.get("SFM_LOCALIZER_PYTHON") or sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified mission authoring / flight pipeline.")
    parser.add_argument(
        "--site-profile",
        default=os.environ.get("SFM_SITE_PROFILE", ""),
        help="JSON profile that atomically selects this site's map and mission assets",
    )
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
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--megaloc-cache", default=None)
    parser.add_argument("--track-landmarks", default=None)
    parser.add_argument("--path-json", default=None)
    parser.add_argument("--poles-json", default=None)
    parser.add_argument("--safezone-dir", default=None)
    parser.add_argument("--ip", default="192.168.42.1")
    parser.add_argument("--controller", default=os.environ.get("SFM_OLYMPE_CONTROLLER", "auto"))
    parser.add_argument("--safety-file", default=str(DEFAULT_SAFETY_FILE))
    parser.add_argument("--secs", type=float, default=20.0)
    parser.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=1)
    args, passthrough = parser.parse_known_args()

    # Emergency safety commands must remain available even when map/profile
    # assets are missing or misconfigured.
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

    site_profile = resolve_mission_site_assets(args, parser)

    if args.safezone_dir:
        safezone = Path(args.safezone_dir)
        args.path_json = str(safezone / "flight_path.json")
        args.poles_json = str(safezone / "poles.json")

    if site_profile is not None:
        print(
            f"[mission_pipeline] site={site_profile.site_id!r} "
            f"name={site_profile.display_name!r} profile={site_profile.source}",
            flush=True,
        )

    env = env_with_mission(args)

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
