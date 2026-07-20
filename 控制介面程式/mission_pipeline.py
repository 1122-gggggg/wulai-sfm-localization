#!/usr/bin/env python3
"""Mission authoring and ANAFI flight entrypoint.

This is the non-expert-facing wrapper for:
  - drawing inspection routes;
  - marking poles/wires;
  - planning/checking the route;
  - running dry-run, live grab-only, or real ANAFI flight.

Operational modes are site-profile first. Legacy per-asset arguments remain available
only behind --allow-legacy-assets so a missing or stale default cannot silently select
another site's map, bundle, camera, or route.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from site_profile import SiteProfile, load_site_profile
from workspace_layout import workspace_from_file


_WS = workspace_from_file(__file__)
SYSTEM_ROOT = _WS.root
MISSION_ROOT = _WS.control
LOC_ROOT = _WS.algorithms
AUTHOR = _WS.control / "authoring"
FLIGHT = _WS.flight_control
DEPLOY = _WS.deploy_code
SCRIPTS = _WS.algorithms / "source" / "sfm_glomap" / "scripts"

DEFAULT_FLIGHT_BUNDLE = _WS.bundles / "your_site_reloc_map_edm.pt"
DEFAULT_MEGALOC_CACHE = ""
DEFAULT_MAP_PLY = _WS.map_ply_dir / "your_site.ply"
DEFAULT_PATH_JSON = _WS.mission_routes / "your_site" / "flight_path.json"
DEFAULT_POLES_JSON = _WS.mission_routes / "your_site" / "poles.json"
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

_PROFILE_FREE_MODES = {"flight-selftest"}
_ROUTE_MUST_EXIST_MODES = {
    "draw-wires",
    "label-route",
    "edit-height",
    "sync-path",
    "plan-path",
    "dry-run",
    "grab-only",
    "fly",
}
_POLES_MUST_EXIST_MODES = {
    "draw-wires",
    "sync-poles",
    "plan-path",
    "dry-run",
    "grab-only",
    "fly",
}
_SITE_ASSET_ENV_VARS = (
    "SFM_MAP_PLY",
    "SFM_FLIGHT_PATH_JSON",
    "SFM_POLES_JSON",
    "SFM_RELOC_BUNDLE",
    "SFM_MEGALOC_CACHE",
    "SFM_TRACK_LANDMARKS",
)


def _profile_mission_path(profile: SiteProfile, filename: str) -> Path:
    return (_WS.mission_routes / profile.site_id / filename).resolve()


def _require_existing_asset(
    parser: argparse.ArgumentParser,
    *,
    mode: str,
    label: str,
    path: Path,
) -> None:
    if not path.is_file():
        parser.error(
            f"mission mode {mode!r} requires existing {label}: {path}; "
            "create it with the authoring modes first or fix the site profile"
        )


def resolve_mission_site_assets(
    args,
    parser: argparse.ArgumentParser,
    *,
    mode: str,
) -> SiteProfile | None:
    """Apply one site profile, or an explicitly opted-in legacy asset set."""
    profile_arg = str(getattr(args, "site_profile", "") or "").strip()
    cli_overrides = [
        flag
        for flag, value in (
            ("--map-ply", getattr(args, "map_ply", None)),
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

        route = profile.route_json or _profile_mission_path(profile, "flight_path.json")
        poles = profile.poles_json or _profile_mission_path(profile, "poles.json")
        route.parent.mkdir(parents=True, exist_ok=True)
        poles.parent.mkdir(parents=True, exist_ok=True)

        if mode in _ROUTE_MUST_EXIST_MODES:
            _require_existing_asset(parser, mode=mode, label="route_json", path=route)
        if mode in _POLES_MUST_EXIST_MODES:
            _require_existing_asset(parser, mode=mode, label="poles_json", path=poles)

        args.site_profile = str(profile.source)
        args.map_ply = str(profile.map_ply)
        args.bundle = str(profile.localization_bundle)
        args.megaloc_cache = str(profile.megaloc_cache or "")
        args.track_landmarks = str(profile.track_landmarks or "")
        args.path_json = str(route)
        args.poles_json = str(poles)
        return profile

    if mode not in _PROFILE_FREE_MODES and not args.allow_legacy_assets:
        parser.error(
            f"mission mode {mode!r} requires --site-profile; "
            "use --allow-legacy-assets only for a reviewed migration run"
        )

    args.site_profile = ""
    args.map_ply = str(DEFAULT_MAP_PLY if args.map_ply is None else args.map_ply)
    args.bundle = str(DEFAULT_FLIGHT_BUNDLE if args.bundle is None else args.bundle)
    args.megaloc_cache = str(
        DEFAULT_MEGALOC_CACHE if args.megaloc_cache is None else args.megaloc_cache
    )
    args.track_landmarks = str(
        os.environ.get("SFM_TRACK_LANDMARKS", "")
        if args.track_landmarks is None
        else args.track_landmarks
    )
    args.path_json = str(DEFAULT_PATH_JSON if args.path_json is None else args.path_json)
    args.poles_json = str(DEFAULT_POLES_JSON if args.poles_json is None else args.poles_json)
    return None


def env_with_mission(args) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(FLIGHT), str(DEPLOY)]
    if SCRIPTS.is_dir():
        python_paths.append(str(SCRIPTS))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["SFM_WORKSPACE_ROOT"] = str(SYSTEM_ROOT)
    env["SFM_RELOC_BUNDLE"] = args.bundle
    env["SFM_MEGALOC_CACHE"] = args.megaloc_cache
    env["SFM_TRACK_LANDMARKS"] = args.track_landmarks
    env["SFM_MAP_PLY"] = args.map_ply
    env["SFM_FLIGHT_PATH_JSON"] = args.path_json
    env["SFM_POLES_JSON"] = args.poles_json
    env["SFM_SAFEZONE_DIR"] = str(Path(args.path_json).resolve().parent)
    env["SFM_MAP_ROOT"] = str((_WS.algorithms / "source" / "sfm_glomap").resolve())
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


DEFAULT_PYTHON = os.environ.get("SFM_LOCALIZER_PYTHON") or sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified mission authoring / flight pipeline.")
    parser.add_argument(
        "--site-profile",
        default=os.environ.get("SFM_SITE_PROFILE", ""),
        help="JSON profile that atomically selects this site's map and mission assets",
    )
    parser.add_argument(
        "--allow-legacy-assets",
        action="store_true",
        help=(
            "migration escape hatch: allow per-asset/default paths without a site profile; "
            "never use for an unreviewed real-flight run"
        ),
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
    parser.add_argument("--map-ply", default=None)
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
        tmp = safety_path.with_suffix(safety_path.suffix + ".tmp")
        tmp.write_text(cmd + "\n")
        os.replace(tmp, safety_path)
        print(f"[mission_pipeline] safety command -> {cmd} ({args.safety_file})", flush=True)
        return

    site_profile = resolve_mission_site_assets(args, parser, mode=args.mode)

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
    elif args.allow_legacy_assets:
        print(
            "[mission_pipeline] WARNING: legacy per-asset mode enabled; "
            "verify every resolved path before continuing",
            file=sys.stderr,
            flush=True,
        )

    env = env_with_mission(args)

    if args.mode == "launch-blender":
        run([args.python, str(AUTHOR / "blender_mcp_launch.py"), *passthrough], env)
        return

    if args.mode in DRAW_SCRIPTS:
        run(
            [
                args.python,
                str(AUTHOR / "blender_send.py"),
                "codefile",
                str(DRAW_SCRIPTS[args.mode]),
                *passthrough,
            ],
            env,
        )
        return

    if args.mode == "plan-path":
        run([args.python, str(FLIGHT / "plan_path.py"), *passthrough], env)
        return

    flight_script = FLIGHT / "path_follow_flight.py"
    if args.mode == "flight-selftest":
        run([args.python, str(flight_script), "--selftest", *passthrough], env)
    elif args.mode == "dry-run":
        run(
            [args.python, str(flight_script), "--dry-run", "--yaw-sign", str(args.yaw_sign), *passthrough],
            env,
        )
    elif args.mode == "grab-only":
        run(
            [
                args.python,
                str(flight_script),
                "--grab-only",
                "--ip",
                args.ip,
                "--controller",
                args.controller,
                "--secs",
                str(args.secs),
                *passthrough,
            ],
            env,
        )
    elif args.mode == "fly":
        run(
            [
                args.python,
                str(flight_script),
                "--fly",
                "--ip",
                args.ip,
                "--controller",
                args.controller,
                "--safety-file",
                args.safety_file,
                "--yaw-sign",
                str(args.yaw_sign),
                *passthrough,
            ],
            env,
        )


if __name__ == "__main__":
    main()
