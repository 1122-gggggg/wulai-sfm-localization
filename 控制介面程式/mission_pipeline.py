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
import hashlib
import hmac
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from site_profile import (
    SiteProfile,
    flight_readiness_errors,
    load_site_profile,
)
from workspace_layout import workspace_from_file


_WS = workspace_from_file(__file__)
SYSTEM_ROOT = _WS.root
MISSION_ROOT = _WS.control
LOC_ROOT = _WS.algorithms
AUTHOR = _WS.control / "authoring"
FLIGHT = _WS.flight_control
DEPLOY = _WS.deploy_code
SCRIPTS = _WS.algorithms / "source" / "sfm_glomap" / "scripts"
if str(FLIGHT) not in sys.path:
    sys.path.insert(0, str(FLIGHT))

from safety_command import safety_file_from_environment, write_safety_command  # noqa: E402

DEFAULT_FLIGHT_BUNDLE = _WS.bundles / "your_site_reloc_map_edm.pt"
DEFAULT_MEGALOC_CACHE = ""
DEFAULT_MAP_PLY = _WS.map_ply_dir / "your_site.ply"
DEFAULT_PATH_JSON = _WS.mission_routes / "your_site" / "flight_path.json"
DEFAULT_POLES_JSON = _WS.mission_routes / "your_site" / "poles.json"
DEFAULT_SAFETY_FILE = safety_file_from_environment()


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
    "fly",
}
_POLES_MUST_EXIST_MODES = {
    "draw-wires",
    "sync-poles",
    "plan-path",
}
_SITE_ASSET_ENV_VARS = (
    "SFM_MAP_PLY",
    "SFM_MAP_ALIGN",
    "SFM_FLIGHT_PATH_JSON",
    "SFM_POLES_JSON",
    "SFM_RELOC_BUNDLE",
    "SFM_MEGALOC_CACHE",
    "SFM_REFERENCE_INDEX",
    "SFM_REFERENCE_INDEX_SHA256",
    "SFM_TRACK_LANDMARKS",
    "SFM_LOCALIZER_BACKEND",
    "SFM_LOCALIZER_DEPLOY_DIR",
    "SFM_LOCALIZER_PROFILE",
    "SFM_QUERY_CAMERA_JSON",
    "SFM_BUNDLE_SHA256",
    "SFM_FLIGHT_CONTRACT_JSON",
)


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _profile_mission_path(profile: SiteProfile, filename: str) -> Path:
    site_packages = _WS.site_packages.resolve()
    for asset in (
        profile.route_json,
        profile.map_ply,
        profile.localization_bundle,
        getattr(profile, "map_reference_poses", None),
        getattr(profile, "map_align", None),
    ):
        if asset is None:
            continue
        try:
            relative = Path(asset).resolve().relative_to(site_packages)
        except ValueError:
            continue
        if relative.parts:
            return (site_packages / relative.parts[0] / "routes" / filename).resolve()
    return (site_packages / profile.site_id / "routes" / filename).resolve()


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


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash {label} {path}: {exc}") from exc
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
        )


def _validate_flight_route(profile: SiteProfile) -> None:
    assert profile.flight is not None
    assert profile.route_json is not None
    try:
        route = json.loads(
            profile.route_json.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise ValueError(f"cannot read flight route {profile.route_json}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid flight route JSON {profile.route_json}: {exc}") from exc
    expected = {
        "schema": "sfm-flight-route/v1",
        "site_id": profile.site_id,
        "coordinate_frame_id": profile.flight.coordinate_frame_id,
        "units": "map",
        "purpose": "flight",
    }
    if not isinstance(route, dict):
        raise ValueError("flight route root must be an object")
    for key, value in expected.items():
        if route.get(key) != value:
            raise ValueError(
                f"flight route {key} must be {value!r}, got {route.get(key)!r}"
            )
    if route.get("closed") is not False:
        raise ValueError(
            f"flight route closed must be False, got {route.get('closed')!r}"
        )
    if route.get("frame") not in {"aligned", "glomap"}:
        raise ValueError("flight route frame must be 'aligned' or 'glomap'")
    waypoints = route.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("flight route must contain at least two waypoints")
    parsed_waypoints = []
    for index, waypoint in enumerate(waypoints):
        if (
            not isinstance(waypoint, list)
            or len(waypoint) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in waypoint
            )
        ):
            raise ValueError(
                f"flight route waypoint[{index}] must be a finite numeric 3-vector"
            )
        parsed_waypoints.append(tuple(float(value) for value in waypoint))
    for index, (start, end) in enumerate(
        zip(parsed_waypoints[:-1], parsed_waypoints[1:])
    ):
        length_sq = sum((b - a) ** 2 for a, b in zip(start, end))
        if length_sq <= 1e-18:
            raise ValueError(f"flight route segment {index}->{index + 1} has zero length")


def validate_profile_flight_assets(profile: SiteProfile) -> None:
    """Validate an approved scale-free flight package before execution."""
    errors = flight_readiness_errors(profile)
    if errors:
        note = "" if profile.flight is None else profile.flight.approval_note
        suffix = f"; note: {note}" if note else ""
        raise ValueError("site is not approved for autonomous flight: " + "; ".join(errors) + suffix)
    assert profile.flight is not None
    assert profile.route_json is not None
    assert profile.map_reference_poses is not None
    assert profile.query_camera is not None
    deploy_path = str(DEPLOY)
    if deploy_path not in sys.path:
        sys.path.append(deploy_path)
    from production_localizer_factory import validate_camera_tuple

    camera = profile.query_camera
    validate_camera_tuple(
        (camera.model, camera.width, camera.height, list(camera.params))
    )
    _validate_flight_route(profile)
    checks = (
        (
            profile.localization_bundle,
            profile.asset_sha256.localization_bundle,
            "localization_bundle",
        ),
        (profile.route_json, profile.asset_sha256.route_json, "route_json"),
        (
            profile.map_reference_poses,
            profile.asset_sha256.map_reference_poses,
            "map_reference_poses",
        ),
        # The flight gate demands this digest; without verifying it the pin was
        # decorative and the alignment on disk could be swapped for another
        # internally-consistent one, silently rotating every commanded body axis.
        (profile.map_align, profile.asset_sha256.map_align, "map_align"),
    )
    if profile.localizer_profile is not None:
        checks += (
            (
                profile.localizer_profile,
                profile.asset_sha256.localizer_profile,
                "localizer_profile",
            ),
        )
    controller = profile.flight.controller
    if controller is not None and controller.inspect_waypoints:
        assert profile.poles_json is not None
        checks += (
            (
                profile.poles_json,
                profile.asset_sha256.poles_json,
                "poles_json",
            ),
        )
    for path, expected, label in checks:
        assert path is not None and expected is not None
        _verify_sha256(path, expected, label)


def shadow_readiness_errors(profile: SiteProfile) -> list[str]:
    """Return offline blockers for a map-unit shadow-algorithm session.

    This checks the inputs that can be verified before travelling to the site.
    It deliberately does *not* require flight approval or a controller contract:
    shadow mode computes and records recommendations only, never commands an
    aircraft.
    """
    errors: list[str] = []
    flight = profile.flight
    if flight is None or not flight.coordinate_frame_id:
        errors.append("missing flight.coordinate_frame_id for route/map matching")
    if profile.route_json is None:
        errors.append("missing route_json")
    if profile.map_reference_poses is None:
        errors.append("missing map_reference_poses")
    if profile.map_align is None:
        errors.append("missing map_align")
    if profile.query_camera is None:
        errors.append("missing query_camera")
    if profile.localizer != "edm" or profile.localizer_profile is None:
        errors.append("missing EDM localizer_profile")
    if errors:
        return errors

    try:
        _validate_flight_route(profile)
    except ValueError as exc:
        errors.append(str(exc))

    checks = (
        (profile.map_ply, profile.asset_sha256.map_ply, "map_ply"),
        (
            profile.localization_bundle,
            profile.asset_sha256.localization_bundle,
            "localization_bundle",
        ),
        (profile.route_json, profile.asset_sha256.route_json, "route_json"),
        (
            profile.map_reference_poses,
            profile.asset_sha256.map_reference_poses,
            "map_reference_poses",
        ),
        (profile.map_align, profile.asset_sha256.map_align, "map_align"),
        (
            profile.localizer_profile,
            profile.asset_sha256.localizer_profile,
            "localizer_profile",
        ),
    )
    for path, expected, label in checks:
        if path is None:
            errors.append(f"missing {label}")
        elif expected is None:
            errors.append(f"missing asset_sha256.{label}")
        else:
            try:
                _verify_sha256(path, expected, label)
            except ValueError as exc:
                errors.append(str(exc))
    return errors


def shadow_authorization_blockers(profile: SiteProfile) -> list[str]:
    """Report the remaining profile authorization blockers."""
    return list(flight_readiness_errors(profile))


def validate_profile_for_flight(profile: SiteProfile) -> None:
    """Require an approved, hash-pinned autonomous flight package."""
    validate_profile_flight_assets(profile)


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
            ("--reference-index", getattr(args, "reference_index", None)),
            (
                "--reference-index-sha256",
                getattr(args, "reference_index_sha256", None),
            ),
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

        if mode == "fly":
            try:
                validate_profile_for_flight(profile)
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
        args.reference_index = str(getattr(profile, "reference_index", None) or "")
        args.reference_index_sha256 = str(
            getattr(getattr(profile, "asset_sha256", None), "reference_index", None)
            or ""
        )
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
    args.reference_index = str(getattr(args, "reference_index", None) or "")
    args.reference_index_sha256 = str(
        getattr(args, "reference_index_sha256", None) or ""
    )
    args.track_landmarks = str(
        os.environ.get("SFM_TRACK_LANDMARKS", "")
        if args.track_landmarks is None
        else args.track_landmarks
    )
    args.path_json = str(DEFAULT_PATH_JSON if args.path_json is None else args.path_json)
    args.poles_json = str(DEFAULT_POLES_JSON if args.poles_json is None else args.poles_json)
    return None


def env_with_mission(args, profile: SiteProfile | None = None) -> dict[str, str]:
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
    env["SFM_REFERENCE_INDEX"] = args.reference_index
    env["SFM_REFERENCE_INDEX_SHA256"] = args.reference_index_sha256
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
    if profile is not None:
        env["SFM_LOCALIZER_BACKEND"] = profile.localizer
        env["SFM_MAP_ALIGN"] = str(profile.map_align or "")
        env["SFM_LOCALIZER_DEPLOY_DIR"] = str(profile.localizer_deploy_dir or DEPLOY)
        env["SFM_LOCALIZER_PROFILE"] = str(profile.localizer_profile or "")
        env["SFM_BUNDLE_SHA256"] = profile.asset_sha256.localization_bundle or ""
        if profile.query_camera is not None:
            env["SFM_QUERY_CAMERA_JSON"] = json.dumps(asdict(profile.query_camera))
        else:
            env.pop("SFM_QUERY_CAMERA_JSON", None)
        if profile.flight is not None:
            env["SFM_FLIGHT_CONTRACT_JSON"] = json.dumps({
                "schema_version": profile.schema_version,
                "approved": profile.flight.approved,
                "site_id": profile.site_id,
                "coordinate_frame_id": profile.flight.coordinate_frame_id,
                "route_clearance_approved": profile.flight.route_clearance_approved,
                "approval_note": profile.flight.approval_note,
                "controller": (
                    None if profile.flight.controller is None
                    else asdict(profile.flight.controller)
                ),
                "query_camera": (
                    None if profile.query_camera is None
                    else asdict(profile.query_camera)
                ),
                "localization_bundle_sha256": (
                    profile.asset_sha256.localization_bundle
                ),
                "route_sha256": profile.asset_sha256.route_json,
                "map_reference_poses": (
                    None if profile.map_reference_poses is None
                    else str(profile.map_reference_poses)
                ),
                "map_reference_poses_sha256": profile.asset_sha256.map_reference_poses,
                "localizer_profile_sha256": profile.asset_sha256.localizer_profile,
                "reference_index": (
                    None
                    if getattr(profile, "reference_index", None) is None
                    else str(profile.reference_index)
                ),
                "reference_index_sha256": getattr(
                    profile.asset_sha256, "reference_index", None
                ),
                "poles_sha256": profile.asset_sha256.poles_json,
            })
        else:
            env.pop("SFM_FLIGHT_CONTRACT_JSON", None)
    else:
        env.pop("SFM_MAP_ALIGN", None)
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
            "shadow-readiness",
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
    parser.add_argument("--reference-index", default=None)
    parser.add_argument("--reference-index-sha256", default=None)
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
        write_safety_command(args.safety_file, cmd)
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

    if args.mode == "shadow-readiness":
        assert site_profile is not None
        shadow_errors = shadow_readiness_errors(site_profile)
        authorization_blockers = shadow_authorization_blockers(site_profile)
        print(
            "[shadow-readiness] package="
            + ("READY" if not shadow_errors else "BLOCKED"),
            flush=True,
        )
        for error in shadow_errors:
            print(f"  package blocker: {error}", flush=True)
        print(
            "[shadow-readiness] autonomous execution="
            + ("APPROVED" if not authorization_blockers else "BLOCKED"),
            flush=True,
        )
        for blocker in authorization_blockers:
            print(f"  human/field blocker: {blocker}", flush=True)
        if shadow_errors:
            raise SystemExit(1)
        return

    env = env_with_mission(args, site_profile)

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
