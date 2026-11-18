"""Compose verified UI route selection with independent mission constituents."""
from __future__ import annotations
import sys
from pathlib import Path

_CONTROL_ROOT = Path(__file__).resolve().parents[1]
if str(_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTROL_ROOT))

from mission_resolver import MissionReadiness  # noqa: E402
from site_profile import flight_readiness_errors  # noqa: E402


def selected_route_mission_readiness(evidence, profile, snapshot) -> MissionReadiness:
    """Replace only route evidence; unknown independent readiness stays fail closed.

    The caller supplies a freshly verified profile_for_selected_route overlay.
    This function binds that receipt to the immutable runtime snapshot as well.
    """
    readiness = MissionReadiness(
        localization_errors=tuple(getattr(evidence, "mission_localization_errors", ()) or ()),
        flight_errors=tuple(getattr(evidence, "mission_flight_errors", ()) or ()),
        route_errors=getattr(evidence, "mission_route_errors", None),
        non_route_flight_errors=getattr(evidence, "mission_non_route_flight_errors", None),
    )
    errors = []
    if profile is None:
        errors.append("selected route requires a validated site profile")
    else:
        errors.extend(flight_readiness_errors(profile))
    if snapshot is None:
        errors.append("no verified operator route selected")
    elif profile is not None:
        try:
            snapshot.verify_file_unchanged()
            if snapshot.sha256 != profile.asset_sha256.route_json:
                errors.append("selected route snapshot SHA-256 does not match verified route approval")
            if snapshot.site_id != profile.site_id:
                errors.append("selected route snapshot site does not match the active map")
            frame_id = None if profile.flight is None else profile.flight.coordinate_frame_id
            if snapshot.coordinate_frame_id != frame_id:
                errors.append("selected route snapshot frame does not match the active map")
            if snapshot.path.resolve() != profile.route_json.resolve():
                errors.append("selected route snapshot path does not match verified route approval")
        except (AttributeError, OSError, ValueError) as exc:
            errors.append(f"selected route snapshot verification failed: {exc}")
    selected = readiness.with_selected_route_errors(tuple(errors))
    if not bool(getattr(evidence, "mission_evaluation_only", False)):
        return selected
    return MissionReadiness(
        localization_errors=selected.localization_errors,
        flight_errors=(*selected.flight_errors, "evaluation-only mission cannot authorize AUTO"),
        route_errors=selected.route_errors,
        non_route_flight_errors=selected.non_route_flight_errors,
    )


def _verify_route_matches_profile(profile, snapshot) -> None:
    profile_site_id = getattr(profile, "site_id", None)
    snapshot_site_id = getattr(snapshot, "site_id", None)
    if profile_site_id and snapshot_site_id and snapshot_site_id != profile_site_id:
        raise ValueError(
            f"route site_id mismatch: expected {profile_site_id!r}, got {snapshot_site_id!r}"
        )

    flight_obj = getattr(profile, "flight", None)
    coord_frame = getattr(profile, "coordinate_frame", None)
    expected_frame_id = (
        getattr(flight_obj, "coordinate_frame_id", None)
        if flight_obj and getattr(flight_obj, "coordinate_frame_id", None)
        else (getattr(coord_frame, "id", None) if coord_frame else None)
    )
    snapshot_frame_id = getattr(snapshot, "coordinate_frame_id", None)
    if expected_frame_id and snapshot_frame_id and snapshot_frame_id != expected_frame_id:
        raise ValueError(
            f"route coordinate_frame_id mismatch: expected {expected_frame_id!r}, got {snapshot_frame_id!r}"
        )

    asset_sha256 = getattr(profile, "asset_sha256", None)
    route_json = getattr(profile, "route_json", None)
    snapshot_path = getattr(snapshot, "path", None)
    snapshot_sha256 = getattr(snapshot, "sha256", None)
    if (
        asset_sha256 is not None
        and route_json is not None
        and getattr(asset_sha256, "route_json", None) is not None
        and snapshot_path is not None
        and Path(route_json).resolve() == Path(snapshot_path).resolve()
        and snapshot_sha256 != asset_sha256.route_json
    ):
        raise ValueError(
            f"route_json SHA-256 mismatch: expected {asset_sha256.route_json}, got {snapshot_sha256}"
        )


def _verify_profile_asset_digests(app, profile, asset_sha256, site_profile_path) -> None:
    is_snapshot_profile = app._is_mission_snapshot_profile(site_profile_path)
    for label in (
        "localization_bundle", "map_reference_poses", "map_align", "reference_index", "track_landmarks",
    ):
        path = getattr(profile, label, None)
        expected = getattr(asset_sha256, label, None)
        if expected is None:
            continue
        if path is None:
            raise ValueError(f"missing asset for {label} (expected {expected})")
        resolved_path = Path(path).resolve()
        if not resolved_path.is_file():
            raise ValueError(f"asset file does not exist for {label}: {path}")
        expected_lower = str(expected).strip().lower()
        if is_snapshot_profile or (resolved_path, expected_lower) in app._SESSION_VERIFIED_ASSETS:
            app._register_session_verified_asset(resolved_path, expected_lower)
            continue
        actual = app.file_sha256(resolved_path)
        if actual != expected_lower:
            raise ValueError(f"{label} ({path}) has sha256 {actual}, expected {expected_lower}")
        app._register_session_verified_asset(resolved_path, expected_lower)


def _verify_flight_approval(app, profile) -> None:
    errors = app.flight_readiness_errors(profile)
    if errors:
        note = "" if getattr(profile, "flight", None) is None else getattr(profile.flight, "approval_note", "")
        suffix = f"; note: {note}" if note else ""
        raise ValueError(
            "site is not approved for autonomous flight: " + "; ".join(errors) + suffix
        )
    if getattr(profile, "flight", None) is None:
        raise ValueError("site profile has no flight section")
    camera = getattr(profile, "query_camera", None)
    if camera is not None:
        deploy_path = str(app._WS.deploy_code)
        if deploy_path not in sys.path:
            sys.path.append(deploy_path)
        from production_localizer_factory import validate_camera_tuple

        validate_camera_tuple((camera.model, camera.width, camera.height, list(camera.params)))


def validate_auto_flight_assets(operator, profile, snapshot) -> None:
    """Re-verify the approved route and assets before AUTO; lookups go through the app module."""
    import flight_operator_app as app

    if callable(getattr(snapshot, "verify_file_unchanged", None)):
        snapshot.verify_file_unchanged()
    if getattr(profile, "source", None) is not None:
        profile = app.profile_for_selected_route(profile.source, snapshot.path)

    current_validator = vars(app).get("validate_profile_flight_assets")
    if (
        current_validator is not None
        and current_validator is not app._ORIGINAL_VALIDATE_PROFILE_FLIGHT_ASSETS
    ):
        current_validator(profile)
        return

    _verify_flight_approval(app, profile)
    _verify_route_matches_profile(profile, snapshot)
    asset_sha256 = getattr(profile, "asset_sha256", None)
    if asset_sha256 is not None:
        _verify_profile_asset_digests(app, profile, asset_sha256, operator.site_profile_path)
