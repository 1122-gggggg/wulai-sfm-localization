#!/usr/bin/env python3
"""Load one atomic set of site-specific localization and mission assets."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, SCHEMA_VERSION)


PRODUCTION_LOCALIZER_BACKEND = "edm"
SHA256_KEYS = (
    "map_ply",
    "localization_bundle",
    "route_json",
    "map_reference_poses",
    "localizer_profile",
    "poles_json",
    "map_align",
)
SITE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
#: display_name stays free-form text -- CJK, spaces, dashes and brackets are all
#: legitimate -- but it comes from a site package authored on another machine and
#: is interpolated into launcher shell commands and window titles. Control
#: characters and the shell quoting/expansion characters have no business in a
#: field whose only job is to be read by a human.
DISPLAY_NAME_REJECT_RE = re.compile("[\x00-\x1f\x7f-\x9f$`\"'\\\\]")


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


@dataclass(frozen=True)
class QueryCamera:
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class CoordinateFrame:
    """Coordinate semantics that a PLY file cannot encode by itself."""

    id: str
    convention: str
    horizontal_axes: tuple[str, str]
    up_axis: str
    handedness: str
    units: str


@dataclass(frozen=True)
class AssetDigests:
    map_ply: str | None = None
    localization_bundle: str | None = None
    route_json: str | None = None
    map_reference_poses: str | None = None
    localizer_profile: str | None = None
    poles_json: str | None = None
    map_align: str | None = None


@dataclass(frozen=True)
class FlightControlProfile:
    model: str
    speed_limit_mps: float
    pose_max_age_ms: float
    speed_max_age_ms: float
    command_ttl_ms: float
    yaw_tolerance_deg: float
    horizontal_axes: tuple[int, int]
    vertical_axis: int
    camera_to_body_yaw_deg: float
    body_right_sign: int
    lookahead_map_units: float
    rejoin_tolerance_map_units: float
    arrival_tolerance_map_units: float
    inspect_radius_map_units: float
    inspect_resume_margin_map_units: float
    max_pose_jump_map_units: float
    max_route_deviation_map_units: float
    progress_jump_slack_map_units: float
    max_progress_regression_map_units: float
    segment_window: int
    progress_speed_factor: float
    inspect_waypoints: tuple[int, ...]


@dataclass(frozen=True)
class FlightReadiness:
    approved: bool
    coordinate_frame_id: str | None
    route_clearance_approved: bool
    approval_note: str
    controller: FlightControlProfile | None


@dataclass(frozen=True)
class HardwareApprovalReference:
    receipt: Path
    sha256: str


@dataclass(frozen=True)
class HardwareApprovalReceipt:
    source: Path
    sha256: str
    approved: bool
    aircraft_product: str
    controller_product: str
    aircraft_firmware_versions: tuple[str, ...]
    controller_firmware_versions: tuple[str, ...]
    olympe_versions: tuple[str, ...]
    approval_note: str


@dataclass(frozen=True)
class SiteProfile:
    source: Path
    schema_version: int
    site_id: str
    display_name: str
    map_ply: Path
    route_json: Path | None
    localization_bundle: Path
    # Production site profiles always use the EDM tracker adapter.
    localizer: str = PRODUCTION_LOCALIZER_BACKEND
    localizer_deploy_dir: Path | None = None
    localizer_profile: Path | None = None
    map_reference_poses: Path | None = None
    #: Measured gravity alignment (T_align_gravity.json). Without it the runtime
    #: falls back to assuming GLOMAP -Y is up, which is wrong on every site
    #: measured so far (1.99 to 22.51 degrees).
    map_align: Path | None = None
    megaloc_cache: Path | None = None
    track_landmarks: Path | None = None
    poles_json: Path | None = None
    query_camera: QueryCamera | None = None
    coordinate_frame: CoordinateFrame | None = None
    asset_sha256: AssetDigests = AssetDigests()
    flight: FlightReadiness | None = None
    hardware_approval: HardwareApprovalReference | None = None


def _resolve_asset(profile_dir: Path, value, key: str, *, required: bool) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"site profile asset {key!r} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"site profile asset {key!r} must be a path string or null")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = profile_dir / path
    return path.resolve()


def _load_query_camera(raw, source: Path) -> QueryCamera | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"site profile query_camera must be an object: {source}")
    camera_keys = {"model", "width", "height", "params"}
    if set(raw) != camera_keys:
        raise ValueError(
            "site profile query_camera must contain exactly "
            f"{sorted(camera_keys)}: {source}"
        )
    model = raw.get("model")
    width = raw.get("width")
    height = raw.get("height")
    params = raw.get("params")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"site profile query_camera.model must be non-empty: {source}")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"site profile query_camera.width must be > 0: {source}")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"site profile query_camera.height must be > 0: {source}")
    if not isinstance(params, list) or not params:
        raise ValueError(f"site profile query_camera.params must be a non-empty list: {source}")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in params
    ):
        raise ValueError(
            f"site profile query_camera.params must contain numbers: {source}"
        )
    parsed_params = tuple(float(value) for value in params)
    if not all(math.isfinite(value) for value in parsed_params):
        raise ValueError(f"site profile query_camera.params must be finite: {source}")
    return QueryCamera(
        model=model.strip().upper(),
        width=width,
        height=height,
        params=parsed_params,
    )


def _load_coordinate_frame(raw, source: Path) -> CoordinateFrame | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"site profile coordinate_frame must be an object: {source}")
    keys = {
        "id",
        "convention",
        "horizontal_axes",
        "up_axis",
        "handedness",
        "units",
    }
    if set(raw) != keys:
        raise ValueError(
            "site profile coordinate_frame fields mismatch: "
            f"unknown={sorted(set(raw) - keys)} missing={sorted(keys - set(raw))}: {source}"
        )
    frame_id = raw.get("id")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError(f"site profile coordinate_frame.id must be non-empty: {source}")
    expected = {
        "convention": "glomap",
        "horizontal_axes": ["x", "z"],
        "up_axis": "-y",
        "handedness": "right",
        "units": "map",
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "site profile coordinate_frame must declare raw GLOMAP coordinates "
            "(X/Z horizontal, -Y up, right-handed, map units): "
            f"{source}"
        )
    return CoordinateFrame(
        id=frame_id.strip(),
        convention="glomap",
        horizontal_axes=("x", "z"),
        up_axis="-y",
        handedness="right",
        units="map",
    )


def _sha256(value, key: str, source: Path) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"site profile asset_sha256.{key} must be a string: {source}")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(
            f"site profile asset_sha256.{key} must be a 64-character SHA-256: {source}"
        )
    return digest


def _load_asset_digests(raw, source: Path) -> AssetDigests:
    if raw is None:
        return AssetDigests()
    if not isinstance(raw, dict):
        raise ValueError(f"site profile asset_sha256 must be an object: {source}")
    unknown = sorted(set(raw) - set(SHA256_KEYS))
    if unknown:
        raise ValueError(f"unknown site profile asset_sha256 keys {unknown}: {source}")
    return AssetDigests(
        **{key: _sha256(raw.get(key), key, source) for key in SHA256_KEYS}
    )


def _finite_number(raw: dict, key: str, source: Path, *, positive: bool) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"site profile flight.controller.{key} must be numeric: {source}")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        constraint = "finite and > 0" if positive else "finite"
        raise ValueError(
            f"site profile flight.controller.{key} must be {constraint}: {source}"
        )
    return parsed


def _load_flight_controller(raw, source: Path) -> FlightControlProfile | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"site profile flight.controller must be an object: {source}")
    positive_keys = (
        "speed_limit_mps",
        "pose_max_age_ms",
        "speed_max_age_ms",
        "command_ttl_ms",
        "yaw_tolerance_deg",
        "lookahead_map_units",
        "rejoin_tolerance_map_units",
        "arrival_tolerance_map_units",
        "inspect_radius_map_units",
        "inspect_resume_margin_map_units",
        "max_pose_jump_map_units",
        "max_route_deviation_map_units",
        "progress_jump_slack_map_units",
        "max_progress_regression_map_units",
        "progress_speed_factor",
    )
    allowed = set(positive_keys) | {
        "model",
        "horizontal_axes",
        "vertical_axis",
        "camera_to_body_yaw_deg",
        "body_right_sign",
        "segment_window",
        "inspect_waypoints",
    }
    unknown = sorted(set(raw) - allowed)
    missing = sorted(allowed - set(raw))
    if unknown or missing:
        raise ValueError(
            f"site profile flight.controller has unknown={unknown} missing={missing}: {source}"
        )
    model = raw["model"]
    if model != "scale_free_direction_speed_guard_v1":
        raise ValueError(
            "site profile flight.controller.model must be "
            f"'scale_free_direction_speed_guard_v1': {source}"
        )
    horizontal_axes = raw["horizontal_axes"]
    vertical_axis = raw["vertical_axis"]
    if (
        not isinstance(horizontal_axes, list)
        or len(horizontal_axes) != 2
        or any(type(value) is not int for value in horizontal_axes)
        or type(vertical_axis) is not int
        or sorted([*horizontal_axes, vertical_axis]) != [0, 1, 2]
    ):
        raise ValueError(
            "site profile flight.controller axes must cover 0, 1, and 2 exactly: "
            f"{source}"
        )
    body_right_sign = raw["body_right_sign"]
    if type(body_right_sign) not in {int, float} or float(body_right_sign) not in {-1.0, 1.0}:
        raise ValueError(
            "site profile flight.controller.body_right_sign must be -1 or 1: "
            f"{source}"
        )
    segment_window = raw["segment_window"]
    if (isinstance(segment_window, bool) or not isinstance(segment_window, int)
            or not 0 <= segment_window <= 20):
        raise ValueError(
            f"site profile flight.controller.segment_window must be an integer in [0,20]: "
            f"{source}"
        )
    inspect_waypoints = raw["inspect_waypoints"]
    if (not isinstance(inspect_waypoints, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in inspect_waypoints)
            or len(inspect_waypoints) != len(set(inspect_waypoints))):
        raise ValueError(
            "site profile flight.controller.inspect_waypoints must be a unique list "
            f"of positive 1-based integers: {source}"
        )
    values = {
        key: _finite_number(raw, key, source, positive=True)
        for key in positive_keys
    }
    # These are safety-envelope ceilings, not tuning defaults.  Site profiles are
    # hand-authored deployment inputs; accepting an extra zero here can otherwise
    # turn a 0.30 m/s or 1.5-map-unit guard into a effectively disabled one.
    ceilings = {
        "speed_limit_mps": 2.0,
        "lookahead_map_units": 5.0,
        "max_pose_jump_map_units": 5.0,
        "max_route_deviation_map_units": 10.0,
    }
    for key, maximum in ceilings.items():
        if values[key] > maximum:
            raise ValueError(
                f"site profile flight.controller.{key} must be <= {maximum:g}: {source}"
            )
    if values["pose_max_age_ms"] > 500.0:
        raise ValueError(
            f"site profile flight.controller.pose_max_age_ms must be <= 500: {source}"
        )
    if values["speed_max_age_ms"] > 500.0:
        raise ValueError(
            f"site profile flight.controller.speed_max_age_ms must be <= 500: {source}"
        )
    if values["command_ttl_ms"] > 250.0:
        raise ValueError(
            f"site profile flight.controller.command_ttl_ms must be <= 250: {source}"
        )
    return FlightControlProfile(
        model=model,
        horizontal_axes=tuple(horizontal_axes),
        vertical_axis=vertical_axis,
        camera_to_body_yaw_deg=_finite_number(
            raw, "camera_to_body_yaw_deg", source, positive=False
        ),
        body_right_sign=int(body_right_sign),
        segment_window=segment_window,
        inspect_waypoints=tuple(inspect_waypoints),
        **values,
    )


def _load_flight_readiness(
    raw, source: Path, *, schema_version: int
) -> FlightReadiness | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"site profile flight must be an object: {source}")
    allowed = {
        "approved",
        "coordinate_frame_id",
        "route_clearance_approved",
        "approval_note",
        "controller",
    }
    if schema_version == 1:
        allowed.add("map_units_per_meter")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown site profile flight keys {unknown}: {source}")
    approved = raw.get("approved", False)
    clearance = raw.get("route_clearance_approved", False)
    if not isinstance(approved, bool) or not isinstance(clearance, bool):
        raise ValueError(
            f"site profile flight approval fields must be boolean: {source}"
        )
    frame_id = raw.get("coordinate_frame_id")
    if frame_id is not None and (
        not isinstance(frame_id, str) or not frame_id.strip()
    ):
        raise ValueError(
            f"site profile flight.coordinate_frame_id must be a non-empty string or null: "
            f"{source}"
        )
    legacy_scale = raw.get("map_units_per_meter")
    if schema_version == 1 and legacy_scale is not None:
        if isinstance(legacy_scale, bool) or not isinstance(legacy_scale, (int, float)):
            raise ValueError(
                f"site profile flight.map_units_per_meter must be numeric or null: {source}"
            )
        legacy_scale = float(legacy_scale)
        if not math.isfinite(legacy_scale) or legacy_scale <= 0.0:
            raise ValueError(
                f"site profile flight.map_units_per_meter must be finite and > 0: {source}"
            )
    if schema_version == 1 and raw.get("controller") is not None:
        raise ValueError(
            "site profile schema v1 flight controllers are retired; migrate to "
            f"schema v2 scale-free controller: {source}"
        )
    note = raw.get("approval_note", "")
    if not isinstance(note, str):
        raise ValueError(f"site profile flight.approval_note must be a string: {source}")
    return FlightReadiness(
        approved=approved,
        coordinate_frame_id=None if frame_id is None else frame_id.strip(),
        route_clearance_approved=clearance,
        approval_note=note.strip(),
        controller=(
            None
            if schema_version == 1
            else _load_flight_controller(raw.get("controller"), source)
        ),
    )


def _load_hardware_approval_reference(
    raw, source: Path, profile_dir: Path
) -> HardwareApprovalReference | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"receipt", "sha256"}:
        raise ValueError(
            "site profile hardware_approval must contain exactly receipt and sha256: "
            f"{source}"
        )
    receipt = _resolve_asset(
        profile_dir, raw.get("receipt"), "hardware_approval.receipt", required=True
    )
    digest = _sha256(raw.get("sha256"), "hardware_approval", source)
    assert receipt is not None and digest is not None
    return HardwareApprovalReference(receipt=receipt, sha256=digest)


def _string_tuple(value, key: str, source: Path) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(
            f"hardware approval {key} must be a non-empty string list: {source}"
        )
    parsed = tuple(item.strip() for item in value)
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"hardware approval {key} must not contain duplicates: {source}")
    return parsed


def load_hardware_approval_receipt(
    reference: HardwareApprovalReference | None,
) -> HardwareApprovalReceipt:
    """Load a hash-pinned real-hardware approval; absence never implies approval."""
    if reference is None:
        raise ValueError("hardware approval receipt is not configured")
    try:
        content = reference.receipt.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read hardware approval receipt {reference.receipt}: {exc}"
        ) from exc
    actual = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(actual, reference.sha256):
        raise ValueError(
            "hardware approval receipt SHA-256 mismatch: "
            f"expected {reference.sha256}, got {actual} ({reference.receipt})"
        )
    try:
        raw = json.loads(content, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid hardware approval receipt {reference.receipt}: {exc}") from exc
    keys = {
        "schema",
        "approved",
        "aircraft_product",
        "controller_product",
        "aircraft_firmware_versions",
        "controller_firmware_versions",
        "olympe_versions",
        "approval_note",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        actual_keys = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(
            "hardware approval receipt fields mismatch: "
            f"unknown={sorted(actual_keys - keys)} missing={sorted(keys - actual_keys)}"
        )
    if raw["schema"] != "anafi-hardware-approval/v1":
        raise ValueError("unsupported hardware approval receipt schema")
    if not isinstance(raw["approved"], bool):
        raise ValueError("hardware approval approved must be boolean")
    for key in ("aircraft_product", "controller_product", "approval_note"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"hardware approval {key} must be non-empty")
    return HardwareApprovalReceipt(
        source=reference.receipt,
        sha256=actual,
        approved=raw["approved"],
        aircraft_product=raw["aircraft_product"].strip(),
        controller_product=raw["controller_product"].strip(),
        aircraft_firmware_versions=_string_tuple(
            raw["aircraft_firmware_versions"],
            "aircraft_firmware_versions",
            reference.receipt,
        ),
        controller_firmware_versions=_string_tuple(
            raw["controller_firmware_versions"],
            "controller_firmware_versions",
            reference.receipt,
        ),
        olympe_versions=_string_tuple(
            raw["olympe_versions"], "olympe_versions", reference.receipt
        ),
        approval_note=raw["approval_note"].strip(),
    )


def flight_readiness_errors(profile: SiteProfile) -> list[str]:
    """Return every reason this site profile is not eligible for autonomous flight."""
    if profile.schema_version != SCHEMA_VERSION:
        return [
            f"schema v{profile.schema_version} is localization-only; migrate to "
            f"scale-free schema v{SCHEMA_VERSION}"
        ]
    flight = profile.flight
    if flight is None:
        return ["missing flight contract"]
    errors = []
    if not flight.approved:
        errors.append("flight.approved is false")
    if not flight.approval_note:
        errors.append("missing flight.approval_note")
    if not flight.coordinate_frame_id:
        errors.append("missing flight.coordinate_frame_id")
    if not flight.route_clearance_approved:
        errors.append("flight.route_clearance_approved is false")
    if flight.controller is None:
        errors.append("missing flight.controller")
    if profile.route_json is None:
        errors.append("missing route_json")
    if profile.map_reference_poses is None:
        errors.append("missing map_reference_poses")
    if profile.map_align is None:
        # Fail closed: without a measured alignment the controller would silently
        # assume GLOMAP -Y is up, which tilts every commanded body axis.
        errors.append("missing map_align (measured T_align_gravity.json)")
    if profile.query_camera is None:
        errors.append("missing query_camera")
    if profile.localizer == "edm" and profile.localizer_profile is None:
        errors.append("missing EDM localizer_profile")
    for key in ("localization_bundle", "route_json", "map_reference_poses",
                "map_align"):
        if getattr(profile.asset_sha256, key) is None:
            errors.append(f"missing asset_sha256.{key}")
    if profile.localizer_profile is not None and profile.asset_sha256.localizer_profile is None:
        errors.append("missing asset_sha256.localizer_profile")
    if (
        flight.controller is not None
        and flight.controller.inspect_waypoints
        and profile.poles_json is None
    ):
        errors.append("inspection waypoints require poles_json")
    if (
        flight.controller is not None
        and flight.controller.inspect_waypoints
        and profile.asset_sha256.poles_json is None
    ):
        errors.append("inspection waypoints require asset_sha256.poles_json")
    return errors


def load_site_profile(path: str | Path, *, validate_files: bool = True) -> SiteProfile:
    """Load a profile and resolve relative asset paths beside the JSON file."""
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise ValueError(f"cannot read site profile {source}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid site profile JSON {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"site profile root must be an object: {source}")
    allowed_top_level = {
        "schema_version",
        "site_id",
        "display_name",
        "localizer",
        "localizer_deploy_dir",
        "localizer_profile",
        "map_reference_poses",
        "map_align",
        "query_camera",
        "coordinate_frame",
        "asset_sha256",
        "hardware_approval",
        "flight",
        "assets",
    }
    unknown_top_level = sorted(set(raw) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(
            f"unknown site profile keys {unknown_top_level}: {source}"
        )
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "site profile schema_version must be one of "
            f"{SUPPORTED_SCHEMA_VERSIONS}: {source}"
        )
    site_id = raw.get("site_id")
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError(f"site profile site_id must be a non-empty string: {source}")
    if not SITE_ID_RE.fullmatch(site_id.strip()):
        raise ValueError(
            "site profile site_id must be a path-safe slug containing only letters, "
            f"digits, dot, underscore, or hyphen: {source}"
        )
    display_name = raw.get("display_name", site_id)
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError(f"site profile display_name must be a non-empty string: {source}")
    if DISPLAY_NAME_REJECT_RE.search(display_name):
        raise ValueError(
            "site profile display_name must not contain control characters or any "
            f"of $ ` \" ' \\ : {source}"
        )
    localizer = raw.get("localizer", PRODUCTION_LOCALIZER_BACKEND)
    if (
        not isinstance(localizer, str)
        or localizer.strip().lower() != PRODUCTION_LOCALIZER_BACKEND
    ):
        raise ValueError(
            f"site profile localizer must be 'edm': {source}"
        )
    localizer = localizer.strip().lower()
    assets = raw.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(f"site profile assets must be an object: {source}")
    asset_keys = {
        "map_ply",
        "route_json",
        "localization_bundle",
        "megaloc_cache",
        "track_landmarks",
        "poles_json",
    }
    unknown_assets = sorted(set(assets) - asset_keys)
    if unknown_assets:
        raise ValueError(
            f"unknown site profile assets keys {unknown_assets}: {source}"
        )

    base = source.parent
    profile = SiteProfile(
        source=source,
        schema_version=schema_version,
        site_id=site_id.strip(),
        display_name=display_name.strip(),
        map_ply=_resolve_asset(base, assets.get("map_ply"), "map_ply", required=True),
        route_json=_resolve_asset(
            base, assets.get("route_json"), "route_json", required=False
        ),
        localization_bundle=_resolve_asset(
            base,
            assets.get("localization_bundle"),
            "localization_bundle",
            required=True,
        ),
        localizer=localizer,
        localizer_deploy_dir=_resolve_asset(
            base,
            raw.get("localizer_deploy_dir"),
            "localizer_deploy_dir",
            required=False,
        ),
        localizer_profile=_resolve_asset(
            base,
            raw.get("localizer_profile"),
            "localizer_profile",
            required=False,
        ),
        map_reference_poses=_resolve_asset(
            base,
            raw.get("map_reference_poses"),
            "map_reference_poses",
            required=False,
        ),
        map_align=_resolve_asset(
            base, raw.get("map_align"), "map_align", required=False
        ),
        megaloc_cache=_resolve_asset(
            base, assets.get("megaloc_cache"), "megaloc_cache", required=False
        ),
        track_landmarks=_resolve_asset(
            base, assets.get("track_landmarks"), "track_landmarks", required=False
        ),
        poles_json=_resolve_asset(
            base, assets.get("poles_json"), "poles_json", required=False
        ),
        query_camera=_load_query_camera(raw.get("query_camera"), source),
        coordinate_frame=_load_coordinate_frame(raw.get("coordinate_frame"), source),
        asset_sha256=_load_asset_digests(raw.get("asset_sha256"), source),
        flight=_load_flight_readiness(
            raw.get("flight"), source, schema_version=schema_version
        ),
        hardware_approval=_load_hardware_approval_reference(
            raw.get("hardware_approval"), source, base
        ),
    )
    if (
        profile.coordinate_frame is not None
        and profile.flight is not None
        and profile.flight.coordinate_frame_id is not None
        and profile.flight.coordinate_frame_id != profile.coordinate_frame.id
    ):
        raise ValueError(
            "site profile flight.coordinate_frame_id must match coordinate_frame.id: "
            f"{source}"
        )
    if validate_files:
        missing = [
            f"{name}={asset}"
            for name, asset in (
                ("map_ply", profile.map_ply),
                ("route_json", profile.route_json),
                ("localization_bundle", profile.localization_bundle),
                ("megaloc_cache", profile.megaloc_cache),
                ("track_landmarks", profile.track_landmarks),
                ("poles_json", profile.poles_json),
                ("map_reference_poses", profile.map_reference_poses),
                ("map_align", profile.map_align),
                ("localizer_profile", profile.localizer_profile),
                (
                    "hardware_approval.receipt",
                    None
                    if profile.hardware_approval is None
                    else profile.hardware_approval.receipt,
                ),
            )
            if asset is not None and not asset.is_file()
        ]
        if profile.localizer_deploy_dir is not None and not profile.localizer_deploy_dir.is_dir():
            missing.append(
                "localizer_deploy_dir=" + str(profile.localizer_deploy_dir)
            )
        if missing:
            raise ValueError(
                f"site profile {source} has missing asset(s): " + "; ".join(missing)
            )
        if profile.hardware_approval is not None:
            load_hardware_approval_receipt(profile.hardware_approval)
    return profile
