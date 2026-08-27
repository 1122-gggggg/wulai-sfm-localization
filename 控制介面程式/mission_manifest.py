"""Versioned, dependency-light contracts for replaceable mission components.

The legacy :mod:`site_profile` document atomically selects a working deployment,
but it also couples the aircraft, reconstruction, localizer and route.  These
contracts keep those identities independent.  ``mission_resolver`` is the only
place that composes them into a runnable, immutable selection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn


SCHEMAS = {
    "vehicle": "sfm-vehicle/v1",
    "site": "sfm-site/v1",
    "map": "sfm-map-revision/v1",
    "localizer": "sfm-localizer-variant/v1",
    "route": "sfm-route-package/v1",
    "calibration": "sfm-calibration-receipt/v1",
    "approval": "sfm-mission-approval/v1",
    "selection": "sfm-mission-selection/v1",
}
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A component manifest is malformed, unsafe or has lost its identity."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> NoReturn:
    raise ManifestError(f"non-finite JSON number is not allowed: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    except (json.JSONDecodeError, ManifestError) as exc:
        raise ManifestError(f"invalid manifest JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest root must be an object: {path}")
    return value


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ManifestError(f"{label} missing keys: {', '.join(missing)}")
    if unknown:
        raise ManifestError(f"{label} has unknown keys: {', '.join(unknown)}")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: object, label: str) -> str:
    parsed = _text(value, label)
    if not ID_RE.fullmatch(parsed):
        raise ManifestError(f"{label} must be a path-safe identifier")
    return parsed


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or positive and parsed <= 0.0:
        suffix = " and > 0" if positive else ""
        raise ManifestError(f"{label} must be finite{suffix}")
    return parsed


def _string_tuple(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be an array")
    parsed = tuple(_text(item, f"{label}[]") for item in value)
    if not allow_empty and not parsed:
        raise ManifestError(f"{label} must not be empty")
    if len(set(parsed)) != len(parsed):
        raise ManifestError(f"{label} must not contain duplicates")
    return parsed


def _root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"workspace root is not a directory: {root}")
    return root


def _safe_path(
    base: Path,
    value: object,
    label: str,
    workspace_root: Path,
    *,
    require_file: bool,
) -> Path:
    raw = str(value) if isinstance(value, Path) else _text(value, label)
    if not raw.strip():
        raise ManifestError(f"{label} must be a non-empty path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ManifestError(f"{label} cannot be resolved: {raw}") from exc
    if not resolved.is_relative_to(workspace_root):
        raise ManifestError(f"{label} escapes workspace root: {raw}")
    current = candidate.absolute()
    while True:
        try:
            if current.is_symlink():
                raise ManifestError(f"{label} must not use symlinks: {raw}")
        except OSError as exc:
            raise ManifestError(f"{label} cannot be inspected: {raw}") from exc
        if current == current.parent or current == workspace_root:
            break
        current = current.parent
    if require_file and not resolved.is_file():
        raise ManifestError(f"{label} is not a file: {resolved}")
    return resolved


def _manifest_document(
    path: str | Path,
    workspace_root: str | Path,
    kind: str,
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> tuple[Path, Path, dict[str, Any]]:
    root = _root(workspace_root)
    source = _safe_path(
        root,
        Path(path).expanduser(),
        f"{kind} manifest",
        root,
        require_file=True,
    )
    raw = _read_json(source)
    _strict_keys(
        raw,
        label=f"{kind} manifest",
        required={"schema", *required},
        optional=optional,
    )
    expected = SCHEMAS[kind]
    if raw["schema"] != expected:
        raise ManifestError(f"{kind} manifest schema must be {expected!r}")
    return root, source, raw


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: Path
    sha256: str

    def verify(self) -> None:
        actual = sha256_file(self.path)
        if not hmac.compare_digest(actual, self.sha256):
            raise ManifestError(
                f"artifact SHA-256 mismatch for {self.path}: expected {self.sha256}, got {actual}"
            )


def _artifact(
    value: object,
    label: str,
    *,
    base: Path,
    workspace_root: Path,
    verify_files: bool,
) -> ArtifactRef:
    raw = _object(value, label)
    _strict_keys(raw, label=label, required={"path", "sha256"})
    digest = _text(raw["sha256"], f"{label}.sha256").lower()
    if not SHA256_RE.fullmatch(digest):
        raise ManifestError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
    reference = ArtifactRef(
        path=_safe_path(
            base,
            raw["path"],
            f"{label}.path",
            workspace_root,
            require_file=verify_files,
        ),
        sha256=digest,
    )
    if verify_files:
        reference.verify()
    return reference


def _artifacts(
    value: object,
    label: str,
    *,
    base: Path,
    workspace_root: Path,
    verify_files: bool,
) -> Mapping[str, ArtifactRef]:
    raw = _object(value, label)
    parsed = {
        _identifier(key, f"{label} key"): _artifact(
            item,
            f"{label}.{key}",
            base=base,
            workspace_root=workspace_root,
            verify_files=verify_files,
        )
        for key, item in raw.items()
    }
    return MappingProxyType(parsed)


@dataclass(frozen=True, slots=True)
class CameraContract:
    camera_id: str
    pipeline_id: str
    model: str
    width: int
    height: int
    params: tuple[float, ...]


def _camera(value: object, label: str) -> CameraContract:
    raw = _object(value, label)
    _strict_keys(
        raw,
        label=label,
        required={"camera_id", "pipeline_id", "model", "width", "height", "params"},
    )
    params_raw = raw["params"]
    if not isinstance(params_raw, list) or not params_raw:
        raise ManifestError(f"{label}.params must be a non-empty array")
    params = tuple(_number(value, f"{label}.params[]") for value in params_raw)
    return CameraContract(
        camera_id=_identifier(raw["camera_id"], f"{label}.camera_id"),
        pipeline_id=_identifier(raw["pipeline_id"], f"{label}.pipeline_id"),
        model=_text(raw["model"], f"{label}.model").upper(),
        width=_integer(raw["width"], f"{label}.width", minimum=1),
        height=_integer(raw["height"], f"{label}.height", minimum=1),
        params=params,
    )


@dataclass(frozen=True, slots=True)
class VehicleManifest:
    source: Path
    vehicle_id: str
    revision: str
    adapter: str
    model: str
    serials: tuple[str, ...]
    capabilities: frozenset[str]
    required_calibrations: tuple[str, ...]
    limits: Mapping[str, float]
    camera: CameraContract

    @property
    def identity(self) -> str:
        return f"{self.vehicle_id}@{self.revision}"


def load_vehicle_manifest(
    path: str | Path,
    *,
    workspace_root: str | Path,
) -> VehicleManifest:
    _root_path, source, raw = _manifest_document(
        path,
        workspace_root,
        "vehicle",
        required={
            "vehicle_id",
            "revision",
            "adapter",
            "model",
            "serials",
            "capabilities",
            "required_calibrations",
            "limits",
            "camera",
        },
    )
    limits_raw = _object(raw["limits"], "vehicle.limits")
    limits = MappingProxyType(
        {
            _identifier(key, "vehicle.limits key"): _number(
                value, f"vehicle.limits.{key}", positive=True
            )
            for key, value in limits_raw.items()
        }
    )
    return VehicleManifest(
        source=source,
        vehicle_id=_identifier(raw["vehicle_id"], "vehicle.vehicle_id"),
        revision=_identifier(raw["revision"], "vehicle.revision"),
        adapter=_identifier(raw["adapter"], "vehicle.adapter"),
        model=_text(raw["model"], "vehicle.model"),
        serials=_string_tuple(raw["serials"], "vehicle.serials", allow_empty=True),
        capabilities=frozenset(_string_tuple(raw["capabilities"], "vehicle.capabilities")),
        required_calibrations=_string_tuple(
            raw["required_calibrations"],
            "vehicle.required_calibrations",
            allow_empty=True,
        ),
        limits=limits,
        camera=_camera(raw["camera"], "vehicle.camera"),
    )


@dataclass(frozen=True, slots=True)
class SiteManifest:
    source: Path
    site_id: str
    display_name: str
    site_frame_id: str
    units: str


def load_site_manifest(path: str | Path, *, workspace_root: str | Path) -> SiteManifest:
    _root_path, source, raw = _manifest_document(
        path,
        workspace_root,
        "site",
        required={"site_id", "display_name", "site_frame_id", "units"},
    )
    units = _text(raw["units"], "site.units")
    if units != "m":
        raise ManifestError("site.units must be 'm'")
    return SiteManifest(
        source=source,
        site_id=_identifier(raw["site_id"], "site.site_id"),
        display_name=_text(raw["display_name"], "site.display_name"),
        site_frame_id=_identifier(raw["site_frame_id"], "site.site_frame_id"),
        units=units,
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _determinant(rows: tuple[tuple[float, float, float], ...]) -> float:
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    """Measured ``p_site = scale * R * p_map + t`` transform."""

    scale: float
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    evidence: str

    def to_site(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        values = tuple(
            self.scale * _dot(row, point) + self.translation[index]
            for index, row in enumerate(self.rotation)
        )
        return values[0], values[1], values[2]

    def to_map(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        shifted = tuple((point[index] - self.translation[index]) / self.scale for index in range(3))
        values = tuple(
            sum(self.rotation[row][column] * shifted[row] for row in range(3))
            for column in range(3)
        )
        return values[0], values[1], values[2]


def _similarity(value: object, label: str) -> SimilarityTransform:
    raw = _object(value, label)
    _strict_keys(
        raw,
        label=label,
        required={"scale", "R", "t", "evidence"},
    )
    rotation_raw = raw["R"]
    if not isinstance(rotation_raw, list) or len(rotation_raw) != 3:
        raise ManifestError(f"{label}.R must be a 3x3 array")
    rows: list[tuple[float, float, float]] = []
    for index, row in enumerate(rotation_raw):
        if not isinstance(row, list) or len(row) != 3:
            raise ManifestError(f"{label}.R[{index}] must contain three numbers")
        values = tuple(_number(item, f"{label}.R[{index}][]") for item in row)
        rows.append((values[0], values[1], values[2]))
    rotation = (rows[0], rows[1], rows[2])
    for index, row in enumerate(rotation):
        if not math.isclose(_dot(row, row), 1.0, abs_tol=1e-6):
            raise ManifestError(f"{label}.R row {index} is not unit length")
        for other in range(index):
            if not math.isclose(_dot(row, rotation[other]), 0.0, abs_tol=1e-6):
                raise ManifestError(f"{label}.R rows are not orthogonal")
    if not math.isclose(_determinant(rotation), 1.0, abs_tol=1e-6):
        raise ManifestError(f"{label}.R must be a right-handed rotation")
    translation_raw = raw["t"]
    if not isinstance(translation_raw, list) or len(translation_raw) != 3:
        raise ManifestError(f"{label}.t must contain three numbers")
    translation_values = tuple(_number(item, f"{label}.t[]") for item in translation_raw)
    return SimilarityTransform(
        scale=_number(raw["scale"], f"{label}.scale", positive=True),
        rotation=rotation,
        translation=(
            translation_values[0],
            translation_values[1],
            translation_values[2],
        ),
        evidence=_text(raw["evidence"], f"{label}.evidence"),
    )


@dataclass(frozen=True, slots=True)
class CoordinateFrameContract:
    frame_id: str
    convention: str
    horizontal_axes: tuple[str, str]
    up_axis: str
    handedness: str
    units: str


def _coordinate_frame(value: object, label: str) -> CoordinateFrameContract:
    raw = _object(value, label)
    _strict_keys(
        raw,
        label=label,
        required={
            "id",
            "convention",
            "horizontal_axes",
            "up_axis",
            "handedness",
            "units",
        },
    )
    axes = _string_tuple(raw["horizontal_axes"], f"{label}.horizontal_axes")
    if len(axes) != 2:
        raise ManifestError(f"{label}.horizontal_axes must contain two axes")
    return CoordinateFrameContract(
        frame_id=_identifier(raw["id"], f"{label}.id"),
        convention=_identifier(raw["convention"], f"{label}.convention"),
        horizontal_axes=(axes[0], axes[1]),
        up_axis=_text(raw["up_axis"], f"{label}.up_axis"),
        handedness=_text(raw["handedness"], f"{label}.handedness"),
        units=_text(raw["units"], f"{label}.units"),
    )


@dataclass(frozen=True, slots=True)
class MapRevisionManifest:
    source: Path
    site_id: str
    map_revision_id: str
    coordinate_frame: CoordinateFrameContract
    assets: Mapping[str, ArtifactRef]
    site_from_map: SimilarityTransform | None


def load_map_manifest(
    path: str | Path,
    *,
    workspace_root: str | Path,
    verify_files: bool = True,
) -> MapRevisionManifest:
    root, source, raw = _manifest_document(
        path,
        workspace_root,
        "map",
        required={"site_id", "map_revision_id", "coordinate_frame", "assets"},
        optional={"site_from_map"},
    )
    assets = _artifacts(
        raw["assets"],
        "map.assets",
        base=source.parent,
        workspace_root=root,
        verify_files=verify_files,
    )
    missing = sorted({"map_ply", "reference_poses", "map_align"} - set(assets))
    if missing:
        raise ManifestError(f"map.assets missing required assets: {', '.join(missing)}")
    site_from_map = raw.get("site_from_map")
    return MapRevisionManifest(
        source=source,
        site_id=_identifier(raw["site_id"], "map.site_id"),
        map_revision_id=_identifier(raw["map_revision_id"], "map.map_revision_id"),
        coordinate_frame=_coordinate_frame(raw["coordinate_frame"], "map.coordinate_frame"),
        assets=assets,
        site_from_map=None
        if site_from_map is None
        else _similarity(site_from_map, "map.site_from_map"),
    )


@dataclass(frozen=True, slots=True)
class LocalizerVariantManifest:
    source: Path
    algorithm_id: str
    variant_id: str
    provider_api_version: int
    pose_contract_version: int
    map_revision_id: str
    coordinate_frame_id: str
    camera_profiles: tuple[str, ...]
    required_vehicle_capabilities: frozenset[str]
    quality_gate_id: str
    artifacts: Mapping[str, ArtifactRef]
    runtime: Mapping[str, Any]


def load_localizer_manifest(
    path: str | Path,
    *,
    workspace_root: str | Path,
    verify_files: bool = True,
) -> LocalizerVariantManifest:
    root, source, raw = _manifest_document(
        path,
        workspace_root,
        "localizer",
        required={
            "algorithm_id",
            "variant_id",
            "provider_api_version",
            "pose_contract_version",
            "map_revision_id",
            "coordinate_frame_id",
            "camera_profiles",
            "required_vehicle_capabilities",
            "quality_gate_id",
            "artifacts",
            "runtime",
        },
    )
    artifacts = _artifacts(
        raw["artifacts"],
        "localizer.artifacts",
        base=source.parent,
        workspace_root=root,
        verify_files=verify_files,
    )
    if "bundle" not in artifacts:
        raise ManifestError("localizer.artifacts must include bundle")
    runtime = _object(raw["runtime"], "localizer.runtime")
    return LocalizerVariantManifest(
        source=source,
        algorithm_id=_identifier(raw["algorithm_id"], "localizer.algorithm_id").lower(),
        variant_id=_identifier(raw["variant_id"], "localizer.variant_id"),
        provider_api_version=_integer(
            raw["provider_api_version"], "localizer.provider_api_version", minimum=1
        ),
        pose_contract_version=_integer(
            raw["pose_contract_version"], "localizer.pose_contract_version", minimum=1
        ),
        map_revision_id=_identifier(raw["map_revision_id"], "localizer.map_revision_id"),
        coordinate_frame_id=_identifier(
            raw["coordinate_frame_id"], "localizer.coordinate_frame_id"
        ),
        camera_profiles=_string_tuple(raw["camera_profiles"], "localizer.camera_profiles"),
        required_vehicle_capabilities=frozenset(
            _string_tuple(
                raw["required_vehicle_capabilities"],
                "localizer.required_vehicle_capabilities",
                allow_empty=True,
            )
        ),
        quality_gate_id=_identifier(raw["quality_gate_id"], "localizer.quality_gate_id"),
        artifacts=artifacts,
        runtime=MappingProxyType(dict(runtime)),
    )


@dataclass(frozen=True, slots=True)
class RoutePackageManifest:
    source: Path
    route_id: str
    revision: str
    site_id: str
    frame_kind: str
    frame_id: str
    route: ArtifactRef

    @property
    def identity(self) -> str:
        return f"{self.route_id}@{self.revision}"


def _validate_route_artifact(
    reference: ArtifactRef,
    *,
    site_id: str,
    frame_kind: str,
    frame_id: str,
) -> None:
    raw = _read_json(reference.path)
    if raw.get("schema") != "sfm-flight-route/v1":
        raise ManifestError("route artifact schema must be 'sfm-flight-route/v1'")
    if raw.get("purpose") != "flight":
        raise ManifestError("route artifact purpose must be 'flight'")
    if raw.get("site_id") != site_id:
        raise ManifestError("route artifact site_id does not match route manifest")
    if raw.get("coordinate_frame_id") != frame_id:
        raise ManifestError("route artifact coordinate_frame_id does not match route manifest")
    expected_units = "map" if frame_kind == "map" else "m"
    if raw.get("units") != expected_units:
        raise ManifestError(f"route artifact units must be {expected_units!r}")
    waypoints = raw.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ManifestError("route artifact requires at least two waypoints")
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, list) or len(waypoint) != 3:
            raise ManifestError(f"route waypoint {index} must contain three numbers")
        for axis, value in enumerate(waypoint):
            _number(value, f"route.waypoints[{index}][{axis}]")


def load_route_manifest(
    path: str | Path,
    *,
    workspace_root: str | Path,
    verify_files: bool = True,
) -> RoutePackageManifest:
    root, source, raw = _manifest_document(
        path,
        workspace_root,
        "route",
        required={"route_id", "revision", "site_id", "frame", "route"},
    )
    frame = _object(raw["frame"], "route.frame")
    _strict_keys(frame, label="route.frame", required={"kind", "id"})
    frame_kind = _text(frame["kind"], "route.frame.kind")
    if frame_kind not in {"site", "map"}:
        raise ManifestError("route.frame.kind must be 'site' or 'map'")
    site_id = _identifier(raw["site_id"], "route.site_id")
    frame_id = _identifier(frame["id"], "route.frame.id")
    route = _artifact(
        raw["route"],
        "route.route",
        base=source.parent,
        workspace_root=root,
        verify_files=verify_files,
    )
    if verify_files:
        _validate_route_artifact(
            route,
            site_id=site_id,
            frame_kind=frame_kind,
            frame_id=frame_id,
        )
    return RoutePackageManifest(
        source=source,
        route_id=_identifier(raw["route_id"], "route.route_id"),
        revision=_identifier(raw["revision"], "route.revision"),
        site_id=site_id,
        frame_kind=frame_kind,
        frame_id=frame_id,
        route=route,
    )


def _timestamp(value: object, label: str) -> datetime:
    raw = _text(value, label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class CalibrationReceipt:
    source: Path
    receipt_id: str
    kind: str
    subject: str
    passed: bool
    issued_at: datetime
    expires_at: datetime | None
    details: Mapping[str, Any]
    artifact: ArtifactRef | None

    def current(self, now: datetime) -> bool:
        normalized = now.astimezone(timezone.utc)
        return self.passed and (self.expires_at is None or normalized <= self.expires_at)


def load_calibration_receipt(
    path: str | Path,
    *,
    workspace_root: str | Path,
) -> CalibrationReceipt:
    root, source, raw = _manifest_document(
        path,
        workspace_root,
        "calibration",
        required={
            "receipt_id",
            "kind",
            "subject",
            "passed",
            "issued_at",
            "expires_at",
            "details",
        },
    )
    passed = raw["passed"]
    if not isinstance(passed, bool):
        raise ManifestError("calibration.passed must be boolean")
    expires_raw = raw["expires_at"]
    expires = None if expires_raw is None else _timestamp(expires_raw, "calibration.expires_at")
    issued = _timestamp(raw["issued_at"], "calibration.issued_at")
    if expires is not None and expires < issued:
        raise ManifestError("calibration.expires_at cannot precede issued_at")
    details = dict(_object(raw["details"], "calibration.details"))
    artifact_raw = details.get("artifact")
    artifact = (
        None
        if artifact_raw is None
        else _artifact(
            artifact_raw,
            "calibration.details.artifact",
            base=source.parent,
            workspace_root=root,
            verify_files=True,
        )
    )
    return CalibrationReceipt(
        source=source,
        receipt_id=_identifier(raw["receipt_id"], "calibration.receipt_id"),
        kind=_identifier(raw["kind"], "calibration.kind"),
        subject=_text(raw["subject"], "calibration.subject"),
        passed=passed,
        issued_at=issued,
        expires_at=expires,
        details=MappingProxyType(details),
        artifact=artifact,
    )


@dataclass(frozen=True, slots=True)
class MissionApproval:
    source: Path
    approval_id: str
    approved: bool
    route_clearance_approved: bool
    component_sha256: Mapping[str, str]
    note: str


def load_mission_approval(
    path: str | Path,
    *,
    workspace_root: str | Path,
) -> MissionApproval:
    _root_path, source, raw = _manifest_document(
        path,
        workspace_root,
        "approval",
        required={
            "approval_id",
            "approved",
            "route_clearance_approved",
            "component_sha256",
            "note",
        },
    )
    approved = raw["approved"]
    route_approved = raw["route_clearance_approved"]
    if not isinstance(approved, bool) or not isinstance(route_approved, bool):
        raise ManifestError("approval flags must be boolean")
    bindings_raw = _object(raw["component_sha256"], "approval.component_sha256")
    bindings: dict[str, str] = {}
    for key, value in bindings_raw.items():
        name = _identifier(key, "approval.component_sha256 key")
        digest = _text(value, f"approval.component_sha256.{name}").lower()
        if not SHA256_RE.fullmatch(digest):
            raise ManifestError(f"approval.component_sha256.{name} must be a SHA-256 digest")
        bindings[name] = digest
    return MissionApproval(
        source=source,
        approval_id=_identifier(raw["approval_id"], "approval.approval_id"),
        approved=approved,
        route_clearance_approved=route_approved,
        component_sha256=MappingProxyType(bindings),
        note=_text(raw["note"], "approval.note"),
    )


@dataclass(frozen=True, slots=True)
class MissionSelection:
    source: Path
    selection_id: str
    vehicle: ArtifactRef
    site: ArtifactRef
    map_revision: ArtifactRef
    localizer: ArtifactRef
    route: ArtifactRef | None
    calibrations: tuple[ArtifactRef, ...]
    approval: ArtifactRef | None

    @property
    def component_sha256(self) -> Mapping[str, str]:
        values = {
            "vehicle": self.vehicle.sha256,
            "site": self.site.sha256,
            "map": self.map_revision.sha256,
            "localizer": self.localizer.sha256,
        }
        if self.route is not None:
            values["route"] = self.route.sha256
        for index, receipt in enumerate(self.calibrations):
            values[f"calibration_{index}"] = receipt.sha256
        return MappingProxyType(values)


def load_mission_selection(
    path: str | Path,
    *,
    workspace_root: str | Path,
    verify_files: bool = True,
) -> MissionSelection:
    root, source, raw = _manifest_document(
        path,
        workspace_root,
        "selection",
        required={
            "selection_id",
            "vehicle",
            "site",
            "map",
            "localizer",
            "route",
            "calibrations",
        },
        optional={"approval"},
    )

    def reference(value: object, label: str) -> ArtifactRef:
        return _artifact(
            value,
            label,
            base=source.parent,
            workspace_root=root,
            verify_files=verify_files,
        )

    calibration_raw = raw["calibrations"]
    if not isinstance(calibration_raw, list):
        raise ManifestError("selection.calibrations must be an array")
    route_raw = raw["route"]
    approval_raw = raw.get("approval")
    return MissionSelection(
        source=source,
        selection_id=_identifier(raw["selection_id"], "selection.selection_id"),
        vehicle=reference(raw["vehicle"], "selection.vehicle"),
        site=reference(raw["site"], "selection.site"),
        map_revision=reference(raw["map"], "selection.map"),
        localizer=reference(raw["localizer"], "selection.localizer"),
        route=None if route_raw is None else reference(route_raw, "selection.route"),
        calibrations=tuple(
            reference(item, f"selection.calibrations[{index}]")
            for index, item in enumerate(calibration_raw)
        ),
        approval=None if approval_raw is None else reference(approval_raw, "selection.approval"),
    )


__all__ = [
    "ArtifactRef",
    "CalibrationReceipt",
    "CameraContract",
    "CoordinateFrameContract",
    "LocalizerVariantManifest",
    "ManifestError",
    "MapRevisionManifest",
    "MissionApproval",
    "MissionSelection",
    "RoutePackageManifest",
    "SCHEMAS",
    "SimilarityTransform",
    "SiteManifest",
    "VehicleManifest",
    "load_calibration_receipt",
    "load_localizer_manifest",
    "load_map_manifest",
    "load_mission_approval",
    "load_mission_selection",
    "load_route_manifest",
    "load_site_manifest",
    "load_vehicle_manifest",
    "sha256_file",
]
