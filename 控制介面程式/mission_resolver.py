"""Compose independent mission manifests into one verified runtime snapshot.

The resolver is deliberately free of UI and aircraft SDK dependencies.  It is
safe to run during packaging, startup and preflight.  Existing launchers can use
``materialize_legacy_site_profile`` while they migrate away from the coupled
``site_profile`` format.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from navigation_pose_runtime import load_pose_calibrations
from mission_manifest import (
    ArtifactRef,
    CalibrationReceipt,
    CoordinateFrameContract,
    LocalizerVariantManifest,
    ManifestError,
    MapRevisionManifest,
    MissionApproval,
    MissionSelection,
    RoutePackageManifest,
    SiteManifest,
    VehicleManifest,
    load_calibration_receipt,
    load_localizer_manifest,
    load_map_manifest,
    load_mission_approval,
    load_mission_selection,
    load_route_manifest,
    load_site_manifest,
    load_vehicle_manifest,
    clear_sha256_cache,
    sha256_file,
)

_FLIGHT_CONTROL = Path(__file__).resolve().parents[1] / "定位演算法" / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))
try:
    from route_domain import validate_flight_route_fields  # noqa: E402
except ImportError:
    # Portable live packages ship no 定位演算法/flight_control. Route geometry
    # is only validated when a selection actually carries a route (below), so
    # keep the import optional and fail closed at that use-site instead of at
    # module import, which every launcher pays even for routeless missions.
    validate_flight_route_fields = None


SUPPORTED_PROVIDER_API_VERSION = 1
SUPPORTED_POSE_CONTRACT_VERSION = 1

#: The single localization blocker an evaluation-only session may waive.  An
#: unvalidated map is precisely what such a session exists to measure, so
#: refusing to start localization on it makes the measurement impossible.
#: Every other localization blocker (digest mismatch, camera-profile mismatch,
#: a missing or wrong-subject receipt) means the snapshot does not describe the
#: hardware in front of the operator, and no session may proceed on it.
EVALUATION_WAIVABLE_PREFIX = "localizer_quality calibration is failed or expired"


@dataclass(frozen=True, slots=True)
class MissionReadiness:
    localization_errors: tuple[str, ...]
    flight_errors: tuple[str, ...]

    @property
    def localization_ready(self) -> bool:
        return not self.localization_errors

    @property
    def flight_ready(self) -> bool:
        return not self.flight_errors

    @property
    def evaluation_only_ready(self) -> bool:
        """Localization may run for measurement only, never for navigation.

        True when the mission is blocked *solely* by an unvalidated map.  This
        never implies :attr:`flight_ready`: the quality error stays in
        ``flight_errors``, so the materialized profile keeps
        ``flight.approved`` and ``route_clearance_approved`` false and the
        autonomous-flight contract still rejects the mission.
        """
        return bool(self.localization_errors) and all(
            error.startswith(EVALUATION_WAIVABLE_PREFIX) for error in self.localization_errors
        )


@dataclass(frozen=True, slots=True)
class PoseChainSelection:
    site_frame_id: str
    vehicle_id: str
    body_frame_id: str
    camera_frame_id: str
    site_alignment: ArtifactRef
    camera_body_extrinsic: ArtifactRef


def _standard_flight_controller(frame: CoordinateFrameContract) -> dict[str, object]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    h_axes = [axis_map.get(str(ax).lstrip("+-").lower(), 0) for ax in frame.horizontal_axes]
    v_axis = axis_map.get(str(frame.up_axis).lstrip("+-").lower(), 1)
    if len(h_axes) != 2 or sorted([*h_axes, v_axis]) != [0, 1, 2]:
        h_axes = [0, 2]
        v_axis = 1
    body_right_sign = 1 if v_axis == 1 else -1
    return {
        "model": "scale_free_direction_speed_guard_v1",
        "speed_limit_mps": 0.60,
        "pose_max_age_ms": 500.0,
        "speed_max_age_ms": 500.0,
        "command_ttl_ms": 150.0,
        "yaw_tolerance_deg": 3.0,
        "horizontal_axes": list(h_axes),
        "vertical_axis": int(v_axis),
        "camera_to_body_yaw_deg": 0.0,
        "body_right_sign": int(body_right_sign),
        "lookahead_map_units": 0.8,
        "rejoin_tolerance_map_units": 0.4,
        "arrival_tolerance_map_units": 0.2,
        "inspect_radius_map_units": 0.5,
        "inspect_resume_margin_map_units": 0.2,
        "max_pose_jump_map_units": 1.0,
        "progress_jump_slack_map_units": 0.5,
        "max_progress_regression_map_units": 0.1,
        "segment_window": 2,
        "progress_speed_factor": 2.0,
        "inspect_waypoints": [],
    }


@dataclass(frozen=True, slots=True)
class ResolvedMission:
    """Hash-bound component graph that cannot silently change during a run."""

    workspace_root: Path
    selection: MissionSelection
    selection_sha256: str
    vehicle: VehicleManifest
    site: SiteManifest
    map_revision: MapRevisionManifest
    localizer: LocalizerVariantManifest
    route: RoutePackageManifest | None
    calibrations: tuple[CalibrationReceipt, ...]
    pose_chain: PoseChainSelection | None
    approval: MissionApproval | None
    readiness: MissionReadiness

    @property
    def identity(self) -> str:
        return f"{self.selection.selection_id}@{self.selection_sha256[:12]}"

    def _artifact_references(self) -> tuple[ArtifactRef, ...]:
        references: list[ArtifactRef] = [
            self.selection.vehicle,
            self.selection.site,
            self.selection.map_revision,
            self.selection.localizer,
            *self.selection.calibrations,
            *self.map_revision.assets.values(),
            *self.localizer.artifacts.values(),
        ]
        if self.selection.route is not None:
            references.append(self.selection.route)
        if self.selection.approval is not None:
            references.append(self.selection.approval)
        if self.route is not None:
            references.append(self.route.route)
        references.extend(
            receipt.artifact for receipt in self.calibrations if receipt.artifact is not None
        )
        return tuple(references)

    def verify_unchanged(self) -> None:
        """Recheck the selection document and every selected artifact."""
        current = sha256_file(self.selection.source)
        if not hmac.compare_digest(current, self.selection_sha256):
            raise ManifestError(
                "mission selection changed after resolution; resolve it again before use"
            )
        seen: set[tuple[Path, str]] = set()
        for reference in self._artifact_references():
            key = (reference.path, reference.sha256)
            if key in seen:
                continue
            reference.verify()
            seen.add(key)

    def legacy_site_profile_document(self) -> dict[str, object]:
        """Return a schema-v2 compatibility profile for existing launchers."""
        map_assets = self.map_revision.assets
        localizer_assets = self.localizer.artifacts
        route = (
            self.route.route if self.route is not None and self.route.frame_kind == "map" else None
        )
        profile = localizer_assets.get("profile")
        reference_index = localizer_assets.get("reference_index")
        megaloc_cache = localizer_assets.get("megaloc_cache")
        track_landmarks = localizer_assets.get("track_landmarks")
        approval_note = (
            self.approval.note
            if self.approval is not None
            else "Static mission approval is not required; operator preflight remains mandatory."
        )
        camera = self.vehicle.camera
        frame = self.map_revision.coordinate_frame
        pose_chain = self.pose_chain
        document = {
            "schema_version": 2,
            "site_id": self.site.site_id,
            "display_name": self.site.display_name,
            "localizer": self.localizer.algorithm_id,
            "localizer_profile": None if profile is None else str(profile.path),
            "map_reference_poses": str(map_assets["reference_poses"].path),
            "map_align": str(map_assets["map_align"].path),
            "query_camera": {
                "model": camera.model,
                "width": camera.width,
                "height": camera.height,
                "params": list(camera.params),
            },
            "coordinate_frame": {
                "id": frame.frame_id,
                "convention": frame.convention,
                "horizontal_axes": list(frame.horizontal_axes),
                "up_axis": frame.up_axis,
                "handedness": frame.handedness,
                "units": frame.units,
            },
            "asset_sha256": {
                "map_ply": map_assets["map_ply"].sha256,
                "localization_bundle": localizer_assets["bundle"].sha256,
                "route_json": None if route is None else route.sha256,
                "map_reference_poses": map_assets["reference_poses"].sha256,
                "localizer_profile": None if profile is None else profile.sha256,
                "reference_index": (None if reference_index is None else reference_index.sha256),
                "poles_json": None,
                "map_align": map_assets["map_align"].sha256,
                "site_alignment": (
                    None if pose_chain is None else pose_chain.site_alignment.sha256
                ),
                "camera_body_extrinsic": (
                    None if pose_chain is None else pose_chain.camera_body_extrinsic.sha256
                ),
            },
            "flight": {
                "approved": self.readiness.flight_ready and not evaluation_only_requested(),
                "coordinate_frame_id": frame.frame_id,
                "route_clearance_approved": self.readiness.flight_ready
                and not evaluation_only_requested(),
                "approval_note": approval_note,
                "controller": (
                    _standard_flight_controller(frame)
                    if self.readiness.flight_ready and not evaluation_only_requested()
                    else None
                ),
            },
            "assets": {
                "map_ply": str(map_assets["map_ply"].path),
                "route_json": None if route is None else str(route.path),
                "poles_json": None,
                "localization_bundle": str(localizer_assets["bundle"].path),
                "megaloc_cache": (None if megaloc_cache is None else str(megaloc_cache.path)),
                "track_landmarks": (None if track_landmarks is None else str(track_landmarks.path)),
                "reference_index": (None if reference_index is None else str(reference_index.path)),
            },
        }
        # A backend whose runtime does not live in the default deployment
        # directory (the direct two-rate localizer lives in sfm_direct_deploy)
        # declares it in the variant manifest; the launchers read it from the
        # compatibility profile.
        deploy_dir = self.localizer.runtime.get("deploy_dir")
        if deploy_dir is not None:
            declared = Path(str(deploy_dir)).expanduser()
            if not declared.is_absolute():
                declared = self.workspace_root / declared
            resolved = declared.resolve()
            if not resolved.is_dir():
                raise ManifestError(f"localizer.runtime.deploy_dir is not a directory: {resolved}")
            document["localizer_deploy_dir"] = str(resolved)
        if pose_chain is not None:
            document["pose_chain"] = {
                "site_frame_id": pose_chain.site_frame_id,
                "vehicle_id": pose_chain.vehicle_id,
                "body_frame_id": pose_chain.body_frame_id,
                "camera_frame_id": pose_chain.camera_frame_id,
                "site_alignment": str(pose_chain.site_alignment.path),
                "camera_body_extrinsic": str(pose_chain.camera_body_extrinsic.path),
            }
        return document

    def materialize_legacy_site_profile(self, output_dir: str | Path) -> Path:
        """Atomically create or reuse the compatibility profile for this snapshot."""
        self.verify_unchanged()
        directory = Path(output_dir).expanduser().resolve()
        if not directory.is_relative_to(self.workspace_root):
            raise ManifestError("compatibility profile output must stay in workspace")
        directory.mkdir(parents=True, exist_ok=True)
        # The document embeds the SFM_EVALUATION_ONLY flavor (flight.approved /
        # route_clearance_approved / controller), so the two flavors must not
        # share one filename: a production run materializing first used to make
        # every later evaluation-only launch fail closed with "disagrees with
        # snapshot" before it reached its own checks, and vice versa.
        flavor = ".eval" if evaluation_only_requested() else ""
        output = directory / f"{self.identity}{flavor}.site_profile.json"
        content = (
            json.dumps(
                self.legacy_site_profile_document(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if output.exists():
            if output.read_text(encoding="utf-8") != content:
                raise ManifestError(
                    f"existing compatibility profile disagrees with snapshot: {output}"
                )
            return output
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, output)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return output


def _calibration_by_kind(
    receipts: Iterable[CalibrationReceipt],
) -> tuple[dict[str, CalibrationReceipt], list[str]]:
    indexed: dict[str, CalibrationReceipt] = {}
    errors: list[str] = []
    for receipt in receipts:
        if receipt.kind in indexed:
            errors.append(f"duplicate calibration kind: {receipt.kind}")
            continue
        indexed[receipt.kind] = receipt
    return indexed, errors


def _receipt_error(
    receipt: CalibrationReceipt | None,
    *,
    kind: str,
    subject: str,
    now: datetime,
    localizer: LocalizerVariantManifest | None = None,
) -> str | None:
    if receipt is None:
        return f"missing current {kind} calibration for {subject}"
    if receipt.subject != subject:
        return f"{kind} calibration subject mismatch: expected {subject}, got {receipt.subject}"
    if not receipt.current(now):
        return f"{kind} calibration is failed or expired for {subject}"
    if localizer is not None:
        details = receipt.details
        bundle_sha = details.get("bundle_sha256")
        if isinstance(bundle_sha, str) and bundle_sha.strip():
            expected = localizer.artifacts.get("bundle")
            if expected is None or bundle_sha.strip().lower() != expected.sha256.lower():
                return (
                    f"{kind} calibration digests do not match the selected artifacts for {subject}"
                )
        profile_sha = details.get("profile_sha256")
        if isinstance(profile_sha, str) and profile_sha.strip():
            expected = localizer.artifacts.get("profile")
            if expected is None or profile_sha.strip().lower() != expected.sha256.lower():
                return (
                    f"{kind} calibration digests do not match the selected artifacts for {subject}"
                )
    return None


def _resolve_pose_chain(
    calibration_by_kind: dict[str, CalibrationReceipt],
    *,
    vehicle: VehicleManifest,
    site: SiteManifest,
    map_revision: MapRevisionManifest,
    now: datetime,
) -> tuple[PoseChainSelection | None, list[str]]:
    errors: list[str] = []
    site_receipt = calibration_by_kind.get("site_alignment")
    camera_receipt = calibration_by_kind.get("camera_body_extrinsic")
    if site_receipt is None and camera_receipt is None:
        return None, []
    site_subject = f"{site.site_id}@{map_revision.map_revision_id}"
    for receipt, kind, subject in (
        (site_receipt, "site_alignment", site_subject),
        (camera_receipt, "camera_body_extrinsic", vehicle.identity),
    ):
        error = _receipt_error(receipt, kind=kind, subject=subject, now=now)
        if error is not None:
            errors.append(error)

    if errors:
        return None, errors
    assert site_receipt is not None and camera_receipt is not None
    if site_receipt.artifact is None:
        errors.append("site_alignment calibration has no hash-bound artifact")
    if camera_receipt.artifact is None:
        errors.append("camera_body_extrinsic calibration has no hash-bound artifact")
    if errors:
        return None, errors
    assert site_receipt.artifact is not None and camera_receipt.artifact is not None

    try:
        alignment, extrinsic = load_pose_calibrations(
            site_alignment=site_receipt.artifact.path,
            camera_body_extrinsic=camera_receipt.artifact.path,
            map_frame_id=map_revision.coordinate_frame.frame_id,
            site_frame_id=site.site_frame_id,
            vehicle_id=vehicle.identity,
        )
    except ValueError as exc:
        return None, [f"pose_chain calibration is invalid: {exc}"]
    return (
        PoseChainSelection(
            site_frame_id=alignment.site_frame_id,
            vehicle_id=extrinsic.vehicle_id,
            body_frame_id=extrinsic.body_frame_id,
            camera_frame_id=extrinsic.camera_frame_id,
            site_alignment=site_receipt.artifact,
            camera_body_extrinsic=camera_receipt.artifact,
        ),
        [],
    )


def _compatibility_errors(
    vehicle: VehicleManifest,
    site: SiteManifest,
    map_revision: MapRevisionManifest,
    localizer: LocalizerVariantManifest,
    route: RoutePackageManifest | None,
) -> tuple[list[str], list[str]]:
    localization: list[str] = []
    route_errors: list[str] = []
    if map_revision.site_id != site.site_id:
        localization.append(f"map site_id {map_revision.site_id!r} does not match {site.site_id!r}")
    if localizer.map_revision_id != map_revision.map_revision_id:
        localization.append("localizer map revision does not match the selected map revision")
    if localizer.coordinate_frame_id != map_revision.coordinate_frame.frame_id:
        localization.append("localizer coordinate frame does not match the selected map frame")
    if localizer.provider_api_version != SUPPORTED_PROVIDER_API_VERSION:
        localization.append(
            f"unsupported localizer provider API version: {localizer.provider_api_version}"
        )
    if localizer.pose_contract_version != SUPPORTED_POSE_CONTRACT_VERSION:
        localization.append(
            f"unsupported localization pose contract version: {localizer.pose_contract_version}"
        )
    if vehicle.camera.pipeline_id not in localizer.camera_profiles:
        localization.append("vehicle camera pipeline is not validated by this localizer variant")
    missing_capabilities = sorted(localizer.required_vehicle_capabilities - vehicle.capabilities)
    if missing_capabilities:
        localization.append(
            "vehicle lacks localizer capabilities: " + ", ".join(missing_capabilities)
        )

    if route is None:
        route_errors.append("no route package selected")
    else:
        if route.site_id != site.site_id:
            route_errors.append("route site_id does not match selected site")
        if route.frame_kind == "map":
            if route.frame_id != map_revision.coordinate_frame.frame_id:
                route_errors.append("route map frame does not match selected map")
        else:
            if route.frame_id != site.site_frame_id:
                route_errors.append("route site frame does not match selected site")
            if map_revision.site_from_map is None:
                route_errors.append("site-frame route requires a measured site_from_map transform")
            route_errors.append(
                "site-frame route conversion is not supported by the legacy flight adapter"
            )
    return localization, route_errors


def resolve_mission(
    selection_path: str | Path,
    *,
    workspace_root: str | Path,
    now: datetime | None = None,
    verify_files: bool = True,
) -> ResolvedMission:
    """Load, hash-check and compatibility-check one mission selection."""
    root = Path(workspace_root).expanduser().resolve()
    selection = load_mission_selection(
        selection_path,
        workspace_root=root,
        verify_files=verify_files,
    )
    vehicle = load_vehicle_manifest(selection.vehicle.path, workspace_root=root)
    site = load_site_manifest(selection.site.path, workspace_root=root)
    map_revision = load_map_manifest(
        selection.map_revision.path,
        workspace_root=root,
        verify_files=verify_files,
    )
    localizer = load_localizer_manifest(
        selection.localizer.path,
        workspace_root=root,
        verify_files=verify_files,
    )
    route = (
        None
        if selection.route is None
        else load_route_manifest(
            selection.route.path,
            workspace_root=root,
            verify_files=verify_files,
        )
    )
    route_geometry_error: str | None = None
    if route is not None:
        if validate_flight_route_fields is None:
            route_geometry_error = (
                "flight route validation is unavailable: "
                "this package ships no flight_control/route_domain"
            )
        else:
            try:
                payload = json.loads(Path(route.route.path).read_text(encoding="utf-8"))
                validate_flight_route_fields(
                    payload,
                    expected_site_id=route.site_id,
                    expected_coordinate_frame_id=route.frame_id,
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                route_geometry_error = str(exc)
    calibrations = tuple(
        load_calibration_receipt(reference.path, workspace_root=root)
        for reference in selection.calibrations
    )
    approval = (
        None
        if selection.approval is None
        else load_mission_approval(selection.approval.path, workspace_root=root)
    )
    checked_at = now or datetime.now(timezone.utc)
    if checked_at.tzinfo is None:
        raise ManifestError("mission resolution time must include a timezone")
    checked_at = checked_at.astimezone(timezone.utc)

    localization_errors, route_errors = _compatibility_errors(
        vehicle,
        site,
        map_revision,
        localizer,
        route,
    )
    if route_geometry_error is not None:
        route_errors.append(route_geometry_error)
    deploy_dir = localizer.runtime.get("deploy_dir")
    if localizer.algorithm_id == "direct" and not deploy_dir:
        localization_errors.append("direct localizer requires runtime.deploy_dir")
    elif deploy_dir is not None:
        declared = Path(str(deploy_dir)).expanduser()
        resolved = (declared if declared.is_absolute() else root / declared).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            localization_errors.append(
                "localizer runtime.deploy_dir must be an existing workspace directory"
            )
    calibration_by_kind, duplicate_errors = _calibration_by_kind(calibrations)
    localization_errors.extend(duplicate_errors)
    quality_error = _receipt_error(
        calibration_by_kind.get("localizer_quality"),
        kind="localizer_quality",
        subject=localizer.quality_gate_id,
        now=checked_at,
        localizer=localizer,
    )
    if quality_error is not None:
        localization_errors.append(quality_error)

    pose_chain, pose_chain_errors = _resolve_pose_chain(
        calibration_by_kind,
        vehicle=vehicle,
        site=site,
        map_revision=map_revision,
        now=checked_at,
    )
    flight_errors = [*localization_errors, *route_errors, *pose_chain_errors]
    for kind in vehicle.required_calibrations:
        if kind == "camera_body_extrinsic":
            continue
        error = _receipt_error(
            calibration_by_kind.get(kind),
            kind=kind,
            subject=vehicle.identity,
            now=checked_at,
        )
        if error is not None:
            flight_errors.append(error)

    return ResolvedMission(
        workspace_root=root,
        selection=selection,
        selection_sha256=sha256_file(selection.source),
        vehicle=vehicle,
        site=site,
        map_revision=map_revision,
        localizer=localizer,
        route=route,
        calibrations=calibrations,
        pose_chain=pose_chain,
        approval=approval,
        readiness=MissionReadiness(
            localization_errors=tuple(localization_errors),
            flight_errors=tuple(flight_errors),
        ),
    )


def evaluation_only_requested() -> bool:
    """True when the operator explicitly asked for an evaluation-only session.

    Opt-in per process via ``SFM_EVALUATION_ONLY=1``.  ``IMU飛行測試.sh`` sets
    it; the ordinary ``一鍵啟動.sh`` entry point deliberately does not.
    """
    return os.environ.get("SFM_EVALUATION_ONLY") == "1"


def evaluation_only_admission(readiness: MissionReadiness) -> tuple[bool, tuple[str, ...]]:
    """Decide whether an evaluation-only session may start, and what it waives.

    Returns ``(admitted, waived_errors)``.  Admission requires both the explicit
    opt-in and a mission whose only localization blocker is the unvalidated map.
    """
    if not evaluation_only_requested() or not readiness.evaluation_only_ready:
        return False, ()
    return True, readiness.localization_errors


__all__ = [
    "EVALUATION_WAIVABLE_PREFIX",
    "MissionReadiness",
    "ResolvedMission",
    "SUPPORTED_POSE_CONTRACT_VERSION",
    "SUPPORTED_PROVIDER_API_VERSION",
    "evaluation_only_admission",
    "evaluation_only_requested",
    "resolve_mission",
    "clear_sha256_cache",
    "sha256_file",
]
