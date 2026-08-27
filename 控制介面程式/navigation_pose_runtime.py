"""Load the measured map/camera/body pose chain used by autonomous flight."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


MIN_APPROVED_CONTROL_POINTS = 6


def _flight_control_imports() -> tuple[Any, Any, Any]:
    flight_control = (
        Path(__file__).resolve().parents[1] / "定位演算法" / "flight_control"
    )
    if str(flight_control) not in sys.path:
        sys.path.insert(0, str(flight_control))
    from pose_frame_chain import (  # type: ignore[import-not-found]
        NavigationPoseTransformer,
        load_camera_body_extrinsic,
    )
    from site_alignment import load_site_alignment  # type: ignore[import-not-found]

    return NavigationPoseTransformer, load_camera_body_extrinsic, load_site_alignment


def load_pose_calibrations(
    *,
    site_alignment: str | Path,
    camera_body_extrinsic: str | Path,
    map_frame_id: str,
    site_frame_id: str,
    vehicle_id: str,
) -> tuple[Any, Any]:
    """Load and validate the two measured documents required before AUTO."""

    (
        _transformer_type,
        load_camera_body_extrinsic,
        load_site_alignment,
    ) = _flight_control_imports()
    alignment = load_site_alignment(
        site_alignment,
        expected_map_frame_id=map_frame_id,
        expected_site_frame_id=site_frame_id,
    )
    if not alignment.approved:
        raise ValueError("site alignment is not approved")
    if len(alignment.control_points) < MIN_APPROVED_CONTROL_POINTS:
        raise ValueError(
            "approved site alignment requires at least "
            f"{MIN_APPROVED_CONTROL_POINTS} measured control points"
        )
    extrinsic = load_camera_body_extrinsic(
        camera_body_extrinsic,
        expected_vehicle_id=vehicle_id,
    )
    if not extrinsic.approved:
        raise ValueError("camera/body extrinsic is not approved")
    return alignment, extrinsic


def load_navigation_pose_transformer(profile: Any) -> Any:
    """Return a verified ``T_W_M * inverse(T_C_M) * T_C_B`` transformer.

    The configuration is intentionally fail-closed.  A transform is usable for
    AUTO only after both measured documents are explicitly approved, hash pins
    are handled by the mission pipeline, and the active frame/vehicle identities
    match exactly.
    """

    pose_chain = getattr(profile, "pose_chain", None)
    if pose_chain is None:
        raise ValueError(
            "missing pose_chain (measured map/site and camera/body calibration)"
        )
    coordinate_frame = getattr(profile, "coordinate_frame", None)
    if coordinate_frame is None:
        raise ValueError("pose_chain requires coordinate_frame")

    transformer_type, _load_extrinsic, _load_alignment = _flight_control_imports()
    alignment, extrinsic = load_pose_calibrations(
        site_alignment=pose_chain.site_alignment,
        camera_body_extrinsic=pose_chain.camera_body_extrinsic,
        map_frame_id=coordinate_frame.id,
        site_frame_id=pose_chain.site_frame_id,
        vehicle_id=pose_chain.vehicle_id,
    )
    if extrinsic.body_frame_id != pose_chain.body_frame_id:
        raise ValueError("camera/body extrinsic body frame does not match pose_chain")
    if extrinsic.camera_frame_id != pose_chain.camera_frame_id:
        raise ValueError("camera/body extrinsic camera frame does not match pose_chain")
    return transformer_type(alignment, extrinsic)


def pose_chain_readiness_errors(profile: Any) -> list[str]:
    """Return the single actionable pose-chain blocker, if any."""

    try:
        load_navigation_pose_transformer(profile)
    except ValueError as exc:
        return [f"pose_chain: {exc}"]
    return []


__all__ = [
    "MIN_APPROVED_CONTROL_POINTS",
    "load_navigation_pose_transformer",
    "load_pose_calibrations",
    "pose_chain_readiness_errors",
]
