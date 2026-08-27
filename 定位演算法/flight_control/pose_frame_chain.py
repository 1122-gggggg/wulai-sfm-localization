"""Explicit camera -> body -> map -> navigation pose composition.

Transform names follow ``T_A_B``: coordinates in frame B are transformed into
frame A.  The visual localizer supplies ``T_C_M`` (map to OpenCV camera); its
inverse is composed with the measured fixed-gimbal ``T_C_B`` and the site's
Sim(3) ``T_W_M``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from site_alignment import SiteAlignment


EXTRINSIC_SCHEMA = "sfm-camera-body-extrinsic/v1"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _vec3(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 3-vector") from exc
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite numeric 3-vector")
    return array


def _proper_rotation(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 3x3 matrix") from exc
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(array)), 1.0, abs_tol=1e-7):
        raise ValueError(f"{label} must be right-handed (det=+1)")
    return array


def _strict_object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return value


@dataclass(frozen=True, slots=True)
class CameraBodyExtrinsic:
    """Measured fixed-gimbal ``T_C_B`` (body FRD to OpenCV camera)."""

    vehicle_id: str
    body_frame_id: str
    camera_frame_id: str
    rotation_camera_from_body: tuple[tuple[float, float, float], ...]
    translation_camera_from_body_m: tuple[float, float, float]
    fixed_gimbal_pitch_deg: float
    gimbal_pitch_tolerance_deg: float
    evidence: str
    approved: bool = False

    def __post_init__(self) -> None:
        for name in ("vehicle_id", "body_frame_id", "camera_frame_id", "evidence"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        rotation = _proper_rotation(
            self.rotation_camera_from_body, "R_camera_from_body"
        )
        object.__setattr__(
            self,
            "rotation_camera_from_body",
            tuple(tuple(float(value) for value in row) for row in rotation),
        )
        translation = _vec3(
            self.translation_camera_from_body_m, "t_camera_from_body_m"
        )
        object.__setattr__(
            self,
            "translation_camera_from_body_m",
            tuple(float(value) for value in translation),
        )
        pitch = float(self.fixed_gimbal_pitch_deg)
        tolerance = float(self.gimbal_pitch_tolerance_deg)
        if not math.isfinite(pitch) or not -90.0 <= pitch <= 90.0:
            raise ValueError("fixed_gimbal_pitch_deg must be between -90 and 90")
        if not math.isfinite(tolerance) or not 0.0 < tolerance <= 10.0:
            raise ValueError("gimbal_pitch_tolerance_deg must be in (0, 10]")
        if not isinstance(self.approved, bool):
            raise ValueError("extrinsic approved must be boolean")
        object.__setattr__(self, "fixed_gimbal_pitch_deg", pitch)
        object.__setattr__(self, "gimbal_pitch_tolerance_deg", tolerance)

    @property
    def R_C_B(self) -> np.ndarray:
        return np.asarray(self.rotation_camera_from_body, dtype=float)

    @property
    def t_C_B(self) -> np.ndarray:
        return np.asarray(self.translation_camera_from_body_m, dtype=float)

    def gimbal_ready(self, pitch_deg: object) -> bool:
        try:
            pitch = float(pitch_deg)
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(
            math.isfinite(pitch)
            and abs(pitch - self.fixed_gimbal_pitch_deg)
            <= self.gimbal_pitch_tolerance_deg
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EXTRINSIC_SCHEMA,
            "vehicle_id": self.vehicle_id,
            "body_frame_id": self.body_frame_id,
            "camera_frame_id": self.camera_frame_id,
            "mode": "fixed_gimbal_pitch",
            "fixed_gimbal_pitch_deg": self.fixed_gimbal_pitch_deg,
            "gimbal_pitch_tolerance_deg": self.gimbal_pitch_tolerance_deg,
            "T_camera_from_body": {
                "R": [list(row) for row in self.rotation_camera_from_body],
                "t_m": list(self.translation_camera_from_body_m),
            },
            "evidence": self.evidence,
            "approved": self.approved,
        }


def camera_body_extrinsic_from_dict(value: object) -> CameraBodyExtrinsic:
    raw = _strict_object(
        value,
        {
            "schema",
            "vehicle_id",
            "body_frame_id",
            "camera_frame_id",
            "mode",
            "fixed_gimbal_pitch_deg",
            "gimbal_pitch_tolerance_deg",
            "T_camera_from_body",
            "evidence",
            "approved",
        },
        "camera/body extrinsic",
    )
    if raw["schema"] != EXTRINSIC_SCHEMA:
        raise ValueError(f"camera/body extrinsic schema must be {EXTRINSIC_SCHEMA}")
    if raw["mode"] != "fixed_gimbal_pitch":
        raise ValueError("only fixed_gimbal_pitch extrinsics are supported")
    transform = _strict_object(
        raw["T_camera_from_body"], {"R", "t_m"}, "T_camera_from_body"
    )
    return CameraBodyExtrinsic(
        vehicle_id=raw["vehicle_id"],
        body_frame_id=raw["body_frame_id"],
        camera_frame_id=raw["camera_frame_id"],
        rotation_camera_from_body=transform["R"],
        translation_camera_from_body_m=transform["t_m"],
        fixed_gimbal_pitch_deg=raw["fixed_gimbal_pitch_deg"],
        gimbal_pitch_tolerance_deg=raw["gimbal_pitch_tolerance_deg"],
        evidence=raw["evidence"],
        approved=raw["approved"],
    )


def load_camera_body_extrinsic(
    path: str | Path,
    *,
    expected_vehicle_id: str | None = None,
) -> CameraBodyExtrinsic:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read camera/body extrinsic {source}: {exc}") from exc
    extrinsic = camera_body_extrinsic_from_dict(raw)
    if expected_vehicle_id is not None and extrinsic.vehicle_id != expected_vehicle_id:
        raise ValueError("camera/body extrinsic does not match the active vehicle")
    return extrinsic


def save_camera_body_extrinsic(
    extrinsic: CameraBodyExtrinsic, path: str | Path
) -> Path:
    if not isinstance(extrinsic, CameraBodyExtrinsic):
        raise TypeError("extrinsic must be a CameraBodyExtrinsic")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(extrinsic.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


@dataclass(frozen=True, slots=True)
class NavigationBodyPose:
    position_site_m: tuple[float, float, float]
    rotation_site_from_body: tuple[tuple[float, float, float], ...]
    yaw_rad: float
    stamp: float

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray(self.position_site_m, dtype=float)

    @property
    def R_W_B(self) -> np.ndarray:
        return np.asarray(self.rotation_site_from_body, dtype=float)


class NavigationPoseTransformer:
    """Compose ``T_W_M * inverse(T_C_M) * T_C_B``."""

    def __init__(
        self,
        site_alignment: SiteAlignment,
        camera_body_extrinsic: CameraBodyExtrinsic,
    ) -> None:
        if not site_alignment.approved:
            raise ValueError("site alignment is not approved")
        if not camera_body_extrinsic.approved:
            raise ValueError("camera/body extrinsic is not approved")
        self.site_alignment = site_alignment
        self.extrinsic = camera_body_extrinsic

    @property
    def scale_m_per_map_unit(self) -> float:
        return self.site_alignment.transform.scale

    def map_point_to_site(self, point: object) -> np.ndarray:
        return self.site_alignment.transform.map_to_site_point(point)

    def map_distance_to_site(self, value: object) -> float:
        distance = float(value)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("map distance must be finite and non-negative")
        return self.scale_m_per_map_unit * distance

    def body_pose(
        self,
        *,
        camera_center_map: object,
        rotation_camera_from_map: object,
        gimbal_pitch_deg: object,
        stamp: object,
    ) -> NavigationBodyPose:
        if not self.extrinsic.gimbal_ready(gimbal_pitch_deg):
            raise ValueError("gimbal pitch does not match the calibrated fixed pitch")
        camera_center = _vec3(camera_center_map, "camera center in map")
        R_C_M = _proper_rotation(rotation_camera_from_map, "R_camera_from_map")
        R_M_C = R_C_M.T
        R_M_B = R_M_C @ self.extrinsic.R_C_B
        p_M_B = camera_center + R_M_C @ self.extrinsic.t_C_B

        transform = self.site_alignment.transform
        p_W_B = transform.map_to_site_point(p_M_B)
        R_W_B = transform.map_to_site_rotation(R_M_B)
        R_W_B = _proper_rotation(R_W_B, "R_site_from_body")
        forward = R_W_B[:, 0]
        horizontal = math.hypot(float(forward[0]), float(forward[1]))
        if horizontal < 1e-6:
            raise ValueError("body forward axis has no navigation-plane heading")
        yaw = math.atan2(float(forward[1]), float(forward[0]))
        pose_stamp = float(stamp)
        if not math.isfinite(pose_stamp):
            raise ValueError("pose stamp must be finite")
        return NavigationBodyPose(
            tuple(float(value) for value in p_W_B),
            tuple(tuple(float(value) for value in row) for row in R_W_B),
            yaw,
            pose_stamp,
        )


__all__ = [
    "CameraBodyExtrinsic",
    "EXTRINSIC_SCHEMA",
    "NavigationBodyPose",
    "NavigationPoseTransformer",
    "camera_body_extrinsic_from_dict",
    "load_camera_body_extrinsic",
    "save_camera_body_extrinsic",
]
