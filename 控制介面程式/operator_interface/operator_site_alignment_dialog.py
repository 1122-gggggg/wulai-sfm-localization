"""Manual and guided map-to-site alignment models and Tk dialogs.

The model owns only measured control points and the resulting alignment file.
It has no connection to a vehicle, command backend, or authority state.  The
current map pose is deliberately supplied by callbacks so an integration can
choose a read-only localization source without coupling this module to it.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np


# Keep this module importable when launched directly from operator_interface,
# while reusing the canonical solver rather than importing any runtime module.
_FLIGHT_CONTROL = Path(__file__).resolve().parents[2] / "定位演算法" / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))

from site_alignment import (  # noqa: E402
    AlignmentFit,
    ControlPoint,
    SiteAlignment,
    save_site_alignment,
    solve_rigid_alignment,
)
from pose_frame_chain import (  # noqa: E402
    CameraBodyExtrinsic,
    save_camera_body_extrinsic,
)


MIN_CONTROL_POINTS = 6
MIN_CALIBRATION_DISPLACEMENT_M = 0.20
RETURN_ORIGIN_TOLERANCE_M = 0.25
PHYSICAL_STICK_DEADZONE = 2000
MapPositionProvider = Callable[[], Iterable[float]]
CameraRotationProvider = Callable[[], Iterable[Iterable[float]]]
MotionObservationProvider = Callable[[], Mapping[str, object]]


# The navigation reference point is deliberately the camera optical centre.
# Its FRD axes follow the airframe, but its translation from the camera is zero.
_R_SITE_FROM_CAMERA_CENTER_FRD_AT_CALIBRATION = np.asarray(
    (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ),
    dtype=float,
)


def _frame_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class GuidedAlignmentStep:
    key: str
    label: str
    site_axis: tuple[int, int, int]
    pcmd_axis: int | None
    pcmd_sign: int
    shortcut: str


GUIDED_ALIGNMENT_STEPS = (
    GuidedAlignmentStep("origin", "原點懸停", (0, 0, 0), None, 0, ""),
    GuidedAlignmentStep("right", "向右", (1, 0, 0), 0, 1, "L"),
    GuidedAlignmentStep("left", "向左", (-1, 0, 0), 0, -1, "J"),
    GuidedAlignmentStep("forward", "向前", (0, 1, 0), 1, 1, "I"),
    GuidedAlignmentStep("back", "向後", (0, -1, 0), 1, -1, "K"),
    GuidedAlignmentStep("up", "向上", (0, 0, 1), 3, 1, "W"),
    GuidedAlignmentStep("down", "向下", (0, 0, -1), 3, -1, "S"),
)


@dataclass(frozen=True)
class MotionObservation:
    """One read-only host-command/flight-telemetry sample."""

    pcmd: tuple[int, int, int, int]
    pcmd_age_s: float | None
    pilot_sticks: bool
    stick_axes: tuple[tuple[int, int], ...]
    speed_north_mps: float | None
    speed_east_mps: float | None
    speed_down_mps: float | None
    yaw_rad: float | None
    gimbal_pitch_deg: float | None
    heading_ok: bool
    telemetry_age_s: float | None
    flight_state: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MotionObservation":
        raw_pcmd = value.get("pcmd")
        if not isinstance(raw_pcmd, (list, tuple)) or len(raw_pcmd) != 4:
            raise ValueError("motion observation requires a four-axis PCMD")

        def optional_number(name: str) -> float | None:
            raw = value.get(name)
            if raw is None:
                return None
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            return number

        pcmd = tuple(int(item) for item in raw_pcmd)
        raw_stick_axes = value.get("stick_axes") or {}
        if not isinstance(raw_stick_axes, Mapping):
            raise ValueError("stick_axes must be a mapping")
        stick_axes = tuple(
            sorted((int(axis), int(position)) for axis, position in raw_stick_axes.items())
        )
        heading = str(value.get("heading_state") or "").strip().lower()
        return cls(
            pcmd=(pcmd[0], pcmd[1], pcmd[2], pcmd[3]),
            pcmd_age_s=optional_number("pcmd_age_s"),
            pilot_sticks=bool(value.get("pilot_sticks")),
            stick_axes=stick_axes,
            speed_north_mps=optional_number("speed_north_mps"),
            speed_east_mps=optional_number("speed_east_mps"),
            speed_down_mps=optional_number("speed_down_mps"),
            yaw_rad=optional_number("yaw_rad"),
            gimbal_pitch_deg=optional_number("gimbal_pitch_deg"),
            heading_ok=heading == "ok",
            telemetry_age_s=optional_number("telemetry_age_s"),
            flight_state=str(value.get("flight_state") or "unknown").strip().lower(),
        )

    @property
    def physical_stick_active(self) -> bool:
        return any(abs(value) > PHYSICAL_STICK_DEADZONE for _axis, value in self.stick_axes)

    def body_velocity(self) -> tuple[float, float, float] | None:
        if None in (
            self.speed_north_mps,
            self.speed_east_mps,
            self.speed_down_mps,
            self.yaw_rad,
        ):
            return None
        yaw = float(self.yaw_rad)
        north = float(self.speed_north_mps)
        east = float(self.speed_east_mps)
        return (
            math.cos(yaw) * north + math.sin(yaw) * east,
            -math.sin(yaw) * north + math.cos(yaw) * east,
            -float(self.speed_down_mps),
        )


def motion_sample_matches_step(
    step: GuidedAlignmentStep, observation: MotionObservation
) -> bool:
    """Return whether one fresh command/telemetry sample proves this direction."""
    if step.pcmd_axis is None:
        return False
    if observation.telemetry_age_s is None or observation.telemetry_age_s > 0.5:
        return False
    velocity_axis = 1 if step.pcmd_axis == 0 else 0 if step.pcmd_axis == 1 else 2
    if velocity_axis != 2 and not observation.heading_ok:
        return False
    velocity = observation.body_velocity()
    if velocity is None:
        return False
    intended = velocity[velocity_axis] * step.pcmd_sign
    cross = max(
        abs(value) for index, value in enumerate(velocity) if index != velocity_axis
    )
    if intended < 0.05 or intended < cross:
        return False
    if observation.pilot_sticks:
        return observation.physical_stick_active
    if observation.pcmd_age_s is None or observation.pcmd_age_s > 0.25:
        return False
    intended_command = abs(observation.pcmd[step.pcmd_axis])
    if intended_command < 3 or observation.pcmd[step.pcmd_axis] * step.pcmd_sign <= 0:
        return False
    other_translation = max(
        abs(observation.pcmd[index])
        for index in (0, 1, 3)
        if index != step.pcmd_axis
    )
    return abs(observation.pcmd[2]) < 3 and other_translation * 2 <= intended_command


def _is_recent_computer_command(
    step: GuidedAlignmentStep, sample: MotionObservation
) -> bool:
    return (
        not sample.pilot_sticks
        and sample.pcmd_age_s is not None
        and sample.pcmd_age_s <= 0.25
        and abs(sample.pcmd[step.pcmd_axis]) >= 3
    )


def _has_recent_translation_command(sample: MotionObservation) -> bool:
    return (
        not sample.pilot_sticks
        and sample.pcmd_age_s is not None
        and sample.pcmd_age_s <= 0.25
        and any(abs(sample.pcmd[index]) >= 3 for index in (0, 1, 3))
    )


def _motion_command_sources(
    step: GuidedAlignmentStep,
    samples: tuple[MotionObservation, ...],
) -> tuple[list[MotionObservation], list[MotionObservation]]:
    computer_samples = [
        sample for sample in samples if _is_recent_computer_command(step, sample)
    ]
    physical_samples = [
        sample
        for sample in samples
        if sample.pilot_sticks and sample.physical_stick_active
    ]
    if computer_samples and physical_samples:
        raise ValueError("控制來源在步驟中切換；請回原點重做")
    unexpected_computer_motion = [
        sample
        for sample in samples
        if _has_recent_translation_command(sample)
        and sample not in computer_samples
    ]
    if computer_samples and unexpected_computer_motion:
        raise ValueError("移動期間收到非預期方向的 PCMD，取樣已拒絕")
    return computer_samples, physical_samples


def _validate_command_shape(
    step: GuidedAlignmentStep, sample: MotionObservation
) -> None:
    intended_command = abs(sample.pcmd[step.pcmd_axis])
    other_translation = max(
        abs(sample.pcmd[index])
        for index in (0, 1, 3)
        if index != step.pcmd_axis
    )
    if abs(sample.pcmd[2]) >= 3 or other_translation * 2 > intended_command:
        raise ValueError("移動期間收到旋轉或斜向 PCMD，取樣已拒絕")


def _matching_motion_commands(
    step: GuidedAlignmentStep,
    computer_samples: list[MotionObservation],
    physical_samples: list[MotionObservation],
) -> list[MotionObservation]:
    if physical_samples:
        return physical_samples
    if not computer_samples:
        raise ValueError(f"未讀到 {step.label} 的實體搖桿或電腦 PCMD")
    matching_commands = [
        sample
        for sample in computer_samples
        if sample.pcmd[step.pcmd_axis] * step.pcmd_sign > 0
    ]
    if len(matching_commands) * 2 < len(computer_samples):
        raise ValueError(f"收到的 PCMD 方向與「{step.label}」相反")
    for sample in matching_commands:
        _validate_command_shape(step, sample)
    return matching_commands


def _measured_motion(
    step: GuidedAlignmentStep,
    matching_commands: list[MotionObservation],
) -> tuple[int, list[tuple[float, float, float]]]:
    velocity_axis = 1 if step.pcmd_axis == 0 else 0 if step.pcmd_axis == 1 else 2
    measured: list[tuple[float, float, float]] = []
    for sample in matching_commands:
        if sample.telemetry_age_s is None or sample.telemetry_age_s > 0.5:
            continue
        if velocity_axis != 2 and not sample.heading_ok:
            continue
        body_velocity = sample.body_velocity()
        if body_velocity is not None:
            measured.append(body_velocity)
    return velocity_axis, measured


def _validate_motion_attitude(
    matching_commands: list[MotionObservation],
) -> None:
    yaw_values = [
        float(sample.yaw_rad)
        for sample in matching_commands
        if sample.yaw_rad is not None
        and sample.telemetry_age_s is not None
        and sample.telemetry_age_s <= 0.5
    ]
    if yaw_values:
        yaw_origin = yaw_values[0]
        yaw_span = max(
            abs((value - yaw_origin + math.pi) % (2.0 * math.pi) - math.pi)
            for value in yaw_values
        )
        if math.degrees(yaw_span) > 8.0:
            raise ValueError("yaw 在校正移動中改變超過 8°")
    gimbal_values = [
        float(sample.gimbal_pitch_deg)
        for sample in matching_commands
        if sample.gimbal_pitch_deg is not None
    ]
    if gimbal_values and max(gimbal_values) - min(gimbal_values) > 1.0:
        raise ValueError("gimbal pitch 在校正移動中改變超過 1°")


def _validate_motion_direction(
    step: GuidedAlignmentStep,
    velocity_axis: int,
    measured: list[tuple[float, float, float]],
) -> None:
    correct = 0
    wrong = 0
    off_axis = 0
    for velocity in measured:
        intended = velocity[velocity_axis] * step.pcmd_sign
        cross = max(
            abs(value) for index, value in enumerate(velocity) if index != velocity_axis
        )
        if intended >= 0.05 and intended >= cross:
            correct += 1
        elif intended <= -0.05:
            wrong += 1
        elif cross >= 0.05:
            off_axis += 1
    if correct < 2 or correct <= wrong + off_axis:
        raise ValueError(
            f"飛控速度顯示機體沒有穩定{step.label}，取樣已拒絕"
        )


def validate_motion_evidence(
    step: GuidedAlignmentStep,
    observations: Iterable[MotionObservation],
) -> None:
    """Reject a labelled move unless command and measured motion agree."""
    samples = tuple(observations)
    if step.pcmd_axis is None:
        return
    computer_samples, physical_samples = _motion_command_sources(step, samples)
    matching_commands = _matching_motion_commands(
        step, computer_samples, physical_samples
    )
    velocity_axis, measured = _measured_motion(step, matching_commands)
    if not measured:
        reason = (
            "航向未鎖定或水平速度遙測過期"
            if velocity_axis != 2
            else "垂直速度遙測過期"
        )
        raise ValueError(f"無法驗證實際{step.label}：{reason}")
    _validate_motion_attitude(matching_commands)
    _validate_motion_direction(step, velocity_axis, measured)


def validate_hover_reference(observation: MotionObservation) -> None:
    """Require a fresh, airborne, stationary origin attitude sample."""
    if observation.flight_state not in {"hovering", "flying"}:
        raise ValueError("請先起飛並穩定懸停")
    if observation.pilot_sticks and observation.physical_stick_active:
        raise ValueError("實體搖桿尚未回中")
    if observation.telemetry_age_s is None or observation.telemetry_age_s > 0.5:
        raise ValueError("飛控遙測過期")
    if not observation.heading_ok or observation.yaw_rad is None:
        raise ValueError("航向尚未鎖定")
    if observation.gimbal_pitch_deg is None:
        raise ValueError("無法讀取 gimbal pitch")
    if (
        not observation.pilot_sticks
        and observation.pcmd_age_s is not None
        and observation.pcmd_age_s <= 0.25
        and any(abs(value) >= 3 for value in observation.pcmd)
    ):
        raise ValueError("飛機仍收到移動或旋轉 PCMD")
    velocity = observation.body_velocity()
    if velocity is None:
        raise ValueError("無法讀取三軸速度")
    if math.hypot(velocity[0], velocity[1]) > 0.20 or abs(velocity[2]) > 0.15:
        raise ValueError("飛機尚未穩定懸停")


class _GuidedSiteAlignmentMixin:
    """Ordered field calibration with command and motion consistency gates."""

    def __init__(
        self,
        *,
        map_frame_id: str,
        site_frame_id: str,
        get_current_map_position: MapPositionProvider,
        get_current_camera_rotation: CameraRotationProvider | None = None,
    ) -> None:
        super().__init__(
            map_frame_id=map_frame_id,
            site_frame_id=site_frame_id,
            get_current_map_position=get_current_map_position,
        )
        if get_current_camera_rotation is not None and not callable(
            get_current_camera_rotation
        ):
            raise TypeError("get_current_camera_rotation must be callable")
        self.get_current_camera_rotation = get_current_camera_rotation
        self._next_step_index = 0
        self._yaw_by_step: list[float] = []
        self._gimbal_pitch_by_step: list[float] = []
        self._camera_rotations: list[np.ndarray] = []

    @property
    def current_step(self) -> GuidedAlignmentStep | None:
        if self._next_step_index >= len(GUIDED_ALIGNMENT_STEPS):
            return None
        return GUIDED_ALIGNMENT_STEPS[self._next_step_index]

    @property
    def complete(self) -> bool:
        return self.current_step is None

    def _site_point(self, step: GuidedAlignmentStep) -> tuple[float, float, float]:
        return tuple(float(value) for value in step.site_axis)

    def _require_step(self, key: str) -> GuidedAlignmentStep:
        step = self.current_step
        if step is None:
            raise ValueError("所有現場取樣步驟都已完成")
        if key != step.key:
            raise ValueError(f"目前應執行「{step.label}」，不是 {key}")
        return step

    def _capture_camera_rotation(self) -> None:
        provider = self.get_current_camera_rotation
        if provider is None:
            return
        rotation = np.asarray(provider(), dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("定位相機旋轉必須是有限 3x3 矩陣")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=0.05):
            raise ValueError("定位相機旋轉不是正交矩陣")
        if float(np.linalg.det(rotation)) < 0.0:
            raise ValueError("定位相機旋轉出現鏡射（det < 0）")
        u, _singular, vt = np.linalg.svd(rotation)
        proper = u @ vt
        if float(np.linalg.det(proper)) < 0.0:
            u[:, -1] *= -1.0
            proper = u @ vt
        self._camera_rotations.append(proper)

    def _record_attitude(self, observations: Iterable[MotionObservation]) -> None:
        samples = tuple(
            sample
            for sample in observations
            if sample.telemetry_age_s is not None and sample.telemetry_age_s <= 0.5
        )
        yaw_values = [
            float(sample.yaw_rad) for sample in samples if sample.yaw_rad is not None
        ]
        gimbal_values = [
            float(sample.gimbal_pitch_deg)
            for sample in samples
            if sample.gimbal_pitch_deg is not None
        ]
        if yaw_values:
            self._yaw_by_step.append(sorted(yaw_values)[len(yaw_values) // 2])
        if gimbal_values:
            self._gimbal_pitch_by_step.append(
                sorted(gimbal_values)[len(gimbal_values) // 2]
            )

    def capture_origin(
        self, observation: MotionObservation | None = None,
    ) -> ControlPoint:
        step = self._require_step("origin")
        if observation is not None:
            validate_hover_reference(observation)
        point = self.capture_current_map_position(self._site_point(step), step.label)
        self._capture_camera_rotation()
        if observation is not None:
            self._record_attitude((observation,))
        self._next_step_index += 1
        return point

    def require_return_to_origin(self, map_point: Iterable[float]) -> None:
        if self.point_count < 1:
            return
        origin = tuple(self.control_points[0].map_point)
        if (
            math.dist(origin, tuple(float(value) for value in map_point))
            > RETURN_ORIGIN_TOLERANCE_M
        ):
            raise ValueError("尚未回到原點附近（誤差需小於 0.25 m）")

    def finish_step(
        self,
        key: str,
        observations: Iterable[MotionObservation],
    ) -> ControlPoint:
        step = self._require_step(key)
        if step.pcmd_axis is None:
            raise ValueError("原點請使用 capture_origin")
        samples = tuple(observations)
        validate_motion_evidence(step, samples)
        self._record_attitude(samples)
        if self.get_current_map_position is None:
            raise RuntimeError("get_current_map_position callback is not configured")
        map_point = tuple(float(value) for value in self.get_current_map_position())
        origin = self.control_points[0].map_point
        displacement = math.dist(origin, map_point)
        if displacement < MIN_CALIBRATION_DISPLACEMENT_M:
            raise ValueError(
                f"定位位移只有 {displacement:.3f} m；"
                f"至少需要 {MIN_CALIBRATION_DISPLACEMENT_M:.2f} m"
            )
        point = self.add_control_point(map_point, self._site_point(step), step.label)
        self._capture_camera_rotation()
        self._next_step_index += 1
        return point

    @staticmethod
    def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        relative = a.T @ b
        cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) * 0.5))
        return math.degrees(math.acos(cosine))

    def _camera_from_navigation_rotation(self) -> np.ndarray:
        fit = self.fit
        if fit is None:
            raise ValueError("請先完成場域座標解算")
        if len(self._camera_rotations) != len(GUIDED_ALIGNMENT_STEPS):
            raise ValueError("缺少七個定位相機姿態，不能解鎖 AUTO")
        map_from_site = fit.transform.R.T
        candidates = [
            rotation_camera_from_map
            @ map_from_site
            @ _R_SITE_FROM_CAMERA_CENTER_FRD_AT_CALIBRATION
            for rotation_camera_from_map in self._camera_rotations
        ]
        u, _singular, vt = np.linalg.svd(sum(candidates))
        mean_rotation = u @ vt
        if float(np.linalg.det(mean_rotation)) < 0.0:
            u[:, -1] *= -1.0
            mean_rotation = u @ vt
        maximum_error = max(
            self._rotation_angle_deg(mean_rotation, candidate)
            for candidate in candidates
        )
        if maximum_error > 5.0:
            raise ValueError(
                f"相機／機體方向在各取樣點差異 {maximum_error:.1f}°，"
                "超過 5°，不能解鎖 AUTO"
            )
        return mean_rotation

    def camera_center_extrinsic(
        self,
        *,
        vehicle_id: str,
        body_frame_id: str,
        camera_frame_id: str,
    ) -> CameraBodyExtrinsic:
        """Build the approved camera-centred FRD reference used by AUTO."""
        if not self.approved:
            raise PermissionError("請先人工核准場域校正")
        if not self._gimbal_pitch_by_step:
            raise ValueError("缺少固定 gimbal pitch 證據")
        pitch = sorted(self._gimbal_pitch_by_step)[
            len(self._gimbal_pitch_by_step) // 2
        ]
        rotation = self._camera_from_navigation_rotation()
        return CameraBodyExtrinsic(
            vehicle_id=vehicle_id,
            body_frame_id=body_frame_id,
            camera_frame_id=camera_frame_id,
            rotation_camera_from_body=tuple(
                tuple(float(value) for value in row) for row in rotation
            ),
            # Operator policy: the camera optical centre is the navigation origin.
            translation_camera_from_body_m=(0.0, 0.0, 0.0),
            fixed_gimbal_pitch_deg=pitch,
            gimbal_pitch_tolerance_deg=1.0,
            evidence=(
                "approved on-site seven-point motion calibration; "
                "navigation origin is the camera optical centre; "
                "T_camera_from_navigation translation fixed to zero by operator policy"
            ),
            approved=True,
        )

    def _validate_recorded_attitude(self) -> None:
        if self._yaw_by_step:
            yaw_origin = self._yaw_by_step[0]
            yaw_span = max(
                abs((value - yaw_origin + math.pi) % (2.0 * math.pi) - math.pi)
                for value in self._yaw_by_step
            )
            if math.degrees(yaw_span) > 8.0:
                raise ValueError("各步驟的 yaw 差異超過 8°")
        if (
            self._gimbal_pitch_by_step
            and max(self._gimbal_pitch_by_step) - min(self._gimbal_pitch_by_step) > 1.0
        ):
            raise ValueError("各步驟的 gimbal pitch 差異超過 1°")

    def _measured_control_points(self) -> list[ControlPoint]:
        control_points = self.control_points
        origin = np.asarray(control_points[0].map_point, dtype=float)
        measured_controls = [control_points[0]]
        for step, point in zip(
            GUIDED_ALIGNMENT_STEPS[1:], control_points[1:]
        ):
            displacement = float(
                np.linalg.norm(np.asarray(point.map_point, dtype=float) - origin)
            )
            if displacement < MIN_CALIBRATION_DISPLACEMENT_M:
                raise ValueError(
                    f"{step.label}定位位移 {displacement:.3f} m 太小；"
                    f"至少需要 {MIN_CALIBRATION_DISPLACEMENT_M:.2f} m"
                )
            measured_controls.append(
                ControlPoint(
                    point.map_point,
                    tuple(displacement * value for value in step.site_axis),
                    point.label,
                )
            )

        return measured_controls

    def _validate_axis_geometry(self) -> None:
        vectors = {
            point.label: tuple(
                float(value) - float(origin)
                for value, origin in zip(
                    point.map_point, self.control_points[0].map_point
                )
            )
            for point in self.control_points[1:]
        }

        def unit(label: str) -> tuple[float, float, float]:
            vector = vectors[label]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm <= 1e-9:
                raise ValueError(f"{label} 定位位移過小")
            return tuple(value / norm for value in vector)

        right, left = unit("向右"), unit("向左")
        forward, back = unit("向前"), unit("向後")
        up, down = unit("向上"), unit("向下")

        def dot(a, b) -> float:
            return sum(x * y for x, y in zip(a, b))

        if dot(right, left) > -0.70:
            raise ValueError("左／右位移不互為反向")
        if dot(forward, back) > -0.70:
            raise ValueError("前／後位移不互為反向")
        if dot(up, down) > -0.70:
            raise ValueError("上／下位移不互為反向")
        if abs(dot(right, forward)) > 0.50:
            raise ValueError("右與前位移夾角不足，無法建立座標軸")
        cross = (
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        )
        cross_norm = math.sqrt(sum(value * value for value in cross))
        cross = tuple(value / cross_norm for value in cross)
        if dot(cross, up) < 0.70:
            raise ValueError("右×前與上方向不符合右手座標系")

    def solve(self) -> AlignmentFit:
        if not self.complete:
            step = self.current_step
            assert step is not None
            raise ValueError(f"尚未完成「{step.label}」取樣")
        self._validate_recorded_attitude()
        self._control_points = self._measured_control_points()
        fit = super().solve()
        if fit.transform.scale != 1.0:
            raise AssertionError("rigid site alignment must keep scale fixed to one")
        self._validate_axis_geometry()
        quality_limit = 0.25
        if fit.quality.max_error_m > quality_limit:
            raise ValueError(
                f"校正殘差 {fit.quality.max_error_m:.3f} m 超過 "
                f"{quality_limit:.3f} m"
            )
        if self.get_current_camera_rotation is not None:
            self._camera_from_navigation_rotation()
        return fit


class ManualSiteAlignmentModel:
    """Collect, solve, approve, and persist a manual site alignment.

    ``get_current_map_position`` is the sole live-data seam.  It is called only
    by :meth:`capture_current_map_position`, and its return value is immediately
    copied into an immutable :class:`ControlPoint`.
    """

    def __init__(
        self,
        *,
        map_frame_id: str,
        site_frame_id: str,
        get_current_map_position: MapPositionProvider | None = None,
    ) -> None:
        self.map_frame_id = _frame_id(map_frame_id, "map_frame_id")
        self.site_frame_id = _frame_id(site_frame_id, "site_frame_id")
        if get_current_map_position is not None and not callable(get_current_map_position):
            raise TypeError("get_current_map_position must be callable")
        self.get_current_map_position = get_current_map_position
        self._control_points: list[ControlPoint] = []
        self._fit: AlignmentFit | None = None
        self._approved = False

    @property
    def control_points(self) -> tuple[ControlPoint, ...]:
        return tuple(self._control_points)

    @property
    def point_count(self) -> int:
        return len(self._control_points)

    @property
    def fit(self) -> AlignmentFit | None:
        return self._fit

    @property
    def approved(self) -> bool:
        return self._approved

    @property
    def can_write(self) -> bool:
        return (
            self.point_count >= MIN_CONTROL_POINTS
            and self._fit is not None
            and self._approved
        )

    def _invalidate_result(self) -> None:
        self._fit = None
        self._approved = False

    def add_control_point(
        self,
        map_point: Iterable[float],
        site_point: Iterable[float],
        label: str | None = None,
    ) -> ControlPoint:
        point = ControlPoint(tuple(map_point), tuple(site_point), label)
        self._control_points.append(point)
        self._invalidate_result()
        return point

    def capture_current_map_position(
        self,
        site_point: Iterable[float],
        label: str | None = None,
    ) -> ControlPoint:
        """Capture the provider's current map XYZ paired with site/world XYZ."""
        if self.get_current_map_position is None:
            raise RuntimeError("get_current_map_position callback is not configured")
        return self.add_control_point(
            self.get_current_map_position(),
            site_point,
            label,
        )

    def remove_control_point(self, index: int) -> ControlPoint:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("control point index must be an integer")
        try:
            point = self._control_points.pop(index)
        except IndexError as exc:
            raise IndexError("control point index is out of range") from exc
        self._invalidate_result()
        return point

    def _require_minimum_points(self) -> None:
        if self.point_count < MIN_CONTROL_POINTS:
            raise ValueError(
                f"alignment requires at least {MIN_CONTROL_POINTS} control points"
            )

    def solve(self) -> AlignmentFit:
        """Solve a fixed-scale rigid transform for the current control points."""
        self._invalidate_result()
        self._require_minimum_points()
        self._fit = solve_rigid_alignment(
            (point.map_point for point in self._control_points),
            (point.site_point for point in self._control_points),
            labels=(point.label for point in self._control_points),
        )
        return self._fit

    def approve(self) -> None:
        """Record the operator's explicit approval for the current solved fit."""
        if self._fit is None:
            raise ValueError("solve the alignment before approval")
        self._require_minimum_points()
        self._approved = True

    def write(self, path: str | Path) -> Path:
        """Atomically write an approved ``sfm-site-alignment/v1`` document."""
        self._require_minimum_points()
        if self._fit is None:
            raise ValueError("solve the alignment before writing")
        if not self._approved:
            raise PermissionError("explicit operator approval is required before writing")
        alignment = SiteAlignment.from_fit(
            map_frame_id=self.map_frame_id,
            site_frame_id=self.site_frame_id,
            fit=self._fit,
            approved=True,
        )
        return Path(save_site_alignment(alignment, path))


class GuidedSiteAlignmentModel(
    _GuidedSiteAlignmentMixin,
    ManualSiteAlignmentModel,
):
    """Concrete guided model; the mixin is defined before the manual base."""


def _format_point(point: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):.6f}" for value in point) + ")"


class SiteAlignmentDialog(tk.Toplevel):
    """Small manual-control-point dialog backed by ``ManualSiteAlignmentModel``."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        map_frame_id: str,
        site_frame_id: str,
        get_current_map_position: MapPositionProvider | None,
        output_path: str | Path | None = None,
        on_saved: Callable[[Path], None] | None = None,
    ) -> None:
        self.model = ManualSiteAlignmentModel(
            map_frame_id=map_frame_id,
            site_frame_id=site_frame_id,
            get_current_map_position=get_current_map_position,
        )
        super().__init__(parent)
        self.title("Manual Site Alignment")
        self.minsize(680, 460)
        self.output_path = Path(output_path) if output_path is not None else None
        self.on_saved = on_saved

        self._map_position_var = tk.StringVar(value="not captured")
        self._status_var = tk.StringVar(value="Add six or more measured control points.")
        self._quality_var = tk.StringVar(value="No solved alignment")
        self._site_vars = [tk.StringVar() for _ in range(3)]
        self._label_var = tk.StringVar()
        self._build()
        self._refresh_points()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self) -> None:
        body = ttk.Frame(self, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        ttk.Label(
            body,
            text=(
                f"Map frame: {self.model.map_frame_id}    "
                f"Site/world frame: {self.model.site_frame_id}"
            ),
        ).grid(row=0, column=0, sticky="w")

        entry_frame = ttk.LabelFrame(body, text="Measured control point")
        entry_frame.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        entry_frame.columnconfigure(1, weight=1)
        ttk.Label(entry_frame, text="Current map XYZ").grid(
            row=0, column=0, padx=8, pady=5, sticky="w"
        )
        ttk.Label(entry_frame, textvariable=self._map_position_var).grid(
            row=0, column=1, padx=8, pady=5, sticky="w"
        )
        ttk.Label(entry_frame, text="Site/world XYZ").grid(
            row=1, column=0, padx=8, pady=5, sticky="w"
        )
        site_entry = ttk.Frame(entry_frame)
        site_entry.grid(row=1, column=1, padx=8, pady=5, sticky="ew")
        for axis, variable in enumerate(self._site_vars):
            ttk.Entry(site_entry, textvariable=variable, width=12).grid(
                row=0, column=axis, padx=(0, 5)
            )
        ttk.Label(entry_frame, text="Label (optional)").grid(
            row=2, column=0, padx=8, pady=5, sticky="w"
        )
        ttk.Entry(entry_frame, textvariable=self._label_var, width=24).grid(
            row=2, column=1, padx=8, pady=5, sticky="w"
        )
        self._capture_button = ttk.Button(
            entry_frame,
            text="Capture current map XYZ + add",
            command=self._capture_and_add,
            state="normal" if self.model.get_current_map_position is not None else "disabled",
        )
        self._capture_button.grid(row=3, column=0, columnspan=2, padx=8, pady=(5, 8), sticky="w")

        list_frame = ttk.LabelFrame(body, text="Control points")
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self._points_tree = ttk.Treeview(
            list_frame,
            columns=("number", "label", "map", "site"),
            show="headings",
            selectmode="browse",
        )
        headings = {
            "number": "#",
            "label": "Label",
            "map": "Map XYZ",
            "site": "Site/world XYZ",
        }
        for column_name, heading in headings.items():
            self._points_tree.heading(column_name, text=heading)
        self._points_tree.column("number", width=40, anchor="center")
        self._points_tree.column("label", width=110)
        self._points_tree.column("map", width=210)
        self._points_tree.column("site", width=210)
        self._points_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._points_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._points_tree.configure(yscrollcommand=scrollbar.set)

        actions = ttk.Frame(body)
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Remove selected", command=self._remove_selected).pack(
            side="left"
        )
        self._solve_button = ttk.Button(actions, text="Solve", command=self._solve)
        self._solve_button.pack(side="left", padx=(8, 0))
        self._approve_button = ttk.Button(
            actions,
            text="Approve alignment for write",
            command=self._approve,
        )
        self._approve_button.pack(side="left", padx=(8, 0))
        self._save_button = ttk.Button(actions, text="Write alignment", command=self._save)
        self._save_button.pack(side="right")

        ttk.Label(body, textvariable=self._quality_var).grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(body, textvariable=self._status_var, wraplength=640).grid(
            row=5, column=0, sticky="w", pady=(4, 0)
        )

    def _parse_site_point(self) -> tuple[float, float, float]:
        values = [float(variable.get().strip()) for variable in self._site_vars]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("site/world XYZ must be finite numbers")
        return values[0], values[1], values[2]

    def _capture_and_add(self) -> None:
        try:
            point = self.model.capture_current_map_position(
                self._parse_site_point(),
                self._label_var.get().strip() or None,
            )
        except Exception as exc:
            self._status_var.set(f"Cannot add control point: {exc}")
            return
        self._map_position_var.set(_format_point(point.map_point))
        for variable in self._site_vars:
            variable.set("")
        self._label_var.set("")
        self._status_var.set(f"Captured control point {self.model.point_count}.")
        self._quality_var.set("No solved alignment")
        self._refresh_points()

    def _refresh_points(self) -> None:
        for item in self._points_tree.get_children():
            self._points_tree.delete(item)
        for index, point in enumerate(self.model.control_points):
            self._points_tree.insert(
                "",
                "end",
                iid=f"point-{index}",
                values=(
                    index + 1,
                    point.label or "",
                    _format_point(point.map_point),
                    _format_point(point.site_point),
                ),
            )
        fit = self.model.fit
        self._solve_button.configure(
            state="normal" if self.model.point_count >= MIN_CONTROL_POINTS else "disabled"
        )
        self._approve_button.configure(
            state="normal" if fit is not None and not self.model.approved else "disabled"
        )
        self._save_button.configure(state="normal" if self.model.can_write else "disabled")

    def _remove_selected(self) -> None:
        selection = self._points_tree.selection()
        if not selection:
            self._status_var.set("Select a control point to remove.")
            return
        try:
            index = int(selection[0].removeprefix("point-"))
            self.model.remove_control_point(index)
        except (IndexError, TypeError, ValueError) as exc:
            self._status_var.set(f"Cannot remove control point: {exc}")
            return
        self._quality_var.set("No solved alignment")
        self._status_var.set("Control point removed; solve again before approval.")
        self._refresh_points()

    def _solve(self) -> None:
        try:
            fit = self.model.solve()
        except ValueError as exc:
            self._quality_var.set("No solved alignment")
            self._status_var.set(f"Cannot solve alignment: {exc}")
            self._refresh_points()
            return
        self._quality_var.set(
            f"RMSE: {fit.quality.rmse_m:.6f} m    "
            f"Max residual: {fit.quality.max_error_m:.6f} m"
        )
        self._status_var.set("Fit ready. Review the residuals, then explicitly approve it.")
        self._refresh_points()

    def _approve(self) -> None:
        try:
            self.model.approve()
        except ValueError as exc:
            self._status_var.set(f"Cannot approve alignment: {exc}")
            return
        self._status_var.set("Alignment approved. Writing is now enabled.")
        self._refresh_points()

    def _save(self) -> None:
        target = self.output_path
        if target is None:
            chosen = filedialog.asksaveasfilename(
                parent=self,
                title="Write site alignment",
                defaultextension=".json",
                filetypes=(("JSON", "*.json"), ("All files", "*")),
            )
            if not chosen:
                return
            target = Path(chosen)
        try:
            saved = self.model.write(target)
        except (OSError, PermissionError, ValueError) as exc:
            self._status_var.set(f"Cannot write alignment: {exc}")
            return
        self.output_path = saved
        self._status_var.set(f"Wrote approved alignment: {saved}")
        if self.on_saved is not None:
            self.on_saved(saved)
        messagebox.showinfo("Site alignment", f"Wrote approved alignment to:\n{saved}", parent=self)


class GuidedSiteAlignmentDialog(tk.Toplevel):
    """Non-modal, read-only flight observer for an operator-flown alignment."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        map_frame_id: str,
        site_frame_id: str,
        get_current_map_position: MapPositionProvider,
        get_current_camera_rotation: CameraRotationProvider,
        get_motion_observation: MotionObservationProvider,
        output_path: str | Path,
        extrinsic_output_path: str | Path,
        vehicle_id: str,
        body_frame_id: str,
        camera_frame_id: str,
        on_saved: Callable[[Path, Path], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.model = GuidedSiteAlignmentModel(
            map_frame_id=map_frame_id,
            site_frame_id=site_frame_id,
            get_current_map_position=get_current_map_position,
            get_current_camera_rotation=get_current_camera_rotation,
        )
        self.get_motion_observation = get_motion_observation
        self.output_path = Path(output_path)
        self.extrinsic_output_path = Path(extrinsic_output_path)
        self.vehicle_id = _frame_id(vehicle_id, "vehicle_id")
        self.body_frame_id = _frame_id(body_frame_id, "body_frame_id")
        self.camera_frame_id = _frame_id(camera_frame_id, "camera_frame_id")
        self.on_saved = on_saved
        self.on_close = on_close
        self._auto_monitoring = False
        self._collecting = False
        self._observations: list[MotionObservation] = []
        self._active_step: GuidedAlignmentStep | None = None
        self._awaiting_origin_return = False
        self._hover_sample_count = 0
        super().__init__(parent)
        self.title("現場座標校正（只讀飛行證據）")
        self.minsize(760, 520)
        self._instruction_var = tk.StringVar()
        self._status_var = tk.StringVar(
            value="本視窗不會送出起飛或移動指令。"
        )
        self._quality_var = tk.StringVar(value="尚未解算")
        self._build()
        self._refresh()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._sample_motion)

    def _build(self) -> None:
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text=(
                "先按主介面「開始定位」並手動起飛懸停。機頭對準場域前方，"
                "固定 yaw 與 gimbal，再按下方「開始自動辨識」。"
            ),
            wraplength=720,
            justify="left",
            font=("Sans", 10, "bold"),
        ).pack(anchor="w", fill="x")
        ttk.Label(
            body,
            text=(
                "可使用實體搖桿、鍵盤或介面虛擬搖桿；每次依提示從原點移動，"
                "放開並懸停後會自動取樣。距離不用量，但至少移動 0.20 m。"
            ),
            wraplength=720,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(4, 8))
        step_box = ttk.LabelFrame(body, text="目前步驟")
        step_box.pack(fill="x")
        ttk.Label(
            step_box,
            textvariable=self._instruction_var,
            wraplength=700,
            font=("Sans", 11, "bold"),
        ).pack(anchor="w", fill="x", padx=8, pady=6)
        action_row = ttk.Frame(step_box)
        action_row.pack(fill="x", padx=6, pady=(0, 6))
        self._begin_button = ttk.Button(
            action_row, text="開始自動辨識", command=self._toggle_auto_monitoring
        )
        self._begin_button.pack(side="left", padx=2)

        self._points_tree = ttk.Treeview(
            body,
            columns=("number", "label", "map", "site"),
            show="headings",
            height=8,
        )
        for name, label, width in (
            ("number", "#", 35),
            ("label", "方向", 90),
            ("map", "Map XYZ", 250),
            ("site", "校正後場域 XYZ (m)", 250),
        ):
            self._points_tree.heading(name, text=label)
            self._points_tree.column(name, width=width)
        self._points_tree.pack(fill="both", expand=True, pady=8)

        result_row = ttk.Frame(body)
        result_row.pack(fill="x")
        self._solve_button = ttk.Button(
            result_row, text="解算固定尺度座標", command=self._solve, state="disabled"
        )
        self._solve_button.pack(side="left", padx=2)
        self._approve_button = ttk.Button(
            result_row, text="人工核准", command=self._approve, state="disabled"
        )
        self._approve_button.pack(side="left", padx=2)
        self._save_button = ttk.Button(
            result_row,
            text="儲存校正並檢查 AUTO",
            command=self._save,
            state="disabled",
        )
        self._save_button.pack(side="left", padx=2)
        ttk.Label(result_row, textvariable=self._quality_var).pack(
            side="left", padx=10
        )
        ttk.Label(
            body,
            textvariable=self._status_var,
            wraplength=720,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(7, 0))

    def _step_instruction(self, step: GuidedAlignmentStep | None) -> str:
        if step is None:
            return "七個取樣點完成，請解算並檢查方向。"
        if step.key == "origin":
            return "1/7：起飛後在安全高度穩定懸停，系統會自動取原點。"
        index = GUIDED_ALIGNMENT_STEPS.index(step) + 1
        if self._awaiting_origin_return:
            return f"{index}/7：請先回到原點附近並穩定懸停。"
        return (
            f"{index}/7：請讓飛機{step.label}至少 0.20 m，可用實體搖桿或"
            f"鍵盤 {step.shortcut}；放開並懸停後會自動取樣。"
        )

    def _refresh(self) -> None:
        for item in self._points_tree.get_children():
            self._points_tree.delete(item)
        for index, point in enumerate(self.model.control_points, start=1):
            site_value = (
                _format_point(point.site_point)
                if self.model.fit is not None or index == 1
                else "待固定尺度解算"
            )
            self._points_tree.insert(
                "",
                "end",
                values=(
                    index,
                    point.label or "",
                    _format_point(point.map_point),
                    site_value,
                ),
            )
        step = self.model.current_step
        self._instruction_var.set(self._step_instruction(step))
        self._begin_button.configure(
            text="停止自動辨識" if self._auto_monitoring else "開始自動辨識",
            state="disabled" if step is None else "normal",
        )
        self._solve_button.configure(
            state="normal"
            if self.model.complete and not self._auto_monitoring
            else "disabled"
        )
        self._approve_button.configure(
            state="normal"
            if self.model.fit is not None and not self.model.approved
            else "disabled"
        )
        self._save_button.configure(
            state="normal" if self.model.can_write else "disabled"
        )

    def _toggle_auto_monitoring(self) -> None:
        if self._auto_monitoring:
            if self._collecting:
                self._awaiting_origin_return = True
            self._auto_monitoring = False
            self._collecting = False
            self._active_step = None
            self._observations.clear()
            self._hover_sample_count = 0
            self._status_var.set("已停止自動辨識；尚未完成的方向不會被採用。")
            self._refresh()
            return
        step = self.model.current_step
        if step is None:
            return
        self._auto_monitoring = True
        self._collecting = False
        self._active_step = None
        self._observations.clear()
        self._hover_sample_count = 0
        self._status_var.set(
            "已開始被動監看；不會送出任何飛行指令。請依目前步驟操作。"
        )
        self._refresh()

    def _sample_motion(self) -> None:
        if not self.winfo_exists():
            return
        if self._auto_monitoring:
            try:
                observation = MotionObservation.from_mapping(
                    self.get_motion_observation()
                )
                self._process_auto_sample(observation)
            except Exception as exc:
                self._status_var.set(f"飛控證據讀取失敗：{exc}")
        self.after(50, self._sample_motion)

    def _process_origin_sample(self, observation: MotionObservation) -> None:
        try:
            validate_hover_reference(observation)
        except ValueError:
            self._hover_sample_count = 0
            return
        self._hover_sample_count += 1
        if self._hover_sample_count < 3:
            return
        try:
            point = self.model.capture_origin(observation)
        except Exception as exc:
            self._hover_sample_count = 0
            self._status_var.set(f"等待有效定位原點：{exc}")
            return
        self._hover_sample_count = 0
        self._status_var.set(
            f"已自動取樣原點：{_format_point(point.map_point)}。請依提示移動。"
        )
        self._refresh()

    def _process_origin_return_sample(
        self,
        step: GuidedAlignmentStep,
        observation: MotionObservation,
    ) -> None:
        try:
            validate_hover_reference(observation)
            assert self.model.get_current_map_position is not None
            self.model.require_return_to_origin(
                self.model.get_current_map_position()
            )
        except Exception:
            self._hover_sample_count = 0
            return
        self._hover_sample_count += 1
        if self._hover_sample_count < 3:
            return
        self._awaiting_origin_return = False
        self._hover_sample_count = 0
        self._status_var.set(f"已回到原點；請開始「{step.label}」。")
        self._refresh()

    def _process_matching_motion_sample(
        self,
        step: GuidedAlignmentStep,
        observation: MotionObservation,
    ) -> bool:
        if not motion_sample_matches_step(step, observation):
            return False
        if not self._collecting:
            self._collecting = True
            self._active_step = step
            self._observations.clear()
            self._status_var.set(
                f"已辨識「{step.label}」；放開控制並穩定懸停即可自動取樣。"
            )
        self._observations.append(observation)
        self._hover_sample_count = 0
        return True

    def _process_auto_sample(self, observation: MotionObservation) -> None:
        step = self.model.current_step
        if step is None:
            self._auto_monitoring = False
            self._refresh()
            return
        if step.key == "origin":
            self._process_origin_sample(observation)
            return

        if self._awaiting_origin_return:
            self._process_origin_return_sample(step, observation)
            return

        if self._process_matching_motion_sample(step, observation):
            return
        if not self._collecting:
            return
        self._observations.append(observation)
        try:
            validate_hover_reference(observation)
        except ValueError:
            self._hover_sample_count = 0
            return
        self._hover_sample_count += 1
        if self._hover_sample_count >= 3:
            self._finish_step()

    def _finish_step(self) -> None:
        step = self._active_step
        self._collecting = False
        self._active_step = None
        self._hover_sample_count = 0
        if step is None:
            self._refresh()
            return
        try:
            point = self.model.finish_step(step.key, tuple(self._observations))
        except Exception as exc:
            self._status_var.set(f"「{step.label}」取樣被拒絕：{exc}")
            self._awaiting_origin_return = True
            self._refresh()
            return
        if self.model.complete:
            self._auto_monitoring = False
            self._status_var.set(
                "七個方向已自動取樣完成；請按「解算固定尺度座標」。"
            )
        else:
            self._awaiting_origin_return = True
            self._status_var.set(
                f"已驗證控制、飛控速度與定位位移：{_format_point(point.map_point)}。"
                "請回到原點。"
            )
        self._refresh()

    def _solve(self) -> None:
        try:
            fit = self.model.solve()
        except Exception as exc:
            self._status_var.set(f"方向／解算檢查失敗：{exc}")
            self._quality_var.set("未通過")
            self._refresh()
            return
        self._quality_var.set(
            f"剛體校正｜RMSE {fit.quality.rmse_m:.3f} m｜"
            f"最大殘差 {fit.quality.max_error_m:.3f} m"
        )
        self._status_var.set(
            "固定尺度、方向、右手性與殘差通過。請人工核對後核准。"
        )
        self._refresh()

    def _approve(self) -> None:
        try:
            self.model.approve()
        except Exception as exc:
            self._status_var.set(f"無法核准：{exc}")
            return
        self._status_var.set("已人工核准，可儲存並進行 AUTO 解鎖檢查。")
        self._refresh()

    def _save(self) -> None:
        try:
            saved = self.model.write(self.output_path)
            extrinsic = self.model.camera_center_extrinsic(
                vehicle_id=self.vehicle_id,
                body_frame_id=self.body_frame_id,
                camera_frame_id=self.camera_frame_id,
            )
            saved_extrinsic = save_camera_body_extrinsic(
                extrinsic, self.extrinsic_output_path
            )
            if self.on_saved is not None:
                self.on_saved(saved, saved_extrinsic)
        except Exception as exc:
            self._status_var.set(
                f"無法儲存或啟用校正：{exc}。AUTO 保持鎖定；手動控制仍可用。"
            )
            return
        self._status_var.set(
            "校正已綁定目前場域與飛機，AUTO 已解鎖；可按「自動飛行」。"
            "實體搖桿仍可立即接管。"
        )

    def _close(self) -> None:
        self._auto_monitoring = False
        self._collecting = False
        self.destroy()
        if self.on_close is not None:
            self.on_close()


__all__ = [
    "GUIDED_ALIGNMENT_STEPS",
    "GuidedAlignmentStep",
    "GuidedSiteAlignmentDialog",
    "GuidedSiteAlignmentModel",
    "MIN_CONTROL_POINTS",
    "ManualSiteAlignmentModel",
    "MapPositionProvider",
    "CameraRotationProvider",
    "MotionObservation",
    "MotionObservationProvider",
    "SiteAlignmentDialog",
    "validate_motion_evidence",
    "validate_hover_reference",
]
