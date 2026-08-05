"""Authoritative scale-free route direction and airframe-speed safety core.

Map coordinates provide direction only. No map-unit-to-metre conversion exists in
this module. Physical speed comes exclusively from fresh airframe telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleFreeConfig:
    speed_limit_mps: float = 0.30
    pose_max_age_s: float = 0.50
    speed_max_age_s: float = 0.50
    command_ttl_s: float = 0.15
    yaw_tolerance_deg: float = 3.0
    horizontal_axes: tuple[int, int] = (0, 1)
    vertical_axis: int = 2
    camera_to_body_yaw_deg: float = 0.0
    body_right_sign: float = -1.0

    def __post_init__(self) -> None:
        for name in (
            "speed_limit_mps",
            "pose_max_age_s",
            "speed_max_age_s",
            "command_ttl_s",
            "yaw_tolerance_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        axes = (*self.horizontal_axes, self.vertical_axis)
        if sorted(axes) != [0, 1, 2]:
            raise ValueError("horizontal_axes and vertical_axis must cover axes 0,1,2")
        if self.body_right_sign not in {-1.0, 1.0}:
            raise ValueError("body_right_sign must be -1.0 or 1.0")


@dataclass(frozen=True)
class ScaleFreeSample:
    position_map: tuple[float, float, float]
    target_map: tuple[float, float, float]
    camera_yaw_rad: float
    pose_mono_ns: int
    localization_state: str
    airframe_horizontal_speed_mps: float | None
    speed_mono_ns: int | None
    route_deviation_ok: bool
    target_reached: bool
    approval_locked: bool


@dataclass(frozen=True)
class ScaleFreeDecision:
    phase: str
    reason: str
    direction_map: tuple[float, float, float]
    body_forward: float
    body_right: float
    body_up: float
    yaw_error_rad: float
    speed_limit_mps: float
    measured_speed_mps: float | None
    allow_translation: bool
    zero_motion: bool
    manual_handoff: bool
    command_expires_mono_ns: int


@dataclass(frozen=True)
class SpeedLimitChange:
    accepted: bool
    reason: str
    old_speed_limit_mps: float
    new_speed_limit_mps: float
    approval_invalidated: bool


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _finite_vector(values: tuple[float, float, float], name: str) -> tuple[float, ...]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    parsed = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _hold(
    reason: str,
    *,
    config: ScaleFreeConfig,
    now_mono_ns: int,
    speed: float | None,
    manual_handoff: bool = True,
) -> ScaleFreeDecision:
    return ScaleFreeDecision(
        phase="HOLD",
        reason=reason,
        direction_map=(0.0, 0.0, 0.0),
        body_forward=0.0,
        body_right=0.0,
        body_up=0.0,
        yaw_error_rad=0.0,
        speed_limit_mps=config.speed_limit_mps,
        measured_speed_mps=speed,
        allow_translation=False,
        zero_motion=True,
        manual_handoff=manual_handoff,
        command_expires_mono_ns=now_mono_ns,
    )


def decide_scale_free(
    sample: ScaleFreeSample,
    *,
    now_mono_ns: int,
    config: ScaleFreeConfig | None = None,
) -> ScaleFreeDecision:
    cfg = config or ScaleFreeConfig()
    now_ns = int(now_mono_ns)
    if now_ns <= 0:
        raise ValueError("now_mono_ns must be positive")
    if sample.approval_locked:
        return _hold(
            "LOCKED_EXTERNAL_APPROVAL",
            config=cfg,
            now_mono_ns=now_ns,
            speed=sample.airframe_horizontal_speed_mps,
        )
    if str(sample.localization_state).upper() != "TRACK":
        return _hold(
            "LOCALIZATION_NOT_TRACK",
            config=cfg,
            now_mono_ns=now_ns,
            speed=sample.airframe_horizontal_speed_mps,
        )

    pose_age_ns = now_ns - int(sample.pose_mono_ns)
    if pose_age_ns < 0 or pose_age_ns > int(cfg.pose_max_age_s * 1e9):
        return _hold(
            "POSE_STALE",
            config=cfg,
            now_mono_ns=now_ns,
            speed=sample.airframe_horizontal_speed_mps,
        )
    speed = sample.airframe_horizontal_speed_mps
    if speed is None or isinstance(speed, bool):
        return _hold("SPEED_UNAVAILABLE", config=cfg, now_mono_ns=now_ns, speed=None)
    speed = float(speed)
    if not math.isfinite(speed) or speed < 0.0:
        return _hold("SPEED_UNAVAILABLE", config=cfg, now_mono_ns=now_ns, speed=None)
    if sample.speed_mono_ns is None:
        return _hold("SPEED_STALE", config=cfg, now_mono_ns=now_ns, speed=speed)
    speed_age_ns = now_ns - int(sample.speed_mono_ns)
    if speed_age_ns < 0 or speed_age_ns > int(cfg.speed_max_age_s * 1e9):
        return _hold("SPEED_STALE", config=cfg, now_mono_ns=now_ns, speed=speed)
    if speed > cfg.speed_limit_mps:
        return _hold("OVERSPEED", config=cfg, now_mono_ns=now_ns, speed=speed)
    if not sample.route_deviation_ok:
        return _hold("ROUTE_DEVIATION", config=cfg, now_mono_ns=now_ns, speed=speed)
    if sample.target_reached:
        decision = _hold(
            "TARGET_REACHED",
            config=cfg,
            now_mono_ns=now_ns,
            speed=speed,
            manual_handoff=False,
        )
        return ScaleFreeDecision(**{**decision.__dict__, "phase": "ARRIVED"})

    position = _finite_vector(sample.position_map, "position_map")
    target = _finite_vector(sample.target_map, "target_map")
    delta = tuple(b - a for a, b in zip(position, target))
    norm = math.sqrt(sum(value * value for value in delta))
    if norm <= 1e-12:
        return _hold(
            "TARGET_VECTOR_ZERO",
            config=cfg,
            now_mono_ns=now_ns,
            speed=speed,
            manual_handoff=False,
        )
    direction = tuple(value / norm for value in delta)
    axis_a, axis_b = cfg.horizontal_axes
    horizontal_norm = math.hypot(direction[axis_a], direction[axis_b])
    camera_yaw = float(sample.camera_yaw_rad)
    if not math.isfinite(camera_yaw):
        return _hold("YAW_INVALID", config=cfg, now_mono_ns=now_ns, speed=speed)
    if horizontal_norm > 1e-12:
        target_yaw = math.atan2(direction[axis_b], direction[axis_a])
        yaw_error = _wrap_angle(target_yaw - camera_yaw)
    else:
        yaw_error = 0.0

    body_yaw = camera_yaw + math.radians(cfg.camera_to_body_yaw_deg)
    world_a = direction[axis_a]
    world_b = direction[axis_b]
    body_forward = world_a * math.cos(body_yaw) + world_b * math.sin(body_yaw)
    body_right = cfg.body_right_sign * (
        -world_a * math.sin(body_yaw) + world_b * math.cos(body_yaw)
    )
    body_up = direction[cfg.vertical_axis]
    expires = now_ns + int(cfg.command_ttl_s * 1e9)
    if horizontal_norm > 1e-12 and abs(math.degrees(yaw_error)) > cfg.yaw_tolerance_deg:
        return ScaleFreeDecision(
            phase="TURN",
            reason="TURN_FIRST",
            direction_map=direction,
            body_forward=0.0,
            body_right=0.0,
            body_up=0.0,
            yaw_error_rad=yaw_error,
            speed_limit_mps=cfg.speed_limit_mps,
            measured_speed_mps=speed,
            allow_translation=False,
            zero_motion=False,
            manual_handoff=False,
            command_expires_mono_ns=expires,
        )
    return ScaleFreeDecision(
        phase="TRANSLATE",
        reason="OK",
        direction_map=direction,
        body_forward=body_forward,
        body_right=body_right,
        body_up=body_up,
        yaw_error_rad=yaw_error,
        speed_limit_mps=cfg.speed_limit_mps,
        measured_speed_mps=speed,
        allow_translation=True,
        zero_motion=False,
        manual_handoff=False,
        command_expires_mono_ns=expires,
    )


def command_is_fresh(command_expires_mono_ns: int, now_mono_ns: int) -> bool:
    return int(now_mono_ns) <= int(command_expires_mono_ns)


def validate_speed_limit_change(
    old_speed_limit_mps: float,
    new_speed_limit_mps: float,
    *,
    landed: bool,
) -> SpeedLimitChange:
    old = float(old_speed_limit_mps)
    new = float(new_speed_limit_mps)
    if not math.isfinite(new) or new <= 0.0:
        return SpeedLimitChange(False, "INVALID_SPEED_LIMIT", old, new, False)
    if not landed:
        return SpeedLimitChange(False, "LANDED_REQUIRED", old, new, False)
    changed = not math.isclose(old, new, rel_tol=0.0, abs_tol=1e-12)
    return SpeedLimitChange(True, "OK", old, new, changed)
