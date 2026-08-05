"""Closed-loop PCMD waypoint following in the Sphinx ENU world."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

from .models import PilotingCommand, TruePosition
from .safety import SPHINX_DRONE_IP, require_sphinx_target
from .scale_free_control import ScaleFreeConfig, ScaleFreeSample, decide_scale_free
from .telemetry import TrueTelemetryCollector


@dataclass(frozen=True)
class Waypoint:
    index: int
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class SpawnPose:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float

    @property
    def sphinx_pose(self) -> str:
        return f"{self.x_m} {self.y_m} {self.z_m} 0 0 {self.yaw_rad}"


@dataclass(frozen=True)
class RoutePlan:
    seed: int
    sphinx_spawn: SpawnPose
    waypoints: tuple[Waypoint, ...]


@dataclass(frozen=True)
class Anafi4KFlightProfile:
    """Standard ANAFI settings plus hardware limits from White Paper v1.4."""

    model: str = "Parrot ANAFI 4K"
    configured_max_tilt_deg: float = 20.0
    configured_max_vertical_speed_m_s: float = 1.0
    configured_max_yaw_rate_deg_s: float = 70.0
    configured_max_pitch_roll_rate_deg_s: float = 150.0
    hardware_max_horizontal_speed_m_s: float = 15.0
    hardware_max_vertical_speed_m_s: float = 4.0
    hardware_max_yaw_rate_deg_s: float = 200.0


ANAFI_4K_STANDARD_PROFILE = Anafi4KFlightProfile()


@dataclass(frozen=True)
class FlightErrorConfig:
    """The only simulated errors: bounded wind displacement and localization error."""

    wind_maximum_displacement_m: float = 0.40
    wind_displacement_interval_s: float = 20.0
    maximum_position_error_m: float = 0.30
    maximum_yaw_error_deg: float = 10.0
    localization_latency_s: float = 0.20
    localization_dropout_probability: float = 0.03
    localization_error_correlation: float = 0.85
    localization_error_basis: str = "operator-observed GlueMap/EDM position and camera-heading caps"


@dataclass(frozen=True)
class NavigationFrameConfig:
    """Explicit map scale and camera-to-body extrinsics used by the controller."""

    localization_frame: str = "Gazebo ENU"
    metres_per_map_unit: float = 1.0
    camera_to_body_forward_m: float = 0.0
    camera_to_body_right_m: float = 0.0
    camera_to_body_up_m: float = 0.0
    camera_yaw_to_body_yaw_deg: float = 0.0


@dataclass(frozen=True)
class RouteConfig:
    arrival_tolerance_m: float = 0.5
    yaw_tolerance_deg: float = 3.0
    yaw_alignment_confirmation_updates: int = 3
    yaw_alignment_max_rate_deg_s: float = 5.0
    minimum_yaw_alignment_distance_m: float = 0.25
    max_translation_pcmd: int = 10
    max_yaw_pcmd: int = 20
    control_period_s: float = 1.0
    pcmd_watchdog_s: float = 0.75
    waypoint_hover_s: float = 0.75
    waypoint_timeout_s: float = 120.0
    arrival_confirmation_updates: int = 3
    arrival_max_speed_m_s: float = 0.25
    slowdown_distance_m: float = 1.5
    maximum_cruise_speed_m_s: float = 0.30
    minimum_localization_confidence: float = 0.20
    maximum_pose_age_s: float = 0.50
    maximum_consecutive_lost_updates: int = 3
    minimum_takeoff_battery_percent: int = 30
    minimum_altitude_m: float = 0.10
    maximum_altitude_m: float = 10.0
    maximum_route_deviation_m: float = 3.0
    frame: NavigationFrameConfig = NavigationFrameConfig()
    error_model: FlightErrorConfig = FlightErrorConfig()


@dataclass(frozen=True)
class RouteStressScenario:
    """One deterministic bounded-error case for the full Sphinx route."""

    name: str
    position_bias: str = "none"
    yaw_error_deg: float = 0.0
    final_approach_wind_m: float = 0.0
    drop_after_nonzero_pcmd: bool = False

    def __post_init__(self) -> None:
        if self.position_bias not in {"none", "toward_target", "away_from_target"}:
            raise ValueError(f"unsupported position bias: {self.position_bias}")
        if self.final_approach_wind_m < 0.0:
            raise ValueError("final_approach_wind_m must not be negative")


def worst_case_route_scenarios(
    config: RouteConfig | None = None,
) -> tuple[RouteStressScenario, ...]:
    """Return fixed cases that place each allowed error at its configured bound."""
    cfg = config or RouteConfig()
    position_cap = cfg.error_model.maximum_position_error_m
    yaw_cap = cfg.error_model.maximum_yaw_error_deg
    wind_cap = cfg.error_model.wind_maximum_displacement_m
    return (
        RouteStressScenario(
            name="false-arrival-bias",
            position_bias="toward_target",
            yaw_error_deg=yaw_cap,
        ),
        RouteStressScenario(
            name="overshoot-bias",
            position_bias="away_from_target",
            yaw_error_deg=-yaw_cap,
        ),
        RouteStressScenario(
            name="final-approach-wind",
            final_approach_wind_m=wind_cap,
        ),
        RouteStressScenario(
            name="post-command-localization-loss",
            drop_after_nonzero_pcmd=True,
        ),
        RouteStressScenario(
            name="combined-bounds",
            position_bias="toward_target" if position_cap > 0.0 else "none",
            yaw_error_deg=yaw_cap,
            final_approach_wind_m=wind_cap,
            drop_after_nonzero_pcmd=True,
        ),
    )


def route_stress_scenario(name: str, config: RouteConfig | None = None) -> RouteStressScenario:
    """Resolve one named deterministic route stress case."""
    for scenario in worst_case_route_scenarios(config):
        if scenario.name == name:
            return scenario
    raise ValueError(f"unknown route stress scenario: {name}")


@dataclass(frozen=True)
class ControlDecision:
    phase: str
    command: PilotingCommand
    dx_m: float
    dy_m: float
    dz_m: float
    horizontal_distance_m: float
    distance_m: float
    camera_to_target_yaw_enu_rad: float
    camera_turn_angle_enu_rad: float
    horizontal_fraction: float
    forward_fraction: float
    right_fraction: float
    vertical_fraction: float
    estimated_speed_m_s: float
    estimated_horizontal_speed_m_s: float
    estimated_closing_speed_m_s: float


@dataclass
class LocalizationErrorState:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    yaw_deg: float = 0.0
    initialized: bool = False


@dataclass
class YawAlignmentTracker:
    """Require a quiet, repeatable yaw estimate before enabling translation."""

    confirmation_count: int = 0
    previous_timestamp_s: float | None = None
    previous_yaw_rad: float | None = None

    def reset(self) -> None:
        self.confirmation_count = 0
        self.previous_timestamp_s = None
        self.previous_yaw_rad = None

    def update(
        self,
        *,
        yaw_error_rad: float,
        estimated_yaw_rad: float,
        timestamp_s: float,
        config: RouteConfig,
    ) -> tuple[bool, float | None]:
        yaw_rate_deg_s: float | None = None
        if self.previous_timestamp_s is not None and self.previous_yaw_rad is not None:
            dt = timestamp_s - self.previous_timestamp_s
            if dt > 1e-6:
                yaw_rate_deg_s = (
                    math.degrees(_wrap_angle_rad(estimated_yaw_rad - self.previous_yaw_rad)) / dt
                )
        self.previous_timestamp_s = timestamp_s
        self.previous_yaw_rad = estimated_yaw_rad

        stable = (
            abs(math.degrees(yaw_error_rad)) <= config.yaw_tolerance_deg
            and yaw_rate_deg_s is not None
            and abs(yaw_rate_deg_s) <= config.yaw_alignment_max_rate_deg_s
        )
        self.confirmation_count = self.confirmation_count + 1 if stable else 0
        return (
            self.confirmation_count >= config.yaw_alignment_confirmation_updates,
            yaw_rate_deg_s,
        )


def pcmd_physical_setpoints(
    command: PilotingCommand,
    profile: Anafi4KFlightProfile = ANAFI_4K_STANDARD_PROFILE,
) -> dict[str, float]:
    """Convert PCMD percentages to the configured ANAFI command setpoints."""
    return {
        "roll_tilt_command_deg": command.roll * profile.configured_max_tilt_deg / 100.0,
        "pitch_tilt_command_deg": command.pitch * profile.configured_max_tilt_deg / 100.0,
        "yaw_rate_command_clockwise_deg_s": (
            command.yaw * profile.configured_max_yaw_rate_deg_s / 100.0
        ),
        "vertical_speed_command_up_m_s": (
            command.gaz * profile.configured_max_vertical_speed_m_s / 100.0
        ),
    }


def estimate_navigation_state(
    position: TruePosition,
    camera_yaw_enu_rad: float,
    *,
    rng: random.Random,
    config: FlightErrorConfig,
    state: LocalizationErrorState | None = None,
) -> tuple[TruePosition, float, dict[str, float]]:
    """Return the position and camera yaw understood by the controller."""
    while True:
        error_x = rng.uniform(-config.maximum_position_error_m, config.maximum_position_error_m)
        error_y = rng.uniform(-config.maximum_position_error_m, config.maximum_position_error_m)
        error_z = rng.uniform(-config.maximum_position_error_m, config.maximum_position_error_m)
        position_error_m = math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z)
        if position_error_m <= config.maximum_position_error_m:
            break
    sampled_yaw_error_deg = rng.uniform(
        -config.maximum_yaw_error_deg,
        config.maximum_yaw_error_deg,
    )
    if state is not None and state.initialized:
        alpha = config.localization_error_correlation
        error_x = alpha * state.x_m + (1.0 - alpha) * error_x
        error_y = alpha * state.y_m + (1.0 - alpha) * error_y
        error_z = alpha * state.z_m + (1.0 - alpha) * error_z
        yaw_error_deg = alpha * state.yaw_deg + (1.0 - alpha) * sampled_yaw_error_deg
        position_error_m = math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z)
    else:
        yaw_error_deg = sampled_yaw_error_deg
    if state is not None:
        state.x_m = error_x
        state.y_m = error_y
        state.z_m = error_z
        state.yaw_deg = yaw_error_deg
        state.initialized = True
    estimated_position = TruePosition(
        timestamp_s=position.timestamp_s,
        x_m=position.x_m + error_x,
        y_m=position.y_m + error_y,
        z_m=position.z_m + error_z,
    )
    estimated_yaw = _wrap_angle_rad(camera_yaw_enu_rad + math.radians(yaw_error_deg))
    return (
        estimated_position,
        estimated_yaw,
        {
            "position_error_x_m": error_x,
            "position_error_y_m": error_y,
            "position_error_z_m": error_z,
            "position_error_m": position_error_m,
            "yaw_error_deg": yaw_error_deg,
            "localization_confidence": max(
                0.0,
                1.0
                - 0.6 * position_error_m / config.maximum_position_error_m
                - 0.4 * abs(yaw_error_deg) / config.maximum_yaw_error_deg,
            ),
        },
    )


def estimate_stress_navigation_state(
    position: TruePosition,
    camera_yaw_enu_rad: float,
    target: Waypoint,
    *,
    config: FlightErrorConfig,
    scenario: RouteStressScenario,
) -> tuple[TruePosition, float, dict[str, float]]:
    """Apply a fixed, maximum bounded bias instead of a random error sample."""
    dx = target.x_m - position.x_m
    dy = target.y_m - position.y_m
    dz = target.z_m - position.z_m
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    sign = 1.0 if scenario.position_bias == "toward_target" else -1.0
    if scenario.position_bias == "none" or distance <= 1e-9:
        error_x = error_y = error_z = 0.0
    else:
        scale = sign * config.maximum_position_error_m / distance
        error_x, error_y, error_z = dx * scale, dy * scale, dz * scale
    yaw_error_deg = max(
        -config.maximum_yaw_error_deg,
        min(config.maximum_yaw_error_deg, scenario.yaw_error_deg),
    )
    position_error_m = math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z)
    return (
        TruePosition(
            timestamp_s=position.timestamp_s,
            x_m=position.x_m + error_x,
            y_m=position.y_m + error_y,
            z_m=position.z_m + error_z,
        ),
        _wrap_angle_rad(camera_yaw_enu_rad + math.radians(yaw_error_deg)),
        {
            "position_error_x_m": error_x,
            "position_error_y_m": error_y,
            "position_error_z_m": error_z,
            "position_error_m": position_error_m,
            "yaw_error_deg": yaw_error_deg,
            # Stress cases test bounded pose error, not EDM confidence gating.
            "localization_confidence": 1.0,
        },
    )


def _wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def enu_yaw_from_olympe_heading(heading_ned_rad: float) -> float:
    """Convert clockwise-from-North firmware heading to CCW-from-East ENU yaw."""
    return _wrap_angle_rad(math.pi / 2 - heading_ned_rad)


def _arsdk_state_name(value: object) -> str | None:
    """Normalize Olympe enum values and plain test strings to the same state name."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name if name is not None else value).rsplit(".", 1)[-1]


def camera_turn_angles(
    camera_position: TruePosition,
    target: Waypoint,
    *,
    camera_yaw_enu_rad: float,
) -> tuple[float, float]:
    """Return camera-to-target XY yaw and the shortest signed turn angle."""
    dx = target.x_m - camera_position.x_m
    dy = target.y_m - camera_position.y_m
    if math.hypot(dx, dy) <= 1e-9:
        return camera_yaw_enu_rad, 0.0
    camera_to_target_yaw = math.atan2(dy, dx)
    turn_angle = _wrap_angle_rad(camera_to_target_yaw - camera_yaw_enu_rad)
    return camera_to_target_yaw, turn_angle


def estimate_velocity_enu(
    previous: TruePosition | None,
    current: TruePosition,
) -> tuple[float, float, float]:
    """Estimate ENU velocity from consecutive localization results."""
    if previous is None:
        return 0.0, 0.0, 0.0
    dt = current.timestamp_s - previous.timestamp_s
    if dt <= 1e-6:
        return 0.0, 0.0, 0.0
    return (
        (current.x_m - previous.x_m) / dt,
        (current.y_m - previous.y_m) / dt,
        (current.z_m - previous.z_m) / dt,
    )


def interpolated_yaw_at(
    samples: list[tuple[float, float]],
    timestamp_s: float,
) -> float | None:
    """Interpolate wrapped ENU yaw at a past timestamp without extrapolating."""
    before: tuple[float, float] | None = None
    for after in samples:
        if after[0] < timestamp_s:
            before = after
            continue
        if after[0] == timestamp_s:
            return after[1]
        if before is None:
            return None
        span_s = after[0] - before[0]
        if span_s <= 0:
            return None
        fraction = (timestamp_s - before[0]) / span_s
        yaw_delta = _wrap_angle_rad(after[1] - before[1])
        return _wrap_angle_rad(before[1] + fraction * yaw_delta)
    return None


def estimated_arrival_tolerance_m(config: RouteConfig) -> float:
    """Reserve the true 0.5 m radius for bounded pose error and latency motion."""
    uncertainty_reserve_m = config.error_model.maximum_position_error_m
    latency_motion_reserve_m = (
        config.arrival_max_speed_m_s * config.error_model.localization_latency_s
    )
    return max(
        0.0,
        config.arrival_tolerance_m - uncertainty_reserve_m - latency_motion_reserve_m,
    )


def route_deviation_m(position: TruePosition, plan: RoutePlan) -> float:
    """Return the shortest 3D distance from a position to the planned polyline."""
    points = [
        (plan.sphinx_spawn.x_m, plan.sphinx_spawn.y_m, plan.sphinx_spawn.z_m),
        *((point.x_m, point.y_m, point.z_m) for point in plan.waypoints),
    ]
    p = (position.x_m, position.y_m, position.z_m)
    distances: list[float] = []
    for start, end in pairwise(points):
        segment = tuple(b - a for a, b in zip(start, end))
        length_squared = sum(value * value for value in segment)
        if length_squared <= 1e-12:
            distances.append(math.dist(p, start))
            continue
        along = sum((value - a) * delta for value, a, delta in zip(p, start, segment))
        t = max(0.0, min(1.0, along / length_squared))
        closest = tuple(a + t * delta for a, delta in zip(start, segment))
        distances.append(math.dist(p, closest))
    return min(distances)


def safety_violation(
    position: TruePosition,
    plan: RoutePlan,
    config: RouteConfig,
    *,
    pose_age_s: float,
    confidence: float,
) -> str | None:
    """Return the first fail-safe reason for a localization result."""
    if pose_age_s > config.maximum_pose_age_s:
        return "stale_localization"
    if confidence < config.minimum_localization_confidence:
        return "low_localization_confidence"
    if not config.minimum_altitude_m <= position.z_m <= config.maximum_altitude_m:
        return "altitude_geofence"
    if route_deviation_m(position, plan) > config.maximum_route_deviation_m:
        return "route_deviation"
    return None


def generate_route(seed: int, *, waypoint_count: int = 10) -> RoutePlan:
    """Generate a reproducible forward route that winds left/right and up/down."""
    if waypoint_count <= 0:
        raise ValueError("waypoint_count must be positive")
    rng = random.Random(seed)
    first = Waypoint(
        index=0,
        x_m=round(rng.uniform(-1.0, -0.8), 3),
        y_m=round(rng.uniform(-0.1, 0.1), 3),
        z_m=round(rng.uniform(1.2, 1.4), 3),
    )
    waypoints = [first]
    x_m = first.x_m
    while len(waypoints) < waypoint_count:
        index = len(waypoints)
        # Keep the final point over the finite collision ground in parrot-ue4-empty.
        # The alternating Y/Z offsets still make each 3D segment about one metre long.
        x_m += rng.uniform(0.35, 0.45)
        side = 1 if index % 2 else -1
        waypoint = Waypoint(
            index=index,
            x_m=round(x_m, 3),
            y_m=round(first.y_m + side * rng.uniform(0.20, 0.35), 3),
            z_m=round(
                rng.uniform(2.0, 2.2) if index % 2 else rng.uniform(1.3, 1.5),
                3,
            ),
        )
        waypoints.append(waypoint)

    sphinx_spawn = SpawnPose(
        x_m=0.0,
        y_m=0.0,
        z_m=0.2,
        yaw_rad=0.0,
    )
    return RoutePlan(
        seed=seed,
        sphinx_spawn=sphinx_spawn,
        waypoints=tuple(waypoints),
    )


def control_decision(
    position: TruePosition,
    target: Waypoint,
    *,
    camera_yaw_enu_rad: float,
    config: RouteConfig,
    require_yaw_alignment: bool = True,
    velocity_enu_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ControlDecision:
    """Recompute the camera-target line, turn angle, and PCMD for one cycle."""
    dx = target.x_m - position.x_m
    dy = target.y_m - position.y_m
    dz = target.z_m - position.z_m
    horizontal = math.hypot(dx, dy)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    camera_to_target_yaw, camera_turn_angle = camera_turn_angles(
        position,
        target,
        camera_yaw_enu_rad=camera_yaw_enu_rad,
    )
    estimated_speed_m_s = math.sqrt(sum(value * value for value in velocity_enu_m_s))
    estimated_horizontal_speed_m_s = math.hypot(
        velocity_enu_m_s[0],
        velocity_enu_m_s[1],
    )
    core_now_ns = max(1, round(position.timestamp_s * 1_000_000_000))
    core = decide_scale_free(
        ScaleFreeSample(
            position_map=(position.x_m, position.y_m, position.z_m),
            target_map=(target.x_m, target.y_m, target.z_m),
            camera_yaw_rad=camera_yaw_enu_rad,
            pose_mono_ns=core_now_ns,
            localization_state="TRACK",
            airframe_horizontal_speed_mps=estimated_horizontal_speed_m_s,
            speed_mono_ns=core_now_ns,
            route_deviation_ok=True,
            target_reached=False,
            approval_locked=False,
        ),
        now_mono_ns=core_now_ns,
        config=ScaleFreeConfig(
            speed_limit_mps=config.maximum_cruise_speed_m_s,
            pose_max_age_s=config.maximum_pose_age_s,
            speed_max_age_s=config.maximum_pose_age_s,
            command_ttl_s=config.pcmd_watchdog_s,
            yaw_tolerance_deg=(
                config.yaw_tolerance_deg
                if require_yaw_alignment and horizontal > config.minimum_yaw_alignment_distance_m
                else 180.0
            ),
            horizontal_axes=(0, 1),
            vertical_axis=2,
            camera_to_body_yaw_deg=config.frame.camera_yaw_to_body_yaw_deg,
            body_right_sign=-1.0,
        ),
    )
    horizontal_fraction = horizontal / distance if distance else 0.0
    forward_fraction = core.body_forward
    right_fraction = core.body_right
    vertical_fraction = core.body_up
    estimated_closing_speed_m_s = (
        sum(value * delta for value, delta in zip(velocity_enu_m_s, (dx, dy, dz))) / distance
        if distance
        else 0.0
    )
    controller_arrival_tolerance_m = estimated_arrival_tolerance_m(config)

    if (
        distance <= controller_arrival_tolerance_m
        and estimated_speed_m_s <= config.arrival_max_speed_m_s
    ):
        phase = "hover"
        command = PilotingCommand.zero()
    elif (
        estimated_horizontal_speed_m_s > config.maximum_cruise_speed_m_s
        or distance <= controller_arrival_tolerance_m
        or (
            distance <= config.slowdown_distance_m
            and estimated_closing_speed_m_s
            > max(
                config.arrival_max_speed_m_s,
                distance - controller_arrival_tolerance_m,
            )
        )
    ):
        phase = "brake"
        command = PilotingCommand.zero()
    elif require_yaw_alignment and (
        horizontal > config.minimum_yaw_alignment_distance_m
        and abs(math.degrees(camera_turn_angle)) > config.yaw_tolerance_deg
    ):
        phase = "turn"
        yaw_strength = min(
            config.max_yaw_pcmd,
            max(
                1,
                round(
                    100
                    * abs(math.degrees(camera_turn_angle))
                    / ANAFI_4K_STANDARD_PROFILE.configured_max_yaw_rate_deg_s
                    / config.control_period_s
                ),
            ),
        )
        # PCMD yaw is positive clockwise; ENU yaw is positive counter-clockwise.
        yaw = -yaw_strength if camera_turn_angle > 0 else yaw_strength
        command = PilotingCommand(roll=0, pitch=0, yaw=yaw, gaz=0)
    else:
        phase = "translate"
        raw_strength = min(
            config.max_translation_pcmd,
            max(2, round(config.max_translation_pcmd * distance / 1.5)),
        )
        remaining_m = max(0.0, distance - controller_arrival_tolerance_m)
        slowdown_span_m = max(
            0.01,
            config.slowdown_distance_m - controller_arrival_tolerance_m,
        )
        slowdown_scale = max(0.25, min(1.0, remaining_m / slowdown_span_m))
        strength = max(1, round(raw_strength * slowdown_scale))
        pitch = round(strength * forward_fraction)
        roll = round(strength * right_fraction)
        gaz = round(strength * vertical_fraction)
        command = PilotingCommand(roll=roll, pitch=pitch, yaw=0, gaz=gaz)

    return ControlDecision(
        phase=phase,
        command=command,
        dx_m=dx,
        dy_m=dy,
        dz_m=dz,
        horizontal_distance_m=horizontal,
        distance_m=distance,
        camera_to_target_yaw_enu_rad=camera_to_target_yaw,
        camera_turn_angle_enu_rad=camera_turn_angle,
        horizontal_fraction=horizontal_fraction,
        forward_fraction=forward_fraction,
        right_fraction=right_fraction,
        vertical_fraction=vertical_fraction,
        estimated_speed_m_s=estimated_speed_m_s,
        estimated_horizontal_speed_m_s=estimated_horizontal_speed_m_s,
        estimated_closing_speed_m_s=estimated_closing_speed_m_s,
    )


class OlympeRouteFollower:
    """Continuously reconnect the drone to each waypoint and correct its PCMD."""

    def __init__(
        self,
        *,
        config: RouteConfig | None = None,
        target_ip: str = SPHINX_DRONE_IP,
        stress_scenario: RouteStressScenario | None = None,
    ) -> None:
        require_sphinx_target(target_ip)
        self.config = config or RouteConfig()
        self.target_ip = target_ip
        self.stress_scenario = stress_scenario
        self.records: list[dict[str, object]] = []
        self.arrivals: list[dict[str, object]] = []
        self.initial_position: TruePosition | None = None
        self.final_position: TruePosition | None = None
        self.takeoff_attempts = 0
        self.landing_attempts = 0
        self.landing_error: str | None = None
        self.landed = False
        self.aircraft_settings: dict[str, object] | None = None
        self.safety_events: list[dict[str, object]] = []

    @staticmethod
    def _expect_success(result: Any, operation: str) -> None:
        if not result.success():
            raise RuntimeError(f"Olympe operation failed: {operation}")

    @staticmethod
    def _read_camera_yaw_enu_rad(drone: Any, attitude_message: Any) -> float:
        """Read horizontal camera yaw; ANAFI's camera has no independent yaw axis."""
        state = drone.get_state(attitude_message)
        heading = state.get("yaw") if state is not None else None
        if heading is None:
            raise RuntimeError("Olympe did not provide an AttitudeChanged yaw state")
        return enu_yaw_from_olympe_heading(float(heading))

    def _apply_anafi_4k_profile(self, drone: Any) -> None:
        """Pin the real ANAFI standard settings and record firmware-confirmed values."""
        from olympe.messages.ardrone3.PilotingSettings import MaxTilt
        from olympe.messages.ardrone3.PilotingSettingsState import MaxTiltChanged
        from olympe.messages.ardrone3.SpeedSettings import (
            MaxPitchRollRotationSpeed,
            MaxRotationSpeed,
            MaxVerticalSpeed,
        )
        from olympe.messages.ardrone3.SpeedSettingsState import (
            MaxPitchRollRotationSpeedChanged,
            MaxRotationSpeedChanged,
            MaxVerticalSpeedChanged,
        )
        from olympe.messages.common.SettingsState import ProductVersionChanged

        profile = ANAFI_4K_STANDARD_PROFILE
        settings = (
            (MaxTilt(profile.configured_max_tilt_deg), "set ANAFI maximum tilt"),
            (
                MaxVerticalSpeed(profile.configured_max_vertical_speed_m_s),
                "set ANAFI maximum vertical speed",
            ),
            (
                MaxRotationSpeed(profile.configured_max_yaw_rate_deg_s),
                "set ANAFI maximum yaw rate",
            ),
            (
                MaxPitchRollRotationSpeed(profile.configured_max_pitch_roll_rate_deg_s),
                "set ANAFI maximum pitch/roll rate",
            ),
        )
        for message, operation in settings:
            self._expect_success(drone(message).wait(), operation)

        self.aircraft_settings = {
            "product_version": dict(drone.get_state(ProductVersionChanged)),
            "max_tilt": dict(drone.get_state(MaxTiltChanged)),
            "max_vertical_speed": dict(drone.get_state(MaxVerticalSpeedChanged)),
            "max_yaw_rate": dict(drone.get_state(MaxRotationSpeedChanged)),
            "max_pitch_roll_rate": dict(drone.get_state(MaxPitchRollRotationSpeedChanged)),
        }

    def _record(
        self,
        *,
        started_at: float,
        true_position: TruePosition,
        estimated_position: TruePosition,
        target: Waypoint,
        true_camera_yaw: float,
        estimated_camera_yaw: float,
        decision: ControlDecision,
        estimation_error: dict[str, float],
        arrival_verified: bool,
        pose_age_s: float,
        arrival_confirmation_count: int,
        yaw_alignment_confirmation_count: int,
        estimated_yaw_rate_deg_s: float | None,
        yaw_alignment_confirmed: bool,
        safety_state: str,
    ) -> None:
        command_setpoints = pcmd_physical_setpoints(decision.command)
        true_distance = math.dist(
            (true_position.x_m, true_position.y_m, true_position.z_m),
            (target.x_m, target.y_m, target.z_m),
        )
        self.records.append(
            {
                "elapsed_s": time.monotonic() - started_at,
                "telemetry_timestamp_s": true_position.timestamp_s,
                "waypoint_index": target.index,
                "phase": decision.phase,
                "camera_x_m": true_position.x_m,
                "camera_y_m": true_position.y_m,
                "camera_z_m": true_position.z_m,
                "estimated_camera_x_m": estimated_position.x_m,
                "estimated_camera_y_m": estimated_position.y_m,
                "estimated_camera_z_m": estimated_position.z_m,
                "localization_source_timestamp_s": estimated_position.timestamp_s,
                "localization_pose_age_s": pose_age_s,
                "target_x_m": target.x_m,
                "target_y_m": target.y_m,
                "target_z_m": target.z_m,
                "estimated_line_dx_m": decision.dx_m,
                "estimated_line_dy_m": decision.dy_m,
                "estimated_line_dz_m": decision.dz_m,
                "estimated_horizontal_distance_m": decision.horizontal_distance_m,
                "estimated_distance_m": decision.distance_m,
                "true_distance_m": true_distance,
                "camera_yaw_enu_deg": math.degrees(true_camera_yaw),
                "estimated_camera_yaw_enu_deg": math.degrees(estimated_camera_yaw),
                "camera_to_target_yaw_enu_deg": math.degrees(decision.camera_to_target_yaw_enu_rad),
                "camera_turn_angle_enu_deg": math.degrees(decision.camera_turn_angle_enu_rad),
                "horizontal_fraction": decision.horizontal_fraction,
                "forward_fraction": decision.forward_fraction,
                "right_fraction": decision.right_fraction,
                "vertical_fraction": decision.vertical_fraction,
                "pcmd_roll": decision.command.roll,
                "pcmd_pitch": decision.command.pitch,
                "pcmd_yaw": decision.command.yaw,
                "pcmd_gaz": decision.command.gaz,
                "estimated_speed_m_s": decision.estimated_speed_m_s,
                "estimated_horizontal_speed_m_s": decision.estimated_horizontal_speed_m_s,
                "estimated_closing_speed_m_s": decision.estimated_closing_speed_m_s,
                "arrival_confirmation_count": arrival_confirmation_count,
                "yaw_alignment_confirmation_count": yaw_alignment_confirmation_count,
                "estimated_yaw_rate_deg_s": estimated_yaw_rate_deg_s,
                "yaw_alignment_confirmed": yaw_alignment_confirmed,
                "safety_state": safety_state,
                "wind_disturbance_model": "bounded horizontal true-position displacement",
                "wind_maximum_displacement_m": (
                    self.config.error_model.wind_maximum_displacement_m
                ),
                "wind_displacement_interval_s": (
                    self.config.error_model.wind_displacement_interval_s
                ),
                "localization_maximum_position_error_m": (
                    self.config.error_model.maximum_position_error_m
                ),
                "localization_maximum_yaw_error_deg": (
                    self.config.error_model.maximum_yaw_error_deg
                ),
                "arrival_verified": arrival_verified,
                **estimation_error,
                **command_setpoints,
            }
        )

    def run(
        self,
        plan: RoutePlan,
        *,
        telemetry: TrueTelemetryCollector | None = None,
        before_landing: Callable[[], None] | None = None,
        control_disturbance: Callable[[TruePosition, Waypoint, bool], None] | None = None,
    ) -> None:
        """Take off, follow every waypoint with PCMD feedback, then land."""
        require_sphinx_target(self.target_ip)
        import olympe
        from olympe.messages.ardrone3.Piloting import Landing, TakeOff
        from olympe.messages.ardrone3.PilotingState import AttitudeChanged, FlyingStateChanged
        from olympe.messages.common.CommonState import BatteryStateChanged

        collector = telemetry or TrueTelemetryCollector()
        drone = olympe.Drone(self.target_ip)
        connected = False
        airborne = False
        piloting_started = False
        started_at = time.monotonic()
        try:
            if not drone.connect():
                raise RuntimeError(f"unable to connect to Sphinx virtual ANAFI at {self.target_ip}")
            connected = True
            time.sleep(2.0)
            self._apply_anafi_4k_profile(drone)
            battery_percent = int(drone.get_state(BatteryStateChanged).get("percent", -1))
            if battery_percent < self.config.minimum_takeoff_battery_percent:
                raise RuntimeError(
                    "takeoff blocked by battery gate: "
                    f"{battery_percent}% < {self.config.minimum_takeoff_battery_percent}%"
                )
            for attempt in range(1, 4):
                self.takeoff_attempts = attempt
                takeoff = drone(
                    TakeOff() >> FlyingStateChanged(state="hovering", _timeout=15)
                ).wait()
                if takeoff.success():
                    airborne = True
                    break
                flying_state = _arsdk_state_name(drone.get_state(FlyingStateChanged).get("state"))
                if flying_state != "landed":
                    airborne = flying_state not in ("emergency", None)
                    self._expect_success(takeoff, f"takeoff and hover (state={flying_state})")
                time.sleep(2.0)
            else:
                self._expect_success(takeoff, "takeoff and hover after 3 attempts")
            collector.start()
            self.initial_position = collector.wait_for_sample(timeout_s=15)
            estimation_rng = random.Random(plan.seed ^ 0xA4AF1)
            error_state = LocalizationErrorState()
            previous_estimated_position: TruePosition | None = None
            last_estimated_position: TruePosition | None = None
            last_estimated_yaw = 0.0
            last_estimation_error = {
                "position_error_x_m": 0.0,
                "position_error_y_m": 0.0,
                "position_error_z_m": 0.0,
                "position_error_m": 0.0,
                "yaw_error_deg": 0.0,
                "localization_confidence": 0.0,
            }
            true_camera_yaw_history: list[tuple[float, float]] = []
            consecutive_lost_updates = 0
            previous_command_nonzero = False
            forced_loss_injected = False

            for target in plan.waypoints:
                deadline = time.monotonic() + self.config.waypoint_timeout_s
                aligned_for_translation = False
                arrival_confirmation_count = 0
                yaw_alignment = YawAlignmentTracker()
                estimated_yaw_rate_deg_s: float | None = None
                while time.monotonic() < deadline:
                    true_position = collector.latest_sample()
                    if control_disturbance is not None:
                        control_disturbance(
                            true_position,
                            target,
                            target.index == plan.waypoints[-1].index,
                        )
                    true_camera_yaw = self._read_camera_yaw_enu_rad(drone, AttitudeChanged)
                    true_camera_yaw_history.append((true_position.timestamp_s, true_camera_yaw))
                    localization_source = collector.interpolated_sample_at(
                        true_position.timestamp_s - self.config.error_model.localization_latency_s
                    )
                    delayed_true_camera_yaw = (
                        interpolated_yaw_at(
                            true_camera_yaw_history,
                            localization_source.timestamp_s,
                        )
                        if localization_source is not None
                        else None
                    )
                    forced_loss = bool(
                        self.stress_scenario is not None
                        and self.stress_scenario.drop_after_nonzero_pcmd
                        and previous_command_nonzero
                        and not forced_loss_injected
                    )
                    localization_dropped = (
                        forced_loss
                        if self.stress_scenario is not None
                        else estimation_rng.random()
                        < self.config.error_model.localization_dropout_probability
                    )
                    if (
                        localization_source is None
                        or delayed_true_camera_yaw is None
                        or localization_dropped
                    ):
                        consecutive_lost_updates += 1
                        arrival_confirmation_count = 0
                        yaw_alignment.reset()
                        estimated_yaw_rate_deg_s = None
                        if forced_loss:
                            forced_loss_injected = True
                        self.safety_events.append(
                            {
                                "elapsed_s": time.monotonic() - started_at,
                                "reason": "localization_lost",
                                "consecutive_updates": consecutive_lost_updates,
                            }
                        )
                        if last_estimated_position is not None:
                            lost_decision = control_decision(
                                last_estimated_position,
                                target,
                                camera_yaw_enu_rad=last_estimated_yaw,
                                config=self.config,
                                require_yaw_alignment=True,
                            )
                            lost_decision = replace(
                                lost_decision,
                                phase="localization_lost",
                                command=PilotingCommand.zero(),
                            )
                            self._record(
                                started_at=started_at,
                                true_position=true_position,
                                estimated_position=last_estimated_position,
                                target=target,
                                true_camera_yaw=true_camera_yaw,
                                estimated_camera_yaw=last_estimated_yaw,
                                decision=lost_decision,
                                estimation_error=last_estimation_error,
                                arrival_verified=False,
                                pose_age_s=(
                                    true_position.timestamp_s - last_estimated_position.timestamp_s
                                ),
                                arrival_confirmation_count=0,
                                yaw_alignment_confirmation_count=0,
                                estimated_yaw_rate_deg_s=None,
                                yaw_alignment_confirmed=False,
                                safety_state="localization_lost_hover",
                            )
                        if not drone.piloting(0, 0, 0, 0, self.config.pcmd_watchdog_s):
                            raise RuntimeError("Olympe refused localization-loss hover PCMD")
                        piloting_started = True
                        previous_command_nonzero = False
                        if consecutive_lost_updates > self.config.maximum_consecutive_lost_updates:
                            raise RuntimeError(
                                "localization lost beyond configured fail-safe limit"
                            )
                        time.sleep(self.config.control_period_s)
                        continue

                    estimated_position, estimated_camera_yaw, estimation_error = (
                        estimate_stress_navigation_state(
                            localization_source,
                            delayed_true_camera_yaw,
                            target,
                            config=self.config.error_model,
                            scenario=self.stress_scenario,
                        )
                        if self.stress_scenario is not None
                        else estimate_navigation_state(
                            localization_source,
                            delayed_true_camera_yaw,
                            rng=estimation_rng,
                            config=self.config.error_model,
                            state=error_state,
                        )
                    )
                    pose_age_s = true_position.timestamp_s - estimated_position.timestamp_s
                    confidence = estimation_error["localization_confidence"]
                    velocity_enu = estimate_velocity_enu(
                        previous_estimated_position,
                        estimated_position,
                    )
                    decision = control_decision(
                        estimated_position,
                        target,
                        camera_yaw_enu_rad=estimated_camera_yaw,
                        config=self.config,
                        require_yaw_alignment=not aligned_for_translation,
                        velocity_enu_m_s=velocity_enu,
                    )
                    violation = safety_violation(
                        estimated_position,
                        plan,
                        self.config,
                        pose_age_s=pose_age_s,
                        confidence=confidence,
                    )
                    if violation is not None:
                        decision = replace(
                            decision,
                            phase="safety_hover",
                            command=PilotingCommand.zero(),
                        )
                        arrival_confirmation_count = 0
                        yaw_alignment.reset()
                        estimated_yaw_rate_deg_s = None
                        consecutive_lost_updates += 1
                        self.safety_events.append(
                            {
                                "elapsed_s": time.monotonic() - started_at,
                                "reason": violation,
                                "consecutive_updates": consecutive_lost_updates,
                            }
                        )
                    else:
                        consecutive_lost_updates = 0
                        previous_estimated_position = estimated_position
                        last_estimated_position = estimated_position
                        last_estimated_yaw = estimated_camera_yaw
                        last_estimation_error = estimation_error
                        alignment_required = (
                            not aligned_for_translation
                            and decision.horizontal_distance_m
                            > self.config.minimum_yaw_alignment_distance_m
                            and decision.phase in {"turn", "translate"}
                        )
                        if alignment_required:
                            alignment_confirmed, estimated_yaw_rate_deg_s = yaw_alignment.update(
                                yaw_error_rad=decision.camera_turn_angle_enu_rad,
                                estimated_yaw_rad=estimated_camera_yaw,
                                timestamp_s=estimated_position.timestamp_s,
                                config=self.config,
                            )
                            if decision.phase == "translate":
                                decision = replace(
                                    decision,
                                    phase=(
                                        "yaw_alignment_confirmed"
                                        if alignment_confirmed
                                        else "yaw_alignment_hold"
                                    ),
                                    command=PilotingCommand.zero(),
                                )
                                if alignment_confirmed:
                                    aligned_for_translation = True
                        elif (
                            not aligned_for_translation
                            and decision.horizontal_distance_m
                            <= self.config.minimum_yaw_alignment_distance_m
                        ):
                            aligned_for_translation = True
                    true_distance = math.dist(
                        (true_position.x_m, true_position.y_m, true_position.z_m),
                        (target.x_m, target.y_m, target.z_m),
                    )
                    if decision.phase == "hover":
                        arrival_confirmation_count += 1
                    else:
                        arrival_confirmation_count = 0
                    arrival_verified = (
                        arrival_confirmation_count >= self.config.arrival_confirmation_updates
                    )
                    self._record(
                        started_at=started_at,
                        true_position=true_position,
                        estimated_position=estimated_position,
                        target=target,
                        true_camera_yaw=true_camera_yaw,
                        estimated_camera_yaw=estimated_camera_yaw,
                        decision=decision,
                        estimation_error=estimation_error,
                        arrival_verified=arrival_verified,
                        pose_age_s=pose_age_s,
                        arrival_confirmation_count=arrival_confirmation_count,
                        yaw_alignment_confirmation_count=yaw_alignment.confirmation_count,
                        estimated_yaw_rate_deg_s=estimated_yaw_rate_deg_s,
                        yaw_alignment_confirmed=aligned_for_translation,
                        safety_state="nominal" if violation is None else violation,
                    )
                    duration = 0.0 if arrival_verified else self.config.pcmd_watchdog_s
                    if not drone.piloting(
                        decision.command.roll,
                        decision.command.pitch,
                        decision.command.yaw,
                        decision.command.gaz,
                        duration,
                    ):
                        raise RuntimeError("Olympe refused a route PCMD command")
                    piloting_started = True
                    previous_command_nonzero = decision.command != PilotingCommand.zero()
                    if violation in {"altitude_geofence", "route_deviation"}:
                        raise RuntimeError(f"flight terminated by safety gate: {violation}")
                    if (
                        violation is not None
                        and consecutive_lost_updates > self.config.maximum_consecutive_lost_updates
                    ):
                        raise RuntimeError(f"flight terminated after repeated {violation}")
                    if arrival_verified:
                        time.sleep(self.config.waypoint_hover_s)
                        self.arrivals.append(
                            {
                                "waypoint_index": target.index,
                                "elapsed_s": time.monotonic() - started_at,
                                "arrival_error_m": true_distance,
                                "estimated_arrival_error_m": decision.distance_m,
                                "estimated_arrival_speed_m_s": decision.estimated_speed_m_s,
                                "confirmation_updates": arrival_confirmation_count,
                                "position": asdict(true_position),
                                "estimated_position": asdict(estimated_position),
                            }
                        )
                        break
                    time.sleep(self.config.control_period_s)
                else:
                    raise TimeoutError(
                        f"waypoint {target.index + 1} was not reached within "
                        f"{self.config.waypoint_timeout_s:.0f} seconds"
                    )
            self.final_position = collector.latest_sample()
        finally:
            if before_landing is not None:
                with contextlib.suppress(Exception):
                    before_landing()
            if connected:
                with contextlib.suppress(Exception):
                    drone.piloting(0, 0, 0, 0, 0)
                if piloting_started:
                    with contextlib.suppress(Exception):
                        drone.stop_piloting()
                if airborne:
                    for attempt in range(1, 4):
                        self.landing_attempts = attempt
                        try:
                            landing = drone(
                                Landing(_no_expect=True)
                                & FlyingStateChanged(state="landed", _timeout=20)
                            ).wait()
                            flying_state = _arsdk_state_name(
                                drone.get_state(FlyingStateChanged).get("state")
                            )
                            if landing.success() or flying_state == "landed":
                                self.landed = True
                                self.landing_error = None
                                break
                            self.landing_error = (
                                "Olympe landing expectation failed "
                                f"(state={flying_state}, attempt={attempt})"
                            )
                        except Exception as error:  # noqa: BLE001 -- cleanup retains diagnostics.
                            self.landing_error = f"{type(error).__name__}: {error}"
                        if attempt < 3:
                            time.sleep(2.0)
                with contextlib.suppress(Exception):
                    drone.disconnect()
            collector.stop()
        if not self.landed:
            details = f": {self.landing_error}" if self.landing_error else ""
            raise RuntimeError(f"route completed but landing was not confirmed{details}")


def route_plan_payload(plan: RoutePlan, config: RouteConfig) -> dict[str, object]:
    first = plan.waypoints[0]
    first_from_spawn = math.dist(
        (plan.sphinx_spawn.x_m, plan.sphinx_spawn.y_m, plan.sphinx_spawn.z_m),
        (first.x_m, first.y_m, first.z_m),
    )
    return {
        "seed": plan.seed,
        "coordinate_frame": "Gazebo ENU metres; yaw is radians CCW from +X",
        "sphinx_ground_spawn": {
            **asdict(plan.sphinx_spawn),
            "sphinx_pose": plan.sphinx_spawn.sphinx_pose,
        },
        "first_waypoint_distance_from_ground_spawn_m": first_from_spawn,
        "waypoint_count": len(plan.waypoints),
        "waypoints": [asdict(waypoint) for waypoint in plan.waypoints],
        "segments": [
            {"from_waypoint_index": index - 1, "to_waypoint_index": index}
            for index in range(1, len(plan.waypoints))
        ],
        "aircraft_profile": asdict(ANAFI_4K_STANDARD_PROFILE),
        "controller": asdict(config),
        "integration_contract": {
            "localization_frame": config.frame.localization_frame,
            "map_scale_metres_per_unit": config.frame.metres_per_map_unit,
            "camera_to_body_translation_m": {
                "forward": config.frame.camera_to_body_forward_m,
                "right": config.frame.camera_to_body_right_m,
                "up": config.frame.camera_to_body_up_m,
            },
            "camera_yaw_to_body_yaw_deg": config.frame.camera_yaw_to_body_yaw_deg,
            "obstacle_clearance": (
                "not simulated: parrot-ue4-empty contains no obstacle course; production must "
                "provide an independent clearance monitor"
            ),
            "operator_override": "Ctrl+C causes zero PCMD, stops piloting, and requests landing",
        },
        "safety_policy": {
            "pose_age_and_confidence_failure": (
                "send zero PCMD; terminate after the configured consecutive-update limit"
            ),
            "altitude_or_route_geofence_failure": "send zero PCMD and terminate immediately",
            "arrival_confirmation": (
                "estimated 3D distance uses a position-error and latency-motion reserve, and "
                "estimated speed must pass for consecutive updates"
            ),
            "estimated_arrival_tolerance_m": estimated_arrival_tolerance_m(config),
            "pcmd_watchdog": ("each command expires before the next one-second feedback update"),
        },
        "control_policy": (
            "after takeoff, reconnect the camera centre to the target every second; "
            f"turn the camera's XY yaw to within {config.yaw_tolerance_deg:g} degrees and keep "
            f"its estimated yaw rate within {config.yaw_alignment_max_rate_deg_s:g} degrees per "
            f"second for {config.yaw_alignment_confirmation_updates} consecutive updates before "
            "translating; then recompute body-forward, body-right, and up PCMD components from "
            "the latest target line every second without re-entering yaw-only alignment; slow and "
            "brake using estimated velocity, then require three consecutive low-speed estimates "
            "inside a 0.15 metre controller radius that reserves the 0.5 metre true acceptance "
            "radius for bounded position error and latency motion; do not let an XY projection shorter than "
            "0.25 metres block vertical capture; calculate PCMD from a correlated, delayed, and "
            "occasionally dropped estimated navigation state while Sphinx applies bounded "
            "wind-like true-position disturbances; apply freshness, confidence, altitude, route "
            "deviation, battery, and PCMD-watchdog gates; "
            "land after the last"
        ),
    }


def write_route_plan(plan: RoutePlan, config: RouteConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "route_plan.json").write_text(
        json.dumps(route_plan_payload(plan, config), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    _write_route_svg(plan, output_dir / "route_preview.svg")


def write_route_artifacts(
    follower: OlympeRouteFollower,
    plan: RoutePlan,
    telemetry: TrueTelemetryCollector,
    output_dir: Path,
    *,
    status: str,
    error: str | None = None,
    wind_events: list[dict[str, float]] | None = None,
) -> None:
    samples = telemetry.samples()
    disturbances = wind_events or []
    report = {
        "status": status,
        "error": error,
        "stress_scenario": (
            asdict(follower.stress_scenario) if follower.stress_scenario is not None else None
        ),
        "completed_waypoint_count": len(follower.arrivals),
        "waypoint_count": len(plan.waypoints),
        "takeoff_attempts": follower.takeoff_attempts,
        "landing_attempts": follower.landing_attempts,
        "landing_error": follower.landing_error,
        "landed": follower.landed,
        "aircraft_profile": asdict(ANAFI_4K_STANDARD_PROFILE),
        "aircraft_settings_confirmed_by_firmware": follower.aircraft_settings,
        "initial_position": asdict(follower.initial_position)
        if follower.initial_position
        else None,
        "final_position": asdict(follower.final_position) if follower.final_position else None,
        "arrivals": follower.arrivals,
        "safety_events": follower.safety_events,
        "safety_event_count": len(follower.safety_events),
        "pcmd_record_count": len(follower.records),
        "true_trajectory_sample_count": len(samples),
        "wind_disturbance_event_count": len(disturbances),
        "maximum_observed_wind_displacement_m": (
            max(event["displacement_m"] for event in disturbances) if disturbances else 0.0
        ),
    }
    (output_dir / "route_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "pcmd_log.csv", follower.records)
    _write_csv(
        output_dir / "true_trajectory.csv",
        [asdict(sample) for sample in samples],
    )
    _write_csv(output_dir / "wind_disturbances.csv", disturbances)
    _write_csv(output_dir / "safety_events.csv", follower.safety_events)
    _write_route_svg(plan, output_dir / "route_result.svg", samples=samples)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_route_svg(
    plan: RoutePlan,
    path: Path,
    *,
    samples: tuple[TruePosition, ...] = (),
) -> None:
    width, height, margin = 800, 620, 60
    points = [(point.x_m, point.y_m) for point in plan.waypoints]
    extents = points + [(plan.sphinx_spawn.x_m, plan.sphinx_spawn.y_m)]
    if samples:
        extents.extend((sample.x_m, sample.y_m) for sample in samples)
    min_x = min(point[0] for point in extents)
    max_x = max(point[0] for point in extents)
    min_y = min(point[1] for point in extents)
    max_y = max(point[1] for point in extents)
    span = max(max_x - min_x, max_y - min_y, 1.0)

    def project(x_m: float, y_m: float) -> tuple[float, float]:
        x = margin + (x_m - min_x) / span * (width - 2 * margin)
        y = height - margin - (y_m - min_y) / span * (height - 2 * margin)
        return x, y

    route_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (project(*p) for p in points))
    start_x, start_y = project(plan.sphinx_spawn.x_m, plan.sphinx_spawn.y_m)
    first_x, first_y = project(plan.waypoints[0].x_m, plan.waypoints[0].y_m)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#101820"/>',
        '<text x="30" y="35" fill="white" font-size="22">Winding 10-waypoint route (top-down ENU)</text>',
        (
            f'<polyline points="{route_points}" fill="none" stroke="#55d6be" '
            'stroke-width="3" stroke-linejoin="round"/>'
        ),
        (
            f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{first_x:.1f}" '
            f'y2="{first_y:.1f}" stroke="#fb4d3d" stroke-width="2" '
            'stroke-dasharray="7 5"/>'
        ),
    ]
    if samples:
        actual = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in (project(sample.x_m, sample.y_m) for sample in samples)
        )
        lines.append(
            f'<polyline points="{actual}" fill="none" stroke="#ffb703" '
            'stroke-width="2" opacity="0.85"/>'
        )
    lines.append(f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="7" fill="#fb4d3d"/>')
    lines.append(
        f'<text x="{start_x + 10:.1f}" y="{start_y - 10:.1f}" fill="#fb4d3d" '
        'font-size="14">drone start XY</text>'
    )
    for waypoint in plan.waypoints:
        x, y = project(waypoint.x_m, waypoint.y_m)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#55d6be"/>')
        lines.append(
            f'<text x="{x + 9:.1f}" y="{y - 9:.1f}" fill="white" font-size="13">'
            f"P{waypoint.index + 1} z={waypoint.z_m:.2f}m</text>"
        )
    lines.append(
        '<text x="30" y="595" fill="#ffb703" font-size="13">orange: true trajectory</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
