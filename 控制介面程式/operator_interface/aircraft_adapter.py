"""Explicit aircraft composition, independent of live/offline stream selection.

Replace CURRENT_AIRCRAFT or inject an AircraftDefinition at the launcher seam.
Factories must not connect hardware: live connection remains an operator action.
The live backend owns its stream; the runtime owns an offline file stream.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend_contract import (
    AircraftCapabilities, ControlAction, ControlRequest, FrameSource, InterfaceMode,
    LegacyFrameSourceAdapter, LiveOperatorBackend, OperatorBackend, StreamOwner,
)
from operator_state import ANAFI, AircraftProfile, DroneState


ANAFI_LIVE_CAPABILITIES = AircraftCapabilities(frozenset({
    ControlAction.TAKEOFF, ControlAction.LAND, ControlAction.HOVER,
    ControlAction.MANUAL, ControlAction.PC_CONTROL, ControlAction.NUDGE_BEGIN,
    ControlAction.NUDGE_HEARTBEAT, ControlAction.NUDGE_END, ControlAction.NUDGE_CLEAR,
    ControlAction.NUDGE_VECTOR, ControlAction.EMERGENCY_STOP, ControlAction.LAND_NOW,
    ControlAction.APPLY_LIMITS, ControlAction.SET_AUTO_SPEED_LIMIT,
    ControlAction.START_LOCALIZATION, ControlAction.START_AUTO, ControlAction.BOOT_LOCK,
    ControlAction.GIMBAL_PITCH, ControlAction.ZOOM, ControlAction.CAMERA_RESET,
    ControlAction.RECORD_ARM, ControlAction.RECORD_DISARM, ControlAction.RECORD_START,
    ControlAction.RECORD_STOP, ControlAction.RECORD_QUALITY,
    ControlAction.DRONE_MAGNETOMETER_START, ControlAction.DRONE_MAGNETOMETER_CANCEL,
    ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
    ControlAction.SKYCONTROLLER_MAGNETOMETER_CANCEL,
}))
ANAFI_SIM_CAPABILITIES = AircraftCapabilities(frozenset({
    ControlAction.TAKEOFF, ControlAction.LAND, ControlAction.HOVER,
    ControlAction.MANUAL, ControlAction.NUDGE_BEGIN, ControlAction.NUDGE_END,
    ControlAction.NUDGE_CLEAR, ControlAction.NUDGE_VECTOR, ControlAction.EMERGENCY_STOP,
    ControlAction.LAND_NOW, ControlAction.SET_AUTO_SPEED_LIMIT,
    ControlAction.START_LOCALIZATION, ControlAction.START_AUTO, ControlAction.BOOT_LOCK,
    ControlAction.GIMBAL_PITCH, ControlAction.ZOOM, ControlAction.CAMERA_RESET,
    ControlAction.RECORD_ARM, ControlAction.RECORD_DISARM, ControlAction.RECORD_START,
    ControlAction.RECORD_STOP, ControlAction.RECORD_QUALITY,
}))


def control_support_error(backend: Any, request: ControlRequest) -> str | None:
    """Fail closed on undeclared operations, including unknown camera envelopes."""
    capabilities = getattr(backend, "capabilities", None)
    if not isinstance(capabilities, AircraftCapabilities) or not capabilities.supports(request.action):
        return "UNSUPPORTED_ACTION"
    profile = getattr(backend, "aircraft_profile", None)
    if request.action in {ControlAction.GIMBAL_PITCH, ControlAction.CAMERA_RESET}:
        bounds = (
            getattr(profile, "gimbal_pitch_min_deg", None),
            getattr(profile, "gimbal_pitch_max_deg", None),
        )
        if not _known_bounds(bounds):
            return "CAMERA_PITCH_BOUNDS_UNKNOWN"
    if request.action in {ControlAction.ZOOM, ControlAction.CAMERA_RESET}:
        bounds = (
            getattr(profile, "digital_zoom_min", None),
            getattr(profile, "digital_zoom_max", None),
        )
        if not _known_bounds(bounds):
            return "CAMERA_ZOOM_BOUNDS_UNKNOWN"
    return None


def _known_bounds(bounds: tuple[Any, Any]) -> bool:
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) for value in bounds
    ) and bounds[0] <= bounds[1]


@dataclass(frozen=True)
class AircraftBuildContext:
    args: Any
    session_logs: Any
    hardware_approval: Any
    profile: AircraftProfile
    capabilities: AircraftCapabilities
    mode: InterfaceMode


@dataclass(frozen=True)
class AircraftDefinition:
    """A local, explicit pair of constructors; there is no plugin discovery.

    Both constructors return an OperatorBackend with this exact profile and
    no capabilities beyond this declaration. A live implementation implements
    LiveOperatorBackend and owns video/video_stream; offline video is attached
    by build_aircraft_runtime and never opens the aircraft SDK.
    """

    profile: AircraftProfile
    simulation_factory: Callable[[AircraftBuildContext], OperatorBackend]
    simulation_capabilities: AircraftCapabilities = AircraftCapabilities()
    live_factory: Callable[[AircraftBuildContext], LiveOperatorBackend] | None = None
    live_capabilities: AircraftCapabilities = AircraftCapabilities()
    offline_link_config: Callable[[], dict | None] | None = None


def _approved_versions(approval: Any, attribute: str) -> tuple[str, ...]:
    if approval is None or not approval.approved:
        return ()
    return tuple(getattr(approval, attribute))


def _build_anafi_simulation(context: AircraftBuildContext) -> OperatorBackend:
    from simulated_backend import DroneBackend

    return DroneBackend(
        session_logs=context.session_logs, aircraft_profile=context.profile,
        capabilities=context.capabilities,
    )


def _build_anafi_live(context: AircraftBuildContext) -> LiveOperatorBackend:
    # Importing the composition module or building a simulator never imports SDK code.
    from olympe_live_backend import OlympeLiveBackend

    args, logs, approval = context.args, context.session_logs, context.hardware_approval
    cmd_log = logs.directory / "commands.jsonl"
    backend = OlympeLiveBackend(
        DroneState, context.profile,
        ip=args.ip,
        controller=args.controller,
        nudge_pct=args.nudge_pct,
        nudge_pulse_s=args.nudge_pulse_s,
        max_altitude_m=args.max_altitude_m,
        max_distance_m=args.max_distance_m,
        distance_geofence=bool(args.distance_geofence),
        min_takeoff_battery_pct=float(args.min_takeoff_battery_pct),
        max_tilt_deg=float(args.max_tilt_deg),
        max_vertical_speed_ms=float(args.max_vertical_speed_ms),
        max_rotation_speed_degs=float(args.max_rotation_speed_degs),
        max_pitch_roll_rotation_speed_degs=float(args.max_pitch_roll_rotation_speed_degs),
        rth_min_altitude_m=float(args.rth_min_altitude_m),
        auto_pc_control=bool(args.auto_pc_control),
        stream_loss_grace_s=float(args.stream_loss_grace_s),
        require_gps_for_geofence=bool(args.require_gps_for_geofence),
        approved_aircraft_firmware=_approved_versions(approval, "aircraft_firmware_versions"),
        approved_controller_firmware=_approved_versions(approval, "controller_firmware_versions"),
        approved_olympe_versions=_approved_versions(approval, "olympe_versions"),
        cmd_log=cmd_log,
        event_log=logs.command_log,
        session_logs=logs,
        with_video=not args.no_live_video,
        defer_connect=True,
    )
    print(f"[live] command log -> {cmd_log}", flush=True)
    return backend


def resolve_anafi_link_sim() -> dict | None:
    """Optional recorded-video link emulation, never a live stream replacement."""
    if os.environ.get("ANAFI_LINK_SIM", "0").strip() not in {"1", "true", "TRUE", "yes"}:
        return None
    try:
        kbps = int(float(os.environ.get("ANAFI_LINK_KBPS", "5000")))
    except ValueError:
        kbps = 5000
    if kbps <= 0:
        return None

    def number(name: str, default: float) -> float:
        try:
            return max(0.0, float(os.environ.get(name, default)))
        except ValueError:
            return default

    return {
        "kbps": kbps,
        "profile": os.environ.get("ANAFI_LINK_PROFILE", "main"),
        "intra_refresh": os.environ.get("ANAFI_LINK_INTRA_REFRESH", "1").strip()
        not in {"0", "false", "FALSE", "no"},
        "latency_ms": number("ANAFI_LINK_LATENCY_MS", 0.0),
        "loss_pct": number("ANAFI_LINK_LOSS_PCT", 0.0),
        "loss_seed": int(number("ANAFI_LINK_LOSS_SEED", 20260726)),
    }


CURRENT_AIRCRAFT = AircraftDefinition(
    profile=ANAFI,
    simulation_factory=_build_anafi_simulation,
    simulation_capabilities=ANAFI_SIM_CAPABILITIES,
    live_factory=_build_anafi_live,
    live_capabilities=ANAFI_LIVE_CAPABILITIES,
    offline_link_config=resolve_anafi_link_sim,
)


def _validate_profile(profile: AircraftProfile) -> None:
    if not profile.aircraft_id or not profile.model:
        raise ValueError("aircraft definition requires an explicit model identity")
    if profile.stream_width <= 0 or profile.stream_height <= 0 or not (
        math.isfinite(profile.stream_fps) and profile.stream_fps > 0.0
    ):
        raise ValueError("aircraft definition requires a valid stream geometry/rate")


def _validate_backend(backend, context: AircraftBuildContext) -> None:
    if not isinstance(backend, OperatorBackend):
        raise TypeError("aircraft factory did not implement OperatorBackend")
    live = context.mode is InterfaceMode.REAL_FLIGHT
    if backend.mode is not context.mode or backend.is_live is not live:
        raise ValueError("aircraft backend/interface mode mismatch")
    if backend.aircraft_profile != context.profile or not (
        isinstance(backend.capabilities, AircraftCapabilities)
        and backend.capabilities.actions <= context.capabilities.actions
    ):
        raise ValueError("aircraft backend/model capability mismatch")
    owner = StreamOwner.BACKEND if live else StreamOwner.RUNTIME
    if backend.stream_owner is not owner:
        raise ValueError("aircraft stream ownership mismatch")


def _offline_stream(backend, args, definition: AircraftDefinition):
    if backend.video is not None:
        raise ValueError("offline source must be composed by the runtime")
    if not args.video:
        return None
    from ffmpeg_frame_stream import FFmpegFrameStream

    profile = definition.profile
    video_stream = FFmpegFrameStream(
        Path(args.video), profile.stream_width, profile.stream_height,
        stride=args.video_stride, fps=profile.stream_fps,
        link_sim=(definition.offline_link_config() if definition.offline_link_config else None),
    )
    backend.video = LegacyFrameSourceAdapter(video_stream, str(Path(args.video).resolve()))
    return video_stream


def _live_runtime(backend):
    if not isinstance(backend, LiveOperatorBackend):
        raise TypeError("live aircraft does not implement the operational live port")
    if backend.video is not None and not isinstance(backend.video, FrameSource):
        raise TypeError("live aircraft source does not implement FrameSource")
    return backend, backend.video_stream, backend


def build_aircraft_runtime(
    args: Any, session_logs: Any, hardware_approval: Any, *,
    definition: AircraftDefinition | None = None,
) -> tuple[OperatorBackend, Any | None, LiveOperatorBackend | None]:
    """Compose one aircraft and one source, rejecting model/mode/source mismatches.

    Offline streams are runtime-owned, matching the existing window shutdown
    coordinator. Live sources are backend-owned and must be closed only through
    its protected cleanup policy. No construction path connects hardware.
    """
    definition = CURRENT_AIRCRAFT if definition is None else definition
    profile = definition.profile
    _validate_profile(profile)
    live = bool(args.live)
    if live and args.video:
        raise ValueError("live aircraft cannot use an offline video source")
    mode = InterfaceMode.REAL_FLIGHT if live else InterfaceMode.SIMULATED_STREAM
    factory = definition.live_factory if live else definition.simulation_factory
    if factory is None:
        raise ValueError("aircraft does not support the selected stream mode")
    capabilities = definition.live_capabilities if live else definition.simulation_capabilities
    context = AircraftBuildContext(args, session_logs, hardware_approval, profile, capabilities, mode)
    backend = factory(context)
    _validate_backend(backend, context)
    if live:
        return _live_runtime(backend)
    return backend, _offline_stream(backend, args, definition), None
