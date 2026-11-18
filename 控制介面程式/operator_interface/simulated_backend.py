"""Offline simulation backend for the desktop operator app.

``DroneBackend`` is the SIMULATED_STREAM backend: it moves a point in memory and
never opens an Olympe/ARSDK connection, so nothing here can arm or fly hardware.
The live counterpart (same method surface) is ``olympe_live_backend``.
"""

from __future__ import annotations

import math
import time

import numpy as np

from backend_contract import (
    AircraftCapabilities,
    CloseResult,
    ControlAction,
    ControlRequest,
    ControlResult,
    FailureReason,
    InterfaceMode,
    RuntimeSwitchKind,
    RuntimeSwitchStatus,
    SessionConfig,
    StartResult,
    StreamOwner,
    control_request_is_current,
    execute_admitted_control,
    gimbal_observation,
)
from operator_state import ANAFI, AircraftProfile, DroneState, TrackerState
from aircraft_adapter import ANAFI_SIM_CAPABILITIES, control_support_error
from recording_quality import (
    DEFAULT_RECORDING_PROFILE,
    format_record_status,
    resolve_recording_profile,
)
from runtime_safety import SessionLogs
from scale_free_control_adapter import validate_speed_limit_change
from simulated_route_test import SimulatedRoutePlant

# Typed requests are submitted immediately before the backend boundary.  Keep
# the same upper bound as the flight-control command TTL: a queued ordinary
# request must never become a delayed movement command.  Explicit safety actions
# below are exempt so an old emergency/landing request remains actionable.
CONTROL_REQUEST_MAX_AGE_NS = 250_000_000


class DroneBackend:
    """In-memory aircraft simulation; never connects to a hardware SDK."""

    mode = InterfaceMode.SIMULATED_STREAM
    is_live = False
    stream_owner = StreamOwner.RUNTIME

    def __init__(
        self, session_logs: SessionLogs | None = None, *,
        aircraft_profile: AircraftProfile = ANAFI,
        capabilities: AircraftCapabilities | None = None,
    ):
        self.aircraft_profile = aircraft_profile
        self.capabilities = (
            ANAFI_SIM_CAPABILITIES if capabilities is None and aircraft_profile == ANAFI
            else AircraftCapabilities() if capabilities is None else capabilities
        )
        self.state = DroneState()
        self.state.aircraft_identity = f"SIMULATED {aircraft_profile.model}"
        self.state.controller_identity = "SIMULATED CONTROLLER"
        self.state.link_latency_ms = aircraft_profile.stream_latency_ms
        self.state.stream_fps = aircraft_profile.stream_fps
        self.state.stream_mbps = aircraft_profile.stream_mbps
        if aircraft_profile.default_gimbal_pitch_deg is not None:
            self.state.gimbal_pitch_deg = aircraft_profile.default_gimbal_pitch_deg
            self.state.gimbal_target_pitch_deg = aircraft_profile.default_gimbal_pitch_deg
        if aircraft_profile.default_zoom is not None:
            self.state.zoom = aircraft_profile.default_zoom
        self.video = None
        self.session_config: SessionConfig | None = None
        self.session_logs = session_logs
        self.started_mono = time.monotonic()
        self.last_poll = self.started_mono
        self.sim_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
        self.sim_yaw = 0.0
        self.target_altitude_m = 0.0
        self.flight_start: float | None = None
        self.route_test_plant = SimulatedRoutePlant(self)
        self.record_on_takeoff = False
        self.recording_active = False
        self.recording_profile = DEFAULT_RECORDING_PROFILE
        self.record_status = format_record_status(
            active=False,
            armed=False,
            profile=self.recording_profile,
        )

    @property
    def connection_target(self) -> str:
        return f"offline:{self.aircraft_profile.aircraft_id}"

    def flight_state_name(self) -> str:
        return "landed" if self.target_altitude_m <= 0.0 and self.state.altitude_m <= 0.0 else "flying"

    def runtime_switch_status(self) -> RuntimeSwitchStatus:
        return RuntimeSwitchStatus(
            RuntimeSwitchKind.NEVER_CONNECTED, "offline simulation has no hardware link",
            self.connection_target, self.flight_state_name(),
        )

    def pilot_override_active(self) -> bool:
        return False

    @property
    def maneuver_in_progress(self) -> str | None:
        return None

    @property
    def runtime_safety_latched(self) -> bool:
        return False

    def gimbal_pitch_sample(self) -> tuple[float, int] | None:
        if self.state.gimbal_stamp_source != "simulated":
            return None
        return gimbal_observation(self.state.gimbal_pitch_deg, self.state.gimbal_mono_ns)

    def set_tracker_state(self, new: TrackerState, *, reason: str) -> TrackerState:
        """Centralized tracker_state transition (F-11).

        Keeps ``str`` compatibility (stores ``new.value``) while logging
        old/new/reason for audit. Returns ``new`` so callers can chain.
        """
        old = self.state.tracker_state
        self.state.tracker_state = new.value
        if self.session_logs is not None:
            try:
                self.session_logs.command(
                    "tracker_state_transition",
                    old=old,
                    new=new.value,
                    reason=reason,
                )
            except (OSError, ValueError, AttributeError, TypeError, RuntimeError):  # Tier3: fallback best-effort — narrow, keep pass
                pass
        return new

    def start(self, config: SessionConfig) -> StartResult:
        if config.interface_mode is not self.mode:
            return StartResult(False, "INTERFACE_MISMATCH")
        if self.session_config is not None and self.session_config != config:
            return StartResult(False, "HOT_SWITCH_PROHIBITED")
        if self.session_config is None:
            self.state.autonomous_speed_limit_mps = config.autonomous_speed_limit_mps
        self.session_config = config
        return StartResult(True, "OK")

    def _typed_command_gate(self, request: ControlRequest) -> ControlResult | None:
        if not control_request_is_current():
            return ControlResult.rejected("CONTROL_REQUEST_CANCELLED", self.state)
        if request.action is ControlAction.TAKEOFF and not request.human_origin:
            return ControlResult.rejected("HUMAN_ORIGIN_REQUIRED", self.state)
        calibration_starts = {
            ControlAction.DRONE_MAGNETOMETER_START,
            ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
        }
        calibration_actions = calibration_starts | {
            ControlAction.DRONE_MAGNETOMETER_CANCEL,
            ControlAction.SKYCONTROLLER_MAGNETOMETER_CANCEL,
        }
        if request.action in calibration_starts and not request.human_origin:
            return ControlResult.rejected("HUMAN_ORIGIN_REQUIRED", self.state)
        if request.action not in {
            ControlAction.LAND_NOW,
            ControlAction.EMERGENCY_STOP,
        }:
            age_ns = time.monotonic_ns() - int(request.submitted_mono_ns)
            if age_ns > CONTROL_REQUEST_MAX_AGE_NS:
                if self.session_logs is not None:
                    self.session_logs.incident(
                        "control_request_stale",
                        request_id=request.request_id,
                        action=request.action.value,
                        age_ms=age_ns / 1_000_000.0,
                    )
                return ControlResult.rejected("STALE_CONTROL_REQUEST", self.state)
        if request.action in calibration_actions:
            return ControlResult.rejected("LIVE_HARDWARE_REQUIRED", self.state)
        support_error = control_support_error(self, request)
        if support_error is not None:
            return ControlResult.rejected(support_error, self.state)
        return None

    def _run_typed_action(self, request: ControlRequest) -> ControlResult | None:
        if request.action is ControlAction.EMERGENCY_STOP:
            return self.fail_safe(FailureReason.EMERGENCY_STOP)
        if request.action is ControlAction.LAND_NOW:
            self.command("land")
            return ControlResult.completed(self.state, raw_result=True)
        if request.action is ControlAction.TAKEOFF:
            self.set_tracker_state(TrackerState.TAKEOFF, reason="takeoff_typed")
            self.target_altitude_m = self.aircraft_profile.takeoff_hover_m
            if self.flight_start is None:
                self.flight_start = time.monotonic()
            return ControlResult.completed(self.state, raw_result=True)
        if request.action is ControlAction.START_LOCALIZATION:
            self.state.loc = "STARTING"
            self.state.last_command = request.action.value
            return ControlResult.completed(self.state, raw_result=True)
        return None

    def _typed_command(self, request: ControlRequest) -> ControlResult:
        rejected = self._typed_command_gate(request)
        if rejected is not None:
            return rejected
        handled = self._run_typed_action(request)
        if handled is not None:
            return handled
        name, payload = request.legacy_call()
        raw = self.command(name, **payload)
        if isinstance(raw, ControlResult):
            explicit = raw.accepted and raw.executed
        elif isinstance(raw, bool):
            explicit = raw
        else:
            explicit = raw is self.state
        return ControlResult.completed(self.state, raw_result=explicit)

    def _command_typed_request(self, request: ControlRequest) -> ControlResult:
        result = execute_admitted_control(request, self.state, self._typed_command)
        if self.session_logs is not None:
            self.session_logs.command(
                "control_request",
                request_id=request.request_id,
                action=request.action.value,
                human_origin=request.human_origin,
                accepted=result.accepted,
                executed=result.executed,
                reason_code=result.reason_code,
            )
        return result

    def _reject_legacy_takeoff(self, name: str) -> ControlResult | None:
        if name == "takeoff":
            # Legacy strings remain compatible for non-flight actions, but a
            # takeoff must carry the typed human-origin proof all the way to the
            # backend boundary.
            if self.session_logs is not None:
                self.session_logs.incident(
                    "legacy_takeoff_rejected",
                    reason="typed_request_required",
                )
            return ControlResult.rejected("TYPED_TAKEOFF_REQUIRED", self.state)
        return None

    def _set_legacy_mode(self, name: str) -> None:
        if name in {"manual", "hover", "land"}:
            self.state.mode = "MANUAL"
        elif name in {"auto", "start_auto"}:
            self.state.mode = "AUTO"
            self.target_altitude_m = max(self.target_altitude_m, self.aircraft_profile.takeoff_hover_m)
            if self.flight_start is None:
                self.flight_start = time.monotonic()

    def _legacy_state_action(self, name: str, payload: dict) -> DroneState | None:
        if name == "hover":
            self.state.tracker_state = TrackerState.HOVER
            self.target_altitude_m = max(0.0, -float(self.sim_xyz[1]))
        elif name == "land":
            self.state.tracker_state = TrackerState.LAND
            self.target_altitude_m = 0.0
        elif name == "boot_lock":
            self.state.tracker_state = TrackerState.BOOT_INIT
        elif name == "gimbal_pitch":
            self.state.gimbal_pitch_deg = float(
                np.clip(
                    float(payload.get("pitch", self.state.gimbal_pitch_deg)),
                    self.aircraft_profile.gimbal_pitch_min_deg,
                    self.aircraft_profile.gimbal_pitch_max_deg,
                )
            )
            self.state.gimbal_target_pitch_deg = self.state.gimbal_pitch_deg
        elif name == "zoom":
            self.state.zoom = float(
                np.clip(
                    float(payload.get("zoom", self.state.zoom)),
                    self.aircraft_profile.digital_zoom_min,
                    self.aircraft_profile.digital_zoom_max,
                )
            )
        elif name in {"camera_reset", "reset_camera", "鏡頭預設", "回復預設"}:
            # UI defaults: slight look-down + 1.0x (DroneState initial values).
            self.state.gimbal_pitch_deg = self.aircraft_profile.default_gimbal_pitch_deg
            self.state.zoom = self.aircraft_profile.default_zoom
        else:
            return None
        return self.state

    def _legacy_record_action(self, name: str, payload: dict) -> DroneState | bool | None:
        if name not in {
            "record_arm",
            "record_on_takeoff",
            "record_disarm",
            "record_start",
            "record_stop",
            "record_quality",
        }:
            return None
        if name == "record_quality":
            if self.recording_active:
                return False
            try:
                self.recording_profile = resolve_recording_profile(
                    payload.get("profile_id"),
                )
            except ValueError:
                return False
        elif name == "record_disarm":
            self.record_on_takeoff = False
        elif name in {"record_arm", "record_on_takeoff"}:
            en = payload.get("enabled", payload.get("value", True))
            if isinstance(en, str):
                en = en.lower() in {"1", "true", "yes", "on"}
            self.record_on_takeoff = bool(en)
        elif name == "record_start":
            self.recording_active = True
        elif name == "record_stop":
            self.recording_active = False
        self.record_status = format_record_status(
            active=self.recording_active,
            armed=self.record_on_takeoff,
            profile=self.recording_profile,
        )
        return self.state

    def _legacy_nudge_action(self, name: str, payload: dict) -> DroneState | None:
        if name in {"nudge_begin", "nudge_press"}:
            d = str(payload.get("dir") or payload.get("name") or "")
            if d:
                self._apply_nudge(d, payload)
        elif name == "nudge_vector":
            self._apply_nudge_vector(payload)
        elif name in {"nudge_end", "nudge_release", "nudge_clear"}:
            self.state.tracker_state = TrackerState.HOVER
            self.state.last_command = "hover"
        else:
            return None
        return self.state

    def _legacy_one_shot_nudge_action(
        self,
        name: str,
        payload: dict,
    ) -> DroneState | None:
        if not (
            name.startswith("nudge_")
            or name
            in {
                "右上前",
                "左上前",
                "右下前",
                "左下前",
                "右上後",
                "左上後",
                "右下後",
                "左下後",
                "前",
                "後",
                "左",
                "右",
                "上",
                "下",
            }
        ):
            return None
        # Legacy one-shot sim step.
        self._apply_nudge(name if not name.startswith("nudge_") else name[6:], payload)
        return self.state

    def _legacy_safety_action(self, name: str, payload: dict) -> object | None:
        if name == "emergency_stop":
            return self.fail_safe(FailureReason.EMERGENCY_STOP)
        if name != "auto_speed_limit_apply":
            return None
        requested = float(payload.get("speed_limit_mps", 0.0))
        enabled = payload.get("enabled", self.state.autonomous_speed_limit_enabled)
        if not isinstance(enabled, bool):
            return False
        landed = self.target_altitude_m <= 0.0 and self.state.altitude_m <= 0.0
        change = validate_speed_limit_change(
            self.state.autonomous_speed_limit_mps,
            requested,
            landed=landed,
        )
        if not change.accepted:
            return False
        enabled_changed = enabled != self.state.autonomous_speed_limit_enabled
        self.state.autonomous_speed_limit_enabled = enabled
        self.state.autonomous_speed_limit_mps = change.new_speed_limit_mps
        self.state.autonomous_speed_guard_status = (
            "SPEED_WAITING" if enabled else "SPEED_LIMIT_DISABLED"
        )
        if change.approval_invalidated or enabled_changed:
            self.state.autonomous_approval_valid = False
        return True

    def _command_legacy(self, name: str, payload: dict) -> DroneState | ControlResult:
        rejected = self._reject_legacy_takeoff(name)
        if rejected is not None:
            return rejected
        self.state.last_command = name
        self._set_legacy_mode(name)
        result = self._legacy_state_action(name, payload)
        if result is None:
            result = self._legacy_record_action(name, payload)
        if result is None:
            result = self._legacy_nudge_action(name, payload)
        if result is None:
            result = self._legacy_safety_action(name, payload)
        if result is None:
            result = self._legacy_one_shot_nudge_action(name, payload)
        return self.state if result is None else result

    def command(
        self,
        name: str | ControlRequest,
        **payload,
    ) -> DroneState | ControlResult:
        if isinstance(name, ControlRequest):
            return self._command_typed_request(name)
        return self._command_legacy(name, payload)

    def fail_safe(self, reason: FailureReason) -> ControlResult:
        self.target_altitude_m = max(0.0, -float(self.sim_xyz[1]))
        self.state.mode = "MANUAL"
        self.state.control_owner = "SIM_MANUAL"
        self.set_tracker_state(TrackerState.FAIL_SAFE_HOVER, reason=f"fail_safe:{reason.value}")
        self.state.last_command = f"fail_safe:{reason.value}"
        self.state.active_incident = reason.value
        if self.session_logs is not None:
            self.session_logs.incident(
                "fail_safe",
                reason=reason.value,
                action="sim_hover_manual",
                auto_resume=False,
            )
        return ControlResult.completed(
            self.state,
            raw_result=True,
            reason_code="HOVER_MANUAL_HANDOFF",
        )

    def close(self, reason: str) -> CloseResult:
        self.state.mode = "CLOSED"
        self.state.last_command = f"close:{reason}"
        return CloseResult(True, "OK")

    def _apply_nudge_vector(self, payload: dict) -> None:
        """Continuous stick input, reusing the discrete nudge geometry."""
        step = float(payload.get("step_m", 0.25))
        roll = float(payload.get("roll", 0.0))
        pitch = float(payload.get("pitch", 0.0))
        gaz = float(payload.get("gaz", 0.0))
        c, s = math.cos(self.sim_yaw), math.sin(self.sim_yaw)
        self.sim_xyz[0] += (pitch * c - roll * s) * step
        self.sim_xyz[2] += (pitch * s + roll * c) * step
        self.sim_xyz[1] -= gaz * step
        self.sim_yaw += float(payload.get("yaw", 0.0)) * 0.1
        self.target_altitude_m = max(0.0, -float(self.sim_xyz[1]))
        self.state.tracker_state = TrackerState.NUDGE
        self.state.last_command = "nudge:vector"

    def _apply_nudge(self, name: str, payload: dict) -> None:
        """Map named diagonal/cardinal nudges onto sim_xyz (GLOMAP-like: -Y up)."""
        # unit body axes: +x right, +z forward, +y up in UI altitude space
        table = {
            "右上前": (+1, +1, +1),
            "左上前": (-1, +1, +1),
            "右下前": (+1, -1, +1),
            "左下前": (-1, -1, +1),
            "右上後": (+1, +1, -1),
            "左上後": (-1, +1, -1),
            "右下後": (+1, -1, -1),
            "左下後": (-1, -1, -1),
            "前": (0, 0, +1),
            "後": (0, 0, -1),
            "左": (-1, 0, 0),
            "右": (+1, 0, 0),
            "上": (0, +1, 0),
            "下": (0, -1, 0),
        }
        if name not in table:
            return
        step = float(payload.get("step_m", 0.25))
        rx, uy, fz = table[name]
        # body -> world using sim_yaw (0 = +X); treat "前" as +heading in XZ
        c, s = math.cos(self.sim_yaw), math.sin(self.sim_yaw)
        # forward along yaw in X/Z, right is perpendicular
        dx = (fz * c - rx * s) * step
        dz = (fz * s + rx * c) * step
        # sim uses -Y as up altitude
        self.sim_xyz[0] += dx
        self.sim_xyz[2] += dz
        self.sim_xyz[1] -= uy * step
        self.target_altitude_m = max(0.0, -float(self.sim_xyz[1]))
        self.state.tracker_state = TrackerState.NUDGE
        self.state.last_command = f"nudge:{name}"

    def stream_lost_hover(self, detail: str = "stream lost") -> DroneState:
        self.set_tracker_state(TrackerState.STREAM_LOST_HOVER, reason="stream_lost_hover")
        self.state.loc = "STREAM_LOST"
        self.state.stream = "LOST"
        self.state.last_command = f"hover: {detail}"
        return self.state

    def _log_sim_session_telemetry(self, now: float) -> None:
        if self.session_logs is not None:
            last_log = getattr(self, "_last_session_telemetry_t", 0.0)
            if now - last_log >= 1.0:
                self._last_session_telemetry_t = now
                self.session_logs.telemetry(
                    "sim_state",
                    mode=self.state.mode,
                    tracker_state=self.state.tracker_state,
                    pose=self.sim_xyz.tolist(),
                    battery_pct=self.state.battery_pct,
                )

    def _advance_sim_orbit(self, t: float, dt: float) -> None:
        orbit_r = 2.0
        cruise_speed = 1.5
        omega = cruise_speed / orbit_r
        target = np.array(
            [
                math.sin(t * omega) * orbit_r,
                self.sim_xyz[1],
                math.cos(t * omega) * orbit_r,
            ],
            dtype=float,
        )
        delta = target - self.sim_xyz
        delta[1] = 0.0
        dist = float(np.linalg.norm(delta))
        max_step = self.aircraft_profile.max_horizontal_speed_mps * dt
        if dist > 1e-6:
            self.sim_xyz += delta / dist * min(dist, max_step)
            target_yaw = math.atan2(float(delta[2]), float(delta[0]))
        else:
            target_yaw = self.sim_yaw
        yaw_delta = (target_yaw - self.sim_yaw + math.pi) % (2.0 * math.pi) - math.pi
        max_yaw = math.radians(self.aircraft_profile.max_yaw_rate_dps) * dt
        if abs(yaw_delta) > max_yaw and max_yaw > 0:
            yaw_delta = math.copysign(max_yaw, yaw_delta)
        self.sim_yaw = (self.sim_yaw + yaw_delta + math.pi) % (2.0 * math.pi) - math.pi

    def _advance_sim_motion(self, t: float, dt: float) -> None:
        if self.state.tracker_state == "LAND":
            self.target_altitude_m = 0.0
        elif self.state.mode == "AUTO":
            self.target_altitude_m = max(self.target_altitude_m, self.aircraft_profile.takeoff_hover_m)

        target_y = -self.target_altitude_m
        dy = float(target_y - self.sim_xyz[1])
        max_dy = self.aircraft_profile.max_vertical_speed_mps * dt
        if abs(dy) > max_dy and max_dy > 0:
            dy = math.copysign(max_dy, dy)
        self.sim_xyz[1] += dy

        if self.state.mode == "AUTO" and not self.route_test_plant.active:
            self._advance_sim_orbit(t, dt)

    def _apply_sim_tracker_state(self) -> bool:
        self.state.pose[:] = [self.sim_xyz[0], self.sim_xyz[1], self.sim_xyz[2], self.sim_yaw]
        self.state.loc = "SIM"
        if self.state.stream == "LOST":
            self.state.tracker_state = TrackerState.STREAM_LOST_HOVER
            return True
        if self.state.mode == "AUTO":
            self.set_tracker_state(TrackerState.TRACK, reason="sim_auto_track")
        elif (
            self.state.tracker_state == "TAKEOFF"
            and abs(self.sim_xyz[1] + self.aircraft_profile.takeoff_hover_m) < 0.03
        ):
            self.state.tracker_state = TrackerState.HOVER
        elif self.state.tracker_state == "LAND" and abs(self.sim_xyz[1]) < 0.03:
            self.sim_xyz[1] = 0.0
            self.target_altitude_m = 0.0
            self.state.tracker_state = TrackerState.HOVER
        elif self.state.tracker_state not in {"LAND", "TAKEOFF", "BOOT_INIT"}:
            self.state.tracker_state = TrackerState.HOVER
        return False

    def _update_sim_telemetry(self, now: float, t: float) -> None:
        self.state.inliers = 180 + int(40 * math.sin(t))
        self.state.reproj = 2.5 + 0.2 * math.cos(t * 0.5)
        elapsed_flight = 0.0 if self.flight_start is None else max(0.0, now - self.flight_start)
        self.state.battery_pct = float(
            np.clip(100.0 * (1.0 - elapsed_flight / self.aircraft_profile.flight_time_s), 0.0, 100.0)
        )
        self.state.altitude_m = max(0.0, -float(self.sim_xyz[1]))
        self.state.link_latency_ms = self.aircraft_profile.stream_latency_ms
        self.state.stream_fps = self.aircraft_profile.stream_fps
        self.state.stream_mbps = self.aircraft_profile.stream_mbps

        self.state.att_yaw = float(self.sim_yaw)

    def poll(self, now_mono_ns: int | None = None) -> DroneState:
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self.last_poll))
        self.last_poll = now
        t = now - self.started_mono
        # Simulated camera attitude observations, never labeled as live events.
        self.state.gimbal_mono_ns = int(now * 1e9)
        self.state.gimbal_stamp_source = "simulated"
        self._log_sim_session_telemetry(now)
        self._advance_sim_motion(t, dt)
        if self._apply_sim_tracker_state():
            return self.state
        self._update_sim_telemetry(now, t)
        return self.state
