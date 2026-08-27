"""Typed in-process contract shared by simulated and real operator backends.

The existing string command API remains as a compatibility shim for older tests and
tools. New operator code must construct :class:`ControlRequest` so unknown actions,
missing payloads, and non-human takeoff requests are rejected before dispatch.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


class InterfaceMode(str, Enum):
    SIMULATED_STREAM = "simulated-stream"
    REAL_FLIGHT = "real-flight"


class ControlAction(str, Enum):
    TAKEOFF = "takeoff"
    LAND = "land"
    HOVER = "hover"
    MANUAL = "manual"
    PC_CONTROL = "pc_control"
    NUDGE_BEGIN = "nudge_begin"
    NUDGE_HEARTBEAT = "nudge_heartbeat"
    NUDGE_END = "nudge_end"
    NUDGE_CLEAR = "nudge_clear"
    NUDGE_VECTOR = "nudge_vector"
    EMERGENCY_STOP = "emergency_stop"
    LAND_NOW = "land_now"
    APPLY_LIMITS = "apply_limits"
    SET_AUTO_SPEED_LIMIT = "set_auto_speed_limit"
    START_LOCALIZATION = "start_localization"
    START_AUTO = "start_auto"
    BOOT_LOCK = "boot_lock"
    GIMBAL_PITCH = "gimbal_pitch"
    ZOOM = "zoom"
    CAMERA_RESET = "camera_reset"
    RECORD_ARM = "record_arm"
    RECORD_DISARM = "record_disarm"
    RECORD_START = "record_start"
    RECORD_STOP = "record_stop"
    RECORD_QUALITY = "record_quality"
    DRONE_MAGNETOMETER_START = "drone_magnetometer_start"
    DRONE_MAGNETOMETER_CANCEL = "drone_magnetometer_cancel"
    SKYCONTROLLER_MAGNETOMETER_START = "skycontroller_magnetometer_start"
    SKYCONTROLLER_MAGNETOMETER_CANCEL = "skycontroller_magnetometer_cancel"


class FailureReason(str, Enum):
    STREAM_STALE = "stream_stale"
    LOCALIZATION_WEAK = "localization_weak"
    LOCALIZATION_LOST = "localization_lost"
    POSE_STALE = "pose_stale"
    WORKER_EXIT = "worker_exit"
    WORKER_STALL = "worker_stall"
    UI_HEARTBEAT_LOST = "ui_heartbeat_lost"
    CONTROL_LINK_LOST = "control_link_lost"
    CONTROLLER_DISCONNECTED = "controller_disconnected"
    BATTERY_CRITICAL = "battery_critical"
    ALTITUDE_LIMIT = "altitude_limit"
    DISTANCE_LIMIT = "distance_limit"
    INVALID_TELEMETRY = "invalid_telemetry"
    DISK_OR_LOG_FAILURE = "disk_or_log_failure"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


class InvalidControlRequest(ValueError):
    pass


def _finite_control_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidControlRequest(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise InvalidControlRequest(f"{label} must be finite")
    return parsed


@dataclass(frozen=True)
class EmptyPayload:
    pass


@dataclass(frozen=True)
class NudgePayload:
    direction: str

    def __post_init__(self) -> None:
        if not self.direction.strip():
            raise InvalidControlRequest("nudge direction is required")


@dataclass(frozen=True)
class NudgeHeartbeatPayload:
    directions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.directions or any(not value.strip() for value in self.directions):
            raise InvalidControlRequest("active nudge directions are required")


@dataclass(frozen=True)
class NudgeVectorPayload:
    """Continuous on-screen stick deflection, one unit value per axis.

    Values are fractions of the same nudge authority the discrete buttons use,
    so a dragged stick can never command more than a button press could.
    """

    roll: float
    pitch: float
    yaw: float
    gaz: float

    def __post_init__(self) -> None:
        for name in ("roll", "pitch", "yaw", "gaz"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidControlRequest(f"nudge vector {name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise InvalidControlRequest(f"nudge vector {name} must be finite")
            if not -1.0 <= value <= 1.0:
                raise InvalidControlRequest(
                    f"nudge vector {name} must be within [-1, 1]"
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LimitsPayload:
    max_altitude_m: float
    max_distance_m: float
    distance_geofence: bool = True

    def __post_init__(self) -> None:
        altitude = _finite_control_float(
            self.max_altitude_m, "max_altitude_m"
        )
        distance = _finite_control_float(
            self.max_distance_m, "max_distance_m"
        )
        if altitude <= 0.0 or distance <= 0.0:
            raise InvalidControlRequest("firmware limits must be positive")
        if not isinstance(self.distance_geofence, bool):
            raise InvalidControlRequest("distance_geofence must be boolean")
        object.__setattr__(self, "max_altitude_m", altitude)
        object.__setattr__(self, "max_distance_m", distance)


@dataclass(frozen=True)
class ScalarPayload:
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value", _finite_control_float(self.value, "scalar value")
        )


@dataclass(frozen=True)
class SpeedLimitPayload:
    speed_limit_mps: float
    enabled: bool = True

    def __post_init__(self) -> None:
        value = _finite_control_float(self.speed_limit_mps, "speed_limit_mps")
        if not isinstance(self.enabled, bool):
            raise InvalidControlRequest("speed limit enabled must be boolean")
        object.__setattr__(self, "speed_limit_mps", value)


@dataclass(frozen=True)
class TogglePayload:
    enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise InvalidControlRequest("toggle enabled must be boolean")


@dataclass(frozen=True)
class CameraResetPayload:
    pitch: float
    zoom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pitch", _finite_control_float(self.pitch, "camera pitch")
        )
        object.__setattr__(
            self, "zoom", _finite_control_float(self.zoom, "camera zoom")
        )


@dataclass(frozen=True)
class RecordingQualityPayload:
    profile_id: str

    def __post_init__(self) -> None:
        from recording_quality import resolve_recording_profile

        try:
            profile = resolve_recording_profile(self.profile_id)
        except ValueError as exc:
            raise InvalidControlRequest(str(exc)) from exc
        object.__setattr__(self, "profile_id", profile.profile_id)


@dataclass(frozen=True)
class MissionRoutePayload:
    route_path: str
    route_sha256: str
    site_id: str
    coordinate_frame_id: str

    def __post_init__(self) -> None:
        path = Path(str(self.route_path or ""))
        if not path.is_absolute():
            raise InvalidControlRequest("start_auto route_path must be absolute")
        digest = str(self.route_sha256 or "").strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise InvalidControlRequest(
                "start_auto route_sha256 must be 64 hexadecimal characters"
            )
        site_id = str(self.site_id or "").strip()
        coordinate_frame_id = str(self.coordinate_frame_id or "").strip()
        if not site_id or not coordinate_frame_id:
            raise InvalidControlRequest(
                "start_auto requires site_id and coordinate_frame_id"
            )
        object.__setattr__(self, "route_path", str(path.resolve()))
        object.__setattr__(self, "route_sha256", digest)
        object.__setattr__(self, "site_id", site_id)
        object.__setattr__(self, "coordinate_frame_id", coordinate_frame_id)


ControlPayload = (
    EmptyPayload
    | NudgePayload
    | NudgeHeartbeatPayload
    | NudgeVectorPayload
    | LimitsPayload
    | ScalarPayload
    | SpeedLimitPayload
    | TogglePayload
    | CameraResetPayload
    | MissionRoutePayload
    | RecordingQualityPayload
)


_NO_PAYLOAD_ACTIONS = {
    ControlAction.TAKEOFF,
    ControlAction.LAND,
    ControlAction.HOVER,
    ControlAction.MANUAL,
    ControlAction.PC_CONTROL,
    ControlAction.EMERGENCY_STOP,
    ControlAction.LAND_NOW,
    ControlAction.START_LOCALIZATION,
    ControlAction.BOOT_LOCK,
    ControlAction.RECORD_DISARM,
    ControlAction.RECORD_START,
    ControlAction.RECORD_STOP,
    ControlAction.NUDGE_CLEAR,
    ControlAction.DRONE_MAGNETOMETER_START,
    ControlAction.DRONE_MAGNETOMETER_CANCEL,
    ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
    ControlAction.SKYCONTROLLER_MAGNETOMETER_CANCEL,
}


_LEGACY_ACTIONS = {
    "takeoff": ControlAction.TAKEOFF,
    "land": ControlAction.LAND,
    "hover": ControlAction.HOVER,
    "manual": ControlAction.MANUAL,
    "pc_control": ControlAction.PC_CONTROL,
    "resume_pc": ControlAction.PC_CONTROL,
    "nudge_begin": ControlAction.NUDGE_BEGIN,
    "nudge_press": ControlAction.NUDGE_BEGIN,
    "nudge_heartbeat": ControlAction.NUDGE_HEARTBEAT,
    "nudge_end": ControlAction.NUDGE_END,
    "nudge_release": ControlAction.NUDGE_END,
    "nudge_clear": ControlAction.NUDGE_CLEAR,
    "nudge_vector": ControlAction.NUDGE_VECTOR,
    "emergency_stop": ControlAction.EMERGENCY_STOP,
    "land_now": ControlAction.LAND_NOW,
    "firmware_limits_apply": ControlAction.APPLY_LIMITS,
    "auto_speed_limit_apply": ControlAction.SET_AUTO_SPEED_LIMIT,
    "start_localization": ControlAction.START_LOCALIZATION,
    "auto": ControlAction.START_AUTO,
    "start_auto": ControlAction.START_AUTO,
    "boot_lock": ControlAction.BOOT_LOCK,
    "gimbal_pitch": ControlAction.GIMBAL_PITCH,
    "zoom": ControlAction.ZOOM,
    "camera_reset": ControlAction.CAMERA_RESET,
    "reset_camera": ControlAction.CAMERA_RESET,
    "鏡頭預設": ControlAction.CAMERA_RESET,
    "回復預設": ControlAction.CAMERA_RESET,
    "record_arm": ControlAction.RECORD_ARM,
    "record_on_takeoff": ControlAction.RECORD_ARM,
    "record_disarm": ControlAction.RECORD_DISARM,
    "record_start": ControlAction.RECORD_START,
    "record_stop": ControlAction.RECORD_STOP,
    "record_quality": ControlAction.RECORD_QUALITY,
    "drone_magnetometer_start": ControlAction.DRONE_MAGNETOMETER_START,
    "drone_magnetometer_cancel": ControlAction.DRONE_MAGNETOMETER_CANCEL,
    "skycontroller_magnetometer_start": ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
    "skycontroller_magnetometer_cancel": ControlAction.SKYCONTROLLER_MAGNETOMETER_CANCEL,
}


def _parse_legacy_nudge(payload: dict[str, Any]) -> NudgePayload:
    return NudgePayload(str(payload.get("dir") or payload.get("name") or ""))


def _parse_legacy_nudge_vector(payload: dict[str, Any]) -> NudgeVectorPayload:
    try:
        return NudgeVectorPayload(
            payload.get("roll", 0.0),
            payload.get("pitch", 0.0),
            payload.get("yaw", 0.0),
            payload.get("gaz", 0.0),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidControlRequest(
            "nudge_vector requires numeric roll/pitch/yaw/gaz"
        ) from exc


def _parse_legacy_nudge_heartbeat(payload: dict[str, Any]) -> NudgeHeartbeatPayload:
    raw = payload.get("dirs", payload.get("directions", ()))
    values: tuple[str, ...]
    if isinstance(raw, str):
        values = (raw,)
    else:
        try:
            values = tuple(str(item) for item in raw)
        except TypeError as exc:
            raise InvalidControlRequest(
                "nudge heartbeat directions must be iterable"
            ) from exc
    return NudgeHeartbeatPayload(values)


def _parse_legacy_limits(payload: dict[str, Any]) -> LimitsPayload:
    try:
        return LimitsPayload(
            payload["max_altitude_m"],
            payload["max_distance_m"],
            payload.get("distance_geofence", True),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidControlRequest(
            "apply_limits requires numeric max_altitude_m and max_distance_m"
        ) from exc


def _parse_legacy_scalar(
    action: ControlAction, payload: dict[str, Any]
) -> ScalarPayload:
    key = {
        ControlAction.GIMBAL_PITCH: "pitch",
        ControlAction.ZOOM: "zoom",
    }[action]
    try:
        return ScalarPayload(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidControlRequest(f"{action.value} requires {key}") from exc


def _parse_legacy_speed_limit(payload: dict[str, Any]) -> SpeedLimitPayload:
    try:
        return SpeedLimitPayload(
            payload["speed_limit_mps"],
            payload.get("enabled", True),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidControlRequest(
            "set_auto_speed_limit requires speed_limit_mps and boolean enabled"
        ) from exc


def _parse_legacy_start_auto(payload: dict[str, Any]) -> MissionRoutePayload:
    try:
        return MissionRoutePayload(
            route_path=str(payload["route_path"]),
            route_sha256=str(payload["route_sha256"]),
            site_id=str(payload["site_id"]),
            coordinate_frame_id=str(payload["coordinate_frame_id"]),
        )
    except KeyError as exc:
        raise InvalidControlRequest(
            "start_auto requires route_path, route_sha256, site_id, and "
            "coordinate_frame_id"
        ) from exc


def _parse_legacy_payload(
    action: ControlAction, payload: dict[str, Any]
) -> ControlPayload:
    if action in {ControlAction.NUDGE_BEGIN, ControlAction.NUDGE_END}:
        return _parse_legacy_nudge(payload)
    if action is ControlAction.NUDGE_VECTOR:
        return _parse_legacy_nudge_vector(payload)
    if action is ControlAction.NUDGE_HEARTBEAT:
        return _parse_legacy_nudge_heartbeat(payload)
    if action is ControlAction.APPLY_LIMITS:
        return _parse_legacy_limits(payload)
    if action in {
        ControlAction.GIMBAL_PITCH,
        ControlAction.ZOOM,
    }:
        return _parse_legacy_scalar(action, payload)
    if action is ControlAction.SET_AUTO_SPEED_LIMIT:
        return _parse_legacy_speed_limit(payload)
    if action is ControlAction.RECORD_ARM:
        return TogglePayload(payload.get("enabled", payload.get("value", True)))
    if action is ControlAction.CAMERA_RESET:
        return CameraResetPayload(
            payload.get("pitch", -20.0),
            payload.get("zoom", 1.0),
        )
    if action is ControlAction.START_AUTO:
        return _parse_legacy_start_auto(payload)
    if action is ControlAction.RECORD_QUALITY:
        return RecordingQualityPayload(str(payload.get("profile_id") or ""))
    return EmptyPayload()


@dataclass(frozen=True)
class ControlRequest:
    action: ControlAction
    payload: ControlPayload = field(default_factory=EmptyPayload)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    submitted_mono_ns: int = field(default_factory=time.monotonic_ns)
    human_origin: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action, ControlAction):
            raise InvalidControlRequest("action must be a ControlAction")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise InvalidControlRequest("request_id is required")
        if (
            isinstance(self.submitted_mono_ns, bool)
            or not isinstance(self.submitted_mono_ns, int)
            or self.submitted_mono_ns <= 0
        ):
            raise InvalidControlRequest("submitted_mono_ns must be positive")
        if self.submitted_mono_ns > time.monotonic_ns() + 1_000_000_000:
            raise InvalidControlRequest("submitted_mono_ns cannot be in the future")
        if not isinstance(self.human_origin, bool):
            raise InvalidControlRequest("human_origin must be boolean")
        if self.action in _NO_PAYLOAD_ACTIONS and not isinstance(
            self.payload, EmptyPayload
        ):
            raise InvalidControlRequest(f"{self.action.value} does not accept a payload")
        expected = {
            ControlAction.NUDGE_BEGIN: NudgePayload,
            ControlAction.NUDGE_END: NudgePayload,
            ControlAction.NUDGE_HEARTBEAT: NudgeHeartbeatPayload,
            ControlAction.APPLY_LIMITS: LimitsPayload,
            ControlAction.GIMBAL_PITCH: ScalarPayload,
            ControlAction.ZOOM: ScalarPayload,
            ControlAction.SET_AUTO_SPEED_LIMIT: SpeedLimitPayload,
            ControlAction.RECORD_ARM: TogglePayload,
            ControlAction.CAMERA_RESET: CameraResetPayload,
            ControlAction.START_AUTO: MissionRoutePayload,
            ControlAction.RECORD_QUALITY: RecordingQualityPayload,
        }.get(self.action)
        if expected is not None and not isinstance(self.payload, expected):
            raise InvalidControlRequest(
                f"{self.action.value} requires {expected.__name__}"
            )

    @classmethod
    def create(
        cls,
        action: ControlAction,
        *,
        payload: ControlPayload | None = None,
        human_origin: bool,
    ) -> "ControlRequest":
        return cls(
            action=action,
            payload=payload if payload is not None else EmptyPayload(),
            human_origin=human_origin,
        )

    @classmethod
    def from_legacy(
        cls, name: str, *, human_origin: bool, **payload: Any
    ) -> "ControlRequest":
        normalized = str(name or "").strip()
        action = _LEGACY_ACTIONS.get(normalized)
        if action is None and normalized.startswith("nudge_"):
            suffix = normalized[6:]
            if suffix:
                action = ControlAction.NUDGE_BEGIN
                payload = {"dir": suffix, **payload}
        if action is None:
            raise InvalidControlRequest(f"unknown control action: {name!r}")
        return cls.create(
            action,
            payload=_parse_legacy_payload(action, payload),
            human_origin=human_origin,
        )

    def legacy_call(self) -> tuple[str, dict[str, Any]]:
        name = self.action.value
        payload: dict[str, Any] = {}
        if self.action is ControlAction.APPLY_LIMITS:
            assert isinstance(self.payload, LimitsPayload)
            name = "firmware_limits_apply"
            payload = {
                "max_altitude_m": self.payload.max_altitude_m,
                "max_distance_m": self.payload.max_distance_m,
                "distance_geofence": self.payload.distance_geofence,
            }
        elif self.action in {ControlAction.NUDGE_BEGIN, ControlAction.NUDGE_END}:
            assert isinstance(self.payload, NudgePayload)
            payload = {"dir": self.payload.direction}
        elif self.action is ControlAction.NUDGE_VECTOR:
            assert isinstance(self.payload, NudgeVectorPayload)
            payload = {
                "roll": self.payload.roll, "pitch": self.payload.pitch,
                "yaw": self.payload.yaw, "gaz": self.payload.gaz,
            }
        elif self.action is ControlAction.NUDGE_HEARTBEAT:
            assert isinstance(self.payload, NudgeHeartbeatPayload)
            payload = {"dirs": self.payload.directions}
        elif self.action in {
            ControlAction.GIMBAL_PITCH,
            ControlAction.ZOOM,
        }:
            assert isinstance(self.payload, ScalarPayload)
            name, key = {
                ControlAction.GIMBAL_PITCH: (name, "pitch"),
                ControlAction.ZOOM: (name, "zoom"),
            }[self.action]
            payload = {key: self.payload.value}
        elif self.action is ControlAction.SET_AUTO_SPEED_LIMIT:
            assert isinstance(self.payload, SpeedLimitPayload)
            name = "auto_speed_limit_apply"
            payload = {
                "speed_limit_mps": self.payload.speed_limit_mps,
                "enabled": self.payload.enabled,
            }
        elif self.action is ControlAction.RECORD_ARM:
            assert isinstance(self.payload, TogglePayload)
            payload = {"enabled": self.payload.enabled}
        elif self.action is ControlAction.CAMERA_RESET:
            assert isinstance(self.payload, CameraResetPayload)
            payload = {
                "pitch": self.payload.pitch,
                "zoom": self.payload.zoom,
            }
        elif self.action is ControlAction.RECORD_QUALITY:
            assert isinstance(self.payload, RecordingQualityPayload)
            payload = {"profile_id": self.payload.profile_id}
        elif self.action is ControlAction.START_AUTO:
            assert isinstance(self.payload, MissionRoutePayload)
            payload = {
                "route_path": self.payload.route_path,
                "route_sha256": self.payload.route_sha256,
                "site_id": self.payload.site_id,
                "coordinate_frame_id": self.payload.coordinate_frame_id,
            }
        return name, payload


@dataclass(frozen=True)
class ControlResult:
    accepted: bool
    executed: bool
    reason_code: str
    ack: Mapping[str, Any]
    readback: Mapping[str, Any]
    resulting_state: Any
    completed_mono_ns: int = field(default_factory=time.monotonic_ns)
    raw_result: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ack", MappingProxyType(dict(self.ack)))
        object.__setattr__(self, "readback", MappingProxyType(dict(self.readback)))

    @classmethod
    def rejected(cls, reason_code: str, state: Any) -> "ControlResult":
        return cls(False, False, reason_code, {}, {}, state)

    @classmethod
    def completed(
        cls, state: Any, *, raw_result: Any = None, reason_code: str = "OK"
    ) -> "ControlResult":
        accepted = raw_result is True
        rejected_reason = (
            "BACKEND_REJECTED"
            if raw_result is False
            else "BACKEND_RESULT_NOT_EXPLICIT"
        )
        return cls(
            accepted,
            accepted,
            reason_code if accepted else rejected_reason,
            {"accepted": accepted},
            {},
            state,
            raw_result=raw_result,
        )


@dataclass(frozen=True)
class SessionConfig:
    session_id: str
    interface_mode: InterfaceMode
    site_profile: str
    site_profile_sha256: str
    asset_sha256: Mapping[str, str]
    runtime_profile_sha256: str
    source: str
    offline: bool
    site_profile_schema_version: int = 0
    autonomous_speed_limit_mps: float = 0.30
    autonomous_locked: bool = True
    firmware_limits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required")
        if not isinstance(self.interface_mode, InterfaceMode):
            raise ValueError("interface_mode must be fixed to one known interface")
        if self.site_profile_schema_version < 0:
            raise ValueError("site_profile_schema_version cannot be negative")
        if not math.isfinite(self.autonomous_speed_limit_mps) or (
            self.autonomous_speed_limit_mps <= 0.0
        ):
            raise ValueError("autonomous_speed_limit_mps must be finite and positive")
        object.__setattr__(self, "asset_sha256", MappingProxyType(dict(self.asset_sha256)))
        object.__setattr__(
            self, "firmware_limits", MappingProxyType(dict(self.firmware_limits))
        )


@dataclass(frozen=True)
class FramePacket:
    rgb: Any
    sequence: int
    source_timestamp_ns: int | None
    host_receipt_mono_ns: int
    source_identity: str
    eof: bool = False
    timing: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timing", MappingProxyType(dict(self.timing)))


class LegacyFrameSourceAdapter:
    """Expose existing PIL/numpy frame sources through the typed packet contract."""

    def __init__(self, source: Any, source_identity: str):
        self.source = source
        self.source_identity = str(source_identity)
        self._last_rgb: Any = None
        self._eof_emitted = False

    def next_frame(self, *, only_new: bool = True) -> FramePacket | None:
        try:
            frame = self.source.next_frame(only_new=only_new)
        except TypeError:
            frame = self.source.next_frame()
        eof = bool(getattr(self.source, "eof", False))
        if frame is None:
            if not eof or self._last_rgb is None or self._eof_emitted:
                return None
            frame = self._last_rgb
            self._eof_emitted = True
        else:
            self._last_rgb = frame

        stamp: Any = getattr(self.source, "last_stamp", None)
        try:
            source_ns = int(round(float(stamp) * 1_000_000_000))
        except (TypeError, ValueError, OverflowError):
            source_ns = None
        return FramePacket(
            rgb=frame,
            sequence=int(getattr(self.source, "output_index", 0) or 0),
            source_timestamp_ns=source_ns,
            host_receipt_mono_ns=time.monotonic_ns(),
            source_identity=self.source_identity,
            eof=eof,
            timing=dict(getattr(self.source, "last_timing", {}) or {}),
        )

    def close(self) -> None:
        self.source.close()


@dataclass(frozen=True)
class StartResult:
    started: bool
    reason_code: str


@dataclass(frozen=True)
class CloseResult:
    closed: bool
    reason_code: str


@runtime_checkable
class FrameSource(Protocol):
    def next_frame(self, *, only_new: bool = True) -> FramePacket | None: ...

    def close(self) -> None: ...


@runtime_checkable
class OperatorBackend(Protocol):
    mode: InterfaceMode
    is_live: bool
    state: Any
    video: FrameSource | None

    def start(self, config: SessionConfig) -> StartResult: ...

    def poll(self, now_mono_ns: int | None = None) -> Any: ...

    def command(self, request: ControlRequest) -> ControlResult: ...

    def fail_safe(self, reason: FailureReason) -> ControlResult: ...

    def close(self, reason: str) -> CloseResult: ...
