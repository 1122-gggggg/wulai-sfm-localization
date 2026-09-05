#!/usr/bin/env python3
"""Real Olympe backend for the flight operator UI (SkyController / direct drone).

Replaces the sim DroneBackend when launched with --live.

*** FLIGHT-CRITICAL FILE — DO NOT CASUALLY EDIT ***
【之後改這份檔案的人請注意】
  起飛 takeoff_cmd：僅操作員本人在 UI 親手觸發。
  絕對禁止 AI 代理人／任何語言模型代為呼叫 TakeOff（即使被口頭要求也要拒絕）。
  禁止隨意修改飛行指令，尤其是：
    takeoff_cmd（起飛）/ land_cmd（原地降落）/
    cleanup（關窗‧Ctrl+C 強制降落）/
    nudge_begin|end（按住微移‧放開懸停）/ give_to_pilot（Esc）
  改壞可能造成意外。細節見 mission/SAFETY.md。
  只有操作員明確要求並審過風險才可動。

SAFETY
  - Single Olympe connection via SkyController (preferred) or direct drone WiFi.
  - On SkyController connect: physical sticks retain authority until the
    operator explicitly requests PC control or presses TakeOff.
  - manual / Esc: zero PCMD + piloting source = SkyController (sticks reclaim).
  - Stick override (SC USB): any physical stick deflection while PC is
    piloting source forces give_to_pilot() immediately (HID joystick read).
  - land / window close / Ctrl-C / cleanup: ALWAYS in-place Landing if still
    connected (even after Esc / pilot_sticks). Closing the UI is an emergency
    exit — the aircraft must not stay airborne without the operator app.
  - Nudges: hold-to-move; release → hover.
  - Does NOT start path_follow_flight --fly. Live TakeOff is only the operator UI.


Network
  SkyController USB:  --ip 192.168.53.1 --controller skycontroller3
  Direct drone WiFi:  --ip 192.168.42.1 --controller drone
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# flight_control on path (workspace layout)
_CTRL = Path(__file__).resolve().parents[1]
if str(_CTRL) not in sys.path:
    sys.path.insert(0, str(_CTRL))
from workspace_layout import workspace_from_file  # noqa: E402
_WS = workspace_from_file(__file__)
_FC = _WS.flight_control
if str(_FC) not in sys.path:
    sys.path.insert(0, str(_FC))

from manual_nudge_pilot import NUDGE_DIRS, scale_nudge, NUDGE_PCT, NUDGE_S  # noqa: E402
from backend_contract import (  # noqa: E402
    CloseResult,
    ControlAction,
    ControlRequest,
    ControlResult,
    FailureReason,
    InterfaceMode,
    LegacyFrameSourceAdapter,
    SessionConfig,
    StartResult,
)
from runtime_safety import assess_disk_space  # noqa: E402
from operator_state import TrackerState  # noqa: E402
from operator_flight_safety import (  # noqa: E402
    AuthorityController,
    TakeoffLandingSupervisor,
    TelemetryFreshnessStore,
)
from scale_free_control_adapter import validate_speed_limit_change  # noqa: E402
from live_safety_config import (  # noqa: E402
    DEFAULT_MAX_ROTATION_SPEED_DEGS,
    DEFAULT_MAX_TILT_DEG,
    DEFAULT_MAX_VERTICAL_SPEED_MS,
    DEFAULT_RTH_MIN_ALTITUDE_M,
    DEFAULT_STREAM_LOSS_GRACE_S,
    LiveSafetyConfig,
    NUDGE_TTL_MAX_S,
)
from recording_quality import (  # noqa: E402
    DEFAULT_RECORDING_PROFILE,
    RecordingProfile,
    format_record_status,
    recording_mode_matches,
    resolve_recording_profile,
)
from skycontroller_stick import (  # noqa: E402
    SkyControllerStickMonitor,
    axes_active,
    stick_log_sample,
    _JS_EVENT_AXIS as _JS_EVENT_AXIS,
    _JS_EVENT_FMT as _JS_EVENT_FMT,
    _STICK_CALLBACK_FAIL_LIMIT as _STICK_CALLBACK_FAIL_LIMIT,
    _STICK_DEADZONE,
    _STICK_FLIGHT_AXES,
)
from live_anafi_video_stream import LiveAnafiVideoStream  # noqa: E402


# Local download dir for flight recordings (stop_recording finalizes on drone media).
_DEFAULT_RECORD_DIR = _WS.flight_logs / "recordings"
_MEDIA_PENDING_TIMEOUT_S = 2.0
_MEDIA_DOWNLOAD_TIMEOUT_S = 15.0
_FIRMWARE_SETTING_TIMEOUT_S = 10.0
# Match the documented flight-control command TTL at the typed backend boundary.
# A queued ordinary request is rejected once it is older than this; explicit
# LAND_NOW and EMERGENCY_STOP requests remain executable regardless of age.
CONTROL_REQUEST_MAX_AGE_NS = 250_000_000
CRITICAL_TELEMETRY_MAX_AGE_S = 1.0


def _clamp_pct(v: int) -> int:
    return max(-100, min(100, int(v)))


def takeoff_battery_blocker(battery: float, battery_floor: float | None) -> str | None:
    """D1: pure fragment of takeoff preflight -- battery-vs-floor decision only.

    The caller has already fetched ``battery`` from live telemetry; this makes
    just the threshold decision testable without a drone.
    """
    if battery_floor is None:
        return "takeoff battery floor is unavailable or invalid"
    if battery < battery_floor:
        return f"battery {battery:.0f}% is below the {battery_floor:.0f}% takeoff floor"
    return None


@dataclass(frozen=True)
class _TakeoffGpsOutcome:
    blocker: str | None
    advisory: str | None


def takeoff_gps_outcome(*, fixed: bool | None, gps_required: bool) -> _TakeoffGpsOutcome:
    """D1: pure fragment of takeoff preflight -- GPS-vs-geofence decision only.

    ``fixed`` is None when the GPS state API itself is unavailable (distinct
    from the API being available but reporting no fix). Either way, a
    configured distance geofence that requires GPS blocks takeoff; otherwise
    it is only an advisory, since a purely local/manual flight has no need
    for GPS.
    """
    if fixed:
        return _TakeoffGpsOutcome(None, None)
    reason = "GPS state API unavailable" if fixed is None else "GPS fix unavailable"
    if gps_required:
        return _TakeoffGpsOutcome(
            f"{reason}; required by configured distance geofence", None
        )
    return _TakeoffGpsOutcome(None, f"{reason}; takeoff remains allowed")


_LINK_FAILURE_CONFIRM_POLLS = 3
# Manual-nudge envelope. The deadman TTL is the "frozen UI decays to hover" timer,
# and nudge_pct scales every held-key command, so both need an upper bound.
#: Sentinel entry in _nudge_held representing a continuous on-screen stick, so the
#: existing hold loop / deadman / release paths treat it exactly like a held button.
_VECTOR_HOLD = "__vector__"
# tracker_state values the firmware flying-state poll must NOT overwrite. The first
# group is "the host is deliberately in this mode"; the second is terminal/abnormal
# outcomes -- overwriting those erased the only on-screen record that a takeoff was
# blocked, a landing was unconfirmed, or a piloting-source handoff failed.
_TRACKER_STATE_STICKY = frozenset({
    "NUDGE", "STICKS", "STREAM_LOST_HOVER", "STREAM_LOST_MANUAL",
    "FAIL_SAFE_MANUAL", "PC_FROZEN", "HOVER", "PC", "RTH",
    "SAFETY_ACTION_PENDING", "STICK_MONITOR_FAIL", "LINK_LOST_ONBOARD",
    "LAND_UNCONFIRMED", "SOURCE_FAIL", "TAKEOFF_FAIL", "TAKEOFF_BLOCKED",
})
#: Age at which a live frame stops counting as fresh FOR THE HANDOFF TIMER.
#: NOT the same decision as path_follow_flight.STREAM_STALE_S (0.5 s), which gates
#: whether the autonomous loop may localize at all. Same word, two different
#: consequences and two different values -- deliberately, so do not "unify" them
#: without deciding what each one is protecting.
_STREAM_STALE_S = 0.75
# How long the stream must be CONTINUOUSLY stale/frozen before control is handed
# back to the sticks. A single 0.75 s gap is a hiccup, not a lost stream: on a
# flaky Wi-Fi link the old single-sample trigger tore PC control away constantly
# (operator decision 2026-08-06). The handoff itself is unchanged -- only how long
# the outage must persist before it fires.
#: Grace for a STALE (late) frame only. A FROZEN feed -- the same picture repeated
#: -- gets no grace at all: riding that out means flying blind for the whole window.
FROZEN_STREAM_GRACE_S = 1.0
_STREAM_DUPLICATE_FRAMES = 15
_ALTITUDE_LIMIT_MARGIN_M = 1.0
_DISTANCE_LIMIT_FRACTION = 0.95
_LOST_LINK_RTH_DELAY_S = 1
_EARTH_RADIUS_M = 6_371_000.0


def quiet_olympe_logs() -> None:
    """Cut Olympe/pdraw INFO spam (AVCC unsupported, renderer empty queue, etc.).

    Does not change flight commands — log level only. Call before connect.
    """
    import logging
    # Force WARNING even if handlers already exist (Olympe reconfigures often).
    logging.basicConfig(level=logging.WARNING)
    # Olympe creates per-device child loggers after connection and may attach
    # their own INFO handlers. The global threshold keeps verbose state payloads
    # out of the operator terminal regardless of later logger reconfiguration.
    logging.disable(logging.INFO)
    for name in (
        "olympe",
        "ulog",
        "olympe.pdraw",
        "olympe.drone",
        "olympe.video",
        "olympe.video.renderer",
        "olympe.backend",
        "olympe.media",
        "olympe.module_loader",
        "olympe.scheduler",
        "olympe.update",
        "olympe.flightplan",
        "olympe.missions",
        "olympe.arsdkng",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.propagate = True


class _CmdLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._f = None
        self.durable = False
        self.healthy = True
        self.last_error = ""
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._f = path.open("w", encoding="utf-8", buffering=1)
                self.event("start", sink="olympe_live_ui")
            except Exception as exc:
                self.healthy = False
                self.last_error = repr(exc)
                self._f = None
                print(
                    f"[live-ui] safety log unavailable path={path}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

    def event(self, event: str, **kw: Any) -> None:
        mono_ns = time.monotonic_ns()
        rec = {
            "t_iso": datetime.now(timezone.utc).isoformat(),
            "t_mono": mono_ns * 1e-9,
            "t_mono_ns": mono_ns,
            "event": event,
            **kw,
        }
        try:
            line = json.dumps(rec, ensure_ascii=False)
        except Exception as exc:
            self.healthy = False
            self.last_error = repr(exc)
            print(
                f"[live-ui] safety event serialization failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return
        print(f"[live-ui] {line}", flush=True)
        if self._f is not None:
            try:
                self._f.write(line + "\n")
                self._f.flush()
                os.fsync(self._f.fileno())
                self.durable = True
            except Exception as exc:
                self.durable = False
                self.healthy = False
                self.last_error = repr(exc)
                print(
                    f"[live-ui] safety log write failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

    def close(self) -> None:
        if self._f is not None:
            try:
                self.event("end")
            except Exception:  # Tier3: teardown best-effort — event flush optional
                pass
            try:
                self._f.close()
            except Exception:  # Tier3: teardown best-effort — file close optional
                pass
            self._f = None


class OlympeLiveBackend:
    """Drop-in live backend for OperatorApp (same methods as DroneBackend)."""

    mode = InterfaceMode.REAL_FLIGHT
    is_live = True

    def __init__(
        self,
        state_factory,
        anafi_profile,
        *,
        ip: str = "192.168.53.1",
        controller: str = "skycontroller3",
        nudge_pct: int = NUDGE_PCT,
        nudge_pulse_s: float = NUDGE_S,
        max_altitude_m: float | None = None,
        max_distance_m: float | None = None,
        max_tilt_deg: float = DEFAULT_MAX_TILT_DEG,
        max_vertical_speed_ms: float = DEFAULT_MAX_VERTICAL_SPEED_MS,
        max_rotation_speed_degs: float = DEFAULT_MAX_ROTATION_SPEED_DEGS,
        rth_min_altitude_m: float = DEFAULT_RTH_MIN_ALTITUDE_M,
        auto_pc_control: bool = True,
        stream_loss_grace_s: float = DEFAULT_STREAM_LOSS_GRACE_S,
        distance_geofence: bool = True,
        min_takeoff_battery_pct: float = 30.0,
        require_gps_for_geofence: bool = True,
        approved_aircraft_firmware: tuple[str, ...] = (),
        approved_controller_firmware: tuple[str, ...] = (),
        approved_olympe_versions: tuple[str, ...] = (),
        cmd_log: Path | None = None,
        event_log: Any | None = None,
        session_logs: Any | None = None,
        with_video: bool = True,
    ):
        """state_factory: callable -> DroneState; anafi_profile: AnafiProfile."""
        requested_ttl_raw = nudge_pulse_s
        safety_config = LiveSafetyConfig.resolve(
            nudge_pct=nudge_pct,
            nudge_pulse_s=requested_ttl_raw,
            max_altitude_m=max_altitude_m,
            max_distance_m=max_distance_m,
            max_tilt_deg=max_tilt_deg,
            max_vertical_speed_ms=max_vertical_speed_ms,
            max_rotation_speed_degs=max_rotation_speed_degs,
            rth_min_altitude_m=rth_min_altitude_m,
            stream_loss_grace_s=stream_loss_grace_s,
            min_takeoff_battery_pct=min_takeoff_battery_pct,
            distance_geofence=distance_geofence,
            require_gps_for_geofence=require_gps_for_geofence,
        )
        requested_ttl = float(requested_ttl_raw)
        self.ANAFI = anafi_profile
        self.state = state_factory()
        self.state.loc = "LIVE"
        self.state.stream = "CONNECTING"
        self.state.mode = "MANUAL"
        # Initial construction, before self.log exists below: not a
        # transition worth an audit trail entry, just the starting state.
        self.state.tracker_state = TrackerState.LINK
        self.ip = ip
        self.controller = controller
        self.nudge_pct = safety_config.nudge_pct
        self.desired_max_altitude_m = safety_config.max_altitude_m
        self.desired_max_tilt_deg = safety_config.max_tilt_deg
        self.desired_max_vertical_speed_ms = safety_config.max_vertical_speed_ms
        self.desired_max_rotation_speed_degs = safety_config.max_rotation_speed_degs
        self.desired_rth_min_altitude_m = safety_config.rth_min_altitude_m
        self.auto_pc_control = bool(auto_pc_control)
        self.stream_loss_grace_s = safety_config.stream_loss_grace_s
        self.desired_max_distance_m = safety_config.max_distance_m
        self.desired_distance_geofence = safety_config.distance_geofence
        self.min_takeoff_battery_pct = safety_config.min_takeoff_battery_pct
        self.require_gps_for_geofence = safety_config.require_gps_for_geofence
        self.approved_aircraft_firmware = frozenset(
            str(value).strip() for value in approved_aircraft_firmware if str(value).strip()
        )
        self.approved_controller_firmware = frozenset(
            str(value).strip() for value in approved_controller_firmware if str(value).strip()
        )
        self.approved_olympe_versions = frozenset(
            str(value).strip() for value in approved_olympe_versions if str(value).strip()
        )
        self.connection_inventory: dict[str, Any] = {}
        self._inventory_takeoff_ready = False
        self._inventory_block_reason = "hardware inventory not read"
        self._firmware_config_ok = False
        self._firmware_config_reason = "not configured"
        self.state.preflight_ok = False
        self.state.preflight_reason = "firmware limits not checked"
        self.state.control_owner = "SKYCONTROLLER"
        self.state.active_incident = ""
        self.state.aircraft_identity = "UNREAD"
        self.state.controller_identity = "UNREAD"
        self.state.autonomous_locked = True
        self.state.home_valid = None
        self.state.rth_policy_valid = None
        self.state.rth_policy_configured = False
        self.state.rth_min_altitude_m = None
        self._last_magnetometer_snapshot: tuple | None = None
        self.state.stick_monitor_ok = not self.via_skycontroller()
        self.state.distance_from_home_m = None
        self.state.airspeed_mps = None
        self.state.autonomous_speed_limit_enabled = True
        self.state.autonomous_speed_limit_mps = 0.30
        self.state.autonomous_approval_valid = False
        self.state.drone_magnetometer_required = None
        self.state.drone_magnetometer_started = None
        self.state.drone_magnetometer_axis = "unknown"
        self.state.drone_magnetometer_x_done = None
        self.state.drone_magnetometer_y_done = None
        self.state.drone_magnetometer_z_done = None
        self.state.drone_magnetometer_failed = None
        self.state.skycontroller_magnetometer_state = (
            "unknown" if self.via_skycontroller() else "not_applicable"
        )
        # Deadman TTL refreshed by UI heartbeats while a nudge remains held. This is
        # the timer that turns "the UI froze while a key was held" into a hover, so it
        # needs a CEILING as well as a floor: an oversized TTL silently removes the
        # guarantee. Clamping a safety timeout downwards is always the safe direction,
        # but it must never be silent.
        self.nudge_pulse_s = safety_config.nudge_pulse_s
        if not math.isclose(self.nudge_pulse_s, requested_ttl, rel_tol=1e-9):
            print(
                f"[live-backend] nudge TTL {requested_ttl:g}s clamped to "
                f"{self.nudge_pulse_s:g}s (allowed 0.1..{NUDGE_TTL_MAX_S:g}s)",
                flush=True,
            )
        self.log = event_log if event_log is not None else _CmdLog(cmd_log)
        self.session_logs = session_logs
        self.disk_guard_path = Path(
            getattr(self.log, "path", None) or cmd_log or _WS.flight_logs
        )
        if self.disk_guard_path.suffix:
            self.disk_guard_path = self.disk_guard_path.parent
        self._last_runtime_storage_check_t = 0.0
        self._runtime_storage_guard_latched = False
        self.drone = None
        self.grabber = None
        self.video_stream: LiveAnafiVideoStream | None = None
        self._video_inventory_logged = False
        self.video = None
        self.session_config: SessionConfig | None = None
        self.with_video = bool(with_video)
        self._lock = threading.RLock()
        self.authority_controller = AuthorityController()
        self._cleanup_lock = threading.Lock()
        self._firmware_limits_lock = threading.Lock()
        self._calibration_lock = threading.Lock()
        self._pulse_token = 0
        # Nudge snapshots may outlive the input/authority state they observed.
        # Keep a motion-only epoch so clearing a nudge cannot be mistaken for a
        # takeoff/source epoch change (which has separate cancellation semantics).
        self._motion_epoch = 0
        self.pilot_sticks = False
        self._cleanup_done = False
        # Set at the beginning of shutdown, before any bounded Landing wait.  It
        # permanently retires PC motion for this backend instance, even when an
        # AUTO worker later wakes after its cancellation join timed out.
        self._shutdown_latched = False
        self._landed = False
        self.flight_start: float | None = None
        # SC stick override: physical stick deflection reclaims SkyController.
        self._stick_monitor: SkyControllerStickMonitor | None = None
        self._last_stick_axes: dict[int, int] = {}
        self._last_stick_axes_t = 0.0
        self._stick_reclaim_lock = threading.Lock()
        self._last_stick_reclaim_mono = 0.0
        self.stick_override_count = 0
        self._link_failure_count = 0
        self._link_was_ok = True
        self._stream_failure_latched = False
        self._stream_stale_since: float | None = None
        # Set while an automated TakeOff/Landing is in flight. Motion PCMD must not
        # interleave with it: the aircraft is executing a firmware manoeuvre and a
        # held nudge would fight it. Zero PCMD stays allowed (it only reinforces hover).
        self._maneuver_in_progress: str | None = None
        self._distance_guard_active: bool | None = None
        self.state.distance_guard_active = None
        self._nudge_vector: tuple[float, float, float, float] | None = None
        self.zero_pcmd_failures = 0
        self.last_zero_pcmd_error: str | None = None
        self._disk_low_noted = False
        self.last_zero_pcmd_call_mono_ns: int | None = None
        self._runtime_safety_action_lock = threading.Lock()
        self._runtime_safety_action_latched = False
        self._runtime_safety_action_reason = ""
        self._runtime_safety_action_thread: threading.Thread | None = None
        self.telemetry_freshness = TelemetryFreshnessStore()
        self.takeoff_landing_supervisor = TakeoffLandingSupervisor()
        self._home_latitude_deg: float | None = None
        self._home_longitude_deg: float | None = None
        # Hold-to-move: keys/buttons held → continuous PCMD; all released → hover.
        self._nudge_held: set[str] = set()
        self._nudge_loop_stop = threading.Event()
        self._nudge_loop_thread: threading.Thread | None = None
        self._nudge_period_s = 0.05  # 20 Hz
        self._nudge_deadline = 0.0
        # Read-only evidence for the on-site alignment wizard.  The wizard never
        # sends through this seam; it only checks that a host PCMD with the
        # expected sign actually preceded the measured motion.
        self._last_pcmd = (0, 0, 0, 0)
        self._last_pcmd_mono_ns: int | None = None
        # Flight recording: arm before takeoff → start on takeoff → stop+save on land.
        self.record_on_takeoff: bool = False
        self.recording_active: bool = False
        self.record_started_iso: str = ""
        self.record_last_path: str = ""
        self.recording_profile: RecordingProfile = DEFAULT_RECORDING_PROFILE
        self.record_status: str = format_record_status(
            active=False, armed=False, profile=self.recording_profile,
        )
        self.record_dir = _DEFAULT_RECORD_DIR
        self.record_dir.mkdir(parents=True, exist_ok=True)
        # sim-compatible attributes used by gravity UI
        self.gravity_sim_phase: str | None = None
        self._gravity_sim_t0 = None
        self.sim_xyz = np.zeros(3, dtype=float)
        self.sim_yaw = 0.0
        self.target_altitude_m = 0.0
        self.started_mono = time.monotonic()
        self.last_poll = self.started_mono

        self._connect()

    def _set_tracker_state(self, new: TrackerState, *, reason: str) -> TrackerState:
        """Centralized tracker_state transition (C2).

        Single write point for the enum values this backend controls (the ANAFI
        firmware's own FlyingStateChanged string, forwarded elsewhere, bypasses
        this on purpose -- see TrackerState's docstring). Logs old/new/reason so
        an incident reconstruction can see why the displayed state changed, not
        just that it did.
        """
        old = self.state.tracker_state
        self.state.tracker_state = new
        if new != old:
            # Several call sites (poll(), the nudge hold loop) re-assert the
            # same state every cycle; only a real transition is log-worthy --
            # logging every re-assertion would flood commands.jsonl/incidents
            # on a healthy connection and add I/O to hot paths for no benefit.
            #
            # `old` is passed as-is, not str(old): TrackerState mixes in str,
            # so it already serializes as its plain value ("LINK"), but
            # Enum.__str__ (not str.__str__) wins for an explicit str() call
            # and would log "TrackerState.LINK" instead.
            self.log.event("tracker_state_transition", old=old, new=new.value, reason=reason)
        return new

    # ------------------------------------------------------------------ connect
    def via_skycontroller(self) -> bool:
        """True when piloting/video go through SC USB (192.168.53.x)."""
        c = str(self.controller or "").lower()
        return ("sky" in c) or str(self.ip).startswith("192.168.53")

    def _connect(self) -> None:
        quiet_olympe_logs()
        import olympe_frame_source as ofs

        self.log.event(
            "connect_begin",
            ip=self.ip,
            controller=self.controller,
            mode="skycontroller" if self.via_skycontroller() else "direct_wifi",
        )
        self.drone = ofs.connect(self.ip, self.controller)
        # Never steal physical sticks merely by opening the UI. Direct WiFi has
        # no SkyController, so the laptop remains the sole controller there.
        initial_source = "SkyController" if self.via_skycontroller() else "Controller"
        if not self._set_piloting_source(initial_source):
            try:
                self.drone.disconnect()
            except Exception:  # Tier3: cleanup after failed piloting source — best-effort
                pass
            self.drone = None
            raise RuntimeError(f"failed to confirm {initial_source} piloting source")
        self.pilot_sticks = self.via_skycontroller()
        # SkyController 3 firmware 1.8.1 does not publish ProductVariantChanged.
        # Start HID monitoring before inventory so its product name can confirm
        # the controller model on the first read instead of leaving step 4 stale.
        if self.via_skycontroller():
            self._start_stick_monitor()
        self._configure_lost_link_policy_if_safe()
        self.read_connection_inventory()
        self._read_magnetometer_calibration_state()
        # Firmware writes are attempted only when both desired limits are set
        # and the aircraft readback says landed. Failure keeps ground UI alive
        # while takeoff remains fail-closed.
        self._configure_firmware_limits_if_safe()
        self.state.stream = "OK"
        self._set_tracker_state(TrackerState.STICKS if self.pilot_sticks else TrackerState.HOVER, reason="_connect")
        self.state.link_ok = True
        self.state.link_status = "OK"
        self.log.event("connect_ok")
        # AFTER the monitor exists: one of the safety conditions for taking PC
        # authority is that something can take it back, so this must not run first.
        self._auto_take_pc_control_if_safe()

        if self.with_video:
            # SC3 RTSP often advertises "DefaultVideo"; direct ANAFI uses "Front camera".
            media = "DefaultVideo" if self.via_skycontroller() else "Front camera"
            try:
                # stale_s low → reject backlog; 720p kept for future localization.
                self.grabber = ofs.OlympePdrawGrabber(
                    self.drone, resize=(1280, 720), stale_s=0.35,
                    media_name=media,
                ).start()
                self.video_stream = LiveAnafiVideoStream(self.grabber)
                self.video = LegacyFrameSourceAdapter(
                    self.video_stream,
                    f"anafi://{self.ip}/{getattr(self.grabber, 'media_name', media)}",
                )
                self.state.stream = "OK"
                self.log.event(
                    "video_ok",
                    media_name=getattr(self.grabber, "media_name", media),
                    stale_s=0.35,
                    note="low_latency_queue",
                )
            except Exception as exc:
                self.grabber = None
                self.video_stream = None
                self.video = None
                self.log.event("video_fail", error=repr(exc))
                # Control still works without video.
                self.state.stream = "NO_VIDEO"

    def _maybe_log_video_inventory(self) -> bool:
        if self._video_inventory_logged or self.video_stream is None:
            return False
        if int(getattr(self.video_stream, "output_index", 0) or 0) <= 0:
            return False
        snapshot = getattr(self.video_stream, "metadata_snapshot", None)
        if not callable(snapshot):
            return False
        try:
            inventory = dict(snapshot())
        except Exception:
            return False
        if not inventory.get("ui_frame_width_px") or not inventory.get(
            "ui_frame_height_px"
        ):
            return False
        self._video_inventory_logged = True
        self.connection_inventory = {
            **dict(self.connection_inventory),
            "video": inventory,
        }
        self.log.event("video_inventory", **inventory)
        if self.session_logs is not None:
            self.session_logs.telemetry("video_inventory", **inventory)
            writer = getattr(self.session_logs, "write_inventory", None)
            if callable(writer) and not writer("video", inventory):
                self.log.event("video_inventory_receipt_failed")
        return True

    def _set_piloting_source(self, source: str) -> bool:
        if self.drone is None:
            return False
        if not self.via_skycontroller():
            # Direct drone WiFi: no SkyController CoPiloting feature.
            self.log.event(
                "piloting_source", source=source, ok=True, note="direct_wifi_noop")
            return True
        try:
            from olympe.messages.skyctrl.CoPiloting import setPilotingSource
            from olympe.messages.skyctrl.CoPilotingState import pilotingSource

            waited = self.drone(
                setPilotingSource(source=source)
            ).wait(_timeout=10)
            ack_ok = bool(getattr(waited, "success", lambda: False)())
            if not ack_ok:
                self.log.event(
                    "piloting_source", source=source, ok=False,
                    note="expectation_failed",
                )
                return False

            st = self.drone.get_state(pilotingSource)
            actual = st.get("source") if isinstance(st, dict) else None

            def source_name(value: Any) -> str:
                return str(getattr(value, "name", value)).rsplit(".", 1)[-1].lower()

            readback_ok = source_name(actual) == source_name(source)
            # Enum values in olympe state are not JSON-serializable; log name only.
            self.log.event(
                "piloting_source",
                source=source,
                state_source=source_name(actual),
                ok=readback_ok,
                note="confirmed" if readback_ok else "readback_mismatch",
            )
            return readback_ok
        except Exception as exc:
            self.log.event("piloting_source", source=source, ok=False, error=repr(exc))
            return False

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    def _observe_telemetry(
        self,
        name: str,
        message: Any,
        fallback_state: dict[str, Any],
        *,
        now_mono_ns: int,
    ) -> dict[str, Any]:
        """Record one SDK event without refreshing an unchanged cache entry."""
        state = dict(fallback_state)
        marker: object = (
            "cache",
            json.dumps(state, sort_keys=True, default=str),
        )
        observed_mono_ns = int(now_mono_ns)
        get_last_event = getattr(self.drone, "get_last_event", None)
        invalid_event_timestamp = False
        if callable(get_last_event):
            try:
                event = get_last_event(message)
                wall_age_s = time.time() - float(event.date.timestamp())
                if not math.isfinite(wall_age_s) or wall_age_s < 0.0:
                    invalid_event_timestamp = True
                    raise ValueError("invalid telemetry event timestamp")
                # Validate the event clock before allowing its payload or marker
                # to replace the last trusted state.
                state = dict(event.args)
                marker = ("event", str(event.uuid))
                observed_mono_ns -= int(max(0.0, wall_age_s) * 1_000_000_000)
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                if invalid_event_timestamp:
                    previous = self.telemetry_freshness.sample(name)
                    if previous is not None:
                        return dict(previous.value)
                    # With no trusted sample, the cache fallback may itself be
                    # the future event payload. Do not let it become current
                    # state or give runtime safety a value with no valid receipt.
                    return {}
        sample = self.telemetry_freshness.observe(
            name,
            state,
            marker=marker,
            observed_mono_ns=max(0, observed_mono_ns),
        )
        setattr(self.state, f"{name}_mono_ns", sample.observed_mono_ns)
        return dict(sample.value)

    @staticmethod
    def _state_value(raw: dict | None, key: str) -> Any:
        value = raw.get(key) if raw is not None else None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(getattr(value, "name", value)).rsplit(".", 1)[-1]

    @classmethod
    def _sensor_state_updates(cls, raw: Any) -> dict[str, bool]:
        """Normalize Olympe's MAP_ITEM cache for SensorsStatesListChanged."""
        if not isinstance(raw, dict):
            return {}
        if "sensorName" in raw and "sensorState" in raw:
            entries = [(raw.get("sensorName"), raw)]
        else:
            entries = list(raw.items())
        updates: dict[str, bool] = {}
        for key, value in entries:
            if isinstance(value, dict):
                name = value.get("sensorName", key)
                sensor_state = value.get("sensorState")
            else:
                name = key
                sensor_state = value
            name_text = str(getattr(name, "name", name)).rsplit(".", 1)[-1]
            if not name_text or name_text in {"None", "list_flags"}:
                continue
            if isinstance(sensor_state, str):
                lowered = sensor_state.strip().lower()
                if lowered in {"1", "true", "ok", "yes", "on"}:
                    sensor_ok = True
                elif lowered in {"0", "false", "fault", "no", "off"}:
                    sensor_ok = False
                else:
                    continue
            elif isinstance(sensor_state, (bool, int)):
                sensor_ok = bool(sensor_state)
            else:
                continue
            updates[name_text] = sensor_ok
        return updates

    @staticmethod
    def _valid_home(latitude: Any, longitude: Any, altitude: Any) -> bool:
        values = []
        for value in (latitude, longitude, altitude):
            if isinstance(value, bool):
                return False
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                return False
            if not math.isfinite(parsed):
                return False
            values.append(parsed)
        return -90.0 <= values[0] <= 90.0 and -180.0 <= values[1] <= 180.0

    def _read_aircraft_inventory_states(
        self, errors: list[str],
    ) -> dict[str, dict | None]:
        try:
            from olympe.messages.common.CommonState import ProductModel
            from olympe.messages.common.SettingsState import (
                BoardIdChanged,
                ProductNameChanged,
                ProductSerialHighChanged,
                ProductSerialLowChanged,
                ProductVersionChanged,
            )
            from olympe.messages.ardrone3.GPSSettingsState import (
                HomeChanged,
                ReturnHomeDelayChanged,
                ReturnHomeMinAltitudeChanged,
            )
        except Exception as exc:
            errors.append(f"aircraft inventory API unavailable: {exc!r}")
            return {}
        common_messages = {
            "name": (ProductNameChanged, "name"),
            "model": (ProductModel, "model"),
            "version": (ProductVersionChanged, None),
            "serial_high": (ProductSerialHighChanged, "high"),
            "serial_low": (ProductSerialLowChanged, "low"),
            "board_id": (BoardIdChanged, "id"),
            "home": (HomeChanged, None),
            "return_home_delay": (ReturnHomeDelayChanged, None),
            "return_home_min_altitude": (ReturnHomeMinAltitudeChanged, None),
        }
        return {
            key: self._state_dict(message)
            for key, (message, _field) in common_messages.items()
        }

    def _read_aircraft_inventory(
        self, states: dict[str, dict | None],
    ) -> dict[str, Any]:
        version = states.get("version")
        high = self._state_value(states.get("serial_high"), "high")
        low = self._state_value(states.get("serial_low"), "low")
        return {
            "name": self._state_value(states.get("name"), "name"),
            "model": self._state_value(states.get("model"), "model"),
            "serial": (
                f"{high or ''}{low or ''}" if high is not None or low is not None else None
            ),
            "software": self._state_value(version, "software"),
            "hardware": self._state_value(version, "hardware"),
            "board_id": self._state_value(states.get("board_id"), "id"),
        }

    def _read_controller_inventory(
        self, errors: list[str],
    ) -> dict[str, Any] | None:
        if not self.via_skycontroller():
            return None
        try:
            from olympe.messages.skyctrl.SettingsState import (
                ProductSerialChanged,
                ProductVariantChanged,
                ProductVersionChanged as ControllerVersionChanged,
            )
        except Exception as exc:
            errors.append(f"controller inventory API unavailable: {exc!r}")
            return None
        controller_version = self._state_dict(ControllerVersionChanged)
        return {
            "variant": self._state_value(
                self._state_dict(ProductVariantChanged), "variant"
            ),
            "serial": self._state_value(
                self._state_dict(ProductSerialChanged), "serialNumber"
            ),
            "software": self._state_value(controller_version, "software"),
            "hardware": self._state_value(controller_version, "hardware"),
        }

    def _read_home_inventory(
        self, states: dict[str, dict | None],
    ) -> dict[str, Any]:
        home_raw = states.get("home")
        home = {
            "latitude": self._state_value(home_raw, "latitude"),
            "longitude": self._state_value(home_raw, "longitude"),
            "altitude": self._state_value(home_raw, "altitude"),
        }
        home["valid"] = self._valid_home(
            home["latitude"], home["longitude"], home["altitude"]
        )
        return home

    def _read_rth_states(
        self, errors: list[str],
    ) -> dict[str, dict | None]:
        try:
            from olympe.messages import rth
        except Exception as exc:
            errors.append(f"RTH inventory API unavailable: {exc!r}")
            return {}
        return {
            "home_reachability": self._state_dict(rth.home_reachability),
            "auto_trigger_mode": self._state_dict(rth.auto_trigger_mode),
            "delay": self._state_dict(rth.delay),
            "ending_behavior": self._state_dict(rth.ending_behavior),
        }

    def _read_lost_link_inventory(
        self,
        states: dict[str, dict | None],
        home: dict[str, Any],
        errors: list[str],
    ) -> tuple[dict[str, Any], bool]:
        rth_states = self._read_rth_states(errors)
        delay_raw = states.get("return_home_delay")
        delay_s = self._finite_float(
            self._state_value(rth_states.get("delay"), "delay")
        )
        if delay_s is None:
            delay_s = self._finite_float(self._state_value(delay_raw, "delay"))
        min_altitude_raw = states.get("return_home_min_altitude")
        reachability = self._state_value(
            rth_states.get("home_reachability"), "status"
        )
        reachability_name = str(reachability or "").rsplit(".", 1)[-1].lower()
        auto_trigger_name = str(
            self._state_value(rth_states.get("auto_trigger_mode"), "mode") or ""
        ).rsplit(".", 1)[-1].lower()
        ending_behavior_name = str(
            self._state_value(
                rth_states.get("ending_behavior"), "ending_behavior"
            ) or ""
        ).rsplit(".", 1)[-1].lower()
        # Split "the aircraft will come down by itself on link loss" from "it can fly
        # home first". The site has GPS but the signal is often poor, so a usable home
        # point must NOT be a takeoff precondition; the onboard policy itself must be.
        # With ending_behavior=landing the aircraft lands at the end of RTH, and with
        # no usable home it simply lands where it is -- which is the required fallback.
        lost_link_policy_ok = bool(
            delay_s is not None
            and int(delay_s) == _LOST_LINK_RTH_DELAY_S
            and auto_trigger_name == "on"
            and ending_behavior_name == "landing"
        )
        home_usable = bool(home["valid"] and reachability_name == "reachable")
        rth_policy_valid = bool(lost_link_policy_ok and home_usable)
        lost_link_fallback = "return_home_then_land" if home_usable else "land_in_place"
        rth_min_altitude = self._finite_float(
            self._state_value(min_altitude_raw, "value")
        )
        # Do NOT overwrite a value the lost-link configuration already read back and
        # confirmed: this ardrone3 mirror can lag or be absent, and blanking it makes
        # _rth_altitude_conflict() refuse RTH for the rest of the flight.
        if rth_min_altitude is not None or self.state.rth_min_altitude_m is None:
            self.state.rth_min_altitude_m = rth_min_altitude
        lost_link = {
            "return_home_delay_s": delay_s,
            "return_home_min_altitude_m": rth_min_altitude,
            "home_reachability": reachability,
            "auto_trigger_mode": self._state_value(
                rth_states.get("auto_trigger_mode"), "mode"
            ),
            "ending_behavior": self._state_value(
                rth_states.get("ending_behavior"), "ending_behavior"
            ),
            "valid_for_b1": rth_policy_valid,
            "policy_confirmed": lost_link_policy_ok,
            "home_usable": home_usable,
            "fallback": lost_link_fallback,
        }
        return lost_link, rth_policy_valid

    def _validate_connection_inventory(
        self,
        aircraft: dict[str, Any],
        controller: dict[str, Any] | None,
        home: dict[str, Any],
        lost_link: dict[str, Any],
        errors: list[str],
    ) -> None:
        aircraft_name = " ".join(
            str(aircraft.get(key) or "") for key in ("name", "model")
        ).lower()
        controller_variant = str((controller or {}).get("variant") or "").lower()
        if "anafi" not in aircraft_name:
            errors.append("connected aircraft is not an approved ANAFI model")
        if not aircraft.get("serial"):
            errors.append("aircraft serial unavailable")
        if not self.via_skycontroller():
            errors.append("formal flight requires SkyController 3")
        else:
            # ProductVariantChanged is never emitted by SkyController 3 firmware
            # 1.8.1 under Olympe 8.4.0 (verified on the reference unit: the state
            # stays uninitialized while serial/version arrive normally), so the
            # variant alone can never confirm the model. Fall back to the HID
            # product string the stick monitor already resolved -- that is the same
            # physical device, read from the kernel, and it is independent evidence
            # rather than an assumption.
            hid_name = str(
                getattr(self._stick_monitor, "device_name", "") or ""
            ).lower()
            confirmed = (
                "skycontroller3" in controller_variant.replace(" ", "")
                or "skycontroller3" in hid_name.replace(" ", "")
            )
            if not confirmed:
                errors.append(
                    "connected controller is not confirmed SkyController 3 "
                    f"(variant={controller_variant or 'unavailable'!r}, "
                    f"hid={hid_name or 'unavailable'!r})"
                )
            if not (controller or {}).get("serial"):
                errors.append("controller serial unavailable")
        if not lost_link["policy_confirmed"]:
            errors.append("lost-link auto-land policy readback is not confirmed")
        if not lost_link["home_usable"]:
            # Not an error: GPS is present but often weak at this site. The aircraft
            # still comes down on link loss, it just lands in place instead of
            # returning home. Make that explicit rather than silently degrading.
            self.log.event(
                "lost_link_fallback",
                fallback=lost_link["fallback"],
                home_valid=bool(home["valid"]),
                home_reachability=str(lost_link["home_reachability"]),
                note="no usable Home Point: link loss lands in place",
            )

    def _olympe_version(self) -> str | None:
        try:
            return importlib.metadata.version("parrot-olympe")
        except importlib.metadata.PackageNotFoundError:
            return None

    def _commit_connection_inventory(
        self,
        inventory: dict[str, Any],
        aircraft: dict[str, Any],
        controller: dict[str, Any] | None,
        home: dict[str, Any],
        rth_policy_valid: bool,
    ) -> None:
        self.connection_inventory = inventory
        self._inventory_takeoff_ready = bool(inventory["takeoff_inventory_ready"])
        self._inventory_block_reason = (
            "; ".join(inventory["block_reasons"])
            if inventory["block_reasons"] else "ready"
        )
        self.state.aircraft_identity = (
            f"{aircraft.get('name') or aircraft.get('model') or 'UNKNOWN'} "
            f"SN={aircraft.get('serial') or 'UNKNOWN'} FW={aircraft.get('software') or 'UNKNOWN'}"
        )
        self.state.controller_identity = (
            "DIRECT WIFI"
            if controller is None
            else f"{controller.get('variant') or 'UNKNOWN'} "
                 f"SN={controller.get('serial') or 'UNKNOWN'} "
                 f"FW={controller.get('software') or 'UNKNOWN'}"
        )
        self.state.home_valid = bool(home["valid"])
        self.state.rth_policy_valid = rth_policy_valid
        self._home_latitude_deg = self._finite_float(home["latitude"])
        self._home_longitude_deg = self._finite_float(home["longitude"])
        self.log.event("hardware_inventory", **inventory)
        if self.session_logs is not None:
            writer = getattr(self.session_logs, "write_inventory", None)
            if callable(writer) and not writer("hardware", inventory):
                self.log.event("hardware_inventory_receipt_failed")

    def read_connection_inventory(self) -> dict[str, Any]:
        """Read hardware, firmware, Home and lost-link state from Olympe cache."""
        errors: list[str] = []
        states = self._read_aircraft_inventory_states(errors)
        aircraft = self._read_aircraft_inventory(states)
        controller = self._read_controller_inventory(errors)
        home = self._read_home_inventory(states)
        lost_link, rth_policy_valid = self._read_lost_link_inventory(
            states, home, errors
        )
        olympe_version = self._olympe_version()
        self._validate_connection_inventory(
            aircraft, controller, home, lost_link, errors
        )
        inventory = {
            "aircraft": aircraft,
            "controller": controller,
            "runtime": {
                "olympe_version": olympe_version,
                "ip": self.ip,
                "transport": "skycontroller" if self.via_skycontroller() else "direct_wifi",
            },
            "home": home,
            "lost_link": lost_link,
            "takeoff_inventory_ready": not errors,
            "block_reasons": errors,
        }
        self._commit_connection_inventory(
            inventory, aircraft, controller, home, rth_policy_valid
        )
        return inventory

    def _state_dict(self, message_type) -> dict | None:
        if self.drone is None:
            return None
        try:
            value = self.drone.get_state(message_type)
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def _clear_firmware_safety_state(self) -> None:
        self.state.max_altitude_m = None
        self.state.max_distance_m = None
        self.state.distance_geofence_enabled = None
        self.state.max_tilt_deg = None
        self.state.max_vertical_speed_mps = None
        self.state.max_rotation_speed_dps = None

    def _read_firmware_safety_state(self) -> dict[str, dict | None]:
        """Refresh UI fields exclusively from Olympe state readback."""
        self._clear_firmware_safety_state()
        try:
            from olympe.messages.ardrone3.PilotingSettingsState import (
                MaxAltitudeChanged,
                MaxDistanceChanged,
                MaxTiltChanged,
                NoFlyOverMaxDistanceChanged,
            )
            from olympe.messages.ardrone3.SpeedSettingsState import (
                MaxRotationSpeedChanged,
                MaxVerticalSpeedChanged,
            )
        except Exception:
            states: dict[str, dict | None] = {}
        else:
            states = {
                "altitude": self._state_dict(MaxAltitudeChanged),
                "distance": self._state_dict(MaxDistanceChanged),
                "geofence": self._state_dict(NoFlyOverMaxDistanceChanged),
                "tilt": self._state_dict(MaxTiltChanged),
                "vertical_speed": self._state_dict(MaxVerticalSpeedChanged),
                "rotation_speed": self._state_dict(MaxRotationSpeedChanged),
            }

        def current(name: str) -> float | None:
            raw = states.get(name)
            return self._finite_float(raw.get("current")) if raw is not None else None

        geofence = states.get("geofence")
        geofence_raw = (
            geofence.get("shouldNotFlyOver") if geofence is not None else None
        )
        try:
            geofence_int = int(geofence_raw)
        except (TypeError, ValueError, OverflowError):
            geofence_int = -1

        self.state.max_altitude_m = current("altitude")
        self.state.max_distance_m = current("distance")
        self.state.distance_geofence_enabled = (
            bool(geofence_int) if geofence_int in {0, 1} else None
        )
        self.state.max_tilt_deg = current("tilt")
        self.state.max_vertical_speed_mps = current("vertical_speed")
        self.state.max_rotation_speed_dps = current("rotation_speed")
        return states

    def _flight_state_name(self) -> str:
        try:
            from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
        except Exception:
            return ""
        raw = self._state_dict(FlyingStateChanged)
        value = raw.get("state") if raw is not None else None
        return str(getattr(value, "name", value)).rsplit(".", 1)[-1].lower()

    def _clear_magnetometer_calibration_state(self) -> None:
        self.state.drone_magnetometer_required = None
        self.state.drone_magnetometer_started = None
        self.state.drone_magnetometer_axis = "unknown"
        self.state.drone_magnetometer_x_done = None
        self.state.drone_magnetometer_y_done = None
        self.state.drone_magnetometer_z_done = None
        self.state.drone_magnetometer_failed = None
        self.state.skycontroller_magnetometer_state = (
            "unknown" if self.via_skycontroller() else "not_applicable"
        )

    def _read_magnetometer_calibration_state(self) -> bool:
        """Read aircraft and SkyController calibration state from Olympe cache."""
        self._clear_magnetometer_calibration_state()
        if self.drone is None:
            return False
        try:
            from olympe.messages.common.CalibrationState import (
                MagnetoCalibrationAxisToCalibrateChanged,
                MagnetoCalibrationRequiredState,
                MagnetoCalibrationStartedChanged,
                MagnetoCalibrationStateChanged,
            )
        except Exception:
            return False

        required_state = self._state_dict(MagnetoCalibrationRequiredState)
        started_state = self._state_dict(MagnetoCalibrationStartedChanged)
        axis_state = self._state_dict(MagnetoCalibrationAxisToCalibrateChanged)
        progress_state = self._state_dict(MagnetoCalibrationStateChanged)

        def integer(raw: dict | None, key: str, allowed: set[int]) -> int | None:
            value = raw.get(key) if raw is not None else None
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return parsed if parsed in allowed else None

        required = integer(required_state, "required", {0, 1, 2})
        started = integer(started_state, "started", {0, 1})
        axis_value = self._state_value(axis_state, "axis")
        self.state.drone_magnetometer_required = required
        self.state.drone_magnetometer_started = (
            None if started is None else bool(started)
        )
        self.state.drone_magnetometer_axis = (
            str(axis_value) if axis_value is not None else "unknown"
        )
        for source, target in (
            ("xAxisCalibration", "drone_magnetometer_x_done"),
            ("yAxisCalibration", "drone_magnetometer_y_done"),
            ("zAxisCalibration", "drone_magnetometer_z_done"),
            ("calibrationFailed", "drone_magnetometer_failed"),
        ):
            value = integer(progress_state, source, {0, 1})
            setattr(self.state, target, None if value is None else bool(value))

        controller_ok = True
        if self.via_skycontroller():
            try:
                from olympe.messages.skyctrl.CalibrationState import (
                    MagnetoCalibrationStateV2,
                )
            except Exception:
                controller_ok = False
            else:
                controller_state = self._state_dict(MagnetoCalibrationStateV2)
                value = self._state_value(controller_state, "state")
                if value is None:
                    controller_ok = False
                else:
                    self.state.skycontroller_magnetometer_state = str(value)
        self._log_magnetometer_change()
        # Completeness describes the AIRCRAFT compass only. Whether the SkyController
        # can report its own calibration is informational: that compass feeds
        # pilot-referenced features this system never uses (operator policy
        # 2026-08-06), so an unreadable controller state must not be reported as an
        # unavailable AIRCRAFT state -- that message sent the operator off
        # calibrating a compass that was already valid.
        self.state.skycontroller_magnetometer_readable = controller_ok
        return required is not None and started is not None

    def _log_magnetometer_change(self) -> None:
        """Log compass state ON CHANGE only.

        This reader runs on every poll (~30 Hz), so an unconditional log would
        drown the command stream. Logging transitions makes the calibration
        outcome -- and whether it survived to the next session -- reviewable
        afterwards instead of being visible only on screen while it happens.
        """
        snapshot = (
            self.state.drone_magnetometer_required,
            self.state.drone_magnetometer_started,
            str(self.state.drone_magnetometer_axis),
            self.state.drone_magnetometer_x_done,
            self.state.drone_magnetometer_y_done,
            self.state.drone_magnetometer_z_done,
            self.state.drone_magnetometer_failed,
            str(self.state.skycontroller_magnetometer_state),
        )
        if snapshot == self._last_magnetometer_snapshot:
            return
        self._last_magnetometer_snapshot = snapshot
        required = self.state.drone_magnetometer_required
        self.log.event(
            "magnetometer_state",
            required={0: "valid", 1: "required", 2: "recommended"}.get(
                required, "unknown"),
            required_code=required,
            started=self.state.drone_magnetometer_started,
            axis=str(self.state.drone_magnetometer_axis),
            x_done=self.state.drone_magnetometer_x_done,
            y_done=self.state.drone_magnetometer_y_done,
            z_done=self.state.drone_magnetometer_z_done,
            failed=self.state.drone_magnetometer_failed,
            skycontroller=str(self.state.skycontroller_magnetometer_state),
        )

    def _magnetometer_control_error(self) -> str | None:
        """Return why takeoff/autonomy is unsafe, or None when calibration is usable."""
        complete = self._read_magnetometer_calibration_state()
        required = self.state.drone_magnetometer_required
        if not complete or required is None:
            return "aircraft magnetometer calibration state is unavailable"
        if not getattr(self.state, "skycontroller_magnetometer_readable", True):
            # Not a blocker -- say so once instead of failing the whole preflight.
            self.log.event(
                "magnetometer_calibration_warning",
                target="skycontroller",
                requirement="state_unreadable_not_required",
                state=str(self.state.skycontroller_magnetometer_state),
            )
        if self.state.drone_magnetometer_failed is True:
            return "aircraft magnetometer calibration failed"
        if self.state.drone_magnetometer_started is True:
            return "aircraft magnetometer calibration is in progress"
        if required == 1:
            return "aircraft magnetometer calibration is required"
        if required == 2:
            self.log.event(
                "magnetometer_calibration_warning",
                target="aircraft",
                requirement="recommended",
            )
        if self.via_skycontroller():
            raw = str(self.state.skycontroller_magnetometer_state or "unknown")
            controller = raw.rsplit(".", 1)[-1].replace("_", "").lower()
            # A calibration actively RUNNING still blocks: the controller is being
            # rotated by hand, so it is not in a state to pilot anything.
            if controller.startswith("calibrating"):
                return "SkyController magnetometer calibration is in progress"
            if controller != "calibrated":
                # Operator decision (2026-08-06): an uncalibrated CONTROLLER compass
                # does not block takeoff. It only feeds pilot-referenced features --
                # RTH preferred_home_type="pilot", follow-me, operator-relative modes
                # -- and this system uses none of them: it never calls
                # set_preferred_home_type (so home is the takeoff point) and flies
                # body-frame PCMD. The AIRCRAFT compass gate above is unchanged.
                # Warned and logged, never silent.
                self.log.event(
                    "magnetometer_calibration_warning",
                    target="skycontroller",
                    requirement="not_required_for_this_configuration",
                    state=raw,
                )
        return None

    def _prepare_magnetometer_calibration(self, target: str) -> bool:
        """Invalidate pending takeoff and clear PC motion before a ground calibration."""
        if self.drone is None or self._cleanup_done:
            self.log.event(
                "magnetometer_calibration",
                target=target,
                action="start",
                ok=False,
                reason="not_available",
            )
            return False
        flight_state = self._flight_state_name()
        if flight_state != "landed":
            self.log.event(
                "magnetometer_calibration",
                target=target,
                action="start",
                ok=False,
                reason="requires_confirmed_landed",
                flight_state=flight_state,
            )
            return False
        self._read_magnetometer_calibration_state()
        controller_state = str(
            self.state.skycontroller_magnetometer_state or ""
        ).replace("_", "").lower()
        if (
            self.state.drone_magnetometer_started is True
            or "calibrating" in controller_state
        ):
            self.log.event(
                "magnetometer_calibration",
                target=target,
                action="start",
                ok=False,
                reason="another_calibration_is_in_progress",
            )
            return False
        self.nudge_clear(reason=f"{target}_magnetometer_calibration")
        zero_sent = False
        with self._lock:
            self._pulse_token += 1
            if not self.pilot_sticks:
                try:
                    zero_sent = self._raw_pcmd(0, 0, 0, 0) is not None
                except Exception:
                    zero_sent = False
        self.state.mode = "MANUAL"
        self.state.last_command = f"{target}_magnetometer_prepare"
        self.log.event(
            "magnetometer_calibration_prepare",
            target=target,
            ok=True,
            zero_pcmd_sent=zero_sent,
            piloting_source=("SkyController" if self.pilot_sticks else "Controller"),
            pending_takeoff_invalidated=True,
        )
        return True

    def _wait_magnetometer_command(
            self, command: Any, *, target: str, action: str,
            timeout_s: float) -> bool:
        try:
            waited = self.drone(command).wait(_timeout=timeout_s)
            ok = bool(getattr(waited, "success", lambda: False)())
        except Exception as exc:
            self.log.event(
                "magnetometer_calibration",
                target=target,
                action=action,
                ok=False,
                error=repr(exc),
            )
            return False
        self._read_magnetometer_calibration_state()
        self.log.event(
            "magnetometer_calibration",
            target=target,
            action=action,
            ok=ok,
        )
        return ok

    def start_drone_magnetometer_calibration(self) -> bool:
        with self._calibration_lock:
            if not self._prepare_magnetometer_calibration("aircraft"):
                return False
            try:
                from olympe.messages.common.Calibration import MagnetoCalibration
            except Exception as exc:
                self.log.event(
                    "magnetometer_calibration",
                    target="aircraft",
                    action="start",
                    ok=False,
                    error=repr(exc),
                )
                return False
            ok = self._wait_magnetometer_command(
                MagnetoCalibration(calibrate=1),
                target="aircraft",
                action="start",
                timeout_s=5.0,
            )
            if ok:
                self._set_preflight_status(
                    False, "aircraft magnetometer calibration is in progress"
                )
            return ok

    def cancel_drone_magnetometer_calibration(self) -> bool:
        with self._calibration_lock:
            if self.drone is None or self._cleanup_done:
                return False
            try:
                from olympe.messages.common.Calibration import MagnetoCalibration
            except Exception:
                return False
            return self._wait_magnetometer_command(
                MagnetoCalibration(calibrate=0),
                target="aircraft",
                action="cancel",
                timeout_s=5.0,
            )

    def start_skycontroller_magnetometer_calibration(self) -> bool:
        with self._calibration_lock:
            if not self.via_skycontroller():
                self.log.event(
                    "magnetometer_calibration",
                    target="skycontroller",
                    action="start",
                    ok=False,
                    reason="not_connected_via_skycontroller",
                )
                return False
            if not self._prepare_magnetometer_calibration("skycontroller"):
                return False
            try:
                from olympe.messages.skyctrl.Calibration import StartCalibration
            except Exception as exc:
                self.log.event(
                    "magnetometer_calibration",
                    target="skycontroller",
                    action="start",
                    ok=False,
                    error=repr(exc),
                )
                return False
            ok = self._wait_magnetometer_command(
                StartCalibration(),
                target="skycontroller",
                action="start",
                timeout_s=12.0,
            )
            if ok:
                self._set_preflight_status(
                    False, "SkyController magnetometer calibration is in progress"
                )
            return ok

    def cancel_skycontroller_magnetometer_calibration(self) -> bool:
        with self._calibration_lock:
            if (
                self.drone is None
                or self._cleanup_done
                or not self.via_skycontroller()
            ):
                return False
            try:
                from olympe.messages.skyctrl.Calibration import AbortCalibration
            except Exception:
                return False
            return self._wait_magnetometer_command(
                AbortCalibration(),
                target="skycontroller",
                action="cancel",
                timeout_s=12.0,
            )

    def _is_airborne(self) -> bool:
        return self._flight_state_name() in {"takingoff", "hovering", "flying"}

    def _configure_lost_link_policy_if_safe(self) -> bool:
        """Set and verify onboard lost-link RTH while confirmed landed."""
        self.state.rth_policy_configured = False
        if self.drone is None or self._flight_state_name() != "landed":
            self.log.event(
                "lost_link_policy",
                ok=False,
                reason="requires_confirmed_landed",
            )
            return False
        try:
            from olympe.messages import rth
        except Exception as exc:
            self.log.event("lost_link_policy", ok=False, error=repr(exc))
            return False

        def name(raw: dict | None, key: str) -> str:
            value = self._state_value(raw, key)
            return str(value or "").rsplit(".", 1)[-1].lower()

        requested = (
            (
                "auto_trigger_mode",
                name(self._state_dict(rth.auto_trigger_mode), "mode") == "on",
                rth.set_auto_trigger_mode(mode="on"),
            ),
            (
                "delay",
                self._finite_float(
                    self._state_value(self._state_dict(rth.delay), "delay")
                ) == float(_LOST_LINK_RTH_DELAY_S),
                rth.set_delay(delay=_LOST_LINK_RTH_DELAY_S),
            ),
            (
                "ending_behavior",
                name(
                    self._state_dict(rth.ending_behavior), "ending_behavior"
                ) == "landing",
                rth.set_ending_behavior(ending_behavior="landing"),
            ),
            (
                "min_altitude",
                self._rth_min_altitude_matches(rth),
                rth.set_min_altitude(altitude=float(self.desired_rth_min_altitude_m)),
            ),
        )
        for field, matches, command in requested:
            if matches:
                continue
            try:
                waited = self.drone(command).wait(_timeout=10)
                ok = bool(getattr(waited, "success", lambda: False)())
            except Exception as exc:
                self.log.event(
                    "lost_link_policy", ok=False, field=field, error=repr(exc)
                )
                return False
            if not ok:
                self.log.event(
                    "lost_link_policy", ok=False, field=field,
                    reason="expectation_failed",
                )
                return False

        delay = self._finite_float(
            self._state_value(self._state_dict(rth.delay), "delay")
        )
        ok = bool(
            name(self._state_dict(rth.auto_trigger_mode), "mode") == "on"
            and delay == float(_LOST_LINK_RTH_DELAY_S)
            and name(
                self._state_dict(rth.ending_behavior), "ending_behavior"
            ) == "landing"
            and self._rth_min_altitude_matches(rth)
        )
        self.state.rth_policy_configured = ok
        self.state.rth_min_altitude_m = self._finite_float(
            self._state_value(self._state_dict(rth.min_altitude), "current")
        )
        self.log.event(
            "lost_link_policy",
            ok=ok,
            auto_trigger="on",
            delay_s=_LOST_LINK_RTH_DELAY_S,
            ending_behavior="landing",
            rth_min_altitude_m=self.state.rth_min_altitude_m,
        )
        return ok

    def _rth_min_altitude_matches(self, rth) -> bool:
        current = self._finite_float(
            self._state_value(self._state_dict(rth.min_altitude), "current")
        )
        if current is None:
            return False
        return math.isclose(
            current, float(self.desired_rth_min_altitude_m),
            rel_tol=1e-6, abs_tol=0.05,
        )

    def _desired_limits_error(self) -> str | None:
        altitude = self.desired_max_altitude_m
        distance = self.desired_max_distance_m
        if altitude is None and distance is None:
            return "max altitude and max distance are unset (ground UI only)"
        if altitude is None or distance is None:
            return "max altitude and max distance must both be set"
        if self._finite_float(altitude) is None or altitude <= 0.0:
            return "desired max altitude must be finite and > 0"
        if self._finite_float(distance) is None or distance <= 0.0:
            return "desired max distance must be finite and > 0"
        for label, value in (
            ("max tilt", self.desired_max_tilt_deg),
            ("max vertical speed", self.desired_max_vertical_speed_ms),
            ("max rotation speed", self.desired_max_rotation_speed_degs),
        ):
            if self._finite_float(value) is None or float(value) <= 0.0:
                return f"desired {label} must be finite and > 0"
        return None

    def _limit_bounds_error(
            self, raw: dict | None, desired: float, label: str) -> str | None:
        minimum = self._finite_float(raw.get("min")) if raw is not None else None
        maximum = self._finite_float(raw.get("max")) if raw is not None else None
        if minimum is None or maximum is None:
            return f"{label} firmware bounds unavailable"
        if not minimum <= desired <= maximum:
            return (
                f"{label} {desired:g} outside firmware range "
                f"[{minimum:g}, {maximum:g}]"
            )
        return None

    def _limit_readback_error(
            self, raw: dict | None, desired: float, label: str) -> str | None:
        actual = self._finite_float(raw.get("current")) if raw is not None else None
        if actual is None or not math.isclose(
                actual, desired, rel_tol=1e-5, abs_tol=0.05):
            return f"{label} readback mismatch: requested={desired:g} actual={actual!r}"
        return None

    def _set_preflight_status(self, ok: bool | None, reason: str) -> None:
        self.state.preflight_ok = ok
        self.state.preflight_reason = reason
        self.log.event("preflight", ok=ok, reason=reason)

    def _configure_firmware_limits_if_safe(self) -> bool:
        """Serialize firmware writes shared by connect, UI, and preflight."""
        with self._firmware_limits_lock:
            return self._configure_firmware_limits_locked()

    def _firmware_limits_fail(self, reason: str) -> bool:
        self._firmware_config_reason = reason
        self._read_firmware_safety_state()
        self._set_preflight_status(False, reason)
        self.log.event("firmware_limits", ok=False, reason=reason)
        return False

    def _firmware_setting_messages(self):
        try:
            from olympe.messages.ardrone3.PilotingSettings import (
                MaxAltitude,
                MaxDistance,
                MaxTilt,
                NoFlyOverMaxDistance,
            )
            from olympe.messages.ardrone3.PilotingSettingsState import (
                MaxAltitudeChanged,
                MaxDistanceChanged,
                MaxTiltChanged,
                NoFlyOverMaxDistanceChanged,
            )
            from olympe.messages.ardrone3.SpeedSettings import (
                MaxRotationSpeed,
                MaxVerticalSpeed,
            )
            from olympe.messages.ardrone3.SpeedSettingsState import (
                MaxRotationSpeedChanged,
                MaxVerticalSpeedChanged,
            )
        except Exception as exc:
            return None, f"firmware settings API unavailable: {exc!r}"
        return {
            "altitude": (MaxAltitude, MaxAltitudeChanged),
            "distance": (MaxDistance, MaxDistanceChanged),
            "tilt": (MaxTilt, MaxTiltChanged),
            "vertical_speed": (MaxVerticalSpeed, MaxVerticalSpeedChanged),
            "rotation_speed": (MaxRotationSpeed, MaxRotationSpeedChanged),
            "geofence": (NoFlyOverMaxDistance, NoFlyOverMaxDistanceChanged),
        }, None

    def _apply_firmware_setting(
        self,
        command,
        state_message,
        desired: float,
        label: str,
    ) -> bool:
        try:
            waited = self.drone(command).wait(_timeout=_FIRMWARE_SETTING_TIMEOUT_S)
            ok = bool(getattr(waited, "success", lambda: False)())
        except Exception as exc:
            return self._firmware_limits_fail(f"{label} command failed: {exc!r}")
        if not ok:
            return self._firmware_limits_fail(f"{label} command failed or timed out")
        readback_error = self._limit_readback_error(
            self._state_dict(state_message), desired, label,
        )
        return (
            True
            if readback_error is None
            else self._firmware_limits_fail(readback_error)
        )

    def _apply_firmware_geofence(self, command, state_message, value: int) -> bool:
        try:
            waited = self.drone(command(shouldNotFlyOver=value)).wait(
                _timeout=_FIRMWARE_SETTING_TIMEOUT_S
            )
            ok = bool(getattr(waited, "success", lambda: False)())
        except Exception as exc:
            return self._firmware_limits_fail(
                f"NoFlyOverMaxDistance command failed: {exc!r}"
            )
        if not ok:
            return self._firmware_limits_fail(
                "NoFlyOverMaxDistance command failed or timed out"
            )
        geofence_state = self._state_dict(state_message)
        actual = (
            geofence_state.get("shouldNotFlyOver")
            if geofence_state is not None else None
        )
        try:
            geofence_matches = int(actual) == value
        except (TypeError, ValueError, OverflowError):
            geofence_matches = False
        if not geofence_matches:
            return self._firmware_limits_fail(
                "NoFlyOverMaxDistance readback mismatch: "
                f"requested={value} actual={actual!r}"
            )
        return True

    def _apply_firmware_settings(
        self,
        messages,
        altitude: float,
        distance: float,
        tilt: float,
        vspeed: float,
        rspeed: float,
    ) -> bool:
        if not self._apply_firmware_setting(
            messages["altitude"][0](current=altitude),
            messages["altitude"][1],
            altitude,
            "MaxAltitude",
        ):
            return False
        if not self._apply_firmware_setting(
            messages["distance"][0](value=distance),
            messages["distance"][1],
            distance,
            "MaxDistance",
        ):
            return False
        # The speed envelope must be pinned and read back like the geofence limits:
        # every operator command is a percentage of these.
        if not self._apply_firmware_setting(
            messages["tilt"][0](current=tilt),
            messages["tilt"][1],
            tilt,
            "MaxTilt",
        ):
            return False
        if not self._apply_firmware_setting(
            messages["vertical_speed"][0](current=vspeed),
            messages["vertical_speed"][1],
            vspeed,
            "MaxVerticalSpeed",
        ):
            return False
        return self._apply_firmware_setting(
            messages["rotation_speed"][0](current=rspeed),
            messages["rotation_speed"][1],
            rspeed,
            "MaxRotationSpeed",
        )

    def _configure_firmware_limits_locked(self) -> bool:
        """Configure desired limits on connect, but only while confirmed landed."""
        self._firmware_config_ok = False
        self._read_firmware_safety_state()

        error = self._desired_limits_error()
        if error is not None:
            return self._firmware_limits_fail(error)
        if self._flight_state_name() != "landed":
            return self._firmware_limits_fail(
                "firmware limits not written: aircraft is not confirmed landed"
            )

        messages, message_error = self._firmware_setting_messages()
        if message_error is not None:
            return self._firmware_limits_fail(message_error)

        altitude = float(self.desired_max_altitude_m)
        distance = float(self.desired_max_distance_m)
        tilt = float(self.desired_max_tilt_deg)
        vspeed = float(self.desired_max_vertical_speed_ms)
        rspeed = float(self.desired_max_rotation_speed_degs)
        states = self._read_firmware_safety_state()
        for bounds_error in (
            self._limit_bounds_error(states.get("altitude"), altitude, "MaxAltitude"),
            self._limit_bounds_error(states.get("distance"), distance, "MaxDistance"),
            self._limit_bounds_error(states.get("tilt"), tilt, "MaxTilt"),
            self._limit_bounds_error(
                states.get("vertical_speed"), vspeed, "MaxVerticalSpeed"),
            self._limit_bounds_error(
                states.get("rotation_speed"), rspeed, "MaxRotationSpeed"),
        ):
            if bounds_error is not None:
                return self._firmware_limits_fail(bounds_error)

        if not self._apply_firmware_settings(
            messages, altitude, distance, tilt, vspeed, rspeed
        ):
            return False

        geofence_value = int(self.desired_distance_geofence)
        if not self._apply_firmware_geofence(
            messages["geofence"][0],
            messages["geofence"][1],
            geofence_value,
        ):
            return False

        self._read_firmware_safety_state()
        self._firmware_config_ok = True
        self._firmware_config_reason = "firmware limits confirmed"
        self._set_preflight_status(None, "firmware limits confirmed; takeoff not checked")
        self.log.event(
            "firmware_limits", ok=True, altitude_m=altitude,
            distance_m=distance, distance_geofence=bool(geofence_value),
            max_tilt_deg=tilt, max_vertical_speed_ms=vspeed,
            max_rotation_speed_degs=rspeed,
        )
        return True

    def apply_firmware_limits(
            self, max_altitude_m: Any, max_distance_m: Any,
            distance_geofence: Any = True) -> bool:
        """Apply operator-selected limits only while the aircraft is landed."""
        altitude = self._finite_float(max_altitude_m)
        distance = self._finite_float(max_distance_m)
        if altitude is None or altitude <= 0.0:
            self._set_preflight_status(False, "desired max altitude must be finite and > 0")
            return False
        if distance is None or distance <= 0.0:
            self._set_preflight_status(False, "desired max distance must be finite and > 0")
            return False

        with self._lock:
            if self._cleanup_done or self.drone is None:
                self._set_preflight_status(False, "firmware limits unavailable after cleanup")
                return False
            if self._flight_state_name() != "landed":
                reason = "firmware limits not written: aircraft is not confirmed landed"
                self._set_preflight_status(False, reason)
                self.log.event("firmware_limits_apply", ok=False, reason=reason)
                return False
            # Supersede any takeoff preflight that started before this update.
            self._pulse_token += 1
            self.desired_max_altitude_m = altitude
            self.desired_max_distance_m = distance
            self.desired_distance_geofence = bool(distance_geofence)
            self._firmware_config_ok = False

        ok = self._configure_firmware_limits_if_safe()
        self.log.event(
            "firmware_limits_apply", ok=ok, altitude_m=altitude,
            distance_m=distance,
            distance_geofence=bool(distance_geofence),
            reason=self._firmware_config_reason,
        )
        return ok

    def _controller_source_confirmed(self) -> bool:
        if not self.via_skycontroller():
            return True
        try:
            from olympe.messages.skyctrl.CoPilotingState import pilotingSource
        except Exception:
            return False
        raw = self._state_dict(pilotingSource)
        value = raw.get("source") if raw is not None else None
        name = str(getattr(value, "name", value)).rsplit(".", 1)[-1].lower()
        return name == "controller"

    def _restore_skycontroller_after_takeoff_abort(self, reason: str) -> bool:
        """Fail closed after a preflight handoff that does not reach TakeOff."""
        if not self.via_skycontroller():
            return True
        with self._lock:
            if self._maneuver_in_progress == "landing":
                self.log.event(
                    "takeoff_source_restore",
                    ok=True,
                    reason=reason,
                    note="landing_confirmation_in_progress",
                )
                return True
            self._zero_pcmd_or_log("takeoff_abort:handoff_zero")
        ok = self._set_piloting_source("SkyController")
        with self._lock:
            # Block PCMD even if the hardware ownership readback failed.
            self.pilot_sticks = True
        if self.state.tracker_state not in {"LAND", "LANDING", "LAND_UNCONFIRMED"}:
            self._set_tracker_state(TrackerState.STICKS if ok else TrackerState.SOURCE_FAIL, reason="_restore_skycontroller_after_takeoff_abort")
        self.log.event("takeoff_source_restore", ok=ok, reason=reason)
        return ok

    def _takeoff_preflight_fail(self, reason: str) -> bool:
        self._set_preflight_status(False, reason)
        return False

    def _takeoff_preflight_initial_checks(self):
        advisories: list[str] = []
        battery_floor = self._finite_float(self.min_takeoff_battery_pct)
        if battery_floor is None or not 0.0 <= battery_floor <= 100.0:
            self._takeoff_preflight_fail(
                "takeoff battery floor must be finite and within [0, 100]"
            )
            return None

        firmware_advisories: list[str] = []
        limit_config_error = self._desired_limits_error()
        if limit_config_error is not None:
            firmware_advisories.append(limit_config_error)
        if self._flight_state_name() != "landed":
            self._takeoff_preflight_fail("takeoff requires confirmed landed state")
            return None
        if self.via_skycontroller() and not self._stick_monitor_ready():
            self._takeoff_preflight_fail(
                "takeoff requires a healthy SkyController stick monitor"
            )
            return None
        if self.via_skycontroller():
            try:
                sticks_active = bool(self._stick_monitor.is_active())
            except Exception:
                self._takeoff_preflight_fail(
                    "takeoff requires readable SkyController stick state"
                )
                return None
            if sticks_active:
                self._takeoff_preflight_fail(
                    "takeoff refused while SkyController sticks are deflected"
                )
                return None

        link_ok = self._probe_link_ok()
        self.state.link_ok = link_ok
        self.state.link_status = "OK" if link_ok else "LOST"
        if not link_ok:
            self._takeoff_preflight_fail("takeoff requires a healthy Olympe link")
            return None

        calibration_error = self._magnetometer_control_error()
        if calibration_error is not None:
            self._takeoff_preflight_fail(calibration_error)
            return None
        if not bool(getattr(self.log, "durable", getattr(self.log, "path", None))):
            self._takeoff_preflight_fail("takeoff requires a durable safety log")
            return None
        if not bool(getattr(self.log, "healthy", True)):
            self._takeoff_preflight_fail("takeoff requires a healthy safety log")
            return None

        disk = assess_disk_space(self.disk_guard_path)
        self.state.disk_free_bytes = disk.free_bytes
        self.state.disk_free_percent = disk.free_percent
        self.state.disk_warning = disk.warning
        if disk.takeoff_blocked:
            self.log.event(
                "disk_low_takeoff_blocked",
                ok=False,
                reason=disk.reason,
                free_percent=disk.free_percent,
                free_bytes=disk.free_bytes,
            )
            self._takeoff_preflight_fail(disk.reason)
            return None
        return battery_floor, advisories, firmware_advisories, limit_config_error

    def _takeoff_preflight_connection_checks(self) -> bool:
        if not self._configure_lost_link_policy_if_safe():
            return self._takeoff_preflight_fail(
                "onboard lost-link RTH policy was not confirmed"
            )
        self.read_connection_inventory()
        if not self._inventory_takeoff_ready:
            return self._takeoff_preflight_fail(self._inventory_block_reason)

        if self.with_video:
            stamp = self._finite_float(
                getattr(self.video_stream, "last_stamp", None)
                if self.video_stream is not None else None
            )
            age = None if stamp is None else max(0.0, time.monotonic() - stamp)
            if stamp is None or stamp <= 0.0 or age is None or age > 1.0:
                return self._takeoff_preflight_fail(
                    "takeoff requires a live video frame newer than 1.0 s"
                )
        return True

    def _takeoff_preflight_battery_gps(
        self, battery_floor: float | None, advisories: list[str]
    ) -> bool:
        try:
            from olympe.messages.common.CommonState import BatteryStateChanged
        except Exception:
            return self._takeoff_preflight_fail("battery state API unavailable")
        else:
            battery_state = self._state_dict(BatteryStateChanged)
            battery = self._finite_float(
                battery_state.get("percent") if battery_state is not None else None
            )
            if battery is None or not 0.0 <= battery <= 100.0:
                return self._takeoff_preflight_fail(
                    "battery state unavailable or invalid"
                )
            self.state.battery_pct = battery
            battery_blocker = takeoff_battery_blocker(battery, battery_floor)
            if battery_blocker is not None:
                return self._takeoff_preflight_fail(battery_blocker)

        gps_required = bool(
            self.desired_distance_geofence and self.require_gps_for_geofence
        )
        try:
            from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
        except Exception:
            self.state.gps_fixed = None
            outcome = takeoff_gps_outcome(fixed=None, gps_required=gps_required)
        else:
            gps_state = self._state_dict(GPSFixStateChanged)
            gps_fixed = gps_state.get("fixed") if gps_state is not None else None
            try:
                fixed = int(gps_fixed) == 1
            except (TypeError, ValueError, OverflowError):
                fixed = False
            self.state.gps_fixed = fixed
            outcome = takeoff_gps_outcome(fixed=fixed, gps_required=gps_required)
        if outcome.blocker is not None:
            return self._takeoff_preflight_fail(outcome.blocker)
        if outcome.advisory is not None:
            advisories.append(outcome.advisory)
        return True

    def _takeoff_preflight_epoch_current(self, expected_epoch: int | None) -> bool:
        if expected_epoch is None:
            return True
        with self._lock:
            return expected_epoch == self._pulse_token and not self._cleanup_done

    def _takeoff_firmware_advisories(
        self, altitude: float, distance: float
    ) -> list[str]:
        states = self._read_firmware_safety_state()
        advisories = []
        for limit_error in (
            self._limit_bounds_error(states.get("altitude"), altitude, "MaxAltitude"),
            self._limit_bounds_error(states.get("distance"), distance, "MaxDistance"),
            self._limit_readback_error(states.get("altitude"), altitude, "MaxAltitude"),
            self._limit_readback_error(states.get("distance"), distance, "MaxDistance"),
        ):
            if limit_error is not None:
                advisories.append(limit_error)

        geofence = states.get("geofence")
        actual = geofence.get("shouldNotFlyOver") if geofence is not None else None
        try:
            geofence_matches = int(actual) == int(self.desired_distance_geofence)
        except (TypeError, ValueError, OverflowError):
            geofence_matches = False
        if not geofence_matches:
            advisories.append(
                "NoFlyOverMaxDistance readback mismatch: "
                f"requested={int(self.desired_distance_geofence)} actual={actual!r}"
            )
        return advisories

    def _takeoff_preflight_firmware(
        self,
        expected_epoch: int | None,
        limit_config_error: str | None,
        firmware_advisories: list[str],
    ) -> bool:
        if limit_config_error is not None:
            # A completely unset limit pair is the intentional ground-UI-only
            # mode. Any malformed configured limit must block takeoff.
            if (
                self.desired_max_altitude_m is not None
                or self.desired_max_distance_m is not None
            ):
                return self._takeoff_preflight_fail(limit_config_error)
            return True

        altitude = float(self.desired_max_altitude_m)
        distance = float(self.desired_max_distance_m)
        if not self._firmware_config_ok:
            if not self._takeoff_preflight_epoch_current(expected_epoch):
                return self._takeoff_preflight_fail("takeoff preflight was superseded")
            if not self._configure_firmware_limits_if_safe():
                return self._takeoff_preflight_fail(
                    "firmware limit configuration retry failed: "
                    f"{self._firmware_config_reason}"
                )
        readback_errors = self._takeoff_firmware_advisories(altitude, distance)
        if readback_errors:
            return self._takeoff_preflight_fail(readback_errors[0])
        return True

    def _takeoff_preflight_finalize(
        self,
        expected_epoch: int | None,
        advisories: list[str],
        firmware_advisories: list[str],
    ) -> bool:
        advisories.extend(firmware_advisories)
        if advisories:
            self.log.event(
                "takeoff_advisory",
                warnings=tuple(dict.fromkeys(advisories)),
                note=(
                    "GPS advisories do not block takeoff when the geofence "
                    "policy permits it"
                ),
            )

        if not self._takeoff_preflight_epoch_current(expected_epoch):
            return self._takeoff_preflight_fail("takeoff preflight was superseded")
        if not self._set_piloting_source("Controller"):
            self._restore_skycontroller_after_takeoff_abort(
                "controller_handoff_unconfirmed",
            )
            return self._takeoff_preflight_fail(
                "Controller piloting source command was not confirmed"
            )
        if not self._controller_source_confirmed():
            self._restore_skycontroller_after_takeoff_abort(
                "controller_readback_unconfirmed",
            )
            return self._takeoff_preflight_fail(
                "Controller piloting source readback is unconfirmed"
            )

        if not firmware_advisories:
            self._firmware_config_ok = True
            self._firmware_config_reason = "firmware limits confirmed"
        status = "ready" if not advisories else "ready with advisory warnings"
        self._set_preflight_status(True, status)
        return True

    def _takeoff_preflight(self, expected_epoch: int | None = None) -> bool:
        """Fail-closed gates with one landed-only config retry if needed."""
        checks = self._takeoff_preflight_initial_checks()
        if checks is None:
            return False
        battery_floor, advisories, firmware_advisories, limit_config_error = checks
        if not self._takeoff_preflight_connection_checks():
            return False
        if not self._takeoff_preflight_battery_gps(battery_floor, advisories):
            return False
        if not self._takeoff_preflight_firmware(
            expected_epoch, limit_config_error, firmware_advisories
        ):
            return False
        return self._takeoff_preflight_finalize(
            expected_epoch, advisories, firmware_advisories
        )

    # ------------------------------------------------------------------ wire
    def _zero_pcmd_or_log(self, reason: str) -> bool:
        """Send the safety zero PCMD. A failure here is never silent.

        Every caller below -- land, cleanup, RTH, manual handoff -- proceeds as if
        the aircraft is now holding still. When the send fails and nobody says so,
        the aircraft keeps flying its last non-zero command instead, and the only
        evidence is that it did not stop.

        Callers already holding self._lock must keep holding it: this does not
        re-acquire, matching the inline sends it replaces.
        """
        try:
            call_mono_ns = self._raw_pcmd(0, 0, 0, 0)
        except Exception as exc:
            self.zero_pcmd_failures += 1
            self.last_zero_pcmd_error = repr(exc)
            self.log.event(
                "pcmd_zero_failed", reason=reason, error=repr(exc),
                failures=self.zero_pcmd_failures,
            )
            return False
        if call_mono_ns is None:
            # No transport to send on (drone is None / disconnected). Nothing
            # was sent, so this must not report success: the aircraft is not
            # "now confirmed holding still", it just never got a command --
            # exactly what a caller trusting True would fail to notice.
            self.log.event("pcmd_zero_skipped_no_drone", reason=reason)
            return False
        # Deliberately NOT logged on success. Callers hold self._lock across this,
        # and a log write per safety zero perturbed the RTH handoff enough to make
        # test_lost_controller_returns_home_when_gps_home_is_usable land instead
        # (reproduced: reverting this helper made the file pass 3/3). What had to
        # stop being silent was FAILURE, and that is still recorded above.
        self.last_zero_pcmd_call_mono_ns = call_mono_ns
        return True

    def _raw_pcmd(self, roll: int, pitch: int, yaw: int, gaz: int) -> int | None:
        r, p, y, g = map(_clamp_pct, (roll, pitch, yaw, gaz))
        if self.drone is None:
            return None
        from olympe.messages.ardrone3.Piloting import PCMD
        call_mono_ns = time.monotonic_ns()
        self.drone(PCMD(1, r, p, y, g, 0))
        self._last_pcmd = (r, p, y, g)
        self._last_pcmd_mono_ns = call_mono_ns
        self.state.last_pcmd_call_mono_ns = call_mono_ns
        return call_mono_ns

    def send_pcmd(
        self, roll: int, pitch: int, yaw: int, gaz: int, *, reason: str,
        motion_epoch: int | None = None,
    ) -> bool:
        with self._lock:
            if self._shutdown_latched or self._cleanup_done or self._landed:
                return False
            if self.pilot_sticks:
                self.log.event("pcmd_blocked_manual", reason=reason,
                               pcmd=(roll, pitch, yaw, gaz))
                return False
            if motion_epoch is not None and motion_epoch != self._motion_epoch:
                self.log.event(
                    "pcmd_blocked_stale_motion",
                    reason=reason,
                    expected_motion_epoch=motion_epoch,
                    motion_epoch=self._motion_epoch,
                    pcmd=(roll, pitch, yaw, gaz),
                )
                return False
            maneuver = self._maneuver_in_progress
            if maneuver is not None and (roll, pitch, yaw, gaz) != (0, 0, 0, 0):
                # An automated TakeOff/Landing is executing. Zero still gets through
                # (it only reinforces hover); motion must not fight the manoeuvre.
                self.log.event("pcmd_blocked_maneuver", reason=reason,
                               maneuver=maneuver, pcmd=(roll, pitch, yaw, gaz))
                return False
            pcmd_call_mono_ns = self._raw_pcmd(roll, pitch, yaw, gaz)
            if pcmd_call_mono_ns is None:
                # drone is None (disconnected): nothing was sent. Reporting
                # True here would tell the caller a command reached the
                # aircraft when none did.
                self.log.event(
                    "pcmd_skipped_no_drone", reason=reason, pcmd=(roll, pitch, yaw, gaz)
                )
                return False
            # Hold loop is 20 Hz — log at most ~2 Hz for hold reasons to cut latency/IO.
            if str(reason).startswith("nudge_hold:"):
                now = time.monotonic()
                last = getattr(self, "_last_hold_log_t", 0.0)
                if now - last < 0.5:
                    return True
                self._last_hold_log_t = now
            self.log.event(
                "pcmd", reason=reason, pcmd=(roll, pitch, yaw, gaz),
                pcmd_call_mono_ns=pcmd_call_mono_ns,
            )
            return True

    def give_to_pilot(self, *, reason: str = "manual") -> bool:
        # Manual/safety intent must pre-empt PCMD immediately, even when a
        # blocking Controller handoff currently owns the serialized transition.
        # The source change itself remains serialized below.
        with self._lock:
            self.pilot_sticks = True
            self._pulse_token += 1
            self._motion_epoch += 1
            self._zero_pcmd_or_log(f"{reason}:manual_intent_zero")
        with self.authority_controller.transition():
            return self._give_to_pilot(reason=reason)

    def _give_to_pilot(self, *, reason: str = "manual") -> bool:
        self.nudge_clear(reason=reason if reason != "manual" else "manual")
        with self._lock:
            # The public entry point already latched manual ownership before it
            # waited for the serialized source transition. Internal restoration
            # calls also reinforce that latch here.
            self.pilot_sticks = True
            self._zero_pcmd_or_log(f"{reason}:manual_handoff_zero")
        if self.via_skycontroller():
            ok = self._set_piloting_source("SkyController")
            if ok:
                detail = "PC silent; SkyController sticks active"
                self._set_tracker_state(TrackerState.STICKS, reason="_give_to_pilot")
            else:
                detail = "PC silent; SkyController source unconfirmed"
                self._set_tracker_state(TrackerState.SOURCE_FAIL, reason="_give_to_pilot")
        else:
            # No sticks on the link: Esc freezes PC PCMD (hover) until 恢復電腦控制.
            ok = True
            detail = "PC PCMD frozen (direct WiFi; no SC sticks)"
            self._set_tracker_state(TrackerState.PC_FROZEN, reason="_give_to_pilot")
        self.state.mode = "MANUAL"
        self.state.control_owner = (
            "SKYCONTROLLER" if self.via_skycontroller() else "PC_MANUAL_ZERO"
        )
        self.state.last_command = reason if reason else "manual"
        self.log.event("manual", detail=detail, ok=ok, reason=reason)
        return ok

    def _start_stick_monitor(self) -> None:
        self._stop_stick_monitor()
        monitor = SkyControllerStickMonitor(
            on_active=self._on_stick_active,
            on_disconnect=self._on_stick_monitor_disconnect,
            log_event=self.log.event,
        )
        if monitor.start():
            self._stick_monitor = monitor
            self.state.stick_monitor_ok = True
        else:
            self._stick_monitor = None
            self.state.stick_monitor_ok = False
            self._set_tracker_state(TrackerState.STICK_MONITOR_FAIL, reason="_start_stick_monitor")
            self.state.active_incident = FailureReason.CONTROLLER_DISCONNECTED.value

    def _stop_stick_monitor(self) -> None:
        mon = self._stick_monitor
        self._stick_monitor = None
        if mon is not None:
            mon.stop()

    def _on_stick_active(self, axes: dict[int, int]) -> None:
        """HID callback: any deliberate stick throw reclaims SkyController."""
        self._maybe_reclaim_from_sticks(axes)

    def _stick_monitor_ready(self) -> bool:
        if not self.via_skycontroller():
            return True
        monitor = self._stick_monitor
        return bool(
            monitor is not None
            and getattr(monitor, "healthy", True)
            and getattr(self.state, "stick_monitor_ok", False)
        )

    def _on_stick_monitor_disconnect(self, reason: str) -> None:
        """Latch HID loss; land if the command link is still usable."""
        self.state.stick_monitor_ok = False
        self.state.active_incident = FailureReason.CONTROLLER_DISCONNECTED.value
        self._set_tracker_state(TrackerState.STICK_MONITOR_FAIL, reason="_on_stick_monitor_disconnect")
        self.log.event(
            "controller_disconnect",
            reason=reason,
            flight_state=self._flight_state_name(),
        )
        if not self._is_airborne():
            return
        if self._probe_link_ok():
            self._schedule_runtime_safety_action(
                FailureReason.CONTROLLER_DISCONNECTED.value
            )
        else:
            self._latch_total_link_loss()

    def _maybe_reclaim_from_sticks(self, axes: dict[int, int] | None = None) -> bool:
        """Force piloting source back to sticks when operator moves them.

        Only while PC still holds authority (pilot_sticks=False) on SC path.
        Idempotent and rate-limited so a held stick does not spam handoffs.
        """
        if not self.via_skycontroller():
            return False
        with self._lock:
            if self._cleanup_done or self.drone is None:
                return False
            if self.pilot_sticks:
                return False
        if axes is None:
            mon = self._stick_monitor
            if mon is None or not mon.is_active():
                return False
            axes = mon.snapshot_axes()
        elif not axes_active(axes):
            return False
        now = time.monotonic()
        with self._stick_reclaim_lock:
            # Allow at most one reclaim attempt every 0.25s while sticks held.
            if (now - self._last_stick_reclaim_mono) < 0.25:
                return False
            self._last_stick_reclaim_mono = now
        ok = self.give_to_pilot(reason="stick_override")
        if ok:
            self.stick_override_count += 1
            active = {
                str(k): int(v)
                for k, v in sorted((axes or {}).items())
                if int(k) in _STICK_FLIGHT_AXES and abs(int(v)) > _STICK_DEADZONE
            }
            self.log.event(
                "stick_override",
                ok=True,
                count=self.stick_override_count,
                axes=active,
                note="physical_stick_forced_skycontroller",
            )
        else:
            self.log.event("stick_override", ok=False, note="give_to_pilot_failed")
        return ok

    def _auto_take_pc_control_if_safe(self) -> bool:
        """Start the session in PC control (operator decision 2026-08-06).

        The three conditions below are the ones that made handing PC authority on
        connect unsafe in the first place, so they are kept:
          * confirmed LANDED -- a UI restarted mid-flight must never seize control
            from the pilot who is currently flying;
          * sticks not deflected -- the pilot is already commanding;
          * stick monitor healthy -- otherwise nothing can take control back.
        Stick movement still forces control back at any time; that path is
        untouched.
        """
        if not self.auto_pc_control:
            return False
        if self._flight_state_name() != "landed":
            self.log.event("auto_pc_control", ok=False, reason="not_confirmed_landed")
            return False
        if self.via_skycontroller() and not self._stick_monitor_ready():
            self.log.event("auto_pc_control", ok=False, reason="stick_monitor_unavailable")
            return False
        mon = self._stick_monitor
        if mon is not None and mon.is_active():
            self.log.event("auto_pc_control", ok=False, reason="sticks_deflected")
            return False
        ok = self.take_pc_control()
        self.log.event("auto_pc_control", ok=ok)
        return ok

    def take_pc_control(self) -> bool:
        with self._lock:
            if self._shutdown_latched:
                self.log.event("pc_control", ok=False, note="shutdown_latched")
                return False
        with self.authority_controller.transition():
            return self._take_pc_control()

    def _take_pc_control(self) -> bool:
        """Explicitly request PC authority; remain PCMD-blocked unless confirmed."""
        if self.via_skycontroller() and not self._stick_monitor_ready():
            with self._lock:
                self.pilot_sticks = True
            self._set_tracker_state(TrackerState.STICK_MONITOR_FAIL, reason="_take_pc_control")
            self.log.event(
                "pc_control", ok=False, note="stick_monitor_unavailable"
            )
            return False
        # If sticks are already deflected, refuse PC grab — pilot has priority.
        mon = self._stick_monitor
        if mon is not None and mon.is_active():
            with self._lock:
                self.pilot_sticks = True
            self._set_tracker_state(TrackerState.STICKS, reason="_take_pc_control")
            self.log.event(
                "pc_control",
                ok=False,
                note="sticks_active_refuse_pc",
            )
            # Ensure firmware source matches sticks when possible.
            self._set_piloting_source("SkyController")
            return False
        self.nudge_clear(reason="pc_control")
        # The piloting-source handoff below blocks (up to 10 s). An EMERGENCY stop,
        # LAND, stream-loss or link-loss handoff can latch during that window, and
        # every one of those paths bumps _pulse_token. Capture the epoch first and
        # refuse to unblock PCMD if it moved -- otherwise this hands PC command
        # authority back AFTER an emergency stop has already taken it away.
        control_epoch = self._pulse_token
        if not self._set_piloting_source("Controller"):
            with self._lock:
                self.pilot_sticks = True
            self._set_tracker_state(TrackerState.SOURCE_FAIL, reason="_take_pc_control")
            self.log.event("pc_control", ok=False, note="source_unconfirmed")
            return False
        # Re-sample the sticks AFTER the blocking handoff: the pilot may have grabbed
        # them during it, and the pre-handoff sample is by then up to 10 s stale.
        mon = self._stick_monitor
        if mon is not None and mon.is_active():
            with self._lock:
                self.pilot_sticks = True
            self._set_tracker_state(TrackerState.STICKS, reason="_take_pc_control")
            self.log.event("pc_control", ok=False, note="sticks_active_during_handoff")
            self._set_piloting_source("SkyController")
            return False
        with self._lock:
            if self._shutdown_latched or self._cleanup_done or self.drone is None:
                self.pilot_sticks = True
                self.log.event("pc_control", ok=False, note="cleanup_or_disconnected")
                return False
            if control_epoch != self._pulse_token:
                self.pilot_sticks = True
                self._set_tracker_state(TrackerState.SOURCE_FAIL, reason="_take_pc_control")
                self.log.event(
                    "pc_control", ok=False, note="safety_latched_during_handoff",
                )
                superseded = True
            else:
                superseded = False
            if not superseded:
                self.pilot_sticks = False
        if superseded:
            # A safety/manual epoch may have changed while the blocking source
            # request was in flight. Restore the physical/manual owner before
            # returning; leaving firmware on Controller makes the False result
            # lie about who actually owns the aircraft.
            self._give_to_pilot(reason="pc_control_superseded")
            self._set_tracker_state(TrackerState.SOURCE_FAIL, reason="_take_pc_control")
            return False
        if not self._landed and not self.send_pcmd(
                0, 0, 0, 0, reason="pc_control_resumed"):
            self._give_to_pilot(reason="pc_control_zero_failed")
            self._set_tracker_state(TrackerState.SOURCE_FAIL, reason="_take_pc_control")
            self.log.event("pc_control", ok=False, note="zero_pcmd_failed")
            return False
        self.state.mode = "MANUAL"
        self._set_tracker_state(TrackerState.PC, reason="_take_pc_control")
        self.log.event("pc_control", ok=True)
        return True

    def hover_cmd(self, reason: str = "hover") -> bool:
        if self.pilot_sticks:
            self.log.event("hover_blocked_manual", reason=reason)
            return False
        with self._lock:
            if self._maneuver_in_progress == "takeoff":
                # Hover is an explicit operator cancellation of an AUTO launch.
                # Invalidate the takeoff epoch before sending zero so a blocking
                # preflight/source handoff cannot schedule TakeOff afterwards.
                self._pulse_token += 1
        self.nudge_clear(reason=reason)
        if not self.send_pcmd(0, 0, 0, 0, reason=reason):
            self.log.event("hover", reason=reason, ok=False)
            return False
        self._set_tracker_state(TrackerState.HOVER, reason="hover_cmd")
        self.state.mode = "MANUAL"
        return True

    def _land_and_confirm(self, reason: str) -> bool:
        """Issue Landing and only report success after the landed state event."""
        if self.drone is None:
            self.log.event("land_cmd", reason=reason, ok=False, error="no_drone")
            return False
        try:
            from olympe.messages.ardrone3.Piloting import Landing
            from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

            exp = self.drone(
                Landing() >> FlyingStateChanged(state="landed", _timeout=20)
            )
            self._set_tracker_state(TrackerState.LANDING, reason="_land_and_confirm")
            waited = exp.wait(_timeout=22)
            ok = bool(getattr(waited, "success", lambda: False)())
        except Exception as exc:
            # Last-ditch command issue is intentionally not treated as touchdown.
            try:
                from olympe.messages.ardrone3.Piloting import Landing
                self.drone(Landing())
                self.log.event(
                    "land_cmd", reason=reason, ok=False, command_issued=True,
                    error=repr(exc), note="touchdown_unconfirmed",
                )
            except Exception as exc2:
                self.log.event(
                    "land_cmd", reason=reason, ok=False, command_issued=False,
                    error=repr(exc2),
                )
            self._set_tracker_state(TrackerState.LAND_UNCONFIRMED, reason="_land_and_confirm")
            return False

        if ok:
            with self._lock:
                self._landed = True
            self._set_tracker_state(TrackerState.LAND, reason="_land_and_confirm")
        else:
            self._set_tracker_state(TrackerState.LAND_UNCONFIRMED, reason="_land_and_confirm")
        self.log.event(
            "land_cmd", reason=reason, ok=ok,
            note="touchdown_confirmed" if ok else "touchdown_unconfirmed",
        )
        return ok

    def _recover_after_scheduled_takeoff_failure(self, reason: str) -> bool:
        """A scheduled TakeOff failure is airborne-unknown until Landing confirms."""
        outcome = self.takeoff_landing_supervisor.ensure_landed(
            reason=f"takeoff_failure:{reason}",
            is_landed=self._is_already_landed,
            land_and_confirm=self._land_and_confirm,
            force_command=True,
        )
        self._restore_skycontroller_after_takeoff_abort(reason)
        self.log.event(
            "takeoff_recovery",
            reason=reason,
            landed=outcome.confirmed,
            reason_code=outcome.reason_code,
        )
        return outcome.confirmed

    def land_cmd(self, reason: str = "land") -> bool:
        # 【原地降落 — 禁止隨意改】UI「原地降落」；語意改壞可能不降或誤降。
        if self._cleanup_done or self.drone is None:
            self.log.event("land_cmd", reason=reason, ok=False, error="not_available")
            return False
        self.nudge_clear(reason=f"land:{reason}")
        with self._lock:
            self._pulse_token += 1
            self._maneuver_in_progress = "landing"
            self.state.mode = "MANUAL"
            self.state.last_command = "land"
            self._zero_pcmd_or_log("land:pre_handoff_zero")
        source_ok = self._set_piloting_source("Controller")
        if source_ok:
            with self._lock:
                self.pilot_sticks = False
        else:
            self.log.event(
                "land_source_unconfirmed", reason=reason,
                note="Landing still attempted as fail-safe",
            )
        with self._lock:
            self._zero_pcmd_or_log("land:post_handoff_zero")

        if self._is_already_landed():
            with self._lock:
                self._landed = True
            self._set_tracker_state(TrackerState.LAND, reason="land_cmd")
            landed = True
            self.log.event(
                "land_cmd", reason=reason, ok=True, note="already_landed",
            )
        else:
            landed = self._land_and_confirm(reason)

        if landed and self.via_skycontroller():
            handed_off = self._set_piloting_source("SkyController")
            with self._lock:
                self.pilot_sticks = handed_off
        elif landed:
            with self._lock:
                self.pilot_sticks = True

        # Recording/media work is deliberately after the landing command and
        # confirmed touchdown so it can never delay the safety action.
        if landed and self.recording_active:
            self.stop_flight_recording(reason=f"land:{reason}")
        elif not landed and self.recording_active:
            self.log.event(
                "record_stop_deferred", reason=f"land:{reason}",
                note="touchdown_unconfirmed",
            )
        # Release the manoeuvre block. On an UNCONFIRMED landing the aircraft may still
        # be airborne, and the operator must keep the ability to reposition it -- a
        # permanently-held block would strand them with hover-only authority.
        self._clear_maneuver("landing")
        return landed

    def _clear_maneuver(self, maneuver: str | None = None) -> None:
        with self._lock:
            if maneuver is None or self._maneuver_in_progress == maneuver:
                self._maneuver_in_progress = None

    def _takeoff_cmd_handle_exception(
        self, exc: Exception, takeoff_epoch: int, takeoff_scheduled: bool
    ) -> bool:
        self.log.event("takeoff", ok=False, error=repr(exc))
        with self._lock:
            current = (
                takeoff_epoch == self._pulse_token
                and not self._cleanup_done
                and not self._landed
            )
        if current and takeoff_scheduled:
            self._recover_after_scheduled_takeoff_failure(
                "takeoff_exception"
            )
            if not self._landed:
                self._set_tracker_state(TrackerState.TAKEOFF_FAIL, reason="_takeoff_cmd_handle_exception")
        else:
            self._restore_skycontroller_after_takeoff_abort(
                "takeoff_exception_superseded"
                if takeoff_scheduled
                else "takeoff_exception_before_schedule"
            )
        return False

    def _takeoff_cmd_run(self, takeoff_epoch: int) -> bool:
        takeoff_scheduled = False
        try:
            from olympe.messages.ardrone3.Piloting import TakeOff
            from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
            takeoff_request = (
                TakeOff() >> FlyingStateChanged(state="hovering", _timeout=15)
            )
            with self._lock:
                # Atomic with land/cleanup epoch increments: if land got this
                # lock first, TakeOff is never scheduled; if TakeOff got it
                # first, a later Landing is necessarily queued afterwards.
                current = (
                    takeoff_epoch == self._pulse_token
                    and not self._cleanup_done
                    and self.drone is not None
                )
                if current:
                    self.pilot_sticks = False
                    self._landed = False
                    takeoff_expectation = self.drone(takeoff_request)
                    takeoff_scheduled = True
                else:
                    takeoff_expectation = None
            if takeoff_expectation is None:
                self._restore_skycontroller_after_takeoff_abort(
                    "superseded_before_schedule",
                )
                self._set_preflight_status(
                    False, "takeoff schedule superseded by a safety command",
                )
                self.log.event(
                    "takeoff", ok=False, note="superseded_before_schedule",
                )
                return False

            # Never hold _lock while waiting up to 17 s; land/cleanup must be
            # able to schedule their command immediately after TakeOff.
            ok = takeoff_expectation.wait(_timeout=17).success()
            with self._lock:
                current = (
                    takeoff_epoch == self._pulse_token
                    and not self._cleanup_done
                    and not self._landed
                )
            if not current:
                self._restore_skycontroller_after_takeoff_abort(
                    "superseded_during_wait",
                )
                self._set_preflight_status(
                    False, "takeoff wait superseded by a safety command",
                )
                self.log.event(
                    "takeoff", ok=False, note="superseded_by_safety_command",
                )
                return False
            if not ok:
                self._recover_after_scheduled_takeoff_failure(
                    "hover_unconfirmed"
                )
                self.log.event("takeoff", ok=False, note="hover_unconfirmed")
                if not self._landed:
                    self._set_tracker_state(TrackerState.TAKEOFF_FAIL, reason="_takeoff_cmd_run")
                return False

            self.log.event("takeoff", ok=True)
            self._set_tracker_state(TrackerState.HOVER, reason="_takeoff_cmd_run")
            if self.flight_start is None:
                self.flight_start = time.monotonic()
            self.send_pcmd(0, 0, 0, 0, reason="post_takeoff")
            # Auto-record only after successful takeoff when operator armed it.
            with self._lock:
                current = (
                    takeoff_epoch == self._pulse_token
                    and not self._cleanup_done
                    and not self._landed
                )
            if current and self.record_on_takeoff:
                self.start_flight_recording(reason="post_takeoff")
            return True
        except Exception as exc:
            return self._takeoff_cmd_handle_exception(
                exc, takeoff_epoch, takeoff_scheduled
            )

    def takeoff_cmd(self) -> bool:
        # 【起飛 — 僅操作員親手按起飛／自動飛行 UI】禁止腳本／AI 代理人／任何 LLM 代為呼叫。
        if self.drone is None or self._shutdown_latched or self._cleanup_done:
            return False
        self.nudge_clear(reason="takeoff")
        with self._lock:
            if self._shutdown_latched or self._cleanup_done or self.drone is None:
                return False
            # Snapshot before the blocking source acknowledgement as well as
            # the takeoff wait. Any later land/cleanup/manual invalidates it.
            self._pulse_token += 1
            takeoff_epoch = self._pulse_token
            self._maneuver_in_progress = "takeoff"
        preflight_ok = self._takeoff_preflight(takeoff_epoch)
        with self._lock:
            current = (
                takeoff_epoch == self._pulse_token
                and not self._cleanup_done
                and self.drone is not None
            )
        if not current:
            self._restore_skycontroller_after_takeoff_abort(
                "superseded_after_preflight",
            )
            self._set_preflight_status(False, "takeoff superseded by a safety command")
            self.log.event(
                "takeoff", ok=False, note="superseded_before_takeoff",
            )
            self._clear_maneuver("takeoff")
            return False
        if not preflight_ok:
            self._set_tracker_state(TrackerState.TAKEOFF_BLOCKED, reason="takeoff_cmd")
            self._clear_maneuver("takeoff")
            self.log.event(
                "takeoff", ok=False, note="preflight_blocked",
                reason=self.state.preflight_reason,
            )
            return False
        try:
            return self._takeoff_cmd_run(takeoff_epoch)
        finally:
            self._clear_maneuver("takeoff")

    # -------------------------------------------------------------- flight recording
    def set_record_on_takeoff(self, enabled: bool) -> None:
        """Arm/disarm: if armed, next successful takeoff starts onboard recording."""
        self.record_on_takeoff = bool(enabled)
        self._refresh_record_status()
        self.log.event("record_arm", enabled=self.record_on_takeoff,
                       status=self.record_status,
                       profile=self.recording_profile.profile_id)

    def _refresh_record_status(self, detail: str = "") -> None:
        self.record_status = format_record_status(
            active=self.recording_active,
            armed=self.record_on_takeoff,
            profile=self.recording_profile,
            detail=detail,
        )

    def set_recording_quality(self, profile_id: object) -> bool:
        """Select onboard SD resolution. Does not change the 720p live stream."""
        try:
            profile = resolve_recording_profile(profile_id)
        except ValueError as exc:
            self._refresh_record_status(detail="畫質無效")
            self.log.event("record_quality", ok=False, error=repr(exc))
            return False
        if self.recording_active:
            self._refresh_record_status()
            self.log.event(
                "record_quality", ok=False, reason="recording_active",
                profile=profile.profile_id,
            )
            return False
        self.recording_profile = profile
        if self.drone is None:
            self._refresh_record_status()
            self.log.event(
                "record_quality", ok=True, note="stored_no_drone",
                profile=profile.profile_id,
            )
            return True
        return self._apply_recording_quality()

    def _apply_recording_quality(self) -> bool:
        profile = self.recording_profile
        try:
            from olympe.messages.camera import (
                recording_mode,
                set_camera_mode,
                set_recording_mode,
            )
            try:
                self.drone(set_camera_mode(cam_id=0, value="recording")).wait(_timeout=5)
            except Exception as exc:
                self.log.event("record_set_mode", ok=False, error=repr(exc))
            exp = self.drone(
                set_recording_mode(
                    cam_id=0,
                    mode=profile.mode,
                    resolution=profile.resolution,
                    framerate=profile.framerate,
                    hyperlapse=profile.hyperlapse,
                )
            ).wait(_timeout=8)
            acked = bool(getattr(exp, "success", lambda: False)())
            try:
                state = self.drone.get_state(recording_mode)
            except Exception:
                state = None
            matched = recording_mode_matches(state, profile)
            if matched is False or (matched is None and not acked):
                self._refresh_record_status(detail="畫質未確認")
                self.log.event(
                    "record_quality", ok=False, reason="readback_mismatch",
                    profile=profile.profile_id, acked=acked, readback=state,
                )
                return False
            self._refresh_record_status()
            self.log.event(
                "record_quality", ok=True, profile=profile.profile_id,
                acked=acked, confirmed=matched is True, readback=state,
            )
            return True
        except Exception as exc:
            self._refresh_record_status(detail="畫質設定失敗")
            self.log.event("record_quality", ok=False, error=repr(exc),
                           profile=profile.profile_id)
            return False

    def start_flight_recording(self, reason: str = "start") -> bool:
        """Start onboard camera recording (ANAFI cam_id=0)."""
        if self.drone is None:
            self._refresh_record_status(detail="失敗（無連線）")
            self.log.event("record_start", ok=False, reason=reason, error="no_drone")
            return False
        if self.recording_active:
            self.log.event("record_start", ok=True, reason=reason, note="already_active")
            return True
        if not self._apply_recording_quality():
            self.log.event(
                "record_start", ok=False, reason=reason, error="quality_unconfirmed",
                profile=self.recording_profile.profile_id,
            )
            return False
        try:
            from olympe.messages.camera import (
                set_camera_mode,
                start_recording,
                recording_state,
            )
            # Ensure camera is in recording mode (not photo).
            try:
                self.drone(set_camera_mode(cam_id=0, value="recording")).wait(_timeout=5)
            except Exception as exc:
                self.log.event("record_set_mode", ok=False, error=repr(exc))
            exp = self.drone(
                start_recording(cam_id=0)
                >> recording_state(cam_id=0, state="active", _policy="check_wait", _timeout=8)
            ).wait(_timeout=10)
            ok = bool(getattr(exp, "success", lambda: False)())
            if not ok:
                # Some firmwares ack start without matching expectation cleanly.
                try:
                    st = self.drone.get_state(recording_state)
                    ok = bool(st and str(st.get("state", "")).lower() in
                              {"active", "recording", "started"})
                except Exception:
                    ok = False
            self.recording_active = ok
            if ok:
                self.record_started_iso = datetime.now(timezone.utc).isoformat()
                self._refresh_record_status()
            else:
                self._refresh_record_status(detail="啟動失敗")
            self.log.event("record_start", ok=ok, reason=reason,
                           status=self.record_status,
                           profile=self.recording_profile.profile_id)
            return ok
        except Exception as exc:
            self.recording_active = False
            self._refresh_record_status(detail=f"失敗 ({exc!r})")
            self.log.event("record_start", ok=False, reason=reason, error=repr(exc))
            return False

    def stop_flight_recording(
            self, reason: str = "stop", *, download: bool = True) -> bool:
        """Finalize onboard recording; optionally perform a bounded PC download."""
        if self.drone is None:
            self.recording_active = False
            self._refresh_record_status(detail="已停（無連線）")
            return False
        ok_stop = False
        try:
            from olympe.messages.camera import stop_recording, recording_state
            exp = self.drone(stop_recording(cam_id=0)).wait(_timeout=10)
            ok_stop = bool(getattr(exp, "success", lambda: True)())
            # Best-effort wait until inactive
            try:
                self.drone(
                    recording_state(cam_id=0, state="inactive",
                                    _policy="check_wait", _timeout=6)
                ).wait(_timeout=8)
            except (OSError, RuntimeError) as exc:  # Tier1: recording stop — narrow, no silent pass
                self.log.event("record_stop_wait_inactive_failed", error=repr(exc), reason=reason)
            self.log.event("record_stop", ok=ok_stop, reason=reason)
        except Exception as exc:
            self.log.event("record_stop", ok=False, reason=reason, error=repr(exc))
        self.recording_active = False

        if not download:
            self._refresh_record_status(detail="已停止（保留於機載媒體）")
            self.log.event(
                "record_saved", path="", reason=reason,
                note="onboard_only_no_synchronous_download",
            )
            return ok_stop

        # Download last media to PC if media API is available (best-effort).
        path = self._try_download_last_recording()
        if path:
            self.record_last_path = path
            self._refresh_record_status(detail=f"已存檔 {Path(path).name}")
            self.log.event("record_saved", path=path, reason=reason)
        else:
            self._refresh_record_status(detail="已停止（檔在機載媒體；下載略過/失敗）")
            self.log.event("record_saved", path="", reason=reason,
                           note="onboard_only_or_download_failed")
        return ok_stop

    def _prepare_recording_download(self, media: Any) -> None:
        try:
            media.download_dir = str(self.record_dir)
        except (OSError, RuntimeError) as exc:  # Tier1: recording media setup — narrow, log, fallback
            self.log.event("record_prepare_download_dir_failed", error=repr(exc), dir=str(self.record_dir))
        # Allow media indexing a moment after stop_recording.
        time.sleep(1.0)
        try:
            media.wait_for_pending_downloads(timeout=_MEDIA_PENDING_TIMEOUT_S)
        except (OSError, RuntimeError) as exc:  # Tier1: recording media setup — narrow, log, fallback
            self.log.event("record_prepare_pending_wait_failed", error=repr(exc))

    def _download_recording_media(self, media: Any, media_id: Any) -> None:
        try:
            from olympe.features.media import download_media
            waited = media(download_media(media_id)).wait(
                _timeout=_MEDIA_DOWNLOAD_TIMEOUT_S,
            )
            if not bool(getattr(waited, "success", lambda: False)()):
                raise TimeoutError("media download failed or timed out")
            media.wait_for_pending_downloads(
                timeout=_MEDIA_PENDING_TIMEOUT_S,
            )
        except Exception as exc:
            self.log.event(
                "record_download", ok=False, media_id=str(media_id),
                error=repr(exc),
            )

    def _recording_download_candidates(self) -> list[Path]:
        # Pick newest file in download dir
        return sorted(
            list(self.record_dir.glob("*.mp4"))
            + list(self.record_dir.glob("*.MP4"))
            + list(self.record_dir.glob("*.mov"))
            + list(self.record_dir.glob("*.MOV")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _try_download_last_recording(self) -> str:
        """Best-effort: download newest video resource into record_dir."""
        if self.drone is None:
            return ""
        try:
            media = getattr(self.drone, "media", None)
            if media is None:
                return ""
            self._prepare_recording_download(media)
            # Prefer explicit download of last media if API exposes it.
            last_id = getattr(media, "last_media_id", None)
            try:
                mid = last_id() if callable(last_id) else last_id
            except Exception:
                mid = None
            if mid:
                self._download_recording_media(media, mid)
            candidates = self._recording_download_candidates()
            if candidates:
                return str(candidates[0])
        except Exception as exc:
            self.log.event("record_download", ok=False, error=repr(exc))
        return ""

    def set_nudge_vector(self, roll: float, pitch: float, yaw: float,
                         gaz: float) -> bool:
        """Continuous stick input, on the SAME path as the discrete nudges.

        Deliberately reuses the hold loop, its deadman, the pilot_sticks block and
        the manoeuvre guard: a second command path would have bypassed every
        safety fix those carry. Axes are unit values in [-1, 1]; releasing the
        on-screen stick calls this with zeros (or clear_nudge_vector).
        """
        if self.pilot_sticks:
            self.log.event("nudge_blocked_manual", reason="vector")
            return False
        if (
            self._shutdown_latched
            or self._cleanup_done
            or self._landed
            or self.drone is None
        ):
            return False
        maneuver = self._maneuver_in_progress
        if maneuver is not None:
            self.log.event("nudge_blocked", name="vector", note=f"maneuver:{maneuver}")
            return False
        axes = []
        for value in (roll, pitch, yaw, gaz):
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(v):
                return False
            axes.append(max(-1.0, min(1.0, v)))
        if not any(abs(v) > 1e-3 for v in axes):
            self.clear_nudge_vector()
            return True
        with self._lock:
            if (
                self._shutdown_latched
                or self._cleanup_done
                or self._landed
                or self.drone is None
                or self.pilot_sticks
            ):
                return False
            maneuver = self._maneuver_in_progress
            if maneuver is not None:
                self.log.event(
                    "nudge_blocked",
                    name="vector",
                    note=f"maneuver:{maneuver}",
                )
                return False
            self._nudge_loop_stop.clear()
            self._nudge_vector = tuple(axes)
            self._nudge_deadline = time.monotonic() + self.nudge_pulse_s
            self._nudge_held.add(_VECTOR_HOLD)
        self._set_tracker_state(TrackerState.NUDGE, reason="set_nudge_vector")
        self._ensure_nudge_loop()
        return True

    def calibration_motion_snapshot(self) -> dict[str, Any]:
        """Return read-only command and telemetry evidence for field alignment.

        This deliberately exposes the last PCMD that reached ``_raw_pcmd`` rather
        than the UI's desired value.  A caller can therefore reject a calibration
        step when authority changed, the command went stale, or the aircraft did
        not move in the commanded body direction.
        """
        now_ns = time.monotonic_ns()
        with self._lock:
            pcmd = tuple(int(value) for value in self._last_pcmd)
            pcmd_stamp_ns = self._last_pcmd_mono_ns
            pilot_sticks = bool(self.pilot_sticks)
        monitor = self._stick_monitor
        try:
            stick_axes = monitor.snapshot_axes() if monitor is not None else {}
        except Exception:
            stick_axes = {}
        telemetry_stamp_ns = getattr(self.state, "telemetry_read_mono_ns", None)
        return {
            "pcmd": pcmd,
            "pcmd_age_s": (
                None
                if pcmd_stamp_ns is None
                else max(0.0, (now_ns - int(pcmd_stamp_ns)) / 1_000_000_000.0)
            ),
            "pilot_sticks": pilot_sticks,
            "stick_axes": stick_axes,
            "speed_north_mps": getattr(self.state, "speed_north_mps", None),
            "speed_east_mps": getattr(self.state, "speed_east_mps", None),
            "speed_down_mps": getattr(self.state, "speed_down_mps", None),
            "yaw_rad": getattr(self.state, "att_yaw", None),
            "gimbal_pitch_deg": getattr(self.state, "gimbal_pitch_deg", None),
            "heading_state": str(getattr(self.state, "heading_state", "UNKNOWN")),
            "telemetry_age_s": (
                None
                if telemetry_stamp_ns is None
                else max(
                    0.0,
                    (now_ns - int(telemetry_stamp_ns)) / 1_000_000_000.0,
                )
            ),
            "flight_state": str(getattr(self.state, "flight_state", "UNKNOWN")),
        }

    def clear_nudge_vector(self) -> None:
        with self._lock:
            self._nudge_vector = None
            self._nudge_held.discard(_VECTOR_HOLD)
            self._motion_epoch += 1
            empty = not self._nudge_held
            if empty:
                self._nudge_deadline = 0.0
        if empty:
            self._nudge_loop_stop.set()
            if not self.pilot_sticks and not self._landed and not self._cleanup_done:
                try:
                    self._raw_pcmd(0, 0, 0, 0)
                    self.log.event("pcmd", reason="vector_release_hover",
                                   pcmd=(0, 0, 0, 0))
                except Exception as exc:
                    self.log.event("pcmd_zero_failed", reason="vector_release",
                                   error=repr(exc))
                self._set_tracker_state(TrackerState.HOVER, reason="clear_nudge_vector")

    def _combined_nudge_pcmd(self) -> tuple[int, int, int, int]:
        """Sum the stick vector AND every held direction, clamped to ±nudge_pct.

        Both inputs contribute: returning the vector alone silently dropped a key
        held at the same time as a stick drag, while the log line still named it.
        Each contribution keeps the scaling it had on its own -- held directions
        through scale_nudge (which de-escalates multi-axis moves), the continuous
        stick by nudge_pct -- so the single-input cases are unchanged.
        """
        with self._lock:
            held = list(self._nudge_held)
            # _VECTOR_HOLD membership is what the deadman, the loop-error path and
            # the total-link-loss latch retire when they clear _nudge_held; they do
            # not null _nudge_vector. Keying on the hold rather than on the vector
            # being non-None stops an EXPIRED stick from re-entering the PCMD on the
            # operator's next button press.
            vector = self._nudge_vector if _VECTOR_HOLD in held else None
        pct = int(self.nudge_pct)
        acc = [0, 0, 0, 0]
        units = [0, 0, 0, 0]
        for name in held:
            u = NUDGE_DIRS.get(name)
            if u is None:
                continue
            for i in range(4):
                units[i] += int(u[i])
        unit = tuple(max(-1, min(1, v)) for v in units)
        if any(unit):
            scaled = scale_nudge(unit, pct)  # type: ignore[arg-type]
            for i in range(4):
                acc[i] += int(scaled[i])
        if vector is not None:
            # Continuous stick: scaled by the same authority envelope, so the
            # on-screen sticks can never exceed what a button press could.
            for i, value in enumerate(vector):
                acc[i] += int(round(float(value) * pct))
        return tuple(max(-pct, min(pct, v)) for v in acc)  # type: ignore[return-value]

    def _nudge_loop_step(self) -> tuple[bool, bool]:
        if (
            self._shutdown_latched
            or self._cleanup_done
            or self._landed
            or self.pilot_sticks
        ):
            return False, False
        deadman_expired = False
        with self._lock:
            held = list(self._nudge_held)
            motion_epoch = self._motion_epoch
            if held and time.monotonic() >= self._nudge_deadline:
                self._nudge_held.clear()
                self._motion_epoch += 1
                held = []
                deadman_expired = True
        if not held:
            return False, deadman_expired
        pcmd = self._combined_nudge_pcmd()
        sent = self.send_pcmd(
            *pcmd,
            reason="nudge_hold:" + "+".join(sorted(held)),
            motion_epoch=motion_epoch,
        )
        if sent:
            self._set_tracker_state(TrackerState.NUDGE, reason="_nudge_loop_step")
        self._nudge_loop_stop.wait(self._nudge_period_s)
        return True, deadman_expired

    def _finish_nudge_loop(
        self, *, deadman_expired: bool, loop_error: BaseException | None,
    ) -> None:
        # Loop exit: zero if nothing held (release → hover)
        with self._lock:
            empty = not self._nudge_held
            owns_thread = self._nudge_loop_thread is threading.current_thread()
            # Atomically retire this loop before deciding whether a new
            # hold arrived during its final iteration. Without this handoff,
            # _ensure_nudge_loop() sees the dying thread as alive and leaves a
            # held nudge without a deadman loop.
            if owns_thread:
                self._nudge_loop_thread = None
            held_for_handoff = bool(self._nudge_held)
        if empty and not self.pilot_sticks and not self._landed and not self._cleanup_done:
            zero_reason = (
                "nudge_loop_error_hover" if loop_error is not None
                else "nudge_deadman_hover" if deadman_expired
                else "nudge_release_hover"
            )
            try:
                self._raw_pcmd(0, 0, 0, 0)
                self.log.event("pcmd", reason=zero_reason, pcmd=(0, 0, 0, 0))
            except Exception as zero_exc:
                # The link itself is gone; the aircraft's own link-loss
                # failsafe is the remaining authority. Record it, never hide it.
                self.log.event("pcmd_zero_failed", reason=zero_reason,
                               error=repr(zero_exc))
            self._set_tracker_state(TrackerState.HOVER, reason="_finish_nudge_loop")
            if loop_error is not None:
                self.state.last_command = "nudge_loop_error_hover"
                self.log.event(
                    "nudge_loop_error", ok=False, error=repr(loop_error),
                )
            elif deadman_expired:
                self.state.last_command = "nudge_deadman_hover"
                self.log.event(
                    "nudge_deadman", ok=True, ttl_s=self.nudge_pulse_s,
                )
        if (
            owns_thread
            and held_for_handoff
            and not self._shutdown_latched
            and not self._cleanup_done
        ):
            self._nudge_loop_stop.clear()
            self._ensure_nudge_loop()

    def _ensure_nudge_loop(self) -> None:
        """Start background loop that streams PCMD while any key/button is held."""
        def _loop() -> None:
            deadman_expired = False
            loop_error: BaseException | None = None
            try:
                while not self._nudge_loop_stop.is_set():
                    keep_running, expired = self._nudge_loop_step()
                    deadman_expired = deadman_expired or expired
                    if not keep_running:
                        break
            except Exception as exc:
                # The deadman timer lives INSIDE this loop. If the loop dies the
                # hold would never expire and the aircraft would keep flying on
                # its last non-zero PCMD until the operator noticed. Drop the
                # hold so the zero below runs, and never die silently.
                loop_error = exc
                with self._lock:
                    self._nudge_held.clear()
                    self._motion_epoch += 1
            finally:
                self._finish_nudge_loop(
                    deadman_expired=deadman_expired,
                    loop_error=loop_error,
                )

        with self._lock:
            if (
                self._shutdown_latched
                or self._cleanup_done
                or self._landed
                or self.drone is None
                or not self._nudge_held
            ):
                return
            if (
                self._nudge_loop_thread is not None
                and self._nudge_loop_thread.is_alive()
            ):
                return
            self._nudge_loop_stop.clear()
            self._nudge_loop_thread = threading.Thread(
                target=_loop, name="nudge-hold-loop", daemon=True
            )
            self._nudge_loop_thread.start()

    def nudge_begin(self, name: str) -> bool:
        """Key/button pressed: hold this direction until nudge_end."""
        if name not in NUDGE_DIRS:
            self.log.event("unknown_nudge", name=name)
            return False
        if self.pilot_sticks:
            self.log.event("nudge_blocked_manual", name=name)
            return False
        if (
            self._shutdown_latched
            or self._cleanup_done
            or self._landed
            or self.drone is None
        ):
            self.log.event("nudge_blocked", name=name, note="not_available")
            return False
        maneuver = self._maneuver_in_progress
        if maneuver is not None:
            self.log.event("nudge_blocked", name=name, note=f"maneuver:{maneuver}")
            return False
        with self._lock:
            if (
                self._shutdown_latched
                or self._cleanup_done
                or self._landed
                or self.drone is None
                or self.pilot_sticks
            ):
                return False
            maneuver = self._maneuver_in_progress
            if maneuver is not None:
                self.log.event(
                    "nudge_blocked", name=name, note=f"maneuver:{maneuver}"
                )
                return False
            self._nudge_loop_stop.clear()
            already = name in self._nudge_held
            self._nudge_held.add(name)
            if not already:
                self._motion_epoch += 1
                self._nudge_deadline = time.monotonic() + self.nudge_pulse_s
        if already:
            return True
        self.log.event("nudge_begin", name=name, held=sorted(self._nudge_held),
                       pcmd=self._combined_nudge_pcmd())
        self.state.last_command = f"nudge_begin:{name}"
        self._set_tracker_state(TrackerState.NUDGE, reason="nudge_begin")
        self._ensure_nudge_loop()
        return True

    def nudge_heartbeat(self, directions: Any) -> bool:
        """Refresh the nudge deadman only for the exact currently held set."""
        if isinstance(directions, str):
            requested = {directions}
        elif isinstance(directions, (list, tuple, set, frozenset)):
            requested = {str(name) for name in directions}
        else:
            requested = set()
        with self._lock:
            # The on-screen stick holds _VECTOR_HOLD in the same set. It refreshes
            # the deadman through its own path, so comparing the UI's list of
            # DIRECTIONS against a set containing the sentinel rejected every
            # heartbeat whenever a key and the stick were held together.
            held = set(self._nudge_held) - {_VECTOR_HOLD}
            matches = bool(held) and requested == held
            if matches and not (
                    self._shutdown_latched
                    or self._cleanup_done
                    or self._landed
                    or self.pilot_sticks):
                self._nudge_deadline = time.monotonic() + self.nudge_pulse_s
                return True
        self.log.event(
            "nudge_heartbeat_rejected",
            requested=sorted(requested), held=sorted(held),
        )
        return False

    def nudge_end(self, name: str) -> bool:
        """Key/button released: drop this direction; hover if none left."""
        with self._lock:
            had_name = name in self._nudge_held
            self._nudge_held.discard(name)
            if had_name:
                self._motion_epoch += 1
            empty = not self._nudge_held
            if empty:
                self._nudge_deadline = 0.0
            else:
                self._nudge_deadline = time.monotonic() + self.nudge_pulse_s
        self.log.event("nudge_end", name=name, held=sorted(self._nudge_held), empty=empty)
        self.state.last_command = f"nudge_end:{name}"
        if empty:
            # Stop loop; it will zero PCMD on exit. Also zero immediately.
            self._nudge_loop_stop.set()
            if not self.pilot_sticks and not self._landed and not self._cleanup_done:
                self._zero_pcmd_or_log("nudge_release_hover")
                self._set_tracker_state(TrackerState.HOVER, reason="nudge_end")
        else:
            # Recompute loop continues with remaining holds.
            self._ensure_nudge_loop()
        return True

    def nudge_clear(self, reason: str = "clear") -> None:
        """Release all held directions AND the stick vector; stop the hold loop.

        The vector is cleared under the same lock the hold loop reads it under:
        an unlocked clear could land between that read and its send, letting one
        more scaled PCMD go out after the release. Advancing the motion epoch
        closes the remaining snapshot-to-send window.
        """
        with self._lock:
            self._nudge_vector = None
            self._nudge_held.clear()
            self._nudge_deadline = 0.0
            self._motion_epoch += 1
        self._nudge_loop_stop.set()
        self.log.event("nudge_clear", reason=reason)

    def nudge(self, name: str) -> None:
        """Backward-compat: one-shot press treated as begin (UI should use begin/end)."""
        self.nudge_begin(name)

    def set_gimbal_pitch(self, pitch_deg: float) -> bool:
        if self.drone is None or self.pilot_sticks:
            self.log.event(
                "gimbal_pitch", ok=False, note="no_drone_or_sticks",
            )
            return False
        pitch = float(np.clip(pitch_deg, self.ANAFI.gimbal_pitch_min_deg,
                              self.ANAFI.gimbal_pitch_max_deg))
        try:
            from olympe.messages.gimbal import set_target
            self.drone(set_target(
                gimbal_id=0,
                control_mode="position",
                yaw_frame_of_reference="none",
                yaw=0.0,
                pitch_frame_of_reference="absolute",
                pitch=pitch,
                roll_frame_of_reference="none",
                roll=0.0,
            ))
            self.state.gimbal_pitch_deg = pitch
            self.log.event("gimbal_pitch", pitch=pitch, ok=True)
            return True
        except Exception as exc:
            # Fallback older API
            try:
                from olympe.messages.ardrone3.Camera import Orientation
                self.drone(Orientation(tilt=int(pitch), pan=0))
                self.state.gimbal_pitch_deg = pitch
                self.log.event("gimbal_pitch_legacy", pitch=pitch, ok=True)
                return True
            except Exception as exc2:
                self.log.event("gimbal_pitch", ok=False, error=repr(exc2 or exc))
                return False

    def set_zoom(self, zoom: float) -> bool:
        """Set digital zoom level (1.0 = default / wide)."""
        z = float(np.clip(float(zoom), 1.0, self.ANAFI.digital_zoom_max))
        if self.drone is None or self.pilot_sticks:
            self.log.event("zoom", zoom=z, ok=False, note="no_drone_or_sticks",
                           applied=getattr(self.state, "zoom", None))
            return False
        try:
            from olympe.messages.camera import set_zoom_target
            from olympe.enums.camera import zoom_control_mode
            self.drone(set_zoom_target(
                cam_id=0,
                control_mode=zoom_control_mode.level,
                target=z,
            ))
        except Exception as exc:
            self.log.event("zoom", zoom=z, ok=False, error=repr(exc),
                           applied=getattr(self.state, "zoom", None))
            return False
        # Commit only once the command was accepted. state.zoom also gates
        # localization (uncalibrated zoom pauses it), so recording a zoom the camera
        # never applied would silently stop localization at the real 1.0x.
        self.state.zoom = z
        self.log.event("zoom", zoom=z, ok=True)
        return True

    def reset_camera_defaults(self, *, pitch: float = -20.0, zoom: float = 1.0) -> bool:
        """Restore default look-down pitch and 1.0x zoom."""
        pitch0 = float(pitch)
        zoom0 = float(zoom)
        pitch_ok = self.set_gimbal_pitch(pitch0)
        zoom_ok = False
        # Prefer dedicated reset_zoom when target is wide; else set level.
        if self.drone is not None and not self.pilot_sticks and abs(zoom0 - 1.0) < 1e-6:
            try:
                from olympe.messages.camera import reset_zoom
                self.drone(reset_zoom(cam_id=0))
                self.state.zoom = 1.0
                self.log.event("zoom_reset", zoom=1.0, ok=True)
                zoom_ok = True
            except Exception as exc:
                self.log.event("zoom_reset", ok=False, error=repr(exc))
                zoom_ok = self.set_zoom(zoom0)
        else:
            zoom_ok = self.set_zoom(zoom0)
        self.state.last_command = "camera_reset"
        ok = bool(pitch_ok and zoom_ok)
        self.log.event("camera_reset", pitch=pitch0, zoom=self.state.zoom, ok=ok)
        return ok

    # -------------------------------------------------------------- UI interface
    def start(self, config: SessionConfig) -> StartResult:
        if config.interface_mode is not self.mode:
            return StartResult(False, "INTERFACE_MISMATCH")
        if self.session_config is not None and self.session_config != config:
            return StartResult(False, "HOT_SWITCH_PROHIBITED")
        self.session_config = config
        return StartResult(True, "OK")

    def _typed_command_gate(self, request: ControlRequest) -> ControlResult | None:
        if request.action is ControlAction.TAKEOFF and not request.human_origin:
            return ControlResult.rejected("HUMAN_ORIGIN_REQUIRED", self.state)
        calibration_starts = {
            ControlAction.DRONE_MAGNETOMETER_START,
            ControlAction.SKYCONTROLLER_MAGNETOMETER_START,
        }
        if request.action in calibration_starts and not request.human_origin:
            return ControlResult.rejected("HUMAN_ORIGIN_REQUIRED", self.state)
        if request.action not in {
            ControlAction.LAND_NOW,
            ControlAction.EMERGENCY_STOP,
        }:
            age_ns = time.monotonic_ns() - int(request.submitted_mono_ns)
            if age_ns > CONTROL_REQUEST_MAX_AGE_NS:
                self.log.event(
                    "control_request_stale",
                    request_id=request.request_id,
                    action=request.action.value,
                    age_ms=age_ns / 1_000_000.0,
                )
                return ControlResult.rejected("STALE_CONTROL_REQUEST", self.state)
        return None

    def _run_typed_action(self, request: ControlRequest) -> ControlResult | None:
        if request.action is ControlAction.START_AUTO:
            calibration_error = self._magnetometer_control_error()
            if calibration_error is not None:
                self.log.event(
                    "auto_rejected",
                    request_id=request.request_id,
                    reason=calibration_error,
                )
                return ControlResult.rejected(
                    "MAGNETOMETER_CALIBRATION_REQUIRED", self.state
                )
            self.log.event(
                "auto_rejected",
                request_id=request.request_id,
                reason="external_approval_required",
            )
            return ControlResult.rejected("LOCKED_EXTERNAL_APPROVAL", self.state)
        if request.action is ControlAction.EMERGENCY_STOP:
            return self.fail_safe(FailureReason.EMERGENCY_STOP)
        if request.action in {ControlAction.LAND_NOW, ControlAction.TAKEOFF}:
            try:
                raw = (
                    self.land_cmd("ui_land_now")
                    if request.action is ControlAction.LAND_NOW
                    else self.takeoff_cmd()
                )
            except Exception as exc:
                self.log.event(
                    "typed_command_exception",
                    request_id=request.request_id,
                    action=request.action.value,
                    error=repr(exc),
                    active_incident=getattr(self.state, "active_incident", ""),
                    tracker_state=getattr(self.state, "tracker_state", ""),
                    mode=getattr(self.state, "mode", ""),
                    control_owner=getattr(self.state, "control_owner", ""),
                )
                return ControlResult.rejected("BACKEND_EXCEPTION", self.state)
            return ControlResult.completed(self.state, raw_result=raw)
        if request.action is ControlAction.START_LOCALIZATION:
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
            return raw
        return ControlResult.completed(self.state, raw_result=raw)

    def _command_typed_request(self, request: ControlRequest) -> ControlResult:
        self.log.event(
            "control_request",
            request_id=request.request_id,
            action=request.action.value,
            human_origin=request.human_origin,
            submitted_mono_ns=request.submitted_mono_ns,
        )
        result = self._typed_command(request)
        self.log.event(
            "control_result",
            request_id=request.request_id,
            action=request.action.value,
            accepted=result.accepted,
            executed=result.executed,
            reason_code=result.reason_code,
            control_owner=getattr(self.state, "control_owner", None),
        )
        return result

    def _legacy_authority_command(self, name: str, payload: dict) -> Any:
        if name == "manual":
            return self.give_to_pilot()
        if name in {"pc_control", "resume_pc", "恢復電腦控制"}:
            accepted = self.take_pc_control()
            if accepted:
                self.state.last_command = "pc_control"
            return accepted
        if name == "hover":
            return self.hover_cmd("ui_hover")
        if name == "land":
            return self.land_cmd("ui_land")
        if name == "emergency_stop":
            self.fail_safe(FailureReason.EMERGENCY_STOP)
            return True
        return self.land_cmd("ui_land_now")

    def _legacy_safety_command(self, name: str, payload: dict) -> Any:
        if name == "firmware_limits_apply":
            return self.apply_firmware_limits(
                payload.get("max_altitude_m"),
                payload.get("max_distance_m"),
                payload.get("distance_geofence", True),
            )
        if name == "auto_speed_limit_apply":
            return self.apply_autonomous_speed_limit(
                payload.get("speed_limit_mps"),
                enabled=payload.get("enabled"),
            )
        if name == "drone_magnetometer_start":
            return self.start_drone_magnetometer_calibration()
        if name == "drone_magnetometer_cancel":
            return self.cancel_drone_magnetometer_calibration()
        if name == "skycontroller_magnetometer_start":
            return self.start_skycontroller_magnetometer_calibration()
        return self.cancel_skycontroller_magnetometer_calibration()

    def _legacy_auto_command(self, name: str, payload: dict) -> Any:
        if name in {"auto", "start_auto"}:
            # Compatibility command only: this backend has no route controller.
            calibration_error = self._magnetometer_control_error()
            if calibration_error is not None:
                self.log.event(
                    "localization_pc_control",
                    ok=False,
                    note="magnetometer_not_ready",
                    reason=calibration_error,
                )
                return False
            accepted = self.take_pc_control()
            if accepted:
                self.state.mode = "PC_CONTROL"
                self._set_tracker_state(TrackerState.LOCALIZATION_ONLY, reason="_legacy_auto_command")
                self.log.event(
                    "localization_pc_control",
                    ok=True,
                    note="PC control taken; no autonomous route controller is running",
                )
            else:
                self.log.event(
                    "localization_pc_control",
                    ok=False,
                    note="source_unconfirmed",
                )
            return accepted
        if name == "boot_lock":
            self._set_tracker_state(TrackerState.BOOT_INIT, reason="_legacy_auto_command")
            return True
        accepted = self.hover_cmd(f"ui_{name}")
        self.log.event(name)
        return accepted


    def _legacy_camera_command(self, name: str, payload: dict) -> Any:
        if name == "gimbal_pitch":
            return self.set_gimbal_pitch(
                float(payload.get("pitch", self.state.gimbal_pitch_deg))
            )
        if name == "zoom":
            return self.set_zoom(float(payload.get("zoom", self.state.zoom)))
        return self.reset_camera_defaults(
            pitch=float(payload.get("pitch", -20.0)),
            zoom=float(payload.get("zoom", 1.0)),
        )

    def _legacy_record_command(self, name: str, payload: dict) -> Any:
        if name in {"record_arm", "record_on_takeoff"}:
            en = payload.get("enabled", payload.get("value", True))
            if isinstance(en, str):
                en = en.lower() in {"1", "true", "yes", "on"}
            self.set_record_on_takeoff(bool(en))
            return True
        if name == "record_disarm":
            self.set_record_on_takeoff(False)
            return True
        if name == "record_quality":
            return self.set_recording_quality(payload.get("profile_id"))
        if name == "record_start":
            return self.start_flight_recording(reason="ui_manual")
        return self.stop_flight_recording(reason="ui_manual")

    def _legacy_nudge_command(self, name: str, payload: dict) -> Any:
        if name in {"nudge_begin", "nudge_press"}:
            d = str(payload.get("dir") or payload.get("name") or "")
            return self.nudge_begin(d)
        if name in {"nudge_end", "nudge_release"}:
            d = str(payload.get("dir") or payload.get("name") or "")
            return self.nudge_end(d)
        if name == "nudge_heartbeat":
            dirs = payload.get("dirs", payload.get("directions"))
            return self.nudge_heartbeat(dirs)
        if name == "nudge_vector":
            return self.set_nudge_vector(
                float(payload.get("roll", 0.0)), float(payload.get("pitch", 0.0)),
                float(payload.get("yaw", 0.0)), float(payload.get("gaz", 0.0)),
            )
        if name == "nudge_clear":
            self.nudge_clear(reason="ui")
            if not self.send_pcmd(0, 0, 0, 0, reason="nudge_clear"):
                return False
            if self._maneuver_in_progress is None:
                self._set_tracker_state(TrackerState.HOVER, reason="_legacy_nudge_command")
                self.state.mode = "MANUAL"
            return True
        # Legacy one-shot name: treat as begin (UI should send begin/end).
        n = name[6:] if name.startswith("nudge_") else name
        return self.nudge_begin(n)

    def _command_legacy(self, name: str, payload: dict) -> Any:
        if name == "takeoff":
            # The compatibility shim intentionally excludes TakeOff. A typed
            # request is the only path that carries the human-origin bit.
            self.log.event("legacy_takeoff_rejected", reason="typed_request_required")
            return ControlResult.rejected("TYPED_TAKEOFF_REQUIRED", self.state)
        self.state.last_command = name
        handler_groups = (
            (
                {"manual", "pc_control", "resume_pc", "恢復電腦控制",
                 "hover", "land", "emergency_stop", "land_now"},
                self._legacy_authority_command,
            ),
            (
                {"firmware_limits_apply", "auto_speed_limit_apply",
                 "drone_magnetometer_start", "drone_magnetometer_cancel",
                 "skycontroller_magnetometer_start",
                 "skycontroller_magnetometer_cancel"},
                self._legacy_safety_command,
            ),
            ({"auto", "start_auto", "boot_lock", "pause", "resume"},
             self._legacy_auto_command),
            ({"gimbal_pitch", "zoom", "camera_reset", "reset_camera",
              "鏡頭預設", "回復預設"}, self._legacy_camera_command),
            ({"record_arm", "record_on_takeoff", "record_disarm",
              "record_start", "record_stop", "record_quality"},
             self._legacy_record_command),
            ({"nudge_begin", "nudge_press", "nudge_end", "nudge_release",
              "nudge_heartbeat", "nudge_vector", "nudge_clear"},
             self._legacy_nudge_command),
        )
        for names, handler in handler_groups:
            if name in names:
                return handler(name, payload)
        if name.startswith("nudge_") or name in NUDGE_DIRS:
            return self._legacy_nudge_command(name, payload)
        self.log.event("unhandled_command", name=name, payload=payload)
        return False

    def command(self, name: str | ControlRequest, **payload) -> Any:
        if isinstance(name, ControlRequest):
            return self._command_typed_request(name)
        return self._command_legacy(name, payload)

    def apply_autonomous_speed_limit(
        self, value: Any, *, enabled: Any = None
    ) -> bool:
        parsed = self._finite_float(value)
        if parsed is None:
            self.log.event("auto_speed_limit", ok=False, reason="INVALID_SPEED_LIMIT")
            return False
        current_enabled = bool(getattr(
            self.state, "autonomous_speed_limit_enabled", True
        ))
        requested_enabled = current_enabled if enabled is None else enabled
        if not isinstance(requested_enabled, bool):
            self.log.event("auto_speed_limit", ok=False, reason="INVALID_ENABLED_STATE")
            return False
        change = validate_speed_limit_change(
            float(self.state.autonomous_speed_limit_mps),
            parsed,
            landed=self._flight_state_name() == "landed",
        )
        if not change.accepted:
            self.log.event("auto_speed_limit", ok=False, reason=change.reason)
            return False
        enabled_changed = requested_enabled != current_enabled
        self.state.autonomous_speed_limit_enabled = requested_enabled
        self.state.autonomous_speed_limit_mps = change.new_speed_limit_mps
        self.state.autonomous_speed_guard_status = (
            "SPEED_WAITING" if requested_enabled else "SPEED_LIMIT_DISABLED"
        )
        approval_invalidated = change.approval_invalidated or enabled_changed
        if approval_invalidated:
            self.state.autonomous_approval_valid = False
            self.state.autonomous_locked = True
        self.log.event(
            "auto_speed_limit",
            ok=True,
            old_mps=change.old_speed_limit_mps,
            new_mps=change.new_speed_limit_mps,
            old_enabled=current_enabled,
            new_enabled=requested_enabled,
            approval_invalidated=approval_invalidated,
            note="requires new PCMD response/braking/low-altitude approval",
        )
        return True

    def fail_safe(self, reason: FailureReason) -> ControlResult:
        """Atomically cancel PC motion, send zero, and require manual recovery."""
        observed_ns = time.monotonic_ns()
        ok = self.give_to_pilot(reason=f"fail_safe:{reason.value}")
        self.state.mode = "MANUAL"
        self.state.active_incident = reason.value
        self.state.last_command = f"fail_safe:{reason.value}"
        self.state.tracker_state = (
            TrackerState.EMERGENCY_MANUAL
            if reason is FailureReason.EMERGENCY_STOP
            else TrackerState.FAIL_SAFE_MANUAL
        )
        self.log.event(
            "fail_safe",
            reason=reason.value,
            observed_mono_ns=observed_ns,
            handoff_ok=ok,
            control_owner=self.state.control_owner,
            auto_resume=False,
        )
        return ControlResult.completed(
            self.state,
            raw_result=ok,
            reason_code="HOVER_MANUAL_HANDOFF",
        )

    def stream_lost_hover(self, detail: str = "stream lost") -> Any:
        self.nudge_clear(reason="stream_stale_hover")
        hovered = self.send_pcmd(
            0, 0, 0, 0, reason="stream_stale_hover"
        )
        pipeline = None
        if self.grabber is not None:
            try:
                pipeline = self.grabber.frame_pipeline_stats
            except Exception:
                pipeline = None
        self.log.event(
            "stream_lost",
            detail=detail,
            hover_sent=hovered,
            pipeline=pipeline,
        )
        self.state.active_incident = FailureReason.STREAM_STALE.value
        self.state.stream = "LOST"
        self.state.loc = "STREAM_LOST"
        self._set_tracker_state(TrackerState.STREAM_LOST_HOVER, reason="stream_lost_hover")
        self.state.last_command = f"stream_lost_hover: {detail}"
        return self.state

    def _video_health_snapshot(self) -> tuple[float | None, int, bool]:
        age_s: float | None = None
        last_frame_age = getattr(self.grabber, "last_frame_age", None)
        if callable(last_frame_age):
            try:
                age_s = self._finite_float(last_frame_age())
            except Exception:
                age_s = None
        pipeline: dict[str, Any] = {}
        try:
            raw_pipeline = getattr(self.grabber, "frame_pipeline_stats", {})
            if isinstance(raw_pipeline, dict):
                pipeline = raw_pipeline
        except Exception:
            pass
        try:
            duplicate_run = int(pipeline.get("duplicate_run", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            duplicate_run = 0
        frozen = bool(
            pipeline.get("frozen", False)
            or duplicate_run >= _STREAM_DUPLICATE_FRAMES
        )
        return age_s, duplicate_run, frozen

    def _mark_video_health_recovered(self, now_mono: float, age_s: float) -> bool:
        if self._stream_stale_since is not None:
            self.log.event(
                "stream_recovered",
                stale_for_s=round(now_mono - self._stream_stale_since, 2),
                age_s=round(age_s, 3),
            )
            self._stream_stale_since = None
        if self._stream_failure_latched:
            # Stream is healthy again. Re-arm, otherwise the FIRST outage of a
            # session is the only one that ever triggers a handoff and every
            # later freeze is silently ignored for the rest of the flight.
            self._stream_failure_latched = False
            self.log.event("stream_health_rearmed", age_s=round(age_s, 3))
        if self.state.active_incident == FailureReason.STREAM_STALE.value:
            self.state.active_incident = ""
            self.state.stream = "OK"
        return False

    def _handle_stale_video(
        self,
        now_mono: float,
        age_s: float,
        duplicate_run: int,
        frozen: bool,
    ) -> bool:
        if self._stream_failure_latched:
            return False            # this outage has already been handled
        # Stale right now -- but only hand control back once it STAYS that way.
        # A LATE frame is a link hiccup worth riding out. A FROZEN one -- the same
        # picture repeated -- is flying blind, and the operator must get the sticks
        # back long before the 10 s latency grace would expire.
        grace_s = (
            min(FROZEN_STREAM_GRACE_S, self.stream_loss_grace_s)
            if frozen else self.stream_loss_grace_s
        )
        if self._stream_stale_since is None:
            self._stream_stale_since = now_mono
            self.log.event(
                "stream_stale_grace_started",
                grace_s=grace_s,
                age_s=round(age_s, 3), frozen=bool(frozen),
            )
        stale_for = now_mono - self._stream_stale_since
        if stale_for < grace_s:
            return False
        self._stream_failure_latched = True
        detail = (
            f"frame frozen duplicate_run={duplicate_run} age_s={age_s:.2f}"
            if frozen
            else f"frame stale age_s={age_s:.2f}"
        ) + f" for {stale_for:.1f}s (grace {grace_s:.1f}s)"
        self.stream_lost_hover(detail)
        return True

    def _evaluate_video_health(self, now: float | None = None) -> bool:
        if self.video_stream is None:
            return False
        age_s, duplicate_run, frozen = self._video_health_snapshot()
        if age_s is None:
            return False
        age_s = max(0.0, age_s)
        now_mono = time.monotonic()
        if not frozen and age_s < _STREAM_STALE_S:
            return self._mark_video_health_recovered(now_mono, age_s)
        return self._handle_stale_video(now_mono, age_s, duplicate_run, frozen)

    @staticmethod
    def _haversine_distance_m(
        latitude_a: float,
        longitude_a: float,
        latitude_b: float,
        longitude_b: float,
    ) -> float:
        lat_a = math.radians(float(latitude_a))
        lat_b = math.radians(float(latitude_b))
        delta_lat = lat_b - lat_a
        delta_lon = math.radians(float(longitude_b) - float(longitude_a))
        value = (
            math.sin(delta_lat / 2.0) ** 2
            + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
        )
        return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))

    def _refresh_home_state(self) -> bool:
        try:
            from olympe.messages import rth

            home = self._state_dict(rth.home_location)
            reachability = self._state_dict(rth.home_reachability)
        except Exception:
            return False
        latitude = self._finite_float(self._state_value(home, "latitude"))
        longitude = self._finite_float(self._state_value(home, "longitude"))
        altitude = self._finite_float(self._state_value(home, "altitude"))
        valid = bool(
            latitude is not None
            and longitude is not None
            and altitude is not None
            and self._valid_home(latitude, longitude, altitude)
        )
        status = str(
            self._state_value(reachability, "status") or ""
        ).rsplit(".", 1)[-1].lower()
        self.state.home_valid = valid
        self.state.home_reachable = bool(valid and status == "reachable")
        if valid:
            self._home_latitude_deg = latitude
            self._home_longitude_deg = longitude
        return self.state.home_reachable

    def _update_distance_from_home(
        self, latitude: float | None, longitude: float | None
    ) -> None:
        if latitude is None or longitude is None:
            self.state.distance_from_home_m = None
            return
        if self._home_latitude_deg is None or self._home_longitude_deg is None:
            self._refresh_home_state()
        if self._home_latitude_deg is None or self._home_longitude_deg is None:
            self.state.distance_from_home_m = None
            return
        self.state.distance_from_home_m = self._haversine_distance_m(
            self._home_latitude_deg,
            self._home_longitude_deg,
            latitude,
            longitude,
        )

    def _request_rth(self, reason: str) -> bool:
        if self.drone is None:
            return False
        try:
            from olympe.messages import rth
        except Exception as exc:
            self.log.event("rth", ok=False, reason=reason, error=repr(exc))
            return False
        self.nudge_clear(reason=f"rth:{reason}")
        with self._lock:
            self._pulse_token += 1
            self._zero_pcmd_or_log("rth:pre_handoff_zero")
        source_ok = self._set_piloting_source("Controller")
        if source_ok:
            with self._lock:
                self.pilot_sticks = False
                self._zero_pcmd_or_log("rth:post_handoff_zero")
        try:
            waited = self.drone(rth.return_to_home()).wait(_timeout=5)
            ok = bool(getattr(waited, "success", lambda: False)())
        except Exception as exc:
            self.log.event("rth", ok=False, reason=reason, error=repr(exc))
            return False
        self.log.event(
            "rth",
            ok=ok,
            reason=reason,
            source_confirmed=source_ok,
            ending_behavior="landing",
        )
        if not ok:
            return False
        self.state.mode = "MANUAL"
        self._set_tracker_state(TrackerState.RTH, reason="_request_rth")
        self.state.last_command = f"rth:{reason}"
        if self.via_skycontroller():
            handed_off = self._set_piloting_source("SkyController")
            with self._lock:
                self.pilot_sticks = handed_off
        return True

    def _rth_altitude_conflict(self) -> str | None:
        """Return why RTH is unsafe under the configured ceiling, else None.

        RTH climbs to the firmware's return_home_min_altitude before heading home.
        That value is set independently of MaxAltitude, so it can be far above the
        ceiling the operator chose for this flight.
        """
        ceiling = self._finite_float(self.desired_max_altitude_m)
        if ceiling is None:
            return None
        climb = self._finite_float(
            getattr(self.state, "rth_min_altitude_m", None)
        )
        if climb is None:
            # state.rth_min_altitude_m is only filled by the hardware readback and
            # by the lost-link configuration; connecting to an ALREADY AIRBORNE
            # aircraft runs neither. Unknown must not mean permitted -- the firmware
            # keeps whatever the last FreeFlight session left (39.2 m measured on the
            # reference unit), which is the exact climb this check exists to stop.
            return (
                "RTH climb altitude was never read back from the aircraft, so it "
                f"cannot be shown to respect the {ceiling:.1f} m altitude limit"
            )
        if climb <= ceiling:
            return None
        return (
            f"RTH would climb to {climb:.1f} m, above the {ceiling:.1f} m "
            "altitude limit for this flight"
        )

    def _execute_runtime_safety_action_impl(self, reason: str) -> None:
        if self._cleanup_done or self.drone is None:
            return
        if self._is_already_landed():
            self.log.event("runtime_safety", reason=reason, action="already_landed")
            return
        # Use RTH only when Home/GPS and the configured altitude ceiling allow it.
        # If RTH is unavailable or fails, hold position instead of landing.
        rth_blocked = self._rth_altitude_conflict()
        if reason == FailureReason.INVALID_TELEMETRY.value:
            rth_blocked = "GPS/geofence telemetry unavailable; hover"
        if rth_blocked is not None:
            self.log.event(
                "runtime_safety_rth_skipped", reason=reason, detail=rth_blocked,
            )
        if rth_blocked is None and self._refresh_home_state() and self._request_rth(reason):
            self.log.event("runtime_safety", reason=reason, action="rth")
            return
        hovered = self.hover_cmd(reason=f"runtime_safety:{reason}:rth_unavailable")
        self.log.event(
            "runtime_safety",
            reason=reason,
            action="hover",
            ok=hovered,
        )
        if not hovered:
            # Neither RTH nor hover succeeded and the aircraft is still airborne.
            # The latch exists to stop one successful action being re-issued; leaving
            # it set after a failed action would disable altitude/distance
            # protection for the rest of the session. Re-arm so the next poll retries.
            with self._runtime_safety_action_lock:
                self._runtime_safety_action_latched = False
            self.log.event("runtime_safety_rearmed", reason=reason)

    def _execute_runtime_safety_action(self, reason: str) -> None:
        """Run the recovery with a last-resort guard around every action step."""
        try:
            self._execute_runtime_safety_action_impl(reason)
        except Exception as exc:
            # A bug in the watchdog thread must be observable and recoverable.  If
            # the latch stayed set here, the next poll would silently stop checking
            # altitude/distance protection for the remainder of the flight.
            try:
                self.log.event(
                    "runtime_safety_exception",
                    reason=reason,
                    error=repr(exc),
                    action="retry_on_next_poll",
                )
            except Exception:
                pass
            with self._runtime_safety_action_lock:
                self._runtime_safety_action_latched = False
            try:
                self.log.event("runtime_safety_rearmed", reason=reason)
            except Exception:
                pass

    def _schedule_runtime_safety_action(self, reason: str) -> bool:
        with self._runtime_safety_action_lock:
            if (
                self._runtime_safety_action_latched
                or self._cleanup_done
                or self.drone is None
            ):
                return False
            self._runtime_safety_action_latched = True
            self._runtime_safety_action_reason = str(reason)
            self.state.active_incident = str(reason)
            self._set_tracker_state(TrackerState.SAFETY_ACTION_PENDING, reason="_schedule_runtime_safety_action")
            self.state.last_command = f"runtime_safety:{reason}"
            thread = threading.Thread(
                target=self._execute_runtime_safety_action,
                args=(str(reason),),
                name="runtime-flight-safety",
                daemon=True,
            )
            self._runtime_safety_action_thread = thread
            thread.start()
        self.log.event("runtime_safety_latched", reason=reason)
        return True

    def _runtime_safety_telemetry(
        self,
        name: str,
        state_attr: str,
        value_key: str,
        now_mono_ns: int,
    ) -> tuple[float | None, bool]:
        sample = self.telemetry_freshness.sample(name)
        if sample is None:
            return self._finite_float(getattr(self.state, state_attr, None)), False
        fresh = self.telemetry_freshness.fresh(
            name,
            max_age_s=CRITICAL_TELEMETRY_MAX_AGE_S,
            now_mono_ns=now_mono_ns,
        )
        value = self._finite_float(None if fresh is None else fresh.get(value_key))
        return value, True

    def _evaluate_runtime_safety(self, now_mono_ns: int | None = None) -> bool:
        flight_state = self._flight_state_name()
        if (
            self._runtime_safety_action_latched
            or self._cleanup_done
            or self.drone is None
            or not bool(getattr(self.state, "link_ok", True))
            or flight_state not in {"hovering", "flying"}
        ):
            return False
        safety_now_ns = (
            time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
        )
        altitude, _altitude_sampled = self._runtime_safety_telemetry(
            "altitude", "drone_altitude_m", "altitude", safety_now_ns
        )
        altitude_limit = self._finite_float(
            getattr(self.state, "max_altitude_m", None)
        )
        if (
            self.desired_max_altitude_m is not None
            and altitude is not None
            and altitude_limit is not None
            and altitude >= max(0.0, altitude_limit - _ALTITUDE_LIMIT_MARGIN_M)
        ):
            return self._schedule_runtime_safety_action(
                FailureReason.ALTITUDE_LIMIT.value
            )
        distance, _distance_sampled = self._runtime_safety_telemetry(
            "distance", "distance_from_home_m", "distance", safety_now_ns
        )
        distance_limit = self._finite_float(
            getattr(self.state, "max_distance_m", None)
        )
        distance_requested = bool(
            self.desired_max_distance_m is not None and self.desired_distance_geofence
        )
        # Truthfulness: with no GPS position or no Home Point this check silently
        # evaluates nothing. Say so instead of letting the UI imply containment.
        guard_active = bool(
            distance_requested and distance is not None and distance_limit is not None
        )
        if distance_requested and guard_active != self._distance_guard_active:
            self.log.event(
                "distance_guard",
                active=guard_active,
                distance_from_home_m=distance,
                max_distance_m=distance_limit,
                note=("distance containment active" if guard_active else
                      "no GPS position/Home Point: distance containment NOT enforced"),
            )
        self._distance_guard_active = guard_active
        self.state.distance_guard_active = guard_active if distance_requested else None
        # No GPS/Home is a normal operating condition at some approved sites.
        # The distance fence is then unavailable and is shown as such, but it does
        # not block takeoff or force a landing. Altitude, stream, link,
        # pose and operator-takeover guards remain active independently.
        if (
            guard_active
            and distance >= distance_limit * _DISTANCE_LIMIT_FRACTION
        ):
            return self._schedule_runtime_safety_action(
                FailureReason.DISTANCE_LIMIT.value
            )
        return False

    def _latch_total_link_loss(self) -> None:
        """Stop host-side assumptions without sending into a dead control link."""
        if self.state.active_incident == FailureReason.CONTROL_LINK_LOST.value:
            return
        with self._lock:
            self._nudge_held.clear()
            self._nudge_deadline = 0.0
            self._nudge_loop_stop.set()
            self._motion_epoch += 1
            self._pulse_token += 1
            self.pilot_sticks = True
            self._link_was_ok = False
            self._link_failure_count = _LINK_FAILURE_CONFIRM_POLLS
        self.state.mode = "MANUAL"
        self.state.control_owner = "ONBOARD_LOST_LINK_POLICY"
        self._set_tracker_state(TrackerState.LINK_LOST_ONBOARD, reason="_latch_total_link_loss")
        self.state.last_command = "host_commands_blocked:control_link_lost"
        self.state.active_incident = FailureReason.CONTROL_LINK_LOST.value
        self.log.event(
            "link_lost",
            policy="onboard_hover_then_verified_rth_or_landing",
            host_command_sent=False,
            auto_resume=False,
        )

    def set_gravity_sim_phase(self, phase: str | None) -> None:
        # Live: never synthesize attitude; real IMU is polled in poll().
        self.gravity_sim_phase = None
        self._gravity_sim_t0 = None
        self.log.event("gravity_phase", phase=phase, mode="live_real_attitude")

    def _probe_host_link_ok(self) -> bool:
        if self.drone is None:
            return False
        host_ok: bool | None = None
        try:
            cs = self.drone.connection_state()
            name = getattr(cs, "name", None) or str(cs)
            s = str(name).lower()
            if "disconnect" in s or "error" in s or s.endswith(".ko"):
                return False
            if "connect" in s and "disconnect" not in s:
                # Created/Connecting/Connected — only Connected is fully OK
                if "created" in s or "connecting" in s:
                    host_ok = getattr(self, "_link_was_ok", False)
                else:
                    host_ok = True
        except Exception:
            pass
        if host_ok is None:
            try:
                # Fallback: any get_state success implies the host command link is up.
                from olympe.messages.common.CommonState import BatteryStateChanged
                self.drone.get_state(BatteryStateChanged)
                host_ok = True
            except Exception:
                return False
        return bool(host_ok)

    def _probe_link_ok(self) -> bool:
        """Best-effort Olympe connection probe (display only; no flight cmds)."""
        host_ok = self._probe_host_link_ok()
        if not host_ok or not self.via_skycontroller():
            return host_ok
        try:
            from olympe.messages.drone_manager import connection_state

            managed = self._state_dict(connection_state)
        except Exception:
            return True
        if managed is None or managed.get("state") is None:
            return True
        state = str(
            getattr(managed.get("state"), "name", managed.get("state"))
        ).rsplit(".", 1)[-1].lower()
        return state == "connected"

    def _poll_get_state(self, message):
        try:
            return self.drone.get_state(message)
        except Exception:
            return None

    def _poll_link_status(self) -> bool:
        observed_link_ok = self._probe_link_ok()
        if observed_link_ok:
            self._link_failure_count = 0
            link_ok = True
            link_status = "OK"
        else:
            self._link_failure_count += 1
            link_ok = self._link_failure_count < _LINK_FAILURE_CONFIRM_POLLS
            link_status = "DEGRADED" if link_ok else "LOST"
        prev = getattr(self, "_link_was_ok", True)
        self._link_was_ok = link_ok
        self.state.link_ok = link_ok
        self.state.link_status = link_status
        if prev and not link_ok:
            self._latch_total_link_loss()
        return link_ok

    def _poll_mandatory_messages(self):
        try:
            from olympe.messages.ardrone3.PilotingState import (
                AttitudeChanged,
                AltitudeChanged,
                FlyingStateChanged,
                SpeedChanged,
            )
            from olympe.messages.common.CommonState import BatteryStateChanged
        except Exception:
            return None
        return AttitudeChanged, AltitudeChanged, FlyingStateChanged, SpeedChanged, BatteryStateChanged

    def _poll_link_loss(self, now: float) -> None:
        self._clear_firmware_safety_state()
        self._clear_magnetometer_calibration_state()
        # Skip heavy telemetry when link is down; keep LOST flags.
        if self.video_stream is not None:
            stamp = float(getattr(self.video_stream, "last_stamp", 0.0) or 0.0)
            if stamp > 0:
                age = max(0.0, (now - stamp) * 1000.0)
                self.state.frame_age_ms = age
                self.state.link_latency_ms = age
        self.state.telemetry_read_mono_ns = time.monotonic_ns()
        last_pcmd_ns = getattr(self.state, "last_pcmd_call_mono_ns", None)
        if last_pcmd_ns is not None:
            self.state.pcmd_to_telemetry_poll_ms = max(
                0.0,
                (self.state.telemetry_read_mono_ns - int(last_pcmd_ns))
                / 1_000_000.0,
            )

    def _poll_mandatory_telemetry(self, messages, poll_mono_ns: int) -> None:
        AttitudeChanged, AltitudeChanged, FlyingStateChanged, SpeedChanged, BatteryStateChanged = messages
        att = self._poll_get_state(AttitudeChanged)
        if att:
            self.state.att_roll = float(att.get("roll", 0.0) or 0.0)
            self.state.att_pitch = float(att.get("pitch", 0.0) or 0.0)
            self.state.att_yaw = float(att.get("yaw", 0.0) or 0.0)
            self.sim_yaw = self.state.att_yaw
        alt = self._poll_get_state(AltitudeChanged)
        if alt and alt.get("altitude") is not None:
            alt = self._observe_telemetry(
                "altitude",
                AltitudeChanged,
                alt,
                now_mono_ns=poll_mono_ns,
            )
            self.state.altitude_m = float(alt["altitude"])
            self.state.drone_altitude_m = float(alt["altitude"])
        speed = self._poll_get_state(SpeedChanged)
        if speed:
            speed = self._observe_telemetry(
                "ground_speed",
                SpeedChanged,
                speed,
                now_mono_ns=poll_mono_ns,
            )
            sx = self._finite_float(speed.get("speedX"))
            sy = self._finite_float(speed.get("speedY"))
            sz = self._finite_float(speed.get("speedZ"))
            self.state.speed_north_mps = sx
            self.state.speed_east_mps = sy
            self.state.speed_down_mps = sz
            self.state.ground_speed_mps = (
                math.hypot(sx, sy) if sx is not None and sy is not None else None
            )
            # Legacy compatibility for logs/tests written before the value
            # was correctly named as NED horizontal ground speed.
            self.state.airspeed_mps = self.state.ground_speed_mps
        bat = self._poll_get_state(BatteryStateChanged)
        if bat and bat.get("percent") is not None:
            bat = self._observe_telemetry(
                "battery",
                BatteryStateChanged,
                bat,
                now_mono_ns=poll_mono_ns,
            )
            self.state.battery_pct = float(bat["percent"])
        fly = self._poll_get_state(FlyingStateChanged)
        if fly and fly.get("state") is not None:
            st = str(self._state_value(fly, "state"))
            self.state.flight_state = st
            if self.state.tracker_state not in _TRACKER_STATE_STICKY:
                self.state.tracker_state = st.upper()

    def _poll_gps_fix(self) -> None:
        try:
            from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
            gps = self._poll_get_state(GPSFixStateChanged)
            if gps is not None and gps.get("fixed") is not None:
                self.state.gps_fixed = bool(int(gps.get("fixed") or 0))
        except Exception:
            pass

    def _poll_navigation_location(self, messages, poll_mono_ns: int) -> None:
        AltitudeAboveGroundChanged, GpsLocationChanged = messages[:2]
        agl = self._poll_get_state(AltitudeAboveGroundChanged)
        if agl and agl.get("altitude") is not None:
            self.state.agl_altitude_m = self._finite_float(agl.get("altitude"))
        location = self._poll_get_state(GpsLocationChanged)
        if not location:
            return
        location = self._observe_telemetry(
            "gps_location",
            GpsLocationChanged,
            location,
            now_mono_ns=poll_mono_ns,
        )
        latitude = self._finite_float(location.get("latitude"))
        longitude = self._finite_float(location.get("longitude"))
        self.state.gps_latitude_deg = (
            latitude if latitude is not None and -90.0 <= latitude <= 90.0
            else None
        )
        self.state.gps_longitude_deg = (
            longitude
            if longitude is not None and -180.0 <= longitude <= 180.0
            else None
        )
        self._update_distance_from_home(
            self.state.gps_latitude_deg,
            self.state.gps_longitude_deg,
        )
        location_sample = self.telemetry_freshness.sample("gps_location")
        if location_sample is not None:
            self.telemetry_freshness.observe(
                "distance",
                {"distance": getattr(self.state, "distance_from_home_m", None)},
                marker=location_sample.marker,
                observed_mono_ns=location_sample.observed_mono_ns,
            )
            self.state.distance_mono_ns = location_sample.observed_mono_ns
        self.state.gps_altitude_m = self._finite_float(location.get("altitude"))
        for source, target in (
            ("latitude_accuracy", "gps_latitude_accuracy_m"),
            ("longitude_accuracy", "gps_longitude_accuracy_m"),
            ("altitude_accuracy", "gps_altitude_accuracy_m"),
        ):
            accuracy = self._finite_float(location.get(source))
            setattr(self.state, target, accuracy if accuracy is not None and accuracy >= 0 else None)

    def _poll_hover_warning(self, hover_warning) -> None:
        if hover_warning:
            if hover_warning.get("no_gps_too_dark") is not None:
                self.state.hover_no_gps_too_dark = bool(
                    int(hover_warning.get("no_gps_too_dark") or 0)
                )
            if hover_warning.get("no_gps_too_high") is not None:
                self.state.hover_no_gps_too_high = bool(
                    int(hover_warning.get("no_gps_too_high") or 0)
                )

    def _poll_navigation_status(self, messages) -> None:
        (
            AlertStateChanged,
            NavigateHomeStateChanged,
            HeadingLockedStateChanged,
            HoveringWarning,
            WindStateChanged,
            VibrationLevelChanged,
        ) = messages
        alert = self._poll_get_state(AlertStateChanged)
        if alert and alert.get("state") is not None:
            self.state.alert_state = str(self._state_value(alert, "state"))
        navigate_home = self._poll_get_state(NavigateHomeStateChanged)
        if navigate_home:
            if navigate_home.get("state") is not None:
                self.state.navigate_home_state = str(
                    self._state_value(navigate_home, "state")
                )
            if navigate_home.get("reason") is not None:
                self.state.navigate_home_reason = str(
                    self._state_value(navigate_home, "reason")
                )
        heading = self._poll_get_state(HeadingLockedStateChanged)
        if heading and heading.get("state") is not None:
            self.state.heading_state = str(self._state_value(heading, "state"))
        hover_warning = self._poll_get_state(HoveringWarning)
        self._poll_hover_warning(hover_warning)
        wind = self._poll_get_state(WindStateChanged)
        if wind and wind.get("state") is not None:
            self.state.wind_state = str(self._state_value(wind, "state"))
        vibration = self._poll_get_state(VibrationLevelChanged)
        if vibration and vibration.get("state") is not None:
            self.state.vibration_state = str(
                self._state_value(vibration, "state")
            )

    def _poll_optional_navigation(self, poll_mono_ns: int) -> None:
        try:
            from olympe.messages.ardrone3.PilotingState import (
                AlertStateChanged,
                AltitudeAboveGroundChanged,
                GpsLocationChanged,
                HeadingLockedStateChanged,
                HoveringWarning,
                NavigateHomeStateChanged,
                VibrationLevelChanged,
                WindStateChanged,
            )
            self._poll_navigation_location(
                (AltitudeAboveGroundChanged, GpsLocationChanged), poll_mono_ns
            )
            self._poll_navigation_status(
                (
                    AlertStateChanged,
                    NavigateHomeStateChanged,
                    HeadingLockedStateChanged,
                    HoveringWarning,
                    WindStateChanged,
                    VibrationLevelChanged,
                )
            )
        except Exception:
            # These read-only events are optional across Olympe/product
            # versions; mandatory battery/attitude/altitude remain visible.
            pass

    def _poll_satellites(self) -> None:
        try:
            from olympe.messages.ardrone3.GPSState import NumberOfSatelliteChanged

            satellites = self._poll_get_state(NumberOfSatelliteChanged)
            if satellites and satellites.get("numberOfSatellite") is not None:
                number = int(satellites["numberOfSatellite"])
                self.state.gps_satellites = number if number >= 0 else None
        except Exception:
            pass

    def _poll_signal_sensors(self) -> None:
        try:
            from olympe.messages.common.CommonState import (
                LinkSignalQuality,
                SensorsStatesListChanged,
                WifiSignalChanged,
            )

            wifi = self._poll_get_state(WifiSignalChanged)
            if wifi and wifi.get("rssi") is not None:
                rssi = self._finite_float(wifi.get("rssi"))
                self.state.wifi_rssi_dbm = None if rssi is None else int(rssi)
            link_quality = self._poll_get_state(LinkSignalQuality)
            if link_quality and link_quality.get("value") is not None:
                quality = int(link_quality["value"])
                self.state.link_signal_quality_raw = (
                    quality if 0 <= quality <= 255 else None
                )
            sensor_updates = self._sensor_state_updates(
                self._poll_get_state(SensorsStatesListChanged)
            )
            if sensor_updates:
                current = dict(getattr(self.state, "sensor_states", {}) or {})
                current.update(sensor_updates)
                self.state.sensor_states = current
        except Exception:
            pass

    def _poll_stick_axes(self, now: float) -> None:
        """Log what the pilot is asking for; see ``stick_log_sample``."""
        if self._stick_monitor is None:
            return
        sample = stick_log_sample(
            self._stick_monitor,
            self._last_stick_axes,
            since_last_s=now - self._last_stick_axes_t,
        )
        if sample is None:
            return
        self._last_stick_axes, fields = sample
        self._last_stick_axes_t = now
        self.session_logs.telemetry(
            "stick_axes",
            t_mono_ns=time.monotonic_ns(),
            pilot_sticks=bool(self.pilot_sticks),
            **fields,
        )

    def _poll_session_telemetry(self, now: float) -> None:
        if self.session_logs is None:
            return
        self._poll_stick_axes(now)
        # High-rate NED velocity + attitude for offline ESEKF/KLT-3D A/B replay
        # (定位演算法/validation/benchmark_esekf_live_replay.py). The 1 Hz
        # "readback" event below is too coarse to feed observe_fused_state.
        last_fused = getattr(self, "_last_fused_odometry_t", 0.0)
        if now - last_fused >= 0.1:
            self._last_fused_odometry_t = now
            self.session_logs.telemetry(
                "fused_odometry",
                t_mono_ns=time.monotonic_ns(),
                speed_north_mps=getattr(self.state, "speed_north_mps", None),
                speed_east_mps=getattr(self.state, "speed_east_mps", None),
                speed_down_mps=getattr(self.state, "speed_down_mps", None),
                att_roll=getattr(self.state, "att_roll", None),
                att_pitch=getattr(self.state, "att_pitch", None),
                att_yaw=getattr(self.state, "att_yaw", None),
                drone_altitude_m=getattr(self.state, "drone_altitude_m", None),
                gps_latitude_deg=getattr(self.state, "gps_latitude_deg", None),
                gps_longitude_deg=getattr(self.state, "gps_longitude_deg", None),
                gps_altitude_m=getattr(self.state, "gps_altitude_m", None),
            )
        if self.session_logs is not None:
            last_session_tel = getattr(self, "_last_session_telemetry_t", 0.0)
            if now - last_session_tel >= 1.0:
                self._last_session_telemetry_t = now
                self.session_logs.telemetry(
                    "readback",
                    battery_pct=getattr(self.state, "battery_pct", None),
                    gps_fixed=getattr(self.state, "gps_fixed", None),
                    home_valid=getattr(self.state, "home_valid", None),
                    home_reachable=getattr(self.state, "home_reachable", None),
                    distance_from_home_m=getattr(
                        self.state, "distance_from_home_m", None
                    ),
                    rth_policy_valid=getattr(self.state, "rth_policy_valid", None),
                    rth_policy_configured=getattr(
                        self.state, "rth_policy_configured", None
                    ),
                    stick_monitor_ok=getattr(self.state, "stick_monitor_ok", None),
                    active_incident=getattr(self.state, "active_incident", None),
                    altitude_m=getattr(self.state, "drone_altitude_m", None),
                    agl_altitude_m=getattr(self.state, "agl_altitude_m", None),
                    ground_speed_mps=getattr(self.state, "ground_speed_mps", None),
                    speed_north_mps=getattr(self.state, "speed_north_mps", None),
                    speed_east_mps=getattr(self.state, "speed_east_mps", None),
                    speed_down_mps=getattr(self.state, "speed_down_mps", None),
                    airspeed_mps=getattr(self.state, "airspeed_mps", None),
                    heading_state=getattr(self.state, "heading_state", None),
                    alert_state=getattr(self.state, "alert_state", None),
                    wind_state=getattr(self.state, "wind_state", None),
                    vibration_state=getattr(self.state, "vibration_state", None),
                    link_status=getattr(self.state, "link_status", None),
                    max_altitude_m=getattr(self.state, "max_altitude_m", None),
                    max_distance_m=getattr(self.state, "max_distance_m", None),
                    distance_geofence=getattr(
                        self.state, "distance_geofence_enabled", None
                    ),
                )

    def _poll_gimbal(self) -> None:
        try:
            from olympe.messages.gimbal import attitude as gimbal_attitude
            g = self._poll_get_state(gimbal_attitude)
            if isinstance(g, dict) and g:
                first = next(iter(g.values())) if g else None
                if isinstance(first, dict) and first.get("pitch_absolute") is not None:
                    self.state.gimbal_pitch_deg = float(first["pitch_absolute"])
        except Exception:
            pass

    def _poll_telemetry(self, poll_mono_ns: int, now: float) -> bool:
        link_ok = self._poll_link_status()
        messages = self._poll_mandatory_messages()
        if messages is None:
            return True
        if not link_ok:
            self._poll_link_loss(now)
            return True
        self._read_magnetometer_calibration_state()
        self._poll_mandatory_telemetry(messages, poll_mono_ns)
        self._poll_gps_fix()
        self._poll_optional_navigation(poll_mono_ns)
        self._poll_satellites()
        self._poll_signal_sensors()
        # Olympe get_state reads the local event cache; these values shown
        # in the UI never come from desired constructor settings.
        self._read_firmware_safety_state()
        self._poll_session_telemetry(now)
        self._poll_gimbal()
        self.state.telemetry_read_mono_ns = time.monotonic_ns()
        last_pcmd_ns = getattr(self.state, "last_pcmd_call_mono_ns", None)
        if last_pcmd_ns is not None:
            self.state.pcmd_to_telemetry_poll_ms = max(
                0.0,
                (self.state.telemetry_read_mono_ns - int(last_pcmd_ns))
                / 1_000_000.0,
            )
        return False

    def poll(self, now_mono_ns: int | None = None) -> Any:
        poll_mono_ns = (
            time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
        )
        now = (
            time.monotonic()
            if now_mono_ns is None
            else float(poll_mono_ns) * 1e-9
        )
        self.last_poll = now
        self._check_runtime_storage_health(now)
        if self.drone is None:
            self.state.link_ok = False
            self.state.link_status = "LOST"
            self._clear_firmware_safety_state()
            self._clear_magnetometer_calibration_state()
            return self.state

        # Belt-and-suspenders: reclaim sticks even if HID callback missed a frame.
        if self.via_skycontroller() and not self.pilot_sticks:
            self._maybe_reclaim_from_sticks()

        # Link + frame-age at UI rate; full telemetry ~8 Hz.
        last_tel = getattr(self, "_last_telemetry_t", 0.0)
        do_tel = (now - last_tel) >= 0.12
        if do_tel:
            self._last_telemetry_t = now
            if self._poll_telemetry(poll_mono_ns, now):
                return self.state
        if self.video_stream is not None:
            self.state.stream_fps = float(getattr(self.video_stream, "fps", 0.0) or 0.0)
            stamp = float(getattr(self.video_stream, "last_stamp", 0.0) or 0.0)
            if stamp > 0:
                age = max(0.0, (now - stamp) * 1000.0)
                self.state.frame_age_ms = age
                # Mapped source/backlog age; fixed camera/network baseline is unknown.
                self.state.link_latency_ms = age
            else:
                self.state.frame_age_ms = None
            self._maybe_log_video_inventory()
            self._evaluate_video_health(now)
        self._evaluate_runtime_safety(poll_mono_ns)
        # Do not clobber UI stream labels (PREVIEW/OK/HOLD) every poll.
        return self.state

    def _check_runtime_storage_health(self, now: float) -> bool:
        """Fail safe once if durable logging or critical disk capacity is lost."""
        if now - self._last_runtime_storage_check_t < 1.0:
            return not self._runtime_storage_guard_latched
        self._last_runtime_storage_check_t = now

        log_healthy = bool(getattr(self.log, "healthy", True))
        log_durable = bool(getattr(self.log, "durable", getattr(self.log, "path", None)))
        disk = None
        disk_error = ""
        try:
            disk = assess_disk_space(self.disk_guard_path)
            self.state.disk_free_bytes = disk.free_bytes
            self.state.disk_free_percent = disk.free_percent
            self.state.disk_warning = disk.warning
        except Exception as exc:
            disk_error = repr(exc)
            self.state.disk_warning = True

        # The documented contract is fail-closed before takeoff at <5 GiB or <5%
        # free. Once airborne, low capacity is recorded only; taking control away
        # from the safety pilot because retention space is low is not safe. An
        # unreadable disk (disk is None) or a log that actually failed still needs
        # the in-flight logging/manual-handoff fail-safe below.
        disk_unreadable = disk is None
        if disk is not None and bool(disk.takeoff_blocked) and not self._disk_low_noted:
            self._disk_low_noted = True
            self.log.event(
                "disk_low_inflight", ok=True, reason=disk.reason,
                free_percent=disk.free_percent,
                note="recording may stop; command authority is unaffected",
            )
        disk_critical = disk_unreadable
        unhealthy = not log_healthy or not log_durable or disk_critical
        if not unhealthy:
            return True
        if self._runtime_storage_guard_latched:
            return False

        self._runtime_storage_guard_latched = True
        reason = (
            disk_error
            or (disk.reason if disk_critical and disk is not None else "")
            or "durable safety logging became unavailable"
        )
        try:
            self.log.event(
                "disk_critical" if disk_critical else "logging_failed",
                reason=reason,
                log_healthy=log_healthy,
                log_durable=log_durable,
                action="zero_pcmd_hover_manual_handoff",
            )
        except Exception:
            pass
        self.fail_safe(FailureReason.DISK_OR_LOG_FAILURE)
        return False

    def _is_already_landed(self) -> bool:
        if self._landed:
            return True
        if self.drone is None:
            return False
        try:
            from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
            st = self.drone.get_state(FlyingStateChanged)
            if isinstance(st, dict):
                value = st.get("state", "")
                s = str(getattr(value, "name", value)).rsplit(".", 1)[-1].lower()
                if s == "landed":
                    with self._lock:
                        self._landed = True
                    return True
        except Exception:
            pass
        return False

    def _cleanup_log(self, event: str, **fields: Any) -> None:
        # Cleanup logging is diagnostic only. A broken sink must never
        # prevent the bounded Landing attempt below.
        try:
            self.log.event(event, **fields)
        except Exception:
            pass

    def latch_shutdown_motion(self, *, reason: str) -> bool:
        """Permanently retire PC motion before a shutdown Landing attempt."""
        with self._lock:
            if self._cleanup_done:
                return True
            self._shutdown_latched = True
            self._pulse_token += 1
            self._motion_epoch += 1
            self._nudge_vector = None
            self._nudge_held.clear()
            self._nudge_deadline = 0.0
            self._nudge_loop_stop.set()
            zero_ok = self._zero_pcmd_or_log(f"shutdown_latch:{reason}")
        self._cleanup_log(
            "shutdown_motion_latched",
            reason=reason,
            zero_ok=zero_ok,
        )
        return zero_ok

    def _cleanup_attempt_landing(self) -> bool:
        # A late AUTO-cancel manual handoff may still be finishing when Ctrl+C
        # reaches cleanup. Keep Controller ownership, Landing, touchdown
        # confirmation, and the final stick handoff in one serialized authority
        # transition so that thread cannot switch back to SkyController between
        # the Landing source handoff and the command itself.
        with self.authority_controller.transition():
            return self._cleanup_attempt_landing_under_authority()

    def _cleanup_attempt_landing_under_authority(self) -> bool:
        # Zero immediately, even before a potentially blocking source
        # acknowledgement. A second zero follows a confirmed handoff.
        try:
            with self._lock:
                self._zero_pcmd_or_log("cleanup:pre_handoff_zero")
        except Exception as exc:
            self._cleanup_log(
                "cleanup_pre_action_error",
                action="pre_handoff_zero",
                error=repr(exc),
            )
        try:
            source_ok = self._set_piloting_source("Controller")
        except Exception as exc:
            source_ok = False
            self._cleanup_log(
                "cleanup_pre_action_error",
                action="piloting_source",
                error=repr(exc),
            )
        if source_ok:
            try:
                with self._lock:
                    self.pilot_sticks = False
                    self._zero_pcmd_or_log("cleanup:post_handoff_zero")
            except Exception as exc:
                self._cleanup_log(
                    "cleanup_pre_action_error",
                    action="post_handoff_zero",
                    error=repr(exc),
                )
        else:
            self._cleanup_log(
                "land_source_unconfirmed",
                reason="cleanup",
                note="Landing still attempted as fail-safe",
            )

        try:
            outcome = self.takeoff_landing_supervisor.ensure_landed(
                reason="cleanup_force_land",
                is_landed=self._is_already_landed,
                land_and_confirm=self._land_and_confirm,
            )
            landed = outcome.confirmed
        except Exception as exc:
            landed = False
            outcome = None
            self._cleanup_log(
                "land_cmd",
                reason="cleanup_force_land",
                ok=False,
                error=repr(exc),
                note="landing_attempt_failed",
            )
        if landed and outcome is not None and outcome.reason_code == "ALREADY_LANDED":
            self._set_tracker_state(TrackerState.LAND, reason="_cleanup_attempt_landing_under_authority")
            self._cleanup_log(
                "land_cmd",
                reason="cleanup_exit",
                ok=True,
                note="already_landed",
            )

        if landed and self.via_skycontroller():
            handed_off = self._set_piloting_source("SkyController")
            with self._lock:
                self.pilot_sticks = handed_off
        elif landed:
            with self._lock:
                self.pilot_sticks = True
        return landed

    def _cleanup_recording(self, landed: bool) -> None:
        # Never let recording finalization or a media download delay the
        # zero/Landing safety path. Skip it if touchdown is unconfirmed.
        if self.recording_active and landed:
            try:
                self.stop_flight_recording(reason="cleanup", download=False)
            except Exception as exc:
                self._cleanup_log(
                    "record_stop", ok=False, reason="cleanup", error=repr(exc)
                )
        elif self.recording_active:
            self._cleanup_log(
                "record_stop_deferred",
                reason="cleanup",
                note="touchdown_unconfirmed",
            )

    def _cleanup_resources(self) -> None:
        try:
            self._stop_stick_monitor()
        except Exception as exc:
            self._cleanup_log("cleanup_stick_monitor_error", error=repr(exc))
        if self.grabber is not None:
            try:
                self.grabber.stop()
            except Exception:  # Tier3: teardown best-effort — grabber stop optional
                pass
            self.grabber = None
        if self.drone is not None:
            try:
                self.drone.disconnect()
            except Exception as exc:
                self._cleanup_log("disconnect_error", error=repr(exc))
            self.drone = None

    def cleanup(self) -> bool:
        """Confirm touchdown before finalizing media or disconnecting.

        Called on: UI window close, Ctrl-C / SIGTERM / SIGHUP, process atexit.

        【強制降落 — 禁止刪除或弱化】關窗／Ctrl+C 時：只要未落地就必須原地 Landing，
        即使曾按 Esc 交回搖桿。弱化此邏輯可能導致空中失控。見 mission/SAFETY.md。
        """
        with self._cleanup_lock:
            with self._lock:
                if self._cleanup_done:
                    return True
            self.latch_shutdown_motion(reason="exit_or_signal")

            self._cleanup_log("cleanup_begin", reason="exit_or_signal")
            try:
                self.nudge_clear(reason="cleanup")
            except Exception as exc:
                self._cleanup_log(
                    "cleanup_pre_action_error",
                    action="nudge_clear",
                    error=repr(exc),
                )
            landed = self.drone is None
            if self.drone is not None:
                landed = self._cleanup_attempt_landing()

            self._cleanup_recording(landed)

            if not landed:
                try:
                    self.give_to_pilot(reason="cleanup_land_unconfirmed")
                except Exception as exc:
                    self._cleanup_log(
                        "cleanup_manual_handoff_error",
                        error=repr(exc),
                    )
                self._cleanup_log(
                    "cleanup_incomplete",
                    reason_code="LAND_UNCONFIRMED",
                    action="connection_preserved_for_retry",
                )
                return False

            self._cleanup_resources()
            with self._lock:
                self._cleanup_done = True
            self._cleanup_log("cleanup_done", touchdown_confirmed=True)
            try:
                self.log.close()
            except Exception:  # Tier3: teardown best-effort — log close optional
                pass
            return True

    def close(self, reason: str) -> CloseResult:
        self.log.event("close_requested", reason=reason)
        closed = self.cleanup()
        return CloseResult(closed, "OK" if closed else "LAND_UNCONFIRMED")
