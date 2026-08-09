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
  - Does NOT auto-start path_follow_flight --fly (mission AUTO remains UI mode
    + localization; arm full AUTO via path_follow_flight.py separately).

Network
  SkyController USB:  --ip 192.168.53.1 --controller skycontroller3
  Direct drone WiFi:  --ip 192.168.42.1 --controller drone
"""
from __future__ import annotations

import glob
import importlib.metadata
import json
import math
import os
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from scale_free_control_adapter import validate_speed_limit_change  # noqa: E402
from live_safety_config import (  # noqa: E402
    CRITICAL_BATTERY_PCT,
    DEFAULT_MAX_ROTATION_SPEED_DEGS,
    DEFAULT_MAX_TILT_DEG,
    DEFAULT_MAX_VERTICAL_SPEED_MS,
    DEFAULT_RTH_MIN_ALTITUDE_M,
    DEFAULT_STREAM_LOSS_GRACE_S,
    LiveSafetyConfig,
    MIN_TAKEOFF_BATTERY_PCT,
    NUDGE_TTL_MAX_S,
)

# Local download dir for flight recordings (stop_recording finalizes on drone media).
_DEFAULT_RECORD_DIR = _WS.flight_logs / "recordings"
_MEDIA_PENDING_TIMEOUT_S = 2.0
_MEDIA_DOWNLOAD_TIMEOUT_S = 15.0
# Match the documented flight-control command TTL at the typed backend boundary.
# A queued ordinary request is rejected once it is older than this; explicit
# LAND_NOW and EMERGENCY_STOP requests remain executable regardless of age.
CONTROL_REQUEST_MAX_AGE_NS = 250_000_000


def _clamp_pct(v: int) -> int:
    return max(-100, min(100, int(v)))


# Linux joystick API (see linux/joystick.h). Axis range is typically ±32767.
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80
_JS_EVENT_FMT = "IhBB"
_JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FMT)
# ~6% of full throw — ignores rest noise, still snaps back on deliberate input.
_STICK_DEADZONE = 2000
# Only the two flight sticks may seize control from the PC. Measured on a Parrot
# SkyController 3 v1.8.1 (2026-08-06, ground bench): axes 0-3 are the two sticks,
# axes 4-5 are the camera/gimbal wheel and shoulder controls. Moving the gimbal
# wheel is a CAMERA action, not "the pilot wants the aircraft" -- counting it made
# every camera adjustment yank flight authority back from the PC.
#: Consecutive override-callback failures before the monitor declares itself
#: unhealthy, which fails the takeoff gate and lands an airborne aircraft.
_STICK_CALLBACK_FAIL_LIMIT = 3
_STICK_FLIGHT_AXES = (0, 1, 2, 3)
_STICK_POLL_S = 0.02  # 50 Hz reclaim path
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
_CRITICAL_BATTERY_PCT = CRITICAL_BATTERY_PCT
_ALTITUDE_LIMIT_MARGIN_M = 1.0
_DISTANCE_LIMIT_FRACTION = 0.95
_LOST_LINK_RTH_DELAY_S = 1
_EARTH_RADIUS_M = 6_371_000.0


def find_skycontroller_joystick_path() -> str | None:
    """Resolve the SC3 HID joystick node (prefer stable by-id symlink)."""
    by_id = sorted(
        glob.glob("/dev/input/by-id/usb-Parrot*Skycontroller*-joystick")
    )
    # Prefer the js node, not event-joystick (name ends with -joystick only).
    for path in by_id:
        if path.endswith("-event-joystick"):
            continue
        if os.path.exists(path):
            return path
    # Fallback: scan js* devices for a Parrot/Skycontroller name.
    for path in sorted(glob.glob("/dev/input/js*")):
        name = _joystick_device_name(path)
        if not name:
            continue
        lowered = name.lower()
        if "skycontroller" in lowered or (
            "parrot" in lowered and "sky" in lowered
        ):
            return path
    return None


def _joystick_device_name(path: str) -> str | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        import array
        import fcntl

        buf = array.array("b", b"\0" * 128)

        def _ioc_read(type_char: str, nr: int, size: int) -> int:
            return (2 << 30) | (size << 16) | (ord(type_char) << 8) | nr

        fcntl.ioctl(fd, _ioc_read("j", 0x13, 128), buf, True)
        return buf.tobytes().split(b"\0", 1)[0].decode("utf-8", "replace")
    except OSError:
        return None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def axes_active(
    axes: dict[int, int],
    *,
    deadzone: int = _STICK_DEADZONE,
    flight_axes: tuple[int, ...] | None = _STICK_FLIGHT_AXES,
) -> bool:
    """True when a FLIGHT stick axis exceeds deadzone (intentional deflection).

    Camera axes (gimbal wheel / shoulder) are ignored: adjusting the camera while
    the PC is flying must not be mistaken for the pilot taking the aircraft back.
    Pass ``flight_axes=None`` to consider every axis.
    """
    dz = max(0, int(deadzone))
    for axis, value in axes.items():
        try:
            if flight_axes is not None and int(axis) not in flight_axes:
                continue
            if abs(int(value)) > dz:
                return True
        except (TypeError, ValueError):
            continue
    return False


class SkyControllerStickMonitor:
    """Non-blocking Linux joystick reader for SC stick override.

    When any stick axis leaves the deadzone, ``on_active`` is invoked. The
    callback must be cheap and re-entrant-safe (backend uses its own locks).
    """

    def __init__(
        self,
        on_active: Callable[[dict[int, int]], None],
        *,
        device_path: str | None = None,
        deadzone: int = _STICK_DEADZONE,
        poll_s: float = _STICK_POLL_S,
        on_disconnect: Callable[[str], None] | None = None,
        log_event: Callable[..., None] | None = None,
    ):
        self._on_active = on_active
        self._on_disconnect = on_disconnect
        self._device_path = device_path
        self.deadzone = int(deadzone)
        self.poll_s = float(poll_s)
        self._log_event = log_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._axes: dict[int, int] = {}
        self._callback_failures = 0
        self._lock = threading.Lock()
        self.resolved_path: str | None = None
        self.device_name: str | None = None
        self.healthy = False
        self.disconnect_reason = ""

    def start(self) -> bool:
        if self._thread is not None:
            return True
        path = self._device_path or find_skycontroller_joystick_path()
        if not path:
            if self._log_event is not None:
                self._log_event(
                    "stick_monitor",
                    ok=False,
                    note="no_skycontroller_joystick",
                )
            return False
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            if self._log_event is not None:
                self._log_event(
                    "stick_monitor",
                    ok=False,
                    path=path,
                    error=repr(exc),
                )
            return False
        self._fd = fd
        self.healthy = True
        self.disconnect_reason = ""
        self.resolved_path = path
        self.device_name = _joystick_device_name(path) or path
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="sc-stick-monitor",
            daemon=True,
        )
        self._thread.start()
        if self._log_event is not None:
            self._log_event(
                "stick_monitor",
                ok=True,
                path=path,
                name=self.device_name,
                deadzone=self.deadzone,
                note="stick_deflection_forces_skycontroller",
            )
        return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def snapshot_axes(self) -> dict[int, int]:
        with self._lock:
            return dict(self._axes)

    def is_active(self) -> bool:
        return axes_active(self.snapshot_axes(), deadzone=self.deadzone)

    def _loop(self) -> None:
        fd = self._fd
        if fd is None:
            return
        while not self._stop.is_set():
            try:
                data = os.read(fd, _JS_EVENT_SIZE * 16)
            except BlockingIOError:
                data = None
            except OSError as exc:
                self._report_disconnect(repr(exc))
                break
            if data == b"":
                self._report_disconnect("joystick EOF")
                break
            if data is not None:
                offset = 0
                n = len(data)
                while offset + _JS_EVENT_SIZE <= n:
                    _t, value, typ, number = struct.unpack_from(
                        _JS_EVENT_FMT, data, offset
                    )
                    offset += _JS_EVENT_SIZE
                    kind = typ & ~_JS_EVENT_INIT
                    if kind != _JS_EVENT_AXIS:
                        continue
                    with self._lock:
                        self._axes[int(number)] = int(value)
                if axes_active(self._axes, deadzone=self.deadzone):
                    try:
                        self._on_active(self.snapshot_axes())
                    except Exception as exc:
                        # This callback IS the stick override. Swallowing it left
                        # the pilot throwing the sticks with the PC still flying
                        # and nothing anywhere saying the takeover had failed.
                        self._callback_failures += 1
                        if self._log_event is not None:
                            self._log_event(
                                "stick_override_callback_failed",
                                error=repr(exc),
                                consecutive=self._callback_failures,
                            )
                        if self._callback_failures >= _STICK_CALLBACK_FAIL_LIMIT:
                            self._report_disconnect(
                                f"override callback failed "
                                f"{self._callback_failures}x: {exc!r}"
                            )
                    else:
                        self._callback_failures = 0
            self._stop.wait(self.poll_s)

    def _report_disconnect(self, reason: str) -> None:
        if self._stop.is_set() or not self.healthy:
            return
        self.healthy = False
        self.disconnect_reason = str(reason)
        if self._log_event is not None:
            self._log_event(
                "stick_monitor_disconnect",
                path=self.resolved_path,
                reason=self.disconnect_reason,
            )
        if self._on_disconnect is not None:
            try:
                self._on_disconnect(self.disconnect_reason)
            except Exception as exc:
                # healthy is already False, so the backend's readiness gate still
                # fails; but a silent handler failure hid why the land never came.
                if self._log_event is not None:
                    self._log_event(
                        "stick_monitor_disconnect_handler_failed", error=repr(exc)
                    )


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
            except Exception:
                pass
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None


class LiveAnafiVideoStream:
    """Latest-frame stream for OperatorApp.

    Returns a **new** frame only when the grabber stamp advances (no duplicate
    work on every UI tick). Keeps numpy RGB for the UI to convert once at
    panel size — cheaper than full-res PIL every tick.
    """

    def __init__(self, grabber):
        self._grab = grabber
        self.output_index = 0
        self.last_frame_name = "live_0"
        self.fps = 0.0
        self._last_stamp: float | None = None
        self.last_stamp: float = 0.0
        self.last_timing: dict = {}
        self._last_frame_width_px: int | None = None
        self._last_frame_height_px: int | None = None

    def next_frame(self, *, only_new: bool = True):
        peek_stamp = getattr(self._grab, "peek_stamp", None)
        if only_new and callable(peek_stamp):
            try:
                stamp_hint = peek_stamp()
            except Exception:
                stamp_hint = None
            if stamp_hint is None or (
                    self._last_stamp is not None and stamp_hint == self._last_stamp):
                return None
        try:
            sample_with_timing = getattr(self._grab, "latest_frame_with_timing", None)
            sample = (sample_with_timing()
                      if callable(sample_with_timing) else self._grab())
        except Exception:
            return None
        if sample is None:
            return None
        timing = {}
        if isinstance(sample, tuple):
            frame, stamp = sample[0], float(sample[1])
            if len(sample) >= 3 and isinstance(sample[2], dict):
                timing = dict(sample[2])
        else:
            frame, stamp = sample, time.monotonic()
        if frame is None:
            return None
        if only_new and self._last_stamp is not None and stamp == self._last_stamp:
            return None
        self._last_stamp = stamp
        self.last_stamp = stamp
        self.last_timing = timing
        # Leave as numpy HxWx3 RGB when possible (UI converts at panel size).
        try:
            import numpy as np
            if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        except Exception:
            pass
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            self._last_frame_height_px = int(shape[0])
            self._last_frame_width_px = int(shape[1])
        else:
            size = getattr(frame, "size", None)
            if isinstance(size, tuple) and len(size) == 2:
                self._last_frame_width_px = int(size[0])
                self._last_frame_height_px = int(size[1])
        self.output_index += 1
        self.last_frame_name = f"live_{self.output_index}"
        self.fps = float(getattr(self._grab, "fps", 0.0) or 0.0)
        return frame

    def metadata_snapshot(self) -> dict[str, Any]:
        raw = getattr(self._grab, "stream_metadata", {})
        try:
            raw = raw() if callable(raw) else raw
            metadata = dict(raw) if isinstance(raw, dict) else {}
        except Exception:
            metadata = {}
        metadata.update(
            {
                "ui_frame_width_px": self._last_frame_width_px,
                "ui_frame_height_px": self._last_frame_height_px,
                "observed_fps": float(self.fps),
                "frames_delivered": int(self.output_index),
            }
        )
        return metadata

    def close(self) -> None:
        try:
            self._grab.stop()
        except Exception:
            pass


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
        self.state.tracker_state = "LINK"
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
        self._firmware_limits_lock = threading.Lock()
        self._calibration_lock = threading.Lock()
        self._pulse_token = 0
        self.pilot_sticks = False
        self._cleanup_done = False
        self._landed = False
        self.flight_start: float | None = None
        # SC stick override: physical stick deflection reclaims SkyController.
        self._stick_monitor: SkyControllerStickMonitor | None = None
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
        self._nudge_vector: tuple[float, float, float, float] | None = None
        self.zero_pcmd_failures = 0
        self.last_zero_pcmd_error: str | None = None
        self._disk_low_noted = False
        self.last_zero_pcmd_call_mono_ns: int | None = None
        self._runtime_safety_action_lock = threading.Lock()
        self._runtime_safety_action_latched = False
        self._runtime_safety_action_reason = ""
        self._runtime_safety_action_thread: threading.Thread | None = None
        self._home_latitude_deg: float | None = None
        self._home_longitude_deg: float | None = None
        # Hold-to-move: keys/buttons held → continuous PCMD; all released → hover.
        self._nudge_held: set[str] = set()
        self._nudge_loop_stop = threading.Event()
        self._nudge_loop_thread: threading.Thread | None = None
        self._nudge_period_s = 0.05  # 20 Hz
        self._nudge_deadline = 0.0
        # Flight recording: arm before takeoff → start on takeoff → stop+save on land.
        self.record_on_takeoff: bool = False
        self.recording_active: bool = False
        self.record_started_iso: str = ""
        self.record_last_path: str = ""
        self.record_status: str = "錄影: 關"
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
            except Exception:
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
        self.state.tracker_state = "STICKS" if self.pilot_sticks else "HOVER"
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

    def read_connection_inventory(self) -> dict[str, Any]:
        """Read hardware, firmware, Home and lost-link state from Olympe cache."""
        errors: list[str] = []
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
            common_messages = None
            errors.append(f"aircraft inventory API unavailable: {exc!r}")
        else:
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

        states: dict[str, dict | None] = {}
        if common_messages is not None:
            states = {
                key: self._state_dict(message)
                for key, (message, _field) in common_messages.items()
            }

        version = states.get("version")
        high = self._state_value(states.get("serial_high"), "high")
        low = self._state_value(states.get("serial_low"), "low")
        aircraft = {
            "name": self._state_value(states.get("name"), "name"),
            "model": self._state_value(states.get("model"), "model"),
            "serial": (
                f"{high or ''}{low or ''}" if high is not None or low is not None else None
            ),
            "software": self._state_value(version, "software"),
            "hardware": self._state_value(version, "hardware"),
            "board_id": self._state_value(states.get("board_id"), "id"),
        }

        controller: dict[str, Any] | None = None
        if self.via_skycontroller():
            try:
                from olympe.messages.skyctrl.SettingsState import (
                    ProductSerialChanged,
                    ProductVariantChanged,
                    ProductVersionChanged as ControllerVersionChanged,
                )
            except Exception as exc:
                errors.append(f"controller inventory API unavailable: {exc!r}")
            else:
                controller_version = self._state_dict(ControllerVersionChanged)
                controller = {
                    "variant": self._state_value(
                        self._state_dict(ProductVariantChanged), "variant"
                    ),
                    "serial": self._state_value(
                        self._state_dict(ProductSerialChanged), "serialNumber"
                    ),
                    "software": self._state_value(controller_version, "software"),
                    "hardware": self._state_value(controller_version, "hardware"),
                }

        home_raw = states.get("home")
        home = {
            "latitude": self._state_value(home_raw, "latitude"),
            "longitude": self._state_value(home_raw, "longitude"),
            "altitude": self._state_value(home_raw, "altitude"),
        }
        home["valid"] = self._valid_home(
            home["latitude"], home["longitude"], home["altitude"]
        )

        rth_states: dict[str, dict | None] = {}
        try:
            from olympe.messages import rth

            rth_states = {
                "home_reachability": self._state_dict(rth.home_reachability),
                "auto_trigger_mode": self._state_dict(rth.auto_trigger_mode),
                "delay": self._state_dict(rth.delay),
                "ending_behavior": self._state_dict(rth.ending_behavior),
            }
        except Exception as exc:
            errors.append(f"RTH inventory API unavailable: {exc!r}")

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

        try:
            olympe_version = importlib.metadata.version("parrot-olympe")
        except importlib.metadata.PackageNotFoundError:
            olympe_version = None
            errors.append("parrot-olympe version unavailable")
        if olympe_version not in self.approved_olympe_versions:
            errors.append(
                f"Olympe version {olympe_version!r} lacks an approved receipt"
            )

        aircraft_name = " ".join(
            str(aircraft.get(key) or "") for key in ("name", "model")
        ).lower()
        controller_variant = str((controller or {}).get("variant") or "").lower()
        if "anafi" not in aircraft_name:
            errors.append("connected aircraft is not an approved ANAFI model")
        if not aircraft.get("serial"):
            errors.append("aircraft serial unavailable")
        if aircraft.get("software") not in self.approved_aircraft_firmware:
            errors.append(
                f"aircraft firmware {aircraft.get('software')!r} lacks an approved receipt"
            )
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
            if (controller or {}).get("software") not in self.approved_controller_firmware:
                errors.append(
                    "controller firmware "
                    f"{(controller or {}).get('software')!r} lacks an approved receipt"
                )
        if not lost_link_policy_ok:
            errors.append("lost-link auto-land policy readback is not confirmed")
        if not home_usable:
            # Not an error: GPS is present but often weak at this site. The aircraft
            # still comes down on link loss, it just lands in place instead of
            # returning home. Make that explicit rather than silently degrading.
            self.log.event(
                "lost_link_fallback",
                fallback=lost_link_fallback,
                home_valid=bool(home["valid"]),
                home_reachability=str(reachability),
                note="no usable Home Point: link loss lands in place",
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
        self.connection_inventory = inventory
        self._inventory_takeoff_ready = not errors
        self._inventory_block_reason = "; ".join(errors) if errors else "ready"
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
        battery_floor = self._finite_float(self.min_takeoff_battery_pct)
        if battery_floor is None or not 0.0 <= battery_floor <= 100.0:
            return "takeoff battery floor must be finite and in [0, 100]"
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

    def _configure_firmware_limits_locked(self) -> bool:
        """Configure desired limits on connect, but only while confirmed landed."""
        self._firmware_config_ok = False
        self._read_firmware_safety_state()

        def fail(reason: str) -> bool:
            self._firmware_config_reason = reason
            self._read_firmware_safety_state()
            self._set_preflight_status(False, reason)
            self.log.event("firmware_limits", ok=False, reason=reason)
            return False

        error = self._desired_limits_error()
        if error is not None:
            return fail(error)
        if self._flight_state_name() != "landed":
            return fail("firmware limits not written: aircraft is not confirmed landed")

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
            return fail(f"firmware settings API unavailable: {exc!r}")

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
                return fail(bounds_error)

        def apply(command, state_message, desired: float, label: str) -> bool:
            try:
                waited = self.drone(command).wait()
                ok = bool(getattr(waited, "success", lambda: False)())
            except Exception as exc:
                return fail(f"{label} command failed: {exc!r}")
            if not ok:
                return fail(f"{label} command failed or timed out")
            readback_error = self._limit_readback_error(
                self._state_dict(state_message), desired, label,
            )
            return True if readback_error is None else fail(readback_error)

        if not apply(MaxAltitude(current=altitude), MaxAltitudeChanged,
                     altitude, "MaxAltitude"):
            return False
        if not apply(MaxDistance(value=distance), MaxDistanceChanged,
                     distance, "MaxDistance"):
            return False
        # The speed envelope must be pinned and read back like the geofence limits:
        # every operator command is a percentage of these.
        if not apply(MaxTilt(current=tilt), MaxTiltChanged, tilt, "MaxTilt"):
            return False
        if not apply(MaxVerticalSpeed(current=vspeed), MaxVerticalSpeedChanged,
                     vspeed, "MaxVerticalSpeed"):
            return False
        if not apply(MaxRotationSpeed(current=rspeed), MaxRotationSpeedChanged,
                     rspeed, "MaxRotationSpeed"):
            return False

        geofence_value = int(self.desired_distance_geofence)
        try:
            waited = self.drone(NoFlyOverMaxDistance(
                shouldNotFlyOver=geofence_value,
            )).wait()
            ok = bool(getattr(waited, "success", lambda: False)())
        except Exception as exc:
            return fail(f"NoFlyOverMaxDistance command failed: {exc!r}")
        if not ok:
            return fail("NoFlyOverMaxDistance command failed or timed out")
        geofence_state = self._state_dict(NoFlyOverMaxDistanceChanged)
        actual = (
            geofence_state.get("shouldNotFlyOver")
            if geofence_state is not None else None
        )
        try:
            geofence_matches = int(actual) == geofence_value
        except (TypeError, ValueError, OverflowError):
            geofence_matches = False
        if not geofence_matches:
            return fail(
                "NoFlyOverMaxDistance readback mismatch: "
                f"requested={geofence_value} actual={actual!r}"
            )

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
            self._zero_pcmd_or_log("takeoff_abort:handoff_zero")
        ok = self._set_piloting_source("SkyController")
        with self._lock:
            # Block PCMD even if the hardware ownership readback failed.
            self.pilot_sticks = True
        if self.state.tracker_state not in {"LAND", "LANDING", "LAND_UNCONFIRMED"}:
            self.state.tracker_state = "STICKS" if ok else "SOURCE_FAIL"
        self.log.event("takeoff_source_restore", ok=ok, reason=reason)
        return ok

    def _takeoff_preflight(self, expected_epoch: int | None = None) -> bool:
        """Fail-closed gates with one landed-only config retry if needed."""
        def fail(reason: str) -> bool:
            self._set_preflight_status(False, reason)
            return False

        battery_floor = self._finite_float(self.min_takeoff_battery_pct)
        if (
            battery_floor is None
            or not MIN_TAKEOFF_BATTERY_PCT <= battery_floor <= 100.0
        ):
            return fail(
                "takeoff battery floor must be finite and within "
                f"[{MIN_TAKEOFF_BATTERY_PCT:g}, 100]"
            )
        advisories: list[str] = []
        firmware_advisories: list[str] = []
        limit_config_error = self._desired_limits_error()
        if limit_config_error is not None:
            firmware_advisories.append(limit_config_error)
        if self._flight_state_name() != "landed":
            return fail("takeoff requires confirmed landed state")
        if self.via_skycontroller() and not self._stick_monitor_ready():
            return fail("takeoff requires a healthy SkyController stick monitor")

        link_ok = self._probe_link_ok()
        self.state.link_ok = link_ok
        self.state.link_status = "OK" if link_ok else "LOST"
        if not link_ok:
            return fail("takeoff requires a healthy Olympe link")

        calibration_error = self._magnetometer_control_error()
        if calibration_error is not None:
            return fail(calibration_error)

        if not bool(getattr(self.log, "durable", getattr(self.log, "path", None))):
            return fail("takeoff requires a durable safety log")
        if not bool(getattr(self.log, "healthy", True)):
            return fail("takeoff requires a healthy safety log")
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
            return fail(disk.reason)

        if not self._configure_lost_link_policy_if_safe():
            return fail("onboard lost-link RTH policy was not confirmed")
        self.read_connection_inventory()
        if not self._inventory_takeoff_ready:
            return fail(self._inventory_block_reason)

        if self.with_video:
            stamp = self._finite_float(
                getattr(self.video_stream, "last_stamp", None)
                if self.video_stream is not None else None
            )
            age = None if stamp is None else max(0.0, time.monotonic() - stamp)
            if stamp is None or stamp <= 0.0 or age is None or age > 1.0:
                return fail("takeoff requires a live video frame newer than 1.0 s")

        try:
            from olympe.messages.common.CommonState import BatteryStateChanged
        except Exception:
            return fail("battery state API unavailable")
        battery_state = self._state_dict(BatteryStateChanged)
        battery = self._finite_float(
            battery_state.get("percent") if battery_state is not None else None
        )
        if battery is None or not 0.0 <= battery <= 100.0:
            return fail("battery state unavailable or invalid")
        self.state.battery_pct = battery
        if battery < battery_floor:
            return fail(
                f"battery {battery:.0f}% is below the "
                f"{battery_floor:.0f}% takeoff floor"
            )

        try:
            from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
        except Exception:
            self.state.gps_fixed = None
            advisories.append("GPS state API unavailable")
        else:
            gps_state = self._state_dict(GPSFixStateChanged)
            gps_fixed = gps_state.get("fixed") if gps_state is not None else None
            try:
                fixed = int(gps_fixed) == 1
            except (TypeError, ValueError, OverflowError):
                fixed = False
            self.state.gps_fixed = fixed
            if not fixed:
                advisories.append("GPS fix unavailable; takeoff remains allowed")

        if limit_config_error is None:
            altitude = float(self.desired_max_altitude_m)
            distance = float(self.desired_max_distance_m)
            if not self._firmware_config_ok:
                if expected_epoch is not None:
                    with self._lock:
                        if expected_epoch != self._pulse_token or self._cleanup_done:
                            return fail("takeoff preflight was superseded")
                if not self._configure_firmware_limits_if_safe():
                    firmware_advisories.append(
                        "firmware limit configuration retry failed: "
                        f"{self._firmware_config_reason}"
                    )

            states = self._read_firmware_safety_state()
            for limit_error in (
                self._limit_bounds_error(states.get("altitude"), altitude, "MaxAltitude"),
                self._limit_bounds_error(states.get("distance"), distance, "MaxDistance"),
                self._limit_readback_error(states.get("altitude"), altitude, "MaxAltitude"),
                self._limit_readback_error(states.get("distance"), distance, "MaxDistance"),
            ):
                if limit_error is not None:
                    firmware_advisories.append(limit_error)

            geofence = states.get("geofence")
            actual = geofence.get("shouldNotFlyOver") if geofence is not None else None
            try:
                geofence_matches = int(actual) == int(self.desired_distance_geofence)
            except (TypeError, ValueError, OverflowError):
                geofence_matches = False
            if not geofence_matches:
                firmware_advisories.append(
                    "NoFlyOverMaxDistance readback mismatch: "
                    f"requested={int(self.desired_distance_geofence)} actual={actual!r}"
                )

        advisories.extend(firmware_advisories)
        if advisories:
            self.log.event(
                "takeoff_advisory",
                warnings=tuple(dict.fromkeys(advisories)),
                note="GPS and firmware height/distance limits do not block takeoff",
            )

        if expected_epoch is not None:
            with self._lock:
                if expected_epoch != self._pulse_token or self._cleanup_done:
                    return fail("takeoff preflight was superseded")
        if not self._set_piloting_source("Controller"):
            self._restore_skycontroller_after_takeoff_abort(
                "controller_handoff_unconfirmed",
            )
            return fail("Controller piloting source command was not confirmed")
        if not self._controller_source_confirmed():
            self._restore_skycontroller_after_takeoff_abort(
                "controller_readback_unconfirmed",
            )
            return fail("Controller piloting source readback is unconfirmed")

        if not firmware_advisories:
            self._firmware_config_ok = True
            self._firmware_config_reason = "firmware limits confirmed"
        status = "ready" if not advisories else "ready with advisory warnings"
        self._set_preflight_status(True, status)
        return True

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
        self.state.last_pcmd_call_mono_ns = call_mono_ns
        return call_mono_ns

    def send_pcmd(self, roll: int, pitch: int, yaw: int, gaz: int, *, reason: str) -> bool:
        with self._lock:
            if self._cleanup_done or self._landed:
                return False
            if self.pilot_sticks:
                self.log.event("pcmd_blocked_manual", reason=reason,
                               pcmd=(roll, pitch, yaw, gaz))
                return False
            maneuver = self._maneuver_in_progress
            if maneuver is not None and (roll, pitch, yaw, gaz) != (0, 0, 0, 0):
                # An automated TakeOff/Landing is executing. Zero still gets through
                # (it only reinforces hover); motion must not fight the manoeuvre.
                self.log.event("pcmd_blocked_maneuver", reason=reason,
                               maneuver=maneuver, pcmd=(roll, pitch, yaw, gaz))
                return False
            pcmd_call_mono_ns = self._raw_pcmd(roll, pitch, yaw, gaz)
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
        self.nudge_clear(reason=reason if reason != "manual" else "manual")
        with self._lock:
            # Block all later PCMD before attempting the source handoff.
            self.pilot_sticks = True
            self._pulse_token += 1
            self._zero_pcmd_or_log(f"{reason}:manual_handoff_zero")
        if self.via_skycontroller():
            ok = self._set_piloting_source("SkyController")
            if ok:
                detail = "PC silent; SkyController sticks active"
                self.state.tracker_state = "STICKS"
            else:
                detail = "PC silent; SkyController source unconfirmed"
                self.state.tracker_state = "SOURCE_FAIL"
        else:
            # No sticks on the link: Esc freezes PC PCMD (hover) until 恢復電腦控制.
            ok = True
            detail = "PC PCMD frozen (direct WiFi; no SC sticks)"
            self.state.tracker_state = "PC_FROZEN"
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
            self.state.tracker_state = "STICK_MONITOR_FAIL"
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
        self.state.tracker_state = "STICK_MONITOR_FAIL"
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
        """Explicitly request PC authority; remain PCMD-blocked unless confirmed."""
        if self.via_skycontroller() and not self._stick_monitor_ready():
            with self._lock:
                self.pilot_sticks = True
            self.state.tracker_state = "STICK_MONITOR_FAIL"
            self.log.event(
                "pc_control", ok=False, note="stick_monitor_unavailable"
            )
            return False
        # If sticks are already deflected, refuse PC grab — pilot has priority.
        mon = self._stick_monitor
        if mon is not None and mon.is_active():
            with self._lock:
                self.pilot_sticks = True
            self.state.tracker_state = "STICKS"
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
            self.state.tracker_state = "SOURCE_FAIL"
            self.log.event("pc_control", ok=False, note="source_unconfirmed")
            return False
        # Re-sample the sticks AFTER the blocking handoff: the pilot may have grabbed
        # them during it, and the pre-handoff sample is by then up to 10 s stale.
        mon = self._stick_monitor
        if mon is not None and mon.is_active():
            with self._lock:
                self.pilot_sticks = True
            self.state.tracker_state = "STICKS"
            self.log.event("pc_control", ok=False, note="sticks_active_during_handoff")
            self._set_piloting_source("SkyController")
            return False
        with self._lock:
            if self._cleanup_done or self.drone is None:
                self.pilot_sticks = True
                self.log.event("pc_control", ok=False, note="cleanup_or_disconnected")
                return False
            if control_epoch != self._pulse_token:
                self.pilot_sticks = True
                self.state.tracker_state = "SOURCE_FAIL"
                self.log.event(
                    "pc_control", ok=False, note="safety_latched_during_handoff",
                )
                return False
            self.pilot_sticks = False
        if not self._landed and not self.send_pcmd(
                0, 0, 0, 0, reason="pc_control_resumed"):
            with self._lock:
                self.pilot_sticks = True
            self.state.tracker_state = "SOURCE_FAIL"
            self.log.event("pc_control", ok=False, note="zero_pcmd_failed")
            return False
        self.state.mode = "MANUAL"
        self.state.tracker_state = "PC"
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
        self.state.tracker_state = "HOVER"
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
            self.state.tracker_state = "LANDING"
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
            self.state.tracker_state = "LAND_UNCONFIRMED"
            return False

        if ok:
            with self._lock:
                self._landed = True
            self.state.tracker_state = "LAND"
        else:
            self.state.tracker_state = "LAND_UNCONFIRMED"
        self.log.event(
            "land_cmd", reason=reason, ok=ok,
            note="touchdown_confirmed" if ok else "touchdown_unconfirmed",
        )
        return ok

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
            self.state.tracker_state = "LAND"
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
        self._clear_maneuver()
        return landed

    def _clear_maneuver(self) -> None:
        with self._lock:
            self._maneuver_in_progress = None

    def takeoff_cmd(self) -> bool:
        # 【起飛 — 僅操作員親手按起飛／自動飛行 UI】禁止腳本／AI 代理人／任何 LLM 代為呼叫。
        if self.drone is None or self._cleanup_done:
            return False
        self.nudge_clear(reason="takeoff")
        with self._lock:
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
            self._clear_maneuver()
            return False
        if not preflight_ok:
            self.state.tracker_state = "TAKEOFF_BLOCKED"
            self._clear_maneuver()
            self.log.event(
                "takeoff", ok=False, note="preflight_blocked",
                reason=self.state.preflight_reason,
            )
            return False
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
                self._restore_skycontroller_after_takeoff_abort(
                    "hover_unconfirmed",
                )
                self.log.event("takeoff", ok=False, note="hover_unconfirmed")
                self.state.tracker_state = "TAKEOFF_FAIL"
                return False

            self.log.event("takeoff", ok=True)
            self.state.tracker_state = "HOVER"
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
            self._restore_skycontroller_after_takeoff_abort("takeoff_exception")
            self.log.event("takeoff", ok=False, error=repr(exc))
            with self._lock:
                current = (
                    takeoff_epoch == self._pulse_token
                    and not self._cleanup_done
                    and not self._landed
                )
            if current:
                self.state.tracker_state = "TAKEOFF_FAIL"
            return False
        finally:
            self._clear_maneuver()

    # -------------------------------------------------------------- flight recording
    def set_record_on_takeoff(self, enabled: bool) -> None:
        """Arm/disarm: if armed, next successful takeoff starts onboard recording."""
        self.record_on_takeoff = bool(enabled)
        if self.recording_active:
            self.record_status = "錄影: 錄影中 ●"
        elif self.record_on_takeoff:
            self.record_status = "錄影: 待命（起飛後自動開始）"
        else:
            self.record_status = "錄影: 關"
        self.log.event("record_arm", enabled=self.record_on_takeoff,
                       status=self.record_status)

    def start_flight_recording(self, reason: str = "start") -> bool:
        """Start onboard camera recording (ANAFI cam_id=0)."""
        if self.drone is None:
            self.record_status = "錄影: 失敗（無連線）"
            self.log.event("record_start", ok=False, reason=reason, error="no_drone")
            return False
        if self.recording_active:
            self.log.event("record_start", ok=True, reason=reason, note="already_active")
            return True
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
                self.record_status = "錄影: 錄影中 ●"
            else:
                self.record_status = "錄影: 啟動失敗"
            self.log.event("record_start", ok=ok, reason=reason,
                           status=self.record_status)
            return ok
        except Exception as exc:
            self.recording_active = False
            self.record_status = f"錄影: 失敗 ({exc!r})"
            self.log.event("record_start", ok=False, reason=reason, error=repr(exc))
            return False

    def stop_flight_recording(
            self, reason: str = "stop", *, download: bool = True) -> bool:
        """Finalize onboard recording; optionally perform a bounded PC download."""
        if self.drone is None:
            self.recording_active = False
            self.record_status = "錄影: 已停（無連線）"
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
            except Exception:
                pass
            self.log.event("record_stop", ok=ok_stop, reason=reason)
        except Exception as exc:
            self.log.event("record_stop", ok=False, reason=reason, error=repr(exc))
        self.recording_active = False

        if not download:
            self.record_status = "錄影: 已停止（保留於機載媒體）"
            self.log.event(
                "record_saved", path="", reason=reason,
                note="onboard_only_no_synchronous_download",
            )
            return ok_stop

        # Download last media to PC if media API is available (best-effort).
        path = self._try_download_last_recording()
        if path:
            self.record_last_path = path
            self.record_status = f"錄影: 已存檔 {Path(path).name}"
            self.log.event("record_saved", path=path, reason=reason)
        else:
            self.record_status = "錄影: 已停止（檔在機載媒體；下載略過/失敗）"
            self.log.event("record_saved", path="", reason=reason,
                           note="onboard_only_or_download_failed")
        return ok_stop

    def _try_download_last_recording(self) -> str:
        """Best-effort: download newest video resource into record_dir."""
        if self.drone is None:
            return ""
        try:
            media = getattr(self.drone, "media", None)
            if media is None:
                return ""
            try:
                media.download_dir = str(self.record_dir)
            except Exception:
                pass
            # Allow media indexing a moment after stop_recording.
            time.sleep(1.0)
            try:
                media.wait_for_pending_downloads(timeout=_MEDIA_PENDING_TIMEOUT_S)
            except Exception:
                pass
            # Prefer explicit download of last media if API exposes it.
            last_id = getattr(media, "last_media_id", None)
            mid = None
            try:
                mid = last_id() if callable(last_id) else last_id
            except Exception:
                mid = None
            if mid:
                try:
                    from olympe.features.media import download_media
                    waited = media(download_media(mid)).wait(
                        _timeout=_MEDIA_DOWNLOAD_TIMEOUT_S,
                    )
                    if not bool(getattr(waited, "success", lambda: False)()):
                        raise TimeoutError("media download failed or timed out")
                    media.wait_for_pending_downloads(
                        timeout=_MEDIA_PENDING_TIMEOUT_S,
                    )
                except Exception as exc:
                    self.log.event("record_download", ok=False, media_id=str(mid),
                                   error=repr(exc))
            # Pick newest file in download dir
            candidates = sorted(
                list(self.record_dir.glob("*.mp4"))
                + list(self.record_dir.glob("*.MP4"))
                + list(self.record_dir.glob("*.mov"))
                + list(self.record_dir.glob("*.MOV")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
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
        if self._cleanup_done or self._landed or self.drone is None:
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
            self._nudge_vector = tuple(axes)
            self._nudge_deadline = time.monotonic() + self.nudge_pulse_s
            self._nudge_held.add(_VECTOR_HOLD)
        self._nudge_loop_stop.clear()
        self.state.tracker_state = "NUDGE"
        self._ensure_nudge_loop()
        return True

    def clear_nudge_vector(self) -> None:
        with self._lock:
            self._nudge_vector = None
            self._nudge_held.discard(_VECTOR_HOLD)
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
                self.state.tracker_state = "HOVER"

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

    def _ensure_nudge_loop(self) -> None:
        """Start background loop that streams PCMD while any key/button is held."""
        if self._nudge_loop_thread is not None and self._nudge_loop_thread.is_alive():
            return
        self._nudge_loop_stop.clear()

        def _loop() -> None:
            deadman_expired = False
            loop_error: BaseException | None = None
            try:
                while not self._nudge_loop_stop.is_set():
                    if self._cleanup_done or self._landed or self.pilot_sticks:
                        break
                    with self._lock:
                        held = list(self._nudge_held)
                        if held and time.monotonic() >= self._nudge_deadline:
                            self._nudge_held.clear()
                            held = []
                            deadman_expired = True
                    if not held:
                        break
                    pcmd = self._combined_nudge_pcmd()
                    self.send_pcmd(*pcmd, reason="nudge_hold:" + "+".join(sorted(held)))
                    self.state.tracker_state = "NUDGE"
                    self._nudge_loop_stop.wait(self._nudge_period_s)
            except Exception as exc:
                # The deadman timer lives INSIDE this loop. If the loop dies the
                # hold would never expire and the aircraft would keep flying on
                # its last non-zero PCMD until the operator noticed. Drop the
                # hold so the zero below runs, and never die silently.
                loop_error = exc
                with self._lock:
                    self._nudge_held.clear()
            finally:
                # Loop exit: zero if nothing held (release → hover)
                with self._lock:
                    empty = not self._nudge_held
                    owns_thread = self._nudge_loop_thread is threading.current_thread()
                    # Atomically retire this loop before deciding whether a new
                    # hold arrived during its final iteration. Without this
                    # handoff, _ensure_nudge_loop() sees the dying thread as
                    # alive and leaves a held nudge without a deadman loop.
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
                    self.state.tracker_state = "HOVER"
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
                if owns_thread and held_for_handoff and not self._cleanup_done:
                    self._nudge_loop_stop.clear()
                    self._ensure_nudge_loop()

        self._nudge_loop_thread = threading.Thread(
            target=_loop, name="nudge-hold-loop", daemon=True)
        self._nudge_loop_thread.start()

    def nudge_begin(self, name: str) -> bool:
        """Key/button pressed: hold this direction until nudge_end."""
        if name not in NUDGE_DIRS:
            self.log.event("unknown_nudge", name=name)
            return False
        if self.pilot_sticks:
            self.log.event("nudge_blocked_manual", name=name)
            return False
        if self._cleanup_done or self._landed or self.drone is None:
            self.log.event("nudge_blocked", name=name, note="not_available")
            return False
        maneuver = self._maneuver_in_progress
        if maneuver is not None:
            self.log.event("nudge_blocked", name=name, note=f"maneuver:{maneuver}")
            return False
        self._nudge_loop_stop.clear()
        with self._lock:
            already = name in self._nudge_held
            self._nudge_held.add(name)
            if not already:
                self._nudge_deadline = time.monotonic() + self.nudge_pulse_s
        if already:
            return True
        self.log.event("nudge_begin", name=name, held=sorted(self._nudge_held),
                       pcmd=self._combined_nudge_pcmd())
        self.state.last_command = f"nudge_begin:{name}"
        self.state.tracker_state = "NUDGE"
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
                    self._cleanup_done or self._landed or self.pilot_sticks):
                self._nudge_deadline = time.monotonic() + self.nudge_pulse_s
                return True
        self.log.event(
            "nudge_heartbeat_rejected",
            requested=sorted(requested), held=sorted(held),
        )
        return False

    def nudge_end(self, name: str) -> None:
        """Key/button released: drop this direction; hover if none left."""
        with self._lock:
            self._nudge_held.discard(name)
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
                self.state.tracker_state = "HOVER"
        else:
            # Recompute loop continues with remaining holds.
            self._ensure_nudge_loop()

    def nudge_clear(self, reason: str = "clear") -> None:
        """Release all held directions AND the stick vector; stop the hold loop.

        The vector is cleared under the same lock the hold loop reads it under:
        an unlocked clear could land between that read and its send, letting one
        more scaled PCMD go out after the release.
        """
        with self._lock:
            self._nudge_vector = None
            self._nudge_held.clear()
            self._nudge_deadline = 0.0
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

    def _typed_command(self, request: ControlRequest) -> ControlResult:
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
        if request.action is ControlAction.LAND_NOW:
            raw = self.land_cmd("ui_land_now")
            return ControlResult.completed(self.state, raw_result=raw)
        if request.action is ControlAction.TAKEOFF:
            raw = self.takeoff_cmd()
            return ControlResult.completed(self.state, raw_result=raw)
        if request.action is ControlAction.START_LOCALIZATION:
            return ControlResult.completed(self.state, raw_result=True)
        name, payload = request.legacy_call()
        raw = self.command(name, **payload)
        return ControlResult.completed(self.state, raw_result=raw)

    def command(self, name: str | ControlRequest, **payload) -> Any:
        if isinstance(name, ControlRequest):
            self.log.event(
                "control_request",
                request_id=name.request_id,
                action=name.action.value,
                human_origin=name.human_origin,
                submitted_mono_ns=name.submitted_mono_ns,
            )
            result = self._typed_command(name)
            self.log.event(
                "control_result",
                request_id=name.request_id,
                action=name.action.value,
                accepted=result.accepted,
                executed=result.executed,
                reason_code=result.reason_code,
                control_owner=getattr(self.state, "control_owner", None),
            )
            return result
        if name == "takeoff":
            # The compatibility shim intentionally excludes TakeOff. A typed
            # request is the only path that carries the human-origin bit.
            self.log.event("legacy_takeoff_rejected", reason="typed_request_required")
            return ControlResult.rejected("TYPED_TAKEOFF_REQUIRED", self.state)
        self.state.last_command = name
        if name == "manual":
            self.give_to_pilot()
        elif name in {"pc_control", "resume_pc", "恢復電腦控制"}:
            if self.take_pc_control():
                self.state.last_command = "pc_control"
        elif name == "hover":
            self.hover_cmd("ui_hover")
        elif name == "land":
            self.land_cmd("ui_land")
        elif name == "emergency_stop":
            self.fail_safe(FailureReason.EMERGENCY_STOP)
            return True
        elif name == "land_now":
            return self.land_cmd("ui_land_now")
        elif name == "firmware_limits_apply":
            return self.apply_firmware_limits(
                payload.get("max_altitude_m"),
                payload.get("max_distance_m"),
                payload.get("distance_geofence", True),
            )
        elif name == "auto_speed_limit_apply":
            return self.apply_autonomous_speed_limit(
                payload.get("speed_limit_mps")
            )
        elif name == "drone_magnetometer_start":
            return self.start_drone_magnetometer_calibration()
        elif name == "drone_magnetometer_cancel":
            return self.cancel_drone_magnetometer_calibration()
        elif name == "skycontroller_magnetometer_start":
            return self.start_skycontroller_magnetometer_calibration()
        elif name == "skycontroller_magnetometer_cancel":
            return self.cancel_skycontroller_magnetometer_calibration()
        elif name in {"auto", "start_auto"}:
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
            if self.take_pc_control():
                self.state.mode = "PC_CONTROL"
                self.state.tracker_state = "LOCALIZATION_ONLY"
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
        elif name == "boot_lock":
            self.state.tracker_state = "BOOT_INIT"
        elif name in {"pause", "resume"}:
            self.hover_cmd(f"ui_{name}")
            self.log.event(name)
        elif name == "gimbal_pitch":
            self.set_gimbal_pitch(float(payload.get("pitch", self.state.gimbal_pitch_deg)))
        elif name == "zoom":
            self.set_zoom(float(payload.get("zoom", self.state.zoom)))
        elif name in {"camera_reset", "reset_camera", "鏡頭預設", "回復預設"}:
            self.reset_camera_defaults(
                pitch=float(payload.get("pitch", -20.0)),
                zoom=float(payload.get("zoom", 1.0)),
            )
        elif name in {"record_arm", "record_on_takeoff"}:
            en = payload.get("enabled", payload.get("value", True))
            if isinstance(en, str):
                en = en.lower() in {"1", "true", "yes", "on"}
            self.set_record_on_takeoff(bool(en))
        elif name in {"record_disarm"}:
            self.set_record_on_takeoff(False)
        elif name in {"record_start"}:
            self.start_flight_recording(reason="ui_manual")
        elif name in {"record_stop"}:
            self.stop_flight_recording(reason="ui_manual")
        elif name in {"nudge_begin", "nudge_press"}:
            d = str(payload.get("dir") or payload.get("name") or "")
            self.nudge_begin(d)
        elif name in {"nudge_end", "nudge_release"}:
            d = str(payload.get("dir") or payload.get("name") or "")
            self.nudge_end(d)
        elif name == "nudge_heartbeat":
            dirs = payload.get("dirs", payload.get("directions"))
            self.nudge_heartbeat(dirs)
        elif name == "nudge_vector":
            return self.set_nudge_vector(
                float(payload.get("roll", 0.0)), float(payload.get("pitch", 0.0)),
                float(payload.get("yaw", 0.0)), float(payload.get("gaz", 0.0)),
            )
        elif name in {"nudge_clear"}:
            self.nudge_clear(reason="ui")
            self.hover_cmd("nudge_clear")
        elif name.startswith("nudge_") or name in NUDGE_DIRS:
            # Legacy one-shot name: treat as begin (UI should send begin/end).
            n = name[6:] if name.startswith("nudge_") else name
            self.nudge_begin(n)
        else:
            self.log.event("unhandled_command", name=name, payload=payload)
        return self.state

    def apply_autonomous_speed_limit(self, value: Any) -> bool:
        parsed = self._finite_float(value)
        if parsed is None:
            self.log.event("auto_speed_limit", ok=False, reason="INVALID_SPEED_LIMIT")
            return False
        change = validate_speed_limit_change(
            float(self.state.autonomous_speed_limit_mps),
            parsed,
            landed=self._flight_state_name() == "landed",
        )
        if not change.accepted:
            self.log.event("auto_speed_limit", ok=False, reason=change.reason)
            return False
        self.state.autonomous_speed_limit_mps = change.new_speed_limit_mps
        if change.approval_invalidated:
            self.state.autonomous_approval_valid = False
            self.state.autonomous_locked = True
        self.log.event(
            "auto_speed_limit",
            ok=True,
            old_mps=change.old_speed_limit_mps,
            new_mps=change.new_speed_limit_mps,
            approval_invalidated=change.approval_invalidated,
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
            "EMERGENCY_MANUAL"
            if reason is FailureReason.EMERGENCY_STOP
            else "FAIL_SAFE_MANUAL"
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
        result = self.fail_safe(FailureReason.STREAM_STALE)
        pipeline = None
        if self.grabber is not None:
            try:
                pipeline = self.grabber.frame_pipeline_stats
            except Exception:
                pipeline = None
        self.log.event(
            "stream_lost",
            detail=detail,
            hover_sent=bool(result.executed),
            pipeline=pipeline,
        )
        self.state.stream = "LOST"
        self.state.loc = "STREAM_LOST"
        self.state.tracker_state = "STREAM_LOST_MANUAL"
        self.state.last_command = f"stream_lost_manual: {detail}"
        return self.state

    def _evaluate_video_health(self, now: float | None = None) -> bool:
        if self.video_stream is None:
            return False
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
        if age_s is None:
            return False
        age_s = max(0.0, age_s)
        now_mono = time.monotonic()
        if not frozen and age_s < _STREAM_STALE_S:
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
            return False
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
        self.state.tracker_state = "RTH"
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
        # Operator policy (2026-08-06): whenever nothing can command the aircraft
        # any more, GPS decides the recovery -- usable Home Point means return home,
        # otherwise land in place. This applies to a lost controller too: the
        # previous code excluded CONTROLLER_DISCONNECTED and always landed where it
        # was, which is the worse outcome when the takeoff point is known.
        #
        # ...but RTH must never break the operator's own altitude limit. The
        # aircraft climbs to return_home_min_altitude before flying home, and that
        # firmware value is independent of MaxAltitude: a 39 m RTH climb under a 3 m
        # limit is not a recovery, it is a new emergency. Land in place instead.
        rth_blocked = self._rth_altitude_conflict()
        if reason == FailureReason.INVALID_TELEMETRY.value:
            rth_blocked = "GPS/geofence telemetry unavailable; land in place"
        if rth_blocked is not None:
            self.log.event(
                "runtime_safety_rth_skipped", reason=reason, detail=rth_blocked,
            )
        if rth_blocked is None and self._refresh_home_state() and self._request_rth(reason):
            self.log.event("runtime_safety", reason=reason, action="rth")
            return
        landed = self.land_cmd(reason=f"runtime_safety:{reason}")
        self.log.event(
            "runtime_safety",
            reason=reason,
            action="land",
            ok=landed,
        )
        if not landed:
            # Neither RTH nor Landing succeeded and the aircraft is still airborne.
            # The latch exists to stop one successful action being re-issued; leaving
            # it set after a FAILED action would disable battery/altitude/distance
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
            # battery/altitude/distance for the remainder of the flight.
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
            self.state.tracker_state = "SAFETY_ACTION_PENDING"
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

    def _evaluate_runtime_safety(self) -> bool:
        if (
            self._runtime_safety_action_latched
            or self._cleanup_done
            or self.drone is None
            or not bool(getattr(self.state, "link_ok", True))
            or not self._is_airborne()
        ):
            return False
        battery = self._finite_float(getattr(self.state, "battery_pct", None))
        if battery is not None and battery <= _CRITICAL_BATTERY_PCT:
            return self._schedule_runtime_safety_action(
                FailureReason.BATTERY_CRITICAL.value
            )
        altitude = self._finite_float(
            getattr(self.state, "drone_altitude_m", None)
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
        distance = self._finite_float(
            getattr(self.state, "distance_from_home_m", None)
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
        # not block takeoff or force a landing. Altitude, battery, stream, link,
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
            self._pulse_token += 1
            self.pilot_sticks = True
            self._link_was_ok = False
            self._link_failure_count = _LINK_FAILURE_CONFIRM_POLLS
        self.state.mode = "MANUAL"
        self.state.control_owner = "ONBOARD_LOST_LINK_POLICY"
        self.state.tracker_state = "LINK_LOST_ONBOARD"
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

    def _probe_link_ok(self) -> bool:
        """Best-effort Olympe connection probe (display only; no flight cmds)."""
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
        if not host_ok or not self.via_skycontroller():
            return bool(host_ok)
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

    def poll(self, now_mono_ns: int | None = None) -> Any:
        now = (
            time.monotonic()
            if now_mono_ns is None
            else float(now_mono_ns) * 1e-9
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
            try:
                from olympe.messages.ardrone3.PilotingState import (
                    AttitudeChanged,
                    AltitudeChanged,
                    FlyingStateChanged,
                    SpeedChanged,
                )
                from olympe.messages.common.CommonState import BatteryStateChanged
            except Exception:
                return self.state

            def gs(msg):
                try:
                    return self.drone.get_state(msg)
                except Exception:
                    return None

            if not link_ok:
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
                return self.state

            self._read_magnetometer_calibration_state()

            att = gs(AttitudeChanged)
            if att:
                self.state.att_roll = float(att.get("roll", 0.0) or 0.0)
                self.state.att_pitch = float(att.get("pitch", 0.0) or 0.0)
                self.state.att_yaw = float(att.get("yaw", 0.0) or 0.0)
                self.sim_yaw = self.state.att_yaw
            alt = gs(AltitudeChanged)
            if alt and alt.get("altitude") is not None:
                self.state.altitude_m = float(alt["altitude"])
                self.state.drone_altitude_m = float(alt["altitude"])
            speed = gs(SpeedChanged)
            if speed:
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
            bat = gs(BatteryStateChanged)
            if bat and bat.get("percent") is not None:
                self.state.battery_pct = float(bat["percent"])
            fly = gs(FlyingStateChanged)
            if fly and fly.get("state") is not None:
                st = str(self._state_value(fly, "state"))
                self.state.flight_state = st
                if self.state.tracker_state not in _TRACKER_STATE_STICKY:
                    self.state.tracker_state = st.upper()
            try:
                from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
                gps = gs(GPSFixStateChanged)
                if gps is not None and gps.get("fixed") is not None:
                    self.state.gps_fixed = bool(int(gps.get("fixed") or 0))
            except Exception:
                pass
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

                agl = gs(AltitudeAboveGroundChanged)
                if agl and agl.get("altitude") is not None:
                    self.state.agl_altitude_m = self._finite_float(agl.get("altitude"))
                location = gs(GpsLocationChanged)
                if location:
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
                    self.state.gps_altitude_m = self._finite_float(
                        location.get("altitude")
                    )
                    for source, target in (
                        ("latitude_accuracy", "gps_latitude_accuracy_m"),
                        ("longitude_accuracy", "gps_longitude_accuracy_m"),
                        ("altitude_accuracy", "gps_altitude_accuracy_m"),
                    ):
                        accuracy = self._finite_float(location.get(source))
                        setattr(self.state, target, accuracy if accuracy is not None and accuracy >= 0 else None)
                alert = gs(AlertStateChanged)
                if alert and alert.get("state") is not None:
                    self.state.alert_state = str(self._state_value(alert, "state"))
                navigate_home = gs(NavigateHomeStateChanged)
                if navigate_home:
                    if navigate_home.get("state") is not None:
                        self.state.navigate_home_state = str(
                            self._state_value(navigate_home, "state")
                        )
                    if navigate_home.get("reason") is not None:
                        self.state.navigate_home_reason = str(
                            self._state_value(navigate_home, "reason")
                        )
                heading = gs(HeadingLockedStateChanged)
                if heading and heading.get("state") is not None:
                    self.state.heading_state = str(self._state_value(heading, "state"))
                hover_warning = gs(HoveringWarning)
                if hover_warning:
                    if hover_warning.get("no_gps_too_dark") is not None:
                        self.state.hover_no_gps_too_dark = bool(
                            int(hover_warning.get("no_gps_too_dark") or 0)
                        )
                    if hover_warning.get("no_gps_too_high") is not None:
                        self.state.hover_no_gps_too_high = bool(
                            int(hover_warning.get("no_gps_too_high") or 0)
                        )
                wind = gs(WindStateChanged)
                if wind and wind.get("state") is not None:
                    self.state.wind_state = str(self._state_value(wind, "state"))
                vibration = gs(VibrationLevelChanged)
                if vibration and vibration.get("state") is not None:
                    self.state.vibration_state = str(
                        self._state_value(vibration, "state")
                    )
            except Exception:
                # These read-only events are optional across Olympe/product
                # versions; mandatory battery/attitude/altitude remain visible.
                pass
            try:
                from olympe.messages.ardrone3.GPSState import NumberOfSatelliteChanged

                satellites = gs(NumberOfSatelliteChanged)
                if satellites and satellites.get("numberOfSatellite") is not None:
                    number = int(satellites["numberOfSatellite"])
                    self.state.gps_satellites = number if number >= 0 else None
            except Exception:
                pass
            try:
                from olympe.messages.common.CommonState import (
                    LinkSignalQuality,
                    SensorsStatesListChanged,
                    WifiSignalChanged,
                )

                wifi = gs(WifiSignalChanged)
                if wifi and wifi.get("rssi") is not None:
                    rssi = self._finite_float(wifi.get("rssi"))
                    self.state.wifi_rssi_dbm = None if rssi is None else int(rssi)
                link_quality = gs(LinkSignalQuality)
                if link_quality and link_quality.get("value") is not None:
                    quality = int(link_quality["value"])
                    self.state.link_signal_quality_raw = (
                        quality if 0 <= quality <= 255 else None
                    )
                sensor_updates = self._sensor_state_updates(gs(SensorsStatesListChanged))
                if sensor_updates:
                    current = dict(getattr(self.state, "sensor_states", {}) or {})
                    current.update(sensor_updates)
                    self.state.sensor_states = current
            except Exception:
                pass
            # Olympe get_state reads the local event cache; these values shown
            # in the UI never come from desired constructor settings.
            self._read_firmware_safety_state()
            if self.session_logs is not None:
                last_session_tel = getattr(self, "_last_session_telemetry_t", 0.0)
                if now - last_session_tel >= 1.0:
                    self._last_session_telemetry_t = now
                    self.session_logs.telemetry(
                        "readback",
                        battery_pct=getattr(self.state, "battery_pct", None),
                        gps_fixed=getattr(self.state, "gps_fixed", None),
                        home_valid=getattr(self.state, "home_valid", None),
                        home_reachable=getattr(
                            self.state, "home_reachable", None
                        ),
                        distance_from_home_m=getattr(
                            self.state, "distance_from_home_m", None
                        ),
                        rth_policy_valid=getattr(self.state, "rth_policy_valid", None),
                        rth_policy_configured=getattr(
                            self.state, "rth_policy_configured", None
                        ),
                        stick_monitor_ok=getattr(
                            self.state, "stick_monitor_ok", None
                        ),
                        active_incident=getattr(
                            self.state, "active_incident", None
                        ),
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
            try:
                from olympe.messages.gimbal import attitude as gimbal_attitude
                g = gs(gimbal_attitude)
                if isinstance(g, dict) and g:
                    first = next(iter(g.values())) if g else None
                    if isinstance(first, dict) and first.get("pitch_absolute") is not None:
                        self.state.gimbal_pitch_deg = float(first["pitch_absolute"])
            except Exception:
                pass
            self.state.telemetry_read_mono_ns = time.monotonic_ns()
            last_pcmd_ns = getattr(self.state, "last_pcmd_call_mono_ns", None)
            if last_pcmd_ns is not None:
                self.state.pcmd_to_telemetry_poll_ms = max(
                    0.0,
                    (self.state.telemetry_read_mono_ns - int(last_pcmd_ns))
                    / 1_000_000.0,
                )
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
        self._evaluate_runtime_safety()
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

    def cleanup(self) -> None:
        """Idempotent: zero PCMD + FORCE Landing + media cleanup + disconnect.

        Called on: UI window close, Ctrl-C / SIGTERM / SIGHUP, process atexit.

        【強制降落 — 禁止刪除或弱化】關窗／Ctrl+C 時：只要未落地就必須原地 Landing，
        即使曾按 Esc 交回搖桿。弱化此邏輯可能導致空中失控。見 mission/SAFETY.md。
        """
        with self._lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True
            self._pulse_token += 1

        def _cleanup_log(event: str, **fields: Any) -> None:
            # Cleanup logging is diagnostic only. A broken sink must never
            # prevent the bounded Landing attempt below (or final disconnect).
            try:
                self.log.event(event, **fields)
            except Exception:
                pass

        try:
            try:
                self._stop_stick_monitor()
            except Exception as exc:
                _cleanup_log("cleanup_stick_monitor_error", error=repr(exc))
            _cleanup_log("cleanup_begin", reason="exit_or_signal")
            try:
                self.nudge_clear(reason="cleanup")
            except Exception as exc:
                # Nudge bookkeeping is best effort; it must never prevent the
                # bounded Landing attempt below.
                _cleanup_log("cleanup_pre_action_error", action="nudge_clear",
                             error=repr(exc))
            landed = False
            if self.drone is not None:
                # Zero immediately, even before a potentially blocking source
                # acknowledgement. A second zero follows a confirmed handoff.
                try:
                    with self._lock:
                        self._zero_pcmd_or_log("cleanup:pre_handoff_zero")
                except Exception as exc:
                    _cleanup_log("cleanup_pre_action_error",
                                 action="pre_handoff_zero", error=repr(exc))
                try:
                    source_ok = self._set_piloting_source("Controller")
                except Exception as exc:
                    source_ok = False
                    _cleanup_log("cleanup_pre_action_error",
                                 action="piloting_source", error=repr(exc))
                if source_ok:
                    try:
                        with self._lock:
                            self.pilot_sticks = False
                            self._zero_pcmd_or_log("cleanup:post_handoff_zero")
                    except Exception as exc:
                        _cleanup_log("cleanup_pre_action_error",
                                     action="post_handoff_zero", error=repr(exc))
                else:
                    _cleanup_log(
                        "land_source_unconfirmed", reason="cleanup",
                        note="Landing still attempted as fail-safe",
                    )

                try:
                    already_landed = self._is_already_landed()
                except Exception as exc:
                    already_landed = False
                    _cleanup_log("cleanup_pre_action_error",
                                 action="landed_probe", error=repr(exc))
                if already_landed:
                    landed = True
                    self.state.tracker_state = "LAND"
                    _cleanup_log(
                        "land_cmd", reason="cleanup_exit", ok=True,
                        note="already_landed",
                    )
                else:
                    try:
                        landed = self._land_and_confirm("cleanup_force_land")
                    except Exception as exc:
                        landed = False
                        _cleanup_log("land_cmd", reason="cleanup_force_land",
                                     ok=False, error=repr(exc),
                                     note="landing_attempt_failed")

                if landed and self.via_skycontroller():
                    handed_off = self._set_piloting_source("SkyController")
                    with self._lock:
                        self.pilot_sticks = handed_off
                elif landed:
                    with self._lock:
                        self.pilot_sticks = True

            # Never let recording finalization or a media download delay the
            # zero/Landing safety path. Skip it if touchdown is unconfirmed.
            if self.recording_active and landed:
                try:
                    self.stop_flight_recording(reason="cleanup", download=False)
                except Exception as exc:
                    _cleanup_log("record_stop", ok=False, reason="cleanup",
                                 error=repr(exc))
            elif self.recording_active:
                _cleanup_log(
                    "record_stop_deferred", reason="cleanup",
                    note="touchdown_unconfirmed",
                )
        finally:
            if self.grabber is not None:
                try:
                    self.grabber.stop()
                except Exception:
                    pass
                self.grabber = None
            if self.drone is not None:
                try:
                    self.drone.disconnect()
                except Exception as exc:
                    _cleanup_log("disconnect_error", error=repr(exc))
                self.drone = None
            _cleanup_log("cleanup_done")
            try:
                self.log.close()
            except Exception:
                pass

    def close(self, reason: str) -> CloseResult:
        self.log.event("close_requested", reason=reason)
        self.cleanup()
        return CloseResult(True, "OK")
