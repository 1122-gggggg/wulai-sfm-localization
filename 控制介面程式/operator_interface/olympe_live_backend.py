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

# Local download dir for flight recordings (stop_recording finalizes on drone media).
_DEFAULT_RECORD_DIR = _WS.flight_logs / "recordings"
_MEDIA_PENDING_TIMEOUT_S = 2.0
_MEDIA_DOWNLOAD_TIMEOUT_S = 15.0


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
_STICK_POLL_S = 0.02  # 50 Hz reclaim path


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


def axes_active(axes: dict[int, int], *, deadzone: int = _STICK_DEADZONE) -> bool:
    """True when any axis exceeds deadzone (intentional stick deflection)."""
    dz = max(0, int(deadzone))
    for value in axes.values():
        try:
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
        log_event: Callable[..., None] | None = None,
    ):
        self._on_active = on_active
        self._device_path = device_path
        self.deadzone = int(deadzone)
        self.poll_s = float(poll_s)
        self._log_event = log_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._axes: dict[int, int] = {}
        self._lock = threading.Lock()
        self.resolved_path: str | None = None
        self.device_name: str | None = None

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
                data = b""
            except OSError:
                break
            if data:
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
                    except Exception:
                        pass
            self._stop.wait(self.poll_s)


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
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = path.open("w", encoding="utf-8", buffering=1)
            self.event("start", sink="olympe_live_ui")

    def event(self, event: str, **kw: Any) -> None:
        mono_ns = time.monotonic_ns()
        rec = {
            "t_iso": datetime.now(timezone.utc).isoformat(),
            "t_mono": mono_ns * 1e-9,
            "t_mono_ns": mono_ns,
            "event": event,
            **kw,
        }
        line = json.dumps(rec, ensure_ascii=False)
        print(f"[live-ui] {line}", flush=True)
        if self._f is not None:
            self._f.write(line + "\n")

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
        self.output_index += 1
        self.last_frame_name = f"live_{self.output_index}"
        self.fps = float(getattr(self._grab, "fps", 0.0) or 0.0)
        return frame

    def close(self) -> None:
        try:
            self._grab.stop()
        except Exception:
            pass


class OlympeLiveBackend:
    """Drop-in live backend for OperatorApp (same methods as DroneBackend)."""

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
        distance_geofence: bool = True,
        min_takeoff_battery_pct: float = 30.0,
        require_gps_for_geofence: bool = True,
        cmd_log: Path | None = None,
        with_video: bool = True,
    ):
        """state_factory: callable -> DroneState; anafi_profile: AnafiProfile."""
        self.ANAFI = anafi_profile
        self.state = state_factory()
        self.state.loc = "LIVE"
        self.state.stream = "CONNECTING"
        self.state.mode = "MANUAL"
        self.state.tracker_state = "LINK"
        self.ip = ip
        self.controller = controller
        self.nudge_pct = int(nudge_pct)
        self.desired_max_altitude_m = (
            None if max_altitude_m is None else float(max_altitude_m)
        )
        self.desired_max_distance_m = (
            None if max_distance_m is None else float(max_distance_m)
        )
        self.desired_distance_geofence = bool(distance_geofence)
        self.min_takeoff_battery_pct = float(min_takeoff_battery_pct)
        self.require_gps_for_geofence = bool(require_gps_for_geofence)
        self._firmware_config_ok = False
        self._firmware_config_reason = "not configured"
        self.state.preflight_ok = False
        self.state.preflight_reason = "firmware limits not checked"
        # Deadman TTL refreshed by UI heartbeats while a nudge remains held.
        requested_ttl = float(nudge_pulse_s)
        if not math.isfinite(requested_ttl):
            requested_ttl = float(NUDGE_S)
        self.nudge_pulse_s = max(0.1, requested_ttl)
        self.log = _CmdLog(cmd_log)
        self.drone = None
        self.grabber = None
        self.video_stream: LiveAnafiVideoStream | None = None
        self.with_video = bool(with_video)
        self._lock = threading.RLock()
        self._firmware_limits_lock = threading.Lock()
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
        self.start = time.monotonic()
        self.last_poll = self.start

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
        # Firmware writes are attempted only when both desired limits are set
        # and the aircraft readback says landed. Failure keeps ground UI alive
        # while takeoff remains fail-closed.
        self._configure_firmware_limits_if_safe()
        self.state.stream = "OK"
        self.state.tracker_state = "STICKS" if self.pilot_sticks else "HOVER"
        self.state.link_ok = True
        self.state.link_status = "OK"
        self.log.event("connect_ok")
        # Stick override only applies on SC USB (HID joystick present).
        if self.via_skycontroller():
            self._start_stick_monitor()

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
                self.log.event("video_fail", error=repr(exc))
                # Control still works without video.
                self.state.stream = "NO_VIDEO"

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
                NoFlyOverMaxDistance,
            )
            from olympe.messages.ardrone3.PilotingSettingsState import (
                MaxAltitudeChanged,
                MaxDistanceChanged,
                NoFlyOverMaxDistanceChanged,
            )
        except Exception as exc:
            return fail(f"firmware settings API unavailable: {exc!r}")

        altitude = float(self.desired_max_altitude_m)
        distance = float(self.desired_max_distance_m)
        states = self._read_firmware_safety_state()
        for bounds_error in (
            self._limit_bounds_error(states.get("altitude"), altitude, "MaxAltitude"),
            self._limit_bounds_error(states.get("distance"), distance, "MaxDistance"),
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
            try:
                self._raw_pcmd(0, 0, 0, 0)
            except Exception:
                pass
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

        error = self._desired_limits_error()
        if error is not None:
            return fail(error)
        if self._flight_state_name() != "landed":
            return fail("takeoff requires confirmed landed state")

        link_ok = self._probe_link_ok()
        self.state.link_ok = link_ok
        self.state.link_status = "OK" if link_ok else "LOST"
        if not link_ok:
            return fail("takeoff requires a healthy Olympe link")

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
        if battery < self.min_takeoff_battery_pct:
            return fail(
                f"battery {battery:.0f}% is below the "
                f"{self.min_takeoff_battery_pct:.0f}% takeoff floor"
            )

        if self.desired_distance_geofence and self.require_gps_for_geofence:
            try:
                from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
            except Exception:
                return fail("GPS state API unavailable")
            gps_state = self._state_dict(GPSFixStateChanged)
            gps_fixed = gps_state.get("fixed") if gps_state is not None else None
            try:
                fixed = int(gps_fixed) == 1
            except (TypeError, ValueError, OverflowError):
                fixed = False
            self.state.gps_fixed = fixed
            if not fixed:
                return fail("enabled distance geofence requires a confirmed GPS fix")

        altitude = float(self.desired_max_altitude_m)
        distance = float(self.desired_max_distance_m)
        if not self._firmware_config_ok:
            if expected_epoch is not None:
                with self._lock:
                    if expected_epoch != self._pulse_token or self._cleanup_done:
                        return fail("takeoff preflight was superseded")
            if not self._configure_firmware_limits_if_safe():
                return fail(
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
                return fail(limit_error)

        geofence = states.get("geofence")
        actual = geofence.get("shouldNotFlyOver") if geofence is not None else None
        try:
            geofence_matches = int(actual) == int(self.desired_distance_geofence)
        except (TypeError, ValueError, OverflowError):
            geofence_matches = False
        if not geofence_matches:
            return fail(
                "NoFlyOverMaxDistance readback mismatch: "
                f"requested={int(self.desired_distance_geofence)} actual={actual!r}"
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

        self._firmware_config_ok = True
        self._firmware_config_reason = "firmware limits confirmed"
        self._set_preflight_status(True, "ready")
        return True

    # ------------------------------------------------------------------ wire
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
            try:
                self._raw_pcmd(0, 0, 0, 0)
            except Exception:
                pass
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
        self.state.last_command = reason if reason else "manual"
        self.log.event("manual", detail=detail, ok=ok, reason=reason)
        return ok

    def _start_stick_monitor(self) -> None:
        self._stop_stick_monitor()
        monitor = SkyControllerStickMonitor(
            on_active=self._on_stick_active,
            log_event=self.log.event,
        )
        if monitor.start():
            self._stick_monitor = monitor
        else:
            self._stick_monitor = None

    def _stop_stick_monitor(self) -> None:
        mon = self._stick_monitor
        self._stick_monitor = None
        if mon is not None:
            mon.stop()

    def _on_stick_active(self, axes: dict[int, int]) -> None:
        """HID callback: any deliberate stick throw reclaims SkyController."""
        self._maybe_reclaim_from_sticks(axes)

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
                if abs(int(v)) > _STICK_DEADZONE
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

    def take_pc_control(self) -> bool:
        """Explicitly request PC authority; remain PCMD-blocked unless confirmed."""
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
        if not self._set_piloting_source("Controller"):
            with self._lock:
                self.pilot_sticks = True
            self.state.tracker_state = "SOURCE_FAIL"
            self.log.event("pc_control", ok=False, note="source_unconfirmed")
            return False
        with self._lock:
            if self._cleanup_done or self.drone is None:
                self.pilot_sticks = True
                self.log.event("pc_control", ok=False, note="cleanup_or_disconnected")
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
            self.state.mode = "MANUAL"
            self.state.last_command = "land"
            try:
                self._raw_pcmd(0, 0, 0, 0)
            except Exception:
                pass
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
            try:
                self._raw_pcmd(0, 0, 0, 0)
            except Exception:
                pass

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
        return landed

    def takeoff_cmd(self) -> bool:
        # 【起飛 — 僅操作員親手按 UI】禁止腳本／AI 代理人／任何 LLM 代為呼叫。
        if self.drone is None or self._cleanup_done:
            return False
        self.nudge_clear(reason="takeoff")
        with self._lock:
            # Snapshot before the blocking source acknowledgement as well as
            # the takeoff wait. Any later land/cleanup/manual invalidates it.
            self._pulse_token += 1
            takeoff_epoch = self._pulse_token
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
            return False
        if not preflight_ok:
            self.state.tracker_state = "TAKEOFF_BLOCKED"
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

    def _combined_nudge_pcmd(self) -> tuple[int, int, int, int]:
        """Sum held direction units, clamp each axis to ±1, then scale_nudge."""
        with self._lock:
            held = list(self._nudge_held)
        if not held:
            return (0, 0, 0, 0)
        acc = [0, 0, 0, 0]
        for name in held:
            u = NUDGE_DIRS.get(name)
            if u is None:
                continue
            for i in range(4):
                acc[i] += int(u[i])
        unit = tuple(max(-1, min(1, v)) for v in acc)
        return scale_nudge(unit, self.nudge_pct)  # type: ignore[arg-type]

    def _ensure_nudge_loop(self) -> None:
        """Start background loop that streams PCMD while any key/button is held."""
        if self._nudge_loop_thread is not None and self._nudge_loop_thread.is_alive():
            return
        self._nudge_loop_stop.clear()

        def _loop() -> None:
            deadman_expired = False
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
            # Loop exit: zero if nothing held (release → hover)
            with self._lock:
                empty = not self._nudge_held
            if empty and not self.pilot_sticks and not self._landed and not self._cleanup_done:
                try:
                    self._raw_pcmd(0, 0, 0, 0)
                    zero_reason = (
                        "nudge_deadman_hover" if deadman_expired
                        else "nudge_release_hover"
                    )
                    self.log.event("pcmd", reason=zero_reason, pcmd=(0, 0, 0, 0))
                except Exception:
                    pass
                self.state.tracker_state = "HOVER"
                if deadman_expired:
                    self.state.last_command = "nudge_deadman_hover"
                    self.log.event(
                        "nudge_deadman", ok=True, ttl_s=self.nudge_pulse_s,
                    )

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
            held = set(self._nudge_held)
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
                try:
                    self._raw_pcmd(0, 0, 0, 0)
                    self.log.event("pcmd", reason="nudge_release_hover", pcmd=(0, 0, 0, 0))
                except Exception:
                    pass
                self.state.tracker_state = "HOVER"
        else:
            # Recompute loop continues with remaining holds.
            self._ensure_nudge_loop()

    def nudge_clear(self, reason: str = "clear") -> None:
        """Release all held directions and stop the hold loop."""
        with self._lock:
            self._nudge_held.clear()
            self._nudge_deadline = 0.0
        self._nudge_loop_stop.set()
        self.log.event("nudge_clear", reason=reason)

    def nudge(self, name: str) -> None:
        """Backward-compat: one-shot press treated as begin (UI should use begin/end)."""
        self.nudge_begin(name)

    def set_gimbal_pitch(self, pitch_deg: float) -> None:
        if self.drone is None or self.pilot_sticks:
            return
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
        except Exception as exc:
            # Fallback older API
            try:
                from olympe.messages.ardrone3.Camera import Orientation
                self.drone(Orientation(tilt=int(pitch), pan=0))
                self.state.gimbal_pitch_deg = pitch
                self.log.event("gimbal_pitch_legacy", pitch=pitch, ok=True)
            except Exception as exc2:
                self.log.event("gimbal_pitch", ok=False, error=repr(exc2 or exc))

    def set_zoom(self, zoom: float) -> None:
        """Set digital zoom level (1.0 = default / wide)."""
        z = float(np.clip(float(zoom), 1.0, self.ANAFI.digital_zoom_max))
        self.state.zoom = z
        if self.drone is None or self.pilot_sticks:
            self.log.event("zoom", zoom=z, ok=False, note="no_drone_or_sticks")
            return
        try:
            from olympe.messages.camera import set_zoom_target
            from olympe.enums.camera import zoom_control_mode
            self.drone(set_zoom_target(
                cam_id=0,
                control_mode=zoom_control_mode.level,
                target=z,
            ))
            self.log.event("zoom", zoom=z, ok=True)
        except Exception as exc:
            # Still keep UI state even if the camera rejects the command.
            self.log.event("zoom", zoom=z, ok=False, error=repr(exc))

    def reset_camera_defaults(self, *, pitch: float = -20.0, zoom: float = 1.0) -> None:
        """Restore default look-down pitch and 1.0x zoom."""
        pitch0 = float(pitch)
        zoom0 = float(zoom)
        self.set_gimbal_pitch(pitch0)
        # Prefer dedicated reset_zoom when target is wide; else set level.
        if self.drone is not None and not self.pilot_sticks and abs(zoom0 - 1.0) < 1e-6:
            try:
                from olympe.messages.camera import reset_zoom
                self.drone(reset_zoom(cam_id=0))
                self.state.zoom = 1.0
                self.log.event("zoom_reset", zoom=1.0, ok=True)
            except Exception as exc:
                self.log.event("zoom_reset", ok=False, error=repr(exc))
                self.set_zoom(zoom0)
        else:
            self.set_zoom(zoom0)
        self.state.last_command = "camera_reset"
        self.log.event("camera_reset", pitch=pitch0, zoom=self.state.zoom)

    # -------------------------------------------------------------- UI interface
    def command(self, name: str, **payload) -> Any:
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
        elif name == "takeoff":
            self.takeoff_cmd()
        elif name == "firmware_limits_apply":
            return self.apply_firmware_limits(
                payload.get("max_altitude_m"),
                payload.get("max_distance_m"),
                payload.get("distance_geofence", True),
            )
        elif name in {"auto", "start_auto"}:
            # Take PC control for operator; full path-follow not auto-armed here.
            if self.take_pc_control():
                self.state.mode = "AUTO"
                self.state.tracker_state = "AUTO_UI"
                self.log.event(
                    "auto_ui", ok=True,
                    note="PC control taken; path_follow --fly not started from UI",
                )
            else:
                self.log.event("auto_ui", ok=False, note="source_unconfirmed")
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

    def stream_lost_hover(self, detail: str = "stream lost") -> Any:
        hovered = self.hover_cmd(f"stream_lost:{detail}")
        pipeline = None
        if self.grabber is not None:
            try:
                pipeline = self.grabber.frame_pipeline_stats
            except Exception:
                pipeline = None
        self.log.event(
            "stream_lost",
            detail=detail,
            hover_sent=bool(hovered),
            pipeline=pipeline,
        )
        self.state.stream = "LOST"
        self.state.loc = "STREAM_LOST"
        if hovered:
            self.state.tracker_state = "STREAM_LOST_HOVER"
            self.state.last_command = f"hover: {detail}"
        else:
            self.state.last_command = f"hover_blocked: {detail}"
        return self.state

    def set_gravity_sim_phase(self, phase: str | None) -> None:
        # Live: never synthesize attitude; real IMU is polled in poll().
        self.gravity_sim_phase = None
        self._gravity_sim_t0 = None
        self.log.event("gravity_phase", phase=phase, mode="live_real_attitude")

    def _probe_link_ok(self) -> bool:
        """Best-effort Olympe connection probe (display only; no flight cmds)."""
        if self.drone is None:
            return False
        try:
            cs = self.drone.connection_state()
            name = getattr(cs, "name", None) or str(cs)
            s = str(name).lower()
            if "disconnect" in s or "error" in s or s.endswith(".ko"):
                return False
            if "connect" in s and "disconnect" not in s:
                # Created/Connecting/Connected — only Connected is fully OK
                if "created" in s or "connecting" in s:
                    return getattr(self, "_link_was_ok", False)
                return True
        except Exception:
            pass
        try:
            # Fallback: any get_state success implies link up.
            from olympe.messages.common.CommonState import BatteryStateChanged
            self.drone.get_state(BatteryStateChanged)
            return True
        except Exception:
            return False

    def poll(self) -> Any:
        now = time.monotonic()
        self.last_poll = now
        if self.drone is None:
            self.state.link_ok = False
            self.state.link_status = "LOST"
            self._clear_firmware_safety_state()
            return self.state

        # Belt-and-suspenders: reclaim sticks even if HID callback missed a frame.
        if self.via_skycontroller() and not self.pilot_sticks:
            self._maybe_reclaim_from_sticks()

        # Link + frame-age at UI rate; full telemetry ~8 Hz.
        last_tel = getattr(self, "_last_telemetry_t", 0.0)
        do_tel = (now - last_tel) >= 0.12
        if do_tel:
            self._last_telemetry_t = now
            link_ok = self._probe_link_ok()
            prev = getattr(self, "_link_was_ok", True)
            self._link_was_ok = link_ok
            self.state.link_ok = link_ok
            self.state.link_status = "OK" if link_ok else "LOST"
            if prev and not link_ok:
                self.log.event("link_lost", note="display_only_no_auto_land")
            try:
                from olympe.messages.ardrone3.PilotingState import (
                    AttitudeChanged,
                    AltitudeChanged,
                    FlyingStateChanged,
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

            att = gs(AttitudeChanged)
            if att:
                self.state.att_roll = float(att.get("roll", 0.0) or 0.0)
                self.state.att_pitch = float(att.get("pitch", 0.0) or 0.0)
                self.state.att_yaw = float(att.get("yaw", 0.0) or 0.0)
                self.sim_yaw = self.state.att_yaw
            alt = gs(AltitudeChanged)
            if alt and alt.get("altitude") is not None:
                self.state.altitude_m = float(alt["altitude"])
            bat = gs(BatteryStateChanged)
            if bat and bat.get("percent") is not None:
                self.state.battery_pct = float(bat["percent"])
            fly = gs(FlyingStateChanged)
            if fly and fly.get("state") is not None:
                st = str(fly["state"])
                if self.state.tracker_state not in {
                    "NUDGE", "STICKS", "STREAM_LOST_HOVER", "PC_FROZEN", "HOVER", "PC",
                }:
                    self.state.tracker_state = st.upper()
            try:
                from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
                gps = gs(GPSFixStateChanged)
                if gps is not None and gps.get("fixed") is not None:
                    self.state.gps_fixed = bool(int(gps.get("fixed") or 0))
            except Exception:
                pass
            # Olympe get_state reads the local event cache; these values shown
            # in the UI never come from desired constructor settings.
            self._read_firmware_safety_state()
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
        # Do not clobber UI stream labels (PREVIEW/OK/HOLD) every poll.
        return self.state

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
        self._stop_stick_monitor()
        self.log.event("cleanup_begin", reason="exit_or_signal")
        try:
            self.nudge_clear(reason="cleanup")
            landed = False
            if self.drone is not None:
                # Zero immediately, even before a potentially blocking source
                # acknowledgement. A second zero follows a confirmed handoff.
                with self._lock:
                    try:
                        self._raw_pcmd(0, 0, 0, 0)
                    except Exception:
                        pass
                source_ok = self._set_piloting_source("Controller")
                if source_ok:
                    with self._lock:
                        self.pilot_sticks = False
                        try:
                            self._raw_pcmd(0, 0, 0, 0)
                        except Exception:
                            pass
                else:
                    self.log.event(
                        "land_source_unconfirmed", reason="cleanup",
                        note="Landing still attempted as fail-safe",
                    )

                if self._is_already_landed():
                    landed = True
                    self.state.tracker_state = "LAND"
                    self.log.event(
                        "land_cmd", reason="cleanup_exit", ok=True,
                        note="already_landed",
                    )
                else:
                    landed = self._land_and_confirm("cleanup_force_land")

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
                    self.log.event("record_stop", ok=False, reason="cleanup",
                                   error=repr(exc))
            elif self.recording_active:
                self.log.event(
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
                    self.log.event("disconnect_error", error=repr(exc))
                self.drone = None
            self.log.event("cleanup_done")
            self.log.close()
