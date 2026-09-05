"""Linux HID joystick adapter for SkyController stick override."""
from __future__ import annotations

import glob
import os
import struct
import threading
from typing import Callable


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
#: Stick-axis logging: ~2% of the +-32767 full throw. Deliberately below the
#: 2000-count override deadzone -- the log records what the pilot asked for,
#: including sub-deadzone trim, and does not decide who is flying.
_STICK_LOG_DELTA = 600
_STICK_LOG_HEARTBEAT_S = 1.0


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

    def _process_stick_data(self, data: bytes) -> None:
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
            self._notify_stick_activity()

    def _notify_stick_activity(self) -> None:
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
                self._process_stick_data(data)
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


def stick_log_sample(
    monitor: "SkyControllerStickMonitor",
    previous: dict[int, int],
    *,
    since_last_s: float,
    heartbeat_s: float = _STICK_LOG_HEARTBEAT_S,
    delta: int = _STICK_LOG_DELTA,
) -> tuple[dict[int, int], dict] | None:
    """One telemetry row for the pilot's stick position, or None to stay quiet.

    A manually flown session issues no PCMD, so ``commands.jsonl`` is empty for
    the whole flight and the recorded IMU has nothing explaining it. These rows
    are that missing input track. Emitted on deflection change plus a heartbeat,
    so a parked stick costs one row per ``heartbeat_s`` and a swept stick keeps
    every poll.
    """
    try:
        axes = monitor.snapshot_axes()
    except OSError:
        return None
    if not axes:
        return None
    moved = any(
        abs(int(value) - int(previous.get(axis, 0))) >= delta
        for axis, value in axes.items()
    )
    if not moved and since_last_s < heartbeat_s:
        return None
    fields = {
        "axes": {str(axis): int(value) for axis, value in sorted(axes.items())},
        "flight_axes_active": bool(axes_active(axes, deadzone=monitor.deadzone)),
        "moved": bool(moved),
    }
    return dict(axes), fields
