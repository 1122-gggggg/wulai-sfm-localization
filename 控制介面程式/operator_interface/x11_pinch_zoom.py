"""Bridge XInput 2.4 touchpad pinch gestures into Tk map zoom callbacks."""

from __future__ import annotations

import ctypes
import math
import tkinter as tk
from collections.abc import Callable


_GENERIC_EVENT = 35
_XI_ALL_DEVICES = 0
_XI_GESTURE_PINCH_BEGIN = 27
_XI_GESTURE_PINCH_UPDATE = 28
_XI_GESTURE_PINCH_END = 29
_XI_GESTURE_CANCELLED = 1 << 0


class _XIEventMask(ctypes.Structure):
    _fields_ = (
        ("deviceid", ctypes.c_int),
        ("mask_len", ctypes.c_int),
        ("mask", ctypes.POINTER(ctypes.c_ubyte)),
    )


class _XGenericEventCookie(ctypes.Structure):
    _fields_ = (
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("extension", ctypes.c_int),
        ("evtype", ctypes.c_int),
        ("cookie", ctypes.c_uint),
        ("data", ctypes.c_void_p),
    )


class _XEvent(ctypes.Union):
    _fields_ = (
        ("type", ctypes.c_int),
        ("xcookie", _XGenericEventCookie),
        ("pad", ctypes.c_long * 24),
    )


class _XIModifierState(ctypes.Structure):
    _fields_ = (
        ("base", ctypes.c_int),
        ("latched", ctypes.c_int),
        ("locked", ctypes.c_int),
        ("effective", ctypes.c_int),
    )


class _XIGesturePinchEvent(ctypes.Structure):
    _fields_ = (
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("extension", ctypes.c_int),
        ("evtype", ctypes.c_int),
        ("time", ctypes.c_ulong),
        ("deviceid", ctypes.c_int),
        ("sourceid", ctypes.c_int),
        ("detail", ctypes.c_int),
        ("root", ctypes.c_ulong),
        ("event", ctypes.c_ulong),
        ("child", ctypes.c_ulong),
        ("root_x", ctypes.c_double),
        ("root_y", ctypes.c_double),
        ("event_x", ctypes.c_double),
        ("event_y", ctypes.c_double),
        ("delta_x", ctypes.c_double),
        ("delta_y", ctypes.c_double),
        ("delta_unaccel_x", ctypes.c_double),
        ("delta_unaccel_y", ctypes.c_double),
        ("scale", ctypes.c_double),
        ("delta_angle", ctypes.c_double),
        ("flags", ctypes.c_int),
        ("mods", _XIModifierState),
        ("group", _XIModifierState),
    )


class PinchZoomSession:
    """Apply each absolute pinch scale to the zoom at gesture start."""

    def __init__(
        self,
        get_zoom: Callable[[], float],
        set_zoom: Callable[[float], None],
    ) -> None:
        self._get_zoom = get_zoom
        self._set_zoom = set_zoom
        self._start_zoom: float | None = None

    def begin(self) -> None:
        self._start_zoom = float(self._get_zoom())

    def update(self, scale: float) -> None:
        if self._start_zoom is None or not math.isfinite(scale) or scale <= 0.0:
            return
        self._set_zoom(self._start_zoom * float(scale))

    def end(self, *, cancelled: bool) -> None:
        if self._start_zoom is not None and cancelled:
            self._set_zoom(self._start_zoom)
        self._start_zoom = None


class X11PinchZoom:
    """Listen for native XInput pinch events on one Tk widget."""

    def __init__(
        self,
        widget,
        get_zoom: Callable[[], float],
        set_zoom: Callable[[float], None],
    ) -> None:
        self._widget = widget
        self._session = PinchZoomSession(get_zoom, set_zoom)
        self._display = None
        self._fd: int | None = None
        self._closed = False

        self._x11 = ctypes.CDLL("libX11.so.6")
        self._xi = ctypes.CDLL("libXi.so.6")
        self._configure_libraries()

        display = self._x11.XOpenDisplay(None)
        if not display:
            raise RuntimeError("cannot open the X11 display")
        self._display = display
        try:
            opcode = ctypes.c_int()
            event_base = ctypes.c_int()
            error_base = ctypes.c_int()
            if not self._x11.XQueryExtension(
                display,
                b"XInputExtension",
                ctypes.byref(opcode),
                ctypes.byref(event_base),
                ctypes.byref(error_base),
            ):
                raise RuntimeError("XInput is unavailable")

            major = ctypes.c_int(2)
            minor = ctypes.c_int(4)
            if self._xi.XIQueryVersion(
                display, ctypes.byref(major), ctypes.byref(minor)
            ) != 0 or (major.value, minor.value) < (2, 4):
                raise RuntimeError("XInput 2.4 gestures are unavailable")
            self._opcode = opcode.value

            self._event_bits = (ctypes.c_ubyte * 4)()
            for event_type in (
                _XI_GESTURE_PINCH_BEGIN,
                _XI_GESTURE_PINCH_UPDATE,
                _XI_GESTURE_PINCH_END,
            ):
                self._event_bits[event_type >> 3] |= 1 << (event_type & 7)
            widget.update_idletasks()
            event_mask = _XIEventMask(
                _XI_ALL_DEVICES, len(self._event_bits), self._event_bits
            )
            if self._xi.XISelectEvents(
                display,
                int(widget.winfo_id()),
                ctypes.byref(event_mask),
                1,
            ) != 0:
                raise RuntimeError("cannot select XInput pinch events")
            self._x11.XSync(display, False)

            self._fd = int(self._x11.XConnectionNumber(display))
            widget.tk.createfilehandler(self._fd, tk.READABLE, self._read_events)
            widget.bind("<Destroy>", self._on_destroy, add="+")
        except Exception:
            self.close()
            raise

    def _configure_libraries(self) -> None:
        self._x11.XOpenDisplay.argtypes = (ctypes.c_char_p,)
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XCloseDisplay.argtypes = (ctypes.c_void_p,)
        self._x11.XQueryExtension.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        )
        self._x11.XQueryExtension.restype = ctypes.c_int
        self._x11.XConnectionNumber.argtypes = (ctypes.c_void_p,)
        self._x11.XConnectionNumber.restype = ctypes.c_int
        self._x11.XSync.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self._x11.XPending.argtypes = (ctypes.c_void_p,)
        self._x11.XPending.restype = ctypes.c_int
        self._x11.XNextEvent.argtypes = (ctypes.c_void_p, ctypes.POINTER(_XEvent))
        self._x11.XGetEventData.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_XGenericEventCookie),
        )
        self._x11.XGetEventData.restype = ctypes.c_int
        self._x11.XFreeEventData.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_XGenericEventCookie),
        )
        self._xi.XIQueryVersion.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        )
        self._xi.XIQueryVersion.restype = ctypes.c_int
        self._xi.XISelectEvents.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(_XIEventMask),
            ctypes.c_int,
        )
        self._xi.XISelectEvents.restype = ctypes.c_int

    def _read_events(self, _fd, _readable) -> None:
        while self._display and self._x11.XPending(self._display):
            event = _XEvent()
            self._x11.XNextEvent(self._display, ctypes.byref(event))
            cookie = event.xcookie
            if (
                cookie.type != _GENERIC_EVENT
                or cookie.extension != self._opcode
                or cookie.evtype
                not in (
                    _XI_GESTURE_PINCH_BEGIN,
                    _XI_GESTURE_PINCH_UPDATE,
                    _XI_GESTURE_PINCH_END,
                )
                or not self._x11.XGetEventData(
                    self._display, ctypes.byref(event.xcookie)
                )
            ):
                continue
            try:
                pinch = ctypes.cast(
                    event.xcookie.data, ctypes.POINTER(_XIGesturePinchEvent)
                ).contents
                if cookie.evtype == _XI_GESTURE_PINCH_BEGIN:
                    self._session.begin()
                elif cookie.evtype == _XI_GESTURE_PINCH_UPDATE:
                    self._session.update(float(pinch.scale))
                else:
                    self._session.end(
                        cancelled=bool(pinch.flags & _XI_GESTURE_CANCELLED)
                    )
            finally:
                self._x11.XFreeEventData(
                    self._display, ctypes.byref(event.xcookie)
                )

    def _on_destroy(self, event) -> None:
        if event.widget is self._widget:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fd is not None:
            try:
                self._widget.tk.deletefilehandler(self._fd)
            except tk.TclError:
                pass
            self._fd = None
        if self._display:
            self._x11.XCloseDisplay(self._display)
            self._display = None


def install_x11_pinch_zoom(
    widget,
    get_zoom: Callable[[], float],
    set_zoom: Callable[[float], None],
) -> X11PinchZoom | None:
    """Install native pinch zoom when this Tk instance runs on X11."""
    try:
        if widget.tk.call("tk", "windowingsystem") != "x11":
            return None
        return X11PinchZoom(widget, get_zoom, set_zoom)
    except (OSError, RuntimeError, tk.TclError):
        return None
