"""Self-centring virtual stick widget for the desktop operator interface."""
from __future__ import annotations

import math

import tkinter as tk


class VirtualStick(tk.Canvas):
    """A round, drag-to-command stick, sized and centred like a real gimbal.

    Deliberately a *self-centring* control: releasing, dragging out of the
    widget, or losing the pointer all snap back to zero and report it, so a
    lost event can only ever decay to hover -- never leave the aircraft with a
    latched deflection.
    """

    SIZE = 118
    KNOB = 15

    def __init__(self, master, *, title: str, x_label: str, y_label: str,
                 on_change) -> None:
        super().__init__(master, width=self.SIZE, height=self.SIZE,
                         highlightthickness=0, takefocus=0)
        self._on_change = on_change
        self.title = title
        self._x = 0.0
        self._y = 0.0
        pad = self.KNOB + 1
        self._radius = self.SIZE / 2.0 - pad
        c = self.SIZE / 2.0
        self.create_oval(c - self._radius - self.KNOB, c - self._radius - self.KNOB,
                         c + self._radius + self.KNOB, c + self._radius + self.KNOB,
                         outline="#8a8a8a", width=2, fill="#ededed")
        self.create_line(c - self._radius, c, c + self._radius, c, fill="#c0c0c0")
        self.create_line(c, c - self._radius, c, c + self._radius, fill="#c0c0c0")
        self.create_text(c, 7, text=y_label, font=("Sans", 7), fill="#666")
        self.create_text(c, self.SIZE - 7, text=y_label.split("/")[-1] if "/" in y_label else "",
                         font=("Sans", 7), fill="#666")
        self.create_text(self.SIZE - 12, c, text=x_label, font=("Sans", 7), fill="#666")
        self._knob = self.create_oval(c - self.KNOB, c - self.KNOB,
                                      c + self.KNOB, c + self.KNOB,
                                      fill="#4a76c8", outline="#26467d", width=2)
        for sequence in ("<ButtonPress-1>", "<B1-Motion>"):
            self.bind(sequence, self._on_drag)
        for sequence in ("<ButtonRelease-1>", "<Leave>"):
            self.bind(sequence, self._on_release)

    @property
    def value(self) -> tuple[float, float]:
        return (self._x, self._y)

    def _on_drag(self, event) -> None:
        c = self.SIZE / 2.0
        dx = (event.x - c) / self._radius
        dy = (c - event.y) / self._radius
        magnitude = math.hypot(dx, dy)
        if magnitude > 1.0:
            dx /= magnitude
            dy /= magnitude
        self._set(dx, dy)

    def _on_release(self, _event=None) -> None:
        self._set(0.0, 0.0)

    def recenter(self) -> None:
        """Snap to zero WITHOUT reporting -- for callers already sending zero."""
        self._x = self._y = 0.0
        self._draw()

    def show_keyboard(self, x: float, y: float) -> None:
        """Display held-key input without emitting a second flight command."""
        self._keyboard_x = float(x)
        self._keyboard_y = float(y)
        self._draw()

    def _set(self, x: float, y: float) -> None:
        self._x, self._y = x, y
        self._draw()
        self._on_change(self, x, y)

    def _draw(self) -> None:
        x = self._x + float(getattr(self, "_keyboard_x", 0.0))
        y = self._y + float(getattr(self, "_keyboard_y", 0.0))
        magnitude = math.hypot(x, y)
        if magnitude > 1.0:
            x /= magnitude
            y /= magnitude
        c = self.SIZE / 2.0
        cx = c + x * self._radius
        cy = c - y * self._radius
        self.coords(self._knob, cx - self.KNOB, cy - self.KNOB,
                    cx + self.KNOB, cy + self.KNOB)
