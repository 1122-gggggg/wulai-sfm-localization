from __future__ import annotations

import math

from x11_pinch_zoom import PinchZoomSession


def test_pinch_zoom_tracks_absolute_scale_from_gesture_start() -> None:
    zoom = 2.0
    updates: list[float] = []

    def set_zoom(value: float) -> None:
        nonlocal zoom
        zoom = value
        updates.append(value)

    session = PinchZoomSession(lambda: zoom, set_zoom)

    session.begin()
    session.update(1.25)
    session.update(1.5)
    session.end(cancelled=False)

    assert updates == [2.5, 3.0]


def test_cancelled_pinch_restores_zoom_at_gesture_start() -> None:
    zoom = 4.0
    updates: list[float] = []

    def set_zoom(value: float) -> None:
        nonlocal zoom
        zoom = value
        updates.append(value)

    session = PinchZoomSession(lambda: zoom, set_zoom)

    session.begin()
    session.update(0.5)
    session.end(cancelled=True)

    assert updates == [2.0, 4.0]


def test_pinch_zoom_ignores_invalid_or_out_of_sequence_updates() -> None:
    updates: list[float] = []
    session = PinchZoomSession(lambda: 3.0, updates.append)

    session.update(2.0)
    session.begin()
    for scale in (0.0, -1.0, math.nan, math.inf):
        session.update(scale)

    assert updates == []
