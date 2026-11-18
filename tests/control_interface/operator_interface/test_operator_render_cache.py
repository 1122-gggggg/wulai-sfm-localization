"""The map's route/history caching must not change what is drawn or detected."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

HERE = (
    Path(__file__).resolve().parents[3]
    / "控制介面程式"
    / "operator_interface"
)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from operator_rendering import (  # noqa: E402
    MapRenderContext,
    _draw_route_and_history,
    _pts_sig,
    _route_sig,
)

WIDTH, HEIGHT, ZOOM, RADIUS = 620, 430, 3.2, 12.0
PAN = np.array([7.5, -3.25])
SCALE = min(WIDTH, HEIGHT) * 0.46 * ZOOM / RADIUS


def _rotate(points) -> np.ndarray:
    """A non-trivial row-wise view transform (yaw plus offset)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3) - [1.0, 2.0, 0.5]
    cy, sy = np.cos(0.7), np.sin(0.7)
    x = cy * pts[:, 0] + sy * pts[:, 2]
    z = -sy * pts[:, 0] + cy * pts[:, 2]
    return np.stack([x, pts[:, 1], z], axis=1)


class _Draw:
    def __init__(self) -> None:
        self.lines: list[tuple[tuple, dict]] = []

    def line(self, xy, **kwargs) -> None:
        self.lines.append((tuple(xy), kwargs))

    def ellipse(self, *_args, **_kwargs) -> None: ...

    def polygon(self, *_args, **_kwargs) -> None: ...


def _context(route, history, health, transform_log=None, **overrides):
    def transform(points):
        if transform_log is not None:
            transform_log.append(len(np.asarray(points).reshape(-1, 3)))
        return _rotate(points)

    return MapRenderContext(
        **{
            **dict(
                width=WIDTH,
                height=HEIGHT,
                map_zoom=ZOOM,
                map_radius=RADIUS,
                map_pan=PAN,
                no_loc_markers=[],
                route_pts=route,
                history=history,
                history_health=health,
                pose=(0.0, 0.0, 0.0, float("nan")),
                camera_axes=None,
                camera_forward=None,
                transform_xyz=transform,
                project_world=lambda *_: (0, 0),
                route_color="#ff3ea5",
                health_color={"LOW": "#e0a92e"},
                route_dot_max=200,
                no_loc_max_markers=40,
                video_aspect_ratio=16 / 9,
                overlay_font=None,
                map_east=np.array([1.0, 0.0, 0.0]),
                map_north=np.array([0.0, 1.0, 0.0]),
                map_up=np.array([0.0, 0.0, 1.0]),
                projection_key=("view-a",),
            ),
            **overrides,
        }
    )


def _expected_screen(points) -> list[tuple[int, int]]:
    view = _rotate(np.asarray(points, dtype=float)[:, :3])
    xs = (WIDTH * 0.5 + PAN[0] + view[:, 0] * SCALE).astype(int)
    ys = (HEIGHT * 0.5 + PAN[1] - view[:, 1] * SCALE).astype(int)
    return list(zip(xs.tolist(), ys.tolist()))


def _route(count: int = 30) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [np.array(p, dtype=float) for p in rng.random((count, 3)) * 10.0]


def _pose(index: int) -> np.ndarray:
    return np.array([index * 0.37, 0.2 * index, 1.0, 0.1 * index])


def _lines(draw: _Draw):
    history = next(xy for xy, kw in draw.lines if kw["fill"] == "#5aa7e8")
    route = next(xy for xy, kw in draw.lines if kw["fill"] == "#ff3ea5")
    return list(history), list(route)


def test_route_and_history_pixels_match_a_direct_projection() -> None:
    route = _route()
    history = deque((_pose(i) for i in range(6)), maxlen=300)
    draw = _Draw()
    _draw_route_and_history(draw, _context(route, history, deque(["OK"] * 6, maxlen=300)))
    drawn_history, drawn_route = _lines(draw)
    assert drawn_history == _expected_screen(list(history))
    assert drawn_route == _expected_screen(route)


def test_new_pose_redraws_the_trail_without_reprojecting_the_static_route() -> None:
    route = _route(37)  # distinct from every trail length used below
    history = deque((_pose(i) for i in range(6)), maxlen=6)
    health = deque(["OK"] * 6, maxlen=6)
    transform_calls: list[int] = []
    _draw_route_and_history(
        _Draw(), _context(route, history, health, transform_calls, projection_key=("v-new",))
    )
    assert 37 in transform_calls

    history.append(_pose(6))  # the capped deque drops the oldest pose
    health.append("OK")
    transform_calls.clear()
    draw = _Draw()
    _draw_route_and_history(
        draw, _context(route, history, health, transform_calls, projection_key=("v-new",))
    )
    drawn_history, drawn_route = _lines(draw)
    assert drawn_history == _expected_screen(list(history))
    assert drawn_route == _expected_screen(route)
    assert 37 not in transform_calls


def test_changed_route_object_or_view_is_reprojected() -> None:
    route = _route(41)
    history = [_pose(0), _pose(1)]
    base = _context(route, history, ["OK", "OK"], projection_key=("v-route",))
    _draw_route_and_history(_Draw(), base)

    moved_route = [point + [0.0, 1.5, 0.0] for point in route]
    draw = _Draw()
    _draw_route_and_history(draw, replace(base, route_pts=moved_route))
    assert _lines(draw)[1] == _expected_screen(moved_route)

    shifted = replace(
        base,
        projection_key=("v-route-shifted",),
        transform_xyz=lambda pts: _rotate(pts) + [2.0, 0.0, 0.0],
    )
    draw = _Draw()
    _draw_route_and_history(draw, shifted)
    expected = [
        (int(WIDTH * 0.5 + PAN[0] + (v[0] + 2.0) * SCALE), int(HEIGHT * 0.5 + PAN[1] - v[1] * SCALE))
        for v in _rotate(np.asarray(route))
    ]
    assert _lines(draw)[1] == expected


def test_route_signature_follows_replacement_and_resizing() -> None:
    route = _route(25)
    first = _route_sig(route)
    assert first == _pts_sig(route)
    assert _route_sig(route) == first

    same_content = [point.copy() for point in route]
    assert _route_sig(same_content) == first

    moved_end = [point.copy() for point in route]
    moved_end[-1] = moved_end[-1] + [0.0, 0.0, 1.0]
    assert _route_sig(moved_end) != first

    moved_start = [point.copy() for point in route]
    moved_start[0] = moved_start[0] + [1.0, 0.0, 0.0]
    assert _route_sig(moved_start) != first

    shrinking = list(route)
    before = _route_sig(shrinking)
    shrinking[:] = shrinking[:2]
    assert _route_sig(shrinking) != before
    assert _route_sig(shrinking) == _pts_sig(shrinking)
    assert _route_sig([]) == 0
