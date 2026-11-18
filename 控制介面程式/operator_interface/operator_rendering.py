"""Pure rendering seams used by the desktop operator interface.

The Tk application owns widget state and the public ``OperatorApp`` methods.
This module owns the drawing decisions once those values have been collected
into explicit arguments.  Keeping the helpers free of ``OperatorApp`` makes
the map/video paths easier to test and keeps the UI tick from accumulating
another implicit state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from localization_result_ui import localization_result_is_weak
from operator_ui_style import ACCENT, BORDER, FONT_BOLD, FONT_SMALL, MUTED, TEXT
from route_editor_wires import wire_color

#: Initial requested map/video canvas height, not a rendering floor. Fixed
#: controls receive their natural height; renderers use the allocated remainder.
VIEWPORT_MIN_H = 96


def _positive_env_int(key: str, default: int) -> int:
    try:
        val = int(os.environ.get(key, default))
        return val if val > 0 else default
    except (ValueError, TypeError):
        return default


def _positive_env_float(key: str, default: float) -> float:
    try:
        val = float(os.environ.get(key, default))
        return val if val > 0.0 else default
    except (ValueError, TypeError):
        return default


LOC_LOW_INLIERS = _positive_env_int("SFM_LOW_CONF_INLIERS", 60)
LOC_HIGH_REPROJ = _positive_env_float("SFM_LOC_HIGH_REPROJ", 4.0)


@dataclass(frozen=True)
class MapRenderContext:
    """All application state needed to paint one map overlay frame."""

    width: int
    height: int
    map_zoom: float
    map_radius: float
    map_pan: np.ndarray
    no_loc_markers: Sequence[object]
    route_pts: Sequence[object]
    history: Sequence[object]
    history_health: Sequence[str]
    pose: Sequence[float]
    camera_axes: np.ndarray | None
    camera_forward: np.ndarray | None
    transform_xyz: Callable[[np.ndarray], np.ndarray]
    project_world: Callable[[np.ndarray, int, int], tuple[int, int]]
    route_color: str
    health_color: Mapping[str, str]
    route_dot_max: int
    no_loc_max_markers: int
    video_aspect_ratio: float
    overlay_font: Any
    #: Map ground plane in raw GLOMAP coordinates -- the measured gravity basis
    #: when the site declares one, else the legacy assumption. The camera
    #: picture plane is a real 3D quad, so it needs the gravity basis to build
    #: orthonormal image axes from an optical axis or from yaw+gimbal telemetry.
    map_east: np.ndarray
    map_north: np.ndarray
    map_up: np.ndarray
    #: Telemetry gimbal pitch in degrees, positive up, for the yaw-only
    #: fallback. Measured camera axes carry their own attitude and ignore it.
    gimbal_pitch_deg: Any = None
    #: Display-only weak trail (VO_ONLY / DEAD_RECKON / WEAK_TRACK /
    #: PREDICTED_ONLY mirrors). Parallel lists, same cap as history. Defaults
    #: keep every existing constructor working.
    history_weak: Sequence[object] = ()
    history_weak_health: Sequence[str] = ()
    map_rotation: tuple[float, float, float] | np.ndarray | None = None
    map_yaw: float = 0.0
    map_pitch: float = 0.0
    map_roll: float = 0.0
    #: Estimate source per weak-trail point (KLT / IMU / OTHER), parallel to
    #: history_weak. Appended last so existing positional constructors keep
    #: working; missing entries fall back to the health colour.
    history_weak_kind: Sequence[str] = ()
    #: Current display-only estimate, independent of the held flight pose.
    #: Empty kind/auto attitude preserve measured/replay camera rendering.
    camera_estimate_kind: str = ""
    camera_attitude: str = "auto"
    camera_estimate_stale: bool = False
    #: Identifies the complete application projection, including its pivot.
    projection_key: tuple = ()
    #: Map-aligned historical submap trails, never live pose/flight history.
    workmap_trajectories: Mapping[int, Sequence[Sequence[float]]] | None = None
    workmap_mapping_diagnostics: Mapping[str, Any] | None = None
    #: Read-only sampled wire curves in raw map coordinates; display only.
    wire_polylines: Sequence[np.ndarray] = ()


#: Camera picture rectangle: a real plane in map space, oriented by the full
#: camera attitude, drawn ahead of the drone like the screen the stream looks
#: at. Yaw spins it through 360 deg, gimbal pitch tilts it, and the pitched map
#: view foreshortens it like any other 3D object -- seen edge-on it collapses
#: to a line, face-on it fills out, exactly as a screen would.
#:
#: The px constants below are on-screen extents, converted to world units with
#: the CURRENT view scale every frame. The orientation is honest 3D, but the
#: extent is normalised rather than fixed in world units, for two measured
#: reasons:
#:   - a world-fixed plane shrank to 6-10 px at the default view (the original
#:     frustum sat at ``map_radius * 0.015``, which cancels against
#:     ``_map_scale``) -- the same size as the position dot under it, and
#:   - a flat screen-space rectangle (the previous symbol) kept one size but
#:     could not show tilt at all: it stayed glued to the map no matter where
#:     the stream pointed, which read as a bearing arrow, not a picture.
#: Pegging the extent to the view scale keeps the calibrated readable size
#: while the facing is free to point anywhere on the sphere.
#:
#: The quad's front side is the stream direction. The white gaze line runs from
#: the drone through the picture and ends in an arrowhead beyond it, so the
#: facing stays readable even when the quad is seen edge-on. Border/dot colour:
#: cyan = camera-measured attitude, green = yaw+gimbal telemetry fallback.
#: Gimbal roll is not symbolised; the video HUD reports it numerically.
#:
#: 72 px overshot: at the default view the symbol covered a real part of the
#: point cloud and the route under it. 44 px is still several times the 8 px
#: position dot, so the bearing stays readable.
CAMERA_VIEW_RECT_PX = 44.0
CAMERA_VIEW_RECT_GAP_PX = 8.0
#: How far the white direction arrow pokes past the picture plane, and the
#: arrowhead's length; its half-width is 0.6 of the length.
CAMERA_VIEW_ARROW_PX = 14.0
CAMERA_VIEW_ARROW_HEAD_PX = 7.0


def _map_scale(context: MapRenderContext) -> float:
    return min(context.width, context.height) * 0.46 * context.map_zoom / context.map_radius


def _basis_from_forward(
    context: MapRenderContext,
    forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(right, image-up, forward) unit world vectors for a known optical axis.

    ``right`` is taken against the map gravity axis. Near nadir/zenith that
    cross product collapses, so map north stands in: the picture top then
    points along the heading, the only defensible guess left when no measured
    image axes exist.
    """
    vector = np.asarray(forward, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    forward = vector / norm
    up_map = np.asarray(context.map_up, dtype=float)
    right = np.cross(forward, up_map)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.cross(forward, np.asarray(context.map_north, dtype=float))
    norm = float(np.linalg.norm(right))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    right = right / norm
    return right, np.cross(right, forward), forward


def _camera_picture_basis(
    context: MapRenderContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """The stream direction as a full 3D camera basis, or None.

    Returns (right, image_up, forward) unit world vectors plus a bool marking
    the yaw+gimbal telemetry fallback (green border) against a
    camera-measured attitude (cyan). The picture rectangle is a real plane, so
    it needs the whole attitude: a ground-flattened azimuth cannot tilt it,
    and tilt is the point of drawing it in 3D.
    """
    if context.camera_attitude == "none":
        return None
    if context.camera_attitude == "heading":
        return _heading_picture_basis(context)
    if context.camera_axes is not None:
        return _axes_picture_basis(context)
    if context.camera_forward is not None:
        return _flagged_basis(_basis_from_forward(context, context.camera_forward), False)
    return _telemetry_picture_basis(context)


def _flagged_basis(
    built: tuple[np.ndarray, np.ndarray, np.ndarray] | None, telemetry_fallback: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    return (*built, telemetry_fallback) if built is not None else None


def _yaw_horizontal(context: MapRenderContext, yaw: float) -> np.ndarray:
    horizontal: np.ndarray = math.cos(yaw) * np.asarray(context.map_east, dtype=float) + math.sin(
        yaw
    ) * np.asarray(context.map_north, dtype=float)
    return horizontal


def _heading_picture_basis(
    context: MapRenderContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    try:
        yaw = float(context.pose[3])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(yaw):
        return None
    return _flagged_basis(_basis_from_forward(context, _yaw_horizontal(context, yaw)), True)


def _axes_picture_basis(
    context: MapRenderContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    axes = np.asarray(context.camera_axes, dtype=float)
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        return None
    right, down, forward = axes
    f_norm = float(np.linalg.norm(forward))
    if f_norm < 1e-9:
        return None
    forward = forward / f_norm
    r_norm = float(np.linalg.norm(right))
    d_norm = float(np.linalg.norm(down))
    if r_norm >= 1e-9 and d_norm >= 1e-9:
        return right / r_norm, -down / d_norm, forward, False
    # Degenerate in-plane rows (e.g. right collapsed at nadir): rebuild
    # them off the optical axis instead of drawing nothing.
    return _flagged_basis(_basis_from_forward(context, forward), False)


def _gimbal_pitch_radians(context: MapRenderContext) -> float:
    try:
        pitch_value = math.radians(float(context.gimbal_pitch_deg))
    except (TypeError, ValueError):
        pitch_value = 0.0
    if not math.isfinite(pitch_value):
        pitch_value = 0.0
    return max(-math.pi / 2 + 1e-6, min(math.pi / 2 - 1e-6, pitch_value))


def _telemetry_picture_basis(
    context: MapRenderContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    try:
        yaw_value = float(context.pose[3])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(yaw_value):
        return None
    pitch_value = _gimbal_pitch_radians(context)
    horizontal = _yaw_horizontal(context, yaw_value)
    forward = math.cos(pitch_value) * horizontal + math.sin(pitch_value) * np.asarray(
        context.map_up, dtype=float
    )
    return _flagged_basis(_basis_from_forward(context, forward), True)


def camera_view_plane_corners(
    drone: np.ndarray,
    right: np.ndarray,
    image_up: np.ndarray,
    forward: np.ndarray,
    scale: float,
    aspect_ratio: float,
) -> np.ndarray:
    """World corners of the picture plane ahead of the drone, in image order.

    Returns top-left, top-right, bottom-right, bottom-left of a quad that is
    perpendicular to the optical axis: its plane sits
    ``CAMERA_VIEW_RECT_GAP_PX`` ahead of the drone, is ``CAMERA_VIEW_RECT_PX``
    wide and width / ``aspect_ratio`` tall -- the stream's picture shape -- all
    converted to world units with the current view scale.
    """
    gap = CAMERA_VIEW_RECT_GAP_PX / scale
    half_width = CAMERA_VIEW_RECT_PX * 0.5 / scale
    half_height = half_width / float(aspect_ratio)
    center = np.asarray(drone, dtype=float) + np.asarray(forward, dtype=float) * gap
    corners: np.ndarray = np.stack(
        [
            center - right * half_width + image_up * half_height,
            center + right * half_width + image_up * half_height,
            center + right * half_width - image_up * half_height,
            center - right * half_width - image_up * half_height,
        ]
    )
    return corners


_PROJECTION_CACHE: dict[tuple, tuple[list[int], list[int]]] = {}
_PROJECTION_CACHE_MAX = 64


def _pts_sig(seq: Sequence[Any] | None) -> tuple | int:
    if seq is None or len(seq) == 0:
        return 0
    # Sampling missed edits in the middle of a capped trail or route. Retain
    # exact coordinates, not references to mutable points or sampled endpoints.
    points = np.asarray(list(seq), dtype=float)
    return (len(points), points[:, :3].tobytes())


#: One-slot memo for the planned route's signature. The ~2000-point route is
#: only ever replaced as a whole object (or emptied/resliced, which changes
#: its length); no site edits its points in place. Re-serialising it every
#: 33 ms tick therefore only buys the same bytes. The memo keeps the sequence
#: itself, so an identity match can never be a recycled ``id()``. The trail,
#: weak trail and no-fix markers do change in place and keep ``_pts_sig``.
_route_sig_memo: tuple[Sequence[Any], int, tuple[bytes, bytes], tuple | int] | None = None


def _point_xyz_bytes(point: Any) -> bytes:
    xyz: np.ndarray = np.asarray(point, dtype=float)[:3]
    data: bytes = xyz.tobytes()
    return data


def _route_sig(route: Sequence[Any] | None) -> tuple | int:
    """``_pts_sig`` of the planned route, recomputed only when it was replaced."""
    global _route_sig_memo
    if route is None or len(route) == 0:
        return 0
    count = len(route)
    ends = (_point_xyz_bytes(route[0]), _point_xyz_bytes(route[-1]))
    memo = _route_sig_memo
    if memo is not None and memo[0] is route and memo[1] == count and memo[2] == ends:
        return memo[3]
    sig = _pts_sig(route)
    _route_sig_memo = (route, count, ends, sig)
    return sig


def _trail_xyz(trail: Sequence[Any]) -> np.ndarray:
    """All trail points as one ``(n, 3)`` array."""
    if len(trail) == 0:
        return np.empty((0, 3), dtype=float)
    points: np.ndarray = np.asarray(list(trail), dtype=float)[:, :3].reshape(-1, 3)
    return points


def _draw_route_and_history(draw: Any, context: MapRenderContext) -> None:
    route_n = len(context.route_pts)
    history_n = len(context.history)
    weak_trail = list(getattr(context, "history_weak", None) or ())
    weak_health = list(getattr(context, "history_weak_health", None) or ())
    weak_n = len(weak_trail)
    if not route_n and not history_n and not weak_n:
        return

    pan_arr = np.asarray(context.map_pan, dtype=float).reshape(-1)
    pan_key = (
        (
            round(float(pan_arr[0]), 4),
            round(float(pan_arr[1]), 4),
        )
        if len(pan_arr) >= 2
        else (0.0, 0.0)
    )
    zoom_key = round(float(context.map_zoom), 6)
    size_key = (int(context.width), int(context.height))
    rot_val = getattr(context, "map_rotation", None)
    if rot_val is not None and len(rot_val) >= 3:
        rotation_key = (
            round(float(rot_val[0]), 4),
            round(float(rot_val[1]), 4),
            round(float(rot_val[2]), 4),
        )
    else:
        rotation_key = (
            round(float(getattr(context, "map_yaw", 0.0)), 4),
            round(float(getattr(context, "map_pitch", 0.0)), 4),
            round(float(getattr(context, "map_roll", 0.0)), 4),
        )

    view_key = (
        pan_key,
        zoom_key,
        size_key,
        rotation_key,
        round(float(getattr(context, "map_radius", 1.0)), 4),
        context.projection_key,
    )

    def project(cache_key: tuple, points: Callable[[], np.ndarray]) -> tuple[list[int], list[int]]:
        cached = _PROJECTION_CACHE.get(cache_key)
        if cached is not None:
            return cached
        view = context.transform_xyz(points())
        scale = _map_scale(context)
        projected = (
            (context.width * 0.5 + context.map_pan[0] + view[:, 0] * scale).astype(int).tolist(),
            (context.height * 0.5 + context.map_pan[1] - view[:, 1] * scale).astype(int).tolist(),
        )
        if len(_PROJECTION_CACHE) >= _PROJECTION_CACHE_MAX:
            _PROJECTION_CACHE.clear()
        _PROJECTION_CACHE[cache_key] = projected
        return projected

    # The static route is keyed by the view and its own signature, so a new
    # pose never re-projects it; trails (<= 300 points each) change per pose.
    route_x, route_y = (
        project(
            ("route", view_key, _route_sig(context.route_pts)),
            lambda: np.asarray(context.route_pts, dtype=float).reshape(-1, 3),
        )
        if route_n
        else ([], [])
    )
    history_xyz = _trail_xyz(context.history) if history_n else None
    history_x, history_y = (
        project(
            ("history", view_key, history_n, history_xyz.tobytes()),
            lambda: history_xyz,
        )
        if history_xyz is not None
        else ([], [])
    )
    weak_xyz = _trail_xyz(weak_trail) if weak_n else None
    weak_x, weak_y = (
        project(
            ("weak", view_key, weak_n, weak_xyz.tobytes()),
            lambda: weak_xyz,
        )
        if weak_xyz is not None
        else ([], [])
    )
    if history_n > 1:
        history_screen = list(zip(history_x, history_y))
        draw.line(history_screen, fill="#5aa7e8", width=3)
        # Explicit alignment: history and history_health are parallel lists,
        # but a dropped append would make zip() silently swallow the tail.
        # Draw markers only over the explicitly shared prefix.
        history_health = list(context.history_health)  # a deque indexes in O(n)
        dot_n = min(history_n, len(history_health))
        for index in range(dot_n):
            x, y = history_screen[index]
            health = history_health[index]
            if health and health != "OK":
                draw.ellipse(
                    (x - 4, y - 4, x + 4, y + 4),
                    fill=context.health_color.get(health, "#e0a92e"),
                )

    if weak_n:
        # Weak fixes (VO_ONLY / DEAD_RECKON / WEAK_TRACK / PREDICTED_ONLY):
        # hollow diamonds coloured by estimate source (KLT green, IMU yellow,
        # anything else the health amber). A solid dot is already the non-OK
        # symbol, so weak reusing it would collide; markers only, no
        # connecting line, so a jump-rejected fix cannot imply a flown
        # segment that never happened.
        weak_screen = list(zip(weak_x, weak_y))
        # Same explicit alignment as the history markers above, and the
        # position has to come from weak_screen: reading a bare x/y here picked
        # up whatever the route or history loop happened to leave bound, which
        # stacked every weak diamond on one wrong point -- and raised
        # UnboundLocalError outright when neither of those loops had run.
        weak_kind = list(getattr(context, "history_weak_kind", None) or ())
        mark_n = min(weak_n, len(weak_health), len(weak_screen))
        for index in range(mark_n):
            x, y = weak_screen[index]
            kind = weak_kind[index] if index < len(weak_kind) else "OTHER"
            colour = WEAK_KIND_COLOR.get(kind)
            if colour is None:
                colour = context.health_color.get(weak_health[index], WEAK_KIND_FALLBACK_COLOR)
            size = 3
            draw.polygon(
                [(x, y - size), (x + size, y), (x, y + size), (x - size, y)],
                outline=colour,
                width=1,
            )

    # Keep the planned path visible when dense estimate markers overlap it.
    if route_n > 1:
        route_screen = list(zip(route_x, route_y))
        draw.line(route_screen, fill=context.route_color, width=2, joint="curve")
        dot_step = max(1, route_n // max(1, context.route_dot_max))
        for x, y in route_screen[::dot_step]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=context.route_color)


def _draw_workmap_trajectories(draw: Any, context: MapRenderContext) -> None:
    trails = context.workmap_trajectories
    if not trails:
        return
    scale = _map_scale(context)
    for points in trails.values():
        if len(points) < 2:
            continue
        view = context.transform_xyz(np.asarray(points, dtype=float))
        screen = list(
            zip(
                (context.width * 0.5 + context.map_pan[0] + view[:, 0] * scale).astype(int),
                (context.height * 0.5 + context.map_pan[1] - view[:, 1] * scale).astype(int),
            )
        )
        # One line per segment, not a bridge across unrelated weak-map runs.
        draw.line(screen, fill="#c084fc", width=2)


def _draw_workmap_mapping_health(draw: Any, context: MapRenderContext) -> None:
    diagnostics = context.workmap_mapping_diagnostics
    if not diagnostics:
        return
    healthy = diagnostics.get("workmap_mapping_healthy")
    health = "正常" if healthy is True else "停止" if healthy is False else "未知"
    state = diagnostics.get("workmap_mapping_state")
    source = diagnostics.get("workmap_output_source")
    error = diagnostics.get("workmap_mapping_error")
    detail = " | ".join(str(value) for value in (state, source, error) if value is not None)
    draw.text(
        (10, 28),
        f"工作圖建圖：{health}" + (f" | {detail}" if detail else ""),
        fill="#ff6b5f" if healthy is False else "#00e6ff",
        font=context.overlay_font,
    )


def _camera_estimate_color(context: MapRenderContext) -> str | None:
    if not context.camera_estimate_kind:
        return None
    return WEAK_KIND_COLOR.get(context.camera_estimate_kind, WEAK_KIND_FALLBACK_COLOR)


def _draw_camera_estimate_label(draw: Any, context: MapRenderContext, x: float, y: float) -> None:
    color = _camera_estimate_color(context)
    if color is None:
        return
    detail = {
        "full": "attitude estimate",
        "direction": "direction-only estimate",
        "heading": "heading-only estimate",
        "none": "position-only estimate",
    }.get(context.camera_attitude, "estimate")
    draw.text(
        (x + 10, y + 12),
        f"{context.camera_estimate_kind} {'stale ' if context.camera_estimate_stale else ''}{detail}",
        fill=color,
        font=context.overlay_font,
    )


def _draw_no_orientation_marker(draw: Any, context: MapRenderContext) -> None:
    """Hollow ring when no orientation source survives: position without bearing."""
    screen_x, screen_y = context.project_world(
        np.asarray(context.pose[:3], dtype=float), context.width, context.height
    )
    draw.ellipse(
        (screen_x - 7, screen_y - 7, screen_x + 7, screen_y + 7),
        outline=_camera_estimate_color(context) or "#3fbf7f",
        width=3,
    )
    _draw_camera_estimate_label(draw, context, screen_x, screen_y)


def _draw_camera_overlay(draw: Any, context: MapRenderContext) -> None:
    basis = _camera_picture_basis(context)
    scale = _map_scale(context)
    if basis is None or not math.isfinite(scale) or scale <= 1e-9:
        _draw_no_orientation_marker(draw, context)
        return
    right, image_up, forward, from_yaw = basis
    aspect = float(context.video_aspect_ratio)
    if not math.isfinite(aspect) or aspect < 1e-6:
        aspect = 16.0 / 9.0
    drone = np.asarray(context.pose[:3], dtype=float)
    corners = camera_view_plane_corners(drone, right, image_up, forward, scale, aspect)
    tip = drone + forward * ((CAMERA_VIEW_RECT_GAP_PX + CAMERA_VIEW_ARROW_PX) / scale)
    head_base = tip - forward * (CAMERA_VIEW_ARROW_HEAD_PX / scale)
    head_half = CAMERA_VIEW_ARROW_HEAD_PX * 0.6 / scale
    # One batched transform for everything: drone, quad corners, arrow tip and
    # arrowhead base -- the same projection the cloud and the route use.
    view = context.transform_xyz(
        np.stack(
            [
                drone,
                *corners,
                tip,
                head_base + right * head_half,
                head_base - right * head_half,
            ]
        )
    )
    screen_x = context.width * 0.5 + context.map_pan[0] + view[:, 0] * scale
    screen_y = context.height * 0.5 + context.map_pan[1] - view[:, 1] * scale
    estimate_color = _camera_estimate_color(context)
    border_color = estimate_color or ("#3fbf7f" if from_yaw else "#00d4ff")
    drone_pt = (screen_x[0], screen_y[0])
    quad = list(zip(screen_x[1:5], screen_y[1:5]))
    # The gaze line runs under the fill, drone -> picture -> arrowhead: one
    # stem, not a cone of corner lines, and it carries the facing direction
    # even when the quad is seen edge-on.
    draw.line((*drone_pt, screen_x[5], screen_y[5]), fill=estimate_color or "#ffffff", width=2)
    # Without observed image axes a picture plane would invent camera roll.
    # Heading/direction estimates therefore draw only their honest arrow.
    if context.camera_attitude not in {"heading", "direction"}:
        draw.polygon(quad, fill="#0a5f73")
        draw.line(quad + [quad[0]], fill=border_color, width=3, joint="curve")
    draw.polygon(
        [(screen_x[5], screen_y[5]), (screen_x[6], screen_y[6]), (screen_x[7], screen_y[7])],
        fill=estimate_color or "#ffffff",
    )
    draw.ellipse(
        (drone_pt[0] - 4, drone_pt[1] - 4, drone_pt[0] + 4, drone_pt[1] + 4),
        fill=border_color,
        outline="#ffffff",
        width=2,
    )
    _draw_camera_estimate_label(draw, context, *drone_pt)


def _draw_wire_overlays(draw: Any, context: MapRenderContext) -> None:
    if not context.wire_polylines:
        return
    scale = _map_scale(context)
    for index, points in enumerate(context.wire_polylines):
        view = context.transform_xyz(points)
        sx = context.width * 0.5 + context.map_pan[0] + view[:, 0] * scale
        sy = context.height * 0.5 + context.map_pan[1] - view[:, 1] * scale
        color = wire_color(index)
        draw.line(tuple(zip(sx, sy)), fill=color, width=3)
        for endpoint in (0, -1):
            x, y = float(sx[endpoint]), float(sy[endpoint])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        middle = len(sx) // 2
        draw.text(
            (float(sx[middle]) + 5, float(sy[middle]) + 5),
            f"W{index + 1}", fill=color, font=context.overlay_font,
        )


def draw_map_overlays(draw: Any, context: MapRenderContext) -> None:
    """Paint dynamic map overlays over an already-rendered map base."""
    _draw_workmap_trajectories(draw, context)
    _draw_wire_overlays(draw, context)
    _draw_route_and_history(draw, context)
    _draw_camera_overlay(draw, context)
    _draw_workmap_mapping_health(draw, context)


def _fit_pil_frame(
    source: Image.Image,
    width: int,
    height: int,
) -> tuple[Image.Image, float]:
    scale = min(width / source.width, height / source.height)
    new_size = (
        max(1, int(source.width * scale)),
        max(1, int(source.height * scale)),
    )
    if source.size == new_size:
        return source, scale
    # Integer box reduction avoids running the bilinear kernel over all 720p
    # source pixels for the much smaller operator pane. No fitted-frame cache
    # or changing metric strings are retained here.
    return source.resize(new_size, Image.Resampling.BILINEAR, reducing_gap=1.0), scale


def prepare_video_frame(
    source: object | None,
    width: int,
    height: int,
) -> tuple[Image.Image | None, float]:
    """Convert and fit one PIL/NumPy video frame to the panel size."""
    if source is None:
        return None, 1.0
    try:
        if isinstance(source, np.ndarray):
            source_height, source_width = (
                int(source.shape[0]),
                int(source.shape[1]),
            )
            scale = min(
                width / max(1, source_width),
                height / max(1, source_height),
            )
            new_size = (
                max(1, int(source_width * scale)),
                max(1, int(source_height * scale)),
            )
            if (source_width, source_height) != new_size:
                import cv2

                source = cv2.resize(
                    source,
                    new_size,
                    interpolation=cv2.INTER_AREA,
                )
            frame = Image.fromarray(source, mode="RGB")
            return frame, scale

        pil_source = (
            source
            if isinstance(source, Image.Image)
            else Image.fromarray(np.asarray(source), mode="RGB")
        )
        return _fit_pil_frame(pil_source, width, height)
    except Exception:
        try:
            pil_source = (
                source
                if isinstance(source, Image.Image)
                else Image.fromarray(np.asarray(source), mode="RGB")
            )
            return _fit_pil_frame(pil_source, width, height)
        except Exception:
            return None, 1.0


class VideoCanvasDraw:
    """Retained Canvas painter shared by the map/video drawing helpers.

    Overlay items are reused and only changed coordinates/options cross Tcl.
    No image pixels are touched when age, telemetry or a banner changes.
    Instances, fonts and all calls belong to the Tk thread.
    """

    def __init__(self, canvas: Any, *, tag: str = "video-overlay") -> None:
        self.canvas = canvas
        self._tag = tag
        self._items: list[tuple[str, int, tuple, dict]] = []
        self._cursor = 0
        self._fonts: dict[tuple[int, bool], Any] = {}
        self._order_dirty = False
        self._hud_layout_cache = None
        self._hud_measure_cache = None

    def font(self, size: int, *, bold: bool = False) -> Any:
        from tkinter import font as tkfont

        key = (size, bold)
        if key not in self._fonts:
            available = set(tkfont.families(self.canvas))
            family = next(
                (
                    name
                    for name in ("Noto Sans CJK TC", "Noto Sans TC", "DejaVu Sans")
                    if name in available
                ),
                "TkDefaultFont",
            )
            self._fonts[key] = tkfont.Font(
                root=self.canvas,
                family=family,
                size=-size,
                weight="bold" if bold else "normal",
            )
        return self._fonts[key]

    def _tk_font(self, font: Any) -> Any:
        if hasattr(font, "measure"):
            return font
        # Detection labels still supply the shared PIL font.
        try:
            _, style = font.getname()
        except AttributeError:
            style = ""
        return self.font(int(getattr(font, "size", 11)), bold="bold" in style.lower())

    def begin(self) -> None:
        self._cursor = 0

    def end(self) -> None:
        for _, item, _, _ in self._items[self._cursor :]:
            self.canvas.delete(item)
        del self._items[self._cursor :]
        if self._order_dirty:
            for _, item, _, _ in self._items:
                self.canvas.tag_raise(item)
            self._order_dirty = False

    def _paint(self, kind: str, coordinates: tuple, options: dict) -> None:
        index = self._cursor
        self._cursor += 1
        if index < len(self._items):
            previous_kind, item, previous_coordinates, previous_options = self._items[index]
            if previous_kind != kind:
                self.canvas.delete(item)
                item = getattr(self.canvas, f"create_{kind}")(
                    *coordinates,
                    **options,
                    tags=self._tag,
                )
                self._order_dirty = True
            else:
                if previous_coordinates != coordinates:
                    self.canvas.coords(item, *coordinates)
                changed = {
                    key: value
                    for key, value in options.items()
                    if previous_options.get(key) != value
                }
                if changed:
                    self.canvas.itemconfigure(item, **changed)
            self._items[index] = (kind, item, coordinates, options)
        else:
            item = getattr(self.canvas, f"create_{kind}")(
                *coordinates,
                **options,
                tags=self._tag,
            )
            self._items.append((kind, item, coordinates, options))
            self._order_dirty = True

    def rectangle(self, coordinates: tuple, **options: Any) -> None:
        self._paint(
            "rectangle",
            tuple(coordinates),
            {
                "fill": options.get("fill", ""),
                "outline": options.get("outline", ""),
                "width": options.get("width", 1),
            },
        )

    def line(self, coordinates: tuple, **options: Any) -> None:
        if coordinates and isinstance(coordinates[0], (tuple, list)):
            coordinates = tuple(value for point in coordinates for value in point)
        self._paint(
            "line",
            tuple(coordinates),
            {
                "fill": options.get("fill", ""),
                "width": options.get("width", 1),
            },
        )

    def ellipse(self, coordinates: tuple, **options: Any) -> None:
        self._paint(
            "oval",
            tuple(coordinates),
            {
                "fill": options.get("fill", ""),
                "outline": options.get("outline", ""),
                "width": options.get("width", 1),
            },
        )

    def polygon(self, coordinates: tuple, **options: Any) -> None:
        self._paint(
            "polygon",
            tuple(value for point in coordinates for value in point),
            {
                "fill": options.get("fill", ""),
                "outline": options.get("outline", ""),
                "width": options.get("width", 1),
                "stipple": options.get("stipple", ""),
            },
        )

    def textlength(self, text: str, *, font: Any) -> float:
        return float(self._tk_font(font).measure(text))

    def text(self, coordinates: tuple, text: str, **options: Any) -> None:
        anchor = "center" if options.get("anchor") == "mm" else "nw"
        self._paint(
            "text",
            tuple(coordinates),
            {
                "text": text,
                "fill": options.get("fill", ""),
                "font": self._tk_font(options["font"]),
                "anchor": anchor,
            },
        )


class MapCanvasPresenter:
    """One view-dependent cloud photo with bounded retained route/trail items."""

    def __init__(self, canvas: Any) -> None:
        self.canvas = canvas
        self.draw = VideoCanvasDraw(canvas, tag="map-overlay")
        self.photo = None
        self._key: tuple | None = None
        self._item = None

    def present(
        self, key: tuple, base: Callable[[], Image.Image], context: MapRenderContext
    ) -> None:
        from PIL import ImageTk

        if key != self._key:
            image = base()
            if self.photo is not None and (self.photo.width(), self.photo.height()) == image.size:
                self.photo.paste(image)
            else:
                self.photo = ImageTk.PhotoImage(image, master=self.canvas)
                if self._item is not None:
                    self.canvas.itemconfigure(self._item, image=self.photo)
            if self._item is None:
                # Replace any startup/export photo without disturbing controls.
                self.canvas.delete("frame")
                self._item = self.canvas.create_image(
                    0,
                    0,
                    image=self.photo,
                    anchor="nw",
                    tags="frame",
                )
                self.canvas.tag_lower(self._item)
            self._key = key
        self.draw.begin()
        draw_map_overlays(self.draw, context)
        self.draw.end()


class VideoCanvasPresenter:
    """One fitted frame and one Tk photo, independently of changing overlays."""

    def __init__(self, canvas: Any) -> None:
        self.canvas = canvas
        self.draw = VideoCanvasDraw(canvas)
        self.photo = None
        self._key: tuple | None = None
        self._source = None
        self._item = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._image_size = (0, 0)
        self.has_frame = False

    def present(self, source: Any, stamp: float, width: int, height: int) -> tuple[float, int, int]:
        from PIL import ImageTk

        source_size = None
        if isinstance(source, Image.Image):
            source_size = source.size
        elif isinstance(source, np.ndarray) and source.ndim >= 2:
            source_size = (int(source.shape[1]), int(source.shape[0]))
        fitted_size = (width, height)
        if source_size is not None and min(source_size) > 0:
            scale = min(width / source_size[0], height / source_size[1])
            fitted_size = (
                max(1, int(source_size[0] * scale)),
                max(1, int(source_size[1] * scale)),
            )
        key = (id(source), stamp, source_size, fitted_size)
        if key != self._key or (not self.has_frame and self._image_size != (width, height)):
            frame, self._scale = prepare_video_frame(source, width, height)
            self.has_frame = frame is not None
            if frame is None:
                frame = Image.new("RGB", (width, height), "#08090b")
            if self.photo is not None and self._image_size == frame.size:
                self.photo.paste(frame)
            else:
                self.photo = ImageTk.PhotoImage(frame, master=self.canvas)
                if self._item is not None:
                    self.canvas.itemconfigure(self._item, image=self.photo)
            self._image_size = frame.size
            self._key = key
            # Keep the source alive so object-id reuse cannot alias a held frame.
            self._source = source
        offset = ((width - self._image_size[0]) // 2, (height - self._image_size[1]) // 2)
        if self._item is None:
            self._item = self.canvas.create_image(
                *offset,
                image=self.photo,
                anchor="nw",
                tags="video-frame",
            )
            self.canvas.tag_lower(self._item)
        elif offset != self._offset:
            self.canvas.coords(self._item, *offset)
        self._offset = offset
        return self._scale, *self._offset


def draw_video_empty_state(
    draw: Any,
    width: int,
    height: int,
    overlay_font: Any,
    live_backend: bool,
) -> None:
    """Paint the placeholder shown before a video frame arrives."""
    for x in range(0, width, 64):
        draw.line((x, 0, x, height), fill="#15181c")
    for y in range(0, height, 64):
        draw.line((0, y, width, y), fill="#15181c")
    draw.text((28, 28), "No video stream", fill="#f0f3f5", font=overlay_font)
    if live_backend:
        draw.text(
            (28, 52),
            "Waiting for live PDRAW frames…",
            fill="#a7b0b8",
            font=overlay_font,
        )
    else:
        draw.text(
            (28, 52),
            "Use --video <file.mp4>",
            fill="#a7b0b8",
            font=overlay_font,
        )


def format_latency_text(latency_ms: float | None) -> str:
    """Format a latency value for HUD display. Pure, no Tk dependency."""
    if latency_ms is None:
        return "-"
    try:
        value = float(latency_ms)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.1f}ms"


def classify_localization_health(loc: dict) -> str:
    """Pure health classification for a localization result. Returns OK/DEGRADED/LOST.

    * OK       – high-confidence fix (enough inliers, low reproj, not weak).
    * DEGRADED – low inliers / high reproj / weak (still a pose but not trusted).
    * LOST     – no pose / success==False.
    No Tk, no self, no mutation — safe to import headless.
    """
    if not hasattr(loc, "get"):
        return "LOST"
    if not loc.get("success"):
        return "LOST"
    try:
        inliers = int(loc.get("inliers", 0) or 0)
    except (TypeError, ValueError):
        inliers = 0
    reproj = loc.get("reproj_rms")
    try:
        weak = localization_result_is_weak(loc)
    except (AttributeError, KeyError, TypeError, ValueError):
        weak = bool(loc.get("weak"))
    if inliers < LOC_LOW_INLIERS:
        return "DEGRADED"
    if reproj is not None:
        try:
            if float(reproj) > LOC_HIGH_REPROJ:
                return "DEGRADED"
        except (TypeError, ValueError):
            pass
    if weak:
        return "DEGRADED"
    return "OK"


#: Map-trail colours by estimate source. Strong fixes keep the blue history
#: line; weak fixes draw hollow diamonds so shape already says "estimate":
#: KLT (optical-flow bridge) green, IMU (fused-yaw bridge) yellow, anything
#: else falls back to the health amber.
WEAK_KIND_COLOR = {
    "KLT": "#3fbf7f",
    "IMU": "#ffd60a",
}
WEAK_KIND_FALLBACK_COLOR = "#e0a92e"


def weak_pose_source_kind(result: Any) -> str:
    """Classify a weak localization result as KLT / IMU / OTHER.

    Pure, no Tk dependency — safe to import headless. Anything unrecognized
    is OTHER so a new estimator can never break the map draw.
    """
    try:
        get = result.get
    except AttributeError:
        return "OTHER"
    try:
        if get("candidate_mode") in ("klt_bridge", "klt_fast"):
            return "KLT"
        if get("pose_status") == "KLT_BRIDGED":
            return "KLT"
        if get("direct_status") in ("FAST_TRACK", "VO_ONLY", "DEAD_RECKON"):
            return "KLT"
        if get("direct_status") == "IMU_BRIDGE" or bool(get("imu_bridge")):
            return "IMU"
    except (AttributeError, KeyError, TypeError, ValueError):
        return "OTHER"
    return "OTHER"


def build_hud_overlay_data(state: Any, loc: Any, telemetry: Any) -> dict:
    """Pure HUD overlay builder. Inputs are not mutated. Returns HUD data dict."""
    # Normalize loc to dict copy
    if isinstance(loc, dict):
        loc_dict = dict(loc)
    elif loc is None:
        loc_dict = {}
    else:
        try:
            loc_dict = {
                k: getattr(loc, k)
                for k in (
                    "success",
                    "inliers",
                    "reproj_rms",
                    "mode",
                    "next_mode",
                    "weak",
                    "health",
                    "loc_health",
                )
                if hasattr(loc, k)
            }
        except (AttributeError, KeyError, TypeError, ValueError):
            loc_dict = {}
    # Normalize state
    if isinstance(state, dict):
        state_dict = dict(state)
    elif state is None:
        state_dict = {}
    else:
        state_dict = {}
        for key in (
            "mode",
            "stream",
            "loc",
            "tracker_state",
            "pose",
            "inliers",
            "reproj",
            "link_ok",
            "gps_fixed",
            "frame_age_ms",
            "flight_state",
            "is_live",
        ):
            try:
                if hasattr(state, key):
                    state_dict[key] = getattr(state, key)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    # Normalize telemetry
    if isinstance(telemetry, dict):
        tele_dict = dict(telemetry)
    elif telemetry is None:
        tele_dict = {}
    else:
        for key in (
            "rth",
            "gps",
            "attitude",
            "velocity",
            "altitude_agl",
            "link_quality",
            "current_auto_speed",
            "olympe_state",
            "olympe_attitude",
            "olympe_altitude",
        ):
            try:
                if hasattr(telemetry, key):
                    tele_dict[key] = getattr(telemetry, key)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    # Health
    health = None
    if isinstance(loc, dict) and "health" in loc:
        health = str(loc["health"])
    elif isinstance(loc, dict) and "loc_health" in loc:
        health = str(loc["loc_health"])
    elif loc_dict:
        if "success" in loc_dict or "inliers" in loc_dict:
            health = classify_localization_health(loc_dict)
        else:
            health = str(loc_dict.get("health", loc_dict.get("loc_health", "OK")))
    else:
        health = "OK"
    # Inliers / reproj
    try:
        inliers = int(loc_dict.get("inliers", loc_dict.get("loc_health_inliers", 0)) or 0)
    except (TypeError, ValueError):
        inliers = 0
    reproj = loc_dict.get("reproj_rms", loc_dict.get("loc_health_reproj"))
    # FPS
    fps_raw = loc_dict.get("fps", loc_dict.get("loc_fps", loc_dict.get("localization_fps")))
    try:
        fps_text = (
            f"{float(fps_raw):.1f}"
            if fps_raw is not None and math.isfinite(float(fps_raw))
            else "-"
        )
    except (TypeError, ValueError):
        fps_text = "-"
    fps_label = str(loc_dict.get("fps_label") or "定位 FPS")
    # Latencies
    latency_raw = loc_dict.get(
        "latency_ms",
        loc_dict.get("loc_latency_ms", loc_dict.get("core_wall_ms", loc_dict.get("wall_ms"))),
    )
    wall_raw = loc_dict.get("wall_ms", loc_dict.get("loc_wall_ms"))
    e2e_raw = loc_dict.get(
        "e2e_ms", loc_dict.get("loc_e2e_ms", loc_dict.get("e2e_submit_to_ui_ms"))
    )
    latency_text = format_latency_text(latency_raw)
    wall_text = format_latency_text(wall_raw)
    e2e_text = format_latency_text(e2e_raw)
    # Telemetry fallbacks
    auto_speed = tele_dict.get(
        "current_auto_speed",
        tele_dict.get("auto_speed", tele_dict.get("current_auto_speed_var", "AUTO PCMD -")),
    )
    attitude = tele_dict.get("attitude", tele_dict.get("olympe_attitude", "飛控融合姿態 -"))
    velocity = tele_dict.get("velocity", tele_dict.get("olympe_velocity", "三軸速度 -"))
    altitude_agl = tele_dict.get(
        "altitude_agl", tele_dict.get("olympe_altitude", "飛控高度 - | AGL -")
    )
    olympe_attitude_val = tele_dict.get("olympe_attitude", None)
    olympe_altitude_val = tele_dict.get("olympe_altitude", None)
    # RTH, GPS, satellite and link-quality figures are deliberately absent from the
    # video HUD (they stay in the header chips and logs).
    altitude_line = str(olympe_altitude_val) if olympe_altitude_val is not None else str(altitude_agl)
    if olympe_attitude_val is not None:
        line1 = " | ".join(
            part
            for part in (str(auto_speed), f"{fps_label} {fps_text}", f"inliers {inliers}")
            if part
        )
        line2 = f"wall_ms {wall_text} | core {latency_text} | e2e {e2e_text}"
        att = str(olympe_attitude_val)
        if " | 三軸速度 " in att:
            att_part, sep, vel_part = att.partition(" | 三軸速度 ")
            attitude_line = att_part
            velocity_line = f"三軸速度 {vel_part}" if sep else str(velocity)
        else:
            attitude_line = att
            velocity_line = (
                str(velocity) if str(velocity).startswith("三軸") else f"三軸速度 {velocity}"
            )
        diagnostic_lines = (line1, line2, altitude_line, attitude_line, velocity_line)
    else:
        diagnostic_lines = (
            " | ".join((str(auto_speed), f"{fps_label} {fps_text}", f"inliers {inliers}")),
            f"wall_ms {wall_text} | core {latency_text} | e2e {e2e_text}",
            altitude_line,
            str(attitude),
            str(velocity) if str(velocity).startswith("三軸") else f"三軸速度 {velocity}",
        )
    health_text_map = {
        "OK": "定位正常",
        "LOW": f"定位信心低 inliers={inliers}",
        "DEGRADED": f"定位信心低 inliers={inliers}",
        "FAIL": "定位失敗",
        "LOST": "定位失敗",
        "PAUSED_ZOOM": "定位暫停：相機縮放未校正",
    }
    health_text = health_text_map.get(str(health), str(health))
    ok_count = loc_dict.get("ok_count", loc_dict.get("_loc_ok_count", loc_dict.get("loc_ok_count")))
    fail_count = loc_dict.get(
        "fail_count", loc_dict.get("_loc_fail_count", loc_dict.get("loc_fail_count"))
    )
    if ok_count is not None or fail_count is not None:
        try:
            oc = int(ok_count or 0)
            fc = int(fail_count or 0)
            if oc or fc:
                health_text += f" | ok={oc} fail={fc}"
        except (TypeError, ValueError):
            pass
    return {
        "health": str(health),
        "health_text": health_text,
        "inliers": inliers,
        "reproj_rms": reproj,
        "fps": fps_text,
        "latency_text": latency_text,
        "wall_text": wall_text,
        "e2e_text": e2e_text,
        "diagnostic_lines": diagnostic_lines,
        "state": dict(state_dict),
        "telemetry": dict(tele_dict),
    }


def heading_arrow_polygon(
    sx: float,
    sy: float,
    hx: float,
    hy: float,
    *,
    length: float = 18.0,
    head_width: float = 8.0,
    tail_length: float = 7.0,
) -> list[tuple[float, float]] | None:
    """Return a screen-space arrow whose tip points from current pose to heading."""
    dx, dy = float(hx - sx), float(hy - sy)
    norm = math.hypot(dx, dy)
    if norm < 1e-3:
        return None
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    tip = (sx + ux * length, sy + uy * length)
    head_base = (sx + ux * 2.0, sy + uy * 2.0)
    tail = (sx - ux * tail_length, sy - uy * tail_length)
    tail_half_width = head_width * 0.42
    return [
        tip,
        (head_base[0] + px * head_width, head_base[1] + py * head_width),
        (tail[0] + px * tail_half_width, tail[1] + py * tail_half_width),
        (tail[0] - px * tail_half_width, tail[1] - py * tail_half_width),
        (head_base[0] - px * head_width, head_base[1] - py * head_width),
    ]


_HudChoice = tuple[Any, Any, int, int, list[list[int]], list[Any], int]
_HudLayout = tuple[int, int, list[tuple[int, int, int, Any]]]


class _HudMeasure:
    """Per-draw text measurement caches shared across HUD layout passes."""

    def __init__(self, draw: Any, row_count: int) -> None:
        self.draw = draw
        # Result metrics change one or two rows per completion. Re-measuring every
        # unchanged CJK telemetry row at every candidate size dominated actual Tk
        # rendering in the offline profile. Keep only the last text per row/font,
        # not an unbounded history of changing metric strings.
        measured: tuple[int, dict[tuple[str, int], tuple[str, float]], dict[str, int]] | None = (
            getattr(draw, "_hud_measure_cache", None)
        )
        if measured is None or measured[0] != row_count:
            measured = (row_count, {}, {})
            if hasattr(draw, "_hud_measure_cache"):
                draw._hud_measure_cache = measured
        self.widths_by_row = measured[1]
        self.font_heights = measured[2]

    def text_width(self, index: int, text: str, font: Any) -> float:
        row_key = (str(font), index)
        previous = self.widths_by_row.get(row_key)
        if previous is None or previous[0] != text:
            previous = (text, self.draw.textlength(text, font=font))
            self.widths_by_row[row_key] = previous
        return previous[1]

    def sized(self, font: Any, size: int) -> Any:
        factory = getattr(self.draw, "font", None)
        if callable(factory):
            return factory(size, bold=True)
        variant = getattr(font, "font_variant", None)
        return variant(size=size) if callable(variant) else font

    def line_height(self, font: Any) -> int:
        # VideoCanvasDraw owns these fonts; their size/style never mutate.
        font_key = str(font)
        if font_key not in self.font_heights:
            metrics = getattr(font, "metrics", None)
            if callable(metrics):
                self.font_heights[font_key] = int(metrics("linespace"))
            else:
                bbox = font.getbbox("Ag國")
                self.font_heights[font_key] = max(1, int(bbox[3]) + 2)
        return self.font_heights[font_key]


def _hud_fit_size(
    measure: _HudMeasure,
    size: int,
    source_line: str,
    diagnostic_lines: Sequence[str],
    hud_font: Any,
    main_hud_font: Any,
    available_width: int,
    available_height: int,
    gap: int,
) -> _HudChoice | None:
    font = measure.sized(hud_font, size)
    main_font = measure.sized(main_hud_font, size + 2)
    row_h, source_h = measure.line_height(font), measure.line_height(main_font)
    # Reserve width for source-age/FPS digits, and include the source row in
    # the layout key so unusually long fresh/stale readouts still reflow.
    if measure.text_width(-1, source_line, main_font) + 40 > available_width:
        main_font = font
        source_h = row_h
    indices = list(range(len(diagnostic_lines)))
    for column_count in range(1, 4):
        rows = max(1, math.ceil(len(indices) / column_count))
        groups = [indices[start : start + rows] for start in range(0, len(indices), rows)]
        needed = source_h + rows * row_h
        if needed > available_height:
            continue
        widths = [
            max(
                (measure.text_width(index, diagnostic_lines[index], font) for index in group),
                default=0,
            )
            for group in groups
        ]
        if sum(widths) + gap * max(0, len(groups) - 1) <= available_width:
            return (font, main_font, row_h, source_h, groups, widths, needed)
    return None


def _hud_scale_only_layout(
    measure: _HudMeasure,
    source_line: str,
    diagnostic_lines: Sequence[str],
    scale_indices: list[int],
    hud_font: Any,
    left: int,
    bottom: int,
    available_width: int,
    available_height: int,
    gap: int,
) -> _HudLayout | None:
    # Two vertical rows may not fit beneath the link-loss banner. Use the
    # remaining band horizontally for freshness and scale, at the largest
    # readable font that fits; do not discard scale to save a row.
    for size in range(18, 7, -1):
        priority_font = measure.sized(hud_font, size)
        priority_h = measure.line_height(priority_font)
        source_width = measure.text_width(-1, source_line, priority_font) + 40
        scale_widths = [
            measure.text_width(index, diagnostic_lines[index], priority_font)
            for index in scale_indices
        ]
        if not (
            priority_h <= available_height
            and source_width + sum(scale_widths) + gap * len(scale_indices) <= available_width
        ):
            continue
        top = bottom - priority_h
        placements = [(left, top, -1, priority_font)]
        x = left + source_width + gap
        for index, scale_width in zip(scale_indices, scale_widths, strict=True):
            placements.append((int(x), top, index, priority_font))
            x += scale_width + gap
        return (top, bottom, placements)
    return None


def _hud_truncated_choice(
    font: Any,
    row_h: int,
    scale_indices: list[int],
    line_count: int,
    available_width: int,
    available_height: int,
) -> _HudChoice:
    priority = scale_indices + [index for index in range(line_count) if index not in scale_indices]
    rows = max(0, available_height // row_h - 1)
    needed = min(available_height, row_h + rows * row_h)
    return (font, font, row_h, row_h, [priority[:rows]], [available_width], needed)


def _hud_place_rows(
    chosen: _HudChoice, left: int, bottom: int, minimum_top: int, gap: int
) -> _HudLayout:
    font, main_font, row_h, source_h, groups, widths, needed = chosen
    top = max(minimum_top, bottom - needed)
    placements = [(left, top, -1, main_font)]
    x = left
    for group, column_width in zip(groups, widths, strict=True):
        for row, index in enumerate(group):
            placements.append((int(x), top + source_h + row * row_h, index, font))
        x += column_width + gap
    return (top, bottom, placements)


def _store_hud_layout(draw: Any, key: tuple, result: _HudLayout) -> _HudLayout:
    if hasattr(draw, "_hud_layout_cache"):
        draw._hud_layout_cache = (key, result)
    return result


def _video_hud_layout(
    draw: Any,
    width: int,
    height: int,
    source_line: str,
    diagnostic_lines: Sequence[str],
    hud_height: int,
    hud_font: Any,
    main_hud_font: Any,
    banner_bottom: int,
) -> tuple[int, int, list[tuple[int, int, int, Any]]]:
    """Fit readable rows inside the panel, reserving active alert-banner space."""
    key = (width, height, source_line, tuple(diagnostic_lines), hud_height, banner_bottom)
    cached: tuple[tuple, _HudLayout] | None = getattr(draw, "_hud_layout_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    measure = _HudMeasure(draw, len(diagnostic_lines))
    margin = 18 if height >= hud_height + 36 else 6
    if banner_bottom and height - banner_bottom < 48:
        margin = 0
    bottom = height - margin - 1  # Reserve the text shadow's one-pixel offset.
    minimum_top = max(margin, banner_bottom)
    available_height = max(0, bottom - minimum_top)
    left, right, gap = 28, width - 28, 20
    available_width = max(1, right - left)
    chosen = None
    for size in range(18, 7, -1):
        chosen = _hud_fit_size(
            measure,
            size,
            source_line,
            diagnostic_lines,
            hud_font,
            main_hud_font,
            available_width,
            available_height,
            gap,
        )
        if chosen is not None:
            break
    if chosen is None:
        # A 96px pane with a 72px safety banner cannot show every engineering
        # metric. Keep source freshness and map-wide scale ahead of secondary
        # device telemetry; never paint text under the banner or past the pane.
        font = measure.sized(hud_font, 8)
        row_h = measure.line_height(font)
        scale_indices = [
            index for index, line in enumerate(diagnostic_lines) if line.startswith("全場")
        ]
        if scale_indices and available_height < 2 * row_h:
            result = _hud_scale_only_layout(
                measure,
                source_line,
                diagnostic_lines,
                scale_indices,
                hud_font,
                left,
                bottom,
                available_width,
                available_height,
                gap,
            )
            if result is not None:
                return _store_hud_layout(draw, key, result)
        chosen = _hud_truncated_choice(
            font, row_h, scale_indices, len(diagnostic_lines), available_width, available_height
        )
    return _store_hud_layout(draw, key, _hud_place_rows(chosen, left, bottom, minimum_top, gap))


def video_hud_summary_text(state: Any, live_backend: bool, stream_latency_ms: float | None) -> str:
    """The HUD's first row: stream fps, frame freshness and gimbal/zoom.

    Shared with the video dirty key so a repaint is requested exactly when
    this text can change, not on every sub-display-precision jitter.
    """
    age = getattr(state, "frame_age_ms", None)
    if age is None and live_backend:
        age = getattr(state, "link_latency_ms", None)
    if age is None:
        age_text = (
            f"~{stream_latency_ms:.0f} ms (profile)"
            if not live_backend and stream_latency_ms is not None
            else "-"
        )
    else:
        age_text = f"{float(age):.0f} ms"
    fps = float(getattr(state, "stream_fps", 0.0) or 0.0)
    fps_text = f"{fps:.1f}" if fps > 0.05 else "-"
    return (
        f"影像 {fps_text} FPS · 影格新鮮度 {age_text} · "
        f"鏡頭 {state.gimbal_pitch_deg:.0f}° / {state.zoom:.1f}x"
    )


def draw_video_hud(
    draw: Any,
    width: int,
    height: int,
    state: Any,
    diagnostic_lines: Sequence[str],
    hud_height: int,
    hud_font: Any,
    main_hud_font: Any,
    overlay_font: Any,
    live_backend: bool,
    stream_latency_ms: float | None,
    *,
    banner_bottom: int = 0,
) -> bool:
    """Paint the border and transparent engineering HUD, returning link status."""
    draw.rectangle(
        (18, 18, width - 18, height - 18),
        outline="#363c44",
        width=2,
    )
    summary_text = video_hud_summary_text(state, live_backend, stream_latency_ms)
    link_ok = bool(getattr(state, "link_ok", True))
    _, _, placements = _video_hud_layout(
        draw,
        width,
        height,
        summary_text,
        diagnostic_lines,
        hud_height,
        hud_font,
        main_hud_font,
        banner_bottom,
    )
    for x, y, index, font in placements:
        text = summary_text if index < 0 else diagnostic_lines[index]
        draw.text((x + 1, y + 1), text, fill="#000000", font=font)
        draw.text(
            (x, y),
            text,
            fill="#f4f4f4",
            font=font,
        )
    return link_ok


def draw_video_banner(
    draw: Any,
    width: int,
    live_backend: bool,
    link_ok: bool,
    lost_holding: bool,
    lost_frame_name: str,
    lost_frame_index: object,
    lost_attempts: int,
    lost_max_attempts: int,
    inspecting: bool,
    health: str,
    health_inliers: int,
    health_reproj: float | None,
    health_color: Mapping[str, str],
    banner_font: Any,
    overlay_font: Any,
) -> None:
    """Paint the highest-priority display-only alert banner."""
    if live_backend and not link_ok:
        draw.rectangle((18, 20, width - 18, 72), fill="#c41e3a")
        draw.text(
            (width // 2, 38),
            "LINK LOST — 連線中斷",
            fill="#ffffff",
            anchor="mm",
            font=banner_font,
        )
        draw.text(
            (width // 2, 58),
            "指令可能送不出去 · 勿依賴本畫面控機",
            fill="#ffd7dc",
            anchor="mm",
            font=overlay_font,
        )
    elif lost_holding:
        draw.rectangle((18, 20, width - 18, 72), fill="#e0a92e")
        draw.text(
            (width // 2, 38),
            "LOST — 串流暫停，MegaLoc 重定位中",
            fill="#0b0c0e",
            anchor="mm",
            font=banner_font,
        )
        draw.text(
            (width // 2, 58),
            f"凍結於 {lost_frame_name or lost_frame_index} · "
            f"重試 {lost_attempts}/{lost_max_attempts}",
            fill="#0b0c0e",
            anchor="mm",
            font=overlay_font,
        )
    elif inspecting and health != "OK":
        color = health_color.get(health, health_color["LOW"])
        if health == "FAIL":
            message = "LOCALIZATION LOST"
        elif health == "PAUSED_ZOOM":
            message = "LOCALIZATION PAUSED — 相機縮放未校正，回到 1.0x 恢復"
        else:
            reprojection = "-" if health_reproj is None else f"{health_reproj:.1f}"
            message = f"LOW CONFIDENCE  inliers={health_inliers} reproj={reprojection}"
        draw.rectangle((18, 20, width - 18, 60), fill=color)
        draw.text(
            (width // 2, 40),
            message,
            fill="#0b0c0e",
            anchor="mm",
            font=banner_font,
        )


def draw_magnetometer_axis(canvas: Any, guide: tuple[str, str, str] | None) -> None:
    """Draw the rotation the firmware is currently asking for.

    ``guide`` is magnetometer_axis_guide()'s result, or None when no axis is
    being requested -- in which case the canvas is cleared rather than left
    showing a stale rotation the operator should not perform. Rotation counts
    are not completion evidence: keep rotating until the firmware changes axis.
    """
    if canvas is None:
        return
    canvas.delete("all")
    if guide is None:
        canvas.create_text(
            8, 26, anchor="w", text="（目前沒有要求校正軸）", fill=MUTED, font=FONT_SMALL
        )
        return
    label, view, instruction = guide
    body, accent, arrow = BORDER, TEXT, ACCENT
    cx, cy = 30, 26

    if view == "top":  # X/roll: seen from above, roll about nose-tail
        canvas.create_oval(cx - 20, cy - 7, cx + 20, cy + 7, fill=body, outline=accent)
        canvas.create_polygon(
            cx + 20, cy, cx + 28, cy - 4, cx + 28, cy + 4, fill=accent, outline=accent
        )
        canvas.create_line(cx - 27, cy, cx + 30, cy, fill=arrow, dash=(3, 2))
        canvas.create_arc(
            cx - 12,
            cy - 20,
            cx + 12,
            cy + 20,
            start=20,
            extent=300,
            style="arc",
            outline=arrow,
            width=2,
        )
        canvas.create_polygon(
            cx + 11, cy - 16, cx + 17, cy - 7, cx + 5, cy - 8, fill=arrow, outline=arrow
        )
    elif view == "side":  # Y/pitch: seen from the left, nose over the top
        canvas.create_oval(cx - 21, cy - 6, cx + 21, cy + 6, fill=body, outline=accent)
        canvas.create_polygon(
            cx + 21, cy, cx + 29, cy - 4, cx + 29, cy + 4, fill=accent, outline=accent
        )
        canvas.create_line(cx, cy - 22, cx, cy + 22, fill=arrow, dash=(3, 2))
        canvas.create_arc(
            cx - 24,
            cy - 19,
            cx + 24,
            cy + 19,
            start=200,
            extent=140,
            style="arc",
            outline=arrow,
            width=2,
        )
        canvas.create_polygon(
            cx + 20, cy - 9, cx + 27, cy - 1, cx + 15, cy - 1, fill=arrow, outline=arrow
        )
    else:  # Z/yaw: seen from above, spin flat
        canvas.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, fill=body, outline=accent)
        canvas.create_polygon(
            cx, cy - 14, cx - 5, cy - 22, cx + 5, cy - 22, fill=accent, outline=accent
        )
        canvas.create_arc(
            cx - 22,
            cy - 22,
            cx + 22,
            cy + 22,
            start=45,
            extent=250,
            style="arc",
            outline=arrow,
            width=2,
        )
        canvas.create_polygon(
            cx + 14, cy - 13, cx + 22, cy - 7, cx + 12, cy - 4, fill=arrow, outline=arrow
        )

    canvas.create_text(
        66,
        2,
        anchor="nw",
        text=f"現在請轉：{label}",
        fill=arrow,
        font=FONT_BOLD,
        width=146,
    )
    canvas.create_text(
        66,
        44,
        anchor="nw",
        text=instruction,
        fill=MUTED,
        font=FONT_SMALL,
        width=146,
    )
