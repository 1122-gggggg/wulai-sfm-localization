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
    if context.camera_axes is not None:
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
        built = _basis_from_forward(context, forward)
        return (*built, False) if built is not None else None
    if context.camera_forward is not None:
        built = _basis_from_forward(context, context.camera_forward)
        return (*built, False) if built is not None else None
    try:
        yaw_value = float(context.pose[3])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(yaw_value):
        return None
    try:
        pitch_value = math.radians(float(context.gimbal_pitch_deg))
    except (TypeError, ValueError):
        pitch_value = 0.0
    if not math.isfinite(pitch_value):
        pitch_value = 0.0
    pitch_value = max(-math.pi / 2 + 1e-6, min(math.pi / 2 - 1e-6, pitch_value))
    horizontal = math.cos(yaw_value) * np.asarray(context.map_east, dtype=float) + math.sin(
        yaw_value
    ) * np.asarray(context.map_north, dtype=float)
    forward = math.cos(pitch_value) * horizontal + math.sin(pitch_value) * np.asarray(
        context.map_up, dtype=float
    )
    built = _basis_from_forward(context, forward)
    return (*built, True) if built is not None else None


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
    if not seq:
        return 0
    n = len(seq)
    if n <= 32:
        return tuple(tuple(round(float(x), 4) for x in p[:3]) for p in seq)
    step = max(1, n // 16)
    sampled = tuple(tuple(round(float(x), 4) for x in seq[i][:3]) for i in range(0, n, step))
    if (n - 1) % step != 0:
        sampled = sampled + (tuple(round(float(x), 4) for x in seq[-1][:3]),)
    return (n, sampled)

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
        round(float(pan_arr[0]), 4),
        round(float(pan_arr[1]), 4),
    ) if len(pan_arr) >= 2 else (0.0, 0.0)
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

    proj_cache_key = (
        pan_key,
        zoom_key,
        size_key,
        rotation_key,
        round(float(getattr(context, "map_radius", 1.0)), 4),
        _pts_sig(context.route_pts),
        _pts_sig(context.history),
        _pts_sig(weak_trail),
    )
    cached = _PROJECTION_CACHE.get(proj_cache_key)
    if cached is not None:
        screen_x, screen_y = cached
    else:
        parts = []
        if route_n:
            parts.append(np.asarray(context.route_pts, dtype=float).reshape(-1, 3))
        if history_n:
            parts.append(
                np.array(
                    [np.asarray(point, dtype=float)[:3] for point in context.history],
                    dtype=float,
                ).reshape(-1, 3)
            )
        if weak_n:
            parts.append(
                np.array(
                    [np.asarray(point, dtype=float)[:3] for point in weak_trail],
                    dtype=float,
                ).reshape(-1, 3)
            )
        view = context.transform_xyz(np.concatenate(parts, axis=0))
        scale = _map_scale(context)
        screen_x = (context.width * 0.5 + context.map_pan[0] + view[:, 0] * scale).astype(int).tolist()
        screen_y = (context.height * 0.5 + context.map_pan[1] - view[:, 1] * scale).astype(int).tolist()
        if len(_PROJECTION_CACHE) >= _PROJECTION_CACHE_MAX:
            _PROJECTION_CACHE.clear()
        _PROJECTION_CACHE[proj_cache_key] = (screen_x, screen_y)
    if route_n > 1:
        route_screen = list(zip(screen_x[:route_n], screen_y[:route_n]))
        draw.line(route_screen, fill=context.route_color, width=2, joint="curve")
        dot_step = max(1, route_n // max(1, context.route_dot_max))
        for x, y in route_screen[::dot_step]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=context.route_color)

    if history_n > 1:
        history_screen = list(
            zip(screen_x[route_n : route_n + history_n], screen_y[route_n : route_n + history_n])
        )
        draw.line(history_screen, fill="#5aa7e8", width=3)
        # Explicit alignment: history and history_health are parallel lists,
        # but a dropped append would make zip() silently swallow the tail.
        # Draw markers only over the explicitly shared prefix.
        dot_n = min(history_n, len(context.history_health))
        for index in range(dot_n):
            x, y = history_screen[index]
            health = context.history_health[index]
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
        weak_screen = list(zip(screen_x[route_n + history_n :], screen_y[route_n + history_n :]))
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
            size = 6
            draw.polygon(
                [(x, y - size), (x + size, y), (x, y + size), (x - size, y)],
                outline=colour,
                width=2,
            )


def _draw_no_orientation_marker(draw: Any, context: MapRenderContext) -> None:
    """Hollow ring when no orientation source survives: position without bearing."""
    screen_x, screen_y = context.project_world(
        np.asarray(context.pose[:3], dtype=float), context.width, context.height
    )
    draw.ellipse(
        (screen_x - 7, screen_y - 7, screen_x + 7, screen_y + 7),
        outline="#3fbf7f",
        width=3,
    )


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
    border_color = "#3fbf7f" if from_yaw else "#00d4ff"
    drone_pt = (screen_x[0], screen_y[0])
    quad = list(zip(screen_x[1:5], screen_y[1:5]))
    # The gaze line runs under the fill, drone -> picture -> arrowhead: one
    # stem, not a cone of corner lines, and it carries the facing direction
    # even when the quad is seen edge-on.
    draw.line((*drone_pt, screen_x[5], screen_y[5]), fill="#ffffff", width=2)
    draw.polygon(quad, fill="#0a5f73")
    draw.line(quad + [quad[0]], fill=border_color, width=3, joint="curve")
    draw.polygon(
        [(screen_x[5], screen_y[5]), (screen_x[6], screen_y[6]), (screen_x[7], screen_y[7])],
        fill="#ffffff",
    )
    draw.ellipse(
        (drone_pt[0] - 4, drone_pt[1] - 4, drone_pt[0] + 4, drone_pt[1] + 4),
        fill=border_color,
        outline="#ffffff",
        width=2,
    )


def draw_map_overlays(draw: Any, context: MapRenderContext) -> None:
    """Paint dynamic map overlays over an already-rendered map base."""
    _draw_route_and_history(draw, context)
    _draw_camera_overlay(draw, context)


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
    return source.resize(new_size, Image.Resampling.BILINEAR), scale


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
                source_height, source_width = new_size[1], new_size[0]
            frame = Image.fromarray(source, mode="RGB")
            scale = min(
                width / max(1, source_width),
                height / max(1, source_height),
            )
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
                for k in ("success", "inliers", "reproj_rms", "mode", "next_mode", "weak", "health", "loc_health")
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
            "mode", "stream", "loc", "tracker_state", "pose", "inliers",
            "reproj", "link_ok", "gps_fixed", "frame_age_ms", "flight_state", "is_live"
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
            "rth", "gps", "attitude", "velocity", "altitude_agl", "link_quality",
            "current_auto_speed", "olympe_state", "olympe_attitude", "olympe_altitude"
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
        fps_text = f"{float(fps_raw):.1f}" if fps_raw is not None and math.isfinite(float(fps_raw)) else "-"
    except (TypeError, ValueError):
        fps_text = "-"
    # Latencies
    latency_raw = loc_dict.get("latency_ms", loc_dict.get("loc_latency_ms", loc_dict.get("core_wall_ms", loc_dict.get("wall_ms"))))
    wall_raw = loc_dict.get("wall_ms", loc_dict.get("loc_wall_ms"))
    e2e_raw = loc_dict.get("e2e_ms", loc_dict.get("loc_e2e_ms", loc_dict.get("e2e_submit_to_ui_ms")))
    latency_text = format_latency_text(latency_raw)
    wall_text = format_latency_text(wall_raw)
    e2e_text = format_latency_text(e2e_raw)
    # Telemetry fallbacks
    auto_speed = tele_dict.get("current_auto_speed", tele_dict.get("auto_speed", tele_dict.get("current_auto_speed_var", "AUTO 地速安全閘門 -")))
    rth = tele_dict.get("rth", tele_dict.get("olympe_state", "RTH ?/?"))
    gps = tele_dict.get("gps", "")
    attitude = tele_dict.get("attitude", tele_dict.get("olympe_attitude", "飛控融合姿態 -"))
    velocity = tele_dict.get("velocity", tele_dict.get("olympe_velocity", "三軸速度 -"))
    altitude_agl = tele_dict.get("altitude_agl", tele_dict.get("olympe_altitude", "飛控高度 - | AGL -"))
    link_quality = tele_dict.get("link_quality", "")
    olympe_state_val = tele_dict.get("olympe_state", None)
    olympe_attitude_val = tele_dict.get("olympe_attitude", None)
    olympe_altitude_val = tele_dict.get("olympe_altitude", None)
    if olympe_state_val is not None or olympe_attitude_val is not None:
        line1 = " | ".join(part for part in (str(auto_speed), f"定位 FPS {fps_text}", f"inliers {inliers}") if part)
        line2 = f"wall_ms {wall_text} | core {latency_text} | e2e {e2e_text}"
        line3 = str(olympe_state_val) if olympe_state_val is not None else (f"{rth} | {gps}".strip(" |") if gps else str(rth))
        line4 = str(olympe_altitude_val) if olympe_altitude_val is not None else (f"{altitude_agl} | {link_quality}".strip(" |") if link_quality else str(altitude_agl))
        att = str(olympe_attitude_val) if olympe_attitude_val is not None else str(attitude)
        if " | 三軸速度 " in att:
            att_part, sep, vel_part = att.partition(" | 三軸速度 ")
            attitude_line = att_part
            velocity_line = f"三軸速度 {vel_part}" if sep else str(velocity)
        else:
            attitude_line = att
            velocity_line = str(velocity) if str(velocity).startswith("三軸") else f"三軸速度 {velocity}"
        diagnostic_lines = (line1, line2, line3, line4, attitude_line, velocity_line)
    else:
        diagnostic_lines = (
            " | ".join((str(auto_speed), f"定位 FPS {fps_text}", f"inliers {inliers}")),
            f"wall_ms {wall_text} | core {latency_text} | e2e {e2e_text}",
            f"{rth} | {gps}".strip(" |") if gps else str(rth),
            f"{altitude_agl} | {link_quality}".strip(" |") if link_quality else str(altitude_agl),
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
    fail_count = loc_dict.get("fail_count", loc_dict.get("_loc_fail_count", loc_dict.get("loc_fail_count")))
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
    stream_latency_ms: float,
) -> bool:
    """Paint the border and engineering HUD, returning the link status."""
    draw.rectangle(
        (18, 18, width - 18, height - 18),
        outline="#363c44",
        width=2,
    )
    hud_top = max(18, height - hud_height)
    age = getattr(state, "frame_age_ms", None)
    if age is None and live_backend:
        age = getattr(state, "link_latency_ms", None)
    if age is None:
        age_text = f"~{stream_latency_ms:.0f} ms (profile)" if not live_backend else "-"
    else:
        age_text = f"{float(age):.0f} ms"
    fps = float(getattr(state, "stream_fps", 0.0) or 0.0)
    fps_text = f"{fps:.1f}" if fps > 0.05 else "-"
    link_ok = bool(getattr(state, "link_ok", True))
    hud_lines = (
        f"影像 {fps_text} FPS · 影格新鮮度 {age_text} · "
        f"鏡頭 {state.gimbal_pitch_deg:.0f}° / {state.zoom:.1f}x",
        *diagnostic_lines,
    )
    line_spacing = max(
        17,
        (height - 26 - hud_top) // len(hud_lines),
    )
    # One solid strip behind plain (stroke-free) text: stroked truetype at
    # 13-15px cost ~44 ms per 960x540 frame on the Tk thread and capped the
    # whole UI tick near 12 Hz; the strip keeps contrast over bright video
    # for ~10 ms total. Display-only; localization input is untouched.
    strip_bottom = min(height - 18, hud_top + 5 + len(hud_lines) * line_spacing)
    draw.rectangle((18, hud_top, width - 18, strip_bottom), fill="#0b0c0e")
    for index, line in enumerate(hud_lines):
        draw.text(
            (28, hud_top + 5 + index * line_spacing),
            line,
            fill="#f4f4f4",
            font=main_hud_font if index == 0 else hud_font,
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
