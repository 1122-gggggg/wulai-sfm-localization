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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


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
    collision_center: Sequence[float] | None
    collision_radius: float
    collision_status: str
    collision_point: Sequence[float] | None
    collision_preview: bool
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
    return (
        min(context.width, context.height)
        * 0.46
        * context.map_zoom
        / context.map_radius
    )


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
    horizontal = (
        math.cos(yaw_value) * np.asarray(context.map_east, dtype=float)
        + math.sin(yaw_value) * np.asarray(context.map_north, dtype=float)
    )
    forward = (
        math.cos(pitch_value) * horizontal
        + math.sin(pitch_value) * np.asarray(context.map_up, dtype=float)
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
    corners: np.ndarray = np.stack([
        center - right * half_width + image_up * half_height,
        center + right * half_width + image_up * half_height,
        center + right * half_width - image_up * half_height,
        center - right * half_width - image_up * half_height,
    ])
    return corners


def _draw_route_and_history(draw: Any, context: MapRenderContext) -> None:
    route_n = len(context.route_pts)
    history_n = len(context.history)
    if not route_n and not history_n:
        return

    parts = []
    if route_n:
        parts.append(np.asarray(context.route_pts, dtype=float).reshape(-1, 3))
    if history_n:
        parts.append(
            np.array(
                [np.asarray(point, dtype=float)[:3] for point in context.history],
                dtype=float,
            )
            .reshape(-1, 3)
        )
    view = context.transform_xyz(np.concatenate(parts, axis=0))
    scale = _map_scale(context)
    screen_x = (
        context.width * 0.5
        + context.map_pan[0]
        + view[:, 0] * scale
    ).astype(int).tolist()
    screen_y = (
        context.height * 0.5
        + context.map_pan[1]
        - view[:, 1] * scale
    ).astype(int).tolist()

    if route_n > 1:
        route_screen = list(zip(screen_x[:route_n], screen_y[:route_n]))
        draw.line(route_screen, fill=context.route_color, width=2, joint="curve")
        dot_step = max(1, route_n // max(1, context.route_dot_max))
        for x, y in route_screen[::dot_step]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=context.route_color)

    if history_n > 1:
        history_screen = list(zip(screen_x[route_n:], screen_y[route_n:]))
        draw.line(history_screen, fill="#5aa7e8", width=3)
        for (x, y), health in zip(history_screen, context.history_health):
            if health and health != "OK":
                draw.ellipse(
                    (x - 4, y - 4, x + 4, y + 4),
                    fill=context.health_color.get(health, "#e0a92e"),
                )


def _draw_collision_guard(draw: Any, context: MapRenderContext) -> None:
    if context.collision_center is None or context.collision_radius <= 0.0:
        return
    center = np.asarray(context.collision_center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        return
    screen_x, screen_y = context.project_world(
        center, context.width, context.height
    )
    radius_px = max(1, int(round(context.collision_radius * _map_scale(context))))
    status = str(context.collision_status)
    color = {
        "CLEAR": "#3fbf7f",
        "COLLISION": "#ff4d4d",
        "PREVIEW_HIT": "#00d4ff",
        "PREVIEW_CLEAR": "#00d4ff",
        "WAITING": "#e0a92e",
        "UNAVAILABLE": "#e0a92e",
        "DISABLED": "#7f8a94",
    }.get(status, "#7f8a94")
    draw.ellipse(
        (
            screen_x - radius_px,
            screen_y - radius_px,
            screen_x + radius_px,
            screen_y + radius_px,
        ),
        outline=color,
        width=3,
    )
    draw.ellipse(
        (screen_x - 5, screen_y - 5, screen_x + 5, screen_y + 5),
        fill=color,
        outline="#ffffff",
        width=1,
    )
    if status in {"COLLISION", "PREVIEW_HIT"} and context.collision_point is not None:
        point = np.asarray(context.collision_point, dtype=float)
        if point.shape == (3,) and np.all(np.isfinite(point)):
            hit_x, hit_y = context.project_world(
                point, context.width, context.height
            )
            hit_color = "#ff4d4d" if status == "COLLISION" else "#e6b94f"
            draw.line((screen_x, screen_y, hit_x, hit_y), fill=hit_color, width=2)
            draw.ellipse(
                (hit_x - 5, hit_y - 5, hit_x + 5, hit_y + 5),
                fill=hit_color,
                outline="#ffffff",
                width=1,
            )
    label = (
        "模擬相機"
        if context.collision_preview
        else "近接命中" if status == "COLLISION" else "相機近接圈"
    )
    draw.text(
        (screen_x + radius_px + 6, screen_y - 8),
        f"{label}  r={context.collision_radius:.3g} u",
        fill=color,
        font=context.overlay_font,
        stroke_width=2,
        stroke_fill="#0b0c0e",
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
    tip = drone + forward * (
        (CAMERA_VIEW_RECT_GAP_PX + CAMERA_VIEW_ARROW_PX) / scale
    )
    head_base = tip - forward * (CAMERA_VIEW_ARROW_HEAD_PX / scale)
    head_half = CAMERA_VIEW_ARROW_HEAD_PX * 0.6 / scale
    # One batched transform for everything: drone, quad corners, arrow tip and
    # arrowhead base -- the same projection the cloud and the route use.
    view = context.transform_xyz(np.stack([
        drone, *corners, tip,
        head_base + right * head_half,
        head_base - right * head_half,
    ]))
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
    _draw_collision_guard(draw, context)
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
        age_text = (
            f"~{stream_latency_ms:.0f} ms (profile)"
            if not live_backend
            else "-"
        )
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
    for index, line in enumerate(hud_lines):
        draw.text(
            (28, hud_top + 5 + index * line_spacing),
            line,
            fill="#000000",
            font=main_hud_font if index == 0 else hud_font,
            stroke_width=1,
            stroke_fill="#f4f4f4",
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
            reprojection = (
                "-" if health_reproj is None else f"{health_reproj:.1f}"
            )
            message = (
                f"LOW CONFIDENCE  inliers={health_inliers} "
                f"reproj={reprojection}"
            )
        draw.rectangle((18, 20, width - 18, 60), fill=color)
        draw.text(
            (width // 2, 40),
            message,
            fill="#0b0c0e",
            anchor="mm",
            font=banner_font,
        )


def draw_gravity_phase_icon(canvas: Any, phase: str | None) -> None:
    """Draw the yaw, pitch, or roll calibration cue on a Tk-like canvas."""
    phase = phase if phase in {"yaw", "pitch", "roll"} else None
    if getattr(canvas, "_gravity_phase_icon", object()) == phase and canvas.find_all():
        return
    canvas.delete("all")
    canvas._gravity_phase_icon = phase
    cx, cy = 36, 32
    body, accent, arrow = "#39424b", "#f0f3f5", "#4ea1ff"
    tag = f"gravity-phase-{phase}" if phase else "gravity-phase-ready"
    if phase is None:
        canvas.create_text(
            cx, cy, text="準備", fill="#7a828a", font=("Sans", 9), tags=(tag,)
        )
        return

    if phase == "yaw":
        canvas.create_oval(
            cx - 9, cy - 18, cx + 9, cy + 18,
            fill=body, outline=accent, tags=(tag, "airframe"),
        )
        canvas.create_polygon(
            cx, cy - 23, cx - 5, cy - 14, cx + 5, cy - 14,
            fill=accent, outline=accent, tags=(tag, "airframe"),
        )
        canvas.create_arc(
            cx - 27, cy - 27, cx + 27, cy + 27,
            start=35, extent=285, style="arc", outline=arrow, width=2,
            tags=(tag, "motion-yaw"),
        )
        canvas.create_polygon(
            cx + 21, cy - 18, cx + 28, cy - 10, cx + 16, cy - 11,
            fill=arrow, outline=arrow, tags=(tag, "motion-yaw"),
        )
        return

    if phase == "pitch":
        canvas.create_oval(
            cx - 22, cy - 7, cx + 17, cy + 7,
            fill=body, outline=accent, tags=(tag, "airframe"),
        )
        canvas.create_polygon(
            cx + 16, cy, cx + 25, cy - 5, cx + 25, cy + 5,
            fill=accent, outline=accent, tags=(tag, "airframe"),
        )
        canvas.create_line(
            cx, cy - 26, cx, cy + 26,
            fill=arrow, width=2, arrow="both", tags=(tag, "motion-pitch"),
        )
        canvas.create_arc(
            cx - 27, cy - 22, cx + 27, cy + 22,
            start=205, extent=130, style="arc", outline=arrow, width=2,
            tags=(tag, "motion-pitch"),
        )
        return

    canvas.create_oval(
        cx - 7, cy - 18, cx + 7, cy + 18,
        fill=body, outline=accent, tags=(tag, "airframe"),
    )
    canvas.create_line(
        cx - 24, cy, cx + 24, cy,
        fill=accent, width=3, tags=(tag, "airframe"),
    )
    canvas.create_line(
        cx - 27, cy, cx + 27, cy,
        fill=arrow, width=2, arrow="both", tags=(tag, "motion-roll"),
    )
    canvas.create_arc(
        cx - 27, cy - 27, cx + 27, cy + 27,
        start=140, extent=80, style="arc", outline=arrow, width=2,
        tags=(tag, "motion-roll"),
    )
