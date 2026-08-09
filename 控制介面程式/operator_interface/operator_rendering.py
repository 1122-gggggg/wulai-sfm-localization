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
    transform_xyz: Callable[[np.ndarray], np.ndarray]
    project_world: Callable[[np.ndarray, int, int], tuple[int, int]]
    camera_frustum_world_points: Callable[..., Sequence[np.ndarray]]
    heading_arrow_polygon: Callable[..., list[tuple[int, int]]]
    route_color: str
    health_color: Mapping[str, str]
    route_dot_max: int
    no_loc_max_markers: int
    video_hfov_deg: float
    video_aspect_ratio: float
    overlay_font: Any


def _map_scale(context: MapRenderContext) -> float:
    return (
        min(context.width, context.height)
        * 0.46
        * context.map_zoom
        / context.map_radius
    )


def _draw_no_localization_markers(draw: Any, context: MapRenderContext) -> None:
    if not context.no_loc_markers:
        return
    view = context.transform_xyz(
        np.asarray(context.no_loc_markers, dtype=float)
    )
    scale = _map_scale(context)
    for point in view:
        x = int(context.width * 0.5 + context.map_pan[0] + point[0] * scale)
        y = int(context.height * 0.5 + context.map_pan[1] - point[1] * scale)
        draw.ellipse(
            (x - 5, y - 5, x + 5, y + 5),
            fill="#ff2a2a",
            outline="#ffffff",
        )


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
            np.array([point[:3] for point in context.history], dtype=float)
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


def _draw_camera_overlay(draw: Any, context: MapRenderContext) -> None:
    x, y, z, yaw = context.pose
    camera_center = np.array([x, y, z], dtype=float)
    screen_x, screen_y = context.project_world(
        camera_center, context.width, context.height
    )
    arrow = None
    arrow_color = "#00d4ff"
    if context.camera_axes is not None:
        frustum = context.camera_frustum_world_points(
            camera_center,
            context.camera_axes,
            length=context.map_radius * 0.015,
            hfov_deg=context.video_hfov_deg,
            aspect_ratio=context.video_aspect_ratio,
        )
        projected = [
            context.project_world(point, context.width, context.height)
            for point in frustum
        ]
        apex, face_center, *face = projected
        for corner in face:
            draw.line((*apex, *corner), fill="#00d4ff", width=2)
        draw.polygon(face, fill="#007f99")
        draw.line(face + [face[0]], fill="#d8fbff", width=2)
        draw.line((*apex, *face_center), fill="#ffffff", width=2)
        draw.ellipse(
            (
                face_center[0] - 2,
                face_center[1] - 2,
                face_center[0] + 2,
                face_center[1] + 2,
            ),
            fill="#ffffff",
        )
        draw.ellipse(
            (screen_x - 3, screen_y - 3, screen_x + 3, screen_y + 3),
            fill="#00d4ff",
        )
    elif context.camera_forward is not None:
        head_x, head_y = context.project_world(
            camera_center + context.camera_forward,
            context.width,
            context.height,
        )
        arrow = context.heading_arrow_polygon(
            screen_x, screen_y, head_x, head_y
        )
    elif np.isfinite(float(yaw)):
        heading = np.array(
            [math.cos(float(yaw)), 0.0, math.sin(float(yaw))],
            dtype=float,
        )
        head_x, head_y = context.project_world(
            camera_center + heading,
            context.width,
            context.height,
        )
        arrow = context.heading_arrow_polygon(
            screen_x, screen_y, head_x, head_y
        )
        arrow_color = "#3fbf7f"
    else:
        draw.ellipse(
            (screen_x - 7, screen_y - 7, screen_x + 7, screen_y + 7),
            outline="#3fbf7f",
            width=3,
        )
        arrow_color = "#3fbf7f"

    if arrow is not None:
        draw.polygon(arrow, fill=arrow_color)
        draw.line(arrow + [arrow[0]], fill="#ffffff", width=2)


def _draw_map_legend(draw: Any, context: MapRenderContext) -> None:
    if not context.no_loc_markers:
        return
    draw.rectangle(
        (4, 4, min(context.width - 4, 420), 26),
        fill="#0b0c0e",
    )
    draw.text(
        (10, 8),
        f"紅點 = 最近無法定位位置（最多 {context.no_loc_max_markers} 處）",
        fill="#ff6a6a",
        font=context.overlay_font,
    )


def draw_map_overlays(draw: Any, context: MapRenderContext) -> None:
    """Paint dynamic map overlays over an already-rendered map base."""
    _draw_no_localization_markers(draw, context)
    _draw_route_and_history(draw, context)
    _draw_camera_overlay(draw, context)
    _draw_map_legend(draw, context)


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

        scale = min(width / source.width, height / source.height)  # type: ignore[attr-defined]
        new_size = (
            max(1, int(source.width * scale)),  # type: ignore[attr-defined]
            max(1, int(source.height * scale)),  # type: ignore[attr-defined]
        )
        frame = source.resize(new_size, Image.Resampling.BILINEAR)  # type: ignore[attr-defined]
        return frame, scale
    except Exception:
        try:
            if not isinstance(source, Image.Image):
                source = Image.fromarray(source, mode="RGB")
            scale = min(source.width / width, source.height / height)
            new_size = (
                max(1, int(source.width * min(width / source.width, height / source.height))),
                max(1, int(source.height * min(width / source.width, height / source.height))),
            )
            frame = source.resize(new_size, Image.Resampling.BILINEAR)
            scale = min(width / source.width, height / source.height)
            return frame, scale
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
