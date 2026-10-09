"""Read-only depth inspection of exactly paired localization captures.

No flight or localization decisions consume these depths. The main video keeps
running; the inspector explicitly displays a captured frame, not a current-frame
measurement. Mesh loading and dense rendering run outside the Tk thread.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


@dataclass(frozen=True)
class DepthCapture:
    frame: object
    name: str
    stamp: float
    center: np.ndarray
    axes: np.ndarray
    meters_per_map_unit: float | None


def paired_capture(result, frames, meters_per_map_unit=None):
    """Reject wrong captures, held poses and attitudes without a full camera basis.

    Weak or unconfirmed fixes are accepted: any frame whose own capture produced a
    finite pose and a full camera attitude can be inspected.
    """
    if result.get("camera_attitude_kind") != "full":
        return None
    try:
        seq = result.get("display_seq")
        frame, name, stamp = frames[seq]
        capture_ns = int(result.get("capture_mono_ns"))
        held_ns = result.get("pose_capture_mono_ns")
        if held_ns is not None and abs(int(held_ns) - capture_ns) > 1000:
            return None
        if result.get("frame_name") != name or abs(capture_ns - round(stamp * 1e9)) > 1000:
            return None
        pose = result.get("pose_raw") or result.get("pose")
        center = np.array([pose[key] for key in ("x", "y", "z")], dtype=float)
        axes = np.array(result.get("camera_axes_world"), dtype=float)
        if (center.shape != (3,) or axes.shape != (3, 3)
                or not np.isfinite(center).all() or not np.isfinite(axes).all()
                or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(axes), 1., atol=1e-5, rtol=0)):
            return None
        if meters_per_map_unit is not None:
            if isinstance(meters_per_map_unit, bool):
                return None
            meters_per_map_unit = float(meters_per_map_unit)
            if not math.isfinite(meters_per_map_unit) or meters_per_map_unit <= 0:
                return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return DepthCapture(frame, name, stamp, center, axes, meters_per_map_unit)


DEPTH_RANGE_M = 30.
VIEW_RENDER_WIDTH = 640
GRID_ROWS = GRID_COLS = 3
STALE_VIEW_S = 2.
DEPTH_VIEW_OFF_TEXT = "深度估計"
DEPTH_VIEW_ON_TEXT = "返回影像"


def depth_to_color(depth, max_m=DEPTH_RANGE_M):
    """RGB depth map: warm = near, cool = far over 0..max_m, black = unknown."""
    import cv2

    valid = np.isfinite(depth) & (depth > 0.)
    near = 1. - np.clip(np.where(valid, depth, max_m) / max_m, 0., 1.)
    color = cv2.cvtColor(cv2.applyColorMap(np.uint8(near * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def map_unit_range(depth):
    """Colour range for unscaled depth: the frame's own 95th percentile, in map units."""
    valid = depth[np.isfinite(depth) & (depth > 0.)]
    return float(np.percentile(valid, 95)) if valid.size else 1.


def depth_scale(capture):
    """(scale passed to SurfaceDepth, unit). Without a reliable scale depth stays in map units."""
    if capture.meters_per_map_unit is None:
        return 1., "u"
    return capture.meters_per_map_unit, "m"


def depth_colors(capture, depth):
    """(RGB colour map, unit, colour range) for one rendered depth array."""
    unit = depth_scale(capture)[1]
    limit = DEPTH_RANGE_M if unit == "m" else map_unit_range(depth)
    return depth_to_color(depth, limit), unit, limit


@dataclass(frozen=True)
class GridCell:
    bounds: tuple  # x0, y0, x1, y1 in depth-image pixels (end exclusive)
    distance_m: float | None
    nearest_xy: tuple | None  # (x, y) depth-image pixel of the smallest depth


def nearest_in_grid(depth, rows=GRID_ROWS, cols=GRID_COLS):
    """Smallest known z-depth and its pixel in each cell of a rows x cols grid, row-major."""
    height, width = depth.shape
    ys = np.linspace(0, height, rows + 1).round().astype(int)
    xs = np.linspace(0, width, cols + 1).round().astype(int)
    cells = []
    for row in range(rows):
        for col in range(cols):
            x0, x1, y0, y1 = int(xs[col]), int(xs[col + 1]), int(ys[row]), int(ys[row + 1])
            block = depth[y0:y1, x0:x1]
            masked = np.where(np.isfinite(block) & (block > 0.), block, np.inf)
            if masked.size == 0 or not np.isfinite(masked.min()):
                cells.append(GridCell((x0, y0, x1, y1), None, None))
                continue
            dy, dx = np.unravel_index(np.argmin(masked), masked.shape)
            cells.append(GridCell((x0, y0, x1, y1), float(masked[dy, dx]), (x0 + int(dx), y0 + int(dy))))
    return tuple(cells)


@dataclass(frozen=True)
class DepthFrame:
    capture: DepthCapture
    image: np.ndarray  # RGB depth colour map
    cells: tuple
    unit: str = "m"  # "u" = map units, shown when no reliable metric scale exists
    range_max: float = DEPTH_RANGE_M


@dataclass(frozen=True)
class DepthViewState:
    frame: DepthFrame | None
    message: str | None


def _star(cx, cy, outer=11., inner=4.5):
    return tuple(
        (cx + (outer if i % 2 == 0 else inner) * math.sin(i * math.pi / 5),
         cy - (outer if i % 2 == 0 else inner) * math.cos(i * math.pi / 5))
        for i in range(10)
    )


def draw_depth_view(draw, state, scale, ox, oy, font, label_font, now_mono):
    """Paint the 3x3 grid, per-cell nearest distance and nearest-point stars on the video canvas."""
    frame = state.frame
    note = "深度估計｜靜態地圖 Z 深度，非即時測距或避障"
    colour = "#f0f3f5"
    if frame is not None:
        age = max(0., now_mono - frame.capture.stamp)
        note += f"｜配對影格 {frame.capture.name}｜擷取距今 {age:.1f} s"
        if frame.unit != "m":
            note += f"｜尺度未校正：單位為地圖單位 u，非公尺｜色階 0–{frame.range_max:.2f} u"
            colour = "#ffb020"
        if age > STALE_VIEW_S:
            colour = "#ffb020"
    if state.message:
        note += f"｜{state.message}"
        colour = "#ffb020"
    # Below the top buttons and the banner (<= 72 px); labels start below this note.
    draw.text((9, 77), note, fill="#000000", font=font)
    draw.text((8, 76), note, fill=colour, font=font)
    if frame is None:
        return
    nearest = min((c.distance_m for c in frame.cells if c.distance_m is not None), default=None)
    for cell in frame.cells:
        x0, y0, x1, y1 = (ox + cell.bounds[0] * scale, oy + cell.bounds[1] * scale,
                          ox + cell.bounds[2] * scale, oy + cell.bounds[3] * scale)
        draw.rectangle((x0, y0, x1, y1), outline="#ffffff", width=1)
        text = ("—" if cell.distance_m is None
                else f"{cell.distance_m:.1f} m" if frame.unit == "m" else f"{cell.distance_m:.2f} u")
        centre = (x0 + (x1 - x0) * .7, y0 + 104)  # right of the bottom-left HUD column
        draw.text((centre[0] + 1, centre[1] + 1), text, fill="#000000", font=label_font, anchor="mm")
        draw.text(centre, text, fill="#ffffff", font=label_font, anchor="mm")
    for cell in frame.cells:
        if cell.nearest_xy is None:
            continue
        px, py = ox + (cell.nearest_xy[0] + .5) * scale, oy + (cell.nearest_xy[1] + .5) * scale
        draw.polygon(_star(px, py), fill="#ff2a2a" if cell.distance_m == nearest else "#ffe14a",
                     outline="#000000", width=1)


class DepthInspector:
    """Bounded capture references and one lazily loaded surface scene per site."""

    # Live full-view state. Class defaults keep partially constructed inspectors valid.
    view_active = False
    view_error = None
    _view = None
    _view_future = None
    _view_submitted = None

    def __init__(self, app):
        self.app = app
        self.frames = OrderedDict()
        self.latest = None
        self.selected = None
        self.window = None
        self.surface = None
        self.depth = None
        self.future = None
        self.executor = None
        self.closed = False
        self.generation = 0
        self.bundle = None
        self.camera = None
        self.frame_id = None
        profile_path = getattr(app, "site_profile_path", None)
        if profile_path is not None:
            profile_path = Path(profile_path)
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            self.bundle = (profile_path.parent / profile["assets"]["localization_bundle"]).resolve()
            self.camera = profile["query_camera"]
            self.frame_id = profile["coordinate_frame"]["id"]
            self.bundle_digest = profile["asset_sha256"]["localization_bundle"]

    def remember(self, seq, name, stamp, frame):
        if self.bundle is None or self.closed:
            return
        # The stream/presenter contract owns immutable frames. Retain references,
        # never copy 720p RGB in the localization submit path.
        self.frames[seq] = (frame, name, float(stamp))
        self.frames.move_to_end(seq)
        while len(self.frames) > 12:
            self.frames.popitem(last=False)

    def receive(self, result):
        if self.closed:
            return
        if result.get("coordinate_frame_id") != self.frame_id:
            return
        scale = None
        estimator = getattr(self.app, "field_scale", None)
        if estimator is not None:
            with estimator.state.lock:
                diagnostics = estimator.state.scale_diagnostics(time.monotonic())
                if diagnostics.get("reliable"):
                    scale = diagnostics.get("scale_m_per_unit")
        capture = paired_capture(result, self.frames, scale)
        if capture is not None:
            self.latest = capture

    def reset(self):
        self.frames.clear()
        self.latest = self.selected = None
        self.depth = None
        self._view = self._view_future = self._view_submitted = None
        self.view_error = None
        self.generation += 1
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def toggle_view(self):
        """Switch the main video between stream and depth view; returns the new state."""
        self.view_active = not self.view_active
        self._view = self._view_future = self._view_submitted = None
        self.view_error = None
        return self.view_active

    def view_token(self):
        """Cheap change marker for the video dirty key; time-based only while active."""
        if not self.view_active:
            return None
        future = self._view_future
        return (id(self._view), id(self.latest), self.view_error,
                future is not None and future.done(), round(time.monotonic() * 10))

    def view_frame(self):
        """Tk-thread tick of the depth view; None when the stream is shown.

        Collects a finished render, then renders the newest paired capture. At most
        one render is in flight; newer captures replace older ones. Depth always
        belongs to the last strong paired localization, never the live frame.
        """
        if not self.view_active or self.closed:
            return None
        if self.bundle is None:
            return DepthViewState(None, "未選擇具有 direct bundle 的場域")
        future = self._view_future
        if future is not None and future.done():
            self._view_future = None
            try:
                self._view = future.result()
                self.view_error = None
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                self.view_error = f"深度無法取得：{exc}"
        latest = self.latest
        if self._view_future is None and latest is not None and latest is not self._view_submitted:
            self._view_submitted = latest
            if self._frame_matches_camera(latest.frame):
                self._view_future = self._worker().submit(self._render_view, latest)
            else:
                self.view_error = "影格與相機校正比例不符，不能猜測裁切後的內參。"
        message = self.view_error
        if message is None and self._view is None:
            message = "等待同影格、完整相機姿態的定位" if latest is None else "計算深度中"
        return DepthViewState(self._view, message)

    def _worker(self):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="surface-depth")
        return self.executor

    def _frame_matches_camera(self, source):
        width, height = (source.size if hasattr(source, "size") and not isinstance(source, np.ndarray)
                         else (source.shape[1], source.shape[0]))
        return width * self.camera["height"] == height * self.camera["width"]

    def _render_array(self, capture):
        # Preview at 640px; API/CLI can render full native resolution.
        width = min(VIEW_RENDER_WIDTH, self.camera["width"])
        height = max(1, round(width * self.camera["height"] / self.camera["width"]))
        return self._load_surface().render(
            self.camera, capture.center, capture.axes, width=width, height=height,
            meters_per_map_unit=depth_scale(capture)[0],
        )

    def _render_view(self, capture):
        depth = self._render_array(capture)
        image, unit, limit = depth_colors(capture, depth)
        return DepthFrame(capture, image, nearest_in_grid(depth), unit, limit)

    def close(self):
        self.closed = True
        self.reset()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def _load_surface(self):
        if self.surface is None:
            deploy = Path(__file__).resolve().parents[2] / "定位演算法/deploy_code/sfm_direct_deploy"
            if str(deploy) not in sys.path:
                sys.path.insert(0, str(deploy))
            from direct_paths import verify_file_sha256
            from surface_depth import SurfaceDepth

            verify_file_sha256(self.bundle, self.bundle_digest)
            self.surface = SurfaceDepth.load(
                self.bundle.parent / "dense_surface.json",
                coordinate_frame_id=self.frame_id, source_bundle_path=self.bundle,
            )
        return self.surface

    def show(self, _event=None):
        import tkinter as tk
        from tkinter import ttk
        from operator_rendering import VideoCanvasPresenter

        if self.window is not None:
            self.window.lift()
            return
        self.window = tk.Toplevel(self.app)
        self.window.title("MVS 絕對深度｜配對影格，非即時測距")
        self.window.geometry("960x640")
        self.window.protocol("WM_DELETE_WINDOW", self._hide)
        bar = ttk.Frame(self.window)
        bar.pack(fill="x")
        ttk.Button(bar, text="更新配對影格", command=self.refresh).pack(side="left")
        ttk.Button(bar, text="產生深度圖", command=self.render_depth).pack(side="left")
        ttk.Button(bar, text="原始影像", command=self.show_rgb).pack(side="left")
        self.status = tk.StringVar(value="")
        ttk.Label(self.window, textvariable=self.status, wraplength=930).pack(fill="x")
        self.canvas = tk.Canvas(self.window, bg="#08090b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.presenter = VideoCanvasPresenter(self.canvas)
        self.canvas.bind("<Configure>", lambda _event: self._paint())
        self.canvas.bind("<Button-1>", self.query)
        self.refresh()

    def _hide(self):
        self.generation += 1
        self.window.destroy()
        self.window = None

    def refresh(self):
        self.generation += 1
        self.selected = self.latest
        self.depth = None
        if self.bundle is None:
            self.status.set("未選擇具有 direct bundle 的場域。")
        elif self.selected is None:
            self.status.set("等待同影格、完整相機姿態的強定位。弱定位或保留舊姿態不提供深度。")
        else:
            self.status.set(
                f"配對影格：{self.selected.name}｜擷取距今 {max(0., time.monotonic()-self.selected.stamp):.1f} s"
                "｜點擊影像查公尺 Z 深度。這是靜態地圖估計，不是現場測距或避障。"
            )
        self._paint()

    def _paint(self):
        if self.window is None:
            return
        source = None if self.selected is None else self.selected.frame
        if self.depth is not None:
            source = self.depth_image
        self.presenter.present(source, 0. if self.selected is None else self.selected.stamp,
                               max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))

    def show_rgb(self):
        self.depth = None
        self._paint()

    def _run(self, fn, consume):
        if self.future is not None and not self.future.done():
            self.status.set("深度查詢仍在執行，請稍候。")
            return
        generation = self.generation
        self.status.set("讀取表面／查詢深度中，不阻塞主串流…")
        self.future = self._worker().submit(fn)
        future = self.future

        def poll():
            if self.closed or self.window is None or generation != self.generation:
                return
            if not future.done():
                self.window.after(30, poll)
                return
            try:
                consume(future.result())
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                self.status.set(f"深度無法取得：{exc}")
        self.window.after(30, poll)

    def query(self, event):
        capture = self.selected
        if capture is None or self.camera is None:
            return
        ox, oy = self.presenter._offset
        image_w, image_h = self.presenter._image_size
        x, y = event.x - ox, event.y - oy
        if not (0 <= x < image_w and 0 <= y < image_h):
            return
        if not self._frame_matches_camera(capture.frame):
            self.status.set("影格與相機校正比例不符，不能猜測裁切後的內參。")
            return
        pixel = ((x + .5) * self.camera["width"] / image_w,
                 (y + .5) * self.camera["height"] / image_h)

        scale, unit = depth_scale(capture)

        def calculate():
            return self._load_surface().sample(
                self.camera, capture.center, capture.axes, [pixel],
                meters_per_map_unit=scale,
            )[0]

        def consume(depth):
            text = f"Z 深度 {float(depth):.3f} {unit}" if np.isfinite(depth) else "深度未知（視線未命中已重建表面）"
            note = "" if unit == "m" else "｜尺度未校正，單位為地圖單位 u"
            self.status.set(f"配對影格 {capture.name}｜像素 ({pixel[0]:.1f}, {pixel[1]:.1f})｜{text}｜靜態地圖估計{note}")
        self._run(calculate, consume)

    def render_depth(self):
        capture = self.selected
        if capture is None or self.camera is None:
            return

        def calculate():
            return self._render_array(capture)

        def consume(depth):
            self.depth = depth
            self.depth_image, unit, limit = depth_colors(capture, depth)
            scale_note = ("公尺 Z 深度圖" if unit == "m"
                          else "尺度未校正，單位為地圖單位 u 的 Z 深度圖")
            self.status.set(f"配對影格 {capture.name}｜{scale_note}：紅近藍遠，色階 0–{limit:.4g} {unit}，黑色未知｜靜態地圖估計")
            self._paint()
        self._run(calculate, consume)
