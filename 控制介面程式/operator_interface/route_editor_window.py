"""Full-screen, self-contained point-cloud route editor built with Tk/Pillow."""

from __future__ import annotations

import queue
import tempfile
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from route_editor_controller import OrthoView, RouteEditorController, pointer_is_drag
from local_site_assets import discover_site_package
from route_editor_model import RouteDocument, glomap_to_editor
from x11_pinch_zoom import install_x11_pinch_zoom

_RED_INTRINSIC = "#dc2020"  # STATUS_RGB red_intrinsic (220,32,32)
_RED_INTRINSIC_RGB = (220, 32, 32)

_BG = (18, 21, 25)
_ARRIVE = "#3fbf7f"
_ARRIVE_FILL = (63, 191, 127, 52)
_SAFE_TUBE = (0, 229, 255, 128)
#: Matches ControlConfig.waypoint_arrive_radius; the exported route carries the
#: value the operator actually confirmed, so the two cannot drift apart.
#: Operator decision 2026-08-06: the arrival sphere is 0.005 to 0.05 map units.
#: 0.3 was tried first and was 5% of urai's camera trajectory -- far too coarse to
#: distinguish adjacent clicked points.
DEFAULT_ARRIVE_RADIUS_U = 0.02
ARRIVE_RADIUS_MIN_U = 0.005
ARRIVE_RADIUS_MAX_U = 0.05
DEFAULT_ROUTE_DEVIATION_U = 0.1
ROUTE_DEVIATION_MIN_U = 0.0
ROUTE_DEVIATION_MAX_U = 0.1
_ROUTE = "#ff3ea5"
_SELECTED = "#ffe169"
_AXIS = ("#ff6b5f", "#55d187", "#5aa7e8")


def _profile_route_deviation_cap(profile) -> float:
    controller = getattr(getattr(profile, "flight", None), "controller", None)
    raw = getattr(controller, "max_route_deviation_map_units", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return DEFAULT_ROUTE_DEVIATION_U
    value = float(raw)
    if not np.isfinite(value) or value < 0.0:
        return DEFAULT_ROUTE_DEVIATION_U
    return min(ROUTE_DEVIATION_MAX_U, max(ROUTE_DEVIATION_MIN_U, value))


def _resolve_map_frame(profile):
    """The site's MEASURED map basis, or the legacy guess when it has none.

    Returns (frame_or_None, align_source). frame None means the legacy [x, z, -y]
    swap, which is what every site without a T_align_gravity.json still gets; the
    exported route records which one was used so the flight loader cannot apply
    the wrong inverse.
    """
    align = getattr(profile, "map_align", None) if profile is not None else None
    if align is None:
        return None, "legacy"
    # A site that declares an alignment we cannot read must NOT quietly become a
    # legacy site: everything drawn and every waypoint marked would then be off by
    # the site's tilt while the export claimed "legacy". Refuse to open instead.
    try:
        from real_path_follow_controller import load_map_frame
    except ImportError as exc:
        raise ValueError(
            f"場域宣告了重力對齊 ({align})，但無法載入讀取它的模組：{exc}"
        ) from exc
    return load_map_frame(align), "measured"


class RouteEditorWindow(tk.Toplevel):
    """No backend reference and no flight-command capability by design."""

    def __init__(
        self,
        parent,
        *,
        profile,
        map_points: np.ndarray,
        map_source: Path | None,
        route_path: Path | None,
        import_route: Callable[[Path], object],
        discover_map_ply: Callable[[Path], list[Path]],
        load_map_ply: Callable[[Path], tuple[object | None, np.ndarray, str]],
        safety_check: Callable[[], tuple[bool, str]],
        map_loaded: Callable[[str], None] | None = None,
        on_close: Callable[[], None] | None = None,
        on_guard_failure: Callable[[str], None] | None = None,
        test_route: Callable[[Path], object] | None = None,
    ):
        bound = profile is not None and profile.coordinate_frame is not None
        if route_path is not None and not bound:
            raise ValueError("目前航線缺少場域座標綁定，不能安全編輯")
        # Resolved BEFORE the document is built: the loader needs the basis this
        # session displays in, so it can convert a route authored in the other one
        # instead of silently relabelling it.
        map_frame, align_source = _resolve_map_frame(profile)
        if route_path is None and bound:
            document = RouteDocument(
                site_id=profile.site_id,
                coordinate_frame_id=profile.coordinate_frame.id,
                align_source=align_source,
            )
        elif route_path is not None:
            document = RouteDocument.load(
                route_path,
                site_id=profile.site_id,
                coordinate_frame_id=profile.coordinate_frame.id,
                map_frame=map_frame,
                align_source=align_source,
            )
        else:
            document = RouteDocument(
                site_id="", coordinate_frame_id="", align_source=align_source)
        super().__init__(parent)
        self.profile = profile
        self.map_frame, self.align_source = map_frame, align_source
        # Reopen the route with the radius it was saved with, clamped to the
        # operator's range so the slider and the value can never disagree.
        saved_radius = document.arrive_radius_map_units
        self.arrive_radius_var = tk.DoubleVar(
            value=DEFAULT_ARRIVE_RADIUS_U if saved_radius is None
            else min(ARRIVE_RADIUS_MAX_U, max(ARRIVE_RADIUS_MIN_U, saved_radius)))
        self.route_deviation_max_u = _profile_route_deviation_cap(profile)
        saved_deviation = document.max_route_deviation_map_units
        route_deviation = (
            self.route_deviation_max_u
            if saved_deviation is None
            else min(
                self.route_deviation_max_u,
                max(ROUTE_DEVIATION_MIN_U, saved_deviation),
            )
        )
        self.route_deviation_var = tk.DoubleVar(value=route_deviation)
        self.discovery_var = tk.StringVar(value="尚未導入資料夾")
        self._discovered = None
        self.show_arrive_var = tk.BooleanVar(value=False)
        self.show_route_deviation_var = tk.BooleanVar(value=False)
        self.document = document
        self.import_route = import_route
        self.discover_map_ply = discover_map_ply
        self.load_map_ply = load_map_ply
        self.safety_check = safety_check
        self.map_loaded_callback = map_loaded
        self.on_close_callback = on_close
        self.on_guard_failure = on_guard_failure
        self.test_route_callback = test_route
        display_name = profile.display_name if bound else "未綁定 PLY"
        self.title(f"航線編輯器 — {display_name}")
        self.configure(bg="#121519")
        self.protocol("WM_DELETE_WINDOW", self.close_editor)
        self.controller = RouteEditorController(self.document.points)
        self.map_points = self._aligned_map_points(map_points)
        try:
            self.red_sphere_points = self._load_aligned_red_spheres(map_source)
        except Exception:
            self.red_sphere_points = np.empty((0, 6), dtype=np.float32)
        center, radius = self._bounds(self.map_points[:, :3])
        self.view = OrthoView(center=center, radius=radius, zoom=1.0)
        self.view.top()
        self._photo = None
        self._canvas_item = None
        self._redraw_pending = False
        self._interacting = False
        self._nav_button: int | None = None
        self._nav_last: tuple[int, int] | None = None
        self._mouse = (0, 0)
        self._move_mouse_start = (0, 0)
        self._left_press: tuple[int, int] | None = None
        self._left_last: tuple[int, int] | None = None
        self._left_dragging = False
        self._left_confirmed_move = False
        self._suppress_left_release = False
        self._single_click_after: str | None = None
        self._asset_loading = False
        self._scan_queue: queue.Queue[
            tuple[list[Path] | None, Path, Exception | None]
        ] = queue.Queue(maxsize=1)
        self._asset_load_queue: queue.Queue[
            tuple[
                object | None,
                np.ndarray | None,
                Path | None,
                str,
                Exception | None,
            ]
        ] = queue.Queue(maxsize=1)
        self._preview_only = not bound
        self._editing_current_route = route_path is not None
        self._map_source = profile.map_ply if bound else map_source
        self._saved_signature = self._signature()
        self._closed = False

        self.title_var = tk.StringVar(value=f"航線編輯器｜{display_name}")
        self.mode_var = tk.StringVar()
        self.selection_var = tk.StringVar()
        self.marking_mode_var = tk.BooleanVar(value=False)
        initial_status = (
            f"正在編輯 {route_path.name}；儲存後只會更新這條航線"
            if self._editing_current_route
            else "正在新增航線；儲存後會保留所有既有航線"
        )
        self.status_var = tk.StringVar(value=initial_status)
        self._build()
        self._bind_controls()
        self._refresh_labels()
        self.attributes("-fullscreen", True)
        self.lift()
        self.canvas.focus_set()
        self.after_idle(self.request_redraw)
        self.after(250, self._guard_tick)

    def _aligned_map_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("點雲資料必須是 Nx3 或 Nx6")
        xyz = glomap_to_editor(values[:, :3], self.map_frame)
        if values.shape[1] >= 6:
            rgb = np.clip(values[:, 3:6], 0, 255)
        else:
            rgb = np.full((len(values), 3), 190.0)
        return np.column_stack((xyz, rgb)).astype(np.float32)

    def _load_aligned_red_spheres(self, map_source: Path | None) -> np.ndarray:
        """Red intrinsic spheres for this site, aligned to the editor frame.

        Probes overlay/red_spheres_only.ply beside map_source, falling back to
        the workspace outputs demo. Empty when missing or unreadable so the
        editor continues to render the base cloud unchanged.
        """
        try:
            from map_point_io import load_red_sphere_points

            raw = load_red_sphere_points(map_source, 250000)
            if raw is None or len(raw) == 0:
                return np.empty((0, 6), dtype=np.float32)
            # Re-use the same glomap->editor transform as the base cloud.
            return self._aligned_map_points(raw)
        except Exception:
            return np.empty((0, 6), dtype=np.float32)

    @staticmethod
    def _bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
        if not len(points):
            return np.zeros(3, dtype=float), 1.0
        lo = np.percentile(points, 0.1, axis=0)
        hi = np.percentile(points, 99.9, axis=0)
        return (lo + hi) * 0.5, max(float(np.max(hi - lo) * 0.5), 1e-3)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure("RouteEditor.TFrame", background="#1a1e23")
        style.configure(
            "RouteEditor.TLabel", background="#1a1e23", foreground="#f2f5f7"
        )
        toolbar = ttk.Frame(self, style="RouteEditor.TFrame")
        toolbar.pack(fill="x")
        ttk.Label(
            toolbar,
            textvariable=self.title_var,
            style="RouteEditor.TLabel",
            font=("Sans", 12, "bold"),
        ).pack(side="left", padx=10, pady=8)
        self.ply_button = ttk.Button(
            toolbar, text="導入地圖資料夾", command=self.choose_map_folder
        )
        self.ply_button.pack(side="left", padx=(4, 2))
        ttk.Button(toolbar, text="俯視", command=self.set_top_view).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="正視", command=self.set_front_view).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="右視", command=self.set_right_view).pack(
            side="left", padx=2
        )
        self.finish_button = ttk.Button(
            toolbar, text="進入第二階段：逐點調整", command=self.finish_layout
        )
        self.finish_button.pack(side="left", padx=(12, 2))
        self.save_button = ttk.Button(
            toolbar,
            text="另存預覽航線" if self._preview_only else "儲存路線",
            command=self.save_route,
        )
        self.save_button.pack(
            side="right", padx=2
        )
        self.test_button = None
        if self.test_route_callback is not None:
            self.test_button = ttk.Button(
                toolbar,
                text="儲存並開始模擬航線",
                command=self.save_and_test_route,
            )
            self.test_button.pack(side="right", padx=2)
        ttk.Button(toolbar, text="離開編輯器", command=self.close_editor).pack(
            side="right", padx=(2, 10)
        )

        body = ttk.Frame(self, style="RouteEditor.TFrame")
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            body, bg="#121519", bd=0, highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        side = ttk.Frame(body, width=300, style="RouteEditor.TFrame")
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(
            side,
            textvariable=self.mode_var,
            style="RouteEditor.TLabel",
            font=("Sans", 11, "bold"),
            wraplength=275,
        ).pack(anchor="w", padx=12, pady=(14, 7))
        self.marking_mode_button = ttk.Checkbutton(
            side,
            text="標路徑點模式",
            variable=self.marking_mode_var,
            command=self.toggle_marking_mode,
        )
        self.marking_mode_button.pack(anchor="w", padx=12, pady=(0, 10))
        discovery = ttk.LabelFrame(side, text="資料夾探索結果")
        discovery.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Label(
            discovery, textvariable=self.discovery_var, style="RouteEditor.TLabel",
            justify="left", wraplength=260, font=("Sans", 9),
        ).pack(anchor="w", padx=6, pady=4)
        arrive = ttk.LabelFrame(side, text="到達半徑確認")
        arrive.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Checkbutton(
            arrive, text="顯示到達球體", variable=self.show_arrive_var,
            command=self.request_redraw,
        ).pack(anchor="w", padx=6, pady=(4, 0))
        self.arrive_label_var = tk.StringVar()
        ttk.Label(arrive, textvariable=self.arrive_label_var,
                  style="RouteEditor.TLabel").pack(anchor="w", padx=6)
        ttk.Scale(
            arrive, from_=ARRIVE_RADIUS_MIN_U, to=ARRIVE_RADIUS_MAX_U,
            variable=self.arrive_radius_var, command=self._on_arrive_radius,
        ).pack(fill="x", padx=6, pady=(0, 6))
        self._on_arrive_radius(None)
        safe_tube = ttk.LabelFrame(side, text="航線安全管確認")
        safe_tube.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Checkbutton(
            safe_tube,
            text="顯示航線安全管",
            variable=self.show_route_deviation_var,
            command=self.request_redraw,
        ).pack(anchor="w", padx=6, pady=(4, 0))
        self.route_deviation_label_var = tk.StringVar()
        ttk.Label(
            safe_tube,
            textvariable=self.route_deviation_label_var,
            style="RouteEditor.TLabel",
            wraplength=260,
        ).pack(anchor="w", padx=6)
        self.route_deviation_scale = ttk.Scale(
            safe_tube,
            from_=ROUTE_DEVIATION_MIN_U,
            to=self.route_deviation_max_u,
            variable=self.route_deviation_var,
            command=self._on_route_deviation,
        )
        self.route_deviation_scale.pack(fill="x", padx=6, pady=(0, 6))
        self._on_route_deviation(None)
        ttk.Label(
            side,
            textvariable=self.selection_var,
            style="RouteEditor.TLabel",
            wraplength=275,
        ).pack(anchor="w", padx=12, pady=(0, 14))
        instructions = (
            "地圖\n"
            "  導入資料夾：自動偵測 PLY\n"
            "  單一檔自動載入，多檔時選擇\n"
            "  SHA-256 匹配場域後才可正式匯入\n\n"
            "第一階段：標路徑點\n"
            "  先開啟「標路徑點模式」\n"
            "  左鍵單擊：新增路徑點\n"
            "  Enter：進入第二階段\n\n"
            "第二階段：逐點調整位置\n"
            "  左鍵：選取航點\n"
            "  G：開始平移\n"
            "  G 後 X / Y / Z：鎖定軸向\n"
            "  左鍵／Enter：確認，Esc：取消\n"
            "  Delete：刪除航點\n"
            "  Ctrl+Z：復原\n"
            "  Ctrl+Shift+Z：重做\n\n"
            "視角\n"
            "  左鍵拖曳：旋轉\n"
            "  左鍵雙擊：切換地圖中心\n"
            "  右鍵拖曳：平移\n"
            "  觸控板捏合／滾輪：縮放\n\n"
            "編輯器座標：Z 永遠向上。\n"
            "新增模式會建立另一條路線；編輯模式只更新所選路線。\n"
            "「儲存路線」會同步 SHA，並將儲存結果設為目前 AUTO 航線。"
        )
        ttk.Label(
            side,
            text=instructions,
            style="RouteEditor.TLabel",
            justify="left",
            wraplength=275,
        ).pack(anchor="w", padx=12)

        status = tk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            bg="#242a31",
            fg="#f2f5f7",
            padx=10,
            pady=6,
        )
        status.pack(fill="x")

    def _bind_controls(self) -> None:
        self.canvas.bind("<Configure>", lambda _event: self.request_redraw())
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Double-Button-1>", self.on_left_double)
        self.canvas.bind("<ButtonPress-3>", self.on_navigation_press)
        self.canvas.bind("<B3-Motion>", self.on_navigation_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_navigation_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
        self.bind("<KeyPress>", self.on_key)
        self._pinch_zoom = install_x11_pinch_zoom(
            self.canvas,
            lambda: self.view.zoom,
            self._set_zoom,
        )

    def _signature(self) -> tuple:
        points = tuple(
            tuple(float(value) for value in point) for point in self.controller.points
        )
        return (
            points,
            float(self.arrive_radius_var.get()),
            float(self.route_deviation_var.get()),
        )

    def _refresh_labels(self) -> None:
        phase = (
            "第一階段：標路徑點"
            if self.controller.phase == "layout"
            else "第二階段：逐點調整位置"
        )
        source = (
            "PLY 單檔預覽（禁止正式匯入）"
            if self._preview_only
            else self.profile.display_name
        )
        self.mode_var.set(
            f"{phase}\n路徑點數：{len(self.controller.points)}\n地圖：{source}"
        )
        selected = self.controller.selected
        if selected is None or not (0 <= selected < len(self.controller.points)):
            self.selection_var.set("選取：無")
        else:
            x, y, z = self.controller.points[selected]
            self.selection_var.set(
                f"選取：航點 {selected + 1}\nX {x:.3f}  Y {y:.3f}  高度 Z {z:.3f}"
            )
        self.finish_button.configure(
            state="normal" if self.controller.phase == "layout" else "disabled"
        )
        self.marking_mode_var.set(self.controller.placement_enabled)
        self.marking_mode_button.configure(
            state="normal" if self.controller.phase == "layout" else "disabled"
        )
        self.save_button.configure(
            state="disabled" if self._asset_loading else "normal"
        )
        if self.test_button is not None:
            self.test_button.configure(
                state=(
                    "disabled"
                    if self._asset_loading or self._preview_only
                    else "normal"
                )
            )
        asset_state = "disabled" if self._asset_loading else "normal"
        self.ply_button.configure(state=asset_state)

    def request_redraw(self) -> None:
        if self._redraw_pending or self._closed:
            return
        self._redraw_pending = True
        self.after_idle(self._redraw)

    def _redraw(self) -> None:
        self._redraw_pending = False
        if self._closed:
            return
        width = max(320, self.canvas.winfo_width())
        height = max(240, self.canvas.winfo_height())
        arr = np.empty((height, width, 3), dtype=np.uint8)
        arr[:] = _BG
        limit = 60000 if self._interacting else 220000
        step = max(1, len(self.map_points) // max(1, limit))
        points = self.map_points[::step]
        if len(points):
            sx, sy, depth = self.view.project(points[:, :3], width, height)
            ix = sx.astype(int)
            iy = sy.astype(int)
            inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            order = np.argsort(depth[inside])
            arr[iy[inside][order], ix[inside][order]] = np.clip(
                points[:, 3:6][inside][order], 0, 255
            ).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        # Red intrinsic spheres — always visible, slightly larger than base points
        try:
            red = getattr(self, "red_sphere_points", None)
            if red is not None and len(red) > 0:
                # Keep all sphere vertices (few k) regardless of interacting limit
                sx_r, sy_r, depth_r = self.view.project(red[:, :3], width, height)
                ix_r = sx_r.astype(int)
                iy_r = sy_r.astype(int)
                inside_r = (ix_r >= 0) & (ix_r < width) & (iy_r >= 0) & (iy_r < height)
                if np.any(inside_r):
                    order_r = np.argsort(depth_r[inside_r])
                    xs = sx_r[inside_r][order_r]
                    ys = sy_r[inside_r][order_r]
                    rgb_r = red[:, 3:6][inside_r][order_r] if red.shape[1] >= 6 else np.full((len(xs), 3), _RED_INTRINSIC_RGB)
                    for x, y, col in zip(xs.tolist(), ys.tolist(), np.clip(rgb_r, 0, 255).astype(np.uint8).tolist()):
                        fill = f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"
                        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=fill, outline="#ffffff", width=1)
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=fill, outline=fill)
        except Exception:
            pass
        self._draw_axes(draw, width, height)
        self._draw_route(draw, width, height)
        photo = ImageTk.PhotoImage(image)
        self._photo = photo
        if self._canvas_item is None:
            self._canvas_item = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self._canvas_item, image=photo)
    def _draw_axes(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        origin = self.view.center
        size = self.view.radius * 0.16
        points = np.vstack((origin, origin + np.eye(3) * size))
        sx, sy, _ = self.view.project(points, width, height)
        for axis, label in enumerate(("X", "Y", "Z")):
            draw.line(
                (sx[0], sy[0], sx[axis + 1], sy[axis + 1]), fill=_AXIS[axis], width=3
            )
            draw.text((sx[axis + 1] + 4, sy[axis + 1] + 3), label, fill=_AXIS[axis])

    def _draw_route(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        if not self.controller.points:
            return
        sx, sy, _ = self.view.project(self.controller.points, width, height)
        projected = list(zip(sx.tolist(), sy.tolist()))
        if len(projected) > 1:
            if self.show_route_deviation_var.get():
                tube_pixels = float(self.route_deviation_var.get()) * self.view.scale(
                    width, height
                )
                if tube_pixels >= 1.0:
                    tube_width = max(2, int(round(tube_pixels * 2.0)))
                    draw.line(
                        projected,
                        fill=_SAFE_TUBE,
                        width=tube_width,
                        joint="curve",
                    )
                    for x, y in projected:
                        draw.ellipse(
                            (
                                x - tube_pixels,
                                y - tube_pixels,
                                x + tube_pixels,
                                y + tube_pixels,
                            ),
                            fill=_SAFE_TUBE,
                        )
            draw.line(projected, fill=_ROUTE, width=3, joint="curve")
        if self.show_arrive_var.get():
            # The arrival sphere in MAP units, drawn at the view's own scale so
            # what the operator confirms here is what the controller will use.
            width_px, height_px = width, height
            pixels = float(self.arrive_radius_var.get()) * self.view.scale(
                width_px, height_px
            )
            if pixels >= 1.0:
                for x, y in projected:
                    draw.ellipse(
                        (x - pixels, y - pixels, x + pixels, y + pixels),
                        fill=_ARRIVE_FILL,
                        outline=_ARRIVE,
                        width=2,
                    )
        for index, (x, y) in enumerate(projected):
            selected = index == self.controller.selected
            radius = 8 if selected else 6
            fill = _SELECTED if selected else _ROUTE
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=fill,
                outline="white",
            )
            draw.text((x + radius + 3, y - radius - 2), str(index + 1), fill="white")

    def _canvas_size(self) -> tuple[int, int]:
        return max(320, self.canvas.winfo_width()), max(240, self.canvas.winfo_height())

    def _nearest_cloud(self, x: float, y: float, threshold: float = 20.0):
        if not len(self.map_points):
            return None
        width, height = self._canvas_size()
        sx, sy, depth = self.view.project(self.map_points[:, :3], width, height)
        distance = (sx - x) ** 2 + (sy - y) ** 2
        candidates = np.flatnonzero(distance <= threshold * threshold)
        if not len(candidates):
            return None
        index = int(candidates[np.argmax(depth[candidates])])
        return self.map_points[index, :3].astype(float)

    def _nearest_waypoint(self, x: float, y: float, threshold: float = 28.0):
        if not self.controller.points:
            return None
        width, height = self._canvas_size()
        sx, sy, _ = self.view.project(self.controller.points, width, height)
        distance = (sx - x) ** 2 + (sy - y) ** 2
        index = int(np.argmin(distance))
        return index if float(distance[index]) <= threshold * threshold else None

    def on_left_press(self, event) -> str:
        self.canvas.focus_set()
        self._mouse = (int(event.x), int(event.y))
        if self.controller.moving:
            self.controller.confirm_move()
            self.status_var.set("已確認航點移動；按「儲存路線」即可更新正式航線")
            self._left_confirmed_move = True
            self._refresh_labels()
            self.request_redraw()
            return "break"
        self._left_press = self._mouse
        self._left_last = self._mouse
        self._left_dragging = False
        self._left_confirmed_move = False
        return "break"

    def on_left_drag(self, event) -> str:
        if (
            self.controller.moving
            or self._left_press is None
            or self._left_last is None
        ):
            return "break"
        x, y = int(event.x), int(event.y)
        if not self._left_dragging and not pointer_is_drag(self._left_press, (x, y)):
            return "break"
        self._left_dragging = True
        self._interacting = True
        dx, dy = x - self._left_last[0], y - self._left_last[1]
        self._left_last = (x, y)
        self.view.yaw -= dx * 0.008
        self.view.pitch = (self.view.pitch + dy * 0.008) % (2.0 * np.pi)
        self.request_redraw()
        return "break"

    def on_left_release(self, event) -> str:
        if self._left_confirmed_move:
            self._left_confirmed_move = False
            return "break"
        if self._suppress_left_release:
            self._suppress_left_release = False
            return "break"
        dragged = self._left_dragging
        self._left_press = None
        self._left_last = None
        self._left_dragging = False
        self._interacting = False
        if dragged:
            self.status_var.set("已旋轉地圖；左鍵單擊仍可新增或選取航點")
            self.request_redraw()
            return "break"
        x, y = int(event.x), int(event.y)
        self._cancel_pending_single_click()
        self._single_click_after = self.after(
            280, lambda: self._handle_single_left(x, y)
        )
        return "break"

    def on_left_double(self, event) -> str:
        self._cancel_pending_single_click()
        self._suppress_left_release = True
        self._left_press = None
        self._left_last = None
        self._left_dragging = False
        point = self._nearest_cloud(event.x, event.y, threshold=35.0)
        if point is None:
            self.status_var.set("雙擊位置附近沒有點雲，地圖中心未變更")
            return "break"
        self.view.center = np.asarray(point, dtype=float)
        self.view.pan_x = 0.0
        self.view.pan_y = 0.0
        self.status_var.set(
            f"地圖中心已切換至 X={point[0]:.2f}, Y={point[1]:.2f}, Z={point[2]:.2f}"
        )
        self.request_redraw()
        return "break"

    def _cancel_pending_single_click(self) -> None:
        if self._single_click_after is None:
            return
        try:
            self.after_cancel(self._single_click_after)
        except tk.TclError:
            pass
        self._single_click_after = None

    def _handle_single_left(self, x: int, y: int) -> None:
        self._single_click_after = None
        if self._closed:
            return
        if self.controller.phase == "layout":
            if not self.controller.placement_enabled:
                self.status_var.set(
                    "第一階段：請先開啟右側「標路徑點模式」再用左鍵單擊"
                )
                return
            point = self._nearest_cloud(x, y)
            if point is None:
                self.status_var.set("游標附近沒有點雲，未新增航點；請放大或重新點選")
                return
            self.controller.add(point)
            self.status_var.set(
                f"已新增路徑點 {len(self.controller.points)}；可繼續單擊標點"
            )
        else:
            selected = self._nearest_waypoint(x, y)
            self.controller.selected = selected
            self.status_var.set(
                "已選取航點，按 G 移動；G 後按 Z 可只調整高度"
                if selected is not None
                else "未選到航點"
            )
        self._refresh_labels()
        self.request_redraw()

    def on_motion(self, event) -> None:
        self._mouse = (int(event.x), int(event.y))
        if not self.controller.moving:
            return
        self._update_move_preview()

    def toggle_marking_mode(self) -> None:
        active = self.controller.set_placement_enabled(self.marking_mode_var.get())
        self.marking_mode_var.set(active)
        self.status_var.set(
            "標路徑點模式已開啟：左鍵單擊點雲可連續新增路徑點"
            if active
            else "標路徑點模式已關閉：左鍵單擊不會新增路徑點"
        )
        self._refresh_labels()
        self.canvas.focus_set()

    def _update_move_preview(self) -> None:
        width, height = self._canvas_size()
        dx = self._mouse[0] - self._move_mouse_start[0]
        dy = self._mouse[1] - self._move_mouse_start[1]
        delta = self.view.screen_delta_to_world(
            dx, dy, width, height, self.controller.move_axis
        )
        self.controller.preview_move(delta)
        self._refresh_labels()
        self.request_redraw()

    def on_navigation_press(self, event) -> str:
        self.canvas.focus_set()
        if self.controller.moving:
            self.status_var.set("請先用左鍵／Enter 確認移動，或用 Esc 取消")
            return "break"
        self._nav_button = int(event.num)
        self._nav_last = (int(event.x), int(event.y))
        self._interacting = True
        return "break"

    def on_navigation_drag(self, event) -> str:
        if self._nav_last is None:
            return "break"
        x, y = int(event.x), int(event.y)
        dx, dy = x - self._nav_last[0], y - self._nav_last[1]
        self._nav_last = (x, y)
        self.view.pan_x += dx
        self.view.pan_y += dy
        self.request_redraw()
        return "break"

    def on_navigation_release(self, _event) -> str:
        self._nav_button = None
        self._nav_last = None
        self._interacting = False
        self.request_redraw()
        return "break"

    def on_wheel(self, event) -> str:
        factor = (
            1.12
            if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
            else 1.0 / 1.12
        )
        self._set_zoom(self.view.zoom * factor)
        return "break"

    def _set_zoom(self, zoom: float) -> None:
        self.view.zoom = float(np.clip(zoom, 0.08, 120.0))
        self.request_redraw()

    def _handle_undo_redo(self, *, shift: bool) -> None:
        changed = self.controller.redo() if shift else self.controller.undo()
        if changed:
            self.status_var.set("已重做" if shift else "已復原")
            self._refresh_labels()
            self.request_redraw()

    def _handle_begin_move(self) -> None:
        if self.controller.begin_move():
            self._move_mouse_start = self._mouse
            self.status_var.set(
                "平移中：按 X / Y / Z 鎖定軸；左鍵或 Enter 確認，Esc 取消"
            )

    def _handle_move_axis(self, key: str) -> None:
        axis = {"x": 0, "y": 1, "z": 2}[key]
        self.controller.constrain(axis)
        self._update_move_preview()
        self.status_var.set(f"平移中：已鎖定 {key.upper()} 軸")

    def _handle_enter(self) -> None:
        if self.controller.moving:
            self.controller.confirm_move()
            self.status_var.set("已確認航點移動")
        elif self.controller.phase == "layout":
            self.finish_layout()
        self._refresh_labels()
        self.request_redraw()

    def _handle_escape(self) -> None:
        if self.controller.cancel_move():
            self.status_var.set("已取消這次移動；請使用右上角按鈕離開編輯器")
            self._refresh_labels()
            self.request_redraw()
        else:
            self.status_var.set("Esc 只取消移動；請使用右上角「離開編輯器」")

    def _handle_delete(self) -> None:
        if self.controller.delete_selected():
            self.status_var.set("已刪除選取航點")
            self._refresh_labels()
            self.request_redraw()

    def on_key(self, event) -> str | None:
        key = str(event.keysym).lower()
        ctrl = bool(int(event.state) & 0x0004)
        shift = bool(int(event.state) & 0x0001)
        if ctrl and key == "z":
            self._handle_undo_redo(shift=shift)
            return "break"
        if (
            key == "g"
            and self.controller.phase == "height"
            and not self.controller.moving
        ):
            self._handle_begin_move()
            return "break"
        if key in {"x", "y", "z"} and self.controller.moving:
            self._handle_move_axis(key)
            return "break"
        if key in {"return", "kp_enter"}:
            self._handle_enter()
            return "break"
        if key == "escape":
            self._handle_escape()
            return "break"
        if key == "delete" and self.controller.phase == "height":
            self._handle_delete()
            return "break"
        return None

    def _on_arrive_radius(self, _value) -> None:
        if _value is not None:
            self.show_arrive_var.set(True)
        radius = float(self.arrive_radius_var.get())
        shortest = self.controller.shortest_leg()
        note = ""
        if self.controller.spheres_overlap(radius):
            # Overlapping spheres mean the drone retires waypoints without ever
            # translating between them: it settles, advances, settles, advances.
            note = f"　⚠ 半徑 >= 最短段 {shortest:.3f}u 的一半，相鄰球體會重疊"
        self.arrive_label_var.set(f"半徑 {radius:.3f} 地圖單位{note}")
        if hasattr(self, "route_deviation_label_var"):
            self._set_route_deviation_label()
        if self.show_arrive_var.get():
            self.request_redraw()

    def _set_route_deviation_label(self) -> None:
        radius = float(self.route_deviation_var.get())
        note = ""
        if radius < float(self.arrive_radius_var.get()):
            note = "；⚠ 小於到達半徑"
        self.route_deviation_label_var.set(
            f"半徑 {radius:.3f} 地圖單位{note}\n超出後立即結束 AUTO 並原地降落"
        )

    def _on_route_deviation(self, _value) -> None:
        if _value is not None:
            self.show_route_deviation_var.set(True)
        self._set_route_deviation_label()
        if self.show_route_deviation_var.get():
            self.request_redraw()

    def finish_layout(self) -> None:
        try:
            self.controller.finish_layout()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        # Stage two is where the arrival spheres matter, so show them by default:
        # the operator confirms the radius against the real geometry before export.
        self.show_arrive_var.set(True)
        self.show_route_deviation_var.set(True)
        self._on_arrive_radius(None)
        self._on_route_deviation(None)
        self.status_var.set(
            "第二階段：綠圈為到達球體、藍綠管為安全範圍，確認兩個半徑後再匯出；"
            "選取路徑點後按 G，可用 X／Y／Z 調整位置"
        )
        self._refresh_labels()
        self.request_redraw()

    def set_top_view(self) -> None:
        self.view.top()
        self.status_var.set("俯視：X/Y 為水平面，Z 為高度")
        self.request_redraw()

    def set_front_view(self) -> None:
        self.view.front()
        self.status_var.set("正視：適合逐點調整 Z 高度")
        self.request_redraw()

    def set_right_view(self) -> None:
        self.view.right()
        self.status_var.set("右視：適合逐點調整 Z 高度")
        self.request_redraw()

    def _confirm_map_replace(self) -> bool:
        if self._signature() == self._saved_signature:
            return True
        return messagebox.askyesno(
            "取代目前地圖",
            "目前航線有尚未儲存的修改。確定放棄修改並載入另一張地圖？",
            parent=self,
        )

    def choose_map_folder(self) -> None:
        if self._asset_loading or not self._confirm_map_replace():
            return
        initial_dir = (
            self._map_source.parent if self._map_source is not None else Path.cwd()
        )
        selected = filedialog.askdirectory(
            parent=self,
            title="選擇包含 PLY 點雲的資料夾",
            initialdir=str(initial_dir),
        )
        if selected:
            self._start_folder_scan(Path(selected).resolve())

    def _start_folder_scan(self, folder: Path) -> None:
        self._asset_loading = True
        self.status_var.set(f"正在偵測資料夾內的 PLY：{folder}…")
        self._refresh_labels()

        def worker() -> None:
            try:
                files = self.discover_map_ply(folder)
                # Best-effort: a discovery failure must not stop the map from
                # loading, it only costs the operator the asset summary.
                try:
                    package = discover_site_package(folder)
                except Exception as discovery_exc:
                    package = None
                    print(f"[route-editor] asset discovery failed: {discovery_exc!r}",
                          flush=True)
                self._scan_queue.put((files, folder, None, package))
            except Exception as exc:
                self._scan_queue.put((None, folder, exc, None))

        threading.Thread(
            target=worker,
            name="route-editor-ply-scan",
            daemon=True,
        ).start()
        self.after(50, self._poll_folder_scan)

    def _poll_folder_scan(self) -> None:
        if self._closed:
            return
        try:
            files, folder, error, package = self._scan_queue.get_nowait()
        except queue.Empty:
            self.after(50, self._poll_folder_scan)
            return
        self._apply_discovery(package)
        if error is not None:
            self._asset_loading = False
            self.status_var.set(f"PLY 偵測失敗：{error}")
            self._refresh_labels()
            return
        assert files is not None
        if not files:
            self._asset_loading = False
            self.status_var.set(f"資料夾內找不到 PLY：{folder}")
            self._refresh_labels()
            return
        if len(files) == 1:
            self.status_var.set(f"偵測到 1 個 PLY，自動載入：{files[0].name}")
            self._start_map_load(files[0])
            return
        selected = self._select_detected_ply(folder, files)
        if selected is None:
            self._asset_loading = False
            self.status_var.set(f"偵測到 {len(files)} 個 PLY；使用者取消選擇")
            self._refresh_labels()
            return
        self._start_map_load(selected)

    def _apply_discovery(self, package) -> None:
        """Report what the folder contains, and say so when a route is missing."""
        self._discovered = package
        if package is None:
            self.discovery_var.set("資產探索失敗；仍可手動選擇 PLY")
            return
        lines = list(package.summary_lines())
        if not package.has_route:
            # The whole reason this editor is open. Say it plainly rather than
            # leaving the operator to notice the missing tick.
            lines.append("→ 這個場域還沒有航線：請按「標路徑點模式」開始畫")
        self.discovery_var.set("\n".join(lines))

    def _select_detected_ply(self, folder: Path, files: list[Path]) -> Path | None:
        dialog = tk.Toplevel(self)
        dialog.title("選擇要導入的 PLY")
        dialog.transient(self)
        dialog.configure(bg="#1a1e23")
        dialog.geometry("760x430")
        ttk.Label(
            dialog,
            text=f"在 {folder.name} 偵測到 {len(files)} 個 PLY，請選擇地圖：",
            wraplength=720,
        ).pack(anchor="w", padx=12, pady=(12, 6))
        listbox = tk.Listbox(
            dialog,
            bg="#121519",
            fg="#f2f5f7",
            selectbackground="#315d86",
            activestyle="none",
        )
        listbox.pack(fill="both", expand=True, padx=12, pady=6)
        for path in files:
            relative = path.relative_to(folder).as_posix()
            size_mib = path.stat().st_size / (1024.0 * 1024.0)
            listbox.insert("end", f"{relative}    ({size_mib:.1f} MiB)")
        listbox.selection_set(0)
        result: list[Path | None] = [None]

        def choose(_event=None) -> str:
            selected = listbox.curselection()
            if selected:
                result[0] = files[int(selected[0])]
                dialog.destroy()
            return "break"

        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=12, pady=(4, 12))
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="導入選取 PLY", command=choose).pack(
            side="right", padx=(0, 8)
        )
        listbox.bind("<Double-Button-1>", choose)
        dialog.bind("<Return>", choose)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        listbox.focus_set()
        self.wait_window(dialog)
        return result[0]

    def _start_map_load(self, source: Path) -> None:
        self._asset_loading = True
        self.status_var.set(f"正在載入 PLY 並比對已匯入場域：{source.name}…")
        self._refresh_labels()

        def worker() -> None:
            try:
                profile, points, message = self.load_map_ply(source)
                self._asset_load_queue.put((profile, points, source, message, None))
            except Exception as exc:
                self._asset_load_queue.put((None, None, source, "", exc))

        threading.Thread(
            target=worker,
            name="route-editor-ply-load",
            daemon=True,
        ).start()
        self.after(50, self._poll_map_load)

    def _poll_map_load(self) -> None:
        if self._closed:
            return
        try:
            profile, points, source, message, error = (
                self._asset_load_queue.get_nowait()
            )
        except queue.Empty:
            self.after(50, self._poll_map_load)
            return
        self._asset_loading = False
        if error is not None:
            self.status_var.set(f"地圖載入失敗：{error}")
            self._refresh_labels()
            return
        assert points is not None and source is not None
        # The swapped-in PLY may belong to a site with a DIFFERENT basis from the
        # one this window opened with. Re-resolve before anything is drawn or
        # marked, or the new cloud is displayed through the old site's frame and
        # every waypoint put on it is off by the difference between them.
        # _resolve_map_frame refuses to guess when a declared alignment cannot be
        # read; report that and keep the previous map rather than dying silently
        # inside the Tk callback with the toolbar stuck on "loading".
        try:
            self.map_frame, self.align_source = _resolve_map_frame(profile)
        except Exception as exc:
            self.status_var.set(f"地圖載入失敗（重力對齊無法讀取）：{exc}")
            self._refresh_labels()
            return
        if profile is not None:
            assert profile.coordinate_frame is not None
            self.profile = profile
            self.document = RouteDocument(
                site_id=profile.site_id,
                coordinate_frame_id=profile.coordinate_frame.id,
                align_source=self.align_source,
            )
            self._preview_only = False
            display_name = profile.display_name
        else:
            self.profile = None
            self.document = RouteDocument(
                site_id="", coordinate_frame_id="", align_source=self.align_source)
            self._preview_only = True
            display_name = "未綁定 PLY"
        self.route_deviation_max_u = _profile_route_deviation_cap(profile)
        self.route_deviation_scale.configure(to=self.route_deviation_max_u)
        self.route_deviation_var.set(
            min(float(self.route_deviation_var.get()), self.route_deviation_max_u)
        )
        self._on_route_deviation(None)
        self._map_source = source
        self.title(f"航線編輯器 — {display_name}")
        self.title_var.set(f"航線編輯器｜{display_name}")
        if self.map_loaded_callback is not None:
            self.map_loaded_callback(message)
        self.controller = RouteEditorController()
        self.map_points = self._aligned_map_points(points)
        try:
            self.red_sphere_points = self._load_aligned_red_spheres(source)
        except Exception:
            self.red_sphere_points = np.empty((0, 6), dtype=np.float32)
        center, radius = self._bounds(self.map_points[:, :3])
        self.view = OrthoView(center=center, radius=radius, zoom=1.0)
        self.view.top()
        self._editing_current_route = False
        self.save_button.configure(
            text="另存預覽航線" if self._preview_only else "儲存路線"
        )
        self._saved_signature = self._signature()
        self.status_var.set(message)
        self._refresh_labels()
        self.request_redraw()

    def _choose_preview_path(self) -> Path | None:
        preview_dir = (
            self._map_source.parent if self._map_source is not None else Path.cwd()
        )
        preview_dir.mkdir(parents=True, exist_ok=True)
        initial = f"route_preview_{time.strftime('%Y%m%d_%H%M%S')}.json"
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="另存預覽航線",
            initialdir=str(preview_dir),
            initialfile=initial,
            defaultextension=".json",
            filetypes=(("航線 JSON", "*.json"),),
        )
        return Path(selected).resolve() if selected else None

    def _update_document(self) -> None:
        if self.controller.moving:
            self.controller.confirm_move()
        self.document.points = self.controller.points
        self.document.arrive_radius_map_units = float(self.arrive_radius_var.get())
        self.document.max_route_deviation_map_units = float(
            self.route_deviation_var.get()
        )

    def save_route(self) -> Path | None:
        self._update_document()
        if self._preview_only:
            path = self._choose_preview_path()
            if path is None:
                return None
            try:
                saved = self.document.save(path, preview_only=True)
            except Exception as exc:
                self.status_var.set(f"預覽航線儲存失敗：{exc}")
                return None
            self._saved_signature = self._signature()
            self.status_var.set(
                f"預覽航線已儲存：{saved}；未綁定場域，不能供 AUTO 使用"
            )
            return saved

        try:
            with tempfile.TemporaryDirectory(prefix="sfm-route-") as temporary:
                staged = self.document.save(Path(temporary) / "flight_route.json")
                result = self.import_route(staged)
        except Exception as exc:
            self.status_var.set(f"航線儲存失敗：{exc}")
            return None
        saved = Path(getattr(result, "asset_path", None) or self.profile.route_json)
        self.document.source_path = saved
        self._saved_signature = self._signature()
        message = str(getattr(result, "message", "航線已匯入"))
        self.status_var.set(f"{message}；SHA 已同步，但不會立即起飛")
        self.close_editor(force=True)
        return saved

    def save_and_test_route(self) -> Path | None:
        """Persist a bound route, then ask the simulation-only host to run it."""
        saved = self.save_route()
        if saved is not None and self.test_route_callback is not None:
            self.test_route_callback(saved)
        return saved

    def close_editor(self, *, force: bool = False) -> None:
        if self._closed:
            return
        if not force and self._signature() != self._saved_signature:
            leave = messagebox.askyesno(
                "尚未儲存",
                "航線有尚未儲存的修改，確定離開？",
                parent=self,
            )
            if not leave:
                return
        self._closed = True
        try:
            self.attributes("-fullscreen", False)
        except tk.TclError:
            pass
        self.destroy()
        if self.on_close_callback is not None:
            self.on_close_callback()

    def _guard_tick(self) -> None:
        if self._closed:
            return
        allowed, reason = self.safety_check()
        if not allowed:
            self.close_editor(force=True)
            if self.on_guard_failure is not None:
                self.on_guard_failure(reason)
            return
        self.after(250, self._guard_tick)
