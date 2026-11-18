"""Self-contained, resizable point-cloud route editor built with Tk/Pillow."""

from __future__ import annotations

import json
import math
import queue
import tempfile
import threading
import time
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from route_editor_controller import OrthoView, RouteEditorController, pointer_is_drag
from route_editor_objects import ObjectBoxController, PointerScale, box_corners
from route_editor_wires import WireController, sample_wire, wire_color
from local_site_assets import discover_site_package
from operator_autonomy_status import waypoint_action_text
from operator_ui_style import (
    BORDER, BORDER_STRONG, CARD, DISABLED, FIELD, FONT_BOLD, FONT_SMALL, FONT_TITLE, MUTED,
    PANEL,
    SELECT_BG, SELECT_FG, SURFACE, TEXT, TEXT_BOX, VIEWPORT_EDITOR_BG,
    apply_operator_styles,
)
from route_editor_model import (
    MAX_ACTIONS_PER_WAYPOINT,
    MapDistanceScale,
    RouteDocument,
    glomap_to_editor,
    load_map_distance_scale,
    parse_waypoint_action,
)
from route_editor_rendering import PointCloudRenderWorker, cloud_raster_key, nearest_point_index
from x11_pinch_zoom import install_x11_pinch_zoom


_ARRIVE = "#3fbf7f"
_ARRIVE_FILL = (63, 191, 127, 52)
#: Matches ControlConfig.waypoint_arrive_radius; the exported route carries the
#: value the operator actually confirmed, so the two cannot drift apart.
#: Operator decision 2026-08-06: the arrival sphere is 0.005 to 0.05 map units;
#: 2026-10-09 the operator raised the maximum to 0.1.
#: 0.3 was tried first and was 5% of urai's camera trajectory -- far too coarse to
#: distinguish adjacent clicked points.
DEFAULT_ARRIVE_RADIUS_U = 0.02
ARRIVE_RADIUS_MIN_U = 0.005
ARRIVE_RADIUS_MAX_U = 0.1
_ROUTE = "#ff3ea5"
_SELECTED = "#ffe169"
_AXIS = ("#ff6b5f", "#55d187", "#5aa7e8")
_BOX_EDGES = tuple(
    (index, index ^ (1 << bit))
    for index in range(8) for bit in range(3)
    if index < (index ^ (1 << bit))
)


def _action_marker(action: dict) -> str:
    """Compact canvas marker: R30 / L45 turn, H3s hover, C-20 camera pitch."""
    if action.get("type") == "yaw":
        degrees = float(action["degrees"])
        return f"{'R' if degrees > 0 else 'L'}{abs(degrees):g}"
    if action.get("type") == "gimbal_pitch":
        return f"C{float(action['pitch_deg']):+g}"
    return f"H{float(action['seconds']):g}s"


def _waypoint_label(controller, index: int) -> str:
    """Canvas label: number, outbound markers in [..], return markers in ret[..]."""
    label = str(index + 1)
    tables = (
        (" [", getattr(controller, "actions", None) or []),
        (" ret[", getattr(controller, "return_actions", None) or []),
    )
    for prefix, table in tables:
        if index < len(table) and table[index]:
            # ASCII only: PIL's default bitmap font has no CJK glyphs.
            label += prefix + " ".join(_action_marker(a) for a in table[index]) + "]"
    return label


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


def _save_failure_hint(exc: Exception) -> str | None:
    """Explain a rejected map binding, not a managed-profile import dead end."""
    text = str(exc)
    if "adoption" in text or "map SHA-256" in text or "map identity" in text:
        return "地圖內容或座標綁定與場域設定不一致；請重新選擇正確地圖後再儲存"
    return None


def _dispatch_measure_key(window, key: str) -> None:
    if key == "g" and window._measurement_move_start is None:
        window._handle_begin_move()
    elif key in {"x", "y", "z"} and window._measurement_move_start is not None:
        window._handle_move_axis(key)
    elif key in {"return", "kp_enter"} and window._measurement_move_start is not None:
        window._handle_enter()
    elif key == "escape":
        window._handle_escape()


def _dispatch_route_key(window, key: str, ctrl: bool, shift: bool) -> str | None:
    if ctrl and key == "z":
        window._handle_undo_redo(shift=shift)
        return "break"
    if (
        key == "g"
        and window.controller.phase == "height"
        and not window.controller.moving
    ):
        window._handle_begin_move()
        return "break"
    if key in {"x", "y", "z"} and window.controller.moving:
        window._handle_move_axis(key)
        return "break"
    if key in {"return", "kp_enter"}:
        window._handle_enter()
        return "break"
    if key == "escape":
        window._handle_escape()
        return "break"
    if key == "delete" and window.controller.phase == "height":
        window._handle_delete()
        return "break"
    return None


EDITOR_MIN_SIZE = (1040, 640)
_SIDE_WIDTH = 384
_SIDE_WRAP = 334

#: Workflow reference shown in the sidebar's 說明 tab: (section title, lines).
_HELP_SECTIONS = (
    ("地圖", (
        "導入資料夾：自動偵測 PLY",
        "單一檔自動載入，多檔時選擇",
        "SHA-256 匹配場域後才可正式匯入",
    )),
    ("第一階段：標路徑點", (
        "先開啟「標路徑點模式」",
        "左鍵單擊：新增路徑點",
        "Enter：進入第二階段",
    )),
    ("第二階段：逐點調整位置", (
        "左鍵：選取航點",
        "航點動作：到點後轉向／懸停／鏡頭俯仰，再轉往下一點",
        "動作可分去程／回程各自設定（第一點與最後一點沒有回程動作）",
        "G：開始平移；G 後 X / Y / Z：鎖定軸向",
        "左鍵／Enter：確認，Esc：取消",
        "Delete：刪除航點",
        "Ctrl+Z：復原；Ctrl+Shift+Z：重做",
    )),
    ("物體長方體", (
        "開啟物體框選模式，拖曳粗框點雲",
        "點框線／中心選取，G 平移、S 縮放",
        "G／S 後 X／Y／Z：限制單軸",
        "可輸入位移（u）／縮放倍數；縮放時 Shift 細調",
        "Ctrl+C／Ctrl+V：複製／貼上選取的長方體",
        "雙擊點雲：切換地圖旋轉中心",
        "與路線儲存，可設定航段機頭朝向",
    )),
    ("電線", (
        "開啟電線模式，依序單擊點雲／航點選兩端",
        "點線選取，S 後移動滑鼠或輸入垂度（u）",
        "點端點 A／B，G 平移，再用 X／Y／Z 鎖軸",
        "可輸入位移（u），例如 G Z 0.1",
        "左鍵／Enter 確認，Esc／右鍵還原",
        "新增電線按鈕可畫下一條；Delete 刪除",
        "Ctrl+Z 復原，Ctrl+Shift+Z 重做電線",
        "隨航線儲存，不修改點雲、不提供自動避障",
    )),
    ("量距", (
        "開啟兩點量距，單擊點雲／航點選 A、B",
        "第三次單擊開始下一組；Esc 清除並離開",
        "公尺以場域尺度換算，估算非實測標定",
    )),
    ("視角", (
        "左鍵拖曳：旋轉",
        "左鍵雙擊：切換地圖中心",
        "右鍵拖曳：平移",
        "觸控板捏合／滾輪：縮放",
    )),
    ("儲存", (
        "編輯器座標：Z 永遠向上。",
        "新增模式會建立另一條路線；編輯模式只更新所選路線。",
        "「儲存路線」會同步 SHA，並將儲存結果設為目前 AUTO 航線。",
        "綁定地圖並成功儲存後才有航線核准；不會起飛，仍須完成主介面安全確認。",
        "未綁定 PLY 只能另存預覽，不能供 AUTO 使用。",
    )),
)


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
        distance_scale_loader: Callable[[object], MapDistanceScale | None] = load_map_distance_scale,
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
        self.distance_scale_loader = distance_scale_loader
        self.distance_scale = distance_scale_loader(profile)
        self._measuring = False
        self.measurement_points: list[list[float]] = []
        self._measurement_selected: int | None = None
        self._measurement_move_start: list[float] | None = None
        self._measurement_move_axis: int | None = None
        self.map_frame, self.align_source = map_frame, align_source
        # Reopen the route with the radius it was saved with, clamped to the
        # operator's range so the slider and the value can never disagree.
        saved_radius = document.arrive_radius_map_units
        self.arrive_radius_var = tk.DoubleVar(
            value=DEFAULT_ARRIVE_RADIUS_U if saved_radius is None
            else min(ARRIVE_RADIUS_MAX_U, max(ARRIVE_RADIUS_MIN_U, saved_radius)))
        self.point_radius_var = tk.StringVar(value=f"{float(self.arrive_radius_var.get()):.3f}")
        self.discovery_var = tk.StringVar(value="尚未導入資料夾")
        self._discovered = None
        self.show_arrive_var = tk.BooleanVar(value=False)
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
        apply_operator_styles(self)
        self.configure(bg=PANEL)
        self.minsize(*EDITOR_MIN_SIZE)
        self.geometry(
            f"{max(EDITOR_MIN_SIZE[0], min(1280, self.winfo_screenwidth() - 80))}"
            f"x{max(EDITOR_MIN_SIZE[1], min(800, self.winfo_screenheight() - 100))}"
        )
        self.protocol("WM_DELETE_WINDOW", self.close_editor)
        self.controller = RouteEditorController(
            self.document.points, self.document.waypoint_actions,
            self.document.segment_look_at, self.document.return_waypoint_actions,
        )
        self.objects = ObjectBoxController(document.object_boxes)
        self.wires = WireController(document.wires)
        self._wiring = False
        self._wire_create_armed = False
        self._wire_numeric = ""
        self._wire_sag_start = 0.0
        self._wire_mouse_y = 0
        self._boxing = False
        self._box_create_armed = False
        self._box_draw_start: tuple[int, int] | None = None
        self._box_rectangle = None
        self._box_numeric = ""
        self.map_points = self._aligned_map_points(map_points)
        center, radius = self._bounds(self.map_points[:, :3])
        self.view = OrthoView(center=center, radius=radius, zoom=1.0)
        self.view.top()
        self._photo = None
        self._canvas_item = None
        self._redraw_pending = False
        self._render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="route-cloud")
        self._render_future = None
        self._render_after = None
        self._cloud_worker = PointCloudRenderWorker()
        self._cloud_revision = 0
        self._raster_cache_key = None
        self._raster_cache = None
        self._distance_labels_dirty = False
        self._displayed_view = None
        self._displayed_size = None
        self._displayed_cloud = None
        self._nav_button: int | None = None
        self._nav_last: tuple[int, int] | None = None
        self._mouse = (0, 0)
        self._mouse_precise = False
        self._scale_gesture: PointerScale | None = None
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
        self.map_status_var = tk.StringVar()
        self.save_note_var = tk.StringVar()
        self.marking_mode_var = tk.BooleanVar(value=False)
        self.measurement_mode_var = tk.BooleanVar(value=False)
        self.object_mode_var = tk.BooleanVar(value=False)
        self.object_info_var = tk.StringVar()
        self.wire_mode_var = tk.BooleanVar(value=False)
        self.wire_info_var = tk.StringVar()
        self.look_at_start_var = tk.StringVar(value="1")
        self.look_at_end_var = tk.StringVar(value="2")
        self.look_at_object_var = tk.StringVar()
        self.look_at_summary_var = tk.StringVar()
        self._look_at_object_key: int | None = None
        self.distance_scale_var = tk.StringVar()
        self.measurement_var = tk.StringVar()
        self.route_total_var = tk.StringVar()
        initial_status = (
            f"正在編輯 {route_path.name}；儲存後只會更新這條航線"
            if self._editing_current_route
            else "正在新增航線；儲存後會保留所有既有航線"
        )
        self.status_var = tk.StringVar(value=initial_status)
        self._build()
        self._bind_controls()
        self._refresh_labels()
        self._fullscreen = False
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


    @staticmethod
    def _bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
        if not len(points):
            return np.zeros(3, dtype=float), 1.0
        lo = np.percentile(points, 0.1, axis=0)
        hi = np.percentile(points, 99.9, axis=0)
        return (lo + hi) * 0.5, max(float(np.max(hi - lo) * 0.5), 1e-3)

    def _build(self) -> None:
        style = ttk.Style(self)
        bg, card, fg, muted, line = PANEL, CARD, TEXT, MUTED, BORDER
        style.configure("RouteEditor.TFrame", background=bg)
        style.configure("RouteEditor.TLabel", background=bg, foreground=fg)
        style.configure(
            "RouteEditor.Title.TLabel", background=bg, foreground=fg, font=FONT_TITLE,
        )
        style.configure(
            "RouteEditor.Hint.TLabel", background=bg, foreground=muted, font=FONT_SMALL,
        )
        style.configure(
            "RouteEditor.Card.TLabel", background=card, foreground=fg, padding=(10, 6),
            relief="solid", borderwidth=1, bordercolor=line, lightcolor=line,
            darkcolor=line,
        )
        style.configure(
            "RouteEditor.TLabelframe", background=bg, bordercolor=line,
            lightcolor=line, darkcolor=line, relief="solid", borderwidth=1,
        )
        style.configure(
            "RouteEditor.TLabelframe.Label", background=bg, foreground=fg,
            font=FONT_BOLD,
        )
        for kind in ("TCheckbutton", "TRadiobutton"):
            style.configure(f"RouteEditor.{kind}", background=bg, foreground=fg)
            style.map(
                f"RouteEditor.{kind}",
                background=[("active", SURFACE)],
                foreground=[("disabled", DISABLED)],
            )
        style.configure("RouteEditor.TNotebook", background=bg, borderwidth=1)
        style.configure(
            "RouteEditor.TNotebook.Tab", padding=(14, 5),
        )
        # The point cloud owns the whole left area. Its controls float on top as
        # compact cards, so no toolbar or button strip takes space from the map.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        style.configure(
            "RouteEditor.Overlay.TFrame", background=card, relief="solid",
            borderwidth=1, bordercolor=BORDER_STRONG, lightcolor=BORDER_STRONG,
            darkcolor=BORDER_STRONG,
        )
        style.configure("RouteEditor.Overlay.TLabel", background=card, foreground=fg)
        style.configure(
            "RouteEditor.Overlay.Title.TLabel", background=card, foreground=fg,
            font=FONT_TITLE,
        )
        style.configure(
            "RouteEditor.Overlay.Hint.TLabel", background=card, foreground=muted,
            font=FONT_SMALL,
        )
        body = ttk.Frame(self, style="RouteEditor.TFrame")
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=_SIDE_WIDTH)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            body, bg=VIEWPORT_EDITOR_BG, bd=0, highlightthickness=0, cursor="crosshair",
            takefocus=True,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        toolbar = ttk.Frame(self.canvas, style="RouteEditor.Overlay.TFrame", padding=(10, 8))
        toolbar.place(x=12, y=12, anchor="nw")
        ttk.Label(
            toolbar, textvariable=self.title_var, style="RouteEditor.Overlay.Title.TLabel",
            anchor="w", justify="left", wraplength=340,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            toolbar, textvariable=self.map_status_var,
            style="RouteEditor.Overlay.Hint.TLabel", anchor="w", justify="left",
            wraplength=340,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))
        self.ply_button = ttk.Button(
            toolbar, text="導入地圖資料夾", command=self.choose_map_folder
        )
        self.ply_button.grid(row=2, column=0, sticky="ew", padx=(0, 6))
        self.fullscreen_button = ttk.Button(
            toolbar, text="全螢幕 F11", command=self._toggle_fullscreen,
        )
        self.fullscreen_button.grid(row=2, column=1, sticky="ew")

        navigation = ttk.Frame(
            self.canvas, style="RouteEditor.Overlay.TFrame", padding=(10, 6),
        )
        navigation.place(relx=0.0, rely=1.0, x=12, y=-12, anchor="sw")
        ttk.Label(navigation, text="視角", style="RouteEditor.Overlay.Hint.TLabel").pack(
            side="left", padx=(0, 8),
        )
        for label, command in (
            ("俯視", self.set_top_view), ("正視", self.set_front_view),
            ("右視", self.set_right_view), ("左視", self.set_left_view),
        ):
            ttk.Button(navigation, text=label, width=6, command=command).pack(
                side="left", padx=(0, 4),
            )
        ttk.Separator(navigation, orient="vertical").pack(
            side="left", fill="y", padx=(6, 10),
        )
        self.finish_button = ttk.Button(
            navigation, text="完成標點 → 逐點調整", command=self.finish_layout,
        )
        self.finish_button.pack(side="left")

        zoom_controls = ttk.Frame(self.canvas, style="MapOverlay.TFrame", padding=4)
        zoom_controls.place(relx=1.0, x=-12, y=12, anchor="ne")
        self.zoom_in_button = ttk.Button(
            zoom_controls,
            text="＋",
            width=3,
            style="MapZoom.TButton",
            command=lambda: self._set_zoom(self.view.zoom * 1.12),
        )
        self.zoom_in_button.pack(pady=(0, 4))
        self.zoom_out_button = ttk.Button(
            zoom_controls,
            text="－",
            width=3,
            style="MapZoom.TButton",
            command=lambda: self._set_zoom(self.view.zoom / 1.12),
        )
        self.zoom_out_button.pack()

        side_host = ttk.Frame(body, width=_SIDE_WIDTH, style="RouteEditor.TFrame")
        side_host.grid(row=0, column=1, sticky="nsew")
        side_host.pack_propagate(False)

        # Pinned header: phase, marking switch and selection stay visible whichever
        # tab is open, so the operator never scrolls to find the current state.
        header = ttk.Frame(side_host, style="RouteEditor.TFrame")
        header.pack(fill="x", padx=12, pady=(12, 6))
        ttk.Label(
            header, textvariable=self.mode_var, style="RouteEditor.Title.TLabel",
            wraplength=_SIDE_WRAP, justify="left",
        ).pack(anchor="w", pady=(0, 8))
        self.marking_mode_button = ttk.Checkbutton(
            header, text="標路徑點模式", variable=self.marking_mode_var,
            command=self.toggle_marking_mode, style="RouteEditor.TCheckbutton",
        )
        self.marking_mode_button.pack(anchor="w", pady=(0, 8))
        ttk.Label(
            header, textvariable=self.selection_var, style="RouteEditor.Card.TLabel",
            wraplength=_SIDE_WRAP - 16, justify="left",
        ).pack(fill="x", pady=(0, 6))
        # Save/exit and the status line are pinned under the tabs: they keep their own
        # reserved rows, never a shrinking toolbar or tab, and never cover the map.
        side_footer = ttk.Frame(
            side_host, style="RouteEditor.TFrame", padding=(8, 8, 6, 10),
        )
        side_footer.pack(side="bottom", fill="x")
        ttk.Label(
            side_footer, textvariable=self.save_note_var, style="RouteEditor.Hint.TLabel",
            justify="left", wraplength=_SIDE_WRAP,
        ).pack(fill="x", pady=(0, 6))
        footer_actions = ttk.Frame(side_footer, style="RouteEditor.TFrame")
        footer_actions.pack(fill="x")
        self.test_button = None
        footer_column = 0
        if self.test_route_callback is not None:
            self.test_button = ttk.Button(
                footer_actions, text="儲存並模擬", command=self.save_and_test_route,
            )
            self.test_button.grid(row=0, column=footer_column, sticky="ew", padx=(0, 4))
            footer_actions.columnconfigure(footer_column, weight=1)
            footer_column += 1
        self.save_button = ttk.Button(
            footer_actions, text="另存預覽航線" if self._preview_only else "儲存路線",
            command=self.save_route, style="Primary.TButton",
        )
        self.save_button.grid(row=0, column=footer_column, sticky="ew", padx=(0, 4))
        footer_actions.columnconfigure(footer_column, weight=1)
        footer_column += 1
        self.exit_button = ttk.Button(
            footer_actions, text="離開編輯器", command=self.close_editor,
        )
        self.exit_button.grid(row=0, column=footer_column, sticky="ew")
        footer_actions.columnconfigure(footer_column, weight=1)
        # Verbose discovery details live in the scrollable help tab, not this header.
        self.side_tabs = ttk.Notebook(side_host, style="RouteEditor.TNotebook")
        self.side_tabs.pack(fill="both", expand=True, padx=(8, 0))
        self._tab_canvases: list[tk.Canvas] = []
        waypoint_tab = self._side_page("航點")
        object_tab = self._side_page("物體")
        measure_tab = self._side_page("量測")
        help_tab = self._side_page("說明")
        wire_tab = self._side_page("電線")

        # ---- 航點: arrival radius + per-waypoint actions -------------------
        arrive = self._side_section(waypoint_tab, "到達半徑確認")
        ttk.Checkbutton(
            arrive, text="顯示到達球體", variable=self.show_arrive_var,
            command=self.request_redraw, style="RouteEditor.TCheckbutton",
        ).pack(anchor="w", padx=8, pady=(6, 0))
        self.arrive_label_var = tk.StringVar()
        ttk.Label(arrive, textvariable=self.arrive_label_var,
                  style="RouteEditor.TLabel", wraplength=_SIDE_WRAP - 20,
                  justify="left").pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Scale(
            arrive, from_=ARRIVE_RADIUS_MIN_U, to=ARRIVE_RADIUS_MAX_U,
            variable=self.arrive_radius_var, command=self._on_arrive_radius,
        ).pack(fill="x", padx=8, pady=(0, 6))
        point_row = ttk.Frame(arrive, style="RouteEditor.TFrame")
        point_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(point_row, text="全部航點半徑", style="RouteEditor.TLabel").pack(side="left")
        self.point_radius_box = ttk.Spinbox(
            point_row, from_=ARRIVE_RADIUS_MIN_U, to=ARRIVE_RADIUS_MAX_U,
            increment=0.001, textvariable=self.point_radius_var, width=7,
            command=self._on_point_radius,
        )
        self.point_radius_box.pack(side="left", padx=6)
        self.point_radius_box.bind("<Return>", self._on_point_radius)
        self.point_radius_box.bind("<FocusOut>", self._on_point_radius)
        self.point_radius_note_var = tk.StringVar()
        ttk.Label(arrive, textvariable=self.point_radius_note_var,
                  style="RouteEditor.Hint.TLabel", wraplength=_SIDE_WRAP - 20,
                  justify="left").pack(anchor="w", padx=8, pady=(0, 8))
        self._on_arrive_radius(None)

        actions = self._side_section(waypoint_tab, "到點動作（依序執行）")
        direction_row = ttk.Frame(actions, style="RouteEditor.TFrame")
        direction_row.pack(fill="x", padx=8, pady=(6, 4))
        self.action_direction_var = tk.StringVar(value="out")
        ttk.Label(direction_row, text="設定哪一趟：", style="RouteEditor.TLabel").pack(side="left")
        for direction_label, direction_value in (("去程", "out"), ("回程", "back")):
            ttk.Radiobutton(
                direction_row, text=direction_label, value=direction_value,
                variable=self.action_direction_var, command=self._sync_action_list,
                style="RouteEditor.TRadiobutton",
            ).pack(side="left", padx=6)
        self.action_list_var = tk.StringVar()
        tk.Label(
            actions, textvariable=self.action_list_var, bg=FIELD, fg=TEXT,
            justify="left", anchor="nw", wraplength=_SIDE_WRAP - 28, padx=8, pady=5,
            highlightthickness=1, highlightbackground=BORDER,
        ).pack(fill="x", padx=8, pady=(0, 6))
        controls = ttk.Frame(actions, style="RouteEditor.TFrame")
        controls.pack(fill="x", padx=8)
        controls.columnconfigure(2, weight=1, uniform="action")
        controls.columnconfigure(3, weight=1, uniform="action")
        self.action_degrees_var = tk.StringVar(value="30")
        self.action_seconds_var = tk.StringVar(value="3")
        self.action_pitch_var = tk.StringVar(value="-20")
        ttk.Spinbox(controls, from_=1, to=180, increment=5, width=6,
                    textvariable=self.action_degrees_var).grid(row=0, column=0, pady=2)
        ttk.Label(controls, text="度", style="RouteEditor.TLabel").grid(
            row=0, column=1, padx=(3, 8))
        ttk.Button(controls, text="左轉", width=-4,
                   command=lambda: self._add_yaw_action(-1)).grid(
            row=0, column=2, sticky="ew", padx=(0, 3), pady=2)
        ttk.Button(controls, text="右轉", width=-4,
                   command=lambda: self._add_yaw_action(+1)).grid(
            row=0, column=3, sticky="ew", pady=2)
        ttk.Spinbox(controls, from_=0.5, to=60, increment=0.5, width=6,
                    textvariable=self.action_seconds_var).grid(row=1, column=0, pady=2)
        ttk.Label(controls, text="秒", style="RouteEditor.TLabel").grid(
            row=1, column=1, padx=(3, 8))
        ttk.Button(controls, text="懸停", command=self._add_hover_action).grid(
            row=1, column=2, columnspan=2, sticky="ew", pady=2)
        ttk.Spinbox(controls, from_=-90, to=90, increment=5, width=6,
                    textvariable=self.action_pitch_var).grid(row=2, column=0, pady=2)
        ttk.Label(controls, text="°", style="RouteEditor.TLabel").grid(
            row=2, column=1, padx=(3, 8))
        ttk.Button(controls, text="鏡頭俯仰°",
                   command=self._add_gimbal_pitch_action).grid(
            row=2, column=2, columnspan=2, sticky="ew", pady=2)
        ttk.Label(actions, text="鏡頭絕對角度：負向下／正向上／0 水平",
                  style="RouteEditor.Hint.TLabel", wraplength=_SIDE_WRAP - 20,
                  justify="left").pack(anchor="w", padx=8, pady=(2, 4))
        ttk.Button(actions, text="刪除最後一個動作",
                   command=self._remove_last_action).pack(fill="x", padx=8, pady=(0, 8))

        # ---- 物體: boxes + look-at -----------------------------------------
        objects = self._side_section(object_tab, "物體長方體（不提供避障）")
        self.object_mode_button = ttk.Checkbutton(
            objects, text="物體框選模式", variable=self.object_mode_var,
            command=self.toggle_object_mode, style="RouteEditor.TCheckbutton",
        )
        self.object_mode_button.pack(anchor="w", padx=8, pady=(6, 4))
        object_buttons = ttk.Frame(objects, style="RouteEditor.TFrame")
        object_buttons.pack(fill="x", padx=8)
        object_buttons.columnconfigure(0, weight=1, uniform="object")
        object_buttons.columnconfigure(1, weight=0)
        ttk.Button(
            object_buttons, text="新增長方體（拖曳框選）", command=self.arm_object_box,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(
            object_buttons, text="刪除選取長方體", command=self._delete_object_box,
        ).grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            objects, textvariable=self.object_info_var, style="RouteEditor.Card.TLabel",
            wraplength=_SIDE_WRAP - 36, justify="left",
        ).pack(fill="x", padx=8, pady=(6, 4))
        ttk.Label(
            objects,
            text="G 平移／S 縮放，再按 X／Y／Z 鎖軸\n"
                 "G Z 1：上移 1u；S X 2：X 長度加倍\n"
                 "左鍵／Enter 確認；Esc／右鍵取消\n"
                 "Ctrl+Z 復原；Ctrl+Shift+Z 重做",
            style="RouteEditor.Hint.TLabel", wraplength=_SIDE_WRAP - 20, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 8))

        look_at = self._side_section(object_tab, "航段機頭朝向物體")
        span = ttk.Frame(look_at, style="RouteEditor.TFrame")
        span.pack(fill="x", padx=8, pady=(6, 4))
        ttk.Label(span, text="航點", style="RouteEditor.TLabel").pack(side="left")
        self.look_at_start_box = ttk.Combobox(
            span, textvariable=self.look_at_start_var, state="readonly", width=4,
        )
        self.look_at_start_box.pack(side="left", padx=3)
        ttk.Label(span, text="→", style="RouteEditor.TLabel").pack(side="left")
        self.look_at_end_box = ttk.Combobox(
            span, textvariable=self.look_at_end_var, state="readonly", width=4,
        )
        self.look_at_end_box.pack(side="left", padx=3)
        self.look_at_object_box = ttk.Combobox(
            look_at, textvariable=self.look_at_object_var, state="readonly", width=20,
        )
        self.look_at_object_box.pack(fill="x", padx=8, pady=2)
        self.look_at_object_box.bind("<<ComboboxSelected>>", self._select_look_at_object)
        look_buttons = ttk.Frame(look_at, style="RouteEditor.TFrame")
        look_buttons.pack(fill="x", padx=8, pady=(4, 0))
        look_buttons.columnconfigure(0, weight=1, uniform="look")
        look_buttons.columnconfigure(1, weight=0)
        self.look_at_apply_button = ttk.Button(
            look_buttons, text="套用所選區間朝向", command=self._apply_look_at_range,
        )
        self.look_at_apply_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.look_at_clear_button = ttk.Button(
            look_buttons, text="清除所選區間朝向",
            command=lambda: self._apply_look_at_range(clear=True),
        )
        self.look_at_clear_button.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            look_at, textvariable=self.look_at_summary_var,
            style="RouteEditor.Card.TLabel", wraplength=_SIDE_WRAP - 36, justify="left",
        ).pack(fill="x", padx=8, pady=(6, 4))
        ttk.Label(
            look_at,
            text="區間內移動時持續朝向長方體中心。\n"
                 "只控制機頭水平朝向，不調整鏡頭。\n"
                 "到點動作優先，只在去程朝向，返程維持一般航向。",
            style="RouteEditor.Hint.TLabel", wraplength=_SIDE_WRAP - 20, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 8))

        wires = self._side_section(wire_tab, "手繪電線（不提供避障）")
        self.wire_mode_button = ttk.Checkbutton(
            wires, text="電線繪製模式", variable=self.wire_mode_var,
            command=self.toggle_wire_mode, style="RouteEditor.TCheckbutton",
        )
        self.wire_mode_button.pack(anchor="w", padx=8, pady=(6, 4))
        ttk.Button(
            wires, text="新增電線（兩點）", command=self.arm_wire,
        ).pack(fill="x", padx=8, pady=4)
        ttk.Button(
            wires, text="移動選取端點（G）", command=self._begin_wire_move,
        ).pack(fill="x", padx=8, pady=4)
        ttk.Button(
            wires, text="調整選取電線垂度（S）", command=self._begin_wire_sag,
        ).pack(fill="x", padx=8, pady=4)
        ttk.Button(
            wires, text="刪除選取電線", command=self._delete_wire,
        ).pack(fill="x", padx=8, pady=4)
        ttk.Label(
            wires, textvariable=self.wire_info_var, style="RouteEditor.Card.TLabel",
            wraplength=_SIDE_WRAP - 36, justify="left",
        ).pack(fill="x", padx=8, pady=6)
        ttk.Label(
            wires, text="依序單擊點雲／航點選兩端，建立直線。\n"
            "點端點 A／B，G 平移，再按 X／Y／Z 鎖軸；可輸入位移（u）。\n"
            "點線選取，S 調垂度：滑鼠下移加深、上移拉直；也可輸入垂度（u）。\n"
            "左鍵／Enter 確認；Esc／右鍵還原。\n"
            "Delete 刪除；Ctrl+Z 復原，Ctrl+Shift+Z 重做。\n"
            "新增中 Esc／右鍵取消端點；滾輪仍縮放。\n"
            "與航線儲存，不修改原始點雲、不啟用自動避障。",
            style="RouteEditor.Hint.TLabel", wraplength=_SIDE_WRAP - 36, justify="left",
        ).pack(fill="x", padx=8, pady=(0, 8))

        # ---- 量測: two-point distance + leg lengths -------------------------
        measurement = self._side_section(measure_tab, "兩點量距（3D 直線）")
        self.measurement_mode_button = ttk.Checkbutton(
            measurement, text="兩點量距模式", variable=self.measurement_mode_var,
            command=self.toggle_measurement_mode, style="RouteEditor.TCheckbutton",
        )
        self.measurement_mode_button.pack(anchor="w", padx=8, pady=(6, 4))
        ttk.Button(
            measurement, text="清除測量", command=self.clear_measurement
        ).pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(
            measurement, textvariable=self.measurement_var,
            style="RouteEditor.Card.TLabel", wraplength=_SIDE_WRAP - 36, justify="left",
        ).pack(fill="x", padx=8, pady=(2, 4))
        ttk.Label(
            measurement, textvariable=self.distance_scale_var,
            style="RouteEditor.Hint.TLabel", wraplength=_SIDE_WRAP - 20, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 8))

        lengths = self._side_section(measure_tab, "完整路線航段（不閉合）")
        ttk.Label(
            lengths, textvariable=self.route_total_var,
            style="RouteEditor.TLabel", wraplength=_SIDE_WRAP - 20,
        ).pack(anchor="w", padx=8, pady=(6, 4))
        leg_rows = ttk.Frame(lengths, style="RouteEditor.TFrame")
        leg_rows.pack(fill="x", padx=8, pady=(0, 6))
        self.route_distance_list = tk.Listbox(
            leg_rows, height=8, width=24, **TEXT_BOX,
            activestyle="none", exportselection=False,
        )
        leg_scroll = ttk.Scrollbar(
            leg_rows, orient="vertical", command=self.route_distance_list.yview
        )
        leg_scroll.pack(side="right", fill="y")
        leg_horizontal = ttk.Scrollbar(
            lengths, orient="horizontal", command=self.route_distance_list.xview
        )
        leg_horizontal.pack(fill="x", padx=8, pady=(0, 8))
        self.route_distance_list.pack(side="left", fill="x", expand=True)
        self.route_distance_list.configure(
            yscrollcommand=leg_scroll.set, xscrollcommand=leg_horizontal.set
        )

        # ---- 說明: workflow reference ----------------------------------------
        discovery = self._side_section(help_tab, "目前地圖資產")
        ttk.Label(
            discovery, textvariable=self.discovery_var, style="RouteEditor.Hint.TLabel",
            wraplength=_SIDE_WRAP - 20, justify="left",
        ).pack(fill="x", padx=8, pady=(6, 8))
        for title, lines in _HELP_SECTIONS:
            section = self._side_section(help_tab, title)
            ttk.Label(
                section, text="\n".join(lines), style="RouteEditor.TLabel",
                justify="left", wraplength=_SIDE_WRAP - 20,
            ).pack(anchor="w", padx=8, pady=(6, 8))

        # Jump to the tab a mode belongs to when it is switched on (edge only, so the
        # per-refresh variable sync never yanks the operator away from another tab).
        self._follow_mode_tab(self.object_mode_var, 1)
        self._follow_mode_tab(self.measurement_mode_var, 2)
        self._follow_mode_tab(self.wire_mode_var, 4)
        self._bind_sidebar_scroll(self.side_tabs)

        status_frame = ttk.Frame(side_footer, style="RouteEditor.TFrame")
        status_frame.pack(fill="x", pady=(8, 0))
        self.status_text = tk.Text(
            status_frame, height=3, width=1, wrap="word", **TEXT_BOX,
            padx=8, pady=3, state="disabled",
        )
        status_scroll = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        status_scroll.pack(side="right", fill="y")
        self.status_text.pack(side="left", fill="x", expand=True)
        self.status_text.configure(yscrollcommand=status_scroll.set)

        def show_status(*_args):
            self.status_text.configure(state="normal")
            self.status_text.delete("1.0", "end")
            self.status_text.insert("1.0", self.status_var.get())
            self.status_text.configure(state="disabled")

        self.status_var.trace_add("write", show_status)
        show_status()

    def _toggle_fullscreen(self, _event=None) -> str:
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)
        self.fullscreen_button.configure(
            text="還原視窗 F11" if self._fullscreen else "全螢幕 F11",
        )
        return "break"

    def _side_page(self, title: str) -> ttk.Frame:
        """One sidebar tab; a scrollbar appears only when the content overflows."""
        page = ttk.Frame(self.side_tabs, style="RouteEditor.TFrame")
        self.side_tabs.add(page, text=title)
        canvas = tk.Canvas(page, bg=PANEL, bd=0, highlightthickness=0)
        bar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas, style="RouteEditor.TFrame")
        item = canvas.create_window(0, 0, window=inner, anchor="nw")

        def sync(_event=None) -> None:
            width = canvas.winfo_width()
            if width > 1:
                canvas.itemconfigure(item, width=width)
            canvas.configure(scrollregion=canvas.bbox("all"))
            overflow = inner.winfo_reqheight() > canvas.winfo_height() > 1
            if overflow and not bar.winfo_ismapped():
                bar.pack(side="right", fill="y", before=canvas)
            elif not overflow and bar.winfo_ismapped():
                bar.pack_forget()
                canvas.yview_moveto(0.0)

        inner.bind("<Configure>", sync)
        canvas.bind("<Configure>", sync)
        self._tab_canvases.append(canvas)
        return inner

    @staticmethod
    def _side_section(parent, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, style="RouteEditor.TLabelframe")
        frame.pack(fill="x", padx=(8, 10), pady=(8, 2))
        return frame

    def _follow_mode_tab(self, variable: tk.BooleanVar, index: int) -> None:
        was_on = [bool(variable.get())]

        def changed(*_args) -> None:
            on = bool(variable.get())
            if on and not was_on[0]:
                self.side_tabs.select(index)
            was_on[0] = on

        variable.trace_add("write", changed)

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
        self.bind("<F11>", self._toggle_fullscreen)
        self._pinch_zoom = install_x11_pinch_zoom(
            self.canvas,
            lambda: self.view.zoom,
            self._set_zoom,
        )

    def _bind_sidebar_scroll(self, widget) -> None:
        """Keep every sidebar control reachable without stealing map zoom."""
        if widget is not self.route_distance_list:
            widget.bind("<MouseWheel>", self._scroll_sidebar, add="+")
            widget.bind("<Button-4>", self._scroll_sidebar, add="+")
            widget.bind("<Button-5>", self._scroll_sidebar, add="+")
        widget.bind("<FocusIn>", self._reveal_sidebar_focus, add="+")
        for child in widget.winfo_children():
            self._bind_sidebar_scroll(child)

    def _scroll_sidebar(self, event) -> str:
        up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        canvas = self._tab_canvases[self.side_tabs.index("current")]
        if canvas.yview() != (0.0, 1.0):
            canvas.yview_scroll(-3 if up else 3, "units")
        return "break"

    def _reveal_sidebar_focus(self, event) -> None:
        """Keyboard tabbing reveals the focused control inside its own page."""
        canvas = self._tab_canvases[self.side_tabs.index("current")]
        widget = event.widget
        # Tk also sends FocusIn to the focused control's ancestors. Revealing
        # those oversized frames would scroll repeatedly before Tk moves them.
        if widget is not self.focus_get():
            return
        parent = widget
        while parent is not None and parent is not canvas:
            parent = getattr(parent, "master", None)
        if parent is None:
            return
        top = widget.winfo_rooty() - canvas.winfo_rooty()
        bottom = top + widget.winfo_height()
        height = canvas.winfo_height()
        bounds = canvas.bbox("all")
        if bounds is None or bounds[3] <= height:
            return
        offset = top - 6 if top < 0 else bottom - height + 6 if bottom > height else 0
        if offset:
            canvas.yview_moveto((canvas.canvasy(0) + offset) / bounds[3])

    def _signature(self) -> tuple:
        points = tuple(
            tuple(float(value) for value in point) for point in self.controller.points
        )
        return (
            points,
            float(self.arrive_radius_var.get()),
            tuple(self.document.waypoint_arrive_radii or ()),
            repr(self.controller.actions),
            repr(self.controller.return_actions),
            repr(self.objects.boxes),
            repr(self.wires.wires),
            repr(self.wires.pending),
            tuple(self.controller.segment_look_at),
            tuple(self.objects.keys),
        )

    def _refresh_labels(self, *, refresh_distances: bool = True) -> None:
        phase = (
            "第一階段：標路徑點"
            if self.controller.phase == "layout"
            else "第二階段：逐點調整位置"
        )
        source = (
            "未綁定 PLY｜僅供預覽，不能供 AUTO 使用"
            if self._preview_only
            else self.profile.display_name
        )
        self.mode_var.set(f"{phase}｜{len(self.controller.points)} 點")
        self.map_status_var.set(f"地圖：{source}")
        self.save_note_var.set(
            "僅另存預覽，不核准 AUTO，不會起飛。"
            if self._preview_only else
            "成功儲存後核准此航線；不會起飛，仍須完成主介面安全確認。"
        )
        selected = self.controller.selected
        if selected is None or not (0 <= selected < len(self.controller.points)):
            self.selection_var.set("選取：無")
        else:
            x, y, z = self.controller.points[selected]
            self.selection_var.set(
                f"選取：航點 {selected + 1}\nX {x:.3f}  Y {y:.3f}  高度 Z {z:.3f}"
            )
        self._sync_point_radius_box()
        self._sync_action_list()
        self.finish_button.configure(
            state="normal" if self.controller.phase == "layout" else "disabled"
        )
        self.marking_mode_var.set(self.controller.placement_enabled)
        self.marking_mode_button.configure(
            state="normal" if self.controller.phase == "layout" else "disabled"
        )
        self.measurement_mode_var.set(self._measuring)
        self.measurement_mode_button.configure(
            state="disabled" if self._asset_loading or self.controller.moving else "normal"
        )
        if refresh_distances:
            self._refresh_distance_labels()
        else:
            self._distance_labels_dirty = True
        self._refresh_object_labels()
        self._refresh_wire_labels()
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

    def _current_raster_key(self) -> tuple:
        width, height = self._canvas_size()
        return cloud_raster_key(
            self.map_points, self._cloud_revision, self.view, width, height
        )

    def request_redraw(self) -> None:
        if self._closed:
            return
        if self._raster_cache_key != self._current_raster_key():
            self._raster_cache_key = None
            self._raster_cache = None
        if self._redraw_pending:
            return
        self._redraw_pending = True
        # One active frame plus one latest-view request, never a drag-event backlog.
        if self._render_future is None:
            self._render_after = self.after(16, self._redraw)

    def _redraw(self) -> None:
        self._render_after = None
        if self._closed:
            return
        self._redraw_pending = False
        if self._distance_labels_dirty:
            self._refresh_distance_labels()
        width, height = self._canvas_size()
        view = replace(self.view, center=self.view.center.copy())
        points = self.map_points
        revision = self._cloud_revision
        key = cloud_raster_key(points, revision, view, width, height)
        if key == self._raster_cache_key:
            self._paint_render(self._raster_cache, view, points, width, height)
            return
        self._raster_cache_key = None
        self._raster_cache = None
        self._render_future = self._render_executor.submit(
            self._cloud_worker.render, points, revision, view, width, height
        )
        self._render_after = self.after(
            8, lambda: self._poll_render(key, view, points, width, height)
        )

    def _poll_render(self, key, view, points, width, height) -> None:
        self._render_after = None
        if self._closed:
            return
        # Sidebar previews must advance even while the cloud worker is blocked.
        if self._distance_labels_dirty:
            self._refresh_distance_labels()
        if not self._render_future.done():
            self._render_after = self.after(
                8, lambda: self._poll_render(key, view, points, width, height)
            )
            return
        future = self._render_future
        self._render_future = None
        try:
            arr = future.result()
        except Exception as exc:
            self.status_var.set(f"完整點雲繪製失敗：{exc}")
        else:
            if key == self._current_raster_key():
                self._raster_cache = Image.fromarray(arr)
                self._raster_cache_key = key
                self._paint_render(self._raster_cache, view, points, width, height)
            else:
                self._redraw_pending = True
        if self._redraw_pending:
            self._render_after = self.after_idle(self._redraw)

    def _paint_render(self, raster, view, points, width, height) -> None:
        # Overlays always start from the pristine cloud, never the previous frame.
        image = raster.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        self._draw_axes(draw, width, height, view)
        self._draw_route(draw, width, height, view)
        self._draw_measurement(draw, width, height, view)
        self._draw_object_boxes(draw, width, height, view)
        self._draw_wires(draw, width, height, view)
        photo = ImageTk.PhotoImage(image)
        self._photo = photo
        self._displayed_view = view
        self._displayed_size = (width, height)
        self._displayed_cloud = points
        if self._canvas_item is None:
            self._canvas_item = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self._canvas_item, image=photo)

    def _draw_axes(
        self, draw: ImageDraw.ImageDraw, width: int, height: int, view: OrthoView | None = None
    ) -> None:
        view = self.view if view is None else view
        origin = view.center
        size = view.radius * 0.16
        points = np.vstack((origin, origin + np.eye(3) * size))
        sx, sy, _ = view.project(points, width, height)
        for axis, label in enumerate(("X", "Y", "Z")):
            draw.line(
                (sx[0], sy[0], sx[axis + 1], sy[axis + 1]), fill=_AXIS[axis], width=3
            )
            draw.text((sx[axis + 1] + 4, sy[axis + 1] + 3), label, fill=_AXIS[axis])

    def _draw_route(
        self, draw: ImageDraw.ImageDraw, width: int, height: int, view: OrthoView | None = None
    ) -> None:
        view = self.view if view is None else view
        if not self.controller.points:
            return
        sx, sy, _ = view.project(self.controller.points, width, height)
        projected = list(zip(sx.tolist(), sy.tolist()))
        if len(projected) > 1:
            draw.line(projected, fill=_ROUTE, width=3, joint="curve")
            for index, target in enumerate(self.controller.segment_look_at):
                if target is None or target not in self.objects.keys:
                    continue
                first, second = projected[index], projected[index + 1]
                draw.line((first, second), fill="#70e5ff", width=4)
                object_index = self.objects.keys.index(target)
                draw.text(
                    ((first[0] + second[0]) * 0.5 + 4, (first[1] + second[1]) * 0.5 - 14),
                    f"B{object_index + 1}", fill="#70e5ff",
                )
        if self.show_arrive_var.get():
            # The arrival sphere in MAP units, drawn at the view's own scale so
            # what the operator confirms here is what the controller will use.
            # Per-point drawn values win where the operator set them.
            width_px, height_px = width, height
            unit = view.scale(width_px, height_px)
            for index, (x, y) in enumerate(projected):
                pixels = self._point_radius_for(index) * unit
                if pixels < 1.0:
                    continue
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
            label = _waypoint_label(self.controller, index)
            draw.text((x + radius + 3, y - radius - 2), label, fill="white")

    def _refresh_wire_labels(self) -> None:
        self.wire_mode_var.set(self._wiring)
        busy = (
            self._asset_loading or self.controller.moving or self.objects.transforming
            or self._measurement_move_start is not None or self._box_draw_start is not None
        )
        self.wire_mode_button.configure(state="disabled" if busy else "normal")
        selected = self.wires.selected
        text = f"電線數：{len(self.wires.wires)}"
        if selected is not None:
            wire = self.wires.wires[selected]
            text += f"\n選取電線 {selected + 1}；垂度 {wire['sag']:.5g} u"
            endpoint = self.wires.selected_endpoint
            if endpoint is not None:
                x, y, z = wire[endpoint]
                text += f"\n端點 {endpoint.upper()}：X {x:.5g}　Y {y:.5g}　Z {z:.5g}"
            if self.wires.editing:
                axis = "自由" if self.wires.move_axis is None else "XYZ"[self.wires.move_axis]
                operation = f"端點平移（{axis}）" if self.wires.moving else "垂度調整"
                text += f"\n{operation}；輸入：{self._wire_numeric or '滑鼠'}"
        if self._wire_create_armed:
            text += "\n請選端點 B" if self.wires.pending is not None else "\n請選端點 A"
        self.wire_info_var.set(text)

    def _leave_wire_mode(self) -> bool:
        if self.wires.editing or self.wires.pending is not None:
            self.status_var.set("請先確認或取消電線編輯，再切換模式")
            return False
        self._wiring = False
        self._wire_create_armed = False
        self.wire_mode_var.set(False)
        return True

    def toggle_wire_mode(self) -> None:
        busy = (
            self._asset_loading or self.controller.moving or self.objects.transforming
            or self._measurement_move_start is not None or self._box_draw_start is not None
        )
        if busy:
            self.wire_mode_var.set(self._wiring)
            self.status_var.set("請先確認或取消目前編輯，並等待地圖載入完成")
            return
        if not self.wire_mode_var.get():
            if not self._leave_wire_mode():
                self.wire_mode_var.set(True)
                return
        else:
            self._wiring = True
            self._boxing = False
            self._box_create_armed = False
            self._measuring = False
            self.controller.set_placement_enabled(False)
            self._wire_create_armed = not self.wires.wires
        self._cancel_pending_single_click()
        self.status_var.set(
            "電線模式：點端點後 G／X／Y／Z 微調，點線後 S 調垂度"
            if self._wiring else "已離開電線模式，保留已完成電線"
        )
        self._refresh_labels()
        self.request_redraw()
        self.canvas.focus_set()

    def arm_wire(self) -> None:
        if not self._wiring:
            self.wire_mode_var.set(True)
            self.toggle_wire_mode()
        if not self._wiring:
            return
        if self.wires.editing or self.wires.pending is not None:
            self.status_var.set("請先確認或取消目前電線編輯")
            return
        self._cancel_pending_single_click()
        self._wire_create_armed = True
        self.status_var.set("請依序單擊點雲／航點選兩端；Esc／右鍵取消")
        self._refresh_wire_labels()
        self.canvas.focus_set()

    def _draw_wires(self, draw, width, height, view) -> None:
        for index, wire in enumerate(self.wires.wires):
            sx, sy, _ = view.project(sample_wire(wire), width, height)
            color = wire_color(index)
            projected = list(zip(sx, sy))
            if self._wiring and index == self.wires.selected:
                draw.line(projected, fill="white", width=6)
            draw.line(projected, fill=color, width=3)
            for endpoint, position in (("a", 0), ("b", -1)):
                x, y = sx[position], sy[position]
                selected = (
                    self._wiring and index == self.wires.selected
                    and endpoint == self.wires.selected_endpoint
                )
                radius = 7 if selected else 4
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color, outline="white" if selected else color, width=2,
                )
                if self._wiring and index == self.wires.selected:
                    draw.text((x + 9, y - 14), endpoint.upper(), fill=color)
            middle = len(sx) // 2
            draw.text((sx[middle] + 6, sy[middle] + 6), f"W{index + 1}", fill=color)
        if self.wires.pending is not None:
            sx, sy, _ = view.project([self.wires.pending], width, height)
            x, y = sx[0], sy[0]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=_SELECTED, width=2)
            draw.text((x + 8, y - 8), "A", fill=_SELECTED)

    def _nearest_wire(self, x, y) -> int | None:
        view = self._displayed_view or self.view
        width, height = self._displayed_size or self._canvas_size()
        pointer = np.array([x, y], dtype=float)
        best, selected = 14.0 ** 2, None
        for index, wire in enumerate(self.wires.wires):
            sx, sy, _ = view.project(sample_wire(wire), width, height)
            points = np.column_stack((sx, sy))
            starts, delta = points[:-1], np.diff(points, axis=0)
            squared = np.sum(delta * delta, axis=1)
            t = np.clip(np.sum((pointer - starts) * delta, axis=1) / np.maximum(squared, 1e-12), 0, 1)
            distance = float(np.min(np.sum((starts + t[:, None] * delta - pointer) ** 2, axis=1)))
            if distance <= best:
                best, selected = distance, index
        return selected

    def _nearest_wire_endpoint(self, x, y) -> tuple[int, str] | None:
        view = self._displayed_view or self.view
        width, height = self._displayed_size or self._canvas_size()
        best, selected = 14.0 ** 2, None
        for index, wire in enumerate(self.wires.wires):
            sx, sy, _ = view.project([wire["a"], wire["b"]], width, height)
            for position, endpoint in enumerate(("a", "b")):
                distance = (sx[position] - x) ** 2 + (sy[position] - y) ** 2
                if distance <= best:
                    best, selected = distance, (index, endpoint)
        return selected

    def _wire_at(self, x, y) -> None:
        if not self._wire_create_armed:
            endpoint = self._nearest_wire_endpoint(x, y)
            if endpoint is not None:
                self.wires.selected, self.wires.selected_endpoint = endpoint
                self.status_var.set(
                    f"已選端點 {endpoint[1].upper()}；G 平移，再按 X／Y／Z 鎖軸；S 調垂度"
                )
            else:
                self.wires.selected = self._nearest_wire(x, y)
                self.wires.selected_endpoint = None
                self.status_var.set(
                    "已選電線；S 調整垂度，點端點後 G 平移，Delete 刪除"
                    if self.wires.selected is not None else "請點電線／端點，或按「新增電線（兩點）」"
                )
        else:
            waypoint = self._nearest_waypoint(x, y)
            point = self.controller.points[waypoint] if waypoint is not None else self._nearest_cloud(x, y)
            if point is None:
                self.status_var.set("附近沒有點雲或航點；請放大後選取電線端點")
                return
            try:
                finished = self.wires.pick_endpoint(point)
            except ValueError as exc:
                self.status_var.set(str(exc))
                return
            self._wire_create_armed = not finished
            self.status_var.set(
                "已建立直線；按 S 調整垂度，或按「新增電線」畫下一條"
                if finished else "已選端點 A；請單擊選端點 B"
            )
        self._refresh_wire_labels()
        self.request_redraw()

    def _begin_wire_move(self) -> None:
        if (
            not self._wiring or self._asset_loading or self._wire_create_armed
            or not self.wires.begin_move()
        ):
            self.status_var.set("請先在電線模式點選端點 A 或 B，再按 G")
            return
        self._cancel_pending_single_click()
        self._wire_numeric = ""
        self._move_mouse_start = self._mouse
        self._wire_move_view = replace(self.view, center=self.view.center.copy())
        self._wire_move_size = self._canvas_size()
        self.status_var.set("端點平移：X／Y／Z 鎖軸，可輸入位移（u）；左鍵／Enter 確認，Esc／右鍵還原")
        self._refresh_wire_labels()
        self.canvas.focus_set()

    def _begin_wire_sag(self) -> None:
        if not self._wiring or self._asset_loading or self._wire_create_armed or not self.wires.begin_sag():
            self.status_var.set("請在電線模式完成兩端並選取電線，再按 S")
            return
        self._cancel_pending_single_click()
        self._wire_numeric = ""
        self._wire_sag_start = self.wires.wires[self.wires.selected]["sag"]
        self._wire_mouse_y = self._mouse[1]
        self._wire_scale = self.view.scale(*self._canvas_size())
        self.status_var.set("垂度調整：滑鼠下移加深、上移拉直；可輸入 u；左鍵／Enter 確認，Esc／右鍵還原")
        self._refresh_wire_labels()
        self.canvas.focus_set()

    def _preview_wire_edit(self) -> bool:
        try:
            if self.wires.moving:
                if self._wire_numeric:
                    value = float(self._wire_numeric)
                    if not math.isfinite(value) or self.wires.move_axis is None:
                        raise ValueError("數字位移必須有限，並先按 X／Y／Z 選軸")
                    delta = np.zeros(3)
                    delta[self.wires.move_axis] = value
                else:
                    dx = self._mouse[0] - self._move_mouse_start[0]
                    dy = self._mouse[1] - self._move_mouse_start[1]
                    delta = self._wire_move_view.screen_delta_to_world(
                        dx, dy, *self._wire_move_size, self.wires.move_axis,
                    )
                self.wires.preview_move(delta)
            else:
                value = (
                    float(self._wire_numeric) if self._wire_numeric else
                    max(0.0, self._wire_sag_start + (self._mouse[1] - self._wire_mouse_y) / self._wire_scale)
                )
                self.wires.preview_sag(value)
        except (ValueError, OverflowError) as exc:
            self.status_var.set(f"電線數值未套用：{exc}；Backspace 修改，Esc 取消")
            self._refresh_wire_labels()
            return False
        self._refresh_wire_labels()
        self.request_redraw()
        return True

    def _confirm_wire_edit(self) -> None:
        if not self._preview_wire_edit():
            return
        self._cancel_pending_single_click()
        self.wires.confirm()
        self._wire_numeric = ""
        self.status_var.set("已確認電線編輯；儲存路線會一併保留端點及垂度")
        self._refresh_wire_labels()
        self.request_redraw()

    def _wire_escape(self) -> None:
        self._cancel_pending_single_click()
        if self.wires.cancel():
            self.status_var.set("已還原本次電線編輯；仍在電線模式")
        elif self._wire_create_armed:
            self.wires.pending = None
            self._wire_create_armed = False
            self.status_var.set("已取消新增電線，保留已完成電線")
        else:
            self._leave_wire_mode()
            self.status_var.set("已離開電線模式")
        self._wire_numeric = ""
        self._refresh_labels()
        self.request_redraw()

    def _delete_wire(self) -> None:
        if self._wiring and not self._wire_create_armed and self.wires.delete_selected():
            self.status_var.set("已刪除選取電線；航點未變更")
            self._refresh_wire_labels()
            self.request_redraw()
        else:
            self.status_var.set("請先完成或取消電線編輯，再選取電線刪除")
        self.canvas.focus_set()

    def _on_wire_key(self, event, key, ctrl, shift) -> str:
        if key == "escape":
            self._wire_escape()
        elif ctrl and key == "z":
            self._cancel_pending_single_click()
            changed = self.wires.redo() if shift else self.wires.undo()
            self._wire_numeric = ""
            self._wire_create_armed = False
            self.status_var.set(("已重做" if shift else "已復原") if changed else "沒有可復原／重做的電線修改")
            self._refresh_wire_labels()
            self.request_redraw()
        elif not ctrl and key == "g" and not self.wires.editing:
            self._begin_wire_move()
        elif not ctrl and key == "s" and not self.wires.editing:
            self._begin_wire_sag()
        elif not ctrl and key in {"x", "y", "z"} and self.wires.moving:
            self.wires.constrain({"x": 0, "y": 1, "z": 2}[key])
            self._preview_wire_edit()
        elif key in {"return", "kp_enter"} and self.wires.editing:
            self._confirm_wire_edit()
        elif self.wires.editing and not ctrl:
            char = getattr(event, "char", "")
            if key == "backspace":
                self._wire_numeric = self._wire_numeric[:-1]
                self._preview_wire_edit()
            elif char and char in "0123456789.-+":
                self._wire_numeric += char
                self._preview_wire_edit()
        elif key == "delete":
            self._delete_wire()
        return "break"

    def _draw_object_boxes(self, draw, width, height, view) -> None:
        for index, box in enumerate(self.objects.boxes):
            sx, sy, _ = view.project(box_corners(box), width, height)
            color = _SELECTED if self._boxing and index == self.objects.selected else "#70e5ff"
            for first, second in _BOX_EDGES:
                draw.line((sx[first], sy[first], sx[second], sy[second]), fill=color, width=2)
            cx, cy, _ = view.project([box["center"]], width, height)
            x, y = float(cx[0]), float(cy[0])
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white")
            draw.text((x + 8, y - 14), f"Box {index + 1}", fill=color)

    def _refresh_object_labels(self) -> None:
        self.object_mode_var.set(self._boxing)
        self.object_mode_button.configure(
            state="disabled" if self._asset_loading or self.controller.moving
            or self._measurement_move_start is not None else "normal"
        )
        index = self.objects.selected
        if index is None:
            text = f"長方體數：{len(self.objects.boxes)}；未選取\n點框線／中心選取；新增時左鍵拖曳"
        else:
            box = self.objects.boxes[index]
            x, y, z = box["center"]
            length, width, height = box["size"]
            text = (
                f"長方體 {index + 1}／{len(self.objects.boxes)}\n"
                f"中心 X {x:.4f}　Y {y:.4f}　Z {z:.4f}\n"
                f"長 X {length:.4f}u\n寬 Y {width:.4f}u\n高 Z {height:.4f}u"
            )
        self.object_info_var.set(text)
        self._refresh_look_at_controls()

    def _refresh_look_at_controls(self) -> None:
        count = len(self.controller.points)
        self.look_at_start_box.configure(values=[str(index) for index in range(1, count)])
        self.look_at_end_box.configure(values=[str(index) for index in range(2, count + 1)])
        try:
            start, end = int(self.look_at_start_var.get()), int(self.look_at_end_var.get())
        except ValueError:
            start, end = 1, 2
        if not 1 <= start < count:
            self.look_at_start_var.set("1" if count >= 2 else "")
        if not 2 <= end <= count:
            self.look_at_end_var.set("2" if count >= 2 else "")
        self.look_at_object_box.configure(
            values=[f"長方體 {index + 1}" for index in range(len(self.objects.boxes))],
        )
        if self._look_at_object_key not in self.objects.keys:
            selected = self.objects.selected
            self._look_at_object_key = (
                self.objects.keys[selected] if selected is not None else
                self.objects.keys[0] if self.objects.keys else None
            )
        if self._look_at_object_key is None:
            self.look_at_object_var.set("")
        else:
            self.look_at_object_var.set(f"長方體 {self.objects.keys.index(self._look_at_object_key) + 1}")
        editing = self._asset_loading or self.controller.moving or self.objects.transforming
        editing = editing or self._measurement_move_start is not None or self._box_draw_start is not None
        enabled = count >= 2 and not editing
        self.look_at_apply_button.configure(
            state="normal" if enabled and self.objects.boxes else "disabled",
        )
        self.look_at_clear_button.configure(state="normal" if enabled else "disabled")
        lines = []
        links = self.controller.segment_look_at
        start = 0
        while start < len(links):
            target = links[start]
            end = start + 1
            while end < len(links) and links[end] == target:
                end += 1
            if target is not None and target in self.objects.keys:
                ordinal = self.objects.keys.index(target) + 1
                lines.append(f"航點 {start + 1} → {end + 1}：長方體 {ordinal}")
            start = end
        self.look_at_summary_var.set("\n".join(lines) if lines else "未設定區間，維持原本航向")

    def _select_look_at_object(self, _event=None) -> None:
        index = self.look_at_object_box.current()
        if 0 <= index < len(self.objects.keys):
            self._look_at_object_key = self.objects.keys[index]
            self.objects.selected = index
            self._refresh_object_labels()
            self.request_redraw()

    def _apply_look_at_range(self, *, clear: bool = False) -> None:
        if (
            self._asset_loading or self.controller.moving or self.objects.transforming
            or self._measurement_move_start is not None or self._box_draw_start is not None
        ):
            self.status_var.set("請先確認或取消編輯，並等待地圖載入完成")
            return
        target = None if clear else self._look_at_object_key
        if not clear and target not in self.objects.keys:
            self.status_var.set("請先建立並選擇一個長方體")
            return
        try:
            start = int(self.look_at_start_var.get()) - 1
            end = int(self.look_at_end_var.get()) - 1
            self.controller.set_look_at_range(start, end, target)
        except ValueError as exc:
            self.status_var.set(f"航點區間未套用：{exc}")
            return
        self.status_var.set(
            f"已清除航點 {start + 1} → {end + 1} 的朝向設定" if clear else
            f"航點 {start + 1} → {end + 1} 移動時，持續朝向長方體中心；請儲存路線"
        )
        self._refresh_labels()
        self.request_redraw()
        self.canvas.focus_set()

    def _repair_look_at_references(self) -> None:
        if self.controller.discard_missing_look_at(self.objects.keys):
            self.status_var.set("復原後有物體框已不存在，已清除其航段朝向設定，未改指向其他物體")

    def toggle_object_mode(self) -> None:
        if not self._leave_wire_mode():
            self.object_mode_var.set(self._boxing)
            return
        if self._asset_loading or self.controller.moving or self._measurement_move_start is not None:
            self.object_mode_var.set(self._boxing)
            self.status_var.set("請先確認或取消移動，並等待地圖載入完成")
            return
        if self.objects.transforming or self._box_draw_start is not None:
            self.object_mode_var.set(self._boxing)
            self.status_var.set("請先確認或取消長方體編輯")
            return
        self._cancel_pending_single_click()
        self._boxing = bool(self.object_mode_var.get())
        self._box_create_armed = self._boxing and not self.objects.boxes
        if self._boxing:
            self._measuring = False
            self.controller.set_placement_enabled(False)
        self.status_var.set(
            "物體框選：左鍵拖曳粗框點雲" if self._box_create_armed else
            "物體框選：點框線／中心選取，G 平移、S 縮放" if self._boxing else
            "已離開物體框選；長方體標註仍保留"
        )
        self._refresh_labels()
        self.request_redraw()
        self.canvas.focus_set()

    def arm_object_box(self) -> None:
        if self.objects.transforming or self._box_draw_start is not None:
            self.status_var.set("請先確認或取消長方體編輯")
            return
        self.object_mode_var.set(True)
        self.toggle_object_mode()
        if self._boxing and not self._asset_loading and not self.controller.moving:
            self._box_create_armed = True
            self.status_var.set("左鍵拖曳框住物體點雲；會以框內點雲估計長寬高，再用 G／S 調整")

    def _clear_object_rectangle(self) -> None:
        if self._box_rectangle is not None:
            self.canvas.delete(self._box_rectangle)
        self._box_rectangle = None
        self._box_draw_start = None

    def _create_object_from_rectangle(self, start, end) -> None:
        if not pointer_is_drag(start, end):
            self.status_var.set("請左鍵拖曳出範圍；Esc 取消新增")
            return
        if self._displayed_cloud is not self.map_points or self._displayed_view is None:
            self.status_var.set("請等待目前點雲顯示完成，再框選")
            return
        width, height = self._displayed_size
        sx, sy, _ = self._displayed_view.project(self.map_points[:, :3], width, height)
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        inside = (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
        points = self.map_points[inside, :3]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            self.status_var.set("框內沒有點雲，未建立長方體；請重新框選")
            return
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) * 0.5
        size = np.maximum(hi - lo, self.view.radius * 0.01)
        self.objects.add_bounds(center - size * 0.5, center + size * 0.5)
        self._box_create_armed = False
        self.status_var.set("已建立長方體；G 平移，S 縮放，接 X／Y／Z 調整單軸。框內背景點也會納入")
        self._refresh_object_labels()
        self.request_redraw()

    def _nearest_object_box(self, x, y) -> int | None:
        view = self._displayed_view or self.view
        width, height = self._displayed_size or self._canvas_size()
        pointer = np.array([x, y], dtype=float)
        best, selected = 14.0 ** 2, None
        for index, box in enumerate(self.objects.boxes):
            sx, sy, _ = view.project(box_corners(box), width, height)
            projected = np.column_stack((sx, sy))
            cx, cy, _ = view.project([box["center"]], width, height)
            distance = float((cx[0] - x) ** 2 + (cy[0] - y) ** 2)
            for first, second in _BOX_EDGES:
                a, b = projected[first], projected[second]
                edge = b - a
                denominator = float(edge @ edge)
                fraction = 0.0 if denominator < 1e-9 else float(
                    np.clip((pointer - a) @ edge / denominator, 0.0, 1.0)
                )
                offset = pointer - (a + fraction * edge)
                distance = min(distance, float(offset @ offset))
            if distance <= best:
                best, selected = distance, index
        return selected

    def _begin_object_transform(self, operation: str) -> None:
        if self._box_create_armed or self._box_draw_start is not None:
            self.status_var.set("請先完成框選或按 Esc 取消新增")
            return
        self._cancel_pending_single_click()
        if not self.objects.begin(operation):
            self.status_var.set("請先點框線／中心選取長方體")
            return
        self._move_mouse_start = self._mouse
        if operation == "s":
            width, height = self._canvas_size()
            box = self.objects.boxes[self.objects.selected]
            sx, sy, _ = self.view.project([box["center"]], width, height)
            self._scale_gesture = PointerScale((sx[0], sy[0]), self._mouse)
        self._box_numeric = ""
        self._object_transform_status()

    def _object_transform_status(self) -> None:
        action = "平移" if self.objects.operation == "g" else "縮放"
        axis = "自由" if self.objects.axis is None else "XYZ"[self.objects.axis] + " 軸"
        number = f"；輸入 {self._box_numeric}" if self._box_numeric else ""
        fine = "；按住 Shift 細調" if self.objects.operation == "s" else ""
        self.status_var.set(
            f"長方體{action}：{axis}{number}{fine}；左鍵／Enter 確認，Esc／右鍵取消"
        )

    def _preview_object_transform(self) -> bool:
        width, height = self._canvas_size()
        try:
            if self._box_numeric:
                value = float(self._box_numeric)
                if not math.isfinite(value):
                    raise ValueError("數值必須有限")
                if self.objects.operation == "g":
                    if self.objects.axis is None:
                        raise ValueError("數字位移請先按 X／Y／Z 選軸")
                    delta = np.zeros(3)
                    delta[self.objects.axis] = value
                    self.objects.preview(delta=delta)
                else:
                    self.objects.preview(factor=value)
            elif self.objects.operation == "g":
                dx = self._mouse[0] - self._move_mouse_start[0]
                dy = self._mouse[1] - self._move_mouse_start[1]
                delta = self.view.screen_delta_to_world(dx, dy, width, height, self.objects.axis)
                self.objects.preview(delta=delta)
            else:
                factor = self._scale_gesture.move(self._mouse, self._mouse_precise)
                self.objects.preview(factor=factor)
        except (ValueError, OverflowError) as exc:
            self.status_var.set(f"長方體數值未套用：{exc}；Backspace 修改，Esc 取消")
            return False
        self._refresh_object_labels()
        self.request_redraw()
        self._object_transform_status()
        return True

    def _finish_object_transform(self) -> None:
        if self._box_numeric and not self._preview_object_transform():
            return
        self.objects.confirm()
        self._box_numeric = ""
        self.status_var.set("已確認長方體變換；儲存路線會一併保留標註")
        self._refresh_object_labels()
        self.request_redraw()

    def _delete_object_box(self) -> None:
        if not self._boxing:
            self.status_var.set("請先開啟物體框選模式並選取長方體")
            return
        if self.objects.transforming or self._box_draw_start is not None:
            self.status_var.set("請先確認或取消長方體編輯")
            return
        selected = self.objects.selected
        if selected is not None and self.objects.keys[selected] in self.controller.segment_look_at:
            self.status_var.set("這個長方體正被航段朝向使用；請先清除相關區間設定再刪除")
            return
        if self.objects.delete_selected():
            self.status_var.set("已刪除選取長方體；航點未變更")
            self._refresh_object_labels()
            self.request_redraw()
        self.canvas.focus_set()

    def _object_escape(self) -> None:
        if self.objects.transforming:
            self.objects.cancel()
            self._box_numeric = ""
            self.status_var.set("已還原長方體；仍在物體框選模式")
        elif self._box_create_armed:
            dragging = self._box_draw_start is not None
            self._clear_object_rectangle()
            self._box_create_armed = False
            self._suppress_left_release = dragging
            self.status_var.set("已取消新增；可點選既有長方體")
        else:
            self.object_mode_var.set(False)
            self.toggle_object_mode()
        self._refresh_object_labels()
        self.request_redraw()

    def _object_undo_redo(self, shift: bool) -> None:
        if self._box_draw_start is not None:
            self._clear_object_rectangle()
            self._suppress_left_release = True
        changed = self.objects.redo() if shift else self.objects.undo()
        self._box_numeric = ""
        if changed:
            self.status_var.set("已重做長方體編輯" if shift else "已復原長方體編輯")
            self._repair_look_at_references()
            self._refresh_object_labels()
            self.request_redraw()

    def _object_copy_paste(self, *, paste: bool) -> None:
        if self.objects.transforming or self._box_draw_start is not None:
            self.status_var.set("請先確認或取消長方體編輯")
        elif not paste:
            copied = self.objects.copy_selected()
            self.status_var.set(
                "已複製長方體；Ctrl+V 貼上" if copied else "請先點框線／中心選取長方體再複製"
            )
        elif self.objects.paste():
            self._box_create_armed = False
            self.status_var.set("已貼上相同長方體並選取；G 平移可移開，Ctrl+Z 復原")
            self._refresh_object_labels()
            self.request_redraw()
        else:
            self.status_var.set("尚未複製長方體；選取後按 Ctrl+C")

    def _object_numeric_key(self, event, key: str) -> None:
        char = getattr(event, "char", "")
        if key == "backspace":
            self._box_numeric = self._box_numeric[:-1]
            self._preview_object_transform()
        elif char and char in "0123456789.-+":
            self._box_numeric += char
            self._preview_object_transform()

    def _on_object_key(self, event, key, ctrl, shift) -> str:
        if key == "escape":
            self._object_escape()
        elif ctrl and key == "z":
            self._object_undo_redo(shift)
        elif ctrl and key in {"c", "v"}:
            self._object_copy_paste(paste=key == "v")
        elif not ctrl and key in {"g", "s"} and not self.objects.transforming:
            self._begin_object_transform(key)
        elif key in {"x", "y", "z"} and self.objects.transforming:
            self.objects.constrain({"x": 0, "y": 1, "z": 2}[key])
            self._preview_object_transform()
        elif key in {"return", "kp_enter"} and self.objects.transforming:
            self._finish_object_transform()
        elif key == "delete":
            self._delete_object_box()
        elif self.objects.transforming and not ctrl:
            self._object_numeric_key(event, key)
        return "break"

    def _draw_measurement(
        self, draw: ImageDraw.ImageDraw, width: int, height: int, view: OrthoView
    ) -> None:
        if not self.measurement_points:
            return
        sx, sy, _ = view.project(self.measurement_points, width, height)
        projected = list(zip(sx.tolist(), sy.tolist()))
        if len(projected) == 2:
            draw.line(projected, fill="#70e5ff", width=3)
        for index, (label, (x, y)) in enumerate(zip(("A", "B"), projected)):
            fill = _SELECTED if index == self._measurement_selected else "#70e5ff"
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=fill, outline="white")
            draw.text((x + 10, y - 12), label, fill=fill)

    def _format_distance(self, units: float, meters: float | None) -> str:
        if meters is None:
            return f"{units:.4f} u｜公尺不可用"
        prefix = "估算 " if self.distance_scale.estimated else ""
        return f"{units:.4f} u｜{prefix}{meters:.3f} m"

    def _refresh_distance_labels(self) -> None:
        self._distance_labels_dirty = False
        scale = self.distance_scale
        if scale is None:
            self.distance_scale_var.set("尺度未取得；僅顯示地圖單位 u，公尺不可用")
        else:
            kind = "估算（非實測標定）" if scale.estimated else "標定"
            details = f"尺度 {scale.m_per_unit:.6g} m/u｜{kind}\n來源：{scale.source}"
            if scale.uncertainty_relative is not None:
                details += f"\n尺度分散參考：±{scale.uncertainty_relative:.1%}（非精度保證）"
            self.distance_scale_var.set(details)
        self.measurement_distance_u = None
        self.measurement_distance_m = None
        if len(self.measurement_points) == 2:
            first, second = self.measurement_points
            self.measurement_distance_u = math.dist(first, second)
            if scale is not None:
                self.measurement_distance_m = scale.distance_m(first, second)
            self.measurement_var.set(
                "A → B：" + self._format_distance(
                    self.measurement_distance_u, self.measurement_distance_m
                )
            )
        else:
            self.measurement_var.set(
                "已選 A，請單擊選 B" if self.measurement_points
                else "開啟量距後單擊選 A、B；第三點開始新一組"
            )
        points = self.controller.points
        self.route_segment_distances_u = [
            math.dist(first, second) for first, second in zip(points, points[1:])
        ]
        self.route_segment_distances_m = (
            None if scale is None else scale.route_segment_distances(points)
        )
        self.route_total_u = sum(self.route_segment_distances_u)
        self.route_total_m = (
            None if self.route_segment_distances_m is None
            else sum(self.route_segment_distances_m)
        )
        self.route_total_var.set(
            "總長：" + self._format_distance(self.route_total_u, self.route_total_m)
        )
        rows = [
            f"{index + 1} → {index + 2}：" + self._format_distance(
                units,
                None if self.route_segment_distances_m is None
                else self.route_segment_distances_m[index],
            )
            for index, units in enumerate(self.route_segment_distances_u)
        ]
        self._update_distance_list(rows)

    def _replace_distance_list_rows(self, previous, rows) -> None:
        """Change only differing rows, retaining unchanged widget entries."""
        for index in range(min(len(previous), len(rows))):
            if previous[index] != rows[index]:
                self.route_distance_list.delete(index)
                self.route_distance_list.insert(index, rows[index])
        if len(previous) > len(rows):
            self.route_distance_list.delete(len(rows), "end")
        for row in rows[len(previous):]:
            self.route_distance_list.insert("end", row)

    def _update_distance_list(self, rows) -> None:
        """Retain scroll, selection and keyboard focus across row edits."""
        previous = self.route_distance_list.get(0, "end")
        if tuple(rows) == tuple(previous):
            return
        scroll = self.route_distance_list.yview()[0]
        horizontal = self.route_distance_list.xview()[0]
        selected = self.route_distance_list.curselection()
        active = self.route_distance_list.index("active")
        anchor = self.route_distance_list.index("anchor")
        self._replace_distance_list_rows(previous, rows)
        self.route_distance_list.selection_clear(0, "end")
        for index in selected:
            if index < len(rows):
                self.route_distance_list.selection_set(index)
        if rows:
            self.route_distance_list.activate(min(active, len(rows) - 1))
            self.route_distance_list.selection_anchor(min(anchor, len(rows) - 1))
        self.route_distance_list.yview_moveto(scroll)
        self.route_distance_list.xview_moveto(horizontal)

    def toggle_measurement_mode(self) -> None:
        if not self._leave_wire_mode():
            self.measurement_mode_var.set(self._measuring)
            return
        if self.objects.transforming or self._box_draw_start is not None:
            self.measurement_mode_var.set(self._measuring)
            self.status_var.set("請先確認或取消長方體編輯")
            return
        if self.controller.moving or self._measurement_move_start is not None:
            self.measurement_mode_var.set(self._measuring)
            self.status_var.set("請先確認或取消球體移動，再切換量距模式")
            return
        self._cancel_pending_single_click()
        self._measuring = bool(self.measurement_mode_var.get())
        if self._measuring:
            self._boxing = False
            self._box_create_armed = False
            self.controller.set_placement_enabled(False)
        self.status_var.set(
            "兩點量距：單擊選 A／B，點測量球後 G、X／Y／Z 移動；Esc 取消／離開"
            if self._measuring else "已離開兩點量距模式"
        )
        self._refresh_labels()
        self.canvas.focus_set()

    def clear_measurement(self) -> None:
        self._cancel_pending_single_click()
        self.measurement_points.clear()
        self._measurement_selected = None
        self._measurement_move_start = None
        self._measurement_move_axis = None
        self._refresh_distance_labels()
        self.request_redraw()

    def _measure_at(self, x: int, y: int) -> None:
        if self.measurement_points:
            view = self.__dict__.get("_displayed_view") or self.view
            width, height = self.__dict__.get("_displayed_size") or self._canvas_size()
            sx, sy, _ = view.project(self.measurement_points, width, height)
            distance = (sx - x) ** 2 + (sy - y) ** 2
            index = int(np.argmin(distance))
            if float(distance[index]) <= 14.0 ** 2:
                self._measurement_selected = index
                self.status_var.set(f"已選測量球 {'AB'[index]}；按 G，再用 X／Y／Z 鎖軸移動")
                self.request_redraw()
                return
        selected = self._nearest_waypoint(x, y)
        point = (
            self.controller.points[selected] if selected is not None
            else self._nearest_cloud(x, y)
        )
        if point is None:
            self.status_var.set("游標附近沒有點雲或航點，測量點未變更")
            return
        if len(self.measurement_points) == 2:
            self.measurement_points.clear()
        self.measurement_points.append([float(value) for value in point])
        self._measurement_selected = len(self.measurement_points) - 1
        self.status_var.set(
            "已選 A，請單擊選 B" if len(self.measurement_points) == 1
            else "已完成 A → B 量距；點測量球可用 G 移動，點其他位置開始新一組"
        )
        self._refresh_distance_labels()
        self.request_redraw()


    def _canvas_size(self) -> tuple[int, int]:
        return max(320, self.canvas.winfo_width()), max(240, self.canvas.winfo_height())

    def _nearest_cloud(self, x: float, y: float, threshold: float = 20.0):
        if not len(self.map_points):
            return None
        if "_displayed_cloud" in self.__dict__ and self._displayed_cloud is not self.map_points:
            return None
        view = self.__dict__.get("_displayed_view") or self.view
        width, height = self.__dict__.get("_displayed_size") or self._canvas_size()
        worker = self.__dict__.get("_cloud_worker")
        executor = self.__dict__.get("_render_executor")
        if worker is not None and executor is not None:
            # The render worker already holds the cloud (on the GPU when CUDA is
            # available); projecting millions of points on the Tk thread per click
            # stalled the editor for ~100 ms.
            index = executor.submit(
                worker.nearest, self.map_points, self._cloud_revision,
                view, width, height, x, y, threshold,
            ).result()
        else:
            index = nearest_point_index(
                self.map_points[:, :3], view, width, height, x, y, threshold
            )
        if index is None:
            return None
        return self.map_points[index, :3].astype(float)

    def _nearest_waypoint(self, x: float, y: float, threshold: float = 28.0):
        if not self.controller.points:
            return None
        view = self.__dict__.get("_displayed_view") or self.view
        width, height = self.__dict__.get("_displayed_size") or self._canvas_size()
        sx, sy, _ = view.project(self.controller.points, width, height)
        distance = (sx - x) ** 2 + (sy - y) ** 2
        index = int(np.argmin(distance))
        return index if float(distance[index]) <= threshold * threshold else None

    def on_left_press(self, event) -> str:
        self.canvas.focus_set()
        self._mouse = (int(event.x), int(event.y))
        if self._wiring and self.wires.editing:
            self._confirm_wire_edit()
            self._left_confirmed_move = True
            return "break"
        if self._boxing:
            self._cancel_pending_single_click()
            if self.objects.transforming:
                self._finish_object_transform()
                self._left_confirmed_move = True
                return "break"
            if self._box_create_armed:
                self._box_draw_start = self._mouse
                self._box_rectangle = self.canvas.create_rectangle(
                    *self._mouse, *self._mouse, outline="#70e5ff", width=2, dash=(6, 3),
                )
                return "break"
        if self._measuring and self._measurement_move_start is not None:
            self._handle_enter()
            self._left_confirmed_move = True
            return "break"
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
        if self._wiring and self.wires.editing:
            self.on_motion(event)
            return "break"
        if self._box_draw_start is not None:
            self.canvas.coords(self._box_rectangle, *self._box_draw_start, event.x, event.y)
            return "break"
        if self.objects.transforming:
            self.on_motion(event)
            return "break"
        if (
            self.controller.moving
            or self._measurement_move_start is not None
            or self._left_press is None
            or self._left_last is None
        ):
            return "break"
        x, y = int(event.x), int(event.y)
        if not self._left_dragging and not pointer_is_drag(self._left_press, (x, y)):
            return "break"
        self._left_dragging = True
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
        if self._box_draw_start is not None:
            start = self._box_draw_start
            self._clear_object_rectangle()
            self._create_object_from_rectangle(start, (int(event.x), int(event.y)))
            return "break"
        dragged = self._left_dragging
        self._left_press = None
        self._left_last = None
        self._left_dragging = False
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
        if self._wiring and self.wires.editing:
            self._cancel_pending_single_click()
            self._suppress_left_release = True
            return "break"
        self._cancel_pending_single_click()
        # The first click of the pair may have started a rough box; drop it.
        self._clear_object_rectangle()
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
        if self._asset_loading:
            return
        if self._wiring:
            self._wire_at(x, y)
            return
        if self._boxing:
            self.objects.selected = self._nearest_object_box(x, y)
            self.status_var.set(
                "已選長方體；G 平移、S 縮放，再按 X／Y／Z 鎖軸"
                if self.objects.selected is not None else "請點長方體框線／中心，或按「新增長方體」"
            )
            self._refresh_object_labels()
            self.request_redraw()
            return
        if self._measuring:
            self._measure_at(x, y)
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
            if self.document.waypoint_arrive_radii is not None:
                self.document.waypoint_arrive_radii.append(
                    float(self.arrive_radius_var.get())
                )
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
        self._mouse_precise = bool(int(getattr(event, "state", 0)) & 0x0001)
        if self._wiring and self.wires.editing:
            self._preview_wire_edit()
            return
        if self.objects.transforming:
            self._preview_object_transform()
            return
        if not self.controller.moving and self._measurement_move_start is None:
            return
        self._update_move_preview()

    def toggle_marking_mode(self) -> None:
        if not self._leave_wire_mode():
            self.marking_mode_var.set(self.controller.placement_enabled)
            return
        if self.objects.transforming or self._box_draw_start is not None:
            self.marking_mode_var.set(self.controller.placement_enabled)
            self.status_var.set("請先確認或取消長方體編輯")
            return
        self._cancel_pending_single_click()
        if self.marking_mode_var.get() and self.controller.phase == "layout":
            self._boxing = False
            self._box_create_armed = False
            self._measuring = False
            self.clear_measurement()
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
        axis = self._measurement_move_axis if self._measuring else self.controller.move_axis
        delta = self.view.screen_delta_to_world(dx, dy, width, height, axis)
        if self._measuring:
            if self._measurement_move_start is None or self._measurement_selected is None:
                return
            self.measurement_points[self._measurement_selected] = (
                np.asarray(self._measurement_move_start) + delta
            ).tolist()
        else:
            self.controller.preview_move(delta)
        self._refresh_labels(refresh_distances=False)
        self.request_redraw()

    def on_navigation_press(self, event) -> str:
        self.canvas.focus_set()
        if self._wiring and (self.wires.editing or self._wire_create_armed):
            self._wire_escape()
            return "break"
        if self._boxing and self.objects.transforming:
            self.objects.cancel()
            self._box_numeric = ""
            self.status_var.set("已取消長方體變換")
            self._refresh_object_labels()
            self.request_redraw()
            return "break"
        if self._box_draw_start is not None:
            self._clear_object_rectangle()
            self.status_var.set("已取消拖曳框選；可重新拖曳")
            return "break"
        if self.controller.moving or self._measurement_move_start is not None:
            self.status_var.set("請先用左鍵／Enter 確認移動，或用 Esc 取消")
            return "break"
        self._nav_button = int(event.num)
        self._nav_last = (int(event.x), int(event.y))
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
        width, height = self._canvas_size()
        before = self.view.scale(width, height)
        self.view.zoom = float(np.clip(zoom, 0.08, 120.0))
        if self.objects.transforming and self.objects.operation == "g":
            # Pointer offset maps to world distance through the zoom; keep the box
            # where it is instead of letting the next mouse move jump it.
            ratio = self.view.scale(width, height) / before
            start_x, start_y = self._move_mouse_start
            self._move_mouse_start = (
                self._mouse[0] - (self._mouse[0] - start_x) * ratio,
                self._mouse[1] - (self._mouse[1] - start_y) * ratio,
            )
        self.request_redraw()


    def _handle_undo_redo(self, *, shift: bool) -> None:
        changed = self.controller.redo() if shift else self.controller.undo()
        if changed:
            self.status_var.set("已重做" if shift else "已復原")
            self._repair_look_at_references()
            self._refresh_labels()
            self.request_redraw()

    def _handle_begin_move(self) -> None:
        if self._measuring:
            if self._measurement_selected is None:
                self.status_var.set("請先單擊選取測量球 A 或 B，再按 G")
                return
            self._cancel_pending_single_click()
            self._measurement_move_start = self.measurement_points[self._measurement_selected].copy()
            self._measurement_move_axis = None
            self._move_mouse_start = self._mouse
            self.status_var.set("測量球平移中：X／Y／Z 鎖軸；左鍵或 Enter 確認，Esc 取消")
            return
        if self.controller.begin_move():
            self._move_mouse_start = self._mouse
            self.status_var.set(
                "平移中：按 X / Y / Z 鎖定軸；左鍵或 Enter 確認，Esc 取消"
            )
            self._refresh_labels()

    def _handle_move_axis(self, key: str) -> None:
        axis = {"x": 0, "y": 1, "z": 2}[key]
        if self._measuring:
            self._measurement_move_axis = axis
        else:
            self.controller.constrain(axis)
        self._update_move_preview()
        self.status_var.set(f"平移中：已鎖定 {key.upper()} 軸")

    def _handle_enter(self) -> None:
        if self._measuring:
            self._measurement_move_start = None
            self._measurement_move_axis = None
            self.status_var.set("已確認測量球移動；航點未變更")
            self._refresh_distance_labels()
            self.request_redraw()
            return
        if self.controller.moving:
            self.controller.confirm_move()
            self.status_var.set("已確認航點移動")
        elif self.controller.phase == "layout":
            self.finish_layout()
        self._refresh_labels()
        self.request_redraw()

    def _handle_escape(self) -> None:
        if self._measuring:
            if self._measurement_move_start is not None:
                self.measurement_points[self._measurement_selected] = self._measurement_move_start
                self._measurement_move_start = None
                self._measurement_move_axis = None
                self.status_var.set("已取消測量球移動；仍在兩點量距模式")
                self._refresh_distance_labels()
                self.request_redraw()
                return
            self._measuring = False
            self.clear_measurement()
            self.status_var.set("已清除測量並離開量距模式")
            self._refresh_labels()
            return
        if self.controller.cancel_move():
            self.status_var.set("已取消這次移動；請使用右上角按鈕離開編輯器")
            self._refresh_labels()
            self.request_redraw()
        else:
            self.status_var.set("Esc 只取消移動；請使用右上角「離開編輯器」")

    def _handle_delete(self) -> None:
        selected = self.controller.selected
        if self.controller.delete_selected():
            radii = self.document.waypoint_arrive_radii
            if radii is not None and selected is not None and 0 <= selected < len(radii):
                del radii[selected]
            self.status_var.set("已刪除選取航點")
            self._refresh_labels()
            self.request_redraw()

    def on_key(self, event) -> str | None:
        key = str(event.keysym).lower()
        ctrl = bool(int(event.state) & 0x0004)
        shift = bool(int(event.state) & 0x0001)
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)):
            return None
        if event.widget is not self.canvas and key != "escape":
            return None
        if self._asset_loading:
            return "break"
        if self._wiring:
            return self._on_wire_key(event, key, ctrl, shift)
        if self._boxing:
            return self._on_object_key(event, key, ctrl, shift)
        if self._measuring:
            _dispatch_measure_key(self, key)
            return "break"
        return _dispatch_route_key(self, key, ctrl, shift)

    def _set_all_arrive_radii(self, radius: float) -> None:
        """One radius for the whole route: global value and every per-point sphere."""
        radius = min(ARRIVE_RADIUS_MAX_U, max(ARRIVE_RADIUS_MIN_U, float(radius)))
        self.arrive_radius_var.set(radius)
        if self.document.waypoint_arrive_radii is not None:
            self.document.waypoint_arrive_radii = [radius] * len(self.controller.points)

    def _on_arrive_radius(self, _value) -> None:
        if _value is not None:
            self.show_arrive_var.set(True)
            self._set_all_arrive_radii(float(self.arrive_radius_var.get()))
        radius = float(self.arrive_radius_var.get())
        shortest = self.controller.shortest_leg()
        note = ""
        if self.controller.spheres_overlap(radius):
            # Overlapping spheres mean the drone retires waypoints without ever
            # translating between them: it settles, advances, settles, advances.
            note = f"　⚠ 半徑 >= 最短段 {shortest:.3f}u 的一半，相鄰球體會重疊"
        self.arrive_label_var.set(f"半徑 {radius:.3f} 地圖單位{note}")
        if self.show_arrive_var.get():
            self.request_redraw()

    def _point_radius_for(self, index: int | None) -> float:
        """Arrival sphere for one waypoint: its drawn value or the global."""
        radii = self.document.waypoint_arrive_radii
        if (
            radii is not None
            and index is not None
            and 0 <= index < len(radii)
            and index < len(self.controller.points)
        ):
            return float(radii[index])
        return float(self.arrive_radius_var.get())

    def _sync_point_radius_box(self) -> None:
        selected = self.controller.selected
        self.point_radius_var.set(f"{self._point_radius_for(selected):.3f}")
        self._update_point_radius_note()

    def _update_point_radius_note(self) -> None:
        selected = self.controller.selected
        points = self.controller.points
        if selected is None or not (0 <= selected < len(points)):
            self.point_radius_note_var.set("")
            return
        radius = self._point_radius_for(selected)
        neighbours = []
        if selected > 0:
            neighbours.append(points[selected - 1])
        if selected + 1 < len(points):
            neighbours.append(points[selected + 1])
        import math as _math
        nearest = min((_math.dist(points[selected], other) for other in neighbours), default=None)
        note = ""
        if nearest is not None and radius >= nearest * 0.5:
            note = f"　⚠ 半徑 >= 相鄰段 {nearest:.3f}u 的一半，相鄰球體會重疊"
        custom = self.document.waypoint_arrive_radii is not None and (
            0 <= selected < len(self.document.waypoint_arrive_radii or [])
        )
        self.point_radius_note_var.set(
            f"航點 {selected + 1} 到達球 {radius:.3f}u{note}"
            + ("" if custom else "（全域值）")
        )

    def _on_point_radius(self, _event=None) -> None:
        try:
            radius = float(str(self.point_radius_var.get()).strip())
        except (TypeError, ValueError):
            radius = float("nan")
        import math as _math
        if not _math.isfinite(radius):
            self._sync_point_radius_box()
            return
        self._set_all_arrive_radii(radius)
        self.show_arrive_var.set(True)
        self._on_arrive_radius(None)
        self._sync_point_radius_box()

    def _sync_action_list(self) -> None:
        selected = self.controller.selected
        if selected is None or not (0 <= selected < len(self.controller.points)):
            self.action_list_var.set("先選取一個航點")
            return
        current = self.action_direction_var.get()
        sections = []
        for title, direction, entries in (
            ("去程", "out", self.controller.actions[selected]),
            ("回程", "back", self.controller.return_actions[selected]),
        ):
            lines = [f"{step}. {waypoint_action_text(action)}"
                     for step, action in enumerate(entries, start=1)]
            marker = "▶ " if direction == current else "　"
            sections.append(
                f"{marker}{title}：" + ("\n" + "\n".join(lines) if lines else "無動作")
            )
        note = ""
        if not self.controller.return_actions_allowed(selected):
            note = "\n（第一點是降落點、最後一點是只飛一次的折返點，沒有回程動作）"
        self.action_list_var.set(f"航點 {selected + 1}\n" + "\n".join(sections) + note)

    def _add_action(self, raw: dict) -> None:
        selected = self.controller.selected
        if selected is None or not (0 <= selected < len(self.controller.points)):
            self.status_var.set("先選取一個航點再加入動作")
            return
        direction = self.action_direction_var.get()
        trip = "回程" if direction == "back" else "去程"
        if direction == "back" and not self.controller.return_actions_allowed(selected):
            self.status_var.set("第一點是降落點、最後一點是只飛一次的折返點，不能設定回程動作")
            return
        table = (
            self.controller.return_actions if direction == "back" else self.controller.actions
        )
        if len(table[selected]) >= MAX_ACTIONS_PER_WAYPOINT:
            self.status_var.set(f"每個航點{trip}最多 {MAX_ACTIONS_PER_WAYPOINT} 個動作")
            return
        try:
            action = parse_waypoint_action(raw, "動作").to_json()
        except ValueError as exc:
            self.status_var.set(f"動作數值不合法：{exc}")
            return
        self.controller.add_action(action, direction)
        self.status_var.set(
            f"航點 {selected + 1} 已加入{trip}動作：{waypoint_action_text(action)}"
        )
        self._refresh_labels()
        self.request_redraw()

    def _add_yaw_action(self, direction: int) -> None:
        try:
            degrees = float(str(self.action_degrees_var.get()).strip())
        except ValueError:
            self.status_var.set("轉向角度必須是數字")
            return
        self._add_action({"type": "yaw", "degrees": direction * degrees})

    def _add_hover_action(self) -> None:
        try:
            seconds = float(str(self.action_seconds_var.get()).strip())
        except ValueError:
            self.status_var.set("懸停秒數必須是數字")
            return
        self._add_action({"type": "hover", "seconds": seconds})

    def _add_gimbal_pitch_action(self) -> None:
        try:
            pitch = float(str(self.action_pitch_var.get()).strip())
        except ValueError:
            self.status_var.set("鏡頭俯仰角度必須是數字（-90 至 +90）")
            return
        self._add_action({"type": "gimbal_pitch", "pitch_deg": pitch})

    def _remove_last_action(self) -> None:
        direction = self.action_direction_var.get()
        trip = "回程" if direction == "back" else "去程"
        if self.controller.remove_last_action(direction):
            self.status_var.set(f"已刪除選取航點{trip}的最後一個動作")
            self._refresh_labels()
            self.request_redraw()
        else:
            self.status_var.set(f"選取航點沒有{trip}動作可刪除")

    def finish_layout(self) -> None:
        try:
            self.controller.finish_layout()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        # Stage two is where the arrival spheres matter, so show them by default:
        # the operator confirms the radius against the real geometry before export.
        self.show_arrive_var.set(True)
        self._on_arrive_radius(None)
        self.status_var.set(
            "第二階段：綠圈為到達球體，確認到達半徑後再匯出；"
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

    def set_left_view(self) -> None:
        self.view.left()
        self.status_var.set("左視：適合逐點調整 Z 高度")
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
        if self.wires.editing or self.wires.pending is not None:
            self.status_var.set("請先確認或取消電線編輯，再更換地圖")
            return
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
        dialog.configure(bg=PANEL)
        dialog.geometry("760x430")
        ttk.Label(
            dialog,
            text=f"在 {folder.name} 偵測到 {len(files)} 個 PLY，請選擇地圖：",
            wraplength=720,
        ).pack(anchor="w", padx=12, pady=(12, 6))
        listbox = tk.Listbox(
            dialog,
            **TEXT_BOX,
            selectbackground=SELECT_BG,
            selectforeground=SELECT_FG,
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
        self._map_source = source
        self._cancel_pending_single_click()
        self._measuring = False
        self.measurement_points.clear()
        self._measurement_selected = None
        self._measurement_move_start = None
        self._measurement_move_axis = None
        self.distance_scale = self.distance_scale_loader(self.profile)
        self.title(f"航線編輯器 — {display_name}")
        self.title_var.set(f"航線編輯器｜{display_name}")
        if self.map_loaded_callback is not None:
            self.map_loaded_callback(message)
        self.controller = RouteEditorController()
        self.objects = ObjectBoxController()
        self.wires = WireController()
        self._wiring = False
        self._wire_create_armed = False
        self._wire_numeric = ""
        self._look_at_object_key = None
        self._boxing = False
        self._box_create_armed = False
        self._box_numeric = ""
        self._clear_object_rectangle()
        self.map_points = self._aligned_map_points(points)
        self._cloud_revision += 1
        self._raster_cache_key = None
        self._raster_cache = None
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
        self.objects.confirm()
        self.document.object_boxes = self.objects.boxes
        self.document.wires = self.wires.wires
        self.document.points = self.controller.points
        index_for_key = {key: index for index, key in enumerate(self.objects.keys)}
        self.document.segment_look_at = [
            None if key is None else index_for_key[key]
            for key in self.controller.segment_look_at
        ]
        actions = getattr(self.controller, "actions", None)
        if actions is not None:
            self.document.waypoint_actions = (
                [list(entry) for entry in actions] if any(actions) else None
            )
        return_actions = getattr(self.controller, "return_actions", None)
        if return_actions is not None:
            self.document.return_waypoint_actions = (
                [list(entry) for entry in return_actions] if any(return_actions) else None
            )
        self.document.arrive_radius_map_units = float(self.arrive_radius_var.get())
        radii = self.document.waypoint_arrive_radii
        if radii is not None:
            # Backstop for undo/redo, which restores points without touching
            # radii: truncate extras, pad additions with the global sphere.
            global_radius = float(self.arrive_radius_var.get())
            self.document.waypoint_arrive_radii = (
                list(radii[: len(self.controller.points)])
                + [global_radius]
                * max(0, len(self.controller.points) - len(radii))
            )

    def save_route(self) -> Path | None:
        if self.wires.editing or self.wires.pending is not None:
            self.status_var.set("請先用左鍵／Enter 確認端點／垂度，或 Esc 取消未完成電線，再儲存")
            return None
        if self.objects.transforming:
            self._finish_object_transform()
            if self.objects.transforming:
                return None
        self._update_document()
        self._refresh_distance_labels()
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
                payload = json.loads(staged.read_text(encoding="utf-8"))
                map_digest = self.profile.asset_sha256.map_ply
                if not isinstance(map_digest, str) or len(map_digest) != 64:
                    raise ValueError("已綁定航線儲存需要地圖 SHA-256")
                payload["map_ply_sha256"] = map_digest
                staged.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                result = self.import_route(staged)
                asset_path = getattr(result, "asset_path", None)
                if asset_path is None or not bool(getattr(result, "approved_for_auto", False)):
                    raise ValueError("航線儲存未取得可驗證的 AUTO 核准")
                saved = Path(asset_path)
        except Exception as exc:
            hint = _save_failure_hint(exc)
            self.status_var.set(f"航線儲存失敗：{hint or exc}")
            return None
        saved = saved.resolve()
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
        if self._render_after is not None:
            self.after_cancel(self._render_after)
            self._render_after = None
        self._render_executor.shutdown(wait=False, cancel_futures=True)
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
