"""新增地圖: browse to a mapping-output folder, preview it, then hand it to the importer.

View only. What a folder contains is decided by local_site_assets.preview_site_folder
and, authoritatively, by the importer the caller runs; this window never copies,
hashes or activates anything itself.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import tkinter as tk
from tkinter import ttk

from local_site_assets import SiteFolderPreview, preview_site_folder
from operator_ui_style import (
    FONT_BOLD, FONT_SMALL_BOLD, HEADING, ON_STATE, STATE_BLOCKED, STATE_GOOD,
    STATE_WAITING, TEXT, TEXT_BLOCKED, TEXT_GOOD, apply_operator_styles,
)

SITE_MANIFEST = "site_profile.json"
_PREVIEW_WRAP = 380


def format_size(size: int | None) -> str:
    if size is None:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024.0:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def list_child_folders(folder: Path) -> list[tuple[Path, bool]]:
    """Visible sub-folders, site folders first, each flagged when it holds a manifest."""
    entries = []
    for child in folder.iterdir():
        if child.name.startswith("."):
            continue
        try:
            if not child.is_dir():
                continue
            is_site = (child / SITE_MANIFEST).is_file()
        except OSError:
            continue
        entries.append((child, is_site))
    return sorted(entries, key=lambda item: (not item[1], item[0].name.casefold()))


class AddMapDialog(tk.Toplevel):
    """Non-modal on purpose: a grab would make 原地降落／停止電腦動作 unclickable."""

    def __init__(
        self,
        parent,
        *,
        start_dir: Path,
        shortcuts: Sequence[tuple[str, Path]],
        on_import: Callable[[Path], None],
        listed_root: Path | None = None,
    ):
        super().__init__(parent)
        self.on_import = on_import
        #: Folders directly under it are already offered by the 「① 地圖」 menu.
        self.listed_root = None if listed_root is None else Path(listed_root).resolve()
        self.title("新增地圖")
        apply_operator_styles(self)
        self.transient(parent)
        self.minsize(880, 560)
        self.geometry("1020x660")
        self.current_dir: Path | None = None
        self.preview: SiteFolderPreview | None = None
        self._entries: list[tuple[Path, bool]] = []
        self.path_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        self.error_var = tk.StringVar()
        self._build(shortcuts)
        self.bind("<Escape>", lambda _event: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.navigate(start_dir)
        self.tree.focus_set()

    # ---- layout --------------------------------------------------------------
    def _build(self, shortcuts: Sequence[tuple[str, Path]]) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self, style="Header.TFrame", padding=(20, 14))
        header.grid(row=0, column=0, sticky="ew")
        title = ttk.Frame(header, style="Header.TFrame")
        title.pack(anchor="w")
        ttk.Label(title, text="◆", style="HeaderBrand.TLabel").pack(side="left", padx=(0, 10))
        ttk.Label(title, text="新增地圖", style="HeaderTitle.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="選擇建圖端輸出的場域資料夾（含 site_profile.json）。"
                 "預檢通過後匯入，並切換到這張地圖。",
            style="HeaderMuted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self, style="Gutter.TFrame", padding=(14, 14, 14, 0))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=11, uniform="body")
        body.columnconfigure(1, weight=9, uniform="body")
        body.rowconfigure(0, weight=1)
        browser = ttk.Frame(body, style="Card.TFrame", padding=14)
        browser.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        preview = ttk.Frame(body, style="Card.TFrame", padding=16)
        preview.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self._build_browser(browser, shortcuts)
        self._build_preview(preview)

        footer = ttk.Frame(self, style="Gutter.TFrame", padding=(14, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        status = ttk.Label(footer, textvariable=self.status_var, style="Gutter.TLabel")
        status.grid(row=0, column=0, sticky="ew")
        status.bind("<Configure>", lambda event: status.configure(wraplength=max(200, event.width)))
        ttk.Button(footer, text="取消", command=self.cancel).grid(row=0, column=1, padx=(10, 0))
        self.import_button = ttk.Button(
            footer, text="匯入並切換到此地圖", style="Primary.TButton",
            command=self.confirm, state="disabled",
        )
        self.import_button.grid(row=0, column=2, padx=(8, 0))

    def _build_browser(self, parent, shortcuts: Sequence[tuple[str, Path]]) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        ttk.Label(parent, text="① 選擇資料夾", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w",
        )
        bar = ttk.Frame(parent)
        bar.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        bar.columnconfigure(1, weight=1)
        self.up_button = ttk.Button(bar, text="↑ 上一層", command=self.go_up)
        self.up_button.grid(row=0, column=0, padx=(0, 6))
        self.path_entry = ttk.Entry(bar, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, sticky="ew")
        self.path_entry.bind("<Return>", self._go_typed_path)
        places = ttk.Frame(parent)
        places.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(places, text="快速位置", style="Secondary.TLabel").pack(side="left", padx=(0, 4))
        for label, path in shortcuts:
            ttk.Button(
                places, text=label, style="Ghost.TButton",
                command=lambda target=path: self.navigate(target),
            ).pack(side="left")
        tree_frame = ttk.Frame(parent)
        tree_frame.grid(row=3, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            tree_frame, columns=("kind",), show="tree headings", selectmode="browse",
        )
        self.tree.heading("#0", text="名稱", anchor="w")
        self.tree.heading("kind", text="類型", anchor="w")
        self.tree.column("#0", stretch=True, minwidth=180)
        self.tree.column("kind", width=110, minwidth=90, stretch=False)
        self.tree.tag_configure("site", foreground=HEADING, font=FONT_BOLD)
        self.tree.tag_configure("folder", foreground=TEXT)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns", padx=(2, 0))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-Button-1>", self._on_open)
        self.tree.bind("<Return>", self._on_open)
        self.tree.bind("<BackSpace>", lambda _event: self.go_up())
        hint = ttk.Label(
            parent,
            text="◆ 為含 site_profile.json 的場域資料夾。雙擊或 Enter 進入，Backspace 回上一層。",
            style="Secondary.TLabel",
        )
        hint.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        hint.bind("<Configure>", lambda event: hint.configure(wraplength=max(200, event.width)))

    def _build_preview(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(8, weight=1)
        ttk.Label(parent, text="② 預檢", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.badge = tk.Label(
            parent, font=FONT_SMALL_BOLD, fg=ON_STATE, bg=STATE_WAITING,
            padx=10, pady=3, anchor="w", borderwidth=0, highlightthickness=0,
        )
        self.badge.grid(row=1, column=0, sticky="w", pady=(10, 8))
        ttk.Label(
            parent, textvariable=self.name_var, style="Display.TLabel",
            wraplength=_PREVIEW_WRAP, justify="left",
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(
            parent, textvariable=self.meta_var, style="Secondary.TLabel",
            wraplength=_PREVIEW_WRAP, justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.error_label = ttk.Label(
            parent, textvariable=self.error_var, style="Error.TLabel",
            wraplength=_PREVIEW_WRAP, justify="left",
        )
        self.error_label.grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Separator(parent).grid(row=5, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(parent, text="必需檔案", style="Heading.TLabel").grid(
            row=6, column=0, sticky="w", pady=(12, 6),
        )
        self.files_frame = ttk.Frame(parent)
        self.files_frame.grid(row=7, column=0, sticky="ew")
        self.files_frame.columnconfigure(2, weight=1)
        ttk.Label(
            parent,
            text="預檢只確認檔案存在。匯入時會重新計算 SHA-256、驗證定位 bundle 與參考影像；"
                 "任何不符都會拒絕，目前地圖保持不變。匯入不會起飛，也不會核准 AUTO。",
            style="Secondary.TLabel", wraplength=_PREVIEW_WRAP, justify="left",
        ).grid(row=9, column=0, sticky="sw", pady=(12, 0))

    # ---- navigation ----------------------------------------------------------
    def navigate(self, folder: str | Path, *, select: Path | None = None) -> None:
        try:
            target = Path(folder).expanduser().resolve()
            entries = list_child_folders(target)
        except OSError as exc:
            self.status_var.set(f"無法開啟 {folder}：{exc}")
            return
        self.current_dir = target
        self._entries = entries
        self.path_var.set(str(target))
        self.tree.delete(*self.tree.get_children())
        for index, (path, is_site) in enumerate(entries):
            self.tree.insert(
                "", "end", iid=str(index),
                text=f"◆  {path.name}" if is_site else f"▸  {path.name}",
                values=("場域資料夾" if is_site else "資料夾",),
                tags=("site" if is_site else "folder",),
            )
        self.up_button.state(["disabled"] if target.parent == target else ["!disabled"])
        sites = sum(1 for _path, is_site in entries if is_site)
        self.status_var.set(
            f"{target}｜{len(entries)} 個資料夾，其中 {sites} 個場域資料夾"
            if entries else f"{target}｜沒有子資料夾"
        )
        chosen = next(
            (str(index) for index, (path, _site) in enumerate(entries) if path == select),
            None,
        )
        if chosen is not None:
            self.tree.selection_set(chosen)
            self.tree.focus(chosen)
            self.tree.see(chosen)
        else:
            self.show_preview(target)

    def go_up(self) -> None:
        if self.current_dir is not None and self.current_dir.parent != self.current_dir:
            self.navigate(self.current_dir.parent, select=self.current_dir)

    def _selected_path(self) -> Path | None:
        selection = self.tree.selection()
        return self._entries[int(selection[0])][0] if selection else None

    def _on_select(self, _event=None) -> None:
        path = self._selected_path()
        self.show_preview(path if path is not None else self.current_dir)

    def _on_open(self, _event=None) -> str:
        path = self._selected_path()
        if path is not None:
            self.navigate(path)
        return "break"

    def _go_typed_path(self, _event=None) -> str:
        typed = Path(self.path_var.get().strip()).expanduser()
        if typed.is_dir():
            self.navigate(typed)
        else:
            self.status_var.set(f"找不到資料夾：{typed}")
        return "break"

    # ---- preview -------------------------------------------------------------
    def show_preview(self, folder: Path | None) -> None:
        if folder is None:
            return
        try:
            preview = preview_site_folder(folder)
        except OSError as exc:
            self.status_var.set(f"無法讀取 {folder}：{exc}")
            preview = SiteFolderPreview(Path(folder))
        self.preview = preview
        for child in self.files_frame.winfo_children():
            child.destroy()
        if not preview.files:
            # No manifest here: nothing to import, keep browsing.
            self._set_badge("○ 不是場域資料夾", STATE_WAITING)
            self.name_var.set(preview.folder.name or str(preview.folder))
            self.meta_var.set("請選擇標示 ◆ 的資料夾，或進入含 site_profile.json 的資料夾。")
            self._set_error("")
        else:
            missing = sum(1 for item in preview.files if item.problem is not None)
            if preview.ready:
                self._set_badge("✓ 預檢通過，可匯入", STATE_GOOD)
            elif preview.error is not None:
                self._set_badge("✕ 無法匯入", STATE_BLOCKED)
            else:
                self._set_badge(f"✕ 缺少 {missing} 個必需檔案", STATE_BLOCKED)
            self.name_var.set(preview.display_name or preview.folder.name)
            meta = [part for part in (
                preview.site_id and f"site_id {preview.site_id}",
                preview.coordinate_frame_id and f"座標系 {preview.coordinate_frame_id}",
                preview.camera and f"定位相機 {preview.camera}",
            ) if part]
            self.meta_var.set("\n".join(meta) or str(preview.folder))
            error = preview.error or ""
            if not preview.ready and self._is_listed(preview.folder):
                error = "此資料夾已在地圖清單中，請關閉本視窗並從「① 地圖」選單直接選用。\n" + error
            self._set_error(error.strip())
            self._show_files(preview)
        self.import_button.state(["!disabled"] if preview.ready else ["disabled"])

    def _is_listed(self, folder: Path) -> bool:
        return self.listed_root is not None and folder.parent == self.listed_root

    def _set_badge(self, text: str, colour: str) -> None:
        self.badge.configure(text=text, bg=colour)

    def _set_error(self, text: str) -> None:
        self.error_var.set(text)
        if text:
            self.error_label.grid()
        else:
            self.error_label.grid_remove()

    def _show_files(self, preview: SiteFolderPreview) -> None:
        for row, item in enumerate(preview.files):
            ok = item.problem is None
            ttk.Label(
                self.files_frame, text="✓" if ok else "✕", font=FONT_BOLD,
                foreground=TEXT_GOOD if ok else TEXT_BLOCKED,
            ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Label(self.files_frame, text=item.label).grid(
                row=row, column=1, sticky="w", padx=(0, 10), pady=2,
            )
            name = "" if item.path is None else item.path.name
            ttk.Label(
                self.files_frame,
                text=name if ok else " ".join(part for part in (name, item.problem) if part),
                style="Secondary.TLabel" if ok else "Error.TLabel",
            ).grid(row=row, column=2, sticky="w", pady=2)
            ttk.Label(
                self.files_frame, text=format_size(item.size_bytes), style="Secondary.TLabel",
            ).grid(row=row, column=3, sticky="e", padx=(10, 0), pady=2)

    # ---- result ----------------------------------------------------------------
    def confirm(self) -> None:
        preview = self.preview
        if preview is None or not preview.ready:
            return
        self.destroy()
        self.on_import(preview.folder)

    def cancel(self) -> None:
        self.destroy()
