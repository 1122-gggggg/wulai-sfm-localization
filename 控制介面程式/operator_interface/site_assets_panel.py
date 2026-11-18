"""Active-map and route selection, with validated new-folder import fallback."""
from __future__ import annotations

import threading
import queue
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from add_map_dialog import AddMapDialog
from local_site_assets import (
    AvailableSitePackage,
    describe_site_routes,
    discover_site_package,
    list_site_packages,
)
from operator_actions import ActionResult, SiteAssetActions
from operator_ui_style import FONT_SMALL, TEXT_BOX


class _RouteSlot:
    """A DiscoveredAsset narrowed to the approved candidates."""

    __slots__ = ("candidates", "path", "found", "ambiguous")

    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        self.path = candidates[0] if len(candidates) == 1 else None
        self.found = bool(candidates)
        self.ambiguous = len(candidates) > 1


class SiteAssetsPanel(ttk.LabelFrame):
    """View only: file pickers delegate all validation and writes to actions."""

    def __init__(
        self,
        parent,
        *,
        actions: SiteAssetActions,
        request_apply: Callable[[Path], None],
        request_route_editor: Callable[[bool], None] | None = None,
        request_route_preview: Callable[[Path], str] | None = None,
        request_route_delete: Callable[[Path], ActionResult] | None = None,
        request_route_edit: Callable[[Path], None] | None = None,
        route_imported: Callable[[ActionResult], None] | None = None,
        site_packages_root: Path | None = None,
        site_pack_root: Path | None = None,
        flight_state_check: Callable[[], tuple[bool, str]] | None = None,
    ):
        super().__init__(
            parent,
            text="地圖與 AUTO 航線",
        )
        self.actions = actions
        self.request_apply = request_apply
        self.request_route_editor = request_route_editor
        #: (path) -> operator message. Draws one route as the map overlay without
        #: touching the site profile.
        self.request_route_preview = request_route_preview
        self.request_route_delete = request_route_delete
        self.request_route_edit = request_route_edit
        self.route_imported = route_imported
        #: Where the ready-made site packages live. None falls back to browsing.
        self.site_packages_root = site_packages_root
        #: The ACTIVE site's own folder. current_profile.parent is not it when the
        #: app was started from a system profile under site_profiles/, which is
        #: what left the route picker opening on a directory holding no routes.
        self.site_pack_root = (
            Path(site_pack_root).expanduser().resolve() if site_pack_root else None
        )
        self._active_profile_path = actions.current_profile
        self._selected_route_path: Path | None = None
        self._map_choices: list[AvailableSitePackage] = []
        self._route_choices = []
        #: () -> (safe, reason). Refuse input grabs and asset changes unless landed.
        self.flight_state_check = flight_state_check or (lambda: (True, ""))
        self.status_var = tk.StringVar(value="選擇地圖，再選取此地圖的 AUTO 航線")
        self._busy = False
        self._buttons: list[ttk.Button] = []
        self._pending_site_folder: Path | None = None
        self._add_map_dialog: AddMapDialog | None = None
        self._last_browse_dir: Path | None = None
        self._result_queue: queue.Queue[tuple[ActionResult | None, Exception | None]] = (
            queue.Queue(maxsize=1)
        )
        self._build()
        self.status_var.trace_add("write", self._show_status)

    def _is_active_site(self, item) -> bool:
        if self.site_pack_root is None:
            return False
        return Path(item.folder).resolve() == self.site_pack_root.resolve()

    def _restore_map_selection(self) -> None:
        active = next(
            (index for index, item in enumerate(self._map_choices)
             if self._is_active_site(item)),
            None,
        )
        if active is None:
            self._map_selector.set("")
        else:
            self._map_selector.current(active)

    def _build_site_shortcuts(self, row: int) -> None:
        available = []
        if self.site_packages_root is not None:
            try:
                available = list_site_packages(self.site_packages_root)
            except Exception as exc:
                self.status_var.set(f"地圖清單讀取失敗，仍可匯入新資料夾：{exc}")
        if self.site_pack_root is not None and not any(
            self._is_active_site(item) for item in available
        ):
            available.insert(0, AvailableSitePackage(
                self.site_pack_root, "", self.site_pack_root.name,
            ))
        self._map_choices = available
        choices = ttk.Frame(self)
        choices.grid(row=row, column=0, sticky="ew", padx=8, pady=(6, 6))
        choices.columnconfigure(1, weight=1)
        ttk.Label(choices, text="① 地圖", style="Heading.TLabel").grid(
            row=0, column=0, padx=(0, 6),
        )
        self._map_selector = ttk.Combobox(
            choices, state="readonly", width=14,
            values=tuple(
                f"{item.label}（目前）" if self._is_active_site(item) else item.label
                for item in available
            ),
        )
        self._map_selector.grid(row=0, column=1, sticky="ew")
        self._map_selector.bind("<<ComboboxSelected>>", self._select_registered_site)
        self._restore_map_selection()
        self.add_map_button = ttk.Button(choices, text="＋ 新增地圖", command=self._choose_site)
        self.add_map_button.grid(row=0, column=2, padx=(6, 0))
        self._buttons.append(self.add_map_button)
        if not available:
            self._map_selector.set("未偵測到地圖")

    def _select_registered_site(self, _event=None) -> None:
        index = self._map_selector.current()
        self._restore_map_selection()
        if index < 0 or index >= len(self._map_choices):
            return
        item = self._map_choices[index]
        if not self._asset_change_allowed("地圖選擇"):
            return
        if item.error:
            self.status_var.set(f"地圖無法套用（{item.folder.name}）：{item.error}")
            return
        if self._is_active_site(item):
            self.status_var.set("目前地圖已啟用；請從下方選取航線")
            return
        self._apply(Path(item.folder) / "site_profile.json")

    def _build_route_choices(self, row: int) -> int:
        """Only the activated runtime's map supplies selectable routes."""
        ttk.Label(self, text="② 此地圖的 AUTO 航線", style="Heading.TLabel").grid(
            row=row, column=0, sticky="w", padx=8, pady=(2, 4)
        )
        try:
            self._route_choices = (
                [] if self.site_pack_root is None
                else describe_site_routes(self.site_pack_root)
            )
            empty_message = (
                "尚未啟用地圖" if self.site_pack_root is None else "此地圖尚無可用航線"
            )
        except Exception as exc:
            self._route_choices = []
            empty_message = f"航線清單讀取失敗：{exc}"
        if not any(item.path == self._selected_route_path for item in self._route_choices):
            self._selected_route_path = None
        choices = ttk.Frame(self)
        choices.grid(row=row + 1, column=0, sticky="nsew", padx=8, pady=(0, 3))
        choices.columnconfigure(0, weight=1)
        choices.rowconfigure(0, weight=1)
        self._route_list = ttk.Treeview(
            choices, show="tree", selectmode="browse", height=5, style="Route.Treeview",
        )
        self._route_list.column("#0", width=180, minwidth=100, stretch=True)
        scrollbar = ttk.Scrollbar(choices, orient="vertical", command=self._route_list.yview)
        self._route_list.configure(yscrollcommand=scrollbar.set)
        self._route_list.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        for index, item in enumerate(self._route_choices):
            self._route_list.insert("", "end", iid=str(index), text=item.label)
        self._restore_route_selection()
        self._route_list.bind("<<TreeviewSelect>>", self._select_route)
        self._route_empty_message = empty_message if not self._route_choices else ""
        if not self._route_choices:
            empty = ttk.Label(
                self._route_list, text=empty_message, wraplength=250,
                style="Secondary.TLabel", padding=6,
            )
            empty.place(x=2, y=2, relwidth=0.95)
            empty.bind(
                "<Configure>",
                lambda event: empty.configure(wraplength=max(100, event.width - 12)),
            )
        controls = ttk.Frame(self)
        controls.grid(row=row + 3, column=0, sticky="ew", padx=8, pady=(0, 3))
        controls.columnconfigure((0, 1, 2), weight=1)
        for column, (label, command) in enumerate((
            ("新增路線", lambda: self._open_route_editor(False)),
            ("編輯選取", lambda: self._route_action(self._edit_existing_route)),
            ("刪除選取", lambda: self._route_action(self._delete_existing_route)),
        )):
            button = ttk.Button(
                controls, text=label, command=command,
                style="Primary.TButton" if column == 0 else "TButton", width=7,
            )
            button.grid(row=0, column=column, sticky="ew", padx=(0, 3))
            self._buttons.append(button)
        self.rowconfigure(row + 1, weight=1, minsize=48)
        return row + 4

    def _restore_route_selection(self) -> None:
        selected = next(
            (str(index) for index, item in enumerate(self._route_choices)
             if item.path == self._selected_route_path),
            None,
        )
        current = self._route_list.selection()
        target = () if selected is None else (selected,)
        if current != target:
            self._route_list.selection_set(target)

    def _select_route(self, _event=None) -> None:
        selection = self._route_list.selection()
        if not selection:
            return
        item = self._route_choices[int(selection[0])]
        if item.path == self._selected_route_path:
            return
        if not item.flight_ready:
            self.status_var.set("此航線僅供顯示，不能選取為 AUTO 航線")
        elif self._preview_existing_route(item.path):
            self._selected_route_path = item.path

    def _route_action(self, action: Callable[[Path], None]) -> None:
        selection = self._route_list.selection()
        if not selection:
            self.status_var.set("請先從此地圖的清單選取一條航線")
            return
        action(self._route_choices[int(selection[0])].path)

    def _asset_change_allowed(self, label: str) -> bool:
        if self._busy:
            self.status_var.set(f"{label}：前一個資產作業尚未完成")
            return False
        safe, reason = self.flight_state_check()
        if not safe:
            self.status_var.set(f"{label}已拒絕：{reason}")
            return False
        return True


    def _edit_existing_route(self, path: Path) -> None:
        if self._busy:
            self.status_var.set("航線編輯：前一個資產作業尚未完成")
            return
        safe, reason = self.flight_state_check()
        if not safe:
            self.status_var.set(f"航線編輯已拒絕：{reason}")
            return
        if self.request_route_edit is None:
            self.status_var.set("航線編輯器尚未接上")
            return
        try:
            self.request_route_edit(path)
        except Exception as exc:
            self.status_var.set(f"開啟航線編輯器失敗（{path.name}）：{exc}")

    def _delete_existing_route(self, path: Path) -> None:
        if self._busy:
            self.status_var.set("航線刪除：前一個資產作業尚未完成")
            return
        if self.request_route_delete is None:
            self.status_var.set("航線刪除尚未接上")
            return
        if not self.ask_yes_no(
            "刪除航線",
            f"確定永久刪除 {path.name}？\n"
            "其他航線會保留；若刪除目前選取的航線，必須重新選取或新增才能 AUTO。",
        ):
            return
        safe, reason = self.flight_state_check()
        if not safe:
            self.status_var.set(f"航線刪除已拒絕：{reason}")
            return
        if self._busy:
            self.status_var.set("航線刪除：前一個資產作業尚未完成")
            return
        try:
            result = self.request_route_delete(path)
        except Exception as exc:
            self.status_var.set(f"航線刪除失敗（{path.name}）：{exc}")
            return
        self._finish_ok(result)

    def _preview_existing_route(self, path: Path) -> bool:
        """Validate and select one route for this session's next AUTO request.

        Deliberately NOT import_route: that needs a managed profile under the site
        packages root (local_site_assets._load_managed_profile) and resets flight
        approval, so it stays behind the route editor's save step. Flipping
        routes changes the immutable session selection only while AUTO is inactive;
        it does not approve, arm, take off, or persist a different site profile.
        """
        if getattr(self, "_busy", False):
            self.status_var.set("航線選擇：前一個資產作業尚未完成")
            return False
        if self.request_route_preview is None:
            self.status_var.set("航線顯示尚未接上")
            return False
        safe, reason = self.flight_state_check()
        if not safe:
            self.status_var.set(f"航線選擇已拒絕：{reason}")
            return False
        try:
            self.status_var.set(self.request_route_preview(path))
        except Exception as exc:
            self.status_var.set(f"航線顯示失敗（{path.name}）：{exc}")
            return False
        return True

    def _import_site_folder(self, folder) -> None:
        if not self._asset_change_allowed("場域匯入"):
            return
        self._pending_site_folder = Path(folder)
        self._run("場域匯入", lambda: self.actions.import_site_folder(str(folder)))

    def _build(self) -> None:
        self._build_site_shortcuts(0)
        status_row = self._build_route_choices(1)
        status = ttk.Frame(self)
        status.grid(row=status_row, column=0, sticky="ew", padx=8, pady=(1, 5))
        self._status_text = tk.Text(
            status, width=1, height=2, wrap="word", state="disabled", **TEXT_BOX,
            font=FONT_SMALL, padx=6, pady=4, takefocus=False,
        )
        scrollbar = ttk.Scrollbar(status, orient="vertical", command=self._status_text.yview)
        self._status_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._status_text.pack(side="left", fill="x", expand=True)
        self._show_status()
        self.columnconfigure(0, weight=1)

    def _show_status(self, *_args) -> None:
        """Keep full pending/error messages readable without growing the workspace."""
        text = getattr(self, "_status_text", None)
        if text is None or not text.winfo_exists():
            return
        text.configure(state="normal")
        text.delete("1.0", "end")
        messages = [self.status_var.get()]
        if self._route_empty_message:
            messages.append(self._route_empty_message)
        selected_map = self._map_selector.get()
        if selected_map:
            messages.append(f"地圖：{selected_map}")
        messages.append("編輯器儲存可核准 AUTO；外部匯入不會自動核准。")
        text.insert("1.0", "\n".join(messages))
        text.configure(state="disabled")
        text.yview_moveto(0)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._map_selector.configure(state="disabled" if busy else "readonly")
        self._route_list.state(["disabled"] if busy else ["!disabled"])
        state = "disabled" if busy else "normal"
        for button in self._buttons:
            button.configure(state=state)

    def _run(self, label: str, action: Callable[[], ActionResult]) -> None:
        if getattr(self, "_busy", False):
            self.status_var.set(f"{label}：前一個資產作業尚未完成")
            return
        self._busy = True
        self._set_busy(True)
        self.status_var.set(f"{label}：驗證中…")

        def worker() -> None:
            try:
                result = action()
            except Exception as exc:
                self._result_queue.put((None, exc))
            else:
                self._result_queue.put((result, None))

        threading.Thread(target=worker, name="site-asset-import", daemon=True).start()
        self.after(50, lambda: self._poll_result(label))

    def _poll_result(self, label: str) -> None:
        try:
            result, error = self._result_queue.get_nowait()
        except queue.Empty:
            self.after(50, lambda: self._poll_result(label))
            return
        if error is not None:
            self._finish_error(label, error)
        else:
            assert result is not None
            self._finish_ok(result)

    def _finish_error(self, label: str, exc: Exception) -> None:
        self._pending_site_folder = None
        self._busy = False
        self._set_busy(False)
        self.status_var.set(f"{label}失敗：{exc}")

    def _finish_ok(self, result: ActionResult) -> None:
        self._busy = False
        self._set_busy(False)
        self.status_var.set(result.message)
        if result.kind in {"route", "route_deleted"}:
            if result.kind == "route" and self.route_imported is not None:
                self.route_imported(result)
            self._refresh()
            self.status_var.set(result.message)
        elif result.kind == "site":
            self._pending_site_folder = None
            self._apply(result.profile_path)

    def _approved_route_slot(self, slot):
        if slot is None or not slot.found:
            return slot, None
        approved = [
            path for path in slot.candidates if "route_drafts" not in path.parts
        ]
        if not approved:
            self.status_var.set(
                f"只找到 {len(slot.candidates)} 份航線草稿（route_drafts/），"
                "不會自動匯入；請用「新增路線」開啟編輯器確認後存檔"
            )
            return None, "drafts_only"
        if len(approved) != len(slot.candidates):
            return _RouteSlot(approved), None
        return slot, None

    def _handle_missing_route(self, folder: Path) -> str:
        if self.ask_yes_no(
            "這個場域還沒有航線",
            f"在 {folder.name} 找不到航線檔。要現在開啟編輯器標路徑點嗎？",
        ):
            self._open_route_editor(False)
            return "drawing"
        self.status_var.set("場域已匯入；尚無航線，之後可從「新增路線」開始")
        return "declined"

    def _import_discovered_route(self, slot) -> str:
        try:
            result = self.actions.import_route(str(slot.path))
        except Exception as exc:
            self.status_var.set(f"自動匯入航線失敗：{exc}")
            if self.ask_yes_no("航線匯入失敗", f"{exc}\n\n要改用編輯器重畫嗎？"):
                self._open_route_editor(False)
                return "drawing"
            return "declined"
        self.status_var.set(f"已自動匯入航線 {slot.path.name}；{result.message}")
        if self.route_imported is not None:
            self.route_imported(result)
        if self.ask_yes_no(
            "已匯入現有航線",
            f"已匯入 {slot.path.name}。要開啟編輯器修改這條航線嗎？",
        ):
            self._open_route_editor(True)
        return "imported"

    def follow_up_route_for_site(self, folder: Path) -> str:
        """After a map import, settle the route without making the operator hunt.

        Returns a short outcome tag so this is testable without Tk dialogs:
        "imported" / "drawing" / "ambiguous" / "declined" / "discovery_failed".
        A site is not usable until it has a route, so the one thing this must never
        do is import a map and say nothing about the route.
        """
        try:
            package = discover_site_package(folder)
        except Exception as exc:
            self.status_var.set(f"{self.status_var.get()}；航線探索失敗：{exc}")
            return "discovery_failed"

        slot, outcome = SiteAssetsPanel._approved_route_slot(
            self,
            package.asset("route_json"),
        )
        if outcome is not None:
            return outcome
        if slot is None or not slot.found:
            return SiteAssetsPanel._handle_missing_route(self, folder)

        if slot.ambiguous:
            # Which route the aircraft flies is not a guess worth making.
            self.status_var.set(
                f"偵測到 {len(slot.candidates)} 個航線檔，請從航線清單點擊選取要飛的一份"
            )
            return "ambiguous"
        return SiteAssetsPanel._import_discovered_route(self, slot)


    def ask_yes_no(self, title: str, message: str) -> bool:
        """Seam so the follow-up flow is testable without a real dialog.

        Refuses to open a dialog unless the aircraft is confirmed landed: a Tk
        message box takes a LOCAL GRAB, which redirects every pointer and key event
        to itself -- 原地降落 and 緊急停止電腦動作 in the always-visible flight bar
        become unclickable for as long as it is up.
        """
        safe, reason = self.flight_state_check()
        if not safe:
            self.status_var.set(f"{title}：飛行中不開對話框（{reason}）")
            return False
        return bool(messagebox.askyesno(title, message, parent=self))

    def _choose_site(self) -> None:
        """Open 新增地圖; the dialog only proposes a folder, the importer decides."""
        if not self._asset_change_allowed("新增地圖"):
            return
        dialog = self._add_map_dialog
        if dialog is not None and dialog.winfo_exists():
            dialog.deiconify()
            dialog.lift()
            return
        # Parented to the window, not this panel: _refresh() destroys the
        # panel's children and would otherwise close the dialog mid-browse.
        self._add_map_dialog = AddMapDialog(
            self.winfo_toplevel(),
            start_dir=self._last_browse_dir or Path.home(),
            shortcuts=self._browse_shortcuts(),
            on_import=self._import_from_dialog,
            listed_root=self.site_packages_root,
        )

    def _browse_shortcuts(self) -> list[tuple[str, Path]]:
        home = Path.home()
        places = [("家目錄", home), ("下載", home / "下載"), ("Downloads", home / "Downloads"),
                  ("桌面", home / "桌面")]
        if self.site_packages_root is not None:
            places.append(("已匯入地圖", Path(self.site_packages_root)))
        return [(label, path) for label, path in places if path.is_dir()]

    def _import_from_dialog(self, folder: Path) -> None:
        self._add_map_dialog = None
        self._last_browse_dir = Path(folder).parent
        self._import_site_folder(folder)

    def _open_route_editor(self, edit_current: bool) -> None:
        if not self._asset_change_allowed("新增路線"):
            return
        if self.request_route_editor is None:
            self.status_var.set("航線編輯器尚未接上")
            return
        try:
            self.request_route_editor(bool(edit_current))
        except Exception as exc:
            self.status_var.set(f"開啟航線編輯器失敗：{exc}")

    def import_authored_route(
        self,
        source: str | Path,
        *,
        replace_route: str | Path | None = None,
    ) -> ActionResult:
        """Validate and approve an editor-produced route for AUTO."""
        result = self.actions.import_authored_route(
            source,
            replace_route=replace_route,
        )
        self._finish_ok(result)
        return result

    def set_status(self, message: str) -> None:
        self.status_var.set(str(message))

    def _choose_targets(self) -> None:
        source = filedialog.askopenfilename(
            title="選擇電桿／目標物 JSON", filetypes=(("JSON", "*.json"),)
        )
        if source:
            self._run("巡檢目標匯入", lambda: self.actions.import_targets(source))


    def _apply(self, profile: Path | None = None) -> None:
        if not self._asset_change_allowed("地圖切換"):
            self.actions.current_profile = self._active_profile_path
            return
        profile = profile if profile is not None else self.actions.current_profile
        if profile is None:
            self.status_var.set("套用失敗：請先匯入建圖端場域資料夾")
            return
        self._set_busy(True)
        self.status_var.set("地圖切換：驗證中，成功啟用後才更新航線…")
        try:
            self.request_apply(profile)
        except Exception as exc:
            if self._active_profile_path is not None:
                self.activate_site(self._active_profile_path, self.site_pack_root)
            self._set_busy(False)
            self.status_var.set(f"套用失敗：{exc}")

    def _refresh(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._buttons.clear()
        self._build()
        self._set_busy(self._busy)

    def activate_site(
        self,
        profile_path: str | Path,
        site_pack_root: str | Path | None,
    ) -> None:
        """Update map/routes only after activation, or restore after a rejected swap."""
        previous_root = self.site_pack_root
        self.actions.current_profile = Path(profile_path).expanduser().resolve()
        self._active_profile_path = self.actions.current_profile
        self.site_pack_root = (
            None
            if site_pack_root is None
            else Path(site_pack_root).expanduser().resolve()
        )
        if previous_root != self.site_pack_root:
            self._selected_route_path = None
        self._busy = False
        self._pending_site_folder = None
        self._result_queue = queue.Queue(maxsize=1)
        self._refresh()
