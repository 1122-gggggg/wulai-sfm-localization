"""Tk layout for the three explicit site-asset import interfaces."""
from __future__ import annotations

import threading
import queue
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from local_site_assets import (
    AvailableSitePackage,
    describe_site_routes,
    discover_site_package,
    list_site_packages,
)
from operator_actions import ActionResult, SiteAssetActions


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
            text="場域資產與航線",
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
        self.site_pack_root = Path(site_pack_root) if site_pack_root else None
        #: Selection is a view of a validated import, not a runtime/AUTO change.
        self._selected_site_pack_root = self.site_pack_root
        self._active_profile_path = actions.current_profile
        self._routes_pending_apply = False
        #: () -> (safe, reason). Anything that grabs input or edits a route must be
        #: refused while the aircraft is not confirmed landed.
        self.flight_state_check = flight_state_check or (lambda: (True, ""))
        self.status_var = tk.StringVar(value="① 請先匯入建圖端輸出資料夾")
        self._busy = False
        self._buttons: list[ttk.Button] = []
        #: Buttons that must stay disabled regardless of the busy state.
        self._permanently_disabled: set[ttk.Button] = set()
        self._pending_site_folder: Path | None = None
        self._result_queue: queue.Queue[tuple[ActionResult | None, Exception | None]] = (
            queue.Queue(maxsize=1)
        )
        self._build()

    @staticmethod
    def _short_site_name(display_name: str) -> str:
        """Operator-facing shorthand: everything before the first bracket or space."""
        for separator in ("（", "(", " ", "\u3000"):
            index = display_name.find(separator)
            if index > 0:
                return display_name[:index]
        return display_name

    def _is_active_site(self, item) -> bool:
        """Whether this listed package is the one already loaded."""
        if self.site_pack_root is None:
            return False
        try:
            return Path(item.folder).resolve() == self.site_pack_root.resolve()
        except OSError:
            return False

    def _active_site_label(self, available) -> str:
        for item in available:
            if self._is_active_site(item):
                return self._short_site_name(item.display_name)
        return ""

    def _build_site_shortcuts(self, row: int) -> None:
        """One button per ready-made site, plus the fallbacks that must always exist.

        Everything except the per-site shortcuts is built unconditionally: an empty
        or unreadable sites directory used to leave this panel with no browse
        button and no route row at all -- a dead end with nothing to click.
        """
        available = []
        if self.site_packages_root is not None:
            try:
                available = list_site_packages(self.site_packages_root)
            except Exception as exc:
                self.status_var.set(f"場域清單讀取失敗，請用「其他資料夾…」：{exc}")
        # Keep the active site available when another import is pending, so the
        # operator can return to its routes without applying the pending site.
        active = self._active_site_label(available)
        if not self._routes_pending_apply:
            available = [item for item in available if not self._is_active_site(item)]
        heading = f"切換場域（目前：{active}）" if active else "切換場域"
        ttk.Label(self, text=heading, font=("Sans", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=8, pady=(4, 2)
        )
        shortcuts = ttk.Frame(self)
        shortcuts.grid(row=row + 1, column=0, sticky="ew", padx=8, pady=(0, 3))
        shortcuts.columnconfigure((0, 1), weight=1)
        if not available:
            text = "（沒有其他場域包）" if active else "（未偵測到場域包）"
            ttk.Label(shortcuts, text=text).grid(row=0, column=0, sticky="w")
        for index, item in enumerate(available):
            button = ttk.Button(
                shortcuts,
                text=self._short_site_name(item.display_name),
                command=lambda folder=item.folder: self._import_site_folder(folder),
            )
            if item.error:
                # Listed but unusable: hiding it would leave the operator wondering
                # where the site went. Recorded so _set_busy cannot re-enable it.
                button.state(["disabled"])
                self._permanently_disabled.add(button)
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0, 4), pady=1)
            self._buttons.append(button)
        browse = ttk.Button(shortcuts, text="其他資料夾…", command=self._choose_site)
        slot = max(1, len(available))
        browse.grid(row=slot // 2, column=slot % 2, sticky="ew", padx=(0, 4), pady=1)
        self._buttons.append(browse)

        # A standing entry point for the route. The switch flow only offers it once,
        # so declining there used to leave no way back to an existing route.
        ttk.Label(self, text="航線", font=("Sans", 10, "bold")).grid(
            row=row + 2, column=0, sticky="w", padx=8, pady=(1, 2)
        )
        route_row = ttk.Frame(self)
        route_row.grid(row=row + 3, column=0, sticky="ew", padx=8, pady=(0, 3))
        route_row.columnconfigure(0, weight=1)
        button = ttk.Button(
            route_row, text="新增路線", command=lambda: self._open_route_editor(False)
        )
        button.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=1)
        self._buttons.append(button)

    def _build_route_choices(self, row: int) -> int:
        """List the selected site's routes without authorizing runtime selection."""
        root = self._selected_site_pack_root
        heading = (
            f"待套用場域：{root.name}（套用後才能選取航線）"
            if self._routes_pending_apply and root is not None
            else "下次 AUTO 航線（點擊選取）"
        )
        ttk.Label(self, text=heading, font=("Sans", 10, "bold"), wraplength=250).grid(
            row=row, column=0, sticky="w", padx=8, pady=(1, 2)
        )
        roots = [] if root is None else [root]
        profile = self.actions.current_profile
        if self._routes_pending_apply and profile is not None:
            managed_root = Path(profile).parent
            if root is None or managed_root.resolve() != root.resolve():
                roots.append(managed_root)
        try:
            routes = [item for folder in roots for item in describe_site_routes(folder)]
        except Exception as exc:
            ttk.Label(self, text=f"航線清單讀取失敗：{exc}", wraplength=250).grid(
                row=row + 1, column=0, sticky="w", padx=8, pady=(0, 3)
            )
            return row + 2
        if not routes:
            ttk.Label(self, text="此場域尚無可用航線", wraplength=250).grid(
                row=row + 1, column=0, sticky="w", padx=8, pady=(0, 3)
            )
            return row + 2
        choices = ttk.Frame(self)
        choices.grid(row=row + 1, column=0, sticky="ew", padx=8, pady=(0, 3))
        choices.columnconfigure((0, 1), weight=1)
        for index, item in enumerate(routes):
            self._build_route_choice(choices, index, item)
        return row + 2

    def _build_route_choice(self, choices, index: int, item) -> None:
        card = ttk.Frame(choices)
        card.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0, 4), pady=1)
        card.columnconfigure(0, weight=1)
        button = ttk.Button(
            card, text=item.label,
            command=lambda: self._preview_existing_route(item.path),
        )
        button.grid(row=0, column=0, sticky="ew")
        if not item.flight_ready or self._routes_pending_apply:
            button.state(["disabled"])
            self._permanently_disabled.add(button)
        controls = ttk.Frame(card)
        edit = ttk.Button(
            controls, text="編輯", width=4,
            command=lambda: self._edit_existing_route(item.path),
        )
        edit.pack(side="left")
        delete = ttk.Button(
            controls, text="×", width=2,
            command=lambda: self._delete_existing_route(item.path),
        )
        delete.pack(side="left")
        self._buttons.extend((button, edit, delete))
        if self._routes_pending_apply:
            for control in (edit, delete):
                control.state(["disabled"])
                self._permanently_disabled.add(control)

        def show_controls(_event=None) -> None:
            controls.place(relx=1.0, y=0, anchor="ne")
            controls.lift()

        def hide_if_outside() -> None:
            if not card.winfo_exists():
                return
            x, y = card.winfo_pointerxy()
            inside = (
                card.winfo_rootx() <= x < card.winfo_rootx() + card.winfo_width()
                and card.winfo_rooty() <= y < card.winfo_rooty() + card.winfo_height()
            )
            if not inside:
                controls.place_forget()

        for widget in (card, button, controls, edit, delete):
            widget.bind("<Enter>", show_controls)
            # Moving onto either hover action is still inside the same card.
            widget.bind("<Leave>", lambda _event: card.after_idle(hide_if_outside))

    def _edit_existing_route(self, path: Path) -> None:
        if getattr(self, "_routes_pending_apply", False):
            self.status_var.set("待套用場域：請先套用場域再編輯清單航線")
            return
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
        if getattr(self, "_routes_pending_apply", False):
            self.status_var.set("待套用場域：請先套用場域再刪除清單航線")
            return
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

    def _preview_existing_route(self, path: Path) -> None:
        """Validate and select one route for this session's next AUTO request.

        Deliberately NOT import_route: that needs a managed profile under the site
        packages root (local_site_assets._load_managed_profile) and resets flight
        approval, so it stays behind the route editor's save step. Flipping
        routes changes the immutable session selection only while AUTO is inactive;
        it does not approve, arm, take off, or persist a different site profile.
        """
        if getattr(self, "_busy", False):
            self.status_var.set("航線選擇：前一個資產作業尚未完成")
            return
        if getattr(self, "_routes_pending_apply", False):
            self.status_var.set("待套用場域：請先套用場域再選取 AUTO 航線")
            return
        if self.request_route_preview is None:
            self.status_var.set("航線顯示尚未接上")
            return
        safe, reason = self.flight_state_check()
        if not safe:
            self.status_var.set(f"航線選擇已拒絕：{reason}")
            return
        try:
            self.status_var.set(self.request_route_preview(path))
        except Exception as exc:
            self.status_var.set(f"航線顯示失敗（{path.name}）：{exc}")

    def _import_site_folder(self, folder) -> None:
        if self._busy:
            self.status_var.set("場域匯入：前一個資產作業尚未完成")
            return
        self._pending_site_folder = Path(folder)
        self._run("場域匯入", lambda: self.actions.import_site_folder(str(folder)))

    def _build(self) -> None:
        # Only the site switch remains (operator decision 2026-08-06). Everything the
        # removed rows did is now automatic: switching a site detects its route and,
        # when there is none, goes straight into marking one. The route-editor and
        # target-import methods are kept -- that automatic flow calls them.
        self._build_site_shortcuts(0)
        status_row = self._build_route_choices(4)
        ttk.Label(self, text="一般匯入不解鎖；請在編輯器明確核准航線。",
                  wraplength=250, font=("Sans", 9)).grid(
            row=status_row, column=0, sticky="w", padx=8, pady=(2, 1)
        )
        ttk.Label(self, textvariable=self.status_var, wraplength=250).grid(
            row=status_row + 1, column=0, sticky="w", padx=8, pady=(1, 2)
        )
        # The standalone 重啟套用 row is gone (operator decision 2026-08-06). Applying
        # is what switching a site MEANS, so it is offered by the switch itself --
        # removing the row without that would leave an imported site never active.
        self.columnconfigure(0, weight=1)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in self._buttons:
            if not busy and button in self._permanently_disabled:
                continue          # unusable package: never becomes clickable
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
            folder, self._pending_site_folder = self._pending_site_folder, None
            outcome = "declined"
            if folder is not None:
                self._select_site_folder(folder, result.profile_path)
                outcome = self.follow_up_route_for_site(folder)
                # Automatic route import can create a managed route; refresh again
                # before offering activation, never leaving the previous list.
                self._refresh()
            # Never switch the runtime out from under an editor that was just
            # opened to draw the route.
            if outcome != "drawing":
                self.offer_apply_after_import()

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

    def pick_site_package(self, available: list[AvailableSitePackage]) -> str | None:
        """Pick from the known packages, or fall back to browsing for a folder."""
        if not available:
            return filedialog.askdirectory(
                title="選擇包含 site_profile.json 的建圖輸出資料夾"
            ) or None

        dialog = tk.Toplevel(self)
        dialog.title("選擇要匯入的地圖")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        ttk.Label(
            dialog, text=f"偵測到 {len(available)} 個場域包，請選擇要匯入的地圖：",
            wraplength=520,
        ).pack(anchor="w", padx=12, pady=(12, 6))
        listbox = tk.Listbox(dialog, height=min(10, len(available)), width=64,
                             exportselection=False)
        for item in available:
            listbox.insert("end", item.label)
        listbox.selection_set(0)
        listbox.pack(fill="both", expand=True, padx=12)

        chosen: dict[str, str | None] = {"folder": None}

        def confirm() -> None:
            index = (listbox.curselection() or (0,))[0]
            item = available[index]
            if item.error:
                # Importing it would fail anyway, and the reason is already on screen.
                self.status_var.set(f"{item.folder.name} 無法匯入：{item.error}")
                return
            chosen["folder"] = str(item.folder)
            dialog.destroy()

        def browse() -> None:
            dialog.destroy()
            chosen["folder"] = filedialog.askdirectory(
                title="選擇包含 site_profile.json 的建圖輸出資料夾"
            ) or None

        row = ttk.Frame(dialog)
        row.pack(fill="x", padx=12, pady=10)
        ttk.Button(row, text="匯入這個地圖", command=confirm).pack(side="left")
        ttk.Button(row, text="其他資料夾…", command=browse).pack(side="left", padx=6)
        ttk.Button(row, text="取消", command=dialog.destroy).pack(side="right")
        listbox.bind("<Double-Button-1>", lambda _event: confirm())
        dialog.wait_window()
        return chosen["folder"]

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
        # Offer the packages that already exist before asking the operator to go
        # find one: on this system the answer is almost always one of three.
        available = []
        if self.site_packages_root is not None:
            try:
                available = list_site_packages(self.site_packages_root)
            except Exception as exc:
                self.status_var.set(f"場域清單讀取失敗，改用資料夾選擇：{exc}")
        folder = self.pick_site_package(available)
        if folder:
            # Remembered so the route follow-up can search the same folder the
            # operator picked, rather than asking them where it went.
            self._import_site_folder(folder)

    def _open_route_editor(self, edit_current: bool) -> None:
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

    def offer_apply_after_import(self) -> bool:
        """Ask whether to activate the imported site. True if applying."""
        if self.actions.current_profile is None:
            return False
        if not self.ask_yes_no(
            "切換到這個場域",
            "匯入完成。要立即重建定位／飛控連線並套用這個場域嗎？"
            "（介面保持開啟，僅限已落地）",
        ):
            self.status_var.set(
                f"{self.status_var.get()}；尚未套用，之後再切換場域即可"
            )
            return False
        self._apply()
        return True

    def _apply(self) -> None:
        profile = self.actions.current_profile
        if profile is None:
            self.status_var.set("套用失敗：請先匯入建圖端場域資料夾")
            return
        try:
            self.request_apply(profile)
        except Exception as exc:
            self.status_var.set(f"套用失敗：{exc}")
            if self._active_profile_path is not None:
                self.activate_site(self._active_profile_path, self.site_pack_root)

    def _select_site_folder(self, folder: Path, profile_path: Path | None) -> None:
        self._selected_site_pack_root = Path(folder).expanduser().resolve()
        imported_profile = None if profile_path is None else Path(profile_path).resolve()
        active_profile = self._active_profile_path
        same_profile = (
            imported_profile is not None
            and active_profile is not None
            and imported_profile == Path(active_profile).resolve()
        )
        same_pack = (
            imported_profile is not None
            and self.site_pack_root is not None
            and imported_profile.parent == self.site_pack_root.resolve()
        )
        self._routes_pending_apply = not (same_profile or same_pack)
        self._refresh()

    def _refresh(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._buttons.clear()
        self._permanently_disabled.clear()
        self._build()

    def activate_site(
        self,
        profile_path: str | Path,
        site_pack_root: str | Path | None,
    ) -> None:
        """Rebuild shortcuts and routes after an in-process runtime swap."""
        self.actions.current_profile = Path(profile_path).expanduser().resolve()
        self._active_profile_path = self.actions.current_profile
        self.site_pack_root = (
            None
            if site_pack_root is None
            else Path(site_pack_root).expanduser().resolve()
        )
        self._selected_site_pack_root = self.site_pack_root
        self._routes_pending_apply = False
        self._pending_site_folder = None
        self._result_queue = queue.Queue(maxsize=1)
        self._refresh()
