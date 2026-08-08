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
    list_site_routes,
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
        route_imported: Callable[[ActionResult], None] | None = None,
        site_packages_root: Path | None = None,
        site_pack_root: Path | None = None,
        flight_state_check: Callable[[], tuple[bool, str]] | None = None,
    ):
        super().__init__(parent, text="場域資產（匯入不會起飛，也不會解鎖自主飛行）")
        self.actions = actions
        self.request_apply = request_apply
        self.request_route_editor = request_route_editor
        #: (path) -> operator message. Draws one route as the map overlay without
        #: touching the site profile.
        self.request_route_preview = request_route_preview
        self.route_imported = route_imported
        #: Where the ready-made site packages live. None falls back to browsing.
        self.site_packages_root = site_packages_root
        #: The ACTIVE site's own folder. current_profile.parent is not it when the
        #: app was started from a system profile under site_profiles/, which is
        #: what left the route picker opening on a directory holding no routes.
        self.site_pack_root = Path(site_pack_root) if site_pack_root else None
        #: () -> (safe, reason). Anything that grabs input or edits a route must be
        #: refused while the aircraft is not confirmed landed.
        self.flight_state_check = flight_state_check or (lambda: (True, ""))
        self.status_var = tk.StringVar(value="① 請先匯入建圖端輸出資料夾")
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
        # Switching to the site already loaded re-imports it and re-execs the whole
        # app to arrive back where it started. Drop that button rather than leave a
        # control whose only effect is a restart.
        active = self._active_site_label(available)
        available = [item for item in available if not self._is_active_site(item)]
        heading = f"切換場域（目前：{active}）" if active else "切換場域"
        ttk.Label(self, text=heading, font=("Sans", 9, "bold")).grid(
            row=row, column=0, sticky="w", padx=(8, 5), pady=(5, 2)
        )
        shortcuts = ttk.Frame(self)
        shortcuts.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=(5, 2))
        if not available:
            text = "（沒有其他場域包）" if active else "（未偵測到場域包）"
            ttk.Label(shortcuts, text=text).pack(side="left", padx=(0, 6))
        for item in available:
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
            button.pack(side="left", padx=(0, 6))
            self._buttons.append(button)
        browse = ttk.Button(shortcuts, text="其他資料夾…", command=self._choose_site)
        browse.pack(side="left", padx=(6, 0))
        self._buttons.append(browse)

        # A standing entry point for the route. The switch flow only offers it once,
        # so declining there used to leave no way back to an existing route.
        ttk.Label(self, text="航線", font=("Sans", 9, "bold")).grid(
            row=row + 1, column=0, sticky="w", padx=(8, 5), pady=(2, 2)
        )
        route_row = ttk.Frame(self)
        route_row.grid(row=row + 1, column=1, columnspan=2, sticky="w",
                       padx=5, pady=(2, 2))
        for label, callback in (
            ("編輯目前航線", lambda: self._open_route_editor(True)),
            ("畫新航線", lambda: self._open_route_editor(False)),
            ("匯入航線 JSON", self._choose_route),
        ):
            button = ttk.Button(route_row, text=label, command=callback)
            button.pack(side="left", padx=(0, 6))
            self._buttons.append(button)

    def _build_route_choices(self, row: int) -> int:
        """One button per route this site already has. Returns the next free row.

        Applying a site re-execs the process, so the active site cannot change
        under this list -- it is built once and stays correct for the session.
        """
        if self.site_pack_root is None:
            return row
        try:
            routes = describe_site_routes(self.site_pack_root)
        except Exception as exc:
            self.status_var.set(f"航線清單讀取失敗，請用「匯入航線 JSON」：{exc}")
            return row
        if not routes:
            return row
        ttk.Label(
            self,
            text="可用航線（選定下一次 AUTO）",
            font=("Sans", 9, "bold"),
        ).grid(
            row=row, column=0, sticky="w", padx=(8, 5), pady=(2, 2)
        )
        choices = ttk.Frame(self)
        choices.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=(2, 2))
        for item in routes:
            button = ttk.Button(
                choices,
                text=item.label,
                command=lambda path=item.path: self._preview_existing_route(path),
            )
            if not item.flight_ready:
                button.state(["disabled"])
                self._permanently_disabled.add(button)
            button.pack(side="left", padx=(0, 6))
            self._buttons.append(button)
        return row + 1

    def _preview_existing_route(self, path: Path) -> None:
        """Validate and select one route for this session's next AUTO request.

        Deliberately NOT import_route: that needs a managed profile under the site
        packages root (local_site_assets._load_managed_profile) and resets flight
        approval, so it stays behind the explicit 匯入航線 JSON action. Flipping
        routes changes the immutable session selection only while AUTO is inactive;
        it does not approve, arm, take off, or persist a different site profile.
        """
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
        self._pending_site_folder = Path(folder)
        self._run("場域匯入", lambda: self.actions.import_site_folder(str(folder)))

    def _build(self) -> None:
        # Only the site switch remains (operator decision 2026-08-06). Everything the
        # removed rows did is now automatic: switching a site detects its route and,
        # when there is none, goes straight into marking one. The route-editor and
        # target-import methods are kept -- that automatic flow calls them.
        self._build_site_shortcuts(0)
        status_row = self._build_route_choices(2)
        ttk.Label(self, textvariable=self.status_var, wraplength=1050).grid(
            row=status_row, column=0, columnspan=3, sticky="w", padx=8, pady=(5, 7)
        )
        # The standalone 重啟套用 row is gone (operator decision 2026-08-06). Applying
        # is what switching a site MEANS, so it is offered by the switch itself --
        # removing the row without that would leave an imported site never active.
        self.columnconfigure(1, weight=1)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in self._buttons:
            if not busy and button in self._permanently_disabled:
                continue          # unusable package: never becomes clickable
            button.configure(state=state)

    def _run(self, label: str, action: Callable[[], ActionResult]) -> None:
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
        self._set_busy(False)
        self.status_var.set(f"{label}失敗：{exc}")

    def _finish_ok(self, result: ActionResult) -> None:
        self._set_busy(False)
        self.status_var.set(result.message)
        if result.kind == "route" and self.route_imported is not None:
            self.route_imported(result)
        elif result.kind == "site":
            folder, self._pending_site_folder = self._pending_site_folder, None
            outcome = "declined"
            if folder is not None:
                outcome = self.follow_up_route_for_site(folder)
            # Applying restarts the interface. Never do that out from under an
            # editor the operator was just sent into to draw a route.
            if outcome != "drawing":
                self.offer_apply_after_import()

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

        slot = package.asset("route_json")
        if slot is not None and slot.found:
            # route_drafts/ holds unapproved editor output. Auto-binding one as the
            # site's flight route would promote a draft nobody confirmed.
            approved = [
                path for path in slot.candidates if "route_drafts" not in path.parts
            ]
            if not approved:
                self.status_var.set(
                    f"只找到 {len(slot.candidates)} 份航線草稿（route_drafts/），"
                    "不會自動匯入；請用「匯入航線 JSON」確認要哪一份"
                )
                return "drafts_only"
            if len(approved) != len(slot.candidates):
                slot = _RouteSlot(approved)
        if slot is None or not slot.found:
            if self.ask_yes_no(
                "這個場域還沒有航線",
                f"在 {folder.name} 找不到航線檔。要現在開啟編輯器標路徑點嗎？",
            ):
                self._open_route_editor(False)
                return "drawing"
            self.status_var.set("場域已匯入；尚無航線，之後可從「畫新航線」開始")
            return "declined"

        if slot.ambiguous:
            # Which route the aircraft flies is not a guess worth making.
            self.status_var.set(
                f"偵測到 {len(slot.candidates)} 個航線檔，請用「匯入航線 JSON」明確選擇"
            )
            return "ambiguous"

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

    def _choose_route(self) -> None:
        # Open where this field's routes and drafts actually are, so picking among
        # several is one click rather than a hunt through the filesystem.
        initial = ""
        # The active site's own folder when we know it. current_profile is the path
        # to site_profile.json, not a SiteProfile (.source raised AttributeError and
        # took the whole button down), and its parent is site_profiles/ rather than
        # the site pack whenever the app was started from a system profile.
        profile = self.actions.current_profile
        search_root = self.site_pack_root or (
            Path(profile).parent if profile is not None else None
        )
        if search_root is not None:
            routes = list_site_routes(search_root)
            if routes:
                initial = str(routes[0].parent)
        source = filedialog.askopenfilename(
            title="選擇要匯入的航線 JSON（新到舊）",
            initialdir=initial or None,
            filetypes=(("JSON", "*.json"),),
        )
        if source:
            self._run("航線匯入", lambda: self.actions.import_route(source))

    def _open_route_editor(self, edit_current: bool) -> None:
        if self.request_route_editor is None:
            self.status_var.set("航線編輯器尚未接上")
            return
        try:
            self.request_route_editor(bool(edit_current))
        except Exception as exc:
            self.status_var.set(f"開啟航線編輯器失敗：{exc}")

    def import_authored_route(self, source: str | Path) -> ActionResult:
        """Import an editor-produced file through the same validated route port."""
        result = self.actions.import_route(source)
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
        """Ask whether to restart into the site just imported. True if applying."""
        if self.actions.current_profile is None:
            return False
        if not self.ask_yes_no(
            "切換到這個場域",
            "匯入完成。要立即重啟介面套用這個場域嗎？（僅限已落地）",
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
