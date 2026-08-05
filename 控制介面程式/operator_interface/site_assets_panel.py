"""Tk layout for the three explicit site-asset import interfaces."""
from __future__ import annotations

import threading
import queue
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, ttk

from operator_actions import ActionResult, SiteAssetActions


class SiteAssetsPanel(ttk.LabelFrame):
    """View only: file pickers delegate all validation and writes to actions."""

    def __init__(
        self,
        parent,
        *,
        actions: SiteAssetActions,
        request_apply: Callable[[Path], None],
    ):
        super().__init__(parent, text="場域資產（匯入不會起飛，也不會解鎖自主飛行）")
        self.actions = actions
        self.request_apply = request_apply
        self.status_var = tk.StringVar(value="① 請先匯入建圖端輸出資料夾")
        self._buttons: list[ttk.Button] = []
        self._result_queue: queue.Queue[tuple[ActionResult | None, Exception | None]] = (
            queue.Queue(maxsize=1)
        )
        self._build()

    def _build(self) -> None:
        rows = (
            (
                "① 場域建圖資料夾（必需）",
                "site_profile.json + PLY + EDM bundle + runtime profile + reference poses",
                "選擇資料夾並驗證匯入",
                self._choose_site,
            ),
            (
                "② 預畫航線 JSON（有航線才需要）",
                "獨立匯入，必須與 site_id / coordinate_frame_id 一致",
                "匯入航線",
                self._choose_route,
            ),
            (
                "③ 電桿／目標物 JSON（巡檢才需要）",
                "獨立匯入；目前接受 aligned 座標",
                "匯入巡檢目標",
                self._choose_targets,
            ),
        )
        for row, (title, detail, button_text, command) in enumerate(rows):
            ttk.Label(self, text=title, font=("Sans", 9, "bold")).grid(
                row=row, column=0, sticky="w", padx=(8, 5), pady=(5, 2)
            )
            ttk.Label(self, text=detail, wraplength=700).grid(
                row=row, column=1, sticky="w", padx=5, pady=(5, 2)
            )
            button = ttk.Button(self, text=button_text, command=command)
            button.grid(row=row, column=2, sticky="e", padx=8, pady=(5, 2))
            self._buttons.append(button)
        ttk.Label(self, textvariable=self.status_var, wraplength=1050).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(5, 7)
        )
        apply_button = ttk.Button(
            self,
            text="重啟套用（僅限已落地）",
            command=self._apply,
        )
        apply_button.grid(row=3, column=2, sticky="e", padx=8, pady=(5, 7))
        self._buttons.append(apply_button)
        self.columnconfigure(1, weight=1)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in self._buttons:
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

    def _choose_site(self) -> None:
        folder = filedialog.askdirectory(title="選擇包含 site_profile.json 的建圖輸出資料夾")
        if folder:
            self._run("場域匯入", lambda: self.actions.import_site_folder(folder))

    def _choose_route(self) -> None:
        source = filedialog.askopenfilename(
            title="選擇預畫航線 JSON", filetypes=(("JSON", "*.json"),)
        )
        if source:
            self._run("航線匯入", lambda: self.actions.import_route(source))

    def _choose_targets(self) -> None:
        source = filedialog.askopenfilename(
            title="選擇電桿／目標物 JSON", filetypes=(("JSON", "*.json"),)
        )
        if source:
            self._run("巡檢目標匯入", lambda: self.actions.import_targets(source))

    def _apply(self) -> None:
        profile = self.actions.current_profile
        if profile is None:
            self.status_var.set("套用失敗：請先匯入建圖端場域資料夾")
            return
        try:
            self.request_apply(profile)
        except Exception as exc:
            self.status_var.set(f"套用失敗：{exc}")
