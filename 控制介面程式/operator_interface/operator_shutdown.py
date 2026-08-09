"""Operator-window shutdown coordination.

The live backend keeps its connection open when touchdown cannot be confirmed.
This small coordinator makes that contract explicit at the UI boundary: only a
successful cleanup may close the session log and destroy the window.
"""
from __future__ import annotations

from typing import Any, Callable


class OperatorShutdownCoordinator:
    """Run backend cleanup before closing the operator session.

    Live cleanup is fail-closed and must return the literal ``True``.  Older
    non-live backends historically had no ``cleanup`` method or returned
    ``None``; both forms remain compatible for simulation/replay use.
    """

    def __init__(
        self,
        *,
        backend: Any,
        session_logs: Any | None,
        write_log: Callable[[str], None] | None,
        destroy: Callable[[], None] | None,
        command_coordinator: Any | None = None,
    ) -> None:
        self.backend = backend
        self.session_logs = session_logs
        self.write_log = write_log
        self.destroy = destroy
        self.command_coordinator = command_coordinator
        self._closed = False

    def _log(self, message: str) -> None:
        if self.write_log is None:
            return
        try:
            self.write_log(message)
        except Exception:
            # A broken Tk/log sink must not change the cleanup decision.
            pass

    def _cleanup_accepted(self, result: object, *, is_live: bool) -> bool:
        if is_live:
            return result is True
        return result is None or result is True

    def _suspend_commands(self) -> bool:
        suspend = getattr(self.command_coordinator, "suspend", None)
        if suspend is None:
            return True
        try:
            suspend()
        except Exception as exc:
            self._log(f"關窗已暫停：無法凍結新控制指令（{exc!r}）；請重試。")
            return False
        return True

    def _resume_commands(self) -> None:
        resume = getattr(self.command_coordinator, "resume", None)
        if resume is None:
            return
        try:
            resume()
        except Exception as exc:
            self._log(f"控制指令恢復失敗（{exc!r}）；請保持手動控制並重試關窗。")

    def _reject_cleanup(self, message: str) -> bool:
        self._log(message)
        self._resume_commands()
        return False

    def _run_cleanup(self, *, is_live: bool) -> bool:
        cleanup = getattr(self.backend, "cleanup", None)
        if cleanup is None:
            if not is_live:
                result = None
            else:
                return self._reject_cleanup(
                    "關窗已暫停：真機後端沒有可驗證的 cleanup 結果；"
                    "請確認機體已落地後重試。"
                )
        elif not callable(cleanup):
            return self._reject_cleanup(
                "關窗已暫停：後端 cleanup 介面無法執行；"
                "請確認機體狀態後重試。"
            )
        else:
            try:
                result = cleanup()
            except Exception as exc:
                return self._reject_cleanup(
                    f"關窗已暫停：後端 cleanup 發生例外（{exc!r}）；"
                    "連線與介面保留，確認落地後重試。"
                )

        if self._cleanup_accepted(result, is_live=is_live):
            return True
        detail = "未確認落地" if is_live else "後端拒絕清理"
        return self._reject_cleanup(
            f"關窗已暫停：{detail}（cleanup={result!r}）；"
            "連線與介面保留，確認落地後重試。"
        )

    def _close_session_logs(self, *, reason: str) -> bool:
        if self.session_logs is None:
            return True
        try:
            self.session_logs.close(reason=reason)
        except Exception as exc:
            self._log(f"關窗已暫停：工作階段記錄關閉失敗（{exc!r}）；請重試。")
            return False
        return True

    def _destroy_window(self) -> bool:
        if self.destroy is None:
            return True
        try:
            self.destroy()
        except Exception as exc:
            self._log(f"介面關閉失敗（{exc!r}）；請重試。")
            return False
        return True

    def shutdown(self, *, reason: str = "ui_window_close") -> bool:
        """Return whether the UI may close; failed cleanup remains retryable."""
        if self._closed:
            return True
        if not self._suspend_commands():
            return False

        is_live = bool(getattr(self.backend, "is_live", False))
        if not self._run_cleanup(is_live=is_live):
            return False

        if not self._close_session_logs(reason=reason):
            return False
        if not self._destroy_window():
            return False
        self._closed = True
        return True

    request_close = shutdown
