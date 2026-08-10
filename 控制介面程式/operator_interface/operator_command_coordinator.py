"""Typed background-command coordination for the operator interface."""
from __future__ import annotations

import queue
import threading
from typing import Callable, Iterable, Protocol

from backend_contract import ControlRequest, ControlResult, InterfaceMode


CommandCompletion = tuple[str, object | None, str | None]


class CommandBackend(Protocol):
    @property
    def mode(self) -> object: ...

    def command(
        self,
        request: ControlRequest | str,
        **payload: object,
    ) -> object: ...


class OperatorCommandCoordinator:
    """Serialize duplicate UI commands and publish bounded async completions."""

    def __init__(
        self,
        *,
        backend: CommandBackend,
        normal_results: queue.Queue[CommandCompletion],
        safety_results: queue.Queue[CommandCompletion],
        inflight: set[str],
        inflight_lock: threading.Lock,
        publish_lock: threading.Lock,
        write_log: Callable[[str], None] | None,
        record_drop: Callable[[str, str], None] | None,
        safety_commands: Iterable[str],
    ) -> None:
        self.backend = backend
        self.normal_results = normal_results
        self.safety_results = safety_results
        self.inflight = inflight
        self.inflight_lock = inflight_lock
        self.publish_lock = publish_lock
        self.write_log = write_log
        self.record_drop = record_drop
        self.safety_commands = frozenset(safety_commands)
        self._accepting = True

    def _log(self, message: str) -> None:
        if self.write_log is None:
            return
        try:
            self.write_log(message)
        except Exception:
            pass

    def suspend(self) -> None:
        with self.inflight_lock:
            self._accepting = False

    def resume(self) -> None:
        with self.inflight_lock:
            self._accepting = True

    @property
    def has_inflight(self) -> bool:
        with self.inflight_lock:
            return bool(self.inflight)

    def execute(self, command: str, payload: dict | None = None) -> object:
        """Dispatch one UI-origin command through the typed backend boundary."""
        values = dict(payload or {})
        mode = getattr(self.backend, "mode", None)
        if isinstance(mode, InterfaceMode):
            try:
                request = ControlRequest.from_legacy(
                    command,
                    human_origin=True,
                    **values,
                )
            except ValueError as exc:
                self._log(f"指令已拒絕: {exc}")
                return False
            result = self.backend.command(request)
            if isinstance(result, ControlResult):
                if not (result.accepted and result.executed):
                    self._log(f"{command}: 已拒絕 ({result.reason_code})")
                    return False
                return result.raw_result if result.raw_result is not None else True
            return result
        # Compatibility for narrow test doubles and pre-contract plugins.
        return self.backend.command(command, **values)

    def dispatch(self, command: str, payload: dict) -> bool:
        """Run one potentially blocking backend expectation off the UI thread."""
        with self.inflight_lock:
            if not self._accepting:
                self._log(f"{command}: 關閉流程進行中，拒絕新指令")
                return False
            if command in self.inflight:
                self._log(f"{command}: 指令仍在等待確認，略過重複送出")
                return False
            self.inflight.add(command)

        def run() -> None:
            result: object | None = None
            error: str | None = None
            try:
                result = self.execute(command, payload)
            except Exception as exc:
                error = repr(exc)
            finally:
                with self.inflight_lock:
                    self.inflight.discard(command)
                self.publish((command, result, error))

        threading.Thread(
            target=run,
            name=f"olympe-ui-{command}",
            daemon=True,
        ).start()
        self._log(f"{command}: 已送至背景飛控執行緒，介面保持可操作")
        return True

    def _evict_normal(self) -> str | None:
        try:
            dropped = self.normal_results.get_nowait()
        except queue.Empty:
            return None
        old_command = str(dropped[0])
        if old_command in self.safety_commands:
            self.safety_results.put(dropped)
            dropped_command = None
        else:
            dropped_command = old_command
        task_done = getattr(self.normal_results, "task_done", None)
        if callable(task_done):
            task_done()
        return dropped_command

    def _enqueue_normal(self, item: CommandCompletion) -> str | None:
        command = str(item[0])
        with self.publish_lock:
            try:
                self.normal_results.put_nowait(item)
                return None
            except queue.Full:
                pass
            dropped_command = self._evict_normal()
            try:
                self.normal_results.put_nowait(item)
            except queue.Full:
                dropped_command = command
        return dropped_command

    def _record_publish_drop(self, command: str, dropped_command: str) -> None:
        self._log(
            f"{command}: 完成佇列已滿，丟棄舊的一般結果 ({dropped_command})"
        )
        if self.record_drop is None:
            return
        try:
            self.record_drop(command, dropped_command)
        except Exception:
            pass

    def publish(self, item: CommandCompletion) -> None:
        """Publish without allowing ordinary completions to evict safety work."""
        command = str(item[0])
        if command in self.safety_commands:
            self.safety_results.put(item)
            return

        dropped_command = self._enqueue_normal(item)
        if dropped_command is None:
            return
        self._record_publish_drop(command, dropped_command)

    def drain(self, handler: Callable[[str, object | None, str | None], None]) -> int:
        """Drain safety completions first, then ordinary completions."""
        count = 0
        for result_queue in (self.safety_results, self.normal_results):
            while True:
                try:
                    command, result, error = result_queue.get_nowait()
                except queue.Empty:
                    break
                handler(command, result, error)
                count += 1
        return count


__all__ = ["CommandCompletion", "OperatorCommandCoordinator"]
