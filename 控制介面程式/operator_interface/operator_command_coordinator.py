"""Typed, non-queuing command arbitration for the operator interface.

Reservations cover the whole backend call, but their lock never covers SDK I/O.
Safety commands can therefore overtake blocked ordinary calls. The adapter still
owns physical cancellation through its authority/motion epochs; this boundary
retires unstarted requests and stale UI completions, not aircraft maneuvers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import queue
import threading
from typing import Callable, Iterable, Protocol

from backend_contract import (
    ControlAction, ControlAdmission, ControlRequest, ControlResult, NudgeVectorPayload,
)


CommandCompletion = tuple[str, object | None, str | None]


class CommandBackend(Protocol):
    def command(self, request: ControlRequest) -> ControlResult: ...


_FLIGHT = frozenset({"flight"})
_FLIGHT_SETTINGS = frozenset({"flight", "settings"})
_CAMERA = frozenset({"camera"})
_COMMAND_RESOURCES = {
    ControlAction.TAKEOFF: _FLIGHT | _CAMERA,
    ControlAction.LAND: _FLIGHT | _CAMERA,
    ControlAction.LAND_NOW: _FLIGHT | _CAMERA,
    ControlAction.HOVER: _FLIGHT,
    ControlAction.MANUAL: _FLIGHT,
    ControlAction.PC_CONTROL: _FLIGHT,
    ControlAction.NUDGE_BEGIN: _FLIGHT,
    ControlAction.NUDGE_HEARTBEAT: _FLIGHT,
    ControlAction.NUDGE_END: _FLIGHT,
    ControlAction.NUDGE_CLEAR: _FLIGHT,
    ControlAction.NUDGE_VECTOR: _FLIGHT,
    ControlAction.EMERGENCY_STOP: _FLIGHT,
    ControlAction.APPLY_LIMITS: _FLIGHT_SETTINGS,
    ControlAction.SET_AUTO_SPEED_LIMIT: _FLIGHT_SETTINGS,
    ControlAction.START_LOCALIZATION: _FLIGHT,
    ControlAction.START_AUTO: _FLIGHT | _CAMERA,
    ControlAction.BOOT_LOCK: _FLIGHT,
    ControlAction.GIMBAL_PITCH: _CAMERA,
    ControlAction.ZOOM: _CAMERA,
    ControlAction.CAMERA_RESET: _CAMERA,
    ControlAction.RECORD_ARM: _CAMERA,
    ControlAction.RECORD_DISARM: _CAMERA,
    ControlAction.RECORD_START: _CAMERA,
    ControlAction.RECORD_STOP: _CAMERA,
    ControlAction.RECORD_QUALITY: _CAMERA,
    ControlAction.DRONE_MAGNETOMETER_START: _FLIGHT_SETTINGS,
    ControlAction.DRONE_MAGNETOMETER_CANCEL: _FLIGHT_SETTINGS,
    ControlAction.SKYCONTROLLER_MAGNETOMETER_START: _FLIGHT_SETTINGS,
    ControlAction.SKYCONTROLLER_MAGNETOMETER_CANCEL: _FLIGHT_SETTINGS,
}
_PREEMPTING_ACTIONS = frozenset({
    ControlAction.LAND,
    ControlAction.LAND_NOW,
    ControlAction.HOVER,
    ControlAction.MANUAL,
    ControlAction.EMERGENCY_STOP,
    ControlAction.NUDGE_END,
    ControlAction.NUDGE_CLEAR,
})


def is_safety_request(request: ControlRequest) -> bool:
    """Identify existing stop/release semantics without trusting command aliases."""
    return request.action in _PREEMPTING_ACTIONS or (
        isinstance(request.payload, NudgeVectorPayload)
        and not any((request.payload.roll, request.payload.pitch,
                     request.payload.yaw, request.payload.gaz))
    )


@dataclass(frozen=True)
class _Reservation:
    command: str
    request: ControlRequest
    family: ControlAction
    resources: frozenset[str]
    safety: bool


class OperatorCommandCoordinator:
    """Reject conflicting work immediately; never queue a stored motion command.

    Each canonical action family has at most one ordinary and one safety
    reservation (so a zero vector can preempt a blocked nonzero vector). Ordinary
    actions sharing a resource cannot overlap. Distinct safety actions retain the
    backend's existing policy instead of acquiring a new UI priority ordering.
    Safety receipts are preserved even when a later safety action supersedes an
    earlier ordinary request. Suspend also fences synchronous execute().

    Worker log messages are bounded and drained by the UI, just like results;
    workers must never call Tk (including Tk.after) while the UI is shutting down.
    """

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
        self._reservations: dict[tuple[bool, ControlAction], _Reservation] = {}
        self._ui_thread = threading.get_ident()
        self._notice_lock = threading.Lock()
        self._notices: deque[tuple[str, str | None]] = deque(maxlen=128)

    def _log(self, message: str) -> None:
        if self.write_log is None:
            return
        if threading.get_ident() == self._ui_thread:
            self.write_log(message)
        else:
            with self._notice_lock:
                self._notices.append((message, None))

    def _drain_notices(self) -> None:
        with self._notice_lock:
            notices = tuple(self._notices)
            self._notices.clear()
        for message, dropped in notices:
            if dropped is None:
                if self.write_log is not None:
                    self.write_log(message)
            elif self.record_drop is not None:
                self.record_drop(message, dropped)

    def _invalidate_ordinary_locked(self) -> None:
        for reservation in self._reservations.values():
            admission = reservation.request.admission
            if admission is not None:
                admission.cancel()
        # A completion already queued before a stop must not restore old UI
        # authority after the stop's receipt. No user callback runs under locks.
        with self.publish_lock:
            while True:
                try:
                    item = self.normal_results.get_nowait()
                except queue.Empty:
                    break
                if item[0] in self.safety_commands:
                    self.safety_results.put_nowait(item)
                self.normal_results.task_done()

    def suspend(self) -> None:
        with self.inflight_lock:
            self._accepting = False
            self._invalidate_ordinary_locked()

    def resume(self) -> None:
        with self.inflight_lock:
            self._accepting = True

    @property
    def has_inflight(self) -> bool:
        with self.inflight_lock:
            return bool(self._reservations)

    def _reserve(self, command: str, payload: dict | None) -> _Reservation | None:
        try:
            request = ControlRequest.from_legacy(
                command, human_origin=True, **dict(payload or {})
            )
        except ValueError as exc:
            self._log(f"指令已拒絕: {exc}")
            return None
        resources = _COMMAND_RESOURCES.get(request.action)
        if resources is None:
            self._log(f"{command}: 指令沒有明確資源宣告，已拒絕")
            return None
        family = (
            ControlAction.LAND
            if request.action is ControlAction.LAND_NOW
            else request.action
        )
        safety = is_safety_request(request)
        if not safety:
            request = replace(request, admission=ControlAdmission())
        key = (safety, family)
        rejection = None
        with self.inflight_lock:
            if not self._accepting and not safety:
                rejection = "關閉流程進行中，拒絕新指令"
            elif key in self._reservations:
                rejection = "指令仍在等待確認，略過重複送出"
            elif not safety and any(
                resources & active.resources for active in self._reservations.values()
            ):
                rejection = "資源仍在等待確認，拒絕衝突指令"
            else:
                if safety:
                    self._invalidate_ordinary_locked()
                reservation = _Reservation(
                    command, request, family, resources, safety
                )
                self._reservations[key] = reservation
                self.inflight.add(command)
                return reservation
        self._log(f"{command}: {rejection}")
        return None

    def _current_locked(self, reservation: _Reservation) -> bool:
        return reservation.safety or (
            self._accepting
            and reservation.request.admission is not None
            and reservation.request.admission.is_current()
        )

    def _invoke(self, reservation: _Reservation) -> bool:
        with self.inflight_lock:
            if not self._current_locked(reservation):
                return False
        result = self.backend.command(reservation.request)
        if not isinstance(result, ControlResult):
            self._log(f"{reservation.command}: 已拒絕 (INVALID_CONTROL_RESULT)")
            return False
        if not (
            result.accepted is True
            and result.executed is True
            and result.ack.get("accepted") is True
            and (result.raw_result is None or result.raw_result is True)
            and isinstance(result.reason_code, str)
            and bool(result.reason_code.strip())
        ):
            self._log(f"{reservation.command}: 已拒絕 ({result.reason_code})")
            return False
        return True

    def _release_locked(self, reservation: _Reservation) -> None:
        key = (reservation.safety, reservation.family)
        if self._reservations.get(key) is reservation:
            del self._reservations[key]
            if not any(
                active.command == reservation.command
                for active in self._reservations.values()
            ):
                self.inflight.discard(reservation.command)

    def execute(self, command: str, payload: dict | None = None) -> bool:
        """Execute synchronously under the same admission rules as dispatch()."""
        reservation = self._reserve(command, payload)
        if reservation is None:
            return False
        try:
            result = self._invoke(reservation)
        except BaseException:
            with self.inflight_lock:
                self._release_locked(reservation)
            raise
        with self.inflight_lock:
            self._release_locked(reservation)
            return result if self._current_locked(reservation) else False

    def dispatch(self, command: str, payload: dict) -> bool:
        """Reserve immediately and start bounded work without waiting for SDK I/O."""
        reservation = self._reserve(command, payload)
        if reservation is None:
            return False

        def run() -> None:
            result: object | None = None
            error: str | None = None
            dropped: str | None = None
            try:
                result = self._invoke(reservation)
            except Exception as exc:
                error = repr(exc)
            finally:
                with self.inflight_lock:
                    self._release_locked(reservation)
                    current = self._current_locked(reservation)
                    if current:
                        dropped = self._publish_locked((command, result, error))
                if error is not None and not current:
                    self._log(f"{command}: 失敗 {error}")
                if dropped is not None:
                    self._record_publish_drop(command, dropped)

        try:
            threading.Thread(
                target=run, name=f"operator-command-{reservation.family.value}",
                daemon=True,
            ).start()
        except BaseException:
            with self.inflight_lock:
                self._release_locked(reservation)
            raise
        self._log(f"{command}: 已送至背景飛控執行緒，介面保持可操作")
        return True

    def _evict_normal(self) -> str | None:
        try:
            dropped = self.normal_results.get_nowait()
        except queue.Empty:
            return None
        old_command = str(dropped[0])
        if old_command in self.safety_commands:
            self.safety_results.put_nowait(dropped)
            dropped_command = None
        else:
            dropped_command = old_command
        self.normal_results.task_done()
        return dropped_command

    def _publish_locked(self, item: CommandCompletion) -> str | None:
        command = str(item[0])
        if command in self.safety_commands:
            self.safety_results.put_nowait(item)
            return None
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
        self._log(f"{command}: 完成佇列已滿，丟棄舊的一般結果 ({dropped_command})")
        if self.record_drop is None:
            return
        if threading.get_ident() == self._ui_thread:
            self.record_drop(command, dropped_command)
        else:
            with self._notice_lock:
                self._notices.append((command, dropped_command))

    def publish(self, item: CommandCompletion) -> None:
        """Publish a receipt; dispatched work also checks its admission token."""
        with self.inflight_lock:
            if not self._accepting and item[0] not in self.safety_commands:
                return
            dropped_command = self._publish_locked(item)
        if dropped_command is not None:
            self._record_publish_drop(str(item[0]), dropped_command)

    def drain(self, handler: Callable[[str, object | None, str | None], None]) -> int:
        """Run UI notices and safety receipts first; never invoke callbacks locked."""
        self._drain_notices()
        count = 0
        for result_queue in (self.safety_results, self.normal_results):
            while True:
                with self.inflight_lock:
                    try:
                        item = result_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    handler(*item)
                    count += 1
                finally:
                    result_queue.task_done()
        return count


__all__ = ["CommandCompletion", "OperatorCommandCoordinator", "is_safety_request"]
