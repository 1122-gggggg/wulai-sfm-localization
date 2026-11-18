from __future__ import annotations

from dataclasses import replace
import queue
import threading

import pytest

from backend_contract import (
    ControlAction, ControlRequest, ControlResult, execute_admitted_control,
)
from operator_command_coordinator import OperatorCommandCoordinator, is_safety_request
from operator_shutdown import OperatorShutdownCoordinator


_WAIT = 2.0


def _coordinator(backend, *, capacity: int = 2, write_log=None):
    normal = queue.Queue(maxsize=capacity)
    safety = queue.Queue()
    inflight: set[str] = set()
    logs: list[str] = []
    drops: list[tuple[str, str]] = []
    coordinator = OperatorCommandCoordinator(
        backend=backend,
        normal_results=normal,
        safety_results=safety,
        inflight=inflight,
        inflight_lock=threading.Lock(),
        publish_lock=threading.Lock(),
        write_log=write_log if write_log is not None else logs.append,
        record_drop=lambda command, dropped: drops.append((command, dropped)),
        safety_commands=frozenset({"land", "land_now", "emergency_stop"}),
    )
    return coordinator, normal, safety, inflight, logs, drops


class _Backend:
    def __init__(self, result=None):
        self.requests: list[ControlRequest] = []
        self.result = result if result is not None else ControlResult.completed(
            object(), raw_result=True
        )

    def command(self, request: ControlRequest) -> ControlResult:
        assert isinstance(request, ControlRequest)
        self.requests.append(request)
        return self.result


@pytest.fixture
def workers(monkeypatch):
    real_thread = threading.Thread
    started = []

    def create_thread(*args, **kwargs):
        worker = real_thread(*args, **kwargs)
        started.append(worker)
        return worker

    monkeypatch.setattr("operator_command_coordinator.threading.Thread", create_thread)
    yield started
    for worker in started:
        if worker.ident is not None:
            worker.join(_WAIT)
            assert not worker.is_alive()


def _join(workers):
    for worker in workers:
        worker.join(_WAIT)
        assert not worker.is_alive()


def test_typed_execution_does_not_trust_backend_mode_labels() -> None:
    backend = _Backend()
    backend.mode = "real-flight"
    coordinator, *_ = _coordinator(backend)

    assert coordinator.execute("resume_pc", {}) is True
    assert backend.requests[0].action is ControlAction.PC_CONTROL
    assert backend.requests[0].human_origin is True
    assert backend.requests[0].admission is not None


@pytest.mark.parametrize("result", [None, True, False, "accepted", object()])
def test_untyped_backend_results_fail_closed(result) -> None:
    backend = _Backend()
    backend.result = result
    coordinator, *_ = _coordinator(backend)

    assert coordinator.execute("pc_control", {}) is False
    assert not coordinator.has_inflight


@pytest.mark.parametrize("changes", [
    {"accepted": False},
    {"accepted": 1},
    {"executed": False},
    {"executed": 1},
    {"ack": {}},
    {"ack": {"accepted": False}},
    {"ack": {"accepted": 1}},
    {"raw_result": False},
    {"raw_result": "accepted"},
    {"reason_code": ""},
])
def test_invalid_or_unconfirmed_typed_results_fail_closed(changes) -> None:
    result = replace(ControlResult.completed(object(), raw_result=True), **changes)
    coordinator, *_ = _coordinator(_Backend(result))

    assert coordinator.execute("pc_control", {}) is False
    assert not coordinator.has_inflight


def test_unknown_action_is_rejected_without_calling_backend() -> None:
    backend = _Backend()
    coordinator, *_ = _coordinator(backend)

    assert coordinator.execute("unknown", {}) is False
    assert coordinator.dispatch("unknown", {}) is False
    assert backend.requests == []


def test_result_backpressure_never_evicts_safety_completion() -> None:
    coordinator, normal, safety, _inflight, _logs, drops = _coordinator(
        _Backend(), capacity=1
    )

    coordinator.publish(("hover", "old", None))
    coordinator.publish(("manual", "new", None))
    coordinator.publish(("land", "landed", None))

    assert normal.get_nowait()[0] == "manual"
    assert safety.get_nowait()[0] == "land"
    assert drops == [("manual", "hover")]


@pytest.mark.parametrize("first,first_payload,second,second_payload", [
    ("takeoff", {}, "takeoff", {}),
    ("pc_control", {}, "resume_pc", {}),
    ("nudge_begin", {"dir": "前"}, "nudge_press", {"dir": "後"}),
    ("nudge_begin", {"dir": "前"}, "nudge_vector", {"pitch": -1.0}),
    ("takeoff", {}, "firmware_limits_apply", {
        "max_altitude_m": 3.0, "max_distance_m": 10.0,
    }),
    ("record_start", {}, "record_stop", {}),
    ("gimbal_pitch", {"pitch": -20.0}, "camera_reset", {}),
    ("camera_reset", {}, "reset_camera", {}),
])
def test_conflicting_commands_and_aliases_never_overlap(
    first, first_payload, second, second_payload, workers,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Backend(_Backend):
        def command(self, request):
            self.requests.append(request)
            entered.set()
            assert release.wait(_WAIT)
            return self.result

    backend = Backend()
    coordinator, normal, _safety, inflight, *_ = _coordinator(backend)
    try:
        assert coordinator.dispatch(first, first_payload) is True
        assert entered.wait(_WAIT)
        assert coordinator.dispatch(second, second_payload) is False
        assert coordinator.execute(second, second_payload) is False
        assert inflight == {first}
        assert len(backend.requests) == 1
    finally:
        release.set()
    _join(workers)
    assert normal.get_nowait() == (first, True, None)
    assert not coordinator.has_inflight
    assert inflight == set()


def test_independent_camera_and_authority_resources_remain_concurrent(workers) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Backend(_Backend):
        def command(self, request):
            self.requests.append(request)
            if request.action is ControlAction.PC_CONTROL:
                entered.set()
                assert release.wait(_WAIT)
            return self.result

    coordinator, *_ = _coordinator(Backend())
    try:
        assert coordinator.dispatch("pc_control", {}) is True
        assert entered.wait(_WAIT)
        assert coordinator.execute("gimbal_pitch", {"pitch": -20.0}) is True
    finally:
        release.set()
    _join(workers)


def test_suspend_fences_sync_and_async_admission_but_not_safety(workers) -> None:
    backend = _Backend()
    coordinator, normal, *_ = _coordinator(backend)

    coordinator.suspend()
    assert coordinator.dispatch("takeoff", {}) is False
    assert coordinator.execute("pc_control", {}) is False
    assert coordinator.execute("manual", {}) is True
    coordinator.resume()
    assert coordinator.dispatch("pc_control", {}) is True
    _join(workers)
    assert normal.get_nowait() == ("pc_control", True, None)
    assert [request.action for request in backend.requests] == [
        ControlAction.MANUAL, ControlAction.PC_CONTROL,
    ]


@pytest.mark.parametrize("synchronous", [False, True], ids=["dispatch", "execute"])
def test_suspend_invalidates_reserved_work_before_invocation(
    monkeypatch, workers, synchronous,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    backend = _Backend()
    coordinator, normal, _safety, inflight, *_ = _coordinator(backend)
    invoke = coordinator._invoke
    results = []

    def delayed(reservation):
        entered.set()
        assert release.wait(_WAIT)
        return invoke(reservation)

    monkeypatch.setattr(coordinator, "_invoke", delayed)
    try:
        if synchronous:
            threading.Thread(target=lambda: results.append(
                coordinator.execute("pc_control", {})
            )).start()
        else:
            assert coordinator.dispatch("pc_control", {}) is True
        assert entered.wait(_WAIT)
        coordinator.suspend()
        coordinator.resume()
    finally:
        release.set()
    _join(workers)
    assert backend.requests == []
    assert normal.empty()
    assert inflight == set()
    if synchronous:
        assert results == [False]
    monkeypatch.setattr(coordinator, "_invoke", invoke)
    assert coordinator.execute("pc_control", {}) is True


@pytest.mark.parametrize("stop", ["suspend", "manual", "hover", "emergency_stop"])
def test_adapter_entry_guard_closes_check_to_call_race(stop, workers) -> None:
    before_adapter_entry = threading.Event()
    enter_adapter = threading.Event()
    committed = []
    adapter_results = []

    class Backend(_Backend):
        def command(self, request):
            if request.action is ControlAction.PC_CONTROL:
                before_adapter_entry.set()
                assert enter_adapter.wait(_WAIT)
            result = execute_admitted_control(request, object(), self.apply)
            if request.action is ControlAction.PC_CONTROL:
                adapter_results.append(result)
            return result

        def apply(self, request):
            committed.append(request.action)
            return self.result

    coordinator, normal, _safety, inflight, *_ = _coordinator(Backend())
    try:
        assert coordinator.dispatch("pc_control", {}) is True
        assert before_adapter_entry.wait(_WAIT)
        if stop == "suspend":
            coordinator.suspend()
        else:
            assert coordinator.execute(stop, {}) is True
    finally:
        enter_adapter.set()
    _join(workers)
    assert ControlAction.PC_CONTROL not in committed
    assert len(adapter_results) == 1
    assert adapter_results[0].reason_code == "CONTROL_REQUEST_CANCELLED"
    assert adapter_results[0].executed is False
    assert normal.empty()
    assert inflight == set()


@pytest.mark.parametrize("stop,payload", [
    ("manual", {}), ("hover", {}), ("land", {}), ("land_now", {}),
    ("emergency_stop", {}), ("nudge_clear", {}),
    ("nudge_end", {"dir": "前"}), ("nudge_vector", {}),
])
def test_safety_overtakes_blocked_normal_and_fences_late_success(
    stop, payload, workers,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    safety_entered = threading.Event()

    class Backend(_Backend):
        def command(self, request):
            self.requests.append(request)
            if request.action is ControlAction.PC_CONTROL:
                entered.set()
                assert release.wait(_WAIT)
            else:
                safety_entered.set()
            # Deliberately return old success: the coordinator must not trust it.
            return self.result

    coordinator, normal, safety, inflight, *_ = _coordinator(Backend())
    try:
        assert coordinator.dispatch("pc_control", {}) is True
        assert entered.wait(_WAIT)
        assert coordinator.dispatch(stop, payload) is True
        assert safety_entered.wait(_WAIT)
        # The still-running old reservation must not be reused after preemption.
        assert coordinator.dispatch("resume_pc", {}) is False
    finally:
        release.set()
    _join(workers)
    receipts = []
    coordinator.drain(lambda *item: receipts.append(item))
    assert receipts == [(stop, True, None)]
    assert normal.empty()
    assert safety.empty()
    assert inflight == set()


def test_zero_vector_can_preempt_blocked_nonzero_vector_with_same_name(workers) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Backend(_Backend):
        def command(self, request):
            if request.payload.pitch:
                entered.set()
                assert release.wait(_WAIT)
            return self.result

    coordinator, normal, *_ = _coordinator(Backend())
    try:
        assert coordinator.dispatch("nudge_vector", {"pitch": 1.0}) is True
        assert entered.wait(_WAIT)
        assert coordinator.execute("nudge_vector", {}) is True
        assert coordinator.has_inflight
    finally:
        release.set()
    _join(workers)
    assert normal.empty()
    assert not coordinator.has_inflight


def test_land_alias_is_deduplicated_but_emergency_is_not_blocked(workers) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Backend(_Backend):
        def command(self, request):
            if request.action is ControlAction.LAND:
                entered.set()
                assert release.wait(_WAIT)
            return self.result

    coordinator, _normal, safety, *_ = _coordinator(Backend())
    try:
        assert coordinator.dispatch("land", {}) is True
        assert entered.wait(_WAIT)
        assert coordinator.dispatch("land_now", {}) is False
        assert coordinator.execute("emergency_stop", {}) is True
    finally:
        release.set()
    _join(workers)
    assert safety.get_nowait() == ("land", True, None)


def test_safety_invalidates_already_queued_ordinary_success(workers) -> None:
    coordinator, normal, *_ = _coordinator(_Backend())
    assert coordinator.dispatch("pc_control", {}) is True
    _join(workers)
    assert not normal.empty()
    assert coordinator.execute("manual", {}) is True
    received = []
    assert coordinator.drain(lambda *item: received.append(item)) == 0
    assert received == []


def test_suspend_preserves_safety_receipts_while_invalidating_ordinary(workers) -> None:
    coordinator, normal, safety, *_ = _coordinator(_Backend())
    assert coordinator.dispatch("land", {}) is True
    _join(workers)
    coordinator.publish(("pc_control", True, None))
    coordinator.suspend()
    coordinator.resume()
    assert normal.empty()
    assert safety.get_nowait() == ("land", True, None)


@pytest.mark.parametrize("stop", ["suspend", "manual", "emergency_stop"])
def test_inflight_synchronous_success_is_fenced_by_stop(stop, workers) -> None:
    entered = threading.Event()
    release = threading.Event()
    returned = []

    class Backend(_Backend):
        def command(self, request):
            if request.action is ControlAction.PC_CONTROL:
                entered.set()
                assert release.wait(_WAIT)
            return self.result

    coordinator, *_ = _coordinator(Backend())
    try:
        threading.Thread(target=lambda: returned.append(
            coordinator.execute("pc_control", {})
        )).start()
        assert entered.wait(_WAIT)
        if stop == "suspend":
            coordinator.suspend()
            coordinator.resume()
        else:
            assert coordinator.execute(stop, {}) is True
    finally:
        release.set()
    _join(workers)
    assert returned == [False]
    assert not coordinator.has_inflight


@pytest.mark.parametrize("failure", [
    RuntimeError("backend failed"), TimeoutError("backend timed out"),
])
def test_async_backend_failure_is_published_and_clears_inflight(failure, workers) -> None:
    class Backend(_Backend):
        def command(self, request):
            raise failure

    coordinator, normal, _safety, inflight, *_ = _coordinator(Backend())
    assert coordinator.dispatch("pc_control", {}) is True
    _join(workers)
    assert normal.get_nowait() == ("pc_control", None, repr(failure))
    assert inflight == set()
    assert not coordinator.has_inflight


def test_explicit_async_failure_never_becomes_success(workers) -> None:
    coordinator, normal, *_ = _coordinator(_Backend(
        ControlResult.rejected("SOURCE_UNCONFIRMED", object())
    ))
    assert coordinator.dispatch("pc_control", {}) is True
    _join(workers)
    assert normal.get_nowait() == ("pc_control", False, None)
    assert not coordinator.has_inflight


def test_sync_exception_propagates_and_releases_reservation() -> None:
    class Backend(_Backend):
        def command(self, request):
            raise RuntimeError("SDK failed")

    coordinator, *_ = _coordinator(Backend())
    with pytest.raises(RuntimeError, match="SDK failed"):
        coordinator.execute("pc_control", {})
    assert not coordinator.has_inflight


@pytest.mark.parametrize("during_construction", [False, True])
def test_thread_start_failure_propagates_and_releases_reservation(
    monkeypatch, during_construction,
) -> None:
    backend = _Backend()
    coordinator, normal, _safety, inflight, *_ = _coordinator(backend)

    class CannotStart:
        def __init__(self, **kwargs):
            if during_construction:
                raise RuntimeError("cannot start thread")

        def start(self):
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr("operator_command_coordinator.threading.Thread", CannotStart)
    with pytest.raises(RuntimeError, match="cannot start thread"):
        coordinator.dispatch("pc_control", {})
    assert not coordinator.has_inflight
    assert inflight == set()
    assert normal.empty()
    assert backend.requests == []
    assert coordinator.execute("resume_pc", {}) is True


@pytest.mark.parametrize("command,payload,expected", [
    ("nudge_release", {"dir": "前"}, True),
    ("nudge_clear", {}, True),
    ("nudge_vector", {}, True),
    ("nudge_vector", {"pitch": 0.5}, False),
    ("manual", {}, True),
    ("land_now", {}, True),
    ("resume_pc", {}, False),
])
def test_runtime_transition_uses_same_typed_safety_classifier(command, payload, expected):
    request = ControlRequest.from_legacy(command, human_origin=True, **payload)
    assert is_safety_request(request) is expected


def test_worker_logging_never_calls_ui_until_drain(workers) -> None:
    ui_thread = threading.get_ident()
    logs = []

    def write_log(message):
        assert threading.get_ident() == ui_thread
        logs.append(message)

    coordinator, *_ = _coordinator(_Backend(
        ControlResult.rejected("EXPLICIT_FAILURE", object())
    ), write_log=write_log)
    assert coordinator.dispatch("pc_control", {}) is True
    _join(workers)
    assert not any("EXPLICIT_FAILURE" in message for message in logs)
    coordinator.drain(lambda *_item: None)
    assert any("EXPLICIT_FAILURE" in message for message in logs)


def test_ui_handler_may_reenter_coordinator_without_locking(workers) -> None:
    coordinator, *_ = _coordinator(_Backend())
    assert coordinator.dispatch("pc_control", {}) is True
    _join(workers)
    results = []
    assert coordinator.drain(lambda *_item: results.append(
        coordinator.execute("manual", {})
    )) == 1
    assert results == [True]


def test_failed_shutdown_resumes_commands_but_success_keeps_them_suspended() -> None:
    class Commands:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def suspend(self) -> None:
            self.calls.append("suspend")

        def resume(self) -> None:
            self.calls.append("resume")

    class Backend:
        is_live = True

        def __init__(self) -> None:
            self.results = iter((False, True))

        def cleanup(self) -> bool:
            return next(self.results)

    commands = Commands()
    shutdown = OperatorShutdownCoordinator(
        backend=Backend(), session_logs=None, write_log=None, destroy=None,
        command_coordinator=commands,
    )

    assert shutdown.shutdown() is False
    assert commands.calls == ["suspend", "resume"]
    assert shutdown.shutdown() is True
    assert commands.calls == ["suspend", "resume", "suspend"]
