from __future__ import annotations

import queue
import threading

import pytest

from backend_contract import ControlRequest, ControlResult, InterfaceMode
from operator_command_coordinator import OperatorCommandCoordinator
from operator_shutdown import OperatorShutdownCoordinator


def _coordinator(backend, *, capacity: int = 2):
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
        write_log=logs.append,
        record_drop=lambda command, dropped: drops.append((command, dropped)),
        safety_commands=frozenset({"land", "emergency_stop"}),
    )
    return coordinator, normal, safety, inflight, logs, drops


def test_typed_execution_requires_accepted_and_executed_result() -> None:
    class Backend:
        mode = InterfaceMode.SIMULATED_STREAM

        def __init__(self) -> None:
            self.requests: list[ControlRequest] = []

        def command(self, request: ControlRequest) -> ControlResult:
            self.requests.append(request)
            return ControlResult(
                accepted=True,
                executed=False,
                reason_code="NOT_EXECUTED",
                ack={},
                readback={},
                resulting_state=object(),
            )

    backend = Backend()
    coordinator, *_ = _coordinator(backend)

    assert coordinator.execute("hover", {}) is False
    assert backend.requests[0].human_origin is True


def test_result_backpressure_never_evicts_safety_completion() -> None:
    coordinator, normal, safety, _inflight, _logs, drops = _coordinator(
        object(), capacity=1
    )

    coordinator.publish(("hover", "old", None))
    coordinator.publish(("manual", "new", None))
    coordinator.publish(("land", "landed", None))

    assert normal.get_nowait()[0] == "manual"
    assert safety.get_nowait()[0] == "land"
    assert drops == [("manual", "hover")]


def test_duplicate_async_command_is_rejected_until_completion() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Backend:
        def command(self, name: str, **payload):
            entered.set()
            assert release.wait(1.0)
            return name

    coordinator, normal, _safety, inflight, _logs, _drops = _coordinator(Backend())

    assert coordinator.dispatch("takeoff", {}) is True
    assert entered.wait(0.5)
    assert coordinator.dispatch("takeoff", {}) is False
    assert inflight == {"takeoff"}
    release.set()

    assert normal.get(timeout=1.0) == ("takeoff", "takeoff", None)


def test_suspended_coordinator_rejects_new_commands_and_can_resume() -> None:
    class Backend:
        def command(self, name: str, **payload):
            return True

    coordinator, normal, *_ = _coordinator(Backend())

    coordinator.suspend()
    assert coordinator.dispatch("manual", {}) is False
    coordinator.resume()
    assert coordinator.dispatch("manual", {}) is True
    assert normal.get(timeout=1.0) == ("manual", True, None)


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("backend failed"), TimeoutError("backend timed out")],
    ids=["exception", "timeout"],
)
def test_async_backend_failure_is_published_and_clears_inflight(failure) -> None:
    class Backend:
        def command(self, name: str, **payload):
            raise failure

    coordinator, normal, _safety, inflight, _logs, _drops = _coordinator(
        Backend()
    )

    assert coordinator.dispatch("manual", {}) is True
    assert normal.get(timeout=1.0) == ("manual", None, repr(failure))
    assert inflight == set()


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
        backend=Backend(),
        session_logs=None,
        write_log=None,
        destroy=None,
        command_coordinator=commands,
    )

    assert shutdown.shutdown() is False
    assert commands.calls == ["suspend", "resume"]
    assert shutdown.shutdown() is True
    assert commands.calls == ["suspend", "resume", "suspend"]
