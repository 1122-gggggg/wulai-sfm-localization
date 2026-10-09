from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

import flight_operator_app as app_module
import operator_launch
from flight_operator_app import OperatorApp
from operator_command_coordinator import OperatorCommandCoordinator
from operator_shutdown import OperatorShutdownCoordinator


class _UiLoop:
    """Model only Tk's main-thread callbacks; no display or hardware required."""

    def __init__(self):
        self.owner = threading.get_ident()
        self.callbacks = deque()
        self.alive = True
        self.in_worker = False
        self.destroyed = 0

    def after(self, _delay, callback):
        assert threading.get_ident() == self.owner
        assert not self.in_worker, "shutdown worker must not call Tk"
        self.callbacks.append(callback)

    def winfo_exists(self):
        assert threading.get_ident() == self.owner
        assert not self.in_worker
        return self.alive

    def destroy(self):
        assert threading.get_ident() == self.owner
        assert not self.in_worker
        self.destroyed += 1
        self.alive = False
        self.callbacks.clear()

    def tick(self):
        self.callbacks.popleft()()


class _Backend:
    is_live = True

    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0
        self.log = SimpleNamespace(event=lambda *_args, **_kwargs: None)

    def cleanup(self):
        self.calls += 1
        return next(self.results)


def _headless_operator(results):
    loop = _UiLoop()
    operator = OperatorApp.__new__(OperatorApp)
    operator.backend = _Backend(results)
    operator.after = loop.after
    operator.winfo_exists = loop.winfo_exists
    operator.destroy = loop.destroy
    operator.write_log = lambda _message: None
    operator._site_switching = False
    operator._flight_inflight_lock = threading.Lock()
    command_coordinator = OperatorCommandCoordinator(
        backend=operator.backend,
        normal_results=queue.Queue(),
        safety_results=queue.Queue(),
        inflight=set(),
        inflight_lock=operator._flight_inflight_lock,
        publish_lock=threading.Lock(),
        write_log=None,
        record_drop=None,
        safety_commands=(),
    )
    closed_logs = []
    operator._shutdown_coordinator = OperatorShutdownCoordinator(
        backend=operator.backend,
        session_logs=SimpleNamespace(close=lambda *, reason: closed_logs.append(reason)),
        write_log=None,
        destroy=None,
        command_coordinator=command_coordinator,
    )
    return operator, loop, closed_logs


def _exercise_locked_signal(signum):
    """Run in isolation so the old handler's self-deadlock is bounded."""
    operator, loop, closed_logs = _headless_operator([True])
    fallback_reasons = []

    def emergency_cleanup(reason):
        fallback_reasons.append(reason)
        return operator._shutdown_coordinator.shutdown(reason=reason)

    operator_launch._install_live_exit_safety(operator, emergency_cleanup)
    with operator._flight_inflight_lock:
        os.kill(os.getpid(), signum)
        assert operator.backend.calls == 0
        assert fallback_reasons == []
        assert loop.destroyed == 0
        assert operator.__dict__.get("_shutdown_in_progress", False) is False

    deadline = time.monotonic() + 5
    while loop.alive and time.monotonic() < deadline:
        loop.tick()
        time.sleep(0.001)
    assert operator.backend.calls == 1
    assert closed_logs == [f"signal_{signum}"]
    assert loop.destroyed == 1
    assert operator._shutdown_completed is True
    assert loop.callbacks == deque()
    assert fallback_reasons == []


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_signal_interrupting_nonreentrant_command_lock_does_not_deadlock(signal_name):
    interface = Path(__file__).resolve().parents[3] / "控制介面程式" / "operator_interface"
    test_directory = Path(__file__).resolve().parent
    child = (
        f"import sys; sys.path[:0] = [{str(interface)!r}, {str(test_directory)!r}]; "
        "import signal; "
        "from test_operator_exit_signals import _exercise_locked_signal; "
        f"_exercise_locked_signal(signal.{signal_name})"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", child],
        cwd=interface,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def controlled_exit(monkeypatch):
    operator, loop, closed_logs = _headless_operator([False, True])
    handlers = {}
    fallback_reasons = []
    atexit_callbacks = []
    workers = deque()

    class _Thread:
        def __init__(self, *, target, name, daemon):
            assert daemon is True
            self.target = target

        def start(self):
            workers.append(self.target)

    monkeypatch.setattr(app_module.threading, "Thread", _Thread)
    monkeypatch.setattr(
        app_module.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler)
    )
    monkeypatch.setattr(app_module.atexit, "register", atexit_callbacks.append)
    operator_launch._install_live_exit_safety(operator, fallback_reasons.append)

    def run_worker():
        loop.in_worker = True
        try:
            workers.popleft()()
        finally:
            loop.in_worker = False

    def finish_attempt():
        run_worker()
        for _ in range(10):
            if not operator.__dict__.get("_shutdown_in_progress", False):
                return
            loop.tick()
        pytest.fail("main-thread poll did not consume completed shutdown")

    return SimpleNamespace(
        operator=operator,
        loop=loop,
        workers=workers,
        handlers=handlers,
        fallback_reasons=fallback_reasons,
        atexit_callbacks=atexit_callbacks,
        closed_logs=closed_logs,
        finish_attempt=finish_attempt,
    )


def test_failed_landing_coalesces_signals_but_allows_a_new_retry(controlled_exit):
    state = controlled_exit
    send = lambda sig: state.handlers[sig](sig, None)
    send(signal.SIGINT)
    send(signal.SIGTERM)
    state.loop.tick()
    assert len(state.workers) == 1
    assert state.operator.backend.calls == 0

    send(signal.SIGHUP)
    state.operator._on_close()
    assert len(state.workers) == 1
    state.finish_attempt()
    assert state.operator.backend.calls == 1
    assert state.loop.alive
    assert state.closed_logs == []
    assert state.loop.destroyed == 0
    for _ in range(3):
        state.loop.tick()
    assert state.workers == deque(), "repeated signals must not auto-retry rejected landing"

    send(signal.SIGTERM)
    state.loop.tick()
    assert len(state.workers) == 1
    state.finish_attempt()
    assert state.operator.backend.calls == 2
    assert state.closed_logs == [f"signal_{signal.SIGTERM}"]
    assert state.loop.destroyed == 1
    assert state.loop.callbacks == deque()
    assert state.fallback_reasons == []
    state.atexit_callbacks[0]()
    assert state.fallback_reasons == ["atexit"]


def test_signal_pending_during_site_switch_is_delivered_after_switch(controlled_exit):
    state = controlled_exit
    state.operator._site_switching = True
    state.handlers[signal.SIGHUP](signal.SIGHUP, None)
    for _ in range(3):
        state.loop.tick()
    assert state.workers == deque()
    assert state.operator.backend.calls == 0
    state.operator._site_switching = False
    state.loop.tick()
    assert len(state.workers) == 1
    state.finish_attempt()
    assert state.operator.backend.calls == 1
    assert state.loop.alive
    assert state.closed_logs == []


def test_window_close_coalesces_an_already_pending_signal(controlled_exit):
    state = controlled_exit
    state.handlers[signal.SIGINT](signal.SIGINT, None)
    state.operator._on_close()
    state.loop.tick()
    assert len(state.workers) == 1
    state.finish_attempt()
    for _ in range(3):
        state.loop.tick()
    assert state.operator.backend.calls == 1
    assert state.workers == deque()
    assert state.loop.alive
    assert state.closed_logs == []


def test_signal_interrupting_close_dispatch_does_not_queue_a_retry(controlled_exit):
    state = controlled_exit
    state.operator.write_log = lambda _message: state.handlers[signal.SIGTERM](
        signal.SIGTERM, None
    )
    state.handlers[signal.SIGINT](signal.SIGINT, None)
    state.loop.tick()
    assert len(state.workers) == 1
    state.finish_attempt()
    for _ in range(3):
        state.loop.tick()
    assert state.operator.backend.calls == 1
    assert state.workers == deque()
    assert state.loop.alive
    assert state.closed_logs == []
