from __future__ import annotations

import threading

from operator_shutdown import OperatorShutdownCoordinator


class _SessionLogs:
    def __init__(self):
        self.reasons = []

    def close(self, *, reason):
        self.reasons.append(reason)


class _LiveBackend:
    is_live = True

    def __init__(self, *cleanup_results):
        self.cleanup_results = iter(cleanup_results)
        self.cleanup_calls = 0
        self.zero_pcmds = []
        self.land_attempts = 0

    def cleanup(self):
        self.cleanup_calls += 1
        self.zero_pcmds.append((0, 0, 0, 0))
        self.land_attempts += 1
        return next(self.cleanup_results)


class _ActiveAutonomy:
    active = True

    def __init__(self, events, *, joins=True):
        self.events = events
        self.joins = joins

    def cancel(self, reason):
        self.events.append(("cancel", reason, (0, 0, 0, 0)))

    def join(self, *, timeout):
        self.events.append(("join", timeout))
        if self.joins:
            self.active = False
        return self.joins


class _OrderedLiveBackend(_LiveBackend):
    def __init__(self, events, *cleanup_results):
        super().__init__(*cleanup_results)
        self.events = events

    def cleanup(self):
        self.events.append("cleanup")
        return super().cleanup()


def test_shutdown_waits_for_touchdown_confirmation_before_destroying() -> None:
    backend = _LiveBackend(False, True)
    logs = _SessionLogs()
    destroyed = []
    messages = []
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=logs,
        write_log=messages.append,
        destroy=lambda: destroyed.append(True),
    )

    assert coordinator.shutdown(reason="signal_SIGTERM") is False
    assert logs.reasons == []
    assert destroyed == []
    assert backend.zero_pcmds == [(0, 0, 0, 0)]
    assert backend.land_attempts == 1

    assert coordinator.shutdown(reason="signal_SIGTERM") is True
    assert logs.reasons == ["signal_SIGTERM"]
    assert destroyed == [True]
    assert backend.cleanup_calls == 2
    assert backend.zero_pcmds == [(0, 0, 0, 0)] * 2
    assert backend.land_attempts == 2


def test_shutdown_is_idempotent_after_confirmed_touchdown() -> None:
    backend = _LiveBackend(True)
    logs = _SessionLogs()
    destroyed = []
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=logs,
        write_log=None,
        destroy=lambda: destroyed.append(True),
    )

    assert coordinator.shutdown(reason="terminal_close") is True
    assert coordinator.shutdown(reason="terminal_close_retry") is True
    assert backend.cleanup_calls == 1
    assert logs.reasons == ["terminal_close"]
    assert destroyed == [True]


def test_concurrent_shutdown_requests_share_one_cleanup() -> None:
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()

    class _BlockingBackend:
        is_live = True

        def __init__(self) -> None:
            self.cleanup_calls = 0

        def cleanup(self) -> bool:
            self.cleanup_calls += 1
            cleanup_started.set()
            assert release_cleanup.wait(timeout=1.0)
            return True

    backend = _BlockingBackend()
    logs = _SessionLogs()
    destroyed = []
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=logs,
        write_log=None,
        destroy=lambda: destroyed.append(True),
    )
    results = []
    second_returned = threading.Event()

    first = threading.Thread(
        target=lambda: results.append(coordinator.shutdown(reason="window"))
    )

    def second_shutdown() -> None:
        results.append(coordinator.shutdown(reason="signal"))
        second_returned.set()

    second = threading.Thread(target=second_shutdown)
    first.start()
    assert cleanup_started.wait(timeout=1.0)
    second.start()
    assert not second_returned.wait(timeout=0.05)
    release_cleanup.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert results == [True, True]
    assert backend.cleanup_calls == 1
    assert logs.reasons == ["window"]
    assert destroyed == [True]


def test_shutdown_cancels_and_joins_active_auto_before_cleanup() -> None:
    events = []
    backend = _OrderedLiveBackend(events, True)
    autonomy = _ActiveAutonomy(events)
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=None,
        write_log=None,
        destroy=None,
        autonomy=autonomy,
    )

    assert coordinator.shutdown() is True
    assert events == [
        ("cancel", "ui_window_close", (0, 0, 0, 0)),
        ("join", 1.0),
        "cleanup",
    ]


def test_shutdown_does_not_cleanup_while_auto_worker_is_still_active() -> None:
    events = []
    backend = _OrderedLiveBackend(events, True)
    autonomy = _ActiveAutonomy(events, joins=False)
    coordinator = OperatorShutdownCoordinator(
        backend=backend,
        session_logs=None,
        write_log=None,
        destroy=None,
        autonomy=autonomy,
    )

    assert coordinator.shutdown() is False
    assert events == [
        ("cancel", "ui_window_close", (0, 0, 0, 0)),
        ("join", 1.0),
    ]
    assert backend.cleanup_calls == 0
