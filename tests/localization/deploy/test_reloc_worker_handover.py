"""The relocalizer handover: what the background worker is actually given.

Retrieval is given the *grey* frame, which the provider replicates to three
channels.  The frozen BoQ bank was built from colour keyframes, so that is a
real distribution mismatch, and the fast loop is holding the colour frame at
the moment it submits -- but handing it over was measured on the P173 holdout
and did not improve anything (see docs/direct_backend_ledger.md).  This file
pins the grey-only contract so the mismatch is not "fixed" again without a
gate, together with the one-job capacity rule.
"""

from __future__ import annotations

import threading
import time

# Source modules are supplied by the repository's pytest pythonpath.
import numpy as np
import pytest

from two_rate_tracker import RelocWorker


class RecordingProvider:
    """Captures what each relocalization was given, without touching a GPU."""

    def __init__(self, *, block: threading.Event | None = None) -> None:
        self.calls: list[tuple[np.ndarray, dict]] = []
        self.entered = threading.Event()
        self._block = block

    def localize_array(self, gray, **kwargs):
        self.entered.set()
        if self._block is not None:
            self._block.wait(timeout=5.0)
        self.calls.append((np.array(gray, copy=True), dict(kwargs)))
        return object()


def drain(worker: RelocWorker, *, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        delivered = worker.poll()
        if delivered is not None:
            return delivered
        time.sleep(0.005)
    raise AssertionError("relocalizer produced no result within the timeout")


@pytest.fixture
def worker_factory():
    started: list[RelocWorker] = []

    def make(provider) -> RelocWorker:
        worker = RelocWorker(provider)
        worker.start()
        started.append(worker)
        return worker

    yield make
    for worker in started:
        worker.close()


def test_the_relocalizer_is_given_the_grey_frame_only(worker_factory) -> None:
    """Grey-only is a measured decision, not an oversight: handing the colour
    frame over was tried on the P173 holdout and did not improve anything."""

    provider = RecordingProvider()
    worker = worker_factory(provider)
    gray = np.full((6, 8), 40, dtype=np.uint8)

    assert worker.submit(gray, 17) is True
    _, ordinal = drain(worker)

    assert ordinal == 17
    seen_gray, extra = provider.calls[0]
    np.testing.assert_array_equal(seen_gray, gray)
    assert extra == {}


def test_submit_refuses_while_the_worker_is_busy(worker_factory) -> None:
    """Capacity stays exactly one job: a queued reloc would land already stale."""

    release = threading.Event()
    provider = RecordingProvider(block=release)
    worker = worker_factory(provider)
    gray = np.zeros((6, 8), dtype=np.uint8)

    assert worker.submit(gray, 0) is True
    assert provider.entered.wait(timeout=5.0)
    assert worker.busy is True
    assert worker.submit(gray, 1) is False

    release.set()
    drain(worker)
    assert len(provider.calls) == 1


def test_a_closed_worker_refuses_work() -> None:
    provider = RecordingProvider()
    worker = RelocWorker(provider)
    worker.start()
    worker.close()

    assert worker.submit(np.zeros((6, 8), dtype=np.uint8), 0) is False
    assert provider.calls == []
