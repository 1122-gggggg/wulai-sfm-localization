"""The relocalizer handover: what the background worker is actually given.

The frozen BoQ reference bank was extracted from colour keyframes, so a grey
query replicated across three channels puts retrieval off the distribution the
bank was built on.  The fast loop holds the colour frame anyway, so it hands it
over; these tests pin that it arrives, and that it arrives as a snapshot rather
than as a view the fast loop is free to overwrite.
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
        self.calls: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.entered = threading.Event()
        self._block = block

    def localize_array(self, gray, *, color_bgr=None):
        self.entered.set()
        if self._block is not None:
            self._block.wait(timeout=5.0)
        self.calls.append(
            (
                np.array(gray, copy=True),
                None if color_bgr is None else np.array(color_bgr, copy=True),
            )
        )
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


def test_the_colour_frame_reaches_the_relocalizer(worker_factory) -> None:
    provider = RecordingProvider()
    worker = worker_factory(provider)
    gray = np.full((6, 8), 40, dtype=np.uint8)
    colour = np.zeros((6, 8, 3), dtype=np.uint8)
    colour[..., 2] = 200  # a red frame no grey replication could produce

    assert worker.submit(gray, colour, 17) is True
    _, ordinal = drain(worker)

    assert ordinal == 17
    seen_gray, seen_colour = provider.calls[0]
    assert seen_colour is not None, "retrieval ran on a grey frame"
    assert seen_colour.shape == (6, 8, 3)
    np.testing.assert_array_equal(seen_colour, colour)
    np.testing.assert_array_equal(seen_gray, gray)


def test_the_colour_frame_is_snapshotted_at_submit(worker_factory) -> None:
    """The fast loop may own the buffer it passed; the worker must not see edits."""

    release = threading.Event()
    provider = RecordingProvider(block=release)
    worker = worker_factory(provider)
    gray = np.zeros((6, 8), dtype=np.uint8)
    colour = np.full((6, 8, 3), 10, dtype=np.uint8)

    assert worker.submit(gray, colour, 0) is True
    assert provider.entered.wait(timeout=5.0), "worker never started the job"
    colour[:] = 250  # the next stream frame lands in the same buffer
    release.set()
    drain(worker)

    _, seen_colour = provider.calls[0]
    assert int(seen_colour.max()) == 10


def test_a_grey_only_submission_is_still_accepted(worker_factory) -> None:
    provider = RecordingProvider()
    worker = worker_factory(provider)

    assert worker.submit(np.zeros((6, 8), dtype=np.uint8), None, 3) is True
    drain(worker)

    assert provider.calls[0][1] is None


def test_submit_refuses_while_the_worker_is_busy(worker_factory) -> None:
    """Capacity stays exactly one job: a queued reloc would land already stale."""

    release = threading.Event()
    provider = RecordingProvider(block=release)
    worker = worker_factory(provider)
    gray = np.zeros((6, 8), dtype=np.uint8)
    colour = np.zeros((6, 8, 3), dtype=np.uint8)

    assert worker.submit(gray, colour, 0) is True
    assert provider.entered.wait(timeout=5.0)
    assert worker.busy is True
    assert worker.submit(gray, colour, 1) is False

    release.set()
    drain(worker)
    assert len(provider.calls) == 1


def test_a_closed_worker_refuses_work() -> None:
    provider = RecordingProvider()
    worker = RelocWorker(provider)
    worker.start()
    worker.close()

    assert worker.submit(np.zeros((6, 8), dtype=np.uint8), None, 0) is False
    assert provider.calls == []
