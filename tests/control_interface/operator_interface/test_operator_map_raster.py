from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

from PIL import Image
import pytest

OPERATOR_INTERFACE = Path(__file__).resolve().parents[3] / "控制介面程式" / "operator_interface"
if str(OPERATOR_INTERFACE) not in sys.path:
    sys.path.insert(0, str(OPERATOR_INTERFACE))

from operator_map_raster import MapRasterWorker  # noqa: E402


def _wait_for(worker: MapRasterWorker, key: tuple):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        worker.poll()
        image = worker.result(key)
        if image is not None:
            return image
        time.sleep(0.002)
    raise AssertionError(f"raster for {key!r} never completed")


@pytest.fixture()
def gated_worker():
    started: list[str] = []
    first_running = threading.Event()
    release_first = threading.Event()

    def render(snapshot: str) -> Image.Image:
        started.append(snapshot)
        if len(started) == 1:
            first_running.set()
            assert release_first.wait(5)
        if snapshot == "boom":
            raise ValueError("render failed")
        return Image.new("RGB", (4, 3), "#123456")

    worker = MapRasterWorker(render)
    yield worker, started, first_running, release_first
    release_first.set()
    worker.close()


def test_superseded_views_are_skipped_while_a_render_is_running(gated_worker) -> None:
    worker, started, first_running, release_first = gated_worker
    worker.request(("a",), lambda: "a")
    assert first_running.wait(5)
    for name in ("b", "c"):
        worker.request((name,), lambda name=name: name)
    release_first.set()

    assert _wait_for(worker, ("c",)).size == (4, 3)
    assert started == ["a", "c"]
    assert worker.result(("a",)) is None
    assert worker.result(("b",)) is None


def test_clear_discards_the_result_of_a_view_that_was_already_running(gated_worker) -> None:
    worker, started, first_running, release_first = gated_worker
    worker.request(("old-site",), lambda: "old")
    assert first_running.wait(5)
    worker.clear()
    release_first.set()
    deadline = time.monotonic() + 1
    while worker._future is not None and time.monotonic() < deadline:
        worker.poll()
        time.sleep(0.002)

    assert worker.result(("old-site",)) is None
    assert worker.latest(4, 3) is None
    worker.request(("old-site",), lambda: "new")
    assert _wait_for(worker, ("old-site",)).size == (4, 3)
    assert started == ["old", "new"]


def test_latest_image_is_only_offered_for_the_same_canvas_size(gated_worker) -> None:
    worker, _started, first_running, release_first = gated_worker
    worker.request(("a",), lambda: "a")
    assert first_running.wait(5)
    release_first.set()
    _wait_for(worker, ("a",))

    assert worker.latest(4, 3) is not None
    assert worker.latest(5, 3) is None


def test_a_failed_render_is_reported_for_that_view_without_stopping_later_views() -> None:
    def render(snapshot: str) -> Image.Image:
        if snapshot == "boom":
            raise ValueError("render failed")
        return Image.new("RGB", (2, 2))

    worker = MapRasterWorker(render)
    try:
        worker.request(("bad",), lambda: "boom")
        deadline = time.monotonic() + 5
        with pytest.raises(RuntimeError, match="map raster failed") as failure:
            while time.monotonic() < deadline:
                worker.result(("bad",))
                time.sleep(0.002)
        assert isinstance(failure.value.__cause__, ValueError)
        worker.request(("good",), lambda: "ok")
        assert _wait_for(worker, ("good",)).size == (2, 2)
    finally:
        worker.close()
