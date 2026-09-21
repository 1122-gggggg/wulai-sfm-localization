"""Pacing contract for pump_localization_frame (30 FPS plan step 5).

Between render ticks the 5 ms result loop owns "accept one new frame and
submit it"; the render tick keeps classification, polling, rendering, and
BOOT/LOST hold retries. No sleeps: stream readiness is scripted, not timed.
"""

from __future__ import annotations

from types import SimpleNamespace

import flight_operator_app as app
import operator_tick


class FakeLiveStream:
    """Live backend stream: only_new reads never block, None means no frame."""

    def __init__(self, frames: list) -> None:
        self._frames = list(frames)
        self.output_index = 0
        self.last_frame_name = ""
        self.last_stamp = 0.0
        self.last_timing: dict = {}
        self.calls = 0

    def next_frame(self, *, only_new: bool = True):
        self.calls += 1
        if not self._frames:
            return None
        frame = self._frames.pop(0)
        self.output_index += 1
        self.last_frame_name = f"frame_{self.output_index:06d}.jpg"
        self.last_stamp = 1000.0 + self.output_index
        self.last_timing = {}
        return frame


class FakeFileStream:
    """Replay stream: plain next_frame(), None while the decoder is empty."""

    def __init__(self) -> None:
        self.queued: list = []
        self.output_index = 0
        self.last_frame_name = ""
        self.last_stamp = 0.0
        self.last_timing: dict = {}

    def next_frame(self):
        if not self.queued:
            return None
        frame = self.queued.pop(0)
        self.output_index += 1
        self.last_frame_name = f"frame_{self.output_index:06d}.jpg"
        self.last_stamp = 2000.0 + self.output_index
        self.last_timing = {}
        return frame


class FakePumpApp:
    """Minimal surface pump_localization_frame and the tick helpers touch."""

    def __init__(
        self,
        stream,
        *,
        live: bool = True,
        boot: bool = False,
        lost: bool = False,
        localizer: bool = True,
    ) -> None:
        self.localizer = object() if localizer else None
        self.video_stream = stream
        self.inspecting = True
        self._live = live
        self._boot = boot
        self._lost = lost
        self.video_frame = None
        self.video_frame_fresh = False
        self.video_display_index = -1
        self.video_display_frame_name = ""
        self._video_frame_stamp = 0.0
        self._video_frame_timing: dict = {}
        self.stream_lost_since = None
        self.processed_frames = 0
        self._stream_frame_times: list = []
        self.stream_fps_instant = 0.0
        self.overall_fps = 0.0
        self.inspect_start = 1.0
        self.submits: list = []

    def _is_live_backend(self) -> bool:
        return self._live

    def boot_holding(self) -> bool:
        return self._boot

    def lost_holding(self) -> bool:
        return self._lost

    def submit_current_frame_for_localization(self) -> None:
        self.submits.append(self.video_display_frame_name)


def test_pump_submits_each_new_frame_once_without_tick() -> None:
    stream = FakeLiveStream(["f1", "f2", "f3", "f4", "f5"])
    operator = FakePumpApp(stream)
    for _ in range(10):
        operator_tick.pump_localization_frame(operator)
    assert operator.submits == [f"frame_{i:06d}.jpg" for i in range(1, 6)]
    assert stream.calls == 10
    assert operator.video_display_frame_name == "frame_000005.jpg"


def test_pump_on_empty_stream_changes_nothing() -> None:
    stream = FakeLiveStream(["f1"])
    operator = FakePumpApp(stream)
    operator_tick.pump_localization_frame(operator)
    assert operator.video_frame_fresh is True
    index, stamp = operator.video_display_index, operator._video_frame_stamp
    operator_tick.pump_localization_frame(operator)
    assert operator.submits != [] and len(operator.submits) == 1
    assert operator.video_display_index == index
    assert operator._video_frame_stamp == stamp
    assert operator.video_frame_fresh is True


def test_pump_never_reads_during_boot_or_lost_hold() -> None:
    for hold in ("boot", "lost"):
        stream = FakeLiveStream(["f1"])
        kwargs = {hold: True}
        operator = FakePumpApp(stream, **kwargs)
        operator_tick.pump_localization_frame(operator)
        assert stream.calls == 0
        assert operator.submits == []


def test_fresh_latch_survives_empty_pump_until_state_consumes_it() -> None:
    stream = FakeLiveStream(["f1"])
    operator = FakePumpApp(stream)
    operator_tick.pump_localization_frame(operator)
    assert operator.video_frame_fresh is True
    operator_tick.pump_localization_frame(operator)
    assert operator.video_frame_fresh is True

    operator.live_new_pose = True
    operator.history = []
    operator.history_health = []
    operator.loc_health = "OK"
    operator.submit_current_frame_for_detection = lambda: None
    operator.state_from_live = lambda state: state
    operator.update_anafi_metrics = lambda state: None
    operator._update_age_readout = lambda state: None
    state = SimpleNamespace(pose=SimpleNamespace(copy=lambda: "pose"))
    operator_tick._update_state_and_history(operator, state)
    assert operator.video_frame_fresh is False
    assert operator.live_new_pose is False


def test_update_stream_still_clears_fresh_without_localizer() -> None:
    operator = SimpleNamespace(
        localizer=None,
        video_stream=None,
        video_frame=object(),
        video_frame_fresh=True,
    )
    state = SimpleNamespace()
    operator_tick._update_stream(
        operator, state, rolling_event_fps=operator_tick._rolling_event_fps
    )
    assert operator.video_frame_fresh is False


def test_poll_order_is_drain_update_then_pump_and_reschedules_5ms(monkeypatch) -> None:
    order: list = []
    scheduled: list = []

    class FakeOperator:
        _loc_result_poll_ms = 5
        localizer = SimpleNamespace(drain_result_notifications=lambda: order.append("drain"))
        poll_localization_results = app.OperatorApp.poll_localization_results

        def update_live_results(self) -> None:
            order.append("update")

        def after(self, delay, callback) -> None:
            scheduled.append((delay, callback))

    monkeypatch.setattr(app, "pump_localization_frame", lambda self: order.append("pump"))
    operator = FakeOperator()
    operator.poll_localization_results()
    assert order == ["drain", "update", "pump"]
    assert len(scheduled) == 1
    assert scheduled[0][0] == 5
    assert scheduled[0][1] == operator.poll_localization_results


def test_attach_handler_keeps_result_loop_at_5ms() -> None:
    seen: list = []

    class FakeOperator:
        _loc_result_poll_ms = 5

        def __init__(self) -> None:
            self.localizer = SimpleNamespace(result_notify_fd=99)

        def createfilehandler(self, fd, mask, callback) -> None:
            seen.append((fd, callback))

        def _on_localizer_result_ready(self, _fd: int, _mask: int) -> None:
            return None

    operator = FakeOperator()
    app.OperatorApp._attach_localizer_file_handler(operator)
    assert operator._loc_file_handler_registered is True
    assert operator._loc_result_poll_ms == 5
    assert [fd for fd, _ in seen] == [99]


def test_result_notification_runs_before_idle_work_and_coalesces() -> None:
    timers, idle, consumed = [], [], []
    operator = SimpleNamespace(
        localizer=SimpleNamespace(drain_result_notifications=lambda: None),
        after=lambda delay, callback: timers.append((delay, callback)),
        after_idle=idle.append,
        update_live_results=lambda: consumed.append(True),
    )
    for _ in range(3):
        app.OperatorApp._on_localizer_result_ready(operator, 99, 0)
    assert consumed == []  # No mutation inside the readable-fd callback.
    assert len(timers) == 1 and timers[0][0] == 0
    assert idle == []
    timers[0][1]()
    assert consumed == [True]
    assert operator._loc_result_idle_pending is False


def test_queued_result_notification_cannot_cross_localizer_replacement() -> None:
    timers, consumed = [], []
    operator = SimpleNamespace(
        localizer=SimpleNamespace(drain_result_notifications=lambda: None),
        after=lambda delay, callback: timers.append(callback),
        update_live_results=lambda: consumed.append(True),
    )
    app.OperatorApp._on_localizer_result_ready(operator, 99, 0)
    operator.localizer = None
    assert len(timers) == 1
    timers[0]()
    assert consumed == []
    assert operator._loc_result_idle_pending is False


def test_file_deadline_waits_for_decoder_then_skips_expired_slots() -> None:
    import time

    stream = FakeFileStream()
    operator = FakePumpApp(stream, live=False)
    operator.next_stream_frame_time = time.monotonic() - 1.0
    operator.stream_period_s = 0.05
    before = operator.next_stream_frame_time

    operator_tick.pump_localization_frame(operator)
    assert operator.submits == []
    assert operator.next_stream_frame_time == before

    stream.queued.append("late-frame")
    operator_tick.pump_localization_frame(operator)
    assert operator.submits == ["frame_000001.jpg"]
    assert operator.next_stream_frame_time > time.monotonic()
