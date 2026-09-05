"""The IMU flight-test recording path: frames, their telemetry, and the sticks."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import imu_flight_test
import olympe_live_backend as backend_module
from imu_flight_test import ImuFlightTestRecorder, create_recorder


def _frame() -> np.ndarray:
    # Small on purpose: this exercises the encode/write path, not 720p timing.
    return np.full((8, 16, 3), 128, dtype=np.uint8)


def _timing(**overrides) -> dict:
    timing = {
        "source_frame_stamp_mono": 1000.25,
        "client_submit_mono": 1000.30,
        "hold_kind": "none",
        "fused_telemetry_mono": 1000.24,
        "fused_roll": 0.01,
        "fused_pitch": -0.02,
        "fused_yaw": 1.5,
        "fused_speed_north": 0.4,
        "fused_speed_east": -0.2,
        "fused_speed_down": 0.05,
    }
    timing.update(overrides)
    return timing


def _rows(directory) -> list[dict]:
    path = directory / "frames.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_recorder_is_off_unless_the_flight_test_env_is_set(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SFM_IMU_FLIGHT_TEST", raising=False)
    assert create_recorder(tmp_path) is None
    monkeypatch.setenv("SFM_IMU_FLIGHT_TEST", "0")
    assert create_recorder(tmp_path) is None

    monkeypatch.setenv("SFM_IMU_FLIGHT_TEST", "1")
    recorder = create_recorder(tmp_path)
    assert recorder is not None
    try:
        assert recorder.directory == tmp_path / "imu_test"
    finally:
        recorder.close()


def test_recorder_pairs_each_frame_with_the_telemetry_that_rode_with_it(tmp_path) -> None:
    # The point of recording host-side frames at all: the image and the IMU
    # sample come out of one submission, so nothing has to be time-aligned
    # afterwards the way an onboard SD recording does.
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        assert recorder.capture(7, "stream_000007", _frame(), _timing()) is True
    finally:
        recorder.close()

    rows = _rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["seq"] == 7
    assert row["file"] == "frames/000001.jpg"
    assert row["capture_stamp_mono"] == pytest.approx(1000.25)
    assert row["fused_telemetry_mono"] == pytest.approx(1000.24)
    assert row["fused_speed_north"] == pytest.approx(0.4)
    assert (tmp_path / "frames" / "000001.jpg").is_file()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["written"] == 1


def test_recorder_row_reports_missing_telemetry_rather_than_omitting_it(tmp_path) -> None:
    # A frame the aircraft gave no velocity for is a finding, not a gap: the
    # report counts those to explain a dormant ESEKF.
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        recorder.capture(1, "stream_000001", _frame(), {"source_frame_stamp_mono": 5.0})
    finally:
        recorder.close()
    row = _rows(tmp_path)[0]
    assert row["fused_speed_north"] is None
    assert row["fused_telemetry_mono"] is None


def test_recorder_drops_instead_of_blocking_the_ui_thread(tmp_path, monkeypatch) -> None:
    # A stalled flight UI is worse than a gap in the recording, so a wedged
    # writer has to cost dropped frames and nothing else.
    gate = threading.Event()

    def _wedged(_image, _quality):
        gate.wait(5.0)
        return b"\xff\xd8\xff\xd9"

    monkeypatch.setattr(imu_flight_test, "_encode_jpeg", _wedged)
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        for seq in range(imu_flight_test._QUEUE_DEPTH + 4):
            assert recorder.capture(seq, f"stream_{seq:06d}", _frame(), _timing()) in {
                True,
                False,
            }
        assert recorder.dropped_queue_full >= 3
    finally:
        gate.set()
        recorder.close()


def test_recorder_stops_at_the_frame_cap(tmp_path) -> None:
    recorder = ImuFlightTestRecorder(tmp_path, max_frames=2)
    try:
        # Written is owned by the writer thread; set it directly so the cap is
        # checked rather than the scheduler.
        recorder.written = 2
        assert recorder.capture(9, "stream_000009", _frame(), _timing()) is False
        assert recorder.stop_reason == "max_frames"
        # The stop latches: later submissions are refused outright.
        assert recorder.capture(10, "stream_000010", _frame(), _timing()) is False
    finally:
        recorder.close()
    assert _rows(tmp_path) == []


def test_recorder_stride_keeps_every_nth_submission(tmp_path) -> None:
    recorder = ImuFlightTestRecorder(tmp_path, every_n=3)
    try:
        for seq in range(6):
            recorder.capture(seq, f"stream_{seq:06d}", _frame(), _timing())
    finally:
        recorder.close()
    assert sorted(row["seq"] for row in _rows(tmp_path)) == [0, 3]
    assert recorder.skipped_stride == 4


def test_recorder_ignores_a_frame_that_is_not_an_image(tmp_path) -> None:
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        assert recorder.capture(0, "stream_000000", None, _timing()) is False
        assert recorder.capture(1, "stream_000001", np.zeros((4, 4)), _timing()) is False
    finally:
        recorder.close()
    assert _rows(tmp_path) == []


def test_recorder_accepts_the_simulated_stream_pil_frame_too(tmp_path) -> None:
    # The live grabber hands over numpy RGB, the simulated stream a PIL image.
    # Rejecting the second silently left imu_test/frames empty on every sim run,
    # which is also the only way to exercise this path without a drone.
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        assert recorder.capture(
            3, "frame_000003", Image.fromarray(_frame()), _timing()
        ) is True
    finally:
        recorder.close()
    assert [row["seq"] for row in _rows(tmp_path)] == [3]
    assert (tmp_path / "frames" / "000001.jpg").is_file()


def test_frame_capture_never_propagates_an_encode_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(imu_flight_test, "_encode_jpeg", lambda *_args: None)
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        assert recorder.capture(0, "stream_000000", _frame(), _timing()) is True
    finally:
        recorder.close()
    assert recorder.written == 0
    assert recorder.dropped_encode_failed == 1


# ---------------------------------------------------------------- stick logging
class _Monitor:
    deadzone = 2000

    def __init__(self, axes: dict[int, int]):
        self.axes = dict(axes)

    def snapshot_axes(self) -> dict[int, int]:
        return dict(self.axes)


class _Sink:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict]] = []

    def telemetry(self, event: str, **fields) -> bool:
        self.rows.append((event, fields))
        return True


def _backend(axes: dict[int, int] | None) -> SimpleNamespace:
    return SimpleNamespace(
        _stick_monitor=None if axes is None else _Monitor(axes),
        _last_stick_axes={},
        _last_stick_axes_t=0.0,
        pilot_sticks=True,
        session_logs=_Sink(),
    )


def _poll(backend, now: float) -> None:
    backend_module.OlympeLiveBackend._poll_stick_axes(backend, now)


def test_stick_axes_are_recorded_so_manual_flight_has_an_input_track() -> None:
    # A manually flown session issues no PCMD, so commands.jsonl stays empty and
    # the recorded IMU has nothing explaining it. These rows are that input.
    backend = _backend({0: 0, 1: 0, 2: 0, 3: 0})
    _poll(backend, 100.0)
    assert [event for event, _ in backend.session_logs.rows] == ["stick_axes"]

    backend._stick_monitor.axes[1] = 9000
    _poll(backend, 100.05)
    assert len(backend.session_logs.rows) == 2
    _event, fields = backend.session_logs.rows[-1]
    assert fields["axes"] == {"0": 0, "1": 9000, "2": 0, "3": 0}
    assert fields["moved"] is True
    assert fields["flight_axes_active"] is True
    assert fields["pilot_sticks"] is True


def test_a_parked_stick_costs_one_heartbeat_row_per_second() -> None:
    backend = _backend({0: 0, 1: 0, 2: 0, 3: 0})
    _poll(backend, 100.0)
    for offset in (0.12, 0.24, 0.36, 0.48):
        _poll(backend, 100.0 + offset)
    assert len(backend.session_logs.rows) == 1
    _poll(backend, 101.1)
    assert len(backend.session_logs.rows) == 2


def test_stick_logging_is_silent_without_a_monitor() -> None:
    backend = _backend(None)
    _poll(backend, 100.0)
    assert backend.session_logs.rows == []

    backend = _backend({})
    _poll(backend, 100.0)
    assert backend.session_logs.rows == []


def test_recorder_stops_before_it_eats_the_flight_safety_disk_margin(
    tmp_path, monkeypatch
) -> None:
    # The frame cap is counted in frames, so on a laptop already near the
    # documented 5 GiB / 5% floor it is not a disk guard at all. Below that
    # floor the backend blocks takeoff and the safety session logs are at risk,
    # so a truncated recording is the correct trade.
    monkeypatch.setattr(
        imu_flight_test,
        "assess_disk_space",
        lambda _path: SimpleNamespace(takeoff_blocked=True, reason="critical disk space"),
    )
    monkeypatch.setattr(imu_flight_test, "_DISK_CHECK_EVERY", 1)
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        recorder.capture(0, "stream_000000", _frame(), _timing())
        recorder.close()
    finally:
        pass
    assert recorder.stop_reason.startswith("disk_floor:")
    assert recorder.capture(1, "stream_000001", _frame(), _timing()) is False


def test_a_held_frame_is_stored_once_and_never_overwrites_the_previous_one(
    tmp_path,
) -> None:
    # Measured on a simulated run: BOOT hold resubmitted stream index 1 forty
    # times. Naming files by that index wrote every one to frames/000001.jpg,
    # so the whole 40-frame cap produced a single image on disk.
    recorder = ImuFlightTestRecorder(tmp_path)
    try:
        for _ in range(5):
            recorder.capture(1, "frame_000002.jpg", _frame(), _timing(hold_kind="boot"))
        recorder.capture(2, "frame_000003.jpg", _frame(), _timing())
        recorder.capture(3, "frame_000004.jpg", _frame(), _timing())
    finally:
        recorder.close()

    rows = _rows(tmp_path)
    assert [row["seq"] for row in rows] == [1, 2, 3]
    assert [row["file"] for row in rows] == [
        "frames/000001.jpg",
        "frames/000002.jpg",
        "frames/000003.jpg",
    ]
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3
    assert recorder.skipped_duplicate == 4
