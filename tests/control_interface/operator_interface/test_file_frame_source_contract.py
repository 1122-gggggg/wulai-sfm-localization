from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import pytest

import flight_operator_app as app
from flight_operator_app import FFmpegFrameStream


class _Process:
    def __init__(self, payload: bytes | object):
        self.stdout = payload if hasattr(payload, "read") else io.BytesIO(payload)
        self.returncode = 0
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminated = True
        self.returncode = -9


class _BlockingStdout:
    def __init__(self):
        self.read_started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def read(self, _size: int) -> bytes:
        self.read_started.set()
        self.release.wait(2.0)
        return b""

    def close(self) -> None:
        self.closed.set()
        self.release.set()


class _ErrorStdout:
    def read(self, _size: int) -> bytes:
        raise RuntimeError("synthetic decoder failure")


def _stream_with_bytes(payload: bytes) -> FFmpegFrameStream:
    stream = FFmpegFrameStream(Path("/definitely/missing.mp4"), 1, 1, fps=30)
    stream.proc = _Process(payload)  # type: ignore[assignment]
    return stream


def _drain_frames(stream: FFmpegFrameStream, expected: int | None = None) -> list:
    """Drive one request at a time while the background reader does pipe I/O."""
    frames = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        frame = stream.next_frame()
        if frame is not None:
            frames.append(frame)
            if expected is not None and len(frames) >= expected:
                return frames
        if (
            expected is None
            and stream.eof
            and not stream._delay_buf
            and stream._reader_queue.empty()
        ):
            return frames
        time.sleep(0.001)
    raise AssertionError("background frame reader did not reach the expected state")


def test_file_stream_defaults_to_no_loop_and_latches_eof() -> None:
    stream = _stream_with_bytes(b"\x01\x02\x03")

    assert stream.loop is False
    assert _drain_frames(stream, expected=1)
    assert stream.output_index == 1
    assert _drain_frames(stream) == []
    assert stream.eof is True
    assert stream.output_index == 1


def test_eof_does_not_restart_decoder() -> None:
    stream = _stream_with_bytes(b"")
    stream.start = lambda: (_ for _ in ()).throw(AssertionError("must not loop"))  # type: ignore[method-assign]

    assert stream.next_frame() is None
    _drain_frames(stream)
    assert stream.eof is True


def test_anafi_link_latency_is_applied_before_frames_reach_localization() -> None:
    frame = b"\x01\x02\x03"
    stream = FFmpegFrameStream(
        Path("/definitely/missing.mp4"),
        1,
        1,
        fps=30,
        link_sim={"latency_ms": 280},
    )
    stream.proc = _Process(frame * 9)  # type: ignore[assignment]

    assert stream.delay_frames == 8
    frames = _drain_frames(stream, expected=9)
    assert frames[0].getpixel((0, 0)) == (1, 2, 3)
    assert stream.output_index == 9


def test_file_stream_drains_delayed_tail_once_at_eof() -> None:
    frames = (
        b"\x01\x02\x03"
        + b"\x04\x05\x06"
        + b"\x07\x08\x09"
    )
    stream = _stream_with_bytes(frames)
    stream.delay_frames = 2

    output = _drain_frames(stream)
    assert [frame.getpixel((0, 0)) for frame in output] == [
        (1, 2, 3), (4, 5, 6), (7, 8, 9),
    ]
    assert stream.output_index == 3


def test_next_frame_never_waits_for_a_blocked_ffmpeg_pipe() -> None:
    stdout = _BlockingStdout()
    stream = FFmpegFrameStream(Path("/definitely/missing.mp4"), 1, 1, fps=30)
    process = _Process(stdout)
    stream.proc = process  # type: ignore[assignment]

    started = time.monotonic()
    assert stream.next_frame() is None
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert stdout.read_started.wait(1.0)

    started = time.monotonic()
    stream.close()
    assert time.monotonic() - started < 0.5
    assert stdout.closed.is_set()
    assert process.terminated is True


def test_reader_latches_decode_errors_without_touching_tk() -> None:
    stream = FFmpegFrameStream(Path("/definitely/missing.mp4"), 1, 1, fps=30)
    stream.proc = _Process(_ErrorStdout())  # type: ignore[assignment]

    assert _drain_frames(stream) == []
    assert stream.eof is True
    assert stream.terminal_state == "DECODE_ERROR_HOLD"


def test_file_stream_preserves_native_rate_instead_of_duplicating_frames(
    tmp_path, monkeypatch
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not decoded in this unit test")
    monkeypatch.setattr(app, "probe_video_fps", lambda _path: 24000 / 1001)
    monkeypatch.setattr(FFmpegFrameStream, "start", lambda self: None)

    stream = FFmpegFrameStream(video, 1280, 720, fps=30)

    assert stream.requested_fps == 30
    assert stream.source_fps == pytest.approx(24000 / 1001)
    assert stream.output_fps == pytest.approx(24000 / 1001)
    assert stream._video_filter() == "scale=1280:720"


def test_file_stream_downsamples_sources_above_requested_rate(
    tmp_path, monkeypatch
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not decoded in this unit test")
    monkeypatch.setattr(app, "probe_video_fps", lambda _path: 60.0)
    monkeypatch.setattr(FFmpegFrameStream, "start", lambda self: None)

    stream = FFmpegFrameStream(video, 1280, 720, fps=30)

    assert stream.output_fps == 30
    assert stream._video_filter() == "fps=30,scale=1280:720"
