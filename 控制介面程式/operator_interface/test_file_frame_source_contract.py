from __future__ import annotations

import io
from pathlib import Path

import pytest

import flight_operator_app as app
from flight_operator_app import FFmpegFrameStream


class _Process:
    def __init__(self, payload: bytes):
        self.stdout = io.BytesIO(payload)


def _stream_with_bytes(payload: bytes) -> FFmpegFrameStream:
    stream = FFmpegFrameStream(Path("/definitely/missing.mp4"), 1, 1, fps=30)
    stream.proc = _Process(payload)  # type: ignore[assignment]
    return stream


def test_file_stream_defaults_to_no_loop_and_latches_eof() -> None:
    stream = _stream_with_bytes(b"\x01\x02\x03")

    assert stream.loop is False
    assert stream.next_frame() is not None
    assert stream.output_index == 1
    assert stream.next_frame() is None
    assert stream.eof is True
    assert stream.output_index == 1


def test_eof_does_not_restart_decoder() -> None:
    stream = _stream_with_bytes(b"")
    stream.start = lambda: (_ for _ in ()).throw(AssertionError("must not loop"))  # type: ignore[method-assign]

    assert stream.next_frame() is None
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
    assert [stream.next_frame() for _ in range(8)] == [None] * 8
    assert stream.next_frame() is not None
    assert stream.output_index == 1


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
