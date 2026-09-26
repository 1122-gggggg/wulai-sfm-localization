from __future__ import annotations

import json

from types import SimpleNamespace
import log_anafi_telemetry as telemetry


def test_partial_video_start_is_stopped_and_drone_disconnects(monkeypatch, tmp_path):
    events = []

    class FakeDrone:
        def disconnect(self):
            events.append("disconnect")

    class PartialGrabber:
        def __init__(self, *_args, **_kwargs):
            events.append("construct")

        def start(self):
            events.append("start")
            raise RuntimeError("partial start")

        def stop(self):
            events.append("stop")

    clock = iter((0.0, 1.0))
    monkeypatch.setattr(telemetry, "_build_message_table", lambda *_: [])
    monkeypatch.setattr(telemetry, "connect", lambda *_args, **_kwargs: FakeDrone())
    monkeypatch.setattr(telemetry, "RtspFrameGrabber", PartialGrabber)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: next(clock))

    telemetry.run_log(
        ip="test.invalid",
        controller="test",
        secs=0.1,
        hz=1.0,
        out=tmp_path / "telemetry.jsonl",
        with_video=True,
    )

    assert events == ["construct", "start", "stop", "disconnect"]


def test_run_log_writes_start_sample_and_end_records(monkeypatch, tmp_path):
    events = []

    class FakeDrone:
        def disconnect(self):
            events.append("disconnect")

    monotonic_values = iter((0.0, 0.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(telemetry, "_build_message_table", lambda *_: [("attitude", object())])
    monkeypatch.setattr(telemetry, "connect", lambda *_args, **_kwargs: FakeDrone())
    monkeypatch.setattr(
        telemetry,
        "sample_once",
        lambda *_args: {
            "t_unix": 1.0,
            "t_mono": 0.0,
            "t_iso": "test",
            "attitude": {"roll": 0.0},
            "_channels_ok": ["attitude"],
            "_channels_missing": [],
        },
    )
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(telemetry.time, "sleep", lambda _seconds: None)
    output = tmp_path / "telemetry.jsonl"

    result = telemetry.run_log(
        ip="test.invalid",
        controller="test",
        secs=0.1,
        hz=1.0,
        out=output,
        with_video=False,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert result == output
    assert records[0]["event"] == "start"
    assert records[0]["safety"] == "passive_no_arm"
    assert records[1]["attitude"] == {"roll": 0.0}
    assert records[2] == {
        "event": "end",
        "samples": 1,
        "video_frames_ok": 0,
        "video_frames_none": 0,
        "t_iso": records[2]["t_iso"],
    }
    assert events == ["disconnect"]


def test_video_log_marks_receipt_stamp_as_degraded():
    frame = SimpleNamespace(shape=(720, 1280, 3))

    class Grabber:
        stream_metadata = {
            "timestamp_degraded": True,
            "stamp_source": "opencv-rtsp-receipt-degraded",
        }

        def __call__(self):
            return frame, 12.5

    row = {}
    assert telemetry._add_video_sample(row, Grabber()) == (1, 0)
    assert row["video.source_timestamp_degraded"] is True
    assert row["video.source_timestamp_trusted"] is False
    assert row["video.stamp_source"] == "opencv-rtsp-receipt-degraded"
    assert row["video.frame"]["stamp_mono"] == 12.5
    assert row["video.frame"]["timestamp_degraded"] is True
    assert row["video.frame"]["source_timestamp_trusted"] is False
