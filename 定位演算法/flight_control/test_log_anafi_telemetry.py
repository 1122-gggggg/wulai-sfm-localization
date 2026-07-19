from __future__ import annotations

import olympe_frame_source as frame_source

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
    monkeypatch.setattr(telemetry, "_build_message_table", lambda: [])
    monkeypatch.setattr(telemetry, "connect", lambda *_args, **_kwargs: FakeDrone())
    monkeypatch.setattr(frame_source, "OlympePdrawGrabber", PartialGrabber)
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
