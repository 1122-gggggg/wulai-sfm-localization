from __future__ import annotations

import json

import passive_flight_session as session


def test_run_session_remains_passive_and_cleans_up(monkeypatch, tmp_path):
    events = []

    class FakeDrone:
        def disconnect(self):
            events.append("disconnect")

    class FakeGrabber:
        fps = 0.0

        def __init__(self, *_args, **_kwargs):
            events.append("construct_video")

        def start(self):
            events.append("start_video")
            return self

        def __call__(self):
            return None

        def is_healthy(self):
            return True

        def stop(self):
            events.append("stop_video")

    monotonic_values = iter((0.0, 0.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(session.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(session.tel, "_build_message_table", lambda: [])
    monkeypatch.setattr(
        session.tel,
        "sample_once",
        lambda *_args: {
            "t_unix": 1.0,
            "t_mono": 0.0,
            "t_iso": "test",
            "_channels_ok": [],
            "_channels_missing": [],
        },
    )
    monkeypatch.setattr(session.ofs, "connect", lambda *_args, **_kwargs: FakeDrone())
    monkeypatch.setattr(session.ofs, "OlympePdrawGrabber", FakeGrabber)
    monkeypatch.setattr(session.pff, "_env_float", lambda *_args, **_kwargs: 1.5)
    monkeypatch.setattr(session.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(session.time, "sleep", lambda _seconds: None)
    output = tmp_path / "passive.jsonl"

    result = session.run_session(
        ip="test.invalid",
        controller="test",
        secs=0.1,
        hz=1.0,
        out=output,
        with_localize=False,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert result == output
    assert records[0]["safety"] == "passive_no_arm"
    assert records[0]["with_localize"] is False
    assert records[1]["video.frame"] is None
    assert records[1]["loc"] is None
    assert records[2]["event"] == "end"
    assert records[2]["samples"] == 1
    assert records[2]["loc_skip"] == 1
    assert events == ["construct_video", "start_video", "stop_video", "disconnect"]


def test_pose_jump_is_rejected_without_replacing_last_good_pose():
    class Pose:
        y = 0.0
        z = 0.0

        def __init__(self, x):
            self.x = x

    stats = session._SessionStats(last_good_xyz=(0.0, 0.0, 0.0))

    finite, jump = session._record_pose_quality(Pose(2.0), stats, 1.5)

    assert finite is True
    assert jump is True
    assert stats.jump_rejects == 1
    assert stats.loc_fail == 1
    assert stats.last_good_xyz == (0.0, 0.0, 0.0)
