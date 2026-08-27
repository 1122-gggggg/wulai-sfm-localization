from __future__ import annotations

import sys
from types import SimpleNamespace

import live_non_map_acceptance as acceptance


def _stub_runtime_imports(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "olympe_live_backend",
        SimpleNamespace(OlympeLiveBackend=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "backend_contract",
        SimpleNamespace(
            ControlAction=object(),
            ControlRequest=object(),
            ControlResult=object,
        ),
    )


def _runner(*, fly: bool):
    runner = object.__new__(acceptance.LiveAcceptance)
    runner.ip = "test.invalid"
    runner.controller = "test"
    runner.fly = fly
    return runner


def test_ground_only_run_never_enters_air_checks(monkeypatch):
    _stub_runtime_imports(monkeypatch)
    runner = _runner(fly=False)
    events = []
    runner.log = lambda *args, **kwargs: events.append(("log", args, kwargs))
    runner._connect_backend = lambda _backend: events.append("connect") or True
    runner._check_telemetry = lambda: events.append("telemetry") or "landed"
    runner._check_video = lambda: events.append("video")
    runner._check_ground_camera = lambda: events.append("ground_camera")
    runner._finish_ground_only = lambda: events.append("finish_ground") or 0
    runner._takeoff = lambda *_args: (_ for _ in ()).throw(
        AssertionError("ground-only run attempted takeoff")
    )

    assert runner.run() == 0
    assert events[1:] == [
        "connect",
        "telemetry",
        "video",
        "ground_camera",
        "finish_ground",
    ]


def test_air_run_orders_checks_after_explicit_takeoff_gate(monkeypatch):
    _stub_runtime_imports(monkeypatch)
    runner = _runner(fly=True)
    events = []
    runner.log = lambda *_args, **_kwargs: None
    runner._connect_backend = lambda _backend: events.append("connect") or True
    runner._check_telemetry = lambda: events.append("telemetry") or "landed"
    runner._check_video = lambda: events.append("video")
    runner._check_ground_camera = lambda: events.append("ground_camera")
    runner._takeoff = lambda *_args: events.append("takeoff") or True
    runner.backend = SimpleNamespace(
        hover_cmd=lambda reason: events.append(("hover", reason))
    )
    runner._check_nudges = lambda: events.append("nudges")
    runner._check_freeze_resume = lambda: events.append("freeze_resume")
    runner._check_air_camera = lambda: events.append("air_camera")
    runner._check_air_video = lambda: events.append("air_video")
    runner._check_landing = lambda: events.append("landing")
    runner._finish_report = lambda: events.append("finish") or 0
    monkeypatch.setattr(acceptance.time, "sleep", lambda _seconds: None)

    assert runner.run() == 0
    assert events == [
        "connect",
        "telemetry",
        "video",
        "ground_camera",
        "takeoff",
        ("hover", "post_takeoff_settle"),
        "nudges",
        "freeze_resume",
        "air_camera",
        "air_video",
        "landing",
        "finish",
    ]
