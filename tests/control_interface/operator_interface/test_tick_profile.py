"""Per-stage UI tick profiling (runbook 0b).

ui_poll_delay_ms p50 was 25 ms with a 61.7% worker duty cycle: the result is
already in the client queue and the Tk loop is busy running the tick. The doc's
per-tick render costs do not add up to 25 ms and its two cheap candidates are
already implemented, so the tick has to say which stage holds the loop. These
tests pin that the instrumentation reports every stage, stays bounded, emits at
a low rate, and cannot break the tick it measures.
"""
from __future__ import annotations

import pytest

import operator_tick


class _Telemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event: str, **fields) -> None:
        self.events.append((event, fields))


class _Logs:
    def __init__(self) -> None:
        self.telemetry = _Telemetry()


class _App:
    def __init__(self) -> None:
        self.session_logs = _Logs()
        self.tick_ms = 33
        self.diagnostic_failures: list = []

    def _record_diagnostic_failure(self, name, exc) -> None:
        self.diagnostic_failures.append((name, exc))


def _profile_one_tick(app, stages=("a", "b", "c")) -> None:
    profile = operator_tick._TickProfile(app, stages[0])
    for name in stages[1:]:
        profile.next(name)
    profile.finish()


def test_every_stage_and_a_total_are_recorded() -> None:
    app = _App()
    _profile_one_tick(app, ("a", "b", "c"))
    samples = app._tick_profile_samples
    assert set(samples) == {"a", "b", "c", "_total"}
    assert all(len(v) == 1 for v in samples.values())


def test_total_is_the_sum_of_the_stages() -> None:
    app = _App()
    _profile_one_tick(app, ("a", "b", "c"))
    samples = app._tick_profile_samples
    staged = sum(samples[name][0] for name in ("a", "b", "c"))
    assert samples["_total"][0] == staged


def test_next_returns_the_stage_name_so_failure_reporting_is_unchanged() -> None:
    profile = operator_tick._TickProfile(_App(), "drain_flight_command_results")
    assert profile.next("poll_backend") == "poll_backend"


def test_elapsed_time_is_filed_under_the_stage_that_ran(monkeypatch) -> None:
    """Off-by-one here would blame every stage for its predecessor's cost."""
    clock = {"t": 0.0}
    monkeypatch.setattr(operator_tick.time, "perf_counter", lambda: clock["t"])
    app = _App()
    profile = operator_tick._TickProfile(app, "slow_stage")
    clock["t"] += 0.010          # slow_stage took 10 ms
    profile.next("fast_stage")
    clock["t"] += 0.001          # fast_stage took 1 ms
    profile.finish()

    samples = app._tick_profile_samples
    assert samples["slow_stage"] == pytest.approx([10.0])
    assert samples["fast_stage"] == pytest.approx([1.0])


def test_first_window_only_arms_the_timer_and_emits_nothing() -> None:
    app = _App()
    _profile_one_tick(app)
    assert app.session_logs.telemetry.events == []
    # Samples are kept for the window that will actually be emitted.
    assert app._tick_profile_samples


def test_emit_is_rate_limited_and_clears_the_window(monkeypatch) -> None:
    app = _App()
    clock = {"t": 1000.0}
    monkeypatch.setattr(operator_tick.time, "monotonic", lambda: clock["t"])

    _profile_one_tick(app)          # arms the timer
    clock["t"] += 1.0
    _profile_one_tick(app)          # inside the window: still silent
    assert app.session_logs.telemetry.events == []

    clock["t"] += operator_tick._TICK_PROFILE_EMIT_S
    _profile_one_tick(app)
    assert len(app.session_logs.telemetry.events) == 1
    event, fields = app.session_logs.telemetry.events[0]
    assert event == "ui_tick_profile"
    assert fields["tick_period_ms"] == 33
    assert fields["ticks"] == 3
    assert set(fields["stages"]) == {"a", "b", "c", "_total"}
    assert set(fields["stages"]["a"]) == {"p50", "p95"}
    # The window is reset, so the next emit describes only new ticks.
    assert app._tick_profile_samples == {}


def test_samples_stay_bounded_when_nothing_ever_emits() -> None:
    app = _App()
    app._tick_profile_last_emit = float("inf")  # never due
    for _ in range(operator_tick._TICK_PROFILE_MAX_SAMPLES + 50):
        _profile_one_tick(app, ("a", "b"))
    for values in app._tick_profile_samples.values():
        assert len(values) == operator_tick._TICK_PROFILE_MAX_SAMPLES


def test_a_failing_telemetry_sink_cannot_break_the_tick(monkeypatch) -> None:
    app = _App()
    clock = {"t": 1000.0}
    monkeypatch.setattr(operator_tick.time, "monotonic", lambda: clock["t"])

    def _boom(_event, **_fields):
        raise OSError("log volume full")

    app.session_logs.telemetry = _boom
    _profile_one_tick(app)
    clock["t"] += operator_tick._TICK_PROFILE_EMIT_S + 1.0
    _profile_one_tick(app)

    assert app.diagnostic_failures  # recorded, not raised
    assert app._tick_profile_samples == {}


def test_an_app_without_a_session_log_still_ticks() -> None:
    class _Bare:
        tick_ms = 33

    app = _Bare()
    _profile_one_tick(app)
    app._tick_profile_last_emit = 0.0
    _profile_one_tick(app)


def test_percentiles_are_ordered() -> None:
    values = [float(v) for v in range(1, 101)]
    assert operator_tick._percentile(values) <= operator_tick._percentile(values, 0.95)
    assert operator_tick._percentile(values, 0.95) == 95.0
