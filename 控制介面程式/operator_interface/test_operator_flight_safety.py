from __future__ import annotations

import threading

from operator_flight_safety import AuthorityController, TelemetryFreshnessStore


def test_authority_controller_serializes_transition_owners() -> None:
    controller = AuthorityController()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    order: list[str] = []

    def first_transition() -> None:
        with controller.transition():
            first_entered.set()
            assert release_first.wait(2.0)
            order.append("first")

    def second_transition() -> None:
        assert first_entered.wait(2.0)
        second_started.set()
        with controller.transition():
            second_entered.set()
            order.append("second")

    first = threading.Thread(target=first_transition)
    second = threading.Thread(target=second_transition)
    first.start()
    second.start()

    assert first_entered.wait(2.0)
    assert second_started.wait(2.0)
    assert not second_entered.wait(0.05)

    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert order == ["first", "second"]


def test_telemetry_store_does_not_refresh_an_unchanged_event_marker() -> None:
    store = TelemetryFreshnessStore()

    store.observe("ground_speed", 0.2, marker="event-1", observed_mono_ns=1_000_000_000)
    store.observe("ground_speed", 0.2, marker="event-1", observed_mono_ns=2_000_000_000)

    assert store.fresh(
        "ground_speed", max_age_s=0.5, now_mono_ns=1_400_000_000
    ) == 0.2
    assert store.fresh(
        "ground_speed", max_age_s=0.5, now_mono_ns=1_600_000_000
    ) is None

    store.observe("ground_speed", 0.3, marker="event-2", observed_mono_ns=2_000_000_000)
    assert store.fresh(
        "ground_speed", max_age_s=0.5, now_mono_ns=2_100_000_000
    ) == 0.3


def test_telemetry_store_does_not_treat_a_future_sample_as_fresh() -> None:
    store = TelemetryFreshnessStore()
    store.observe(
        "ground_speed", 0.2, marker="future", observed_mono_ns=2_000_000_000
    )

    assert store.fresh(
        "ground_speed", max_age_s=0.5, now_mono_ns=1_900_000_000
    ) is None
