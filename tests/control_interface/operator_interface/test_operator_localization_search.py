from __future__ import annotations

import math
import threading

import pytest

from operator_localization_search import (
    BoundedYawSearch,
    ManualLocalizationSearch,
    YawSearchConfig,
)


def test_bounded_search_waits_then_alternates_at_the_angular_limits() -> None:
    config = YawSearchConfig(
        wait_before_search_s=1.0,
        max_search_s=6.0,
        sweep_angle_deg=20.0,
        yaw_pcmd=7,
    )
    search = BoundedYawSearch(config)
    search.begin(10.0)

    assert search.step(now=10.5, aircraft_yaw_rad=0.0).yaw_pcmd == 0
    assert search.step(now=11.0, aircraft_yaw_rad=0.0).yaw_pcmd == 7
    right_limit = search.step(
        now=11.5, aircraft_yaw_rad=math.radians(21.0)
    )
    assert right_limit.yaw_pcmd == -7
    assert right_limit.state == "searching_left"
    left_limit = search.step(
        now=12.0, aircraft_yaw_rad=math.radians(-21.0)
    )
    assert left_limit.yaw_pcmd == 7
    assert left_limit.state == "searching_right"


def test_bounded_search_handles_yaw_wrap_without_false_reversal() -> None:
    search = BoundedYawSearch(
        YawSearchConfig(wait_before_search_s=0.1, max_search_s=2.0)
    )
    search.begin(0.0)
    search.step(now=0.1, aircraft_yaw_rad=math.radians(179.0))
    step = search.step(now=0.2, aircraft_yaw_rad=math.radians(-179.0))

    assert step.yaw_pcmd > 0
    assert step.relative_yaw_deg == pytest.approx(2.0)


def test_bounded_search_refuses_to_turn_without_finite_yaw_and_times_out() -> None:
    config = YawSearchConfig(wait_before_search_s=0.5, max_search_s=1.0)
    search = BoundedYawSearch(config)
    search.begin(3.0)

    unavailable = search.step(now=3.5, aircraft_yaw_rad=float("nan"))
    assert unavailable.state == "no_yaw_telemetry"
    assert unavailable.yaw_pcmd == 0
    timed_out = search.step(now=4.5, aircraft_yaw_rad=None)
    assert timed_out.state == "timed_out"
    assert timed_out.yaw_pcmd == 0


@pytest.mark.parametrize("terminal", ["pose_found", "unsafe"])
def test_bounded_search_terminal_conditions_always_return_zero(terminal: str) -> None:
    search = BoundedYawSearch(
        YawSearchConfig(wait_before_search_s=0.1, max_search_s=1.0)
    )
    kwargs = {"pose_found": terminal == "pose_found", "safe": terminal != "unsafe"}
    step = search.step(now=1.0, aircraft_yaw_rad=0.0, **kwargs)

    assert step.terminal
    assert step.state == terminal
    assert step.yaw_pcmd == 0


def test_manual_search_cancels_synchronously_to_zero() -> None:
    calls = []
    entered_sleep = threading.Event()
    release_sleep = threading.Event()

    def send(roll, pitch, yaw, gaz, reason):
        calls.append((roll, pitch, yaw, gaz, reason))
        return True

    def sleep(_duration):
        entered_sleep.set()
        release_sleep.wait(1.0)

    search = ManualLocalizationSearch(
        send_pcmd=send,
        aircraft_yaw=lambda: 0.0,
        pose_found=lambda: False,
        safe_to_search=lambda: True,
        force_relocalize=lambda: None,
        config=YawSearchConfig(
            wait_before_search_s=0.1,
            max_search_s=1.0,
            tick_s=0.1,
        ),
        now=lambda: 1.0,
        sleep=sleep,
    )

    assert search.start()
    assert entered_sleep.wait(1.0)
    search.cancel("keyboard")
    assert calls[-1][:4] == (0, 0, 0, 0)
    assert calls[-1][4] == "manual_localization_search_keyboard"
    release_sleep.set()
    assert search.join(1.0)
    assert search.finished_state == "cancelled"
    assert calls[-1][:4] == (0, 0, 0, 0)


def test_manual_search_never_starts_when_authority_gate_is_closed() -> None:
    calls = []
    search = ManualLocalizationSearch(
        send_pcmd=lambda *values: calls.append(values) or True,
        aircraft_yaw=lambda: 0.0,
        pose_found=lambda: False,
        safe_to_search=lambda: False,
        force_relocalize=lambda: None,
    )

    assert not search.start()
    assert calls == []
