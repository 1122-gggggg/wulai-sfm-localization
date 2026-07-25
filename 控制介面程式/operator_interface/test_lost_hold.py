from __future__ import annotations

import pytest

from flight_operator_app import LostHoldPolicy


def fail(next_mode: str = "LOST") -> dict:
    return {"success": False, "next_mode": next_mode}


def track_fix() -> dict:
    return {"success": True, "next_mode": "TRACK", "relocalize_requested": True}


def feed(policy: LostHoldPolicy, result: dict, *, frame_index: int = 7,
         now: float = 0.0) -> str | None:
    return policy.on_result(
        success=bool(result["success"]), low_confidence=False,
        strong_relocalize=bool(result.get("relocalize_requested")),
        next_mode=result["next_mode"],
        frame_index=frame_index, now=now)


def test_hold_waits_for_lost_instead_of_engaging_on_weak_failure():
    policy = LostHoldPolicy()

    assert feed(policy, fail("WEAK_TRACK"), frame_index=41) is None
    assert not policy.active
    assert feed(policy, fail("LOST"), frame_index=42) == "ENGAGE_FAIL"
    assert policy.active
    assert policy.frame_index == 42
    assert policy.attempts == 0


def test_held_frame_is_retried_then_released_by_a_fix():
    policy = LostHoldPolicy(max_attempts=5)
    feed(policy, fail())

    for expected in (1, 2):
        assert policy.wants_retry()
        policy.note_submit()
        assert policy.attempts == expected
        assert feed(policy, fail()) is None       # still LOST -> keep holding
        assert policy.active

    policy.note_submit()
    assert feed(policy, track_fix()) == "RELEASE_FIX"
    assert not policy.active
    assert policy.armed                            # a fix re-arms the next episode
    assert policy.attempts == 0


def test_automatic_lost_recovery_fix_does_not_enqueue_duplicate_retry():
    policy = LostHoldPolicy(max_attempts=5)
    assert feed(policy, fail()) == "ENGAGE_FAIL"

    automatic_fix = {
        "success": True,
        "next_mode": "TRACK",
        "relocalize_requested": False,
    }
    assert feed(policy, automatic_fix) == "RELEASE_FIX"
    assert not policy.active
    assert not policy.wants_retry()


def test_unsolvable_frame_releases_the_stream_after_max_attempts():
    policy = LostHoldPolicy(max_attempts=3)
    feed(policy, fail())

    for _ in range(2):                             # attempts 1 and 2 keep the hold
        assert policy.wants_retry()
        policy.note_submit()
        assert feed(policy, fail()) is None

    policy.note_submit()                           # third and last allowed attempt
    assert policy.attempts == 3
    assert not policy.wants_retry()                # budget spent
    assert feed(policy, fail()) == "RELEASE_ATTEMPTS"
    assert not policy.active

    # Disarmed: a still-LOST tracker must not immediately re-freeze the stream,
    # which would crawl the video one frame per exhausted retry budget.
    assert feed(policy, fail()) is None
    assert not policy.active


def test_a_fix_rearms_the_hold_after_a_released_episode():
    policy = LostHoldPolicy(max_attempts=1)
    feed(policy, fail())
    policy.note_submit()
    assert feed(policy, fail()) == "RELEASE_ATTEMPTS"

    assert feed(policy, track_fix()) is None       # recovered on a fresh frame
    assert policy.armed
    assert feed(policy, fail()) == "ENGAGE_FAIL"   # next lost episode holds again


def test_stalled_retries_time_out_so_the_ui_cannot_freeze():
    policy = LostHoldPolicy(max_attempts=5, timeout_s=10.0)
    feed(policy, fail(), now=100.0)

    assert policy.check_timeout(109.9) is None
    assert policy.active

    assert policy.check_timeout(110.0) == "RELEASE_TIMEOUT"
    assert not policy.active
    assert not policy.armed


def test_timeout_is_disabled_when_zero():
    policy = LostHoldPolicy(timeout_s=0.0)
    feed(policy, fail(), now=0.0)
    assert policy.check_timeout(1e6) is None
    assert policy.active


@pytest.mark.parametrize("mode", ["TRACK", "WEAK_TRACK", "BOOT_INIT", ""])
def test_non_lost_missing_pose_does_not_pause_the_stream(mode):
    policy = LostHoldPolicy()
    assert feed(policy, fail(mode)) is None
    assert not policy.active


def test_low_confidence_never_engages_megaloc_hold():
    policy = LostHoldPolicy(low_confidence_results=2)

    for now in (1.0, 2.0, 3.0):
        assert policy.on_result(
            success=True, low_confidence=True, strong_relocalize=False,
            next_mode="WEAK_TRACK", frame_index=10, now=now,
        ) is None
    assert not policy.active


def test_low_confidence_hold_engages_after_the_configured_run():
    """Accuracy-first mode: a run of low-confidence fixes pauses the stream."""
    policy = LostHoldPolicy(low_confidence_results=2, hold_on_low_confidence=True)

    assert policy.on_result(
        success=True, low_confidence=True, strong_relocalize=False,
        next_mode="WEAK_TRACK", frame_index=10, now=1.0,
    ) is None
    assert not policy.active

    assert policy.on_result(
        success=True, low_confidence=True, strong_relocalize=False,
        next_mode="WEAK_TRACK", frame_index=11, now=2.0,
    ) == "ENGAGE_LOW_CONF"
    assert policy.active
    assert policy.frame_index == 11
    # The held frame must be retried, which is what escalates it to LOST recovery.
    assert policy.wants_retry()


def test_low_confidence_hold_releases_on_a_trustworthy_fix():
    policy = LostHoldPolicy(low_confidence_results=1, hold_on_low_confidence=True)
    assert policy.on_result(
        success=True, low_confidence=True, strong_relocalize=False,
        next_mode="WEAK_TRACK", frame_index=4, now=1.0,
    ) == "ENGAGE_LOW_CONF"
    assert policy.on_result(
        success=True, low_confidence=False, strong_relocalize=False,
        next_mode="TRACK", frame_index=4, now=2.0,
    ) == "RELEASE_FIX"
    assert not policy.active


def test_a_confident_fix_clears_the_low_confidence_streak():
    policy = LostHoldPolicy(low_confidence_results=2, hold_on_low_confidence=True)
    policy.on_result(success=True, low_confidence=True, strong_relocalize=False,
                     next_mode="WEAK_TRACK", frame_index=1, now=1.0)
    policy.on_result(success=True, low_confidence=False, strong_relocalize=False,
                     next_mode="TRACK", frame_index=2, now=2.0)
    assert policy.low_streak == 0
    assert policy.on_result(
        success=True, low_confidence=True, strong_relocalize=False,
        next_mode="WEAK_TRACK", frame_index=3, now=3.0,
    ) is None
    assert not policy.active
