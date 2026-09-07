from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from trajectory_plausibility import (  # noqa: E402
    TrajectoryWindow,
    vote_allow,
)


#: The active river map's calibrated 0.40*S jump budget, used where a test
#: needs production-scale numbers instead of a round one.
RIVER_MAX_JUMP = 0.9338766098022462


def _walk(n=5, step=0.05, dt=0.1, start=100.0):
    w = TrajectoryWindow(max_points=8, max_age_s=3.0)
    for i in range(n):
        w.append((i * step, 0.0, 0.0), 0.0, start + i * dt)
    return w, start + n * dt


def test_allows_normal_walk():
    w, now = _walk()
    allow, reason = vote_allow(w, (5 * 0.05, 0.0, 0.0), 0.0, now, max_jump=2.0)
    assert allow is True
    assert reason == "ok"


def test_vetoes_teleport_under_max_jump():
    # 1.2 is under max_jump=2.0, so the single-step gate allows it; the
    # median step here is 0.05, so 1.2 >> 4 * median must veto.
    w, now = _walk()
    allow, reason = vote_allow(w, (4 * 0.05 + 1.2, 0.0, 0.0), 0.0, now, max_jump=2.0)
    assert allow is False
    assert reason == "jump_vs_median"


def test_allows_fast_but_consistent_flight():
    # Every step is large: median is large too, so no veto (protects legal
    # fast flight from the median check).
    w, now = _walk(n=5, step=0.8, dt=0.1)
    allow, _ = vote_allow(w, (5 * 0.8, 0.0, 0.0), 0.0, now, max_jump=2.0)
    assert allow is True


def test_vetoes_yaw_flip():
    w, now = _walk()
    allow, reason = vote_allow(w, (5 * 0.05, 0.0, 0.0), math.pi, now, max_jump=2.0)
    assert allow is False
    assert reason == "yaw_rate"


def test_single_frame_yaw_noise_does_not_veto():
    """Regression: measured P168/P119 false kills.

    Single-frame PnP yaw carried 15-21 deg of noise between adjacent frames
    at dt=0.125 s (120-170 deg/s apparent rate) on frames that were healthy
    (60-137 inliers, step < 0.09). Differencing against the last point alone
    vetoed them; the median baseline must not.
    """
    w = TrajectoryWindow(max_points=8, max_age_s=3.0)
    stamps = [100.0 + i * 0.125 for i in range(6)]
    noise = [0.0, 0.30, -0.28, 0.31, -0.29, 0.36]  # rad, ~17 deg peak
    for i, (s, n) in enumerate(zip(stamps, noise)):
        w.append((i * 0.02, 0.0, 0.0), n, s)
    allow, reason = vote_allow(
        w, (6 * 0.02, 0.0, 0.0), -0.30, stamps[-1] + 0.125, max_jump=0.9338766098022462
    )
    assert allow is True, reason


def test_short_baseline_skips_yaw_check():
    # Two points 0.1 s apart: baseline below MIN_YAW_BASELINE_S -> allow,
    # so the voter never judges yaw off a span too short to average noise.
    w = TrajectoryWindow(max_points=8, max_age_s=3.0)
    for i in range(3):
        w.append((i * 0.01, 0.0, 0.0), 0.0, 100.0 + i * 0.05)
    allow, _ = vote_allow(w, (0.03, 0.0, 0.0), math.pi, 100.15, max_jump=2.0)
    assert allow is True


def test_vetoes_reversal_teleport():
    w = TrajectoryWindow(max_points=8, max_age_s=30.0)
    w.append((0.0, 0.0, 0.0), 0.0, 100.0)
    w.append((0.6, 0.0, 0.0), 0.0, 100.1)
    w.append((1.2, 0.0, 0.0), 0.0, 100.2)
    # Back to the middle: 180 deg reversal, both legs significant vs max_jump.
    allow, reason = vote_allow(w, (0.6, 0.0, 0.0), 0.0, 100.3, max_jump=2.0)
    assert allow is False
    assert reason == "reversal"


def test_hover_jitter_never_trips_reversal():
    w = TrajectoryWindow(max_points=8, max_age_s=30.0)
    w.append((0.0, 0.0, 0.0), 0.0, 100.0)
    w.append((0.01, 0.0, 0.0), 0.0, 100.1)
    allow, _ = vote_allow(w, (0.0, 0.0, 0.0), 0.0, 100.2, max_jump=2.0)
    assert allow is True


def test_vetoes_vertical_jump():
    # Fast horizontal walk, so the speed check has a high median and stays
    # quiet; the candidate's motion is almost entirely vertical, which is a
    # floor/ceiling relocalization rather than flight.
    w, now = _walk(n=5, step=0.5, dt=0.1)
    allow, reason = vote_allow(w, (4 * 0.5 + 0.05, 0.0, 1.1), 0.0, now, max_jump=2.0)
    assert allow is False
    assert reason == "vertical"


def test_stale_history_allows():
    w = TrajectoryWindow(max_points=8, max_age_s=3.0)
    w.append((0.0, 0.0, 0.0), 0.0, 100.0)
    w.append((0.05, 0.0, 0.0), 0.0, 100.1)
    w.append((0.10, 0.0, 0.0), 0.0, 100.2)
    # 10 s later: history expired (LOST-boundary re-acquisition) -> allow.
    allow, reason = vote_allow(w, (50.0, 0.0, 0.0), 3.0, 110.2, max_jump=2.0)
    assert allow is True
    assert reason == "too_few_points"


def test_too_few_points_allows():
    w = TrajectoryWindow(max_points=8, max_age_s=3.0)
    w.append((0.0, 0.0, 0.0), 0.0, 100.0)
    allow, _ = vote_allow(w, (50.0, 0.0, 0.0), 3.0, 100.1, max_jump=2.0)
    assert allow is True


@pytest.mark.parametrize("bad", [None, (math.nan, 0.0, 0.0), (0.0, math.inf, 0.0)])
def test_nonfinite_candidate_allows(bad):
    w, now = _walk()
    allow, _ = vote_allow(w, bad, 0.0, now, max_jump=2.0)
    assert allow is True


def test_bad_max_jump_allows():
    w, now = _walk()
    allow, _ = vote_allow(w, (99.0, 0.0, 0.0), 0.0, now, max_jump=0.0)
    assert allow is True


def test_fault_injection_detection_vs_false_kill():
    # 200 clean accepts + 20 teleports injected just under max_jump.
    detected = 0
    for i in range(20):
        w, now = _walk(n=5, step=0.05, dt=0.1, start=100.0 + i * 10.0)
        allow, _ = vote_allow(w, (4 * 0.05 + 1.2, 0.0, 0.0), 0.0, now, max_jump=2.0)
        detected += not allow
    assert detected / 20 >= 0.95
    false_kills = 0
    for i in range(200):
        w, now = _walk(n=5, step=0.05, dt=0.1, start=500.0 + i * 10.0)
        allow, _ = vote_allow(w, (5 * 0.05, 0.0, 0.0), 0.0, now, max_jump=2.0)
        false_kills += not allow
    assert false_kills / 200 <= 0.005


def test_long_gap_after_misses_is_not_judged():
    """Regression: the 2026-09-06 seven-video gate failure.

    After a run of missed frames the previous accept is 1-2 s old. Measured
    on P167 / 河濱_P117, a legal 0.3-0.7 map-unit move across such a gap looked
    like a teleport against an adjacent-frame median, the veto then blocked
    the frame that would have refreshed the window, and the stale window
    vetoed the next accept too (-35 and -30 successes). A gap longer than
    MAX_CONTINUITY_GAP_S must not be judged at all.
    """
    w, _ = _walk(n=5, step=0.02, dt=0.125, start=100.0)
    late = 100.0 + 4 * 0.125 + 1.0
    allow, reason = vote_allow(w, (0.08 + 0.35, 0.0, 0.0), 0.0, late, max_jump=0.9338766098022462)
    assert allow is True
    assert reason == "gap_too_long"


def test_climb_across_a_second_is_not_vetoed():
    # Same measured case, vertical flavour: 0.31 up over 1 s with a small
    # horizontal step is a climb, not a floor jump.
    w, _ = _walk(n=5, step=0.02, dt=0.125, start=200.0)
    late = 200.0 + 4 * 0.125 + 1.0
    allow, _ = vote_allow(w, (0.08, 0.0, 0.31), 0.0, late, max_jump=0.9338766098022462)
    assert allow is True


def test_consecutive_vetoes_stand_down():
    """A voter must never be able to sustain itself on its own refusals."""
    w = TrajectoryWindow(max_points=8, max_age_s=3.0)
    for i in range(5):
        w.append((i * 0.5, 0.0, 0.0), 0.0, 300.0 + i * 0.1)
    now = 300.0 + 5 * 0.1
    candidate = (4 * 0.5 + 0.05, 0.0, 1.1)
    reasons = []
    for _ in range(4):
        allow, reason = vote_allow(w, candidate, 0.0, now, max_jump=2.0)
        reasons.append(reason)
        if not allow:
            w.note_veto()
    assert reasons[:2] == ["vertical", "vertical"]
    assert reasons[2] == "stand_down"
    assert len(w) == 0


def test_speed_semantics_not_distance():
    """Same distances, different frame spacing -> different verdicts.

    Both windows advance at the same 0.5 map-units/s median speed and both
    candidates move the same 0.5 units. Only the elapsed time differs: 0.1 s
    (5 units/s, ten times the median -> teleport) versus 0.4 s (1.25 units/s,
    ordinary motion). A distance-only check cannot tell them apart, which is
    exactly the defect the seven-video gate exposed.
    """
    fast = TrajectoryWindow(max_points=8, max_age_s=3.0)
    for i in range(5):
        fast.append((i * 0.05, 0.0, 0.0), 0.0, 400.0 + i * 0.1)
    allow_fast, reason_fast = vote_allow(
        fast, (4 * 0.05 + 0.5, 0.0, 0.0), 0.0, 400.0 + 5 * 0.1, max_jump=RIVER_MAX_JUMP
    )
    slow = TrajectoryWindow(max_points=8, max_age_s=3.0)
    for i in range(5):
        slow.append((i * 0.2, 0.0, 0.0), 0.0, 500.0 + i * 0.4)
    allow_slow, reason_slow = vote_allow(
        slow, (4 * 0.2 + 0.5, 0.0, 0.0), 0.0, 500.0 + 5 * 0.4, max_jump=RIVER_MAX_JUMP
    )
    assert allow_fast is False, reason_fast
    assert reason_fast == "jump_vs_median"
    assert allow_slow is True, reason_slow
