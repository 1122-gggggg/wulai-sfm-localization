"""Mapping state machine and the LOCALIZE gate on session mapping."""
from __future__ import annotations

import numpy as np
import pytest

from test_workmap_observations import _pixels, _scene
from test_workmap_promotion import _online_tracker  # noqa: F401  (fixture)
from two_rate_tracker import SeedProvenance
from workmap_state import (
    DETACH_FRAMES,
    LOCALIZE_HOLD_FRAMES,
    MappingInputs,
    MappingState,
    MappingStateMachine,
    frozen_ratio,
)


def _inputs(*, map_inliers=100, n_corr=100, status="FAST_TRACK", pose_ok=True,
            hull=0.5, detached_open=False, relocalizing=False, merge_pending=False):
    return MappingInputs(
        pose_ok=pose_ok, status=status, map_inliers=map_inliers, n_corr=n_corr,
        hull_coverage=hull, detached_open=detached_open,
        relocalizing=relocalizing, merge_pending=merge_pending, strong_inliers=80, min_inliers=20,
    )


def _localize(machine):
    for _ in range(LOCALIZE_HOLD_FRAMES):
        machine.update(_inputs())
    assert machine.state is MappingState.LOCALIZE


def test_localize_needs_eight_consecutive_dominant_frames():
    machine = MappingStateMachine()
    for _ in range(LOCALIZE_HOLD_FRAMES - 1):
        assert machine.update(_inputs()) is MappingState.ANCHORED_EXPAND
    assert machine.update(_inputs()) is MappingState.LOCALIZE


def test_a_weak_frame_resets_the_localize_entry_run():
    machine = MappingStateMachine()
    for _ in range(LOCALIZE_HOLD_FRAMES - 1):
        machine.update(_inputs())
    machine.update(_inputs(n_corr=400))                           # R_G = 100 / 400
    for _ in range(LOCALIZE_HOLD_FRAMES - 1):
        assert machine.update(_inputs()) is MappingState.ANCHORED_EXPAND
    assert machine.update(_inputs()) is MappingState.LOCALIZE


def test_localize_has_hysteresis_between_entry_and_exit_ratio():
    machine = MappingStateMachine()
    _localize(machine)
    dominant = _inputs(map_inliers=150, n_corr=200)               # R_G = 0.75
    weaker = _inputs(map_inliers=150, n_corr=230)                 # R_G = 0.65
    assert frozen_ratio(dominant) == pytest.approx(0.75)
    assert machine.update(dominant) is MappingState.LOCALIZE
    assert machine.update(weaker) is MappingState.ANCHORED_EXPAND


def test_localize_needs_hull_coverage_and_strong_support():
    machine = MappingStateMachine()
    for _ in range(2 * LOCALIZE_HOLD_FRAMES):
        assert machine.update(_inputs(hull=0.1)) is MappingState.ANCHORED_EXPAND
    for _ in range(2 * LOCALIZE_HOLD_FRAMES):
        assert machine.update(_inputs(map_inliers=60, n_corr=60)) is MappingState.ANCHORED_EXPAND


def test_detach_is_due_on_the_twelfth_unanchored_frame():
    machine = MappingStateMachine()
    machine.update(_inputs())
    due = []
    for _ in range(DETACH_FRAMES):
        machine.update(_inputs(status="VO_ONLY", map_inliers=0, n_corr=60))
        due.append(machine.detach_due)
    assert due == [False] * (DETACH_FRAMES - 1) + [True]
    assert machine.state is MappingState.DETACHED_EXPAND
    machine.update(_inputs(status="VO_ONLY", map_inliers=0, n_corr=60))
    assert not machine.detach_due
    assert machine.update(_inputs()) is MappingState.ANCHORED_EXPAND


def test_short_dropout_keeps_the_previous_expanding_state():
    machine = MappingStateMachine()
    machine.update(_inputs())
    for _ in range(DETACH_FRAMES - 1):
        assert machine.update(_inputs(pose_ok=False, status="NO_POSE", map_inliers=0)) is (
            MappingState.ANCHORED_EXPAND
        )


def test_submap_driven_states_override_the_coverage_rules():
    machine = MappingStateMachine()
    _localize(machine)
    assert machine.update(_inputs(merge_pending=True, relocalizing=True, detached_open=True)) is MappingState.MERGE
    assert machine.update(_inputs(relocalizing=True, detached_open=True)) is MappingState.RELOCALIZE
    assert machine.update(_inputs(detached_open=True)) is MappingState.DETACHED_EXPAND


def _seed_one_batch(tracker):
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:15] + [0.07, 0.03, 0.18]
    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = _pixels(tracker, new_xyz, seed_pose).copy()
    tracker._vo_xy = _pixels(tracker, new_xyz, current_pose).copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    tracker._vo_seed_meta = {
        0: SeedProvenance(ordinal=0, stamp=1.0, source_epoch=tracker._source_epoch, admissible=True)
    }
    tracker.approve_capture(
        current_pose, status="FAST_TRACK", map_inliers=100, n_corr=120, ordinal=8, stamp=1.25
    )
    return current_pose


def _count_keyframes(tracker):
    calls = []
    original = tracker.session_workmap.record_keyframe

    def spy(*args, **kwargs):
        calls.append(kwargs.get("ordinal"))
        return original(*args, **kwargs)

    tracker.session_workmap.record_keyframe = spy
    return calls


def test_localize_state_maps_nothing(_online_tracker):  # noqa: F811
    tracker = _online_tracker
    pose = _seed_one_batch(tracker)
    keyframes = _count_keyframes(tracker)
    tracker._mapping_state = MappingState.LOCALIZE
    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), pose)
    assert keyframes == []
    assert not tracker._workmap_landmarks and not tracker.session_workmap.landmarks
    assert len(tracker._live_ids)                                  # temporary VO still runs


def test_expanding_state_maps_the_same_batch(_online_tracker):  # noqa: F811
    tracker = _online_tracker
    pose = _seed_one_batch(tracker)
    keyframes = _count_keyframes(tracker)
    tracker._mapping_state = MappingState.ANCHORED_EXPAND
    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), pose)
    assert keyframes == [8]
    assert len(tracker._workmap_landmarks) == 15


def test_live_cap_policy_follows_the_state(_online_tracker):  # noqa: F811
    tracker = _online_tracker
    tracker._mapping_state = MappingState.LOCALIZE
    assert tracker._live_cap_policy() == "frozen_first"
    for state in set(MappingState) - {MappingState.LOCALIZE}:
        tracker._mapping_state = state
        assert tracker._live_cap_policy() == "spread"
