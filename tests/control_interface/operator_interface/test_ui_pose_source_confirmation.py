"""Desktop live poses no longer pause AUTO for source confirmation."""

from types import SimpleNamespace

import numpy as np

import operator_tick
from live_localizer_worker import _build_success_payload
from operator_rendering import classify_localization_health


def _worker_result(*, confirming: bool, weak: bool, hold: bool = False):
    pose = SimpleNamespace(x=1., y=2., z=3., yaw=0.4, stamp=10.)
    tracker = SimpleNamespace(last_info={
        "mode": "WEAK_TRACK" if weak else "TRACK",
        "next_mode": "WEAK_TRACK" if weak else "TRACK",
        "weak": weak, "inliers": 80, "inlier_ratio": 1.,
        "direct_status": "FAST_TRACK",
        "pose_source_confirming": confirming,
        "confidence_hold_active": hold,
    })
    return _build_success_payload(
        args=SimpleNamespace(force_track_bench=False),
        seq=1, pose=pose, tracker=tracker, capture_stamp=10.1,
        worker_read_done_mono_ns=10_100_000_000, worker_read_done_mono=10.1,
        worker_core_start_mono=10.11, worker_core_start_mono_ns=10_110_000_000,
        worker_core_done_mono_ns=10_120_000_000,
        wall_t0=0., core_t0=0., gpu_timing_profiled=False, gpu_span_ms=None,
        tracker_variant="direct", force_track_label=None,
        force_track_ref_resolved=None, force_track_seed_ms=0.,
        benchmark_mode_requested="normal", benchmark_mode_active="normal",
        benchmark_setup_ms=0., benchmark_seeded=False,
    )


def test_ui_does_not_pause_auto_for_source_confirmation():
    pauses = []
    app = SimpleNamespace(
        _live_pending=object(),
        _pause_integrated_auto=pauses.append,
    )
    xyz = np.array([0.113277, 0.0, 0.0])
    assert operator_tick._live_pose_is_continuous(app, xyz) is True
    assert app._live_pending is None
    assert pauses == []


def test_worker_confirmation_reaches_autonomy_without_becoming_a_new_observation():
    xyz = np.array([1., 2., 3.])
    for hold in (False, True):
        pending = _worker_result(confirming=True, weak=True, hold=hold)
        assert pending["pose_source_confirming"] is True
        assert classify_localization_health(pending) == "DEGRADED"
        pending_pose = operator_tick._make_autonomy_pose(pending, xyz, 0.4, 10.)
        assert (pending_pose.x, pending_pose.y, pending_pose.z, pending_pose.yaw) == (
            1., 2., 3., 0.4,
        )
        assert not pending_pose.position_observed
        assert not pending_pose.map_confirmed

    confirmed = _worker_result(confirming=False, weak=False)
    assert confirmed["pose_source_confirming"] is False
    assert classify_localization_health(confirmed) == "OK"
    confirmed_pose = operator_tick._make_autonomy_pose(confirmed, xyz, 0.4, 10.)
    assert confirmed_pose.position_observed
    assert confirmed_pose.map_confirmed

    weak_observed = _worker_result(confirming=False, weak=True, hold=True)
    observed_pose = operator_tick._make_autonomy_pose(weak_observed, xyz, 0.4, 10.)
    assert observed_pose.position_observed
    assert not observed_pose.map_confirmed
