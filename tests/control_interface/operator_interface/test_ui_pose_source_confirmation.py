"""Desktop live poses no longer pause AUTO for source confirmation."""

from types import SimpleNamespace

import numpy as np

import operator_tick
from live_localizer_worker import _build_success_payload
from operator_rendering import classify_localization_health


def _worker_result(
    *, confirming: bool, weak: bool, hold: bool = False,
    camera_axes=None, direct_status="FAST_TRACK", camera_attitude_kind=None,
    prediction=None,
):
    pose = SimpleNamespace(x=1., y=2., z=3., yaw=0.4, stamp=10.)
    tracker = SimpleNamespace(last_info={
        "mode": "WEAK_TRACK" if weak else "TRACK",
        "next_mode": "WEAK_TRACK" if weak else "TRACK",
        "weak": weak, "inliers": 80, "inlier_ratio": 1.,
        "direct_status": direct_status,
        "camera_axes_world": camera_axes,
        "camera_forward_world": camera_axes[2] if camera_axes is not None else None,
        "camera_attitude_kind": camera_attitude_kind,
        "pose_source_confirming": confirming,
        "confidence_hold_active": hold,
    })
    if prediction is not None:
        pose = None
        tracker.last_info.update(prediction)
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


def test_worker_estimated_camera_metadata_survives_typed_contract_and_adapter():
    from localization_contract import LocalizationResult
    from localization_result_ui import estimated_camera_display_pose
    from operator_rendering import weak_pose_source_kind

    axes = [[0., 0., -1.], [0., -1., 0.], [1., 0., 0.]]
    for status, kind, attitude_kind, display_attitude in (
        ("VO_ONLY", "KLT", "full", "full"),
        ("IMU_BRIDGE", "IMU", "heading_only", "heading"),
    ):
        payload = _worker_result(
            confirming=False, weak=True, hold=True, camera_axes=axes,
            direct_status=status, camera_attitude_kind=attitude_kind,
        )
        typed = LocalizationResult.from_payload(payload, now_mono_ns=11_000_000_000)
        assert typed["camera_attitude_kind"] == attitude_kind
        assert typed["camera_axes_world"] == axes
        assert weak_pose_source_kind(typed) == kind
        display = estimated_camera_display_pose(typed, now=10.2)
        assert display is not None
        assert display.attitude == display_attitude
        assert not display.stale
        # The worker contract encodes weakness in mode/next_mode rather than
        # publishing the adapter's optional "weak" boolean.
        assert typed["mode"] == typed["next_mode"] == "WEAK_TRACK"
        assert classify_localization_health(typed) == "DEGRADED"
        control_pose = operator_tick._make_autonomy_pose(typed, np.array([1., 2., 3.]), 0.4, 10.)
        assert not control_pose.map_confirmed


def test_prediction_placeholder_yaw_never_becomes_camera_orientation():
    from localization_result_ui import estimated_camera_display_pose

    for yaw, expected in ((None, "none"), (float("nan"), "none"), (0.7, "heading")):
        payload = _worker_result(
            confirming=False, weak=True, prediction={
                "prediction_valid": True, "pose_status": "PREDICTED_ONLY",
                "predicted_center": [1., 2., 3.], "predicted_yaw": yaw,
            },
        )
        snapshot = estimated_camera_display_pose(payload, now=10.2)
        assert snapshot is not None
        assert snapshot.attitude == expected
        assert snapshot.camera_axes is snapshot.camera_forward is None
        if expected == "none":
            assert np.isnan(snapshot.pose[3])
