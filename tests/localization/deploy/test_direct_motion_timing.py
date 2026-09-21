"""Motion must use a pose and image from the same exposure, in seconds."""

from types import SimpleNamespace

import numpy as np
import pytest

from two_rate_tracker import TwoRateTracker


def pose_at(x):
    return np.column_stack((np.eye(3), [-x, 0.0, 0.0]))


@pytest.fixture
def tracker():
    tracker = object.__new__(TwoRateTracker)
    tracker._reset_tracking_state()
    tracker.profile = SimpleNamespace(
        dead_reckon=SimpleNamespace(enabled=True, max_frames=300, max_age_s=10.0)
    )
    tracker._detect_features = lambda gray, occupied: np.zeros((8, 2))
    tracker._inside = lambda points: np.ones(len(points), dtype=bool)
    return tracker


def record(tracker, x, ordinal, stamp, status="FAST_TRACK"):
    pose = None if x is None else pose_at(x)
    center = None if x is None else np.array([x, 0.0, 0.0])
    tracker._step_bookkeeping(
        center, pose, ordinal, np.full((8, 8), ordinal, dtype=np.uint8), stamp, status
    )


def install_relative_probe(tracker):
    observed = {}

    def track_pair(previous, current, points):
        observed["image_pair"] = (int(previous[0, 0]), int(current[0, 0]))
        return points, np.ones(len(points), dtype=bool)

    def recover(previous, current, pose, step):
        observed["pose"] = pose.copy()
        observed["step"] = step
        return pose, 8

    tracker._tracker = SimpleNamespace(track_pair=track_pair)
    tracker._recover_relative = recover
    return observed


def test_no_pose_keeps_the_image_that_belongs_to_the_last_pose(tracker):
    record(tracker, 0.0, 0, 10.0)
    record(tracker, 0.1, 1, 10.1)
    record(tracker, None, 2, 10.2, "NO_POSE")
    observed = install_relative_probe(tracker)
    tracker._step_dead_reckon(np.full((8, 8), 3, dtype=np.uint8), 3, None, None, 10.3)

    assert observed["image_pair"] == (1, 3)
    np.testing.assert_array_equal(observed["pose"], pose_at(0.1))
    assert observed["step"] == pytest.approx(0.2)


@pytest.mark.parametrize("dt", [1 / 30, 1 / 15, 0.25])
def test_dropped_frames_scale_translation_by_elapsed_time(tracker, dt):
    record(tracker, 0.0, 0, 10.0)
    record(tracker, 0.1, 1, 10.1)
    observed = install_relative_probe(tracker)
    tracker._step_dead_reckon(np.zeros((8, 8), dtype=np.uint8), 2, None, None, 10.1 + dt)
    assert observed["step"] == pytest.approx(dt)


def test_relocalization_correction_is_not_a_velocity_measurement(tracker):
    record(tracker, 0.0, 0, 10.0)
    record(tracker, 0.1, 1, 10.1)
    record(tracker, 8.0, 2, 10.2, "RELOC_SEED")
    assert not tracker._speed_history
    record(tracker, 8.1, 3, 10.3)
    assert tracker._speed_history == pytest.approx([1.0])


def test_dead_reckoning_does_not_train_its_own_speed(tracker):
    record(tracker, 0.0, 0, 10.0)
    record(tracker, 0.1, 1, 10.1)
    record(tracker, 0.3, 2, 10.2, "DEAD_RECKON")
    record(tracker, 0.5, 3, 10.3, "DEAD_RECKON")
    assert tracker._speed_history == pytest.approx([1.0])


def test_time_limit_expires_even_if_failed_frames_do_not_age_dr(tracker):
    record(tracker, 0.0, 0, 10.0)
    record(tracker, 0.1, 1, 10.1)
    record(tracker, None, 2, 21.0, "NO_POSE")
    observed = install_relative_probe(tracker)
    result = tracker._step_dead_reckon(np.zeros((8, 8), dtype=np.uint8), 3, None, None, 21.1)
    assert result[0] is None
    assert not observed


@pytest.mark.parametrize("stamp", [10.1, 10.0])
def test_nonadvancing_capture_time_cannot_produce_a_dr_step(tracker, stamp):
    record(tracker, 0.0, 0, 10.0)
    record(tracker, 0.1, 1, 10.1)
    observed = install_relative_probe(tracker)
    result = tracker._step_dead_reckon(np.zeros((8, 8), dtype=np.uint8), 2, None, None, stamp)
    assert result[0] is None
    assert not observed


def test_missing_capture_stamp_is_not_a_fresh_pose():
    from direct_localizer_adapter import DirectTrackerAdapter

    adapter = object.__new__(DirectTrackerAdapter)
    adapter._last_info = {}
    called = []

    def forbidden_step(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("trk.step must not run without a capture stamp")

    adapter.trk = SimpleNamespace(step=forbidden_step)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    pose = adapter.localize_frame(frame, capture_stamp=None)
    assert pose is None
    assert called == []
    assert adapter.last_info["mode"] == "LOST"
    assert adapter.last_info["weak"] is True
    assert adapter.last_info["direct_status"] == "NO_POSE"


def test_direct_adapter_preserves_handover_capture_metadata():
    from direct_localizer_adapter import DirectTrackerAdapter, RuntimeState

    adapter = object.__new__(DirectTrackerAdapter)
    adapter.profile = SimpleNamespace(
        fast_loop=SimpleNamespace(resolution=(8, 8)),
        reloc=SimpleNamespace(
            top_k=2,
            frozen_pnp_thresholds={"strong_inliers": 80},
        ),
        max_jump_u=1.5,
    )
    adapter.trk = SimpleNamespace(map_frame=None)
    adapter.map_frame = None
    adapter._fused_sample = None
    adapter.state = RuntimeState()
    adapter._pose_continuity = SimpleNamespace()
    adapter._pose_centers = []
    adapter._reseed_confirming = False
    adapter._confirm_pose = lambda pose, info, mode, weak, status: (
        pose,
        mode,
        weak,
        False,
    )
    adapter._advance_state = lambda *_args: None

    info = {
        "status": "FAST_TRACK",
        "ok": True,
        "center": np.zeros(3),
        "yaw": 0.0,
        "R": np.eye(3),
        "inliers": 80,
        "map_inliers": 80,
        "vo_inliers": 0,
        "live_points": 80,
        "n_corr": 80,
        "loop_ms": 5.0,
        "track_ms": 2.0,
        "handover_ms": 0.2,
        "bookkeeping_ms": 0.1,
        "pnp_ms": 1.0,
        "vo_ms": 0.0,
        "gray_ms": 0.1,
        "map_constraint_age_s": 0.0,
        "dead_reckon_age": 0,
        "reloc_status": "LOCALIZED_STRONG",
        "handover_points": 120,
        "handover_dropped": 0,
        "handover_capture_age_s": 0.37,
        "reloc_ms": 160.0,
        "reloc_retrieval_ms": 10.0,
        "reloc_match_lift_ms": 140.0,
        "reloc_pnp_ms": 7.0,
        "reloc_reference_count": 2,
        "reloc_capture_stamp_mono": 10.0,
        "reloc_source_epoch": 4,
        "reloc_ordinal": 100,
        "reloc_delivered": True,
        "reloc_busy": False,
        "vo_candidates": 0,
        "step": None,
        "reference_names": ("r0",),
        "reloc_submitted": False,
    }
    adapter.trk = SimpleNamespace(map_frame=None, step=lambda _gray, _stamp: info)

    pose = adapter.localize_frame(np.zeros((8, 8, 3), dtype=np.uint8), capture_stamp=10.37)

    assert pose is not None
    assert adapter.last_info["handover_capture_age_s"] == pytest.approx(0.37)
    assert adapter.last_info["reloc_capture_stamp_mono"] == pytest.approx(10.0)
    assert adapter.last_info["reloc_source_epoch"] == 4
    assert adapter.last_info["reloc_ordinal"] == 100
