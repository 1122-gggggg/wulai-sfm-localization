from types import SimpleNamespace

import numpy as np

from direct_localizer_adapter import DirectTrackerAdapter, RuntimeState
from localization_continuity import PoseSourceConfirmation
from pose_types import Pose


def test_direct_jump_holds_original_pose_until_distinct_confirmation():
    adapter = object.__new__(DirectTrackerAdapter)
    adapter.profile = SimpleNamespace(max_jump_u=1.5)
    adapter.state = RuntimeState()
    adapter._pose_continuity = PoseSourceConfirmation()
    adapter._pose_centers = []
    adapter._reseed_confirming = False
    adapter.trk = SimpleNamespace(_speed_history=[100.0])

    def offer(x, stamp, status="FAST_TRACK", weak=False):
        pose = Pose(x=x, y=0, z=0, yaw=0, stamp=stamp)
        info = {"center": np.array([x, 0, 0])}
        accepted, mode, weak, pending = adapter._confirm_pose(pose, info, "TRACK", weak, status)
        adapter._advance_state(accepted, info, mode, status)
        return accepted, weak, pending

    original, _, _ = offer(0.0, 10.0)
    held, weak, pending = offer(2.0, 10.1)
    assert held is original and weak and pending
    assert not adapter.trk._speed_history
    assert offer(2.0, 10.1)[0] is original
    accepted, weak, pending = offer(2.0, 10.2)
    assert accepted.x == 2.0 and not weak and not pending
    # RELOC_SEED and the following FAST_TRACK share one map-fix source: the
    # second consistent capture releases without waiting for two more.
    assert offer(2.01, 10.3, "RELOC_SEED")[2]
    assert offer(2.01, 10.4)[2] is False


def test_only_the_run_a_reseed_starts_is_reseed_confirming():
    adapter = object.__new__(DirectTrackerAdapter)
    adapter.profile = SimpleNamespace(max_jump_u=1.5)
    adapter.state = RuntimeState()
    adapter._pose_continuity = PoseSourceConfirmation()
    adapter._pose_centers = []
    adapter._reseed_confirming = False
    adapter.trk = SimpleNamespace(_speed_history=[])

    def offer(x, stamp, status="FAST_TRACK", weak=False):
        pose = Pose(x=x, y=0, z=0, yaw=0, stamp=stamp)
        info = {"center": np.array([x, 0, 0])}
        accepted, mode, _weak, _pending = adapter._confirm_pose(pose, info, "TRACK", weak, status)
        adapter._advance_state(accepted, info, mode, status)
        return adapter._reseed_confirming

    assert offer(0.0, 10.0) is False
    # Flight 2026-09-15 14:25: the seed frame is held; the next consistent
    # FAST_TRACK releases the shared map-fix source and ends the reseed run.
    assert offer(0.01, 10.1, "RELOC_SEED") is True
    assert offer(0.01, 10.2) is False
    assert adapter.state.last_pose is not None and adapter.state.last_pose.x == 0.01
    assert offer(0.01, 10.3) is False
    # A VO-only frame ends the reseed run even though confirmation continues.
    assert offer(0.02, 10.4, "RELOC_SEED") is True
    assert offer(0.02, 10.5, "VO_ONLY", weak=True) is False
    assert offer(0.02, 10.6) is True
    assert offer(0.02, 10.7) is False
    # A jump confirmation is not a reseed.
    assert offer(2.0, 10.8) is True


def test_weak_jump_cannot_bypass_the_shared_continuity_gate():
    gate = PoseSourceConfirmation()
    assert gate.accept(np.zeros(3), 10.0, reliable=True, radius=0.02)
    assert not gate.accept(np.ones(3), 10.1, reliable=False, radius=0.02)
    assert gate.pending is not None
    assert not gate.accept(np.ones(3), 10.2, reliable=True, radius=0.02)
    assert gate.accept(np.ones(3), 10.3, reliable=True, radius=0.02)


def test_direct_worker_pose_keeps_accepted_camera_basis_while_a_jump_is_pending():
    adapter = object.__new__(DirectTrackerAdapter)
    adapter.profile = SimpleNamespace(
        fast_loop=SimpleNamespace(resolution=(8, 8)),
        reloc=SimpleNamespace(top_k=2, frozen_pnp_thresholds={"strong_inliers": 60}),
        max_jump_u=1.5,
    )
    adapter.state = RuntimeState()
    adapter._pose_continuity = PoseSourceConfirmation()
    adapter._pose_centers = []
    adapter._reseed_confirming = False
    adapter._accepted_camera_axes_world = None
    adapter._fused_sample = None
    old_rotation = np.eye(3)
    new_rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    samples = [
        (0.0, 10.0, old_rotation, np.pi / 2),
        (2.0, 10.1, new_rotation, 0.0),
        (2.0, 10.2, new_rotation, 0.0),
    ]

    def step(_gray, _stamp):
        x, stamp, rotation, yaw = samples.pop(0)
        assert _stamp == stamp
        return {
            "status": "FAST_TRACK",
            "ok": True,
            "center": np.array([x, 0.0, 0.0]),
            "R": rotation,
            "yaw": yaw,
            "inliers": 80,
            "map_inliers": 80,
            "vo_inliers": 0,
            "live_points": 80,
            "n_corr": 80,
            "loop_ms": 5.0,
            "track_ms": 2.0,
            "pnp_ms": 1.0,
            "vo_ms": 0.0,
            "dead_reckon_age": 0,
            "reloc_status": "IDLE",
            "handover_points": 0,
            "handover_dropped": 0,
            "reloc_ms": None,
            "reloc_delivered": False,
            "reloc_busy": False,
            "vo_candidates": 0,
            "step": None,
        }

    adapter.trk = SimpleNamespace(
        map_frame=None,
        step=step,
        _speed_history=[1.0],
    )
    frame = np.zeros((8, 8), dtype=np.uint8)
    first = adapter.localize_frame(frame, capture_stamp=10.0)
    assert (first.x, first.yaw) == (0.0, np.pi / 2)
    assert adapter.last_info["camera_axes_world"] == old_rotation.tolist()
    held = adapter.localize_frame(frame, capture_stamp=10.1)
    held_info = adapter.last_info
    assert held is first
    assert held_info["pose_source_confirming"] is True
    assert held_info["mode"] == "WEAK_TRACK" and held_info["weak"] is True
    assert held_info["camera_axes_world"] == old_rotation.tolist()
    assert held_info["camera_forward_world"] == old_rotation[2].tolist()
    assert not adapter.trk._speed_history
    confirmed = adapter.localize_frame(frame, capture_stamp=10.2)
    assert (confirmed.x, confirmed.yaw) == (2.0, 0.0)
    assert adapter.last_info["camera_axes_world"] == new_rotation.tolist()
    assert adapter.last_info["camera_forward_world"] == new_rotation[2].tolist()
    assert adapter.last_info["mode"] == "TRACK"
    assert adapter.last_info["weak"] is False
    assert adapter.last_info["pose_source_confirming"] is False
