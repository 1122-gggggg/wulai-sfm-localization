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
    # A source change also needs confirmation even when the displacement is small.
    assert offer(2.01, 10.3, "RELOC_SEED")[2]
    assert offer(2.01, 10.4, "RELOC_SEED")[2] is False


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
    # Flight 2026-09-15 14:25: the seed frame and the next FAST_TRACK frame are held.
    assert offer(0.01, 10.1, "RELOC_SEED") is True
    assert offer(0.01, 10.2) is True
    assert offer(0.01, 10.3) is False
    # A VO-only frame ends the reseed run even though confirmation continues.
    assert offer(0.02, 10.4, "RELOC_SEED") is True
    assert offer(0.02, 10.5, "VO_ONLY", weak=True) is False
    assert offer(0.02, 10.6) is False
    assert offer(0.02, 10.7) is False
    # A jump confirmation is not a reseed.
    assert offer(2.0, 10.8) is False


def test_weak_jump_cannot_bypass_the_shared_continuity_gate():
    gate = PoseSourceConfirmation()
    assert gate.accept(np.zeros(3), 10.0, reliable=True, radius=0.02)
    assert not gate.accept(np.ones(3), 10.1, reliable=False, radius=0.02)
    assert gate.pending is not None
    assert not gate.accept(np.ones(3), 10.2, reliable=True, radius=0.02)
    assert gate.accept(np.ones(3), 10.3, reliable=True, radius=0.02)
