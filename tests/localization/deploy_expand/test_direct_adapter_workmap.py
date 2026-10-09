"""Single-front-end publication, weak-pose rules, and mapping failure isolation."""
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from direct_localizer_adapter import DirectTrackerAdapter, RuntimeState
from live_provider import LiveMapEDMProvider, RelocFix
from localization_continuity import PoseSourceConfirmation
from pose_types import Pose


class Tracker:
    def __init__(self):
        self._speed_history = []
        self.map_frame = None
        self.calls = 0
        self.resets = 0
        self.closed = False
        self.mapping_enabled = True
        self.mapping_error = None
        self.raise_on_step = None
        self.info = dict(
            status="FAST_TRACK", ok=True, center=np.array([1., 1., 1.]), yaw=0.,
            R=np.eye(3), inliers=100, n_corr=100, loop_ms=1., track_ms=1., pnp_ms=1.,
            vo_ms=0., map_inliers=100, vo_inliers=0, live_points=100, dead_reckon_age=0,
            reloc_status="LOCALIZED_STRONG", handover_points=0, handover_dropped=0,
            reloc_ms=0., reloc_delivered=False, step=0., reference_scope="global",
        )

    def step(self, gray, stamp):
        self.calls += 1
        if self.raise_on_step is not None:
            raise self.raise_on_step
        return dict(self.info)

    def close(self):
        self.closed = True

    def reset(self):
        self.resets += 1

    def disable_mapping(self, reason):
        self.mapping_enabled = False
        self.mapping_error = reason


def adapter(trk):
    obj = object.__new__(DirectTrackerAdapter)
    obj.profile = SimpleNamespace(
        fast_loop=SimpleNamespace(resolution=(16, 16)),
        reloc=SimpleNamespace(top_k=2, frozen_pnp_thresholds={"strong_inliers": 80}),
        max_jump_u=1.5,
    )
    obj.trk = trk
    obj.state = RuntimeState()
    obj._pose_continuity = PoseSourceConfirmation()
    obj._pose_centers = []
    obj._reseed_confirming = False
    obj._last_output_pose = None
    obj._fused_sample = None
    obj._workmap_diagnostics = {}
    obj._last_info = {}
    return obj


def test_workmap_points_delta_drains_the_display_queue_in_bounded_batches():
    from direct_localizer_adapter import WORKMAP_POINTS_EVERY, WORKMAP_POINTS_PER_UPDATE

    queue = [(-key, [float(key), 0., 0.]) for key in range(1, WORKMAP_POINTS_PER_UPDATE + 3)]

    def pop(limit):
        batch, queue[:] = queue[:limit], queue[limit:]
        return [k for k, _ in batch], [xyz for _, xyz in batch], [[1, 2, 3]] * len(batch), 99

    trk = Tracker()
    trk.session_workmap = SimpleNamespace(pop_display_points=pop)
    obj = adapter(trk)
    deltas = [obj._workmap_points_delta() for _ in range(WORKMAP_POINTS_EVERY * 3)]
    sent = [d for d in deltas if d]
    assert [len(d["ids"]) for d in sent] == [WORKMAP_POINTS_PER_UPDATE, 2]
    assert sent[0]["total"] == 99 and sent[0]["rgb"][0] == [1, 2, 3] and sent[1]["ids"] == [-(WORKMAP_POINTS_PER_UPDATE + 1), -(WORKMAP_POINTS_PER_UPDATE + 2)]


def _weak_walk(monkeypatch, no_freeze):
    import direct_localizer_adapter as module

    monkeypatch.setattr(module, "ORIGINAL_NO_FREEZE", no_freeze)
    trk = Tracker()
    obj = adapter(trk)
    image = np.zeros((16, 16), np.uint8)
    obj.localize_frame(image, 10.)                                   # strong fix at x = 1
    trk.info.update(status="VO_ONLY", map_inliers=0, vo_inliers=100)
    published, stamp = [], 10.
    for x in (1.4, 1.9, 2.5, 3.1, 3.7, 4.3):                         # drifts 3.3 u past the fix
        stamp += .1
        trk.info.update(center=np.array([x, 1., 1.]))
        pose = obj.localize_frame(image, stamp)
        published.append(None if pose is None else (pose.x, obj.last_info["pose_source_confirming"]))
    return published


def test_original_weak_pose_keeps_updating_past_the_old_freeze_radius(monkeypatch):
    published = _weak_walk(monkeypatch, True)
    assert [p[0] for p in published] == pytest.approx([1.4, 1.9, 2.5, 3.1, 3.7, 4.3])
    assert not any(confirming for _, confirming in published)


def test_original_weak_pose_still_freezes_when_the_switch_is_off(monkeypatch):
    published = _weak_walk(monkeypatch, False)
    assert published[-1][0] == pytest.approx(published[-2][0]) and published[-1][1]
    assert max(x for x, _ in published) < 3.0


def test_a_single_frame_jump_is_still_held_when_not_frozen(monkeypatch):
    import direct_localizer_adapter as module

    monkeypatch.setattr(module, "ORIGINAL_NO_FREEZE", True)
    trk = Tracker()
    obj = adapter(trk)
    image = np.zeros((16, 16), np.uint8)
    first = obj.localize_frame(image, 10.)
    trk.info.update(status="VO_ONLY", map_inliers=0, center=np.array([4., 1., 1.]))
    assert obj.localize_frame(image, 10.1) is first                  # 3 u in one frame: held


def test_stale_pose_at_large_monotonic_time_is_not_fresh():
    obj = adapter(Tracker())
    obj._confirm_pose = lambda *args: (Pose(1., 1., 1., 0., 1e8 - .01), "TRACK", False, False)
    obj.localize_frame(np.zeros((16, 16), np.uint8), 1e8)
    assert obj.last_info["workmap_output_source"] == "baseline"


def test_mapping_failure_keeps_the_front_end_publishing():
    trk = Tracker()
    obj = adapter(trk)
    image = np.zeros((16, 16), np.uint8)
    assert obj.localize_frame(image, 10.).x == 1.
    assert obj.last_info["workmap_mapping_healthy"]

    trk.raise_on_step = RuntimeError("writer exploded")
    assert obj.localize_frame(image, 10.1) is None
    assert obj.last_info["direct_status"] == "NO_POSE"
    assert not obj.last_info["workmap_mapping_healthy"]
    assert obj.last_info["workmap_mapping_error"] == "RuntimeError: writer exploded"
    assert trk.resets == 1

    trk.raise_on_step = None
    pose = obj.localize_frame(image, 10.2)
    assert pose is not None and pose.x == 1.
    assert trk.calls == 3
    assert not obj.last_info["workmap_mapping_healthy"]


def test_front_end_failure_after_mapping_is_disabled_propagates():
    trk = Tracker()
    trk.disable_mapping("earlier")
    trk.raise_on_step = RuntimeError("front end")
    with pytest.raises(RuntimeError, match="front end"):
        adapter(trk).localize_frame(np.zeros((16, 16), np.uint8), 10.)


@pytest.mark.parametrize("landmarks, source", [(3, "workmap"), (0, "baseline")])
def test_output_source_is_workmap_only_when_the_published_pose_used_session_landmarks(landmarks, source):
    trk = Tracker()
    trk.info["workmap_pose_landmarks"] = landmarks
    obj = adapter(trk)
    assert obj.localize_frame(np.zeros((16, 16), np.uint8), 10.) is not None
    assert obj.last_info["workmap_output_source"] == source
    assert obj.last_info["workmap_pose_landmarks"] == landmarks


def test_a_held_pose_is_not_labelled_workmap():
    trk = Tracker()
    trk.info["workmap_pose_landmarks"] = 3
    obj = adapter(trk)
    image = np.zeros((16, 16), np.uint8)
    obj.localize_frame(image, 10.)
    trk.info.update(center=np.array([9., 1., 1.]))                    # strong jump: held for confirmation
    obj.localize_frame(image, 10.1)
    assert obj.last_info["pose_source_confirming"]
    assert obj.last_info["workmap_output_source"] == "baseline"


def test_auxiliary_only_queries_do_not_change_frozen_retrieval_history(monkeypatch):
    import live_provider

    def fix(scope="global"):
        return replace(
            RelocFix.abstained("fixture", 1.),
            query_xy=np.array([[1., 2.]]), point3d_ids=np.array([7]),
            point_xyz=np.array([[3., 4., 5.]]), reference_scope=scope,
            workmap_snapshot={"nested": {"values": np.array([1, 2])}},
        )

    provider = object.__new__(LiveMapEDMProvider)
    provider._lock = threading.RLock()
    provider._index = object()
    provider._query_matches = {}
    provider._query_lifts = {}
    provider._workmap_descriptor_count = 1
    provider._last_strong_refs = (3, 7)
    provider._vpr = lambda: None
    monkeypatch.setattr(live_provider, "boq_descriptor_from_array", lambda *args: np.zeros(2))

    def global_query(*args):
        provider._last_strong_refs = (99,)
        raise AssertionError("workmap-scope request ran a global solve")

    provider._global_relocalization = global_query
    provider._workmap_relocalization = lambda *args: fix("workmap")
    result = provider.localize_array(np.zeros((8, 8), np.uint8), scope="workmap")
    assert result.reference_scope == "workmap"
    assert provider._last_strong_refs == (3, 7)
