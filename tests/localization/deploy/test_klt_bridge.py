"""KLT-bridge scheduler gates (SFM_EDM_KLT_BRIDGE_INTERVAL).

The bridge skips the EDM match on non-keyframes and carries the pose with KLT.
It must stay off by default and refuse to engage outside steady TRACK or out of
a weak EDM anchor -- the naive version (no anchor gate) regressed P168 by ~100
successes. These tests pin the pre-``_track_klt_prior`` gate conditions.
"""
from __future__ import annotations

import numpy as np

from production_edm_tracker import EDMConfig, ProductionEDMTracker, RuntimeState


def _tracker(interval: int) -> ProductionEDMTracker:
    t = object.__new__(ProductionEDMTracker)
    t.cfg = EDMConfig()
    t.st = RuntimeState()
    t.st.state = "TRACK"
    t.st.center = np.zeros(3, dtype=float)
    t.st.last_capture_stamp = 1.0
    t._klt_bridge_interval = interval
    t._klt_bridge_run = 0
    t._klt_bridge_min_ratio = 0.66
    t._klt_bridge_max_consec = 0
    t._klt_bridge_drift_guard = True
    t._klt_anchor_inliers = 200
    t._klt_anchor_ratio = 0.9
    return t


def _mock_prior(center=(0.05, 0.05, 0.02), reproj=1.0, ratio=0.9):
    return lambda _g, _s: {
        "center": np.array(center, dtype=float),
        "yaw": 0.5,
        "R": np.eye(3),
        "cam_from_world": None,
        "tracked": 90,
        "inliers": 80,
        "reproj_rms": reproj,
        "inlier_ratio": ratio,
    }


def _fail_prior(*_a, **_k):  # would raise if the gate let us through
    raise AssertionError("_track_klt_prior should not be reached")


def test_disabled_by_default():
    t = _tracker(interval=0)
    t._track_klt_prior = _fail_prior
    assert t._try_klt_bridge(None, None, 1.1, 0.0) is None


def test_interval_one_is_off():
    t = _tracker(interval=1)
    t._track_klt_prior = _fail_prior
    assert t._try_klt_bridge(None, None, 1.1, 0.0) is None


def test_not_track_state():
    t = _tracker(interval=3)
    t._track_klt_prior = _fail_prior
    t.st.state = "WEAK_TRACK"
    assert t._try_klt_bridge(None, None, 1.1, 0.0) is None


def test_keyframe_forces_edm():
    t = _tracker(interval=3)
    t._track_klt_prior = _fail_prior
    t._klt_bridge_run = 2  # >= n - 1 -> keyframe
    assert t._try_klt_bridge(None, None, 1.1, 0.0) is None


def test_weak_anchor_blocks_bridge():
    t = _tracker(interval=3)
    t._track_klt_prior = _fail_prior
    t._klt_anchor_inliers = 40  # < max(2*weak_min_inliers, 60)
    assert t._try_klt_bridge(None, None, 1.1, 0.0) is None


def test_low_ratio_anchor_blocks_bridge():
    t = _tracker(interval=3)
    t._track_klt_prior = _fail_prior
    t._klt_anchor_ratio = 0.4
    assert t._try_klt_bridge(None, None, 1.1, 0.0) is None


def test_strong_anchor_reaches_prior_and_accepts():
    t = _tracker(interval=3)
    t._klt_prior_allowed = lambda _stamp: True
    t._store_visual_motion_cache = lambda *_a, **_k: None
    t.name_of = {}
    t.pose_guided = None
    t._track_klt_prior = _mock_prior(center=(0.05, 0.05, 0.02))
    info = t._try_klt_bridge(np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.uint8), 1.2, 0.0)
    assert info is not None
    assert info["ok"] is True
    assert info["candidate_mode"] == "klt_bridge"
    # Bridged frames carry no fresh EDM match: distinct status, still ok=True.
    assert info["pose_status"] == "KLT_BRIDGED"
    assert info["bridge"] is True
    assert info["state_in"] == "TRACK" and info["state_out"] == "TRACK"
    assert t._klt_bridge_run == 1  # clean bridge, drift guard did not trip
    assert np.allclose(t.st.center, [0.05, 0.05, 0.02])


def test_drift_guard_forces_edm_next_frame_on_big_step():
    t = _tracker(interval=4)  # cap = 3, so run would normally be 1 here
    t._klt_prior_allowed = lambda _stamp: True
    t._store_visual_motion_cache = lambda *_a, **_k: None
    t.name_of = {}
    t.pose_guided = None
    # step from origin ~1.9 map units > 0.50 * max_jump (1.0)
    t._track_klt_prior = _mock_prior(center=(1.1, 1.1, 1.1))
    info = t._try_klt_bridge(np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.uint8), 1.2, 0.0)
    assert info is not None and info["ok"] is True
    assert t._klt_bridge_run == 3  # bumped to cap -> next _try_klt_bridge returns None


def test_drift_guard_off_keeps_bridging():
    t = _tracker(interval=4)
    t._klt_bridge_drift_guard = False
    t._klt_prior_allowed = lambda _stamp: True
    t._store_visual_motion_cache = lambda *_a, **_k: None
    t.name_of = {}
    t.pose_guided = None
    t._track_klt_prior = _mock_prior(center=(1.1, 1.1, 1.1))
    t._try_klt_bridge(np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.uint8), 1.2, 0.0)
    assert t._klt_bridge_run == 1  # no trip


def test_max_consec_caps_below_interval():
    t = _tracker(interval=6)
    t._klt_bridge_max_consec = 1
    t._klt_bridge_run = 1  # already one bridge; cap = min(5, 1) = 1
    t._track_klt_prior = _fail_prior
    assert t._try_klt_bridge(None, None, 1.1, 0.0) is None
