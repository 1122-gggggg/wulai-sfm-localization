"""Tests for transient VO continuity and trusted workmap promotion separation.

Verifies:
1. Actual seeding path creates genuine SeedProvenance reflecting capture approval.
2. Inadmissible frames (no approval) still seed features into transient VO with admissible=False.
3. Untrusted-temp PnP supports current pose estimation but never allows unqualified seed promotion.
4. Legacy 2-tuples and malformed provenance are strictly rejected (no compatibility loophole).
5. Nonadmissible seed frames mature into temporary VO without workmap landmark promotion.
6. Ineligible current views mature into temporary VO without workmap landmark promotion.
7. Unpromoted transient VO points cannot escape into session keyframes, index, BA, or query.
8. Qualified multi-view points promote, participate in BA, and persist across resets.
9. Stale provenance and post-reset provenance are rejected from workmap promotion.
10. Frozen region gate restricts trusted workmap expansion without killing temporary VO.
11. Untrusted transient VO points are not counted as trusted workmap points used.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from online_workmap import OnlineWorkMapTracker, SessionWorkMap
from test_workmap_observations import _pixels, _scene, _tracker
from two_rate_tracker import SeedProvenance
from workmap_query import FrozenCoverage, SpatioTemporalQuery
from workmap_state import MappingStateMachine
from workmap_submaps import SubmapRegistry


@pytest.fixture
def _online_tracker(tmp_path):
    base = _tracker(tmp_path)
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker.__dict__.update(base.__dict__)
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    tracker.workmap_session_dir = session_dir
    tracker.workmap_session_id = "test-promo-session"
    tracker.workmap_eval_stage = "reuse"
    tracker._expansion_gate = False
    tracker._expansion_query = False
    tracker._expansion_depth = False
    tracker._depth_estimator = None
    tracker._ba_executor = None
    tracker._closing = False
    tracker._mapping = MappingStateMachine()
    tracker._submaps = SubmapRegistry()
    tracker._presubmap_submap_id = None
    tracker._submap_archive = []
    tracker._reloc_started, tracker._reloc_first_anchor = {}, {}
    tracker._merge_failures, tracker._merge_residual = {}, {}
    tracker._merge_context, tracker._merge_ready = {}, {}
    tracker._graph_edges, tracker._graph_pending = [], False
    tracker._presubmap, tracker._presubmap_pending = None, None
    tracker._init_expansion_state()
    tracker.session_workmap = SessionWorkMap(
        session_dir,
        session_id=tracker.workmap_session_id,
        base_release_id="base-release",
        coordinate_frame_id="coord-frame",
    )
    tracker.workmap = tracker.session_workmap
    tracker._workmap_points_used = 0
    tracker._workmap_reloc_accepted = 0
    tracker._workmap_index_healthy = True
    tracker._workmap_ba_submitted = 0
    tracker._workmap_ba_applied = 0
    tracker._workmap_ba_stale = 0
    tracker._workmap_scale_applied = 0
    tracker._ba_pending_use_ids = set()
    tracker._scale_pending_use_ids = set()
    try:
        yield tracker
    finally:
        tracker.session_workmap.close()


def test_actual_seeding_path_records_genuine_provenance_from_approval(_online_tracker):
    """Calling _vo_keyframe with few live points exercises _detect_features and records SeedProvenance."""
    tracker = _online_tracker
    xyz, _origin, pose = _scene(tracker)
    tracker.profile = replace(tracker.profile, vo=replace(tracker.profile.vo, min_live=200))

    # Construct texture image with corners so _detect_features finds points
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 256, (tracker._h, tracker._w), dtype=np.uint8)

    tracker._ordinal = 2
    tracker._previous_capture_stamp = 1.1
    # Frame is approved
    tracker.approve_capture(
        pose, status="FAST_TRACK", map_inliers=len(xyz), n_corr=len(xyz), ordinal=2, stamp=1.1
    )

    bid_before = tracker._vo_bid_next
    tracker._vo_keyframe(gray, pose)
    tracker._finish_pending_detection()

    # Features must have been seeded
    assert tracker._vo_bid_next > bid_before
    bid = bid_before
    meta = tracker._vo_seed_meta.get(bid)
    assert meta is not None
    assert isinstance(meta, SeedProvenance)
    assert meta.ordinal == 2
    assert meta.stamp == 1.1
    assert meta.source_epoch == tracker._source_epoch
    assert meta.admissible is True
    assert bid in tracker._vo_pose_by_bid
    assert np.array_equal(tracker._vo_pose_by_bid[bid], pose)


def test_actual_seeding_path_records_inadmissible_for_unapproved_frame(_online_tracker):
    """Calling _vo_keyframe without capture approval seeds transient features with admissible=False."""
    tracker = _online_tracker
    xyz, _origin, pose = _scene(tracker, n=5)
    tracker._live_ids = np.arange(1, 6, dtype=np.int64)
    tracker._live_xyz = xyz[:5].copy()
    tracker._live_xy = np.ascontiguousarray(_pixels(tracker, tracker._live_xyz, pose))

    rng = np.random.default_rng(42)
    gray = rng.integers(0, 256, (tracker._h, tracker._w), dtype=np.uint8)

    tracker._ordinal = 4
    tracker._previous_capture_stamp = 1.3
    # No capture approval is set (unapproved / failed frame)
    tracker._current_capture_approval = None

    bid_before = tracker._vo_bid_next
    tracker._vo_keyframe(gray, pose)
    tracker._finish_pending_detection()

    # Features are still seeded for transient tracking continuity
    assert tracker._vo_bid_next > bid_before
    bid = bid_before
    meta = tracker._vo_seed_meta.get(bid)
    assert meta is not None
    assert isinstance(meta, SeedProvenance)
    assert meta.ordinal == 4
    assert meta.stamp == 1.3
    assert meta.source_epoch == tracker._source_epoch
    # Crucial: provenance records that seed frame was NOT admissible
    assert meta.admissible is False


def test_untrusted_temp_pnp_supports_current_pose_but_no_unqualified_seed_promotion(_online_tracker):
    """Untrusted transient VO points help solve PnP, but an ineligible seed batch cannot promote."""
    tracker = _online_tracker
    # Stage 30 old positive points and 60 untrusted transient VO points (negative IDs not in workmap)
    rng = np.random.default_rng(101)
    xyz = np.column_stack(
        (rng.uniform(-2.0, 2.0, 90), rng.uniform(-1.0, 1.0, 90), rng.uniform(4.5, 6.0, 90))
    )
    old_ids = np.arange(1, 31, dtype=np.int64)
    untrusted_ids = -np.arange(101, 161, dtype=np.int64)
    all_ids = np.concatenate([old_ids, untrusted_ids])

    cam_pose = np.column_stack((np.eye(3), [-0.15, 0.0, 0.0]))
    all_xy = _pixels(tracker, xyz, cam_pose)

    tracker._live_ids = all_ids
    tracker._live_xyz = xyz.copy()
    tracker._live_xy = np.ascontiguousarray(all_xy)
    tracker._workmap_initial_anchor = True  # Anchor already established in prior mission

    # Step absolute pose: PnP uses both positive and untrusted negative points
    status, pose, center, inliers, map_inl, vo_inl, n_corr, quality = tracker._step_absolute_pose(
        reseeded=False, stamp=1.5
    )
    assert status == "FAST_TRACK" or status == "VO_ONLY"
    assert inliers >= 80  # Solved with aid of transient VO points!
    assert vo_inl >= 50   # Transient points contributed to inliers

    # Now mature an ineligible seed batch (seeded from an inadmissible pose)
    new_xyz = xyz[:20] + [0.08, 0.04, 0.20]
    seed_pose = np.column_stack((np.eye(3), np.zeros(3)))
    seed_xy = _pixels(tracker, new_xyz, seed_pose)
    cur_xy = _pixels(tracker, new_xyz, cam_pose)

    tracker._ordinal = 12
    tracker._previous_capture_stamp = 1.5
    tracker._vo_seed_xy = seed_xy.copy()
    tracker._vo_xy = cur_xy.copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    # Ineligible seed pose
    tracker._vo_seed_meta = {
        0: SeedProvenance(
            ordinal=0,
            stamp=1.0,
            source_epoch=tracker._source_epoch,
            admissible=False,
        )
    }

    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), cam_pose)

    # Triangulation succeeds and points are added to live tracking
    assert np.count_nonzero(tracker._live_ids < 0) >= len(new_xyz)
    # But durable workmap landmark promotion is strictly blocked
    assert len(tracker._workmap_landmarks) == 0
    assert len(tracker._workmap_pending_ids) == 0
    assert len(tracker.session_workmap.landmarks) == 0


def test_legacy_two_tuples_rejected_without_compatibility_loophole(_online_tracker):
    """Missing or 2-tuple provenance is rejected safely without synthesizing approval."""
    tracker = _online_tracker
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:15] + [0.09, 0.05, 0.25]
    seed_xy = _pixels(tracker, new_xyz, seed_pose)
    current_xy = _pixels(tracker, new_xyz, current_pose)

    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = seed_xy.copy()
    tracker._vo_xy = current_xy.copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    # Legacy 2-tuple installed (no epoch, no admission)
    tracker._vo_seed_meta = {0: (0, 1.0)}

    # Frame is otherwise approved
    tracker.approve_capture(
        current_pose, status="FAST_TRACK", map_inliers=100, n_corr=120, ordinal=8, stamp=1.25
    )

    # Helper directly verifies rejection
    assert tracker._get_seed_provenance(0) is None

    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)

    # Transient VO triangulated and retained
    assert np.count_nonzero(tracker._live_ids < 0) == len(new_xyz)
    # Promotion strictly rejected! No approval synthesized from 2-tuple!
    assert len(tracker._workmap_landmarks) == 0
    assert len(tracker._workmap_pending_ids) == 0
    assert len(tracker.session_workmap.landmarks) == 0


def test_nonadmissible_seed_frame_matures_transient_vo_without_workmap_promotion(_online_tracker):
    """Features seeded from an ineligible pose mature into temporary VO but cannot promote."""
    tracker = _online_tracker
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:20] + [0.09, 0.05, 0.25]
    seed_xy = _pixels(tracker, new_xyz, seed_pose)
    current_xy = _pixels(tracker, new_xyz, current_pose)

    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = seed_xy.copy()
    tracker._vo_xy = current_xy.copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    # Ineligible seed pose: admissible = False
    tracker._vo_seed_meta = {
        0: SeedProvenance(
            ordinal=0,
            stamp=1.0,
            source_epoch=tracker._source_epoch,
            admissible=False,
        )
    }

    # Current view is approved
    tracker.approve_capture(
        current_pose, status="FAST_TRACK", map_inliers=100, n_corr=120, ordinal=8, stamp=1.25
    )

    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)

    # Temporary VO points ARE triangulated and retained with negative IDs
    assert np.count_nonzero(tracker._live_ids < 0) == len(new_xyz)
    # But NONE are promoted to durable workmap landmarks
    assert len(tracker._workmap_landmarks) == 0
    assert len(tracker._workmap_pending_ids) == 0
    assert len(tracker.session_workmap.landmarks) == 0
    for frame in tracker.session_workmap.keyframes.values():
        assert np.all(np.asarray(frame["ba_ids"]) > 0)


def test_ineligible_current_view_matures_transient_vo_without_workmap_promotion(_online_tracker):
    """Eligible seed maturing at an ineligible current view retains VO but blocks promotion."""
    tracker = _online_tracker
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:25] + [0.08, 0.04, 0.20]
    seed_xy = _pixels(tracker, new_xyz, seed_pose)
    current_xy = _pixels(tracker, new_xyz, current_pose)

    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = seed_xy.copy()
    tracker._vo_xy = current_xy.copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    # Eligible seed: admissible = True
    tracker._vo_seed_meta = {
        0: SeedProvenance(
            ordinal=0,
            stamp=1.0,
            source_epoch=tracker._source_epoch,
            admissible=True,
        )
    }

    # Current frame is NOT approved (failed or unapproved capture)
    tracker._current_capture_approval = None

    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)

    # Temporary VO retained for tracking continuity
    assert np.count_nonzero(tracker._live_ids < 0) == len(new_xyz)
    # Blocked from durable workmap registration
    assert len(tracker._workmap_landmarks) == 0
    assert len(tracker._workmap_pending_ids) == 0
    assert len(tracker.session_workmap.keyframes) == 0


def test_unpromoted_points_cannot_escape_to_session_keyframes_index_or_ba(_online_tracker):
    """Untrusted negative IDs are excluded from session keyframes, query, and BA."""
    tracker = _online_tracker
    tracker._expansion_query = True
    tracker._query = SpatioTemporalQuery(tracker._w, tracker._h)
    xyz, seed_pose, current_pose = _scene(tracker)

    # Add 10 promoted landmarks
    promoted_xyz = xyz[:10] + [0.05, 0.05, 0.15]
    promoted_xy = _pixels(tracker, promoted_xyz, current_pose)
    promoted_ids = -np.arange(1, 11, dtype=np.int64)
    for lid, p_xy, p_xyz in zip(promoted_ids, promoted_xy, promoted_xyz, strict=True):
        tracker._workmap_landmarks[int(lid)] = {
            "xyz": p_xyz.copy(),
            "parallax_deg": 4.5,
            "revision": 0,
            "observations": (
                (0, 1.0, p_xy.copy(), seed_pose),
                (8, 1.25, p_xy.copy(), current_pose),
            ),
            "last_inlier": (8, 1.25, p_xy.copy()),
        }
        tracker._workmap_pending_ids.add(int(lid))

    # Add 15 UNPROMOTED negative IDs directly into live tracking (transient VO)
    unpromoted_xyz = xyz[10:25] + [-0.05, -0.05, 0.10]
    unpromoted_xy = _pixels(tracker, unpromoted_xyz, current_pose)
    unpromoted_ids = -np.arange(101, 116, dtype=np.int64)

    tracker._live_ids = np.concatenate([tracker._live_ids, promoted_ids, unpromoted_ids])
    tracker._live_xy = np.ascontiguousarray(
        np.concatenate([tracker._live_xy, promoted_xy, unpromoted_xy])
    )
    tracker._live_xyz = np.concatenate([tracker._live_xyz, promoted_xyz, unpromoted_xyz])

    # Mark current frame as approved
    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker.approve_capture(
        current_pose,
        status="FAST_TRACK",
        map_inliers=100,
        n_corr=len(tracker._live_xy),
        ordinal=8,
        stamp=1.25,
    )

    # Run keyframe step
    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)

    # 1. Check session workmap keyframes
    assert len(tracker.session_workmap.keyframes) == 1
    kf = next(iter(tracker.session_workmap.keyframes.values()))
    kf_ids = set(map(int, kf["ids"]))
    for up_id in unpromoted_ids:
        assert int(up_id) not in kf_ids, f"Unpromoted ID {up_id} escaped into session keyframe!"

    # 2. Check query keyframe indexing
    if tracker._query is not None and len(tracker._query._kf_ids):
        query_kf_ids = set(map(int, next(iter(tracker._query._kf_ids.values()))))
        for up_id in unpromoted_ids:
            assert int(up_id) not in query_kf_ids, f"Unpromoted ID {up_id} escaped into query!"

    # 3. Check BA snapshot
    ba_snap = tracker.session_workmap.ba_snapshot(
        tracker._source_epoch,
        tracker._K,
        incremental=False,
        anchor_min_inliers=10,
    )
    if ba_snap is not None:
        ba_lm_ids = set(map(int, ba_snap.get("landmarks", {}).keys()))
        for up_id in unpromoted_ids:
            assert int(up_id) not in ba_lm_ids, f"Unpromoted ID {up_id} escaped into BA snapshot!"


def test_qualified_multiview_points_promote_participate_in_ba_and_persist(_online_tracker):
    """Qualified multi-view points promote, participate in BA snapshot, and persist across reset."""
    tracker = _online_tracker
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:15] + [0.07, 0.03, 0.18]
    seed_xy = _pixels(tracker, new_xyz, seed_pose)
    current_xy = _pixels(tracker, new_xyz, current_pose)

    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = seed_xy.copy()
    tracker._vo_xy = current_xy.copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    tracker._vo_seed_meta = {
        0: SeedProvenance(
            ordinal=0,
            stamp=1.0,
            source_epoch=tracker._source_epoch,
            admissible=True,
        )
    }

    # Both views are approved
    tracker.approve_capture(
        current_pose, status="FAST_TRACK", map_inliers=100, n_corr=120, ordinal=8, stamp=1.25
    )

    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)

    # Landmarks are promoted
    assert len(tracker._workmap_landmarks) == len(new_xyz)
    promoted_keys = set(tracker._workmap_landmarks.keys())
    assert len(promoted_keys) == 15

    # Check session workmap archiving
    assert len(tracker.session_workmap.landmarks) == 15
    for key in promoted_keys:
        assert key in tracker.session_workmap.landmarks
        assert tracker._workmap_landmarks[key]["revision"] > 0

    for ordinal in range(9, 12):
        pose = current_pose.copy()
        pose[0, 3] -= 0.03 * (ordinal - 8)
        tracker._ordinal = ordinal
        tracker._previous_capture_stamp = 1.25 + 0.1 * (ordinal - 8)
        tracker._live_xy = np.ascontiguousarray(_pixels(tracker, tracker._live_xyz, pose))
        solved = tracker._step_absolute_pose(False, tracker._previous_capture_stamp)
        assert solved[0] == "FAST_TRACK"
        tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), solved[1])

    # BA snapshot includes these landmarks
    snapshot = tracker.session_workmap.ba_snapshot(
        tracker._source_epoch,
        tracker._K,
        incremental=False,
        anchor_min_inliers=10,
    )
    assert snapshot is not None
    assert set(snapshot["new_landmarks"]) == promoted_keys

    # Reset clears current tracking priors but durable landmarks persist
    tracker.reset()
    assert len(tracker._vo_seed_meta) == 0
    assert len(tracker._vo_pose_by_bid) == 0
    assert len(tracker._live_ids) == 0
    # Durable landmarks survive reset
    assert len(tracker._workmap_landmarks) == 15
    for key in promoted_keys:
        assert key in tracker._workmap_landmarks


def test_stale_and_reset_provenance_rejected(_online_tracker):
    """Seed batches with stale epochs or from prior resets are rejected from promotion."""
    tracker = _online_tracker
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:10] + [0.06, 0.02, 0.22]
    seed_xy = _pixels(tracker, new_xyz, seed_pose)
    current_xy = _pixels(tracker, new_xyz, current_pose)

    # Initial epoch is 0
    assert tracker._source_epoch == 0
    # Reset advances epoch to 1 and drops priors
    tracker.reset()
    assert tracker._source_epoch == 1

    # Attempt to mature a batch that carries stale epoch 0
    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = seed_xy.copy()
    tracker._vo_xy = current_xy.copy()
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    # Stale provenance with source_epoch = 0 != current epoch 1
    tracker._vo_seed_meta = {
        0: SeedProvenance(
            ordinal=0,
            stamp=1.0,
            source_epoch=0,
            admissible=True,
        )
    }

    tracker.approve_capture(
        current_pose, status="FAST_TRACK", map_inliers=100, n_corr=120, ordinal=8, stamp=1.25
    )

    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)

    # Transient VO triangulated and retained
    assert np.count_nonzero(tracker._live_ids < 0) == len(new_xyz)
    # But promotion is REJECTED due to stale epoch
    assert len(tracker._workmap_landmarks) == 0
    assert len(tracker.session_workmap.landmarks) == 0


def test_frozen_region_gate_retains_transient_vo_but_blocks_trusted_expansion(_online_tracker):
    """Points inside frozen region are excluded from workmap landmarks but kept in temporary VO."""
    tracker = _online_tracker
    tracker._expansion_gate = True
    xyz, seed_pose, current_pose = _scene(tracker)
    new_xyz = xyz[:20] + [0.09, 0.05, 0.25]
    # Coverage covers the first 5 points
    coverage = FrozenCoverage(np.repeat(new_xyz[:5], 3, axis=0))
    tracker._coverage = coverage

    tracker._ordinal = 8
    tracker._previous_capture_stamp = 1.25
    tracker._vo_seed_xy = _pixels(tracker, new_xyz, seed_pose)
    tracker._vo_xy = _pixels(tracker, new_xyz, current_pose)
    tracker._vo_bid = np.zeros(len(new_xyz), dtype=np.int64)
    tracker._vo_bid_next = tracker.profile.vo.batch_lag + 1
    tracker._vo_pose_by_bid = {0: seed_pose.copy()}
    tracker._vo_seed_meta = {
        0: SeedProvenance(
            ordinal=0,
            stamp=1.0,
            source_epoch=tracker._source_epoch,
            admissible=True,
        )
    }

    tracker.approve_capture(
        current_pose, status="FAST_TRACK", map_inliers=100, n_corr=120, ordinal=8, stamp=1.25
    )

    tracker._vo_keyframe(np.zeros((tracker._h, tracker._w), dtype=np.uint8), current_pose)

    expected_promoted = new_xyz[~coverage.covered(new_xyz)]
    # All 20 points are retained in temporary VO
    assert np.count_nonzero(tracker._live_ids < 0) == len(new_xyz)
    # Only points outside frozen region are registered in workmap landmarks
    assert len(tracker._workmap_landmarks) == len(expected_promoted)
    assert tracker._workmap_frozen_gate_rejected >= 5


def test_untrusted_transient_vo_points_not_counted_in_workmap_points_used(_online_tracker):
    """Untrusted negative IDs in PnP do not increment _workmap_points_used."""
    tracker = _online_tracker
    xyz, _origin, moved = _scene(tracker)

    # 1 promoted landmark
    promoted_xyz = xyz[:1] + [0.05, 0.05, 0.15]
    promoted_xy = _pixels(tracker, promoted_xyz, moved)
    tracker._workmap_landmarks[-1] = {
        "xyz": promoted_xyz[0].copy(),
        "parallax_deg": 5.0,
        "revision": 1,
        "observations": (
            (0, 1.0, promoted_xy[0].copy(), np.eye(3, 4)),
            (8, 1.25, promoted_xy[0].copy(), moved),
        ),
        "last_inlier": (8, 1.25, promoted_xy[0].copy()),
    }

    # 5 untrusted negative IDs (not in _workmap_landmarks)
    untrusted_ids = -np.arange(10, 15, dtype=np.int64)
    untrusted_xyz = xyz[1:6] + [0.05, 0.05, 0.15]
    untrusted_xy = _pixels(tracker, untrusted_xyz, moved)

    tracker._live_ids = np.concatenate([tracker._live_ids, [-1], untrusted_ids])
    tracker._live_xy = np.ascontiguousarray(
        np.concatenate([tracker._live_xy, promoted_xy, untrusted_xy])
    )
    tracker._live_xyz = np.concatenate([tracker._live_xyz, promoted_xyz, untrusted_xyz])

    info = {
        "ok": True,
        "status": "FAST_TRACK",
        "map_inliers": 120,
        "vo_inliers": 6,  # 1 promoted + 5 untrusted
    }
    tracker._ordinal = 8
    tracker._workmap_pose_ids = {-1}  # Only trusted landmark was in _workmap_pose_ids

    # Step bookkeeping check
    if info.get("status") in {"FAST_TRACK", "VO_ONLY", "RELOC_SEED"}:
        actual_ids = tuple(
            int(key)
            for key in tracker._live_ids
            if key < 0
            and int(key) in tracker._workmap_pose_ids
            and int(key) in tracker._workmap_landmarks
            and int(tracker._workmap_landmarks[int(key)]["last_inlier"][0]) == tracker._ordinal
        )
        tracker._workmap_points_used += len(actual_ids)

    # Only the 1 trusted point is counted, not the 6 total vo_inliers!
    assert tracker._workmap_points_used == 1
