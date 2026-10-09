"""A session landmark that re-observes a GlueMap point is fused into it; GlueMap never moves."""
from __future__ import annotations

import numpy as np

from online_workmap import FUSE_MAX_DISTANCE_U, FUSE_MAX_REPROJ_PX, OnlineWorkMapTracker
from test_workmap_ba_integration import _scene
from workmap_query import SpatioTemporalQuery
from workmap_submaps import SubmapRegistry

K = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]])
POSE = np.column_stack((np.eye(3), [0.0, 0.0, 0.0]))
OLD_ID = 999


def _candidate(session, new_id, *, offset_u, offset_px):
    landmark = session.landmarks[new_id]
    old_xyz = np.asarray(landmark["xyz"]) + [offset_u, 0.0, 0.0]
    camera = old_xyz @ POSE[:, :3].T + POSE[:, 3]
    pixel = (K @ camera)[:2] / camera[2]
    return {
        "new_id": new_id, "old_id": OLD_ID, "old_xyz": old_xyz,
        "new_xy": pixel + [offset_px, 0.0], "pose": POSE, "K": K, "keyframe": "kf",
    }


def _tracker(session):
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker.session_workmap = session
    tracker._submaps = SubmapRegistry()
    tracker._workmap_landmarks = {
        key: dict(value) for key, value in session.tracker_landmarks().items()
    }
    tracker._workmap_pending_ids = set()
    tracker._query = SpatioTemporalQuery(64, 64)
    tracker._workmap_landmarks_fused = 0
    tracker._live_ids = np.array([5, -1, -2], dtype=np.int64)
    tracker._live_xy = np.zeros((3, 2))
    tracker._live_xyz = np.array([[0.0, 0.0, 4.0], *[v["xyz"] for v in (
        tracker._workmap_landmarks[-1], tracker._workmap_landmarks[-2])]])
    return tracker


def test_close_and_well_reprojected_pairs_fuse_and_relabel_the_live_rows(tmp_path):
    session, _old, _append, _K = _scene(tmp_path)
    try:
        tracker = _tracker(session)
        near = _candidate(session, -1, offset_u=FUSE_MAX_DISTANCE_U - 0.01, offset_px=2.0)
        tracker._apply_fusions([near])
        assert tracker._workmap_landmarks_fused == 1
        assert session.landmarks[-1]["fused_to"] == OLD_ID
        np.testing.assert_array_equal(session.landmarks[-1]["xyz"], near["old_xyz"])
        assert -1 not in tracker._workmap_landmarks
        np.testing.assert_array_equal(tracker._live_ids, [5, OLD_ID, -2])
        np.testing.assert_array_equal(tracker._live_xyz[1], near["old_xyz"])
        # An already fused point is not fused twice.
        revision = session.revision
        tracker._apply_fusions([near])
        assert session.revision == revision and tracker._workmap_landmarks_fused == 1
    finally:
        session.close()


def test_pairs_beyond_the_distance_or_reprojection_limits_do_not_fuse(tmp_path):
    session, _old, _append, _K = _scene(tmp_path)
    try:
        tracker = _tracker(session)
        far = _candidate(session, -1, offset_u=FUSE_MAX_DISTANCE_U + 0.01, offset_px=0.0)
        blurry = _candidate(session, -2, offset_u=0.01, offset_px=FUSE_MAX_REPROJ_PX + 1.0)
        tracker._apply_fusions([far, blurry])
        assert tracker._workmap_landmarks_fused == 0
        assert "fused_to" not in session.landmarks[-1] and "fused_to" not in session.landmarks[-2]
        np.testing.assert_array_equal(tracker._live_ids, [5, -1, -2])
    finally:
        session.close()


def test_a_fused_point_is_a_glue_constant_in_the_next_ba_snapshot(tmp_path):
    session, _old, _append, K_scene = _scene(tmp_path)
    try:
        before = session.ba_snapshot(0, K_scene)
        assert -1 in before["new_landmarks"]
        xyz = np.asarray(session.landmarks[-1]["xyz"])
        assert session.fuse_landmark(-1, OLD_ID, xyz, distance_u=0.0, reprojection_px=0.0)
        session.flush_writer()
        after = session.ba_snapshot(0, K_scene)
        assert -1 not in after["new_landmarks"] and -1 not in after["landmark_revisions"]
        assert OLD_ID in after["old_landmarks"]
        np.testing.assert_allclose(after["old_landmarks"][OLD_ID], xyz)
        for frame in after["keyframes"]:
            assert -1 not in frame["ids"]
            assert len(set(frame["ids"])) == len(frame["ids"])
        assert any(OLD_ID in frame["ids"] for frame in after["keyframes"])
    finally:
        session.close()
