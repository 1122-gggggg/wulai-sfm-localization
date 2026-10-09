"""Real BA submission against durable session keyframes and revision checks."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from online_workmap import OnlineWorkMapTracker, SessionWorkMap
from two_rate_tracker import SeedProvenance
from workmap_ba import solve_ba_snapshot


def _scene(tmp_path):
    rng = np.random.default_rng(20260927)
    K = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]])
    old = rng.uniform([-0.55, -0.5, 3.5], [0.55, 0.5, 5.0], size=(48, 3))
    fresh = rng.uniform([-0.4, -0.35, 3.6], [0.4, 0.35, 4.9], size=(6, 3))
    old_ids = np.arange(1, len(old) + 1, dtype=np.int64)
    new_ids = -np.arange(1, len(fresh) + 1, dtype=np.int64)
    session = SessionWorkMap(
        tmp_path / "session",
        session_id="ba-session",
        base_release_id="base-r1",
        coordinate_frame_id="frame-r1",
        min_reference_points=6,
    )

    def pose(index):
        return np.column_stack((np.eye(3), [-0.065 * index, 0.0, 0.0]))

    def project(points, matrix):
        camera = points @ matrix[:, :3].T + matrix[:, 3]
        homogeneous = camera @ K.T
        return homogeneous[:, :2] / homogeneous[:, 2, None]

    landmarks = {}
    for index, point in enumerate(fresh):
        landmarks[-index - 1] = {
            "xyz": point + [0.003, -0.002, 0.005],
            "observations": (
                (0, 1.0, project(point[None], pose(0))[0], pose(0)),
                (1, 1.1, project(point[None], pose(1))[0], pose(1)),
            ),
            "last_inlier": (1, 1.1, project(point[None], pose(1))[0]),
            "parallax_deg": 2.0,
            "valid_observation_count": 2,
        }
    image = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    def append_frame(index, include_negative=True):
        world = np.concatenate((old, fresh)) if include_negative else old
        ids = np.concatenate((old_ids, new_ids)) if include_negative else old_ids
        session.record_keyframe(
            image,
            ordinal=index,
            stamp=1.0 + index * 0.1,
            source_epoch=0,
            pose=pose(index),
            K=K,
            image_size=(64, 64),
            ids=ids,
            xy=project(world, pose(index)),
            xyz=np.concatenate((old, np.stack([landmarks[int(key)]["xyz"] for key in new_ids])))
            if include_negative
            else old,
            landmarks=landmarks if include_negative else {},
        )
        session.flush_writer()

    append_frame(0, False)
    session.archive_landmarks(landmarks)
    for index in range(1, 5):
        append_frame(index)
    return session, old, append_frame, K


def test_ba_commit_merges_new_keyframes_and_keeps_frozen_map_unchanged(tmp_path):
    session, old, append_frame, K = _scene(tmp_path)
    try:
        snapshot = session.ba_snapshot(0, K)
        assert snapshot is not None
        original_old = {
            key: np.asarray(value).copy() for key, value in snapshot["old_landmarks"].items()
        }
        candidate = solve_ba_snapshot(snapshot)
        assert candidate["ok"], candidate.get("reason")
        assert candidate["updates"]

        # A later keyframe arrives after the solver read its private four-view snapshot.
        append_frame(5)
        late_name = next(name for name, frame in session.keyframes.items() if frame["ordinal"] == 5)
        revision_before = session.revision
        assert session.apply_ba_candidate(snapshot, candidate)
        assert session.revision > revision_before
        assert late_name in session.keyframes
        assert all(
            np.array_equal(snapshot["old_landmarks"][key], value)
            for key, value in original_old.items()
        )
        assert all(np.array_equal(old[key - 1], value) for key, value in original_old.items())
        assert not session.apply_ba_candidate(snapshot, candidate)
    finally:
        session.close()
    updates = [
        json.loads(row)
        for row in (tmp_path / "session" / "geometry_updates.jsonl").read_text().splitlines()
    ]
    assert any(
        row["event"] == "ba_geometry_committed" and row["after_cost"] < row["before_cost"]
        for row in updates
    )


def test_ba_rejects_changed_input_revision_and_old_support_loss(tmp_path):
    session, _old, _append_frame, K = _scene(tmp_path)
    try:
        snapshot = session.ba_snapshot(0, K)
        candidate = solve_ba_snapshot(snapshot)
        assert candidate["ok"], candidate.get("reason")
        damaged = {**candidate, "poses": dict(candidate["poses"])}
        moved = np.array(damaged["poses"][snapshot["keyframes"][-1]["name"]], dtype=float)
        moved[:, 3] += [0.4, 0.0, 0.0]
        damaged["poses"][snapshot["keyframes"][-1]["name"]] = moved.tolist()
        assert not session.apply_ba_candidate(snapshot, damaged)
        assert session.revision == snapshot["snapshot_revision"]
        point_id = next(iter(snapshot["landmark_revisions"]))
        session.landmarks[point_id]["revision"] += 1
        assert not session.apply_ba_candidate(snapshot, candidate)
        session.update_source_epoch(1)
        assert not session.apply_ba_candidate(snapshot, candidate)
    finally:
        session.close()


def test_fixed_old_pose_fallback_improves_new_geometry_without_moving_old_map(tmp_path):
    session, _old, _append_frame, K = _scene(tmp_path)
    try:
        snapshot = session.ba_snapshot(0, K)
        fallback = solve_ba_snapshot({**snapshot, "freeze_poses": True})
        assert fallback["ok"], fallback.get("reason")
        assert fallback["after_cost"] < fallback["before_cost"]
        for frame in snapshot["keyframes"]:
            assert np.array_equal(fallback["poses"][frame["name"]], frame["pose"])
        assert session.apply_ba_candidate(snapshot, fallback)
    finally:
        session.close()


def test_local_window_is_ten_keyframes_plus_a_fixed_boundary(tmp_path):
    from workmap_ba import BA_WINDOW_KEYFRAMES

    session, old, append_frame, K = _scene(tmp_path)
    try:
        for index in range(5, 16):
            append_frame(index)
        snapshot = session.ba_snapshot(0, K, anchor_min_inliers=len(old) + 1)
        frames = snapshot["keyframes"]
        assert len(frames) == BA_WINDOW_KEYFRAMES + 1
        assert [f["ordinal"] for f in frames[1:]] == list(range(6, 16))      # the 10 newest
        assert frames[0]["ordinal"] < 6                                       # the older boundary
        assert [f["fixed"] for f in frames] == [True] + [False] * BA_WINDOW_KEYFRAMES
        anchored = session.ba_snapshot(0, K, anchor_min_inliers=1)
        assert all(f["fixed"] for f in anchored["keyframes"])        # every view holds GlueMap points
    finally:
        session.close()


@pytest.mark.parametrize("stale", [False, True])
def test_pending_vo_triangulation_uses_only_committed_ba_seed_pose(tmp_path, stale):
    session, _old, _append_frame, K = _scene(tmp_path)
    try:
        seed_frame = next(frame for frame in session.keyframes.values() if frame["ordinal"] == 4)
        seed_frame["pose"] = np.asarray(seed_frame["pose"]).copy()
        seed_frame["pose"][0, 3] += 0.04
        snapshot = session.ba_snapshot(0, K, anchor_min_inliers=100)
        candidate = solve_ba_snapshot(snapshot)
        tracker = object.__new__(OnlineWorkMapTracker)
        tracker.session_workmap = session
        tracker._source_epoch = 0
        tracker._workmap_ba_stale = tracker._workmap_ba_rejected = tracker._workmap_ba_applied = 0
        tracker._ba_pending_use_ids = set()
        tracker._workmap_landmarks = {}
        tracker._live_ids = np.empty(0, dtype=np.int64)
        tracker._live_xyz = np.empty((0, 3))
        tracker._query = None
        tracker._last_pose_ordinal = 5
        tracker._vo_seed_meta = {3: SeedProvenance(4, 1.4, 0, True)}
        initial = np.asarray(seed_frame["pose"]).copy()
        tracker._vo_pose_by_bid = {3: initial.copy()}
        if stale:
            seed_frame["pose_revision"] += 1
        assert tracker._apply_ba_result(snapshot, candidate) is not stale
        if stale:
            np.testing.assert_array_equal(tracker._vo_pose_by_bid[3], initial)
            return

        point = np.array([0.1, 0.1, 4.0])
        true_seed = np.column_stack((np.eye(3), [-0.065 * 4, 0.0, 0.0]))
        current = np.column_stack((np.eye(3), [-0.065 * 6, 0.0, 0.0]))

        def project(pose):
            camera = pose[:, :3] @ point + pose[:, 3]
            pixel = K @ camera
            return (pixel[:2] / pixel[2]).reshape(2, 1)

        homogeneous = cv2.triangulatePoints(
            K @ tracker._vo_pose_by_bid[3], K @ current, project(true_seed), project(current)
        )
        triangulated = homogeneous[:3, 0] / homogeneous[3, 0]
        np.testing.assert_allclose(triangulated, point, atol=1e-3)
    finally:
        session.close()


def _window_variable_ids(snapshot):
    counts = {}
    for frame in snapshot["keyframes"]:
        for key in frame["ids"]:
            if key < 0:
                counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= 2}


def _observer_scene(tmp_path):
    session, old, append_frame, K = _scene(tmp_path)
    for index in range(5, 16):
        append_frame(index)
    snapshot = session.ba_snapshot(0, K)
    return session, snapshot


def test_outside_keyframes_enter_the_snapshot_as_fixed_observers_of_variable_points(tmp_path):
    from workmap_ba import BA_MAX_OBSERVER_KEYFRAMES, BA_WINDOW_KEYFRAMES

    session, snapshot = _observer_scene(tmp_path)
    try:
        window = {frame["name"] for frame in snapshot["keyframes"]}
        assert len(window) == BA_WINDOW_KEYFRAMES + 1
        observers = snapshot["observers"]
        assert 0 < len(observers) <= BA_MAX_OBSERVER_KEYFRAMES
        # Boundary keyframe is ordinal 5; the others that see the points follow, newest first.
        assert [item["ordinal"] for item in observers] == [4, 3, 2, 1, 0]
        variable = _window_variable_ids(snapshot)
        for item in observers:
            assert item["name"] not in window and item["fixed"] is True
            assert item["ids"] and set(item["ids"]) <= variable
            assert np.array_equal(item["pose"], session.keyframes[item["name"]]["pose"])
            assert item["pose_revision"] == session.keyframes[item["name"]]["pose_revision"]
    finally:
        session.close()


def test_observers_are_part_of_the_cost_and_a_fixed_camera_may_not_lose_support(tmp_path):
    session, snapshot = _observer_scene(tmp_path)
    try:
        candidate = solve_ba_snapshot(snapshot)
        assert candidate["ok"], candidate.get("reason")
        without = solve_ba_snapshot({**snapshot, "observers": []})
        assert without["ok"], without.get("reason")
        assert candidate["before_cost"] > without["before_cost"]
        assert set(candidate["poses"]) == {frame["name"] for frame in snapshot["keyframes"]}

        damaged = {**candidate, "updates": dict(candidate["updates"])}
        key = next(iter(damaged["updates"]))
        damaged["updates"][key] = (np.asarray(damaged["updates"][key]) + [0.5, 0.0, 0.0]).tolist()
        revision = session.revision
        assert not session.apply_ba_candidate(snapshot, damaged)
        assert session.last_ba_rejection_reason == "observer_support_count"
        # The same damage is judged per observation when the camera is not a declared observer.
        assert not session.apply_ba_candidate({**snapshot, "observers": []}, damaged)
        assert session.last_ba_rejection_reason == "later_keyframe_reprojection"
        assert session.revision == revision

        assert session.apply_ba_candidate(snapshot, candidate)
    finally:
        session.close()


def test_an_observer_whose_pose_changed_after_the_snapshot_rejects_the_candidate(tmp_path):
    session, snapshot = _observer_scene(tmp_path)
    try:
        candidate = solve_ba_snapshot(snapshot)
        assert candidate["ok"], candidate.get("reason")
        session.keyframes[snapshot["observers"][0]["name"]]["pose_revision"] += 1
        assert not session.apply_ba_candidate(snapshot, candidate)
        assert session.last_ba_rejection_reason == "keyframe_pose_revision"
    finally:
        session.close()
