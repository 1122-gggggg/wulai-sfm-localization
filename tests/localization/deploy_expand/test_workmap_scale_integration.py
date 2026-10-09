"""Measured cross-map overlap may weld only a proven session component."""

from __future__ import annotations

import json

import numpy as np

from online_workmap import SessionWorkMap, _fit_overlap_job, _transform_pose_sim3


def _overlap_scene(tmp_path, *, protected_old_support: bool):
    rng = np.random.default_rng(20260928)
    K = np.array([[180.0, 0.0, 160.0], [0.0, 180.0, 120.0], [0.0, 0.0, 1.0]])
    fresh = rng.uniform([-0.7, -0.5, 3.5], [0.7, 0.5, 5.0], size=(24, 3))
    scale = 1.22
    rotation = np.eye(3)
    offset = np.array([0.12, -0.08, 0.06])
    frozen_matches = scale * fresh + offset
    frozen_support = rng.uniform([-0.6, -0.5, 3.5], [0.6, 0.5, 5.2], size=(12, 3))
    negative_ids = -np.arange(1, len(fresh) + 1, dtype=np.int64)
    positive_ids = np.arange(101, 101 + len(frozen_support), dtype=np.int64)

    def local_pose(i):
        return np.column_stack((np.eye(3), [-0.17 * i, 0.03 * i, 0.0]))

    def project(xyz, pose):
        camera = xyz @ pose[:, :3].T + pose[:, 3]
        homogeneous = camera @ K.T
        return homogeneous[:, :2] / homogeneous[:, 2, None]

    session = SessionWorkMap(
        tmp_path / "session",
        session_id="measured-scale-session",
        base_release_id="immutable-r1",
        coordinate_frame_id="fixed-global-frame",
        min_reference_points=6,
    )
    landmarks = {}
    for row, point in enumerate(fresh):
        first = 1 if row >= 18 else 0
        second = first + 1
        landmarks[int(negative_ids[row])] = {
            "xyz": point,
            "observations": (
                (
                    first,
                    1.0 + first * 0.1,
                    project(point[None], local_pose(first))[0],
                    local_pose(first),
                ),
                (
                    second,
                    1.0 + second * 0.1,
                    project(point[None], local_pose(second))[0],
                    local_pose(second),
                ),
            ),
            "last_inlier": (
                second,
                1.0 + second * 0.1,
                project(point[None], local_pose(second))[0],
            ),
            "parallax_deg": 3.0,
            "valid_observation_count": 2,
        }
    for key, revision in session.archive_landmarks(landmarks).items():
        landmarks[key]["revision"] = revision
    image = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    for ordinal in range(3):
        pose = local_pose(ordinal)
        corrected = _transform_pose_sim3(pose, scale, rotation, offset)
        support_pose = pose if protected_old_support else corrected
        shown = 12 + ordinal * 6
        ids = np.concatenate((positive_ids, negative_ids[:shown]))
        xy = np.concatenate((project(frozen_support, support_pose), project(fresh[:shown], pose)))
        xyz = np.concatenate((frozen_support, fresh[:shown]))
        name = session.record_keyframe(
            image,
            ordinal=ordinal,
            stamp=1.0 + ordinal * 0.1,
            source_epoch=0,
            pose=pose,
            K=K,
            image_size=(320, 240),
            ids=ids,
            xy=xy,
            xyz=xyz,
            landmarks=landmarks,
        )
        assert name is not None
        session.flush_writer()
        frame = session.keyframes[name]
        reference = session._reference_from_frame_locked(frame)
        session.mark_indexed(reference, f"indexed-{ordinal}")
        assert frame["index_ready"]
        measured = {
            "session_id": session.session_id,
            "source_epoch": 0,
            "keyframe_name": name,
            "keyframe_revision": frame["revision"],
            "pose_revision": frame["pose_revision"],
            "landmark_revisions": frame["landmark_revisions"],
            "keyframe_pose": pose,
            "keyframe_K": K,
            "image_size": (320, 240),
            "pairs": [
                {
                    "new_id": int(negative_ids[row]),
                    "old_id": row + 1,
                    "new_xyz": fresh[row],
                    "old_xyz": frozen_matches[row],
                    "new_xy": project(fresh[row : row + 1], pose)[0],
                    "old_xy": project(
                        frozen_matches[row : row + 1], np.column_stack((np.eye(3), np.zeros(3)))
                    )[0],
                    "old_reference_name": "frozen-reference",
                    "confidence": 1.0,
                }
                for row in range(shown)
            ],
        }
        assert session.record_overlap_evidence(measured, None)
        session.flush_writer()
    snapshot = session.overlap_snapshot()
    assert snapshot is not None
    return session, snapshot, frozen_matches, frozen_support


def test_measured_overlap_corrects_only_new_geometry_and_rejects_stale_replay(tmp_path):
    session, snapshot, expected_new, frozen_support = _overlap_scene(
        tmp_path, protected_old_support=False
    )
    try:
        candidate = _fit_overlap_job(snapshot)
        assert candidate["ok"], candidate.get("reason")
        before_revision = session.revision
        corrected = session.apply_scale_candidate(snapshot, candidate)
        assert len(corrected) == len(expected_new)
        assert session.revision > before_revision
        for index, point in enumerate(expected_new, 1):
            np.testing.assert_allclose(session.landmarks[-index]["xyz"], point, atol=1e-5)
        for frame in session.keyframes.values():
            np.testing.assert_array_equal(frame["ba_xyz"][: len(frozen_support)], frozen_support)
            camera = frame["ba_xyz"] @ frame["pose"][:, :3].T + frame["pose"][:, 3]
            projected = camera @ frame["K"].T
            pixels = projected[:, :2] / projected[:, 2, None]
            np.testing.assert_allclose(pixels, frame["ba_xy"], atol=1e-5)
        assert not session.apply_scale_candidate(snapshot, candidate)
        assert session.last_scale_rejection_reason in {
            "keyframe_pose_revision",
            "landmark_revision",
            "session_epoch_or_writer",
        }
    finally:
        session.close()
    records = [
        json.loads(row)
        for row in (tmp_path / "session" / "geometry_updates.jsonl").read_text().splitlines()
    ]
    assert any(
        row["event"] == "scale_geometry_committed" and row["after_2d_p90"] < row["before_2d_p90"]
        for row in records
    )


def test_overlap_candidate_cannot_damage_existing_old_map_support(tmp_path):
    session, snapshot, _new, old = _overlap_scene(tmp_path, protected_old_support=True)
    try:
        anchored = next(iter(session.keyframes.values()))
        reference = session._reference_from_frame_locked(anchored)
        assert not session.can_match_overlap(
            reference, strong_old_inliers=10, maximum_old_p90_px=3.0
        )
        assert session.can_match_overlap(reference, strong_old_inliers=80, maximum_old_p90_px=3.0)
        candidate = _fit_overlap_job(snapshot)
        assert candidate["ok"], candidate.get("reason")
        prior_poses = {name: frame["pose"].copy() for name, frame in session.keyframes.items()}
        assert not session.apply_scale_candidate(snapshot, candidate)
        assert session.last_scale_rejection_reason in {"old_support_count", "old_reprojection_p90"}
        for name, pose in prior_poses.items():
            np.testing.assert_array_equal(session.keyframes[name]["pose"], pose)
        assert session.revision == snapshot["snapshot_revision"]
        assert all(
            np.array_equal(frame["ba_xyz"][: len(old)], old) for frame in session.keyframes.values()
        )
    finally:
        session.close()
