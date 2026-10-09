"""Sim(3) algebra, anchor consistency, and the submap registry."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from workmap_submaps import (
    ANCHOR_MIN_COUNT,
    AnchorPair,
    Sim3,
    SubmapRegistry,
    anchor_consistency,
    fit_anchor_similarity,
)


def _rot(axis, degrees):
    return cv2.Rodrigues(np.asarray(axis, dtype=float).reshape(3, 1) * np.radians(degrees))[0]


TRUE = Sim3(1.1, _rot([0, 0, 1], 3.0), np.array([0.2, -0.1, 0.05]))


def _local_pose(index):
    rotation = _rot([0, 1, 0], 0.5 * index)
    center = np.array([0.1 * index, 0.02 * np.sin(index), 0.03 * index])
    return np.hstack([rotation, (-rotation @ center).reshape(3, 1)])


def _anchors(count=ANCHOR_MIN_COUNT, step=2):
    out = []
    for index in range(count):
        local = _local_pose(index * step)
        out.append(
            AnchorPair(index * step, TRUE.apply_pose(local), local, np.arange(40), np.zeros((40, 2)))
        )
    return out


def _shift_center(pose, offset):
    center = -pose[:, :3].T @ pose[:, 3] + offset
    return np.hstack([pose[:, :3], (-pose[:, :3] @ center).reshape(3, 1)])


def test_sim3_compose_inverse_and_apply_round_trip():
    other = Sim3(0.8, _rot([1, 0, 0], 20.0), np.array([1.0, 2.0, 3.0]))
    points = np.random.default_rng(0).normal(size=(5, 3))
    np.testing.assert_allclose(
        TRUE.compose(other).apply_points(points), TRUE.apply_points(other.apply_points(points)), atol=1e-12
    )
    np.testing.assert_allclose(TRUE.inverse().apply_points(TRUE.apply_points(points)), points, atol=1e-12)
    identity = TRUE.compose(TRUE.inverse())
    assert identity.scale == pytest.approx(1.0)
    np.testing.assert_allclose(identity.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(identity.translation, 0.0, atol=1e-12)


def test_sim3_vector_round_trip_and_pose_mapping_keeps_the_projection():
    restored = Sim3.from_vector(TRUE.to_vector())
    assert restored.scale == pytest.approx(TRUE.scale)
    np.testing.assert_allclose(restored.rotation, TRUE.rotation, atol=1e-12)
    np.testing.assert_allclose(restored.translation, TRUE.translation, atol=1e-12)
    pose, point = _local_pose(3), np.array([[0.4, 0.1, 5.0]])
    mapped_pose, mapped_point = TRUE.apply_pose(pose), TRUE.apply_points(point)
    camera_local = point @ pose[:, :3].T + pose[:, 3]
    camera_mapped = mapped_point @ mapped_pose[:, :3].T + mapped_pose[:, 3]
    np.testing.assert_allclose(camera_mapped[0] / camera_mapped[0, 2], camera_local[0] / camera_local[0, 2], atol=1e-9)


def test_consistent_anchors_fit_the_true_similarity_and_pass():
    anchors = _anchors()
    sim, residual = fit_anchor_similarity(anchors)
    assert sim.scale == pytest.approx(TRUE.scale, rel=1e-6) and residual < 1e-9
    assert anchor_consistency(anchors, sim) is None


def test_fewer_anchors_than_required_never_pass():
    anchors = _anchors()[: ANCHOR_MIN_COUNT - 1]
    assert anchor_consistency(anchors, TRUE) == "insufficient_anchors"


def test_anchors_packed_into_too_short_a_span_never_pass():
    assert anchor_consistency(_anchors(step=1)[:ANCHOR_MIN_COUNT], TRUE) == "insufficient_anchors"


def test_one_anchor_rotated_three_degrees_is_rejected():
    anchors = _anchors()
    bad = anchors[3]
    rotated = _rot([0, 0, 1], 3.0) @ bad.map_pose[:, :3]
    center = -bad.map_pose[:, :3].T @ bad.map_pose[:, 3]
    pose = np.hstack([rotated, (-rotated @ center).reshape(3, 1)])
    anchors[3] = AnchorPair(bad.ordinal, pose, bad.local_pose, bad.ids, bad.xy)
    assert anchor_consistency(anchors, TRUE) == "rotation_residual"


def test_alternating_small_centre_errors_fail_the_pairwise_check():
    anchors = _anchors()
    shifted = []
    for index, anchor in enumerate(anchors):
        offset = np.array([0.0, 0.0, 0.015 * (1 if index % 2 else -1)])
        shifted.append(AnchorPair(anchor.ordinal, _shift_center(anchor.map_pose, offset), anchor.local_pose, anchor.ids, anchor.xy))
    assert anchor_consistency(shifted, TRUE) == "pair_translation"


def test_large_centre_error_and_scale_bounds_are_named():
    anchors = _anchors()
    anchors[4] = AnchorPair(
        anchors[4].ordinal, _shift_center(anchors[4].map_pose, np.array([0.05, 0, 0])),
        anchors[4].local_pose, anchors[4].ids, anchors[4].xy,
    )
    assert anchor_consistency(anchors, TRUE) == "center_residual"
    assert anchor_consistency(_anchors(), Sim3(2.5, TRUE.rotation, TRUE.translation)) == "scale_bounds"


def test_registry_labels_states_and_landmark_sets():
    registry = SubmapRegistry()
    assert registry.active_id == 0 and registry.get(0)["state"] == "anchored"
    child = registry.open("detached", 0, ordinal=40)
    registry.set_active(child)
    registry.tag_landmarks([-3, -1, -2], child)
    registry.tag_keyframe("kf1", child)
    assert registry.submap_of(-2) == child and registry.submap_of(-99) == 0 and registry.submap_of(7) == 0
    np.testing.assert_array_equal(registry.landmark_ids(child), [-3, -2, -1])
    assert registry.keyframe_submap("kf1") == child
    registry.tag_landmarks([-2], 0)                                   # a re-tag invalidates the cached array
    np.testing.assert_array_equal(registry.landmark_ids(child), [-3, -1])

    registry.set_state(child, "relocalizing")
    assert registry.relocalizing_id == child
    registry.set_state(child, "merging")
    assert registry.relocalizing_id is None
    registry.set_applied(child, TRUE)
    records = registry.records()
    assert [r["id"] for r in records] == [0, child]
    assert records[1]["state"] == "merging" and records[1]["landmarks"] == 2
    np.testing.assert_allclose(records[1]["applied"], TRUE.to_vector())
    with pytest.raises(ValueError):
        registry.set_state(0, "unmerged")
    with pytest.raises(ValueError):
        registry.set_state(child, "nonsense")
