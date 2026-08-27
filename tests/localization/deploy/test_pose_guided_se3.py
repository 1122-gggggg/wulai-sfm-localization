from __future__ import annotations

import math

import numpy as np
import pytest

from pose_guided.se3 import (
    cam_from_world,
    camera_center_from_cam_from_world,
    compose_transforms,
    identity_transform,
    invert_transform,
    matrix_from_quaternion_wxyz,
    olympe_yaw_to_map_ccw,
    quaternion_wxyz_from_matrix,
    slerp,
    transform_from_rt,
    transform_point,
    wrap_pi,
)
from pose_guided.visual_anchor import predict_from_fixed_map_from_odom, predict_from_odom_delta


def test_identity_and_axis_translations() -> None:
    identity = identity_transform()
    assert transform_point(identity, [1.0, 2.0, 3.0]) == pytest.approx([1.0, 2.0, 3.0])
    for axis, delta in enumerate(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])):
        transform = transform_from_rt(np.eye(3), delta)
        moved = transform_point(transform, [0.0, 0.0, 0.0])
        expected = [0.0, 0.0, 0.0]
        expected[axis] = 1.0
        assert moved == pytest.approx(expected)


def test_inverse_and_quaternion_roundtrip() -> None:
    yaw = math.pi / 3.0
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    transform = transform_from_rt(rotation, [0.4, -0.2, 1.5])
    restored = compose_transforms(invert_transform(transform), transform)
    assert restored == pytest.approx(np.eye(4), abs=1e-9)
    quat = quaternion_wxyz_from_matrix(rotation)
    assert matrix_from_quaternion_wxyz(quat) == pytest.approx(rotation, abs=1e-9)


def test_yaw90_body_forward_rotates_in_map() -> None:
    yaw = math.pi / 2.0
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    pose = transform_from_rt(rotation, [0.0, 0.0, 0.0])
    moved = transform_point(pose, [1.0, 0.0, 0.0])
    assert moved == pytest.approx([0.0, 1.0, 0.0])


def test_cam_from_world_center_and_inverse() -> None:
    rotation = np.eye(3)
    translation = np.array([0.0, 0.0, 2.0])
    center = camera_center_from_cam_from_world(rotation, translation)
    assert center == pytest.approx([0.0, 0.0, -2.0])
    world = invert_transform(cam_from_world(rotation, translation))
    assert world[:3, 3] == pytest.approx(center)


def test_slerp_halfway_is_45_degrees() -> None:
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = quaternion_wxyz_from_matrix(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    mid = slerp(q0, q1, 0.5)
    rotation = matrix_from_quaternion_wxyz(mid)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    assert angle == pytest.approx(math.pi / 4.0, abs=1e-6)


def test_impossible_quaternion_is_rejected() -> None:
    with pytest.raises(ValueError, match="near-zero"):
        matrix_from_quaternion_wxyz([0.0, 0.0, 0.0, 0.0])


def test_olympe_yaw_conversion_matches_heading_estimator() -> None:
    assert olympe_yaw_to_map_ccw(0.0) == pytest.approx(math.pi / 2.0)
    assert wrap_pi(olympe_yaw_to_map_ccw(math.pi / 2.0)) == pytest.approx(0.0)


def test_two_anchor_formulations_are_equivalent() -> None:
    yaw0 = 0.0
    yaw1 = math.pi / 2.0
    r0 = np.array(
        [[math.cos(yaw0), -math.sin(yaw0), 0.0], [math.sin(yaw0), math.cos(yaw0), 0.0], [0.0, 0.0, 1.0]]
    )
    r1 = np.array(
        [[math.cos(yaw1), -math.sin(yaw1), 0.0], [math.sin(yaw1), math.cos(yaw1), 0.0], [0.0, 0.0, 1.0]]
    )
    map_t0 = transform_from_rt(np.eye(3), [10.0, 20.0, 1.0])
    odom_t0 = transform_from_rt(r0, [0.0, 0.0, 0.0])
    odom_t1 = transform_from_rt(r1, r1 @ np.array([1.0, 0.0, 0.0]))
    first = predict_from_odom_delta(map_t0, odom_t0, odom_t1)
    second = predict_from_fixed_map_from_odom(map_t0, odom_t0, odom_t1)
    assert first == pytest.approx(second)
    assert first[:3, 3] == pytest.approx([10.0, 21.0, 1.0])
