from __future__ import annotations

import math

import numpy as np
import pytest

from pose_frame_chain import (
    CameraBodyExtrinsic,
    NavigationPoseTransformer,
    camera_body_extrinsic_from_dict,
    load_camera_body_extrinsic,
    save_camera_body_extrinsic,
)
from site_alignment import SiteAlignment, solve_similarity_alignment


R_BODY_FROM_CAMERA_IDEAL = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)


def _alignment(*, approved: bool = True) -> SiteAlignment:
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 1], [1, 1, 1], [-1, 1, 0]],
        dtype=float,
    )
    return SiteAlignment.from_fit(
        map_frame_id="map-v1",
        site_frame_id="site-v1",
        fit=solve_similarity_alignment(points, points),
        approved=approved,
    )


def _extrinsic(*, approved: bool = True) -> CameraBodyExtrinsic:
    R_C_B = R_BODY_FROM_CAMERA_IDEAL.T
    camera_origin_in_body = np.array([0.2, 0.0, 0.0])
    t_C_B = -R_C_B @ camera_origin_in_body
    return CameraBodyExtrinsic(
        vehicle_id="parrot_anafi@720p_v1",
        body_frame_id="parrot_anafi_frd",
        camera_frame_id="front_camera_opencv",
        rotation_camera_from_body=tuple(tuple(value for value in row) for row in R_C_B),
        translation_camera_from_body_m=tuple(t_C_B),
        fixed_gimbal_pitch_deg=-10.0,
        gimbal_pitch_tolerance_deg=1.0,
        evidence="measured fixture",
        approved=approved,
    )


def test_chain_recovers_body_origin_and_orientation_from_colmap_pose() -> None:
    transformer = NavigationPoseTransformer(_alignment(), _extrinsic())
    R_M_C = R_BODY_FROM_CAMERA_IDEAL

    pose = transformer.body_pose(
        camera_center_map=[0.2, 0.0, 0.0],
        rotation_camera_from_map=R_M_C.T,
        gimbal_pitch_deg=-10.2,
        stamp=4.0,
    )

    assert pose.xyz == pytest.approx([0.0, 0.0, 0.0])
    assert pose.R_W_B == pytest.approx(np.eye(3))
    assert pose.yaw_rad == pytest.approx(0.0)


def test_chain_uses_body_forward_for_navigation_yaw() -> None:
    transformer = NavigationPoseTransformer(_alignment(), _extrinsic())
    yaw = math.pi / 2.0
    R_W_B = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    R_M_C = R_W_B @ R_BODY_FROM_CAMERA_IDEAL

    pose = transformer.body_pose(
        camera_center_map=R_W_B @ [0.2, 0.0, 0.0],
        rotation_camera_from_map=R_M_C.T,
        gimbal_pitch_deg=-10.0,
        stamp=5.0,
    )

    assert pose.xyz == pytest.approx([0.0, 0.0, 0.0])
    assert pose.R_W_B == pytest.approx(R_W_B)
    assert pose.yaw_rad == pytest.approx(math.pi / 2.0)


def test_gimbal_mismatch_and_unapproved_calibrations_fail_closed() -> None:
    with pytest.raises(ValueError, match="site alignment"):
        NavigationPoseTransformer(_alignment(approved=False), _extrinsic())
    with pytest.raises(ValueError, match="extrinsic"):
        NavigationPoseTransformer(_alignment(), _extrinsic(approved=False))

    transformer = NavigationPoseTransformer(_alignment(), _extrinsic())
    with pytest.raises(ValueError, match="gimbal pitch"):
        transformer.body_pose(
            camera_center_map=[0, 0, 0],
            rotation_camera_from_map=np.eye(3),
            gimbal_pitch_deg=-20.0,
            stamp=1.0,
        )


def test_extrinsic_json_roundtrip_and_vehicle_binding(tmp_path) -> None:
    path = save_camera_body_extrinsic(_extrinsic(), tmp_path / "extrinsic.json")

    assert load_camera_body_extrinsic(
        path, expected_vehicle_id="parrot_anafi@720p_v1"
    ) == _extrinsic()
    with pytest.raises(ValueError, match="active vehicle"):
        load_camera_body_extrinsic(path, expected_vehicle_id="other")


def test_malformed_or_left_handed_extrinsic_is_rejected() -> None:
    raw = _extrinsic().to_dict()
    raw["T_camera_from_body"]["R"][0][1] *= -1.0

    with pytest.raises(ValueError):
        camera_body_extrinsic_from_dict(raw)
    raw = _extrinsic().to_dict()
    raw["extra"] = True
    with pytest.raises(ValueError, match="exactly"):
        camera_body_extrinsic_from_dict(raw)


def test_map_distance_scales_to_meters() -> None:
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 1], [1, 1, 1], [-1, 1, 0]],
        dtype=float,
    )
    alignment = SiteAlignment.from_fit(
        map_frame_id="map-v1",
        site_frame_id="site-v1",
        fit=solve_similarity_alignment(points, points * 2.5),
        approved=True,
    )
    transformer = NavigationPoseTransformer(alignment, _extrinsic())

    assert transformer.map_distance_to_site(0.4) == pytest.approx(1.0)
