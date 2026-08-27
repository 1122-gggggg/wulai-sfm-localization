from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


VALIDATION_DIR = Path(__file__).resolve().parents[1]
TRACKER_DIR = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
for directory in (VALIDATION_DIR, TRACKER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from projection_guided_tracker import (
    TrackLandmarkSidecar,
    extrapolate_tcw,
    match_projected_landmarks,
    ordered_names_sha256,
    project_visible_landmarks,
)


def _write_sidecar(path: Path, names: list[str], bundle_hash: str) -> None:
    descriptors = np.zeros((2, 64), dtype=np.float16)
    descriptors[0, 0] = 1.0
    descriptors[1, 1] = 1.0
    np.savez(
        path,
        point3D_id=np.array([-1, 10], dtype=np.int64),
        xyz=np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32),
        descriptor=descriptors,
        obs_offsets=np.array([0, 1, 2], dtype=np.int64),
        obs_ref_idx=np.array([0, 1], dtype=np.int32),
        obs_kp_idx=np.array([0, 0], dtype=np.int32),
        ref_kp_offsets=np.array([0, 2, 3], dtype=np.int64),
        ref_kp_landmark_idx=np.array([0, 1, 1], dtype=np.int32),
        schema_name=np.array("xfeat-track-landmarks"),
        schema_version=np.array(1, dtype=np.int32),
        ref_bundle_sha256=np.array(bundle_hash),
        ref_names_sha256=np.array(ordered_names_sha256(names)),
        source_features_sha256=np.array("1" * 64),
        source_images_sha256=np.array("2" * 64),
        source_points3D_sha256=np.array("3" * 64),
    )


def test_sidecar_validates_bundle_names_and_reference_lookup(tmp_path: Path) -> None:
    path = tmp_path / "landmarks.npz"
    names = ["a.jpg", "b.jpg"]
    bundle_hash = "a" * 64
    _write_sidecar(path, names, bundle_hash)

    sidecar = TrackLandmarkSidecar.load(path, names, bundle_hash)

    assert sidecar.landmarks_for_refs([1]).tolist() == [1]
    assert sidecar.landmarks_for_refs([0, 1]).tolist() == [0, 1]
    with pytest.raises(ValueError, match="ordered reference names"):
        TrackLandmarkSidecar.load(path, list(reversed(names)), bundle_hash)
    with pytest.raises(ValueError, match="bundle SHA-256"):
        TrackLandmarkSidecar.load(path, names, "b" * 64)


def test_se3_constant_velocity_extrapolates_translation_and_rotation() -> None:
    previous = np.eye(4)
    last = np.eye(4)
    last[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(10.0)]))[0]
    last[0, 3] = 1.0

    predicted, scale = extrapolate_tcw(previous, last, 0.0, 1.0, 2.0)

    assert scale == pytest.approx(1.0)
    predicted_angle = np.linalg.norm(cv2.Rodrigues(predicted[:3, :3])[0])
    assert np.rad2deg(predicted_angle) == pytest.approx(20.0, abs=1e-5)
    # A constant SE(3) screw motion rotates the second translation increment.
    expected_translation = [1.0 + np.cos(np.deg2rad(10.0)),
                            np.sin(np.deg2rad(10.0)), 0.0]
    np.testing.assert_allclose(predicted[:3, 3], expected_translation, atol=1e-6)


def test_full_opencv_projection_filters_negative_depth_and_image_bounds() -> None:
    cam = SimpleNamespace(
        model="FULL_OPENCV",
        width=100,
        height=80,
        params=[100.0, 100.0, 50.0, 40.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0],
    )
    xyz = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [10.0, 0.0, 1.0],
    ], dtype=np.float32)

    indices, projected = project_visible_landmarks(xyz, np.eye(4), cam)

    assert indices.tolist() == [0]
    np.testing.assert_allclose(projected, [[50.0, 40.0]], atol=1e-6)


def test_spatial_cosine_matching_is_one_to_one_and_radius_limited() -> None:
    landmark_descriptor = torch.zeros((3, 64), dtype=torch.float32)
    landmark_descriptor[0, 0] = 1.0
    landmark_descriptor[1, 1] = 1.0
    landmark_descriptor[2, 0] = 1.0
    query_descriptor = torch.zeros((2, 64), dtype=torch.float32)
    query_descriptor[0, 0] = 1.0
    query_descriptor[1, 1] = 1.0
    projected = np.array([[10.0, 10.0], [30.0, 10.0], [11.0, 10.0]], np.float32)
    query = np.array([[11.0, 10.0], [31.0, 10.0]], np.float32)

    results = match_projected_landmarks(
        projected,
        np.array([0, 1, 2]),
        query,
        query_descriptor,
        landmark_descriptor,
        radii=(1.5, 5.0),
        min_score=0.85,
        ratio=0.8,
    )

    landmark_ids, query_ids, scores = results[5.0]
    assert len(landmark_ids) == len(np.unique(landmark_ids))
    assert len(query_ids) == len(np.unique(query_ids))
    assert set(query_ids.tolist()) == {0, 1}
    assert np.all(scores >= 0.85)
    assert len(results[1.5][0]) == 2
