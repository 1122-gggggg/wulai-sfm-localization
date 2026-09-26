import copy

import cv2
import numpy as np
import pytest

from audit_camera_calibration import rectify_observations, raw_anafi_rectifier
from ekf_capture_contract import camera_calibration_evidence, verify_image_domain


INTRINSICS = {
    "image_width": 640,
    "image_height": 480,
    "K": [[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]],
    "dist": [-0.02, 0.01, 0.002, 0.003, -0.002],
}
CAMERA = {"model": "PINHOLE", "width": 640, "height": 480, "params": [500.0, 510.0, 320.0, 240.0]}


@pytest.mark.parametrize("params", [500.0, [500.0], [500.0, 510.0, 320.0], [float("nan")] * 4])
def test_camera_parameters_cannot_broadcast_into_a_false_match(params):
    evidence = camera_calibration_evidence(INTRINSICS, {**CAMERA, "params": params})
    assert not evidence["intrinsics_match_active_pinhole"]


def test_invalid_intrinsic_bottom_row_and_distortion_are_reported():
    intrinsics = copy.deepcopy(INTRINSICS)
    intrinsics["K"][2][2] = 2.0
    intrinsics["dist"][0] = float("nan")
    evidence = camera_calibration_evidence(intrinsics, CAMERA)
    assert not evidence["intrinsics_valid"]
    assert not evidence["distortion_coefficients_valid"]


def test_operator_domain_evidence_is_bound_to_the_exact_calibration():
    provenance = {
        "intrinsics_sha256": "abc",
        "confirmed_by": "operator",
        "image_domain": "anafi_raw_stream",
    }
    evidence = verify_image_domain(provenance, "abc")
    assert evidence["distortion_domain_verified"]
    assert not evidence["pinhole_image_domain_verified"]
    assert not verify_image_domain(provenance, "changed")["distortion_domain_verified"]


def test_offline_frame_rectification_preserves_zero_distortion_and_rejects_wrong_camera():
    intrinsics = {**INTRINSICS, "dist": [0.0] * 5}
    rectify = raw_anafi_rectifier(intrinsics, CAMERA, (640, 480))
    frame = np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    np.testing.assert_array_equal(rectify(frame), frame)
    with pytest.raises(ValueError, match="aspect ratio"):
        rectify(frame[:300])
    with pytest.raises(ValueError, match="match the active query camera"):
        raw_anafi_rectifier(intrinsics, {**CAMERA, "width": 1280}, (640, 480))


def test_scaled_rectification_inverts_known_distortion_without_changing_pinhole_K():
    points = np.array([[-1.0, -0.7, 3.0], [0.8, 0.6, 2.0], [0.0, 0.0, 4.0]])
    K = np.asarray(INTRINSICS["K"])
    distorted, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), K, np.asarray(INTRINSICS["dist"])
    )
    corrected, scaled_K = rectify_observations(distorted.reshape(-1, 2) / 2, INTRINSICS, (320, 240))
    projected = points @ scaled_K.T
    np.testing.assert_allclose(corrected, projected[:, :2] / projected[:, 2:], atol=1e-5)
    np.testing.assert_allclose(scaled_K[:2], K[:2] / 2)
    with pytest.raises(ValueError):
        rectify_observations(distorted, INTRINSICS, (640, 400))
