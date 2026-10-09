"""Offline tests for the visual-only pose ESKF experiment."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "定位演算法" / "validation"))

from pose_filter_experiment import (  # noqa: E402
    VisualPoseESKF,
    _exp_so3,
    _left_jacobian,
    _log_so3,
)


def pose(center, rotation=None):
    rotation_wc = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    rotation_cw = rotation_wc.T
    return np.column_stack((rotation_cw, -rotation_cw @ np.asarray(center, dtype=float)))


def position_from(result):
    matrix = np.asarray(result["pose"], dtype=float)
    return -matrix[:, :3].T @ matrix[:, 3]


def rotation_from(result):
    return np.asarray(result["pose"], dtype=float)[:, :3].T


def test_constant_velocity_prediction_survives_a_visual_dropout():
    filt = VisualPoseESKF(acceleration_sigma=2.0, angular_acceleration_sigma=0.2)
    first = filt.update(
        0.0, pose([0.0, 0.0, 0.0]), epoch=4, position_sigma=0.01, rotation_sigma=0.01
    )
    assert first["accepted"] and first["reset"]
    second = filt.update(
        0.1, pose([0.1, 0.0, 0.0]), epoch=4, position_sigma=0.01, rotation_sigma=0.01
    )
    assert second["accepted"] and not second["predicted_only"]

    dropout = filt.update(0.2, None, epoch=4, position_sigma=0.01, rotation_sigma=0.01)
    assert dropout["pose"] is not None
    assert dropout["predicted_only"] and not dropout["accepted"]
    assert position_from(dropout)[0] > 0.05
    assert np.min(np.linalg.eigvalsh(filt._P)) >= -1e-9


def test_turning_measurements_teach_angular_velocity_for_prediction():
    filt = VisualPoseESKF(acceleration_sigma=0.2, angular_acceleration_sigma=2.0)
    sigma_p = 0.01
    sigma_r = 1e-5
    assert filt.update(
        0.0, pose([0.0, 0.0, 0.0]), epoch=1, position_sigma=sigma_p, rotation_sigma=sigma_r
    )["accepted"]
    assert filt.update(
        0.1,
        pose([0.0, 0.0, 0.0], _exp_so3(np.array([0.0, 0.0, 0.1]))),
        epoch=1,
        position_sigma=sigma_p,
        rotation_sigma=sigma_r,
    )["accepted"]
    turning = filt.update(0.2, None, epoch=1, position_sigma=sigma_p, rotation_sigma=sigma_r)
    assert turning["predicted_only"]
    predicted_angle = math.atan2(rotation_from(turning)[1, 0], rotation_from(turning)[0, 0])
    assert predicted_angle > 0.01


def test_isolated_pose_outlier_is_gated_without_becoming_state():
    filt = VisualPoseESKF(acceleration_sigma=2.0, angular_acceleration_sigma=0.2)
    kwargs = {"epoch": 2, "position_sigma": 0.05, "rotation_sigma": 0.01}
    assert filt.update(0.0, pose([0.0, 0.0, 0.0]), **kwargs)["accepted"]
    assert filt.update(0.1, pose([0.1, 0.0, 0.0]), **kwargs)["accepted"]
    outlier = filt.update(0.2, pose([100.0, 0.0, 0.0]), **kwargs)
    assert not outlier["accepted"] and outlier["predicted_only"]
    assert outlier["innovation_mahalanobis"] is not None
    assert outlier["innovation_mahalanobis"] > 12.592
    assert position_from(outlier)[0] < 1.0
    recovered = filt.update(0.3, pose([0.3, 0.0, 0.0]), **kwargs)
    assert recovered["accepted"]


def test_epoch_change_and_large_gap_reseed_from_the_next_visual_pose():
    filt = VisualPoseESKF(max_gap_s=0.5)
    kwargs = {"position_sigma": 0.02, "rotation_sigma": 0.02}
    assert filt.update(10.0, pose([1.0, 0.0, 0.0]), epoch=1, **kwargs)["accepted"]
    changed = filt.update(0.0, pose([5.0, 0.0, 0.0]), epoch=2, **kwargs)
    assert changed["accepted"] and changed["reset"]
    assert np.isclose(position_from(changed)[0], 5.0)
    gapped = filt.update(1.0, pose([9.0, 0.0, 0.0]), epoch=2, **kwargs)
    assert gapped["accepted"] and gapped["reset"]
    assert np.isclose(position_from(gapped)[0], 9.0)


def test_invalid_and_reverse_timestamps_are_rejected_without_state_corruption():
    filt = VisualPoseESKF()
    kwargs = {"epoch": 7, "position_sigma": 0.02, "rotation_sigma": 0.02}
    accepted = filt.update(1.0, pose([2.0, 0.0, 0.0]), **kwargs)
    assert accepted["accepted"]
    before = np.asarray(accepted["pose"], dtype=float).copy()
    reverse = filt.update(0.5, pose([20.0, 0.0, 0.0]), **kwargs)
    assert not reverse["accepted"] and not reverse["reset"]
    np.testing.assert_allclose(reverse["pose"], before)
    invalid = filt.update(1.1, np.zeros((2, 4)), **kwargs)
    assert not invalid["accepted"] and invalid["predicted_only"]


def test_prediction_stops_after_max_gap_until_a_new_visual_seed():
    filt = VisualPoseESKF(max_gap_s=0.5)
    kwargs = {"epoch": 3, "position_sigma": 0.02, "rotation_sigma": 0.02}
    assert filt.update(0.0, pose([0.0, 0.0, 0.0]), **kwargs)["accepted"]
    short = filt.update(0.2, None, **kwargs)
    assert short["predicted_only"] and short["pose"] is not None
    expired = filt.update(0.6, None, **kwargs)
    assert expired["reset"] and not expired["predicted_only"] and expired["pose"] is None
    reseeded = filt.update(0.7, pose([1.0, 0.0, 0.0]), **kwargs)
    assert reseeded["accepted"] and reseeded["reset"]


def test_so3_left_jacobian_matches_finite_difference_transport():
    phi = np.array([0.2, -0.1, 0.3])
    direction = np.array([0.4, 0.7, -0.2])
    direction /= np.linalg.norm(direction)
    epsilon = 1e-7
    finite_difference = _log_so3(_exp_so3(phi + epsilon * direction) @ _exp_so3(-phi)) / epsilon
    np.testing.assert_allclose(finite_difference, _left_jacobian(phi) @ direction, atol=1e-6)


def test_turning_covariance_keeps_theta_angular_velocity_correlation():
    filt = VisualPoseESKF(angular_acceleration_sigma=2.0)
    kwargs = {"epoch": 5, "position_sigma": 0.02, "rotation_sigma": 0.01}
    assert filt.update(0.0, pose([0.0, 0.0, 0.0]), **kwargs)["accepted"]
    predicted = filt.update(0.1, None, **kwargs)
    assert predicted["predicted_only"]
    assert np.linalg.norm(filt._P[6:9, 9:12]) > 0.0
    measured = filt.update(
        0.2,
        pose([0.0, 0.0, 0.0], _exp_so3(np.array([0.0, 0.0, 0.1]))),
        **kwargs,
    )
    assert measured["accepted"]
    np.testing.assert_allclose(filt._P, filt._P.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(filt._P)) >= -1e-9


def test_nonfinite_innovation_reports_none_for_json_safe_diagnostics():
    filt = VisualPoseESKF()
    kwargs = {"epoch": 6, "position_sigma": 0.02, "rotation_sigma": 0.02}
    assert filt.update(0.0, pose([0.0, 0.0, 0.0]), **kwargs)["accepted"]
    filt._P[0, 0] = np.nan
    accepted, innovation = filt._measurement_update(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0.02, 0.02
    )
    assert not accepted and innovation is None
