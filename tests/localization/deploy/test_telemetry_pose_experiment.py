import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "定位演算法/validation"))
from telemetry_pose_experiment import TelemetryPoseESKF


def pose(x=0.0):
    return np.column_stack((np.eye(3), [-x, 0.0, 0.0]))


def seeded():
    f = TelemetryPoseESKF(np.eye(3)[:2])
    f.update(0.0, pose(), epoch=0, position_sigma=0.02, rotation_sigma=0.1)
    return f


def test_velocity_reduces_dropout_prediction_error_without_rewriting_visual_anchor():
    f = seeded()
    assert f.update_velocity(0.0, [1.0, 0.0], epoch=0, sigma=0.02)["accepted"]
    result = f.update(0.4, None, epoch=0, position_sigma=0.02, rotation_sigma=0.1)
    predicted = -result["pose"][:, :3].T @ result["pose"][:, 3]
    assert predicted[0] == pytest.approx(0.4, abs=0.005)
    assert result["predicted_only"] and not result["accepted"]
    assert f._last_accepted_stamp == 0.0
    assert np.linalg.eigvalsh(f._P).min() > 0.0


def test_duplicate_velocity_cannot_reduce_covariance_twice():
    f = seeded()
    f.update_velocity(0.1, [1.0, 0.0], epoch=0, sigma=0.1)
    covariance = f._P.copy()
    result = f.update_velocity(0.1, [1.0, 0.0], epoch=0, sigma=0.1)
    assert not result["accepted"]
    np.testing.assert_array_equal(f._P, covariance)


def test_velocity_rejects_outlier_and_does_not_hide_prolonged_visual_loss():
    f = seeded()
    assert not f.update_velocity(0.1, [100.0, 0.0], epoch=0, sigma=0.1)["accepted"]
    result = f.update_velocity(0.6, [1.0, 0.0], epoch=0, sigma=0.1)
    assert result["reason"] == "no_recent_visual_anchor"
    assert f._pose_matrix() is None


def test_epoch_change_and_reversed_clock_do_not_mix_coordinate_frames():
    f = seeded()
    f.update_velocity(0.1, [1.0, 0.0], epoch=0, sigma=0.1)
    assert not f.update_velocity(0.05, [1.0, 0.0], epoch=0, sigma=0.1)["accepted"]
    assert not f.update_velocity(0.2, [1.0, 0.0], epoch=1, sigma=0.1)["accepted"]


def test_invalid_projection_and_velocity_fail_before_state_update():
    with pytest.raises(ValueError):
        TelemetryPoseESKF(np.ones((2, 3)))
    f = seeded()
    assert not f.update_velocity(0.1, [np.nan, 0.0], epoch=0, sigma=0.1)["accepted"]
