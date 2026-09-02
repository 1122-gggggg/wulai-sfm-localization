"""Unit tests for the ESEKF error-state Kalman filter.

ESEKF (定位演算法/deploy_code/sfm_glomap_deploy/esekf.py) is the only pose-fusion
filter actually wired into the live tracker (production_edm_tracker.py calls
esekf.predict()/update_visual() every frame), but until this file it had zero
dedicated tests anywhere in the repo -- only end-to-end tracker tests that
never isolate its predict/update/gating math. Pure numpy: no CUDA, no torch.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from esekf import (  # noqa: E402
    EKFConfig,
    ESEKF,
    _normalize_q,
    _quat_mul,
    _quat_to_rot,
)

IDENTITY_Q = [1.0, 0.0, 0.0, 0.0]


def _fresh(config: EKFConfig | None = None, *, v=(0.0, 0.0, 0.0)) -> ESEKF:
    ekf = ESEKF(config)
    ekf.reset_from_pose([0.0, 0.0, 0.0], IDENTITY_Q, v=list(v), timestamp=0.0)
    return ekf


# ---------------------------------------------------------------------------
# quaternion helpers
# ---------------------------------------------------------------------------


def test_normalize_q_rejects_non_finite_and_zero_norm_by_returning_identity():
    assert _normalize_q(np.array([0.0, 0.0, 0.0, 0.0])).tolist() == IDENTITY_Q
    assert _normalize_q(np.array([np.nan, 0.0, 0.0, 1.0])).tolist() == IDENTITY_Q


def test_normalize_q_keeps_w_non_negative_for_shortest_path():
    q = _normalize_q(np.array([-1.0, 0.0, 0.0, 0.0]))
    assert q[0] >= 0.0


def test_quat_mul_identity_is_a_no_op():
    q = _normalize_q(np.array([0.9, 0.1, 0.2, 0.3]))
    assert _quat_mul(np.array(IDENTITY_Q), q) == pytest.approx(q)


def test_quat_to_rot_identity_is_the_identity_matrix():
    assert _quat_to_rot(np.array(IDENTITY_Q)) == pytest.approx(np.eye(3))


def test_quat_to_rot_is_always_orthonormal():
    q = _normalize_q(np.array([0.2, -0.4, 0.8, 0.1]))
    R = _quat_to_rot(q)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# construction / config overrides
# ---------------------------------------------------------------------------


def test_init_defaults_to_a_stock_ekfconfig():
    ekf = ESEKF()
    assert ekf.config == EKFConfig()


def test_init_rejects_an_unknown_config_override_instead_of_silently_ignoring_it():
    """A typo'd override must not silently run the filter with default noise."""
    with pytest.raises(TypeError, match="pos_var_typo"):
        ESEKF(pos_var_typo=1.0)


def test_init_applies_a_valid_kwargs_override():
    ekf = ESEKF(pos_var=5.0)
    assert ekf.config.pos_var == 5.0
    assert ekf.config.vel_var == EKFConfig().vel_var  # untouched fields keep defaults


def test_init_applies_metres_per_map_unit_and_gate_threshold_together():
    """These two must not be mutually exclusive; both were requested."""
    ekf = ESEKF(metres_per_map_unit=2.0, gate_threshold=50.0)
    assert ekf.config.metres_per_map_unit == pytest.approx(2.0)
    assert ekf.config.gate_threshold == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_before_reset_returns_zero_state_without_crashing():
    ekf = ESEKF()
    p, q, P, dtheta = ekf.predict(1.0, velocity_ned=(1.0, 2.0, 3.0))
    assert p.tolist() == [0.0, 0.0, 0.0]
    assert dtheta == 0.0


def test_predict_velocity_mode_integrates_constant_velocity():
    ekf = _fresh()
    p, q, P, dtheta = ekf.predict(0.1, velocity_ned=(2.0, -1.0, 0.5))
    assert p == pytest.approx([0.2, -0.1, 0.05])
    assert ekf.v == pytest.approx([2.0, -1.0, 0.5])
    assert dtheta == 0.0
    assert q == pytest.approx(IDENTITY_Q)  # velocity-only mode never rotates q


def test_predict_scales_velocity_ned_by_metres_per_map_unit():
    ekf = _fresh(EKFConfig(metres_per_map_unit=2.0))
    p, *_ = ekf.predict(0.1, velocity_ned=(2.0, 0.0, 0.0))
    assert p == pytest.approx([0.1, 0.0, 0.0])  # (2.0 m/s / 2.0 m-per-unit) * 0.1s


def test_predict_clamps_dt_to_configured_max_dt():
    ekf = _fresh(EKFConfig(max_dt=0.2))
    p, *_ = ekf.predict(10.0, velocity_ned=(1.0, 0.0, 0.0))
    assert p == pytest.approx([0.2, 0.0, 0.0])


def test_predict_treats_a_backwards_timestamp_as_a_no_op():
    ekf = _fresh(v=(1.0, 0.0, 0.0))
    ekf.predict(1.0, velocity_ned=(1.0, 0.0, 0.0))
    p_before = ekf.p.copy()
    p, *_ = ekf.predict(0.5, velocity_ned=(1.0, 0.0, 0.0))  # earlier than last_predict
    assert p == pytest.approx(p_before)


def test_predict_with_a_non_finite_timestamp_does_not_permanently_freeze_the_clock():
    """A single bad timestamp must not corrupt last_predict for all future calls."""
    ekf = _fresh()
    ekf.predict(float("nan"), velocity_ned=(1.0, 0.0, 0.0))
    assert math.isfinite(ekf.last_predict)
    p_before = ekf.p.copy()
    p, *_ = ekf.predict(0.5, velocity_ned=(1.0, 0.0, 0.0))
    assert not np.allclose(p, p_before)  # dt integrated normally, not stuck at 0


def test_predict_age_frames_increments_every_call():
    ekf = _fresh()
    assert ekf._age_frames == 0
    ekf.predict(0.1, velocity_ned=(0.0, 0.0, 0.0))
    ekf.predict(0.2, velocity_ned=(0.0, 0.0, 0.0))
    assert ekf._age_frames == 2


# ---------------------------------------------------------------------------
# update_visual()
# ---------------------------------------------------------------------------


def test_update_visual_before_reset_is_rejected_without_crashing():
    ekf = ESEKF()
    accepted, d2, nu = ekf.update_visual([0.0, 0.0, 0.0], IDENTITY_Q)
    assert accepted is False
    assert d2 == float("inf")
    assert nu.tolist() == [0.0] * 6


def test_update_visual_accepts_a_close_measurement_and_moves_state_toward_it():
    ekf = _fresh()
    accepted, d2, nu = ekf.update_visual(
        [0.3, 0.0, 0.0], IDENTITY_Q, R_visual_pos=0.3, timestamp=0.0
    )
    assert accepted is True
    assert d2 < ekf.config.gate_threshold
    assert 0.0 < ekf.p[0] < 0.3  # pulled toward the measurement, not snapped to it


def test_update_visual_rejects_a_measurement_far_outside_the_gate_and_leaves_state_unchanged():
    ekf = _fresh()
    p_before = ekf.p.copy()
    P_before = ekf.P.copy()
    accepted, d2, nu = ekf.update_visual(
        [100.0, 0.0, 0.0], IDENTITY_Q, R_visual_pos=0.3, timestamp=0.0
    )
    assert accepted is False
    assert d2 > ekf.config.gate_threshold
    assert ekf.p == pytest.approx(p_before)
    assert ekf.P == pytest.approx(P_before)


def test_update_visual_rejects_a_non_finite_measurement():
    ekf = _fresh()
    accepted, d2, nu = ekf.update_visual([float("nan"), 0.0, 0.0], IDENTITY_Q)
    assert accepted is False
    assert d2 == float("inf")


def test_update_visual_accept_resets_age_frames():
    ekf = _fresh()
    ekf.predict(0.1, velocity_ned=(0.0, 0.0, 0.0))
    ekf.predict(0.2, velocity_ned=(0.0, 0.0, 0.0))
    assert ekf._age_frames == 2
    accepted, _, _ = ekf.update_visual(
        [0.01, 0.0, 0.0], IDENTITY_Q, R_visual_pos=0.3, timestamp=0.2
    )
    assert accepted is True
    assert ekf._age_frames == 0


def test_covariance_stays_symmetric_and_non_negative_across_predict_update_cycles():
    ekf = _fresh(v=(0.3, 0.0, 0.0))
    t = 0.0
    for i in range(20):
        t += 0.1
        ekf.predict(t, velocity_ned=(0.3, 0.0, 0.0))
        ekf.update_visual(
            [0.3 * (i + 1), 0.0, 0.0], IDENTITY_Q, R_visual_pos=0.3, timestamp=t
        )
    P = ekf.get_covariance()
    assert P == pytest.approx(P.T)
    assert np.all(np.diag(P) >= 0.0)
    assert np.all(np.isfinite(P))


# ---------------------------------------------------------------------------
# prediction_allowed()
# ---------------------------------------------------------------------------


def test_prediction_allowed_is_false_before_reset():
    ekf = ESEKF()
    assert ekf.prediction_allowed() is False


def test_prediction_allowed_is_true_right_after_a_fresh_reset():
    ekf = _fresh()
    assert ekf.prediction_allowed() is True


def test_prediction_allowed_becomes_false_after_max_age_frames_without_an_update():
    ekf = _fresh(EKFConfig(max_age_frames=3))
    t = 0.0
    for _ in range(3):
        t += 0.05
        ekf.predict(t, velocity_ned=(0.0, 0.0, 0.0))
    assert ekf.prediction_allowed() is False


def test_prediction_allowed_becomes_false_once_position_covariance_grows_too_large():
    ekf = _fresh(EKFConfig(pos_trace_threshold=1.0))
    assert ekf.prediction_allowed() is True
    ekf.P[0:3, 0:3] = np.eye(3) * 10.0  # simulate large accumulated position uncertainty
    assert ekf.prediction_allowed() is False


def test_prediction_allowed_becomes_false_once_yaw_uncertainty_grows_too_large():
    ekf = _fresh(EKFConfig(yaw_sigma_threshold_deg=15.0))
    assert ekf.prediction_allowed() is True
    ekf.P[6:9, 6:9] = np.eye(3) * (math.radians(45.0) ** 2)
    assert ekf.prediction_allowed() is False
