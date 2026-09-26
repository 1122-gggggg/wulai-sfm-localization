from __future__ import annotations

import math
from pathlib import Path

import pytest

import read_only_flight_advisor as advisor


def test_ned_to_map_requires_explicit_metric_and_heading_calibration() -> None:
    calibration = advisor.NedMapCalibration(
        north_yaw_map_rad=0.0,
        map_units_per_meter=2.0,
    )

    mapped = advisor.ned_velocity_to_map(
        north_mps=1.0,
        east_mps=1.0,
        down_mps=0.5,
        calibration=calibration,
    )

    assert mapped.x == pytest.approx(2.0)
    assert mapped.z == pytest.approx(-2.0)
    assert mapped.up == pytest.approx(-1.0)


def test_map_velocity_rotates_into_physical_body_axes() -> None:
    forward = advisor.map_velocity_to_body(
        advisor.MapVelocity(x=1.0, z=0.0, up=0.2),
        body_yaw_map_rad=0.0,
    )
    left_of_body = advisor.map_velocity_to_body(
        advisor.MapVelocity(x=0.0, z=1.0, up=0.0),
        body_yaw_map_rad=0.0,
    )
    forward_after_turn = advisor.map_velocity_to_body(
        advisor.MapVelocity(x=0.0, z=1.0, up=0.0),
        body_yaw_map_rad=math.pi / 2.0,
    )

    assert forward == advisor.BodyVelocity(forward=1.0, right=0.0, up=0.2)
    assert left_of_body.forward == pytest.approx(0.0)
    assert left_of_body.right == pytest.approx(-1.0)
    assert forward_after_turn.forward == pytest.approx(1.0)
    assert forward_after_turn.right == pytest.approx(0.0, abs=1e-12)


def test_velocity_advisor_closes_speed_error_and_saturates_without_device_units() -> None:
    controller = advisor.VelocityAdvisor(
        forward=advisor.AxisPid(kp=1.0, output_limit=1.0),
        right=advisor.AxisPid(kp=10.0, output_limit=0.5),
        up=advisor.AxisPid(kp=1.0, output_limit=1.0),
    )

    recommendation = controller.recommend(
        target=advisor.BodyVelocity(forward=1.0, right=1.0, up=0.0),
        measured=advisor.BodyVelocity(forward=0.4, right=0.0, up=0.0),
        dt_s=0.1,
    )
    settled = controller.recommend(
        target=advisor.BodyVelocity(forward=1.0, right=0.0, up=0.0),
        measured=advisor.BodyVelocity(forward=1.0, right=0.0, up=0.0),
        dt_s=0.1,
    )

    assert recommendation.pitch_forward == pytest.approx(0.6)
    assert recommendation.roll_right == pytest.approx(0.5)
    assert recommendation.gaz_up == pytest.approx(0.0)
    assert settled.pitch_forward == pytest.approx(0.0)


def test_visual_imu_yaw_gate_rejects_a_single_visual_jump() -> None:
    accepted = advisor.check_visual_imu_yaw_delta(
        previous_visual_yaw_rad=math.radians(30.0),
        visual_yaw_rad=math.radians(32.0),
        previous_imu_yaw_rad=math.radians(31.0),
        imu_yaw_rad=math.radians(33.0),
        max_mismatch_rad=math.radians(5.0),
    )
    rejected = advisor.check_visual_imu_yaw_delta(
        previous_visual_yaw_rad=math.radians(31.0),
        visual_yaw_rad=math.radians(95.0),
        previous_imu_yaw_rad=math.radians(30.0),
        imu_yaw_rad=math.radians(32.0),
        max_mismatch_rad=math.radians(5.0),
    )

    assert accepted.accepted is True
    assert rejected.accepted is False
    assert rejected.reason == "VISUAL_IMU_YAW_INCREMENT_MISMATCH"
    assert math.degrees(rejected.mismatch_rad) == pytest.approx(62.0)


def test_short_horizon_prediction_stops_after_the_declared_bound() -> None:
    pose = advisor.MapPose(x=10.0, z=0.0, up=2.0, yaw_map_rad=0.0, stamp_s=1.0)
    velocity = advisor.MapVelocity(x=0.5, z=0.0, up=0.0)

    predicted = advisor.predict_short_horizon(
        pose=pose,
        velocity=velocity,
        yaw_rate_rad_s=0.0,
        to_stamp_s=1.05,
        max_horizon_s=0.20,
    )
    stale = advisor.predict_short_horizon(
        pose=pose,
        velocity=velocity,
        yaw_rate_rad_s=0.0,
        to_stamp_s=1.21,
        max_horizon_s=0.20,
    )

    assert predicted is not None
    assert predicted.x == pytest.approx(10.025)
    assert stale is None


def test_scale_free_fusion_uses_visual_map_velocity_without_metric_calibration() -> None:
    fusion = advisor.ScaleFreeVisualImuFusion(
        max_pose_gap_s=0.20,
        max_yaw_mismatch_rad=math.radians(5.0),
    )
    first = fusion.update(
        visual_pose=advisor.MapPose(
            x=10.0, z=2.0, up=1.0, yaw_map_rad=0.20, stamp_s=1.0
        ),
        roll_rad=0.01,
        pitch_rad=-0.02,
        imu_yaw_rad=0.50,
    )
    second = fusion.update(
        visual_pose=advisor.MapPose(
            x=10.5, z=2.1, up=1.0, yaw_map_rad=0.22, stamp_s=1.1
        ),
        roll_rad=0.03,
        pitch_rad=-0.04,
        imu_yaw_rad=0.52,
    )

    assert first.accepted is False
    assert first.reason == "INITIALIZING_NEEDS_SECOND_VISUAL_POSE"
    assert second.accepted is True
    assert second.state is not None
    assert second.state.velocity.x == pytest.approx(5.0)
    assert second.state.velocity.z == pytest.approx(1.0)
    assert second.state.velocity.up == pytest.approx(0.0)
    assert second.state.roll_rad == pytest.approx(0.03)
    assert second.state.pitch_rad == pytest.approx(-0.04)


def test_scale_free_fusion_rejects_yaw_jump_and_stale_visual_gap() -> None:
    fusion = advisor.ScaleFreeVisualImuFusion(
        max_pose_gap_s=0.20,
        max_yaw_mismatch_rad=math.radians(5.0),
    )
    fusion.update(
        visual_pose=advisor.MapPose(
            x=0.0, z=0.0, up=0.0, yaw_map_rad=0.0, stamp_s=1.0
        ),
        roll_rad=0.0,
        pitch_rad=0.0,
        imu_yaw_rad=0.0,
    )
    jump = fusion.update(
        visual_pose=advisor.MapPose(
            x=0.1, z=0.0, up=0.0, yaw_map_rad=1.0, stamp_s=1.1
        ),
        roll_rad=0.0,
        pitch_rad=0.0,
        imu_yaw_rad=0.02,
    )
    stale = fusion.update(
        visual_pose=advisor.MapPose(
            x=0.2, z=0.0, up=0.0, yaw_map_rad=0.04, stamp_s=1.4
        ),
        roll_rad=0.0,
        pitch_rad=0.0,
        imu_yaw_rad=0.04,
    )

    assert jump.accepted is False
    assert jump.reason == "VISUAL_IMU_YAW_INCREMENT_MISMATCH"
    assert stale.accepted is False
    assert stale.reason == "VISUAL_POSE_GAP_TOO_LARGE_REINITIALIZED"


def test_constant_velocity_kalman_fuses_only_calibrated_map_observations() -> None:
    estimator = advisor.ConstantVelocityKalman(max_prediction_s=0.20)
    estimator.update_visual(
        advisor.MapPose(x=0.0, z=0.0, up=1.0, yaw_map_rad=0.0, stamp_s=1.0),
        position_variance=1e-6,
        yaw_variance=1e-6,
    )
    estimator.update_telemetry(
        stamp_s=1.0,
        velocity=advisor.MapVelocity(x=1.0, z=0.0, up=0.0),
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_map_rad=0.0,
        velocity_variance=1e-6,
        attitude_variance=1e-6,
    )

    estimate = estimator.predict(1.05)

    assert estimate is not None
    assert estimate.pose.x == pytest.approx(0.05, abs=1e-4)
    assert estimate.pose.up == pytest.approx(1.0, abs=1e-4)
    assert estimate.velocity.x == pytest.approx(1.0, abs=1e-4)
    assert estimator.predict(1.30) is None


def test_kalman_accepts_only_an_altitude_already_aligned_to_map_up() -> None:
    estimator = advisor.ConstantVelocityKalman()
    estimator.update_visual(
        advisor.MapPose(x=0.0, z=0.0, up=1.0, yaw_map_rad=0.0, stamp_s=1.0),
        position_variance=1e-6,
        yaw_variance=1e-6,
    )

    estimate = estimator.update_altitude(
        stamp_s=1.0,
        up=1.2,
        variance=1e-6,
    )

    # Equal measurement variances produce the midpoint, rather than letting one
    # altitude source overwrite the visual observation.
    assert estimate.pose.up == pytest.approx(1.1, abs=1e-4)


def test_safety_advisor_escalates_to_recommendations_only() -> None:
    safety = advisor.SafetyAdvisor(drift_samples_before_hover=2)

    first_drift = safety.assess(
        roll_rad=0.0,
        pitch_rad=0.0,
        ground_speed_mps=1.5,
        hover_is_expected=True,
    )
    persistent_drift = safety.assess(
        roll_rad=0.0,
        pitch_rad=0.0,
        ground_speed_mps=1.5,
        hover_is_expected=True,
    )
    high_tilt = safety.assess(
        roll_rad=math.radians(26.0),
        pitch_rad=0.0,
        ground_speed_mps=0.0,
        hover_is_expected=False,
    )
    stale = safety.assess(
        roll_rad=0.0,
        pitch_rad=0.0,
        ground_speed_mps=0.0,
        hover_is_expected=False,
        telemetry_fresh=False,
    )

    assert first_drift.recommendation is advisor.SafetyRecommendation.DRIFT_WARNING
    assert persistent_drift.recommendation is advisor.SafetyRecommendation.HOVER_RECOMMENDED
    assert high_tilt.recommendation is advisor.SafetyRecommendation.HOVER_RECOMMENDED
    assert stale.recommendation is advisor.SafetyRecommendation.MANUAL_TAKEOVER_RECOMMENDED


def test_advisor_has_no_live_backend_or_protocol_dependency() -> None:
    source = Path(advisor.__file__).read_text(encoding="utf-8").lower()

    assert "import olympe" not in source
    assert "olympe_live_backend" not in source
    assert "flight_operator_app" not in source
