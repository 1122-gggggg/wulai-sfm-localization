import math
import random
from enum import Enum

import pytest

from anafi_pcmd_sim.models import PilotingCommand, TruePosition
from anafi_pcmd_sim.route import (
    ANAFI_4K_STANDARD_PROFILE,
    FlightErrorConfig,
    LocalizationErrorState,
    NavigationFrameConfig,
    RouteConfig,
    RouteStressScenario,
    Waypoint,
    YawAlignmentTracker,
    _arsdk_state_name,
    camera_turn_angles,
    control_decision,
    enu_yaw_from_olympe_heading,
    estimate_navigation_state,
    estimate_stress_navigation_state,
    estimate_velocity_enu,
    estimated_arrival_tolerance_m,
    generate_route,
    interpolated_yaw_at,
    pcmd_physical_setpoints,
    route_deviation_m,
    safety_violation,
    worst_case_route_scenarios,
)


def test_route_has_ten_reproducible_points_and_first_is_within_two_metres() -> None:
    plan = generate_route(42)

    assert plan == generate_route(42)
    assert len(plan.waypoints) == 10
    assert 3.0 <= plan.waypoints[-1].x_m - plan.waypoints[0].x_m <= 4.1
    for index, (previous, current) in enumerate(
        zip(plan.waypoints, plan.waypoints[1:], strict=False),
        start=1,
    ):
        assert current.x_m > previous.x_m
        assert 0.35 <= current.x_m - previous.x_m <= 0.45
        assert abs(current.y_m - plan.waypoints[0].y_m) <= 0.35
        assert abs(current.z_m - previous.z_m) >= 0.35
        if index % 2:
            assert current.y_m > plan.waypoints[0].y_m
            assert current.z_m >= 2.0
        else:
            assert current.y_m < plan.waypoints[0].y_m
            assert current.z_m <= 1.5
    first = plan.waypoints[0]
    assert (
        math.dist(
            (
                plan.sphinx_spawn.x_m,
                plan.sphinx_spawn.y_m,
                plan.sphinx_spawn.z_m,
            ),
            (first.x_m, first.y_m, first.z_m),
        )
        <= 2.0
    )
    assert plan.sphinx_spawn.sphinx_pose == "0.0 0.0 0.2 0 0 0.0"


def test_olympe_ned_heading_is_converted_to_sphinx_enu_yaw() -> None:
    assert enu_yaw_from_olympe_heading(math.pi / 2) == pytest.approx(0.0)
    assert enu_yaw_from_olympe_heading(0.0) == pytest.approx(math.pi / 2)


def test_olympe_flying_state_enum_is_normalized_for_takeoff_retry() -> None:
    class FlyingState(Enum):
        landed = 0
        hovering = 1

    assert _arsdk_state_name(FlyingState.landed) == "landed"
    assert _arsdk_state_name("hovering") == "hovering"
    assert _arsdk_state_name(None) is None


def test_camera_turn_angle_uses_the_shortest_signed_xy_angle() -> None:
    target_angle = math.radians(-179)
    target = Waypoint(
        index=0,
        x_m=math.cos(target_angle),
        y_m=math.sin(target_angle),
        z_m=1,
    )

    _, turn_angle = camera_turn_angles(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=1),
        target,
        camera_yaw_enu_rad=math.radians(179),
    )

    assert math.degrees(turn_angle) == pytest.approx(2.0)


def test_controller_turns_before_translating() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=1),
        Waypoint(index=0, x_m=1, y_m=1, z_m=1),
        camera_yaw_enu_rad=0,
        config=RouteConfig(),
    )

    assert decision.phase == "turn"
    assert decision.camera_turn_angle_enu_rad == pytest.approx(math.pi / 4)
    assert decision.command.roll == 0
    assert decision.command.pitch == 0
    assert decision.command.yaw < 0
    assert decision.command.gaz == 0


def test_yaw_alignment_requires_three_quiet_in_tolerance_updates() -> None:
    config = RouteConfig(
        yaw_tolerance_deg=3.0,
        yaw_alignment_confirmation_updates=3,
        yaw_alignment_max_rate_deg_s=5.0,
    )
    tracker = YawAlignmentTracker()

    assert tracker.update(
        yaw_error_rad=math.radians(2.0),
        estimated_yaw_rad=math.radians(1.0),
        timestamp_s=0.0,
        config=config,
    ) == (False, None)
    assert (
        tracker.update(
            yaw_error_rad=math.radians(2.0),
            estimated_yaw_rad=math.radians(3.0),
            timestamp_s=1.0,
            config=config,
        )[0]
        is False
    )
    assert (
        tracker.update(
            yaw_error_rad=math.radians(2.0),
            estimated_yaw_rad=math.radians(4.0),
            timestamp_s=2.0,
            config=config,
        )[0]
        is False
    )
    confirmed, yaw_rate = tracker.update(
        yaw_error_rad=math.radians(2.0),
        estimated_yaw_rad=math.radians(4.5),
        timestamp_s=3.0,
        config=config,
    )

    assert confirmed is True
    assert yaw_rate == pytest.approx(0.5)


def test_yaw_alignment_resets_when_angle_or_rate_leaves_the_gate() -> None:
    config = RouteConfig(yaw_alignment_confirmation_updates=2)
    tracker = YawAlignmentTracker()
    tracker.update(
        yaw_error_rad=0.0,
        estimated_yaw_rad=0.0,
        timestamp_s=0.0,
        config=config,
    )
    tracker.update(
        yaw_error_rad=0.0,
        estimated_yaw_rad=0.0,
        timestamp_s=1.0,
        config=config,
    )

    confirmed, _ = tracker.update(
        yaw_error_rad=math.radians(4.0),
        estimated_yaw_rad=math.radians(4.0),
        timestamp_s=2.0,
        config=config,
    )

    assert confirmed is False
    assert tracker.confirmation_count == 0


def test_aligned_controller_decomposes_the_3d_line_into_pitch_and_gaz() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=0),
        Waypoint(index=0, x_m=1, y_m=1, z_m=1),
        camera_yaw_enu_rad=math.pi / 4,
        config=RouteConfig(),
    )

    assert decision.phase == "translate"
    assert decision.command.roll == 0
    assert decision.command.pitch > 0
    assert decision.command.yaw == 0
    assert decision.command.gaz > 0
    assert decision.horizontal_fraction == pytest.approx(math.sqrt(2 / 3))
    assert decision.vertical_fraction == pytest.approx(1 / math.sqrt(3))


def test_translation_recomputes_body_forward_right_and_up_components() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=0),
        Waypoint(index=0, x_m=1, y_m=-1, z_m=1),
        camera_yaw_enu_rad=0,
        config=RouteConfig(),
        require_yaw_alignment=False,
    )

    assert decision.phase == "translate"
    assert decision.command.pitch > 0
    assert decision.command.roll > 0
    assert decision.command.yaw == 0
    assert decision.command.gaz > 0
    assert decision.forward_fraction == pytest.approx(1 / math.sqrt(3))
    assert decision.right_fraction == pytest.approx(1 / math.sqrt(3))


def test_pcmd_percentages_map_to_standard_anafi_setpoints() -> None:
    setpoints = pcmd_physical_setpoints(PilotingCommand(roll=-10, pitch=15, yaw=20, gaz=-25))

    assert setpoints == {
        "roll_tilt_command_deg": -2.0,
        "pitch_tilt_command_deg": 3.0,
        "yaw_rate_command_clockwise_deg_s": 14.0,
        "vertical_speed_command_up_m_s": -0.25,
    }
    assert ANAFI_4K_STANDARD_PROFILE.hardware_max_horizontal_speed_m_s == 15.0
    assert ANAFI_4K_STANDARD_PROFILE.hardware_max_vertical_speed_m_s == 4.0
    assert ANAFI_4K_STANDARD_PROFILE.hardware_max_yaw_rate_deg_s == 200.0


def test_navigation_estimation_errors_are_seeded_and_bounded() -> None:
    config = RouteConfig()
    position = TruePosition(timestamp_s=1, x_m=2, y_m=3, z_m=4)
    camera_yaw = math.radians(30)

    first = estimate_navigation_state(
        position,
        camera_yaw,
        rng=random.Random(42),
        config=config.error_model,
    )
    repeated = estimate_navigation_state(
        position,
        camera_yaw,
        rng=random.Random(42),
        config=config.error_model,
    )

    assert config.control_period_s == 1.0
    assert first == repeated
    estimated_position, estimated_yaw, errors = first
    assert abs(estimated_position.x_m - position.x_m) <= 0.30
    assert abs(estimated_position.y_m - position.y_m) <= 0.30
    assert abs(estimated_position.z_m - position.z_m) <= 0.30
    assert errors["position_error_m"] <= 0.30
    assert abs(math.degrees(estimated_yaw - camera_yaw)) <= 10.0


def test_navigation_error_is_time_correlated_without_exceeding_its_caps() -> None:
    config = FlightErrorConfig(localization_error_correlation=1.0)
    state = LocalizationErrorState()
    rng = random.Random(42)
    position = TruePosition(timestamp_s=1, x_m=2, y_m=3, z_m=4)

    first = estimate_navigation_state(position, 0.0, rng=rng, config=config, state=state)
    second = estimate_navigation_state(position, 0.0, rng=rng, config=config, state=state)

    assert first[2] == second[2]
    assert first[2]["position_error_m"] <= 0.30
    assert abs(first[2]["yaw_error_deg"]) <= 10.0


@pytest.mark.parametrize(
    ("bias", "expected_distance"),
    (("toward_target", 0.7), ("away_from_target", 1.3)),
)
def test_stress_position_bias_uses_the_full_thirty_centimetre_bound(
    bias: str,
    expected_distance: float,
) -> None:
    position = TruePosition(timestamp_s=1.0, x_m=0.0, y_m=0.0, z_m=1.0)
    target = Waypoint(index=0, x_m=1.0, y_m=0.0, z_m=1.0)
    estimated, yaw, errors = estimate_stress_navigation_state(
        position,
        0.0,
        target,
        config=FlightErrorConfig(),
        scenario=RouteStressScenario(bias, position_bias=bias, yaw_error_deg=10.0),
    )

    assert math.dist(
        (estimated.x_m, estimated.y_m, estimated.z_m),
        (target.x_m, target.y_m, target.z_m),
    ) == pytest.approx(expected_distance)
    assert errors["position_error_m"] == pytest.approx(0.30)
    assert math.degrees(yaw) == pytest.approx(10.0)
    assert errors["localization_confidence"] == 1.0


def test_worst_case_matrix_is_deterministic_and_covers_every_requested_trigger() -> None:
    scenarios = worst_case_route_scenarios()
    by_name = {scenario.name: scenario for scenario in scenarios}

    assert tuple(by_name) == (
        "false-arrival-bias",
        "overshoot-bias",
        "final-approach-wind",
        "post-command-localization-loss",
        "combined-bounds",
    )
    assert by_name["false-arrival-bias"].position_bias == "toward_target"
    assert by_name["overshoot-bias"].position_bias == "away_from_target"
    assert by_name["final-approach-wind"].final_approach_wind_m == pytest.approx(0.40)
    assert by_name["post-command-localization-loss"].drop_after_nonzero_pcmd is True


def test_velocity_estimate_and_speed_gate_brake_before_arrival() -> None:
    previous = TruePosition(timestamp_s=1, x_m=0, y_m=0, z_m=1)
    current = TruePosition(timestamp_s=2, x_m=0.6, y_m=0, z_m=1)
    velocity = estimate_velocity_enu(previous, current)

    decision = control_decision(
        current,
        Waypoint(index=0, x_m=1, y_m=0, z_m=1),
        camera_yaw_enu_rad=0,
        config=RouteConfig(),
        velocity_enu_m_s=velocity,
    )

    assert velocity == pytest.approx((0.6, 0.0, 0.0))
    assert decision.phase == "brake"
    assert decision.command == PilotingCommand.zero()
    assert decision.estimated_horizontal_speed_m_s == pytest.approx(0.6)
    assert RouteConfig().maximum_cruise_speed_m_s == pytest.approx(0.30)


def test_horizontal_speed_gate_does_not_treat_vertical_speed_as_overspeed() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=2, x_m=0, y_m=0, z_m=1),
        Waypoint(index=0, x_m=3, y_m=0, z_m=1),
        camera_yaw_enu_rad=0,
        config=RouteConfig(),
        velocity_enu_m_s=(0.0, 0.0, 0.31),
    )

    assert decision.estimated_speed_m_s == pytest.approx(0.31)
    assert decision.estimated_horizontal_speed_m_s == pytest.approx(0.0)
    assert decision.phase == "translate"
    assert decision.command.pitch > 0


def test_yaw_latency_interpolation_uses_shortest_wrapped_angle() -> None:
    samples = [
        (1.0, math.radians(170)),
        (2.0, math.radians(-170)),
    ]

    selected = interpolated_yaw_at(samples, 1.5)

    assert selected is not None
    assert abs(math.degrees(selected)) == pytest.approx(180.0)
    assert interpolated_yaw_at(samples, 0.5) is None
    assert interpolated_yaw_at(samples, 2.5) is None


def test_camera_to_body_yaw_extrinsic_rotates_translation_components() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=1),
        Waypoint(index=0, x_m=1, y_m=0, z_m=1),
        camera_yaw_enu_rad=0,
        config=RouteConfig(
            frame=NavigationFrameConfig(camera_yaw_to_body_yaw_deg=90.0),
        ),
        require_yaw_alignment=False,
    )

    assert decision.command.pitch == 0
    assert decision.command.roll > 0


def test_safety_gates_pose_age_confidence_altitude_and_route_deviation() -> None:
    plan = generate_route(42)
    on_route = TruePosition(
        timestamp_s=0,
        x_m=plan.waypoints[0].x_m,
        y_m=plan.waypoints[0].y_m,
        z_m=plan.waypoints[0].z_m,
    )
    config = RouteConfig()

    assert safety_violation(on_route, plan, config, pose_age_s=0.2, confidence=0.8) is None
    assert (
        safety_violation(on_route, plan, config, pose_age_s=0.6, confidence=0.8)
        == "stale_localization"
    )
    assert (
        safety_violation(on_route, plan, config, pose_age_s=0.2, confidence=0.1)
        == "low_localization_confidence"
    )
    assert (
        safety_violation(
            TruePosition(timestamp_s=0, x_m=on_route.x_m, y_m=on_route.y_m, z_m=20),
            plan,
            config,
            pose_age_s=0.2,
            confidence=0.8,
        )
        == "altitude_geofence"
    )
    far = TruePosition(timestamp_s=0, x_m=0, y_m=20, z_m=1)
    assert route_deviation_m(far, plan) > config.maximum_route_deviation_m
    assert safety_violation(far, plan, config, pose_age_s=0.2, confidence=0.8) == "route_deviation"


def test_error_model_contains_only_wind_and_navigation_estimation() -> None:
    assert FlightErrorConfig().__dict__ == {
        "wind_maximum_displacement_m": 0.40,
        "wind_displacement_interval_s": 20.0,
        "maximum_position_error_m": 0.30,
        "maximum_yaw_error_deg": 10.0,
        "localization_latency_s": 0.20,
        "localization_dropout_probability": 0.03,
        "localization_error_correlation": 0.85,
        "localization_error_basis": (
            "operator-observed GlueMap/EDM position and camera-heading caps"
        ),
    }


def test_translation_components_are_recomputed_from_the_latest_position() -> None:
    config = RouteConfig()
    target = Waypoint(index=0, x_m=2, y_m=0, z_m=2)
    first = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=0),
        target,
        camera_yaw_enu_rad=0,
        config=config,
    )
    corrected = control_decision(
        TruePosition(timestamp_s=1, x_m=1.8, y_m=0, z_m=0.2),
        target,
        camera_yaw_enu_rad=0,
        config=config,
    )

    assert corrected.command.pitch < first.command.pitch
    assert corrected.command.gaz > first.command.gaz
    assert corrected.horizontal_fraction < first.horizontal_fraction
    assert corrected.vertical_fraction > first.vertical_fraction


def test_controller_hovers_inside_the_arrival_radius() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=1),
        Waypoint(index=0, x_m=0.1, y_m=0, z_m=1),
        camera_yaw_enu_rad=0,
        config=RouteConfig(),
    )

    assert decision.phase == "hover"
    assert decision.command.roll == 0
    assert decision.command.pitch == 0
    assert decision.command.yaw == 0
    assert decision.command.gaz == 0


def test_estimated_arrival_radius_reserves_true_error_and_latency_motion() -> None:
    config = RouteConfig()

    assert estimated_arrival_tolerance_m(config) == pytest.approx(0.15)

    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=1),
        Waypoint(index=0, x_m=0.4, y_m=0, z_m=1),
        camera_yaw_enu_rad=0,
        config=config,
    )

    assert decision.phase == "translate"


def test_controller_keeps_translating_outside_half_metre_arrival_radius() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=1),
        Waypoint(index=0, x_m=0.6, y_m=0, z_m=1),
        camera_yaw_enu_rad=0,
        config=RouteConfig(),
    )

    assert decision.phase == "translate"
    assert decision.command.pitch > 0


def test_short_xy_projection_does_not_block_vertical_capture() -> None:
    decision = control_decision(
        TruePosition(timestamp_s=0, x_m=0, y_m=0, z_m=1),
        Waypoint(index=0, x_m=0.1, y_m=0.1, z_m=1.6),
        camera_yaw_enu_rad=-math.pi / 2,
        config=RouteConfig(),
    )

    assert decision.distance_m > 0.5
    assert decision.horizontal_distance_m < 0.25
    assert abs(math.degrees(decision.camera_turn_angle_enu_rad)) > 1.0
    assert decision.phase == "translate"
    assert decision.command.gaz > 0
