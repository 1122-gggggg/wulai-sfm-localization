"""Offline position-servo regressions; the simple plant is not an aircraft model."""

from dataclasses import replace
import math

import numpy as np
import pytest

import real_path_follow_controller as rpf


def _config():
    # Smallest sphere and slowdown span in the latest 2026-09-18 flight plan.
    return replace(
        rpf.production_auto_control_config(rpf.LEGACY_MAP_FRAME),
        slowdown_distance=0.4322554798454335,
        waypoint_arrive_radius=0.051909,
    )


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
@pytest.mark.parametrize("wind_mps", [-0.4, -0.3, 0.0, 0.3])
def test_centering_converges_inside_arrival_sphere_under_sustained_disturbance(yaw, wind_mps):
    cfg = _config()
    gate = rpf.YawAlignedPcmdController(cfg)
    cmd = rpf.Command("FOLLOW", np.zeros(3), yaw, np.zeros(3), 0.0, 0.0)
    # Fixed world-east disturbance, first-order PCMD response, 15 m/map unit.
    # Keep asking for the same point to also check settling after first entry.
    x, velocity = -0.12, 0.0
    final_errors = []
    for index in range(1200):
        now = 1.0 + index * 0.05
        ground_speed = velocity + wind_mps
        body_velocity = (ground_speed * math.cos(yaw), ground_speed * math.sin(yaw), now)
        roll, pitch, turn, gaz = gate.update(
            cmd, rpf.Pose(x, 0.0, 0.0, yaw, stamp=now), now,
            target_key=0, body_velocity=body_velocity,
        )
        assert turn == gaz == 0
        assert max(abs(roll), abs(pitch)) <= cfg.max_translation_pcmd
        east_pcmd = pitch * math.cos(yaw) + roll * math.sin(yaw)
        velocity += (east_pcmd * 0.018 - velocity) * (1.0 - math.exp(-0.05 / 0.4))
        x += (velocity + wind_mps) * 0.05 / 15.0
        if index >= 1000:
            final_errors.append(abs(x))
    assert max(final_errors) < cfg.waypoint_arrive_radius


@pytest.mark.parametrize("body_velocity", [None, (0.0, 0.0, -10.0)])
def test_missing_or_stale_speed_decays_existing_wind_compensation(body_velocity):
    gate = rpf.YawAlignedPcmdController(_config())
    cmd = rpf.Command("FOLLOW", np.zeros(3), 0.0, np.zeros(3), 0.0, 0.0)
    for index in range(41):
        now = 1.0 + index * 0.05
        gate.update(cmd, rpf.Pose(-0.1, 0.0, 0.0, 0.0, stamp=now), now,
                    target_key=0, body_velocity=(0.0, 0.0, now))
    accumulated = float(np.linalg.norm(gate._integral_map))
    assert accumulated > 0.0
    for index in range(1, 41):
        now = 3.0 + index * 0.05
        gate.update(cmd, rpf.Pose(-0.1, 0.0, 0.0, 0.0, stamp=now), now,
                    target_key=0, body_velocity=body_velocity)
    assert np.linalg.norm(gate._integral_map) == pytest.approx(accumulated * math.exp(-1.0))


def test_repeated_capture_does_not_accumulate_more_wind_compensation():
    gate = rpf.YawAlignedPcmdController(_config())
    pose = rpf.Pose(-0.1, 0.0, 0.0, 0.0, stamp=1.0)
    gate._position_integral([0.1, 0.0], pose, 1.0, 0.03, 50)
    pose.stamp = 1.1
    gate._position_integral([0.1, 0.0], pose, 1.1, 0.03, 50)
    accumulated = gate._integral_map.copy()
    for now in (1.2, 1.3, 1.4):
        gate._position_integral([0.1, 0.0], pose, now, 0.03, 50)
    np.testing.assert_allclose(gate._integral_map, accumulated)
    # An expired capture still sheds compensation and resets the capture anchor.
    assert gate._position_integral([0.1, 0.0], pose, 1.7, 0.03, 50) == (0.0, 0.0)
    assert np.linalg.norm(gate._integral_map) < np.linalg.norm(accumulated)
    assert gate._integral_now_pose is None


def test_saturated_integral_is_bounded_and_opposing_error_unloads_it():
    gate = rpf.YawAlignedPcmdController(_config())
    for index in range(80):
        now = 1.0 + index * 0.05
        gate._position_integral([10.0, 0.0], rpf.Pose(0, 0, 0, 0, stamp=now), now, 0.03, 50)
    accumulated = gate._integral_map.copy()
    assert np.linalg.norm(accumulated) <= 3.0 * 0.5 * gate.config.slowdown_distance
    gate._position_integral([10.0, 0.0], rpf.Pose(0, 0, 0, 0, stamp=5.0), 5.0,
                            0.03, 50, saturated=True)
    np.testing.assert_allclose(gate._integral_map, accumulated)
    gate._position_integral([-1.0, 0.0], rpf.Pose(0, 0, 0, 0, stamp=5.05), 5.05,
                            0.03, 50, saturated=True)
    assert np.linalg.norm(gate._integral_map) < np.linalg.norm(accumulated)
