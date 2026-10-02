import math

import numpy as np
import pytest

from control_state_estimator import ControlStateEstimator
import path_follow_flight as pff
import real_path_follow_controller as rpf


def _trained(*, scale=15., offset=0.):
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(901):
        stamp = index * .05
        # True NED east velocity .5 m/s, expressed in the rotated map basis.
        estimator.observe_telemetry((math.pi / 2, stamp), (0., .5, stamp), stamp)
        pose = rpf.Pose(.5 * stamp / scale * math.cos(offset), 0.,
                        .5 * stamp / scale * math.sin(offset), offset, stamp=stamp)
        estimator.observe_pose(pose, stamp, offset)
    return estimator, pose


def test_delayed_yaw_is_interpolated_across_the_pi_wrap():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    estimator.observe_telemetry((math.radians(179), 1.), None, 1.)
    estimator.observe_telemetry((math.radians(-179), 1.2), None, 1.2)
    assert abs(estimator.yaw_at(1.1)) == pytest.approx(math.pi)
    assert estimator.yaw_at(.9) is None
    assert estimator.yaw_at(1.3) is None


def test_cached_future_and_stale_samples_do_not_refresh_history():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for now, stamp in [(1., 1.), (1.1, 1.), (1.2, 2.), (2., 1.), (2., float('nan'))]:
        estimator.observe_telemetry((0., stamp), (0., .5, stamp), now)
    assert len(estimator.yaws) == len(estimator.velocities) == 1
    assert estimator.displacement(1., 2., extrapolate=True) is None


@pytest.mark.parametrize("offset", [0., .7, -1.5])
def test_observed_motion_teaches_scale_and_predicts_without_retiming_pose(offset):
    estimator, pose = _trained(offset=offset)
    assert estimator.scale_m_per_unit == pytest.approx(15.)
    stamp = pose.stamp
    xyz, info = estimator.predict(pose, stamp + .2, offset, .05)
    expected = np.array([math.cos(offset), 0., math.sin(offset)]) * .1 / 15.
    np.testing.assert_allclose(xyz - pose.xyz, expected, atol=1e-9)
    assert info['applied'] and not info['limited']
    assert pose.stamp == stamp and pose.map_confirmed and pose.position_observed


def test_unscaled_stationary_or_weak_positions_never_enable_prediction():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(901):
        now = index * .05
        estimator.observe_telemetry((0., now), (0., .5, now), now)
        pose = rpf.Pose(now / 30, 0., 0., 0., stamp=now, map_confirmed=False)
        estimator.observe_pose(pose, now, 0.)
    xyz, info = estimator.predict(pose, now + .1, 0., .05)
    np.testing.assert_array_equal(xyz, pose.xyz)
    assert not info['applied'] and estimator.scale_m_per_unit is None


def test_bridge_uses_observation_time_and_cannot_extend_prediction_forever():
    estimator, observed = _trained()
    bridge = rpf.Pose(*observed.xyz, 0., stamp=observed.stamp + .3,
                      map_confirmed=False, position_observed=False)
    estimator.observe_pose(bridge, bridge.stamp, 0.)
    xyz, info = estimator.predict(bridge, bridge.stamp, 0., .05)
    assert info['horizon_s'] == pytest.approx(.3)
    assert xyz[0] > bridge.x
    assert not bridge.map_confirmed and not bridge.position_observed
    later = bridge.stamp + .3
    estimator.observe_telemetry((0., later), (0., .5, later), later)
    xyz, info = estimator.predict(bridge, later, 0., .05)
    assert not info['applied'] and info['reason'] == 'prediction_horizon_expired'


def test_prediction_is_bounded_and_reset_discards_scale_and_anchor():
    estimator, pose = _trained()
    xyz, info = estimator.predict(pose, pose.stamp + .5, 0., .002)
    assert info['limited']
    assert np.linalg.norm(xyz - pose.xyz) == pytest.approx(.001)
    estimator.reset_position()
    _, info = estimator.predict(pose, pose.stamp + .1, 0., .05)
    assert not info['applied']


def test_velocity_gap_is_not_interpolated_as_observed_motion():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for stamp in (1., 1.1, 2., 2.1):
        estimator.observe_telemetry((0., stamp), (0., .5, stamp), stamp)
    assert estimator.displacement(1.05, 2.05) is None


def test_scale_learning_waits_for_delayed_telemetry_without_extrapolating_evidence():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(1100):
        now = 1. + index * .05
        estimator.observe_telemetry((math.pi / 2, now), (0., .5, now - .3), now)
        pose = rpf.Pose((now - .1) / 30., 0, 0, 0, stamp=now - .1)
        estimator.observe_pose(pose, now, 0.)
    assert estimator.scale_m_per_unit == pytest.approx(15.)
    assert estimator.scale_stamp <= now - .3


def test_flight_loop_anchors_heading_at_exposure_time_during_a_turn():
    clock = [1.]
    config = rpf.ControlConfig(inspect_waypoints=())
    ctrl = rpf.RouteAutoController([np.zeros(3), np.array([10., 0., 0.])], config=config)
    hooks = pff.LoopHooks(
        get_pose=lambda: rpf.Pose(.5, 0, 0, .3 * (clock[0] - .2), stamp=clock[0] - .2),
        olympe_yaw=lambda: math.pi / 2 - .3 * clock[0],
        attitude_sample=lambda: (math.pi / 2 - .3 * clock[0], clock[0]),
        send_pcmd=lambda *_args: None, now=lambda: clock[0],
    )
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False,
                                   enforce_weak_pose_gate=False)
    record = {}
    for index in range(30):
        clock[0] = 1. + index * .05
        record = {}
        runner._tick(clock[0], 'AUTO', record)
    assert record['heading_time_aligned']
    assert runner.heading.offset == pytest.approx(0., abs=1e-10)
    assert runner.heading.heading(hooks.olympe_yaw()) == pytest.approx(.3 * clock[0])


def test_control_prediction_inside_sphere_does_not_confirm_raw_position_outside():
    estimator, pose = _trained()
    cfg = rpf.ControlConfig(inspect_waypoints=(), waypoint_arrive_radius=.01)
    ctrl = rpf.RouteAutoController([np.zeros(3), np.array([pose.x + .012, 0., 0.])], config=cfg)
    ctrl.target_index = 1
    sent = []
    hooks = pff.LoopHooks(get_pose=lambda: pose, olympe_yaw=lambda: math.pi / 2,
                          ned_velocity=lambda: (0., .5, pose.stamp),
                          send_pcmd=lambda *pcmd: sent.append(pcmd))
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False,
                                   enforce_weak_pose_gate=False)
    runner.control_state = estimator
    runner.heading.update(0., math.pi / 2)
    record = {}
    runner._localized_motion_outcome(pose, math.pi / 2, pose.stamp + .2, record)
    assert record['control_prediction']['applied']
    assert np.linalg.norm(ctrl.wp[1] - record['control_pose_u']) < cfg.waypoint_arrive_radius
    assert np.linalg.norm(ctrl.wp[1] - record['pose_u']) > cfg.waypoint_arrive_radius
    assert ctrl._final_confirm_frames == 0
    assert record['pose_stamp'] == pose.stamp
    assert ctrl.state != 'LANDING'


def test_unmatched_capture_cannot_move_an_existing_heading_anchor():
    ctrl = rpf.RouteAutoController([np.zeros(3), np.ones(3)], config=rpf.ControlConfig(inspect_waypoints=()))
    hooks = pff.LoopHooks(get_pose=lambda: None, olympe_yaw=lambda: 0.,
                          attitude_sample=lambda: None, send_pcmd=lambda *_args: None)
    runner = pff._FlightLoopRunner(hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False,
                                   enforce_weak_pose_gate=False)
    runner.heading.update(.2, math.pi / 2)
    anchor_offset = runner.heading.offset
    record = {}
    assert runner._anchor_heading(rpf.Pose(0, 0, 0, 1., stamp=1.), 0., record)
    assert not record['heading_time_aligned']
    assert runner.heading.offset == anchor_offset
    assert runner.heading.heading(None) == pytest.approx(1.0)


def test_anchor_heading_missing_imu_preserves_offset_and_refreshes_visual_fallback():
    ctrl = rpf.RouteAutoController(
        [np.zeros(3), np.ones(3)], config=rpf.ControlConfig(inspect_waypoints=())
    )
    hooks = pff.LoopHooks(
        get_pose=lambda: None,
        olympe_yaw=lambda: 0.0,
        attitude_sample=lambda: None,
        send_pcmd=lambda *_args: None,
    )
    runner = pff._FlightLoopRunner(
        hooks, ctrl, ctrl.wp, yaw_sign=1, verbose=False, enforce_weak_pose_gate=False
    )
    runner.heading.update(0.0, math.pi / 2)
    established_offset = runner.heading.offset
    assert established_offset is not None

    record = {}
    pose = rpf.Pose(0, 0, 0, math.pi / 2, stamp=1.0)
    assert runner._anchor_heading(pose, 0.0, record)
    assert not record["heading_time_aligned"]
    assert runner.heading.offset == established_offset
    assert runner.heading.heading(None) == pytest.approx(math.pi / 2)
