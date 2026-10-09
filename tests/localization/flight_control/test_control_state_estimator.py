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


def test_map_scale_is_cumulative_and_uniform_for_the_whole_map():
    estimator, pose = _trained()
    info = estimator.scale_diagnostics(pose.stamp)
    assert info["reliable"]
    assert info["candidate_m_per_unit"] == pytest.approx(15.)
    assert info["scale_segments"] > 12
    assert info["cumulative_distance_m"] >= 10.
    assert info["relative_iqr"] == pytest.approx(0., abs=1e-12)
    assert info["relative_drift"] == pytest.approx(0., abs=1e-12)
    assert estimator.metric_distance(2., pose.stamp) == pytest.approx(30.)
    np.testing.assert_allclose(
        estimator.metric_xyz([2., -3., 4.], pose.stamp), [30., -45., 60.]
    )
    # Same-map observation/controller holds break pairs, not global evidence.
    estimator.reset_observation()
    assert estimator.scale_diagnostics(pose.stamp)["scale_segments"] == info["scale_segments"]
    assert estimator.metric_distance(2., pose.stamp) == pytest.approx(30.)
    assert estimator.last_observed is None
    estimator.reset_position()
    assert estimator.metric_distance(2., pose.stamp) is None
    assert estimator.scale_diagnostics(pose.stamp)["scale_segments"] == 0


def test_candidate_is_visible_before_reliable_but_never_converts_map_units():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(121):
        stamp = index * .05
        estimator.observe_telemetry(None, (0., .5, stamp), stamp)
        pose = rpf.Pose(stamp / 30., 0., 0., 0., stamp=stamp)
        estimator.observe_pose(pose, stamp, 0.)
    info = estimator.scale_diagnostics(stamp)
    assert info["candidate_m_per_unit"] == pytest.approx(15.)
    assert not info["reliable"] and info["reason"] == "insufficient_segments"
    assert estimator.metric_distance(1., stamp) is None
    assert estimator.metric_xyz([1., 2., 3.], stamp) is None


def test_slow_start_retains_usable_endpoints_at_interval_expiry():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    distance = 0.
    previous_speed = 0.
    for index in range(97):
        stamp = index * .05
        speed = .6 if stamp > 3. else 0.
        distance += .5 * (previous_speed + speed) * .05
        previous_speed = speed
        estimator.observe_telemetry(None, (0., speed, stamp), stamp)
        estimator.observe_pose(rpf.Pose(distance / 15., 0., 0., 0., stamp=stamp), stamp, 0.)
    # The old reset-to-current at 4.5s discarded the usable 3.0–4.8s leg.
    assert len(estimator.scale_samples) == 1
    assert estimator.scale_candidate == pytest.approx(15.)
    assert estimator.scale_diagnostics(stamp)["rejection_counts"]["interval_coverage_expired"] > 0
    assert estimator.scale_m_per_unit is None


@pytest.mark.parametrize(
    ("direction", "visual_speed", "reason"),
    [(1., .005, "scale_outlier"), (-1., .5 / 15., "direction_inconsistent")],
)
def test_bad_segment_disables_reliability_without_erasing_coherent_map_evidence(
    direction, visual_speed, reason
):
    estimator, pose = _trained()
    count = len(estimator.scale_samples)
    estimator.reset_observation()
    for index in range(51):
        stamp = pose.stamp + index * .05
        estimator.observe_telemetry(None, (0., .5, stamp), stamp)
        sample = rpf.Pose(
            pose.x + direction * visual_speed * (stamp - pose.stamp),
            0., 0., 0., stamp=stamp,
        )
        estimator.observe_pose(sample, stamp, 0.)
    # Outlier visual motion must also pass the original .02u displacement
    # gate; use 4.5s of covered motion for that case.
    if reason == "scale_outlier":
        for index in range(51, 90):
            stamp = pose.stamp + index * .05
            estimator.observe_telemetry(None, (0., .5, stamp), stamp)
            sample = rpf.Pose(
                pose.x + visual_speed * (stamp - pose.stamp),
                0., 0., 0., stamp=stamp,
            )
            estimator.observe_pose(sample, stamp, 0.)
    info = estimator.scale_diagnostics(stamp)
    assert info["scale_segments"] == count + (reason == "scale_outlier")
    assert info["candidate_m_per_unit"] == pytest.approx(15.)
    assert info["rejection_counts"][reason] >= 1
    assert not info["reliable"]
    assert estimator.metric_distance(1., stamp) is None
    assert not estimator.predict(sample, stamp + .1, 0., .05)[1]["applied"]


def test_global_scale_freshness_and_bad_speed_refusals_are_explicit():
    estimator, pose = _trained()
    for sample in [(0., float("nan"), pose.stamp + .1),
                   (0., .5, pose.stamp - 1.), (0., .5, pose.stamp + 1.)]:
        estimator.observe_telemetry(None, sample, pose.stamp)
    info = estimator.scale_diagnostics(pose.stamp)
    assert info["rejection_counts"]["invalid_velocity_sample"] == 1
    assert info["rejection_counts"]["velocity_stale"] == 1
    assert info["rejection_counts"]["velocity_future"] == 1
    stale = estimator.scale_diagnostics(pose.stamp + 30.1)
    assert not stale["reliable"] and stale["reason"] == "scale_stale"
    assert stale["candidate_m_per_unit"] == pytest.approx(15.)
    assert estimator.metric_xyz([1., 2., 3.], pose.stamp + 30.1) is None
    assert estimator.metric_distance(float("nan"), pose.stamp) is None
    assert estimator.metric_xyz([1., float("inf"), 3.], pose.stamp) is None


def test_unstable_cumulative_scale_never_enables_metric_conversion():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    # Independent, direction-consistent intervals can still expose actual
    # monocular scale drift. Accumulation alone is not proof of uniform scale.
    estimator.scale_samples = [(10., 1.5)] * 4 + [(15., 1.5)] * 4
    estimator.scale_distance = 12.
    estimator._update_scale(20.)
    info = estimator.scale_diagnostics(20.)
    assert info["candidate_m_per_unit"] == pytest.approx(12.5)
    assert info["relative_iqr"] > .20
    assert info["relative_drift"] > .20
    assert not info["reliable"]
    assert estimator.metric_distance(1., 20.) is None


def test_cached_velocity_integral_handles_partial_samples_and_refuses_gaps():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for stamp, speed in [(1., 0.), (1.2, 2.), (1.4, 4.)]:
        estimator.observe_telemetry(None, (0., speed, stamp), stamp)
    np.testing.assert_allclose(estimator.displacement(1.1, 1.3), [0., .4])
    cache = estimator._velocity_cache
    np.testing.assert_allclose(estimator.displacement(1.3, 1.6, extrapolate=True), [0., 1.15])
    assert estimator._velocity_cache is cache
    estimator.observe_telemetry(None, (0., 4., 2.1), 2.1)
    assert estimator._velocity_cache is None
    assert estimator.displacement(1.3, 2.1) is None


def test_converging_global_candidate_accumulates_trajectory_without_forgetting():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    x = 0.
    early = None
    for index in range(1201):
        stamp = index * .05
        ratio = (13., 14., 16., 17.)[min(3, int(stamp / 3.))] if stamp < 12. else 15.
        if index:
            x += .5 * .05 / ratio
        estimator.observe_telemetry(None, (0., .5, stamp), stamp)
        estimator.observe_pose(rpf.Pose(x, 0., 0., 0., stamp=stamp), stamp, 0.)
        if index == 240:
            early = estimator.scale_diagnostics(stamp)
    final = estimator.scale_diagnostics(stamp)
    assert final["scale_segments"] > early["scale_segments"] + 12
    assert final["cumulative_distance_m"] > early["cumulative_distance_m"]
    assert final["relative_iqr"] < early["relative_iqr"]
    assert final["candidate_m_per_unit"] == pytest.approx(15., abs=.01)
    assert final["reliable"]


def test_eight_independent_intervals_still_require_ten_metres():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(341):
        stamp = index * .05
        estimator.observe_telemetry(None, (0., .5, stamp), stamp)
        estimator.observe_pose(rpf.Pose(stamp / 30., 0., 0., 0., stamp=stamp), stamp, 0.)
    info = estimator.scale_diagnostics(stamp)
    assert info["scale_segments"] >= 8
    assert info["cumulative_distance_m"] < 10.
    assert info["reason"] == "insufficient_displacement"
    assert not info["reliable"]
    for index in range(341, 451):
        stamp = index * .05
        estimator.observe_telemetry(None, (0., .5, stamp), stamp)
        estimator.observe_pose(rpf.Pose(stamp / 30., 0., 0., 0., stamp=stamp), stamp, 0.)
    assert estimator.scale_diagnostics(stamp)["reliable"]


def test_session_estimator_transaction_lock_is_reentrant_and_survives_resets():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    lock = estimator.lock
    with lock:
        with estimator.lock:
            estimator.reset_observation()
            assert estimator.lock is lock
            estimator.reset_position()
            assert estimator.lock is lock


@pytest.mark.parametrize("interruption", ["weak", "missing", "bridge", "reseed"])
def test_strong_endpoints_cannot_teach_scale_across_untrusted_visual_holes(interruption):
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(1201):
        stamp = index * .05
        estimator.observe_telemetry(None, (0., .5, stamp), stamp)
        strong = index % 40 == 0
        pose = rpf.Pose(
            stamp / 30., 0., 0., 0., stamp=stamp,
            map_confirmed=strong or interruption != "weak",
            position_observed=strong or interruption != "bridge",
            reseed_confirming=not strong and interruption == "reseed",
        )
        estimator.observe_pose(None if not strong and interruption == "missing" else pose, stamp, 0.)
    info = estimator.scale_diagnostics(stamp)
    assert info["scale_segments"] == 0
    assert info["candidate_m_per_unit"] is None
    assert not info["reliable"]
    assert info["rejection_counts"]["visual_interval_untrusted"] > 0


def test_delayed_velocity_tail_uses_the_endpoint_heading_not_a_later_anchor():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(5):
        stamp = index * .2
        estimator.observe_telemetry(None, (0., 1., stamp), stamp)
        estimator.observe_pose(rpf.Pose(stamp / 15., 0., 0., 0., stamp=stamp), stamp, 0.)
    pose = rpf.Pose(1.1 / 15., 0., 0., 0., stamp=1.1)
    estimator.observe_pose(pose, 1.1, 0.)
    assert estimator.scale_candidate is None
    estimator.observe_telemetry(None, (0., 1., 1.2), 1.2)
    estimator.observe_pose(pose, 1.2, math.pi / 2.)
    info = estimator.scale_diagnostics(1.2)
    assert info["candidate_m_per_unit"] == pytest.approx(15.)
    assert info["scale_segments"] == 1
    assert info["last_evidence_stamp"] == 1.1
    assert pose.stamp == 1.1
    assert "direction_inconsistent" not in info["rejection_counts"]


def test_heading_frame_mismatch_inside_interval_cannot_hide_at_matching_endpoints():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(25):
        stamp = index * .05
        estimator.observe_telemetry(None, (0., 1., stamp), stamp)
        offset = math.pi / 2. if index == 8 else 0.
        estimator.observe_pose(rpf.Pose(stamp / 15., 0., 0., 0., stamp=stamp), stamp, offset)
    info = estimator.scale_diagnostics(stamp)
    assert info["scale_segments"] == 0
    assert info["rejection_counts"]["heading_interval_inconsistent"] == 1
    assert not info["reliable"]


def test_changed_scale_ratios_remain_in_cumulative_dispersion_instead_of_self_filtering():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(1281):
        stamp = index * .05
        x = min(stamp, 24.) / 30. + max(0., stamp - 24.) / 60.
        estimator.observe_telemetry(None, (0., .5, stamp), stamp)
        estimator.observe_pose(rpf.Pose(x, 0., 0., 0., stamp=stamp), stamp, 0.)
    info = estimator.scale_diagnostics(stamp)
    assert info["scale_segments"] == 32
    assert info["cumulative_distance_m"] == pytest.approx(32.)
    assert info["candidate_m_per_unit"] == pytest.approx(30.)
    assert info["relative_iqr"] > .2 and info["relative_drift"] > .2
    assert info["rejection_counts"]["scale_outlier"] > 0
    assert not info["reliable"]
    assert estimator.metric_distance(1., stamp) is None


def test_old_pending_pose_cannot_refresh_scale_when_telemetry_finally_arrives():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for index in range(13):
        stamp = 1. + index * .05
        estimator.observe_pose(rpf.Pose((stamp - 1.) / 15., 0., 0., 0., stamp=stamp), stamp, 0.)
    estimator.observe_telemetry(None, (0., 1., 1.6), 1.6)
    estimator.observe_pose(rpf.Pose(.6 / 15., 0., 0., 0., stamp=1.6), 1.6, 0.)
    info = estimator.scale_diagnostics(1.6)
    assert info["last_evidence_stamp"] is None
    assert info["scale_segments"] == 0
    assert info["rejection_counts"]["pending_pose_stale"] == 1
    assert not info["reliable"]


@pytest.mark.parametrize("source", ["attitude", "velocity"])
def test_invalid_new_source_event_cannot_be_interpolated_through_between_valid_receipts(source):
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for stamp in (1., 1.1):
        estimator.observe_telemetry((0., stamp), (0., 1., stamp), stamp)
    estimator.observe_telemetry(
        (float("nan"), 1.15) if source == "attitude" else None,
        (0., float("nan"), 1.15) if source == "velocity" else None,
        1.15,
    )
    for stamp in (1.2, 1.3):
        estimator.observe_telemetry((0., stamp), (0., 1., stamp), stamp)

    # A sub-0.5s gap is NOT sufficient coverage when a known invalid source
    # event lies inside the interpolation bracket, even for a partial query.
    if source == "attitude":
        assert estimator.yaw_at(1.12) is None
        assert estimator.yaw_at(1.05) == pytest.approx(0.)
        assert estimator.yaw_at(1.2) == pytest.approx(0.)
        assert estimator.yaw_at(1.25) == pytest.approx(0.)
    else:
        assert estimator.displacement(1.11, 1.13) is None
        assert estimator.displacement(1.05, 1.25) is None
        np.testing.assert_allclose(estimator.displacement(1.02, 1.08), [0., .06])
        np.testing.assert_allclose(estimator.displacement(1.2, 1.3), [0., .1])


def test_prediction_extrapolation_cannot_cross_a_known_invalid_velocity_receipt():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    estimator.observe_telemetry(None, (0., 1., 1.), 1.)
    estimator.observe_telemetry(None, (0., float("nan"), 1.2), 1.2)
    np.testing.assert_allclose(estimator.displacement(1., 1.1, extrapolate=True), [0., .1])
    assert estimator.displacement(1., 1.2, extrapolate=True) is None


def test_cached_valid_duplicate_future_or_late_receipts_do_not_create_source_barriers():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    for stamp in (1., 1.1, 1.2):
        estimator.observe_telemetry((0., stamp), (0., 1., stamp), stamp)
    for stamp in (1.2, .1, 2.):
        estimator.observe_telemetry((0., stamp), (0., 1., stamp), 1.2)
    assert estimator.yaw_at(1.15) == pytest.approx(0.)
    np.testing.assert_allclose(estimator.displacement(1.05, 1.15), [0., .1])
    assert all(not barriers for barriers in estimator._invalid_sources.values())


def test_repeated_unstampable_invalid_payload_fences_once_not_on_every_poll():
    estimator = ControlStateEstimator(rpf.LEGACY_MAP_FRAME)
    estimator.observe_telemetry(None, (0., 1., 1.), 1.)
    for now in (1.1, 1.2, 1.3):
        estimator.observe_telemetry(None, ("bad",), now)
    assert tuple(estimator._invalid_sources["velocity"]) == (1.1,)
    estimator.observe_telemetry(None, (0., 1., 1.4), 1.4)
    estimator.observe_telemetry(None, (0., 1., 1.5), 1.5)
    np.testing.assert_allclose(estimator.displacement(1.4, 1.5), [0., .1])
    # A new valid source receipt ends the cached-invalid episode. The same
    # malformed payload arriving after recovery is new invalid evidence.
    estimator.observe_telemetry(None, ("bad",), 1.6)
    assert tuple(estimator._invalid_sources["velocity"]) == (1.1, 1.6)
    assert estimator.displacement(1.5, 1.6, extrapolate=True) is None
