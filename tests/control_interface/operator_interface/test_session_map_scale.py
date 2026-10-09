"""Passive trajectory scale and AUTO consumption, with no aircraft SDK."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import path_follow_flight as pff
import real_path_follow_controller as rpf
from localization_metrics import SessionMapScale
from operator_hud_text import format_map_scale


def _flight_sample(learner, now, *, weak=False):
    backend = SimpleNamespace(
        control_telemetry_history=lambda _cursor: (
            [(math.pi / 2, now)],
            [(0.6, 0.8, now)],
        )
    )
    pose = rpf.Pose(0.08 * now, 0.0, 0.06 * now, 0.0, stamp=now, map_confirmed=not weak)
    return learner.observe(pose, backend, now)


def test_manual_trajectory_restores_one_scale_for_the_whole_field_and_auto():
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    for tick in range(321):
        now = 1.0 + 0.05 * tick
        result = _flight_sample(learner, now)
    assert result["reliable"]
    assert result["scale_m_per_unit"] == pytest.approx(10.0)
    assert result["scale_segments"] >= 8
    np.testing.assert_allclose(learner.state.metric_xyz([3.0, -2.0, 4.0], now), [30.0, -20.0, 40.0])
    assert learner.state.metric_distance(5.0, now) == pytest.approx(50.0)

    cfg = rpf.ControlConfig(inspect_waypoints=())
    ctrl = rpf.RouteAutoController([np.zeros(3), np.array([2.0, 0.0, 0.0])], config=cfg)
    runner = pff._FlightLoopRunner(
        pff.LoopHooks(
            get_pose=lambda: None,
            olympe_yaw=lambda: None,
            send_pcmd=lambda *_: None,
            control_state=learner.state,
        ),
        ctrl,
        ctrl.wp,
        yaw_sign=1,
        verbose=False,
        enforce_weak_pose_gate=True,
    )
    runner.heading.offset = 0.0
    pose = rpf.Pose(1.0, 0.0, 0.0, 0.0, stamp=now)
    command = rpf.Command("FOLLOW", np.zeros(3), 0.0, np.array([2.0, 0.0, 0.0]), 0.0, 0.5)
    record = {}
    runner._predict_control_pose(command, pose, now, record)
    assert record["metric_target_distance_m"] == pytest.approx(10.0)
    assert record["metric_pose_m"] == pytest.approx([10.0, 0.0, 0.0])
    segments = result["scale_segments"]
    runner._reset_pcmd_controller()
    for tick in range(1, 81):
        result = _flight_sample(learner, now + 0.05 * tick)
    assert result["scale_segments"] > segments
    assert result["scale_m_per_unit"] == pytest.approx(10.0)


def test_events_received_after_observation_time_teach_scale_on_next_tick():
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    attitudes, velocities = [], []

    def history(cursor):
        attitude_stamp, velocity_stamp = cursor or (None, None)
        return (
            [row for row in attitudes if attitude_stamp is None or row[-1] > attitude_stamp],
            [row for row in velocities if velocity_stamp is None or row[-1] > velocity_stamp],
        )

    backend = SimpleNamespace(control_telemetry_history=history)
    for tick in range(421):
        now = 1.0 + tick * 0.05
        attitudes.append((math.pi / 2, now + 0.025))
        velocities.append((0.6, 0.8, now + 0.025))
        pose = rpf.Pose(0.08 * now, 0.0, 0.06 * now, 0.0, stamp=now)
        result = learner.observe(pose, backend, now)

    assert result["reliable"]
    assert result["scale_m_per_unit"] == pytest.approx(10.0)
    np.testing.assert_allclose(learner.state.metric_xyz([3.0, -2.0, 4.0], now), [30, -20, 40])


def test_weak_only_trajectory_cannot_restore_metric_field():
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    for tick in range(321):
        result = _flight_sample(learner, 1.0 + 0.05 * tick, weak=True)
    assert result["scale_segments"] == 0
    assert not result["reliable"]
    assert learner.state.metric_xyz([1.0, 2.0, 3.0], 17.0) is None


def test_stale_scale_does_not_remain_a_metric_readout():
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    for tick in range(321):
        _flight_sample(learner, 1.0 + 0.05 * tick)
    snapshot = learner.state.scale_diagnostics(60.0)
    assert not snapshot["reliable"]
    assert snapshot["scale_m_per_unit"] is None
    assert snapshot["candidate_m_per_unit"] == pytest.approx(10.0)
    assert learner.state.metric_distance(5.0, 60.0) is None
    assert "候選" in format_map_scale(snapshot)
    assert "證據過期" in format_map_scale(snapshot)


def test_auto_rejects_another_fields_basis_instead_of_reusing_its_scale():
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    other = rpf.MapFrame(east=(0.0, 0.0, 1.0), north=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    ctrl = rpf.RouteAutoController(
        [np.zeros(3), np.array([2.0, 0.0, 0.0])],
        config=rpf.ControlConfig(inspect_waypoints=(), map_frame=other),
    )
    with pytest.raises(ValueError, match="different map frame"):
        pff._FlightLoopRunner(
            pff.LoopHooks(
                get_pose=lambda: None,
                olympe_yaw=lambda: None,
                send_pcmd=lambda *_: None,
                control_state=learner.state,
            ),
            ctrl,
            ctrl.wp,
            yaw_sign=1,
            verbose=False,
            enforce_weak_pose_gate=True,
        )


@pytest.mark.parametrize("scenario", ["delay_turn", "weak_dropout", "telemetry_dropout", "reseed"])
def test_capture_aligned_delayed_turn_dropout_and_reseed_keep_known_scale_unbiased(scenario):
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    attitudes, velocities = [], []

    def history(cursor):
        attitude_stamp, velocity_stamp = cursor or (None, None)
        return (
            [row for row in attitudes if attitude_stamp is None or row[-1] > attitude_stamp],
            [row for row in velocities if velocity_stamp is None or row[-1] > velocity_stamp],
        )

    backend = SimpleNamespace(control_telemetry_history=history)
    times = 1. + np.arange(1201) * .05
    heading = .35 * np.sin((times - 1.) * .15)
    ned = np.column_stack((.6 * np.sin(heading), .6 * np.cos(heading)))
    metric = np.zeros((len(times), 2))
    metric[1:] = np.cumsum(.5 * (ned[1:] + ned[:-1]) * .05, axis=0)
    for index, now in enumerate(times):
        if index >= 4:
            attitudes.append((math.pi / 2. - heading[index - 4], times[index - 4]))
        if index >= 6 and not (scenario == "telemetry_dropout" and 400 <= index <= 425):
            velocities.append((*ned[index - 6], times[index - 6]))
        if index < 2:
            continue
        capture = index - 2
        weak = scenario == "weak_dropout" and 400 <= index <= 425
        reseed = scenario == "reseed" and 400 <= index <= 405
        correction = .2 if scenario == "reseed" and index >= 400 else 0.
        pose = rpf.Pose(
            metric[capture, 1] / 15. + correction, 0., metric[capture, 0] / 15.,
            heading[capture], stamp=times[capture],
            map_confirmed=not weak, reseed_confirming=reseed,
        )
        result = learner.observe(pose, backend, now)
        assert pose.stamp == times[capture]
        if result["last_evidence_stamp"] is not None:
            assert 0. <= now - result["last_evidence_stamp"]
            assert result["last_evidence_stamp"] <= velocities[-1][-1]
        if reseed or weak:
            # Untrusted frames cannot refresh an already learned conversion.
            assert result["last_evidence_stamp"] < times[398]
    assert result["reliable"]
    assert result["scale_m_per_unit"] == pytest.approx(15.)
    assert result["scale_segments"] >= 8
    assert result["cumulative_distance_m"] >= 10.
    assert result["relative_iqr"] < 1e-10
    assert result["relative_drift"] < 1e-10
    if scenario == "weak_dropout":
        assert result["rejection_counts"]["visual_interval_untrusted"] > 0
    elif scenario == "telemetry_dropout":
        assert result["rejection_counts"]["pending_pose_stale"] > 0
    assert not learner.state.scale_diagnostics(now + 30.1)["reliable"]


def test_heading_without_a_capture_time_attitude_pair_never_teaches_metric_scale():
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    backend = SimpleNamespace(control_telemetry_history=lambda _cursor: ([], []))
    for index in range(1201):
        now = 1. + index * .05
        backend.control_telemetry_history = lambda _cursor, t=now: ([], [(0., .5, t)])
        pose = rpf.Pose(now / 30., 0., 0., 0., stamp=now)
        result = learner.observe(pose, backend, now)
    assert result["scale_segments"] == 0
    assert result["candidate_m_per_unit"] is None
    assert not result["reliable"]
    assert learner.state.last_observed[0] == pose.stamp


@pytest.mark.parametrize("source", ["attitude", "velocity"])
def test_invalid_source_intervals_cannot_certify_scale_and_valid_source_recovers(source):
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    attitudes, velocities = [], []

    def history(cursor):
        a, v = cursor or (None, None)
        return (
            [row for row in attitudes if a is None or row[-1] > a],
            [row for row in velocities if v is None or row[-1] > v],
        )

    backend = SimpleNamespace(control_telemetry_history=history)
    for index in range(601):
        now = 1. + index * .05
        invalid = index % 20 == 10
        attitudes.append((float("nan") if invalid and source == "attitude" else math.pi / 2., now))
        velocities.append((0., float("nan") if invalid and source == "velocity" else 1., now))
        result = learner.observe(rpf.Pose((now - 1.) / 15., 0., 0., 0., stamp=now), backend, now)
    # Previously the invalid samples were dropped, leaving <=0.5s numeric
    # brackets that falsely certified 30 apparently coherent segments.
    assert result["scale_segments"] == 0
    assert result["candidate_m_per_unit"] is None
    assert not result["reliable"]
    assert result["rejection_counts"][f"invalid_{source}_sample"] == 30
    if source == "attitude":
        assert result["rejection_counts"]["heading_capture_uncovered"] == 30
        assert not learner._pending_heading
    for index in range(601, 1001):
        now = 1. + index * .05
        attitudes.append((math.pi / 2., now))
        velocities.append((0., 1., now))
        result = learner.observe(rpf.Pose((now - 1.) / 15., 0., 0., 0., stamp=now), backend, now)
    assert result["reliable"]
    assert result["scale_m_per_unit"] == pytest.approx(15.)


def test_phase_shifted_timestamp_grids_during_large_turns_preserve_known_scale():
    learner = SessionMapScale(rpf.LEGACY_MAP_FRAME)
    attitudes, velocities = [], []

    def history(cursor):
        a, v = cursor or (None, None)
        return (
            [row for row in attitudes if a is None or row[-1] > a],
            [row for row in velocities if v is None or row[-1] > v],
        )

    backend = SimpleNamespace(control_telemetry_history=history)
    scale, offset = 12.7, 1.2
    velocity_times = 1.017 + np.arange(1301) * .05
    attitude_times = 1.029 + np.arange(1301) * .05

    def heading(t):
        return .8 * np.sin(.23 * (t - 1.)) + .3 * np.sin(.73 * (t - 1.))

    ned = np.column_stack((.8 * np.sin(heading(velocity_times)), .8 * np.cos(heading(velocity_times))))
    metric_integral = np.zeros_like(ned)
    metric_integral[1:] = np.cumsum(.5 * (ned[1:] + ned[:-1]) * .05, axis=0)
    attitude_index = velocity_index = 0
    for now in 1.4 + np.arange(1201) * .05:
        while attitude_times[attitude_index] + .18 <= now:
            stamp = float(attitude_times[attitude_index])
            attitudes.append((float(math.pi / 2. - heading(stamp)), stamp))
            attitude_index += 1
        while velocity_times[velocity_index] + .27 <= now:
            stamp = float(velocity_times[velocity_index])
            velocities.append((*ned[velocity_index], stamp))
            velocity_index += 1
        capture = float(now - .10)
        index = int(np.searchsorted(velocity_times, capture, side="right")) - 1
        dt = capture - velocity_times[index]
        metric = (metric_integral[index] + ned[index] * dt
                  + .5 * (ned[index + 1] - ned[index]) * dt * dt / .05)
        horizontal = learner.state._map_horizontal(metric, offset)
        pose = rpf.Pose(horizontal[0] / scale, 0., horizontal[1] / scale,
                        float(heading(capture) + offset), stamp=capture)
        result = learner.observe(pose, backend, float(now))
        if result["last_evidence_stamp"] is not None:
            assert result["last_evidence_stamp"] <= velocities[-1][-1]
        assert pose.stamp == capture
    assert np.ptp(heading(velocity_times)) > math.radians(120.)
    assert result["reliable"]
    assert result["scale_segments"] >= 8
    assert result["cumulative_distance_m"] >= 10.
    assert result["scale_m_per_unit"] == pytest.approx(scale)
    assert result["relative_iqr"] < 1e-10
    assert result["relative_drift"] < 1e-10
    assert not learner.state.scale_diagnostics(float(now) + 30.1)["reliable"]
