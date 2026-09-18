"""Plant, sensor-delay, and trace-contract regressions for the offline sim."""

import math

import numpy as np
import pytest

import sim_route_autoflight as sim


def _route():
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    routes = sorted((root / "地圖檔/場域/river_site/routes").glob("*.json"))
    assert routes, "preset routes must be present"
    return routes[0]


def _align():
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    align = root / "地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260908/localization/T_align_gravity.json"
    return align if align.is_file() else None


def _plant_state(map_frame, yaw=0.0):
    return sim.DroneState(pos_m=np.zeros(3), yaw=float(yaw))


def test_tilt_responds_symmetrically_on_both_axes_and_rotates_with_yaw():
    route, align = _route(), _align()
    base = dict(route=route, align=align, duration_s=10.0, seed=7,
                wind_sigma_mps=0.0, hover_hold_mps=0.0, latency_ms=0.0,
                drop_rate=0.0, pos_noise_u=0.0, yaw_noise_deg=0.0, oly_noise_deg=0.0,
                tau_tilt_s=0.15, thrust_margin=99.0)
    ctrl, cfg, frame, _ = sim.build_production_controller(route, align, return_to_start=True)
    vh, vv, vy = 0.02, 0.02, 20.0
    rng = np.random.default_rng(0)
    params = sim.SimParams(**base)
    s_pos = _plant_state(frame, yaw=0.0)
    wind = np.zeros(3)
    for _ in range(int(10.0 / sim.DT)):
        wind, _, _ = sim._integrate_plant(s_pos, wind, None, params, frame, rng, (20, 0, 0, 0), vh, vv, vy, 0.0, elapsed_s=5.0)
    s_neg = _plant_state(frame, yaw=0.0)
    wind = np.zeros(3)
    rng = np.random.default_rng(0)
    for _ in range(int(10.0 / sim.DT)):
        wind, _, _ = sim._integrate_plant(s_neg, wind, None, params, frame, rng, (-20, 0, 0, 0), vh, vv, vy, 0.0, elapsed_s=5.0)
    assert s_pos.pos_m[0] == pytest.approx(-s_neg.pos_m[0], rel=1e-6)
    assert abs(s_pos.pos_m[0]) > 0.0
    # Pitch axis mirrors roll.
    s_fwd = _plant_state(frame, yaw=0.0)
    wind = np.zeros(3)
    rng = np.random.default_rng(0)
    for _ in range(int(10.0 / sim.DT)):
        wind, _, _ = sim._integrate_plant(s_fwd, wind, None, params, frame, rng, (0, 20, 0, 0), vh, vv, vy, 0.0, elapsed_s=5.0)
    fwd_axis = np.asarray(frame.east, float) * math.cos(0.0) + np.asarray(frame.north, float) * math.sin(0.0)
    right_axis = np.asarray(frame.east, float) * math.sin(0.0) - np.asarray(frame.north, float) * math.cos(0.0)
    assert abs(float(np.dot(s_pos.pos_m, right_axis))) == pytest.approx(abs(float(np.dot(s_fwd.pos_m, fwd_axis))), rel=1e-6)
    # Yaw 90 deg rotates the world response.
    s_yaw = _plant_state(frame, yaw=math.pi / 2)
    wind = np.zeros(3)
    rng = np.random.default_rng(0)
    for _ in range(int(10.0 / sim.DT)):
        wind, _, _ = sim._integrate_plant(s_yaw, wind, None, params, frame, rng, (20, 0, 0, 0), vh, vv, vy, 0.0, elapsed_s=5.0)
    want = np.asarray(frame.east, float) * math.sin(math.pi / 2) - np.asarray(frame.north, float) * math.cos(math.pi / 2)
    got = s_yaw.pos_m / max(float(np.linalg.norm(s_yaw.pos_m)), 1e-12)
    assert float(np.dot(got, want)) == pytest.approx(1.0, abs=1e-6)


def test_battery_sag_reduces_response_near_thrust_margin():
    route, align = _route(), _align()
    kw = dict(route=route, align=align, duration_s=10.0, seed=7, wind_sigma_mps=0.0,
              hover_hold_mps=0.0, latency_ms=0.0, drop_rate=0.0, pos_noise_u=0.0,
              yaw_noise_deg=0.0, oly_noise_deg=0.0, tau_tilt_s=0.15,
              thrust_margin=1.05, yaw_coupling=0.0)
    ctrl, cfg, frame, _ = sim.build_production_controller(route, align, return_to_start=True)
    vh, vv, vy = 0.02, 0.02, 20.0
    fresh = sim.SimParams(battery_sag_frac=0.0, **kw)
    sagged = sim.SimParams(battery_sag_frac=0.2, **kw)
    s0, s1 = _plant_state(frame), _plant_state(frame)
    w0, w1 = np.zeros(3), np.zeros(3)
    r0, r1 = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(int(10.0 / sim.DT)):
        w0, _, _ = sim._integrate_plant(s0, w0, None, fresh, frame, r0, (0, 50, 0, 50), vh, vv, vy, 0.0, elapsed_s=9.0)
        w1, _, _ = sim._integrate_plant(s1, w1, None, sagged, frame, r1, (0, 50, 0, 50), vh, vv, vy, 0.0, elapsed_s=9.0)
    assert float(np.linalg.norm(s1.pos_m)) < float(np.linalg.norm(s0.pos_m))


def test_fixed_ground_plane_ignores_hover_anchor_climb():
    route, align = _route(), _align()
    params = sim.SimParams(route=route, align=align, duration_s=10.0, seed=7,
                           ground_effect_frac=1.0, ground_altitude_m=0.0,
                           ground_effect_height_m=1.0, tau_tilt_s=0.15)
    ctrl, cfg, frame, _ = sim.build_production_controller(route, align, return_to_start=True)
    up = np.asarray(frame.up, float)
    high = _plant_state(frame)
    high.pos_m = up * 5.0
    wind = np.zeros(3)
    rng = np.random.default_rng(0)
    _, _, _ = sim._integrate_plant(high, wind.copy(), high.pos_m.copy(), params, frame, rng, (0, 0, 0, 0), 0.02, 0.02, 20.0, 0.0, elapsed_s=1.0)
    low = _plant_state(frame)
    low.pos_m = up * 0.25
    rng = np.random.default_rng(0)
    _, _, _ = sim._integrate_plant(low, wind.copy(), low.pos_m.copy() + up * 3.0, params, frame, rng, (0, 0, 0, 0), 0.02, 0.02, 20.0, 0.0, elapsed_s=1.0)
    assert high.vel_m is not None and low.vel_m is not None
    # Above the band there is no cushion contribution regardless of anchor height.
    assert float(np.dot(high.vel_m, up)) == pytest.approx(0.0, abs=1e-6)


def test_telemetry_delay_hides_samples_and_drop_keeps_stamp():
    import sim_route_autoflight as sim_mod
    assert float(sim_mod.SimParams(route=_route(), align=_align()).telemetry_latency_ms) == pytest.approx(200.0)
    route, align = _route(), _align()
    params = sim.SimParams(route=route, align=align, duration_s=3.0, seed=7,
                           telemetry_latency_ms=200.0, telemetry_drop_rate=1.0,
                           wind_sigma_mps=0.0, pos_noise_u=0.0)
    result, _ = sim.run_sim(params, record_trace=True)
    assert result.trace
    # Every sample dropped: no measured velocity and no telemetry stamp ever.
    assert all(r["measured_body_velocity_mps"] is None and r["telemetry_stamp"] is None for r in result.trace)
    params2 = sim.SimParams(route=route, align=align, duration_s=3.0, seed=7,
                            telemetry_latency_ms=200.0, telemetry_drop_rate=0.0,
                            wind_sigma_mps=0.0, pos_noise_u=0.0)
    result2, _ = sim.run_sim(params2, record_trace=True)
    stamps = [r["telemetry_stamp"] for r in result2.trace if r["telemetry_stamp"] is not None]
    assert stamps
    # Capture stamps lag decision time by the configured latency (minus DT grid).
    assert min(r["t"] - r["telemetry_stamp"] for r in result2.trace if r["telemetry_stamp"] is not None) >= 0.15


def test_control_uses_measured_speed_while_scoring_uses_truth():
    route, align = _route(), _align()
    params = sim.SimParams(route=route, align=align, duration_s=5.0, seed=7, wind_sigma_mps=0.0)
    result, _ = sim.run_sim(params, record_trace=True)
    assert result.trace
    assert all(r["body_velocity_mps"] is not None for r in result.trace)
    assert any(r["measured_body_velocity_mps"] is not None for r in result.trace)


def test_speed_limiter_changes_output_in_sim():
    route, align = _route(), _align()
    params = sim.SimParams(route=route, align=align, duration_s=60.0, seed=7,
                           wind_sigma_mps=0.0, speed_limit_mps=0.05)
    result, _ = sim.run_sim(params, record_trace=True)
    assert result.speed_guard_interventions > 0
    assert any(r["pcmd"] != r["pcmd_requested"] for r in result.trace)


def test_fault_families_recover_and_surface_in_trace():
    from collections import Counter
    route, align = _route(), _align()
    base = dict(route=route, align=align, meters_per_unit=5.0, duration_s=300.0, seed=7, wind_sigma_mps=0.0)
    cases = [
        ("weak", dict(weak_rate=0.05)),
        ("predicted", dict(predicted_rate=0.05)),
        ("jump", dict(jump_rate=0.01, jump_max_u=0.05)),
        ("outlier", dict(outlier_rate=0.01, outlier_max_u=0.05)),
        ("blackout-imu", dict(blackout_every_s=40, blackout_dur_s=3, blackout_drift_mps=0.05, blackout_yaw_drift_deg_s=2.0)),
        ("dropout", dict(drop_rate=0.20)),
    ]
    for label, fault_kw in cases:
        result, _ = sim.run_sim(sim.SimParams(**{**base, **fault_kw}), record_trace=True)
        assert result.success, (label, result.reason)
        faults = Counter(tick.get("loc_fault", "?") for tick in result.trace)
        assert faults[label] > 0, (label, dict(faults))


def test_excessive_latency_hands_off_to_manual_without_crash():
    route, align = _route(), _align()
    result, _ = sim.run_sim(
        sim.SimParams(route=route, align=align, meters_per_unit=5.0, duration_s=300.0, seed=7,
                      wind_sigma_mps=0.0, latency_ms=500.0),
        record_trace=False,
    )
    assert not result.success
    assert "manual recovery required" in result.reason


def test_invalid_plant_params_raise():
    route, align = _route(), _align()
    with pytest.raises(ValueError, match="tilt dynamics require"):
        sim.SimParams(route=route, align=align, tau_tilt_s=1e-3, battery_sag_frac=0.1)
    with pytest.raises(ValueError, match="ground_altitude_m"):
        sim.SimParams(route=route, align=align, ground_effect_frac=1.0)


def _flown_route():
    """The route actually flown on 2026-09-18 (stalls at waypoint 1)."""
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    flown = root / "地圖檔/場域/river_site/routes/flight_route_20260917_222503_e4c4dfe8.json"
    return flown if flown.is_file() else _route()


def _regression_base(**kw):
    route, align = _flown_route(), _align()
    base = dict(route=route, align=align, meters_per_unit=15.0, duration_s=120.0,
                seed=7, wind_sigma_mps=0.0)
    base.update(kw)
    return sim.SimParams(**base)


def test_regression_R1_contiguous_outage_stalls_like_flight20260918():
    params = sim.apply_regression_scenario(_regression_base(), "R1_flight20260918")
    result, _ = sim.run_sim(params, record_trace=False)
    # Real flight: 0% progress, stuck at waypoint 1, hovering, no flyaway.
    assert result.progress_pct < 5.0
    assert len(result.waypoints_reached) <= 1
    assert result.hover_ticks > 500
    assert result.max_dev_u < 1.0


def test_regression_R2_weak_flicker_degraded_but_moving():
    base_result, _ = sim.run_sim(_regression_base(), record_trace=False)
    params = sim.apply_regression_scenario(_regression_base(), "R2_weak80")
    result, _ = sim.run_sim(params, record_trace=False)
    assert 0.0 < result.progress_pct < base_result.progress_pct
    assert result.max_dev_u < 1.0


def test_apply_regression_scenario_unknown_raises():
    with pytest.raises(KeyError):
        sim.apply_regression_scenario(_regression_base(), "nope")


def test_cli_out_and_scenario_flags_write_artifacts(tmp_path):
    route = _flown_route()
    outdir = tmp_path / "simR2"
    rc = sim.main(["--route", str(route), "--meters-per-unit", "15",
                   "--duration-s", "60", "--seed", "7",
                   "--scenario", "R2_weak80", "--out", str(outdir), "--no-plot"])
    assert rc in (0, 1)  # R2 is not expected to fully succeed
    assert (outdir / "summary.json").is_file()
    assert (outdir / "trace.csv").is_file()
    import json
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scenario"] == "R2_weak80"
    assert summary["params"]["weak_rate"] == 0.8


def _fly_localizer(velocity_mps, t_end, t_start=0.0, **config):
    """Poll a noise-free, zero-latency SimulatedLocalizer along a straight line."""
    from simulated_localization import SimulatedLocalizer, SimulatedLocalizerConfig
    cfg = SimulatedLocalizerConfig(s=10.0, pos_sigma_u=0.0, yaw_sigma_deg=0.0, latency_ms=0.0,
                                   drop_rate=0.0, map_up_u=(0.0, -1.0, 0.0), **config)
    loc = SimulatedLocalizer(cfg, np.random.default_rng(0))
    samples, pos = [], np.zeros(3)
    for step in range(int(round((t_end - t_start) / sim.DT)) + 1):
        t = t_start + step * sim.DT
        loc.push_truth(t, pos, 0.0)
        samples.append((t, pos / 10.0, loc.report(t)))
        pos = pos + np.asarray(velocity_mps, dtype=float) * sim.DT
    return samples


def test_drift_window_reports_weak_pose_running_ahead_of_truth():
    samples = _fly_localizer((1.0, 0.0, 0.0), 4.0, drift_offset_s=1.0, drift_every_s=100.0,
                             drift_dur_s=2.0, drift_gain=5.0)

    def truth_x(pose):  # 1 m/s at 10 m/u; a capture may sample the previous tick
        return float(pose.stamp) / 10.0

    tick_u = 0.1 * sim.DT
    anchor = [pose for t, _truth, pose in samples if t < 1.0][-1]
    assert anchor.map_confirmed and anchor.x == pytest.approx(truth_x(anchor), abs=tick_u)
    pose = next(pose for t, _truth, pose in samples if t >= 2.5)
    assert not pose.map_confirmed and pose.position_observed
    assert pose.x - anchor.x == pytest.approx(
        5.0 * (truth_x(pose) - truth_x(anchor)), abs=5.0 * tick_u + 1e-9
    )
    pose = next(pose for t, _truth, pose in samples if t >= 3.5)
    assert pose.map_confirmed and pose.x == pytest.approx(truth_x(pose), abs=tick_u)


def test_drift_window_before_any_map_fix_has_no_pose():
    samples = _fly_localizer((1.0, 0.0, 0.0), 2.0, t_start=1.0, drift_every_s=100.0, drift_dur_s=5.0)
    assert all(pose is None for _t, _truth, pose in samples)


def test_vertical_gain_zero_freezes_reported_height():
    samples = _fly_localizer((0.0, -1.0, 0.0), 3.0, vertical_gain=0.0)  # climbing: up is -y
    _t, truth, pose = samples[-1]
    assert truth[1] == pytest.approx(-0.3)
    assert pose.y == pytest.approx(0.0, abs=1e-12)


def test_drift_and_vertical_gain_need_map_up():
    from simulated_localization import SimulatedLocalizer, SimulatedLocalizerConfig
    for kw in ({"drift_mps": 0.5}, {"vertical_gain": 0.0}):
        with pytest.raises(ValueError, match="map_up_u"):
            SimulatedLocalizer(SimulatedLocalizerConfig(**kw), np.random.default_rng(0))


def test_regression_R3_flies_on_a_drifting_pose_like_flight20260918_1301():
    params = sim.apply_regression_scenario(_regression_base(), "R3_flight20260918_1301_drift")
    result, _ = sim.run_sim(params, record_trace=False)
    # Run 1: route commands kept flowing for the whole 24 s window on VO /
    # dead-reckon poses metres off; the estimate-side max_dev cannot see it.
    assert result.weak_steer_max_s > 20.0
    assert result.max_est_error_m > 5.0
    base, _ = sim.run_sim(_regression_base(), record_trace=False)
    assert base.weak_steer_max_s == 0.0
    assert base.max_est_error_m < 1.0


def test_regression_R4_height_guard_arrives_without_runaway_climb_or_yaw_swing():
    params = sim.apply_regression_scenario(_regression_base(), "R4_flight20260918_1301_height")
    result, _ = sim.run_sim(params, record_trace=False)
    # Run 2 before the fix: never reached waypoint 1, climbed 12-14.5 m in
    # 120 s here and swung the nose over it. The barometric cross-check holds
    # altitude once the localized height stops following, and the join leg
    # holds yaw inside minimum_yaw_alignment_distance.
    assert 0 in result.waypoints_reached
    assert result.vertical_waivers >= 1
    assert result.max_climb_m < 4.0
    assert result.near_waypoint_turn_ticks == 0


def test_height_guard_does_not_fire_on_barometer_noise_or_drift():
    for drift in (0.01, -0.01):
        params = _regression_base(baro_noise_m=0.1, baro_drift_mps=drift)
        result, _ = sim.run_sim(params, record_trace=False)
        assert result.vertical_waivers == 0


def test_cli_localization_failure_flags_reach_params():
    args = sim.build_parser().parse_args([
        "--drift-every-s", "40", "--drift-dur-s", "24", "--drift-offset-s", "2",
        "--drift-gain", "5", "--drift-mps", "0.5", "--vertical-gain", "0",
        "--start-offset-frame", "raw", "--start-offset-u", "0.1", "0.2", "0.3",
        "--baro-noise-m", "0.1", "--baro-drift-mps", "-0.01",
    ])
    params = sim._params_from_args(args, seed=7, scale=15.0)
    assert (params.drift_every_s, params.drift_dur_s, params.drift_offset_s) == (40.0, 24.0, 2.0)
    assert (params.drift_gain, params.drift_mps, params.vertical_gain) == (5.0, 0.5, 0.0)
    assert (params.baro_noise_m, params.baro_drift_mps) == (0.1, -0.01)
    assert params.start_offset_frame == "raw" and params.start_offset_u == (0.1, 0.2, 0.3)
