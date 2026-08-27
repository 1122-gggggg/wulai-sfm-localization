import json
from pathlib import Path

import numpy as np
import pytest

from anafi_profile import ANAFI_PROFILE
from controllers import ALGORITHM_ORDER, CtrlParams, make_controller
from make_browser_sandbox import parse_args as parse_browser_args, run_sandbox
from route_geometry import RouteModel
from run_sphinx_anafi_convergence import build_trial_plan, parse_args
from telemetry_sources import KinematicParams


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def test_white_paper_profile_is_the_single_vehicle_spec():
    spec = ANAFI_PROFILE
    assert spec.name == "Parrot ANAFI"
    assert spec.mass_kg == pytest.approx(0.320)
    assert spec.max_horizontal_speed_mps == pytest.approx(15.0)
    assert spec.max_ascent_speed_mps == pytest.approx(4.0)
    assert spec.max_descent_speed_mps == pytest.approx(4.0)
    assert spec.max_angular_speed_deg_s == pytest.approx(200.0)
    assert spec.wind_resistance_kmh == pytest.approx(50.0)
    assert spec.wind_gust_kmh == pytest.approx(80.0)
    assert spec.takeoff_hover_height_m == pytest.approx(1.0)
    assert spec.hover_accuracy_m_at_1m == pytest.approx(0.015)
    assert spec.internal_control_loop_hz == pytest.approx(200.0)
    assert (spec.video_width_px, spec.video_height_px, spec.video_fps) == (1280, 720, 30)
    assert spec.video_bitrate_bps == 5_000_000
    assert spec.video_latency_ms == pytest.approx(280.0)
    assert (spec.gimbal_pitch_min_deg, spec.gimbal_pitch_max_deg) == (-90.0, 90.0)
    assert spec.gimbal_max_speed_deg_s == pytest.approx(180.0)
    assert spec.gps_position_std_m == pytest.approx(1.2)
    assert spec.gps_speed_std_mps == pytest.approx(0.5)
    assert spec.barometer_noise_std_m == pytest.approx(0.2)
    assert "white-paper_anafi-v1.4-en.pdf" in spec.source_url


def test_kinematic_defaults_reference_profile_but_controller_caps_stay_conservative():
    kp = KinematicParams()
    assert kp.v_max_horiz == ANAFI_PROFILE.max_horizontal_speed_mps
    assert kp.v_max_up == ANAFI_PROFILE.max_ascent_speed_mps
    assert kp.v_max_down == ANAFI_PROFILE.max_descent_speed_mps
    assert np.degrees(kp.yaw_rate_max) == pytest.approx(
        ANAFI_PROFILE.max_angular_speed_deg_s
    )
    assert kp.integration_hz == ANAFI_PROFILE.internal_control_loop_hz

    caps = CtrlParams()
    assert (caps.max_pitch, caps.max_roll, caps.max_yaw, caps.max_gaz) == (10, 6, 25, 15)


def test_browser_defaults_and_metadata_reference_profile():
    args = parse_browser_args([])
    assert args.video_e2e_latency_ms == ANAFI_PROFILE.video_latency_ms
    assert args.decode_localization_latency_ms == 0.0
    assert args.telemetry_delay_ms is None

    args.duration = 0.1
    args.num_waypoints = 2
    args.return_to_start = False
    args.inspection_poles = False
    args.pose_source = "noisy_estimated"
    args.wind_gust_interval_s = 0.0
    data = run_sandbox(args)
    assert data["meta"]["anafiProfile"] == ANAFI_PROFILE.to_metadata()
    assert data["meta"]["backendKind"] == "kinematic_approximation"
    assert data["meta"]["telemetryDelayMs"] == ANAFI_PROFILE.video_latency_ms
    assert data["meta"]["telemetryDelayIsLowerBound"] is True


def test_compare_plan_pairs_identical_scenarios_across_algorithms():
    args = parse_args([
        "--backend", "kinematic",
        "--rejoin-algorithm", "compare",
        "--num-trials", "3",
        "--random-seed", "17",
    ])
    plan = build_trial_plan(args, np.random.default_rng(args.random_seed))

    by_scenario = {}
    for trial in plan:
        by_scenario.setdefault(trial["scenario_id"], []).append(trial)
    assert len(by_scenario) == 3
    assert all(len(rows) == len(ALGORITHM_ORDER) for rows in by_scenario.values())
    for rows in by_scenario.values():
        paired = {
            (row["seed"], row["quadrant"], row["yaw_error_deg"], row["perturbation"])
            for row in rows
        }
        assert len(paired) == 1


def test_equals_style_cli_flag_wins_over_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"duration": 60.0, "num_trials": 20}))
    args = parse_args([f"--config={cfg}", "--duration=4", "--num-trials=2"])
    assert args.duration == 4.0
    assert args.num_trials == 2


def test_default_compare_config_preserves_algorithm_variants():
    cfg = EXPERIMENT_DIR / "configs" / "algorithm_compare.json"
    args = parse_args(["--config", str(cfg)])
    assert args.rejoin_algorithm == "compare"
    assert args.horizontal_control_mode is None

    route = RouteModel([np.zeros(3), np.array([4.0, 0.0, 0.0])])
    lateral = make_controller("segment_corridor_lateral", route)
    nose_first = make_controller("segment_corridor_nose_first", route)
    assert lateral.p.horizontal_control_mode == "small_lateral_assist"
    assert nose_first.p.horizontal_control_mode == "nose_first"


@pytest.mark.parametrize("argv", [
    ["--num-trials", "0"],
    ["--num-trials", "-1"],
    ["--duration", "nan"],
    ["--duration", "inf"],
    ["--start-radius-m", "-0.1"],
    ["--start-radius-m", "0.5"],
    ["--telemetry-drop-rate", "1.01"],
    ["--min-corridor-time-ratio", "-0.1"],
    ["--yaw-error-list", "0,nan"],
    ["--map-units-per-meter", "0"],
])
def test_runner_rejects_nonfinite_and_out_of_range_benchmark_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_runner_rejects_wrong_numeric_type_from_config(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"duration": "60"}))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(cfg)])


@pytest.mark.parametrize("argv", [
    ["--duration", "nan"],
    ["--num-waypoints", "1"],
    ["--max-frames", "0"],
    ["--video-e2e-latency-ms", "-1"],
    ["--decode-localization-latency-ms", "inf"],
    ["--telemetry-delay-ms", "-0.1"],
])
def test_browser_rejects_nonfinite_and_out_of_range_values(argv):
    with pytest.raises(SystemExit):
        parse_browser_args(argv)
