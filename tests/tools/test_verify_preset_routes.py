"""All available preset routes must work from each nearest-waypoint start."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

import sim_route_autoflight as sim
from verify_preset_routes import evaluate, evaluate_gust_recovery


ROOT = Path(__file__).resolve().parents[2]
ROUTES = ROOT / "地圖檔/場域/river_site/routes"
ALIGN = (
    ROOT
    / "地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260908/localization/T_align_gravity.json"
)


@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_every_preset_route_from_every_nearest_waypoint(tmp_path):
    routes = sorted(ROUTES.glob("*.json"))
    assert routes, "preset routes must be present for route acceptance"
    expected_count = sum(len(json.loads(path.read_text())["waypoints"]) for path in routes)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/verify_preset_routes.py"),
            "--quick",
            "--localization-wait-s",
            "0",
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads((tmp_path / "results.json").read_text())
    assert len(rows) == expected_count
    assert {row["route"] for row in rows} == {path.name for path in routes}
    for row in rows:
        assert row["accepted"] and row["completed"] and row["sequence_ok"]
        assert row["join_waypoint"] == row["expected_join_waypoint"]
        assert row["observed_target_order"] in row["accepted_target_orders"]
        assert row["retired_target_order"] == row["expected_waypoint_indices"]
        assert row["waypoint_indices_reached"] == row["expected_waypoint_indices"]
        assert row["arrival_truth_ok"]
        assert all(item["ok"] for item in row["arrival_truth"])
        assert row["params"]["max_vertical_speed_mps"] == 2.0
        assert row["params"]["max_rotation_speed_deg_s"] == 20.0
        assert row["phase_contract_violations"] == 0
        assert row["yaw_translation_overlap_samples"] == 0
        assert row["vertical_translation_overlap_samples"] == 0


@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_gust_parameter_applies_the_requested_physical_displacement():
    route = next(ROUTES.glob("*.json"))
    params = sim.SimParams(
        route=route, align=ALIGN, duration_s=0.05, gust_every_s=0.05, gust_m=0.3, wind_sigma_mps=0
    )
    result, _ = sim.run_sim(params)
    assert np.linalg.norm(result.trace[0]["gust_displacement_m"]) == pytest.approx(0.3)


@pytest.mark.parametrize("route", sorted(ROUTES.glob("*.json")), ids=lambda path: path.stem)
@pytest.mark.parametrize("gust_m", [0.2, 0.5])
@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_every_route_recovers_from_repeated_wind_displacement(route, gust_m):
    result = evaluate_gust_recovery(
        sim.SimParams(
            route=route,
            align=ALIGN,
            duration_s=300,
            meters_per_unit=5,
            seed=53,
            gust_m=gust_m,
            gust_every_s=20,
        )
    )
    assert result["accepted"], result
    assert all(item["displacement_m"] == pytest.approx(gust_m) for item in result["gusts"])


@pytest.mark.parametrize("join_index", [0, 3, 5], ids=["waypoint_1", "waypoint_4", "waypoint_6"])
@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_six_point_route_converges_in_previously_failing_delayed_noisy_cases(join_index):
    """Regress the timeout, excessive cross-track error and terminal stall cases."""
    route = ROUTES / "flight_route_20260914_141505_e87a66fb.json"
    controller, _config, frame, _doc = sim.build_production_controller(
        route, ALIGN, return_to_start=True
    )
    offset = controller.wp[join_index] - controller.wp[0] - 0.08 * frame.east
    result = evaluate(
        sim.SimParams(
            route=route,
            align=ALIGN,
            meters_per_unit=5.0,
            duration_s=300.0,
            seed=41,
            start_offset_frame="raw",
            start_offset_u=tuple(offset),
            initial_yaw_error_deg=170.0,
            latency_ms=300.0,
            pos_noise_u=0.006,
            yaw_noise_deg=2.0,
            drop_rate=0.05,
            wind_sigma_mps=0.08,
            outage_every_s=60.0,
            outage_dur_s=0.7,
        ),
        start_label=f"delayed_noisy_waypoint_{join_index + 1}",
    )
    # Waypoint-1 start policy (start_after_nearest_waypoint always targets
    # index 0): the start label records the takeoff neighborhood, the join
    # target stays waypoint 1.
    assert result["join_waypoint"] == 1
    assert result["accepted"], result


@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_initial_localization_wait_consumes_the_total_mission_budget():
    route = next(ROUTES.glob("*.json"))
    result, info = sim.run_sim(
        sim.SimParams(route=route, align=ALIGN, duration_s=10, localization_wait_s=20)
    )
    assert not result.success
    assert result.steps == 0 and result.time_s == 10
    assert result.dist_flown_m == 0
    assert info["route_time_s"] == 0
    assert "localization wait" in result.reason


@pytest.mark.parametrize("wait", [-1, float("nan"), float("inf")])
def test_invalid_localization_wait_is_rejected(wait):
    with pytest.raises(ValueError, match="localization_wait_s"):
        sim._simulation_budget(
            sim.SimParams(route=Path("unused"), align=None, localization_wait_s=wait)
        )


@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_short_visual_blackout_recovers_on_imu_hold():
    """Total visual loss rides the production IMU-bridge law, then recovers.

    During the blackout the estimate holds the last visual anchor (it must
    not track truth motion: scale-free map has no velocity source) and the
    run still completes once vision returns.
    """
    import numpy as np
    from collections import Counter

    route = next(ROUTES.glob("*.json"))
    params = sim.SimParams(
        route=route,
        align=ALIGN,
        meters_per_unit=10,
        seed=7,
        blackout_every_s=40,
        blackout_dur_s=3,
        blackout_drift_mps=0.05,
        blackout_yaw_drift_deg_s=2.0,
    )
    result, _info = sim.run_sim(params, record_trace=True)
    assert result.success, result.reason
    faults = Counter(tick.get("loc_fault", "?") for tick in result.trace)
    assert faults["blackout-imu"] > 0
    est = np.array([tick["est_map_u"] for tick in result.trace])
    tru = np.array([tick["true_map_u"] for tick in result.trace])
    est_step = np.linalg.norm(np.diff(est, axis=0), axis=1)
    tru_step = np.linalg.norm(np.diff(tru, axis=0), axis=1)
    bo = [i for i, tick in enumerate(result.trace) if tick.get("loc_fault") == "blackout-imu"]
    bo = [i for i in bo if i > 0]
    assert bo
    assert float(np.mean([est_step[i - 1] for i in bo])) < float(
        np.mean([tru_step[i - 1] for i in bo])
    )
