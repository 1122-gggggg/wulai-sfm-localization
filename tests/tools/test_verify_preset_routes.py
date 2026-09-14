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
ALIGN = ROOT / "地圖檔/場域/river_site/releases/river_gluemap_all8_direct_20260908/localization/T_align_gravity.json"


@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_every_preset_route_from_every_nearest_waypoint(tmp_path):
    routes = sorted(ROUTES.glob("*.json"))
    assert routes, "preset routes must be present for route acceptance"
    expected_count = sum(len(json.loads(path.read_text())["waypoints"]) for path in routes)
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/verify_preset_routes.py"), "--quick", "--out", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads((tmp_path / "results.json").read_text())
    assert len(rows) == expected_count
    assert {row["route"] for row in rows} == {path.name for path in routes}
    for row in rows:
        assert row["accepted"] and row["completed"] and row["sequence_ok"]
        assert row["join_waypoint"] == row["expected_join_waypoint"]
        assert row["observed_target_order"] == row["expected_waypoint_indices"]
        assert row["params"]["max_vertical_speed_mps"] == 2.0
        assert row["params"]["max_rotation_speed_deg_s"] == 20.0
        assert row["yaw_horizontal_overlap_samples"] == 0


@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_gust_parameter_applies_the_requested_physical_displacement():
    route = next(ROUTES.glob("*.json"))
    params = sim.SimParams(route=route, align=ALIGN, duration_s=0.05,
                           gust_every_s=0.05, gust_m=0.3, wind_sigma_mps=0)
    result, _ = sim.run_sim(params)
    assert np.linalg.norm(result.trace[0]["gust_displacement_m"]) == pytest.approx(0.3)


@pytest.mark.parametrize("route", sorted(ROUTES.glob("*.json")), ids=lambda path: path.stem)
@pytest.mark.parametrize("gust_m", [0.2, 0.5])
@pytest.mark.skipif(not ALIGN.is_file(), reason="requires the operator's river site assets")
def test_every_route_recovers_from_repeated_wind_displacement(route, gust_m):
    result = evaluate_gust_recovery(sim.SimParams(
        route=route, align=ALIGN, duration_s=300, meters_per_unit=5,
        seed=53, gust_m=gust_m, gust_every_s=20,
    ))
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
    offset = controller.wp[join_index] - controller.wp[0] - .08 * frame.east
    result = evaluate(sim.SimParams(
        route=route, align=ALIGN, meters_per_unit=5., duration_s=300., seed=41,
        start_offset_frame="raw", start_offset_u=tuple(offset),
        initial_yaw_error_deg=170., latency_ms=300., pos_noise_u=.006,
        yaw_noise_deg=2., drop_rate=.05, wind_sigma_mps=.08,
        outage_every_s=60., outage_dur_s=.7,
    ), start_label=f"delayed_noisy_waypoint_{join_index + 1}")
    # Waypoint-1 start policy (start_after_nearest_waypoint always targets
    # index 0): the start label records the takeoff neighborhood, the join
    # target stays waypoint 1.
    assert result["join_waypoint"] == 1
    assert result["accepted"], result
