import math
from types import SimpleNamespace

import pytest

from make_browser_sandbox import run_sandbox, wind_gust_vector


def _args(**overrides):
    data = {
        "algorithm": "translational_waypoint",
        "route_style": "complex",
        "num_waypoints": 4,
        "return_to_start": True,
        "inspection_poles": True,
        "inspection_pole_waypoints": "6,7",
        "inspection_pole_right_offset_m": 1.2,
        "inspection_pole_top_above_waypoint_m": 2.0,
        "map_size_m": 20.0,
        "map_margin_m": None,
        "segment_length_m": 3.0,
        "height_amp_m": 0.2,
        "start_radius_m": 0.8,
        "start_vert_jitter_m": 0.0,
        "start_quadrant": 1,
        "yaw_error_deg": 8.0,
        "duration": 2.0,
        "seed": 11,
        "max_frames": 200,
        "min_segment_time_s": 0.6,
        "max_segment_time_s": 20.0,
        "waypoint_hover_s": 0.3,
        "arrival_radius": 1.0,
        "arrival_vertical_radius": 1.0,
        "pose_source": "noisy_estimated",
        "pose_noise_m": 0.25,
        "pose_error_max_m": 1.0,
        "camera_yaw_noise_deg": 5.0,
        "camera_yaw_error_max_deg": 20.0,
        "pose_noise_seed": 123,
        "telemetry_delay_ms": 0.0,
        "telemetry_delay_jitter_ms": 0.0,
        "hloc_outage_interval_s": 0.0,
        "hloc_outage_duration_s": 0.0,
        "hloc_outage_start_s": 0.0,
        "wind_gust_interval_s": 5.0,
        "wind_gust_m": 0.5,
        "wind_gust_seed": 321,
        "route_tube_radius": 1.0,
        "route_tube_segment_window": 1,
        "route_tube_exit_s": 0.3,
        "route_tube_exit_updates": 2,
        "route_tube_initial_grace_s": 5.0,
        "camera_yaw_offset_deg": 0.0,
        "pcmd_rate_limit_pct_per_s": 200.0,
        "max_pose_age_s": 0.6,
        "pose_loss_short_s": 1.0,
        "lost_abort_s": 8.0,
        "final_landing_radius": 0.5,
        "final_landing_hold_s": 1.0,
        "final_landing_max_est_speed": 0.8,
        "final_landing_yaw_stable_deg": 20.0,
        "ideal_success_error_m": 0.5,
        "max_allowed_error_m": 3.0,
        "min_corridor_time_ratio": 0.8,
        "success_hold_s": 1.0,
        "out": "unused.html",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_browser_sandbox_noisy_estimated_pose_is_controller_input():
    data = run_sandbox(_args())

    assert data["meta"]["poseSource"] == "noisy_estimated"
    assert data["meta"]["controlPoseUsesTruth"] is False
    assert data["meta"]["distanceUnitLabel"] == "map u"
    assert data["meta"]["telemetryDelayMs"] == 0.0
    assert data["meta"]["mapSizeM"] == 20.0
    assert data["meta"]["returnToStart"] is True
    assert data["meta"]["controlWaypointCount"] == 7
    assert data["meta"]["poseErrorMaxM"] == 1.0
    assert data["meta"]["cameraYawErrorMaxDeg"] == 20.0
    assert data["meta"]["arrivalRadius"] == 1.0
    assert data["tubeRadius"] == 1.0
    assert data["bounds"] == {"minX": -10.0, "maxX": 10.0, "minZ": -10.0, "maxZ": 10.0}
    assert len(data["plannedRoute"]) == 4
    assert len(data["route"]) == 7
    assert data["routeWaypointLabels"] == ["W1", "W2", "W3", "W4", "W3", "W2", "W1"]
    assert data["route"][-1] == data["plannedRoute"][0]
    for x, _, z in data["plannedRoute"]:
        assert -10.0 <= x <= 10.0
        assert -10.0 <= z <= 10.0
    route_x = [p[0] for p in data["plannedRoute"]]
    route_z = [p[2] for p in data["plannedRoute"]]
    route_up = [-p[1] for p in data["plannedRoute"]]
    assert all(a < b for a, b in zip(route_x, route_x[1:]))
    assert max(route_z) - min(route_z) < 2.2
    assert max(route_up) - min(route_up) > 1.0
    assert -10.0 <= data["start"][0] <= 10.0
    assert -10.0 <= data["start"][2] <= 10.0

    errors = [f["poseError"] for f in data["frames"] if f["poseAvailable"]]
    assert errors
    assert max(errors) > 0.01
    assert max(errors) <= 1.0 + 1e-9
    assert any((f["px"], f["py"], f["pz"]) != (f["x"], f["y"], f["z"])
               for f in data["frames"])
    yaw_errors = [abs(f["yawErrorDeg"]) for f in data["frames"]]
    assert max(yaw_errors) <= 20.0 + 1e-9
    assert max(yaw_errors) > 1.0


def test_browser_sandbox_truth_pose_mode_is_explicit():
    data = run_sandbox(_args(pose_source="truth", pose_noise_m=0.0,
                             pose_error_max_m=0.0,
                             camera_yaw_noise_deg=0.0,
                             camera_yaw_error_max_deg=0.0))

    assert data["meta"]["poseSource"] == "truth"
    assert data["meta"]["controlPoseUsesTruth"] is True
    errors = [f["poseError"] for f in data["frames"] if f["poseAvailable"]]
    assert max(errors) < 1e-9
    assert max(abs(f["yawErrorDeg"]) for f in data["frames"]) < 1e-9


def test_browser_sandbox_can_simulate_hloc_delay_and_age_gate():
    data = run_sandbox(_args(duration=2.0, telemetry_delay_ms=250.0,
                             telemetry_delay_jitter_ms=0.0,
                             max_pose_age_s=0.6))

    assert data["meta"]["telemetryDelayMs"] == 250.0
    ages = [f["poseAgeS"] for f in data["frames"] if f["poseAgeS"] is not None]
    assert ages
    assert max(ages) >= 0.25
    assert any(f["poseAvailable"] is False or f["poseAgeS"] is None
               for f in data["frames"][:6])


def test_browser_sandbox_outage_enters_pose_loss_stage():
    data = run_sandbox(_args(duration=4.0, telemetry_delay_ms=0.0,
                             hloc_outage_interval_s=1.0,
                             hloc_outage_duration_s=0.8,
                             max_pose_age_s=0.2,
                             pose_loss_short_s=0.6,
                             lost_abort_s=5.0))

    assert data["meta"]["hlocOutageIntervalS"] == 1.0
    assert data["metrics"]["telemetryLostTimeS"] > 0.0
    assert any(reason.startswith("telemetry_stale_")
               for reason in data["metrics"]["failureReasons"])
    stale = [f for f in data["frames"] if f["telemetryStale"]]
    assert stale
    assert any(f["poseLossStage"] in {"short_hover_hold", "medium_wait_relocalize"}
               for f in stale)


def test_browser_sandbox_can_disable_return_leg():
    data = run_sandbox(_args(return_to_start=False))

    assert data["meta"]["returnToStart"] is False
    assert len(data["plannedRoute"]) == 4
    assert len(data["route"]) == 4
    assert data["routeWaypointLabels"] == ["W1", "W2", "W3", "W4"]


def test_browser_sandbox_inserts_w6_w7_pole_inspection_waypoints():
    data = run_sandbox(_args(num_waypoints=10, duration=1.0,
                             route_tube_initial_grace_s=5.0))

    labels = data["routeWaypointLabels"]
    assert len(data["plannedRoute"]) == 10
    assert len(data["poles"]) == 2
    assert {p["waypoint"] for p in data["poles"]} == {"W6", "W7"}
    assert labels.count("W6 pole top") == 2
    assert labels.count("W6 return height") == 2
    assert labels.count("W7 pole top") == 2
    assert labels.count("W7 return height") == 2
    assert data["meta"]["controlWaypointCount"] == 27

    for pole in data["poles"]:
        wp_idx = int(pole["waypoint"][1:]) - 1
        wp = data["plannedRoute"][wp_idx]
        assert math.hypot(pole["base"][0] - wp[0], pole["base"][2] - wp[2]) == pytest.approx(1.2)
        assert -pole["top"][1] == pytest.approx(-wp[1] + 2.0)


def test_wind_gust_vector_is_position_only_one_meter_impulse():
    rng = __import__("numpy").random.default_rng(7)
    yaw = 0.7
    for _ in range(20):
        vec, label = wind_gust_vector(yaw, rng, 1.0)
        assert label in {"forward", "back", "right", "left", "up", "down"}
        assert sum(float(v) ** 2 for v in vec) ** 0.5 == pytest.approx(1.0)


def test_browser_sandbox_records_wind_gust_events():
    data = run_sandbox(_args(duration=0.8, wind_gust_interval_s=0.2,
                             route_tube_initial_grace_s=5.0))

    gusts = [f for f in data["frames"] if f["windGust"]]
    assert gusts
    assert {round(f["windMagnitude"], 6) for f in gusts} == {0.5}
