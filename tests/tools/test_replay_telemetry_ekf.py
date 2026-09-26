import json
import math

import numpy as np
import pytest

from replay_telemetry_ekf import run
from fusion_session_data import fit_horizontal_calibration, load_session


def synthetic_session(path, seed=7):
    """Known metric circle, independent camera noise and delayed fused telemetry."""
    path.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    scale, theta = 2.5, 0.35
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    visual = [
        {
            "event": "auto_route_plan",
            "coordinate_frame_id": "synthetic",
            "control_config": {"map_frame": dict(zip(("east", "north", "up"), np.eye(3).tolist()))},
        }
    ]
    truth = {}
    for index, stamp in enumerate(np.arange(100.0, 200.0, 0.05)):
        phase = (stamp - 100.0) * 0.12
        en = 10.0 * np.array([math.cos(phase) - 1.0, math.sin(phase)])
        xyz = np.r_[rotation @ en / scale, 0.0]
        truth[float(stamp)] = xyz
        noise = rng.normal(0, 0.004, 3)
        # A short visual outage each ten seconds; no fabricated weak anchors.
        observed = index % 200 not in {150, 151, 152, 153, 154}
        visual.append(
            {
                "event": "pose_result",
                "source_frame_stamp_mono": float(stamp),
                "t_mono_ns": round((stamp + 0.1) * 1e9),
                "success": observed,
                "direct_status": "FAST_TRACK" if observed else "VO_ONLY",
                "pose_raw": dict(zip(("x", "y", "z"), (xyz + noise).tolist())),
                "camera_axes_world": np.eye(3).tolist(),
            }
        )
    velocities = []
    for stamp in np.arange(100.0, 200.0, 0.1):
        phase = (stamp - 100.0) * 0.12
        e, n = 1.2 * np.array([-math.sin(phase), math.cos(phase)]) + rng.normal(0, 0.015, 2)
        row = {
            "event": "fused_odometry",
            "t_mono_ns": round((stamp + 0.12) * 1e9),
            "speed_mono_ns": round(stamp * 1e9),
            "speed_north_mps": n,
            "speed_east_mps": e,
            "speed_down_mps": 0.0,
        }
        velocities.extend([row, row])  # polls do not create extra sensor samples
    for name, rows in (("localization.jsonl", visual), ("telemetry.jsonl", velocities)):
        (path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return truth


def test_complete_replay_calibrates_only_prefix_and_keeps_predictions_separate(tmp_path):
    session = tmp_path / "synthetic"
    truth = synthetic_session(session)
    result = run(session, tmp_path / "out")
    calibration = result["calibration"]
    assert calibration["ready"]
    assert calibration["scale_m_per_unit"] == pytest.approx(2.5, rel=0.03)
    assert calibration["yaw_offset_rad"] == pytest.approx(0.35, abs=0.02)
    assert not result["formal_control_ready"]
    assert result["distinct_velocity_samples"] == 1000
    evaluation = result["evaluation"]
    assert evaluation["velocity_accepted"] > 400
    assert evaluation["common_heldout_position_residual_u"]["telemetry_esekf"]["count"] > 100
    rows = [json.loads(s) for s in (tmp_path / "out/poses.jsonl").read_text().splitlines()]
    error = []
    for row in rows:
        filtered = row["telemetry_esekf"]
        if row["held_out"]:
            assert not filtered["accepted"]
        if filtered["pose"] is not None:
            matrix = np.asarray(filtered["pose"])
            error.append(np.linalg.norm(-matrix[:, :3].T @ matrix[:, 3] - truth[row["stamp"]]))
            assert min(filtered["covariance_diagonal"]) > 0.0
    assert np.percentile(error, 95) < 0.03


def test_insufficient_scale_is_reported_without_inventing_calibration():
    result = fit_horizontal_calibration([(np.array([1.0, 0.0]), np.array([2.0, 0.0]))])
    assert not result["ready"] and "scale_m_per_unit" not in result


def test_sensor_clock_and_coordinate_frame_conflicts_are_rejected(tmp_path):
    session = tmp_path / "synthetic"
    synthetic_session(session)
    telemetry = session / "telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    for row in rows:
        row["speed_mono_ns"] = row["t_mono_ns"] + 100_000_000
    telemetry.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _, velocities, _, _ = load_session(session)
    assert len(velocities) == 0
    with (session / "localization.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "auto_route_plan",
                    "coordinate_frame_id": "other",
                    "control_config": {
                        "map_frame": dict(zip(("east", "north", "up"), np.eye(3).tolist()))
                    },
                }
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="one recorded map frame"):
        load_session(session)


def test_outage_replay_uses_prefix_only_and_never_claims_flight_evidence(tmp_path):
    from replay_outage_bridge import run as replay_outages

    session = tmp_path / "synthetic"
    synthetic_session(session)
    report = replay_outages(session, tmp_path / "outages")
    assert report["calibration"]["ready"]
    assert not report["formal_control_ready"]
    assert report["production_prediction_horizon_s"] == 0.5
    for case in report["evaluation"]:
        errors = case["horizontal_reference_residual_u"]
        assert errors["ned_velocity"]["count"] > 5
        assert errors["ned_velocity"]["p95"] < errors["frozen"]["p95"]
    rows = [
        json.loads(line) for line in (tmp_path / "outages/outages.jsonl").read_text().splitlines()
    ]
    assert all(row["start_mono"] > report["calibration_cutoff_mono"] for row in rows)
    assert all(not row["map_confirmed"] and not row["arrival_evidence"] for row in rows)


def test_outage_velocity_never_uses_future_samples_or_crosses_stale_gaps():
    from replay_outage_bridge import causal_displacement

    velocity = np.array([[1.0, 0.0, 1.0, 0.0], [1.1, 0.0, 1.0, 0.0], [1.4, 0.0, 100.0, 0.0]])
    np.testing.assert_allclose(causal_displacement(velocity, 1.0, 1.3), [0.0, 0.3])
    assert causal_displacement(velocity, 0.9, 1.3) is None
    assert causal_displacement(velocity, 1.0, 2.0) is None
    assert (
        causal_displacement(np.array([[1.0, 0.0, 1.0, 0.0], [2.0, 0.0, 1.0, 0.0]]), 1.0, 2.0)
        is None
    )


def test_outage_replay_refuses_inconsistent_calibration(tmp_path, monkeypatch):
    import replay_outage_bridge

    session = tmp_path / "synthetic"
    synthetic_session(session)
    monkeypatch.setattr(
        replay_outage_bridge,
        "fit_horizontal_calibration",
        lambda _: {
            "ready": False,
            "reason": "inconsistent scale/yaw",
            "scale_m_per_unit": 2.5,
        },
    )
    report = replay_outage_bridge.run(session, tmp_path / "outages")
    assert report["evaluation"]["unavailable"] == "inconsistent scale/yaw"
    assert not (tmp_path / "outages/outages.jsonl").read_text()


def test_recorded_motion_history_preserves_samples_between_ui_polls(tmp_path):
    session = tmp_path / "synthetic"
    synthetic_session(session)
    (session / "telemetry.jsonl").write_text(
        json.dumps(
            {
                "event": "control_motion_history",
                "t_mono_ns": 100_200_000_000,
                "horizontal_ned_samples": [[0.0, 1.0, 100.0], [0.0, 2.0, 100.1], [0.0, 1.0, 100.2]],
            }
        )
        + "\n"
    )
    _, velocity, _, _ = load_session(session)
    assert len(velocity) == 3
    np.testing.assert_allclose(velocity[:, 2], [1.0, 2.0, 1.0])
