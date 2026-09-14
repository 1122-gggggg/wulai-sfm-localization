"""The turn-gate debug bundle splits spin causes without offline recomputation."""

from __future__ import annotations

import json
import math
import pytest
from pathlib import Path

import flight_debug_bundle as bundle


def _write(tmp_path: Path, files: dict[str, list[dict]]) -> Path:
    session = tmp_path / "session_20260914T000000Z_real-flight_test1234"
    session.mkdir(parents=True)
    for name, rows in files.items():
        (session / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return session


def _tick(step: int, **overrides) -> dict:
    row: dict = {
        "event": "auto_route_tick",
        "t": 100.0 + step * 0.05,
        "t_mono_ns": int((100.0 + step * 0.05) * 1e9),
        "step": step,
        "action": "FOLLOW",
        "pcmd_phase": "turn",
        "target_index": 0,
        "target_u": [1.0, 0.0, 0.0],
        "pose_u": [0.0, 0.0, 0.0],
        "heading_deg": 0.0,
        "imu_yaw_ned_rad": math.pi / 2.0,
        "pcmd": [0, 0, 20, 0],
        "blocked": False,
        "reason": "FOLLOW",
    }
    row.update(overrides)
    return row


def _gps(t: float, lat: float, lon: float) -> dict:
    return {
        "event": "fused_odometry",
        "t_mono_ns": int(t * 1e9),
        "gps_latitude_deg": lat,
        "gps_longitude_deg": lon,
    }


def _source(t: float, source: str) -> dict:
    return {
        "event": "piloting_source",
        "t_mono_ns": int(t * 1e9),
        "source": source,
    }


def test_logged_error_needs_no_recomputation(tmp_path: Path) -> None:
    ticks = [
        _tick(0, yaw_error_deg=12.0, pcmd_phase="translate"),
        _tick(1, yaw_error_deg=11.0, pcmd_phase="translate"),
    ]
    report = bundle.analyze_session(
        _write(tmp_path, {"localization.jsonl": ticks, "commands.jsonl": [],
                           "telemetry.jsonl": [], "incidents.jsonl": []})
    )
    assert report["turn_gate"]["logged_error_ticks"] == 2
    assert report["turn_gate"]["recomputed_error_ticks"] == 0
    assert report["turn_gate"]["max_abs_yaw_error_deg"] == 12.0
    assert report["verdict"] == "TURN_CONVERGED"


def test_segment_heading_is_not_recomputed_as_waypoint_bearing():
    ticks = [_tick(0, yaw_target_deg=90, yaw_error_deg=90)]
    report = bundle.analyze_turn_gate(ticks)
    assert report["error_agreement_deg"]["max"] == 0


def test_twenty_hz_durations_are_not_rounded_per_tick():
    report = bundle.analyze_turn_gate([_tick(i) for i in range(21)])
    assert report["turn_seconds"] == 1.0


def test_crossing_angle_wrap_is_not_a_large_fusion_jump():
    rows = [_tick(0, heading_deg=179), _tick(1, heading_deg=-179)]
    assert bundle.analyze_heading_reconciliation(rows)["offset_walk_deg"] == 2


def test_planned_target_change_is_separated_from_position_drift():
    rows = [_tick(0, target_index=0, yaw_target_deg=0),
            _tick(1, target_index=1, yaw_target_deg=90)]
    assert bundle.analyze_heading_reconciliation(rows)["verdict"] == "multiple_targets_review_per_leg"
    assert [target["target_index"] for target in bundle.analyze_auto_runs(rows)[0]["targets"]] == [0, 1]


@pytest.mark.parametrize("action,phase,verdict", [
    ("final path reached -> LAND", "idle", "ROUTE_FINAL_REACHED"),
    ("REJOIN", "route_rejoin", "POSITION_CORRECTION_ACTIVE"),
    ("FOLLOW", "height_adjust", "POSITION_CORRECTION_ACTIVE"),
])
def test_arrival_and_position_repair_are_not_reported_as_failed_turns(tmp_path, action, phase, verdict):
    session = _write(tmp_path, {"localization.jsonl": [_tick(0, action=action, pcmd_phase=phase)]})
    assert bundle.analyze_session(session)["verdict"] == verdict


def test_auto_runs_keep_their_own_clock_and_plan():
    rows = [
        {"event": "auto_route_plan", "auto_run_id": "first"},
        _tick(0, auto_run_id="first", yaw_error_deg=10),
        _tick(1, auto_run_id="first", yaw_error_deg=5, pcmd_phase="translate"),
        {"event": "auto_route_plan", "auto_run_id": "second"},
        _tick(0, t=500.0, auto_run_id="second", yaw_error_deg=30),
    ]
    runs = bundle.analyze_auto_runs(rows)
    assert [run["tick_count"] for run in runs] == [2, 1]
    assert runs[0]["turn_gate"]["convergence"]["time_to_converge_s"] == pytest.approx(0.05)
    assert runs[1]["turn_gate"]["convergence"] is None


def test_error_only_in_band_is_not_convergence(tmp_path: Path) -> None:
    # A 12 deg starting error is "in band" at tick 0 while the gate still
    # spins (measured 2026-09-14: 21 s of pure-yaw PCMD before translate).
    # Convergence requires a translate-phase tick, not an in-band error.
    ticks = [
        _tick(0, pcmd_phase="turn", pcmd=[0, 0, -16, 0]),
        _tick(1, pcmd_phase="turn", pcmd=[0, 0, -16, 0]),
        _tick(2, pcmd_phase="translate", pcmd=[0, 3, 0, 1],
              yaw_error_deg=12.0),
    ]
    report = bundle.analyze_session(
        _write(tmp_path, {"localization.jsonl": ticks, "commands.jsonl": [],
                           "telemetry.jsonl": [], "incidents.jsonl": []})
    )
    assert report["turn_gate"]["convergence"]["tick"] == 2
    assert report["turn_gate"]["final_phase"] == "translate"


def test_regressed_leg_is_not_converged(tmp_path: Path) -> None:
    ticks = [
        _tick(0, pcmd_phase="translate", yaw_error_deg=5.0),
        _tick(1, pcmd_phase="turn", pcmd=[0, 0, -50, 0]),
    ]
    report = bundle.analyze_session(
        _write(tmp_path, {"localization.jsonl": ticks, "commands.jsonl": [],
                           "telemetry.jsonl": [], "incidents.jsonl": []})
    )
    assert report["turn_gate"]["convergence"]["tick"] == 0
    assert report["turn_gate"]["final_phase"] == "turn"
    assert report["verdict"] == "TURN_REGRESSED"


def test_drift_splits_auto_from_manual_by_piloting_source(tmp_path: Path) -> None:
    telemetry = [
        _gps(100.0, 25.0, 121.5),
        _gps(101.0, 25.0, 121.5),
        _gps(102.0, 25.0001, 121.5),  # ~11 m north, manual window
        _gps(103.0, 25.0001, 121.5),
    ]
    commands = [_source(99.0, "Controller"), _source(101.5, "SkyController")]
    report = bundle.analyze_session(
        _write(tmp_path, {"localization.jsonl": [], "commands.jsonl": commands,
                           "telemetry.jsonl": telemetry, "incidents.jsonl": []})
    )
    assert report["verdict"] == "NO_AUTO"
    assert report["drift"]["gps_excursion_auto_m"] == 0.0
    assert report["drift"]["gps_excursion_manual_m"] == 0.0
    assert report["drift"]["gps_excursion_m"] > 10.0


def test_heading_reconciliation_names_bearing_walk(tmp_path: Path) -> None:
    # Hold the nose fixed while the position walks sideways past the target:
    # the target bearing sweeps >10 deg while the fusion offset holds.
    ticks = [
        _tick(s, pose_u=[0.0, 0.0, s * 0.05],
              heading_deg=0.0, imu_yaw_ned_rad=math.pi / 2.0)
        for s in range(10)
    ]
    report = bundle.analyze_session(
        _write(tmp_path, {"localization.jsonl": ticks, "commands.jsonl": [],
                           "telemetry.jsonl": [], "incidents.jsonl": []})
    )
    assert report["heading"]["verdict"] == "target_bearing_walk_position_drift"
    assert report["heading"]["offset_walk_deg"] < 2.0
