"""The IMU flight-test report has one job: refuse a recording that cannot be evaluated."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import imu_flight_test_report as report


def _telemetry_rows(
    *,
    samples: int = 60,
    velocity: bool = True,
    attitude: bool = True,
    moving: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    for index in range(samples):
        row: dict = {
            "event": "fused_odometry",
            "t_mono_ns": int((1000.0 + index * 0.125) * 1e9),
        }
        if attitude:
            sweep = math.radians(index * (1.0 if moving else 0.0))
            row.update({"att_roll": 0.02, "att_pitch": -0.03, "att_yaw": sweep})
        if velocity:
            speed = 2.5 if moving else 0.0
            row.update(
                {
                    "speed_north_mps": speed,
                    "speed_east_mps": 0.1,
                    "speed_down_mps": -0.2,
                }
            )
        rows.append(row)
    for index in range(10):
        rows.append(
            {
                "event": "stick_axes",
                "t_mono_ns": int((1000.0 + index) * 1e9),
                "axes": {"0": 0, "1": 8000 if moving else 0, "2": 0, "3": 0},
                "moved": bool(moving),
                "flight_axes_active": bool(moving),
                "pilot_sticks": True,
            }
        )
    return rows


def _localization_rows(*, fused: bool = True, lost: bool = True, esekf: bool = True):
    rows = []
    states = ["BOOT", "TRACK", "TRACK", "WEAK_TRACK", "LOST", "TRACK"] if lost else ["TRACK"] * 6
    for index, state in enumerate(states):
        row: dict = {
            "event": "pose_result",
            "next_mode": state,
            "success": state != "LOST",
            "pose_status": "VISUALLY_CONFIRMED" if state != "LOST" else "NONE",
            "source_frame_stamp_mono": 1000.0 + index * 0.2,
        }
        if fused:
            row.update(
                {
                    "fused_telemetry_mono": 1000.0 + index * 0.2 - 0.02,
                    "fused_roll": 0.01,
                    "fused_speed_north": 2.5,
                }
            )
        if esekf:
            row.update(
                {
                    "prediction_mode": "esekf",
                    "esekf_update_accepted": True,
                    "esekf_d2": 1.4,
                    "esekf_pos_trace": 0.2,
                }
            )
        rows.append(row)
    return rows


def _write_session(tmp_path: Path, telemetry, localization, frames=6) -> Path:
    session = tmp_path / "session_20260905T000000Z_real-flight_abcd1234"
    (session / "imu_test" / "frames").mkdir(parents=True)
    (session / "telemetry.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in telemetry),
        encoding="utf-8",
    )
    (session / "localization.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in localization),
        encoding="utf-8",
    )
    index = []
    for seq in range(frames):
        (session / "imu_test" / "frames" / f"{seq:06d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        index.append(
            {
                "seq": seq,
                "file": f"frames/{seq:06d}.jpg",
                "capture_stamp_mono": 1000.0 + seq * 0.2,
                "fused_speed_north": 2.5,
            }
        )
    (session / "imu_test" / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index), encoding="utf-8"
    )
    (session / "imu_test" / "summary.json").write_text(
        json.dumps({"written": frames, "dropped_queue_full": 0, "stop_reason": ""}),
        encoding="utf-8",
    )
    return session


def test_a_complete_recording_is_reported_usable(tmp_path) -> None:
    session = _write_session(tmp_path, _telemetry_rows(), _localization_rows())
    built = report.build_report(session)
    assert built["verdict"] == "USABLE"
    assert built["notes"] == []
    assert built["imu"]["with_velocity"] == 60
    assert built["imu"]["rate_hz"] == pytest.approx(8.0, rel=0.05)
    assert built["localization"]["lost_episodes"] == 1
    assert built["localization"]["fused_paired"] == 6
    assert built["localization"]["esekf_predictions"] == 6
    assert built["frames"]["indexed"] == 6 and built["frames"]["on_disk"] == 6
    # Rendering must not raise on a fully populated report.
    assert "IMU 飛行測試報告" in report.render(built)


def test_a_recording_without_velocity_cannot_be_evaluated(tmp_path) -> None:
    # ESEKF.prediction_allowed() is gated on live NED velocity; with none the
    # offline A/B can only return INVALID, so the report must say so here
    # rather than after a full GPU replay.
    session = _write_session(
        tmp_path,
        _telemetry_rows(velocity=False),
        _localization_rows(fused=False, esekf=False),
    )
    built = report.build_report(session)
    assert built["verdict"] == "UNUSABLE"
    assert any("速度" in note for note in built["notes"])


def test_localization_never_started_is_a_blocker(tmp_path) -> None:
    session = _write_session(tmp_path, _telemetry_rows(), [], frames=0)
    built = report.build_report(session)
    assert built["verdict"] == "UNUSABLE"
    assert any("沒有任何一幀" in note for note in built["notes"])


def test_a_parked_flight_is_flagged_but_not_rejected(tmp_path) -> None:
    session = _write_session(
        tmp_path,
        _telemetry_rows(moving=False),
        _localization_rows(lost=False, esekf=False),
    )
    built = report.build_report(session)
    assert built["verdict"] == "USABLE WITH GAPS"
    joined = " ".join(built["notes"])
    assert "靜止" in joined
    assert "LOST" in joined
    assert "ESEKF" in joined


def test_stale_telemetry_pairs_are_counted_against_the_protocol_bound(tmp_path) -> None:
    rows = _localization_rows()
    for row in rows:
        row["fused_telemetry_mono"] = row["source_frame_stamp_mono"] - 0.4
    session = _write_session(tmp_path, _telemetry_rows(), rows)
    built = report.build_report(session)
    assert built["localization"]["sync_error_over_bound"] == len(rows)
    assert built["verdict"] == "USABLE WITH GAPS"
    assert any(str(report.MAX_FUSED_SYNC_ERROR_S) in note for note in built["notes"])


def test_cli_exits_nonzero_only_for_an_unusable_session(tmp_path, capsys) -> None:
    good = _write_session(tmp_path / "good", _telemetry_rows(), _localization_rows())
    assert report.main(["--session", str(good)]) == 0
    bad = _write_session(tmp_path / "bad", _telemetry_rows(velocity=False), [])
    assert report.main(["--session", str(bad)]) == 1
    assert "IMU 飛行測試報告" in capsys.readouterr().out


def test_tick_profile_names_the_stage_that_holds_the_event_loop() -> None:
    """runbook 0b's open question, answered from one flight's telemetry."""
    from imu_flight_test_report import summarize_tick_profile

    summary = summarize_tick_profile([
        {
            "event": "ui_tick_profile",
            "ticks": 150,
            "tick_period_ms": 33,
            "stages": {
                "_total": {"p50": 20.0, "p95": 41.0},
                "render_if_dirty": {"p50": 3.0, "p95": 9.0},
                "update_stream": {"p50": 14.0, "p95": 30.0},
            },
        },
        {
            "event": "ui_tick_profile",
            "ticks": 150,
            "tick_period_ms": 33,
            "stages": {
                "_total": {"p50": 22.0, "p95": 44.0},
                "render_if_dirty": {"p50": 3.5, "p95": 10.0},
                "update_stream": {"p50": 15.0, "p95": 33.0},
            },
        },
    ])

    assert summary["windows"] == 2
    assert summary["ticks"] == 300
    assert summary["tick_period_ms"] == 33
    # _total is excluded from the ranking, or it would always "win".
    assert summary["worst"] == "update_stream"
    assert summary["stages"]["update_stream"]["max_ms"] == 15.0


def test_tick_profile_is_absent_rather_than_wrong_when_nothing_was_logged() -> None:
    from imu_flight_test_report import summarize_tick_profile

    summary = summarize_tick_profile([])
    assert summary["ticks"] == 0
    assert summary["worst"] is None
