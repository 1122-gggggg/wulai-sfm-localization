"""Pure-function tests for benchmark_esekf_live_replay.

The GPU replay path needs a recorded live flight (onboard video + a session
telemetry.jsonl with fused_odometry events) and is exercised on the trip; here
we pin the telemetry parsing, interpolation, recovery accounting, and verdict.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_VALIDATION = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "benchmark_esekf_live_replay", _VALIDATION / "benchmark_esekf_live_replay.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_load_velocity_track_prefers_fused(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _write_jsonl(
        session / "telemetry.jsonl",
        [
            {
                "event": "readback",
                "t_mono_ns": 0,
                "speed_north_mps": 9.0,
                "speed_east_mps": 9.0,
                "speed_down_mps": 9.0,
            },
            {
                "event": "fused_odometry",
                "t_mono_ns": 1_000_000_000,
                "speed_north_mps": 1.0,
                "speed_east_mps": 2.0,
                "speed_down_mps": 3.0,
            },
            {
                "event": "fused_odometry",
                "t_mono_ns": 1_100_000_000,
                "speed_north_mps": 1.5,
                "speed_east_mps": 2.5,
                "speed_down_mps": 3.5,
            },
        ],
    )
    t_rel, vel, kind = m.load_velocity_track(session)
    assert kind == "fused_odometry"
    assert t_rel == pytest.approx([0.0, 0.1])
    assert vel[0].tolist() == [1.0, 2.0, 3.0]


def test_load_velocity_track_falls_back_to_readback(tmp_path):
    jsonl = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "event": "readback",
                "t_mono_ns": 5_000_000_000,
                "speed_north_mps": 0.2,
                "speed_east_mps": 0.1,
                "speed_down_mps": 0.0,
            },
            {"event": "sim_state", "t_mono_ns": 6_000_000_000},
        ],
    )
    t_rel, vel, kind = m.load_velocity_track(jsonl)
    assert kind == "readback"
    assert len(t_rel) == 1


def test_load_velocity_track_rejects_no_velocity(tmp_path):
    jsonl = tmp_path / "telemetry.jsonl"
    _write_jsonl(jsonl, [{"event": "sim_state", "t_mono_ns": 1, "pose": [0, 0, 0]}])
    with pytest.raises(SystemExit):
        m.load_velocity_track(jsonl)


def test_velocity_at_interpolates_and_gaps():
    t_rel = [0.0, 1.0, 2.0]
    vel = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [2.0, 4.0, 6.0]])
    mid = m.velocity_at(t_rel, vel, 0.5, max_gap_s=2.0)
    assert mid.tolist() == [1.0, 2.0, 3.0]
    assert m.velocity_at(t_rel, vel, 9.0, max_gap_s=2.0) is None
    edge = m.velocity_at(t_rel, vel, 2.5, max_gap_s=1.0)
    assert edge.tolist() == [2.0, 4.0, 6.0]


def test_recovery_stats_counts_runs_and_recoveries():
    rows = [
        {"next_mode": "TRACK"},
        {"next_mode": "LOST"},
        {"next_mode": "LOST"},
        {"next_mode": "TRACK"},  # recovered after a 2-frame run
        {"next_mode": "LOST"},
        {"next_mode": "WEAK_TRACK"},  # run ended without a TRACK recovery
    ]
    stats = m._recovery_stats(rows)
    assert stats["lost_runs"] == 2
    assert stats["longest_lost_run"] == 2
    assert stats["recoveries_to_track"] == 1
    assert stats["mean_frames_to_recover"] == 2.0


def test_summarize_shape():
    rows = [
        {
            "success": True,
            "next_mode": "TRACK",
            "wall_ms": 20.0,
            "prediction_allowed": False,
            "prediction_source": None,
            "esekf_update_accepted": None,
        },
        {
            "success": False,
            "next_mode": "LOST",
            "wall_ms": 40.0,
            "prediction_allowed": True,
            "prediction_source": "esekf",
            "esekf_update_accepted": True,
        },
    ]
    s = m.summarize(rows)
    assert s["frames"] == 2
    assert s["successes"] == 1
    assert s["state_counts"] == {"TRACK": 1, "LOST": 1}
    assert s["esekf_prediction_allowed_frames"] == 1
    assert s["esekf_superseded_frames"] == 1
    assert s["esekf_update_accepted_frames"] == 1


def _variant(frames_fed, pred_allowed, successes, lost, p95):
    return {
        "site_profile": "x",
        "site_profile_sha256": "abc123abc123",
        "video": "v",
        "video_sha256": "def456def456",
        "session": "s",
        "telemetry_samples": 100,
        "telemetry_event_kind": "fused_odometry",
        "telemetry_span_s": 120.0,
        "telemetry_offset_s": 0.0,
        "telemetry_scale": 1.0,
        "max_gap_s": 2.0,
        "frames_fed_velocity": frames_fed,
        "stride": 3,
        "pnp_random_seed": 0,
        "device": "cuda",
        "summary": {
            "frames": 200,
            "successes": successes,
            "success_rate": successes / 200.0,
            "state_counts": {"TRACK": successes, "LOST": lost},
            "wall_ms": {"p50": 22.0, "p95": p95},
            "recovery": {
                "lost_runs": 3,
                "longest_lost_run": 10,
                "recoveries_to_track": 2,
                "mean_frames_to_recover": 5.0,
            },
            "esekf_prediction_allowed_frames": pred_allowed,
            "esekf_superseded_frames": 0,
            "esekf_update_accepted_frames": 0,
        },
    }


def test_verdict_invalid_when_no_velocity_fed():
    md = m.compare_markdown(_variant(0, 0, 150, 20, 80.0), _variant(0, 0, 150, 20, 80.0))
    assert "INVALID" in md


def test_verdict_dormant_when_never_armed():
    md = m.compare_markdown(_variant(180, 0, 150, 20, 80.0), _variant(0, 0, 150, 20, 80.0))
    assert "DORMANT" in md


def test_verdict_helps_and_regresses():
    helps = m.compare_markdown(_variant(180, 40, 160, 15, 82.0), _variant(0, 0, 150, 20, 80.0))
    assert "ESEKF HELPS" in helps
    regress = m.compare_markdown(_variant(180, 40, 140, 25, 95.0), _variant(0, 0, 150, 20, 80.0))
    assert "REGRESSES" in regress
