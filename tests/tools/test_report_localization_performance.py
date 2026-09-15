"""Tests for tools/report_localization_performance.py (30 FPS plan step 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import report_localization_performance as rep


def _manifest(**over: object) -> dict:
    base: dict = {
        "mode": "simulated-stream",
        "source": "/nonexistent/video.MP4",
        "mission_snapshot_id": "snap-1",
        "site_profile_sha256": "site-1",
        "asset_sha256": {"map_ply": "abc"},
        "runtime_profile_sha256": "run-1",
        "argv": ["app.py"],
        "source_sha256": "src-1",
    }
    base.update(over)
    return base


def _loc_row(seq: int, t: float, **over: object) -> dict:
    row: dict = {
        "event": "pose_result",
        "display_seq": seq,
        "frame_name": f"frame_{seq:06d}.jpg",
        "ui_arrival_mono": t,
        "t_mono": t + 0.001,
        "success": True,
        "pose": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw_raw": 0.1},
        "direct_status": "FAST_TRACK",
        "inliers": 400,
        "reproj_rms": 0.5,
        "core_wall_ms": 9.0,
        "e2e_submit_to_ui_ms": 22.0,
        "ui_poll_delay_ms": 9.0,
        "source_stamp_age_at_ui_ms": 40.0,
        "client_roundtrip_ms": 10.0,
    }
    row.update(over)
    return row


def _write_session(path: Path, manifest: dict, loc: list[dict], tick=None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "localization.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in loc), encoding="utf-8"
    )
    (path / "telemetry.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (tick or [])), encoding="utf-8"
    )
    return path


def _thirty_hz_session(n=300, span=10.0, t0=1000.0):
    """300 distinct counted rows over exactly `span` seconds."""
    rows = []
    for i in range(n):
        t = t0 + (i * span / (n - 1) if n > 1 else 0.0)
        rows.append(_loc_row(1000 + i, t))
    return rows


def test_mixed_rows_count_fps_and_pass(tmp_path, capsys) -> None:
    loc = _thirty_hz_session()
    # 50 hold_retry rows (own identities, excluded from every FPS).
    for i in range(50):
        loc.append(_loc_row(5000 + i, 1000.5 + i * 0.01, hold_retry=True, hold_kind="boot"))
    # 30 duplicate identities of the first 30 counted rows.
    for i in range(30):
        dup = _loc_row(1000 + i, 1011.0 + i * 0.001)
        loc.append(dup)
    session = _write_session(tmp_path / "s", _manifest(), loc)
    code = rep.main(
        [
            "--session",
            str(session),
            "--window-s",
            "5",
            "--warmup-s",
            "0",
            "--nominal-fps",
            "30",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    rep0 = out["sessions"][0]
    assert rep0["counts"]["hold_retry_rows"] == 50
    assert rep0["counts"]["duplicate_rows"] == 30
    assert rep0["counts"]["counted_rows"] == 300
    assert rep0["fps"]["unique_result_fps"] == pytest.approx(30.0, abs=0.1)
    assert rep0["acceptance"]["pass"] is True


def test_status_and_success_gates_map_confirmed(tmp_path) -> None:
    loc = [
        _loc_row(1, 1000.0, direct_status="VO_ONLY"),
        _loc_row(2, 1000.1, direct_status="DEAD_RECKON"),
        _loc_row(3, 1000.2, direct_status="NO_POSE"),
        _loc_row(4, 1000.3, direct_status="FAST_TRACK"),
        _loc_row(5, 1000.4, direct_status="RELOC_SEED"),
        _loc_row(6, 1000.5, success=False, pose=None, direct_status="FAST_TRACK"),
        _loc_row(7, 1000.6),  # valid FAST_TRACK
    ]
    session = _write_session(tmp_path / "s", _manifest(), loc)
    got = rep.build_session_report(session, window_s=0.2, warmup_s=0.0, nominal_override=30.0)
    span = 0.6
    assert got["fps"]["valid_pose_fps"] == pytest.approx(6 / span)
    # Only the two valid FAST_TRACK/RELOC_SEED rows (seq 4, 5, 7 -> 4,5,7? seq6
    # invalid). seq 4 FAST_TRACK valid, seq 5 RELOC_SEED valid, seq 7 valid.
    assert got["fps"]["map_confirmed_fps"] == pytest.approx(3 / span)
    assert got["quality"]["status_counts"]["VO_ONLY"] == 1
    assert got["quality"]["status_counts"]["DEAD_RECKON"] == 1
    assert got["quality"]["status_counts"]["NO_POSE"] == 1


def test_missing_display_seq_excluded(tmp_path) -> None:
    loc = [_thirty_hz_session(n=10, span=1.0)[i] for i in range(10)]
    bad = _loc_row(999, 1000.5)
    del bad["display_seq"]
    loc.append(bad)
    session = _write_session(tmp_path / "s", _manifest(), loc)
    got = rep.build_session_report(session, window_s=0.2, warmup_s=0.0, nominal_override=30.0)
    assert got["counts"]["missing_identity_rows"] == 1
    assert got["counts"]["counted_rows"] == 10


def test_all_clocks_missing_is_insufficient(tmp_path, capsys) -> None:
    loc = []
    for i in range(10):
        row = _loc_row(1000 + i, 1000.0 + i * 0.1)
        del row["ui_arrival_mono"]
        del row["t_mono"]
        loc.append(row)
    session = _write_session(tmp_path / "s", _manifest(), loc)
    code = rep.main(["--session", str(session), "--nominal-fps", "30"])
    out = json.loads(capsys.readouterr().out)
    assert code == 3
    assert out["sessions"][0]["acceptance"]["pass"] is not True


def test_too_short_after_warmup_is_insufficient(tmp_path, capsys) -> None:
    session = _write_session(tmp_path / "s", _manifest(), _thirty_hz_session(n=30, span=3.0))
    code = rep.main(["--session", str(session), "--nominal-fps", "30"])
    out = json.loads(capsys.readouterr().out)
    assert code == 3
    assert out["sessions"][0]["insufficient"] is not None


def test_slow_session_fails_with_per_window_values(tmp_path, capsys) -> None:
    # ~24/s against a nominal 30 -> evaluated (exit 1), windows listed.
    loc = []
    n = 241
    for i in range(n):
        loc.append(_loc_row(1000 + i, 1000.0 + i / 24.0))
    session = _write_session(tmp_path / "s", _manifest(), loc)
    code = rep.main(
        [
            "--session",
            str(session),
            "--window-s",
            "5",
            "--warmup-s",
            "0",
            "--nominal-fps",
            "30",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    rep0 = out["sessions"][0]
    assert rep0["acceptance"]["pass"] is False
    for window in rep0["windows"]:
        assert window["fps"] == pytest.approx(24.0, abs=0.5)


def test_file_source_semantics_and_live_default(tmp_path) -> None:
    session = _write_session(
        tmp_path / "s", _manifest(mode="real-flight", source=""), _thirty_hz_session(n=5, span=1.0)
    )
    got = rep.build_session_report(session, window_s=60.0, warmup_s=0.0, nominal_override=None)
    assert got["source"]["source_kind"] == "live"
    assert got["source"]["nominal_fps"] == 30.0
    assert got["source"]["nominal_fps_source"] == "live_default"


def test_none_hold_kind_does_not_flag_window(tmp_path) -> None:
    # The logger writes hold_kind="none" on ordinary rows; only a real hold
    # (or LOST mode) may mark a window hold_or_lost.
    loc = _thirty_hz_session(n=120, span=12.0)
    for row in loc:
        row["hold_kind"] = "none"
        row["mode"] = "TRACK"
        row["next_mode"] = "TRACK"
    session = _write_session(tmp_path / "s", _manifest(), loc)
    got = rep.build_session_report(session, window_s=5.0, warmup_s=0.0, nominal_override=10.0)
    assert len(got["windows"]) == 2
    assert all(not w["hold_or_lost"] for w in got["windows"])
    assert got["acceptance"]["pass"] is True


def test_continuity_and_reloc_latency_count_distinct_results_not_cached_values(tmp_path):
    statuses = [
        "NO_POSE",
        "FAST_TRACK",
        "VO_ONLY",
        "VO_ONLY",
        "DEAD_RECKON",
        "RELOC_SEED",
        "NO_POSE",
        "NO_POSE",
    ]
    rows = [
        _loc_row(
            i,
            float(i),
            direct_status=status,
            success=status != "NO_POSE",
            map_constraint_age_s=[None, 0, 1, 2, 3, 0, 1, 2][i],
            reloc_ms=100.0 if i < 5 else 300.0,
            reloc_delivered=i in (1, 5),
        )
        for i, status in enumerate(statuses)
    ]
    rows.append({**rows[3], "ui_arrival_mono": 100.0})
    session = _write_session(tmp_path / "s", _manifest(), rows)
    got = rep.build_session_report(session, window_s=1.0, warmup_s=0, nominal_override=1)
    quality = got["quality"]
    assert quality["map_confirmed_fraction"] == 2 / 8
    assert quality["longest_vo_only_streak_s"] == 2.0
    assert quality["longest_map_unconfirmed_streak_s"] == 3.0
    assert quality["longest_no_pose_streak_s"] == 1.0
    assert quality["max_map_constraint_age_s"] == 3.0
    assert got["latency_ms"]["reloc_ms"]["samples"] == 2
    assert got["latency_ms"]["reloc_ms"]["p50"] == 200.0


def test_all_vo_session_does_not_invent_a_map_fix(tmp_path):
    rows = [_loc_row(i, float(i), direct_status="VO_ONLY") for i in range(4)]
    session = _write_session(tmp_path / "s", _manifest(), rows)
    got = rep.build_session_report(session, window_s=1, warmup_s=0, nominal_override=1)
    assert got["quality"]["map_confirmed_fraction"] == 0
    assert got["quality"]["longest_vo_only_streak_s"] == 3.0
    assert got["quality"]["longest_map_unconfirmed_streak_s"] == 3.0
    assert got["quality"]["max_map_constraint_age_s"] is None
    assert got["quality"]["longest_no_pose_streak_s"] == 0.0
    assert got["latency_ms"]["reloc_ms"]["samples"] == 0
    assert got["latency_ms"]["reloc_ms"]["p95"] is None
