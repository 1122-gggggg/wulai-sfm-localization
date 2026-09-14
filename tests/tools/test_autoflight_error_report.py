"""The AUTO error debrief counts trajectory error only after route join."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import autoflight_error_report as report


def _write_session(tmp_path: Path, rows: list[dict]) -> Path:
    session = tmp_path / "session_20260913T000000Z_real-flight_test1234"
    session.mkdir(parents=True)
    (session / "localization.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return session


def _plan() -> dict:
    return {
        "event": "auto_route_plan",
        "waypoints_u": [[0.2, 0.0, -1.8], [0.5, 0.0, -1.5]],
        "drawn_waypoint_count": 2,
        "return_to_start": False,
        "waypoint_arrive_radius_u": 0.2,
        "target_index": 1,
    }


def _tick(step: int, *, target: int | None, distance: float | None) -> dict:
    row: dict = {
        "event": "auto_route_tick",
        "t": 100.0 + step * 0.1,
        "step": step,
        "action": "FOLLOW" if target is not None else None,
        "pcmd_phase": "translate" if target is not None else None,
        "target_index": target,
        "progress": 0.1 + step * 0.01,
        "pcmd": [0, 1, 0, 0],
        "blocked": False,
        "reason": "FOLLOW",
    }
    if distance is not None:
        row["route_distance_u"] = distance
    return row


def test_join_boundary_excludes_approach_wander_from_error(tmp_path: Path) -> None:
    rows = [_plan()]
    # Join phase: far off route, must not leak into the error statistics.
    rows += [_tick(step, target=1, distance=0.6 - step * 0.05) for step in range(5)]
    # Joined at step 5; post-join error is a known 0.10/0.12 alternating pair.
    rows += [
        _tick(5 + step, target=2, distance=0.10 if step % 2 == 0 else 0.12)
        for step in range(6)
    ]
    result = report.analyze_session(_write_session(tmp_path, rows))

    assert result["verdict"] == "JOINED"
    assert result["join"]["joined"] is True
    assert result["join"]["time_to_join_s"] == pytest.approx(0.5)
    post = result["post_join"]
    assert post["samples"] == 6
    assert post["cross_track_u"]["mean"] == 0.11
    assert post["cross_track_u"]["max"] == 0.12
    assert post["cross_track_u"]["p95"] == 0.12
    assert result["arrivals"] == [{"target_index": 2, "t": 100.5}]


def test_never_joined_reports_closest_approach_without_error(tmp_path: Path) -> None:
    rows = [_plan()]
    rows += [_tick(step, target=1, distance=0.5 - step * 0.05) for step in range(5)]
    result = report.analyze_session(_write_session(tmp_path, rows))

    assert result["verdict"] == "NEVER_JOINED"
    assert result["join"]["closest_approach_u"] == 0.3
    assert "post_join" not in result


def test_session_without_auto_is_explicit(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [{"event": "pose_result", "mode": "TRACK"}])
    result = report.analyze_session(session)

    assert result["verdict"] == "NO_AUTO"
    assert result["has_auto"] is False


def test_post_join_vo_share_attributes_ticks_to_poses(tmp_path: Path) -> None:
    rows = [_plan()]
    rows += [_tick(step, target=1, distance=0.5) for step in range(2)]
    rows += [_tick(2 + step, target=2, distance=0.1) for step in range(4)]
    poses = [
        {"event": "pose_result", "success": True, "mode": "TRACK",
         "direct_status": status, "t_mono_ns": int(stamp * 1e9)}
        for stamp, status in (
            (100.0, "FAST_TRACK"),
            (100.2, "FAST_TRACK"),
            (100.3, "VO_ONLY"),
            (100.5, "VO_ONLY"),
        )
    ]
    result = report.analyze_session(_write_session(tmp_path, rows + poses))

    assert result["verdict"] == "JOINED"
    assert result["post_join"]["vo_share"] == 0.75


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("AUTO speed guard latched -> HOVER", "guard_hover"),
        ("AUTO speed limit 0.3 m/s reached -> HOVER", "guard_hover"),
        ("AUTO ground speed unavailable or stale -> HOVER", "guard_hover"),
        ("no fresh pose -> hover", "search_lost"),
        ("AUTO paused", "hover_other"),
    ],
)
def test_blocked_commands_override_the_planned_phase(reason, expected):
    tick = {"pcmd_phase": "translate", "pcmd": [0, 0, 0, 0],
            "blocked": True, "reason": reason}
    assert report._phase_bucket(tick) == expected


def test_accepted_speed_guard_command_is_still_translation():
    tick = {"pcmd_phase": "translate", "pcmd": [0, 1, 0, 0],
            "blocked": False, "reason": "desktop AUTO fresh-speed guard + command cap"}
    assert report._phase_bucket(tick) == "translate"


def test_phase_durations_accumulate_before_rounding_at_20hz(tmp_path):
    rows = [_plan(), _tick(0, target=0, distance=0.5)]
    for index in range(21):
        tick = _tick(index, target=1, distance=0.1)
        tick.update(t=101.0 + index * 0.05, blocked=True,
                    reason="AUTO speed limit reached -> HOVER", pcmd=[0, 0, 0, 0])
        rows.append(tick)
    result = report.analyze_session(_write_session(tmp_path, rows))
    assert result["post_join"]["phase_seconds"] == {"guard_hover": 1.0}
    assert result["post_join"]["blocked_seconds"] == 1.0

def test_two_auto_runs_do_not_share_join_timing_or_error(tmp_path: Path) -> None:
    first = [_plan()]
    first += [_tick(step, target=0, distance=0.6 - step * 0.05) for step in range(3)]
    first += [_tick(3 + step, target=1, distance=0.1) for step in range(3)]
    second = [_plan()]
    second += [_tick(10 + step, target=0, distance=0.9 - step * 0.05) for step in range(3)]
    second += [_tick(13 + step, target=1, distance=0.2) for step in range(3)]
    # Simulate a manual gap: the second run restarts the clock and steps.
    for tick in second[1:]:
        tick["t"] += 100.0
    result = report.analyze_session(_write_session(tmp_path, first + second))

    assert len(result["auto_runs"]) == 2
    assert result["auto_runs"][0]["join"]["time_to_join_s"] == 0.3
    assert result["auto_runs"][1]["join"]["time_to_join_s"] == 0.3
    assert result["auto_runs"][0]["post_join"]["cross_track_u"]["mean"] == 0.1
    assert result["auto_runs"][1]["post_join"]["cross_track_u"]["mean"] == 0.2
    # Legacy top-level shape still exposes the latest run.
    assert result["join"]["time_to_join_s"] == 0.3
    assert result["verdict"] == "JOINED"


def test_runs_split_by_auto_run_id_even_without_a_time_gap(tmp_path: Path) -> None:
    rows = [_plan()]
    rows[0]["auto_run_id"] = "run-a"
    rows += [_tick(step, target=0, distance=0.5) for step in range(2)]
    for tick in rows[1:]:
        tick["auto_run_id"] = "run-a"
    second = [_plan()]
    second[0]["auto_run_id"] = "run-b"
    second += [_tick(10 + step, target=0, distance=0.2) for step in range(2)]
    second += [_tick(12, target=1, distance=0.2)]
    for tick in second[1:]:
        tick["auto_run_id"] = "run-b"
        tick["t"] += 0.2
    result = report.analyze_session(_write_session(tmp_path, rows + second))

    assert [run["auto_run_id"] for run in result["auto_runs"]] == ["run-a", "run-b"]
    assert result["auto_runs"][0]["verdict"] == "NEVER_JOINED"
    assert result["auto_runs"][1]["verdict"] == "JOINED"
