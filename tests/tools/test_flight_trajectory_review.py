import csv
import json

from flight_trajectory_review import build_replay, export_review, segment_error


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def plan(run="a", t=1):
    return {
        "event": "auto_route_plan",
        "auto_run_id": run,
        "t": t,
        "waypoints_u": [[0, 0, 0], [10, 0, 0], [10, 10, 0]],
        "target_index": 0,
        "drawn_waypoint_count": 3,
        "control_config": {"map_frame": {"east": [1, 0, 0], "north": [0, 1, 0], "up": [0, 0, 1]}},
    }


def tick(run="a", t=2, **fields):
    return {
        "event": "auto_route_tick",
        "auto_run_id": run,
        "t": t,
        "pose_u": [5, 3, 4],
        "target_index": 1,
        "pose_age": 0.1,
        "map_pose_confirmed": True,
        **fields,
    }


def test_known_segment_distance_and_degenerate_segment():
    assert segment_error([5, 3, 4], [0, 0, 0], [10, 0, 0]) == 5
    assert segment_error([13, 4, 0], [0, 0, 0], [10, 0, 0]) == 5
    assert segment_error([3, 4, 0], [0, 0, 0], [0, 0, 0]) == 5


def test_archive_survives_without_diagnostics_and_splits_runs(tmp_path):
    write_rows(
        tmp_path / "trajectory.jsonl",
        [plan(), tick(), plan("b", 10), tick("b", 11, pose_u=[5, 0, 0])],
    )
    result = build_replay(tmp_path)
    assert [r["id"] for r in result["runs"]] == ["a", "b"]
    assert result["runs"][0]["stats"]["mean_u"] == 5
    assert result["runs"][1]["stats"]["max_u"] == 0
    assert result["runs"][0]["axes"] == ["East", "North", "Up"]


def test_missing_stale_and_prediction_labels_are_preserved(tmp_path):
    write_rows(
        tmp_path / "localization.jsonl",
        [
            plan(),
            tick(pose_age=2),
            tick(t=3, pose_u=None),
            tick(t=4, pose_u=[float("nan"), 0, 0]),
            tick(t=5, position_observed=False, map_pose_confirmed=False),
            tick(t=6, map_pose_confirmed=False),
        ],
    )
    run = build_replay(tmp_path)["runs"][0]
    assert run["stats"]["missing"] == 3
    assert run["stats"]["samples"] == 2
    assert [s["quality"] for s in run["samples"][-2:]] == ["IMU／預測", "弱定位／未確認"]


def test_boot_only_auto_retains_estimates_but_has_no_route_error(tmp_path):
    write_rows(
        tmp_path / "trajectory.jsonl",
        [
            {"event": "autonomy_event", "auto_run_id": "boot", "t": 1, "kind": "boot_hover"},
            {
                "event": "pose_result",
                "t": 2,
                "pose": {"x": 1, "y": 2, "z": 3},
                "direct_status": "IMU_BRIDGE",
            },
            {"event": "pose_result", "t": 3, "pose": None, "success": False},
            {"event": "autonomy_event", "auto_run_id": "boot", "t": 4, "kind": "auto_failed"},
        ],
    )
    run = build_replay(tmp_path)["runs"][0]
    assert len(run["samples"]) == 2
    assert run["samples"][0]["quality"] == "IMU／預測"
    assert run["stats"]["mean_u"] is None


def test_export_csv_and_html_escape_session_data(tmp_path):
    write_rows(
        tmp_path / "trajectory.jsonl",
        [plan("</script><script>alert(1)</script>"), tick("</script><script>alert(1)</script>")],
    )
    output = tmp_path / "review"
    export_review(tmp_path, output)
    rows = list(csv.DictReader((output / "trajectory.csv").open(encoding="utf-8-sig")))
    assert float(rows[0]["active_segment_error_u"]) == 5
    html = (output / "index.html").read_text()
    assert "</script><script>alert(1)</script>" not in html
    assert "__TRAJECTORY_DATA__" not in html
    assert json.loads((output / "trajectory.json").read_text())["runs"]


def test_empty_session_exports_explicit_empty_view(tmp_path):
    assert export_review(tmp_path, tmp_path / "out")["runs"] == []
