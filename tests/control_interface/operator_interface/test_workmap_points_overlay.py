"""Session landmarks stream into the map-pane overlay without touching the base cloud."""
from types import SimpleNamespace

import numpy as np

import operator_tick


def test_worker_delta_merges_by_id_and_throttles_redraws(monkeypatch):
    monkeypatch.setattr(operator_tick.time, "monotonic", lambda: 10.0)
    app = SimpleNamespace()
    delta = {"ids": [-1, -2], "xyz": [[1, 2, 3], [4, 5, 6]]}
    operator_tick._ingest_workmap_points(app, {"workmap_points": delta})
    assert app.workmap_points_version == 1
    operator_tick._ingest_workmap_points(
        app, {"workmap_points": {"ids": [-2], "xyz": [[4, 5, 7]]}}
    )
    np.testing.assert_array_equal(app.workmap_points[-2], [4, 5, 7])
    assert len(app.workmap_points) == 2
    assert app.workmap_points_version == 1


def test_malformed_delta_is_ignored():
    app = SimpleNamespace()
    for bad in ({"ids": [1], "xyz": [[1, 2]]}, {"ids": [1], "xyz": [[np.nan, 0, 0]]},
                {"ids": [1, 2], "xyz": [[1, 2, 3]]}, "x", {}):
        operator_tick._ingest_workmap_points(app, {"workmap_points": bad})
    operator_tick._ingest_workmap_points(app, "not a dict")
    assert not getattr(app, "workmap_points", {})


def test_worker_colours_are_kept_per_point_with_a_fallback_for_missing_ones():
    app = SimpleNamespace()
    delta = {"ids": [-1, -2, -3], "xyz": [[1, 2, 3]] * 3, "rgb": [[200, 100, 50], None, [999, -5, 7]]}
    operator_tick._ingest_workmap_points(app, {"workmap_points": delta})
    np.testing.assert_array_equal(app.workmap_point_rgb[-1], [200, 100, 50])
    assert -2 not in app.workmap_point_rgb
    np.testing.assert_array_equal(app.workmap_point_rgb[-3], [255, 0, 7])


def test_aligned_submap_trajectory_replaces_one_segment_without_linking_runs():
    app = SimpleNamespace()
    first_id = (1 << 32) | 7
    second_id = (2 << 32) | 9
    first = {"segment_id": first_id, "points": [[2, 1, 2, 3], [3, 4, 5, 6]]}
    operator_tick._ingest_workmap_trajectory(app, {"workmap_trajectory_backfill": first})
    assert app.workmap_trajectories == {first_id: [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]}
    operator_tick._ingest_workmap_trajectory(
        app, {"workmap_trajectory_backfill": {
            "segment_id": second_id, "points": [[4, 10, 2, 3], [5, 12, 2, 3]],
        }},
    )
    operator_tick._ingest_workmap_trajectory(
        app, {"workmap_trajectory_backfill": {
            "segment_id": first_id, "points": [[2, -1, 2, 3], [3, -4, 5, 6]],
        }},
    )
    assert app.workmap_trajectory_version == 3
    assert app.workmap_trajectories == {
        first_id: [(-1.0, 2.0, 3.0), (-4.0, 5.0, 6.0)],
        second_id: [(10.0, 2.0, 3.0), (12.0, 2.0, 3.0)],
    }


def test_invalid_or_unanchored_trajectory_is_ignored_atomically():
    app = SimpleNamespace()
    good = {"segment_id": 2, "points": [[1, 0, 0, 0], [2, 1, 0, 0]]}
    operator_tick._ingest_workmap_trajectory(app, {"workmap_trajectory_backfill": good})
    original = dict(app.workmap_trajectories)
    for bad in (
        {"segment_id": 2, "points": [[1, 1, 1, 1], [1, 2, 2, 2]]},
        {"segment_id": 2, "points": [[2, 1, 1, 1], [1, 2, 2, 2]]},
        {"segment_id": 2, "points": [[1, 1, 1, 1], [2, np.nan, 2, 2]]},
        {"segment_id": 2, "points": [[1, 1_000_001, 0, 0]]},
        {"segment_id": 2, "points": [[1, 0, 0, 0]] * 301},
        {"segment_id": "2", "points": [[1, 0, 0, 0]]},
        {"segment_id": 2, "points": [[-1, 0, 0, 0]]},
        {"segment_id": 2, "points": []},
        {"segment_id": -1, "points": [[1, 0, 0, 0]]},
        {"segment_id": 1 << 63, "points": [[1, 0, 0, 0]]},
    ):
        operator_tick._ingest_workmap_trajectory(app, {"workmap_trajectory_backfill": bad})
    operator_tick._ingest_workmap_trajectory(app, {"unanchored_submap": good})
    assert app.workmap_trajectories == original
    assert app.workmap_trajectory_version == 1


def test_session_identity_change_discards_old_points_colours_and_backfill():
    app = SimpleNamespace(live_pose=np.array([1, 2, 3, 4]), history=[(1, 2, 3)])
    identity = {"workmap_session_id": "one", "workmap_session_dir": "/sessions/one"}
    operator_tick._ingest_workmap_session(app, identity)
    operator_tick._ingest_workmap_points(app, {"workmap_points": {
        "ids": [-1], "xyz": [[4, 5, 6]], "rgb": [[20, 30, 40]],
    }})
    operator_tick._ingest_workmap_trajectory(app, {"workmap_trajectory_backfill": {
        "segment_id": 2, "points": [[1, 0, 0, 0], [2, 1, 0, 0]],
    }})
    points_version = app.workmap_points_version
    trail_version = app.workmap_trajectory_version
    operator_tick._ingest_workmap_session(app, identity)
    assert -1 in app.workmap_points
    operator_tick._ingest_workmap_session(app, {
        "workmap_session_id": "two", "workmap_session_dir": "/sessions/two",
        "workmap_mapping_healthy": False, "workmap_mapping_error": "writer failed",
    })
    assert app.workmap_points == app.workmap_point_rgb == app.workmap_trajectories == {}
    assert app.workmap_points_version > points_version
    assert app.workmap_trajectory_version > trail_version
    assert app.workmap_mapping_diagnostics["workmap_mapping_healthy"] is False
    np.testing.assert_array_equal(app.live_pose, [1, 2, 3, 4])
    assert app.history == [(1, 2, 3)]


def test_last_throttled_point_update_eventually_invalidates_map(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(operator_tick.time, "monotonic", lambda: clock[0])
    app = SimpleNamespace(
        live_pose=np.array([0, 0, 0, 0]), camera_axes_world=None,
        camera_forward_world=None, history=[], route_pts=[], no_loc_markers=[],
        map_base_key=lambda width, height: (width, height, app.workmap_points_version),
    )
    delta = {"workmap_points": {"ids": [-1], "xyz": [[1, 2, 3]]}}
    operator_tick._ingest_workmap_points(app, delta)
    state = SimpleNamespace(pose=app.live_pose, gimbal_pitch_deg=None)
    first = operator_tick._map_dirty_key(app, state, width=600, height=400)
    clock[0] = 10.5
    operator_tick._ingest_workmap_points(app, {
        "workmap_points": {"ids": [-1], "xyz": [[4, 5, 6]]},
    })
    assert operator_tick._map_dirty_key(app, state, width=600, height=400) == first
    clock[0] = 11.1
    assert operator_tick._map_dirty_key(app, state, width=600, height=400) != first
    assert app._workmap_points_pending is False


def test_point_overlay_renders_worker_rgb_and_fallback_without_mutating_base_cloud():
    from flight_operator_app import OperatorApp, WORKMAP_POINT_RGB

    width, height = 320, 240
    scale = min(width, height) * 0.46 / 10.0
    base_cloud = np.empty((0, 6))
    app = SimpleNamespace(
        map_points=base_cloud, map_zoom=1.0, map_radius=10.0,
        map_pan=np.zeros(2), map_center=np.zeros(3), pivot_point=np.zeros(3),
        map_yaw=0.0, map_pitch=0.0, map_roll=0.0,
        _map_detail_points=lambda: 1000,
        _map_axis_basis=lambda: (
            np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), True,
        ),
        project_world=lambda point, w, h: (
            int(w * 0.5 + point[0] * scale), int(h * 0.5 - point[1] * scale),
        ),
    )
    baseline = np.asarray(OperatorApp.render_map_base(app, width, height))
    operator_tick._ingest_workmap_points(app, {"workmap_points": {
        "ids": [-1, -2], "xyz": [[3, 3, 0], [-3, 3, 0]],
        "rgb": [[20, 30, 40], None],
    }})
    overlay = np.asarray(OperatorApp.render_map_base(app, width, height))
    for point, color in (([3, 3, 0], [20, 30, 40]), ([-3, 3, 0], WORKMAP_POINT_RGB)):
        x, y = app.project_world(point, width, height)
        np.testing.assert_array_equal(overlay[y:y + 2, x:x + 2], np.tile(color, (2, 2, 1)))
        assert not np.array_equal(baseline[y:y + 2, x:x + 2], overlay[y:y + 2, x:x + 2])
    assert app.map_points is base_cloud
    assert len(base_cloud) == 0
