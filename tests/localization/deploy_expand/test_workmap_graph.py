"""Incremental visual factors, immutable anchors, and session publication."""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("gtsam")

from workmap_graph import IncrementalWorkMapGraph
from workmap_ba import solve_ba_snapshot
from test_workmap_ba_solver import _scene
from test_workmap_ba_integration import _scene as _session_scene


def _eight_views():
    payload, old, fresh = _scene()
    points = np.concatenate((old, fresh))
    for index in range(4, 8):
        pose = np.column_stack((np.eye(3), [-0.12 * index, 0.0, 0.0]))
        camera = points @ pose[:, :3].T + pose[:, 3]
        projected = camera @ payload["K"].T
        estimate = pose.copy()
        estimate[:, 3] += [0.015, -0.01, 0.003]
        payload["keyframes"].append(
            {
                "name": f"kf-{index}",
                "ordinal": index * 3,
                "pose": estimate,
                "pose_revision": 10 + index,
                "fixed": False,
                "ids": payload["keyframes"][0]["ids"].copy(),
                "xy": projected[:, :2] / projected[:, 2, None],
            }
        )
    return payload, fresh


def test_incremental_graph_keeps_earlier_observations_without_duplicating_factors():
    payload, truth = _eight_views()
    original = copy.deepcopy(payload)
    graph = IncrementalWorkMapGraph()
    first = graph.solve({**payload, "keyframes": payload["keyframes"][:4]})
    assert first["ok"], first["reason"]
    result = graph.solve(payload)
    assert result["ok"], result["reason"]
    assert not result["graph_rebuilt"]
    assert result["graph_keyframes"] == 8
    assert result["graph_observations"] == 8 * len(payload["keyframes"][0]["ids"])
    assert result["graph_new_observations"] == 4 * len(payload["keyframes"][0]["ids"])
    assert result["after_cost"] < result["before_cost"] * 0.05
    assert np.allclose(
        np.array([result["updates"][-i - 1] for i in range(len(truth))]), truth, atol=0.002
    )
    repeated = graph.solve(payload)
    assert repeated["graph_new_observations"] == 0
    assert repeated["graph_observations"] == result["graph_observations"]
    assert set(result["updates"]) == set(payload["new_landmarks"])
    for key, value in original["old_landmarks"].items():
        assert np.array_equal(payload["old_landmarks"][key], value)
    assert np.array_equal(result["poses"]["kf-0"], payload["keyframes"][0]["pose"])


def test_graph_fixes_scale_without_old_points_and_rejects_degenerate_gauge():
    payload, _old, _fresh = _scene(old_count=0, new_count=12, no_old=True)
    result = IncrementalWorkMapGraph().solve(payload)
    assert result["ok"], result["reason"]
    for frame in payload["keyframes"][:2]:
        assert np.array_equal(result["poses"][frame["name"]], frame["pose"])
    payload["keyframes"][1]["pose"] = payload["keyframes"][0]["pose"].copy()
    invalid = IncrementalWorkMapGraph().solve(payload)
    assert not invalid["ok"]
    assert "baseline" in invalid["reason"]


def test_graph_rebuilds_after_external_geometry_change_or_epoch_reset():
    payload, _ = _eight_views()
    graph = IncrementalWorkMapGraph()
    assert graph.solve(payload)["ok"]
    result = graph.solve({**payload, "solver_generation": 1})
    assert result["ok"] and result["graph_rebuilt"]
    result = graph.solve({**payload, "source_epoch": 8})
    assert result["ok"] and result["graph_rebuilt"]
    damaged = copy.deepcopy(payload)
    damaged["new_landmarks"][-1][2] = -1.0
    assert not graph.solve(damaged)["ok"]


def test_anchored_keyframes_append_without_rebuilding_constant_factors():
    payload, _ = _eight_views()
    for frame in payload["keyframes"]:
        frame["fixed_pose"] = True
    graph = IncrementalWorkMapGraph()
    first = graph.solve({**payload, "keyframes": payload["keyframes"][:4]})
    assert first["ok"], first["reason"]
    result = graph.solve(payload)
    assert result["ok"], result["reason"]
    assert not result["graph_rebuilt"]
    assert result["graph_factors"] < result["graph_observations"]
    for frame in payload["keyframes"]:
        assert np.array_equal(result["poses"][frame["name"]], frame["pose"])


def test_history_outside_local_ba_window_is_actually_corrected():
    payload, _ = _eight_views()
    window = [payload["keyframes"][i] for i in (0, 5, 6, 7)]
    local = solve_ba_snapshot({**payload, "keyframes": window})
    whole = IncrementalWorkMapGraph().solve(payload)
    assert local["ok"] and whole["ok"]
    before, after = [], []
    for index in range(1, 5):
        name = f"kf-{index}"
        truth = np.array([-0.12 * index, 0.0, 0.0])
        assert name not in local["poses"]
        before.append(np.linalg.norm(payload["keyframes"][index]["pose"][:, 3] - truth))
        after.append(np.linalg.norm(np.asarray(whole["poses"][name])[:, 3] - truth))
    assert np.mean(after) < np.mean(before) * 0.1


def test_session_accepts_graph_beyond_four_views_and_rejects_stale_results(tmp_path):
    session, old, append_frame, K = _session_scene(tmp_path)
    try:
        for index in (5, 6, 7):
            append_frame(index)
        payload = session.ba_snapshot(0, K, incremental=True)
        assert len(payload["keyframes"]) > 4
        result = IncrementalWorkMapGraph().solve({**payload, "freeze_poses": True})
        assert result["ok"], result["reason"]
        assert session.apply_ba_candidate(payload, result)
        assert not session.apply_ba_candidate(payload, result)
        for key, value in payload["old_landmarks"].items():
            assert np.array_equal(old[key - 1], value)
    finally:
        session.close()


def test_vector_snapshot_is_detached_and_checks_all_frozen_observations(tmp_path):
    session, _old, _append, K = _session_scene(tmp_path)
    try:
        payload = session.ba_snapshot(0, K, incremental=True, anchor_min_inliers=40)
        last = payload["keyframes"][-1]
        pixels = last["xy"].copy()
        frame = session.keyframes[last["name"]]
        frame["ba_xy"] += 0.01
        assert np.array_equal(last["xy"], pixels)
        row = np.flatnonzero(frame["ba_ids"] > 0)[0]
        frame["ba_xyz"][row] += [0.0, 0.0, 0.1]
        assert session.ba_snapshot(0, K, incremental=True) is None
        candidate = IncrementalWorkMapGraph().solve(payload)
        assert candidate["ok"], candidate["reason"]
        assert not session.apply_ba_candidate(payload, candidate)
        assert session.last_ba_rejection_reason == "frozen_landmark_changed"
    finally:
        session.close()
