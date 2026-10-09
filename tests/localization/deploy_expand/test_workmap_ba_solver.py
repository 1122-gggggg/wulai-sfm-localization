from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DIR = ROOT / "定位演算法" / "deploy_code_expand" / "sfm_direct_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from workmap_ba import solve_ba_snapshot  # noqa: E402


def _scene(
    *,
    old_count: int = 24,
    new_count: int = 6,
    no_old: bool = False,
    keyframes_count: int = 4,
    fixed: frozenset[int] | None = None,
):
    fixed = frozenset({0, 1} if no_old else {0}) if fixed is None else fixed
    rng = np.random.default_rng(20260926)
    K = np.array([[540.0, 0.0, 320.0], [0.0, 540.0, 240.0], [0.0, 0.0, 1.0]])
    old_xyz = np.column_stack(
        (
            rng.uniform(-0.7, 0.7, old_count),
            rng.uniform(-0.45, 0.45, old_count),
            rng.uniform(4.0, 6.0, old_count),
        )
    )
    true_new = np.column_stack(
        (
            rng.uniform(-0.55, 0.55, new_count),
            rng.uniform(-0.35, 0.35, new_count),
            rng.uniform(4.1, 5.8, new_count),
        )
    )
    initial_new = true_new + rng.uniform(-0.012, 0.012, true_new.shape)
    old_landmarks = {index + 1: point.copy() for index, point in enumerate(old_xyz)}
    new_landmarks = {-(index + 1): point.copy() for index, point in enumerate(initial_new)}
    keyframes = []
    all_true_xyz = np.concatenate((old_xyz, true_new)) if old_count else true_new
    ids = np.concatenate(
        (
            np.arange(1, old_count + 1, dtype=np.int64),
            -np.arange(1, new_count + 1, dtype=np.int64),
        )
    )

    for index in range(keyframes_count):
        center_x = 0.12 * index
        true_pose = np.column_stack((np.eye(3), [-center_x, 0.0, 0.0]))
        camera_xyz = all_true_xyz @ true_pose[:, :3].T + true_pose[:, 3]
        projected = camera_xyz @ K.T
        xy = projected[:, :2] / projected[:, 2, None]
        initial_pose = true_pose.copy()
        if index not in fixed:
            initial_pose[:, 3] += np.array([0.003, -0.004, 0.001]) * index
        keyframes.append(
            {
                "name": f"kf-{index}",
                "ordinal": index * 3,
                "pose": initial_pose,
                "pose_revision": 10 + index,
                "ids": ids.copy(),
                "xy": xy,
                "fixed": index in fixed,
            }
        )

    payload = {
        "session_id": "solver-test-session",
        "source_epoch": 7,
        "snapshot_revision": 23,
        "K": K,
        "keyframes": keyframes,
        "old_landmarks": old_landmarks,
        "new_landmarks": new_landmarks,
        "landmark_revisions": {-(index + 1): 30 + index for index in range(new_count)},
    }
    return payload, old_xyz, true_new


def _reprojection_rmse(payload, old_xyz, true_new, *, result=None):
    old_points = {index + 1: point for index, point in enumerate(old_xyz)}
    new_points = {-(index + 1): point for index, point in enumerate(true_new)}
    total_squared = 0.0
    count = 0
    K = np.asarray(payload["K"], dtype=np.float64)
    for frame in payload["keyframes"]:
        pose = (
            np.asarray(result["poses"][frame["name"]], dtype=np.float64)
            if result is not None
            else np.asarray(frame["pose"], dtype=np.float64)
        )
        for landmark_id, observed in zip(frame["ids"], frame["xy"]):
            key = int(landmark_id)
            point = old_points[key] if key > 0 else new_points[key]
            camera_xyz = pose[:, :3] @ point + pose[:, 3]
            projected = K @ camera_xyz
            error = projected[:2] / projected[2] - observed
            total_squared += float(error @ error)
            count += 1
    return float(np.sqrt(total_squared / count))


def test_real_scipy_ba_reduces_reprojection_error_without_mutating_frozen_points():
    payload, old_xyz, true_new = _scene()
    old_before = {
        key: np.asarray(value).tobytes() for key, value in payload["old_landmarks"].items()
    }
    result = solve_ba_snapshot(payload)

    assert result["ok"], result["reason"]
    assert result["session_id"] == payload["session_id"]
    assert result["source_epoch"] == payload["source_epoch"]
    assert result["snapshot_revision"] == payload["snapshot_revision"]
    assert result["keyframe_revisions"] == {
        frame["name"]: frame["pose_revision"] for frame in payload["keyframes"]
    }
    assert result["landmark_revisions"] == payload["landmark_revisions"]
    assert result["before_cost"] > result["after_cost"]
    assert _reprojection_rmse(payload, old_xyz, true_new) > _reprojection_rmse(
        payload, old_xyz, true_new, result=result
    )
    assert set(result["updates"]) == set(payload["new_landmarks"])
    assert not set(result["updates"]) & set(payload["old_landmarks"])
    assert {
        key: np.asarray(value).tobytes() for key, value in payload["old_landmarks"].items()
    } == old_before


def test_no_old_support_fixes_first_and_nonzero_baseline_second_keyframes():
    payload, old_xyz, true_new = _scene(old_count=0, new_count=12, no_old=True)
    result = solve_ba_snapshot(payload)

    assert result["ok"], result["reason"]
    for frame in payload["keyframes"][:2]:
        assert np.array_equal(result["poses"][frame["name"]], frame["pose"])
    assert _reprojection_rmse(payload, old_xyz, true_new) > _reprojection_rmse(
        payload, old_xyz, true_new, result=result
    )


def test_no_old_support_rejects_zero_second_keyframe_baseline():
    payload, _old_xyz, _true_new = _scene(old_count=0, new_count=12, no_old=True)
    payload["keyframes"][1]["pose"] = payload["keyframes"][0]["pose"].copy()

    result = solve_ba_snapshot(payload)

    assert not result["ok"]
    assert "baseline" in result["reason"]
    assert result["updates"] == {}


def test_rejects_snapshot_with_non_positive_observation_depth():
    payload, _old_xyz, _true_new = _scene()
    payload["new_landmarks"][-1][2] = -2.0

    result = solve_ba_snapshot(payload)

    assert not result["ok"]
    assert "depth" in result["reason"]
    assert result["updates"] == {}


def test_ten_keyframe_window_keeps_fixed_poses_and_glue_points_and_pulls_the_rest_to_truth():
    fixed = frozenset({0, 1, 2})
    payload, old_xyz, true_new = _scene(keyframes_count=11, fixed=fixed)
    old_before = {key: np.asarray(v).tobytes() for key, v in payload["old_landmarks"].items()}
    initial = {f["name"]: np.asarray(f["pose"]).copy() for f in payload["keyframes"]}
    result = solve_ba_snapshot(payload)

    assert result["ok"], result["reason"]
    assert len(result["poses"]) == 11
    for index, frame in enumerate(payload["keyframes"]):
        solved = np.asarray(result["poses"][frame["name"]])
        if index in fixed:
            assert np.array_equal(solved, initial[frame["name"]])
    # GlueMap points are constants: never variables, never moved.
    assert not set(result["updates"]) & set(payload["old_landmarks"])
    assert {k: np.asarray(v).tobytes() for k, v in payload["old_landmarks"].items()} == old_before

    def center_error(poses):
        return max(
            abs(-np.asarray(poses[f"kf-{index}"])[0, 3] - 0.12 * index) for index in range(3, 11)
        )

    assert center_error(result["poses"]) < center_error(initial)
    assert _reprojection_rmse(payload, old_xyz, true_new) > _reprojection_rmse(
        payload, old_xyz, true_new, result=result
    )


def test_a_window_without_fixed_flags_or_with_too_many_keyframes_is_refused():
    payload, _old, _new = _scene(keyframes_count=12)
    assert "keyframes are required" in solve_ba_snapshot(payload)["reason"]
    payload, _old, _new = _scene()
    del payload["keyframes"][0]["fixed"]
    assert "fixed flag" in solve_ba_snapshot(payload)["reason"]


def _observer_payload(*, window=4, observers=2):
    payload, old_xyz, true_new = _scene(keyframes_count=window + observers)
    frames = payload["keyframes"]
    payload["keyframes"] = frames[:window]
    payload["observers"] = [
        {**frame, "ids": frame["ids"][frame["ids"] < 0], "xy": frame["xy"][frame["ids"] < 0], "fixed": True}
        for frame in frames[window:]
    ]
    return payload


def test_fixed_observers_add_residuals_but_not_poses():
    payload = _observer_payload()
    plain = solve_ba_snapshot({**payload, "observers": []})
    result = solve_ba_snapshot(payload)
    assert plain["ok"] and result["ok"], result.get("reason")
    assert result["before_cost"] > plain["before_cost"]
    assert set(result["poses"]) == {frame["name"] for frame in payload["keyframes"]}
    assert result["keyframe_revisions"] == plain["keyframe_revisions"]
    assert set(result["updates"]) == set(payload["new_landmarks"])


def test_an_observer_behind_its_camera_is_dropped_not_fatal():
    payload = _observer_payload()
    observer = payload["observers"][0]
    pose = np.asarray(observer["pose"], dtype=float).copy()
    pose[2, 3] -= 100.0                                   # every point lies behind this camera
    observer["pose"] = pose
    result = solve_ba_snapshot(payload)
    assert result["ok"], result.get("reason")
    assert result["before_cost"] == pytest.approx(
        solve_ba_snapshot({**payload, "observers": payload["observers"][1:]})["before_cost"]
    )


@pytest.mark.parametrize("damage", ["not_fixed", "positive_id", "duplicate_name", "too_many", "not_a_list"])
def test_invalid_observers_fail_the_snapshot(damage):
    payload = _observer_payload()
    first = payload["observers"][0]
    if damage == "not_fixed":
        first["fixed"] = False
    elif damage == "positive_id":
        first["ids"] = np.array([1, *first["ids"][1:]])
    elif damage == "duplicate_name":
        first["name"] = payload["keyframes"][0]["name"]
    elif damage == "too_many":
        payload["observers"] = [
            {**first, "name": f"extra-{index}", "ordinal": 100 + index} for index in range(21)
        ]
    else:
        payload["observers"] = {"name": "x"}
    result = solve_ba_snapshot(payload)
    assert not result["ok"] and result["reason"] == "invalid observer keyframe"
    assert result["updates"] == {}
