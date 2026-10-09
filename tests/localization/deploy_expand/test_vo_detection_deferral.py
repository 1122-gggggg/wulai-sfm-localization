"""Differential tests: Step-1 front-end refactors against the code they replaced."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from online_workmap import (
    SessionWorkMap,
    _copy_landmark,
    _keyframe_observation_rows,
    _landmark_observation_rows,
)
from test_workmap_observations import _scene, _tracker


def _legacy_detect(vo, gray, occupied):
    """``TwoRateTracker._detect_features`` as it was before the corner search moved."""
    min_distance = int(vo.min_distance)
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    if occupied.size:
        pts = np.asarray(occupied, dtype=int)
        valid = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < gray.shape[1])
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < gray.shape[0])
        )
        for x, y in pts[valid]:
            cv2.circle(mask, (int(x), int(y)), min_distance, 0, -1)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=int(vo.detect_cap), qualityLevel=0.01, minDistance=min_distance, mask=mask
    )
    return np.zeros((0, 2), dtype=float) if corners is None else corners.reshape(-1, 2).astype(float)


def test_detect_features_equals_the_implementation_it_replaced(tmp_path):
    tracker = _tracker(tmp_path)
    rng = np.random.default_rng(3)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (720, 1280), dtype=np.uint8), (0, 0), 1.2)
    inside = rng.uniform([0, 0], [1280, 720], (300, 2))
    edges = np.array([[0, 0], [1279, 719], [0, 719], [1279, 0], [-0.4, 5.0], [-3, 7], [1280, 4], [1e6, -1e6]])
    for occupied in (np.zeros((0, 2)), inside, np.concatenate([inside, edges, inside[:20]])):
        expected = _legacy_detect(tracker.profile.vo, gray, occupied)
        assert len(expected) > 20
        assert np.array_equal(tracker._detect_features(gray, occupied), expected)
    blank = np.full((720, 1280), 7, dtype=np.uint8)
    assert tracker._detect_features(blank, np.zeros((0, 2))).shape == (0, 2)


def _state(tracker):
    return {
        "vo_xy": tracker._vo_xy, "vo_seed_xy": tracker._vo_seed_xy, "vo_bid": tracker._vo_bid,
        "vo_pose_by_bid": tracker._vo_pose_by_bid, "vo_seed_meta": tracker._vo_seed_meta,
        "vo_bid_next": tracker._vo_bid_next,
    }


def _seeded_pair(tmp_path, *, active_provenance=True):
    pair = []
    gray = np.random.default_rng(7).integers(0, 256, (720, 1280), dtype=np.uint8)
    for deferred in (False, True):
        directory = tmp_path / ("deferred" if deferred else "sync")
        directory.mkdir()
        tracker = _tracker(directory)
        tracker._online_workmap_active = active_provenance
        _, _, pose = _scene(tracker, n=40)
        tracker.profile = replace(tracker.profile, vo=replace(tracker.profile.vo, min_live=200))
        tracker._vo_detection_deferrable = lambda deferred=deferred: deferred
        tracker._ordinal, tracker._previous_capture_stamp = 6, 1.7
        tracker._vo_seed_xy = tracker._vo_xy = np.array([[300.0, 200.0], [640.5, 360.5]])
        tracker._vo_bid = np.array([0, 0], dtype=np.int64)
        tracker._vo_bid_next = 1
        tracker._vo_pose_by_bid = {0: pose.copy()}
        pair.append((tracker, pose))
    return gray, pair


@pytest.mark.parametrize("active", [True, False])
def test_deferred_detection_registers_exactly_what_synchronous_detection_does(tmp_path, active):
    gray, ((sync, pose), (deferred, _)) = _seeded_pair(tmp_path, active_provenance=active)
    sync._vo_keyframe(gray, pose)
    deferred._vo_keyframe(gray, pose.copy())
    assert deferred._pending_detection is not None
    assert deferred._vo_bid_next == 1 and set(deferred._vo_pose_by_bid) == {0}
    deferred._finish_pending_detection()
    assert deferred._pending_detection is None
    expected, actual = _state(sync), _state(deferred)
    assert sync._vo_bid_next == 2 and len(sync._vo_xy) > 2
    assert np.array_equal(actual["vo_xy"], expected["vo_xy"])
    assert np.array_equal(actual["vo_seed_xy"], expected["vo_seed_xy"])
    assert np.array_equal(actual["vo_bid"], expected["vo_bid"])
    assert actual["vo_bid_next"] == expected["vo_bid_next"]
    assert actual["vo_seed_meta"] == expected["vo_seed_meta"]
    assert (1 in actual["vo_seed_meta"]) is active
    assert set(actual["vo_pose_by_bid"]) == set(expected["vo_pose_by_bid"])
    for key, value in expected["vo_pose_by_bid"].items():
        assert np.array_equal(actual["vo_pose_by_bid"][key], value)
    deferred._finish_pending_detection()  # second call is a no-op
    assert deferred._vo_bid_next == 2


def test_deferred_seed_keeps_the_pose_and_provenance_of_the_detecting_frame(tmp_path):
    gray, (_, (deferred, pose)) = _seeded_pair(tmp_path)
    detecting_pose = pose.copy()
    deferred._vo_keyframe(gray, pose)
    pose[:] = 0.0                      # the caller reuses its pose buffer
    deferred._ordinal, deferred._previous_capture_stamp = 7, 1.8
    deferred._finish_pending_detection()
    assert np.array_equal(deferred._vo_pose_by_bid[1], detecting_pose)
    meta = deferred._vo_seed_meta[1]
    assert (meta.ordinal, meta.stamp) == (6, 1.7)


def test_reset_and_close_discard_a_pending_detection(tmp_path):
    gray, (_, (tracker, pose)) = _seeded_pair(tmp_path)
    tracker._vo_keyframe(gray, pose)
    assert tracker._pending_detection is not None
    tracker.reset()
    assert tracker._pending_detection is None and tracker._vo_bid_next == 0
    assert len(tracker._vo_xy) == 0
    tracker._ordinal, tracker._previous_capture_stamp = 8, 2.1
    tracker._vo_keyframe(gray, pose)
    assert tracker._pending_detection is not None
    executor = tracker._detect_executor
    tracker.close()
    assert tracker._pending_detection is None and tracker._detect_executor is None
    with pytest.raises(RuntimeError):
        executor.submit(lambda: None)


def _legacy_copy_landmark(value):
    extras = {
        k: v for k, v in value.items() if k not in ("xyz", "observations", "last_inlier", "revision")
    }
    result = copy.deepcopy(extras) if extras else {}
    result["xyz"] = np.asarray(value["xyz"], dtype=np.float64).reshape(3).copy()
    result["observations"] = tuple(
        (int(o), float(s), np.asarray(xy, dtype=np.float64).reshape(2).copy(),
         np.asarray(p, dtype=np.float64).reshape(3, 4).copy())
        for o, s, xy, p in value.get("observations", ())
    )
    if value.get("last_inlier") is not None:
        o, s, xy = value["last_inlier"]
        result["last_inlier"] = (int(o), float(s), np.asarray(xy, dtype=np.float64).reshape(2).copy())
    result["revision"] = int(value.get("revision", 0))
    return result


def _landmark(extras=True):
    pose = np.column_stack((np.eye(3), [0.1, -0.0, 1e-7]))
    value = {
        "xyz": np.array([1.0, -0.0, 3.5]),
        "observations": ((1, 0.5, np.array([10.0, 20.0]), pose), (2, 0.6, np.array([11.0, 21.0]), pose)),
        "last_inlier": (2, 0.6, np.array([11.0, 21.0])),
        "revision": 3,
    }
    if extras:
        value.update(
            source="vo", count=4, flag=True, nothing=None, scalar=np.float32(1.5),
            vector=np.arange(3.0), numbers=[1, 2.5], pair=(1, 2), nested={"a": [1, {"b": np.zeros(2)}]},
            mixed=[np.zeros(2), 3],
        )
    return value


def _same(left, right):
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right) and left.dtype == right.dtype
    else:
        assert left == right and type(left) is type(right)


@pytest.mark.parametrize("extras", [True, False])
def test_copy_landmark_matches_deepcopy_version_and_is_independent(extras):
    source = _landmark(extras)
    pristine = copy.deepcopy(source)
    copied = _copy_landmark(source)
    _same(copied, _legacy_copy_landmark(source))
    # mutate every mutable container of the copy; the source must not notice
    copied["xyz"][:] = 9.0
    for key in ("vector", "numbers", "nested", "mixed"):
        if key in copied:
            value = copied[key]
            if isinstance(value, np.ndarray):
                value[:] = 9.0
            elif isinstance(value, list):
                value.append(99)
                if value and isinstance(value[0], np.ndarray):
                    value[0][:] = 9.0
            else:
                value["a"].append(99)
                value["a"][1]["b"][:] = 9.0
    _same(source, pristine)


def _legacy_carry(session, chain_names, poses, updates):
    for key, landmark in session.landmarks.items():
        names = session._frames_by_landmark.get(int(key), set())
        if key >= 0 or key in updates or not names or not names <= chain_names:
            continue
        host = min(names, key=lambda name: int(session.keyframes[name]["ordinal"]))
        old = np.asarray(session.keyframes[host]["pose"], dtype=np.float64)
        new = poses[host]
        camera = old[:, :3] @ np.asarray(landmark["xyz"], dtype=np.float64) + old[:, 3]
        updates[int(key)] = new[:, :3].T @ (camera - new[:, 3])


def test_single_view_carry_matches_full_scan():
    rng = np.random.default_rng(3)
    session = object.__new__(SessionWorkMap)
    session.landmarks, session.keyframes, session._frames_by_landmark = {}, {}, {}
    frame_ids = {"a": [-1, -2, -3], "b": [-2, -3, -4], "c": [-3, -4, 5], "outside": [-4, -6]}
    for index, name in enumerate(frame_ids):
        pose = np.column_stack((np.eye(3), rng.normal(size=3)))
        ids = np.array(frame_ids[name], dtype=np.int64)
        session.keyframes[name] = {"ordinal": 10 - index, "pose": pose, "ba_ids": ids}
        for key in ids:
            if key < 0:
                session._frames_by_landmark.setdefault(int(key), set()).add(name)
    for key in (-1, -2, -3, -4, 5, -6, -7):
        session.landmarks[key] = {"xyz": rng.normal(size=3)}
    chain = {"a", "b", "c"}
    poses = {name: np.column_stack((np.eye(3), rng.normal(size=3))) for name in chain}
    for preset in ({}, {-2: np.zeros(3)}):
        expected, actual = dict(preset), dict(preset)
        _legacy_carry(session, chain, poses, expected)
        session._carry_single_view_points_locked(chain, poses, actual)
        assert list(actual) == list(expected) and expected
        for key in expected:
            assert np.array_equal(actual[key], expected[key])


def test_observation_rows_serialize_identically_on_the_fast_path(tmp_path):
    pose = np.column_stack((np.eye(3), [0.1, -0.0, 1e-7]))
    landmarks = {-2: _landmark(False), 7: _landmark(False), -1: _landmark(False)}
    landmarks[-1]["observations"] = ()
    rows = _landmark_observation_rows("sid", landmarks, epoch_by_capture={(1, 0.5): np.int64(2)})
    task = {
        "ordinal": np.int64(4), "stamp": 1.25, "pose": pose,
        "landmark_revisions": {-2: 3, -1: np.int64(1)},
        "ids": np.array([-2, -1]), "xy": np.array([[1.0, -0.0], [2.0, 1e-7]]),
        "xyz": np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        "all_ids": np.array([9, -2, 4]), "all_xy": np.zeros((3, 2), np.float32),
        "all_xyz": np.ones((3, 3), np.float32),
    }
    rows.extend(_keyframe_observation_rows("sid", task, "kf-1", Path("/tmp/kf-1.png")))
    rows.append({"event": "geometry_archive_complete", "name": "kf-1", "revision": np.int64(2)})
    assert len(rows) > 6
    slow, fast = tmp_path / "slow.jsonl", tmp_path / "fast.jsonl"
    SessionWorkMap._append_jsonl_rows(slow, rows)
    SessionWorkMap._append_jsonl_rows(fast, rows, str_keyed_rows=True)
    assert fast.read_bytes() == slow.read_bytes()
    assert all(json.loads(line) for line in fast.read_text().splitlines())
    with pytest.raises(ValueError):
        SessionWorkMap._append_jsonl_rows(tmp_path / "nan.jsonl", [{"x": np.array([np.nan])}], str_keyed_rows=True)
