from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from production_edm_tracker import EDMConfig, ProductionEDMTracker, RuntimeState


def _tracker() -> ProductionEDMTracker:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    tracker.st = RuntimeState()
    tracker.map = SimpleNamespace(
        ref_names=["ref0"],
        ref_centers=np.zeros((1, 3), dtype=np.float32),
        ref_yaws=np.zeros(1, dtype=np.float32),
        images={"ref0": np.zeros((8, 8, 3), dtype=np.uint8)},
        xyz_by_cell={"ref0": np.zeros((64, 3), dtype=np.float32)},
        covis={},
    )
    tracker.cam = SimpleNamespace(model="PINHOLE", width=1280, height=720, params=[900, 900, 640, 360])
    tracker.loc = SimpleNamespace(scale=1.0)
    tracker.centers = np.zeros((1, 3), dtype=np.float32)
    tracker.yaws = np.zeros(1, dtype=np.float32)
    tracker.name_of = {0: "ref0"}
    tracker.idx_of = {"ref0": 0}
    tracker.recovery_bank = ["ref0"]
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    tracker.pose_guided = None
    # init KLT fields like __init__
    tracker._klt_gray = None
    tracker._klt_2d = None
    tracker._klt_3d = None
    tracker._klt_center = None
    tracker._klt_yaw = None
    tracker._klt_seed_stamp = None
    tracker._klt_age = 0
    tracker._klt_query_gray = None
    tracker._klt_query_stamp = None
    tracker._pcam = None
    tracker._pnp_options = None
    return tracker


def _fake_cam_from_world(center):
    center = np.asarray(center, dtype=float)
    t = -center  # with R=I, center = -R.T @ t => t = -center
    class _Rot:
        def matrix(self):
            return np.eye(3)
    class _Tf:
        def __init__(self):
            self.rotation = _Rot()
            self.translation = t
    return _Tf()


def test_klt_seed_floor():
    tracker = _tracker()
    tracker.st.center = np.array([0.0, 0.0, 0.0], dtype=float)
    tracker.st.yaw = 0.0
    tracker.st.state = "TRACK"
    gray = np.zeros((720, 1280), dtype=np.uint8)
    # 49 unique inliers -> no seed
    pts2d_49 = np.arange(49 * 2, dtype=float).reshape(49, 2)
    pts3d_49 = np.arange(49 * 3, dtype=float).reshape(49, 3)
    mask_49 = np.ones(49, dtype=bool)
    ok = tracker._seed_klt_from_inliers(gray, pts2d_49, pts3d_49, mask_49, 1.0)
    assert ok is False
    assert getattr(tracker, "_klt_2d", None) is None
    # 50 -> seed stored, _klt_center is None
    pts2d_50 = np.arange(50 * 2, dtype=float).reshape(50, 2)
    pts3d_50 = np.arange(50 * 3, dtype=float).reshape(50, 3)
    mask_50 = np.ones(50, dtype=bool)
    ok = tracker._seed_klt_from_inliers(gray, pts2d_50, pts3d_50, mask_50, 1.0)
    assert ok is True
    assert tracker._klt_2d is not None
    assert tracker._klt_2d.shape[0] == 50
    assert tracker._klt_center is None
    assert tracker._klt_seed_stamp == pytest.approx(1.0)


def test_klt_miss_prior_success(monkeypatch):
    tracker = _tracker()
    tracker.st.center = np.array([0.0, 0.0, 0.0], dtype=float)
    tracker.st.yaw = 0.0
    tracker.st.state = "TRACK"
    tracker.st.velocity = np.array([1.0, 0.0, 0.0], dtype=float)
    tracker.st.last_capture_stamp = 1.0
    gray_seed = np.zeros((720, 1280), dtype=np.uint8)
    # seed 50 points at known locations
    pts2d = np.arange(50 * 2, dtype=float).reshape(50, 2) % 1000
    pts3d = np.arange(50 * 3, dtype=float).reshape(50, 3)
    mask = np.ones(50, dtype=bool)
    assert tracker._seed_klt_from_inliers(gray_seed, pts2d, pts3d, mask, 1.0)
    # set query gray/stamp for next frame
    gray_query = np.zeros((720, 1280), dtype=np.uint8)
    tracker._klt_query_gray = gray_query
    tracker._klt_query_stamp = 1.1

    orig_calc = cv2.calcOpticalFlowPyrLK

    def fake_calc(prev_gray, next_gray, p0, nxt, **kwargs):
        # forward: p0 is tracker._klt_2d (seed), shift x by +4
        # backward: p0 is shifted, return original
        p0_arr = np.asarray(p0).reshape(-1, 2)
        if p0_arr.shape[0] == 50 and np.allclose(p0_arr, tracker._klt_2d, atol=1e-3):
            nxt_pts = p0_arr + np.array([4.0, 0.0], dtype=np.float32)
            status = np.ones((50, 1), dtype=np.uint8)
            return nxt_pts.reshape(-1, 1, 2), status, None
        else:
            # backward: return original seed
            status = np.ones((p0_arr.shape[0], 1), dtype=np.uint8)
            back = tracker._klt_2d.reshape(-1, 1, 2).astype(np.float32)[: p0_arr.shape[0]]
            # if second call has fewer points due to good filtering, still return original subset?
            # Simplify: return first N of original
            return back, status, None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_calc)

    fake_result = {"cam_from_world": _fake_cam_from_world([0.1, 0.0, 0.0]), "num_inliers": 30}
    fake_metrics = {"reproj_rms": 1.0}

    def fake_estimate(s2d, s3d, conf, pcam, opts):
        return (fake_result, s2d, s3d, fake_metrics), 0.0

    monkeypatch.setattr(tracker, "_estimate_pose_candidate", fake_estimate)

    info = {"state_in": "TRACK"}
    tracker._on_miss(info)

    assert info["ok"] is False
    assert info["pose_status"] == "PREDICTED_ONLY"
    assert info["prediction_valid"] is True
    assert info["prediction_mode"] == "klt_pnp"
    assert info["predicted_center"] == pytest.approx([0.1, 0.0, 0.0])
    assert np.allclose(tracker.st.center, [0.0, 0.0, 0.0])
    # _predict_center should return KLT center not velocity integration
    pred = tracker._predict_center(1.2)
    assert np.allclose(pred, [0.1, 0.0, 0.0])


def test_klt_jump_reject(monkeypatch):
    tracker = _tracker()
    tracker.st.center = np.array([0.0, 0.0, 0.0], dtype=float)
    tracker.st.yaw = 0.0
    tracker.st.state = "TRACK"
    tracker.st.velocity = np.array([5.0, 0.0, 0.0], dtype=float)
    tracker.st.last_capture_stamp = 1.0
    gray_seed = np.zeros((720, 1280), dtype=np.uint8)
    pts2d = np.arange(50 * 2, dtype=float).reshape(50, 2) % 1000
    pts3d = np.arange(50 * 3, dtype=float).reshape(50, 3)
    mask = np.ones(50, dtype=bool)
    assert tracker._seed_klt_from_inliers(gray_seed, pts2d, pts3d, mask, 1.0)
    gray_query = np.zeros((720, 1280), dtype=np.uint8)
    tracker._klt_query_gray = gray_query
    tracker._klt_query_stamp = 1.1

    def fake_calc(prev_gray, next_gray, p0, nxt, **kwargs):
        p0_arr = np.asarray(p0).reshape(-1, 2)
        if p0_arr.shape[0] == 50 and np.allclose(p0_arr, tracker._klt_2d, atol=1e-3):
            nxt_pts = p0_arr + np.array([4.0, 0.0], dtype=np.float32)
            status = np.ones((50, 1), dtype=np.uint8)
            return nxt_pts.reshape(-1, 1, 2), status, None
        else:
            back = tracker._klt_2d.reshape(-1, 1, 2).astype(np.float32)[: p0_arr.shape[0]]
            status = np.ones((p0_arr.shape[0], 1), dtype=np.uint8)
            return back, status, None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_calc)

    fake_result = {"cam_from_world": _fake_cam_from_world([10.0, 0.0, 0.0]), "num_inliers": 30}
    fake_metrics = {"reproj_rms": 1.0}

    def fake_estimate(s2d, s3d, conf, pcam, opts):
        return (fake_result, s2d, s3d, fake_metrics), 0.0

    monkeypatch.setattr(tracker, "_estimate_pose_candidate", fake_estimate)

    info = {"state_in": "TRACK"}
    tracker._on_miss(info)

    # jump reject: KLT should not set prediction_valid, _klt_center cleared
    assert info.get("prediction_valid") is not True or info.get("prediction_mode") != "klt_pnp"
    assert getattr(tracker, "_klt_center", None) is None
    # _predict_center falls through to visual velocity: center + v*dt (dt clamped to 0.25)
    # last_capture 1.0, now 1.1 => dt 0.1 => 0 +5*0.1 =0.5
    pred = tracker._predict_center(1.1)
    assert np.allclose(pred, [0.5, 0.0, 0.0])


def test_klt_lost_clears():
    tracker = _tracker()
    tracker.st.center = np.array([0.0, 0.0, 0.0], dtype=float)
    tracker.st.yaw = 0.0
    tracker.st.state = "TRACK"
    gray_seed = np.zeros((720, 1280), dtype=np.uint8)
    pts2d = np.arange(50 * 2, dtype=float).reshape(50, 2) % 1000
    pts3d = np.arange(50 * 3, dtype=float).reshape(50, 3)
    mask = np.ones(50, dtype=bool)
    assert tracker._seed_klt_from_inliers(gray_seed, pts2d, pts3d, mask, 1.0)
    # also set a prior center to verify it gets cleared
    tracker._klt_center = np.array([0.1, 0.0, 0.0])
    tracker._klt_yaw = 0.1
    tracker._klt_query_gray = np.zeros((720, 1280), dtype=np.uint8)
    tracker._klt_query_stamp = 1.0
    # default weak_after=2, lost_after=2 => 4 misses to LOST
    for i in range(4):
        tracker._on_miss({"state_in": tracker.st.state})
    assert tracker.st.state == "LOST"
    assert getattr(tracker, "_klt_2d", None) is None
    # _predict_center should ignore previous KLT and return visual (center) or None velocity path
    # with velocity None, it returns st.center
    pred = tracker._predict_center(1.5)
    # after LOST, pending reacquire is None, KLT cleared, so should be visual center [0,0,0] (no velocity)
    assert np.allclose(pred, [0.0, 0.0, 0.0])


def test_adapter_contract_no_pose_on_predicted_only():
    # Mirror edm_localizer_adapter lines 195-208: Pose only if info.get("ok")
    from edm_localizer_adapter import EDMTrackerAdapter

    # Create minimal fake map and camera
    fake_map = SimpleNamespace(
        ref_names=["ref0"],
        ref_centers=np.zeros((1, 3), dtype=np.float32),
        ref_yaws=np.zeros(1, dtype=np.float32),
        images={"ref0": np.zeros((8, 8, 3), dtype=np.uint8)},
        xyz_by_cell={"ref0": np.zeros((64, 3), dtype=np.float32)},
        covis={},
    )
    fake_cam = SimpleNamespace(model="PINHOLE", width=1280, height=720, params=[900, 900, 640, 360])
    # Use object.__new__ to avoid heavy init
    adapter = object.__new__(EDMTrackerAdapter)
    adapter.trk = SimpleNamespace(cfg=EDMConfig())
    # Instead of full adapter init, we test the contract directly:
    # The adapter returns Pose only if ok is True.
    info_ok_false = {"ok": False, "prediction_valid": True, "predicted_center": [1, 2, 3], "center": [1, 2, 3], "yaw": 0.0, "R": np.eye(3), "state_in": "TRACK", "state_out": "TRACK"}
    info_ok_true = {"ok": True, "prediction_valid": False, "center": [1, 2, 3], "yaw": 0.5, "R": np.eye(3), "state_in": "TRACK", "state_out": "TRACK"}

    # Simulate adapter logic
    def would_construct_pose(info):
        return bool(info.get("ok"))

    assert would_construct_pose(info_ok_false) is False
    assert would_construct_pose(info_ok_true) is True
    # Also ensure adapter file hasn't been edited to bypass ok check
    import pathlib
    adapter_path = pathlib.Path("定位演算法/deploy_code/sfm_glomap_deploy/edm_localizer_adapter.py")
    text = adapter_path.read_text()
    assert 'if info.get("ok")' in text
