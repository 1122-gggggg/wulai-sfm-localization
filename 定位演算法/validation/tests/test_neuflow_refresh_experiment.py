from __future__ import annotations

import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch


VALIDATION_DIR = Path(__file__).resolve().parents[1]
TRACKER_DIR = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
for directory in (VALIDATION_DIR, TRACKER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import neuflow_refresh_experiment as neuflow_experiment
from neuflow_refresh_experiment import NeuFlowRefreshTracker, sample_flow_at_points
from pose_types import Pose


def test_sparse_flow_sampling_preserves_source_pixel_units():
    flow = torch.zeros((1, 2, 4, 8), dtype=torch.float32)
    flow[:, 0] = 1.0
    points = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=np.float32)

    tracked, valid = sample_flow_at_points(flow, points, 8, 4)

    np.testing.assert_allclose(tracked, points + [1.0, 0.0], atol=1e-6)
    assert valid.tolist() == [True, True]


def test_sparse_flow_sampling_rescales_model_displacement():
    flow = torch.zeros((1, 2, 2, 4), dtype=torch.float32)
    flow[:, 0] = 1.0
    points = np.array([[2.0, 1.0]], dtype=np.float32)

    tracked, valid = sample_flow_at_points(flow, points, 8, 4)

    np.testing.assert_allclose(tracked, points + [2.0, 0.0], atol=1e-6)
    assert valid.tolist() == [True]


def test_refresh_interval_runs_deep_path_on_schedule():
    tracker = NeuFlowRefreshTracker.__new__(NeuFlowRefreshTracker)
    tracker.state = SimpleNamespace(mode="TRACK")
    tracker.neuflow_refresh_interval = 3
    tracker._nf_since_refresh = 2
    tracker._nf_2d = np.zeros((100, 2), np.float32)
    tracker._nf_3d = np.zeros((100, 3), np.float32)
    tracker._nf_previous_frame = np.zeros((4, 8, 3), np.uint8)
    calls = []
    expected = Pose(1.0, 2.0, 3.0, 0.1, 4.0)

    def fake_deep(self, frame, stage="refresh"):
        calls.append((frame, stage))
        return expected

    tracker._deep_refresh = MethodType(fake_deep, tracker)
    frame = np.ones((4, 8, 3), np.uint8)

    actual = tracker.localize_frame(frame, capture_stamp=time.monotonic())

    assert actual is expected
    assert len(calls) == 1
    assert calls[0][0] is frame
    assert calls[0][1] == "refresh"


def test_lost_flow_falls_back_to_deep_on_same_frame():
    tracker = NeuFlowRefreshTracker.__new__(NeuFlowRefreshTracker)
    tracker.state = SimpleNamespace(mode="TRACK")
    tracker.neuflow_refresh_interval = 3
    tracker._nf_since_refresh = 0
    tracker._nf_2d = np.zeros((100, 2), np.float32)
    tracker._nf_3d = np.zeros((100, 3), np.float32)
    tracker._nf_previous_frame = np.zeros((4, 8, 3), np.uint8)
    tracker._nf_min_track = 100
    tracker._nf_min_seed = 100
    tracker._last_inl_2d = None
    tracker._last_inl_3d = None
    tracker.neuflow_backend = SimpleNamespace(
        track_points=lambda *_args: (
            np.zeros((100, 2), np.float32),
            np.zeros(100, dtype=bool),
            {"prepare_ms": 1.0, "gpu_ms": 2.0, "total_ms": 3.0},
        )
    )
    expected = Pose(1.0, 2.0, 3.0, 0.1, 4.0)
    deep_frames = []

    def fake_deep(self, frame):
        deep_frames.append(frame)
        self._last_info = {
            "mode": "TRACK", "accepted": False, "weak": True, "inliers": 0,
        }
        return expected

    tracker._localize_frame_deep = MethodType(fake_deep, tracker)
    frame = np.ones((4, 8, 3), np.uint8)

    actual = tracker.localize_frame(frame, capture_stamp=time.monotonic())

    assert actual is expected
    assert len(deep_frames) == 1
    assert deep_frames[0] is frame
    assert tracker._last_info["neuflow_fallback"] is True
    assert tracker._last_info["neuflow_stage"] == "track_lost"
    assert tracker._last_info["neuflow_attempt_tracked"] == 0


def test_invalid_flow_shape_is_rejected():
    with pytest.raises(ValueError, match="expected flow"):
        sample_flow_at_points(torch.zeros((2, 2, 4, 8)), np.zeros((1, 2)), 8, 4)


def test_tracking_history_reset_also_clears_neuflow_anchors(monkeypatch):
    tracker = NeuFlowRefreshTracker.__new__(NeuFlowRefreshTracker)
    tracker._nf_2d = np.ones((2, 2), np.float32)
    tracker._nf_3d = np.ones((2, 3), np.float32)
    tracker._nf_previous_frame = np.ones((4, 8, 3), np.uint8)
    tracker._nf_since_refresh = 2
    base_calls = []
    monkeypatch.setattr(
        neuflow_experiment.ProductionXFeatTracker,
        "_clear_tracking_history",
        lambda _self: base_calls.append(True),
    )

    tracker._clear_tracking_history()

    assert base_calls == [True]
    assert tracker._nf_2d is None
    assert tracker._nf_3d is None
    assert tracker._nf_previous_frame is None
    assert tracker._nf_since_refresh == 0
