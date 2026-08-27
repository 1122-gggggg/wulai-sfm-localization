from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pose_guided.config import PoseGuidedConfig, load_pose_guided_config
from pose_guided.controller import PoseGuidedController
from pose_guided.types import FusedOdometrySample, PropagationMode
from production_edm_tracker import EDMConfig, ProductionEDMTracker, RuntimeState


class FakeLocalizer:
    def retrieve(self, _rgb, topk, candidates=None):
        return ["ref0"]

    def correspondences(self, _gray, refs, **_kwargs):
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), [0] * len(refs)


def _tracker() -> ProductionEDMTracker:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig()
    tracker.st = RuntimeState()
    tracker.map = SimpleNamespace(
        ref_names=["ref0"],
        images={"ref0": np.zeros((8, 8), dtype=np.uint8)},
        xyz_by_cell={"ref0": np.zeros((1, 3), dtype=np.float32)},
        covis={},
    )
    tracker.cam = SimpleNamespace(model="PINHOLE", width=1280, height=720, params=[900, 900, 640, 360])
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((1, 3), dtype=np.float32)
    tracker.yaws = np.zeros(1, dtype=np.float32)
    tracker.name_of = {0: "ref0"}
    tracker.idx_of = {"ref0": 0}
    tracker.recovery_bank = ["ref0"]
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    tracker.pose_guided = None
    return tracker


def test_disabled_controller_does_not_change_prediction() -> None:
    tracker = _tracker()
    tracker.st.center = np.array([1.0, 2.0, 3.0])
    tracker.st.velocity = np.array([1.0, 0.0, 0.0])
    tracker.st.last_capture_stamp = 1.0
    values = dict(load_pose_guided_config().__dict__)
    values["enabled"] = False
    tracker.attach_pose_guided(PoseGuidedController(PoseGuidedConfig(**values)))
    predicted = tracker._predict_center(1.1)
    assert predicted == pytest.approx([1.1, 2.0, 3.0])

def test_enabled_predict_center_integrates_visual_velocity_once() -> None:
    tracker = _tracker()
    tracker.st.center = np.array([1.0, 2.0, 3.0])
    tracker.st.velocity = np.array([1.0, 0.0, 0.0])
    tracker.st.last_capture_stamp = 1.0
    tracker.st.yaw = 0.25
    values = dict(load_pose_guided_config().__dict__)
    values["enabled"] = True
    tracker.attach_pose_guided(PoseGuidedController(PoseGuidedConfig(**values)))
    predicted = tracker._predict_center(1.1)
    assert predicted == pytest.approx([1.1, 2.0, 3.0])
    prediction = tracker.pose_guided.last_prediction
    assert prediction is not None
    assert prediction.valid
    assert prediction.position == pytest.approx([1.1, 2.0, 3.0])
    assert prediction.yaw == pytest.approx(0.25)



def test_miss_never_confirms_predicted_pose() -> None:
    tracker = _tracker()
    values = dict(load_pose_guided_config().__dict__)
    values["enabled"] = True
    tracker.attach_pose_guided(PoseGuidedController(PoseGuidedConfig(**values)))
    info = {"state_in": "TRACK"}
    tracker.st.state = "TRACK"

    tracker._on_miss(info)
    assert info["ok"] is False
    assert info["pose_status"] != "VISUALLY_CONFIRMED"
    tracker._on_miss(info)
    assert tracker.st.state == "WEAK_TRACK"
    tracker._on_miss(info)
    tracker._on_miss(info)
    assert tracker.st.state == "LOST"



def test_lost_keeps_velocity_and_guesses_with_imu() -> None:
    tracker = _tracker()
    tracker.st.center = np.array([0.0, 0.0, 0.0], dtype=float)
    tracker.st.velocity = np.array([1.0, 0.0, 0.0], dtype=float)
    tracker.st.last_capture_stamp = 1.0
    tracker.st.yaw = 0.0
    tracker.st.state = "TRACK"
    values = dict(load_pose_guided_config().__dict__)
    values["enabled"] = False
    controller = PoseGuidedController(PoseGuidedConfig(**values))
    tracker.attach_pose_guided(controller)
    controller.observe_fused(
        FusedOdometrySample.from_anafi(timestamp=1.0, roll=0.0, pitch=0.0, yaw=0.0)
    )
    controller.maybe_update_anchor(
        {
            "ok": True,
            "state_in": "TRACK",
            "inliers": 90,
            "inlier_ratio": 0.4,
            "inlier_grid_cells": 8,
            "reproj_rms": 1.2,
        },
        timestamp=1.0,
        rotation_cam_from_world=np.eye(3),
        center=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
    )
    info = {"state_in": "TRACK"}
    tracker._on_miss(info)
    tracker._on_miss(info)
    tracker._on_miss(info)
    tracker._on_miss(info)
    assert tracker.st.state == "LOST"
    assert tracker.st.velocity == pytest.approx([1.0, 0.0, 0.0])
    assert controller.anchors.anchor is not None
    controller.observe_fused(
        FusedOdometrySample.from_anafi(timestamp=2.0, roll=0.0, pitch=0.0, yaw=0.0)
    )
    predicted = tracker._predict_center(2.0)
    assert predicted == pytest.approx([1.0, 0.0, 0.0])
    assert controller.last_prediction.propagation_mode is PropagationMode.VELOCITY_INTEGRATION
    assert controller.last_prediction.confirmation.value == "PREDICTED_ONLY"

