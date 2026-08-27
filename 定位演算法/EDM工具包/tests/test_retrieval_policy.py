from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from production_edm_tracker import (  # noqa: E402
    EDMConfig,
    ProductionEDMTracker,
    RuntimeState,
)


class FakeLocalizer:
    def __init__(self) -> None:
        self.retrieve_calls = 0
        self.retrieve_topks: list[int] = []
        self.retrieve_candidates: list[list[str] | None] = []

    def retrieve(
        self,
        _rgb: np.ndarray,
        topk: int,
        candidates: list[str] | None = None,
    ) -> list[str]:
        self.retrieve_calls += 1
        self.retrieve_topks.append(topk)
        self.retrieve_candidates.append(None if candidates is None else list(candidates))
        return ["ref0"]

    def correspondences(self, _gray: np.ndarray, refs: list[str], **_kwargs):
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), [0] * len(refs)

    def correspondences_by_ref(self, gray: np.ndarray, refs: list[str], **kwargs):
        points2d, points3d, confidence, counts = self.correspondences(gray, refs, **kwargs)
        return [
            (
                points2d[0:0],
                points3d[0:0],
                confidence[0:0],
                count,
            )
            for count in counts
        ]


def make_tracker() -> tuple[ProductionEDMTracker, FakeLocalizer, list[int]]:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        global_retrieval_policy="boot_and_lost_once",
        local_topk=1,
        weak_local_topk=3,
        lost_local_topk=5,
        lost_local_grace_frames=2,
    )
    tracker.st = RuntimeState()
    tracker.map = SimpleNamespace(
        ref_names=["ref0"],
        images={"ref0": np.zeros((576, 1024), dtype=np.uint8)},
        xyz_by_cell={"ref0": np.zeros((1, 3), dtype=np.float32)},
        covis={},
    )
    tracker.cam = SimpleNamespace(
        model="PINHOLE", width=1280, height=720,
        params=np.array([900.0, 900.0, 640.0, 360.0]),
    )
    tracker.loc = FakeLocalizer()
    tracker.centers = np.zeros((1, 3), dtype=np.float32)
    tracker.yaws = np.zeros(1, dtype=np.float32)
    tracker.name_of = {0: "ref0"}
    tracker.idx_of = {"ref0": 0}
    tracker.recovery_bank = ["ref0"]
    tracker.temporal_gray = None
    tracker.temporal_xyz_by_cell = None
    requested_topk: list[int] = []

    def candidates(topk: int, _capture_stamp: float | None = None) -> list[str]:
        requested_topk.append(topk)
        return ["ref0"]

    tracker._track_candidates = candidates
    return tracker, tracker.loc, requested_topk


def test_boot_and_each_lost_episode_stage_two_megaloc_attempts() -> None:
    tracker, localizer, _ = make_tracker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    first = tracker.localize(frame)
    second = tracker.localize(frame)
    third = tracker.localize(frame)
    assert first["candidate_mode"] == "megaloc_boot"
    assert second["candidate_mode"] == "megaloc_boot_retry"
    assert third["candidate_mode"] == "edm_boot_refs"
    assert localizer.retrieve_topks == [10, 20]
    assert localizer.retrieve_candidates == [None, None]

    tracker.st.state = "LOST"
    tracker.st.lost_frames = 1
    tracker.st.center = np.zeros(3, dtype=np.float32)
    tracker.st.last_capture_stamp = 1.0
    lost_first = tracker.localize(frame, capture_stamp=1.1)
    lost_second = tracker.localize(frame, capture_stamp=1.2)
    lost_third = tracker.localize(frame, capture_stamp=1.3)
    lost_fourth = tracker.localize(frame, capture_stamp=1.4)
    assert lost_first["candidate_mode"] == "edm_local_recovery"
    assert lost_second["candidate_mode"] == "edm_local_recovery"
    assert lost_third["candidate_mode"] == "megaloc_lost_near"
    assert lost_fourth["candidate_mode"] == "edm_map_scan"
    assert localizer.retrieve_topks == [10, 20, 5]
    assert localizer.retrieve_candidates == [None, None, ["ref0"]]
    assert localizer.retrieve_calls == 3

    tracker.st.state = "TRACK"
    tracker.st.lost_global_retrieval_attempts = 0
    tracker.st.lost_global_retrieval_done = False
    tracker.st.lost_frames = 1
    tracker.st.state = "LOST"
    tracker.localize(frame, capture_stamp=1.5)
    tracker.localize(frame, capture_stamp=1.6)
    tracker.localize(frame, capture_stamp=1.7)
    assert localizer.retrieve_topks == [10, 20, 5, 5]
    assert localizer.retrieve_candidates[-1] == ["ref0"]


def test_lost_megaloc_uses_global_search_when_the_pose_prior_is_stale() -> None:
    tracker, localizer, _ = make_tracker()
    tracker.st.state = "LOST"
    tracker.st.lost_frames = 3
    tracker.st.center = np.zeros(3, dtype=np.float32)
    tracker.st.last_capture_stamp = 1.0

    result = tracker.localize(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        capture_stamp=1.0 + tracker.cfg.lost_prior_max_age_s + 0.1,
    )

    assert result["candidate_mode"] == "megaloc_lost"
    assert localizer.retrieve_candidates == [None]
    assert localizer.retrieve_topks == [5]


def test_boot_once_policy_keeps_one_megaloc_attempt() -> None:
    tracker, localizer, _ = make_tracker()
    tracker.cfg.global_retrieval_policy = "boot_once"
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    first = tracker.localize(frame)
    second = tracker.localize(frame)

    assert first["candidate_mode"] == "megaloc_boot"
    assert second["candidate_mode"] == "edm_boot_refs"
    assert localizer.retrieve_topks == [10]


def test_track_and_weak_only_change_local_candidate_count() -> None:
    tracker, localizer, requested_topk = make_tracker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref0"]
    tracker.st.center = np.zeros(3, dtype=np.float32)
    tracker.st.last_refs = [0]

    tracker.st.state = "TRACK"
    tracker.localize(frame)
    tracker.st.state = "WEAK_TRACK"
    tracker.localize(frame)

    assert requested_topk == [1, 3]
    assert localizer.retrieve_calls == 0


def test_entering_lost_rearms_one_global_retrieval() -> None:
    tracker, _, _ = make_tracker()
    tracker.st.state = "WEAK_TRACK"
    tracker.st.misses = tracker.cfg.weak_after + tracker.cfg.lost_after - 1
    tracker.st.lost_global_retrieval_attempts = 2
    tracker.st.lost_global_retrieval_done = True

    info: dict = {}
    tracker._on_miss(info)

    assert tracker.st.state == "LOST"
    assert tracker.st.lost_global_retrieval_attempts == 0
    assert not tracker.st.lost_global_retrieval_done


def test_lost_skips_megaloc_during_local_grace() -> None:
    tracker, localizer, requested_topk = make_tracker()
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref0"]
    tracker.st.state = "LOST"
    tracker.st.lost_frames = 1
    tracker.st.center = np.zeros(3, dtype=np.float32)
    tracker.st.last_capture_stamp = 1.0
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    first = tracker.localize(frame, capture_stamp=1.1)
    second = tracker.localize(frame, capture_stamp=1.2)

    assert first["candidate_mode"] == "edm_local_recovery"
    assert second["candidate_mode"] == "edm_local_recovery"
    assert requested_topk == [5, 5]
    assert localizer.retrieve_calls == 0


def test_lost_fires_one_megaloc_shot_after_grace_then_scans() -> None:
    tracker, localizer, requested_topk = make_tracker()
    tracker.st.global_retrieval_calls = 1
    tracker.st.boot_refs = ["ref0"]
    tracker.st.state = "LOST"
    tracker.st.lost_frames = 3
    tracker.st.center = np.zeros(3, dtype=np.float32)
    tracker.st.last_capture_stamp = 1.0
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    first = tracker.localize(frame, capture_stamp=1.1)
    second = tracker.localize(frame, capture_stamp=1.2)

    assert first["candidate_mode"] == "megaloc_lost_near"
    assert second["candidate_mode"] == "edm_map_scan"
    assert localizer.retrieve_calls == 1
    assert localizer.retrieve_topks == [5]
    assert requested_topk == [24]
