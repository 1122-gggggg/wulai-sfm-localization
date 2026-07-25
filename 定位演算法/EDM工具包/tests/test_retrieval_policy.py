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

    def retrieve(self, _rgb: np.ndarray, _topk: int) -> list[str]:
        self.retrieve_calls += 1
        return ["ref0"]

    def correspondences(self, _gray: np.ndarray, refs: list[str], **_kwargs):
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), [0] * len(refs)


def make_tracker() -> tuple[ProductionEDMTracker, FakeLocalizer, list[int]]:
    tracker = object.__new__(ProductionEDMTracker)
    tracker.cfg = EDMConfig(
        global_retrieval_policy="boot_and_lost_once",
        local_topk=1,
        weak_local_topk=3,
        lost_local_topk=5,
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


def test_boot_and_each_lost_episode_run_megaloc_once() -> None:
    tracker, localizer, _ = make_tracker()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    first = tracker.localize(frame)
    second = tracker.localize(frame)
    assert first["candidate_mode"] == "megaloc_boot"
    assert second["candidate_mode"] == "edm_boot_refs"
    assert localizer.retrieve_calls == 1

    tracker.st.state = "LOST"
    lost_first = tracker.localize(frame)
    lost_second = tracker.localize(frame)
    assert lost_first["candidate_mode"] == "megaloc_lost"
    assert lost_second["candidate_mode"] == "edm_local_recovery"
    assert localizer.retrieve_calls == 2

    tracker.st.state = "TRACK"
    tracker.st.lost_global_retrieval_done = False
    tracker.st.state = "LOST"
    tracker.localize(frame)
    assert localizer.retrieve_calls == 3


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
    tracker.st.lost_global_retrieval_done = True

    info: dict = {}
    tracker._on_miss(info)

    assert tracker.st.state == "LOST"
    assert not tracker.st.lost_global_retrieval_done
