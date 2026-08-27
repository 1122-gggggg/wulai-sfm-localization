"""Offline characterization for the permanently locked legacy autoflight loop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


FLIGHT_CONTROL_ROOT = Path(__file__).resolve().parents[3] / "定位演算法" / "flight_control"
sys.path.insert(0, str(FLIGHT_CONTROL_ROOT))

import autoflight


class FakeSDF:
    def __init__(self, clearance: float):
        self.value = clearance

    def clearance(self, _point) -> float:
        return self.value

    def gradient(self, _point) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])


class StaticFollower:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def carrot(self, _position) -> np.ndarray:
        return self.target.copy()


def _controller(clearance: float = 2.0) -> tuple[autoflight.AutoFlight, FakeSDF]:
    sdf = FakeSDF(clearance)
    target = np.array([10.0, 0.0, 1.0])
    return autoflight.AutoFlight(sdf, StaticFollower(target), [target]), sdf


def _pose() -> autoflight.Pose:
    return autoflight.Pose(0.0, 0.0, 1.0, 0.0, 1.0)


def test_clearance_breach_hovers_until_hysteresis_clears() -> None:
    controller, sdf = _controller(autoflight.STOP_MARGIN)

    command = controller.step(_pose(), None, now=1.0)
    assert command[:4] == (0, 0, 0, 0)
    assert command[4].startswith("RECOVER hover")
    assert controller.state == "RECOVER"

    sdf.value = autoflight.RESUME_MARGIN
    resumed = controller.step(_pose(), None, now=1.1)
    assert controller.state == "NAV"
    assert resumed[4].startswith("NAV->")


def test_inspection_completion_advances_to_done() -> None:
    controller, _sdf = _controller()
    controller.state = "INSPECT"
    controller._t_inspect = 1.0

    command = controller.step(_pose(), None, now=1.0 + autoflight.INSPECT_SECS)

    assert command == (0, 0, 0, 0, "INSPECT done -> DONE")
    assert controller.state == "DONE"


def test_real_legacy_entrypoint_remains_locked(monkeypatch) -> None:
    controller, _sdf = _controller()
    monkeypatch.setattr(autoflight.signal, "signal", lambda *_args: None)

    with pytest.raises(SystemExit, match="legacy TakeOff is permanently locked"):
        autoflight.run(
            autoflight.Localizer(),
            autoflight.PoleDetector(),
            controller,
            dry_run=False,
        )
