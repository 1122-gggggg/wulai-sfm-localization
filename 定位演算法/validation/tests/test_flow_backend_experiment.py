"""Flow-backend swap used to A/B DIS / FastFlowNet against the KLT bridge.

The backends must be drop-in for ``cv2.calcOpticalFlowPyrLK``'s first two
return values -- an (N,1,2) float32 point array and an (N,1) uint8 status --
because ``_track_klt_prior`` feeds them straight into its forward-backward
gate. Production keeps ``_FLOW_BACKEND = None``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

import production_edm_tracker as pet

_VALIDATION = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "flow_backend_experiment", _VALIDATION / "flow_backend_experiment.py"
)
fbe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fbe)


def _pair(shift: int = 3):
    rng = np.random.default_rng(11)
    raw = rng.integers(0, 255, size=(240, 320)).astype(np.float32)
    a = cv2.normalize(cv2.GaussianBlur(raw, (0, 0), 2.0), None, 0, 255,
                      cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(a), np.ascontiguousarray(np.roll(a, shift, axis=1))


def test_production_default_is_no_backend():
    assert pet._FLOW_BACKEND is None


def test_install_klt_clears_the_hook():
    fbe.install("dis_fast")
    assert pet._FLOW_BACKEND is not None
    assert fbe.install("klt") == "klt"
    assert pet._FLOW_BACKEND is None


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        fbe.build("no_such_flow")


@pytest.mark.parametrize("name", sorted(fbe.DIS_PRESETS))
def test_dis_backend_matches_lk_return_contract(name):
    a, b = _pair(shift=3)
    pts = np.array([[x, y] for x in range(40, 280, 20) for y in range(40, 200, 20)],
                   dtype=np.float64)
    nxt, status = fbe.build(name)(a, b, pts)

    assert nxt.shape == (len(pts), 1, 2) and nxt.dtype == np.float32
    assert status.shape == (len(pts), 1) and status.dtype == np.uint8
    # a pure 3px horizontal shift must come back as a ~3px horizontal flow
    d = nxt.reshape(-1, 2) - pts
    assert np.median(d[:, 0]) == pytest.approx(3.0, abs=0.5)
    assert abs(float(np.median(d[:, 1]))) < 0.5


def test_dis_status_flags_points_pushed_out_of_frame():
    a, b = _pair(shift=3)
    pts = np.array([[10.0, 10.0], [-5.0, 120.0], [1000.0, 120.0]], dtype=np.float64)
    _, status = fbe.build("dis_fast")(a, b, pts)
    assert status[0, 0] == 1
    assert status[1, 0] == 0 and status[2, 0] == 0


def test_installed_backend_drives_the_bridge_gate(monkeypatch):
    """The hook must actually replace both LK passes inside _track_klt_prior."""
    calls = []

    def spy(prev_gray, gray, points):
        calls.append((prev_gray.shape, len(np.asarray(points).reshape(-1, 2))))
        p = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return p, np.ones((p.shape[0], 1), dtype=np.uint8)

    monkeypatch.setattr(pet, "_FLOW_BACKEND", spy)
    a, b = _pair(shift=0)
    tracker = object.__new__(pet.ProductionEDMTracker)
    tracker.cfg = pet.EDMConfig()
    tracker.st = pet.RuntimeState()
    tracker.st.state = "TRACK"
    tracker.st.center = np.zeros(3, dtype=float)
    tracker.esekf = None
    tracker._klt_age = 0
    tracker._klt_seed_stamp = 0.0
    tracker._klt_gray = a
    tracker._klt_2d = np.array([[x, y] for x in range(40, 280, 8)
                                for y in range(40, 200, 8)], dtype=float)
    tracker._klt_3d = np.hstack(
        [tracker._klt_2d, np.full((len(tracker._klt_2d), 1), 10.0)])
    seen = []
    tracker._new_pose_estimation_context = lambda _s: (None, None)
    tracker._pnp_random_seed = lambda: 0
    tracker._estimate_pose_candidate = lambda s2d, *a_, **k_: (seen.append(len(s2d)), (None, 0.0))[1]

    n_seed = len(tracker._klt_3d)
    tracker._track_klt_prior(b, 0.05)
    # forward and backward pass both went through the hook, and an identity
    # flow yields e_FB = 0, so every seeded point survives the gate
    assert len(calls) == 2
    assert seen and seen[0] == n_seed
