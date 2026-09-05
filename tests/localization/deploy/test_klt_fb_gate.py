"""KLT forward-backward consistency gate in the EDM bridge.

The bridge tracks p_t -> p_{t+1} with LK, tracks the result back
p_{t+1} -> p_hat_t, and keeps a point only when e_FB = ||p_t - p_hat_t|| < tau.
Without it LK reports status=1 on points it has actually lost (occlusion,
repeated texture) and those bias the PnP that carries the pose between EDM
keyframes. These tests pin the gate and its tau knob (SFM_EDM_KLT_FB_PX).
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import production_edm_tracker as pet  # noqa: F401
from production_edm_tracker import EDMConfig, ProductionEDMTracker, RuntimeState


def _rng_texture(seed: int) -> np.ndarray:
    """Band-limited texture: smooth enough for the LK pyramid to be reliable,
    so any rejection the gate makes comes from the corruption below and not
    from an unresolvable corner."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 255, size=(720, 1280)).astype(np.float32)
    return cv2.normalize(
        cv2.GaussianBlur(raw, (0, 0), 2.0), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def _grid_points(n_side: int = 12) -> np.ndarray:
    xs = np.linspace(120.0, 1160.0, n_side)
    ys = np.linspace(100.0, 620.0, n_side)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)


def _tracker(fb_px: float | None = None) -> ProductionEDMTracker:
    t = object.__new__(ProductionEDMTracker)
    t.cfg = EDMConfig()
    t.st = RuntimeState()
    t.st.state = "TRACK"
    t.st.center = np.zeros(3, dtype=float)
    t.cam = SimpleNamespace(
        model="PINHOLE", width=1280, height=720, params=[900.0, 900.0, 640.0, 360.0]
    )
    t.esekf = None
    t._klt_age = 0
    t._klt_seed_stamp = 0.0
    t._pcam = None
    t._pnp_options = None
    if fb_px is not None:
        t._klt_fb_px = float(fb_px)
    return t


def _solver_spy(tracker: ProductionEDMTracker) -> list[np.ndarray]:
    """Capture the 2D set that survives the gate, then abort the estimate."""
    seen: list[np.ndarray] = []
    tracker._new_pose_estimation_context = lambda _seed: (None, None)
    tracker._pnp_random_seed = lambda: 0

    def _estimate(s2d, _s3d, _w, _pcam, _options):
        seen.append(np.asarray(s2d, dtype=float).copy())
        return None, 0.0

    tracker._estimate_pose_candidate = _estimate
    return seen


_BAND = slice(240, 480)


def _corrupted_pair(occlude: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame pair with a pure 3px shift, except one horizontal band that is
    replaced by unrelated texture -- points there are genuinely lost."""
    a = _rng_texture(7)
    b = np.roll(a, shift=3, axis=1)
    b[occlude, :] = _rng_texture(99)[occlude, :]
    return a, b, _grid_points()


def test_shipped_defaults_are_klt_with_tau_one_px():
    """Pin the shipped bridge config: sparse pyramidal LK (no swapped flow
    backend) gated at e_FB < 1.0 px. Both were re-measured 2026-09-03; tau=0.5
    and the DIS / FastFlowNet backends were rejected (see the ledger)."""
    assert pet._KLT_FB_PX == 1.0
    assert pet._FLOW_BACKEND is None


def test_gate_drops_points_in_the_corrupted_band():
    a, b, pts = _corrupted_pair(_BAND)
    t = _tracker(fb_px=1.0)
    t._klt_gray = np.ascontiguousarray(a)
    t._klt_2d = pts.copy()
    t._klt_3d = np.hstack([pts, np.full((len(pts), 1), 10.0)])
    seen = _solver_spy(t)

    assert t._track_klt_prior(np.ascontiguousarray(b), 0.05) is None  # solver aborted
    assert seen, "gate rejected every point; expected the clean ones to survive"
    kept = seen[0]
    seeded_in_band = (pts[:, 1] >= _BAND.start) & (pts[:, 1] <= _BAND.stop)
    kept_in_band = (kept[:, 1] >= _BAND.start) & (kept[:, 1] <= _BAND.stop)
    # the gate must throw out the large majority of the genuinely lost tracks
    # (it is not perfect: a few land FB-consistent on the replacement texture)
    assert kept_in_band.sum() <= 0.2 * seeded_in_band.sum()
    # and it must not cost a single clean track
    assert len(kept) - kept_in_band.sum() == int((~seeded_in_band).sum())


def test_zero_tau_rejects_everything():
    a, b, pts = _corrupted_pair(_BAND)
    t = _tracker(fb_px=0.0)
    t._klt_gray = np.ascontiguousarray(a)
    t._klt_2d = pts.copy()
    t._klt_3d = np.hstack([pts, np.full((len(pts), 1), 10.0)])
    seen = _solver_spy(t)

    assert t._track_klt_prior(np.ascontiguousarray(b), 0.05) is None
    assert not seen, "tau=0 must reject every track before the pose solve"
    assert t._klt_2d is None  # cache cleared


def test_tau_is_monotonic_in_kept_tracks():
    a, b, pts = _corrupted_pair(_BAND)
    counts = {}
    for tau in (0.25, 1.0, 5.0):
        t = _tracker(fb_px=tau)
        t._klt_gray = np.ascontiguousarray(a)
        t._klt_2d = pts.copy()
        t._klt_3d = np.hstack([pts, np.full((len(pts), 1), 10.0)])
        seen = _solver_spy(t)
        t._track_klt_prior(np.ascontiguousarray(b), 0.05)
        counts[tau] = len(seen[0]) if seen else 0
    assert counts[0.25] <= counts[1.0] <= counts[5.0]
    assert counts[5.0] > counts[0.25], "tau must actually change the kept set"


def test_missing_attr_falls_back_to_module_default():
    """Trackers built with object.__new__ (older tests, pickled state) must
    still gate at the module default rather than crash."""
    a, b, pts = _corrupted_pair(_BAND)
    t = _tracker(fb_px=None)
    assert not hasattr(t, "_klt_fb_px")
    t._klt_gray = np.ascontiguousarray(a)
    t._klt_2d = pts.copy()
    t._klt_3d = np.hstack([pts, np.full((len(pts), 1), 10.0)])
    seen = _solver_spy(t)
    t._track_klt_prior(np.ascontiguousarray(b), 0.05)
    assert seen
