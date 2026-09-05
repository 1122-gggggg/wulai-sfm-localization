"""KLT prediction in LOST (SFM_EDM_KLT_LOST_PREDICT) and its shadow evaluator.

Both prediction branches are gated to TRACK/WEAK_TRACK, so a LOST frame reports
no pose at all. The opt-in lets the KLT chain keep re-solving PnP against its
fixed 3D anchor set while LOST. These tests pin three things: the gate is off by
default, the horizon is configurable so it can be swept against measured drift,
and the shadow evaluator cannot perturb the production cache -- the last is what
makes it safe to score accuracy on frames the tracker also solved.
"""
from __future__ import annotations

import numpy as np
import pytest

from production_edm_tracker import (
    _KLT_MAX_AGE_S,
    _KLT_MAX_FRAMES,
    EDMConfig,
    ProductionEDMTracker,
    RuntimeState,
)


def _tracker(state: str = "LOST", **attrs) -> ProductionEDMTracker:
    t = object.__new__(ProductionEDMTracker)
    t.cfg = EDMConfig()
    t.st = RuntimeState()
    t.st.state = state
    t.st.center = np.zeros(3, dtype=float)
    t._klt_lost_predict = False
    t._klt_shadow_eval = False
    t._klt_max_frames = int(_KLT_MAX_FRAMES)
    t._klt_max_age_s = float(_KLT_MAX_AGE_S)
    t._klt_2d = np.zeros((60, 2), dtype=float)
    t._klt_3d = np.zeros((60, 3), dtype=float)
    t._klt_gray = np.zeros((16, 16), dtype=np.uint8)
    t._klt_age = 0
    t._klt_seed_stamp = 0.0
    t._klt_center = None
    t._klt_yaw = None
    t._shadow_klt = None
    t._shadow_klt_anchor_frames = 0
    for key, value in attrs.items():
        setattr(t, key, value)
    return t


def test_lost_is_refused_by_default() -> None:
    assert not _tracker("LOST")._klt_prior_allowed(0.1)


def test_lost_is_allowed_once_opted_in() -> None:
    assert _tracker("LOST", _klt_lost_predict=True)._klt_prior_allowed(0.1)


def test_shadow_eval_also_opens_lost_so_the_chain_can_be_measured() -> None:
    assert _tracker("LOST", _klt_shadow_eval=True)._klt_prior_allowed(0.1)


@pytest.mark.parametrize("state", ["TRACK", "WEAK_TRACK"])
def test_the_states_that_always_predicted_still_do(state: str) -> None:
    assert _tracker(state)._klt_prior_allowed(0.1)


@pytest.mark.parametrize("state", ["BOOT_INIT", "HOVER_LOCK", ""])
def test_no_opt_in_opens_a_state_that_was_never_allowed(state: str) -> None:
    assert not _tracker(state, _klt_lost_predict=True)._klt_prior_allowed(0.1)


def test_frame_horizon_is_configurable_and_still_bounds() -> None:
    t = _tracker("TRACK", _klt_max_frames=20, _klt_age=19)
    assert t._klt_prior_allowed(0.1)
    t._klt_age = 20
    assert not t._klt_prior_allowed(0.1)


def test_age_horizon_is_configurable_and_still_bounds() -> None:
    t = _tracker("TRACK", _klt_max_age_s=2.0)
    assert t._klt_prior_allowed(1.99)
    assert not t._klt_prior_allowed(2.01)


def test_default_horizon_matches_the_shipped_constants() -> None:
    t = _tracker("TRACK")
    assert t._klt_max_frames == _KLT_MAX_FRAMES
    assert t._klt_max_age_s == _KLT_MAX_AGE_S
    # The shipped 0.5 s cap is only four frames at the measured 8 Hz cadence.
    assert t._klt_max_age_s / 0.1251 < 5.0


def _snapshot(t: ProductionEDMTracker) -> dict:
    return {
        name: getattr(t, name)
        for name in ProductionEDMTracker._SHADOW_KLT_FIELDS
    }


def test_shadow_step_restores_the_production_cache_on_success() -> None:
    t = _tracker("TRACK", _klt_shadow_eval=True)
    t._shadow_klt = {
        "_klt_2d": np.ones((60, 2)), "_klt_3d": np.ones((60, 3)),
        "_klt_gray": np.ones((16, 16), dtype=np.uint8), "_klt_seed_stamp": 0.0,
        "_klt_age": 3, "_klt_center": None, "_klt_yaw": None,
    }
    before = _snapshot(t)
    t._track_klt_prior = lambda _g, _s: {  # type: ignore[method-assign]
        "center": np.array([1.0, 2.0, 3.0]), "yaw": 0.1, "inliers": 40,
        "tracked": 55, "reproj_rms": 1.2,
    }
    info: dict = {}
    t._klt_shadow_step(np.zeros((16, 16), np.uint8), 1.0, np.array([1.0, 2.0, 3.5]), info)

    after = _snapshot(t)
    for name, value in before.items():
        assert np.array_equal(np.asarray(after[name]), np.asarray(value)) or after[name] is value
    assert info["klt_shadow_alive"] is True
    assert info["klt_shadow_error"] == pytest.approx(0.5)


def test_shadow_step_restores_the_production_cache_when_the_chain_dies() -> None:
    t = _tracker("TRACK", _klt_shadow_eval=True)
    t._shadow_klt = dict(_snapshot(t))
    before = _snapshot(t)
    t._track_klt_prior = lambda _g, _s: None  # type: ignore[method-assign]
    info: dict = {}
    t._klt_shadow_step(np.zeros((16, 16), np.uint8), 1.0, np.zeros(3), info)

    after = _snapshot(t)
    for name, value in before.items():
        assert after[name] is value
    assert info["klt_shadow_alive"] is False
    assert t._shadow_klt is None


def test_shadow_step_is_inert_before_the_chain_is_seeded() -> None:
    t = _tracker("TRACK", _klt_shadow_eval=True)
    before = _snapshot(t)
    t._track_klt_prior = lambda _g, _s: pytest.fail(  # type: ignore[method-assign]
        "an unseeded shadow must not run the KLT chain"
    )
    info: dict = {}
    t._klt_shadow_step(np.zeros((16, 16), np.uint8), 1.0, np.zeros(3), info)
    assert _snapshot(t) == before
    assert info == {}


def test_shadow_reseed_does_not_disturb_the_production_cache() -> None:
    t = _tracker("TRACK", _klt_shadow_eval=True)
    before = _snapshot(t)
    seeded: dict = {}

    def _seed(gray, p2d, p3d, mask, stamp):
        t._klt_2d = np.full((60, 2), 7.0)
        t._klt_age = 0
        seeded["called"] = True
        return True

    t._seed_klt_from_inliers = _seed  # type: ignore[method-assign]
    t._klt_shadow_reseed(
        np.zeros((16, 16), np.uint8),
        np.zeros((60, 2)),
        np.zeros((60, 3)),
        np.ones(60, dtype=bool),
        1.0,
    )
    assert seeded["called"]
    assert t._shadow_klt is not None
    after = _snapshot(t)
    for name, value in before.items():
        assert after[name] is value


def test_shadow_reseed_leaves_a_live_chain_alone() -> None:
    t = _tracker("TRACK", _klt_shadow_eval=True)
    live = {"_klt_age": 4}
    t._shadow_klt = live
    t._seed_klt_from_inliers = lambda *_a: pytest.fail(  # type: ignore[method-assign]
        "a live shadow chain must not be re-seeded"
    )
    t._klt_shadow_reseed(
        np.zeros((16, 16), np.uint8),
        np.zeros((60, 2)),
        np.zeros((60, 3)),
        np.ones(60, dtype=bool),
        1.0,
    )
    assert t._shadow_klt is live
