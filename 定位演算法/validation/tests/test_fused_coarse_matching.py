"""The fused EDM coarse tail must select the same matches as the upstream one.

Upstream materialises a [B, L, L] fp32 confidence matrix and reads it twice more
(row max, then a gather for mconf). edm_matcher patches that into a single
torch.compile'd reduction. This is the only test in the repo that runs the real
EDM coarse head, so it is also the guard against the fp16 dual-softmax overflow
(exp -> +inf -> every mconf passes the threshold -> broken poses).

Needs CUDA and the pinned EDM checkpoint; skipped otherwise.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import edm_matcher as edm_matcher_module  # noqa: E402
from edm_matcher import DEFAULT_CKPT, EDM_H, EDM_W, EDMMatcher  # noqa: E402

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Path(DEFAULT_CKPT).is_file(),
    reason="fused EDM coarse tail needs CUDA and the pinned EDM checkpoint",
)


def _textured_pair() -> tuple[np.ndarray, np.ndarray]:
    """A reference and a shifted copy, so real correspondences exist."""
    import cv2

    rng = np.random.default_rng(20260803)
    base = rng.integers(0, 255, (EDM_H, EDM_W), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.7)
    query = np.roll(np.roll(base, 8, axis=1), 4, axis=0)
    return base, query


@pytest.fixture(scope="module")
def matcher() -> EDMMatcher:
    return EDMMatcher()


def _run(matcher: EDMMatcher, ref: np.ndarray, query: np.ndarray, batch: int):
    matcher._reference_tensor_cache.clear()
    return matcher.match_many_to_one([ref] * batch, query)


def _upstream(matcher: EDMMatcher, ref, query, batch):
    state = edm_matcher_module._FUSED_COARSE_STATE
    coarse_cls = state["coarse_class"]
    fused_forward = coarse_cls.forward
    coarse_cls.forward = state["original_forward"]
    try:
        return _run(matcher, ref, query, batch)
    finally:
        coarse_cls.forward = fused_forward


def test_patch_is_installed_and_reversible(matcher: EDMMatcher) -> None:
    state = edm_matcher_module._FUSED_COARSE_STATE
    assert state["installed"] is True
    assert state["reason"] == "fused coarse tail active", state["reason"]
    assert state["original_forward"] is not None
    assert state["coarse_class"] is not None
    assert state["coarse_class"].forward is not state["original_forward"]


@pytest.mark.parametrize("batch", [1, 2])
def test_fused_tail_selects_the_same_match_set(matcher: EDMMatcher, batch: int) -> None:
    ref, query = _textured_pair()
    fused = _run(matcher, ref, query, batch)
    upstream = _upstream(matcher, ref, query, batch)

    assert len(fused) == len(upstream) == batch
    for slot, (a, b) in enumerate(zip(upstream, fused)):
        assert len(a["mkpts0"]) == len(b["mkpts0"]), f"match count differs in slot {slot}"
        assert len(a["mkpts0"]) > 0, "degenerate pair produced no matches"
        for key in ("mkpts0", "mkpts1"):
            up = set(map(tuple, np.round(a[key].astype(np.float64), 4)))
            fu = set(map(tuple, np.round(b[key].astype(np.float64), 4)))
            assert up == fu, (
                f"{key} differs in slot {slot}: "
                f"only_upstream={len(up - fu)} only_fused={len(fu - up)}"
            )
        # Order may differ where confidences tie, so compare the sorted spectrum.
        up_conf = np.sort(a["mconf"].astype(np.float64))
        fu_conf = np.sort(b["mconf"].astype(np.float64))
        assert np.allclose(up_conf, fu_conf, atol=1e-5), (
            f"mconf spectrum differs in slot {slot}: "
            f"max|diff|={np.abs(up_conf - fu_conf).max():.3e}"
        )


@pytest.mark.parametrize("batch", [1, 2])
def test_confidences_stay_finite_and_bounded(matcher: EDMMatcher, batch: int) -> None:
    """Guards the fp16 dual-softmax overflow: exp() must run in fp32."""
    ref, query = _textured_pair()
    for result in _run(matcher, ref, query, batch):
        mconf = result["mconf"]
        assert np.isfinite(mconf).all(), "non-finite mconf -> fp16 exp() overflow"
        assert (mconf >= 0.0).all() and (mconf <= 1.0 + 1e-6).all()


def test_batch_ids_stay_contiguous_and_equal_length(matcher: EDMMatcher) -> None:
    """edm.py reshapes with K = i_ids.size // bs, so each batch row needs the same K."""
    ref, query = _textured_pair()
    captured: dict = {}
    state = edm_matcher_module._FUSED_COARSE_STATE
    coarse_cls = state["coarse_class"]
    patched = coarse_cls.forward

    def spy(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        out = patched(self, feat_c0, feat_c1, data, mask_c0, mask_c1)
        captured["b_ids"] = data["b_ids"].detach().cpu().numpy()
        captured["i_ids"] = data["i_ids"].detach().cpu().numpy()
        return out

    coarse_cls.forward = spy
    try:
        _run(matcher, ref, query, 2)
    finally:
        coarse_cls.forward = patched

    b_ids = captured["b_ids"]
    assert b_ids.size == captured["i_ids"].size
    counts = np.bincount(b_ids, minlength=2)
    assert counts[0] == counts[1], "batch rows must carry the same match count"
    assert np.array_equal(b_ids, np.repeat([0, 1], counts[0])), "b_ids must be batch-major"


def test_fused_tail_is_not_slower(matcher: EDMMatcher) -> None:
    ref, query = _textured_pair()

    def timed(fn, iters=6):
        fn()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000.0 / iters

    fused_ms = timed(lambda: _run(matcher, ref, query, 1))
    upstream_ms = timed(lambda: _upstream(matcher, ref, query, 1))
    assert fused_ms < upstream_ms, (
        f"fused {fused_ms:.1f} ms is not faster than upstream {upstream_ms:.1f} ms"
    )


def test_disable_flag_keeps_the_upstream_tail(monkeypatch) -> None:
    """SFM_EDM_FUSED_COARSE=0 must be an honest escape hatch."""
    monkeypatch.setenv("SFM_EDM_FUSED_COARSE", "0")
    saved = dict(edm_matcher_module._FUSED_COARSE_STATE)
    edm_matcher_module._FUSED_COARSE_STATE.update(
        installed=False, compiled=None, reason="")
    try:
        reason = edm_matcher_module._install_fused_coarse_matching()
        assert reason == "disabled by SFM_EDM_FUSED_COARSE=0"
        assert edm_matcher_module._FUSED_COARSE_STATE["compiled"] is None
    finally:
        edm_matcher_module._FUSED_COARSE_STATE.clear()
        edm_matcher_module._FUSED_COARSE_STATE.update(saved)
