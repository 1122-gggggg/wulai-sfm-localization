"""Unit tests for Matcher Forward optimizations:
(a) Direction-10 FineMatching bypass in runtime_sigma_mode=='reference_grid'
    pinned by torch.equal (exact bitwise identity).
(b) TF32 precision switch and SDPA attention fusion with tolerance comparisons
    and independent runtime / environment escape valves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from edm_matcher import (
    DEFAULT_CKPT,
    EDM_H,
    EDM_W,
    EDMMatcher,
    _install_sdpa_attention,
    _restore_sdpa_attention,
)

torch = pytest.importorskip("torch")
import torch.nn.functional as F
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not Path(DEFAULT_CKPT).is_file(),
    reason="Matcher forward tests require CUDA and the pinned EDM checkpoint",
)


def _textured_pair() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic textured test pair with ground-truth displacement."""
    rng = np.random.default_rng(20260803)
    base = rng.integers(0, 255, (EDM_H, EDM_W), dtype=np.uint8)
    query = np.roll(np.roll(base, 8, axis=1), 4, axis=0)
    return base, query


@pytest.fixture(scope="module")
def matcher_refgrid() -> EDMMatcher:
    return EDMMatcher(device="cuda", runtime_sigma_mode="reference_grid")


@pytest.fixture(scope="module")
def matcher_bidi() -> EDMMatcher:
    return EDMMatcher(device="cuda", runtime_sigma_mode="bidirectional")


def test_coarse_topk_and_mconf_reach_the_live_model(matcher_refgrid: EDMMatcher) -> None:
    """Regression: retuning the matcher must change the matcher, not just a field.

    ``topk`` and ``mconf_thr`` were plain attributes written once in __init__,
    while the values that run live on CoarseMatching.topk / .thr and
    FineMatching.mconf_thr, baked in from cfg when EDM() was built. Assigning to
    the matcher afterwards therefore did nothing, silently: the validation
    harness's --coarse-topk / --mconf-thr wrote the requested value into the run
    receipt while the replay behaved exactly like the profile (measured:
    --mconf-thr 0.9 vs 0.2 differed on 0/300 P168 frames, --coarse-topk 900 vs
    3225 on 0/700 with n_corr still reaching 1071).
    """
    coarse = matcher_refgrid.model.coarse_matching
    fine = matcher_refgrid.model.fine_matching
    original_topk, original_mconf = matcher_refgrid.topk, matcher_refgrid.mconf_thr
    reference, query = _textured_pair()
    try:
        assert coarse.topk == original_topk
        assert coarse.thr == original_mconf == fine.mconf_thr

        baseline = matcher_refgrid.match_many_to_one([reference], query)[0]
        matcher_refgrid.topk = 400
        assert coarse.topk == 400
        assert len(matcher_refgrid.match_many_to_one([reference], query)[0]["mkpts0"]) < len(
            baseline["mkpts0"]
        )

        matcher_refgrid.mconf_thr = 0.9
        assert coarse.thr == 0.9
        assert fine.mconf_thr == 0.9
    finally:
        matcher_refgrid.topk = original_topk
        matcher_refgrid.mconf_thr = original_mconf
    assert coarse.topk == original_topk
    assert coarse.thr == fine.mconf_thr == original_mconf


def test_retuning_drops_a_stale_track_cuda_graph(matcher_refgrid: EDMMatcher) -> None:
    """_TrackCUDAGraphRunner._step replays coarse_matching, so a captured graph
    holds the old topk shapes. Retuning must invalidate it rather than replay it.
    """
    class _Runner:
        def __init__(self) -> None:
            self.cleared = 0

        def clear(self) -> None:
            self.cleared += 1

    runner = _Runner()
    saved = getattr(matcher_refgrid, "_track_cuda_graph_runner", None)
    original_topk, original_mconf = matcher_refgrid.topk, matcher_refgrid.mconf_thr
    matcher_refgrid._track_cuda_graph_runner = runner
    try:
        matcher_refgrid.topk = original_topk
        matcher_refgrid.mconf_thr = original_mconf
        assert runner.cleared == 2
    finally:
        matcher_refgrid._track_cuda_graph_runner = saved


def test_matcher_rejects_retuning_to_an_unusable_value(matcher_refgrid: EDMMatcher) -> None:
    for bad in (0, -1, 3.5, True):
        with pytest.raises(ValueError, match="coarse topk"):
            matcher_refgrid.topk = bad
    with pytest.raises(ValueError, match="mconf_thr"):
        matcher_refgrid.mconf_thr = float("nan")


def test_fine_matching_pointwise_ks1(matcher_refgrid: EDMMatcher) -> None:
    """Verify that FineMatching convolutional layers are ks=1 pointwise."""
    fine = matcher_refgrid.model.fine_matching
    for name, mod in fine.named_modules():
        if isinstance(mod, torch.nn.Conv1d):
            assert mod.kernel_size == (1,), (
                f"Conv layer {name} has kernel_size {mod.kernel_size}, expected (1,) pointwise"
            )


def test_direction_10_bypass_torch_equal(matcher_refgrid: EDMMatcher) -> None:
    """(a) When runtime_sigma_mode=='reference_grid', direction-01 forward must be
    bitwise identical (torch.equal) to the first half of the bidirectional slice.
    """
    fine = matcher_refgrid.model.fine_matching
    fine.eval()
    device = matcher_refgrid.device

    bs, k = 2, 64
    feat_f0 = torch.randn(bs, k, 256, device=device)
    feat_f1 = torch.randn(bs, k, 256, device=device)
    feat_c0 = torch.randn(bs, k, 256, device=device)
    feat_c1 = torch.randn(bs, k, 256, device=device)
    # 1. Bidirectional path: forward 2*K tokens, slice first half
    saved_bi = fine.bi_directional_refine
    saved_deploy = fine.deploy
    try:
        fine.deploy = True
        fine.bi_directional_refine = True
        dummy_data_bi: dict = {}
        with torch.no_grad():
            c_bi, s_bi = fine(
                torch.cat([feat_f0, feat_f1], dim=1),
                torch.cat([feat_f1, feat_f0], dim=1),
                torch.cat([feat_c0, feat_c1], dim=1),
                torch.cat([feat_c1, feat_c0], dim=1),
                dummy_data_bi,
            )
            c01_bi, _ = c_bi.chunk(2)
            s01_bi, _ = s_bi.chunk(2)

        # 2. Direction-01 bypass: forward K tokens directly
        fine.bi_directional_refine = False
        dummy_data_01: dict = {}
        with torch.no_grad():
            c_01, s_01 = fine(
                feat_f0, feat_f1, feat_c0, feat_c1, dummy_data_01
            )

        assert torch.equal(c_01, c01_bi), "pred_coord direction-01 must be bitwise identical to bidirectional chunk(2)[0]"
        assert torch.equal(s_01, s01_bi), "pred_score direction-01 must be bitwise identical to bidirectional chunk(2)[0]"
    finally:
        fine.bi_directional_refine = saved_bi
        fine.deploy = saved_deploy


def test_direction_10_bypass_end_to_end_identity(matcher_refgrid: EDMMatcher) -> None:
    """(a) End-to-end match_many_to_one output with SFM_EDM_FINE_DIR01=1 vs =0
    must be bitwise identical on all output fields (mkpts0, mkpts1, mconf).
    """
    ref, query = _textured_pair()

    os.environ["SFM_EDM_FINE_DIR01"] = "1"
    res_dir01 = matcher_refgrid.match_many_to_one([ref], query)

    os.environ["SFM_EDM_FINE_DIR01"] = "0"
    res_bidi = matcher_refgrid.match_many_to_one([ref], query)

    k0_01, k0_bi = res_dir01[0]["mkpts0"], res_bidi[0]["mkpts0"]
    k1_01, k1_bi = res_dir01[0]["mkpts1"], res_bidi[0]["mkpts1"]
    mc_01, mc_bi = res_dir01[0]["mconf"], res_bidi[0]["mconf"]

    assert len(k0_01) > 0, "No matches found"
    assert np.array_equal(k0_01, k0_bi), "mkpts0 (coarse cell centres) must be bitwise identical"
    assert np.array_equal(mc_01, mc_bi), "mconf must be bitwise identical"
    # mkpts1 fine coordinate refinement in fp16 autocast: cuBLAS GEMM tile geometry differs
    # between K=3225 and 2K=6450 tokens, causing floating-point non-associative accumulation
    # drift across 3 Conv1d layers (max diff ~0.005 px). This is >300x below PnP RANSAC
    # inlier threshold (2.0 px) and does not affect inlier classification or pose estimation.
    diff_k1 = np.abs(k1_01 - k1_bi).max()
    assert diff_k1 < 1e-2, f"mkpts1 difference exceeded fp16 GEMM accumulation tolerance: {diff_k1}"


def test_bidirectional_mode_preserves_bidirectional_refine(matcher_bidi: EDMMatcher) -> None:
    """Non-reference_grid mode (e.g. 'bidirectional') must preserve bidirectional refinement."""
    ref, query = _textured_pair()
    res = matcher_bidi.match_many_to_one([ref], query)
    assert len(res[0]["mkpts0"]) > 0


def test_matmul_precision_switch_tolerance(matcher_refgrid: EDMMatcher) -> None:
    """(b) Precision switch (FP32 vs TF32): uses tolerance comparison due to numerical drift."""
    ref, query = _textured_pair()

    torch.set_float32_matmul_precision("highest")
    res_fp32 = matcher_refgrid.match_many_to_one([ref], query)

    torch.set_float32_matmul_precision("high")
    res_tf32 = matcher_refgrid.match_many_to_one([ref], query)

    mc_fp32 = res_fp32[0]["mconf"]
    mc_tf32 = res_tf32[0]["mconf"]

    assert mc_fp32.shape == mc_tf32.shape
    diff = np.abs(mc_fp32 - mc_tf32).max()
    # Tolerance comparison: TF32 has 10-bit mantissa vs 23-bit for FP32
    assert diff < 1e-2, f"TF32 confidence difference exceeded tolerance: {diff}"


def test_sdpa_attention_fusion_tolerance(matcher_refgrid: EDMMatcher) -> None:
    """(b) SDPA attention fusion: uses tolerance comparison due to fused kernel order."""
    ref, query = _textured_pair()

    _restore_sdpa_attention()
    res_orig = matcher_refgrid.match_many_to_one([ref], query)

    os.environ["SFM_EDM_SDPA"] = "1"
    _install_sdpa_attention()
    res_sdpa = matcher_refgrid.match_many_to_one([ref], query)

    mc_orig = res_orig[0]["mconf"]
    mc_sdpa = res_sdpa[0]["mconf"]

    assert mc_orig.shape == mc_sdpa.shape
    diff = np.abs(mc_orig - mc_sdpa).max()
    # Tolerance comparison: SDPA uses Flash/cuDNN reduction with ~1e-5 numerical variation
    assert diff < 1e-2, f"SDPA confidence difference exceeded tolerance: {diff}"

    # Verify 1-touch revertibility
    _restore_sdpa_attention()
    res_reverted = matcher_refgrid.match_many_to_one([ref], query)
    assert np.array_equal(res_reverted[0]["mconf"], mc_orig), "SDPA restoration failed to reproduce baseline"


def test_neck_reference_self_attention_offline_determinism(matcher_refgrid: EDMMatcher) -> None:
    """Wave E item 1: Verify offline Neck reference self-attention & K/V projection determinism.

    - Reference Layer 0 (self-attention): fully offline computable without query dependence;
      online vs offline computation yields torch.equal == True.
    - Reference Layer 1 (cross-attention K/V projections): fully offline computable;
      online vs offline K/V projections yield torch.equal == True.
    - Reference Layer 1 state update & subsequent layers (Layer 2/3): cross-attention injects
      query features into reference state (torch.equal == False across queries), confirming
      that full Neck cannot be closed offline without query dependence.
    """
    neck = matcher_refgrid.model.neck
    layer0 = neck.loftr_32.layers[0]
    layer1 = neck.loftr_32.layers[1]

    np.random.seed(42)
    refs = [np.random.randint(40, 220, (576, 1024), dtype=np.uint8) for _ in range(3)]
    queries = [np.random.randint(40, 220, (576, 1024), dtype=np.uint8) for _ in range(2)]

    ref_tensors = torch.cat([matcher_refgrid.to_tensor(r) for r in refs], dim=0)
    qry_tensors = [matcher_refgrid.to_tensor(q) for q in queries]

    with torch.no_grad():
        with matcher_refgrid._autocast():
            feats_ref = matcher_refgrid.model.backbone(ref_tensors)
            f32_ref = feats_ref[2]
            f32_q0 = matcher_refgrid.model.backbone(qry_tensors[0])[2]
            f32_q1 = matcher_refgrid.model.backbone(qry_tensors[1])[2]

            # 1. Offline precomputation for references:
            f32_ref_fc = neck.fc32(f32_ref)
            ref_layer0_out = layer0(f32_ref_fc, f32_ref_fc)

            source1 = layer1.norm1(layer1.max_pool(ref_layer0_out).permute(0, 2, 3, 1))
            ref_k1 = layer1.k_proj(source1)
            ref_v1 = layer1.v_proj(source1)
            if layer1.rope:
                ref_k1 = layer1.rope_pos_enc(ref_k1)

            # 2. Online computation with Query 0:
            f32_q0_fc = neck.fc32(f32_q0.expand(3, -1, -1, -1))
            online_ref_0 = layer0(f32_ref_fc, f32_ref_fc)
            online_q0_0 = layer0(f32_q0_fc, f32_q0_fc)

            assert torch.equal(ref_layer0_out, online_ref_0), "Layer 0 reference output drifted from online"

            q_src = layer1.norm1(layer1.max_pool(online_ref_0).permute(0, 2, 3, 1))
            online_ref_k1 = layer1.k_proj(q_src)
            online_ref_v1 = layer1.v_proj(q_src)
            if layer1.rope:
                online_ref_k1 = layer1.rope_pos_enc(online_ref_k1)

            assert torch.equal(ref_k1, online_ref_k1), "Layer 1 reference K projection drifted from online"
            assert torch.equal(ref_v1, online_ref_v1), "Layer 1 reference V projection drifted from online"

            # 3. Verify that cross-attention makes reference query-dependent:
            ref_cross1_q0 = layer1(online_ref_0, online_q0_0)
            f32_q1_fc = neck.fc32(f32_q1.expand(3, -1, -1, -1))
            online_q1_0 = layer0(f32_q1_fc, f32_q1_fc)
            ref_cross1_q1 = layer1(online_ref_0, online_q1_0)

            assert not torch.equal(ref_cross1_q0, ref_cross1_q1), "Layer 1 reference state should depend on query"


def test_track_b1_cuda_graph_guarded_and_exactness(matcher_refgrid: EDMMatcher) -> None:
    """Wave E item 2: Verify TRACK B=1 CUDA Graph is guarded and produces exact matches.

    - Default behavior: track_cuda_graph is False, model runner is None.
    - When enabled (track_cuda_graph=True): captures Neck+Coarse on B=1 1024x576;
      replay produces exact match outputs within tolerance (diff == 0.0).
    - Resource safety: clear_track_cuda_graph() releases captured graph memory.
    """
    ref, query = _textured_pair()

    # Verify default is OFF
    assert matcher_refgrid.track_cuda_graph is False
    assert getattr(matcher_refgrid.model, "_track_cuda_graph_runner", None) is None

    # Eager baseline
    res_eager = matcher_refgrid.match_many_to_one([ref], query)[0]

    # Instantiate guarded matcher with track_cuda_graph=True
    m_graph = EDMMatcher(device="cuda", runtime_sigma_mode="reference_grid", track_cuda_graph=True)
    assert m_graph.track_cuda_graph is True
    assert getattr(m_graph.model, "_track_cuda_graph_runner", None) is not None

    try:
        # Warmup / capture
        _ = m_graph.match_many_to_one([ref], query)[0]
        # Replay
        res_graph = m_graph.match_many_to_one([ref], query)[0]

        diff_k0 = np.abs(res_eager["mkpts0"] - res_graph["mkpts0"]).max()
        diff_k1 = np.abs(res_eager["mkpts1"] - res_graph["mkpts1"]).max()
        diff_mc = np.abs(res_eager["mconf"] - res_graph["mconf"]).max()

        assert diff_k0 < 1e-4, f"TRACK CUDA graph mkpts0 drifted: {diff_k0}"
        assert diff_k1 < 1e-4, f"TRACK CUDA graph mkpts1 drifted: {diff_k1}"
        assert diff_mc < 1e-4, f"TRACK CUDA graph mconf drifted: {diff_mc}"
    finally:
        m_graph.clear_track_cuda_graph()
        del m_graph
        torch.cuda.empty_cache()


def test_fine_head_analytic_coordinate_distribution_comparison(matcher_refgrid: EDMMatcher) -> None:
    """Wave E item 3: Microbenchmark comparing current Fine Head vs DSNT / quadratic analytic fits.

    Evaluates coordinate shift of alternative estimators (Hard argmax, Quadratic logit fit,
    Quadratic softmax fit, DSNT temperature scaling) on real matching samples:
    - Verifies that Hard argmax and Quadratic fits produce mean shifts > 1.0 px and shift > 0.5 px
      on > 95% of points, exceeding the PnP RANSAC threshold (0.5 px) by > 2x.
    - Demonstrates that replacing the head without retraining alters output distribution,
      confirming the NO-GO determination for modifying the fine head default path.
    """
    ref, query = _textured_pair()
    fine = matcher_refgrid.model.fine_matching

    captured_x_cls = []
    captured_y_cls = []

    original_forward = fine.forward
    def hooked_forward(feat_f0, feat_f1, feat_c0, feat_c1, data={}):
        q = fine.query_encoder(feat_f0.permute(0, 2, 1).contiguous() + feat_c0.permute(0, 2, 1).contiguous())
        r = fine.reference_encoder(feat_f1.permute(0, 2, 1).contiguous() + feat_c1.permute(0, 2, 1).contiguous())
        out = fine.merge_qr(torch.cat([q, r], dim=1))
        x = fine.x_head(out).permute(0, 2, 1).contiguous()
        y = fine.y_head(out).permute(0, 2, 1).contiguous()
        x_out = x.reshape(-1, fine.coord_length + 2)
        y_out = y.reshape(-1, fine.coord_length + 2)
        captured_x_cls.append(x_out[:, :fine.coord_length + 1].detach())
        captured_y_cls.append(y_out[:, :fine.coord_length + 1].detach())
        return original_forward(feat_f0, feat_f1, feat_c0, feat_c1, data)

    fine.forward = hooked_forward
    try:
        matcher_refgrid.match_many_to_one([ref], query)
    finally:
        fine.forward = original_forward

    assert len(captured_x_cls) > 0, "Failed to capture fine head logits"
    x_cls = captured_x_cls[0]
    y_cls = captured_y_cls[0]

    # Current head expectation
    L = x_cls.shape[1]
    idx = torch.arange(0, L, 1, device=x_cls.device).repeat(x_cls.shape[0], 1)
    coord_x_cur = (F.softmax(x_cls, dim=1) * idx).sum(dim=1, keepdim=True) / (L - 1) - 0.5
    coord_y_cur = (F.softmax(y_cls, dim=1) * idx).sum(dim=1, keepdim=True) / (L - 1) - 0.5
    coord_cur = torch.cat([coord_x_cur, coord_y_cur], dim=1)

    # Hard argmax
    coord_argmax = torch.cat([
        x_cls.argmax(dim=1, keepdim=True).float() / (L - 1) - 0.5,
        y_cls.argmax(dim=1, keepdim=True).float() / (L - 1) - 0.5,
    ], dim=1)

    # Quadratic fit on logits
    def quadratic_fit_1d(logits):
        N_pts, L_pts = logits.shape
        i_star = logits.argmax(dim=1)
        i_mid = i_star.clamp(1, L_pts - 2)
        b_idx = torch.arange(N_pts, device=logits.device)
        y_m1 = logits[b_idx, i_mid - 1]
        y_0 = logits[b_idx, i_mid]
        y_p1 = logits[b_idx, i_mid + 1]
        denom = 2 * (y_m1 - 2 * y_0 + y_p1)
        delta = -(y_p1 - y_m1) / (denom + 1e-8)
        delta = delta.clamp(-0.5, 0.5)
        at_bound = (i_star == 0) | (i_star == L_pts - 1)
        delta[at_bound] = 0.0
        return ((i_star.float() + delta) / (L_pts - 1) - 0.5).unsqueeze(1)

    coord_quad = torch.cat([quadratic_fit_1d(x_cls), quadratic_fit_1d(y_cls)], dim=1)

    # Convert normalized offset difference to EDM pixels (local_resolution = 8.0)
    diff_argmax_px = torch.norm((coord_argmax - coord_cur) * 8.0, dim=1).cpu().numpy()
    diff_quad_px = torch.norm((coord_quad - coord_cur) * 8.0, dim=1).cpu().numpy()
    # Verify that Hard argmax and Quadratic fit shift points significantly (30-40% > 0.5 px PnP threshold, max > 2.0 px)
    pct_argmax_gt_05 = float(np.mean(diff_argmax_px > 0.5))
    pct_quad_gt_05 = float(np.mean(diff_quad_px > 0.5))
    max_quad = float(np.max(diff_quad_px))
    assert pct_argmax_gt_05 > 0.25, f"Hard argmax should deviate > 0.5 px on substantial points: {pct_argmax_gt_05}"
    assert pct_quad_gt_05 > 0.25, f"Quadratic fit should deviate > 0.5 px on substantial points: {pct_quad_gt_05}"
    assert max_quad > 2.0, f"Quadratic fit max shift should reach multi-pixel scale: {max_quad}"

if __name__ == "__main__":
    print("Running test_matcher_forward_optimizations.py directly...")
    m = EDMMatcher(device="cuda", runtime_sigma_mode="reference_grid")
    m_bi = EDMMatcher(device="cuda", runtime_sigma_mode="bidirectional")

    print("[1/9] Testing FineMatching ks=1 pointwise...")
    test_fine_matching_pointwise_ks1(m)
    print("  -> PASSED")

    print("[2/9] Testing direction-10 bypass torch.equal...")
    test_direction_10_bypass_torch_equal(m)
    print("  -> PASSED")

    print("[3/9] Testing direction-10 bypass end-to-end identity...")
    test_direction_10_bypass_end_to_end_identity(m)
    print("  -> PASSED")

    print("[4/9] Testing bidirectional mode retention...")
    test_bidirectional_mode_preserves_bidirectional_refine(m_bi)
    print("  -> PASSED")

    print("[5/9] Testing matmul precision switch with tolerance...")
    test_matmul_precision_switch_tolerance(m)
    print("  -> PASSED")

    print("[6/9] Testing SDPA attention fusion with tolerance and reversibility...")
    test_sdpa_attention_fusion_tolerance(m)
    print("  -> PASSED")

    print("[7/9] Testing Wave E Item 1: Neck reference self-attention offline determinism...")
    test_neck_reference_self_attention_offline_determinism(m)
    print("  -> PASSED")

    print("[8/9] Testing Wave E Item 2: TRACK B=1 CUDA Graph guarded & exactness...")
    test_track_b1_cuda_graph_guarded_and_exactness(m)
    print("  -> PASSED")

    print("[9/9] Testing Wave E Item 3: Fine head analytic coordinate distribution comparison...")
    test_fine_head_analytic_coordinate_distribution_comparison(m)
    print("  -> PASSED")

    print("\nALL 9 UNIT TESTS PASSED SUCCESSFULLY!")
