#!/usr/bin/env python3
"""EDM (ICCV'25, Efficient Deep feature Matching) wrapper for map build + relocalization.

Replaces the XFeat + LighterGlue + mutual-NN local matcher. EDM is detector-free:
there are no reusable per-image descriptors, so both the map build and the runtime
localizer must feed it the actual reference IMAGE.

THE CELL TRICK (why this module can drop the KD-tree / depth-map machinery):
  EDM matches at a 1/8 coarse grid, then regresses a subpixel offset that
  `fine_matching.final_matching_selection` CLAMPS to +/- LOCAL_RESOLUTION/2 = +/-4px
  (fine_matching.py:313-319). Every surviving match therefore stays inside the coarse
  cell it was born in, on BOTH sides, whichever direction won the sigma selection.
  => cell = round(kpt / 8) exactly recovers the coarse cell for any match.

  The cell index is a deterministic, image-intrinsic keypoint identity that is stable
  across pairs -- which is exactly what a detector-free matcher normally lacks. The map
  build keys triangulated 3D by cell; the localizer looks 3D up by cell in O(1).

  Consequence for the localizer: pass the REFERENCE as image0 and the QUERY as image1.
  The reference side is then only ever used as an identity (cell -> xyz), and the query
  side keeps its full subpixel precision for PnP.

Input contract: grayscale, /255, (B,1,H,W), H and W divisible by 32.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
import hashlib
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from artifact_integrity import verify_sha256


def _find_edm_repo() -> Path:
    """The EDM checkout sits at third_party/EDM in the dev tree and runtime/EDM in the
    transfer package. $EDM_REPO overrides both."""
    if os.environ.get("EDM_REPO"):
        return Path(os.environ["EDM_REPO"])
    root = Path(__file__).resolve().parent.parent
    for rel in ("third_party/EDM", "runtime/EDM"):
        if (root / rel / "src").is_dir():
            return root / rel
    raise FileNotFoundError(f"no EDM checkout under {root} (looked in third_party/, runtime/)")


EDM_REPO = _find_edm_repo()
DEFAULT_CKPT = EDM_REPO / "weights" / "edm_outdoor.ckpt"
DEFAULT_CKPT_SHA256 = "f686bebdd9705bf6918621a1a83695f83d698cbd8c3eed932847fe3678d13a97"
DEFAULT_CFG = EDM_REPO / "configs" / "edm" / "outdoor" / "edm_base.py"

# 1280x720 (the map camera) -> 1024x576: exactly 16:9, both sides divisible by 32,
# so no padding, no mask, no aspect distortion. Coarse grid is 128x72.
EDM_W, EDM_H = 1024, 576
COARSE_STRIDE = 8  # EDM.LOCAL_RESOLUTION
GRID_W, GRID_H = EDM_W // COARSE_STRIDE, EDM_H // COARSE_STRIDE  # 128 x 72
REFERENCE_FEATURE_STORE_SCHEMA = 1
REFERENCE_FEATURE_SHARD_SCHEMA = 2
REFERENCE_FEATURE_SHARD_ENTRIES = 64
# The all-eight-video River map has 1,045 references (~8.4 GiB of FP16 levels).
REFERENCE_FEATURE_STORE_MAX_BYTES = 12 * 1024 * 1024 * 1024
# PyTorch's ZIP writer can produce an unreadable archive once tensor data crosses
# the 32-bit ZIP boundary. Large banks use its legacy stream format instead.
REFERENCE_FEATURE_STORE_ZIP_MAX_BYTES = 3 * 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Fused coarse tail
#
# Upstream's coarse head (src/edm/head/coarse_matching.py) builds a [B, L, L]
# confidence matrix -- 9216x9216 fp32 = 324 MiB per batch element at 1024x576 --
# through four separate kernels (exp, normalize dim=1, normalize dim=2, product)
# and then reads it a fifth time for the row-wise max. EDM is detector-free, so
# it pays a full forward per reference and that tail is a large slice of the
# per-frame budget.
#
# torch.compile fuses exp + dual-softmax + row-max into one reduction, so the
# matrix is never materialised. Measured on an RTX 5060 Laptop with real 720p
# frames at 1024x576, coarse topk 3225:
#
#   b=1  (TRACK, one reference)     41.52 ms -> 25.57 ms   peak 1457 -> 494 MiB
#   b=2  (WEAK/LOST reference pair) 89.50 ms -> 57.36 ms
#
# The selected match SET is identical to upstream (verified: 3095/3095 common,
# 0 only-upstream, 0 only-fused; per-row argmax column 0/3225 disagreements;
# mconf max|diff| 4.8e-07). Only the torch.topk tie-break ORDER differs, among
# confidences that agree to ~5e-7. Downstream that reaches exactly two places,
# both already order- or seed-dependent: the np.lexsort tiebreak inside
# spatially_cap_indices (binds only when n_corr exceeds max_corr_total) and the
# RANSAC sample draw (pnp_ransac_random_seed=-1). See
# 定位演算法/validation/tests/test_fused_coarse_matching.py.
#
# Set SFM_EDM_FUSED_COARSE=0 to fall back to the upstream tail.
# ---------------------------------------------------------------------------
_FUSED_COARSE_STATE: dict = {
    "installed": False,
    "compiled": None,
    "reason": "",
    # Kept so an A/B test can restore the upstream tail in-process.
    "original_forward": None,
    "coarse_class": None,
}


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    """A malformed operational knob must not stop a flight from booting."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[edm_matcher] ignoring non-integer {name}={raw!r}", file=sys.stderr, flush=True)
        return default


def _inductor_cache_dir() -> Path | None:
    """Keep compiled kernels in the workspace so the cost is paid once, offline."""
    configured = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        runtime = parent / "執行環境"
        if runtime.is_dir():
            return runtime / "inductor_cache"
    return None


def _runtime_cache_root() -> Path | None:
    configured = os.environ.get("SFM_EDM_REFERENCE_FEATURE_STORE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    for parent in Path(__file__).resolve().parents:
        runtime = parent / "執行環境"
        if runtime.is_dir():
            return runtime / "models" / "edm_reference_features"
    return None


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _dual_softmax_rowmax(sim: torch.Tensor):
    """exp -> L1 dual-softmax -> row max, in fp32, without storing the matrix.

    fp32 is explicit rather than inherited from the caller's autocast state:
    exp() on the fp16 similarity overflows to +inf, which makes every mconf
    pass the threshold and produces broken poses (the same failure the ONNX
    FP16 backend hit, see edm_onnx_matcher.py).
    """
    e = torch.exp(sim.to(torch.float32))
    d1 = e.sum(dim=1, keepdim=True).clamp_min(1e-12)
    d2 = e.sum(dim=2, keepdim=True).clamp_min(1e-12)
    conf = (e / d1) * (e / d2)
    return torch.max(conf, dim=2)


def _make_fused_forward(original_forward, compiled_tail):
    """Build a CoarseMatching.forward that skips the confidence-matrix buffer."""

    def fused_forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        # Training needs conf_matrix for the loss; deploy=True runs its own
        # duplicated tail in edm.py (that is what ONNX export traces); ds_opt=False
        # selects the native dual softmax; masks only appear with padded batches.
        # All four keep the upstream path.
        if (
            self.training
            or self.deploy
            or not self.ds_opt
            or mask_c0 is not None
            or mask_c1 is not None
        ):
            return original_forward(self, feat_c0, feat_c1, data, mask_c0, mask_c1)

        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1])
        with torch.autocast(enabled=False, device_type="cuda"):
            if _env_flag("SFM_EDM_COARSE_BMM", True):
                sim_matrix = torch.bmm(feat_c0, feat_c1.transpose(1, 2)) / self.temperature
            else:
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        row_max_val, row_max_idx = compiled_tail(sim_matrix)
        del sim_matrix
        k = self.topk
        if k == -1 or k > row_max_val.shape[-1]:
            k = row_max_val.shape[-1]
        # sorted=True: the descending order is the last tie-break feeding
        # spatially_cap_indices, so keep it identical to upstream.
        topk_val, topk_idx = torch.topk(row_max_val, k)
        b_ids = (
            torch.arange(row_max_val.shape[0], device=row_max_val.device)
            .unsqueeze(1)
            .repeat(1, k)
            .flatten()
        )
        i_ids = topk_idx.flatten()
        j_ids = row_max_idx[b_ids, i_ids].flatten()
        # conf_matrix[b, i, row_max_idx[b, i]] IS row_max_val[b, i], and topk_val
        # is exactly that gathered at topk_idx -- so the upstream gather back into
        # the full matrix is redundant, not merely expensive.
        mconf = topk_val.flatten()

        scale = data["hw0_i"][0] / data["hw0_c"][0]
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
        mkpts0_c = (
            torch.stack(
                [
                    i_ids % data["hw0_c"][1],
                    torch.div(i_ids, data["hw0_c"][1], rounding_mode="floor"),
                ],
                dim=1,
            )
            * scale0
        )
        mkpts1_c = (
            torch.stack(
                [
                    j_ids % data["hw1_c"][1],
                    torch.div(j_ids, data["hw1_c"][1], rounding_mode="floor"),
                ],
                dim=1,
            )
            * scale1
        )
        data.update(
            {
                "mconf": mconf,
                "mkpts0_c": mkpts0_c,
                "mkpts1_c": mkpts1_c,
                "b_ids": b_ids,
                "i_ids": i_ids,
                "j_ids": j_ids,
            }
        )
        return None

    return fused_forward


def _install_fused_coarse_matching() -> str:
    """Patch CoarseMatching in place. Idempotent; never raises into the caller.

    Called from _import_edm() rather than at module import: edm_onnx_matcher
    imports this module at module scope, and importing an ONNX backend must not
    drag in the EDM source tree or set up a compiler.
    """
    if _FUSED_COARSE_STATE["installed"]:
        return _FUSED_COARSE_STATE["reason"]
    if not _env_flag("SFM_EDM_FUSED_COARSE", True):
        _FUSED_COARSE_STATE.update(installed=True, reason="disabled by SFM_EDM_FUSED_COARSE=0")
        return _FUSED_COARSE_STATE["reason"]
    try:
        cache_dir = _inductor_cache_dir()
        if cache_dir is not None and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        from src.edm.head.coarse_matching import CoarseMatching

        compiled = torch.compile(_dual_softmax_rowmax, dynamic=False)
        original_forward = CoarseMatching.forward
        CoarseMatching.forward = _make_fused_forward(original_forward, compiled)
        _FUSED_COARSE_STATE.update(
            installed=True,
            compiled=compiled,
            reason="fused coarse tail active",
            original_forward=original_forward,
            coarse_class=CoarseMatching,
        )
    except Exception as exc:  # keep the upstream tail on any failure
        _FUSED_COARSE_STATE.update(
            installed=True, compiled=None, reason=f"fused coarse tail unavailable: {exc!r}"
        )
    return _FUSED_COARSE_STATE["reason"]


_SDPA_STATE: dict = {
    "installed": False,
    "reason": "",
    "original_attention": None,
    "attention_class": None,
}


def _install_sdpa_attention() -> str:
    """Patch Attention in LocalFeatureTransformer to use F.scaled_dot_product_attention.

    Controlled by SFM_EDM_SDPA (default: 0 / off).
    Revertible at runtime via _restore_sdpa_attention() or via SFM_EDM_SDPA=0.
    """
    if _SDPA_STATE["installed"]:
        return _SDPA_STATE["reason"]
    if not _env_flag("SFM_EDM_SDPA", False):
        _SDPA_STATE.update(installed=True, reason="disabled by SFM_EDM_SDPA=0")
        return _SDPA_STATE["reason"]
    try:
        from src.edm.neck.loftr_module.transformer import Attention

        orig_attention = Attention.attention

        def sdpa_attention(self, query, key, value, q_mask=None, kv_mask=None):
            if q_mask is not None or kv_mask is not None:
                return orig_attention(self, query, key, value, q_mask=q_mask, kv_mask=kv_mask)
            query = F.normalize(query, p=2, dim=3)
            key = F.normalize(key, p=2, dim=3)
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v, scale=20.0)
            return out.permute(0, 2, 1, 3).contiguous()

        Attention.attention = sdpa_attention
        _SDPA_STATE.update(
            installed=True,
            reason="SDPA attention active",
            original_attention=orig_attention,
            attention_class=Attention,
        )
    except Exception as exc:
        _SDPA_STATE.update(
            installed=True, reason=f"SDPA attention unavailable: {exc!r}"
        )
    return _SDPA_STATE["reason"]


def _restore_sdpa_attention() -> None:
    """Restore the unpatched Attention.attention."""
    orig = _SDPA_STATE.get("original_attention")
    cls_ = _SDPA_STATE.get("attention_class")
    if orig is not None and cls_ is not None:
        cls_.attention = orig
        _SDPA_STATE["installed"] = False
        _SDPA_STATE["reason"] = "restored"

def _import_edm():
    if str(EDM_REPO) not in sys.path:
        sys.path.insert(0, str(EDM_REPO))
    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config

    _install_fused_coarse_matching()
    _install_sdpa_attention()
    return get_cfg_defaults, EDM, lower_config


def _load_edm_state_dict(
    ckpt: str | Path,
    expected_sha256: str | None = None,
) -> Mapping:
    checkpoint_path = Path(ckpt).expanduser().resolve()
    verify_sha256(checkpoint_path, expected_sha256)
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("EDM checkpoint must contain a mapping")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("EDM checkpoint has no non-empty state_dict mapping")
    return state


def _checkpoint_sha256_for_load(
    ckpt: str | Path,
    expected_sha256: str | None,
) -> str | None:
    if expected_sha256:
        return expected_sha256
    if Path(ckpt).expanduser().resolve() == DEFAULT_CKPT.expanduser().resolve():
        return DEFAULT_CKPT_SHA256
    return None


# ---------------------------------------------------------------------------
# Reference backbone cache
#
# EDM extracts features with ONE backbone call over cat([image0, image1])
# (edm.py:56). The localizer always passes map references as image0 and the same
# query tiled as image1 (match_many_to_one), so every frame recomputes:
#   - the reference features, which depend only on immutable map imagery, and
#   - b-1 duplicate copies of the query.
# BatchNorm runs on stored statistics in eval(), so a reference's features are
# image-local and safe to keep. Caching them and extracting the query once
# removes both redundancies.
#
# Measured on an RTX 5060 Laptop with real map imagery, bracketed A/B/A
# (2.5-3.5% drift between brackets), production chunking at match_batch_size=2:
#
#   TRACK, 1 reference    27.17 ms -> 22.90 warm (-15.7%), 28.70 cold (+5.6%)
#   WEAK,  3 references   87.67 ms -> 69.56 warm (-20.7%), 86.05 cold (-1.8%)
#
# TRACK therefore needs a 26% hit rate to break even; WEAK is not slower even
# when every reference misses. Two details carry that:
#   - every image is extracted in a batch of ONE. Batching the misses together
#     with the query is measurably faster on a cold frame (cold TRACK becomes
#     upstream-identical, +2.0%), but then a frame's matches depend on how many
#     references happened to be cached, and match_many_to_one stops being a pure
#     function of its inputs. Determinism is worth more here than 5.6% on the
#     minority of frames that miss.
#   - the assembled levels are restored to channels_last. Concatenating an
#     expanded query silently produces NCHW, and feeding that to the neck cost
#     more than the cache saved (cold TRACK was +26% before this).
#
# Against upstream the warm path still moves ~1% of anchor cells and shifts the
# shared ones by ~0.005 px on average, because the query is extracted alone
# rather than beside the reference. That is the same fp16 batch-shape sensitivity
# upstream already has between TRACK (b=1) and WEAK (b=2), and it is why this
# needs a replay A/B, not just a unit test.
#
# Cost is 8.16 MiB of fp16 features per reference, so the validated default 192
# entries hold about 1.53 GiB. Entries are keyed like reference_tensor(): by id() with the
# source object retained, so an id cannot be recycled behind the cache. A
# temporal reference (use_temporal_reference, off in every shipped profile) is
# a per-frame array and would occupy an entry until evicted.
#
# Set SFM_EDM_REF_FEATURE_CACHE=0 to disable, or to another size to retune.
# ---------------------------------------------------------------------------
DEFAULT_REF_FEATURE_CACHE = 192
DEFAULT_HOST_REF_FEATURE_CACHE = 0
DEFAULT_TEMPORAL_FEATURE_CACHE = 0
RUNTIME_SIGMA_MODES = ("bidirectional", "reference_grid")
FEATURE_CACHE_KINDS = ("map", "temporal")
MAX_MATCHER_CACHE_BYTES = 8 * 1024**3
MAX_HOST_FEATURE_CACHE_BYTES = 4 * 1024**3
# Conservative vs the measured 8.16 MiB fp16 feature blob per reference.
FEATURE_CACHE_BYTES_PER_ENTRY = 9 * 1024 * 1024
TENSOR_CACHE_BYTES_PER_ENTRY = EDM_W * EDM_H * 4


def _non_negative_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_matcher_cache_budget(
    *,
    map_feature_entries: int,
    temporal_feature_entries: int,
    map_tensor_entries: int,
    temporal_tensor_entries: int,
    host_feature_entries: int = 0,
) -> None:
    total = (map_feature_entries + temporal_feature_entries) * FEATURE_CACHE_BYTES_PER_ENTRY + (
        map_tensor_entries + temporal_tensor_entries
    ) * TENSOR_CACHE_BYTES_PER_ENTRY
    if total > MAX_MATCHER_CACHE_BYTES:
        raise ValueError("EDM matcher cache exceeds the 8 GiB bound")
    if host_feature_entries * FEATURE_CACHE_BYTES_PER_ENTRY > MAX_HOST_FEATURE_CACHE_BYTES:
        raise ValueError("EDM pinned host feature cache exceeds the 4 GiB bound")


def select_reference_grid_matches(fine, data) -> None:
    """Keep direction-01 (reference on its coarse cell) and apply confidence filters.

    Map-build keeps the upstream bidirectional sigma winner. Runtime lookup can
    only consume reference-grid anchors, so a higher direction-10 sigma must not
    discard a usable direction-01 correspondence.
    """
    offset = data["pred_coord"] * fine.local_resolution
    clamped = torch.clamp(offset, -fine.local_resolution / 2, fine.local_resolution / 2)
    if fine.bi_directional_refine and clamped.shape[0] == 2 * data["b_ids"].shape[0]:
        fine_offset01, _fine_offset10 = clamped.chunk(2)
        pred_score01, _pred_score10 = data["pred_score"].chunk(2)
    else:
        fine_offset01 = clamped
        pred_score01 = data["pred_score"]

    h0, w0 = data["hw0_i"]
    h1, w1 = data["hw1_i"]
    scale0 = data["scale0"][data["b_ids"]] if "scale0" in data else 1.0
    scale1 = data["scale1"][data["b_ids"]] if "scale1" in data else 1.0
    scale0_w = scale0[:, 0] if "scale0" in data else 1.0
    scale0_h = scale0[:, 1] if "scale0" in data else 1.0
    scale1_w = scale1[:, 0] if "scale1" in data else 1.0
    scale1_h = scale1[:, 1] if "scale1" in data else 1.0

    mkpts0_f = data["mkpts0_c"]
    mkpts1_f = data["mkpts1_c"] + fine_offset01 * scale1
    mconf = data["mconf"]
    mask = (
        torch.isfinite(mconf)
        & (mconf > fine.mconf_thr)
        & torch.isfinite(pred_score01)
        & (pred_score01 > fine.sigma_thr)
        & torch.isfinite(mkpts0_f).all(dim=1)
        & torch.isfinite(mkpts1_f).all(dim=1)
        & (mkpts0_f[:, 0] >= fine.border_rm)
        & (mkpts0_f[:, 0] <= w0 * scale0_w - fine.border_rm)
        & (mkpts0_f[:, 1] >= fine.border_rm)
        & (mkpts0_f[:, 1] <= h0 * scale0_h - fine.border_rm)
        & (mkpts1_f[:, 0] >= fine.border_rm)
        & (mkpts1_f[:, 0] <= w1 * scale1_w - fine.border_rm)
        & (mkpts1_f[:, 1] >= fine.border_rm)
        & (mkpts1_f[:, 1] <= h1 * scale1_h - fine.border_rm)
    )
    data.update(
        {
            "m_bids": data["b_ids"][mask],
            "mkpts0_f": mkpts0_f[mask],
            "mkpts1_f": mkpts1_f[mask],
            "mconf": mconf[mask],
        }
    )

class _TrackCUDAGraphRunner:
    """Fixed-shape (B=1, 1024x576) CUDA Graph runner for Neck + Coarse matching.

    Guarded by SFM_EDM_TRACK_CUDAGRAPH (default: False / off).
    WARNING: Capturing Neck + Coarse in a CUDAGraph creates private memory pools
    that retain gigabytes of VRAM (measured ~3.4 GiB on RTX 5060). On 8GB GPUs,
    this risks OutOfMemoryError when running alongside persistent feature caches.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.graph = None
        self.static_f8 = None
        self.static_f16 = None
        self.static_f32 = None
        self.static_outputs = None
        self.capture_stream = None
        self.captured_hw = None

    def clear(self) -> None:
        self.graph = None
        self.static_f8 = None
        self.static_f16 = None
        self.static_f32 = None
        self.static_outputs = None
        self.capture_stream = None
        self.captured_hw = None

    def _step(self, f8, f16, f32, data):
        ms_feats = (f8, f16, f32)
        feat_c0, feat_c1 = self.model.neck(ms_feats)
        feat_c0_flat = feat_c0.flatten(2).permute(0, 2, 1)
        feat_c1_flat = feat_c1.flatten(2).permute(0, 2, 1)
        data.update({
            "hw0_c": feat_c0.shape[2:],
            "hw1_c": feat_c1.shape[2:],
            "hw0_f": feat_c0.shape[2:] * self.model.config["local_resolution"],
            "hw1_f": feat_c1.shape[2:] * self.model.config["local_resolution"],
        })
        self.model.coarse_matching(feat_c0_flat, feat_c1_flat, data)
        return (
            feat_c0_flat,
            feat_c1_flat,
            data["mconf"],
            data["mkpts0_c"],
            data["mkpts1_c"],
            data["b_ids"],
            data["i_ids"],
            data["j_ids"],
        )

    def forward(self, ms_feats, data):
        f8, f16, f32 = ms_feats
        if self.graph is None:
            self.static_f8 = f8.clone()
            self.static_f16 = f16.clone()
            self.static_f32 = f32.clone()
            self.capture_stream = torch.cuda.Stream(device=f8.device)
            current_stream = torch.cuda.current_stream(device=f8.device)
            self.capture_stream.wait_stream(current_stream)

            with torch.cuda.stream(self.capture_stream):
                for _ in range(3):
                    d = dict(data)
                    _ = self._step(self.static_f8, self.static_f16, self.static_f32, d)
            current_stream.wait_stream(self.capture_stream)

            self.graph = torch.cuda.CUDAGraph()
            d_cap = dict(data)
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self.static_outputs = self._step(
                    self.static_f8, self.static_f16, self.static_f32, d_cap
                )
            current_stream.wait_stream(self.capture_stream)
            self.captured_hw = (d_cap["hw0_c"], d_cap["hw1_c"], d_cap["hw0_f"], d_cap["hw1_f"])

        self.static_f8.copy_(f8)
        self.static_f16.copy_(f16)
        self.static_f32.copy_(f32)
        self.graph.replay()
        c0, c1, mconf, mk0, mk1, b_ids, i_ids, j_ids = self.static_outputs
        hw0_c, hw1_c, hw0_f, hw1_f = self.captured_hw
        data.update({
            "hw0_c": hw0_c,
            "hw1_c": hw1_c,
            "hw0_f": hw0_f,
            "hw1_f": hw1_f,
            "mconf": mconf,
            "mkpts0_c": mk0,
            "mkpts1_c": mk1,
            "b_ids": b_ids,
            "i_ids": i_ids,
            "j_ids": j_ids,
        })
        return c0, c1


class EDMMatcher:
    """Detector-free matcher. Coordinates in/out are EDM-input pixels (1024x576)."""

    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        cfg_path: str | Path = DEFAULT_CFG,
        mconf_thr: float = 0.2,
        topk: int | None = None,
        border_rm: int = 2,
        device: str = "cuda",
        fp16: bool = True,
        reference_cache_size: int = 32,
        ckpt_sha256: str | None = None,
        reference_feature_cache_size: int | None = None,
        host_reference_feature_cache_size: int | None = None,
        temporal_feature_cache_size: int = DEFAULT_TEMPORAL_FEATURE_CACHE,
        runtime_sigma_mode: str = "bidirectional",
        temporal_feature_promotion: bool | None = None,
        inclusive_host_feature_cache: bool | None = None,
        input_size: tuple[int, int] | None = None,
        query_cuda_graph: bool = False,
        track_cuda_graph: bool | None = None,
    ):
        # fp16 autocast + channels_last: 1.6x faster with byte-identical match counts on
        # this site (tests/bench_edm_speed.py: 3085 matches/ref either way). On by default
        # because EDM pays a full forward per reference and that cost is the whole budget.
        get_cfg_defaults, EDM, lower_config = _import_edm()
        resolved_input = (EDM_W, EDM_H) if input_size is None else tuple(input_size)
        if (
            len(resolved_input) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in resolved_input
            )
            or any(value <= 0 or value % 32 for value in resolved_input)
        ):
            raise ValueError("EDM input_size must contain positive multiples of 32")
        self.input_w, self.input_h = resolved_input
        self.grid_w = self.input_w // COARSE_STRIDE
        self.grid_h = self.input_h // COARSE_STRIDE
        cfg = get_cfg_defaults()
        cfg.merge_from_file(str(cfg_path))
        cfg.EDM.COARSE.MCONF_THR = mconf_thr
        cfg.EDM.COARSE.BORDER_RM = border_rm
        # Paper's rule of thumb: topk ~= 0.35 * coarse grid size.
        cfg.EDM.COARSE.TOPK = int(topk if topk else self.grid_w * self.grid_h * 0.35)
        cfg.EDM.TEST_RES_H, cfg.EDM.TEST_RES_W = self.input_h, self.input_w
        # RoPE is rescaled from the training resolution to the test resolution
        # (transformer.py:31-41). Without this the positional encoding is wrong for
        # any input that is not 832x832.
        cfg.EDM.NECK.NPE = [
            cfg.EDM.TRAIN_RES_H,
            cfg.EDM.TRAIN_RES_W,
            self.input_h,
            self.input_w,
        ]

        self.cfg = cfg
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        self.topk = cfg.EDM.COARSE.TOPK
        self.mconf_thr = mconf_thr
        self.reference_cache_size = max(0, int(reference_cache_size))
        self._reference_tensor_cache: OrderedDict[int, tuple[object, torch.Tensor]] = OrderedDict()
        if runtime_sigma_mode not in RUNTIME_SIGMA_MODES:
            raise ValueError("runtime_sigma_mode must be 'bidirectional' or 'reference_grid'")
        self.runtime_sigma_mode = runtime_sigma_mode
        resolved_feature_cache = (
            _env_int("SFM_EDM_REF_FEATURE_CACHE", DEFAULT_REF_FEATURE_CACHE)
            if reference_feature_cache_size is None
            else reference_feature_cache_size
        )
        self.reference_feature_cache_size = _non_negative_int(
            "reference_feature_cache_size", resolved_feature_cache
        )
        if host_reference_feature_cache_size is None:
            resolved_host_cache = (
                0
                if reference_feature_cache_size is not None
                and int(reference_feature_cache_size) == 0
                else _env_int(
                    "SFM_EDM_HOST_REF_FEATURE_CACHE",
                    DEFAULT_HOST_REF_FEATURE_CACHE,
                )
            )
        else:
            resolved_host_cache = host_reference_feature_cache_size
        self.host_reference_feature_cache_size = _non_negative_int(
            "host_reference_feature_cache_size", resolved_host_cache
        )
        if not device.startswith("cuda"):
            self.host_reference_feature_cache_size = 0
        self.temporal_feature_cache_size = _non_negative_int(
            "temporal_feature_cache_size", temporal_feature_cache_size
        )
        self.temporal_feature_promotion = (
            _env_flag("SFM_EDM_TEMPORAL_FEATURE_PROMOTION", False)
            if temporal_feature_promotion is None
            else bool(temporal_feature_promotion)
        )
        self.inclusive_host_feature_cache = (
            _env_flag("SFM_EDM_INCLUSIVE_HOST_FEATURE_CACHE", False)
            if inclusive_host_feature_cache is None
            else bool(inclusive_host_feature_cache)
        )
        self.query_feature_reuse = _env_flag("SFM_EDM_QUERY_FEATURE_REUSE", True)
        self.query_batch_expand = _env_flag("SFM_EDM_QUERY_BATCH_EXPAND", True)
        if not isinstance(query_cuda_graph, bool):
            raise ValueError("query_cuda_graph must be boolean")
        if query_cuda_graph and not str(device).startswith("cuda"):
            raise ValueError("query_cuda_graph requires a CUDA device")
        self.query_cuda_graph = query_cuda_graph
        self._query_graph = None
        self._query_graph_input = None
        self._query_graph_features = None
        if track_cuda_graph is None:
            track_cuda_graph = _env_flag("SFM_EDM_TRACK_CUDAGRAPH", False)
        if not isinstance(track_cuda_graph, bool):
            raise ValueError("track_cuda_graph must be boolean")
        if track_cuda_graph and not str(device).startswith("cuda"):
            raise ValueError("track_cuda_graph requires a CUDA device")
        self.track_cuda_graph = track_cuda_graph
        self._track_cuda_graph_runner = None
        _validate_matcher_cache_budget(
            map_feature_entries=self.reference_feature_cache_size,
            temporal_feature_entries=self.temporal_feature_cache_size,
            map_tensor_entries=self.reference_cache_size,
            temporal_tensor_entries=self.temporal_feature_cache_size,
            host_feature_entries=self.host_reference_feature_cache_size,
        )
        self._feature_caches: dict[str, OrderedDict] = {
            "map": OrderedDict(),
            "temporal": OrderedDict(),
        }
        self._feature_cache_capacity = {
            "map": self.reference_feature_cache_size,
            "temporal": self.temporal_feature_cache_size,
        }
        self._feature_cache_stats = {
            "map": {"hits": 0, "misses": 0, "evictions": 0},
            "temporal": {"hits": 0, "misses": 0, "evictions": 0},
        }
        # Alias preserved for the existing map-cache unit tests.
        self._reference_feature_cache = self._feature_caches["map"]
        self._host_reference_feature_cache: OrderedDict[int, tuple[object, tuple]] = OrderedDict()
        self._host_feature_cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "stores": 0,
        }
        self._temporal_tensor_cache: OrderedDict[int, tuple[object, torch.Tensor]] = OrderedDict()
        # Set only while a match_many_to_one forward is in flight; it tells the
        # patched backbone which rows of its input are cacheable references.
        self._backbone_plan: list | None = None
        self._prepared_query_features: tuple | None = None
        self._sigma_plan: str | None = None

        model = EDM(config=lower_config(cfg)["edm"])
        trusted_ckpt_sha256 = _checkpoint_sha256_for_load(ckpt, ckpt_sha256)
        state = _load_edm_state_dict(ckpt, trusted_ckpt_sha256)
        model.load_state_dict(state)
        model = model.eval().to(device)
        model_identity = (
            f"schema={REFERENCE_FEATURE_STORE_SCHEMA}|ckpt={trusted_ckpt_sha256}|"
            f"cfg={_file_sha256(cfg_path)}|input={self.input_w}x{self.input_h}|"
            f"fp16={int(self.fp16)}|torch={torch.__version__}"
        )
        self.reference_feature_model_key = hashlib.sha256(
            model_identity.encode("utf-8")
        ).hexdigest()
        self._persistent_feature_sources: dict[int, tuple[object, int]] = {}
        self._persistent_feature_names: tuple[str, ...] = ()
        self._persistent_feature_digests: tuple[str, ...] = ()
        self._persistent_feature_levels: tuple[torch.Tensor, ...] | None = None
        self._persistent_feature_shards: tuple[tuple[torch.Tensor, ...], ...] | None = None
        self._persistent_feature_shard_size = 0
        self._persistent_feature_store_path: Path | None = None
        self._persistent_feature_stats = {"hits": 0, "misses": 0}
        if self.fp16:
            model = model.to(memory_format=torch.channels_last)
        self.model = model
        self.fused_coarse = _FUSED_COARSE_STATE["reason"]
        self._install_reference_feature_cache()
        self._install_runtime_sigma_selection()
        if self.track_cuda_graph:
            self._track_cuda_graph_runner = _TrackCUDAGraphRunner(self.model, self.device)
            self.model._track_cuda_graph_runner = self._track_cuda_graph_runner
        self.warmup_fused_coarse()

    # ``topk`` and ``mconf_thr`` used to be plain attributes written once here.
    # The values that actually run live on the model's own submodules --
    # CoarseMatching.topk / .thr and FineMatching.mconf_thr -- baked in from
    # ``cfg`` when EDM() was constructed, so assigning to the matcher afterwards
    # changed nothing. It failed silently: 定位演算法/validation's --coarse-topk
    # and --mconf-thr wrote the requested value straight into the run receipt
    # while the replay behaved exactly like the profile, so an A/B on either knob
    # returned a false "no effect" (measured: --mconf-thr 0.9 vs 0.2 differed on
    # 0/300 P168 frames; --coarse-topk 900 vs 3225 on 0/700, with n_corr still
    # reaching 1071). These properties keep the record and the live modules in
    # step. fused_forward reads CoarseMatching.topk outside the compiled tail, so
    # a change lands on the next forward with no recompile.
    @property
    def topk(self) -> int:
        return self._topk

    @topk.setter
    def topk(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("EDM coarse topk must be a positive integer")
        self._topk = value
        coarse = getattr(getattr(self, "model", None), "coarse_matching", None)
        if coarse is not None:
            coarse.topk = value
        self._invalidate_track_cuda_graph()

    @property
    def mconf_thr(self) -> float:
        return self._mconf_thr

    @mconf_thr.setter
    def mconf_thr(self, value: float) -> None:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("EDM mconf_thr must be finite")
        self._mconf_thr = number
        model = getattr(self, "model", None)
        coarse = getattr(model, "coarse_matching", None)
        if coarse is not None:
            coarse.thr = number
        fine = getattr(model, "fine_matching", None)
        if fine is not None:
            fine.mconf_thr = number
        self._invalidate_track_cuda_graph()

    def _invalidate_track_cuda_graph(self) -> None:
        """Drop the TRACK B=1 graph, which captured coarse_matching's old shapes.

        _TrackCUDAGraphRunner._step replays model.coarse_matching, so a captured
        graph holds the topk-dependent tensor sizes (and, with the fused tail off,
        the confidence threshold). Replaying it after a retune would quietly serve
        the pre-retune value -- the same silent-staleness this pair of properties
        exists to remove. The graph is off by default and rebuilt on next use.
        """
        runner = getattr(self, "_track_cuda_graph_runner", None)
        if runner is not None:
            runner.clear()

    def clear_track_cuda_graph(self) -> None:
        """Release private CUDA graph memory pools for TRACK B=1."""
        if getattr(self, "_track_cuda_graph_runner", None) is not None:
            self._track_cuda_graph_runner.clear()

    # ---------- reference backbone cache ----------
    def _install_reference_feature_cache(self) -> None:
        """Patch this model's backbone only; other matchers keep the plain one."""
        if (
            self.reference_feature_cache_size <= 0
            and self.temporal_feature_cache_size <= 0
            and self.host_reference_feature_cache_size <= 0
        ):
            return
        backbone = self.model.backbone
        original = backbone.forward
        self._backbone_original = original

        def cached_forward(x: torch.Tensor):
            plan = self._backbone_plan
            # Anything that is not one runtime match_many_to_one batch -- the map
            # build, match(), match_one_to_many(), a warmup probe -- runs upstream.
            if plan is None or x.shape[0] != 2 * len(plan):
                return original(x)
            return self._backbone_from_cache(original, x, plan)

        backbone.forward = cached_forward

    def _install_runtime_sigma_selection(self) -> None:
        """Runtime-only: prefer direction-01. Map-build match() keeps upstream sigma."""
        if self.runtime_sigma_mode != "reference_grid":
            return
        fine = self.model.fine_matching
        original = fine.final_matching_selection

        def runtime_selection(data):
            if self._sigma_plan == "reference_grid":
                select_reference_grid_matches(fine, data)
                return
            return original(data)

        fine.final_matching_selection = runtime_selection

    @staticmethod
    def _own_row(level: torch.Tensor, index: int) -> torch.Tensor:
        """Detach one batch row into its own storage, keeping the layout."""
        row = level[index : index + 1]
        if level.dim() == 4 and level.is_contiguous(memory_format=torch.channels_last):
            return row.contiguous(memory_format=torch.channels_last)
        return row.contiguous()

    @staticmethod
    def _stack_rows(rows: list, like: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat(rows, dim=0)
        if like.dim() == 4 and like.is_contiguous(memory_format=torch.channels_last):
            return stacked.contiguous(memory_format=torch.channels_last)
        return stacked

    @staticmethod
    def _broadcast_query(tensor: torch.Tensor, batch: int) -> torch.Tensor:
        """Broadcast one immutable query without allocating a repeated input."""
        if batch <= 0 or tensor.shape[0] != 1:
            raise ValueError("query broadcast requires one row and a positive batch")
        return tensor if batch == 1 else tensor.expand(batch, -1, -1, -1)

    def _store_reference_features(self, source, features: tuple, kind: str = "map") -> None:
        capacity = self._feature_cache_capacity[kind]
        stats = self._feature_cache_stats[kind]
        if capacity <= 0:
            return
        cache = self._feature_caches[kind]
        key = id(source)
        cache[key] = (source, features)
        cache.move_to_end(key)
        while len(cache) > capacity:
            _evicted_key, (evicted_source, evicted_features) = cache.popitem(last=False)
            if kind == "map":
                self._store_host_reference_features(evicted_source, evicted_features)
            stats["evictions"] += 1

    def _store_host_reference_features(self, source, features: tuple) -> None:
        capacity = self.host_reference_feature_cache_size
        if capacity <= 0:
            return
        cache = self._host_reference_feature_cache
        key = id(source)
        existing = cache.get(key)
        if self.inclusive_host_feature_cache and existing is not None and existing[0] is source:
            # Host storage is an immutable backing copy. Keep it inclusive while
            # the same features are promoted to GPU so a later GPU eviction does
            # not synchronously copy the blob back to host again.
            cache.move_to_end(key)
            return
        host_features = []
        for level in features:
            host = torch.empty_like(
                level,
                device="cpu",
                pin_memory=True,
                memory_format=torch.preserve_format,
            )
            host.copy_(level, non_blocking=False)
            host_features.append(host)
        cache[key] = (source, tuple(host_features))
        cache.move_to_end(key)
        self._host_feature_cache_stats["stores"] += 1
        while len(cache) > capacity:
            cache.popitem(last=False)
            self._host_feature_cache_stats["evictions"] += 1

    def _restore_host_reference_features(self, source) -> tuple | None:
        cache = self._host_reference_feature_cache
        key = id(source)
        entry = cache.get(key) if self.inclusive_host_feature_cache else cache.pop(key, None)
        if entry is None or entry[0] is not source:
            self._host_feature_cache_stats["misses"] += 1
            return None
        if self.inclusive_host_feature_cache:
            cache.move_to_end(key)
        self._host_feature_cache_stats["hits"] += 1
        return tuple(level.to(self.device, non_blocking=True) for level in entry[1])

    def _extract_one(self, original, image: torch.Tensor) -> tuple:
        """Always a batch of one, so an image's features never depend on what it
        was extracted alongside -- see the note above about output purity."""
        return tuple(self._own_row(level, 0) for level in original(image))

    def _plan_source(self, item) -> tuple[object, str]:
        if isinstance(item, tuple) and len(item) == 2 and item[1] in FEATURE_CACHE_KINDS:
            return item[0], item[1]
        return item, "map"

    def _all_reference_features_available(self, sources) -> bool:
        """True when every reference's features resolve without running the
        backbone on the batched input pixels: a GPU feature-cache hit or a
        persistent map restore. Mirrors _backbone_from_cache's lookup order."""
        for item in sources:
            source, kind = self._plan_source(item)
            cache = self._feature_caches[kind]
            entry = cache.get(id(source))
            if entry is not None and entry[0] is source:
                continue
            if kind != "map":
                return False
            levels = self._persistent_feature_levels
            shards = self._persistent_feature_shards
            bound = self._persistent_feature_sources.get(id(source))
            if (
                (levels is None and shards is None)
                or bound is None
                or bound[0] is not source
            ):
                return False
            if levels is None:
                if shards is None or self._persistent_feature_shard_size <= 0:
                    return False
                shard_index, _ = divmod(
                    bound[1],
                    self._persistent_feature_shard_size,
                )
                if shard_index >= len(shards):
                    return False
        return True

    def _backbone_from_cache(self, original, x: torch.Tensor, sources: list):
        b = len(sources)
        per_reference: list = []
        for index, item in enumerate(sources):
            source, kind = self._plan_source(item)
            cache = self._feature_caches[kind]
            stats = self._feature_cache_stats[kind]
            entry = cache.get(id(source))
            if entry is not None and entry[0] is source:
                cache.move_to_end(id(source))
                stats["hits"] += 1
                per_reference.append(entry[1])
                continue
            stats["misses"] += 1
            features = (
                self._restore_persistent_reference_features(source) if kind == "map" else None
            )
            if features is None:
                features = self._restore_host_reference_features(source) if kind == "map" else None
            if features is None:
                features = self._extract_one(original, x[index : index + 1])
            self._store_reference_features(source, features, kind)
            per_reference.append(features)
        query = self._prepared_query_features
        if query is None:
            query = self._extract_one(original, x[b : b + 1])

        return tuple(
            self._stack_rows(
                [features[depth] for features in per_reference] + [query[depth]] * b,
                query[depth],
            )
            for depth in range(len(query))
        )

    def reference_feature_cache_stats(self) -> dict:
        """Hit accounting for replay telemetry; counters are cumulative map-cache."""
        map_stats = self._feature_cache_stats["map"]
        return {
            **map_stats,
            "size": len(self._feature_caches["map"]),
            "capacity": self.reference_feature_cache_size,
        }

    def feature_cache_stats_by_class(self) -> dict:
        """Per-class LRU telemetry for map vs transient temporal features."""
        return {
            kind: {
                **self._feature_cache_stats[kind],
                "size": len(self._feature_caches[kind]),
                "capacity": self._feature_cache_capacity[kind],
            }
            for kind in FEATURE_CACHE_KINDS
        }

    def host_reference_feature_cache_stats(self) -> dict:
        return {
            **self._host_feature_cache_stats,
            "size": len(self._host_reference_feature_cache),
            "capacity": self.host_reference_feature_cache_size,
        }

    @staticmethod
    def _reference_source_sha256(source) -> str:
        gray = EDMMatcher.load_gray(source)
        digest = hashlib.sha256()
        digest.update(str(gray.shape).encode("ascii"))
        digest.update(str(gray.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(gray).tobytes())
        return digest.hexdigest()

    def reference_feature_store_path(self, bundle_sha256: str) -> Path | None:
        root = _runtime_cache_root()
        if root is None:
            return None
        digest = str(bundle_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("reference feature store requires the verified bundle SHA-256")
        return root / (f"{digest[:20]}_{self.reference_feature_model_key[:20]}.pt")

    @staticmethod
    def _sharded_reference_feature_store_path(path: Path) -> Path:
        return path.with_name(f"{path.name}.shards")

    def bind_reference_feature_store(
        self,
        sources: Mapping[str, object],
        *,
        bundle_sha256: str,
        build_if_missing: bool = False,
    ) -> Path | None:
        """Bind immutable map images to one model- and bundle-scoped feature bank."""
        names = tuple(str(name) for name in sources)
        values = tuple(sources[name] for name in sources)
        digests = tuple(self._reference_source_sha256(source) for source in values)
        self._persistent_feature_sources = {
            id(source): (source, index) for index, source in enumerate(values)
        }
        self._persistent_feature_names = names
        self._persistent_feature_digests = digests
        path = self.reference_feature_store_path(bundle_sha256)
        if path is None:
            return None
        sharded = self._sharded_reference_feature_store_path(path)
        selected = sharded if sharded.is_dir() else path
        self._persistent_feature_store_path = selected
        if selected.is_file() or selected.is_dir():
            try:
                self._load_reference_feature_store(selected)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self._persistent_feature_levels = None
                self._persistent_feature_shards = None
                print(
                    f"[edm_matcher] ignoring invalid reference feature store {selected}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if build_if_missing:
                    selected = self.build_reference_feature_store(path)
        elif build_if_missing:
            selected = self.build_reference_feature_store(path)
        return selected

    def _validate_reference_feature_identity(
        self,
        payload: object,
        *,
        schema_version: int,
    ) -> Mapping:
        if not isinstance(payload, Mapping):
            raise ValueError("EDM reference feature store must contain an object")
        if payload.get("schema_version") != schema_version:
            raise ValueError("EDM reference feature store schema mismatch")
        if payload.get("model_key") != self.reference_feature_model_key:
            raise ValueError("EDM reference feature store model mismatch")
        if tuple(payload.get("source_names", ())) != self._persistent_feature_names:
            raise ValueError("EDM reference feature store names mismatch")
        if tuple(payload.get("source_sha256", ())) != self._persistent_feature_digests:
            raise ValueError("EDM reference feature store image mismatch")
        return payload

    def _validate_reference_feature_payload(
        self,
        payload: object,
    ) -> tuple[torch.Tensor, ...]:
        payload = self._validate_reference_feature_identity(
            payload,
            schema_version=REFERENCE_FEATURE_STORE_SCHEMA,
        )
        raw_levels = payload.get("levels")
        if not isinstance(raw_levels, (list, tuple)) or not raw_levels:
            raise ValueError("EDM reference feature store has no feature levels")
        count = len(self._persistent_feature_names)
        levels = tuple(raw_levels)
        if any(
            not torch.is_tensor(level)
            or level.device.type != "cpu"
            or level.dim() != 4
            or level.shape[0] != count
            or not level.is_floating_point()
            for level in levels
        ):
            raise ValueError("EDM reference feature store tensor contract mismatch")
        return levels

    def _load_sharded_reference_feature_store(self, path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"EDM sharded feature store must be a regular directory: {path}")
        manifest_path = path / "manifest.pt"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("EDM sharded feature store manifest is missing")
        manifest = torch.load(
            manifest_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        manifest = self._validate_reference_feature_identity(
            manifest,
            schema_version=REFERENCE_FEATURE_SHARD_SCHEMA,
        )
        shard_size = manifest.get("shard_size")
        shard_files = manifest.get("shard_files")
        if (
            isinstance(shard_size, bool)
            or not isinstance(shard_size, int)
            or shard_size <= 0
            or not isinstance(shard_files, (list, tuple))
            or not shard_files
        ):
            raise ValueError("EDM sharded feature store manifest is invalid")
        expected_start = 0
        total_bytes = manifest_path.stat().st_size
        shards = []
        level_shapes = None
        for filename in shard_files:
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("EDM sharded feature store filename is invalid")
            shard_path = path / filename
            if shard_path.is_symlink() or not shard_path.is_file():
                raise ValueError(f"EDM reference feature shard is missing: {filename}")
            total_bytes += shard_path.stat().st_size
            if total_bytes > REFERENCE_FEATURE_STORE_MAX_BYTES:
                raise ValueError("EDM sharded feature store exceeds the size budget")
            payload = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if not isinstance(payload, Mapping):
                raise ValueError("EDM reference feature shard must contain an object")
            if payload.get("schema_version") != REFERENCE_FEATURE_SHARD_SCHEMA:
                raise ValueError("EDM reference feature shard schema mismatch")
            start, stop = payload.get("start"), payload.get("stop")
            levels = payload.get("levels")
            if (
                start != expected_start
                or isinstance(stop, bool)
                or not isinstance(stop, int)
                or stop <= start
                or not isinstance(levels, (list, tuple))
                or not levels
            ):
                raise ValueError("EDM reference feature shard range is invalid")
            count = stop - start
            checked = tuple(levels)
            if any(
                not torch.is_tensor(level)
                or level.device.type != "cpu"
                or level.dim() != 4
                or level.shape[0] != count
                or not level.is_floating_point()
                for level in checked
            ):
                raise ValueError("EDM reference feature shard tensor contract mismatch")
            shapes = tuple((tuple(level.shape[1:]), level.dtype) for level in checked)
            if level_shapes is None:
                level_shapes = shapes
            elif shapes != level_shapes:
                raise ValueError("EDM reference feature shard levels are inconsistent")
            shards.append(checked)
            expected_start = stop
        if expected_start != len(self._persistent_feature_names):
            raise ValueError("EDM sharded feature store entry count mismatch")
        self._persistent_feature_levels = None
        self._persistent_feature_shards = tuple(shards)
        self._persistent_feature_shard_size = shard_size

    def _load_reference_feature_store(self, path: Path) -> None:
        if path.is_dir():
            self._load_sharded_reference_feature_store(path)
            return
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"EDM reference feature store must be a regular file: {path}")
        size = path.stat().st_size
        if size <= 0 or size > REFERENCE_FEATURE_STORE_MAX_BYTES:
            raise ValueError(f"EDM reference feature store size is invalid: {size}")
        with path.open("rb") as stream:
            zip_serialized = stream.read(4).startswith(b"PK")
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=zip_serialized,
        )
        self._persistent_feature_levels = self._validate_reference_feature_payload(payload)
        self._persistent_feature_shards = None
        self._persistent_feature_shard_size = 0

    def _write_sharded_reference_feature_store(
        self,
        target: Path,
        banks: list[torch.Tensor],
    ) -> Path:
        sharded = self._sharded_reference_feature_store_path(target)
        temporary = sharded.with_name(f".{sharded.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        shard_files = []
        try:
            for shard_index, start in enumerate(
                range(0, len(self._persistent_feature_names), REFERENCE_FEATURE_SHARD_ENTRIES)
            ):
                stop = min(
                    len(self._persistent_feature_names),
                    start + REFERENCE_FEATURE_SHARD_ENTRIES,
                )
                filename = f"shard_{shard_index:04d}.pt"
                torch.save(
                    {
                        "schema_version": REFERENCE_FEATURE_SHARD_SCHEMA,
                        "start": start,
                        "stop": stop,
                        # Clone each slice so torch.save cannot serialize the full
                        # backing bank once per shard through shared storage.
                        "levels": [bank[start:stop].clone() for bank in banks],
                    },
                    temporary / filename,
                )
                shard_files.append(filename)
            torch.save(
                {
                    "schema_version": REFERENCE_FEATURE_SHARD_SCHEMA,
                    "model_key": self.reference_feature_model_key,
                    "source_names": list(self._persistent_feature_names),
                    "source_sha256": list(self._persistent_feature_digests),
                    "shard_size": REFERENCE_FEATURE_SHARD_ENTRIES,
                    "shard_files": shard_files,
                },
                temporary / "manifest.pt",
            )
            os.replace(temporary, sharded)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return sharded

    def build_reference_feature_store(self, path: Path | None = None) -> Path:
        """Extract each bound immutable reference once and atomically persist it."""
        target = path or self._persistent_feature_store_path
        if target is None:
            raise ValueError("reference feature store path is not bound")
        sources = sorted(
            self._persistent_feature_sources.values(),
            key=lambda item: item[1],
        )
        if not sources:
            raise ValueError("reference feature store has no bound sources")
        original = getattr(self, "_backbone_original", self.model.backbone)
        banks: list[torch.Tensor] | None = None
        with torch.no_grad(), self._autocast():
            for source, index in sources:
                tensor = self.to_tensor(self.load_input_gray(source))
                features = self._extract_one(original, tensor)
                if banks is None:
                    banks = [
                        torch.empty(
                            (len(sources), *level.shape[1:]),
                            dtype=level.dtype,
                            device="cpu",
                        )
                        for level in features
                    ]
                for bank, level in zip(banks, features):
                    bank[index].copy_(level[0].to("cpu"))
        assert banks is not None
        payload = {
            "schema_version": REFERENCE_FEATURE_STORE_SCHEMA,
            "model_key": self.reference_feature_model_key,
            "source_names": list(self._persistent_feature_names),
            "source_sha256": list(self._persistent_feature_digests),
            "levels": banks,
        }
        serialized_bytes = sum(level.numel() * level.element_size() for level in banks)
        target.parent.mkdir(parents=True, exist_ok=True)
        if serialized_bytes > REFERENCE_FEATURE_STORE_ZIP_MAX_BYTES:
            stored = self._write_sharded_reference_feature_store(target, banks)
        else:
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                torch.save(payload, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            stored = target
        self._persistent_feature_store_path = stored
        self._load_reference_feature_store(stored)
        return stored

    def _restore_persistent_reference_features(self, source) -> tuple | None:
        levels = self._persistent_feature_levels
        shards = self._persistent_feature_shards
        entry = self._persistent_feature_sources.get(id(source))
        if (levels is None and shards is None) or entry is None or entry[0] is not source:
            self._persistent_feature_stats["misses"] += 1
            return None
        index = entry[1]
        local_index = index
        if levels is None:
            assert shards is not None and self._persistent_feature_shard_size > 0
            shard_index, local_index = divmod(
                index,
                self._persistent_feature_shard_size,
            )
            if shard_index >= len(shards):
                self._persistent_feature_stats["misses"] += 1
                return None
            levels = shards[shard_index]
        self._persistent_feature_stats["hits"] += 1
        # The four pyramid levels have different HxW. Flatten the CPU rows and
        # move them in one host-to-device transfer, then split and view each
        # level back to its original shape.
        rows = [bank[local_index : local_index + 1] for bank in levels]
        sizes = [row.numel() for row in rows]
        packed = torch.cat([row.reshape(-1) for row in rows]).to(self.device)
        restored = []
        for row, part in zip(rows, packed.split(sizes)):
            level = part.view(row.shape)
            if level.dim() == 4:
                level = level.contiguous(memory_format=torch.channels_last)
            restored.append(level)
        return tuple(restored)

    def persistent_reference_feature_cache_stats(self) -> dict[str, int | str | None]:
        return {
            **self._persistent_feature_stats,
            "entries": (
                len(self._persistent_feature_names)
                if (
                    self._persistent_feature_levels is not None
                    or self._persistent_feature_shards is not None
                )
                else 0
            ),
            "path": (
                None
                if self._persistent_feature_store_path is None
                else str(self._persistent_feature_store_path)
            ),
        }

    def warmup_fused_coarse(self, batch_sizes: tuple | list | None = None) -> None:
        """Compile the fused tail now, not on the operator's first frame.

        The live worker is killed and respawned if it stalls past
        SFM_WORKER_WARMUP_S, so a first-frame compile would look like a hung
        worker. Compiled kernels are cached under 執行環境/inductor_cache, so
        only the very first run on a machine pays the full cost.

        Only the reference count varies per call -- one query frame at a time --
        so the batch sizes are 1..match_batch_size, i.e. {1, 2} in production.
        """
        compiled = _FUSED_COARSE_STATE.get("compiled")
        if compiled is None or not str(self.device).startswith("cuda"):
            return
        if batch_sizes is None:
            raw = os.environ.get("SFM_EDM_COARSE_WARMUP_BATCHES", "1,2")
            batch_sizes = [int(v) for v in raw.replace(" ", "").split(",") if v]
        cells = self.grid_w * self.grid_h
        for size in batch_sizes:
            if size <= 0:
                continue
            probe = torch.zeros((int(size), cells, cells), device=self.device, dtype=torch.float16)
            try:
                with torch.no_grad():
                    compiled(probe)
            except Exception as exc:  # a failed warmup must not block boot
                print(
                    f"[edm_matcher] fused coarse warmup b={size} failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                del probe
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16)

    def _capture_query_graph(self, original, tensor: torch.Tensor) -> None:
        static_input = torch.empty_like(tensor)
        static_input.copy_(tensor)
        capture_stream = torch.cuda.Stream(device=tensor.device)
        current_stream = torch.cuda.current_stream(device=tensor.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                original(static_input)
        current_stream.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = original(static_input)
        graph.replay()
        self._query_graph = graph
        self._query_graph_input = static_input
        self._query_graph_features = tuple(self._own_row(level, 0) for level in outputs)

    def _prepare_query_features(
        self,
        original,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        if not self.query_cuda_graph:
            return tensor, self._extract_one(original, tensor)
        if self._query_graph is None:
            self._capture_query_graph(original, tensor)
        else:
            self._query_graph_input.copy_(tensor)
            self._query_graph.replay()
        return self._query_graph_input, self._query_graph_features

    @torch.no_grad()
    def prepare_query(self, img1):
        """Extract one fixed-shape query backbone once for all reference chunks."""
        if not self.query_feature_reuse:
            return None
        original = getattr(self, "_backbone_original", self.model.backbone)
        tensor = self.to_tensor(self.load_input_gray(img1) if not torch.is_tensor(img1) else img1)
        with self._autocast():
            return self._prepare_query_features(original, tensor)

    def promote_prepared_query(self, source, prepared_query) -> None:
        """Reuse an accepted query as the next temporal reference exactly.

        The accepted grayscale copy has identical pixels to the prepared query;
        only its Python identity changes when the tracker takes ownership.
        """
        if (
            not self.temporal_feature_promotion
            or prepared_query is None
            or self.temporal_feature_cache_size <= 0
        ):
            return
        tensor, features = prepared_query
        if self.query_cuda_graph:
            tensor = tensor.clone(memory_format=torch.preserve_format)
            features = tuple(level.clone(memory_format=torch.preserve_format) for level in features)
        cache = self._temporal_tensor_cache
        key = id(source)
        cache[key] = (source, tensor)
        cache.move_to_end(key)
        while len(cache) > self.temporal_feature_cache_size:
            cache.popitem(last=False)
        self._store_reference_features(source, features, "temporal")

    # ---------- image io ----------
    @staticmethod
    def _load_gray_raw(src: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(src, (str, Path)):
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"cannot read image: {src}")
        else:
            img = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def load_gray(src: str | Path | np.ndarray) -> np.ndarray:
        """-> uint8 (EDM_H, EDM_W) grayscale. Accepts a path or a BGR/gray array."""
        img = EDMMatcher._load_gray_raw(src)
        if img.shape[:2] != (EDM_H, EDM_W):
            img = cv2.resize(img, (EDM_W, EDM_H), interpolation=cv2.INTER_AREA)
        return img

    def load_input_gray(self, src: str | Path | np.ndarray) -> np.ndarray:
        img = self._load_gray_raw(src)
        input_h = getattr(self, "input_h", EDM_H)
        input_w = getattr(self, "input_w", EDM_W)
        if img.shape[:2] != (input_h, input_w):
            img = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_AREA,
            )
        return img

    def to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(gray)[None][None].to(self.device).float() / 255.0

    def reference_tensor(self, source, kind: str = "map") -> torch.Tensor:
        """Return a cached device tensor for an immutable map or transient image."""
        if kind not in FEATURE_CACHE_KINDS:
            raise ValueError(f"unsupported EDM cache kind: {kind!r}")
        cache = self._temporal_tensor_cache if kind == "temporal" else self._reference_tensor_cache
        capacity = (
            self.temporal_feature_cache_size if kind == "temporal" else self.reference_cache_size
        )
        key = id(source)
        cached = cache.get(key)
        if cached is not None and cached[0] is source:
            cache.move_to_end(key)
            return cached[1]
        tensor = self.to_tensor(self.load_input_gray(source))
        if capacity > 0:
            cache[key] = (source, tensor)
            cache.move_to_end(key)
            while len(cache) > capacity:
                cache.popitem(last=False)
        return tensor

    # ---------- cells ----------
    @staticmethod
    def cell_ids(kpts: np.ndarray) -> np.ndarray:
        """(N,2) EDM-res keypoints -> (N,) int64 coarse-cell ids.

        Exact because the fine offset is clamped to +/-4px, i.e. half a cell.
        """
        cx = np.rint(kpts[:, 0] / COARSE_STRIDE).astype(np.int64)
        cy = np.rint(kpts[:, 1] / COARSE_STRIDE).astype(np.int64)
        np.clip(cx, 0, GRID_W - 1, out=cx)
        np.clip(cy, 0, GRID_H - 1, out=cy)
        return cy * GRID_W + cx

    @staticmethod
    def is_refined(kpts: np.ndarray) -> np.ndarray:
        """True where this side carries the subpixel offset (i.e. is NOT on the grid).

        The other side of the same match sits exactly on a multiple of COARSE_STRIDE.
        """
        snapped = np.rint(kpts / COARSE_STRIDE) * COARSE_STRIDE
        return (np.abs(kpts - snapped) > 1e-4).any(axis=1)

    @staticmethod
    def _match_outputs_to_numpy(batch: Mapping[str, torch.Tensor]):
        """Pack match outputs on-device so one host transfer is the only sync."""
        k0 = batch["mkpts0_f"]
        k1 = batch["mkpts1_f"]
        confidence = batch["mconf"]
        columns = [k0, k1, confidence[:, None]]
        bids = batch.get("m_bids")
        if bids is not None:
            columns.insert(0, bids[:, None].to(dtype=k0.dtype))
        packed = (
            torch.cat(
                [column.to(dtype=k0.dtype) for column in columns],
                dim=1,
            )
            .cpu()
            .numpy()
        )
        offset = 1 if bids is not None else 0
        host_bids = None if bids is None else packed[:, 0].astype(np.int64, copy=False)
        return (
            host_bids,
            packed[:, offset : offset + 2],
            packed[:, offset + 2 : offset + 4],
            packed[:, offset + 4],
        )

    @staticmethod
    def _split_match_outputs(
        bids: np.ndarray,
        k0: np.ndarray,
        k1: np.ndarray,
        confidence: np.ndarray,
        batch: int,
    ) -> list[dict]:
        """Split EDM's batch-sorted packed output into zero-copy NumPy views."""
        if batch <= 0 or len(bids) != len(k0) or len(k0) != len(k1) or len(k1) != len(confidence):
            raise ValueError("EDM packed match outputs are misaligned")
        if len(bids) and (
            bids[0] < 0
            or bids[-1] >= batch
            or np.any(bids[1:] < bids[:-1])
        ):
            raise ValueError("EDM batch ids must be sorted and within range")
        edges = np.searchsorted(bids, np.arange(batch + 1))
        return [
            {
                "mkpts0": k0[edges[index] : edges[index + 1]],
                "mkpts1": k1[edges[index] : edges[index + 1]],
                "mconf": confidence[edges[index] : edges[index + 1]],
            }
            for index in range(batch)
        ]

    # ---------- matching ----------
    @torch.no_grad()
    def match(self, img0, img1) -> dict:
        """One pair. img0/img1: path or array or preloaded gray. -> mkpts0/mkpts1/mconf."""
        t0 = self.to_tensor(self.load_input_gray(img0) if not torch.is_tensor(img0) else img0)
        t1 = self.to_tensor(self.load_input_gray(img1) if not torch.is_tensor(img1) else img1)
        batch = {"image0": t0, "image1": t1}
        with self._autocast():
            self.model(batch)
        _bids, k0, k1, confidence = self._match_outputs_to_numpy(batch)
        return {"mkpts0": k0, "mkpts1": k1, "mconf": confidence}

    @torch.no_grad()
    def match_one_to_many(self, img0, imgs1: list) -> list[dict]:
        """Match ONE image0 against B image1s in a single batched forward.

        This is the runtime path: image0 = query-independent reference is NOT what we
        want; we pass image0 = the reference and image1 = the query only in the 1-vs-1
        case. For 1 query vs K refs we instead tile the QUERY as image1 and stack the
        refs as image0, so every reference keeps its exact cell identity.
        """
        b = len(imgs1)
        t0 = self.to_tensor(self.load_input_gray(img0)).repeat(b, 1, 1, 1)
        t1 = torch.cat([self.to_tensor(self.load_input_gray(i)) for i in imgs1], dim=0)
        batch = {"image0": t0, "image1": t1}
        with self._autocast():
            self.model(batch)
        bids, k0, k1, mc = self._match_outputs_to_numpy(batch)
        assert bids is not None
        return self._split_match_outputs(bids, k0, k1, mc, b)

    @torch.no_grad()
    def match_many_to_one(
        self,
        imgs0: list,
        img1,
        source_kinds: list[str] | None = None,
        *,
        prepared_query=None,
    ) -> list[dict]:
        """K references (image0, batched) vs ONE query (image1, tiled). The localizer path."""
        b = len(imgs0)
        if b == 0:
            return []
        if source_kinds is None:
            kinds = ["map"] * b
        else:
            if len(source_kinds) != b:
                raise ValueError("source_kinds must match the reference count")
            kinds = list(source_kinds)
            unknown = sorted({kind for kind in kinds if kind not in FEATURE_CACHE_KINDS})
            if unknown:
                raise ValueError(f"unsupported EDM cache kind: {unknown[0]!r}")
        plan = list(zip(imgs0, kinds))
        # Packed D2H: _match_outputs_to_numpy packs mkpts/mconf/bids into one
        # contiguous buffer and does a single cpu().numpy() transfer. Cache-hit
        # path saves the dummy torch.cat([image0,image1], dim=0) allocation (9.4MB):
        # when every reference is cached and query features are prepared the
        # backbone reuses features and never reads image pixels, so the EDM
        # can skip the dummy cat and directly reuse cached features via
        # _backbone_from_cache. Guard keeps fallback cat when miss to preserve
        # exact behavior on cold frames.
        cache_fully_hit = prepared_query is not None and self._all_reference_features_available(plan)
        cache_hit = cache_fully_hit  # alias for existing branches
        if cache_hit:
            # Cache-hit: backbone will reuse features, so image pixels are
            # dummy. Use a shared expanded view to avoid per-call allocation
            # and save the subsequent torch.cat([image0, image1], dim=0) (9.4MB) inside EDM.
            # Guard: if getattr(self, '_backbone_plan', None) is not None and cache_fully_hit: bypass cat and call _backbone_from_cache path
            if (
                not hasattr(self, "_dummy_cache_image")
                or self._dummy_cache_image is None
                or self._dummy_cache_image.shape[2] != self.input_h
                or self._dummy_cache_image.shape[3] != self.input_w
                or self._dummy_cache_image.device != self.device
            ):
                self._dummy_cache_image = torch.empty(
                    (1, 1, self.input_h, self.input_w),
                    dtype=torch.float32,
                    device=self.device,
                )
            t0 = (
                self._dummy_cache_image
                if b == 1
                else self._dummy_cache_image.expand(b, -1, -1, -1)
            )
            query_tensor, query_features = prepared_query
        else:
            references = [
                self.reference_tensor(source, kind) for source, kind in plan
            ]
            t0 = references[0] if b == 1 else torch.cat(references, dim=0)
            if prepared_query is None:
                query_tensor = self.to_tensor(self.load_input_gray(img1))
                query_features = None
            else:
                query_tensor, query_features = prepared_query
        t1 = (
            self._broadcast_query(query_tensor, b)
            if self.query_batch_expand
            else query_tensor.repeat(b, 1, 1, 1)
        )
        batch = {"image0": t0, "image1": t1}
        self._backbone_plan = plan
        self._prepared_query_features = query_features
        self._sigma_plan = "reference_grid" if self.runtime_sigma_mode == "reference_grid" else None
        # Expose plan on the model so EDM can guard its dummy cat:
        # `if getattr(self, '_backbone_plan', None) is not None and cache_fully_hit:` before torch.cat([image0,image1]) and reuse cached features via _backbone_from_cache / _cached_pyramid; keep fallback cat when miss.
        if hasattr(self, "model"):
            try:
                self.model._backbone_plan = plan if cache_fully_hit else None
                self.model._cache_fully_hit = bool(cache_fully_hit)
                self.model._sigma_plan = self._sigma_plan
                if getattr(self, '_backbone_plan', None) is not None and cache_fully_hit:
                    try:
                        # Precompute pyramid via _backbone_from_cache to truly skip 9.4MB cat inside EDM.
                        # Dummy x with correct batch dim but no pixel alloc; _backbone_from_cache ignores content on hit.
                        _dummy = torch.empty((2 * b, 1, 1, 1), device=self.device, dtype=torch.float32)
                        self.model._cached_pyramid = self._backbone_from_cache(self._backbone_original, _dummy, plan)
                    except Exception:
                        self.model._cached_pyramid = None
                else:
                    self.model._cached_pyramid = None
            except Exception:
                pass
        try:
            if self._sigma_plan is not None and isinstance(batch, dict):
                batch["_sigma_plan"] = self._sigma_plan
            with self._autocast():
                self.model(batch)
        finally:
            self._backbone_plan = None
            self._prepared_query_features = None
            self._sigma_plan = None
            if hasattr(self, "model") and hasattr(self.model, "_backbone_plan"):
                try:
                    self.model._backbone_plan = None
                    self.model._cache_fully_hit = False
                    self.model._cached_pyramid = None
                    self.model._sigma_plan = None
                except Exception:
                    pass
        bids, k0, k1, mc = self._match_outputs_to_numpy(batch)
        assert bids is not None
        return self._split_match_outputs(bids, k0, k1, mc, b)


if __name__ == "__main__":
    print(__doc__)
    print(f"EDM input {EDM_W}x{EDM_H}, coarse grid {GRID_W}x{GRID_H} = {GRID_W * GRID_H} cells")
