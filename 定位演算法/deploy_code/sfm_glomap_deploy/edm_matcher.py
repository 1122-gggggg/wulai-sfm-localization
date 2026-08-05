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
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

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
DEFAULT_CKPT_SHA256 = (
    "f686bebdd9705bf6918621a1a83695f83d698cbd8c3eed932847fe3678d13a97"
)
DEFAULT_CFG = EDM_REPO / "configs" / "edm" / "outdoor" / "edm_base.py"

# 1280x720 (the map camera) -> 1024x576: exactly 16:9, both sides divisible by 32,
# so no padding, no mask, no aspect distortion. Coarse grid is 128x72.
EDM_W, EDM_H = 1024, 576
COARSE_STRIDE = 8  # EDM.LOCAL_RESOLUTION
GRID_W, GRID_H = EDM_W // COARSE_STRIDE, EDM_H // COARSE_STRIDE  # 128 x 72


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
        print(f"[edm_matcher] ignoring non-integer {name}={raw!r}",
              file=sys.stderr, flush=True)
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
        if (self.training or self.deploy or not self.ds_opt
                or mask_c0 is not None or mask_c1 is not None):
            return original_forward(self, feat_c0, feat_c1, data, mask_c0, mask_c1)

        feat_c0, feat_c1 = map(
            lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1]
        )
        with torch.autocast(enabled=False, device_type="cuda"):
            sim_matrix = (
                torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            )
        del feat_c0, feat_c1
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
    except Exception as exc:                       # keep the upstream tail on any failure
        _FUSED_COARSE_STATE.update(
            installed=True, compiled=None, reason=f"fused coarse tail unavailable: {exc!r}"
        )
    return _FUSED_COARSE_STATE["reason"]


def _import_edm():
    if str(EDM_REPO) not in sys.path:
        sys.path.insert(0, str(EDM_REPO))
    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config
    _install_fused_coarse_matching()
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
# Cost is 8.16 MiB of fp16 features per reference, so the default 16 entries
# hold 130 MiB. Entries are keyed like reference_tensor(): by id() with the
# source object retained, so an id cannot be recycled behind the cache. A
# temporal reference (use_temporal_reference, off in every shipped profile) is
# a per-frame array and would occupy an entry until evicted.
#
# Set SFM_EDM_REF_FEATURE_CACHE=0 to disable, or to another size to retune.
# ---------------------------------------------------------------------------
DEFAULT_REF_FEATURE_CACHE = 16


class EDMMatcher:
    """Detector-free matcher. Coordinates in/out are EDM-input pixels (1024x576)."""

    def __init__(self, ckpt: str | Path = DEFAULT_CKPT,
                 cfg_path: str | Path = DEFAULT_CFG,
                 mconf_thr: float = 0.2, topk: int | None = None, border_rm: int = 2,
                 device: str = "cuda", fp16: bool = True,
                 reference_cache_size: int = 32,
                 ckpt_sha256: str | None = None,
                 reference_feature_cache_size: int | None = None):
        # fp16 autocast + channels_last: 1.6x faster with byte-identical match counts on
        # this site (tests/bench_edm_speed.py: 3085 matches/ref either way). On by default
        # because EDM pays a full forward per reference and that cost is the whole budget.
        get_cfg_defaults, EDM, lower_config = _import_edm()
        cfg = get_cfg_defaults()
        cfg.merge_from_file(str(cfg_path))
        cfg.EDM.COARSE.MCONF_THR = mconf_thr
        cfg.EDM.COARSE.BORDER_RM = border_rm
        # Paper's rule of thumb: topk ~= 0.35 * coarse grid size.
        cfg.EDM.COARSE.TOPK = int(topk if topk else GRID_W * GRID_H * 0.35)
        cfg.EDM.TEST_RES_H, cfg.EDM.TEST_RES_W = EDM_H, EDM_W
        # RoPE is rescaled from the training resolution to the test resolution
        # (transformer.py:31-41). Without this the positional encoding is wrong for
        # any input that is not 832x832.
        cfg.EDM.NECK.NPE = [cfg.EDM.TRAIN_RES_H, cfg.EDM.TRAIN_RES_W, EDM_H, EDM_W]

        self.cfg = cfg
        self.device = device
        self.topk = cfg.EDM.COARSE.TOPK
        self.mconf_thr = mconf_thr
        self.reference_cache_size = max(0, int(reference_cache_size))
        self._reference_tensor_cache: OrderedDict[
            int, tuple[object, torch.Tensor]
        ] = OrderedDict()
        self.reference_feature_cache_size = (
            _env_int("SFM_EDM_REF_FEATURE_CACHE", DEFAULT_REF_FEATURE_CACHE)
            if reference_feature_cache_size is None
            else max(0, int(reference_feature_cache_size))
        )
        self._reference_feature_cache: OrderedDict[
            int, tuple[object, tuple]
        ] = OrderedDict()
        self._feature_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        # Set only while a match_many_to_one forward is in flight; it tells the
        # patched backbone which rows of its input are cacheable references.
        self._backbone_plan: list | None = None

        self.fp16 = fp16 and device.startswith("cuda")
        model = EDM(config=lower_config(cfg)["edm"])
        state = _load_edm_state_dict(
            ckpt,
            _checkpoint_sha256_for_load(ckpt, ckpt_sha256),
        )
        model.load_state_dict(state)
        model = model.eval().to(device)
        if self.fp16:
            model = model.to(memory_format=torch.channels_last)
        self.model = model
        self.fused_coarse = _FUSED_COARSE_STATE["reason"]
        self._install_reference_feature_cache()
        self.warmup_fused_coarse()

    # ---------- reference backbone cache ----------
    def _install_reference_feature_cache(self) -> None:
        """Patch this model's backbone only; other matchers keep the plain one."""
        if self.reference_feature_cache_size <= 0:
            return
        backbone = self.model.backbone
        original = backbone.forward

        def cached_forward(x: torch.Tensor):
            plan = self._backbone_plan
            # Anything that is not one runtime match_many_to_one batch -- the map
            # build, match(), match_one_to_many(), a warmup probe -- runs upstream.
            if plan is None or x.shape[0] != 2 * len(plan):
                return original(x)
            return self._backbone_from_cache(original, x, plan)

        backbone.forward = cached_forward

    @staticmethod
    def _own_row(level: torch.Tensor, index: int) -> torch.Tensor:
        """Detach one batch row into its own storage, keeping the layout."""
        row = level[index:index + 1]
        if level.dim() == 4 and level.is_contiguous(memory_format=torch.channels_last):
            return row.contiguous(memory_format=torch.channels_last)
        return row.contiguous()

    @staticmethod
    def _stack_rows(rows: list, like: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat(rows, dim=0)
        if like.dim() == 4 and like.is_contiguous(memory_format=torch.channels_last):
            return stacked.contiguous(memory_format=torch.channels_last)
        return stacked

    def _store_reference_features(self, source, features: tuple) -> None:
        cache = self._reference_feature_cache
        key = id(source)
        cache[key] = (source, features)
        cache.move_to_end(key)
        while len(cache) > self.reference_feature_cache_size:
            cache.popitem(last=False)
            self._feature_cache_stats["evictions"] += 1

    def _extract_one(self, original, image: torch.Tensor) -> tuple:
        """Always a batch of one, so an image's features never depend on what it
        was extracted alongside -- see the note above about output purity."""
        return tuple(self._own_row(level, 0) for level in original(image))

    def _backbone_from_cache(self, original, x: torch.Tensor, sources: list):
        b = len(sources)
        per_reference: list = []
        for index, source in enumerate(sources):
            entry = self._reference_feature_cache.get(id(source))
            if entry is not None and entry[0] is source:
                self._reference_feature_cache.move_to_end(id(source))
                self._feature_cache_stats["hits"] += 1
                per_reference.append(entry[1])
                continue
            self._feature_cache_stats["misses"] += 1
            features = self._extract_one(original, x[index:index + 1])
            self._store_reference_features(source, features)
            per_reference.append(features)
        query = self._extract_one(original, x[b:b + 1])

        return tuple(
            self._stack_rows(
                [features[depth] for features in per_reference] + [query[depth]] * b,
                query[depth],
            )
            for depth in range(len(query))
        )

    def reference_feature_cache_stats(self) -> dict:
        """Hit accounting for replay telemetry; counters are cumulative."""
        return {
            **self._feature_cache_stats,
            "size": len(self._reference_feature_cache),
            "capacity": self.reference_feature_cache_size,
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
        cells = GRID_W * GRID_H
        for size in batch_sizes:
            if size <= 0:
                continue
            probe = torch.zeros(
                (int(size), cells, cells), device=self.device, dtype=torch.float16)
            try:
                with torch.no_grad():
                    compiled(probe)
            except Exception as exc:               # a failed warmup must not block boot
                print(f"[edm_matcher] fused coarse warmup b={size} failed: {exc!r}",
                      file=sys.stderr, flush=True)
            finally:
                del probe
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16)

    # ---------- image io ----------
    @staticmethod
    def load_gray(src: str | Path | np.ndarray) -> np.ndarray:
        """-> uint8 (EDM_H, EDM_W) grayscale. Accepts a path or a BGR/gray array."""
        if isinstance(src, (str, Path)):
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"cannot read image: {src}")
        else:
            img = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if img.shape[:2] != (EDM_H, EDM_W):
            img = cv2.resize(img, (EDM_W, EDM_H), interpolation=cv2.INTER_AREA)
        return img

    def to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(gray)[None][None].to(self.device).float() / 255.0

    def reference_tensor(self, source) -> torch.Tensor:
        """Return a cached device tensor for an immutable map reference image."""
        key = id(source)
        cached = self._reference_tensor_cache.get(key)
        if cached is not None and cached[0] is source:
            self._reference_tensor_cache.move_to_end(key)
            return cached[1]
        tensor = self.to_tensor(self.load_gray(source))
        if self.reference_cache_size > 0:
            self._reference_tensor_cache[key] = (source, tensor)
            self._reference_tensor_cache.move_to_end(key)
            while len(self._reference_tensor_cache) > self.reference_cache_size:
                self._reference_tensor_cache.popitem(last=False)
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

    # ---------- matching ----------
    @torch.no_grad()
    def match(self, img0, img1) -> dict:
        """One pair. img0/img1: path or array or preloaded gray. -> mkpts0/mkpts1/mconf."""
        t0 = self.to_tensor(self.load_gray(img0) if not torch.is_tensor(img0) else img0)
        t1 = self.to_tensor(self.load_gray(img1) if not torch.is_tensor(img1) else img1)
        batch = {"image0": t0, "image1": t1}
        with self._autocast():
            self.model(batch)
        return {
            "mkpts0": batch["mkpts0_f"].cpu().numpy(),
            "mkpts1": batch["mkpts1_f"].cpu().numpy(),
            "mconf": batch["mconf"].cpu().numpy(),
        }

    @torch.no_grad()
    def match_one_to_many(self, img0, imgs1: list) -> list[dict]:
        """Match ONE image0 against B image1s in a single batched forward.

        This is the runtime path: image0 = query-independent reference is NOT what we
        want; we pass image0 = the reference and image1 = the query only in the 1-vs-1
        case. For 1 query vs K refs we instead tile the QUERY as image1 and stack the
        refs as image0, so every reference keeps its exact cell identity.
        """
        b = len(imgs1)
        t0 = self.to_tensor(self.load_gray(img0)).repeat(b, 1, 1, 1)
        t1 = torch.cat([self.to_tensor(self.load_gray(i)) for i in imgs1], dim=0)
        batch = {"image0": t0, "image1": t1}
        with self._autocast():
            self.model(batch)
        bids = batch["m_bids"].cpu().numpy()
        k0 = batch["mkpts0_f"].cpu().numpy()
        k1 = batch["mkpts1_f"].cpu().numpy()
        mc = batch["mconf"].cpu().numpy()
        return [{"mkpts0": k0[bids == i], "mkpts1": k1[bids == i], "mconf": mc[bids == i]}
                for i in range(b)]

    @torch.no_grad()
    def match_many_to_one(self, imgs0: list, img1) -> list[dict]:
        """K references (image0, batched) vs ONE query (image1, tiled). The localizer path."""
        b = len(imgs0)
        if b == 0:
            return []
        references = [self.reference_tensor(source) for source in imgs0]
        t0 = references[0] if b == 1 else torch.cat(references, dim=0)
        t1 = self.to_tensor(self.load_gray(img1)).repeat(b, 1, 1, 1)
        batch = {"image0": t0, "image1": t1}
        self._backbone_plan = list(imgs0)
        try:
            with self._autocast():
                self.model(batch)
        finally:
            self._backbone_plan = None
        bids = batch["m_bids"].cpu().numpy()
        k0 = batch["mkpts0_f"].cpu().numpy()
        k1 = batch["mkpts1_f"].cpu().numpy()
        mc = batch["mconf"].cpu().numpy()
        return [{"mkpts0": k0[bids == i], "mkpts1": k1[bids == i], "mconf": mc[bids == i]}
                for i in range(b)]


if __name__ == "__main__":
    print(__doc__)
    print(f"EDM input {EDM_W}x{EDM_H}, coarse grid {GRID_W}x{GRID_H} = {GRID_W*GRID_H} cells")
