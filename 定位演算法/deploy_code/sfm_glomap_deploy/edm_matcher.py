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

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

def _find_edm_repo() -> Path:
    """The EDM checkout sits at third_party/EDM in the dev tree and runtime/EDM in the
    transfer package. $EDM_REPO overrides both."""
    import os
    if os.environ.get("EDM_REPO"):
        return Path(os.environ["EDM_REPO"])
    root = Path(__file__).resolve().parent.parent
    for rel in ("third_party/EDM", "runtime/EDM"):
        if (root / rel / "src").is_dir():
            return root / rel
    raise FileNotFoundError(f"no EDM checkout under {root} (looked in third_party/, runtime/)")


EDM_REPO = _find_edm_repo()
DEFAULT_CKPT = EDM_REPO / "weights" / "edm_outdoor.ckpt"
DEFAULT_CFG = EDM_REPO / "configs" / "edm" / "outdoor" / "edm_base.py"

# 1280x720 (the map camera) -> 1024x576: exactly 16:9, both sides divisible by 32,
# so no padding, no mask, no aspect distortion. Coarse grid is 128x72.
EDM_W, EDM_H = 1024, 576
COARSE_STRIDE = 8  # EDM.LOCAL_RESOLUTION
GRID_W, GRID_H = EDM_W // COARSE_STRIDE, EDM_H // COARSE_STRIDE  # 128 x 72


def _import_edm():
    if str(EDM_REPO) not in sys.path:
        sys.path.insert(0, str(EDM_REPO))
    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config
    return get_cfg_defaults, EDM, lower_config


class EDMMatcher:
    """Detector-free matcher. Coordinates in/out are EDM-input pixels (1024x576)."""

    def __init__(self, ckpt: str | Path = DEFAULT_CKPT, cfg_path: str | Path = DEFAULT_CFG,
                 mconf_thr: float = 0.2, topk: int | None = None, border_rm: int = 2,
                 device: str = "cuda", fp16: bool = True):
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

        self.fp16 = fp16 and device.startswith("cuda")
        model = EDM(config=lower_config(cfg)["edm"])
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
        model = model.eval().to(device)
        if self.fp16:
            model = model.to(memory_format=torch.channels_last)
        self.model = model

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
        t0 = torch.cat([self.to_tensor(self.load_gray(i)) for i in imgs0], dim=0)
        t1 = self.to_tensor(self.load_gray(img1)).repeat(b, 1, 1, 1)
        batch = {"image0": t0, "image1": t1}
        with self._autocast():
            self.model(batch)
        bids = batch["m_bids"].cpu().numpy()
        k0 = batch["mkpts0_f"].cpu().numpy()
        k1 = batch["mkpts1_f"].cpu().numpy()
        mc = batch["mconf"].cpu().numpy()
        return [{"mkpts0": k0[bids == i], "mkpts1": k1[bids == i], "mconf": mc[bids == i]}
                for i in range(b)]


if __name__ == "__main__":
    print(__doc__)
    print(f"EDM input {EDM_W}x{EDM_H}, coarse grid {GRID_W}x{GRID_H} = {GRID_W*GRID_H} cells")
