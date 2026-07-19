#!/usr/bin/env python3
"""EDM relocalizer: MegaLoc retrieval -> EDM matching -> PnP.

Drop-in replacement for the XFeat + LighterGlue + mutual-NN local matcher. Retrieval is
untouched: the bundle carries the same MegaLoc global descriptors, so only the local
matching and the 2D->3D lookup change.

WHY THERE IS NO KD-TREE / DEPTH MAP / SNAP RADIUS HERE:
  The reference is passed as image0, so every direction-01 match puts the reference side
  exactly on its coarse cell centre -- the same anchor the map build triangulated. The 3D
  lookup is therefore an exact O(1) index into xyz_by_cell, with no positional error
  between what was triangulated and what is matched at flight time.

  Direction-10 matches (reference side refined, query side on the grid) are DISCARDED:
  their reference point is not an anchor, so their 3D would be off by up to half a cell.
  Half the matches are dropped and there are still ~1000+ left per reference.

The bundle embeds the reference images (EDM must SEE them), so no external image
directory is needed at flight time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from edm_matcher import EDM_W, EDMMatcher

MEGALOC_INPUT = 322
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EDMRelocMap:
    ref_names: list
    ref_global: np.ndarray                 # (N, 8448) L2-normalised MegaLoc
    xyz_by_cell: dict                      # name -> (N_CELLS, 3) float32, NaN = unanchored
    images: dict                           # name -> (576, 1024) uint8 grayscale
    meta: dict
    ref_centers: np.ndarray | None = None
    ref_yaws: np.ndarray | None = None
    covis: dict | None = None

    @staticmethod
    def load(path: str | Path) -> "EDMRelocMap":
        b = torch.load(str(path), map_location="cpu", weights_only=False)
        if str(b["meta"].get("feature", "")).lower() != "edm":
            raise ValueError(f"not an EDM bundle: feature={b['meta'].get('feature')!r}")
        names = list(b["ref_names"])
        xyz, imgs = {}, {}
        for name in names:
            e = b["refs"][name]
            xyz[name] = np.asarray(e["xyz_by_cell"], np.float32)
            imgs[name] = cv2.imdecode(np.asarray(e["image_jpg"], np.uint8), cv2.IMREAD_GRAYSCALE)
        g = np.asarray(b["ref_global"], np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        return EDMRelocMap(
            ref_names=names, ref_global=g, xyz_by_cell=xyz, images=imgs, meta=dict(b["meta"]),
            ref_centers=None if "ref_centers" not in b else np.asarray(b["ref_centers"], np.float32),
            ref_yaws=None if "ref_yaws" not in b else np.asarray(b["ref_yaws"], np.float32),
            covis=b.get("covis"),
        )


class MegaLocQuery:
    """Same retrieval front-end as the XFeat localizer -- the bundle's ref_global is MegaLoc."""

    def __init__(self, device: str = DEVICE, input_size: int = MEGALOC_INPUT):
        from PIL import Image
        from torchvision import transforms as T
        self._Image = Image
        self.device = device
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model",
                                    trust_repo=True).eval().to(device)
        self.tf = T.Compose([
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def extract_one(self, rgb: np.ndarray) -> np.ndarray:
        x = self.tf(self._Image.fromarray(rgb[..., :3]).convert("RGB")).unsqueeze(0).to(self.device)
        d = self.model(x).float().cpu().numpy()[0].astype(np.float32)
        return d / (np.linalg.norm(d) + 1e-12)


@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: list = field(default_factory=list)


class EDMLocalizer:
    def __init__(self, reloc_map: EDMRelocMap, camera: Camera, matcher: EDMMatcher | None = None,
                 megaloc: MegaLocQuery | None = None, topk: int = 5, min_conf: float = 0.2,
                 pnp_max_error: float = 5.0, min_inliers: int = 50):
        self.map = reloc_map
        self.cam = camera
        self.matcher = matcher or EDMMatcher(mconf_thr=min_conf)
        self._megaloc = megaloc
        self.topk = topk
        self.min_inliers = min_inliers
        self.pnp_max_error = pnp_max_error
        self.scale = camera.width / EDM_W        # EDM px -> camera px (1280/1024 = 1.25)

    @property
    def megaloc(self) -> MegaLocQuery:
        if self._megaloc is None:
            self._megaloc = MegaLocQuery()
        return self._megaloc

    def retrieve(self, frame_rgb: np.ndarray, k: int, exclude: set[str] | None = None) -> list[str]:
        d = self.megaloc.extract_one(frame_rgb)
        sim = self.map.ref_global @ d
        order = np.argsort(-sim)
        out = []
        for i in order:
            name = self.map.ref_names[i]
            if exclude and name in exclude:
                continue
            out.append(name)
            if len(out) == k:
                break
        return out

    def correspondences(
        self,
        query_gray: np.ndarray,
        ref_names: list[str],
        batch_size: int | None = None,
    ):
        """-> (pts2d in camera px, pts3d in map units, per-ref match counts)."""
        by_ref = self.correspondences_by_ref(query_gray, ref_names, batch_size=batch_size)
        p2 = [row[0] for row in by_ref if len(row[0])]
        p3 = [row[1] for row in by_ref if len(row[1])]
        counts = [row[2] for row in by_ref]
        if not p2:
            return np.zeros((0, 2)), np.zeros((0, 3)), counts
        return np.concatenate(p2), np.concatenate(p3), counts

    def correspondences_by_ref(
        self,
        query_gray: np.ndarray,
        ref_names: list[str],
        batch_size: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Return 2D/3D correspondences separately for each reference.

        Keeping these boundaries lets EDM-only map recovery score each unrelated
        global reference with its own PnP hypothesis instead of mixing their 3D points.
        """
        return self.correspondences_for_sources(
            query_gray,
            [self.map.images[name] for name in ref_names],
            [self.map.xyz_by_cell[name] for name in ref_names],
            batch_size=batch_size,
        )

    def correspondences_for_sources(
        self,
        query_gray: np.ndarray,
        reference_images: list[np.ndarray],
        xyz_luts: list[np.ndarray],
        batch_size: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Match arbitrary reference images whose coarse cells carry map-frame 3D."""
        if len(reference_images) != len(xyz_luts):
            raise ValueError("reference_images and xyz_luts must have equal length")
        size = len(reference_images) if not batch_size or batch_size <= 0 else batch_size
        results = []
        for start in range(0, len(reference_images), max(size, 1)):
            images = reference_images[start : start + max(size, 1)]
            results.extend(
                self.matcher.match_many_to_one(images, query_gray)
            )
        out: list[tuple[np.ndarray, np.ndarray, int]] = []
        for xyz_lut, r in zip(xyz_luts, results):
            k0, k1 = r["mkpts0"], r["mkpts1"]
            if len(k0) == 0:
                out.append((np.zeros((0, 2)), np.zeros((0, 3)), 0))
                continue
            # keep direction-01 only: the reference side must sit on its cell centre,
            # which is the anchor the map build triangulated.
            d01 = ~EDMMatcher.is_refined(k0)
            if not d01.any():
                out.append((np.zeros((0, 2)), np.zeros((0, 3)), 0))
                continue
            cells = EDMMatcher.cell_ids(k0[d01])
            xyz = xyz_lut[cells]
            ok = np.isfinite(xyz).all(1)
            if ok.any():
                out.append((k1[d01][ok] * self.scale, xyz[ok], int(ok.sum())))
            else:
                out.append((np.zeros((0, 2)), np.zeros((0, 3)), 0))
        return out

    def localize(self, frame_bgr: np.ndarray, ref_names: list[str] | None = None,
                 exclude: set[str] | None = None) -> dict | None:
        import pycolmap
        gray = EDMMatcher.load_gray(frame_bgr)
        if ref_names is None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr.ndim == 3 else \
                cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
            ref_names = self.retrieve(rgb, self.topk, exclude=exclude)

        p2, p3, counts = self.correspondences(gray, ref_names)
        if len(p3) < 6:
            return None
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        opts = pycolmap.AbsolutePoseEstimationOptions()
        opts.ransac.max_error = self.pnp_max_error
        ret = pycolmap.estimate_and_refine_absolute_pose(
            np.asarray(p2, float), np.asarray(p3, float), cam, opts)
        if ret is None:
            return None
        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        t = np.asarray(T.translation)
        C = -R.T @ t
        fwd = R.T @ np.array([0, 0, 1.0])
        n_in = int(ret["num_inliers"])
        return {
            "R": R, "t": t, "center": C,
            "yaw": float(math.atan2(fwd[1], fwd[0])),
            "inliers": n_in,
            "n_corr": int(len(p3)),
            "refs": list(ref_names),
            "per_ref": counts,
            "ok": n_in >= self.min_inliers,
        }


if __name__ == "__main__":
    print(__doc__)
