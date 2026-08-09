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
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from artifact_integrity import verify_sha256
from edm_matcher import EDM_W, EDMMatcher
from reference_index import ReferenceIndex

MEGALOC_INPUT = 322
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEGALOC_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
MEGALOC_WEIGHTS_SHA256 = (
    "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
)
MEGALOC_HUBCONF_SHA256 = (
    "0ebf9fc9c455ca38b9e52c69bcfee4b136a0f8872fdfc4307d00469013228b98"
)
MEGALOC_MODEL_SOURCE_SHA256 = (
    "3cbf1d20515b1da423998a8edab787031eaa7bb273c5a86a5c41c4f6d84e2a6d"
)


def _torch_hub_cache() -> Path:
    configured = os.environ.get("SFM_TORCH_HUB_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        direct = parent / "torch_hub_cache"
        if direct.is_dir():
            return direct
        runtime = parent / "執行環境" / "torch_hub_cache"
        if runtime.is_dir():
            return runtime
    return Path(__file__).resolve().parents[3] / "執行環境" / "torch_hub_cache"


TORCH_HUB_DIR = _torch_hub_cache()
MEGALOC_REPO_DIR = TORCH_HUB_DIR / "gmberton_MegaLoc_main"
MEGALOC_WEIGHTS = (
    TORCH_HUB_DIR / "checkpoints" / "megaloc" / MEGALOC_REVISION
    / "model.safetensors"
)


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
    def load(
        path: str | Path,
        expected_sha256: str | None = None,
    ) -> "EDMRelocMap":
        artifact = Path(path)
        verify_sha256(artifact, expected_sha256)
        dtypes = (
            np.float16,
            np.float32,
            np.float64,
            np.uint8,
            np.int32,
            np.int64,
            np.bool_,
        )
        numpy_safe_globals = [
            np._core.multiarray._reconstruct,
            np._core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            *(type(np.dtype(dtype)) for dtype in dtypes),
        ]
        with torch.serialization.safe_globals(numpy_safe_globals):
            b = torch.load(
                artifact,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        b = _validate_edm_bundle_schema(b)
        names = list(b["ref_names"])
        xyz, imgs = {}, {}
        for name in names:
            e = b["refs"][name]
            xyz[name] = np.asarray(e["xyz_by_cell"], np.float32)
            decoded = cv2.imdecode(
                np.asarray(e["image_jpg"], np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            if decoded is None or decoded.shape != (576, 1024):
                raise ValueError(f"EDM reference image failed to decode at 1024x576: {name}")
            imgs[name] = decoded
        g = np.asarray(b["ref_global"], np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        return EDMRelocMap(
            ref_names=names, ref_global=g, xyz_by_cell=xyz, images=imgs, meta=dict(b["meta"]),
            ref_centers=None if "ref_centers" not in b else np.asarray(b["ref_centers"], np.float32),
            ref_yaws=None if "ref_yaws" not in b else np.asarray(b["ref_yaws"], np.float32),
            covis=b.get("covis"),
        )


def _validate_edm_bundle_schema(bundle: object) -> dict:
    if not isinstance(bundle, dict):
        raise ValueError("EDM relocation bundle must be a dictionary")
    required = {"meta", "ref_names", "ref_global", "refs"}
    allowed = required | {"ref_centers", "ref_yaws", "covis"}
    if not required.issubset(bundle) or not set(bundle).issubset(allowed):
        raise ValueError(
            f"unexpected EDM bundle keys: {sorted(set(bundle) - allowed)}; "
            f"missing: {sorted(required - set(bundle))}"
        )
    meta = bundle["meta"]
    if not isinstance(meta, dict) or str(meta.get("feature", "")).lower() != "edm":
        feature = None if not isinstance(meta, dict) else meta.get("feature")
        raise ValueError(f"not an EDM bundle: feature={feature!r}")
    names = bundle["ref_names"]
    refs = bundle["refs"]
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(names) != len(set(names))):
        raise ValueError("EDM bundle ref_names must be unique non-empty strings")
    if not isinstance(refs, dict) or set(refs) != set(names):
        raise ValueError("EDM bundle refs do not exactly match ref_names")
    dimensions = (
        meta.get("edm_grid_w"),
        meta.get("edm_grid_h"),
        meta.get("edm_input_w"),
        meta.get("edm_input_h"),
    )
    if (not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in dimensions)
            or dimensions != (128, 72, 1024, 576)):
        raise ValueError("EDM bundle grid/input dimensions are incompatible with this runtime")
    grid_w, grid_h, input_w, input_h = dimensions
    ref_global = bundle["ref_global"]
    if (not isinstance(ref_global, np.ndarray) or ref_global.ndim != 2
            or ref_global.shape[0] != len(names) or ref_global.shape[1] == 0
            or not np.issubdtype(ref_global.dtype, np.floating)
            or not np.isfinite(ref_global).all()
            or np.any(np.linalg.norm(ref_global, axis=1) <= 1e-12)):
        raise ValueError("EDM bundle ref_global has an invalid shape, dtype, or value")
    cell_count = grid_w * grid_h
    for name in names:
        entry = refs[name]
        if not isinstance(entry, dict) or set(entry) != {"xyz_by_cell", "image_jpg"}:
            raise ValueError(f"invalid EDM reference entry: {name}")
        xyz = entry["xyz_by_cell"]
        image_jpg = entry["image_jpg"]
        if (not isinstance(xyz, np.ndarray) or xyz.shape != (cell_count, 3)
                or xyz.dtype != np.float32):
            raise ValueError(f"invalid EDM xyz_by_cell: {name}")
        valid_xyz_rows = np.isfinite(xyz).all(axis=1) | np.isnan(xyz).all(axis=1)
        if np.isinf(xyz).any() or not valid_xyz_rows.all():
            raise ValueError(f"invalid non-finite EDM anchors: {name}")
        if (not isinstance(image_jpg, np.ndarray) or image_jpg.ndim != 1
                or image_jpg.dtype != np.uint8 or image_jpg.size == 0):
            raise ValueError(f"invalid embedded EDM reference image: {name}")
    for key, shape in (("ref_centers", (len(names), 3)), ("ref_yaws", (len(names),))):
        value = bundle.get(key)
        if value is not None and (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid EDM bundle {key}")
    covis = bundle.get("covis")
    if covis is not None:
        if not isinstance(covis, dict) or set(covis) != set(names):
            raise ValueError("invalid EDM bundle covis keys")
        for index, name in enumerate(names):
            neighbors = covis[name]
            if (not isinstance(neighbors, list)
                    or any(type(value) is not int or value < 0 or value >= len(names)
                           for value in neighbors)
                    or index in neighbors or len(neighbors) != len(set(neighbors))):
                raise ValueError(f"invalid EDM bundle covis neighbors: {name}")
    return bundle


class MegaLocQuery:
    """Same retrieval front-end as the XFeat localizer -- the bundle's ref_global is MegaLoc."""

    def __init__(self, device: str = DEVICE, input_size: int = MEGALOC_INPUT):
        from PIL import Image
        from torchvision import transforms as T
        self._Image = Image
        self.device = device
        verify_sha256(MEGALOC_REPO_DIR / "hubconf.py", MEGALOC_HUBCONF_SHA256)
        verify_sha256(
            MEGALOC_REPO_DIR / "megaloc_model.py", MEGALOC_MODEL_SOURCE_SHA256
        )
        verify_sha256(MEGALOC_WEIGHTS, MEGALOC_WEIGHTS_SHA256)
        torch.hub.set_dir(str(TORCH_HUB_DIR))
        self.model = torch.hub.load(
            str(MEGALOC_REPO_DIR),
            "get_trained_model",
            source="local",
            weights_path=str(MEGALOC_WEIGHTS),
        ).eval().to(device)
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
                 pnp_max_error: float = 5.0, min_inliers: int = 50,
                 reference_index: ReferenceIndex | None = None):
        self.map = reloc_map
        self.cam = camera
        self.matcher = matcher or EDMMatcher(mconf_thr=min_conf)
        self._megaloc = megaloc
        self.topk = topk
        self.min_inliers = min_inliers
        self.pnp_max_error = pnp_max_error
        self.scale = camera.width / EDM_W        # EDM px -> camera px (1280/1024 = 1.25)
        self.reference_index = reference_index
        if reference_index is not None:
            names = tuple(reloc_map.ref_names)
            if reference_index.count != len(names) or set(reference_index.names) != set(names):
                raise ValueError(
                    "reference index names do not match EDM relocation bundle"
                )

    @property
    def megaloc(self) -> MegaLocQuery:
        if self._megaloc is None:
            self._megaloc = MegaLocQuery()
        return self._megaloc

    def retrieve(self, frame_rgb: np.ndarray, k: int, exclude: set[str] | None = None) -> list[str]:
        d = self.megaloc.extract_one(frame_rgb)
        if self.reference_index is not None:
            requested = max(0, int(k))
            if requested == 0:
                return []
            candidate_count = min(
                self.reference_index.count,
                requested + len(exclude or ()),
            )
            matches = self.reference_index.query(d, top_k=candidate_count)
            out = [
                match.name
                for match in matches
                if not exclude or match.name not in exclude
            ]
            return out[:requested]
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
        """-> (camera-pixel 2D, map 3D, confidence, per-ref match counts)."""
        by_ref = self.correspondences_by_ref(query_gray, ref_names, batch_size=batch_size)
        p2 = [row[0] for row in by_ref if len(row[0])]
        p3 = [row[1] for row in by_ref if len(row[1])]
        confidence = [row[2] for row in by_ref if len(row[2])]
        counts = [row[3] for row in by_ref]
        if not p2:
            return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), counts
        return np.concatenate(p2), np.concatenate(p3), np.concatenate(confidence), counts

    def correspondences_by_ref(
        self,
        query_gray: np.ndarray,
        ref_names: list[str],
        batch_size: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
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
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
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
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        for xyz_lut, r in zip(xyz_luts, results):
            k0, k1 = r["mkpts0"], r["mkpts1"]
            confidence = np.asarray(r["mconf"], dtype=np.float32)
            if len(k0) == 0:
                out.append((np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), 0))
                continue
            # keep direction-01 only: the reference side must sit on its cell centre,
            # which is the anchor the map build triangulated.
            d01 = ~EDMMatcher.is_refined(k0)
            if not d01.any():
                out.append((np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), 0))
                continue
            cells = EDMMatcher.cell_ids(k0[d01])
            xyz = xyz_lut[cells]
            ok = np.isfinite(xyz).all(1)
            if ok.any():
                out.append((
                    k1[d01][ok] * self.scale,
                    xyz[ok],
                    confidence[d01][ok],
                    int(ok.sum()),
                ))
            else:
                out.append((np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0), 0))
        return out

    def localize(self, frame_bgr: np.ndarray, ref_names: list[str] | None = None,
                 exclude: set[str] | None = None) -> dict | None:
        import pycolmap
        gray = EDMMatcher.load_gray(frame_bgr)
        if ref_names is None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr.ndim == 3 else \
                cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
            ref_names = self.retrieve(rgb, self.topk, exclude=exclude)

        p2, p3, _confidence, counts = self.correspondences(gray, ref_names)
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
