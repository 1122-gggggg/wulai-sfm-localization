#!/usr/bin/env python3
"""Deployment relocalizer using XFeat + the official XFeat LighterGlue matcher + PnP.

This is a non-destructive alternative to reloc_localizer.py (ALIKED+LightGlue).
It loads the bundle produced by build_reloc_map_xfeat.py.

Source grounding:
- XFeat official repo: https://github.com/verlab/accelerated_features
  README documents torch.hub usage and the XFeat+LightGlue/LighterGlue path.
- The repo's notebook `xfeat+lg_torch_hub.ipynb` uses:
    output = xfeat.detectAndCompute(image, top_k=4096)[0]
    output.update({'image_size': (W, H)})
    mkpts0, mkpts1 = xfeat.match_lighterglue(output0, output1)
  We follow that API exactly. The bundled matcher is named LighterGlue by XFeat;
  it is a smaller/faster LightGlue-family matcher trained for XFeat descriptors.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from pose_types import Localizer, Pose

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_TOPK = 10
XFEAT_MAX_KP = 2048
MIN_INLIERS = 30
MEGALOC_INPUT = 322


class MegaLocQuery:
    """MegaLoc query extractor for bundles whose ref_global is MegaLoc."""

    def __init__(self, device: str = DEVICE, input_size: int = MEGALOC_INPUT):
        self.device = device
        self.input_size = input_size
        self.model = torch.hub.load(
            "gmberton/MegaLoc", "get_trained_model", trust_repo=True
        ).eval().to(device)
        self.tf = T.Compose([
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def extract_one(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame[..., :3]).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        d = self.model(x).float().cpu().numpy()[0].astype(np.float32)
        d /= np.linalg.norm(d) + 1e-12
        return d


def bundle_vpr_kind(meta: dict) -> str:
    v = str(meta.get("bundle_vpr") or meta.get("vpr") or "").lower()
    if not v.startswith("megaloc"):
        raise ValueError(
            "This localizer is MegaLoc-only. Rebuild or override the bundle "
            "ref_global descriptors with MegaLoc before deployment."
        )
    return "megaloc"


@dataclass
class XFeatRefEntry:
    feats: dict          # unbatched XFeat dict: keypoints/descriptors/scores/image_size on CPU
    xyz: np.ndarray      # (N,3), NaN means no 3D anchor


@dataclass
class XFeatRelocMap:
    ref_names: list
    ref_global: np.ndarray
    refs: dict
    meta: dict
    ref_centers: np.ndarray | None = None
    ref_yaws: np.ndarray | None = None
    covis: dict | None = None

    @staticmethod
    def load(path: str) -> "XFeatRelocMap":
        b = torch.load(path, map_location="cpu", weights_only=False)
        refs = {}
        for name, e in b["refs"].items():
            f = e["feats"]
            # LighterGlue expects fp32 descriptors. Store fp16 on disk if desired,
            # but cast back here before matching.
            if torch.is_tensor(f.get("descriptors")) and f["descriptors"].dtype != torch.float32:
                f = {**f, "descriptors": f["descriptors"].float()}
            refs[name] = XFeatRefEntry(feats=f, xyz=np.asarray(e["xyz"], np.float32))
        ref_global = np.asarray(b["ref_global"], np.float32)
        ref_global /= np.linalg.norm(ref_global, axis=1, keepdims=True) + 1e-12
        return XFeatRelocMap(
            ref_names=list(b["ref_names"]),
            ref_global=ref_global,
            refs=refs,
            meta=dict(b.get("meta", {})),
            ref_centers=None if "ref_centers" not in b else np.asarray(b["ref_centers"], np.float32),
            ref_yaws=None if "ref_yaws" not in b else np.asarray(b["ref_yaws"], np.float32),
            covis=b.get("covis"),
        )


@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: list


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_xfeat(top_k: int = XFEAT_MAX_KP):
    """Load official XFeat via torch.hub. Uses cached repo after first run."""
    model = torch.hub.load(
        "verlab/accelerated_features", "XFeat", pretrained=True,
        top_k=top_k, trust_repo=True,
    )
    model.top_k = top_k
    return model


@torch.inference_mode()
def extract_xfeat(xfeat, frame: np.ndarray, top_k: int = XFEAT_MAX_KP) -> dict:
    """Extract unbatched XFeat features from an HxWx3 uint8 RGB/BGR-like array.

    XFeat's official preprocess accepts uint8-like numpy arrays directly and
    internally resizes to dimensions divisible by 32; keypoints are returned in
    original image pixel coordinates.
    """
    h, w = frame.shape[:2]
    out = xfeat.detectAndCompute(frame, top_k=top_k)[0]
    out = {k: v.detach() if torch.is_tensor(v) else v for k, v in out.items()}
    out["image_size"] = (w, h)
    return out


def _to_device_feats(feats: dict, device: str = DEVICE) -> dict:
    out = {}
    for k, v in feats.items():
        if torch.is_tensor(v):
            vv = v.to(device)
            if k == "descriptors" and vv.dtype != torch.float32:
                vv = vv.float()
            out[k] = vv
        else:
            out[k] = v
    return out


# NOTE: dead code — the production localizer is ProductionXFeatTracker
# (production_xfeat_tracker.py). This class is unused; kept for reference.
class XFeatLightGlueLocalizer(Localizer):
    """MegaLoc retrieval + XFeat + XFeat-LighterGlue + pycolmap PnP."""

    def __init__(self, reloc_map: XFeatRelocMap, frame_source, query_cam: Camera,
                 query_topk: int = QUERY_TOPK, xfeat_max_kp: int = XFEAT_MAX_KP,
                 min_inliers: int = MIN_INLIERS, min_conf: float = 0.1):
        self.map = reloc_map
        self.frame_source = frame_source
        self.cam = query_cam
        self.query_topk = query_topk
        self.xfeat_max_kp = xfeat_max_kp
        self.min_inliers = min_inliers
        self.min_conf = min_conf
        self._vpr = None
        self._xfeat = None
        self._vpr_kind = bundle_vpr_kind(reloc_map.meta)
        self._last_inliers = 0
        self._last_timing = {}

    def _models(self):
        if self._vpr is None:
            self._vpr = MegaLocQuery(DEVICE)
            self._xfeat = load_xfeat(self.xfeat_max_kp)
        return self._vpr, self._xfeat

    def _extract_global(self, frame: np.ndarray) -> np.ndarray:
        return self._vpr.extract_one(frame)

    @torch.inference_mode()
    def get_pose(self) -> Pose | None:
        frame = self.frame_source()
        if frame is None:
            return None
        _, xfeat = self._models()

        timings = {}
        _sync(); t0 = time.perf_counter()
        q_glob = self._extract_global(frame)
        sims = self.map.ref_global @ q_glob
        topk = np.argsort(-sims)[:self.query_topk]
        _sync(); timings["vpr_ms"] = (time.perf_counter() - t0) * 1000

        _sync(); t0 = time.perf_counter()
        q_feats = extract_xfeat(xfeat, frame, self.xfeat_max_kp)
        _sync(); timings["xfeat_ms"] = (time.perf_counter() - t0) * 1000

        pts2d, pts3d = [], []
        _sync(); t0 = time.perf_counter()
        q_dev = _to_device_feats(q_feats)
        for idx in topk:
            name = self.map.ref_names[idx]
            ref = self.map.refs[name]
            r_dev = _to_device_feats(ref.feats)
            _mk0, _mk1, mi = xfeat.match_lighterglue(q_dev, r_dev, min_conf=self.min_conf)
            if len(mi) == 0:
                continue
            qkp = q_feats["keypoints"].detach().cpu().numpy()
            for qa, rb in mi:
                X = ref.xyz[int(rb)]
                if not np.any(np.isnan(X)):
                    pts2d.append(qkp[int(qa)])
                    pts3d.append(X)
        _sync(); timings["match_ms"] = (time.perf_counter() - t0) * 1000

        if len(pts3d) < self.min_inliers:
            self._last_inliers = 0
            self._last_timing = timings
            return None

        _sync(); t0 = time.perf_counter()
        import pycolmap
        cam = pycolmap.Camera(model=self.cam.model, width=self.cam.width,
                              height=self.cam.height, params=self.cam.params)
        ret = pycolmap.estimate_and_refine_absolute_pose(
            np.asarray(pts2d, float), np.asarray(pts3d, float), cam)
        timings["pnp_ms"] = (time.perf_counter() - t0) * 1000
        self._last_timing = timings
        if ret is None or ret.get("num_inliers", 0) < self.min_inliers:
            self._last_inliers = 0 if ret is None else int(ret.get("num_inliers", 0))
            return None
        self._last_inliers = int(ret["num_inliers"])

        T = ret["cam_from_world"]
        R = T.rotation.matrix()
        t = np.asarray(T.translation)
        C = -R.T @ t
        fwd = R.T @ np.array([0.0, 0.0, 1.0])
        yaw = float(np.arctan2(fwd[1], fwd[0]))
        return Pose(x=float(C[0]), y=float(C[1]), z=float(C[2]), yaw=yaw,
                    stamp=time.monotonic())


if __name__ == "__main__":
    print(__doc__)
