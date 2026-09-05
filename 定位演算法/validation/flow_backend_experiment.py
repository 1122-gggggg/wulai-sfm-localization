#!/usr/bin/env python3
"""Swap the KLT bridge's optical-flow backend for a replay A/B.

Production tracks the EDM anchor's inlier points between keyframes with sparse
pyramidal Lucas-Kanade. This module supplies drop-in alternatives so the same
replay gate can score them end to end:

* ``dis_*`` -- DIS, Dense Inverse Search (Kroeger et al., arXiv:1603.03590),
  ``cv2.DISOpticalFlow``. Dense field, sampled bilinearly at the anchor points.
* ``fastflownet`` -- FastFlowNet (Kong et al., ICRA 2021, arXiv:2103.04524).
  Needs the upstream checkpoint + compiled correlation op; point
  ``SFM_FASTFLOWNET_ROOT`` at a built checkout.

Install with :func:`install`, which sets ``production_edm_tracker._FLOW_BACKEND``.
The production default (``None``) leaves the LK calls untouched.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

DIS_PRESETS = {
    "dis_ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "dis_fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "dis_medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}


def _sample(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinearly sample a HxWx2 dense flow field at float image points."""
    xs = points[:, 0].astype(np.float32).reshape(-1, 1)
    ys = points[:, 1].astype(np.float32).reshape(-1, 1)
    fx = cv2.remap(flow[..., 0], xs, ys, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE).ravel()
    fy = cv2.remap(flow[..., 1], xs, ys, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE).ravel()
    return points + np.stack([fx, fy], axis=1)


def _as_status(points: np.ndarray, height: int, width: int) -> np.ndarray:
    """Mimic the LK status flag: 1 when the tracked point stayed in frame."""
    ok = (np.isfinite(points).all(axis=1)
          & (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
          & (points[:, 1] >= 0) & (points[:, 1] <= height - 1))
    return ok.astype(np.uint8).reshape(-1, 1)


def dis_backend(preset: int):
    """DIS (arXiv:1603.03590) as a sparse-point tracker."""
    dis = cv2.DISOpticalFlow_create(preset)

    def track(prev_gray, gray, points):
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        flow = dis.calc(prev_gray, gray, None)
        nxt = _sample(flow, p)
        status = _as_status(nxt, gray.shape[0], gray.shape[1])
        return nxt.reshape(-1, 1, 2).astype(np.float32), status
    return track


def fastflownet_backend(checkpoint: str = "fastflownet_ft_mix.pth", half: bool = True):
    """FastFlowNet (arXiv:2103.04524) as a sparse-point tracker.

    The bridge hands us grayscale; FastFlowNet expects 3-channel input, so the
    grayscale plane is replicated. Flow is predicted at a /64-padded size and
    resampled back to source pixels.
    """
    import torch
    import torch.nn.functional as F

    root = os.environ.get("SFM_FASTFLOWNET_ROOT")
    if not root:
        raise RuntimeError("set SFM_FASTFLOWNET_ROOT to a built FastFlowNet checkout")
    for path in (root, os.path.join(root, "models", "correlation_package")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from models.FastFlowNet import FastFlowNet  # noqa: E402

    model = FastFlowNet().cuda().eval()
    model.load_state_dict(
        torch.load(os.path.join(root, "checkpoints", checkpoint), map_location="cuda")
    )
    if half:
        model = model.half()
    dtype = torch.half if half else torch.float
    div = 64

    def track(prev_gray, gray, points):
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        h, w = gray.shape[:2]
        ih, iw = int(div * np.ceil(h / div)), int(div * np.ceil(w / div))
        a = torch.from_numpy(np.ascontiguousarray(prev_gray)).cuda()[None, None].to(dtype) / 255.0
        b = torch.from_numpy(np.ascontiguousarray(gray)).cuda()[None, None].to(dtype) / 255.0
        a, b = a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1)
        mean = torch.cat([a, b], 2).view(1, 3, -1).mean(2).view(1, 3, 1, 1)
        a, b = a - mean, b - mean
        a = F.interpolate(a, (ih, iw), mode="bilinear", align_corners=False)
        b = F.interpolate(b, (ih, iw), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = model(torch.cat([a, b], 1)).float()
        flow = 20.0 * F.interpolate(out, (h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / iw
        flow[:, 1] *= h / ih
        field = flow[0].permute(1, 2, 0).cpu().numpy()
        nxt = _sample(field, p)
        return nxt.reshape(-1, 1, 2).astype(np.float32), _as_status(nxt, h, w)
    return track


def build(name: str):
    if name in DIS_PRESETS:
        return dis_backend(DIS_PRESETS[name])
    if name == "fastflownet":
        return fastflownet_backend()
    raise ValueError(f"unknown flow backend {name!r}")


def install(name: str | None = None) -> str:
    """Point the tracker's bridge at ``name`` (env ``SFM_EDM_FLOW_BACKEND``)."""
    import production_edm_tracker as pet

    name = name or os.environ.get("SFM_EDM_FLOW_BACKEND", "klt")
    if name in ("", "klt", "lk", "none"):
        pet._FLOW_BACKEND = None
        return "klt"
    pet._FLOW_BACKEND = build(name)
    return name
