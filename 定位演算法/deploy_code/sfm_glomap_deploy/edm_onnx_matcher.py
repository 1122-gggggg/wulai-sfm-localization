#!/usr/bin/env python3
"""ONNX Runtime backend for EDM matching (CUDA EP / TensorRT EP).

Drop-in alternative to the PyTorch EDMMatcher for the pair-matching stage.
I/O matches the official EDM deploy export:
  input  : float32 (N, 2, H, W) = concat(ref, query) along channel
  output : float32 (K, 11) raw matches (deploy mode)
Post-process mirrors deploy/run_onnx.py and returns mkpts0/mkpts1/mconf in EDM
pixels so the cell-trick 3D lookup stays valid.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from edm_matcher import COARSE_STRIDE, EDM_H, EDM_W, EDMMatcher

DEFAULT_ONNX = (
    Path(__file__).resolve().parent.parent
    / "runtime" / "EDM" / "weights"
    / f"edm_w{EDM_W}_h{EDM_H}_topk{int((EDM_H//8)*(EDM_W//8)*0.35)}_outdoor.onnx"
)


def _postprocess_deploy_output(
    output: np.ndarray,
    *,
    width: int,
    height: int,
    mconf_thr: float = 0.2,
    local_resolution: int = 8,
    border_rm: int = 16,
    sigma_thr: float = 1e-6,
) -> dict:
    """Convert deploy [K,11] tensor to filtered mkpts0/mkpts1/mconf."""
    out = np.asarray(output, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] < 11:
        raise ValueError(f"unexpected EDM ONNX output shape: {out.shape}")

    kpts0_c = out[:, :2]
    kpts1_c = out[:, 2:4]
    fine_offset01 = out[:, 4:6]
    fine_offset10 = out[:, 6:8]
    pred_score01 = out[:, 8]
    pred_score10 = out[:, 9]
    mconf = out[:, 10]
    # TensorRT FP16 can overflow dual-softmax confidences to +inf; treat those as
    # invalid rather than letting every pair pass the threshold.
    if not np.isfinite(mconf).all():
        finite = np.isfinite(mconf)
        if finite.any():
            fill = float(np.median(mconf[finite]))
        else:
            fill = 0.0
        mconf = np.where(finite, mconf, fill).astype(np.float32)

    # Direction 01: ref on grid, query refined.
    mkpts0_f = kpts0_c
    mkpts1_f = kpts1_c + fine_offset01 * local_resolution
    # Direction 10: ref refined, query on grid.
    mkpts0_f_ = kpts0_c + fine_offset10 * local_resolution
    mkpts1_f_ = kpts1_c

    mkpts0 = np.concatenate([mkpts0_f, mkpts0_f_], axis=0)
    mkpts1 = np.concatenate([mkpts1_f, mkpts1_f_], axis=0)
    pred_score = np.concatenate([pred_score01, pred_score10], axis=0)
    mconf_all = np.concatenate([mconf, mconf], axis=0)

    mask = (
        np.isfinite(mconf_all)
        & np.isfinite(mkpts0).all(axis=1)
        & np.isfinite(mkpts1).all(axis=1)
        & (mconf_all > mconf_thr)
        & (mkpts0[:, 0] >= border_rm)
        & (mkpts0[:, 0] <= width - border_rm)
        & (mkpts0[:, 1] >= border_rm)
        & (mkpts0[:, 1] <= height - border_rm)
        & (mkpts1[:, 0] >= border_rm)
        & (mkpts1[:, 0] <= width - border_rm)
        & (mkpts1[:, 1] >= border_rm)
        & (mkpts1[:, 1] <= height - border_rm)
    )

    # Keep the more confident direction of each bi-directional pair.
    pred_score_mask = pred_score01 > pred_score10
    pred_score_mask = np.concatenate([pred_score_mask, ~pred_score_mask])
    pred_score_mask &= pred_score > sigma_thr
    mask &= pred_score_mask

    return {
        "mkpts0": mkpts0[mask],
        "mkpts1": mkpts1[mask],
        "mconf": mconf_all[mask],
    }


class EDMOnnxMatcher:
    """ORT session wrapper with the same gray/cell helpers as EDMMatcher."""

    load_gray = staticmethod(EDMMatcher.load_gray)
    cell_ids = staticmethod(EDMMatcher.cell_ids)
    is_refined = staticmethod(EDMMatcher.is_refined)

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_ONNX,
        *,
        provider: str = "tensorrt",
        mconf_thr: float = 0.2,
        device_id: int = 0,
        trt_cache_dir: str | Path | None = None,
        # FP16 TRT overflows coarse conf_matrix -> mconf=inf and broken poses on
        # this model/resolution. FP32 TRT matches torch accuracy (verified).
        trt_fp16: bool = False,
        width: int = EDM_W,
        height: int = EDM_H,
    ):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"EDM ONNX model not found: {self.onnx_path}")
        self.mconf_thr = float(mconf_thr)
        self.width = int(width)
        self.height = int(height)
        self.provider = provider.strip().lower()
        self.border_rm = COARSE_STRIDE * 2

        available = ort.get_available_providers()
        providers: list = []
        # 8 GB laptop GPUs OOM on the 9216x9216 coarse matrix if the CUDA EP
        # arena retains intermediates; keep limits conservative and prefer TRT.
        cuda_opts = {
            "device_id": int(device_id),
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "HEURISTIC",
        }
        if self.provider in {"tensorrt", "trt"}:
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError(
                    f"TensorrtExecutionProvider unavailable; have {available}")
            cache = Path(
                trt_cache_dir
                or (self.onnx_path.parent / "trt_cache" / self.onnx_path.stem)
            )
            cache.mkdir(parents=True, exist_ok=True)
            # Isolate FP16/FP32 engine caches so a previous broken FP16 engine is
            # not reused after flipping precision.
            cache = cache / ("fp16" if trt_fp16 else "fp32")
            cache.mkdir(parents=True, exist_ok=True)
            trt_opts = {
                "device_id": int(device_id),
                "trt_max_workspace_size": 3 * 1024 * 1024 * 1024,
                "trt_fp16_enable": bool(trt_fp16),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
                "trt_builder_optimization_level": 3,
                "trt_timing_cache_enable": True,
            }
            providers = [
                ("TensorrtExecutionProvider", trt_opts),
                ("CUDAExecutionProvider", cuda_opts),
                "CPUExecutionProvider",
            ]
        elif self.provider in {"cuda", "gpu"}:
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    f"CUDAExecutionProvider unavailable; have {available}")
            providers = [
                ("CUDAExecutionProvider", cuda_opts),
                "CPUExecutionProvider",
            ]
        elif self.provider == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(
                f"provider must be tensorrt|cuda|cpu, got {provider!r}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable residual mem patterns that fragment 8 GB cards after the
        # coarse ArgMax (needs ~2.7 GB peak on 1024x576).
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        # First TensorRT build can be long; do not kill the process early.
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_providers = list(self.session.get_providers())
        print(
            f"[edm_onnx] model={self.onnx_path.name} "
            f"requested={self.provider} active={self.active_providers}",
            flush=True,
        )

    def _to_nchw(self, gray: np.ndarray) -> np.ndarray:
        g = self.load_gray(gray)
        if g.shape != (self.height, self.width):
            g = cv2.resize(g, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return (g.astype(np.float32) / 255.0)[None, None]

    def match(self, img0, img1) -> dict:
        t0 = self._to_nchw(img0)
        t1 = self._to_nchw(img1)
        data = np.concatenate([t0, t1], axis=1)  # (1,2,H,W)
        outputs = self.session.run(None, {self.input_name: data})
        return _postprocess_deploy_output(
            outputs[0],
            width=self.width,
            height=self.height,
            mconf_thr=self.mconf_thr,
            local_resolution=COARSE_STRIDE,
            border_rm=self.border_rm,
        )

    def match_many_to_one(self, imgs0: list, img1) -> list[dict]:
        """K refs vs one query. Runs sequential N=1 (TRT engine is fixed batch-1)."""
        query = self.load_gray(img1)
        return [self.match(ref, query) for ref in imgs0]


def make_matcher(backend: str = "torch", **kwargs):
    """Factory used by stream benchmarks.

    backend: torch | onnx_cuda | onnx_tensorrt | onnx_cpu
    """
    backend = backend.strip().lower()
    if backend in {"torch", "pytorch", "fp16"}:
        return EDMMatcher(fp16=True, **{k: v for k, v in kwargs.items()
                                        if k in {"mconf_thr", "device", "max_batch", "ckpt"}})
    if backend in {"torch_fp32", "fp32"}:
        return EDMMatcher(fp16=False, **{k: v for k, v in kwargs.items()
                                         if k in {"mconf_thr", "device", "max_batch", "ckpt"}})
    provider = {
        "onnx_cuda": "cuda",
        "onnx_gpu": "cuda",
        "onnx_tensorrt": "tensorrt",
        "onnx_trt": "tensorrt",
        "onnx_cpu": "cpu",
        "cuda": "cuda",
        "tensorrt": "tensorrt",
        "trt": "tensorrt",
        "cpu": "cpu",
    }.get(backend)
    if provider is None:
        raise ValueError(f"unknown matcher backend: {backend!r}")
    onnx_path = kwargs.pop("onnx_path", os.environ.get("SFM_EDM_ONNX", str(DEFAULT_ONNX)))
    return EDMOnnxMatcher(onnx_path=onnx_path, provider=provider, **{
        k: v for k, v in kwargs.items()
        if k in {"mconf_thr", "device_id", "trt_cache_dir", "trt_fp16", "width", "height"}
    })
