"""BoQ-ResNet50 global retrieval front-end (production VPR).

Replaces MegaLoc: the bundle's ``ref_global`` is a BoQ bank
(``BOQ_DIM``-dimensional, L2-normalised), and this module extracts the
matching query descriptor. Same duck-type contract the localizers use:

- ``.device`` (str)
- ``.extract_one_tensor(rgb)`` -> ``torch.Tensor`` ``(BOQ_DIM,)`` on
  ``self.device``, L2-normalised
- ``.extract_one(rgb)`` -> ``np.ndarray`` ``(BOQ_DIM,)`` float32
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from artifact_integrity import verify_sha256
from boq_net import load_boq_model

BOQ_INPUT = 384
BOQ_DIM = 16384
BOQ_MODEL_IDENTITY = "boq:resnet50-16384"
BOQ_WEIGHTS_SHA256 = "4691d1545db847da2c0ba911f34e6c520a5da63f8b94c6415fe87d0d8800ebaa"


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
BOQ_WEIGHTS = TORCH_HUB_DIR / "checkpoints" / "boq" / "resnet50_16384.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fp16_enabled(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    raw = os.environ.get("SFM_BOQ_FP16", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class BoQQuery:
    """BoQ-ResNet50 query extractor for bundles whose ref_global is BoQ."""

    def __init__(
        self,
        device: str = DEVICE,
        input_size: int = BOQ_INPUT,
        *,
        weights_path: str | Path | None = None,
        weights_sha256: str | None = None,
        fp16: bool | None = None,
    ):
        self.device = str(device)
        self.input_size = int(input_size)
        resolved = Path(
            weights_path or os.environ.get("SFM_BOQ_WEIGHTS", "") or BOQ_WEIGHTS
        )
        verify_sha256(
            resolved, weights_sha256 or os.environ.get("SFM_BOQ_WEIGHTS_SHA256", "")
            or BOQ_WEIGHTS_SHA256
        )
        self.model = load_boq_model(resolved).to(self.device)
        torch_device = torch.device(self.device)
        if fp16 is None:
            fp16 = _fp16_enabled(torch_device)
        self.fp16 = bool(fp16) and torch_device.type == "cuda"
        self._input_staging = None
        self._normalization_tensors = None

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """Transfer uint8 once, then resize and normalize on the target device."""
        source = torch.from_numpy(np.ascontiguousarray(rgb[..., :3])).permute(2, 0, 1)
        device = torch.device(self.device)
        if device.type == "cuda":
            staging = self._input_staging
            if staging is None or tuple(staging.shape) != tuple(source.shape):
                staging = torch.empty(tuple(source.shape), dtype=torch.uint8, pin_memory=True)
                self._input_staging = staging
            staging.copy_(source)
            x = staging.to(device, non_blocking=True)
        else:
            x = source.to(device)
        x = x.unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        normalization = self._normalization_tensors
        if normalization is None or normalization[0].device != device:
            mean = torch.tensor(
                [0.485, 0.456, 0.406], dtype=torch.float32, device=device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [0.229, 0.224, 0.225], dtype=torch.float32, device=device
            ).view(1, 3, 1, 1)
            normalization = (mean, std)
            self._normalization_tensors = normalization
        return ((x - normalization[0]) / normalization[1]).contiguous()

    @torch.inference_mode()
    def extract_one_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        x = self._preprocess(rgb)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.fp16
        ):
            output = self.model(x)
        return F.normalize(output.float(), dim=1, eps=1e-12)[0]

    @torch.inference_mode()
    def extract_one(self, rgb: np.ndarray) -> np.ndarray:
        return self.extract_one_tensor(rgb).cpu().numpy().astype(np.float32, copy=False)
