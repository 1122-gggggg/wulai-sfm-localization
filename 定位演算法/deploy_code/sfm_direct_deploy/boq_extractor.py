"""BoQ-ResNet50 global descriptor extractor for the direct deployment path.

Self-contained VPR front-end: ResNet50-crop backbone + BoQ aggregator producing
a 16384-d L2-normalised descriptor per image. Vendored from upstream
Bag-of-Queries (``src/backbones.py`` ResNet + ``src/boq.py`` aggregator, MIT,
Amar Ali-bey) via the deleted ``sfm_glomap_deploy/boq_net.py`` /
``boq_query.py``; it deliberately imports no deleted repo module and depends
only on torch / torchvision / numpy / cv2.

Weights (read-only, never copied into the repo)::

    /home/allen/.cache/torch/hub/checkpoints/resnet50_16384.pth

loaded with ``strict=True`` and a hard-pinned SHA-256 (``BOQ_WEIGHTS_SHA256``);
any mismatch raises before the model is built.

Preprocessing contract (must stay bit-for-bit identical to the old
``boq_query.BoQQuery._preprocess`` — the frozen reference bank was extracted
with exactly this transform; any deviation silently re-ranks retrieval):

* input: HxWx3 uint8 RGB array (``extract_bank`` reads files with cv2 and
  converts BGR→RGB first),
* uint8 → float32 / 255,
* ``F.interpolate`` to 384×384 square, ``mode="bicubic"``,
  ``align_corners=False``, ``antialias=True``,
* ImageNet mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225),
* model forward, then L2-normalise (``F.normalize`` eps 1e-12).

Public API::

    extractor = BoQExtractor()                      # default cuda
    desc = extractor.extract_from_array(rgb)        # (16384,) float32, norm 1
    bank = extractor.extract_bank(image_paths)      # (N, 16384) float32

Fidelity gate: descriptors re-extracted with this module must reach cosine
≥ 0.999 against the matching rows of the FairVPRRace bank
``/tmp/vpr_fair_out/boq_references_sideview.npy``. If fidelity ever drops
below that, suspect a preprocessing divergence first — do not retune.

Phase 2 handover — switching live_provider from MegaLoc to this module.
``live_provider.py`` is untouched by the present change; Phase 2 owns it.
Exact touch points (line numbers against the current file):

1. ``megaloc_descriptor_from_array`` (~line 125): replace the whole function
   with a thin wrapper over ``BoQExtractor.extract_from_array``. The two
   transforms differ in every constant that matters: bilinear 322×322
   (``runtime.input_size``) → bicubic 384×384 (``BOQ_INPUT``); output
   8448-d → 16384-d (``BOQ_DIM``). Keep the finite / zero-norm guards.
2. ``LiveMapEDMProvider.__init__`` kwargs ``megaloc_source`` /
   ``megaloc_checkpoint`` (~lines 185-186) → a single BoQ weights path
   (default ``BOQ_WEIGHTS`` below, env ``SFM_BOQ_WEIGHTS`` override); update
   the matching ``super().__init__`` arguments (~lines 207-208) and check the
   vendored ``FinalMapEDMProvider`` constructor for the same two parameters.
3. ``_megaloc()`` loader (~lines 307-317, ``load_offline_megaloc_runtime``)
   → construct ``BoQExtractor`` once and cache it on the instance.
4. ``ensure_models`` warm-up (~line 327) and ``_localize_array`` descriptor
   call (~line 379): unchanged shapes of code, new extractor object.
5. ``self.last_retrieval_source = "megaloc"`` (~line 399) → ``"boq"``.
6. Docstrings mentioning MegaLoc (module docstring lines 1-21, the
   descriptor docstring ~128-135, ``localize_array`` ~344-351): rewrite to BoQ.
7. ``direct_paths.py``: MegaLoc source/checkpoint resolution
   (``_DEFAULT_MEGALOC_SOURCE``, HF-cache fallback) → BoQ weights resolution.
8. Descriptor dimension 8448→16384: retrieval itself
   (``rank_reference_indices``, ``apply_reference_bank``) is dim-agnostic as
   long as query and bank agree, and ``direct_map._load_reference_names``
   only checks row counts — no dim assert needs editing there.
9. Bank files (Phase 3 regenerates the release; names are fixed here so both
   phases agree): descriptors ``megaloc_references_sideview.npy`` →
   ``boq_references_sideview.npy``, names
   ``megaloc_references_sideview.names.json`` →
   ``boq_references_sideview.names.json``, ``reference_bank.name``
   ``"sideview"`` → ``"sideview-boq"`` (bundle ``reference_bank`` object +
   profile ``reloc.reference_bank`` string).

OOD threshold note: the direct path (``two_rate_tracker.py``,
``live_provider.py``, ``direct_map.py``, ``direct_profile.py``,
``direct_paths.py``, ``direct_localizer_adapter.py``, ``vendor/``) contains
no reference to OOD / ``ood_early_exit`` / ``vpr_top1`` (word-boundary grep
over those files returns nothing; the only ``good`` hits are unrelated local
variable names). 結論:無引用，不需重調. (The glomap-side ``ood_early_exit``
default ``vpr_top1_out = 0.32`` is already on the BoQ score scale.)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

BOQ_INPUT = 384
BOQ_DIM = 16384
BOQ_MODEL_IDENTITY = "boq:resnet50-16384"
BOQ_WEIGHTS_SHA256 = (
    "4691d1545db847da2c0ba911f34e6c520a5da63f8b94c6415fe87d0d8800ebaa"
)
BOQ_WEIGHTS = Path("/home/allen/.cache/torch/hub/checkpoints/resnet50_16384.pth")
BOQ_WEIGHTS_ENV = "SFM_BOQ_WEIGHTS"
BOQ_FP16_ENV = "SFM_BOQ_FP16"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetCrop(nn.Module):
    """ResNet backbone cropped after layer3 (upstream BoQ training config)."""

    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
    }

    def __init__(self, backbone_name: str = "resnet50") -> None:
        super().__init__()
        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Backbone {backbone_name} is not recognized! "
                f"Supported: {list(self.AVAILABLE_MODELS.keys())}"
            )
        # weights=None on purpose: the BoQ checkpoint below overwrites every
        # parameter via strict load, so ImageNet init would only cost a
        # download and nondeterminism before the load.
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=None)
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        if backbone_name in ("resnet18", "resnet34"):
            self.out_channels = resnet.layer3[-1].conv2.out_channels
        else:
            self.out_channels = resnet.layer3[-1].conv3.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim: int, num_queries: int, nheads: int = 8):
        super().__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nheads,
            dim_feedforward=4 * in_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # Training-only stability path (cached in eval, kept for key compat).
        self.self_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_q = torch.nn.LayerNorm(in_dim)

        self.cross_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor):
        batch = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(batch, 1, 1)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQAggregator(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        proj_channels: int = 512,
        num_queries: int = 32,
        num_layers: int = 2,
        row_dim: int = 32,
    ):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(
            in_channels, proj_channels, kernel_size=3, padding=1
        )
        self.norm_input = torch.nn.LayerNorm(proj_channels)

        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList(
            [
                BoQBlock(in_dim, num_queries, nheads=in_dim // 64)
                for _ in range(num_layers)
            ]
        )

        self.fc = torch.nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns


class BoQVPRModel(torch.nn.Module):
    """ResNet50-crop + BoQ aggregator producing a 16384-d global descriptor."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNetCrop(backbone_name="resnet50")
        self.aggregator = BoQAggregator(
            in_channels=self.backbone.out_channels,
            proj_channels=512,
            num_queries=64,
            num_layers=2,
            row_dim=32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        descriptors, _ = self.aggregator(x)
        return descriptors


def resolve_weights_path(weights_path: str | Path | None = None) -> Path:
    """Resolve the BoQ checkpoint location (explicit > env > default)."""
    configured = (os.environ.get(BOQ_WEIGHTS_ENV, "") or "").strip()
    return Path(weights_path or configured or BOQ_WEIGHTS).expanduser()


def verify_weights_sha256(
    path: Path, expected: str = BOQ_WEIGHTS_SHA256
) -> Path:
    """SHA-pin the checkpoint; raise on any mismatch or absent file."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"BoQ weights are absent: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"BoQ weights SHA-256 mismatch: {resolved} "
            f"has {actual}, expected {expected}"
        )
    return resolved


def load_boq_model(weights_path: str | Path) -> BoQVPRModel:
    """Strict-load the sanctioned BoQ checkpoint (SHA verified by the caller)."""
    model = BoQVPRModel()
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _default_device(device: str | None) -> str:
    if device is not None and str(device).strip():
        wanted = str(device).strip()
        if wanted == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return wanted
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fp16_enabled(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    raw = os.environ.get(BOQ_FP16_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class BoQExtractor:
    """BoQ-ResNet50 descriptor extractor (``boq:resnet50-16384``)."""

    def __init__(
        self,
        device: str = "cuda",
        *,
        weights_path: str | Path | None = None,
        weights_sha256: str | None = None,
        fp16: bool | None = None,
        input_size: int = BOQ_INPUT,
    ) -> None:
        self.device = _default_device(device)
        self.input_size = int(input_size)
        resolved = resolve_weights_path(weights_path)
        verify_weights_sha256(resolved, weights_sha256 or BOQ_WEIGHTS_SHA256)
        self.weights_path = resolved
        self.model = load_boq_model(resolved).to(self.device)
        torch_device = torch.device(self.device)
        if fp16 is None:
            fp16 = _fp16_enabled(torch_device)
        self.fp16 = bool(fp16) and torch_device.type == "cuda"
        self._normalization_tensors = None

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """uint8 RGB → float /255 → bicubic 384 → ImageNet-normalised tensor.

        Identical to the old ``boq_query.BoQQuery._preprocess``; keep it so.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("BoQ input must be an HxWx3 uint8 RGB array")
        source = torch.from_numpy(np.ascontiguousarray(rgb[..., :3])).permute(2, 0, 1)
        x = source.to(torch.device(self.device))
        x = x.unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        normalization = self._normalization_tensors
        device = torch.device(self.device)
        if normalization is None or normalization[0].device != device:
            mean = torch.tensor(
                list(_IMAGENET_MEAN), dtype=torch.float32, device=device
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                list(_IMAGENET_STD), dtype=torch.float32, device=device
            ).view(1, 3, 1, 1)
            normalization = (mean, std)
            self._normalization_tensors = normalization
        return ((x - normalization[0]) / normalization[1]).contiguous()

    @torch.inference_mode()
    def extract_one_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """One L2-normalised ``(16384,)`` tensor on ``self.device``."""
        x = self._preprocess(rgb)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.fp16
        ):
            output = self.model(x)
        desc = F.normalize(output.float(), dim=1, eps=1e-12)[0]
        if desc.shape != (BOQ_DIM,):
            raise RuntimeError(
                f"BoQ descriptor shape {tuple(desc.shape)!r} violates "
                f"{BOQ_DIM}-D contract"
            )
        return desc

    @torch.inference_mode()
    def extract_from_array(self, rgb: np.ndarray) -> np.ndarray:
        """One L2-normalised ``(16384,)`` float32 descriptor from uint8 RGB."""
        desc = self.extract_one_tensor(rgb).cpu().numpy().astype(
            np.float32, copy=False
        )
        if not np.isfinite(desc).all():
            raise RuntimeError("BoQ emitted a non-finite descriptor")
        norm = float(np.linalg.norm(desc))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("BoQ emitted a zero-norm descriptor")
        return np.ascontiguousarray(desc / norm, dtype=np.float32)

    @torch.inference_mode()
    def extract_bank(
        self, image_paths: Sequence[str | Path], *, batch_size: int = 8
    ) -> np.ndarray:
        """Batched ``(N, 16384)`` float32 L2-normalised bank from image files."""
        paths = [Path(p) for p in image_paths]
        if not paths:
            raise ValueError("cannot extract a BoQ bank from an empty path list")
        if batch_size <= 0:
            raise ValueError("BoQ batch_size must be positive")
        device = torch.device(self.device)
        out: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start : start + batch_size]:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"BoQ bank image is unreadable: {path}")
                batch.append(self._preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            x = torch.cat(batch, dim=0).to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.fp16
            ):
                raw = self.model(x)
            array = F.normalize(raw.float(), dim=1, eps=1e-12).cpu().numpy()
            if array.ndim != 2 or array.shape[1] != BOQ_DIM:
                raise RuntimeError(
                    f"BoQ bank batch shape {array.shape!r} violates "
                    f"{BOQ_DIM}-D contract"
                )
            if not np.isfinite(array).all():
                raise RuntimeError("BoQ emitted a non-finite bank batch")
            norms = np.linalg.norm(array, axis=1)
            if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
                raise RuntimeError("BoQ emitted a zero-norm bank row")
            out.append(np.ascontiguousarray(array / norms[:, None], dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(out, axis=0), dtype=np.float32)
