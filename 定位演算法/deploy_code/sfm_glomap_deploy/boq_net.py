# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root (MIT). Vendored from upstream
# Bag-of-Queries src/backbones.py (ResNet) + src/boq.py (aggregator) and
# adapted for offline deployment: no torch.hub calls, no sys.path hacks,
# ImageNet-pretrained init replaced by explicit weights load (the BoQ
# checkpoint carries every parameter, so random init + strict load is exact
# and needs no network).
# ----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision


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


def load_boq_model(weights_path: str | Path) -> BoQVPRModel:
    """Strict-load the sanctioned BoQ checkpoint (SHA verified by the caller)."""
    model = BoQVPRModel()
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()
