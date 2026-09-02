"""Training-free MegaLoc token reduction for online VPR queries.

Implements the query-only L2 and EViT variants evaluated by Jin et al.
(arXiv:2607.15563v1).  The pinned MegaLoc source and weights stay untouched;
this module patches one already-loaded model instance for an opt-in experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


TOKEN_REDUCTION_METHODS = ("none", "l2", "evit")


@dataclass(frozen=True)
class TokenReductionConfig:
    method: str = "none"
    keep_ratio: float = 1.0
    layer: int = 6
    fuse_inattentive: bool = True

    def validate(self, depth: int | None = None) -> None:
        if self.method not in TOKEN_REDUCTION_METHODS:
            raise ValueError(
                f"token reduction method must be one of {TOKEN_REDUCTION_METHODS}"
            )
        if (
            isinstance(self.keep_ratio, bool)
            or not isinstance(self.keep_ratio, (int, float))
            or not math.isfinite(float(self.keep_ratio))
            or not 0.0 < float(self.keep_ratio) <= 1.0
        ):
            raise ValueError("token reduction keep_ratio must be within (0, 1]")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer <= 0:
            raise ValueError("token reduction layer must be a positive integer")
        if depth is not None and self.layer > depth:
            raise ValueError(
                f"token reduction layer {self.layer} exceeds backbone depth {depth}"
            )
        if not isinstance(self.fuse_inattentive, bool):
            raise ValueError("fuse_inattentive must be boolean")
        if self.method == "none" and float(self.keep_ratio) != 1.0:
            raise ValueError("token reduction method 'none' requires keep_ratio=1.0")


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        tokens,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )


def _l2_reduce(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    cls, patches = x[:, :1], x[:, 1:]
    keep = max(1, math.floor(float(keep_ratio) * patches.shape[1]))
    if keep >= patches.shape[1]:
        return x
    scores = torch.linalg.vector_norm(patches, ord=2, dim=-1)
    indices = torch.topk(scores, keep, dim=1, largest=True, sorted=False).indices
    # Preserve the original spatial order. ViT attention is permutation equivariant,
    # but stable ordering removes an avoidable floating-point reduction difference.
    indices = torch.sort(indices, dim=1).values
    return torch.cat((cls, _gather_tokens(patches, indices)), dim=1)


def _attention_with_probs(attention, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, channels = x.shape
    qkv = attention.qkv(x).reshape(
        batch,
        tokens,
        3,
        attention.num_heads,
        attention.head_dim,
    )
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    probs = ((query @ key.transpose(-2, -1)) * attention.scale).softmax(dim=-1)
    probs = attention.attn_drop(probs)
    output = (probs @ value).transpose(1, 2).reshape(batch, tokens, channels)
    output = attention.proj_drop(attention.proj(output))
    return output, probs


def _evit_block(block, x: torch.Tensor, config: TokenReductionConfig) -> torch.Tensor:
    attention_output, attention = _attention_with_probs(block.attn, block.norm1(x))
    x = x + block.ls1(attention_output)
    patches = x[:, 1:]
    keep = max(1, math.ceil(float(config.keep_ratio) * patches.shape[1]))
    if keep < patches.shape[1]:
        scores = attention[:, :, 0, 1:].mean(dim=1)
        indices = torch.topk(scores, keep, dim=1, largest=True, sorted=True).indices
        retained = _gather_tokens(patches, indices)
        if config.fuse_inattentive:
            selected = torch.zeros_like(scores, dtype=torch.bool)
            selected.scatter_(1, indices, True)
            discarded = patches[~selected].reshape(
                patches.shape[0], patches.shape[1] - keep, patches.shape[2]
            )
            discarded_scores = scores[~selected].reshape(
                scores.shape[0], scores.shape[1] - keep
            )
            fused = torch.sum(
                discarded * discarded_scores.unsqueeze(-1), dim=1, keepdim=True
            )
            retained = torch.cat((retained, fused), dim=1)
        x = torch.cat((x[:, :1], retained), dim=1)
    return x + block.ls2(block.mlp(block.norm2(x)))


def install_token_reduction(model, config: TokenReductionConfig) -> None:
    """Install one query-only token-reduced forward on a loaded MegaLoc model."""
    config.validate()
    if config.method == "none":
        model.token_reduction = config
        return
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if backbone is None or blocks is None:
        raise ValueError("MegaLoc model does not expose a compatible ViT backbone")
    config.validate(len(blocks))
    if getattr(backbone, "_token_reduction_installed", False):
        raise ValueError("token reduction is already installed on this MegaLoc model")

    aggregator = getattr(getattr(model, "aggregator", None), "agg", None)
    min_tokens = int(getattr(aggregator, "num_clusters", 0)) + 1

    def reduced_forward(images: torch.Tensor):
        batch, _channels, height, width = images.shape
        x = backbone.patch_embed(images)
        cls = backbone.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + backbone.interpolate_pos_encoding(x, height, width)

        for index, block in enumerate(backbone.blocks, start=1):
            if config.method == "evit" and index == config.layer:
                x = _evit_block(block, x, config)
            else:
                x = block(x)
                if config.method == "l2" and index == config.layer:
                    x = _l2_reduce(x, config.keep_ratio)

        x = backbone.norm(x)
        patch_tokens = x[:, 1:]
        if patch_tokens.shape[1] < min_tokens:
            raise ValueError(
                "token reduction leaves too few patches for MegaLoc aggregation: "
                f"{patch_tokens.shape[1]} < {min_tokens}"
            )
        backbone.last_token_count = int(patch_tokens.shape[1])
        # MegaLoc's 1x1-convolutional aggregator is spatially permutation invariant;
        # an N x 1 map therefore supports the irregular retained token sequence.
        patch_features = patch_tokens.transpose(1, 2).unsqueeze(-1)
        return patch_features, x[:, 0]

    backbone.forward = reduced_forward
    backbone._token_reduction_installed = True
    model.token_reduction = config
