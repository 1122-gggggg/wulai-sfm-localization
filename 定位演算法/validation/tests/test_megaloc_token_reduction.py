from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
MEGALOC = (
    Path(__file__).resolve().parents[3]
    / "執行環境"
    / "torch_hub_cache"
    / "gmberton_MegaLoc_main"
)
for candidate in (DEPLOY, MEGALOC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

torch = pytest.importorskip("torch")

from megaloc_model import DINOv2  # noqa: E402
from megaloc_token_reduction import (  # noqa: E402
    TokenReductionConfig,
    install_token_reduction,
)


def _toy_model():
    return SimpleNamespace(
        backbone=DINOv2(
            image_size=8,
            patch_size=2,
            embed_dim=8,
            depth=4,
            num_heads=2,
        ).eval(),
        aggregator=SimpleNamespace(agg=SimpleNamespace(num_clusters=2)),
    )


@pytest.mark.parametrize(
    ("method", "expected_tokens"),
    (("l2", 8), ("evit", 9)),
)
def test_query_token_reduction_returns_irregular_aggregator_input(
    method: str,
    expected_tokens: int,
) -> None:
    model = _toy_model()
    install_token_reduction(
        model,
        TokenReductionConfig(method=method, keep_ratio=0.5, layer=2),
    )

    features, cls = model.backbone(torch.randn(1, 3, 8, 8))

    assert features.shape == (1, 8, expected_tokens, 1)
    assert cls.shape == (1, 8)
    assert model.backbone.last_token_count == expected_tokens


def test_none_mode_does_not_require_or_patch_a_backbone() -> None:
    model = SimpleNamespace()
    config = TokenReductionConfig()

    install_token_reduction(model, config)

    assert model.token_reduction == config


def test_token_reduction_rejects_invalid_or_aggregator_incompatible_settings() -> None:
    with pytest.raises(ValueError, match="keep_ratio"):
        TokenReductionConfig(method="l2", keep_ratio=0.0).validate()
    with pytest.raises(ValueError, match="exceeds backbone depth"):
        install_token_reduction(
            _toy_model(),
            TokenReductionConfig(method="l2", keep_ratio=0.5, layer=5),
        )

    model = _toy_model()
    model.aggregator.agg.num_clusters = 15
    install_token_reduction(
        model,
        TokenReductionConfig(method="l2", keep_ratio=0.5, layer=2),
    )
    with pytest.raises(ValueError, match="too few patches"):
        model.backbone(torch.randn(1, 3, 8, 8))

