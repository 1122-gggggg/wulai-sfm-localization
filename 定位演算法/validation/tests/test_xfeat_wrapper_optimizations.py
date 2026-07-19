from __future__ import annotations

from pathlib import Path
import sys

import torch


XFEAT_REPO = (
    Path(__file__).resolve().parents[4]
    / "torch_hub_cache/verlab_accelerated_features_main"
)
if str(XFEAT_REPO) not in sys.path:
    sys.path.insert(0, str(XFEAT_REPO))

from modules.xfeat import XFeat


class _FakeLighterGlue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_sizes = []

    def forward(self, data, min_conf=0.1):
        self.image_sizes.append((data["image_size0"], data["image_size1"]))
        return {"matches": [torch.tensor([[0, 0]], dtype=torch.long)]}


def test_lighterglue_keeps_indices_on_device_and_reuses_image_size_tensors():
    xfeat = XFeat.__new__(XFeat)
    torch.nn.Module.__init__(xfeat)
    xfeat.dev = torch.device("cpu")
    xfeat.kornia_available = True
    xfeat.lighterglue = _FakeLighterGlue()
    query = {
        "keypoints": torch.tensor([[1.0, 2.0]]),
        "descriptors": torch.ones((1, 64)),
        "image_size": (16, 12),
    }
    reference = {
        "keypoints": torch.tensor([[3.0, 4.0]]),
        "descriptors": torch.ones((1, 64)),
        "image_size": (16, 12),
    }

    first = xfeat.match_lighterglue_indices_tensor(query, reference)
    second = xfeat.match_lighterglue_indices_tensor(query, reference)

    assert torch.equal(first, torch.tensor([[0, 0]]))
    assert torch.equal(second, first)
    first_sizes, second_sizes = xfeat.lighterglue.image_sizes
    assert first_sizes[0] is second_sizes[0]
    assert first_sizes[1] is second_sizes[1]
