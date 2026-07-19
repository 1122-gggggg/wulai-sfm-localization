#!/usr/bin/env python3
"""Load the shipped localization models while all socket connects are blocked."""
from __future__ import annotations

import argparse
import gc
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HUB = ROOT / "torch_hub_cache"
DEPLOY = ROOT / "sfm_system" / "定位" / "deploy_code" / "sfm_glomap_deploy"


class OfflineSocket(socket.socket):
    def connect(self, address):
        raise RuntimeError(f"network access attempted during offline smoke test: {address}")

    def connect_ex(self, address):
        raise RuntimeError(f"network access attempted during offline smoke test: {address}")


def blocked_getaddrinfo(*args, **kwargs):
    raise RuntimeError(f"DNS access attempted during offline smoke test: {args[:2]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "megaloc", "xfeat"], default="all")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    socket.socket = OfflineSocket
    socket.getaddrinfo = blocked_getaddrinfo
    sys.path.insert(0, str(DEPLOY))
    from production_xfeat_tracker import MegaLocLayer
    from reloc_localizer_xfeat import MEGALOC_REVISION, load_xfeat

    if args.model in ("all", "megaloc"):
        model = MegaLocLayer(
            np.empty((0, 8448), dtype=np.float32), input_size=322, device="cpu"
        ).model()
        print(f"MegaLoc offline load OK: revision={MEGALOC_REVISION} params={sum(p.numel() for p in model.parameters())}")
        del model
        gc.collect()

    if args.model in ("all", "xfeat"):
        model = load_xfeat(2048)
        device = model.dev
        keypoints = torch.stack((torch.linspace(16, 624, 32, device=device),
                                 torch.linspace(16, 464, 32, device=device)), dim=1)
        descriptors = torch.nn.functional.normalize(
            torch.arange(32 * 64, dtype=torch.float32, device=device).reshape(32, 64), dim=1
        )
        features = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image_size": (640, 480),
        }
        _mk0, _mk1, matches = model.match_lighterglue(features, features, min_conf=0.1)
        if model.lighterglue is None:
            raise RuntimeError("LighterGlue remained unloaded after matcher smoke test")
        print(
            f"XFeat+LighterGlue offline load/match OK: "
            f"params={sum(p.numel() for p in model.parameters())} matches={len(matches)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
