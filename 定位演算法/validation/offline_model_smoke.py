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


LOCALIZATION_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOCALIZATION_ROOT.parent
HUB = WORKSPACE_ROOT / "執行環境" / "torch_hub_cache"
DEPLOY = LOCALIZATION_ROOT / "deploy_code" / "sfm_glomap_deploy"


class OfflineSocket(socket.socket):
    def connect(self, address):
        raise RuntimeError(f"network access attempted during offline smoke test: {address}")

    def connect_ex(self, address):
        raise RuntimeError(f"network access attempted during offline smoke test: {address}")


def blocked_getaddrinfo(*args, **kwargs):
    raise RuntimeError(f"DNS access attempted during offline smoke test: {args[:2]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["all", "edm", "megaloc", "xfeat"], default="all"
    )
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SFM_WORKSPACE_ROOT"] = str(WORKSPACE_ROOT)
    os.environ["SFM_TORCH_HUB_CACHE"] = str(HUB)
    socket.socket = OfflineSocket
    socket.getaddrinfo = blocked_getaddrinfo
    sys.path.insert(0, str(DEPLOY))

    if args.model in ("all", "edm"):
        from edm_matcher import EDMMatcher
        from reloc_localizer_edm import MEGALOC_REVISION, MegaLocQuery

        extractor = MegaLocQuery(device="cpu")
        descriptor = extractor.extract_one(np.zeros((64, 64, 3), dtype=np.uint8))
        if descriptor.ndim != 1 or not np.isfinite(descriptor).all():
            raise RuntimeError("EDM MegaLoc produced an invalid descriptor")
        if not np.isclose(np.linalg.norm(descriptor), 1.0, atol=1e-4):
            raise RuntimeError("EDM MegaLoc descriptor is not L2-normalized")
        print(
            f"EDM MegaLoc offline load/inference OK: revision={MEGALOC_REVISION} "
            f"dim={descriptor.size}"
        )
        del extractor, descriptor
        gc.collect()

        if not torch.cuda.is_available():
            raise RuntimeError("production EDM offline smoke requires CUDA")
        matcher = EDMMatcher(
            device="cuda",
            fp16=True,
            topk=3225,
            reference_cache_size=1,
        )
        yy, xx = np.indices((576, 1024))
        image = ((xx * 3 + yy * 5) % 256).astype(np.uint8)
        result = matcher.match(image, image)
        lengths = {len(np.asarray(result[key])) for key in ("mkpts0", "mkpts1", "mconf")}
        if len(lengths) != 1:
            raise RuntimeError("EDM matcher returned inconsistent match arrays")
        if not all(
            np.isfinite(np.asarray(result[key])).all()
            for key in ("mkpts0", "mkpts1", "mconf")
        ):
            raise RuntimeError("EDM matcher returned non-finite values")
        print(
            "EDM matcher offline CUDA load/inference OK: "
            f"topk={matcher.topk} fp16={matcher.fp16} matches={len(result['mconf'])}"
        )
        del matcher, result, image, xx, yy
        gc.collect()
        torch.cuda.empty_cache()

    if args.model in ("all", "megaloc"):
        from production_xfeat_tracker import MegaLocLayer
        from reloc_localizer_edm import MEGALOC_REVISION

        model = MegaLocLayer(
            np.empty((0, 8448), dtype=np.float32), input_size=322, device="cpu"
        ).model()
        print(f"MegaLoc offline load OK: revision={MEGALOC_REVISION} params={sum(p.numel() for p in model.parameters())}")
        del model
        gc.collect()

    if args.model in ("all", "xfeat"):
        from reloc_localizer_xfeat import load_xfeat

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
