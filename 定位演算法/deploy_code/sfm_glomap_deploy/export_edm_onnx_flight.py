#!/usr/bin/env python3
"""Export EDM outdoor weights to ONNX at the flight resolution (1024x576).

The upstream deploy export is 640x480 indoor. River-site flight matching uses
1024x576 (same 16:9 as 1280x720, divisible by 32) with outdoor weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
EDM_REPO = ROOT / "runtime" / "EDM"
DEFAULT_CKPT = EDM_REPO / "weights" / "edm_outdoor.ckpt"
DEFAULT_CFG = EDM_REPO / "configs" / "edm" / "outdoor" / "edm_base.py"
SAMPLE0 = EDM_REPO / "deploy" / "edm_onnx_cpp" / "scene0707_00_15.jpg"
SAMPLE1 = EDM_REPO / "deploy" / "edm_onnx_cpp" / "scene0707_00_45.jpg"

# Flight resolution used by edm_matcher.py
W, H = 1024, 576


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=W)
    ap.add_argument("--height", type=int, default=H)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    w, h = int(args.width), int(args.height)
    if w % 32 or h % 32:
        raise SystemExit(f"resolution must be divisible by 32, got {w}x{h}")
    topk = int((h // 8) * (w // 8) * 0.35)
    out = args.out or (EDM_REPO / "weights" / f"edm_w{w}_h{h}_topk{topk}_outdoor.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(EDM_REPO))
    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(args.cfg))
    cfg.EDM.DEPLOY = True
    cfg.EDM.TEST_RES_H = h
    cfg.EDM.TEST_RES_W = w
    cfg.EDM.COARSE.TOPK = topk
    # Match production localizer confidence threshold (post-filter still at 0.2).
    cfg.EDM.COARSE.MCONF_THR = 0.2
    cfg.EDM.COARSE.BORDER_RM = 2
    cfg.EDM.NECK.NPE = [cfg.EDM.TRAIN_RES_H, cfg.EDM.TRAIN_RES_W, h, w]

    model = EDM(config=lower_config(cfg)["edm"])
    state = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    model = model.eval().cpu()

    def read_gray(path: Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Synthetic fallback if sample images are missing.
            img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        else:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img)[None, None].float() / 255.0

    img0 = read_gray(SAMPLE0 if SAMPLE0.is_file() else Path("/dev/null"))
    img1 = read_gray(SAMPLE1 if SAMPLE1.is_file() else Path("/dev/null"))
    data = torch.cat([img0, img1], dim=1)  # (1, 2, H, W)
    print(f"[export] input={tuple(data.shape)} topk={topk} out={out}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (data,),
            str(out),
            input_names=["input"],
            output_names=["output"],
            opset_version=int(args.opset),
            dynamo=False,
        )

    if args.simplify:
        import onnx
        from onnxsim import simplify

        print("[export] simplifying ...")
        model_onnx = onnx.load(str(out))
        model_simp, check = simplify(model_onnx)
        if not check:
            raise SystemExit("onnxsim failed validation")
        onnx.save(model_simp, str(out))

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"[export] wrote {out} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
