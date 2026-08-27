#!/usr/bin/env python3
"""Fail closed unless the production RTX 5060 CUDA runtime executes a tensor op."""
from __future__ import annotations

import json

import torch


def main() -> int:
    result: dict[str, object] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device": None,
        "torch": torch.__version__,
        "ok": False,
    }
    if not torch.cuda.is_available():
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    device = torch.cuda.get_device_name(0)
    result["device"] = device
    if "rtx 5060" not in device.lower():
        result["reason"] = "production requires an RTX 5060"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    value = torch.tensor([1.0, 2.0], device="cuda").square().sum()
    torch.cuda.synchronize()
    if float(value.cpu()) != 5.0:
        result["reason"] = "CUDA computation readback mismatch"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    result["ok"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
