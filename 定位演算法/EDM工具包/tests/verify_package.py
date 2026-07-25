#!/usr/bin/env python3
"""Validate an extracted EDM package before flight or TensorRT conversion."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in (path, *path.parents):
        raise ValueError(f"manifest/config path escapes package: {relative}")
    return path


def verify_manifest(root: Path) -> int:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {manifest}")
    checked = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = _resolve_inside(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"manifest file missing: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


def load_config(root: Path) -> dict:
    config = json.loads((root / "config.json").read_text())
    for key, relative in config["paths"].items():
        if not _resolve_inside(root, relative).exists():
            raise FileNotFoundError(f"config path {key!r} missing: {relative}")
    onnx = config["onnx_reference"]
    if onnx["production_profile_compatible"] is not False:
        raise ValueError("reference ONNX must not be selected as the production backend")
    return config


def configure_caches(root: Path) -> tuple[Path, Path]:
    packaged_torch = root / "runtime/torch_hub_cache"
    packaged_hf = root / "runtime/hf_cache"
    if packaged_torch.is_dir():
        os.environ.setdefault("TORCH_HOME", str(packaged_torch))
    if packaged_hf.is_dir():
        os.environ.setdefault("HF_HOME", str(packaged_hf))
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return torch_home, hf_home


def verify_megaloc_cache(torch_home: Path, hf_home: Path) -> None:
    required = (
        torch_home / "hub/gmberton_MegaLoc_main/hubconf.py",
        torch_home / "hub/facebookresearch_dinov2_main/hubconf.py",
        torch_home / "hub/checkpoints/dinov2_vitl14_pretrain.pth",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"MegaLoc cache missing: {path}")
    models = list(
        (hf_home / "hub/models--gberton--MegaLoc/snapshots").glob(
            "*/model.safetensors"
        )
    )
    if not models or not any(path.is_file() for path in models):
        raise FileNotFoundError(
            f"MegaLoc model.safetensors missing below {hf_home / 'hub'}"
        )


def runtime_smoke(config: dict, require_rtx5060: bool, max_vram_mib: int) -> dict:
    import cv2
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if require_rtx5060 and "RTX 5060" not in device_name:
        raise RuntimeError(f"expected RTX 5060, detected {device_name}")
    if capability < (12, 0) or "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f"PyTorch build does not cover Blackwell sm_120: capability={capability}, "
            f"arch_list={torch.cuda.get_arch_list()}"
        )

    sys.path.insert(0, str(ROOT / "deploy"))
    from edm_matcher import EDMMatcher
    from production_edm_tracker import EDMConfig, ProductionEDMTracker
    from reloc_localizer_edm import Camera, EDMRelocMap

    bundle = _resolve_inside(ROOT, config["paths"]["bundle"])
    reloc_map = EDMRelocMap.load(bundle)
    profile_path = ROOT / "deploy/profiles/balanced_rtx5060_candidate.json"
    profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
    coarse_topk = int(profile.get("matcher", {}).get("coarse_topk", 2304))
    matcher = EDMMatcher(topk=coarse_topk, fp16=True)

    first_ref = reloc_map.ref_names[0]
    self_match = matcher.match(reloc_map.images[first_ref], reloc_map.images[first_ref])
    if len(self_match["mkpts0"]) < 100:
        raise RuntimeError(f"EDM self-match returned only {len(self_match['mkpts0'])} matches")
    one_grid_side = np.logical_xor(
        EDMMatcher.is_refined(self_match["mkpts0"]),
        EDMMatcher.is_refined(self_match["mkpts1"]),
    )
    if float(one_grid_side.mean()) < 0.99:
        raise RuntimeError("EDM cell-anchor contract failed")

    pc = config["pnp_camera"]
    camera = Camera(pc["model"], pc["width"], pc["height"], list(pc["params"]))
    allowed = {field.name for field in dataclasses.fields(EDMConfig)}
    tracker_kwargs = {
        key: value for key, value in config.get("tracker_defaults", {}).items()
        if key in allowed
    }
    torch.cuda.reset_peak_memory_stats()
    tracker = ProductionEDMTracker(
        reloc_map,
        camera,
        EDMConfig(**tracker_kwargs),
        matcher=matcher,
    )
    frame = cv2.cvtColor(reloc_map.images[first_ref], cv2.COLOR_GRAY2BGR)
    result = tracker.localize(frame)
    if not result.get("ok"):
        raise RuntimeError(f"end-to-end localization smoke failed: {result}")
    if result.get("global_retrieval_calls") != 1:
        raise RuntimeError(f"unexpected MegaLoc call contract: {result}")
    peak_reserved_mib = int(torch.cuda.max_memory_reserved() / 2**20)
    if peak_reserved_mib > max_vram_mib:
        raise RuntimeError(
            f"PyTorch peak reserved VRAM {peak_reserved_mib} MiB exceeds {max_vram_mib} MiB"
        )
    return {
        "device": device_name,
        "compute_capability": list(capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "refs": len(reloc_map.ref_names),
        "self_matches": len(self_match["mkpts0"]),
        "pnp_inliers": int(result["inliers"]),
        "peak_reserved_mib": peak_reserved_mib,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--require-rtx5060", action="store_true")
    parser.add_argument("--max-vram-mib", type=int, default=7600)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    files = verify_manifest(ROOT)
    config = load_config(ROOT)
    torch_home, hf_home = configure_caches(ROOT)
    report = {
        "manifest_files": files,
        "torch_home": str(torch_home),
        "hf_home": str(hf_home),
        "onnx_reference_only": True,
    }
    if not args.static_only:
        verify_megaloc_cache(torch_home, hf_home)
        report.update(runtime_smoke(config, args.require_rtx5060, args.max_vram_mib))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("PACKAGE VERIFY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
