#!/usr/bin/env python3
"""Rebuild a production EDM bundle's global retrieval bank with BoQ-ResNet50.

Surgical migration: only ``ref_global`` + VPR meta fields change. Reference
JPEGs, xyz LUTs, centers/yaws/covis and every other meta key are byte-untouched,
so geometry and the EDM matcher path are exactly preserved.

Usage:
    python3 rebuild_bundle_vpr_boq.py --in bundle.pt --out bundle.boq.pt
    # verify, then move over the original and update every SHA pin:
    # site_profile.json asset_sha256.localization_bundle,
    # compat/localizer_edm_manifest.json artifacts.bundle.sha256,
    # mission selection localizer.sha256, MANIFEST.tsv, SHA256SUMS.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEPLOY = Path(__file__).resolve().parents[2] / "deploy_code" / "sfm_glomap_deploy"
sys.path.insert(0, str(DEPLOY))

from boq_query import BOQ_DIM, BOQ_INPUT, BoQQuery  # noqa: E402

BATCH = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    print(f"load {src}", flush=True)
    bundle = torch.load(str(src), map_location="cpu", weights_only=False)
    names = list(bundle["ref_names"])
    old = np.asarray(bundle["ref_global"], dtype=np.float32)
    print(f"refs={len(names)} old ref_global={old.shape}", flush=True)
    assert old.shape == (len(names), 8448), old.shape

    boq = BoQQuery(device="cuda" if torch.cuda.is_available() else "cpu")
    descs: list[np.ndarray] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        batch: list[torch.Tensor] = []
        for i, name in enumerate(names):
            arr = np.frombuffer(bundle["refs"][name]["image_jpg"], np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"reference JPEG failed to decode: {name}")
            batch.append(boq._preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if len(batch) == BATCH or i == len(names) - 1:
                x = torch.cat(batch, dim=0)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=boq.fp16
                ):
                    out = boq.model(x)
                descs.append(
                    F.normalize(out.float(), dim=1, eps=1e-12).cpu().numpy().astype(
                        np.float32
                    )
                )
                batch = []
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)
    bank = np.concatenate(descs, axis=0)
    assert bank.shape == (len(names), BOQ_DIM), bank.shape
    norms = np.linalg.norm(bank, axis=1)
    print(
        f"bank {bank.shape} norm {norms.min():.6f}/{norms.max():.6f} "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    bundle["ref_global"] = np.ascontiguousarray(bank)
    meta = dict(bundle["meta"])
    meta["vpr"] = "BoQ"
    meta["bundle_vpr"] = "boq"
    meta["vpr_input"] = BOQ_INPUT
    bundle["meta"] = meta

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(bundle, str(tmp))
    tmp.replace(dst)
    print(f"wrote {dst} sha256={sha256_file(dst)}", flush=True)


if __name__ == "__main__":
    main()
