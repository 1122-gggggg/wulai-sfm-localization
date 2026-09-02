#!/usr/bin/env python3
"""First-round BoQ-ResNet50 replacement experiment for MegaLoc retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONTROL = WORKSPACE / "控制介面程式"
DEPLOY = ROOT / "deploy_code" / "sfm_glomap_deploy"
for candidate in (CONTROL, DEPLOY, Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_edm_input_resolution import _estimate_best, _rows_from_matches  # noqa: E402
from edm_matcher import EDM_H, EDM_W, EDMMatcher  # noqa: E402
from reloc_localizer_edm import EDMRelocMap, MegaLocQuery  # noqa: E402
from site_profile import load_site_profile  # noqa: E402


DEFAULT_SITE = WORKSPACE / "地圖檔" / "場域" / "river_site" / "site_profile.json"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(values):
    data = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "mean": float(np.mean(data)),
    }


def _boq_input(images: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    tensor = F.interpolate(
        tensor,
        size=(384, 384),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.inference_mode()
def _boq_descriptors(model, images: np.ndarray, device: str) -> torch.Tensor:
    descriptors, _attention = model(_boq_input(images, device))
    return F.normalize(descriptors.float(), dim=1)


def _build_boq_bank(model, reloc_map, device: str, batch_size: int):
    rows = []
    started = time.perf_counter()
    for offset in range(0, len(reloc_map.ref_names), batch_size):
        names = reloc_map.ref_names[offset : offset + batch_size]
        gray = np.stack([reloc_map.images[name] for name in names])
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        rows.append(_boq_descriptors(model, rgb, device).cpu())
    torch.cuda.synchronize()
    return torch.cat(rows).to(device), (time.perf_counter() - started) * 1e3


def _top_names(scores: torch.Tensor, names: list[str], count: int) -> list[str]:
    order = torch.topk(scores, count).indices.cpu().tolist()
    return [names[index] for index in order]


def _edm_pose(matcher, reloc_map, gray, ref_names, camera, camera_scale):
    refs = [reloc_map.images[name] for name in ref_names]
    prepared = matcher.prepare_query(gray)
    torch.cuda.synchronize()
    started = time.perf_counter()
    matches = []
    for offset in range(0, len(refs), 2):
        matches.extend(
            matcher.match_many_to_one(
                refs[offset : offset + 2],
                gray,
                prepared_query=prepared,
            )
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    geometry = (1.0, 0, 0, EDM_W, EDM_H)
    rows = _rows_from_matches(
        reloc_map, ref_names, matches, geometry, camera_scale
    )
    return _estimate_best(rows, camera), elapsed_ms


def _queries(args):
    capture = cv2.VideoCapture(str(args.video))
    source_index = 0
    selected = 0
    try:
        while selected < args.max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            index = source_index
            source_index += 1
            if index % args.stride:
                continue
            bgr = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = EDMMatcher.load_gray(bgr)
            yield index, np.ascontiguousarray(rgb), gray
            selected += 1
    finally:
        capture.release()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boq-repo", type=Path, required=True)
    parser.add_argument("--boq-weights", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--references", type=int, default=5)
    parser.add_argument("--bank-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    site = load_site_profile(args.site_profile.expanduser().resolve())
    reloc_map = EDMRelocMap.load(
        site.localization_bundle,
        expected_sha256=site.asset_sha256.localization_bundle,
    )
    boq = torch.hub.load(
        str(args.boq_repo.expanduser().resolve()),
        "get_trained_boq",
        source="local",
        backbone_name="resnet50",
        output_dim=16384,
    ).eval().cuda()
    boq_bank, bank_build_ms = _build_boq_bank(
        boq, reloc_map, "cuda", args.bank_batch_size
    )
    megaloc = MegaLocQuery(backend="tensorrt")
    megaloc_bank = torch.from_numpy(reloc_map.ref_global).to(megaloc.device)
    matcher = EDMMatcher(
        reference_feature_cache_size=192,
        host_reference_feature_cache_size=0,
        runtime_sigma_mode="reference_grid",
    )
    camera = pycolmap.Camera(
        model=site.query_camera.model,
        width=site.query_camera.width,
        height=site.query_camera.height,
        params=site.query_camera.params,
    )
    camera_scale = np.asarray(
        (site.query_camera.width / EDM_W, site.query_camera.height / EDM_H),
        np.float32,
    )
    rows = []
    vpr_times = {"megaloc": [], "boq_color": [], "boq_gray": []}
    edm_times = {"megaloc": [], "boq_color": [], "boq_gray": []}
    accepted = {"megaloc": 0, "boq_color": 0, "boq_gray": 0}
    for index, rgb, gray in _queries(args):
        query_images = {
            "boq_color": rgb,
            "boq_gray": np.repeat(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1
            ),
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        dense_descriptor = megaloc.extract_one_tensor(rgb)
        dense_refs = _top_names(
            megaloc_bank @ dense_descriptor,
            reloc_map.ref_names,
            args.references,
        )
        torch.cuda.synchronize()
        vpr_times["megaloc"].append((time.perf_counter() - started) * 1e3)
        ref_sets = {"megaloc": dense_refs}
        for mode, image in query_images.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            descriptor = _boq_descriptors(boq, image[None], "cuda")[0]
            ref_sets[mode] = _top_names(
                boq_bank @ descriptor,
                reloc_map.ref_names,
                args.references,
            )
            torch.cuda.synchronize()
            vpr_times[mode].append((time.perf_counter() - started) * 1e3)
        poses = {}
        for mode, refs in ref_sets.items():
            pose, edm_ms = _edm_pose(
                matcher, reloc_map, gray, refs, camera, camera_scale
            )
            poses[mode] = pose
            edm_times[mode].append(edm_ms)
            accepted[mode] += int(pose["accepted"])
        rows.append(
            {
                "source_index": index,
                "refs": ref_sets,
                "accepted": {key: value["accepted"] for key, value in poses.items()},
                "inliers": {key: value["inliers"] for key, value in poses.items()},
            }
        )
    result = {
        "schema": "boq-resnet50-first-round/v1",
        "official_repo": {
            "path": str(args.boq_repo.expanduser().resolve()),
            "commit": "1a4965ea7dfd9bd0dd846adf7a0e430f68101d12",
            "license": "MIT",
            "weights": str(args.boq_weights.expanduser().resolve()),
            "weights_sha256": _sha256(args.boq_weights.expanduser().resolve()),
        },
        "reference_images": "embedded grayscale JPEG replicated to RGB",
        "bank_build_ms": bank_build_ms,
        "frames": len(rows),
        "accepted": accepted,
        "vpr_ms": {key: _metric(value) for key, value in vpr_times.items()},
        "edm_ms": {key: _metric(value) for key, value in edm_times.items()},
        "no_loss": {
            key: accepted[key] >= accepted["megaloc"]
            for key in ("boq_color", "boq_gray")
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "rows"}, indent=2))
    return 0 if any(result["no_loss"].values()) else 4


if __name__ == "__main__":
    raise SystemExit(main())

