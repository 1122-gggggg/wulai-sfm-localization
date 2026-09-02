#!/usr/bin/env python3
"""Deterministic query-only MegaLoc token-reduction screening benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONTROL = WORKSPACE / "控制介面程式"
DEPLOY = ROOT / "deploy_code" / "sfm_glomap_deploy"
for candidate in (CONTROL, DEPLOY):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from reloc_localizer_edm import EDMRelocMap, MegaLocQuery  # noqa: E402
from site_profile import load_site_profile  # noqa: E402


DEFAULT_SITE = WORKSPACE / "地圖檔" / "場域" / "river_site" / "site_profile.json"


def _percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "mean": float(np.mean(data)),
    }


def _selected_frames(video: Path, stride: int, limit: int):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    source_index = 0
    selected = 0
    try:
        while not limit or selected < limit:
            ok, bgr = capture.read()
            if not ok:
                break
            index = source_index
            source_index += 1
            if index % stride:
                continue
            rgb = cv2.cvtColor(cv2.resize(bgr, (1280, 720)), cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            yield index, hashlib.sha256(rgb.tobytes()).hexdigest(), rgb
            selected += 1
    finally:
        capture.release()


def _extract_pass(args, *, method: str, expected_schedule=None):
    ratio = 1.0 if method == "none" else float(args.keep_ratio)
    load_started = time.perf_counter()
    query = MegaLocQuery(
        backend=args.backend,
        token_reduction_method=method,
        token_reduction_keep_ratio=ratio,
        token_reduction_layer=args.layer,
        token_reduction_fuse=not args.no_fuse,
    )
    load_ms = (time.perf_counter() - load_started) * 1e3
    descriptors = []
    schedule = []
    timings = []
    for ordinal, (index, digest, rgb) in enumerate(
        _selected_frames(args.video, args.stride, args.max_frames)
    ):
        schedule.append((index, digest))
        if expected_schedule is not None and schedule[-1] != expected_schedule[ordinal]:
            raise RuntimeError("candidate frame schedule differs from baseline")
        if ordinal == 0:
            for _ in range(args.warmup):
                query.extract_one_tensor(rgb)
            torch.cuda.synchronize()
        started = time.perf_counter()
        descriptor = query.extract_one_tensor(rgb)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1e3)
        descriptors.append(descriptor.detach().cpu().numpy().astype(np.float32, copy=False))
    if expected_schedule is not None and schedule != expected_schedule:
        raise RuntimeError("candidate frame count differs from baseline")
    return {
        "method": method,
        "load_ms": load_ms,
        "timing_ms": _percentiles(timings),
        "descriptors": descriptors,
        "schedule": schedule,
        "last_token_count": int(
            getattr(query.model.backbone, "last_token_count", 529)
        ),
    }


def _topk(refs: np.ndarray, descriptor: np.ndarray, count: int) -> np.ndarray:
    scores = refs @ descriptor
    return np.argsort(-scores, kind="stable")[:count]


def _compare(refs: np.ndarray, baseline: list[np.ndarray], candidate: list[np.ndarray], k: int):
    exact = top1 = baseline_top1_in_candidate = 0
    cosines = []
    for dense, reduced in zip(baseline, candidate):
        dense_order = _topk(refs, dense, k)
        reduced_order = _topk(refs, reduced, k)
        exact += int(np.array_equal(dense_order, reduced_order))
        top1 += int(dense_order[0] == reduced_order[0])
        baseline_top1_in_candidate += int(dense_order[0] in set(reduced_order.tolist()))
        cosines.append(float(np.dot(dense, reduced)))
    total = len(baseline)
    return {
        "frames": total,
        "exact_topk": exact,
        "exact_topk_rate": exact / total,
        "top1_agreement": top1,
        "top1_agreement_rate": top1 / total,
        "baseline_top1_in_candidate_topk": baseline_top1_in_candidate,
        "baseline_top1_in_candidate_topk_rate": baseline_top1_in_candidate / total,
        "descriptor_cosine": _percentiles(cosines),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("l2", "evit"), required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--backend", choices=("pytorch", "pytorch_fp16"), default="pytorch")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-fuse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not 0.0 < args.keep_ratio <= 1.0 or args.layer <= 0:
        raise SystemExit("keep-ratio must be in (0,1] and layer must be positive")
    site = load_site_profile(args.site_profile.expanduser().resolve())
    reloc_map = EDMRelocMap.load(
        site.localization_bundle,
        expected_sha256=site.asset_sha256.localization_bundle,
    )
    baseline = _extract_pass(args, method="none")
    torch.cuda.empty_cache()
    candidate = _extract_pass(
        args,
        method=args.method,
        expected_schedule=baseline["schedule"],
    )
    comparison = _compare(
        reloc_map.ref_global,
        baseline.pop("descriptors"),
        candidate.pop("descriptors"),
        args.topk,
    )
    schedule = baseline.pop("schedule")
    candidate.pop("schedule")
    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    speedup = baseline["timing_ms"]["p50"] / candidate["timing_ms"]["p50"]
    result = {
        "schema": "megaloc-token-reduction-benchmark/v1",
        "paper": "arXiv:2607.15563v1",
        "video": str(args.video.expanduser().resolve()),
        "bundle": str(site.localization_bundle),
        "schedule_sha256": schedule_sha256,
        "config": {
            "method": args.method,
            "keep_ratio": args.keep_ratio,
            "layer": args.layer,
            "fuse": not args.no_fuse,
            "backend": args.backend,
            "topk": args.topk,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "p50_speedup": speedup,
        "no_loss_gate": {
            "passed": comparison["exact_topk_rate"] == 1.0,
            "criterion": f"all top-{args.topk} reference rankings exactly match",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["no_loss_gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

