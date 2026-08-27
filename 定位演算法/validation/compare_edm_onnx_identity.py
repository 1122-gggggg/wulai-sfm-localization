#!/usr/bin/env python3
"""Compare EDM ONNX/TensorRT matches against production PyTorch FP16.

Protocol matches test_fused_coarse_matching: match-set equality on rounded
keypoints, sorted mconf spectrum, and direction-01 coarse-cell argmax.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

DEPLOY = Path(__file__).resolve().parents[1] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from edm_matcher import EDM_H, EDM_W, EDMMatcher  # noqa: E402
from edm_onnx_matcher import DEFAULT_ONNX, make_matcher  # noqa: E402


def textured_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260803)
    base = rng.integers(0, 255, (EDM_H, EDM_W), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.7)
    query = np.roll(np.roll(base, 8, axis=1), 4, axis=0)
    return base, query


def river_pair(bundle: Path, video: Path | None, bundle_sha256: str) -> tuple[np.ndarray, np.ndarray, str]:
    from reloc_localizer_edm import EDMRelocMap

    reloc = EDMRelocMap.load(bundle, expected_sha256=bundle_sha256)
    ref_name = reloc.ref_names[0]
    ref = reloc.images[ref_name]
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read first frame: {video}")
        query = EDMMatcher.load_gray(bgr)
        label = f"bundle:{ref_name} vs {video.name}[0]"
    else:
        query = np.roll(ref, 6, axis=1)
        label = f"bundle:{ref_name} rolled"
    return ref, query, label


def point_set(points: np.ndarray) -> set[tuple[float, float]]:
    return set(map(tuple, np.round(np.asarray(points, np.float64), 4)))


def direction01_cells(result: dict) -> dict[int, tuple[float, float]]:
    k0 = np.asarray(result["mkpts0"], np.float64)
    k1 = np.asarray(result["mkpts1"], np.float64)
    if len(k0) == 0:
        return {}
    keep = ~EDMMatcher.is_refined(k0)
    if not keep.any():
        return {}
    cells = EDMMatcher.cell_ids(k0[keep])
    return {int(cell): (float(pt[0]), float(pt[1])) for cell, pt in zip(cells, k1[keep])}


def compare_one(baseline: dict, candidate: dict) -> dict:
    base0 = point_set(baseline["mkpts0"])
    cand0 = point_set(candidate["mkpts0"])
    base1 = point_set(baseline["mkpts1"])
    cand1 = point_set(candidate["mkpts1"])
    base_cells = direction01_cells(baseline)
    cand_cells = direction01_cells(candidate)
    shared = set(base_cells) & set(cand_cells)
    cell_deltas = [
        float(np.hypot(base_cells[c][0] - cand_cells[c][0], base_cells[c][1] - cand_cells[c][1]))
        for c in shared
    ]
    base_conf = np.sort(np.asarray(baseline["mconf"], np.float64))
    cand_conf = np.sort(np.asarray(candidate["mconf"], np.float64))
    conf_len = min(len(base_conf), len(cand_conf))
    conf_max_diff = (
        float(np.abs(base_conf[:conf_len] - cand_conf[:conf_len]).max())
        if conf_len
        else None
    )
    set_equal = base0 == cand0 and base1 == cand1
    cells_equal = set(base_cells) == set(cand_cells)
    conf_close = (
        len(base_conf) == len(cand_conf)
        and conf_max_diff is not None
        and conf_max_diff <= 1e-5
    )
    return {
        "baseline_matches": int(len(baseline["mconf"])),
        "candidate_matches": int(len(candidate["mconf"])),
        "only_baseline_mkpts0": int(len(base0 - cand0)),
        "only_candidate_mkpts0": int(len(cand0 - base0)),
        "only_baseline_cells": int(len(set(base_cells) - set(cand_cells))),
        "only_candidate_cells": int(len(set(cand_cells) - set(base_cells))),
        "shared_cells": int(len(shared)),
        "cell_query_mean_px": None if not cell_deltas else float(np.mean(cell_deltas)),
        "cell_query_max_px": None if not cell_deltas else float(np.max(cell_deltas)),
        "mconf_len_equal": len(base_conf) == len(cand_conf),
        "mconf_max_abs_diff": conf_max_diff,
        "match_set_equal": set_equal,
        "cell_argmax_equal": cells_equal,
        "mconf_spectrum_close": conf_close,
        "identity_ok": bool(set_equal and cells_equal and conf_close),
    }


def time_match(matcher, refs: list[np.ndarray], query: np.ndarray, repeats: int) -> float:
    matcher.match_many_to_one(refs, query)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        matcher.match_many_to_one(refs, query)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repeats


def run_backend(name: str, refs: list[np.ndarray], query: np.ndarray) -> tuple[object, list[dict]]:
    matcher = make_matcher(name, mconf_thr=0.2)
    return matcher, matcher.match_many_to_one(refs, query)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "/home/allen/localization/地圖檔/場域/river_site/releases/"
            "river_site_official69_map_v000_20260811/localization/localization_bundle.pt"
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("/home/allen/localization/模擬器/測試影片/P1670167_720p.MP4"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/allen/localization/outputs/validation/edm_onnx_identity.json"),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--bundle-sha256",
        default="a1ac3c8effce34fdb9891eaaa16e7c91e7b3f0bca68f0903f634f074f2bf6bae",
    )
    args = parser.parse_args()

    pairs = [("synthetic_shift", *textured_pair()[:2], "synthetic")]
    if args.bundle.is_file():
        ref, query, label = river_pair(
            args.bundle,
            args.video if args.video.is_file() else None,
            args.bundle_sha256,
        )
        pairs.append(("river_production_like", ref, query, label))

    backends = ["torch"]
    available = []
    try:
        import onnxruntime as ort  # noqa: F401
        available = list(ort.get_available_providers())
    except Exception as exc:
        report = {"error": f"onnxruntime unavailable: {exc!r}", "identity_ok": False}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2
    if "CUDAExecutionProvider" in available:
        backends.append("onnx_cuda")
    if "TensorrtExecutionProvider" in available:
        backends.append("onnx_tensorrt")

    report = {
        "onnx_path": str(DEFAULT_ONNX),
        "onnx_exists": DEFAULT_ONNX.is_file(),
        "ort_providers": available,
        "backends": backends,
        "pairs": [],
        "identity_ok": True,
    }
    if not DEFAULT_ONNX.is_file():
        report["identity_ok"] = False
        report["error"] = f"missing {DEFAULT_ONNX}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    torch_matcher = None
    for pair_name, ref, query, label in pairs:
        pair_row = {"name": pair_name, "label": label, "comparisons": {}, "timings_ms": {}}
        torch_matcher, torch_out = run_backend("torch", [ref], query)
        pair_row["timings_ms"]["torch"] = time_match(torch_matcher, [ref], query, args.repeats)
        for backend in backends:
            if backend == "torch":
                continue
            try:
                cand_matcher, cand_out = run_backend(backend, [ref], query)
                comparison = compare_one(torch_out[0], cand_out[0])
                pair_row["comparisons"][backend] = comparison
                pair_row["timings_ms"][backend] = time_match(
                    cand_matcher, [ref], query, args.repeats
                )
                if not comparison["identity_ok"]:
                    report["identity_ok"] = False
            except Exception as exc:
                pair_row["comparisons"][backend] = {"error": repr(exc), "identity_ok": False}
                report["identity_ok"] = False
        report["pairs"].append(pair_row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["identity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
