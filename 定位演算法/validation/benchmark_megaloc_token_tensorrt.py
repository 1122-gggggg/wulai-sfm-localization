#!/usr/bin/env python3
"""End-to-end A/B for full-token and token-reduced MegaLoc TensorRT engines."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

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

from benchmark_boq_resnet50_first_round import _edm_pose, _metric, _queries  # noqa: E402
from edm_matcher import EDM_H, EDM_W, EDMMatcher  # noqa: E402
from reloc_localizer_edm import (  # noqa: E402
    EDM_GLOBAL_DESCRIPTOR_DIM,
    MEGALOC_INPUT,
    EDMRelocMap,
    MegaLocQuery,
    _resolve_cuda_device,
    _validate_megaloc_engine_file,
    _validate_megaloc_tensorrt_compatibility,
)
from site_profile import load_site_profile  # noqa: E402


DEFAULT_SITE = WORKSPACE / "地圖檔" / "場域" / "river_site" / "site_profile.json"


class ExperimentalTensorRTEngine:
    """Strict fixed-shape runner without the production engine size/hash pin."""

    def __init__(self, engine_path: Path, device: str = "cuda"):
        import tensorrt as trt

        self.device = _resolve_cuda_device(device)
        _validate_megaloc_engine_file(engine_path)
        _validate_megaloc_tensorrt_compatibility(self.device.index, trt)
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        with engine_path.open("rb") as stream:
            self._engine = self._runtime.deserialize_cuda_engine(stream.read())
        if self._engine is None:
            raise RuntimeError("failed to deserialize token-reduced MegaLoc engine")
        expected = {
            "images": (trt.TensorIOMode.INPUT, (1, 3, MEGALOC_INPUT, MEGALOC_INPUT)),
            "descriptors": (
                trt.TensorIOMode.OUTPUT,
                (1, EDM_GLOBAL_DESCRIPTOR_DIM),
            ),
        }
        names = {
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        }
        if names != set(expected):
            raise ValueError(f"unexpected token-reduced engine tensors: {names}")
        for name, (mode, shape) in expected.items():
            if (
                self._engine.get_tensor_mode(name) != mode
                or tuple(self._engine.get_tensor_shape(name)) != shape
                or self._engine.get_tensor_dtype(name) != trt.float32
            ):
                raise ValueError(f"unexpected token-reduced tensor contract: {name}")
        self._context = self._engine.create_execution_context()
        self._output = torch.empty(
            (1, EDM_GLOBAL_DESCRIPTOR_DIM), dtype=torch.float32, device=self.device
        )
        self._context.set_tensor_address("descriptors", self._output.data_ptr())

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = images.contiguous()
        self._context.set_tensor_address("images", images.data_ptr())
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not self._context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("token-reduced MegaLoc TensorRT inference failed")
        return self._output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_names(scores: torch.Tensor, names: list[str], count: int) -> list[str]:
    indices = torch.topk(scores, count).indices.cpu().tolist()
    return [names[index] for index in indices]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-profile", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("l2", "evit"), default="l2")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=75)
    parser.add_argument("--references", type=int, default=5)
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
    full = MegaLocQuery(backend="tensorrt")
    reduced = ExperimentalTensorRTEngine(args.candidate_engine)
    descriptor_bank = torch.from_numpy(reloc_map.ref_global).cuda()
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
    vpr_times = {"full": [], "reduced": []}
    edm_times = {"full": [], "reduced": []}
    accepted = {"full": 0, "reduced": 0}
    top1_agreement = 0
    full_top1_in_reduced = 0
    descriptor_cosines = []
    for index, rgb, gray in _queries(args):
        torch.cuda.synchronize()
        started = time.perf_counter()
        full_descriptor = full.extract_one_tensor(rgb)
        full_refs = _top_names(
            descriptor_bank @ full_descriptor,
            reloc_map.ref_names,
            args.references,
        )
        torch.cuda.synchronize()
        vpr_times["full"].append((time.perf_counter() - started) * 1e3)

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            reduced_descriptor = F.normalize(
                reduced(full._preprocess(rgb)).float(), dim=1
            )[0]
        reduced_refs = _top_names(
            descriptor_bank @ reduced_descriptor,
            reloc_map.ref_names,
            args.references,
        )
        torch.cuda.synchronize()
        vpr_times["reduced"].append((time.perf_counter() - started) * 1e3)
        top1_agreement += int(full_refs[0] == reduced_refs[0])
        full_top1_in_reduced += int(full_refs[0] in set(reduced_refs))
        descriptor_cosines.append(float(torch.dot(full_descriptor, reduced_descriptor)))

        poses = {}
        for mode, refs in (("full", full_refs), ("reduced", reduced_refs)):
            pose, edm_ms = _edm_pose(
                matcher, reloc_map, gray, refs, camera, camera_scale
            )
            poses[mode] = pose
            edm_times[mode].append(edm_ms)
            accepted[mode] += int(pose["accepted"])
        rows.append(
            {
                "source_index": index,
                "refs": {"full": full_refs, "reduced": reduced_refs},
                "accepted": {key: value["accepted"] for key, value in poses.items()},
                "inliers": {key: value["inliers"] for key, value in poses.items()},
            }
        )
    result = {
        "schema": "megaloc-token-tensorrt-benchmark/v1",
        "candidate": {
            "engine": str(args.candidate_engine.expanduser().resolve()),
            "engine_sha256": _sha256(args.candidate_engine),
            "method": args.method,
            "keep_ratio": args.keep_ratio,
            "layer": args.layer,
        },
        "frames": len(rows),
        "references": args.references,
        "accepted": accepted,
        "vpr_ms": {key: _metric(value) for key, value in vpr_times.items()},
        "edm_ms": {key: _metric(value) for key, value in edm_times.items()},
        "top1_agreement": top1_agreement,
        "full_top1_in_reduced_topk": full_top1_in_reduced,
        "descriptor_cosine": _metric(descriptor_cosines),
        "total_p50_ms": {
            key: _metric(vpr_times[key])["p50"] + _metric(edm_times[key])["p50"]
            for key in ("full", "reduced")
        },
        "no_loss": accepted["reduced"] >= accepted["full"],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if result["no_loss"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
