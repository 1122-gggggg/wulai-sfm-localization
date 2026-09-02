#!/usr/bin/env python3
"""Build and validate EDM directly with the installed native TensorRT runtime."""
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
DEPLOY = ROOT / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_edm_onnx_identity import compare_one, textured_pair  # noqa: E402
from edm_matcher import EDM_H, EDM_W, EDMMatcher  # noqa: E402
from edm_onnx_matcher import DEFAULT_ONNX, _postprocess_deploy_output  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_engine(onnx_path: Path, engine_path: Path, *, fp16: bool) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    flags = 0 if explicit is None else 1 << int(explicit)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 * 1024**3)
    config.builder_optimization_level = 3
    if fp16:
        if not hasattr(trt.BuilderFlag, "FP16"):
            raise RuntimeError(
                "TensorRT 11 requires an explicitly typed FP16 ONNX graph; "
                "the current EDM flight ONNX graph is FP32"
            )
        config.set_flag(trt.BuilderFlag.FP16)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_ms = (time.perf_counter() - started) * 1e3
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the EDM engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return {
        "build_ms": build_ms,
        "engine_bytes": engine_path.stat().st_size,
        "engine_sha256": _sha256(engine_path),
    }


class NativeTensorRTMatcher:
    def __init__(self, engine_path: Path, *, mconf_thr: float = 0.2):
        import tensorrt as trt

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError("TensorRT failed to deserialize the EDM engine")
        self._context = self._engine.create_execution_context()
        names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        ]
        inputs = [
            name
            for name in names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        outputs = [name for name in names if name not in inputs]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"unexpected EDM TensorRT tensors: {names}")
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.input_shape = tuple(self._engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self._engine.get_tensor_shape(self.output_name))
        if self.input_shape != (1, 2, EDM_H, EDM_W):
            raise ValueError(f"unexpected EDM TensorRT input shape: {self.input_shape}")
        self.mconf_thr = float(mconf_thr)

    def match(self, ref: np.ndarray, query: np.ndarray) -> dict:
        ref = EDMMatcher.load_gray(ref)
        query = EDMMatcher.load_gray(query)
        array = np.stack((ref, query), axis=0)[None].astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(array).cuda()
        output_tensor = torch.empty(
            self.output_shape,
            dtype=torch.float32,
            device="cuda",
        )
        self._context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self._context.set_tensor_address(self.output_name, output_tensor.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not self._context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT EDM inference failed")
        output = output_tensor.cpu().numpy()
        return _postprocess_deploy_output(
            output,
            width=EDM_W,
            height=EDM_H,
            mconf_thr=self.mconf_thr,
        )


def _timed(matcher, ref, query, repeats: int) -> tuple[dict, dict]:
    result = matcher.match(ref, query)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = matcher.match(ref, query)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1e3)
    data = np.asarray(values, dtype=float)
    return result, {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "mean": float(np.mean(data)),
    }


def _river_pair(bundle: Path, bundle_sha256: str, video: Path):
    from reloc_localizer_edm import EDMRelocMap

    reloc_map = EDMRelocMap.load(bundle, expected_sha256=bundle_sha256)
    ref_name = reloc_map.ref_names[0]
    capture = cv2.VideoCapture(str(video))
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read {video}")
    return reloc_map.images[ref_name], EDMMatcher.load_gray(bgr), ref_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import tensorrt as trt

    build = None
    if args.rebuild or not args.engine.is_file():
        build = build_engine(args.onnx, args.engine, fp16=args.fp16)
    torch_matcher = EDMMatcher(fp16=True)
    trt_matcher = NativeTensorRTMatcher(args.engine)
    synthetic = textured_pair()
    river_ref, river_query, ref_name = _river_pair(
        args.bundle, args.bundle_sha256, args.video
    )
    pairs = (
        ("synthetic_shift", synthetic[0], synthetic[1]),
        (f"river:{ref_name}", river_ref, river_query),
    )
    rows = []
    identity_ok = True
    for name, ref, query in pairs:
        torch_result, torch_ms = _timed(torch_matcher, ref, query, args.repeats)
        trt_result, trt_ms = _timed(trt_matcher, ref, query, args.repeats)
        comparison = compare_one(torch_result, trt_result)
        identity_ok &= comparison["identity_ok"]
        rows.append(
            {
                "name": name,
                "comparison": comparison,
                "torch_ms": torch_ms,
                "tensorrt_ms": trt_ms,
                "p50_speedup": torch_ms["p50"] / trt_ms["p50"],
            }
        )
    result = {
        "schema": "edm-native-tensorrt-feasibility/v1",
        "tensorrt_version": trt.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "onnx": str(args.onnx),
        "onnx_sha256": _sha256(args.onnx),
        "engine": str(args.engine),
        "engine_sha256": _sha256(args.engine),
        "fp16": bool(args.fp16),
        "build": build,
        "input_shape": list(trt_matcher.input_shape),
        "output_shape": list(trt_matcher.output_shape),
        "pairs": rows,
        "identity_ok": bool(identity_ok),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if identity_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
