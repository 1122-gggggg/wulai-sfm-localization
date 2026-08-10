#!/usr/bin/env python3
"""Export and validate a static XFeat LighterGlue matcher.

This is an experiment-only adapter for fabio-sim/LightGlue-ONNX.  It keeps
XFeat's 64 -> 96 descriptor projection, six one-head transformer layers,
isotropic keypoint normalization, and unequal query/reference lengths.  Point
pruning and early stopping are intentionally disabled so ONNX parity is
measurable before any production integration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx import symbolic_helper


FILE = Path(__file__).resolve()
PACKAGE_ROOT = FILE.parents[3]
DEPLOY_DIR = FILE.parents[1] / "deploy_code" / "sfm_glomap_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))


class FourierPositionEncoding(nn.Module):
    def __init__(self, input_dim: int, descriptor_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = descriptor_dim // num_heads
        self.Wr = nn.Linear(input_dim, head_dim // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        embedding = torch.stack((torch.cos(projected), torch.sin(projected)))
        return embedding.repeat_interleave(2, dim=3).repeat(1, 1, 1, self.num_heads).unsqueeze(4)


FUSED_ATTENTION = None
USE_FUSED_ATTENTION = False


def plain_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, query_count, dim = q.shape
    reference_count = k.shape[1]
    head_dim = dim // num_heads
    q = q.reshape(batch, query_count, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, reference_count, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, reference_count, num_heads, head_dim).transpose(1, 2)
    result = F.scaled_dot_product_attention(q, k, v)
    return result.transpose(1, 2).reshape(batch, query_count, dim)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    if USE_FUSED_ATTENTION and FUSED_ATTENTION is not None:
        return FUSED_ATTENTION(q, k, v, num_heads)
    return plain_attention(q, k, v, num_heads)


@symbolic_helper.parse_args("v", "v", "v", "i")
def symbolic_attention(graph, q, k, v, num_heads):
    return graph.op(
        "com.microsoft::MultiHeadAttention", q, k, v, num_heads_i=num_heads
    ).setType(q.type())


def enable_fused_attention() -> None:
    global FUSED_ATTENTION, USE_FUSED_ATTENTION
    USE_FUSED_ATTENTION = True
    FUSED_ATTENTION = torch.library.custom_op(
        "xfeat_onnx::multi_head_attention", mutates_args=()
    )(plain_attention)
    torch.onnx.register_custom_op_symbolic(
        "xfeat_onnx::multi_head_attention", symbolic_attention, 9
    )


class SelfBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def rotate_half(self, qk: torch.Tensor) -> torch.Tensor:
        batch, count, _, _ = qk.shape
        qk = qk.reshape(batch, count, self.num_heads, self.head_dim // 2, 2, 2)
        qk = torch.stack((-qk[..., 1, :], qk[..., 0, :]), dim=4)
        return qk.reshape(batch, count, self.dim, 2)

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        batch, count, _ = x.shape
        qkv = self.Wqkv(x).reshape(batch, count, self.dim, 3)
        qk, value = qkv[..., :2], qkv[..., 2]
        qk = qk * encoding[0] + self.rotate_half(qk) * encoding[1]
        message = self.out_proj(attention(qk[..., 0], qk[..., 1], value, self.num_heads))
        return x + self.ffn(torch.cat((x, message), dim=2))


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.to_qk = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def update(self, descriptors: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        message = self.to_out(message)
        return descriptors + self.ffn(torch.cat((descriptors, message), dim=2))

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qk0, qk1 = self.to_qk(desc0), self.to_qk(desc1)
        value0, value1 = self.to_v(desc0), self.to_v(desc1)
        out0 = self.update(desc0, attention(qk0, qk1, value1, self.num_heads))
        out1 = self.update(desc1, attention(qk1, qk0, value0, self.num_heads))
        return out0, out1


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = SelfBlock(dim, num_heads)
        self.cross_attn = CrossBlock(dim, num_heads)

    def forward(
        self, desc0: torch.Tensor, desc1: torch.Tensor, enc0: torch.Tensor, enc1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cross_attn(self.self_attn(desc0, enc0), self.self_attn(desc1, enc1))


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.25
        self.final_proj = nn.Linear(dim, dim)
        self.matchability = nn.Linear(dim, 1)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        matched0 = self.final_proj(desc0) / self.scale
        matched1 = self.final_proj(desc1) / self.scale
        similarity = matched0 @ matched1.transpose(1, 2)
        certainty0 = self.matchability(desc0)
        certainty1 = self.matchability(desc1).transpose(1, 2)
        return (
            F.log_softmax(similarity, dim=2)
            + F.log_softmax(similarity, dim=1)
            + F.logsigmoid(certainty0)
            + F.logsigmoid(certainty1)
        )


class TokenConfidence(nn.Module):
    """Retained only so every official XFeat checkpoint tensor loads strictly."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())


class StaticXFeatLighterGlue(nn.Module):
    def __init__(self, threshold: float = 0.1, selection: str = "max") -> None:
        super().__init__()
        input_dim, dim, heads, layers = 64, 96, 1, 6
        self.threshold = threshold
        self.selection = selection
        self.input_proj = nn.Linear(input_dim, dim)
        self.posenc = FourierPositionEncoding(2, dim, heads)
        self.transformers = nn.ModuleList([TransformerLayer(dim, heads) for _ in range(layers)])
        self.log_assignment = nn.ModuleList([MatchAssignment(dim) for _ in range(layers)])
        self.token_confidence = nn.ModuleList([TokenConfidence(dim) for _ in range(layers - 1)])

    @staticmethod
    def normalize(keypoints: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        shift = image_size[:, None, :] / 2
        scale = image_size.max(dim=1, keepdim=True).values[:, None, :] / 2
        return (keypoints - shift) / scale

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        image_size0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        desc0 = self.input_proj(descriptors0)
        desc1 = self.input_proj(descriptors1)
        enc0 = self.posenc(self.normalize(keypoints0, image_size0))
        enc1 = self.posenc(self.normalize(keypoints1, image_size1))
        for layer in self.transformers:
            desc0, desc1 = layer(desc0, desc1, enc0, enc1)

        scores = self.log_assignment[-1](desc0, desc1)
        if self.selection == "topk":
            values0, matches0 = torch.topk(scores, k=1, dim=2)
            _, matches1 = torch.topk(scores, k=1, dim=1)
            values0 = values0.squeeze(2)
            matches0 = matches0.squeeze(2)
            matches1 = matches1.squeeze(1)
        else:
            max0, max1 = scores.max(dim=2), scores.max(dim=1)
            values0, matches0, matches1 = max0.values, max0.indices, max1.indices
        query_indices = torch.arange(matches0.shape[1], device=matches0.device)[None]
        mutual = query_indices == matches1.gather(1, matches0)
        match_scores = torch.where(mutual, values0.exp(), values0.new_tensor(0.0))
        valid = mutual & (match_scores > self.threshold)
        matches0 = torch.where(valid, matches0, torch.full_like(matches0, -1)).to(torch.int32)
        return matches0, match_scores


def load_model(weights: Path, selection: str = "max") -> StaticXFeatLighterGlue:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("matcher."): value
        for key, value in checkpoint.items()
        if key.startswith("matcher.")
    }
    model = StaticXFeatLighterGlue(selection=selection).eval()
    model.load_state_dict(state, strict=True)
    return model


class OnnxXFeatMatcher:
    """Experimental drop-in for XFeat.match_lighterglue_indices."""

    def __init__(self, models: dict[int, Path], provider: str, reference_topk: int = 2048) -> None:
        import onnxruntime as ort

        ort.preload_dlls()
        if provider in {"tensorrt", "tensorrt_fp32"}:
            providers = [
                ("TensorrtExecutionProvider", {"trt_fp16_enable": provider == "tensorrt"}),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise ValueError(f"unsupported ONNX matcher provider: {provider}")
        self.sessions = {
            count: ort.InferenceSession(str(path), providers=providers)
            for count, path in models.items()
        }
        self.reference_topk = reference_topk
        self.fallback = None

    @staticmethod
    def _image_size(value, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=device, dtype=torch.float32).reshape(1, 2).contiguous()
        return torch.tensor([value], device=device, dtype=torch.float32)

    def __call__(self, data0: dict, data1: dict, min_conf: float = 0.1) -> np.ndarray:
        query_count = len(data0["keypoints"])
        session = self.sessions.get(query_count)
        if (
            session is None
            or len(data1["keypoints"]) != self.reference_topk
            or abs(float(min_conf) - 0.1) > 1e-9
        ):
            if self.fallback is None:
                raise ValueError(
                    f"no static ONNX matcher for query={query_count}, "
                    f"reference={len(data1['keypoints'])}, min_conf={min_conf}"
                )
            return self.fallback(data0, data1, min_conf=min_conf)

        device = data0["keypoints"].device
        tensors = {
            "keypoints0": data0["keypoints"].reshape(1, query_count, 2).float().contiguous(),
            "descriptors0": data0["descriptors"].reshape(1, query_count, 64).float().contiguous(),
            "image_size0": self._image_size(data0["image_size"], device),
            "keypoints1": data1["keypoints"][:self.reference_topk].reshape(
                1, self.reference_topk, 2
            ).float().contiguous(),
            "descriptors1": data1["descriptors"][:self.reference_topk].reshape(
                1, self.reference_topk, 64
            ).float().contiguous(),
            "image_size1": self._image_size(data1["image_size"], device),
        }
        binding = session.io_binding()
        device_id = device.index or 0
        for name, tensor in tensors.items():
            binding.bind_input(
                name, "cuda", device_id, np.float32, tuple(tensor.shape), tensor.data_ptr()
            )
        binding.bind_output("matches0", "cpu")
        binding.bind_output("scores0", "cpu")
        session.run_with_iobinding(binding)
        matches0 = binding.copy_outputs_to_cpu()[0][0]
        query_indices = np.flatnonzero(matches0 >= 0)
        return np.column_stack((query_indices, matches0[query_indices])).astype(np.int64, copy=False)


def dummy_inputs(query_topk: int, reference_topk: int) -> tuple[torch.Tensor, ...]:
    size = torch.tensor([[1280.0, 720.0]])
    return (
        torch.rand(1, query_topk, 2) * size[:, None],
        torch.randn(1, query_topk, 64),
        size,
        torch.rand(1, reference_topk, 2) * size[:, None],
        torch.randn(1, reference_topk, 64),
        size.clone(),
    )


def export_model(args: argparse.Namespace) -> None:
    if args.fuse_mha:
        if args.exporter != "legacy":
            raise ValueError("fused MHA is an ONNX Runtime-only legacy export")
        enable_fused_attention()
    model = load_model(args.weights, args.selection)
    inputs = dummy_inputs(args.query_topk, args.reference_topk)
    names = [
        "keypoints0", "descriptors0", "image_size0",
        "keypoints1", "descriptors1", "image_size1",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(args.output),
        input_names=names,
        output_names=["matches0", "scores0"],
        opset_version=18,
        dynamo=args.exporter == "dynamo",
    )
    import onnx

    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "nodes": len(graph.graph.node),
        "exporter": args.exporter,
        "selection": args.selection,
        "fused_mha": args.fuse_mha,
        "query_topk": args.query_topk,
        "reference_topk": args.reference_topk,
    }, indent=2))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), q))


def _select_covisible_pairs(
    reloc_map,
    query_topk: int,
    reference_topk: int,
    limit: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for query_index, name in enumerate(reloc_map.ref_names):
        for reference_index in (reloc_map.covis or {}).get(name, []):
            query = reloc_map.refs[name].feats
            reference = reloc_map.refs[reloc_map.ref_names[reference_index]].feats
            if len(query["keypoints"]) >= query_topk and len(reference["keypoints"]) >= reference_topk:
                pairs.append((query_index, reference_index))
                break
        if len(pairs) >= limit:
            break
    return pairs


def _run_engine(
    context,
    stream: int,
    feed: dict[str, torch.Tensor],
    matches_output: torch.Tensor,
    scores_output: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    for name, tensor in feed.items():
        context.set_tensor_address(name, tensor.data_ptr())
    if not context.execute_async_v3(stream_handle=stream):
        raise RuntimeError("TensorRT execute_async_v3 failed")
    torch.cuda.synchronize()
    return matches_output.cpu().numpy().copy(), scores_output.cpu().numpy().copy()


def quantize_model(args: argparse.Namespace) -> None:
    import modelopt.onnx.quantization as moq
    from reloc_localizer_xfeat import XFeatRelocMap

    reloc_map = XFeatRelocMap.load(str(args.bundle), expected_sha256=args.bundle_sha256)
    samples: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "keypoints0", "descriptors0", "image_size0",
            "keypoints1", "descriptors1", "image_size1",
        )
    }
    for query_index, name in enumerate(reloc_map.ref_names):
        neighbors = (reloc_map.covis or {}).get(name, [])
        if not neighbors:
            continue
        reference_index = neighbors[0]
        query = reloc_map.refs[name].feats
        reference = reloc_map.refs[reloc_map.ref_names[reference_index]].feats
        if len(query["keypoints"]) < args.query_topk or len(reference["keypoints"]) < args.reference_topk:
            continue
        values = {
            "keypoints0": query["keypoints"][:args.query_topk].numpy()[None],
            "descriptors0": query["descriptors"][:args.query_topk].float().numpy()[None],
            "image_size0": np.asarray(query["image_size"], np.float32)[None],
            "keypoints1": reference["keypoints"][:args.reference_topk].numpy()[None],
            "descriptors1": reference["descriptors"][:args.reference_topk].float().numpy()[None],
            "image_size1": np.asarray(reference["image_size"], np.float32)[None],
        }
        for key, value in values.items():
            samples[key].append(value.astype(np.float32, copy=False))
        if len(samples["keypoints0"]) >= args.pairs:
            break
    if len(samples["keypoints0"]) < args.pairs:
        raise RuntimeError(f"only found {len(samples['keypoints0'])} calibration pairs")

    calibration_data = {key: np.concatenate(values, axis=0) for key, values in samples.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    moq.quantize(
        onnx_path=str(args.model),
        quantize_mode=args.mode,
        calibration_data=calibration_data,
        calibration_method="rtn_dq" if args.mode == "int4" else args.calibration_method,
        calibration_eps=["cuda:0", "cpu"],
        high_precision_dtype="fp32" if args.mode == "fp8" else "fp16",
        dq_only=True,
        simplify=False,
        output_path=str(args.output),
    )
    print(json.dumps({
        "model": str(args.model),
        "output": str(args.output),
        "mode": args.mode,
        "calibration_pairs": args.pairs,
        "bytes": args.output.stat().st_size,
    }, indent=2))


def benchmark_onnx(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    from reloc_localizer_xfeat import XFeatRelocMap

    ort.preload_dlls()
    providers = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        "tensorrt_fp32": [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": False}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    }[args.provider]
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_dtypes = {
        node.name: np.float16 if node.type == "tensor(float16)" else np.float32
        for node in session.get_inputs()
    }
    reloc_map = XFeatRelocMap.load(str(args.bundle), expected_sha256=args.bundle_sha256)

    pair_indices = _select_covisible_pairs(
        reloc_map, args.query_topk, args.reference_topk, args.pairs
    )
    if len(pair_indices) < args.pairs:
        raise RuntimeError(f"only found {len(pair_indices)} usable covisible pairs")

    feeds = []
    for query_index, reference_index in pair_indices:
        query = reloc_map.refs[reloc_map.ref_names[query_index]].feats
        reference = reloc_map.refs[reloc_map.ref_names[reference_index]].feats
        feed = {
            "keypoints0": query["keypoints"][:args.query_topk].numpy()[None],
            "descriptors0": query["descriptors"][:args.query_topk].float().numpy()[None],
            "image_size0": np.asarray(query["image_size"], np.float32)[None],
            "keypoints1": reference["keypoints"][:args.reference_topk].numpy()[None],
            "descriptors1": reference["descriptors"][:args.reference_topk].float().numpy()[None],
            "image_size1": np.asarray(reference["image_size"], np.float32)[None],
        }
        feeds.append({name: value.astype(input_dtypes[name], copy=False) for name, value in feed.items()})

    for _ in range(args.warmup):
        session.run(None, feeds[0])
    latencies, valid_matches = [], []
    outputs = []
    for feed in feeds:
        start = time.perf_counter()
        output = session.run(None, feed)
        latencies.append((time.perf_counter() - start) * 1000)
        valid_matches.append(int((output[0] >= 0).sum()))
        outputs.append(output)

    parity = None
    if args.parity:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        baseline = load_model(args.weights, args.selection).to(device)
        exact, jaccards, score_diffs = 0, [], []
        with torch.inference_mode():
            for feed, output in zip(feeds, outputs):
                tensors = tuple(
                    torch.from_numpy(feed[node.name]).float().to(device) for node in session.get_inputs()
                )
                expected_matches, expected_scores = baseline(*tensors)
                expected_matches = expected_matches.cpu().numpy()
                expected_scores = expected_scores.cpu().numpy()
                exact += int(np.array_equal(expected_matches, output[0]))
                expected_set = set(np.flatnonzero(expected_matches[0] >= 0).tolist())
                actual_set = set(np.flatnonzero(output[0][0] >= 0).tolist())
                jaccards.append(len(expected_set & actual_set) / max(1, len(expected_set | actual_set)))
                score_diffs.append(float(np.max(np.abs(expected_scores - output[1]))))
        parity = {
            "exact_match_pairs": exact,
            "median_query_match_jaccard": percentile(jaccards, 50),
            "minimum_query_match_jaccard": min(jaccards),
            "maximum_score_abs_diff": max(score_diffs),
        }

    result = {
        "model": str(args.model),
        "provider_requested": args.provider,
        "providers_active": session.get_providers(),
        "query_topk": args.query_topk,
        "reference_topk": args.reference_topk,
        "pairs": len(feeds),
        "latency_ms": {
            "median": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "mean": float(np.mean(latencies)),
        },
        "valid_matches": {
            "median": percentile(valid_matches, 50),
            "minimum": min(valid_matches),
        },
        "parity": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def benchmark_engine(args: argparse.Namespace) -> None:
    import tensorrt as trt
    from reloc_localizer_xfeat import XFeatRelocMap

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {args.engine}")
    context = engine.create_execution_context()
    reloc_map = XFeatRelocMap.load(str(args.bundle), expected_sha256=args.bundle_sha256)

    pairs = _select_covisible_pairs(
        reloc_map, args.query_topk, args.reference_topk, args.pairs
    )
    if len(pairs) < args.pairs:
        raise RuntimeError(f"only found {len(pairs)} usable covisible pairs")

    device = torch.device("cuda")
    prepared = []
    for query_index, reference_index in pairs:
        query = reloc_map.refs[reloc_map.ref_names[query_index]].feats
        reference = reloc_map.refs[reloc_map.ref_names[reference_index]].feats
        prepared.append({
            "keypoints0": query["keypoints"][:args.query_topk].to(device)[None].contiguous(),
            "descriptors0": query["descriptors"][:args.query_topk].float().to(device)[None].contiguous(),
            "image_size0": torch.tensor([query["image_size"]], dtype=torch.float32, device=device),
            "keypoints1": reference["keypoints"][:args.reference_topk].to(device)[None].contiguous(),
            "descriptors1": reference["descriptors"][:args.reference_topk].float().to(device)[None].contiguous(),
            "image_size1": torch.tensor([reference["image_size"]], dtype=torch.float32, device=device),
        })

    matches_output = torch.empty((1, args.query_topk), dtype=torch.int32, device=device)
    scores_output = torch.empty((1, args.query_topk), dtype=torch.float32, device=device)
    context.set_tensor_address("matches0", matches_output.data_ptr())
    context.set_tensor_address("scores0", scores_output.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    for _ in range(args.warmup):
        _run_engine(context, stream, prepared[0], matches_output, scores_output)
    baseline = load_model(args.weights, "max").to(device)
    latencies, jaccards, score_diffs, valid_matches = [], [], [], []
    with torch.inference_mode():
        for feed in prepared:
            start = time.perf_counter()
            actual_matches, actual_scores = _run_engine(
                context, stream, feed, matches_output, scores_output
            )
            latencies.append((time.perf_counter() - start) * 1000)
            expected_matches, expected_scores = baseline(
                feed["keypoints0"], feed["descriptors0"], feed["image_size0"],
                feed["keypoints1"], feed["descriptors1"], feed["image_size1"],
            )
            expected_matches = expected_matches.cpu().numpy()
            expected_scores = expected_scores.cpu().numpy()
            expected_set = set(np.flatnonzero(expected_matches[0] >= 0).tolist())
            actual_set = set(np.flatnonzero(actual_matches[0] >= 0).tolist())
            jaccards.append(len(expected_set & actual_set) / max(1, len(expected_set | actual_set)))
            score_diffs.append(float(np.max(np.abs(expected_scores - actual_scores))))
            valid_matches.append(len(actual_set))

    result = {
        "engine": str(args.engine),
        "query_topk": args.query_topk,
        "reference_topk": args.reference_topk,
        "pairs": len(prepared),
        "latency_ms": {
            "median": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "mean": float(np.mean(latencies)),
        },
        "valid_matches": {"median": percentile(valid_matches, 50), "minimum": min(valid_matches)},
        "parity": {
            "median_query_match_jaccard": percentile(jaccards, 50),
            "minimum_query_match_jaccard": min(jaccards),
            "maximum_score_abs_diff": max(score_diffs),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    default_weights = PACKAGE_ROOT / "torch_hub_cache/verlab_accelerated_features_main/weights/xfeat-lighterglue.pt"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--weights", type=Path, default=default_weights)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--query-topk", type=int, required=True)
    export_parser.add_argument("--reference-topk", type=int, required=True)
    export_parser.add_argument("--exporter", choices=("legacy", "dynamo"), default="legacy")
    export_parser.add_argument("--selection", choices=("max", "topk"), default="max")
    export_parser.add_argument("--fuse-mha", action="store_true")
    export_parser.set_defaults(run=export_model)

    quantize_parser = subparsers.add_parser("quantize")
    quantize_parser.add_argument("--model", type=Path, required=True)
    quantize_parser.add_argument("--bundle", type=Path, required=True)
    quantize_parser.add_argument("--bundle-sha256")
    quantize_parser.add_argument("--output", type=Path, required=True)
    quantize_parser.add_argument("--mode", choices=("fp8", "int8", "int4"), required=True)
    quantize_parser.add_argument("--calibration-method", choices=("entropy", "max"), default="max")
    quantize_parser.add_argument("--query-topk", type=int, required=True)
    quantize_parser.add_argument("--reference-topk", type=int, required=True)
    quantize_parser.add_argument("--pairs", type=int, default=8)
    quantize_parser.set_defaults(run=quantize_model)

    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("--model", type=Path, required=True)
    bench_parser.add_argument("--weights", type=Path, default=default_weights)
    bench_parser.add_argument("--bundle", type=Path, required=True)
    bench_parser.add_argument("--bundle-sha256")
    bench_parser.add_argument("--output", type=Path, required=True)
    bench_parser.add_argument(
        "--provider", choices=("cpu", "cuda", "tensorrt", "tensorrt_fp32"), required=True
    )
    bench_parser.add_argument("--query-topk", type=int, required=True)
    bench_parser.add_argument("--reference-topk", type=int, required=True)
    bench_parser.add_argument("--pairs", type=int, default=24)
    bench_parser.add_argument("--warmup", type=int, default=5)
    bench_parser.add_argument("--parity", action="store_true")
    bench_parser.add_argument("--selection", choices=("max", "topk"), default="max")
    bench_parser.set_defaults(run=benchmark_onnx)

    engine_parser = subparsers.add_parser("engine-benchmark")
    engine_parser.add_argument("--engine", type=Path, required=True)
    engine_parser.add_argument("--weights", type=Path, default=default_weights)
    engine_parser.add_argument("--bundle", type=Path, required=True)
    engine_parser.add_argument("--bundle-sha256")
    engine_parser.add_argument("--output", type=Path, required=True)
    engine_parser.add_argument("--query-topk", type=int, required=True)
    engine_parser.add_argument("--reference-topk", type=int, required=True)
    engine_parser.add_argument("--pairs", type=int, default=24)
    engine_parser.add_argument("--warmup", type=int, default=5)
    engine_parser.set_defaults(run=benchmark_engine)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.run(parsed)
