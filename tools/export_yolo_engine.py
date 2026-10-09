#!/usr/bin/env python3
"""Build a batch-one TensorRT engine matching the live worker's FP16 letterbox.

Run with .venv-yolo/bin/python on the target GPU. No downloads, calibration,
INT8 conversion, aircraft or runtime auto-export. Engines are hardware-specific.
"""

from __future__ import annotations

import argparse
import ast
import math
import tempfile
from pathlib import Path


def _metadata_value(value: str) -> object:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.model.is_file() or args.model.suffix != ".pt":
        parser.error("--model requires an existing .pt checkpoint")
    if args.output.suffix != ".engine" or args.output.exists():
        parser.error("--output requires a new .engine path")
    if min(args.width, args.height, args.imgsz) <= 0:
        parser.error("width, height and imgsz must be positive")

    import onnx
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.export.engine import onnx2engine

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires the target CUDA GPU")
    model = YOLO(str(args.model))
    if model.task not in {"detect", "segment"}:
        raise ValueError("only detection and instance segmentation are supported")
    stride = int(model.model.stride.max())
    size = math.ceil(args.imgsz / stride) * stride
    scale = size / max(args.height, args.width)
    shape = [
        math.ceil(round(dimension * scale) / stride) * stride
        for dimension in (args.height, args.width)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yolo-engine-") as directory:
        # The exporter uses pt_path as the output stem. Never write beside or
        # replace the operator's checkpoint.
        model.model.pt_path = str(Path(directory) / args.model.name)
        exported = model.export(
            format="onnx",
            quantize=16,
            device=0,
            imgsz=shape,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=18,
        )
        graph = onnx.load(exported)
        metadata = {item.key: _metadata_value(item.value) for item in graph.metadata_props}
        # Precision is already explicit in the FP16 ONNX graph. In particular,
        # TensorRT 11 is strongly typed and no longer has BuilderFlag.FP16.
        # Do not re-quantize this graph or calibrate it on unrelated images.
        if graph.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
            raise RuntimeError("export did not produce FP16 input")
        onnx2engine(
            str(exported), str(args.output), workspace=1, metadata=metadata, shape=(1, 3, *shape)
        )
    print(f"Built {args.output}: task={model.task}, FP16, batch=1, shape={shape}")


if __name__ == "__main__":
    main()
