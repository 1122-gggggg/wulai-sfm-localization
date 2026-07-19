"""Load benchmark camera intrinsics from the shipped JSON artifact."""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_scaled_simple_radial(path: str | Path, size: tuple[int, int]) -> list[float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("model") != "SIMPLE_RADIAL":
        raise ValueError(f"unsupported camera model in {path}: {data.get('model')!r}")
    cameras = data.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"no cameras in {path}")

    width, height = map(int, size)
    key = f"{width}x{height}"
    recommended = data.get("query_recommended", {}).get(key, {})
    selected = cameras.get(recommended.get("use_scene"))
    if selected is None:
        candidates = [camera for camera in cameras.values()
                      if int(camera.get("width", 0)) > 0 and int(camera.get("height", 0)) > 0]
        if not candidates:
            raise ValueError(f"no usable cameras in {path}")
        target_aspect = width / height
        selected = min(
            candidates,
            key=lambda camera: (
                abs(int(camera["width"]) / int(camera["height"]) - target_aspect),
                abs(int(camera["width"]) - width) + abs(int(camera["height"]) - height),
            ),
        )

    source_width = int(selected.get("width", 0))
    source_height = int(selected.get("height", 0))
    params = selected.get("params", [])
    if source_width <= 0 or source_height <= 0 or len(params) != 4:
        raise ValueError(f"invalid SIMPLE_RADIAL entry in {path}")
    values = [float(value) for value in params]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite camera parameters in {path}")
    scale_x = width / source_width
    scale_y = height / source_height
    if not math.isclose(scale_x, scale_y, rel_tol=1e-3, abs_tol=1e-6):
        raise ValueError(
            f"cannot scale intrinsics across aspect ratios: {source_width}x{source_height} -> {width}x{height}"
        )
    return [values[0] * scale_x, values[1] * scale_x, values[2] * scale_y, values[3]]
