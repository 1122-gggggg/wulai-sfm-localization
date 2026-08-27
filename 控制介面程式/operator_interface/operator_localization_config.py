"""Shared localization display policy with no operator-app dependency."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


def positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


LIVE_STATUS_PATH = Path("/tmp/sfm_flight_operator_live_status.json")
POSE_JUMP_U = positive_env_float("SFM_MAX_POSE_JUMP_U", 1.5)
LOCALIZATION_BENCHMARK_LABELS = {
    "auto": "AUTO 狀態機",
    "global": "BOOT_INIT / LOST",
    "weak": "WEAK_TRACK",
    "track": "TRACK 正常路徑",
}


def normalize_camera_forward(value: object) -> np.ndarray | None:
    """Return a finite unit camera optical-axis vector, or no orientation."""
    try:
        forward = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if forward.shape != (3,) or not np.isfinite(forward).all():
        return None
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return None
    return forward / norm


def normalize_camera_axes(value: object) -> np.ndarray | None:
    """Validate right/down/forward camera axes expressed in world coordinates."""
    try:
        axes = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        return None
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms < 1e-6):
        return None
    axes = axes / norms[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=0.05):
        return None
    return axes
