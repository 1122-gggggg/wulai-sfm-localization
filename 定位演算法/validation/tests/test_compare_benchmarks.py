from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


VALIDATION = Path(__file__).resolve().parents[1]


def load_compare_module():
    path = VALIDATION / "compare_benchmarks.py"
    spec = importlib.util.spec_from_file_location("compare_benchmarks_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result(*, x: float = 1.0, inliers: int = 80, stage: str = "nn_fast_accept") -> dict:
    return {"rows": [{
        "idx": 0,
        "frame": "000001.jpg",
        "success": True,
        "mode": "TRACK",
        "next_mode": "TRACK",
        "composite_stage": stage,
        "accepted": True,
        "weak": False,
        "inliers": inliers,
        "corr3d": 120,
        "pnp_failed": False,
        "jump_rejected": False,
        "candidates": [3, 4],
        "used_refs": [3, 4],
        "reproj_rms": 1.25,
        "pose": {"x": x, "y": 2.0, "z": 3.0, "yaw": 0.2},
    }]}


def test_frame_equivalence_accepts_only_small_numeric_noise():
    compare = load_compare_module()
    report = compare.compare_frame_rows(
        result(), result(x=1.0 + 5e-6), pose_atol=1e-5, reproj_atol=1e-5,
    )
    assert report["ok"]
    assert report["mismatch_count"] == 0


def test_frame_equivalence_rejects_accuracy_or_stage_change():
    compare = load_compare_module()
    report = compare.compare_frame_rows(
        result(), result(x=1.01, inliers=79, stage="lg_full_after_nn"),
        pose_atol=1e-5, reproj_atol=1e-5,
    )
    assert not report["ok"]
    assert report["mismatch_count"] == 3
    assert any("inliers" in item for item in report["examples"])
    assert any("composite_stage" in item for item in report["examples"])
    assert any("pose.x" in item for item in report["examples"])
