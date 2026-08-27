from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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


def _receipt(**overrides):
    receipt = {
        "radius": 0.5881852149963379,
        "mconf_thr": 0.2,
        "coarse_topk": 3225,
        "cache_capacity": 32,
        "lost_strategy": "boot_and_lost_once",
        "lost_global_retrieval_interval": 15,
        "fused_coarse_mode": "fused",
        "runtime_sigma_mode": "reference_grid",
        "temporal_feature_cache_size": 0,
        "acquire_stage_mode": "full_set",
        "lost_prior_strategy": "restrict_nearby",
        "lost_prior_fusion_weight": 1.0,
        "worker_mode": "sequential",
    }
    receipt.update(overrides)
    return receipt



def _series_result(values, receipt=None):
    payload = {
        "rows": [
            {"idx": i, "wall_ms": v, "match_ms": v * 0.8, "pnp_ms": 3.03}
            for i, v in enumerate(values)
        ],
    }
    if receipt is not None:
        payload["receipt"] = receipt
    return payload


def test_contiguous_block_bootstrap_reports_ci_without_pass_threshold():
    compare = load_compare_module()
    values = [28.27] * 20 + [105.51] * 5
    report = compare.contiguous_block_bootstrap(
        values, block_size=5, n_resamples=50, seed=0,
    )
    assert report["n"] == 25
    assert report["mean"] == pytest.approx(sum(values) / len(values))
    assert report["ci_low"] <= report["mean"] <= report["ci_high"]
    assert "pass" not in report
    assert report.get("verdict") is None


def test_baab_reports_drift_and_effect_without_inventing_a_gate():
    compare = load_compare_module()
    receipt = _receipt()
    b1 = _series_result([28.27] * 8, receipt)
    a1 = _series_result([24.01] * 8, receipt)
    a2 = _series_result([24.50] * 8, receipt)
    b2 = _series_result([28.40] * 8, receipt)
    report = compare.compare_baab_runs(
        b1, a1, a2, b2, metrics=("wall_ms",), block_size=4, n_resamples=20, seed=1,
    )
    assert report["order"] == ["B", "A", "A", "B"]
    assert report["verdict"] is None
    wall = report["metrics"]["wall_ms"]
    assert wall["comparable"] is True
    assert wall["effect"] < 0
    assert wall["bootstrap"]["A1_minus_B1"]["mean"] == pytest.approx(24.01 - 28.27)
    assert "pass threshold" not in report["notes"].lower() or "does not invent" in report["notes"]
    assert report["receipt_identity_failures"] == []
    assert report["receipts"]["B1"] == receipt



def test_paired_bootstrap_fails_closed_on_length_mismatch():
    compare = load_compare_module()
    try:
        compare.paired_block_bootstrap([1.0, 2.0], [1.0], block_size=1, n_resamples=2)
    except ValueError as exc:
        assert "length mismatch" in str(exc)
    else:
        raise AssertionError("unequal paired series were accepted")


def test_legacy_receiptless_baseline_is_identity_mismatch():
    compare = load_compare_module()
    actual = _receipt()
    assert compare.receipt_identity_failures({}, actual) == [
        "baseline is missing receipt identity",
    ]
    assert compare.receipt_identity_failures({"video_sha256": "a" * 64}, actual) == [
        "baseline is missing receipt identity",
    ]
    assert compare.receipt_identity_failures({"receipt": None}, actual) == [
        "baseline is missing receipt identity",
    ]
    nested = {"input_identity": {"receipt": actual}}
    assert compare.receipt_identity_failures(nested, actual) == []
    assert compare.result_receipt(nested) == actual


def test_receipt_mismatch_and_missing_candidate_fail_closed():
    compare = load_compare_module()
    actual = _receipt()
    failures = compare.receipt_identity_failures(
        {"receipt": _receipt(worker_mode="production-path")}, actual,
    )
    assert any("receipt.worker_mode mismatch" in item for item in failures)
    assert compare.receipt_identity_failures({"receipt": actual}, None) == [
        "candidate is missing receipt identity",
    ]


def test_baab_legacy_receiptless_runs_fail_closed_without_pass_threshold():
    compare = load_compare_module()
    report = compare.compare_baab_runs(
        _series_result([28.27] * 8),
        _series_result([24.01] * 8),
        _series_result([24.50] * 8),
        _series_result([28.40] * 8),
        metrics=("wall_ms",),
        block_size=4,
        n_resamples=20,
        seed=1,
    )
    assert report["verdict"] is None
    assert "pass" not in report
    assert any("missing receipt identity" in item for item in report["receipt_identity_failures"])
    assert report["metrics"]["wall_ms"]["comparable"] is False
    assert report["metrics"]["wall_ms"]["bootstrap"] is None


def test_baab_nested_input_identity_receipts_are_consumed():
    compare = load_compare_module()
    receipt = _receipt(worker_mode="production-path")

    def wrapped(values):
        payload = _series_result(values)
        payload["input_identity"] = {"receipt": receipt}
        return payload

    report = compare.compare_baab_runs(
        wrapped([28.27] * 4),
        wrapped([24.01] * 4),
        wrapped([24.50] * 4),
        wrapped([28.40] * 4),
        metrics=("wall_ms",),
        block_size=2,
        n_resamples=10,
        seed=2,
    )
    assert report["verdict"] is None
    assert report["receipt_identity_failures"] == []
    assert report["receipts"]["A1"]["worker_mode"] == "production-path"
    assert report["metrics"]["wall_ms"]["comparable"] is True



def test_baab_identity_failures_name_missing_tuning_keys():
    compare = load_compare_module()
    full = _receipt()
    missing = dict(full)
    del missing["fused_coarse_mode"]
    del missing["runtime_sigma_mode"]
    del missing["temporal_feature_cache_size"]
    del missing["acquire_stage_mode"]
    del missing["lost_prior_strategy"]
    del missing["lost_prior_fusion_weight"]
    failures = compare.receipt_identity_failures({"receipt": full}, missing)
    assert "candidate is missing receipt.fused_coarse_mode" in failures
    assert "candidate is missing receipt.runtime_sigma_mode" in failures
    assert "candidate is missing receipt.temporal_feature_cache_size" in failures
    assert "candidate is missing receipt.acquire_stage_mode" in failures
    assert "candidate is missing receipt.lost_prior_strategy" in failures
    assert "candidate is missing receipt.lost_prior_fusion_weight" in failures


def test_fused_coarse_mode_is_not_runtime_sigma_mode():
    compare = load_compare_module()
    expected = _receipt(fused_coarse_mode="fused", runtime_sigma_mode="bidirectional")
    coarse_only = _receipt(
        fused_coarse_mode="upstream", runtime_sigma_mode="bidirectional",
    )
    failures = compare.receipt_identity_failures({"receipt": expected}, coarse_only)
    assert any("receipt.fused_coarse_mode mismatch" in item for item in failures)
    assert not any("runtime_sigma_mode mismatch" in item for item in failures)
    sigma_only = _receipt(
        fused_coarse_mode="fused", runtime_sigma_mode="reference_grid",
    )
    failures = compare.receipt_identity_failures({"receipt": expected}, sigma_only)
    assert any("receipt.runtime_sigma_mode mismatch" in item for item in failures)
    assert not any("fused_coarse_mode mismatch" in item for item in failures)


def test_lost_strategy_is_not_lost_prior_strategy():
    compare = load_compare_module()
    expected = _receipt(
        lost_strategy="boot_and_lost_once",
        lost_prior_strategy="restrict_nearby",
    )
    actual = _receipt(
        lost_strategy="boot_and_lost_once",
        lost_prior_strategy="full_global",
    )
    failures = compare.receipt_identity_failures({"receipt": expected}, actual)
    assert any("receipt.lost_prior_strategy mismatch" in item for item in failures)
    assert not any("receipt.lost_strategy mismatch" in item for item in failures)
