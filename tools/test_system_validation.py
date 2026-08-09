from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_validation_module():
    path = ROOT / "system_validation.py"
    spec = importlib.util.spec_from_file_location("system_validation_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_receipt_tracks_operator_runtime_seams() -> None:
    module = _load_validation_module()
    required = {
        "outputs/README.md",
        "控制介面程式/operator_interface/operator_autonomy.py",
        "控制介面程式/operator_interface/operator_preflight.py",
        "控制介面程式/operator_interface/operator_rendering.py",
        "控制介面程式/operator_interface/operator_tick.py",
    }

    assert required <= set(module.RELEASE_FILES)


def test_production_offline_smoke_requires_edm_not_research_xfeat() -> None:
    module = _load_validation_module()
    steps = {
        step.name: step
        for step in module._steps(
            p119=False,
            accept_p119=False,
            p119_quality=False,
            quality_out=None,
        )
    }

    offline = steps["offline_model_smoke"]
    assert offline.argv[-2:] == ("--model", "edm")

    workspace = steps["workspace_layout"]
    assert workspace.argv[-2:] == ("--strict-output-names", "--no-sizes")

    assert steps["root_ruff_check"].argv[-3:] == ("ruff", "check", ".")
    assert steps["root_ruff_format"].argv[2:5] == ("ruff", "format", "--check")
    assert steps["maintainability_budget"].argv[-1].endswith(
        "tools/check_maintainability.py"
    )
    assert "--cov" in steps["root_pytest"].argv
    assert "--timeout=300" in steps["root_pytest"].argv


def test_parrot_simulator_validation_stays_in_its_python311_environment() -> None:
    module = _load_validation_module()
    steps = {
        step.name: step
        for step in module._steps(
            p119=False,
            accept_p119=False,
            p119_quality=False,
            quality_out=None,
        )
    }

    for name in (
        "parrot_pytest",
        "parrot_ruff_check",
        "parrot_ruff_format",
        "parrot_preflight",
    ):
        assert steps[name].argv[0] == str(module.PARROT_PYTHON)
        assert steps[name].cwd == str(module.PARROT_ROOT)
    assert steps["parrot_lock_check"].cwd == str(module.PARROT_ROOT)
    assert steps["parrot_preflight"].argv[-1] == "preflight"


def test_p119_quality_validation_runs_the_pinned_replay_gate(tmp_path) -> None:
    module = _load_validation_module()

    steps = {
        step.name: step
        for step in module._steps(
            p119=True,
            accept_p119=True,
            p119_quality=True,
            quality_out=tmp_path / "quality.json",
        )
    }

    quality = steps["p119_quality"]
    assert "benchmark_edm_site_replay.py" in " ".join(quality.argv)
    assert "--quality-baseline" in quality.argv
    assert "--accept-known-incomplete" in quality.argv
    assert "--require-cuda" in quality.argv


def test_portable_output_is_verified_and_bound_to_source(tmp_path: Path) -> None:
    module = _load_validation_module()
    package = tmp_path / "portable"
    package.mkdir()
    source_release = {
        "manifest_sha256": module._sha256(module.ROOT / "MANIFEST.tsv"),
        "sha256sums_sha256": module._sha256(module.ROOT / "SHA256SUMS"),
    }
    source_release.update(
        {"commit": "deadbeef", "version": "release-deadbeef", "dirty": False}
    )
    (package / "PORTABLE_PACKAGE.json").write_text(
        json.dumps({"source_release": source_release}), encoding="utf-8"
    )
    (package / "MANIFEST.tsv").write_text("manifest", encoding="utf-8")
    (package / "SHA256SUMS").write_text("sums", encoding="utf-8")

    metadata = module._portable_metadata(package)
    steps = {
        step.name: step
        for step in module._steps(
            p119=False,
            accept_p119=False,
            p119_quality=False,
            quality_out=None,
            portable_package=package,
        )
    }

    assert metadata["complete"]
    assert metadata["source_release_matches"]
    assert steps["portable_output_manifest"].cwd == str(package.resolve())


def test_optional_release_gates_are_explicit_steps() -> None:
    module = _load_validation_module()
    steps = {
        step.name: step
        for step in module._steps(
            p119=False,
            accept_p119=False,
            p119_quality=False,
            quality_out=None,
            clean_install=True,
            ui_smoke=True,
        )
    }

    assert steps["clean_install_full_runtime"].timeout_s == 3600
    assert steps["simulated_ui_smoke"].timeout_s == 180


def test_actual_portable_gets_clean_install_and_valid_pose_gate(tmp_path: Path) -> None:
    module = _load_validation_module()
    package = tmp_path / "portable"
    steps = {
        step.name: step
        for step in module._steps(
            p119=False,
            accept_p119=False,
            p119_quality=False,
            quality_out=None,
            portable_package=package,
            clean_install=True,
            ui_smoke=True,
        )
    }

    gate = steps["portable_clean_install_ui_pose"]
    assert gate.argv[-1] == str(package.resolve())
    assert gate.timeout_s == 3600


def test_system_validation_has_a_read_only_hardware_receipt_step() -> None:
    module = _load_validation_module()
    steps = {
        step.name: step
        for step in module._steps(
            p119=False,
            accept_p119=False,
            p119_quality=False,
            quality_out=None,
        )
    }
    hardware = steps["hardware_snapshot"]
    assert "monitor_hardware.py" in " ".join(hardware.argv)
    assert "--output" in hardware.argv
    assert hardware.argv[hardware.argv.index("--output") + 1] == "-"
    assert "--samples" in hardware.argv
    assert hardware.timeout_s <= 120


def test_simulator_preflight_receipt_includes_collision_monitor_policy() -> None:
    module = _load_validation_module()
    steps = {
        step.name: step
        for step in module._steps(
            p119=False,
            accept_p119=False,
            p119_quality=False,
            quality_out=None,
        )
    }
    preflight = steps["portable_simulator_preflight"]
    assert "--json" in preflight.argv


def test_dirty_release_is_failed_unless_development_opt_out(monkeypatch) -> None:
    module = _load_validation_module()
    monkeypatch.setattr(
        module,
        "_git_metadata",
        lambda: {"commit": "deadbeef", "version": "release-deadbeef", "dirty": True, "changed_path_count": 1},
    )
    assert module._release_gate(module._git_metadata(), allow_dirty=False)["passed"] is False
    assert module._release_gate(module._git_metadata(), allow_dirty=True)["passed"] is True


def test_pytest_summary_records_conditional_skips_for_receipts() -> None:
    module = _load_validation_module()
    summary = module._pytest_summary(
        "test_a.py::test_ok PASSED\n"
        "================ 2 passed, 3 skipped, 1 warning in 0.4s ================\n",
    )

    assert summary is not None
    assert summary["passed"] == 2
    assert summary["skipped"] == 3
    assert summary["warnings"] == 1

    fallback = module._pytest_summary("SKIPPED [3] optional CUDA test\n")
    assert fallback is not None and fallback["skipped"] == 3
