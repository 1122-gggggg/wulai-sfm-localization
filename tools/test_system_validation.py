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
