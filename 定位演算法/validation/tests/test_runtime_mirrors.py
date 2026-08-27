from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "check_runtime_mirrors.py"
SPEC = importlib.util.spec_from_file_location("check_runtime_mirrors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_runtime_mirrors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_runtime_mirrors)

ALGO_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_CONTROL = ALGO_ROOT / "flight_control"
DEPLOY = ALGO_ROOT / "deploy_code" / "sfm_glomap_deploy"


def test_runtime_modules_have_one_authoritative_copy() -> None:
    assert check_runtime_mirrors.check_mirrors() == []

    for canonical_rel in check_runtime_mirrors.CANONICAL_MODULES.values():
        canonical = ALGO_ROOT / canonical_rel
        assert canonical.is_file()
        assert not (ALGO_ROOT / check_runtime_mirrors._other_runtime_path(canonical_rel)).exists()


def test_only_documentation_has_the_same_name_in_both_runtime_trees() -> None:
    assert check_runtime_mirrors.same_named_files() == {"README.md"}


def test_canonical_frame_source_preserves_live_safety_contract() -> None:
    source = (FLIGHT_CONTROL / "olympe_frame_source.py").read_text(encoding="utf-8")

    assert not (DEPLOY / "olympe_frame_source.py").exists()
    assert "SFM_ALLOW_LEGACY_FLIGHT" not in source
    assert "TakeOff()" not in source
    assert ".wait()" not in source
    assert ".join()" not in source
    assert ".wait(timeout=0.5)" in source
    assert ".join(timeout=5.0)" in source
    assert "require_source_timestamps" in source
    assert "_source_timing_trusted_locked" in source
