from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "check_runtime_mirrors.py"
SPEC = importlib.util.spec_from_file_location("check_runtime_mirrors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_runtime_mirrors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_runtime_mirrors)


def test_retained_runtime_mirrors_are_identical():
    assert check_runtime_mirrors.check_mirrors() == []
