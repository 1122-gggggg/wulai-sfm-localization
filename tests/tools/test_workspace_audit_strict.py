from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "workspace_audit.py"
SPEC = importlib.util.spec_from_file_location("workspace_audit_strict", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
workspace_audit = importlib.util.module_from_spec(SPEC)  # type: ignore[attr-defined]
sys.modules[SPEC.name] = workspace_audit
SPEC.loader.exec_module(workspace_audit)  # type: ignore[union-attr]

classify_output = workspace_audit.classify_output
audit_workspace = workspace_audit.audit_workspace
REQUIRED_DIRECTORIES = workspace_audit.REQUIRED_DIRECTORIES
REQUIRED_FILES = workspace_audit.REQUIRED_FILES
CANONICAL_ENTRYPOINTS = workspace_audit.CANONICAL_ENTRYPOINTS


def _minimal_workspace(root: Path) -> None:
    for name in REQUIRED_DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_FILES, *CANONICAL_ENTRYPOINTS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")


def test_audit_20260805_is_governance_not_failure(tmp_path: Path) -> None:
    # AUDIT_OUTPUT_RE = r"audit_\\d{8}" must treat this date-stamped audit as governance
    assert classify_output("audit_20260805") == "governance"
    assert classify_output("audit_20260805") != "unclassified"

    # workspace-level: strict-output-names must not flag audit_20260805 as failure
    _minimal_workspace(tmp_path)
    (tmp_path / "outputs" / "audit_20260805").mkdir(parents=True)
    result = audit_workspace(tmp_path, include_sizes=False)
    assert "audit_20260805" in result["output_classes"]["governance"]
    assert "audit_20260805" not in result["output_classes"]["unclassified"]
    assert result["output_classes"]["unclassified"] == []
