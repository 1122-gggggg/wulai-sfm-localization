from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "workspace_audit.py"
SPEC = importlib.util.spec_from_file_location("workspace_audit_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
workspace_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workspace_audit
SPEC.loader.exec_module(workspace_audit)

CANONICAL_ENTRYPOINTS = workspace_audit.CANONICAL_ENTRYPOINTS
REQUIRED_DIRECTORIES = workspace_audit.REQUIRED_DIRECTORIES
REQUIRED_FILES = workspace_audit.REQUIRED_FILES
audit_workspace = workspace_audit.audit_workspace
classify_output = workspace_audit.classify_output


def _minimal_workspace(root: Path) -> None:
    for name in REQUIRED_DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_FILES, *CANONICAL_ENTRYPOINTS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")


def test_workspace_audit_accepts_the_canonical_layout(tmp_path: Path) -> None:
    _minimal_workspace(tmp_path)
    (tmp_path / "outputs" / "flight_logs").mkdir()
    (tmp_path / "outputs" / "validation_receipts").mkdir()

    result = audit_workspace(tmp_path, include_sizes=False)

    assert result["ok"]
    assert result["structural_failures"] == []
    assert result["output_classes"]["operations"] == ["flight_logs"]
    assert result["output_classes"]["validation"] == ["validation_receipts"]


def test_workspace_audit_requires_output_governance_readme(tmp_path: Path) -> None:
    _minimal_workspace(tmp_path)
    (tmp_path / "outputs/README.md").unlink()

    result = audit_workspace(tmp_path, include_sizes=False)

    assert not result["ok"]
    assert "missing file: outputs/README.md" in result["structural_failures"]


def test_workspace_audit_reports_missing_paths_and_broken_links(tmp_path: Path) -> None:
    _minimal_workspace(tmp_path)
    (tmp_path / "文件/SYSTEM_SPEC.md").unlink()
    os.symlink("missing-target", tmp_path / "broken-link")

    result = audit_workspace(tmp_path, include_sizes=False)

    assert not result["ok"]
    assert "missing file: 文件/SYSTEM_SPEC.md" in result["structural_failures"]
    assert "broken symlink: broken-link" in result["structural_failures"]


def test_workspace_audit_requires_embedded_parrot_control_core(tmp_path: Path) -> None:
    _minimal_workspace(tmp_path)
    core = (
        tmp_path
        / "模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py"
    )
    core.unlink()

    result = audit_workspace(tmp_path, include_sizes=False)

    assert not result["ok"]
    assert (
        "missing file: 模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py"
        in result["structural_failures"]
    )


def test_output_classification_keeps_new_names_visible() -> None:
    assert classify_output("edm_speed_20260803") == "experiment_evidence"
    assert classify_output("validation_receipts") == "validation"
    assert classify_output("security") == "validation"
    assert classify_output("flight_logs") == "operations"
    assert classify_output("misc") == "unclassified"


def test_workspace_audit_classifies_date_stamped_audits_and_warns_on_storage(
    tmp_path: Path, monkeypatch
) -> None:
    _minimal_workspace(tmp_path)
    (tmp_path / "outputs" / "audit_20260807").mkdir()
    monkeypatch.setattr(
        workspace_audit.shutil,
        "disk_usage",
        lambda _path: workspace_audit.shutil._ntuple_diskusage(
            total=100 * 1024**3,
            used=86 * 1024**3,
            free=14 * 1024**3,
        ),
    )

    result = audit_workspace(tmp_path, include_sizes=False)

    assert result["output_classes"]["governance"] == ["README.md", "audit_20260807"]
    assert any("free space is below 15%" in warning for warning in result["storage_warnings"])


def test_workspace_audit_warns_when_workspace_exceeds_20_gib(tmp_path: Path, monkeypatch) -> None:
    _minimal_workspace(tmp_path)
    monkeypatch.setattr(workspace_audit, "directory_size", lambda _path: 21 * 1024**3)
    monkeypatch.setattr(
        workspace_audit.shutil,
        "disk_usage",
        lambda _path: workspace_audit.shutil._ntuple_diskusage(
            total=100 * 1024**3,
            used=50 * 1024**3,
            free=50 * 1024**3,
        ),
    )

    result = audit_workspace(tmp_path, include_sizes=True)

    assert any("workspace size is above 20 GiB" in warning for warning in result["storage_warnings"])
