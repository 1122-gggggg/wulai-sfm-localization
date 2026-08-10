#!/usr/bin/env python3
"""Read-only structure and storage audit for the localization workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
import re
from typing import Any

REQUIRED_DIRECTORIES = (
    "定位演算法",
    "控制介面程式",
    "地圖檔",
    "模擬器",
    "執行環境",
    "outputs",
    "文件",
    "tools",
)
REQUIRED_FILES = (
    "README.md",
    "MANIFEST.tsv",
    "SHA256SUMS",
    "requirements/README.md",
    "requirements/runtime.txt",
    "requirements/runtime-lock.txt",
    "requirements/test.txt",
    "requirements/test-lock.txt",
    "requirements/quality.txt",
    "requirements/quality-lock.txt",
    "pytest.ini",
    "pyproject.toml",
    "文件/README.md",
    "文件/ARCHITECTURE.md",
    "文件/SYSTEM_SPEC.md",
    "outputs/README.md",
    "tools/README.md",
    "tools/install_runtime.sh",
    "tools/offline_wheelhouse.py",
    "tools/test_clean_install.sh",
    "tools/simulated_ui_smoke.sh",
    "tools/export_simulator_package.py",
    "tools/package_manifest.py",
    "tools/simulator_preflight.py",
    "tools/system_validation.py",
    "tools/release_contract.py",
    "tools/release_activation.py",
    "驗證系統.sh",
    "模擬器/parrot_stimulate/pyproject.toml",
    "模擬器/parrot_stimulate/firmware/manifest.json",
    "模擬器/parrot_stimulate/firmware/anafi-pc.ext2.zip",
    "模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py",
    "控制介面程式/影片模擬串流/選擇啟動.sh",
    "控制介面程式/影片模擬串流/選擇啟動.py",
)
CANONICAL_ENTRYPOINTS = (
    "控制介面程式/影片模擬串流/啟動.sh",
    "控制介面程式/真機串流/啟動.sh",
    "控制介面程式/mission_pipeline.py",
    "定位演算法/deploy_code/sfm_glomap_deploy/production_edm_tracker.py",
    "定位演算法/configs/edm_production_profile.json",
    "定位演算法/validation/check_runtime_mirrors.py",
)
GENERATED_TOP_LEVEL = {
    ".codegraph",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
OUTPUT_GOVERNANCE_FILES = {
    "README.md",
    "EXPERIMENT_BACKLOG.md",
    "EXPERIMENT_LEDGER.md",
    "EXPERIMENT_RECORD_TEMPLATE.json",
    "LAB_MANUAL_FLIGHT_CHECKLIST_20260710.md",
    "PROJECT_REVIEW_20260719.md",
    "測試組合效能比較_20260715.md",
}
OUTPUT_EVIDENCE_PREFIXES = (
    "edm_",
    "exact_latency_",
    "localization_fps_",
    "onnx_flow_",
    "optimization_",
    "regression_",
    "reverse_topk_",
    "video720_",
)
WORKSPACE_SIZE_WARNING_BYTES = 20 * 1024**3
FREE_SPACE_WARNING_PERCENT = 15.0
AUDIT_OUTPUT_RE = re.compile(r"audit_\d{8}$")


def _walk_without_following_links(root: Path) -> Iterable[tuple[Path, os.stat_result]]:
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            try:
                yield path, path.lstat()
            except FileNotFoundError:
                continue


def directory_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child, stat in _walk_without_following_links(path):
        if child.is_symlink() or not child.is_file():
            continue
        total += stat.st_size
    return total


def classify_output(name: str) -> str:
    if name in OUTPUT_GOVERNANCE_FILES:
        return "governance"
    if name == "flight_logs":
        return "operations"
    if name in {
        "validation",
        "validation_receipts",
        "production_stream_bench",
        "security",
    }:
        return "validation"
    if AUDIT_OUTPUT_RE.fullmatch(name):
        return "governance"
    if name.startswith(OUTPUT_EVIDENCE_PREFIXES):
        return "experiment_evidence"
    return "unclassified"


def _missing_workspace_paths(workspace: Path) -> tuple[list[str], list[str]]:
    missing_directories = [name for name in REQUIRED_DIRECTORIES if not (workspace / name).is_dir()]
    required_paths = (*REQUIRED_FILES, *CANONICAL_ENTRYPOINTS)
    missing_files = [name for name in required_paths if not (workspace / name).is_file()]
    return missing_directories, missing_files


def _workspace_symlinks(
    workspace: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    symlinks: list[dict[str, Any]] = []
    broken_symlinks: list[str] = []
    if not workspace.is_dir():
        return symlinks, broken_symlinks
    for path, _stat in _walk_without_following_links(workspace):
        if not path.is_symlink():
            continue
        relative = str(path.relative_to(workspace))
        target = os.readlink(path)
        resolved = path.resolve(strict=False)
        exists = resolved.exists()
        symlinks.append({"path": relative, "target": target, "valid": exists})
        if not exists:
            broken_symlinks.append(relative)
    return symlinks, broken_symlinks


def _workspace_top_level_sizes(workspace: Path, *, include_sizes: bool) -> dict[str, int]:
    top_level_sizes: dict[str, int] = {}
    if not include_sizes or not workspace.is_dir():
        return top_level_sizes
    for path in sorted(workspace.iterdir(), key=lambda item: item.name):
        if path.name == ".git":
            continue
        top_level_sizes[path.name] = directory_size(path)
    return top_level_sizes


def _workspace_output_classes(workspace: Path) -> dict[str, list[str]]:
    output_classes: dict[str, list[str]] = {
        "operations": [],
        "validation": [],
        "experiment_evidence": [],
        "governance": [],
        "unclassified": [],
    }
    output_root = workspace / "outputs"
    if output_root.is_dir():
        for path in sorted(output_root.iterdir(), key=lambda item: item.name):
            output_classes[classify_output(path.name)].append(path.name)
    return output_classes


def _workspace_storage_report(
    workspace: Path,
    top_level_sizes: dict[str, int],
    *,
    include_sizes: bool,
) -> tuple[int | None, list[str], dict[str, int | float]]:
    usage_target = workspace if workspace.exists() else workspace.parent
    usage = shutil.disk_usage(usage_target)
    workspace_size_bytes = sum(top_level_sizes.values()) if include_sizes else None
    storage_warnings: list[str] = []
    if workspace_size_bytes is not None and workspace_size_bytes > WORKSPACE_SIZE_WARNING_BYTES:
        storage_warnings.append(
            "workspace size is above 20 GiB; review generated data before release"
        )
    free_percent = usage.free / usage.total * 100.0 if usage.total else 0.0
    if free_percent < FREE_SPACE_WARNING_PERCENT:
        storage_warnings.append(
            "free space is below 15%; stop before creating large validation artifacts"
        )
    disk = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_percent": free_percent,
    }
    return workspace_size_bytes, storage_warnings, disk


def audit_workspace(root: str | Path, *, include_sizes: bool = True) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    missing_directories, missing_files = _missing_workspace_paths(workspace)
    symlinks, broken_symlinks = _workspace_symlinks(workspace)
    top_level_sizes = _workspace_top_level_sizes(workspace, include_sizes=include_sizes)
    output_classes = _workspace_output_classes(workspace)
    workspace_size_bytes, storage_warnings, disk = _workspace_storage_report(
        workspace,
        top_level_sizes,
        include_sizes=include_sizes,
    )
    structural_failures = [
        *(f"missing directory: {name}" for name in missing_directories),
        *(f"missing file: {name}" for name in missing_files),
        *(f"broken symlink: {name}" for name in broken_symlinks),
    ]
    return {
        "schema": "localization-workspace-audit/v1",
        "root": str(workspace),
        "ok": not structural_failures,
        "structural_failures": structural_failures,
        "missing_directories": missing_directories,
        "missing_files": missing_files,
        "symlinks": symlinks,
        "top_level_bytes": top_level_sizes,
        "output_classes": output_classes,
        "workspace_size_bytes": workspace_size_bytes,
        "storage_warnings": storage_warnings,
        "generated_top_level": sorted(
            name for name in GENERATED_TOP_LEVEL if (workspace / name).exists()
        ),
        "disk": disk,
    }


def _gib(value: int) -> float:
    return value / (1024**3)


def format_human(report: dict[str, Any]) -> str:
    lines = [
        f"workspace: {report['root']}",
        f"layout: {'OK' if report['ok'] else 'FAIL'}",
        (
            "disk: "
            f"{_gib(report['disk']['free_bytes']):.2f} GiB free "
            f"({report['disk']['free_percent']:.1f}%)"
        ),
    ]
    for warning in report.get("storage_warnings", []):
        lines.append(f"WARNING: {warning}")
    sizes = report.get("top_level_bytes", {})
    if sizes:
        lines.append("largest top-level entries:")
        for name, size in sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:10]:
            lines.append(f"  {name}: {_gib(size):.2f} GiB")
    unclassified = report["output_classes"]["unclassified"]
    lines.append(f"output entries not classified: {len(unclassified)}")
    for failure in report["structural_failures"]:
        lines.append(f"ERROR: {failure}")
    for name in unclassified:
        lines.append(f"WARN: outputs/{name}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="workspace root (default: parent of tools/)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--no-sizes",
        action="store_true",
        help="skip recursive top-level size accounting",
    )
    parser.add_argument(
        "--strict-output-names",
        action="store_true",
        help="fail when outputs/ contains an unclassified top-level entry",
    )
    args = parser.parse_args()
    report = audit_workspace(args.root, include_sizes=not args.no_sizes)
    print(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.json
        else format_human(report)
    )
    failed = not report["ok"]
    if args.strict_output_names and report["output_classes"]["unclassified"]:
        failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
