#!/usr/bin/env python3
"""Export fixed simulator/UI/runtime assets for another computer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY_DIRS = (
    "控制介面程式",
    "定位演算法",
    "模擬器/parrot_stimulate",
    "文件",
    "tools",
)
COPY_FILES = (
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-test.txt",
    "requirements-test-lock.txt",
    "驗證系統.sh",
)
EXCLUDED_NAMES = {
    ".git",
    ".codegraph",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "EDM工具包",
    "inductor_cache",
    "outputs",
    "package_git",
    "report_20260714",
    "source_videos",
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDED_NAMES}


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=_ignore)


def _source_release() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from tools.package_manifest import verify
    from tools.release_contract import source_release_identity

    issues = verify(ROOT)
    if issues:
        raise ValueError(f"source package manifest is stale: {issues[0]}")
    return source_release_identity(ROOT)


def _write_metadata(destination: Path, source_release: dict[str, object]) -> None:
    metadata = {
        "schema": "sfm-portable-simulator/v2",
        "package_kind": "simulated-interface-runtime",
        "entrypoint": "控制介面程式/影片模擬串流/選擇啟動.sh",
        "cli_entrypoint": "控制介面程式/影片模擬串流/啟動.sh",
        "install": "bash tools/install_runtime.sh",
        "fixed_assets": [
            "定位演算法/deploy_code/runtime/EDM/weights/edm_outdoor.ckpt",
            "執行環境/torch_hub_cache/gmberton_MegaLoc_main/hubconf.py",
            "執行環境/torch_hub_cache/gmberton_MegaLoc_main/megaloc_model.py",
            "執行環境/torch_hub_cache/checkpoints/megaloc/7cb9f7970d366fdf059963d04d372e503e8e9df9/model.safetensors",
            "模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py",
        ],
        "site_import": {
            "maps_root": "地圖檔/場域/<site>/",
            "videos_root": "模擬器/測試影片/",
            "requires_complete_site_package": True,
            "requires_site_profile": True,
        },
        "excluded_from_package": sorted(EXCLUDED_NAMES),
        "source_release": source_release,
        "supported_gpu": "NVIDIA RTX 5060 + CUDA 12.8 runtime (validated target)",
    }
    (destination / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export(destination: Path) -> list[str]:
    destination = destination.expanduser().resolve()
    if (
        destination == ROOT
        or ROOT.is_relative_to(destination)
        or destination.is_relative_to(ROOT)
    ):
        raise ValueError("destination and source workspace must not contain each other")
    source_release = _source_release()
    if destination.exists():
        if any(destination.iterdir()):
            raise ValueError(f"destination must be empty: {destination}")
    else:
        destination.mkdir(parents=True)

    for relative in COPY_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in COPY_DIRS:
        source = ROOT / relative
        if not source.is_dir():
            raise FileNotFoundError(source)
        _copy_tree(source, destination / relative)

    runtime_source = ROOT / "執行環境"
    runtime_target = destination / "執行環境"
    runtime_target.mkdir(parents=True, exist_ok=True)
    for relative in ("requirements_runtime.txt", "requirements_test.txt", "README.md"):
        shutil.copy2(runtime_source / relative, runtime_target / relative)
    _copy_tree(runtime_source / "torch_hub_cache", runtime_target / "torch_hub_cache")
    _write_metadata(destination, source_release)

    sys.path.insert(0, str(ROOT))
    from tools.package_manifest import generate

    entries = generate(destination)
    return [entry.path for entry in entries]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        files = export(args.destination)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"portable package exported: {args.destination.resolve()}")
    print(f"manifest entries: {len(files)}")
    print("verify: python tools/package_manifest.py verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
