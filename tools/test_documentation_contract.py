from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIVING_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "文件" / "ARCHITECTURE.md",
    ROOT / "文件" / "SYSTEM_SPEC.md",
    ROOT / "定位演算法" / "README.md",
    ROOT / "定位演算法" / "flight_control" / "README.md",
    ROOT / "定位演算法" / "deploy_code" / "sfm_glomap_deploy" / "README.md",
    ROOT / "控制介面程式" / "README.md",
    ROOT / "控制介面程式" / "SAFETY.md",
    ROOT / "控制介面程式" / "operator_interface" / "README.md",
    ROOT / "tools" / "README.md",
)
LINK_PATTERN = re.compile(r"\[[^]]+]\(([^)]+)\)")


def _ownership_module():
    script = ROOT / "定位演算法" / "validation" / "check_runtime_mirrors.py"
    spec = importlib.util.spec_from_file_location("documentation_module_ownership", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_living_documents_do_not_restore_retired_paths_or_mirror_policy() -> None:
    forbidden = {
        "/tmp/sfm_drone_safety.cmd",
        "飛控鏡像",
        "相容鏡像",
        "mission/operator_interface",
        "sfm_system/定位/mission",
    }

    for document in LIVING_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        stale = sorted(token for token in forbidden if token in text)
        assert not stale, f"{document.relative_to(ROOT)} contains stale contracts: {stale}"


def test_architecture_lists_every_canonical_runtime_module() -> None:
    architecture = (ROOT / "文件" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    for name in _ownership_module().CANONICAL_MODULES:
        assert f"`{name}`" in architecture


def test_relative_links_in_living_documents_exist() -> None:
    missing: list[str] = []
    for document in LIVING_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "missing documentation targets:\n" + "\n".join(missing)
