from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools" / "install_runtime.sh"
PORTABLE_RUNTIME = ROOT / "tools" / "test_portable_runtime.sh"


def _offline_install_complete(metadata_path: Path) -> bool:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    offline_install = metadata.get("offline_install")
    return isinstance(offline_install, dict) and offline_install.get("complete") is True


def test_installer_documents_and_constructs_offline_pip_mode() -> None:
    script = INSTALLER.read_text(encoding="utf-8")

    assert "--offline" in script
    assert "SFM_INSTALL_OFFLINE=1" in script
    assert 'offline_wheelhouse="$root_dir/執行環境/offline_wheelhouse"' in script
    assert 'offline_wheelhouse_manifest="$offline_wheelhouse/WHEELHOUSE.json"' in script
    assert 'offline_wheelhouse_tool="$root_dir/tools/offline_wheelhouse.py"' in script
    assert '"$python_bin" "$offline_wheelhouse_tool" verify' in script
    assert script.count('--requirements "$requirements_') == 3
    assert '"$offline_wheelhouse_tool" prepare-lock' in script
    assert script.count("install_requirements_") >= 6
    assert "pip_install_args=(--require-hashes)" in script
    assert "pip_global_args=(--disable-pip-version-check)" in script
    assert "pip_global_args+=(--isolated)" in script
    assert (
        "pip_install_args+=(--no-index --only-binary=:all: "
        '--find-links "$offline_wheelhouse")' in script
    )
    assert script.count('"${pip_install_args[@]}"') == 3
    assert script.index("prepare-lock") < script.index("pip_install_args=")


def test_portable_runtime_requires_literal_offline_marker_and_payload() -> None:
    script = PORTABLE_RUNTIME.read_text(encoding="utf-8")

    assert '"$portable_root/PORTABLE_PACKAGE.json"' in script
    assert '"$portable_root/執行環境/offline_wheelhouse/WHEELHOUSE.json"' in script
    assert 'offline_install.get("complete") is not True' in script
    assert 'find "$offline_wheelhouse" -type f ! -name WHEELHOUSE.json' in script


def test_portable_runtime_forces_pip_offline_and_runs_offline_installer() -> None:
    script = PORTABLE_RUNTIME.read_text(encoding="utf-8")

    assert "PIP_NO_INDEX=1" in script
    assert "PIP_INDEX_URL=http://127.0.0.1:9/invalid" in script
    assert "PIP_EXTRA_INDEX_URL= " in script
    assert 'bash "$portable_root/tools/install_runtime.sh" --offline' in script
    assert "looking in indexes:" in script
    assert "https?://" in script


def test_offline_completion_marker_is_a_strict_json_boolean(tmp_path: Path) -> None:
    metadata_path = tmp_path / "PORTABLE_PACKAGE.json"

    metadata_path.write_text(json.dumps({"offline_install": {"complete": True}}), encoding="utf-8")
    assert _offline_install_complete(metadata_path)

    for value in (1, "true", "True", False, None):
        metadata_path.write_text(
            json.dumps({"offline_install": {"complete": value}}), encoding="utf-8"
        )
        assert not _offline_install_complete(metadata_path)
