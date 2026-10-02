from __future__ import annotations

import os
import hashlib
import json
import subprocess
import sys

import pytest
from pathlib import Path

import export_simulator_package as exporter
from package_manifest import verify


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "一鍵啟動.sh"


def _make_portable_validator(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    tools_dir = source_root / "tools"
    tools_dir.mkdir(parents=True)
    venv_bin = source_root / ".venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    validator_script = tools_dir / "test_portable_runtime.sh"
    validator_script.write_bytes((ROOT / "tools/test_portable_runtime.sh").read_bytes())
    validator_script.chmod(0o755)
    return validator_script


def test_live_minimal_export_uses_live_entrypoint_and_filters_tests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "portable"
    (source / "code").mkdir(parents=True)
    (source / "code/runtime.py").write_text("runtime\n", encoding="utf-8")
    (source / "code/test_runtime.py").write_text("test\n", encoding="utf-8")
    artifact = source / "artifact.bin"
    artifact.write_bytes(b"trusted")
    (source / "RUNTIME_ARTIFACTS.json").write_text(
        json.dumps(
            {
                "schema": "sfm-runtime-artifacts/v1",
                "seed": {"layout": "repository-relative", "hint": "fixture"},
                "artifacts": [
                    {
                        "name": "fixture",
                        "path": "artifact.bin",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exporter, "ROOT", source)
    monkeypatch.setattr(exporter, "LIVE_MINIMAL_COPY_DIRS", ("code",))
    monkeypatch.setattr(exporter, "LIVE_MINIMAL_COPY_FILES", ("RUNTIME_ARTIFACTS.json",))
    monkeypatch.setattr(exporter, "LIVE_MINIMAL_COPY_MAPPINGS", ())
    monkeypatch.setattr(exporter, "_source_release", lambda: {})

    exporter.export(destination, live_minimal=True)

    metadata = json.loads((destination / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "sfm-portable-live-runtime/v1"
    assert metadata["package_kind"] == "live-operator-runtime"
    assert metadata["entrypoint"] == "一鍵啟動.sh"
    assert (destination / "code/runtime.py").is_file()
    assert not (destination / "code/test_runtime.py").exists()
    assert verify(destination) == []


@pytest.mark.skipif(
    not (ROOT / "地圖檔/場域/river_site/site_profile.json").is_file(),
    reason="requires private river site bundle",
)
def test_one_click_launcher_admits_operator_accepted_map_for_localization_and_flight() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SFM_LAUNCH_DRY_RUN": "1",
            "SFM_UI_PYTHON": sys.executable,
            "SFM_LOCALIZER_PYTHON": sys.executable,
            "SFM_MAX_PERFORMANCE": "0",
        }
    )
    environment.pop("SFM_EVALUATION_ONLY", None)

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mission-selection:" in completed.stdout
    assert "dry-run command" in completed.stdout
    assert "EVALUATION-ONLY session" not in completed.stderr


def test_one_click_rejects_direct_site_profile_override() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SFM_LAUNCH_DRY_RUN": "1",
            "SFM_UI_PYTHON": sys.executable,
            "SFM_MAX_PERFORMANCE": "0",
        }
    )
    environment.pop("SFM_SITE_PROFILE", None)
    environment.pop("SFM_MISSION_SELECTION", None)

    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--site-profile=/tmp/old site.json"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 2
    assert "mission selection" in completed.stderr


def test_one_click_rejects_existing_site_profile_environment() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SFM_LAUNCH_DRY_RUN": "1",
            "SFM_UI_PYTHON": sys.executable,
            "SFM_MAX_PERFORMANCE": "0",
            "SFM_SITE_PROFILE": "/tmp/old site.json",
        }
    )
    environment.pop("SFM_MISSION_SELECTION", None)

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 2
    assert "SFM_SITE_PROFILE" in completed.stderr


def test_portable_runtime_smoke_rejects_symlinked_outputs(
    tmp_path: Path,
) -> None:
    validator = _make_portable_validator(tmp_path)
    portable = tmp_path / "portable"
    victim = tmp_path / "victim"
    portable.mkdir()
    victim.mkdir()
    marker = victim / "must-survive.txt"
    marker.write_text("keep\n", encoding="utf-8")
    (portable / "outputs").symlink_to(victim, target_is_directory=True)

    completed = subprocess.run(
        ["bash", str(validator), str(portable)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 2
    assert "refusing symlinked package path" in completed.stderr
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_one_click_rejects_unpackaged_non_git_directory(tmp_path: Path) -> None:
    launcher = tmp_path / "一鍵啟動.sh"
    launcher.write_bytes(LAUNCHER.read_bytes())

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env={**os.environ, "SFM_PYTHON": sys.executable},
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 1
    assert "PORTABLE_PACKAGE.json" in completed.stderr


def test_one_click_rejects_untrusted_existing_portable_venv(
    tmp_path: Path,
) -> None:
    portable = tmp_path / "portable"
    (portable / "tools").mkdir(parents=True)
    (portable / ".venv/bin").mkdir(parents=True)
    (portable / "一鍵啟動.sh").write_bytes(LAUNCHER.read_bytes())
    (portable / "PORTABLE_PACKAGE.json").write_text("{}\n", encoding="utf-8")
    (portable / "MANIFEST.tsv").write_text("fixture\n", encoding="utf-8")
    (portable / "tools/package_manifest.py").write_text("pass\n", encoding="utf-8")
    fake_python = portable / ".venv/bin/python"
    fake_python.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    fake_python.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(portable / "一鍵啟動.sh")],
        cwd=portable,
        env={**os.environ, "SFM_PYTHON": sys.executable},
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 1
    assert "拒絕未由目前 portable package 建立" in completed.stderr


def test_portable_runtime_rejects_unknown_package_kind(
    tmp_path: Path,
) -> None:
    validator = _make_portable_validator(tmp_path)
    portable = tmp_path / "portable"
    portable.mkdir()
    (portable / "outputs").mkdir()
    (portable / "outputs/README.md").write_text("governance\n", encoding="utf-8")
    (portable / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "schema": "sfm-portable-simulator/v2",
                "package_kind": "unknown-runtime",
                "entrypoint": "tools/unknown.sh",
                "offline_install": {"complete": True},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(validator), str(portable)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode != 0
    assert "unknown package_kind" in completed.stderr


def test_portable_runtime_rejects_non_string_package_kind(
    tmp_path: Path,
) -> None:
    validator = _make_portable_validator(tmp_path)
    portable = tmp_path / "portable"
    portable.mkdir()
    (portable / "outputs").mkdir()
    (portable / "outputs/README.md").write_text("governance\n", encoding="utf-8")
    (portable / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "schema": "sfm-portable-simulator/v2",
                "package_kind": ["unhashable", "list"],
                "entrypoint": "tools/unknown.sh",
                "offline_install": {"complete": True},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(validator), str(portable)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode != 0
    assert "package_kind must be a string" in completed.stderr


def test_portable_runtime_rejects_malformed_metadata(
    tmp_path: Path,
) -> None:
    validator = _make_portable_validator(tmp_path)
    portable = tmp_path / "portable"
    portable.mkdir()
    (portable / "outputs").mkdir()
    (portable / "outputs/README.md").write_text("governance\n", encoding="utf-8")
    (portable / "PORTABLE_PACKAGE.json").write_text("{not json", encoding="utf-8")

    completed = subprocess.run(
        ["bash", str(validator), str(portable)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode != 0
    assert "malformed PORTABLE_PACKAGE.json" in completed.stderr


def test_portable_runtime_checks_simulated_interface_entrypoint(
    tmp_path: Path,
) -> None:
    validator = _make_portable_validator(tmp_path)
    portable = tmp_path / "portable"
    portable.mkdir()
    (portable / "outputs").mkdir()
    (portable / "outputs/README.md").write_text("governance\n", encoding="utf-8")
    (portable / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "schema": "sfm-portable-simulator/v2",
                "package_kind": "simulated-interface-runtime",
                "entrypoint": "控制介面程式/影片模擬串流/選擇啟動.sh",
                "offline_install": {"complete": True},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(validator), str(portable)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode != 0
    assert "控制介面程式/影片模擬串流/選擇啟動.sh" in completed.stderr
    assert "一鍵啟動.sh" not in completed.stderr


def test_portable_runtime_checks_live_operator_entrypoint(
    tmp_path: Path,
) -> None:
    validator = _make_portable_validator(tmp_path)
    portable = tmp_path / "portable"
    portable.mkdir()
    (portable / "outputs").mkdir()
    (portable / "outputs/README.md").write_text("governance\n", encoding="utf-8")
    (portable / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "schema": "sfm-portable-live-runtime/v1",
                "package_kind": "live-operator-runtime",
                "entrypoint": "一鍵啟動.sh",
                "offline_install": {"complete": True},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(validator), str(portable)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode != 0
    assert "一鍵啟動.sh" in completed.stderr
