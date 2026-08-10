from __future__ import annotations

import os
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import export_simulator_package as exporter
from package_manifest import verify


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "一鍵啟動.sh"


def test_live_minimal_export_contract_excludes_development_payload() -> None:
    assert "一鍵啟動.sh" in exporter.LIVE_MINIMAL_COPY_FILES
    assert "requirements/runtime-lock.txt" in exporter.LIVE_MINIMAL_COPY_FILES
    assert "tools/simulator_preflight.py" in exporter.LIVE_MINIMAL_COPY_FILES
    assert (
        "模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py"
        in exporter.LIVE_MINIMAL_COPY_FILES
    )
    assert (
        "模擬器/測試影片/河濱_P1180118_first_2s.mp4",
        "執行環境/smoke/river_site_first_2s.mp4",
    ) in exporter.LIVE_MINIMAL_COPY_MAPPINGS
    assert "控制介面程式" in exporter.LIVE_MINIMAL_COPY_DIRS
    assert "定位演算法/flight_control" in exporter.LIVE_MINIMAL_COPY_DIRS
    assert "模擬器/parrot_stimulate" not in exporter.LIVE_MINIMAL_COPY_DIRS
    assert "文件" not in exporter.LIVE_MINIMAL_COPY_DIRS
    assert "pyproject.toml" not in exporter.LIVE_MINIMAL_COPY_FILES
    assert "pytest.ini" not in exporter.LIVE_MINIMAL_COPY_FILES


def test_live_minimal_export_uses_live_entrypoint_and_filters_tests(
    tmp_path: Path, monkeypatch,
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
    monkeypatch.setattr(
        exporter, "LIVE_MINIMAL_COPY_FILES", ("RUNTIME_ARTIFACTS.json",)
    )
    monkeypatch.setattr(exporter, "LIVE_MINIMAL_COPY_MAPPINGS", ())
    monkeypatch.setattr(exporter, "_source_release", lambda: {})

    exporter.export(destination, live_minimal=True)

    metadata = json.loads(
        (destination / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8")
    )
    assert metadata["schema"] == "sfm-portable-live-runtime/v1"
    assert metadata["package_kind"] == "live-operator-runtime"
    assert metadata["entrypoint"] == "一鍵啟動.sh"
    assert (destination / "code/runtime.py").is_file()
    assert not (destination / "code/test_runtime.py").exists()
    assert verify(destination) == []


def test_one_click_launcher_installs_offline_and_selects_river_profile() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "tools/install_runtime.sh\" --offline" in script
    assert "tools/package_manifest.py\" verify" in script
    assert "控制介面程式/site_profiles/river_site_edm.json" in script
    assert "控制介面程式/真機串流/啟動.sh" in script
    assert ".sfm-portable-runtime" in script
    assert "非 Git 原始碼目錄必須包含 PORTABLE_PACKAGE.json" in script
    assert "exec" in script

    smoke_script = (ROOT / "tools/simulated_ui_smoke.sh").read_text(encoding="utf-8")
    assert "執行環境/smoke/river_site_first_2s.mp4" in smoke_script
    assert 'SFM_WORKSPACE_ROOT="$root_dir" setsid' in smoke_script


def test_export_cli_forwards_live_minimal_mode(
    tmp_path: Path, monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_export(destination: Path, **kwargs: object) -> list[str]:
        calls["destination"] = destination
        calls.update(kwargs)
        return []

    monkeypatch.setattr(exporter, "export", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_simulator_package.py", str(tmp_path / "portable"), "--live-minimal"],
    )

    assert exporter.main() == 0
    assert calls["live_minimal"] is True


def test_one_click_launcher_real_flight_dry_run_does_not_connect() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SFM_LAUNCH_DRY_RUN": "1",
            "SFM_UI_PYTHON": sys.executable,
            "SFM_LOCALIZER_PYTHON": sys.executable,
            "SFM_MAX_PERFORMANCE": "0",
        }
    )

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
    assert "--interface real-flight" in completed.stdout
    assert "river_site_edm.json" in completed.stdout


def test_portable_runtime_smoke_rejects_symlinked_outputs(
    tmp_path: Path,
) -> None:
    portable = tmp_path / "portable"
    victim = tmp_path / "victim"
    portable.mkdir()
    victim.mkdir()
    marker = victim / "must-survive.txt"
    marker.write_text("keep\n", encoding="utf-8")
    (portable / "outputs").symlink_to(victim, target_is_directory=True)

    completed = subprocess.run(
        ["bash", str(ROOT / "tools/test_portable_runtime.sh"), str(portable)],
        cwd=ROOT,
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
