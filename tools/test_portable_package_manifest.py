from __future__ import annotations

from pathlib import Path

import export_simulator_package
import pytest
from package_manifest import generate, included, verify


def test_current_manifest_excludes_imported_assets_and_runtime_caches() -> None:
    assert not included(Path(".venv/bin/python"))
    assert not included(Path("outputs/validation.json"))
    assert not included(Path("地圖檔/場域/river_site/maps/site.ply"))
    assert not included(Path("模擬器/測試影片/replay.mp4"))
    assert not included(Path("模擬器/封存/old_worktree/src/module.py"))
    assert not included(Path("執行環境/inductor_cache/kernel.py"))
    assert included(Path("控制介面程式/影片模擬串流/啟動.sh"))


def test_manifest_excludes_editor_and_transient_release_files() -> None:
    for path in (
        Path(".vscode/settings.json"),
        Path(".idea/workspace.xml"),
        Path("notes.swp"),
        Path("notes~"),
        Path("report.tmp"),
        Path(".DS_Store"),
        Path("audit/final_validation.md"),
        Path(".cursor/session.json"),
        Path("env/bin/python"),
        Path(".coverage"),
        Path(".coverage.host.123.random"),
        Path("coverage.xml"),
        Path("htmlcov/index.html"),
    ):
        assert not included(path)
    assert included(Path(".coveragerc"))


def test_release_tools_have_one_authoritative_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "執行環境/tools/package_manifest.py").exists()
    assert not (root / "執行環境/MANIFEST.tsv").exists()
    assert not (root / "執行環境/SHA256SUMS").exists()
    assert not (root / "定位演算法/sync_mirror_check.sh").exists()


def test_portable_export_keeps_output_governance_readme() -> None:
    import export_simulator_package as exporter

    assert "outputs/README.md" in exporter.COPY_FILES

    runtime_script = (Path(__file__).parent / "test_portable_runtime.sh").read_text(
        encoding="utf-8"
    )
    assert '"$portable_root/outputs/README.md"' in runtime_script
    assert 'rm -r -- "$staged_output_root/flight_logs"' in runtime_script
    assert 'rm -r -- "$staged_output_root"' not in runtime_script
    assert "routes/authored/route_20260807_013811.json" in runtime_script
    assert "maps/T_align_gravity.json" in runtime_script
    assert "reports/hardware_approval_20260806.json" in runtime_script
    assert "river_site_safezone/flight_path.json" not in runtime_script


def test_manifest_round_trip(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.txt").write_text("b", encoding="utf-8")
    generate(tmp_path)
    assert verify(tmp_path) == []
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    assert any(
        marker in issue
        for issue in verify(tmp_path)
        for marker in ("SHA-256 mismatch: a.txt", "size mismatch: a.txt")
    )


def test_export_rejects_a_destination_inside_the_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(export_simulator_package, "ROOT", source)

    with pytest.raises(ValueError, match="must not contain each other"):
        export_simulator_package.export(source / "portable")
