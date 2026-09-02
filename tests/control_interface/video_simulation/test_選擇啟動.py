from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "控制介面程式"
    / "影片模擬串流"
    / "選擇啟動.py"
)
SPEC = importlib.util.spec_from_file_location("simulator_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


ROOT = SCRIPT.parents[2]
RIVER_MAP = (
    ROOT
    / "地圖檔/場域/river_site/releases"
    / "river_gluemap_all8_direct_20260831/map/map.ply"
)
RIVER_PROFILE = ROOT / "地圖檔/場域/river_site/site_profile.json"


def test_map_selection_resolves_the_matching_profile(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"test-video")
    profile_path, map_path, _video = MODULE.select_inputs(
        ROOT,
        map_path=RIVER_MAP,
        video_path=video_path,
    )
    assert profile_path == RIVER_PROFILE.resolve()
    assert map_path == RIVER_MAP.resolve()


def test_unknown_map_fails_closed(tmp_path: Path) -> None:
    unknown_map = tmp_path / "unknown.ply"
    unknown_map.write_bytes(b"ply\nend_header\n")
    with pytest.raises(MODULE.SelectionError, match="找不到與此 PLY"):
        MODULE.profile_for_map(ROOT, unknown_map)


def test_python_executable_preserves_venv_symlink(tmp_path: Path, monkeypatch) -> None:
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(MODULE.sys.executable).resolve())
    monkeypatch.setenv("SFM_UI_PYTHON", str(venv_python))

    assert MODULE.python_executable() == venv_python
    assert MODULE.python_executable() != venv_python.resolve()
