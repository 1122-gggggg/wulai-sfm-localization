from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
    / "river_gluemap_all8_direct_20260908/map/map.ply"
)
RIVER_PROFILE = ROOT / "地圖檔/場域/river_site/site_profile.json"


@pytest.mark.skipif(not RIVER_MAP.is_file(), reason="requires private river site bundle")
def test_map_selection_resolves_the_matching_profile(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"test-video")
    profile_path, map_path, _video = MODULE.select_inputs(
        ROOT,
        map_path=RIVER_MAP,
        video_path=video_path,
        mode=MODULE.MODE_LOCALIZE,
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


def _profile(tmp_path: Path, name: str, workmap, map_sha: str = "m") -> SimpleNamespace:
    runtime = tmp_path / f"{name}.json"
    body = {} if workmap is None else {"online_workmap": {"enabled": workmap}}
    runtime.write_text(json.dumps(body), encoding="utf-8")
    return SimpleNamespace(
        localizer_profile=runtime, asset_sha256=SimpleNamespace(map_ply=map_sha)
    )


def test_profile_mode_follows_the_bound_runtime_profile(tmp_path: Path) -> None:
    assert MODULE.profile_mode(_profile(tmp_path, "a", None)) == MODULE.MODE_LOCALIZE
    assert MODULE.profile_mode(_profile(tmp_path, "b", False)) == MODULE.MODE_LOCALIZE
    assert MODULE.profile_mode(_profile(tmp_path, "c", True)) == MODULE.MODE_EXPAND


def test_expansion_mode_swaps_only_to_a_package_on_the_same_map(
    tmp_path: Path, monkeypatch
) -> None:
    pure = _profile(tmp_path, "pure", None, "map-1")
    other = (tmp_path / "other.json", _profile(tmp_path, "other", True, "map-2"))
    same = (tmp_path / "same.json", _profile(tmp_path, "same", True, "map-1"))
    monkeypatch.setattr(MODULE, "discover_expansion_profiles", lambda _root: [other, same])
    pure_path = tmp_path / "pure_profile.json"
    assert (
        MODULE.resolve_mode_profile(
            tmp_path, pure_path, pure, MODULE.MODE_EXPAND, explicit=False
        )
        == same
    )
    assert MODULE.resolve_mode_profile(
        tmp_path, pure_path, pure, MODULE.MODE_LOCALIZE, explicit=False
    ) == (pure_path, pure)


def test_expansion_without_a_matching_package_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    pure = _profile(tmp_path, "pure", None, "map-1")
    other = (tmp_path / "other.json", _profile(tmp_path, "other", True, "map-2"))
    monkeypatch.setattr(MODULE, "discover_expansion_profiles", lambda _root: [other])
    path = tmp_path / "p.json"
    with pytest.raises(MODULE.SelectionError, match="沒有對應的定位＋擴圖場域包"):
        MODULE.resolve_mode_profile(tmp_path, path, pure, MODULE.MODE_EXPAND, explicit=False)
    # The dialog must receive the reason so the expansion choice is shown as blocked.
    asked: dict[str, str | None] = {}

    def choose(*, expand_blocked):
        asked["blocked"] = expand_blocked
        return MODULE.MODE_LOCALIZE

    monkeypatch.setattr(MODULE, "choose_mode", choose)
    assert MODULE.resolve_mode_profile(tmp_path, path, pure, None, explicit=False) == (path, pure)
    assert "沒有對應的定位＋擴圖場域包" in str(asked["blocked"])


def test_explicit_profile_is_never_swapped_and_conflicting_mode_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    pure = _profile(tmp_path, "pure", None, "map-1")
    expansion = _profile(tmp_path, "exp", True, "map-1")
    monkeypatch.setattr(
        MODULE, "choose_mode", lambda **_kw: pytest.fail("explicit profile must not prompt")
    )
    path = tmp_path / "p.json"
    assert MODULE.resolve_mode_profile(tmp_path, path, pure, None, explicit=True) == (path, pure)
    assert MODULE.resolve_mode_profile(tmp_path, path, expansion, None, explicit=False) == (
        path,
        expansion,
    )
    with pytest.raises(MODULE.SelectionError, match="衝突"):
        MODULE.resolve_mode_profile(
            tmp_path, path, expansion, MODULE.MODE_LOCALIZE, explicit=True
        )
    with pytest.raises(MODULE.SelectionError, match="衝突"):
        MODULE.resolve_mode_profile(tmp_path, path, pure, MODULE.MODE_EXPAND, explicit=True)


EXPANSION_RELEASE = (
    ROOT
    / "地圖檔/場域/river_site/releases"
    / "river_gluemap_all8_online_ui_20260929/site_profile.json"
)


@pytest.mark.skipif(
    not (EXPANSION_RELEASE.is_file() and RIVER_MAP.is_file()),
    reason="requires private river site bundle and the expansion release",
)
def test_real_river_map_pairs_with_the_expansion_release(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"test-video")
    profile_path, _map, _video = MODULE.select_inputs(
        ROOT, map_path=RIVER_MAP, video_path=video_path, mode=MODULE.MODE_EXPAND
    )
    assert profile_path == EXPANSION_RELEASE.resolve()
