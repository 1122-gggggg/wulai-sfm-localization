#!/usr/bin/env python3
"""Choose a validated site map/profile and replay video before opening the UI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


class SelectionError(RuntimeError):
    """A map/profile or video could not be selected safely."""


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_executable() -> Path:
    value = Path(os.environ.get("SFM_UI_PYTHON", sys.executable)).expanduser()
    return Path(os.path.abspath(value))


def load_profile(profile_path: Path):
    control_dir = (
        profile_path.parents[1]
        if profile_path.parent.name == "site_profiles"
        else workspace_root() / "控制介面程式"
    )
    if str(control_dir) not in sys.path:
        sys.path.insert(0, str(control_dir))
    from site_profile import load_site_profile

    try:
        return load_site_profile(profile_path)
    except ValueError as exc:
        raise SelectionError(str(exc)) from exc


def discover_profiles(root: Path) -> list[tuple[Path, object]]:
    profiles: list[tuple[Path, object]] = []
    profile_dir = root / "控制介面程式/site_profiles"
    paths = [
        *profile_dir.glob("*.json"),
        *(root / "地圖檔/場域").glob("*/site_profile.json"),
    ]
    for path in sorted(paths):
        try:
            profile = load_profile(path)
        except SelectionError:
            continue
        if profile.site_id == "your_site_edm":
            continue
        profiles.append((path.resolve(), profile))
    return profiles


def profile_for_map(root: Path, map_path: Path) -> tuple[Path, object]:
    selected = map_path.expanduser().resolve()
    if selected.suffix.lower() == ".json":
        profile = load_profile(selected)
        return selected, profile
    if selected.suffix.lower() != ".ply":
        raise SelectionError("請選擇地圖 PLY 或 site profile JSON。")
    matches = [
        (path, profile)
        for path, profile in discover_profiles(root)
        if profile.map_ply.resolve() == selected
    ]
    if not matches:
        raise SelectionError(
            "找不到與此 PLY 配對的有效 site profile。"
            " PLY 不能單獨定位，請把完整場域包與 profile 放入工作區。"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path, _profile in matches)
        raise SelectionError(
            f"一個 PLY 對應多個 profile，請直接選 profile JSON: {names}"
        )
    return matches[0]


def choose_file(
    root, *, title: str, initialdir: Path, filetypes: list[tuple[str, str]]
) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        app = tk.Tk()
        app.withdraw()
        app.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=str(initialdir if initialdir.is_dir() else root),
            filetypes=filetypes,
        )
        app.destroy()
    except Exception as exc:
        raise SelectionError(f"無法開啟檔案選擇器，請確認 tkinter/X11: {exc}") from exc
    if not selected:
        raise SelectionError("已取消選擇。")
    return Path(selected).expanduser().resolve()


def select_inputs(
    root: Path,
    *,
    map_path: Path | None = None,
    profile_path: Path | None = None,
    video_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    if profile_path is None and map_path is None:
        map_path = choose_file(
            root,
            title="選擇完整地圖 PLY 或 site profile JSON",
            initialdir=root / "地圖檔/場域",
            filetypes=[
                ("地圖 PLY / site profile", "*.ply *.json"),
                ("地圖 PLY", "*.ply"),
                ("site profile JSON", "*.json"),
            ],
        )
    selected_profile, profile = (
        (profile_path.expanduser().resolve(), load_profile(profile_path))
        if profile_path is not None
        else profile_for_map(root, map_path)
    )
    if video_path is None:
        video_path = choose_file(
            root,
            title=f"選擇 {profile.site_id} 的模擬影片",
            initialdir=root / "模擬器/測試影片",
            filetypes=[
                ("模擬影片", "*.mp4 *.mov *.mkv"),
                ("所有檔案", "*"),
            ],
        )
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise SelectionError(f"影片不存在或為空檔案: {video_path}")
    return selected_profile, profile.map_ply.resolve(), video_path


def launch(root: Path, profile_path: Path, video_path: Path, *, dry_run: bool) -> int:
    launcher = root / "控制介面程式/影片模擬串流/啟動.sh"
    python_bin = python_executable()
    env = os.environ.copy()
    env.update(
        {
            "SFM_WORKSPACE_ROOT": str(root.resolve()),
            "SFM_TORCH_HUB_CACHE": str(root / "執行環境/torch_hub_cache"),
            "SFM_UI_PYTHON": str(python_bin),
            "SFM_LOCALIZER_PYTHON": str(python_bin),
            "SFM_SITE_PROFILE": str(profile_path.resolve()),
            "VIDEO": str(video_path.resolve()),
        }
    )
    if dry_run:
        env["SFM_LAUNCH_DRY_RUN"] = "1"
    return subprocess.run([str(launcher)], cwd=root, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_path", type=Path)
    parser.add_argument("--site-profile", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = workspace_root()
    try:
        profile_path, selected_map, video_path = select_inputs(
            root,
            map_path=args.map_path,
            profile_path=args.site_profile,
            video_path=args.video,
        )
        print(f"[選擇介面] site_profile={profile_path}")
        print(f"[選擇介面] map={selected_map}")
        print(f"[選擇介面] video={video_path}")
        return launch(root, profile_path, video_path, dry_run=args.dry_run)
    except SelectionError as exc:
        print(f"[選擇介面] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
