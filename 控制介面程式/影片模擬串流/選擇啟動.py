#!/usr/bin/env python3
"""Choose a validated site map/profile and replay video before opening the UI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


class SelectionError(RuntimeError):
    """A map/profile or video could not be selected safely."""


MODE_LOCALIZE = "localize"
MODE_EXPAND = "expand"
MODES = (MODE_LOCALIZE, MODE_EXPAND)
MODE_LABELS = {
    MODE_LOCALIZE: "純定位（主程式原版）",
    MODE_EXPAND: "定位＋擴圖（實驗版）",
}
MODE_NOTES = {
    MODE_LOCALIZE: "只用既有地圖定位，不新增地圖點。",
    MODE_EXPAND: (
        "離開舊圖範圍時建立暫時點與子圖，回到舊圖後縫合；"
        "實驗演算法，離線驗證尚未全面勝過純定位。"
    ),
}
#: Expansion site packages are the workmap-enabled releases under 地圖檔/場域/<site>/releases.
EXPANSION_SITES_DIR = "地圖檔/場域"


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


def profile_mode(profile) -> str:
    """Expansion iff the profile's SHA-bound direct runtime profile enables the workmap."""
    path = profile.localizer_profile
    if path is None:
        return MODE_LOCALIZE
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SelectionError(f"無法讀取定位 profile {path}: {exc}") from exc
    section = raw.get("online_workmap") if isinstance(raw, dict) else None
    enabled = isinstance(section, dict) and section.get("enabled") is True
    return MODE_EXPAND if enabled else MODE_LOCALIZE


def discover_expansion_profiles(root: Path) -> list[tuple[Path, object]]:
    base = root / EXPANSION_SITES_DIR
    paths = [
        *base.glob("*/site_profile.json"),
        *base.glob("*/releases/*/site_profile.json"),
    ]
    found: list[tuple[Path, object]] = []
    for path in sorted(paths):
        try:
            profile = load_profile(path)
            if profile_mode(profile) == MODE_EXPAND:
                found.append((path.resolve(), profile))
        except SelectionError:
            continue
    return found


def expansion_profile_for(root: Path, profile) -> tuple[Path, object]:
    """The expansion site package built on the same map PLY (same SHA-256)."""
    map_sha = profile.asset_sha256.map_ply
    matches = [
        (path, candidate)
        for path, candidate in discover_expansion_profiles(root)
        if map_sha is not None and candidate.asset_sha256.map_ply == map_sha
    ]
    if not matches:
        raise SelectionError(
            f"這張地圖沒有對應的定位＋擴圖場域包（需放在 {EXPANSION_SITES_DIR} 且地圖 PLY 相同）。"
        )
    if len(matches) > 1:
        names = ", ".join(str(path) for path, _candidate in matches)
        raise SelectionError(f"這張地圖對應多個定位＋擴圖場域包，請直接選 profile JSON: {names}")
    return matches[0]


def choose_mode(*, expand_blocked: str | None) -> str:
    """Ask which localization to run; the expansion choice is disabled when blocked."""
    result: dict[str, str | None] = {"mode": None}
    try:
        import tkinter as tk

        app = tk.Tk()
        app.title("選擇定位模式")
        app.attributes("-topmost", True)
        app.resizable(False, False)
        variable = tk.StringVar(value=MODE_LOCALIZE)
        tk.Label(app, text="這次要使用哪一種定位？", font=("Sans", 12, "bold")).pack(
            anchor="w", padx=14, pady=(12, 6)
        )
        for mode in MODES:
            tk.Radiobutton(
                app,
                text=MODE_LABELS[mode],
                variable=variable,
                value=mode,
                anchor="w",
                state="disabled" if mode == MODE_EXPAND and expand_blocked else "normal",
            ).pack(anchor="w", padx=14)
            tk.Label(
                app, text=MODE_NOTES[mode], wraplength=380, justify="left", fg="gray30"
            ).pack(anchor="w", padx=36, pady=(0, 6))
        if expand_blocked:
            tk.Label(
                app, text=expand_blocked, wraplength=380, justify="left", fg="red"
            ).pack(anchor="w", padx=14, pady=(0, 6))

        def confirm() -> None:
            result["mode"] = variable.get()
            app.destroy()

        row = tk.Frame(app)
        row.pack(fill="x", padx=14, pady=10)
        tk.Button(row, text="開始", command=confirm).pack(side="left")
        tk.Button(row, text="取消", command=app.destroy).pack(side="right")
        app.protocol("WM_DELETE_WINDOW", app.destroy)
        app.mainloop()
    except Exception as exc:
        raise SelectionError(f"無法開啟模式選擇視窗，請確認 tkinter/X11: {exc}") from exc
    if result["mode"] is None:
        raise SelectionError("已取消選擇。")
    return str(result["mode"])


def resolve_mode_profile(
    root: Path, profile_path: Path, profile, mode: str | None, *, explicit: bool
) -> tuple[Path, object]:
    """Return the site profile that runs the chosen localization mode.

    An explicitly named profile is never swapped: it fixes the mode, and a
    conflicting --mode is refused rather than silently ignored.
    """
    if mode is not None and mode not in MODES:
        raise SelectionError(f"未知的定位模式: {mode}")
    current = profile_mode(profile)
    if explicit or current == MODE_EXPAND:
        if mode is not None and mode != current:
            raise SelectionError(
                f"指定的 site profile 是「{MODE_LABELS[current]}」，與 --mode {mode} 衝突；"
                "要換模式請改選地圖 PLY。"
            )
        return profile_path, profile
    blocked = None
    expansion = None
    try:
        expansion = expansion_profile_for(root, profile)
    except SelectionError as exc:
        blocked = str(exc)
    if mode is None:
        mode = choose_mode(expand_blocked=blocked)
    if mode == MODE_EXPAND:
        if expansion is None:
            raise SelectionError(blocked or "沒有對應的定位＋擴圖場域包")
        return expansion
    return profile_path, profile


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
    mode: str | None = None,
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
    explicit = profile_path is not None or (
        map_path is not None and map_path.suffix.lower() == ".json"
    )
    selected_profile, profile = (
        (profile_path.expanduser().resolve(), load_profile(profile_path))
        if profile_path is not None
        else profile_for_map(root, map_path)
    )
    selected_profile, profile = resolve_mode_profile(
        root, selected_profile, profile, mode, explicit=explicit
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
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="localize=純定位, expand=定位＋擴圖；省略時用視窗選擇（明確指定 profile 時由 profile 決定）",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = workspace_root()
    try:
        profile_path, selected_map, video_path = select_inputs(
            root,
            map_path=args.map_path,
            profile_path=args.site_profile,
            video_path=args.video,
            mode=args.mode,
        )
        print(f"[選擇介面] mode={MODE_LABELS[profile_mode(load_profile(profile_path))]}")
        print(f"[選擇介面] site_profile={profile_path}")
        print(f"[選擇介面] map={selected_map}")
        print(f"[選擇介面] video={video_path}")
        return launch(root, profile_path, video_path, dry_run=args.dry_run)
    except SelectionError as exc:
        print(f"[選擇介面] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
