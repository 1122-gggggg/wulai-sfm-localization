from __future__ import annotations

import os
from pathlib import Path

import pytest
from package_manifest import included, package_files


def _reference_package_files(
    root: Path, *, source_only: bool = False, include_site_assets: bool = False
) -> list[Path]:
    """Full-tree ``rglob`` walk with post-hoc filtering (the pre-pruning algorithm)."""
    files: list[Path] = []
    nested_checkouts = [marker.parent for marker in root.rglob(".git") if marker.parent != root]
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(checkout in path.parents for checkout in nested_checkouts):
            continue
        if not included(relative, source_only=source_only, include_site_assets=include_site_assets):
            continue
        if path.is_symlink():
            raise ValueError(f"portable package does not permit symlinks: {relative}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relative, encoding="utf-8")


def _build_tree(root: Path) -> None:
    for relative in (
        "keep.py",
        "pkg/mod.py",
        "pkg/deep/data.json",
        ".gitignore",
        ".coveragerc",
        # excluded path parts, at the top level and nested
        ".venv/lib/site.py",
        ".venv-yolo/lib/site.py",
        "pkg/__pycache__/mod.pyc",
        "pkg/outputs/result.json",
        "執行環境/inductor_cache/kernel.py",
        "執行環境/mission_snapshots/snap.json",
        "模擬器/封存/old/src.py",
        # excluded prefixes
        "地圖檔/場域/river_site/map.ply",
        "模擬器/測試影片/clip.mp4",
        "定位演算法/validation/report_a/summary.json",
        "定位演算法/validation/source_videos/v.mp4",
        "定位演算法/validation/keep_me.py",
        "river-localization-bundle-20260831/bundle.pt",
        # excluded names / suffixes apply to the entry only, not to its children
        "notes.swp",
        "dir.tmp/inner.py",
        "backup~/inner.py",
        ".coverage",
        "coverage.xml",
        # source-only external prefixes, with the .gitignore exception
        "執行環境/models/boq/weights.bin",
        "執行環境/models/boq/.gitignore",
        # a nested checkout is a separate repository
        "nested_repo/.git/HEAD",
        "nested_repo/src/a.py",
        "nested_repo/deeper/b.py",
        "pkg/sub_worktree/.git",
        "pkg/sub_worktree/c.py",
        # control files are only special at the root
        "MANIFEST.tsv",
        "SHA256SUMS",
        "pkg/MANIFEST.tsv",
        # the root's own repository is not a nested checkout
        ".git/config",
    ):
        _touch(root, relative)
    # Links under excluded parts, nested checkouts or excluded names are skipped silently.
    os.symlink(root / "keep.py", root / "pkg" / "__pycache__" / "link.pyc")
    os.symlink(root / "pkg", root / ".venv" / "dirlink")
    os.symlink(root / "pkg", root / ".git" / "dirlink")
    os.symlink(root / "keep.py", root / "nested_repo" / "src" / "link.py")
    os.symlink(root / "keep.py", root / "dir.tmp" / "also.tmp")


@pytest.mark.parametrize("source_only", [False, True])
@pytest.mark.parametrize("include_site_assets", [False, True])
def test_pruned_walk_matches_full_tree_filter(
    tmp_path: Path, source_only: bool, include_site_assets: bool
) -> None:
    _build_tree(tmp_path)
    options = {"source_only": source_only, "include_site_assets": include_site_assets}

    result = package_files(tmp_path, **options)

    assert result == _reference_package_files(tmp_path, **options)
    relative = {path.relative_to(tmp_path).as_posix() for path in result}
    assert {"keep.py", "pkg/mod.py", ".coveragerc", "dir.tmp/inner.py"} <= relative
    assert "backup~/inner.py" in relative
    assert "定位演算法/validation/keep_me.py" in relative
    assert not any(value.startswith((".venv", ".git/")) for value in relative)
    assert not any(value.startswith("nested_repo/") for value in relative)
    assert "pkg/sub_worktree/c.py" not in relative
    assert ("地圖檔/場域/river_site/map.ply" in relative) is include_site_assets
    assert ("執行環境/models/boq/weights.bin" in relative) is not source_only
    assert "執行環境/models/boq/.gitignore" in relative
    assert "MANIFEST.tsv" not in relative and "pkg/MANIFEST.tsv" in relative


@pytest.mark.parametrize(
    "link",
    ["pkg/link.py", "pkg/linked_dir"],
)
def test_included_symlink_is_rejected_like_before(tmp_path: Path, link: str) -> None:
    _touch(tmp_path, "pkg/real.py")
    target = tmp_path / "pkg" / ("real.py" if link.endswith(".py") else ".")
    os.symlink(target, tmp_path / link)

    with pytest.raises(ValueError, match="does not permit symlinks"):
        package_files(tmp_path)
    with pytest.raises(ValueError, match="does not permit symlinks"):
        _reference_package_files(tmp_path)


def test_excluded_directory_is_not_entered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch(tmp_path, "keep.py")
    _touch(tmp_path, ".venv/lib/a.py")
    _touch(tmp_path, "地圖檔/場域/x.ply")
    entered: list[str] = []
    real_walk = os.walk

    def recording_walk(top, *args, **kwargs):
        for current, dirnames, filenames in real_walk(top, *args, **kwargs):
            entered.append(Path(current).relative_to(tmp_path).as_posix())
            yield current, dirnames, filenames

    monkeypatch.setattr(os, "walk", recording_walk)

    assert [path.name for path in package_files(tmp_path)] == ["keep.py"]
    assert entered == ["."]
