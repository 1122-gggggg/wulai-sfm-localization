#!/usr/bin/env python3
"""Generate or verify manifests for the current portable package layout."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

CONTROL_FILES = {"MANIFEST.tsv", "SHA256SUMS"}
EXCLUDED_PARTS = {
    ".git",
    ".codegraph",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "env",
    "__pycache__",
    "artifacts",
    "封存",
    "EDM工具包",
    "inductor_cache",
    "outputs",
    "package_git",
    ".idea",
    ".vscode",
    ".fleet",
    "node_modules",
    ".cursor",
    "audit",
}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp", ".swo", ".tmp", ".bak", ".orig", ".rej"}
EXCLUDED_PREFIXES = (
    "地圖檔/",
    "模擬器/測試影片/",
    "定位演算法/validation/report_",
    "定位演算法/validation/source_videos/",
)


@dataclass(frozen=True)
class Entry:
    size: int
    sha256: str
    path: str


def included(relative: Path) -> bool:
    value = relative.as_posix()
    return (
        value not in CONTROL_FILES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and relative.name not in EXCLUDED_NAMES
        and not relative.name.endswith("~")
        and relative.suffix not in EXCLUDED_SUFFIXES
        and not any(value.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    )


def package_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not included(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"portable package does not permit symlinks: {relative}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def entries(root: Path) -> list[Entry]:
    return [
        Entry(path.stat().st_size, digest(path), path.relative_to(root).as_posix())
        for path in package_files(root)
    ]


def generate(root: str | Path) -> list[Entry]:
    root = Path(root).resolve()
    result = entries(root)
    manifest = ["size_bytes\tsha256\tpath"]
    manifest.extend(f"{entry.size}\t{entry.sha256}\t./{entry.path}" for entry in result)
    (root / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    sums = [f"{entry.sha256}  ./{entry.path}" for entry in result]
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return result


def read_manifest(root: Path) -> list[Entry]:
    lines = (root / "MANIFEST.tsv").read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "size_bytes\tsha256\tpath":
        raise ValueError("MANIFEST.tsv has an unsupported header")
    result: list[Entry] = []
    for line in lines[1:]:
        size, sha256, path = line.split("\t", 2)
        if not path.startswith("./"):
            raise ValueError(f"manifest path is not relative: {path}")
        result.append(Entry(int(size), sha256, path[2:]))
    if result != sorted(result, key=lambda entry: entry.path):
        raise ValueError("MANIFEST.tsv is not sorted")
    if len({entry.path for entry in result}) != len(result):
        raise ValueError("MANIFEST.tsv contains duplicate paths")
    return result


def verify(root: str | Path) -> list[str]:
    root = Path(root).resolve()
    try:
        expected = read_manifest(root)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    issues: list[str] = []
    expected_by_path = {entry.path: entry for entry in expected}
    try:
        actual_paths = {
            path.relative_to(root).as_posix(): path for path in package_files(root)
        }
    except ValueError as exc:
        return [str(exc)]

    for missing in sorted(set(expected_by_path) - set(actual_paths)):
        issues.append(f"missing: {missing}")
    for extra in sorted(set(actual_paths) - set(expected_by_path)):
        issues.append(f"unexpected: {extra}")
    for name in sorted(set(expected_by_path) & set(actual_paths)):
        expected_entry = expected_by_path[name]
        path = actual_paths[name]
        if path.stat().st_size != expected_entry.size:
            issues.append(
                f"size mismatch: {name} expected={expected_entry.size} "
                f"actual={path.stat().st_size}"
            )
            continue
        actual_digest = digest(path)
        if actual_digest != expected_entry.sha256:
            issues.append(
                f"SHA-256 mismatch: {name} expected={expected_entry.sha256} "
                f"actual={actual_digest}"
            )

    expected_sums = "".join(f"{entry.sha256}  ./{entry.path}\n" for entry in expected)
    try:
        if (root / "SHA256SUMS").read_text(encoding="utf-8") != expected_sums:
            issues.append("SHA256SUMS does not match MANIFEST.tsv")
    except OSError as exc:
        issues.append(str(exc))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args.root)
        print(f"generated MANIFEST.tsv and SHA256SUMS for {len(result)} files")
        return 0
    issues = verify(args.root)
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("portable package manifest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
