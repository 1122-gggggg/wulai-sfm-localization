#!/usr/bin/env python3
"""Generate or verify the transfer package's size and SHA-256 manifests."""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path


CONTROL_FILES = {"MANIFEST.tsv", "SHA256SUMS"}
IGNORED_PARTS = {
    ".git",
    ".codegraph",
    ".cursor",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "venv",
}
IGNORED_PREFIXES = (
    "sfm_system/定位/outputs/",  # generated benchmark reports
    # Generated simulator/convergence reports. Do not ignore mission/outputs/:
    # it contains the tracked production route and safe-zone inputs.
    "sfm_system/定位/experiments/sphinx_anafi_path_convergence/outputs/",
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
        and not any(part in IGNORED_PARTS for part in relative.parts)
        and relative.suffix not in {".pyc", ".pyo"}
        and not value.startswith(IGNORED_PREFIXES)
    )


def package_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not included(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"package manifest does not permit symlinks: {relative.as_posix()}")
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
    result = []
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
    issues = []
    expected_by_path = {entry.path: entry for entry in expected}
    try:
        actual_paths = {path.relative_to(root).as_posix(): path for path in package_files(root)}
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
                f"size mismatch: {name} expected={expected_entry.size} actual={path.stat().st_size}"
            )
            continue
        actual_digest = digest(path)
        if actual_digest != expected_entry.sha256:
            issues.append(
                f"SHA-256 mismatch: {name} expected={expected_entry.sha256} actual={actual_digest}"
            )

    sums_path = root / "SHA256SUMS"
    expected_sums = "".join(f"{entry.sha256}  ./{entry.path}\n" for entry in expected)
    try:
        if sums_path.read_text(encoding="utf-8") != expected_sums:
            issues.append("SHA256SUMS does not match MANIFEST.tsv")
    except OSError as exc:
        issues.append(str(exc))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "verify"])
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
    print("package manifest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
