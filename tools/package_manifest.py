#!/usr/bin/env python3
"""Generate or verify source-only and portable-package manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CONTROL_FILES = {"MANIFEST.tsv", "SHA256SUMS"}
SITE_ASSET_MANIFEST = "PORTABLE_SITE_ASSETS.json"
EXCLUDED_PARTS = {
    ".git",
    ".codegraph",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "htmlcov",
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
EXCLUDED_NAMES = {".coverage", "coverage.xml", ".DS_Store", "Thumbs.db", "desktop.ini"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".swp", ".swo", ".tmp", ".bak", ".orig", ".rej"}
EXCLUDED_PREFIXES = (
    "地圖檔/",
    "模擬器/測試影片/",
    "定位演算法/validation/report_",
    "定位演算法/validation/source_videos/",
)
SOURCE_EXTERNAL_PARTS = {"torch_hub_cache"}
SOURCE_EXTERNAL_PREFIXES = ("定位演算法/deploy_code/runtime/EDM/weights/",)


@dataclass(frozen=True)
class Entry:
    size: int
    sha256: str
    path: str


def included(
    relative: Path, *, source_only: bool = False, include_site_assets: bool = False
) -> bool:
    value = relative.as_posix()
    return (
        value not in CONTROL_FILES
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and not (source_only and any(part in SOURCE_EXTERNAL_PARTS for part in relative.parts))
        and relative.name not in EXCLUDED_NAMES
        and not relative.name.startswith(".coverage.")
        and not relative.name.endswith("~")
        and relative.suffix not in EXCLUDED_SUFFIXES
        and not (
            not include_site_assets
            and value.startswith("地圖檔/")
        )
        and not any(
            value.startswith(prefix)
            for prefix in EXCLUDED_PREFIXES
            if prefix != "地圖檔/"
        )
        and not (
            source_only
            and any(value.startswith(prefix) for prefix in SOURCE_EXTERNAL_PREFIXES)
            and relative.name != ".gitignore"
        )
    )


def package_files(
    root: Path, *, source_only: bool = False, include_site_assets: bool = False
) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not included(
            relative,
            source_only=source_only,
            include_site_assets=include_site_assets,
        ):
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


def entries(
    root: Path, *, source_only: bool = False, include_site_assets: bool = False
) -> list[Entry]:
    return [
        Entry(path.stat().st_size, digest(path), path.relative_to(root).as_posix())
        for path in package_files(
            root,
            source_only=source_only,
            include_site_assets=include_site_assets,
        )
    ]


def generate(
    root: str | Path,
    *,
    source_only: bool = False,
    include_site_assets: bool | None = None,
) -> list[Entry]:
    root = Path(root).resolve()
    if include_site_assets is None:
        include_site_assets = not source_only and (root / SITE_ASSET_MANIFEST).is_file()
    result = entries(
        root,
        source_only=source_only,
        include_site_assets=include_site_assets,
    )
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


def _runtime_artifact_entry_issue(root: Path, artifact: object) -> str | None:
    if not isinstance(artifact, dict):
        return "RUNTIME_ARTIFACTS.json contains a non-object artifact"
    relative = artifact.get("path")
    expected_size = artifact.get("size_bytes")
    expected_sha256 = artifact.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or not isinstance(expected_sha256, str)
    ):
        return f"invalid runtime artifact entry: {artifact!r}"
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return f"runtime artifact path is not relative: {relative}"
    path = root / relative
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        return f"runtime artifact path cannot be resolved: {relative}: {exc}"
    if path.is_symlink() or not resolved.is_relative_to(root):
        return f"runtime artifact path escapes package root: {relative}"
    if not path.is_file():
        return f"runtime artifact missing: {relative}"
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return (
            f"runtime artifact size mismatch: {relative} "
            f"expected={expected_size} actual={actual_size}"
        )
    actual_sha256 = digest(path)
    if actual_sha256 != expected_sha256:
        return (
            f"runtime artifact SHA-256 mismatch: {relative} "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return None


def _runtime_artifact_issues(root: Path) -> list[str]:
    manifest = root / "RUNTIME_ARTIFACTS.json"
    if not manifest.is_file():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"RUNTIME_ARTIFACTS.json cannot be read: {exc}"]
    if not isinstance(data, dict) or data.get("schema") != "sfm-runtime-artifacts/v1":
        return ["RUNTIME_ARTIFACTS.json has an unsupported schema"]
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return ["RUNTIME_ARTIFACTS.json has no artifacts list"]
    return [
        issue
        for artifact in artifacts
        if (issue := _runtime_artifact_entry_issue(root, artifact)) is not None
    ]


def _portable_runtime_issues(root: Path, source_only: bool) -> list[str]:
    return [] if source_only else _runtime_artifact_issues(root)


def _site_asset_path_issue(root: Path, relative: object, *, label: str) -> tuple[str | None, Path | None]:
    if not isinstance(relative, str) or not relative.strip():
        return f"{SITE_ASSET_MANIFEST} {label} path is invalid: {relative!r}", None
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return f"{SITE_ASSET_MANIFEST} {label} path is not relative: {relative}", None
    path = root / relative_path
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        return f"{SITE_ASSET_MANIFEST} {label} path cannot be resolved: {relative}: {exc}", None
    if path.is_symlink() or not resolved.is_relative_to(root):
        return f"{SITE_ASSET_MANIFEST} {label} path escapes package root: {relative}", None
    return None, path


def _site_asset_entry_issues(
    root: Path,
    raw: object,
    entries_by_path: dict[str, dict[str, object]],
) -> list[str]:
    if not isinstance(raw, dict):
        return [f"{SITE_ASSET_MANIFEST} contains a non-object file"]
    relative = raw.get("path")
    issue, path = _site_asset_path_issue(root, relative, label="asset")
    if issue is not None:
        return [issue]
    assert isinstance(relative, str) and path is not None
    if relative in entries_by_path:
        return [f"{SITE_ASSET_MANIFEST} contains duplicate path: {relative}"]
    entries_by_path[relative] = raw
    expected_size = raw.get("size_bytes")
    expected_sha256 = raw.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        return [f"{SITE_ASSET_MANIFEST} has invalid digest entry: {relative}"]
    if not path.is_file():
        return [f"{SITE_ASSET_MANIFEST} asset missing: {relative}"]
    if path.stat().st_size != expected_size:
        return [
            f"{SITE_ASSET_MANIFEST} size mismatch: {relative} "
            f"expected={expected_size} actual={path.stat().st_size}"
        ]
    actual_sha256 = digest(path)
    if actual_sha256 != expected_sha256:
        return [
            f"{SITE_ASSET_MANIFEST} SHA-256 mismatch: {relative} "
            f"expected={expected_sha256} actual={actual_sha256}"
        ]
    return []


def _reference_index_issues(
    root: Path,
    relative: str,
    entries_by_path: dict[str, dict[str, object]],
) -> list[str]:
    try:
        index = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"reference index manifest cannot be read: {relative}: {exc}"]
    if not isinstance(index, dict) or index.get("schema") != "localization-reference-index-sha256":
        return [f"reference index manifest has unsupported schema: {relative}"]
    index_files = index.get("files")
    if not isinstance(index_files, dict) or not index_files:
        return [f"reference index manifest has no files: {relative}"]
    issues: list[str] = []
    for sibling_name, expected_sha256 in index_files.items():
        if (
            not isinstance(sibling_name, str)
            or Path(sibling_name).name != sibling_name
            or "\\" in sibling_name
        ):
            issues.append(f"reference index sibling path is unsafe: {relative}/{sibling_name}")
            continue
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            issues.append(f"reference index sibling digest is invalid: {relative}/{sibling_name}")
            continue
        sibling_relative = (Path(relative).parent / sibling_name).as_posix()
        sibling_issue, sibling_path = _site_asset_path_issue(
            root, sibling_relative, label="reference index sibling"
        )
        if sibling_issue is not None or sibling_path is None:
            issues.append(sibling_issue or f"reference index sibling missing: {sibling_relative}")
            continue
        if sibling_relative not in entries_by_path:
            issues.append(f"reference index sibling is not bundled: {sibling_relative}")
            continue
        if not sibling_path.is_file() or digest(sibling_path) != expected_sha256:
            issues.append(f"reference index sibling SHA-256 mismatch: {sibling_relative}")
    return issues


def _site_asset_issues(root: Path) -> list[str]:
    manifest_path = root / SITE_ASSET_MANIFEST
    if not manifest_path.is_file():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{SITE_ASSET_MANIFEST} cannot be read: {exc}"]
    if not isinstance(data, dict) or data.get("schema") != "sfm-portable-site-assets/v1":
        return [f"{SITE_ASSET_MANIFEST} has an unsupported schema"]
    files = data.get("files")
    if not isinstance(files, list) or not files:
        return [f"{SITE_ASSET_MANIFEST} has no files list"]
    issues: list[str] = []
    entries_by_path: dict[str, dict[str, object]] = {}
    for raw in files:
        issues.extend(_site_asset_entry_issues(root, raw, entries_by_path))
    for relative, raw in entries_by_path.items():
        if raw.get("role") == "reference_index_manifest":
            issues.extend(_reference_index_issues(root, relative, entries_by_path))
    return issues


def _manifest_entries_issues(
    root: Path,
    expected: list[Entry],
    *,
    source_only: bool,
    include_site_assets: bool,
) -> list[str]:
    expected_by_path = {entry.path: entry for entry in expected}
    try:
        actual_paths = {
            path.relative_to(root).as_posix(): path
            for path in package_files(
                root,
                source_only=source_only,
                include_site_assets=include_site_assets,
            )
        }
    except ValueError as exc:
        return [str(exc)]
    issues = [f"missing: {name}" for name in sorted(set(expected_by_path) - set(actual_paths))]
    issues.extend(f"unexpected: {name}" for name in sorted(set(actual_paths) - set(expected_by_path)))
    for name in sorted(set(expected_by_path) & set(actual_paths)):
        expected_entry = expected_by_path[name]
        path = actual_paths[name]
        actual_size = path.stat().st_size
        if actual_size != expected_entry.size:
            issues.append(
                f"size mismatch: {name} expected={expected_entry.size} actual={actual_size}"
            )
            continue
        actual_sha256 = digest(path)
        if actual_sha256 != expected_entry.sha256:
            issues.append(
                f"SHA-256 mismatch: {name} expected={expected_entry.sha256} actual={actual_sha256}"
            )
    return issues


def _manifest_checksum_issue(root: Path, expected: list[Entry]) -> str | None:
    expected_sums = "".join(f"{entry.sha256}  ./{entry.path}\n" for entry in expected)
    try:
        matches = (root / "SHA256SUMS").read_text(encoding="utf-8") == expected_sums
    except OSError as exc:
        return str(exc)
    return None if matches else "SHA256SUMS does not match MANIFEST.tsv"


def verify(
    root: str | Path,
    *,
    source_only: bool = False,
    include_site_assets: bool | None = None,
) -> list[str]:
    root = Path(root).resolve()
    if include_site_assets is None:
        include_site_assets = not source_only and (root / SITE_ASSET_MANIFEST).is_file()
    try:
        expected = read_manifest(root)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    issues = _manifest_entries_issues(
        root,
        expected,
        source_only=source_only,
        include_site_assets=include_site_assets,
    )
    if checksum_issue := _manifest_checksum_issue(root, expected):
        issues.append(checksum_issue)
    issues.extend(_portable_runtime_issues(root, source_only))
    if not source_only:
        issues.extend(_site_asset_issues(root))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="scope the manifest to Git/source files and exclude external runtime artifacts",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source_only = args.source_only or (
        (root / "RUNTIME_ARTIFACTS.json").is_file()
        and not (root / "PORTABLE_PACKAGE.json").is_file()
    )
    if args.command == "generate":
        result = generate(args.root, source_only=source_only)
        print(f"generated MANIFEST.tsv and SHA256SUMS for {len(result)} files")
        return 0
    issues = verify(args.root, source_only=source_only)
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("portable package manifest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
