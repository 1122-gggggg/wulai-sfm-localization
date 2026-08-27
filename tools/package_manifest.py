#!/usr/bin/env python3
"""Generate or verify source-only and portable-package manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CONTROL_FILES = {"MANIFEST.tsv", "SHA256SUMS"}
SITE_ASSET_MANIFEST = "PORTABLE_SITE_ASSETS.json"
OFFLINE_WHEELHOUSE_RELATIVE = "執行環境/offline_wheelhouse"
OFFLINE_REQUIREMENTS_RELATIVE = "requirements"
_SITE_PROFILE_TOP_LEVEL_ASSET_KEYS = (
    "localizer_profile",
    "map_reference_poses",
    "map_align",
)
_SITE_PROFILE_HARDWARE_PATH_KEYS = ("receipt", "signature", "trust_store")
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
    "mission_snapshots",
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
SOURCE_EXTERNAL_PARTS = {"offline_wheelhouse", "torch_hub_cache"}
SOURCE_EXTERNAL_PREFIXES = ("定位演算法/deploy_code/runtime/EDM/weights/",)


@dataclass(frozen=True)
class Entry:
    size: int
    sha256: str
    path: str


def _contains_symlink(path: Path) -> bool:
    try:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            if current.is_symlink():
                return True
    except (OSError, RuntimeError):
        return True
    return False


def _control_file_issue(root: Path) -> str | None:
    for name in sorted(CONTROL_FILES):
        if _contains_symlink(root / name):
            return f"{name} or its parent path contains a symlink"
    return None


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
    root_path = Path(root)
    if issue := _control_file_issue(root_path):
        raise ValueError(issue)
    root = root_path.resolve()
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
    if source_only:
        return []
    return [*_runtime_artifact_issues(root), *_offline_wheelhouse_issues(root)]


def _offline_install_metadata_issues(
    root: Path,
    offline_install: dict[str, object],
    wheelhouse_metadata: Mapping[str, object],
) -> list[str]:
    from offline_wheelhouse import MANIFEST_NAME

    requirements = wheelhouse_metadata["requirements"]
    wheels = wheelhouse_metadata["wheels"]
    assert isinstance(requirements, list)
    assert isinstance(wheels, list)
    expected = {
        "complete": True,
        "mode": "no-index",
        "wheelhouse": OFFLINE_WHEELHOUSE_RELATIVE,
        "manifest": f"{OFFLINE_WHEELHOUSE_RELATIVE}/{MANIFEST_NAME}",
        "manifest_sha256": digest(root / OFFLINE_WHEELHOUSE_RELATIVE / MANIFEST_NAME),
        "lock_digests": {
            str(entry["name"]): str(entry["sha256"])
            for entry in requirements
        },
        "wheel_count": len(wheels),
    }
    return (
        []
        if offline_install == expected
        else ["PORTABLE_PACKAGE.json offline_install does not match WHEELHOUSE.json"]
    )


def _offline_lock_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, dict) or not value:
        raise ValueError("PORTABLE_PACKAGE.json offline_install.lock_digests is invalid")
    names = tuple(sorted(value))
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("PORTABLE_PACKAGE.json offline install lock name is invalid")
    return names


def _offline_wheelhouse_issues(root: Path) -> list[str]:
    package_metadata = root / "PORTABLE_PACKAGE.json"
    if not package_metadata.is_file():
        return []
    try:
        data = json.loads(package_metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"PORTABLE_PACKAGE.json cannot be read: {exc}"]
    if not isinstance(data, dict):
        return ["PORTABLE_PACKAGE.json must contain an object"]
    offline_install = data.get("offline_install")
    if offline_install is None:
        return []
    if not isinstance(offline_install, dict):
        return ["PORTABLE_PACKAGE.json offline_install must be an object"]
    complete = offline_install.get("complete")
    if complete is False:
        return []
    if complete is not True:
        return ["PORTABLE_PACKAGE.json offline_install.complete must be a boolean"]

    try:
        lock_names = _offline_lock_names(offline_install.get("lock_digests"))
    except ValueError as exc:
        return [str(exc)]

    try:
        from offline_wheelhouse import WheelhouseError, verify_wheelhouse

        metadata = verify_wheelhouse(
            root / OFFLINE_WHEELHOUSE_RELATIVE,
            tuple(root / OFFLINE_REQUIREMENTS_RELATIVE / name for name in lock_names),
        )
        return _offline_install_metadata_issues(root, offline_install, metadata)
    except (OSError, WheelhouseError) as exc:
        return [f"offline wheelhouse is invalid: {exc}"]


def _site_asset_path_issue(
    root: Path, relative: object, *, label: str
) -> tuple[str | None, Path | None]:
    if not isinstance(relative, str) or not relative.strip():
        return f"{SITE_ASSET_MANIFEST} {label} path is invalid: {relative!r}", None
    if "\\" in relative:
        return f"{SITE_ASSET_MANIFEST} {label} path is unsafe: {relative}", None
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return f"{SITE_ASSET_MANIFEST} {label} path is not relative: {relative}", None
    path = root / relative_path
    if _contains_symlink(path):
        return f"{SITE_ASSET_MANIFEST} {label} path contains a symlink: {relative}", None
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        return f"{SITE_ASSET_MANIFEST} {label} path cannot be resolved: {relative}: {exc}", None
    if not resolved.is_relative_to(root):
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
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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


def _site_profile_asset_path(
    root: Path,
    profile_path: Path,
    relative: object,
    *,
    label: str,
) -> tuple[str | None, str | None, Path | None]:
    if not isinstance(relative, str) or not relative.strip():
        return (
            f"{SITE_ASSET_MANIFEST} {label} path is invalid: {relative!r}",
            None,
            None,
        )
    if "\\" in relative:
        return (
            f"{SITE_ASSET_MANIFEST} {label} path is unsafe: {relative}",
            None,
            None,
        )
    try:
        candidate = Path(relative).expanduser()
        if not candidate.is_absolute():
            candidate = profile_path.parent / candidate
        if _contains_symlink(candidate):
            return (
                f"{SITE_ASSET_MANIFEST} {label} path contains a symlink: {relative}",
                None,
                None,
            )
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        return (
            f"{SITE_ASSET_MANIFEST} {label} path cannot be resolved: {relative}: {exc}",
            None,
            None,
        )
    if not resolved.is_relative_to(root):
        return (
            f"{SITE_ASSET_MANIFEST} {label} path escapes package root: {relative}",
            None,
            None,
        )
    return None, resolved.relative_to(root).as_posix(), resolved


def _site_profile_files_issues(
    root: Path,
    profile_relative: str,
    raw_profile: dict[str, object],
    entries_by_path: dict[str, dict[str, object]],
) -> tuple[list[str], set[str]]:
    issues: list[str] = []
    listed_files = raw_profile.get("files")
    listed_paths: set[str] = set()
    if not isinstance(listed_files, list) or not listed_files:
        issues.append(f"{SITE_ASSET_MANIFEST} profile has no files list: {profile_relative}")
    else:
        for listed in listed_files:
            listed_issue, listed_path = _site_asset_path_issue(
                root, listed, label="profile file"
            )
            if listed_issue is not None or listed_path is None:
                issues.append(listed_issue or f"{SITE_ASSET_MANIFEST} profile file is invalid")
                continue
            assert isinstance(listed, str)
            listed_paths.add(listed)
            if listed not in entries_by_path:
                issues.append(f"{SITE_ASSET_MANIFEST} profile file is not bundled: {listed}")
            elif not listed_path.is_file():
                issues.append(f"{SITE_ASSET_MANIFEST} profile file missing: {listed}")
    if profile_relative not in listed_paths:
        issues.append(f"{SITE_ASSET_MANIFEST} profile does not list itself: {profile_relative}")
    return issues, listed_paths


def _read_site_profile(
    profile_path: Path, profile_relative: str
) -> tuple[list[str], dict[str, object] | None]:
    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{SITE_ASSET_MANIFEST} profile cannot be read: {profile_relative}: {exc}"], None
    if not isinstance(profile_data, dict):
        return [f"{SITE_ASSET_MANIFEST} profile must contain an object: {profile_relative}"], None
    return [], profile_data


def _site_profile_declared_assets(
    profile_data: dict[str, object], profile_relative: str
) -> tuple[list[str], dict[str, object]]:
    issues: list[str] = []
    declared_assets: dict[str, object] = {}
    profile_assets = profile_data.get("assets", {})
    if not isinstance(profile_assets, dict):
        issues.append(
            f"{SITE_ASSET_MANIFEST} profile assets must be an object: {profile_relative}"
        )
    else:
        declared_assets.update(profile_assets)
    for key in _SITE_PROFILE_TOP_LEVEL_ASSET_KEYS:
        if key in profile_data:
            declared_assets[key] = profile_data[key]

    hardware = profile_data.get("hardware_approval")
    if hardware is not None:
        if not isinstance(hardware, dict):
            issues.append(
                f"{SITE_ASSET_MANIFEST} profile hardware_approval must be an object: "
                f"{profile_relative}"
            )
        else:
            for key in _SITE_PROFILE_HARDWARE_PATH_KEYS:
                if key in hardware:
                    declared_assets[f"hardware_approval.{key}"] = hardware[key]
    return issues, declared_assets


def _site_profile_asset_issues(
    root: Path,
    profile_path: Path,
    profile_relative: str,
    declared_assets: dict[str, object],
    listed_paths: set[str],
    entries_by_path: dict[str, dict[str, object]],
    checked_reference_indexes: set[str],
) -> list[str]:
    issues: list[str] = []
    for label, value in declared_assets.items():
        if value in (None, ""):
            continue
        asset_issue, asset_relative, asset_path = _site_profile_asset_path(
            root,
            profile_path,
            value,
            label=f"profile {profile_relative} {label}",
        )
        if asset_issue is not None or asset_relative is None or asset_path is None:
            issues.append(asset_issue or f"{SITE_ASSET_MANIFEST} profile asset path is invalid")
            continue
        if asset_relative not in entries_by_path:
            issues.append(
                f"{SITE_ASSET_MANIFEST} profile {profile_relative} asset {label} "
                f"is not bundled: {asset_relative}"
            )
        elif not asset_path.is_file():
            issues.append(
                f"{SITE_ASSET_MANIFEST} profile {profile_relative} asset {label} "
                f"is missing: {asset_relative}"
            )
        if asset_relative not in listed_paths:
            issues.append(
                f"{SITE_ASSET_MANIFEST} profile {profile_relative} asset {label} "
                f"is not listed in profile files: {asset_relative}"
            )
        if label == "reference_index" and asset_relative not in checked_reference_indexes:
            issues.extend(_reference_index_issues(root, asset_relative, entries_by_path))
            checked_reference_indexes.add(asset_relative)
    return issues


def _site_profile_entry_issues(
    root: Path,
    raw_profile: object,
    seen_profiles: set[str],
    entries_by_path: dict[str, dict[str, object]],
    checked_reference_indexes: set[str],
) -> list[str]:
    if not isinstance(raw_profile, dict):
        return [f"{SITE_ASSET_MANIFEST} contains a non-object profile"]
    profile_relative = raw_profile.get("path")
    issue, profile_path = _site_asset_path_issue(root, profile_relative, label="profile")
    if issue is not None or profile_path is None:
        return [issue or f"{SITE_ASSET_MANIFEST} profile path is invalid"]
    assert isinstance(profile_relative, str)
    if profile_relative in seen_profiles:
        return [f"{SITE_ASSET_MANIFEST} contains duplicate profile: {profile_relative}"]
    seen_profiles.add(profile_relative)
    issues: list[str] = []
    profile_entry = entries_by_path.get(profile_relative)
    if profile_entry is None:
        issues.append(f"{SITE_ASSET_MANIFEST} profile is not bundled: {profile_relative}")
    elif profile_entry.get("role") != "site_profile":
        issues.append(
            f"{SITE_ASSET_MANIFEST} profile has an invalid file role: {profile_relative}"
        )
    if not profile_path.is_file():
        issues.append(f"{SITE_ASSET_MANIFEST} profile missing: {profile_relative}")
        return issues

    file_issues, listed_paths = _site_profile_files_issues(
        root, profile_relative, raw_profile, entries_by_path
    )
    issues.extend(file_issues)
    profile_issues, profile_data = _read_site_profile(profile_path, profile_relative)
    issues.extend(profile_issues)
    if profile_data is None:
        return issues
    declaration_issues, declared_assets = _site_profile_declared_assets(
        profile_data, profile_relative
    )
    issues.extend(declaration_issues)
    issues.extend(
        _site_profile_asset_issues(
            root,
            profile_path,
            profile_relative,
            declared_assets,
            listed_paths,
            entries_by_path,
            checked_reference_indexes,
        )
    )
    return issues


def _site_profile_issues(
    root: Path,
    data: dict[str, object],
    entries_by_path: dict[str, dict[str, object]],
    checked_reference_indexes: set[str],
) -> list[str]:
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        return [f"{SITE_ASSET_MANIFEST} has no profiles list"]
    issues: list[str] = []
    seen_profiles: set[str] = set()
    for raw_profile in profiles:
        issues.extend(
            _site_profile_entry_issues(
                root,
                raw_profile,
                seen_profiles,
                entries_by_path,
                checked_reference_indexes,
            )
        )
    return issues


def _site_asset_issues(root: Path) -> list[str]:
    manifest_path = root / SITE_ASSET_MANIFEST
    if not manifest_path.is_file():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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
    checked_reference_indexes: set[str] = set()
    for relative, raw in entries_by_path.items():
        if raw.get("role") == "reference_index_manifest":
            issues.extend(_reference_index_issues(root, relative, entries_by_path))
            checked_reference_indexes.add(relative)
    issues.extend(
        _site_profile_issues(root, data, entries_by_path, checked_reference_indexes)
    )
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
    issues.extend(
        f"unexpected: {name}"
        for name in sorted(set(actual_paths) - set(expected_by_path))
    )
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
    root_path = Path(root)
    if issue := _control_file_issue(root_path):
        return [issue]
    root = root_path.resolve()
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
