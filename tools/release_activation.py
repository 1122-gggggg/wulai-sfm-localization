#!/usr/bin/env python3
"""Stage, atomically activate, and roll back portable release directories.

The activation root is separate from the source workspace.  A release is copied
to a versioned directory, validated, and then selected by replacing a temporary
symlink with ``os.replace``.  ``current`` and ``previous`` therefore always
point at complete release directories; a failed copy never becomes active.
Activation also requires an independently trusted expected source-release
binding; self-reported package metadata is not sufficient.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from package_manifest import verify
from release_contract import validate_source_release


def _metadata(path: Path) -> dict:
    metadata_path = path / "PORTABLE_PACKAGE.json"
    if not metadata_path.is_file():
        raise ValueError(f"release is missing PORTABLE_PACKAGE.json: {path}")
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PORTABLE_PACKAGE.json: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("source_release"), dict):
        raise ValueError("PORTABLE_PACKAGE.json has no source_release object")
    return value


def _expected_source_release(
    expected: Mapping[str, Any] | str | Path | None,
    *,
    release: Path,
) -> dict[str, Any]:
    if expected is None:
        raise ValueError("trusted expected source release is required")
    if isinstance(expected, Mapping):
        return dict(expected)

    expected_path = Path(expected).expanduser().resolve()
    if expected_path == release or expected_path.is_relative_to(release):
        raise ValueError("trusted expected source release must be external to the release")
    try:
        if expected_path.is_dir():
            value = _metadata(expected_path)
        else:
            value = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid trusted expected source release: {exc}") from exc

    if not isinstance(value, dict):
        raise ValueError("trusted expected source release must be a JSON object")
    source = value.get("source_release")
    if isinstance(source, dict):
        return source
    release_inputs = value.get("release_inputs")
    files = release_inputs.get("files") if isinstance(release_inputs, dict) else None
    git = value.get("git")
    if isinstance(files, dict) and isinstance(git, dict):
        manifest = files.get("MANIFEST.tsv")
        sha256sums = files.get("SHA256SUMS")
        if isinstance(manifest, dict) and isinstance(sha256sums, dict):
            return {
                "manifest_sha256": manifest.get("sha256"),
                "sha256sums_sha256": sha256sums.get("sha256"),
                "commit": git.get("commit"),
                "version": git.get("version"),
                "dirty": git.get("dirty"),
            }
    return value


def _resolve_within(root: Path, path: Path, *, label: str) -> Path:
    try:
        resolved_root = root.resolve()
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be resolved safely: {exc}") from exc
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} resolves outside activation root: {resolved_path}"
        ) from exc
    return resolved_path


def _validate_version(version: str) -> None:
    if (
        not isinstance(version, str)
        or not version
        or "\x00" in version
        or version in {".", ".."}
        or Path(version).is_absolute()
        or any(separator in version for separator in ("/", "\\"))
    ):
        raise ValueError("version must be a non-empty single path component")


def _link_target(root: Path, link: Path, *, label: str, required: bool = False) -> Path | None:
    if not link.is_symlink():
        if required:
            raise ValueError("rollback requires both current and previous releases")
        return None
    return _resolve_within(root, link.parent / os.readlink(link), label=label)


def validate_release(
    path: str | Path,
    *,
    allow_dirty: bool = False,
    expected_source_release: Mapping[str, Any] | str | Path | None = None,
) -> dict:
    release = Path(path).expanduser().resolve()
    metadata = _metadata(release)
    issues = verify(release)
    if issues:
        raise ValueError(f"release manifest is invalid: {issues[0]}")
    offline_install = metadata.get("offline_install")
    if (
        not isinstance(offline_install, dict)
        or offline_install.get("complete") is not True
    ):
        raise ValueError("release has no complete offline install bundle")
    source = metadata["source_release"]
    expected = _expected_source_release(expected_source_release, release=release)
    binding_issues = validate_source_release(source, expected=expected)
    if binding_issues:
        raise ValueError(binding_issues[0])
    if source.get("dirty") is not False and not allow_dirty:
        raise ValueError("release source is dirty; activation is refused")
    return metadata


def stage(
    source: str | Path,
    activation_root: str | Path,
    *,
    version: str,
    allow_dirty: bool = False,
    expected_source_release: Mapping[str, Any] | str | Path | None = None,
) -> str:
    """Copy and validate a complete release, returning its version directory."""
    _validate_version(version)
    source_path = Path(source).expanduser().resolve()
    metadata = validate_release(
        source_path,
        allow_dirty=allow_dirty,
        expected_source_release=expected_source_release,
    )
    declared_version = metadata["source_release"].get("version")
    if declared_version != version:
        raise ValueError(
            f"release version mismatch: metadata={declared_version!r} requested={version!r}"
        )
    root = Path(activation_root).expanduser().resolve()
    releases = _resolve_within(root, root / "releases", label="releases directory")
    releases.mkdir(parents=True, exist_ok=True)
    destination = _resolve_within(root, releases / version, label="release destination")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"release version already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=releases))
    try:
        shutil.copytree(source_path, temporary, dirs_exist_ok=True)
        validate_release(
            temporary,
            allow_dirty=allow_dirty,
            expected_source_release=expected_source_release,
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return version


def _replace_link(path: Path, target: str) -> None:
    temporary = path.with_name(f".{path.name}.new")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(target)
    os.replace(temporary, path)


def activate(
    activation_root: str | Path,
    version: str,
    *,
    allow_dirty: bool = False,
    expected_source_release: Mapping[str, Any] | str | Path | None = None,
) -> Path:
    """Atomically make a staged version current and retain the former current."""
    _validate_version(version)
    root = Path(activation_root).expanduser().resolve()
    releases = _resolve_within(root, root / "releases", label="releases directory")
    target = _resolve_within(root, releases / version, label="release target")
    validate_release(
        target,
        allow_dirty=allow_dirty,
        expected_source_release=expected_source_release,
    )
    current = root / "current"
    previous = root / "previous"
    _link_target(root, previous, label="previous symlink target")
    current_target = _link_target(root, current, label="current symlink target")
    if current_target is not None:
        _replace_link(previous, os.path.relpath(current_target, root))
    _replace_link(current, os.path.relpath(target, root))
    return target


def rollback(
    activation_root: str | Path,
    *,
    allow_dirty: bool = False,
    expected_source_release: Mapping[str, Any] | str | Path | None = None,
) -> Path:
    """Atomically swap ``current`` and ``previous`` after validating both."""
    root = Path(activation_root).expanduser().resolve()
    current = root / "current"
    previous = root / "previous"
    current_target = _link_target(
        root,
        current,
        label="current symlink target",
        required=True,
    )
    previous_target = _link_target(
        root,
        previous,
        label="previous symlink target",
        required=True,
    )
    validate_release(
        previous_target,
        allow_dirty=allow_dirty,
        expected_source_release=expected_source_release,
    )
    _replace_link(current, os.path.relpath(previous_target, root))
    _replace_link(previous, os.path.relpath(current_target, root))
    return previous_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("source", type=Path)
    stage_parser.add_argument("activation_root", type=Path)
    stage_parser.add_argument("--version", required=True)
    stage_parser.add_argument(
        "--expected-source-release",
        type=Path,
        help="trusted JSON source-release receipt (required)",
    )
    stage_parser.add_argument("--allow-dirty", "--development", action="store_true")
    activate_parser = subparsers.add_parser("activate")
    activate_parser.add_argument("activation_root", type=Path)
    activate_parser.add_argument("version")
    activate_parser.add_argument(
        "--expected-source-release",
        type=Path,
        help="trusted JSON source-release receipt (required)",
    )
    activate_parser.add_argument("--allow-dirty", "--development", action="store_true")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("activation_root", type=Path)
    rollback_parser.add_argument(
        "--expected-source-release",
        type=Path,
        help="trusted JSON source-release receipt (required)",
    )
    rollback_parser.add_argument("--allow-dirty", "--development", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "stage":
        print(
            stage(
                args.source,
                args.activation_root,
                version=args.version,
                allow_dirty=args.allow_dirty,
                expected_source_release=args.expected_source_release,
            )
        )
    elif args.command == "activate":
        print(
            activate(
                args.activation_root,
                args.version,
                allow_dirty=args.allow_dirty,
                expected_source_release=args.expected_source_release,
            )
        )
    else:
        print(
            rollback(
                args.activation_root,
                allow_dirty=args.allow_dirty,
                expected_source_release=args.expected_source_release,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
