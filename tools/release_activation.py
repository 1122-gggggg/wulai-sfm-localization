#!/usr/bin/env python3
"""Stage, atomically activate, and roll back portable release directories.

The activation root is separate from the source workspace.  A release is copied
to a versioned directory, validated, and then selected by replacing a temporary
symlink with ``os.replace``.  ``current`` and ``previous`` therefore always
point at complete release directories; a failed copy never becomes active.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

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


def validate_release(path: str | Path, *, allow_dirty: bool = False) -> dict:
    release = Path(path).expanduser().resolve()
    metadata = _metadata(release)
    issues = verify(release)
    if issues:
        raise ValueError(f"release manifest is invalid: {issues[0]}")
    source = metadata["source_release"]
    binding_issues = validate_source_release(source, expected=source)
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
) -> str:
    """Copy and validate a complete release, returning its version directory."""
    if not version or version in {".", ".."} or "/" in version:
        raise ValueError("version must be a non-empty single path component")
    source_path = Path(source).expanduser().resolve()
    metadata = validate_release(source_path, allow_dirty=allow_dirty)
    declared_version = metadata["source_release"].get("version")
    if declared_version != version:
        raise ValueError(
            f"release version mismatch: metadata={declared_version!r} requested={version!r}"
        )
    root = Path(activation_root).expanduser().resolve()
    releases = root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    destination = releases / version
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"release version already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=releases))
    try:
        shutil.copytree(source_path, temporary, dirs_exist_ok=True)
        validate_release(temporary, allow_dirty=allow_dirty)
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


def activate(activation_root: str | Path, version: str, *, allow_dirty: bool = False) -> Path:
    """Atomically make a staged version current and retain the former current."""
    root = Path(activation_root).expanduser().resolve()
    target = root / "releases" / version
    validate_release(target, allow_dirty=allow_dirty)
    current = root / "current"
    previous = root / "previous"
    if current.is_symlink():
        _replace_link(previous, os.readlink(current))
    _replace_link(current, os.path.relpath(target, root))
    return target


def rollback(activation_root: str | Path, *, allow_dirty: bool = False) -> Path:
    """Atomically swap ``current`` and ``previous`` after validating both."""
    root = Path(activation_root).expanduser().resolve()
    current = root / "current"
    previous = root / "previous"
    if not current.is_symlink() or not previous.is_symlink():
        raise ValueError("rollback requires both current and previous releases")
    current_target = (root / os.readlink(current)).resolve()
    previous_target = (root / os.readlink(previous)).resolve()
    validate_release(previous_target, allow_dirty=allow_dirty)
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
    stage_parser.add_argument("--allow-dirty", "--development", action="store_true")
    activate_parser = subparsers.add_parser("activate")
    activate_parser.add_argument("activation_root", type=Path)
    activate_parser.add_argument("version")
    activate_parser.add_argument("--allow-dirty", "--development", action="store_true")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("activation_root", type=Path)
    rollback_parser.add_argument("--allow-dirty", "--development", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "stage":
        print(
            stage(
                args.source,
                args.activation_root,
                version=args.version,
                allow_dirty=args.allow_dirty,
            )
        )
    elif args.command == "activate":
        print(activate(args.activation_root, args.version, allow_dirty=args.allow_dirty))
    else:
        print(rollback(args.activation_root, allow_dirty=args.allow_dirty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
