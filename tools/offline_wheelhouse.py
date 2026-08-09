#!/usr/bin/env python3
"""Build and verify a hash-locked, target-specific offline wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, TypedDict, cast

MANIFEST_NAME = "WHEELHOUSE.json"
SCHEMA = "sfm-offline-wheelhouse/v1"
TARGET = {
    "implementation": "CPython",
    "python": "3.10",
    "platform": "linux",
    "machine": "x86_64",
}


class RequirementRecord(TypedDict):
    name: str
    sha256: str


class WheelRecord(TypedDict):
    name: str
    size_bytes: int
    sha256: str


class WheelhouseManifest(TypedDict):
    schema: str
    target: dict[str, str]
    requirements: list[RequirementRecord]
    wheels: list[WheelRecord]


class WheelhouseError(ValueError):
    """Raised when a wheelhouse cannot be trusted or built."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise WheelhouseError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _lock_paths(requirement_locks: Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    if isinstance(requirement_locks, (str, os.PathLike)):
        requirement_locks = (requirement_locks,)
    try:
        paths = tuple(Path(lock) for lock in requirement_locks)
    except (TypeError, ValueError) as exc:
        raise WheelhouseError("requirement_locks must contain filesystem paths") from exc
    if not paths:
        raise WheelhouseError("at least one requirement lock is required")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise WheelhouseError("requirement lock basenames must be unique")
    return tuple(sorted(paths, key=lambda path: path.name))


def _requirement_records(requirement_locks: tuple[Path, ...]) -> list[RequirementRecord]:
    records: list[RequirementRecord] = []
    for lock in requirement_locks:
        if lock.is_symlink() or not lock.is_file():
            raise WheelhouseError(f"requirement lock is not a regular file: {lock}")
        name = lock.name
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise WheelhouseError(f"invalid requirement lock basename: {name!r}")
        records.append({"name": name, "sha256": _sha256(lock)})
    return records


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_wheel_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and value.endswith(".whl")
    )


def _actual_wheels(root: Path) -> dict[str, Path]:
    wheels: dict[str, Path] = {}
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise WheelhouseError(f"cannot inspect wheelhouse {root}: {exc}") from exc
    for path in children:
        if path.name == MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise WheelhouseError(f"wheelhouse contains a symlink: {path.name}")
        if not path.name.endswith(".whl") or not path.is_file():
            raise WheelhouseError(f"wheelhouse contains an unexpected entry: {path.name}")
        wheels[path.name] = path
    return wheels


def _valid_requirement_entry(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and set(entry) == {"name", "sha256"}
        and isinstance(entry["name"], str)
        and bool(entry["name"])
        and _valid_sha256(entry["sha256"])
    )


def _valid_wheel_entry(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and set(entry) == {"name", "size_bytes", "sha256"}
        and _valid_wheel_name(entry["name"])
        and isinstance(entry["size_bytes"], int)
        and not isinstance(entry["size_bytes"], bool)
        and entry["size_bytes"] > 0
        and _valid_sha256(entry["sha256"])
    )


def _validate_requirements(requirements: object, expected: list[RequirementRecord]) -> None:
    if not isinstance(requirements, list) or not all(
        _valid_requirement_entry(entry) for entry in requirements
    ):
        raise WheelhouseError(f"{MANIFEST_NAME} contains invalid requirement entries")
    if requirements != expected:
        raise WheelhouseError(f"{MANIFEST_NAME} requirements do not match the lock files")


def _validate_wheels(wheels: object) -> list[WheelRecord]:
    if (
        not isinstance(wheels, list)
        or not wheels
        or not all(_valid_wheel_entry(entry) for entry in wheels)
    ):
        raise WheelhouseError(f"{MANIFEST_NAME} contains invalid wheel entries")
    names = [entry["name"] for entry in wheels]
    if names != sorted(names):
        raise WheelhouseError(f"{MANIFEST_NAME} wheel entries are not sorted")
    if len(names) != len(set(names)):
        raise WheelhouseError(f"{MANIFEST_NAME} contains duplicate wheels")
    return cast(list[WheelRecord], wheels)


def _validate_manifest(
    data: object, expected_requirements: list[RequirementRecord]
) -> WheelhouseManifest:
    required_keys = {"schema", "target", "requirements", "wheels"}
    if not isinstance(data, dict) or set(data) != required_keys:
        raise WheelhouseError(f"{MANIFEST_NAME} has an invalid structure")
    if data["schema"] != SCHEMA:
        raise WheelhouseError(f"{MANIFEST_NAME} has an unsupported schema")
    if data["target"] != TARGET:
        raise WheelhouseError(f"{MANIFEST_NAME} has an unsupported target")
    _validate_requirements(data["requirements"], expected_requirements)
    _validate_wheels(data["wheels"])
    return cast(WheelhouseManifest, data)


def verify_wheelhouse(
    wheelhouse: str | os.PathLike[str],
    requirement_locks: Iterable[str | os.PathLike[str]],
) -> WheelhouseManifest:
    """Verify a wheelhouse and return its validated manifest metadata."""
    root = Path(wheelhouse)
    if root.is_symlink() or not root.is_dir():
        raise WheelhouseError(f"wheelhouse root is not a non-symlink directory: {root}")

    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise WheelhouseError(f"wheelhouse is missing a regular {MANIFEST_NAME}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WheelhouseError(f"cannot read {MANIFEST_NAME}: {exc}") from exc

    locks = _lock_paths(requirement_locks)
    expected_requirements = _requirement_records(locks)
    manifest = _validate_manifest(data, expected_requirements)
    wheel_entries = manifest["wheels"]
    actual_wheels = _actual_wheels(root)
    expected_names = {entry["name"] for entry in wheel_entries}
    if set(actual_wheels) != expected_names:
        raise WheelhouseError(
            f"wheelhouse wheel set does not match {MANIFEST_NAME}: "
            f"expected={sorted(expected_names)} actual={sorted(actual_wheels)}"
        )

    for entry in wheel_entries:
        name = entry["name"]
        path = actual_wheels[name]
        if path.is_symlink():
            raise WheelhouseError(f"wheelhouse contains a symlink wheel: {name}")
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise WheelhouseError(f"cannot stat wheel {name}: {exc}") from exc
        if actual_size != entry["size_bytes"]:
            raise WheelhouseError(
                f"wheel size mismatch for {name}: "
                f"expected={entry['size_bytes']} actual={actual_size}"
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != entry["sha256"]:
            raise WheelhouseError(
                f"wheel SHA-256 mismatch for {name}: "
                f"expected={entry['sha256']} actual={actual_sha256}"
            )
    return manifest


def _offline_requirement_line(source: Path, line_number: int, line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line
    option = stripped.split(maxsplit=1)[0].split("=", 1)[0]
    if option in {"--index-url", "--extra-index-url"}:
        return None
    if stripped.startswith("-") and not stripped.startswith("--hash=sha256:"):
        raise WheelhouseError(f"unsupported offline requirement option at {source}:{line_number}")
    if "://" in stripped:
        raise WheelhouseError(f"direct URL is not permitted at {source}:{line_number}")
    return line


def write_offline_requirements(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> Path:
    """Write a network-free derivative after the original lock was verified."""
    source_path = Path(source)
    output_path = Path(output)
    if source_path.is_symlink() or not source_path.is_file():
        raise WheelhouseError(f"requirement lock is not a regular file: {source_path}")
    if output_path.exists() or output_path.is_symlink():
        raise WheelhouseError(f"offline requirement output already exists: {output_path}")
    if output_path.parent.is_symlink() or not output_path.parent.is_dir():
        raise WheelhouseError(
            f"offline requirement output parent is not a directory: {output_path.parent}"
        )
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError) as exc:
        raise WheelhouseError(f"cannot read requirement lock {source_path}: {exc}") from exc

    result = [
        prepared
        for line_number, line in enumerate(lines, start=1)
        if (prepared := _offline_requirement_line(source_path, line_number, line)) is not None
    ]
    try:
        output_path.write_text("".join(result), encoding="utf-8")
    except OSError as exc:
        raise WheelhouseError(f"cannot write offline requirement file: {exc}") from exc
    return output_path


def _check_build_host() -> None:
    version = sys.version_info
    if (
        getattr(sys.implementation, "name", "").lower() != "cpython"
        or tuple(version[:2]) != (3, 10)
        or sys.platform != "linux"
        or platform.system() != "Linux"
        or platform.machine().lower() != "x86_64"
    ):
        raise WheelhouseError("wheelhouse build requires CPython 3.10 on Linux x86_64")


def _check_python_executable(python_executable: str | os.PathLike[str]) -> None:
    configured = os.fspath(python_executable)
    candidate = shutil.which(configured) if not os.path.isabs(configured) else configured
    if not candidate:
        raise WheelhouseError(f"cannot resolve Python executable: {configured}")
    try:
        matches_current_process = os.path.samefile(candidate, sys.executable)
    except OSError as exc:
        raise WheelhouseError(f"cannot resolve Python executable: {configured}") from exc
    if not matches_current_process:
        raise WheelhouseError("python_executable must identify the current CPython 3.10 process")


def _wheel_entries(stage: Path) -> list[WheelRecord]:
    wheels = _actual_wheels(stage)
    entries: list[WheelRecord] = []
    for name in sorted(wheels):
        path = wheels[name]
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise WheelhouseError(f"cannot stat downloaded wheel {name}: {exc}") from exc
        if size_bytes <= 0:
            raise WheelhouseError(f"downloaded wheel is empty: {name}")
        entries.append({"name": name, "size_bytes": size_bytes, "sha256": _sha256(path)})
    return entries


def _download(lock: Path, stage: Path, python_executable: str | os.PathLike[str]) -> None:
    command = [
        str(python_executable),
        "-m",
        "pip",
        "download",
        "--require-hashes",
        "--only-binary=:all:",
        "--dest",
        str(stage),
        "--requirement",
        str(lock),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WheelhouseError(f"pip download failed for {lock}: {exc}") from exc


def build_wheelhouse(
    output: str | os.PathLike[str],
    requirement_locks: Iterable[str | os.PathLike[str]],
    python_executable: str | os.PathLike[str],
) -> WheelhouseManifest:
    """Download locked wheels into an atomically published wheelhouse."""
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise WheelhouseError(f"wheelhouse output already exists: {destination}")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise WheelhouseError(f"wheelhouse output parent is not a directory: {parent}")

    _check_build_host()
    locks = _lock_paths(requirement_locks)
    _check_python_executable(python_executable)
    requirements = _requirement_records(locks)
    stage: Path | None = None
    try:
        try:
            stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(parent)))
        except OSError as exc:
            raise WheelhouseError(f"cannot create wheelhouse staging directory: {exc}") from exc

        for lock in locks:
            _download(lock, stage, python_executable)
        wheels = _wheel_entries(stage)
        manifest: WheelhouseManifest = {
            "schema": SCHEMA,
            "target": dict(TARGET),
            "requirements": requirements,
            "wheels": wheels,
        }
        manifest_path = stage / MANIFEST_NAME
        if manifest_path.exists() or manifest_path.is_symlink():
            raise WheelhouseError(f"staging directory already contains {MANIFEST_NAME}")
        try:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise WheelhouseError(f"cannot write {MANIFEST_NAME}: {exc}") from exc
        verify_wheelhouse(stage, locks)

        if destination.exists() or destination.is_symlink():
            raise WheelhouseError(f"wheelhouse output appeared during build: {destination}")
        try:
            os.replace(stage, destination)
        except OSError as exc:
            raise WheelhouseError(f"cannot publish wheelhouse: {exc}") from exc
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="download and verify a wheelhouse")
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--requirements", action="append", default=[], type=Path)

    verify = commands.add_parser("verify", help="verify an existing wheelhouse")
    verify.add_argument("--wheelhouse", required=True, type=Path)
    verify.add_argument("--requirements", action="append", default=[], type=Path)

    prepare = commands.add_parser(
        "prepare-lock", help="remove only package-index declarations from a verified lock"
    )
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "build":
            metadata = build_wheelhouse(args.output, args.requirements, sys.executable)
        elif args.command == "verify":
            metadata = verify_wheelhouse(args.wheelhouse, args.requirements)
        else:
            output = write_offline_requirements(args.source, args.output)
            print(output)
            return 0
    except WheelhouseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
