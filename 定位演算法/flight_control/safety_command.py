"""Secure path and atomic-write helpers for operator safety commands."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path


VALID_COMMANDS = frozenset({"auto", "hover", "manual", "land", "emergency"})


def default_safety_file(
    environ: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
) -> Path:
    """Return the owner-private safety command path for this user."""
    values = os.environ if environ is None else environ
    runtime = values.get("XDG_RUNTIME_DIR", "").strip()
    user_home = Path.home() if home is None else Path(home)
    base = Path(runtime).expanduser() if runtime else user_home / ".local" / "state"
    if not base.is_absolute():
        base = user_home / ".local" / "state"
    return base / "sfm_drone" / "safety.cmd"


def safety_file_from_environment(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve ``SFM_SAFETY_FILE`` without permitting relative paths."""
    values = os.environ if environ is None else environ
    configured = values.get("SFM_SAFETY_FILE", "").strip()
    if not configured:
        return default_safety_file(values)
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise RuntimeError("SFM_SAFETY_FILE must be an absolute path")
    return path


def validate_safety_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"safety directory does not exist: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise RuntimeError(f"safety directory must be an owner-only directory: {path}")


def validate_safety_file_stat(path: Path, metadata: os.stat_result) -> None:
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o022
        or not metadata.st_mode & 0o200
    ):
        raise RuntimeError(
            f"safety command must be an owner-owned file without group/other write: {path}"
        )


def prepare_safety_file(path: Path) -> os.stat_result:
    """Create a fail-closed safety file if absent and validate any existing file."""
    validate_safety_directory(path.parent, create=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        pass
    else:
        try:
            os.write(descriptor, b"hover\n")
        finally:
            os.close(descriptor)
    metadata = path.lstat()
    validate_safety_file_stat(path, metadata)
    return metadata


def write_safety_command(path: str | Path, command: str) -> None:
    """Atomically publish one validated command in an owner-private directory."""
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        raise RuntimeError("safety command path must be absolute")
    token = command.strip().lower()
    if token not in VALID_COMMANDS:
        raise ValueError(f"unsupported safety command: {command!r}")

    validate_safety_directory(destination.parent, create=True)
    if destination.exists() or destination.is_symlink():
        validate_safety_file_stat(destination, destination.lstat())

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            os.chmod(stream.name, 0o600)
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        validate_safety_file_stat(destination, destination.lstat())
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
