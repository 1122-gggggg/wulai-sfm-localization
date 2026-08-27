#!/usr/bin/env python3
"""Shared release identity and worktree policy helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def git_identity(root: str | Path) -> dict[str, Any]:
    """Return the immutable source revision and whether the worktree is dirty."""
    workspace = Path(root).expanduser().resolve()
    status = _git(workspace, "status", "--porcelain").splitlines()
    commit = _git(workspace, "rev-parse", "HEAD")
    version = _git(workspace, "describe", "--tags", "--always", "--dirty")
    if not version:
        version = commit or "unknown"
    return {
        "commit": commit,
        "version": version,
        "dirty": bool(status),
        "changed_path_count": len(status),
    }


def source_release_identity(root: str | Path) -> dict[str, Any]:
    """Bind the generated package to the source commit and authoritative manifests."""
    workspace = Path(root).expanduser().resolve()
    identity = git_identity(workspace)
    return {
        "manifest_sha256": sha256_file(workspace / "MANIFEST.tsv"),
        "sha256sums_sha256": sha256_file(workspace / "SHA256SUMS"),
        "commit": identity["commit"],
        "version": identity["version"],
        "dirty": identity["dirty"],
    }


def validate_source_release(actual: dict[str, Any], *, expected: dict[str, Any]) -> list[str]:
    """Return release-binding failures without mutating either input."""
    issues: list[str] = []
    for key, label in (
        ("manifest_sha256", "source release manifest SHA-256"),
        ("sha256sums_sha256", "source release SHA256SUMS SHA-256"),
    ):
        value = actual.get(key)
        if not isinstance(value, str) or len(value) != 64:
            issues.append(f"{label} is missing or invalid")
        elif value != expected.get(key):
            issues.append(f"{label} mismatch: expected {expected.get(key)} actual {value}")
    for key, label in (("commit", "source release commit"), ("version", "source release version")):
        value = actual.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{label} is missing")
        elif value != expected.get(key):
            issues.append(f"{label} mismatch: expected {expected.get(key)} actual {value}")
    return issues


def release_verdict(git: dict[str, Any], *, allow_dirty: bool) -> list[str]:
    """Fail a release/deployment verdict for dirty source unless explicitly opted out."""
    if git.get("dirty") and not allow_dirty:
        return ["git worktree is dirty; release/deployment validation is refused"]
    return []
