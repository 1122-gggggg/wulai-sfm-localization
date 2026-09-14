"""Filesystem anchors for the ``direct`` localization backend.

Two responsibilities, deliberately kept apart from every accuracy knob:

* ``vendor_sys_path`` puts the frozen ``river_map_quality`` / ``sfm_diagnosis``
  packages that ship inside this deploy directory on ``sys.path``.
* ``default_runtime_paths`` answers "where are the model weights on *this*
  machine".  Those three locations are machine facts, not tuning parameters, so
  they may be overridden through the environment.  Everything that can change a
  pose lives in ``direct_localizer_profile.json`` instead.

Modules in this deploy directory import each other flatly (``import
direct_paths``), matching the ``sfm_glomap_deploy`` convention: callers insert
the deploy directory itself onto ``sys.path``.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sys
from dataclasses import dataclass
from pathlib import Path


_SHA256_CHUNK = 8 * 1024 * 1024
_SHA256_CACHE: dict[tuple[int, int, int], str] = {}


def sha256_file(path: str | Path) -> str:
    """Stream one file through SHA-256 without holding it in memory."""
    resolved = Path(path).resolve()
    try:
        st = resolved.stat()
        key = (st.st_ino, int(st.st_mtime_ns), st.st_size)
    except OSError:
        key = None
    if key is not None and key in _SHA256_CACHE:
        return _SHA256_CACHE[key]

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_SHA256_CHUNK), b""):
            digest.update(chunk)
    val = digest.hexdigest()
    if key is not None:
        _SHA256_CACHE[key] = val
    return val


def verify_file_sha256(path: str | Path, expected: str | None) -> str:
    """Digest ``path``; when ``expected`` is given, refuse any mismatch."""

    actual = sha256_file(path)
    if expected is None:
        return actual
    trusted = str(expected).strip().lower()
    if len(trusted) != 64 or any(character not in "0123456789abcdef" for character in trusted):
        raise ValueError(f"expected SHA-256 for {path} is not a 64-character digest")
    if not hmac.compare_digest(actual, trusted):
        raise ValueError(f"SHA-256 mismatch for {path}: expected {trusted}, got {actual}")
    return actual


DEPLOY_DIR = Path(__file__).resolve().parent
VENDOR_DIR = DEPLOY_DIR / "vendor"
WORKSPACE_ROOT = DEPLOY_DIR.parents[2]

EDM_REPO_ENV = "SFM_EDM_REPO"
EDM_CHECKPOINT_ENV = "SFM_EDM_CHECKPOINT"
BOQ_WEIGHTS_ENV = "SFM_BOQ_WEIGHTS"

_DEFAULT_EDM_REPO = WORKSPACE_ROOT / "定位演算法" / "deploy_code" / "runtime" / "EDM"
_DEFAULT_EDM_CHECKPOINT = _DEFAULT_EDM_REPO / "weights" / "edm_outdoor.ckpt"
# Authoritative BoQ weights live under 執行環境/models/boq/ (RUNTIME_ARTIFACTS.json).
# Three-way exists chain: authoritative -> repo torch_hub_cache -> user cache.
_BOQ_WEIGHTS_CANDIDATES = (
    WORKSPACE_ROOT / "執行環境" / "models" / "boq" / "resnet50_16384.pth",
    WORKSPACE_ROOT / "執行環境" / "torch_hub_cache" / "checkpoints" / "boq" / "resnet50_16384.pth",
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50_16384.pth",
)
_DEFAULT_BOQ_WEIGHTS = _BOQ_WEIGHTS_CANDIDATES[0]
_DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "執行環境" / "direct_reloc_cache"


def resolve_boq_weights_path() -> Path:
    """Resolve BoQ weights via 3-way exists chain (authoritative -> repo cache -> home cache)."""
    for candidate in _BOQ_WEIGHTS_CANDIDATES:
        resolved = candidate.expanduser()
        if resolved.is_file():
            return resolved.resolve()
    return _DEFAULT_BOQ_WEIGHTS


def vendor_sys_path() -> None:
    """Expose the vendored map-quality/diagnosis packages exactly once."""

    entry = str(VENDOR_DIR)
    if not VENDOR_DIR.is_dir():
        raise FileNotFoundError(f"vendored packages are missing: {VENDOR_DIR}")
    if entry not in sys.path:
        sys.path.insert(0, entry)


@dataclass(frozen=True)
class DirectRuntimePaths:
    """Resolved, existing locations of the EDM and BoQ weights."""

    edm_repo: Path
    edm_checkpoint: Path
    boq_weights: Path


def _from_env(variable: str, *, directory: bool) -> Path | None:
    raw = os.environ.get(variable)
    if not raw:
        return None
    path = Path(raw).expanduser().resolve(strict=True)
    if directory and not path.is_dir():
        raise NotADirectoryError(f"{variable} must name a directory: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{variable} must name a file: {path}")
    return path


def _require(path: Path, *, directory: bool, what: str) -> Path:
    resolved = path.expanduser()
    if directory and not resolved.is_dir():
        raise NotADirectoryError(f"{what} directory is absent: {resolved}")
    if not directory and not resolved.is_file():
        raise FileNotFoundError(f"{what} file is absent: {resolved}")
    return resolved.resolve(strict=True)


def default_runtime_paths() -> DirectRuntimePaths:
    """Resolve the EDM/BoQ weight locations for this workstation."""

    edm_repo = _from_env(EDM_REPO_ENV, directory=True) or _require(
        _DEFAULT_EDM_REPO, directory=True, what="EDM repository"
    )
    edm_checkpoint = _from_env(EDM_CHECKPOINT_ENV, directory=False) or _require(
        _DEFAULT_EDM_CHECKPOINT, directory=False, what="EDM checkpoint"
    )
    boq_weights = _from_env(BOQ_WEIGHTS_ENV, directory=False) or _require(
        resolve_boq_weights_path(), directory=False, what="BoQ weights"
    )
    return DirectRuntimePaths(
        edm_repo=edm_repo,
        edm_checkpoint=edm_checkpoint,
        boq_weights=boq_weights,
    )


def default_cache_dir() -> Path:
    """Writable scratch directory for the direct backend's runtime artefacts."""

    return _DEFAULT_CACHE_DIR
