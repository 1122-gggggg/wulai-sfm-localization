from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "控制介面程式"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import mission_manifest  # noqa: E402
from mission_manifest import clear_sha256_cache, sha256_file  # noqa: E402

# Captured before any test patches the module attribute, so expected digests
# are not counted as file reads.
_REAL_SHA256 = hashlib.sha256


@pytest.fixture(autouse=True)
def _empty_cache():
    clear_sha256_cache()
    yield
    clear_sha256_cache()


@pytest.fixture
def reads(monkeypatch):
    """Count how many times a file is actually streamed through SHA-256."""
    calls: list[None] = []

    def counting(*args, **kwargs):
        calls.append(None)
        return _REAL_SHA256(*args, **kwargs)

    monkeypatch.setattr(mission_manifest.hashlib, "sha256", counting)
    return calls


def _digest(data: bytes) -> str:
    return _REAL_SHA256(data).hexdigest()


def test_digest_matches_and_unchanged_file_is_hashed_once(tmp_path: Path, reads) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"alpha")

    assert sha256_file(path) == _digest(b"alpha")
    assert sha256_file(str(path)) == _digest(b"alpha")
    assert len(reads) == 1


def test_modified_file_is_hashed_again(tmp_path: Path, reads) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"alpha")
    before = path.stat()
    assert sha256_file(path) == _digest(b"alpha")

    # Same size, new mtime: the stale digest must not be served.
    path.write_bytes(b"omega")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
    assert sha256_file(path) == _digest(b"omega")

    # A different size is detected as well.
    path.write_bytes(b"omega-longer")
    assert sha256_file(path) == _digest(b"omega-longer")
    assert len(reads) == 3


def test_identical_copy_at_another_path_is_hashed_on_its_own(
    tmp_path: Path, reads
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    copy = tmp_path / "copy.bin"
    shutil.copy2(source, copy)  # same size and same mtime_ns, different path

    assert sha256_file(source) == sha256_file(copy) == _digest(b"payload")
    assert len(reads) == 2


def test_missing_file_raises_instead_of_returning_a_digest(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        sha256_file(tmp_path / "absent.bin")
