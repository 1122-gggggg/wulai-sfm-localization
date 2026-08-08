from __future__ import annotations

import json
from pathlib import Path

import pytest

from package_manifest import generate
from release_activation import activate, rollback, stage
from release_contract import release_verdict, validate_source_release


def test_dirty_release_requires_an_explicit_development_opt_out() -> None:
    assert release_verdict({"dirty": True}, allow_dirty=False) == [
        "git worktree is dirty; release/deployment validation is refused"
    ]
    assert release_verdict({"dirty": True}, allow_dirty=True) == []
    assert release_verdict({"dirty": False}, allow_dirty=False) == []


def test_source_release_binding_requires_commit_and_version() -> None:
    expected = {
        "manifest_sha256": "a" * 64,
        "sha256sums_sha256": "b" * 64,
        "commit": "deadbeef",
        "version": "release-deadbeef",
    }
    assert validate_source_release(expected, expected=expected) == []
    missing_commit = dict(expected)
    missing_commit.pop("commit")
    assert "source release commit is missing" in validate_source_release(
        missing_commit, expected=expected
    )


def test_activation_is_atomic_and_rollback_is_verifiable(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "schema": "sfm-portable-simulator/v2",
                "source_release": {
                    "commit": "deadbeef",
                    "version": "release-one",
                    "manifest_sha256": "a" * 64,
                    "sha256sums_sha256": "b" * 64,
                    "dirty": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "payload.txt").write_text("one", encoding="utf-8")
    generate(source)

    activation_root = tmp_path / "activation"
    first = stage(source, activation_root, version="release-one")
    activate(activation_root, first)
    assert (activation_root / "current" / "payload.txt").read_text() == "one"

    second_source = tmp_path / "package-two"
    second_source.mkdir()
    for path in source.iterdir():
        (second_source / path.name).write_bytes(path.read_bytes())
    (second_source / "payload.txt").write_text("two", encoding="utf-8")
    metadata = json.loads(
        (second_source / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8")
    )
    metadata["source_release"]["version"] = "release-two"
    (second_source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    generate(second_source)
    second = stage(second_source, activation_root, version="release-two")
    activate(activation_root, second)
    assert (activation_root / "current" / "payload.txt").read_text() == "two"

    rollback(activation_root)
    assert (activation_root / "current" / "payload.txt").read_text() == "one"


def test_activation_rejects_missing_release_metadata(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "payload.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="PORTABLE_PACKAGE.json"):
        stage(source, tmp_path / "activation", version="bad")


def test_activation_rejects_dirty_release_without_development_opt_out(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "source_release": {
                    "commit": "deadbeef",
                    "version": "release-dirty",
                    "manifest_sha256": "a" * 64,
                    "sha256sums_sha256": "b" * 64,
                    "dirty": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (source / "payload.txt").write_text("dirty", encoding="utf-8")
    generate(source)

    with pytest.raises(ValueError, match="dirty"):
        stage(source, tmp_path / "activation", version="release-dirty")

    assert stage(
        source, tmp_path / "activation-dev", version="release-dirty", allow_dirty=True
    ) == "release-dirty"
