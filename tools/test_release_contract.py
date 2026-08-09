from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from package_manifest import generate
from release_activation import activate, main, rollback, stage
from release_contract import release_verdict, validate_source_release


_OFFLINE_LOCKS = (
    "requirements-lock.txt",
    "requirements-test-lock.txt",
    "requirements-quality-lock.txt",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_offline_bundle(package: Path) -> None:
    for name in _OFFLINE_LOCKS:
        (package / name).write_text(f"{name}\n", encoding="utf-8")
    wheelhouse = package / "執行環境/offline_wheelhouse"
    wheelhouse.mkdir(parents=True)
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    manifest = wheelhouse / "WHEELHOUSE.json"
    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {"name": name, "sha256": _sha256(package / name)}
                    for name in sorted(_OFFLINE_LOCKS)
                ],
                "schema": "sfm-offline-wheelhouse/v1",
                "target": {
                    "implementation": "CPython",
                    "machine": "x86_64",
                    "platform": "linux",
                    "python": "3.10",
                },
                "wheels": [
                    {
                        "name": wheel.name,
                        "sha256": _sha256(wheel),
                        "size_bytes": wheel.stat().st_size,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = package / "PORTABLE_PACKAGE.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["offline_install"] = {
        "complete": True,
        "lock_digests": {
            name: _sha256(package / name) for name in sorted(_OFFLINE_LOCKS)
        },
        "manifest": "執行環境/offline_wheelhouse/WHEELHOUSE.json",
        "manifest_sha256": _sha256(manifest),
        "mode": "no-index",
        "wheel_count": 1,
        "wheelhouse": "執行環境/offline_wheelhouse",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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
    _add_offline_bundle(source)
    (source / "payload.txt").write_text("one", encoding="utf-8")
    generate(source)

    expected = json.loads((source / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8"))[
        "source_release"
    ]

    activation_root = tmp_path / "activation"
    first = stage(
        source,
        activation_root,
        version="release-one",
        expected_source_release=expected,
    )
    activate(activation_root, first, expected_source_release=expected)
    assert (activation_root / "current" / "payload.txt").read_text() == "one"

    second_source = tmp_path / "package-two"
    second_source.mkdir()
    for path in source.iterdir():
        target = second_source / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    (second_source / "payload.txt").write_text("two", encoding="utf-8")
    metadata = json.loads(
        (second_source / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8")
    )
    metadata["source_release"]["version"] = "release-two"
    (second_source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    generate(second_source)
    expected_two = metadata["source_release"]
    second = stage(
        second_source,
        activation_root,
        version="release-two",
        expected_source_release=expected_two,
    )
    activate(activation_root, second, expected_source_release=expected_two)
    assert (activation_root / "current" / "payload.txt").read_text() == "two"

    rollback(activation_root, expected_source_release=expected)
    assert (activation_root / "current" / "payload.txt").read_text() == "one"


def test_stage_rejects_symlinks_even_under_manifest_excluded_paths(
    tmp_path: Path,
) -> None:
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
    _add_offline_bundle(source)
    (source / "payload.txt").write_text("one", encoding="utf-8")
    generate(source)
    expected = json.loads(
        (source / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8")
    )["source_release"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (source / "outputs").mkdir()
    (source / "outputs/leak").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        stage(
            source,
            tmp_path / "activation",
            version="release-one",
            expected_source_release=expected,
        )

    assert not (tmp_path / "activation/releases/release-one").exists()


def test_activation_rejects_missing_release_metadata(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "payload.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="PORTABLE_PACKAGE.json"):
        stage(source, tmp_path / "activation", version="bad")


def test_activation_rejects_incomplete_offline_bundle(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    source_release = {
        "commit": "deadbeef",
        "version": "release-no-wheels",
        "manifest_sha256": "a" * 64,
        "sha256sums_sha256": "b" * 64,
        "dirty": False,
    }
    (source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "source_release": source_release,
                "offline_install": {"complete": False},
            }
        ),
        encoding="utf-8",
    )
    generate(source)

    with pytest.raises(ValueError, match="offline install bundle"):
        stage(
            source,
            tmp_path / "activation",
            version="release-no-wheels",
            expected_source_release=source_release,
        )


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
    _add_offline_bundle(source)
    (source / "payload.txt").write_text("dirty", encoding="utf-8")
    generate(source)

    expected = json.loads((source / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8"))[
        "source_release"
    ]

    with pytest.raises(ValueError, match="dirty"):
        stage(
            source,
            tmp_path / "activation",
            version="release-dirty",
            expected_source_release=expected,
        )

    assert (
        stage(
            source,
            tmp_path / "activation-dev",
            version="release-dirty",
            allow_dirty=True,
            expected_source_release=expected,
        )
        == "release-dirty"
    )


def test_activation_rejects_self_bound_metadata_without_trusted_expected(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "source_release": {
                    "commit": "attacker-controlled",
                    "version": "release-self-bound",
                    "manifest_sha256": "a" * 64,
                    "sha256sums_sha256": "b" * 64,
                    "dirty": False,
                }
            }
        ),
        encoding="utf-8",
    )
    _add_offline_bundle(source)
    (source / "payload.txt").write_text("untrusted", encoding="utf-8")
    generate(source)

    with pytest.raises(ValueError, match="trusted expected source release"):
        stage(source, tmp_path / "activation", version="release-self-bound")


@pytest.mark.parametrize(
    "version",
    ("../escape", "/absolute", "nested/release", "nested\\release", "\\absolute"),
)
def test_activation_rejects_path_like_versions(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="single path component"):
        stage(tmp_path / "missing-source", tmp_path / "activation", version=version)


def test_activation_rejects_releases_symlink_outside_root(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "source_release": {
                    "commit": "trusted-commit",
                    "version": "release-one",
                    "manifest_sha256": "a" * 64,
                    "sha256sums_sha256": "b" * 64,
                    "dirty": False,
                }
            }
        ),
        encoding="utf-8",
    )
    _add_offline_bundle(source)
    (source / "payload.txt").write_text("one", encoding="utf-8")
    generate(source)
    expected = json.loads((source / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8"))[
        "source_release"
    ]

    activation_root = tmp_path / "activation"
    activation_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (activation_root / "releases").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="activation root"):
        stage(
            source,
            activation_root,
            version="release-one",
            expected_source_release=expected,
        )


def test_activation_rejects_current_symlink_outside_root(tmp_path: Path) -> None:
    source = tmp_path / "package"
    source.mkdir()
    (source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(
            {
                "source_release": {
                    "commit": "trusted-commit",
                    "version": "release-one",
                    "manifest_sha256": "a" * 64,
                    "sha256sums_sha256": "b" * 64,
                    "dirty": False,
                }
            }
        ),
        encoding="utf-8",
    )
    _add_offline_bundle(source)
    (source / "payload.txt").write_text("one", encoding="utf-8")
    generate(source)
    expected = json.loads((source / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8"))[
        "source_release"
    ]

    activation_root = tmp_path / "activation"
    stage(
        source,
        activation_root,
        version="release-one",
        expected_source_release=expected,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (activation_root / "current").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="activation root"):
        activate(activation_root, "release-one", expected_source_release=expected)


def test_cli_accepts_a_trusted_expected_receipt(tmp_path: Path, capsys) -> None:
    source = tmp_path / "package"
    source.mkdir()
    expected = {
        "commit": "trusted-commit",
        "version": "release-one",
        "manifest_sha256": "a" * 64,
        "sha256sums_sha256": "b" * 64,
        "dirty": False,
    }
    (source / "PORTABLE_PACKAGE.json").write_text(
        json.dumps({"source_release": expected}), encoding="utf-8"
    )
    _add_offline_bundle(source)
    (source / "payload.txt").write_text("one", encoding="utf-8")
    generate(source)
    receipt = tmp_path / "trusted-receipt.json"
    receipt.write_text(json.dumps({"source_release": expected}), encoding="utf-8")
    activation_root = tmp_path / "activation"

    assert (
        main(
            [
                "stage",
                str(source),
                str(activation_root),
                "--version",
                "release-one",
                "--expected-source-release",
                str(receipt),
            ]
        )
        == 0
    )
    assert (activation_root / "releases" / "release-one" / "payload.txt").read_text() == "one"
    capsys.readouterr()

    assert (
        main(
            [
                "activate",
                str(activation_root),
                "release-one",
                "--expected-source-release",
                str(receipt),
            ]
        )
        == 0
    )
    assert (activation_root / "current" / "payload.txt").read_text() == "one"
