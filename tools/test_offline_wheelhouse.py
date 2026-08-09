from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from tools import offline_wheelhouse

ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wheel_files: dict[str, bytes]
) -> tuple[Path, Path, dict[str, object], list[list[str]]]:
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text(
        "demo==1.0 lock\n",
        encoding="utf-8",
    )
    output = tmp_path / "wheelhouse"
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        destination = Path(command[command.index("--dest") + 1])
        for name, payload in wheel_files.items():
            (destination / name).write_bytes(payload)

    monkeypatch.setattr(offline_wheelhouse.subprocess, "run", fake_run)
    metadata = offline_wheelhouse.build_wheelhouse(output, [lock], sys.executable)
    return output, lock, metadata, calls


def _read_manifest(wheelhouse: Path) -> dict[str, object]:
    return json.loads((wheelhouse / offline_wheelhouse.MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(wheelhouse: Path, manifest: dict[str, object]) -> None:
    (wheelhouse / offline_wheelhouse.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_build_and_verify_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheelhouse, lock, metadata, calls = _build_fixture(
        tmp_path,
        monkeypatch,
        {"demo-1.0-py3-none-any.whl": b"wheel payload"},
    )

    assert metadata == offline_wheelhouse.verify_wheelhouse(wheelhouse, [lock])
    assert metadata["schema"] == "sfm-offline-wheelhouse/v1"
    assert metadata["target"] == offline_wheelhouse.TARGET
    assert metadata["requirements"] == [{"name": lock.name, "sha256": _digest(lock)}]
    assert "tmp" not in (wheelhouse / offline_wheelhouse.MANIFEST_NAME).read_text()
    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--require-hashes",
            "--only-binary=:all:",
            "--dest",
            calls[0][calls[0].index("--dest") + 1],
            "--requirement",
            str(lock),
        ]
    ]


def test_subset_wheelhouse_is_network_free_and_binds_only_selected_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime_lock, _, _ = _build_fixture(
        tmp_path,
        monkeypatch,
        {
            "runtime-1.0-py3-none-any.whl": b"runtime",
            "quality-1.0-py3-none-any.whl": b"quality",
        },
    )
    quality_lock = tmp_path / "requirements-quality-lock.txt"
    quality_lock.write_text("quality==1.0 lock\n", encoding="utf-8")
    manifest = _read_manifest(source)
    manifest["requirements"] = sorted(
        [
            *manifest["requirements"],
            {"name": quality_lock.name, "sha256": _digest(quality_lock)},
        ],
        key=lambda entry: entry["name"],
    )
    _write_manifest(source, manifest)
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        destination = Path(command[command.index("--dest") + 1])
        (destination / "runtime-1.0-py3-none-any.whl").write_bytes(b"runtime")

    monkeypatch.setattr(offline_wheelhouse.subprocess, "run", fake_run)
    output = tmp_path / "runtime-only"

    metadata = offline_wheelhouse.subset_wheelhouse(
        source,
        output,
        source_requirement_locks=(runtime_lock, quality_lock),
        requirement_locks=(runtime_lock,),
        python_executable=sys.executable,
    )

    assert metadata == offline_wheelhouse.verify_wheelhouse(output, (runtime_lock,))
    assert [entry["name"] for entry in metadata["requirements"]] == [runtime_lock.name]
    assert [entry["name"] for entry in metadata["wheels"]] == ["runtime-1.0-py3-none-any.whl"]
    command = calls[0]
    assert "--no-index" in command
    assert command[command.index("--find-links") + 1] == str(source.resolve())


def test_lock_digest_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheelhouse, lock, _, _ = _build_fixture(
        tmp_path, monkeypatch, {"demo-1.0-py3-none-any.whl": b"payload"}
    )
    lock.write_text(lock.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.verify_wheelhouse(wheelhouse, [lock])


@pytest.mark.parametrize("mutation", ["extra", "missing", "tampered"])
def test_wheel_set_and_hash_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    wheelhouse, lock, _, _ = _build_fixture(
        tmp_path, monkeypatch, {"demo-1.0-py3-none-any.whl": b"payload"}
    )
    wheel = wheelhouse / "demo-1.0-py3-none-any.whl"
    if mutation == "extra":
        (wheelhouse / "extra-1.0-py3-none-any.whl").write_bytes(b"extra")
    elif mutation == "missing":
        wheel.unlink()
    else:
        wheel.write_bytes(b"tampered")

    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.verify_wheelhouse(wheelhouse, [lock])


def test_traversal_duplicate_and_unsorted_entries_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheelhouse, lock, _, _ = _build_fixture(
        tmp_path,
        monkeypatch,
        {"a-1.0-py3-none-any.whl": b"a", "b-1.0-py3-none-any.whl": b"b"},
    )
    manifest = _read_manifest(wheelhouse)
    wheels = manifest["wheels"]
    assert isinstance(wheels, list)

    wheels[0]["name"] = "../escape.whl"
    _write_manifest(wheelhouse, manifest)
    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.verify_wheelhouse(wheelhouse, [lock])

    manifest = _read_manifest(wheelhouse)
    wheels = manifest["wheels"]
    assert isinstance(wheels, list)
    wheels[1] = wheels[0].copy()
    _write_manifest(wheelhouse, manifest)
    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.verify_wheelhouse(wheelhouse, [lock])

    manifest = _read_manifest(wheelhouse)
    wheels = manifest["wheels"]
    assert isinstance(wheels, list)
    wheels.reverse()
    _write_manifest(wheelhouse, manifest)
    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.verify_wheelhouse(wheelhouse, [lock])


def test_wheel_symlink_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheelhouse, lock, _, _ = _build_fixture(
        tmp_path, monkeypatch, {"demo-1.0-py3-none-any.whl": b"payload"}
    )
    target = wheelhouse / "demo-1.0-py3-none-any.whl"
    target.rename(wheelhouse / "real.whl")
    os.symlink("real.whl", target)

    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.verify_wheelhouse(wheelhouse, [lock])


def test_build_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "wheelhouse"
    output.mkdir()
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("lock\n", encoding="utf-8")
    monkeypatch.setattr(
        offline_wheelhouse.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("download must not run"),
    )

    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.build_wheelhouse(output, [lock], "/fake/python")


@pytest.mark.parametrize("locks", [[], ["same", "same"]])
def test_build_requires_nonempty_unique_lock_basenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    locks: list[str],
) -> None:
    paths: list[Path] = []
    for index, name in enumerate(locks):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / name
        path.write_text("lock\n", encoding="utf-8")
        paths.append(path)
    monkeypatch.setattr(
        offline_wheelhouse.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("download must not run"),
    )

    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.build_wheelhouse(tmp_path / "wheelhouse", paths, "/fake/python")


def test_build_rejects_an_empty_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("demo==1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        offline_wheelhouse.subprocess,
        "run",
        lambda _command, *, check: None,
    )

    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.build_wheelhouse(tmp_path / "wheelhouse", [lock], sys.executable)


def test_build_rejects_a_different_python_executable(tmp_path: Path) -> None:
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("demo==1.0\n", encoding="utf-8")

    with pytest.raises(offline_wheelhouse.WheelhouseError, match="Python executable"):
        offline_wheelhouse.build_wheelhouse(
            tmp_path / "wheelhouse", [lock], "/not/the/current/python"
        )


def test_prepare_lock_removes_only_index_declarations(tmp_path: Path) -> None:
    source = tmp_path / "requirements-lock.txt"
    source.write_text(
        "# locked\n"
        "--index-url https://example.invalid/simple\n"
        "--extra-index-url=https://extra.invalid/simple\n"
        "demo==1.0 \\\n"
        "    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "offline.txt"

    assert offline_wheelhouse.write_offline_requirements(source, output) == output
    assert output.read_text(encoding="utf-8") == (
        "# locked\ndemo==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"
    )


@pytest.mark.parametrize(
    "unsafe_line",
    (
        "--find-links https://example.invalid/wheels\n",
        "-r nested-lock.txt\n",
        "demo @ https://example.invalid/demo.whl\n",
    ),
)
def test_prepare_lock_rejects_network_and_nested_inputs(tmp_path: Path, unsafe_line: str) -> None:
    source = tmp_path / "requirements-lock.txt"
    source.write_text(unsafe_line, encoding="utf-8")

    with pytest.raises(offline_wheelhouse.WheelhouseError):
        offline_wheelhouse.write_offline_requirements(source, tmp_path / "offline.txt")


def test_cli_prepare_lock_refuses_to_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "requirements-lock.txt"
    source.write_text("demo==1.0\n", encoding="utf-8")
    output = tmp_path / "offline.txt"
    output.write_text("existing\n", encoding="utf-8")

    assert (
        offline_wheelhouse.main(
            [
                "prepare-lock",
                "--source",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name",
    (
        "requirements-lock.txt",
        "requirements-test-lock.txt",
        "requirements-quality-lock.txt",
    ),
)
def test_project_locks_have_a_network_free_install_derivative(tmp_path: Path, name: str) -> None:
    output = tmp_path / name

    offline_wheelhouse.write_offline_requirements(ROOT / name, output)

    contents = output.read_text(encoding="utf-8")
    assert "://" not in contents
    assert "--index-url" not in contents
    assert "--extra-index-url" not in contents
    assert "--hash=sha256:" in contents


def test_cli_verify_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wheelhouse, lock, _, _ = _build_fixture(
        tmp_path, monkeypatch, {"demo-1.0-py3-none-any.whl": b"payload"}
    )
    assert (
        offline_wheelhouse.main(
            ["verify", "--wheelhouse", str(wheelhouse), "--requirements", str(lock)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["schema"] == offline_wheelhouse.SCHEMA

    lock.write_text(lock.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    assert (
        offline_wheelhouse.main(
            ["verify", "--wheelhouse", str(wheelhouse), "--requirements", str(lock)]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


def test_cli_build_uses_repeated_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lock_one = tmp_path / "first-lock.txt"
    lock_two = tmp_path / "second-lock.txt"
    lock_one.write_text("first\n", encoding="utf-8")
    lock_two.write_text("second\n", encoding="utf-8")
    output = tmp_path / "wheelhouse"

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        destination = Path(command[command.index("--dest") + 1])
        (destination / "demo-1.0-py3-none-any.whl").write_bytes(b"payload")

    monkeypatch.setattr(offline_wheelhouse.subprocess, "run", fake_run)
    assert (
        offline_wheelhouse.main(
            [
                "build",
                "--output",
                str(output),
                "--requirements",
                str(lock_one),
                "--requirements",
                str(lock_two),
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in manifest["requirements"]] == [
        lock_one.name,
        lock_two.name,
    ]
