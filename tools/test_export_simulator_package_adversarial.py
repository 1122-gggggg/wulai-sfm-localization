from __future__ import annotations

import hashlib
import json
from pathlib import Path

import export_simulator_package as exporter
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(root: Path, relative: str, payload: bytes = b"artifact") -> Path:
    artifact = root / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(payload)
    (root / exporter.ARTIFACT_MANIFEST).write_text(
        json.dumps(
            {
                "schema": "sfm-runtime-artifacts/v1",
                "artifacts": [
                    {
                        "name": "fixture",
                        "path": relative,
                        "size_bytes": len(payload),
                        "sha256": _sha256(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _minimal_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "code").mkdir()
    (source / "code/main.py").write_text("main\n", encoding="utf-8")
    runtime = source / "執行環境"
    runtime.mkdir()
    for name in ("requirements_runtime.txt", "requirements_test.txt", "README.md"):
        (runtime / name).write_text(f"{name}\n", encoding="utf-8")
    _write_registry(source, "artifact.bin")
    monkeypatch.setattr(exporter, "ROOT", source)
    monkeypatch.setattr(exporter, "COPY_DIRS", ("code",))
    monkeypatch.setattr(exporter, "COPY_FILES", (exporter.ARTIFACT_MANIFEST,))
    monkeypatch.setattr(exporter, "_source_release", lambda: {})
    return source


@pytest.mark.parametrize("link_kind", ("leaf", "parent"))
def test_runtime_artifacts_reject_symlink_leaf_and_parent(
    tmp_path: Path, link_kind: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_artifact = outside / "model.bin"
    outside_artifact.write_bytes(b"artifact")
    relative = "nested/model.bin"
    (source / exporter.ARTIFACT_MANIFEST).write_text(
        json.dumps(
            {
                "schema": "sfm-runtime-artifacts/v1",
                "artifacts": [
                    {
                        "name": "fixture",
                        "path": relative,
                        "size_bytes": outside_artifact.stat().st_size,
                        "sha256": _sha256(outside_artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if link_kind == "leaf":
        (source / "nested").mkdir()
        (source / relative).symlink_to(outside_artifact)
    else:
        (source / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(exporter.ArtifactResolutionError, match="escapes|symlink"):
        exporter.resolve_runtime_artifacts(source)


def test_runtime_artifacts_reject_symlink_parent_in_seed_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    seed = tmp_path / "seed"
    outside = tmp_path / "outside"
    source.mkdir()
    seed.mkdir()
    outside.mkdir()
    artifact = outside / "model.bin"
    artifact.write_bytes(b"artifact")
    relative = "nested/model.bin"
    (source / exporter.ARTIFACT_MANIFEST).write_text(
        json.dumps(
            {
                "schema": "sfm-runtime-artifacts/v1",
                "artifacts": [
                    {
                        "name": "fixture",
                        "path": relative,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _sha256(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (seed / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(exporter.ArtifactResolutionError, match="escapes|symlink"):
        exporter.resolve_runtime_artifacts(source, artifact_root=seed)


def test_runtime_artifact_copy_rejects_symlink_destination_parent(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"artifact")
    destination = tmp_path / "destination"
    destination.mkdir()
    safe_parent = destination / "safe"
    safe_parent.mkdir()
    (destination / "nested").symlink_to(safe_parent, target_is_directory=True)
    spec = exporter.RuntimeArtifact(
        name="fixture",
        path="nested/model.bin",
        size_bytes=source.stat().st_size,
        sha256=_sha256(source),
    )

    with pytest.raises(exporter.ArtifactResolutionError, match="symlink|escapes"):
        exporter._copy_resolved_runtime_artifacts(
            (exporter.ResolvedRuntimeArtifact(spec, source),), destination
        )
    assert not (safe_parent / "model.bin").exists()


def test_site_bundle_requires_digest_for_non_empty_legacy_assets(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    cache = root / "cache.npz"
    landmarks = root / "landmarks.npz"
    cache.write_bytes(b"cache")
    landmarks.write_bytes(b"landmarks")
    profile = root / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "site_id": "fixture",
                "assets": {
                    "megaloc_cache": cache.name,
                    "track_landmarks": landmarks.name,
                },
                "asset_sha256": {"track_landmarks": _sha256(landmarks)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(exporter.ArtifactResolutionError, match="SHA-256"):
        exporter.collect_site_bundle(root, [profile])

    profile_data = json.loads(profile.read_text(encoding="utf-8"))
    profile_data["asset_sha256"]["megaloc_cache"] = _sha256(cache)
    profile.write_text(json.dumps(profile_data), encoding="utf-8")
    bundle = exporter.collect_site_bundle(root, [profile])
    assert {item.path for item in bundle.files} == {
        "profile.json",
        "cache.npz",
        "landmarks.npz",
    }


def test_export_failure_does_not_pollute_requested_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _minimal_source(tmp_path, monkeypatch)
    destination = tmp_path / "portable"
    destination.mkdir()

    def fail_after_partial_copy(stage: Path, *_args: object, **_kwargs: object) -> None:
        (stage / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("copy failed")

    monkeypatch.setattr(exporter, "_copy_export_payload", fail_after_partial_copy)
    with pytest.raises(RuntimeError, match="copy failed"):
        exporter.export(destination)

    assert list(destination.iterdir()) == []
    assert not any(path.name.startswith(".portable.") for path in tmp_path.iterdir())


def test_export_rejects_source_identity_change_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _minimal_source(tmp_path, monkeypatch)
    destination = tmp_path / "portable"
    destination.mkdir()
    identities = iter(({"release": "old"}, {"release": "new"}))
    monkeypatch.setattr(exporter, "_source_release", lambda: next(identities))

    with pytest.raises(ValueError, match="source release changed"):
        exporter.export(destination)

    assert list(destination.iterdir()) == []


def test_export_preserves_source_symlink_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _minimal_source(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (exporter.ROOT / "code/leak.txt").symlink_to(outside)
    destination = tmp_path / "portable"
    destination.mkdir()

    with pytest.raises(exporter.ArtifactResolutionError, match="symlink"):
        exporter.export(destination)

    assert list(destination.iterdir()) == []
    assert outside.read_text(encoding="utf-8") == "outside"
