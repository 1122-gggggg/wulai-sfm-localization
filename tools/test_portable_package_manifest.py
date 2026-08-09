from __future__ import annotations

import hashlib
import json
from pathlib import Path

import export_simulator_package
import pytest
from package_manifest import generate, included, verify


def test_current_manifest_excludes_imported_assets_and_runtime_caches() -> None:
    assert not included(Path(".venv/bin/python"))
    assert not included(Path("outputs/validation.json"))
    assert not included(Path("地圖檔/場域/river_site/maps/site.ply"))
    assert not included(Path("模擬器/測試影片/replay.mp4"))
    assert not included(Path("模擬器/封存/old_worktree/src/module.py"))
    assert not included(Path("執行環境/inductor_cache/kernel.py"))
    assert included(Path("控制介面程式/影片模擬串流/啟動.sh"))


def test_manifest_excludes_editor_and_transient_release_files() -> None:
    for path in (
        Path(".vscode/settings.json"),
        Path(".idea/workspace.xml"),
        Path("notes.swp"),
        Path("notes~"),
        Path("report.tmp"),
        Path(".DS_Store"),
        Path("audit/final_validation.md"),
        Path(".cursor/session.json"),
        Path("env/bin/python"),
        Path(".coverage"),
        Path(".coverage.host.123.random"),
        Path("coverage.xml"),
        Path("htmlcov/index.html"),
    ):
        assert not included(path)
    assert included(Path(".coveragerc"))


def test_release_tools_have_one_authoritative_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "執行環境/tools/package_manifest.py").exists()
    assert not (root / "執行環境/MANIFEST.tsv").exists()
    assert not (root / "執行環境/SHA256SUMS").exists()
    assert not (root / "定位演算法/sync_mirror_check.sh").exists()


def test_portable_export_keeps_output_governance_readme() -> None:
    import export_simulator_package as exporter

    assert "outputs/README.md" in exporter.COPY_FILES
    assert {
        "pyproject.toml",
        "pytest.ini",
        "requirements-quality.txt",
        "requirements-quality-lock.txt",
    }.issubset(exporter.COPY_FILES)

    runtime_script = (Path(__file__).parent / "test_portable_runtime.sh").read_text(
        encoding="utf-8"
    )
    assert '"$portable_root/outputs/README.md"' in runtime_script
    assert 'rm -r -- "$staged_output_root/flight_logs"' in runtime_script
    assert 'rm -r -- "$staged_output_root"' not in runtime_script
    assert "routes/authored/route_20260807_013811.json" in runtime_script
    assert "maps/T_align_gravity.json" in runtime_script
    assert '"$portable_root/PORTABLE_SITE_ASSETS.json"' in runtime_script
    assert "reports/hardware_approval_20260806.json" not in runtime_script
    assert "river_site_safezone/flight_path.json" not in runtime_script


def test_manifest_round_trip(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.txt").write_text("b", encoding="utf-8")
    generate(tmp_path)
    assert verify(tmp_path) == []
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    assert any(
        marker in issue
        for issue in verify(tmp_path)
        for marker in ("SHA-256 mismatch: a.txt", "size mismatch: a.txt")
    )


def _write_runtime_registry(root: Path, *artifacts: dict[str, object]) -> None:
    (root / "RUNTIME_ARTIFACTS.json").write_text(
        json.dumps(
            {
                "schema": "sfm-runtime-artifacts/v1",
                "seed": {
                    "layout": "repository-relative",
                    "hint": "seed from an approved offline runtime artifact bundle",
                },
                "artifacts": list(artifacts),
            }
        ),
        encoding="utf-8",
    )


def test_source_manifest_separates_external_runtime_artifacts() -> None:
    assert not included(
        Path("執行環境/torch_hub_cache/checkpoints/model.safetensors"),
        source_only=True,
    )
    assert not included(
        Path("定位演算法/deploy_code/runtime/EDM/weights/edm_outdoor.ckpt"),
        source_only=True,
    )
    assert included(
        Path("定位演算法/deploy_code/runtime/EDM/weights/.gitignore"),
        source_only=True,
    )


def test_runtime_artifact_resolver_reports_seed_guidance_for_clean_checkout(
    tmp_path: Path,
) -> None:
    _write_runtime_registry(
        tmp_path,
        {
            "name": "demo-model",
            "path": "執行環境/torch_hub_cache/checkpoints/model.bin",
            "size_bytes": 3,
            "sha256": "a" * 64,
        },
    )

    with pytest.raises(export_simulator_package.ArtifactResolutionError) as exc_info:
        export_simulator_package.resolve_runtime_artifacts(tmp_path)

    message = str(exc_info.value)
    assert "demo-model" in message
    assert "a" * 64 in message
    assert "fetch/seed" in message
    assert "--artifact-root" in message
    assert "network" in message.lower()


def test_runtime_artifact_copy_uses_only_digest_bound_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    seed = tmp_path / "seed"
    destination = tmp_path / "portable"
    source.mkdir()
    seed.mkdir()
    payload = b"abc"
    artifact_path = seed / "執行環境/torch_hub_cache/checkpoints/model.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(payload)
    (seed / "執行環境/torch_hub_cache/checkpoints/unlisted.bin").write_bytes(
        b"must not be copied"
    )
    _write_runtime_registry(
        source,
        {
            "name": "demo-model",
            "path": "執行環境/torch_hub_cache/checkpoints/model.bin",
            "size_bytes": len(payload),
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
    )

    copied = export_simulator_package.copy_runtime_artifacts(
        source, destination, artifact_root=seed
    )

    assert copied == ["執行環境/torch_hub_cache/checkpoints/model.bin"]
    assert (destination / copied[0]).read_bytes() == payload
    assert not (destination / "執行環境/torch_hub_cache/checkpoints/unlisted.bin").exists()


def test_runtime_artifact_copy_rejects_digest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "artifact.bin"
    artifact.write_bytes(b"wrong")
    _write_runtime_registry(
        source,
        {
            "name": "demo-model",
            "path": "artifact.bin",
            "size_bytes": 5,
            "sha256": "a" * 64,
        },
    )

    with pytest.raises(export_simulator_package.ArtifactResolutionError, match="SHA-256"):
        export_simulator_package.copy_runtime_artifacts(source, tmp_path / "portable")


def test_exporter_does_not_walk_unlisted_runtime_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "portable"
    source.mkdir()
    (source / "code").mkdir()
    (source / "code/main.py").write_text("main\n", encoding="utf-8")
    runtime = source / "執行環境"
    runtime.mkdir()
    for name in ("requirements_runtime.txt", "requirements_test.txt", "README.md"):
        (runtime / name).write_text(f"{name}\n", encoding="utf-8")
    selected = runtime / "torch_hub_cache/checkpoints/model.bin"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"abc")
    (selected.parent / "unlisted.bin").write_bytes(b"do not copy")
    _write_runtime_registry(
        source,
        {
            "name": "demo-model",
            "path": "執行環境/torch_hub_cache/checkpoints/model.bin",
            "size_bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
    )
    monkeypatch.setattr(export_simulator_package, "ROOT", source)
    monkeypatch.setattr(export_simulator_package, "COPY_DIRS", ("code",))
    monkeypatch.setattr(export_simulator_package, "COPY_FILES", ("RUNTIME_ARTIFACTS.json",))
    monkeypatch.setattr(export_simulator_package, "_source_release", lambda: {})

    export_simulator_package.export(destination)

    assert (destination / "code/main.py").is_file()
    assert (destination / "執行環境/torch_hub_cache/checkpoints/model.bin").is_file()
    assert not (destination / "執行環境/torch_hub_cache/checkpoints/unlisted.bin").exists()
    metadata = json.loads((destination / "PORTABLE_PACKAGE.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["scope"] == "git-source"
    assert metadata["runtime_artifacts"]["scope"] == "external-runtime-artifacts"
    assert metadata["runtime_artifacts"]["artifact_count"] == 1
    assert verify(destination) == []


def test_portable_manifest_rechecks_registry_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")
    _write_runtime_registry(
        tmp_path,
        {
            "name": "demo-model",
            "path": "artifact.bin",
            "size_bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
    )
    (tmp_path / "PORTABLE_PACKAGE.json").write_text("{}", encoding="utf-8")
    generate(tmp_path)
    assert verify(tmp_path) == []

    artifact.write_bytes(b"xyz")
    issues = verify(tmp_path)
    assert any("runtime artifact SHA-256 mismatch" in issue for issue in issues)


def test_export_rejects_a_destination_inside_the_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(export_simulator_package, "ROOT", source)

    with pytest.raises(ValueError, match="must not contain each other"):
        export_simulator_package.export(source / "portable")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_site_profile(root: Path, *, hardware: dict[str, object] | None = None) -> Path:
    profile = root / "控制介面程式/site_profiles/demo.json"
    profile.parent.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": 2,
        "site_id": "demo_site",
        "assets": {
            "reference_index": "../../maps/index/SHA256SUMS.json",
        },
        "asset_sha256": {},
    }
    if hardware is not None:
        payload["hardware_approval"] = hardware
    profile.write_text(json.dumps(payload), encoding="utf-8")
    return profile


def _write_reference_index(root: Path) -> Path:
    index_root = root / "maps/index"
    index_root.mkdir(parents=True)
    files: dict[str, str] = {}
    for name, content in (
        ("metadata.json", b'{"count": 2}\n'),
        ("descriptors.npy", b"descriptors"),
        ("names.json", b'["a", "b"]\n'),
    ):
        path = index_root / name
        path.write_bytes(content)
        files[name] = _sha256(path)
    manifest = index_root / "SHA256SUMS.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "localization-reference-index-sha256",
                "format_version": 1,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_site_bundle_copies_and_verifies_all_reference_index_siblings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    destination = tmp_path / "bundle"
    index_manifest = _write_reference_index(root)
    profile = _write_site_profile(root)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")

    copied = export_simulator_package.copy_site_bundle(root, destination, [profile])

    assert set(copied) == {
        "控制介面程式/site_profiles/demo.json",
        "maps/index/SHA256SUMS.json",
        "maps/index/descriptors.npy",
        "maps/index/metadata.json",
        "maps/index/names.json",
    }
    assert verify(destination) == []
    marker = json.loads(
        (destination / "PORTABLE_SITE_ASSETS.json").read_text(encoding="utf-8")
    )
    assert marker["schema"] == "sfm-portable-site-assets/v1"
    assert {item["path"] for item in marker["files"]} == set(copied)


def test_site_bundle_rejects_reference_index_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "source"
    index_manifest = _write_reference_index(root)
    profile = _write_site_profile(root)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")
    index_manifest.write_text(
        json.dumps(
            {
                "schema": "localization-reference-index-sha256",
                "files": {"../escape.bin": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(export_simulator_package.ArtifactResolutionError, match="unsafe"):
        export_simulator_package.collect_site_bundle(root, [profile])


def test_site_bundle_rejects_profile_reference_index_pin_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _write_reference_index(root)
    profile = _write_site_profile(root)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw["asset_sha256"]["reference_index"] = "0" * 64
    profile.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(export_simulator_package.ArtifactResolutionError, match="SHA-256 mismatch"):
        export_simulator_package.collect_site_bundle(root, [profile])


def test_site_bundle_rejects_reference_index_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "source"
    index_manifest = _write_reference_index(root)
    profile = _write_site_profile(root)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    sibling = index_manifest.parent / "names.json"
    sibling.unlink()
    sibling.symlink_to(outside)
    index_data = json.loads(index_manifest.read_text(encoding="utf-8"))
    index_data["files"]["names.json"] = _sha256(outside)
    index_manifest.write_text(json.dumps(index_data), encoding="utf-8")
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(export_simulator_package.ArtifactResolutionError, match="symlinks"):
        export_simulator_package.collect_site_bundle(root, [profile])


def test_site_bundle_rejects_partial_signed_hardware_reference(tmp_path: Path) -> None:
    root = tmp_path / "source"
    index_manifest = _write_reference_index(root)
    receipt = root / "maps/reports/receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("receipt\n", encoding="utf-8")
    profile = _write_site_profile(
        root,
        hardware={
            "receipt": "../../maps/reports/receipt.json",
            "sha256": _sha256(receipt),
            "signature": "../../maps/reports/missing.sig",
        },
    )
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(export_simulator_package.ArtifactResolutionError, match="sidecars"):
        export_simulator_package.collect_site_bundle(root, [profile])


def test_site_bundle_includes_signed_hardware_sidecars_and_checks_pins(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    index_manifest = _write_reference_index(root)
    reports = root / "maps/reports"
    reports.mkdir(parents=True)
    receipt = reports / "receipt.json"
    signature = reports / "receipt.sig"
    trust_store = reports / "trust.json"
    receipt.write_text('{"schema":"anafi-hardware-approval/v2"}\n', encoding="utf-8")
    signature.write_text("signature\n", encoding="utf-8")
    trust_store.write_text('{"keys":[]}\n', encoding="utf-8")
    hardware = {
        "receipt": "../../maps/reports/receipt.json",
        "sha256": _sha256(receipt),
        "signature": "../../maps/reports/receipt.sig",
        "signature_sha256": _sha256(signature),
        "trust_store": "../../maps/reports/trust.json",
        "trust_store_sha256": _sha256(trust_store),
    }
    profile = _write_site_profile(root, hardware=hardware)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")

    export_simulator_package.copy_site_bundle(root, tmp_path / "bundle", [profile])
    bundle = json.loads(
        (tmp_path / "bundle/PORTABLE_SITE_ASSETS.json").read_text(encoding="utf-8")
    )
    profile_record = bundle["profiles"][0]
    assert profile_record["signed_hardware_approval"] is True
    assert {
        item["path"]
        for item in bundle["files"]
        if item["role"].startswith("hardware_approval_")
    } == {
        "maps/reports/receipt.json",
        "maps/reports/receipt.sig",
        "maps/reports/trust.json",
    }

    (tmp_path / "bundle/maps/reports/trust.json").write_text(
        "tampered\n", encoding="utf-8"
    )
    assert any(
        "PORTABLE_SITE_ASSETS.json" in issue
        and "maps/reports/trust.json" in issue
        for issue in verify(tmp_path / "bundle")
    )


def test_site_bundle_allows_legacy_unsigned_hardware_receipt_without_signing_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    index_manifest = _write_reference_index(root)
    receipt = root / "maps/reports/legacy-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"approved":false}\n', encoding="utf-8")
    profile = _write_site_profile(
        root,
        hardware={
            "receipt": "../../maps/reports/legacy-receipt.json",
            "sha256": _sha256(receipt),
        },
    )
    raw = json.loads(profile.read_text(encoding="utf-8"))
    raw["asset_sha256"]["reference_index"] = _sha256(index_manifest)
    profile.write_text(json.dumps(raw), encoding="utf-8")

    export_simulator_package.copy_site_bundle(root, tmp_path / "bundle", [profile])
    bundle = json.loads(
        (tmp_path / "bundle/PORTABLE_SITE_ASSETS.json").read_text(encoding="utf-8")
    )
    assert bundle["profiles"][0]["signed_hardware_approval"] is False
    assert verify(tmp_path / "bundle") == []
