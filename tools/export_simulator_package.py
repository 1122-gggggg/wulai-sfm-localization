#!/usr/bin/env python3
"""Export fixed simulator/UI/runtime assets for another computer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_MANIFEST = "RUNTIME_ARTIFACTS.json"
ARTIFACT_ROOT_ENV = "SFM_RUNTIME_ARTIFACT_ROOT"
SITE_ASSET_MANIFEST = "PORTABLE_SITE_ASSETS.json"
REFERENCE_INDEX_SCHEMA = "localization-reference-index-sha256"
COPY_DIRS = (
    "控制介面程式",
    "定位演算法",
    "模擬器/parrot_stimulate",
    "文件",
    "tools",
)
COPY_FILES = (
    "README.md",
    "outputs/README.md",
    ARTIFACT_MANIFEST,
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-test.txt",
    "requirements-test-lock.txt",
    "requirements-quality.txt",
    "requirements-quality-lock.txt",
    "pyproject.toml",
    "pytest.ini",
    "驗證系統.sh",
)
EXCLUDED_NAMES = {
    ".git",
    ".codegraph",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "EDM工具包",
    "inductor_cache",
    "outputs",
    "package_git",
    "report_20260714",
    "source_videos",
}


@dataclass(frozen=True)
class RuntimeArtifact:
    name: str
    path: str
    size_bytes: int
    sha256: str
    fetch_hint: str = ""


@dataclass(frozen=True)
class ResolvedRuntimeArtifact:
    spec: RuntimeArtifact
    source: Path


@dataclass(frozen=True)
class SiteBundleFile:
    path: str
    size_bytes: int
    sha256: str
    role: str
    profile: str


@dataclass(frozen=True)
class SiteBundle:
    profiles: tuple[dict[str, object], ...]
    files: tuple[SiteBundleFile, ...]


class ArtifactResolutionError(ValueError):
    """Raised when an external runtime artifact is absent or not trusted."""


_PROFILE_ASSET_KEYS = (
    "map_ply",
    "localization_bundle",
    "route_json",
    "map_reference_poses",
    "localizer_profile",
    "reference_index",
    "poles_json",
    "map_align",
)
_SIGNED_HARDWARE_KEYS = (
    "signature",
    "signature_sha256",
    "trust_store",
    "trust_store_sha256",
)


def _validate_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactResolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_repo_file(root: Path, candidate: Path, *, label: str) -> Path:
    """Resolve a profile asset without allowing traversal or symlink escape."""
    try:
        current = candidate
        while True:
            if current.is_symlink():
                raise ArtifactResolutionError(f"{label} must not use symlinks: {candidate}")
            if current == root:
                break
            if current == current.parent:
                break
            current = current.parent
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ArtifactResolutionError(f"{label} cannot be resolved: {candidate}") from exc
    if not resolved.is_relative_to(root):
        raise ArtifactResolutionError(f"{label} escapes repository root: {candidate}")
    if not candidate.is_file():
        raise ArtifactResolutionError(f"{label} is missing or not a regular file: {candidate}")
    return resolved


def _profile_asset_path(root: Path, profile: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactResolutionError(f"{label} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = profile.parent / candidate
    return _safe_repo_file(root, candidate, label=label)


def _verify_bound_file(path: Path, expected: object, *, label: str) -> str:
    expected_sha256 = _validate_sha256(expected, label=label)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ArtifactResolutionError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    return actual_sha256


def _load_reference_index_files(
    root: Path,
    manifest_path: Path,
    *,
    expected_sha256: object,
    profile_label: str,
) -> tuple[tuple[Path, str], ...]:
    if manifest_path.name != "SHA256SUMS.json":
        raise ArtifactResolutionError(
            f"{profile_label} reference_index must point to SHA256SUMS.json"
        )
    _verify_bound_file(
        manifest_path,
        expected_sha256,
        label=f"{profile_label} reference_index",
    )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactResolutionError(
            f"{profile_label} reference index manifest cannot be read"
        ) from exc
    if not isinstance(data, dict) or data.get("schema") != REFERENCE_INDEX_SCHEMA:
        raise ArtifactResolutionError(
            f"{profile_label} reference index manifest has unsupported schema"
        )
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactResolutionError(
            f"{profile_label} reference index manifest has no files"
        )
    result: list[tuple[Path, str]] = [(manifest_path, "reference_index_manifest")]
    for name, expected in sorted(files.items()):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "\\" in name
        ):
            raise ArtifactResolutionError(
                f"{profile_label} reference index sibling path is unsafe: {name!r}"
            )
        sibling = _safe_repo_file(
            root,
            manifest_path.parent / name,
            label=f"{profile_label} reference index sibling",
        )
        _verify_bound_file(
            sibling,
            expected,
            label=f"{profile_label} reference index sibling {name}",
        )
        result.append((sibling, "reference_index_sibling"))
    return tuple(result)


def _read_site_profile(root: Path, raw_profile: str | Path) -> tuple[Path, dict[str, object]]:
    candidate = Path(raw_profile).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    profile_path = _safe_repo_file(root, candidate, label="site profile")
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactResolutionError(f"site profile cannot be read: {profile_path}") from exc
    if not isinstance(raw, dict):
        raise ArtifactResolutionError(f"site profile must contain an object: {profile_path}")
    return profile_path, raw


def _collect_profile_assets(
    root: Path,
    profile_path: Path,
    raw: dict[str, object],
    profile_label: str,
    add_file: Callable[..., str],
) -> list[str]:
    assets = raw.get("assets", {})
    digests = raw.get("asset_sha256", {})
    if not isinstance(assets, dict) or not isinstance(digests, dict):
        raise ArtifactResolutionError(
            f"site profile assets/digests must be objects: {profile_path}"
        )
    profile_files: list[str] = []
    top_level = {"localizer_profile", "map_reference_poses", "map_align"}
    for key in _PROFILE_ASSET_KEYS:
        value = raw.get(key) if key in top_level else assets.get(key)
        if value in (None, ""):
            continue
        asset_path = _profile_asset_path(
            root, profile_path, value, label=f"{profile_label} {key}"
        )
        expected = digests.get(key)
        if key == "reference_index":
            for index_path, role in _load_reference_index_files(
                root,
                asset_path,
                expected_sha256=expected,
                profile_label=profile_label,
            ):
                profile_files.append(
                    add_file(
                        index_path,
                        role=role,
                        profile_label=profile_label,
                        expected=(expected if role == "reference_index_manifest" else None),
                    )
                )
        else:
            profile_files.append(
                add_file(
                    asset_path,
                    role=key,
                    profile_label=profile_label,
                    expected=expected,
                )
            )
    return profile_files


def _collect_hardware_assets(
    root: Path,
    profile_path: Path,
    hardware: object,
    profile_label: str,
    add_file: Callable[..., str],
) -> tuple[list[str], bool]:
    if hardware is None:
        return [], False
    if not isinstance(hardware, dict):
        raise ArtifactResolutionError(
            f"{profile_label} hardware_approval must be an object"
        )
    receipt = _profile_asset_path(
        root,
        profile_path,
        hardware.get("receipt"),
        label=f"{profile_label} hardware approval receipt",
    )
    files = [
        add_file(
            receipt,
            role="hardware_approval_receipt",
            profile_label=profile_label,
            expected=hardware.get("sha256"),
        )
    ]
    sidecars = [key in hardware for key in _SIGNED_HARDWARE_KEYS]
    if not any(sidecars):
        return files, False
    if not all(sidecars):
        raise ArtifactResolutionError(
            f"{profile_label} signed hardware approval must pin all sidecars"
        )
    for key, role in (
        ("signature", "hardware_approval_signature"),
        ("trust_store", "hardware_approval_trust_store"),
    ):
        sidecar = _profile_asset_path(
            root,
            profile_path,
            hardware.get(key),
            label=f"{profile_label} {key}",
        )
        files.append(
            add_file(
                sidecar,
                role=role,
                profile_label=profile_label,
                expected=hardware.get(f"{key}_sha256"),
            )
        )
    return files, True


def collect_site_bundle(
    root: str | Path,
    profile_paths: Iterable[str | Path],
) -> SiteBundle:
    """Collect selected site assets with profile and index digest bindings.

    The caller must explicitly select profiles.  This keeps the normal simulator
    exporter lightweight while making a selected offline site bundle complete.
    """
    root = Path(root).expanduser().resolve()
    files: dict[str, SiteBundleFile] = {}
    profiles: list[dict[str, object]] = []
    seen_profiles: set[str] = set()

    def add_file(path: Path, *, role: str, profile_label: str, expected: object = None) -> str:
        relative = path.relative_to(root).as_posix()
        actual_sha256 = _sha256(path)
        if expected is not None:
            _verify_bound_file(path, expected, label=f"{profile_label} {role}")
        existing = files.get(relative)
        if existing is None:
            files[relative] = SiteBundleFile(
                path=relative,
                size_bytes=path.stat().st_size,
                sha256=actual_sha256,
                role=role,
                profile=profile_label,
            )
        elif existing.sha256 != actual_sha256:
            raise ArtifactResolutionError(f"site asset changed while collecting: {relative}")
        return relative

    for raw_profile in profile_paths:
        profile_path, raw = _read_site_profile(root, raw_profile)
        profile_relative = profile_path.relative_to(root).as_posix()
        if profile_relative in seen_profiles:
            continue
        seen_profiles.add(profile_relative)
        profile_label = str(raw.get("site_id") or profile_relative)
        profile_files = [add_file(profile_path, role="site_profile", profile_label=profile_label)]
        profile_files.extend(
            _collect_profile_assets(
                root, profile_path, raw, profile_label, add_file
            )
        )
        hardware_files, signed_hardware = _collect_hardware_assets(
            root, profile_path, raw.get("hardware_approval"), profile_label, add_file
        )
        profile_files.extend(hardware_files)
        profiles.append(
            {
                "path": profile_relative,
                "site_id": raw.get("site_id"),
                "signed_hardware_approval": signed_hardware,
                "files": sorted(set(profile_files)),
            }
        )
    if not profiles:
        raise ArtifactResolutionError("at least one site profile is required")
    return SiteBundle(tuple(profiles), tuple(files.values()))


def _write_site_bundle_manifest(destination: Path, bundle: SiteBundle) -> None:
    payload = {
        "schema": "sfm-portable-site-assets/v1",
        "profiles": list(bundle.profiles),
        "files": [
            {
                "path": item.path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "role": item.role,
                "profile": item.profile,
            }
            for item in sorted(bundle.files, key=lambda item: item.path)
        ],
    }
    (destination / SITE_ASSET_MANIFEST).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_destination_path(destination: Path, relative: str) -> Path:
    target = destination / relative
    try:
        resolved = target.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ArtifactResolutionError(f"site bundle destination cannot be resolved: {relative}") from exc
    if not resolved.is_relative_to(destination):
        raise ArtifactResolutionError(f"site bundle destination escapes root: {relative}")
    current = target
    while current != destination:
        if current.is_symlink():
            raise ArtifactResolutionError(f"site bundle destination contains symlink: {relative}")
        if current == current.parent:
            break
        current = current.parent
    return target


def _copy_site_bundle_files(
    root: Path, destination: Path, bundle: SiteBundle
) -> list[str]:
    copied: list[str] = []
    for item in sorted(bundle.files, key=lambda item: item.path):
        source = _safe_repo_file(root, root / item.path, label="site bundle asset")
        target = _safe_destination_path(destination, item.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise ArtifactResolutionError(f"site bundle target is not a regular file: {target}")
            if _sha256(target) != item.sha256:
                raise ArtifactResolutionError(f"site bundle target conflicts: {item.path}")
        else:
            shutil.copy2(source, target)
        copied.append(item.path)
    _write_site_bundle_manifest(destination, bundle)
    return copied


def copy_site_bundle(
    root: str | Path,
    destination: str | Path,
    profile_paths: Iterable[str | Path],
) -> list[str]:
    """Copy a selected, digest-bound site bundle without network access."""
    root_path = Path(root).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if destination_path == root_path or root_path.is_relative_to(destination_path):
        raise ArtifactResolutionError("site bundle destination must be outside source root")
    destination_path.mkdir(parents=True, exist_ok=True)
    bundle = collect_site_bundle(root_path, profile_paths)
    copied = _copy_site_bundle_files(root_path, destination_path, bundle)
    sys.path.insert(0, str(root_path))
    from tools.package_manifest import generate

    generate(destination_path, include_site_assets=True)
    return copied


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in EXCLUDED_NAMES}
    path = Path(directory)
    if path.name == "weights" and path.parent.name == "EDM":
        ignored.update(name for name in names if name != ".gitignore")
    return ignored


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=_ignore)


def _source_release() -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from tools.package_manifest import verify
    from tools.release_contract import source_release_identity

    issues = verify(ROOT, source_only=True)
    if issues:
        if any(ARTIFACT_MANIFEST in issue for issue in issues):
            raise ArtifactResolutionError(
                f"source checkout is missing {ARTIFACT_MANIFEST}; "
                "fetch/seed the registry from the Git source checkout; "
                "no network fetch was attempted"
            )
        raise ValueError(f"source package manifest is stale: {issues[0]}")
    return source_release_identity(ROOT)


def _sha256(path: Path) -> str:
    from tools.package_manifest import digest

    return digest(path)


def _load_runtime_artifact_data(root: Path) -> dict[str, object]:
    manifest_path = root / ARTIFACT_MANIFEST
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactResolutionError(
            f"missing {ARTIFACT_MANIFEST}; clean checkout has no runtime artifact registry"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactResolutionError(
            f"cannot read {ARTIFACT_MANIFEST}: {exc}"
        ) from exc
    if not isinstance(data, dict) or data.get("schema") != "sfm-runtime-artifacts/v1":
        raise ArtifactResolutionError(
            f"{ARTIFACT_MANIFEST} has unsupported schema; expected sfm-runtime-artifacts/v1"
        )
    if not isinstance(data.get("artifacts"), list) or not data["artifacts"]:
        raise ArtifactResolutionError(f"{ARTIFACT_MANIFEST} has no runtime artifacts")
    return data


def load_runtime_artifacts(root: str | Path) -> tuple[RuntimeArtifact, ...]:
    """Load the checked-in allowlist for external runtime artifacts."""
    root = Path(root).expanduser().resolve()
    data = _load_runtime_artifact_data(root)
    result: list[RuntimeArtifact] = []
    names: set[str] = set()
    paths: set[str] = set()
    for raw in data["artifacts"]:
        if not isinstance(raw, dict):
            raise ArtifactResolutionError(f"{ARTIFACT_MANIFEST} contains a non-object artifact")
        name = raw.get("name")
        relative = raw.get("path")
        size_bytes = raw.get("size_bytes")
        sha256 = raw.get("sha256")
        fetch_hint = raw.get("fetch_hint", "")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(relative, str)
            or not relative.strip()
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(fetch_hint, str)
        ):
            raise ArtifactResolutionError(
                f"{ARTIFACT_MANIFEST} contains an invalid artifact entry: {raw!r}"
            )
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ArtifactResolutionError(
                f"{ARTIFACT_MANIFEST} artifact path must be repository-relative: {relative}"
            )
        normalized = candidate.as_posix()
        if name in names or normalized in paths:
            raise ArtifactResolutionError(
                f"{ARTIFACT_MANIFEST} contains duplicate artifact name or path: {name}"
            )
        names.add(name)
        paths.add(normalized)
        result.append(RuntimeArtifact(name, normalized, size_bytes, sha256, fetch_hint))
    return tuple(result)


def _artifact_root(root: Path, artifact_root: str | Path | None) -> Path | None:
    configured = artifact_root
    if configured is None:
        configured = os.environ.get(ARTIFACT_ROOT_ENV, "").strip() or None
    if configured is None:
        return None
    resolved = Path(configured).expanduser().resolve()
    return None if resolved == root else resolved


def resolve_runtime_artifacts(
    root: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> tuple[ResolvedRuntimeArtifact, ...]:
    """Resolve and verify every allowlisted artifact without downloading anything."""
    root = Path(root).expanduser().resolve()
    specs = load_runtime_artifacts(root)
    seed_root = _artifact_root(root, artifact_root)
    resolved: list[ResolvedRuntimeArtifact] = []
    failures: list[str] = []
    for spec in specs:
        candidates = [root / spec.path]
        if seed_root is not None:
            candidates.append(seed_root / spec.path)
        found = False
        for candidate in candidates:
            if not candidate.exists() and not candidate.is_symlink():
                continue
            found = True
            if candidate.is_symlink() or not candidate.is_file():
                failures.append(f"{spec.name}: not a regular file: {candidate}")
                break
            actual_size = candidate.stat().st_size
            if actual_size != spec.size_bytes:
                hint = f"; {spec.fetch_hint}" if spec.fetch_hint else ""
                failures.append(
                    f"{spec.name}: size mismatch at {candidate}; "
                    f"expected {spec.size_bytes}, got {actual_size}{hint}"
                )
                break
            actual_sha256 = _sha256(candidate)
            if actual_sha256 != spec.sha256:
                hint = f"; {spec.fetch_hint}" if spec.fetch_hint else ""
                failures.append(
                    f"{spec.name}: SHA-256 mismatch at {candidate}; "
                    f"expected {spec.sha256}, got {actual_sha256}{hint}"
                )
                break
            resolved.append(ResolvedRuntimeArtifact(spec, candidate))
            break
        if not found:
            hint = f"; {spec.fetch_hint}" if spec.fetch_hint else ""
            failures.append(
                f"{spec.name}: missing {spec.path} (expected SHA-256 {spec.sha256}){hint}"
            )
    if failures:
        guidance = [
            "runtime artifact resolution failed; no network fetch was attempted",
            "fetch/seed an approved offline runtime artifact bundle, preserving "
            "repository-relative paths, then retry",
            f"seed root: pass --artifact-root PATH or set {ARTIFACT_ROOT_ENV}=PATH",
        ]
        raise ArtifactResolutionError(
            "\n".join([*guidance, *[f"- {failure}" for failure in failures]])
        )
    return tuple(resolved)


def _copy_resolved_runtime_artifacts(
    resolved: tuple[ResolvedRuntimeArtifact, ...], destination: Path
) -> list[str]:
    copied: list[str] = []
    for artifact in resolved:
        target = destination / artifact.spec.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact.source, target)
        copied.append(artifact.spec.path)
    return copied


def copy_runtime_artifacts(
    root: str | Path,
    destination: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> list[str]:
    """Copy only verified allowlisted artifacts into a portable package."""
    resolved = resolve_runtime_artifacts(root, artifact_root=artifact_root)
    return _copy_resolved_runtime_artifacts(resolved, Path(destination).expanduser().resolve())


def _write_metadata(
    destination: Path,
    source_release: dict[str, object],
    artifacts: tuple[RuntimeArtifact, ...],
) -> None:
    artifact_manifest_sha256 = _sha256(destination / ARTIFACT_MANIFEST)
    metadata = {
        "schema": "sfm-portable-simulator/v2",
        "package_kind": "simulated-interface-runtime",
        "entrypoint": "控制介面程式/影片模擬串流/選擇啟動.sh",
        "cli_entrypoint": "控制介面程式/影片模擬串流/啟動.sh",
        "install": "bash tools/install_runtime.sh",
        "fixed_assets": [
            *(artifact.path for artifact in artifacts),
            "模擬器/parrot_stimulate/src/anafi_pcmd_sim/scale_free_control.py",
        ],
        "source_manifest": {
            "scope": "git-source",
            "manifest": "MANIFEST.tsv",
            "sha256sums": "SHA256SUMS",
        },
        "runtime_artifacts": {
            "scope": "external-runtime-artifacts",
            "manifest": ARTIFACT_MANIFEST,
            "manifest_sha256": artifact_manifest_sha256,
            "artifact_count": len(artifacts),
        },
        "site_import": {
            "maps_root": "地圖檔/場域/<site>/",
            "videos_root": "模擬器/測試影片/",
            "requires_complete_site_package": True,
            "requires_site_profile": True,
        },
        "excluded_from_package": sorted(EXCLUDED_NAMES),
        "source_release": source_release,
        "supported_gpu": "NVIDIA RTX 5060 + CUDA 12.8 runtime (validated target)",
    }
    (destination / "PORTABLE_PACKAGE.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_export_destination(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)


def _copy_export_payload(
    destination: Path,
    resolved_artifacts: tuple[ResolvedRuntimeArtifact, ...],
) -> None:
    for relative in COPY_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in COPY_DIRS:
        source = ROOT / relative
        if not source.is_dir():
            raise FileNotFoundError(source)
        _copy_tree(source, destination / relative)
    runtime_source = ROOT / "執行環境"
    runtime_target = destination / "執行環境"
    runtime_target.mkdir(parents=True, exist_ok=True)
    for relative in ("requirements_runtime.txt", "requirements_test.txt", "README.md"):
        shutil.copy2(runtime_source / relative, runtime_target / relative)
    _copy_resolved_runtime_artifacts(resolved_artifacts, destination)


def export(
    destination: Path,
    *,
    artifact_root: str | Path | None = None,
    site_profiles: Iterable[str | Path] | None = None,
) -> list[str]:
    destination = destination.expanduser().resolve()
    if (
        destination == ROOT
        or ROOT.is_relative_to(destination)
        or destination.is_relative_to(ROOT)
    ):
        raise ValueError("destination and source workspace must not contain each other")
    source_release = _source_release()
    _prepare_export_destination(destination)
    resolved_artifacts = resolve_runtime_artifacts(ROOT, artifact_root=artifact_root)
    artifact_specs = tuple(artifact.spec for artifact in resolved_artifacts)
    site_bundle = (
        collect_site_bundle(ROOT, site_profiles)
        if site_profiles is not None
        else None
    )
    _copy_export_payload(destination, resolved_artifacts)
    if site_bundle is not None:
        _copy_site_bundle_files(ROOT, destination, site_bundle)
    _write_metadata(destination, source_release, artifact_specs)

    sys.path.insert(0, str(ROOT))
    from tools.package_manifest import generate

    entries = generate(destination, include_site_assets=site_bundle is not None)
    return [entry.path for entry in entries]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "seed root mirroring repository-relative runtime artifact paths "
            f"(or set {ARTIFACT_ROOT_ENV})"
        ),
    )
    parser.add_argument(
        "--site-profile",
        action="append",
        default=[],
        help=(
            "include a selected site profile and all digest-bound assets in the "
            "portable site bundle; repeat for multiple profiles"
        ),
    )
    args = parser.parse_args()
    try:
        files = export(
            args.destination,
            artifact_root=args.artifact_root,
            site_profiles=args.site_profile or None,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"portable package exported: {args.destination.resolve()}")
    print(f"manifest entries: {len(files)}")
    print("verify: python tools/package_manifest.py verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
