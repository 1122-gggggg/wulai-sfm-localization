#!/usr/bin/env python3
"""Build an offline workmap deploy pack from a completed, SHA-bound session.

The output is a new pack for tools/build_direct_site_release.py; this command
never selects a site release, updates a mission, or grants flight approval.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any
from build_direct_site_release import frame_digest, resolve_map_dir, sha256_file
from workmap_pack_assets import assemble_workmap_pack
from workmap_reconstruction import reconstruct_workmap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "定位演算法/deploy_code_expand/sfm_direct_deploy"))
from direct_map import DirectMapAssets  # noqa: E402


class WorkmapBuildError(ValueError):
    """A source binding or publishing precondition failed."""


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkmapBuildError(f"expected a JSON object: {path}")
    return value


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _base_release_identity(base_release: Path) -> tuple[str, Any, str, str]:
    """Bundle SHA, assets, model digest, and coordinate identity of the frozen base."""
    profile = _load_object(base_release / "site_profile.json")
    bundle_sha = profile["asset_sha256"]["localization_bundle"]
    bundle_path = base_release / "localization/direct_bundle.json"
    if sha256_file(bundle_path) != bundle_sha:
        raise WorkmapBuildError("base release bundle does not match site profile SHA-256")
    assets = DirectMapAssets.load(bundle_path, expected_sha256=bundle_sha)
    model_sha = frame_digest(assets.model_dir)
    identity = f"{assets.map_revision_id}_{model_sha[:12]}"
    poses = _load_object(base_release / "localization/reference_poses.json")
    if (
        profile.get("site_id") != assets.map_revision_id
        or profile.get("coordinate_frame", {}).get("id") != identity
        or assets.coordinate_frame_id != identity
        or poses.get("source_model_sha256") != model_sha
    ):
        raise WorkmapBuildError("base release identity does not match its COLMAP model bytes")
    return bundle_sha, assets, model_sha, identity


def _check_session_binding(
    metadata: dict[str, Any],
    snapshot: dict[str, Any],
    map_revision_id: str,
    identity: str,
    bundle_sha: str,
) -> None:
    if metadata.get("schema") != "direct-online-workmap-session/v1":
        raise WorkmapBuildError("unknown workmap session schema")
    for field, expected in (
        ("session_id", metadata.get("session_id")),
        ("base_release_id", map_revision_id),
        ("coordinate_frame_id", identity),
        ("base_bundle_sha256", bundle_sha),
        ("source_epoch", metadata.get("source_epoch")),
    ):
        if expected is None or metadata.get(field) != expected or snapshot.get(field) != expected:
            raise WorkmapBuildError(f"session and base release {field} do not match")
    if not isinstance(metadata["session_id"], str) or not metadata["session_id"]:
        raise WorkmapBuildError("session_id must be nonempty")


def _check_session_finished(metadata: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if (
        snapshot.get("schema") != metadata["schema"]
        or not isinstance(metadata.get("finished_unix_s"), (float, int))
        or not isinstance(snapshot.get("finished_unix_s"), (float, int))
        or metadata.get("writer_healthy") is not True
        or snapshot.get("statistics", {}).get("writer_healthy") is not True
        or snapshot["statistics"].get("writer_overflow_count") != 0
        or snapshot["statistics"].get("writer_error") is not None
    ):
        raise WorkmapBuildError("session is unfinished or its durable writer is unhealthy")
    if not isinstance(snapshot.get("keyframes"), dict) or not snapshot["keyframes"]:
        raise WorkmapBuildError("completed session has no durable keyframes")
    if not isinstance(snapshot.get("landmarks"), dict) or not snapshot["landmarks"]:
        raise WorkmapBuildError("completed session has no measured landmarks")


def validate_source(session_dir: Path, base_release: Path) -> dict[str, Any]:
    """Prove the session belongs to the exact, unmodified source bundle."""
    if _inside(session_dir, base_release):
        raise WorkmapBuildError("session must be outside the immutable base release")
    bundle_sha, assets, model_sha, identity = _base_release_identity(base_release)
    metadata = _load_object(session_dir / "session_meta.json")
    snapshot = _load_object(session_dir / "snapshot.json")
    _check_session_binding(metadata, snapshot, assets.map_revision_id, identity, bundle_sha)
    _check_session_finished(metadata, snapshot)
    for item in session_dir.rglob("*"):
        if item.is_symlink():
            raise WorkmapBuildError(f"session contains a symlink: {item}")
    return {"snapshot": snapshot, "base_bundle_sha256": bundle_sha, "base_model_sha256": model_sha}


def validate_pack(staging: Path, base_release: Path) -> None:
    """Refuse incomplete, linked or symlinked packs before publishing."""
    map_dir = resolve_map_dir(staging, "")
    required = (
        map_dir / "keyframes/keyframes.jsonl",
        map_dir / "localization/reference_manifest.jsonl",
        map_dir / "localization/boq_references_sideview.npy",
        map_dir / "localization/boq_references_sideview.names.json",
        map_dir / "localization/fim_lwtl_intersection_cells.json",
        map_dir / "experiments/edm_fixedpose_retriangulate/model/occupancy.json",
        staging / "README.md",
        staging / "P174_AND_NEXT.md",
    )
    if any(not path.is_file() for path in required):
        raise WorkmapBuildError("deploy pack is missing a required builder input")
    base_inodes = {
        (entry.stat().st_dev, entry.stat().st_ino)
        for entry in base_release.rglob("*")
        if entry.is_file()
    }
    for entry in staging.rglob("*"):
        if entry.is_symlink():
            raise WorkmapBuildError(f"deploy pack contains a symlink: {entry}")
        if entry.is_file() and (entry.stat().st_dev, entry.stat().st_ino) in base_inodes:
            raise WorkmapBuildError(f"deploy pack shares a frozen-base inode: {entry}")


def build(session_dir: Path, base_release: Path, output_pack: Path) -> dict[str, Any]:
    """Verify source, construct and validate an isolated pack, then publish it."""
    session_dir = session_dir.expanduser().resolve(strict=True)
    base_release = base_release.expanduser().resolve(strict=True)
    parent = output_pack.expanduser().parent.resolve(strict=True)
    output_pack = parent / output_pack.name
    if output_pack.exists() or output_pack.is_symlink():
        raise WorkmapBuildError(f"output pack already exists: {output_pack}")
    if any(
        _inside(output_pack, root) or _inside(root, output_pack)
        for root in (base_release, session_dir)
    ):
        raise WorkmapBuildError("output pack must be separate from the base release and session")
    context = validate_source(session_dir, base_release)
    staging = parent / f".staging-{output_pack.name}-{uuid.uuid4().hex}"
    model_dir = staging / "map/experiments/edm_fixedpose_retriangulate/model"
    try:
        staging.mkdir()
        model_receipt = reconstruct_workmap(
            base_release / "model", context["snapshot"], session_dir, model_dir
        )
        pack_receipt = assemble_workmap_pack(
            base_release, session_dir, model_dir, staging, model_receipt
        )
        if not isinstance(model_receipt, dict) or not isinstance(pack_receipt, dict):
            raise WorkmapBuildError("reconstruction and pack must return verifiable receipts")
        receipt = {
            "schema": "offline-workmap-pack/v1",
            "source_session_id": context["snapshot"]["session_id"],
            "base_release_id": context["snapshot"]["base_release_id"],
            "base_bundle_sha256": context["base_bundle_sha256"],
            "base_model_sha256": context["base_model_sha256"],
            "new_model_sha256": frame_digest(model_dir),
            "model": model_receipt,
            "pack": pack_receipt,
            "flight_approved": False,
            "mission_compatible": False,
        }
        intersection = pack_receipt.get("intersection_cells_status")
        if not isinstance(intersection, dict) or intersection.get("status") != "recomputed":
            raise WorkmapBuildError("pack has no newly scored FIM/LWTL safety-cell provenance")
        validate_pack(staging, base_release)
        (staging / "workmap_build_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output_pack.exists() or output_pack.is_symlink():
            raise WorkmapBuildError(f"output pack appeared during build: {output_pack}")
        os.rename(staging, output_pack)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, type=Path, help="completed session directory")
    parser.add_argument("--base-release", required=True, type=Path, help="immutable source release")
    parser.add_argument("--output-pack", required=True, type=Path, help="new deploy pack path")
    args = parser.parse_args(argv)
    try:
        receipt = build(args.session, args.base_release, args.output_pack)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"[workmap-pack] FAIL: {exc}\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
