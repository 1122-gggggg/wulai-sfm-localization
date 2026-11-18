#!/usr/bin/env python3
"""Swap the flight route of a hash-pinned mission selection, fail closed.

Performs the documented manual steps in one command: hash the new route file,
rewrite the route package to point at it, hash the route package, rewrite the
selection's route reference, then resolve the mission in-process.  Nothing is
written unless the new mission resolves with no blocker that the current one
does not already have; a failure after writing restores both original files.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from pin_mission_selection import ROOT, file_reference

from mission_manifest import ManifestError, clear_sha256_cache, sha256_file
from mission_resolver import ResolvedMission, resolve_mission

ROUTE_PACKAGE_SCHEMA = "sfm-route-package/v1"
#: File name convention of the route editor: ``flight_route_<revision>.json``.
ROUTE_FILE_PREFIX = "flight_route_"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object: {path}")
    return value


def _dump(document: dict[str, Any]) -> str:
    # Key order is preserved so the diff of a committed manifest is only the changed values.
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def _workspace_file(value: str | Path, *, root: Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ManifestError(f"{label} must stay inside the workspace: {path}")
    if not path.is_file() or path.is_symlink():
        raise ManifestError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _baseline_errors(selection: Path, *, root: Path) -> frozenset[str] | None:
    """Flight blockers the selection already has, or None if it does not even parse.

    Artifact hashes are deliberately not checked here: the route being replaced may be
    stale or edited, and this only decides which blockers are not caused by the swap.
    """
    try:
        mission = resolve_mission(selection, workspace_root=root, verify_files=False)
    except ManifestError:
        return None
    return frozenset(mission.readiness.flight_errors)


def _blockers(mission: ResolvedMission, baseline: frozenset[str] | None) -> list[str]:
    """Reasons the swapped mission must not be written."""
    readiness = mission.readiness
    if not readiness.localization_ready:
        return list(readiness.localization_errors)
    known = baseline or frozenset()
    return [error for error in readiness.flight_errors if error not in known]


def _write_atomic(path: Path, content: bytes | str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if isinstance(content, bytes):
            temporary.write_bytes(content)
        else:
            temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stale_selections(selection: Path, package: Path, package_sha256: str) -> list[str]:
    """Other selections in the same directory bound to *package* with an outdated hash."""
    stale: list[str] = []
    for candidate in sorted(selection.parent.glob("*.json")):
        if candidate == selection:
            continue
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
            reference = document["route"]
            bound = (candidate.parent / reference["path"]).resolve()
            bound_sha256 = reference["sha256"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if bound == package and bound_sha256 != package_sha256:
            stale.append(str(candidate))
    return stale


def _validate_candidate(
    selection: Path,
    selection_document: dict[str, Any],
    package: Path,
    package_text: str,
    *,
    baseline: frozenset[str] | None,
    root: Path,
) -> tuple[str, ResolvedMission]:
    """Resolve the swapped mission from side files; the real files are not touched.

    Candidates sit next to the real files so every relative path resolves the same way;
    the candidate selection binds the candidate package until the real files are replaced.
    """
    # A unique token keeps the in-process SHA-256 memo from ever matching an older candidate.
    token = f"{os.getpid()}.{secrets.token_hex(4)}"
    candidate_package = package.with_name(f".{package.name}.{token}.candidate")
    candidate_selection = selection.with_name(f".{selection.name}.{token}.candidate")
    try:
        candidate_package.write_text(package_text, encoding="utf-8")
        # Step 3: hash the route package.
        package_sha256 = sha256_file(candidate_package)
        # Step 4: bind the selection to the new package hash.
        probe = dict(selection_document)
        probe["route"] = {
            **selection_document["route"],
            "path": os.path.relpath(candidate_package, selection.parent),
            "sha256": package_sha256,
        }
        candidate_selection.write_text(_dump(probe), encoding="utf-8")
        # Step 5: resolver validation, before anything real is touched.
        mission = resolve_mission(candidate_selection, workspace_root=root)
        blockers = _blockers(mission, baseline)
        if blockers:
            raise ManifestError("new route is not acceptable: " + "; ".join(blockers))
        return package_sha256, mission
    finally:
        candidate_package.unlink(missing_ok=True)
        candidate_selection.unlink(missing_ok=True)


def update_route(
    selection_path: str | Path,
    route_path: str | Path,
    *,
    root: Path = ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebind *selection_path* to *route_path*; return the report printed by ``main``."""
    root = root.resolve()
    selection = _workspace_file(selection_path, root=root, label="selection")
    clear_sha256_cache()

    selection_document = _read_object(selection, "selection")
    route_reference = selection_document.get("route")
    if not isinstance(route_reference, dict) or not isinstance(route_reference.get("path"), str):
        raise ManifestError("selection has no route package to update")
    package = _workspace_file(
        selection.parent / route_reference["path"], root=root, label="route package"
    )
    package_document = _read_object(package, "route package")
    if package_document.get("schema") != ROUTE_PACKAGE_SCHEMA:
        raise ManifestError(f"route package schema must be {ROUTE_PACKAGE_SCHEMA!r}")

    route = _workspace_file(route_path, root=root, label="route")
    baseline = _baseline_errors(selection, root=root)
    original_selection = selection.read_bytes()
    original_package = package.read_bytes()

    # Steps 1-2: hash the new route file and point the route package at it.
    new_package = dict(package_document)
    new_package["route"] = file_reference(route, base=package.parent, label="route", root=root)
    if route.name.startswith(ROUTE_FILE_PREFIX) and route.suffix == ".json":
        new_package["revision"] = route.name[len(ROUTE_FILE_PREFIX) : -len(route.suffix)]
    package_text = _dump(new_package)

    package_sha256, mission = _validate_candidate(
        selection,
        selection_document,
        package,
        package_text,
        baseline=baseline,
        root=root,
    )

    final_selection = dict(selection_document)
    final_selection["route"] = {**route_reference, "sha256": package_sha256}
    if not dry_run:
        try:
            _write_atomic(package, package_text)
            _write_atomic(selection, _dump(final_selection))
            mission = resolve_mission(selection, workspace_root=root)
            blockers = _blockers(mission, baseline)
            if blockers:
                raise ManifestError("updated selection failed validation: " + "; ".join(blockers))
        except BaseException:
            _write_atomic(package, original_package)
            _write_atomic(selection, original_selection)
            raise

    return {
        "selection": str(selection),
        "route_package": str(package),
        "route": str(route),
        "route_revision": new_package["revision"],
        "route_sha256": new_package["route"]["sha256"],
        "route_package_sha256": package_sha256,
        "localization_ready": mission.readiness.localization_ready,
        "flight_ready": mission.readiness.flight_ready,
        "flight_errors": list(mission.readiness.flight_errors),
        "stale_selections": _stale_selections(selection, package, package_sha256),
        "dry_run": dry_run,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="sfm-mission-selection/v1 to update")
    parser.add_argument("--route", required=True, help="new sfm-flight-route/v1 JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the swap and print the report without writing any file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = update_route(args.selection, args.route, dry_run=args.dry_run)
    except (ManifestError, OSError, ValueError) as exc:
        print(f"cannot update mission route: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
