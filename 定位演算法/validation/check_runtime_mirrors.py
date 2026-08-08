#!/usr/bin/env python3
"""Enforce one authoritative implementation for each shared runtime module.

The flight and deployment directories are both added to ``sys.path`` by live
entry points.  A module with the same name in both directories is therefore
order-dependent and can drift silently.  Keep exactly one copy of every shared
runtime module and make its owner explicit here.

The file name and ``check_mirrors`` function remain for compatibility with CI
and existing callers; the policy now rejects mirrors instead of comparing them.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FLIGHT_PREFIX = Path("flight_control")
DEPLOY_PREFIX = Path("deploy_code/sfm_glomap_deploy")

# Module name -> authoritative path, relative to 定位演算法/.
CANONICAL_MODULES = {
    "artifact_integrity.py": DEPLOY_PREFIX / "artifact_integrity.py",
    "autoflight.py": FLIGHT_PREFIX / "autoflight.py",
    "manual_nudge_pilot.py": FLIGHT_PREFIX / "manual_nudge_pilot.py",
    "megaloc_cache.py": DEPLOY_PREFIX / "megaloc_cache.py",
    "olympe_frame_source.py": FLIGHT_PREFIX / "olympe_frame_source.py",
    "path_follow_flight.py": FLIGHT_PREFIX / "path_follow_flight.py",
    "plan_path.py": FLIGHT_PREFIX / "plan_path.py",
    "pose_types.py": DEPLOY_PREFIX / "pose_types.py",
    "production_xfeat_tracker.py": DEPLOY_PREFIX / "production_xfeat_tracker.py",
    "real_path_follow_controller.py": FLIGHT_PREFIX / "real_path_follow_controller.py",
    "reloc_localizer_xfeat.py": DEPLOY_PREFIX / "reloc_localizer_xfeat.py",
}

# Documentation describes different directory responsibilities, so the two
# README files intentionally share a name while containing different material.
INTENTIONALLY_DIVERGENT = frozenset({"README.md"})


def algorithm_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def same_named_files(algorithm_root: str | Path | None = None) -> set[str]:
    root = _resolve_algorithm_root(algorithm_root)
    flight_control = root / FLIGHT_PREFIX
    deploy = root / DEPLOY_PREFIX
    if not flight_control.is_dir() or not deploy.is_dir():
        return set()
    flight_names = {path.name for path in flight_control.iterdir() if path.is_file()}
    deploy_names = {path.name for path in deploy.iterdir() if path.is_file()}
    return flight_names & deploy_names


def _resolve_algorithm_root(algorithm_root: str | Path | None) -> Path:
    if algorithm_root is None:
        return algorithm_root_from_file()
    return Path(algorithm_root).expanduser().resolve()


def _other_runtime_path(canonical: Path) -> Path:
    if canonical.parent == FLIGHT_PREFIX:
        return DEPLOY_PREFIX / canonical.name
    return FLIGHT_PREFIX / canonical.name


def check_mirrors(algorithm_root: str | Path | None = None) -> list[str]:
    """Return ownership-policy violations; an empty list means the tree is valid."""
    root = _resolve_algorithm_root(algorithm_root)
    failures: list[str] = []
    for name, canonical_rel in CANONICAL_MODULES.items():
        canonical = root / canonical_rel
        duplicate_rel = _other_runtime_path(canonical_rel)
        if not canonical.is_file():
            failures.append(f"missing canonical module: {canonical_rel}")
        if (root / duplicate_rel).exists():
            failures.append(
                f"duplicate runtime module: {duplicate_rel}; canonical is {canonical_rel}"
            )
        if canonical_rel.name != name:
            failures.append(
                f"invalid ownership entry: key {name} does not match {canonical_rel.name}"
            )

    unexpected = same_named_files(root) - INTENTIONALLY_DIVERGENT
    for name in sorted(unexpected):
        failures.append(f"unclassified same-named runtime file: {name}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm-root",
        default="",
        help="override the 定位演算法 directory used for validation",
    )
    args = parser.parse_args()
    failures = check_mirrors(args.algorithm_root or None)
    if failures:
        for failure in failures:
            print(f"[module-ownership] {failure}")
        raise SystemExit(1)
    print(f"[module-ownership] OK: {len(CANONICAL_MODULES)} canonical modules; no runtime mirrors")


if __name__ == "__main__":
    main()
