#!/usr/bin/env python3
"""Enforce the authoritative deploy/flight compatibility-mirror policy.

Only ``MIRROR_PAIRS`` are byte-identical compatibility mirrors.  The three
same-named files in ``INTENTIONALLY_DIVERGENT`` are separate implementations
with documented reasons and must not be silently added to the mirror list.
``UNENFORCED_BUT_MUST_MATCH`` covers transitional files that are required to
remain byte-identical until an ownership decision is made.  This Python checker
is the only executable mirror policy; the former shell duplicate was removed.
"""
from __future__ import annotations

import argparse
from pathlib import Path


# (logical owner, compatibility mirror), relative to 定位演算法/.
# Files not listed here are independent even when their names happen to match.
MIRROR_PAIRS = (
    (
        "deploy_code/sfm_glomap_deploy/artifact_integrity.py",
        "flight_control/artifact_integrity.py",
    ),
    (
        "flight_control/autoflight.py",
        "deploy_code/sfm_glomap_deploy/autoflight.py",
    ),
    (
        "deploy_code/sfm_glomap_deploy/megaloc_cache.py",
        "flight_control/megaloc_cache.py",
    ),
    (
        "flight_control/path_follow_flight.py",
        "deploy_code/sfm_glomap_deploy/path_follow_flight.py",
    ),
    (
        "flight_control/plan_path.py",
        "deploy_code/sfm_glomap_deploy/plan_path.py",
    ),
    (
        "deploy_code/sfm_glomap_deploy/pose_types.py",
        "flight_control/pose_types.py",
    ),
    (
        "deploy_code/sfm_glomap_deploy/production_xfeat_tracker.py",
        "flight_control/production_xfeat_tracker.py",
    ),
    (
        "flight_control/real_path_follow_controller.py",
        "deploy_code/sfm_glomap_deploy/real_path_follow_controller.py",
    ),
)

# Same-named files that intentionally differ between the two trees.  Keep the
# rationale next to the executable policy so a new checker cannot silently
# treat one of these files as a compatibility mirror.
INTENTIONALLY_DIVERGENT = {
    "olympe_frame_source.py": "mission copy carries operator/SkyController integration",
    "reloc_localizer_xfeat.py": "different nesting depth requires different parents[N]",
    "README.md": "mission copy documents the nudge UI",
}

# Transitional same-named files that are byte-identical today but are not yet
# owned by a retained compatibility pair.
UNENFORCED_BUT_MUST_MATCH = frozenset({"manual_nudge_pilot.py"})


def algorithm_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def same_named_files(algorithm_root: str | Path | None = None) -> set[str]:
    root = (
        algorithm_root_from_file()
        if algorithm_root is None
        else Path(algorithm_root).expanduser().resolve()
    )
    flight_control = root / "flight_control"
    deploy = root / "deploy_code" / "sfm_glomap_deploy"
    if not flight_control.is_dir() or not deploy.is_dir():
        return set()
    flight_names = {path.name for path in flight_control.iterdir() if path.is_file()}
    deploy_names = {path.name for path in deploy.iterdir() if path.is_file()}
    return flight_names & deploy_names


def check_mirrors(algorithm_root: str | Path | None = None) -> list[str]:
    root = (
        algorithm_root_from_file()
        if algorithm_root is None
        else Path(algorithm_root).expanduser().resolve()
    )
    failures: list[str] = []
    for owner_rel, mirror_rel in MIRROR_PAIRS:
        owner = root / owner_rel
        mirror = root / mirror_rel
        if not owner.is_file():
            failures.append(f"missing owner: {owner_rel}")
            continue
        if not mirror.is_file():
            failures.append(f"missing mirror: {mirror_rel}")
            continue
        if owner.read_bytes() != mirror.read_bytes():
            failures.append(f"drift: {mirror_rel} != {owner_rel}")
    enforced_names = {
        Path(relative).name
        for pair in MIRROR_PAIRS
        for relative in pair
    }
    classified_names = enforced_names | set(INTENTIONALLY_DIVERGENT) | set(
        UNENFORCED_BUT_MUST_MATCH
    )
    for name in sorted(same_named_files(root) - classified_names):
        failures.append(
            "unclassified same-named file: "
            f"{name}; add it to MIRROR_PAIRS, INTENTIONALLY_DIVERGENT, or "
            "UNENFORCED_BUT_MUST_MATCH"
        )
    for name in sorted(UNENFORCED_BUT_MUST_MATCH):
        owner = root / "flight_control" / name
        mirror = root / "deploy_code" / "sfm_glomap_deploy" / name
        if owner.is_file() and mirror.is_file() and owner.read_bytes() != mirror.read_bytes():
            failures.append(f"drift: {name} is transitional but must remain identical")
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
            print(f"[mirror-check] {failure}")
        raise SystemExit(1)
    print(f"[mirror-check] OK: {len(MIRROR_PAIRS)} compatibility pairs")


if __name__ == "__main__":
    main()
