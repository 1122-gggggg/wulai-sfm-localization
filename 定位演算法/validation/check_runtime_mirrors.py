#!/usr/bin/env python3
"""Fail when an explicitly retained compatibility mirror drifts from its owner."""
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


def algorithm_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


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
