#!/usr/bin/env python3
"""Measure which way is UP in a GLOMAP reconstruction, and write T_align_gravity.json.

A GLOMAP/COLMAP reconstruction is gauge-free: it is determined only up to a
similarity transform, so its raw XYZ axes carry no physical meaning at all --
they fall out of whichever camera happened to initialise the model. Assuming one
of them is vertical is how target_site_v1 ended up 22.5 degrees off.

Gravity IS recoverable from the camera poses, given one physical fact about how
the data was captured: the gimbal holds roll near zero, so every camera's right
axis is horizontal. The direction orthogonal to all of them is gravity, obtained
as the smallest eigenvector of sum(r_i r_i^T).

What this CANNOT recover:
  * azimuth -- rotating the whole reconstruction about gravity reproduces it
    exactly, so "which way is north" is simply not in the data. The convention
    here (keep the GLOMAP +X azimuth) is an arbitrary but fixed reference; the
    real heading is resolved at runtime by HeadingEstimator against Olympe yaw.
  * scale -- unchanged by this transform, and deliberately not used downstream.

    python 定位演算法/validation/derive_map_gravity.py --poses <ref_poses.json>
    python 定位演算法/validation/derive_map_gravity.py --poses <...> --check
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

#: Below this many poses the eigen-decomposition is not a measurement.
MIN_POSES = 100
#: p95 of |r_i . g|. Above this the "gimbal roll ~ 0" premise did not hold, so the
#: result is not trustworthy and writing it would launder a guess into an asset.
MAX_RESIDUAL_DEG = 5.0
#: |mean(camera_down . g)|. Near zero the sign of the eigenvector is undetermined
#: and "up" could come out inverted, which is worse than having no file at all.
MIN_SIGN_AGREEMENT = 0.30
#: Tolerance when --check compares a stored file against a fresh derivation.
CHECK_TOLERANCE_DEG = 0.05


def _reject_json_constant(value: str):
    raise ValueError(f"reference poses must not contain {value}")


def load_rotations(poses_json: Path) -> np.ndarray:
    data = json.loads(poses_json.read_text(encoding="utf-8"),
                      parse_constant=_reject_json_constant)
    poses = data.get("poses") if isinstance(data, dict) else None
    if not isinstance(poses, dict) or not poses:
        raise ValueError(f"reference poses file has no 'poses' object: {poses_json}")
    rotations = []
    for name, pose in poses.items():
        raw = pose.get("R") if isinstance(pose, dict) else None
        if raw is None:
            raise ValueError(f"pose {name!r} has no rotation R")
        R = np.asarray(raw, dtype=float).reshape(3, 3)
        if not np.isfinite(R).all():
            raise ValueError(f"pose {name!r} has a non-finite rotation")
        rotations.append(R)
    return np.asarray(rotations)


def camera_centres(poses_json: Path) -> np.ndarray:
    """C = -R^T t, the camera position in world coordinates."""
    data = json.loads(poses_json.read_text(encoding="utf-8"),
                      parse_constant=_reject_json_constant)
    centres = []
    for name, pose in data["poses"].items():
        R = np.asarray(pose["R"], dtype=float).reshape(3, 3)
        t = np.asarray(pose["t"], dtype=float).reshape(3)
        if not np.isfinite(t).all():
            raise ValueError(f"pose {name!r} has a non-finite translation")
        centres.append(-R.T @ t)
    return np.asarray(centres)


def derive_gravity(rotations: np.ndarray) -> dict:
    """Return the measured gravity (DOWN) direction plus its quality evidence."""
    if len(rotations) < MIN_POSES:
        raise ValueError(
            f"need at least {MIN_POSES} poses to measure gravity, got {len(rotations)}"
        )
    right = rotations[:, 0]                      # camera +X in world coordinates
    down = rotations[:, 1]                       # camera +Y in world coordinates
    scatter = np.einsum("ni,nj->ij", right, right)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    gravity = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])

    # The eigenvector's sign is arbitrary. Resolve it against the cameras' own
    # down axis, which is physical -- NOT against a world axis, since assuming a
    # world axis points down is exactly the error this tool exists to remove.
    agreement = float(np.mean(down @ gravity))
    if agreement < 0.0:
        gravity, agreement = -gravity, -agreement
    if agreement < MIN_SIGN_AGREEMENT:
        raise ValueError(
            "cannot determine which end of the gravity axis is DOWN: mean camera "
            f"down-axis agreement {agreement:.3f} < {MIN_SIGN_AGREEMENT}"
        )

    residual = np.abs(right @ gravity)
    residual_p95 = float(np.percentile(residual, 95))
    residual_deg = math.degrees(math.asin(min(1.0, residual_p95)))
    if residual_deg > MAX_RESIDUAL_DEG:
        raise ValueError(
            f"camera right axes are not horizontal enough: p95 residual "
            f"{residual_deg:.2f} deg > {MAX_RESIDUAL_DEG} deg. The gimbal-roll~0 "
            "premise does not hold for this capture; gravity was NOT measured."
        )
    return {
        "gravity": gravity,
        "n_poses": int(len(rotations)),
        "residual_p95": residual_p95,
        "residual_deg": residual_deg,
        "sign_agreement": agreement,
        "eigenvalues": eigenvalues.tolist(),
    }


def basis_from_gravity(gravity: np.ndarray) -> np.ndarray:
    """Rows: [east, north, up]. east keeps the GLOMAP +X azimuth, east x north = up."""
    up = -np.asarray(gravity, dtype=float)
    up = up / np.linalg.norm(up)
    x_axis = np.array([1.0, 0.0, 0.0])
    east = x_axis - float(np.dot(x_axis, up)) * up
    norm = float(np.linalg.norm(east))
    if norm < 1e-6:
        raise ValueError("gravity is parallel to GLOMAP +X; the azimuth reference "
                         "is degenerate and no heading convention can be defined")
    east = east / norm
    return np.vstack([east, np.cross(up, east), up])


def ground_percentile(ply_path: Path, R: np.ndarray) -> float | None:
    """1st percentile of the cloud's aligned Z, i.e. roughly the ground plane."""
    with ply_path.open("rb") as handle:
        header, count, binary = [], 0, False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY has no end_header: {ply_path}")
            text = line.decode("ascii", "replace").strip()
            header.append(text)
            if text.startswith("element vertex"):
                count = int(text.split()[-1])
            if text.startswith("format"):
                binary = "ascii" not in text
            if text == "end_header":
                break
        if binary:
            print(f"[gravity] {ply_path.name} is binary PLY; skipping ground_z_p01",
                  file=sys.stderr)
            return None
        zs = []
        for _ in range(count):
            parts = handle.readline().split()
            if len(parts) < 3:
                break
            point = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            zs.append(float(R[2] @ point))
    return float(np.percentile(zs, 1)) if zs else None


def build_document(poses_json: Path, frame_name: str, ply: Path | None) -> dict:
    rotations = load_rotations(poses_json)
    measured = derive_gravity(rotations)
    gravity = measured["gravity"]
    R = basis_from_gravity(gravity)

    centres = camera_centres(poses_json)
    aligned = centres @ R.T
    extent = (aligned.max(axis=0) - aligned.min(axis=0)).tolist()

    legacy_up = np.array([0.0, -1.0, 0.0])
    deviation = math.degrees(math.acos(
        float(np.clip(np.dot(R[2], legacy_up), -1.0, 1.0))))

    aligned_extent = {
        "camera_bbox": extent,
        "height_range": extent[2],
    }
    if ply is not None:
        ground = ground_percentile(ply, R)
        if ground is not None:
            aligned_extent["ground_z_p01"] = ground

    return {
        "schema": "sfm-align/v2",
        "frame_from": f"{frame_name}_glomap",
        "frame_to": f"{frame_name}_gravity_aligned_Zup",
        "R": R.tolist(),
        "t": [0.0, 0.0, 0.0],
        "note": ("p_aligned = R @ p_glomap. Z is up (anti-gravity), "
                 "X keeps the GLOMAP +X azimuth."),
        "gravity_glomap": gravity.tolist(),
        "derivation": {
            "method": "smallest eigenvector of sum(right_i right_i^T) over camera X axes",
            "assumption": "gimbal holds roll ~0, so every camera right axis is horizontal",
            "sign_convention": ("gravity points DOWN, resolved against the mean camera "
                                "down axis (never against a world axis)"),
            "n_poses": measured["n_poses"],
            "residual_right_dot_g_p95": measured["residual_p95"],
            "residual_deg_p95": measured["residual_deg"],
            "sign_agreement": measured["sign_agreement"],
            "deviation_from_old_Yup_deg": deviation,
            "source_poses": poses_json.name,
            "tool": "定位演算法/validation/derive_map_gravity.py",
        },
        "aligned_extent": aligned_extent,
    }


def compare(existing: dict, fresh: dict) -> list[str]:
    problems = []
    if existing.get("schema") != fresh["schema"]:
        problems.append(f"schema {existing.get('schema')!r} != {fresh['schema']!r}")
    # R is what every consumer actually applies. Comparing only gravity_glomap let
    # --check pass a document whose published rotation contradicts the direction it
    # claims to be derived from.
    old_r = np.asarray(existing.get("R", []), dtype=float)
    new_r = np.asarray(fresh["R"], dtype=float)
    if old_r.shape != (3, 3):
        problems.append("stored R is missing or not 3x3")
    elif not np.allclose(old_r, new_r, atol=1e-6):
        worst = float(np.max(np.abs(old_r - new_r)))
        problems.append(f"stored R differs from a fresh derivation (max {worst:.2e})")
    old_t = np.asarray(existing.get("t", [0.0, 0.0, 0.0]), dtype=float)
    if old_t.shape != (3,) or not np.allclose(old_t, 0.0, atol=1e-9):
        problems.append("stored t must be zero: the alignment is a pure rotation")
    old_g = np.asarray(existing.get("gravity_glomap", [0.0, 0.0, 0.0]), dtype=float)
    new_g = np.asarray(fresh["gravity_glomap"], dtype=float)
    if float(np.linalg.norm(old_g)) < 1e-9:
        problems.append("stored gravity_glomap is missing or zero")
    else:
        angle = math.degrees(math.acos(float(np.clip(
            np.dot(old_g / np.linalg.norm(old_g), new_g), -1.0, 1.0))))
        if angle > CHECK_TOLERANCE_DEG:
            problems.append(
                f"stored gravity differs from a fresh derivation by {angle:.3f} deg"
            )
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", required=True,
                        help="site reference poses JSON (COLMAP world->camera R, t)")
    parser.add_argument("--out", default="",
                        help="output path (default: T_align_gravity.json beside --poses)")
    parser.add_argument("--frame-name", default="",
                        help="frame id prefix (default: derived from --poses)")
    parser.add_argument("--ply", default="",
                        help="optional ASCII dense cloud, for aligned_extent.ground_z_p01")
    parser.add_argument("--check", action="store_true",
                        help="verify the existing file matches a fresh derivation; write nothing")
    args = parser.parse_args(argv)

    poses = Path(args.poses).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out \
        else poses.parent / "T_align_gravity.json"
    name = args.frame_name or poses.stem.replace("_ref_poses", "")
    ply = Path(args.ply).expanduser().resolve() if args.ply else None

    document = build_document(poses, name, ply)
    derivation = document["derivation"]
    print(f"[gravity] {poses.name}: n={derivation['n_poses']} "
          f"residual_p95={derivation['residual_deg_p95']:.2f}deg "
          f"sign_agreement={derivation['sign_agreement']:.3f} "
          f"deviation_from_legacy_Yup={derivation['deviation_from_old_Yup_deg']:.2f}deg")

    if args.check:
        if not out.is_file():
            print(f"[gravity] FAIL: {out} does not exist", file=sys.stderr)
            return 1
        problems = compare(json.loads(out.read_text(encoding="utf-8")), document)
        for problem in problems:
            print(f"[gravity] FAIL: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"[gravity] OK: {out.name} matches a fresh derivation")
        return 0

    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[gravity] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
