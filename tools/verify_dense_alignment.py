#!/usr/bin/env python3
"""Prove a dense cloud lies in a COLMAP model's frame, from the keyframes' own geometry.

The coarse cell-overlap check in ``build_dense_display_ply`` cannot tell a cloud that
is a few map units off (or rotated) from a correct one: a mismatched Wulai cloud passed
it. Waypoints are drawn on the dense cloud and flown in the localization frame, so a
mismatch is a flight hazard, not a cosmetic bug.

Here every sampled registered image contributes the 3D points it observed. Those points
are visible from that image by construction, so a cloud in the same frame must have its
nearest surface at (almost) the same depth along the same pixel. A cloud-to-image
z-buffer gives that surface; the score is the fraction of observed points whose depth
agrees within ``DEPTH_TOLERANCE``. Measured on Wulai all9: the correct MVSFormer++
cloud scores 0.60-0.63, the model's own 147k-point map.ply 0.37 (it is sparser), the
mismatched earlier cloud 0.02.

Usage
-----
    .venv/bin/python tools/verify_dense_alignment.py \\
        --model 地圖檔/場域/wulai_all9_direct_20261019/model --cloud <dense.ply>

Exit status is non-zero when the score is below ``--min-fraction``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_OPERATOR = REPO_ROOT / "控制介面程式" / "operator_interface"
if str(_OPERATOR) not in sys.path:
    sys.path.insert(0, str(_OPERATOR))

from map_point_io import read_ply_points  # noqa: E402

DEPTH_TOLERANCE = 0.05  # |ln(z_observed / z_cloud)|, i.e. about 5 %
MIN_OBSERVATIONS = 150  # images with fewer observed points are not informative
DEFAULT_VIEWS = 60
DEFAULT_CLOUD_POINTS = 3_000_000
DEFAULT_MIN_FRACTION = 0.30
RENDER_WIDTH = 640
MIN_DEPTH = 0.3


def depth_agreement(
    cloud_xyz: np.ndarray, views: list[dict]
) -> tuple[float, int]:
    """(fraction of observed points whose depth the cloud reproduces, observations).

    A view is ``{"R": 3x3, "t": 3, "K": (fx, fy, cx, cy), "size": (W, H), "X": Nx3}``
    with ``R, t`` mapping world to camera. Observed points that no cloud point
    covers count as disagreeing, so a cloud that simply misses the scene cannot win.
    """
    cloud = np.asarray(cloud_xyz, dtype=np.float64).reshape(-1, 3)
    agree = 0
    total = 0
    for view in views:
        width, height = view["size"]
        fx, fy, cx, cy = view["K"]
        camera = cloud @ view["R"].T + view["t"]
        z = camera[:, 2]
        keep = z > MIN_DEPTH
        u = np.rint(fx * camera[keep, 0] / z[keep] + cx).astype(np.int64)
        v = np.rint(fy * camera[keep, 1] / z[keep] + cy).astype(np.int64)
        z = z[keep]
        inside = (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
        u, v, z = u[inside], v[inside], z[inside]
        buffer = np.full(width * height, np.inf)
        # Far to near, so the nearest point is the last write to each pixel.
        order = np.argsort(-z)
        u, v, z = u[order], v[order], z[order]
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                buffer[(v + dv) * width + (u + du)] = z
        observed = np.asarray(view["X"], dtype=np.float64) @ view["R"].T + view["t"]
        ze = observed[:, 2]
        ok = ze > MIN_DEPTH
        ue = np.clip(np.rint(fx * observed[ok, 0] / ze[ok] + cx).astype(np.int64), 0, width - 1)
        ve = np.clip(np.rint(fy * observed[ok, 1] / ze[ok] + cy).astype(np.int64), 0, height - 1)
        surface = buffer[ve * width + ue]
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.abs(np.log(ze[ok] / surface))
        agree += int((error < DEPTH_TOLERANCE).sum())
        total += int(ok.sum())
    if total == 0:
        raise ValueError("no observed points to compare against")
    return agree / total, total


def load_views(model: Path, count: int = DEFAULT_VIEWS, seed: int = 0) -> list[dict]:
    """Sample registered images and the 3D points each one observed (needs pycolmap)."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model))
    informative = [
        image_id
        for image_id, image in reconstruction.images.items()
        if sum(1 for p in image.points2D if p.has_point3D()) >= MIN_OBSERVATIONS
    ]
    if not informative:
        raise ValueError(f"no image in {model} observes {MIN_OBSERVATIONS}+ points")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(informative, min(count, len(informative)), replace=False)
    views = []
    for image_id in chosen:
        image = reconstruction.images[int(image_id)]
        camera = reconstruction.cameras[image.camera_id]
        if str(camera.model) not in {"CameraModelId.PINHOLE", "PINHOLE"}:
            raise ValueError(f"only PINHOLE cameras are supported, got {camera.model}")
        scale = RENDER_WIDTH / camera.width
        fx, fy, cx, cy = (float(p) * scale for p in camera.params[:4])
        matrix = np.asarray(image.cam_from_world().matrix())
        observed = np.array(
            [reconstruction.points3D[p.point3D_id].xyz for p in image.points2D if p.has_point3D()]
        )
        views.append(
            {
                "R": matrix[:, :3],
                "t": matrix[:, 3],
                "K": (fx, fy, cx, cy),
                "size": (RENDER_WIDTH, int(round(camera.height * scale))),
                "X": observed,
            }
        )
    return views


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="COLMAP model folder (.bin)")
    parser.add_argument("--cloud", type=Path, required=True, help="dense PLY to check")
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS)
    parser.add_argument("--cloud-points", type=int, default=DEFAULT_CLOUD_POINTS)
    parser.add_argument("--min-fraction", type=float, default=DEFAULT_MIN_FRACTION)
    args = parser.parse_args(argv)
    cloud = read_ply_points(args.cloud, args.cloud_points)[:, :3]
    views = load_views(args.model, args.views)
    fraction, observations = depth_agreement(cloud, views)
    passed = fraction >= args.min_fraction
    print(
        json.dumps(
            {
                "cloud": str(args.cloud),
                "cloud_points_sampled": int(len(cloud)),
                "views": len(views),
                "observations": observations,
                "depth_tolerance": DEPTH_TOLERANCE,
                "agreement_fraction": round(fraction, 4),
                "min_fraction": args.min_fraction,
                "passed": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
