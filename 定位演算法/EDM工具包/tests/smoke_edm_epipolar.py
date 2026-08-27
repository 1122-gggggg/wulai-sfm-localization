#!/usr/bin/env python3
"""Smoke test: does EDM actually match THIS site's images, and is the cell trick exact?

Not a "did it run" test. It checks the two things the whole design rests on:
  1. GEOMETRY: matches are verified against the site's known GLUEMAP poses via the
     symmetric epipolar distance. A matcher that "runs" but matches noise shows up here.
  2. CELL TRICK: every match has exactly one side sitting on the 8px coarse grid, and
     round(kpt/8) recovers that cell exactly. The map build and the localizer both
     depend on this; if it fails, 3D lookup by cell id is unsound.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
from edm_matcher import EDM_H, EDM_W, EDMMatcher  # noqa: E402

def cam_K(cam) -> np.ndarray:
    fx, fy, cx, cy = cam.params
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def symmetric_epipolar(k0, k1, K, T0, T1):
    """k0,k1 in FULL-res (1280x720) pixels; T = 3x4 world->cam. -> per-match distance (px)."""
    R0, t0 = T0[:3, :3], T0[:3, 3]
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R = R1 @ R0.T
    t = t1 - R @ t0
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K).T @ (tx @ R) @ np.linalg.inv(K)
    x0 = np.hstack([k0, np.ones((len(k0), 1))])
    x1 = np.hstack([k1, np.ones((len(k1), 1))])
    Fx0 = (F @ x0.T).T
    Ftx1 = (F.T @ x1.T).T
    num = np.sum(x1 * Fx0, axis=1) ** 2
    d = num * (1.0 / (Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2) + 1.0 / (Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2))
    return np.sqrt(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-root", required=True)
    args = parser.parse_args()
    model = Path(args.model).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()

    rec = pycolmap.Reconstruction(str(model))
    cam = list(rec.cameras.values())[0]
    K = cam_K(cam)
    scale = cam.width / EDM_W  # EDM-res -> full-res (1280/1024 = 1.25)

    # covisibility from the model: pick pairs sharing many 3D points
    imgs = {im.name: im for im in rec.images.values()}
    pts_of = {n: {p.point3D_id for p in im.points2D if p.has_point3D()} for n, im in imgs.items()}
    names = sorted(imgs)
    anchor = names[len(names) // 2]
    shared = sorted(((len(pts_of[anchor] & pts_of[n]), n) for n in names if n != anchor), reverse=True)
    pairs = [(anchor, shared[0][1]), (anchor, shared[10][1]), (anchor, shared[40][1])]

    m = EDMMatcher(mconf_thr=0.2)
    print(f"EDM {EDM_W}x{EDM_H}, topk={m.topk}, mconf_thr={m.mconf_thr}\n")

    ok = True
    for n0, n1 in pairs:
        covis = len(pts_of[n0] & pts_of[n1])
        r = m.match(image_root / n0, image_root / n1)
        k0, k1, mc = r["mkpts0"], r["mkpts1"], r["mconf"]
        if len(k0) < 10:
            print(f"{n0} <-> {n1}: only {len(k0)} matches  FAIL")
            ok = False
            continue

        # pycolmap 4.x (rig model): cam_from_world is a method, not a property.
        T0 = imgs[n0].cam_from_world().matrix()
        T1 = imgs[n1].cam_from_world().matrix()
        d = symmetric_epipolar(k0 * scale, k1 * scale, K, T0, T1)
        med = float(np.median(d))
        in1 = float((d < 1.0).mean()) * 100
        in3 = float((d < 3.0).mean()) * 100

        # cell trick: exactly one side of each match must lie on the coarse grid
        ref0 = EDMMatcher.is_refined(k0)
        ref1 = EDMMatcher.is_refined(k1)
        one_side_on_grid = float((ref0 ^ ref1).mean()) * 100
        # round-trip: recovered cell must equal the grid side's own cell
        grid_side = np.where(ref0[:, None], k1, k0)
        cells = EDMMatcher.cell_ids(grid_side)
        exact = np.rint(grid_side / 8).astype(np.int64)
        rt = float((cells == (exact[:, 1] * (EDM_W // 8) + exact[:, 0])).mean()) * 100

        print(f"{n0} <-> {n1}  covis3D={covis}")
        print(f"  matches={len(k0):5d}  epipolar median={med:5.2f}px  <1px={in1:5.1f}%  <3px={in3:5.1f}%")
        print(f"  cell trick: one-side-on-grid={one_side_on_grid:5.1f}%  cell round-trip exact={rt:5.1f}%")
        if med > 3.0 or in3 < 60 or one_side_on_grid < 99 or rt < 99.9:
            ok = False
            print("  ^^ FAIL")
        print()

    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
