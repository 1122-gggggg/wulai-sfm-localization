#!/usr/bin/env python3
"""How much do a cell's subpixel observations actually disagree?

The map build assumed a cell's refined observations across pairs all describe the same
physical point, so it averaged them and dropped 'ambiguous' cells. But EDM's refinement
is anchored on the PARTNER's cell centre, so different partners legitimately refine to
slightly different sub-locations. This measures the real spread, so the merge policy is
set by data instead of by assumption.

Also measures how far the mean refined position sits from the cell centre, which is the
systematic 2D-3D error the localizer would inherit if the 3D is keyed to the mean but
the runtime match anchors on the centre.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
from edm_matcher import COARSE_STRIDE, GRID_W, EDMMatcher  # noqa: E402

WORK = Path(__file__).resolve().parent.parent / "outputs" / "river_edm_work"


def main():
    pairs = [l.split() for l in (WORK / "pairs-edm-covis-top20.txt").read_text().splitlines() if l.strip()]
    # a manageable slice: every pair touching the first 40 reference images
    names = sorted({n for p in pairs for n in p})[:40]
    keep = set(names)
    pairs = [(a, b) for a, b in pairs if a in keep and b in keep]
    print(f"{len(names)} images, {len(pairs)} pairs")

    obs: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    from hloc.utils.io import names_to_pair
    with h5py.File(WORK / "edm_pairs.h5", "r") as h5:
        for a, b in pairs:
            g = h5[names_to_pair(a, b)]
            for side, name in ((0, a), (1, b)):
                c = g[f"cell{side}"][:]
                k = g[f"kpt{side}"][:]
                r = EDMMatcher.is_refined(k)
                for cc, pp in zip(c[r], k[r]):
                    obs[(name, int(cc))].append(pp)

    stds, offs, counts = [], [], []
    for (name, cell), pts in obs.items():
        pts = np.asarray(pts)
        counts.append(len(pts))
        if len(pts) >= 2:
            stds.append(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
        cy, cx = divmod(cell, GRID_W)
        centre = np.array([cx, cy]) * COARSE_STRIDE
        offs.append(np.linalg.norm(pts.mean(0) - centre))

    stds = np.array(stds)
    offs = np.array(offs)
    counts = np.array(counts)
    print(f"\ncells with >=1 refined obs: {len(counts)}   obs/cell: "
          f"median={np.median(counts):.0f} mean={counts.mean():.1f} max={counts.max()}")
    print(f"cells with >=2 obs: {len(stds)}")
    print("\nper-cell spread of refined observations (EDM px):")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:<3} = {np.percentile(stds, q):.2f}")
    print(f"  frac > 1.5px (the gate the build used) = {(stds > 1.5).mean()*100:.1f}%")
    print(f"  frac > 3.0px                            = {(stds > 3.0).mean()*100:.1f}%")
    print("\n|mean refined - cell centre| (EDM px; the runtime anchor mismatch):")
    for q in (50, 75, 90, 99):
        print(f"  p{q:<3} = {np.percentile(offs, q):.2f}")
    print(f"  (clamp bound is {COARSE_STRIDE/2:.0f}px by construction)")


if __name__ == "__main__":
    main()
