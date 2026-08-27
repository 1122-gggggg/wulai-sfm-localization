#!/usr/bin/env python3
"""Re-derive safezone/poles.json from the edited pole_NNN box objects in Blender.

The draw_poles.py tool makes axis-aligned vertical boxes. To FIX a box that
doesn't hug its pole, exit the tool (Enter), select the pole_NNN cube and move
it (G) / resize it (S) in Blender, then run this to write the new positions back
to poles.json. Boxes stay axis-aligned -- rotating (R) only grows the bbox, so
straighten with G/S, not R.

Run (Blender up):
  python3 scripts/blender_send.py codefile scripts/sync_poles.py
"""
import os

import bpy
import numpy as np

ROOT = os.environ.get("SFM_MAP_ROOT", "").strip()
if not ROOT:
    raise RuntimeError("SFM_MAP_ROOT must explicitly select the authoring workspace")
OUTD = os.environ.get("SFM_SAFEZONE_DIR", f"{ROOT}/safezone")
POLES_JSON = f"{OUTD}/poles.json"

# reuse draw_poles' box geometry + writer so json/ply stay identical
import importlib.util
spec = importlib.util.spec_from_file_location("draw_poles", f"{ROOT}/scripts/draw_poles.py")
dp = importlib.util.module_from_spec(spec)
# draw_poles.register() runs on import; that's fine (idempotent)
spec.loader.exec_module(dp)


def _oriented_box(obj):
    """Oriented box from an edited pole object: world center, local half-extents
    (after scale), and orientation R. Handles G (move), S/S-Z (stretch) and
    R (tilt). The mesh's local verts are an axis-aligned box; matrix_world =
    T * R * S turns it into an oriented world box."""
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    c_l = (co.min(0) + co.max(0)) * 0.5          # local center
    h_l = (co.max(0) - co.min(0)) * 0.5          # local half-extents (axis-aligned)
    loc, rot, scale = obj.matrix_world.decompose()
    R = np.array(rot.to_matrix())                # 3x3 world orientation
    M = np.array(obj.matrix_world)
    center = (M[:3, :3] @ c_l) + M[:3, 3]        # world center
    half = h_l * np.array(scale)                 # world half-extents along box axes
    return center, half, R


def main():
    col = bpy.data.collections.get("Poles")
    objs = sorted([o for o in (col.objects if col else []) if o.name.startswith("pole_")],
                  key=lambda o: o.name)
    poles = []
    for o in objs:
        c, h, R = _oriented_box(o)
        ez = R[:, 2]                              # pole axis (box local +Z) in world
        base = c - ez * h[2]
        top = c + ez * h[2]
        r = float((h[0] + h[1]) * 0.5)
        poles.append({
            "center": [float(v) for v in c],
            "half_extents": [float(v) for v in h],
            "R": [[float(v) for v in row] for row in R],   # world orientation (identity if un-tilted)
            "base": [float(v) for v in base],
            "top": [float(v) for v in top],
            "radius": r,
        })
    os.makedirs(OUTD, exist_ok=True)
    dp._save(poles)                      # writes poles.json + poles.ply, redraws not needed
    print(f"[sync] {len(poles)} poles synced from Blender objects -> {POLES_JSON}")


main()
