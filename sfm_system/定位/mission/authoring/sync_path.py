#!/usr/bin/env python3
"""Re-derive safezone/flight_path.json from the moved wp_NNN markers in Blender.

draw_path.py only wheel-adjusts the LAST waypoint. To set per-waypoint (per-leg)
heights: exit the path tool (Enter), then in a SIDE view grab each wp_NNN marker
and move it up/down (G Z). Run this to write the new heights back to the path
and redraw the green line through them.

Run (Blender up):
  python3 scripts/blender_send.py codefile scripts/sync_path.py
"""
import json
import os

import bpy

ROOT = os.environ.get("SFM_MAP_ROOT", "/media/cihcilab/新增磁碟區/sfm_glomap")
OUTD = os.environ.get("SFM_SAFEZONE_DIR", f"{ROOT}/safezone")
PATH_JSON = f"{OUTD}/flight_path.json"

import importlib.util
spec = importlib.util.spec_from_file_location("draw_path", f"{ROOT}/scripts/draw_path.py")
dpth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpth)          # registers the operator (idempotent)


def main():
    col = bpy.data.collections.get("FlightPath")
    markers = sorted([o for o in (col.objects if col else []) if o.name.startswith("wp_")],
                     key=lambda o: o.name)
    wps = [[float(c) for c in o.matrix_world.translation] for o in markers]
    closed = False
    if os.path.exists(PATH_JSON):
        try:
            closed = bool(json.load(open(PATH_JSON)).get("closed", False))
        except Exception:
            pass
    if len(wps) < 2:
        print(f"[sync_path] only {len(wps)} markers -- nothing to write")
        return
    dpth._save(wps, closed)            # write flight_path.json + .ply
    dpth._draw(wps, closed)            # redraw the green line through the new heights
    zs = [round(w[2], 2) for w in wps]
    print(f"[sync_path] {len(wps)} waypoints synced, heights={zs}")


main()
