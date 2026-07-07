#!/usr/bin/env python3
"""Manually draw power lines on the aligned map cloud, in Blender, by clicking.

This registers a modal operator that lets you click endpoints in the 3D
viewport; each click SNAPS to the nearest visible cloud vertex (so endpoints sit
on the real reconstructed tower/bridge attachment points, not in empty space).
Two clicks -> a CATENARY (hanging-cable) curve is generated between them; the
mouse wheel adjusts that span's sag. The wires live in the SAME aligned frame as
map_aligned.ply / safe_volume.ply, so they drop straight into the safe-zone /
obstacle layer.

Why catenary, not a straight line: a free-hanging cable dips in the middle
(y = a*cosh(...)); a straight chord would under-estimate how low the wire hangs
and give a wrong clearance for flight planning.

Run (Blender must already be up via blender_mcp_launch.py):
  python3 scripts/blender_send.py codefile scripts/draw_wires.py
then in the 3D viewport press F3 -> "Draw Catenary Wires" (or Ctrl+Shift+W).

Controls (while the operator is running):
  LEFT CLICK   place an endpoint (snaps to nearest visible cloud vertex)
  WHEEL UP/DN  adjust sag of the LAST finished wire (deeper / shallower)
  Z            undo last wire
  S            save -> safezone/wires.json + safezone/wires.ply (sampled points)
  ENTER / ESC  save and finish

Output:
  safezone/wires.json  [{"a":[x,y,z], "b":[x,y,z], "sag":float}, ...]  (aligned frame)
  safezone/wires.ply   dense points sampled along every catenary (obstacle layer)
  Blender objects under collection "Wires" (red bevelled curves) for inspection.
"""
import json
import math
import os

import bpy
import numpy as np
from bpy_extras import view3d_utils  # noqa: F401  (kept for reference / fallback)

ROOT = os.environ.get("SFM_MAP_ROOT", "/media/cihcilab/新增磁碟區/sfm_glomap")
OUTD = os.environ.get("SFM_SAFEZONE_DIR", f"{ROOT}/safezone")
WIRES_JSON = f"{OUTD}/wires.json"
WIRES_PLY = f"{OUTD}/wires.ply"

CLOUD_NAMES = ["gs_cloud", "map_cloud"]   # snap to the dense GS cloud if loaded, else sparse
SNAP_RADIUS_PX = 14.0         # click must land within this many px of a cloud vertex
SAMPLES_PER_SPAN = 80         # catenary polyline resolution
SAG_STEP = 1.15               # wheel multiplier on sag
DEFAULT_SAG_FRAC = 0.04       # initial sag = 4% of horizontal span length


# ---------------------------------------------------------------- catenary math
def _solve_a(d: float, sag: float) -> float:
    """Catenary parameter a solving  sag = a*(cosh(d/(2a)) - 1)  (level span)."""
    if sag <= 1e-9 or d <= 1e-9:
        return 1e9
    lo, hi = 1e-4, 1e6                       # f(a)=a*(cosh(d/2a)-1) is decreasing in a
    for _ in range(80):
        a = 0.5 * (lo + hi)
        f = a * (math.cosh(d / (2.0 * a)) - 1.0)
        if f > sag:
            lo = a                            # need bigger a to reduce sag
        else:
            hi = a
    return 0.5 * (lo + hi)


def catenary_points(a3, b3, sag, n=SAMPLES_PER_SPAN):
    """Sample a sagging cable between 3D endpoints a3,b3 (aligned frame, +Z up).

    Vertical sag is added below the straight chord so the curve hits both
    endpoints exactly and dips by `sag` (map units) at mid-span.
    """
    a3 = np.asarray(a3, float)
    b3 = np.asarray(b3, float)
    dxy = b3[:2] - a3[:2]
    d = float(np.linalg.norm(dxy))
    a = _solve_a(d, sag)
    pts = []
    for i in range(n + 1):
        t = i / n
        u = t * d
        chord_z = a3[2] + t * (b3[2] - a3[2])
        # dip(u): 0 at ends, -sag at centre
        dip = a * (math.cosh((u - d / 2.0) / a) - math.cosh(d / (2.0 * a))) if d > 1e-9 else 0.0
        xy = a3[:2] + t * dxy
        pts.append((float(xy[0]), float(xy[1]), float(chord_z + dip)))
    return pts


# ---------------------------------------------------------------- blender helpers
def _cloud_coords():
    """Snap targets = vertices of every visible candidate cloud (prefer gs_cloud)."""
    chunks = []
    for nm in CLOUD_NAMES:
        obj = bpy.data.objects.get(nm)
        if obj is None or obj.type != "MESH" or obj.hide_get() or obj.hide_viewport:
            continue
        me = obj.data
        n = len(me.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(obj.matrix_world)
        chunks.append((co @ M[:3, :3].T) + M[:3, 3])   # to world (aligned) frame
    return np.concatenate(chunks, axis=0) if chunks else None


def _wires_collection():
    col = bpy.data.collections.get("Wires")
    if col is None:
        col = bpy.data.collections.new("Wires")
        bpy.context.scene.collection.children.link(col)
    return col


def _wire_material():
    mat = bpy.data.materials.get("wire_mat")
    if mat is None:
        mat = bpy.data.materials.new("wire_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.95, 0.1, 0.1, 1)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (0.95, 0.1, 0.1, 1)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


def _make_curve(idx, pts, span_len):
    name = f"wire_{idx:03d}"
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (p[0], p[1], p[2], 1.0)
    cu.bevel_depth = max(span_len * 0.004, 1e-3)   # thin visible cable
    cu.bevel_resolution = 1
    obj = bpy.data.objects.new(name, cu)
    obj.data.materials.append(_wire_material())
    _wires_collection().objects.link(obj)
    return obj


def _save(wires):
    os.makedirs(OUTD, exist_ok=True)
    with open(WIRES_JSON, "w") as f:
        json.dump([{"a": list(w["a"]), "b": list(w["b"]), "sag": w["sag"]} for w in wires],
                  f, indent=2)
    # dense point sampling for the obstacle layer
    allp = []
    for w in wires:
        allp.extend(catenary_points(w["a"], w["b"], w["sag"]))
    with open(WIRES_PLY, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(allp)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in allp:
            f.write(f"{p[0]} {p[1]} {p[2]} 245 26 26\n")
    print(f"[wires] saved {len(wires)} wires -> {WIRES_JSON} ; {len(allp)} pts -> {WIRES_PLY}")


# ---------------------------------------------------------------- modal operator
class WIRE_OT_draw(bpy.types.Operator):
    bl_idname = "view3d.draw_catenary_wires"
    bl_label = "Draw Catenary Wires"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        self._cloud = _cloud_coords()
        if self._cloud is None:
            self.report({"ERROR"}, f"no visible cloud among {CLOUD_NAMES}")
            return {"CANCELLED"}
        self._wires = []
        # reload any previously saved wires so you can keep editing
        if os.path.exists(WIRES_JSON):
            try:
                for w in json.load(open(WIRES_JSON)):
                    self._wires.append({"a": w["a"], "b": w["b"], "sag": float(w["sag"])})
            except Exception as e:
                print("[wires] could not reload:", e)
        self._pending = None
        self._rebuild_all()
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    # ---- snapping: click -> nearest visible cloud vertex
    def _snap(self, context, event):
        region = context.region
        rv3d = context.region_data
        if rv3d is None:
            return None
        P = np.array(rv3d.perspective_matrix)               # world -> clip
        co = self._cloud
        hom = np.concatenate([co, np.ones((len(co), 1))], axis=1)
        clip = hom @ P.T
        w = clip[:, 3]
        front = w > 1e-6
        ndc = np.empty((len(co), 2))
        ndc[front] = clip[front, :2] / w[front, None]
        sx = (ndc[:, 0] * 0.5 + 0.5) * region.width
        sy = (ndc[:, 1] * 0.5 + 0.5) * region.height
        mx, my = event.mouse_region_x, event.mouse_region_y
        d2 = (sx - mx) ** 2 + (sy - my) ** 2
        d2[~front] = np.inf
        near = np.where(d2 < SNAP_RADIUS_PX ** 2)[0]
        if len(near) == 0:
            return None
        # among the on-screen candidates pick the one closest to the camera
        cam = np.array(rv3d.view_matrix.inverted().translation)
        depth = np.linalg.norm(co[near] - cam, axis=1)
        return co[near[np.argmin(depth)]].tolist()

    def _rebuild_all(self):
        for w in list(self._wires):
            span = float(np.linalg.norm(np.array(w["b"][:2]) - np.array(w["a"][:2])))
            _make_curve(self._wires.index(w), catenary_points(w["a"], w["b"], w["sag"]), span)

    def _status(self, context):
        n = len(self._wires)
        msg = f"Wires:{n}  " + ("click END point" if self._pending else "click START point")
        msg += "  |  Alt+drag=orbit  Wheel=sag  Z=undo  S=save  Enter/Esc=finish"
        context.area.header_text_set(msg)
        context.area.tag_redraw()

    def modal(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            return {"PASS_THROUGH"}

        # navigation always wins: emulated MMB orbit = Alt+Left, pan = Shift+Alt+Left,
        # zoom = Alt+wheel/Ctrl+Alt-drag. Holding Alt -> let Blender navigate.
        if event.alt:
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            p = self._snap(context, event)
            if p is None:
                self.report({"WARNING"}, "no cloud vertex under cursor (zoom in / aim at a point)")
                return {"RUNNING_MODAL"}
            if self._pending is None:
                self._pending = p
            else:
                a, b = self._pending, p
                span = float(np.linalg.norm(np.array(b[:2]) - np.array(a[:2])))
                sag = max(span * DEFAULT_SAG_FRAC, 1e-3)
                self._wires.append({"a": a, "b": b, "sag": sag})
                _make_curve(len(self._wires) - 1, catenary_points(a, b, sag), span)
                self._pending = None
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS" and self._wires:
            w = self._wires[-1]
            w["sag"] *= SAG_STEP if event.type == "WHEELUPMOUSE" else (1.0 / SAG_STEP)
            span = float(np.linalg.norm(np.array(w["b"][:2]) - np.array(w["a"][:2])))
            _make_curve(len(self._wires) - 1, catenary_points(w["a"], w["b"], w["sag"]), span)
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type == "Z" and event.value == "PRESS" and self._wires:
            self._wires.pop()
            obj = bpy.data.objects.get(f"wire_{len(self._wires):03d}")
            if obj:
                bpy.data.objects.remove(obj, do_unlink=True)
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type == "S" and event.value == "PRESS":
            _save(self._wires)
            self.report({"INFO"}, f"saved {len(self._wires)} wires")
            return {"RUNNING_MODAL"}

        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            _save(self._wires)
            context.area.header_text_set(None)
            self.report({"INFO"}, f"done, {len(self._wires)} wires saved")
            return {"FINISHED"}

        # let the user orbit/zoom the view
        if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM",
                          "NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4", "NUMPAD_5",
                          "NUMPAD_6", "NUMPAD_7", "NUMPAD_8", "NUMPAD_9"}:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


_addon_keymaps = []


def _menu_func(self, context):
    self.layout.operator(WIRE_OT_draw.bl_idname, text="Draw Catenary Wires", icon="CURVE_DATA")


def register():
    try:
        bpy.utils.unregister_class(WIRE_OT_draw)
    except Exception:
        pass
    bpy.utils.register_class(WIRE_OT_draw)

    # 1) make custom operators show up in the F3 search menu
    try:
        bpy.context.preferences.view.show_developer_ui = True
    except Exception as e:
        print("[wires] could not enable developer extras:", e)

    # 2) clickable entry in the 3D viewport's "View" menu (idempotent on re-push)
    mt = bpy.types.VIEW3D_MT_view
    try:
        existing = list(getattr(mt.draw, "_draw_funcs", []))
        for f in existing:
            if getattr(f, "__name__", "") == "_menu_func":
                mt.remove(f)
    except Exception:
        pass
    mt.append(_menu_func)

    # 3) keyboard shortcut Ctrl+Shift+W (try addon then user keyconfig)
    wm = bpy.context.window_manager
    for kc in (wm.keyconfigs.addon, wm.keyconfigs.user, wm.keyconfigs.active):
        if not kc:
            continue
        km = kc.keymaps.get("3D View") or kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        if not any(k.idname == WIRE_OT_draw.bl_idname for k in km.keymap_items):
            kmi = km.keymap_items.new(WIRE_OT_draw.bl_idname, "W", "PRESS", ctrl=True, shift=True)
            _addon_keymaps.append((km, kmi))
        break
    print("[wires] registered. Open it 3 ways: 3D viewport top menu 'View' > "
          "'Draw Catenary Wires'  |  F3 search 'Draw Catenary Wires'  |  Ctrl+Shift+W")


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()
    try:
        bpy.utils.unregister_class(WIRE_OT_draw)
    except Exception:
        pass


if __name__ == "__main__" or __name__ == "__builtin__" or True:
    # executed via blender_send.py / --python : (re)register so F3 finds it
    register()
