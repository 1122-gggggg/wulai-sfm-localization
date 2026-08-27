#!/usr/bin/env python3
"""Mark utility POLES as 3D boxes on the aligned map cloud, in Blender, by clicking.

For pole inspection: each pole becomes an axis-aligned vertical CUBOID in the
ALIGNED map frame (ground z=0, +Z up) -- same frame as map_aligned.ply /
true_polygon.json / wires.ply / flight_path.json. The boxes feed two consumers:

  1. OBSTACLE / no-go  -> baked into the planner SDF WITH a buffer, so
     deploy/plan_path.py keeps standoff from each pole by construction.
  2. INSPECTION targets -> deploy/load_poles.py turns each box into standoff
     waypoints (a ring at a chosen distance) for plan_tour / autoflight.

A pole is tall and thin, so you mark it with TWO clicks:
  click 1 = the pole BASE (snaps to nearest visible cloud vertex -> XY + z_base)
  click 2 = the pole TOP  (snaps -> z_top; box spans z_base..z_top at base XY)
The square footprint half-width (radius) defaults small and is wheel-adjustable;
the SDF buffer is added later at bake time, so keep this hugging the real pole.

Run (Blender already up via scripts/blender_mcp_launch.py):
  python3 scripts/blender_send.py codefile scripts/draw_poles.py
then in the 3D viewport press F3 -> "Mark Poles (Boxes)" (or Ctrl+Shift+B), or
use the 3D viewport "View" menu entry.

Controls (while the operator is running):
  LEFT CLICK    1st = pole base, 2nd = pole top (both snap to the cloud)
  WHEEL UP/DN   grow / shrink the LAST pole's footprint radius
  Z             undo last pole (or cancel a pending base)
  S             save -> safezone/poles.json + safezone/poles.ply
  ENTER / ESC   save and finish
  Hold ALT      orbit/pan/zoom the view (navigation always wins)

Output:
  safezone/poles.json
      {"poles":[{"center":[x,y,z],"half_extents":[hx,hy,hz],
                 "base":[x,y,z],"top":[x,y,z],"radius":r}, ...],
       "frame":"aligned","units":"map","source":"blender_draw_poles"}
  safezone/poles.ply   triangulated boxes (mesh: vertices + faces) for the
                       obstacle layer / SDF bake and visual inspection.
  Blender objects under collection "Poles" (orange wire boxes) for inspection.
"""
import json
import os

import bpy
import numpy as np

ROOT = os.environ.get("SFM_MAP_ROOT", "").strip()
if not ROOT:
    raise RuntimeError("SFM_MAP_ROOT must explicitly select the authoring workspace")
OUTD = os.environ.get("SFM_SAFEZONE_DIR", f"{ROOT}/safezone")
POLES_JSON = f"{OUTD}/poles.json"
POLES_PLY = f"{OUTD}/poles.ply"

CLOUD_NAMES = ["gs_cloud", "map_cloud"]   # snap to the dense GS cloud if loaded, else sparse
SNAP_RADIUS_PX = 16.0          # click within this many px of a cloud vertex -> snap
DEFAULT_RADIUS = 0.08         # footprint half-width (map units); wheel-tunable
RADIUS_STEP = 0.02            # map units per wheel notch
MIN_RADIUS = 0.01


# ---------------------------------------------------------------- box geometry
def box_corners(center, half, R=None):
    """8 corners of a box. Axis-aligned unless an orientation R (3x3) is given,
    in which case corners are rotated about the center (oriented/tilted pole)."""
    cx, cy, cz = center
    hx, hy, hz = half
    signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    if R is None:
        return [(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx, sy, sz in signs]
    R = np.asarray(R, float)
    out = []
    for sx, sy, sz in signs:
        local = np.array([sx * hx, sy * hy, sz * hz])
        w = R @ local + np.array([cx, cy, cz])
        out.append((float(w[0]), float(w[1]), float(w[2])))
    return out


# CCW outward-facing triangles (for a watertight box mesh)
_BOX_TRIS = [
    (0, 2, 1), (0, 3, 2),   # bottom (-z)
    (4, 5, 6), (4, 6, 7),   # top (+z)
    (0, 1, 5), (0, 5, 4),   # -y
    (1, 2, 6), (1, 6, 5),   # +x
    (2, 3, 7), (2, 7, 6),   # +y
    (3, 0, 4), (3, 4, 7),   # -x
]


def pole_box(base, top, radius):
    """Axis-aligned cuboid spanning base.z..top.z at the base XY."""
    cx, cy = float(base[0]), float(base[1])
    z0, z1 = float(base[2]), float(top[2])
    if z1 < z0:
        z0, z1 = z1, z0
    cz = 0.5 * (z0 + z1)
    hz = max(0.5 * (z1 - z0), MIN_RADIUS)
    return {
        "center": [cx, cy, cz],
        "half_extents": [radius, radius, hz],
        "base": [cx, cy, z0],
        "top": [cx, cy, z1],
        "radius": radius,
    }


# ---------------------------------------------------------------- blender helpers
def _cloud_coords():
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
        chunks.append((co @ M[:3, :3].T) + M[:3, 3])
    return np.concatenate(chunks, axis=0) if chunks else None


def _poles_collection():
    col = bpy.data.collections.get("Poles")
    if col is None:
        col = bpy.data.collections.new("Poles")
        bpy.context.scene.collection.children.link(col)
    return col


def _pole_material():
    mat = bpy.data.materials.get("pole_mat")
    if mat is None:
        mat = bpy.data.materials.new("pole_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 0.55, 0.05, 1)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (1.0, 0.55, 0.05, 1)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


def _make_box_obj(idx, pole):
    name = f"pole_{idx:03d}"
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    verts = box_corners(pole["center"], pole["half_extents"], pole.get("R"))
    me = bpy.data.meshes.new(name)
    me.from_pydata([list(v) for v in verts], [], [list(t) for t in _BOX_TRIS])
    me.update()
    obj = bpy.data.objects.new(name, me)
    obj.data.materials.append(_pole_material())
    obj.display_type = "WIRE"          # see the cloud pole inside the box
    obj.show_in_front = True
    _poles_collection().objects.link(obj)
    return obj


def _clear_pole_objects():
    for obj in list(_poles_collection().objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _redraw(poles):
    _clear_pole_objects()
    for i, p in enumerate(poles):
        _make_box_obj(i, p)


def _save(poles):
    os.makedirs(OUTD, exist_ok=True)
    with open(POLES_JSON, "w") as f:
        json.dump({
            "poles": poles,
            "frame": "aligned",
            "units": "map",
            "source": "blender_draw_poles",
        }, f, indent=2)
    # triangulated boxes -> one mesh PLY (vertices + faces)
    V, F = [], []
    for p in poles:
        base = len(V)
        V.extend(box_corners(p["center"], p["half_extents"], p.get("R")))
        F.extend([(base + a, base + b, base + c) for (a, b, c) in _BOX_TRIS])
    with open(POLES_PLY, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in V:
            f.write(f"{v[0]} {v[1]} {v[2]} 255 140 12\n")
        for tri in F:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    print(f"[poles] saved {len(poles)} poles -> {POLES_JSON} ; "
          f"{len(V)} verts / {len(F)} tris -> {POLES_PLY}")


def _poles_edit_event(operator, context, event):
    if event.type == "LEFTMOUSE" and event.value == "PRESS":
        p = operator._snap(context, event)
        if p is None:
            operator.report({"WARNING"}, "no cloud vertex under cursor (zoom in / aim at the pole)")
            return {"RUNNING_MODAL"}
        if operator._pending_base is None:
            operator._pending_base = p
        else:
            pole = pole_box(operator._pending_base, p, DEFAULT_RADIUS)
            operator._poles.append(pole)
            operator._pending_base = None
            _redraw(operator._poles)
        operator._status(context)
        return {"RUNNING_MODAL"}

    if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS" \
            and operator._poles:
        p = operator._poles[-1]
        r = p["radius"] + (RADIUS_STEP if event.type == "WHEELUPMOUSE" else -RADIUS_STEP)
        r = max(MIN_RADIUS, r)
        p["radius"] = r
        p["half_extents"][0] = r
        p["half_extents"][1] = r
        _redraw(operator._poles)
        operator._status(context)
        return {"RUNNING_MODAL"}

    if event.type == "Z" and event.value == "PRESS":
        if operator._pending_base is not None:
            operator._pending_base = None
        elif operator._poles:
            operator._poles.pop()
            _redraw(operator._poles)
        operator._status(context)
        return {"RUNNING_MODAL"}
    return None


def _poles_finish_event(operator, context, event):
    if event.type == "S" and event.value == "PRESS":
        _save(operator._poles)
        operator.report({"INFO"}, f"saved {len(operator._poles)} poles")
        return {"RUNNING_MODAL"}

    if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
        _save(operator._poles)
        context.area.header_text_set(None)
        operator.report({"INFO"}, f"done, {len(operator._poles)} poles saved")
        return {"FINISHED"}

    if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM",
                      "NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4", "NUMPAD_5",
                      "NUMPAD_6", "NUMPAD_7", "NUMPAD_8", "NUMPAD_9"}:
        return {"PASS_THROUGH"}
    return None


# ---------------------------------------------------------------- modal operator
class POLE_OT_draw(bpy.types.Operator):
    bl_idname = "view3d.mark_poles_boxes"
    bl_label = "Mark Poles (Boxes)"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        self._cloud = _cloud_coords()
        if self._cloud is None:
            self.report({"ERROR"}, f"no visible cloud among {CLOUD_NAMES}")
            return {"CANCELLED"}
        self._poles = []
        if os.path.exists(POLES_JSON):
            try:
                for p in json.load(open(POLES_JSON)).get("poles", []):
                    self._poles.append(p)
            except Exception as e:
                print("[poles] could not reload:", e)
        self._pending_base = None
        _redraw(self._poles)
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _snap(self, context, event):
        region = context.region
        rv3d = context.region_data
        if rv3d is None:
            return None
        P = np.array(rv3d.perspective_matrix)
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
        cam = np.array(rv3d.view_matrix.inverted().translation)
        depth = np.linalg.norm(co[near] - cam, axis=1)
        return co[near[np.argmin(depth)]].tolist()

    def _status(self, context):
        n = len(self._poles)
        stage = "click pole TOP" if self._pending_base is not None else "click pole BASE"
        msg = (f"Poles:{n}  {stage}  |  Click base->top  Wheel=radius  Z=undo  "
               "S=save  Enter/Esc=finish  Alt=orbit")
        context.area.header_text_set(msg)
        context.area.tag_redraw()

    def modal(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            return {"PASS_THROUGH"}
        if event.alt:
            return {"PASS_THROUGH"}

        result = _poles_edit_event(self, context, event)
        if result is not None:
            return result
        result = _poles_finish_event(self, context, event)
        if result is not None:
            return result
        return {"RUNNING_MODAL"}


_addon_keymaps = []


def _menu_func(self, context):
    self.layout.operator(POLE_OT_draw.bl_idname, text="Mark Poles (Boxes)", icon="MESH_CUBE")


def register():
    try:
        bpy.utils.unregister_class(POLE_OT_draw)
    except Exception:
        pass
    bpy.utils.register_class(POLE_OT_draw)

    try:
        bpy.context.preferences.view.show_developer_ui = True
    except Exception as e:
        print("[poles] could not enable developer extras:", e)

    mt = bpy.types.VIEW3D_MT_view
    try:
        existing = list(getattr(mt.draw, "_draw_funcs", []))
        for f in existing:
            if getattr(f, "__name__", "") == "_menu_func":
                mt.remove(f)
    except Exception:
        pass
    mt.append(_menu_func)

    wm = bpy.context.window_manager
    for kc in (wm.keyconfigs.addon, wm.keyconfigs.user, wm.keyconfigs.active):
        if not kc:
            continue
        km = kc.keymaps.get("3D View") or kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        if not any(k.idname == POLE_OT_draw.bl_idname for k in km.keymap_items):
            kmi = km.keymap_items.new(POLE_OT_draw.bl_idname, "B", "PRESS", ctrl=True, shift=True)
            _addon_keymaps.append((km, kmi))
        break
    print("[poles] registered. Open 3 ways: 3D viewport 'View' menu > 'Mark Poles (Boxes)'  |  "
          "F3 search 'Mark Poles'  |  Ctrl+Shift+B")


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()
    try:
        bpy.utils.unregister_class(POLE_OT_draw)
    except Exception:
        pass


if __name__ == "__main__" or __name__ == "__builtin__" or True:
    register()
