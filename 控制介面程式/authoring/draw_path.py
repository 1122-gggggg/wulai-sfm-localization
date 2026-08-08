#!/usr/bin/env python3
"""Hand-draw a drone FLIGHT PATH on the aligned map cloud, in Blender, by clicking.

This is the manual counterpart to deploy/plan_path.py's automatic A* planner:
instead of letting the planner route between targets, you click the waypoints
yourself. The result is a single connected polyline of 3D waypoints in the SAME
aligned frame as map_aligned.ply / true_polygon.json / wires.ply (ground = z=0,
up = +Z), so it drops straight into the executor:

    from load_path import load_waypoints          # deploy/load_path.py
    follower = PathFollower(load_waypoints())      # cruise_geofence.PathFollower

Why waypoints live in free space (not snapped to a surface like the wires):
a cruise path flies ABOVE the structures. Each click fixes the waypoint's XY
(snapped to the cloud under the cursor, with a ground-plane fallback over gaps)
and its Z is the working CRUISE ALTITUDE in the aligned frame -- ground is z=0
here (align_report median z=0.008), so altitude reads as height above ground.
Pick a top-down view, trace the route, raise/lower altitude with the wheel.

Run (Blender already up via scripts/blender_mcp_launch.py):
  python3 scripts/blender_send.py codefile scripts/draw_path.py
then in the 3D viewport press F3 -> "Draw Flight Path" (or Ctrl+Shift+P), or use
the 3D viewport "View" menu entry.

Controls (while the operator is running):
  LEFT CLICK    add a waypoint (XY snaps to nearest visible cloud vertex; if none
                under the cursor it drops onto the z=0 ground plane along the ray)
  WHEEL UP/DN   raise / lower the LAST waypoint's altitude
  A             set ALL waypoints to the last one's altitude (flat cruise)
  C             toggle CLOSED loop (return to start -- patrol) on/off
  Z             undo last waypoint
  S             save -> safezone/flight_path.json + safezone/flight_path.ply
  ENTER / ESC   save and finish
  Hold ALT      orbit/pan/zoom the view (navigation always wins)

Output:
  safezone/flight_path.json
      {"waypoints":[[x,y,z],...], "closed":bool, "frame":"aligned",
       "units":"map", "source":"blender_draw_path", "default_alt":float}
  safezone/flight_path.ply   green polyline densely sampled along the legs
  Blender objects under collection "FlightPath" (green tube + waypoint markers).
"""
import json
import os

import bpy
import numpy as np

ROOT = os.environ.get("SFM_MAP_ROOT", "").strip()
if not ROOT:
    raise RuntimeError("SFM_MAP_ROOT must explicitly select the authoring workspace")
OUTD = os.environ.get("SFM_SAFEZONE_DIR", f"{ROOT}/safezone")
PATH_JSON = f"{OUTD}/flight_path.json"
PATH_PLY = f"{OUTD}/flight_path.ply"
REPORT = f"{OUTD}/align_report.json"

CLOUD_NAMES = ["gs_cloud", "map_cloud"]   # snap to the dense GS cloud if loaded, else sparse
SNAP_RADIUS_PX = 16.0          # click within this many px of a cloud vertex -> snap its XY
ALT_STEP = 0.25               # map units per wheel notch on altitude
SAMPLES_PER_LEG = 24          # polyline resolution for the exported .ply


def _default_alt():
    """Working cruise altitude: clear the 99th-pct of cloud height, + margin."""
    try:
        q = json.load(open(REPORT))["z_pct_1_5_50_90_95_99"]
        return round(float(q[5]) + 1.0, 2)     # z_99 + 1.0 map unit
    except Exception:
        return 4.5


DEFAULT_ALT = _default_alt()


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


def _path_collection():
    col = bpy.data.collections.get("FlightPath")
    if col is None:
        col = bpy.data.collections.new("FlightPath")
        bpy.context.scene.collection.children.link(col)
    return col


def _path_material(name, rgb):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (*rgb, 1)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


def _clear_path_objects():
    col = _path_collection()
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _span(pts):
    if len(pts) < 2:
        return 1.0
    a = np.asarray(pts, float)
    return float(np.linalg.norm(a.max(0) - a.min(0))) or 1.0


def _draw(waypoints, closed):
    """(Re)build the green path tube + waypoint marker spheres."""
    _clear_path_objects()
    if not waypoints:
        return
    col = _path_collection()
    span = _span(waypoints)
    line_mat = _path_material("path_mat", (0.90, 0.05, 0.05))
    node_mat = _path_material("path_node_mat", (1.0, 0.85, 0.05))

    # the route tube
    loop = waypoints + [waypoints[0]] if (closed and len(waypoints) > 2) else waypoints
    if len(loop) >= 2:
        cu = bpy.data.curves.new("flight_path", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(len(loop) - 1)
        for i, p in enumerate(loop):
            sp.points[i].co = (p[0], p[1], p[2], 1.0)
        cu.bevel_depth = max(span * 0.004, 1e-3)
        cu.bevel_resolution = 2
        obj = bpy.data.objects.new("flight_path", cu)
        obj.data.materials.append(line_mat)
        col.objects.link(obj)

    # waypoint markers (numbered by creation order)
    r = max(span * 0.010, 1e-3)
    for i, p in enumerate(waypoints):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(p[0], p[1], p[2]))
        m = bpy.context.active_object
        m.name = f"wp_{i:03d}"
        m.data.materials.append(node_mat)
        # move from scene collection into FlightPath
        for c in list(m.users_collection):
            c.objects.unlink(m)
        col.objects.link(m)


def _resample(waypoints, closed):
    """Dense points along every leg for the inspection .ply."""
    loop = waypoints + [waypoints[0]] if (closed and len(waypoints) > 2) else waypoints
    out = []
    for a, b in zip(loop[:-1], loop[1:]):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        for i in range(SAMPLES_PER_LEG):
            t = i / SAMPLES_PER_LEG
            p = a + t * (b - a)
            out.append((float(p[0]), float(p[1]), float(p[2])))
    if loop:
        p = np.asarray(loop[-1], float)
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def _save(waypoints, closed):
    os.makedirs(OUTD, exist_ok=True)
    with open(PATH_JSON, "w") as f:
        json.dump({
            "waypoints": [[float(c) for c in p] for p in waypoints],
            "closed": bool(closed),
            "frame": "aligned",
            "units": "map",
            "source": "blender_draw_path",
            "default_alt": DEFAULT_ALT,
        }, f, indent=2)
    allp = _resample(waypoints, closed)
    with open(PATH_PLY, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(allp)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in allp:
            f.write(f"{p[0]} {p[1]} {p[2]} 40 230 60\n")
    print(f"[path] saved {len(waypoints)} waypoints (closed={closed}) -> {PATH_JSON} ; "
          f"{len(allp)} pts -> {PATH_PLY}")


# ---------------------------------------------------------------- modal operator
class PATH_OT_draw(bpy.types.Operator):
    bl_idname = "view3d.draw_flight_path"
    bl_label = "Draw Flight Path"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        self._cloud = _cloud_coords()
        if self._cloud is None:
            self.report({"ERROR"}, f"no visible cloud among {CLOUD_NAMES}")
            return {"CANCELLED"}
        self._wp = []
        self._closed = False
        # reload a previously drawn path so you can keep editing it
        if os.path.exists(PATH_JSON):
            try:
                d = json.load(open(PATH_JSON))
                self._wp = [list(map(float, p)) for p in d.get("waypoints", [])]
                self._closed = bool(d.get("closed", False))
            except Exception as e:
                print("[path] could not reload:", e)
        _draw(self._wp, self._closed)
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    # ---- XY from nearest visible cloud vertex; fallback = ray hits z=0 plane
    def _pick_xy(self, context, event):
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
        if len(near):
            cam = np.array(rv3d.view_matrix.inverted().translation)
            depth = np.linalg.norm(co[near] - cam, axis=1)
            p = co[near[np.argmin(depth)]]
            return float(p[0]), float(p[1])
        # fallback: intersect the view ray with the ground plane z=0
        from bpy_extras import view3d_utils
        coord = (mx, my)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        if abs(direction[2]) < 1e-6:
            return None
        t = -origin[2] / direction[2]
        if t <= 0:
            return None
        hit = origin + t * direction
        return float(hit[0]), float(hit[1])

    def _last_alt(self):
        return self._wp[-1][2] if self._wp else DEFAULT_ALT

    def _status(self, context):
        n = len(self._wp)
        alt = self._last_alt()
        msg = (f"FlightPath  waypoints:{n}  alt(last):{alt:.2f}  closed:{self._closed}  |  "
               "Click=add  Wheel=alt  A=flat  C=loop  Z=undo  S=save  Enter/Esc=finish  Alt=orbit")
        context.area.header_text_set(msg)
        context.area.tag_redraw()

    def modal(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            return {"PASS_THROUGH"}

        # --- view navigation: pass Shift/Alt + LMB through to the nav keymap items
        # added in register() (Shift+LMB=pan, Alt+LMB=orbit). Plain LMB still draws. ---
        if event.type == "LEFTMOUSE" and (event.shift or event.alt):
            return {"PASS_THROUGH"}
        if event.alt:                                    # Alt+wheel zoom etc.
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            xy = self._pick_xy(context, event)
            if xy is None:
                self.report({"WARNING"}, "could not resolve a point under the cursor")
                return {"RUNNING_MODAL"}
            z = self._last_alt()                     # inherit previous altitude
            self._wp.append([xy[0], xy[1], z])
            _draw(self._wp, self._closed)
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS" and self._wp:
            self._wp[-1][2] += ALT_STEP if event.type == "WHEELUPMOUSE" else -ALT_STEP
            _draw(self._wp, self._closed)
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type == "A" and event.value == "PRESS" and self._wp:
            z = self._wp[-1][2]
            for p in self._wp:
                p[2] = z
            _draw(self._wp, self._closed)
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type == "C" and event.value == "PRESS":
            self._closed = not self._closed
            _draw(self._wp, self._closed)
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type == "Z" and event.value == "PRESS" and self._wp:
            self._wp.pop()
            _draw(self._wp, self._closed)
            self._status(context)
            return {"RUNNING_MODAL"}

        if event.type == "S" and event.value == "PRESS":
            _save(self._wp, self._closed)
            self.report({"INFO"}, f"saved {len(self._wp)} waypoints")
            return {"RUNNING_MODAL"}

        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            _save(self._wp, self._closed)
            context.area.header_text_set(None)
            self.report({"INFO"}, f"done, {len(self._wp)} waypoints saved")
            return {"FINISHED"}

        if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM",
                          "NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4", "NUMPAD_5",
                          "NUMPAD_6", "NUMPAD_7", "NUMPAD_8", "NUMPAD_9"}:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


_addon_keymaps = []


def _menu_func(self, context):
    self.layout.operator(PATH_OT_draw.bl_idname, text="Draw Flight Path", icon="CURVE_PATH")


def register():
    try:
        bpy.utils.unregister_class(PATH_OT_draw)
    except Exception:
        pass
    bpy.utils.register_class(PATH_OT_draw)

    try:
        bpy.context.preferences.view.show_developer_ui = True
    except Exception as e:
        print("[path] could not enable developer extras:", e)

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
        if not any(k.idname == PATH_OT_draw.bl_idname for k in km.keymap_items):
            kmi = km.keymap_items.new(PATH_OT_draw.bl_idname, "P", "PRESS", ctrl=True, shift=True)
            _addon_keymaps.append((km, kmi))
        # LEFT-mouse navigation so plain LMB stays free for drawing:
        #   Shift+LMB drag = pan (view3d.move), Alt+LMB drag = orbit (view3d.rotate).
        # Remove any prior copies first so reloading register() stays idempotent.
        for k in list(km.keymap_items):
            if k.type == "LEFTMOUSE" and k.idname in ("view3d.move", "view3d.rotate") \
                    and (k.shift or k.alt):
                km.keymap_items.remove(k)
        _addon_keymaps.append((km, km.keymap_items.new("view3d.move", "LEFTMOUSE", "PRESS", shift=True)))
        _addon_keymaps.append((km, km.keymap_items.new("view3d.rotate", "LEFTMOUSE", "PRESS", alt=True)))
        break
    print(f"[path] registered (default cruise alt={DEFAULT_ALT}). Open 3 ways: "
          "3D viewport 'View' menu > 'Draw Flight Path'  |  F3 search 'Draw Flight Path'  |  "
          "Ctrl+Shift+P")


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()
    try:
        bpy.utils.unregister_class(PATH_OT_draw)
    except Exception:
        pass


if __name__ == "__main__" or __name__ == "__builtin__" or True:
    register()
