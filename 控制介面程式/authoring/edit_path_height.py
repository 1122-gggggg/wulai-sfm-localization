#!/usr/bin/env python3
"""Per-waypoint HEIGHT editor for the drawn flight path (companion to draw_path.py).

draw_path sets every waypoint to the cruise altitude; this tool lets you adjust
EACH point's height individually. Switch to a SIDE view (Numpad 1 front / Numpad 3
right) so altitude reads vertically, click a waypoint to select it, scroll to
raise/lower just that point. The green path tube updates live; S saves back to
safezone/flight_path.json (same file/frame draw_path uses).

Open: F3 -> "Edit Path Height", or Ctrl+Shift+H.
Controls:
  LEFT CLICK    select the nearest waypoint (highlighted yellow->white)
  WHEEL UP/DN   raise / lower the SELECTED waypoint's altitude
  [ / ]         previous / next waypoint
  PAGEUP/DN     big altitude step on selected
  L             ramp: linearly interpolate altitude between first & last selected ends
  S             save -> safezone/flight_path.json
  ENTER / ESC   save and finish
  Hold ALT      orbit/pan/zoom
"""
import json
import os

import bpy
import numpy as np

ROOT = os.environ.get("SFM_MAP_ROOT", "").strip()
if not ROOT:
    raise RuntimeError("SFM_MAP_ROOT must explicitly select the authoring workspace")
PATH_JSON = os.environ.get("SFM_FLIGHT_PATH_JSON", f"{ROOT}/safezone/flight_path.json")
ALT_STEP = 0.25
ALT_STEP_BIG = 1.0


def _load():
    d = json.load(open(PATH_JSON))
    return [list(map(float, p)) for p in d.get("waypoints", [])], bool(d.get("closed", False)), d


def _save(wp, closed):
    d = json.load(open(PATH_JSON)) if os.path.exists(PATH_JSON) else {}
    d.update(waypoints=[[float(c) for c in p] for p in wp], closed=closed)
    json.dump(d, open(PATH_JSON, "w"), indent=2)


def _col():
    c = bpy.data.collections.get("FlightPath")
    if c is None:
        c = bpy.data.collections.new("FlightPath")
        bpy.context.scene.collection.children.link(c)
    return c


def _mat(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    if not any(n.type == "EMISSION" for n in nt.nodes):
        nt.nodes.clear()
        o = nt.nodes.new("ShaderNodeOutputMaterial")
        e = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(e.outputs[0], o.inputs["Surface"])
    for n in nt.nodes:
        if n.type == "EMISSION":
            n.inputs["Color"].default_value = (*rgb, 1)
    return m


def _span(wp):
    a = np.asarray(wp, float)
    return float(np.linalg.norm(a.max(0) - a.min(0))) or 1.0 if len(wp) > 1 else 1.0


def _draw(wp, closed, sel):
    col = _col()
    for o in list(col.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    if not wp:
        return
    span = _span(wp)
    line_mat = _mat("path_mat", (0.05, 0.9, 0.1))
    node_mat = _mat("path_node_mat", (1.0, 0.85, 0.05))
    sel_mat = _mat("path_sel_mat", (1.0, 1.0, 1.0))
    loop = wp + [wp[0]] if (closed and len(wp) > 2) else wp
    if len(loop) >= 2:
        cu = bpy.data.curves.new("flight_path", "CURVE"); cu.dimensions = "3D"
        sp = cu.splines.new("POLY"); sp.points.add(len(loop) - 1)
        for i, p in enumerate(loop):
            sp.points[i].co = (p[0], p[1], p[2], 1.0)
        cu.bevel_depth = max(span * 0.004, 1e-3); cu.bevel_resolution = 2
        ob = bpy.data.objects.new("flight_path", cu); ob.data.materials.append(line_mat)
        col.objects.link(ob)
    r = max(span * 0.004, 1e-3)              # smaller markers
    for i, p in enumerate(wp):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r * (1.5 if i == sel else 1.0),
                                             location=(p[0], p[1], p[2]))
        m = bpy.context.active_object; m.name = f"wp_{i:03d}"
        m.data.materials.append(sel_mat if i == sel else node_mat)
        for c in list(m.users_collection):
            c.objects.unlink(m)
        col.objects.link(m)


def _height_edit_event(operator, context, event):
    e = event.type
    if e == "LEFTMOUSE" and event.value == "PRESS":
        operator._sel = operator._pick(context, event)
        _draw(operator._wp, operator._closed, operator._sel)
        operator._status(context)
        return {"RUNNING_MODAL"}

    if e in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and event.value == "PRESS":
        operator._wp[operator._sel][2] += ALT_STEP if e == "WHEELUPMOUSE" else -ALT_STEP
        _draw(operator._wp, operator._closed, operator._sel)
        operator._status(context)
        return {"RUNNING_MODAL"}

    if e in {"PAGE_UP", "PAGE_DOWN"} and event.value == "PRESS":
        operator._wp[operator._sel][2] += ALT_STEP_BIG if e == "PAGE_UP" else -ALT_STEP_BIG
        _draw(operator._wp, operator._closed, operator._sel)
        operator._status(context)
        return {"RUNNING_MODAL"}

    if e in {"LEFT_BRACKET", "RIGHT_BRACKET"} and event.value == "PRESS":
        operator._sel = (operator._sel + (1 if e == "RIGHT_BRACKET" else -1)) % len(operator._wp)
        _draw(operator._wp, operator._closed, operator._sel)
        operator._status(context)
        return {"RUNNING_MODAL"}

    if e == "L" and event.value == "PRESS" and len(operator._wp) >= 2:
        z0, z1 = operator._wp[0][2], operator._wp[-1][2]
        for i in range(len(operator._wp)):
            operator._wp[i][2] = z0 + (z1 - z0) * i / (len(operator._wp) - 1)
        _draw(operator._wp, operator._closed, operator._sel)
        operator._status(context)
        return {"RUNNING_MODAL"}
    return None


def _height_finish_event(operator, context, event):
    e = event.type
    if e == "S" and event.value == "PRESS":
        _save(operator._wp, operator._closed)
        operator.report({"INFO"}, "saved heights")
        return {"RUNNING_MODAL"}

    if e in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
        _save(operator._wp, operator._closed)
        context.area.header_text_set(None)
        operator.report({"INFO"}, f"done, {len(operator._wp)} waypoints saved")
        return {"FINISHED"}

    if e in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM"} or e.startswith("NUMPAD_"):
        return {"PASS_THROUGH"}
    return None


class PATH_OT_edit_height(bpy.types.Operator):
    bl_idname = "view3d.edit_path_height"
    bl_label = "Edit Path Height"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        if not os.path.exists(PATH_JSON):
            self.report({"ERROR"}, f"no path at {PATH_JSON} -- draw it first (Ctrl+Shift+P)")
            return {"CANCELLED"}
        self._wp, self._closed, _ = _load()
        if len(self._wp) < 1:
            self.report({"ERROR"}, "path has no waypoints")
            return {"CANCELLED"}
        self._sel = 0
        _draw(self._wp, self._closed, self._sel)
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _pick(self, context, event):
        rv = context.region_data
        if rv is None:
            return self._sel
        region = context.region
        P = np.array(rv.perspective_matrix)
        co = np.array(self._wp, float)
        hom = np.concatenate([co, np.ones((len(co), 1))], 1)
        clip = hom @ P.T
        w = clip[:, 3]; w[np.abs(w) < 1e-9] = 1e-9
        ndc = clip[:, :2] / w[:, None]
        sx = (ndc[:, 0] * 0.5 + 0.5) * region.width
        sy = (ndc[:, 1] * 0.5 + 0.5) * region.height
        d2 = (sx - event.mouse_region_x) ** 2 + (sy - event.mouse_region_y) ** 2
        d2[w < 0] = np.inf
        return int(np.argmin(d2))

    def _status(self, context):
        z = self._wp[self._sel][2]
        context.area.header_text_set(
            f"EditHeight  wp {self._sel+1}/{len(self._wp)}  alt={z:.2f}  |  "
            "Click=select  Wheel=alt  [ ]=prev/next  PgUp/Dn=big  L=ramp  S=save  Enter/Esc=done  Alt=orbit")
        context.area.tag_redraw()

    def modal(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            return {"PASS_THROUGH"}
        if event.alt:
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.shift:   # Shift+LMB = pan (global keymap)
            return {"PASS_THROUGH"}
        result = _height_edit_event(self, context, event)
        if result is not None:
            return result
        result = _height_finish_event(self, context, event)
        if result is not None:
            return result
        return {"RUNNING_MODAL"}


class PATH_OT_rebuild_from_markers(bpy.types.Operator):
    """Re-connect the path from the current wp_NNN marker positions (use after
    moving balls with native G/grab) and save to flight_path.json."""
    bl_idname = "view3d.rebuild_path_from_markers"
    bl_label = "Rebuild Path From Markers"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wp, i = [], 0
        while True:
            o = bpy.data.objects.get(f"wp_{i:03d}")
            if o is None:
                break
            wp.append([float(o.location.x), float(o.location.y), float(o.location.z)])
            i += 1
        if len(wp) < 2:
            self.report({"ERROR"}, "need >=2 wp_NNN markers")
            return {"CANCELLED"}
        closed = _load()[1] if os.path.exists(PATH_JSON) else False
        _draw(wp, closed, -1)
        _save(wp, closed)
        self.report({"INFO"}, f"rebuilt path from {len(wp)} markers -> saved")
        return {"FINISHED"}


_km = []


def _menu(self, context):
    self.layout.operator(PATH_OT_edit_height.bl_idname, text="Edit Path Height", icon="SORTSIZE")
    self.layout.operator(PATH_OT_rebuild_from_markers.bl_idname, text="Rebuild Path From Markers", icon="CURVE_PATH")


def register():
    for cls in (PATH_OT_edit_height, PATH_OT_rebuild_from_markers):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_view.append(_menu)
    wm = bpy.context.window_manager
    for kc in (wm.keyconfigs.addon, wm.keyconfigs.user, wm.keyconfigs.active):
        if not kc:
            continue
        km = kc.keymaps.get("3D View") or kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        if not any(k.idname == PATH_OT_edit_height.bl_idname for k in km.keymap_items):
            _km.append((km, km.keymap_items.new(PATH_OT_edit_height.bl_idname, "H", "PRESS", ctrl=True, shift=True)))
        if not any(k.idname == PATH_OT_rebuild_from_markers.bl_idname for k in km.keymap_items):
            _km.append((km, km.keymap_items.new(PATH_OT_rebuild_from_markers.bl_idname, "R", "PRESS", ctrl=True, shift=True)))
        break
    print("[edit_height] registered: Ctrl+Shift+H edit height | Ctrl+Shift+R rebuild from markers. Use a SIDE view.")


if __name__ == "__main__" or True:
    register()
