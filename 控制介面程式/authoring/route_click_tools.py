#!/usr/bin/env python3
"""Click waypoints on the cloud, then adjust each one's height, then export.

Registered into a running Blender by blender_route_setup.py. Two stages, matching
how a cruise route is actually authored:

  Stage 1 (top-down)  click waypoints; XY snaps to the cloud under the cursor,
                      falling back to the z=0 ground plane over gaps. They are
                      joined into a single polyline as you go.
  Stage 2 (any view)  each waypoint is a marker object. Select one and press
                      G then Z to slide it vertically, exactly like any other
                      Blender object. Then export.

Output goes to the site's route dir as flight_path.json + flight_path.ply, in the
aligned Z-up frame that flight_operator_app.load_route_glomap reads.

Panel: 3D viewport sidebar (press N) -> "航線" tab.
"""
import json
import os

import bpy
import bmesh
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector

WP_COLLECTION = "FlightPathWaypoints"
CURVE_NAME = "FlightPath"
MARKER_PREFIX = "WP_"


def _out_dir() -> str:
    return bpy.context.scene.get("sfm_route_out_dir", "/tmp")


def _cloud():
    return bpy.data.objects.get("SiteCloud")


def _collection() -> "bpy.types.Collection":
    coll = bpy.data.collections.get(WP_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(WP_COLLECTION)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _markers() -> list:
    coll = bpy.data.collections.get(WP_COLLECTION)
    if coll is None:
        return []
    items = [o for o in coll.objects if o.name.startswith(MARKER_PREFIX)]
    return sorted(items, key=lambda o: o.name)


def _marker_radius() -> float:
    """Scale markers to the map so they stay clickable on any site."""
    cloud = _cloud()
    if cloud is None:
        return 0.02
    dims = cloud.dimensions
    span = max(float(dims.x), float(dims.y)) or 1.0
    return max(span * 0.004, 1e-4)


def _add_marker(index: int, location) -> "bpy.types.Object":
    radius = _marker_radius()
    mesh = bpy.data.meshes.new(f"{MARKER_PREFIX}{index:03d}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(f"{MARKER_PREFIX}{index:03d}", mesh)
    obj.location = location
    obj.show_in_front = True
    mat = bpy.data.materials.get("WaypointMat")
    if mat is None:
        mat = bpy.data.materials.new("WaypointMat")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        emit = mat.node_tree.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = (1.0, 0.85, 0.1, 1.0)
        emit.inputs["Strength"].default_value = 2.0
        mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mesh.materials.append(mat)
    _collection().objects.link(obj)
    return obj


def rebuild_curve() -> None:
    """Redraw the polyline through the current marker positions."""
    pts = [tuple(o.location) for o in _markers()]
    old = bpy.data.objects.get(CURVE_NAME)
    if old is not None:
        data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if isinstance(data, bpy.types.Curve) and data.users == 0:
            bpy.data.curves.remove(data)
    if len(pts) < 2:
        return
    curve = bpy.data.curves.new(CURVE_NAME, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = _marker_radius() * 0.35
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        spline.points[i].co = (p[0], p[1], p[2], 1.0)
    obj = bpy.data.objects.new(CURVE_NAME, curve)
    mat = bpy.data.materials.get("FlightPathMat")
    if mat is None:
        mat = bpy.data.materials.new("FlightPathMat")
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        emit = mat.node_tree.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = (0.1, 1.0, 0.35, 1.0)
        emit.inputs["Strength"].default_value = 2.0
        mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    curve.materials.append(mat)
    _collection().objects.link(obj)


def _snap_to_cloud(context, event):
    """Ray from the cursor -> nearest cloud point, else the z=0 ground plane."""
    region = context.region
    r3d = context.space_data.region_3d
    coord = (event.mouse_region_x, event.mouse_region_y)
    direction = view3d_utils.region_2d_to_vector_3d(region, r3d, coord)
    origin = view3d_utils.region_2d_to_origin_3d(region, r3d, coord)

    cloud = _cloud()
    if cloud is not None and len(cloud.data.vertices):
        mw = cloud.matrix_world
        pts = np.array([(mw @ v.co)[:] for v in cloud.data.vertices], dtype=np.float64)
        o = np.array(origin[:], dtype=np.float64)
        d = np.array(direction[:], dtype=np.float64)
        d /= (np.linalg.norm(d) or 1.0)
        rel = pts - o
        t = rel @ d
        ahead = t > 0
        if ahead.any():
            perp = np.linalg.norm(rel[ahead] - np.outer(t[ahead], d), axis=1)
            span = max(float(cloud.dimensions.x), float(cloud.dimensions.y)) or 1.0
            if perp.min() < span * 0.02:
                return Vector(pts[ahead][int(perp.argmin())])

    if abs(direction.z) < 1e-9:
        return None
    t = -origin.z / direction.z
    if t <= 0:
        return None
    return origin + direction * t


class ROUTE_OT_click(bpy.types.Operator):
    """Click waypoints on the map; they auto-connect into one route."""

    bl_idname = "route.click_waypoints"
    bl_label = "點選航點"

    def modal(self, context, event):
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            return {"PASS_THROUGH"}
        # Ctrl-modified drags are navigation (orbit/pan), not waypoint clicks.
        if event.type == "LEFTMOUSE" and event.ctrl:
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = _snap_to_cloud(context, event)
            if hit is not None:
                _add_marker(len(_markers()) + 1, hit)
                rebuild_curve()
                context.area.header_text_set(f"航點 {len(_markers())} 個 | ENTER 完成 | Z 復原")
            return {"RUNNING_MODAL"}
        if event.type == "Z" and event.value == "PRESS" and not event.ctrl:
            items = _markers()
            if items:
                bpy.data.objects.remove(items[-1], do_unlink=True)
                rebuild_curve()
                context.area.header_text_set(f"航點 {len(_markers())} 個 | ENTER 完成 | Z 復原")
            return {"RUNNING_MODAL"}
        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            context.area.header_text_set(None)
            self.report({"INFO"}, f"完成，共 {len(_markers())} 個航點。接著調高度，然後匯出。")
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        if context.space_data.type != "VIEW_3D":
            self.report({"WARNING"}, "請在 3D 視圖中執行")
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("左鍵點航點 | Ctrl+左鍵旋轉 | Ctrl+Shift+左鍵平移 | Z 復原 | ENTER 完成")
        return {"RUNNING_MODAL"}


class ROUTE_OT_export(bpy.types.Operator):
    """Write flight_path.json + flight_path.ply from the current markers."""

    bl_idname = "route.export"
    bl_label = "匯出航線"

    def execute(self, context):
        pts = [[float(c) for c in o.location] for o in _markers()]
        if len(pts) < 2:
            self.report({"WARNING"}, "至少需要 2 個航點")
            return {"CANCELLED"}
        out = _out_dir()
        os.makedirs(out, exist_ok=True)
        path_json = os.path.join(out, "flight_path.json")
        with open(path_json, "w", encoding="utf-8") as fh:
            json.dump({
                "waypoints": pts,
                "closed": False,
                "frame": "aligned",
                "units": "map",
                "source": "blender_route_click_tools",
                "map_ply": context.scene.get("sfm_map_ply", ""),
            }, fh, ensure_ascii=False, indent=2)

        # Densely sampled polyline so the UI overlay and Blender agree.
        samples = []
        for a, b in zip(pts, pts[1:]):
            a_v, b_v = np.array(a), np.array(b)
            n = max(2, int(np.linalg.norm(b_v - a_v) / (_marker_radius() * 0.5)) + 1)
            for i in range(n):
                samples.append(a_v + (b_v - a_v) * (i / n))
        samples.append(np.array(pts[-1]))
        ply = os.path.join(out, "flight_path.ply")
        with open(ply, "w", encoding="ascii") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(samples)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write(f"element edge {len(samples) - 1}\n")
            fh.write("property int vertex1\nproperty int vertex2\nend_header\n")
            for s in samples:
                fh.write(f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f} 26 255 89\n")
            for i in range(len(samples) - 1):
                fh.write(f"{i} {i + 1}\n")

        self.report({"INFO"}, f"已寫出 {len(pts)} 個航點 -> {path_json}")
        print(f"[route] wrote {path_json}", flush=True)
        print(f"[route] wrote {ply}", flush=True)
        return {"FINISHED"}


class ROUTE_OT_refresh(bpy.types.Operator):
    """Redraw the polyline after markers were moved with G/Z."""

    bl_idname = "route.refresh_curve"
    bl_label = "更新航線"

    def execute(self, context):
        rebuild_curve()
        return {"FINISHED"}


class ROUTE_PT_panel(bpy.types.Panel):
    bl_label = "航線編修"
    bl_idname = "ROUTE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "航線"

    def draw(self, context):
        col = self.layout.column()
        col.label(text=f"航點：{len(_markers())} 個")
        col.operator("route.click_waypoints", icon="MOUSE_LMB")
        col.separator()
        col.label(text="調高度：選航點按 G 再按 Z")
        col.operator("route.refresh_curve", icon="FILE_REFRESH")
        col.separator()
        col.operator("route.export", icon="EXPORT")
        col.separator()
        box = col.box()
        box.label(text="Ctrl+左鍵：旋轉")
        box.label(text="Ctrl+Shift+左鍵：平移")
        box.label(text="滾輪：縮放")


_CLASSES = (ROUTE_OT_click, ROUTE_OT_export, ROUTE_OT_refresh, ROUTE_PT_panel)


def register() -> None:
    for cls in _CLASSES:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass


register()
print("[route] click tools ready -- 3D 視圖按 N，選「航線」分頁", flush=True)
