#!/usr/bin/env python3
"""Show numbered route labels for safezone/flight_path.json in Blender.

Run with live Blender/MCP:
  python3 scripts/blender_send.py codefile scripts/label_route_points.py

Creates text objects route_label_001, route_label_002, ... in the same ALIGNED
Blender frame as the drawn flight path.  Labels are visual aids only; the route
itself remains safezone/flight_path.json.
"""
import json
import os

import bpy

ROOT = os.environ.get("SFM_MAP_ROOT", "").strip()
if not ROOT:
    raise RuntimeError("SFM_MAP_ROOT must explicitly select the authoring workspace")
PATH_JSON = os.environ.get("SFM_FLIGHT_PATH_JSON", f"{ROOT}/safezone/flight_path.json")
COLLECTION = "RoutePointLabels"


def _label_material():
    mat = bpy.data.materials.get("route_label_mat")
    if mat is None:
        mat = bpy.data.materials.new("route_label_mat")
        mat.diffuse_color = (1.0, 0.9, 0.05, 1.0)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 0.9, 0.05, 1.0)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (1.0, 0.85, 0.05, 1.0)
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 2.0
    return mat


def _collection():
    col = bpy.data.collections.get(COLLECTION)
    if col is None:
        col = bpy.data.collections.new(COLLECTION)
        bpy.context.scene.collection.children.link(col)
    return col


def _clear_labels():
    col = _collection()
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _load_waypoints():
    if not os.path.exists(PATH_JSON):
        raise FileNotFoundError(PATH_JSON)
    d = json.load(open(PATH_JSON))
    return [list(map(float, p)) for p in d.get("waypoints", [])]


def _make_label(i, p, mat):
    # p is in aligned Blender frame [x, y, z] with +Z up.
    curve = bpy.data.curves.new(f"route_label_{i:03d}", type="FONT")
    curve.body = str(i)
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.size = 0.32
    curve.resolution_u = 12
    obj = bpy.data.objects.new(f"route_label_{i:03d}", curve)
    obj.location = (p[0], p[1], p[2] + 0.42)
    obj.show_in_front = True
    obj.data.materials.append(mat)
    _collection().objects.link(obj)

    # A small black backing disk makes the number legible on bright point clouds.
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.23, depth=0.015,
                                       location=(p[0], p[1], p[2] + 0.39),
                                       rotation=(0, 0, 0))
    disk = bpy.context.active_object
    disk.name = f"route_label_back_{i:03d}"
    disk.show_in_front = True
    black = bpy.data.materials.get("route_label_back_mat") or bpy.data.materials.new("route_label_back_mat")
    black.diffuse_color = (0.0, 0.0, 0.0, 0.75)
    disk.data.materials.append(black)
    # Move disk from master collection into label collection.
    for col in list(disk.users_collection):
        col.objects.unlink(disk)
    _collection().objects.link(disk)


def refresh_labels():
    wps = _load_waypoints()
    _clear_labels()
    mat = _label_material()
    for i, p in enumerate(wps, start=1):
        _make_label(i, p, mat)
    print(f"[route-labels] created {len(wps)} labels from {PATH_JSON}")
    return len(wps)


refresh_labels()
