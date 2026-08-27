#!/usr/bin/env python3
"""Load the Target Site RGB cloud into Blender's aligned Z-up authoring frame."""
import os
import json

import bpy
import bmesh
import numpy as np
from mathutils import Matrix


MAP_PLY = os.environ["SFM_MAP_PLY"]
REFERENCE_POSES = os.environ.get("SFM_MAP_REFERENCE_POSES", "")
POINT_RADIUS = 0.004

# Original GLOMAP [x,y,z] -> Blender aligned [x,z,-y]. This is the exact inverse
# of flight_operator_app.load_route_glomap / aligned_to_glomap.
GLOMAP_TO_ALIGNED = Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection != bpy.context.scene.collection:
            bpy.data.collections.remove(collection)


def point_material():
    """Render the PLY's point-domain Col attribute as the point color."""
    material = bpy.data.materials.new("TargetMapRGB")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    color = nodes.new("ShaderNodeAttribute")
    color.attribute_name = "Col"
    links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        links.new(color.outputs["Color"], bsdf.inputs["Emission Color"])
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.2
    bsdf.inputs["Roughness"].default_value = 1.0
    return material


def add_point_display(obj, radius):
    """Show a vertex-only PLY as points while retaining its mesh for click snapping."""
    group = bpy.data.node_groups.new("TargetMapPoints", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    node_in = group.nodes.new("NodeGroupInput")
    node_out = group.nodes.new("NodeGroupOutput")
    to_points = group.nodes.new("GeometryNodeMeshToPoints")
    set_material = group.nodes.new("GeometryNodeSetMaterial")
    to_points.mode = "VERTICES"
    to_points.inputs["Radius"].default_value = float(radius)
    set_material.inputs["Material"].default_value = point_material()
    group.links.new(node_in.outputs["Geometry"], to_points.inputs["Mesh"])
    group.links.new(to_points.outputs["Points"], set_material.inputs["Geometry"])
    group.links.new(set_material.outputs["Geometry"], node_out.inputs["Geometry"])
    modifier = obj.modifiers.new("TargetMapPointDisplay", "NODES")
    modifier.node_group = group


def reference_bounds():
    if not REFERENCE_POSES:
        raise RuntimeError("SFM_MAP_REFERENCE_POSES is required for outlier filtering")
    poses = json.load(open(REFERENCE_POSES, encoding="utf-8")).get("poses", {})
    centers = []
    for pose in poses.values():
        R = np.asarray(pose["R"], dtype=float)
        t = np.asarray(pose["t"], dtype=float)
        centers.append(-R.T @ t)
    if not centers:
        raise RuntimeError(f"no reference poses in {REFERENCE_POSES}")
    centers = np.asarray(centers)
    return centers.min(axis=0) - 5.0, centers.max(axis=0) + 5.0


clear_scene()
bpy.ops.wm.ply_import(filepath=MAP_PLY)
obj = bpy.context.active_object
if obj is None or obj.type != "MESH":
    raise RuntimeError(f"Target Site PLY did not import as a mesh: {MAP_PLY}")
obj.name = "map_cloud"
obj.data.name = "target_site_v1_rgb_cloud"

coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
obj.data.vertices.foreach_get("co", coords)
coords = coords.reshape(-1, 3)
raw_count = len(coords)
raw_lo, raw_hi = reference_bounds()
keep = ((coords >= raw_lo) & (coords <= raw_hi)).all(axis=1)
mesh = bmesh.new()
mesh.from_mesh(obj.data)
mesh.verts.ensure_lookup_table()
bmesh.ops.delete(
    mesh,
    geom=[vertex for vertex, selected in zip(mesh.verts, keep) if not selected],
    context="VERTS",
)
mesh.to_mesh(obj.data)
mesh.free()
obj.data.update()

coords = coords[keep]
obj.matrix_world = GLOMAP_TO_ALIGNED
aligned = np.column_stack((coords[:, 0], coords[:, 2], -coords[:, 1]))
lo = aligned.min(axis=0)
hi = aligned.max(axis=0)
center = 0.5 * (lo + hi)
span = float(np.linalg.norm(hi - lo)) or 1.0
add_point_display(obj, radius=POINT_RADIUS)

obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.context.scene["coordinate_frame"] = "target_site_v1_aligned_Zup"
bpy.context.scene["glomap_to_aligned"] = "[x,z,-y]"
bpy.context.scene["map_ply"] = MAP_PLY

for area in bpy.context.screen.areas:
    if area.type != "VIEW_3D":
        continue
    area.spaces.active.shading.type = "MATERIAL"
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    if region is None:
        continue
    with bpy.context.temp_override(area=area, region=region, space_data=area.spaces.active):
        bpy.ops.view3d.view_axis(type="TOP", align_active=False)
        bpy.ops.view3d.view_all(center=False)
    area.spaces.active.region_3d.view_location = center
    area.spaces.active.region_3d.view_distance = span * 0.65

print(
    f"[target-site] reference-bound filter kept {len(obj.data.vertices)}/{raw_count} map points; "
    f"aligned bounds={lo.tolist()}..{hi.tolist()} frame=[x,z,-y]",
    flush=True,
)
