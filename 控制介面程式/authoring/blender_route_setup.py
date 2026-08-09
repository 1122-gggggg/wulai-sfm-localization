#!/usr/bin/env python3
"""Blender startup for route authoring: load a site's cloud, arm the click tools.

Runs inside Blender (``blender --python this_file``). Reads the site from
SFM_MAP_PLY / SFM_SAFEZONE_DIR, which 航線編修.sh fills in from a site profile.

Frame: the cloud is stored in the GLOMAP frame and drawn in Blender's aligned
Z-up frame (ground = z=0, up = +Z), the same convention
flight_operator_app.load_route_glomap expects, so a saved path drops straight
into a site profile's route_json.

Navigation is remapped to what the operator asked for:
  Ctrl + LMB drag          orbit
  Ctrl + Shift + LMB drag  pan
  Wheel                    zoom
The view opens top-down orthographic so the route can be traced over the map,
then heights are adjusted per waypoint from a side view with G then Z.
"""
import json
import os

import bpy
import numpy as np
from mathutils import Matrix

MAP_PLY = os.environ["SFM_MAP_PLY"]
OUT_DIR = os.environ.get("SFM_SAFEZONE_DIR", "/tmp")
REFERENCE_POSES = os.environ.get("SFM_MAP_REFERENCE_POSES", "")
TOOLS_SCRIPT = os.environ.get("SFM_ROUTE_TOOLS", "")
# Display radius of each cloud point, in map units. Small enough to read fine
# structure (wires, poles) rather than a blob of overlapping spheres.
POINT_RADIUS = float(os.environ.get("SFM_CLOUD_POINT_RADIUS", "0.004"))

# Legacy alignment: GLOMAP [x,y,z] -> [x,z,-y], i.e. it assumes GLOMAP +Y points
# down. That assumption is wrong for target_site_v1 by 22.5 degrees, so a
# per-site T_align_gravity.json (measured from the camera poses) wins when present.
LEGACY_GLOMAP_TO_ALIGNED = Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


def load_alignment():
    """Gravity-aligned transform for this site, else the legacy axis swap.

    Getting this wrong tilts the whole authoring frame: 'up' in Blender stops
    being up, so raising a waypoint with G,Z also shoves it sideways.
    """
    path = os.environ.get("SFM_MAP_ALIGN", "")
    if not path:
        guess = os.path.join(os.path.dirname(MAP_PLY), "T_align_gravity.json")
        path = guess if os.path.isfile(guess) else ""
    if not path or not os.path.isfile(path):
        print("[route] alignment: legacy [x,z,-y] (no T_align_gravity.json found)", flush=True)
        return LEGACY_GLOMAP_TO_ALIGNED, None
    data = json.load(open(path, encoding="utf-8"))
    R = data["R"]
    m = Matrix((
        (R[0][0], R[0][1], R[0][2], 0.0),
        (R[1][0], R[1][1], R[1][2], 0.0),
        (R[2][0], R[2][1], R[2][2], 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    dev = (data.get("derivation") or {}).get("deviation_from_old_Yup_deg")
    print(f"[route] alignment: {os.path.basename(path)} "
          f"(gravity measured from poses; legacy axis swap was off by {dev:.2f} deg)",
          flush=True)
    return m, data


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def import_cloud(path: str, align):
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=path)
    elif hasattr(bpy.ops.import_mesh, "ply"):
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise RuntimeError("this Blender has no PLY import operator")
    obj = bpy.context.selected_objects[0]
    obj.name = "SiteCloud"
    obj.matrix_world = align @ obj.matrix_world
    return obj


def point_cloud_display(obj, radius: float) -> None:
    """Render the loose PLY vertices as small coloured spheres.

    A PLY of bare vertices draws as nothing in Material Preview, so the colour
    attribute never shows. Geometry Nodes turns each vertex into a point of a
    fixed radius, which both makes the colour visible and gives the clicks
    something with real extent to land on.
    """
    mod = obj.modifiers.new("CloudPoints", "NODES")
    tree = bpy.data.node_groups.new("SiteCloudPoints", "GeometryNodeTree")
    mod.node_group = tree
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    n_in = nodes.new("NodeGroupInput")
    n_out = nodes.new("NodeGroupOutput")
    to_points = nodes.new("GeometryNodeMeshToPoints")
    to_points.mode = "VERTICES"
    to_points.inputs["Radius"].default_value = radius
    set_mat = nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = obj.data.materials[0]
    links.new(n_in.outputs[0], to_points.inputs["Mesh"])
    links.new(to_points.outputs["Points"], set_mat.inputs["Geometry"])
    links.new(set_mat.outputs["Geometry"], n_out.inputs[0])
    print(f"[route] point cloud radius {radius:.5f} map units", flush=True)


def vertex_colour_material(mesh):
    """Show the PLY's per-point colour instead of flat grey."""
    names = {a.name for a in getattr(mesh, "color_attributes", [])}
    names |= {a.name for a in getattr(mesh, "attributes", [])}
    attr_name = next((c for c in ("Col", "Color", "color", "rgb", "RGB") if c in names), None)
    mat = bpy.data.materials.new("SiteCloudRGB")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    if attr_name:
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = attr_name
        links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mesh.materials.append(mat)
    return mat


def remap_navigation() -> None:
    """Ctrl+LMB orbit, Ctrl+Shift+LMB pan, wheel zoom.

    Added on top of the default keymap rather than replacing it, so the stock
    middle-mouse navigation still works for anyone used to it.
    """
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is None:
        return
    km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
    km.keymap_items.new("view3d.rotate", "LEFTMOUSE", "PRESS", ctrl=True)
    km.keymap_items.new("view3d.move", "LEFTMOUSE", "PRESS", ctrl=True, shift=True)
    km.keymap_items.new("view3d.zoom", "WHEELINMOUSE", "PRESS")
    km.keymap_items.new("view3d.zoom", "WHEELOUTMOUSE", "PRESS")
    print("[route] navigation: Ctrl+LMB orbit, Ctrl+Shift+LMB pan, wheel zoom", flush=True)


def frame_top_down(obj) -> None:
    """Open top-down orthographic, framed on the cloud's robust extent."""
    verts = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
    if len(verts) == 0:
        return
    world = np.array(obj.matrix_world.to_3x3()) @ verts.T
    world = (world.T + np.array(obj.matrix_world.translation)).astype(np.float64)
    lo = np.percentile(world, 0.1, axis=0)
    hi = np.percentile(world, 99.9, axis=0)
    centre = (lo + hi) / 2.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) or 1.0
    screen = getattr(bpy.context, "screen", None)
    if screen is None:                       # --background has no viewport
        print(f"[route] (background) span {span:.3f} map units", flush=True)
        return
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        space.shading.type = "MATERIAL"
        space.overlay.show_floor = False
        space.overlay.show_axis_x = True
        space.overlay.show_axis_y = True
        space.clip_start = 0.001
        space.clip_end = max(1000.0, span * 20.0)
        r3d = space.region_3d
        r3d.view_perspective = "ORTHO"
        r3d.view_rotation = (1.0, 0.0, 0.0, 0.0)      # looking straight down -Z
        r3d.view_location = (float(centre[0]), float(centre[1]), float(centre[2]))
        r3d.view_distance = span * 1.2
    print(f"[route] top-down ortho, span {span:.3f} map units", flush=True)


def main() -> None:
    clear_scene()
    align, align_meta = load_alignment()
    obj = import_cloud(MAP_PLY, align)
    vertex_colour_material(obj.data)
    point_cloud_display(obj, POINT_RADIUS)
    if align_meta is not None:
        bpy.context.scene["sfm_align"] = json.dumps(align_meta.get("R"))
        bpy.context.scene["sfm_gravity_glomap"] = json.dumps(align_meta.get("gravity_glomap"))
    remap_navigation()
    frame_top_down(obj)
    os.makedirs(OUT_DIR, exist_ok=True)
    bpy.context.scene["sfm_route_out_dir"] = OUT_DIR
    bpy.context.scene["sfm_map_ply"] = MAP_PLY
    if TOOLS_SCRIPT and os.path.isfile(TOOLS_SCRIPT):
        # Register the click/height operators in this same Blender session.
        with open(TOOLS_SCRIPT, encoding="utf-8") as fh:
            exec(compile(fh.read(), TOOLS_SCRIPT, "exec"), {"__name__": "__route_tools__"})  # noqa: S102 - Blender must load the selected authoring operators in-process.
    print(f"[route] cloud={MAP_PLY}", flush=True)
    print(f"[route] output dir={OUT_DIR}", flush=True)


main()
