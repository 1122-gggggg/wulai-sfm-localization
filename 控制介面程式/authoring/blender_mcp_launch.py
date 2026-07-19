#!/usr/bin/env python3
"""Blender startup: safezone scene + the OFFICIAL blender-mcp addon server.

Runs the safezone setup (cloud + starter SafeVolume), then registers
ahujasid/blender-mcp's addon and starts its socket server on 127.0.0.1:9876.
That server is what the blender-mcp MCP server (uvx blender-mcp) talks to, and
it also accepts {"type":"execute_code","params":{"code":...}} directly, so it
can be driven live via scripts/blender_send.py.

Launch:
  DISPLAY=:1 .../blender --python scripts/blender_mcp_launch.py
"""
import os
import sys

import bpy

ROOT = os.environ.get("SFM_MAP_ROOT", "/media/cihcilab/新增磁碟區/sfm_glomap")
TOOLS = os.environ.get("SFM_TOOLS_ROOT", "/media/cihcilab/新增磁碟區/tools")
SETUP = os.environ.get("SFM_BLENDER_SETUP", f"{ROOT}/scripts/blender_safezone.py")

# 1) build the safezone scene
with open(SETUP) as f:
    exec(compile(f.read(), SETUP, "exec"), {"__name__": "__setup__"})

# 2) register the official blender-mcp addon
sys.path.insert(0, TOOLS)
import blender_mcp_addon as bmcp                       # noqa: E402
try:
    bmcp.register()
    print("[mcp] addon registered", flush=True)
except Exception as e:                                 # already-registered etc.
    print("[mcp] register warning:", e, flush=True)

# 3) start the addon's socket server on 9876
sc = bpy.context.scene
try:
    if not getattr(bpy.types, "blendermcp_server", None):
        bpy.types.blendermcp_server = bmcp.BlenderMCPServer(
            port=getattr(sc, "blendermcp_port", 9876))
    bpy.types.blendermcp_server.start()
    try:
        sc.blendermcp_server_running = True
    except Exception:
        pass
    print("[mcp] blender-mcp server started on 127.0.0.1:9876", flush=True)
except Exception as e:
    print("[mcp] start error:", e, flush=True)
