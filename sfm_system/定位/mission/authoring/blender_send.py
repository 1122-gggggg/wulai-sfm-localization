#!/usr/bin/env python3
"""Tiny client for the blender-mcp addon socket (127.0.0.1:9876).

Drives the live Blender that scripts/blender_mcp_launch.py started, using the
official protocol {"type": ..., "params": {...}}. Lets me run code in Blender
and read its printed output without the MCP layer.

  python3 blender_send.py codefile /tmp/op.py     # run a .py file in Blender
  python3 blender_send.py code "import bpy; print(len(bpy.data.objects))"
  python3 blender_send.py get_scene_info           # any addon command, no params
"""
import json
import socket
import sys


def send(cmd: dict, host="127.0.0.1", port=9876, timeout=120) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((host, port))
    s.sendall(json.dumps(cmd).encode("utf-8"))
    buf = b""
    while True:
        try:
            d = s.recv(65536)
        except socket.timeout:
            break
        if not d:
            break
        buf += d
        try:
            json.loads(buf.decode("utf-8"))
            break
        except Exception:
            continue
    s.close()
    try:
        return json.loads(buf.decode("utf-8"))
    except Exception:
        return {"status": "error", "message": "no/invalid response", "raw": buf.decode("utf-8", "replace")}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "get_scene_info"
    if mode == "codefile":
        code = open(sys.argv[2]).read()
        resp = send({"type": "execute_code", "params": {"code": code}})
    elif mode == "code":
        resp = send({"type": "execute_code", "params": {"code": sys.argv[2]}})
    else:
        resp = send({"type": mode, "params": {}})
    # surface execute_code stdout nicely
    if resp.get("status") == "success" and isinstance(resp.get("result"), dict) \
            and "result" in resp["result"]:
        out = resp["result"]["result"]
        print(out if out else "[ok] (no stdout)")
    else:
        print(json.dumps(resp, ensure_ascii=False, indent=2))
