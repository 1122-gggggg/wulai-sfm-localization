#!/usr/bin/env python3
"""Serve the flight operator interface with stub APIs.

This is intentionally dependency-free. It is a bridge point for the real Olympe
backend: replace the stub handlers for /api/control, /api/state, /api/video.jpg,
and /api/map_points with the live drone/localizer state.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATE = {
    "mode": "MANUAL",
    "loc": "SIM",
    "trackerState": "HOVER",
    "inliers": 0,
    "reproj": None,
}
START = time.monotonic()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[operator-ui] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, payload, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/flight_operator_interface.html"
            return super().do_GET()
        if self.path.startswith("/api/state"):
            t = time.monotonic() - START
            mode = str(STATE.get("mode", "MANUAL"))
            tracker = "TRACK" if mode == "AUTO" else str(STATE.get("trackerState", "HOVER"))
            return self._json({
                "connected": True,
                "mode": mode,
                "loc": "SIM",
                "trackerState": tracker,
                "pose": {
                    "x": math.sin(t * 0.18) * 1.8,
                    "y": -1.2,
                    "z": math.cos(t * 0.18) * 1.2,
                    "yaw": t * 0.18,
                },
                "inliers": 180 + int(40 * math.sin(t)),
                "reproj": 2.5 + 0.2 * math.cos(t * 0.5),
            })
        if self.path.startswith("/api/map_points"):
            pts = []
            for i in range(2400):
                a = i * 0.037
                r = 1.0 + (i % 97) / 35.0
                pts.append([math.cos(a) * r, -1.0 + ((i % 17) / 20.0), math.sin(a) * r])
            return self._json({"points": pts})
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/control"):
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}
            command = str(payload.get("command", ""))
            if command in {"manual", "hover", "land"}:
                STATE["mode"] = "MANUAL"
            elif command in {"auto", "start_auto"}:
                STATE["mode"] = "AUTO"
            if command == "hover":
                STATE["trackerState"] = "HOVER"
            elif command == "land":
                STATE["trackerState"] = "LAND"
            return self._json({"ok": True, "command": command, "state": STATE})
        return self._json({"ok": False, "error": "unknown endpoint"}, status=404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"operator interface: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
