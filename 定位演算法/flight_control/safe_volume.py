#!/usr/bin/env python3
"""Treat the safe zone as a hand-modelled 3D VOLUME (a closed mesh).

RETIRED: no production or legacy flight code reads a safety tube, slow band,
or clearance field. This module is kept only so old analysis scripts that
import SafeVolume keep importing; nothing steers, slows, or repels off it.

NOTE: not wired into the production flight path (path_follow_flight).
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


class SafeVolume:
    def __init__(self, mesh_path: str):
        m = o3d.io.read_triangle_mesh(mesh_path)
        if not m.has_triangles():
            raise ValueError(f"{mesh_path} has no triangles")
        self.watertight = m.is_watertight()
        if not self.watertight:
            # A non-watertight mesh makes compute_signed_distance's inside/outside sign
            # meaningless -> inside()/clearance() would silently lie. Fail closed.
            raise ValueError(
                f"{mesh_path} is not watertight; signed-distance containment is unreliable. "
                "Repair the mesh (closed, outward normals) before using it as a geofence.")
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    def _sd(self, pts: np.ndarray) -> np.ndarray:
        q = o3d.core.Tensor(np.atleast_2d(pts).astype(np.float32))
        # o3d signed distance: >0 OUTSIDE, <0 INSIDE for a watertight mesh
        return self.scene.compute_signed_distance(q).numpy()

    def clearance(self, x: float, y: float, z: float) -> float:
        """>0 inside (distance to nearest wall), <0 outside."""
        return float(-self._sd(np.array([x, y, z]))[0])

    def inside(self, x: float, y: float, z: float) -> bool:
        return self.clearance(x, y, z) > 0.0

    def inward_dir(self, x: float, y: float, z: float, eps: float = 0.05):
        """Unit vector toward increasing clearance (the interior)."""
        p = np.array([x, y, z], float)
        g = np.zeros(3)
        for ax in range(3):
            d = np.zeros(3)
            d[ax] = eps
            g[ax] = (-self._sd(p + d)[0]) - (-self._sd(p - d)[0])
        n = np.linalg.norm(g)
        return (g / n) if n > 1e-9 else g
