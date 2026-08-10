#!/usr/bin/env python3
"""Global safest-path planner for pole inspection, on an SDF voxel grid.

Two-layer architecture (agreed):
  GLOBAL (this file): at mission start, plan a CLEARANCE-PREFERRING path that
    visits the inspection targets (poles) while staying inside the safe zone and
    clear of structures. Poles/wires are baked into the SDF as no-go WITH buffer,
    so the path keeps standoff by construction.
  LOCAL (cruise_geofence.py, reactive): follow this path while reacting live to
    SDF clearance (geofence repulsion), YOLO pole bearing + bbox-area standoff,
    and localization; stop/hover/recover is the hard safety override.

SDF convention: sdf[i,j,k] = signed distance (MAP UNITS) to the nearest UNSAFE
surface; POSITIVE inside the flyable & structure-clear region, <=0 outside or
inside a structure (pole/wire no-go). Built offline from the dense MVS map +
drawn safe zone + pole/wire buffers. Everything here is scale-free (map units);
only the YOLO standoff in the LOCAL layer needs a metric notion.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

# 26-connected neighborhood
_NB = [
    (a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)
]


@dataclass
class SDFGrid:
    sdf: np.ndarray  # (nx,ny,nz) signed distance in map units
    origin: np.ndarray  # world coord of voxel (0,0,0)
    voxel: float  # voxel edge length (map units)

    def w2g(self, p):
        return (np.asarray(p, float) - self.origin) / self.voxel

    def g2w(self, idx):
        return self.origin + np.asarray(idx, float) * self.voxel

    def clearance(self, p) -> float:
        """Trilinear-interpolated signed distance at world point p.

        Out-of-grid points are treated as UNSAFE (negative) instead of clamped to the
        nearest edge, so a pose that leaves the modeled SDF volume never reads as clear.
        """
        nx, ny, nz = self.sdf.shape
        f = self.w2g(p)
        if not (
            0.0 <= f[0] <= nx - 1.001 and 0.0 <= f[1] <= ny - 1.001 and 0.0 <= f[2] <= nz - 1.001
        ):
            return -1.0
        x = min(max(f[0], 0.0), nx - 1.001)
        y = min(max(f[1], 0.0), ny - 1.001)
        z = min(max(f[2], 0.0), nz - 1.001)
        i, j, k = int(x), int(y), int(z)
        dx, dy, dz = x - i, y - j, z - k
        s = self.sdf
        c00 = s[i, j, k] * (1 - dx) + s[i + 1, j, k] * dx
        c01 = s[i, j, k + 1] * (1 - dx) + s[i + 1, j, k + 1] * dx
        c10 = s[i, j + 1, k] * (1 - dx) + s[i + 1, j + 1, k] * dx
        c11 = s[i, j + 1, k + 1] * (1 - dx) + s[i + 1, j + 1, k + 1] * dx
        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy
        return float(c0 * (1 - dz) + c1 * dz)

    def gradient(self, p):
        """Unit vector toward increasing clearance (the 'most open' direction)."""
        eps = self.voxel
        g = np.zeros(3)
        for ax in range(3):
            d = np.zeros(3)
            d[ax] = eps
            g[ax] = (self.clearance(np.asarray(p) + d) - self.clearance(np.asarray(p) - d)) / (
                2 * eps
            )
        n = np.linalg.norm(g)
        return g / n if n > 1e-9 else g


def _is_free(
    sdf: np.ndarray,
    shape: tuple[int, int, int],
    index: tuple[int, int, int],
    standoff_min: float,
) -> bool:
    i, j, k = index
    nx, ny, nz = shape
    return 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and sdf[i, j, k] > standoff_min


def _risk(sdf: np.ndarray, index: tuple[int, int, int], pref_clear: float) -> float:
    return max(0.0, (pref_clear - sdf[index]) / pref_clear)


def _astar_predecessors(
    grid: SDFGrid,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    *,
    standoff_min: float,
    pref_clear: float,
    risk_w: float,
) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    sdf = grid.sdf
    shape = sdf.shape

    def heuristic(index: tuple[int, int, int]) -> float:
        return math.dist(index, goal) * grid.voxel

    open_heap = [(heuristic(start), 0.0, start)]
    predecessors: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    costs = {start: 0.0}
    closed: set[tuple[int, int, int]] = set()
    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current == goal:
            break
        if current in closed:
            continue
        closed.add(current)
        for offset in _NB:
            neighbor = tuple(current[axis] + offset[axis] for axis in range(3))
            if neighbor in closed or not _is_free(sdf, shape, neighbor, standoff_min):
                continue
            step = math.dist((0, 0, 0), offset) * grid.voxel
            next_cost = cost + step * (1.0 + risk_w * _risk(sdf, neighbor, pref_clear))
            if next_cost < costs.get(neighbor, 1e18):
                costs[neighbor] = next_cost
                predecessors[neighbor] = current
                heapq.heappush(
                    open_heap,
                    (next_cost + heuristic(neighbor), next_cost, neighbor),
                )
    return predecessors


def _reconstruct_indices(
    predecessors: dict[tuple[int, int, int], tuple[int, int, int]],
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
) -> list[tuple[int, int, int]] | None:
    if goal != start and goal not in predecessors:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(predecessors[path[-1]])
    path.reverse()
    return path


def plan(
    grid: SDFGrid,
    start,
    goal,
    standoff_min: float = 0.0,
    pref_clear: float | None = None,
    risk_w: float = 3.0,
):
    """Clearance-preferring A* (world coords -> list of world waypoints).

    Voxels with clearance <= standoff_min are BLOCKED. Step cost is scaled by a
    risk term that grows as clearance drops toward standoff_min, so the path
    hugs high-clearance corridors ('safest'), not just shortest.
    Returns None if unreachable.
    """
    pref_clear = pref_clear if pref_clear is not None else 5 * grid.voxel
    si = tuple(int(round(v)) for v in grid.w2g(start))
    gi = tuple(int(round(v)) for v in grid.w2g(goal))
    if not _is_free(grid.sdf, grid.sdf.shape, si, standoff_min):
        raise ValueError(f"start not free (clearance={grid.clearance(start):.3f})")
    if not _is_free(grid.sdf, grid.sdf.shape, gi, standoff_min):
        raise ValueError(f"goal not free (clearance={grid.clearance(goal):.3f})")
    predecessors = _astar_predecessors(
        grid,
        si,
        gi,
        standoff_min=standoff_min,
        pref_clear=pref_clear,
        risk_w=risk_w,
    )
    path = _reconstruct_indices(predecessors, si, gi)
    if path is None:
        return None
    return [grid.g2w(index) for index in path]


def plan_tour(grid: SDFGrid, start, targets, **kw):
    """Greedy nearest-neighbor visit order; A* between consecutive targets.
    Returns (full_path_world, visit_order). Targets are pole STANDOFF points
    (already offset from the pole by the inspection distance), in map coords."""
    remaining = list(targets)
    cur = tuple(start)
    full = [tuple(start)]
    order = []
    while remaining:
        nxt = min(remaining, key=lambda t: math.dist(cur, t))
        remaining.remove(nxt)
        seg = plan(grid, cur, nxt, **kw)
        if seg is None:
            print(f"[plan_tour] WARN target {nxt} unreachable, skipped")
            continue
        seg = smooth(
            grid, seg, standoff_min=kw.get("standoff_min", 0.0)
        )  # per-leg: keeps target endpoints
        full += seg[1:]
        order.append(nxt)
        cur = nxt
    return full, order


def smooth(grid: SDFGrid, path, standoff_min: float = 0.0, step: float | None = None):
    """Shortcut smoothing: keep a waypoint only if the straight skip-segment
    from the last kept point would leave free space."""
    if not path or len(path) < 3:
        return path
    step = step or grid.voxel
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _seg_clear(grid, path[i], path[j], standoff_min, step):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _seg_clear(grid: SDFGrid, a, b, standoff_min, step) -> bool:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    n = max(1, int(L / step))
    for t in np.linspace(0.0, 1.0, n + 1):
        if grid.clearance(a + t * (b - a)) <= standoff_min:
            return False
    return True


# --------------------------------------------------------------------------
# Self-test on a SYNTHETIC sdf: safe box with a pole (no-go cylinder + buffer)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    vox = 0.5
    L, H = 30.0, 10.0
    nx, ny, nz = int(L / vox), int(L / vox), int(H / vox)
    origin = np.zeros(3)
    sdf = np.empty((nx, ny, nz), np.float32)
    for i in range(nx):
        x = i * vox
        for j in range(ny):
            y = j * vox
            d_pole = math.hypot(x - 15.0, y - 15.0) - 3.0  # pole r+buffer=3
            for k in range(nz):
                z = k * vox
                d_box = min(x, L - x, y, L - y, z, H - z)  # inside box (+)
                sdf[i, j, k] = min(d_box, d_pole)
    grid = SDFGrid(sdf, origin, vox)

    start, goal = (3, 3, 5), (27, 27, 5)
    straight_ok = _seg_clear(grid, start, goal, 0.0, vox)
    path = plan(grid, start, goal, standoff_min=0.0, pref_clear=2.0, risk_w=3.0)
    sm = smooth(grid, path)
    mn = min(grid.clearance(p) for p in path)
    print(f"straight start->goal clear? {straight_ok}  (False => path must detour around pole)")
    print(
        f"A* path pts={len(path)}  smoothed={len(sm)}  min clearance along path={mn:.2f} (>0 = never enters no-go)"
    )
    g = grid.gradient((15 - 3 + 0.0, 15, 5))  # near pole edge -> should point away from pole
    print(f"gradient near pole edge (most-open dir) = [{g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}]")
