"""Toolkit-independent camera and waypoint operations for route authoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def pointer_is_drag(
    start: tuple[int, int], current: tuple[int, int], threshold_px: int = 5
) -> bool:
    dx = int(current[0]) - int(start[0])
    dy = int(current[1]) - int(start[1])
    return dx * dx + dy * dy > int(threshold_px) ** 2


@dataclass
class OrthoView:
    center: np.ndarray
    radius: float
    yaw: float = 0.0
    pitch: float = 0.0
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    def rotation(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        yaw = np.array(((cy, sy, 0.0), (-sy, cy, 0.0), (0.0, 0.0, 1.0)))
        pitch = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)))
        return pitch @ yaw

    def transform(self, points) -> np.ndarray:
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        return (values - self.center[None, :]) @ self.rotation().T

    def scale(self, width: int, height: int) -> float:
        return min(width, height) * 0.46 * self.zoom / max(self.radius, 1e-9)

    def project(self, points, width: int, height: int) -> tuple[np.ndarray, ...]:
        view = self.transform(points)
        scale = self.scale(width, height)
        sx = width * 0.5 + self.pan_x + view[:, 0] * scale
        sy = height * 0.5 + self.pan_y - view[:, 1] * scale
        return sx, sy, view[:, 2]

    def screen_delta_to_world(
        self,
        dx: float,
        dy: float,
        width: int,
        height: int,
        axis: int | None,
    ) -> np.ndarray:
        scale = self.scale(width, height)
        if axis is None:
            view_delta = np.array((dx / scale, -dy / scale, 0.0))
            return self.rotation().T @ view_delta
        world_axis = np.zeros(3, dtype=float)
        world_axis[axis] = 1.0
        view_axis = self.rotation() @ world_axis
        screen_axis = np.array((view_axis[0] * scale, -view_axis[1] * scale))
        denominator = float(screen_axis @ screen_axis)
        if denominator < 1e-9:
            amount = -dy / scale
        else:
            amount = float(np.array((dx, dy)) @ screen_axis) / denominator
        return world_axis * amount

    def top(self) -> None:
        self.yaw = 0.0
        self.pitch = 0.0

    def front(self) -> None:
        self.yaw = 0.0
        self.pitch = -math.pi / 2.0

    def right(self) -> None:
        self.yaw = math.pi / 2.0
        self.pitch = -math.pi / 2.0


class RouteEditorController:
    """State changes for the two-stage placement/position-edit workflow."""

    def __init__(self, points=None):
        self.points = [list(map(float, point)) for point in (points or [])]
        self.phase = "height" if self.points else "layout"
        self.placement_enabled = False
        self.selected: int | None = 0 if self.points else None
        self._undo: list[list[list[float]]] = []
        self._redo: list[list[list[float]]] = []
        self._move_start: list[float] | None = None
        self._move_snapshot: list[list[float]] | None = None
        self.move_axis: int | None = None

    @property
    def moving(self) -> bool:
        return self._move_start is not None

    def _snapshot(self) -> list[list[float]]:
        return [point.copy() for point in self.points]

    def _record(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()

    def set_placement_enabled(self, enabled: bool) -> bool:
        self.placement_enabled = bool(enabled) and self.phase == "layout"
        return self.placement_enabled

    def add(self, point) -> int:
        if self.phase != "layout":
            raise ValueError("只能在第一階段新增路徑點")
        if not self.placement_enabled:
            raise ValueError("請先開啟「標路徑點模式」")
        self._record()
        self.points.append([float(value) for value in point])
        self.selected = len(self.points) - 1
        return self.selected

    def delete_selected(self) -> bool:
        if self.selected is None or not (0 <= self.selected < len(self.points)):
            return False
        self._record()
        del self.points[self.selected]
        self.selected = (
            min(self.selected, len(self.points) - 1) if self.points else None
        )
        return True

    def spheres_overlap(self, radius: float) -> bool:
        """True when arrival spheres of `radius` would touch on the shortest leg.

        Overlapping spheres mean the drone retires waypoints without ever
        translating between them: settle, advance, settle, advance.
        """
        shortest = self.shortest_leg()
        return shortest is not None and float(radius) >= shortest * 0.5

    def shortest_leg(self) -> float | None:
        """Shortest distance between consecutive waypoints, or None below two.

        An arrival radius at or above half of this makes neighbouring spheres
        overlap, so the drone retires waypoints without translating between them.
        """
        points = self.points
        if len(points) < 2:
            return None
        import math as _math
        return min(
            _math.dist(a, b) for a, b in zip(points, points[1:])
        )

    def finish_layout(self) -> None:
        if len(self.points) < 2:
            raise ValueError("至少標注兩個路徑點才能進入第二階段")
        self.phase = "height"
        self.placement_enabled = False
        self.selected = 0

    def begin_move(self) -> bool:
        if self.phase != "height":
            return False
        if self.selected is None or not (0 <= self.selected < len(self.points)):
            return False
        self._move_start = self.points[self.selected].copy()
        self._move_snapshot = self._snapshot()
        self.move_axis = None
        return True

    def constrain(self, axis: int) -> None:
        if self.moving:
            self.move_axis = int(axis)

    def preview_move(self, delta) -> None:
        if not self.moving or self.selected is None or self._move_start is None:
            return
        self.points[self.selected] = (
            np.asarray(self._move_start) + np.asarray(delta, dtype=float)
        ).tolist()

    def confirm_move(self) -> bool:
        if not self.moving or self._move_snapshot is None:
            return False
        changed = self.points != self._move_snapshot
        if changed:
            self._undo.append(self._move_snapshot)
            self._redo.clear()
        self._move_start = None
        self._move_snapshot = None
        self.move_axis = None
        return changed

    def cancel_move(self) -> bool:
        if not self.moving or self._move_snapshot is None:
            return False
        self.points = self._move_snapshot
        self._move_start = None
        self._move_snapshot = None
        self.move_axis = None
        return True

    def undo(self) -> bool:
        if self.moving:
            self.cancel_move()
            return True
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.points = self._undo.pop()
        self.selected = (
            min(self.selected or 0, len(self.points) - 1) if self.points else None
        )
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.points = self._redo.pop()
        self.selected = (
            min(self.selected or 0, len(self.points) - 1) if self.points else None
        )
        return True
