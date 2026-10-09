"""Toolkit-independent camera and waypoint operations for route authoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


_RouteSnapshot = tuple[
    list[list[float]], list[list[dict]], list[int | None], list[list[dict]]
]


def _look_at_key(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("look-at object key must be a nonnegative integer or None")
    return int(value)


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

    def __init__(self, points=None, actions=None, segment_look_at=None, return_actions=None):
        self.points = [list(map(float, point)) for point in (points or [])]
        #: Per-waypoint action dicts, index-aligned with points so undo/redo
        #: can never shift an action onto a different waypoint.
        self.actions = (
            [[dict(action) for action in entry] for entry in actions]
            if actions is not None else [[] for _ in self.points]
        )
        if len(self.actions) != len(self.points):
            raise ValueError("航點動作數量與航點數量不一致")
        #: Same shape for the return pass: the actions run when the return trip
        #: reaches that waypoint (the first/last waypoints cannot carry any).
        self.return_actions = (
            [[dict(action) for action in entry] for entry in return_actions]
            if return_actions is not None else [[] for _ in self.points]
        )
        if len(self.return_actions) != len(self.points):
            raise ValueError("回程動作數量與航點數量不一致")
        self.segment_look_at: list[int | None] = (
            [_look_at_key(key) for key in segment_look_at]
            if segment_look_at is not None else [None] * max(0, len(self.points) - 1)
        )
        if len(self.segment_look_at) != max(0, len(self.points) - 1):
            raise ValueError("look-at segment count must match the waypoint legs")
        self.phase = "height" if self.points else "layout"
        self.placement_enabled = False
        self.selected: int | None = 0 if self.points else None
        self._undo: list[_RouteSnapshot] = []
        self._redo: list[_RouteSnapshot] = []
        self._move_start: list[float] | None = None
        self._move_snapshot: _RouteSnapshot | None = None
        self.move_axis: int | None = None

    @property
    def moving(self) -> bool:
        return self._move_start is not None

    def _snapshot(self) -> list[list[float]]:
        return [point.copy() for point in self.points]

    def _actions_snapshot(self) -> list[list[dict]]:
        return [[dict(action) for action in entry] for entry in self.actions]

    def _return_actions_snapshot(self) -> list[list[dict]]:
        return [[dict(action) for action in entry] for entry in self.return_actions]

    def _state_snapshot(self) -> _RouteSnapshot:
        return (
            self._snapshot(), self._actions_snapshot(), self.segment_look_at.copy(),
            self._return_actions_snapshot(),
        )

    def _record(self) -> None:
        self._undo.append(self._state_snapshot())
        self._redo.clear()

    def set_look_at_range(self, start: int, end: int, object_key: int | None) -> bool:
        """Set the legs from waypoint start through end (zero-based, end-exclusive)."""
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, np.integer))
            or not isinstance(end, (int, np.integer))
            or not 0 <= start < end < len(self.points)
        ):
            raise ValueError("look-at range must satisfy 0 <= start < end < waypoint count")
        key = _look_at_key(object_key)
        targets = [key] * (end - start)
        if self.segment_look_at[start:end] == targets:
            return False
        self._record()
        self.segment_look_at[start:end] = targets
        return True

    def discard_missing_look_at(self, valid_keys: Iterable[int]) -> bool:
        """Clear missing object links without adding an independent undo step."""
        keys = set(valid_keys)
        changed = False
        for index, key in enumerate(self.segment_look_at):
            if key is not None and key not in keys:
                self.segment_look_at[index] = None
                changed = True
        return changed

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
        self.actions.append([])
        self.return_actions.append([])
        if len(self.points) > 1:
            self.segment_look_at.append(None)
        self.selected = len(self.points) - 1
        return self.selected

    def delete_selected(self) -> bool:
        if self.selected is None or not (0 <= self.selected < len(self.points)):
            return False
        self._record()
        index = self.selected
        if len(self.points) > 1:
            if index == 0:
                self.segment_look_at.pop(0)
            elif index == len(self.points) - 1:
                self.segment_look_at.pop()
            else:
                incoming, outgoing = self.segment_look_at[index - 1:index + 1]
                self.segment_look_at[index - 1:index + 1] = [
                    incoming if incoming == outgoing else None
                ]
        del self.points[self.selected]
        del self.actions[self.selected]
        del self.return_actions[self.selected]
        self.selected = (
            min(self.selected, len(self.points) - 1) if self.points else None
        )
        return True

    def _actions_for(self, direction: str) -> list[list[dict]]:
        if direction == "out":
            return self.actions
        if direction == "back":
            return self.return_actions
        raise ValueError(f"unknown action direction: {direction!r}")

    def return_actions_allowed(self, index: int | None) -> bool:
        """The first waypoint is the landing point; the last is the turnaround, flown once."""
        return index is not None and 0 < index < len(self.points) - 1

    def add_action(self, action: dict, direction: str = "out") -> bool:
        """Append one already-validated action to the selected waypoint (out or back)."""
        table = self._actions_for(direction)
        if self.selected is None or not (0 <= self.selected < len(self.points)):
            return False
        if direction == "back" and not self.return_actions_allowed(self.selected):
            return False
        self._record()
        table[self.selected].append(dict(action))
        return True

    def remove_last_action(self, direction: str = "out") -> bool:
        table = self._actions_for(direction)
        if self.selected is None or not (0 <= self.selected < len(self.points)):
            return False
        if not table[self.selected]:
            return False
        self._record()
        table[self.selected].pop()
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
        self._move_snapshot = self._state_snapshot()
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
        changed = self.points != self._move_snapshot[0]
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
        self.points, self.actions, self.segment_look_at, self.return_actions = (
            self._move_snapshot
        )
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
        self._redo.append(self._state_snapshot())
        self.points, self.actions, self.segment_look_at, self.return_actions = self._undo.pop()
        self.selected = (
            min(self.selected or 0, len(self.points) - 1) if self.points else None
        )
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._state_snapshot())
        self.points, self.actions, self.segment_look_at, self.return_actions = self._redo.pop()
        self.selected = (
            min(self.selected or 0, len(self.points) - 1) if self.points else None
        )
        return True
