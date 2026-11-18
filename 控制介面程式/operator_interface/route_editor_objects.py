"""Toolkit-independent, editor-Z-up axis-aligned object box operations."""

from __future__ import annotations

import itertools
import math

import numpy as np


Box = dict[str, list[float]]
_ObjectSnapshot = tuple[list[Box], list[int]]


def _vector(values, name: str) -> list[float]:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite numbers") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return vector.tolist()


def _box(value) -> Box:
    try:
        center = _vector(value["center"], "center")
        size = _vector(value["size"], "size")
    except (KeyError, TypeError) as exc:
        raise ValueError("box must contain center and size") from exc
    if any(dimension <= 0.0 for dimension in size):
        raise ValueError("box dimensions must be strictly positive")
    return {"center": center, "size": size}


def _copy_boxes(boxes: list[Box]) -> list[Box]:
    return [{"center": box["center"].copy(), "size": box["size"].copy()} for box in boxes]


def box_corners(box: Box) -> np.ndarray:
    """Return eight corners in itertools.product((-1, 1), repeat=3) order."""
    value = _box(box)
    return np.asarray([
        [center + sign * (size * 0.5)
         for center, size, sign in zip(value["center"], value["size"], signs)]
        for signs in itertools.product((-1, 1), repeat=3)
    ], dtype=float)


#: Screen pixels of pointer travel that multiply a box dimension by e (about 2.7x).
SCALE_PIXELS_PER_E = 250.0
#: Holding Shift divides pointer sensitivity by this factor.
SCALE_PRECISE_DIVISOR = 10.0
#: Start closer to the box centre than this has no outward direction to follow.
_SCALE_MIN_ANCHOR_PX = 8.0
_SCALE_LOG_LIMIT = 12.0
#: Fallback travel axis (screen right/up grows, left/down shrinks).
_SCALE_FALLBACK_DIRECTION = np.array((1.0, -1.0)) / math.sqrt(2.0)


class PointerScale:
    """Pointer-to-scale mapping that is exponential, unbounded both ways and jump-free.

    The factor is exp(t / SCALE_PIXELS_PER_E) where t is pointer travel projected on
    one fixed screen axis, so the first pixels change the size by about 0.4% instead
    of a share of a tiny start-to-centre distance, and the box can shrink as far as
    it can grow. Travel is accumulated per event, so toggling precision never makes
    the box jump, and the axis is frozen at construction so zooming the map
    mid-gesture cannot change it.
    """

    def __init__(self, center, start):
        self._last = np.asarray(start, dtype=float)
        offset = self._last - np.asarray(center, dtype=float)
        distance = float(np.linalg.norm(offset))
        # Away from the box centre grows, towards it shrinks; started on the centre
        # there is no such direction, so right/up grows.
        self._direction = (
            offset / distance if distance >= _SCALE_MIN_ANCHOR_PX
            else _SCALE_FALLBACK_DIRECTION
        )
        self._log = 0.0

    def move(self, point, precise: bool = False) -> float:
        point = np.asarray(point, dtype=float)
        travel = float((point - self._last) @ self._direction)
        self._last = point
        if precise:
            travel /= SCALE_PRECISE_DIVISOR
        self._log = float(np.clip(
            self._log + travel / SCALE_PIXELS_PER_E, -_SCALE_LOG_LIMIT, _SCALE_LOG_LIMIT))
        return math.exp(self._log)


class ObjectBoxController:
    """Object-only undo history; previews always derive from their start snapshot."""

    def __init__(self, boxes=None):
        self.boxes: list[Box] = [_box(box) for box in ([] if boxes is None else boxes)]
        self.keys: list[int] = list(range(len(self.boxes)))
        self._clipboard: Box | None = None
        # Deliberately excluded from history: undone identities must never be reused.
        self._next_key = len(self.boxes)
        self.selected: int | None = 0 if self.boxes else None
        self.operation: str | None = None
        self.axis: int | None = None
        self._undo: list[_ObjectSnapshot] = []
        self._redo: list[_ObjectSnapshot] = []
        self._transform_snapshot: _ObjectSnapshot | None = None
        self._transform_index: int | None = None

    @property
    def transforming(self) -> bool:
        return self._transform_snapshot is not None

    def _snapshot(self) -> _ObjectSnapshot:
        return _copy_boxes(self.boxes), self.keys.copy()

    def _record(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()

    def _has_selection(self) -> bool:
        return self.selected is not None and 0 <= self.selected < len(self.boxes)

    def _restore_selection(self) -> None:
        self.selected = min(self.selected or 0, len(self.boxes) - 1) if self.boxes else None

    def _finish(self) -> None:
        self._transform_snapshot = None
        self._transform_index = None
        self.operation = None
        self.axis = None

    def add_bounds(self, lo, hi) -> int:
        lower = _vector(lo, "lower bounds")
        upper = _vector(hi, "upper bounds")
        value = _box({
            "center": [low * 0.5 + high * 0.5 for low, high in zip(lower, upper)],
            "size": [high - low for low, high in zip(lower, upper)],
        })
        if self.transforming:
            raise ValueError("finish the current transform before adding a box")
        self._record()
        self.boxes.append(value)
        self.keys.append(self._next_key)
        self._next_key += 1
        self.selected = len(self.boxes) - 1
        return self.selected

    def begin(self, operation: str) -> bool:
        if self.transforming:
            return False
        if operation not in ("g", "s"):
            raise ValueError("operation must be 'g' or 's'")
        if not self._has_selection():
            return False
        self._transform_snapshot = self._snapshot()
        self._transform_index = self.selected
        self.operation = operation
        self.axis = None
        return True

    def constrain(self, axis: int | None) -> None:
        if axis is not None and (
            not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2)
        ):
            raise ValueError("axis must be 0, 1, 2 or None")
        if self.transforming:
            self.axis = None if self.axis == axis else axis

    def preview(self, delta=None, factor: float | None = None) -> None:
        if self._transform_snapshot is None or self._transform_index is None:
            return
        original = self._transform_snapshot[0][self._transform_index]
        if self.operation == "g":
            if delta is None:
                return
            displacement = _vector(delta, "displacement")
            center = [
                coordinate + (displacement[index] if self.axis in (None, index) else 0.0)
                for index, coordinate in enumerate(original["center"])
            ]
            value = _box({"center": center, "size": original["size"]})
        else:
            if factor is None:
                return
            try:
                scale = float(factor)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("scale factor must be finite and strictly positive") from exc
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("scale factor must be finite and strictly positive")
            size = [
                dimension * (scale if self.axis in (None, index) else 1.0)
                for index, dimension in enumerate(original["size"])
            ]
            value = _box({"center": original["center"], "size": size})
        self.boxes[self._transform_index] = value

    def confirm(self) -> bool:
        if self._transform_snapshot is None:
            return False
        changed = self.boxes != self._transform_snapshot[0]
        if changed:
            self._undo.append(self._transform_snapshot)
            self._redo.clear()
        self._finish()
        return changed

    def cancel(self) -> bool:
        if self._transform_snapshot is None:
            return False
        self.boxes, self.keys = self._transform_snapshot
        self._finish()
        return True

    def delete_selected(self) -> bool:
        if self.transforming or not self._has_selection():
            return False
        self._record()
        del self.boxes[self.selected]
        del self.keys[self.selected]
        self._restore_selection()
        return True

    def undo(self) -> bool:
        if self.transforming:
            return self.cancel()
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.boxes, self.keys = self._undo.pop()
        self._restore_selection()
        return True

    def redo(self) -> bool:
        if self.transforming or not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.boxes, self.keys = self._redo.pop()
        self._restore_selection()
        return True

    def copy_selected(self) -> bool:
        """Remember the selected box; the clipboard is outside undo history."""
        if self.transforming or not self._has_selection():
            return False
        self._clipboard = _copy_boxes([self.boxes[self.selected]])[0]
        return True

    def paste(self) -> bool:
        """Add an identical box with a fresh identity and select it."""
        if self.transforming or self._clipboard is None:
            return False
        self._record()
        self.boxes.append(_copy_boxes([self._clipboard])[0])
        self.keys.append(self._next_key)
        self._next_key += 1
        self.selected = len(self.boxes) - 1
        return True
