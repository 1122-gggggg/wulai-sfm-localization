"""Offline wire annotations in the route editor's aligned coordinate basis."""
from __future__ import annotations

import copy
import math
from colorsys import hsv_to_rgb

import numpy as np


def wire_color(index: int) -> str:
    """Shared editor/operator colour by span order; no short repeating palette."""
    channels = hsv_to_rgb((0.075 + index * 0.618033988749895) % 1.0, 0.65, 1.0)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in channels)


def _point(raw, label: str) -> list[float]:
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != 3:
        raise ValueError(f"{label} 必須包含三個有限座標")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) for v in raw):
        raise ValueError(f"{label} 必須包含三個有限座標")
    try:
        values = [float(v) for v in raw]
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} 必須包含三個有限座標") from exc
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{label} 必須包含三個有限座標")
    return values


def _sag(raw) -> float:
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, float, np.number)):
        raise ValueError("垂度必須是有限且不小於零的數值")
    try:
        value = float(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("垂度必須是有限且不小於零的數值") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("垂度必須是有限且不小於零的數值")
    return value


def wire_annotations(raw) -> list[dict]:
    """Validate and detach annotations; absent wires are represented by []."""
    if not isinstance(raw, list):
        raise ValueError("route wires 必須是清單")
    wires = []
    for wire in raw:
        if not isinstance(wire, dict) or set(wire) != {"a", "b", "sag", "up"}:
            raise ValueError("每條電線必須包含 a、b、sag、up")
        first, second = _point(wire["a"], "端點 A"), _point(wire["b"], "端點 B")
        if first == second or not math.isfinite(math.dist(first, second)):
            raise ValueError("電線兩端必須不同且距離有限")
        up = _point(wire["up"], "電線向上軸")
        if not math.isclose(math.hypot(*up), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("電線向上軸必須是單位向量")
        sag = _sag(wire["sag"])
        # Reject overflow before committing state or rendering with Pillow.
        middle = np.asarray(first) * 0.5 + np.asarray(second) * 0.5 - np.asarray(up) * sag
        if not np.isfinite(middle).all():
            raise ValueError("垂度超出可表示範圍")
        wires.append({"a": first, "b": second, "sag": sag, "up": up})
    return wires


def sample_wire(wire: dict, samples: int = 65) -> np.ndarray:
    """Parabolic visual annotation, midpoint sag below the chord, not a cable simulation."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
        raise ValueError("電線取樣數至少為三")
    t = np.linspace(0.0, 1.0, samples)[:, None]
    first, second = np.asarray(wire["a"]), np.asarray(wire["b"])
    return (1.0 - t) * first + t * second - (4.0 * t * (1.0 - t) * wire["sag"]) * np.asarray(wire["up"])


class WireController:
    """Independent wire history; preview changes are committed as one undo step."""

    def __init__(self, wires=None):
        self.wires = wire_annotations([] if wires is None else wires)
        self.selected: int | None = None
        self.selected_endpoint: str | None = None
        self.pending: list[float] | None = None
        self._start: dict | None = None
        self._operation: str | None = None
        self.move_axis: int | None = None
        self._undo: list[list[dict]] = []
        self._redo: list[list[dict]] = []

    @property
    def adjusting(self) -> bool:
        return self._operation == "s"

    @property
    def moving(self) -> bool:
        return self._operation == "g"

    @property
    def editing(self) -> bool:
        return self._start is not None

    def _record(self) -> None:
        self._undo.append(copy.deepcopy(self.wires))
        self._redo.clear()

    def pick_endpoint(self, point) -> bool:
        if self.editing:
            raise ValueError("請先確認或取消電線編輯")
        point = _point(point, "電線端點")
        if self.pending is None:
            self.pending = point
            return False
        wire = wire_annotations([{"a": self.pending, "b": point, "sag": 0.0, "up": [0.0, 0.0, 1.0]}])[0]
        self._record()
        self.wires.append(wire)
        self.selected = len(self.wires) - 1
        self.selected_endpoint = "b"
        self.pending = None
        return True

    def begin_sag(self) -> bool:
        if self.editing or self.pending is not None or self.selected is None:
            return False
        self._start = copy.deepcopy(self.wires[self.selected])
        self._operation = "s"
        return True

    def preview_sag(self, value) -> None:
        if not self.adjusting:
            raise ValueError("請先按 S 開始調整垂度")
        candidate = dict(self.wires[self.selected], sag=_sag(value))
        self.wires[self.selected] = wire_annotations([candidate])[0]

    def begin_move(self) -> bool:
        if (
            self.editing or self.pending is not None or self.selected is None
            or self.selected_endpoint not in {"a", "b"}
        ):
            return False
        self._start = copy.deepcopy(self.wires[self.selected])
        self._operation = "g"
        self.move_axis = None
        return True

    def constrain(self, axis: int) -> None:
        if axis not in (0, 1, 2):
            raise ValueError("移動軸必須是 X、Y 或 Z")
        if self.moving:
            self.move_axis = None if self.move_axis == axis else axis

    def preview_move(self, delta) -> None:
        if not self.moving:
            raise ValueError("請先按 G 開始移動端點")
        offset = _point(delta, "端點位移")
        if self.move_axis is not None:
            offset = [value if index == self.move_axis else 0.0 for index, value in enumerate(offset)]
        origin = self._start[self.selected_endpoint]
        point = [value + change for value, change in zip(origin, offset)]
        candidate = dict(self._start, **{self.selected_endpoint: point})
        self.wires[self.selected] = wire_annotations([candidate])[0]

    def confirm(self) -> bool:
        if not self.editing:
            return False
        value = self.wires[self.selected]
        self.wires[self.selected] = self._start
        changed = value != self._start
        if changed:
            self._record()
        self.wires[self.selected] = value
        self._start = None
        self._operation = None
        self.move_axis = None
        return changed

    def cancel(self) -> bool:
        if not self.editing:
            return False
        self.wires[self.selected] = self._start
        self._start = None
        self._operation = None
        self.move_axis = None
        return True

    def delete_selected(self) -> bool:
        if self.editing or self.pending is not None or self.selected is None:
            return False
        self._record()
        del self.wires[self.selected]
        self.selected = min(self.selected, len(self.wires) - 1) if self.wires else None
        self.selected_endpoint = None
        return True

    def undo(self) -> bool:
        self.pending = None
        self.cancel()
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self.wires))
        self.wires = self._undo.pop()
        self.selected = len(self.wires) - 1 if self.wires else None
        self.selected_endpoint = None
        return True

    def redo(self) -> bool:
        self.pending = None
        self.cancel()
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self.wires))
        self.wires = self._redo.pop()
        self.selected = len(self.wires) - 1 if self.wires else None
        self.selected_endpoint = None
        return True
