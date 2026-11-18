"""Immutable map raster jobs; no Tk objects or callbacks cross into the worker."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import threading
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw


def transform_map_xyz(xyz: np.ndarray, center: np.ndarray, rotation: tuple) -> np.ndarray:
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3) - center[None, :]
    yaw, pitch, roll = rotation
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    x1 = cy * pts[:, 0] + sy * pts[:, 2]
    z1 = -sy * pts[:, 0] + cy * pts[:, 2]
    y1 = pts[:, 1]
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1
    x3 = cr * x1 - sr * y2
    y3 = sr * x1 + cr * y2
    return np.stack([x3, y3, z2], axis=1)


def readonly_copy(value: Any, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MapRasterSnapshot:
    width: int
    height: int
    points: np.ndarray
    workmap_xyz: np.ndarray
    workmap_rgb: np.ndarray
    center: np.ndarray
    pivot: np.ndarray
    pan: np.ndarray
    zoom: float
    radius: float
    rotation: tuple[float, float, float]
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    measured_axes: bool
    font: Any
    workmap_color: tuple[int, int, int]


def render_map_raster(view: MapRasterSnapshot) -> Image.Image:
    """Render one complete view, preserving depth ties and the native map units."""
    width, height = view.width, view.height
    arr = np.empty((max(1, height), max(1, width), 3), dtype=np.uint8)
    arr[:] = (0x15, 0x18, 0x1C)
    scale = min(width, height) * 0.46 * view.zoom / view.radius

    def project(xyz):
        transformed = transform_map_xyz(xyz, view.center, view.rotation)
        sx = (width * 0.5 + view.pan[0] + transformed[:, 0] * scale).astype(int)
        sy = (height * 0.5 + view.pan[1] - transformed[:, 1] * scale).astype(int)
        return transformed, sx, sy

    if len(view.points):
        transformed, sx, sy = project(view.points[:, :3])
        inside = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        order = np.argsort(transformed[inside, 2])
        rgb = np.clip(view.points[:, 3:6][inside][order], 0, 255).astype(np.uint8)
        arr[sy[inside][order], sx[inside][order]] = rgb
    if len(view.workmap_xyz):
        _, sx, sy = project(view.workmap_xyz)
        for dx in (0, 1):
            for dy in (0, 1):
                px, py = sx + dx, sy + dy
                inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                arr[py[inside], px[inside]] = view.workmap_rgb[inside]
    image = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(image)
    if len(view.workmap_xyz):
        draw.text(
            (10, 8),
            f"工作圖新點 {len(view.workmap_xyz)}（原始 RGB，較大的點；僅顯示，不參與定位授權）",
            fill=view.workmap_color, font=view.font,
        )

    def pixel(xyz):
        # Scalar conversion truncates, exactly as OperatorApp.project_world did.
        transformed = transform_map_xyz(np.asarray(xyz).reshape(1, 3), view.center, view.rotation)[0]
        return (
            int(width * 0.5 + view.pan[0] + transformed[0] * scale),
            int(height * 0.5 + view.pan[1] - transformed[1] * scale),
        )

    ox, oy = pixel(view.center)
    axis_len = view.radius * 0.18
    labels = ("東", "上", "北") if view.measured_axes else ("X", "UP*", "Z")
    east, north, up = view.axes
    for direction, color, label in (
        (east, "#ff6b5f", labels[0]), (up, "#3fbf7f", labels[1]), (north, "#5aa7e8", labels[2]),
    ):
        ex, ey = pixel(view.center + direction * axis_len)
        draw.line((ox, oy, ex, ey), fill=color, width=2)
        draw.text((ex + 4, ey + 4), label, fill=color, font=view.font)
    px, py = pixel(view.pivot)
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline="#e6b94f", width=2)
    draw.line((px - 12, py, px + 12, py), fill="#e6b94f", width=1)
    draw.line((px, py - 12, px, py + 12), fill="#e6b94f", width=1)
    return image


class MapRasterWorker:
    """At most one running job and one latest pending snapshot, never a queue."""

    def __init__(self, render: Callable = render_map_raster) -> None:
        self._render = render
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-raster")
        self._lock = threading.Lock()
        self._future: Future | None = None
        self._running_key: tuple | None = None
        self._running_epoch = 0
        self._epoch = 0
        self._pending: tuple | None = None
        self._wanted: tuple | None = None
        self._completed: tuple | None = None
        self._closed = False
        self._version = 0
        self._latest: Image.Image | None = None

    def _poll_locked(self) -> None:
        if self._future is not None and self._future.done():
            try:
                image, error = self._future.result(), None
            except Exception as exc:
                image, error = None, exc
            if error is None and self._running_epoch == self._epoch and not self._closed:
                self._latest = image
            if (not self._closed and self._running_key == self._wanted
                    and self._running_epoch == self._epoch):
                self._completed = (self._running_key, image, error)
                self._version += 1
            self._future = None
        if not self._closed and self._future is None and self._pending is not None:
            self._running_key, snapshot = self._pending
            self._running_epoch = self._epoch
            self._pending = None
            self._future = self._executor.submit(self._render, snapshot)

    def poll(self) -> int:
        with self._lock:
            self._poll_locked()
            return self._version

    def request(self, key: tuple, snapshot: Callable[[], MapRasterSnapshot]) -> None:
        with self._lock:
            self._poll_locked()
            if self._closed or key == self._wanted:
                return
            self._wanted = key
            self._completed = None
            self._pending = (key, snapshot())
            self._poll_locked()

    def result(self, key: tuple) -> Image.Image | None:
        with self._lock:
            self._poll_locked()
            if self._completed is None or self._completed[0] != key:
                return None
            _, image, error = self._completed
            if error is not None:
                raise RuntimeError("map raster failed") from error
            return image

    def latest(self, width: int, height: int) -> Image.Image | None:
        with self._lock:
            image = self._latest
            return image if image is not None and image.size == (width, height) else None

    def clear(self) -> None:
        with self._lock:
            self._wanted = self._pending = self._completed = self._latest = None
            self._epoch += 1
            self._version += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._wanted = self._pending = self._completed = self._latest = None
            if self._future is not None:
                self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
