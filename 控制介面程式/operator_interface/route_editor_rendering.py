"""Full-cloud orthographic rasterization, independent of Tk and flight controls."""
from __future__ import annotations

import numpy as np

from route_editor_controller import OrthoView

BACKGROUND = (18, 21, 25)


class PointCloudRenderer:
    """Retain every point; resolve visibility per pixel instead of sorting the cloud.

    Owned by a single rendering worker. CUDA keeps geometry/colors resident;
    NumPy uses the same depth and deterministic equal-depth tie policy on CPU.
    """

    def __init__(self, points: np.ndarray, *, use_cuda: bool = True):
        self.xyz = np.ascontiguousarray(points[:, :3], dtype=np.float32)
        self.rgb = np.ascontiguousarray(np.clip(points[:, 3:6], 0, 255), dtype=np.uint8)
        self.point_ids = np.arange(len(points), dtype=np.int64)
        self.device = "cpu"
        self._gpu = None
        if use_cuda and len(points):
            try:
                import torch

                if torch.cuda.is_available():
                    self._gpu = (
                        torch,
                        torch.from_numpy(self.xyz).to("cuda"),
                        torch.from_numpy(self.rgb).to("cuda"),
                        torch.from_numpy(self.point_ids).to("cuda"),
                    )
                    self.device = "cuda"
            except (ImportError, RuntimeError) as exc:
                print(f"[route-editor] CUDA unavailable, using full-cloud CPU: {exc}", flush=True)
        self._gpu_size = None
        self._gpu_depth = None
        self._gpu_winners = None

    def render(self, view: OrthoView, width: int, height: int) -> np.ndarray:
        if self._gpu is not None:
            try:
                return self._render_cuda(view, width, height)
            except RuntimeError as exc:
                # GPU pressure must not force point decimation or strand the UI.
                print(f"[route-editor] CUDA rendering failed, using full-cloud CPU: {exc}", flush=True)
                self._gpu = None
                self._gpu_depth = self._gpu_winners = None
                self.device = "cpu"
        return self._render_cpu(view, width, height)

    def _render_cpu(self, view: OrthoView, width: int, height: int) -> np.ndarray:
        image = np.empty((height * width, 3), dtype=np.uint8)
        image[:] = BACKGROUND
        if not len(self.xyz):
            return image.reshape(height, width, 3)
        rotation = view.rotation().astype(np.float32)
        projected = self.xyz @ rotation.T - np.asarray(view.center, dtype=np.float32) @ rotation.T
        scale = view.scale(width, height)
        sx = width * 0.5 + view.pan_x + projected[:, 0] * scale
        sy = height * 0.5 + view.pan_y - projected[:, 1] * scale
        inside = (
            np.isfinite(projected).all(axis=1)
            & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        )
        pixels = sy[inside].astype(np.int64) * width + sx[inside].astype(np.int64)
        depth = projected[:, 2][inside]
        ids = self.point_ids[inside]
        zbuffer = np.full(width * height, -np.inf, dtype=np.float32)
        np.maximum.at(zbuffer, pixels, depth)
        front = depth == zbuffer[pixels]
        winners = np.full(width * height, -1, dtype=np.int64)
        np.maximum.at(winners, pixels[front], ids[front])
        visible = winners >= 0
        image[visible] = self.rgb[winners[visible]]
        return image.reshape(height, width, 3)

    def _render_cuda(self, view: OrthoView, width: int, height: int) -> np.ndarray:
        torch, xyz, rgb, ids = self._gpu
        with torch.inference_mode():
            rotation = torch.as_tensor(view.rotation(), dtype=torch.float32, device="cuda")
            center = torch.as_tensor(view.center, dtype=torch.float32, device="cuda")
            projected = xyz @ rotation.T - center @ rotation.T
            scale = view.scale(width, height)
            sx = width * 0.5 + view.pan_x + projected[:, 0] * scale
            sy = height * 0.5 + view.pan_y - projected[:, 1] * scale
            inside = (
                torch.isfinite(projected).all(dim=1)
                & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
            )
            pixels = sy[inside].to(torch.int64) * width + sx[inside].to(torch.int64)
            depth = projected[:, 2][inside]
            point_ids = ids[inside]
            if self._gpu_size != (width, height):
                self._gpu_depth = torch.empty(width * height, device="cuda")
                self._gpu_winners = torch.empty(width * height, dtype=torch.int64, device="cuda")
                self._gpu_size = (width, height)
            zbuffer = self._gpu_depth.fill_(-torch.inf)
            zbuffer.scatter_reduce_(0, pixels, depth, reduce="amax", include_self=True)
            front = depth == zbuffer[pixels]
            winners = self._gpu_winners.fill_(-1)
            winners.scatter_reduce_(0, pixels[front], point_ids[front], reduce="amax", include_self=True)
            image = rgb[winners.clamp_min(0)]
            image[winners < 0] = torch.tensor(BACKGROUND, dtype=torch.uint8, device="cuda")
            return image.reshape(height, width, 3).cpu().numpy()
