"""Digest-bound surface queries in raw COLMAP coordinates, with metric z-depth.

Open3D is imported only by ``load``; distorted cameras import pycolmap only
when queried. No geometry, pose, or ray origin is scaled into meters. The
caller must supply a reliable session scale or a calibrated manifest scale.
"""

from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping

import numpy as np

from direct_paths import verify_file_sha256

SURFACE_SCHEMA = "mvs-surface/v1"
SURFACE_MANIFEST = "dense_surface.json"
_MAX_RAYS = 65536
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "coordinate_frame_id",
        "source_bundle_sha256",
        "mesh",
        "mesh_sha256",
        "geometry_units",
        "meters_per_map_unit",
        "scale_source",
    }
)


def _positive_scale(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite meters-per-map-unit scale")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite meters-per-map-unit scale")
    return result


def _dimension(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _pose(center, axes_world):
    center = np.asarray(center, dtype=np.float64)
    axes = np.asarray(axes_world, dtype=np.float64)
    if (
        center.shape != (3,)
        or not np.all(np.isfinite(center))
        or np.any(np.abs(center) > np.finfo(np.float32).max)
    ):
        raise ValueError("center must be a finite raw-map XYZ vector representable in float32")
    if (
        axes.shape != (3, 3)
        or not np.all(np.isfinite(axes))
        or not np.allclose(axes @ axes.T, np.eye(3), rtol=0, atol=1e-5)
        or not np.isclose(np.linalg.det(axes), 1.0, rtol=0, atol=1e-5)
    ):
        raise ValueError("axes_world must be a proper rotation with right/down/forward rows")
    return center, axes


def _calibration(camera):
    if not isinstance(camera, Mapping):
        raise ValueError("camera must contain model, width, height and params")
    try:
        model = camera["model"]
        width = _dimension(camera["width"], "camera width")
        height = _dimension(camera["height"], "camera height")
        params = np.asarray(camera["params"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"camera is missing {exc.args[0]}") from exc
    if not isinstance(model, str) or not model:
        raise ValueError("camera model must be a non-empty COLMAP model name")
    if params.ndim != 1 or not np.all(np.isfinite(params)):
        raise ValueError("camera params must be a finite parameter vector")
    if model in ("PINHOLE", "SIMPLE_PINHOLE"):
        expected = 4 if model == "PINHOLE" else 3
        if len(params) != expected:
            raise ValueError(f"{model} requires {expected} camera params")
        if model == "SIMPLE_PINHOLE":
            params = params[[0, 0, 1, 2]]
        if np.any(params[:2] <= 0):
            raise ValueError("camera focal lengths must be positive")
        return width, height, params, None

    try:
        import pycolmap
    except ImportError as exc:
        raise ImportError("distorted surface-depth cameras require pycolmap") from exc
    try:
        colmap_camera = pycolmap.Camera(model=model, width=width, height=height, params=params)
        if not colmap_camera.verify_params():
            raise ValueError("parameter count does not match the camera model")
        focal_idxs = colmap_camera.focal_length_idxs()
        if not focal_idxs or np.any(params[focal_idxs] <= 0):
            raise ValueError("camera focal lengths must be positive")
        if not callable(getattr(colmap_camera, "cam_from_img", None)):
            raise ValueError("installed pycolmap lacks camera.cam_from_img")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid COLMAP camera calibration: {exc}") from exc
    return width, height, params, colmap_camera


def _unproject(calibration, pixels):
    _, _, params, camera = calibration
    if camera is None:
        return (pixels - params[2:]) / params[:2]
    try:
        xy = camera.cam_from_img(np.ascontiguousarray(pixels))
    except TypeError:
        # Older pycolmap versions expose only the single-point overload.
        xy = np.full((len(pixels), 2), np.nan, dtype=np.float64)
        for index, pixel in enumerate(pixels):
            point = camera.cam_from_img(pixel)
            if point is not None:
                xy[index] = point
    if xy is None:
        return np.full((len(pixels), 2), np.nan, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (len(pixels), 2):
        raise ValueError("camera.cam_from_img returned invalid unprojection dimensions")
    return xy


class SurfaceDepth:
    """One verified mesh and one reusable CPU raycasting acceleration scene.

    ``axes_world`` is the world-to-camera rotation R: its rows are camera
    right, down, forward in map coordinates. ``center`` is camera center C,
    not the translation -R C. Pixel coordinates use native camera calibration.
    Missing, out-of-image, and failed-unprojection samples remain NaN.
    """

    def __init__(self, manifest_path, mesh_path, manifest, scene, open3d):
        self.manifest_path = manifest_path
        self.mesh_path = mesh_path
        self.coordinate_frame_id = manifest["coordinate_frame_id"]
        self.meters_per_map_unit = manifest["meters_per_map_unit"]
        self.scale_source = manifest["scale_source"]
        self._scene = scene
        self._open3d = open3d

    @classmethod
    def load(cls, manifest_path, *, coordinate_frame_id, source_bundle_path):
        """Validate provenance and raw-map mesh, then build its CPU scene once."""
        manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        bundle_path = Path(source_bundle_path).expanduser().resolve(strict=True)
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict) or _REQUIRED_FIELDS - manifest.keys():
            raise ValueError("surface manifest is missing required mvs-surface/v1 fields")
        if manifest["schema"] != SURFACE_SCHEMA:
            raise ValueError("unsupported surface manifest schema")
        if (
            not isinstance(coordinate_frame_id, str)
            or not coordinate_frame_id
            or manifest["coordinate_frame_id"] != coordinate_frame_id
        ):
            raise ValueError("surface coordinate frame mismatch")
        if manifest["geometry_units"] != "map":
            raise ValueError("surface geometry_units must be map, not meters/display coordinates")
        scale = manifest["meters_per_map_unit"]
        if scale is not None:
            manifest["meters_per_map_unit"] = _positive_scale(scale, "manifest scale")
        source = manifest["scale_source"]
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("surface scale_source must be a non-empty string or null")
        for field in ("source_bundle_sha256", "mesh_sha256"):
            if not isinstance(manifest[field], str) or not manifest[field]:
                raise ValueError(f"surface {field} must be a SHA-256 digest")
        verify_file_sha256(bundle_path, manifest["source_bundle_sha256"])
        with bundle_path.open(encoding="utf-8") as stream:
            bundle = json.load(stream)
        if not isinstance(bundle, dict) or bundle.get("coordinate_frame_id") != coordinate_frame_id:
            raise ValueError("source bundle coordinate frame mismatch")
        mesh_name = manifest["mesh"]
        if not isinstance(mesh_name, str) or not mesh_name or Path(mesh_name).is_absolute():
            raise ValueError("surface mesh must be a relative path")
        mesh_path = (manifest_path.parent / mesh_name).resolve(strict=True)
        if not mesh_path.is_file():
            raise ValueError("surface mesh must be a regular file")
        verify_file_sha256(mesh_path, manifest["mesh_sha256"])

        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("surface depth requires the optional Open3D CPU dependency") from exc
        mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.all(np.isfinite(vertices))
            or np.any(np.abs(vertices) > np.finfo(np.float32).max)
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or not len(triangles)
            or np.any(triangles < 0)
            or np.any(triangles >= len(vertices))
        ):
            raise ValueError("surface mesh requires finite map vertices and valid triangle indices")
        vertices = vertices.astype(np.float32)
        # Reject triangles that collapse at the CPU scene's float32 precision.
        for start in range(0, len(triangles), _MAX_RAYS):
            points = vertices[triangles[start : start + _MAX_RAYS]].astype(np.float64)
            cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
            if np.any(np.all(cross == 0, axis=1)):
                raise ValueError("surface mesh contains degenerate triangles")
        device = o3d.core.Device("CPU:0")
        scene = o3d.t.geometry.RaycastingScene(device=device)
        scene.add_triangles(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32, device=device),
            o3d.core.Tensor(
                triangles.astype(np.uint32), dtype=o3d.core.Dtype.UInt32, device=device
            ),
        )
        return cls(manifest_path, mesh_path, manifest, scene, o3d)

    def _metric_scale(self, explicit):
        if explicit is not None:
            return _positive_scale(explicit, "session scale")
        if self.meters_per_map_unit is None or self.scale_source is None:
            raise ValueError(
                "metric surface depth requires a reliable session or calibrated manifest scale"
            )
        return self.meters_per_map_unit

    def _sample(self, calibration, center, axes, pixels, scale):
        output = np.full(len(pixels), np.nan, dtype=np.float32)
        width, height, _, _ = calibration
        for start in range(0, len(pixels), _MAX_RAYS):
            batch = pixels[start : start + _MAX_RAYS]
            valid = (
                np.all(np.isfinite(batch), axis=1)
                & (batch[:, 0] >= 0)
                & (batch[:, 0] < width)
                & (batch[:, 1] >= 0)
                & (batch[:, 1] < height)
            )
            indices = np.flatnonzero(valid)
            if not len(indices):
                continue
            xy = _unproject(calibration, batch[indices])
            camera_rays = np.column_stack((xy, np.ones(len(xy))))
            world_rays = camera_rays @ axes
            ray_valid = np.all(np.isfinite(world_rays), axis=1) & np.all(
                np.abs(world_rays) <= np.finfo(np.float32).max, axis=1
            )
            indices = indices[ray_valid]
            if not len(indices):
                continue
            rays = np.empty((len(indices), 6), dtype=np.float32)
            rays[:, :3] = center
            rays[:, 3:] = world_rays[ray_valid]
            hits = self._scene.cast_rays(self._open3d.core.Tensor(rays))["t_hit"].numpy()
            # t_hit is a ray parameter, not range. Project the actual float32
            # direction onto camera forward to obtain z before the sole scale.
            depth = hits.astype(np.float64) * (rays[:, 3:] @ axes[2]) * scale
            finite = np.isfinite(depth) & (depth > 0) & (depth <= np.finfo(np.float32).max)
            converted = depth[finite].astype(np.float32)
            nonzero = converted > 0
            output[start + indices[finite][nonzero]] = converted[nonzero]
        return output

    def sample(self, camera, center, axes_world, pixels, *, meters_per_map_unit=None):
        """Return N float32 camera-z depths in meters for native pixel xy rows.

        An explicit positive scale is the caller's assertion of reliable session
        calibration and overrides the manifest. Invalid supplied scales raise;
        they never silently fall back. No metric scale means no metric output.
        """
        calibration = _calibration(camera)
        center, axes = _pose(center, axes_world)
        scale = self._metric_scale(meters_per_map_unit)
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixels must have shape (N, 2) in native camera coordinates")
        return self._sample(calibration, center, axes, pixels, scale)

    def render(
        self, camera, center, axes_world, *, width=None, height=None, meters_per_map_unit=None
    ):
        """Return HxW float32 meter z-depth at pixel centers, without hole fill.

        Optional output dimensions resample the native calibration, rather than
        treating reduced-resolution pixel coordinates as native coordinates.
        Output (x,y) samples native ((x+.5)*native_width/width,
        (y+.5)*native_height/height), the COLMAP pixel-center convention.
        """
        calibration = _calibration(camera)
        center, axes = _pose(center, axes_world)
        scale = self._metric_scale(meters_per_map_unit)
        native_width, native_height, _, _ = calibration
        width = native_width if width is None else _dimension(width, "render width")
        height = native_height if height is None else _dimension(height, "render height")
        output = np.empty((height, width), dtype=np.float32)
        flat = output.reshape(-1)
        for start in range(0, len(flat), _MAX_RAYS):
            indices = np.arange(start, min(start + _MAX_RAYS, len(flat)))
            pixels = np.column_stack(
                (
                    (indices % width + 0.5) * (native_width / width),
                    (indices // width + 0.5) * (native_height / height),
                )
            )
            flat[start : start + len(indices)] = self._sample(
                calibration, center, axes, pixels, scale
            )
        return output
