"""Consumer-visible raw-map surface, calibration, provenance and meter-depth invariants."""

import hashlib
import json

import numpy as np
import pytest

from surface_depth import SurfaceDepth


CAMERA = {"model": "PINHOLE", "width": 200, "height": 120, "params": [50, 60, 100, 60]}
FRAME = "raw-colmap-test-frame"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quad(x0, x1, y0, y1, z):
    points = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=float)
    return points, np.array([[0, 1, 2], [0, 2, 3]])


def _combine(*meshes):
    vertices, triangles, offset = [], [], 0
    for points, faces in meshes:
        vertices.append(points)
        triangles.append(faces + offset)
        offset += len(points)
    return np.concatenate(vertices), np.concatenate(triangles)


def _write_mesh(path, vertices, triangles):
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    points = "".join(" ".join(str(float(value)) for value in point) + "\n" for point in vertices)
    faces = "".join("3 " + " ".join(str(int(value)) for value in face) + "\n" for face in triangles)
    path.write_text(header + points + faces, encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    def create(mesh=None, *, scale=1.0, scale_source="surveyed-baseline"):
        if mesh is None:
            mesh = _quad(-8, 8, -8, 8, 4)
        bundle = tmp_path / "direct_bundle.json"
        bundle.write_text(
            json.dumps(
                {
                    "schema": "direct-localization-bundle/v1",
                    "coordinate_frame_id": FRAME,
                }
            ),
            encoding="utf-8",
        )
        mesh_path = tmp_path / "dense_surface" / "surface.ply"
        mesh_path.parent.mkdir(exist_ok=True)
        _write_mesh(mesh_path, *mesh)
        manifest = {
            "schema": "mvs-surface/v1",
            "coordinate_frame_id": FRAME,
            "source_bundle_sha256": _digest(bundle),
            "mesh": "dense_surface/surface.ply",
            "mesh_sha256": _digest(mesh_path),
            "geometry_units": "map",
            "meters_per_map_unit": scale,
            "scale_source": scale_source,
            "reconstruction": {"algorithm": "fixture triangles"},
        }
        path = tmp_path / "dense_surface.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, bundle, mesh_path

    return create


def _load(paths, *, frame=FRAME):
    return SurfaceDepth.load(paths[0], coordinate_frame_id=frame, source_bundle_path=paths[1])


def _update_manifest(paths, **changes):
    path = paths[0]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(changes)
    path.write_text(json.dumps(raw), encoding="utf-8")


def _require_open3d():
    pytest.importorskip("open3d", reason="surface raycasting requires optional Open3D CPU")


def test_frontmost_surface_returns_z_not_range_and_excludes_behind_camera(release):
    _require_open3d()
    mesh = _combine(
        _quad(-8, 8, -8, 8, -1),
        _quad(-8, 8, -8, 8, 4),
        _quad(-1, 1, -1, 1, 2),
    )
    surface = _load(release(mesh, scale=1.7))
    depth = surface.sample(CAMERA, [0, 0, 0], np.eye(3), [[100, 60], [120, 60], [150, 60]])
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [3.4, 3.4, 6.8], rtol=2e-6)
    # The last ray has camera x/z=1, so Euclidean range is sqrt(2) times z.
    assert not np.isclose(depth[-1], 6.8 * np.sqrt(2))


def test_raw_map_center_and_world_to_camera_rows_are_not_display_transformed(release):
    _require_open3d()
    yaw, pitch, roll = 0.7, -0.4, 0.2
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    axes = rz @ ry @ rx
    center = np.array([13.0, -6.0, 11.0])
    vertices, triangles = _combine(_quad(-8, 8, -8, 8, 4), _quad(-1, 1, -1, 1, 2))
    surface = _load(release((vertices @ axes + center, triangles), scale=0.3))
    depth = surface.sample(CAMERA, center, axes, [[100, 60], [150, 60], [100, 110]])
    np.testing.assert_allclose(depth, [0.6, 1.2, 1.2], rtol=2e-6, atol=1e-6)


def test_unknown_holes_out_of_image_and_invalid_pixels_stay_nan(release):
    _require_open3d()
    surface = _load(release(_combine(_quad(-6, -1, -6, 6, 4), _quad(1, 6, -6, 6, 4))))
    pixels = [[100, 60], [125, 60], [-1, 60], [200, 60], [100, 120], [np.nan, 60], [np.inf, 60]]
    depths = surface.sample(CAMERA, [0, 0, 0], np.eye(3), pixels)
    assert np.isnan(depths[[0, 2, 3, 4, 5, 6]]).all()
    assert depths[1] == pytest.approx(4)
    assert surface.sample(CAMERA, [0, 0, 0], np.eye(3), np.empty((0, 2))).shape == (0,)
    image = surface.render(CAMERA, [0, 0, 0], np.eye(3), width=10, height=6)
    assert np.isnan(image[:, 4:6]).all()


def test_only_behind_camera_mesh_has_no_depth(release):
    _require_open3d()
    surface = _load(release(_quad(-8, 8, -8, 8, -4)))
    assert np.isnan(surface.sample(CAMERA, [0, 0, 0], np.eye(3), [[100, 60], [150, 60]])).all()


def test_simple_pinhole_has_same_depth_as_equivalent_pinhole(release):
    _require_open3d()
    surface = _load(release())
    simple = {"model": "SIMPLE_PINHOLE", "width": 200, "height": 120, "params": [50, 100, 60]}
    pinhole = dict(CAMERA, params=[50, 50, 100, 60])
    pixels = [[100, 60], [130, 80], [170, 110]]
    np.testing.assert_array_equal(
        surface.sample(simple, [0, 0, 0], np.eye(3), pixels),
        surface.sample(pinhole, [0, 0, 0], np.eye(3), pixels),
    )


def test_distortion_unprojects_to_surface_instead_of_using_pinhole_approximation(release):
    _require_open3d()
    pycolmap = pytest.importorskip("pycolmap")
    if not hasattr(pycolmap.Camera, "cam_from_img"):
        pytest.skip("installed pycolmap has no calibrated camera unprojection")
    camera = dict(CAMERA, model="OPENCV", params=[50, 60, 100, 60, 0.6, 0.1, 0.01, -0.02])
    colmap = pycolmap.Camera(**camera)
    surface = _load(release(_quad(0.7, 0.9, -0.2, 0.2, 4)))
    # This narrow real surface is visible at x/z=.224 but a distorted pixel is
    # outside its pinhole projection. Compare physical geometry, not internals.
    pixel = np.asarray(colmap.img_from_cam(np.array([0.896, 0.0, 4.0])))
    result = surface.sample(camera, [0, 0, 0], np.eye(3), pixel.reshape(1, 2))
    assert result[0] == pytest.approx(4, rel=2e-6)
    pinhole = dict(CAMERA)
    assert np.isnan(surface.sample(pinhole, [0, 0, 0], np.eye(3), pixel.reshape(1, 2))[0])


def test_distorted_camera_rotation_matches_raw_map_geometry(release):
    _require_open3d()
    pycolmap = pytest.importorskip("pycolmap")
    if not hasattr(pycolmap.Camera, "cam_from_img"):
        pytest.skip("installed pycolmap has no calibrated camera unprojection")
    camera = dict(CAMERA, model="SIMPLE_RADIAL", params=[50, 100, 60, 0.25])
    colmap = pycolmap.Camera(**camera)
    axes = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)
    center = np.array([20, -4, 6], dtype=float)
    vertices, triangles = _quad(-2, 2, -2, 2, 4)
    surface = _load(release((vertices @ axes + center, triangles), scale=2.0))
    pixel = np.asarray(colmap.img_from_cam(np.array([1.0, 0.5, 4.0])))
    depth = surface.sample(camera, center, axes, pixel.reshape(1, 2))
    assert depth[0] == pytest.approx(8, rel=2e-6)


def test_session_scale_overrides_manifest_without_scaling_geometry_twice(release):
    _require_open3d()
    surface = _load(release(scale=2.0))
    pixels = [[100, 60], [150, 60]]
    default = surface.sample(CAMERA, [0, 0, 1], np.eye(3), pixels)
    override = surface.sample(CAMERA, [0, 0, 1], np.eye(3), pixels, meters_per_map_unit=0.3)
    np.testing.assert_allclose(default, [6, 6])
    np.testing.assert_allclose(override, [0.9, 0.9])
    # A subsequent query must still see map-space geometry and manifest scale.
    np.testing.assert_array_equal(surface.sample(CAMERA, [0, 0, 1], np.eye(3), pixels), default)


@pytest.mark.parametrize("scale,source", [(None, None), (2.0, None)])
def test_uncalibrated_manifest_rejects_metric_depth_but_session_scale_enables_it(
    release, scale, source
):
    _require_open3d()
    surface = _load(release(scale=scale, scale_source=source))
    with pytest.raises(ValueError, match="requires a reliable"):
        surface.sample(CAMERA, [0, 0, 0], np.eye(3), [[100, 60]])
    with pytest.raises(ValueError, match="requires a reliable"):
        surface.render(CAMERA, [0, 0, 0], np.eye(3), width=2, height=2)
    assert surface.sample(CAMERA, [0, 0, 0], np.eye(3), [[100, 60]], meters_per_map_unit=0.25)[
        0
    ] == pytest.approx(1, rel=2e-6)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "2"])
def test_invalid_explicit_scale_never_falls_back_to_manifest(release, scale):
    _require_open3d()
    surface = _load(release())
    with pytest.raises(ValueError, match="session scale"):
        surface.sample(CAMERA, [0, 0, 0], np.eye(3), [[100, 60]], meters_per_map_unit=scale)


def test_tiled_render_matches_pixel_center_calibration_at_resized_resolution(release):
    _require_open3d()
    vertices, triangles = _quad(-100, 100, -100, 100, 4)
    vertices[:, 2] = 4 + 0.3 * vertices[:, 0] + 0.1 * vertices[:, 1]
    surface = _load(release((vertices, triangles), scale=1.2))
    image = surface.render(CAMERA, [0, 0, 0], np.eye(3), width=257, height=257)
    assert image.shape == (257, 257)
    assert image.dtype == np.float32
    yy, xx = np.indices(image.shape)
    native_x = (xx + 0.5) * CAMERA["width"] / image.shape[1]
    native_y = (yy + 0.5) * CAMERA["height"] / image.shape[0]
    expected = 4.8 / (1 - 0.3 * (native_x - 100) / 50 - 0.1 * (native_y - 60) / 60)
    np.testing.assert_allclose(image, expected, rtol=3e-6, atol=2e-6)
    pixels = np.column_stack((native_x.ravel(), native_y.ravel()))
    sampled = surface.sample(CAMERA, [0, 0, 0], np.eye(3), pixels)
    np.testing.assert_array_equal(image.ravel(), sampled)


@pytest.mark.parametrize(
    "center,axes",
    [
        ([0, 0], np.eye(3)),
        ([0, np.nan, 0], np.eye(3)),
        ([0, 0, 0], np.diag([1, 1, -1])),
        ([0, 0, 0], np.eye(3) * 2),
        ([0, 0, 0], np.zeros((3, 3))),
        ([0, 0, 0], np.full((3, 3), np.nan)),
    ],
)
def test_invalid_centers_and_nonrotations_are_rejected(release, center, axes):
    _require_open3d()
    surface = _load(release())
    with pytest.raises(ValueError, match="center|axes_world"):
        surface.sample(CAMERA, center, axes, [[100, 60]])


@pytest.mark.parametrize(
    "change",
    [
        {"width": 0},
        {"height": True},
        {"width": 200.5},
        {"params": [50, 60, 100]},
        {"params": [0, 60, 100, 60]},
        {"params": [50, -1, 100, 60]},
        {"params": [50, 60, np.nan, 60]},
    ],
)
def test_invalid_calibration_is_rejected(release, change):
    _require_open3d()
    surface = _load(release())
    with pytest.raises(ValueError):
        surface.sample(dict(CAMERA, **change), [0, 0, 0], np.eye(3), [[100, 60]])


def test_surface_rejects_requested_coordinate_frame_mismatch_before_dependency_load(release):
    with pytest.raises(ValueError, match="coordinate frame mismatch"):
        _load(release(), frame="gravity-aligned-display-frame")


def test_surface_rejects_bundle_frame_mismatch_even_when_digest_matches(release):
    paths = release()
    paths[1].write_text(json.dumps({"coordinate_frame_id": "other-map"}), encoding="utf-8")
    _update_manifest(paths, source_bundle_sha256=_digest(paths[1]))
    with pytest.raises(ValueError, match="source bundle coordinate frame mismatch"):
        _load(paths)


@pytest.mark.parametrize("target", [1, 2])
def test_surface_is_bound_to_exact_bundle_and_mesh_bytes(release, target):
    paths = release()
    with paths[target].open("ab") as stream:
        stream.write(b"\nchanged")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load(paths)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"schema": "mvs-surface/v2"}, "schema"),
        ({"geometry_units": "meters"}, "geometry_units"),
        ({"meters_per_map_unit": 0}, "manifest scale"),
        ({"meters_per_map_unit": -1}, "manifest scale"),
        ({"meters_per_map_unit": float("inf")}, "manifest scale"),
        ({"meters_per_map_unit": True}, "manifest scale"),
        ({"scale_source": ""}, "scale_source"),
        ({"mesh": "/tmp/unbound-surface.ply"}, "relative path"),
        ({"mesh_sha256": None}, "SHA-256 digest"),
    ],
)
def test_manifest_rejects_nonmap_geometry_and_invalid_provenance(release, change, error):
    paths = release()
    _update_manifest(paths, **change)
    with pytest.raises(ValueError, match=error):
        _load(paths)


def test_manifest_requires_explicit_scale_and_units_fields(release):
    paths = release()
    raw = json.loads(paths[0].read_text(encoding="utf-8"))
    del raw["meters_per_map_unit"]
    paths[0].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        _load(paths)


@pytest.mark.parametrize(
    "mesh",
    [
        (np.array([[0, 0, 4], [1, 0, 4], [2, 0, 4]], dtype=float), np.array([[0, 1, 2]])),
        (np.array([[0, 0, 4], [1, 0, 4], [0, 1, 4]], dtype=float), np.empty((0, 3), dtype=int)),
    ],
)
def test_empty_or_degenerate_surfaces_are_not_fabricated_as_depth(release, mesh):
    _require_open3d()
    with pytest.raises(ValueError, match="triangle"):
        _load(release(mesh))


def test_surface_and_bundle_remain_valid_when_release_is_relocated(release, tmp_path):
    _require_open3d()
    paths = release(scale=0.5)
    moved = tmp_path.parent / (tmp_path.name + "-relocated")
    tmp_path.rename(moved)
    relocated = tuple(moved / path.relative_to(tmp_path) for path in paths)
    surface = _load(relocated)
    assert surface.manifest_path == relocated[0]
    assert surface.mesh_path == relocated[2]
    assert surface.sample(CAMERA, [0, 0, 0], np.eye(3), [[100, 60]])[0] == pytest.approx(
        2, rel=2e-6
    )
