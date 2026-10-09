"""Offline builder contracts and small CPU geometry fixtures; no GPU or hardware."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import build_mvs_surface as tool


@pytest.mark.parametrize(
    "scale,source",
    [
        (0, "survey"),
        (-1, "survey"),
        (float("nan"), "survey"),
        (float("inf"), "survey"),
        (1, None),
        (None, "survey"),
        (1, " "),
    ],
)
def test_scale_requires_positive_finite_value_and_provenance(scale, source):
    with pytest.raises(ValueError):
        tool.validate_scale(scale, source)


def test_cpu_only_pycolmap_rejects_cuda_reconstruction(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pycolmap", SimpleNamespace(__version__="4.0.4", has_cuda=False)
    )
    with pytest.raises(RuntimeError):
        tool.load_cuda_pycolmap()


def test_cuda_binding_requires_verified_version(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pycolmap", SimpleNamespace(__version__="4.1.0", has_cuda=True)
    )
    with pytest.raises(RuntimeError, match="4.0.4"):
        tool.load_cuda_pycolmap()


def _registered_source(tmp_path, *, wrong_size=False, corrupt=False):
    Image = pytest.importorskip("PIL.Image")
    paths = {}
    hashes = {}
    images = {}
    for index in range(3):
        name = f"video/{index}.png"
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (16, 8) if index or not wrong_size else (8, 4)).save(path)
        hashes[name] = tool.sha256_file(path)
        paths[name] = path
        images[index] = SimpleNamespace(name=name, camera_id=1)
    if corrupt:
        paths["video/1.png"].write_bytes(b"changed")
    verified = []

    def verify(name):
        verified.append(name)
        if tool.sha256_file(paths[name]) != hashes[name]:
            raise ValueError("reference image digest mismatch")
        return paths[name]

    assets = SimpleNamespace(verify_reference_image=verify)
    reconstruction = SimpleNamespace(
        reg_image_ids=lambda: [2, 0, 1],
        image=lambda index: images[index],
        camera=lambda _index: SimpleNamespace(width=16, height=8),
    )
    return assets, reconstruction, verified


def test_all_registered_images_are_verified_at_model_resolution(tmp_path):
    assets, reconstruction, verified = _registered_source(tmp_path)
    assert tool.verify_registered_images(assets, reconstruction) == [
        "video/0.png",
        "video/1.png",
        "video/2.png",
    ]


def test_resized_registered_image_is_rejected(tmp_path):
    assets, reconstruction, _ = _registered_source(tmp_path, wrong_size=True)
    with pytest.raises(ValueError, match="COLMAP camera.*original"):
        tool.verify_registered_images(assets, reconstruction)


def test_corrupted_registered_image_is_rejected(tmp_path):
    assets, reconstruction, _ = _registered_source(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="digest mismatch"):
        tool.verify_registered_images(assets, reconstruction)


def test_default_uses_all_registered_images_and_selection_must_be_registered():
    registered = ["a", "b", "c", "d"]
    assert tool.select_images(registered, None) == registered
    assert tool.select_images(registered, ["c", "a", "b", "a"]) == ["a", "b", "c"]
    with pytest.raises(ValueError, match="not registered"):
        tool.select_images(registered, ["a", "b", "missing"])
    with pytest.raises(ValueError, match="at least three"):
        tool.select_images(registered, ["a", "b"])


def _track_model():
    def point(point_id):
        return SimpleNamespace(point3D_id=point_id, has_point3D=lambda: True)

    images = {
        index: SimpleNamespace(name=name, points2D=[point(1), point(2)])
        for index, name in enumerate(["a", "b", "c", "outside_subset"])
    }
    return SimpleNamespace(reg_image_ids=lambda: list(images), image=lambda index: images[index])


def test_partial_config_never_auto_selects_unavailable_sources(tmp_path):
    path = tmp_path / "patch-match.cfg"
    tool.write_patch_match_config(path, _track_model(), ["a", "b", "c"], partial=True)
    assert path.read_text() == "a\nb, c\nb\na, c\nc\na, b\n"
    assert "outside_subset" not in path.read_text()


def test_full_config_uses_colmap_overlap_selection(tmp_path):
    path = tmp_path / "patch-match.cfg"
    tool.write_patch_match_config(path, _track_model(), ["a", "b", "c"], partial=False)
    assert path.read_text() == "a\n__auto__, 8\nb\n__auto__, 8\nc\n__auto__, 8\n"


def test_nonoverlapping_partial_frames_fail_before_patch_match(tmp_path):
    model = _track_model()
    model.image(0).points2D = []
    with pytest.raises(ValueError, match="two overlapping selected sources"):
        tool.write_patch_match_config(
            tmp_path / "patch-match.cfg", model, ["a", "b", "c"], partial=True
        )


def _measurements():
    return np.asarray([(x, y, 0) for x in [0, 1, 2, 10, 11, 12] for y in range(3)], dtype=float)


def test_support_filter_keeps_measured_patch_but_rejects_bridge_and_unseen_area():
    pytest.importorskip("scipy.spatial")
    vertices = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [10, 0, 0],
            [10, 1, 0],
            [100, 0, 0],
            [101, 0, 0],
            [100, 1, 0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 4], [5, 6, 7], [0, 0, 0]])
    mask, spacing = tool.supported_faces(vertices, faces, _measurements())
    np.testing.assert_array_equal(mask, [True, False, False, False])
    assert spacing > 0


def test_support_filter_is_in_original_units_not_assumed_meters():
    pytest.importorskip("scipy.spatial")
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2]])
    original, spacing = tool.supported_faces(vertices, faces, _measurements())
    scaled, scaled_spacing = tool.supported_faces(vertices * 7, faces, _measurements() * 7)
    np.testing.assert_array_equal(original, scaled)
    assert scaled_spacing == pytest.approx(spacing * 7)


def test_mesh_filter_preserves_coordinates_and_never_adds_faces(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy.spatial")
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [100, 0, 0], [101, 0, 0], [100, 1, 0]])
    raw = tmp_path / "poisson.ply"
    fused = tmp_path / "fused.ply"
    target = tmp_path / "supported.ply"
    trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False).export(raw)
    trimesh.points.PointCloud(_measurements()).export(fused)
    metadata = tool.filter_surface(raw, fused, target)
    mesh = trimesh.load(target, process=False)
    assert metadata["poisson_triangles"] == 2
    assert metadata["supported_triangles"] == 1
    np.testing.assert_array_equal(mesh.vertices, vertices[:3])
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])


def test_mesh_filter_refuses_empty_supported_result(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy.spatial")
    raw = tmp_path / "poisson.ply"
    fused = tmp_path / "fused.ply"
    target = tmp_path / "supported.ply"
    trimesh.Trimesh(
        vertices=[[100, 0, 0], [101, 0, 0], [100, 1, 0]], faces=[[0, 1, 2]], process=False
    ).export(raw)
    trimesh.points.PointCloud(_measurements()).export(fused)
    with pytest.raises(ValueError, match="No measurement-supported triangles"):
        tool.filter_surface(raw, fused, target)
    assert not target.exists()


def _publish_source(tmp_path):
    bundle_dir = tmp_path / "release" / "localization"
    bundle_dir.mkdir(parents=True)
    bundle = bundle_dir / "direct_bundle.json"
    bundle.write_text("{}")
    mesh = tmp_path / "supported.ply"
    mesh.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n"
        "property float y\nproperty float z\nelement face 1\n"
        "property list uchar int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )
    assets = SimpleNamespace(
        bundle_path=bundle, coordinate_frame_id="raw_frame", sha256=tool.sha256_file(bundle)
    )
    return assets, mesh


def test_atomic_manifest_is_next_to_bundle_and_points_to_immutable_hashed_mesh(
    tmp_path, monkeypatch
):
    assets, mesh = _publish_source(tmp_path)
    replacements = []
    replace = tool.os.replace

    def record_replace(source, destination):
        replacements.append(Path(destination))
        return replace(source, destination)

    monkeypatch.setattr(tool.os, "replace", record_replace)
    manifest = tool.publish_surface(
        assets, mesh, {"coverage_complete": False}, meters_per_map_unit=None, scale_source=None
    )
    path = assets.bundle_path.parent / "dense_surface.json"
    assert json.loads(path.read_text()) == manifest
    assert manifest["schema"] == "mvs-surface/v1"
    assert manifest["coordinate_frame_id"] == "raw_frame"
    assert manifest["geometry_units"] == "map"
    assert manifest["meters_per_map_unit"] is None
    assert manifest["scale_source"] is None
    published_mesh = path.parent / manifest["mesh"]
    assert published_mesh.read_bytes() == mesh.read_bytes()
    assert tool.sha256_file(published_mesh) == manifest["mesh_sha256"]
    assert manifest["source_bundle_sha256"] == assets.sha256
    assert replacements == [published_mesh, path]
    assert not list(path.parent.glob(".dense_surface-*"))


def test_publication_failure_keeps_existing_manifest(tmp_path):
    assets, mesh = _publish_source(tmp_path)
    path = assets.bundle_path.parent / "dense_surface.json"
    path.write_text('{"old": true}\n')
    with pytest.raises(ValueError):
        tool.publish_surface(
            assets, mesh, {"invalid": float("nan")}, meters_per_map_unit=None, scale_source=None
        )
    assert path.read_text() == '{"old": true}\n'
    assert not list(path.parent.glob(".dense_surface-*"))


def test_changed_bundle_is_not_published(tmp_path):
    assets, mesh = _publish_source(tmp_path)
    assets.bundle_path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed during reconstruction"):
        tool.publish_surface(assets, mesh, {}, meters_per_map_unit=None, scale_source=None)
    assert not (assets.bundle_path.parent / "dense_surface.json").exists()


def test_mesh_triangles_degenerate_in_float32_are_removed_before_publication(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy.spatial")
    raw = tmp_path / "poisson.ply"
    fused = tmp_path / "fused.ply"
    target = tmp_path / "supported.ply"
    raw.write_text(
        "ply\nformat ascii 1.0\nelement vertex 6\n"
        "property double x\nproperty double y\nproperty double z\n"
        "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"
        "1 0 0\n1.00000001 0 0\n1 0.00000001 0\n"
        "3 0 1 2\n3 3 4 5\n"
    )
    trimesh.points.PointCloud(_measurements()).export(fused)
    metadata = tool.filter_surface(raw, fused, target)
    assert metadata["poisson_triangles"] == 2
    assert metadata["supported_triangles"] == 1
    mesh = trimesh.load(target, process=False)
    np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_dense_observed_plane_survives_large_extent_without_bridging_gap(tmp_path):
    o3d = pytest.importorskip("open3d")
    coordinates = np.linspace(-0.3, 0.3, 151)
    x, y = np.meshgrid(coordinates, coordinates)
    plane = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 5.0)))
    # A distant observed patch expands the field bounds but must not erase the
    # dense near patch through coarse meshing, or authorize a surface in the gap.
    distant = plane[::30].copy()
    distant[:, 0] += 16.0
    points = np.vstack((plane, distant))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, -1.0], (len(points), 1)))
    fused, raw, supported = (tmp_path / name for name in ("fused.ply", "raw.ply", "supported.ply"))
    o3d.io.write_point_cloud(str(fused), cloud)
    tool.poisson_surface(fused, raw)
    tool.filter_surface(raw, fused, supported)
    mesh = o3d.io.read_triangle_mesh(str(supported))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    x, y = np.meshgrid(np.linspace(-0.25, 0.25, 9), np.linspace(-0.25, 0.25, 9))
    targets = np.vstack(
        (np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 5.0))), [8.0, 0.0, 5.0])
    )
    rays = np.column_stack((np.zeros_like(targets), targets / 5.0)).astype(np.float32)
    depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    np.testing.assert_allclose(depth[:-1], 5.0, atol=0.02, rtol=0)
    assert np.isinf(depth[-1])
