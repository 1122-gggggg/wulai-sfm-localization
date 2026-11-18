"""Bounded binary cloud reads preserve the original global stride sample."""

import io

import numpy as np
import pytest

from map_point_io import _read_binary_ply_points


@pytest.mark.parametrize(
    "normals,colors,alpha", [(False, False, False), (False, True, False), (True, True, True)]
)
@pytest.mark.parametrize("step", [1, 7, 100003])
def test_binary_cloud_sampling_crosses_chunks_without_changing_points(normals, colors, alpha, step):
    fields = [("pos", "<f4", (3,))]
    if normals:
        fields.append(("norm", "<f4", (3,)))
    if colors:
        fields.append(("rgb", "u1", (4 if alpha else 3,)))
    data = np.zeros(131077, dtype=np.dtype(fields))
    data["pos"] = np.arange(len(data) * 3).reshape(-1, 3) * 0.25
    if colors:
        data["rgb"][:, :3] = np.arange(len(data))[:, None] % 251

    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 65536 * data.dtype.itemsize
            return super().read(size)

    actual = _read_binary_ply_points(
        BoundedReader(data.tobytes()),
        len(data),
        step,
        has_normals=normals,
        has_colors=colors,
        has_alpha=alpha,
    )
    np.testing.assert_array_equal(actual[:, :3], data["pos"][::step])
    expected_rgb = data["rgb"][::step, :3] if colors else np.full((len(actual), 3), 200)
    np.testing.assert_array_equal(actual[:, 3:], expected_rgb)


def test_truncated_binary_cloud_keeps_only_complete_records():
    points = np.arange(18, dtype="<f4").reshape(-1, 3)
    result = _read_binary_ply_points(
        io.BytesIO(points.tobytes() + b"partial"),
        20,
        2,
        has_normals=False,
        has_colors=False,
    )
    np.testing.assert_array_equal(result[:, :3], points[::2])


def test_double_precision_positions_are_read_as_coordinates(tmp_path):
    from map_point_io import read_ply_points

    xyz = np.array([[1.5, -2.25, 3.125], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    records = np.zeros(3, dtype=[("pos", "<f8", (3,)), ("rgb", "u1", (3,))])
    records["pos"] = xyz
    records["rgb"] = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    path = tmp_path / "double.ply"
    path.write_bytes(
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        ).encode("ascii")
        + records.tobytes()
    )
    points = read_ply_points(path, 10)
    np.testing.assert_allclose(points[:, :3], xyz)
    np.testing.assert_array_equal(points[:, 3:], records["rgb"])



def _write_dense_ply(path, *, frame_id, map_sha):
    path.write_bytes(
        (
            "ply\nformat binary_little_endian 1.0\n"
            f"comment coordinate_frame_id {frame_id}\n"
            f"comment display_for_map_ply_sha256 {map_sha}\n"
            "element vertex 0\nproperty float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        ).encode("ascii")
    )


def test_dense_display_ply_requires_matching_frame_and_map_digest(tmp_path):
    from map_point_io import dense_display_ply

    map_ply = tmp_path / "map.ply"
    map_ply.write_bytes(b"sparse")
    dense = tmp_path / "map_dense.ply"
    assert dense_display_ply(map_ply, coordinate_frame_id="f", map_ply_sha256="a" * 64) is None

    _write_dense_ply(dense, frame_id="f", map_sha="a" * 64)
    assert dense_display_ply(map_ply, coordinate_frame_id="f", map_ply_sha256="a" * 64) == dense
    # Another release's cloud would move every waypoint snapped to it.
    assert dense_display_ply(map_ply, coordinate_frame_id="g", map_ply_sha256="a" * 64) is None
    assert dense_display_ply(map_ply, coordinate_frame_id="f", map_ply_sha256="b" * 64) is None
    assert dense_display_ply(map_ply, coordinate_frame_id="f", map_ply_sha256=None) is None


def test_folder_discovery_never_offers_the_dense_display_cloud(tmp_path):
    from local_site_assets import discover_ply_files, discover_site_package

    (tmp_path / "map.ply").write_bytes(b"ply\n")
    (tmp_path / "map_dense.ply").write_bytes(b"ply\n")
    assert [path.name for path in discover_ply_files(tmp_path)] == ["map.ply"]
    package = discover_site_package(tmp_path)
    offered = next(asset for asset in package.assets if asset.key == "map_ply")
    assert [path.name for path in offered.candidates] == ["map.ply"]
