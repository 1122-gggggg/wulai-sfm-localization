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
