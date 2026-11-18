"""The VO detection mask must stay pixel-identical to the erode-based original."""

from __future__ import annotations

# Source modules are supplied by the repository's pytest pythonpath.
import cv2
import numpy as np
import pytest

from two_rate_tracker import _occupied_exclusion_mask


def erode_reference(shape: tuple[int, int], occupied: np.ndarray, min_distance: int) -> np.ndarray:
    """The pre-optimisation full-frame erosion, unchanged. Oracle only."""

    mask = np.full(shape, 255, dtype=np.uint8)
    if occupied.size:
        pts = np.asarray(occupied, dtype=int)
        valid = (
            (pts[:, 0] >= 0) & (pts[:, 0] < shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < shape[0])
        )
        pts = pts[valid]
        if len(pts):
            mask[pts[:, 1], pts[:, 0]] = 0
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * min_distance + 1, 2 * min_distance + 1)
            )
            mask = cv2.erode(mask, kernel)
    return mask


@pytest.mark.parametrize("min_distance", [1, 2, 8, 15])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mask_matches_erosion(min_distance: int, seed: int) -> None:
    shape = (120, 200)
    rng = np.random.default_rng(seed)
    interior = np.column_stack([rng.integers(0, shape[1], 40), rng.integers(0, shape[0], 40)])
    corners = np.array([[0, 0], [shape[1] - 1, 0], [0, shape[0] - 1], [shape[1] - 1, shape[0] - 1]])
    out_of_frame = np.array([[-3, 5], [shape[1], 5], [7, -1], [7, shape[0]], [-50, 900]])
    occupied = np.vstack([interior, corners, out_of_frame]).astype(float)
    # Fractional coordinates truncate toward zero exactly as `dtype=int` did.
    occupied += rng.uniform(0.0, 0.9, occupied.shape)
    np.testing.assert_array_equal(
        _occupied_exclusion_mask(shape, occupied, min_distance),
        erode_reference(shape, occupied, min_distance),
    )


@pytest.mark.parametrize(
    "occupied",
    [
        np.zeros((0, 2)),
        np.array([[-5.0, -5.0], [1000.0, 1000.0]]),  # nothing inside the frame
        np.array([[10.0, 10.0]]),
        np.array([[10.0, 10.0], [10.0, 10.0]]),  # duplicates
    ],
)
def test_degenerate_inputs(occupied: np.ndarray) -> None:
    shape = (60, 80)
    np.testing.assert_array_equal(
        _occupied_exclusion_mask(shape, occupied, 8), erode_reference(shape, occupied, 8)
    )
