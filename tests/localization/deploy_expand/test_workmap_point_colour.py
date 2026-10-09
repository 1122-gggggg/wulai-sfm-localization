"""New session points take their colour from the capture they were triangulated in."""
from __future__ import annotations

import numpy as np

from online_workmap import OnlineWorkMapTracker


def _tracker(frame, size=(8, 4)):
    tracker = object.__new__(OnlineWorkMapTracker)
    tracker._frame_rgb = frame
    tracker._w, tracker._h = size
    return tracker


def test_colour_is_the_3x3_mean_at_the_scaled_pixel():
    frame = np.zeros((8, 16, 3), np.uint8)          # 2x the 8x4 fast-loop grid
    frame[4:7, 6:9] = (90, 180, 45)                 # block around fast-loop pixel (3.5, 2.5) -> (7, 5)
    colours = _tracker(frame)._sample_rgb(np.array([[3.5, 2.5], [0.0, 0.0]]))
    np.testing.assert_array_equal(colours, [[90, 180, 45], [0, 0, 0]])


def test_no_colour_without_a_colour_frame_or_points_and_edges_are_clamped():
    assert _tracker(None)._sample_rgb(np.array([[1.0, 1.0]])) is None
    assert _tracker(np.zeros((8, 16), np.uint8))._sample_rgb(np.array([[1.0, 1.0]])) is None
    assert _tracker(np.zeros((8, 16, 3), np.uint8))._sample_rgb(np.zeros((0, 2))) is None
    frame = np.full((8, 16, 3), 7, np.uint8)
    np.testing.assert_array_equal(_tracker(frame)._sample_rgb(np.array([[99.0, -9.0]])), [[7, 7, 7]])
