from __future__ import annotations

import numpy as np

from olympe_live_backend import LiveAnafiVideoStream


class _StampedGrabber:
    def __init__(self):
        self.stamp = 1.0
        self.calls = 0

    def peek_stamp(self):
        return self.stamp

    def __call__(self):
        self.calls += 1
        return np.zeros((2, 3, 3), dtype=np.uint8), self.stamp


def test_duplicate_stamp_is_filtered_before_frame_copy():
    grabber = _StampedGrabber()
    stream = LiveAnafiVideoStream(grabber)

    assert stream.next_frame(only_new=True) is not None
    assert grabber.calls == 1

    assert stream.next_frame(only_new=True) is None
    assert grabber.calls == 1

    grabber.stamp = 2.0
    assert stream.next_frame(only_new=True) is not None
    assert grabber.calls == 2


def test_live_stream_uses_atomic_no_copy_sample_with_timing():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    class Grabber:
        fps = 30.0

        @staticmethod
        def peek_stamp():
            return 3.0

        @staticmethod
        def latest_frame_with_timing():
            return frame, 3.0, {"frame_callback_enter_mono_ns": 123}

        def __call__(self):
            raise AssertionError("copying __call__ path must not be used")

    stream = LiveAnafiVideoStream(Grabber())
    assert stream.next_frame() is frame
    assert stream.last_timing == {"frame_callback_enter_mono_ns": 123}
