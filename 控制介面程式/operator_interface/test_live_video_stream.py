from __future__ import annotations

import time

import numpy as np
import pytest

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


def test_live_stream_reports_observed_frame_and_pdraw_metadata():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    class Grabber:
        fps = 29.5
        stream_metadata = {
            "codec": "H.264",
            "codec_evidence": "configured-pdraw-input-contract",
            "codec_observed": False,
            "pdraw_media_name": "DefaultVideo",
            "pdraw_stream_mode": "play",
            "source_width_px": 1920,
            "source_height_px": 1080,
            "source_timestamp_capable": True,
            "source_timestamp_trusted": True,
            "stamp_source": "ntp-mapped",
        }

        @staticmethod
        def peek_stamp():
            return 4.0

        @staticmethod
        def latest_frame_with_timing():
            return frame, 4.0, {"frame_callback_enter_mono_ns": 456}

    stream = LiveAnafiVideoStream(Grabber())
    assert stream.next_frame() is frame

    metadata = stream.metadata_snapshot()
    assert metadata["codec"] == "H.264"
    assert metadata["codec_observed"] is False
    assert metadata["source_width_px"] == 1920
    assert metadata["source_height_px"] == 1080
    assert metadata["ui_frame_width_px"] == 1280
    assert metadata["ui_frame_height_px"] == 720
    assert metadata["observed_fps"] == 29.5
    assert metadata["frames_delivered"] == 1
    assert metadata["source_timestamp_capable"] is True


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), -float("inf")])
def test_live_stream_rejects_non_finite_frame_timestamps(stamp):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    class Grabber:
        @staticmethod
        def peek_stamp():
            return stamp

        @staticmethod
        def latest_frame_with_timing():
            return frame, stamp, {}

    stream = LiveAnafiVideoStream(Grabber())

    assert stream.next_frame() is None
    assert stream.last_stamp == 0.0
    assert stream.output_index == 0


def test_live_stream_rejects_a_future_frame_timestamp():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    stamp = time.monotonic() + 1.0

    class Grabber:
        @staticmethod
        def peek_stamp():
            return stamp

        @staticmethod
        def latest_frame_with_timing():
            return frame, stamp, {}

    stream = LiveAnafiVideoStream(Grabber())

    assert stream.next_frame() is None
    assert stream.last_stamp == 0.0
    assert stream.output_index == 0
