#!/usr/bin/env python3
"""Pure-Python contract tests for the latest-only Parrot RTSP frame source."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
FLIGHT_CONTROL_ROOT = ROOT / "定位演算法" / "flight_control"
sys.path.insert(0, str(FLIGHT_CONTROL_ROOT))

import rtsp_frame_source as rfs


def _grabber(*, stale_s=5.0, require_source_timestamps=False):
    return rfs.RtspFrameGrabber(
        SimpleNamespace(address="192.168.42.1"),
        resize=None,
        stale_s=stale_s,
        require_source_timestamps=require_source_timestamps,
    )


def test_returns_latest_rgb_with_explicitly_degraded_receipt_timing():
    grabber = _grabber()
    frame = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    received = time.monotonic()

    grabber._store(frame, receipt_stamp=received)

    sample = grabber()
    assert sample is not None
    rgb, stamp = sample
    assert rgb is not grabber._latest
    np.testing.assert_array_equal(rgb, frame)
    assert stamp == pytest.approx(received)
    latest = grabber.latest_frame_with_timing()
    assert latest is not None
    assert latest[0] is grabber._latest
    assert latest[1] == pytest.approx(received)
    assert latest[2]["source_timestamp_degraded"] is True
    metadata = grabber.stream_metadata
    assert metadata["transport"] == "RTSP"
    assert metadata["rtsp_path"] == "/live"
    assert metadata["source_timestamp_capable"] is False
    assert metadata["source_timestamp_trusted"] is False
    assert metadata["stamp_source"] == rfs.RTSP_TIMESTAMP_KIND


def test_stale_frames_are_not_returned():
    grabber = _grabber(stale_s=0.05)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    grabber._store(frame, receipt_stamp=time.monotonic() - 1.0)

    assert grabber() is None
    assert grabber.latest_frame_with_timing() is None
    assert grabber.peek_stamp() is None
    assert not grabber.is_healthy()


def test_frozen_frames_are_rejected_until_a_distinct_frame_arrives():
    grabber = _grabber()
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    for _ in range(rfs.FROZEN_DUP_FRAMES + 1):
        grabber._store(frame)

    assert grabber() is None
    assert grabber.latest_frame_with_timing() is None
    assert grabber.peek_stamp() is None
    assert not grabber.is_healthy()

    grabber._store(frame + 1)
    assert grabber() is not None
    assert grabber.is_healthy()


def test_source_timestamp_requirement_fails_closed():
    grabber = _grabber(require_source_timestamps=True)
    grabber._store(np.zeros((8, 12, 3), dtype=np.uint8))

    assert grabber() is None
    assert grabber.latest_frame_with_timing() is None
    assert grabber.peek_stamp() is None
    assert not grabber.is_healthy()
    assert grabber.timestamp_degraded
    assert grabber.latest_source_ntp_us() is None
    assert grabber.inspection_sample_after(1_000_000, min_source_advance_s=1.0) is None


@pytest.mark.parametrize("controller", ["skycontroller3", "drone"])
def test_connect_populates_initial_aircraft_telemetry(controller, monkeypatch):
    from log_anafi_telemetry import sample_once

    class Query:
        def __init__(self, fullname):
            self.fullname = fullname

        def __call__(self):
            return self

    class FakeClient:
        def __init__(self, _address, *, device_type, port, lib_dir):
            assert port == 44445  # Discovery announces the command port (44444).
            self.messages = SimpleNamespace(
                Skyctrl=SimpleNamespace(
                    Common=SimpleNamespace(AllStates=Query("Skyctrl.Common.AllStates")),
                    Settings=SimpleNamespace(AllSettings=Query("Skyctrl.Settings.AllSettings")),
                ),
                Common=SimpleNamespace(
                    Common=SimpleNamespace(AllStates=Query("Common.Common.AllStates")),
                    Settings=SimpleNamespace(AllSettings=Query("Common.Settings.AllSettings")),
                    CommonState=SimpleNamespace(BatteryStateChanged=Query("battery")),
                ),
                Ardrone3=SimpleNamespace(PilotingState=SimpleNamespace(
                    FlyingStateChanged=Query("flying"),
                )),
            )
            self.states = {}
            self.skycontroller = device_type == rfs.ARSDK_DEVICE_SKYCTRL_3
            self.sky_ready = not self.skycontroller

        def wait_connected(self, _timeout):
            pass

        def __call__(self, query):
            def complete(_timeout):
                if query.fullname == "Skyctrl.Common.AllStates":
                    self.sky_ready = True
                if query.fullname == "Common.Common.AllStates" and self.sky_ready:
                    self.states["battery"] = {"percent": 62}
                    self.states["flying"] = {"state": "LANDED"}
                return SimpleNamespace(success=lambda: True)

            return SimpleNamespace(wait=complete)

        def get_state(self, query):
            return self.states.get(query.fullname)

        def disconnect(self):
            pass

    monkeypatch.setitem(
        sys.modules, "arsdk_ng",
        SimpleNamespace(ArsdkNgClient=FakeClient, ArsdkNgError=RuntimeError),
    )
    client = rfs.connect("192.168.53.1", controller=controller)
    row = sample_once(client, [
        ("battery", client.messages.Common.CommonState.BatteryStateChanged),
        ("flying", client.messages.Ardrone3.PilotingState.FlyingStateChanged),
    ])
    assert row["battery"] == {"percent": 62}
    assert row["flying"] == {"state": "LANDED"}
