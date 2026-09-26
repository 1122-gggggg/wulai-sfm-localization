#!/usr/bin/env python3
"""Direct ARSDK-NG connection and latest-only Parrot RTSP frame source.

The ARSDK-NG client owns the one control connection. Video is received
independently from the aircraft's RTSP ``/live`` endpoint through OpenCV/FFmpeg.
OpenCV exposes no trustworthy camera capture timestamp here: every frame stamp
is the host monotonic time at which ``VideoCapture.read()`` returned, and is
explicitly marked degraded rather than presented as a sensor timestamp.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

DRONE_IP_REAL = "192.168.42.1"
DRONE_IP_USB = "192.168.43.1"
DRONE_IP_SKYCTRL = "192.168.53.1"
DRONE_IP_SIM = "10.202.0.1"
ARSDK_DEVICE_ANAFI4K = 0x0914
ARSDK_DEVICE_SKYCTRL_3 = 0x0918
ARSDK_PORT = 44445  # Discovery; the device advertises its separate command port.
RTSP_PORT = 554
RTSP_PATH = "/live"
RTSP_TIMESTAMP_KIND = "opencv-rtsp-receipt-degraded"
FROZEN_DUP_FRAMES = 15


def _operator_interface_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "控制介面程式" / "operator_interface"


def connect(
    ip: str = DRONE_IP_REAL,
    controller: str = "auto",
    *,
    timeout: float = 15.0,
    lib_dir: str | os.PathLike[str] | None = None,
):
    """Open the single ARSDK-NG connection for the selected device/address."""
    address = str(ip).strip()
    if not address:
        raise ValueError("ARSDK device address must not be empty")
    selected = str(controller or "auto").strip().casefold().replace("-", "").replace("_", "")
    if selected == "auto":
        selected = "skycontroller3" if address == DRONE_IP_SKYCTRL else "drone"
    if selected in {"skycontroller3", "skyctrl3", "sc3", "skycontroller"}:
        device_type = ARSDK_DEVICE_SKYCTRL_3
        selected = "skycontroller3"
    elif selected in {"drone", "anafi", "direct", "anafi4k"}:
        device_type = ARSDK_DEVICE_ANAFI4K
        selected = "drone"
    else:
        raise ValueError(
            f"unsupported controller={controller!r}; use auto, drone, anafi, or skycontroller3"
        )

    interface_dir = _operator_interface_dir()
    if str(interface_dir) not in sys.path:
        sys.path.insert(0, str(interface_dir))
    from arsdk_ng import ArsdkNgClient, ArsdkNgError

    client = ArsdkNgClient(
        address,
        device_type=device_type,
        port=ARSDK_PORT,
        lib_dir=lib_dir,
    )
    try:
        client.wait_connected(float(timeout))
        # Discovery only opens the command interface; request the initial state
        # snapshot before readers rely on the event cache.
        if selected == "skycontroller3":
            queries = (
                client.messages.Skyctrl.Common.AllStates,
                client.messages.Skyctrl.Settings.AllSettings,
            )
        else:
            queries = ()
        queries += (
            client.messages.Common.Common.AllStates,
            client.messages.Common.Settings.AllSettings,
        )
        for query in queries:
            result = client(query()).wait(float(timeout))
            if not result.success():
                raise ArsdkNgError(f"{query.fullname}: {result.reason}")
    except BaseException:
        client.disconnect()
        raise
    return client


class RtspFrameGrabber:
    """Latest-only BGR-to-RGB RTSP reader with fail-closed health checks.

    ``__call__`` preserves the legacy ``(rgb, stamp) | None`` source contract;
    ``stamp`` is always a host-monotonic receipt time, never an OpenCV-derived
    source/capture timestamp. Callers that require a source timestamp must set
    ``require_source_timestamps=True``; because this path cannot provide one,
    such a grabber remains unhealthy and returns no frame.
    """

    def __init__(
        self,
        client: Any,
        resize: tuple[int, int] | None = (1280, 720),
        stale_s: float = 0.5,
        require_source_timestamps: bool = False,
        *,
        open_timeout_s: float = 10.0,
        read_timeout_ms: int = 1000,
    ) -> None:
        address = str(getattr(client, "address", "") or "").strip()
        if not address:
            raise ValueError("ARSDK client must expose its connected address")
        stale = float(stale_s)
        if not np.isfinite(stale) or stale <= 0:
            raise ValueError("stale_s must be finite and positive")
        if resize is not None:
            if len(resize) != 2 or int(resize[0]) <= 0 or int(resize[1]) <= 0:
                raise ValueError("resize must be None or a positive (width, height) pair")
            resize = (int(resize[0]), int(resize[1]))
        self.client = client
        self.address = address
        self.resize = resize
        self.stale_s = stale
        self.require_source_timestamps = bool(require_source_timestamps)
        self.open_timeout_s = max(0.1, float(open_timeout_s))
        self.read_timeout_ms = max(100, int(read_timeout_ms))
        self.rtsp_url = f"rtsp://{address}:{RTSP_PORT}{RTSP_PATH}"

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_timing: dict[str, Any] = {}
        self._stamp = 0.0
        self._receipt_stamp = 0.0
        self._n = 0
        self._started_at: float | None = None
        self._digest: bytes | None = None
        self._dup_n = 0
        self._frames_received = 0
        self._read_failures = 0
        self._decode_failures = 0
        self._source_width_px: int | None = None
        self._source_height_px: int | None = None
        self._output_width_px: int | None = None
        self._output_height_px: int | None = None
        self._capture_fourcc: str | None = None
        self._capture = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._opened = threading.Event()
        self._open_error: BaseException | None = None
        self._last_read_mono_ns: int | None = None

    def start(self):
        worker = self._worker
        if worker is not None and worker.is_alive():
            return self
        self._stop_event.clear()
        self._opened.clear()
        self._open_error = None
        worker = threading.Thread(
            target=self._reader_loop,
            name="parrot-rtsp-latest-frame",
            daemon=True,
        )
        self._worker = worker
        worker.start()
        if not self._opened.wait(self.open_timeout_s):
            self.stop()
            raise TimeoutError(f"RTSP stream did not open within {self.open_timeout_s:.1f}s")
        if self._open_error is not None:
            error = self._open_error
            self.stop()
            raise RuntimeError(f"could not open Parrot RTSP stream {self.rtsp_url}: {error}") from error
        return self

    def _open_capture(self, cv2):
        capture = None
        cap_ffmpeg = getattr(cv2, "CAP_FFMPEG", 0)
        open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_timeout is not None and read_timeout is not None:
            try:
                capture = cv2.VideoCapture(
                    self.rtsp_url,
                    cap_ffmpeg,
                    [
                        int(open_timeout), int(self.open_timeout_s * 1000),
                        int(read_timeout), self.read_timeout_ms,
                    ],
                )
            except (TypeError, ValueError, cv2.error):
                capture = None
        if capture is None:
            capture = cv2.VideoCapture(self.rtsp_url, cap_ffmpeg)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except (AttributeError, cv2.error):
            pass
        return capture

    def _reader_loop(self) -> None:
        import cv2

        capture = None
        try:
            capture = self._open_capture(cv2)
            self._capture = capture
            if not capture.isOpened():
                self._open_error = RuntimeError("OpenCV/FFmpeg rejected the RTSP URI")
                self._opened.set()
                return
            try:
                fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
                fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ")
                self._capture_fourcc = fourcc or None
            except (AttributeError, TypeError, ValueError, cv2.error):
                self._capture_fourcc = None
            self._started_at = time.monotonic()
            self._opened.set()

            while not self._stop_event.is_set():
                ok, bgr = capture.read()
                receipt_mono_ns = time.monotonic_ns()
                if not ok or bgr is None or getattr(bgr, "ndim", 0) != 3 or bgr.shape[2] != 3:
                    with self._lock:
                        self._read_failures += 1
                    if self._stop_event.wait(0.02):
                        break
                    continue
                try:
                    preprocess_start_mono_ns = time.monotonic_ns()
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    source_height, source_width = rgb.shape[:2]
                    if self.resize is not None and (source_width, source_height) != self.resize:
                        rgb = cv2.resize(rgb, self.resize)
                    preprocess_done_mono_ns = time.monotonic_ns()
                    timing = {
                        "frame_receipt_mono_ns": int(receipt_mono_ns),
                        "frame_preprocess_start_mono_ns": int(preprocess_start_mono_ns),
                        "frame_preprocess_done_mono_ns": int(preprocess_done_mono_ns),
                        "frame_store_mono_ns": int(time.monotonic_ns()),
                        "frame_stamp_mono": receipt_mono_ns * 1e-9,
                        "source_timestamp_kind": RTSP_TIMESTAMP_KIND,
                        "source_timestamp_degraded": True,
                        "source_timestamp_trusted": False,
                        "source_ntp_timestamp_us": None,
                    }
                except Exception:
                    with self._lock:
                        self._decode_failures += 1
                    continue
                self._store(
                    rgb,
                    receipt_stamp=receipt_mono_ns * 1e-9,
                    receipt_mono_ns=receipt_mono_ns,
                    timing=timing,
                    source_shape=(int(source_width), int(source_height)),
                )
        except BaseException as exc:
            if not self._opened.is_set():
                self._open_error = exc
                self._opened.set()
            elif not self._stop_event.is_set():
                with self._lock:
                    self._read_failures += 1
        finally:
            self._opened.set()
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            self._capture = None

    def _store(
        self,
        rgb: np.ndarray,
        *,
        receipt_stamp: float | None = None,
        receipt_mono_ns: int | None = None,
        timing: dict[str, Any] | None = None,
        source_shape: tuple[int, int] | None = None,
    ) -> None:
        receipt = time.monotonic() if receipt_stamp is None else float(receipt_stamp)
        now = time.monotonic()
        if not np.isfinite(receipt) or receipt > now:
            return
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            return
        digest = rgb[::64, ::64].tobytes()
        store_ns = time.monotonic_ns()
        timing_copy = dict(timing or {})
        timing_copy.setdefault(
            "frame_receipt_mono_ns",
            int(receipt_mono_ns if receipt_mono_ns is not None else round(receipt * 1e9)),
        )
        timing_copy["frame_store_mono_ns"] = store_ns
        timing_copy["frame_stamp_mono"] = receipt
        timing_copy["source_timestamp_kind"] = RTSP_TIMESTAMP_KIND
        timing_copy["source_timestamp_degraded"] = True
        timing_copy["source_timestamp_trusted"] = False
        timing_copy["source_ntp_timestamp_us"] = None
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        with self._lock:
            if self._latest is not None and receipt < self._stamp:
                return
            self._frames_received += 1
            if digest == self._digest:
                self._dup_n += 1
            else:
                self._digest = digest
                self._dup_n = 0
            self._latest = rgb
            self._stamp = receipt
            self._receipt_stamp = receipt
            self._latest_timing = timing_copy
            self._n += 1
            self._last_read_mono_ns = timing_copy["frame_receipt_mono_ns"]
            self._output_width_px = width
            self._output_height_px = height
            if source_shape is None:
                self._source_width_px, self._source_height_px = width, height
            else:
                self._source_width_px, self._source_height_px = source_shape

    def _usable_locked(self, now: float | None = None) -> bool:
        if self._latest is None or self._dup_n >= FROZEN_DUP_FRAMES:
            return False
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now) or now - self._stamp > self.stale_s:
            return False
        return not self.require_source_timestamps

    def __call__(self):
        with self._lock:
            if not self._usable_locked():
                return None
            return self._latest.copy(), float(self._stamp)

    def latest_frame_with_timing(self):
        """Return the immutable latest RGB frame, receipt stamp, and metadata."""
        with self._lock:
            if not self._usable_locked():
                return None
            return self._latest, float(self._stamp), dict(self._latest_timing)

    def peek_stamp(self) -> float | None:
        with self._lock:
            if not self._usable_locked():
                return None
            return float(self._stamp)

    def last_frame_age(self) -> float | None:
        with self._lock:
            if self._latest is None:
                return None
            return max(0.0, time.monotonic() - self._stamp)

    def is_healthy(self) -> bool:
        with self._lock:
            return self._usable_locked()

    @property
    def timestamp_degraded(self) -> bool:
        return True

    @property
    def stamp_source(self) -> str:
        return RTSP_TIMESTAMP_KIND

    def latest_source_ntp_us(self) -> None:
        """RTSP/OpenCV does not expose a trusted source clock."""
        return None

    def inspection_sample_after(self, *_args, **_kwargs):
        """Fail closed: an RTSP receipt time cannot satisfy a source-clock drain."""
        return None

    @property
    def fps(self) -> float:
        with self._lock:
            started_at = self._started_at
            count = self._n
        elapsed = time.monotonic() - started_at if started_at is not None else 0.0
        return count / elapsed if elapsed > 0 else 0.0

    @property
    def stream_metadata(self) -> dict[str, Any]:
        """Describe the RTSP stream and explicitly degraded receipt timestamps."""
        with self._lock:
            started_at = self._started_at
            elapsed = time.monotonic() - started_at if started_at is not None else 0.0
            observed_fps = self._n / elapsed if elapsed > 0 else 0.0
            return {
                "transport": "RTSP",
                "rtsp_uri": self.rtsp_url,
                "rtsp_path": RTSP_PATH,
                "codec": self._capture_fourcc,
                "codec_observed": self._capture_fourcc is not None,
                "source_width_px": self._source_width_px,
                "source_height_px": self._source_height_px,
                "output_width_px": self._output_width_px,
                "output_height_px": self._output_height_px,
                "observed_fps": float(observed_fps),
                "source_timestamp_capable": False,
                "source_timestamp_trusted": False,
                "timestamp_degraded": True,
                "stamp_source": RTSP_TIMESTAMP_KIND,
                "latest_age_s": (
                    max(0.0, time.monotonic() - self._stamp) if self._latest is not None else None
                ),
                "healthy": self._usable_locked(),
                "frozen_duplicate_count": self._dup_n,
            }

    @property
    def frame_pipeline_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "received": self._frames_received,
                "read_failures": self._read_failures,
                "decode_failures": self._decode_failures,
                "duplicate_run": self._dup_n,
                "frozen": self._dup_n >= FROZEN_DUP_FRAMES,
                "stamp_source": RTSP_TIMESTAMP_KIND,
                "latest_age_ms": (
                    max(0.0, time.monotonic() - self._stamp) * 1000.0
                    if self._latest is not None else None
                ),
            }

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(2.0, self.read_timeout_ms / 1000.0 + 1.0))
            if worker.is_alive():
                capture = self._capture
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass
                worker.join(timeout=1.0)
        if worker is not None and not worker.is_alive():
            self._worker = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.stop()
 
def run_real(drone, loc, det, ctrl):
    """Keep the retired legacy takeoff path permanently locked."""
    raise SystemExit(
        "rtsp_frame_source.run_real legacy TakeOff is permanently locked; "
        "use the operator-approved flight path instead"
    )
