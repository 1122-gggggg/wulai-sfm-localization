"""Opt-in flight recorder for judging whether IMU-aided localization is viable.

Turned on by ``SFM_IMU_FLIGHT_TEST=1``; the ``IMU飛行測試.sh`` launcher sets it.
Off by default, so no ordinary flight pays for any of this.

Why it exists
-------------
``docs/esekf_live_eval_runbook.md`` sends the operator out with the onboard SD
recording and then replays that MP4 offline against the session
``telemetry.jsonl``. That works, but the two clocks are independent: the runbook
has a whole section on hand-tuning ``--telemetry-offset-s`` until the velocity
actually lands on the right frames, and a mis-set offset shows up as a
``DORMANT``/``INVALID`` verdict rather than as an error.

This recorder writes the frames the localizer *actually saw*, each one already
carrying the fused telemetry that was attached to that same submission, on the
one host monotonic clock. Nothing to align afterwards: row N of ``frames.jsonl``
and ``frames/<seq>.jpg`` are the exact (image, IMU) pair the tracker was fed.

Cost and safety
---------------
The UI thread only copies the frame and drops it in a bounded queue; JPEG
encoding and disk writes happen on one daemon writer thread. A full queue drops
the frame and counts it -- recording must never stall the flight UI, and a gap
in the recording is a recoverable data problem while a stalled UI is not.
Frame and byte caps stop the recorder rather than fill the disk the session
logs and the safety fail-closed path depend on.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from disk_policy import assess_disk_space

#: Copied verbatim out of the submission's timing metadata into every frame row.
#: Same keys ``attach_fused_localization_telemetry`` puts on the wire, so a row
#: says exactly what the worker received for that image -- including ``None``
#: for a field the aircraft did not report, which is itself the finding.
FUSED_FIELDS = (
    "fused_telemetry_mono",
    "fused_stamp_source",
    "fused_attitude_mono",
    "fused_speed_mono",
    "fused_altitude_mono",
    "fused_roll",
    "fused_pitch",
    "fused_yaw",
    "fused_speed_north",
    "fused_speed_east",
    "fused_speed_down",
    "fused_gps_mono",
    "fused_gps_latitude",
    "fused_gps_longitude",
    "fused_gps_altitude",
    "fused_gps_latitude_accuracy",
    "fused_gps_longitude_accuracy",
    "fused_gps_altitude_accuracy",
)

DEFAULT_MAX_FRAMES = 20000
DEFAULT_MAX_MEGABYTES = 2048
DEFAULT_JPEG_QUALITY = 90
#: Three in flight plus one being encoded. Deeper only buys latency: at 2.8 MB
#: per 720p frame the queue is the recorder's whole memory footprint.
_QUEUE_DEPTH = 4
_INDEX_SYNC_EVERY = 16
#: How often the writer re-checks free space. The frame cap alone is not a disk
#: guard: it is sized in frames, and the deployment laptop has run under 10 GiB
#: free. Checked on the writer thread only, so the UI never pays for the statvfs.
_DISK_CHECK_EVERY = 32


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, value)


def imu_flight_test_enabled() -> bool:
    return os.environ.get("SFM_IMU_FLIGHT_TEST") == "1"


def _as_rgb_array(frame: Any) -> np.ndarray | None:
    """Own a contiguous RGB copy of a frame, or None if it is not an image.

    The grabber reuses its buffer, so the copy is required, not an optimization.
    Both frame types the UI can hold are accepted: the live ANAFI grabber hands
    over a numpy RGB array, the simulated stream a PIL image.
    """
    if isinstance(frame, Image.Image):
        frame = np.asarray(frame.convert("RGB"))
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    try:
        # ``ascontiguousarray`` returns its input when it is already contiguous.
        # The live grabber reuses that buffer, so this recorder must always own
        # an independent copy before handing it to the writer thread.
        owned: np.ndarray = np.array(
            frame[..., :3], dtype=np.uint8, order="C", copy=True
        )
    except (TypeError, ValueError):
        return None
    return owned


def _encode_jpeg(frame_rgb: np.ndarray, quality: int) -> bytes | None:
    try:
        import cv2

        ok, buffer = cv2.imencode(
            ".jpg",
            frame_rgb[..., ::-1],
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        return bytes(buffer) if ok else None
    except Exception:
        return None


class ImuFlightTestRecorder:
    """Write the localizer's own frames plus their fused telemetry to disk."""

    def __init__(
        self,
        directory: Path,
        *,
        max_frames: int | None = None,
        max_megabytes: int | None = None,
        jpeg_quality: int | None = None,
        every_n: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.frames_directory = self.directory / "frames"
        self.frames_directory.mkdir(parents=True, exist_ok=True)
        self.max_frames = (
            _env_int("SFM_IMU_TEST_MAX_FRAMES", DEFAULT_MAX_FRAMES)
            if max_frames is None
            else int(max_frames)
        )
        self.max_bytes = (
            _env_int("SFM_IMU_TEST_MAX_MB", DEFAULT_MAX_MEGABYTES)
            if max_megabytes is None
            else int(max_megabytes)
        ) * 1024 * 1024
        self.jpeg_quality = (
            _env_int("SFM_IMU_TEST_JPEG_QUALITY", DEFAULT_JPEG_QUALITY)
            if jpeg_quality is None
            else int(jpeg_quality)
        )
        self.every_n = (
            _env_int("SFM_IMU_TEST_EVERY_N", 1) if every_n is None else max(1, int(every_n))
        )

        self.captured = 0
        self.written = 0
        self.dropped_queue_full = 0
        self.dropped_encode_failed = 0
        self.skipped_stride = 0
        self.skipped_duplicate = 0
        self.bytes_written = 0
        self._file_index = 0
        self._last_seq: int | None = None
        self.stop_reason = ""
        self.last_error = ""
        self._state_lock = threading.Lock()
        self._summary_lock = threading.Lock()
        self._writer_finished = threading.Event()
        self._incomplete = False
        self._writer_shutdown_timeout = False
        self._writer_error = ""
        self._summary_revision = 0

        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._closed = threading.Event()
        self._index = (self.directory / "frames.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self._index_pending = 0
        self._since_disk_check = 0
        self._thread = threading.Thread(
            target=self._run, name="imu-flight-test-writer", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ input
    def capture(
        self,
        seq: int,
        frame_name: str,
        frame: Any,
        timing_metadata: dict[str, Any] | None,
    ) -> bool:
        """Queue one submitted frame. Never blocks and never raises."""
        if self._closed.is_set() or self.stop_reason:
            return False
        image = _as_rgb_array(frame)
        if image is None:
            return False
        # BOOT and LOST hold resubmit the same frozen image until they recover.
        # One copy of it is the whole picture, and storing every retry used to
        # burn the frame cap on a single held frame; each retry's own telemetry
        # is still its own row in localization.jsonl.
        if self._last_seq is not None and int(seq) == self._last_seq:
            self.skipped_duplicate += 1
            return False
        self._last_seq = int(seq)
        self.captured += 1
        if (self.captured - 1) % self.every_n:
            self.skipped_stride += 1
            return False
        if self.written >= self.max_frames:
            self._stop("max_frames")
            return False
        if self.bytes_written >= self.max_bytes:
            self._stop("max_bytes")
            return False

        timing = timing_metadata or {}
        # Named by the recorder's own counter, not the stream index: the stream
        # index repeats across a hold and repeated names silently overwrote the
        # previous image. Gaps here mean dropped frames, which is worth seeing.
        self._file_index += 1
        row = {
            "seq": int(seq),
            "frame_name": str(frame_name),
            "file": f"frames/{self._file_index:06d}.jpg",
            "capture_stamp_mono": timing.get("source_frame_stamp_mono"),
            "enqueue_mono": time.monotonic(),
            "hold_kind": timing.get("hold_kind"),
        }
        row.update((field, timing.get(field)) for field in FUSED_FIELDS)
        try:
            self._queue.put_nowait((row, image))
        except queue.Full:
            self.dropped_queue_full += 1
            self._incomplete = True
            return False
        return True

    # ----------------------------------------------------------------- writer
    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._closed.is_set():
                    break
                continue
            if item is None:
                break
            try:
                self._write(*item)
            except Exception as exc:
                self._record_writer_error(exc)
        try:
            self._flush_index()
        finally:
            # The writer is the sole owner of the index handle. close() may
            # return while this thread is still blocked on an encoder or disk,
            # so the caller must never close the handle behind it.
            self._close_index()
            with self._summary_lock:
                while True:
                    with self._state_lock:
                        revision = self._summary_revision
                    self._write_summary_locked(writer_alive=False)
                    with self._state_lock:
                        if self._summary_revision == revision:
                            self._writer_finished.set()
                            break

    def _write(self, row: dict[str, Any], image: np.ndarray) -> None:
        payload = _encode_jpeg(image, self.jpeg_quality)
        if payload is None:
            self.dropped_encode_failed += 1
            self._incomplete = True
            return
        (self.directory / str(row["file"])).write_bytes(payload)
        row["bytes"] = len(payload)
        row["written_mono"] = time.monotonic()
        self._index.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._index_pending += 1
        if self._index_pending >= _INDEX_SYNC_EVERY:
            self._flush_index()
        self.written += 1
        self.bytes_written += len(payload)
        self._since_disk_check += 1
        if self._since_disk_check >= _DISK_CHECK_EVERY:
            self._since_disk_check = 0
            self._check_disk()

    def _check_disk(self) -> None:
        """Stop before the recording eats the flight-safety disk margin.

        Below the documented critical floor the backend blocks takeoff and the
        session logs the safety path depends on are at risk. A truncated
        recording is recoverable; losing those is not.
        """
        try:
            status = assess_disk_space(self.directory)
        except Exception as exc:
            error = repr(exc)
            with self._state_lock:
                self.last_error = error
                self._writer_error = error
                self._incomplete = True
            return
        if status.takeoff_blocked:
            self._stop(f"disk_floor:{status.reason}")

    def _flush_index(self) -> None:
        try:
            self._index.flush()
            os.fsync(self._index.fileno())
            self._index_pending = 0
        except Exception as exc:
            self._record_writer_error(exc, reason="writer_flush_error")

    def _close_index(self) -> None:
        try:
            if not self._index.closed:
                self._index.close()
        except Exception as exc:
            self._record_writer_error(exc, reason="writer_close_error")

    def _record_writer_error(self, exc: BaseException, *, reason: str = "writer_error") -> None:
        error = repr(exc)
        with self._state_lock:
            self.last_error = error
            self._writer_error = error
            self._incomplete = True
            if not self.stop_reason:
                self.stop_reason = reason

    def _stop(self, reason: str) -> None:
        if not self.stop_reason:
            self.stop_reason = reason

    def _summary_payload(self, *, writer_alive: bool) -> dict[str, Any]:
        with self._state_lock:
            incomplete = bool(self._incomplete or writer_alive or self._writer_shutdown_timeout)
            writer_error = self._writer_error or None
            shutdown_timeout = bool(self._writer_shutdown_timeout)
        return {
            "schema_version": 1,
            "closed_mono": time.monotonic(),
            "incomplete": incomplete,
            "writer_alive": bool(writer_alive),
            "writer_shutdown_timeout": shutdown_timeout,
            "writer_error": writer_error,
            **self.stats(),
        }

    def _write_summary_locked(self, *, writer_alive: bool) -> None:
        summary = self._summary_payload(writer_alive=writer_alive)
        try:
            temp = self.directory / ".summary.json.tmp"
            temp.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, self.directory / "summary.json")
        except Exception as exc:
            # There may be no usable path left for a second diagnostic, but keep
            # the in-memory error for callers and for any later retry.
            self._record_writer_error(exc, reason="summary_write_error")

    # ------------------------------------------------------------------ close
    def stats(self) -> dict[str, Any]:
        return {
            "captured": self.captured,
            "written": self.written,
            "skipped_stride": self.skipped_stride,
            "skipped_duplicate": self.skipped_duplicate,
            "dropped_queue_full": self.dropped_queue_full,
            "dropped_encode_failed": self.dropped_encode_failed,
            "bytes_written": self.bytes_written,
            "every_n": self.every_n,
            "jpeg_quality": self.jpeg_quality,
            "max_frames": self.max_frames,
            "max_bytes": self.max_bytes,
            "stop_reason": self.stop_reason,
            "last_error": self.last_error,
        }

    def close(self, *, timeout: float = 5.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        wait_s = max(0.0, float(timeout))
        self._thread.join(timeout=wait_s)
        with self._state_lock:
            writer_alive = not self._writer_finished.is_set()
            if writer_alive:
                self._writer_shutdown_timeout = True
                self._incomplete = True
                self._summary_revision += 1
        if writer_alive and self._summary_lock.acquire(blocking=False):
            try:
                # A timed-out writer may still own the index. Publish a truthful
                # provisional summary only when the summary sink is available;
                # the writer's revision check prevents an older snapshot from
                # overwriting the timeout evidence.
                with self._state_lock:
                    still_alive = not self._writer_finished.is_set()
                if still_alive:
                    self._write_summary_locked(writer_alive=True)
            finally:
                self._summary_lock.release()


def create_recorder(
    session_directory: Path | None,
    *,
    note: Any = None,
) -> ImuFlightTestRecorder | None:
    """Build a recorder for this session, or None when the mode is off."""
    if session_directory is None or not imu_flight_test_enabled():
        return None
    try:
        recorder = ImuFlightTestRecorder(Path(session_directory) / "imu_test")
    except Exception:
        return None
    # The UI closes the recorder on a clean shutdown; this covers Ctrl-C and an
    # unhandled exit, where the queued tail and the summary would be lost.
    # close() is idempotent, so the normal path still wins the race harmlessly.
    atexit.register(recorder.close)
    if callable(note):
        note(f"IMU 飛行測試錄製 -> {recorder.directory}")
    return recorder


def close_recorder(recorder: Any) -> None:
    """Finalize a recorder if there is one. Never raises into the flight UI."""
    if recorder is None:
        return
    try:
        recorder.close()
    except (OSError, ValueError, AttributeError, TypeError, RuntimeError):
        pass


def capture_frame(
    recorder: Any,
    seq: int,
    frame_name: str,
    frame: Any,
    timing_metadata: dict[str, Any] | None,
) -> None:
    """Keep the exact image the tracker was handed, with its own telemetry."""
    if recorder is not None:
        recorder.capture(seq, frame_name, frame, timing_metadata)


__all__ = [
    "FUSED_FIELDS",
    "ImuFlightTestRecorder",
    "capture_frame",
    "close_recorder",
    "create_recorder",
    "imu_flight_test_enabled",
]
