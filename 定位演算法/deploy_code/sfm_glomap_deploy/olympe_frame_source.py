#!/usr/bin/env python3
"""Live ANAFI PDRAW frame grabber -> a reloc_localizer frame_source().

Swaps the sim's SplatFrameSource for the REAL drone with ZERO change to the
localizer or controller: both consume the same `frame_source() -> (HxWx3 uint8
RGB, frame_monotonic_stamp) | None` contract. Same map M, same ALIKED+LightGlue+PnP, same scale-free
controller -- that is the whole point of testing through one interface.

  sim :  RelocLocalizer(M, SplatFrameSource(sim, renderer, cam), qcam)
  real:  RelocLocalizer(M, OlympePdrawGrabber(drone),           qcam)

ONE connection rule: ANAFI accepts a single controller connection, and Olympe
streams video over that same connection. So the grabber attaches to an
ALREADY-CONNECTED `olympe.Drone`, and PCMD control must reuse THAT drone (not a
second one). `run_real()` below does exactly that, mirroring autoflight.run()'s
loop on the shared drone -- so you never open two connections and never edit
autoflight.py.

Live stream IPs:  real ANAFI 192.168.42.1 ; SkyController 192.168.53.1 ;
                  simulated (Sphinx) drone 10.202.0.1.
720p live stream matches the sc1_AB_720 query intrinsics used by the sim.

Deps: olympe (Parrot Ground SDK) + cv2 (already used by streaming_localize.py).
The YUV format enum / frame layout is the one version-sensitive bit -> `# VERIFY`.

!!! SAFETY: real flight is irreversible and can injure people or destroy the
drone. Validate the FULL stack in sim (sim_harness) first. Then bench-test this
grabber with PROPS OFF (--selftest-live just grabs video, never arms motors).
Only fly run_real() in open space, with a human holding the manual-override
controller, after verifying PCMD signs on YOUR airframe. Speeds are tiny. !!!
"""
from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np

DRONE_IP_REAL = "192.168.42.1"
DRONE_IP_SKYCTRL = "192.168.53.1"
DRONE_IP_SIM = "10.202.0.1"

# A stuck decoder that keeps re-delivering the SAME frame defeats the staleness
# check (fresh timestamps, frozen picture): localization returns a constant pose
# while the drone physically drifts. Real decoded video has sensor noise, so
# byte-identical consecutive frames only happen on a pipeline freeze/duplication.
FROZEN_DUP_FRAMES = 15          # consecutive identical frames -> stream unhealthy (~0.5s @30fps)


# ----------------------------- connection ----------------------------------
def connect(ip: str = DRONE_IP_REAL, controller: str = "auto"):
    """Open + connect one Olympe controller.

    Olympe 8.4 API rule:
      - direct ANAFI connection uses olympe.Drone / olympe.Anafi;
      - ANAFI through SkyController 3 uses olympe.SkyController3.

    Reuse this one controller object for BOTH video streaming and PCMD control.
    """
    import olympe
    ctrl = str(controller or "auto").lower()
    if ctrl == "auto":
        ctrl = "skycontroller3" if str(ip) == DRONE_IP_SKYCTRL else "drone"
    if ctrl in {"skycontroller3", "skyctrl3", "sc3"}:
        klass = getattr(olympe, "SkyController3")
    elif ctrl in {"drone", "anafi", "direct"}:
        klass = getattr(olympe, "Anafi", None) or getattr(olympe, "Drone")
    else:
        raise SystemExit(f"unsupported Olympe controller={controller}; use auto, drone, anafi, skycontroller3")
    drone = klass(ip)
    if not drone.connect():
        raise SystemExit(f"could not connect to {ctrl} at {ip}")
    return drone


def _wait_success(expectation, label: str):
    res = expectation.wait()
    if hasattr(res, "success") and not res.success():
        raise SystemExit(f"{label} failed or timed out")


# ----------------------------- grabber -------------------------------------
class OlympePdrawGrabber:
    """Latest-frame PDRAW grabber. __call__() returns the most recent RGB frame
    and its capture-time monotonic stamp if fresher than `stale_s`, else None.
    The localizer must preserve this stamp on Pose so inference latency cannot
    make an old image look fresh to the flight controller."""

    def __init__(self, drone, resize: tuple | None = (1280, 720), stale_s: float = 0.5,
                 media_name: str = "Front camera",
                 require_source_timestamps: bool = False):
        self.drone = drone
        self.resize = resize
        self.stale_s = stale_s
        self.media_name = media_name
        self.require_source_timestamps = bool(require_source_timestamps)
        self._lock = threading.Lock()
        self._frame_cv = threading.Condition(self._lock)
        self._latest = None          # RGB uint8 HxWx3
        self._latest_timing = {}     # host monotonic_ns pipeline markers
        self._stamp = 0.0
        self._n = 0                  # frames received (for fps)
        self._digest = None          # sparse pixel sample of the last stored frame
        self._dup_n = 0              # consecutive identical frames (freeze detector)
        self._frozen_warned = False
        self._t0 = None
        self._cvt = None             # lazy: {olympe format -> cv2 code}
        self._default_code = None
        self._decode_warned = False
        self._stream_mode = None
        self._source_ntp_us = None
        self._source_to_monotonic = None
        self._stamp_source = "none"
        self._stored_source_ntp_us = None
        self._receipt_stamp = 0.0
        self._timestamp_fallbacks = 0
        self._pending_yuv = None
        self._frame_worker = None
        self._frame_worker_stop = False
        self._accept_frames = False
        self._flush_depth = 0
        self._frames_received = 0
        self._frames_enqueued = 0
        self._frames_converted = 0
        self._queue_drops = 0
        self._flush_drops = 0
        self._stopped_drops = 0
        self._timestamp_drops = 0
        self._decode_failures = 0
        self._convert_attempts = 0
        self._convert_total_s = 0.0
        self._convert_last_s = 0.0
        self._convert_max_s = 0.0
        self._convert_inflight = 0

    # ---- lifecycle ----
    def start(self):
        # Olympe >=8.4 documented API: drone.streaming.play(raw_cb=...).
        # Older Olympe builds exposed set_callbacks()+start(); keep it as a
        # compatibility fallback because this workstation has older scripts.
        self._start_frame_worker()
        try:
            if hasattr(self.drone.streaming, "play"):
                try:
                    self._start_with_play()
                    self._stream_mode = "play"
                except Exception as exc:
                    print(f"[olympe_frame_source] streaming.play failed ({exc}); fallback to start API")
                    self._start_with_legacy_callbacks()
                    self._stream_mode = "legacy"
            else:
                self._start_with_legacy_callbacks()
                self._stream_mode = "legacy"
        except Exception:
            self._stop_frame_worker()
            raise
        self._t0 = time.monotonic()
        return self

    def _start_with_play(self):
        ok = self.drone.streaming.play(
            media_name=self.media_name,
            raw_cb=self._yuv_cb,
            flush_raw_cb=self._flush_cb,
            has_renderer=False,
        )
        if ok is False:
            raise RuntimeError("streaming.play returned False")

    def _start_with_legacy_callbacks(self):
        self.drone.streaming.set_callbacks(
            raw_cb=self._yuv_cb, h264_cb=None, start_cb=None, end_cb=None,
            flush_raw_cb=self._flush_cb)
        self.drone.streaming.start()

    def _start_frame_worker(self):
        with self._frame_cv:
            worker = self._frame_worker
            if worker is not None and worker.is_alive():
                self._accept_frames = True
                return
            self._frame_worker_stop = False
            self._accept_frames = True
            worker = threading.Thread(
                target=self._frame_worker_loop,
                name="olympe-yuv-latest",
                daemon=True,
            )
            self._frame_worker = worker
            worker.start()

    def _pause_frame_input(self):
        pending = None
        with self._frame_cv:
            self._accept_frames = False
            pending = self._pending_yuv
            self._pending_yuv = None
            if pending is not None:
                self._stopped_drops += 1
            self._frame_cv.notify_all()
        if pending is not None:
            pending[0].unref()

    def _stop_frame_worker(self):
        self._pause_frame_input()
        with self._frame_cv:
            self._frame_worker_stop = True
            worker = self._frame_worker
            self._frame_cv.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        with self._frame_cv:
            if self._frame_worker is worker:
                self._frame_worker = None

    def stop(self):
        self._pause_frame_input()
        try:
            self.drone.streaming.stop()
        except Exception:
            pass
        finally:
            self._stop_frame_worker()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- olympe callbacks (run on olympe's thread) ----
    def _formats(self):
        if self._cvt is None:
            import cv2, olympe
            # VERIFY against your olympe version: I420 / NV12 are the usual yuv
            # raw formats for the ANAFI live stream.
            cvt = {}
            for nm, code in (("VDEF_I420", cv2.COLOR_YUV2RGB_I420),
                             ("VDEF_NV12", cv2.COLOR_YUV2RGB_NV12)):
                if hasattr(olympe, nm):
                    cvt[getattr(olympe, nm)] = code
            # publish fully-built maps under the lock (callback may run on another thread)
            with self._lock:
                self._default_code = cv2.COLOR_YUV2RGB_I420
                self._cvt = cvt
        return self._cvt

    def _yuv_cb(self, yuv_frame):
        # Pdraw thread: ref + replace the single pending slot, then return.
        # Metadata access, ndarray conversion, resize, inference and logging all
        # stay on downstream workers so callback pressure cannot build a queue.
        receipt_mono_ns = time.monotonic_ns()
        try:
            yuv_frame.ref()
        except Exception:
            return
        keep_ref = False
        replaced = None
        try:
            with self._frame_cv:
                self._frames_received += 1
                if not self._accept_frames:
                    self._stopped_drops += 1
                    return
                if self._flush_depth:
                    self._flush_drops += 1
                    return
                replaced = self._pending_yuv
                self._pending_yuv = (yuv_frame, receipt_mono_ns)
                self._frames_enqueued += 1
                if replaced is not None:
                    self._queue_drops += 1
                keep_ref = True
                self._frame_cv.notify()
        finally:
            if replaced is not None:
                replaced[0].unref()
            if not keep_ref:
                yuv_frame.unref()

    def _convert_yuv_frame(self, yuv_frame, receipt_stamp: float,
                           capture_stamp: float, stamp_source: str,
                           source_ntp_us: int | None,
                           callback_mono_ns: int | None = None) -> bool:
        import cv2
        preprocess_start_mono_ns = time.monotonic_ns()
        arr = yuv_frame.as_ndarray()              # full YUV plane (H*3/2, W)
        yuv_ready_mono_ns = time.monotonic_ns()
        if arr is None:
            return False
        code = self._formats().get(yuv_frame.format(), self._default_code)
        rgb = cv2.cvtColor(arr, code)             # -> RGB HxWx3 uint8
        if self.resize is not None and (rgb.shape[1], rgb.shape[0]) != self.resize:
            rgb = cv2.resize(rgb, self.resize)
        preprocess_done_mono_ns = time.monotonic_ns()
        self._store(
            rgb, stamp=capture_stamp, stamp_source=stamp_source,
            source_ntp_us=source_ntp_us, receipt_stamp=receipt_stamp,
            timing={
                "frame_callback_enter_mono_ns": int(
                    callback_mono_ns
                    if callback_mono_ns is not None
                    else round(float(receipt_stamp) * 1_000_000_000)
                ),
                "frame_preprocess_start_mono_ns": preprocess_start_mono_ns,
                "frame_yuv_ready_mono_ns": yuv_ready_mono_ns,
                "frame_preprocess_done_mono_ns": preprocess_done_mono_ns,
            })
        return True

    def _frame_worker_loop(self):
        while True:
            with self._frame_cv:
                while self._pending_yuv is None and not self._frame_worker_stop:
                    self._frame_cv.wait()
                if self._frame_worker_stop:
                    return
                pending = self._pending_yuv
                self._pending_yuv = None
                self._convert_inflight += 1

            started = time.perf_counter()
            converted = False
            failed = False
            try:
                yuv_frame, receipt_mono_ns = pending
                receipt_stamp = int(receipt_mono_ns) * 1e-9
                # VideoFrame.info() can be non-trivial. Read it here, never on
                # Pdraw's callback thread.
                capture_stamp, stamp_source, source_ntp_us = self._capture_stamp(
                    yuv_frame, receipt_stamp)
                if capture_stamp is None:
                    with self._lock:
                        self._timestamp_drops += 1
                if capture_stamp is not None:
                    converted = self._convert_yuv_frame(
                        yuv_frame, receipt_stamp, capture_stamp,
                        stamp_source, source_ntp_us, receipt_mono_ns)
                    failed = not converted
            except Exception as exc:
                failed = True
                # A single bad frame must NOT kill the worker. Sustained failure
                # makes the stored frame stale, so the controller HOVERs.
                if not self._decode_warned:
                    self._decode_warned = True
                    print(f"[olympe_frame_source] frame decode failed ({exc!r}); "
                          "dropping frame(s), stream stays alive "
                          "(controller hovers on staleness)", flush=True)
            finally:
                elapsed = time.perf_counter() - started
                try:
                    pending[0].unref()
                finally:
                    with self._frame_cv:
                        self._convert_attempts += 1
                        self._convert_total_s += elapsed
                        self._convert_last_s = elapsed
                        self._convert_max_s = max(self._convert_max_s, elapsed)
                        if converted:
                            self._frames_converted += 1
                        if failed:
                            self._decode_failures += 1
                        self._convert_inflight -= 1
                        self._frame_cv.notify_all()

    def _flush_cb(self, *_):
        pending = None
        with self._frame_cv:
            self._flush_depth += 1
            pending = self._pending_yuv
            self._pending_yuv = None
            if pending is not None:
                self._flush_drops += 1
            self._frame_cv.notify_all()
        if pending is not None:
            pending[0].unref()
        with self._frame_cv:
            while self._convert_inflight:
                self._frame_cv.wait()
            self._flush_depth -= 1
            self._frame_cv.notify_all()
        return True

    # ---- store / read (the frame_source contract) ----
    def _capture_stamp(self, yuv_frame, receipt_stamp: float):
        """Map Olympe's source NTP microseconds onto host monotonic time.

        Olympe 8.4's official streaming example uses
        ``VideoFrame.info()['ntp_raw_timestamp']`` in microseconds. The first
        valid frame anchors that clock to callback-entry monotonic time. Later
        frames preserve source deltas, exposing transport/decode backlog.
        Missing metadata falls back to the already-captured callback receipt
        time and is explicitly marked. Equal/out-of-order source frames are
        dropped; a large backward jump is treated as a stream-clock restart.
        """
        receipt = float(receipt_stamp)
        if not np.isfinite(receipt):
            return None, "invalid-receipt", None
        try:
            info = yuv_frame.info()
            source_us = int(info["ntp_raw_timestamp"])
            if source_us <= 0:
                raise ValueError("non-positive ntp_raw_timestamp")
        except Exception:
            self._timestamp_fallbacks = getattr(self, "_timestamp_fallbacks", 0) + 1
            return receipt, "callback-receipt", None

        last_us = getattr(self, "_source_ntp_us", None)
        offset = getattr(self, "_source_to_monotonic", None)
        if last_us is None or offset is None:
            self._source_ntp_us = source_us
            self._source_to_monotonic = receipt - source_us * 1e-6
            return receipt, "ntp-anchor", source_us
        if source_us == last_us:
            return None, "ntp-nonincreasing", source_us
        if source_us < last_us:
            if last_us - source_us < 1_000_000:
                return None, "ntp-nonincreasing", source_us
            self._source_ntp_us = source_us
            self._source_to_monotonic = receipt - source_us * 1e-6
            return receipt, "ntp-restart-receipt", source_us

        # Keep the smallest observed transport offset. It is the safest local
        # approximation of capture time and never makes a queued frame newer.
        observed_offset = receipt - source_us * 1e-6
        self._source_to_monotonic = min(float(offset), observed_offset)
        self._source_ntp_us = source_us
        mapped = min(receipt, source_us * 1e-6 + self._source_to_monotonic)
        return mapped, "ntp-mapped", source_us

    def _store(self, rgb: np.ndarray, stamp: float | None = None,
               stamp_source: str = "callback-receipt",
               source_ntp_us: int | None = None,
               receipt_stamp: float | None = None,
               timing: dict | None = None):
        stamp = time.monotonic() if stamp is None else float(stamp)
        receipt = time.monotonic() if receipt_stamp is None else float(receipt_stamp)
        if not np.isfinite(stamp) or stamp > time.monotonic() + 0.05:
            return
        if not np.isfinite(receipt):
            return
        try:
            source_us = None if source_ntp_us is None else int(source_ntp_us)
        except (TypeError, ValueError, OverflowError):
            source_us = None
        if source_us is not None and source_us <= 0:
            source_us = None
        digest = rgb[::64, ::64].tobytes()   # ~720B sample; identical only if the frame repeats
        stored_timing = dict(timing or {})
        stored_timing["frame_store_mono_ns"] = time.monotonic_ns()
        with self._lock:
            if digest == self._digest:
                self._dup_n += 1
                warn = self._dup_n == FROZEN_DUP_FRAMES and not self._frozen_warned
                if warn:
                    self._frozen_warned = True
            else:
                self._digest = digest
                self._dup_n = 0
                self._frozen_warned = False   # recovered; warn again on the next freeze
                warn = False
            self._latest = rgb
            self._stamp = stamp
            self._stamp_source = str(stamp_source)
            self._stored_source_ntp_us = source_us
            self._receipt_stamp = receipt
            self._latest_timing = stored_timing
            self._n += 1
        if warn:
            print(f"[olympe_frame_source] stream FROZEN: {FROZEN_DUP_FRAMES} identical "
                  "consecutive frames; treating as unhealthy (controller hovers)", flush=True)

    def __call__(self):
        with self._lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._stamp > self.stale_s:
                return None                            # stale -> controller HOVERs
            if self._dup_n >= FROZEN_DUP_FRAMES:
                return None                            # frozen -> controller HOVERs
            if (getattr(self, "require_source_timestamps", False)
                    and not self._source_timing_trusted_locked()):
                return None                            # degraded timing -> true-flight HOVER
            return self._latest.copy(), float(self._stamp)

    def latest_frame_with_timing(self):
        """Return the latest immutable RGB object and atomic timing snapshot.

        The worker never mutates a stored array after publication. The live UI
        may therefore serialize it without another full-frame copy; the legacy
        ``__call__`` contract keeps its defensive copy for flight callers.
        """
        with self._lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._stamp > self.stale_s:
                return None
            if self._dup_n >= FROZEN_DUP_FRAMES:
                return None
            if (getattr(self, "require_source_timestamps", False)
                    and not self._source_timing_trusted_locked()):
                return None
            return self._latest, float(self._stamp), dict(
                getattr(self, "_latest_timing", {}))

    def peek_stamp(self) -> float | None:
        """Return the available frame stamp without copying the image."""
        with self._lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._stamp > self.stale_s:
                return None
            if self._dup_n >= FROZEN_DUP_FRAMES:
                return None
            if (getattr(self, "require_source_timestamps", False)
                    and not self._source_timing_trusted_locked()):
                return None
            return float(self._stamp)

    def _source_timing_trusted_locked(self) -> bool:
        return (getattr(self, "_stamp_source", "none") == "ntp-mapped"
                and isinstance(getattr(self, "_stored_source_ntp_us", None), int)
                and self._stored_source_ntp_us > 0)

    def last_frame_age(self) -> float | None:
        with self._lock:
            if self._latest is None:
                return None
            return time.monotonic() - self._stamp

    def is_healthy(self) -> bool:
        with self._lock:
            if self._dup_n >= FROZEN_DUP_FRAMES:
                return False                           # frozen picture with fresh stamps
            if (getattr(self, "require_source_timestamps", False)
                    and not self._source_timing_trusted_locked()):
                return False
            if self._latest is None:
                return False
            return time.monotonic() - self._stamp <= self.stale_s

    @property
    def timestamp_degraded(self) -> bool:
        with self._lock:
            return not self._source_timing_trusted_locked()

    def latest_source_ntp_us(self) -> int | None:
        with self._lock:
            if not self._source_timing_trusted_locked():
                return None
            return int(self._stored_source_ntp_us)

    def inspection_sample_after(self, baseline_source_ntp_us: int,
                                min_source_advance_s: float,
                                receipt_after: float):
        """Return a frame only after both source-clock and host-receipt drains.

        Raw NTP is treated as a sequence clock, not as host-synchronized time.
        Requiring both conditions prevents a pre-confirmation frame already in
        the video pipeline from satisfying an inspection acknowledgement.
        """
        try:
            baseline = int(baseline_source_ntp_us)
            advance_us = int(math.ceil(float(min_source_advance_s) * 1_000_000.0))
            receipt_min = float(receipt_after)
        except (TypeError, ValueError, OverflowError):
            return None
        if baseline <= 0 or advance_us <= 0 or not np.isfinite(receipt_min):
            return None
        with self._lock:
            if (self._latest is None or self._dup_n >= FROZEN_DUP_FRAMES
                    or not self._source_timing_trusted_locked()
                    or time.monotonic() - self._stamp > self.stale_s
                    or self._stored_source_ntp_us < baseline + advance_us
                    or getattr(self, "_receipt_stamp", 0.0) < receipt_min):
                return None
            return self._latest.copy(), float(self._stamp)

    @property
    def stamp_source(self) -> str:
        with self._lock:
            return str(getattr(self, "_stamp_source", "none"))

    @property
    def fps(self) -> float:
        dt = (time.monotonic() - self._t0) if self._t0 else 0.0
        return self._n / dt if dt > 0 else 0.0

    @property
    def frame_pipeline_stats(self) -> dict:
        """Snapshot latest-frame queue drops and worker conversion timing."""
        with self._lock:
            attempts = self._convert_attempts
            timing = dict(getattr(self, "_latest_timing", {}))

            def timing_ms(start: str, end: str):
                a, b = timing.get(start), timing.get(end)
                if a is None or b is None:
                    return None
                return max(0.0, (int(b) - int(a)) / 1_000_000.0)

            return {
                "received": self._frames_received,
                "enqueued": self._frames_enqueued,
                "converted": self._frames_converted,
                "queue_drops": self._queue_drops,
                "flush_drops": self._flush_drops,
                "stopped_drops": self._stopped_drops,
                "timestamp_drops": self._timestamp_drops,
                "timestamp_fallbacks": self._timestamp_fallbacks,
                "decode_failures": self._decode_failures,
                "duplicate_run": self._dup_n,
                "frozen": self._dup_n >= FROZEN_DUP_FRAMES,
                "stamp_source": self._stamp_source,
                "latest_age_ms": (
                    (time.monotonic() - self._stamp) * 1000.0
                    if self._latest is not None else None),
                "convert_attempts": attempts,
                "pending": int(self._pending_yuv is not None),
                "inflight": self._convert_inflight,
                "convert_ms_last": self._convert_last_s * 1000.0,
                "convert_ms_mean": (
                    self._convert_total_s * 1000.0 / attempts if attempts else 0.0),
                "convert_ms_max": self._convert_max_s * 1000.0,
                "callback_to_preprocess_ms": timing_ms(
                    "frame_callback_enter_mono_ns", "frame_preprocess_start_mono_ns"),
                "yuv_view_ms": timing_ms(
                    "frame_preprocess_start_mono_ns", "frame_yuv_ready_mono_ns"),
                "preprocess_ms": timing_ms(
                    "frame_preprocess_start_mono_ns", "frame_preprocess_done_mono_ns"),
                "callback_to_store_ms": timing_ms(
                    "frame_callback_enter_mono_ns", "frame_store_mono_ns"),
            }


# ----------------------------- real flight loop ----------------------------
def run_real(drone, loc, det, ctrl):
    """Closed-loop REAL flight on the SHARED drone (video grabber already started
    on this same drone). Mirrors autoflight.run() but reuses `drone` for PCMD so
    there is exactly one connection. Localizer pose is scale-free (MAP UNITS);
    only the controller gains map that to PCMD percentages.

    SAFETY: see module header. Validate in sim first; PROPS-OFF bench; open area
    with manual override; verify PCMD signs. Ctrl-C lands.
    """
    import math
    import os
    import signal
    import autoflight as af
    from olympe.messages.ardrone3.Piloting import TakeOff, PCMD, Landing
    from olympe.messages.ardrone3.PilotingState import FlyingStateChanged

    if os.environ.get("SFM_ALLOW_LEGACY_FLIGHT") != "1":
        raise SystemExit(
            "olympe_frame_source.run_real is a legacy arming path with no safety "
            "switch and no localization lock before takeoff. Use "
            "path_follow_flight.py fly, or set SFM_ALLOW_LEGACY_FLIGHT=1 only "
            "for controlled legacy testing."
        )

    print("[real] takeoff")
    _wait_success(drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=12)), "takeoff/hovering")

    stop = {"f": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("f", True))
    period = 1.0 / af.CTRL_HZ
    last_good = lost_since = None
    steps = 0
    try:
        while not stop["f"]:
            t0 = time.monotonic()
            p = loc.get_pose(); now = time.monotonic()
            fresh = p is not None and (now - p.stamp) <= af.POSE_STALE_S
            if fresh:
                last_good, lost_since = p, None
            elif last_good and (now - last_good.stamp) <= af.POSE_STALE_S:
                p, fresh = last_good, True

            if not fresh:
                drone(PCMD(1, 0, 0, 0, 0, 0)); lost_since = lost_since or now
                if now - lost_since >= af.LOST_LAND_S:
                    print("[real] localization lost -> land"); break
            else:
                roll, pitch, yaw, gaz, info = ctrl.step(p, det.detect(), now)
                drone(PCMD(1, roll, pitch, yaw, gaz, 0))
                if steps % 10 == 0:
                    print(f"[{info:28s}] pos=({p.x:5.1f},{p.y:5.1f},{p.z:4.1f}) "
                          f"yaw={math.degrees(p.yaw):6.1f} PCMD(p={pitch:+d},y={yaw:+d},g={gaz:+d})")
                if ctrl.state == "DONE":
                    print("[real] all targets done -> land"); break
            steps += 1
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        print("[real] landing")
        try:
            drone(PCMD(1, 0, 0, 0, 0, 0))
            _wait_success(drone(Landing()), "landing")   # raises SystemExit on timeout
        except (Exception, SystemExit) as exc:
            print(f"[real] warning: landing raised: {exc}")
        try:
            drone.disconnect()
        except Exception:
            pass


# ----------------------------- self-test / smoke ---------------------------
def _selftest():
    """Pure-python: the frame_source freshness/None contract (no olympe/cv2)."""
    g = OlympePdrawGrabber.__new__(OlympePdrawGrabber)
    g._lock = threading.Lock(); g._latest = None; g._stamp = 0.0; g._n = 0
    g._digest = None; g._dup_n = 0; g._frozen_warned = False
    g.stale_s = 0.1
    assert g() is None, "no frame yet -> None"
    frame = (np.arange(720 * 1280 * 3, dtype=np.uint8) % 255).reshape(720, 1280, 3)
    g._store(frame)
    sample = g()
    assert sample is not None, "fresh frame returned"
    out, capture_stamp = sample
    assert out.shape == (720, 1280, 3), "fresh frame returned"
    assert out is not g._latest, "must return a COPY, not the live buffer"
    assert np.array_equal(out, frame)
    assert capture_stamp == g._stamp, "capture stamp must travel atomically with the frame"
    g._stamp = time.monotonic() - 1.0                 # force stale
    assert g() is None, "stale frame -> None (controller HOVERs)"
    assert not g.is_healthy(), "stale stream should be unhealthy"
    # frozen stream: the SAME frame re-delivered with fresh stamps must go unhealthy
    for _ in range(FROZEN_DUP_FRAMES + 1):
        g._store(frame)
    assert g() is None, "frozen (duplicated) frames -> None (controller HOVERs)"
    assert not g.is_healthy(), "frozen stream should be unhealthy despite fresh stamps"
    g._store(frame.copy() + 1)                        # a genuinely new frame recovers
    assert g() is not None and g.is_healthy(), "new distinct frame -> healthy again"
    print("olympe_frame_source self-test: OK (None-before-frame, fresh copy, stale->None, "
          "stale->unhealthy, frozen->unhealthy, distinct-frame recovery)")


def _smoke_live(ip: str, secs: float, controller: str):
    """Connect, grab video for `secs`, report fps, save one sample. NO arming."""
    import cv2
    drone = connect(ip, controller)
    with OlympePdrawGrabber(drone) as g:
        t0 = time.monotonic(); last = None
        while time.monotonic() - t0 < secs:
            sample = g()
            if sample is not None:
                last, _capture_stamp = sample
            time.sleep(0.05)
        print(f"[smoke] fps~{g.fps:.1f}  last_frame={None if last is None else last.shape}")
        print(f"[smoke] pipeline={g.frame_pipeline_stats}")
        if last is not None:
            cv2.imwrite("/tmp/anafi_grab_sample.png", cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
            print("[smoke] saved /tmp/anafi_grab_sample.png")
    drone.disconnect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["selftest", "smoke"],
                    help="selftest=no deps; smoke=live video grab (props off, no arming)")
    ap.add_argument("--ip", default=DRONE_IP_REAL, help="192.168.42.1 real / 10.202.0.1 sim")
    ap.add_argument("--controller", default="auto", help="auto / drone / anafi / skycontroller3")
    ap.add_argument("--secs", type=float, default=5.0)
    args = ap.parse_args()
    if args.mode == "selftest":
        _selftest()
    else:
        _smoke_live(args.ip, args.secs, args.controller)
