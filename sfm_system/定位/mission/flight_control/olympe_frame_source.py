#!/usr/bin/env python3
"""Live ANAFI PDRAW frame grabber -> a reloc_localizer frame_source().

Swaps the sim's SplatFrameSource for the REAL drone with ZERO change to the
localizer or controller: both consume the same `frame_source() -> HxWx3 uint8
RGB | None` contract. Same map M, same ALIKED+LightGlue+PnP, same scale-free
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
    (uint8 HxWx3) if fresher than `stale_s`, else None -- which the controller's
    POSE_STALE / HOVER path already handles."""

    def __init__(self, drone, resize: tuple | None = (1280, 720), stale_s: float = 0.5,
                 media_name: str = "Front camera"):
        self.drone = drone
        self.resize = resize
        self.stale_s = stale_s
        self.media_name = media_name
        self._lock = threading.Lock()
        self._latest = None          # RGB uint8 HxWx3
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

    # ---- lifecycle ----
    def start(self):
        # Olympe >=8.4 documented API: drone.streaming.play(raw_cb=...).
        # Older Olympe builds exposed set_callbacks()+start(); keep it as a
        # compatibility fallback because this workstation has older scripts.
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

    def stop(self):
        try:
            self.drone.streaming.stop()
        except Exception:
            pass

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
        import cv2
        yuv_frame.ref()
        try:
            arr = yuv_frame.as_ndarray()              # full YUV plane (H*3/2, W)
            code = self._formats().get(yuv_frame.format(), self._default_code)
            rgb = cv2.cvtColor(arr, code)             # -> RGB HxWx3 uint8
            if self.resize is not None and (rgb.shape[1], rgb.shape[0]) != self.resize:
                rgb = cv2.resize(rgb, self.resize)
            self._store(rgb)
        except Exception as exc:
            # A single bad frame must NOT kill Olympe's stream thread -> just drop it.
            # Sustained failure -> frames go stale -> the controller HOVERs on staleness.
            if not self._decode_warned:
                self._decode_warned = True
                print(f"[olympe_frame_source] frame decode failed ({exc!r}); dropping frame(s), "
                      "stream stays alive (controller hovers on staleness)", flush=True)
        finally:
            yuv_frame.unref()

    def _flush_cb(self, *_):
        return True

    # ---- store / read (the frame_source contract) ----
    def _store(self, rgb: np.ndarray):
        digest = rgb[::64, ::64].tobytes()   # ~720B sample; identical only if the frame repeats
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
            self._stamp = time.monotonic()
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
            return self._latest.copy()

    def last_frame_age(self) -> float | None:
        with self._lock:
            if self._latest is None:
                return None
            return time.monotonic() - self._stamp

    def is_healthy(self) -> bool:
        with self._lock:
            if self._dup_n >= FROZEN_DUP_FRAMES:
                return False                           # frozen picture with fresh stamps
        age = self.last_frame_age()
        return age is not None and age <= self.stale_s

    @property
    def fps(self) -> float:
        dt = (time.monotonic() - self._t0) if self._t0 else 0.0
        return self._n / dt if dt > 0 else 0.0


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
    out = g()
    assert out is not None and out.shape == (720, 1280, 3), "fresh frame returned"
    assert out is not g._latest, "must return a COPY, not the live buffer"
    assert np.array_equal(out, frame)
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
            f = g()
            if f is not None:
                last = f
            time.sleep(0.05)
        print(f"[smoke] fps~{g.fps:.1f}  last_frame={None if last is None else last.shape}")
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
