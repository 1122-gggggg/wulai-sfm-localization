"""File-stream decode for the desktop operator interface.

Decode a recorded file to 720p RGB frames, optionally through an ANAFI-like
radio-link simulation. Tk must never wait on an FFmpeg pipe.
"""
from __future__ import annotations

import collections
import math
import queue
import random
import subprocess
import threading
from pathlib import Path

from PIL import Image


def _annex_b_nal_starts(buffer: bytearray) -> list[int]:
    starts: list[int] = []
    offset = 0
    while True:
        start = buffer.find(b"\x00\x00\x01", offset)
        if start < 0:
            return starts
        starts.append(start)
        offset = start + 3


def _write_nal_bytes(dst, data: bytes | bytearray) -> None:
    if not data:
        return
    dst.write(data)
    dst.flush()


def _write_complete_nals(
        buffer: bytearray, starts: list[int], dst,
        rng: random.Random, loss_fraction: float) -> None:
    output = bytearray()
    for start, end in zip(starts, starts[1:]):
        nal = buffer[start:end]
        nal_type = nal[3] & 0x1F if len(nal) > 3 else 0
        if nal_type == 1 and rng.random() < loss_fraction:
            continue
        output += nal
    _write_nal_bytes(dst, output)
    del buffer[:starts[-1]]


def _close_nal_destination(dst) -> None:
    try:
        dst.close()
    except (BrokenPipeError, OSError):
        pass


class FFmpegFrameStream:
    """Decode a file to 720p RGB frames, optionally through an ANAFI-like radio link.

    Without link_sim the source is only rescaled, so the localizer sees near-original
    detail -- optimistic compared to the real drone. With link_sim the frames are
    re-encoded to the ANAFI v1.4 live-stream contract (white paper §5.2: 720p, H264
    main profile, up to 5 Mb/s, 45 slices of 16 px with periodic intra-refresh) and
    decoded back, so EDM sees the same compression artefacts it will see in flight.
    """

    _READ_EOF = object()

    def __init__(self, video_path: Path, width: int, height: int, stride: int = 1,
                 fps: float = 30.0, loop: bool = False,
                 link_sim: dict | None = None):
        self.video_path = Path(video_path)
        self.width = int(width)
        self.height = int(height)
        self.stride = max(1, int(stride))
        self.requested_fps = float(fps)
        if not math.isfinite(self.requested_fps) or self.requested_fps <= 0.0:
            raise ValueError("fps must be finite and > 0")
        import flight_operator_app as _app
        self.source_fps = _app.probe_video_fps(self.video_path) if self.video_path.is_file() else None
        if self.source_fps is None:
            self.output_fps = self.requested_fps / self.stride
        elif self.stride > 1:
            self.output_fps = self.source_fps / self.stride
        else:
            self.output_fps = min(self.source_fps, self.requested_fps)
        self.fps = self.output_fps
        self.loop = bool(loop)
        self.link_sim = dict(link_sim) if link_sim else None
        self.proc: subprocess.Popen | None = None
        self.enc_proc: subprocess.Popen | None = None
        self._nal_thread: threading.Thread | None = None
        # Tk must never wait on an FFmpeg pipe.  The reader is request-driven so
        # replay frames are not consumed faster than the UI asks for them, while
        # the one-item queue keeps process output bounded.
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._reader_request = threading.Event()
        self._reader_queue: queue.Queue[bytes | object] = queue.Queue(maxsize=1)
        self._reader_pending = False
        self._reader_generation: object | None = None
        self._reader_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        # End-to-end link latency, expressed as a frame backlog (280 ms at 30 fps ~ 8).
        latency_ms = float((self.link_sim or {}).get("latency_ms", 0.0) or 0.0)
        self.delay_frames = int(round(latency_ms * self.fps / 1000.0)) if latency_ms > 0 else 0
        self._delay_buf: collections.deque = collections.deque()
        self.frame_size = self.width * self.height * 3
        self.output_index = 0
        self.last_frame_name = ""
        self.eof = False
        self.terminal_state = "RUNNING"
        if self.video_path.exists():
            self.start()

    def _video_filter(self) -> str:
        if self.stride > 1:
            return f"select=not(mod(n\\,{self.stride})),scale={self.width}:{self.height}"
        if self.source_fps is not None and self.source_fps <= self.requested_fps:
            return f"scale={self.width}:{self.height}"
        return f"fps={self.requested_fps:g},scale={self.width}:{self.height}"

    def _encoder_argv(self, vf: str) -> list:
        sim = self.link_sim or {}
        kbps = int(sim.get("kbps", 5000))
        slices = int(sim.get("slices", max(1, self.height // 16)))
        x264opts = f"slices={slices}:bframes=0"
        if sim.get("intra_refresh", True):
            x264opts = "intra-refresh=1:" + x264opts
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(self.video_path),
            "-vf", vf,
            "-vsync", "vfr",
            "-c:v", "libx264",
            "-profile:v", str(sim.get("profile", "main")),
            "-preset", "veryfast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k",
            "-bufsize", f"{max(1, kbps // 4)}k",
            "-x264opts", x264opts,
            "-f", "h264", "pipe:1",
        ]

    @staticmethod
    def _pump_nals(src, dst, loss_pct: float, seed: int) -> None:
        """Copy an Annex-B H264 stream, dropping a fraction of the slice NALs.

        Models Wi-Fi packet loss: the decoder still produces a frame but conceals the
        missing slices, which is what the real link does (white paper §5.2.1.3). SPS/PPS
        and IDR slices are never dropped -- losing those kills the stream rather than
        degrading it, which is not the failure mode we want to reproduce.
        """
        rng = random.Random(seed)
        buf = bytearray()
        try:
            while True:
                chunk = src.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                starts = _annex_b_nal_starts(buf)
                if len(starts) < 2:
                    continue
                _write_complete_nals(buf, starts, dst, rng, loss_pct / 100.0)
            _write_nal_bytes(dst, buf)
        except (BrokenPipeError, ValueError, OSError):
            pass
        finally:
            _close_nal_destination(dst)

    def _spawn_processes(self) -> None:
        """Start the encoder/decoder pair without touching reader state."""
        vf = self._video_filter()
        if self.link_sim:
            # Radio-link simulation: encode to the ANAFI stream contract, then decode.
            self.enc_proc = subprocess.Popen(
                self._encoder_argv(vf),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            loss_pct = float(self.link_sim.get("loss_pct", 0.0) or 0.0)
            dec_stdin = self.enc_proc.stdout
            if loss_pct > 0.0:
                # Drop slice NALs between encoder and decoder.
                self.proc = subprocess.Popen(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-err_detect", "ignore_err",
                        "-f", "h264", "-i", "pipe:0",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                self._nal_thread = threading.Thread(
                    target=self._pump_nals,
                    args=(dec_stdin, self.proc.stdin, loss_pct,
                          int(self.link_sim.get("loss_seed", 20260726))),
                    name="anafi-nal-loss", daemon=True,
                )
                self._nal_thread.start()
                return
            self.proc = subprocess.Popen(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "h264", "-i", "pipe:0",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                ],
                stdin=dec_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # Only the decoder reads the encoder pipe; drop our handle so the encoder
            # gets SIGPIPE when the decoder goes away.
            if self.enc_proc.stdout is not None:
                self.enc_proc.stdout.close()
            return
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(self.video_path),
                "-vf", vf,
                "-vsync", "vfr",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _close_pipe(proc: object, attribute: str) -> None:
        stream = getattr(proc, attribute, None)
        close = getattr(stream, "close", None)
        if not callable(close):
            return
        try:
            close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def _terminate_processes(self) -> None:
        """Stop child processes and pipe workers with bounded waits."""
        processes = [proc for proc in (self.proc, self.enc_proc) if proc is not None]
        self.proc = None
        self.enc_proc = None

        # Closing the parent-side pipes also releases a reader blocked in a fake or
        # real pipe whose child has not observed SIGTERM yet.
        seen: set[int] = set()
        for proc in processes:
            if id(proc) in seen:
                continue
            seen.add(id(proc))
            for attribute in ("stdin", "stdout", "stderr"):
                self._close_pipe(proc, attribute)

        for proc in processes:
            terminate = getattr(proc, "terminate", None)
            if callable(terminate):
                try:
                    terminate()
                except (OSError, ValueError):
                    pass
            wait = getattr(proc, "wait", None)
            if not callable(wait):
                continue
            try:
                wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                kill = getattr(proc, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except (OSError, ValueError):
                        pass
                try:
                    wait(timeout=0.5)
                except (subprocess.TimeoutExpired, OSError, ValueError):
                    pass
            except (OSError, ValueError):
                pass

        nal_thread = self._nal_thread
        self._nal_thread = None
        if nal_thread is not None and nal_thread is not threading.current_thread():
            nal_thread.join(timeout=1.0)

    def _clear_reader_queue(self) -> None:
        while True:
            try:
                self._reader_queue.get_nowait()
            except queue.Empty:
                return

    def _reader_is_current(self, generation: object, stop: threading.Event) -> bool:
        with self._reader_lock:
            return (
                not stop.is_set()
                and not self._closed
                and self._reader_generation is generation
            )

    def _reader_publish(
            self, generation: object, stop: threading.Event, item: bytes | object,
    ) -> None:
        with self._reader_lock:
            if not self._reader_is_current(generation, stop):
                return
            try:
                self._reader_queue.put_nowait(item)
            except queue.Full:
                # There is normally one result per request.  If shutdown races
                # with publication, retain the newest bounded result.
                try:
                    self._reader_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._reader_queue.put_nowait(item)
                except queue.Full:
                    pass

    @staticmethod
    def _read_complete_frame(
            stdout: object, frame_size: int, stop: threading.Event,
    ) -> bytes | None:
        raw = bytearray()
        while len(raw) < frame_size:
            if stop.is_set():
                return None
            read = getattr(stdout, "read", None)
            if not callable(read):
                return None
            chunk = read(frame_size - len(raw))
            if not chunk:
                break
            raw.extend(chunk)
        return bytes(raw) if len(raw) == frame_size else None

    def _set_reader_terminal(
            self, generation: object, stop: threading.Event, proc: object,
            *, decode_error: bool = False,
    ) -> None:
        with self._reader_lock:
            if not self._reader_is_current(generation, stop) or self.proc is not proc:
                return
            poll = getattr(proc, "poll", None)
            try:
                return_code = poll() if callable(poll) else 0
            except (OSError, ValueError):
                return_code = 1
            self.eof = True
            self.terminal_state = (
                "DECODE_ERROR_HOLD"
                if decode_error or return_code not in {None, 0}
                else "EOF_HOLD"
            )

    def _restart_process_for_reader(
            self, generation: object, stop: threading.Event,
    ) -> bool:
        """Restart a looping replay from the reader thread, never from Tk."""
        with self._lifecycle_lock:
            if not self._reader_is_current(generation, stop):
                return False
            self._terminate_processes()
            self.output_index = 0
            self.last_frame_name = ""
            self.eof = False
            self.terminal_state = "RUNNING"
            self._delay_buf.clear()
            try:
                self._spawn_processes()
            except Exception:
                self.eof = True
                self.terminal_state = "DECODE_ERROR_HOLD"
                return False
            return self.proc is not None

    def _reader_loop(
            self, generation: object, stop: threading.Event,
            request: threading.Event,
    ) -> None:
        while self._reader_is_current(generation, stop):
            request.wait(0.1)
            if not self._reader_is_current(generation, stop):
                return
            request.clear()
            if self.eof:
                if not self.loop:
                    return
                if not self._restart_process_for_reader(generation, stop):
                    self._reader_publish(generation, stop, self._READ_EOF)
                    return
            proc = self.proc
            stdout = getattr(proc, "stdout", None) if proc is not None else None
            if proc is None or stdout is None:
                self._set_reader_terminal(
                    generation, stop, proc, decode_error=True,
                )
                self._reader_publish(generation, stop, self._READ_EOF)
                return
            decode_error = False
            try:
                raw = self._read_complete_frame(stdout, self.frame_size, stop)
            except Exception:
                raw = None
                decode_error = True
            if not self._reader_is_current(generation, stop):
                return
            if raw is None:
                self._set_reader_terminal(
                    generation, stop, proc, decode_error=decode_error,
                )
                self._reader_publish(generation, stop, self._READ_EOF)
                if not self.loop:
                    return
            else:
                self._reader_publish(generation, stop, raw)

    def _ensure_reader(self) -> None:
        # A loop restart or shutdown may be stopping a child in the background.
        # Never make the Tk callback wait for that lifecycle lock; the next tick
        # can retry the request after the reader is available again.
        if not self._lifecycle_lock.acquire(blocking=False):
            return
        try:
            if self._closed or self.proc is None:
                return
            if self.eof and not self.loop:
                return
            if self._reader_thread is not None and self._reader_thread.is_alive():
                return
            self._reader_stop = threading.Event()
            self._reader_request = threading.Event()
            self._reader_pending = False
            self._clear_reader_queue()
            generation = object()
            self._reader_generation = generation
            thread = threading.Thread(
                target=self._reader_loop,
                args=(generation, self._reader_stop, self._reader_request),
                name="ffmpeg-frame-reader",
                daemon=True,
            )
            self._reader_thread = thread
            thread.start()
            # Prime one bounded request while the rest of the operator runtime
            # starts, so the first Tk tick normally has a frame ready without
            # ever reading the pipe itself.
            self._reader_pending = True
            self._reader_request.set()
        finally:
            self._lifecycle_lock.release()

    def _request_frame(self) -> None:
        self._ensure_reader()
        with self._reader_lock:
            if self._closed or self.proc is None:
                return
            if self.eof and not self.loop:
                return
            if self._reader_pending:
                return
            if self._reader_thread is None or not self._reader_thread.is_alive():
                return
            self._reader_pending = True
            self._reader_request.set()

    def start(self) -> None:
        self.close()
        with self._lifecycle_lock:
            self._closed = False
            self.output_index = 0
            self.last_frame_name = ""
            self.eof = False
            self.terminal_state = "RUNNING"
            self._delay_buf.clear()
            self._clear_reader_queue()
            self._reader_stop = threading.Event()
            self._reader_request = threading.Event()
            self._reader_pending = False
            generation = object()
            self._reader_generation = generation
            try:
                self._spawn_processes()
            except Exception:
                self._closed = True
                self._reader_generation = None
                self._terminate_processes()
                raise
            thread = threading.Thread(
                target=self._reader_loop,
                args=(generation, self._reader_stop, self._reader_request),
                name="ffmpeg-frame-reader",
                daemon=True,
            )
            self._reader_thread = thread
            thread.start()
            self._reader_pending = True
            self._reader_request.set()

    def close(self) -> None:
        with self._reader_lock:
            self._closed = True
            stop = self._reader_stop
            request = self._reader_request
            reader = self._reader_thread
            self._reader_generation = None
            self._reader_pending = False
        stop.set()
        request.set()
        with self._lifecycle_lock:
            self._terminate_processes()
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        with self._reader_lock:
            if self._reader_thread is reader:
                self._reader_thread = None
            self._reader_pending = False
            self._clear_reader_queue()

    def _read_raw(self) -> bytes | None:
        self._ensure_reader()
        try:
            item = self._reader_queue.get_nowait()
        except queue.Empty:
            self._request_frame()
            return None
        with self._reader_lock:
            self._reader_pending = False
        if item is self._READ_EOF:
            return None
        raw = bytes(item)
        self._request_frame()
        return raw

    def next_frame(self) -> Image.Image | None:
        raw = self._read_raw()
        if raw is None:
            if self.eof and self._delay_buf:
                # At EOF the decoder cannot refill the latency backlog. Drain it
                # directly; appending these tail frames again would rotate them
                # forever and never expose the final frames to the localizer.
                raw = self._delay_buf.popleft()
            else:
                return None
        elif self.delay_frames > 0:
            # Hold a fixed backlog so the consumer always sees a frame captured
            # delay_frames earlier, matching the link's end-to-end latency.
            #
            # The backlog fills one frame per call. Draining it synchronously here
            # instead would block the UI thread for delay_frames reads on the very
            # first tick, and the localizer worker -- still loading its bundle --
            # gets torn down as a stalled worker, taking its frame shm with it.
            self._delay_buf.append(raw)
            if len(self._delay_buf) <= self.delay_frames:
                return None
            raw = self._delay_buf.popleft()
        if raw is None:
            return None
        self.output_index += 1
        self.last_frame_name = f"frame_{self.output_index:06d}.jpg"
        return Image.frombytes("RGB", (self.width, self.height), raw)
