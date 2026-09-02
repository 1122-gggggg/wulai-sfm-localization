"""Subprocess worker clients for live localization and detection."""

from __future__ import annotations

import json
import os
import queue
import select
import subprocess
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from PIL import Image

from live_localizer_protocol import (
    FusedTelemetry,
    encode_fused_request,
    encode_mode,
    encode_request,
)
from localization_metrics import CIRCUIT_BREAKER_STATES, RESTART_REASONS
from operator_localization_config import LOCALIZATION_BENCHMARK_LABELS


class _WorkerResponseDesync(RuntimeError):
    """A worker response could not be paired with the request that produced it."""


class LiveWorkerClient:
    """Shared queue/subprocess plumbing for the worker clients.

    Subclasses supply the worker argv, stderr log path, thread name, a short label
    for error messages, and any extra keys to include in error-result payloads.
    """

    MAX_RESPONSE_BYTES = 4 * 1024 * 1024
    MAX_PENDING_RESULTS = 32
    MAX_FATAL_RESTART_ATTEMPTS = 3
    FATAL_RESTART_BACKOFF_S = 1.0
    FATAL_RESTART_BACKOFF_MAX_S = 10.0

    def __init__(
        self,
        cmd: list,
        width: int,
        height: int,
        log_path: str,
        thread_name: str,
        label: str,
        error_defaults: dict | None = None,
        timeout_s: float = 8.0,
        use_shared_frames: bool = False,
        expect_ready_event: bool = False,
    ):
        self.width = int(width)
        self.height = int(height)
        self.label = label
        self.error_defaults = error_defaults or {}
        self.cmd = list(cmd)
        self.log_path = log_path
        self.timeout_s = float(os.environ.get("SFM_WORKER_TIMEOUT_S", timeout_s))
        self.restart_warmup_s = float(os.environ.get("SFM_WORKER_WARMUP_S", "20"))
        self._last_restart = 0.0
        self._spawned_at = 0.0
        self.max_fatal_restart_attempts = self.MAX_FATAL_RESTART_ATTEMPTS
        self.fatal_restart_backoff_s = self.FATAL_RESTART_BACKOFF_S
        self.fatal_restart_backoff_max_s = self.FATAL_RESTART_BACKOFF_MAX_S
        self._fatal_restart_attempts = 0
        self._worker_unavailable = False
        self._unavailable_notice_published = False
        # The client/worker protocol is strictly synchronous and carries no request
        # id, so a response can only be paired with a request by position. The worker
        # numbers its responses from 0, so the response to the k-th request written
        # since the worker started must carry seq == k-1. Any other value means the
        # stream has slipped and the payload belongs to an older frame.
        self._worker_requests_written = 0
        # os.read may return multiple newline-delimited worker responses. Keep
        # bytes after the first line for the next request instead of discarding
        # them and desynchronizing the frame/response protocol.
        self._stdout_buffer = bytearray()
        self._expect_ready_event = bool(expect_ready_event)
        self._ready_event = threading.Event()
        self._startup_info: dict = {}
        self._startup_error: str | None = None
        if not self._expect_ready_event:
            self._ready_event.set()
        self._frame_size = self.width * self.height * 3
        self._frame_shm_slots = 2 if use_shared_frames else 0
        self._frame_shm: shared_memory.SharedMemory | None = None
        self._active_shm_slot: int | None = None
        self._coalesce_scratch: bytearray | None = None
        if self._frame_shm_slots:
            self._frame_shm = shared_memory.SharedMemory(
                create=True, size=self._frame_size * self._frame_shm_slots
            )
            self._coalesce_scratch = bytearray(self._frame_size)
            self.cmd.extend(
                [
                    "--frame-shm-name",
                    self._frame_shm.name,
                    "--frame-shm-slots",
                    str(self._frame_shm_slots),
                ]
            )
        # Capacity-1 pipeline: never queue old work. While the worker is busy,
        # submit() overwrites a single coalesce slot so the next run uses the
        # newest frame (drop intermediate frames).
        self.pending: queue.Queue[tuple[int, str, bytes, bytes | memoryview | int, dict]] = (
            queue.Queue(maxsize=1)
        )
        self.results: queue.Queue[dict] = queue.Queue(maxsize=self.MAX_PENDING_RESULTS)
        self.in_flight = False
        self._coalesce: tuple[int, str, bytes, bytes | memoryview | int, dict] | None = None
        self._coalesce_drops = 0
        self._lock = threading.Lock()
        self._proc_lock = threading.RLock()
        self._closed = threading.Event()
        self._restart_reason: str | None = None
        self._outage_start_mono: float | None = None
        self._outage_duration_s: float | None = None
        self._rejected_submits = 0
        self._ready_mono: float | None = None
        self._ready_latency_ms: float | None = None
        self._first_result_latency_ms: float | None = None
        self._circuit_breaker_state = "closed"
        self._pending_oom_transition = False
        self._result_notify_read_fd, self._result_notify_write_fd = os.pipe()
        os.set_blocking(self._result_notify_read_fd, False)
        os.set_blocking(self._result_notify_write_fd, False)
        try:
            self.proc = self._spawn("w")
            if not self._expect_ready_event:
                self._note_worker_ready()
            self.thread = threading.Thread(target=self._loop, name=thread_name, daemon=True)
            self.thread.start()
        except Exception:
            os.close(self._result_notify_read_fd)
            os.close(self._result_notify_write_fd)
            if self._frame_shm is not None:
                self._frame_shm.close()
                self._frame_shm.unlink()
            raise

    @property
    def result_notify_fd(self) -> int:
        return self._result_notify_read_fd

    @property
    def ready(self) -> bool:
        return self._ready_event.is_set()

    @property
    def startup_info(self) -> dict:
        return dict(self._startup_info)

    @property
    def startup_error(self) -> str | None:
        return self._startup_error

    @property
    def unavailable(self) -> bool:
        return bool(getattr(self, "_worker_unavailable", False))

    def _publish_result(self, payload: dict) -> None:
        self._attach_lifecycle(payload)
        try:
            self.results.put_nowait(payload)
        except queue.Full:
            # The UI consumes only current state. If its event loop stalls, discard
            # the oldest result instead of allowing an unbounded latency/memory tail.
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            try:
                self.results.put_nowait(payload)
            except queue.Full:
                return
        try:
            os.write(self._result_notify_write_fd, b"\x01")
        except (BlockingIOError, BrokenPipeError, OSError):
            pass

    @staticmethod
    def _bounded_restart_reason(reason: str) -> str:
        if reason in RESTART_REASONS:
            return reason
        return "fatal"

    @staticmethod
    def _bounded_circuit_state(state: str) -> str:
        if state in CIRCUIT_BREAKER_STATES:
            return state
        return "open"

    def _reject_submit(self) -> bool:
        with self._lock:
            self._rejected_submits += 1
        return False

    def _note_outage_start(self, reason: str) -> None:
        self._restart_reason = self._bounded_restart_reason(reason)
        if self._outage_start_mono is None:
            self._outage_start_mono = time.monotonic()

    def _note_worker_ready(self) -> None:
        now = time.monotonic()
        self._ready_event.set()
        self._ready_mono = now
        self._ready_latency_ms = max(0.0, (now - self._spawned_at) * 1000.0)
        if self._outage_start_mono is not None:
            self._outage_duration_s = max(0.0, now - self._outage_start_mono)
            self._outage_start_mono = None
        self._circuit_breaker_state = self._bounded_circuit_state("closed")

    def _attach_lifecycle(self, payload: dict) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            if self._first_result_latency_ms is None and self._ready_mono is not None:
                self._first_result_latency_ms = max(
                    0.0, (time.monotonic() - self._ready_mono) * 1000.0
                )
            oom_transition = bool(self._pending_oom_transition)
            self._pending_oom_transition = False
            payload.update(
                {
                    "restart_reason": self._restart_reason,
                    "outage_duration_s": self._outage_duration_s,
                    "rejected_submits": self._rejected_submits,
                    "coalesced_submit_drops": getattr(self, "_coalesce_drops", 0),
                    "ready_latency_ms": self._ready_latency_ms,
                    "first_result_latency_ms": self._first_result_latency_ms,
                    "circuit_breaker_state": self._circuit_breaker_state,
                    "oom_transition": oom_transition,
                }
            )

    def drain_result_notifications(self) -> None:
        while True:
            try:
                if not os.read(self._result_notify_read_fd, 4096):
                    break
            except BlockingIOError:
                break
            except OSError:
                break

    def _spawn(self, mode: str) -> subprocess.Popen:
        self._stdout_buffer.clear()
        log = open(self.log_path, mode, encoding="utf-8")
        try:
            proc = subprocess.Popen(
                self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0
            )
        finally:
            log.close()  # the child owns its duplicated stderr fd
        self._spawned_at = time.monotonic()
        return proc

    def _request_prefix(self, timing_metadata: dict | None = None) -> bytes:
        """Per-request control bytes; generic image workers use the raw frame only."""
        return b""

    @staticmethod
    def _stop_process(proc: subprocess.Popen) -> None:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        if proc.poll() is None:
            try:
                # stdin EOF is the worker's normal shutdown signal. Give it a
                # short bounded chance to finish before escalating to SIGTERM.
                proc.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        else:
            proc.wait(timeout=0.0)  # reap an already-exited child
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass

    def _mark_worker_unavailable(self, reason: str) -> None:
        self._worker_unavailable = True
        self._ready_event.clear()
        self._note_outage_start("circuit_open")
        self._circuit_breaker_state = self._bounded_circuit_state("open")
        self._startup_error = (
            f"{self.label} worker unavailable after "
            f"{self._fatal_restart_attempts} fatal restart attempts: {reason}"
        )
        print(f"[operator] {self._startup_error}", flush=True)

    def _publish_unavailable_result(self, seq: int, frame_name: str) -> None:
        if getattr(self, "_unavailable_notice_published", False):
            return
        payload = self._error_result(
            seq,
            frame_name,
            self._startup_error or f"{self.label} worker unavailable",
        )
        payload.update(
            {
                "worker_unavailable": True,
                "failure_kind": "worker_unavailable",
                "restart_required": False,
            }
        )
        self._publish_result(payload)
        self._unavailable_notice_published = True

    def _note_healthy_response(self, payload: dict) -> None:
        """Reset fatal restart history only after a normal worker response."""
        if payload.get("restart_required") is True or self.unavailable:
            return
        self._fatal_restart_attempts = 0
        self._startup_error = None

    def _fatal_restart_delay(self, attempt: int) -> float:
        if attempt <= 1:
            return 0.0
        base = max(0.0, float(self.fatal_restart_backoff_s))
        ceiling = max(base, float(self.fatal_restart_backoff_max_s))
        return min(base * (2 ** (attempt - 2)), ceiling)

    def _write_all(self, proc: subprocess.Popen, raw: bytes | memoryview) -> None:
        """Write a frame without allowing a non-reading worker to block forever."""
        if proc.stdin is None:
            raise RuntimeError(f"{self.label} worker stdin closed")
        fd = proc.stdin.fileno()
        os.set_blocking(fd, False)
        startup_left = max(0.0, self.restart_warmup_s - (time.monotonic() - self._spawned_at))
        deadline = time.monotonic() + max(self.timeout_s, startup_left)
        view = memoryview(raw)
        while view:
            if self._closed.is_set():
                raise RuntimeError(f"{self.label} worker client closed")
            if proc.poll() is not None:
                raise RuntimeError(f"{self.label} worker exited")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.label} worker input stalled > {self.timeout_s:.0f}s")
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                raise TimeoutError(f"{self.label} worker input stalled > {self.timeout_s:.0f}s")
            try:
                written = os.write(fd, view[:65536])
            except BlockingIOError:
                continue
            if written <= 0:
                raise BrokenPipeError(f"{self.label} worker input pipe closed")
            view = view[written:]

    def _readline_with_timeout(
        self,
        proc: subprocess.Popen,
        timeout_s: float | None = None,
    ) -> bytes:
        """Read one complete response line without blocking after a partial write."""
        if proc.stdout is None:
            raise RuntimeError(f"{self.label} worker stdout closed")
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[: newline + 1])
                del self._stdout_buffer[: newline + 1]
                return line
            if len(self._stdout_buffer) > self.MAX_RESPONSE_BYTES:
                raise RuntimeError(
                    f"{self.label} worker response exceeds {self.MAX_RESPONSE_BYTES} bytes"
                )
            if self._closed.is_set():
                raise RuntimeError(f"{self.label} worker client closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self.label} worker returned an incomplete response > {timeout:.1f}s"
                )
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError(
                    f"{self.label} worker returned an incomplete response > {timeout:.1f}s"
                )
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeError(f"{self.label} worker exited with an incomplete response")
            self._stdout_buffer.extend(chunk)

    def _restart_blocked(self) -> bool:
        return self._closed.is_set() or self.unavailable

    def _plan_fatal_restart(self) -> float | None:
        max_attempts = max(0, int(self.max_fatal_restart_attempts))
        if self._fatal_restart_attempts >= max_attempts:
            self._mark_worker_unavailable("fatal restart budget exhausted")
            self._stop_process(self.proc)
            return None
        self._fatal_restart_attempts += 1
        return self._fatal_restart_delay(self._fatal_restart_attempts)

    def _cooldown_blocks_restart(self, *, force: bool, now: float) -> bool:
        return not force and now - self._last_restart < 30.0

    def _respawn_worker(self, delay: float) -> bool:
        try:
            self._stop_process(self.proc)
            if delay > 0.0 and self._closed.wait(delay):
                return False
            self._first_result_latency_ms = None
            self._ready_mono = None
            self._ready_latency_ms = None
            if self._expect_ready_event:
                self._ready_event.clear()
            self._startup_info = {}
            self._startup_error = None
            self._worker_requests_written = 0
            self.proc = self._spawn("a")
            if not self._expect_ready_event:
                self._note_worker_ready()
            print(f"[operator] {self.label} worker restarted after stall/exit", flush=True)
            return True
        except Exception as exc:
            print(f"[operator] {self.label} worker restart failed: {exc!r}", flush=True)
            return False

    def _restart(self, *, force: bool = False, fatal: bool = False, reason: str = "stall") -> bool:
        """Respawn a hung/exited worker so localization can recover. Cooldown-guarded so a
        freshly-restarted worker (which needs ~15s to reload models) is not thrashed.

        `force` bypasses the cooldown. It is for the response-desync case only: once
        request/response pairing has slipped, every later response is attributed to the
        wrong frame, so leaving the stalled worker attached is worse than thrashing.
        `fatal` applies a bounded circuit breaker for poisoned workers such as an
        EDM CUDA OOM or a worker that never completes its startup handshake.
        """
        if self._restart_blocked():
            return False
        if fatal:
            # submit() checks readiness before taking _proc_lock. Clear it before
            # the bounded backoff so Tk never waits behind a sleeping restart.
            self._ready_event.clear()
            self._note_outage_start(reason)
        with self._proc_lock:
            if self._restart_blocked():
                return False
            now = time.monotonic()
            delay = 0.0
            if fatal:
                planned = self._plan_fatal_restart()
                if planned is None:
                    return False
                delay = planned
            elif self._cooldown_blocks_restart(force=force, now=now):
                return False
            self._note_outage_start(reason)
            self._last_restart = now
            return self._respawn_worker(delay)

    def _error_result(self, seq: int, frame_name: str, error: str) -> dict:
        now_ns = time.monotonic_ns()
        payload = {
            "display_seq": seq,
            "frame_name": frame_name,
            "frame_id": frame_name,
            "capture_mono_ns": now_ns,
            "pose_mono_ns": now_ns,
            "localization_contract_version": 1,
            "validity": False,
            "confidence": 0.0,
            "success": False,
        }
        payload.update(self.error_defaults)
        payload["error"] = error
        return payload

    @staticmethod
    def _mark_mono(timing: dict, key: str) -> int:
        stamp_ns = time.monotonic_ns()
        timing[key] = stamp_ns * 1e-9
        timing[f"{key}_ns"] = stamp_ns
        return stamp_ns

    @staticmethod
    def _attach_client_timing(payload: dict, timing: dict) -> None:
        payload.update(timing)

        def duration_ms(start_key: str, end_key: str, out_key: str) -> None:
            start, end = payload.get(start_key), payload.get(end_key)
            if start is None or end is None:
                return
            try:
                payload[out_key] = max(0.0, (float(end) - float(start)) * 1000.0)
            except (TypeError, ValueError, OverflowError):
                pass

        def duration_ns(start_key: str, end_key: str, out_key: str) -> None:
            start, end = payload.get(start_key), payload.get(end_key)
            if start is None or end is None:
                return
            try:
                payload[out_key] = max(0.0, (int(end) - int(start)) / 1_000_000.0)
            except (TypeError, ValueError, OverflowError):
                pass

        duration_ms("client_submit_mono", "client_dequeue_mono", "client_queue_wait_ms")
        duration_ms("client_write_start_mono", "client_write_done_mono", "client_pipe_write_ms")
        duration_ms("client_submit_mono", "client_response_mono", "client_roundtrip_ms")
        duration_ms("client_submit_mono", "worker_read_done_mono", "submit_to_worker_read_ms")
        duration_ms("worker_core_done_mono", "client_response_mono", "worker_done_to_client_ms")
        duration_ms(
            "source_frame_stamp_mono", "client_submit_mono", "source_stamp_age_at_submit_ms"
        )
        duration_ns("client_submit_mono_ns", "client_dequeue_mono_ns", "client_queue_wait_ns_ms")
        duration_ns(
            "client_write_start_mono_ns", "client_write_done_mono_ns", "client_pipe_write_ns_ms"
        )
        duration_ns("client_submit_mono_ns", "client_response_mono_ns", "client_roundtrip_ns_ms")
        duration_ns(
            "frame_callback_enter_mono_ns",
            "frame_preprocess_start_mono_ns",
            "callback_to_preprocess_start_ms",
        )
        duration_ns("frame_preprocess_start_mono_ns", "frame_yuv_ready_mono_ns", "yuv_view_ms")
        duration_ns(
            "frame_preprocess_start_mono_ns", "frame_preprocess_done_mono_ns", "frame_preprocess_ms"
        )
        duration_ns(
            "frame_preprocess_done_mono_ns", "client_submit_mono_ns", "preprocess_done_to_submit_ms"
        )
        duration_ns(
            "frame_callback_enter_mono_ns", "client_submit_mono_ns", "callback_to_submit_ms"
        )
        duration_ns(
            "frame_callback_enter_mono_ns",
            "worker_core_start_mono_ns",
            "callback_to_inference_start_ms",
        )
        duration_ns(
            "frame_callback_enter_mono_ns",
            "worker_core_done_mono_ns",
            "callback_to_localization_done_ms",
        )

    def _take_work_item(self):
        """Pop one work item: prefer coalesce (newest), else pending queue."""
        if self._frame_shm is not None:
            try:
                return self.pending.get(timeout=0.1)
            except queue.Empty:
                return None
        with self._lock:
            if self._coalesce is not None:
                item = self._coalesce
                self._coalesce = None
                return item
        try:
            return self.pending.get(timeout=0.1)
        except queue.Empty:
            return None

    def _read_startup_handshake(self) -> None:
        with self._proc_lock:
            proc = self.proc
        try:
            line = self._readline_with_timeout(proc, timeout_s=self.restart_warmup_s)
            payload = json.loads(line.decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("event") != "ready":
                raise RuntimeError(f"{self.label} worker sent an invalid startup handshake")
            self._startup_info = payload
            self._startup_error = None
            self._note_worker_ready()
        except (TimeoutError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            if not self._closed.is_set():
                self._startup_error = repr(exc)
                self._restart(fatal=True, reason="startup")
                self._closed.wait(0.1)

    def _validate_worker_sequence(self, payload: dict, expected: int) -> None:
        worker_seq = payload.get("seq")
        if type(worker_seq) is int and worker_seq == expected:
            return
        # Pairing has slipped: never publish an older response as the newest pose.
        raise _WorkerResponseDesync(
            f"{self.label} worker response desync: got seq={worker_seq!r}, expected {expected}"
        )

    def _exchange_work_item(self, item) -> None:
        seq, frame_name, prefix, raw, timing = item
        self._mark_mono(timing, "client_dequeue_mono")
        if self._frame_shm is None:
            with self._lock:
                self.in_flight = True
        with self._proc_lock:
            proc = self.proc
        if proc.stdout is None:
            raise RuntimeError(f"{self.label} worker pipe closed")
        self._mark_mono(timing, "client_write_start_mono")
        if prefix:
            self._write_all(proc, prefix)
        request = bytes((int(raw),)) if self._frame_shm is not None else raw
        self._write_all(proc, request)
        self._worker_requests_written += 1
        expected_worker_seq = self._worker_requests_written - 1
        self._mark_mono(timing, "client_write_done_mono")
        line = self._readline_with_timeout(proc)
        self._mark_mono(timing, "client_response_mono")
        payload = json.loads(line.decode("utf-8"))
        self._validate_worker_sequence(payload, expected_worker_seq)
        payload.update({"display_seq": seq, "frame_name": frame_name, "frame_id": frame_name})
        self._attach_client_timing(payload, timing)
        oom = payload.get("failure_kind") == "cuda_oom" or payload.get("error") == "cuda_oom"
        if oom:
            with self._lock:
                self._pending_oom_transition = True
                self._restart_reason = "oom"
        self._publish_result(payload)
        if payload.get("restart_required") is True:
            self._restart(
                force=True,
                fatal=True,
                reason="oom" if oom else "fatal",
            )
        else:
            self._note_healthy_response(payload)

    def _handle_work_error(
        self, item, exc: Exception, *, announce: bool = False, fatal: bool = False
    ) -> None:
        if self._closed.is_set():
            return
        seq, frame_name, _prefix, _raw, timing = item
        if announce:
            print(f"[operator] {exc}", flush=True)
        self._mark_mono(timing, "client_response_mono")
        payload = self._error_result(seq, frame_name, repr(exc))
        self._attach_client_timing(payload, timing)
        self._publish_result(payload)
        if fatal:
            self._restart(force=True, fatal=True, reason="response_desync")
        else:
            self._restart(reason="stall")

    def _write_shm_slot(self, slot: int, payload: bytes | bytearray | memoryview) -> None:
        assert self._frame_shm is not None
        start = slot * self._frame_size
        self._frame_shm.buf[start : start + self._frame_size] = payload

    @staticmethod
    def _immutable_frame_bytes(raw: memoryview) -> bytes | None:
        obj = getattr(raw, "obj", None)
        if type(obj) is bytes and raw.readonly and raw.nbytes == len(obj):
            return obj
        return None

    def _coalesce_payload(self, raw: memoryview) -> bytes | memoryview:
        immutable = self._immutable_frame_bytes(raw)
        if immutable is not None:
            return immutable
        scratch = self._coalesce_scratch
        assert scratch is not None
        scratch[:] = raw
        return memoryview(scratch)

    def _promote_coalesced_work(self) -> None:
        with self._lock:
            nxt = self._coalesce
            self._coalesce = None
            if self._frame_shm is not None:
                if nxt is None or self._closed.is_set() or self._frame_shm is None:
                    self.in_flight = False
                    self._active_shm_slot = None
                    nxt = None
                else:
                    seq, frame_name, prefix, payload, timing = nxt
                    assert self._active_shm_slot is not None
                    slot = 1 - self._active_shm_slot
                    self._write_shm_slot(slot, payload)
                    nxt = (seq, frame_name, prefix, slot, timing)
                    self.in_flight = True
                    self._active_shm_slot = slot
            else:
                self.in_flight = False
        if nxt is None:
            return
        self._discard_pending(count_drops=False)
        try:
            self.pending.put_nowait(nxt)
        except queue.Full:
            pass

    def _process_work_item(self, item) -> None:
        try:
            self._exchange_work_item(item)
        except _WorkerResponseDesync as exc:
            self._handle_work_error(item, exc, announce=True, fatal=True)
        except (TimeoutError, RuntimeError, BrokenPipeError, OSError, json.JSONDecodeError) as exc:
            self._handle_work_error(item, exc)
        except Exception as exc:
            # Unexpected protocol/client failures can also leave pairing unknown.
            self._handle_work_error(item, exc)
        finally:
            self._promote_coalesced_work()

    def _loop(self) -> None:
        while not self._closed.is_set():
            if self.unavailable:
                self._closed.wait(0.1)
                continue
            if not self._ready_event.is_set():
                self._read_startup_handshake()
                continue
            item = self._take_work_item()
            if item is None:
                continue
            self._process_work_item(item)

    @staticmethod
    def _frame_memoryview(
        frame: Image.Image | np.ndarray | bytes | bytearray | memoryview,
    ) -> memoryview:
        if isinstance(frame, memoryview):
            return frame
        if isinstance(frame, (bytes, bytearray)):
            return memoryview(frame)
        if (
            isinstance(frame, np.ndarray)
            and frame.dtype == np.uint8
            and frame.flags["C_CONTIGUOUS"]
        ):
            return memoryview(frame).cast("B")
        return memoryview(frame.tobytes())

    def _discard_pending(self, *, count_drops: bool) -> None:
        try:
            while True:
                self.pending.get_nowait()
                if count_drops:
                    self._coalesce_drops += 1
        except queue.Empty:
            pass

    def _store_coalesced(self, item) -> bool:
        if self._coalesce is not None:
            self._coalesce_drops += 1
        self._coalesce = item
        return True

    def _submit_shared_frame(
        self, seq: int, frame_name: str, prefix: bytes, raw: memoryview, timing: dict
    ) -> bool:
        assert self._frame_shm is not None
        with self._lock:
            if self.in_flight:
                item = (
                    seq,
                    frame_name,
                    prefix,
                    self._coalesce_payload(raw),
                    timing,
                )
                return self._store_coalesced(item)
            slot = 0
            self._write_shm_slot(slot, raw)
            item = (seq, frame_name, prefix, slot, timing)
            self.in_flight = True
            self._active_shm_slot = slot
            self._discard_pending(count_drops=True)
            try:
                self.pending.put_nowait(item)
            except queue.Full:
                self.in_flight = False
                self._active_shm_slot = None
                return False
        return True

    def _submit_pipe_frame(
        self, seq: int, frame_name: str, prefix: bytes, raw: memoryview, timing: dict
    ) -> bool:
        item = (seq, frame_name, prefix, bytes(raw), timing)
        with self._lock:
            if self.in_flight:
                return self._store_coalesced(item)
        self._discard_pending(count_drops=True)
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            return False
        return True

    def submit(
        self,
        seq: int,
        frame_name: str,
        frame: Image.Image | np.ndarray | bytes | bytearray | memoryview,
        *,
        timing_metadata: dict | None = None,
    ) -> bool:
        """Enqueue at most one pending request; always prefer the newest frame.

        While inference is in flight, the request overwrites a single coalesce
        slot (drop intermediate frames). Never builds a multi-frame queue.
        """
        if self._closed.is_set():
            return self._reject_submit()
        if self.unavailable:
            self._publish_unavailable_result(seq, frame_name)
            return self._reject_submit()
        if not self._ready_event.is_set():
            return self._reject_submit()
        if (
            not self._expect_ready_event
            and time.monotonic() - self._last_restart < self.restart_warmup_s
        ):
            return self._reject_submit()
        with self._proc_lock:
            proc = self.proc
        if proc.poll() is not None:
            self._publish_result(
                self._error_result(
                    seq, frame_name, f"{self.label} worker exited code={proc.returncode}"
                )
            )
            self._restart(reason="worker_exit")
            return self._reject_submit()
        raw = self._frame_memoryview(frame)
        timing = dict(timing_metadata or {})
        try:
            prefix = self._request_prefix(timing)
        except ValueError:
            return self._reject_submit()
        timing["timing_clock"] = "host_monotonic_seconds"
        timing["timing_clock_ns"] = "host_monotonic_ns"
        self._mark_mono(timing, "client_submit_mono")
        if raw.nbytes != self._frame_size:
            return self._reject_submit()
        if self._frame_shm is not None:
            accepted = self._submit_shared_frame(seq, frame_name, prefix, raw, timing)
        else:
            accepted = self._submit_pipe_frame(seq, frame_name, prefix, raw, timing)
        if not accepted:
            return self._reject_submit()
        return True

    def poll_results(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                break
        return out

    def busy(self) -> bool:
        """True while a frame is mid-inference (UI may still coalesce submits)."""
        with self._lock:
            return self.in_flight

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._lock:
            self._coalesce = None
            self.in_flight = False
            self._active_shm_slot = None
        self._discard_pending(count_drops=False)
        with self._proc_lock:
            try:
                self._stop_process(self.proc)
            except (OSError, subprocess.SubprocessError):
                pass
        self.thread.join(timeout=2.0)
        for fd in (self._result_notify_read_fd, self._result_notify_write_fd):
            try:
                os.close(fd)
            except OSError:
                pass
        if self._frame_shm is not None:
            self._frame_shm.close()
            try:
                self._frame_shm.unlink()
            except FileNotFoundError:
                pass
            self._frame_shm = None


def _append_localizer_backend_args(
    command: list[str],
    *,
    backend: str,
    bundle_sha256: str,
    deploy_dir: str | Path,
    profile: str | Path,
    profile_sha256: str,
    matcher_mode: str,
) -> None:
    if bundle_sha256:
        command.extend(["--bundle-sha256", str(bundle_sha256)])
    if backend == "edm":
        command.extend(["--edm-matcher", "torch"])
        if deploy_dir:
            command.extend(["--deploy-dir", str(deploy_dir)])
        if profile:
            command.extend(["--production-profile", str(profile)])
        if profile_sha256:
            command.extend(["--production-profile-sha256", str(profile_sha256)])
    elif matcher_mode:
        command.extend(["--matcher-mode", str(matcher_mode)])


def _append_localizer_query_args(
    command: list[str], *, local_topk: int, query_camera, map_align: str | Path
) -> None:
    if local_topk and int(local_topk) > 0:
        command.extend(["--local-topk", str(int(local_topk))])
    if query_camera is not None:
        command.extend(
            [
                "--query-camera-model",
                str(query_camera.model),
                "--query-camera-width",
                str(int(query_camera.width)),
                "--query-camera-height",
                str(int(query_camera.height)),
                "--query-camera-params",
                *(str(float(value)) for value in query_camera.params),
            ]
        )
    if map_align:
        command.extend(["--map-align", str(map_align)])


def _append_localizer_reference_args(
    command: list[str],
    *,
    megaloc_cache: str | Path,
    reference_index: str | Path,
    reference_index_sha256: str,
    megaloc_backend: str,
    megaloc_engine: str | Path,
    megaloc_engine_sha256: str,
) -> None:
    # Pass an explicit empty cache value so an inherited environment variable
    # cannot override the site profile's bundle descriptors.
    command.extend(["--megaloc-cache", str(megaloc_cache)])
    command.extend(["--megaloc-backend", str(megaloc_backend)])
    if megaloc_engine:
        command.extend(["--megaloc-engine", str(megaloc_engine)])
    if megaloc_engine_sha256:
        command.extend(["--megaloc-engine-sha256", str(megaloc_engine_sha256)])
    if reference_index:
        command.extend(["--reference-index", str(reference_index)])
    if reference_index_sha256:
        command.extend(["--reference-index-sha256", str(reference_index_sha256)])


def _append_localizer_tracking_args(
    command: list[str],
    *,
    neuflow_track: bool,
    projection_track: bool,
    track_landmarks: str | Path,
    force_track_bench: bool,
    force_track_ref: int,
) -> None:
    if neuflow_track:
        command.append("--neuflow-track")
    if projection_track:
        command.extend(["--projection-track", "--track-landmarks", str(track_landmarks)])
    if force_track_bench:
        command.append("--force-track-bench")
    else:
        command.append("--runtime-benchmark-control")
    command.extend(["--force-track-ref", str(int(force_track_ref))])


class LiveLocalizerClient(LiveWorkerClient):
    def __init__(
        self,
        worker_py: Path,
        python_bin: str,
        width: int,
        height: int,
        bundle: Path,
        megaloc_cache: str | Path = "",
        megaloc_backend: str = "tensorrt",
        megaloc_engine: str | Path = "",
        megaloc_engine_sha256: str = "",
        reference_index: str | Path = "",
        reference_index_sha256: str = "",
        force_track_bench: bool = False,
        force_track_ref: int = -1,
        neuflow_track: bool = False,
        projection_track: bool = False,
        track_landmarks: str | Path = "",
        matcher_mode: str = "",
        localizer_backend: str = "auto",
        localizer_deploy_dir: str | Path = "",
        localizer_profile: str | Path = "",
        bundle_sha256: str = "",
        localizer_profile_sha256: str = "",
        local_topk: int = 0,
        query_camera=None,
        map_align: str | Path = "",
    ):
        self._runtime_benchmark_control = not bool(force_track_bench)
        self._benchmark_mode_lock = threading.Lock()
        self._benchmark_mode = "track" if force_track_bench else "auto"
        self._relocalize_once = False
        import flight_operator_app as _app

        self.localizer_backend = _app.resolve_localizer_backend(localizer_backend, Path(bundle))
        self.edm_matcher = "torch"
        cmd = [
            str(python_bin),
            str(worker_py),
            "--width",
            str(width),
            "--height",
            str(height),
            "--bundle",
            str(bundle),
            "--localizer-backend",
            self.localizer_backend,
        ]
        _append_localizer_backend_args(
            cmd,
            backend=self.localizer_backend,
            bundle_sha256=bundle_sha256,
            deploy_dir=localizer_deploy_dir,
            profile=localizer_profile,
            profile_sha256=localizer_profile_sha256,
            matcher_mode=matcher_mode,
        )
        _append_localizer_query_args(
            cmd,
            local_topk=local_topk,
            query_camera=query_camera,
            map_align=map_align,
        )
        _append_localizer_reference_args(
            cmd,
            megaloc_cache=megaloc_cache,
            megaloc_backend=megaloc_backend,
            megaloc_engine=megaloc_engine,
            megaloc_engine_sha256=megaloc_engine_sha256,
            reference_index=reference_index,
            reference_index_sha256=reference_index_sha256,
        )
        _append_localizer_tracking_args(
            cmd,
            neuflow_track=neuflow_track,
            projection_track=projection_track,
            track_landmarks=track_landmarks,
            force_track_bench=force_track_bench,
            force_track_ref=force_track_ref,
        )
        cmd.append("--startup-handshake")
        # The client owns the shared-memory segment. Attached workers unregister it
        # from resource_tracker, so a killed/restarted worker cannot unlink it.
        # SFM_SHARED_FRAMES=0 remains the slower pipe fallback.
        super().__init__(
            cmd,
            width,
            height,
            "/tmp/sfm_live_localizer_worker.log",
            "live-localizer-client",
            "localizer",
            error_defaults={
                "mode": "LOST",
                "next_mode": "LOST",
                "localization_exception": True,
            },
            use_shared_frames=os.environ.get("SFM_SHARED_FRAMES", "1").strip()
            not in {"0", "false", "no"},
            expect_ready_event=True,
        )

    @property
    def benchmark_mode(self) -> str:
        with self._benchmark_mode_lock:
            return self._benchmark_mode

    def set_benchmark_mode(self, mode: str) -> str:
        if not self._runtime_benchmark_control:
            raise RuntimeError("runtime mode switching is disabled by --force-track-bench")
        mode = str(mode)
        if mode not in LOCALIZATION_BENCHMARK_LABELS:
            raise ValueError(f"unsupported localization benchmark mode: {mode!r}")
        with self._benchmark_mode_lock:
            self._benchmark_mode = mode
        return mode

    def request_relocalize(self) -> None:
        """Force the next submitted frame through LOST recovery.

        The tracker stages MegaLoc top-k 10 then 20 per LOST episode; later
        held-frame requests stay on EDM local/map recovery.
        """
        if not self._runtime_benchmark_control:
            return
        with self._benchmark_mode_lock:
            self._relocalize_once = True

    def _request_prefix(self, timing_metadata: dict | None = None) -> bytes:
        if not self._runtime_benchmark_control:
            return b""
        with self._benchmark_mode_lock:
            mode = "relocalize" if self._relocalize_once else self._benchmark_mode
            self._relocalize_once = False
        timing = timing_metadata or {}
        capture_stamp = timing.get("source_frame_stamp_mono")
        if capture_stamp:
            fused_stamp = timing.get("fused_telemetry_mono")
            telemetry_stamp = None
            if fused_stamp is not None and fused_stamp != "":
                try:
                    telemetry_stamp = float(fused_stamp)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"invalid fused telemetry stamp: {fused_stamp!r}") from exc
            fused = FusedTelemetry(
                stamp=telemetry_stamp,
                roll=timing.get("fused_roll"),
                pitch=timing.get("fused_pitch"),
                yaw=timing.get("fused_yaw"),
                speed_north=timing.get("fused_speed_north"),
                speed_east=timing.get("fused_speed_east"),
                speed_down=timing.get("fused_speed_down"),
                gps_stamp=timing.get("fused_gps_mono"),
                latitude=timing.get("fused_gps_latitude"),
                longitude=timing.get("fused_gps_longitude"),
                altitude=timing.get("fused_gps_altitude"),
                latitude_accuracy=timing.get("fused_gps_latitude_accuracy"),
                longitude_accuracy=timing.get("fused_gps_longitude_accuracy"),
                altitude_accuracy=timing.get("fused_gps_altitude_accuracy"),
            )
            if telemetry_stamp is not None and (
                fused.has_attitude or fused.has_velocity or fused.has_gnss
            ):
                return encode_fused_request(
                    mode,
                    float(capture_stamp),
                    fused,
                    telemetry_stamp=telemetry_stamp,
                )
            return encode_request(mode, capture_stamp)
        return encode_mode(mode)


class LiveDetectorClient(LiveWorkerClient):
    def __init__(
        self,
        worker_py: Path,
        python_bin: str,
        width: int,
        height: int,
        model: Path,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
    ):
        cmd = [
            str(python_bin),
            str(worker_py),
            "--width",
            str(width),
            "--height",
            str(height),
            "--model",
            str(model),
            "--imgsz",
            str(int(imgsz)),
            "--conf",
            str(float(conf)),
            "--iou",
            str(float(iou)),
            "--max-det",
            str(int(max_det)),
        ]
        super().__init__(
            cmd,
            width,
            height,
            "/tmp/sfm_live_detector_worker.log",
            "live-detector-client",
            "detector",
            error_defaults={"count": 0, "boxes": []},
        )
