"""Durable per-session JSONL logs, hardware/video inventory receipts, and the
offline runtime identity recorded into the session manifest.

Split out of the former runtime_safety.py (D2): this file owns session/log
lifecycle only. Disk-pressure/retention policy lives in disk_policy.py,
offline-network enforcement in network_policy.py, and the autonomy arming
gate in arming_gate.py -- each independently readable and testable.
"""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import os
import platform
from collections import deque
import sys
import time
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from backend_contract import InterfaceMode
from localization_metrics import DIRECT_INFO_FIELDS


_BULK_LOG_SYNC_EVERY = 16
_DIAGNOSTIC_QUEUE_CAPACITY = 512

_PERMANENT_LOG_NAMES = {
    "trajectory.jsonl",
    "commands.jsonl",
    "hardware_inventory.json",
    "incidents.jsonl",
    "session_manifest.json",
    "session_summary.json",
    "video_inventory.json",
}
_INCIDENT_EVENTS = {
    "controller_disconnect",
    "fail_safe",
    "stream_lost",
    "link_lost",
    "lost_link_policy",
    "rth",
    "runtime_safety",
    "runtime_safety_latched",
    "runtime_safety_exception",
    "runtime_safety_rearmed",
    "distance_guard",
    "distance_guard_unavailable",
    "worker_exit",
    "worker_stall",
    "backend_poll_failed",
    "diagnostic_write_failed",
    "legacy_takeoff_rejected",
    "logging_failed",
    "uncommanded_landing",
    "rth_in_progress",
    "disk_critical",
    "cleanup_begin",
    "cleanup_done",
    "land_cmd",
}

_RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "opencv-python",
    "parrot-olympe",
    "Pillow",
    "pycolmap",
    "torch",
    "torchvision",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_runtime_identity() -> dict[str, Any]:
    """Collect a JSON-safe, offline runtime identity for the session manifest."""
    packages: dict[str, str | None] = {}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None

    # The operator process never loads the models, so importing torch here only
    # delayed the window by ~1.2 s. The worker that really owns CUDA reports it
    # at startup; until then (or if it never starts) the value is unknown, not False.
    cuda: dict[str, Any] = {
        "available": None,
        "status": "pending_worker_report",
        "runtime_version": None,
        "device_names": [],
    }

    driver_path = Path("/proc/driver/nvidia/version")
    nvidia_driver: dict[str, Any] = {
        "path": str(driver_path),
        "version_line": None,
    }
    try:
        lines = driver_path.read_text(encoding="utf-8", errors="replace").splitlines()
        nvidia_driver["version_line"] = lines[0] if lines else None
    except OSError as exc:
        nvidia_driver["error"] = repr(exc)

    return {
        "python": {
            "version": platform.python_version(),
            "build": sys.version,
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "nvidia_driver": nvidia_driver,
    }


def _record(event: str, fields: dict[str, Any]) -> dict[str, Any]:
    return _finite_json({
        "t_utc": _utc_now(),
        "t_mono_ns": time.monotonic_ns(),
        "event": event,
        **fields,
    })


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip().rsplit(".", 1)[-1]
    return str(value)


def flight_debug_contract() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "定位演算法/flight_control/real_path_follow_controller.py",
        "定位演算法/flight_control/path_follow_flight.py",
        "定位演算法/deploy_code/sfm_direct_deploy/two_rate_tracker.py",
        "控制介面程式/operator_interface/operator_autonomy.py",
        "控制介面程式/operator_interface/olympe_live_backend.py",
        "定位演算法/flight_control/control_state_estimator.py",
        "定位演算法/flight_control/wind_observer.py",
        "定位演算法/flight_control/heading_fusion.py",
        "定位演算法/flight_control/pose_frame_chain.py",
        "定位演算法/flight_control/olympe_frame_source.py",
        "控制介面程式/operator_interface/imu_flight_test.py",
        "控制介面程式/operator_interface/ekf_capture_contract.py",
        "控制介面程式/operator_interface/localization_metrics.py",
        "控制介面程式/operator_interface/live_localizer_worker.py",
        "控制介面程式/operator_interface/live_worker_clients.py",
        "控制介面程式/operator_interface/backend_telemetry.py",
        "控制介面程式/operator_interface/session_logs.py",
        "控制介面程式/operator_interface/flight_operator_app.py",
        "tools/imu_flight_test_report.py",
    )
    return {
        "version": 3,
        "clock": "host_monotonic; *_mono_ns nanoseconds, *_mono seconds",
        "map_position_units": "map units; not metres without measured scale",
        "velocity": "firmware-fused NED, metres/second",
        "attitude": "firmware-fused Euler radians; yaw clockwise from North",
        "raw_accelerometer_available": False, "raw_gyroscope_available": False,
        "sensor_covariance_available": False,
        "position_reference": "camera_center; coincident body-center approximation requested by operator",
        "body_orientation_is_camera_orientation": False,
        "ekf_enabled": False,
        "pcmd_order": ["roll", "pitch", "yaw", "gaz"],
        "pcmd_dispatch_semantics": "SDK call; not firmware execution acknowledgement",
        "source_files_at_session_start": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths if (root / name).is_file()
        },
    }


class _JsonlSink:
    def __init__(self, path: Path, *, sync_every: int = 1):
        self.path = path
        self._handle = path.open("a", encoding="utf-8", buffering=1)
        self._sync_every = max(1, int(sync_every))
        self._pending_writes = 0
        self._lock = threading.Lock()
        self.healthy = True
        self.last_error = ""
        self.durable = False
        try:
            # A durable property is only truthful after the kernel accepted an
            # fsync on this exact sink.  The probe also makes a fresh session
            # fail closed before its first safety command is issued.
            self._sync()
        except Exception as exc:
            self.healthy = False
            self.last_error = repr(exc)
            print(
                f"[session-log] durability probe failed path={self.path}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending_writes = 0
        self.durable = True

    def write(self, value: dict[str, Any]) -> bool:
        with self._lock:
            if not self.healthy:
                return False
            try:
                self._handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                self._pending_writes += 1
                if self._pending_writes >= self._sync_every:
                    self._sync()
                return True
            except Exception as exc:
                self.healthy = False
                self.durable = False
                self.last_error = repr(exc)
                print(
                    f"[session-log] write failed path={self.path}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                return False

    def close(self) -> None:
        with self._lock:
            try:
                if self.healthy and self._pending_writes:
                    self._sync()
            except Exception as exc:
                self.healthy = False
                self.durable = False
                self.last_error = repr(exc)
            finally:
                try:
                    self._handle.close()
                except Exception as exc:
                    self.healthy = False
                    self.durable = False
                    self.last_error = repr(exc)


class SessionCommandLog:
    """Adapter used by OlympeLiveBackend in place of its legacy single JSONL."""

    def __init__(self, session: "SessionLogs"):
        self._session = session
        self.path = session.directory / "commands.jsonl"

    @property
    def healthy(self) -> bool:
        return self._session.healthy

    @property
    def durable(self) -> bool:
        return self._session.durable

    def event(self, event: str, **fields: Any) -> None:
        self._session.command(event, **fields)
        if event in _INCIDENT_EVENTS:
            self._session.incident(event, **fields)

    def close(self) -> None:
        # Session summary is finalized by the owning UI/process after cleanup.
        return None


class SessionLogs:
    """Durable safety receipts and bounded, ordered diagnostic persistence.

    Commands/incidents return only after their own sink fsync succeeds.
    Localization/telemetry/deferred telemetry return queue admission, not disk
    success; failures and overflow immediately affect health and receive incident
    receipts. Offline readers must use wait_for_diagnostics or close. Close stops
    admission, drains the writer and final-syncs sinks before summary. The daemon
    writer permits launch's atexit cleanup to run; that cleanup explicitly joins
    the writer, so accepted records are not abandoned at normal interpreter exit.
    """

    REQUIRED_STREAMS = ("commands", "localization", "telemetry", "incidents", "trajectory")

    def __init__(self, directory: Path, mode: InterfaceMode):
        self.directory = directory
        self.mode = mode
        self._closed = False
        self._closing = False
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._close_lock = threading.Lock()
        self._inventory_lock = threading.Lock()
        self._active_inventory_writes = 0
        self._active_safety_writes = 0
        self._summary_error: str | None = None
        self._summary_write_failures = 0
        self._inventory_error: str | None = None
        self._runtime_cuda: dict[str, Any] = {
            "available": None,
            "status": "worker_not_reported",
            "runtime_version": None,
            "device_names": [],
        }
        self._runtime_cuda_error: str | None = None
        self._counts = {name: 0 for name in self.REQUIRED_STREAMS}
        self._write_failures = {name: 0 for name in self.REQUIRED_STREAMS}
        self._failure_reported: set[str] = set()
        self._latency_samples_ms: list[float] = []
        self._unresolved_incidents: set[str] = set()
        # One slot owns a source row and its trajectory mirror. Admission never
        # performs diagnostic I/O, and partial queue admission cannot lose a mirror.
        self._diagnostics: deque = deque()
        self._diagnostic_accepted = 0
        self._diagnostic_completed = 0
        self._diagnostic_dropped = {name: 0 for name in self.REQUIRED_STREAMS}
        self._drop_reported = dict(self._diagnostic_dropped)
        self._deferred_dropped = 0
        self._sinks = {
            name: _JsonlSink(
                directory / f"{name}.jsonl",
                sync_every=(
                    1 if name in {"commands", "incidents"} else _BULK_LOG_SYNC_EVERY
                ),
            )
            for name in self.REQUIRED_STREAMS
        }
        self.command_log = SessionCommandLog(self)
        self._writer = threading.Thread(
            target=self._write_diagnostics,
            name=f"session-diagnostics-{directory.name}",
            daemon=True,
        )
        self._writer.start()

    @classmethod
    def create(
        cls,
        log_root: str | Path,
        *,
        mode: InterfaceMode,
        manifest: dict[str, Any],
    ) -> "SessionLogs":
        root = Path(log_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = root / f"session_{stamp}_{mode.value}_{uuid.uuid4().hex[:8]}"
        directory.mkdir(mode=0o750)
        payload = {
            "schema_version": 1,
            "session_id": directory.name,
            "mode": mode.value,
            "created_utc": _utc_now(),
            "created_mono_ns": time.monotonic_ns(),
            "debug_contract": flight_debug_contract(),
            **manifest,
        }
        temp = directory / ".session_manifest.json.tmp"
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, directory / "session_manifest.json")
        return cls(directory, mode)

    @property
    def healthy(self) -> bool:
        return (
            not self._closed
            and not any(self._diagnostic_dropped.values())
            and not any(self._write_failures.values())
            and self._inventory_error is None
            and all(sink.healthy for sink in self._sinks.values())
        )

    @property
    def durable(self) -> bool:
        return (
            self._inventory_error is None
            and self._summary_error is None
            and all(sink.durable for sink in self._sinks.values())
        )

    def write_inventory(self, kind: str, inventory: dict[str, Any]) -> bool:
        """Synchronously persist one immutable hardware or video receipt.

        Inventory serialization never holds the producer/admission lock. Close
        waits for every admitted operation, including those queued on this lock.
        """
        with self._condition:
            if self._closing or self._closed:
                return False
            self._active_inventory_writes += 1
            self._condition.notify_all()
        try:
            with self._inventory_lock:
                return self._persist_inventory(kind, inventory)
        finally:
            with self._condition:
                self._active_inventory_writes -= 1
                self._condition.notify_all()

    def _persist_inventory(self, kind: str, inventory: dict[str, Any]) -> bool:
        if kind not in {"hardware", "video"}:
            raise ValueError("inventory kind must be 'hardware' or 'video'")
        path = self.directory / f"{kind}_inventory.json"
        try:
            if path.exists():
                return True
            payload = {
                "schema_version": 1,
                "session_id": self.directory.name,
                "kind": kind,
                "captured_utc": _utc_now(),
                "captured_mono_ns": time.monotonic_ns(),
                "inventory": dict(inventory),
            }
            temp = self.directory / f".{kind}_inventory.json.tmp"
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temp, path)
        except Exception as exc:
            with self._condition:
                self._inventory_error = repr(exc)
            return False
        return True

    def record_worker_cuda(self, cuda: Mapping[str, Any]) -> bool:
        """Replace the manifest's pending CUDA identity with the worker's measurement."""
        try:
            names = [str(name) for name in cuda["device_names"]]
            report = {
                "available": cuda["available"] if isinstance(cuda["available"], bool) else None,
                "status": "worker_reported",
                "source": "live_localizer_worker",
                "runtime_version": cuda.get("runtime_version"),
                "device_names": names,
            }
            if report["available"] is None:
                raise ValueError("worker CUDA availability must be boolean")
        except (KeyError, TypeError, ValueError) as exc:
            self._runtime_cuda_error = repr(exc)
            return False
        with self._condition:
            if self._closing or self._closed:
                return False
            self._active_inventory_writes += 1
            self._condition.notify_all()
        try:
            with self._inventory_lock:
                path = self.directory / "session_manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["runtime"] = {**(manifest.get("runtime") or {}), "cuda": report}
                temp = self.directory / ".session_manifest.json.tmp"
                temp.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temp, path)
                self._runtime_cuda = report
                self._runtime_cuda_error = None
            return True
        except Exception as exc:
            self._runtime_cuda_error = repr(exc)
            return False
        finally:
            with self._condition:
                self._active_inventory_writes -= 1
                self._condition.notify_all()

    def _write(self, stream: str, event: str, fields: dict[str, Any]) -> bool:
        """Safety streams return durable success; diagnostics return admission."""
        with self._condition:
            if self._closing or self._closed:
                return False
            # _record recursively copies/normalizes producer-owned collections.
            # The worker never observes subsequent mutations or restamps a row.
            row = _record(event, {"session_id": self.directory.name, **fields})
            if stream not in {"commands", "incidents"}:
                return self._enqueue_locked(stream, row)
            self._active_safety_writes += 1
            if stream == "incidents":
                self._track_incident_locked(row)
        try:
            ok = self._persist(stream, row)
            mirror = self._trajectory_row(stream, row)
            if mirror is not None:
                with self._condition:
                    self._enqueue_locked("trajectory", mirror)
            return ok
        finally:
            with self._condition:
                self._active_safety_writes -= 1
                self._condition.notify_all()

    def _enqueue_locked(self, stream: str, row: dict[str, Any], *, deferred=False) -> bool:
        if len(self._diagnostics) >= _DIAGNOSTIC_QUEUE_CAPACITY:
            self._diagnostic_dropped[stream] += 1
            if deferred:
                self._deferred_dropped += 1
            self._condition.notify_all()
            return False
        self._diagnostics.append((stream, row, self._trajectory_row(stream, row)))
        self._diagnostic_accepted += 1
        self._condition.notify_all()
        return True

    def _persist(self, stream: str, row: dict[str, Any], *, report=True) -> bool:
        # No session/admission lock surrounds write, flush or fsync. Safety
        # sinks have independent locks and never wait behind diagnostic sinks.
        sink = self._sinks[stream]
        ok = sink.write(row)
        with self._condition:
            if ok:
                self._counts[stream] += 1
                for key in ("e2e_submit_to_ui_ms", "e2e_ms", "wall_ms", "core_wall_ms"):
                    value = row.get(key)
                    if value is None:
                        continue
                    try:
                        value = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if math.isfinite(value) and value >= 0.0:
                        self._latency_samples_ms.append(value)
                        break
            else:
                self._write_failures[stream] += 1
            first_failure = not ok and report and stream not in self._failure_reported
            if first_failure:
                self._failure_reported.add(stream)
        if first_failure:
            self._diagnostic_incident(
                "diagnostic_write_failed",
                failed_stream=stream,
                failed_event=row["event"],
                last_error=sink.last_error,
            )
        return ok

    def _track_incident_locked(self, row: dict[str, Any]) -> None:
        event = str(row["event"])
        if row.get("resolved") is True:
            self._unresolved_incidents.discard(event)
        else:
            self._unresolved_incidents.add(event)

    def _diagnostic_incident(self, event: str, **fields: Any) -> None:
        row = _record(event, {"session_id": self.directory.name, **fields})
        with self._condition:
            self._track_incident_locked(row)
        # Internal reports also run during close-drain, after public admission
        # has stopped. A broken incident sink cannot recursively report itself.
        self._persist("incidents", row, report=False)
        if threading.current_thread() is self._writer:
            self._persist("trajectory", row)
        else:
            with self._condition:
                self._enqueue_locked("trajectory", row)

    def _trajectory_row(self, stream: str, row: dict[str, Any]) -> dict[str, Any] | None:
        event = row["event"]
        if stream == "localization" and event == "pose_result":
            # Compact pose evidence survives diagnostic-log retention. Keep
            # failed fixes too, so reconstruction never silently bridges loss.
            keys = (
                "t_utc", "t_mono_ns", "pose", "pose_raw", "success",
                "source_frame_stamp_mono", "pose_capture_mono_ns", "display_seq",
                "mode", "direct_status", "pose_status", "confidence_low",
                "imu_bridge", "map_inliers", "vo_inliers", "inliers",
                "reproj_rms", "map_constraint_age_s", "reseed_confirming",
                "camera_axes_world", "camera_forward_world",
                "camera_attitude_kind",
                "camera_pitch_error_deg", "camera_roll_error_deg", "pnp_observation_sample",
                "n_corr", "inlier_ratio", "inlier_grid_cells", "inlier_hull_coverage",
                "positive_depth_ratio", "reproj_p90", "imu_bridge_reason",
                "frame_name", "hold_retry", "hold_kind", "fused_telemetry_mono",
                "fused_attitude_mono", "fused_speed_mono", "fused_altitude_mono",
                "fused_roll", "fused_pitch", "fused_yaw", "fused_speed_north",
                "fused_speed_east", "fused_speed_down", "fused_altitude_m",
                "fused_agl_altitude_m", "fused_stamp_source", "fused_gps_mono",
                "client_submit_mono_ns", "worker_core_start_mono_ns",
                "worker_core_done_mono_ns", "ui_arrival_mono_ns",
                "frame_callback_enter_mono_ns", "source_stamp_age_at_ui_ms",
                "restart_reason", "relocalize_requested", "reloc_capture_stamp_mono",
                "reloc_source_epoch", "reloc_ordinal", "reloc_ms", "reloc_retrieval_ms",
                "reloc_match_lift_ms", "reloc_edm_match_ms", "reloc_lift_prepare_ms", "reloc_pnp_ms",
                "source_ntp_timestamp_us", "source_timestamp_kind", "source_to_host_offset_s",
                "frame_receipt_mono_ns", "gimbal_pitch_deg", "gimbal_yaw_deg", "gimbal_roll_deg",
                "gimbal_relative_pitch_deg", "gimbal_relative_yaw_deg", "gimbal_relative_roll_deg",
                "gimbal_mono_ns", "gimbal_stamp_source", "zoom", "flight_state", "control_owner",
                "speed_stamp_source", "altitude_stamp_source",
                "camera_zoom_level", "camera_zoom_mono_ns", "camera_zoom_stamp_source",
                "pose_filter", "prediction_valid", "prediction_mode", "prediction_source",
                "esekf_pos_trace", "esekf_d2", "esekf_update_accepted",
                "esekf_update_exceptions", "error", "auto_run_id",
            ) + DIRECT_INFO_FIELDS
            return {
                "session_id": row["session_id"],
                "event": event,
                **{key: row[key] for key in keys if key in row},
            }
        elif (stream == "localization" and event in {
                  "auto_route_plan", "auto_route_tick", "auto_route_error_summary"}
              or stream == "incidents" and event in {
                  "diagnostic_write_failed", "diagnostic_records_dropped"}
              or stream == "telemetry" and event in {
                  "autonomy_event", "fused_odometry", "pcmd_dispatch", "stick_axes",
                  "control_motion_history"}):
            return row
        return None

    def _write_diagnostics(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._diagnostics
                    or self._diagnostic_dropped != self._drop_reported
                    or (self._closing and self._active_safety_writes == 0)
                )
                drops = {
                    name: (total - self._drop_reported[name], total)
                    for name, total in self._diagnostic_dropped.items()
                    if total != self._drop_reported[name]
                }
                batch = self._diagnostics.popleft() if self._diagnostics else None
                if batch is None and not drops and self._closing:
                    return
            for stream, (delta, total) in drops.items():
                self._diagnostic_incident(
                    "diagnostic_records_dropped",
                    stream=stream,
                    queue_capacity=_DIAGNOSTIC_QUEUE_CAPACITY,
                    dropped_since_marker=delta,
                    dropped_total=total,
                )
                with self._condition:
                    self._drop_reported[stream] = total
                    self._condition.notify_all()
            if batch is not None:
                stream, row, mirror = batch
                self._persist(stream, row)
                if mirror is not None:
                    self._persist("trajectory", mirror)
                with self._condition:
                    self._diagnostic_completed += 1
                    self._condition.notify_all()

    def wait_for_diagnostics(self, timeout: float | None = None) -> bool:
        """Wait for rows/loss markers admitted before this call (not an fsync).

        Explicit offline readback barrier; never call from the control/UI poll.
        Bulk fsync cadence remains unchanged; close provides final durability.
        """
        with self._condition:
            accepted = self._diagnostic_accepted
            dropped = dict(self._diagnostic_dropped)
            return self._condition.wait_for(
                lambda: self._diagnostic_completed >= accepted
                and all(self._drop_reported[name] >= total for name, total in dropped.items()),
                timeout=timeout,
            )

    def command(self, event: str, **fields: Any) -> bool:
        return self._write("commands", event, fields)

    def localization(self, event: str, **fields: Any) -> bool:
        return self._write("localization", event, fields)

    def telemetry(self, event: str, **fields: Any) -> bool:
        return self._write("telemetry", event, fields)

    def defer_telemetry(self, event: str, **fields: Any) -> bool:
        """Admit SDK diagnostics to the same ordered, bounded background writer."""
        with self._condition:
            if self._closing or self._closed:
                return False
            row = _record(event, {"session_id": self.directory.name, **fields})
            return self._enqueue_locked("telemetry", row, deferred=True)

    def incident(self, event: str, **fields: Any) -> bool:
        return self._write("incidents", event, fields)

    def close(self, *, reason: str, extra: dict[str, Any] | None = None) -> None:
        # Serialize closers, not producers. The writer may update health and
        # persist incidents during drain, so join must not hold the session lock.
        with self._close_lock:
            with self._condition:
                if self._closed:
                    retry_summary = self._summary_error is not None
                else:
                    retry_summary = False
                    self._closing = True
                    self._condition.notify_all()
            if self._closed:
                if retry_summary:
                    self._write_summary(reason=reason, extra=extra)
                return
            self._writer.join()
            with self._condition:
                self._condition.wait_for(
                    lambda: self._active_safety_writes == 0
                    and self._active_inventory_writes == 0
                )
            # Capture final fsync/close failures in the permanent summary.
            for sink in self._sinks.values():
                sink.close()
            with self._condition:
                self._closed = True
            self._write_summary(reason=reason, extra=extra)

    def _write_summary(self, *, reason: str, extra: dict[str, Any] | None) -> None:
        samples = sorted(self._latency_samples_ms)
        p95 = None
        maximum = None
        if samples:
            p95 = samples[min(len(samples) - 1, max(0, math.ceil(0.95 * len(samples)) - 1))]
            maximum = samples[-1]
        summary = {
            **(extra or {}),
            "schema_version": 1,
            "session_id": self.directory.name,
            "mode": self.mode.value,
            "reason": reason,
            "closed_utc": _utc_now(),
            "closed_mono_ns": time.monotonic_ns(),
            "event_counts": dict(self._counts),
            "debug_deferred_dropped": self._deferred_dropped,
            "diagnostic_dropped": dict(self._diagnostic_dropped),
            "write_failures": dict(self._write_failures),
            "summary_write_failures": self._summary_write_failures,
            "recording_complete": (
                not any(self._diagnostic_dropped.values())
                and not any(self._write_failures.values())
                and self._inventory_error is None
                and all(sink.healthy and sink.durable for sink in self._sinks.values())
            ),
            "inventory_error": self._inventory_error,
            "runtime_cuda": dict(self._runtime_cuda),
            "runtime_cuda_error": self._runtime_cuda_error,
            "metrics": {
                "p95_ms": p95,
                "max_ms": maximum,
                "sample_count": len(samples),
            },
            "p95_ms": p95,
            "max_ms": maximum,
            "unresolved": {
                "count": len(self._unresolved_incidents),
                "events": sorted(self._unresolved_incidents),
            },
            "log_health": {
                name: {
                    "healthy": sink.healthy,
                    "last_error": sink.last_error,
                    "durable": sink.durable,
                }
                for name, sink in self._sinks.items()
            },
        }
        temp = self.directory / ".session_summary.json.tmp"
        try:
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(_finite_json(summary), ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.directory / "session_summary.json")
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._summary_error = None
        except Exception as exc:
            self._summary_error = repr(exc)
            self._summary_write_failures += 1
            print(f"[session-log] summary persistence failed: {exc!r}", file=sys.stderr, flush=True)
            raise


__all__ = [
    "SessionCommandLog",
    "SessionLogs",
    "collect_runtime_identity",
]
