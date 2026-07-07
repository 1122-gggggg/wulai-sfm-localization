#!/usr/bin/env python3
"""Single-process desktop flight operator app.

No browser, no localhost server, no network API. The backend class is the single
place to connect Olympe/localization/control later.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import select
import subprocess
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import tkinter as tk
from tkinter import ttk


def find_system_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "sfm_system":
            return p
    env = os.environ.get("SFM_SYSTEM_ROOT")
    if env:
        return Path(env)
    raise SystemExit(f"could not locate sfm_system root from {start}; set SFM_SYSTEM_ROOT")


def default_worker_python(env_var: str, legacy: str) -> str:
    """Interpreter for spawned workers (needs torch/pycolmap/cv2).

    Priority: explicit env var > the original machine's known-good interpreter
    (if it still exists) > whatever python is running this app.
    """
    from_env = os.environ.get(env_var, "")
    if from_env:
        return from_env
    if Path(legacy).exists():
        return legacy
    return sys.executable


SYSTEM_ROOT = find_system_root(Path(__file__).resolve())
DEFAULT_MAP = SYSTEM_ROOT / "定位" / "maps" / "current_realrgb_v3.ply"
_DRAWN_ROUTE = SYSTEM_ROOT / "定位" / "mission" / "outputs" / "drawn_safezone" / "flight_path.json"
_CUR_ROUTE = SYSTEM_ROOT / "定位" / "mission" / "outputs" / "current_safezone" / "flight_path.json"
DEFAULT_ROUTE = _DRAWN_ROUTE if _DRAWN_ROUTE.exists() else _CUR_ROUTE
DEFAULT_VIDEO = Path("/home/cihcilab/Downloads/P0230023.MP4")
DEFAULT_REPLAY_JSON = (
    SYSTEM_ROOT / "定位" / "outputs" / "downloads_validation_20260702" / "P0230023_v3_temporal.json"
)
DEFAULT_WORKER = SYSTEM_ROOT / "定位" / "mission" / "operator_interface" / "live_localizer_worker.py"
DEFAULT_DETECTOR_WORKER = SYSTEM_ROOT / "定位" / "mission" / "operator_interface" / "object_detector_worker.py"
DEFAULT_BUNDLE = SYSTEM_ROOT / "定位" / "bundles" / "current_reloc_map_updated_v3.pt"
DEFAULT_MEGALOC = ""
DEFAULT_DETECTOR_MODEL = (
    SYSTEM_ROOT / "定位" / "object_detection" / "models" / "power_equipment_yolo26n_640_fp16.engine"
)
LIVE_STATUS_PATH = Path("/tmp/sfm_flight_operator_live_status.json")
LIVE_DETECTION_STATUS_PATH = Path("/tmp/sfm_flight_operator_detection_status.json")
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
ROUTE_COLOR = "#ff3ea5"          # planned drawn route overlay (distinct from the flown track)
POSE_JUMP_U = float(os.environ.get("SFM_MAX_POSE_JUMP_U", "1.5"))  # reject fixes that break trajectory continuity
LOC_LOW_INLIERS = int(os.environ.get("SFM_LOW_CONF_INLIERS", "60"))    # below -> low-confidence alert
LOC_HIGH_REPROJ = float(os.environ.get("SFM_LOC_HIGH_REPROJ", "4.0"))  # above -> low-confidence alert
HEALTH_COLOR = {"OK": "#3fbf7f", "LOW": "#e0a92e", "FAIL": "#e2483d"}
NO_LOC_DEDUP_U = float(os.environ.get("SFM_NO_LOC_DEDUP_U", "1.0"))  # merge no-loc markers within this


def load_route_glomap(path_json: str) -> list:
    """Load a Blender-drawn flight path (aligned frame) and convert each waypoint to the
    GLOMAP map frame used by the point cloud / localizer: aligned(x,y,z) -> (x, -z, y)."""
    try:
        data = json.loads(Path(path_json).read_text(encoding="utf-8"))
    except Exception:
        return []
    return [np.array([float(p[0]), -float(p[2]), float(p[1])], dtype=float)
            for p in data.get("waypoints", []) if len(p) >= 3]
DEFAULT_MAP_YAW = 0.0
DEFAULT_MAP_PITCH = (math.radians(78.0) + math.pi) % (2.0 * math.pi)
DEFAULT_MAP_ROLL = 0.0
DEFAULT_MAP_ZOOM = 3.2


@dataclass(frozen=True)
class AnafiProfile:
    model: str = "Parrot ANAFI"
    weight_g: int = 320
    max_horizontal_speed_mps: float = 15.0
    max_vertical_speed_mps: float = 4.0
    max_yaw_rate_dps: float = 200.0
    max_wind_kmh: float = 50.0
    flight_time_s: float = 25.0 * 60.0
    takeoff_hover_m: float = 1.0
    gimbal_pitch_min_deg: float = -90.0
    gimbal_pitch_max_deg: float = 90.0
    gimbal_pitch_rate_dps: float = 180.0
    stream_width: int = STREAM_WIDTH
    stream_height: int = STREAM_HEIGHT
    stream_fps: float = 30.0
    stream_latency_ms: float = 280.0
    stream_mbps: float = 5.0
    video_hfov_deg: float = 69.0
    digital_zoom_max: float = 3.0
    lossless_zoom_fhd: float = 2.8


ANAFI = AnafiProfile()


@dataclass
class DroneState:
    mode: str = "MANUAL"
    tracker_state: str = "HOVER"
    loc: str = "SIM"
    stream: str = "WAIT"
    pose: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))  # x,y,z,yaw
    inliers: int = 0
    reproj: float | None = None
    battery_pct: float = 100.0
    altitude_m: float = 0.0
    gimbal_pitch_deg: float = -20.0
    zoom: float = 1.0
    link_latency_ms: float = ANAFI.stream_latency_ms
    stream_fps: float = ANAFI.stream_fps
    stream_mbps: float = ANAFI.stream_mbps
    last_command: str = "ready"


class DroneBackend:
    """Stub backend. Replace methods here with real Olympe/localizer calls."""

    def __init__(self):
        self.state = DroneState()
        self.start = time.monotonic()
        self.last_poll = self.start
        self.sim_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
        self.sim_yaw = 0.0
        self.target_altitude_m = 0.0
        self.flight_start: float | None = None

    def command(self, name: str, **payload) -> DroneState:
        self.state.last_command = name
        if name in {"manual", "hover", "land"}:
            self.state.mode = "MANUAL"
        elif name in {"auto", "start_auto"}:
            self.state.mode = "AUTO"
            self.target_altitude_m = max(self.target_altitude_m, ANAFI.takeoff_hover_m)
            if self.flight_start is None:
                self.flight_start = time.monotonic()
        if name == "hover":
            self.state.tracker_state = "HOVER"
            self.target_altitude_m = max(0.0, -float(self.sim_xyz[1]))
        elif name == "land":
            self.state.tracker_state = "LAND"
            self.target_altitude_m = 0.0
        elif name == "takeoff":
            self.state.tracker_state = "TAKEOFF"
            self.target_altitude_m = ANAFI.takeoff_hover_m
            if self.flight_start is None:
                self.flight_start = time.monotonic()
        elif name == "boot_lock":
            self.state.tracker_state = "BOOT_INIT"
        elif name == "gimbal_pitch":
            self.state.gimbal_pitch_deg = float(np.clip(
                float(payload.get("pitch", self.state.gimbal_pitch_deg)),
                ANAFI.gimbal_pitch_min_deg,
                ANAFI.gimbal_pitch_max_deg,
            ))
        elif name == "zoom":
            self.state.zoom = float(np.clip(
                float(payload.get("zoom", self.state.zoom)),
                1.0,
                ANAFI.digital_zoom_max,
            ))
        return self.state

    def stream_lost_hover(self, detail: str = "stream lost") -> DroneState:
        self.state.mode = "MANUAL"
        self.state.tracker_state = "STREAM_LOST_HOVER"
        self.state.loc = "STREAM_LOST"
        self.state.stream = "LOST"
        self.state.last_command = f"hover: {detail}"
        return self.state

    def poll(self) -> DroneState:
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self.last_poll))
        self.last_poll = now
        t = now - self.start

        if self.state.tracker_state == "LAND":
            self.target_altitude_m = 0.0
        elif self.state.mode == "AUTO":
            self.target_altitude_m = max(self.target_altitude_m, ANAFI.takeoff_hover_m)

        target_y = -self.target_altitude_m
        dy = float(target_y - self.sim_xyz[1])
        max_dy = ANAFI.max_vertical_speed_mps * dt
        if abs(dy) > max_dy and max_dy > 0:
            dy = math.copysign(max_dy, dy)
        self.sim_xyz[1] += dy

        if self.state.mode == "AUTO":
            orbit_r = 2.0
            cruise_speed = 1.5
            omega = cruise_speed / orbit_r
            target = np.array([
                math.sin(t * omega) * orbit_r,
                self.sim_xyz[1],
                math.cos(t * omega) * orbit_r,
            ], dtype=float)
            delta = target - self.sim_xyz
            delta[1] = 0.0
            dist = float(np.linalg.norm(delta))
            max_step = ANAFI.max_horizontal_speed_mps * dt
            if dist > 1e-6:
                self.sim_xyz += delta / dist * min(dist, max_step)
                target_yaw = math.atan2(float(delta[2]), float(delta[0]))
            else:
                target_yaw = self.sim_yaw
            yaw_delta = (target_yaw - self.sim_yaw + math.pi) % (2.0 * math.pi) - math.pi
            max_yaw = math.radians(ANAFI.max_yaw_rate_dps) * dt
            if abs(yaw_delta) > max_yaw and max_yaw > 0:
                yaw_delta = math.copysign(max_yaw, yaw_delta)
            self.sim_yaw = (self.sim_yaw + yaw_delta + math.pi) % (2.0 * math.pi) - math.pi

        self.state.pose[:] = [self.sim_xyz[0], self.sim_xyz[1], self.sim_xyz[2], self.sim_yaw]
        self.state.loc = "SIM"
        if self.state.stream == "LOST":
            self.state.tracker_state = "STREAM_LOST_HOVER"
            return self.state
        if self.state.mode == "AUTO":
            self.state.tracker_state = "TRACK"
        elif self.state.tracker_state == "TAKEOFF" and abs(self.sim_xyz[1] + ANAFI.takeoff_hover_m) < 0.03:
            self.state.tracker_state = "HOVER"
        elif self.state.tracker_state == "LAND" and abs(self.sim_xyz[1]) < 0.03:
            self.sim_xyz[1] = 0.0
            self.target_altitude_m = 0.0
            self.state.tracker_state = "HOVER"
        elif self.state.tracker_state not in {"LAND", "TAKEOFF", "BOOT_INIT"}:
            self.state.tracker_state = "HOVER"
        self.state.inliers = 180 + int(40 * math.sin(t))
        self.state.reproj = 2.5 + 0.2 * math.cos(t * 0.5)
        elapsed_flight = 0.0 if self.flight_start is None else max(0.0, now - self.flight_start)
        self.state.battery_pct = float(np.clip(100.0 * (1.0 - elapsed_flight / ANAFI.flight_time_s), 0.0, 100.0))
        self.state.altitude_m = max(0.0, -float(self.sim_xyz[1]))
        self.state.link_latency_ms = ANAFI.stream_latency_ms
        self.state.stream_fps = ANAFI.stream_fps
        self.state.stream_mbps = ANAFI.stream_mbps
        return self.state


class FFmpegFrameStream:
    def __init__(self, video_path: Path, width: int, height: int, stride: int = 1,
                 fps: float = ANAFI.stream_fps, loop: bool = True):
        self.video_path = Path(video_path)
        self.width = int(width)
        self.height = int(height)
        self.stride = max(1, int(stride))
        self.fps = float(fps)
        self.loop = bool(loop)
        self.proc: subprocess.Popen | None = None
        self.frame_size = self.width * self.height * 3
        self.output_index = 0
        self.last_frame_name = ""
        if self.video_path.exists():
            self.start()

    def start(self) -> None:
        self.close()
        self.output_index = 0
        self.last_frame_name = ""
        if self.stride > 1:
            vf = f"select=not(mod(n\\,{self.stride})),scale={self.width}:{self.height}"
        else:
            vf = f"fps={self.fps:g},scale={self.width}:{self.height}"
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

    def close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def next_frame(self) -> Image.Image | None:
        if self.proc is None or self.proc.stdout is None:
            return None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            if self.loop:
                self.start()
                if self.proc is None or self.proc.stdout is None:
                    return None
                raw = self.proc.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                return None
        self.output_index += 1
        self.last_frame_name = f"frame_{self.output_index:06d}.jpg"
        return Image.frombytes("RGB", (self.width, self.height), raw)


class LiveWorkerClient:
    """Shared queue/subprocess plumbing for the worker clients.

    Subclasses supply the worker argv, stderr log path, thread name, a short label
    for error messages, and any extra keys to include in error-result payloads.
    """

    def __init__(self, cmd: list, width: int, height: int, log_path: str,
                 thread_name: str, label: str, error_defaults: dict | None = None,
                 timeout_s: float = 8.0):
        self.width = int(width)
        self.height = int(height)
        self.label = label
        self.error_defaults = error_defaults or {}
        self.cmd = cmd
        self.log_path = log_path
        self.timeout_s = float(os.environ.get("SFM_WORKER_TIMEOUT_S", timeout_s))
        self.restart_warmup_s = 20.0     # after a restart, skip submits while the worker reloads models
        self._last_restart = 0.0
        self.pending: queue.Queue[tuple[int, str, bytes]] = queue.Queue(maxsize=1)
        self.results: queue.Queue[dict] = queue.Queue()
        self.in_flight = False
        self._lock = threading.Lock()
        self.proc = self._spawn("w")
        self.thread = threading.Thread(target=self._loop, name=thread_name, daemon=True)
        self.thread.start()

    def _spawn(self, mode: str) -> subprocess.Popen:
        return subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(self.log_path, mode, encoding="utf-8"), bufsize=0)

    def _restart(self) -> None:
        """Respawn a hung/exited worker so localization can recover. Cooldown-guarded so a
        freshly-restarted worker (which needs ~15s to reload models) is not thrashed."""
        now = time.monotonic()
        if now - self._last_restart < 30.0:
            return
        self._last_restart = now
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc = self._spawn("a")
            print(f"[operator] {self.label} worker restarted after stall/exit", flush=True)
        except Exception as exc:
            print(f"[operator] {self.label} worker restart failed: {exc!r}", flush=True)

    def _error_result(self, seq: int, frame_name: str, error: str) -> dict:
        payload = {
            "display_seq": seq,
            "frame_name": frame_name,
            "success": False,
        }
        payload.update(self.error_defaults)
        payload["error"] = error
        return payload

    def _loop(self) -> None:
        while True:
            item = self.pending.get()
            if item is None:
                return
            seq, frame_name, raw = item
            with self._lock:
                self.in_flight = True
            try:
                if self.proc.stdin is None or self.proc.stdout is None:
                    raise RuntimeError(f"{self.label} worker pipe closed")
                self.proc.stdin.write(raw)
                self.proc.stdin.flush()
                # bound the read: a hung worker (e.g. GPU stall in localize) must not block
                # this thread forever (would freeze localization on the last result).
                ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout_s)
                if not ready:
                    raise TimeoutError(f"{self.label} worker stalled > {self.timeout_s:.0f}s")
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"{self.label} worker exited")
                payload = json.loads(line.decode("utf-8"))
                payload["display_seq"] = seq
                payload["frame_name"] = frame_name
                self.results.put(payload)
            except (TimeoutError, RuntimeError, BrokenPipeError, OSError) as exc:
                self.results.put(self._error_result(seq, frame_name, repr(exc)))
                self._restart()                      # hung/exited worker -> respawn (cooldown-guarded)
            except Exception as exc:
                self.results.put(self._error_result(seq, frame_name, repr(exc)))
            finally:
                with self._lock:
                    self.in_flight = False

    def submit(self, seq: int, frame_name: str, frame: Image.Image) -> bool:
        if time.monotonic() - self._last_restart < self.restart_warmup_s:
            return False                             # worker reloading models after a restart
        if self.proc.poll() is not None:
            self.results.put(self._error_result(
                seq, frame_name, f"{self.label} worker exited code={self.proc.returncode}"))
            return False
        if not self.pending.empty():
            return False
        self.pending.put((seq, frame_name, frame.tobytes()))
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
        with self._lock:
            return self.in_flight or not self.pending.empty()

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=1.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class LiveLocalizerClient(LiveWorkerClient):
    def __init__(self, worker_py: Path, python_bin: str, width: int, height: int,
                 bundle: Path, megaloc_cache: str | Path = ""):
        cmd = [
            str(python_bin), str(worker_py),
            "--width", str(width),
            "--height", str(height),
            "--bundle", str(bundle),
        ]
        if str(megaloc_cache):
            cmd.extend(["--megaloc-cache", str(megaloc_cache)])
        super().__init__(cmd, width, height,
                         "/tmp/sfm_live_localizer_worker.log",
                         "live-localizer-client", "localizer")


class LiveDetectorClient(LiveWorkerClient):
    def __init__(self, worker_py: Path, python_bin: str, width: int, height: int,
                 model: Path, imgsz: int = 640, conf: float = 0.25,
                 iou: float = 0.7, max_det: int = 300):
        cmd = [
            str(python_bin), str(worker_py),
            "--width", str(width),
            "--height", str(height),
            "--model", str(model),
            "--imgsz", str(int(imgsz)),
            "--conf", str(float(conf)),
            "--iou", str(float(iou)),
            "--max-det", str(int(max_det)),
        ]
        super().__init__(cmd, width, height,
                         "/tmp/sfm_live_detector_worker.log",
                         "live-detector-client", "detector",
                         error_defaults={"count": 0, "boxes": []})


def read_ply_points(path: Path, max_points: int) -> np.ndarray:
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY ended before end_header")
            text = line.decode("ascii", "replace").strip()
            header.append(text)
            if text == "end_header":
                break
        fmt = next((h.split()[1] for h in header if h.startswith("format ")), "")
        vertex_line = next((h for h in header if h.startswith("element vertex ")), "")
        n_vertices = int(vertex_line.split()[-1]) if vertex_line else 0
        step = max(1, n_vertices // max(1, int(max_points)))

        points = []
        if fmt == "binary_little_endian":
            rec = struct.Struct("<fffBBB")
            for i in range(n_vertices):
                raw = f.read(rec.size)
                if len(raw) != rec.size:
                    break
                if i % step != 0:
                    continue
                x, y, z, r, g, b = rec.unpack(raw)
                points.append((x, y, z, r, g, b))
        elif fmt == "ascii":
            for i in range(n_vertices):
                raw = f.readline()
                if not raw:
                    break
                if i % step != 0:
                    continue
                vals = raw.split()
                if len(vals) >= 6:
                    points.append((
                        float(vals[0]), float(vals[1]), float(vals[2]),
                        int(vals[3]), int(vals[4]), int(vals[5]),
                    ))
        else:
            raise ValueError(f"unsupported PLY format: {fmt}")
    return np.asarray(points, dtype=np.float32)


class OperatorApp(tk.Tk):
    def __init__(self, backend: DroneBackend, map_points: np.ndarray,
                 video_stream: FFmpegFrameStream | None = None,
                 localizer: LiveLocalizerClient | None = None,
                 detector: LiveDetectorClient | None = None,
                 replay_rows: list[dict] | None = None,
                 tick_ms: int = 200,
                 boot_lock_ms: int = 2500,
                 detect_every_n_frames: int = 3):
        super().__init__()
        self.backend = backend
        self.map_points = map_points
        self.video_stream = video_stream
        self.localizer = localizer
        self.detector = detector
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.video_frame: Image.Image | None = None
        self.video_frame_fresh = False
        self.video_display_index = -1
        self.video_display_frame_name = ""
        self.live_result_frame_name = ""
        self.live_pending_frame_name = ""
        self.live_result: dict | None = None
        self.live_result_times: list[float] = []
        self.detection_result_frame_name = ""
        self.detection_pending_frame_name = ""
        self.detection_result: dict | None = None
        self.detection_result_times: list[float] = []
        self.det_fps = 0.0
        self.det_core_fps = 0.0
        self.det_latency_ms: float | None = None
        self.det_count = 0
        self.det_status = "OFF"
        self.loc_fps = 0.0
        self.loc_core_fps = 0.0
        self.loc_latency_ms: float | None = None
        self.loc_stage = "-"
        self._last_status_write = 0.0          # throttle debug status files to ~5Hz
        self._last_det_write = 0.0
        self.loc_health = "OK"                 # OK / LOW / FAIL — operator localization alert
        self.loc_health_inliers = 0
        self.loc_health_reproj = None
        self.history_health: list[str] = []    # per-history-point health, for map markers
        # 巡檢 gate: the 720p stream is only fed to the localizer AFTER 開始巡檢 is
        # pressed. Overall FPS = frames actually processed since inspection started.
        self.inspecting = False
        self.inspect_start: float | None = None
        self.processed_frames = 0
        self.overall_fps = 0.0
        self.live_last_xyz: np.ndarray | None = None
        self._live_pending: np.ndarray | None = None   # continuity gate: jump awaiting a confirming fix
        self.live_heading: float | None = None
        self.live_pose = np.array([0.0, 0.0, 0.0, np.nan], dtype=float)
        self.live_locked = False
        self.live_new_pose = False
        self.last_submitted_index = -1
        self.last_detect_submitted_index = -1
        self.stream_lost_since: float | None = None
        self.replay_rows = replay_rows or []
        self.replay_headings = self._derive_motion_headings(self.replay_rows)
        self.replay_index = 0
        self.replay_last_pose = np.zeros(4, dtype=float)
        self.tick_ms = max(10, int(tick_ms))
        self.boot_lock_s = max(0.0, float(boot_lock_ms) / 1000.0)
        self.boot_lock_start: float | None = None
        self.boot_lock_done = self.boot_lock_s <= 0.0
        self.stream_period_s = 1.0 / ANAFI.stream_fps
        self.next_stream_frame_time = 0.0
        self.history: list[np.ndarray] = []
        self.route_pts: list = []          # planned drawn route (GLOMAP frame), overlay
        self.current_state = self.backend.state
        self.base_map_image = None
        self.map_photo = None
        self.video_photo = None
        self.map_base_cache_key = None
        self.map_base_cache: Image.Image | None = None
        self._map_dirty_key = None       # skip map re-render+PhotoImage when nothing drawn changed
        self._video_dirty_key = None     # skip video re-render+PhotoImage when the frame/overlays are unchanged
        self.default_map_center, self.map_radius = self._map_view_bounds(map_points)
        # Permanent no-localization markers: each time localization fails, drop a red dot at the
        # last-known position (deduped within NO_LOC_DEDUP_U) so the operator builds a persistent
        # map of where localization can't hold. Never capped/cleared.
        self.no_loc_markers: list[np.ndarray] = []
        self.map_center = self.default_map_center.copy()
        self.pivot_point = self.map_center.copy()
        self.map_yaw = DEFAULT_MAP_YAW
        self.map_pitch = DEFAULT_MAP_PITCH
        self.map_roll = DEFAULT_MAP_ROLL
        self.map_zoom = DEFAULT_MAP_ZOOM
        self.map_pan = np.zeros(2, dtype=float)
        self._drag_button: int | None = None
        self._drag_last: tuple[int, int] | None = None
        self.title("SfM Flight Operator - Parrot ANAFI profile")
        self.geometry("1440x900")
        self.minsize(980, 640)
        self.configure(bg="#111316")
        self._build_ui()
        self.bind("<space>", lambda _e: self.send("hover"))
        self.bind("<Escape>", lambda _e: self.send("manual"))
        self.after(100, self.tick)

    def boot_holding(self) -> bool:
        return self.inspecting and not self.boot_lock_done

    def update_boot_lock(self) -> None:
        if self.boot_lock_done:
            return
        if not self.inspecting:                 # boot lock only engages after 開始巡檢 (AUTO consent)
            return
        now = time.monotonic()
        if self.boot_lock_start is None:
            self.boot_lock_start = now
            self.write_log("BOOT_INIT: holding first 720p frame until live MegaLoc/PnP lock")
        elif now - self.boot_lock_start >= self.boot_lock_s:
            self.boot_lock_done = True           # timeout: release the hold so the UI never freezes
            self.write_log(f"BOOT_INIT: no lock in {self.boot_lock_s:.1f}s; releasing hold "
                           "(localization keeps trying, stream resumes)")

    @staticmethod
    def _derive_motion_headings(rows: list[dict]) -> list[float | None]:
        headings: list[float | None] = [None] * len(rows)
        poses = [r.get("pose") for r in rows]
        last_heading: float | None = None
        for i, pose in enumerate(poses):
            if not pose:
                headings[i] = last_heading
                continue
            heading = None
            prev = None
            nxt = None
            for j in range(max(0, i - 12), i):
                if poses[j]:
                    prev = poses[j]
                    break
            for j in range(min(len(poses) - 1, i + 12), i, -1):
                if poses[j]:
                    nxt = poses[j]
                    break
            if prev is not None and nxt is not None:
                dx = float(nxt.get("x", 0.0)) - float(prev.get("x", 0.0))
                dz = float(nxt.get("z", 0.0)) - float(prev.get("z", 0.0))
                if math.hypot(dx, dz) >= 0.12:
                    heading = math.atan2(dz, dx)
            if heading is None and nxt is not None:
                dx = float(nxt.get("x", 0.0)) - float(pose.get("x", 0.0))
                dz = float(nxt.get("z", 0.0)) - float(pose.get("z", 0.0))
                if math.hypot(dx, dz) >= 0.12:
                    heading = math.atan2(dz, dx)
            if heading is None and prev is not None:
                dx = float(pose.get("x", 0.0)) - float(prev.get("x", 0.0))
                dz = float(pose.get("z", 0.0)) - float(prev.get("z", 0.0))
                if math.hypot(dx, dz) >= 0.12:
                    heading = math.atan2(dz, dx)
            if heading is not None:
                last_heading = heading
            headings[i] = last_heading
        first_heading = next((h for h in headings if h is not None), None)
        if first_heading is not None:
            for i, h in enumerate(headings):
                if h is not None:
                    break
                headings[i] = first_heading
        return headings

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#111316")
        style.configure("Panel.TFrame", background="#1a1d21")
        style.configure("TLabel", background="#111316", foreground="#f0f3f5")
        style.configure("TButton", padding=(10, 6))

        top = ttk.Frame(self, style="TFrame")
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="SfM Flight Operator | Parrot ANAFI", font=("Sans", 13, "bold")).pack(side="left")
        self.status = ttk.Label(top, text="MANUAL | WAIT | HOVER")
        self.status.pack(side="right")

        mid = ttk.Frame(self, style="TFrame")
        mid.pack(fill="both", expand=True, padx=10)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(0, weight=1)

        self.map_label = tk.Canvas(mid, bg="#15181c", highlightthickness=0, bd=0,
                                   width=640, height=420)
        self.map_label.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.map_label.bind("<ButtonPress-1>", self.on_map_press)
        self.map_label.bind("<B1-Motion>", self.on_map_drag)
        self.map_label.bind("<ButtonRelease-1>", self.on_map_release)
        self.map_label.bind("<Double-Button-1>", self.set_rotation_pivot)
        self.map_label.bind("<ButtonPress-3>", self.on_map_press)
        self.map_label.bind("<B3-Motion>", self.on_map_drag)
        self.map_label.bind("<ButtonRelease-3>", self.on_map_release)
        self.map_label.bind("<ButtonPress-2>", self.on_map_press)
        self.map_label.bind("<B2-Motion>", self.on_map_drag)
        self.map_label.bind("<ButtonRelease-2>", self.on_map_release)
        self.map_label.bind("<MouseWheel>", self.on_map_wheel)
        self.map_label.bind("<Button-4>", self.on_map_wheel)
        self.map_label.bind("<Button-5>", self.on_map_wheel)
        self.video_label = tk.Canvas(mid, bg="#08090b", highlightthickness=0, bd=0,
                                     width=640, height=420)
        self.video_label.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        bottom = ttk.Frame(self, style="Panel.TFrame")
        bottom.pack(fill="x", padx=10, pady=10)

        flight = ttk.LabelFrame(bottom, text="飛行模式")
        flight.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        mission = ttk.LabelFrame(bottom, text="任務控制")
        mission.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6, pady=6)
        camera = ttk.LabelFrame(bottom, text="鏡頭")
        camera.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        telemetry = ttk.LabelFrame(bottom, text="定位儀表")
        telemetry.grid(row=1, column=2, sticky="ew", padx=6, pady=(0, 6))
        anafi_panel = ttk.LabelFrame(bottom, text="ANAFI 模擬")
        anafi_panel.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        bottom.columnconfigure((0, 1, 2), weight=1)

        for text, cmd in [
            ("手動", "manual"), ("自動", "auto"), ("懸停", "hover"), ("原地降落", "land")
        ]:
            ttk.Button(flight, text=text, command=lambda c=cmd: self.send(c)).pack(side="left", padx=4, pady=8)
        for text, cmd in [
            ("起飛", "takeoff"), ("定位鎖定", "boot_lock"), ("開始巡檢", "start_auto"),
            ("暫停路徑", "pause"), ("繼續路徑", "resume")
        ]:
            ttk.Button(mission, text=text, command=lambda c=cmd: self.send(c)).pack(side="left", padx=4, pady=8)

        self.pitch = tk.DoubleVar(value=-20)
        self.zoom = tk.DoubleVar(value=1)
        ttk.Label(camera, text="俯仰").pack(side="left", padx=(6, 2))
        ttk.Scale(camera, from_=ANAFI.gimbal_pitch_min_deg, to=ANAFI.gimbal_pitch_max_deg, variable=self.pitch,
                  command=lambda _v: self.send("gimbal_pitch", pitch=self.pitch.get())).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Label(camera, text="縮放").pack(side="left", padx=(8, 2))
        ttk.Scale(camera, from_=1, to=ANAFI.digital_zoom_max, variable=self.zoom,
                  command=lambda _v: self.send("zoom", zoom=self.zoom.get())).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(camera, text="重設地圖", command=self.reset_map_view).pack(side="left", padx=4, pady=8)
        ttk.Button(camera, text="上下翻面", command=self.flip_map_vertical).pack(side="left", padx=4, pady=8)

        self.loc_health_label = ttk.Label(telemetry, text="定位 待命", font=("Sans", 14, "bold"))
        self.loc_health_label.pack(anchor="w", padx=8, pady=(6, 2))
        self.overall_fps_var = tk.StringVar(value="整體 FPS - (待命，按開始巡檢)")
        self.loc_fps_var = tk.StringVar(value="定位 FPS -")
        self.loc_latency_var = tk.StringVar(value="延遲 -")
        self.loc_quality_var = tk.StringVar(value="inliers - | reproj -")
        self.det_fps_var = tk.StringVar(value="偵測 FPS -")
        self.det_latency_var = tk.StringVar(value="YOLO 延遲 -")
        self.det_count_var = tk.StringVar(value="objects -")
        ttk.Label(telemetry, textvariable=self.overall_fps_var, font=("Sans", 11, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
        ttk.Label(telemetry, textvariable=self.loc_fps_var, font=("Sans", 10, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
        ttk.Label(telemetry, textvariable=self.loc_latency_var, font=("Sans", 10, "bold")).pack(anchor="w", padx=8)
        ttk.Label(telemetry, textvariable=self.loc_quality_var).pack(anchor="w", padx=8)
        ttk.Label(telemetry, textvariable=self.det_fps_var, font=("Sans", 10, "bold")).pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(telemetry, textvariable=self.det_latency_var, font=("Sans", 10, "bold")).pack(anchor="w", padx=8)
        ttk.Label(telemetry, textvariable=self.det_count_var).pack(anchor="w", padx=8, pady=(0, 5))

        self.anafi_flight_var = tk.StringVar(value="battery 100% | alt 0.0m | gimbal -20° | zoom 1.0x")
        self.anafi_stream_var = tk.StringVar(value="720p30 | 5 Mb/s | latency 280 ms | HFOV 69°")
        self.anafi_limit_var = tk.StringVar(value="limit H 15m/s | V 4m/s | yaw 200°/s | gimbal 180°")
        ttk.Label(anafi_panel, textvariable=self.anafi_flight_var, font=("Sans", 10, "bold")).pack(side="left", padx=(8, 18), pady=5)
        ttk.Label(anafi_panel, textvariable=self.anafi_stream_var).pack(side="left", padx=(0, 18), pady=5)
        ttk.Label(anafi_panel, textvariable=self.anafi_limit_var).pack(side="left", padx=(0, 8), pady=5)

        self.log = tk.Text(bottom, height=4, bg="#15181c", fg="#a7b0b8", insertbackground="#f0f3f5")
        self.log.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        self.write_log("ready")

    def send(self, command: str, **payload) -> None:
        if command in {"manual", "hover", "land"}:
            self.stream_lost_since = None
            self.backend.state.stream = "MANUAL"
        if command == "start_auto" and not self.inspecting:
            # 開始巡檢: begin feeding the 720p stream to the localizer and start FPS timing.
            self.inspecting = True
            self.inspect_start = time.monotonic()
            self.processed_frames = 0
            self.overall_fps = 0.0
            self.next_stream_frame_time = 0.0
            self.write_log("開始巡檢：串流輸入 + 定位啟動")
        self.backend.command(command, **payload)
        self.write_log(command)

    def write_log(self, text: str) -> None:
        self.log.insert("1.0", f"{time.strftime('%H:%M:%S')} {text}\n")

    def _record_no_loc(self) -> None:
        """Drop a permanent red marker at the last-known position when localization fails
        (deduped within NO_LOC_DEDUP_U so a sustained failure = one dot, not thousands)."""
        p = self.live_last_xyz
        if p is None:
            return
        for q in self.no_loc_markers:
            if float(np.linalg.norm(p - q)) <= NO_LOC_DEDUP_U:
                return
        self.no_loc_markers.append(np.asarray(p, dtype=float).copy())

    def update_localization_metrics(self, result: dict) -> None:
        now = time.monotonic()
        self.live_result_times.append(now)
        cutoff = now - 5.0
        self.live_result_times = [t for t in self.live_result_times[-80:] if t >= cutoff]
        if len(self.live_result_times) >= 2:
            span = self.live_result_times[-1] - self.live_result_times[0]
            self.loc_fps = (len(self.live_result_times) - 1) / span if span > 0 else 0.0
        wall_ms = result.get("wall_ms")
        self.loc_latency_ms = float(wall_ms) if wall_ms is not None else None
        self.loc_core_fps = 1000.0 / self.loc_latency_ms if self.loc_latency_ms and self.loc_latency_ms > 0 else 0.0
        self.loc_stage = str(
            result.get("composite_stage")
            or result.get("next_mode")
            or result.get("mode")
            or ("FAIL" if not result.get("success") else "-")
        )

        inliers = int(result.get("inliers", 0) or 0)
        reproj = result.get("reproj_rms")
        latency_text = "-" if self.loc_latency_ms is None else f"{self.loc_latency_ms:.1f} ms"
        reproj_text = "-" if reproj is None else f"{float(reproj):.2f}"
        # localization health for the operator alert (FAIL = no fix, LOW = weak confidence)
        weak = self.loc_stage.upper() in ("WEAK_TRACK", "LOST", "FAIL", "TRACK_FAIL", "TRACK_LOST")
        if not result.get("success"):
            self.loc_health = "FAIL"
            self._record_no_loc()
        elif inliers < LOC_LOW_INLIERS or (reproj is not None and reproj > LOC_HIGH_REPROJ) or weak:
            self.loc_health = "LOW"
        else:
            self.loc_health = "OK"
        self.loc_health_inliers, self.loc_health_reproj = inliers, reproj
        if hasattr(self, "loc_health_label"):
            txt = {"OK": "定位正常", "LOW": f"定位信心低 inliers={inliers}", "FAIL": "定位失敗"}[self.loc_health]
            self.loc_health_label.configure(text=txt, foreground=HEALTH_COLOR[self.loc_health])
        if hasattr(self, "loc_fps_var"):
            self.loc_fps_var.set(f"定位 FPS {self.loc_fps:.1f} | 核心 {self.loc_core_fps:.1f}")
            self.loc_latency_var.set(f"延遲 {latency_text}")
            self.loc_quality_var.set(f"inliers {inliers} | reproj {reproj_text} | {self.loc_stage}")

    def update_detection_metrics(self, result: dict) -> None:
        now = time.monotonic()
        self.detection_result_times.append(now)
        cutoff = now - 5.0
        self.detection_result_times = [t for t in self.detection_result_times[-80:] if t >= cutoff]
        if len(self.detection_result_times) >= 2:
            span = self.detection_result_times[-1] - self.detection_result_times[0]
            self.det_fps = (len(self.detection_result_times) - 1) / span if span > 0 else 0.0
        wall_ms = result.get("wall_ms")
        self.det_latency_ms = float(wall_ms) if wall_ms is not None else None
        self.det_core_fps = 1000.0 / self.det_latency_ms if self.det_latency_ms and self.det_latency_ms > 0 else 0.0
        self.det_count = int(result.get("count", 0) or 0)
        if not result.get("success"):
            self.det_status = "FAIL"
        elif self.detector is not None and self.detector.busy():
            self.det_status = "BUSY"
        else:
            self.det_status = "OK"

        latency_text = "-" if self.det_latency_ms is None else f"{self.det_latency_ms:.1f} ms"
        if hasattr(self, "det_fps_var"):
            self.det_fps_var.set(f"偵測 FPS {self.det_fps:.1f} | 核心 {self.det_core_fps:.1f}")
            self.det_latency_var.set(f"YOLO 延遲 {latency_text} | every {self.detect_every_n_frames}f")
            self.det_count_var.set(f"objects {self.det_count} | {self.det_status}")

    def update_anafi_metrics(self, st: DroneState) -> None:
        if not hasattr(self, "anafi_flight_var"):
            return
        # Altitude comes from the scale-free GLOMAP map: label as map-units,
        # not meters (no metric anchor in this system).
        self.anafi_flight_var.set(
            f"battery {st.battery_pct:.0f}% | alt {st.altitude_m:.1f}u(map) | "
            f"gimbal {st.gimbal_pitch_deg:.0f}° | zoom {st.zoom:.1f}x"
        )
        self.anafi_stream_var.set(
            f"{ANAFI.stream_width}x{ANAFI.stream_height}@{ANAFI.stream_fps:.0f} | "
            f"{ANAFI.stream_mbps:.0f} Mb/s | latency {ANAFI.stream_latency_ms:.0f} ms | HFOV {ANAFI.video_hfov_deg:.0f}°"
        )
        self.anafi_limit_var.set(
            f"limit H {ANAFI.max_horizontal_speed_mps:.0f}m/s | V {ANAFI.max_vertical_speed_mps:.0f}m/s | "
            f"yaw {ANAFI.max_yaw_rate_dps:.0f}°/s | gimbal ±90°"
        )

    @staticmethod
    def _map_view_bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
        if len(points) == 0:
            return np.zeros(3, dtype=float), 3.0
        xyz = points[:, :3].astype(float)
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        center = (lo + hi) * 0.5
        radius = float(np.max(hi - lo) * 0.5)
        return center, max(radius, 1.0)

    def on_map_press(self, event) -> None:
        self._drag_button = int(event.num)
        self._drag_last = (int(event.x), int(event.y))

    def on_map_drag(self, event) -> None:
        if self._drag_last is None:
            return
        x, y = int(event.x), int(event.y)
        last_x, last_y = self._drag_last
        dx = x - last_x
        dy = y - last_y
        self._drag_last = (x, y)
        if self._drag_button == 1:
            self.map_yaw -= dx * 0.008
            self.map_pitch = (self.map_pitch + dy * 0.008) % (2.0 * math.pi)
        elif self._drag_button == 2:
            self.map_roll = (self.map_roll + dx * 0.008) % (2.0 * math.pi)
        elif self._drag_button == 3:
            self.map_pan += np.array([dx, dy], dtype=float)
        self.redraw_map_only()

    def on_map_release(self, _event) -> None:
        self._drag_button = None
        self._drag_last = None

    def on_map_wheel(self, event) -> None:
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = 1.12
        else:
            factor = 1.0 / 1.12
        self.map_zoom = float(np.clip(self.map_zoom * factor, 0.08, 120.0))
        self.redraw_map_only()

    def reset_map_view(self, _event=None) -> None:
        self.map_center = self.default_map_center.copy()
        self.pivot_point = self.map_center.copy()
        self.map_yaw = DEFAULT_MAP_YAW
        self.map_pitch = DEFAULT_MAP_PITCH
        self.map_roll = DEFAULT_MAP_ROLL
        self.map_zoom = DEFAULT_MAP_ZOOM
        self.map_pan[:] = 0.0
        self.redraw_map_only()

    def flip_map_vertical(self) -> None:
        self.map_pitch = (self.map_pitch + math.pi) % (2.0 * math.pi)
        self.redraw_map_only()

    def set_rotation_pivot(self, event) -> None:
        if len(self.map_points) == 0:
            return
        width = max(300, self.map_label.winfo_width())
        height = max(220, self.map_label.winfo_height())
        step = max(1, len(self.map_points) // 90000)
        pts = self.map_points[::step, :3].astype(float)
        view = self.transform_xyz(pts)
        scale = min(width, height) * 0.46 * self.map_zoom / self.map_radius
        sx = width * 0.5 + self.map_pan[0] + view[:, 0] * scale
        sy = height * 0.5 + self.map_pan[1] - view[:, 1] * scale
        inside = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        if not np.any(inside):
            return
        dx = sx[inside] - float(event.x)
        dy = sy[inside] - float(event.y)
        dist2 = dx * dx + dy * dy
        local_idx = int(np.argmin(dist2))
        if float(dist2[local_idx]) > 45.0 * 45.0:
            return
        world = pts[inside][local_idx]
        self.map_center = world.copy()
        self.pivot_point = world.copy()
        self.map_pan[:] = [float(event.x) - width * 0.5, float(event.y) - height * 0.5]
        self.write_log(f"map pivot set: x={world[0]:.2f} y={world[1]:.2f} z={world[2]:.2f}")
        self.redraw_map_only()

    def transform_xyz(self, xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xyz, dtype=float).reshape(-1, 3) - self.map_center[None, :]
        cy, sy = math.cos(self.map_yaw), math.sin(self.map_yaw)
        cp, sp = math.cos(self.map_pitch), math.sin(self.map_pitch)
        cr, sr = math.cos(self.map_roll), math.sin(self.map_roll)
        x1 = cy * pts[:, 0] + sy * pts[:, 2]
        z1 = -sy * pts[:, 0] + cy * pts[:, 2]
        y1 = pts[:, 1]
        y2 = cp * y1 - sp * z1
        z2 = sp * y1 + cp * z1
        x3 = cr * x1 - sr * y2
        y3 = sr * x1 + cr * y2
        return np.stack([x3, y3, z2], axis=1)

    def project_world(self, xyz: np.ndarray, width: int, height: int) -> tuple[int, int]:
        v = self.transform_xyz(np.asarray(xyz, dtype=float).reshape(1, 3))[0]
        scale = min(width, height) * 0.46 * self.map_zoom / self.map_radius
        sx = int(width * 0.5 + self.map_pan[0] + v[0] * scale)
        sy = int(height * 0.5 + self.map_pan[1] - v[1] * scale)
        return sx, sy

    def redraw_map_only(self) -> None:
        mw = max(300, self.map_label.winfo_width())
        mh = max(220, self.map_label.winfo_height())
        self.map_photo = ImageTk.PhotoImage(self.render_map(mw, mh, self.current_state))
        self.map_label.delete("frame")
        self.map_label.create_image(0, 0, image=self.map_photo, anchor="nw", tags="frame")

    def map_base_key(self, width: int, height: int) -> tuple:
        return (
            int(width), int(height),
            round(float(self.map_zoom), 4),
            round(float(self.map_yaw), 4),
            round(float(self.map_pitch), 4),
            round(float(self.map_roll), 4),
            tuple(np.round(self.map_center.astype(float), 4)),
            tuple(np.round(self.pivot_point.astype(float), 4)),
            tuple(np.round(self.map_pan.astype(float), 2)),
            len(self.map_points),
        )

    def render_map_base(self, width: int, height: int) -> Image.Image:
        arr = np.empty((max(1, height), max(1, width), 3), dtype=np.uint8)
        arr[:] = (0x15, 0x18, 0x1c)   # "#15181c"
        scale = min(width, height) * 0.46 * self.map_zoom / self.map_radius

        step = max(1, len(self.map_points) // 70000)
        pts = self.map_points[::step]
        if len(pts):
            view = self.transform_xyz(pts[:, :3])
            sx = (width * 0.5 + self.map_pan[0] + view[:, 0] * scale).astype(int)
            sy = (height * 0.5 + self.map_pan[1] - view[:, 1] * scale).astype(int)
            inside = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
            order = np.argsort(view[inside, 2])
            sx_i = sx[inside][order]
            sy_i = sy[inside][order]
            rgb = np.clip(pts[:, 3:6][inside][order], 0, 255).astype(np.uint8)
            arr[sy_i, sx_i] = rgb   # depth-sorted: later (nearer) points overwrite, like draw.point order
        img = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(img)

        # Draw map axes in the current view.
        axis_len = self.map_radius * 0.18
        origin = self.map_center
        ox, oy = self.project_world(origin, width, height)
        axes = [
            (origin + np.array([axis_len, 0, 0]), "#ff6b5f", "X"),
            (origin + np.array([0, -axis_len, 0]), "#3fbf7f", "UP"),
            (origin + np.array([0, 0, axis_len]), "#5aa7e8", "Z"),
        ]
        for end, color, label in axes:
            ex, ey = self.project_world(end, width, height)
            draw.line((ox, oy, ex, ey), fill=color, width=2)
            draw.text((ex + 4, ey + 4), label, fill=color)

        px, py = self.project_world(self.pivot_point, width, height)
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline="#e6b94f", width=2)
        draw.line((px - 12, py, px + 12, py), fill="#e6b94f", width=1)
        draw.line((px, py - 12, px, py + 12), fill="#e6b94f", width=1)
        return img

    def map_base_image(self, width: int, height: int) -> Image.Image:
        key = self.map_base_key(width, height)
        if self.map_base_cache is None or self.map_base_cache_key != key:
            self.map_base_cache = self.render_map_base(width, height)
            self.map_base_cache_key = key
        return self.map_base_cache

    def render_map(self, width: int, height: int, st: DroneState) -> Image.Image:
        img = self.map_base_image(width, height).copy()
        draw = ImageDraw.Draw(img)

        # permanent no-localization markers (red): every spot where localization has failed,
        # accumulated over the flight (drawn under the live track so the track stays readable).
        if self.no_loc_markers:
            mv = self.transform_xyz(np.asarray(self.no_loc_markers, dtype=float))
            msc = min(width, height) * 0.46 * self.map_zoom / self.map_radius
            for mp in mv:
                mx = int(width * 0.5 + self.map_pan[0] + mp[0] * msc)
                my = int(height * 0.5 + self.map_pan[1] - mp[1] * msc)
                draw.ellipse((mx - 5, my - 5, mx + 5, my + 5), fill="#ff2a2a", outline="#ffffff")

        # Batch route + flown-track + health-marker points through ONE transform_xyz
        # call (was hundreds of per-point project_world calls per tick).
        route_pts = self.route_pts if getattr(self, "route_pts", None) else []
        route_n = len(route_pts)
        hist_n = len(self.history)
        if route_n or hist_n:
            parts = []
            if route_n:
                parts.append(np.asarray(route_pts, dtype=float).reshape(-1, 3))
            if hist_n:
                parts.append(np.array([p[:3] for p in self.history], dtype=float).reshape(-1, 3))
            view = self.transform_xyz(np.concatenate(parts, axis=0))
            scale = min(width, height) * 0.46 * self.map_zoom / self.map_radius
            sx = (width * 0.5 + self.map_pan[0] + view[:, 0] * scale).astype(int).tolist()
            sy = (height * 0.5 + self.map_pan[1] - view[:, 1] * scale).astype(int).tolist()

            # planned drawn route (aligned->glomap), distinct colour from the flown track
            if route_n > 1:
                rp = list(zip(sx[:route_n], sy[:route_n]))
                draw.line(rp, fill=ROUTE_COLOR, width=2, joint="curve")
                for qx, qy in rp:
                    draw.ellipse((qx - 3, qy - 3, qx + 3, qy + 3), fill=ROUTE_COLOR)

            if hist_n > 1:
                hsx = sx[route_n:]
                hsy = sy[route_n:]
                hist = list(zip(hsx, hsy))
                draw.line(hist, fill="#5aa7e8", width=3)
                # mark WHERE localization was weak/failed along the flown track
                for (hx, hy), h in zip(hist, self.history_health):
                    if h and h != "OK":
                        draw.ellipse((hx - 4, hy - 4, hx + 4, hy + 4), fill=HEALTH_COLOR.get(h, "#e0a92e"))

        x, y, z, yaw = st.pose
        sx, sy = self.project_world(np.array([x, y, z], dtype=float), width, height)
        if np.isfinite(float(yaw)):
            heading = np.array([math.cos(float(yaw)), 0.0, math.sin(float(yaw))], dtype=float)
            center = np.array([x, y, z], dtype=float)
            hx, hy = self.project_world(center + heading, width, height)
            vx = float(hx - sx)
            vy = float(hy - sy)
            norm = math.hypot(vx, vy)
            if norm < 1e-3:
                vx, vy, norm = 1.0, 0.0, 1.0
            ux, uy = vx / norm, vy / norm
            px_dir, py_dir = -uy, ux
            arrow_len_px = 9.0
            arrow_w_px = 3.5
            tx, ty = sx - ux * 3.0, sy - uy * 3.0
            hx, hy = sx + ux * arrow_len_px, sy + uy * arrow_len_px
            bx, by = sx + ux * 1.0, sy + uy * 1.0
            lx, ly = bx + px_dir * arrow_w_px, by + py_dir * arrow_w_px
            rx, ry = bx - px_dir * arrow_w_px, by - py_dir * arrow_w_px
            draw.line((tx, ty, hx, hy), fill="#3fbf7f", width=1)
            draw.polygon([(hx, hy), (lx, ly), (rx, ry)], fill="#3fbf7f")
            draw.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill="#d8ffe8")
        else:
            draw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), outline="#3fbf7f", width=3)
        draw.text((10, 10), "左鍵360旋轉 | 左鍵雙擊設旋轉中心 | 中鍵滾轉 | 右鍵平移 | 滾輪縮放", fill="#f0f3f5")
        draw.text((10, 30), f"x={x:.2f} y={y:.2f} z={z:.2f} | inliers={st.inliers} "
                  f"reproj={'-' if st.reproj is None else f'{st.reproj:.2f}'}", fill="#a7b0b8")
        draw.text((10, 50), f"zoom={self.map_zoom:.2f} yaw={math.degrees(self.map_yaw)%360:.0f} "
                  f"pitch={math.degrees(self.map_pitch)%360:.0f} roll={math.degrees(self.map_roll)%360:.0f}",
                  fill="#a7b0b8")
        if self.no_loc_markers:
            draw.text((10, 70), f"紅點 = 無法定位位置（{len(self.no_loc_markers)} 處，永久標記）", fill="#ff6a6a")
        return img

    def draw_detections(self, draw: ImageDraw.ImageDraw, scale: float, ox: int, oy: int,
                        width: int, height: int) -> None:
        if not self.detection_result or not self.detection_result.get("success"):
            return
        boxes = self.detection_result.get("boxes") or []
        if not boxes:
            return
        palette = [
            "#ff6b5f", "#5aa7e8", "#3fbf7f", "#e6b94f", "#c084fc", "#f97316",
            "#22d3ee", "#f43f5e", "#84cc16", "#facc15", "#38bdf8", "#fb7185",
        ]
        for box in boxes:
            xyxy = box.get("xyxy") or []
            if len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            sx1 = int(ox + x1 * scale)
            sy1 = int(oy + y1 * scale)
            sx2 = int(ox + x2 * scale)
            sy2 = int(oy + y2 * scale)
            if sx2 < 0 or sy2 < 0 or sx1 >= width or sy1 >= height:
                continue
            sx1 = max(0, min(width - 1, sx1))
            sy1 = max(0, min(height - 1, sy1))
            sx2 = max(0, min(width - 1, sx2))
            sy2 = max(0, min(height - 1, sy2))
            class_id = int(box.get("class_id", 0) or 0)
            color = palette[class_id % len(palette)]
            conf = float(box.get("confidence", 0.0) or 0.0)
            label = f"{box.get('class_name', f'class_{class_id}')} {conf:.2f}"
            draw.rectangle((sx1, sy1, sx2, sy2), outline=color, width=2)
            tw = max(60, min(width - sx1 - 2, len(label) * 7 + 8))
            ty1 = max(0, sy1 - 18)
            draw.rectangle((sx1, ty1, sx1 + tw, ty1 + 16), fill="#08090b", outline=color)
            draw.text((sx1 + 4, ty1 + 2), label[:28], fill=color)

    def render_video(self, width: int, height: int, st: DroneState) -> Image.Image:
        img = Image.new("RGB", (max(1, width), max(1, height)), "#08090b")
        draw = ImageDraw.Draw(img)
        if self.video_frame is not None:
            src = self.video_frame
            scale = min(width / src.width, height / src.height)
            new_size = (max(1, int(src.width * scale)), max(1, int(src.height * scale)))
            frame = src.resize(new_size, Image.Resampling.BILINEAR)
            ox = (width - frame.width) // 2
            oy = (height - frame.height) // 2
            img.paste(frame, (ox, oy))
            self.draw_detections(draw, scale, ox, oy, width, height)
        else:
            for i in range(0, width, 64):
                draw.line((i, 0, i, height), fill="#15181c")
            for j in range(0, height, 64):
                draw.line((0, j, width, j), fill="#15181c")
            draw.text((28, 28), "No video stream", fill="#f0f3f5")
            draw.text((28, 52), "Use --video /home/cihcilab/Downloads/P0230023.MP4", fill="#a7b0b8")
        draw.rectangle((18, 18, width - 18, height - 18), outline="#363c44", width=2)
        hud_h = 100
        draw.rectangle((18, height - hud_h, width - 18, height - 18), fill="#08090b", outline="#363c44")
        draw.text((28, height - hud_h + 10), f"{ANAFI.model} | {st.mode} | {st.tracker_state} | {st.loc} | stream={st.stream}", fill="#e6b94f")
        draw.text((28, height - hud_h + 32), f"720p30 H264/RTP sim | {ANAFI.stream_mbps:.0f} Mb/s | link latency {ANAFI.stream_latency_ms:.0f} ms | HFOV {ANAFI.video_hfov_deg:.0f}°",
                  fill="#f0f3f5")
        draw.text((28, height - hud_h + 54), f"battery={st.battery_pct:.0f}% alt={st.altitude_m:.1f}u(map) "
                  f"gimbal={st.gimbal_pitch_deg:.0f}° zoom={st.zoom:.1f}x | "
                  f"inliers={st.inliers} reproj={'-' if st.reproj is None else f'{st.reproj:.2f}'}",
                  fill="#f0f3f5")
        draw.text((28, height - hud_h + 76), f"video={self.video_display_frame_name or '-'} "
                  f"loc={self.live_result_frame_name or self.live_pending_frame_name or '-'} "
                  f"det={self.detection_result_frame_name or self.detection_pending_frame_name or '-'} "
                  f"obj={self.det_count}",
                  fill="#a7b0b8")
        # localization alert banner: operator sees WHEN localization fails / is low-confidence
        health = getattr(self, "loc_health", "OK")
        if self.inspecting and health != "OK":
            col = HEALTH_COLOR[health]
            msg = ("LOCALIZATION LOST" if health == "FAIL"
                   else f"LOW CONFIDENCE  inliers={self.loc_health_inliers}"
                        f" reproj={'-' if self.loc_health_reproj is None else f'{self.loc_health_reproj:.1f}'}")
            draw.rectangle((18, 20, width - 18, 60), fill=col)
            draw.text((width // 2, 40), msg, fill="#0b0c0e", anchor="mm")
        return img

    def state_from_replay(self, base: DroneState) -> DroneState:
        if not self.replay_rows:
            return base
        if base.stream == "LOST":
            return base
        row = self.replay_rows[min(self.replay_index, len(self.replay_rows) - 1)]
        pose = row.get("pose")
        if pose:
            raw_yaw = float(pose.get("yaw", 0.0))
            display_yaw = self.replay_headings[min(self.replay_index, len(self.replay_headings) - 1)]
            if display_yaw is None:
                display_yaw = raw_yaw
            self.replay_last_pose[:] = [
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("z", 0.0)),
                float(display_yaw),
            ]
        base.pose[:] = self.replay_last_pose
        base.mode = "REPLAY"
        base.loc = "OK" if row.get("success") else "FAIL"
        base.tracker_state = str(row.get("next_mode") or row.get("mode") or "TRACK")
        base.inliers = int(row.get("inliers", 0) or 0)
        base.reproj = row.get("reproj_rms")
        if self.boot_holding():
            base.mode = "BOOT_INIT"
            base.loc = "MEGALOC_LOCKED" if row.get("success") else "MEGALOC_LOCKING"
            base.tracker_state = "HOVER_LOCK"
            base.stream = "HOLD_720P"
            return base
        base.stream = "OK" if self.video_frame_fresh else "WAIT"
        if self.video_frame_fresh:
            self.replay_index = (self.replay_index + 1) % len(self.replay_rows)
        return base

    def submit_current_frame_for_localization(self) -> None:
        if not self.inspecting:
            return
        if self.localizer is None or self.video_frame is None:
            return
        if self.video_display_index < 0 or self.video_display_index == self.last_submitted_index:
            return
        if self.localizer.busy():
            return
        frame_name = self.video_display_frame_name or f"stream_{self.video_display_index:06d}"
        if self.localizer.submit(self.video_display_index, frame_name, self.video_frame):
            self.last_submitted_index = self.video_display_index
            self.live_pending_frame_name = frame_name

    def submit_current_frame_for_detection(self) -> None:
        if self.detector is None or self.video_frame is None:
            return
        if self.video_display_index < 0 or self.video_display_index == self.last_detect_submitted_index:
            return
        if self.video_display_index % self.detect_every_n_frames != 0:
            return
        if self.detector.busy():
            return
        frame_name = self.video_display_frame_name or f"stream_{self.video_display_index:06d}"
        if self.detector.submit(self.video_display_index, frame_name, self.video_frame):
            self.last_detect_submitted_index = self.video_display_index
            self.detection_pending_frame_name = frame_name
            self.det_status = "RUN"

    def update_live_results(self) -> None:
        self.live_new_pose = False
        if self.localizer is None:
            return
        for result in self.localizer.poll_results():
            self.live_result = result
            self.live_result_frame_name = str(result.get("frame_name", ""))
            self.update_localization_metrics(result)
            _tw = time.monotonic()
            if _tw - self._last_status_write >= 0.2:       # throttle debug status file to ~5Hz
                self._last_status_write = _tw
                try:
                    LIVE_STATUS_PATH.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            if not result.get("success") or not result.get("pose"):
                self.write_log(f"LIVE_LOCALIZE_FAIL {self.live_result_frame_name}: {result.get('error', 'no pose')}")
                continue
            pose = result["pose"]
            xyz = np.array([
                float(pose.get("x", 0.0)),
                float(pose.get("y", 0.0)),
                float(pose.get("z", 0.0)),
            ], dtype=float)
            # continuity gate: a fix that BREAKS the continuous trajectory (jumps too far
            # from the last accepted point) is not considered; a second agreeing fix is
            # accepted as genuine relocalization. Mirrors the flight controller's guard.
            if self.live_last_xyz is not None and \
                    float(np.linalg.norm(xyz - self.live_last_xyz)) > POSE_JUMP_U:
                if self._live_pending is not None and \
                        float(np.linalg.norm(xyz - self._live_pending)) <= POSE_JUMP_U:
                    self._live_pending = None                # two agreeing fixes -> real reloc, accept
                else:
                    self._live_pending = xyz
                    self.write_log(
                        f"POSE_JUMP_REJECT {self.live_result_frame_name}: "
                        f"{float(np.linalg.norm(xyz - self.live_last_xyz)):.2f}u > {POSE_JUMP_U}u; "
                        "trajectory break, skipped")
                    self.loc_health = "FAIL"                 # discontinuous fix -> operator alert
                    self._record_no_loc()
                    continue                                 # discontinuous -> don't consider this fix
            else:
                self._live_pending = None
            if self.live_last_xyz is not None:
                d = xyz - self.live_last_xyz
                if math.hypot(float(d[0]), float(d[2])) >= 0.05:
                    self.live_heading = math.atan2(float(d[2]), float(d[0]))
            self.live_last_xyz = xyz
            self.live_pose[:] = [
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                np.nan if self.live_heading is None else float(self.live_heading),
            ]
            self.live_locked = True
            self.live_new_pose = True
            if self.boot_holding():
                self.boot_lock_done = True
                self.write_log(f"BOOT_INIT: live MegaLoc/PnP locked on {self.live_result_frame_name}")

    def update_detection_results(self) -> None:
        if self.detector is None:
            if hasattr(self, "det_count_var"):
                self.det_count_var.set("objects - | OFF")
            return
        for result in self.detector.poll_results():
            self.detection_result = result
            self.detection_result_frame_name = str(result.get("frame_name", ""))
            self.update_detection_metrics(result)
            _tw = time.monotonic()
            if _tw - self._last_det_write >= 0.2:          # throttle debug status file to ~5Hz
                self._last_det_write = _tw
                try:
                    LIVE_DETECTION_STATUS_PATH.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            if not result.get("success"):
                self.write_log(f"LIVE_DETECT_FAIL {self.detection_result_frame_name}: {result.get('error', 'no boxes')}")

    def state_from_live(self, base: DroneState) -> DroneState:
        base.mode = "LIVE"
        base.pose[:] = self.live_pose
        if self.live_result is not None:
            base.inliers = int(self.live_result.get("inliers", 0) or 0)
            base.reproj = self.live_result.get("reproj_rms")
            base.tracker_state = str(self.live_result.get("next_mode") or self.live_result.get("mode") or "TRACK")
            base.loc = "OK" if self.live_result.get("success") else "FAIL"
        base.altitude_m = max(0.0, -float(base.pose[1]))
        if self.boot_holding():
            base.mode = "BOOT_INIT"
            base.loc = "MEGALOC_LOCKING"
            base.tracker_state = "HOVER_LOCK"
            base.stream = "HOLD_720P"
        elif self.localizer is not None and self.localizer.busy():
            base.stream = "LOCALIZING"
        elif self.video_frame_fresh:
            base.stream = "OK"
        return base

    def tick(self) -> None:
        st = self.backend.poll()
        self.video_frame_fresh = False
        self.update_boot_lock()
        self.update_live_results()
        self.update_detection_results()
        if self.video_stream is not None and self.inspecting:
            frame = None
            now = time.monotonic()
            stream_due = now >= self.next_stream_frame_time
            can_read = stream_due and (not self.boot_holding() or self.video_frame is None)
            if can_read:
                frame = self.video_stream.next_frame()
                if self.next_stream_frame_time <= 0.0:
                    self.next_stream_frame_time = now
                while self.next_stream_frame_time <= now:
                    self.next_stream_frame_time += self.stream_period_s
            if frame is not None:
                self.video_frame = frame
                self.video_display_index = max(0, self.video_stream.output_index - 1)
                self.video_display_frame_name = self.video_stream.last_frame_name
                self.video_frame_fresh = not self.boot_holding()
                self.stream_lost_since = None
                st.stream = "HOLD_720P" if self.boot_holding() else "OK"
                # overall pipeline FPS = frames actually processed since 開始巡檢
                self.processed_frames += 1
                if self.inspect_start is not None:
                    elapsed = now - self.inspect_start
                    if elapsed > 0:
                        self.overall_fps = self.processed_frames / elapsed
            else:
                if self.boot_holding() and self.video_frame is not None:
                    st.stream = "HOLD_720P"
                elif self.stream_lost_since is None:
                    self.stream_lost_since = time.monotonic()
                    self.backend.stream_lost_hover("720p frame source returned None")
                    self.write_log("STREAM_LOST_HOVER: 720p frame source returned None")
                st = self.backend.state
        elif self.video_stream is not None:
            st.stream = "STANDBY"                         # 待命：尚未按開始巡檢
        elif self.video_frame is None:
            st.stream = "NO_SOURCE"
        self.submit_current_frame_for_localization()
        self.submit_current_frame_for_detection()
        if self.localizer is not None:
            st = self.state_from_live(st)
        else:
            st = self.state_from_replay(st)
        self.current_state = st
        self.update_anafi_metrics(st)
        if (self.localizer is not None and self.live_new_pose) or (self.localizer is None):
            self.history.append(st.pose.copy())
            self.history_health.append(getattr(self, "loc_health", "OK"))
            if len(self.history) > 300:
                self.history.pop(0)
                if self.history_health:
                    self.history_health.pop(0)
        self.status.configure(
            text=f"{st.mode} | {st.loc} | {st.tracker_state} | stream {st.stream} | "
                 f"battery {st.battery_pct:.0f}% | inliers {st.inliers} | objects {self.det_count}"
        )
        if hasattr(self, "overall_fps_var"):
            if not self.inspecting:
                self.overall_fps_var.set("整體 FPS - (待命，按開始巡檢)")
            else:
                self.overall_fps_var.set(
                    f"整體 FPS {self.overall_fps:.1f} | 已處理 {self.processed_frames} 幀")

        mw = max(300, self.map_label.winfo_width())
        mh = max(220, self.map_label.winfo_height())
        vw = max(300, self.video_label.winfo_width())
        vh = max(220, self.video_label.winfo_height())

        # Map: re-render only when the view, flown history, drawn route or pose/quality
        # readouts changed. Between live fixes these are all static, so reuse the PhotoImage.
        pose_key = tuple(round(float(v), 4) if np.isfinite(v) else None
                         for v in np.asarray(st.pose, dtype=float))
        map_key = (
            self.map_base_key(mw, mh),
            len(self.history),
            id(self.history[-1]) if self.history else 0,
            id(self.history[0]) if self.history else 0,
            len(self.route_pts),
            len(self.no_loc_markers),
            pose_key,
            int(st.inliers),
            None if st.reproj is None else round(float(st.reproj), 4),
        )
        if map_key != self._map_dirty_key:
            self._map_dirty_key = map_key
            self.map_photo = ImageTk.PhotoImage(self.render_map(mw, mh, st))
            self.map_label.delete("frame")
            self.map_label.create_image(0, 0, image=self.map_photo, anchor="nw", tags="frame")

        # Video: re-render only when the source frame, panel size, the localization
        # alert banner inputs or the detection overlay changed. The expensive 720p resize
        # + PhotoImage is skipped on ticks where the same frame is shown again.
        video_key = (
            id(self.video_frame), vw, vh,
            self.inspecting,
            self.loc_health, self.loc_health_inliers,
            None if self.loc_health_reproj is None else round(float(self.loc_health_reproj), 4),
            id(self.detection_result),
        )
        if video_key != self._video_dirty_key:
            self._video_dirty_key = video_key
            self.video_photo = ImageTk.PhotoImage(self.render_video(vw, vh, st))
            self.video_label.delete("frame")
            self.video_label.create_image(0, 0, image=self.video_photo, anchor="nw", tags="frame")
        self.after(self.tick_ms, self.tick)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-ply", default=str(DEFAULT_MAP))
    ap.add_argument("--max-points", type=int, default=90000)
    ap.add_argument("--video", default=str(DEFAULT_VIDEO) if DEFAULT_VIDEO.exists() else "")
    ap.add_argument("--video-stride", type=int, default=1,
                    help="1 means ANAFI-like 720p30 stream; >1 keeps every Nth source frame")
    ap.add_argument("--replay-json", default=str(DEFAULT_REPLAY_JSON) if DEFAULT_REPLAY_JSON.exists() else "")
    ap.add_argument("--live-localize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--localizer-python",
                    default=default_worker_python("SFM_LOCALIZER_PYTHON", "/usr/bin/python3.12"))
    ap.add_argument("--localizer-worker", default=str(DEFAULT_WORKER))
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--megaloc-cache", default=str(DEFAULT_MEGALOC))
    ap.add_argument("--live-detect", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--detector-python",
                    default=default_worker_python("SFM_DETECTOR_PYTHON", "/home/cihcilab/miniconda3/bin/python3"))
    ap.add_argument("--detector-worker", default=str(DEFAULT_DETECTOR_WORKER))
    ap.add_argument("--detector-model", default=str(DEFAULT_DETECTOR_MODEL))
    ap.add_argument("--detect-every-n-frames", type=int, default=3)
    ap.add_argument("--detector-conf", type=float, default=0.25)
    ap.add_argument("--detector-iou", type=float, default=0.7)
    ap.add_argument("--detector-max-det", type=int, default=300)
    ap.add_argument("--tick-ms", type=int, default=16)
    ap.add_argument("--stream-fps", type=float, default=ANAFI.stream_fps,
                    help="720p frame-source rate fed to the localizer (real ANAFI live stream is 30)")
    ap.add_argument("--boot-lock-ms", type=int, default=2500,
                    help="hold the first 720p frame to simulate takeoff hover + MegaLoc BOOT_INIT")
    ap.add_argument("--route-json", default=str(DEFAULT_ROUTE),
                    help="drawn flight path (aligned frame) to overlay on the map in a distinct colour")
    ap.add_argument("--layout-selftest", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    print("[mode] REPLAY/UI: no drone commands -- this app uses a sim backend and never "
          "sends TakeOff/PCMD/Landing/Emergency to a real drone", flush=True)
    # AnafiProfile is a frozen dataclass; set in place so FFmpegFrameStream + the app's
    # read timing pick up the requested stream rate.
    object.__setattr__(ANAFI, "stream_fps", float(args.stream_fps))

    points = read_ply_points(Path(args.map_ply), args.max_points)
    if args.selftest:
        print(f"loaded {len(points)} sampled map points from {args.map_ply}")
        if len(points):
            xyz = points[:, :3]
            print(f"bbox min={xyz.min(axis=0).round(3).tolist()} max={xyz.max(axis=0).round(3).tolist()}")
        if args.video:
            print(f"video stream fixed at {STREAM_WIDTH}x{STREAM_HEIGHT}@{ANAFI.stream_fps:g}: {args.video} stride={args.video_stride}")
        if args.replay_json:
            replay = json.loads(Path(args.replay_json).read_text(encoding="utf-8"))
            print(f"loaded replay rows={len(replay.get('rows', []))} from {args.replay_json}")
        print(f"live localize={args.live_localize} worker={args.localizer_worker}")
        print(
            f"live detect={args.live_detect} worker={args.detector_worker} "
            f"model={args.detector_model} every={args.detect_every_n_frames}f"
        )
        print(f"boot lock hold={args.boot_lock_ms} ms")
        return

    if args.layout_selftest:
        app = OperatorApp(DroneBackend(), points, tick_ms=args.tick_ms,
                          boot_lock_ms=0)
        app.update_idletasks()
        sizes = []
        for _ in range(5):
            app.tick()
            app.update_idletasks()
            sizes.append((app.winfo_reqwidth(), app.winfo_reqheight()))
        print(f"layout requested sizes: {sizes}")
        if len(set(sizes[-3:])) != 1:
            raise SystemExit("layout requested size is not stable")
        app.destroy()
        return

    video_stream = None
    if args.video:
        video_stream = FFmpegFrameStream(Path(args.video), STREAM_WIDTH, STREAM_HEIGHT,
                                         stride=args.video_stride, fps=ANAFI.stream_fps)
    localizer = None
    if args.live_localize:
        localizer = LiveLocalizerClient(
            Path(args.localizer_worker),
            args.localizer_python,
            STREAM_WIDTH,
            STREAM_HEIGHT,
            Path(args.bundle),
            args.megaloc_cache,
        )
    detector = None
    if args.live_detect:
        detector = LiveDetectorClient(
            Path(args.detector_worker),
            args.detector_python,
            STREAM_WIDTH,
            STREAM_HEIGHT,
            Path(args.detector_model),
            imgsz=640,
            conf=args.detector_conf,
            iou=args.detector_iou,
            max_det=args.detector_max_det,
        )
    replay_rows = []
    if args.replay_json and not args.live_localize:
        replay_rows = json.loads(Path(args.replay_json).read_text(encoding="utf-8")).get("rows", [])
    app = OperatorApp(DroneBackend(), points, video_stream=video_stream, localizer=localizer,
                      detector=detector, detect_every_n_frames=args.detect_every_n_frames,
                      replay_rows=replay_rows, tick_ms=args.tick_ms,
                      boot_lock_ms=args.boot_lock_ms)
    app.route_pts = load_route_glomap(args.route_json)
    if app.route_pts:
        print(f"[operator] overlaying drawn route: {len(app.route_pts)} waypoints "
              f"from {args.route_json}", flush=True)
    try:
        app.mainloop()
    finally:
        if video_stream is not None:
            video_stream.close()
        if localizer is not None:
            localizer.close()
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    main()
